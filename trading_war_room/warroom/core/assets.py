"""Asset classes, instrument specs, the 24/7 crypto schedule, and the router
that sends each symbol to the right rules.

Equities: NYSE calendar, whole shares, per-share commission, shorting allowed.
Crypto:   trades 24/7, fractional quantities in exchange lot steps, minimum
          order size/notional, percentage maker/taker fees, spot (no shorting
          by default). The "session" is the UTC day: it anchors VWAP and the
          prior-day levels, and there is no opening range, holiday or EOD flat.
          Positions are capped by a maximum holding time instead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from warroom.core.clock import UTC, ensure_utc
from warroom.core.market_calendar import MarketCalendar, MarketStatus, TradingSession

CRYPTO_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}-(USD|USDT|USDC|EUR|CAD)$")


class AssetClass(StrEnum):
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    qty_step: float = 1.0          # lot size (1 share for equities)
    min_qty: float = 1.0
    min_notional: float = 0.0      # minimum order value in quote currency
    price_decimals: int = 2
    taker_fee_pct: float = 0.0     # market orders (entries, stops, flattens)
    maker_fee_pct: float = 0.0     # resting limit orders (targets)
    allow_short: bool = True

    @property
    def is_crypto(self) -> bool:
        return self.asset_class is AssetClass.CRYPTO

    @property
    def base(self) -> str:
        return self.symbol.split("-")[0]

    @property
    def unit(self) -> str:
        return self.base if self.is_crypto else "shares"

    def floor_qty(self, qty: float) -> float:
        """Round DOWN to the lot step (never round a size up)."""
        if qty <= 0 or not math.isfinite(qty):
            return 0.0
        steps = math.floor(round(qty / self.qty_step, 6))  # snap float noise to the step grid, then floor
        out = steps * self.qty_step
        return float(round(out, 10)) if self.is_crypto else float(int(out))

    def round_price(self, price: float) -> float:
        return round(price, self.price_decimals)

    def meets_minimum(self, qty: float, price: float) -> bool:
        return qty >= self.min_qty - 1e-12 and qty * price >= self.min_notional - 1e-9 and qty > 0

    def fmt_qty(self, qty: float) -> str:
        if not self.is_crypto:
            return f"{int(round(qty))} shares"
        s = f"{qty:.8f}".rstrip("0").rstrip(".")
        return f"{s} {self.base}"


def price_decimals_for(price: float) -> int:
    """Sensible tick precision for an unknown crypto pair."""
    if price >= 1000:
        return 1
    if price >= 10:
        return 2
    if price >= 1:
        return 4
    return 6


class CryptoSchedule:
    """Calendar-compatible schedule for a market that never closes.

    Implements the subset of :class:`MarketCalendar` the agents use. Each UTC
    day is one "session": ``orb_end == open`` (no opening range) and
    ``eod_flat is None`` (no end-of-day flat).
    """

    def session(self, d: date) -> TradingSession:
        start = datetime.combine(d, time(0, 0), tzinfo=UTC)
        return TradingSession(day=d, open=start, orb_end=start, eod_flat=None,
                              close=start + timedelta(days=1), early_close=False)

    def session_for(self, dt: datetime) -> TradingSession:
        return self.session(ensure_utc(dt).date())

    def status(self, dt: datetime) -> MarketStatus:
        return MarketStatus.OPEN

    def is_open(self, dt: datetime) -> bool:
        return True

    def is_trading_day(self, d: date) -> bool:
        return True

    def is_holiday(self, d: date) -> bool:
        return False

    def holiday_name(self, d: date) -> str | None:
        return None

    def next_session(self, dt: datetime) -> TradingSession:
        return self.session_for(dt)

    def latest_session_on_or_before(self, d: date) -> TradingSession:
        return self.session(d)

    def previous_trading_days(self, d: date, n: int, include_today: bool = True) -> list[date]:
        start = d if include_today else d - timedelta(days=1)
        return [start - timedelta(days=i) for i in range(n)]


class AssetRouter:
    """Classifies symbols and hands out the matching schedule and instrument spec."""

    def __init__(self, equity_calendar: MarketCalendar, crypto_cfg=None, equity_commission_per_share: float = 0.0) -> None:
        self.equity_calendar = equity_calendar
        self.crypto_schedule = CryptoSchedule()
        self.crypto_cfg = crypto_cfg
        self._overrides: dict[str, Instrument] = {}
        self.crypto_symbols: list[str] = list(crypto_cfg.symbols) if crypto_cfg is not None and crypto_cfg.enabled else []

    def is_crypto(self, symbol: str) -> bool:
        return symbol in self.crypto_symbols or bool(CRYPTO_SYMBOL_RE.match(symbol))

    def asset_class(self, symbol: str) -> AssetClass:
        return AssetClass.CRYPTO if self.is_crypto(symbol) else AssetClass.EQUITY

    def schedule_for(self, symbol: str):
        return self.crypto_schedule if self.is_crypto(symbol) else self.equity_calendar

    def normalize(self, text: str) -> str:
        """'btc' / 'BTC' / 'btc-usd' -> 'BTC-USD' when it names a configured crypto pair."""
        t = text.strip().upper().lstrip("$")
        if t in self.crypto_symbols:
            return t
        for s in self.crypto_symbols:
            if s.split("-")[0] == t:
                return s
        return t

    def set_override(self, inst: Instrument) -> None:
        """Exchange-provided specs (e.g. Kraken AssetPairs) take precedence over config defaults."""
        self._overrides[inst.symbol] = inst

    def instrument(self, symbol: str, price_hint: float | None = None) -> Instrument:
        if not self.is_crypto(symbol):
            return Instrument(symbol=symbol, asset_class=AssetClass.EQUITY)
        if symbol in self._overrides:
            return self._overrides[symbol]
        c = self.crypto_cfg
        spec = (c.instruments.get(symbol) if c is not None else None)
        base = Instrument(
            symbol=symbol, asset_class=AssetClass.CRYPTO,
            qty_step=1e-8, min_qty=0.0, min_notional=5.0,
            price_decimals=price_decimals_for(price_hint) if price_hint else 2,
            taker_fee_pct=c.taker_fee_pct if c is not None else 0.004,
            maker_fee_pct=c.maker_fee_pct if c is not None else 0.0025,
            allow_short=c.allow_short if c is not None else False,
        )
        if spec is not None:
            base = replace(base, qty_step=spec.qty_step, min_qty=spec.min_qty, min_notional=spec.min_notional,
                           price_decimals=spec.price_decimals)
        return base


def format_qty(symbol: str, qty: float) -> str:
    """'249 shares' for equities, '0.01234 BTC' for crypto pairs."""
    if CRYPTO_SYMBOL_RE.match(symbol):
        s = f"{qty:.8f}".rstrip("0").rstrip(".")
        return f"{s or '0'} {symbol.split('-')[0]}"
    return f"{int(round(qty))} shares"


def format_price(symbol: str, price: float | None) -> str:
    if price is None:
        return "n/a"
    if CRYPTO_SYMBOL_RE.match(symbol):
        return f"{price:,.{max(2, price_decimals_for(price))}f}"
    return f"{price:.2f}"
