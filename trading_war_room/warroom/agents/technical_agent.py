"""Market Scanner & Technical Agent.

Timeframes: 1m for price, VWAP, opening range and premarket levels; 5m for
EMAs, RSI, ATR, RVOL, swings and the setup trigger bar; 15m for trend
confirmation. Indicators are computed over multi-day regular-hours 5m history
so EMA50/ATR are warmed up from the first scan of the day.

Setups are evaluated on the LAST COMPLETED 5m bar only (no peeking at the bar
still forming) and both LONG and SHORT variants are checked.

De-duplication: the War Room is not re-convened on the same (ticker,
direction) within ``setup_cooldown_minutes`` of clock time. On-demand
``analyze <TICKER>`` requests bypass the gate.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import pandas as pd

from warroom.analysis import indicators as ind
from warroom.config import AppConfig
from warroom.core.assets import AssetClass, AssetRouter, Instrument
from warroom.core.clock import Clock, ensure_utc, et_to_utc
from warroom.core.market_calendar import MarketCalendar, TradingSession
from warroom.models.schemas import (
    Direction,
    IndicatorSnapshot,
    MarketRegime,
    Quote,
    SetupType,
    Timeframe,
    TradeSetup,
)
from warroom.providers.base import MarketDataProvider, ProviderError
from warroom.strategies.base import StrategyContext
from warroom.strategies.registry import build_registry

log = logging.getLogger(__name__)


@dataclass
class TickerContext:
    """Everything the War Room needs to know about one ticker at one moment."""

    ticker: str
    snapshot: IndicatorSnapshot
    session: TradingSession
    quote: Quote | None
    has_real_quotes: bool
    last_bar_close: datetime              # close time of the latest completed 1m bar
    prior_high: float | None = None
    prior_low: float | None = None
    prior_close: float | None = None
    trend_15m: Direction | None = None
    bars_5m: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)  # setup bars (15m for crypto)
    possible_halt: bool = False
    asset_class: AssetClass = AssetClass.EQUITY
    instrument: Instrument | None = None

    @property
    def is_crypto(self) -> bool:
        return self.asset_class is AssetClass.CRYPTO

    def data_age_seconds(self, now: datetime) -> float:
        return (ensure_utc(now) - self.last_bar_close).total_seconds()


@dataclass
class MacroBias:
    regime: MarketRegime
    direction: Direction | None
    detail: dict[str, str]


class TechnicalAgent:
    def __init__(self, cfg: AppConfig, market: MarketDataProvider, calendar: MarketCalendar, clock: Clock,
                 router: AssetRouter | None = None) -> None:
        self.cfg = cfg
        self.t = cfg.technical
        self.market = market
        self.calendar = calendar
        self.clock = clock
        self.router = router or AssetRouter(calendar, cfg.crypto)
        self.registry = build_registry(cfg)
        self._last_debated: dict[tuple[str, Direction], datetime] = {}

    # ------------------------------------------------------------------ cooldown gate

    def on_cooldown(self, ticker: str, direction: Direction) -> bool:
        last = self._last_debated.get((ticker, direction))
        if last is None:
            return False
        return self.clock.now() - last < timedelta(minutes=self.t.setup_cooldown_minutes)

    def mark_debated(self, ticker: str, direction: Direction) -> None:
        self._last_debated[(ticker, direction)] = self.clock.now()

    def cooldown_remaining(self, ticker: str, direction: Direction) -> float:
        last = self._last_debated.get((ticker, direction))
        if last is None:
            return 0.0
        left = timedelta(minutes=self.t.setup_cooldown_minutes) - (self.clock.now() - last)
        return max(0.0, left.total_seconds())

    # ------------------------------------------------------------------ context

    def _real_quotes(self, ticker: str) -> bool:
        f = getattr(self.market, "real_quotes_for", None)
        return f(ticker) if f is not None else self.market.has_real_quotes

    async def build_context(self, ticker: str) -> TickerContext | None:
        if self.router.is_crypto(ticker):
            return await self._build_crypto_context(ticker)
        now = self.clock.now()
        session = self.calendar.session_for(now)
        if session is None:
            return None
        try:
            hist_start = now - timedelta(days=self.t.history_days)
            b1, b5, b15, quote = await asyncio.gather(
                self.market.get_bars(ticker, Timeframe.M1, session.open - timedelta(hours=5, minutes=30)),  # 04:00 ET
                self.market.get_bars(ticker, Timeframe.M5, hist_start),
                self.market.get_bars(ticker, Timeframe.M15, hist_start),
                self.market.get_quote(ticker),
            )
        except ProviderError as exc:
            log.warning("market data for %s unavailable: %s", ticker, exc)
            return None
        if b1.empty:
            return None

        today_rth_1m = ind.session_slice(b1, session.open, session.close)
        rth5 = ind.regular_hours(b5, self.calendar)
        rth15 = ind.regular_hours(b15, self.calendar)
        price = float(b1["close"].iloc[-1])
        last_bar_close = b1.index[-1].to_pydatetime() + timedelta(minutes=1)

        vw = ind.vwap(today_rth_1m).iloc[-1] if len(today_rth_1m) else None
        ema_f, ema_m, ema_s = (ind.ema(rth5["close"], p).iloc[-1] if len(rth5) else None for p in self.t.ema_periods)
        rsi_v = ind.rsi(rth5["close"], self.t.rsi_period).iloc[-1] if len(rth5) else None
        atr_v = ind.atr(rth5, self.t.atr_period).iloc[-1] if len(rth5) else None
        orb = ind.opening_range(b1, session) if now >= session.orb_end else None
        pm = ind.premarket_range(b1, session)
        prior = ind.prior_day_levels(rth5, session.day)
        rvol_v = ind.relative_volume(rth5, now, self.t.rvol_lookback_days)
        today5 = ind.session_slice(rth5, session.open, session.close)
        swings = ind.swing_levels(today5 if len(today5) >= 3 else rth5, self.t.swing_lookback_bars)

        trend15 = None
        if len(rth15) >= 21:
            f, m = ind.ema(rth15["close"], 9).iloc[-1], ind.ema(rth15["close"], 21).iloc[-1]
            trend15 = Direction.LONG if f > m else Direction.SHORT if f < m else None

        def fin(x):
            return None if x is None or pd.isna(x) else float(x)

        spread = quote.spread_pct if quote is not None else None
        snap = IndicatorSnapshot(
            ticker=ticker, timestamp=b1.index[-1].to_pydatetime(), price=price,
            vwap=fin(vw), ema9=fin(ema_f), ema21=fin(ema_m), ema50=fin(ema_s),
            rsi14=fin(rsi_v), atr14=fin(atr_v), rvol=fin(rvol_v),
            orb_high=orb[0] if orb else None, orb_low=orb[1] if orb else None,
            premarket_high=pm[0] if pm else None, premarket_low=pm[1] if pm else None,
            spread_pct=spread, swing_high=swings[0] if swings else None, swing_low=swings[1] if swings else None,
        )
        return TickerContext(
            ticker=ticker, snapshot=snap, session=session, quote=quote,
            has_real_quotes=self._real_quotes(ticker) and quote is not None,
            last_bar_close=last_bar_close,
            prior_high=prior[0] if prior else None, prior_low=prior[1] if prior else None,
            prior_close=prior[2] if prior else None, trend_15m=trend15, bars_5m=today5,
            instrument=self.router.instrument(ticker),
        )

    async def _build_crypto_context(self, ticker: str) -> TickerContext | None:
        """24/7 crypto: VWAP anchored at 00:00 UTC (on 5m bars - Kraken keeps only 12h of 1m),
        EMAs/RSI/ATR/swings/range on the setup timeframe (15m default), 1h trend, prior
        UTC-day levels, and a 7-day time-of-day RVOL baseline."""
        c = self.cfg.crypto
        now = self.clock.now()
        session = self.router.crypto_schedule.session_for(now)
        tf = Timeframe.M15 if c.timeframe == "15m" else Timeframe.M5
        hist_days = 7 if tf is Timeframe.M15 else 2.4
        try:
            b1, b5, bset, quote = await asyncio.gather(
                self.market.get_bars(ticker, Timeframe.M1, now - timedelta(hours=2)),
                self.market.get_bars(ticker, Timeframe.M5, min(session.open, now - timedelta(hours=2))),
                self.market.get_bars(ticker, tf, now - timedelta(days=hist_days)),
                self.market.get_quote(ticker),
            )
        except ProviderError as exc:
            log.warning("crypto data for %s unavailable: %s", ticker, exc)
            return None
        if b1.empty or bset.empty:
            return None
        price = float(b1["close"].iloc[-1])
        today5 = ind.session_slice(b5, session.open, session.close)
        vw = ind.vwap(today5 if len(today5) >= 3 else b5.iloc[-12:]).iloc[-1] if len(b5) else None
        ema_f, ema_m, ema_s = (ind.ema(bset["close"], p).iloc[-1] for p in self.t.ema_periods)
        rsi_v = ind.rsi(bset["close"], self.t.rsi_period).iloc[-1]
        atr_v = ind.atr(bset, self.t.atr_period).iloc[-1]
        prior = ind.prior_day_levels(bset, session.day, tz="UTC")
        rvol_v = ind.relative_volume(bset, now, c.rvol_lookback_days, tz="UTC")
        rng = ind.rolling_range(bset, c.range_lookback_minutes, now)
        setup_bars = bset.iloc[-48:]
        swings = ind.swing_levels(setup_bars, self.t.swing_lookback_bars)
        hourly = ind.resample_ohlcv(bset, 60)
        trend = None
        if len(hourly) >= 21:
            f, m = ind.ema(hourly["close"], 9).iloc[-1], ind.ema(hourly["close"], 21).iloc[-1]
            trend = Direction.LONG if f > m else Direction.SHORT if f < m else None

        def fin(x):
            return None if x is None or pd.isna(x) else float(x)

        snap = IndicatorSnapshot(
            ticker=ticker, timestamp=b1.index[-1].to_pydatetime(), price=price, vwap=fin(vw),
            ema9=fin(ema_f), ema21=fin(ema_m), ema50=fin(ema_s), rsi14=fin(rsi_v), atr14=fin(atr_v), rvol=fin(rvol_v),
            spread_pct=quote.spread_pct if quote is not None else None,
            swing_high=swings[0] if swings else None, swing_low=swings[1] if swings else None,
            range_high=rng[0] if rng else None, range_low=rng[1] if rng else None,
        )
        return TickerContext(
            ticker=ticker, snapshot=snap, session=session, quote=quote,
            has_real_quotes=self._real_quotes(ticker) and quote is not None,
            last_bar_close=b1.index[-1].to_pydatetime() + timedelta(minutes=1),
            prior_high=prior[0] if prior else None, prior_low=prior[1] if prior else None,
            prior_close=prior[2] if prior else None, trend_15m=trend, bars_5m=setup_bars,
            asset_class=AssetClass.CRYPTO, instrument=self.router.instrument(ticker, price),
        )

    async def macro_bias(self, asset_class: AssetClass = AssetClass.EQUITY) -> MacroBias:
        """SPY/QQQ for equities; BTC (crypto.macro_symbol) for crypto."""
        votes: dict[str, Direction | None] = {}
        detail: dict[str, str] = {}
        symbols = [self.cfg.crypto.macro_symbol] if asset_class is AssetClass.CRYPTO else self.cfg.app.macro_tickers
        for sym in symbols:
            ctx = await self.build_context(sym)
            if ctx is None or ctx.snapshot.vwap is None or ctx.snapshot.ema9 is None or ctx.snapshot.ema21 is None:
                votes[sym] = None
                detail[sym] = "no data"
                continue
            s = ctx.snapshot
            above_vwap = s.price > s.vwap
            up = s.ema9 > s.ema21
            if above_vwap and up:
                votes[sym] = Direction.LONG
            elif not above_vwap and not up:
                votes[sym] = Direction.SHORT
            else:
                votes[sym] = None
            detail[sym] = f"{'above' if above_vwap else 'below'} VWAP, EMA9 {'>' if up else '<'} EMA21"
        vals = [v for v in votes.values()]
        if vals and all(v is Direction.LONG for v in vals):
            return MacroBias(MarketRegime.RISK_ON, Direction.LONG, detail)
        if vals and all(v is Direction.SHORT for v in vals):
            return MacroBias(MarketRegime.RISK_OFF, Direction.SHORT, detail)
        if all(v is None and detail.get(k) == "no data" for k, v in votes.items()):
            return MacroBias(MarketRegime.UNKNOWN, None, detail)
        return MacroBias(MarketRegime.MIXED, None, detail)

    # ------------------------------------------------------------------ setups

    def detect_setups(self, ctx: TickerContext, bias: MacroBias, on_demand: bool = False) -> list[TradeSetup]:
        """Run every enabled strategy on the last completed setup bar, then apply the shared
        confidence adjustments (RVOL, index trend, higher-timeframe trend) and keep the best
        setup per direction."""
        s = ctx.snapshot
        bars = ctx.bars_5m
        now = self.clock.now()
        if len(bars) < 2 or None in (s.vwap, s.ema9, s.ema21, s.atr14, s.rsi14):
            return []
        sc = StrategyContext(ctx=ctx, cfg=self.cfg, bias=bias, last=bars.iloc[-1], prev=bars.iloc[-2], bars=bars)
        rvol_ok = s.rvol is not None and s.rvol >= self.t.rvol_threshold
        tape = "BTC" if ctx.is_crypto else "SPY/QQQ"
        found: list[TradeSetup] = []
        for strat in self.registry.active():
            if not strat.applies_to(ctx.asset_class):
                continue
            try:
                signals = strat.evaluate(sc)
            except Exception:  # noqa: BLE001 - one broken strategy must not stop the others
                log.exception("strategy %s failed on %s", strat.name, ctx.ticker)
                continue
            for sig in signals:
                conf, why = sig.base_confidence, list(sig.rationale)
                if rvol_ok:
                    conf += 0.1
                    why.append(f"RVOL {s.rvol:.1f}x confirms participation")
                if bias.direction is sig.direction:
                    conf += 0.1
                    why.append(f"aligned with {bias.regime.value} tape ({tape})")
                elif bias.direction is sig.direction.opposite:
                    conf -= 0.15
                    why.append(f"against {bias.regime.value} tape ({tape})")
                if ctx.trend_15m is sig.direction:
                    conf += 0.05
                    why.append("1h trend agrees" if ctx.is_crypto else "15m trend agrees")
                found.append(TradeSetup(
                    ticker=ctx.ticker, direction=sig.direction, setup_type=sig.setup_type, detected_at=now,
                    trigger_price=s.price, confidence=round(min(1.0, max(0.0, conf)), 2), rationale=why, indicators=s,
                    macro_bias=bias.direction, on_demand=on_demand, strategy=sig.strategy,
                ))

        # spot crypto cannot be shorted: don't convene debates that can only end in a veto
        if ctx.instrument is not None and not ctx.instrument.allow_short:
            found = [st for st in found if st.direction is Direction.LONG]
        # keep the best setup per direction
        best: dict[Direction, TradeSetup] = {}
        for st in found:
            if st.direction not in best or st.confidence > best[st.direction].confidence:
                best[st.direction] = st
        return list(best.values())

    def lean(self, ctx: TickerContext, bias: MacroBias) -> TradeSetup:
        """For on-demand analysis when no setup fired: the best *unqualified* idea,
        so the War Room can explain why it is not a trade."""
        s = ctx.snapshot
        up_votes = sum([s.vwap is not None and s.price > s.vwap, s.ema9 is not None and s.ema21 is not None and s.ema9 > s.ema21,
                        ctx.trend_15m is Direction.LONG])
        direction = Direction.LONG if up_votes >= 2 else Direction.SHORT
        return TradeSetup(
            ticker=ctx.ticker, direction=direction,
            setup_type=SetupType.VWAP_BOUNCE if direction is Direction.LONG else SetupType.VWAP_REJECTION,
            detected_at=self.clock.now(), trigger_price=s.price, confidence=0.2,
            rationale=["no qualified setup on the last completed 5m bar",
                       f"structure leans {direction.value.lower()} ({up_votes}/3 bullish signals)"],
            indicators=s, macro_bias=bias.direction, on_demand=True, qualified=False,
        )

    # ------------------------------------------------------------------ scanning

    async def scan(self, tickers: list[str], bias: MacroBias | None = None) -> list[tuple[TradeSetup, TickerContext]]:
        """Build contexts and return setups that are NOT on cooldown, best confidence first."""
        tickers = [t for t in tickers if self.entry_window_open(symbol=t)]
        if not tickers:
            return []
        bias = bias or await self.macro_bias(self.router.asset_class(tickers[0]))
        contexts = [c for c in await asyncio.gather(*(self.build_context(t) for t in tickers)) if c is not None]
        return self.scan_contexts(contexts, bias)

    def scan_contexts(self, contexts: list[TickerContext], bias: MacroBias) -> list[tuple[TradeSetup, TickerContext]]:
        """Setup detection over already-built contexts (the orchestrator builds them once per
        cycle and reuses them for the watchlist display)."""
        out: list[tuple[TradeSetup, TickerContext]] = []
        now = self.clock.now()
        for ctx in contexts:
            if not self.entry_window_open(now, ctx.ticker):
                continue
            self.flag_possible_halt(ctx, [c for c in contexts if c.asset_class is ctx.asset_class], now)
            for st in self.detect_setups(ctx, bias):
                if self.on_cooldown(st.ticker, st.direction):
                    log.debug("cooldown: skip %s %s (%.0fs left)", st.ticker, st.direction.value,
                              self.cooldown_remaining(st.ticker, st.direction))
                    continue
                out.append((st, ctx))
        return sorted(out, key=lambda x: x[0].confidence, reverse=True)

    def entry_window_open(self, now: datetime | None = None, symbol: str | None = None) -> bool:
        """Equities: between the end of the opening range and the entry cutoff (outside it Risk
        would veto everything anyway). Crypto: always open."""
        now = now or self.clock.now()
        if symbol is not None and self.router.is_crypto(symbol):
            return True
        sess = self.calendar.session_for(now)
        if sess is None:
            return False
        hh, mm = map(int, self.t.no_new_entries_after_et.split(":"))
        cutoff = min(et_to_utc(datetime.combine(sess.day, time(hh, mm))), sess.eod_flat)
        return sess.orb_end <= now < cutoff

    def flag_possible_halt(self, ctx: TickerContext, peers: list[TickerContext], now: datetime) -> None:
        """No fresh bar for this ticker while most peers are current: probable halt.
        A data-driven backstop for when the halt feed is slow or unavailable."""
        max_age = self.cfg.risk.max_data_age_seconds
        others = [p for p in peers if p.ticker != ctx.ticker]
        fresh_others = [p for p in others if p.data_age_seconds(now) <= max_age]
        if ctx.data_age_seconds(now) > max_age and others and len(fresh_others) >= max(1, len(others) // 2):
            ctx.possible_halt = True

    async def analyze(self, ticker: str, bias: MacroBias | None = None) -> tuple[TradeSetup, TickerContext] | None:
        """On-demand analysis (bypasses cooldown). Returns the best setup, or an
        unqualified lean when nothing fires, or None without data."""
        ticker = self.router.normalize(ticker)
        bias = bias or await self.macro_bias(self.router.asset_class(ticker))
        ctx = await self.build_context(ticker)
        if ctx is None:
            return None
        setups = self.detect_setups(ctx, bias, on_demand=True)
        if setups:
            return max(setups, key=lambda s: s.confidence), ctx
        return self.lean(ctx, bias), ctx
