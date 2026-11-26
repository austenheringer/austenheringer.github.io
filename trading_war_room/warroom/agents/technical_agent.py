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
    bars_5m: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    possible_halt: bool = False

    def data_age_seconds(self, now: datetime) -> float:
        return (ensure_utc(now) - self.last_bar_close).total_seconds()


@dataclass
class MacroBias:
    regime: MarketRegime
    direction: Direction | None
    detail: dict[str, str]


class TechnicalAgent:
    def __init__(self, cfg: AppConfig, market: MarketDataProvider, calendar: MarketCalendar, clock: Clock) -> None:
        self.cfg = cfg
        self.t = cfg.technical
        self.market = market
        self.calendar = calendar
        self.clock = clock
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

    async def build_context(self, ticker: str) -> TickerContext | None:
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
            has_real_quotes=self.market.has_real_quotes and quote is not None,
            last_bar_close=last_bar_close,
            prior_high=prior[0] if prior else None, prior_low=prior[1] if prior else None,
            prior_close=prior[2] if prior else None, trend_15m=trend15, bars_5m=today5,
        )

    async def macro_bias(self) -> MacroBias:
        votes: dict[str, Direction | None] = {}
        detail: dict[str, str] = {}
        for sym in self.cfg.app.macro_tickers:
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
        s = ctx.snapshot
        bars = ctx.bars_5m
        now = self.clock.now()
        if len(bars) < 2 or None in (s.vwap, s.ema9, s.ema21, s.atr14, s.rsi14):
            return []
        last, prev = bars.iloc[-1], bars.iloc[-2]
        found: list[TradeSetup] = []
        rvol_ok = s.rvol is not None and s.rvol >= self.t.rvol_threshold
        touch = self.t.vwap_touch_pct
        rng = max(last["high"] - last["low"], 1e-9)
        close_pos = (last["close"] - last["low"]) / rng  # 0 = closed at low, 1 = at high

        def add(direction: Direction, setup: SetupType, base: float, why: list[str]) -> None:
            conf = base
            if rvol_ok:
                conf += 0.1
                why.append(f"RVOL {s.rvol:.1f}x confirms participation")
            if bias.direction is direction:
                conf += 0.1
                why.append(f"aligned with {bias.regime.value} tape (SPY/QQQ)")
            elif bias.direction is direction.opposite:
                conf -= 0.15
                why.append(f"against {bias.regime.value} tape (SPY/QQQ)")
            if ctx.trend_15m is direction:
                conf += 0.05
                why.append("15m trend agrees")
            found.append(TradeSetup(
                ticker=ctx.ticker, direction=direction, setup_type=setup, detected_at=now, trigger_price=s.price,
                confidence=round(min(1.0, max(0.0, conf)), 2), rationale=why, indicators=s,
                macro_bias=bias.direction, on_demand=on_demand,
            ))

        # --- Opening range breakout / breakdown (fresh break on the last completed 5m bar)
        if s.orb_high is not None and s.orb_low is not None:
            # no chasing exhausted moves: RSI > 80 into a breakout / < 20 into a breakdown
            if last["close"] > s.orb_high >= prev["close"] and s.price > s.vwap and s.rsi14 <= 80:
                add(Direction.LONG, SetupType.ORB_BREAKOUT, 0.5,
                    [f"5m close {last['close']:.2f} broke ORB high {s.orb_high:.2f}", "holding above VWAP"])
            if last["close"] < s.orb_low <= prev["close"] and s.price < s.vwap and s.rsi14 >= 20:
                add(Direction.SHORT, SetupType.ORB_BREAKDOWN, 0.5,
                    [f"5m close {last['close']:.2f} broke ORB low {s.orb_low:.2f}", "holding below VWAP"])

        # --- VWAP bounce / rejection
        if last["low"] <= s.vwap * (1 + touch) and last["close"] > s.vwap and last["close"] > last["open"] and s.ema9 > s.ema21:
            add(Direction.LONG, SetupType.VWAP_BOUNCE, 0.45,
                [f"tested VWAP {s.vwap:.2f} and closed back above", "EMA9 above EMA21"])
        if last["high"] >= s.vwap * (1 - touch) and last["close"] < s.vwap and last["close"] < last["open"] and s.ema9 < s.ema21:
            add(Direction.SHORT, SetupType.VWAP_REJECTION, 0.45,
                [f"rejected at VWAP {s.vwap:.2f}", "EMA9 below EMA21"])

        # --- EMA pullback in an aligned trend
        if s.ema50 is not None:
            if (s.ema9 > s.ema21 > s.ema50 and last["low"] <= s.ema9 and last["close"] > s.ema9
                    and 40 <= s.rsi14 <= 65 and s.price > s.vwap):
                add(Direction.LONG, SetupType.EMA_PULLBACK, 0.45,
                    [f"pullback to EMA9 {s.ema9:.2f} in a 9>21>50 uptrend", f"RSI {s.rsi14:.0f} reset, not overbought"])
            if (s.ema9 < s.ema21 < s.ema50 and last["high"] >= s.ema9 and last["close"] < s.ema9
                    and 35 <= s.rsi14 <= 60 and s.price < s.vwap):
                add(Direction.SHORT, SetupType.EMA_PULLBACK, 0.45,
                    [f"rally into EMA9 {s.ema9:.2f} in a 9<21<50 downtrend", f"RSI {s.rsi14:.0f} reset, not oversold"])

        # --- High-RVOL momentum
        if s.rvol is not None and s.rvol >= self.t.rvol_threshold * 1.3:
            if s.price > s.vwap and s.ema9 > s.ema21 and 55 <= s.rsi14 <= 75 and last["close"] > last["open"] and close_pos >= 0.7:
                add(Direction.LONG, SetupType.RVOL_MOMENTUM, 0.4,
                    [f"RVOL {s.rvol:.1f}x momentum bar closing near highs", f"RSI {s.rsi14:.0f}"])
            if s.price < s.vwap and s.ema9 < s.ema21 and 25 <= s.rsi14 <= 45 and last["close"] < last["open"] and close_pos <= 0.3:
                add(Direction.SHORT, SetupType.RVOL_MOMENTUM, 0.4,
                    [f"RVOL {s.rvol:.1f}x momentum bar closing near lows", f"RSI {s.rsi14:.0f}"])

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
        if not self.entry_window_open():
            return []
        bias = bias or await self.macro_bias()
        contexts = [c for c in await asyncio.gather(*(self.build_context(t) for t in tickers)) if c is not None]
        return self.scan_contexts(contexts, bias)

    def scan_contexts(self, contexts: list[TickerContext], bias: MacroBias) -> list[tuple[TradeSetup, TickerContext]]:
        """Setup detection over already-built contexts (the orchestrator builds them once per
        cycle and reuses them for the watchlist display)."""
        if not self.entry_window_open():
            return []
        out: list[tuple[TradeSetup, TickerContext]] = []
        now = self.clock.now()
        for ctx in contexts:
            self.flag_possible_halt(ctx, contexts, now)
            for st in self.detect_setups(ctx, bias):
                if self.on_cooldown(st.ticker, st.direction):
                    log.debug("cooldown: skip %s %s (%.0fs left)", st.ticker, st.direction.value,
                              self.cooldown_remaining(st.ticker, st.direction))
                    continue
                out.append((st, ctx))
        return sorted(out, key=lambda x: x[0].confidence, reverse=True)

    def entry_window_open(self, now: datetime | None = None) -> bool:
        """Scanning only proposes setups between the end of the opening range and the
        entry cutoff; outside that window Risk would veto everything anyway."""
        now = now or self.clock.now()
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
        bias = bias or await self.macro_bias()
        ctx = await self.build_context(ticker.upper())
        if ctx is None:
            return None
        setups = self.detect_setups(ctx, bias, on_demand=True)
        if setups:
            return max(setups, key=lambda s: s.confidence), ctx
        return self.lean(ctx, bias), ctx
