"""Risk Management & Portfolio Guard Agent - the absolute veto.

Level math (all pure functions, unit-tested):

* Entry estimate: ask (LONG) / bid (SHORT) when a real quote exists, else
  last price; then adverse slippage is applied, so size is computed on the
  price we expect to actually pay.
* Stop (``stop_policy: structure_within_atr``): swing low/high +/- a small
  ATR buffer, used only if it is on the correct side of entry and no farther
  than ``atr_stop_multiplier`` x ATR; otherwise the ATR stop. Never tighter
  than ``min_stop_atr`` x ATR (noise would take it out).
* Targets: ``tp1_r_multiple`` / ``tp2_r_multiple`` x R, then pulled in to just
  before the nearest opposing levels (prior-day H/L, premarket H/L, ORB H/L,
  swing H/L) under ``target_policy: structural_cap``. R:R is measured
  entry -> TP1 AFTER capping.
* Ordering invariants: LONG stop < entry < tp1 < tp2; SHORT the mirror.
  Zero/negative risk per share is an immediate veto.

Sizing: ``min(floor(equity*max_risk_pct / R), floor(cash*max_position_pct / entry))``,
then reduced to fit the remaining daily-loss budget; veto if < 1 share.

Every applicable veto reason is collected (not short-circuited) so "why did
you veto?" can be answered completely.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from warroom.config import AppConfig
from warroom.core.clock import Clock, ensure_utc, et_to_utc, to_et
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import (
    Direction,
    Mode,
    NewsAssessment,
    Position,
    RiskAssessment,
    RiskState,
    Severity,
    TradeSetup,
    validate_levels,
)

log = logging.getLogger(__name__)


# ============================================================================ pure math


@dataclass
class StopResult:
    stop: float
    source: str               # "structure" | "atr" | "min_distance"
    atr_multiple: float       # stop distance in ATRs


def compute_stop(direction: Direction, entry: float, atr: float, structural: float | None,
                 atr_mult: float, buffer_atr: float, min_stop_atr: float,
                 policy: str = "structure_within_atr") -> StopResult:
    """Choose the stop level. Raises ValueError on non-positive ATR."""
    if not (atr > 0) or not (entry > 0):
        raise ValueError(f"invalid entry/ATR ({entry}, {atr})")
    s = direction.sign
    atr_stop = entry - s * atr_mult * atr
    stop, source = atr_stop, "atr"
    if policy == "structure_within_atr" and structural is not None:
        struct_stop = structural - s * buffer_atr * atr
        dist = (entry - struct_stop) * s
        if 0 < dist <= atr_mult * atr:
            stop, source = struct_stop, "structure"
    min_dist = min_stop_atr * atr
    if (entry - stop) * s < min_dist:
        stop, source = entry - s * min_dist, "min_distance"
    return StopResult(stop=stop, source=source, atr_multiple=abs(entry - stop) / atr)


@dataclass
class TargetResult:
    tp1: float
    tp2: float
    capped_by: list[str] = field(default_factory=list)


def compute_targets(direction: Direction, entry: float, stop: float, tp1_r: float, tp2_r: float,
                    levels: dict[str, float | None], atr: float, policy: str = "structural_cap") -> TargetResult:
    """R-multiple targets, optionally capped just before opposing structure."""
    s = direction.sign
    r = abs(entry - stop)
    tp1, tp2 = entry + s * tp1_r * r, entry + s * tp2_r * r
    capped: list[str] = []
    if policy == "structural_cap":
        front_run = 0.05 * atr  # exit slightly before the level, where offers/bids stack
        ahead = sorted(((v - entry) * s, name, v) for name, v in levels.items() if v is not None and (v - entry) * s > 0)
        if ahead:
            _, name1, lvl1 = ahead[0]
            cap1 = lvl1 - s * front_run
            if (tp1 - cap1) * s > 0:
                tp1 = cap1
                capped.append(f"TP1 capped by {name1} {lvl1:.2f}")
            beyond = [a for a in ahead if (a[2] - tp1) * s > front_run]
            if beyond:
                _, name2, lvl2 = beyond[0]
                cap2 = lvl2 - s * front_run
                if (tp2 - cap2) * s > 0:
                    tp2 = cap2
                    capped.append(f"TP2 capped by {name2} {lvl2:.2f}")
            elif (tp2 - cap1) * s > 0 and capped:
                tp2 = tp1 + s * 0.25 * r  # nothing beyond the first wall: runner gets a token extension
                capped.append("TP2 limited: no clear air beyond TP1")
        if (tp2 - tp1) * s <= 0:
            tp2 = tp1 + s * 0.25 * r
    return TargetResult(tp1=tp1, tp2=tp2, capped_by=capped)


def position_size(equity: float, available_cash: float, entry: float, stop: float, max_risk_pct: float,
                  max_position_pct: float) -> tuple[int, int, int]:
    """(risk_shares, buying_power_shares, position_size). Zero distance -> zeros."""
    dist = abs(entry - stop)
    if dist <= 0 or entry <= 0 or equity <= 0:
        return 0, 0, 0
    risk_shares = math.floor((equity * max_risk_pct) / dist)
    bp_shares = math.floor((max(0.0, available_cash) * max_position_pct) / entry)
    return risk_shares, bp_shares, max(0, min(risk_shares, bp_shares))


def reward_risk(direction: Direction, entry: float, stop: float, target: float) -> float:
    r = abs(entry - stop)
    return ((target - entry) * direction.sign) / r if r > 0 else 0.0


# ============================================================================ agent


@dataclass
class PortfolioView:
    equity: float
    available_cash: float
    start_of_day_equity: float
    unrealized_pnl: float
    open_positions: list[Position]


@dataclass
class RiskNotes:
    """Context the War Room uses to argue; not part of the persisted verdict."""

    stop_source: str = ""
    stop_atr_multiple: float = 0.0
    target_caps: list[str] = field(default_factory=list)
    blended_rr: float | None = None
    counter_trend: bool = False
    day_pnl: float = 0.0
    daily_budget_left: float = 0.0
    entry_basis: str = ""
    data_age_seconds: float | None = None
    stop_pct: float | None = None


class RiskAgent:
    def __init__(self, cfg: AppConfig, calendar: MarketCalendar, clock: Clock, mode: Mode,
                 state: RiskState | None = None) -> None:
        self.cfg = cfg
        self.r = cfg.risk
        self.calendar = calendar
        self.clock = clock
        self.mode = mode
        self.state = state or RiskState(trading_date=clock.now_et().date())

    # ------------------------------------------------------------------ state

    def roll_day(self) -> bool:
        today = self.clock.now_et().date()
        if self.state.trading_date == today:
            return False
        self.state = RiskState(trading_date=today, entries_paused=self.state.entries_paused,
                               day_trade_dates=list(self.state.day_trade_dates))
        return True

    def day_trades_in_window(self, today: date | None = None) -> int:
        today = today or self.clock.now_et().date()
        window = set(self.calendar.previous_trading_days(today, 5))
        return sum(1 for d in self.state.day_trade_dates if d in window)

    def on_fill(self, realized_pnl: float, commission: float) -> None:
        self.roll_day()
        self.state.realized_pnl_today = round(self.state.realized_pnl_today + realized_pnl - commission, 2)

    def on_position_closed(self, p: Position) -> None:
        self.roll_day()
        now = self.clock.now()
        net = p.realized_pnl - p.commissions
        if net < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_at = p.closed_at or now
            if self.state.consecutive_losses >= self.r.consecutive_loss_limit:
                self.state.cooldown_until = (p.closed_at or now) + timedelta(minutes=self.r.consecutive_loss_cooldown_minutes)
        else:
            self.state.consecutive_losses = 0
        if p.closed_at and to_et(p.opened_at).date() == to_et(p.closed_at).date():
            self.state.day_trade_dates = [*self.state.day_trade_dates, to_et(p.closed_at).date()]

    def pause(self) -> None:
        self.state.entries_paused = True

    def resume(self) -> None:
        self.state.entries_paused = False

    def day_pnl(self, portfolio: PortfolioView) -> float:
        pnl = self.state.realized_pnl_today
        if self.r.daily_loss_includes_unrealized:
            pnl += portfolio.unrealized_pnl
        return pnl

    def daily_limit(self, portfolio: PortfolioView) -> float:
        return self.r.max_daily_loss_pct * portfolio.start_of_day_equity

    def check_daily_halt(self, portfolio: PortfolioView) -> bool:
        if self.day_pnl(portfolio) <= -self.daily_limit(portfolio):
            self.state.halted_for_day = True
        return self.state.halted_for_day

    def circuit_status(self, portfolio: PortfolioView | None = None) -> str:
        now = self.clock.now()
        if self.state.halted_for_day:
            return "HALTED (daily loss limit)"
        if self.state.cooldown_until and now < self.state.cooldown_until:
            return f"COOLDOWN until {to_et(self.state.cooldown_until):%H:%M} ET"
        if self.state.entries_paused:
            return "PAUSED (manual)"
        return "ARMED"

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, setup: TradeSetup, ctx, news: NewsAssessment, portfolio: PortfolioView,
                 macro_direction: Direction | None) -> tuple[RiskAssessment, RiskNotes]:
        """Full risk review. ``ctx`` is a technical_agent.TickerContext."""
        self.roll_day()
        now = self.clock.now()
        d = setup.direction
        s = d.sign
        snap = ctx.snapshot
        vetoes: list[str] = []
        warns: list[str] = []
        notes = RiskNotes(counter_trend=macro_direction is d.opposite)

        # ---- circuit breakers & portfolio guards
        if self.state.entries_paused:
            vetoes.append("new entries are paused by the user")
        day_pnl = self.day_pnl(portfolio)
        limit = self.daily_limit(portfolio)
        notes.day_pnl = round(day_pnl, 2)
        budget_left = limit + day_pnl
        notes.daily_budget_left = round(budget_left, 2)
        if self.check_daily_halt(portfolio):
            vetoes.append(f"daily loss limit hit: day P&L {day_pnl:+.2f} vs limit -{limit:.2f}; entries halted for the day")
        if self.state.cooldown_until and now < self.state.cooldown_until:
            vetoes.append(f"{self.state.consecutive_losses} consecutive losses: cooling down until "
                          f"{to_et(self.state.cooldown_until):%H:%M} ET")
        open_pos = [p for p in portfolio.open_positions if p.qty > 0]
        if len(open_pos) >= self.r.max_open_positions:
            vetoes.append(f"max concurrent positions reached ({len(open_pos)}/{self.r.max_open_positions})")
        if any(p.ticker == setup.ticker for p in open_pos):
            vetoes.append(f"already exposed to {setup.ticker}: no duplicate positions")
        if self.r.enforce_pdt_rule and portfolio.equity < self.r.pdt_equity_threshold:
            n = self.day_trades_in_window()
            if n >= self.r.pdt_max_day_trades:
                vetoes.append(f"PDT rule: {n} day trades in the last 5 sessions with equity "
                              f"${portfolio.equity:,.0f} < ${self.r.pdt_equity_threshold:,.0f}")

        # ---- session timing
        sess = ctx.session
        cutoff_h, cutoff_m = map(int, self.cfg.technical.no_new_entries_after_et.split(":"))
        cutoff = min(et_to_utc(datetime.combine(sess.day, time(cutoff_h, cutoff_m))), sess.eod_flat)
        if not sess.contains(now):
            vetoes.append("market is closed")
        elif now < sess.orb_end:
            vetoes.append(f"inside the opening-range window (until {to_et(sess.orb_end):%H:%M} ET)")
        elif now >= cutoff:
            vetoes.append(f"past the {to_et(cutoff):%H:%M} ET cutoff for new entries")

        # ---- data quality
        age = ctx.data_age_seconds(now)
        notes.data_age_seconds = round(age, 1)
        if age > self.r.max_data_age_seconds:
            msg = f"stale data: latest candle is {age:.0f}s old (max {self.r.max_data_age_seconds}s)"
            if getattr(ctx, "possible_halt", False):
                msg += "; peers are current, so the symbol may be halted"
            vetoes.append(msg)
        if ctx.has_real_quotes and ctx.quote is not None:
            if ctx.quote.spread_pct > self.r.max_spread_pct:
                vetoes.append(f"spread {ctx.quote.spread_pct:.3%} exceeds {self.r.max_spread_pct:.3%}")
        elif self.r.unknown_spread_policy == "veto":
            vetoes.append("no real bid/ask available: spread cannot be verified (risk.unknown_spread_policy=veto)")
        else:
            warns.append("spread unverified (no real bid/ask)")

        # ---- news / catalysts
        if news.in_blackout:
            vetoes.append(f"news blackout: {news.blackout_reason}")
        if news.conflicts_with_direction and (news.severity is Severity.HIGH or abs(news.sentiment) >= 0.6):
            kind = "high-impact" if news.severity is Severity.HIGH else "strongly"
            vetoes.append(f"{kind} {'news' if kind == 'high-impact' else 'adverse news'} against the "
                          f"{d.value.lower()} (sentiment {news.sentiment:+.2f})")
        elif news.conflicts_with_direction:
            warns.append(f"news sentiment {news.sentiment:+.2f} leans against the trade")
        if not setup.qualified:
            warns.append("no qualified technical trigger")

        # ---- levels
        atr = snap.atr14 or 0.0
        if ctx.has_real_quotes and ctx.quote is not None:
            ref = ctx.quote.ask if d is Direction.LONG else ctx.quote.bid
            notes.entry_basis = "ask" if d is Direction.LONG else "bid"
        else:
            ref = snap.price
            notes.entry_basis = "last"
        entry = ref * (1 + s * self.cfg.execution.slippage_pct)
        entry_r = round(entry, 2)
        stop = tp1 = tp2 = None
        rr = None
        risk_sh = bp_sh = size = 0
        dollar_risk = 0.0
        if atr <= 0:
            vetoes.append("ATR unavailable: cannot place a volatility-based stop")
        else:
            structural = snap.swing_low if d is Direction.LONG else snap.swing_high
            st = compute_stop(d, entry, atr, structural, self.r.atr_stop_multiplier, self.r.stop_buffer_atr,
                              self.r.min_stop_atr, self.r.stop_policy)
            notes.stop_source, notes.stop_atr_multiple = st.source, round(st.atr_multiple, 2)
            stop_r = round(st.stop, 2)
            # Only levels established BEFORE the signal count as opposing structure. The recent
            # swing high/low is excluded: after a breakout it is the bar that just broke out.
            levels = (
                {"prior-day high": ctx.prior_high, "premarket high": snap.premarket_high, "ORB high": snap.orb_high}
                if d is Direction.LONG else
                {"prior-day low": ctx.prior_low, "premarket low": snap.premarket_low, "ORB low": snap.orb_low}
            )
            tg = compute_targets(d, entry_r, stop_r, self.r.tp1_r_multiple, self.r.tp2_r_multiple, levels, atr,
                                 self.r.target_policy)
            stop, tp1, tp2 = stop_r, round(tg.tp1, 2), round(tg.tp2, 2)
            notes.target_caps = tg.capped_by
            bad = validate_levels(d, entry_r, stop, tp1, tp2)
            if bad and tg.capped_by and (tp1 - entry_r) * s <= 0:
                vetoes.append(f"no room to a first target: {tg.capped_by[0]} sits right at entry")
                rr = 0.0
            elif bad:
                vetoes.append(f"invalid levels: {bad}")
            else:
                rr = round(reward_risk(d, entry_r, stop, tp1), 2)
                rr2 = reward_risk(d, entry_r, stop, tp2)
                frac = self.r.tp1_scale_out_pct
                notes.blended_rr = round(frac * rr + (1 - frac) * rr2, 2)
                if rr < self.r.min_reward_risk:
                    why = f" ({tg.capped_by[0]})" if tg.capped_by else ""
                    vetoes.append(f"R:R {rr:.2f} to TP1 is below {self.r.min_reward_risk:.1f}{why}")
                if notes.counter_trend and notes.blended_rr < self.r.counter_trend_min_reward_risk:
                    vetoes.append(f"counter-trend vs SPY/QQQ: blended R:R {notes.blended_rr:.2f} is below "
                                  f"{self.r.counter_trend_min_reward_risk:.1f}")
                notes.stop_pct = round(abs(entry_r - stop) / entry_r, 4)
                if notes.stop_pct > 0.02:
                    warns.append(f"wide stop ({notes.stop_pct:.1%} of price); size scaled down accordingly")

                risk_sh, bp_sh, size = position_size(portfolio.equity, portfolio.available_cash, entry_r, stop,
                                                     self.r.max_risk_pct, self.r.max_position_pct)
                per_share = abs(entry_r - stop)
                if d is Direction.SHORT:
                    margin_cap = math.floor(max(0.0, portfolio.available_cash) /
                                            (entry_r * self.cfg.execution.short_margin_requirement))
                    size = min(size, margin_cap)
                if size >= 1 and budget_left > 0 and size * per_share > budget_left:
                    reduced = math.floor(budget_left / per_share)
                    warns.append(f"size cut {size} -> {reduced} shares to fit the remaining daily loss budget "
                                 f"(${budget_left:,.2f})")
                    size = reduced
                if size < 1:
                    vetoes.append(f"position size < 1 share (risk cap {risk_sh}, buying-power cap {bp_sh})")
                dollar_risk = round(size * per_share, 2)

        approved = not vetoes
        assessment = RiskAssessment(
            ticker=setup.ticker, direction=d, approved=approved, veto_reasons=vetoes, warnings=warns,
            entry=entry_r, stop_loss=stop, tp1=tp1, tp2=tp2, risk_shares=risk_sh, buying_power_shares=bp_sh,
            position_size=size if approved else max(0, size), dollar_risk=dollar_risk if size >= 1 else 0.0,
            reward_risk=rr,
        )
        return assessment, notes
