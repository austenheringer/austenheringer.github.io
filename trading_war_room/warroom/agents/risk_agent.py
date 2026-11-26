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
from warroom.core.assets import AssetRouter, format_qty
from warroom.core.clock import Clock, et_to_utc, to_et
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


@dataclass(frozen=True)
class FeeModel:
    """Trading costs. ``taker_pct`` applies to market orders (entry, stop, flatten),
    ``maker_pct`` to resting limit targets, ``per_unit`` is a per-share commission."""

    taker_pct: float = 0.0
    maker_pct: float = 0.0
    per_unit: float = 0.0

    @property
    def free(self) -> bool:
        return self.taker_pct == 0 and self.maker_pct == 0 and self.per_unit == 0


def risk_per_unit(entry: float, stop: float, fees: FeeModel | None = None) -> float:
    """Loss per unit if stopped: price distance + entry fee + stop (market) fee."""
    f = fees or FeeModel()
    return abs(entry - stop) + entry * f.taker_pct + stop * f.taker_pct + 2 * f.per_unit


def net_reward(direction: Direction, entry: float, target: float, fees: FeeModel | None = None) -> float:
    """Gain per unit at a limit target after the entry (taker) and target (maker) fees."""
    f = fees or FeeModel()
    return (target - entry) * direction.sign - entry * f.taker_pct - target * f.maker_pct - 2 * f.per_unit


def net_reward_risk(direction: Direction, entry: float, stop: float, target: float, fees: FeeModel | None = None) -> float:
    r = risk_per_unit(entry, stop, fees)
    return net_reward(direction, entry, target, fees) / r if r > 0 else 0.0


def fee_aware_target(direction: Direction, entry: float, stop: float, k: float, fees: FeeModel | None = None) -> float:
    """Price at which the NET reward equals ``k`` x the NET risk."""
    f = fees or FeeModel()
    want = k * risk_per_unit(entry, stop, f) + 2 * f.per_unit
    if direction is Direction.LONG:
        return (entry * (1 + f.taker_pct) + want) / (1 - f.maker_pct)
    return (entry * (1 - f.taker_pct) - want) / (1 + f.maker_pct)


@dataclass
class TargetResult:
    tp1: float
    tp2: float
    capped_by: list[str] = field(default_factory=list)


def compute_targets(direction: Direction, entry: float, stop: float, tp1_r: float, tp2_r: float,
                    levels: dict[str, float | None], atr: float, policy: str = "structural_cap",
                    fees: FeeModel | None = None) -> TargetResult:
    """R-multiple targets (net of fees when a FeeModel is given), optionally capped
    just before opposing structure."""
    s = direction.sign
    r = abs(entry - stop)
    if fees is None or fees.free:
        tp1, tp2 = entry + s * tp1_r * r, entry + s * tp2_r * r
    else:
        tp1, tp2 = fee_aware_target(direction, entry, stop, tp1_r, fees), fee_aware_target(direction, entry, stop, tp2_r, fees)
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
                  max_position_pct: float, *, per_unit_risk: float | None = None,
                  entry_cost: float | None = None) -> tuple[float, float, float]:
    """(risk_units, buying_power_units, position_size). Zero distance -> zeros.

    Defaults are the spec's whole-share formulas:
      risk_shares = floor(equity * max_risk_pct / |entry - stop|)
      buying_power_shares = floor(cash * max_position_pct / entry)
    ``per_unit_risk`` / ``entry_cost`` switch to fee-inclusive, unfloored values; the
    caller then rounds down to the instrument's lot step (fractional for crypto)."""
    dist = abs(entry - stop)
    if dist <= 0 or entry <= 0 or equity <= 0:
        return 0, 0, 0
    if per_unit_risk is None and entry_cost is None:
        risk_shares = math.floor((equity * max_risk_pct) / dist)
        bp_shares = math.floor((max(0.0, available_cash) * max_position_pct) / entry)
        return risk_shares, bp_shares, max(0, min(risk_shares, bp_shares))
    risk_units = (equity * max_risk_pct) / (per_unit_risk or dist)
    bp_units = (max(0.0, available_cash) * max_position_pct) / (entry_cost or entry)
    return risk_units, bp_units, max(0.0, min(risk_units, bp_units))


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
    est_fees: float = 0.0


class RiskAgent:
    def __init__(self, cfg: AppConfig, calendar: MarketCalendar, clock: Clock, mode: Mode,
                 state: RiskState | None = None, router: AssetRouter | None = None) -> None:
        self.cfg = cfg
        self.router = router or AssetRouter(calendar, cfg.crypto)
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
        # PDT is an equities rule: crypto round trips do not count
        if p.closed_at and not self.router.is_crypto(p.ticker) and to_et(p.opened_at).date() == to_et(p.closed_at).date():
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

        inst = ctx.instrument or self.router.instrument(setup.ticker, snap.price)
        is_crypto = inst.is_crypto
        risk_pct = self.r.max_risk_pct

        # ---- asset-class guards
        if is_crypto:
            c = self.cfg.crypto
            if d is Direction.SHORT and not inst.allow_short:
                vetoes.append(f"short selling is not available for {setup.ticker} on a spot account (crypto.allow_short=false)")
            n_crypto = sum(1 for p in open_pos if self.router.is_crypto(p.ticker))
            if n_crypto >= c.max_open_positions:
                vetoes.append(f"max concurrent crypto positions reached ({n_crypto}/{c.max_open_positions})")
            if now.weekday() >= 5 and c.weekend_risk_multiplier < 1:
                risk_pct *= c.weekend_risk_multiplier
                warns.append(f"weekend: risk per trade cut to {risk_pct:.2%} (thinner order books)")

        # ---- session timing (equities only: crypto never closes)
        if not is_crypto:
            sess = ctx.session
            cutoff_h, cutoff_m = map(int, self.cfg.technical.no_new_entries_after_et.split(":"))
            cutoff = et_to_utc(datetime.combine(sess.day, time(cutoff_h, cutoff_m)))
            if sess.eod_flat is not None:
                cutoff = min(cutoff, sess.eod_flat)
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
        max_spread = self.cfg.crypto.max_spread_pct if is_crypto else self.r.max_spread_pct
        if ctx.has_real_quotes and ctx.quote is not None:
            if ctx.quote.spread_pct > max_spread:
                vetoes.append(f"spread {ctx.quote.spread_pct:.3%} exceeds {max_spread:.3%}")
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

        # ---- levels (fee-aware)
        fees = FeeModel(inst.taker_fee_pct, inst.maker_fee_pct,
                        0.0 if is_crypto else self.cfg.execution.commission_per_share)
        atr = snap.atr14 or 0.0
        if ctx.has_real_quotes and ctx.quote is not None:
            ref = ctx.quote.ask if d is Direction.LONG else ctx.quote.bid
            notes.entry_basis = "ask" if d is Direction.LONG else "bid"
        else:
            ref = snap.price
            notes.entry_basis = "last"
        entry_r = inst.round_price(ref * (1 + s * self.cfg.execution.slippage_pct))
        stop = tp1 = tp2 = None
        rr = None
        risk_sh = bp_sh = size = 0.0
        dollar_risk = 0.0
        if atr <= 0:
            vetoes.append("ATR unavailable: cannot place a volatility-based stop")
        else:
            structural = snap.swing_low if d is Direction.LONG else snap.swing_high
            st = compute_stop(d, entry_r, atr, structural, self.r.atr_stop_multiplier, self.r.stop_buffer_atr,
                              self.r.min_stop_atr, self.r.stop_policy)
            notes.stop_source, notes.stop_atr_multiple = st.source, round(st.atr_multiple, 2)
            stop = inst.round_price(st.stop)
            # Only levels established BEFORE the signal count as opposing structure. The recent
            # swing high/low (and the breakout range itself) is excluded.
            if is_crypto:
                levels = ({"prior-day high": ctx.prior_high} if d is Direction.LONG else {"prior-day low": ctx.prior_low})
            else:
                levels = (
                    {"prior-day high": ctx.prior_high, "premarket high": snap.premarket_high, "ORB high": snap.orb_high}
                    if d is Direction.LONG else
                    {"prior-day low": ctx.prior_low, "premarket low": snap.premarket_low, "ORB low": snap.orb_low}
                )
            tg = compute_targets(d, entry_r, stop, self.r.tp1_r_multiple, self.r.tp2_r_multiple, levels, atr,
                                 self.r.target_policy, fees=fees)
            tp1, tp2 = inst.round_price(tg.tp1), inst.round_price(tg.tp2)
            notes.target_caps = tg.capped_by
            bad = validate_levels(d, entry_r, stop, tp1, tp2)
            if bad and tg.capped_by and (tp1 - entry_r) * s <= 0:
                vetoes.append(f"no room to a first target: {tg.capped_by[0]} sits right at entry")
                rr = 0.0
            elif bad:
                vetoes.append(f"invalid levels: {bad}")
            else:
                rr = round(net_reward_risk(d, entry_r, stop, tp1, fees), 2)
                rr2 = net_reward_risk(d, entry_r, stop, tp2, fees)
                frac = self.r.tp1_scale_out_pct
                notes.blended_rr = round(frac * rr + (1 - frac) * rr2, 2)
                if net_reward(d, entry_r, tp1, fees) <= 0:
                    vetoes.append("fees exceed the reward to TP1")
                elif rr < self.r.min_reward_risk:
                    why = f" ({tg.capped_by[0]})" if tg.capped_by else ""
                    vetoes.append(f"R:R {rr:.2f} to TP1 is below {self.r.min_reward_risk:.1f}{why}")
                if notes.counter_trend and notes.blended_rr < self.r.counter_trend_min_reward_risk:
                    vetoes.append(f"counter-trend vs {'BTC' if is_crypto else 'SPY/QQQ'}: blended R:R "
                                  f"{notes.blended_rr:.2f} is below {self.r.counter_trend_min_reward_risk:.1f}")
                notes.stop_pct = round(abs(entry_r - stop) / entry_r, 4)
                wide = self.cfg.crypto.wide_stop_pct if is_crypto else 0.02
                if notes.stop_pct > wide:
                    warns.append(f"wide stop ({notes.stop_pct:.1%} of price); size scaled down accordingly")

                per_unit = risk_per_unit(entry_r, stop, fees)  # price risk + entry/exit fees
                risk_sh, bp_sh, size = position_size(portfolio.equity, portfolio.available_cash, entry_r, stop,
                                                     risk_pct, self.r.max_position_pct, per_unit_risk=per_unit,
                                                     entry_cost=entry_r * (1 + fees.taker_pct) + fees.per_unit)
                risk_sh, bp_sh, size = inst.floor_qty(risk_sh), inst.floor_qty(bp_sh), inst.floor_qty(size)
                if d is Direction.SHORT:
                    margin_cap = inst.floor_qty(max(0.0, portfolio.available_cash) /
                                                (entry_r * self.cfg.execution.short_margin_requirement))
                    size = min(size, margin_cap)
                if size > 0 and budget_left > 0 and size * per_unit > budget_left:
                    reduced = inst.floor_qty(budget_left / per_unit)
                    warns.append(f"size cut {format_qty(setup.ticker, size)} -> {format_qty(setup.ticker, reduced)} "
                                 f"to fit the remaining daily loss budget (${budget_left:,.2f})")
                    size = reduced
                if size <= 0 or not inst.meets_minimum(size, entry_r):
                    if is_crypto and size > 0:
                        vetoes.append(f"size {format_qty(setup.ticker, size)} is below the exchange minimum "
                                      f"({format_qty(setup.ticker, inst.min_qty)} / ${inst.min_notional:,.2f})")
                    else:
                        vetoes.append(f"position size < 1 share (risk cap {risk_sh:g}, buying-power cap {bp_sh:g})")
                        size = 0.0
                dollar_risk = round(size * per_unit, 2)
                notes.est_fees = round(size * (entry_r * fees.taker_pct + stop * fees.taker_pct + 2 * fees.per_unit), 2)

        approved = not vetoes
        assessment = RiskAssessment(
            ticker=setup.ticker, direction=d, approved=approved, veto_reasons=vetoes, warnings=warns,
            entry=entry_r, stop_loss=stop, tp1=tp1, tp2=tp2, risk_shares=risk_sh, buying_power_shares=bp_sh,
            position_size=max(0.0, size), dollar_risk=dollar_risk if size > 0 else 0.0, reward_risk=rr,
        )
        return assessment, notes
