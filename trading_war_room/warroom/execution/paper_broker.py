"""Paper broker - simulated fills only. There is no live order routing anywhere
in this project, by design.

Accounting
----------
* LONG entry: cash -= qty*fill + commission. The position's market value is
  qty*last.
* SHORT entry: margin of ``short_margin_requirement`` x notional moves from
  cash to ``reserved_margin``; commission from cash. Covering releases the
  margin pro rata and settles P&L into cash.
* Equity = cash + reserved_margin + sum(long qty*last) + sum(short unrealized).
  This identity is asserted in tests after every kind of exit.

Fills
-----
* Market orders (entries, stops, EOD/manual flattens) pay ``slippage_pct``
  adversely. A stop gapped through fills at the bar's open, not the stop.
* Targets are limit orders: filled at the target or better (a gap through
  fills at the open), no slippage.
* Same-bar ambiguity: if one 1m bar touches both the stop and a target, the
  STOP is assumed to fill first. Intrabar order is unknowable from OHLC; the
  conservative assumption keeps paper results from flattering the strategy.
* TP1 sells ``tp1_scale_out_pct`` of the initial size (min 1 share) and
  moves the stop to breakeven (entry price). TP2 closes the rest.
* At/after the session's EOD flat time (15:55 ET; 12:55 on early-close days)
  everything is closed at the bar close. Positions left over from a previous
  session (app was down) are flattened on the first tick.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

from warroom.config import ExecutionSection, RiskSection
from warroom.core.clock import Clock, to_et
from warroom.core.event_bus import EventBus
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import (
    AccountSnapshot,
    Direction,
    ExitReason,
    Fill,
    Position,
    PositionStatus,
    TradeExecuted,
    TradePlan,
    validate_levels,
)

log = logging.getLogger(__name__)


class OrderRejected(Exception):
    pass


@dataclass(frozen=True)
class Bar:
    """Minimal OHLC for tick processing (a completed 1m bar)."""

    open: float
    high: float
    low: float
    close: float

    @classmethod
    def flat(cls, price: float) -> "Bar":
        return cls(price, price, price, price)


class PaperBroker:
    def __init__(self, execution: ExecutionSection, risk_cfg: RiskSection, clock: Clock, calendar: MarketCalendar,
                 starting_cash: float, *, db=None, risk_agent=None, bus: EventBus | None = None) -> None:
        self.ex = execution
        self.risk_cfg = risk_cfg
        self.clock = clock
        self.calendar = calendar
        self.cash = float(starting_cash)
        self.reserved_margin = 0.0
        self.positions: dict[str, Position] = {}
        self.closed_today: list[Position] = []
        self.fills_today: list[Fill] = []
        self.db = db
        self.risk_agent = risk_agent
        self.bus = bus
        self._day: date = clock.now_et().date()
        self.last_prices: dict[str, float] = {}

    # ------------------------------------------------------------------ state

    def restore(self, cash: float | None, reserved_margin: float, positions: list[Position]) -> None:
        if cash is not None:
            self.cash = cash
        self.reserved_margin = reserved_margin
        self.positions = {p.id: p for p in positions if p.status is PositionStatus.OPEN and p.qty > 0}
        for p in self.positions.values():
            if p.last_price:
                self.last_prices[p.ticker] = p.last_price

    def _roll_day(self) -> None:
        today = self.clock.now_et().date()
        if today != self._day:
            self._day = today
            self.closed_today = []
            self.fills_today = []

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.qty > 0]

    def position_for(self, ticker: str) -> Position | None:
        return next((p for p in self.open_positions if p.ticker == ticker), None)

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl() for p in self.open_positions)

    def long_market_value(self) -> float:
        return sum((p.last_price or p.entry_price) * p.qty for p in self.open_positions if p.direction is Direction.LONG)

    def equity(self) -> float:
        short_unreal = sum(p.unrealized_pnl() for p in self.open_positions if p.direction is Direction.SHORT)
        return self.cash + self.reserved_margin + self.long_market_value() + short_unreal

    def realized_pnl_today(self) -> float:
        self._roll_day()
        return round(sum(f.realized_pnl - f.commission for f in self.fills_today), 2)

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=self.clock.now(), equity=round(self.equity(), 2), cash=round(self.cash, 2),
            reserved_margin=round(self.reserved_margin, 2), realized_pnl_today=self.realized_pnl_today(),
            unrealized_pnl=round(self.unrealized_pnl(), 2), open_positions=len(self.open_positions),
        )

    # ------------------------------------------------------------------ pricing

    def commission(self, qty: int) -> float:
        c = self.ex.commission_per_share * qty
        return round(max(self.ex.commission_min, c) if c > 0 or self.ex.commission_min > 0 else 0.0, 4)

    def _market_fill(self, ref: float, buying: bool) -> float:
        slip = self.ex.slippage_pct
        return round(ref * (1 + slip) if buying else ref * (1 - slip), 4)

    # ------------------------------------------------------------------ entries

    async def open_from_plan(self, plan: TradePlan, bid: float | None, ask: float | None, last: float) -> Fill:
        """Enter at market. Re-validates levels against the ACTUAL fill: if price has run
        through the stop or past TP1 since the debate, the order is rejected rather than
        entered with broken geometry."""
        self._roll_day()
        if plan.position_size < 1 or None in (plan.stop_loss, plan.tp1, plan.tp2):
            raise OrderRejected("plan is not executable")
        if self.position_for(plan.ticker):
            raise OrderRejected(f"already holding {plan.ticker}")
        d = plan.direction
        buying = d is Direction.LONG
        ref = (ask if buying else bid) or last
        fill = self._market_fill(ref, buying)
        bad = validate_levels(d, fill, plan.stop_loss, plan.tp1, plan.tp2)  # type: ignore[arg-type]
        if bad:
            raise OrderRejected(f"price moved since the debate ({fill:.2f}): {bad}")
        qty = plan.position_size
        if buying:
            affordable = math.floor(max(0.0, self.cash - self.commission(qty)) / fill)
            qty = min(qty, affordable)
        else:
            affordable = math.floor(max(0.0, self.cash - self.commission(qty)) / (fill * self.ex.short_margin_requirement))
            qty = min(qty, affordable)
        if qty < 1:
            raise OrderRejected("insufficient cash for even one share")
        comm = self.commission(qty)
        margin = 0.0
        if buying:
            self.cash -= qty * fill + comm
        else:
            margin = qty * fill * self.ex.short_margin_requirement
            self.cash -= margin + comm
            self.reserved_margin += margin
        now = self.clock.now()
        pos = Position(plan_id=plan.id, ticker=plan.ticker, direction=d, entry_price=fill, stop_loss=plan.stop_loss,
                       tp1=plan.tp1, tp2=plan.tp2, initial_qty=qty, qty=qty, reserved_margin=round(margin, 4),
                       commissions=comm, opened_at=now, last_price=last)
        self.positions[pos.id] = pos
        self.last_prices[pos.ticker] = last
        f = Fill(position_id=pos.id, ticker=pos.ticker, direction=d, is_entry=True, qty=qty, price=fill,
                 commission=comm, realized_pnl=0.0, timestamp=now)
        await self._record(f, pos, closed=False)
        return f

    # ------------------------------------------------------------------ exits

    async def _exit(self, pos: Position, qty: int, price: float, reason: ExitReason) -> Fill:
        qty = min(qty, pos.qty)
        s = pos.direction.sign
        pnl = round((price - pos.entry_price) * qty * s, 4)
        comm = self.commission(qty)
        if pos.direction is Direction.LONG:
            self.cash += qty * price - comm
        else:
            release = pos.reserved_margin * qty / pos.qty
            pos.reserved_margin = round(pos.reserved_margin - release, 4)
            self.reserved_margin = max(0.0, self.reserved_margin - release)
            self.cash += release + pnl - comm
        pos.qty -= qty
        pos.realized_pnl = round(pos.realized_pnl + pnl, 4)
        pos.commissions = round(pos.commissions + comm, 4)
        pos.last_price = price
        closed = pos.qty == 0
        if closed:
            pos.status = PositionStatus.CLOSED
            pos.closed_at = self.clock.now()
            pos.exit_reason = reason
            pos.reserved_margin = 0.0
            self.positions.pop(pos.id, None)
            self.closed_today.append(pos)
        f = Fill(position_id=pos.id, ticker=pos.ticker, direction=pos.direction, is_entry=False, qty=qty, price=price,
                 commission=comm, realized_pnl=pnl, reason=reason, timestamp=self.clock.now())
        await self._record(f, pos, closed=closed)
        return f

    async def _record(self, f: Fill, pos: Position, closed: bool) -> None:
        self.fills_today.append(f)
        if self.db is not None:
            await self.db.record_fill(f, pos, round(self.cash, 4), round(self.reserved_margin, 4))
        if self.risk_agent is not None:
            if not f.is_entry or f.commission:
                self.risk_agent.on_fill(f.realized_pnl, f.commission)
            if closed:
                self.risk_agent.on_position_closed(pos)
        if self.bus is not None:
            self.bus.publish(TradeExecuted(timestamp=f.timestamp, fill=f, position=pos.model_copy()))

    async def on_bar(self, ticker: str, bar: Bar) -> list[Fill]:
        """Process one completed 1m bar for ``ticker``: stops, targets, EOD flat."""
        self._roll_day()
        self.last_prices[ticker] = bar.close
        fills: list[Fill] = []
        now = self.clock.now()
        sess = self.calendar.session_for(now)
        for pos in [p for p in self.open_positions if p.ticker == ticker]:
            pos.last_price = bar.close
            stale = to_et(pos.opened_at).date() < to_et(now).date()
            if stale or (sess is not None and sess.past_eod_flat(now)):
                fills.append(await self._exit(pos, pos.qty, self._market_fill(bar.close, pos.direction is Direction.SHORT),
                                              ExitReason.EOD_FLAT))
                continue
            if sess is None:
                continue  # market closed: no level checks on stray prints
            fills.extend(await self._check_levels(pos, bar))
        return fills

    async def _check_levels(self, pos: Position, bar: Bar) -> list[Fill]:
        out: list[Fill] = []
        long = pos.direction is Direction.LONG
        stop_hit = bar.low <= pos.stop_loss if long else bar.high >= pos.stop_loss
        tp1_hit = (bar.high >= pos.tp1) if long else (bar.low <= pos.tp1)
        tp2_hit = (bar.high >= pos.tp2) if long else (bar.low <= pos.tp2)

        if stop_hit:  # conservative: stop wins any same-bar tie
            gap_through = bar.open < pos.stop_loss if long else bar.open > pos.stop_loss
            ref = bar.open if gap_through else pos.stop_loss
            reason = ExitReason.BREAKEVEN_STOP if pos.tp1_hit else ExitReason.STOP_LOSS
            out.append(await self._exit(pos, pos.qty, self._market_fill(ref, buying=not long), reason))
            return out
        if not pos.tp1_hit and tp1_hit:
            px = max(pos.tp1, bar.open) if long else min(pos.tp1, bar.open)
            scale = max(1, math.floor(pos.initial_qty * self.risk_cfg.tp1_scale_out_pct))
            if scale >= pos.qty:
                out.append(await self._exit(pos, pos.qty, px, ExitReason.TP1))
                return out
            pos.tp1_hit = True
            pos.stop_loss = pos.entry_price  # breakeven
            out.append(await self._exit(pos, scale, px, ExitReason.TP1))
        if pos.tp1_hit and pos.qty > 0 and tp2_hit:
            px = max(pos.tp2, bar.open) if long else min(pos.tp2, bar.open)
            out.append(await self._exit(pos, pos.qty, px, ExitReason.TP2))
        return out

    async def flatten_all(self, reason: ExitReason = ExitReason.MANUAL_FLATTEN,
                          prices: dict[str, float] | None = None) -> list[Fill]:
        fills = []
        for pos in list(self.open_positions):
            last = (prices or {}).get(pos.ticker) or self.last_prices.get(pos.ticker) or pos.last_price or pos.entry_price
            fills.append(await self._exit(pos, pos.qty, self._market_fill(last, pos.direction is Direction.SHORT), reason))
        return fills

    def mark(self, prices: dict[str, float]) -> None:
        for p in self.open_positions:
            if p.ticker in prices:
                p.last_price = prices[p.ticker]
                self.last_prices[p.ticker] = prices[p.ticker]
