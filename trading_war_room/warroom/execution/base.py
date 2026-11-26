"""Broker interface. ``PaperBroker`` is the only implementation, by design.

A future real-money broker (see docs/LIVE_TRADING_PLAN.md) must implement this
exact interface so the orchestrator, War Room and Risk agent are unchanged, and
must additionally place protective stops ON THE EXCHANGE immediately after each
fill so protection survives an app crash. ``execution.live_trading_enabled`` is
rejected by config validation until that work exists and has passed the plan's
gates.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from warroom.models.schemas import AccountSnapshot, ExitReason, Fill, Position, TradePlan


@runtime_checkable
class Broker(Protocol):
    cash: float
    reserved_margin: float

    @property
    def open_positions(self) -> list[Position]: ...

    def position_for(self, ticker: str) -> Position | None: ...

    def equity(self) -> float: ...

    def unrealized_pnl(self) -> float: ...

    def snapshot(self) -> AccountSnapshot: ...

    def mark(self, prices: dict[str, float]) -> None: ...

    def restore(self, cash: float | None, reserved_margin: float, positions: list[Position]) -> None: ...

    async def open_from_plan(self, plan: TradePlan, bid: float | None, ask: float | None, last: float) -> Fill: ...

    async def on_bar(self, ticker: str, bar, bar_time=None) -> list[Fill]: ...

    async def flatten_all(self, reason: ExitReason = ExitReason.MANUAL_FLATTEN,
                          prices: dict[str, float] | None = None, symbols: set[str] | None = None) -> list[Fill]: ...
