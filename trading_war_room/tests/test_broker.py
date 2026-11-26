"""PaperBroker: SL / TP1 (scale + breakeven) / TP2 / EOD, LONG and SHORT, accounting identity."""

from datetime import timedelta

import pytest

from tests.conftest import D, et
from warroom.agents.risk_agent import RiskAgent
from warroom.config import ExecutionSection, RiskSection
from warroom.core.clock import SimulatedClock
from warroom.core.event_bus import EventBus
from warroom.core.market_calendar import MarketCalendar
from warroom.execution.paper_broker import Bar, OrderRejected, PaperBroker
from warroom.models.schemas import Direction, ExitReason, Mode, PlanStatus, TradeExecuted, TradePlan
from warroom.storage.database import Database

L, S = Direction.LONG, Direction.SHORT
NO_SLIP = ExecutionSection(slippage_pct=0.0)


def plan(direction=L, size=100, entry=100.0, stop=None, tp1=None, tp2=None, ticker="AAPL"):
    s = direction.sign
    return TradePlan(debate_id="d", ticker=ticker, direction=direction, setup_type="ORB_BREAKOUT",
                     status=PlanStatus.APPROVED, entry=entry, stop_loss=stop or entry - s * 1.0,
                     tp1=tp1 or entry + s * 2.0, tp2=tp2 or entry + s * 3.0, position_size=size, dollar_risk=size * 1.0,
                     created_at=et(11))


def broker(ex=NO_SLIP, cash=100_000.0, now=None, **kw):
    clk = SimulatedClock(now or et(11, 0))
    return PaperBroker(ex, RiskSection(), clk, MarketCalendar(), cash, **kw), clk


def identity(b: PaperBroker, start=100_000.0):
    """equity == start + realized(net) + unrealized"""
    realized = sum(f.realized_pnl - f.commission for f in b.fills_today)
    assert b.equity() == pytest.approx(start + realized + b.unrealized_pnl(), abs=1e-6)


async def test_long_entry_accounting():
    b, _ = broker()
    f = await b.open_from_plan(plan(), bid=99.98, ask=100.0, last=99.99)
    assert f.price == 100.0 and f.qty == 100 and b.cash == pytest.approx(90_000.0)
    b.mark({"AAPL": 101.0})
    assert b.unrealized_pnl() == pytest.approx(100.0) and b.equity() == pytest.approx(100_100.0)
    identity(b)


async def test_long_stop_loss():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    fills = await b.on_bar("AAPL", Bar(99.5, 99.6, 98.8, 99.0))
    assert [f.reason for f in fills] == [ExitReason.STOP_LOSS] and fills[0].price == 99.0
    assert fills[0].realized_pnl == pytest.approx(-100.0) and not b.open_positions
    assert b.cash == pytest.approx(99_900.0)
    identity(b)


async def test_long_stop_gap_fills_at_open():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    fills = await b.on_bar("AAPL", Bar(98.0, 98.2, 97.5, 98.1))  # gapped through 99.0
    assert fills[0].price == 98.0 and fills[0].realized_pnl == pytest.approx(-200.0)


async def test_long_tp1_scale_breakeven_then_tp2():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    f1 = await b.on_bar("AAPL", Bar(101.5, 102.2, 101.4, 102.0))
    assert [f.reason for f in f1] == [ExitReason.TP1] and f1[0].qty == 50 and f1[0].price == 102.0
    pos = b.position_for("AAPL")
    assert pos.qty == 50 and pos.tp1_hit and pos.stop_loss == pos.entry_price == 100.0
    identity(b)
    f2 = await b.on_bar("AAPL", Bar(102.5, 103.4, 102.4, 103.2))
    assert [f.reason for f in f2] == [ExitReason.TP2] and f2[0].price == 103.0
    assert b.cash == pytest.approx(100_000 + 50 * 2 + 50 * 3)
    closed = b.closed_today[-1]
    assert closed.exit_reason is ExitReason.TP2 and closed.realized_pnl == pytest.approx(250.0)
    identity(b)


async def test_long_breakeven_stop_after_tp1():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    await b.on_bar("AAPL", Bar(101.5, 102.1, 101.4, 102.0))
    fills = await b.on_bar("AAPL", Bar(100.5, 100.6, 99.9, 100.1))
    assert fills[0].reason is ExitReason.BREAKEVEN_STOP and fills[0].realized_pnl == pytest.approx(0.0)
    assert b.closed_today[-1].realized_pnl == pytest.approx(100.0)  # only the TP1 half made money


async def test_tp1_and_tp2_in_one_bar():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    fills = await b.on_bar("AAPL", Bar(100.5, 103.5, 100.4, 103.2))
    assert [f.reason for f in fills] == [ExitReason.TP1, ExitReason.TP2] and not b.open_positions


async def test_same_bar_stop_and_target_assumes_stop_first():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    fills = await b.on_bar("AAPL", Bar(100.0, 102.5, 98.9, 101.0))
    assert [f.reason for f in fills] == [ExitReason.STOP_LOSS]


async def test_short_lifecycle_and_margin():
    b, _ = broker()
    await b.open_from_plan(plan(S), bid=100.0, ask=100.02, last=100.01)
    pos = b.position_for("AAPL")
    assert pos.reserved_margin == pytest.approx(100 * 100.0 * 1.5) and b.reserved_margin == pytest.approx(15_000.0)
    assert b.cash == pytest.approx(85_000.0) and b.equity() == pytest.approx(100_000.0, abs=1.5)
    f1 = await b.on_bar("AAPL", Bar(98.5, 98.6, 97.8, 98.0))
    assert f1[0].reason is ExitReason.TP1 and f1[0].price == 98.0 and f1[0].realized_pnl == pytest.approx(100.0)
    assert b.reserved_margin == pytest.approx(7_500.0) and b.position_for("AAPL").stop_loss == 100.0
    identity(b)
    f2 = await b.on_bar("AAPL", Bar(97.5, 97.6, 96.9, 97.1))
    assert f2[0].reason is ExitReason.TP2 and f2[0].price == 97.0
    assert b.reserved_margin == pytest.approx(0.0) and b.cash == pytest.approx(100_000 + 100 + 150)
    identity(b)


async def test_short_stop_and_gap():
    b, _ = broker()
    await b.open_from_plan(plan(S), 100.0, 100.02, 100.0)
    fills = await b.on_bar("AAPL", Bar(101.5, 101.8, 101.4, 101.6))  # gap above stop 101
    assert fills[0].reason is ExitReason.STOP_LOSS and fills[0].price == 101.5
    assert fills[0].realized_pnl == pytest.approx(-150.0) and b.cash == pytest.approx(100_000 - 150)


async def test_eod_flat_long_and_short():
    b, clk = broker()
    await b.open_from_plan(plan(L, ticker="AAPL"), 99.98, 100.0, 100.0)
    await b.open_from_plan(plan(S, ticker="MSFT", entry=50.0), 50.0, 50.02, 50.0)
    clk.set(et(15, 54))
    assert await b.on_bar("AAPL", Bar.flat(100.5)) == []
    clk.set(et(15, 55))
    fa = await b.on_bar("AAPL", Bar.flat(100.5))
    fm = await b.on_bar("MSFT", Bar.flat(49.5))
    assert fa[0].reason is ExitReason.EOD_FLAT and fm[0].reason is ExitReason.EOD_FLAT and not b.open_positions
    identity(b)


async def test_eod_flat_on_early_close_day():
    from datetime import date
    b, clk = broker(now=et(11, 0, date(2025, 11, 28)))
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    clk.set(et(12, 55, date(2025, 11, 28)))
    assert (await b.on_bar("AAPL", Bar.flat(100.2)))[0].reason is ExitReason.EOD_FLAT


async def test_stale_overnight_position_flattened_on_first_tick():
    b, clk = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    clk.set(et(9, 31, D + timedelta(days=1)))
    assert (await b.on_bar("AAPL", Bar.flat(101.0)))[0].reason is ExitReason.EOD_FLAT


async def test_slippage_and_commissions():
    ex = ExecutionSection(slippage_pct=0.001, commission_per_share=0.005, commission_min=1.0)
    b, _ = broker(ex)
    f = await b.open_from_plan(plan(size=100), 99.9, 100.0, 99.95)
    assert f.price == pytest.approx(100.1) and f.commission == pytest.approx(1.0)  # 0.50 < $1 minimum
    s = await b.on_bar("AAPL", Bar(99.5, 99.6, 98.8, 99.0))
    assert s[0].price == pytest.approx(99.0 * 0.999)  # stop is a market order: slips
    identity(b)


async def test_rejects_when_price_ran_through_levels_or_cash_short():
    b, _ = broker()
    with pytest.raises(OrderRejected, match="price moved"):
        await b.open_from_plan(plan(), 98.9, 98.95, 98.9)   # already below the stop
    with pytest.raises(OrderRejected, match="price moved"):
        await b.open_from_plan(plan(), 102.5, 102.6, 102.5)  # already past TP1
    small, _ = broker(cash=50.0)
    with pytest.raises(OrderRejected, match="insufficient"):
        await small.open_from_plan(plan(), 99.98, 100.0, 100.0)
    capped, _ = broker(cash=5_000.0)
    f = await capped.open_from_plan(plan(size=100), 99.98, 100.0, 100.0)
    assert f.qty == 50  # reduced to what cash allows


async def test_no_duplicate_ticker_and_manual_flatten():
    b, _ = broker()
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    with pytest.raises(OrderRejected, match="already holding"):
        await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    fills = await b.flatten_all(prices={"AAPL": 100.4})
    assert fills[0].reason is ExitReason.MANUAL_FLATTEN and fills[0].price == 100.4 and not b.open_positions


async def test_persistence_risk_callbacks_events_and_recovery(tmp_path, cfg):
    db = Database(tmp_path / "b.db")
    await db.open()
    clk = SimulatedClock(et(11, 0))
    cal = MarketCalendar()
    risk = RiskAgent(cfg, cal, clk, Mode.PAPER)
    bus = EventBus()
    sub = bus.subscribe(TradeExecuted)
    b = PaperBroker(NO_SLIP, RiskSection(), clk, cal, 100_000.0, db=db, risk_agent=risk, bus=bus)
    await b.open_from_plan(plan(), 99.98, 100.0, 100.0)
    await b.on_bar("AAPL", Bar(99.5, 99.6, 98.8, 99.0))           # -100 loss
    await b.open_from_plan(plan(S, ticker="MSFT"), 100.0, 100.02, 100.0)
    assert sub.pending() == 3
    assert risk.state.realized_pnl_today == pytest.approx(-100.0) and risk.state.consecutive_losses == 1
    await db.close()

    db2 = Database(tmp_path / "b.db")
    await db2.open()
    rec = await db2.recover(clk.now(), cal, 3, 30)
    b2 = PaperBroker(NO_SLIP, RiskSection(), clk, cal, 100_000.0)
    b2.restore(rec.cash, rec.reserved_margin, rec.open_positions)
    assert rec.risk_state.realized_pnl_today == pytest.approx(-100.0) and rec.risk_state.consecutive_losses == 1
    assert b2.cash == pytest.approx(b.cash) and b2.reserved_margin == pytest.approx(b.reserved_margin)
    assert [p.ticker for p in b2.open_positions] == ["MSFT"]
    assert b2.equity() == pytest.approx(b.equity())
    await db2.close()
