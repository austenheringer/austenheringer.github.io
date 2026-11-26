import asyncio
import os
from datetime import timedelta

import pytest

from tests.conftest import D, et
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import (
    AccountSnapshot, AgentMessage, AgentRole, Fill, Position, RiskState, TradePlan, WarRoomTranscript,
)
from warroom.storage.database import Database, derive_risk_state


def mkpos(pid, opened, closed, pnl, status="CLOSED"):
    return Position(id=pid, ticker="AAPL", direction="LONG", entry_price=100, stop_loss=99, tp1=102, tp2=103,
                    initial_qty=10, qty=0 if status == "CLOSED" else 10, opened_at=opened, closed_at=closed,
                    realized_pnl=pnl, status=status, exit_reason="STOP_LOSS" if status == "CLOSED" else None)


@pytest.fixture
async def db(tmp_path):
    d = Database(tmp_path / "t.db")
    await d.open()
    yield d
    await d.close()


async def test_wal_mode(db):
    assert await db._run(lambda c: c.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"


async def _seed(db):
    y = D - timedelta(days=1)
    seq = [("y1", et(10, 0, y), et(11, 0, y), 50.0), ("a", et(9, 40), et(9, 50), 30.0),
           ("b", et(9, 51), et(9, 55), -10.0), ("c", et(9, 56), et(9, 58), -10.0), ("d", et(9, 58), et(9, 59), -10.0)]
    cash = 100_000.0
    for pid, o, c, pnl in seq:
        cash += pnl
        await db.record_fill(Fill(position_id=pid, ticker="AAPL", direction="LONG", is_entry=False, qty=10, price=100,
                                  commission=0.5, realized_pnl=pnl, reason="STOP_LOSS", timestamp=c),
                             mkpos(pid, o, c, pnl), cash, 0.0)
    await db.upsert_position(mkpos("open1", et(9, 59), None, 0.0, status="OPEN"))
    await db.save_risk_state(RiskState(trading_date=D, entries_paused=True))
    await db.save_equity_snapshot(AccountSnapshot(timestamp=et(16, 0, y), equity=100_050, cash=100_050,
                                                  reserved_margin=0, realized_pnl_today=50, unrealized_pnl=0, open_positions=0))
    return cash


async def test_startup_recovery_survives_restart(tmp_path, cal):
    path = tmp_path / "r.db"
    db = Database(path)
    await db.open()
    cash = await _seed(db)
    await db.close()
    assert not os.path.exists(str(path) + "-wal") or os.path.getsize(str(path) + "-wal") == 0  # checkpointed

    db = Database(path)
    await db.open()
    rec = await db.recover(et(10, 0), cal, consecutive_loss_limit=3, cooldown_minutes=30)
    rs = rec.risk_state
    assert rs.realized_pnl_today == pytest.approx(30 - 30 - 4 * 0.5)
    assert rs.consecutive_losses == 3 and rs.cooldown_until == et(10, 29)
    assert rs.entries_paused                                   # restored flag
    assert len(rs.day_trade_dates) == 5                        # yesterday's + today's 4
    assert [p.id for p in rec.open_positions] == ["open1"]
    assert rec.cash == pytest.approx(cash) and rec.start_of_day_equity == 100_050
    await db.close()


def test_streak_breaks_on_win_and_daily_halt_does_not_carry(cal):
    closed = [mkpos("a", et(9, 40), et(9, 50), -5), mkpos("b", et(9, 51), et(9, 55), 5), mkpos("c", et(9, 56), et(9, 58), -5)]
    yesterday = RiskState(trading_date=D - timedelta(days=1), halted_for_day=True, entries_paused=True)
    st = derive_risk_state(D, closed, [], yesterday, cal, 3, 30)
    assert st.consecutive_losses == 1 and st.cooldown_until is None
    assert not st.halted_for_day and st.entries_paused


def test_overnight_position_is_not_a_day_trade(cal):
    p = mkpos("o", et(15, 0, D - timedelta(days=1)), et(10, 0), 5)
    assert derive_risk_state(D, [p], [], None, cal, 3, 30).day_trade_dates == []


async def test_record_fill_is_atomic(db):
    f = Fill(id="dup", position_id="a", ticker="AAPL", direction="LONG", is_entry=False, qty=1, price=1, commission=0,
             timestamp=et(10, 5))
    await db.record_fill(f, mkpos("a", et(9, 40), et(9, 50), 30.0), 1.0, 0.0)
    with pytest.raises(Exception):
        await db.record_fill(f, mkpos("a", et(9, 40), et(9, 50), 30.0), 999.0, 0.0)
    assert (await db.load_account_state())[0] == 1.0


async def test_transcript_roundtrip_and_veto_log(db):
    did = "deb1"
    msgs = [AgentMessage(debate_id=did, sender=s, round_number=r, stance="NEUTRAL", message="m", timestamp=et(10))
            for r, s in ((1, AgentRole.TECHNICAL), (2, AgentRole.NEWS), (3, AgentRole.RISK))]
    plan = TradePlan(debate_id=did, ticker="AAPL", direction="LONG", setup_type="ORB_BREAKOUT", status="VETOED",
                     veto_reasons=["R:R 1.2 < 2.0"], created_at=et(10))
    await db.save_transcript(WarRoomTranscript(debate_id=did, ticker="AAPL", direction="LONG", started_at=et(10),
                                               finished_at=et(10, 1), messages=msgs, plan=plan, briefing="b"))
    t = await db.latest_transcript("aapl")
    assert t.plan.veto_reasons == ["R:R 1.2 < 2.0"] and await db.count("risk_vetoes") == 1
    assert await db.count("agent_messages") == 3


async def test_concurrent_writes_serialize(db):
    await asyncio.gather(*(db.save_message(AgentMessage(debate_id="x", sender="RISK", round_number=1, stance="VETO",
                                                        message=f"m{i}", timestamp=et(10))) for i in range(200)))
    assert await db.count("agent_messages") == 200


async def test_llm_usage_persisted(db):
    await db.set_llm_usage(D, 7)
    assert await db.get_llm_usage(D) == 7
