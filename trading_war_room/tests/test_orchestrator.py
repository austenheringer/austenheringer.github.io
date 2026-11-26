"""Orchestrator wiring, commands, recovery, standby, replay, TUI and CLI."""

import asyncio
from datetime import date, datetime, time, timedelta

import pytest

from tests.conftest import D, et
from warroom.config import Secrets, parse_config
from warroom.core.clock import SimulatedClock, et_to_utc
from warroom.models.schemas import JarvisBriefing, MarketUpdate, Mode, PlanStatus, WarRoomMessage
from warroom.orchestrator import Orchestrator

SECRETS = Secrets.from_env({})


async def _orch(cfg, mode=Mode.SMOKE, now=None, db_path=":memory:", start=True, **kw):
    clk = SimulatedClock(now or et(10, 30))
    o = await Orchestrator.build(cfg, SECRETS, mode, clock=clk, db_path=db_path, **kw)
    if start:
        await o.start()
    return o, clk


async def test_cycles_publish_and_debate(cfg):
    o, clk = await _orch(cfg, start=False)
    sub = o.bus.subscribe(name="t", maxsize=10_000)
    await o.start()
    debates = []
    for _ in range(20):
        debates += (await o.run_cycle()).debates
        clk.advance(timedelta(minutes=5))
    events = []
    while (e := sub.get_nowait()) is not None:
        events.append(e)
    assert any(isinstance(e, MarketUpdate) for e in events)
    assert debates and any(isinstance(e, WarRoomMessage) for e in events)
    assert any(isinstance(e, JarvisBriefing) and e.kind == "greeting" for e in events)
    assert await o.db.count("equity_snapshots") >= 20
    await o.shutdown()


async def test_jarvis_commands_through_orchestrator(cfg):
    o, clk = await _orch(cfg, now=et(11, 0))
    await o.run_cycle()
    assert (await o.handle_command("status")).startswith("Sir, SMOKE mode")
    reply = await o.handle_command("analyze NVDA")
    assert reply.startswith("Sir,") and o.war_room.transcripts.get("NVDA")
    why = await o.handle_command("why NVDA")
    assert "NVDA" in why
    tr = await o.handle_command("transcript NVDA")
    assert "War Room" in tr and "R3 [RISK]" in tr
    await o.handle_command("pause")
    assert o.risk.state.entries_paused
    await o.handle_command("resume")
    assert not o.risk.state.entries_paused
    flat = await o.handle_command("flatten")
    assert "already flat" in flat or "Closed at market" in flat
    assert not o.broker.open_positions
    await o.shutdown()


async def test_flatten_closes_positions(cfg):
    o, clk = await _orch(cfg, now=et(11, 0))
    from tests.test_broker import plan
    await o.broker.open_from_plan(plan(ticker="AAPL", entry=200.0), 199.9, 200.0, 200.0)
    reply = await o.handle_command("flatten")
    assert "AAPL" in reply and not o.broker.open_positions
    await o.shutdown()


async def test_restart_cannot_bypass_circuit_breakers(cfg, tmp_path):
    path = str(tmp_path / "o.db")
    o, clk = await _orch(cfg, db_path=path, now=et(11, 0))
    from tests.test_broker import plan
    from warroom.execution.paper_broker import Bar
    for i, t in enumerate(("AAPL", "MSFT", "AMD")):
        await o.broker.open_from_plan(plan(ticker=t, entry=100.0), 99.9, 100.0, 100.0)
        clk.advance(60)
        await o.broker.on_bar(t, Bar(99.0, 99.1, 98.5, 98.6))  # stopped out
    o.pause()
    await asyncio.sleep(0)
    assert o.risk.state.consecutive_losses == 3 and o.risk.state.cooldown_until
    await o.shutdown()

    o2, _ = await _orch(cfg, db_path=path, now=clk.now())
    st = o2.risk.state
    assert st.consecutive_losses == 3 and st.cooldown_until is not None and st.entries_paused
    assert st.realized_pnl_today < 0 and o2.broker.cash == pytest.approx(o.broker.cash)
    assert "COOLDOWN" in o2.risk.circuit_status()
    await o2.shutdown()


async def test_paper_mode_standby_when_closed(cfg):
    from warroom.core.market_calendar import MarketCalendar
    from warroom.providers.market_data import MockDataProvider
    from warroom.providers.news_data import CalendarOnlyNewsProvider, StaticMacroCalendar

    sat = et_to_utc(datetime(2025, 1, 18, 11, 0))
    clk = SimulatedClock(sat)
    cal = MarketCalendar()
    o = await Orchestrator.build(cfg, SECRETS, Mode.PAPER, clock=clk, db_path=":memory:",
                                 market=MockDataProvider(clk, cal),
                                 news_provider=CalendarOnlyNewsProvider(clk, StaticMacroCalendar(cal, None)))
    await o.start()
    sub = o.bus.subscribe(JarvisBriefing)
    r1 = await o.run_cycle()
    r2 = await o.run_cycle()
    msgs = []
    while (e := sub.get_nowait()) is not None:
        msgs.append(e.text)
    assert r1.standby and r2.standby and not r1.debates
    assert sum("Standing by until Tuesday 09:30 ET" in m for m in msgs) == 1  # announced once (MLK Monday)
    await o.shutdown()


async def test_run_forever_stops_and_shutdown_idempotent(cfg):
    o, clk = await _orch(cfg)
    stop = asyncio.Event()
    task = asyncio.create_task(o.run_forever(stop, interval=60))
    await asyncio.sleep(0.3)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert o.cycles >= 1
    await o.shutdown()
    await o.shutdown()


async def test_cycle_errors_are_contained(cfg):
    o, clk = await _orch(cfg)
    boom = {"n": 0}

    async def bad_cycle():
        boom["n"] += 1
        raise RuntimeError("provider exploded")

    o.run_cycle = bad_cycle
    stop = asyncio.Event()
    task = asyncio.create_task(o.run_forever(stop, interval=60))
    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert boom["n"] >= 2  # kept going after the first failure
    await o.shutdown()


async def test_replay_engine_mock_slice(cfg):
    from warroom.backtest.replay_engine import ReplayEngine

    raw = cfg.model_dump(mode="json")
    raw["replay"]["source"] = "mock"
    eng = ReplayEngine(parse_config(raw), SECRETS, D, ["NVDA", "TSLA"])
    report = await eng.run(max_minutes=90)
    assert report.source.startswith("mock") and report.debates >= 1
    assert report.approved + report.watchlist + report.vetoed == report.debates
    text = report.format()
    assert "Total trades" in text and "SYNTHETIC" in text


async def test_replay_rejects_missing_data_and_holidays(cfg, tmp_path):
    from warroom.backtest.replay_engine import ReplayDataUnavailable, ReplayEngine

    raw = cfg.model_dump(mode="json")
    raw["replay"]["data_dir"] = str(tmp_path)
    with pytest.raises(ReplayDataUnavailable, match="No historical data"):
        await ReplayEngine(parse_config(raw), SECRETS, date(2024, 3, 5), ["AAPL"]).run()
    with pytest.raises(ReplayDataUnavailable, match="Christmas"):
        await ReplayEngine(parse_config(raw), SECRETS, date(2025, 12, 25), ["AAPL"]).run()


# ---------------------------------------------------------------- TUI (headless pilot)

async def test_tui_renders_and_accepts_commands(cfg):
    from textual.widgets import DataTable, RichLog

    from warroom.dashboard.tui_app import WarRoomApp

    o, clk = await _orch(cfg, now=et(11, 0))
    app = WarRoomApp(o, run_loop=False)
    async with app.run_test(size=(160, 48)) as pilot:
        for _ in range(3):
            await o.run_cycle()
            clk.advance(timedelta(minutes=5))
        await o.handle_command("analyze NVDA")
        await pilot.pause(0.2)
        assert app.query_one("#watch", DataTable).row_count == len(cfg.app.tickers)
        assert len(app.query_one("#warroom", RichLog).lines) > 0
        await pilot.click("#cmd")
        await pilot.press(*"status", "enter")
        await pilot.pause(0.3)
        jarvis_text = "\n".join(str(line.text) for line in app.query_one("#jarvis", RichLog).lines)
        assert "SMOKE mode" in jarvis_text and "> status" in jarvis_text
        header = str(app.query_one("#header").render())
        assert "SMOKE" in header and "Equity" in header and "Breakers" in header
        await app.action_quit()
    assert o._closed


# ---------------------------------------------------------------- CLI

def test_cli_argument_validation():
    import main

    with pytest.raises(SystemExit):
        main.parse_args([])
    with pytest.raises(SystemExit):
        main.parse_args(["--demo", "--paper"])
    with pytest.raises(SystemExit):
        main.parse_args(["--replay", "15-01-2025"])
    with pytest.raises(SystemExit):
        main.parse_args(["--smoke-test", "--voice"])
    a = main.parse_args(["--replay", "2025-01-15", "--tickers", "AAPL,NVDA"])
    assert a.replay_date == date(2025, 1, 15)


def test_cli_config_error_exit_code(tmp_path, capsys):
    import main

    bad = tmp_path / "c.yaml"
    bad.write_text("risk:\n  max_risk_pct: 0.9\n")
    assert main.main(["--smoke-test", "--config", str(bad), "--log-file", str(tmp_path / "l.log")]) == 2
    assert "max_risk_pct" in capsys.readouterr().err


def test_cli_paper_without_real_provider_fails_fast(tmp_path, capsys):
    import main

    assert main.main(["--paper", "--headless", "--env-file", str(tmp_path / "none"), "--log-file", str(tmp_path / "l.log")]) == 2
    assert "real market data provider" in capsys.readouterr().err


def test_cli_smoke_test_passes(tmp_path, capsys):
    import main

    assert main.main(["--smoke-test", "--log-file", str(tmp_path / "l.log")]) == 0
    out = capsys.readouterr().out
    assert "SMOKE TEST PASSED" in out and "J.A.R.V.I.S.:" in out
