"""Trading War Room - CLI entrypoint. PAPER / ADVISORY ONLY: no live orders, ever.

  python main.py --demo                    mock data, TUI, zero API keys
  python main.py --demo --voice            + J.A.R.V.I.S. voice briefings
  python main.py --paper                   real data providers (config.yaml + .env), paper broker
  python main.py --replay 2025-01-15 --tickers AAPL,NVDA,TSLA
  python main.py --smoke-test              3 headless cycles; exit 0 on a full debate + briefing
  python main.py --check-providers         live contract checks of configured providers
Add --headless to --demo/--paper to stream events to the console instead of the TUI.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import date, datetime, time
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2
SMOKE_DAY = date(2025, 1, 15)  # fixed, known session: the smoke test is reproducible on any machine, any hour


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="main.py", description="Multi-agent paper-trading War Room with J.A.R.V.I.S.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="mock market/news, live debates, TUI (no keys needed)")
    mode.add_argument("--paper", action="store_true", help="paper trading on configured real data providers")
    mode.add_argument("--replay", metavar="YYYY-MM-DD", help="replay a historical session and print a report")
    mode.add_argument("--smoke-test", action="store_true", help="3 headless cycles; exit 0 if a debate + briefing succeed")
    mode.add_argument("--check-providers", action="store_true", help="live checks of configured providers")
    p.add_argument("--tickers", help="comma-separated tickers (replay; overrides config)")
    p.add_argument("--replay-source", choices=["auto", "csv", "alpaca", "yfinance", "mock"],
                   help="override replay.source (mock = synthetic dry run, no data needed)")
    p.add_argument("--voice", action="store_true", help="speak J.A.R.V.I.S. briefings (edge-tts/pyttsx3 if installed)")
    p.add_argument("--headless", action="store_true", help="--demo/--paper without the TUI (console event stream)")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--env-file", default=".env", help="path to .env")
    p.add_argument("--log-file", default="data/warroom.log")
    args = p.parse_args(argv)
    if args.replay:
        try:
            args.replay_date = date.fromisoformat(args.replay)
        except ValueError:
            p.error("--replay expects a date as YYYY-MM-DD")
    if args.voice and not (args.demo or args.paper):
        p.error("--voice applies to --demo and --paper")
    return args


def setup_logging(level: str, log_file: str, console: bool) -> None:
    handlers: list[logging.Handler] = []
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except OSError:
        pass
    if console:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.WARNING)
        handlers.append(h)
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load(args: argparse.Namespace, mode):
    from dotenv import load_dotenv

    from warroom.config import Secrets, load_config, validate_for_mode

    if Path(args.env_file).exists():
        load_dotenv(args.env_file, override=False)
    cfg = load_config(args.config)
    secrets = Secrets.from_env()
    warnings = validate_for_mode(cfg, secrets, mode)
    return cfg, secrets, warnings


def install_signal_handlers(loop: asyncio.AbstractEventLoop, callback) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, callback)
        except (NotImplementedError, RuntimeError):  # Windows / non-main thread
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(callback))


# ---------------------------------------------------------------------------- modes


async def run_live(args, cfg, secrets, mode) -> int:
    from warroom.orchestrator import Orchestrator

    orch = await Orchestrator.build(cfg, secrets, mode, voice=args.voice)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    if args.headless:
        install_signal_handlers(loop, stop.set)
        printer = asyncio.create_task(_console_stream(orch))
        await orch.start()
        try:
            await orch.run_forever(stop)
        finally:
            printer.cancel()
            await orch.shutdown()
        return EXIT_OK

    from warroom.dashboard.tui_app import WarRoomApp

    app = WarRoomApp(orch, stop)
    await orch.start()
    try:
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.ensure_future(app.action_quit()))
    except (NotImplementedError, RuntimeError):
        pass
    try:
        await app.run_async()
    finally:
        await app.shutdown()  # idempotent; covers abnormal exits
    return EXIT_OK


async def _console_stream(orch) -> None:
    from warroom.core.clock import to_et
    from warroom.models.schemas import JarvisBriefing, SystemAlert, TradeExecuted, WarRoomMessage

    sub = orch.bus.subscribe(WarRoomMessage, JarvisBriefing, TradeExecuted, SystemAlert, name="console")
    async for e in sub:
        ts = f"{to_et(e.timestamp):%H:%M}"
        if isinstance(e, WarRoomMessage):
            m = e.message
            print(f"{ts} R{m.round_number} [{m.sender.value}] {e.ticker} ({m.stance.value}) {m.message}", flush=True)
        elif isinstance(e, JarvisBriefing):
            print(f"{ts} J.A.R.V.I.S.: {e.text}", flush=True)
        elif isinstance(e, TradeExecuted):
            f = e.fill
            print(f"{ts} [PAPER {'ENTRY' if f.is_entry else 'EXIT ' + (f.reason.value if f.reason else '')}] "
                  f"{f.ticker} {f.qty} @ {f.price:.2f}", flush=True)
        else:
            print(f"{ts} [{e.level.value}] {e.source}: {e.message}", flush=True)


async def run_replay(args, cfg, secrets) -> int:
    from warroom.backtest.replay_engine import ReplayEngine

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    if args.replay_source:
        from warroom.config import parse_config

        raw = cfg.model_dump(mode="json")
        raw["replay"]["source"] = args.replay_source
        cfg = parse_config(raw, source="--replay-source")
    engine = ReplayEngine(cfg, secrets, args.replay_date, tickers)
    print(f"Replaying {args.replay_date} through the War Room pipeline...", file=sys.stderr, flush=True)
    report = await engine.run()
    print(report.format())
    return EXIT_OK


async def run_smoke(cfg, secrets) -> int:
    """3 simulated cycles, headless, no audio, in-memory DB. Exit 0 iff a full 3-round
    War Room debate AND a J.A.R.V.I.S. briefing were produced."""
    from datetime import timedelta

    from warroom.core.clock import SimulatedClock, et_to_utc
    from warroom.models.schemas import JarvisBriefing, Mode
    from warroom.orchestrator import Orchestrator

    clock = SimulatedClock(et_to_utc(datetime.combine(SMOKE_DAY, time(10, 30))))
    orch = await Orchestrator.build(cfg, secrets, Mode.SMOKE, clock=clock, db_path=":memory:")
    briefings = orch.bus.subscribe(JarvisBriefing, name="smoke")
    ok = False
    try:
        await orch.start()
        debates = []
        for i in range(3):
            res = await orch.run_cycle()
            debates.extend(res.debates)
            print(f"cycle {i + 1}: {to_et_str(res.at)} ET, {len(res.debates)} debate(s), {res.fills} fill(s)")
            clock.advance(timedelta(minutes=1))
        if not debates:
            print("no organic setup in 3 cycles; convening an on-demand War Room")
            t = await orch.analyze(cfg.app.tickers[0])
            if not isinstance(t, str):
                debates.append(t)
        status = await orch.handle_command("status report")
        got_briefing = False
        while (e := briefings.get_nowait()) is not None:
            if e.kind == "briefing" and e.text:
                got_briefing = True
        full = [t for t in debates if {m.round_number for m in t.messages} == {1, 2, 3} and t.briefing]
        for t in full[:3]:
            print(f"  {t.ticker} {t.direction.value}: {t.plan.status.value} ({len(t.messages)} messages)")
            print(f"    J.A.R.V.I.S.: {t.briefing}")
        print(f"  status: {status}")
        ok = bool(full) and got_briefing and status.startswith(cfg.app.user_title)
    finally:
        await orch.shutdown()
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return EXIT_OK if ok else EXIT_FAIL


def to_et_str(dt) -> str:
    from warroom.core.clock import to_et

    return f"{to_et(dt):%Y-%m-%d %H:%M}"


async def run_check(cfg, secrets) -> int:
    from warroom.core.clock import SystemClock
    from warroom.providers import factory
    from warroom.providers.healthcheck import check_providers

    results = await check_providers(cfg, secrets, SystemClock(), factory.build_calendar(cfg))
    for r in results:
        print(r.line())
    return EXIT_OK if all(r.ok for r in results) else EXIT_FAIL


# ---------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from warroom.config import ConfigError
    from warroom.models.schemas import Mode

    mode = (Mode.DEMO if args.demo else Mode.PAPER if (args.paper or args.check_providers)
            else Mode.REPLAY if args.replay else Mode.SMOKE)
    try:
        cfg, secrets, warnings = load(args, mode)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    tui = (args.demo or args.paper) and not args.headless
    setup_logging(cfg.app.log_level, args.log_file, console=not tui)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    try:
        if args.smoke_test:
            return asyncio.run(run_smoke(cfg, secrets))
        if args.replay:
            return asyncio.run(run_replay(args, cfg, secrets))
        if args.check_providers:
            return asyncio.run(run_check(cfg, secrets))
        return asyncio.run(run_live(args, cfg, secrets, mode))
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
