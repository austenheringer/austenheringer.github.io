# CLAUDE.md — Trading War Room

Multi-agent day-trading **analysis, risk management and PAPER trading** system with
an Agent War Room (Technical / News / Risk debate in 3 rounds) and a J.A.R.V.I.S.-style
console. **Advisory/paper only. Never add live real-money order execution.**

This project lives in `trading_war_room/` inside a GitHub Pages repo; keep all of it
inside this directory.

## Commands (run from `trading_war_room/`)

```bash
pip install -e ".[dev]"            # core + pytest; add [all] for every optional provider
python main.py --demo              # mock data, TUI, zero API keys
python main.py --demo --voice      # + British voice briefings (edge-tts/pygame/pyttsx3 if present)
python main.py --paper             # real data providers from config.yaml + .env, paper broker
python main.py --replay 2025-01-15 --tickers AAPL,NVDA,TSLA
python main.py --smoke-test        # 3 headless cycles, exit 0 on success
pytest -q
```

## Layout

```
main.py                      CLI entrypoint
config.yaml                  non-secret settings (validated by warroom.config.AppConfig)
.env                         secrets only (see .env.example)
warroom/
  config.py                  AppConfig / Secrets / load_config / validate_for_mode
  models/schemas.py          ALL domain models + event types + validate_levels()
  core/clock.py              Clock, SystemClock, SimulatedClock, ET, ensure_utc, to_et
  core/event_bus.py          typed non-blocking async pub/sub
  core/market_calendar.py    NYSE holidays, early closes, sessions (ORB end, EOD flat)
  providers/                 base.py interfaces; market_data, news_data, llm_provider, http retry
  storage/database.py        SQLite (WAL, serialized writes) + startup state recovery
  agents/                    technical_agent, news_agent, risk_agent, jarvis_interface
  war_room.py                3-round debate engine (per-ticker asyncio.Lock)
  orchestrator.py            scan loop wiring agents, war room, broker, bus, storage
  execution/paper_broker.py  simulated fills, SL / TP1 (50% + BE) / TP2 / EOD flat
  backtest/replay_engine.py  SimulatedClock-driven replay through the same pipeline
  dashboard/tui_app.py       Textual UI (bus subscriber)
  voice/speech.py            async speech queue (edge-tts -> pyttsx3 -> text-only)
  notifications/alerts.py    Discord / Telegram
tests/
```

## Invariants (do not break)

1. **No `datetime.now()` / `time.time()` in trading logic.** Take a `Clock`. Models have no
   wall-clock default factories for timestamps — pass `clock.now()`.
2. **UTC everywhere internally.** `UTCDatetime` rejects naive datetimes. Convert with
   `to_et()` only for market-hours logic. Session boundaries come from `MarketCalendar`.
3. **Blocking I/O goes through `asyncio.to_thread()`** (yfinance, sqlite3, pyttsx3,
   feedparser, pygame). The event loop and TUI must never freeze.
4. **Risk veto is absolute.** `TradePlan(status=APPROVED)` validates level ordering and
   size >= 1 itself; use `validate_levels()` for the LONG/SHORT ordering rule.
5. **Optional deps are lazily imported inside `try/except ImportError`.** `--demo` and
   `--smoke-test` must run with only core dependencies installed.
6. **EventBus is lossy by design** (drop-oldest per subscriber). Anything that must not lose
   data (SQLite persistence) is called directly, not via the bus.
7. **Config is strict** (`extra="forbid"`). Add new keys to both `config.yaml` and the
   matching section model in `warroom/config.py`.
8. Secrets only in `.env`; never log `Secrets` values. `ALPACA_PAPER` must stay true.

## Conventions

- Python 3.11+, type hints throughout, Pydantic v2 models, `StrEnum` for enums.
- Package imports are absolute (`from warroom.core.clock import ...`).
- Money/price math in `float`; share counts are `int` via `math.floor`.
- Tests use `SimulatedClock`, mock providers and a temp SQLite path; no network.
