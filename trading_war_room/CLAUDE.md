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
python main.py --replay 2025-01-15 --replay-source mock   # replay on the synthetic tape
python main.py --check-providers   # live checks of configured providers/keys
python main.py --demo --headless   # console event stream instead of the TUI
pytest -q                          # ~35s, no network
```

## Layout

```
main.py                      CLI entrypoint
config.yaml                  non-secret settings (validated by warroom.config.AppConfig)
.env                         secrets only (see .env.example)
warroom/
  config.py                  AppConfig / Secrets / load_config / validate_for_mode
  models/schemas.py          ALL domain models + event types + validate_levels()
  core/clock.py              Clock, SystemClock (--paper), DemoClock (--demo), SimulatedClock (--replay/tests)
  core/event_bus.py          typed non-blocking async pub/sub
  core/market_calendar.py    NYSE holidays, early closes, sessions (ORB end, EOD flat; eod_flat=None for crypto)
  core/assets.py             AssetClass, Instrument (lot step/min order/fees), CryptoSchedule (24/7), AssetRouter
  providers/base.py          MarketDataProvider / NewsProvider / LLMProvider + bar helpers
  providers/http.py          RetryingHttpClient (429/5xx backoff + full jitter, Retry-After)
  providers/market_data.py   Mock (deterministic scenarios), yfinance, Alpaca REST
  providers/news_data.py     Mock, RSS, Alpaca, Finnhub, CalendarOnly + StaticMacroCalendar (FOMC dates)
  providers/sentiment.py     lexicon headline scorer (negation/contrast; accuracy-floor test)
  providers/calendar_feeds.py Fed FOMC + BLS schedule updater (cached), Nasdaq halt feed, HaltAwareNewsProvider
  providers/healthcheck.py   live contract checks (`main.py --check-providers`)
  analysis/indicators.py     pure vectorised EMA/RSI(Wilder)/ATR/VWAP/ORB/premarket/PDH-PDL/RVOL/swings
  providers/llm_provider.py  DialogueEngine (rewrite-only, number guard, daily cap) + Anthropic/OpenAI/RuleBased
  providers/crypto_data.py   Kraken public REST, MockCryptoProvider (24/7, bridged days), crypto RSS, Routed* providers
  providers/factory.py       builds providers per mode (DEMO/SMOKE force mocks; PAPER never silently falls back)
  storage/database.py        SQLite (WAL, serialized writes) + startup state recovery
  agents/technical_agent.py  TickerContext, macro bias, LONG/SHORT setups, cooldown gate, halt inference
  agents/news_agent.py       blackout_status() (pure), recency-weighted sentiment, conflicts
  agents/risk_agent.py       compute_stop/compute_targets/position_size (pure) + RiskAgent veto engine
  agents/jarvis_interface.py command parser, briefings, status/why/transcript, JarvisContext protocol
  war_room.py                3-round debate; per-ticker lock + global portfolio lock around vote+execution
  orchestrator.py            build()/start()/run_cycle()/run_forever()/shutdown(); implements JarvisContext
  execution/base.py          Broker protocol (PaperBroker is the only implementation)
  execution/paper_broker.py  simulated fills, SL / TP1 (50% + BE) / TP2 / EOD flat (equities) / time stop (crypto)
  backtest/replay_engine.py  SimulatedClock replay through the real Orchestrator (csv/alpaca/yfinance/mock)
  backtest/metrics.py        ReplayReport, profit factor, max drawdown
  dashboard/tui_app.py       Textual UI (bus subscriber); tested headlessly via App.run_test()
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
9. **No look-ahead.** Providers return only completed bars (`open + timeframe <= now`) and only
   news with `published_at <= now`. Mock data is deterministic per (seed, ticker, date).
10. **LLMs are cosmetic.** Agents compose a deterministic draft; `DialogueEngine.render` may only
    rephrase it and rejects any rewrite that changes/adds a number. Never let LLM output feed
    risk math or verdicts.
11. **One database per mode** (`data/trades_{mode}.db`) so demo trades never touch paper-mode
    circuit breakers. Recovery is derived from the fills/positions ledger; restored flags are
    OR-ed in (a restart can only make the system more cautious).
12. Bars travel as pandas DataFrames (UTC index of bar open times, columns open/high/low/close/volume),
    normalised by `providers.base.normalize_bars`.

13. **Quantities are floats.** Always round sizes DOWN with `Instrument.floor_qty` (lot step: 1 share,
    1e-8 BTC) and check `Instrument.meets_minimum`. Print with `format_qty` / `format_price`.
14. **Crypto R:R, sizing and targets are net of fees** (`FeeModel`, `risk_per_unit`, `fee_aware_target`).
    With zero commission the math reduces exactly to the equity formulas.
15. **No real-money execution.** `execution.live_trading_enabled` is rejected by config validation;
    see docs/LIVE_TRADING_PLAN.md before touching `execution/`.
16. Invariant test: no replayed trade may lose much more than its risk budget
    (`_assert_losses_bounded`). Keep it green; it caught the stale-bar and overnight-gap bugs.

## Risk policy decisions (config `risk.*`)

- Stop: swing level ± `stop_buffer_atr`·ATR unless farther than `atr_stop_multiplier`·ATR → ATR stop.
- Targets: R multiples pulled in to the nearest opposing structure (PDH/PDL, premarket H/L, ORB H/L);
  R:R is measured entry→TP1 after capping, so the 2.0 gate actually bites.
- Daily loss limit counts realized + unrealized P&L against start-of-day equity.
- No real bid/ask (yfinance) → `unknown_spread_policy` (default veto).
- Counter-trend (SPY/QQQ against the trade): TP1 ≥ `min_reward_risk` AND blended R:R ≥
  `counter_trend_min_reward_risk`.
- Only pre-existing levels (PDH/PDL, premarket, ORB) cap targets; the recent swing is excluded
  because after a breakout it is the breakout bar itself.
- Risk collects every veto reason (no short-circuit) so `why <TICKER>` is complete.
- Round 3 re-evaluates against the live portfolio under the global portfolio lock and executes
  inside the same lock: two debates can never both take the last slot.

## Exit codes (main.py)

0 success · 1 smoke/provider check failed · 2 configuration or data-availability error.

## Conventions

- Python 3.11+, type hints throughout, Pydantic v2 models, `StrEnum` for enums.
- Package imports are absolute (`from warroom.core.clock import ...`).
- Money/price math in `float`; share counts are `int` via `math.floor`.
- Tests use `SimulatedClock`, mock providers and a temp SQLite path; no network.
