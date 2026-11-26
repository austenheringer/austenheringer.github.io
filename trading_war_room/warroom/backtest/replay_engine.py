"""Historical replay: drives a SimulatedClock minute by minute through the SAME
Orchestrator / War Room / Risk / PaperBroker pipeline used live.

Data source resolution (``replay.source``):
* ``csv``      - ``<data_dir>/<YYYY-MM-DD>/<TICKER>.csv`` (plus prior days for warm-up)
* ``alpaca``   - Alpaca historical bars (needs keys)
* ``yfinance`` - only for dates within ~29 days (Yahoo's 1m limit)
* ``mock``     - deterministic synthetic tape (offline demo of the replay itself)
* ``auto``     - csv if every ticker has a file for the day, else alpaca if keys,
                 else yfinance if installed and recent enough, else a clear error.

Historical bid/ask is not available from these sources, so quotes are
synthesised from bar closes with ``replay.assumed_spread_pct`` (noted in the
report). Historical headlines are not available for free either: replay news
is the macro calendar only (FOMC/BLS/user events, so blackouts still apply).
The database is in-memory: a replay can never touch live/paper state.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta

import pandas as pd

from warroom.backtest.metrics import ReplayReport
from warroom.config import AppConfig, ConfigError, Secrets, parse_config
from warroom.core.clock import Clock, SimulatedClock, et_to_utc, to_et
from warroom.models.schemas import Mode, PlanStatus, Quote, Timeframe
from warroom.orchestrator import Orchestrator
from warroom.providers import factory
from warroom.providers.base import MarketDataProvider, ProviderUnavailable, completed_only
from warroom.providers.market_data import AlpacaDataProvider, CsvDataProvider, MockDataProvider, YFinanceDataProvider


class ReplayDataUnavailable(ConfigError):
    pass


def veto_label(reason: str) -> str:
    """Group veto reasons by type: drop parentheticals and numbers ('R:R 1.42 to TP1 ...' -> 'R:R # to TP1 ...')."""
    import re

    base = reason.split(" (")[0]
    base = re.sub(r"(?<![A-Za-z])\$?-?\d[\d,]*(\.\d+)?%?", "#", base)
    base = re.sub(r"(already exposed to|for) [A-Z.]{1,6}\b", r"\1 <ticker>", base)
    return base.split(";")[0][:70]


class SyntheticQuoteAdapter(MarketDataProvider):
    """Passes bars through; synthesises quotes from the last completed 1m close."""

    has_real_quotes = True

    def __init__(self, inner: MarketDataProvider, clock: Clock, spread_pct: float) -> None:
        super().__init__(clock)
        self.inner = inner
        self.spread = spread_pct
        self.name = f"{inner.name}+synthetic-quotes"

    async def get_bars(self, ticker, timeframe, start, end=None) -> pd.DataFrame:
        return await self.inner.get_bars(ticker, timeframe, start, end)

    async def get_quote(self, ticker: str) -> Quote | None:
        now = self.clock.now()
        df = await self.inner.get_bars(ticker, Timeframe.M1, now - timedelta(minutes=30))
        df = completed_only(df, Timeframe.M1, now)
        if df.empty:
            return None
        last = float(df["close"].iloc[-1])
        half = last * self.spread / 2
        return Quote(ticker=ticker, timestamp=df.index[-1].to_pydatetime() + timedelta(minutes=1),
                     bid=round(last - half, 4), ask=round(last + half, 4), last=last)

    async def aclose(self) -> None:
        await self.inner.aclose()


def replay_config(cfg: AppConfig, tickers: list[str] | None) -> AppConfig:
    raw = cfg.model_dump(mode="json")
    if tickers:
        raw["app"]["tickers"] = [t for t in tickers if t not in raw["app"]["macro_tickers"]]
    raw["account"]["starting_equity"] = cfg.replay.starting_equity
    raw["account"]["starting_cash"] = cfg.replay.starting_equity
    raw["notifications"]["discord_enabled"] = False
    raw["notifications"]["telegram_enabled"] = False
    return parse_config(raw, source="replay overrides")


def resolve_source(cfg: AppConfig, secrets: Secrets, day: date, clock: Clock) -> tuple[MarketDataProvider, str]:
    src = cfg.replay.source
    calendar = factory.build_calendar(cfg)
    symbols = cfg.app.tickers + cfg.app.macro_tickers
    tried: list[str] = []

    def csv():
        p = CsvDataProvider(clock, cfg.replay.data_dir, cfg.replay.assumed_spread_pct)
        missing = [s for s in symbols if not p.available(s, day)]
        if missing:
            raise ReplayDataUnavailable(f"CSV missing for {day}: {', '.join(missing)} "
                                        f"(expected {cfg.replay.data_dir}/{day}/<TICKER>.csv)")
        return p, "csv"

    def alpaca():
        if not (secrets.alpaca_api_key and secrets.alpaca_secret_key):
            raise ReplayDataUnavailable("Alpaca keys not set (ALPACA_API_KEY / ALPACA_SECRET_KEY)")
        inner = AlpacaDataProvider(clock, secrets.alpaca_api_key.get_secret_value(),
                                   secrets.alpaca_secret_key.get_secret_value(), cfg.http, feed=cfg.providers.alpaca_feed)
        return SyntheticQuoteAdapter(inner, clock, cfg.replay.assumed_spread_pct), "alpaca"

    def yfin():
        if (date.today() - day).days > 29:
            raise ReplayDataUnavailable(f"yfinance only serves 1m bars for ~29 days; {day} is too old")
        try:
            inner = YFinanceDataProvider(clock)
        except ProviderUnavailable as exc:
            raise ReplayDataUnavailable(str(exc)) from exc
        return SyntheticQuoteAdapter(inner, clock, cfg.replay.assumed_spread_pct), "yfinance"

    def mock():
        return MockDataProvider(clock, calendar, seed=cfg.providers.mock_seed), "mock (synthetic)"

    order = {"csv": [csv], "alpaca": [alpaca], "yfinance": [yfin], "mock": [mock], "auto": [csv, alpaca, yfin]}[src]
    for fn in order:
        try:
            return fn()
        except ReplayDataUnavailable as exc:
            tried.append(str(exc))
    raise ReplayDataUnavailable(
        f"No historical data for {day}:\n  - " + "\n  - ".join(tried)
        + "\nOptions: add CSVs, set Alpaca keys, or use replay.source: mock for a synthetic dry run.")


class ReplayEngine:
    def __init__(self, cfg: AppConfig, secrets: Secrets, day: date, tickers: list[str] | None = None) -> None:
        self.cfg = replay_config(cfg, tickers)
        self.secrets = secrets
        self.day = day
        self.calendar = factory.build_calendar(self.cfg)

    async def run(self, progress=None, max_minutes: int | None = None) -> ReplayReport:
        """Replay the session. ``max_minutes`` truncates the run (tests / quick looks)."""
        sess = self.calendar.session(self.day)
        if sess is None:
            raise ReplayDataUnavailable(f"{self.day} is not a trading day "
                                        f"({self.calendar.holiday_name(self.day) or 'weekend'})")
        clock = SimulatedClock(sess.open + timedelta(minutes=1))
        market, source = resolve_source(self.cfg, self.secrets, self.day, clock)
        orch = await Orchestrator.build(self.cfg, self.secrets, Mode.REPLAY, clock=clock, db_path=":memory:",
                                        market=market)
        await orch.start()
        start_equity = orch.broker.equity()
        curve: list[tuple[datetime, float]] = [(clock.now(), start_equity)]
        closed = []
        step = timedelta(minutes=self.cfg.replay.scan_every_minutes)
        try:
            end = sess.close + timedelta(minutes=1)
            if max_minutes is not None:
                end = min(end, clock.now() + timedelta(minutes=max_minutes))
            while clock.now() <= end:
                await orch.run_cycle()
                curve.append((clock.now(), orch.broker.equity()))
                closed.extend(p for p in orch.broker.closed_today if p not in closed)
                if progress:
                    progress(clock.now(), orch)
                clock.advance(step)
            # anything still open after the bell (should not happen: EOD flat) is closed for the report
            if orch.broker.open_positions:
                await orch.broker.flatten_all()
                closed.extend(p for p in orch.broker.closed_today if p not in closed)
            history = list(orch.war_room.history)
            report = ReplayReport(
                day=self.day, tickers=self.cfg.app.tickers, source=source, starting_equity=start_equity,
                ending_equity=round(orch.broker.equity(), 2), trades=closed, equity_curve=curve,
                debates=len(history),
                approved=sum(t.plan.status is PlanStatus.APPROVED for t in history),
                watchlist=sum(t.plan.status is PlanStatus.WATCHLIST for t in history),
                vetoed=sum(t.plan.status is PlanStatus.VETOED for t in history),
                veto_reasons=Counter(veto_label(r) for t in history for r in t.plan.veto_reasons),
            )
            report.notes.append(f"quotes synthesised with a {self.cfg.replay.assumed_spread_pct:.2%} spread; "
                                "news = macro calendar only (no historical headlines)")
            if source.startswith("mock"):
                report.notes.append("SYNTHETIC tape: this measures the pipeline, not a real day's market")
            return report
        finally:
            await orch.shutdown()
