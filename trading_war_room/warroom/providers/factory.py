"""Build providers from config + secrets for a given mode.

Policy:
* DEMO / SMOKE always use mock market data and mock news (zero keys needed).
* REPLAY uses the replay data source (Phase 4) and calendar-only news.
* PAPER uses exactly what config.yaml says; a missing package or key is a
  hard ConfigError. Silently swapping in mock data during paper trading
  would make every risk decision meaningless.
* The LLM always degrades to rule-based dialogue (it is cosmetic by design).
"""

from __future__ import annotations

import logging
from datetime import datetime, time

from warroom.config import AppConfig, ConfigError, Secrets
from warroom.core.clock import Clock, DemoClock, et_to_utc
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import Mode
from warroom.providers.base import LLMProvider, MarketDataProvider, NewsProvider, ProviderUnavailable
from warroom.core.assets import AssetRouter
from warroom.providers.crypto_data import (
    CryptoNewsProvider,
    KrakenDataProvider,
    MockCryptoNewsProvider,
    MockCryptoProvider,
    RoutedMarketData,
    RoutedNewsProvider,
)
from warroom.providers.calendar_feeds import HaltAwareNewsProvider, MacroCalendarUpdater, NasdaqHaltProvider
from warroom.providers.llm_provider import AnthropicProvider, DialogueEngine, OpenAIProvider
from warroom.providers.market_data import AlpacaDataProvider, MockDataProvider, YFinanceDataProvider
from warroom.providers.news_data import (
    AlpacaNewsProvider,
    CalendarOnlyNewsProvider,
    FinnhubNewsProvider,
    MockNewsProvider,
    RSSNewsProvider,
    StaticMacroCalendar,
)

log = logging.getLogger(__name__)


def _secret(s) -> str:
    return s.get_secret_value() if s is not None else ""


def build_calendar(cfg: AppConfig) -> MarketCalendar:
    return MarketCalendar(
        extra_holidays=cfg.calendar.extra_holidays,
        extra_early_closes=cfg.calendar.extra_early_closes,
        orb_minutes=cfg.technical.orb_minutes,
        eod_flat_minutes_before_close=cfg.execution.eod_flat_minutes_before_close,
    )


def build_demo_clock(cfg: AppConfig, calendar: MarketCalendar, real_now: datetime,
                     resume_after: datetime | None = None) -> DemoClock:
    """Latest trading session on/before today (ET), starting at demo.start_time_et.

    ``resume_after``: the last persisted demo activity. If it falls inside that same
    session, the demo resumes one minute later instead of rewinding to the start time,
    so a restart never puts the clock behind saved positions, fills or cooldowns."""
    from datetime import timedelta

    from warroom.core.clock import to_et

    sess = calendar.latest_session_on_or_before(to_et(real_now).date())
    hh, mm = map(int, cfg.demo.start_time_et.split(":"))
    start = max(sess.open, min(et_to_utc(datetime.combine(sess.day, time(hh, mm))), sess.eod_flat))
    if resume_after is not None and sess.open <= resume_after < sess.close:
        start = max(start, resume_after + timedelta(minutes=1))
    return DemoClock(start, cfg.demo.speed_multiplier)


def build_router(cfg: AppConfig, calendar: MarketCalendar) -> AssetRouter:
    return AssetRouter(calendar, cfg.crypto, cfg.execution.commission_per_share)


def build_crypto_market_data(cfg: AppConfig, clock: Clock, mode: Mode) -> MarketDataProvider | None:
    if not cfg.crypto.enabled or not cfg.crypto.symbols:
        return None
    if mode in (Mode.DEMO, Mode.SMOKE) or cfg.crypto.data_provider == "mock":
        if mode is Mode.PAPER:
            raise ConfigError("--paper with crypto.data_provider: mock would paper-trade on fake prices; use kraken")
        return MockCryptoProvider(clock, seed=cfg.providers.mock_seed)
    return KrakenDataProvider(clock, cfg.http)


def build_market_data(cfg: AppConfig, secrets: Secrets, clock: Clock, calendar: MarketCalendar,
                      mode: Mode) -> MarketDataProvider:
    """Equity provider wrapped with the crypto provider in a symbol router."""
    equity = build_equity_market_data(cfg, secrets, clock, calendar, mode)
    crypto = build_crypto_market_data(cfg, clock, mode)
    return RoutedMarketData(build_router(cfg, calendar), equity, crypto) if crypto is not None else equity


def build_equity_market_data(cfg: AppConfig, secrets: Secrets, clock: Clock, calendar: MarketCalendar,
                             mode: Mode) -> MarketDataProvider:
    choice = "mock" if mode in (Mode.DEMO, Mode.SMOKE) else cfg.providers.market_data
    try:
        if choice == "mock":
            return MockDataProvider(clock, calendar, seed=cfg.providers.mock_seed)
        if choice == "yfinance":
            return YFinanceDataProvider(clock)
        if choice == "alpaca":
            return AlpacaDataProvider(clock, _secret(secrets.alpaca_api_key), _secret(secrets.alpaca_secret_key),
                                      cfg.http, feed=cfg.providers.alpaca_feed)
    except ProviderUnavailable as exc:
        raise ConfigError(f"market data provider '{choice}' unavailable: {exc}") from exc
    raise ConfigError(f"unknown market data provider {choice!r}")  # pragma: no cover


async def build_macro_calendar(cfg: AppConfig, calendar: MarketCalendar, clock: Clock,
                               mode: Mode) -> tuple[StaticMacroCalendar, list[str]]:
    """PAPER refreshes the Fed/BLS schedules (weekly cache); other modes use cache + built-ins
    without touching the network. Returns the calendar and human-readable notes/warnings."""
    updater = MacroCalendarUpdater(cfg.http, cfg.news.macro_cache_path, cfg.news.macro_refresh_days)
    if mode is Mode.PAPER:
        fomc, events, notes = await updater.refresh(clock.now())
    else:
        fomc, events, _ = updater.load_cache()
        notes = []
    macro = StaticMacroCalendar(calendar, fetched_fomc=fomc, fetched_events=events)
    if mode is Mode.PAPER:
        notes.extend(macro.coverage_warnings(clock.now()))
    return macro, notes


def build_news(cfg: AppConfig, secrets: Secrets, clock: Clock, calendar: MarketCalendar, mode: Mode,
               market: MarketDataProvider | None = None, macro: StaticMacroCalendar | None = None) -> NewsProvider:
    macro = macro or StaticMacroCalendar(calendar)
    equity_market = market.equity if isinstance(market, RoutedMarketData) else market
    equity_news = build_equity_news(cfg, secrets, clock, calendar, mode, equity_market, macro)
    if not (cfg.crypto.enabled and cfg.crypto.symbols):
        return equity_news
    crypto_market = market.crypto if isinstance(market, RoutedMarketData) else None
    if mode in (Mode.DEMO, Mode.SMOKE):
        crypto_news: NewsProvider = MockCryptoNewsProvider(
            clock, cfg.crypto.symbols, equity_mock_news=equity_news if isinstance(equity_news, MockNewsProvider) else None,
            seed=cfg.providers.mock_seed, scenario_fn=getattr(crypto_market, "scenario", None))
    elif mode is Mode.REPLAY:
        crypto_news = CalendarOnlyNewsProvider(clock, macro)
    else:
        crypto_news = CryptoNewsProvider(clock, cfg.crypto.news_feeds, cfg.http, macro, crypto_market)
    return RoutedNewsProvider(build_router(cfg, calendar), equity_news, crypto_news)


def build_equity_news(cfg: AppConfig, secrets: Secrets, clock: Clock, calendar: MarketCalendar, mode: Mode,
                      market: MarketDataProvider | None, macro: StaticMacroCalendar) -> NewsProvider:
    if mode in (Mode.DEMO, Mode.SMOKE):
        scenario_fn = market.scenario if isinstance(market, MockDataProvider) else None
        return MockNewsProvider(clock, calendar, cfg.app.tickers, seed=cfg.providers.mock_seed, scenario_fn=scenario_fn)
    if mode is Mode.REPLAY:
        return CalendarOnlyNewsProvider(clock, macro)
    choice = cfg.providers.news
    inner: NewsProvider
    try:
        if choice == "mock":
            return MockNewsProvider(clock, calendar, cfg.app.tickers, seed=cfg.providers.mock_seed)
        if choice == "rss":
            inner = RSSNewsProvider(clock, cfg.providers.rss_feeds, cfg.http, macro)
        elif choice == "alpaca":
            inner = AlpacaNewsProvider(clock, _secret(secrets.alpaca_api_key), _secret(secrets.alpaca_secret_key),
                                       cfg.http, macro)
        elif choice == "finnhub":
            inner = FinnhubNewsProvider(clock, _secret(secrets.finnhub_api_key), cfg.http, macro, cfg.app.tickers)
        else:  # pragma: no cover
            raise ConfigError(f"unknown news provider {choice!r}")
    except ProviderUnavailable as exc:
        raise ConfigError(f"news provider '{choice}' unavailable: {exc}") from exc
    if cfg.news.halt_feed_enabled:
        return HaltAwareNewsProvider(inner, NasdaqHaltProvider(clock, cfg.http), cfg.app.tickers)
    return inner


def build_llm(cfg: AppConfig, secrets: Secrets, mode: Mode) -> LLMProvider | None:
    """Return the configured LLM or None (rule-based). Never raises."""
    if mode is Mode.SMOKE:
        return None  # smoke test must be deterministic and offline
    p = cfg.providers
    try:
        if p.llm == "anthropic":
            return AnthropicProvider(_secret(secrets.anthropic_api_key), p.anthropic_model, p.llm_timeout_seconds)
        if p.llm == "openai":
            return OpenAIProvider(_secret(secrets.openai_api_key), p.openai_model, p.llm_timeout_seconds)
    except ProviderUnavailable as exc:
        log.warning("LLM '%s' unavailable (%s); using rule-based dialogue", p.llm, exc)
    return None


async def build_dialogue(cfg: AppConfig, secrets: Secrets, clock: Clock, mode: Mode, db=None) -> DialogueEngine:
    primary = build_llm(cfg, secrets, mode)
    today = clock.now_et().date()
    initial = (today, await db.get_llm_usage(today)) if db is not None else None
    return DialogueEngine(
        clock, primary,
        max_daily_calls=cfg.providers.max_daily_llm_calls,
        timeout=cfg.providers.llm_timeout_seconds,
        user_title=cfg.app.user_title,
        initial_usage=initial,
        on_call=db.set_llm_usage if db is not None else None,
    )
