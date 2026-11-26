"""Configuration loading and validation.

``config.yaml`` -> :class:`AppConfig` (non-secret settings, strict: unknown
keys are errors so a typo like ``max_risk_pt`` can't silently fall back to a
default). ``.env`` -> :class:`Secrets`. :func:`validate_for_mode` adds
cross-cutting checks that depend on the CLI mode.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

from warroom.models.schemas import Mode


class ConfigError(Exception):
    """Raised for any invalid configuration. Message is user-facing."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSection(_Section):
    user_title: str = Field(default="Sir", min_length=1, max_length=40)
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "NVDA", "TSLA"], min_length=1, max_length=50)
    macro_tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"], min_length=1)
    scan_interval_seconds: float = Field(default=15, ge=1, le=3600)
    max_debates_per_cycle: int = Field(default=3, ge=1, le=20)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("tickers", "macro_tickers")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for t in v:
            t = t.strip().upper()
            if not t or not all(c.isalnum() or c in ".-" for c in t) or len(t) > 10:
                raise ValueError(f"invalid ticker symbol {t!r}")
            if t not in out:
                out.append(t)
        return out


class DemoSection(_Section):
    start_time_et: str = "09:50"
    speed_multiplier: float = Field(default=10.0, gt=0, le=600)
    scan_interval_seconds: float = Field(default=60, ge=1, le=3600)

    @field_validator("start_time_et")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        try:
            hh, mm = map(int, v.split(":"))
        except ValueError:
            raise ValueError("start_time_et must be HH:MM") from None
        if not (9 * 60 + 30 <= hh * 60 + mm < 16 * 60):
            raise ValueError("start_time_et must be within regular hours (09:30-15:59)")
        return f"{hh:02d}:{mm:02d}"


class AccountSection(_Section):
    starting_equity: float = Field(default=100_000.0, gt=0)
    starting_cash: float = Field(default=100_000.0, gt=0)

    @model_validator(mode="after")
    def _cash_le_equity(self) -> "AccountSection":
        if self.starting_cash > self.starting_equity:
            raise ValueError("starting_cash cannot exceed starting_equity")
        return self


class RiskSection(_Section):
    max_risk_pct: float = Field(default=0.01, gt=0, le=0.05)
    max_position_pct: float = Field(default=0.25, gt=0, le=1.0)
    min_reward_risk: float = Field(default=2.0, ge=1.0, le=20)
    counter_trend_min_reward_risk: float = Field(default=2.5, ge=1.0, le=20)
    atr_stop_multiplier: float = Field(default=1.5, gt=0, le=10)
    stop_policy: Literal["structure_within_atr", "atr_only"] = "structure_within_atr"
    stop_buffer_atr: float = Field(default=0.1, ge=0, le=2)
    min_stop_atr: float = Field(default=0.25, gt=0, le=5)
    target_policy: Literal["structural_cap", "fixed_r"] = "structural_cap"
    tp1_r_multiple: float = Field(default=2.0, gt=0)
    tp2_r_multiple: float = Field(default=3.0, gt=0)
    tp1_scale_out_pct: float = Field(default=0.5, gt=0, lt=1)
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=0.2)
    daily_loss_includes_unrealized: bool = True
    consecutive_loss_limit: int = Field(default=3, ge=1, le=20)
    consecutive_loss_cooldown_minutes: int = Field(default=30, ge=0, le=24 * 60)
    max_open_positions: int = Field(default=3, ge=1, le=50)
    max_spread_pct: float = Field(default=0.0015, gt=0, le=0.05)
    unknown_spread_policy: Literal["veto", "allow"] = "veto"
    max_data_age_seconds: int = Field(default=120, ge=5, le=3600)
    enforce_pdt_rule: bool = False
    pdt_equity_threshold: float = Field(default=25_000.0, ge=0)
    pdt_max_day_trades: int = Field(default=3, ge=0, le=10)

    @model_validator(mode="after")
    def _targets(self) -> "RiskSection":
        if self.tp2_r_multiple <= self.tp1_r_multiple:
            raise ValueError("tp2_r_multiple must be greater than tp1_r_multiple")
        if self.counter_trend_min_reward_risk < self.min_reward_risk:
            raise ValueError("counter_trend_min_reward_risk must be >= min_reward_risk")
        if self.min_stop_atr >= self.atr_stop_multiplier:
            raise ValueError("min_stop_atr must be smaller than atr_stop_multiplier")
        if self.tp1_r_multiple < self.min_reward_risk:
            raise ValueError("tp1_r_multiple is below min_reward_risk: every trade would be vetoed")
        if self.max_risk_pct >= self.max_daily_loss_pct:
            raise ValueError("max_risk_pct must be smaller than max_daily_loss_pct (one trade could halt the day)")
        return self


class TechnicalSection(_Section):
    setup_cooldown_minutes: int = Field(default=15, ge=0, le=390)
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50])
    rsi_period: int = Field(default=14, ge=2, le=100)
    atr_period: int = Field(default=14, ge=2, le=100)
    rvol_lookback_days: int = Field(default=10, ge=1, le=60)
    rvol_threshold: float = Field(default=1.5, gt=0)
    orb_minutes: int = Field(default=15, ge=1, le=120)
    swing_lookback_bars: int = Field(default=10, ge=2, le=200)
    history_days: int = Field(default=12, ge=3, le=60)
    vwap_touch_pct: float = Field(default=0.0015, ge=0, le=0.02)
    min_confidence_to_trade: float = Field(default=0.55, ge=0, le=1)
    no_new_entries_after_et: str = "15:30"

    @field_validator("no_new_entries_after_et")
    @classmethod
    def _cutoff(cls, v: str) -> str:
        try:
            hh, mm = map(int, v.split(":"))
        except ValueError:
            raise ValueError("no_new_entries_after_et must be HH:MM") from None
        if not (10 * 60 <= hh * 60 + mm <= 16 * 60):
            raise ValueError("no_new_entries_after_et must be between 10:00 and 16:00")
        return f"{hh:02d}:{mm:02d}"

    @field_validator("ema_periods")
    @classmethod
    def _emas(cls, v: list[int]) -> list[int]:
        if len(v) != 3 or any(p < 2 for p in v) or not (v[0] < v[1] < v[2]):
            raise ValueError("ema_periods must be three ascending periods >= 2 (fast, mid, slow)")
        return v


class NewsSection(_Section):
    blackout_minutes_before: int = Field(default=15, ge=0, le=240)
    blackout_minutes_after: int = Field(default=15, ge=0, le=240)
    conflict_sentiment_threshold: float = Field(default=0.3, gt=0, le=1)
    halt_feed_enabled: bool = True
    macro_cache_path: str = "data/macro_cache.json"
    macro_refresh_days: int = Field(default=7, ge=1, le=90)


class ExecutionSection(_Section):
    slippage_pct: float = Field(default=0.0005, ge=0, le=0.02)
    commission_per_share: float = Field(default=0.0, ge=0)
    commission_min: float = Field(default=0.0, ge=0)
    short_margin_requirement: float = Field(default=1.5, ge=1.0, le=3.0)
    eod_flat_minutes_before_close: int = Field(default=5, ge=1, le=60)


class ProvidersSection(_Section):
    market_data: Literal["mock", "yfinance", "alpaca"] = "mock"
    news: Literal["mock", "rss", "alpaca", "finnhub"] = "mock"
    llm: Literal["rule_based", "anthropic", "openai"] = "rule_based"
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=20, gt=0, le=120)
    max_daily_llm_calls: int = Field(default=200, ge=0)
    rss_feeds: list[str] = Field(default_factory=list)
    mock_seed: int = 42
    alpaca_feed: Literal["iex", "sip"] = "iex"

    @field_validator("rss_feeds")
    @classmethod
    def _https(cls, v: list[str]) -> list[str]:
        for url in v:
            if not url.startswith(("https://", "http://")):
                raise ValueError(f"rss feed must be an http(s) URL: {url!r}")
        return v


class HttpSection(_Section):
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_base_seconds: float = Field(default=0.5, gt=0)
    backoff_max_seconds: float = Field(default=30, gt=0)


class VoiceSection(_Section):
    edge_voice: str = "en-GB-RyanNeural"
    rate: str = "+0%"
    max_queue: int = Field(default=10, ge=1, le=100)


class StorageSection(_Section):
    db_path: str = "data/trades_{mode}.db"

    def resolve_db_path(self, mode: Mode) -> str:
        """Per-mode database so demo/smoke trades never feed paper-mode circuit breakers."""
        return self.db_path.replace("{mode}", mode.value.lower())


class ReplaySection(_Section):
    source: Literal["auto", "csv", "alpaca", "yfinance", "mock"] = "auto"
    data_dir: str = "data/history"
    starting_equity: float = Field(default=100_000.0, gt=0)
    assumed_spread_pct: float = Field(default=0.0005, ge=0, le=0.02)
    scan_every_minutes: int = Field(default=1, ge=1, le=60)


class NotificationsSection(_Section):
    discord_enabled: bool = False
    telegram_enabled: bool = False
    notify_on: list[Literal["APPROVED", "VETOED", "TRADE_EXECUTED", "CIRCUIT_BREAKER"]] = Field(
        default_factory=lambda: ["APPROVED", "TRADE_EXECUTED", "CIRCUIT_BREAKER"]
    )


class CalendarSection(_Section):
    extra_holidays: list[date] = Field(default_factory=list)
    extra_early_closes: list[date] = Field(default_factory=list)


class AppConfig(_Section):
    app: AppSection = Field(default_factory=AppSection)
    demo: DemoSection = Field(default_factory=DemoSection)
    account: AccountSection = Field(default_factory=AccountSection)
    risk: RiskSection = Field(default_factory=RiskSection)
    technical: TechnicalSection = Field(default_factory=TechnicalSection)
    news: NewsSection = Field(default_factory=NewsSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    providers: ProvidersSection = Field(default_factory=ProvidersSection)
    http: HttpSection = Field(default_factory=HttpSection)
    voice: VoiceSection = Field(default_factory=VoiceSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    replay: ReplaySection = Field(default_factory=ReplaySection)
    notifications: NotificationsSection = Field(default_factory=NotificationsSection)
    calendar: CalendarSection = Field(default_factory=CalendarSection)

    @model_validator(mode="after")
    def _cross(self) -> "AppConfig":
        overlap = set(self.app.tickers) & set(self.app.macro_tickers)
        if overlap:
            raise ValueError(f"tickers and macro_tickers overlap: {sorted(overlap)}")
        return self


class Secrets(BaseModel):
    """Values read from the environment (.env). Never logged."""

    model_config = ConfigDict(frozen=True)

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    alpaca_paper: bool = True
    finnhub_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    discord_webhook_url: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Secrets":
        env = dict(os.environ) if env is None else env

        def get(name: str) -> str | None:
            val = (env.get(name) or "").strip()
            return val or None

        paper_raw = (get("ALPACA_PAPER") or "true").lower()
        return cls(
            alpaca_api_key=get("ALPACA_API_KEY"),
            alpaca_secret_key=get("ALPACA_SECRET_KEY"),
            alpaca_paper=paper_raw not in ("0", "false", "no"),
            finnhub_api_key=get("FINNHUB_API_KEY"),
            anthropic_api_key=get("ANTHROPIC_API_KEY"),
            openai_api_key=get("OPENAI_API_KEY"),
            discord_webhook_url=get("DISCORD_WEBHOOK_URL"),
            telegram_bot_token=get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=get("TELEGRAM_CHAT_ID"),
        )


def _format_validation_error(err: ValidationError, source: str) -> str:
    lines = [f"Invalid configuration in {source}:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate ``config.yaml``. Raises :class:`ConfigError`."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return parse_config(raw, source=str(path))


def parse_config(raw: dict, source: str = "<dict>") -> AppConfig:
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc


def validate_for_mode(config: AppConfig, secrets: Secrets, mode: Mode) -> list[str]:
    """Mode-dependent checks. Returns warnings; raises ConfigError on fatal problems."""
    errors: list[str] = []
    warnings: list[str] = []
    p = config.providers

    if mode is Mode.PAPER:
        if p.market_data == "mock":
            errors.append("--paper requires a real market data provider (providers.market_data: yfinance | alpaca)")
        if p.market_data == "alpaca" or p.news == "alpaca":
            if not (secrets.alpaca_api_key and secrets.alpaca_secret_key):
                errors.append("Alpaca provider selected but ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
        if p.news == "finnhub" and not secrets.finnhub_api_key:
            errors.append("providers.news is finnhub but FINNHUB_API_KEY is not set")
        if p.news == "mock":
            warnings.append("--paper is using MOCK news: blackout windows will not reflect real events")
        if p.market_data == "yfinance":
            warnings.append(
                "yfinance data is delayed/unofficial with no real bid/ask: expect stale-data vetoes"
                + (" and spread vetoes on every setup (risk.unknown_spread_policy=veto)"
                   if config.risk.unknown_spread_policy == "veto" else "")
            )
        if "{mode}" not in config.storage.db_path:
            warnings.append("storage.db_path has no {mode} placeholder: demo and paper share one database")
        if p.news == "rss" and not p.rss_feeds:
            errors.append("providers.news is rss but providers.rss_feeds is empty")
    if not secrets.alpaca_paper:
        errors.append("ALPACA_PAPER must be true: this system is paper/advisory only")

    if p.llm == "anthropic" and not secrets.anthropic_api_key:
        warnings.append("providers.llm is anthropic but ANTHROPIC_API_KEY is unset; using rule-based dialogue")
    if p.llm == "openai" and not secrets.openai_api_key:
        warnings.append("providers.llm is openai but OPENAI_API_KEY is unset; using rule-based dialogue")
    if config.notifications.discord_enabled and not secrets.discord_webhook_url:
        warnings.append("Discord notifications enabled but DISCORD_WEBHOOK_URL is unset; disabled")
    if config.notifications.telegram_enabled and not (secrets.telegram_bot_token and secrets.telegram_chat_id):
        warnings.append("Telegram notifications enabled but TELEGRAM_BOT_TOKEN/CHAT_ID unset; disabled")

    if errors:
        raise ConfigError("Configuration not valid for mode " + mode.value + ":\n  - " + "\n  - ".join(errors))
    return warnings
