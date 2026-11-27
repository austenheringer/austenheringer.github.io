"""Change any setting by talking to J.A.R.V.I.S.

* Every setting is addressable by its dotted path (``risk.max_risk_pct``) or a friendly
  alias ("risk per trade").
* A proposed change is applied to a COPY of the config and run through the same strict
  ``AppConfig`` validation as config.yaml, so chat can never produce an invalid or unsafe
  configuration (e.g. risk per trade above 5%, or live trading).
* Accepted changes are written to ``data/config_overrides.yaml`` (config.yaml and its
  comments are never rewritten) and merged on every start, so they survive restarts.
  ``reset settings`` deletes the overrides file.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from warroom.config import AppConfig, ConfigError, parse_config

DEFAULT_OVERRIDES = "data/config_overrides.yaml"

ALIASES: dict[str, str] = {
    "risk per trade": "risk.max_risk_pct", "max risk": "risk.max_risk_pct", "risk": "risk.max_risk_pct",
    "daily loss limit": "risk.max_daily_loss_pct", "daily loss": "risk.max_daily_loss_pct",
    "max positions": "risk.max_open_positions", "max open positions": "risk.max_open_positions",
    "position size cap": "risk.max_position_pct", "max position": "risk.max_position_pct",
    "min reward risk": "risk.min_reward_risk", "reward to risk": "risk.min_reward_risk", "min rr": "risk.min_reward_risk",
    "r:r": "risk.min_reward_risk", "counter trend rr": "risk.counter_trend_min_reward_risk",
    "stop atr": "risk.atr_stop_multiplier", "atr multiplier": "risk.atr_stop_multiplier",
    "tp1": "risk.tp1_r_multiple", "first target": "risk.tp1_r_multiple", "tp2": "risk.tp2_r_multiple",
    "second target": "risk.tp2_r_multiple", "scale out": "risk.tp1_scale_out_pct",
    "spread limit": "risk.max_spread_pct", "max spread": "risk.max_spread_pct",
    "stale data": "risk.max_data_age_seconds", "loss streak": "risk.consecutive_loss_limit",
    "loss cooldown": "risk.consecutive_loss_cooldown_minutes", "pdt": "risk.enforce_pdt_rule",
    "unknown spread": "risk.unknown_spread_policy", "target policy": "risk.target_policy",
    "cooldown": "technical.setup_cooldown_minutes", "setup cooldown": "technical.setup_cooldown_minutes",
    "rvol threshold": "technical.rvol_threshold", "entry cutoff": "technical.no_new_entries_after_et",
    "min confidence": "technical.min_confidence_to_trade", "confidence": "technical.min_confidence_to_trade",
    "opening range": "technical.orb_minutes", "blackout before": "news.blackout_minutes_before",
    "blackout after": "news.blackout_minutes_after", "slippage": "execution.slippage_pct",
    "commission": "execution.commission_per_share", "title": "app.user_title", "call me": "app.user_title",
    "scan interval": "app.scan_interval_seconds", "debates per cycle": "app.max_debates_per_cycle",
    "stocks": "app.tickers", "tickers": "app.tickers", "watchlist": "app.tickers",
    "coins": "crypto.symbols", "crypto": "crypto.symbols", "crypto symbols": "crypto.symbols",
    "taker fee": "crypto.taker_fee_pct", "maker fee": "crypto.maker_fee_pct", "max hold": "crypto.max_hold_minutes",
    "weekend risk": "crypto.weekend_risk_multiplier", "crypto positions": "crypto.max_open_positions",
    "crypto max positions": "crypto.max_open_positions", "crypto spread": "crypto.max_spread_pct",
    "crypto timeframe": "crypto.timeframe", "range lookback": "crypto.range_lookback_minutes",
    "demo speed": "demo.speed_multiplier", "llm calls": "providers.max_daily_llm_calls",
}
BLOCKED = {"execution.live_trading_enabled": "real-money trading is not implemented (docs/LIVE_TRADING_PLAN.md)",
           "storage.db_path": "the database location can only be changed in config.yaml"}
# Changes under these take effect after a restart (providers/clients are built once at startup).
RESTART_PREFIXES = ("providers.", "http.", "storage.", "voice.", "notifications.", "calendar.", "replay.", "account.",
                    "crypto.enabled", "crypto.data_provider", "crypto.exchange", "crypto.news_feeds")
SECTIONS = ("app", "risk", "technical", "news", "execution", "crypto", "strategies", "demo", "providers", "voice",
            "notifications", "calendar", "replay", "http", "storage", "account")


@dataclass
class Change:
    path: str
    old: Any
    new: Any
    config: AppConfig
    restart: bool

    def describe(self) -> str:
        when = " (takes effect after a restart)" if self.restart else ""
        return f"{self.path}: {fmt_value(self.path, self.old)} -> {fmt_value(self.path, self.new)}{when}"


class SettingError(ValueError):
    """A user-facing reason a setting can't be found or changed."""


def fmt_value(path: str, v: Any) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and path.endswith("_pct"):
        return f"{v * 100:.3f}".rstrip("0").rstrip(".") + "%"
    if isinstance(v, list):
        return ", ".join(map(str, v)) or "(empty)"
    return str(v)


def _model_for(path: str) -> tuple[type[BaseModel], str]:
    """Return (section model class, field name) for 'section.field'; raises KeyError."""
    parts = path.split(".")
    model: type[BaseModel] = AppConfig
    for p in parts[:-1]:
        model = model.model_fields[p].annotation  # type: ignore[assignment]
    if parts[-1] not in model.model_fields:
        raise KeyError(path)
    return model, parts[-1]


def resolve(name: str) -> str:
    """Friendly alias or dotted path -> validated dotted path."""
    key = re.sub(r"\s+", " ", name.strip().lower().replace("_", " ")).strip(" ?.")
    if key in ALIASES:
        return ALIASES[key]
    dotted = name.strip().lower().replace(" ", "_")
    candidates = [dotted] if "." in dotted else [f"{sec}.{dotted}" for sec in SECTIONS]
    for c in candidates:
        try:
            _model_for(c)
            return c
        except (KeyError, AttributeError, TypeError):
            continue
    raise SettingError(f"I don't know a setting called '{name}'. Try 'settings' to see them.")


def get_value(cfg: AppConfig, path: str) -> Any:
    obj: Any = cfg
    for p in path.split("."):
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    return obj


def parse_value(path: str, raw: str, current: Any) -> Any:
    """'0.5%' -> 0.005 for *_pct settings; on/off -> bool; comma lists; numbers."""
    r = raw.strip().strip("'\"")
    low = r.lower()
    if isinstance(current, bool):
        if low in ("on", "true", "yes", "enable", "enabled", "1"):
            return True
        if low in ("off", "false", "no", "disable", "disabled", "0"):
            return False
        raise SettingError(f"expected on/off for {path}, got '{raw}'")
    if isinstance(current, list):
        return [x.strip().upper() if path.endswith(("tickers", "symbols")) else x.strip()
                for x in re.split(r"[,\s]+", r) if x.strip()]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(%|x|r|minutes?|mins?|seconds?|secs?|s)?", low)
        if not m:
            raise SettingError(f"expected a number for {path}, got '{raw}'")
        v = float(m.group(1))
        if path.endswith("_pct"):
            # "0.5%" or "0.5 %" -> 0.005; a bare value >= 1 is read as a percentage too (1 -> 1%)
            if m.group(2) == "%" or v >= 1:
                v /= 100
        return int(v) if isinstance(current, int) and v.is_integer() else v
    return r


class SettingsManager:
    """``overrides_path=None`` keeps changes in memory only (smoke tests, replays, unit tests)."""

    def __init__(self, overrides_path: str | Path | None = DEFAULT_OVERRIDES) -> None:
        self.path = Path(overrides_path) if overrides_path is not None else None
        self._memory: dict = {}

    # ------------------------------------------------------------------ persistence

    def load(self) -> dict:
        if self.path is None:
            return copy.deepcopy(self._memory)
        if not self.path.exists():
            return {}
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        if self.path is None:
            self._memory = data
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("# Changes made by talking to J.A.R.V.I.S. Merged over config.yaml on start.\n"
                       "# Delete this file (or say 'reset settings') to go back to config.yaml.\n"
                       + yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def persist(self, change: Change) -> None:
        data = self.load()
        node = data
        parts = change.path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = change.new
        self._write(data)

    def reset(self) -> bool:
        if self.path is None:
            had, self._memory = bool(self._memory), {}
            return had
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    # ------------------------------------------------------------------ planning

    def plan(self, cfg: AppConfig, path: str, value: Any) -> Change:
        """Validate a change against the full config schema. Raises ValueError with a readable reason."""
        if path in BLOCKED:
            raise SettingError(f"{path} can't be changed from chat: {BLOCKED[path]}")
        old = get_value(cfg, path)
        raw = cfg.model_dump(mode="json")
        node = raw
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
        try:
            new_cfg = parse_config(raw, source="your change")
        except ConfigError as exc:
            reason = "; ".join(line.strip(" -") for line in str(exc).splitlines()[1:]) or str(exc)
            raise SettingError(f"that change isn't allowed: {reason}") from exc
        return Change(path=path, old=copy.deepcopy(old), new=get_value(new_cfg, path), config=new_cfg,
                      restart=path.startswith(RESTART_PREFIXES))

    def plan_text(self, cfg: AppConfig, name: str, raw_value: str) -> Change:
        path = resolve(name)
        return self.plan(cfg, path, parse_value(path, raw_value, get_value(cfg, path)))

    # ------------------------------------------------------------------ display

    def describe(self, cfg: AppConfig, section: str | None = None) -> str:
        overrides = self.load()
        rev = {}
        for alias, path in ALIASES.items():
            rev.setdefault(path, alias)
        lines = []
        for sec in ([section] if section else ["app", "risk", "technical", "news", "crypto", "strategies", "execution"]):
            if sec not in SECTIONS:
                raise SettingError(f"unknown section '{sec}' (try: {', '.join(SECTIONS[:8])})")
            lines.append(f"[{sec}]")
            for field, value in getattr(cfg, sec).model_dump().items():
                path = f"{sec}.{field}"
                if isinstance(value, dict) and field in ("instruments",):
                    continue
                mark = " *" if field in (overrides.get(sec) or {}) else ""
                alias = f'  ("{rev[path]}")' if path in rev else ""
                lines.append(f"  {field} = {fmt_value(path, value)}{mark}{alias}")
        lines.append("(* = changed by you; say 'set <name> to <value>')")
        return "\n".join(lines)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
