"""Strategy interface and registry.

A strategy only answers one question: "on the last completed bar, is there a
LONG and/or SHORT setup, and why?" It returns :class:`Signal` objects. Everything
after that - confidence adjustments for RVOL / index trend / higher-timeframe
trend, cooldowns, the War Room debate, Risk's veto, sizing and execution - is
shared and applies identically to every strategy, built-in or user-defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from warroom.core.assets import AssetClass
from warroom.models.schemas import Direction, IndicatorSnapshot, SetupType

if TYPE_CHECKING:  # pragma: no cover
    from warroom.agents.technical_agent import MacroBias, TickerContext
    from warroom.config import AppConfig


@dataclass
class Signal:
    direction: Direction
    setup_type: SetupType
    base_confidence: float
    rationale: list[str]
    strategy: str = ""


@dataclass
class StrategyContext:
    """What a strategy may look at. ``last`` / ``prev`` are the last two COMPLETED
    setup bars (5m for equities, 15m for crypto by default)."""

    ctx: "TickerContext"
    cfg: "AppConfig"
    bias: "MacroBias"
    last: pd.Series
    prev: pd.Series
    bars: pd.DataFrame

    @property
    def s(self) -> IndicatorSnapshot:
        return self.ctx.snapshot

    @property
    def close_pos(self) -> float:
        """Where the last bar closed within its range: 0 = at the low, 1 = at the high."""
        rng = max(float(self.last["high"] - self.last["low"]), 1e-9)
        return float((self.last["close"] - self.last["low"]) / rng)


class Strategy(ABC):
    name: str = "base"
    description: str = ""
    asset_classes: frozenset[AssetClass] = frozenset({AssetClass.EQUITY, AssetClass.CRYPTO})
    defaults: dict[str, Any] = {}
    builtin: bool = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError(f"strategy {self.name}: unknown parameter(s) {sorted(unknown)}; "
                             f"valid: {sorted(self.defaults) or 'none'}")
        self.params = {**self.defaults, **(params or {})}

    def applies_to(self, asset_class: AssetClass) -> bool:
        return asset_class in self.asset_classes

    @abstractmethod
    def evaluate(self, sc: StrategyContext) -> list[Signal]: ...

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        assets = "/".join(sorted(a.value.lower() for a in self.asset_classes))
        return f"{self.description} [{assets}]" + (f" ({p})" if p else "")

    def sig(self, direction: Direction, setup_type: SetupType, base: float, why: list[str]) -> Signal:
        return Signal(direction, setup_type, base, why, self.name)


@dataclass
class StrategyRegistry:
    """All known strategies (built-in + rule files) and which are enabled."""

    available: dict[str, Strategy] = field(default_factory=dict)
    enabled: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # rule files that failed to load

    def active(self) -> list[Strategy]:
        return [self.available[n] for n in self.enabled if n in self.available]

    def get(self, name: str) -> Strategy | None:
        return self.available.get(name)
