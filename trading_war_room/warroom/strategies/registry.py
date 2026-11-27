"""Build the strategy registry from config: built-ins (with parameter overrides) plus
rule files from ``strategies.rules_dir``."""

from __future__ import annotations

from warroom.config import AppConfig
from warroom.strategies.base import StrategyRegistry
from warroom.strategies.builtin import BUILTINS
from warroom.strategies.rules import load_rules_dir


def build_registry(cfg: AppConfig) -> StrategyRegistry:
    reg = StrategyRegistry()
    sc = cfg.strategies
    for name, cls in BUILTINS.items():
        try:
            reg.available[name] = cls(sc.params.get(name))
        except ValueError as exc:
            reg.errors.append(str(exc))
            reg.available[name] = cls()
    rules, errors = load_rules_dir(sc.rules_dir, reserved=set(BUILTINS))
    reg.available.update(rules)
    reg.errors.extend(errors)
    for n in sc.enabled:
        if n in reg.available:
            reg.enabled.append(n)
        else:
            reg.errors.append(f"enabled strategy '{n}' not found (built-in or {sc.rules_dir}/{n}.yaml)")
    return reg
