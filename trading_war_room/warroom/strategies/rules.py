"""User strategies as declarative rule files - never executable code.

A rule file (``strategies/<name>.yaml``)::

    name: rsi_oversold_reversal
    description: Buy oversold dips that reclaim VWAP inside an uptrend
    assets: [equity, crypto]          # optional, default both
    confidence: 0.5                   # base confidence before shared adjustments
    long_when:                        # ALL conditions must hold (optional)
      - rsi14 < 35
      - close > ema50
      - close crosses_above vwap
    short_when: []                    # optional

Each condition is ``<operand> <op> <operand>`` or ``<field> between <a> <b>``:

* operands: a field name, a number, or ``field * number`` (e.g. ``volume > avg_volume * 2``)
* ops: ``<  <=  >  >=  crosses_above  crosses_below``
* ``crosses_above`` / ``crosses_below`` need a bar field on the left (close/high/low/open):
  it compares the previous bar's value with the current one.

Conditions are parsed against a fixed whitelist of fields and operators. Nothing
is ``eval``-ed; an unknown word is a validation error, not code. A missing value
(e.g. ``orb_high`` on crypto) makes the condition false, so the strategy stays quiet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from warroom.core.assets import AssetClass
from warroom.models.schemas import Direction, SetupType
from warroom.strategies.base import Signal, Strategy, StrategyContext

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # drop NaN


def _dir(d) -> float:
    return 0.0 if d is None else (1.0 if d is Direction.LONG else -1.0)


FIELDS: dict[str, tuple[str, Callable[[StrategyContext], float | None]]] = {
    "price": ("latest price", lambda c: c.s.price),
    "open": ("last bar open", lambda c: _num(c.last["open"])),
    "high": ("last bar high", lambda c: _num(c.last["high"])),
    "low": ("last bar low", lambda c: _num(c.last["low"])),
    "close": ("last bar close", lambda c: _num(c.last["close"])),
    "volume": ("last bar volume", lambda c: _num(c.last["volume"])),
    "prev_open": ("previous bar open", lambda c: _num(c.prev["open"])),
    "prev_high": ("previous bar high", lambda c: _num(c.prev["high"])),
    "prev_low": ("previous bar low", lambda c: _num(c.prev["low"])),
    "prev_close": ("previous bar close", lambda c: _num(c.prev["close"])),
    "prev_volume": ("previous bar volume", lambda c: _num(c.prev["volume"])),
    "avg_volume": ("average volume of the 20 bars before the last", lambda c: _num(c.bars["volume"].iloc[-21:-1].mean())
                   if len(c.bars) > 1 else None),
    "bar_range": ("last bar high - low", lambda c: _num(c.last["high"] - c.last["low"])),
    "close_position": ("where the last bar closed in its range, 0=low 1=high", lambda c: c.close_pos),
    "vwap": ("session VWAP (UTC-day for crypto)", lambda c: c.s.vwap),
    "ema9": ("EMA 9", lambda c: c.s.ema9),
    "ema21": ("EMA 21", lambda c: c.s.ema21),
    "ema50": ("EMA 50", lambda c: c.s.ema50),
    "rsi14": ("RSI 14 (0-100)", lambda c: c.s.rsi14),
    "atr14": ("ATR 14", lambda c: c.s.atr14),
    "rvol": ("relative volume vs the same time of day (1.0 = normal)", lambda c: c.s.rvol),
    "spread_pct": ("bid/ask spread as a fraction", lambda c: c.s.spread_pct),
    "orb_high": ("opening-range high (stocks)", lambda c: c.s.orb_high),
    "orb_low": ("opening-range low (stocks)", lambda c: c.s.orb_low),
    "premarket_high": ("premarket high (stocks)", lambda c: c.s.premarket_high),
    "premarket_low": ("premarket low (stocks)", lambda c: c.s.premarket_low),
    "prior_high": ("prior session high", lambda c: c.ctx.prior_high),
    "prior_low": ("prior session low", lambda c: c.ctx.prior_low),
    "prior_close": ("prior session close", lambda c: c.ctx.prior_close),
    "swing_high": ("recent swing high", lambda c: c.s.swing_high),
    "swing_low": ("recent swing low", lambda c: c.s.swing_low),
    "range_high": ("rolling range high (crypto)", lambda c: c.s.range_high),
    "range_low": ("rolling range low (crypto)", lambda c: c.s.range_low),
    "trend": ("higher-timeframe trend: 1 up, -1 down, 0 flat", lambda c: _dir(c.ctx.trend_15m)),
    "market_trend": ("index trend (SPY/QQQ; BTC for crypto): 1 up, -1 down, 0 mixed", lambda c: _dir(c.bias.direction)),
    "minutes_since_open": ("minutes since the session/UTC-day opened",
                           lambda c: (c.ctx.last_bar_close - c.ctx.session.open).total_seconds() / 60),
    "hour_utc": ("hour of day, UTC", lambda c: float(c.ctx.last_bar_close.hour)),
}
BAR_PREV = {"close": "prev_close", "open": "prev_open", "high": "prev_high", "low": "prev_low"}
OPS = ("crosses_above", "crosses_below", "<=", ">=", "<", ">")

_ALIASES = [
    (r"\brsi\s*\(\s*14\s*\)|\brsi\s*14\b|\brsi\b", "rsi14"), (r"\batr\s*\(\s*14\s*\)|\batr\s*14\b|\batr\b", "atr14"),
    (r"\bema\s*\(\s*(9|21|50)\s*\)", r"ema\1"), (r"\bema\s*(9|21|50)\b", r"ema\1"), (r"\b(9|21|50)\s*-?\s*ema\b", r"ema\1"),
    (r"\brelative volume\b", "rvol"), (r"\bprevious close\b", "prev_close"), (r"\bprior day high\b", "prior_high"),
    (r"\bprior day low\b", "prior_low"), (r"\bcrosses above\b", "crosses_above"), (r"\bcrosses below\b", "crosses_below"),
    (r"\bis above\b|\babove\b|\bover\b|\bgreater than\b", ">"), (r"\bis below\b|\bbelow\b|\bunder\b|\bless than\b", "<"),
    (r"\bat least\b", ">="), (r"\bat most\b", "<="), (r"\bthe\b", " "),
]


def normalize_condition(text: str) -> str:
    t = " " + text.strip().lower().rstrip(".,") + " "
    for pat, rep in _ALIASES:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Operand:
    field: str | None
    value: float | None = None
    mult: float = 1.0

    def eval(self, c: StrategyContext, prev: bool = False) -> float | None:
        if self.field is None:
            return self.value
        name = BAR_PREV.get(self.field, self.field) if prev else self.field
        v = FIELDS[name][1](c)
        return None if v is None else v * self.mult

    def __str__(self) -> str:
        if self.field is None:
            return f"{self.value:g}"
        return self.field if self.mult == 1.0 else f"{self.field} * {self.mult:g}"


def parse_operand(tok: str) -> Operand:
    tok = tok.strip()
    m = re.fullmatch(r"([a-z_0-9]+)\s*\*\s*(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)\s*\*\s*([a-z_0-9]+)", tok)
    if m:
        field, mult = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        if field not in FIELDS:
            raise RuleError(f"unknown field '{field}'")
        return Operand(field, mult=float(mult))
    if re.fullmatch(r"-?\d+(?:\.\d+)?%?", tok):
        return Operand(None, float(tok.rstrip("%")) / (100 if tok.endswith("%") else 1))
    if tok in FIELDS:
        return Operand(tok)
    raise RuleError(f"unknown field or value '{tok}' (see `strategy fields` for the list)")


@dataclass(frozen=True)
class Condition:
    left: Operand
    op: str
    right: Operand
    right2: Operand | None = None  # for between
    text: str = ""

    def check(self, c: StrategyContext) -> tuple[bool, str]:
        a = self.left.eval(c)
        if self.op == "between":
            lo, hi = self.right.eval(c), self.right2.eval(c)  # type: ignore[union-attr]
            if None in (a, lo, hi):
                return False, ""
            return lo <= a <= hi, f"{self.left} {a:.4g} between {lo:.4g} and {hi:.4g}"
        b = self.right.eval(c)
        if a is None or b is None:
            return False, ""
        if self.op in ("crosses_above", "crosses_below"):
            ap = self.left.eval(c, prev=True)
            bp = self.right.eval(c, prev=True) if self.right.field in BAR_PREV else b
            if ap is None or bp is None:
                return False, ""
            ok = (ap <= bp and a > b) if self.op == "crosses_above" else (ap >= bp and a < b)
            return ok, f"{self.left} {ap:.4g} -> {a:.4g} {self.op.replace('_', ' ')} {self.right} {b:.4g}"
        ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[self.op]
        return ok, f"{self.left} {a:.4g} {self.op} {self.right} {b:.4g}"


def parse_condition(text: str) -> Condition:
    t = normalize_condition(text)
    m = re.fullmatch(r"(.+?)\s+between\s+(\S+)\s+(?:and\s+)?(\S+)", t)
    if m:
        return Condition(parse_operand(m.group(1)), "between", parse_operand(m.group(2)), parse_operand(m.group(3)), text=t)
    for op in OPS:
        parts = re.split(rf"\s*{re.escape(op)}\s*" if op in ("<", ">", "<=", ">=") else rf"\s+{op}\s+", t, maxsplit=1)
        if len(parts) == 2 and parts[0] and parts[1]:
            left, right = parse_operand(parts[0]), parse_operand(parts[1])
            if op.startswith("crosses") and left.field not in BAR_PREV:
                raise RuleError(f"'{op}' needs a bar field on the left (close, high, low, open): '{t}'")
            if left.field is None and right.field is None:
                raise RuleError(f"condition compares two numbers: '{t}'")
            return Condition(left, op, right, text=t)
    raise RuleError(f"can't read condition '{text}': expected e.g. 'rsi14 < 30' or 'close crosses_above vwap'")


class RuleStrategy(Strategy):
    builtin = False
    defaults: dict[str, Any] = {}

    def __init__(self, spec: dict[str, Any], source: str = "") -> None:
        super().__init__({})
        self.spec = spec
        self.source = source
        self.name = spec["name"]
        self.description = spec.get("description", "") or "user rule strategy"
        self.asset_classes = frozenset(AssetClass(a.upper()) for a in spec.get("assets", ["equity", "crypto"]))
        self.base = float(spec.get("confidence", 0.5))
        self.long = [parse_condition(c) for c in spec.get("long_when") or []]
        self.short = [parse_condition(c) for c in spec.get("short_when") or []]

    def _all(self, conds: list[Condition], sc: StrategyContext) -> list[str] | None:
        why = []
        for c in conds:
            ok, msg = c.check(sc)
            if not ok:
                return None
            why.append(msg)
        return why

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        out = []
        for conds, d in ((self.long, Direction.LONG), (self.short, Direction.SHORT)):
            if conds and (why := self._all(conds, sc)) is not None:
                out.append(self.sig(d, SetupType.CUSTOM, self.base, [f"[{self.name}] " + "; ".join(why)]))
        return out

    def describe(self) -> str:
        parts = [self.description]
        if self.long:
            parts.append("LONG when " + " AND ".join(c.text for c in self.long))
        if self.short:
            parts.append("SHORT when " + " AND ".join(c.text for c in self.short))
        assets = "/".join(sorted(a.value.lower() for a in self.asset_classes))
        return f"{' | '.join(parts)} [{assets}, base confidence {self.base:.2f}]"


def validate_spec(spec: Any, reserved: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """Normalise and validate a rule spec; raises RuleError with every problem found."""
    if not isinstance(spec, dict):
        raise RuleError("a strategy must be a mapping with name / long_when / short_when")
    allowed = {"name", "description", "assets", "confidence", "long_when", "short_when"}
    problems = [f"unknown key '{k}'" for k in spec if k not in allowed]
    name = str(spec.get("name", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if not NAME_RE.match(name):
        problems.append(f"name '{name}' must be 3-41 chars: lowercase letters, digits, underscores")
    elif name in reserved:
        problems.append(f"name '{name}' is a built-in strategy")
    assets = spec.get("assets") or ["equity", "crypto"]
    if isinstance(assets, str):
        assets = [assets]
    for a in assets:
        if str(a).upper() not in AssetClass.__members__:
            problems.append(f"unknown asset class '{a}' (equity, crypto)")
    try:
        conf = float(spec.get("confidence", 0.5))
        if not 0.1 <= conf <= 0.9:
            problems.append("confidence must be between 0.1 and 0.9")
    except (TypeError, ValueError):
        problems.append("confidence must be a number")
        conf = 0.5
    out: dict[str, Any] = {"name": name, "description": str(spec.get("description", "")).strip()[:200],
                           "assets": [str(a).lower() for a in assets], "confidence": conf}
    for key in ("long_when", "short_when"):
        conds = spec.get(key) or []
        if isinstance(conds, str):
            conds = [conds]
        cleaned = []
        for c in conds:
            try:
                cleaned.append(parse_condition(str(c)).text)
            except RuleError as exc:
                problems.append(f"{key}: {exc}")
        if len(cleaned) > 12:
            problems.append(f"{key}: at most 12 conditions")
        out[key] = cleaned
    if not out["long_when"] and not out["short_when"]:
        problems.append("needs at least one condition in long_when or short_when")
    if problems:
        raise RuleError("; ".join(problems))
    return out


def parse_plain_rules(name: str, text: str) -> dict[str, Any]:
    """'long when rsi < 30 and close above ema50; short when ...' -> spec (no LLM needed)."""
    spec: dict[str, Any] = {"name": name, "long_when": [], "short_when": []}
    found = False
    for seg in re.split(r";|\n", text):
        seg = seg.strip()
        m = re.match(r"(?i)(long|buy|go long|short|sell short|go short)\s+(?:when|if)\s+(.+)", seg)
        if not m:
            continue
        found = True
        key = "short_when" if "short" in m.group(1).lower() else "long_when"
        spec[key] += [c.strip() for c in re.split(r"(?i)\s+and\s+", m.group(2)) if c.strip()]
    if not found:
        raise RuleError("write it as 'long when <condition> and <condition>; short when ...'")
    return spec


def to_yaml(spec: dict[str, Any]) -> str:
    clean = {k: v for k, v in spec.items() if v not in (None, [], "")}
    return yaml.safe_dump(clean, sort_keys=False, default_flow_style=False, allow_unicode=True)


def load_rules_dir(path: str | Path, reserved: set[str] | frozenset[str] = frozenset()) -> tuple[dict[str, RuleStrategy], list[str]]:
    out: dict[str, RuleStrategy] = {}
    errors: list[str] = []
    root = Path(path)
    if not root.exists():
        return out, errors
    for f in sorted(root.glob("*.yaml")):
        try:
            spec = validate_spec(yaml.safe_load(f.read_text(encoding="utf-8")), reserved)
            if spec["name"] in out:
                raise RuleError(f"duplicate strategy name '{spec['name']}'")
            out[spec["name"]] = RuleStrategy(spec, source=str(f))
        except (RuleError, yaml.YAMLError) as exc:
            errors.append(f"{f.name}: {exc}")
    return out, errors


def save_rule(spec: dict[str, Any], path: str | Path) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{spec['name']}.yaml"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(to_yaml(spec), encoding="utf-8")
    tmp.replace(target)
    return target


def field_help() -> str:
    return "\n".join(f"  {k:<20} {v[0]}" for k, v in FIELDS.items())
