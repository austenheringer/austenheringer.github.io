"""J.A.R.V.I.S. control desk: change strategies and settings by talking.

Safety model
------------
* Every change goes through ``SettingsManager.plan`` = the same strict config validation
  as config.yaml. Chat cannot create an invalid or unsafe setting, and real-money trading
  cannot be switched on.
* Numeric / risk changes, new strategies and resets are PROPOSED first (before -> after)
  and applied only on ``confirm`` (10-minute expiry). Switching a strategy on/off and
  adding/removing symbols apply immediately (low risk, easily undone).
* New strategies (typed rules, rule files or screenshots) become declarative rule specs,
  validated against a whitelist of fields and operators. LLM output is treated as untrusted
  text: it is parsed and validated, never executed.
* Applied changes are written to data/config_overrides.yaml and the settings_changes table.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from warroom.backtest.metrics import format_strategy_breakdown, strategy_breakdown
from warroom.core.assets import CRYPTO_SYMBOL_RE
from warroom.settings_manager import Change, SettingError, SettingsManager, fmt_value
from warroom.strategies.builtin import BUILTINS
from warroom.strategies.rules import (
    FIELDS,
    RuleError,
    field_help,
    parse_plain_rules,
    save_rule,
    to_yaml,
    validate_spec,
)

if TYPE_CHECKING:  # pragma: no cover
    from warroom.orchestrator import Orchestrator

log = logging.getLogger(__name__)

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
PENDING_TTL = timedelta(minutes=10)

STRATEGY_PROMPT = """You convert a day-trading strategy (described in text and/or shown in a chart screenshot)
into a strict YAML rule file. Output ONLY the YAML - no prose, no code fences.

Format:
name: snake_case_name
description: one sentence (mention any rule you had to approximate)
assets: [equity, crypto]        # or just one of them
confidence: 0.5                 # 0.1-0.9, lower for looser rules
long_when:                      # ALL must be true to go long (omit or [] if long not described)
  - <condition>
short_when:                     # ALL must be true to go short (omit or [] if not described)
  - <condition>

A condition is "<operand> <op> <operand>" or "<field> between <a> <b>".
Operands: a field, a number, or "field * number". Ops: < <= > >= crosses_above crosses_below
crosses_above / crosses_below need close, high, low or open on the left.
Fields (use ONLY these; never invent fields or indicators):
{fields}

If the strategy uses something not available (e.g. MACD, Bollinger Bands, a specific
candlestick pattern), approximate it with the closest fields above and say so in description.
Stops, targets and position sizing are handled by the Risk agent: do not include them.
{name_hint}"""

COMMAND_PROMPT = """Translate the user's request into exactly ONE command from this list, or reply UNKNOWN.
Reply with the command only.

status | analyze <TICKER> | why <TICKER> | transcript [TICKER] | pause | resume | flatten
strategies | show strategy <name> | enable <name> | disable <name> | delete strategy <name>
set strategy <name> <param> to <value> | strategy fields | stats
settings [section] | set <setting> to <value> | add stock <TICKER> | remove stock <TICKER>
add coin <SYMBOL> | remove coin <SYMBOL> | reset settings | confirm | cancel
new strategy <name>: long when <cond> and <cond>; short when <cond>

Setting names include: risk per trade, daily loss limit, max positions, min reward risk,
stop atr, first target, second target, spread limit, cooldown, rvol threshold, entry cutoff,
min confidence, blackout before, blackout after, slippage, title, max hold, weekend risk,
crypto positions, taker fee, maker fee. Strategy names: {strategies}.

User: {text}"""


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:ya?ml)?\s*(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()


@dataclass
class Pending:
    kind: str            # "setting" | "strategy" | "reset"
    payload: Any
    summary: str
    created: datetime


class ControlDesk:
    def __init__(self, orch: "Orchestrator", settings: SettingsManager | None = None) -> None:
        self.o = orch
        self.settings = settings or SettingsManager()
        self.pending: Pending | None = None

    @property
    def title(self) -> str:
        return self.o.cfg.app.user_title

    # ------------------------------------------------------------------ helpers

    def _resolve_strategy(self, name: str) -> str | None:
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        names = list(self.o.technical.registry.available)
        if key in names:
            return key
        matches = [n for n in names if n.startswith(key) or key in n]
        return matches[0] if len(matches) == 1 else None

    async def _apply(self, change: Change, source: str = "chat") -> str:
        self.settings.persist(change)
        try:
            await self.o.db.log_setting_change(self.o.clock.now(), change.path, change.old, change.new, source)
        except Exception:  # noqa: BLE001 - the audit log must not block the change
            log.exception("failed to audit setting change")
        if not change.restart:
            self.o.apply_config(change.config)
        else:
            self.o.cfg_pending_restart = True
        return change.describe()

    def _propose(self, kind: str, payload: Any, summary: str) -> str:
        self.pending = Pending(kind, payload, summary, self.o.clock.now())
        return f"{self.title}, proposed: {summary}. Say 'confirm' to apply or 'cancel' to discard."

    # ------------------------------------------------------------------ strategies

    def strategies(self) -> str:
        reg = self.o.technical.registry
        lines = [f"{self.title}, strategies ({len(reg.enabled)} on):"]
        for name, strat in reg.available.items():
            on = "ON " if name in reg.enabled else "off"
            kind = "built-in" if strat.builtin else "yours"
            lines.append(f"  [{on}] {name} ({kind}): {strat.describe()}")
        if reg.errors:
            lines.append("  Problems: " + "; ".join(reg.errors))
        lines.append("Say 'enable <name>', 'disable <name>', 'show strategy <name>', 'stats', or 'new strategy ...'.")
        return "\n".join(lines)

    def show_strategy(self, name: str) -> str:
        n = self._resolve_strategy(name)
        if n is None:
            return f"I don't have a strategy called '{name}', {self.title}. Say 'strategies' for the list."
        s = self.o.technical.registry.available[n]
        state = "on" if n in self.o.technical.registry.enabled else "off"
        extra = f"\nRule file: {getattr(s, 'source', '')}\n{to_yaml(s.spec)}" if not s.builtin else ""
        return f"{n} ({state}): {s.describe()}{extra}"

    async def set_enabled(self, name: str, on: bool) -> str:
        n = self._resolve_strategy(name)
        if n is None:
            return f"I don't have a strategy called '{name}', {self.title}. Say 'strategies' for the list."
        enabled = list(self.o.cfg.strategies.enabled)
        if on and n not in enabled:
            enabled.append(n)
        elif not on and n in enabled:
            enabled.remove(n)
        else:
            return f"{n} is already {'on' if on else 'off'}, {self.title}."
        if not enabled:
            return f"{self.title}, that would switch off every strategy; say 'pause' instead to stop new trades."
        change = self.settings.plan(self.o.cfg, "strategies.enabled", enabled)
        await self._apply(change)
        return f"Done, {self.title}. {n} is now {'on' if on else 'off'}."

    def set_param(self, name: str, param: str, raw: str) -> str:
        n = self._resolve_strategy(name)
        if n is None:
            return f"I don't have a strategy called '{name}', {self.title}."
        strat = self.o.technical.registry.available[n]
        if not strat.builtin:
            return (f"{n} is a rule strategy; change it with a new rule set, e.g. "
                    f"'new strategy {n}: long when ...' (it will replace the file after you confirm).")
        param = param.strip().lower().replace(" ", "_")
        if param not in strat.defaults:
            return f"{n} has no parameter '{param}'. Parameters: {', '.join(strat.defaults) or 'none'}."
        try:
            value = float(raw.strip().rstrip("%x"))
        except ValueError:
            return f"'{raw}' isn't a number, {self.title}."
        params = {k: dict(v) for k, v in self.o.cfg.strategies.params.items()}
        params.setdefault(n, {})[param] = value
        try:
            change = self.settings.plan(self.o.cfg, "strategies.params", params)
        except SettingError as exc:
            return f"{self.title}, {exc}"
        old = strat.params[param]
        return self._propose("setting", change, f"{n}.{param}: {old:g} -> {value:g}")

    def fields(self) -> str:
        return ("Fields you can use in strategy rules:\n" + field_help() +
                "\nOperators: < <= > >= crosses_above crosses_below, and 'field between a b'. "
                "Example: new strategy dip_buyer: long when rsi14 < 30 and close > ema50")

    async def new_strategy(self, text: str) -> str:
        """'new strategy NAME: long when ...', 'new strategy from <image or .yaml path>',
        or a plain-English description (needs an LLM)."""
        text = text.strip()
        m = re.match(r"(?i)from\s+(.+)$", text)
        if m:
            return await self._strategy_from_file(m.group(1).strip().strip("'\""))
        name, rules = None, text
        m = re.match(r"^([A-Za-z][\w\- ]{1,40}?)\s*:\s*(.+)$", text, re.S)
        if m:
            name, rules = m.group(1).strip().lower().replace(" ", "_").replace("-", "_"), m.group(2)
        spec, err = None, None
        if name:
            try:
                spec = validate_spec(parse_plain_rules(name, rules), reserved=set(BUILTINS))
            except RuleError as exc:
                err = str(exc)
        if spec is None:
            spec, llm_err = await self._llm_strategy(rules, name_hint=name)
            if spec is None:
                why = err or llm_err
                return (f"{self.title}, I couldn't turn that into rules: {why}\n"
                        "Write it like: new strategy dip_buyer: long when rsi14 < 30 and close > ema50; "
                        "short when rsi14 > 70 and close < ema50   (say 'strategy fields' for the list)")
        return self._propose_strategy(spec)

    async def _strategy_from_file(self, raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            return f"{self.title}, I can't find a file at {path}."
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                spec = validate_spec(yaml.safe_load(path.read_text(encoding="utf-8")), reserved=set(BUILTINS))
            except (RuleError, yaml.YAMLError) as exc:
                return f"{self.title}, that rule file isn't valid: {exc}"
            return self._propose_strategy(spec)
        if suffix not in IMAGE_TYPES:
            return f"{self.title}, I can read .png/.jpg/.gif/.webp screenshots or .yaml rule files, not {suffix or 'that'}."
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return f"{self.title}, that image is over 5 MB; please send a smaller screenshot."
        if not self.o.dialogue.can_see:
            return (f"{self.title}, reading screenshots needs an AI model with vision: set ANTHROPIC_API_KEY (or "
                    "OPENAI_API_KEY) in .env and providers.llm in config.yaml. You can also type the rules: "
                    "new strategy NAME: long when ...")
        data = await asyncio.to_thread(path.read_bytes)
        spec, err = await self._llm_strategy("Read the strategy rules shown in this screenshot.",
                                             image=(data, IMAGE_TYPES[suffix]))
        if spec is None:
            return f"{self.title}, I couldn't extract usable rules from that screenshot: {err}"
        return self._propose_strategy(spec)

    async def _llm_strategy(self, text: str, name_hint: str | None = None,
                            image: tuple[bytes, str] | None = None) -> tuple[dict | None, str]:
        if not self.o.dialogue.can_ask:
            return None, "plain-English strategies need an AI model (set ANTHROPIC_API_KEY or OPENAI_API_KEY)"
        hint = f"Use this name: {name_hint}" if name_hint else "Choose a short descriptive snake_case name."
        system = STRATEGY_PROMPT.format(fields="\n".join(f"  {k}: {v[0]}" for k, v in FIELDS.items()), name_hint=hint)
        answer = await self.o.dialogue.ask(system, text, max_tokens=900, image=image)
        if not answer:
            return None, "the AI model did not answer"
        try:
            raw = yaml.safe_load(_strip_fences(answer))
            return validate_spec(raw, reserved=set(BUILTINS)), ""
        except (RuleError, yaml.YAMLError) as exc:
            return None, f"the AI's rules failed validation ({exc})"

    def _propose_strategy(self, spec: dict) -> str:
        exists = spec["name"] in self.o.technical.registry.available
        verb = "replace" if exists else "save"
        summary = (f"{verb} strategy '{spec['name']}' and switch it on - "
                   + (f"LONG when {' AND '.join(spec['long_when'])}" if spec["long_when"] else "")
                   + ("; " if spec["long_when"] and spec["short_when"] else "")
                   + (f"SHORT when {' AND '.join(spec['short_when'])}" if spec["short_when"] else "")
                   + f" [{'/'.join(spec['assets'])}]")
        return self._propose("strategy", spec, summary)

    async def delete_strategy(self, name: str) -> str:
        n = self._resolve_strategy(name)
        if n is None:
            return f"I don't have a strategy called '{name}', {self.title}."
        strat = self.o.technical.registry.available[n]
        if strat.builtin:
            return f"{n} is built in and can't be deleted; say 'disable {n}' to switch it off."
        if n in self.o.cfg.strategies.enabled:
            await self.set_enabled(n, False)
        Path(strat.source).unlink(missing_ok=True)
        self.o.reload_strategies()
        return f"Deleted strategy {n}, {self.title}."

    # ------------------------------------------------------------------ settings

    def show_settings(self, section: str | None) -> str:
        try:
            return self.settings.describe(self.o.cfg, section)
        except SettingError as exc:
            return f"{self.title}, {exc}"

    def propose_setting(self, name: str, raw: str) -> str:
        try:
            change = self.settings.plan_text(self.o.cfg, name, raw)
        except SettingError as exc:
            return f"{self.title}, {exc}"
        if change.old == change.new:
            return f"{change.path} is already {fmt_value(change.path, change.new)}, {self.title}."
        return self._propose("setting", change, change.describe())

    async def modify_symbols(self, kind: str, op: str, symbols: list[str]) -> str:
        crypto = kind == "coin"
        path = "crypto.symbols" if crypto else "app.tickers"
        current = list(self.o.cfg.crypto.symbols if crypto else self.o.cfg.app.tickers)
        changed = []
        for raw in symbols:
            sym = raw.strip().upper().lstrip("$")
            if crypto and not CRYPTO_SYMBOL_RE.match(sym):
                sym = f"{sym}-USD"
            if not crypto and CRYPTO_SYMBOL_RE.match(sym):
                return f"{sym} looks like a crypto pair, {self.title}; say 'add coin {sym.split('-')[0]}'."
            if op == "add" and sym not in current:
                current.append(sym)
                changed.append(sym)
            elif op == "remove" and sym in current:
                current.remove(sym)
                changed.append(sym)
        if not changed:
            return f"Nothing to change, {self.title}: {'already watching' if op == 'add' else 'not watching'} that."
        try:
            change = self.settings.plan(self.o.cfg, path, current)
        except SettingError as exc:
            return f"{self.title}, {exc}"
        if crypto and op == "add" and getattr(self.o.market, "crypto", None) is None:
            change.restart = True
        await self._apply(change)
        note = " The crypto desk starts after a restart." if change.restart else ""
        return f"Done, {self.title}. {'Added' if op == 'add' else 'Removed'} {', '.join(changed)}.{note}"

    # ------------------------------------------------------------------ confirm / cancel / reset

    async def confirm(self) -> str:
        p = self.pending
        if p is None:
            return f"There's nothing waiting for confirmation, {self.title}."
        self.pending = None
        if self.o.clock.now() - p.created > PENDING_TTL:
            return f"That proposal expired, {self.title}; please ask again."
        if p.kind == "setting":
            try:
                # re-plan against the CURRENT config so a stale proposal can't overwrite newer changes
                fresh = self.settings.plan(self.o.cfg, p.payload.path, p.payload.new)
            except SettingError as exc:
                return f"{self.title}, that change is no longer valid: {exc}"
            done = await self._apply(fresh)
            return f"Applied, {self.title}: {done}."
        if p.kind == "strategy":
            spec = p.payload
            path = await asyncio.to_thread(save_rule, spec, self.o.cfg.strategies.rules_dir)
            self.o.reload_strategies()
            if spec["name"] not in self.o.cfg.strategies.enabled:
                await self.set_enabled(spec["name"], True)
            else:
                self.o.reload_strategies()
            return (f"Saved {path} and switched '{spec['name']}' on, {self.title}. Every signal it finds still goes "
                    "through the War Room and Risk's veto. Run a replay to see how it would have done.")
        if p.kind == "reset":
            self.settings.reset()
            return f"Overrides removed, {self.title}. Restart to return fully to config.yaml."
        return "Unknown proposal."

    def cancel(self) -> str:
        had = self.pending is not None
        self.pending = None
        return f"Discarded, {self.title}." if had else f"Nothing to cancel, {self.title}."

    def reset_settings(self) -> str:
        if not self.settings.load():
            return f"You have no chat-made changes, {self.title}; everything comes from config.yaml."
        return self._propose("reset", None, "remove all chat-made setting changes (back to config.yaml)")

    # ------------------------------------------------------------------ stats

    async def stats(self) -> str:
        closed = await self.o.db.load_all_closed_positions()
        closed += [p for p in self.o.broker.closed_today if p.id not in {c.id for c in closed}]
        if not closed:
            return f"No closed trades yet, {self.title}. Run a replay (--replay) or let paper trading run."
        return f"{self.title}, results by strategy (net of fees):\n" + format_strategy_breakdown(strategy_breakdown(closed))

    # ------------------------------------------------------------------ free text

    async def translate(self, text: str) -> str | None:
        """Use the LLM (if any) to map free text onto ONE known command. The result is parsed
        by the normal command parser; nothing is executed from the model's text directly."""
        if not self.o.dialogue.can_ask:
            return None
        names = ", ".join(self.o.technical.registry.available) or "none"
        answer = await self.o.dialogue.ask("You are a strict command translator.",
                                           COMMAND_PROMPT.format(strategies=names, text=text[:500]), max_tokens=60)
        if not answer:
            return None
        line = answer.strip().splitlines()[0].strip().strip("`")
        return None if line.upper().startswith("UNKNOWN") else line
