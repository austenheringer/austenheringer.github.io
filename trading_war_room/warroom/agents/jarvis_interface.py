"""J.A.R.V.I.S. - Lead Strategist & user interface.

Persona: calm, articulate, proactive, dry British wit, addresses the user by
``app.user_title``. Never panics, never uses hype, slang or exclamation marks.

Every reply is composed deterministically first (exact numbers from the
transcript / portfolio) and only then optionally rephrased by the
DialogueEngine, which rejects any rewrite that alters a number.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from warroom.config import AppConfig
from warroom.core.assets import format_price, format_qty
from warroom.models.schemas import (
    AgentRole,
    MarketRegime,
    PlanStatus,
    Stance,
    WarRoomTranscript,
)
from warroom.providers.llm_provider import DialogueEngine

SETUP_NAMES = {
    "VWAP_BOUNCE": "VWAP bounce", "VWAP_REJECTION": "VWAP rejection", "ORB_BREAKOUT": "opening-range breakout",
    "ORB_BREAKDOWN": "opening-range breakdown", "EMA_PULLBACK": "EMA pullback", "RVOL_MOMENTUM": "high-RVOL momentum",
    "RANGE_BREAKOUT": "range breakout", "RANGE_BREAKDOWN": "range breakdown",
}

def setup_label(setup_type, strategy: str = "") -> str:
    """Human name for a setup: built-in pattern names, or the user strategy's name."""
    value = getattr(setup_type, "value", str(setup_type))
    if value == "CUSTOM" and strategy:
        return strategy.replace("_", " ") + " setup"
    return SETUP_NAMES.get(value, value.lower().replace("_", " "))


_WIT = [
    "The market, as ever, remains unmoved by our enthusiasm.",
    "I shall keep the champagne on ice, but not open it.",
    "Discipline is less exciting than heroics, but considerably cheaper.",
    "Patience, I am told, is also a position.",
    "Not every opportunity is an invitation.",
]


class CommandKind(StrEnum):
    STATUS = "status"
    ANALYZE = "analyze"
    WHY = "why"
    TRANSCRIPT = "transcript"
    PAUSE = "pause"
    RESUME = "resume"
    FLATTEN = "flatten"
    HELP = "help"
    STRATEGIES = "strategies"
    SHOW_STRATEGY = "show_strategy"
    ENABLE = "enable"
    DISABLE = "disable"
    NEW_STRATEGY = "new_strategy"
    DELETE_STRATEGY = "delete_strategy"
    STRATEGY_PARAM = "strategy_param"
    FIELDS = "fields"
    STATS = "stats"
    SETTINGS = "settings"
    SET = "set"
    ADD_SYMBOL = "add_symbol"
    REMOVE_SYMBOL = "remove_symbol"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    RESET_SETTINGS = "reset_settings"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    ticker: str | None = None
    raw: str = ""
    args: tuple = ()


_STOPWORDS = {
    "I", "A", "AN", "THE", "WHY", "DID", "YOU", "VETO", "ANALYZE", "ANALYSE", "STATUS", "REPORT", "PLEASE", "ON",
    "ME", "FOR", "OF", "TO", "IS", "IT", "WAS", "AND", "SHOW", "LAST", "TRANSCRIPT", "JARVIS", "SIR", "LOOK", "AT",
    "RUN", "WHAT", "ABOUT", "REJECT", "PASS", "SKIP", "DEBATE", "HAPPENED", "WITH", "THAT", "TRADE", "CHECK",
}
_TICKER_RE = re.compile(r"\$?\b([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")


def _extract_ticker(text: str, prefer_after: str | None = None) -> str | None:
    if prefer_after:
        m = re.search(rf"\b{prefer_after}\s+(?:on\s+|at\s+)?\$?([A-Za-z]{{1,5}}(?:\.[A-Za-z])?)\b", text, re.I)
        if m and m.group(1).upper() not in _STOPWORDS:
            return m.group(1).upper()
    for m in _TICKER_RE.finditer(text):
        tok = m.group(1)
        if tok.upper() in _STOPWORDS:
            continue
        if tok.isupper() or text.strip().startswith("$") or m.group(0).startswith("$"):
            return tok.upper()
    return None


def parse_command(text: str) -> Command:
    raw = text.strip()
    t = raw.lower().strip(" ?!.")
    if not t:
        return Command(CommandKind.UNKNOWN, raw=raw)
    if t in ("help", "commands", "?") or t.startswith("help"):
        return Command(CommandKind.HELP, raw=raw)
    control = _parse_control(raw, t)
    if control is not None:
        return control
    if re.search(r"\b(status|sitrep|how are we|where do we stand|p&l|pnl)\b", t):
        return Command(CommandKind.STATUS, raw=raw)
    if re.search(r"\b(flatten|close (all|everything)|liquidate|get me flat|go flat)\b", t):
        return Command(CommandKind.FLATTEN, raw=raw)
    if re.search(r"\b(resume|unpause|continue trading|start trading|re-?arm)\b", t):
        return Command(CommandKind.RESUME, raw=raw)
    if re.search(r"\b(pause|hold fire|stop trading|stand down)\b", t):
        return Command(CommandKind.PAUSE, raw=raw)
    if t.startswith("why") or re.search(r"\bwhy\b", t):
        return Command(CommandKind.WHY, _extract_ticker(raw, "veto") or _extract_ticker(raw), raw=raw)
    if re.search(r"\btranscript\b", t):
        return Command(CommandKind.TRANSCRIPT, _extract_ticker(raw, "transcript"), raw=raw)
    m = re.search(r"\b(analy[sz]e|look at|check|debate|run)\b", t)
    if m:
        tk = _extract_ticker(raw, m.group(1))
        return Command(CommandKind.ANALYZE if tk else CommandKind.UNKNOWN, tk, raw=raw)
    # a bare ticker means "analyze it"
    if re.fullmatch(r"\$?[A-Za-z]{1,5}", raw) and raw.lstrip("$").upper() not in _STOPWORDS:
        return Command(CommandKind.ANALYZE, raw.lstrip("$").upper(), raw=raw)
    return Command(CommandKind.UNKNOWN, raw=raw)


_SYMBOL_KIND = r"(stock|stocks|ticker|tickers|share|shares|coin|coins|crypto|cryptos|token|tokens)"


def _parse_control(raw: str, t: str) -> Command | None:
    """Strategy and settings management commands."""
    def cmd(kind: CommandKind, *args) -> Command:
        return Command(kind, raw=raw, args=tuple(a.strip() if isinstance(a, str) else a for a in args))

    if t in ("confirm", "yes", "yes please", "do it", "apply", "apply it", "confirmed", "go ahead"):
        return cmd(CommandKind.CONFIRM)
    if t in ("cancel", "no", "never mind", "nevermind", "discard", "abort", "don't"):
        return cmd(CommandKind.CANCEL)
    m = re.match(r"(?i)^\s*(?:new|create|add|make)\s+(?:a\s+)?strategy\b\s*:?\s*(.*)$", raw, re.S)
    if m:
        return cmd(CommandKind.NEW_STRATEGY, m.group(1))
    m = re.match(r"^(?:delete|remove)\s+strategy\s+(.+)$", t)
    if m:
        return cmd(CommandKind.DELETE_STRATEGY, m.group(1))
    if re.match(r"^(?:strategy fields|rule fields|fields|what fields.*)$", t):
        return cmd(CommandKind.FIELDS)
    m = re.match(r"^(?:set|change)\s+strategy\s+(\S+)\s+(\S+)\s+(?:to\s+|=\s*)?(\S+)$", t)
    if m:
        return cmd(CommandKind.STRATEGY_PARAM, m.group(1), m.group(2), m.group(3))
    m = re.match(r"^(?:show|describe|explain)\s+(?:the\s+)?strategy\s+(.+)$", t)
    if m:
        return cmd(CommandKind.SHOW_STRATEGY, m.group(1))
    if re.match(r"^(?:strategies|list strategies|show strategies|my strategies|what strategies.*|which strategies are (on|active).*)$", t):
        return cmd(CommandKind.STRATEGIES)
    if re.match(r"^(?:stats|statistics|performance|results|strategy stats|which strategies (work|are working|make money).*|how are (my )?strategies doing.*)$", t):
        return cmd(CommandKind.STATS)
    m = re.match(r"^(?:enable|turn on|switch on|activate)\s+(?:the\s+)?(.+?)(?:\s+strategy)?$", t)
    if m:
        return cmd(CommandKind.ENABLE, m.group(1))
    m = re.match(r"^(?:disable|turn off|switch off|deactivate)\s+(?:the\s+)?(.+?)(?:\s+strategy)?$", t)
    if m:
        return cmd(CommandKind.DISABLE, m.group(1))
    if re.match(r"^reset (?:all )?(?:settings|config|configuration)$", t):
        return cmd(CommandKind.RESET_SETTINGS)
    m = re.match(r"^(?:settings|show settings|config|show config|configuration)(?:\s+(?:for\s+)?(\w+))?$", t)
    if m:
        return cmd(CommandKind.SETTINGS, m.group(1) or "")
    m = re.match(rf"^(?:add|watch|track)\s+{_SYMBOL_KIND}\s+(.+)$", t)
    if m:
        kind = "coin" if m.group(1).startswith(("coin", "crypto", "token")) else "stock"
        return cmd(CommandKind.ADD_SYMBOL, kind, m.group(2))
    m = re.match(rf"^(?:remove|drop|unwatch|stop watching)\s+{_SYMBOL_KIND}\s+(.+)$", t)
    if m:
        kind = "coin" if m.group(1).startswith(("coin", "crypto", "token")) else "stock"
        return cmd(CommandKind.REMOVE_SYMBOL, kind, m.group(2))
    # match the ORIGINAL text so values keep their case ("set title to Boss")
    m = re.match(r"(?i)^\s*(?:set|change|make|update|put)\s+(?:my\s+|the\s+)?(.+?)\s+(?:to|=|at)\s+(.+?)[\s.!?]*$", raw)
    if m:
        return cmd(CommandKind.SET, m.group(1).lower(), m.group(2))
    return None


HELP_TEXT = (
    "Commands: 'status'; 'analyze TSLA'; 'why AAPL'; 'transcript [TICKER]'; 'pause' / 'resume'; 'flatten'. "
    "Strategies: 'strategies', 'show strategy NAME', 'enable NAME', 'disable NAME', 'stats', 'strategy fields', "
    "'new strategy NAME: long when rsi14 < 30 and close > ema50', 'new strategy from ~/chart.png', "
    "'set strategy NAME PARAM to VALUE', 'delete strategy NAME'. Settings: 'settings [risk]', "
    "'set risk per trade to 0.5%', 'add stock AMZN', 'add coin DOGE', 'remove stock TSLA', 'reset settings'. "
    "Changes to numbers and new strategies wait for 'confirm' (or 'cancel')."
)


@dataclass
class PositionLine:
    ticker: str
    direction: str
    qty: float
    entry: float
    stop: float
    tp1: float
    tp2: float
    last: float | None
    unrealized: float


@dataclass
class StatusReport:
    mode: str
    equity: float
    day_pnl: float
    day_pnl_pct: float
    regime: MarketRegime
    circuit: str
    positions: list[PositionLine] = field(default_factory=list)
    llm: str = "rule_based"
    voice: str = "text-only"
    crypto_regime: MarketRegime | None = None
    market_status: str = "OPEN"


class JarvisContext(Protocol):
    """Implemented by the orchestrator."""

    async def status(self) -> StatusReport: ...
    async def analyze(self, ticker: str) -> WarRoomTranscript | str: ...
    async def latest_transcript(self, ticker: str | None) -> WarRoomTranscript | None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    async def flatten(self) -> list[str]: ...


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _money(x: float) -> str:
    return f"${abs(x):,.2f}" if x >= 0 else f"-${abs(x):,.2f}"


class JarvisInterface:
    def __init__(self, cfg: AppConfig, dialogue: DialogueEngine) -> None:
        self.title = cfg.app.user_title
        self.dialogue = dialogue

    def _wit(self, key: str) -> str:
        return _WIT[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(_WIT)]

    # ------------------------------------------------------------------ briefings

    def briefing_draft(self, t: WarRoomTranscript) -> str:
        p = t.plan
        setup = setup_label(p.setup_type, p.strategy)
        side = p.direction.value.lower()
        disagreements = [m for m in t.messages if m.round_number == 2 and m.stance in (Stance.CHALLENGE, Stance.CAUTION)]
        dispute = ""
        if disagreements:
            first = disagreements[0]
            dispute = f" {first.sender.value.title()} challenged it: {first.data_payload.get('point', 'see transcript')}."
        if p.status is PlanStatus.APPROVED:
            px = lambda v: format_price(t.ticker, v)  # noqa: E731
            return (f"{self.title}, {t.ticker} presents {_a(setup)} {side}.{dispute} Risk approved: entry {px(p.entry)}, "
                    f"stop {px(p.stop_loss)}, targets {px(p.tp1)} and {px(p.tp2)}. "
                    f"{format_qty(t.ticker, p.position_size)}, risking {_money(p.dollar_risk)}.")
        if p.status is PlanStatus.WATCHLIST:
            levels = (f" Levels for reference: entry {format_price(t.ticker, p.entry)}, stop "
                      f"{format_price(t.ticker, p.stop_loss)}, TP1 {format_price(t.ticker, p.tp1)}."
                      if p.entry and p.stop_loss and p.tp1 else "")
            return (f"{self.title}, {t.ticker} {side} goes on the watchlist rather than the book: {p.summary}.{dispute}"
                    f"{levels} {self._wit(t.debate_id)}")
        reason = p.veto_reasons[0] if p.veto_reasons else "risk limits"
        more = f" and {len(p.veto_reasons) - 1} further objection(s)" if len(p.veto_reasons) > 1 else ""
        return (f"{self.title}, I have declined {_a(t.ticker)} {setup} {side}. Risk vetoed it: {reason}{more}. "
                f"No position taken. {self._wit(t.debate_id)}")

    async def briefing(self, t: WarRoomTranscript) -> str:
        return await self.dialogue.render(AgentRole.JARVIS, self.briefing_draft(t), max_sentences=4)

    # ------------------------------------------------------------------ command responses

    def status_draft(self, s: StatusReport) -> str:
        pos = "no open positions" if not s.positions else "; ".join(
            f"{p.ticker} {p.direction.lower()} {format_qty(p.ticker, p.qty)} @ {format_price(p.ticker, p.entry)} "
            f"(stop {format_price(p.ticker, p.stop)}, P&L {_money(p.unrealized)})"
            for p in s.positions)
        return (f"{self.title}, {s.mode} mode. Equity {_money(s.equity)}, day P&L {_money(s.day_pnl)} "
                f"({s.day_pnl_pct:+.2%}). {pos[:1].upper() + pos[1:]}. Market regime {s.regime.value.replace('_', '-').lower()}"
                + (f", crypto regime {s.crypto_regime.value.replace('_', '-').lower()}" if s.crypto_regime else "")
                + f", circuit breakers {s.circuit.lower()}.")

    def why_draft(self, t: WarRoomTranscript | None, ticker: str | None) -> str:
        if t is None:
            return f"{self.title}, I have no War Room record for {ticker or 'that'} today."
        p = t.plan
        if p.status is PlanStatus.VETOED:
            reasons = "; ".join(p.veto_reasons)
            news = next((m for m in t.messages if m.sender is AgentRole.NEWS and m.round_number == 1), None)
            news_bit = f" The news desk's view: {news.message}" if news else ""
            return f"{self.title}, {t.ticker} {p.direction.value.lower()} was vetoed because: {reasons}.{news_bit}"
        if p.status is PlanStatus.WATCHLIST:
            return f"{self.title}, {t.ticker} was not vetoed; it was parked on the watchlist: {p.summary}."
        return (f"{self.title}, {t.ticker} was approved, not vetoed: R:R {p.reward_risk:.2f} to TP1, "
                f"{format_qty(t.ticker, p.position_size)} risking {_money(p.dollar_risk)}.")

    @staticmethod
    def format_transcript(t: WarRoomTranscript) -> str:
        lines = [f"War Room {t.debate_id}: {t.ticker} {t.direction.value} -> {t.plan.status.value}"]
        for m in t.messages:
            tgt = "" if m.recipient is AgentRole.ALL else f" -> {m.recipient.value}"
            lines.append(f"  R{m.round_number} [{m.sender.value}{tgt}] ({m.stance.value}) {m.message}")
        if t.briefing:
            lines.append(f"  J.A.R.V.I.S.: {t.briefing}")
        return "\n".join(lines)

    async def handle(self, text: str, ctx: JarvisContext) -> str:
        cmd = parse_command(text)
        prefix = ""
        control = getattr(ctx, "control", None)
        if cmd.kind is CommandKind.UNKNOWN and control is not None and text.strip():
            translated = await control.translate(text)
            if translated:
                cmd2 = parse_command(translated)
                if cmd2.kind is not CommandKind.UNKNOWN:
                    cmd, prefix = cmd2, f"(Taking that as: {translated}) "
        reply = await self._dispatch(cmd, ctx, control)
        return prefix + reply

    async def _dispatch(self, cmd: Command, ctx: JarvisContext, control) -> str:
        k = cmd.kind
        if control is not None:
            a = cmd.args
            if k is CommandKind.STRATEGIES:
                return control.strategies()
            if k is CommandKind.SHOW_STRATEGY:
                return control.show_strategy(a[0])
            if k is CommandKind.ENABLE:
                return await control.set_enabled(a[0], True)
            if k is CommandKind.DISABLE:
                return await control.set_enabled(a[0], False)
            if k is CommandKind.NEW_STRATEGY:
                return await control.new_strategy(a[0]) if a[0] else control.fields()
            if k is CommandKind.DELETE_STRATEGY:
                return await control.delete_strategy(a[0])
            if k is CommandKind.STRATEGY_PARAM:
                return control.set_param(*a)
            if k is CommandKind.FIELDS:
                return control.fields()
            if k is CommandKind.STATS:
                return await control.stats()
            if k is CommandKind.SETTINGS:
                return control.show_settings(a[0] or None)
            if k is CommandKind.SET:
                return control.propose_setting(a[0], a[1])
            if k in (CommandKind.ADD_SYMBOL, CommandKind.REMOVE_SYMBOL):
                syms = [x for x in re.split(r"[,\s]+(?:and\s+)?", a[1]) if x and x != "and"]
                return await control.modify_symbols(a[0], "add" if k is CommandKind.ADD_SYMBOL else "remove", syms)
            if k is CommandKind.CONFIRM:
                return await control.confirm()
            if k is CommandKind.CANCEL:
                return control.cancel()
            if k is CommandKind.RESET_SETTINGS:
                return control.reset_settings()
        elif k not in (CommandKind.STATUS, CommandKind.ANALYZE, CommandKind.WHY, CommandKind.TRANSCRIPT,
                       CommandKind.PAUSE, CommandKind.RESUME, CommandKind.FLATTEN, CommandKind.HELP, CommandKind.UNKNOWN):
            return f"{self.title}, strategy and settings changes aren't available in this mode."
        if k is CommandKind.HELP:
            return f"{self.title}, {HELP_TEXT}"
        if k is CommandKind.STATUS:
            return await self.dialogue.render(AgentRole.JARVIS, self.status_draft(await ctx.status()), max_sentences=5)
        if k is CommandKind.ANALYZE and cmd.ticker:
            result = await ctx.analyze(cmd.ticker)
            if isinstance(result, str):
                return result
            return result.briefing or self.briefing_draft(result)
        if k is CommandKind.WHY:
            t = await ctx.latest_transcript(cmd.ticker)
            return await self.dialogue.render(AgentRole.JARVIS, self.why_draft(t, cmd.ticker), max_sentences=5)
        if k is CommandKind.TRANSCRIPT:
            t = await ctx.latest_transcript(cmd.ticker)
            return self.format_transcript(t) if t else f"{self.title}, there is no debate on record yet."
        if k is CommandKind.PAUSE:
            ctx.pause()
            return (f"Very good, {self.title}. New entries are paused; open positions remain managed "
                    "by their stops and targets.")
        if k is CommandKind.RESUME:
            ctx.resume()
            return f"Resuming, {self.title}. The War Room is open for business, circuit breakers permitting."
        if k is CommandKind.FLATTEN:
            closed = await ctx.flatten()
            if not closed:
                return f"{self.title}, we are already flat. Nothing to close."
            return f"Done, {self.title}. Closed at market: " + "; ".join(closed) + "."
        return f"I'm afraid I didn't catch that, {self.title}. {HELP_TEXT}"
