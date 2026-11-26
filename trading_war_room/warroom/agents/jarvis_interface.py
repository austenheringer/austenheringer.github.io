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
}

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
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    ticker: str | None = None
    raw: str = ""


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


HELP_TEXT = (
    "Commands: 'status' (equity, P&L, positions, regime, breakers); 'analyze TSLA' (convene the War Room now); "
    "'why AAPL' (explain the last verdict); 'transcript [TICKER]' (full last debate); 'pause' / 'resume' "
    "(new entries); 'flatten' (close all paper positions at market)."
)


@dataclass
class PositionLine:
    ticker: str
    direction: str
    qty: int
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
        setup = SETUP_NAMES.get(p.setup_type.value, p.setup_type.value.lower())
        side = p.direction.value.lower()
        disagreements = [m for m in t.messages if m.round_number == 2 and m.stance in (Stance.CHALLENGE, Stance.CAUTION)]
        dispute = ""
        if disagreements:
            first = disagreements[0]
            dispute = f" {first.sender.value.title()} challenged it: {first.data_payload.get('point', 'see transcript')}."
        if p.status is PlanStatus.APPROVED:
            return (f"{self.title}, {t.ticker} presents {_a(setup)} {side}.{dispute} Risk approved: entry {p.entry:.2f}, "
                    f"stop {p.stop_loss:.2f}, targets {p.tp1:.2f} and {p.tp2:.2f}. "
                    f"{p.position_size} shares, risking {_money(p.dollar_risk)}.")
        if p.status is PlanStatus.WATCHLIST:
            levels = (f" Levels for reference: entry {p.entry:.2f}, stop {p.stop_loss:.2f}, TP1 {p.tp1:.2f}."
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
            f"{p.ticker} {p.direction.lower()} {p.qty} @ {p.entry:.2f} (stop {p.stop:.2f}, P&L {_money(p.unrealized)})"
            for p in s.positions)
        return (f"{self.title}, {s.mode} mode. Equity {_money(s.equity)}, day P&L {_money(s.day_pnl)} "
                f"({s.day_pnl_pct:+.2%}). {pos[:1].upper() + pos[1:]}. Market regime {s.regime.value.replace('_', '-').lower()}, "
                f"circuit breakers {s.circuit.lower()}.")

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
                f"{p.position_size} shares risking {_money(p.dollar_risk)}.")

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
        k = cmd.kind
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
