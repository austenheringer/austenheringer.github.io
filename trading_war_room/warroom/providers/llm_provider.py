"""LLM dialogue layer with a deterministic, always-available fallback.

Design: agents ALWAYS compose their message deterministically first (the
"draft"). An LLM, when configured, may only *rephrase* that draft in the
speaker's voice. :class:`DialogueEngine` then checks the rewrite preserved
every number in the draft (prices, share counts, dollar risk). If the model
dropped, changed or invented a number, the draft is used instead. An LLM can
therefore make the War Room read better; it can never change a stop, a size
or a verdict.

The rule-based path is used when no API key is set, the SDK is missing, a
call errors or times out, or ``max_daily_llm_calls`` is reached. The daily
counter is keyed by the ET trading date from the injected Clock and can be
persisted (see storage) so a restart does not reset the budget.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import date

from warroom.core.clock import Clock
from warroom.models.schemas import AgentRole
from warroom.providers.base import LLMProvider, ProviderUnavailable

log = logging.getLogger(__name__)

PERSONAS: dict[AgentRole, str] = {
    AgentRole.TECHNICAL: (
        "You are the Technical analyst in a trading war room. Terse, confident, chart-focused. "
        "You argue from price, VWAP, EMAs, RSI, ATR and relative volume."
    ),
    AgentRole.NEWS: (
        "You are the News & Macro analyst in a trading war room. Skeptical, evidence-driven, "
        "you cite catalysts and scheduled macro events and challenge setups that ignore them."
    ),
    AgentRole.RISK: (
        "You are the Risk manager in a trading war room. Unemotional, precise, protective of capital. "
        "You hold an absolute veto and speak in terms of R, stops, size and limits."
    ),
    AgentRole.JARVIS: (
        "You are J.A.R.V.I.S., a calm, articulate British AI aide with a slightly dry wit. "
        "Address the user as '{title}'. Never use hype, slang or exclamation marks. Never panic."
    ),
}

_REWRITE_INSTRUCTIONS = (
    "Rewrite the DRAFT below in your voice. Rules: keep every number exactly as written "
    "(same digits, same decimals); do not add any new numbers, facts, tickers or recommendations; "
    "keep the same verdict; at most {max_sentences} sentences; plain text only.\n\nDRAFT:\n{draft}"
)

_NUM_RE = re.compile(r"(?<![A-Za-z])-?\$?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[str]:
    """Canonical numeric tokens ('$1,234.50' -> '1234.50', '2.0%' -> '2.0')."""
    out = []
    for m in _NUM_RE.findall(text):
        tok = m.replace("$", "").replace(",", "").rstrip("%")
        if tok.startswith("-"):
            tok = tok[1:]
        if tok:
            out.append(tok)
    return out


def numbers_preserved(draft: str, rewrite: str) -> bool:
    """Every number in the draft must appear in the rewrite, and the rewrite
    may not introduce numbers absent from the draft."""
    d, r = extract_numbers(draft), extract_numbers(rewrite)
    return set(d) <= set(r) and set(r) <= set(d)


class RuleBasedDialogueProvider(LLMProvider):
    """Deterministic fallback. Returns the draft (lightly normalised)."""

    name = "rule_based"

    async def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        marker = "DRAFT:\n"
        text = prompt.split(marker, 1)[1] if marker in prompt else prompt
        return self.polish(text)

    @staticmethod
    def polish(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".?!":
            text += "."
        return text


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailable("anthropic SDK not installed: pip install anthropic") from exc
        if not api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)

    async def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        resp = await self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content).strip()

    async def aclose(self) -> None:
        await self._client.close()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ProviderUnavailable("openai SDK not installed: pip install openai") from exc
        if not api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        self.model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=2)

    async def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    async def aclose(self) -> None:
        await self._client.close()


class DialogueEngine:
    def __init__(
        self,
        clock: Clock,
        primary: LLMProvider | None = None,
        *,
        max_daily_calls: int = 200,
        timeout: float = 20.0,
        user_title: str = "Sir",
        initial_usage: tuple[date, int] | None = None,
        on_call: Callable[[date, int], Awaitable[None]] | None = None,
    ) -> None:
        self.clock = clock
        self.primary = primary
        self.fallback = RuleBasedDialogueProvider()
        self.max_daily_calls = max_daily_calls
        self.timeout = timeout
        self.user_title = user_title
        self._on_call = on_call
        self._usage_day, self._calls = initial_usage or (clock.now_et().date(), 0)
        self.failures = 0
        self.rejected_rewrites = 0

    @property
    def active_provider(self) -> str:
        return self.primary.name if self.primary and not self.budget_exhausted else self.fallback.name

    @property
    def calls_today(self) -> int:
        self._roll_day()
        return self._calls

    @property
    def budget_exhausted(self) -> bool:
        return self.calls_today >= self.max_daily_calls

    def _roll_day(self) -> None:
        today = self.clock.now_et().date()
        if today != self._usage_day:
            self._usage_day, self._calls = today, 0

    def system_prompt(self, role: AgentRole) -> str:
        return PERSONAS.get(role, PERSONAS[AgentRole.JARVIS]).replace("{title}", self.user_title)

    async def render(self, role: AgentRole, draft: str, max_sentences: int = 4) -> str:
        """Return a persona-styled version of ``draft``; never raises."""
        draft = self.fallback.polish(draft)
        if self.primary is None or self.budget_exhausted:
            return draft
        self._calls += 1  # count attempts, not successes: a failing API still costs quota
        if self._on_call:
            try:
                await self._on_call(self._usage_day, self._calls)
            except Exception:  # noqa: BLE001 - persistence trouble must not break dialogue
                log.exception("failed to persist LLM usage")
        prompt = _REWRITE_INSTRUCTIONS.format(draft=draft, max_sentences=max_sentences)
        try:
            out = await asyncio.wait_for(self.primary.complete(self.system_prompt(role), prompt), self.timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - timeouts, HTTP errors, SDK errors -> fallback
            self.failures += 1
            log.warning("LLM %s failed (%s); using rule-based dialogue", self.primary.name, type(exc).__name__)
            return draft
        out = self.fallback.polish(out)
        if not out or len(out) > max(400, len(draft) * 3) or not numbers_preserved(draft, out):
            self.rejected_rewrites += 1
            log.info("LLM rewrite rejected (numbers changed or malformed); using draft")
            return draft
        return out

    async def aclose(self) -> None:
        if self.primary:
            await self.primary.aclose()
