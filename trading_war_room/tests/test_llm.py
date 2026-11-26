import asyncio
from datetime import timedelta

from tests.conftest import et
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import AgentRole
from warroom.providers.base import LLMProvider
from warroom.providers.llm_provider import DialogueEngine, extract_numbers, numbers_preserved

DRAFT = "Long NVDA at $140.25, stop 138.10, TP1 144.55, size 46 shares, risking $98.90"


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def complete(self, system, prompt, max_tokens=300):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply == "slow":
            await asyncio.sleep(5)
        return self.reply


GOOD = "Sir, long NVDA from $140.25 with the stop at 138.10, TP1 144.55, 46 shares and $98.90 at risk."


def test_number_extraction_and_guard():
    assert extract_numbers("TP1 at $1,234.50 (2.0%)") == ["1234.50", "2.0"]
    assert numbers_preserved(DRAFT, GOOD)
    assert not numbers_preserved(DRAFT, GOOD.replace("138.10", "138.00"))   # changed stop
    assert not numbers_preserved(DRAFT, GOOD + " Expect 5% upside.")       # invented number
    assert not numbers_preserved(DRAFT, "Long NVDA, stop 138.10.")            # dropped numbers


async def test_rule_based_without_llm():
    eng = DialogueEngine(SimulatedClock(et(10)), None)
    assert await eng.render(AgentRole.RISK, DRAFT) == DRAFT + "."
    assert eng.active_provider == "rule_based"


async def test_good_rewrite_accepted_bad_rejected():
    eng = DialogueEngine(SimulatedClock(et(10)), FakeLLM(GOOD))
    assert (await eng.render(AgentRole.JARVIS, DRAFT)).startswith("Sir")
    eng.primary = FakeLLM("Long NVDA at 141, stop 138, size 50.")
    assert await eng.render(AgentRole.JARVIS, DRAFT) == DRAFT + "." and eng.rejected_rewrites == 1


async def test_errors_and_timeouts_fall_back():
    eng = DialogueEngine(SimulatedClock(et(10)), FakeLLM(RuntimeError("HTTP 500")))
    assert await eng.render(AgentRole.NEWS, DRAFT) == DRAFT + "." and eng.failures == 1
    eng = DialogueEngine(SimulatedClock(et(10)), FakeLLM("slow"), timeout=0.05)
    assert await eng.render(AgentRole.NEWS, DRAFT) == DRAFT + "."


async def test_daily_cap_counts_attempts_persists_and_resets():
    clk = SimulatedClock(et(10))
    saved = []

    async def on_call(day, n):
        saved.append((day, n))

    eng = DialogueEngine(clk, FakeLLM(RuntimeError("x")), max_daily_calls=2, on_call=on_call)
    await eng.render(AgentRole.RISK, DRAFT)
    await eng.render(AgentRole.RISK, DRAFT)
    assert eng.budget_exhausted and saved[-1][1] == 2
    eng.primary = FakeLLM(GOOD)
    await eng.render(AgentRole.RISK, DRAFT)
    assert eng.primary.calls == 0
    clk.advance(timedelta(days=1))
    assert eng.calls_today == 0 and not eng.budget_exhausted


async def test_restored_usage_respected():
    clk = SimulatedClock(et(10))
    eng = DialogueEngine(clk, FakeLLM(GOOD), max_daily_calls=5, initial_usage=(clk.now_et().date(), 5))
    await eng.render(AgentRole.RISK, DRAFT)
    assert eng.primary.calls == 0
