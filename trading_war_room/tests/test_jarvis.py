import pytest

from tests.conftest import et
from warroom.agents.jarvis_interface import (
    CommandKind, JarvisInterface, PositionLine, StatusReport, parse_command,
)
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import AgentMessage, AgentRole, MarketRegime, PlanStatus, TradePlan, WarRoomTranscript
from warroom.providers.llm_provider import DialogueEngine


@pytest.mark.parametrize("text,kind,ticker", [
    ("status", CommandKind.STATUS, None),
    ("Status report", CommandKind.STATUS, None),
    ("How are we doing?", CommandKind.STATUS, None),
    ("analyze TSLA", CommandKind.ANALYZE, "TSLA"),
    ("Analyze tsla", CommandKind.ANALYZE, "TSLA"),
    ("Analyse $NVDA please", CommandKind.ANALYZE, "NVDA"),
    ("look at amd", CommandKind.ANALYZE, "AMD"),
    ("MSFT", CommandKind.ANALYZE, "MSFT"),
    ("why AAPL", CommandKind.WHY, "AAPL"),
    ("Why did you veto AAPL?", CommandKind.WHY, "AAPL"),
    ("why did you veto aapl", CommandKind.WHY, "AAPL"),
    ("why?", CommandKind.WHY, None),
    ("transcript", CommandKind.TRANSCRIPT, None),
    ("transcript NVDA", CommandKind.TRANSCRIPT, "NVDA"),
    ("pause", CommandKind.PAUSE, None),
    ("Stand down", CommandKind.PAUSE, None),
    ("resume", CommandKind.RESUME, None),
    ("flatten", CommandKind.FLATTEN, None),
    ("close everything", CommandKind.FLATTEN, None),
    ("help", CommandKind.HELP, None),
    ("make me rich", CommandKind.UNKNOWN, None),
    ("", CommandKind.UNKNOWN, None),
])
def test_parse_command(text, kind, ticker):
    c = parse_command(text)
    assert c.kind is kind and c.ticker == ticker


def _transcript(status=PlanStatus.VETOED):
    did = "d1"
    msgs = [AgentMessage(debate_id=did, sender=s, round_number=r, stance="NEUTRAL", message=f"{s.value} says", timestamp=et(11))
            for r, s in ((1, AgentRole.TECHNICAL), (1, AgentRole.NEWS), (2, AgentRole.RISK), (3, AgentRole.RISK))]
    kw = dict(entry=100.07, stop_loss=98.9, tp1=102.41, tp2=103.58, position_size=249, dollar_risk=291.33, reward_risk=2.0)
    plan = TradePlan(debate_id=did, ticker="AAPL", direction="LONG", setup_type="ORB_BREAKOUT", status=status,
                     veto_reasons=["R:R 1.20 to TP1 is below 2.0"] if status is PlanStatus.VETOED else [],
                     summary="confidence 40% is below the 55% execution bar", created_at=et(11),
                     **({} if status is PlanStatus.VETOED else kw))
    return WarRoomTranscript(debate_id=did, ticker="AAPL", direction="LONG", started_at=et(11), finished_at=et(11),
                             messages=msgs, plan=plan)


class Ctx:
    def __init__(self):
        self.paused = False
        self.t = _transcript()

    async def status(self):
        return StatusReport(mode="DEMO", equity=100_250.5, day_pnl=250.5, day_pnl_pct=0.0025, regime=MarketRegime.RISK_ON,
                            circuit="ARMED", positions=[PositionLine("NVDA", "LONG", 170, 146.46, 145.01, 149.36, 150.81, 147.0, 91.8)])

    async def analyze(self, ticker):
        return _transcript(PlanStatus.APPROVED)

    async def latest_transcript(self, ticker):
        return self.t if ticker in (None, "AAPL") else None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def flatten(self):
        return ["NVDA long 170 @ 147.00"]


@pytest.fixture
def jarvis(cfg):
    return JarvisInterface(cfg, DialogueEngine(SimulatedClock(et(11))))


def test_briefings_cover_all_statuses(jarvis):
    v = jarvis.briefing_draft(_transcript(PlanStatus.VETOED))
    assert v.startswith("Sir,") and "R:R 1.20" in v and "!" not in v
    a = jarvis.briefing_draft(_transcript(PlanStatus.APPROVED))
    for token in ("100.07", "98.90", "102.41", "103.58", "249 shares", "$291.33"):
        assert token in a
    w = jarvis.briefing_draft(_transcript(PlanStatus.WATCHLIST))
    assert "watchlist" in w
    assert 2 <= a.count(".") + a.count(";") and len(a.split(". ")) <= 6


async def test_handle_commands(jarvis):
    ctx = Ctx()
    s = await jarvis.handle("Status report", ctx)
    assert "$100,250.50" in s and "+0.25%" in s and "NVDA long 170" in s and "armed" in s
    assert "Risk approved" in await jarvis.handle("analyze TSLA", ctx)
    why = await jarvis.handle("Why did you veto AAPL?", ctx)
    assert "vetoed because: R:R 1.20" in why and "NEWS says" in why
    assert "no War Room record" in await jarvis.handle("why ZZZZ", ctx)
    tr = await jarvis.handle("transcript", ctx)
    assert tr.count("\n") == 4 and "R2 [RISK]" in tr
    assert "paused" in (await jarvis.handle("pause", ctx)).lower() and ctx.paused
    await jarvis.handle("resume", ctx)
    assert not ctx.paused
    assert "NVDA long 170" in await jarvis.handle("flatten", ctx)
    assert "didn't catch" in await jarvis.handle("buy the dip", ctx)
    assert "Commands" in await jarvis.handle("help", ctx)


def test_custom_title(cfg):
    raw = cfg.model_dump()
    raw["app"]["user_title"] = "Ma'am"
    from warroom.config import parse_config
    j = JarvisInterface(parse_config(raw), DialogueEngine(SimulatedClock(et(11))))
    assert j.briefing_draft(_transcript()).startswith("Ma'am,")
