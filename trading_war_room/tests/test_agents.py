"""News blackout windows, technical setups/cooldowns, and the War Room debate."""

import asyncio
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.conftest import D, et
from tests.helpers import context, portfolio, position, snapshot
from warroom.agents.jarvis_interface import JarvisInterface
from warroom.agents.news_agent import NewsAgent, blackout_status, weighted_sentiment
from warroom.agents.risk_agent import RiskAgent
from warroom.agents.technical_agent import MacroBias, TechnicalAgent
from warroom.config import parse_config
from warroom.core.clock import SimulatedClock
from warroom.core.event_bus import EventBus
from warroom.models.schemas import (
    AgentRole, Direction, JarvisBriefing, MacroEvent, MacroEventKind, MarketRegime, Mode, NewsItem, PlanStatus,
    Position, SetupType, Severity, Stance, WarRoomMessage,
)
from warroom.providers.llm_provider import DialogueEngine
from warroom.providers.market_data import MockDataProvider
from warroom.providers.news_data import CalendarOnlyNewsProvider, StaticMacroCalendar
from warroom.storage.database import Database
from warroom.war_room import WarRoom

L, S = Direction.LONG, Direction.SHORT


def ev(name, h, m, sev=Severity.HIGH, kind=MacroEventKind.FOMC, ticker=None, **kw):
    return MacroEvent(name=name, kind=kind, scheduled_at=et(h, m), severity=sev, ticker=ticker, **kw)


# ================================================================== news / blackout

@pytest.mark.parametrize("now_h,now_m,active", [(13, 44, False), (13, 45, True), (14, 0, True), (14, 15, True), (14, 16, False)])
def test_blackout_window_edges(now_h, now_m, active):
    r = blackout_status(et(now_h, now_m), [ev("FOMC", 14, 0)], "AAPL", 15, 15)
    assert r.active is active


def test_blackout_ignores_low_impact_and_other_tickers():
    events = [ev("Housing starts", 11, 0, sev=Severity.MEDIUM), ev("MSFT earnings", 11, 0, kind=MacroEventKind.EARNINGS, ticker="MSFT")]
    assert not blackout_status(et(11, 0), events, "AAPL", 15, 15).active
    assert blackout_status(et(11, 0), events, "MSFT", 15, 15).active


def test_halt_blackouts():
    resumed = ev("NVDA halt", 10, 30, kind=MacroEventKind.HALT, ticker="NVDA", ends_at=et(10, 35))
    assert blackout_status(et(10, 45), [resumed], "NVDA", 15, 15).active      # within 15m of resumption
    assert not blackout_status(et(10, 51), [resumed], "NVDA", 15, 15).active
    ongoing = ev("TSLA halt", 10, 30, kind=MacroEventKind.HALT, ticker="TSLA", ongoing=True)
    r = blackout_status(et(15, 0), [ongoing], "TSLA", 15, 15, session_close=et(16, 0))
    assert r.active and r.until == et(16, 0) and "no resumption" in r.reason


def test_upcoming_event_flagged_outside_window():
    r = blackout_status(et(13, 15), [ev("FOMC", 14, 0)], "AAPL", 15, 15)
    assert not r.active and r.upcoming is not None


def test_weighted_sentiment_favours_recent():
    old = NewsItem(ticker="A", headline="up", source="s", published_at=et(6, 0), sentiment=0.8)
    new = NewsItem(ticker="A", headline="down", source="s", published_at=et(10, 50), sentiment=-0.6)
    assert weighted_sentiment([old, new], et(11, 0)) < 0


async def test_news_agent_assessment(cfg, cal):
    clk = SimulatedClock(et(13, 50))
    smc = StaticMacroCalendar(cal, None)
    agent = NewsAgent(cfg, CalendarOnlyNewsProvider(clk, smc), cal, clk)
    a, bo = await agent.assess("AAPL", L)
    assert not a.in_blackout  # 2025-01-15 is not an FOMC day
    clk2 = SimulatedClock(et(13, 50, D.replace(day=29)))      # 2025-01-29 FOMC 14:00
    agent2 = NewsAgent(cfg, CalendarOnlyNewsProvider(clk2, smc), cal, clk2)
    a2, _ = await agent2.assess("AAPL", L)
    assert a2.in_blackout and "FOMC" in a2.blackout_reason


async def test_news_conflict_detection(cfg, cal):
    clk = SimulatedClock(et(11, 0))

    class Feed(CalendarOnlyNewsProvider):
        async def get_news(self, ticker, since):
            return [NewsItem(ticker=ticker, headline="AAPL downgraded to sell", source="x", published_at=et(10, 55),
                             sentiment=-0.7, severity=Severity.MEDIUM)]

    agent = NewsAgent(cfg, Feed(clk, StaticMacroCalendar(cal, None)), cal, clk)
    assert (await agent.assess("AAPL", L))[0].conflicts_with_direction
    assert not (await agent.assess("AAPL", S))[0].conflicts_with_direction


# ================================================================== technical

def _bars(closes, opens=None, highs=None, lows=None, start=None):
    n = len(closes)
    idx = pd.date_range(start or et(9, 30), periods=n, freq="5min", tz="UTC")
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else c - 0.1
    return pd.DataFrame({"open": o, "high": highs if highs is not None else np.maximum(o, c) + 0.1,
                         "low": lows if lows is not None else np.minimum(o, c) - 0.1, "close": c, "volume": 1000.0}, index=idx)


def _tech(cfg, cal, now=None):
    clk = SimulatedClock(now or et(10, 30))
    return TechnicalAgent(cfg, MockDataProvider(clk, cal), cal, clk), clk


NEUTRAL = MacroBias(MarketRegime.MIXED, None, {})


def test_orb_breakout_long_detected(cfg, cal):
    tech, _ = _tech(cfg, cal)
    snap = snapshot(price=101.2, vwap=100.5, ema9=100.9, ema21=100.6, ema50=100.2, rsi14=62, atr14=0.8,
                    orb_high=101.0, orb_low=99.5, rvol=1.8)
    ctx = context(snap)
    ctx.bars_5m = _bars([100.2, 100.6, 100.9, 101.2])
    setups = tech.detect_setups(ctx, MacroBias(MarketRegime.RISK_ON, L, {}))
    best = {s.direction: s for s in setups}
    assert best[L].setup_type is SetupType.ORB_BREAKOUT and best[L].confidence >= 0.7 and S not in best


def test_orb_breakdown_short_detected(cfg, cal):
    tech, _ = _tech(cfg, cal)
    snap = snapshot(price=99.2, vwap=99.9, ema9=99.4, ema21=99.7, ema50=100.1, rsi14=38, atr14=0.8,
                    orb_high=101.0, orb_low=99.5, rvol=1.0)
    ctx = context(snap)
    ctx.bars_5m = _bars([100.0, 99.8, 99.6, 99.2], opens=[100.1, 100.0, 99.8, 99.6])
    setups = tech.detect_setups(ctx, MacroBias(MarketRegime.RISK_ON, L, {}))
    st = next(s for s in setups if s.direction is S)
    assert st.setup_type is SetupType.ORB_BREAKDOWN and any("against" in r for r in st.rationale)


def test_vwap_rejection_and_bounce(cfg, cal):
    tech, _ = _tech(cfg, cal)
    snap = snapshot(price=99.8, vwap=100.0, ema9=99.85, ema21=99.95, ema50=99.0, rsi14=45, atr14=0.8, orb_high=None,
                    orb_low=None, rvol=0.9)
    ctx = context(snap)
    ctx.bars_5m = _bars([100.2, 99.8], opens=[100.1, 100.05], highs=np.array([100.3, 100.02]), lows=np.array([100.0, 99.7]))
    assert any(s.setup_type is SetupType.VWAP_REJECTION for s in tech.detect_setups(ctx, NEUTRAL))
    snap2 = snapshot(price=100.3, vwap=100.0, ema9=100.2, ema21=100.1, ema50=99.0, rsi14=55, atr14=0.8, orb_high=None,
                     orb_low=None, rvol=0.9)
    ctx2 = context(snap2)
    ctx2.bars_5m = _bars([100.1, 100.3], opens=[100.2, 100.05], lows=np.array([100.0, 99.98]))
    assert any(s.setup_type is SetupType.VWAP_BOUNCE for s in tech.detect_setups(ctx2, NEUTRAL))


def test_no_setup_without_indicators(cfg, cal):
    tech, _ = _tech(cfg, cal)
    ctx = context(snapshot(atr14=None))
    ctx.bars_5m = _bars([100, 101])
    assert tech.detect_setups(ctx, NEUTRAL) == []


def test_cooldown_dedup_and_expiry(cfg, cal):
    tech, clk = _tech(cfg, cal)
    assert not tech.on_cooldown("AAPL", L)
    tech.mark_debated("AAPL", L)
    assert tech.on_cooldown("AAPL", L) and not tech.on_cooldown("AAPL", S) and not tech.on_cooldown("MSFT", L)
    clk.advance(timedelta(minutes=14, seconds=59))
    assert tech.on_cooldown("AAPL", L) and tech.cooldown_remaining("AAPL", L) == pytest.approx(1.0)
    clk.advance(timedelta(seconds=1))
    assert not tech.on_cooldown("AAPL", L)


async def test_scan_respects_cooldown_but_analyze_bypasses(cfg, cal):
    tech, clk = _tech(cfg, cal, now=et(11, 0))
    found = await tech.scan(cfg.app.tickers, NEUTRAL)
    for st, _ in found:
        tech.mark_debated(st.ticker, st.direction)
    again = await tech.scan(cfg.app.tickers, NEUTRAL)
    assert not again or all(not tech.on_cooldown(s.ticker, s.direction) for s, _ in again)
    if found:
        st, _ = found[0]
        result = await tech.analyze(st.ticker, NEUTRAL)
        assert result is not None and result[0].on_demand


async def test_analyze_returns_unqualified_lean_when_no_setup(cfg, cal):
    tech, _ = _tech(cfg, cal, now=et(12, 7))
    res = await tech.analyze("AAPL", NEUTRAL)
    assert res is not None
    st, ctx = res
    assert st.on_demand and ctx.snapshot.vwap is not None


async def test_scan_quiet_outside_entry_window(cfg, cal):
    for t in (et(9, 40), et(15, 45)):
        tech, _ = _tech(cfg, cal, now=t)
        assert await tech.scan(cfg.app.tickers, NEUTRAL) == []


def test_possible_halt_flag(cfg, cal):
    tech, _ = _tech(cfg, cal)
    now = et(11, 0)
    stale = context(snapshot("NVDA"), age_s=600)
    peers = [stale] + [context(snapshot(t), age_s=20) for t in ("AAPL", "MSFT", "TSLA")]
    tech.flag_possible_halt(stale, peers, now)
    assert stale.possible_halt
    everyone_stale = [context(snapshot(t), age_s=600) for t in ("AAPL", "MSFT")]
    tech.flag_possible_halt(everyone_stale[0], everyone_stale, now)
    assert not everyone_stale[0].possible_halt  # whole feed stale: a data problem, not a halt


async def test_macro_bias_from_mock(cfg, cal):
    tech, _ = _tech(cfg, cal, now=et(11, 0))
    b = await tech.macro_bias()
    assert b.regime in MarketRegime and set(b.detail) == {"SPY", "QQQ"}


# ================================================================== war room

class Rig:
    """A War Room wired to fake technical context, real News/Risk/J.A.R.V.I.S."""

    def __init__(self, cfg, cal, now=None, db=None, executor=None, positions=None, risk_over=None):
        if risk_over:
            raw = cfg.model_dump()
            raw["risk"].update(risk_over)
            cfg = parse_config(raw)
        self.cfg = cfg
        self.clk = SimulatedClock(now or et(11, 0))
        self.bus = EventBus()
        self.events = self.bus.subscribe(name="test")
        md = MockDataProvider(self.clk, cal)
        self.tech = TechnicalAgent(cfg, md, cal, self.clk)
        self.news = NewsAgent(cfg, CalendarOnlyNewsProvider(self.clk, StaticMacroCalendar(cal, None)), cal, self.clk)
        self.risk = RiskAgent(cfg, cal, self.clk, Mode.DEMO)
        self.dialogue = DialogueEngine(self.clk)
        self.jarvis = JarvisInterface(cfg, self.dialogue)
        self.positions = positions if positions is not None else []
        self.wr = WarRoom(self.tech, self.news, self.risk, self.jarvis, self.dialogue, self.bus, self.clk,
                          lambda: portfolio(positions=self.positions), db=db, executor=executor)

    def drain(self):
        out = []
        while (e := self.events.get_nowait()) is not None:
            out.append(e)
        return out


def _setup_ctx(ticker="AAPL", direction=L, **kw):
    from tests.helpers import setup as mk_setup

    snap = snapshot(ticker, **kw)
    return mk_setup(ticker, direction, snap=snap), context(snap)


async def test_three_round_debate_approves_clean_setup(cfg, cal, tmp_path):
    db = Database(tmp_path / "w.db")
    await db.open()
    rig = Rig(cfg, cal, db=db)
    st, ctx = _setup_ctx()
    t = await rig.wr.convene(st, ctx, NEUTRAL)
    assert {m.round_number for m in t.messages} == {1, 2, 3}
    r1 = [m.sender for m in t.messages if m.round_number == 1]
    assert r1 == [AgentRole.TECHNICAL, AgentRole.NEWS, AgentRole.RISK]
    assert any(m.round_number == 2 for m in t.messages)
    verdict = [m for m in t.messages if m.round_number == 3 and m.sender is AgentRole.RISK][0]
    assert verdict.stance is Stance.APPROVE and t.plan.status is PlanStatus.APPROVED
    assert t.plan.stop_loss < t.plan.entry < t.plan.tp1 < t.plan.tp2 and t.plan.position_size >= 1
    assert t.briefing.startswith("Sir,") and "Risk approved" in t.briefing
    events = rig.drain()
    assert sum(isinstance(e, WarRoomMessage) for e in events) == len(t.messages)
    assert any(isinstance(e, JarvisBriefing) for e in events)
    assert await db.count("agent_messages") == len(t.messages) and (await db.latest_transcript("AAPL")).debate_id == t.debate_id
    assert rig.tech.on_cooldown("AAPL", L)
    await db.close()


async def test_veto_debate_has_challenge_and_binding_veto(cfg, cal):
    rig = Rig(cfg, cal)
    st, ctx = _setup_ctx()
    ctx.prior_high = 100.5  # caps TP1 -> R:R < 2
    t = await rig.wr.convene(st, ctx, NEUTRAL)
    r2 = [m for m in t.messages if m.round_number == 2]
    assert any(m.sender is AgentRole.RISK and m.stance is Stance.CHALLENGE and m.recipient is AgentRole.TECHNICAL for m in r2)
    assert t.plan.status is PlanStatus.VETOED and t.plan.veto_reasons and t.plan.position_size == 0
    assert "declined" in t.briefing


async def test_counter_trend_cross_examination(cfg, cal):
    rig = Rig(cfg, cal)
    st, ctx = _setup_ctx()
    t = await rig.wr.convene(st, ctx, MacroBias(MarketRegime.RISK_OFF, S, {}))
    r2 = [m for m in t.messages if m.round_number == 2]
    assert any("fights the index" in m.message for m in r2) and any(m.stance is Stance.DEFEND for m in r2)


async def test_low_confidence_goes_to_watchlist(cfg, cal):
    rig = Rig(cfg, cal)
    from tests.helpers import setup as mk_setup

    snap = snapshot()
    t = await rig.wr.convene(mk_setup(snap=snap, confidence=0.4), context(snap), NEUTRAL)
    assert t.plan.status is PlanStatus.WATCHLIST and "confidence" in t.plan.summary
    t2 = await rig.wr.convene(mk_setup(snap=snap, qualified=False, confidence=0.9), context(snap), NEUTRAL)
    assert t2.plan.status is PlanStatus.WATCHLIST


async def test_executor_called_only_for_approved_and_failures_downgrade(cfg, cal):
    calls = []

    async def executor(plan, ctx):
        calls.append(plan.ticker)

    rig = Rig(cfg, cal, executor=executor)
    st, ctx = _setup_ctx()
    await rig.wr.convene(st, ctx, NEUTRAL)
    ctx2 = context(snapshot("MSFT"))
    ctx2.prior_high = 100.3
    from tests.helpers import setup as mk_setup
    await rig.wr.convene(mk_setup("MSFT", snap=ctx2.snapshot), ctx2, NEUTRAL)
    assert calls == ["AAPL"]

    async def broken(plan, ctx):
        raise RuntimeError("broker offline")

    rig2 = Rig(cfg, cal, executor=broken)
    t = await rig2.wr.convene(*_setup_ctx(), NEUTRAL)
    assert t.plan.status is PlanStatus.WATCHLIST and "execution failed" in t.plan.summary


async def test_portfolio_lock_prevents_overfilling_last_slot(cfg, cal):
    """Two debates on different tickers race for the last open slot: exactly one wins."""
    holder: list[Position] = [position("MSFT"), position("NVDA")]

    async def executor(plan, ctx):
        await asyncio.sleep(0)  # yield mid-execution to invite interleaving
        holder.append(position(plan.ticker))

    rig = Rig(cfg, cal, executor=executor, positions=holder)
    a = _setup_ctx("AAPL")
    b = _setup_ctx("AMD")
    t1, t2 = await asyncio.gather(rig.wr.convene(*a, NEUTRAL), rig.wr.convene(*b, NEUTRAL))
    statuses = sorted([t1.plan.status, t2.plan.status])
    assert statuses == [PlanStatus.APPROVED, PlanStatus.VETOED] and len(holder) == 3
    loser = t1 if t1.plan.status is PlanStatus.VETOED else t2
    assert any("max concurrent" in v for v in loser.plan.veto_reasons)


async def test_same_ticker_debates_serialize(cfg, cal):
    rig = Rig(cfg, cal)
    order = []
    orig = rig.wr._debate

    async def traced(setup, ctx, bias):
        order.append(("start", setup.direction))
        await asyncio.sleep(0.01)
        t = await orig(setup, ctx, bias)
        order.append(("end", setup.direction))
        return t

    rig.wr._debate = traced
    st_l, ctx = _setup_ctx(direction=L)
    from tests.helpers import setup as mk_setup
    st_s = mk_setup(direction=S, snap=ctx.snapshot)
    await asyncio.gather(rig.wr.convene(st_l, ctx, NEUTRAL), rig.wr.convene(st_s, ctx, NEUTRAL))
    assert [o[0] for o in order] == ["start", "end", "start", "end"]


async def test_full_pipeline_on_mock_day(cfg, cal):
    """Scan a mock session and debate every setup: transcripts must always be well formed."""
    rig = Rig(cfg, cal, now=et(10, 0))
    rig.tech.market.clock = rig.clk
    seen = 0
    for _ in range(0, 300, 10):
        bias = await rig.tech.macro_bias()
        for st, ctx in await rig.tech.scan(cfg.app.tickers, bias):
            t = await rig.wr.convene(st, ctx, bias)
            seen += 1
            if t.plan.status is PlanStatus.APPROVED:
                assert t.plan.position_size >= 1
                assert (t.plan.stop_loss < t.plan.entry < t.plan.tp1 < t.plan.tp2) if t.direction is L else \
                    (t.plan.stop_loss > t.plan.entry > t.plan.tp1 > t.plan.tp2)
        rig.clk.advance(timedelta(minutes=10))
    assert seen > 0


def test_breakout_not_chased_when_rsi_exhausted(cfg, cal):
    tech, _ = _tech(cfg, cal)
    snap = snapshot(price=101.2, vwap=100.5, ema9=100.9, ema21=100.6, ema50=100.2, rsi14=86, atr14=0.8,
                    orb_high=101.0, orb_low=99.5, rvol=1.8)
    ctx = context(snap)
    ctx.bars_5m = _bars([100.2, 100.6, 100.9, 101.2])
    assert not any(s.setup_type is SetupType.ORB_BREAKOUT for s in tech.detect_setups(ctx, NEUTRAL))
