"""The Agent War Room: a 3-round structured debate per setup.

Round 1 - Pitch & evidence: Technical pitches; News reports catalysts and
          blackouts; Risk computes levels, size, R:R and portfolio checks.
Round 2 - Cross-examination: agents challenge each other on concrete
          conflicts (news vs tape, index headwind, capped targets, wide
          stops, data/spread quality) and the challenged agent defends or
          concedes.
Round 3 - Binding verdict: Risk re-evaluates against the LIVE portfolio and
          casts APPROVE/VETO; J.A.R.V.I.S. synthesises the TradePlan
          (APPROVED / WATCHLIST / VETOED).

Concurrency:
* a per-ticker ``asyncio.Lock`` stops two debates on one symbol colliding;
* a single portfolio lock wraps the binding vote AND execution, so two
  debates on different tickers can't both take the last position slot.

Every message is persisted and published on the EventBus as it is produced.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from warroom.agents.jarvis_interface import SETUP_NAMES, JarvisInterface
from warroom.agents.news_agent import BlackoutResult, NewsAgent
from warroom.agents.risk_agent import PortfolioView, RiskAgent, RiskNotes
from warroom.agents.technical_agent import MacroBias, TechnicalAgent, TickerContext
from warroom.core.clock import Clock, to_et
from warroom.core.event_bus import EventBus
from warroom.models.schemas import (
    AgentMessage,
    AgentRole,
    Direction,
    JarvisBriefing,
    NewsAssessment,
    PlanStatus,
    RiskAssessment,
    Stance,
    SystemAlert,
    TradePlan,
    TradeSetup,
    WarRoomMessage,
    WarRoomTranscript,
    new_id,
)
from warroom.providers.llm_provider import DialogueEngine

log = logging.getLogger(__name__)

Executor = Callable[[TradePlan, TickerContext], Awaitable[None]]


class WarRoom:
    def __init__(
        self,
        technical: TechnicalAgent,
        news: NewsAgent,
        risk: RiskAgent,
        jarvis: JarvisInterface,
        dialogue: DialogueEngine,
        bus: EventBus,
        clock: Clock,
        portfolio: Callable[[], PortfolioView],
        *,
        db=None,
        executor: Executor | None = None,
        portfolio_lock: asyncio.Lock | None = None,
        min_confidence: float = 0.55,
    ) -> None:
        self.technical = technical
        self.news = news
        self.risk = risk
        self.jarvis = jarvis
        self.dialogue = dialogue
        self.bus = bus
        self.clock = clock
        self.portfolio = portfolio
        self.db = db
        self.executor = executor
        self.portfolio_lock = portfolio_lock or asyncio.Lock()
        self.min_confidence = min_confidence
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_transcript: WarRoomTranscript | None = None
        self.transcripts: dict[str, WarRoomTranscript] = {}

    def lock_for(self, ticker: str) -> asyncio.Lock:
        return self._locks[ticker]

    # ------------------------------------------------------------------ messaging

    async def _say(self, debate: list[AgentMessage], debate_id: str, ticker: str, sender: AgentRole, rnd: int,
                   stance: Stance, draft: str, recipient: AgentRole = AgentRole.ALL, **payload) -> AgentMessage:
        text = await self.dialogue.render(sender, draft)
        msg = AgentMessage(debate_id=debate_id, sender=sender, recipient=recipient, round_number=rnd, stance=stance,
                           message=text, data_payload=payload, timestamp=self.clock.now())
        debate.append(msg)
        if self.db is not None:
            try:
                await self.db.save_message(msg)
            except Exception:  # noqa: BLE001 - persistence failure must not abort a debate
                log.exception("failed to persist war room message")
        self.bus.publish(WarRoomMessage(timestamp=msg.timestamp, message=msg, ticker=ticker))
        return msg

    # ------------------------------------------------------------------ the debate

    async def convene(self, setup: TradeSetup, ctx: TickerContext, bias: MacroBias) -> WarRoomTranscript:
        async with self.lock_for(setup.ticker):
            self.technical.mark_debated(setup.ticker, setup.direction)
            return await self._debate(setup, ctx, bias)

    async def _debate(self, setup: TradeSetup, ctx: TickerContext, bias: MacroBias) -> WarRoomTranscript:
        did = new_id()
        started = self.clock.now()
        msgs: list[AgentMessage] = []
        tk, d = setup.ticker, setup.direction
        s = ctx.snapshot
        name = SETUP_NAMES.get(setup.setup_type.value, setup.setup_type.value)

        news, bo = await self.news.assess(tk, d)
        prelim, notes = self.risk.evaluate(setup, ctx, news, self.portfolio(), bias.direction)

        # ================= ROUND 1: pitch & evidence
        vw = f"{s.vwap:.2f}" if s.vwap else "n/a"
        dist = f" ({s.vwap_distance_pct:+.2%})" if s.vwap_distance_pct is not None else ""
        rvol = f"{s.rvol:.1f}x" if s.rvol is not None else "n/a"
        rsi = f"{s.rsi14:.0f}" if s.rsi14 is not None else "n/a"
        atr = f"{s.atr14:.2f}" if s.atr14 is not None else "n/a"
        pitch_stance = (Stance.NEUTRAL if not setup.qualified
                        else Stance.BULLISH if d is Direction.LONG else Stance.BEARISH)
        lead = f"{tk} {name}, {d.value.lower()}." if setup.qualified else f"{tk}: no qualified setup."
        await self._say(msgs, did, tk, AgentRole.TECHNICAL, 1, pitch_stance,
                        f"{lead} Price {s.price:.2f} vs VWAP {vw}{dist}, RSI {rsi}, RVOL {rvol}, ATR {atr}. "
                        f"{'; '.join(setup.rationale)}. Confidence {setup.confidence:.0%}.",
                        setup_type=setup.setup_type.value, confidence=setup.confidence, price=s.price,
                        vwap=s.vwap, rsi=s.rsi14, rvol=s.rvol, atr=s.atr14, qualified=setup.qualified)

        await self._say(msgs, did, tk, AgentRole.NEWS, 1, self._news_stance(news, bo, d), self._news_report(news, bo),
                        sentiment=news.sentiment, severity=news.severity.value, blackout=news.in_blackout,
                        blackout_reason=news.blackout_reason, conflict=news.conflicts_with_direction)

        await self._say(msgs, did, tk, AgentRole.RISK, 1, Stance.CAUTION if prelim.veto_reasons else Stance.NEUTRAL,
                        self._risk_numbers(prelim, notes, preliminary=True),
                        entry=prelim.entry, stop=prelim.stop_loss, tp1=prelim.tp1, tp2=prelim.tp2,
                        size=prelim.position_size, dollar_risk=prelim.dollar_risk, rr=prelim.reward_risk,
                        flags=prelim.veto_reasons)

        # ================= ROUND 2: cross-examination
        await self._cross_examine(msgs, did, setup, ctx, bias, news, bo, prelim, notes)

        # ================= ROUND 3: binding verdict (atomic with execution)
        async with self.portfolio_lock:
            final, fnotes = self.risk.evaluate(setup, ctx, news, self.portfolio(), bias.direction)
            if final.approved:
                await self._say(msgs, did, tk, AgentRole.RISK, 3, Stance.APPROVE,
                                f"APPROVE. Binding check passed against the live book. "
                                f"{self._risk_numbers(final, fnotes, preliminary=False)}",
                                verdict="APPROVE", size=final.position_size, dollar_risk=final.dollar_risk)
            else:
                await self._say(msgs, did, tk, AgentRole.RISK, 3, Stance.VETO,
                                "VETO. " + "; ".join(final.veto_reasons) + ".",
                                verdict="VETO", reasons=final.veto_reasons)
            plan = self._synthesise(did, setup, final, news)
            if plan.status is PlanStatus.APPROVED and self.executor is not None:
                try:
                    await self.executor(plan, ctx)
                except Exception as exc:  # noqa: BLE001
                    log.exception("execution failed for %s", tk)
                    self.bus.publish(SystemAlert(timestamp=self.clock.now(), level="CRITICAL", source="war_room",
                                                 message=f"execution failed for {tk}: {exc}"))
                    plan = plan.model_copy(update={"status": PlanStatus.WATCHLIST,
                                                   "summary": f"approved but paper execution failed ({exc})"})
        await self._say(msgs, did, tk, AgentRole.JARVIS, 3, self._plan_stance(plan.status),
                        f"Verdict: {plan.status.value}. {plan.summary}.", status=plan.status.value)

        draft = WarRoomTranscript(debate_id=did, ticker=tk, direction=d, started_at=started,
                                  finished_at=self.clock.now(), messages=msgs, plan=plan)
        briefing = await self.jarvis.briefing(draft)
        transcript = draft.model_copy(update={"briefing": briefing})
        self.last_transcript = transcript
        self.transcripts[tk] = transcript
        if self.db is not None:
            try:
                await self.db.save_transcript(transcript)
            except Exception:  # noqa: BLE001
                log.exception("failed to persist transcript")
        self.bus.publish(JarvisBriefing(timestamp=self.clock.now(), text=briefing, debate_id=did,
                                        speak=plan.status is not PlanStatus.VETOED or setup.on_demand,
                                        status=plan.status.value))
        return transcript

    # ------------------------------------------------------------------ round 2

    async def _cross_examine(self, msgs, did, setup: TradeSetup, ctx: TickerContext, bias: MacroBias,
                             news: NewsAssessment, bo: BlackoutResult, risk: RiskAssessment, notes: RiskNotes) -> None:
        tk, d = setup.ticker, setup.direction
        s = ctx.snapshot
        T, N, R = AgentRole.TECHNICAL, AgentRole.NEWS, AgentRole.RISK
        rvol_ok = s.rvol is not None and s.rvol >= self.technical.t.rvol_threshold
        spoke = False

        if news.in_blackout:
            point = f"no entries during the blackout ({news.blackout_reason})"
            await self._say(msgs, did, tk, N, 2, Stance.CHALLENGE, f"Stand down: {point}.", recipient=T, point=point)
            until = f" until {to_et(news.blackout_until):%H:%M} ET" if news.blackout_until else ""
            await self._say(msgs, did, tk, T, 2, Stance.NEUTRAL,
                            f"Conceded. The structure will still be there{until}; I will re-pitch if it holds.",
                            recipient=N)
            spoke = True
        elif news.conflicts_with_direction:
            h = NewsAgent.top_headline(news)
            head = f"'{h.headline}'" if h else "recent headlines"
            point = f"{head} scores {news.sentiment:+.2f} against a {d.value.lower()}"
            await self._say(msgs, did, tk, N, 2, Stance.CHALLENGE,
                            f"{point}. Price may simply not have caught up with the news.", recipient=T, point=point)
            if rvol_ok:
                side = "above" if d is Direction.LONG else "below"
                await self._say(msgs, did, tk, T, 2, Stance.DEFEND,
                                f"The tape disagrees: RVOL {s.rvol:.1f}x with price holding {side} VWAP. If the headline "
                                "mattered, volume would be leaving, not arriving.", recipient=N)
            else:
                await self._say(msgs, did, tk, T, 2, Stance.NEUTRAL,
                                "Fair challenge. Without volume confirmation I cannot argue the tape over the headline.",
                                recipient=N)
            spoke = True

        if notes.counter_trend:
            need = self.risk.r.counter_trend_min_reward_risk
            have = f"{notes.blended_rr:.2f}" if notes.blended_rr is not None else "n/a"
            point = f"SPY/QQQ are {bias.regime.value.replace('_', '-').lower()}; this {d.value.lower()} fights the index"
            await self._say(msgs, did, tk, R, 2, Stance.CHALLENGE,
                            f"{point}. Counter-trend needs blended R:R {need:.1f}; this offers {have}.",
                            recipient=T, point=point)
            await self._say(msgs, did, tk, T, 2, Stance.DEFEND,
                            f"Relative strength is the thesis: {tk} is {s.vwap_distance_pct:+.2%} from VWAP while the "
                            "index leans the other way." if s.vwap_distance_pct is not None
                            else "Relative strength is the thesis, though I concede the index is not helping.",
                            recipient=R)
            spoke = True

        if notes.target_caps and risk.reward_risk is not None:
            point = f"{notes.target_caps[0]}, so TP1 pays only {risk.reward_risk:.2f}R"
            await self._say(msgs, did, tk, R, 2, Stance.CHALLENGE, f"{point}.", recipient=T, point=point)
            if risk.reward_risk >= self.risk.r.min_reward_risk:
                await self._say(msgs, did, tk, T, 2, Stance.DEFEND,
                                f"Even capped, {risk.reward_risk:.2f}R before that level clears the bar.", recipient=R)
            else:
                wall = "overhead supply" if d is Direction.LONG else "support underneath"
                await self._say(msgs, did, tk, T, 2, Stance.NEUTRAL,
                                f"Agreed. That level is {wall}; the trade lacks room.", recipient=R)
            spoke = True

        if notes.stop_pct is not None and (notes.stop_atr_multiple >= 1.4 or notes.stop_pct > 0.02):
            point = f"stop is {notes.stop_atr_multiple:.2f}x ATR ({notes.stop_pct:.2%}) away ({notes.stop_source})"
            await self._say(msgs, did, tk, R, 2, Stance.CHALLENGE,
                            f"The {point}; size drops to {risk.position_size} shares.", recipient=T, point=point)
            await self._say(msgs, did, tk, T, 2, Stance.DEFEND,
                            "The stop sits where the structure is; anything tighter would live inside the noise.",
                            recipient=R)
            spoke = True

        quality = [v for v in risk.veto_reasons if v.startswith(("stale data", "spread", "no real bid/ask"))]
        if quality:
            point = quality[0]
            await self._say(msgs, did, tk, R, 2, Stance.CHALLENGE, f"Execution quality problem: {point}.",
                            recipient=T, point=point)
            spoke = True

        if not spoke:
            upcoming = (f" Note {bo.upcoming.name} at {to_et(bo.upcoming.scheduled_at):%H:%M} ET."
                        if bo.upcoming else "")
            await self._say(msgs, did, tk, N, 2, Stance.NEUTRAL,
                            f"No objection from the news desk: nothing material against the {d.value.lower()}.{upcoming}",
                            recipient=T)
            await self._say(msgs, did, tk, R, 2, Stance.NEUTRAL,
                            "Numbers hold on my side. No objection pending the binding check.", recipient=T)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _news_stance(news: NewsAssessment, bo: BlackoutResult, d: Direction) -> Stance:
        if news.in_blackout or news.conflicts_with_direction or bo.upcoming is not None:
            return Stance.CAUTION
        if news.sentiment * d.sign >= 0.3:
            return Stance.BULLISH if d is Direction.LONG else Stance.BEARISH
        return Stance.NEUTRAL

    @staticmethod
    def _news_report(news: NewsAssessment, bo: BlackoutResult) -> str:
        parts = [f"Sentiment {news.sentiment:+.2f}, catalyst severity {news.severity.value}."]
        h = NewsAgent.top_headline(news)
        if h:
            parts.append(f"Top headline: '{h.headline}' ({h.source}, {to_et(h.published_at):%H:%M} ET).")
        else:
            parts.append("No ticker-specific headlines.")
        if news.in_blackout:
            parts.append(f"BLACKOUT: {news.blackout_reason}.")
        elif bo.upcoming:
            parts.append(f"Upcoming: {bo.upcoming.name} at {to_et(bo.upcoming.scheduled_at):%H:%M} ET.")
        return " ".join(parts)

    @staticmethod
    def _risk_numbers(a: RiskAssessment, n: RiskNotes, preliminary: bool) -> str:
        if a.stop_loss is None or a.tp1 is None:
            core = "Levels could not be computed."
        else:
            rr = f"{a.reward_risk:.2f}" if a.reward_risk is not None else "n/a"
            core = (f"Entry {a.entry:.2f} ({n.entry_basis}), stop {a.stop_loss:.2f} ({n.stop_source}, "
                    f"{n.stop_atr_multiple:.2f}x ATR), TP1 {a.tp1:.2f}, TP2 {a.tp2:.2f}; R:R {rr} to TP1. "
                    f"Size {a.position_size} shares (risk cap {a.risk_shares}, buying-power cap "
                    f"{a.buying_power_shares}), ${a.dollar_risk:,.2f} at risk.")
        if preliminary and a.veto_reasons:
            core += " Flags: " + "; ".join(a.veto_reasons) + "."
        if a.warnings:
            core += " Notes: " + "; ".join(a.warnings) + "."
        return core

    def _synthesise(self, did: str, setup: TradeSetup, a: RiskAssessment, news: NewsAssessment) -> TradePlan:
        base = dict(debate_id=did, ticker=setup.ticker, direction=setup.direction, setup_type=setup.setup_type,
                    entry=a.entry, stop_loss=a.stop_loss, tp1=a.tp1, tp2=a.tp2, reward_risk=a.reward_risk,
                    created_at=self.clock.now())
        if not a.approved:
            return TradePlan(status=PlanStatus.VETOED, veto_reasons=a.veto_reasons, position_size=0, dollar_risk=0.0,
                             summary=f"Risk veto: {a.veto_reasons[0]}", **base)
        if not setup.qualified:
            reason = "no qualified technical trigger yet"
        elif setup.confidence < self.min_confidence:
            reason = f"confidence {setup.confidence:.0%} is below the {self.min_confidence:.0%} execution bar"
        else:
            reason = None
        if reason:
            return TradePlan(status=PlanStatus.WATCHLIST, position_size=a.position_size, dollar_risk=a.dollar_risk,
                             summary=reason, **base)
        return TradePlan(status=PlanStatus.APPROVED, position_size=a.position_size, dollar_risk=a.dollar_risk,
                         summary=(f"{a.position_size} shares, entry {a.entry:.2f}, stop {a.stop_loss:.2f}, "
                                  f"TP1 {a.tp1:.2f}, TP2 {a.tp2:.2f}, risking ${a.dollar_risk:,.2f}"), **base)

    @staticmethod
    def _plan_stance(status: PlanStatus) -> Stance:
        return {PlanStatus.APPROVED: Stance.APPROVE, PlanStatus.VETOED: Stance.VETO}.get(status, Stance.CAUTION)
