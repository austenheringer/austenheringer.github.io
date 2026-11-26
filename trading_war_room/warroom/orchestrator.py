"""Orchestrator: wires providers, agents, War Room, broker, storage, voice,
notifications and the EventBus, and runs the scan/execute cycle.

One cycle (``run_cycle``):
 1. roll the trading day if needed (resets daily breakers, keeps manual pause);
 2. standby if the market is closed (PAPER only; DEMO/REPLAY bypass);
 3. feed every NEW completed 1m bar to the paper broker for tickers with open
    positions (stops, targets, EOD flat) - in order, none skipped;
 4. build one TickerContext per watched ticker, publish MarketUpdates;
 5. compute the SPY/QQQ macro bias;
 6. detect setups (cooldown-gated) and convene the War Room for the best few;
    approved plans execute inside the War Room's portfolio lock;
 7. persist an equity snapshot and the risk state.

The orchestrator also implements ``JarvisContext`` (status / analyze / why /
transcript / pause / resume / flatten) for the command bar.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from warroom.agents.jarvis_interface import JarvisInterface, PositionLine, StatusReport
from warroom.agents.news_agent import NewsAgent
from warroom.agents.risk_agent import PortfolioView, RiskAgent
from warroom.agents.technical_agent import TechnicalAgent, TickerContext
from warroom.config import AppConfig, Secrets
from warroom.core.assets import format_price, format_qty
from warroom.core.clock import Clock, SystemClock, to_et
from warroom.core.event_bus import EventBus
from warroom.core.market_calendar import MarketCalendar, MarketStatus
from warroom.execution.paper_broker import Bar, OrderRejected, PaperBroker
from warroom.models.schemas import (
    AlertLevel,
    ExitReason,
    JarvisBriefing,
    MarketRegime,
    MarketUpdate,
    Mode,
    SystemAlert,
    Timeframe,
    TradePlan,
    WarRoomTranscript,
)
from warroom.notifications.alerts import NotificationHub, build_notifications
from warroom.providers import factory
from warroom.providers.base import MarketDataProvider, NewsProvider, ProviderError
from warroom.providers.llm_provider import DialogueEngine
from warroom.storage.database import Database
from warroom.voice.speech import SpeechQueue
from warroom.war_room import WarRoom

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    at: datetime
    standby: bool = False
    debates: list[WarRoomTranscript] = field(default_factory=list)
    fills: int = 0
    error: str | None = None


class Orchestrator:
    def __init__(self, cfg: AppConfig, secrets: Secrets, mode: Mode, clock: Clock, calendar: MarketCalendar,
                 market: MarketDataProvider, news_provider: NewsProvider, dialogue: DialogueEngine, db: Database,
                 bus: EventBus, speech: SpeechQueue | None = None, notifications: NotificationHub | None = None) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.mode = mode
        self.clock = clock
        self.calendar = calendar
        self.market = market
        self.news_provider = news_provider
        self.dialogue = dialogue
        self.db = db
        self.bus = bus
        self.speech = speech or SpeechQueue(False)
        self.notifications = notifications or NotificationHub([], [])

        self.technical = TechnicalAgent(cfg, market, calendar, clock)
        self.news = NewsAgent(cfg, news_provider, calendar, clock)
        self.risk = RiskAgent(cfg, calendar, clock, mode)
        self.broker = PaperBroker(cfg.execution, cfg.risk, clock, calendar, cfg.account.starting_cash,
                                  db=db, risk_agent=self.risk, bus=bus)
        self.jarvis = JarvisInterface(cfg, dialogue)
        self.war_room = WarRoom(self.technical, self.news, self.risk, self.jarvis, dialogue, bus, clock,
                                self.portfolio_view, db=db, executor=self._execute,
                                min_confidence=cfg.technical.min_confidence_to_trade)
        self.sod_equity = cfg.account.starting_equity
        self._sod_day = clock.now_et().date()
        self.regime = MarketRegime.UNKNOWN
        self.contexts: dict[str, TickerContext] = {}
        self._last_bar: dict[str, datetime] = {}
        self._standby_announced = False
        self._halt_announced = False
        self._bg: set[asyncio.Task] = set()
        self._closed = False
        self.cycles = 0

    # ------------------------------------------------------------------ construction

    @classmethod
    async def build(cls, cfg: AppConfig, secrets: Secrets, mode: Mode, *, clock: Clock | None = None,
                    db_path: str | None = None, voice: bool = False, market: MarketDataProvider | None = None,
                    news_provider: NewsProvider | None = None) -> "Orchestrator":
        calendar = factory.build_calendar(cfg)
        db = Database(db_path or cfg.storage.resolve_db_path(mode))
        await db.open()
        if clock is None:
            if mode is Mode.DEMO:
                clock = factory.build_demo_clock(cfg, calendar, SystemClock().now(),
                                                 resume_after=await db.latest_activity())
            else:
                clock = SystemClock()
        bus = EventBus()
        notes: list[str] = []
        market = market or factory.build_market_data(cfg, secrets, clock, calendar, mode)
        if news_provider is None:
            macro, notes = await factory.build_macro_calendar(cfg, calendar, clock, mode)
            news_provider = factory.build_news(cfg, secrets, clock, calendar, mode, market=market, macro=macro)
        dialogue = await factory.build_dialogue(cfg, secrets, clock, mode, db=db)
        speech = SpeechQueue(voice and mode not in (Mode.SMOKE, Mode.REPLAY), cfg.voice)
        notifications = build_notifications(cfg, secrets) if mode in (Mode.PAPER, Mode.DEMO) else NotificationHub([], [])
        orch = cls(cfg, secrets, mode, clock, calendar, market, news_provider, dialogue, db, bus, speech, notifications)
        orch._startup_notes = notes
        return orch

    async def start(self) -> list[str]:
        """Recover persisted state, start voice/notifications, greet. Returns startup notes."""
        rec = await self.db.recover(self.clock.now(), self.calendar, self.cfg.risk.consecutive_loss_limit,
                                    self.cfg.risk.consecutive_loss_cooldown_minutes)
        self.broker.restore(rec.cash, rec.reserved_margin, rec.open_positions)
        self.risk.state = rec.risk_state
        self.sod_equity = rec.start_of_day_equity or self.broker.equity()
        backend = await self.speech.start()
        self.notifications.start(self.bus)
        notes = list(getattr(self, "_startup_notes", [])) + rec.notes
        for n in notes:
            level = AlertLevel.WARNING if ("WARNING" in n or "coverage" in n or "failed" in n) else AlertLevel.INFO
            self.bus.publish(SystemAlert(timestamp=self.clock.now(), level=level, source="startup", message=n))
        t = self.cfg.app.user_title
        now_et = self.clock.now_et()
        greeting = (f"Good {'morning' if now_et.hour < 12 else 'afternoon'}, {t}. {self.mode.value} mode, "
                    f"equity ${self.broker.equity():,.2f}, watching {', '.join(self.cfg.app.tickers)}. "
                    f"Circuit breakers {self.risk.circuit_status().lower()}.")
        if self.broker.open_positions:
            greeting += f" I have restored {len(self.broker.open_positions)} open paper position(s)."
        self.say(greeting, kind="greeting")
        log.info("orchestrator started (mode=%s, voice=%s, llm=%s)", self.mode.value, backend,
                 self.dialogue.active_provider)
        return notes

    # ------------------------------------------------------------------ helpers

    def say(self, text: str, kind: str = "response", speak: bool = True) -> None:
        self.bus.publish(JarvisBriefing(timestamp=self.clock.now(), text=text, kind=kind, speak=speak))
        if speak:
            self.speech.say(text)

    def portfolio_view(self) -> PortfolioView:
        return PortfolioView(equity=self.broker.equity(), available_cash=self.broker.cash,
                             start_of_day_equity=self.sod_equity, unrealized_pnl=self.broker.unrealized_pnl(),
                             open_positions=self.broker.open_positions)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _roll_day(self) -> None:
        today = self.clock.now_et().date()
        if today != self._sod_day:
            self._sod_day = today
            self.sod_equity = self.broker.equity()
            self.risk.roll_day()
            self._standby_announced = False
            self._halt_announced = False

    async def _execute(self, plan: TradePlan, ctx: TickerContext) -> None:
        q = ctx.quote
        try:
            fill = await self.broker.open_from_plan(plan, q.bid if q else None, q.ask if q else None, ctx.snapshot.price)
        except OrderRejected as exc:
            raise RuntimeError(f"order rejected: {exc}") from exc
        self._last_bar.setdefault(plan.ticker, ctx.last_bar_close - timedelta(minutes=1))
        log.info("paper entry %s %s %d @ %.2f", plan.ticker, plan.direction.value, fill.qty, fill.price)

    # ------------------------------------------------------------------ the cycle

    async def _feed_bars(self) -> int:
        """Feed every new completed 1m bar (in order) for tickers with open positions."""
        fills = 0
        now = self.clock.now()
        for ticker in sorted({p.ticker for p in self.broker.open_positions}):
            since = self._last_bar.get(ticker)
            start = (since + timedelta(minutes=1)) if since else now - timedelta(minutes=5)
            try:
                bars = await self.market.get_bars(ticker, Timeframe.M1, start)
            except ProviderError as exc:
                log.warning("bars for %s unavailable: %s", ticker, exc)
                continue
            for ts, row in bars.iterrows():
                fills += len(await self.broker.on_bar(ticker, Bar(row.open, row.high, row.low, row.close)))
                self._last_bar[ticker] = ts.to_pydatetime()
                if not self.broker.position_for(ticker):
                    break
        return fills

    async def run_cycle(self) -> CycleResult:
        self.cycles += 1
        self._roll_day()
        now = self.clock.now()
        res = CycleResult(at=now)
        status = self.calendar.status(now)
        if self.mode is Mode.PAPER and status is not MarketStatus.OPEN:
            if self.broker.open_positions and self.calendar.session_for(now) is None:
                await self.broker.flatten_all(ExitReason.EOD_FLAT)
            if not self._standby_announced:
                nxt = self.calendar.next_session(now)
                self.say(f"Markets are {status.value.lower().replace('_', '-')}, {self.cfg.app.user_title}. Standing by "
                         f"until {to_et(nxt.open):%A %H:%M} ET.", speak=False)
                self._standby_announced = True
            res.standby = True
            return res
        self._standby_announced = False

        res.fills += await self._feed_bars()

        tickers = self.cfg.app.tickers
        built = await asyncio.gather(*(self.technical.build_context(t) for t in tickers), return_exceptions=True)
        contexts = [c for c in built if isinstance(c, TickerContext)]
        self.contexts = {c.ticker: c for c in contexts}
        bias = await self.technical.macro_bias()
        self.regime = bias.regime
        prices = {c.ticker: c.snapshot.price for c in contexts}
        self.broker.mark(prices)
        for c in contexts:
            self.bus.publish(MarketUpdate(timestamp=now, ticker=c.ticker, price=c.snapshot.price,
                                          snapshot=c.snapshot, regime=bias.regime))

        for setup, ctx in self.technical.scan_contexts(contexts, bias)[: self.cfg.app.max_debates_per_cycle]:
            t = await self.war_room.convene(setup, ctx, bias)
            res.debates.append(t)
            if t.plan.status.value != "VETOED" or setup.on_demand:
                self.speech.say(t.briefing)

        if self.risk.state.halted_for_day and not self._halt_announced:
            self._halt_announced = True
            self.bus.publish(SystemAlert(timestamp=now, level=AlertLevel.CRITICAL, source="risk",
                                         message="daily loss limit reached: new entries halted for the day"))
        await self.db.save_equity_snapshot(self.broker.snapshot())
        await self.db.save_risk_state(self.risk.state)
        return res

    async def run_forever(self, stop: asyncio.Event, interval: float | None = None) -> None:
        interval = interval or (self.cfg.demo.scan_interval_seconds if self.mode is Mode.DEMO
                                else self.cfg.app.scan_interval_seconds)
        while not stop.is_set():
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the session
                log.exception("cycle failed")
                self.bus.publish(SystemAlert(timestamp=self.clock.now(), level=AlertLevel.WARNING,
                                             source="orchestrator", message=f"cycle error: {exc}"))
            sleeper = asyncio.create_task(self.clock.sleep(interval))
            waiter = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()

    # ------------------------------------------------------------------ JarvisContext

    async def status(self) -> StatusReport:
        eq = self.broker.equity()
        day_pnl = self.risk.day_pnl(self.portfolio_view())
        lines = [PositionLine(p.ticker, p.direction.value, p.qty, p.entry_price, p.stop_loss, p.tp1, p.tp2,
                              p.last_price, round(p.unrealized_pnl(), 2)) for p in self.broker.open_positions]
        return StatusReport(mode=self.mode.value, equity=round(eq, 2), day_pnl=round(day_pnl, 2),
                            day_pnl_pct=day_pnl / self.sod_equity if self.sod_equity else 0.0, regime=self.regime,
                            circuit=self.risk.circuit_status(), positions=lines, llm=self.dialogue.active_provider,
                            voice=self.speech.backend_name, market_status=self.calendar.status(self.clock.now()).value)

    async def analyze(self, ticker: str) -> WarRoomTranscript | str:
        ticker = ticker.upper()
        bias = await self.technical.macro_bias()
        res = await self.technical.analyze(ticker, bias)
        if res is None:
            return f"I'm afraid I have no market data for {ticker}, {self.cfg.app.user_title}."
        setup, ctx = res
        return await self.war_room.convene(setup, ctx, bias)

    async def latest_transcript(self, ticker: str | None) -> WarRoomTranscript | None:
        if ticker is None:
            return self.war_room.last_transcript or await self.db.latest_transcript(None)
        return self.war_room.transcripts.get(ticker.upper()) or await self.db.latest_transcript(ticker)

    def pause(self) -> None:
        self.risk.pause()
        self._spawn(self.db.save_risk_state(self.risk.state))

    def resume(self) -> None:
        self.risk.resume()
        self._spawn(self.db.save_risk_state(self.risk.state))

    async def flatten(self) -> list[str]:
        fills = await self.broker.flatten_all(ExitReason.MANUAL_FLATTEN, {t: c.snapshot.price for t, c in self.contexts.items()})
        return [f"{f.ticker} {f.direction.value.lower()} {format_qty(f.ticker, f.qty)} @ {format_price(f.ticker, f.price)} "
                f"(P&L {f.realized_pnl - f.commission:+.2f})"
                for f in fills]

    async def handle_command(self, text: str) -> str:
        try:
            reply = await self.jarvis.handle(text, self)
        except Exception as exc:  # noqa: BLE001
            log.exception("command failed: %s", text)
            reply = f"My apologies, {self.cfg.app.user_title}; that command failed ({exc})."
        multiline = "\n" in reply
        self.say(reply, kind="response", speak=not multiline)
        return reply

    # ------------------------------------------------------------------ shutdown

    async def shutdown(self) -> None:
        """Cancel background work, stop audio, close HTTP clients, flush SQLite."""
        if self._closed:
            return
        self._closed = True
        for t in list(self._bg):
            t.cancel()
        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)
        await self.speech.stop()
        await self.notifications.stop()
        for closer in (self.market.aclose, self.news_provider.aclose, self.dialogue.aclose):
            try:
                await closer()
            except Exception:  # noqa: BLE001
                log.exception("error closing provider")
        try:
            await self.db.save_equity_snapshot(self.broker.snapshot())
            await self.db.save_risk_state(self.risk.state)
        except Exception:  # noqa: BLE001
            log.exception("final persistence failed")
        await self.db.close()
        self.bus.close()
        log.info("orchestrator shut down cleanly")
