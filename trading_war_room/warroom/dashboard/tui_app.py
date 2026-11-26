"""Textual terminal dashboard. A pure EventBus subscriber plus a command bar.

Layout
  Header bar  : mode, equity, day P&L ($ and %), circuit breakers, regime, voice, clock
  Top-left    : watchlist (price, VWAP %, RSI, RVOL) and open paper positions
  Top-right   : live War Room debate, colour-coded by agent
  Bottom      : J.A.R.V.I.S. console and command input (works while scanning runs)

The orchestrator loop runs as a background task on the app's event loop; all
blocking work inside it is already off-loaded to threads, so the UI stays
responsive. Quitting (q / Ctrl+Q / Ctrl+C) or SIGTERM triggers a clean
orchestrator shutdown.
"""

from __future__ import annotations

import asyncio
import logging

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, RichLog, Static

from warroom.core.assets import format_price, format_qty
from warroom.core.clock import to_et
from warroom.core.event_bus import BusClosed
from warroom.models.schemas import (
    AgentRole,
    AlertLevel,
    BaseEvent,
    JarvisBriefing,
    MarketRegime,
    MarketUpdate,
    Stance,
    SystemAlert,
    TradeExecuted,
    WarRoomMessage,
)
from warroom.orchestrator import Orchestrator

log = logging.getLogger(__name__)

AGENT_STYLE = {
    AgentRole.TECHNICAL: "bold cyan",
    AgentRole.NEWS: "bold yellow",
    AgentRole.RISK: "bold red",
    AgentRole.JARVIS: "bold magenta",
}
STANCE_STYLE = {
    Stance.APPROVE: "green", Stance.VETO: "red", Stance.CHALLENGE: "orange1", Stance.DEFEND: "cyan",
    Stance.CAUTION: "yellow", Stance.BULLISH: "green", Stance.BEARISH: "red",
}


class WarRoomApp(App):
    TITLE = "Trading War Room"
    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; background: $boost; color: $text; padding: 0 1; }
    #top { height: 1fr; }
    #left { width: 46%; }
    #watch { height: 1fr; border: round $primary; }
    #positions { height: 1fr; border: round $secondary; }
    #warroom { width: 1fr; border: round $accent; }
    #bottom { height: 14; }
    #jarvis { height: 1fr; border: round magenta; }
    #cmd { height: 3; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("f2", "command('status')", "Status"),
        Binding("f3", "command('transcript')", "Transcript"),
    ]

    def __init__(self, orch: Orchestrator, stop: asyncio.Event | None = None, run_loop: bool = True) -> None:
        super().__init__()
        self.orch = orch
        self.stop_event = stop or asyncio.Event()
        self.run_loop = run_loop
        self._tasks: list[asyncio.Task] = []
        self._rows: dict[str, MarketUpdate] = {}
        self._sub = orch.bus.subscribe(maxsize=2000, name="tui")

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="top"):
            with Vertical(id="left"):
                yield DataTable(id="watch", zebra_stripes=True)
                yield DataTable(id="positions", zebra_stripes=True)
            yield RichLog(id="warroom", wrap=True, markup=False, highlight=False, max_lines=2000)
        with Vertical(id="bottom"):
            yield RichLog(id="jarvis", wrap=True, markup=False, max_lines=1000)
            yield Input(placeholder="J.A.R.V.I.S. >  status | analyze TSLA | why AAPL | transcript | pause | resume | flatten",
                        id="cmd")

    async def on_mount(self) -> None:
        w = self.query_one("#watch", DataTable)
        w.border_title = "Watchlist"
        w.add_columns("Ticker", "Price", "VWAP %", "RSI", "RVOL", "ATR")
        p = self.query_one("#positions", DataTable)
        p.border_title = "Open paper positions"
        p.add_columns("Ticker", "Side", "Qty", "Entry", "Stop", "TP1", "TP2", "Last", "Unreal $")
        self.query_one("#warroom", RichLog).border_title = "Agent War Room"
        self.query_one("#jarvis", RichLog).border_title = "J.A.R.V.I.S."
        self._tasks.append(asyncio.create_task(self._consume(), name="tui-consume"))
        if self.run_loop:
            self._tasks.append(asyncio.create_task(self.orch.run_forever(self.stop_event), name="orchestrator"))
        self.set_interval(1.0, self.refresh_header)
        self.refresh_header()
        self.query_one("#cmd", Input).focus()

    # ------------------------------------------------------------------ rendering

    def refresh_header(self) -> None:
        o = self.orch
        eq = o.broker.equity()
        pnl = o.risk.day_pnl(o.portfolio_view())
        pct = pnl / o.sod_equity if o.sod_equity else 0.0
        circuit = o.risk.circuit_status()
        regime = o.regime.value.replace("_", "-")
        col = "green" if pnl >= 0 else "red"
        ccol = "green" if circuit == "ARMED" else "yellow" if "PAUSED" in circuit or "COOLDOWN" in circuit else "red"
        rcol = {MarketRegime.RISK_ON: "green", MarketRegime.RISK_OFF: "red"}.get(o.regime, "yellow")
        t = Text.assemble(
            (f" {o.mode.value} ", "bold reverse"), "  Equity ", (f"${eq:,.2f}", "bold"), "  Day P&L ",
            (f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} ({pct:+.2%})", f"bold {col}"),
            "  Breakers ", (circuit, f"bold {ccol}"), "  Regime ", (regime, f"bold {rcol}"),
            "  Voice ", ("ON" if o.speech.active else "OFF", "bold"),
            "  LLM ", (o.dialogue.active_provider, "dim"),
            f"  {to_et(o.clock.now()):%Y-%m-%d %H:%M:%S} ET",
        )
        self.query_one("#header", Static).update(t)
        self._render_positions()

    def _render_watch(self) -> None:
        table = self.query_one("#watch", DataTable)
        table.clear()
        for tk in sorted(self._rows):
            s = self._rows[tk].snapshot
            if s is None:
                continue
            vd = s.vwap_distance_pct
            table.add_row(
                tk, f"{s.price:,.2f}",
                Text(f"{vd:+.2%}" if vd is not None else "-", style="green" if (vd or 0) >= 0 else "red"),
                f"{s.rsi14:.0f}" if s.rsi14 is not None else "-",
                Text(f"{s.rvol:.1f}x" if s.rvol is not None else "-", style="bold" if (s.rvol or 0) >= 1.5 else ""),
                f"{s.atr14:.2f}" if s.atr14 is not None else "-",
            )

    def _render_positions(self) -> None:
        table = self.query_one("#positions", DataTable)
        table.clear()
        for p in self.orch.broker.open_positions:
            u = p.unrealized_pnl()
            table.add_row(p.ticker, p.direction.value, str(p.qty), f"{p.entry_price:.2f}", f"{p.stop_loss:.2f}",
                          f"{p.tp1:.2f}", f"{p.tp2:.2f}", f"{p.last_price or 0:.2f}",
                          Text(f"{u:+,.2f}", style="green" if u >= 0 else "red"))

    def _war_line(self, e: WarRoomMessage) -> Text:
        m = e.message
        tgt = "" if m.recipient is AgentRole.ALL else f"→{m.recipient.value}"
        return Text.assemble(
            (f"{to_et(m.timestamp):%H:%M} ", "dim"), (f"R{m.round_number} ", "dim"),
            (f"[{m.sender.value}{tgt}] ", AGENT_STYLE.get(m.sender, "bold")),
            (f"{e.ticker} ", "bold"), (f"({m.stance.value}) ", STANCE_STYLE.get(m.stance, "")), m.message,
        )

    def handle_event(self, e: BaseEvent) -> None:
        if isinstance(e, MarketUpdate):
            self._rows[e.ticker] = e
            if len(self._rows) >= len(self.orch.cfg.app.tickers) or e.ticker == self.orch.cfg.app.tickers[-1]:
                self._render_watch()
        elif isinstance(e, WarRoomMessage):
            log_ = self.query_one("#warroom", RichLog)
            if e.message.round_number == 1 and e.message.sender is AgentRole.TECHNICAL:
                log_.write(Text(f"── War Room: {e.ticker} ──", style="bold"))
            log_.write(self._war_line(e))
        elif isinstance(e, JarvisBriefing):
            style = {"APPROVED": "bold green", "VETOED": "red", "WATCHLIST": "yellow"}.get(e.status or "", "magenta")
            self.query_one("#jarvis", RichLog).write(
                Text.assemble((f"{to_et(e.timestamp):%H:%M} ", "dim"), ("J.A.R.V.I.S.: ", "bold magenta"), (e.text, style)))
        elif isinstance(e, TradeExecuted):
            f = e.fill
            what = "ENTRY" if f.is_entry else f"EXIT {f.reason.value if f.reason else ''}"
            pnl = "" if f.is_entry else f"  P&L {f.realized_pnl - f.commission:+,.2f}"
            self.query_one("#jarvis", RichLog).write(Text(
                f"{to_et(f.timestamp):%H:%M} [PAPER {what}] {f.ticker} {e.position.direction.value} {format_qty(f.ticker, f.qty)} @ {format_price(f.ticker, f.price)}{pnl}",
                style="bold green" if f.is_entry or (f.realized_pnl - f.commission) >= 0 else "bold red"))
            self._render_positions()
        elif isinstance(e, SystemAlert):
            style = {AlertLevel.CRITICAL: "bold white on red", AlertLevel.WARNING: "yellow"}.get(e.level, "dim")
            self.query_one("#jarvis", RichLog).write(Text(f"[{e.level.value}] {e.source}: {e.message}", style=style))

    async def _consume(self) -> None:
        try:
            async for e in self._sub:
                try:
                    self.handle_event(e)
                except Exception:  # noqa: BLE001 - a rendering glitch must not stop the feed
                    log.exception("render failed for %s", type(e).__name__)
        except BusClosed:
            pass

    # ------------------------------------------------------------------ input

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in ("quit", "exit"):
            await self.action_quit()
            return
        self.query_one("#jarvis", RichLog).write(Text(f"> {text}", style="bold"))
        self._tasks.append(asyncio.create_task(self.orch.handle_command(text)))

    async def action_command(self, text: str) -> None:
        self._tasks.append(asyncio.create_task(self.orch.handle_command(text)))

    async def action_quit(self) -> None:
        await self.shutdown()
        self.exit()

    async def shutdown(self) -> None:
        self.stop_event.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._sub.unsubscribe()
        await self.orch.shutdown()
