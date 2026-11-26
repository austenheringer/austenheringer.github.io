"""SQLite persistence (WAL, serialized access) and startup state recovery.

Concurrency model: one ``sqlite3`` connection, every operation executed in a
worker thread via ``asyncio.to_thread`` while holding a single ``asyncio.Lock``.
This serializes writes (no "database is locked") and keeps the event loop free.

Atomicity: a fill, the position it changes and the resulting account cash are
written in ONE transaction (:meth:`Database.record_fill`), so a crash can
never leave cash and positions disagreeing.

Recovery is *derived from the ledger* (fills + positions), not from a cached
counter that could be stale: today's realized P&L, the consecutive-loss
streak, the PDT round-trip window and open positions are recomputed on every
start. Flags that cannot be derived (manual pause, daily halt) are restored
from the persisted RiskState and OR-ed in, so a restart can only make the
system *more* cautious, never less.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

from warroom.core.clock import ET, ensure_utc, to_et
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import (
    AccountSnapshot,
    AgentMessage,
    Fill,
    Position,
    PositionStatus,
    RiskState,
    TradeSetup,
    WarRoomTranscript,
)

log = logging.getLogger(__name__)
T = TypeVar("T")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, ticker TEXT NOT NULL, direction TEXT NOT NULL, setup_type TEXT NOT NULL,
    detected_at TEXT NOT NULL, confidence REAL NOT NULL, on_demand INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_signals_time ON signals(detected_at);
CREATE TABLE IF NOT EXISTS debates (
    debate_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, direction TEXT NOT NULL, status TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT NOT NULL, briefing TEXT NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_debates_ticker ON debates(ticker, finished_at);
CREATE TABLE IF NOT EXISTS agent_messages (
    id TEXT PRIMARY KEY, debate_id TEXT NOT NULL, sender TEXT NOT NULL, recipient TEXT NOT NULL,
    round_number INTEGER NOT NULL, stance TEXT NOT NULL, message TEXT NOT NULL,
    data_payload TEXT NOT NULL, timestamp TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_messages_debate ON agent_messages(debate_id, timestamp);
CREATE TABLE IF NOT EXISTS risk_vetoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, debate_id TEXT, ticker TEXT NOT NULL, direction TEXT NOT NULL,
    reasons TEXT NOT NULL, timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, ticker TEXT NOT NULL, direction TEXT NOT NULL, status TEXT NOT NULL,
    opened_at TEXT NOT NULL, closed_at TEXT, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_positions_status ON positions(status);
CREATE TABLE IF NOT EXISTS fills (
    id TEXT PRIMARY KEY, position_id TEXT NOT NULL, ticker TEXT NOT NULL, is_entry INTEGER NOT NULL,
    qty INTEGER NOT NULL, price REAL NOT NULL, commission REAL NOT NULL, realized_pnl REAL NOT NULL,
    reason TEXT, timestamp TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_fills_time ON fills(timestamp);
CREATE TABLE IF NOT EXISTS account_state (
    id INTEGER PRIMARY KEY CHECK (id = 1), cash REAL NOT NULL, reserved_margin REAL NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL,
    reserved_margin REAL NOT NULL, realized_pnl_today REAL NOT NULL, unrealized_pnl REAL NOT NULL,
    open_positions INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS risk_state (trading_date TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_usage (day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
"""


def ts(dt: datetime) -> str:
    """Fixed-width ISO UTC string: lexicographic order == chronological order."""
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def et_day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time(0, 0), tzinfo=ET)
    return ensure_utc(start), ensure_utc(start + timedelta(days=1))


# --------------------------------------------------------------------------- recovery (pure)


@dataclass
class RecoveredState:
    risk_state: RiskState
    open_positions: list[Position] = field(default_factory=list)
    cash: float | None = None             # None -> no account history; use starting cash
    reserved_margin: float = 0.0
    start_of_day_equity: float | None = None
    notes: list[str] = field(default_factory=list)


def position_net_pnl(p: Position) -> float:
    return p.realized_pnl - p.commissions


def derive_risk_state(
    today: date,
    closed_positions: list[Position],
    fills_today: list[Fill],
    persisted: RiskState | None,
    calendar: MarketCalendar,
    consecutive_loss_limit: int,
    cooldown_minutes: int,
) -> RiskState:
    """Rebuild circuit-breaker state for ``today`` (ET) from the ledger."""
    realized = sum(f.realized_pnl - f.commission for f in fills_today)

    todays_closed = sorted(
        (p for p in closed_positions if p.closed_at and to_et(p.closed_at).date() == today),
        key=lambda p: p.closed_at,  # type: ignore[arg-type,return-value]
    )
    streak = 0
    last_loss_at = None
    for p in reversed(todays_closed):
        if position_net_pnl(p) < 0:
            streak += 1
            last_loss_at = last_loss_at or p.closed_at
        else:
            break
    cooldown_until = None
    if streak >= consecutive_loss_limit and last_loss_at is not None:
        cooldown_until = last_loss_at + timedelta(minutes=cooldown_minutes)

    window = set(calendar.previous_trading_days(today, 5))
    day_trades = sorted(
        to_et(p.closed_at).date()  # type: ignore[arg-type]
        for p in closed_positions
        if p.closed_at
        and to_et(p.opened_at).date() == to_et(p.closed_at).date()
        and to_et(p.closed_at).date() in window
    )

    state = RiskState(
        trading_date=today,
        realized_pnl_today=round(realized, 2),
        consecutive_losses=streak,
        last_loss_at=last_loss_at,
        cooldown_until=cooldown_until,
        day_trade_dates=day_trades,
    )
    if persisted is not None and persisted.trading_date == today:
        state.halted_for_day = persisted.halted_for_day
        state.entries_paused = persisted.entries_paused
        if persisted.cooldown_until and (state.cooldown_until is None or persisted.cooldown_until > state.cooldown_until):
            state.cooldown_until = persisted.cooldown_until
    elif persisted is not None:
        # manual pause carries across days; a daily halt does not
        state.entries_paused = persisted.entries_paused
    return state


# --------------------------------------------------------------------------- database


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    # ---- lifecycle ----------------------------------------------------------

    async def open(self) -> None:
        if self._conn is not None:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        def _open() -> sqlite3.Connection:
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10)
            conn.row_factory = sqlite3.Row
            mode = conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
            if self.path != ":memory:" and str(mode).lower() != "wal":
                log.warning("SQLite did not enter WAL mode (got %s)", mode)
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            return conn

        self._conn = await asyncio.to_thread(_open)

    async def close(self) -> None:
        """Checkpoint the WAL into the main file and close."""
        if self._conn is None:
            return
        async with self._lock:
            conn, self._conn = self._conn, None

            def _close() -> None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                finally:
                    conn.close()

            await asyncio.to_thread(_close)

    async def __aenter__(self) -> "Database":
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if self._conn is None:
            raise RuntimeError("database is not open")
        async with self._lock:
            conn = self._conn
            return await asyncio.to_thread(fn, conn)

    async def _tx(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def _wrapped(conn: sqlite3.Connection) -> T:
            conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return out

        return await self._run(_wrapped)

    # ---- writes ---------------------------------------------------------------

    async def save_signal(self, s: TradeSetup) -> None:
        await self._tx(lambda c: c.execute(
            "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?)",
            (s.id, s.ticker, s.direction.value, s.setup_type.value, ts(s.detected_at), s.confidence,
             int(s.on_demand), s.model_dump_json()),
        ))

    @staticmethod
    def _insert_message(c: sqlite3.Connection, m: AgentMessage) -> None:
        c.execute(
            "INSERT OR REPLACE INTO agent_messages VALUES (?,?,?,?,?,?,?,?,?)",
            (m.id, m.debate_id, m.sender.value, m.recipient.value, m.round_number, m.stance.value, m.message,
             json.dumps(m.data_payload, default=str), ts(m.timestamp)),
        )

    async def save_message(self, m: AgentMessage) -> None:
        await self._tx(lambda c: self._insert_message(c, m))

    async def save_transcript(self, t: WarRoomTranscript) -> None:
        def _do(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT OR REPLACE INTO debates VALUES (?,?,?,?,?,?,?,?)",
                (t.debate_id, t.ticker, t.direction.value, t.plan.status.value, ts(t.started_at),
                 ts(t.finished_at), t.briefing, t.model_dump_json()),
            )
            for m in t.messages:
                self._insert_message(c, m)
            if t.plan.veto_reasons:
                c.execute(
                    "INSERT INTO risk_vetoes(debate_id, ticker, direction, reasons, timestamp) VALUES (?,?,?,?,?)",
                    (t.debate_id, t.ticker, t.direction.value, json.dumps(t.plan.veto_reasons), ts(t.finished_at)),
                )

        await self._tx(_do)

    async def upsert_position(self, p: Position) -> None:
        await self._tx(lambda c: self._upsert_position(c, p))

    @staticmethod
    def _upsert_position(c: sqlite3.Connection, p: Position) -> None:
        c.execute(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)",
            (p.id, p.ticker, p.direction.value, p.status.value, ts(p.opened_at),
             ts(p.closed_at) if p.closed_at else None, p.model_dump_json()),
        )

    async def record_fill(self, fill: Fill, position: Position, cash: float, reserved_margin: float) -> None:
        """Atomically persist a fill, the updated position and account cash."""

        def _do(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                (fill.id, fill.position_id, fill.ticker, int(fill.is_entry), fill.qty, fill.price, fill.commission,
                 fill.realized_pnl, fill.reason.value if fill.reason else None, ts(fill.timestamp)),
            )
            self._upsert_position(c, position)
            c.execute(
                "INSERT INTO account_state(id, cash, reserved_margin, updated_at) VALUES (1,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET cash=excluded.cash, reserved_margin=excluded.reserved_margin, "
                "updated_at=excluded.updated_at",
                (cash, reserved_margin, ts(fill.timestamp)),
            )

        await self._tx(_do)

    async def save_equity_snapshot(self, s: AccountSnapshot) -> None:
        await self._tx(lambda c: c.execute(
            "INSERT INTO equity_snapshots(timestamp, equity, cash, reserved_margin, realized_pnl_today, "
            "unrealized_pnl, open_positions) VALUES (?,?,?,?,?,?,?)",
            (ts(s.timestamp), s.equity, s.cash, s.reserved_margin, s.realized_pnl_today, s.unrealized_pnl,
             s.open_positions),
        ))

    async def save_risk_state(self, state: RiskState) -> None:
        await self._tx(lambda c: c.execute(
            "INSERT OR REPLACE INTO risk_state VALUES (?,?)", (state.trading_date.isoformat(), state.model_dump_json())
        ))

    async def set_llm_usage(self, day: date, calls: int) -> None:
        await self._tx(lambda c: c.execute("INSERT OR REPLACE INTO llm_usage VALUES (?,?)", (day.isoformat(), calls)))

    # ---- reads ----------------------------------------------------------------

    async def get_llm_usage(self, day: date) -> int:
        row = await self._run(lambda c: c.execute("SELECT calls FROM llm_usage WHERE day=?", (day.isoformat(),)).fetchone())
        return int(row["calls"]) if row else 0

    async def load_open_positions(self) -> list[Position]:
        rows = await self._run(lambda c: c.execute(
            "SELECT payload FROM positions WHERE status=? ORDER BY opened_at", (PositionStatus.OPEN.value,)).fetchall())
        return [Position.model_validate_json(r["payload"]) for r in rows]

    async def load_closed_positions_since(self, since: datetime) -> list[Position]:
        rows = await self._run(lambda c: c.execute(
            "SELECT payload FROM positions WHERE status=? AND closed_at >= ? ORDER BY closed_at",
            (PositionStatus.CLOSED.value, ts(since))).fetchall())
        return [Position.model_validate_json(r["payload"]) for r in rows]

    async def load_fills_between(self, start: datetime, end: datetime) -> list[Fill]:
        rows = await self._run(lambda c: c.execute(
            "SELECT f.*, p.direction AS pdir FROM fills f JOIN positions p ON p.id = f.position_id "
            "WHERE f.timestamp >= ? AND f.timestamp < ? ORDER BY f.timestamp", (ts(start), ts(end))).fetchall())
        return [
            Fill(id=r["id"], position_id=r["position_id"], ticker=r["ticker"], direction=r["pdir"],
                 is_entry=bool(r["is_entry"]), qty=r["qty"], price=r["price"], commission=r["commission"],
                 realized_pnl=r["realized_pnl"], reason=r["reason"], timestamp=datetime.fromisoformat(r["timestamp"]))
            for r in rows
        ]

    async def load_account_state(self) -> tuple[float, float] | None:
        row = await self._run(lambda c: c.execute("SELECT cash, reserved_margin FROM account_state WHERE id=1").fetchone())
        return (float(row["cash"]), float(row["reserved_margin"])) if row else None

    async def load_risk_state(self, latest_on_or_before: date) -> RiskState | None:
        row = await self._run(lambda c: c.execute(
            "SELECT payload FROM risk_state WHERE trading_date <= ? ORDER BY trading_date DESC LIMIT 1",
            (latest_on_or_before.isoformat(),)).fetchone())
        return RiskState.model_validate_json(row["payload"]) if row else None

    async def start_of_day_equity(self, day: date) -> float | None:
        """Equity of the last snapshot before today's ET midnight (yesterday's close)."""
        start, _ = et_day_bounds(day)
        row = await self._run(lambda c: c.execute(
            "SELECT equity FROM equity_snapshots WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            (ts(start),)).fetchone())
        return float(row["equity"]) if row else None

    async def latest_transcript(self, ticker: str | None = None) -> WarRoomTranscript | None:
        def _q(c: sqlite3.Connection) -> Any:
            if ticker:
                return c.execute("SELECT payload FROM debates WHERE ticker=? ORDER BY finished_at DESC LIMIT 1",
                                 (ticker.upper(),)).fetchone()
            return c.execute("SELECT payload FROM debates ORDER BY finished_at DESC LIMIT 1").fetchone()

        row = await self._run(_q)
        return WarRoomTranscript.model_validate_json(row["payload"]) if row else None

    async def count(self, table: str) -> int:
        if table not in {"signals", "debates", "agent_messages", "risk_vetoes", "positions", "fills", "equity_snapshots"}:
            raise ValueError(table)
        row = await self._run(lambda c: c.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone())  # noqa: S608
        return int(row["n"])

    # ---- startup recovery -----------------------------------------------------

    async def recover(
        self,
        now: datetime,
        calendar: MarketCalendar,
        consecutive_loss_limit: int,
        cooldown_minutes: int,
    ) -> RecoveredState:
        today = to_et(now).date()
        day_start, day_end = et_day_bounds(today)
        lookback_start, _ = et_day_bounds(min(calendar.previous_trading_days(today, 5)))
        closed = await self.load_closed_positions_since(lookback_start)
        fills_today = await self.load_fills_between(day_start, day_end)
        persisted = await self.load_risk_state(today)
        state = derive_risk_state(today, closed, fills_today, persisted, calendar,
                                  consecutive_loss_limit, cooldown_minutes)
        open_positions = await self.load_open_positions()
        acct = await self.load_account_state()
        rec = RecoveredState(
            risk_state=state,
            open_positions=open_positions,
            cash=acct[0] if acct else None,
            reserved_margin=acct[1] if acct else 0.0,
            start_of_day_equity=await self.start_of_day_equity(today),
        )
        if open_positions:
            rec.notes.append(f"restored {len(open_positions)} open paper position(s)")
        if state.realized_pnl_today:
            rec.notes.append(f"restored realized P&L today: {state.realized_pnl_today:+.2f}")
        if state.consecutive_losses:
            rec.notes.append(f"restored loss streak: {state.consecutive_losses}")
        if state.day_trade_dates:
            rec.notes.append(f"restored {len(state.day_trade_dates)} day trade(s) in the 5-day PDT window")
        stale = [p for p in open_positions if to_et(p.opened_at).date() < today]
        if stale:
            rec.notes.append(
                f"WARNING: {len(stale)} position(s) from a previous session are still open "
                "(the app was down at EOD); they will be flattened at the next tick"
            )
        await self.save_risk_state(state)
        return rec
