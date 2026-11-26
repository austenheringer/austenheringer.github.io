"""Typed domain models and event-bus events.

Conventions
-----------
* Every timestamp is an aware UTC datetime (``UTCDatetime``). Naive datetimes
  are rejected at validation time.
* No model has a wall-clock ``default_factory`` for timestamps: callers must
  pass ``clock.now()``. This keeps replay deterministic.
* Messages and events are frozen (immutable once published). Positions are
  mutable because the broker updates them in place.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from warroom.core.clock import ensure_utc

UTCDatetime = Annotated[datetime, AfterValidator(ensure_utc)]
PositivePrice = Annotated[float, Field(gt=0)]


def new_id() -> str:
    return uuid4().hex[:12]


# --------------------------------------------------------------------------- enums


class Mode(StrEnum):
    DEMO = "DEMO"
    PAPER = "PAPER"
    REPLAY = "REPLAY"
    SMOKE = "SMOKE"


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"

    @property
    def minutes(self) -> int:
        return int(self.value[:-1])


class SetupType(StrEnum):
    VWAP_BOUNCE = "VWAP_BOUNCE"
    VWAP_REJECTION = "VWAP_REJECTION"
    ORB_BREAKOUT = "ORB_BREAKOUT"
    ORB_BREAKDOWN = "ORB_BREAKDOWN"
    EMA_PULLBACK = "EMA_PULLBACK"
    RVOL_MOMENTUM = "RVOL_MOMENTUM"


class AgentRole(StrEnum):
    TECHNICAL = "TECHNICAL"
    NEWS = "NEWS"
    RISK = "RISK"
    JARVIS = "JARVIS"
    ALL = "ALL"


class Stance(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"
    CHALLENGE = "CHALLENGE"
    DEFEND = "DEFEND"
    APPROVE = "APPROVE"
    VETO = "VETO"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MacroEventKind(StrEnum):
    FOMC = "FOMC"
    CPI = "CPI"
    NFP = "NFP"
    EARNINGS = "EARNINGS"
    HALT = "HALT"
    OTHER = "OTHER"


class PlanStatus(StrEnum):
    APPROVED = "APPROVED"
    WATCHLIST = "WATCHLIST"
    VETOED = "VETOED"


class PositionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    TP1 = "TP1"
    TP2 = "TP2"
    EOD_FLAT = "EOD_FLAT"
    MANUAL_FLATTEN = "MANUAL_FLATTEN"


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MarketRegime(StrEnum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- level math


def validate_levels(
    direction: Direction, entry: float, stop: float, tp1: float, tp2: float
) -> str | None:
    """Return a human-readable reason if levels are invalid, else ``None``.

    LONG:  stop < entry < tp1 < tp2
    SHORT: stop > entry > tp1 > tp2
    Shared by the Risk agent (to veto) and TradePlan (last line of defence).
    """
    for name, v in (("entry", entry), ("stop", stop), ("tp1", tp1), ("tp2", tp2)):
        if not (v > 0) or v != v or v in (float("inf"), float("-inf")):  # rejects NaN/inf/<=0
            return f"{name} price is invalid ({v})"
    if abs(entry - stop) <= 0:
        return "zero distance between entry and stop"
    if direction is Direction.LONG:
        if not (stop < entry < tp1 < tp2):
            return (
                f"LONG level ordering violated: need stop<entry<tp1<tp2, "
                f"got {stop:.4f}/{entry:.4f}/{tp1:.4f}/{tp2:.4f}"
            )
    else:
        if not (stop > entry > tp1 > tp2):
            return (
                f"SHORT level ordering violated: need stop>entry>tp1>tp2, "
                f"got {stop:.4f}/{entry:.4f}/{tp1:.4f}/{tp2:.4f}"
            )
    return None


# --------------------------------------------------------------------------- market data


class Candle(_Frozen):
    ticker: str
    timeframe: Timeframe
    timestamp: UTCDatetime  # bar OPEN time
    open: PositivePrice
    high: PositivePrice
    low: PositivePrice
    close: PositivePrice
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_ohlc(self) -> "Candle":
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"inconsistent OHLC for {self.ticker} @ {self.timestamp}")
        if self.low > self.high:
            raise ValueError("low > high")
        return self


class Quote(_Frozen):
    ticker: str
    timestamp: UTCDatetime
    bid: PositivePrice
    ask: PositivePrice
    last: PositivePrice

    @model_validator(mode="after")
    def _check_book(self) -> "Quote":
        if self.ask < self.bid:
            raise ValueError(f"crossed quote for {self.ticker}: bid {self.bid} > ask {self.ask}")
        return self

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid


class IndicatorSnapshot(_Frozen):
    ticker: str
    timestamp: UTCDatetime  # time of the latest candle used
    price: PositivePrice
    vwap: float | None = None
    ema9: float | None = None
    ema21: float | None = None
    ema50: float | None = None
    rsi14: float | None = Field(default=None, ge=0, le=100)
    atr14: float | None = Field(default=None, ge=0)
    rvol: float | None = Field(default=None, ge=0)
    orb_high: float | None = None
    orb_low: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    spread_pct: float | None = Field(default=None, ge=0)
    swing_high: float | None = None
    swing_low: float | None = None

    @property
    def vwap_distance_pct(self) -> float | None:
        if not self.vwap:
            return None
        return (self.price - self.vwap) / self.vwap


class TradeSetup(_Frozen):
    id: str = Field(default_factory=new_id)
    ticker: str
    direction: Direction
    setup_type: SetupType
    detected_at: UTCDatetime
    trigger_price: PositivePrice
    confidence: float = Field(ge=0, le=1)
    rationale: list[str] = Field(default_factory=list)
    indicators: IndicatorSnapshot
    macro_bias: Direction | None = None
    on_demand: bool = False


# --------------------------------------------------------------------------- news


class NewsItem(_Frozen):
    id: str = Field(default_factory=new_id)
    ticker: str | None = None
    headline: str
    source: str
    published_at: UTCDatetime
    url: str | None = None
    sentiment: float = Field(default=0.0, ge=-1, le=1)
    severity: Severity = Severity.LOW


class MacroEvent(_Frozen):
    name: str
    kind: MacroEventKind
    scheduled_at: UTCDatetime
    severity: Severity
    ticker: str | None = None  # set for earnings / halts


class NewsAssessment(_Frozen):
    ticker: str
    sentiment: float = Field(ge=-1, le=1)
    severity: Severity
    headlines: list[NewsItem] = Field(default_factory=list)
    in_blackout: bool = False
    blackout_reason: str | None = None
    blackout_until: UTCDatetime | None = None
    conflicts_with_direction: bool = False


# --------------------------------------------------------------------------- risk


class RiskAssessment(_Frozen):
    ticker: str
    direction: Direction
    approved: bool
    veto_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    entry: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    risk_shares: int = 0
    buying_power_shares: int = 0
    position_size: int = 0
    dollar_risk: float = 0.0
    reward_risk: float | None = None

    @model_validator(mode="after")
    def _approved_needs_levels(self) -> "RiskAssessment":
        if self.approved:
            if self.veto_reasons:
                raise ValueError("an approved assessment cannot carry veto reasons")
            if None in (self.entry, self.stop_loss, self.tp1, self.tp2):
                raise ValueError("approved assessment requires all levels")
            reason = validate_levels(self.direction, self.entry, self.stop_loss, self.tp1, self.tp2)  # type: ignore[arg-type]
            if reason:
                raise ValueError(reason)
            if self.position_size < 1:
                raise ValueError("approved assessment requires position_size >= 1")
        return self


class RiskState(_Mutable):
    """Persisted circuit-breaker state, reloaded on startup so a restart
    cannot reset the daily loss limit, loss streak or PDT count."""

    trading_date: date
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    last_loss_at: UTCDatetime | None = None
    cooldown_until: UTCDatetime | None = None
    halted_for_day: bool = False
    day_trade_dates: list[date] = Field(default_factory=list)  # one entry per round trip
    entries_paused: bool = False


# --------------------------------------------------------------------------- war room


class AgentMessage(_Frozen):
    id: str = Field(default_factory=new_id)
    debate_id: str
    sender: AgentRole
    recipient: AgentRole = AgentRole.ALL
    round_number: int = Field(ge=1, le=3)
    stance: Stance
    message: str = Field(min_length=1)
    data_payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: UTCDatetime


class TradePlan(_Frozen):
    id: str = Field(default_factory=new_id)
    debate_id: str
    ticker: str
    direction: Direction
    setup_type: SetupType
    status: PlanStatus
    entry: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    position_size: int = Field(default=0, ge=0)
    dollar_risk: float = Field(default=0.0, ge=0)
    reward_risk: float | None = None
    veto_reasons: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: UTCDatetime

    @model_validator(mode="after")
    def _approved_is_executable(self) -> "TradePlan":
        if self.status is PlanStatus.APPROVED:
            if None in (self.entry, self.stop_loss, self.tp1, self.tp2):
                raise ValueError("APPROVED plan requires entry, stop and both targets")
            reason = validate_levels(self.direction, self.entry, self.stop_loss, self.tp1, self.tp2)  # type: ignore[arg-type]
            if reason:
                raise ValueError(f"APPROVED plan has invalid levels: {reason}")
            if self.position_size < 1:
                raise ValueError("APPROVED plan requires position_size >= 1")
            if self.veto_reasons:
                raise ValueError("APPROVED plan cannot carry veto reasons")
        return self


class WarRoomTranscript(_Frozen):
    debate_id: str
    ticker: str
    direction: Direction
    started_at: UTCDatetime
    finished_at: UTCDatetime
    messages: list[AgentMessage]
    plan: TradePlan
    briefing: str = ""

    @model_validator(mode="after")
    def _three_rounds(self) -> "WarRoomTranscript":
        rounds = {m.round_number for m in self.messages}
        if rounds != {1, 2, 3}:
            raise ValueError(f"transcript must contain rounds 1-3, got {sorted(rounds)}")
        if any(m.debate_id != self.debate_id for m in self.messages):
            raise ValueError("message debate_id mismatch")
        return self


# --------------------------------------------------------------------------- execution


class Position(_Mutable):
    id: str = Field(default_factory=new_id)
    plan_id: str | None = None
    ticker: str
    direction: Direction
    entry_price: PositivePrice
    stop_loss: PositivePrice
    tp1: PositivePrice
    tp2: PositivePrice
    initial_qty: int = Field(ge=1)
    qty: int = Field(ge=0)
    tp1_hit: bool = False
    reserved_margin: float = Field(default=0.0, ge=0)
    realized_pnl: float = 0.0
    commissions: float = Field(default=0.0, ge=0)
    status: PositionStatus = PositionStatus.OPEN
    opened_at: UTCDatetime
    closed_at: UTCDatetime | None = None
    exit_reason: ExitReason | None = None
    last_price: float | None = None

    def unrealized_pnl(self, price: float | None = None) -> float:
        px = price if price is not None else self.last_price
        if px is None or self.qty == 0:
            return 0.0
        return (px - self.entry_price) * self.qty * self.direction.sign


class Fill(_Frozen):
    id: str = Field(default_factory=new_id)
    position_id: str
    ticker: str
    direction: Direction  # direction of the POSITION, not the order side
    is_entry: bool
    qty: int = Field(ge=1)
    price: PositivePrice
    commission: float = Field(ge=0)
    realized_pnl: float = 0.0
    reason: ExitReason | None = None
    timestamp: UTCDatetime


class AccountSnapshot(_Frozen):
    timestamp: UTCDatetime
    equity: float
    cash: float
    reserved_margin: float
    realized_pnl_today: float
    unrealized_pnl: float
    open_positions: int


# --------------------------------------------------------------------------- events


class BaseEvent(_Frozen):
    timestamp: UTCDatetime


class MarketUpdate(BaseEvent):
    ticker: str
    price: PositivePrice
    snapshot: IndicatorSnapshot | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN


class WarRoomMessage(BaseEvent):
    message: AgentMessage
    ticker: str


class JarvisBriefing(BaseEvent):
    text: str
    debate_id: str | None = None
    kind: str = "briefing"  # briefing | response | greeting
    speak: bool = True


class TradeExecuted(BaseEvent):
    fill: Fill
    position: Position


class SystemAlert(BaseEvent):
    level: AlertLevel
    source: str
    message: str


EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    MarketUpdate,
    WarRoomMessage,
    JarvisBriefing,
    TradeExecuted,
    SystemAlert,
)
