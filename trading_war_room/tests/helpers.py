"""Builders for agent-level tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from tests.conftest import D, et
from warroom.agents.risk_agent import PortfolioView
from warroom.agents.technical_agent import TickerContext
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import (
    Direction, IndicatorSnapshot, NewsAssessment, Position, Quote, SetupType, Severity, TradeSetup,
)


def snapshot(ticker="AAPL", price=100.0, atr=1.0, swing_low=99.0, swing_high=101.0, ts=None, **kw) -> IndicatorSnapshot:
    base = dict(ticker=ticker, timestamp=ts or et(10, 59), price=price, vwap=price * 0.998, ema9=price * 0.999,
                ema21=price * 0.997, ema50=price * 0.99, rsi14=55.0, atr14=atr, rvol=1.6,
                swing_low=swing_low, swing_high=swing_high)
    base.update(kw)
    return IndicatorSnapshot(**base)


def context(snap: IndicatorSnapshot | None = None, *, now: datetime | None = None, spread_pct: float = 0.0004,
            real_quotes=True, prior_high=None, prior_low=None, age_s: float = 30, cal=None) -> TickerContext:
    snap = snap or snapshot()
    cal = cal or MarketCalendar()
    now = now or et(11, 0)
    half = snap.price * spread_pct / 2
    q = Quote(ticker=snap.ticker, timestamp=now, bid=round(snap.price - half, 4), ask=round(snap.price + half, 4),
              last=snap.price) if real_quotes else None
    return TickerContext(ticker=snap.ticker, snapshot=snap, session=cal.session(D), quote=q,
                         has_real_quotes=real_quotes, last_bar_close=now - timedelta(seconds=age_s),
                         prior_high=prior_high, prior_low=prior_low)


def setup(ticker="AAPL", direction=Direction.LONG, price=100.0, snap=None, confidence=0.7, qualified=True,
          setup_type=SetupType.ORB_BREAKOUT) -> TradeSetup:
    return TradeSetup(ticker=ticker, direction=direction, setup_type=setup_type, detected_at=et(11, 0),
                      trigger_price=price, confidence=confidence, rationale=["test"],
                      indicators=snap or snapshot(ticker, price), qualified=qualified)


def news(ticker="AAPL", sentiment=0.0, severity=Severity.LOW, blackout=False, conflict=False) -> NewsAssessment:
    return NewsAssessment(ticker=ticker, sentiment=sentiment, severity=severity, in_blackout=blackout,
                          blackout_reason="FOMC rate decision in 5m" if blackout else None, conflicts_with_direction=conflict)


def portfolio(equity=100_000.0, cash=100_000.0, sod=100_000.0, unrealized=0.0, positions=None) -> PortfolioView:
    return PortfolioView(equity=equity, available_cash=cash, start_of_day_equity=sod, unrealized_pnl=unrealized,
                         open_positions=positions or [])


def position(ticker="MSFT", direction=Direction.LONG, qty=10) -> Position:
    return Position(ticker=ticker, direction=direction, entry_price=100, stop_loss=99 if direction is Direction.LONG else 101,
                    tp1=102 if direction is Direction.LONG else 98, tp2=103 if direction is Direction.LONG else 97,
                    initial_qty=qty, qty=qty, opened_at=et(10, 0))
