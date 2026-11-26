"""News, Catalyst & Macro Agent.

Produces a :class:`NewsAssessment` per (ticker, direction):

* sentiment: recency-weighted mean of the ticker's headlines (half-life 2h),
  so a stale morning upgrade does not outweigh a fresh downgrade;
* severity: the highest severity seen in the last 4 hours;
* blackout: no new entries from ``blackout_minutes_before`` a HIGH-impact
  scheduled event until ``blackout_minutes_after`` it, and for a halted
  ticker from the halt until ``blackout_minutes_after`` its resumption
  (an ongoing halt blacks out the rest of the session);
* conflict: sentiment against the trade direction beyond the threshold.

:func:`blackout_status` is pure so it can be tested without providers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from warroom.config import AppConfig
from warroom.core.assets import AssetRouter
from warroom.core.clock import Clock, ensure_utc
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import (
    Direction,
    MacroEvent,
    MacroEventKind,
    NewsAssessment,
    NewsItem,
    Severity,
)
from warroom.providers.base import NewsProvider, ProviderError

log = logging.getLogger(__name__)

_SEV_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}


@dataclass
class BlackoutResult:
    active: bool
    reason: str | None = None
    until: datetime | None = None
    upcoming: MacroEvent | None = None  # next HIGH event outside the window but within the hour


def blackout_status(
    now: datetime,
    events: list[MacroEvent],
    ticker: str,
    before_min: int,
    after_min: int,
    session_close: datetime | None = None,
) -> BlackoutResult:
    now = ensure_utc(now)
    before, after = timedelta(minutes=before_min), timedelta(minutes=after_min)
    active: list[tuple[datetime, str]] = []
    upcoming: MacroEvent | None = None
    for e in events:
        if e.ticker is not None and e.ticker != ticker:
            continue
        if e.kind is MacroEventKind.HALT:
            if e.ongoing:
                end = session_close or (e.scheduled_at + timedelta(hours=8))
                if e.scheduled_at <= now:
                    active.append((end, f"{e.name}: trading halted, no resumption announced"))
                continue
            end = (e.ends_at or e.scheduled_at) + after
            if e.scheduled_at <= now <= end:
                active.append((end, f"{e.name}: within {after_min}m of the halt/resumption"))
            continue
        if e.severity is not Severity.HIGH:
            continue
        start, end = e.scheduled_at - before, e.scheduled_at + after
        if start <= now <= end:
            when = "in" if now < e.scheduled_at else "since"
            mins = abs(int((e.scheduled_at - now).total_seconds() // 60))
            active.append((end, f"{e.name} {when} {mins}m (blackout {before_min}m before / {after_min}m after)"))
        elif now < start <= now + timedelta(minutes=60) and (upcoming is None or e.scheduled_at < upcoming.scheduled_at):
            upcoming = e
    if not active:
        return BlackoutResult(False, upcoming=upcoming)
    until, reason = max(active, key=lambda x: x[0])
    return BlackoutResult(True, reason=reason, until=until, upcoming=upcoming)


def weighted_sentiment(items: list[NewsItem], now: datetime, half_life_minutes: float = 120.0) -> float:
    if not items:
        return 0.0
    num = den = 0.0
    for it in items:
        age_min = max(0.0, (ensure_utc(now) - it.published_at).total_seconds() / 60)
        w = math.pow(0.5, age_min / half_life_minutes) * (1 + _SEV_RANK[it.severity] * 0.5)
        num += it.sentiment * w
        den += w
    return round(num / den, 3) if den else 0.0


class NewsAgent:
    def __init__(self, cfg: AppConfig, news: NewsProvider, calendar: MarketCalendar, clock: Clock,
                 router: AssetRouter | None = None) -> None:
        self.cfg = cfg
        self.n = cfg.news
        self.news = news
        self.calendar = calendar
        self.clock = clock
        self.router = router or AssetRouter(calendar, cfg.crypto)
        self.last_error: str | None = None

    async def events_window(self, ticker: str | None = None) -> list[MacroEvent]:
        """Macro events around now. Routed providers return the symbol's own set
        (crypto: macro releases + exchange status; equities: + earnings and halts)."""
        now = self.clock.now()
        start, end = now - timedelta(hours=8), now + timedelta(hours=2)
        try:
            per_symbol = getattr(self.news, "macro_events_for", None)
            if ticker is not None and per_symbol is not None:
                return await per_symbol(ticker, start, end)
            return await self.news.get_macro_events(start, end)
        except ProviderError as exc:
            self.last_error = str(exc)
            log.warning("macro events unavailable: %s", exc)
            return []

    async def assess(self, ticker: str, direction: Direction) -> tuple[NewsAssessment, BlackoutResult]:
        now = self.clock.now()
        try:
            items = await self.news.get_news(ticker, now - timedelta(hours=18))
            self.last_error = None
        except ProviderError as exc:
            self.last_error = str(exc)
            log.warning("news for %s unavailable: %s", ticker, exc)
            items = []
        events = await self.events_window(ticker)
        sess = self.router.schedule_for(ticker).session_for(now)
        bo = blackout_status(now, events, ticker, self.n.blackout_minutes_before, self.n.blackout_minutes_after,
                             sess.close if sess else None)
        recent = [i for i in items if now - i.published_at <= timedelta(hours=4)]
        severity = max((i.severity for i in recent), key=lambda s: _SEV_RANK[s], default=Severity.LOW)
        sentiment = weighted_sentiment(items, now)
        conflict = sentiment * direction.sign <= -self.n.conflict_sentiment_threshold
        return NewsAssessment(
            ticker=ticker, sentiment=sentiment, severity=severity,
            headlines=sorted(items, key=lambda i: i.published_at, reverse=True)[:5],
            in_blackout=bo.active, blackout_reason=bo.reason, blackout_until=bo.until,
            conflicts_with_direction=conflict,
        ), bo

    @staticmethod
    def top_headline(a: NewsAssessment) -> NewsItem | None:
        return max(a.headlines, key=lambda i: (abs(i.sentiment), i.published_at), default=None)
