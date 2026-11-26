"""US equity market calendar (NYSE/Nasdaq regular session).

Holidays are computed by rule (not a hard-coded list that silently goes stale)
plus a table of known one-off closures. All boundaries are produced as aware
UTC datetimes derived from America/New_York wall-clock times, so DST is
handled by zoneinfo, not by arithmetic.

Rules implemented (NYSE):
* New Year's Day - Sunday -> Monday; Saturday -> NOT observed (NYSE rule 7.2).
* MLK Day (3rd Mon Jan, since 1998), Presidents' Day (3rd Mon Feb),
  Good Friday, Memorial Day (last Mon May), Juneteenth (since 2022),
  Independence Day, Labor Day (1st Mon Sep), Thanksgiving (4th Thu Nov),
  Christmas. Fixed-date holidays: Saturday -> Friday, Sunday -> Monday.
* Early close (13:00 ET): July 3 when it is a trading day Mon-Thu, the day
  after Thanksgiving, and Dec 24 when it is a trading day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from functools import lru_cache
from typing import Iterable

from warroom.core.clock import ET, ensure_utc, to_et

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
PREMARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

# One-off closures that no rule can predict.
KNOWN_SPECIAL_CLOSURES: dict[date, str] = {
    date(2012, 10, 29): "Hurricane Sandy",
    date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "National Day of Mourning (George H.W. Bush)",
    date(2025, 1, 9): "National Day of Mourning (Jimmy Carter)",
}


class MarketStatus(StrEnum):
    CLOSED = "CLOSED"          # holiday / weekend / overnight
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    AFTER_HOURS = "AFTER_HOURS"


@dataclass(frozen=True)
class TradingSession:
    day: date
    open: datetime       # UTC
    orb_end: datetime    # UTC
    eod_flat: datetime   # UTC
    close: datetime      # UTC
    early_close: bool

    def contains(self, dt: datetime) -> bool:
        dt = ensure_utc(dt)
        return self.open <= dt < self.close

    def in_orb_window(self, dt: datetime) -> bool:
        dt = ensure_utc(dt)
        return self.open <= dt < self.orb_end

    def past_eod_flat(self, dt: datetime) -> bool:
        return ensure_utc(dt) >= self.eod_flat


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def rule_holidays(year: int) -> dict[date, str]:
    h: dict[date, str] = {}
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        h[ny + timedelta(days=1)] = "New Year's Day (observed)"
    elif ny.weekday() != 5:
        h[ny] = "New Year's Day"
    if year >= 1998:
        h[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    h[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    h[_easter(year) - timedelta(days=2)] = "Good Friday"
    h[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        h[_observed(date(year, 6, 19))] = "Juneteenth"
    h[_observed(date(year, 7, 4))] = "Independence Day"
    h[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    h[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    h[_observed(date(year, 12, 25))] = "Christmas Day"
    # Next year's New Year's never observes back into Dec 31 on NYSE.
    return h


class MarketCalendar:
    def __init__(
        self,
        extra_holidays: Iterable[date] = (),
        extra_early_closes: Iterable[date] = (),
        orb_minutes: int = 15,
        eod_flat_minutes_before_close: int = 5,
    ) -> None:
        self._extra_holidays = set(extra_holidays)
        self._extra_early = set(extra_early_closes)
        self.orb_minutes = orb_minutes
        self.eod_flat_minutes = eod_flat_minutes_before_close

    # ---- day classification -------------------------------------------------

    def holiday_name(self, d: date) -> str | None:
        if d in self._extra_holidays:
            return "Configured closure"
        if d in KNOWN_SPECIAL_CLOSURES:
            return KNOWN_SPECIAL_CLOSURES[d]
        return rule_holidays(d.year).get(d)

    def is_holiday(self, d: date) -> bool:
        return self.holiday_name(d) is not None

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self.is_holiday(d)

    def is_early_close(self, d: date) -> bool:
        if not self.is_trading_day(d):
            return False
        if d in self._extra_early:
            return True
        if d.month == 7 and d.day == 3 and d.weekday() <= 3:
            return True
        if d.month == 11 and d == _nth_weekday(d.year, 11, 3, 4) + timedelta(days=1):
            return True
        if d.month == 12 and d.day == 24:
            return True
        return False

    # ---- sessions -------------------------------------------------------------

    def session(self, d: date) -> TradingSession | None:
        if not self.is_trading_day(d):
            return None
        early = self.is_early_close(d)
        open_et = datetime.combine(d, REGULAR_OPEN, tzinfo=ET)
        close_et = datetime.combine(d, EARLY_CLOSE if early else REGULAR_CLOSE, tzinfo=ET)
        return TradingSession(
            day=d,
            open=ensure_utc(open_et),
            orb_end=ensure_utc(open_et + timedelta(minutes=self.orb_minutes)),
            eod_flat=ensure_utc(close_et - timedelta(minutes=self.eod_flat_minutes)),
            close=ensure_utc(close_et),
            early_close=early,
        )

    def session_for(self, dt: datetime) -> TradingSession | None:
        """Session for the ET calendar date of ``dt`` (None if not a trading day)."""
        return self.session(to_et(dt).date())

    def status(self, dt: datetime) -> MarketStatus:
        et = to_et(dt)
        sess = self.session(et.date())
        if sess is None:
            return MarketStatus.CLOSED
        utc = ensure_utc(dt)
        if sess.open <= utc < sess.close:
            return MarketStatus.OPEN
        if PREMARKET_OPEN <= et.time() and utc < sess.open:
            return MarketStatus.PRE_MARKET
        if utc >= sess.close and et.time() < AFTER_HOURS_CLOSE:
            return MarketStatus.AFTER_HOURS
        return MarketStatus.CLOSED

    def is_open(self, dt: datetime) -> bool:
        return self.status(dt) is MarketStatus.OPEN

    def next_session(self, dt: datetime) -> TradingSession:
        """The current session if ``dt`` is before its close, else the next one."""
        utc = ensure_utc(dt)
        d = to_et(dt).date()
        for _ in range(15):  # longest realistic gap is ~4 days
            sess = self.session(d)
            if sess is not None and utc < sess.close:
                return sess
            d += timedelta(days=1)
        raise RuntimeError(f"no trading session found within 15 days of {dt}")  # pragma: no cover

    def previous_trading_days(self, d: date, n: int, include_today: bool = True) -> list[date]:
        """The last ``n`` trading days ending at ``d`` (most recent first).
        Used for the PDT rolling 5-business-day window."""
        out: list[date] = []
        cur = d if include_today else d - timedelta(days=1)
        guard = 0
        while len(out) < n:
            if self.is_trading_day(cur):
                out.append(cur)
            cur -= timedelta(days=1)
            guard += 1
            if guard > n * 4 + 15:  # pragma: no cover
                break
        return out
