"""Clock abstraction.

Trading logic must never call ``datetime.now()`` directly. Everything that
needs "now" receives a :class:`Clock`, so ORB windows, cooldowns, blackout
windows and EOD flat closes behave identically in live and replay modes.

All clocks return timezone-aware UTC datetimes. Convert with :func:`to_et`
for market-hours logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime. Naive datetimes are rejected:
    guessing their zone is exactly the bug class this module exists to stop."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"naive datetime not allowed: {dt!r}")
    return dt.astimezone(UTC)


def to_et(dt: datetime) -> datetime:
    """Convert an aware datetime to America/New_York."""
    return ensure_utc(dt).astimezone(ET)


def et_to_utc(dt_et: datetime) -> datetime:
    """Interpret a naive wall-clock datetime as ET (or convert an aware one) and return UTC."""
    if dt_et.tzinfo is None:
        dt_et = dt_et.replace(tzinfo=ET)
    return dt_et.astimezone(UTC)


class Clock(ABC):
    """Source of truth for 'now'."""

    @abstractmethod
    def now(self) -> datetime:
        """Current time, aware UTC."""

    def now_et(self) -> datetime:
        return to_et(self.now())

    @abstractmethod
    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` of *clock* time."""


class SystemClock(Clock):
    """Wall-clock time. Used by --demo and --paper."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class SimulatedClock(Clock):
    """Manually driven clock for --replay and tests.

    ``sleep`` advances simulated time instantly (yielding to the loop once)
    so replay runs as fast as the CPU allows. Time never moves backwards.
    """

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def set(self, dt: datetime) -> None:
        dt = ensure_utc(dt)
        if dt < self._now:
            raise ValueError(f"SimulatedClock cannot move backwards: {dt} < {self._now}")
        self._now = dt

    def advance(self, delta: timedelta | float) -> datetime:
        if isinstance(delta, (int, float)):
            delta = timedelta(seconds=delta)
        if delta < timedelta(0):
            raise ValueError("cannot advance by a negative delta")
        self._now = self._now + delta
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.advance(max(0.0, seconds))
        await asyncio.sleep(0)


class DemoClock(Clock):
    """Wall-clock driven, but mapped onto a real trading session and optionally sped up.

    ``--demo`` must be watchable at any hour: with a plain SystemClock a 3 a.m.
    or holiday demo would sit outside market hours, so ORB windows, blackout
    windows and EOD flat logic would never fire. DemoClock starts at
    ``start`` (e.g. 09:50 ET on the latest trading day) and advances
    ``speed`` clock-seconds per real second. It is still monotonic and
    wall-driven (unlike SimulatedClock, it does not jump on ``sleep``).
    """

    def __init__(self, start: datetime, speed: float = 1.0) -> None:
        import time as _time

        if speed <= 0:
            raise ValueError("speed must be positive")
        self._start = ensure_utc(start)
        self.speed = speed
        self._mono = _time.monotonic
        self._t0 = self._mono()

    def now(self) -> datetime:
        return self._start + timedelta(seconds=(self._mono() - self._t0) * self.speed)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds) / self.speed)
