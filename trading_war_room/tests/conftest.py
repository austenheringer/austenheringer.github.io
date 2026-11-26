"""Shared fixtures. Tests never touch the network or the real data/ directory."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from warroom.config import AppConfig, load_config
from warroom.core.clock import SimulatedClock, et_to_utc
from warroom.core.market_calendar import MarketCalendar

D = date(2025, 1, 15)  # a normal Wednesday session


def et(h: int, m: int = 0, d: date = D) -> datetime:
    """Aware UTC datetime for an ET wall-clock time."""
    return et_to_utc(datetime(d.year, d.month, d.day, h, m))


@pytest.fixture
def cfg() -> AppConfig:
    return load_config("config.yaml")


@pytest.fixture
def cal() -> MarketCalendar:
    return MarketCalendar()


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(et(10, 0))
