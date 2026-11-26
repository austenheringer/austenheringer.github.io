from datetime import date, datetime, timedelta, timezone

import pytest

from tests.conftest import et
from warroom.core.clock import DemoClock, SimulatedClock, ensure_utc, to_et
from warroom.core.market_calendar import MarketStatus
from warroom.providers import factory

HOLIDAYS = {
    2025: ["2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
           "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"],
    2026: ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
           "2026-09-07", "2026-11-26", "2026-12-25"],
    2027: ["2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
           "2027-09-06", "2027-11-25", "2027-12-24"],
}


def _days(y):
    d = date(y, 1, 1)
    while d.year == y:
        yield d
        d += timedelta(days=1)


@pytest.mark.parametrize("year", sorted(HOLIDAYS))
def test_nyse_holidays(cal, year):
    got = [str(d) for d in _days(year) if d.weekday() < 5 and cal.is_holiday(d)]
    assert got == HOLIDAYS[year]


def test_new_year_on_saturday_not_observed(cal):
    assert cal.is_trading_day(date(2022, 12, 30))


def test_early_closes(cal):
    assert [str(d) for d in _days(2025) if cal.is_early_close(d)] == ["2025-07-03", "2025-11-28", "2025-12-24"]
    assert [str(d) for d in _days(2026) if cal.is_early_close(d)] == ["2026-11-27", "2026-12-24"]


def test_session_times_and_dst(cal):
    s = cal.session(date(2025, 1, 15))
    assert (s.open.hour, s.open.minute) == (14, 30)  # EST
    assert s.orb_end == s.open + timedelta(minutes=15)
    assert s.eod_flat == datetime(2025, 1, 15, 20, 55, tzinfo=timezone.utc)
    assert cal.session(date(2025, 7, 15)).open.hour == 13  # EDT
    early = cal.session(date(2025, 11, 28))
    assert early.early_close and to_et(early.eod_flat).strftime("%H:%M") == "12:55"


def test_status_and_next_session(cal):
    assert cal.status(et(8, 0)) is MarketStatus.PRE_MARKET
    assert cal.status(et(10, 0)) is MarketStatus.OPEN
    assert cal.status(et(17, 0)) is MarketStatus.AFTER_HOURS
    assert cal.status(et(12, 0, date(2025, 1, 18))) is MarketStatus.CLOSED
    assert cal.next_session(et(17, 0, date(2025, 1, 17))).day == date(2025, 1, 21)  # MLK skipped
    assert cal.previous_trading_days(date(2025, 1, 21), 5)[1] == date(2025, 1, 17)


def test_naive_datetimes_rejected():
    with pytest.raises(ValueError):
        ensure_utc(datetime(2025, 1, 1))


def test_simulated_clock_monotonic():
    c = SimulatedClock(et(10, 0))
    c.advance(60)
    assert c.now() == et(10, 1)
    with pytest.raises(ValueError):
        c.set(et(9, 0))


async def test_simulated_sleep_advances_instantly():
    c = SimulatedClock(et(10, 0))
    await c.sleep(300)
    assert c.now() == et(10, 5)


def test_demo_clock_maps_holiday_to_last_session(cfg, cal):
    dc = factory.build_demo_clock(cfg, cal, datetime(2026, 11, 26, 8, tzinfo=timezone.utc))  # Thanksgiving
    assert to_et(dc.now()).date() == date(2026, 11, 25)
    assert to_et(dc.now()).strftime("%H:%M") == "09:50"
    assert isinstance(dc, DemoClock) and dc.speed == cfg.demo.speed_multiplier
