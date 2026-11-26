"""Fed / BLS / Nasdaq-halt parsers and the halt-aware news decorator.

Fixtures mirror the real documents' structure (Fed calendar panel markup,
BLS iCalendar with folded lines and TZID, Nasdaq ndaq: namespaced RSS)."""

import json
from datetime import date, timedelta

import httpx

from tests.conftest import et
from warroom.config import HttpSection
from warroom.core.clock import SimulatedClock, to_et
from warroom.models.schemas import MacroEventKind
from warroom.providers.calendar_feeds import (
    HaltAwareNewsProvider,
    MacroCalendarUpdater,
    NasdaqHaltProvider,
    parse_bls_ics,
    parse_fomc_page,
    parse_nasdaq_halts,
    validate_fomc_dates,
)
from warroom.providers.news_data import CalendarOnlyNewsProvider, StaticMacroCalendar


def _meeting(month, days):
    return (f'<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>{month}</strong></div>'
            f'<div class="fomc-meeting__date col-xs-4">{days}</div></div>')


FOMC_HTML = (
    "<html><script>var x='2099 FOMC Meetings';</script><body>"
    '<div class="panel"><h4><a id="1">2027 FOMC Meetings</a></h4>'
    + "".join(_meeting(m, d) for m, d in [("January", "26-27"), ("March", "16-17*"), ("Apr/May", "27-28"),
                                          ("June", "8-9*"), ("July", "27-28"), ("September", "21-22*"),
                                          ("October", "26-27"), ("December", "7-8*")])
    + '</div><div class="panel"><h4>2020 FOMC Meetings</h4>'
    + "".join(_meeting(m, d) for m, d in [("January", "28-29"), ("March", "3 (unscheduled)"), ("March", "15 (unscheduled)"),
                                          ("Apr/May", "28-29"), ("June", "9-10*"), ("July", "28-29"), ("Jul/Aug", "31-1"),
                                          ("September", "15-16*"), ("November", "4-5"), ("December", "15-16*")])
    + "</div></body></html>"
)


def test_parse_fomc_page():
    dates = parse_fomc_page(FOMC_HTML)
    d2027 = [d for d in dates if d.year == 2027]
    assert d2027[0] == date(2027, 1, 27) and date(2027, 4, 28) in d2027 and len(d2027) == 8
    assert date(2020, 8, 1) in dates                       # month rollover "Jul/Aug 31-1"
    assert date(2020, 3, 3) not in dates                   # unscheduled skipped
    assert not any(d.year == 2099 for d in dates)          # script content ignored


def test_validate_fomc_dates_drops_implausible_years():
    good = [date(2027, m, 10) for m in (1, 3, 5, 6, 7, 9, 11, 12)]
    good = [d - timedelta(days=d.weekday() - 2) if d.weekday() > 4 else d for d in good]
    assert validate_fomc_dates(good + [date(2030, 1, 5)]) == sorted(good)  # 2030: 1 date -> rejected


BLS_ICS = (
    "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
    "BEGIN:VEVENT\r\nDTSTART;TZID=US-Eastern:20270108T083000\r\nSUMMARY:Employment Situation for\r\n  December 2026\r\nEND:VEVENT\r\n"
    "BEGIN:VEVENT\r\nDTSTART:20270113T133000Z\r\nSUMMARY:Consumer Price Index for December 2026\r\nEND:VEVENT\r\n"
    "BEGIN:VEVENT\r\nDTSTART;VALUE=DATE:20270114\r\nSUMMARY:Producer Price Index for December 2026\r\nEND:VEVENT\r\n"
    "BEGIN:VEVENT\r\nDTSTART:20270115T150000Z\r\nSUMMARY:Metropolitan Area Employment\r\nEND:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)


def test_parse_bls_ics():
    ev = parse_bls_ics(BLS_ICS)
    assert [e.kind for e in ev] == [MacroEventKind.NFP, MacroEventKind.CPI, MacroEventKind.OTHER]
    assert all(to_et(e.scheduled_at).strftime("%H:%M") == "08:30" for e in ev)
    assert to_et(ev[0].scheduled_at).date() == date(2027, 1, 8)


def test_exact_bls_dates_replace_nfp_rule(cal):
    ev = parse_bls_ics(BLS_ICS)
    smc = StaticMacroCalendar(cal, user_file=None, fetched_events=ev)
    jan = smc.events(et(0, 0, date(2027, 1, 1)), et(23, 0, date(2027, 1, 31)))
    nfps = [e for e in jan if e.kind is MacroEventKind.NFP]
    assert len(nfps) == 1 and nfps[0].source == "bls"


def test_coverage_warnings(cal):
    smc = StaticMacroCalendar(cal, user_file=None)
    assert smc.coverage_warnings(et(10, 0, date(2025, 6, 1)))[0].startswith("No upcoming CPI")
    late = smc.coverage_warnings(et(10, 0, date(2027, 2, 1)))
    assert any("FOMC calendar coverage ends 2026-12-09" in w for w in late)
    fresh = StaticMacroCalendar(cal, user_file=None, fetched_fomc=parse_fomc_page(FOMC_HTML), fetched_events=parse_bls_ics(BLS_ICS))
    assert not any("FOMC" in w for w in fresh.coverage_warnings(et(10, 0, date(2027, 1, 2))))


async def test_updater_fetches_caches_and_falls_back(tmp_path):
    calls = {"n": 0}

    def ok(req):
        calls["n"] += 1
        return httpx.Response(200, text=FOMC_HTML if "federalreserve" in str(req.url) else BLS_ICS)

    path = tmp_path / "macro.json"
    up = MacroCalendarUpdater(HttpSection(max_retries=0), path, transport=httpx.MockTransport(ok))
    fomc, events, notes = await up.refresh(et(10))
    assert date(2027, 1, 27) in fomc and len(events) == 3 and path.exists()
    # within refresh window: served from cache, no network
    fomc2, _, notes2 = await up.refresh(et(11))
    assert calls["n"] == 2 and fomc2 == fomc and "cache" in notes2[0]
    # network down after expiry: cache survives
    down = MacroCalendarUpdater(HttpSection(max_retries=0), path, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    fomc3, events3, notes3 = await down.refresh(et(10) + timedelta(days=30))
    assert fomc3 == fomc and len(events3) == 3 and any("failed" in n for n in notes3)
    # corrupt cache: ignored, not fatal
    path.write_text("{not json")
    assert MacroCalendarUpdater(HttpSection(), path).load_cache() == ([], [], None)


HALTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/"><channel><title>NASDAQTrader.com</title>
<item><title>NVDA</title><ndaq:HaltDate>01/15/2025</ndaq:HaltDate><ndaq:HaltTime>10:31:05</ndaq:HaltTime>
<ndaq:IssueSymbol>NVDA</ndaq:IssueSymbol><ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
<ndaq:ResumptionDate>01/15/2025</ndaq:ResumptionDate><ndaq:ResumptionQuoteTime>10:31:05</ndaq:ResumptionQuoteTime>
<ndaq:ResumptionTradeTime>10:36:05</ndaq:ResumptionTradeTime></item>
<item><title>TSLA</title><ndaq:HaltDate>01/15/2025</ndaq:HaltDate><ndaq:HaltTime>11:00:00</ndaq:HaltTime>
<ndaq:IssueSymbol>TSLA</ndaq:IssueSymbol><ndaq:ReasonCode>T1</ndaq:ReasonCode>
<ndaq:ResumptionDate></ndaq:ResumptionDate><ndaq:ResumptionTradeTime></ndaq:ResumptionTradeTime></item>
<item><title>AMD</title><ndaq:HaltDate>01/14/2025</ndaq:HaltDate><ndaq:HaltTime>15:00:00</ndaq:HaltTime>
<ndaq:IssueSymbol>AMD</ndaq:IssueSymbol><ndaq:ReasonCode>T1</ndaq:ReasonCode></item>
<item><title>ZZZZ</title><ndaq:HaltDate>01/15/2025</ndaq:HaltDate><ndaq:HaltTime>10:00:00</ndaq:HaltTime>
<ndaq:IssueSymbol>ZZZZ</ndaq:IssueSymbol></item>
</channel></rss>"""


def test_parse_nasdaq_halts():
    h = {e.ticker: e for e in parse_nasdaq_halts(HALTS_XML)}
    assert h["NVDA"].ends_at == et(10, 36) + timedelta(seconds=5) and not h["NVDA"].ongoing
    assert h["TSLA"].ongoing and h["TSLA"].ends_at is None and "T1" in h["TSLA"].name


async def test_halt_aware_news(cal):
    clk = SimulatedClock(et(11, 30))
    halts = NasdaqHaltProvider(clk, HttpSection(max_retries=0), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=HALTS_XML)))
    news = HaltAwareNewsProvider(CalendarOnlyNewsProvider(clk, StaticMacroCalendar(cal, None)), halts, ["NVDA", "TSLA", "AMD"])
    ev = await news.get_macro_events(et(11, 15), et(11, 45))
    tickers = {e.ticker for e in ev if e.kind is MacroEventKind.HALT}
    # TSLA ongoing -> included; NVDA resumed 10:36 (before window) -> excluded;
    # AMD ongoing but from yesterday -> stale, excluded; ZZZZ not watched
    assert tickers == {"TSLA"}
    assert news.name == "calendar_only+halts"
    await news.aclose()


async def test_halt_feed_outage_keeps_last_known(cal):
    clk = SimulatedClock(et(11, 0))
    state = {"up": True}
    hp = NasdaqHaltProvider(clk, HttpSection(max_retries=0), min_interval_seconds=30, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=HALTS_XML) if state["up"] else httpx.Response(503)))
    assert len(await hp.halts()) == 4
    state["up"] = False
    clk.advance(10)
    assert len(await hp.halts()) == 4 and hp.last_error is None   # throttled: no refetch
    clk.advance(60)
    assert len(await hp.halts()) == 4 and hp.last_error            # outage: last known kept
    await hp.aclose()
