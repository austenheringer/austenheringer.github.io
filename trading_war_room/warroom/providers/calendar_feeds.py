"""Live schedule feeds: FOMC calendar, BLS release schedule, Nasdaq trading halts.

* :func:`parse_fomc_page` - federalreserve.gov FOMC calendar HTML -> decision dates.
* :func:`parse_bls_ics`   - bls.gov release-schedule iCalendar -> CPI / NFP / PPI / JOLTS events.
* :class:`MacroCalendarUpdater` - fetches both, validates them, caches to JSON
  (refresh weekly) and falls back to the cache / built-ins when offline.
* :class:`NasdaqHaltProvider` - Nasdaq Trader trade-halt RSS (covers halts on
  all US listing venues), parsed with the stdlib XML parser.
* :class:`HaltAwareNewsProvider` - wraps any NewsProvider so halts on watched
  tickers appear as HALT macro events and trigger blackouts.

All parsing is defensive: a page-layout change yields "no data" plus a
logged warning, never a crash and never a fabricated date.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET_XML
from datetime import date, datetime, time, timedelta
from pathlib import Path

from warroom.config import HttpSection
from warroom.core.clock import UTC, Clock, ensure_utc, et_to_utc, to_et
from warroom.models.schemas import MacroEvent, MacroEventKind, NewsItem, Severity
from warroom.providers.base import NewsProvider, ProviderError
from warroom.providers.http import RetryingHttpClient

log = logging.getLogger(__name__)

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
NASDAQ_HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}


# --------------------------------------------------------------------------- FOMC


def _month_num(token: str) -> int | None:
    t = token.strip().lower().rstrip(".")
    return _MONTHS.get(t) or _MONTH_ABBR.get(t[:3])


def parse_fomc_page(html: str) -> list[date]:
    """Extract FOMC *decision* dates (last day of each scheduled meeting).

    The Fed page groups meetings under "<YEAR> FOMC Meetings" headings and
    renders each meeting as a month cell ("January", "Apr/May") followed by a
    date cell ("27-28", "30-1*"). Notation votes and unscheduled meetings are
    skipped (no 14:00 statement on a known date).
    """
    out: set[date] = set()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    sections = re.split(r"(\d{4})\s+FOMC\s+Meetings", text)
    # sections = [pre, year1, body1, year2, body2, ...]
    for i in range(1, len(sections) - 1, 2):
        year = int(sections[i])
        body = sections[i + 1]
        cells = [re.sub(r"<[^>]+>|&nbsp;", " ", c) for c in re.findall(
            r'class="[^"]*fomc-meeting__(?:month|date)[^"]*"[^>]*>(.*?)</div>', body, flags=re.S | re.I)]
        cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
        for month_cell, date_cell in zip(cells[0::2], cells[1::2]):
            if re.search(r"notation|unscheduled|cancel", date_cell, re.I):
                continue
            months = [m for m in (_month_num(t) for t in month_cell.split("/")) if m]
            days = [int(d) for d in re.findall(r"\d{1,2}", date_cell)]
            if not months or not days:
                continue
            last_day = days[-1]
            # "Apr/May" + "30-1": the meeting ends in the second month
            month = months[-1] if len(months) > 1 and len(days) > 1 and last_day < days[0] else months[0]
            try:
                out.add(date(year, month, last_day))
            except ValueError:
                log.warning("FOMC parser: skipping impossible date %s %s %s", year, month_cell, date_cell)
    return sorted(out)


def validate_fomc_dates(dates: list[date]) -> list[date]:
    """Sanity check: 6-10 scheduled meetings per year, all weekdays (Tue/Wed in practice)."""
    by_year: dict[int, list[date]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    good: list[date] = []
    for year, ds in by_year.items():
        if not (6 <= len(ds) <= 10) or any(d.weekday() >= 5 for d in ds):
            log.warning("FOMC dates for %s failed validation (%d dates); ignoring that year", year, len(ds))
            continue
        good.extend(ds)
    return sorted(good)


# --------------------------------------------------------------------------- BLS


_BLS_MAP: list[tuple[str, MacroEventKind, Severity, str]] = [
    ("employment situation", MacroEventKind.NFP, Severity.HIGH, "Nonfarm Payrolls (Employment Situation)"),
    ("consumer price index", MacroEventKind.CPI, Severity.HIGH, "CPI release"),
    ("producer price index", MacroEventKind.OTHER, Severity.MEDIUM, "PPI release"),
    ("job openings and labor turnover", MacroEventKind.OTHER, Severity.MEDIUM, "JOLTS release"),
    ("employment cost index", MacroEventKind.OTHER, Severity.MEDIUM, "Employment Cost Index"),
]


def _unfold_ics(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_ics_dt(key: str, value: str) -> datetime | None:
    value = value.strip()
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if "T" in value:
            naive = datetime.strptime(value, "%Y%m%dT%H%M%S")
            # TZID=US-Eastern / America/New_York, or floating time: BLS publishes in ET
            return et_to_utc(naive)
        d = datetime.strptime(value, "%Y%m%d").date()
        return et_to_utc(datetime.combine(d, time(8, 30)))  # all-day entry: BLS releases at 08:30 ET
    except ValueError:
        return None


def parse_bls_ics(text: str) -> list[MacroEvent]:
    events: list[MacroEvent] = []
    current: dict[str, tuple[str, str]] = {}
    for line in _unfold_ics(text):
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT":
            summary = current.get("SUMMARY", ("", ""))[1].lower()
            dt_key, dt_val = current.get("DTSTART", ("", ""))
            when = _parse_ics_dt(dt_key, dt_val) if dt_val else None
            if when is None:
                continue
            for needle, kind, sev, name in _BLS_MAP:
                if needle in summary:
                    events.append(MacroEvent(name=name, kind=kind, scheduled_at=when, severity=sev, source="bls"))
                    break
        elif ":" in line:
            head, val = line.split(":", 1)
            key = head.split(";", 1)[0].upper()
            current[key] = (head, val)
    return sorted(events, key=lambda e: e.scheduled_at)


# --------------------------------------------------------------------------- updater


class MacroCalendarUpdater:
    """Fetch FOMC + BLS schedules, cache them, and serve the best available copy."""

    def __init__(self, http_cfg: HttpSection, cache_path: str | Path = "data/macro_cache.json",
                 refresh_days: int = 7, transport=None) -> None:
        self.cache_path = Path(cache_path)
        self.refresh_days = refresh_days
        self._http_cfg = http_cfg
        self._transport = transport

    def load_cache(self) -> tuple[list[date], list[MacroEvent], datetime | None]:
        if not self.cache_path.exists():
            return [], [], None
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            fomc = [date.fromisoformat(d) for d in raw.get("fomc", [])]
            events = [MacroEvent.model_validate(e) for e in raw.get("events", [])]
            fetched = datetime.fromisoformat(raw["fetched_at"]) if raw.get("fetched_at") else None
            return fomc, events, fetched
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not stop startup
            log.warning("macro cache %s unreadable (%s); ignoring", self.cache_path, exc)
            return [], [], None

    def _save(self, fomc: list[date], events: list[MacroEvent], now: datetime) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "fetched_at": ensure_utc(now).isoformat(),
            "fomc": [d.isoformat() for d in fomc],
            "events": [json.loads(e.model_dump_json()) for e in events],
        }, indent=1), encoding="utf-8")
        tmp.replace(self.cache_path)  # atomic

    async def refresh(self, now: datetime, force: bool = False) -> tuple[list[date], list[MacroEvent], list[str]]:
        """Return (fomc_dates, events, notes). Network failures fall back to cache."""
        fomc, events, fetched_at = self.load_cache()
        notes: list[str] = []
        if not force and fetched_at and (ensure_utc(now) - fetched_at) < timedelta(days=self.refresh_days):
            return fomc, events, [f"macro calendar cache from {fetched_at.date()}"]
        http = RetryingHttpClient(self._http_cfg, headers={"User-Agent": _BROWSER_UA}, transport=self._transport)
        try:
            new_fomc: list[date] = []
            new_events: list[MacroEvent] = []
            try:
                new_fomc = validate_fomc_dates(parse_fomc_page((await http.get(FOMC_URL)).text))
                notes.append(f"FOMC calendar: {len(new_fomc)} dates fetched")
            except ProviderError as exc:
                notes.append(f"FOMC calendar fetch failed ({exc}); using cache/built-ins")
            try:
                new_events = parse_bls_ics((await http.get(BLS_ICS_URL)).text)
                notes.append(f"BLS schedule: {len(new_events)} releases fetched")
            except ProviderError as exc:
                notes.append(f"BLS schedule fetch failed ({exc}); using cache/rules")
        finally:
            await http.aclose()
        fomc = new_fomc or fomc
        events = new_events or events
        if new_fomc or new_events:
            self._save(fomc, events, now)
        return fomc, events, notes


# --------------------------------------------------------------------------- halts


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_nasdaq_halts(xml_text: str) -> list[MacroEvent]:
    """Parse the Nasdaq Trader trade-halt RSS. Times in the feed are US Eastern."""
    try:
        root = ET_XML.fromstring(xml_text)
    except ET_XML.ParseError as exc:
        raise ProviderError(f"halt feed is not valid XML: {exc}") from exc
    out: list[MacroEvent] = []
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        f = {_local(c.tag): (c.text or "").strip() for c in item}
        sym = f.get("issuesymbol") or f.get("title", "")
        if not sym or not f.get("haltdate") or not f.get("halttime"):
            continue
        try:
            halted = et_to_utc(datetime.strptime(f"{f['haltdate']} {f['halttime']}", "%m/%d/%Y %H:%M:%S"))
        except ValueError:
            continue
        resume = None
        if f.get("resumptiondate") and f.get("resumptiontradetime"):
            try:
                resume = et_to_utc(datetime.strptime(f"{f['resumptiondate']} {f['resumptiontradetime']}",
                                                     "%m/%d/%Y %H:%M:%S"))
            except ValueError:
                resume = None
        code = f.get("reasoncode", "")
        out.append(MacroEvent(
            name=f"{sym.upper()} trading halt ({code or 'unknown reason'})", kind=MacroEventKind.HALT,
            scheduled_at=halted, severity=Severity.HIGH, ticker=sym.upper(), ends_at=resume,
            ongoing=resume is None, source="nasdaq",
        ))
    return out


class NasdaqHaltProvider:
    """Polls the halt feed at most every ``min_interval_seconds`` of clock time."""

    def __init__(self, clock: Clock, http_cfg: HttpSection, min_interval_seconds: float = 30,
                 transport=None) -> None:
        self.clock = clock
        self.min_interval = timedelta(seconds=min_interval_seconds)
        self.http = RetryingHttpClient(http_cfg, headers={"User-Agent": _BROWSER_UA}, transport=transport)
        self._cache: list[MacroEvent] = []
        self._fetched_at: datetime | None = None
        self.last_error: str | None = None

    async def halts(self) -> list[MacroEvent]:
        now = self.clock.now()
        if self._fetched_at and now - self._fetched_at < self.min_interval:
            return self._cache
        try:
            self._cache = parse_nasdaq_halts((await self.http.get(NASDAQ_HALTS_URL)).text)
            self.last_error = None
        except ProviderError as exc:
            self.last_error = str(exc)
            log.warning("halt feed unavailable: %s (keeping last %d halts)", exc, len(self._cache))
        self._fetched_at = now
        return self._cache

    async def aclose(self) -> None:
        await self.http.aclose()


class HaltAwareNewsProvider(NewsProvider):
    """Decorates a NewsProvider: halts on watched tickers become HALT macro events."""

    def __init__(self, inner: NewsProvider, halts: NasdaqHaltProvider, tickers: list[str]) -> None:
        super().__init__(inner.clock)
        self.inner = inner
        self.halt_feed = halts
        self.tickers = set(tickers)
        self.name = f"{inner.name}+halts"

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        return await self.inner.get_news(ticker, since)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        events = await self.inner.get_macro_events(start, end)
        start, end = ensure_utc(start), ensure_utc(end)
        today = to_et(self.clock.now()).date()
        for h in await self.halt_feed.halts():
            if h.ticker not in self.tickers:
                continue
            if h.ongoing and to_et(h.scheduled_at).date() != today:
                continue  # stale feed entry from a previous session; never block forever
            # a halt matters if it overlaps the window: started before end and not resumed before start
            if h.scheduled_at <= end and (h.ongoing or (h.ends_at is not None and h.ends_at >= start)):
                events.append(h)
        return sorted(events, key=lambda e: e.scheduled_at)

    async def aclose(self) -> None:
        await self.inner.aclose()
        await self.halt_feed.aclose()
