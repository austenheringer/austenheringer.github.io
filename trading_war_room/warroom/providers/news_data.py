"""News & macro providers: Mock, RSS, Alpaca News, Finnhub.

Macro calendar
--------------
Free news APIs do not provide a reliable economic calendar. Every real
provider therefore combines its headlines with :class:`StaticMacroCalendar`:
built-in FOMC decision dates (14:00 ET - the only routinely scheduled
high-impact release that lands *inside* regular hours) plus rule-based
CPI/NFP placeholders (08:30 ET, pre-open), plus any events you add in
``data/macro_events.yaml``. Keep that file current; the blackout gate is only
as good as the calendar behind it.

Trading halts are not covered by any of these feeds. The stale-data guard
(no fresh candles) is the backstop for a halted symbol.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from warroom.config import HttpSection
from warroom.core.clock import UTC, Clock, ensure_utc, et_to_utc, to_et
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import MacroEvent, MacroEventKind, NewsItem, Severity
from warroom.providers.base import NewsProvider, ProviderError, ProviderUnavailable
from warroom.providers.http import RetryingHttpClient
from warroom.providers.sentiment import score_headline

log = logging.getLogger(__name__)

# FOMC statement days (second day of each meeting), statement at 14:00 ET.
FOMC_DECISION_DATES: tuple[date, ...] = (
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
)


class StaticMacroCalendar:
    def __init__(self, calendar: MarketCalendar, user_file: str | Path | None = "data/macro_events.yaml") -> None:
        self.calendar = calendar
        self._user: list[MacroEvent] = []
        if user_file and Path(user_file).exists():
            self._user = self._load(Path(user_file))

    @staticmethod
    def _load(path: Path) -> list[MacroEvent]:
        """YAML list of {name, kind, date: YYYY-MM-DD, time_et: "HH:MM", severity, ticker?}."""
        out: list[MacroEvent] = []
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        for i, row in enumerate(raw):
            try:
                hh, mm = map(int, str(row["time_et"]).split(":"))
                d = date.fromisoformat(str(row["date"]))
                out.append(MacroEvent(
                    name=row["name"], kind=MacroEventKind(row.get("kind", "OTHER")),
                    scheduled_at=et_to_utc(datetime.combine(d, time(hh, mm))),
                    severity=Severity(row.get("severity", "HIGH")), ticker=row.get("ticker"),
                ))
            except Exception as exc:  # noqa: BLE001 - surface which row is wrong
                raise ProviderError(f"{path}: invalid macro event #{i}: {exc}") from exc
        return out

    def events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        start, end = ensure_utc(start), ensure_utc(end)
        out: list[MacroEvent] = []
        d = to_et(start).date()
        while d <= to_et(end).date():
            if d in FOMC_DECISION_DATES:
                out.append(MacroEvent(name="FOMC rate decision", kind=MacroEventKind.FOMC,
                                      scheduled_at=et_to_utc(datetime.combine(d, time(14, 0))), severity=Severity.HIGH))
            # NFP: first Friday of the month, 08:30 ET (approximate; BLS occasionally shifts it)
            if d.weekday() == 4 and d.day <= 7:
                out.append(MacroEvent(name="Nonfarm Payrolls (approx. date)", kind=MacroEventKind.NFP,
                                      scheduled_at=et_to_utc(datetime.combine(d, time(8, 30))), severity=Severity.HIGH))
            d += timedelta(days=1)
        out.extend(self._user)
        return sorted((e for e in out if start <= e.scheduled_at <= end), key=lambda e: e.scheduled_at)


def _item(ticker: str | None, headline: str, source: str, published: datetime, url: str | None = None,
          item_id: str | None = None) -> NewsItem:
    sentiment, severity, _ = score_headline(headline)
    kwargs = {"id": item_id} if item_id else {}
    return NewsItem(ticker=ticker, headline=headline, source=source, published_at=ensure_utc(published),
                    url=url, sentiment=sentiment, severity=severity, **kwargs)


# --------------------------------------------------------------------------- mock


_MOCK_TEMPLATES = {
    1: ["{t} surges after analyst upgrade to buy", "{t} beats estimates, raises guidance",
        "{t} wins contract with major cloud provider", "{t} record revenue on strong demand"],
    -1: ["{t} tumbles after downgrade to sell", "{t} misses estimates as margins slump",
         "{t} faces SEC investigation over accounting", "{t} cuts guidance citing weak demand"],
    0: ["{t} to present at investor conference next week", "{t} shares active in morning trade",
        "{t} options volume elevated ahead of product event"],
}


class MockNewsProvider(NewsProvider):
    """Deterministic simulated news and macro calendar for --demo / tests.

    Headline tone is correlated with the ticker's mock price scenario (so News
    sometimes agrees with Technical and sometimes contradicts it) and every day
    includes an intraday high-impact macro event so blackout windows are
    exercised. Items are only returned once ``published_at <= now``.
    """

    name = "mock"
    _SCENARIO_TONE = {"orb_breakout": 1, "trend_up_pullbacks": 1, "vwap_reclaim": 1,
                      "orb_breakdown": -1, "trend_down_rallies": -1, "spike_fade": -1, "chop": 0}

    def __init__(self, clock: Clock, calendar: MarketCalendar, tickers: list[str], seed: int = 42,
                 scenario_fn=None) -> None:
        super().__init__(clock)
        self.calendar = calendar
        self.tickers = tickers
        self.seed = seed
        self._scenario_fn = scenario_fn  # (ticker, date) -> scenario name, from MockDataProvider

    def _rng(self, *parts: object) -> np.random.Generator:
        h = hashlib.sha256("|".join(map(str, (self.seed, *parts))).encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(h, "little"))

    def _day_news(self, d: date) -> list[NewsItem]:
        sess = self.calendar.session(d)
        if sess is None:
            return []
        items: list[NewsItem] = []
        for t in self.tickers:
            rng = self._rng("news", t, d)
            tone = self._SCENARIO_TONE.get(self._scenario_fn(t, d), 0) if self._scenario_fn else 0
            # 25% of the time the news tone contradicts the tape - fuel for cross-examination
            if tone and rng.random() < 0.25:
                tone = -tone
            for _ in range(int(rng.integers(1, 4))):
                minutes = int(rng.integers(-300, int((sess.close - sess.open).total_seconds() // 60) - 30))
                pub = sess.open + timedelta(minutes=minutes)
                pick = tone if rng.random() < 0.7 else 0
                tmpl = _MOCK_TEMPLATES[pick][int(rng.integers(0, len(_MOCK_TEMPLATES[pick])))]
                items.append(_item(t, tmpl.format(t=t), "MockWire", pub,
                                   item_id=f"mock-{d}-{t}-{minutes}"))
        return items

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        since = ensure_utc(since)
        out: list[NewsItem] = []
        d = to_et(since).date()
        while d <= to_et(now).date():
            out.extend(self._day_news(d))
            d += timedelta(days=1)
        return sorted(
            (n for n in out if since <= n.published_at <= now and (ticker is None or n.ticker == ticker)),
            key=lambda n: n.published_at,
        )

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        start, end = ensure_utc(start), ensure_utc(end)
        out: list[MacroEvent] = []
        d = to_et(start).date()
        while d <= to_et(end).date():
            sess = self.calendar.session(d)
            if sess is not None:
                rng = self._rng("macro", d)
                out.append(MacroEvent(name="CPI release", kind=MacroEventKind.CPI,
                                      scheduled_at=et_to_utc(datetime.combine(d, time(8, 30))), severity=Severity.HIGH))
                # one high-impact intraday event, somewhere between 10:30 and 14:30 ET
                k = int(rng.integers(60, 300))
                out.append(MacroEvent(name=str(rng.choice(["FOMC rate decision", "Fed Chair testimony"])),
                                      kind=MacroEventKind.FOMC, scheduled_at=sess.open + timedelta(minutes=k),
                                      severity=Severity.HIGH))
                if self.tickers:
                    t = self.tickers[int(rng.integers(0, len(self.tickers)))]
                    out.append(MacroEvent(name=f"{t} earnings (after close)", kind=MacroEventKind.EARNINGS,
                                          scheduled_at=sess.close + timedelta(minutes=5), severity=Severity.HIGH,
                                          ticker=t))
                    if rng.random() < 0.35:
                        h = self.tickers[int(rng.integers(0, len(self.tickers)))]
                        out.append(MacroEvent(name=f"{h} LULD volatility halt", kind=MacroEventKind.HALT,
                                              scheduled_at=sess.open + timedelta(minutes=int(rng.integers(30, 330))),
                                              severity=Severity.HIGH, ticker=h))
            d += timedelta(days=1)
        return sorted((e for e in out if start <= e.scheduled_at <= end), key=lambda e: e.scheduled_at)


# --------------------------------------------------------------------------- real providers


def _mentions(headline: str, ticker: str) -> bool:
    return re.search(rf"(?<![A-Za-z])\$?{re.escape(ticker)}(?![A-Za-z])", headline) is not None


class RSSNewsProvider(NewsProvider):
    name = "rss"

    def __init__(self, clock: Clock, feeds: list[str], http_cfg: HttpSection, macro: StaticMacroCalendar,
                 transport=None) -> None:
        super().__init__(clock)
        try:
            import feedparser  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailable("feedparser is not installed: pip install feedparser") from exc
        if not feeds:
            raise ProviderUnavailable("no RSS feeds configured (providers.rss_feeds)")
        self.feeds = feeds
        self.macro = macro
        self.http = RetryingHttpClient(http_cfg, transport=transport)

    async def _fetch(self, url: str) -> list[NewsItem]:
        import feedparser

        try:
            body = (await self.http.get(url)).content
        except ProviderError as exc:
            log.warning("RSS feed %s failed: %s", url.split("?")[0], exc)
            return []
        parsed = await asyncio.to_thread(feedparser.parse, body)
        items = []
        for e in parsed.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            if not ts:
                continue
            pub = datetime(*ts[:6], tzinfo=UTC)  # feedparser normalises *_parsed to UTC
            items.append(_item(None, e.get("title", "").strip(), parsed.feed.get("title", "RSS"), pub,
                               url=e.get("link"), item_id=hashlib.sha1((e.get("link") or e.get("title", "")).encode()).hexdigest()[:12]))
        return items

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        batches = await asyncio.gather(*(self._fetch(u) for u in self.feeds))
        out = []
        for n in (i for b in batches for i in b):
            if not (ensure_utc(since) <= n.published_at <= now) or not n.headline:
                continue
            if ticker is None:
                out.append(n)
            elif _mentions(n.headline, ticker):
                out.append(n.model_copy(update={"ticker": ticker}))
        return sorted(out, key=lambda n: n.published_at)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        return self.macro.events(start, end)

    async def aclose(self) -> None:
        await self.http.aclose()


class AlpacaNewsProvider(NewsProvider):
    name = "alpaca"
    URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, clock: Clock, api_key: str, secret_key: str, http_cfg: HttpSection,
                 macro: StaticMacroCalendar, transport=None) -> None:
        super().__init__(clock)
        if not api_key or not secret_key:
            raise ProviderUnavailable("Alpaca news requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
        self.macro = macro
        self.http = RetryingHttpClient(
            http_cfg, headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}, transport=transport
        )

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        params = {"start": ensure_utc(since).isoformat().replace("+00:00", "Z"),
                  "end": now.isoformat().replace("+00:00", "Z"), "limit": 50, "sort": "desc"}
        if ticker:
            params["symbols"] = ticker
        data = (await self.http.get(self.URL, params=params)).json()
        out = []
        for n in data.get("news") or []:
            pub = pd.Timestamp(n["created_at"]).tz_convert("UTC").to_pydatetime()
            if pub > now:
                continue
            out.append(_item(ticker, n.get("headline", ""), n.get("source", "alpaca"), pub, n.get("url"),
                             item_id=f"alp-{n.get('id')}"))
        return sorted(out, key=lambda x: x.published_at)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        return self.macro.events(start, end)

    async def aclose(self) -> None:
        await self.http.aclose()


class FinnhubNewsProvider(NewsProvider):
    """Finnhub company news + earnings calendar (token sent as a header, never in the URL)."""

    name = "finnhub"
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, clock: Clock, api_key: str, http_cfg: HttpSection, macro: StaticMacroCalendar,
                 tickers: list[str], transport=None) -> None:
        super().__init__(clock)
        if not api_key:
            raise ProviderUnavailable("Finnhub requires FINNHUB_API_KEY")
        self.macro = macro
        self.tickers = tickers
        self.http = RetryingHttpClient(http_cfg, headers={"X-Finnhub-Token": api_key}, transport=transport)

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        if ticker is None:
            params = {"category": "general"}
            url = f"{self.BASE}/news"
        else:
            params = {"symbol": ticker, "from": to_et(since).date().isoformat(), "to": to_et(now).date().isoformat()}
            url = f"{self.BASE}/company-news"
        data = (await self.http.get(url, params=params)).json() or []
        out = []
        for n in data:
            pub = datetime.fromtimestamp(int(n.get("datetime", 0)), tz=UTC)
            if ensure_utc(since) <= pub <= now:
                out.append(_item(ticker, n.get("headline", ""), n.get("source", "finnhub"), pub, n.get("url"),
                                 item_id=f"fh-{n.get('id')}"))
        return sorted(out, key=lambda x: x.published_at)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        events = self.macro.events(start, end)
        try:
            data = (await self.http.get(f"{self.BASE}/calendar/earnings", params={
                "from": to_et(start).date().isoformat(), "to": to_et(end).date().isoformat()})).json()
        except ProviderError as exc:
            log.warning("finnhub earnings calendar unavailable: %s", exc)
            return events
        watch = set(self.tickers)
        for row in (data or {}).get("earningsCalendar", []):
            sym = row.get("symbol")
            if sym not in watch:
                continue
            d = date.fromisoformat(row["date"])
            hour = row.get("hour", "")
            t = time(8, 0) if hour == "bmo" else time(16, 5) if hour == "amc" else time(12, 0)
            when = et_to_utc(datetime.combine(d, t))
            if ensure_utc(start) <= when <= ensure_utc(end):
                events.append(MacroEvent(name=f"{sym} earnings ({hour or 'time tbd'})", kind=MacroEventKind.EARNINGS,
                                         scheduled_at=when, severity=Severity.HIGH, ticker=sym))
        return sorted(events, key=lambda e: e.scheduled_at)

    async def aclose(self) -> None:
        await self.http.aclose()


class CalendarOnlyNewsProvider(NewsProvider):
    """No headlines, macro calendar only. Used by --replay, where historical
    headlines are not available for free; blackout windows still apply."""

    name = "calendar_only"

    def __init__(self, clock: Clock, macro: StaticMacroCalendar) -> None:
        super().__init__(clock)
        self.macro = macro

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        return []

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        return self.macro.events(start, end)
