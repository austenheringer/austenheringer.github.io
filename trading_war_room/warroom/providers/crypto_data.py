"""Crypto data: Kraken public REST (no keys), a synthetic 24/7 mock, crypto news,
and routers that send each symbol to the equity or crypto provider.

Kraken notes
------------
* Bitcoin is ``XBT`` on Kraken (BTC-USD -> XBTUSD). Response keys use legacy
  names (``XXBTZUSD``), so results are read by position, not by name.
* OHLC returns at most the 720 most recent bars per interval: 12 h of 1m,
  2.5 days of 5m, 7.5 days of 15m. Enough for EMA50/ATR warm-up and a 7-day
  time-of-day RVOL baseline on 15m bars, NOT for deep history (use CSVs for
  replay).
* Errors come back as HTTP 200 with ``{"error": [...]}``. Rate-limit errors
  are retried with exponential backoff; anything else raises ProviderError.
* ``/SystemStatus`` != ``online`` (maintenance, cancel_only, post_only) is
  surfaced as an ongoing exchange halt, which blacks out new crypto entries.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET_XML
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from warroom.config import HttpSection
from warroom.core.assets import AssetClass, AssetRouter, Instrument
from warroom.core.clock import UTC, Clock, ensure_utc
from warroom.models.schemas import MacroEvent, MacroEventKind, NewsItem, Quote, Severity, Timeframe
from warroom.providers.base import (
    MarketDataProvider,
    NewsProvider,
    ProviderError,
    completed_only,
    empty_bars,
    normalize_bars,
    resample_bars,
)
from warroom.providers.http import RetryingHttpClient, backoff_delay
from warroom.providers.sentiment import score_headline

log = logging.getLogger(__name__)

_KRAKEN_ALIASES = {"BTC": "XBT", "DOGE": "XDG"}


def kraken_pair(symbol: str) -> str:
    base, quote = symbol.split("-")
    return f"{_KRAKEN_ALIASES.get(base, base)}{quote}"


# ============================================================================ Kraken


class KrakenDataProvider(MarketDataProvider):
    name = "kraken"
    has_real_quotes = True
    BASE = "https://api.kraken.com/0/public"

    def __init__(self, clock: Clock, http_cfg: HttpSection, *, min_refresh_seconds: float = 20,
                 transport=None, sleep=asyncio.sleep) -> None:
        super().__init__(clock)
        self.http_cfg = http_cfg
        self.http = RetryingHttpClient(http_cfg, transport=transport, sleep=sleep)
        self._sleep = sleep
        self.min_refresh = timedelta(seconds=min_refresh_seconds)
        self._cache: dict[tuple[str, int], tuple[pd.DataFrame, datetime]] = {}

    async def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(self.http_cfg.max_retries + 1):
            data = (await self.http.get(f"{self.BASE}/{path}", params=params)).json()
            errors = data.get("error") or []
            if not errors:
                return data.get("result") or {}
            if any("rate limit" in e.lower() or "too many" in e.lower() for e in errors) and attempt < self.http_cfg.max_retries:
                await self._sleep(backoff_delay(attempt, self.http_cfg.backoff_base_seconds, self.http_cfg.backoff_max_seconds))
                continue
            raise ProviderError(f"Kraken {path}: {'; '.join(errors)}")
        raise ProviderError(f"Kraken {path}: rate limited")  # pragma: no cover

    @staticmethod
    def _first(result: dict) -> object:
        for k, v in result.items():
            if k != "last":
                return v
        raise ProviderError("Kraken returned an empty result")

    async def _ohlc(self, symbol: str, interval: int, since: datetime) -> pd.DataFrame:
        result = await self._get("OHLC", {"pair": kraken_pair(symbol), "interval": interval,
                                          "since": int(ensure_utc(since).timestamp())})
        rows = self._first(result)
        if not rows:
            return empty_bars()
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
        return normalize_bars(df[["open", "high", "low", "close", "volume"]].astype(float))

    async def get_bars(self, ticker: str, timeframe: Timeframe, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        now = self.clock.now()
        start = ensure_utc(start)
        end = min(ensure_utc(end), now) if end else now
        key = (ticker, timeframe.minutes)
        cached = self._cache.get(key)
        if cached is None or now - cached[1] >= self.min_refresh or (not cached[0].empty and cached[0].index[0] > pd.Timestamp(start)
                                                                   and cached[1] < now - timedelta(minutes=timeframe.minutes)):
            since = start if cached is None or cached[0].empty else min(start, cached[0].index[-1].to_pydatetime())
            fresh = await self._ohlc(ticker, timeframe.minutes, since - timedelta(minutes=timeframe.minutes))
            merged = fresh if cached is None else pd.concat([cached[0], fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._cache[key] = (merged, now)
        df = self._cache[key][0]
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return completed_only(df, timeframe, now)  # Kraken's last row is the bar still forming

    async def get_quote(self, ticker: str) -> Quote | None:
        result = await self._get("Ticker", {"pair": kraken_pair(ticker)})
        t = self._first(result)
        try:
            ask, bid, last = float(t["a"][0]), float(t["b"][0]), float(t["c"][0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Kraken ticker for {ticker} malformed: {exc}") from exc
        if bid <= 0 or ask < bid:
            return None
        return Quote(ticker=ticker, timestamp=self.clock.now(), bid=bid, ask=ask, last=last or (bid + ask) / 2)

    async def system_status(self) -> str:
        return str((await self._get("SystemStatus")).get("status", "unknown"))

    async def instruments(self, symbols: list[str], taker: float, maker: float, allow_short: bool) -> list[Instrument]:
        """Live lot sizes / minimums from Kraken AssetPairs (overrides config defaults)."""
        result = await self._get("AssetPairs", {"pair": ",".join(kraken_pair(s) for s in symbols)})
        by_ws = {v.get("wsname", ""): v for v in result.values()}
        out = []
        for s in symbols:
            base, quote = s.split("-")
            info = by_ws.get(f"{_KRAKEN_ALIASES.get(base, base)}/{quote}") or by_ws.get(f"{base}/{quote}")
            if not info:
                continue
            lot = int(info.get("lot_decimals", 8))
            out.append(Instrument(symbol=s, asset_class=AssetClass.CRYPTO, qty_step=10 ** -lot,
                                  min_qty=float(info.get("ordermin", 0) or 0), min_notional=float(info.get("costmin", 0) or 0),
                                  price_decimals=int(info.get("pair_decimals", 2)), taker_fee_pct=taker,
                                  maker_fee_pct=maker, allow_short=allow_short))
        return out

    async def aclose(self) -> None:
        await self.http.aclose()


# ============================================================================ mock (24/7)

# symbol -> (reference price, avg 1m volume in coins, per-minute return vol, beta to BTC, base spread)
_CRYPTO_PROFILES = {
    "BTC-USD": (95_000.0, 6.0, 0.0010, 1.0, 0.00008),
    "ETH-USD": (3_300.0, 70.0, 0.0013, 1.2, 0.00012),
    "SOL-USD": (180.0, 900.0, 0.0019, 1.5, 0.00025),
}
CRYPTO_SCENARIOS = ("range_breakout", "range_breakdown", "trend_up", "trend_down", "squeeze_fade", "chop")


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _crypto_profile(symbol: str):
    if symbol in _CRYPTO_PROFILES:
        return _CRYPTO_PROFILES[symbol]
    r = np.random.default_rng(_h("cprofile", symbol))
    return (float(r.uniform(0.1, 500)), float(r.uniform(500, 50_000)), float(r.uniform(0.0015, 0.003)),
            float(r.uniform(1.2, 2.0)), float(r.uniform(0.0003, 0.001)))


@lru_cache(maxsize=256)
def _btc_factor(seed: int, day: date) -> np.ndarray:
    rng = np.random.default_rng(_h(seed, "btcfactor", day))
    mood = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
    return rng.standard_t(4, 1440) * 0.0008 + mood * 0.00004


@lru_cache(maxsize=4096)
def _cshock(seed: int, symbol: str, ordinal: int) -> float:
    return float(np.random.default_rng(_h(seed, "cshock", symbol, ordinal)).normal(0, 0.012))


@lru_cache(maxsize=512)
def generate_crypto_day(seed: int, symbol: str, day: date) -> pd.DataFrame:
    """1440 one-minute bars for a UTC day (plus a 'spread' column)."""
    ref, base_vol, vol, beta, base_spread = _crypto_profile(symbol)
    rng = np.random.default_rng(_h(seed, symbol, day))
    anchor = ref * float(np.exp(sum(_cshock(seed, symbol, day.toordinal() - k) for k in range(30))))
    scn = CRYPTO_SCENARIOS[_h(seed, "cscn", symbol, day) % len(CRYPTO_SCENARIOS)]
    rets = _btc_factor(seed, day) * beta + rng.standard_t(4, 1440) * vol * 0.6
    t = np.arange(1440)
    drift = np.zeros(1440)
    vmult = np.ones(1440)
    k = int(rng.integers(300, 1100))  # event minute
    if scn in ("range_breakout", "range_breakdown"):
        s = 1 if scn == "range_breakout" else -1
        drift[k:k + 20] = s * 0.5 * vol
        vmult[k:k + 20] = 3.5
        drift[k + 20:k + 200] = s * 0.08 * vol
    elif scn in ("trend_up", "trend_down"):
        s = 1 if scn == "trend_up" else -1
        drift += s * 0.06 * vol + 0.2 * vol * np.sin(2 * np.pi * t / 90) * -s
    elif scn == "squeeze_fade":
        drift[k:k + 6] = 2.5 * vol
        vmult[k:k + 6] = 6.0
        drift[k + 6:k + 120] = -0.3 * vol
    rets += drift
    log_path = np.log(anchor) + np.cumsum(rets)
    # Bridge to the next day's anchor so 00:00 UTC is continuous: a 24/7 market has no overnight gap.
    next_anchor = ref * float(np.exp(sum(_cshock(seed, symbol, day.toordinal() + 1 - k) for k in range(30))))
    log_path += (np.log(next_anchor) - log_path[-1]) * (t + 1) / 1440
    close = np.exp(log_path)
    open_ = np.concatenate([[anchor], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.6, 1440)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.6, 1440)))
    hour = t / 60.0
    seasonal = 0.6 + 0.9 * np.exp(-((hour - 14.5) ** 2) / 8) + 0.3 * np.exp(-((hour - 2) ** 2) / 6)  # US + Asia peaks
    weekend = 0.6 if day.weekday() >= 5 else 1.0
    volume = base_vol * seasonal * vmult * weekend * (0.6 + 0.4 * np.minimum(np.abs(rets) / vol, 4)) \
        * rng.lognormal(0, 0.3, 1440)
    spread = base_spread * rng.lognormal(0, 0.25, 1440) * (1.5 if day.weekday() >= 5 else 1.0)
    spread[vmult > 3] *= 6
    idx = pd.date_range(datetime.combine(day, datetime.min.time(), tzinfo=UTC), periods=1440, freq="1min", tz="UTC",
                        name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "spread": spread},
                        index=idx)


class MockCryptoProvider(MarketDataProvider):
    name = "mock-crypto"
    has_real_quotes = True

    def __init__(self, clock: Clock, seed: int = 42) -> None:
        super().__init__(clock)
        self.seed = seed

    def _bars_1m(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        frames, d = [], ensure_utc(start).date()
        while d <= ensure_utc(end).date():
            frames.append(generate_crypto_day(self.seed, symbol, d))
            d += timedelta(days=1)
        df = pd.concat(frames)
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    async def get_bars(self, ticker: str, timeframe: Timeframe, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        now = self.clock.now()
        start = ensure_utc(start)
        end = min(ensure_utc(end), now) if end else now
        if start > end:
            return empty_bars()
        df = completed_only(self._bars_1m(ticker, start, end), Timeframe.M1, now).drop(columns=["spread"])
        return normalize_bars(completed_only(resample_bars(df, timeframe), timeframe, now))

    async def get_quote(self, ticker: str) -> Quote | None:
        now = self.clock.now()
        df = completed_only(self._bars_1m(ticker, now - timedelta(hours=2), now), Timeframe.M1, now)
        if df.empty:
            return None
        last, sp = float(df["close"].iloc[-1]), float(df["spread"].iloc[-1])
        half = last * sp / 2
        return Quote(ticker=ticker, timestamp=df.index[-1].to_pydatetime() + timedelta(minutes=1),
                     bid=last - half, ask=last + half, last=last)

    def scenario(self, symbol: str, d: date) -> str:
        return CRYPTO_SCENARIOS[_h(self.seed, "cscn", symbol, d) % len(CRYPTO_SCENARIOS)]

    async def system_status(self) -> str:
        return "online"


# ============================================================================ news

COIN_NAMES = {
    "BTC": ("bitcoin", "btc"), "ETH": ("ethereum", "ether", "eth"), "SOL": ("solana", "sol"),
    "XRP": ("xrp", "ripple"), "DOGE": ("dogecoin", "doge"), "ADA": ("cardano", "ada"),
    "AVAX": ("avalanche", "avax"), "LINK": ("chainlink", "link"), "LTC": ("litecoin", "ltc"),
}


def mentions_coin(headline: str, symbol: str) -> bool:
    import re

    base = symbol.split("-")[0]
    names = COIN_NAMES.get(base, (base.lower(),))
    low = headline.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", low) for n in names)


def parse_rss(xml_text: str, source_default: str = "RSS") -> list[tuple[str, datetime, str | None, str]]:
    """Stdlib RSS 2.0 / Atom parser -> [(title, published_utc, link, source)]."""
    try:
        root = ET_XML.fromstring(xml_text)
    except ET_XML.ParseError as exc:
        raise ProviderError(f"feed is not valid XML: {exc}") from exc
    loc = lambda tag: tag.rsplit("}", 1)[-1].lower()  # noqa: E731
    source = source_default
    for el in root.iter():
        if loc(el.tag) == "title" and el.text:
            source = el.text.strip()
            break
    out = []
    for item in root.iter():
        if loc(item.tag) not in ("item", "entry"):
            continue
        f: dict[str, str] = {}
        for c in item:
            name = loc(c.tag)
            if name == "link" and c.get("href"):
                f["link"] = c.get("href")
            elif c.text:
                f.setdefault(name, c.text.strip())
        title = f.get("title")
        raw = f.get("pubdate") or f.get("published") or f.get("updated")
        if not title or not raw:
            continue
        try:
            pub = parsedate_to_datetime(raw) if "," in raw or raw[:3].isalpha() else datetime.fromisoformat(raw.replace("Z", "+00:00"))
            pub = ensure_utc(pub)
        except (TypeError, ValueError):
            continue
        out.append((title, pub, f.get("link"), source))
    return out


class CryptoNewsProvider(NewsProvider):
    """Crypto RSS headlines + macro calendar (FOMC/CPI/NFP hit crypto hard, and crypto is
    trading when they print) + exchange status as a halt."""

    name = "crypto-rss"

    def __init__(self, clock: Clock, feeds: list[str], http_cfg: HttpSection, macro, market: MarketDataProvider | None,
                 transport=None) -> None:
        super().__init__(clock)
        self.feeds = feeds
        self.macro = macro
        self.market = market
        self.http = RetryingHttpClient(http_cfg, headers={"User-Agent": "Mozilla/5.0 trading-war-room"}, transport=transport)

    async def _fetch(self, url: str):
        try:
            return parse_rss((await self.http.get(url)).text)
        except ProviderError as exc:
            log.warning("crypto feed %s failed: %s", url.split("?")[0], exc)
            return []

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        batches = await asyncio.gather(*(self._fetch(u) for u in self.feeds))
        out = []
        for title, pub, link, source in (x for b in batches for x in b):
            if not (ensure_utc(since) <= pub <= now):
                continue
            if ticker is None or mentions_coin(title, ticker):
                s, sev, _ = score_headline(title)
                out.append(NewsItem(id=hashlib.sha1((link or title).encode()).hexdigest()[:12], ticker=ticker,
                                    headline=title, source=source, published_at=pub, url=link, sentiment=s, severity=sev))
        return sorted(out, key=lambda n: n.published_at)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        events = [e for e in self.macro.events(start, end) if e.ticker is None]  # no equity earnings
        status = await exchange_status_event(self.market, self.clock)
        return sorted(events + ([status] if status else []), key=lambda e: e.scheduled_at)

    async def aclose(self) -> None:
        await self.http.aclose()


async def exchange_status_event(market: MarketDataProvider | None, clock: Clock) -> MacroEvent | None:
    getter = getattr(market, "system_status", None)
    if getter is None:
        return None
    try:
        status = await getter()
    except ProviderError as exc:
        log.warning("exchange status unavailable: %s", exc)
        return None
    if status == "online":
        return None
    return MacroEvent(name=f"Exchange status '{status}'", kind=MacroEventKind.HALT, scheduled_at=clock.now() - timedelta(seconds=1),
                      severity=Severity.HIGH, ongoing=True, source="exchange")


_CRYPTO_TEMPLATES = {
    1: ["{n} rallies as spot ETF inflows hit a record", "{n} surges after major exchange listing",
        "{n} jumps as network upgrade goes live", "Institutional buyers accumulate {n}, on-chain data shows"],
    -1: ["{n} slides after exchange hack drains hot wallet", "{n} tumbles as long liquidations top $500 million",
         "Regulator sues issuer, {n} falls", "{n} drops as ETF outflows accelerate"],
    0: ["{n} trades sideways ahead of US data", "{n} options open interest builds into month-end"],
}


class MockCryptoNewsProvider(NewsProvider):
    """Deterministic crypto headlines + the mock macro schedule. No look-ahead."""

    name = "mock-crypto-news"
    _TONE = {"range_breakout": 1, "trend_up": 1, "range_breakdown": -1, "trend_down": -1, "squeeze_fade": -1, "chop": 0}

    def __init__(self, clock: Clock, symbols: list[str], equity_mock_news=None, seed: int = 42, scenario_fn=None) -> None:
        super().__init__(clock)
        self.symbols = symbols
        self.seed = seed
        self.equity_news = equity_mock_news
        self._scenario_fn = scenario_fn

    def _day(self, d: date) -> list[NewsItem]:
        out = []
        for s in self.symbols:
            rng = np.random.default_rng(_h(self.seed, "cnews", s, d))
            tone = self._TONE.get(self._scenario_fn(s, d), 0) if self._scenario_fn else 0
            if tone and rng.random() < 0.25:
                tone = -tone
            name = COIN_NAMES.get(s.split("-")[0], (s,))[0].title()
            for _ in range(int(rng.integers(1, 4))):
                m = int(rng.integers(0, 1440))
                pick = tone if rng.random() < 0.7 else 0
                tmpl = _CRYPTO_TEMPLATES[pick][int(rng.integers(0, len(_CRYPTO_TEMPLATES[pick])))]
                title = tmpl.format(n=name)
                sent, sev, _ = score_headline(title)
                out.append(NewsItem(id=f"cmock-{d}-{s}-{m}", ticker=s, headline=title, source="MockChain",
                                    published_at=datetime.combine(d, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=m),
                                    sentiment=sent, severity=sev))
        return out

    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        now = self.clock.now()
        out, d = [], ensure_utc(since).date()
        while d <= now.date():
            out.extend(self._day(d))
            d += timedelta(days=1)
        return sorted((n for n in out if ensure_utc(since) <= n.published_at <= now and (ticker is None or n.ticker == ticker)),
                      key=lambda n: n.published_at)

    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        if self.equity_news is None:
            return []
        return [e for e in await self.equity_news.get_macro_events(start, end) if e.ticker is None]


# ============================================================================ routers


class RoutedMarketData(MarketDataProvider):
    """Dispatches by asset class so agents keep using one provider."""

    def __init__(self, router: AssetRouter, equity: MarketDataProvider, crypto: MarketDataProvider | None) -> None:
        super().__init__(equity.clock)
        self.router = router
        self.equity = equity
        self.crypto = crypto
        self.name = f"{equity.name}+{crypto.name}" if crypto else equity.name

    def _for(self, symbol: str) -> MarketDataProvider:
        if self.router.is_crypto(symbol):
            if self.crypto is None:
                raise ProviderError(f"crypto is disabled; no data for {symbol}")
            return self.crypto
        return self.equity

    @property
    def has_real_quotes(self) -> bool:  # type: ignore[override]
        return self.equity.has_real_quotes

    def real_quotes_for(self, symbol: str) -> bool:
        return self._for(symbol).has_real_quotes

    async def get_bars(self, ticker, timeframe, start, end=None):
        return await self._for(ticker).get_bars(ticker, timeframe, start, end)

    async def get_quote(self, ticker):
        return await self._for(ticker).get_quote(ticker)

    def scenario(self, symbol: str, d: date) -> str:
        p = self._for(symbol)
        return p.scenario(symbol, d) if hasattr(p, "scenario") else "chop"

    async def aclose(self) -> None:
        await self.equity.aclose()
        if self.crypto is not None:
            await self.crypto.aclose()


class RoutedNewsProvider(NewsProvider):
    def __init__(self, router: AssetRouter, equity: NewsProvider, crypto: NewsProvider | None) -> None:
        super().__init__(equity.clock)
        self.router = router
        self.equity = equity
        self.crypto = crypto
        self.name = f"{equity.name}+{crypto.name}" if crypto else equity.name
        self._current: str | None = None

    def _for(self, symbol: str | None) -> NewsProvider:
        if symbol and self.router.is_crypto(symbol) and self.crypto is not None:
            return self.crypto
        return self.equity

    async def get_news(self, ticker, since):
        return await self._for(ticker).get_news(ticker, since)

    async def get_macro_events(self, start, end):
        return await self.equity.get_macro_events(start, end)

    async def macro_events_for(self, symbol: str, start, end):
        return await self._for(symbol).get_macro_events(start, end)

    async def aclose(self) -> None:
        await self.equity.aclose()
        if self.crypto is not None:
            await self.crypto.aclose()
