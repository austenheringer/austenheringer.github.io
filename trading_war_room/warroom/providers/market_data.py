"""Market data providers: Mock (synthetic), yfinance, Alpaca (REST via httpx).

MockDataProvider
----------------
Deterministic per (seed, ticker, date), so a demo, a replay and a test all
see the same tape. Each day has:

* a shared *market path* (drives SPY/QQQ and, via beta, every stock), so the
  macro bias the Technical agent computes actually means something;
* a per-ticker *scenario* (ORB breakout/breakdown, trending with pullbacks,
  VWAP reclaim, spike-and-fade, chop) layered on fat-tailed noise;
* a U-shaped intraday volume profile, low pre/post-market volume, volume
  surges on breakout segments (so RVOL lights up), and occasional volatility
  spikes during which the quoted spread widens (exercises the spread veto).

Only completed bars up to ``clock.now()`` are ever returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from warroom.config import HttpSection
from warroom.core.clock import ET, Clock, ensure_utc, to_et
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import Quote, Timeframe
from warroom.providers.base import (
    MarketDataProvider,
    ProviderError,
    ProviderUnavailable,
    completed_only,
    empty_bars,
    normalize_bars,
    resample_bars,
)
from warroom.providers.http import RetryingHttpClient

log = logging.getLogger(__name__)

# ticker -> (reference price, avg session 1m volume, per-minute return vol, beta, base spread pct)
_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "SPY": (590.0, 120_000, 0.00045, 1.0, 0.00002),
    "QQQ": (510.0, 80_000, 0.00060, 1.2, 0.00003),
    "AAPL": (230.0, 90_000, 0.00090, 1.1, 0.00005),
    "MSFT": (420.0, 45_000, 0.00080, 1.0, 0.00006),
    "NVDA": (140.0, 600_000, 0.00160, 1.7, 0.00008),
    "TSLA": (350.0, 250_000, 0.00200, 2.0, 0.00010),
    "AMD": (140.0, 110_000, 0.00150, 1.6, 0.00010),
}
SCENARIOS = (
    "orb_breakout", "orb_breakdown", "trend_up_pullbacks", "trend_down_rallies",
    "vwap_reclaim", "spike_fade", "chop",
)
_DAY_START = time(4, 0)
_N_MIN = 16 * 60  # 04:00 -> 20:00 ET
_OPEN_IDX = 330    # 09:30 is minute 330 after 04:00


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _profile(ticker: str) -> tuple[float, float, float, float, float]:
    if ticker in _PROFILES:
        return _PROFILES[ticker]
    r = np.random.default_rng(_seed("profile", ticker))
    return (float(r.uniform(20, 300)), float(r.uniform(20_000, 200_000)), float(r.uniform(0.0008, 0.002)),
            float(r.uniform(0.8, 1.8)), float(r.uniform(0.00005, 0.0004)))


def scenario_for(seed: int, ticker: str, day: date) -> str:
    if ticker in ("SPY", "QQQ"):
        return "market"
    return SCENARIOS[_seed(seed, "scenario", ticker, day) % len(SCENARIOS)]


@lru_cache(maxsize=64)
def _market_returns(seed: int, day: date, session_len: int) -> np.ndarray:
    """Per-minute market (SPY-like) log returns for the whole 04:00-20:00 grid."""
    rng = np.random.default_rng(_seed(seed, "market", day))
    vol = 0.00045
    mood = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
    rets = rng.standard_t(5, _N_MIN) * vol * 0.75
    sess = slice(_OPEN_IDX, _OPEN_IDX + session_len)
    rets[sess] += mood * vol * 0.08
    rets[:_OPEN_IDX] *= 0.35
    rets[_OPEN_IDX + session_len:] *= 0.3
    return rets


def _scenario_drift(scn: str, vol: float, session_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (drift, volume multiplier) arrays over session minutes."""
    t = np.arange(session_len, dtype=float)
    drift = np.zeros(session_len)
    vmult = np.ones(session_len)

    def seg(a: int, b: int) -> slice:
        return slice(max(0, min(a, session_len)), max(0, min(b, session_len)))

    if scn in ("orb_breakout", "orb_breakdown"):
        s = 1 if scn == "orb_breakout" else -1
        start = int(rng.integers(18, 40))
        drift[seg(start, start + 12)] = s * 0.55 * vol
        vmult[seg(start, start + 12)] = 3.5
        drift[seg(start + 12, start + 150)] = s * 0.10 * vol
        vmult[seg(start + 12, start + 60)] = 1.8
    elif scn in ("trend_up_pullbacks", "trend_down_rallies"):
        s = 1 if scn == "trend_up_pullbacks" else -1
        drift[seg(0, 240)] = s * (0.12 * vol) + 0.28 * vol * np.sin(2 * np.pi * t[seg(0, 240)] / 45) * -s
        vmult[seg(0, 240)] = 1.4
    elif scn == "vwap_reclaim":
        drift[seg(0, 50)] = -0.22 * vol
        drift[seg(50, 170)] = 0.26 * vol
        vmult[seg(50, 80)] = 2.8
    elif scn == "spike_fade":
        k = int(rng.integers(60, 180))
        drift[seg(k, k + 5)] = 2.5 * vol
        vmult[seg(k, k + 5)] = 6.0
        drift[seg(k + 5, k + 70)] = -0.35 * vol
        vmult[seg(k + 5, k + 30)] = 2.0
    # chop: no drift
    return drift, vmult


@lru_cache(maxsize=512)
def generate_day(seed: int, ticker: str, day: date, session_len: int) -> pd.DataFrame:
    """Full 04:00-20:00 ET 1m bars for one ticker/day, plus a 'spread' column."""
    ref, base_vol, vol, beta, base_spread = _profile(ticker)
    rng = np.random.default_rng(_seed(seed, ticker, day))
    mkt = _market_returns(seed, day, session_len)
    scn = scenario_for(seed, ticker, day)

    # day-level anchor: deterministic wander around the reference price, plus an overnight gap
    anchor = ref * float(np.exp(np.random.default_rng(_seed(seed, "anchor", ticker, day.toordinal() // 5)).normal(0, 0.04)))
    gap = float(rng.normal(0, 0.006)) + (0.012 if scn == "orb_breakout" else -0.012 if scn == "orb_breakdown" else 0.0)
    p0 = anchor * (1 + gap)

    if ticker in ("SPY", "QQQ"):
        idio = rng.standard_t(5, _N_MIN) * vol * 0.25
        rets = mkt * beta + idio
    else:
        idio = rng.standard_t(4, _N_MIN) * vol * 0.7
        rets = mkt * beta + idio
    rets[:_OPEN_IDX] *= 0.5

    drift, vmult_s = _scenario_drift(scn, vol, session_len, rng)
    sess = slice(_OPEN_IDX, _OPEN_IDX + session_len)
    rets[sess] += drift

    # volatility spikes (also widen spreads)
    spike = np.zeros(_N_MIN, dtype=bool)
    for _ in range(int(rng.integers(0, 3))):
        k = _OPEN_IDX + int(rng.integers(20, max(21, session_len - 20)))
        n = int(rng.integers(4, 10))
        rets[k:k + n] *= 3.0
        spike[k:k + n] = True

    close = p0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[p0], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol * 0.6, _N_MIN))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.6, _N_MIN)))

    # volume: U-shape in session, thin outside, bursts on scenario segments
    volume = np.full(_N_MIN, base_vol * 0.04)
    st = np.arange(session_len, dtype=float)
    ushape = 1 + 2.5 * np.exp(-st / 25) + 1.6 * np.exp(-(session_len - st) / 25)
    volume[sess] = base_vol * ushape / ushape.mean() * vmult_s
    volume[_OPEN_IDX + session_len:] = base_vol * 0.06
    in_play = 1.8 if scn in ("orb_breakout", "orb_breakdown", "spike_fade", "vwap_reclaim") else 1.0
    day_mult = float(rng.lognormal(0, 0.25)) * in_play
    z = np.abs(rets) / vol
    volume = volume * day_mult * (0.6 + 0.4 * np.minimum(z, 4)) * rng.lognormal(0, 0.3, _N_MIN)
    volume = np.round(volume)

    spread = np.full(_N_MIN, base_spread) * rng.lognormal(0, 0.2, _N_MIN)
    spread[:_OPEN_IDX] *= 4
    spread[_OPEN_IDX + session_len:] *= 4
    spread[spike] *= 12

    start_utc = ensure_utc(datetime.combine(day, _DAY_START, tzinfo=ET))
    idx = pd.date_range(start_utc, periods=_N_MIN, freq="1min", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "spread": spread}, index=idx
    )


class MockDataProvider(MarketDataProvider):
    name = "mock"
    has_real_quotes = True  # synthetic, but internally consistent bid/ask by construction

    def __init__(self, clock: Clock, calendar: MarketCalendar, seed: int = 42) -> None:
        super().__init__(clock)
        self.calendar = calendar
        self.seed = seed

    def _day_frame(self, ticker: str, d: date) -> pd.DataFrame:
        sess = self.calendar.session(d)
        if sess is None:
            return empty_bars()
        session_len = int((sess.close - sess.open).total_seconds() // 60)
        return generate_day(self.seed, ticker, d, session_len)

    def _bars_1m(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        d = to_et(start).date()
        last = to_et(end).date()
        while d <= last:
            df = self._day_frame(ticker, d)
            if not df.empty:
                frames.append(df)
            d += timedelta(days=1)
        if not frames:
            return empty_bars()
        df = pd.concat(frames)
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    async def get_bars(self, ticker: str, timeframe: Timeframe, start: datetime,
                       end: datetime | None = None) -> pd.DataFrame:
        now = self.clock.now()
        start = ensure_utc(start)
        end = min(ensure_utc(end), now) if end else now
        if start > end:
            return empty_bars()
        df = self._bars_1m(ticker, start, end)
        df = completed_only(df, Timeframe.M1, now).drop(columns=["spread"], errors="ignore")
        df = resample_bars(df, timeframe)
        return normalize_bars(completed_only(df, timeframe, now))

    async def get_quote(self, ticker: str) -> Quote | None:
        now = self.clock.now()
        df = self._bars_1m(ticker, now - timedelta(days=4), now)
        df = completed_only(df, Timeframe.M1, now)
        if df.empty:
            return None
        row = df.iloc[-1]
        last = float(row["close"])
        half = last * float(row["spread"]) / 2
        return Quote(ticker=ticker, timestamp=df.index[-1].to_pydatetime() + timedelta(minutes=1),
                     bid=round(last - half, 4), ask=round(last + half, 4), last=round(last, 4))

    def scenario(self, ticker: str, d: date) -> str:
        return scenario_for(self.seed, ticker, d)


class YFinanceDataProvider(MarketDataProvider):
    """Yahoo Finance via the optional ``yfinance`` package. Unofficial, delayed,
    rate-limited and without bid/ask: suitable for experimentation only.
    Limits: 1m bars only for the last ~30 days (max 8 days per request);
    5m/15m for ~60 days."""

    name = "yfinance"
    has_real_quotes = False

    def __init__(self, clock: Clock) -> None:
        super().__init__(clock)
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailable("yfinance is not installed: pip install yfinance") from exc

    async def get_bars(self, ticker: str, timeframe: Timeframe, start: datetime,
                       end: datetime | None = None) -> pd.DataFrame:
        now = self.clock.now()
        start = ensure_utc(start)
        end = min(ensure_utc(end), now) if end else now
        history_limit_days = 29 if timeframe is Timeframe.M1 else 59
        max_days = 7 if timeframe is Timeframe.M1 else 59  # per-request window
        if (now - start) > timedelta(days=history_limit_days):
            raise ProviderError(f"yfinance does not serve {timeframe.value} bars that old ({start.date()})")

        def _fetch() -> pd.DataFrame:
            import yfinance as yf

            frames = []
            chunk = timedelta(days=max_days)
            s = start
            while s < end:
                e = min(s + chunk, end + timedelta(minutes=1))
                df = yf.Ticker(ticker).history(start=s, end=e, interval=timeframe.value,
                                               prepost=True, auto_adjust=False, raise_errors=False)
                if df is not None and len(df):
                    frames.append(df)
                s = e
            return pd.concat(frames) if frames else pd.DataFrame()

        try:
            raw = await asyncio.to_thread(_fetch)
        except Exception as exc:  # yfinance raises a zoo of exception types
            raise ProviderError(f"yfinance fetch failed for {ticker}: {exc}") from exc
        if raw.empty:
            return empty_bars()
        df = normalize_bars(raw)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return completed_only(df, timeframe, now)

    async def get_quote(self, ticker: str) -> Quote | None:
        return None  # no genuine bid/ask; Risk applies risk.unknown_spread_policy


class AlpacaDataProvider(MarketDataProvider):
    """Alpaca Market Data v2 over plain REST (httpx + retry/backoff).

    Uses the ``iex`` feed by default (free tier). IEX is a single venue: its
    quotes are real but often wider and thinner than the consolidated NBBO, so
    expect occasional spread vetoes on less liquid names. ``alpaca-py`` is not
    required.
    """

    name = "alpaca"
    has_real_quotes = True
    BASE = "https://data.alpaca.markets"
    _TF = {Timeframe.M1: "1Min", Timeframe.M5: "5Min", Timeframe.M15: "15Min"}

    def __init__(self, clock: Clock, api_key: str, secret_key: str, http_cfg: HttpSection,
                 feed: str = "iex", transport=None) -> None:
        super().__init__(clock)
        if not api_key or not secret_key:
            raise ProviderUnavailable("Alpaca data requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
        self.feed = feed
        self.http = RetryingHttpClient(
            http_cfg, headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}, transport=transport
        )

    async def get_bars(self, ticker: str, timeframe: Timeframe, start: datetime,
                       end: datetime | None = None) -> pd.DataFrame:
        now = self.clock.now()
        start = ensure_utc(start)
        end = min(ensure_utc(end), now) if end else now
        params = {
            "timeframe": self._TF[timeframe], "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"), "limit": 10000, "adjustment": "raw", "feed": self.feed,
        }
        rows: list[dict] = []
        token = None
        for _ in range(50):
            if token:
                params["page_token"] = token
            data = (await self.http.get(f"{self.BASE}/v2/stocks/{ticker}/bars", params=params)).json()
            rows.extend(data.get("bars") or [])
            token = data.get("next_page_token")
            if not token:
                break
        if not rows:
            return empty_bars()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], utc=True)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return completed_only(normalize_bars(df), timeframe, now)

    async def get_quote(self, ticker: str) -> Quote | None:
        data = (await self.http.get(f"{self.BASE}/v2/stocks/{ticker}/snapshot", params={"feed": self.feed})).json()
        q = data.get("latestQuote") or {}
        tr = data.get("latestTrade") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        last = float(tr.get("p") or 0) or (bid + ask) / 2
        if bid <= 0 or ask <= 0 or ask < bid or last <= 0:
            return None
        ts = pd.Timestamp(q.get("t") or tr.get("t")).tz_convert("UTC").to_pydatetime()
        return Quote(ticker=ticker, timestamp=ts, bid=bid, ask=ask, last=last)

    async def aclose(self) -> None:
        await self.http.aclose()
