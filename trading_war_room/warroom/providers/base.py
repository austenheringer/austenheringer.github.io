"""Provider interfaces and shared bar utilities.

Bars are exchanged as pandas DataFrames (index: tz-aware UTC DatetimeIndex of
bar OPEN times; columns: open, high, low, close, volume). Indicator math is
vectorised, so building thousands of pydantic Candles per scan would be pure
overhead; :func:`bars_to_candles` exists for the few places that need models.

Look-ahead rule: providers return only COMPLETED bars, i.e. bars whose
``open_time + timeframe <= clock.now()``. A bar still forming is never
visible to agents, in live or replay mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from warroom.core.clock import Clock, ensure_utc
from warroom.models.schemas import Candle, MacroEvent, NewsItem, Quote, Timeframe

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


class ProviderError(Exception):
    """A provider failed in a way the caller should handle (retry later / degrade)."""


class ProviderUnavailable(ProviderError):
    """A provider cannot be constructed (missing optional package or credentials)."""


# --------------------------------------------------------------------------- bar helpers


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in BAR_COLUMNS},
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce provider output into the canonical bar frame and drop bad rows.

    Rejects naive indexes (refuse to guess a timezone), sorts, de-duplicates,
    drops NaN/non-positive prices and repairs nothing silently except
    clamping high/low to contain open/close (vendor rounding noise).
    """
    if df is None or len(df) == 0:
        return empty_bars()
    df = df.rename(columns=str.lower)
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ProviderError(f"bar frame missing columns {missing}")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ProviderError("bar index is timezone-naive; providers must return aware timestamps")
    df = df[BAR_COLUMNS].astype("float64").copy()
    df.index = idx.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


def completed_only(df: pd.DataFrame, timeframe: Timeframe, now: datetime) -> pd.DataFrame:
    """Drop bars that have not finished forming at ``now``."""
    if df.empty:
        return df
    cutoff = pd.Timestamp(ensure_utc(now)) - pd.Timedelta(minutes=timeframe.minutes)
    return df[df.index <= cutoff]


def resample_bars(df_1m: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate 1m bars into 5m/15m bars aligned to the epoch (09:30 ET aligns
    for both EST and EDT). Bars labelled by their open time."""
    if timeframe is Timeframe.M1 or df_1m.empty:
        return df_1m
    out = df_1m.resample(
        f"{timeframe.minutes}min", label="left", closed="left", origin="epoch"
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"])


def bars_to_candles(df: pd.DataFrame, ticker: str, timeframe: Timeframe) -> list[Candle]:
    return [
        Candle(
            ticker=ticker, timeframe=timeframe, timestamp=ts.to_pydatetime(),
            open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
        )
        for ts, r in zip(df.index, df.itertuples(index=False))
    ]


# --------------------------------------------------------------------------- interfaces


class MarketDataProvider(ABC):
    name: str = "base"
    #: False when the source has no genuine bid/ask (spread checks must not trust synthetic quotes).
    has_real_quotes: bool = False

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    @abstractmethod
    async def get_bars(
        self, ticker: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """Completed bars with open time in [start, min(end, now)]."""

    @abstractmethod
    async def get_quote(self, ticker: str) -> Quote | None:
        """Latest quote, or None if the source has no real bid/ask."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network clients."""


class NewsProvider(ABC):
    name: str = "base"

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    @abstractmethod
    async def get_news(self, ticker: str | None, since: datetime) -> list[NewsItem]:
        """Headlines published in [since, now]. Never returns future items."""

    @abstractmethod
    async def get_macro_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        """Scheduled macro/earnings/halt events in [start, end] (may be in the future)."""

    async def aclose(self) -> None:  # noqa: B027
        pass


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        """Return the model's text reply. Raise on any failure."""

    async def aclose(self) -> None:  # noqa: B027
        pass


def window(now: datetime, minutes_before: int, minutes_after: int) -> tuple[datetime, datetime]:
    now = ensure_utc(now)
    return now - timedelta(minutes=minutes_before), now + timedelta(minutes=minutes_after)
