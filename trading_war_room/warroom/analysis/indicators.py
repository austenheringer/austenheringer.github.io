"""Pure, vectorised indicator math on canonical bar frames.

Conventions: RSI and ATR use Wilder's smoothing (alpha = 1/period), the
standard definition; EMAs use span-based smoothing with ``adjust=False`` (the
recursive form every charting platform uses). VWAP is anchored at the
regular-session open and uses the typical price (H+L+C)/3.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from warroom.core.clock import ensure_utc
from warroom.core.market_calendar import MarketCalendar, TradingSession


ET_TZ = "America/New_York"


def et_index(df: pd.DataFrame, tz: str = ET_TZ) -> pd.DatetimeIndex:
    """Vectorised UTC -> local conversion of a bar index (ET for equities, UTC for crypto)."""
    return df.index.tz_convert(tz)


def et_dates(df: pd.DataFrame, tz: str = ET_TZ) -> np.ndarray:
    return np.array(et_index(df, tz).date)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # flat or one-sided windows: no losses -> 100, no gains -> 0, neither -> 50
    out = out.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return out.where(avg_gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1)
    return tr.max(axis=1, skipna=True)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP over the given frame (pass ONE session's regular-hours bars)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    return (typical * df["volume"]).cumsum() / cum_vol.replace(0, np.nan)


def session_slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Bars with open time in [start, end)."""
    return df[(df.index >= pd.Timestamp(ensure_utc(start))) & (df.index < pd.Timestamp(ensure_utc(end)))]


def regular_hours(df: pd.DataFrame, calendar: MarketCalendar) -> pd.DataFrame:
    """Keep only bars inside each day's regular session (drops pre/post-market)."""
    if df.empty:
        return df
    days = et_dates(df)
    keep = np.zeros(len(df), dtype=bool)
    idx = df.index
    for d in np.unique(days):
        sess = calendar.session(d)
        if sess is None:
            continue
        keep |= (days == d) & (idx >= pd.Timestamp(sess.open)) & (idx < pd.Timestamp(sess.close))
    return df[keep]


def opening_range(df_1m: pd.DataFrame, session: TradingSession) -> tuple[float, float] | None:
    """High/low of the first ``orb_minutes`` of the session; None until the window has closed."""
    rng = session_slice(df_1m, session.open, session.orb_end)
    expected = int((session.orb_end - session.open).total_seconds() // 60)
    if len(rng) < max(1, expected - 2):  # tolerate a couple of missing prints, not a missing window
        return None
    return float(rng["high"].max()), float(rng["low"].min())


def premarket_range(df_1m: pd.DataFrame, session: TradingSession) -> tuple[float, float] | None:
    """High/low of bars on the session's ET date before the 09:30 open (data starts 04:00 ET)."""
    if df_1m.empty:
        return None
    pm = df_1m[(df_1m.index < pd.Timestamp(session.open)) & (et_dates(df_1m) == session.day)]
    if pm.empty:
        return None
    return float(pm["high"].max()), float(pm["low"].min())


def prior_day_levels(df_5m_rth: pd.DataFrame, today: date, tz: str = ET_TZ) -> tuple[float, float, float] | None:
    """(high, low, close) of the most recent session before ``today`` (ET dates; UTC for crypto)."""
    if df_5m_rth.empty:
        return None
    days = et_dates(df_5m_rth, tz)
    prior = sorted(d for d in np.unique(days) if d < today)
    if not prior:
        return None
    sub = df_5m_rth[days == prior[-1]]
    return float(sub["high"].max()), float(sub["low"].min()), float(sub["close"].iloc[-1])


def relative_volume(df_5m_rth: pd.DataFrame, now: datetime, lookback_days: int, tz: str = ET_TZ) -> float | None:
    """Cumulative volume today up to the current time-of-day, divided by the average
    cumulative volume at the same time-of-day over the prior ``lookback_days`` sessions.
    Time-of-day matching matters: comparing 10:00 volume to a full day's average
    would make every morning look dead."""
    if df_5m_rth.empty:
        return None
    eti = et_index(df_5m_rth, tz)
    days = np.array(eti.date)
    minutes = eti.hour * 60 + eti.minute
    vol = df_5m_rth["volume"].to_numpy()
    today = pd.Timestamp(now).tz_convert(tz).date()
    today_mask = days == today
    if not today_mask.any():
        return None
    cutoff = int(minutes[today_mask].max())
    today_cum = float(vol[today_mask].sum())
    prior_days = sorted(d for d in np.unique(days) if d < today)[-lookback_days:]
    hist = [float(vol[(days == d) & (minutes <= cutoff)].sum()) for d in prior_days]
    hist = [h for h in hist if h > 0]
    if not hist:
        return None
    return today_cum / float(np.mean(hist))


def rolling_range(df: pd.DataFrame, minutes: int, now: datetime) -> tuple[float, float] | None:
    """(high, low) of bars opened in the last ``minutes`` EXCLUDING the latest bar, so a
    breakout bar is compared with the range it is breaking, not with itself."""
    if len(df) < 3:
        return None
    prior = df.iloc[:-1]
    window = prior[prior.index >= pd.Timestamp(now) - pd.Timedelta(minutes=minutes)]
    if len(window) < 2:
        return None
    return float(window["high"].max()), float(window["low"].min())


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"])


def swing_levels(df: pd.DataFrame, lookback: int) -> tuple[float, float] | None:
    """(swing_high, swing_low) over the last ``lookback`` completed bars."""
    if len(df) < 2:
        return None
    tail = df.iloc[-lookback:]
    return float(tail["high"].max()), float(tail["low"].min())
