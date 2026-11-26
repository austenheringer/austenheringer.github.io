"""Indicator math vs. independent straightforward reference implementations."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.conftest import D, et
from warroom.analysis import indicators as ind


def _frame(closes, highs=None, lows=None, vols=None, start=None):
    n = len(closes)
    idx = pd.date_range(start or et(9, 30), periods=n, freq="5min", tz="UTC")
    closes = np.asarray(closes, float)
    return pd.DataFrame({"open": closes, "high": highs if highs is not None else closes + 0.5,
                         "low": lows if lows is not None else closes - 0.5, "close": closes,
                         "volume": vols if vols is not None else np.full(n, 1000.0)}, index=idx)


def test_ema_matches_recursive_definition():
    x = pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17], dtype=float)
    p = 3
    k = 2 / (p + 1)
    ref = [x[0]]
    for v in x[1:]:
        ref.append(v * k + ref[-1] * (1 - k))
    got = ind.ema(x, p)
    assert got.iloc[:p - 1].isna().all()
    assert np.allclose(got.iloc[p - 1:], ref[p - 1:])


def _wilder_rsi(closes, n):
    d = np.diff(closes)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[0], l[0]
    out = [np.nan]
    for i in range(len(d)):
        if i > 0:
            ag = (ag * (n - 1) + g[i]) / n
            al = (al * (n - 1) + l[i]) / n
        out.append(np.nan if i < n - 1 else (100.0 if al == 0 else 100 - 100 / (1 + ag / al)))
    return np.array(out)


def test_rsi_wilder():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 60))
    got = ind.rsi(pd.Series(closes), 14).to_numpy()
    ref = _wilder_rsi(closes, 14)
    mask = ~np.isnan(ref)
    assert np.allclose(got[mask], ref[mask], atol=1e-9)
    assert ((got[mask] >= 0) & (got[mask] <= 100)).all()


def test_rsi_edge_cases():
    assert ind.rsi(pd.Series(np.arange(1, 40, dtype=float)), 14).iloc[-1] == 100.0     # only gains
    assert ind.rsi(pd.Series(np.arange(40, 1, -1, dtype=float)), 14).iloc[-1] == pytest.approx(0.0)
    assert ind.rsi(pd.Series(np.full(30, 5.0)), 14).iloc[-1] == 50.0                     # flat


def test_true_range_and_atr():
    df = _frame([10, 12, 9], highs=np.array([10.5, 12.5, 11.0]), lows=np.array([9.5, 10.0, 8.5]))
    tr = ind.true_range(df).to_numpy()
    assert tr.tolist() == [1.0, 2.5, 3.5]  # 2nd: max(2.5, |12.5-10|, |10-10|); 3rd: max(2.5, |11-12|, |8.5-12|)
    flat = _frame(np.full(40, 50.0))
    assert ind.atr(flat, 14).iloc[-1] == pytest.approx(1.0)  # constant range of 1.0


def test_vwap_manual():
    df = _frame([10, 11, 12], highs=np.array([10, 11, 12.0]), lows=np.array([10, 11, 12.0]),
                vols=np.array([100.0, 200.0, 700.0]))
    assert ind.vwap(df).iloc[-1] == pytest.approx((10 * 100 + 11 * 200 + 12 * 700) / 1000)
    df.loc[:, "volume"] = 0
    assert ind.vwap(df).isna().all()  # no volume -> undefined, not a crash


def test_opening_range_requires_full_window(cal):
    sess = cal.session(D)
    idx = pd.date_range(sess.open, periods=20, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": np.r_[np.full(15, 101.0), np.full(5, 105.0)],
                       "low": np.r_[np.full(15, 99.0), np.full(5, 98.0)], "close": 100.0, "volume": 1.0}, index=idx)
    assert ind.opening_range(df, sess) == (101.0, 99.0)       # bars after 09:45 excluded
    assert ind.opening_range(df.iloc[:5], sess) is None       # window not complete


def test_premarket_and_prior_day(cal):
    sess = cal.session(D)
    idx = pd.date_range(et(8, 0), periods=120, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    df.loc[pd.Timestamp(et(8, 10)), "high"] = 103.0
    df.loc[pd.Timestamp(et(9, 45)), "high"] = 110.0          # regular hours: not premarket
    assert ind.premarket_range(df, sess) == (103.0, 100.0)
    y = D - timedelta(days=1)
    five = pd.concat([_frame([50, 52, 51], start=et(9, 30, y)), _frame([60, 61], start=et(9, 30))])
    assert ind.prior_day_levels(five, D) == (52.5, 49.5, 51.0)


def test_relative_volume_time_of_day_matched():
    frames = [_frame(np.full(78, 100.0), vols=np.full(78, 1000.0), start=et(9, 30, D - timedelta(days=k)))
              for k in (1, 2, 6)]  # three prior sessions (skips the weekend), 1000/bar
    today = _frame(np.full(6, 100.0), vols=np.full(6, 2000.0), start=et(9, 30))  # 09:30..09:55 at 2x
    df = pd.concat(frames + [today]).sort_index()
    assert ind.relative_volume(df, et(10, 0), 10) == pytest.approx(2.0)


def test_regular_hours_filter(cal):
    idx = pd.date_range(et(8, 0), et(17, 0), freq="30min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    rth = ind.regular_hours(df, cal)
    assert rth.index[0] == pd.Timestamp(et(9, 30)) and rth.index[-1] == pd.Timestamp(et(15, 30))


def test_swing_levels():
    df = _frame([10, 12, 11, 9, 10], highs=np.array([10, 13, 11, 9, 10.0]), lows=np.array([9, 11, 10, 8, 9.0]))
    assert ind.swing_levels(df, 3) == (11.0, 8.0)
