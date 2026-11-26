from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import pytest

from tests.conftest import D, et
from warroom.config import HttpSection
from warroom.core.clock import SimulatedClock, et_to_utc, to_et
from warroom.models.schemas import MacroEventKind, Severity, Timeframe
from warroom.providers.base import ProviderError, normalize_bars
from warroom.providers.http import HttpError, RetryingHttpClient, backoff_delay
from warroom.providers.market_data import AlpacaDataProvider, MockDataProvider
from warroom.providers.news_data import AlpacaNewsProvider, FinnhubNewsProvider, MockNewsProvider, StaticMacroCalendar


# ---------------------------------------------------------------- mock market data

async def test_only_completed_bars_are_visible(cal):
    md = MockDataProvider(SimulatedClock(et(10, 7)), cal)
    b1 = await md.get_bars("NVDA", Timeframe.M1, et(4, 0))
    assert b1.index[-1] == pd.Timestamp(et(10, 6))      # the 10:06 bar completes at 10:07
    b5 = await md.get_bars("NVDA", Timeframe.M5, et(9, 30))
    assert b5.index[-1] == pd.Timestamp(et(10, 0))      # the 10:05 bar is still forming


async def test_resample_matches_1m(cal):
    md = MockDataProvider(SimulatedClock(et(11, 0)), cal)
    b1 = await md.get_bars("TSLA", Timeframe.M1, et(9, 30))
    b15 = await md.get_bars("TSLA", Timeframe.M15, et(9, 30))
    first = b1.loc[pd.Timestamp(et(9, 30)):pd.Timestamp(et(9, 44))]
    assert b15.iloc[0].high == pytest.approx(first.high.max())
    assert b15.iloc[0].low == pytest.approx(first.low.min())
    assert b15.iloc[0].volume == pytest.approx(first.volume.sum())
    assert b15.iloc[0].open == pytest.approx(first.open.iloc[0]) and b15.iloc[0].close == pytest.approx(first.close.iloc[-1])


async def test_history_is_immutable_and_deterministic(cal):
    clk = SimulatedClock(et(10, 30))
    md = MockDataProvider(clk, cal, seed=7)
    early = await md.get_bars("AAPL", Timeframe.M1, et(4, 0))
    clk.advance(timedelta(hours=2))
    later = await md.get_bars("AAPL", Timeframe.M1, et(4, 0))
    assert later.loc[early.index].equals(early)
    other = MockDataProvider(SimulatedClock(clk.now()), cal, seed=7)
    assert (await other.get_bars("AAPL", Timeframe.M1, et(4, 0))).equals(later)


async def test_market_structure(cal):
    md = MockDataProvider(SimulatedClock(et(15, 0)), cal)
    spy = (await md.get_bars("SPY", Timeframe.M1, et(9, 30))).close.pct_change().dropna()
    qqq = (await md.get_bars("QQQ", Timeframe.M1, et(9, 30))).close.pct_change().dropna()
    assert spy.corr(qqq) > 0.6
    b = await md.get_bars("NVDA", Timeframe.M1, et(4, 0))
    pre = b.loc[:pd.Timestamp(et(9, 29))].volume.mean()
    opening = b.loc[pd.Timestamp(et(9, 30)):pd.Timestamp(et(9, 45))].volume.mean()
    assert opening > 5 * pre
    q = await md.get_quote("NVDA")
    assert q.bid < q.ask and q.timestamp <= md.clock.now()


async def test_closed_days_have_no_bars(cal):
    md = MockDataProvider(SimulatedClock(et_to_utc(datetime(2025, 1, 18, 12))), cal)
    assert (await md.get_bars("AAPL", Timeframe.M1, et_to_utc(datetime(2025, 1, 18, 0)))).empty
    hist = await md.get_bars("AAPL", Timeframe.M5, et(9, 30, date(2025, 1, 6)))
    assert date(2025, 1, 9) not in {to_et(t).date() for t in hist.index}  # Carter closure


def test_normalize_bars_rejects_naive_and_cleans():
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]},
                      index=pd.to_datetime(["2025-01-15 15:00", "2025-01-15 15:01"]))
    with pytest.raises(ProviderError, match="naive"):
        normalize_bars(df)
    df.index = df.index.tz_localize("UTC")
    df.loc[df.index[0], "close"] = -1
    assert len(normalize_bars(df)) == 1


# ---------------------------------------------------------------- news

async def test_mock_news_has_no_lookahead(cal, cfg):
    clk = SimulatedClock(et(11, 0))
    md = MockDataProvider(clk, cal)
    nw = MockNewsProvider(clk, cal, cfg.app.tickers, scenario_fn=md.scenario)
    early = await nw.get_news(None, et(0, 0))
    assert early and all(i.published_at <= clk.now() for i in early)
    clk.advance(timedelta(hours=5))
    later = await nw.get_news(None, et(0, 0))
    assert {i.id for i in early} <= {i.id for i in later}
    events = await nw.get_macro_events(et(0, 0), et(23, 0))
    assert any(e.kind is MacroEventKind.FOMC for e in events)


def test_static_macro_calendar(cal):
    smc = StaticMacroCalendar(cal, user_file=None)
    fomc = smc.events(et(0, 0, date(2025, 1, 29)), et(23, 0, date(2025, 1, 29)))
    assert to_et(fomc[0].scheduled_at).strftime("%H:%M") == "14:00"
    nfp = smc.events(et(0, 0, date(2025, 2, 7)), et(23, 0, date(2025, 2, 7)))
    assert nfp[0].kind is MacroEventKind.NFP and nfp[0].source == "rule"


def test_user_macro_events(cal, tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("- {name: Powell speech, kind: OTHER, date: '2025-01-15', time_et: '11:00', severity: HIGH}\n")
    ev = StaticMacroCalendar(cal, user_file=f).events(et(0), et(23))
    assert [e.name for e in ev] == ["Powell speech"]
    f.write_text("- {name: broken, date: 'not-a-date', time_et: '11:00'}\n")
    with pytest.raises(ProviderError, match="invalid macro event"):
        StaticMacroCalendar(cal, user_file=f)


# ---------------------------------------------------------------- HTTP retry

def _client(handler, slept, retries=3):
    async def fake_sleep(s):
        slept.append(s)
    return RetryingHttpClient(HttpSection(max_retries=retries, backoff_base_seconds=0.5, backoff_max_seconds=30),
                              transport=httpx.MockTransport(handler), sleep=fake_sleep, rng=lambda: 0.5)


async def test_retry_honours_retry_after():
    n = {"c": 0}
    slept: list[float] = []

    def h(req):
        n["c"] += 1
        return httpx.Response(429, headers={"Retry-After": "2"}) if n["c"] < 3 else httpx.Response(200, json={"ok": 1})

    c = _client(h, slept)
    assert (await c.get("https://x.test/a")).json() == {"ok": 1}
    assert slept == [2.0, 2.0]
    await c.aclose()


async def test_retry_gives_up_on_5xx_with_exponential_jitter():
    slept: list[float] = []
    c = _client(lambda r: httpx.Response(503), slept)
    with pytest.raises(HttpError) as ei:
        await c.get("https://x.test/down")
    assert ei.value.status == 503 and slept == [0.25, 0.5, 1.0]
    await c.aclose()


async def test_no_retry_on_4xx_and_url_redacted():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        return httpx.Response(401)

    c = _client(h, [])
    with pytest.raises(HttpError) as ei:
        await c.get("https://x.test/a?token=SECRET")
    assert n["c"] == 1 and "SECRET" not in str(ei.value)
    await c.aclose()


async def test_transport_errors_retry():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        if n["c"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200)

    c = _client(h, [])
    assert (await c.get("https://x.test")).status_code == 200
    await c.aclose()


def test_backoff_cap():
    assert backoff_delay(10, 0.5, 30, lambda: 1.0) == 30
    assert backoff_delay(0, 0.5, 30, lambda: 1.0) == 0.5


# ---------------------------------------------------------------- Alpaca / Finnhub parsing

def _alpaca_handler(req):
    p = req.url.path
    if p.endswith("/bars"):
        if "page_token" not in req.url.params:
            return httpx.Response(200, json={"bars": [{"t": "2025-01-15T15:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}],
                                             "next_page_token": "p2"})
        return httpx.Response(200, json={"bars": [{"t": "2025-01-15T15:01:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 50},
                                                  {"t": "2025-01-15T15:02:00Z", "o": 1.8, "h": 2, "l": 1, "c": 1.9, "v": 50}],
                                         "next_page_token": None})
    if p.endswith("/snapshot"):
        return httpx.Response(200, json={"latestQuote": {"t": "2025-01-15T15:02:59Z", "bp": 1.89, "ap": 1.91},
                                         "latestTrade": {"p": 1.9, "t": "2025-01-15T15:02:58Z"}})
    if p.endswith("/news"):
        return httpx.Response(200, json={"news": [
            {"id": 1, "headline": "NVDA beats estimates, raises guidance", "created_at": "2025-01-15T14:00:00Z", "source": "b"},
            {"id": 2, "headline": "future item", "created_at": "2025-01-15T18:00:00Z"}]})
    return httpx.Response(404)


async def test_alpaca_bars_pagination_and_completion(cal):
    a = AlpacaDataProvider(SimulatedClock(et(10, 3)), "k", "s", HttpSection(), transport=httpx.MockTransport(_alpaca_handler))
    bars = await a.get_bars("NVDA", Timeframe.M1, et(9, 30))
    assert len(bars) == 3 and str(bars.index.tz) == "UTC"
    a.clock = SimulatedClock(et(10, 2))
    assert len(await a.get_bars("NVDA", Timeframe.M1, et(9, 30))) == 2
    q = await a.get_quote("NVDA")
    assert (q.bid, q.ask, q.last) == (1.89, 1.91, 1.9)
    await a.aclose()


async def test_alpaca_news_filters_future_and_scores(cal):
    n = AlpacaNewsProvider(SimulatedClock(et(10, 3)), "k", "s", HttpSection(), StaticMacroCalendar(cal, None),
                           transport=httpx.MockTransport(_alpaca_handler))
    items = await n.get_news("NVDA", et(0, 0))
    assert len(items) == 1 and items[0].sentiment > 0.5 and items[0].severity is Severity.HIGH
    await n.aclose()


async def test_finnhub_news_and_earnings(cal):
    def h(req):
        assert req.headers.get("X-Finnhub-Token") == "tok" and "token" not in req.url.params
        if req.url.path.endswith("/company-news"):
            return httpx.Response(200, json=[{"id": 9, "headline": "AAPL tumbles after downgrade to sell",
                                              "datetime": int(et(9, 0).timestamp()), "source": "x"}])
        return httpx.Response(200, json={"earningsCalendar": [
            {"symbol": "AAPL", "date": "2025-01-15", "hour": "amc"}, {"symbol": "ZZZZ", "date": "2025-01-15", "hour": "bmo"}]})

    f = FinnhubNewsProvider(SimulatedClock(et(10, 0)), "tok", HttpSection(), StaticMacroCalendar(cal, None), ["AAPL"],
                            transport=httpx.MockTransport(h))
    items = await f.get_news("AAPL", et(0, 0))
    assert items[0].sentiment < 0
    ev = await f.get_macro_events(et(0, 0), et(23, 0))
    assert [e.ticker for e in ev if e.kind is MacroEventKind.EARNINGS] == ["AAPL"]
    assert to_et(ev[-1].scheduled_at).strftime("%H:%M") == "16:05"
    await f.aclose()
