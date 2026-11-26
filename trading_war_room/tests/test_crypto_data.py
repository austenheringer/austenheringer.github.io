"""Kraken provider (mocked HTTP), synthetic 24/7 tape, crypto RSS/news, routers, asset specs."""

from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from warroom.config import HttpSection
from warroom.core.assets import AssetClass, AssetRouter, CryptoSchedule, Instrument, format_price, format_qty
from warroom.core.clock import SimulatedClock
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import MacroEventKind, Severity, Timeframe
from warroom.providers.base import ProviderError
from warroom.providers.crypto_data import (
    CryptoNewsProvider, KrakenDataProvider, MockCryptoNewsProvider, MockCryptoProvider, RoutedMarketData,
    exchange_status_event, kraken_pair, mentions_coin, parse_rss,
)
from warroom.providers.market_data import MockDataProvider
from warroom.providers.news_data import StaticMacroCalendar

UTC = timezone.utc
T0 = datetime(2025, 1, 18, 14, 3, tzinfo=UTC)  # a Saturday: crypto still trades


# ---------------------------------------------------------------- assets

def test_instrument_rounding_minimums_and_format(cfg):
    r = AssetRouter(MarketCalendar(), cfg.crypto)
    btc = r.instrument("BTC-USD")
    assert btc.is_crypto and btc.floor_qty(0.0123456789) == 0.01234567 and btc.floor_qty(-1) == 0
    assert not btc.meets_minimum(0.00001, 95_000) and btc.meets_minimum(0.0001, 95_000)
    assert btc.taker_fee_pct == 0.004 and btc.maker_fee_pct == 0.0025 and not btc.allow_short
    aapl = r.instrument("AAPL")
    assert aapl.asset_class is AssetClass.EQUITY and aapl.floor_qty(170.9) == 170 and aapl.allow_short
    assert r.normalize("btc") == "BTC-USD" and r.normalize("$eth") == "ETH-USD" and r.normalize("tsla") == "TSLA"
    assert format_qty("BTC-USD", 0.012340) == "0.01234 BTC" and format_qty("AAPL", 170.0) == "170 shares"
    assert format_price("BTC-USD", 95123.456) == "95,123.46" and format_price("DOGE-USD", 0.123456) == "0.123456" and format_price("AAPL", 101.234) == "101.23"
    assert r.instrument("DOGE-USD", price_hint=0.15).price_decimals == 6


def test_crypto_schedule_never_closes():
    s = CryptoSchedule()
    sess = s.session_for(T0)
    assert s.is_open(T0) and sess.eod_flat is None and sess.orb_end == sess.open and not sess.past_eod_flat(T0)
    assert sess.open == datetime(2025, 1, 18, tzinfo=UTC) and len(s.previous_trading_days(date(2025, 1, 18), 5)) == 5


# ---------------------------------------------------------------- Kraken

def _ohlc_rows(start, n, step_min, price=95_000.0):
    rows = []
    for i in range(n):
        t = int((start + timedelta(minutes=i * step_min)).timestamp())
        p = price + i
        rows.append([t, str(p), str(p + 5), str(p - 5), str(p + 1), str(p), "1.5", 42])
    return rows


def _kraken(handler_state):
    def h(req):
        path = req.url.path
        handler_state.setdefault("calls", []).append(path)
        if path.endswith("/OHLC"):
            if handler_state.get("rate_limit", 0) > 0:
                handler_state["rate_limit"] -= 1
                return httpx.Response(200, json={"error": ["EAPI:Rate limit exceeded"], "result": {}})
            interval = int(req.url.params["interval"])
            start = datetime.fromtimestamp(int(req.url.params["since"]), UTC)
            n = int((T0 - start).total_seconds() // (60 * interval)) + 1
            return httpx.Response(200, json={"error": [], "result": {"XXBTZUSD": _ohlc_rows(start, max(n, 0), interval), "last": 0}})
        if path.endswith("/Ticker"):
            return httpx.Response(200, json={"error": [], "result": {"XXBTZUSD": {"a": ["95010.0", "1", "1.0"],
                                                                                  "b": ["95000.0", "1", "1.0"], "c": ["95005.0", "0.1"]}}})
        if path.endswith("/SystemStatus"):
            return httpx.Response(200, json={"error": [], "result": {"status": handler_state.get("status", "online")}})
        if path.endswith("/AssetPairs"):
            return httpx.Response(200, json={"error": [], "result": {"XXBTZUSD": {"wsname": "XBT/USD", "lot_decimals": 8,
                                                                                   "ordermin": "0.0001", "costmin": "0.5",
                                                                                   "pair_decimals": 1}}})
        return httpx.Response(404)
    return h


async def _no_sleep(_):
    pass


async def test_kraken_bars_drop_forming_bar_and_parse():
    state = {}
    k = KrakenDataProvider(SimulatedClock(T0), HttpSection(max_retries=2), transport=httpx.MockTransport(_kraken(state)),
                           sleep=_no_sleep)
    assert kraken_pair("BTC-USD") == "XBTUSD" and kraken_pair("ETH-USD") == "ETHUSD"
    bars = await k.get_bars("BTC-USD", Timeframe.M1, T0 - timedelta(minutes=30))
    assert str(bars.index.tz) == "UTC" and bars.index[-1] == pd.Timestamp(T0 - timedelta(minutes=1))  # 14:03 bar forming
    assert (bars["high"] >= bars["close"]).all()
    q = await k.get_quote("BTC-USD")
    assert (q.bid, q.ask, q.last) == (95000.0, 95010.0, 95005.0)
    await k.aclose()


async def test_kraken_cache_and_rate_limit_retry():
    state = {"rate_limit": 2}
    clk = SimulatedClock(T0)
    k = KrakenDataProvider(clk, HttpSection(max_retries=3), transport=httpx.MockTransport(_kraken(state)), sleep=_no_sleep)
    await k.get_bars("BTC-USD", Timeframe.M5, T0 - timedelta(hours=2))
    n = len(state["calls"])
    await k.get_bars("BTC-USD", Timeframe.M5, T0 - timedelta(hours=2))  # within refresh window: cached
    assert len(state["calls"]) == n
    await k.aclose()


async def test_kraken_errors_raise():
    def h(req):
        return httpx.Response(200, json={"error": ["EQuery:Unknown asset pair"], "result": {}})
    k = KrakenDataProvider(SimulatedClock(T0), HttpSection(max_retries=1), transport=httpx.MockTransport(h), sleep=_no_sleep)
    with pytest.raises(ProviderError, match="Unknown asset pair"):
        await k.get_quote("FOO-USD")
    await k.aclose()


async def test_kraken_status_and_asset_pairs():
    state = {"status": "maintenance"}
    k = KrakenDataProvider(SimulatedClock(T0), HttpSection(), transport=httpx.MockTransport(_kraken(state)), sleep=_no_sleep)
    ev = await exchange_status_event(k, k.clock)
    assert ev.kind is MacroEventKind.HALT and ev.ongoing and "maintenance" in ev.name
    state["status"] = "online"
    assert await exchange_status_event(k, k.clock) is None
    inst = await k.instruments(["BTC-USD"], 0.004, 0.0025, False)
    assert inst[0].min_qty == 0.0001 and inst[0].qty_step == 1e-8 and inst[0].price_decimals == 1
    await k.aclose()


# ---------------------------------------------------------------- mock 24/7 tape

async def test_mock_crypto_trades_weekends_without_lookahead():
    clk = SimulatedClock(T0)
    m = MockCryptoProvider(clk)
    b1 = await m.get_bars("BTC-USD", Timeframe.M1, T0 - timedelta(hours=3))
    assert len(b1) == 180 and b1.index[-1] == pd.Timestamp(T0 - timedelta(minutes=1))
    b15 = await m.get_bars("ETH-USD", Timeframe.M15, T0 - timedelta(days=7))
    assert len(b15) > 600  # 7 days, 24/7
    clk.advance(timedelta(hours=1))
    later = await m.get_bars("BTC-USD", Timeframe.M1, T0 - timedelta(hours=3))
    assert later.loc[b1.index].equals(b1)
    q = await m.get_quote("SOL-USD")
    assert q.bid < q.ask and 50 < q.last < 1000


async def test_mock_btc_leads_alts():
    m = MockCryptoProvider(SimulatedClock(T0))
    btc = (await m.get_bars("BTC-USD", Timeframe.M5, T0 - timedelta(hours=12))).close.pct_change().dropna()
    eth = (await m.get_bars("ETH-USD", Timeframe.M5, T0 - timedelta(hours=12))).close.pct_change().dropna()
    assert btc.corr(eth) > 0.4


# ---------------------------------------------------------------- news

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>CoinDesk</title>
<item><title>Bitcoin slides after exchange hack drains hot wallet</title><link>https://x/1</link>
<pubDate>Sat, 18 Jan 2025 13:30:00 +0000</pubDate></item>
<item><title>Solana rallies on ETF inflows</title><link>https://x/2</link><pubDate>Sat, 18 Jan 2025 12:00:00 GMT</pubDate></item>
<item><title>Future story</title><link>https://x/3</link><pubDate>Sat, 18 Jan 2025 20:00:00 +0000</pubDate></item>
<item><title>No date</title></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Feed</title>
<entry><title>Ethereum network upgrade goes live</title><link href="https://y/1"/><updated>2025-01-18T10:00:00Z</updated></entry></feed>"""


def test_parse_rss_and_atom():
    items = parse_rss(RSS)
    assert len(items) == 3 and items[0][3] == "CoinDesk" and items[0][1] == datetime(2025, 1, 18, 13, 30, tzinfo=UTC)
    assert parse_rss(ATOM)[0][2] == "https://y/1"
    with pytest.raises(ProviderError):
        parse_rss("not xml")


def test_coin_matching():
    assert mentions_coin("Bitcoin slides", "BTC-USD") and mentions_coin("BTC breaks out", "BTC-USD")
    assert mentions_coin("Ether ETFs see inflows", "ETH-USD") and not mentions_coin("Solar stocks rally", "SOL-USD")


async def test_crypto_news_provider(cal):
    clk = SimulatedClock(T0)
    feed = lambda r: httpx.Response(200, text=RSS)  # noqa: E731
    p = CryptoNewsProvider(clk, ["https://feed.test/rss"], HttpSection(), StaticMacroCalendar(cal, None),
                           MockCryptoProvider(clk), transport=httpx.MockTransport(feed))
    btc = await p.get_news("BTC-USD", T0 - timedelta(hours=6))
    assert [n.headline[:7] for n in btc] == ["Bitcoin"] and btc[0].sentiment < 0 and btc[0].severity is Severity.HIGH
    assert all(n.published_at <= T0 for n in await p.get_news(None, T0 - timedelta(days=1)))
    await p.aclose()


async def test_mock_crypto_news_no_lookahead(cfg, cal):
    clk = SimulatedClock(T0)
    m = MockCryptoProvider(clk)
    n = MockCryptoNewsProvider(clk, cfg.crypto.symbols, scenario_fn=m.scenario)
    items = await n.get_news(None, T0 - timedelta(days=1))
    assert items and all(i.published_at <= T0 for i in items)


# ---------------------------------------------------------------- router

async def test_routed_market_data(cfg, cal):
    clk = SimulatedClock(datetime(2025, 1, 15, 16, 0, tzinfo=UTC))
    r = AssetRouter(cal, cfg.crypto)
    routed = RoutedMarketData(r, MockDataProvider(clk, cal), MockCryptoProvider(clk))
    assert (await routed.get_quote("AAPL")).last < 1000 and (await routed.get_quote("BTC-USD")).last > 10_000
    assert routed.real_quotes_for("BTC-USD") and routed.scenario("BTC-USD", date(2025, 1, 15))
    no_crypto = RoutedMarketData(r, MockDataProvider(clk, cal), None)
    with pytest.raises(ProviderError, match="disabled"):
        await no_crypto.get_quote("BTC-USD")
