"""Crypto trading rules: fee-aware risk math, fractional sizing and minimums, spot no-short,
24/7 timing, time stops, maker/taker fees, desk routing, and the bar-bookmark regression."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.helpers import news, portfolio, position
from warroom.agents.risk_agent import (
    FeeModel, RiskAgent, compute_targets, fee_aware_target, net_reward, net_reward_risk, risk_per_unit,
)
from warroom.agents.technical_agent import MacroBias, TechnicalAgent, TickerContext
from warroom.config import ExecutionSection, RiskSection, Secrets, parse_config
from warroom.core.assets import AssetClass, AssetRouter, CryptoSchedule
from warroom.core.clock import SimulatedClock
from warroom.core.market_calendar import MarketCalendar
from warroom.execution.paper_broker import Bar, OrderRejected, PaperBroker
from warroom.models.schemas import (
    Direction, ExitReason, IndicatorSnapshot, MarketRegime, Mode, PlanStatus, Quote, SetupType, TradePlan, TradeSetup,
)

UTC = timezone.utc
L, S = Direction.LONG, Direction.SHORT
WED = datetime(2025, 1, 15, 3, 0, tzinfo=UTC)   # 22:00 ET Tuesday: stocks closed, crypto open
SAT = datetime(2025, 1, 18, 12, 0, tzinfo=UTC)
KRAKEN = FeeModel(0.004, 0.0025)


# ================================================================== fee math

def test_fee_model_math():
    assert risk_per_unit(100, 98, KRAKEN) == pytest.approx(2 + 0.4 + 0.392)
    assert net_reward(L, 100, 104, KRAKEN) == pytest.approx(4 - 0.4 - 0.26)
    tp = fee_aware_target(L, 100, 98, 2.0, KRAKEN)
    assert net_reward_risk(L, 100, 98, tp, KRAKEN) == pytest.approx(2.0)          # pays 2R AFTER fees
    tps = fee_aware_target(S, 100, 102, 2.0, KRAKEN)
    assert tps < 100 and net_reward_risk(S, 100, 102, tps, KRAKEN) == pytest.approx(2.0)


def test_zero_fees_reduce_to_plain_r_multiples():
    assert fee_aware_target(L, 100, 99, 2.0, FeeModel()) == pytest.approx(102.0)
    t = compute_targets(L, 100, 99, 2.0, 3.0, {}, atr=1.0, fees=FeeModel())
    assert (t.tp1, t.tp2) == (102.0, 103.0)


def test_tight_stops_cannot_pay_fees():
    """The reason crypto uses 15m bars: a 0.2% stop needs a 2.9% move to net 2R at Kraken fees."""
    tp = fee_aware_target(L, 100, 99.8, 2.0, KRAKEN)
    assert (tp - 100) / 100 > 0.025


# ================================================================== risk agent (crypto)

def _cfg(cfg, **crypto):
    raw = cfg.model_dump(mode="json")
    raw["crypto"].update(crypto)
    return parse_config(raw)


def _snap(symbol="BTC-USD", price=95_000.0, atr=600.0, **kw):
    base = dict(ticker=symbol, timestamp=WED - timedelta(minutes=1), price=price, vwap=price * 0.995, ema9=price * 0.999,
                ema21=price * 0.996, ema50=price * 0.99, rsi14=58.0, atr14=atr, rvol=1.4,
                swing_low=price - 0.8 * atr, swing_high=price + 0.8 * atr, range_high=price * 0.998, range_low=price * 0.97)
    base.update(kw)
    return IndicatorSnapshot(**base)


def _ctx(router, snap, now=WED, spread=0.0002, prior_high=None, prior_low=None):
    half = snap.price * spread / 2
    q = Quote(ticker=snap.ticker, timestamp=now, bid=snap.price - half, ask=snap.price + half, last=snap.price)
    return TickerContext(ticker=snap.ticker, snapshot=snap, session=CryptoSchedule().session_for(now), quote=q,
                         has_real_quotes=True, last_bar_close=now - timedelta(seconds=30), prior_high=prior_high,
                         prior_low=prior_low, asset_class=AssetClass.CRYPTO, instrument=router.instrument(snap.ticker, snap.price))


def _setup(snap, direction=L, conf=0.7):
    return TradeSetup(ticker=snap.ticker, direction=direction, setup_type=SetupType.RANGE_BREAKOUT, detected_at=WED,
                      trigger_price=snap.price, confidence=conf, indicators=snap)


def _risk(cfg, now=WED, **crypto):
    c = _cfg(cfg, **crypto) if crypto else cfg
    cal = MarketCalendar()
    router = AssetRouter(cal, c.crypto)
    return RiskAgent(c, cal, SimulatedClock(now), Mode.PAPER, router=router), router


def test_crypto_approved_overnight_with_fractional_size_and_net_rr(cfg):
    agent, router = _risk(cfg)
    snap = _snap()
    a, n = agent.evaluate(_setup(snap), _ctx(router, snap), news("BTC-USD"), portfolio(), None)
    assert a.approved, a.veto_reasons                      # 22:00 ET: equities closed, crypto fine
    assert 0 < a.position_size < 1 and a.position_size == router.instrument("BTC-USD").floor_qty(a.position_size)
    assert a.reward_risk == pytest.approx(2.0, abs=0.02)    # measured AFTER fees
    assert a.position_size == min(a.risk_shares, a.buying_power_shares)    # dual cap (25% of cash binds for BTC)
    assert 0 < a.dollar_risk <= 1000 + 1e-6                                 # never above 1% of equity incl. fees
    assert n.est_fees > 0 and n.entry_basis == "ask"
    assert a.entry == round(a.entry, 1)                     # BTC price precision from instrument


def test_crypto_short_vetoed_on_spot(cfg):
    agent, router = _risk(cfg)
    snap = _snap()
    a, _ = agent.evaluate(_setup(snap, S), _ctx(router, snap), news("BTC-USD"), portfolio(), None)
    assert any("short selling is not available" in v for v in a.veto_reasons)


def test_crypto_position_cap_and_weekend_risk(cfg):
    agent, router = _risk(cfg)
    snap = _snap()
    held = [position("ETH-USD"), position("SOL-USD")]
    a, _ = agent.evaluate(_setup(snap), _ctx(router, snap), news("BTC-USD"), portfolio(positions=held), None)
    assert any("crypto positions" in v for v in a.veto_reasons)
    wk, router2 = _risk(cfg, now=SAT)
    a2, _ = wk.evaluate(_setup(snap), _ctx(router2, snap, now=SAT), news("BTC-USD"), portfolio(), None)
    assert a2.approved and any("weekend" in w for w in a2.warnings)
    weekday, _ = agent.evaluate(_setup(snap), _ctx(router, snap), news("BTC-USD"), portfolio(), None)
    assert a2.risk_shares == pytest.approx(weekday.risk_shares / 2, rel=1e-6)   # half the risk budget


def test_crypto_minimum_order_veto(cfg):
    agent, router = _risk(cfg)
    snap = _snap()
    a, _ = agent.evaluate(_setup(snap), _ctx(router, snap), news("BTC-USD"),
                          portfolio(equity=10, cash=10, sod=10), None)  # $2.50 buying power < 0.00005 BTC
    assert any("exchange minimum" in v for v in a.veto_reasons)


def test_crypto_spread_threshold_and_fees_exceed_reward(cfg):
    agent, router = _risk(cfg)
    snap = _snap()
    a, _ = agent.evaluate(_setup(snap), _ctx(router, snap, spread=0.004), news("BTC-USD"), portfolio(), None)
    assert any(v.startswith("spread") for v in a.veto_reasons)
    capped = _ctx(router, snap, prior_high=snap.price * 1.006)  # wall right above: gross move < fees
    a2, _ = agent.evaluate(_setup(snap), capped, news("BTC-USD"), portfolio(), None)
    assert any("fees exceed the reward" in v or "no room" in v or v.startswith("R:R") for v in a2.veto_reasons)


def test_crypto_round_trips_do_not_count_for_pdt(cfg):
    agent, _ = _risk(cfg)
    p = position("BTC-USD")
    p.qty, p.closed_at, p.status = 0, WED + timedelta(minutes=30), "CLOSED"
    agent.on_position_closed(p)
    assert agent.state.day_trade_dates == []


# ================================================================== broker (crypto)

def _broker(cfg, now=WED, cash=100_000.0):
    clk = SimulatedClock(now)
    cal = MarketCalendar()
    return PaperBroker(ExecutionSection(slippage_pct=0.0), RiskSection(), clk, cal, cash,
                       router=AssetRouter(cal, cfg.crypto)), clk


def _plan(symbol="BTC-USD", size=0.5, entry=95_000.0, stop=94_000.0, tp1=97_500.0, tp2=99_000.0, d=L):
    return TradePlan(debate_id="d", ticker=symbol, direction=d, setup_type="RANGE_BREAKOUT", status=PlanStatus.APPROVED,
                     entry=entry, stop_loss=stop, tp1=tp1, tp2=tp2, position_size=size, dollar_risk=500, created_at=WED)


async def test_crypto_fees_taker_entry_maker_targets_and_accounting(cfg):
    b, _ = _broker(cfg)
    f = await b.open_from_plan(_plan(), 94_990.0, 95_000.0, 94_995.0)
    assert f.qty == 0.5 and f.commission == pytest.approx(0.5 * 95_000 * 0.004)          # taker
    f1 = await b.on_bar("BTC-USD", Bar(97_000, 97_600, 96_900, 97_550), bar_time=WED + timedelta(minutes=5))
    assert f1[0].reason is ExitReason.TP1 and f1[0].qty == 0.25
    assert f1[0].commission == pytest.approx(0.25 * 97_500 * 0.0025)                      # maker
    assert b.position_for("BTC-USD").stop_loss == 95_000.0                                 # breakeven
    f2 = await b.on_bar("BTC-USD", Bar(98_800, 99_100, 98_700, 99_050), bar_time=WED + timedelta(minutes=9))
    assert f2[0].reason is ExitReason.TP2 and not b.open_positions
    net = 0.25 * 2_500 + 0.25 * 4_000 - (190 + 0.25 * 97_500 * 0.0025 + 0.25 * 99_000 * 0.0025)
    assert b.equity() == pytest.approx(100_000 + net, abs=1e-6)


async def test_crypto_no_eod_flat_but_time_stop(cfg):
    b, clk = _broker(cfg)
    await b.open_from_plan(_plan(), 94_990.0, 95_000.0, 95_000.0)
    clk.set(WED + timedelta(hours=20, minutes=55))   # 15:55 ET-equivalent passes: nothing happens
    assert await b.on_bar("BTC-USD", Bar.flat(95_100), bar_time=clk.now()) == []
    clk.set(WED + timedelta(hours=24))
    fills = await b.on_bar("BTC-USD", Bar.flat(95_200), bar_time=clk.now())
    assert fills[0].reason is ExitReason.TIME_STOP


async def test_crypto_tp1_takes_all_when_half_is_below_minimum(cfg):
    b, _ = _broker(cfg)
    await b.open_from_plan(_plan(size=0.00008), 94_990.0, 95_000.0, 95_000.0)  # half = 0.00004 < min 0.00005
    f = await b.on_bar("BTC-USD", Bar(97_000, 97_600, 96_900, 97_550), bar_time=WED + timedelta(minutes=5))
    assert f[0].reason is ExitReason.TP1 and f[0].qty == 0.00008 and not b.open_positions


async def test_crypto_order_rejections(cfg):
    b, _ = _broker(cfg, cash=3.0)
    with pytest.raises(OrderRejected, match="exchange minimum"):
        await b.open_from_plan(_plan(size=0.01), 94_990.0, 95_000.0, 95_000.0)
    b2, _ = _broker(cfg)
    with pytest.raises(OrderRejected, match="short selling"):
        await b2.open_from_plan(_plan(d=S, stop=96_000, tp1=93_000, tp2=92_000), 95_000, 95_010, 95_005)


async def test_pre_entry_bars_are_ignored(cfg):
    """Regression: bars older than the position must never trigger its stop."""
    b, clk = _broker(cfg)
    await b.open_from_plan(_plan(), 94_990.0, 95_000.0, 95_000.0)
    stale = await b.on_bar("BTC-USD", Bar(93_000, 93_100, 92_000, 92_500), bar_time=WED - timedelta(minutes=30))
    assert stale == [] and b.position_for("BTC-USD") is not None


# ================================================================== technical (crypto)

async def test_crypto_context_and_range_breakout(cfg):
    from warroom.providers.crypto_data import MockCryptoProvider

    clk = SimulatedClock(SAT)
    cal = MarketCalendar()
    tech = TechnicalAgent(cfg, MockCryptoProvider(clk), cal, clk)
    ctx = await tech.build_context("BTC-USD")
    s = ctx.snapshot
    assert ctx.is_crypto and ctx.session.eod_flat is None and s.vwap and s.atr14 and s.range_high and s.range_low
    assert s.orb_high is None and s.premarket_high is None and ctx.prior_high is not None
    assert tech.entry_window_open(SAT, "BTC-USD") and not tech.entry_window_open(SAT, "AAPL")
    # engineered breakout bar
    idx = pd.date_range(SAT - timedelta(minutes=30), periods=2, freq="15min", tz="UTC")
    rh = s.range_high
    ctx.bars_5m = pd.DataFrame({"open": [rh * 0.998, rh * 0.999], "high": [rh * 0.999, rh * 1.004],
                                "low": [rh * 0.995, rh * 0.998], "close": [rh * 0.998, rh * 1.003], "volume": [10, 30]},
                               index=idx)
    ctx.snapshot = s.model_copy(update={"price": rh * 1.003, "vwap": rh * 0.99, "rsi14": 62})
    setups = tech.detect_setups(ctx, MacroBias(MarketRegime.RISK_ON, L, {}))
    assert any(st.setup_type is SetupType.RANGE_BREAKOUT and "BTC" in " ".join(st.rationale) for st in setups)
    assert all(st.direction is L for st in setups)          # spot: shorts never proposed


async def test_crypto_macro_bias_uses_btc(cfg):
    from warroom.providers.crypto_data import MockCryptoProvider

    clk = SimulatedClock(SAT)
    tech = TechnicalAgent(cfg, MockCryptoProvider(clk), MarketCalendar(), clk)
    b = await tech.macro_bias(AssetClass.CRYPTO)
    assert set(b.detail) == {"BTC-USD"}


# ================================================================== orchestrator (both desks)

async def test_crypto_desk_runs_while_stocks_stand_by(cfg):
    from warroom.core.market_calendar import MarketCalendar as MC
    from warroom.orchestrator import Orchestrator
    from warroom.providers.crypto_data import (
        MockCryptoNewsProvider, MockCryptoProvider, RoutedMarketData, RoutedNewsProvider,
    )
    from warroom.providers.market_data import MockDataProvider
    from warroom.providers.news_data import CalendarOnlyNewsProvider, StaticMacroCalendar

    clk = SimulatedClock(SAT)
    cal = MC()
    router = AssetRouter(cal, cfg.crypto)
    market = RoutedMarketData(router, MockDataProvider(clk, cal), MockCryptoProvider(clk))
    newsp = RoutedNewsProvider(router, CalendarOnlyNewsProvider(clk, StaticMacroCalendar(cal, None)),
                               MockCryptoNewsProvider(clk, cfg.crypto.symbols))
    o = await Orchestrator.build(cfg, Secrets.from_env({}), Mode.PAPER, clock=clk, db_path=":memory:",
                                 market=market, news_provider=newsp)
    await o.start()
    sub = o.bus.subscribe(name="t", maxsize=10_000)
    debates = []
    for _ in range(40):
        r = await o.run_cycle()
        assert r.standby                                   # equities closed on Saturday
        debates += r.debates
        clk.advance(timedelta(minutes=15))
    updates = set()
    while (e := sub.get_nowait()) is not None:
        if type(e).__name__ == "MarketUpdate":
            updates.add(e.ticker)
    assert updates == set(cfg.crypto.symbols)              # only crypto marked while stocks stand by
    assert all(t.ticker in cfg.crypto.symbols for t in debates)
    status = await o.handle_command("status")
    assert "crypto regime" in status
    reply = await o.handle_command("analyze btc")
    assert "BTC-USD" in reply and o.war_room.transcripts.get("BTC-USD")
    assert "BTC-USD" in await o.handle_command("why BTC")
    await o.shutdown()


async def test_execute_resets_bar_bookmark(cfg):
    from tests.conftest import et as _et
    from warroom.orchestrator import Orchestrator

    clk = SimulatedClock(_et(11, 0))
    o = await Orchestrator.build(cfg, Secrets.from_env({}), Mode.SMOKE, clock=clk, db_path=":memory:")
    await o.start()
    o._last_bar["AAPL"] = _et(9, 31)  # stale bookmark from an earlier trade
    await o.run_cycle()
    ctx = o.contexts["AAPL"]
    from tests.test_broker import plan as eq_plan
    p = eq_plan(ticker="AAPL", entry=ctx.snapshot.price, stop=ctx.snapshot.price * 0.99,
                tp1=ctx.snapshot.price * 1.02, tp2=ctx.snapshot.price * 1.03)
    await o._execute(p, ctx)
    assert o._last_bar["AAPL"] == ctx.last_bar_close - timedelta(minutes=1)
    await o.shutdown()
