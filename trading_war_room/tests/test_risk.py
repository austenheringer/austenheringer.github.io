"""Risk agent: exact level math, dual-capped sizing and every veto path."""

from datetime import date, timedelta

import pytest

from tests.conftest import D, et
from tests.helpers import context, news, portfolio, position, setup, snapshot
from warroom.agents.risk_agent import (
    RiskAgent, compute_stop, compute_targets, position_size, reward_risk,
)
from warroom.config import parse_config
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import Direction, Mode, Position, RiskState, Severity, validate_levels

L, S = Direction.LONG, Direction.SHORT


# ------------------------------------------------------------------ pure math

def test_long_stop_uses_structure_when_within_atr():
    r = compute_stop(L, entry=100, atr=1.0, structural=99.0, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25)
    assert r.source == "structure" and r.stop == pytest.approx(98.9)


def test_long_stop_falls_back_to_atr_when_structure_too_far():
    r = compute_stop(L, 100, 1.0, structural=97.0, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25)
    assert r.source == "atr" and r.stop == pytest.approx(98.5)


def test_structure_on_wrong_side_is_ignored():
    r = compute_stop(L, 100, 1.0, structural=100.5, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25)
    assert r.source == "atr" and r.stop == pytest.approx(98.5)


def test_short_stop_mirrors():
    r = compute_stop(S, 100, 1.0, structural=101.0, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25)
    assert r.source == "structure" and r.stop == pytest.approx(101.1)
    r = compute_stop(S, 100, 1.0, structural=103.0, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25)
    assert r.source == "atr" and r.stop == pytest.approx(101.5)


def test_min_stop_distance_enforced():
    r = compute_stop(L, 100, 1.0, structural=99.95, atr_mult=1.5, buffer_atr=0.0, min_stop_atr=0.25)
    assert r.source == "min_distance" and r.stop == pytest.approx(99.75)


def test_atr_only_policy():
    r = compute_stop(L, 100, 1.0, structural=99.5, atr_mult=1.5, buffer_atr=0.1, min_stop_atr=0.25, policy="atr_only")
    assert r.source == "atr"


def test_invalid_atr_raises():
    with pytest.raises(ValueError):
        compute_stop(L, 100, 0.0, None, 1.5, 0.1, 0.25)


def test_long_targets_uncapped_and_ordering():
    t = compute_targets(L, 100, 99, 2.0, 3.0, {}, atr=1.0)
    assert (t.tp1, t.tp2) == (102.0, 103.0)
    assert validate_levels(L, 100, 99, t.tp1, t.tp2) is None


def test_short_targets_uncapped_and_ordering():
    t = compute_targets(S, 100, 101, 2.0, 3.0, {}, atr=1.0)
    assert (t.tp1, t.tp2) == (98.0, 97.0)
    assert validate_levels(S, 100, 101, t.tp1, t.tp2) is None


def test_long_target_capped_by_prior_day_high():
    t = compute_targets(L, 100, 99, 2.0, 3.0, {"prior-day high": 101.5}, atr=1.0)
    assert t.tp1 == pytest.approx(101.45) and "prior-day high" in t.capped_by[0]
    assert t.tp2 > t.tp1                                  # ordering preserved
    assert reward_risk(L, 100, 99, t.tp1) == pytest.approx(1.45)


def test_short_target_capped_by_two_levels():
    t = compute_targets(S, 100, 101, 2.0, 3.0, {"premarket low": 98.5, "prior-day low": 97.5}, atr=1.0)
    assert t.tp1 == pytest.approx(98.55) and t.tp2 == pytest.approx(97.55)


def test_levels_behind_entry_do_not_cap():
    t = compute_targets(L, 100, 99, 2.0, 3.0, {"ORB high": 99.5}, atr=1.0)
    assert (t.tp1, t.tp2) == (102.0, 103.0) and not t.capped_by


def test_fixed_r_policy_ignores_levels():
    t = compute_targets(L, 100, 99, 2.0, 3.0, {"prior-day high": 100.5}, atr=1.0, policy="fixed_r")
    assert t.tp1 == 102.0


def test_validate_levels_rejects_everything_wrong():
    assert validate_levels(L, 100, 100, 102, 103)                  # zero distance
    assert validate_levels(L, 100, 101, 102, 103)                  # stop above entry
    assert validate_levels(L, 100, 99, 103, 102)                   # tp2 < tp1
    assert validate_levels(S, 100, 99, 98, 97)                     # short stop below entry
    assert validate_levels(S, 100, 101, 98, 99)                    # short tp2 above tp1
    assert validate_levels(L, float("nan"), 99, 102, 103)


# ------------------------------------------------------------------ sizing

def test_dual_capped_sizing_risk_binds():
    # risk: 100k * 1% / 2.00 = 500 ; buying power: 100k * 25% / 50 = 500 -> tie; make risk bind
    assert position_size(100_000, 100_000, 50.0, 47.0, 0.01, 0.25) == (333, 500, 333)


def test_dual_capped_sizing_buying_power_binds():
    # risk: 1000 / 0.50 = 2000 ; buying power: 25_000 / 100 = 250
    assert position_size(100_000, 100_000, 100.0, 99.5, 0.01, 0.25) == (2000, 250, 250)


def test_sizing_uses_available_cash_not_equity():
    assert position_size(100_000, 20_000, 100.0, 99.0, 0.01, 0.25)[1] == 50


def test_zero_division_guard():
    assert position_size(100_000, 100_000, 100.0, 100.0, 0.01, 0.25) == (0, 0, 0)
    assert position_size(100_000, -5.0, 100.0, 99.0, 0.01, 0.25)[2] == 0


def test_short_sizing_math_is_symmetric():
    assert position_size(100_000, 100_000, 50.0, 53.0, 0.01, 0.25) == (333, 500, 333)


# ------------------------------------------------------------------ agent

def _agent(cfg, now=None, state=None, mode=Mode.PAPER, **risk_over):
    if risk_over:
        raw = cfg.model_dump()
        raw["risk"].update(risk_over)
        cfg = parse_config(raw)
    clk = SimulatedClock(now or et(11, 0))
    from warroom.core.market_calendar import MarketCalendar
    return RiskAgent(cfg, MarketCalendar(), clk, mode, state=state), clk


def test_clean_long_is_approved_with_exact_numbers(cfg):
    agent, _ = _agent(cfg)
    a, n = agent.evaluate(setup(), context(), news(), portfolio(), None)
    assert a.approved, a.veto_reasons
    entry = round(100.02 * 1.0005, 2)                 # ask * (1 + slippage)
    assert a.entry == entry and a.stop_loss == round(entry - (entry - 98.9), 2) == 98.9
    r = entry - 98.9
    assert a.tp1 == round(entry + 2 * r, 2) and a.tp2 == round(entry + 3 * r, 2)
    assert a.reward_risk == pytest.approx(2.0, abs=0.01)
    assert a.position_size == min(int(1000 // r), int(25_000 // entry)) and a.position_size >= 1
    assert a.dollar_risk == pytest.approx(a.position_size * r, abs=0.01)
    assert n.stop_source == "structure" and n.entry_basis == "ask"


def test_clean_short_is_approved_and_ordered(cfg):
    agent, _ = _agent(cfg)
    snap = snapshot(price=100.0, vwap=100.3, ema9=99.9, ema21=100.1)
    a, n = agent.evaluate(setup(direction=S, snap=snap), context(snap), news(), portfolio(), None)
    assert a.approved, a.veto_reasons
    assert a.stop_loss > a.entry > a.tp1 > a.tp2 and n.entry_basis == "bid"


def test_rr_veto_when_target_capped(cfg):
    agent, _ = _agent(cfg)
    a, n = agent.evaluate(setup(), context(prior_high=101.0), news(), portfolio(), None)
    assert not a.approved and any(v.startswith("R:R") and "prior-day high" in v for v in a.veto_reasons)
    assert a.position_size >= 0 and "TP1 capped" in n.target_caps[0]


def test_no_room_veto_when_level_at_entry(cfg):
    agent, _ = _agent(cfg)
    a, _ = agent.evaluate(setup(), context(prior_high=100.09), news(), portfolio(), None)
    assert any("no room" in v for v in a.veto_reasons)


def test_counter_trend_needs_blended_rr(cfg):
    agent, _ = _agent(cfg, counter_trend_min_reward_risk=2.6)
    a, n = agent.evaluate(setup(), context(), news(), portfolio(), macro_direction=S)
    assert n.counter_trend and any("counter-trend" in v for v in a.veto_reasons)
    agent2, _ = _agent(cfg)  # default 2.5: uncapped 2R/3R blends to exactly 2.5 -> passes
    assert agent2.evaluate(setup(), context(), news(), portfolio(), macro_direction=S)[0].approved


def test_daily_loss_limit_halts_including_unrealized(cfg):
    agent, _ = _agent(cfg, state=RiskState(trading_date=D, realized_pnl_today=-2000.0))
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(unrealized=-1000.0), None)
    assert any("daily loss limit" in v for v in a.veto_reasons) and agent.state.halted_for_day
    # realized-only configuration would not have halted at -2000
    agent2, _ = _agent(cfg, state=RiskState(trading_date=D, realized_pnl_today=-2000.0),
                       daily_loss_includes_unrealized=False)
    assert agent2.evaluate(setup(), context(), news(), portfolio(unrealized=-1000.0), None)[0].approved


def test_size_cut_to_remaining_daily_budget(cfg):
    agent, _ = _agent(cfg, state=RiskState(trading_date=D, realized_pnl_today=-2900.0))
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(), None)
    assert a.approved and a.dollar_risk <= 100.0 + 1e-6
    assert any("remaining daily loss budget" in w for w in a.warnings)


def test_consecutive_loss_cooldown(cfg):
    agent, clk = _agent(cfg)
    for i in range(3):
        p = Position(ticker="X", direction=L, entry_price=10, stop_loss=9, tp1=12, tp2=13, initial_qty=1, qty=0,
                     opened_at=et(10, 0), closed_at=et(10, 30 + i), realized_pnl=-1.0, status="CLOSED")
        agent.on_position_closed(p)
    assert agent.state.cooldown_until == et(10, 32) + timedelta(minutes=30)
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(), None)
    assert any("consecutive losses" in v for v in a.veto_reasons)
    clk.set(et(11, 3))
    assert agent.evaluate(setup(), context(now=et(11, 3)), news(), portfolio(), None)[0].approved


def test_win_resets_streak(cfg):
    agent, _ = _agent(cfg)
    mk = lambda pnl, m: Position(ticker="X", direction=L, entry_price=10, stop_loss=9, tp1=12, tp2=13, initial_qty=1,
                                 qty=0, opened_at=et(10, 0), closed_at=et(10, m), realized_pnl=pnl, status="CLOSED")
    agent.on_position_closed(mk(-1, 1))
    agent.on_position_closed(mk(-1, 2))
    agent.on_position_closed(mk(+1, 3))
    assert agent.state.consecutive_losses == 0


def test_max_positions_and_duplicate_exposure(cfg):
    agent, _ = _agent(cfg)
    three = [position("MSFT"), position("NVDA"), position("TSLA")]
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(positions=three), None)
    assert any("max concurrent" in v for v in a.veto_reasons)
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(positions=[position("AAPL")]), None)
    assert any("duplicate" in v for v in a.veto_reasons)


def test_spread_veto_and_unknown_spread_policy(cfg):
    agent, _ = _agent(cfg)
    a, _ = agent.evaluate(setup(), context(spread_pct=0.004), news(), portfolio(), None)
    assert any(v.startswith("spread") for v in a.veto_reasons)
    a, _ = agent.evaluate(setup(), context(real_quotes=False), news(), portfolio(), None)
    assert any("no real bid/ask" in v for v in a.veto_reasons)
    allow, _ = _agent(cfg, unknown_spread_policy="allow")
    a, _ = allow.evaluate(setup(), context(real_quotes=False), news(), portfolio(), None)
    assert a.approved and any("unverified" in w for w in a.warnings)


def test_stale_data_veto_flags_possible_halt(cfg):
    agent, _ = _agent(cfg)
    ctx = context(age_s=400)
    ctx.possible_halt = True
    a, _ = agent.evaluate(setup(), ctx, news(), portfolio(), None)
    assert any("stale data" in v and "halted" in v for v in a.veto_reasons)


def test_pdt_tracker(cfg):
    trades = [D - timedelta(days=1), D - timedelta(days=2), D]
    agent, _ = _agent(cfg, state=RiskState(trading_date=D, day_trade_dates=trades), enforce_pdt_rule=True)
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(equity=20_000, cash=20_000, sod=20_000), None)
    assert any("PDT" in v for v in a.veto_reasons)
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(equity=30_000, cash=30_000, sod=30_000), None)
    assert not any("PDT" in v for v in a.veto_reasons)       # above $25k: rule does not apply
    old = [date(2024, 12, 1)] * 5                              # outside the 5-session window
    agent2, _ = _agent(cfg, state=RiskState(trading_date=D, day_trade_dates=old), enforce_pdt_rule=True)
    assert agent2.evaluate(setup(), context(), news(), portfolio(equity=20_000, cash=20_000, sod=20_000), None)[0].approved


def test_pdt_disabled_by_default(cfg):
    agent, _ = _agent(cfg, state=RiskState(trading_date=D, day_trade_dates=[D] * 5))
    assert agent.evaluate(setup(), context(), news(), portfolio(equity=20_000, cash=20_000, sod=20_000), None)[0].approved


def test_news_blackout_and_high_impact_conflict(cfg):
    agent, _ = _agent(cfg)
    a, _ = agent.evaluate(setup(), context(), news(blackout=True), portfolio(), None)
    assert any("blackout" in v for v in a.veto_reasons)
    a, _ = agent.evaluate(setup(), context(), news(sentiment=-0.7, severity=Severity.HIGH, conflict=True), portfolio(), None)
    assert any("high-impact news" in v for v in a.veto_reasons)
    a, _ = agent.evaluate(setup(), context(), news(sentiment=-0.4, severity=Severity.MEDIUM, conflict=True), portfolio(), None)
    assert a.approved and any("leans against" in w for w in a.warnings)


@pytest.mark.parametrize("hh,mm,needle", [(9, 40, "opening-range window"), (15, 31, "cutoff"), (16, 30, "closed")])
def test_session_timing_vetoes(cfg, hh, mm, needle):
    agent, _ = _agent(cfg, now=et(hh, mm))
    a, _ = agent.evaluate(setup(), context(now=et(hh, mm)), news(), portfolio(), None)
    assert any(needle in v for v in a.veto_reasons)


def test_pause_and_resume(cfg):
    agent, _ = _agent(cfg)
    agent.pause()
    assert any("paused" in v for v in agent.evaluate(setup(), context(), news(), portfolio(), None)[0].veto_reasons)
    agent.resume()
    assert agent.evaluate(setup(), context(), news(), portfolio(), None)[0].approved


def test_missing_atr_vetoes_without_crashing(cfg):
    agent, _ = _agent(cfg)
    snap = snapshot(atr=None)
    a, _ = agent.evaluate(setup(snap=snap), context(snap), news(), portfolio(), None)
    assert any("ATR unavailable" in v for v in a.veto_reasons) and a.position_size == 0


def test_tiny_account_vetoes_when_size_below_one(cfg):
    agent, _ = _agent(cfg)
    a, _ = agent.evaluate(setup(), context(), news(), portfolio(equity=300, cash=300, sod=300), None)
    assert any("size < 1" in v for v in a.veto_reasons)


def test_all_vetoes_are_collected(cfg):
    agent, _ = _agent(cfg, now=et(9, 40))
    agent.pause()
    a, _ = agent.evaluate(setup(), context(now=et(9, 40), spread_pct=0.01, age_s=500), news(blackout=True),
                          portfolio(positions=[position("AAPL")]), None)
    assert len(a.veto_reasons) >= 5


def test_day_roll_resets_but_keeps_pause(cfg):
    agent, clk = _agent(cfg, state=RiskState(trading_date=D, realized_pnl_today=-500, consecutive_losses=2,
                                              halted_for_day=True, entries_paused=True))
    clk.set(et(11, 0, D + timedelta(days=1)))
    assert agent.roll_day()
    st = agent.state
    assert st.realized_pnl_today == 0 and st.consecutive_losses == 0 and not st.halted_for_day and st.entries_paused
