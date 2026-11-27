"""Strategy plugins, the rule language, and changing everything by talking to J.A.R.V.I.S."""

from datetime import timedelta

import pandas as pd
import pytest
import yaml

from tests.conftest import et
from tests.helpers import context, snapshot
from warroom.agents.jarvis_interface import CommandKind, parse_command
from warroom.agents.technical_agent import MacroBias
from warroom.config import Secrets, load_config, parse_config
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import Direction, MarketRegime, Mode, SetupType
from warroom.providers.base import LLMProvider
from warroom.settings_manager import SettingError, SettingsManager, parse_value, resolve
from warroom.strategies.base import StrategyContext
from warroom.strategies.builtin import BUILTINS
from warroom.strategies.registry import build_registry
from warroom.strategies.rules import (
    RuleError, RuleStrategy, load_rules_dir, parse_condition, parse_plain_rules, validate_spec,
)

NEUTRAL = MacroBias(MarketRegime.MIXED, None, {})


# ================================================================== rule language

def _sc(snap=None, bars=None):
    snap = snap or snapshot(price=101.0, vwap=100.5, ema9=100.8, ema21=100.4, ema50=99.8, rsi14=28.0, atr14=1.0, rvol=1.8)
    ctx = context(snap)
    bars = bars if bars is not None else pd.DataFrame(
        {"open": [100.0, 100.2], "high": [100.6, 101.3], "low": [99.8, 100.1], "close": [100.3, 101.0],
         "volume": [1000.0, 3000.0]}, index=pd.date_range(et(10, 50), periods=2, freq="5min", tz="UTC"))
    ctx.bars_5m = bars
    return StrategyContext(ctx=ctx, cfg=parse_config({}), bias=NEUTRAL, last=bars.iloc[-1], prev=bars.iloc[-2], bars=bars)


@pytest.mark.parametrize("text,ok", [
    ("rsi14 < 30", True), ("close > ema50", True), ("rvol >= 1.5", True), ("close crosses_above vwap", True),
    ("rsi14 between 20 35", True), ("volume > avg_volume * 2", True), ("close > ema50 * 1.01", True),
    ("RSI under 30", True), ("close crosses above the 50 EMA", False),  # 50 ema != vwap: checks value, not just parse
])
def test_condition_parsing_and_evaluation(text, ok):
    cond = parse_condition(text)
    result, why = cond.check(_sc())
    assert result is ok and (why or not ok)


@pytest.mark.parametrize("bad", [
    "__import__('os').system('rm -rf /') > 1", "macd > 0", "rsi14 < ", "1 < 2", "ema50 crosses_above vwap",
    "open(file) > 3", "rsi14 < 30; import os",
])
def test_rule_language_rejects_unknown_or_unsafe(bad):
    with pytest.raises(RuleError):
        parse_condition(bad)


def test_missing_values_make_conditions_false():
    snap = snapshot(orb_high=None)
    assert parse_condition("close > orb_high").check(_sc(snap))[0] is False


def test_validate_spec_and_plain_rules():
    spec = validate_spec(parse_plain_rules("dip_buyer", "long when RSI below 30 and close above ema50; "
                                                        "short when rsi14 > 70 and close < ema50"))
    assert spec["long_when"] == ["rsi14 < 30", "close > ema50"] and spec["short_when"][0] == "rsi14 > 70"
    for bad, needle in [({"name": "x"}, "name"), ({"name": "orb_breakout", "long_when": ["rsi14 < 30"]}, "built-in"),
                        ({"name": "abc", "long_when": []}, "at least one"),
                        ({"name": "abc", "long_when": ["rsi14 < 30"], "exec": "x"}, "unknown key"),
                        ({"name": "abc", "long_when": ["rsi14 < 30"], "confidence": 5}, "confidence")]:
        with pytest.raises(RuleError, match=needle):
            validate_spec(bad, reserved=set(BUILTINS))


def test_rule_strategy_signals_with_explanations():
    s = RuleStrategy(validate_spec({"name": "oversold_reclaim", "long_when": ["rsi14 < 35", "close crosses_above vwap"],
                                    "assets": ["equity"]}))
    sigs = s.evaluate(_sc())
    assert len(sigs) == 1 and sigs[0].direction is Direction.LONG and sigs[0].setup_type is SetupType.CUSTOM
    assert "rsi14 28 < 35" in sigs[0].rationale[0] and sigs[0].strategy == "oversold_reclaim"


def test_rules_dir_loading_reports_bad_files(tmp_path):
    (tmp_path / "good.yaml").write_text(yaml.safe_dump({"name": "good_one", "long_when": ["rsi14 < 30"]}))
    (tmp_path / "bad.yaml").write_text("name: bad_one\nlong_when: [macd > 0]\n")
    (tmp_path / "broken.yaml").write_text("{not yaml")
    rules, errors = load_rules_dir(tmp_path)
    assert list(rules) == ["good_one"] and len(errors) == 2


# ================================================================== registry & builtins

def test_registry_enable_and_params(cfg, tmp_path):
    raw = cfg.model_dump(mode="json")
    raw["strategies"].update(enabled=["vwap_bounce", "nope"], rules_dir=str(tmp_path),
                             params={"rvol_momentum": {"rvol_multiple": 2.0}})
    reg = build_registry(parse_config(raw))
    assert [s.name for s in reg.active()] == ["vwap_bounce"] and any("nope" in e for e in reg.errors)
    assert reg.available["rvol_momentum"].params["rvol_multiple"] == 2.0
    raw["strategies"]["params"] = {"vwap_bounce": {"bogus": 1}}
    assert any("bogus" in e for e in build_registry(parse_config(raw)).errors)


def test_disabled_strategy_never_fires(cfg, cal):
    from tests.test_agents import _bars, _tech

    tech, _ = _tech(cfg, cal)
    snap = snapshot(price=101.2, vwap=100.5, ema9=100.9, ema21=100.6, ema50=100.2, rsi14=62, atr14=0.8,
                    orb_high=101.0, orb_low=99.5, rvol=1.8)
    ctx = context(snap)
    ctx.bars_5m = _bars([100.2, 100.6, 100.9, 101.2])
    assert any(s.strategy == "orb_breakout" for s in tech.detect_setups(ctx, NEUTRAL))
    tech.registry.enabled.remove("orb_breakout")
    assert not any(s.strategy == "orb_breakout" for s in tech.detect_setups(ctx, NEUTRAL))


# ================================================================== settings manager

def test_settings_resolve_parse_and_validate(cfg, tmp_path):
    assert resolve("risk per trade") == "risk.max_risk_pct" and resolve("max_open_positions") == "risk.max_open_positions"
    assert parse_value("risk.max_risk_pct", "0.5%", 0.01) == 0.005 and parse_value("risk.max_risk_pct", "1", 0.01) == 0.01
    assert parse_value("risk.enforce_pdt_rule", "on", False) is True
    assert parse_value("app.tickers", "aapl, msft amzn", ["X"]) == ["AAPL", "MSFT", "AMZN"]
    m = SettingsManager(tmp_path / "o.yaml")
    ch = m.plan_text(cfg, "risk per trade", "0.5%")
    assert ch.old == 0.01 and ch.new == 0.005 and not ch.restart and ch.config.risk.max_risk_pct == 0.005
    with pytest.raises(SettingError, match="isn't allowed"):
        m.plan_text(cfg, "risk per trade", "10%")
    with pytest.raises(SettingError, match="can't be changed from chat"):
        m.plan_text(cfg, "live_trading_enabled", "on")
    with pytest.raises(SettingError, match="don't know"):
        m.plan_text(cfg, "warp drive", "9")
    assert m.plan_text(cfg, "slippage", "0.1%").path == "execution.slippage_pct"
    assert m.plan_text(cfg, "llm calls", "50").restart


def test_overrides_persist_and_merge(cfg, tmp_path):
    m = SettingsManager(tmp_path / "o.yaml")
    m.persist(m.plan_text(cfg, "risk per trade", "0.5%"))
    m.persist(m.plan_text(cfg, "max positions", "2"))
    merged = load_config("config.yaml", overrides=tmp_path / "o.yaml")
    assert merged.risk.max_risk_pct == 0.005 and merged.risk.max_open_positions == 2
    assert load_config("config.yaml").risk.max_risk_pct == 0.01      # base file untouched
    assert "max_risk_pct = 0.5% *" in m.describe(merged, "risk")
    assert m.reset() and not (tmp_path / "o.yaml").exists()


# ================================================================== command parsing

@pytest.mark.parametrize("text,kind,args", [
    ("strategies", CommandKind.STRATEGIES, ()),
    ("which strategies are on?", CommandKind.STRATEGIES, ()),
    ("turn off vwap bounce", CommandKind.DISABLE, ("vwap bounce",)),
    ("enable my_dip_strategy", CommandKind.ENABLE, ("my_dip_strategy",)),
    ("disable the orb_breakout strategy", CommandKind.DISABLE, ("orb_breakout",)),
    ("show strategy ema_pullback", CommandKind.SHOW_STRATEGY, ("ema_pullback",)),
    ("set risk per trade to 0.5%", CommandKind.SET, ("risk per trade", "0.5%")),
    ("change max positions to 5", CommandKind.SET, ("max positions", "5")),
    ("set strategy rvol_momentum rvol_multiple to 1.5", CommandKind.STRATEGY_PARAM, ("rvol_momentum", "rvol_multiple", "1.5")),
    ("add stock AMZN", CommandKind.ADD_SYMBOL, ("stock", "amzn")),
    ("add coin doge", CommandKind.ADD_SYMBOL, ("coin", "doge")),
    ("stop watching stock tsla", CommandKind.REMOVE_SYMBOL, ("stock", "tsla")),
    ("settings risk", CommandKind.SETTINGS, ("risk",)),
    ("stats", CommandKind.STATS, ()),
    ("which strategies work?", CommandKind.STATS, ()),
    ("confirm", CommandKind.CONFIRM, ()), ("yes", CommandKind.CONFIRM, ()), ("cancel", CommandKind.CANCEL, ()),
    ("reset settings", CommandKind.RESET_SETTINGS, ()),
    ("strategy fields", CommandKind.FIELDS, ()),
    ("delete strategy dip_buyer", CommandKind.DELETE_STRATEGY, ("dip_buyer",)),
])
def test_control_command_parsing(text, kind, args):
    c = parse_command(text)
    assert c.kind is kind and c.args == args


def test_new_strategy_keeps_original_case_and_paths():
    c = parse_command("new strategy from ~/Desktop/My Chart.PNG")
    assert c.kind is CommandKind.NEW_STRATEGY and c.args == ("from ~/Desktop/My Chart.PNG",)
    assert parse_command("analyze TSLA").kind is CommandKind.ANALYZE and parse_command("status").kind is CommandKind.STATUS


# ================================================================== end-to-end through J.A.R.V.I.S.

class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.images = 0

    async def complete(self, system, prompt, max_tokens=300):
        return self.replies.pop(0)

    async def complete_with_image(self, system, prompt, image, media_type, max_tokens=800):
        self.images += 1
        assert image.startswith(b"\x89PNG") and media_type == "image/png"
        return self.replies.pop(0)


async def _orch(cfg, tmp_path, llm=None):
    from warroom.orchestrator import Orchestrator

    raw = cfg.model_dump(mode="json")
    raw["strategies"]["rules_dir"] = str(tmp_path / "strategies")
    c = parse_config(raw)
    o = await Orchestrator.build(c, Secrets.from_env({}), Mode.SMOKE, clock=SimulatedClock(et(11, 0)),
                                 db_path=":memory:", overrides_path=str(tmp_path / "overrides.yaml"))
    if llm is not None:
        o.dialogue.primary = llm
    await o.start()
    return o


async def test_change_settings_by_talking(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    r = await o.handle_command("set risk per trade to 0.5%")
    assert "proposed" in r and "1% -> 0.5%" in r and o.risk.r.max_risk_pct == 0.01   # not applied yet
    r = await o.handle_command("confirm")
    assert "Applied" in r and o.risk.r.max_risk_pct == 0.005 and o.cfg.risk.max_risk_pct == 0.005
    assert yaml.safe_load((tmp_path / "overrides.yaml").read_text())["risk"]["max_risk_pct"] == 0.005
    assert await o.db.count("settings_changes") == 1
    assert "isn't allowed" in await o.handle_command("set risk per trade to 9%")
    assert "can't be changed" in await o.handle_command("set live_trading_enabled to on")
    await o.handle_command("set max positions to 1")
    assert "Discarded" in await o.handle_command("cancel") and o.risk.r.max_open_positions == 3
    assert "nothing waiting" in (await o.handle_command("confirm")).lower()
    r = await o.handle_command("set title to Boss")
    await o.handle_command("confirm")
    assert (await o.handle_command("status")).startswith("Boss,")
    assert "max_risk_pct = 0.5%" in await o.handle_command("settings risk")
    await o.shutdown()


async def test_proposals_expire(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    await o.handle_command("set cooldown to 30")
    o.clock.advance(timedelta(minutes=11))
    assert "expired" in await o.handle_command("confirm") and o.cfg.technical.setup_cooldown_minutes == 15
    await o.shutdown()


async def test_strategy_toggle_symbols_and_restart_persistence(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    r = await o.handle_command("turn off vwap bounce")
    assert "vwap_bounce is now off" in r and "vwap_bounce" not in o.technical.registry.enabled
    assert "[off] vwap_bounce" in await o.handle_command("strategies")
    assert "Added AMZN" in await o.handle_command("add stock AMZN") and "AMZN" in o.cfg.app.tickers
    assert "Added DOGE-USD" in await o.handle_command("add coin doge") and "DOGE-USD" in o.router.crypto_symbols
    assert "Removed TSLA" in await o.handle_command("remove stock tsla") and "TSLA" not in o.watch_symbols
    await o.shutdown()
    merged = load_config("config.yaml", overrides=tmp_path / "overrides.yaml")
    assert "vwap_bounce" not in merged.strategies.enabled and "AMZN" in merged.app.tickers  # survives restart


async def test_new_strategy_from_typed_rules(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    r = await o.handle_command("new strategy dip_buyer: long when rsi14 < 35 and close > ema50")
    assert "proposed" in r and "LONG when rsi14 < 35 AND close > ema50" in r
    assert not (tmp_path / "strategies" / "dip_buyer.yaml").exists()                 # nothing saved yet
    r = await o.handle_command("confirm")
    assert "switched 'dip_buyer' on" in r and (tmp_path / "strategies" / "dip_buyer.yaml").exists()
    assert "dip_buyer" in o.technical.registry.enabled
    assert "dip_buyer (on)" in await o.handle_command("show strategy dip_buyer")
    assert "Deleted" in await o.handle_command("delete strategy dip_buyer")
    assert "dip_buyer" not in o.technical.registry.available
    assert "built in" in await o.handle_command("delete strategy orb_breakout")
    await o.shutdown()


async def test_new_strategy_needs_llm_for_plain_english_and_images(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    assert "need an AI model" in await o.handle_command("new strategy: buy strong stocks that dip in the morning")
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    assert "vision" in await o.handle_command(f"new strategy from {img}")
    assert "can't find" in await o.handle_command("new strategy from /nope/chart.png")
    await o.shutdown()


async def test_new_strategy_from_screenshot_and_text_with_llm(cfg, tmp_path):
    good = "```yaml\nname: morning_dip\ndescription: dip buy\nassets: [equity]\nlong_when:\n  - rsi14 < 35\n  - close > ema50\n```"
    evil = "name: evil\nlong_when: [\"__import__('os').system('x') > 1\"]\nexec: rm -rf /"
    llm = FakeLLM([good, evil, "name: gap_fader\nshort_when: [close > prior_high, rsi14 > 70]\n"])
    o = await _orch(cfg, tmp_path, llm=llm)
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    r = await o.handle_command(f"new strategy from {img}")
    assert llm.images == 1 and "morning_dip" in r and "proposed" in r
    await o.handle_command("confirm")
    assert "morning_dip" in o.technical.registry.enabled
    r = await o.handle_command(f"new strategy from {img}")                 # model returns something malicious
    assert "failed validation" in r and "evil" not in o.technical.registry.available
    r = await o.handle_command("new strategy: fade gaps above yesterday's high when overbought")
    assert "gap_fader" in r and "SHORT when close > prior_high AND rsi14 > 70" in r
    await o.shutdown()


async def test_free_text_goes_through_the_command_parser(cfg, tmp_path):
    llm = FakeLLM(["disable rvol_momentum", "rm -rf /", "UNKNOWN"])
    o = await _orch(cfg, tmp_path, llm=llm)
    r = await o.handle_command("I don't trust the volume spike thing, stop using it")
    assert r.startswith("(Taking that as: disable rvol_momentum)") and "rvol_momentum" not in o.technical.registry.enabled
    assert "didn't catch" in await o.handle_command("please wipe the disk")       # nonsense translation ignored
    assert "didn't catch" in await o.handle_command("tell me a joke")
    await o.shutdown()


async def test_strategy_param_change_needs_confirm(cfg, tmp_path):
    o = await _orch(cfg, tmp_path)
    r = await o.handle_command("set strategy rvol_momentum rvol_multiple to 1.6")
    assert "1.3 -> 1.6" in r
    await o.handle_command("confirm")
    assert o.technical.registry.available["rvol_momentum"].params["rvol_multiple"] == 1.6
    assert "no parameter" in await o.handle_command("set strategy rvol_momentum banana to 2")
    await o.shutdown()


async def test_stats_by_strategy(cfg, tmp_path):
    from warroom.models.schemas import Position

    o = await _orch(cfg, tmp_path)
    assert "No closed trades" in await o.handle_command("stats")
    for i, (strat, pnl) in enumerate([("orb_breakout", 300.0), ("orb_breakout", -100.0), ("dip_buyer", -50.0)]):
        p = Position(ticker="AAPL", direction="LONG", entry_price=100, stop_loss=99, tp1=102, tp2=103, initial_qty=10,
                     qty=0, opened_at=et(10, i), closed_at=et(10, 30 + i), realized_pnl=pnl, status="CLOSED",
                     exit_reason="TP2", strategy=strat)
        await o.db.upsert_position(p)
    r = await o.handle_command("which strategies work?")
    lines = r.splitlines()
    assert "orb_breakout" in lines[2] and "+200.00" in lines[2] and "dip_buyer" in lines[3]
    await o.shutdown()


def test_replay_report_breaks_down_by_strategy():
    from warroom.backtest.metrics import ReplayReport
    from warroom.models.schemas import Position
    from tests.conftest import D

    t = [Position(ticker="AAPL", direction="LONG", entry_price=100, stop_loss=99, tp1=102, tp2=103, initial_qty=10,
                  qty=0, opened_at=et(10), closed_at=et(11), realized_pnl=50, status="CLOSED", exit_reason="TP2",
                  strategy="vwap_bounce")]
    text = ReplayReport(day=D, tickers=["AAPL"], source="x", starting_equity=1e5, ending_equity=1e5 + 50, trades=t).format()
    assert "By strategy" in text and "vwap_bounce" in text
