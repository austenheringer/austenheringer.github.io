"""The built-in strategies (the original Technical-agent setups, unchanged in logic)."""

from __future__ import annotations

from warroom.core.assets import AssetClass
from warroom.models.schemas import Direction, SetupType
from warroom.strategies.base import Signal, Strategy, StrategyContext

L, S = Direction.LONG, Direction.SHORT


class OpeningRangeBreakout(Strategy):
    name = "orb_breakout"
    description = "Fresh break of the first-15-minute range, with VWAP on side and RSI not exhausted"
    asset_classes = frozenset({AssetClass.EQUITY})
    defaults = {"max_rsi_long": 80, "min_rsi_short": 20, "base_confidence": 0.5}

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        s, last, prev, p = sc.s, sc.last, sc.prev, self.params
        out = []
        if s.orb_high is None or s.orb_low is None:
            return out
        if last["close"] > s.orb_high >= prev["close"] and s.price > s.vwap and s.rsi14 <= p["max_rsi_long"]:
            out.append(self.sig(L, SetupType.ORB_BREAKOUT, p["base_confidence"],
                                [f"5m close {last['close']:.2f} broke ORB high {s.orb_high:.2f}", "holding above VWAP"]))
        if last["close"] < s.orb_low <= prev["close"] and s.price < s.vwap and s.rsi14 >= p["min_rsi_short"]:
            out.append(self.sig(S, SetupType.ORB_BREAKDOWN, p["base_confidence"],
                                [f"5m close {last['close']:.2f} broke ORB low {s.orb_low:.2f}", "holding below VWAP"]))
        return out


class RangeBreakout(Strategy):
    name = "range_breakout"
    description = "Break of the rolling multi-hour range (crypto's opening-range substitute)"
    asset_classes = frozenset({AssetClass.CRYPTO})
    defaults = {"max_rsi_long": 80, "min_rsi_short": 20, "base_confidence": 0.5}

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        s, last, prev, p = sc.s, sc.last, sc.prev, self.params
        out = []
        if s.range_high is None or s.range_low is None:
            return out
        hrs = sc.cfg.crypto.range_lookback_minutes // 60
        if last["close"] > s.range_high >= prev["close"] and s.price > s.vwap and s.rsi14 <= p["max_rsi_long"]:
            out.append(self.sig(L, SetupType.RANGE_BREAKOUT, p["base_confidence"],
                                [f"close {last['close']:.2f} broke the {hrs}h range high {s.range_high:.2f}",
                                 "above the UTC-day VWAP"]))
        if last["close"] < s.range_low <= prev["close"] and s.price < s.vwap and s.rsi14 >= p["min_rsi_short"]:
            out.append(self.sig(S, SetupType.RANGE_BREAKDOWN, p["base_confidence"],
                                [f"close {last['close']:.2f} broke the {hrs}h range low {s.range_low:.2f}",
                                 "below the UTC-day VWAP"]))
        return out


class VwapBounce(Strategy):
    name = "vwap_bounce"
    description = "Test of VWAP that closes back on the trend side (bounce long / rejection short)"
    defaults = {"base_confidence": 0.45}

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        s, last = sc.s, sc.last
        touch = sc.cfg.technical.vwap_touch_pct
        b = self.params["base_confidence"]
        out = []
        if last["low"] <= s.vwap * (1 + touch) and last["close"] > s.vwap and last["close"] > last["open"] and s.ema9 > s.ema21:
            out.append(self.sig(L, SetupType.VWAP_BOUNCE, b, [f"tested VWAP {s.vwap:.2f} and closed back above",
                                                              "EMA9 above EMA21"]))
        if last["high"] >= s.vwap * (1 - touch) and last["close"] < s.vwap and last["close"] < last["open"] and s.ema9 < s.ema21:
            out.append(self.sig(S, SetupType.VWAP_REJECTION, b, [f"rejected at VWAP {s.vwap:.2f}", "EMA9 below EMA21"]))
        return out


class EmaPullback(Strategy):
    name = "ema_pullback"
    description = "Pullback to EMA9 inside a 9/21/50 stacked trend, RSI reset"
    defaults = {"rsi_long_min": 40, "rsi_long_max": 65, "rsi_short_min": 35, "rsi_short_max": 60, "base_confidence": 0.45}

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        s, last, p = sc.s, sc.last, self.params
        out = []
        if s.ema50 is None:
            return out
        if (s.ema9 > s.ema21 > s.ema50 and last["low"] <= s.ema9 and last["close"] > s.ema9
                and p["rsi_long_min"] <= s.rsi14 <= p["rsi_long_max"] and s.price > s.vwap):
            out.append(self.sig(L, SetupType.EMA_PULLBACK, p["base_confidence"],
                                [f"pullback to EMA9 {s.ema9:.2f} in a 9>21>50 uptrend", f"RSI {s.rsi14:.0f} reset, not overbought"]))
        if (s.ema9 < s.ema21 < s.ema50 and last["high"] >= s.ema9 and last["close"] < s.ema9
                and p["rsi_short_min"] <= s.rsi14 <= p["rsi_short_max"] and s.price < s.vwap):
            out.append(self.sig(S, SetupType.EMA_PULLBACK, p["base_confidence"],
                                [f"rally into EMA9 {s.ema9:.2f} in a 9<21<50 downtrend", f"RSI {s.rsi14:.0f} reset, not oversold"]))
        return out


class RvolMomentum(Strategy):
    name = "rvol_momentum"
    description = "Unusually high relative volume with a strong bar closing near its extreme"
    defaults = {"rvol_multiple": 1.3, "close_position": 0.7, "base_confidence": 0.4}

    def evaluate(self, sc: StrategyContext) -> list[Signal]:
        s, last, p = sc.s, sc.last, self.params
        out = []
        if s.rvol is None or s.rvol < sc.cfg.technical.rvol_threshold * p["rvol_multiple"]:
            return out
        cp = sc.close_pos
        if s.price > s.vwap and s.ema9 > s.ema21 and 55 <= s.rsi14 <= 75 and last["close"] > last["open"] and cp >= p["close_position"]:
            out.append(self.sig(L, SetupType.RVOL_MOMENTUM, p["base_confidence"],
                                [f"RVOL {s.rvol:.1f}x momentum bar closing near highs", f"RSI {s.rsi14:.0f}"]))
        if (s.price < s.vwap and s.ema9 < s.ema21 and 25 <= s.rsi14 <= 45 and last["close"] < last["open"]
                and cp <= 1 - p["close_position"]):
            out.append(self.sig(S, SetupType.RVOL_MOMENTUM, p["base_confidence"],
                                [f"RVOL {s.rvol:.1f}x momentum bar closing near lows", f"RSI {s.rsi14:.0f}"]))
        return out


BUILTINS: dict[str, type[Strategy]] = {cls.name: cls for cls in
                                       (OpeningRangeBreakout, RangeBreakout, VwapBounce, EmaPullback, RvolMomentum)}
