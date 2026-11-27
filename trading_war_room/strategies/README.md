# Your strategies

Each `.yaml` file here is one strategy. You don't need to write code: describe **when to go long
and/or short**, and the system handles everything else. Stops, targets, position size, the War Room
debate, Risk's veto and all safety limits apply exactly as they do to the built-in strategies.

## The easy way: tell J.A.R.V.I.S.

| You type | What happens |
|---|---|
| `new strategy dip_buyer: long when rsi14 < 30 and close > ema50` | Turned into rules; you `confirm` to save and switch it on |
| `new strategy: buy oversold dips in an uptrend` | Plain English (needs an AI key in `.env`) |
| `new strategy from ~/Desktop/chart.png` | Reads a screenshot of your rules (needs an AI key with vision) |
| `new strategy from ~/my_rules.yaml` | Loads a rule file you wrote |
| `strategies` / `enable NAME` / `disable NAME` / `show strategy NAME` / `delete strategy NAME` | Manage them |
| `stats` | Win rate and net P&L for each strategy |
| `strategy fields` | Everything you can use in a rule |

Whatever the source, you always see the exact rules before anything is saved. The AI only *translates*
your description into this rule format. It can't add anything else, and its output is checked like any
other rule file.

## The file format

```yaml
name: dip_buyer                  # lowercase_with_underscores
description: Buy oversold dips in an uptrend
assets: [equity, crypto]         # or just one
confidence: 0.5                  # 0.1-0.9 starting confidence (the system adjusts it)
long_when:                       # ALL must be true
  - rsi14 < 30
  - close > ema50
  - close crosses_above vwap
short_when:                      # optional
  - rsi14 > 70
  - close < ema50
```

A condition is `A op B` or `field between low high`:

- `op` is one of `<  <=  >  >=  crosses_above  crosses_below`.
- `A` and `B` can be a field, a number, or `field * number`, e.g. `volume > avg_volume * 2`.
- `crosses_above` / `crosses_below` need `close`, `high`, `low` or `open` on the left.
- Plain words work too: "RSI under 30", "close above the 50 EMA", "crosses above".

**Fields:**
- **Last bar:** `price close open high low volume`, the same with `prev_` in front, `avg_volume bar_range close_position`
- **Indicators:** `vwap ema9 ema21 ema50 rsi14 atr14 rvol spread_pct`
- **Levels:** `orb_high orb_low premarket_high premarket_low` (stocks), `range_high range_low` (crypto),
  `prior_high prior_low prior_close swing_high swing_low`
- **Context:** `trend market_trend` (1 up / -1 down / 0), `minutes_since_open hour_utc`

If a field is missing (e.g. `orb_high` on crypto), the condition is false and the strategy stays quiet.

## Test before trusting

```bash
python main.py --replay 2025-01-15 --replay-source mock      # report includes "By strategy"
```

Then let it run in `--paper` and check `stats`. Switch off anything that doesn't pay after fees.
