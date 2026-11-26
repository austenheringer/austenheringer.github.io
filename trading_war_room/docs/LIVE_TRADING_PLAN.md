# Plan: a small, capped real-money crypto mode (NOT built yet)

The system is paper-only. `execution.live_trading_enabled: true` is rejected by config validation.
This document is the contract for when, and how, a real-money mode may be added.

## Gate 1: evidence first (all must hold, measured in `--paper` on live Kraken data)

| Requirement | Threshold |
|---|---|
| Paper trading duration | at least 8 consecutive weeks |
| Closed crypto paper trades | at least 100 |
| Net expectancy per trade, after fees and slippage | > 0, and the lower bound of a 95% confidence interval > 0 |
| Profit factor (net) | at least 1.3 |
| Max drawdown | below 10% of paper equity |
| Circuit-breaker behaviour | every daily-loss halt / cooldown verified in the logs |
| Fill realism | paper entries within 0.1% of Kraken's book at signal time on a spot-checked sample |

If any line fails, there is no real-money mode. A strategy that loses on paper loses faster with money.

## Gate 2: safety architecture (must exist before the first order)

1. **A separate Kraken API key with trade-only permission.** No withdrawal permission, IP-allowlisted,
   stored only in `.env`. The app refuses keys that can withdraw (checked at startup).
2. **Exchange-side protection.** The stop-loss is placed as a resting order on Kraken immediately after
   each fill, so it survives an app crash, a network loss or a laptop lid closing.
3. **Hard caps, enforced in code and not overridable from the command bar:**
   - total capital allocated to live trading (start at $500);
   - maximum notional per order;
   - daily loss in dollars;
   - one open live position at a time to begin with.
4. **Startup reconciliation.** Exchange balances, open orders and fills are compared with the local
   ledger. Any mismatch halts trading until a human resolves it.
5. **Idempotent orders.** Client order IDs, so retries can never double-fill.
6. **Kill switch.** `flatten` cancels all orders and closes all positions. A dead-man timer halts new
   entries if data goes stale or the connection drops.
7. **Human in the loop first.** Phase A: J.A.R.V.I.S. proposes and you confirm each order. Phase B
   (only after 4+ weeks of Phase A without incident) allows automatic entries within the caps.
8. **Tax records.** An export of every fill with fees, in the format your accountant needs.

## Where it plugs in

`warroom/execution/base.py` defines the `Broker` interface. `PaperBroker` implements it, and a
`KrakenLiveBroker` would implement the same interface. The orchestrator, War Room and Risk agent do
not change. Risk's veto, the circuit breakers and restart recovery apply unchanged.
