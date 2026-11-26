"""Replay performance metrics (pure functions)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from warroom.models.schemas import Position


def net_pnl(p: Position) -> float:
    return p.realized_pnl - p.commissions


def max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """(max drawdown in $, as a fraction of the running peak)."""
    peak = None
    dd_abs = dd_pct = 0.0
    for e in equity_curve:
        peak = e if peak is None or e > peak else peak
        drop = peak - e
        if drop > dd_abs:
            dd_abs = drop
        if peak > 0 and drop / peak > dd_pct:
            dd_pct = drop / peak
    return round(dd_abs, 2), round(dd_pct, 6)


def profit_factor(pnls: list[float]) -> float | None:
    """Gross profit / gross loss. None when there are no trades; inf with no losers."""
    if not pnls:
        return None
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if gains > 0 else None
    return round(gains / losses, 3)


@dataclass
class ReplayReport:
    day: date
    tickers: list[str]
    source: str
    starting_equity: float
    ending_equity: float
    trades: list[Position] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    debates: int = 0
    approved: int = 0
    watchlist: int = 0
    vetoed: int = 0
    veto_reasons: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if net_pnl(t) > 0)

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.total_trades if self.trades else None

    @property
    def total_pnl(self) -> float:
        return round(sum(net_pnl(t) for t in self.trades), 2)

    @property
    def profit_factor(self) -> float | None:
        return profit_factor([net_pnl(t) for t in self.trades])

    @property
    def drawdown(self) -> tuple[float, float]:
        return max_drawdown([e for _, e in self.equity_curve] or [self.starting_equity])

    def format(self) -> str:
        pf = self.profit_factor
        pf_s = "n/a" if pf is None else ("inf (no losing trades)" if pf == float("inf") else f"{pf:.2f}")
        wr = "n/a" if self.win_rate is None else f"{self.win_rate:.1%}"
        dd_abs, dd_pct = self.drawdown
        lines = [
            f"REPLAY {self.day}  tickers={','.join(self.tickers)}  data={self.source}",
            "-" * 64,
            f"Total trades     {self.total_trades}",
            f"Win rate         {wr}  ({self.wins}W / {self.total_trades - self.wins}L)",
            f"Profit factor    {pf_s}",
            f"Max drawdown     ${dd_abs:,.2f} ({dd_pct:.2%})",
            f"Total P&L (net)  ${self.total_pnl:+,.2f}",
            f"Equity           ${self.starting_equity:,.2f} -> ${self.ending_equity:,.2f}",
            f"War Room         {self.debates} debates: {self.approved} approved, {self.watchlist} watchlist, "
            f"{self.vetoed} vetoed",
        ]
        if self.veto_reasons:
            lines.append("Top veto reasons:")
            lines += [f"  {n:>3}x {r}" for r, n in self.veto_reasons.most_common(5)]
        if self.trades:
            lines.append("Trades:")
            for t in self.trades:
                lines.append(f"  {t.ticker:<5} {t.direction.value:<5} {t.initial_qty:>5} @ {t.entry_price:>9.2f} "
                             f"-> {t.exit_reason.value if t.exit_reason else '?':<14} net {net_pnl(t):+9.2f}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)
