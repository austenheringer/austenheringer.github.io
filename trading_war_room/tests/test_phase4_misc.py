"""Notifications, CSV replay data, and replay metrics."""

import json
from collections import Counter

import httpx
import pandas as pd
import pytest

from tests.conftest import D, et
from warroom.backtest.metrics import ReplayReport, max_drawdown, profit_factor
from warroom.config import HttpSection, Secrets, parse_config
from warroom.core.clock import SimulatedClock
from warroom.models.schemas import (
    AlertLevel, Fill, JarvisBriefing, Position, SystemAlert, Timeframe, TradeExecuted,
)
from warroom.notifications.alerts import (
    DiscordWebhookNotifier, NotificationHub, TelegramNotifier, build_notifications, format_event,
)
from warroom.providers.base import ProviderError
from warroom.providers.market_data import CsvDataProvider


# ---------------------------------------------------------------- notifications

def _pos():
    return Position(ticker="NVDA", direction="LONG", entry_price=100, stop_loss=99, tp1=102, tp2=103, initial_qty=10,
                    qty=10, opened_at=et(11))


def test_format_event_filters():
    on = {"APPROVED", "TRADE_EXECUTED", "CIRCUIT_BREAKER"}
    assert "APPROVED" in format_event(JarvisBriefing(timestamp=et(11), text="go", status="APPROVED"), on)
    assert format_event(JarvisBriefing(timestamp=et(11), text="no", status="VETOED"), on) is None
    f = Fill(position_id="p", ticker="NVDA", direction="LONG", is_entry=True, qty=10, price=100, commission=0, timestamp=et(11))
    assert "OPEN" in format_event(TradeExecuted(timestamp=et(11), fill=f, position=_pos()), on)
    assert format_event(SystemAlert(timestamp=et(11), level=AlertLevel.CRITICAL, source="risk", message="halt"), on)
    assert format_event(SystemAlert(timestamp=et(11), level=AlertLevel.INFO, source="ui", message="x"), on) is None


async def test_discord_and_telegram_payloads_and_redaction():
    sent = []

    def h(req):
        sent.append((str(req.url), json.loads(req.content)))
        return httpx.Response(204 if "discord" in str(req.url) else 200)

    d = DiscordWebhookNotifier("https://discord.com/api/webhooks/1/abc", HttpSection(), transport=httpx.MockTransport(h))
    t = TelegramNotifier("123:SECRET", "42", HttpSection(), transport=httpx.MockTransport(h))
    await d.send("hello @everyone")
    await t.send("hi")
    assert sent[0][1]["allowed_mentions"] == {"parse": []} and sent[1][1]["chat_id"] == "42"
    bad = TelegramNotifier("123:SECRET", "42", HttpSection(max_retries=0), transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(ProviderError) as ei:
        await bad.send("x")
    assert "SECRET" not in str(ei.value)
    for n in (d, t, bad):
        await n.aclose()
    with pytest.raises(ValueError):
        DiscordWebhookNotifier("http://insecure", HttpSection())


async def test_hub_dedupes_and_survives_failures():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def send(self, text):
            self.calls += 1
            raise ProviderError("down")

        async def aclose(self):
            pass

    class Ok(Flaky):
        name = "ok"

        async def send(self, text):
            self.calls += 1

    flaky, ok = Flaky(), Ok()
    hub = NotificationHub([flaky, ok], ["APPROVED"], min_interval_seconds=0)
    ev = JarvisBriefing(timestamp=et(11), text="NVDA approved", status="APPROVED")
    await hub.handle(ev)
    await hub.handle(ev)  # duplicate within 5 minutes
    assert ok.calls == 1 and flaky.calls == 1 and hub.failed == 1 and hub.sent == 1


def test_build_notifications_requires_secrets(cfg):
    raw = cfg.model_dump()
    raw["notifications"].update(discord_enabled=True, telegram_enabled=True)
    c = parse_config(raw)
    assert not build_notifications(c, Secrets.from_env({})).enabled
    hub = build_notifications(c, Secrets.from_env({"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/1/x",
                                                   "TELEGRAM_BOT_TOKEN": "1:x", "TELEGRAM_CHAT_ID": "9"}))
    assert [n.name for n in hub.notifiers] == ["discord", "telegram"]


# ---------------------------------------------------------------- CSV provider

def _write_csv(root, d, ticker, start_et, n=60, tz=True):
    idx = pd.date_range(et(*start_et, d=d), periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"timestamp": [t.isoformat() if tz else t.tz_localize(None).isoformat() for t in idx],
                       "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000})
    (root / d.isoformat()).mkdir(parents=True, exist_ok=True)
    df.to_csv(root / d.isoformat() / f"{ticker}.csv", index=False)


async def test_csv_provider(tmp_path):
    _write_csv(tmp_path, D, "AAPL", (9, 30))
    clk = SimulatedClock(et(10, 0))
    p = CsvDataProvider(clk, str(tmp_path), assumed_spread_pct=0.001)
    b1 = await p.get_bars("AAPL", Timeframe.M1, et(4, 0))
    assert len(b1) == 30 and b1.index[-1] == pd.Timestamp(et(9, 59))   # no look-ahead
    b5 = await p.get_bars("AAPL", Timeframe.M5, et(4, 0))
    assert len(b5) == 6 and b5.volume.iloc[0] == 5000
    q = await p.get_quote("AAPL")
    assert q.spread_pct == pytest.approx(0.001) and p.available("AAPL", D) and not p.available("MSFT", D)


async def test_csv_rejects_naive_timestamps(tmp_path):
    _write_csv(tmp_path, D, "AAPL", (9, 30), tz=False)
    p = CsvDataProvider(SimulatedClock(et(10, 0)), str(tmp_path))
    with pytest.raises(ProviderError, match="timezone"):
        await p.get_bars("AAPL", Timeframe.M1, et(4, 0))


# ---------------------------------------------------------------- metrics

def test_drawdown_and_profit_factor():
    assert max_drawdown([100, 110, 90, 120, 100]) == (20.0, pytest.approx(20 / 110, rel=1e-4))
    assert max_drawdown([100, 101, 102]) == (0.0, 0.0)
    assert profit_factor([100, -50, 50]) == 3.0
    assert profit_factor([10]) == float("inf") and profit_factor([]) is None and profit_factor([-5]) == 0.0


def test_report_format():
    def closed(pnl, comm=1.0):
        return Position(ticker="AAPL", direction="LONG", entry_price=100, stop_loss=99, tp1=102, tp2=103, initial_qty=10,
                        qty=0, opened_at=et(10), closed_at=et(11), realized_pnl=pnl, commissions=comm, status="CLOSED",
                        exit_reason="TP2" if pnl > 0 else "STOP_LOSS")

    r = ReplayReport(day=D, tickers=["AAPL"], source="csv", starting_equity=100_000, ending_equity=100_048,
                     trades=[closed(100), closed(-50)], equity_curve=[(et(10), 100_000), (et(10, 30), 100_099),
                                                                     (et(11), 100_048)],
                     debates=5, approved=2, vetoed=3, veto_reasons=Counter({"R:R": 3}))
    assert r.total_trades == 2 and r.wins == 1 and r.win_rate == 0.5 and r.total_pnl == 48.0
    assert r.profit_factor == pytest.approx(99 / 51, abs=1e-3)
    text = r.format()
    for s in ("Total trades     2", "Win rate         50.0%", "Max drawdown", "Total P&L (net)  $+48.00", "3x R:R"):
        assert s in text
