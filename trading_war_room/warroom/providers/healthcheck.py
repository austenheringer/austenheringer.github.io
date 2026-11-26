"""Live contract checks for configured providers (``python main.py --check-providers``).

Each check calls the real service with your keys and verifies the response
has the shape the rest of the system relies on: tz-aware UTC bars in order,
positive prices, completed bars only, sane bar age during market hours, a
non-crossed quote, parseable news, a non-empty FOMC/BLS schedule and a
parseable halt feed. Nothing is traded, nothing is posted to notifiers.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import timedelta

from warroom.config import AppConfig, Secrets
from warroom.core.clock import Clock
from warroom.core.market_calendar import MarketCalendar
from warroom.models.schemas import Mode, Timeframe
from warroom.providers import factory
from warroom.providers.calendar_feeds import NasdaqHaltProvider


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name:<28} {self.detail} ({self.seconds:.1f}s)"


async def _timed(name: str, coro) -> CheckResult:
    t0 = _time.perf_counter()
    try:
        ok, detail = await coro
    except Exception as exc:  # noqa: BLE001 - report every failure, keep checking
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return CheckResult(name, ok, detail, _time.perf_counter() - t0)


async def check_providers(cfg: AppConfig, secrets: Secrets, clock: Clock, calendar: MarketCalendar) -> list[CheckResult]:
    results: list[CheckResult] = []
    ticker = cfg.app.tickers[0]
    now = clock.now()
    market_open = calendar.is_open(now)

    md = factory.build_market_data(cfg, secrets, clock, calendar, Mode.PAPER)

    async def bars_check():
        df = await md.get_bars(ticker, Timeframe.M5, now - timedelta(days=4))
        if df.empty:
            return False, "no 5m bars returned for the last 4 days"
        problems = []
        if str(df.index.tz) != "UTC":
            problems.append(f"index tz {df.index.tz}")
        if not df.index.is_monotonic_increasing:
            problems.append("bars out of order")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            problems.append("non-positive prices")
        if df.index[-1].to_pydatetime() + timedelta(minutes=5) > now:
            problems.append("incomplete (still-forming) bar returned")
        age = (now - df.index[-1].to_pydatetime()).total_seconds()
        if market_open and age > cfg.risk.max_data_age_seconds + 300:
            problems.append(f"latest bar is {age:.0f}s old during market hours (stale-data veto will fire)")
        return (not problems), (", ".join(problems) or f"{len(df)} bars, latest {df.index[-1]} (age {age:.0f}s)")

    async def quote_check():
        q = await md.get_quote(ticker)
        if q is None:
            return (not md.has_real_quotes), (
                "no real bid/ask from this provider; spread checks use risk.unknown_spread_policy="
                f"{cfg.risk.unknown_spread_policy}")
        return q.ask >= q.bid > 0, f"bid {q.bid} ask {q.ask} spread {q.spread_pct:.4%}"

    results.append(await _timed(f"market data ({md.name}) bars", bars_check()))
    results.append(await _timed(f"market data ({md.name}) quote", quote_check()))

    crypto = getattr(md, "crypto", None)
    if crypto is not None and cfg.crypto.symbols:
        sym = cfg.crypto.symbols[0]

        async def crypto_bars():
            df = await md.get_bars(sym, Timeframe.M15, now - timedelta(days=2))
            if df.empty:
                return False, f"no 15m bars for {sym}"
            age = (now - df.index[-1].to_pydatetime()).total_seconds()
            ok = age <= 15 * 60 + cfg.risk.max_data_age_seconds
            return ok, f"{len(df)} bars, latest {df.index[-1]} (age {age:.0f}s)"

        async def crypto_quote():
            q = await md.get_quote(sym)
            if q is None:
                return False, "no quote"
            ok = q.spread_pct <= cfg.crypto.max_spread_pct
            return ok, f"bid {q.bid} ask {q.ask} spread {q.spread_pct:.4%} (max {cfg.crypto.max_spread_pct:.2%})"

        async def crypto_status():
            st = await crypto.system_status()
            return st == "online", f"exchange status: {st}"

        async def crypto_specs():
            specs = await crypto.instruments(cfg.crypto.symbols, cfg.crypto.taker_fee_pct, cfg.crypto.maker_fee_pct,
                                             cfg.crypto.allow_short)
            missing = set(cfg.crypto.symbols) - {i.symbol for i in specs}
            return not missing, (f"lot sizes for {len(specs)} pair(s)" + (f"; unknown: {sorted(missing)}" if missing else ""))

        results.append(await _timed(f"crypto ({crypto.name}) bars", crypto_bars()))
        results.append(await _timed(f"crypto ({crypto.name}) quote", crypto_quote()))
        results.append(await _timed(f"crypto ({crypto.name}) status", crypto_status()))
        results.append(await _timed(f"crypto ({crypto.name}) pairs", crypto_specs()))
    await md.aclose()

    macro, notes = await factory.build_macro_calendar(cfg, calendar, clock, Mode.PAPER)

    async def macro_check():
        warns = macro.coverage_warnings(now)
        return (not warns), "; ".join(notes + warns) or "ok"

    results.append(await _timed("macro calendar (Fed/BLS)", macro_check()))

    news = factory.build_news(cfg, secrets, clock, calendar, Mode.PAPER, macro=macro)

    async def news_check():
        items = await news.get_news(ticker, now - timedelta(days=3))
        bad = [i for i in items if i.published_at > now]
        if bad:
            return False, f"{len(bad)} items dated in the future"
        return True, f"{len(items)} headlines for {ticker} in 3 days" + ("" if items else " (none: check the feed/ticker)")

    results.append(await _timed(f"news ({news.name})", news_check()))
    await news.aclose()

    if cfg.news.halt_feed_enabled:
        halts = NasdaqHaltProvider(clock, cfg.http)

        async def halt_check():
            hs = await halts.halts()
            if halts.last_error:
                return False, halts.last_error
            return True, f"feed parsed, {len(hs)} halt(s) listed"

        results.append(await _timed("halt feed (Nasdaq)", halt_check()))
        await halts.aclose()

    llm = factory.build_llm(cfg, secrets, Mode.PAPER)
    if llm is not None:
        async def llm_check():
            out = await llm.complete("Reply with the single word: ready", "Say ready.", max_tokens=5)
            return bool(out), f"{llm.name} replied {out[:30]!r}"

        results.append(await _timed(f"llm ({llm.name})", llm_check()))
        await llm.aclose()
    return results
