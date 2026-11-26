"""Shared async HTTP client with exponential backoff + full jitter.

Retries on HTTP 429, 5xx, timeouts and transport errors. Honours a numeric
``Retry-After`` header (capped at ``backoff_max_seconds``). Never retries
other 4xx: a 401/403/404 will not fix itself.

Backoff sleeps use real ``asyncio.sleep`` on purpose: they wait on a remote
server, not on market time, so the Clock abstraction does not apply.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from warroom.config import HttpSection
from warroom.providers.base import ProviderError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpError(ProviderError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def backoff_delay(attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    return rng() * min(cap, base * (2 ** attempt))


class RetryingHttpClient:
    def __init__(
        self,
        cfg: HttpSection,
        headers: dict[str, str] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.cfg = cfg
        self._sleep = sleep
        self._rng = rng
        self._client = httpx.AsyncClient(
            timeout=cfg.timeout_seconds,
            headers={"User-Agent": "trading-war-room/0.1", **(headers or {})},
            transport=transport,
            follow_redirects=True,
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = await self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                status = None
                retry_after = None
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS:
                    raise HttpError(f"{method} {_redact(url)} -> HTTP {resp.status_code}", resp.status_code)
                status = resp.status_code
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                last_exc = HttpError(f"{method} {_redact(url)} -> HTTP {status}", status)

            if attempt >= self.cfg.max_retries:
                break
            delay = backoff_delay(attempt, self.cfg.backoff_base_seconds, self.cfg.backoff_max_seconds, self._rng)
            if retry_after is not None:
                delay = max(delay, min(retry_after, self.cfg.backoff_max_seconds))
            log.warning("retrying %s %s in %.2fs (attempt %d, %s)", method, _redact(url), delay, attempt + 1,
                        status or type(last_exc).__name__)
            await self._sleep(delay)

        if isinstance(last_exc, HttpError):
            raise last_exc
        raise HttpError(f"{method} {_redact(url)} failed after retries: {last_exc!r}") from last_exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form: fall back to computed backoff


def _redact(url: str) -> str:
    """Strip query strings from logged URLs (some APIs put tokens there)."""
    return url.split("?", 1)[0]
