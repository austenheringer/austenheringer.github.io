"""Optional Discord / Telegram notifications, driven by EventBus events.

Configured by ``notifications.*`` in config.yaml and webhook/token values in
.env. Failures are logged (with secrets redacted) and never propagate: a
dead webhook must not affect trading. Messages are rate limited and
de-duplicated so a flapping alert cannot spam a channel.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from abc import ABC, abstractmethod

from warroom.config import AppConfig, HttpSection, Secrets
from warroom.core.event_bus import EventBus, Subscription
from warroom.models.schemas import AlertLevel, BaseEvent, JarvisBriefing, SystemAlert, TradeExecuted
from warroom.providers.base import ProviderError
from warroom.providers.http import RetryingHttpClient

log = logging.getLogger(__name__)


class Notifier(ABC):
    name = "base"

    @abstractmethod
    async def send(self, text: str) -> None: ...

    async def aclose(self) -> None:  # noqa: B027
        pass


class DiscordWebhookNotifier(Notifier):
    name = "discord"

    def __init__(self, webhook_url: str, http_cfg: HttpSection, transport=None) -> None:
        if not webhook_url.startswith("https://"):
            raise ValueError("Discord webhook URL must be https")
        self._url = webhook_url
        self.http = RetryingHttpClient(http_cfg, transport=transport)

    async def send(self, text: str) -> None:
        await self.http.post(self._url, json={"content": text[:1900], "allowed_mentions": {"parse": []}})

    async def aclose(self) -> None:
        await self.http.aclose()


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, http_cfg: HttpSection, transport=None) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat = chat_id
        self.http = RetryingHttpClient(http_cfg, transport=transport)

    async def send(self, text: str) -> None:
        await self.http.post(self._url, json={"chat_id": self._chat, "text": text[:4000], "disable_web_page_preview": True})

    async def aclose(self) -> None:
        await self.http.aclose()


def format_event(event: BaseEvent, notify_on: set[str]) -> str | None:
    """Return the notification text for an event, or None if it is not wanted."""
    if isinstance(event, JarvisBriefing) and event.kind == "briefing" and event.status in notify_on:
        return f"[War Room {event.status}] {event.text}"
    if isinstance(event, TradeExecuted) and "TRADE_EXECUTED" in notify_on:
        f, p = event.fill, event.position
        action = "OPEN" if f.is_entry else f"CLOSE ({f.reason.value if f.reason else 'exit'})"
        pnl = "" if f.is_entry else f" P&L {f.realized_pnl - f.commission:+.2f}"
        return f"[Paper {action}] {p.ticker} {p.direction.value} {f.qty} @ {f.price:.2f}{pnl}"
    if isinstance(event, SystemAlert) and "CIRCUIT_BREAKER" in notify_on and (
            event.level is AlertLevel.CRITICAL or event.source == "risk"):
        return f"[{event.level.value}] {event.source}: {event.message}"
    return None


class NotificationHub:
    def __init__(self, notifiers: list[Notifier], notify_on: list[str], min_interval_seconds: float = 1.0) -> None:
        self.notifiers = notifiers
        self.notify_on = set(notify_on)
        self.min_interval = min_interval_seconds
        self._last_sent: dict[str, float] = {}
        self._last_any = 0.0
        self._task: asyncio.Task | None = None
        self._sub: Subscription | None = None
        self.sent = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.notifiers)

    def start(self, bus: EventBus) -> None:
        if not self.notifiers:
            return
        self._sub = bus.subscribe(JarvisBriefing, TradeExecuted, SystemAlert, maxsize=200, name="notifications")
        self._task = bus.attach_handler(self._sub, self.handle)

    async def handle(self, event: BaseEvent) -> None:
        text = format_event(event, self.notify_on)
        if text is None:
            return
        now = _time.monotonic()
        if now - self._last_sent.get(text, -1e9) < 300:  # identical text within 5 minutes: drop
            return
        wait = self.min_interval - (now - self._last_any)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_any = _time.monotonic()
        self._last_sent[text] = self._last_any
        for n in self.notifiers:
            try:
                await n.send(text)
                self.sent += 1
            except (ProviderError, ValueError) as exc:
                self.failed += 1
                log.warning("%s notification failed: %s", n.name, exc)

    async def stop(self) -> None:
        if self._sub is not None:
            self._sub.unsubscribe()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for n in self.notifiers:
            await n.aclose()


def build_notifications(cfg: AppConfig, secrets: Secrets) -> NotificationHub:
    out: list[Notifier] = []
    n = cfg.notifications
    if n.discord_enabled and secrets.discord_webhook_url:
        try:
            out.append(DiscordWebhookNotifier(secrets.discord_webhook_url.get_secret_value(), cfg.http))
        except ValueError as exc:
            log.warning("Discord disabled: %s", exc)
    if n.telegram_enabled and secrets.telegram_bot_token and secrets.telegram_chat_id:
        out.append(TelegramNotifier(secrets.telegram_bot_token.get_secret_value(), secrets.telegram_chat_id, cfg.http))
    return NotificationHub(out, list(n.notify_on))
