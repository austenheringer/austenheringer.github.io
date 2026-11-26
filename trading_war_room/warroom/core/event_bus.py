"""Typed async pub/sub event bus.

Decouples the orchestrator (producer) from the TUI, headless reporters,
notifiers and the voice queue (consumers).

Design choices
--------------
* ``publish`` is synchronous and never blocks: a slow consumer (e.g. a TUI
  redraw) must never stall the trading loop. Each subscription owns a bounded
  queue; on overflow the OLDEST event is dropped and counted. Consumers that
  cannot tolerate loss (the database) are called directly by the
  orchestrator, not fed through the bus.
* Subscriptions filter by event class (``isinstance``), so subscribing to
  ``BaseEvent`` receives everything.
* Must be used from the event-loop thread. Worker threads use
  :meth:`EventBus.publish_threadsafe`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Generic, TypeVar

from warroom.models.schemas import BaseEvent

log = logging.getLogger(__name__)

E = TypeVar("E", bound=BaseEvent)

_CLOSED = object()


class BusClosed(Exception):
    pass


class Subscription(Generic[E]):
    def __init__(self, bus: "EventBus", types: tuple[type[E], ...], maxsize: int, name: str) -> None:
        self._bus = bus
        self.types = types
        self.name = name
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def matches(self, event: BaseEvent) -> bool:
        return isinstance(event, self.types)

    def _offer(self, item: object) -> None:
        if self.closed and item is not _CLOSED:
            return
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                    if self.dropped in (1, 100) or self.dropped % 1000 == 0:
                        log.warning("subscription %s overflowing; dropped %d events", self.name, self.dropped)
                except asyncio.QueueEmpty:  # pragma: no cover - race-free in single thread
                    pass

    def pending(self) -> int:
        return self._queue.qsize()

    async def get(self, timeout: float | None = None) -> E:
        item = await (asyncio.wait_for(self._queue.get(), timeout) if timeout is not None else self._queue.get())
        if item is _CLOSED:
            self.closed = True
            raise BusClosed(self.name)
        return item  # type: ignore[return-value]

    def get_nowait(self) -> E | None:
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if item is _CLOSED:
            self.closed = True
            raise BusClosed(self.name)
        return item  # type: ignore[return-value]

    def __aiter__(self) -> AsyncIterator[E]:
        return self

    async def __anext__(self) -> E:
        try:
            return await self.get()
        except BusClosed:
            raise StopAsyncIteration from None

    def unsubscribe(self) -> None:
        self._bus._remove(self)
        if not self.closed:
            self.closed = True
            self._offer(_CLOSED)


class EventBus:
    def __init__(self, default_maxsize: int = 1000) -> None:
        self._subs: list[Subscription] = []
        self._default_maxsize = default_maxsize
        self._closed = False
        self.published = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def subscribe(
        self, *types: type[E], maxsize: int | None = None, name: str = "subscriber"
    ) -> Subscription[E]:
        """Subscribe to one or more event classes (none = all events)."""
        if self._closed:
            raise BusClosed("cannot subscribe to a closed bus")
        for t in types:
            if not (isinstance(t, type) and issubclass(t, BaseEvent)):
                raise TypeError(f"{t!r} is not a BaseEvent subclass")
        sub: Subscription[E] = Subscription(
            self, types or (BaseEvent,), maxsize or self._default_maxsize, name  # type: ignore[arg-type]
        )
        self._subs.append(sub)
        return sub

    def publish(self, event: BaseEvent) -> int:
        """Deliver ``event`` to matching subscribers. Returns delivery count."""
        if not isinstance(event, BaseEvent):
            raise TypeError(f"can only publish BaseEvent instances, got {type(event).__name__}")
        if self._closed:
            return 0
        self.published += 1
        delivered = 0
        for sub in list(self._subs):
            if sub.matches(event):
                sub._offer(event)
                delivered += 1
        return delivered

    def publish_threadsafe(self, event: BaseEvent, loop: asyncio.AbstractEventLoop) -> None:
        loop.call_soon_threadsafe(self.publish, event)

    def attach_handler(
        self,
        sub: Subscription[E],
        handler: Callable[[E], Awaitable[None] | None],
    ) -> asyncio.Task[None]:
        """Run ``handler`` for every event on ``sub`` in a background task.
        Handler exceptions are logged, never propagated into the bus."""

        async def _pump() -> None:
            async for event in sub:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - a bad consumer must not kill the pump
                    log.exception("event handler %s failed on %s", sub.name, type(event).__name__)

        return asyncio.create_task(_pump(), name=f"bus-handler-{sub.name}")

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    def close(self) -> None:
        """Signal end-of-stream to all subscribers; further publishes are ignored."""
        if self._closed:
            return
        self._closed = True
        for sub in list(self._subs):
            sub.closed = True
            sub._offer(_CLOSED)
        self._subs.clear()
