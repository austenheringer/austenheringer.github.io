import asyncio

import pytest

from tests.conftest import et
from warroom.core.event_bus import BusClosed, EventBus
from warroom.models.schemas import MarketUpdate, SystemAlert


def _alert(msg="hi"):
    return SystemAlert(timestamp=et(10), level="INFO", source="t", message=msg)


async def test_typed_filtering_and_fanout():
    bus = EventBus()
    alerts = bus.subscribe(SystemAlert)
    everything = bus.subscribe()
    bus.publish(_alert())
    bus.publish(MarketUpdate(timestamp=et(10), ticker="A", price=1.0))
    assert (await alerts.get()).message == "hi" and alerts.pending() == 0
    assert everything.pending() == 2


async def test_overflow_drops_oldest_without_blocking():
    bus = EventBus()
    sub = bus.subscribe(maxsize=2)
    for p in (1.0, 2.0, 3.0):
        bus.publish(MarketUpdate(timestamp=et(10), ticker="A", price=p))
    assert sub.dropped == 1
    assert [(await sub.get()).price, (await sub.get()).price] == [2.0, 3.0]


async def test_close_ends_iteration():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(_alert("last"))
    bus.close()
    assert [e.message async for e in sub] == ["last"]
    assert bus.publish(_alert()) == 0
    with pytest.raises(BusClosed):
        bus.subscribe()


async def test_rejects_non_events():
    with pytest.raises(TypeError):
        EventBus().publish({"not": "an event"})


async def test_handler_errors_do_not_kill_pump():
    bus = EventBus()
    seen = []

    def handler(e):
        if e.message == "boom":
            raise RuntimeError("bad consumer")
        seen.append(e.message)

    task = bus.attach_handler(bus.subscribe(SystemAlert, name="h"), handler)
    bus.publish(_alert("boom"))
    bus.publish(_alert("ok"))
    await asyncio.sleep(0.01)
    bus.close()
    await asyncio.wait_for(task, 1)
    assert seen == ["ok"]
