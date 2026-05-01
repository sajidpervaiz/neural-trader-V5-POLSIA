"""REQ-ARC-003 / REQ-MON-001: EventBus must expose backpressure observability."""
from __future__ import annotations

import asyncio

import pytest

from core.event_bus import EventBus


@pytest.mark.asyncio
async def test_stats_initial_shape() -> None:
    bus = EventBus()
    s = bus.stats()
    assert s["queue_size"] == 0
    assert s["queue_capacity"] >= 1
    assert s["queue_pct"] == 0.0
    assert s["dropped_count"] == 0
    assert s["backpressure_warned"] is False
    assert s["subscribed_topics"] == 0
    assert s["subscriber_counts"] == {}
    assert s["running"] is False


@pytest.mark.asyncio
async def test_stats_reflects_subscribers() -> None:
    bus = EventBus()
    async def _h1(p): return None
    async def _h2(p): return None
    bus.subscribe("X", _h1)
    bus.subscribe("X", _h2)
    bus.subscribe("Y", _h1)
    s = bus.stats()
    assert s["subscribed_topics"] == 2
    assert s["subscriber_counts"]["X"] == 2
    assert s["subscriber_counts"]["Y"] == 1


@pytest.mark.asyncio
async def test_dropped_count_increments_on_full_queue() -> None:
    bus = EventBus()
    # Fill the queue to capacity using publish_nowait so non-critical events
    # would be dropped instead of awaited.
    cap = bus._queue.maxsize
    for i in range(cap):
        bus.publish_nowait("TICK", payload=i)
    # One more triggers the drop path on a non-critical event.
    bus.publish_nowait("TICK", payload="overflow")
    s = bus.stats()
    assert s["queue_size"] == cap
    assert s["queue_pct"] == 100.0
    assert s["dropped_count"] == 1
