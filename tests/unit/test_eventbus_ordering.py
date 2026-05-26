from __future__ import annotations

import asyncio

from core.event_bus import EventBus


def test_market_data_events_are_ordered_per_symbol() -> None:
    seen: list[int] = []
    bus = EventBus()

    async def handler(payload: dict) -> None:
        await asyncio.sleep(0.02 if payload["n"] == 1 else 0.0)
        seen.append(payload["n"])

    async def scenario() -> None:
        bus.subscribe("TICK", handler)
        await bus._dispatch("TICK", {"exchange": "binance", "symbol": "BTC/USDT:USDT", "n": 1})
        await bus._dispatch("TICK", {"exchange": "binance", "symbol": "BTC/USDT:USDT", "n": 2})
        await asyncio.sleep(0.08)

    asyncio.run(scenario())

    assert seen == [1, 2]
    assert bus.stats()["ordered_lanes"] == 1


def test_eventbus_stamps_event_metadata() -> None:
    bus = EventBus()
    payload = {"exchange": "binance", "symbol": "ETH/USDT:USDT"}

    asyncio.run(bus.publish("TICK", payload))

    meta = payload["_event_meta"]
    assert meta["event_type"] == "TICK"
    assert meta["event_sequence"] == 1
    assert meta["event_id"] == "TICK:1"
    assert meta["order_key"] == "binance:ETH/USDT:USDT"
