from __future__ import annotations

import asyncio
import json

from data_ingestion.orderbook_feed import OrderbookFeed


class _ConfigStub:
    def __init__(self) -> None:
        self._data = {
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "testnet": False,
                    "symbols": ["BTC/USDT:USDT"],
                }
            },
            "data_ingestion": {
                "orderbook_feed": {
                    "mode": "hybrid",
                    "depth": 20,
                    "update_speed_ms": 250,
                    "rest_poll_interval_seconds": 30.0,
                }
            },
        }

    def get_value(self, *keys, default=None):
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def publish(self, event: str, payload=None) -> None:
        self.events.append((event, payload))


def test_ws_partial_depth_message_publishes_orderbook_update() -> None:
    cfg = _ConfigStub()
    bus = _Bus()
    feed = OrderbookFeed(cfg, bus)
    raw = json.dumps({
        "stream": "btcusdt@depth20@250ms",
        "data": {
            "e": "depthUpdate",
            "E": 1710000000000,
            "T": 1710000000001,
            "s": "BTCUSDT",
            "U": 100,
            "u": 101,
            "pu": 99,
            "b": [["50000.0", "1.5"]],
            "a": [["50001.0", "1.25"]],
        },
    })

    asyncio.run(feed._handle_ws_message(raw, {"BTCUSDT": "BTC/USDT:USDT"}))

    assert bus.events[0][0] == "ORDERBOOK_UPDATE"
    payload = bus.events[0][1]
    assert payload["source"] == "ws_partial_depth"
    assert payload["symbol"] == "BTC/USDT:USDT"
    assert payload["bids"] == [(50000.0, 1.5)]
    assert payload["asks"] == [(50001.0, 1.25)]
    assert payload["last_update_id"] == 101


def test_ws_partial_depth_gap_publishes_market_data_gap() -> None:
    cfg = _ConfigStub()
    bus = _Bus()
    feed = OrderbookFeed(cfg, bus)
    feed._last_update_id["binance:BTC/USDT:USDT"] = 101
    raw = json.dumps({
        "data": {
            "s": "BTCUSDT",
            "U": 110,
            "u": 111,
            "pu": 105,
            "b": [["50000.0", "1.5"]],
            "a": [["50001.0", "1.25"]],
        },
    })

    asyncio.run(feed._handle_ws_message(raw, {"BTCUSDT": "BTC/USDT:USDT"}))

    assert bus.events[0][0] == "MARKET_DATA_GAP"
    assert bus.events[0][1]["source"] == "orderbook_ws"
    assert bus.events[1][0] == "ORDERBOOK_UPDATE"


def test_ws_partial_depth_uses_stream_name_when_symbol_missing() -> None:
    cfg = _ConfigStub()
    bus = _Bus()
    feed = OrderbookFeed(cfg, bus)
    raw = json.dumps({
        "stream": "btcusdt@depth20@250ms",
        "data": {
            "lastUpdateId": 202,
            "bids": [["50000.0", "1.5"]],
            "asks": [["50001.0", "1.25"]],
        },
    })

    asyncio.run(feed._handle_ws_message(raw, {"BTCUSDT": "BTC/USDT:USDT"}))

    assert bus.events[0][0] == "ORDERBOOK_UPDATE"
    assert bus.events[0][1]["symbol"] == "BTC/USDT:USDT"
    assert bus.events[0][1]["last_update_id"] == 202
