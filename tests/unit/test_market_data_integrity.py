from __future__ import annotations

import asyncio
import time
from typing import Any

from core.event_bus import EventBus
from data_ingestion.market_data_integrity import MarketDataIntegrityMonitor
from data_ingestion.normalizer import Candle


class _Config:
    paper_mode = True

    def __init__(self) -> None:
        self._cfg = {
            "enabled": True,
            "enforce_signal_gate": True,
            "tick_stale_seconds": 10.0,
            "orderbook_stale_seconds": 90.0,
            "signal_event_stale_seconds": 5.0,
            "candle_stale_multiple": 2.5,
            "max_clock_drift_seconds": 2.0,
            "gap_quarantine_seconds": 30.0,
            "check_interval_seconds": 1.0,
        }

    def get_value(self, *keys: str, default: Any = None) -> Any:
        if not keys:
            return default
        if keys[0] != "market_data_integrity":
            return default
        if len(keys) == 1:
            return dict(self._cfg)
        return self._cfg.get(keys[1], default)


def _monitor() -> MarketDataIntegrityMonitor:
    return MarketDataIntegrityMonitor(_Config(), EventBus())  # type: ignore[arg-type]


def test_fresh_candle_allows_signal_gate() -> None:
    mon = _monitor()
    now_us = time.time_ns() // 1000
    candle = Candle(
        exchange="binance",
        symbol="BTC/USDT:USDT",
        timeframe="15m",
        timestamp=int(time.time()),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
        receive_time_us=now_us,
        source_tick_timestamp_us=now_us,
    )

    asyncio.run(mon._handle_candle(candle))
    ok, reason, health = mon.check_signal_allowed(
        "binance",
        "BTC/USDT:USDT",
        timeframe="15m",
        receive_time_us=candle.receive_time_us,
        source_tick_timestamp_us=candle.source_tick_timestamp_us,
    )

    assert ok is True
    assert reason.startswith("fresh_event_age_")
    assert health["status"] == "OK"


def test_sequence_gap_blocks_even_with_fresh_event() -> None:
    mon = _monitor()
    asyncio.run(mon._handle_market_data_gap({
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "expected": 2,
        "actual": 5,
        "gap": 3,
    }))

    ok, reason, health = mon.check_signal_allowed(
        "binance",
        "BTC/USDT:USDT",
        timeframe="15m",
        receive_time_us=time.time_ns() // 1000,
    )

    assert ok is False
    assert reason == "sequence_gap_3"
    assert health["status"] == "GAP"
    assert health["sequence_gap_count"] == 3


def test_crossed_orderbook_marks_feed_invalid() -> None:
    mon = _monitor()
    asyncio.run(mon._handle_orderbook_update({
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "bids": [(101.0, 1.0)],
        "asks": [(100.0, 1.0)],
        "receive_time_us": time.time_ns() // 1000,
    }))

    snap = mon.snapshot()
    assert snap["status"] == "BLOCKED"
    assert snap["feeds"][0]["status"] == "INVALID"
    assert snap["feeds"][0]["reason"] == "crossed_orderbook"


def test_partial_depth_update_ids_do_not_trigger_sequence_gap() -> None:
    mon = _monitor()
    now_us = time.time_ns() // 1000

    asyncio.run(mon._handle_orderbook_update({
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "bids": [(50000.0, 1.0)],
        "asks": [(50001.0, 1.0)],
        "receive_time_us": now_us,
        "timestamp_us": now_us,
        "last_update_id": 100,
        "source": "ws_partial_depth",
    }))
    asyncio.run(mon._handle_orderbook_update({
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "bids": [(50002.0, 1.0)],
        "asks": [(50003.0, 1.0)],
        "receive_time_us": now_us,
        "timestamp_us": now_us,
        "last_update_id": 100000,
        "source": "ws_partial_depth",
    }))

    snap = mon.snapshot()
    assert snap["status"] == "OK"
    assert snap["feeds"][0]["status"] == "OK"
    assert snap["feeds"][0]["sequence_gap_count"] == 0


def test_candle_timeframes_are_gated_independently() -> None:
    mon = _monitor()
    now_us = time.time_ns() // 1000

    for tf in ("1m", "1h"):
        mon.observe_candle(Candle(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            timeframe=tf,
            timestamp=int(time.time()),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
            receive_time_us=now_us,
            source_tick_timestamp_us=now_us,
        ))

    stale_cutoff = time.time() - 600.0
    for health in mon._health.values():
        if health.last_candle_timeframe == "1m":
            health.last_event_ts = stale_cutoff
            health.updated_at = stale_cutoff

    ok_1m, reason_1m, health_1m = mon.check_signal_allowed(
        "binance",
        "BTC/USDT:USDT",
        timeframe="1m",
    )
    ok_1h, reason_1h, health_1h = mon.check_signal_allowed(
        "binance",
        "BTC/USDT:USDT",
        timeframe="1h",
    )

    assert ok_1m is False
    assert "market_data_stale" in reason_1m
    assert health_1m["last_candle_timeframe"] == "1m"
    assert ok_1h is True
    assert reason_1h == "market_data_current"
    assert health_1h["last_candle_timeframe"] == "1h"
