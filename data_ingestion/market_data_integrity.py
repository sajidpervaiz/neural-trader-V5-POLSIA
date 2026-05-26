from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.config import Config
from core.error_handling import sanitize_exception
from core.event_bus import EventBus


_TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def _get(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timeframe_seconds(timeframe: str | None, default: int = 60) -> int:
    return _TIMEFRAME_SECONDS.get(str(timeframe or "").strip(), default)


@dataclass
class FeedHealth:
    exchange: str
    symbol: str
    channel: str = "market"
    status: str = "WARMING"
    healthy: bool = False
    reason: str = "waiting_for_market_data"
    last_event_type: str = ""
    last_event_ts: float = 0.0
    last_receive_time_us: int = 0
    last_exchange_ts_us: int = 0
    last_sequence: int = 0
    last_candle_timeframe: str = ""
    age_ms: float = 0.0
    exchange_lag_ms: float = 0.0
    sequence_gap_count: int = 0
    out_of_order_count: int = 0
    invalid_book_count: int = 0
    tick_count: int = 0
    candle_count: int = 0
    orderbook_count: int = 0
    last_gap: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        ts = now if now is not None else time.time()
        age_s = max(0.0, ts - self.last_event_ts) if self.last_event_ts else 0.0
        age_ms = round(age_s * 1000.0, 1)
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "channel": self.channel,
            "status": self.status,
            "healthy": self.healthy,
            "reason": self.reason,
            "last_event_type": self.last_event_type,
            "last_event_ts": self.last_event_ts,
            "last_receive_time_us": self.last_receive_time_us,
            "last_exchange_ts_us": self.last_exchange_ts_us,
            "last_sequence": self.last_sequence,
            "last_candle_timeframe": self.last_candle_timeframe,
            "age_ms": age_ms,
            "exchange_lag_ms": round(self.exchange_lag_ms, 3),
            "sequence_gap_count": self.sequence_gap_count,
            "out_of_order_count": self.out_of_order_count,
            "invalid_book_count": self.invalid_book_count,
            "tick_count": self.tick_count,
            "candle_count": self.candle_count,
            "orderbook_count": self.orderbook_count,
            "last_gap": dict(self.last_gap),
            "updated_at": self.updated_at,
        }


class MarketDataIntegrityMonitor:
    """L0 feed-health gate for ticks, candles, order books, and gaps.

    The monitor is deliberately independent of a specific exchange client. Feed
    components publish normal TICK/CANDLE/ORDERBOOK_UPDATE events and, when
    available, MARKET_DATA_GAP events. This class turns those into one compact
    health contract that the signal layer and dashboard can consume.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        cfg = config.get_value("market_data_integrity", default={}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.enforce_signal_gate = bool(cfg.get("enforce_signal_gate", True))
        self.tick_stale_seconds = float(cfg.get("tick_stale_seconds", 10.0))
        self.orderbook_stale_seconds = float(cfg.get("orderbook_stale_seconds", 90.0))
        self.signal_event_stale_seconds = float(cfg.get("signal_event_stale_seconds", 5.0))
        self.candle_stale_multiple = float(cfg.get("candle_stale_multiple", 2.5))
        self.max_clock_drift_seconds = float(cfg.get("max_clock_drift_seconds", 2.0))
        self.gap_quarantine_seconds = float(cfg.get("gap_quarantine_seconds", 30.0))
        self.check_interval_seconds = float(cfg.get("check_interval_seconds", 1.0))
        self._health: dict[str, FeedHealth] = {}
        self._running = False

    def _key(self, exchange: str, symbol: str, channel: str = "", timeframe: str = "") -> str:
        event_channel = str(channel or "").upper()
        if event_channel == "CANDLE":
            return f"{exchange}:{symbol}:CANDLE:{timeframe or 'unknown'}"
        if event_channel:
            return f"{exchange}:{symbol}:{event_channel}"
        return f"{exchange}:{symbol}"

    def _health_for(self, exchange: str, symbol: str, channel: str = "", timeframe: str = "") -> FeedHealth:
        key = self._key(exchange, symbol, channel=channel, timeframe=timeframe)
        if key not in self._health:
            self._health[key] = FeedHealth(
                exchange=exchange,
                symbol=symbol,
                channel=str(channel or "market").upper() or "market",
                last_candle_timeframe=timeframe if str(channel or "").upper() == "CANDLE" else "",
            )
        return self._health[key]

    def _feeds_for_symbol(self, exchange: str, symbol: str) -> list[FeedHealth]:
        return [
            health
            for health in self._health.values()
            if health.exchange == exchange and health.symbol == symbol
        ]

    def _select_signal_health(self, exchange: str, symbol: str, timeframe: str = "") -> FeedHealth | None:
        feeds = self._feeds_for_symbol(exchange, symbol)
        if not feeds:
            return None
        if timeframe:
            exact = [
                health
                for health in feeds
                if health.channel == "CANDLE" and health.last_candle_timeframe == timeframe
            ]
            if exact:
                return max(exact, key=lambda h: h.updated_at)
        preferred = [health for health in feeds if health.channel in {"TICK", "ORDERBOOK_UPDATE"}]
        if preferred:
            return max(preferred, key=lambda h: h.updated_at)
        return max(feeds, key=lambda h: h.updated_at)

    def _mark_event(
        self,
        payload: Any,
        event_type: str,
        *,
        receive_time_us: int = 0,
        exchange_ts_us: int = 0,
        sequence: int = 0,
        timeframe: str = "",
    ) -> FeedHealth | None:
        exchange = str(_get(payload, "exchange", "") or "")
        symbol = str(_get(payload, "symbol", "") or "")
        if not exchange or not symbol:
            return None

        now = time.time()
        if receive_time_us <= 0:
            receive_time_us = int(now * 1_000_000)
        health = self._health_for(exchange, symbol, channel=event_type, timeframe=timeframe)
        previous_ts = health.last_exchange_ts_us
        previous_seq = health.last_sequence

        health.last_event_type = event_type
        health.last_event_ts = now
        health.last_receive_time_us = int(receive_time_us)
        health.last_exchange_ts_us = int(exchange_ts_us or 0)
        health.updated_at = now
        health.status = "OK"
        health.healthy = True
        health.reason = "market_data_current"
        if timeframe:
            health.last_candle_timeframe = timeframe

        if event_type == "TICK":
            health.tick_count += 1
        elif event_type == "CANDLE":
            health.candle_count += 1
        elif event_type == "ORDERBOOK_UPDATE":
            health.orderbook_count += 1

        if exchange_ts_us > 0:
            health.exchange_lag_ms = max(0.0, (receive_time_us - exchange_ts_us) / 1000.0)
            future_drift_s = (exchange_ts_us / 1_000_000.0) - now
            if future_drift_s > self.max_clock_drift_seconds:
                health.status = "INVALID"
                health.healthy = False
                health.reason = f"exchange_timestamp_future_{future_drift_s:.2f}s"
            elif event_type != "CANDLE" and previous_ts > 0 and exchange_ts_us < previous_ts:
                health.out_of_order_count += 1
                health.status = "OUT_OF_ORDER"
                health.healthy = False
                health.reason = "exchange_timestamp_out_of_order"

        if sequence > 0:
            if previous_seq > 0 and sequence > previous_seq + 1:
                gap = sequence - previous_seq - 1
                health.sequence_gap_count += gap
                health.last_gap = {
                    "expected": previous_seq + 1,
                    "actual": sequence,
                    "gap": gap,
                    "ts": int(now),
                }
                health.status = "GAP"
                health.healthy = False
                health.reason = f"sequence_gap_{gap}"
            health.last_sequence = sequence

        return health

    async def _handle_tick(self, payload: Any) -> None:
        sequence = _int(_get(payload, "sequence", 0))
        gap = _int(_get(payload, "sequence_gap", 0))
        health = self._mark_event(
            payload,
            "TICK",
            receive_time_us=_int(_get(payload, "receive_time_us", 0)),
            exchange_ts_us=_int(_get(payload, "timestamp_us", 0)),
            sequence=sequence,
        )
        if health is not None and gap > 0:
            health.sequence_gap_count += gap
            health.last_gap = {"gap": gap, "source": "tick", "ts": int(time.time())}
            health.status = "GAP"
            health.healthy = False
            health.reason = f"sequence_gap_{gap}"

    def observe_candle(self, payload: Any) -> FeedHealth | None:
        now_us = time.time_ns() // 1000
        receive_us = _int(_get(payload, "receive_time_us", 0), now_us)
        exchange_ts_us = _int(_get(payload, "source_tick_timestamp_us", 0))
        if exchange_ts_us <= 0:
            exchange_ts_us = _int(_get(payload, "timestamp", 0)) * 1_000_000
        return self._mark_event(
            payload,
            "CANDLE",
            receive_time_us=receive_us,
            exchange_ts_us=exchange_ts_us,
            timeframe=str(_get(payload, "timeframe", "") or ""),
        )

    async def _handle_candle(self, payload: Any) -> None:
        self.observe_candle(payload)

    async def _handle_orderbook_update(self, payload: Any) -> None:
        bids = list(_get(payload, "bids", []) or [])
        asks = list(_get(payload, "asks", []) or [])
        # Partial-depth streams and REST snapshots expose exchange update IDs,
        # but those IDs are not a contiguous event sequence. Treating
        # last_update_id as +1 ordered produced false L0 sequence gaps and kept
        # paper mode blocked. Only explicitly marked contiguous order-book
        # streams may drive sequence quarantine; gap-capable feeds should emit
        # MARKET_DATA_GAP themselves when their own continuity rules fail.
        sequence_contiguous = bool(_get(payload, "sequence_contiguous", False))
        orderbook_sequence = _int(_get(payload, "sequence", 0)) if sequence_contiguous else 0
        health = self._mark_event(
            payload,
            "ORDERBOOK_UPDATE",
            receive_time_us=_int(_get(payload, "receive_time_us", 0), time.time_ns() // 1000),
            exchange_ts_us=_int(_get(payload, "timestamp_us", 0)),
            sequence=orderbook_sequence,
        )
        if health is None:
            return
        try:
            best_bid = _float(bids[0][0]) if bids else 0.0
            best_ask = _float(asks[0][0]) if asks else 0.0
            if best_bid > 0.0 and best_ask > 0.0 and best_bid >= best_ask:
                health.invalid_book_count += 1
                health.status = "INVALID"
                health.healthy = False
                health.reason = "crossed_orderbook"
        except Exception:
            health.invalid_book_count += 1
            health.status = "INVALID"
            health.healthy = False
            health.reason = "orderbook_parse_error"

    async def _handle_market_data_gap(self, payload: Any) -> None:
        exchange = str(_get(payload, "exchange", "") or "")
        symbol = str(_get(payload, "symbol", "") or "")
        if not exchange or not symbol:
            return
        source = str(_get(payload, "source", "") or "").lower()
        channel = "ORDERBOOK_UPDATE" if "orderbook" in source else "TICK"
        health = self._health_for(exchange, symbol, channel=channel)
        gap = max(1, _int(_get(payload, "gap", 1), 1))
        health.sequence_gap_count += gap
        health.last_event_type = "MARKET_DATA_GAP"
        health.last_event_ts = time.time()
        health.updated_at = health.last_event_ts
        health.last_gap = {
            "expected": _int(_get(payload, "expected", 0)),
            "actual": _int(_get(payload, "actual", 0)),
            "gap": gap,
            "ts": _int(_get(payload, "ts", int(time.time()))),
        }
        health.status = "GAP"
        health.healthy = False
        health.reason = f"sequence_gap_{gap}"

    async def _handle_alert(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "ws_prolonged_disconnect":
            return
        exchange = str(payload.get("exchange", "") or "")
        if not exchange:
            return
        now = time.time()
        for health in self._health.values():
            if health.exchange != exchange:
                continue
            health.status = "STALE"
            health.healthy = False
            health.reason = "websocket_prolonged_disconnect"
            health.updated_at = now

    def _max_age_seconds(self, health: FeedHealth) -> float:
        if health.last_event_type == "ORDERBOOK_UPDATE":
            return self.orderbook_stale_seconds
        if health.last_event_type == "CANDLE":
            tf_seconds = _timeframe_seconds(health.last_candle_timeframe, default=60)
            return max(self.signal_event_stale_seconds, tf_seconds * self.candle_stale_multiple)
        return self.tick_stale_seconds

    def _refresh_stale_states(self) -> None:
        now = time.time()
        for health in self._health.values():
            if not health.last_event_ts:
                continue
            age_s = now - health.last_event_ts
            if age_s <= self._max_age_seconds(health):
                continue
            health.status = "STALE"
            health.healthy = False
            health.reason = f"market_data_stale_{age_s:.1f}s"
            health.updated_at = now

    def check_signal_allowed(
        self,
        exchange: str,
        symbol: str,
        *,
        timeframe: str = "",
        receive_time_us: int = 0,
        source_tick_timestamp_us: int = 0,
    ) -> tuple[bool, str, dict[str, Any]]:
        if not self.enabled:
            return True, "market_data_integrity_disabled", {"enabled": False}
        if not self.enforce_signal_gate:
            return True, "market_data_integrity_observe_only", {"enabled": True, "enforced": False}

        now = time.time()
        health = self._select_signal_health(exchange, symbol, timeframe=timeframe)
        for blocked_health in self._feeds_for_symbol(exchange, symbol):
            if blocked_health.status in {"GAP", "INVALID", "OUT_OF_ORDER"}:
                if (now - blocked_health.updated_at) <= self.gap_quarantine_seconds:
                    return False, blocked_health.reason, blocked_health.to_dict(now)

        if receive_time_us > 0:
            event_age_s = now - (receive_time_us / 1_000_000.0)
            if event_age_s <= self.signal_event_stale_seconds:
                return True, f"fresh_event_age_{event_age_s:.3f}s", (health.to_dict(now) if health else {})
            return False, f"fresh_event_stale_{event_age_s:.1f}s", (health.to_dict(now) if health else {})

        if health is None:
            if self.config.paper_mode:
                return False, "waiting_for_paper_market_data", {}
            return False, "waiting_for_live_market_data", {}

        self._refresh_stale_states()
        health_payload = health.to_dict(now)
        if health.healthy:
            return True, health.reason, health_payload
        return False, health.reason, health_payload

    def snapshot(self) -> dict[str, Any]:
        self._refresh_stale_states()
        now = time.time()
        feeds = [health.to_dict(now) for health in self._health.values()]
        healthy = [feed for feed in feeds if bool(feed.get("healthy"))]
        unhealthy = [feed for feed in feeds if not bool(feed.get("healthy"))]
        status = "OK"
        if unhealthy and healthy:
            status = "DEGRADED"
        elif unhealthy and not healthy:
            status = "BLOCKED"
        elif not feeds:
            status = "WARMING"
        return {
            "enabled": self.enabled,
            "enforced": self.enforce_signal_gate,
            "status": status,
            "healthy": status == "OK",
            "feed_count": len(feeds),
            "healthy_count": len(healthy),
            "unhealthy_count": len(unhealthy),
            "thresholds": {
                "tick_stale_seconds": self.tick_stale_seconds,
                "orderbook_stale_seconds": self.orderbook_stale_seconds,
                "signal_event_stale_seconds": self.signal_event_stale_seconds,
                "candle_stale_multiple": self.candle_stale_multiple,
                "max_clock_drift_seconds": self.max_clock_drift_seconds,
                "gap_quarantine_seconds": self.gap_quarantine_seconds,
            },
            "feeds": feeds,
        }

    async def run(self) -> None:
        self._running = True
        self.event_bus.subscribe("TICK", self._handle_tick)
        self.event_bus.subscribe("CANDLE", self._handle_candle)
        self.event_bus.subscribe("ORDERBOOK_UPDATE", self._handle_orderbook_update)
        self.event_bus.subscribe("MARKET_DATA_GAP", self._handle_market_data_gap)
        self.event_bus.subscribe("ALERT_CRITICAL", self._handle_alert)
        logger.info(
            "MarketDataIntegrityMonitor started (enabled={}, enforced={}, tick_stale={}s)",
            self.enabled,
            self.enforce_signal_gate,
            self.tick_stale_seconds,
        )
        try:
            while self._running:
                self._refresh_stale_states()
                await asyncio.sleep(self.check_interval_seconds)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("MarketDataIntegrityMonitor crashed: {}", sanitize_exception(exc))
            raise

    async def stop(self) -> None:
        self._running = False
        self.event_bus.unsubscribe("TICK", self._handle_tick)
        self.event_bus.unsubscribe("CANDLE", self._handle_candle)
        self.event_bus.unsubscribe("ORDERBOOK_UPDATE", self._handle_orderbook_update)
        self.event_bus.unsubscribe("MARKET_DATA_GAP", self._handle_market_data_gap)
        self.event_bus.unsubscribe("ALERT_CRITICAL", self._handle_alert)
