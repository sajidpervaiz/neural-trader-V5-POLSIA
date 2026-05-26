"""Orderbook depth feed.

Uses Binance futures partial-depth WebSocket updates for low-latency top-of-book
state, with periodic REST snapshots as a resilience fallback.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import orjson
import websockets
from loguru import logger

from core.config import Config
from core.event_bus import EventBus


_BINANCE_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
_BINANCE_TESTNET_DEPTH_URL = "https://testnet.binancefuture.com/fapi/v1/depth"
_BINANCE_DEPTH_WS_URL = "wss://fstream.binance.com/stream?streams={streams}"
_BINANCE_TESTNET_DEPTH_WS_URL = "wss://stream.binancefuture.com/stream?streams={streams}"


def _symbol_to_binance(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "").upper()


class OrderbookFeed:
    """Maintains orderbook updates for signal scoring, routing, and risk."""

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        poll_interval: float = 30.0,
        depth: int = 20,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        ob_cfg = self._get_orderbook_cfg()
        self._mode = str(ob_cfg.get("mode", "hybrid")).lower()
        self._interval = float(
            ob_cfg.get(
                "rest_poll_interval_seconds",
                ob_cfg.get("poll_interval_seconds", poll_interval),
            )
        )
        self._depth = int(ob_cfg.get("depth", depth))
        self._depth = 20 if self._depth not in {5, 10, 20} else self._depth
        self._update_speed_ms = int(ob_cfg.get("update_speed_ms", 250))
        self._update_speed_ms = 250 if self._update_speed_ms not in {100, 250, 500} else self._update_speed_ms
        self._reconnect_min_s = float(ob_cfg.get("ws_reconnect_min_seconds", 1.0))
        self._reconnect_max_s = float(ob_cfg.get("ws_reconnect_max_seconds", 30.0))
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._ws_connections: dict[str, Any] = {}
        self._last_update_id: dict[str, int] = {}

    def _get_orderbook_cfg(self) -> dict[str, Any]:
        data_cfg = self.config.get_value("data_ingestion", default={}) or {}
        if not isinstance(data_cfg, dict):
            return {}
        ob_cfg = data_cfg.get("orderbook_feed", {}) or {}
        return ob_cfg if isinstance(ob_cfg, dict) else {}

    def _get_symbols(self) -> list[str]:
        binance_cfg = self.config.get_value("exchanges", "binance") or {}
        return list(binance_cfg.get("symbols", []) or [])

    def _get_base_url(self) -> str:
        binance_cfg = self.config.get_value("exchanges", "binance") or {}
        if binance_cfg.get("testnet", True):
            return _BINANCE_TESTNET_DEPTH_URL
        return _BINANCE_DEPTH_URL

    def _get_ws_url(self) -> str:
        binance_cfg = self.config.get_value("exchanges", "binance") or {}
        if binance_cfg.get("testnet", True):
            return _BINANCE_TESTNET_DEPTH_WS_URL
        return _BINANCE_DEPTH_WS_URL

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def _fetch_depth(self, symbol: str) -> dict[str, Any] | None:
        session = await self._get_session()
        url = self._get_base_url()
        try:
            async with session.get(
                url,
                params={"symbol": _symbol_to_binance(symbol), "limit": self._depth},
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception as exc:
            logger.debug("Orderbook REST fetch error for {}: {}", symbol, exc)
            return None

    async def _publish_orderbook(
        self,
        *,
        symbol: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        timestamp_us: int,
        receive_time_us: int,
        last_update_id: int,
        source: str,
        first_update_id: int = 0,
        previous_update_id: int = 0,
    ) -> None:
        if not bids and not asks:
            return
        await self.event_bus.publish("ORDERBOOK_UPDATE", {
            "exchange": "binance",
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp_us": timestamp_us,
            "receive_time_us": receive_time_us,
            "last_update_id": last_update_id,
            "first_update_id": first_update_id,
            "previous_update_id": previous_update_id,
            "source": source,
        })

    async def _poll_once(self) -> None:
        for symbol in self._get_symbols():
            data = await self._fetch_depth(symbol)
            if data is None:
                continue
            bids = [
                (float(price), float(qty))
                for price, qty in data.get("bids", [])
            ]
            asks = [
                (float(price), float(qty))
                for price, qty in data.get("asks", [])
            ]
            await self._publish_orderbook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp_us=int(data.get("E", 0) or data.get("T", 0) or 0) * 1000,
                receive_time_us=int(time.time_ns() // 1000),
                last_update_id=int(data.get("lastUpdateId", 0) or 0),
                source="rest_snapshot",
            )

    def _stream_url(self, symbols: list[str]) -> str:
        streams = "/".join(
            f"{_symbol_to_binance(symbol).lower()}@depth{self._depth}@{self._update_speed_ms}ms"
            for symbol in symbols
        )
        return self._get_ws_url().format(streams=streams)

    async def _handle_ws_message(
        self,
        raw: str | bytes,
        symbol_map: dict[str, str],
    ) -> None:
        receive_time_us = int(time.time_ns() // 1000)
        try:
            payload = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError, TypeError):
            try:
                payload = json.loads(raw)
            except Exception:
                return

        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        if not isinstance(data, dict):
            return

        raw_symbol = str(data.get("s") or data.get("symbol") or "").upper()
        if not raw_symbol and isinstance(payload, dict):
            stream_name = str(payload.get("stream", ""))
            raw_symbol = stream_name.split("@", 1)[0].upper()
        symbol = symbol_map.get(raw_symbol)
        if not symbol:
            return

        bid_rows = data.get("b") or data.get("bids") or []
        ask_rows = data.get("a") or data.get("asks") or []
        bids = [
            (float(price), float(qty))
            for price, qty in bid_rows[: self._depth]
        ]
        asks = [
            (float(price), float(qty))
            for price, qty in ask_rows[: self._depth]
        ]

        first_update_id = int(data.get("U", 0) or 0)
        last_update_id = int(data.get("u", 0) or data.get("lastUpdateId", 0) or 0)
        previous_update_id = int(data.get("pu", 0) or 0)
        key = f"binance:{symbol}"
        prior_update_id = self._last_update_id.get(key, 0)

        has_gap = False
        expected = prior_update_id + 1 if prior_update_id else 0
        actual = last_update_id
        if prior_update_id > 0 and previous_update_id > 0 and previous_update_id != prior_update_id:
            has_gap = True
            expected = prior_update_id
            actual = previous_update_id
        elif prior_update_id > 0 and first_update_id > 0 and first_update_id > prior_update_id + 1:
            has_gap = True
            expected = prior_update_id + 1
            actual = first_update_id

        if has_gap:
            await self.event_bus.publish("MARKET_DATA_GAP", {
                "exchange": "binance",
                "symbol": symbol,
                "expected": expected,
                "actual": actual,
                "gap": max(0, actual - expected),
                "source": "orderbook_ws",
                "ts": int(time.time()),
            })
            logger.warning(
                "binance {} orderbook sequence gap: expected {} got {}",
                symbol, expected, actual,
            )

        if last_update_id > 0:
            self._last_update_id[key] = last_update_id

        await self._publish_orderbook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp_us=int(data.get("E", 0) or data.get("T", 0) or 0) * 1000,
            receive_time_us=receive_time_us,
            last_update_id=last_update_id,
            first_update_id=first_update_id,
            previous_update_id=previous_update_id,
            source="ws_partial_depth",
        )

    async def _run_ws_partial_depth(self) -> None:
        symbols = self._get_symbols()
        if not symbols:
            logger.warning("OrderbookFeed WS idle: no Binance symbols configured")
            while self._running:
                await asyncio.sleep(5)
            return

        symbol_map = {_symbol_to_binance(symbol): symbol for symbol in symbols}
        url = self._stream_url(symbols)
        reconnect_delay = self._reconnect_min_s

        while self._running:
            try:
                logger.info(
                    "OrderbookFeed WS connecting (depth={}, speed={}ms, symbols={})",
                    self._depth, self._update_speed_ms, len(symbols),
                )
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                    max_queue=2048,
                ) as ws:
                    self._ws_connections["binance"] = ws
                    reconnect_delay = self._reconnect_min_s
                    logger.info("OrderbookFeed WS connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_ws_message(raw, symbol_map)
            except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
                logger.warning("OrderbookFeed WS disconnected: {} - retry in {}s", exc, reconnect_delay)
            except Exception as exc:
                logger.warning("OrderbookFeed WS error: {} - retry in {}s", exc, reconnect_delay)
            finally:
                self._ws_connections.pop("binance", None)

            if self._running:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._reconnect_max_s)

    async def _run_rest_poll(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.warning("OrderbookFeed REST poll error: {}", exc)
            await asyncio.sleep(self._interval)

    async def run(self) -> None:
        self._running = True
        symbols = self._get_symbols()
        logger.info(
            "OrderbookFeed started (mode={}, rest_interval={}s, depth={}, speed={}ms, symbols={})",
            self._mode, self._interval, self._depth, self._update_speed_ms, len(symbols),
        )
        tasks: list[asyncio.Task] = []
        if self._mode in {"rest", "hybrid"}:
            tasks.append(asyncio.create_task(self._run_rest_poll(), name="orderbook_rest_poll"))
        if self._mode in {"ws", "websocket", "ws_partial", "hybrid"}:
            tasks.append(asyncio.create_task(self._run_ws_partial_depth(), name="orderbook_ws_partial"))
        if not tasks:
            logger.warning("OrderbookFeed mode '{}' unknown; falling back to REST polling", self._mode)
            tasks.append(asyncio.create_task(self._run_rest_poll(), name="orderbook_rest_poll"))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False
        for ws in list(self._ws_connections.values()):
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("OrderbookFeed WS close error: {}", exc)
        if self._session and not self._session.closed:
            await self._session.close()
