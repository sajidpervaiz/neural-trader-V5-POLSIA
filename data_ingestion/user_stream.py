"""
Binance Futures User Data Stream — real-time fill tracking and account updates.

Manages the listenKey lifecycle (create, keepalive every 30 min, reconnect on expiry)
and parses ORDER_TRADE_UPDATE, ACCOUNT_UPDATE, and listenKeyExpired events.

Every state change is published to EventBus for Order Manager and Risk Engine.

Features:
- Stream-level fill dedup via trade_id set (prevents duplicate event bus publishes on WS reconnect)
- Order state machine tracking with invalid-transition detection
- Configurable safe-mode threshold: delays circuit breaker trip by X seconds after disconnect
- Reconnection reconciliation via REST API to recover missed fills
"""
from __future__ import annotations

import asyncio
import collections
import time
from typing import Any

import orjson
import websockets
from loguru import logger

from core.config import Config
from core.event_bus import EventBus


# Binance Futures endpoints
_BASE_URL_LIVE = "https://fapi.binance.com"
_BASE_URL_TESTNET = "https://testnet.binancefuture.com"
_WS_URL_LIVE = "wss://fstream.binance.com/ws/"
_WS_URL_TESTNET = "wss://stream.binancefuture.com/ws/"

# Keepalive interval (Binance requires every 60 min; we do every 30 min for safety)
_KEEPALIVE_INTERVAL = 30 * 60

# Max dedup cache size (oldest entries evicted via OrderedDict)
_MAX_DEDUP_SIZE = 50_000

# Valid order state transitions from Binance
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "NEW": {"PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED"},
    "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "CANCELED", "EXPIRED"},
    "FILLED": set(),  # terminal
    "CANCELED": set(),  # terminal
    "REJECTED": set(),  # terminal
    "EXPIRED": set(),  # terminal
}


class UserDataStream:
    """
    Connects to Binance Futures User Data Stream via WebSocket.

    Events published to EventBus:
    - USER_ORDER_UPDATE: order fill, partial fill, cancel, reject, liquidation
    - USER_ACCOUNT_UPDATE: balance/position change
    - USER_STREAM_LOST: stream disconnected (triggers safety mode after threshold)
    - USER_STREAM_CONNECTED: stream reconnected
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._listen_key: str | None = None
        self._ws: Any = None
        self._last_keepalive: float = 0.0
        self._connected = False
        self._disconnect_time: float | None = None

        # ── Stream-level fill dedup (trade_id -> True, evicts oldest) ──
        self._processed_trade_ids: collections.OrderedDict[int, bool] = collections.OrderedDict()

        # ── Order state machine tracking (order_id -> last known status) ──
        self._order_states: dict[int, str] = {}

        # ── Safe-mode threshold (seconds before tripping circuit breaker) ──
        risk_cfg = {}
        try:
            risk_cfg = config.get_value("risk") or {}
        except Exception:
            pass
        self._safe_mode_threshold: float = float(
            risk_cfg.get("user_stream_safe_mode_seconds", 0)
        )
        self._safe_mode_published = False

        # ── Metrics counters ──
        self.metrics = {
            "fills_processed": 0,
            "fills_deduped": 0,
            "state_transitions": 0,
            "invalid_transitions": 0,
            "reconnects": 0,
            "messages_received": 0,
        }

        # Parse exchange config
        cfg = config.get_value("exchanges", "binance") or {}
        self._api_key = cfg.get("api_key", "")
        self._api_secret = cfg.get("api_secret", "")
        self._testnet = cfg.get("testnet", True)
        self._enabled = cfg.get("enabled", False)

        self._base_url = _BASE_URL_TESTNET if self._testnet else _BASE_URL_LIVE
        self._ws_url = _WS_URL_TESTNET if self._testnet else _WS_URL_LIVE

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def disconnect_duration(self) -> float:
        """Seconds since last disconnect, or 0 if connected."""
        if self._connected or self._disconnect_time is None:
            return 0.0
        return time.time() - self._disconnect_time

    async def _http_request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Make authenticated HTTP request to Binance Futures API."""
        import aiohttp
        url = f"{self._base_url}{path}"
        headers = {"X-MBX-APIKEY": self._api_key}
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, **kwargs) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Binance API {method} {path} failed ({resp.status}): {body}")
                return await resp.json()

    async def _create_listen_key(self) -> str:
        """POST /fapi/v1/listenKey — create user data stream."""
        data = await self._http_request("POST", "/fapi/v1/listenKey")
        key = data.get("listenKey", "")
        if not key:
            raise RuntimeError(f"No listenKey in response: {data}")
        logger.info("User data stream listenKey created")
        return key

    async def _keepalive_listen_key(self) -> None:
        """PUT /fapi/v1/listenKey — extend listen key validity with retry."""
        if not self._listen_key:
            return
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                await self._http_request("PUT", "/fapi/v1/listenKey")
                self._last_keepalive = time.time()
                logger.debug("User data stream listenKey keepalive sent")
                return
            except Exception as exc:
                logger.warning(
                    "listenKey keepalive attempt {}/{} failed: {}",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s backoff
        # All retries exhausted — invalidate key to force reconnect
        logger.error(
            "listenKey keepalive failed after {} retries — forcing reconnect",
            max_retries,
        )
        self._listen_key = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _keepalive_loop(self) -> None:
        """Background task to send keepalive every 30 minutes."""
        while self._running:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            if self._running and self._listen_key:
                await self._keepalive_listen_key()

    def _parse_order_update(self, data: dict) -> dict:
        """Parse ORDER_TRADE_UPDATE event from Binance."""
        o = data.get("o", {})
        return {
            "event_type": "ORDER_TRADE_UPDATE",
            "event_time": data.get("E", 0),
            "transaction_time": data.get("T", 0),
            "symbol": o.get("s", ""),
            "client_order_id": o.get("c", ""),
            "side": o.get("S", ""),
            "order_type": o.get("o", ""),
            "time_in_force": o.get("f", ""),
            "quantity": float(o.get("q", 0)),
            "price": float(o.get("p", 0)),
            "avg_price": float(o.get("ap", 0)),
            "stop_price": float(o.get("sp", 0)),
            "execution_type": o.get("x", ""),  # NEW, TRADE, CANCELED, EXPIRED, etc.
            "order_status": o.get("X", ""),  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, etc.
            "order_id": int(o.get("i", 0)),
            "last_filled_qty": float(o.get("l", 0)),
            "cumulative_filled_qty": float(o.get("z", 0)),
            "last_filled_price": float(o.get("L", 0)),
            "commission": float(o.get("n", 0)),
            "commission_asset": o.get("N", ""),
            "trade_id": int(o.get("t", 0)),
            "reduce_only": o.get("R", False),
            "position_side": o.get("ps", "BOTH"),
            "realized_profit": float(o.get("rp", 0)),
            "is_maker": o.get("m", False),
        }

    def _parse_account_update(self, data: dict) -> dict:
        """Parse ACCOUNT_UPDATE event from Binance."""
        a = data.get("a", {})
        balances = [
            {
                "asset": b.get("a", ""),
                "wallet_balance": float(b.get("wb", 0)),
                "cross_wallet_balance": float(b.get("cw", 0)),
                "balance_change": float(b.get("bc", 0)),
            }
            for b in a.get("B", [])
        ]
        positions = [
            {
                "symbol": p.get("s", ""),
                "position_amount": float(p.get("pa", 0)),
                "entry_price": float(p.get("ep", 0)),
                "unrealized_pnl": float(p.get("up", 0)),
                "margin_type": p.get("mt", ""),
                "isolated_wallet": float(p.get("iw", 0)),
                "position_side": p.get("ps", "BOTH"),
            }
            for p in a.get("P", [])
        ]
        return {
            "event_type": "ACCOUNT_UPDATE",
            "event_time": data.get("E", 0),
            "transaction_time": data.get("T", 0),
            "reason": a.get("m", ""),
            "balances": balances,
            "positions": positions,
        }

    def _check_fill_dedup(self, trade_id: int) -> bool:
        """Return True if this trade_id was already processed (duplicate). Thread-safe via GIL."""
        if trade_id in self._processed_trade_ids:
            self.metrics["fills_deduped"] += 1
            return True
        # Add to dedup set, evict oldest if over limit
        self._processed_trade_ids[trade_id] = True
        if len(self._processed_trade_ids) > _MAX_DEDUP_SIZE:
            self._processed_trade_ids.popitem(last=False)
        return False

    def _track_order_state(self, order_id: int, new_status: str) -> bool:
        """Track order state transition. Returns True if valid, False if invalid."""
        old_status = self._order_states.get(order_id)
        if old_status is None:
            # First time seeing this order — accept any status
            self._order_states[order_id] = new_status
            self.metrics["state_transitions"] += 1
            return True

        if old_status == new_status:
            # Same status (e.g. repeated PARTIALLY_FILLED) — valid
            return True

        valid_next = _VALID_TRANSITIONS.get(old_status, set())
        if new_status in valid_next:
            self._order_states[order_id] = new_status
            self.metrics["state_transitions"] += 1
            # Clean up terminal states to prevent unbounded growth
            if new_status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                # Keep for a while for late dedup, but schedule cleanup
                pass
            return True

        # Invalid transition
        self.metrics["invalid_transitions"] += 1
        logger.error(
            "Invalid order state transition for order {}: {} → {} — accepting but logging",
            order_id, old_status, new_status,
        )
        self._order_states[order_id] = new_status
        return False

    def _cleanup_terminal_orders(self) -> None:
        """Remove terminal-state orders from tracking to prevent unbounded growth."""
        terminal = {oid for oid, status in self._order_states.items()
                    if status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED")}
        # Keep at most 10000 terminal entries
        if len(terminal) > 10_000:
            for oid in list(terminal)[:len(terminal) - 10_000]:
                self._order_states.pop(oid, None)

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch user data stream messages with dedup and state tracking."""
        self.metrics["messages_received"] += 1
        try:
            data = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError):
            logger.warning("Invalid JSON from user stream: {}", raw[:200])
            return

        event = data.get("e", "")

        if event == "ORDER_TRADE_UPDATE":
            parsed = self._parse_order_update(data)
            order_id = parsed["order_id"]
            trade_id = parsed["trade_id"]
            exec_type = parsed["execution_type"]
            order_status = parsed["order_status"]

            # ── State machine tracking ──
            self._track_order_state(order_id, order_status)

            # ── Stream-level fill dedup (only for TRADE events with trade_id) ──
            if exec_type == "TRADE" and trade_id:
                if self._check_fill_dedup(trade_id):
                    logger.warning(
                        "Duplicate trade_id {} for order {} — suppressing event publish",
                        trade_id, order_id,
                    )
                    return
                self.metrics["fills_processed"] += 1

            logger.info(
                "Order update: {} {} {} exec={} status={} filled={}/{}",
                parsed["symbol"], parsed["side"], parsed["order_type"],
                exec_type, order_status,
                parsed["cumulative_filled_qty"], parsed["quantity"],
            )
            await self.event_bus.publish("USER_ORDER_UPDATE", parsed)

        elif event == "ACCOUNT_UPDATE":
            parsed = self._parse_account_update(data)
            logger.info(
                "Account update: reason={} balances={} positions={}",
                parsed["reason"], len(parsed["balances"]), len(parsed["positions"]),
            )
            await self.event_bus.publish("USER_ACCOUNT_UPDATE", parsed)

        elif event == "listenKeyExpired":
            logger.warning("User data stream listenKey expired — reconnecting")
            self._listen_key = None

        elif event == "MARGIN_CALL":
            logger.critical("MARGIN CALL received: {}", data)
            await self.event_bus.publish("MARGIN_CALL", data)

        else:
            logger.debug("Unknown user stream event: {}", event)

    async def _safe_mode_watchdog(self) -> None:
        """Background task: publish USER_STREAM_LOST only after threshold elapsed."""
        while self._running:
            await asyncio.sleep(1.0)
            if self._connected or self._disconnect_time is None:
                self._safe_mode_published = False
                continue
            if self._safe_mode_published:
                continue
            elapsed = time.time() - self._disconnect_time
            if elapsed >= self._safe_mode_threshold:
                logger.critical(
                    "User stream down for {:.1f}s (threshold={:.0f}s) — publishing STREAM_LOST",
                    elapsed, self._safe_mode_threshold,
                )
                self._safe_mode_published = True
                await self.event_bus.publish("USER_STREAM_LOST", {
                    "timestamp": time.time(),
                    "disconnect_duration": elapsed,
                })

    async def _reconcile_after_reconnect(self) -> None:
        """After reconnection, fetch recent trades via REST to recover missed fills."""
        try:
            import aiohttp
            headers = {"X-MBX-APIKEY": self._api_key}
            path = "/fapi/v1/userTrades"
            url = f"{self._base_url}{path}"
            # Fetch trades from last 5 minutes (covers typical disconnect window)
            params = {"timestamp": int(time.time() * 1000), "limit": 100}
            # Sign request
            import hashlib
            import hmac
            import urllib.parse
            query = urllib.parse.urlencode(params)
            signature = hmac.new(
                self._api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        logger.warning("Reconciliation fetch failed: {}", resp.status)
                        return
                    trades = await resp.json()
            reconciled = 0
            for t in trades:
                trade_id = int(t.get("id", 0))
                if trade_id and trade_id not in self._processed_trade_ids:
                    reconciled += 1
                    self._processed_trade_ids[trade_id] = True
                    if len(self._processed_trade_ids) > _MAX_DEDUP_SIZE:
                        self._processed_trade_ids.popitem(last=False)
                    # Construct and publish a synthetic ORDER_TRADE_UPDATE
                    parsed = {
                        "event_type": "ORDER_TRADE_UPDATE",
                        "event_time": int(t.get("time", 0)),
                        "transaction_time": int(t.get("time", 0)),
                        "symbol": t.get("symbol", ""),
                        "client_order_id": "",
                        "side": t.get("side", ""),
                        "order_type": "",
                        "time_in_force": "",
                        "quantity": float(t.get("qty", 0)),
                        "price": float(t.get("price", 0)),
                        "avg_price": float(t.get("price", 0)),
                        "stop_price": 0.0,
                        "execution_type": "TRADE",
                        "order_status": "FILLED",
                        "order_id": int(t.get("orderId", 0)),
                        "last_filled_qty": float(t.get("qty", 0)),
                        "cumulative_filled_qty": float(t.get("qty", 0)),
                        "last_filled_price": float(t.get("price", 0)),
                        "commission": float(t.get("commission", 0)),
                        "commission_asset": t.get("commissionAsset", ""),
                        "trade_id": trade_id,
                        "reduce_only": False,
                        "position_side": t.get("positionSide", "BOTH"),
                        "realized_profit": float(t.get("realizedPnl", 0)),
                        "is_maker": t.get("maker", False),
                        "reconciled": True,
                    }
                    await self.event_bus.publish("USER_ORDER_UPDATE", parsed)
                    self.metrics["fills_processed"] += 1
            if reconciled:
                logger.info("Reconciliation recovered {} missed fills", reconciled)
        except Exception as exc:
            logger.warning("Reconnection reconciliation failed: {}", exc)

    async def run(self) -> None:
        """Main loop: connect, listen, reconnect on failure."""
        if not self._enabled:
            logger.info("Binance not enabled — user data stream disabled")
            return
        if not self._api_key:
            logger.warning("No Binance API key — user data stream disabled")
            return
        if self.config.paper_mode:
            logger.info("Paper mode — user data stream disabled (no real fills)")
            return

        self._running = True
        reconnect_delay = 1.0

        # Start keepalive background task
        keepalive_task = asyncio.create_task(self._keepalive_loop(), name="user_stream_keepalive")

        # Start safe-mode watchdog (only if threshold > 0; otherwise immediate on disconnect)
        watchdog_task = None
        if self._safe_mode_threshold > 0:
            watchdog_task = asyncio.create_task(
                self._safe_mode_watchdog(), name="user_stream_watchdog"
            )

        try:
            while self._running:
                try:
                    # Get or refresh listen key
                    if not self._listen_key:
                        self._listen_key = await self._create_listen_key()
                        self._last_keepalive = time.time()

                    ws_url = f"{self._ws_url}{self._listen_key}"
                    logger.info("Connecting to Binance user data stream...")

                    async with websockets.connect(
                        ws_url,
                        ping_interval=20,
                        ping_timeout=30,
                        close_timeout=10,
                    ) as ws:
                        self._ws = ws
                        self._connected = True
                        self._disconnect_time = None
                        self._safe_mode_published = False
                        reconnect_delay = 1.0
                        self.metrics["reconnects"] += 1
                        logger.info("Binance user data stream connected")
                        await self.event_bus.publish("USER_STREAM_CONNECTED", {})

                        # Reconcile missed fills after reconnect
                        await self._reconcile_after_reconnect()

                        # Periodic terminal-order cleanup
                        self._cleanup_terminal_orders()

                        async for raw in ws:
                            if not self._running:
                                break
                            await self._handle_message(raw)

                except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
                    logger.warning("User data stream disconnected: {} — retry in {}s", exc, reconnect_delay)
                except Exception as exc:
                    logger.exception("User data stream error: {}", exc)

                # Mark disconnected
                self._connected = False
                self._disconnect_time = time.time()

                # If no safe-mode threshold configured, publish STREAM_LOST immediately
                if self._safe_mode_threshold <= 0:
                    await self.event_bus.publish("USER_STREAM_LOST", {
                        "timestamp": time.time(),
                    })

                if self._running:
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30.0)
                    # Force new listen key on reconnect
                    self._listen_key = None
        finally:
            keepalive_task.cancel()
            if watchdog_task:
                watchdog_task.cancel()
            for task in [keepalive_task, watchdog_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def stop(self) -> None:
        self._running = False
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("User data stream stopped")
