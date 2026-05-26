from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from core.circuit_breaker import CircuitBreaker, CircuitState
from core.idempotency import IdempotencyManager
from core.retry import RetryPolicy, with_retry


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


@dataclass
class L2DepthLevel:
    price: float
    quantity: float
    orders_count: int = 0


@dataclass
class OrderbookSnapshot:
    symbol: str
    timestamp_ms: int
    bids: list[L2DepthLevel]
    asks: list[L2DepthLevel]
    sequence: int = 0


@dataclass
class AlgoOrder:
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    filled_quantity: float = 0.0
    algo_type: str = "TWAP"
    duration_seconds: int = 60
    slices: int = 12
    status: str = "PENDING"
    child_orders: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ImbalanceSignal:
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float
    direction: str
    confidence: float


class LegacyCEXExecutorBase:
    """Shared order lifecycle for legacy venue-specific executors."""

    exchange_label = "CEX"
    supports_reduce_only = False

    def _init_common_state(self, enable_paper_trading: bool = True) -> None:
        self.paper_trading = enable_paper_trading
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
        )
        self.idempotency = IdempotencyManager(ttl=3600)
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
        )
        self.orderbook_cache: dict[str, OrderbookSnapshot] = {}
        self.algo_orders: dict[str, AlgoOrder] = {}
        self.balance_cache: dict[str, float] = {}
        self.position_cache: dict[str, Any] = {}
        self._running = False

    async def initialize(self) -> None:
        """Initialize the executor with connectivity checks."""
        try:
            await self.exchange.load_markets()
            logger.info("{} executor initialized successfully", self.exchange_label)
            if self.paper_trading:
                logger.info("Running in PAPER TRADING mode - no real orders will be placed")
        except Exception as exc:
            logger.error("Failed to initialize {} executor: {}", self.exchange_label, exc)
            raise

    def _build_order_params(
        self,
        time_in_force: TimeInForce | None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.supports_reduce_only:
            params["reduceOnly"] = reduce_only
        if time_in_force:
            params["timeInForce"] = time_in_force.value
        return params

    @with_retry()
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        idempotency_key: str | None = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place an order with idempotency, circuit breaking, and retry logic."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is open - rejecting order placement")

        if idempotency_key and self.idempotency.check_and_set(idempotency_key):
            logger.info("Duplicate order detected: {}", idempotency_key)
            return {"status": "DUPLICATE", "idempotency_key": idempotency_key}

        if self.paper_trading:
            logger.info("[PAPER] Placing {} {} {} @ {}", side.value, quantity, symbol, order_type.value)
            return {
                "id": f"paper_{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "amount": quantity,
                "price": price,
                "status": "closed",
                "filled": quantity,
                "remaining": 0.0,
            }

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type.value,
                side=side.value,
                amount=quantity,
                price=price,
                params=self._build_order_params(time_in_force, reduce_only=reduce_only),
            )
            self.circuit_breaker.record_success()
            logger.info("Order placed successfully: {}", order["id"])
            return order
        except Exception as exc:
            self.circuit_breaker.record_failure()
            logger.error("Failed to place order: {}", exc)
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order."""
        if self.paper_trading:
            logger.info("[PAPER] Cancelled order {}", order_id)
            return True

        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info("Order cancelled: {}", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order {}: {}", order_id, exc)
            return False

    async def get_orderbook_snapshot(self, symbol: str, depth: int = 20) -> OrderbookSnapshot:
        """Fetch and cache an L2 orderbook snapshot."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is open")

        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=depth)
            bids = [L2DepthLevel(price=b[0], quantity=b[1]) for b in orderbook["bids"][:depth]]
            asks = [L2DepthLevel(price=a[0], quantity=a[1]) for a in orderbook["asks"][:depth]]
            snapshot = OrderbookSnapshot(
                symbol=symbol,
                timestamp_ms=int(time.time() * 1000),
                bids=bids,
                asks=asks,
                sequence=orderbook.get("timestamp", 0),
            )
            self.orderbook_cache[symbol] = snapshot
            return snapshot
        except Exception as exc:
            self.circuit_breaker.record_failure()
            logger.error("Failed to fetch orderbook for {}: {}", symbol, exc)
            raise

    async def get_orderbook_from_cache(self, symbol: str) -> OrderbookSnapshot | None:
        """Get cached orderbook snapshot."""
        return self.orderbook_cache.get(symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if self.paper_trading:
            logger.info("[PAPER] Setting leverage for {} to {}x", symbol, leverage)
            return True

        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info("Leverage set for {} to {}x", symbol, leverage)
            return True
        except Exception as exc:
            logger.error("Failed to set leverage: {}", exc)
            return False

    async def close(self) -> None:
        """Clean up resources."""
        self._running = False
        await self.exchange.close()
        logger.info("{} executor closed", self.exchange_label)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
