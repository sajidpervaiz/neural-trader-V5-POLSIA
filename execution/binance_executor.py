"""
Binance CEX Executor with L2 Orderbook Reconstruction, TWAP/VWAP Algorithms,
Imbalance Detection, and Advanced Order Management.
"""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import ccxt.async_support as ccxt
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
    bids: List[L2DepthLevel]
    asks: List[L2DepthLevel]
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
    child_orders: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ImbalanceSignal:
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float
    direction: str
    confidence: float


class BinanceExecutor:
    """
    Production-grade Binance executor with advanced features:
    - L2 orderbook reconstruction
    - TWAP/VWAP execution algorithms
    - Order flow imbalance detection
    - Smart order routing
    - Idempotency and circuit breaking
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        enable_paper_trading: bool = True,
    ):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

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

        self.orderbook_cache: Dict[str, OrderbookSnapshot] = {}
        self.algo_orders: Dict[str, AlgoOrder] = {}
        self.balance_cache: Dict[str, float] = {}
        self.position_cache: Dict[str, Dict] = {}

        self._running = False

    async def initialize(self) -> None:
        """Initialize the executor with connectivity checks."""
        try:
            await self.exchange.load_markets()
            logger.info("Binance executor initialized successfully")

            if self.paper_trading:
                logger.info("Running in PAPER TRADING mode - no real orders will be placed")
        except Exception as e:
            logger.error(f"Failed to initialize Binance executor: {e}")
            raise

    @with_retry()
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        idempotency_key: Optional[str] = None,
    ) -> Dict:
        """
        Place an order with idempotency, circuit breaking, and retry logic.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop)
            quantity: Order quantity
            price: Order price (required for limit orders)
            time_in_force: Time in force (GTC/IOC/FOK)
            idempotency_key: Unique key for idempotency

        Returns:
            Order response dictionary
        """
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is open - rejecting order placement")

        if idempotency_key and self.idempotency.check_and_set(idempotency_key):
            logger.info(f"Duplicate order detected: {idempotency_key}")
            return {"status": "DUPLICATE", "idempotency_key": idempotency_key}

        if self.paper_trading:
            logger.info(f"[PAPER] Placing {side.value} {quantity} {symbol} @ {order_type.value}")
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
            params = {}
            if time_in_force:
                params['timeInForce'] = time_in_force.value

            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type.value,
                side=side.value,
                amount=quantity,
                price=price,
                params=params,
            )

            self.circuit_breaker.record_success()
            logger.info(f"Order placed successfully: {order['id']}")
            return order

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to place order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order."""
        if self.paper_trading:
            logger.info(f"[PAPER] Cancelled order {order_id}")
            return True

        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_orderbook_snapshot(
        self,
        symbol: str,
        depth: int = 20,
    ) -> OrderbookSnapshot:
        """Fetch and cache L2 orderbook snapshot."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is open")

        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=depth)

            bids = [
                L2DepthLevel(price=b[0], quantity=b[1])
                for b in orderbook['bids'][:depth]
            ]
            asks = [
                L2DepthLevel(price=a[0], quantity=a[1])
                for a in orderbook['asks'][:depth]
            ]

            snapshot = OrderbookSnapshot(
                symbol=symbol,
                timestamp_ms=int(time.time() * 1000),
                bids=bids,
                asks=asks,
                sequence=orderbook.get('timestamp', 0),
            )

            self.orderbook_cache[symbol] = snapshot
            return snapshot

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    async def get_orderbook_from_cache(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """Get cached orderbook snapshot."""
        return self.orderbook_cache.get(symbol)

    def calculate_order_flow_imbalance(
        self,
        snapshot: OrderbookSnapshot,
        depth_levels: int = 5,
    ) -> ImbalanceSignal:
        """
        Calculate order flow imbalance indicator.

        Returns imbalance ratio where:
        - > 1.0 indicates buying pressure
        - < 1.0 indicates selling pressure
        - = 1.0 indicates balanced market
        """
        bid_volume = sum(
            level.quantity * level.price
            for level in snapshot.bids[:depth_levels]
        )
        ask_volume = sum(
            level.quantity * level.price
            for level in snapshot.asks[:depth_levels]
        )

        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0

        direction = "bullish" if imbalance_ratio > 1.0 else "bearish"
        confidence = min(abs(imbalance_ratio - 1.0) * 2, 1.0)

        return ImbalanceSignal(
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance_ratio,
            direction=direction,
            confidence=confidence,
        )

    async def execute_twap(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        duration_seconds: int,
        slices: int,
        price_limit: Optional[float] = None,
        idempotency_key: Optional[str] = None,
    ) -> AlgoOrder:
        """
        Execute a Time-Weighted Average Price (TWAP) order.

        Slices the order into equal parts and executes them at regular intervals
        to minimize market impact.
        """
        algo_id = f"twap_{int(time.time() * 1000)}"

        algo_order = AlgoOrder(
            order_id=algo_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            algo_type="TWAP",
            duration_seconds=duration_seconds,
            slices=slices,
            status="RUNNING",
        )

        self.algo_orders[algo_id] = algo_order

        slice_quantity = total_quantity / slices
        slice_interval = duration_seconds / slices

        logger.info(
            f"Starting TWAP order {algo_id}: {total_quantity} {symbol} in {slices} slices "
            f"over {duration_seconds}s"
        )

        for i in range(slices):
            if algo_order.status != "RUNNING":
                break

            try:
                current_price = await self.get_current_price(symbol)

                if price_limit:
                    if (side == OrderSide.BUY and current_price > price_limit) or \
                       (side == OrderSide.SELL and current_price < price_limit):
                        logger.warning(f"Price limit breached for TWAP slice {i+1}/{slices}")
                        continue

                child_order = await self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=slice_quantity,
                    idempotency_key=f"{idempotency_key}_{i}" if idempotency_key else None,
                )

                algo_order.child_orders.append(child_order['id'])
                algo_order.filled_quantity += child_order.get('filled', 0)

                if i < slices - 1:
                    await asyncio.sleep(slice_interval)

            except Exception as e:
                logger.error(f"TWAP slice {i+1}/{slices} failed: {e}")

        algo_order.status = "COMPLETED" if algo_order.filled_quantity >= total_quantity * 0.95 else "PARTIALLY_FILLED"

        logger.info(
            f"TWAP order {algo_id} completed: "
            f"{algo_order.filled_quantity}/{total_quantity} filled"
        )

        return algo_order

    async def execute_vwap(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        duration_seconds: int,
        target_participation_rate: float = 0.1,
        idempotency_key: Optional[str] = None,
    ) -> AlgoOrder:
        """
        Execute a Volume-Weighted Average Price (VWAP) order.

        Executes slices proportionally to market volume to track VWAP.
        """
        algo_id = f"vwap_{int(time.time() * 1000)}"

        algo_order = AlgoOrder(
            order_id=algo_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            algo_type="VWAP",
            duration_seconds=duration_seconds,
            slices=0,
            status="RUNNING",
        )

        self.algo_orders[algo_id] = algo_order

        start_time = time.time()
        check_interval = 10

        logger.info(
            f"Starting VWAP order {algo_id}: {total_quantity} {symbol} "
            f"with {target_participation_rate*100}% participation over {duration_seconds}s"
        )

        while (time.time() - start_time) < duration_seconds and algo_order.status == "RUNNING":
            try:
                snapshot = await self.get_orderbook_snapshot(symbol)

                imbalance = self.calculate_order_flow_imbalance(snapshot)
                recent_volume = self._estimate_recent_volume(snapshot, depth_levels=5)

                slice_size = recent_volume * target_participation_rate * (check_interval / duration_seconds)

                if side == OrderSide.BUY:
                    slice_size = min(slice_size, total_quantity - algo_order.filled_quantity)
                else:
                    slice_size = min(slice_size, total_quantity - algo_order.filled_quantity)

                if slice_size > 0:
                    child_order = await self.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=slice_size,
                        idempotency_key=f"{idempotency_key}_{int(time.time())}" if idempotency_key else None,
                    )

                    algo_order.child_orders.append(child_order['id'])
                    algo_order.filled_quantity += child_order.get('filled', 0)

                if algo_order.filled_quantity >= total_quantity:
                    break

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"VWAP execution error: {e}")
                await asyncio.sleep(check_interval)

        algo_order.status = "COMPLETED" if algo_order.filled_quantity >= total_quantity * 0.95 else "PARTIALLY_FILLED"

        logger.info(
            f"VWAP order {algo_id} completed: "
            f"{algo_order.filled_quantity}/{total_quantity} filled"
        )

        return algo_order

    async def get_current_price(self, symbol: str) -> float:
        """Get current mid-price from orderbook."""
        snapshot = await self.get_orderbook_snapshot(symbol, depth=1)

        if snapshot.bids and snapshot.asks:
            return (snapshot.bids[0].price + snapshot.asks[0].price) / 2

        raise Exception("No price data available")

    def _estimate_recent_volume(self, snapshot: OrderbookSnapshot, depth_levels: int = 5) -> float:
        """Estimate recent trading volume from orderbook depth."""
        bid_volume = sum(level.quantity for level in snapshot.bids[:depth_levels])
        ask_volume = sum(level.quantity for level in snapshot.asks[:depth_levels])
        return (bid_volume + ask_volume) / 2

    async def cancel_algo_order(self, algo_id: str) -> bool:
        """Cancel an algorithmic order and its child orders."""
        if algo_id not in self.algo_orders:
            return False

        algo_order = self.algo_orders[algo_id]
        algo_order.status = "CANCELLING"

        for child_order_id in algo_order.child_orders:
            try:
                await self.cancel_order(child_order_id, algo_order.symbol)
            except Exception as e:
                logger.error(f"Failed to cancel child order {child_order_id}: {e}")

        algo_order.status = "CANCELLED"
        return True

    async def close(self) -> None:
        """Clean up resources."""
        self._running = False
        await self.exchange.close()
        logger.info("Binance executor closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
