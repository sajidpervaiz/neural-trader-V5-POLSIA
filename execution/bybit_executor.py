"""
Bybit V5 CEX Executor with 500ms orderbook snapshots, position mode switching,
and enhanced risk management.
"""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import ccxt.async_support as ccxt
from loguru import logger
from execution.binance_executor import OrderSide, OrderType, TimeInForce, AlgoOrder, OrderbookSnapshot, L2DepthLevel
from core.circuit_breaker import CircuitBreaker, CircuitState
from core.idempotency import IdempotencyManager
from core.retry import RetryPolicy, with_retry


class PositionMode(Enum):
    HEDGE = "Both"
    ONE_WAY = "Merged"


@dataclass
class PositionInfo:
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float
    mode: PositionMode


class BybitExecutor:
    """
    Production-grade Bybit V5 executor with:
    - V5 API support
    - 500ms orderbook snapshots
    - Position mode switching
    - Enhanced risk management
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        enable_paper_trading: bool = True,
    ):
        self.exchange = ccxt.bybit({
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
        self.position_cache: Dict[str, PositionInfo] = {}

        self._running = False
        self._orderbook_task = None

    async def initialize(self) -> None:
        """Initialize executor with connectivity checks."""
        try:
            await self.exchange.load_markets()
            logger.info("Bybit V5 executor initialized successfully")

            if self.paper_trading:
                logger.info("Running in PAPER TRADING mode - no real orders will be placed")

        except Exception as e:
            logger.error(f"Failed to initialize Bybit executor: {e}")
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
        reduce_only: bool = False,
    ) -> Dict:
        """Place an order with idempotency and circuit breaking."""
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
            params = {
                'reduceOnly': reduce_only,
            }

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

    async def get_positions(self) -> List[PositionInfo]:
        """Get all open positions."""
        if self.paper_trading:
            return list(self.position_cache.values())

        try:
            positions_data = await self.exchange.fetch_positions()

            positions = []
            for pos in positions_data:
                if float(pos['contracts']) != 0:
                    position_info = PositionInfo(
                        symbol=pos['symbol'],
                        side=pos['side'],
                        size=float(pos['contracts']),
                        entry_price=float(pos['entryPrice'] or 0),
                        unrealized_pnl=float(pos['unrealizedPnl'] or 0),
                        leverage=int(pos['leverage'] or 1),
                        liquidation_price=float(pos['liquidationPrice'] or 0),
                        mode=PositionMode.HEDGE,
                    )
                    positions.append(position_info)
                    self.position_cache[pos['symbol']] = position_info

            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return list(self.position_cache.values())

    async def set_position_mode(self, mode: PositionMode) -> bool:
        """Set position mode (HEDGE or ONE_WAY)."""
        if self.paper_trading:
            logger.info(f"[PAPER] Setting position mode to {mode.value}")
            return True

        try:
            await self.exchange.set_position_mode(mode.value)
            logger.info(f"Position mode set to {mode.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set position mode: {e}")
            return False

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if self.paper_trading:
            logger.info(f"[PAPER] Setting leverage for {symbol} to {leverage}x")
            return True

        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_current_price(self, symbol: str) -> float:
        """Get current mid-price from orderbook."""
        snapshot = await self.get_orderbook_snapshot(symbol, depth=1)

        if snapshot.bids and snapshot.asks:
            return (snapshot.bids[0].price + snapshot.asks[0].price) / 2

        raise Exception("No price data available")

    async def start_orderbook_stream(
        self,
        symbols: List[str],
        interval_ms: int = 500,
    ) -> None:
        """Start continuous orderbook streaming."""
        self._running = True

        async def _orderbook_loop():
            while self._running:
                for symbol in symbols:
                    try:
                        await self.get_orderbook_snapshot(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching orderbook for {symbol}: {e}")

                await asyncio.sleep(interval_ms / 1000)

        self._orderbook_task = asyncio.create_task(_orderbook_loop())
        logger.info(f"Started orderbook stream for {len(symbols)} symbols")

    async def stop_orderbook_stream(self) -> None:
        """Stop orderbook streaming."""
        self._running = False
        if self._orderbook_task:
            self._orderbook_task.cancel()
            try:
                await self._orderbook_task
            except asyncio.CancelledError:
                logger.debug("Bybit orderbook stream task cancelled")
        logger.info("Stopped orderbook stream")

    async def close(self) -> None:
        """Clean up resources."""
        self._running = False
        await self.stop_orderbook_stream()
        await self.exchange.close()
        logger.info("Bybit executor closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
