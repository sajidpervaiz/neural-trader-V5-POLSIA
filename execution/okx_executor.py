"""
OKX V5 CEX Executor with V5 depth books, funding arbitrage, portfolio margin support.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import ccxt.async_support as ccxt
from loguru import logger
from execution.binance_executor import OrderSide, OrderType, TimeInForce, OrderbookSnapshot, L2DepthLevel
from core.circuit_breaker import CircuitBreaker, CircuitState
from core.idempotency import IdempotencyManager
from core.retry import RetryPolicy, with_retry


@dataclass
class FundingRate:
    symbol: str
    funding_rate: float
    funding_time: int
    predicted_rate: float
    mark_price: float


class OKXExecutor:
    """
    Production-grade OKX V5 executor with:
    - V5 API support
    - Depth orderbooks
    - Funding rate arbitrage
    - Portfolio margin
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        testnet: bool = False,
        enable_paper_trading: bool = True,
    ):
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
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
        self.balance_cache: Dict[str, float] = {}
        self.position_cache: Dict[str, Dict] = {}

        self._running = False

    async def initialize(self) -> None:
        """Initialize executor with connectivity checks."""
        try:
            await self.exchange.load_markets()
            logger.info("OKX V5 executor initialized successfully")

            if self.paper_trading:
                logger.info("Running in PAPER TRADING mode - no real orders will be placed")

        except Exception as e:
            logger.error(f"Failed to initialize OKX executor: {e}")
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

    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Get current and predicted funding rate."""
        try:
            funding_data = await self.exchange.fetch_funding_rate(symbol)

            return FundingRate(
                symbol=symbol,
                funding_rate=float(funding_data.get('fundingRate', 0)),
                funding_time=int(funding_data.get('fundingTimestamp', 0)),
                predicted_rate=float(funding_data.get('predictedFundingRate', 0)),
                mark_price=float(funding_data.get('markPrice', 0)),
            )

        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def calculate_funding_arbitrage(
        self,
        symbol: str,
        funding_threshold: float = 0.0001,
    ) -> Optional[Dict]:
        """
        Calculate funding arbitrage opportunity.

        Returns arbitrage strategy if funding rate exceeds threshold.
        """
        funding = await self.get_funding_rate(symbol)

        if not funding:
            return None

        if abs(funding.funding_rate) > funding_threshold:
            # positive funding_rate → longs pay → go short to receive funding
            # negative funding_rate → shorts pay → go long to receive funding
            direction = "short" if funding.funding_rate > 0 else "long"

            return {
                "symbol": symbol,
                "direction": direction,
                "funding_rate": funding.funding_rate,
                "predicted_rate": funding.predicted_rate,
                "mark_price": funding.mark_price,
                "opportunity_score": abs(funding.funding_rate) / funding_threshold,
            }

        return None

    async def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if self.paper_trading:
            return list(self.position_cache.values())

        try:
            positions_data = await self.exchange.fetch_positions()

            positions = []
            for pos in positions_data:
                if float(pos['contracts']) != 0:
                    positions.append({
                        "symbol": pos['symbol'],
                        "side": pos['side'],
                        "size": float(pos['contracts']),
                        "entry_price": float(pos['entryPrice'] or 0),
                        "unrealized_pnl": float(pos['unrealizedPnl'] or 0),
                        "leverage": int(pos['leverage'] or 1),
                    })

            self.position_cache = {p["symbol"]: p for p in positions}

            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return list(self.position_cache.values())

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

    async def close(self) -> None:
        """Clean up resources."""
        self._running = False
        await self.exchange.close()
        logger.info("OKX executor closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
