"""
Binance CEX Executor with L2 Orderbook Reconstruction, TWAP/VWAP Algorithms,
Imbalance Detection, and Advanced Order Management.
"""

import asyncio
import time
from typing import Optional
import ccxt.async_support as ccxt
from loguru import logger
from execution.legacy_cex_common import (
    AlgoOrder,
    ImbalanceSignal,
    L2DepthLevel,
    LegacyCEXExecutorBase,
    OrderSide,
    OrderType,
    OrderbookSnapshot,
    TimeInForce,
)


class BinanceExecutor(LegacyCEXExecutorBase):
    """
    Production-grade Binance executor with advanced features:
    - L2 orderbook reconstruction
    - TWAP/VWAP execution algorithms
    - Order flow imbalance detection
    - Smart order routing
    - Idempotency and circuit breaking
    """

    exchange_label = "Binance"

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
            # CCXT removed set_sandbox_mode for binance futures (announcement #92).
            # Override fapi* URLs directly so REST hits testnet.binancefuture.com.
            try:
                api_urls = self.exchange.urls.get('api') or {}
                for k, v in list(api_urls.items()):
                    if isinstance(v, str) and 'fapi.binance.com' in v:
                        api_urls[k] = v.replace('https://fapi.binance.com', 'https://testnet.binancefuture.com')
                self.exchange.urls['api'] = api_urls
            except Exception:
                # legacy CCXT path — fall through to set_sandbox_mode if URL override fails
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception:
                    pass

        self._init_common_state(enable_paper_trading)

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
