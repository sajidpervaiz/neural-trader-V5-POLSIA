"""
Bybit V5 CEX Executor with 500ms orderbook snapshots, position mode switching,
and enhanced risk management.
"""

import asyncio
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import ccxt.async_support as ccxt
from loguru import logger
from execution.legacy_cex_common import (
    AlgoOrder,
    LegacyCEXExecutorBase,
    OrderSide,
    OrderType,
    OrderbookSnapshot,
    TimeInForce,
)


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


class BybitExecutor(LegacyCEXExecutorBase):
    """
    Production-grade Bybit V5 executor with:
    - V5 API support
    - 500ms orderbook snapshots
    - Position mode switching
    - Enhanced risk management
    """

    exchange_label = "Bybit V5"
    supports_reduce_only = True

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

        self._init_common_state(enable_paper_trading)
        self.position_cache: dict[str, PositionInfo] = {}
        self._orderbook_task = None

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
