"""
OKX V5 CEX Executor with V5 depth books, funding arbitrage, portfolio margin support.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import ccxt.async_support as ccxt
from loguru import logger
from execution.legacy_cex_common import (
    LegacyCEXExecutorBase,
    OrderSide,
    OrderType,
    OrderbookSnapshot,
    TimeInForce,
)


@dataclass
class FundingRate:
    symbol: str
    funding_rate: float
    funding_time: int
    predicted_rate: float
    mark_price: float


class OKXExecutor(LegacyCEXExecutorBase):
    """
    Production-grade OKX V5 executor with:
    - V5 API support
    - Depth orderbooks
    - Funding rate arbitrage
    - Portfolio margin
    """

    exchange_label = "OKX V5"
    supports_reduce_only = True

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

        self._init_common_state(enable_paper_trading)

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
