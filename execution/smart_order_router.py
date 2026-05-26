"""
Smart Order Router with venue scoring and liquidity-based routing.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from core.error_handling import sanitize_exception
from execution.binance_executor import BinanceExecutor
from execution.bybit_executor import BybitExecutor
from execution.legacy_cex_common import OrderSide, OrderType
from execution.okx_executor import OKXExecutor


class Venue(Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    HYPERLIQUID = "hyperliquid"


@dataclass
class VenueScore:
    venue: Venue
    score: float
    liquidity: float
    spread: float
    fee: float
    latency_ms: float
    reliability: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class Route:
    venue: Venue
    quantity: float
    expected_fill_price: float
    estimated_cost: float
    execution_time_ms: float
    score: float


@dataclass
class RoutingDecision:
    routes: List[Route]
    total_quantity: float
    expected_avg_price: float
    total_cost: float
    recommended_venue: Venue
    confidence: float


@dataclass
class _RouterDepthLevel:
    price: float
    quantity: float


@dataclass
class _RouterOrderbookSnapshot:
    bids: List[_RouterDepthLevel]
    asks: List[_RouterDepthLevel]


class SmartOrderRouter:
    """
    Production-grade smart order router with:
    - Venue scoring based on multiple factors
    - Liquidity-based routing
    - Fee optimization
    - Latency considerations
    - Reliability tracking
    """

    def __init__(
        self,
        binance_executor: Optional[BinanceExecutor] = None,
        bybit_executor: Optional[BybitExecutor] = None,
        okx_executor: Optional[OKXExecutor] = None,
    ):
        self.executors = {
            Venue.BINANCE: binance_executor,
            Venue.BYBIT: bybit_executor,
            Venue.OKX: okx_executor,
        }

        self.venue_scores: Dict[Venue, VenueScore] = {}
        self.historical_performance: Dict[str, List[float]] = {}

        self.routing_weights = {
            "liquidity": 0.35,
            "spread": 0.25,
            "fee": 0.15,
            "latency": 0.15,
            "reliability": 0.10,
        }

        self.score_refresh_interval_sec = 30
        self._last_score_refresh = 0.0

        self._lock = asyncio.Lock()

        # Live tracking of venue metrics (EWMA-updated from real fills)
        self._observed_fees: Dict[Venue, List[float]] = {v: [] for v in Venue}
        self._observed_latencies: Dict[Venue, List[float]] = {v: [] for v in Venue}
        self._fill_success_count: Dict[Venue, int] = {v: 0 for v in Venue}
        self._fill_total_count: Dict[Venue, int] = {v: 0 for v in Venue}

    async def initialize(self) -> None:
        """Initialize router by fetching initial venue scores."""
        logger.info("Initializing Smart Order Router")

        for venue, executor in self.executors.items():
            if executor:
                score = await self._calculate_venue_score(venue)
                self.venue_scores[venue] = score
                logger.info(f"{venue.value} score: {score.score:.2f}")

    async def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        max_venues: int = 1,
        min_score_threshold: float = 0.5,
    ) -> Optional[RoutingDecision]:
        """
        Route order to optimal venue(s).

        Args:
            symbol: Trading pair symbol
            side: Order side
            quantity: Order quantity
            max_venues: Maximum number of venues to use
            min_score_threshold: Minimum score for venue selection

        Returns:
            Routing decision with venue(s) and execution plan
        """
        # Refresh venue scores OUTSIDE the lock to avoid blocking
        # other coroutines during network I/O (orderbook fetches).
        await self._refresh_scores_if_needed(symbol)

        async with self._lock:
            available_venues = [
                (venue, score)
                for venue, score in self.venue_scores.items()
                if score.liquidity >= quantity
                and score.score >= min_score_threshold
            ]

            if not available_venues:
                logger.warning(f"No suitable venues for {symbol} with quantity {quantity}")
                return None

            available_venues.sort(key=lambda x: x[1].score, reverse=True)

            selected_venues = available_venues[:max_venues]
            routes = []

            remaining_qty = quantity

            for venue, score in selected_venues:
                if remaining_qty <= 0:
                    break

                route_qty = min(remaining_qty, score.liquidity * 0.8)

                expected_price = await self._get_expected_price(venue, symbol, side)

                route = Route(
                    venue=venue,
                    quantity=route_qty,
                    expected_fill_price=expected_price,
                    estimated_cost=self._calculate_estimated_cost(score, route_qty, expected_price),
                    execution_time_ms=score.latency_ms,
                    score=score.score,
                )

                routes.append(route)
                remaining_qty -= route_qty

            if remaining_qty > 0.001:
                logger.warning(
                    f"Could not route full quantity: {remaining_qty:.6f} remaining"
                )

            total_qty = sum(r.quantity for r in routes)
            if total_qty <= 0:
                logger.warning("No routable quantity available for {}", symbol)
                return None

            weighted_price = sum(r.quantity * r.expected_fill_price for r in routes) / total_qty
            total_cost = sum(r.estimated_cost for r in routes)

            return RoutingDecision(
                routes=routes,
                total_quantity=total_qty,
                expected_avg_price=weighted_price,
                total_cost=total_cost,
                recommended_venue=selected_venues[0][0],
                confidence=selected_venues[0][1].score,
            )

    async def _refresh_scores_if_needed(self, symbol: str) -> None:
        now = time.time()
        if self.venue_scores and (now - self._last_score_refresh) < self.score_refresh_interval_sec:
            return

        # Claim the refresh slot first to prevent concurrent refresh storms
        async with self._lock:
            # Double-check after acquiring lock
            if self.venue_scores and (time.time() - self._last_score_refresh) < self.score_refresh_interval_sec:
                return
            self._last_score_refresh = time.time()

        # Fetch scores OUTSIDE the lock to avoid blocking order routing
        refreshed: Dict[Venue, VenueScore] = {}
        for venue, executor in self.executors.items():
            if executor is None:
                continue
            refreshed[venue] = await self._calculate_venue_score(venue, symbol)

        if refreshed:
            async with self._lock:
                self.venue_scores = refreshed

    async def execute_routed_order(
        self,
        decision: RoutingDecision,
        symbol: str,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> List[Dict]:
        """Execute order across selected venues."""
        results = []

        for route in decision.routes:
            executor = self.executors.get(route.venue)
            if not executor:
                logger.warning(f"Executor not available for {route.venue.value}")
                continue

            try:
                result = await executor.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=route.quantity,
                    price=price,
                )

                results.append({
                    "venue": route.venue.value,
                    "route": route,
                    "result": result,
                    "status": "success" if result else "failed",
                })

            except Exception as e:
                logger.error(f"Failed to execute on {route.venue.value}: {e}")
                results.append({
                    "venue": route.venue.value,
                    "route": route,
                    "result": None,
                    "status": "error",
                    "error": sanitize_exception(e),
                })

        return results

    async def _calculate_venue_score(self, venue: Venue, symbol: str = "BTC/USDT") -> VenueScore:
        """Calculate comprehensive venue score."""
        executor = self.executors.get(venue)
        if not executor:
            return VenueScore(
                venue=venue,
                score=0.0,
                liquidity=0.0,
                spread=float('inf'),
                fee=0.0,
                latency_ms=float('inf'),
                reliability=0.0,
            )

        try:
            snapshot = await self._get_orderbook_snapshot(executor, symbol, depth=10)

            liquidity = sum(
                level.quantity
                for level in snapshot.bids[:5] + snapshot.asks[:5]
            )

            if snapshot.bids and snapshot.asks:
                spread = snapshot.asks[0].price - snapshot.bids[0].price
                mid_price = (snapshot.bids[0].price + snapshot.asks[0].price) / 2
                spread_pct = (spread / mid_price) * 100
            else:
                spread = float('inf')
                spread_pct = float('inf')

            fee = self._get_venue_fee(venue)
            latency_ms = self._get_venue_latency(venue)
            reliability = self._get_venue_reliability(venue)

            normalized_liquidity = min(liquidity / 1000, 1.0)
            normalized_spread = max(0, 1 - (spread_pct / 0.1))
            normalized_fee = max(0, 1 - (fee / 0.01))
            normalized_latency = max(0, 1 - (latency_ms / 1000))

            score = (
                self.routing_weights["liquidity"] * normalized_liquidity +
                self.routing_weights["spread"] * normalized_spread +
                self.routing_weights["fee"] * normalized_fee +
                self.routing_weights["latency"] * normalized_latency +
                self.routing_weights["reliability"] * reliability
            )

            return VenueScore(
                venue=venue,
                score=score,
                liquidity=liquidity,
                spread=spread,
                fee=fee,
                latency_ms=latency_ms,
                reliability=reliability,
            )

        except Exception as e:
            logger.error(f"Error calculating score for {venue.value}: {e}")
            return VenueScore(
                venue=venue,
                score=0.0,
                liquidity=0.0,
                spread=float('inf'),
                fee=0.0,
                latency_ms=float('inf'),
                reliability=0.0,
            )

    async def _get_expected_price(
        self,
        venue: Venue,
        symbol: str,
        side: OrderSide,
    ) -> float:
        """Get expected execution price for venue."""
        executor = self.executors.get(venue)
        if not executor:
            return 0.0

        try:
            snapshot = await self._get_orderbook_snapshot(executor, symbol, depth=1)

            if side == OrderSide.BUY:
                return snapshot.asks[0].price if snapshot.asks else 0.0
            else:
                return snapshot.bids[0].price if snapshot.bids else 0.0

        except Exception:
            return 0.0

    async def _get_orderbook_snapshot(
        self, executor: Any, symbol: str, depth: int
    ) -> _RouterOrderbookSnapshot:
        """Fetch an orderbook snapshot from either legacy or runtime CEX executors."""
        if hasattr(executor, "get_orderbook_snapshot"):
            snapshot = await executor.get_orderbook_snapshot(symbol, depth=depth)
            return _RouterOrderbookSnapshot(
                bids=[_RouterDepthLevel(float(level.price), float(level.quantity)) for level in snapshot.bids],
                asks=[_RouterDepthLevel(float(level.price), float(level.quantity)) for level in snapshot.asks],
            )

        client = getattr(executor, "_client", None)
        if client is None and hasattr(executor, "_init_client"):
            await executor._init_client()
            client = getattr(executor, "_client", None)

        if client is None or not hasattr(client, "fetch_order_book"):
            raise RuntimeError("executor has no orderbook-capable client")

        orderbook = await client.fetch_order_book(symbol, limit=depth)
        bids = [
            _RouterDepthLevel(price=float(level[0]), quantity=float(level[1]))
            for level in orderbook.get("bids", [])[:depth]
            if len(level) >= 2
        ]
        asks = [
            _RouterDepthLevel(price=float(level[0]), quantity=float(level[1]))
            for level in orderbook.get("asks", [])[:depth]
            if len(level) >= 2
        ]
        return _RouterOrderbookSnapshot(bids=bids, asks=asks)

    def _calculate_estimated_cost(self, score: VenueScore, quantity: float, expected_price: float) -> float:
        """Calculate estimated execution cost including fees and slippage."""
        if quantity <= 0:
            return 0.0

        if expected_price <= 0:
            expected_price = 0.0

        slippage_cost = (score.spread / 2) * quantity
        fee_cost = expected_price * quantity * score.fee

        return slippage_cost + fee_cost

    def _get_venue_fee(self, venue: Venue) -> float:
        """Get venue's trading fee — uses observed average if available."""
        if self._observed_fees.get(venue):
            return sum(self._observed_fees[venue]) / len(self._observed_fees[venue])
        fee_map = {
            Venue.BINANCE: 0.001,
            Venue.BYBIT: 0.001,
            Venue.OKX: 0.0008,
            Venue.HYPERLIQUID: 0.00025,
        }
        return fee_map.get(venue, 0.001)

    def _get_venue_latency(self, venue: Venue) -> float:
        """Get venue's average latency in ms — uses observed average if available."""
        if self._observed_latencies.get(venue):
            return sum(self._observed_latencies[venue]) / len(self._observed_latencies[venue])
        latency_map = {
            Venue.BINANCE: 50,
            Venue.BYBIT: 45,
            Venue.OKX: 60,
            Venue.HYPERLIQUID: 100,
        }
        return latency_map.get(venue, 100)

    def _get_venue_reliability(self, venue: Venue) -> float:
        """Get venue's reliability score (0-1) — uses observed fill rate if available."""
        total = self._fill_total_count.get(venue, 0)
        if total >= 10:  # Need minimum sample before trusting observed data
            return self._fill_success_count.get(venue, 0) / total
        reliability_map = {
            Venue.BINANCE: 0.99,
            Venue.BYBIT: 0.97,
            Venue.OKX: 0.96,
            Venue.HYPERLIQUID: 0.90,
        }
        return reliability_map.get(venue, 0.95)

    def record_fill_metrics(
        self,
        venue: Venue,
        fee_rate: float,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record observed fill metrics for live tracking. Call after each order execution."""
        max_history = 200
        fees = self._observed_fees.setdefault(venue, [])
        fees.append(fee_rate)
        if len(fees) > max_history:
            self._observed_fees[venue] = fees[-max_history:]

        lats = self._observed_latencies.setdefault(venue, [])
        lats.append(latency_ms)
        if len(lats) > max_history:
            self._observed_latencies[venue] = lats[-max_history:]

        self._fill_total_count[venue] = self._fill_total_count.get(venue, 0) + 1
        if success:
            self._fill_success_count[venue] = self._fill_success_count.get(venue, 0) + 1

    def update_routing_weights(self, weights: Dict[str, float]) -> None:
        """Update routing weight configuration."""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.routing_weights = weights
        logger.info(f"Routing weights updated: {weights}")

    def get_venue_scores(self) -> Dict[Venue, VenueScore]:
        """Get current venue scores."""
        return self.venue_scores.copy()
