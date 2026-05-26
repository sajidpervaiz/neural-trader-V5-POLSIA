"""
ARMS-V2.1 Order Splitting — TWAP and Iceberg execution (§13).

Splits large orders into smaller child orders to reduce market impact.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger


@dataclass
class ChildOrder:
    """Individual child order within a split execution."""
    parent_id: str
    index: int
    size: float
    price: float | None  # None = market order
    status: str = "pending"  # pending | filled | failed
    fill_price: float = 0.0
    fill_time: float = 0.0


@dataclass
class SplitOrder:
    """Parent order that manages child orders."""
    order_id: str
    symbol: str
    direction: str
    total_size: float
    strategy: str  # "twap" | "iceberg"
    children: list[ChildOrder] = field(default_factory=list)
    status: str = "active"
    created_at: float = field(default_factory=time.time)

    @property
    def filled_size(self) -> float:
        return sum(c.size for c in self.children if c.status == "filled")

    @property
    def remaining_size(self) -> float:
        return self.total_size - self.filled_size

    @property
    def avg_fill_price(self) -> float:
        filled = [c for c in self.children if c.status == "filled"]
        if not filled:
            return 0.0
        total_notional = sum(c.size * c.fill_price for c in filled)
        total_size = sum(c.size for c in filled)
        return total_notional / total_size if total_size > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        return all(c.status in ("filled", "failed") for c in self.children)


class TWAPExecutor:
    """Time-Weighted Average Price — splits order into equal slices over a duration.

    Parameters:
        num_slices: Number of child orders (default: 5)
        duration_seconds: Total execution window (default: 300s = 5 minutes)
        size_threshold_usd: Minimum order size to trigger TWAP (default: $50,000)
    """

    def __init__(
        self,
        num_slices: int = 5,
        duration_seconds: float = 300.0,
        size_threshold_usd: float = 50_000.0,
    ) -> None:
        self._num_slices = max(2, num_slices)
        self._duration = duration_seconds
        self._threshold = size_threshold_usd
        self._active_orders: dict[str, SplitOrder] = {}

    def should_split(self, size_usd: float) -> bool:
        """Returns True if order size exceeds TWAP threshold."""
        return size_usd >= self._threshold

    def create_split(
        self, order_id: str, symbol: str, direction: str, total_size: float,
    ) -> SplitOrder:
        """Create a TWAP split plan."""
        slice_size = total_size / self._num_slices
        children: list[ChildOrder] = []
        for i in range(self._num_slices):
            children.append(ChildOrder(
                parent_id=order_id,
                index=i,
                size=slice_size,
                price=None,  # Market orders
            ))

        split = SplitOrder(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_size=total_size,
            strategy="twap",
            children=children,
        )
        self._active_orders[order_id] = split
        logger.info(
            "TWAP split created: {} {} {} total={:.4f} slices={} interval={:.0f}s",
            order_id, symbol, direction, total_size, self._num_slices,
            self._duration / self._num_slices,
        )
        return split

    async def execute(
        self,
        split: SplitOrder,
        submit_fn: Callable[[str, str, float], Coroutine[Any, Any, float]],
    ) -> SplitOrder:
        """Execute TWAP slices with timed intervals.

        submit_fn(symbol, direction, size) → fill_price
        """
        interval = self._duration / self._num_slices

        for child in split.children:
            try:
                fill_price = await submit_fn(split.symbol, split.direction, child.size)
                child.status = "filled"
                child.fill_price = fill_price
                child.fill_time = time.time()
                logger.debug(
                    "TWAP slice {}/{} filled: {:.4f} @ {:.2f}",
                    child.index + 1, self._num_slices, child.size, fill_price,
                )
            except Exception as e:
                child.status = "failed"
                logger.error("TWAP slice {}/{} failed: {}", child.index + 1, self._num_slices, e)

            # Wait before next slice (except after the last one)
            if child.index < self._num_slices - 1:
                await asyncio.sleep(interval)

        split.status = "complete" if split.is_complete else "partial"
        self._active_orders.pop(split.order_id, None)
        logger.info(
            "TWAP {} complete: filled={:.4f}/{:.4f} avg_price={:.2f}",
            split.order_id, split.filled_size, split.total_size, split.avg_fill_price,
        )
        return split

    def get_active_orders(self) -> dict[str, SplitOrder]:
        return dict(self._active_orders)


class IcebergExecutor:
    """Iceberg order — shows only a small visible portion to the market.

    Parameters:
        visible_pct: Percentage of total order shown (default: 20%)
        refill_threshold_pct: Refill when visible portion is this % filled (default: 80%)
        size_threshold_usd: Minimum size to trigger iceberg (default: $100,000)
    """

    def __init__(
        self,
        visible_pct: float = 0.20,
        refill_threshold_pct: float = 0.80,
        size_threshold_usd: float = 100_000.0,
    ) -> None:
        self._visible_pct = visible_pct
        self._refill_threshold = refill_threshold_pct
        self._threshold = size_threshold_usd
        self._active_orders: dict[str, SplitOrder] = {}

    def should_split(self, size_usd: float) -> bool:
        return size_usd >= self._threshold

    def get_active_orders(self) -> dict[str, SplitOrder]:
        return dict(self._active_orders)

    def create_split(
        self, order_id: str, symbol: str, direction: str, total_size: float,
    ) -> SplitOrder:
        """Create an iceberg split plan (multiple visible slices)."""
        visible_size = total_size * self._visible_pct
        num_slices = max(2, int(1.0 / self._visible_pct))
        children: list[ChildOrder] = []
        remaining = total_size
        for i in range(num_slices):
            slice_size = min(visible_size, remaining)
            if slice_size <= 0:
                break
            children.append(ChildOrder(
                parent_id=order_id,
                index=i,
                size=slice_size,
                price=None,
            ))
            remaining -= slice_size

        split = SplitOrder(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_size=total_size,
            strategy="iceberg",
            children=children,
        )
        self._active_orders[order_id] = split
        logger.info(
            "Iceberg split created: {} {} {} total={:.4f} visible={:.4f} slices={}",
            order_id, symbol, direction, total_size, visible_size, len(children),
        )
        return split

    async def execute(
        self,
        split: SplitOrder,
        submit_fn: Callable[[str, str, float], Coroutine[Any, Any, float]],
    ) -> SplitOrder:
        """Execute iceberg slices sequentially (refill after each fill)."""
        for child in split.children:
            try:
                fill_price = await submit_fn(split.symbol, split.direction, child.size)
                child.status = "filled"
                child.fill_price = fill_price
                child.fill_time = time.time()
                logger.debug(
                    "Iceberg slice {}/{} filled: {:.4f} @ {:.2f}",
                    child.index + 1, len(split.children), child.size, fill_price,
                )
            except Exception as e:
                child.status = "failed"
                logger.error("Iceberg slice {}/{} failed: {}", child.index + 1, len(split.children), e)
                break  # Stop on failure for iceberg

        split.status = "complete" if split.is_complete else "partial"
        self._active_orders.pop(split.order_id, None)
        return split
