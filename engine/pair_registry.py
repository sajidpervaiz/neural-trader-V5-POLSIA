"""
ARMS-V2.1 Pair Tier Registry — 4-tier classification with dynamic filtering.

Tier 1: > $200M daily volume (full allocation)
Tier 2: $75M–$200M (reduced sizing)
Tier 3: $30M–$75M (minimum sizing)
Tier 4: < $30M (excluded)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from loguru import logger


class PairTier(IntEnum):
    TIER_1 = 1  # > $200M — full allocation
    TIER_2 = 2  # $75M–$200M — reduced
    TIER_3 = 3  # $30M–$75M — minimum
    TIER_4 = 4  # < $30M — excluded


@dataclass
class PairInfo:
    symbol: str
    tier: PairTier
    daily_volume_usd: float = 0.0
    avg_spread_bps: float = 0.0
    wick_anomaly_score: float = 0.0
    orderbook_depth_usd: float = 0.0
    last_updated: float = 0.0
    manual_override: PairTier | None = None
    demoted_from: PairTier | None = None
    promoted_from: PairTier | None = None

    @property
    def effective_tier(self) -> PairTier:
        return self.manual_override if self.manual_override is not None else self.tier

    @property
    def is_tradeable(self) -> bool:
        return self.effective_tier != PairTier.TIER_4


# Tier sizing multipliers
TIER_SIZE_MULTIPLIER: dict[PairTier, float] = {
    PairTier.TIER_1: 1.0,
    PairTier.TIER_2: 0.6,
    PairTier.TIER_3: 0.3,
    PairTier.TIER_4: 0.0,
}


class PairRegistry:
    """4-tier pair classification with dynamic filtering on a 4H cycle."""

    VOLUME_THRESHOLDS: dict[PairTier, float] = {
        PairTier.TIER_1: 200_000_000,
        PairTier.TIER_2: 75_000_000,
        PairTier.TIER_3: 30_000_000,
    }

    def __init__(
        self,
        max_spread_bps: float = 15.0,
        max_wick_anomaly: float = 3.0,
        min_depth_usd: float = 50_000.0,
        filter_interval_seconds: float = 14400,  # 4H
    ) -> None:
        self._pairs: dict[str, PairInfo] = {}
        self._max_spread = max_spread_bps
        self._max_wick = max_wick_anomaly
        self._min_depth = min_depth_usd
        self._filter_interval = filter_interval_seconds
        self._last_filter_time = 0.0

    def register(self, symbol: str, daily_volume_usd: float = 0.0) -> PairInfo:
        tier = self._classify_tier(daily_volume_usd)
        info = PairInfo(
            symbol=symbol,
            tier=tier,
            daily_volume_usd=daily_volume_usd,
            last_updated=time.time(),
        )
        self._pairs[symbol] = info
        return info

    def get(self, symbol: str) -> PairInfo | None:
        return self._pairs.get(symbol)

    def get_tier(self, symbol: str) -> PairTier:
        info = self._pairs.get(symbol)
        if info is None:
            return PairTier.TIER_4
        return info.effective_tier

    def get_size_multiplier(self, symbol: str) -> float:
        return TIER_SIZE_MULTIPLIER.get(self.get_tier(symbol), 0.0)

    def is_tradeable(self, symbol: str) -> bool:
        tier = self.get_tier(symbol)
        return tier != PairTier.TIER_4

    def _classify_tier(self, volume_usd: float) -> PairTier:
        if volume_usd >= self.VOLUME_THRESHOLDS[PairTier.TIER_1]:
            return PairTier.TIER_1
        if volume_usd >= self.VOLUME_THRESHOLDS[PairTier.TIER_2]:
            return PairTier.TIER_2
        if volume_usd >= self.VOLUME_THRESHOLDS[PairTier.TIER_3]:
            return PairTier.TIER_3
        return PairTier.TIER_4

    def update_metrics(
        self,
        symbol: str,
        daily_volume_usd: float | None = None,
        avg_spread_bps: float | None = None,
        wick_anomaly_score: float | None = None,
        orderbook_depth_usd: float | None = None,
    ) -> PairInfo | None:
        info = self._pairs.get(symbol)
        if info is None:
            return None

        if daily_volume_usd is not None:
            info.daily_volume_usd = daily_volume_usd
        if avg_spread_bps is not None:
            info.avg_spread_bps = avg_spread_bps
        if wick_anomaly_score is not None:
            info.wick_anomaly_score = wick_anomaly_score
        if orderbook_depth_usd is not None:
            info.orderbook_depth_usd = orderbook_depth_usd
        info.last_updated = time.time()
        return info

    def run_dynamic_filter(self) -> dict[str, str]:
        """Run 4H dynamic filter cycle. Returns dict of symbol → action taken."""
        now = time.time()
        if now - self._last_filter_time < self._filter_interval:
            return {}

        self._last_filter_time = now
        actions: dict[str, str] = {}

        for symbol, info in self._pairs.items():
            if info.manual_override is not None:
                continue

            old_tier = info.tier
            new_tier = self._classify_tier(info.daily_volume_usd)

            # Dynamic demotion: bad spread, wick anomalies, low depth
            demote = False
            reasons = []
            if info.avg_spread_bps > self._max_spread:
                demote = True
                reasons.append(f"spread={info.avg_spread_bps:.1f}bps")
            if info.wick_anomaly_score > self._max_wick:
                demote = True
                reasons.append(f"wick={info.wick_anomaly_score:.1f}")
            if info.orderbook_depth_usd > 0 and info.orderbook_depth_usd < self._min_depth:
                demote = True
                reasons.append(f"depth=${info.orderbook_depth_usd:.0f}")

            if demote and new_tier.value < 4:
                new_tier = PairTier(min(new_tier.value + 1, 4))
                info.demoted_from = old_tier
                actions[symbol] = f"demoted:{','.join(reasons)}"
            elif not demote and old_tier.value > new_tier.value:
                info.promoted_from = old_tier
                actions[symbol] = "promoted:volume"

            if new_tier != old_tier:
                info.tier = new_tier
                logger.info("Pair {} tier change: {} → {} [{}]",
                            symbol, old_tier.name, new_tier.name,
                            actions.get(symbol, "reclassified"))

        return actions

    def set_override(self, symbol: str, tier: PairTier | None) -> None:
        info = self._pairs.get(symbol)
        if info is not None:
            info.manual_override = tier

    @property
    def tradeable_pairs(self) -> list[str]:
        return [s for s, i in self._pairs.items() if i.is_tradeable]

    def get_all(self) -> dict[str, PairInfo]:
        return dict(self._pairs)

    def get_snapshot(self) -> dict[str, Any]:
        return {
            symbol: {
                "tier": info.effective_tier.name,
                "volume_usd": info.daily_volume_usd,
                "spread_bps": info.avg_spread_bps,
                "depth_usd": info.orderbook_depth_usd,
                "tradeable": info.is_tradeable,
                "size_mult": TIER_SIZE_MULTIPLIER[info.effective_tier],
            }
            for symbol, info in self._pairs.items()
        }
