"""
Macro Aggregator with unified macro publisher and regime classification.
"""

import asyncio
import inspect
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger

from data_ingestion.fed_calendar import FedCalendar, FedEventType
from data_ingestion.economic_releases import EconomicReleases


class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    TRANSITION = "TRANSITION"


@dataclass
class MacroSignal:
    signal_id: str
    timestamp: datetime
    source: str
    event_type: str
    importance: int
    bias: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    metadata: Dict


@dataclass
class RegimeClassification:
    regime: MarketRegime
    confidence: float
    transition_probability: float
    key_drivers: List[str]
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"


class MacroAggregator:
    """
    Macro aggregator with:
    - Unified macro publisher
    - Regime classification
    - Signal generation
    - Risk assessment
    """

    def __init__(self, enable_paper_mode: bool = True):
        self.paper_mode = enable_paper_mode

        self.fed_calendar = FedCalendar(enable_paper_mode=enable_paper_mode)
        self.economic_releases = EconomicReleases(enable_paper_mode=enable_paper_mode)

        self.signals: deque = deque(maxlen=1000)
        self.current_regime: Optional[RegimeClassification] = None

        self._subscribers: List[callable] = []

    async def initialize(self) -> None:
        """Initialize macro aggregator."""
        await self.fed_calendar.initialize()
        await self.economic_releases.initialize()

        self._classify_regime()
        logger.info("Macro Aggregator initialized")

    def subscribe(self, callback: callable) -> None:
        """Subscribe to macro signals."""
        self._subscribers.append(callback)

    async def publish_signal(self, signal: MacroSignal) -> None:
        """Publish macro signal to subscribers."""
        self.signals.append(signal)

        for callback in self._subscribers:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    async def process_fed_events(self) -> List[MacroSignal]:
        """Process upcoming Fed events into signals."""
        upcoming_events = self.fed_calendar.get_upcoming_events(days=7, min_importance=3)
        signals = []

        for event in upcoming_events:
            signal_id = f"fed_{event.event_id}_{int(datetime.now().timestamp())}"

            bias = "neutral"
            if event.event_type == FedEventType.FOMC_DECISION:
                bias = "bearish" if event.importance >= 5 else "neutral"

            signal = MacroSignal(
                signal_id=signal_id,
                timestamp=datetime.now(),
                source="fed_calendar",
                event_type=event.event_type.value,
                importance=event.importance,
                bias=bias,
                confidence=0.8,
                metadata={
                    "title": event.title,
                    "description": event.description,
                    "date": event.date.isoformat(),
                    "speaker": event.speaker,
                    "location": event.location,
                },
            )

            signals.append(signal)
            await self.publish_signal(signal)

        return signals

    async def process_economic_releases(self) -> List[MacroSignal]:
        """Process economic releases into signals."""
        upcoming = self.economic_releases.get_upcoming_releases(days=7, min_importance=3)
        signals = []

        for release in upcoming:
            signal_id = f"eco_{release.release_id}_{int(datetime.now().timestamp())}"

            deviation_impact = 0.0
            if release.deviation_pct:
                deviation_impact = min(abs(release.deviation_pct) / 20, 1.0)

            bias = "neutral"
            if release.actual and release.consensus:
                if release.actual > release.consensus:
                    bias = "bullish"
                elif release.actual < release.consensus:
                    bias = "bearish"

            signal = MacroSignal(
                signal_id=signal_id,
                timestamp=datetime.now(),
                source="economic_releases",
                event_type=release.indicator_type.value,
                importance=release.importance,
                bias=bias,
                confidence=0.7 + (deviation_impact * 0.3),
                metadata={
                    "title": release.title,
                    "actual": release.actual,
                    "consensus": release.consensus,
                    "deviation": release.deviation,
                    "deviation_pct": release.deviation_pct,
                },
            )

            signals.append(signal)
            await self.publish_signal(signal)

        return signals

    def _classify_regime(self) -> RegimeClassification:
        """Classify current market regime."""
        recent_signals = self.signals[-100:] if len(self.signals) >= 100 else self.signals

        bullish_count = sum(1 for s in recent_signals if s.bias == "bullish")
        bearish_count = sum(1 for s in recent_signals if s.bias == "bearish")
        total_signals = len(recent_signals)

        if total_signals == 0:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
        else:
            bullish_ratio = bullish_count / total_signals
            bearish_ratio = bearish_count / total_signals

            avg_importance = sum(s.importance for s in recent_signals) / total_signals if total_signals > 0 else 0

            if bullish_ratio > 0.6 and avg_importance >= 4:
                regime = MarketRegime.BULL
                confidence = min(bullish_ratio, 0.9)
            elif bearish_ratio > 0.6 and avg_importance >= 4:
                regime = MarketRegime.BEAR
                confidence = min(bearish_ratio, 0.9)
            elif abs(bullish_ratio - bearish_ratio) < 0.2:
                regime = MarketRegime.SIDEWAYS
                confidence = 1.0 - abs(bullish_ratio - bearish_ratio)
            else:
                regime = MarketRegime.TRANSITION
                confidence = 0.6

        high_importance_count = sum(1 for s in recent_signals if s.importance >= 5)
        vol_signal = high_importance_count / total_signals if total_signals > 0 else 0

        if vol_signal > 0.3:
            regime = MarketRegime.VOLATILE

        key_drivers = [
            s.event_type
            for s in recent_signals
            if s.importance >= 4
        ][:5]

        risk_level = self._calculate_risk_level(recent_signals)

        self.current_regime = RegimeClassification(
            regime=regime,
            confidence=confidence,
            transition_probability=1.0 - confidence,
            key_drivers=key_drivers,
            risk_level=risk_level,
        )

        logger.info(f"Regime classified: {regime.value} (confidence: {confidence:.2f})")

        return self.current_regime

    def _calculate_risk_level(self, signals: List[MacroSignal]) -> str:
        """Calculate overall risk level from signals."""
        if not signals:
            return "LOW"

        high_impact_count = sum(1 for s in signals if s.importance >= 5)
        total_count = len(signals)

        high_ratio = high_impact_count / total_count

        if high_ratio > 0.4:
            return "EXTREME"
        elif high_ratio > 0.25:
            return "HIGH"
        elif high_ratio > 0.1:
            return "MEDIUM"
        else:
            return "LOW"

    def get_current_regime(self) -> Optional[RegimeClassification]:
        """Get current regime classification."""
        return self.current_regime

    def get_recent_signals(self, hours: int = 24) -> List[MacroSignal]:
        """Get recent macro signals."""
        cutoff = datetime.now().timestamp() - (hours * 3600)

        return [
            signal
            for signal in self.signals
            if signal.timestamp.timestamp() >= cutoff
        ]

    def get_risk_adjusted_position_size(
        self,
        base_size: float,
        risk_tolerance: float = 1.0,
    ) -> float:
        """
        Calculate risk-adjusted position size based on regime.

        Args:
            base_size: Base position size
            risk_tolerance: Risk tolerance factor (0.0-1.0)

        Returns:
            Adjusted position size
        """
        if not self.current_regime:
            return base_size

        risk_multipliers = {
            "LOW": 1.0,
            "MEDIUM": 0.8,
            "HIGH": 0.5,
            "EXTREME": 0.25,
        }

        risk_mult = risk_multipliers.get(self.current_regime.risk_level, 0.8)
        regime_mult = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: 0.6,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.TRANSITION: 0.8,
        }.get(self.current_regime.regime, 1.0)

        adjusted_size = base_size * risk_mult * regime_mult * risk_tolerance * self.current_regime.confidence

        return max(adjusted_size, 0.1)

    async def close(self) -> None:
        """Clean up resources."""
        await self.fed_calendar.close()
        await self.economic_releases.close()
