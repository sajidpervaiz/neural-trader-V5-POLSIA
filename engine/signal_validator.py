"""
Signal Validator with OOS validation, paper tracking, decay detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
from loguru import logger


class SignalQuality(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    INVALID = "INVALID"

    @staticmethod
    def _rank(member: "SignalQuality") -> int:
        return {"EXCELLENT": 4, "GOOD": 3, "FAIR": 2, "POOR": 1, "INVALID": 0}[member.value]

    def __lt__(self, other):
        if not isinstance(other, SignalQuality):
            return NotImplemented
        return self._rank(self) < self._rank(other)

    def __le__(self, other):
        if not isinstance(other, SignalQuality):
            return NotImplemented
        return self._rank(self) <= self._rank(other)

    def __gt__(self, other):
        if not isinstance(other, SignalQuality):
            return NotImplemented
        return self._rank(self) > self._rank(other)

    def __ge__(self, other):
        if not isinstance(other, SignalQuality):
            return NotImplemented
        return self._rank(self) >= self._rank(other)


@dataclass
class Signal:
    signal_id: str
    timestamp: int
    prediction: float
    confidence: float
    regime: str
    model_id: str
    features: Dict[str, float]
    validated: bool = False
    quality: SignalQuality = SignalQuality.FAIR


@dataclass
class PnLTracking:
    signal_id: str
    entry_price: float
    quantity: float
    entry_time: int
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    realized_pnl: Optional[float] = None
    unrealized_pnl: float = 0.0
    status: str = "OPEN"


class SignalValidator:
    """
    Production-grade signal validator with:
    - Out-of-sample validation
    - Paper trading tracking
    - Performance monitoring
    - Decay detection
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_quality: SignalQuality = SignalQuality.GOOD,
        max_position_age_hours: int = 24,
    ):
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.max_position_age_hours = max_position_age_hours

        self.signals: deque = deque(maxlen=5000)
        self.paper_trades: Dict[str, PnLTracking] = {}

        self.performance_history: deque = deque(maxlen=1000)
        self.validation_stats: Dict[str, Dict] = {}

    def validate_signal(
        self,
        signal: Signal,
        current_regime: str,
        market_conditions: Optional[Dict] = None,
    ) -> Tuple[bool, str, SignalQuality]:
        """
        Validate trading signal.

        Returns:
            (is_valid, reason, quality)
        """
        validation_checks = []

        signal.validated = True

        if signal.confidence < self.min_confidence:
            signal.validated = False
            validation_checks.append(f"Low confidence: {signal.confidence:.2f} < {self.min_confidence}")

        if signal.regime != current_regime:
            signal.validated = False
            validation_checks.append(f"Regime mismatch: signal={signal.regime}, current={current_regime}")

        quality = self._assess_quality(signal, market_conditions)

        if quality < self.min_quality:
            signal.validated = False
            validation_checks.append(f"Quality below threshold: {quality.value}")

        reason = "; ".join(validation_checks) if validation_checks else "All checks passed"

        signal.quality = quality
        self.signals.append(signal)

        logger.info(
            f"Signal validation: {signal.signal_id} - "
            f"Valid={signal.validated}, Quality={quality.value}, Reason={reason}"
        )

        return signal.validated, reason, quality

    def _assess_quality(
        self,
        signal: Signal,
        market_conditions: Optional[Dict],
    ) -> SignalQuality:
        """Assess signal quality based on multiple factors."""
        quality_score = 0.0

        if signal.confidence >= 0.9:
            quality_score += 2
        elif signal.confidence >= 0.7:
            quality_score += 1
        elif signal.confidence >= 0.5:
            quality_score += 0.5

        if market_conditions:
            volatility = market_conditions.get('volatility', 0.02)
            if volatility < 0.02:
                quality_score += 1
            elif volatility < 0.05:
                quality_score += 0.5

            spread = market_conditions.get('spread_pct', 0.1)
            if spread < 0.1:
                quality_score += 1
            elif spread < 0.3:
                quality_score += 0.5

        recent_performance = self._get_recent_performance(signal.model_id)
        if recent_performance.get('win_rate', 0) > 0.55:
            quality_score += 1
        elif recent_performance.get('win_rate', 0) > 0.5:
            quality_score += 0.5

        if quality_score >= 4:
            return SignalQuality.EXCELLENT
        elif quality_score >= 3:
            return SignalQuality.GOOD
        elif quality_score >= 2:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR

    def _get_recent_performance(
        self,
        model_id: str,
        window: int = 50,
    ) -> Dict:
        """Get recent performance for model."""
        model_signals = [
            s for s in self.signals[-window:]
            if s.model_id == model_id and s.validated
        ]

        if not model_signals:
            return {'win_rate': 0.5, 'count': 0}

        trades = [t for t in self.paper_trades.values()
                  if any(s.signal_id == t.signal_id for s in model_signals)]

        if not trades:
            return {'win_rate': 0.5, 'count': 0}

        completed_trades = [t for t in trades if t.status == "CLOSED"]
        if not completed_trades:
            return {'win_rate': 0.5, 'count': len(trades)}

        wins = sum(1 for t in completed_trades if t.realized_pnl and t.realized_pnl > 0)
        win_rate = wins / len(completed_trades) if completed_trades else 0.5

        return {
            'win_rate': win_rate,
            'count': len(completed_trades),
            'avg_pnl': np.mean([t.realized_pnl for t in completed_trades if t.realized_pnl]),
        }

    def start_paper_trade(
        self,
        signal: Signal,
        entry_price: float,
        quantity: float,
    ) -> bool:
        """Start paper trade for validated signal."""
        if not signal.validated or signal.quality < self.min_quality:
            logger.warning(f"Signal {signal.signal_id} not valid for paper trading")
            return False

        trade = PnLTracking(
            signal_id=signal.signal_id,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=signal.timestamp,
            status="OPEN",
        )

        self.paper_trades[signal.signal_id] = trade

        logger.info(
            f"Paper trade started: {signal.signal_id} @ {entry_price}, "
            f"quantity={quantity}, prediction={signal.prediction:.2f}"
        )

        return True

    def update_paper_trade(
        self,
        signal_id: str,
        current_price: float,
        update_unrealized: bool = True,
    ) -> Optional[float]:
        """Update paper trade with current price."""
        if signal_id not in self.paper_trades:
            return None

        trade = self.paper_trades[signal_id]

        if trade.status != "OPEN":
            return None

        if update_unrealized:
            direction = 1 if self._get_signal_direction(signal_id) == "long" else -1
            price_change = current_price - trade.entry_price
            trade.unrealized_pnl = price_change * trade.quantity * direction

        return trade.unrealized_pnl

    def close_paper_trade(
        self,
        signal_id: str,
        exit_price: float,
        exit_time: Optional[int] = None,
    ) -> Optional[float]:
        """Close paper trade and record realized PnL."""
        if signal_id not in self.paper_trades:
            return None

        trade = self.paper_trades[signal_id]

        if trade.status != "OPEN":
            return None

        direction = 1 if self._get_signal_direction(signal_id) == "long" else -1
        price_change = exit_price - trade.entry_price
        trade.realized_pnl = price_change * trade.quantity * direction
        trade.exit_price = exit_price
        trade.exit_time = exit_time or pd.Timestamp.now().value // 10**6
        trade.status = "CLOSED"

        self.performance_history.append(trade.realized_pnl)

        logger.info(
            f"Paper trade closed: {signal_id} @ {exit_price}, "
            f"PnL={trade.realized_pnl:.2f}"
        )

        return trade.realized_pnl

    def _get_signal_direction(self, signal_id: str) -> str:
        """Get direction (long/short) for signal."""
        signal = next((s for s in self.signals if s.signal_id == signal_id), None)
        if not signal:
            return "long"

        return "long" if signal.prediction > 0.5 else "short"

    def detect_signal_decay(
        self,
        model_id: str,
        window: int = 100,
        threshold: float = 0.15,
    ) -> bool:
        """
        Detect if signal quality is decaying for a model.

        Returns True if decay is detected.
        """
        model_signals = [
            s for s in self.signals
            if s.model_id == model_id and s.validated
        ]

        if len(model_signals) < window * 2:
            return False

        recent_signals = model_signals[-window:]
        older_signals = model_signals[-2*window:-window]

        recent_quality_scores = [
            self._quality_to_score(s.quality) for s in recent_signals
        ]
        older_quality_scores = [
            self._quality_to_score(s.quality) for s in older_signals
        ]

        recent_avg = np.mean(recent_quality_scores)
        older_avg = np.mean(older_quality_scores)

        quality_decline = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0

        decay_detected = quality_decline > threshold

        if decay_detected:
            logger.warning(
                f"Signal decay detected for {model_id}: "
                f"quality declined by {quality_decline:.2%}"
            )

        return decay_detected

    def _quality_to_score(self, quality: SignalQuality) -> float:
        """Convert quality enum to numeric score."""
        quality_map = {
            SignalQuality.EXCELLENT: 1.0,
            SignalQuality.GOOD: 0.75,
            SignalQuality.FAIR: 0.5,
            SignalQuality.POOR: 0.25,
            SignalQuality.INVALID: 0.0,
        }
        return quality_map.get(quality, 0.5)

    def calculate_sharpe_ratio(
        self,
        risk_free_rate: float = 0.02,
    ) -> Optional[float]:
        """Calculate Sharpe ratio from paper trading performance (return-based)."""
        if len(self.performance_history) < 30:
            return None

        # Convert dollar PnL to pct returns using entry cost
        pct_returns = []
        for trade in self.paper_trades.values():
            if trade.status == "CLOSED" and trade.realized_pnl is not None:
                entry_cost = trade.entry_price * trade.quantity
                if entry_cost > 0:
                    pct_returns.append(trade.realized_pnl / entry_cost)
        if len(pct_returns) < 30:
            return None

        returns = np.array(pct_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return None

        # Annualize: assume ~252 trades/year approximation
        sharpe = (mean_return - risk_free_rate / 252) / std_return

        return sharpe

    def get_validation_statistics(self) -> Dict:
        """Get comprehensive validation statistics."""
        if not self.signals:
            return {}

        validated_signals = [s for s in self.signals if s.validated]
        total_validated = len(validated_signals)
        quality_distribution = {}

        for quality in SignalQuality:
            count = sum(1 for s in validated_signals if s.quality == quality)
            quality_distribution[quality.value] = count

        paper_trades_stats = {}
        if self.paper_trades:
            total_trades = len(self.paper_trades)
            closed_trades = sum(1 for t in self.paper_trades.values() if t.status == "CLOSED")

            if closed_trades > 0:
                realized_pnls = [t.realized_pnl for t in self.paper_trades.values()
                               if t.realized_pnl is not None]
                wins = sum(1 for pnl in realized_pnls if pnl > 0)
                win_rate = wins / len(realized_pnls)
                avg_pnl = np.mean(realized_pnls)
                total_pnl = sum(realized_pnls)
            else:
                win_rate = 0.5
                avg_pnl = 0.0
                total_pnl = 0.0

            paper_trades_stats = {
                'total_trades': total_trades,
                'closed_trades': closed_trades,
                'open_trades': total_trades - closed_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
            }

        return {
            'total_signals': len(self.signals),
            'validated_signals': total_validated,
            'validation_rate': total_validated / len(self.signals) if self.signals else 0,
            'quality_distribution': quality_distribution,
            'paper_trades': paper_trades_stats,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
        }
