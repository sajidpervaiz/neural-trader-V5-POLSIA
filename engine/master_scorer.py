"""
Master Signal Scoring Engine — V6 Specification §11

TotalScore = 0.25×L3_score + 0.20×L4_score + 0.20×L5_score
           + 0.20×NeuralConfidence + 0.15×ExecLiquidityScore

Where:
  L3_score          : SMC Confluence soft score (0-100)
  L4_score          : Momentum Matrix soft score (0-100)
  L5_score          : Volume Confirmation soft score (0-100)
  NeuralConfidence  : ML model confidence (0-100), 0 when drift detected
  ExecLiquidityScore: 100 if OB depth ≥ 5× position, 50 if ≥ 2×, 0 if < 2×

Trade Decision:
  TotalScore ≥ 90  → "boost"   (25% size increase, subject to risk limits)
  TotalScore ≥ 75  → "trade"
  TotalScore < 75  → "abstain"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TradeDecision = Literal["trade", "boost", "abstain"]


@dataclass
class MasterScoreResult:
    total_score: float
    decision: TradeDecision
    size_multiplier: float          # 1.0 normal, 1.25 on boost
    size_boost: bool                # True when decision == "boost" (score ≥ 90)
    component_scores: dict[str, float]
    explanation: str


def compute_exec_liquidity_score(
    orderbook_depth_usd: float,
    position_size_usd: float,
) -> float:
    """Execution liquidity score per spec §11.

    100  if OB depth ≥ 5× position size
    50   if OB depth ≥ 2× position size
    0    if OB depth < 2× position size
    """
    if position_size_usd <= 0:
        return 100.0   # Unknown — assume liquid
    ratio = orderbook_depth_usd / position_size_usd
    if ratio >= 5.0:
        return 100.0
    if ratio >= 2.0:
        return 50.0
    return 0.0


def compute_master_score(
    layer3_score: float,
    layer4_score: float,
    layer5_score: float,
    neural_confidence: float,
    exec_liquidity_score: float,
    drift_detected: bool = False,
) -> MasterScoreResult:
    """Compute the master signal score and trade decision.

    Args:
        layer3_score      : SMC Confluence score 0-100
        layer4_score      : Momentum Matrix score 0-100
        layer5_score      : Volume Confirmation score 0-100
        neural_confidence : ML confidence 0-100 (ignored when drift detected)
        exec_liquidity_score: Execution liquidity score (0/50/100)
        drift_detected    : If True, neural component set to 0

    Returns MasterScoreResult with total_score, decision, size_multiplier.
    """
    # When drift is detected, neural confidence is zeroed (spec §8.5)
    effective_neural = 0.0 if drift_detected else float(neural_confidence)

    components = {
        "layer3_smc":          round(float(layer3_score), 2),
        "layer4_momentum":     round(float(layer4_score), 2),
        "layer5_volume":       round(float(layer5_score), 2),
        "neural_confidence":   round(effective_neural, 2),
        "exec_liquidity":      round(float(exec_liquidity_score), 2),
        "drift_zeroed":        float(drift_detected),
    }

    total = (
        0.25 * layer3_score +
        0.20 * layer4_score +
        0.20 * layer5_score +
        0.20 * effective_neural +
        0.15 * exec_liquidity_score
    )
    total = round(min(100.0, max(0.0, total)), 2)

    if total >= 90.0:
        decision: TradeDecision = "boost"
        size_mult = 1.25
    elif total >= 75.0:
        decision = "trade"
        size_mult = 1.0
    else:
        decision = "abstain"
        size_mult = 0.0

    parts = [
        f"L3={layer3_score:.0f}×0.25={layer3_score*0.25:.1f}",
        f"L4={layer4_score:.0f}×0.20={layer4_score*0.20:.1f}",
        f"L5={layer5_score:.0f}×0.20={layer5_score*0.20:.1f}",
        f"Neural={effective_neural:.0f}×0.20={effective_neural*0.20:.1f}",
        f"Liq={exec_liquidity_score:.0f}×0.15={exec_liquidity_score*0.15:.1f}",
    ]
    if drift_detected:
        parts.append("[DRIFT-neural zeroed]")

    explanation = f"TotalScore={total:.1f} → {decision.upper()}  " + "  ".join(parts)

    return MasterScoreResult(
        total_score=total,
        decision=decision,
        size_multiplier=size_mult,
        size_boost=(decision == "boost"),
        component_scores=components,
        explanation=explanation,
    )


def neural_confidence_to_score(
    prob: float,
    min_confidence: float = 0.05,
) -> float:
    """Convert ML model probability output to 0-100 neural confidence score.

    prob: P(up) in [0, 1] from the ensemble scorer
    Converts |prob - 0.5| × 2 → confidence, then scales to 0-100.
    Returns 0 when confidence is below min_confidence threshold.
    """
    confidence = abs(prob - 0.5) * 2.0   # 0 at 0.5, 1 at 0 or 1
    if confidence < min_confidence:
        return 0.0
    return round(min(100.0, confidence * 100.0), 2)
