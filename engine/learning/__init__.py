"""Adaptive ML Learning System — full-capacity engine.

Components:
  features   — 55+ engineered features (multi-TF, regime, macro, candlestick)
  drift      — Page-Hinkley concept drift + PSI feature drift detection
  wfo        — Walk-forward optimization with Sharpe/Calmar objective
  ensemble   — Regime-aware LightGBM+XGBoost+Ridge ensemble + online weight adaptation
  scorer     — AdaptiveMLScorer: drop-in replacement for MLScorer with Kelly sizing
"""
from engine.learning.scorer import AdaptiveMLScorer
from engine.learning.features import engineer_features, FEATURE_COLS
from engine.learning.drift import PageHinkleyDetector, FeatureDriftMonitor
from engine.learning.wfo import WalkForwardOptimizer, WFOResult
from engine.learning.ensemble import RegimeAwareEnsemble

__all__ = [
    "AdaptiveMLScorer",
    "engineer_features",
    "FEATURE_COLS",
    "PageHinkleyDetector",
    "FeatureDriftMonitor",
    "WalkForwardOptimizer",
    "WFOResult",
    "RegimeAwareEnsemble",
]
