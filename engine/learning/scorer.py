"""AdaptiveMLScorer — drop-in replacement for MLScorer with full adaptive capabilities.

Replaces the static LightGBM scorer with:
  • 55+ engineered features (multi-TF, regime, macro, candlestick)
  • Regime-aware ensemble (LGB + XGB + Ridge per ARMS regime)
  • Walk-forward optimised training (Sharpe/Calmar objective)
  • Isotonic calibration → reliable probabilities
  • Kelly criterion position sizing
  • Online feedback loop (record_outcome → adapt weights)
  • Concept drift detection (Page-Hinkley + PSI)
  • Full persistence with hot-reload

Backward compatibility:
  score(df) → float in [-1, 1]  (same as MLScorer.score)
Extended API:
  score_with_kelly(df, regime_state, rr_ratio) → (score, kelly_mult, confidence)
  record_outcome(pnl_pct, regime)
  get_status() → dict
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from engine.learning.ensemble import RegimeAwareEnsemble, EnsemblePrediction
from engine.learning.features import engineer_features, FEATURE_COLS
from engine.learning.drift import PageHinkleyDetector, FeatureDriftMonitor
from engine.learning.wfo import WalkForwardOptimizer, WFOResult


_DEFAULT_ENSEMBLE_PATH = Path("models/ensemble_state.pkl")
_DEFAULT_LGB_PATH      = Path("models/ml_signal.lgb")


class AdaptiveMLScorer:
    """Full adaptive ML scoring engine.

    Designed as a drop-in replacement for MLScorer so existing signal
    generator code continues to work without modification.

    Args:
        ensemble_path: Where to persist the ensemble (pickle).
        lgb_fallback:  Path to legacy static LightGBM model as fallback.
        rr_ratio:      Reward/risk ratio for Kelly computation.
        drift_threshold: Page-Hinkley threshold before forcing retrain flag.
        min_confidence: Below this |prob-0.5|, abstain (return score=0).
    """

    def __init__(
        self,
        ensemble_path: str | Path = _DEFAULT_ENSEMBLE_PATH,
        lgb_fallback:  str | Path = _DEFAULT_LGB_PATH,
        rr_ratio:      float = 2.0,
        drift_threshold: float = 50.0,
        min_confidence:  float = 0.05,
    ) -> None:
        self._ensemble_path  = Path(ensemble_path)
        self._lgb_path       = Path(lgb_fallback)
        self._rr             = rr_ratio
        self._min_confidence = min_confidence

        # Components
        self._ensemble:   RegimeAwareEnsemble | None = None
        self._lgb_model:  Any = None          # legacy fallback
        self._lgb_features: list[str] = []

        # Drift detection
        self._ph_detector    = PageHinkleyDetector(threshold=drift_threshold)
        self._feature_monitor = FeatureDriftMonitor(n_bins=10)

        # Stats
        self._n_scored   = 0
        self._n_kelly    = 0
        self._n_fallback = 0
        self._n_outcomes = 0
        self._last_score = 0.0
        self._last_kelly = 0.0
        self._last_confidence = 0.0
        self._drift_flagged  = False

        # Load persisted ensemble (if exists)
        self._try_load_ensemble()
        # Load legacy LGB as fallback
        self._try_load_lgb()

    # ── Loading ──────────────────────────────────────────────────────────────
    def _try_load_ensemble(self) -> None:
        if self._ensemble_path.exists():
            try:
                self._ensemble = RegimeAwareEnsemble.load(self._ensemble_path)
                logger.info("AdaptiveMLScorer: ensemble loaded from {}", self._ensemble_path)
            except Exception as exc:
                logger.warning("AdaptiveMLScorer: ensemble load failed: {}", exc)
                self._ensemble = None

    def _try_load_lgb(self) -> None:
        if not self._lgb_path.exists():
            return
        try:
            import lightgbm as lgb
            self._lgb_model = lgb.Booster(model_file=str(self._lgb_path))
            self._lgb_features = self._lgb_model.feature_name()
            logger.info("AdaptiveMLScorer: LGB fallback loaded ({} features)", len(self._lgb_features))
        except Exception as exc:
            logger.debug("AdaptiveMLScorer: LGB fallback not available: {}", exc)

    def reload(self) -> bool:
        """Hot-reload ensemble from disk. Called after retrainer saves new model."""
        self._try_load_ensemble()
        self._try_load_lgb()
        return self._ensemble is not None or self._lgb_model is not None

    # ── Core scoring ─────────────────────────────────────────────────────────
    def score(self, df: pd.DataFrame, regime_state: Any = None) -> float:
        """Backward-compatible score. Returns float in [-1, 1].

        Regime state is optional — if None, uses DEFAULT_REGIME.
        """
        pred = self._predict(df, regime_state)
        self._n_scored += 1
        self._last_score = pred.score
        return pred.score

    def score_with_kelly(
        self,
        df:           pd.DataFrame,
        regime_state: Any   = None,
        rr_ratio:     float | None = None,
    ) -> tuple[float, float, float]:
        """Extended scoring returning (score, kelly_mult, confidence).

        score:      [-1, 1] signal strength
        kelly_mult: [0, 1] fraction of max position size (Kelly-sized)
        confidence: [0, 1] calibrated confidence
        """
        pred = self._predict(df, regime_state)
        self._n_scored += 1
        self._n_kelly  += 1
        self._last_score      = pred.score
        self._last_kelly      = pred.kelly_mult
        self._last_confidence = pred.confidence
        return pred.score, pred.kelly_mult, pred.confidence

    def _predict(self, df: pd.DataFrame, regime_state: Any = None) -> EnsemblePrediction:
        """Internal prediction pipeline."""
        regime = self._extract_regime(regime_state)

        # Feature engineering
        try:
            feat_df = engineer_features(df, regime_state=regime_state)
            feat_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
            if len(feat_cols) == 0:
                return self._fallback_predict(df)
            x = feat_df[feat_cols].iloc[-1].values.astype(np.float32)
            if np.isnan(x).all():
                return self._fallback_predict(df)
            # Replace NaN with 0 (median fill would need training data)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            logger.debug("AdaptiveMLScorer: feature engineering failed: {}", exc)
            return self._fallback_predict(df)

        # Drift check (non-blocking)
        try:
            self._feature_monitor.score(feat_df[feat_cols].tail(50))
        except Exception:
            pass

        # Ensemble prediction
        if self._ensemble is not None and self._ensemble._fitted:
            try:
                pred = self._ensemble.predict(x, regime)
                # Apply minimum confidence filter
                if pred.confidence < self._min_confidence:
                    pred.score = 0.0
                    pred.kelly_mult = 0.0
                return pred
            except Exception as exc:
                logger.debug("AdaptiveMLScorer: ensemble predict failed: {}", exc)

        return self._fallback_predict(df)

    def _fallback_predict(self, df: pd.DataFrame) -> EnsemblePrediction:
        """Use legacy LGB model when ensemble is unavailable."""
        from engine.learning.ensemble import EnsemblePrediction, DEFAULT_REGIME
        self._n_fallback += 1

        if self._lgb_model is None:
            return EnsemblePrediction(0.5, 0.0, 0.0, 0.0, DEFAULT_REGIME)

        try:
            # Build a row with ALL model features — fill missing ones with 0
            row = df.tail(1)
            x = np.zeros((1, len(self._lgb_features)), dtype=np.float32)
            for i, feat in enumerate(self._lgb_features):
                if feat in row.columns:
                    val = row[feat].iloc[0]
                    x[0, i] = 0.0 if (val != val) else float(val)  # NaN → 0
            prob = float(self._lgb_model.predict(x)[0])
            score = float(np.clip((prob - 0.5) * 2, -1, 1))
            confidence = float(abs(prob - 0.5) * 2)
            return EnsemblePrediction(prob, score, 0.0, confidence, DEFAULT_REGIME)
        except Exception as exc:
            logger.debug("AdaptiveMLScorer: LGB fallback failed: {}", exc)
            return EnsemblePrediction(0.5, 0.0, 0.0, 0.0, DEFAULT_REGIME)

    @staticmethod
    def _extract_regime(regime_state: Any) -> str | None:
        if regime_state is None:
            return None
        if isinstance(regime_state, str):
            return regime_state
        # ARMS RegimeState object
        for attr in ("regime", "current_regime", "state", "name"):
            if hasattr(regime_state, attr):
                v = getattr(regime_state, attr)
                return str(v) if v is not None else None
        return None

    # ── Online feedback ───────────────────────────────────────────────────────
    def record_outcome(self, pnl_pct: float, regime: str | None = None) -> None:
        """Feed trade outcome for online adaptation.

        pnl_pct: signed P&L percentage (positive = profit, negative = loss)
        regime:  ARMS regime string at time of signal
        """
        label = 1 if pnl_pct > 0 else 0
        prob  = self._last_score * 0.5 + 0.5   # convert [-1,1] back to prob

        # Update drift detector
        error = abs(prob - label)
        was_drifting = self._ph_detector.is_drifting
        self._ph_detector.update(error)
        if self._ph_detector.is_drifting and not was_drifting:
            self._drift_flagged = True
            logger.warning("AdaptiveMLScorer: concept drift flagged — retrain recommended")

        # Update ensemble weights
        if self._ensemble is not None and self._ensemble._fitted:
            try:
                self._ensemble.record_outcome(prob, label, regime)
            except Exception:
                pass

        self._n_outcomes += 1

    # ── Training integration ──────────────────────────────────────────────────
    def fit_from_dataframe(
        self,
        df:             pd.DataFrame,
        label_col:      str = "label",
        regime_col:     str | None = None,
        n_folds:        int = 5,
        min_train_rows: int = 500,
    ) -> WFOResult | None:
        """Full WFO + ensemble fit from a labelled DataFrame.

        df must have FEATURE_COLS and label_col columns.
        Optionally regime_col for regime-aware training.
        Returns WFOResult (metrics) or None on failure.
        """
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        if len(feat_cols) < 5:
            logger.error("fit_from_dataframe: too few features ({} available)", len(feat_cols))
            return None

        X = df[feat_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int32)

        # Mask out NaN rows
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        if len(X) < min_train_rows:
            logger.warning("fit_from_dataframe: insufficient rows ({} after NaN drop)", len(X))
            return None

        # Walk-forward optimisation
        logger.info("AdaptiveMLScorer: running WFO on {} rows × {} features", len(X), len(feat_cols))
        wfo = WalkForwardOptimizer(n_folds=n_folds, min_train_rows=200, test_size=200)
        result = wfo.run(X, y, feat_cols)

        # Fit ensemble with best WFO params
        regime_labels = np.array(["RANGE_CHOP"] * len(X))
        if regime_col and regime_col in df.columns:
            rl = df[regime_col].values[valid]
            regime_labels = np.array([str(r) if r else "RANGE_CHOP" for r in rl])

        logger.info("AdaptiveMLScorer: fitting ensemble with best params={}", result.best_params)
        ensemble = RegimeAwareEnsemble(rr_ratio=self._rr)
        ensemble.fit(X, y, regime_labels, feat_cols, result.best_params)

        # Fit feature drift monitor reference
        self._feature_monitor.fit(df[feat_cols], feat_cols)

        # Save and hot-reload
        ensemble.save(self._ensemble_path)
        self._ensemble = ensemble
        self._drift_flagged = False
        self._ph_detector.reset()

        logger.info(
            "AdaptiveMLScorer: training complete. WFO sharpe={:.3f} calmar={:.3f} auc={:.4f}",
            result.sharpe_mean, result.calmar_mean, result.auc_mean,
        )
        return result

    # ── Bootstrap from raw candles ────────────────────────────────────────────
    def bootstrap_from_ohlcv(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        min_rows: int = 300,
    ) -> bool:
        """Fit ensemble from raw OHLCV data using forward-return labels.

        Designed to be called once at startup after historical candles are seeded,
        so ML scoring works immediately instead of waiting for ModelRetrainer.

        Args:
            df: OHLCV DataFrame with columns open/high/low/close/volume.
            horizon: bars ahead used to define label (1 = next bar return).
            min_rows: minimum rows required to attempt fitting.

        Returns True if fitting succeeded.
        """
        if self._ensemble is not None and self._ensemble._fitted:
            logger.debug("AdaptiveMLScorer: ensemble already fitted — skipping bootstrap")
            return True

        if df is None or len(df) < min_rows:
            logger.warning(
                "AdaptiveMLScorer: bootstrap skipped — only {} rows (need {})",
                0 if df is None else len(df), min_rows,
            )
            return False

        try:
            # Engineer features
            feat_df = engineer_features(df.copy(), regime_state=None)

            # Forward-return label: 1 if close[t+horizon] > close[t], else 0
            fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1.0
            feat_df["label"] = (fwd_ret > 0).astype(int)
            feat_df = feat_df.dropna(subset=["label", "close"])

            # Remove the last `horizon` rows (no label available)
            feat_df = feat_df.iloc[:-horizon]

            result = self.fit_from_dataframe(feat_df, label_col="label", min_train_rows=min_rows)
            if result is not None:
                logger.info(
                    "AdaptiveMLScorer: bootstrap complete — {} rows, WFO auc={:.4f}",
                    len(feat_df), result.auc_mean,
                )
                return True
            return False
        except Exception as exc:
            logger.warning("AdaptiveMLScorer: bootstrap failed: {}", exc)
            return False

    # ── Status ───────────────────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        ensemble_status = self._ensemble.get_status() if self._ensemble else {}
        drift_status = self._ph_detector.get_status()
        return {
            "fitted":          self._ensemble is not None and self._ensemble._fitted,
            "n_scored":        self._n_scored,
            "n_kelly_calls":   self._n_kelly,
            "n_fallback":      self._n_fallback,
            "n_outcomes":      self._n_outcomes,
            "last_score":      round(self._last_score, 4),
            "last_kelly":      round(self._last_kelly, 4),
            "last_confidence": round(self._last_confidence, 4),
            "drift_flagged":   self._drift_flagged,
            "drift_detector":  drift_status,
            "ensemble":        ensemble_status,
            "has_lgb_fallback": self._lgb_model is not None,
        }
