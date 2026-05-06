"""Concept drift detection — know when the market has changed.

Two detectors:
  PageHinkleyDetector  — tracks cumulative prediction error,
                         fires when error exceeds adaptive threshold.
  FeatureDriftMonitor  — Population Stability Index (PSI) to detect
                         when input feature distributions shift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class PageHinkleyDetector:
    """Page-Hinkley sequential change-point test.

    Standard in quantitative finance for detecting regime shifts.
    Fires when model's cumulative error drifts above threshold,
    indicating the model is no longer suited to current conditions.

    Args:
        threshold:   Detection threshold (λ). Higher = less sensitive.
        min_samples: Minimum samples before drift can be declared.
        alpha:       Forgetting factor for mean update (0 = no forgetting).
    """

    def __init__(
        self,
        threshold: float = 50.0,
        min_samples: int = 30,
        alpha: float = 0.01,
    ) -> None:
        self._threshold  = threshold
        self._min_samples = min_samples
        self._alpha      = alpha
        self.reset()

    def reset(self) -> None:
        self._n         = 0
        self._sum       = 0.0
        self._min_sum   = 0.0
        self._mean      = 0.0
        self._drift_detected = False
        self._last_drift_n: int | None = None

    def update(self, error: float) -> bool:
        """Feed next prediction absolute error. Returns True if drift detected."""
        self._n += 1
        # Update running mean with forgetting
        self._mean = (1 - self._alpha) * self._mean + self._alpha * abs(error)
        self._sum += abs(error) - self._mean - 0.01  # small epsilon for sensitivity
        self._min_sum = min(self._min_sum, self._sum)

        if self._n < self._min_samples:
            return False

        ph_stat = self._sum - self._min_sum
        if ph_stat > self._threshold:
            if not self._drift_detected:
                logger.warning(
                    "PageHinkley: drift detected at sample {} (stat={:.2f} > threshold={})",
                    self._n, ph_stat, self._threshold,
                )
                self._drift_detected = True
                self._last_drift_n   = self._n
            return True

        if self._drift_detected and ph_stat < self._threshold * 0.5:
            self._drift_detected = False
        return False

    @property
    def is_drifting(self) -> bool:
        return self._drift_detected

    @property
    def samples_since_drift(self) -> int | None:
        if self._last_drift_n is None:
            return None
        return self._n - self._last_drift_n

    def get_status(self) -> dict[str, Any]:
        return {
            "n_samples":     self._n,
            "is_drifting":   self._drift_detected,
            "threshold":     self._threshold,
            "last_drift_at": self._last_drift_n,
        }


class FeatureDriftMonitor:
    """Population Stability Index (PSI) for feature distribution drift.

    PSI < 0.1  → stable   (no action needed)
    PSI 0.1-0.2 → moderate drift  (monitor closely)
    PSI > 0.2  → significant drift (retrain recommended)
    PSI > 0.5  → severe drift   (model likely invalid)
    """

    PSI_STABLE    = 0.10
    PSI_MODERATE  = 0.20
    PSI_SEVERE    = 0.50

    def __init__(self, n_bins: int = 10, min_samples: int = 100) -> None:
        self._n_bins    = n_bins
        self._min_samples = min_samples
        self._reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # feature → (bin_edges, ref_pcts)

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Learn reference distributions from training data."""
        self._reference.clear()
        for col in feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < self._min_samples:
                continue
            counts, edges = np.histogram(vals, bins=self._n_bins)
            pcts = counts / counts.sum()
            pcts = np.where(pcts == 0, 1e-4, pcts)  # avoid log(0)
            self._reference[col] = (edges, pcts)

    def score(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute PSI for each monitored feature. Returns {feature: psi}."""
        if not self._reference:
            return {}
        results: dict[str, float] = {}
        for col, (edges, ref_pcts) in self._reference.items():
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue
            counts, _ = np.histogram(vals, bins=edges)
            curr_pcts = counts / max(counts.sum(), 1)
            curr_pcts = np.where(curr_pcts == 0, 1e-4, curr_pcts)
            psi = float(np.sum((curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts)))
            results[col] = round(psi, 4)
        return results

    def max_psi(self, df: pd.DataFrame) -> float:
        """Max PSI across all features — quick health check."""
        scores = self.score(df)
        return max(scores.values(), default=0.0)

    def is_drifting(self, df: pd.DataFrame) -> bool:
        return self.max_psi(df) > self.PSI_MODERATE

    def drift_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        scores = self.score(df)
        if not scores:
            return {"status": "no_reference", "max_psi": 0.0, "drifted_features": []}
        max_psi = max(scores.values(), default=0.0)
        drifted = [f for f, v in scores.items() if v > self.PSI_MODERATE]
        status = "stable"
        if max_psi > self.PSI_SEVERE:
            status = "severe"
        elif max_psi > self.PSI_MODERATE:
            status = "moderate"
        elif max_psi > self.PSI_STABLE:
            status = "minor"
        return {
            "status":           status,
            "max_psi":          round(max_psi, 4),
            "drifted_features": drifted,
            "feature_psi":      scores,
        }
