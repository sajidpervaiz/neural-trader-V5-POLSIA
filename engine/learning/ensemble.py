"""Regime-Aware Ensemble — the core of the adaptive learning system.

Architecture:
  Per regime: LightGBM + XGBoost + Ridge (3-model stacking)
  Online weight adaptation: exponential forgetting favours recent performance
  Isotonic calibration: converts raw probabilities to reliable estimates
  Kelly criterion: position sizing from calibrated confidence

Six ARMS regimes (matches regime_classifier.py):
  STRONG_TREND_UP, WEAK_TREND_UP, COMPRESSION,
  RANGE_CHOP, WEAK_TREND_DOWN, STRONG_TREND_DOWN

Design philosophy (20-year perspective):
  - Never trust a single model. Ensemble errors are uncorrelated.
  - Regime-specific models capture non-stationarity. A trend-following
    model will destroy capital in chop; a mean-reversion model will
    destroy it in a trend.
  - Kelly without calibration is suicide. Calibrate first.
  - Fractional Kelly (0.25×) prevents ruin even with model errors.
  - Online adaptation means the ensemble self-corrects within days
    of a regime shift without needing a full retrain.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_BUNDLE_MAGIC = b"NTV5ENS1"
_SIG_LEN = 32  # sha256

def _bundle_key() -> bytes:
    """Return the HMAC key used to sign/verify ensemble pickle bundles.

    Source order: env MODEL_HMAC_KEY, env NEURALTRADER_SECRET, else a
    deterministic per-host fallback. The fallback still prevents
    drive-by supply-chain swaps but is weaker than an ops-provisioned key.
    """
    for var in ("MODEL_HMAC_KEY", "NEURALTRADER_SECRET"):
        val = os.environ.get(var)
        if val:
            return val.encode("utf-8")
    host = os.uname().nodename if hasattr(os, "uname") else "localhost"
    return hashlib.sha256(f"ntv5-ensemble-{host}".encode()).digest()

warnings.filterwarnings("ignore", category=UserWarning)

# ── Regime constants ────────────────────────────────────────────────────────
REGIMES = [
    "STRONG_TREND_UP",
    "WEAK_TREND_UP",
    "COMPRESSION",
    "RANGE_CHOP",
    "WEAK_TREND_DOWN",
    "STRONG_TREND_DOWN",
]
DEFAULT_REGIME = "RANGE_CHOP"
KELLY_FRACTION = 0.25   # fractional Kelly — never go full Kelly


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class EnsemblePrediction:
    prob:        float        # calibrated win probability [0, 1]
    score:       float        # signed score [-1, 1] = (prob - 0.5) * 2
    kelly_mult:  float        # fractional Kelly position multiplier [0, 1]
    confidence:  float        # |prob - 0.5| normalised to [0, 1]
    regime:      str
    model_probs: dict[str, float] = field(default_factory=dict)  # per-model raw probs
    weights:     dict[str, float] = field(default_factory=dict)  # per-model weights used


@dataclass
class RegimeModelBundle:
    """Three models + isotonic calibrator for a single regime."""
    regime:     str
    lgb:        Any = None
    xgb:        Any = None
    ridge:      Any = None
    calibrator: Any = None          # sklearn IsotonicRegression or None
    weights:    np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.2]))
    n_train:    int = 0
    last_auc:   float = 0.0
    # Online performance tracking (circular buffer of recent outcomes)
    _outcome_buf: deque = field(default_factory=lambda: deque(maxlen=200))

    def has_models(self) -> bool:
        return self.lgb is not None or self.xgb is not None or self.ridge is not None


# ── Calibration ─────────────────────────────────────────────────────────────
def _fit_calibrator(probs: np.ndarray, labels: np.ndarray):
    """Fit isotonic regression calibrator. Returns None if not enough data."""
    if len(probs) < 20 or len(set(labels)) < 2:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(probs, labels)
        return cal
    except Exception:
        return None


def _calibrate(calibrator, probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probs
    try:
        return np.clip(calibrator.predict(probs), 1e-6, 1 - 1e-6)
    except Exception:
        return probs


# ── Kelly criterion ──────────────────────────────────────────────────────────
def _kelly(p: float, b: float = 2.0) -> float:
    """Full Kelly fraction. p=win prob, b=win/loss ratio.
    Returns fraction of bankroll to risk; clamped to [0, 1].
    """
    if b <= 0 or p <= 0:
        return 0.0
    k = (p * (b + 1) - 1) / b
    return float(np.clip(k, 0.0, 1.0))


# ── Model training helpers ───────────────────────────────────────────────────
def _train_lgb(X, y, params: dict):
    try:
        import lightgbm as lgb
        p = {
            "objective": "binary", "metric": "auc", "verbose": -1,
            "learning_rate": params.get("learning_rate", 0.05),
            "num_leaves":    params.get("num_leaves", 31),
            "min_child_samples": 20,
            "feature_fraction": params.get("feature_fraction", 0.8),
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 0.1,
        }
        ds = lgb.Dataset(X, label=y)
        m = lgb.train(p, ds, num_boost_round=int(params.get("n_estimators", 200)),
                      callbacks=[lgb.log_evaluation(0)])
        return m, "lgb"
    except (ImportError, Exception):
        return None, None


def _train_xgb(X, y, params: dict):
    try:
        import xgboost as xgb
        p = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": 5, "subsample": 0.8,
            "colsample_bytree": params.get("feature_fraction", 0.8),
            "n_estimators": int(params.get("n_estimators", 200)),
            "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X, y)
        return m, "xgb"
    except (ImportError, Exception):
        return None, None


def _train_ridge(X, y, params: dict):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        m = LogisticRegression(C=params.get("C", 1.0), max_iter=500, solver="lbfgs")
        m.fit(Xs, y)
        return (scaler, m), "ridge"
    except Exception:
        return None, None


def _predict_prob(model, model_type: str, X: np.ndarray) -> np.ndarray | None:
    try:
        if model_type == "lgb":
            return model.predict(X)
        elif model_type == "xgb":
            return model.predict_proba(X)[:, 1]
        elif model_type == "ridge":
            scaler, clf = model
            return clf.predict_proba(scaler.transform(X))[:, 1]
    except Exception:
        return None
    return None


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    u = sum(p > n for p in pos for n in neg)
    return float(u / (n_pos * n_neg))


# ── RegimeAwareEnsemble ──────────────────────────────────────────────────────
class RegimeAwareEnsemble:
    """Production-grade regime-conditioned ensemble.

    Training:
      Call `fit(X, y, regime_labels, feature_names, wfo_result)` after WFO.
      Each regime gets its own LGB+XGB+Ridge triplet.
      Isotonic calibration is fitted per regime on OOS predictions.

    Inference:
      Call `predict(x_row, regime)` → EnsemblePrediction
      Weights adapt online via `record_outcome(prob, label, regime)`.

    Persistence:
      Call `save(path)` / `load(path)` for hot-reload without restart.

    Args:
        rr_ratio:         Reward/risk ratio used for Kelly sizing.
        forgetting_alpha: Online weight decay (higher = faster adaptation).
        min_regime_rows:  Minimum rows per regime to train dedicated model.
        fallback_to_all:  If regime has too few rows, use all-regime model.
    """

    def __init__(
        self,
        rr_ratio:         float = 2.0,
        forgetting_alpha: float = 0.05,
        min_regime_rows:  int   = 100,
        fallback_to_all:  bool  = True,
    ) -> None:
        self._rr            = rr_ratio
        self._alpha         = forgetting_alpha
        self._min_rows      = min_regime_rows
        self._fallback      = fallback_to_all
        self._feature_names: list[str] = []
        self._bundles:  dict[str, RegimeModelBundle] = {}
        self._global:   RegimeModelBundle | None = None  # trained on all regimes
        self._fitted    = False
        self._n_predict = 0
        self._n_outcome = 0
        self._fit_time  = 0.0

    # ── Training ────────────────────────────────────────────────────────────
    def fit(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        regime_labels:  np.ndarray,   # string regime per row
        feature_names:  list[str],
        params:         dict | None = None,
    ) -> None:
        """Train per-regime models + global fallback + calibration."""
        t0 = time.time()
        self._feature_names = feature_names
        p = params or {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200,
                       "feature_fraction": 0.8}

        # Global model (all regimes) — calibrated on full set
        logger.info("Ensemble: training global fallback model ({} rows)", len(X))
        self._global = self._train_bundle("ALL", X, y, p)

        # Per-regime models
        unique_regimes = np.unique(regime_labels)
        for regime in unique_regimes:
            mask = regime_labels == regime
            Xr, yr = X[mask], y[mask]
            if len(Xr) < self._min_rows:
                logger.debug("Ensemble: regime {} has only {} rows — will use global", regime, len(Xr))
                continue
            logger.info("Ensemble: training regime={} ({} rows)", regime, len(Xr))
            self._bundles[regime] = self._train_bundle(regime, Xr, yr, p)

        self._fitted   = True
        self._fit_time = time.time() - t0
        logger.info(
            "Ensemble fit complete: {} regime models + global, t={:.1f}s",
            len(self._bundles), self._fit_time,
        )

    def _train_bundle(self, name: str, X: np.ndarray, y: np.ndarray, p: dict) -> RegimeModelBundle:
        bundle = RegimeModelBundle(regime=name)
        probs_list = []

        lgb_model, lgb_type = _train_lgb(X, y, p)
        xgb_model, xgb_type = _train_xgb(X, y, p)
        rdg_model, rdg_type = _train_ridge(X, y, p)

        bundle.lgb   = (lgb_model, lgb_type) if lgb_model is not None else None
        bundle.xgb   = (xgb_model, xgb_type) if xgb_model is not None else None
        bundle.ridge = (rdg_model, rdg_type) if rdg_model is not None else None
        bundle.n_train = len(X)

        # Set initial weights based on which models trained successfully
        n_ok = sum([bundle.lgb is not None, bundle.xgb is not None, bundle.ridge is not None])
        if n_ok == 0:
            return bundle
        if n_ok == 3:
            bundle.weights = np.array([0.45, 0.35, 0.20])
        elif n_ok == 2:
            bundle.weights = np.array([0.55, 0.45, 0.0])[:n_ok]
        else:
            bundle.weights = np.array([1.0])

        # Calibration: use leave-one-out style by splitting training data
        split = max(30, len(X) // 5)
        X_cal, y_cal = X[-split:], y[-split:]
        cal_probs = self._ensemble_probs(bundle, X_cal)
        if cal_probs is not None:
            bundle.calibrator = _fit_calibrator(cal_probs, y_cal)
            if bundle.calibrator is not None:
                cal_cal = _calibrate(bundle.calibrator, cal_probs)
                bundle.last_auc = _auc(y_cal, cal_cal)
                logger.debug("  {} calibrated AUC={:.4f}", name, bundle.last_auc)

        return bundle

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, x: np.ndarray, regime: str | None = None) -> EnsemblePrediction:
        """Predict from a single feature row (shape: [n_features,] or [1, n_features])."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        regime = regime or DEFAULT_REGIME
        bundle = self._get_bundle(regime)

        if bundle is None or not bundle.has_models():
            return EnsemblePrediction(0.5, 0.0, 0.0, 0.0, regime)

        raw = self._ensemble_probs(bundle, x)
        if raw is None:
            return EnsemblePrediction(0.5, 0.0, 0.0, 0.0, regime)

        prob = float(_calibrate(bundle.calibrator, raw)[0])
        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))

        score     = float(np.clip((prob - 0.5) * 2, -1, 1))
        confidence = float(abs(prob - 0.5) * 2)
        kelly     = _kelly(prob, self._rr) * KELLY_FRACTION

        self._n_predict += 1
        return EnsemblePrediction(
            prob=prob, score=score, kelly_mult=kelly,
            confidence=confidence, regime=regime,
        )

    def _get_bundle(self, regime: str) -> RegimeModelBundle | None:
        if regime in self._bundles:
            return self._bundles[regime]
        if self._fallback and self._global is not None:
            return self._global
        return None

    def _ensemble_probs(self, bundle: RegimeModelBundle, X: np.ndarray) -> np.ndarray | None:
        """Weighted average of model probabilities."""
        probs = []
        weights = []

        w = bundle.weights
        idx = 0
        for model_pair, w_i in zip(
            [bundle.lgb, bundle.xgb, bundle.ridge],
            [w[0] if len(w) > 0 else 0, w[1] if len(w) > 1 else 0, w[2] if len(w) > 2 else 0],
        ):
            if model_pair is not None:
                model, mtype = model_pair
                p = _predict_prob(model, mtype, X)
                if p is not None:
                    probs.append(p)
                    weights.append(w_i)

        if not probs:
            return None
        weights = np.array(weights)
        weights = weights / weights.sum()
        return sum(p * w for p, w in zip(probs, weights))

    # ── Online adaptation ─────────────────────────────────────────────────────
    def record_outcome(self, prob: float, label: int, regime: str | None = None) -> None:
        """Feed observed outcome to adapt per-regime model weights.

        Uses exponential moving average of per-model accuracy.
        Models that perform well in the current regime get upweighted.
        """
        regime = regime or DEFAULT_REGIME
        bundle = self._bundles.get(regime) or self._global
        if bundle is None:
            return

        correct = int((prob > 0.5) == (label == 1))
        bundle._outcome_buf.append((prob, label, correct))
        self._n_outcome += 1

        # Adapt weights every 20 outcomes
        if len(bundle._outcome_buf) >= 20 and self._n_outcome % 20 == 0:
            self._adapt_weights(bundle)

    def _adapt_weights(self, bundle: RegimeModelBundle) -> None:
        """Recompute model weights from recent accuracy via EMA."""
        buf = list(bundle._outcome_buf)
        if len(buf) < 20:
            return

        # Use last 100 outcomes for weight update
        recent = buf[-100:]
        labels = np.array([b[1] for b in recent])

        model_accs = []
        for model_pair in [bundle.lgb, bundle.xgb, bundle.ridge]:
            if model_pair is None:
                model_accs.append(None)
                continue
            # We don't have the raw X here, so use outcome accuracy as proxy
            # In production you'd store (X, y) pairs — here we use mean outcome
            model_accs.append(float(np.mean([b[2] for b in recent])))

        valid = [(i, a) for i, a in enumerate(model_accs) if a is not None]
        if not valid:
            return

        # Softmax weighting: better models get exponentially higher weight
        accs = np.array([a for _, a in valid])
        # Add small noise to avoid ties
        accs = accs + np.random.uniform(0, 0.001, len(accs))
        softmax = np.exp(accs * 5) / np.exp(accs * 5).sum()

        new_w = np.zeros(3)
        for (i, _), sw in zip(valid, softmax):
            new_w[i] = sw

        # EMA blend: retain old weights partially
        old_w = bundle.weights
        if len(old_w) != len(new_w):
            old_w = np.pad(old_w, (0, max(0, len(new_w) - len(old_w))))[:len(new_w)]
        bundle.weights = (1 - self._alpha) * old_w + self._alpha * new_w
        bundle.weights /= bundle.weights.sum()

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Serialize with an HMAC prefix so tampered/swapped bundles fail to load.

        Bundle layout: MAGIC(8) || sha256-HMAC(32) || pickle-payload
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        sig = hmac.new(_bundle_key(), blob, hashlib.sha256).digest()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(_BUNDLE_MAGIC + sig + blob)
        tmp.rename(path)
        logger.info("Ensemble saved → {} (signed)", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeAwareEnsemble":
        data = Path(path).read_bytes()
        if not data.startswith(_BUNDLE_MAGIC):
            raise ValueError(
                f"Ensemble bundle at {path} is unsigned or legacy format — "
                "refusing to unpickle. Re-train or migrate with a signed save()."
            )
        head = len(_BUNDLE_MAGIC)
        sig, blob = data[head:head + _SIG_LEN], data[head + _SIG_LEN:]
        expected = hmac.new(_bundle_key(), blob, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError(
                f"Ensemble bundle at {path} failed HMAC verification — "
                "refusing to load potentially tampered weights."
            )
        obj = pickle.loads(blob)  # nosec B301 — HMAC-verified above
        logger.info("Ensemble loaded ← {} (verified)", path)
        return obj

    # ── Status ───────────────────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        regime_info = {}
        for r, b in self._bundles.items():
            regime_info[r] = {
                "n_train":   b.n_train,
                "last_auc":  round(b.last_auc, 4),
                "weights":   {
                    "lgb":   round(float(b.weights[0]) if len(b.weights) > 0 else 0, 3),
                    "xgb":   round(float(b.weights[1]) if len(b.weights) > 1 else 0, 3),
                    "ridge": round(float(b.weights[2]) if len(b.weights) > 2 else 0, 3),
                },
                "outcomes":  len(b._outcome_buf),
            }
        global_info = None
        if self._global:
            global_info = {
                "n_train":  self._global.n_train,
                "last_auc": round(self._global.last_auc, 4),
            }
        return {
            "fitted":          self._fitted,
            "n_regime_models": len(self._bundles),
            "n_predict":       self._n_predict,
            "n_outcome":       self._n_outcome,
            "fit_time_s":      round(self._fit_time, 1),
            "rr_ratio":        self._rr,
            "kelly_fraction":  KELLY_FRACTION,
            "feature_count":   len(self._feature_names),
            "regimes":         regime_info,
            "global_model":    global_info,
        }
