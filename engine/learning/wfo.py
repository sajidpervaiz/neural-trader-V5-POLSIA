"""Walk-Forward Optimization — the only valid way to validate time-series models.

A 20-year trader never backtests on the full dataset — that's data snooping.
Walk-forward uses an expanding training window and rolls the test forward,
mimicking real trading conditions where you train → deploy → retrain.

Objective: Sharpe ratio (risk-adjusted returns), not raw AUC.
Secondary: Calmar ratio (return / max drawdown).

Process:
  1. Split data into N folds
  2. For each fold: train on [0 → fold_start], test on [fold_start → fold_end]
  3. Aggregate out-of-sample predictions → compute Sharpe/Calmar
  4. Return best hyperparams + feature importances + fold diagnostics
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FoldResult:
    fold:       int
    train_rows: int
    test_rows:  int
    accuracy:   float
    auc:        float
    sharpe:     float
    calmar:     float
    params:     dict[str, Any]


@dataclass
class WFOResult:
    sharpe_mean:         float
    calmar_mean:         float
    accuracy_mean:       float
    auc_mean:            float
    best_params:         dict[str, Any]
    fold_results:        list[FoldResult]
    feature_importances: dict[str, float]
    oos_predictions:     np.ndarray   # full out-of-sample predictions
    oos_actuals:         np.ndarray   # corresponding actuals
    duration_s:          float


def _sharpe_from_predictions(preds: np.ndarray, actuals: np.ndarray, rr: float = 2.0) -> float:
    """Simulate strategy returns from binary predictions.
    Positive pred → long, negative → short (or flat).
    Returns daily Sharpe (annualised with √252).
    """
    if len(preds) == 0:
        return 0.0
    # Simulated daily P&L: +rr if correct, -1 if wrong, 0 if abstain
    directions = np.sign(preds)
    actuals_dir = np.where(actuals > 0, 1, -1)
    pnl = np.where(directions == 0, 0,
                   np.where(directions == actuals_dir, rr, -1.0))
    if pnl.std() == 0:
        return 0.0
    return float((pnl.mean() / pnl.std()) * np.sqrt(252))


def _calmar_from_predictions(preds: np.ndarray, actuals: np.ndarray, rr: float = 2.0) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    directions = np.sign(preds)
    actuals_dir = np.where(actuals > 0, 1, -1)
    pnl = np.where(directions == 0, 0,
                   np.where(directions == actuals_dir, rr, -1.0))
    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    max_dd = drawdown.max()
    if max_dd == 0:
        return 0.0
    annualised_return = pnl.mean() * 252
    return float(annualised_return / max_dd)


def _auc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Simple AUC without sklearn dependency."""
    if len(set(y_true)) < 2:
        return 0.5
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Mann-Whitney U statistic
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    u = sum(p > n for p in pos_probs for n in neg_probs)
    return float(u / (n_pos * n_neg))


def _train_model(X_train, y_train, params: dict, model_type: str = "lightgbm"):
    """Train a single model with given hyperparameters."""
    try:
        import lightgbm as lgb
        p = {
            "objective": "binary", "metric": "auc",
            "learning_rate": params.get("learning_rate", 0.05),
            "num_leaves": params.get("num_leaves", 31),
            "min_child_samples": params.get("min_child_samples", 30),
            "feature_fraction": params.get("feature_fraction", 0.8),
            "bagging_fraction": params.get("bagging_fraction", 0.8),
            "bagging_freq": 5, "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1,
        }
        ds = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(p, ds, num_boost_round=int(params.get("n_estimators", 200)),
                          callbacks=[lgb.log_evaluation(0)])
        return model, "lightgbm"
    except ImportError:
        pass

    try:
        import xgboost as xgb
        p = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "subsample": params.get("bagging_fraction", 0.8),
            "colsample_bytree": params.get("feature_fraction", 0.8),
            "n_estimators": int(params.get("n_estimators", 200)),
            "use_label_encoder": False, "verbosity": 0,
        }
        model = xgb.XGBClassifier(**p)
        model.fit(X_train, y_train)
        return model, "xgboost"
    except ImportError:
        pass

    # Minimal linear fallback
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500, C=params.get("C", 1.0))
    model.fit(X_train, y_train)
    return model, "logistic"


def _predict(model, model_type: str, X) -> np.ndarray:
    if model_type == "lightgbm":
        return model.predict(X)
    elif model_type == "xgboost":
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


def _feature_importances(model, model_type: str, feature_names: list[str]) -> dict[str, float]:
    try:
        if model_type == "lightgbm":
            imp = model.feature_importance(importance_type="gain")
            total = imp.sum() or 1
            return {f: float(imp[i] / total) for i, f in enumerate(feature_names)
                    if i < len(imp)}
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            total = imp.sum() or 1
            return {f: float(imp[i] / total) for i, f in enumerate(feature_names)
                    if i < len(imp)}
    except Exception:
        pass
    return {}


class WalkForwardOptimizer:
    """Anchored walk-forward optimization.

    Trains on an expanding window and evaluates on the next unseen window.
    Optimises for out-of-sample Sharpe ratio.

    Args:
        n_folds:        Number of test folds.
        min_train_rows: Minimum rows required to start training.
        test_size:      Number of rows per test fold.
        rr_ratio:       Reward/risk ratio used for Sharpe simulation.
    """

    PARAM_GRID = [
        {"learning_rate": 0.01, "num_leaves": 15, "n_estimators": 300, "feature_fraction": 0.7},
        {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200, "feature_fraction": 0.8},
        {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 200, "feature_fraction": 0.9},
        {"learning_rate": 0.10, "num_leaves": 31, "n_estimators": 150, "feature_fraction": 0.8},
    ]

    def __init__(
        self,
        n_folds: int = 5,
        min_train_rows: int = 200,
        test_size: int = 200,
        rr_ratio: float = 2.0,
    ) -> None:
        self._n_folds = n_folds
        self._min_train = min_train_rows
        self._test_size = test_size
        self._rr = rr_ratio

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        param_grid: list[dict] | None = None,
    ) -> WFOResult:
        """Run walk-forward optimization. Returns WFOResult."""
        t0 = time.time()
        n = len(X)
        grid = param_grid or self.PARAM_GRID

        if n < self._min_train + self._test_size:
            logger.warning("WFO: insufficient data ({} rows)", n)
            return WFOResult(0.0, 0.0, 0.0, 0.0, grid[0], [], {}, np.array([]), np.array([]), 0.0)

        # Build fold boundaries: each fold tests [test_start, test_end)
        total_test = min(n - self._min_train, self._n_folds * self._test_size)
        test_start_global = n - total_test
        folds = []
        for i in range(self._n_folds):
            ts = test_start_global + i * (total_test // self._n_folds)
            te = min(ts + (total_test // self._n_folds), n)
            if ts >= te or ts < self._min_train:
                continue
            folds.append((ts, te))

        if not folds:
            return WFOResult(0.0, 0.0, 0.0, 0.0, grid[0], [], {}, np.array([]), np.array([]), 0.0)

        # Grid search: evaluate each param set on first fold only (fast)
        best_params = grid[0]
        best_sharpe = float("-inf")
        ts0, te0 = folds[0]
        X_tr0, y_tr0 = X[:ts0], y[:ts0]
        X_te0, y_te0 = X[ts0:te0], y[ts0:te0]
        for params in grid:
            try:
                model, mt = _train_model(X_tr0, y_tr0, params)
                preds = _predict(model, mt, X_te0)
                sh = _sharpe_from_predictions(preds - 0.5, y_te0, self._rr)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = params
            except Exception as exc:
                logger.debug("WFO grid param failed: {}", exc)

        # Full WFO with best params
        fold_results: list[FoldResult] = []
        oos_preds: list[np.ndarray] = []
        oos_acts:  list[np.ndarray] = []
        agg_importances: dict[str, float] = {}

        for i, (ts, te) in enumerate(folds):
            X_tr, y_tr = X[:ts], y[:ts]
            X_te, y_te = X[ts:te], y[ts:te]
            if len(X_tr) < self._min_train:
                continue
            try:
                model, mt = _train_model(X_tr, y_tr, best_params)
                preds = _predict(model, mt, X_te)
                oos_preds.append(preds)
                oos_acts.append(y_te)

                signed = preds - 0.5
                sh = _sharpe_from_predictions(signed, y_te, self._rr)
                cal = _calmar_from_predictions(signed, y_te, self._rr)
                auc = _auc_binary(y_te, preds)
                acc = float(((preds > 0.5) == y_te).mean())

                fold_results.append(FoldResult(
                    fold=i, train_rows=len(X_tr), test_rows=len(X_te),
                    accuracy=acc, auc=auc, sharpe=sh, calmar=cal, params=best_params,
                ))

                # Accumulate feature importances
                fi = _feature_importances(model, mt, feature_names)
                for f, v in fi.items():
                    agg_importances[f] = agg_importances.get(f, 0.0) + v / len(folds)

            except Exception as exc:
                logger.warning("WFO fold {} failed: {}", i, exc)

        if not fold_results:
            return WFOResult(0.0, 0.0, 0.0, 0.0, best_params, [], {}, np.array([]), np.array([]), time.time() - t0)

        oos_all_preds  = np.concatenate(oos_preds)
        oos_all_actuals = np.concatenate(oos_acts)

        sorted_fi = dict(sorted(agg_importances.items(), key=lambda x: x[1], reverse=True))
        result = WFOResult(
            sharpe_mean   = float(np.mean([f.sharpe for f in fold_results])),
            calmar_mean   = float(np.mean([f.calmar for f in fold_results])),
            accuracy_mean = float(np.mean([f.accuracy for f in fold_results])),
            auc_mean      = float(np.mean([f.auc for f in fold_results])),
            best_params   = best_params,
            fold_results  = fold_results,
            feature_importances = sorted_fi,
            oos_predictions = oos_all_preds,
            oos_actuals     = oos_all_actuals,
            duration_s      = time.time() - t0,
        )
        logger.info(
            "WFO complete: folds={} sharpe={:.3f} calmar={:.3f} auc={:.4f} t={:.1f}s",
            len(fold_results), result.sharpe_mean, result.calmar_mean,
            result.auc_mean, result.duration_s,
        )
        return result
