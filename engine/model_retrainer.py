"""Automatic periodic ML model retrainer.

Loads accumulated candle + position data from SQLite, retrains the
LightGBM signal model, saves it to models/ml_signal.lgb, then
hot-reloads the MLScorer inside the running engine — no restart needed.

Schedule: configurable interval (default 7 days), minimum trade history
required before first retrain (default 100 closed trades).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    pass

# ── Feature engineering (mirrors scripts/train_model.py) ─────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["returns_1"]  = c.pct_change(1)
    df["returns_5"]  = c.pct_change(5)
    df["returns_10"] = c.pct_change(10)
    df["returns_20"] = c.pct_change(20)

    df["vol_10"] = df["returns_1"].rolling(10).std()
    df["vol_20"] = df["returns_1"].rolling(20).std()
    df["vol_60"] = df["returns_1"].rolling(60).std()

    df["ema_12"]       = c.ewm(span=12, adjust=False).mean()
    df["ema_26"]       = c.ewm(span=26, adjust=False).mean()
    df["ema_cross"]    = (df["ema_12"] - df["ema_26"]) / c
    df["ema_50"]       = c.ewm(span=50, adjust=False).mean()
    df["ema_200"]      = c.ewm(span=200, adjust=False).mean()
    df["trend_50_200"] = (df["ema_50"] - df["ema_200"]) / c

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"]  = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50

    macd          = df["ema_12"] - df["ema_26"]
    macd_signal   = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - macd_signal) / c

    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    df["bb_pct"]  = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    h, l = df["high"], df["low"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / c

    if "volume" in df.columns:
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    else:
        df["vol_ratio"] = 1.0

    return df


FEATURE_COLS = [
    "returns_1", "returns_5", "returns_10", "returns_20",
    "vol_10", "vol_20", "vol_60",
    "ema_cross", "trend_50_200",
    "rsi_norm", "macd_hist", "bb_pct", "atr_pct", "vol_ratio",
]


def _label_bars(df: pd.DataFrame, horizon: int = 5, threshold_atr: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    future_return   = df["close"].shift(-horizon) / df["close"] - 1
    atr_threshold   = df["atr_pct"] * threshold_atr
    df["label"]     = (future_return > atr_threshold).astype(int)
    df = df.dropna()
    if len(df) > horizon:
        df = df.iloc[:-horizon]
    return df


# ── Data loading from SQLite ──────────────────────────────────────────────────

def _load_candles_sqlite(db_path: Path, min_rows: int = 500) -> pd.DataFrame | None:
    try:
        import sqlite3
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql(
            "SELECT time_ns, open, high, low, close, volume "
            "FROM candles WHERE timeframe='15m' ORDER BY time_ns",
            conn,
        )
        conn.close()
        if len(df) < min_rows:
            return None
        df["close"]  = df["close"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as exc:
        logger.warning("ModelRetrainer: SQLite candle load failed: {}", exc)
        return None


def _count_closed_trades(db_path: Path) -> int:
    try:
        import sqlite3
        if not db_path.exists():
            return 0
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT COUNT(*) FROM positions WHERE close_time_ns IS NOT NULL")
        n = cur.fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


# ── Core retrain logic ────────────────────────────────────────────────────────

class RetrainResult:
    def __init__(
        self,
        success: bool,
        auc: float = 0.0,
        rows: int = 0,
        duration_s: float = 0.0,
        reason: str = "",
        sha256: str = "",
    ) -> None:
        self.success    = success
        self.auc        = auc
        self.rows       = rows
        self.duration_s = duration_s
        self.reason     = reason
        self.sha256     = sha256

    def __repr__(self) -> str:
        if self.success:
            return f"RetrainResult(ok rows={self.rows} auc={self.auc:.4f} t={self.duration_s:.1f}s)"
        return f"RetrainResult(FAILED reason={self.reason})"


def _retrain_sync(
    df: pd.DataFrame,
    model_path: Path,
    features_path: Path,
    horizon: int = 5,
) -> RetrainResult:
    """Blocking retrain — runs in a thread executor."""
    t0 = time.time()
    try:
        import lightgbm as lgb
    except ImportError:
        return RetrainResult(False, reason="lightgbm not installed")

    try:
        df = _add_features(df)
        df = _label_bars(df, horizon=horizon)

        available = [f for f in FEATURE_COLS if f in df.columns]
        if len(available) < 5:
            return RetrainResult(False, reason=f"too few features: {len(available)}")

        X = df[available].values
        y = df["label"].values

        if len(X) < 200:
            return RetrainResult(False, reason=f"insufficient rows after labeling: {len(X)}")

        pos_rate = float(y.mean())
        logger.info("ModelRetrainer: training on {} rows, {:.1%} positive", len(X), pos_rate)

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective":        "binary",
            "metric":           "auc",
            "learning_rate":    0.05,
            "num_leaves":       31,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "lambda_l1":        0.1,
            "lambda_l2":        0.1,
            "verbose":          -1,
        }
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        auc = float(model.best_score.get("valid_0", {}).get("auc", 0.0))

        # Atomic write: save to .tmp then rename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_model    = model_path.with_suffix(".lgb.tmp")
        tmp_features = features_path.with_suffix(".json.tmp")

        model.save_model(str(tmp_model))
        tmp_features.write_text(json.dumps(available, indent=2))
        tmp_model.replace(model_path)
        tmp_features.replace(features_path)

        sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        duration = time.time() - t0
        logger.info(
            "ModelRetrainer: saved model auc={:.4f} rows={} sha256={}…",
            auc, len(X), sha256[:12],
        )
        return RetrainResult(True, auc=auc, rows=len(X), duration_s=duration, sha256=sha256)

    except Exception as exc:
        logger.error("ModelRetrainer: retrain failed: {}", exc)
        return RetrainResult(False, reason=str(exc), duration_s=time.time() - t0)


# ── Async service ─────────────────────────────────────────────────────────────

class ModelRetrainer:
    """Periodic background retrainer.

    Every ``retrain_interval_seconds``, it:
    1. Checks there are enough closed trades (``min_trades``)
    2. Loads 15-min candles from SQLite
    3. Retrains LightGBM in a thread executor (non-blocking)
    4. Hot-reloads the MLScorer instance so the live engine uses the
       new model immediately — no restart required
    5. Emits an alert via AlertManager (if wired)
    """

    DEFAULT_INTERVAL = 7 * 24 * 3600  # 7 days
    DEFAULT_MIN_TRADES = 100

    def __init__(
        self,
        ml_scorer: Any,                       # MLScorer instance
        sqlite_path: str = "data/neural_trader.db",
        model_path: str = "models/ml_signal.lgb",
        features_path: str = "models/ml_features.json",
        retrain_interval_seconds: float | None = None,
        min_trades: int | None = None,
        alert_manager: Any = None,
        config: Any = None,
    ) -> None:
        self._ml_scorer = ml_scorer
        self._sqlite_path = Path(sqlite_path)
        self._model_path = Path(model_path)
        self._features_path = Path(features_path)
        self._alert_manager = alert_manager

        # Read from config if provided
        ml_cfg: dict = {}
        if config is not None:
            ml_cfg = (config.get_value("ml_retrainer") or {})

        self._interval = retrain_interval_seconds or float(
            ml_cfg.get("retrain_interval_seconds", self.DEFAULT_INTERVAL)
        )
        self._min_trades = min_trades or int(
            ml_cfg.get("min_trades", self.DEFAULT_MIN_TRADES)
        )
        self._horizon = int(ml_cfg.get("label_horizon_bars", 5))

        self._running = False
        self._last_retrain: float = 0.0
        self._retrain_count: int = 0
        self._last_result: RetrainResult | None = None

    async def run(self) -> None:
        self._running = True
        logger.info(
            "ModelRetrainer started — interval={}h min_trades={}",
            self._interval / 3600, self._min_trades,
        )
        # Stagger first check so startup noise settles
        await asyncio.sleep(300)
        while self._running:
            try:
                await self._maybe_retrain()
            except Exception as exc:
                logger.error("ModelRetrainer loop error: {}", exc)
            # Sleep in small chunks so stop() is responsive
            elapsed = 0.0
            while self._running and elapsed < self._interval:
                await asyncio.sleep(min(60, self._interval - elapsed))
                elapsed += 60

    async def _maybe_retrain(self) -> None:
        now = time.time()
        if now - self._last_retrain < self._interval:
            return

        n_trades = _count_closed_trades(self._sqlite_path)
        if n_trades < self._min_trades:
            logger.info(
                "ModelRetrainer: skipping — only {} closed trades (need {})",
                n_trades, self._min_trades,
            )
            return

        logger.info("ModelRetrainer: starting retrain ({} closed trades)…", n_trades)
        df = _load_candles_sqlite(self._sqlite_path)
        if df is None:
            logger.warning("ModelRetrainer: no candle data in SQLite — skipping")
            return

        # Prefer AdaptiveMLScorer path (WFO + ensemble + calibration)
        adaptive = getattr(self._ml_scorer, "fit_from_dataframe", None)
        if adaptive is not None:
            result = await self._retrain_adaptive(df, adaptive)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _retrain_sync, df, self._model_path, self._features_path, self._horizon
            )

        self._last_result = result
        if result.success:
            self._last_retrain = now
            self._retrain_count += 1
            self._hot_reload()
            await self._send_alert(result)
        else:
            logger.warning("ModelRetrainer: retrain failed — {}", result.reason)

    async def _retrain_adaptive(self, df: pd.DataFrame, fit_fn) -> RetrainResult:
        """Use AdaptiveMLScorer.fit_from_dataframe for full WFO + ensemble retrain."""
        import time as _time
        t0 = _time.time()
        try:
            from engine.learning.features import engineer_features
            feat_df = await asyncio.get_event_loop().run_in_executor(
                None, engineer_features, df
            )
            # Label: price rises more than 1×ATR within horizon bars
            horizon = self._horizon
            atr = feat_df.get("atr_14_pct", feat_df["close"].pct_change().rolling(14).std())
            future_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
            feat_df["label"] = (future_ret > atr).astype(int)
            feat_df = feat_df.dropna(subset=["label"])
            if len(feat_df) > horizon:
                feat_df = feat_df.iloc[:-horizon]

            if len(feat_df) < 300:
                return RetrainResult(False, reason=f"insufficient rows after feature engineering: {len(feat_df)}")

            wfo_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fit_fn(feat_df, label_col="label", n_folds=5, min_train_rows=300),
            )
            if wfo_result is None:
                return RetrainResult(False, reason="AdaptiveMLScorer.fit_from_dataframe returned None")

            return RetrainResult(
                success=True,
                auc=float(wfo_result.auc_mean),
                rows=len(feat_df),
                duration_s=_time.time() - t0,
                reason=f"WFO sharpe={wfo_result.sharpe_mean:.3f} calmar={wfo_result.calmar_mean:.3f}",
            )
        except Exception as exc:
            logger.error("AdaptiveMLScorer retrain failed: {}", exc)
            return RetrainResult(False, reason=str(exc), duration_s=_time.time() - t0)

    def _hot_reload(self) -> None:
        """Hot-reload the scorer without restarting.

        AdaptiveMLScorer: calls reload() which reads the ensemble pkl from disk.
        Legacy MLScorer: directly swaps the LGB booster.
        """
        try:
            if hasattr(self._ml_scorer, "reload"):
                ok = self._ml_scorer.reload()
                logger.info(
                    "ModelRetrainer: scorer hot-reloaded ok={} (retrain #{})",
                    ok, self._retrain_count,
                )
            else:
                # Legacy path: directly swap LGB model
                import lightgbm as lgb
                new_model = lgb.Booster(model_file=str(self._model_path))
                new_feats = json.loads(self._features_path.read_text())
                self._ml_scorer._model    = new_model
                self._ml_scorer._features = new_feats
                self._ml_scorer._load_attempted = True
                logger.info(
                    "ModelRetrainer: LGB hot-reloaded ({} features, retrain #{})",
                    len(new_feats), self._retrain_count,
                )
        except Exception as exc:
            logger.error("ModelRetrainer: hot-reload failed: {}", exc)

    async def _send_alert(self, result: RetrainResult) -> None:
        if self._alert_manager is None:
            return
        try:
            from monitoring.alert_manager import Alert, AlertType, AlertSeverity
            await self._alert_manager.send(Alert(
                alert_type=AlertType.CUSTOM,
                severity=AlertSeverity.INFO,
                title="ML Model Retrained",
                message=(
                    f"New model trained on {result.rows} bars — "
                    f"AUC={result.auc:.4f} in {result.duration_s:.0f}s"
                ),
                metadata={
                    "auc": result.auc,
                    "rows": result.rows,
                    "retrain_count": self._retrain_count,
                    "sha256_prefix": result.sha256[:16],
                },
            ))
        except Exception as exc:
            logger.debug("ModelRetrainer alert failed: {}", exc)

    async def trigger_now(self) -> RetrainResult | None:
        """Force an immediate retrain (bypasses interval check)."""
        self._last_retrain = 0.0
        await self._maybe_retrain()
        return self._last_result

    async def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict[str, Any]:
        next_in = max(0.0, self._interval - (time.time() - self._last_retrain))
        return {
            "enabled": True,
            "retrain_count": self._retrain_count,
            "last_retrain_ts": self._last_retrain or None,
            "next_retrain_in_hours": round(next_in / 3600, 1),
            "interval_hours": round(self._interval / 3600, 1),
            "min_trades": self._min_trades,
            "last_result": {
                "success":    self._last_result.success,
                "auc":        self._last_result.auc,
                "rows":       self._last_result.rows,
                "duration_s": self._last_result.duration_s,
            } if self._last_result else None,
        }
