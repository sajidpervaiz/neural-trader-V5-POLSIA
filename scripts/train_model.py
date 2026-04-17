#!/usr/bin/env python3
"""Train the LightGBM signal model and save to models/ml_signal.lgb.

Usage:
    python scripts/train_model.py [--data-path data/candles.parquet] [--symbols BTC/USDT ETH/USDT]

The script:
1. Loads OHLCV + indicator data from SQLite or a Parquet file
2. Engineers features (returns, volatility, RSI, MACD, BB, ATR, EMA cross)
3. Labels each bar: +1 if next N bars close > entry + threshold, -1 if < entry - threshold, 0 otherwise
4. Trains a LightGBM binary classifier (long vs not-long)
5. Saves model to models/ml_signal.lgb and features to models/ml_features.json
6. Prints the SHA-256 of the saved model (add to ML_MODEL_SHA256 env var)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    # Returns
    df["returns_1"] = c.pct_change(1)
    df["returns_5"] = c.pct_change(5)
    df["returns_10"] = c.pct_change(10)
    df["returns_20"] = c.pct_change(20)

    # Volatility
    df["vol_10"] = df["returns_1"].rolling(10).std()
    df["vol_20"] = df["returns_1"].rolling(20).std()
    df["vol_60"] = df["returns_1"].rolling(60).std()

    # EMA crossover
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()
    df["ema_cross"] = (df["ema_12"] - df["ema_26"]) / c

    # Trend EMAs
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()
    df["trend_50_200"] = (df["ema_50"] - df["ema_200"]) / c

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50  # centre on 0

    # MACD
    macd = df["ema_12"] - df["ema_26"]
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - macd_signal) / c

    # Bollinger Bands %B
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ATR
    h, l = df["high"], df["low"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / c

    # Volume features
    if "volume" in df.columns:
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    else:
        df["vol_ratio"] = 1.0

    return df


def label_bars(df: pd.DataFrame, horizon: int = 5, threshold_atr: float = 1.0) -> pd.DataFrame:
    """Binary label: 1 if price rises > threshold_atr * ATR over next horizon bars."""
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    atr_threshold = df["atr_pct"] * threshold_atr
    df["label"] = (future_return > atr_threshold).astype(int)
    # Drop bars where we can't compute future returns or features are NaN
    df = df.dropna()
    df = df[df.index < df.index[-horizon]]  # drop last horizon bars (no label)
    return df


FEATURE_COLS = [
    "returns_1", "returns_5", "returns_10", "returns_20",
    "vol_10", "vol_20", "vol_60",
    "ema_cross", "trend_50_200",
    "rsi_norm", "macd_hist", "bb_pct", "atr_pct", "vol_ratio",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_sqlite(db_path: str) -> pd.DataFrame | None:
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT time_ns, exchange, symbol, open, high, low, close, volume "
            "FROM candles ORDER BY time_ns",
            conn,
        )
        conn.close()
        df["time"] = pd.to_datetime(df["time_ns"], unit="ns")
        df = df.set_index("time")
        return df
    except Exception as exc:
        print(f"[warn] SQLite load failed: {exc}")
        return None


def load_from_parquet(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as exc:
        print(f"[warn] Parquet load failed: {exc}")
        return None


def generate_synthetic_data(n: int = 10_000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for smoke-testing when no real data is available."""
    print("[info] No real data found — generating synthetic training data for smoke test")
    np.random.seed(42)
    price = 30_000.0
    prices, highs, lows, volumes = [], [], [], []
    for _ in range(n):
        ret = np.random.normal(0, 0.002)
        price *= (1 + ret)
        h = price * (1 + abs(np.random.normal(0, 0.001)))
        l = price * (1 - abs(np.random.normal(0, 0.001)))
        prices.append(price)
        highs.append(h)
        lows.append(l)
        volumes.append(np.random.exponential(100))
    df = pd.DataFrame({"close": prices, "high": highs, "low": lows,
                        "open": prices, "volume": volumes})
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("[error] lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    df = add_features(df)
    df = label_bars(df, horizon=5, threshold_atr=1.0)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features]
    y = df["label"]

    pos_rate = y.mean()
    print(f"[info] Dataset: {len(X)} rows, {len(available_features)} features, "
          f"positive rate={pos_rate:.2%}")

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ml_signal.lgb"
    features_path = output_dir / "ml_features.json"

    model.save_model(str(model_path))
    features_path.write_text(json.dumps(available_features, indent=2))

    # Print SHA-256 for ML_MODEL_SHA256 env var
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    print(f"\n[ok] Model saved to {model_path}")
    print(f"[ok] Features saved to {features_path}")
    print(f"\nAdd this to your .env or environment:")
    print(f"  ML_MODEL_SHA256={sha256}")
    print(f"\nValidation AUC: {model.best_score['valid_0']['auc']:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural-trader ML signal model")
    parser.add_argument("--data-path", default=None, help="Path to parquet file with OHLCV data")
    parser.add_argument("--sqlite-path", default="data/neural_trader.db", help="SQLite DB path")
    parser.add_argument("--output-dir", default="models", help="Directory to save model files")
    parser.add_argument("--horizon", type=int, default=5, help="Label horizon in bars")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df: pd.DataFrame | None = None

    if args.data_path:
        df = load_from_parquet(args.data_path)

    if df is None:
        df = load_from_sqlite(args.sqlite_path)

    if df is None or len(df) < 500:
        df = generate_synthetic_data()

    train(df, output_dir)


if __name__ == "__main__":
    main()
