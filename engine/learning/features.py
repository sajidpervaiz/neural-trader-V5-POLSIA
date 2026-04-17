"""55+ feature engineering pipeline.

A 20-year trader knows: alpha comes from features the crowd doesn't use.
Beyond basic RSI/MACD, we compute:
  - Multi-lag returns (Fibonacci sequence)
  - Parkinson & Garman-Klass volatility estimators (2-5x more efficient than close-only)
  - Hurst exponent (trend persistence vs mean-reversion)
  - Autocorrelation (serial dependence)
  - Candlestick anatomy (body, shadows, gap)
  - Regime features (one-hot + confidence + duration)
  - Multi-timeframe trend alignment
  - Macro overlay (funding, OI, sentiment)
  - Cyclical time encoding (hour/day of week)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── Feature column lists ──────────────────────────────────────────────────────

# Core price/vol features computed from OHLCV
PRICE_FEATURES = [
    "returns_1", "returns_2", "returns_3", "returns_5", "returns_8",
    "returns_13", "returns_21", "returns_34",
    "log_ret_5", "log_ret_20", "log_ret_60",
    "vol_5", "vol_10", "vol_20", "vol_40", "vol_60",
    "parkinson_vol_14", "garman_klass_vol_14",
    "vol_ratio", "vol_expanding",
    "atr_pct_7", "atr_pct_14", "atr_pct_21",
]

MOMENTUM_FEATURES = [
    "rsi_7", "rsi_14", "rsi_21",
    "rsi_14_slope",
    "rsi_14_norm",
    "macd_hist", "macd_cross",
    "cci_14",
    "williams_r_14",
    "roc_5", "roc_10", "roc_20",
    "stoch_k", "stoch_d",
    "obv_momentum",
]

TREND_FEATURES = [
    "ema_cross_9_21", "ema_cross_50_200",
    "ema200_slope",
    "price_vs_sma20_pct", "price_vs_ema50_pct", "price_vs_ema200_pct",
    "adx_14", "di_spread",
    "hurst_20", "hurst_50",
    "autocorr_1", "autocorr_5",
    "bb_pct", "bb_width_pctile",
    "keltner_squeeze",
]

CANDLE_FEATURES = [
    "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
    "is_bullish", "gap_pct",
    "hl_range_vs_atr",
]

REGIME_FEATURES = [
    "regime_strong_up", "regime_weak_up", "regime_compression",
    "regime_range_chop", "regime_weak_down", "regime_strong_down",
    "regime_confidence", "regime_duration_norm",
]

MACRO_FEATURES = [
    "funding_rate", "funding_sign",
    "oi_change_pct",
    "macro_score", "sentiment_score",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

MULTI_TF_FEATURES = [
    "htf_1h_trend", "htf_4h_trend",
    "htf_1h_rsi_norm", "htf_4h_rsi_norm",
    "htf_1h_adx", "htf_4h_adx",
]

FEATURE_COLS: list[str] = (
    PRICE_FEATURES + MOMENTUM_FEATURES + TREND_FEATURES
    + CANDLE_FEATURES + REGIME_FEATURES + MACRO_FEATURES
    + TIME_FEATURES + MULTI_TF_FEATURES
)


# ── Individual feature computers ──────────────────────────────────────────────

def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """High-low range estimator — 5× more efficient than close-only vol."""
    ln_hl = np.log(high / low.replace(0, np.nan))
    return (ln_hl ** 2 / (4 * np.log(2))).rolling(window).mean() ** 0.5


def _garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14,
) -> pd.Series:
    """Garman-Klass OHLC estimator — most efficient close-to-close analogue."""
    c1 = 0.5 * np.log(high / low.replace(0, np.nan)) ** 2
    c2 = (2 * np.log(2) - 1) * np.log(close / open_.replace(0, np.nan)) ** 2
    gk = (c1 - c2).rolling(window).mean()
    return gk.clip(lower=0) ** 0.5


def _hurst(series: pd.Series, lags: int = 20) -> float:
    """Hurst exponent via R/S analysis.
    >0.5 = trending, <0.5 = mean-reverting, ~0.5 = random walk.
    """
    prices = series.dropna().values
    if len(prices) < lags + 5:
        return 0.5
    try:
        ts = [2, 4, 8, 16, min(lags, len(prices) // 4)]
        rs_vals = []
        for lag in ts:
            sub = prices[-lag * 4:]
            if len(sub) < lag:
                continue
            chunks = [sub[i:i + lag] for i in range(0, len(sub) - lag + 1, lag)]
            rs_list = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                m = np.mean(chunk)
                dev = np.cumsum(chunk - m)
                r = dev.max() - dev.min()
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                rs_vals.append((lag, np.mean(rs_list)))
        if len(rs_vals) < 2:
            return 0.5
        lags_arr = np.log([x[0] for x in rs_vals])
        rs_arr   = np.log([x[1] for x in rs_vals])
        poly = np.polyfit(lags_arr, rs_arr, 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low  = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tp  = (high + low + close) / 3
    ma  = tp.rolling(period).mean()
    md  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _obv_momentum(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv.diff(period) / volume.rolling(period).mean().replace(0, np.nan)


# ── Main feature engineering function ────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    regime_state: Any = None,
    htf_dfs: dict[str, pd.DataFrame] | None = None,
    macro_context: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute all 55+ features and return augmented DataFrame.

    Args:
        df: Primary timeframe OHLCV DataFrame (already has basic indicators).
        regime_state: RegimeState from ARMS regime detector.
        htf_dfs: Higher timeframe DataFrames {timeframe: df}.
        macro_context: Live macro values (funding_rate, oi_change_pct, etc.).

    Returns:
        DataFrame with all FEATURE_COLS columns added (NaN-safe).
    """
    if df is None or len(df) < 50:
        return df

    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]
    o = out.get("open", c)
    v = out.get("volume", pd.Series(1.0, index=out.index))

    # ── Multi-lag returns (Fibonacci) ────────────────────────────────────
    for lag in [1, 2, 3, 5, 8, 13, 21, 34]:
        col = f"returns_{lag}"
        if col not in out.columns:
            out[col] = c.pct_change(lag)

    out["log_ret_5"]  = np.log(c / c.shift(5))
    out["log_ret_20"] = np.log(c / c.shift(20))
    out["log_ret_60"] = np.log(c / c.shift(60))

    # ── Volatility estimators ─────────────────────────────────────────────
    for w in [5, 10, 20, 40, 60]:
        col = f"vol_{w}"
        if col not in out.columns:
            out[col] = out["returns_1"].rolling(w).std() if "returns_1" in out.columns else c.pct_change().rolling(w).std()

    out["parkinson_vol_14"]    = _parkinson_vol(h, l, 14)
    out["garman_klass_vol_14"] = _garman_klass_vol(o, h, l, c, 14)

    if "vol_ratio" not in out.columns:
        ret1 = c.pct_change()
        rolling_vol = ret1.rolling(20).std()
        out["vol_ratio"] = rolling_vol / rolling_vol.rolling(60).mean().replace(0, np.nan)
    out["vol_expanding"] = (out.get("vol_20", c.pct_change().rolling(20).std())
                            > out.get("vol_20", c.pct_change().rolling(20).std()).shift(5)).astype(float)

    # ── ATR% at multiple periods ──────────────────────────────────────────
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    for period in [7, 14, 21]:
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        out[f"atr_pct_{period}"] = atr / c.replace(0, np.nan)

    # ── RSI at multiple periods ───────────────────────────────────────────
    delta = c.diff()
    for period in [7, 14, 21]:
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        out[f"rsi_{period}"] = rsi

    if "rsi_14" not in out.columns:
        out["rsi_14"] = out["rsi_14"]  # already computed above
    out["rsi_14_slope"] = out["rsi_14"].diff(3) / 3
    out["rsi_14_norm"]  = (out["rsi_14"] - 50) / 50

    # ── MACD ─────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    if "macd_hist" not in out.columns:
        macd_sig     = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"]  = (macd - macd_sig) / c.replace(0, np.nan)
    out["macd_cross"] = ((macd > macd.ewm(span=9, adjust=False).mean()) &
                         (macd.shift(1) <= macd.ewm(span=9, adjust=False).mean().shift(1))).astype(float)

    # ── CCI, Williams %R, ROC ─────────────────────────────────────────────
    out["cci_14"]       = _cci(h, l, c, 14).clip(-5, 5) / 5  # normalised
    out["williams_r_14"] = ((h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan) - 0.5) * 2
    for period in [5, 10, 20]:
        out[f"roc_{period}"] = c.pct_change(period)

    # ── Stochastic ────────────────────────────────────────────────────────
    sk, sd = _stochastic(h, l, c)
    out["stoch_k"] = (sk - 50) / 50
    out["stoch_d"] = (sd - 50) / 50

    # ── OBV Momentum ─────────────────────────────────────────────────────
    out["obv_momentum"] = _obv_momentum(c, v)

    # ── EMA crossovers ────────────────────────────────────────────────────
    ema9  = c.ewm(span=9,   adjust=False).mean()
    ema21 = c.ewm(span=21,  adjust=False).mean()
    ema50 = c.ewm(span=50,  adjust=False).mean()
    ema200= c.ewm(span=200, adjust=False).mean()

    out["ema_cross_9_21"]   = (ema9 - ema21) / c.replace(0, np.nan)
    out["ema_cross_50_200"] = (ema50 - ema200) / c.replace(0, np.nan)
    if "ema200_slope" not in out.columns:
        out["ema200_slope"] = (ema200 - ema200.shift(10)) / ema200.shift(10).replace(0, np.nan)

    # ── Price vs key EMAs ─────────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    out["price_vs_sma20_pct"]  = (c - sma20) / sma20.replace(0, np.nan)
    out["price_vs_ema50_pct"]  = (c - ema50) / ema50.replace(0, np.nan)
    out["price_vs_ema200_pct"] = (c - ema200) / ema200.replace(0, np.nan)

    # ── ADX + DI spread ───────────────────────────────────────────────────
    dm_plus  = (h - h.shift(1)).clip(lower=0)
    dm_minus = (l.shift(1) - l).clip(lower=0)
    mask = dm_plus <= dm_minus; dm_plus[mask] = 0
    mask2 = dm_minus <= dm_plus; dm_minus[mask2] = 0
    atr14 = tr.ewm(com=13, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    if "adx_14" not in out.columns:
        out["adx_14"] = dx.ewm(com=13, adjust=False).mean()
    out["di_spread"] = (di_plus - di_minus) / 100

    # ── Hurst exponent (rolling approximation) ────────────────────────────
    out["hurst_20"] = c.rolling(60).apply(lambda x: _hurst(pd.Series(x), 20), raw=False)
    out["hurst_50"] = c.rolling(150).apply(lambda x: _hurst(pd.Series(x), 50), raw=False)

    # ── Autocorrelation ───────────────────────────────────────────────────
    ret1 = c.pct_change()
    out["autocorr_1"] = ret1.rolling(30).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    out["autocorr_5"] = ret1.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=5), raw=False)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    if "bb_pct" not in out.columns:
        out["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std.replace(0, np.nan))
    bb_width = 2 * bb_std / bb_mid.replace(0, np.nan)
    if "bb_width_pctile" not in out.columns:
        out["bb_width_pctile"] = bb_width.rolling(200, min_periods=20).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False
        )

    # ── Keltner squeeze ───────────────────────────────────────────────────
    kc_atr = atr14
    ema20  = c.ewm(span=20, adjust=False).mean()
    kc_u   = ema20 + 1.5 * kc_atr
    kc_l   = ema20 - 1.5 * kc_atr
    bb_u   = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    if "keltner_squeeze" not in out.columns:
        out["keltner_squeeze"] = ((bb_u < kc_u) & (bb_lo > kc_l)).astype(float)

    # ── Candlestick anatomy ───────────────────────────────────────────────
    body   = (c - o).abs()
    hi_lo  = (h - l).replace(0, np.nan)
    atr_ref = atr14.replace(0, np.nan)
    out["body_ratio"]         = body / hi_lo
    out["upper_shadow_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / atr_ref
    out["lower_shadow_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / atr_ref
    out["is_bullish"]         = (c > o).astype(float)
    out["gap_pct"]            = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)
    out["hl_range_vs_atr"]    = hi_lo / atr_ref

    # ── Regime features (one-hot + metadata) ─────────────────────────────
    regime_cols = {
        "regime_strong_up": 0.0, "regime_weak_up": 0.0, "regime_compression": 0.0,
        "regime_range_chop": 0.0, "regime_weak_down": 0.0, "regime_strong_down": 0.0,
        "regime_confidence": 0.5, "regime_duration_norm": 0.0,
    }
    if regime_state is not None:
        regime_name = getattr(regime_state, "regime", None)
        if regime_name is not None:
            rv = str(regime_name.value) if hasattr(regime_name, "value") else str(regime_name)
            if "strong_trend_up"   in rv: regime_cols["regime_strong_up"]    = 1.0
            elif "weak_trend_up"   in rv: regime_cols["regime_weak_up"]      = 1.0
            elif "compression"     in rv: regime_cols["regime_compression"]  = 1.0
            elif "range_chop"      in rv: regime_cols["regime_range_chop"]   = 1.0
            elif "weak_trend_down" in rv: regime_cols["regime_weak_down"]    = 1.0
            elif "strong_trend_down" in rv: regime_cols["regime_strong_down"] = 1.0
        regime_cols["regime_confidence"]    = float(getattr(regime_state, "confidence", 0.5))
        dur = float(getattr(regime_state, "candles_in_state", 0))
        regime_cols["regime_duration_norm"] = min(1.0, dur / 50)

    for col, val in regime_cols.items():
        out[col] = val

    # ── Macro features ────────────────────────────────────────────────────
    mc = macro_context or {}
    out["funding_rate"]  = float(mc.get("funding_rate", 0.0))
    out["funding_sign"]  = float(np.sign(mc.get("funding_rate", 0.0)))
    out["oi_change_pct"] = float(mc.get("oi_change_pct", 0.0))
    out["macro_score"]   = float(mc.get("macro_score", 0.0))
    out["sentiment_score"] = float(mc.get("sentiment_score", 0.0))

    # ── Time features (cyclical) ──────────────────────────────────────────
    if hasattr(out.index, "hour"):
        hour = out.index.hour
        dow  = out.index.dayofweek
    else:
        import time as _time
        now = _time.localtime()
        hour = pd.Series([now.tm_hour] * len(out), index=out.index)
        dow  = pd.Series([now.tm_wday] * len(out), index=out.index)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    # ── Multi-timeframe features ──────────────────────────────────────────
    for col in MULTI_TF_FEATURES:
        out[col] = 0.0

    if htf_dfs:
        for tf_label, tf_key in [("1h", "1h"), ("4h", "4h")]:
            htf = htf_dfs.get(tf_key)
            if htf is not None and len(htf) > 5:
                htf_c = htf["close"]
                htf_e12 = htf_c.ewm(span=12, adjust=False).mean()
                htf_e26 = htf_c.ewm(span=26, adjust=False).mean()
                trend_val = float(np.sign(htf_e12.iloc[-1] - htf_e26.iloc[-1]))
                out[f"htf_{tf_label}_trend"] = trend_val

                htf_delta = htf_c.diff()
                htf_gain  = htf_delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                htf_loss  = (-htf_delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                htf_rs    = htf_gain / htf_loss.replace(0, np.nan)
                htf_rsi   = 100 - 100 / (1 + htf_rs)
                out[f"htf_{tf_label}_rsi_norm"] = float((htf_rsi.iloc[-1] - 50) / 50) if not htf_rsi.empty else 0.0

                htf_h = htf["high"]; htf_l = htf["low"]
                htf_tr = pd.concat([htf_h - htf_l, (htf_h - htf_c.shift()).abs(), (htf_l - htf_c.shift()).abs()], axis=1).max(axis=1)
                htf_atr = htf_tr.ewm(com=13, adjust=False).mean()
                htf_dm_p = (htf_h - htf_h.shift()).clip(lower=0)
                htf_dm_m = (htf_l.shift() - htf_l).clip(lower=0)
                m1 = htf_dm_p <= htf_dm_m; htf_dm_p[m1] = 0
                m2 = htf_dm_m <= htf_dm_p; htf_dm_m[m2] = 0
                htf_dip = 100 * htf_dm_p.ewm(com=13, adjust=False).mean() / htf_atr.replace(0, np.nan)
                htf_dim = 100 * htf_dm_m.ewm(com=13, adjust=False).mean() / htf_atr.replace(0, np.nan)
                htf_dx  = 100 * (htf_dip - htf_dim).abs() / (htf_dip + htf_dim).replace(0, np.nan)
                htf_adx = htf_dx.ewm(com=13, adjust=False).mean()
                out[f"htf_{tf_label}_adx"] = float(htf_adx.iloc[-1]) / 100 if not htf_adx.empty else 0.0

    # ── Final cleanup — clip extreme values, fill NaN ─────────────────────
    feature_cols_present = [c for c in FEATURE_COLS if c in out.columns]
    out[feature_cols_present] = out[feature_cols_present].replace([np.inf, -np.inf], np.nan)
    # Clip gross outliers (5σ)
    for col in feature_cols_present:
        if out[col].std() > 0:
            mu, sigma = out[col].mean(), out[col].std()
            out[col] = out[col].clip(mu - 5 * sigma, mu + 5 * sigma)
    out[feature_cols_present] = out[feature_cols_present].fillna(0)
    return out
