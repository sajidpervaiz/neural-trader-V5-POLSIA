"""
Volume Profile & Order Flow Engine — V6 Specification §6

Components:
  • VPFR          — Fixed Range Volume Profile (24 bins, POC/VAH/VAL/LVN/HVN)
  • AnchoredVWAP  — VWAP anchored to any reference bar with slope detection
  • OBVDivergence — Bull/bear divergence detector (min 5-bar span)
  • VolumeDelta   — Approximated buy/sell pressure delta from OHLCV
  • layer5_volume_score — Layer 5 soft score (0-100) per master scorer spec

All functions are pure/stateless (take DataFrames, return values).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── VPFR (Fixed Range Volume Profile) ────────────────────────────────────────

def compute_vpfr(
    df: pd.DataFrame,
    n_bins: int = 24,
    value_area_pct: float = 0.70,
) -> dict[str, Any]:
    """Fixed Range Volume Profile over the given DataFrame slice.

    Distributes each candle's volume across price bins proportional to the
    candle range overlap with each bin (more accurate than close-only).

    Returns dict with:
      poc          : price mid of highest-volume bin
      vah          : value area high boundary
      val          : value area low boundary
      lvn_levels   : list of price mids in <30th pct volume bins
      hvn_levels   : list of price mids in >70th pct volume bins
      bins         : list of {price_mid, price_low, price_high, volume}
      total_volume : total volume in range
      p30_threshold: volume threshold for LVN classification
      p70_threshold: volume threshold for HVN classification
    """
    if df is None or len(df) < 2:
        return {}

    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return {}

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_volumes = np.zeros(n_bins)

    for _, row in df.iterrows():
        c_lo = float(row["low"])
        c_hi = float(row["high"])
        c_vol = float(row.get("volume", 0))
        if c_vol == 0 or np.isnan(c_vol):
            continue
        span = c_hi - c_lo if c_hi > c_lo else 1e-10
        for b in range(n_bins):
            b_lo, b_hi = bin_edges[b], bin_edges[b + 1]
            overlap = max(0.0, min(c_hi, b_hi) - max(c_lo, b_lo))
            bin_volumes[b] += c_vol * (overlap / span)

    # Point of Control: bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    # Value Area (70% of total volume) — expand from POC outward
    total_vol = float(bin_volumes.sum())
    if total_vol == 0:
        return {}
    target_vol = total_vol * value_area_pct

    va_indices = {poc_idx}
    va_vol = bin_volumes[poc_idx]
    lo_ptr, hi_ptr = poc_idx, poc_idx

    while va_vol < target_vol:
        next_lo = lo_ptr - 1 if lo_ptr > 0 else None
        next_hi = hi_ptr + 1 if hi_ptr < n_bins - 1 else None
        vol_lo = bin_volumes[next_lo] if next_lo is not None else -1.0
        vol_hi = bin_volumes[next_hi] if next_hi is not None else -1.0
        if vol_lo < 0 and vol_hi < 0:
            break
        if vol_hi >= vol_lo:
            hi_ptr += 1
            va_indices.add(hi_ptr)
            va_vol += bin_volumes[hi_ptr]
        else:
            lo_ptr -= 1
            va_indices.add(lo_ptr)
            va_vol += bin_volumes[lo_ptr]

    vah = float(bin_edges[max(va_indices) + 1])
    val = float(bin_edges[min(va_indices)])

    # LVN / HVN thresholds
    p30 = float(np.percentile(bin_volumes[bin_volumes > 0], 30)) if (bin_volumes > 0).any() else 0.0
    p70 = float(np.percentile(bin_volumes[bin_volumes > 0], 70)) if (bin_volumes > 0).any() else 0.0

    lvn_levels = [
        float((bin_edges[b] + bin_edges[b + 1]) / 2)
        for b in range(n_bins) if bin_volumes[b] <= p30 and bin_volumes[b] > 0
    ]
    hvn_levels = [
        float((bin_edges[b] + bin_edges[b + 1]) / 2)
        for b in range(n_bins) if bin_volumes[b] >= p70
    ]

    bins = [
        {
            "price_mid": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "price_low": float(bin_edges[b]),
            "price_high": float(bin_edges[b + 1]),
            "volume": float(bin_volumes[b]),
        }
        for b in range(n_bins)
    ]

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "lvn_levels": lvn_levels,
        "hvn_levels": hvn_levels,
        "bins": bins,
        "total_volume": total_vol,
        "p30_threshold": p30,
        "p70_threshold": p70,
    }


def price_in_lvn(price: float, vpfr: dict, tolerance_pct: float = 0.002) -> bool:
    """True if ``price`` is within ±tolerance of any LVN level."""
    for lvn in vpfr.get("lvn_levels", []):
        if price > 0 and abs(price - lvn) / price <= tolerance_pct:
            return True
    return False


def price_near_poc(price: float, vpfr: dict, tolerance_pct: float = 0.005) -> bool:
    """True if ``price`` is within ±0.5 % of the POC."""
    poc = vpfr.get("poc", 0.0)
    if poc <= 0 or price <= 0:
        return False
    return abs(price - poc) / price <= tolerance_pct


def breakout_above_vah(price: float, vpfr: dict) -> bool:
    """True if price has broken above the VAH (requires volume confirmation separately)."""
    return price > vpfr.get("vah", float("inf"))


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def compute_anchored_vwap(
    df: pd.DataFrame,
    anchor_idx: int = 0,
    slope_period: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """VWAP anchored to ``anchor_idx`` (index into df.iloc).

    Returns:
      vwap_series  : pd.Series aligned to df.index (NaN before anchor)
      slope_series : normalised 5-period linear regression slope (rate/price)
    """
    if df is None or len(df) < 2:
        return pd.Series(dtype=float, index=df.index if df is not None else None), \
               pd.Series(dtype=float, index=df.index if df is not None else None)

    sub = df.iloc[anchor_idx:].copy()
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3
    cum_vol = sub["volume"].cumsum()
    cum_tp_vol = (typical * sub["volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Slope: linear regression of VWAP over last slope_period bars, normalised
    slope_vals = pd.Series(0.0, index=vwap.index)
    x = np.arange(slope_period, dtype=float)
    for i in range(slope_period - 1, len(vwap)):
        y = vwap.iloc[i - slope_period + 1:i + 1].values
        if np.isnan(y).any():
            continue
        m = np.polyfit(x, y, 1)[0]
        ref = y[-1] if y[-1] != 0 else 1.0
        slope_vals.iloc[i] = m / ref

    # Re-index to full df (NaN before anchor)
    full_vwap = pd.Series(np.nan, index=df.index)
    full_slope = pd.Series(0.0, index=df.index)
    full_vwap.iloc[anchor_idx:] = vwap.values
    full_slope.iloc[anchor_idx:] = slope_vals.values

    return full_vwap, full_slope


def vwap_signal(vwap: pd.Series, slope: pd.Series, price: float, direction: str) -> bool:
    """True if price + VWAP slope are aligned with trade direction.

    Spec §6.2:
      Longs:  price above VWAP AND slope positive
      Shorts: price below VWAP AND slope negative
    """
    if vwap.empty or slope.empty:
        return False
    vwap_val = float(vwap.iloc[-1])
    slope_val = float(slope.iloc[-1])
    if np.isnan(vwap_val):
        return False
    if direction == "long":
        return price > vwap_val and slope_val > 0
    else:
        return price < vwap_val and slope_val < 0


# ── OBV Divergence ────────────────────────────────────────────────────────────

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Standard On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def detect_obv_divergence(
    close: pd.Series,
    obv: pd.Series,
    min_span: int = 5,
    lookback: int = 50,
) -> dict[str, bool]:
    """Detect bullish/bearish OBV divergence in recent ``lookback`` bars.

    Bullish  — price makes lower low,  OBV makes higher low  (spec §6.3)
    Bearish  — price makes higher high, OBV makes lower high

    Minimum span between extrema: ``min_span`` candles.

    Returns {'bullish': bool, 'bearish': bool}
    """
    result = {"bullish": False, "bearish": False}
    n = min(lookback, len(close))
    if n < min_span * 2 + 2:
        return result

    rc = close.iloc[-n:].values
    ro = obv.iloc[-n:].values

    half = min_span

    def _local_low_indices() -> list[int]:
        out = []
        for i in range(half, n - half):
            window = rc[i - half:i + half + 1]
            if rc[i] <= window.min():
                out.append(i)
        return out

    def _local_high_indices() -> list[int]:
        out = []
        for i in range(half, n - half):
            window = rc[i - half:i + half + 1]
            if rc[i] >= window.max():
                out.append(i)
        return out

    lows = _local_low_indices()
    highs = _local_high_indices()

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if rc[i2] < rc[i1] and ro[i2] > ro[i1]:
            result["bullish"] = True

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if rc[i2] > rc[i1] and ro[i2] < ro[i1]:
            result["bearish"] = True

    return result


# ── Volume Delta ──────────────────────────────────────────────────────────────

def compute_volume_delta(
    open_: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Approximated buy−sell volume delta from OHLCV candle data.

    Method: body bias = (close−open) / |close−open| weighted by volume.
      Positive close → buying pressure
      Negative close → selling pressure
    Returns rolling ``window``-bar cumulative delta.
    """
    body = close - open_
    abs_body = body.abs()
    bias = body / abs_body.replace(0, np.nan).fillna(0)
    delta = bias * volume
    return delta.rolling(window=window, min_periods=1).sum()


def delta_aligned(delta_series: pd.Series, direction: str) -> bool:
    """True if cumulative delta is aligned with ``direction``."""
    if delta_series.empty:
        return False
    val = float(delta_series.iloc[-1])
    return (direction == "long" and val > 0) or (direction == "short" and val < 0)


# ── Layer 5 Volume Confirmation Score ─────────────────────────────────────────

def layer5_volume_score(
    df: pd.DataFrame,
    direction: str,
    vpfr: dict | None = None,
    n_bins: int = 24,
    obv_lookback: int = 50,
) -> float:
    """Layer 5 (Volume Confirmation) soft score, 0–100.

    Scoring rubric (spec §10 Layer 5):
      Entry price in LVN (VPFR)  : +25
      Volume delta aligned        : +25
      VWAP slope aligned          : +20
      CMF aligned (>±0.05)        : +15
      OBV divergence confirmed    : +15
      ──────────────────────────────
      Max                         : 100
    Minimum required to pass: 60 (enforced by layer_validator).
    """
    score = 0.0
    if df is None or len(df) < 10:
        return score

    last = df.iloc[-1]
    price = float(last.get("close", 0))
    if price <= 0:
        return score

    # 1. LVN (+25)
    if vpfr is None and len(df) >= 10:
        vpfr = compute_vpfr(df.tail(min(50, len(df))), n_bins=n_bins)
    if vpfr and price_in_lvn(price, vpfr):
        score += 25.0

    # 2. Volume delta (+25)
    if "open" in df.columns and "volume" in df.columns:
        delta = compute_volume_delta(df["open"], df["close"], df["volume"])
        if delta_aligned(delta, direction):
            score += 25.0

    # 3. VWAP slope (+20) — use pre-computed vwap column if available
    vwap_col = next((c for c in ("vwap", "anchored_vwap") if c in df.columns), None)
    if vwap_col and len(df) >= 5:
        vw = df[vwap_col].dropna()
        if len(vw) >= 5:
            y = vw.iloc[-5:].values
            if not np.isnan(y).any():
                x = np.arange(5, dtype=float)
                m = np.polyfit(x, y, 1)[0]
                slope_norm = m / y[-1] if y[-1] != 0 else 0.0
                if (direction == "long" and slope_norm > 0) or \
                   (direction == "short" and slope_norm < 0):
                    score += 20.0

    # 4. CMF (+15)
    cmf = float(last.get("cmf_20", 0.0))
    if (direction == "long" and cmf > 0.05) or (direction == "short" and cmf < -0.05):
        score += 15.0

    # 5. OBV divergence (+15)
    if "close" in df.columns and "volume" in df.columns:
        obv = compute_obv(df["close"], df["volume"])
        div = detect_obv_divergence(df["close"], obv, lookback=min(obv_lookback, len(df)))
        if direction == "long" and div["bullish"]:
            score += 15.0
        elif direction == "short" and div["bearish"]:
            score += 15.0

    return min(100.0, score)
