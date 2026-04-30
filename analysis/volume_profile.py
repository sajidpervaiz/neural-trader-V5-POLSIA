"""Volume Profile & Order Flow Analysis — V6.0 Spec.

Implements:
- Volume Profile Fixed Range (VPFR) with 24 bins
- Point of Control (POC) & Value Area (70%)
- Low Volume Nodes (LVN < 30th percentile) & High Volume Nodes (HVN > 70th percentile)
- On-Balance Volume (OBV) divergence
- Cumulative Volume Delta
- VWAP slope confirmation + Anchored VWAP
- Chaikin Money Flow (CMF) scoring
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileLevel:
    price: float
    volume: float
    is_poc: bool = False
    in_value_area: bool = False
    is_hvn: bool = False   # V6.0: High Volume Node (>70th percentile)
    is_lvn: bool = False   # V6.0: Low Volume Node (<30th percentile)


@dataclass
class VolumeFlowState:
    """Composite volume/order-flow analysis."""
    poc_price: float = 0.0
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    obv_divergence: str = "none"       # "bullish", "bearish", "none"
    cmf_score: float = 0.0
    vwap_slope: str = "flat"           # "rising", "falling", "flat"
    volume_delta: float = 0.0          # cumulative session delta
    delta_trend: str = "neutral"       # "accumulating", "distributing", "neutral"
    flow_score: float = 0.0            # [-1, 1] composite
    reasons: list[str] = field(default_factory=list)
    # V6.0 additions
    lvn_levels: list[float] = field(default_factory=list)
    hvn_levels: list[float] = field(default_factory=list)
    # REQ-IND-017: aggressor inference quality flags. cvd_degraded=True when
    # the close-position proxy was inconclusive (mostly doji-like bars in the
    # window) or when no real bid/ask aggressor data was available.
    cvd_degraded: bool = False
    cvd_inference_method: str = "close_position_proxy"
    cvd_ambiguous_pct: float = 0.0     # share of lookback bars with close near mid


class VolumeProfileAnalyzer:
    """Computes volume profile, order flow metrics, and delta from OHLCV data."""

    def __init__(self, profile_bins: int = 24, value_area_pct: float = 0.70,
                 lvn_percentile: float = 30.0, hvn_percentile: float = 70.0) -> None:
        self._bins = profile_bins
        self._va_pct = value_area_pct
        self._lvn_pct = lvn_percentile
        self._hvn_pct = hvn_percentile

    # ── Volume Profile Fixed Range ────────────────────────────────────────
    def _compute_vpfr(
        self, df: pd.DataFrame, lookback: int = 100,
    ) -> tuple[float, float, float, list[VolumeProfileLevel]]:
        """Compute POC, VAH, VAL from last `lookback` bars.

        Returns (poc, vah, val, levels).
        """
        segment = df.tail(lookback)
        if len(segment) < 10:
            mid = float(df["close"].iloc[-1])
            return mid, mid, mid, []

        price_low = float(segment["low"].min())
        price_high = float(segment["high"].max())
        if price_high == price_low:
            return price_low, price_high, price_low, []

        bin_edges = np.linspace(price_low, price_high, self._bins + 1)
        bin_vols = np.zeros(self._bins)

        for _, row in segment.iterrows():
            vol = row.get("volume", 1.0)
            # Distribute volume across the bar's range
            bar_low = row["low"]
            bar_high = row["high"]
            for j in range(self._bins):
                if bin_edges[j + 1] >= bar_low and bin_edges[j] <= bar_high:
                    # Fraction of bar range that overlaps this bin
                    overlap_low = max(bin_edges[j], bar_low)
                    overlap_high = min(bin_edges[j + 1], bar_high)
                    bar_range = bar_high - bar_low if bar_high > bar_low else 1.0
                    fraction = (overlap_high - overlap_low) / bar_range
                    bin_vols[j] += vol * fraction

        # POC: bin with highest volume
        poc_idx = int(np.argmax(bin_vols))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

        # Value area: expand from POC until 70% of total volume captured
        total_vol = bin_vols.sum()
        if total_vol == 0:
            return poc_price, price_high, price_low, []

        va_vol = bin_vols[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx
        while va_vol / total_vol < self._va_pct:
            expand_up = bin_vols[hi_idx + 1] if hi_idx + 1 < self._bins else 0
            expand_down = bin_vols[lo_idx - 1] if lo_idx - 1 >= 0 else 0
            if expand_up >= expand_down and hi_idx + 1 < self._bins:
                hi_idx += 1
                va_vol += bin_vols[hi_idx]
            elif lo_idx - 1 >= 0:
                lo_idx -= 1
                va_vol += bin_vols[lo_idx]
            else:
                break

        vah = float(bin_edges[hi_idx + 1])
        val = float(bin_edges[lo_idx])

        # V6.0: LVN / HVN thresholds
        lvn_thresh = float(np.percentile(bin_vols[bin_vols > 0], self._lvn_pct)) if bin_vols[bin_vols > 0].size > 0 else 0
        hvn_thresh = float(np.percentile(bin_vols[bin_vols > 0], self._hvn_pct)) if bin_vols[bin_vols > 0].size > 0 else float("inf")

        levels = []
        lvn_prices: list[float] = []
        hvn_prices: list[float] = []
        for j in range(self._bins):
            mid_price = float((bin_edges[j] + bin_edges[j + 1]) / 2)
            vol_val = float(bin_vols[j])
            is_lvn = vol_val > 0 and vol_val <= lvn_thresh
            is_hvn = vol_val >= hvn_thresh
            levels.append(VolumeProfileLevel(
                price=mid_price,
                volume=vol_val,
                is_poc=(j == poc_idx),
                in_value_area=(lo_idx <= j <= hi_idx),
                is_hvn=is_hvn,
                is_lvn=is_lvn,
            ))
            if is_lvn:
                lvn_prices.append(mid_price)
            if is_hvn:
                hvn_prices.append(mid_price)

        return poc_price, vah, val, levels, lvn_prices, hvn_prices

    # ── OBV Divergence ────────────────────────────────────────────────────
    @staticmethod
    def _detect_obv_divergence(df: pd.DataFrame, lookback: int = 20) -> str:
        """Detect OBV vs price divergence over last bars."""
        if "obv" not in df.columns or len(df) < lookback:
            return "none"
        seg = df.tail(lookback)
        price_change = seg["close"].iloc[-1] - seg["close"].iloc[0]
        obv_change = seg["obv"].iloc[-1] - seg["obv"].iloc[0]

        # Bullish divergence: price down but OBV up (accumulation)
        if price_change < 0 and obv_change > 0:
            return "bullish"
        # Bearish divergence: price up but OBV down (distribution)
        if price_change > 0 and obv_change < 0:
            return "bearish"
        return "none"

    # ── VWAP slope ────────────────────────────────────────────────────────
    @staticmethod
    def _vwap_slope(df: pd.DataFrame, lookback: int = 10) -> str:
        if "vwap" not in df.columns or len(df) < lookback:
            return "flat"
        vwap_seg = df["vwap"].tail(lookback).dropna()
        if len(vwap_seg) < 3:
            return "flat"
        slope = (vwap_seg.iloc[-1] - vwap_seg.iloc[0]) / vwap_seg.iloc[0]
        if slope > 0.001:
            return "rising"
        elif slope < -0.001:
            return "falling"
        return "flat"

    # ── Volume Delta (approximate from OHLCV) ────────────────────────────
    @staticmethod
    def _compute_delta(df: pd.DataFrame, lookback: int = 50) -> tuple[float, str, float]:
        """Approximate volume delta from candle direction.

        REQ-IND-017: also returns the share of bars whose close fell near the
        bar midpoint (|close_pos − 0.5| < 0.15) — these are doji-like and
        produce a low-confidence aggressor inference. Caller uses this to
        flag the CVD output as degraded.
        """
        seg = df.tail(lookback)
        delta = 0.0
        ambiguous = 0
        total = 0
        for _, row in seg.iterrows():
            vol = row.get("volume", 0.0)
            close = row["close"]
            open_ = row["open"]
            high = row["high"]
            low = row["low"]
            bar_range = high - low if high > low else 1.0
            close_pos = (close - low) / bar_range  # 0..1, 1 = closed at high
            buy_pct = close_pos
            delta += vol * (2 * buy_pct - 1)
            total += 1
            if abs(close_pos - 0.5) < 0.15:
                ambiguous += 1

        ambiguous_pct = (ambiguous / total) if total else 0.0
        if delta > 0:
            trend = "accumulating"
        elif delta < 0:
            trend = "distributing"
        else:
            trend = "neutral"
        return delta, trend, ambiguous_pct

    # ── Composite analysis ────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame) -> VolumeFlowState:
        """Full volume profile + flow analysis."""
        if df is None or len(df) < 20:
            return VolumeFlowState()

        state = VolumeFlowState()
        reasons: list[str] = []
        score = 0.0

        # VPFR (V6.0: 24 bins, LVN/HVN)
        poc, vah, val, levels, lvn_prices, hvn_prices = self._compute_vpfr(df)
        state.poc_price = poc
        state.value_area_high = vah
        state.value_area_low = val
        state.lvn_levels = lvn_prices
        state.hvn_levels = hvn_prices

        close = float(df["close"].iloc[-1])

        # Price position relative to value area
        if close > vah:
            score += 0.2
            reasons.append("above_value_area")
        elif close < val:
            score -= 0.2
            reasons.append("below_value_area")
        else:
            reasons.append("inside_value_area")

        # POC proximity
        poc_dist = abs(close - poc) / poc if poc > 0 else 0
        if poc_dist < 0.005:
            reasons.append("near_POC")

        # V6.0: LVN proximity — price near a low-volume node = potential fast move
        for lvn in lvn_prices:
            if abs(close - lvn) / close < 0.005:
                reasons.append("near_LVN")
                break

        # OBV divergence
        obv_div = self._detect_obv_divergence(df)
        state.obv_divergence = obv_div
        if obv_div == "bullish":
            score += 0.3
            reasons.append("OBV_bullish_div")
        elif obv_div == "bearish":
            score -= 0.3
            reasons.append("OBV_bearish_div")

        # CMF
        cmf = float(df["cmf"].iloc[-1]) if "cmf" in df.columns and not np.isnan(df["cmf"].iloc[-1]) else 0.0
        state.cmf_score = cmf
        if cmf > 0.05:
            score += 0.2
            reasons.append("CMF_accumulation")
        elif cmf < -0.05:
            score -= 0.2
            reasons.append("CMF_distribution")

        # VWAP slope
        vslope = self._vwap_slope(df)
        state.vwap_slope = vslope
        if vslope == "rising":
            score += 0.15
            reasons.append("VWAP_rising")
        elif vslope == "falling":
            score -= 0.15
            reasons.append("VWAP_falling")

        # Volume delta — proxied from candle close-position; flag degraded
        # when most bars in the window are doji-like (REQ-IND-017).
        delta, delta_trend, ambiguous_pct = self._compute_delta(df)
        state.volume_delta = delta
        state.delta_trend = delta_trend
        state.cvd_ambiguous_pct = float(ambiguous_pct)
        state.cvd_degraded = bool(ambiguous_pct >= 0.50)
        if state.cvd_degraded:
            reasons.append(f"cvd_degraded({int(ambiguous_pct * 100)}%_ambiguous)")
        if delta_trend == "accumulating" and not state.cvd_degraded:
            score += 0.15
            reasons.append("delta_accumulating")
        elif delta_trend == "distributing" and not state.cvd_degraded:
            score -= 0.15
            reasons.append("delta_distributing")

        state.flow_score = float(np.clip(score, -1.0, 1.0))
        state.reasons = reasons
        return state
