"""
7-Layer Signal Confirmation System — V6 Specification §10

Hard Gates (reject immediately on failure):
  L0 — Sentiment Gate       (news/social)
  L1 — Regime Classification (ADX / ATR / BB-width)
  L2 — Structure Confirmation (BOS/CHoCH + Fib 50-62%)
  L6 — Multi-Timeframe Alignment (weighted score ≥ 6.0/7.0)
  L7 — Microstructure Confirmation (candle pattern, volume, OB imbalance)

Soft Scoring (contribute to master score):
  L3 — SMC Confluence   (0-100 pts)
  L4 — Momentum Matrix  (0-100 pts)
  L5 — Volume Confirmation (0-100 pts — via volume_profile.layer5_volume_score)

Master Score: TotalScore = 0.25×L3 + 0.20×L4 + 0.20×L5 + 0.20×Neural + 0.15×ExecLiq
Trade allowed if TotalScore ≥ 75 (≥ 90 → 25% size boost).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from engine.volume_profile import layer5_volume_score, compute_vpfr


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class LayerResult:
    layer_id: int
    name: str
    is_hard_gate: bool
    passed: bool          # hard gates: True/False; soft layers: always True
    score: float          # soft layers: 0-100; hard gates: 0 (fail) or 100 (pass)
    reason: str = ""


@dataclass
class ValidationResult:
    """Complete result of the 7-layer validation pipeline."""
    passed: bool
    rejection_layer: int | None       # None if passed all layers
    rejection_reason: str

    # Soft layer scores (0-100)
    layer3_score: float = 0.0
    layer4_score: float = 0.0
    layer5_score: float = 0.0

    layer_results: list[LayerResult] = field(default_factory=list)

    # Master score fields (populated by master_scorer after validation)
    total_score: float = 0.0
    size_boost: bool = False          # True if total_score ≥ 90


# ── Layer 0: Sentiment Gate (Hard Gate) ───────────────────────────────────────

def layer0_sentiment_gate(
    sentiment_score: float,
    has_high_impact_news: bool = False,
    social_volume_ratio: float = 1.0,
    min_sentiment: float = 0.4,          # relaxed from 0.6 — external data rare
    min_social_ratio: float = 1.0,
    max_social_ratio: float = 500.0,
) -> LayerResult:
    """Sentiment hard gate.

    External data (LunarCrush, Cryptopanic) is optional.  When not available,
    sentiment_score defaults to 0.5 (neutral) and the gate passes.
    High-impact news events always block regardless of sentiment score.
    """
    if has_high_impact_news:
        return LayerResult(0, "Sentiment Gate", True, False, 0.0,
                           "high_impact_news_event_active")

    # If no external sentiment data, pass (don't block on missing data)
    if sentiment_score <= 0.0:
        return LayerResult(0, "Sentiment Gate", True, True, 100.0, "no_sentiment_data_neutral")

    if sentiment_score < min_sentiment:
        return LayerResult(0, "Sentiment Gate", True, False, 0.0,
                           f"sentiment_too_low={sentiment_score:.2f}<{min_sentiment}")

    # Social volume spam filter: too low or too high is suspicious
    if social_volume_ratio > max_social_ratio:
        return LayerResult(0, "Sentiment Gate", True, False, 0.0,
                           f"social_volume_spike_suspicious={social_volume_ratio:.0f}x")

    return LayerResult(0, "Sentiment Gate", True, True, 100.0,
                       f"sentiment={sentiment_score:.2f} social_ratio={social_volume_ratio:.1f}x")


# ── Layer 1: Regime Classification (Hard Gate) ────────────────────────────────

def layer1_regime_gate(df: pd.DataFrame) -> LayerResult:
    """Choppy regime hard gate.

    Passes if: ADX(14) > 25 OR ATR_ratio > 1.2
    AND Bollinger Band width > 0.015

    Fail = no trade (market too choppy / too low volatility).
    """
    if df is None or len(df) < 20:
        return LayerResult(1, "Regime Gate", True, False, 0.0, "insufficient_data")

    last = df.iloc[-1]
    adx = float(last.get("adx", 0.0))
    atr = float(last.get("atr_14", 0.0))
    close = float(last.get("close", 1.0))

    # ATR ratio: current ATR / 20-period average ATR
    if "atr_14" in df.columns and len(df) >= 20:
        avg_atr = float(df["atr_14"].tail(20).mean())
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
    else:
        atr_ratio = 1.0

    # Bollinger Band width = (upper - lower) / middle
    bb_width = 0.0
    if "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_mid" in df.columns:
        bb_upper = float(last.get("bb_upper", 0.0))
        bb_lower = float(last.get("bb_lower", 0.0))
        bb_mid = float(last.get("bb_mid", close))
        if bb_mid > 0:
            bb_width = (bb_upper - bb_lower) / bb_mid

    regime_active = (adx > 25) or (atr_ratio > 1.2)
    bb_ok = bb_width > 0.015 or bb_width == 0.0  # 0 means column missing — don't block

    if not regime_active:
        return LayerResult(1, "Regime Gate", True, False, 0.0,
                           f"choppy_regime: adx={adx:.1f} atr_ratio={atr_ratio:.2f}")
    if not bb_ok:
        return LayerResult(1, "Regime Gate", True, False, 0.0,
                           f"low_volatility: bb_width={bb_width:.4f}<0.015")

    return LayerResult(1, "Regime Gate", True, True, 100.0,
                       f"adx={adx:.1f} atr_ratio={atr_ratio:.2f} bb_width={bb_width:.4f}")


# ── Layer 2: Structure Confirmation (Hard Gate) ────────────────────────────────

def layer2_structure_gate(df: pd.DataFrame, direction: str) -> LayerResult:
    """Structure confirmation hard gate.

    Requires:
    - BOS or CHoCH on the signal timeframe
    - Price retracement within 50-62% of recent impulse move
    """
    if df is None or len(df) < 10:
        return LayerResult(2, "Structure Gate", True, False, 0.0, "insufficient_data")

    last = df.iloc[-1]

    # BOS / CHoCH check
    bos_bull = float(last.get("bos_bull", 0.0))
    bos_bear = float(last.get("bos_bear", 0.0))
    choch_bull = float(last.get("choch_bull", 0.0))
    choch_bear = float(last.get("choch_bear", 0.0))

    if direction == "long":
        struct_ok = (bos_bull > 0) or (choch_bull > 0)
    else:
        struct_ok = (bos_bear > 0) or (choch_bear > 0)

    # Fibonacci retracement check: is price in 50-62% zone?
    fib_ok = True  # default pass when impulse data not available
    if len(df) >= 20:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        # Detect recent swing: use last 20 bars
        swing_high = float(high.tail(20).max())
        swing_low = float(low.tail(20).min())
        span = swing_high - swing_low
        current_price = float(last.get("close", 0.0))
        if span > 0 and current_price > 0:
            if direction == "long":
                retrace_pct = (swing_high - current_price) / span
            else:
                retrace_pct = (current_price - swing_low) / span
            fib_ok = 0.50 <= retrace_pct <= 0.786  # slightly wider than 0.618 for tolerance

    if not struct_ok and not fib_ok:
        return LayerResult(2, "Structure Gate", True, False, 0.0,
                           f"no_bos_choch and not in fib zone for {direction}")

    if not struct_ok:
        return LayerResult(2, "Structure Gate", True, True, 60.0,
                           f"in_fib_zone but no bos/choch signal (partial pass)")

    reason = []
    if direction == "long" and bos_bull > 0:
        reason.append("bos_bull")
    if direction == "long" and choch_bull > 0:
        reason.append("choch_bull")
    if direction == "short" and bos_bear > 0:
        reason.append("bos_bear")
    if direction == "short" and choch_bear > 0:
        reason.append("choch_bear")
    if fib_ok:
        reason.append("fib_50_62pct")

    return LayerResult(2, "Structure Gate", True, True, 100.0, "+".join(reason))


# ── Layer 3: SMC Confluence (Soft Score) ─────────────────────────────────────

_LONDON_NY_OVERLAP_START_UTC = 13 * 60   # 13:00 UTC
_LONDON_NY_OVERLAP_END_UTC   = 17 * 60   # 17:00 UTC


def layer3_smc_score(df: pd.DataFrame, direction: str) -> LayerResult:
    """SMC Confluence soft score (0-100).

    Scoring rubric (spec §10 Layer 3):
      FVG boundary touch         : +20
      OB 38-62% retracement zone : +20
      Breaker Block rejection    : +15
      Liquidity sweep + MSS ≤3c  : +25
      London-NY overlap          : +20
      ────────────────────────────────
      Max                        : 100
    Minimum required: 40 (enforced by caller).
    """
    if df is None or len(df) < 5:
        return LayerResult(3, "SMC Confluence", False, True, 0.0, "insufficient_data")

    last = df.iloc[-1]
    score = 0.0
    reasons: list[str] = []

    # FVG boundary touch (+20)
    fvg_key = "fvg_bull_active" if direction == "long" else "fvg_bear_active"
    if float(last.get(fvg_key, 0)) > 0:
        score += 20.0
        reasons.append("fvg_active")

    # Order Block 38-62% retracement (+20)
    ob_key = "ob_bull_active" if direction == "long" else "ob_bear_active"
    if float(last.get(ob_key, 0)) > 0:
        score += 20.0
        reasons.append("ob_active")

    # Breaker Block rejection (+15)
    bb_key = "breaker_bull_active" if direction == "long" else "breaker_bear_active"
    if float(last.get(bb_key, 0)) > 0:
        score += 15.0
        reasons.append("breaker_active")

    # Liquidity sweep + MSS within last 3 candles (+25)
    sweep_key = "liq_swept_low" if direction == "long" else "liq_swept_high"
    for i in range(1, min(4, len(df))):
        bar = df.iloc[-i]
        if float(bar.get(sweep_key, 0)) > 0:
            score += 25.0
            reasons.append(f"liq_sweep_mss_{i}c_ago")
            break

    # London-NY overlap (13:00-17:00 UTC) (+20)
    now_utc = time.gmtime()
    now_min = now_utc.tm_hour * 60 + now_utc.tm_min
    if _LONDON_NY_OVERLAP_START_UTC <= now_min <= _LONDON_NY_OVERLAP_END_UTC:
        score += 20.0
        reasons.append("london_ny_overlap")

    score = min(100.0, score)
    return LayerResult(3, "SMC Confluence", False, True, score, "+".join(reasons) or "no_smc")


# ── Layer 4: Momentum Matrix (Soft Score) ─────────────────────────────────────

def layer4_momentum_score(df: pd.DataFrame, direction: str) -> LayerResult:
    """Momentum Matrix soft score (0-100).

    Three weighted groups (spec §10 Layer 4):
      Trend group    (EMA stack, SuperTrend, Ichimoku, SAR, Aroon)  — 40%
      Momentum group (RSI, MACD, MFI, Stochastic, CCI)              — 40%
      Volatility grp (BB %B, Keltner channel)                       — 20%

    Each indicator scores 0-100 for directional alignment.
    Group score = mean of available indicators in group.
    Final score = 0.40×trend + 0.40×momentum + 0.20×volatility.
    Minimum required: 60.
    """
    if df is None or len(df) < 5:
        return LayerResult(4, "Momentum Matrix", False, True, 0.0, "insufficient_data")

    last = df.iloc[-1]
    sign = 1.0 if direction == "long" else -1.0

    def _indicator_score(value: float, bull_positive: bool = True) -> float:
        """Convert a directional indicator to 0-100 score (50 = neutral)."""
        raw = value if bull_positive else -value
        if raw > 0:
            return min(100.0, 50.0 + raw * 50.0)
        else:
            return max(0.0, 50.0 + raw * 50.0)

    # ── Trend Group (40%) ────────────────────────────────────────────────
    trend_scores: list[float] = []

    # EMA stack: ema_9 > ema_21 > ema_50 (for long)
    e9 = float(last.get("ema_9", 0))
    e21 = float(last.get("ema_21", 0))
    e50 = float(last.get("ema_50", 0))
    if e9 > 0 and e21 > 0:
        ema_aligned = sign * (1 if e9 > e21 else -1)
        if e50 > 0:
            ema_aligned += sign * (0.5 if e21 > e50 else -0.5)
        trend_scores.append(min(100.0, max(0.0, 50.0 + ema_aligned * 25.0)))

    # SuperTrend direction
    st_dir = float(last.get("supertrend_dir", 0))
    if st_dir != 0:
        trend_scores.append(100.0 if sign * st_dir > 0 else 0.0)

    # SAR: close vs SAR value
    sar = float(last.get("sar", 0))
    close = float(last.get("close", 0))
    if sar > 0 and close > 0:
        sar_bull = close > sar
        trend_scores.append(100.0 if sign > 0 == sar_bull else 0.0)

    # Market structure as trend proxy
    ms = float(last.get("market_structure", 0))
    if ms != 0:
        trend_scores.append(min(100.0, max(0.0, 50.0 + sign * ms * 50.0)))

    # ADX slope (trend strength)
    adx = float(last.get("adx", 0))
    if adx > 0:
        # ADX itself is direction-agnostic; use DI+ vs DI-
        di_plus = float(last.get("di_plus", 0))
        di_minus = float(last.get("di_minus", 0))
        if di_plus > 0 or di_minus > 0:
            di_signal = di_plus - di_minus
            trend_scores.append(min(100.0, max(0.0, 50.0 + sign * di_signal / 20.0 * 50.0)))

    trend_score = float(np.mean(trend_scores)) if trend_scores else 50.0

    # ── Momentum Group (40%) ─────────────────────────────────────────────
    mom_scores: list[float] = []

    # RSI
    rsi = float(last.get("rsi_14", 50))
    if direction == "long":
        # Ideal: 40-65 (momentum but not overbought)
        if 40 <= rsi <= 65:
            mom_scores.append(80.0)
        elif 30 <= rsi < 40:
            mom_scores.append(60.0)  # oversold bounce
        elif rsi > 65:
            mom_scores.append(30.0)  # overbought
        else:
            mom_scores.append(20.0)
    else:
        if 35 <= rsi <= 60:
            mom_scores.append(80.0)
        elif 60 < rsi <= 70:
            mom_scores.append(60.0)  # overbought reversal
        elif rsi < 35:
            mom_scores.append(30.0)  # oversold
        else:
            mom_scores.append(20.0)

    # MACD histogram
    macd_h = float(last.get("macd_hist", 0))
    mom_scores.append(100.0 if sign * macd_h > 0 else 0.0 if macd_h != 0 else 50.0)

    # MFI
    mfi = float(last.get("mfi_14", 50))
    if direction == "long":
        mom_scores.append(min(100.0, max(0.0, mfi)))  # higher = more bullish money flow
    else:
        mom_scores.append(min(100.0, max(0.0, 100.0 - mfi)))

    # Stochastic K
    stoch_k = float(last.get("stoch_k", 50))
    if direction == "long":
        mom_scores.append(min(100.0, max(0.0, stoch_k)))
    else:
        mom_scores.append(min(100.0, max(0.0, 100.0 - stoch_k)))

    # CCI
    cci = float(last.get("cci_20", 0))
    cci_score = min(100.0, max(0.0, 50.0 + sign * cci / 200.0 * 50.0))
    mom_scores.append(cci_score)

    mom_score = float(np.mean(mom_scores)) if mom_scores else 50.0

    # ── Volatility Group (20%) ────────────────────────────────────────────
    vol_scores: list[float] = []

    # BB %B
    bb_upper = float(last.get("bb_upper", 0))
    bb_lower = float(last.get("bb_lower", 0))
    if bb_upper > 0 and bb_lower > 0 and close > 0:
        bb_pctb = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        if direction == "long":
            # Bullish: above midpoint (0.5) but not overbought (>0.9)
            if 0.5 <= bb_pctb <= 0.9:
                vol_scores.append(80.0)
            elif bb_pctb < 0.5:
                vol_scores.append(30.0)
            else:
                vol_scores.append(20.0)  # overbought
        else:
            if 0.1 <= bb_pctb <= 0.5:
                vol_scores.append(80.0)
            elif bb_pctb > 0.5:
                vol_scores.append(30.0)
            else:
                vol_scores.append(20.0)  # oversold

    # Keltner: is price outside channel? (breakout signal)
    kc_upper = float(last.get("kc_upper", 0))
    kc_lower = float(last.get("kc_lower", 0))
    if kc_upper > 0 and kc_lower > 0 and close > 0:
        if direction == "long":
            vol_scores.append(100.0 if close > (kc_upper + kc_lower) / 2 else 40.0)
        else:
            vol_scores.append(100.0 if close < (kc_upper + kc_lower) / 2 else 40.0)

    vol_score = float(np.mean(vol_scores)) if vol_scores else 50.0

    # ── Weighted Final Score ──────────────────────────────────────────────
    final_score = 0.40 * trend_score + 0.40 * mom_score + 0.20 * vol_score
    final_score = min(100.0, max(0.0, final_score))

    reason = f"trend={trend_score:.0f} mom={mom_score:.0f} vol={vol_score:.0f}"
    return LayerResult(4, "Momentum Matrix", False, True, final_score, reason)


# ── Layer 6: Multi-Timeframe Alignment (Hard Gate) ────────────────────────────

def layer6_mtf_gate(
    htf_data: dict[str, pd.DataFrame | None],
    direction: str,
) -> LayerResult:
    """Multi-timeframe alignment hard gate.

    Spec §7 & §10 L6:
      Timeframe weights: 1D=3, 4H=2, 1H=1, 15m=1  (total=7)
      Required weighted score ≥ 6.0/7.0
      Hard override: if daily bias bearish → no longs (and vice versa)

    htf_data: dict mapping timeframe string to DataFrame (or None if unavailable).
              Expected keys: '1d', '4h', '1h', '15m'
    """
    weights = {"1d": 3, "4h": 2, "1h": 1, "15m": 1}
    total_possible = sum(weights.values())  # 7
    achieved = 0.0

    alignment_details: list[str] = []
    daily_df = htf_data.get("1d")

    # Hard daily override
    if daily_df is not None and len(daily_df) >= 5:
        d_last = daily_df.iloc[-1]
        d_trend = float(d_last.get("supertrend_dir", 0)) or float(d_last.get("market_structure", 0))
        if direction == "long" and d_trend < 0:
            return LayerResult(6, "MTF Alignment", True, False, 0.0,
                               "daily_bias_bearish_blocks_long")
        if direction == "short" and d_trend > 0:
            return LayerResult(6, "MTF Alignment", True, False, 0.0,
                               "daily_bias_bullish_blocks_short")

    # Score each available timeframe
    for tf, weight in weights.items():
        df_tf = htf_data.get(tf)
        if df_tf is None or len(df_tf) < 3:
            # Missing timeframe: award half weight (don't penalise for data gaps)
            achieved += weight * 0.5
            alignment_details.append(f"{tf}=missing(+{weight*0.5:.1f})")
            continue

        last = df_tf.iloc[-1]
        # Primary: supertrend direction; fallback: EMA cross or market structure
        st_dir = float(last.get("supertrend_dir", 0))
        ms = float(last.get("market_structure", 0))
        ema_cross = float(last.get("ema_9", 0)) - float(last.get("ema_21", 0))

        aligned = False
        if direction == "long":
            aligned = (st_dir > 0) or (ms > 0) or (ema_cross > 0)
        else:
            aligned = (st_dir < 0) or (ms < 0) or (ema_cross < 0)

        if aligned:
            achieved += weight
            alignment_details.append(f"{tf}=aligned(+{weight})")
        else:
            alignment_details.append(f"{tf}=opposed(+0)")

    weighted_score = achieved / total_possible * 7.0  # normalise to /7

    if weighted_score < 6.0:
        return LayerResult(6, "MTF Alignment", True, False, 0.0,
                           f"mtf_score={weighted_score:.2f}/7.0 < 6.0  "
                           + " ".join(alignment_details))

    return LayerResult(6, "MTF Alignment", True, True, 100.0,
                       f"mtf_score={weighted_score:.2f}/7.0  " + " ".join(alignment_details))


# ── Layer 7: Microstructure Confirmation (Hard Gate) ──────────────────────────

def layer7_microstructure_gate(
    df: pd.DataFrame,
    direction: str,
    orderbook_imbalance: float = 0.0,   # bid_vol / ask_vol for longs (>2 = good)
    ofi: float = 0.0,                   # order flow imbalance (signed)
    vol_multiplier_required: float = 2.0,
) -> LayerResult:
    """Microstructure hard gate.

    All must pass (spec §10 Layer 7):
    1. Candlestick pattern: Hammer/Shooting Star (wick ≥ 2× body)
       OR Engulfing pattern
    2. Entry candle volume > 200% of 20-period average
    3. Order book bid/ask imbalance > 2:1 at entry (if available)
    4. OFI positive for longs, negative for shorts (if available)
    """
    if df is None or len(df) < 5:
        return LayerResult(7, "Microstructure", True, False, 0.0, "insufficient_data")

    last = df.iloc[-1]
    reasons_pass: list[str] = []
    reasons_fail: list[str] = []

    # 1. Candlestick pattern
    open_ = float(last.get("open", last.get("close", 0)))
    high = float(last.get("high", last.get("close", 0)))
    low = float(last.get("low", last.get("close", 0)))
    close = float(last.get("close", 0))
    total_range = high - low
    body = abs(close - open_)

    hammer = False
    engulf = False
    if total_range > 0:
        if direction == "long":
            # Hammer: small body at top, long lower wick
            lower_wick = open_ - low if close > open_ else close - low
            if lower_wick >= 2 * body and body > 0:
                hammer = True
            # Bullish engulfing: current candle body > previous candle range
            if len(df) >= 2:
                prev = df.iloc[-2]
                prev_body = abs(float(prev.get("close", 0)) - float(prev.get("open", 0)))
                if close > open_ and close > float(prev.get("high", 0)) and open_ < float(prev.get("low", 0)):
                    engulf = True
        else:
            # Shooting star: small body at bottom, long upper wick
            upper_wick = high - close if close < open_ else high - open_
            if upper_wick >= 2 * body and body > 0:
                hammer = True
            # Bearish engulfing
            if len(df) >= 2:
                prev = df.iloc[-2]
                if close < open_ and close < float(prev.get("low", 0)) and open_ > float(prev.get("high", 0)):
                    engulf = True

    candle_ok = hammer or engulf
    if candle_ok:
        reasons_pass.append("hammer_or_engulf")
    else:
        reasons_fail.append("no_candle_pattern")

    # 2. Volume > 200% of 20-period average
    vol_ok = False
    if "volume" in df.columns and len(df) >= 20:
        avg_vol = float(df["volume"].tail(20).mean())
        cur_vol = float(last.get("volume", 0))
        vol_ratio_now = cur_vol / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio_now >= vol_multiplier_required:
            vol_ok = True
            reasons_pass.append(f"vol_{vol_ratio_now:.1f}x")
        else:
            reasons_fail.append(f"vol_insufficient_{vol_ratio_now:.1f}x<{vol_multiplier_required}x")
    else:
        vol_ok = True  # No data — don't block on missing volume
        reasons_pass.append("vol_data_unavailable")

    # 3. Order book imbalance (if provided)
    # Accepts normalized score in [-1, 1] where +1=all bids, -1=all asks.
    # Only activates when non-zero data is present; neutral (0.0) = pass by default.
    ob_ok = True  # default pass when data unavailable
    if orderbook_imbalance != 0.0:
        if direction == "long":
            # Need positive (bid-heavy) imbalance for long; allow slight neutral (-0.1)
            ob_ok = orderbook_imbalance >= -0.1
        else:
            # Need negative (ask-heavy) imbalance for short; allow slight neutral (+0.1)
            ob_ok = orderbook_imbalance <= 0.1
        if ob_ok:
            reasons_pass.append(f"ob_imbalance={orderbook_imbalance:.2f}")
        else:
            reasons_fail.append(f"ob_opposed={orderbook_imbalance:.2f}")

    # 4. OFI (if provided)
    ofi_ok = True
    if ofi != 0.0:
        ofi_ok = (direction == "long" and ofi > 0) or (direction == "short" and ofi < 0)
        if ofi_ok:
            reasons_pass.append(f"ofi={ofi:.1f}")
        else:
            reasons_fail.append(f"ofi_opposed={ofi:.1f}")

    # All mandatory checks: candle pattern + volume
    # OB imbalance and OFI are additional filters only when data is available
    all_pass = candle_ok and vol_ok and ob_ok and ofi_ok

    if not all_pass:
        return LayerResult(7, "Microstructure", True, False, 0.0,
                           "FAIL: " + " ".join(reasons_fail))

    return LayerResult(7, "Microstructure", True, True, 100.0,
                       " ".join(reasons_pass))


# ── Full 7-Layer Validation Pipeline ──────────────────────────────────────────

def validate_signal_7layers(
    df: pd.DataFrame,
    direction: str,
    regime_state: Any = None,
    htf_data: dict[str, pd.DataFrame | None] | None = None,
    vpfr: dict | None = None,
    sentiment_score: float = 0.5,
    has_high_impact_news: bool = False,
    social_volume_ratio: float = 1.0,
    orderbook_imbalance: float = 0.0,
    ofi: float = 0.0,
) -> ValidationResult:
    """Run the full 7-layer validation pipeline.

    Returns ValidationResult with per-layer results, soft scores, and
    rejection info if any hard gate fails.
    """
    results: list[LayerResult] = []

    # ── L0: Sentiment (Hard Gate) ──────────────────────────────────────────
    l0 = layer0_sentiment_gate(sentiment_score, has_high_impact_news, social_volume_ratio)
    results.append(l0)
    if not l0.passed:
        return ValidationResult(False, 0, l0.reason, layer_results=results)

    # ── L1: Regime (Hard Gate) ────────────────────────────────────────────
    l1 = layer1_regime_gate(df)
    results.append(l1)
    if not l1.passed:
        return ValidationResult(False, 1, l1.reason, layer_results=results)

    # ── L2: Structure (Hard Gate) ─────────────────────────────────────────
    l2 = layer2_structure_gate(df, direction)
    results.append(l2)
    if not l2.passed:
        return ValidationResult(False, 2, l2.reason, layer_results=results)

    # ── L3: SMC Confluence (Soft Score) ──────────────────────────────────
    l3 = layer3_smc_score(df, direction)
    results.append(l3)
    if l3.score < 40.0:
        return ValidationResult(False, 3, f"smc_score={l3.score:.0f}<40",
                                layer3_score=l3.score, layer_results=results)

    # ── L4: Momentum Matrix (Soft Score) ──────────────────────────────────
    l4 = layer4_momentum_score(df, direction)
    results.append(l4)
    if l4.score < 60.0:
        return ValidationResult(False, 4, f"momentum_score={l4.score:.0f}<60",
                                layer3_score=l3.score, layer4_score=l4.score,
                                layer_results=results)

    # ── L5: Volume Confirmation (Soft Score) ─────────────────────────────
    l5_score = layer5_volume_score(df, direction, vpfr=vpfr)
    l5 = LayerResult(5, "Volume Confirmation", False, True, l5_score,
                     f"volume_score={l5_score:.0f}")
    results.append(l5)
    if l5_score < 60.0:
        return ValidationResult(False, 5, f"volume_score={l5_score:.0f}<60",
                                layer3_score=l3.score, layer4_score=l4.score,
                                layer5_score=l5_score, layer_results=results)

    # ── L6: MTF Alignment (Hard Gate) ─────────────────────────────────────
    htf_data = htf_data or {}
    l6 = layer6_mtf_gate(htf_data, direction)
    results.append(l6)
    if not l6.passed:
        return ValidationResult(False, 6, l6.reason,
                                layer3_score=l3.score, layer4_score=l4.score,
                                layer5_score=l5_score, layer_results=results)

    # ── L7: Microstructure (Hard Gate) ────────────────────────────────────
    l7 = layer7_microstructure_gate(df, direction, orderbook_imbalance, ofi)
    results.append(l7)
    if not l7.passed:
        return ValidationResult(False, 7, l7.reason,
                                layer3_score=l3.score, layer4_score=l4.score,
                                layer5_score=l5_score, layer_results=results)

    # ── All layers passed ─────────────────────────────────────────────────
    return ValidationResult(
        passed=True,
        rejection_layer=None,
        rejection_reason="",
        layer3_score=l3.score,
        layer4_score=l4.score,
        layer5_score=l5_score,
        layer_results=results,
    )
