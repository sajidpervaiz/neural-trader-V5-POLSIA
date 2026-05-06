"""
ARMS-V2.1 Strategy Modules — Regime-gated trading strategies.

Three modules:
1. TrendContinuationModule: Triple EMA alignment, adaptive RSI, MACD histogram resumption.
2. BreakoutExpansionModule: BB/Keltner squeeze, volume breakout, retest filter.
3. MeanReversionModule: ADX<18 ranging, RSI extremes, BB band touch, limited holds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.regime import MarketRegime, RegimeState


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_rsi_percentile(df: pd.DataFrame, period: int = 100) -> float:
    """100-bar rolling percentile rank of RSI(14) — adaptive zone detection."""
    if "rsi_14" not in df.columns or len(df) < period:
        return 50.0
    rsi_series = df["rsi_14"].tail(period)
    current = float(rsi_series.iloc[-1])
    rank = float((rsi_series < current).sum()) / len(rsi_series) * 100
    return rank


def _detect_candlestick_pattern(df: pd.DataFrame) -> str | None:
    """Detect bullish/bearish candlestick patterns on the latest candle.

    Returns pattern name or None: bullish_engulfing, bearish_engulfing,
    hammer, shooting_star, doji.
    """
    if len(df) < 3 or "open" not in df.columns:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    o = float(last.get("open", 0))
    h = float(last.get("high", 0))
    l_ = float(last.get("low", 0))
    c = float(last.get("close", 0))
    po = float(prev.get("open", 0))
    pc = float(prev.get("close", 0))

    body = abs(c - o)
    rng = h - l_ if h > l_ else 0.001
    prev_body = abs(pc - po)

    # Bullish engulfing
    if pc < po and c > o and c > po and o < pc and body > prev_body:
        return "bullish_engulfing"
    # Bearish engulfing
    if pc > po and c < o and c < po and o > pc and body > prev_body:
        return "bearish_engulfing"
    # Hammer (bullish)
    lower_wick = min(o, c) - l_
    upper_wick = h - max(o, c)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and rng > 0 and body / rng < 0.35:
        return "hammer"
    # Shooting star (bearish)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and rng > 0 and body / rng < 0.35:
        return "shooting_star"
    # Doji
    if rng > 0 and body / rng < 0.1:
        return "doji"
    return None


def _find_sr_zones(
    df: pd.DataFrame, lookback: int = 50, tolerance_pct: float = 0.005,
) -> dict[str, list[float]]:
    """Find support and resistance zones from 3-bar pivot points."""
    if len(df) < lookback or "high" not in df.columns:
        return {"support": [], "resistance": []}

    window = df.tail(lookback)
    highs = window["high"].values.astype(float)
    lows = window["low"].values.astype(float)

    resistance_levels: list[float] = []
    support_levels: list[float] = []

    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistance_levels.append(float(highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            support_levels.append(float(lows[i]))

    def _cluster(levels: list[float], tol: float) -> list[float]:
        if not levels:
            return []
        levels_sorted = sorted(levels)
        clusters: list[float] = []
        current = [levels_sorted[0]]
        for lvl in levels_sorted[1:]:
            if current[-1] > 0 and abs(lvl - current[-1]) / current[-1] < tol:
                current.append(lvl)
            else:
                clusters.append(sum(current) / len(current))
                current = [lvl]
        clusters.append(sum(current) / len(current))
        return clusters

    return {
        "support": _cluster(support_levels, tolerance_pct),
        "resistance": _cluster(resistance_levels, tolerance_pct),
    }


def _check_confirmation_candle(df: pd.DataFrame, direction: str) -> bool:
    """Check if latest candle body is > 60% of range (confirmation candle)."""
    if len(df) < 2 or "open" not in df.columns:
        return False
    last = df.iloc[-1]
    o = float(last.get("open", 0))
    h = float(last.get("high", 0))
    l_ = float(last.get("low", 0))
    c = float(last.get("close", 0))
    rng = h - l_ if h > l_ else 0.001
    body = abs(c - o)
    body_ratio = body / rng
    if body_ratio < 0.6:
        return False
    if direction == "long" and c > o:
        return True
    if direction == "short" and c < o:
        return True
    return False


@dataclass
class StrategySignal:
    """Output from a strategy module."""
    strategy: str
    direction: str  # "long" or "short"
    score: float  # 0..1
    price: float
    atr: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.direction == "long"


# ── Strategy 1: Trend Continuation ──────────────────────────────────────────

class TrendContinuationModule:
    """Fires in STRONG_TREND_UP/DOWN or WEAK_TREND_UP/DOWN regimes.

    Entry criteria:
    - Triple EMA alignment: EMA9 > EMA21 > EMA55 (long) or reversed (short)
    - Adaptive RSI: pullback to RSI percentile zone without breaking trend
    - MACD histogram resumption: histogram was negative, now turning positive (long)
    - Volume at least 1.2x SMA(20)
    """

    ALLOWED_REGIMES = {
        MarketRegime.STRONG_TREND_UP,
        MarketRegime.WEAK_TREND_UP,
        MarketRegime.STRONG_TREND_DOWN,
        MarketRegime.WEAK_TREND_DOWN,
    }

    def __init__(
        self,
        rsi_pullback_low: float = 40.0,
        rsi_pullback_high: float = 60.0,
        volume_min_ratio: float = 1.2,
        rsi_percentile_low: float = 25.0,
        rsi_percentile_high: float = 75.0,
        ema_touch_pct: float = 0.005,
    ) -> None:
        self._rsi_low = rsi_pullback_low
        self._rsi_high = rsi_pullback_high
        self._volume_min = volume_min_ratio
        self._rsi_pctile_low = rsi_percentile_low
        self._rsi_pctile_high = rsi_percentile_high
        self._ema_touch_pct = ema_touch_pct

    def evaluate(self, df: pd.DataFrame, regime: RegimeState) -> StrategySignal | None:
        if regime.regime not in self.ALLOWED_REGIMES:
            return None
        if df is None or len(df) < 55:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        reasons: list[str] = []

        ema9 = last.get("ema_9", 0)
        ema21 = last.get("ema_21", 0)
        ema55 = last.get("ema_55", 0)
        rsi = last.get("rsi_14", 50)
        macd_hist = last.get("macd_hist", 0)
        prev_macd_hist = prev.get("macd_hist", 0)
        vol_ratio = last.get("volume_ratio", 1.0)
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))

        if any(v == 0 for v in [ema9, ema21, ema55]):
            return None

        # Determine direction from regime
        is_bull = regime.regime in {MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP}

        if is_bull:
            # Triple EMA alignment: EMA9 > EMA21 > EMA55
            if not (ema9 > ema21 > ema55):
                return None
            reasons.append("triple_ema_bull")

            # ARMS-V2.1: Adaptive RSI percentile (100-bar rolling rank)
            rsi_pctile = _compute_rsi_percentile(df)
            if rsi_pctile > self._rsi_pctile_high:
                return None  # Too extended
            if rsi_pctile < self._rsi_pctile_low:
                return None  # Pullback too deep
            reasons.append(f"rsi_pctile_{rsi_pctile:.0f}")

            # ARMS-V2.1: EMA pullback touch — price within 0.5% of EMA21
            ema_dist = abs(price - ema21) / ema21 if ema21 > 0 else 1.0
            if ema_dist > self._ema_touch_pct:
                return None
            reasons.append("ema21_touch")

            # MACD histogram resumption: was negative/zero, now positive
            if not (prev_macd_hist <= 0 and macd_hist > 0):
                # Alternative: histogram increasing
                if not (macd_hist > prev_macd_hist and macd_hist > 0):
                    return None
            reasons.append("macd_hist_resumption")

            # ARMS-V2.1: Confirmation candle (body > 60% of range)
            if not _check_confirmation_candle(df, "long"):
                return None
            reasons.append("confirmation_candle")

            direction = "long"
        else:
            # Short: reversed alignment
            if not (ema9 < ema21 < ema55):
                return None
            reasons.append("triple_ema_bear")

            rsi_pctile = _compute_rsi_percentile(df)
            if rsi_pctile < self._rsi_pctile_low:
                return None
            if rsi_pctile > self._rsi_pctile_high:
                return None
            reasons.append(f"rsi_pctile_{rsi_pctile:.0f}")

            ema_dist = abs(price - ema21) / ema21 if ema21 > 0 else 1.0
            if ema_dist > self._ema_touch_pct:
                return None
            reasons.append("ema21_touch")

            if not (prev_macd_hist >= 0 and macd_hist < 0):
                if not (macd_hist < prev_macd_hist and macd_hist < 0):
                    return None
            reasons.append("macd_hist_resumption")

            if not _check_confirmation_candle(df, "short"):
                return None
            reasons.append("confirmation_candle")

            direction = "short"

        # Volume filter
        if vol_ratio < self._volume_min:
            return None
        reasons.append(f"volume_{vol_ratio:.1f}x")

        # Score based on regime strength, EMA spacing, and histogram momentum
        ema_spread = abs(ema9 - ema55) / price if price > 0 else 0
        regime_bonus = 0.2 if regime.regime in {MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN} else 0.0
        score = min(1.0, 0.5 + ema_spread * 100 + regime_bonus + abs(macd_hist) / atr * 0.1)

        return StrategySignal(
            strategy="trend_continuation",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"regime": regime.regime.value, "regime_confidence": regime.confidence},
        )


# ── Strategy 2: Breakout Expansion ──────────────────────────────────────────

class BreakoutExpansionModule:
    """Fires in COMPRESSION regime (awaiting breakout).

    Entry criteria:
    - BB width < 20th percentile (squeeze)
    - Keltner squeeze active
    - Price breaks above 50-candle high (long) or below 50-candle low (short)
    - Volume > 2.0x SMA(20) on breakout candle
    - Retest/fakeout filter: close must be past the breakout level
    """

    ALLOWED_REGIMES = {MarketRegime.COMPRESSION}

    def __init__(
        self,
        bb_squeeze_pctile: float = 20.0,
        volume_breakout_ratio: float = 2.0,
        breakout_margin_pct: float = 0.001,
        consolidation_range_pct: float = 0.02,
        volume_decay_candles: int = 3,
    ) -> None:
        self._bb_squeeze = bb_squeeze_pctile
        self._vol_breakout = volume_breakout_ratio
        self._margin_pct = breakout_margin_pct
        self._consolidation_range = consolidation_range_pct
        self._volume_decay_candles = volume_decay_candles

    def evaluate(self, df: pd.DataFrame, regime: RegimeState) -> StrategySignal | None:
        if regime.regime not in self.ALLOWED_REGIMES:
            return None
        if df is None or len(df) < 55:
            return None

        last = df.iloc[-1]
        reasons: list[str] = []

        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        bb_pctile = float(last.get("bb_width_percentile", 50))
        keltner_sq = bool(last.get("keltner_squeeze", False))
        vol_ratio = float(last.get("volume_ratio", 1.0))
        high_50 = float(last.get("high_50", price))
        low_50 = float(last.get("low_50", price))
        high_val = float(last.get("high", price))
        low_val = float(last.get("low", price))

        # Squeeze must be present
        if bb_pctile > self._bb_squeeze and not keltner_sq:
            return None
        reasons.append(f"squeeze_bb{bb_pctile:.0f}pct")
        if keltner_sq:
            reasons.append("keltner_squeeze")

        # ARMS-V2.1: Pre-breakout range consolidation — 50-candle range < 2%
        if len(df) >= 50:
            range_50 = df["high"].tail(50).max() - df["low"].tail(50).min()
            mid_50 = (df["high"].tail(50).max() + df["low"].tail(50).min()) / 2
            if mid_50 > 0 and range_50 / mid_50 > self._consolidation_range:
                return None
            reasons.append(f"consolidation_{range_50 / mid_50 * 100:.1f}pct")

        # ARMS-V2.1: Volume decay pre-breakout — volume declining before spike
        if len(df) >= self._volume_decay_candles + 2 and "volume" in df.columns:
            pre_vols = df["volume"].iloc[-(self._volume_decay_candles + 1):-1].values
            decaying = all(pre_vols[i] >= pre_vols[i + 1] for i in range(len(pre_vols) - 1))
            if decaying:
                reasons.append("volume_decay_confirmed")

        # Volume must be elevated on breakout
        if vol_ratio < self._vol_breakout:
            return None
        reasons.append(f"volume_{vol_ratio:.1f}x")

        # Breakout detection
        margin = price * self._margin_pct
        if high_val > high_50 and price > high_50 - margin:
            # Bullish breakout — close must confirm (above previous resistance)
            if price <= high_50 * (1 - self._margin_pct):
                return None  # Fakeout — close back below
            direction = "long"
            reasons.append("break_50high")
        elif low_val < low_50 and price < low_50 + margin:
            if price >= low_50 * (1 + self._margin_pct):
                return None
            direction = "short"
            reasons.append("break_50low")
        else:
            return None  # No breakout

        # ARMS-V2.1: Follow-through candle validation
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("follow_through_candle")

        # ARMS-V2.1: Retest validation — previous candle broke out, current confirms
        if len(df) >= 3:
            prev_2 = df.iloc[-2]
            if direction == "long":
                prev_broke = float(prev_2.get("high", 0)) > high_50
                current_holds = price > high_50
                if prev_broke and current_holds:
                    reasons.append("retest_confirmed")
            else:
                prev_broke = float(prev_2.get("low", float("inf"))) < low_50
                current_holds = price < low_50
                if prev_broke and current_holds:
                    reasons.append("retest_confirmed")

        score = min(1.0, 0.6 + (vol_ratio - self._vol_breakout) * 0.1 + regime.confidence * 0.2)

        return StrategySignal(
            strategy="breakout_expansion",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"regime": regime.regime.value, "bb_width_pctile": bb_pctile},
        )


# ── Strategy 3: Mean Reversion ──────────────────────────────────────────────

class MeanReversionModule:
    """Fires in RANGE_CHOP regime.

    Entry criteria:
    - ADX < 18 for 5+ candles (ranging confirmed)
    - Price at BB lower band + RSI < 30 + Stochastic < 20 (long)
    - Price at BB upper band + RSI > 70 + Stochastic > 80 (short)
    - 12-candle max hold
    - Max 2 trades per 24 hours
    """

    ALLOWED_REGIMES = {MarketRegime.RANGE_CHOP}

    def __init__(
        self,
        adx_max: float = 18.0,
        adx_confirm_candles: int = 5,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
        bb_touch_threshold: float = 0.05,
        max_trades_per_24h: int = 2,
        max_hold_candles: int = 12,
    ) -> None:
        self._adx_max = adx_max
        self._adx_confirm = adx_confirm_candles
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._stoch_oversold = stoch_oversold
        self._stoch_overbought = stoch_overbought
        self._bb_touch = bb_touch_threshold
        self._max_trades_24h = max_trades_per_24h
        self._max_hold = max_hold_candles
        self._trade_times: list[float] = []

    @property
    def max_hold_candles(self) -> int:
        return self._max_hold

    def _prune_trade_times(self) -> None:
        cutoff = time.time() - 86400
        self._trade_times = [t for t in self._trade_times if t >= cutoff]

    def record_trade(self, ts: float | None = None) -> None:
        self._trade_times.append(ts or time.time())
        self._prune_trade_times()

    def evaluate(self, df: pd.DataFrame, regime: RegimeState) -> StrategySignal | None:
        if regime.regime not in self.ALLOWED_REGIMES:
            return None
        if df is None or len(df) < 55:
            return None

        # 24h trade limit
        self._prune_trade_times()
        if len(self._trade_times) >= self._max_trades_24h:
            return None

        last = df.iloc[-1]
        reasons: list[str] = []

        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        rsi = float(last.get("rsi_14", 50))
        stoch_k = float(last.get("stoch_k", 50))
        bb_lower = float(last.get("bb_lower", price))
        bb_upper = float(last.get("bb_upper", price))
        bb_pct = float(last.get("bb_pct", 0.5))

        # ADX must be low for at least N candles
        if "adx" in df.columns:
            recent_adx = df["adx"].tail(self._adx_confirm)
            if len(recent_adx) < self._adx_confirm:
                return None
            if (recent_adx > self._adx_max).any():
                return None
            reasons.append(f"adx_low_{recent_adx.iloc[-1]:.0f}")
        else:
            if regime.adx > self._adx_max:
                return None

        # Long: price near BB lower + RSI oversold + Stochastic oversold
        if bb_pct < self._bb_touch and rsi < self._rsi_oversold and stoch_k < self._stoch_oversold:
            direction = "long"
            reasons.extend([f"bb_lower_touch_{bb_pct:.2f}", f"rsi_{rsi:.0f}", f"stoch_{stoch_k:.0f}"])
        # Short: price near BB upper + RSI overbought + Stochastic overbought
        elif bb_pct > (1 - self._bb_touch) and rsi > self._rsi_overbought and stoch_k > self._stoch_overbought:
            direction = "short"
            reasons.extend([f"bb_upper_touch_{bb_pct:.2f}", f"rsi_{rsi:.0f}", f"stoch_{stoch_k:.0f}"])
        else:
            return None

        # ARMS-V2.1: Candlestick pattern confirmation
        pattern = _detect_candlestick_pattern(df)
        if pattern is None:
            return None  # Require pattern confirmation
        # Validate pattern aligns with direction
        bullish_patterns = {"bullish_engulfing", "hammer", "doji"}
        bearish_patterns = {"bearish_engulfing", "shooting_star", "doji"}
        if direction == "long" and pattern not in bullish_patterns:
            return None
        if direction == "short" and pattern not in bearish_patterns:
            return None
        reasons.append(f"pattern:{pattern}")

        # ARMS-V2.1: S/R zone proximity check
        sr_zones = _find_sr_zones(df)
        sr_bonus = 0.0
        if direction == "long" and sr_zones["support"]:
            nearest_support = min(sr_zones["support"], key=lambda s: abs(s - price))
            if price > 0 and abs(nearest_support - price) / price < 0.01:
                reasons.append(f"near_support_{nearest_support:.2f}")
                sr_bonus = 0.1
        elif direction == "short" and sr_zones["resistance"]:
            nearest_resistance = min(sr_zones["resistance"], key=lambda r: abs(r - price))
            if price > 0 and abs(nearest_resistance - price) / price < 0.01:
                reasons.append(f"near_resistance_{nearest_resistance:.2f}")
                sr_bonus = 0.1

        score = min(1.0, 0.5 + (1.0 - regime.adx / self._adx_max) * 0.3 + regime.confidence * 0.2 + sr_bonus)

        return StrategySignal(
            strategy="mean_reversion",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={
                "regime": regime.regime.value,
                "max_hold_candles": self._max_hold,
                "trades_today": len(self._trade_times),
            },
        )


# ── Strategy Selector ────────────────────────────────────────────────────────

class StrategySelector:
    """Regime-gated strategy selection — picks the appropriate strategy module
    based on the current market regime."""

    def __init__(
        self,
        trend_module: TrendContinuationModule | None = None,
        breakout_module: BreakoutExpansionModule | None = None,
        mean_reversion_module: MeanReversionModule | None = None,
    ) -> None:
        self.trend = trend_module or TrendContinuationModule()
        self.breakout = breakout_module or BreakoutExpansionModule()
        self.mean_reversion = mean_reversion_module or MeanReversionModule()

    def select_and_evaluate(
        self, df: pd.DataFrame, regime: RegimeState,
    ) -> StrategySignal | None:
        """Evaluate the strategy module appropriate for the current regime."""
        if regime.regime in TrendContinuationModule.ALLOWED_REGIMES:
            return self.trend.evaluate(df, regime)
        elif regime.regime in BreakoutExpansionModule.ALLOWED_REGIMES:
            return self.breakout.evaluate(df, regime)
        elif regime.regime in MeanReversionModule.ALLOWED_REGIMES:
            return self.mean_reversion.evaluate(df, regime)
        return None
