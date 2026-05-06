from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ── ARMS-V2.1 6-State Regime Classification ──────────────────────────────────

class MarketRegime(str, Enum):
    STRONG_TREND_UP = "strong_trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    COMPRESSION = "compression"
    RANGE_CHOP = "range_chop"
    WEAK_TREND_DOWN = "weak_trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    UNKNOWN = "unknown"

    # Legacy aliases for backward compatibility
    TRENDING_UP = "strong_trend_up"
    TRENDING_DOWN = "strong_trend_down"
    RANGING = "range_chop"
    VOLATILE = "compression"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    adx: float
    plus_di: float
    minus_di: float
    trend_slope: float
    realized_vol: float
    hurst_exponent: float
    ema200_slope: float
    bb_width_percentile: float
    keltner_squeeze: bool
    timestamp: int
    candles_in_state: int = 0

    @property
    def tradeable(self) -> bool:
        """Spec L1: ADX<25 with RANGE_CHOP/COMPRESSION = NO TRADE."""
        if self.regime in (MarketRegime.RANGE_CHOP,) and self.adx < 25:
            return False
        return self.regime != MarketRegime.UNKNOWN

    @property
    def position_size_pct(self) -> float:
        """Spec L1: Position sizing by regime.
        Strong Trend → 100%, Moderate → 75%, Choppy → 0%."""
        if not self.tradeable:
            return 0.0
        if self.regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            return 1.0
        if self.regime in (MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN):
            return 0.75
        if self.regime == MarketRegime.COMPRESSION:
            return 0.5
        return 0.0


# ── Transition rules: which transitions are allowed ──────────────────────────
_ALLOWED_TRANSITIONS: dict[MarketRegime, set[MarketRegime]] = {
    MarketRegime.STRONG_TREND_UP: {
        MarketRegime.WEAK_TREND_UP, MarketRegime.COMPRESSION, MarketRegime.UNKNOWN,
    },
    MarketRegime.WEAK_TREND_UP: {
        MarketRegime.STRONG_TREND_UP, MarketRegime.RANGE_CHOP,
        MarketRegime.COMPRESSION, MarketRegime.UNKNOWN,
    },
    MarketRegime.COMPRESSION: {
        MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN,
        MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN,
        MarketRegime.RANGE_CHOP, MarketRegime.UNKNOWN,
    },
    MarketRegime.RANGE_CHOP: {
        MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN,
        MarketRegime.COMPRESSION, MarketRegime.UNKNOWN,
    },
    MarketRegime.WEAK_TREND_DOWN: {
        MarketRegime.STRONG_TREND_DOWN, MarketRegime.RANGE_CHOP,
        MarketRegime.COMPRESSION, MarketRegime.UNKNOWN,
    },
    MarketRegime.STRONG_TREND_DOWN: {
        MarketRegime.WEAK_TREND_DOWN, MarketRegime.COMPRESSION, MarketRegime.UNKNOWN,
    },
    MarketRegime.UNKNOWN: set(MarketRegime),
}


class RegimeDetector:
    """ARMS-V2.1 regime detector with 6-state classification, 2-candle confirmation,
    transition rules, and cooldown."""

    def __init__(
        self,
        adx_threshold: float = 25.0,
        strong_adx_threshold: float = 30.0,
        vol_threshold: float = 0.6,
        hurst_threshold: float = 0.55,
        ema200_slope_threshold: float = 0.0015,
        bb_squeeze_percentile: float = 20.0,
        confirmation_candles: int = 3,
        cooldown_candles: int = 10,
    ) -> None:
        self._adx_threshold = adx_threshold
        self._strong_adx_threshold = strong_adx_threshold
        self._vol_threshold = vol_threshold
        self._hurst_threshold = hurst_threshold
        self._ema200_slope_threshold = ema200_slope_threshold
        self._bb_squeeze_percentile = bb_squeeze_percentile
        self._confirmation_candles = confirmation_candles
        self._cooldown_candles = cooldown_candles

        # State machine — REQ-REG-005 hysteresis is enforced via the
        # confirmation_candles / cooldown_candles fields. The fields below
        # are read by RegimeDetector.get_hysteresis_snapshot() so the
        # dashboard can show "regime since N candles" + any pending switch.
        self._current_regime = MarketRegime.UNKNOWN
        self._candles_in_state = 0
        self._pending_regime: MarketRegime | None = None
        self._pending_count = 0
        self._cooldown_remaining = 0
        self._history: deque[MarketRegime] = deque(maxlen=50)
        self._last_transition_ts: float = 0.0

    def _calc_adx_di(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
    ) -> tuple[float, float, float]:
        """Returns (adx, +DI, -DI)."""
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        dm_plus = (high - prev_high).clip(lower=0)
        dm_minus = (prev_low - low).clip(lower=0)
        mask = dm_plus <= dm_minus
        dm_plus[mask] = 0
        mask2 = dm_minus <= dm_plus
        dm_minus[mask2] = 0

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_s = tr.ewm(com=period - 1, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, adjust=False).mean()

        adx_val = float(adx.iloc[-1]) if not adx.empty and not np.isnan(adx.iloc[-1]) else 0.0
        dip_val = float(di_plus.iloc[-1]) if not di_plus.empty and not np.isnan(di_plus.iloc[-1]) else 0.0
        dim_val = float(di_minus.iloc[-1]) if not di_minus.empty and not np.isnan(di_minus.iloc[-1]) else 0.0
        return adx_val, dip_val, dim_val

    # Keep legacy single-value method for backward compat
    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        adx, _, _ = self._calc_adx_di(high, low, close, period)
        return adx

    def _calc_hurst(self, series: pd.Series, max_lag: int = 20) -> float:
        prices = series.dropna().values
        if len(prices) < 30:
            return 0.5
        lags = range(2, min(max_lag, len(prices) // 4))
        tau = []
        for lag in lags:
            diffs = prices[lag:] - prices[:-lag]
            tau.append(np.sqrt(np.var(diffs)))
        if len(tau) < 2:
            return 0.5
        try:
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return float(poly[0])
        except Exception:
            return 0.5

    def _calc_realized_vol(self, close: pd.Series, periods: int = 20) -> float:
        returns = close.pct_change().dropna()
        if len(returns) < periods:
            return 0.0
        return float(returns.tail(periods).std() * np.sqrt(252))

    @staticmethod
    def _calc_ema200_slope(close: pd.Series, lookback: int = 10) -> float:
        """EMA200 slope as percentage change over lookback bars."""
        ema200 = close.ewm(span=200, adjust=False).mean()
        if len(ema200) < lookback + 1:
            return 0.0
        curr = float(ema200.iloc[-1])
        prev = float(ema200.iloc[-lookback])
        if prev == 0:
            return 0.0
        return (curr - prev) / prev

    @staticmethod
    def _calc_bb_width_percentile(close: pd.Series, period: int = 20, window: int = 200) -> float:
        """Rolling percentile rank of BB width over `window` candles."""
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        width = (2 * std) / mid.replace(0, np.nan)
        if len(width.dropna()) < 2:
            return 50.0
        rank = width.rolling(window, min_periods=20).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100, raw=False,
        )
        val = rank.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

    @staticmethod
    def _detect_keltner_squeeze(close: pd.Series, high: pd.Series, low: pd.Series,
                                 bb_period: int = 20, kc_mult: float = 1.5) -> bool:
        """True when Bollinger Bands are inside Keltner Channels (squeeze)."""
        mid = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_upper = mid + 2 * bb_std
        bb_lower = mid - 2 * bb_std

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_s = tr.ewm(com=bb_period - 1, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        kc_upper = ema20 + kc_mult * atr_s
        kc_lower = ema20 - kc_mult * atr_s

        if bb_upper.empty or kc_upper.empty:
            return False

        return bool(bb_upper.iloc[-1] < kc_upper.iloc[-1] and bb_lower.iloc[-1] > kc_lower.iloc[-1])

    def _classify_raw(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        trend_slope: float,
        ema200_slope: float,
        bb_width_pctile: float,
        keltner_squeeze: bool,
        realized_vol: float,
        hurst: float,
    ) -> tuple[MarketRegime, float]:
        """Pure classification logic — returns (regime, confidence)."""
        # Compression: BB squeeze or BB width < threshold percentile + Keltner squeeze
        if bb_width_pctile < self._bb_squeeze_percentile and keltner_squeeze:
            return MarketRegime.COMPRESSION, min(1.0, (self._bb_squeeze_percentile - bb_width_pctile) / self._bb_squeeze_percentile + 0.5)

        if bb_width_pctile < self._bb_squeeze_percentile and adx < self._adx_threshold:
            return MarketRegime.COMPRESSION, min(1.0, 0.6 + (self._adx_threshold - adx) / self._adx_threshold * 0.4)

        # Strong Trend: ADX > strong threshold + DI confirms direction + EMA200 slope confirms
        if adx > self._strong_adx_threshold:
            if plus_di > minus_di and ema200_slope > self._ema200_slope_threshold:
                conf = min(1.0, adx / 50.0)
                return MarketRegime.STRONG_TREND_UP, conf
            if minus_di > plus_di and ema200_slope < -self._ema200_slope_threshold:
                conf = min(1.0, adx / 50.0)
                return MarketRegime.STRONG_TREND_DOWN, conf

        # Weak Trend: ADX > base threshold and DI confirms, but not strong enough
        if adx > self._adx_threshold:
            if plus_di > minus_di and trend_slope > 0:
                conf = min(1.0, adx / 40.0 * 0.7)
                return MarketRegime.WEAK_TREND_UP, conf
            if minus_di > plus_di and trend_slope < 0:
                conf = min(1.0, adx / 40.0 * 0.7)
                return MarketRegime.WEAK_TREND_DOWN, conf

        # Range/Chop: low ADX, no clear DI dominance
        if adx < self._adx_threshold:
            conf = max(0.3, 1.0 - adx / self._adx_threshold)
            return MarketRegime.RANGE_CHOP, conf

        # Fallback: weak trend based on slope
        if trend_slope > 0:
            return MarketRegime.WEAK_TREND_UP, 0.4
        elif trend_slope < 0:
            return MarketRegime.WEAK_TREND_DOWN, 0.4
        return MarketRegime.RANGE_CHOP, 0.3

    def _apply_state_machine(self, raw_regime: MarketRegime) -> MarketRegime:
        """2-candle confirmation + transition rules + cooldown."""
        # Cooldown: don't allow any transition while cooldown is active
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._candles_in_state += 1
            return self._current_regime

        # Same as current — reset pending, increment candles
        if raw_regime == self._current_regime:
            self._pending_regime = None
            self._pending_count = 0
            self._candles_in_state += 1
            return self._current_regime

        # Check if transition is allowed
        allowed = _ALLOWED_TRANSITIONS.get(self._current_regime, set(MarketRegime))
        if raw_regime not in allowed:
            self._candles_in_state += 1
            return self._current_regime

        # 2-candle confirmation
        if raw_regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        if self._pending_count >= self._confirmation_candles:
            old = self._current_regime
            self._current_regime = raw_regime
            self._candles_in_state = 0
            self._pending_regime = None
            self._pending_count = 0
            self._cooldown_remaining = self._cooldown_candles
            self._history.append(raw_regime)
            self._last_transition_ts = time.time()
            logger.info("Regime transition: {} → {} (cooldown={})", old.value, raw_regime.value, self._cooldown_candles)
            return self._current_regime

        self._candles_in_state += 1
        return self._current_regime

    def get_hysteresis_snapshot(self) -> dict[str, Any]:
        """REQ-REG-005: expose the hysteresis state for dashboard rendering.

        Returns the current regime + how many candles we've held it,
        any pending regime swap (and how many of the required confirmation
        candles it has accumulated), and remaining post-transition cooldown.
        """
        return {
            "current_regime": self._current_regime.value,
            "candles_in_state": int(self._candles_in_state),
            "pending_regime": (self._pending_regime.value if self._pending_regime is not None else None),
            "pending_count": int(self._pending_count),
            "confirmation_required": int(self._confirmation_candles),
            "cooldown_remaining": int(self._cooldown_remaining),
            "cooldown_total": int(self._cooldown_candles),
            "last_transition_ts": float(self._last_transition_ts),
            "recent_regimes": [r.value for r in list(self._history)[-10:]],
        }

    def detect(self, df: pd.DataFrame) -> RegimeState:
        if len(df) < 30:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx=0.0,
                plus_di=0.0,
                minus_di=0.0,
                trend_slope=0.0,
                realized_vol=0.0,
                hurst_exponent=0.5,
                ema200_slope=0.0,
                bb_width_percentile=50.0,
                keltner_squeeze=False,
                timestamp=int(time.time()),
            )

        high = df["high"]
        low = df["low"]
        close = df["close"]

        adx, plus_di, minus_di = self._calc_adx_di(high, low, close)
        realized_vol = self._calc_realized_vol(close)
        hurst = self._calc_hurst(close)
        ema200_slope = self._calc_ema200_slope(close)
        bb_width_pctile = self._calc_bb_width_percentile(close)
        keltner_squeeze = self._detect_keltner_squeeze(close, high, low)

        recent = close.tail(20).values
        trend_slope = float(np.polyfit(range(len(recent)), recent, 1)[0]) / float(recent.mean()) if recent.mean() > 0 else 0.0

        raw_regime, confidence = self._classify_raw(
            adx, plus_di, minus_di, trend_slope, ema200_slope,
            bb_width_pctile, keltner_squeeze, realized_vol, hurst,
        )

        confirmed_regime = self._apply_state_machine(raw_regime)
        # Use raw confidence if we haven't transitioned yet
        if confirmed_regime != raw_regime:
            confidence = max(0.3, confidence * 0.7)

        return RegimeState(
            regime=confirmed_regime,
            confidence=confidence,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            trend_slope=trend_slope,
            realized_vol=realized_vol,
            hurst_exponent=hurst,
            ema200_slope=ema200_slope,
            bb_width_percentile=bb_width_pctile,
            keltner_squeeze=keltner_squeeze,
            timestamp=int(time.time()),
            candles_in_state=self._candles_in_state,
        )

    def reset(self) -> None:
        """Reset detector state machine."""
        self._current_regime = MarketRegime.UNKNOWN
        self._candles_in_state = 0
        self._pending_regime = None
        self._pending_count = 0
        self._cooldown_remaining = 0
        self._history.clear()
