from __future__ import annotations

import asyncio
import datetime
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from analysis.data_manager import DataManager
from analysis.regime import MarketRegime, RegimeState
from analysis.smart_money import SmartMoneyAnalyzer, SMCState
from analysis.volume_profile import VolumeProfileAnalyzer, VolumeFlowState
from engine.ai_agent import TradingAIAgent
from engine.model_trainer import ModelTrainer
from engine.strategy_modules import StrategySelector, StrategySignal

try:
    from engine.learning.scorer import AdaptiveMLScorer
    _ADAPTIVE_ML_AVAILABLE = True
except Exception:
    AdaptiveMLScorer = None  # type: ignore
    _ADAPTIVE_ML_AVAILABLE = False

try:
    from engine.session_filter import SessionFilter
    _SESSION_FILTER_AVAILABLE = True
except Exception:
    SessionFilter = None  # type: ignore
    _SESSION_FILTER_AVAILABLE = False

try:
    from engine.smc_signal_modules import SMCStrategySelector
    _SMC_SELECTOR_AVAILABLE = True
except Exception:
    SMCStrategySelector = None  # type: ignore
    _SMC_SELECTOR_AVAILABLE = False

try:
    from engine.master_scorer import compute_master_score, compute_exec_liquidity_score
    _MASTER_SCORER_AVAILABLE = True
except Exception:
    compute_master_score = None  # type: ignore
    compute_exec_liquidity_score = None  # type: ignore
    _MASTER_SCORER_AVAILABLE = False

try:
    from engine.pair_registry import PairRegistry, PairTier
    _PAIR_REGISTRY_AVAILABLE = True
except Exception:
    PairRegistry = None  # type: ignore
    PairTier = None  # type: ignore
    _PAIR_REGISTRY_AVAILABLE = False

try:
    from execution.estop import EStopManager
    _ESTOP_AVAILABLE = True
except Exception:
    EStopManager = None  # type: ignore
    _ESTOP_AVAILABLE = False


# ── Signal Type Classification (§3) ──────────────────────────────────────────

class SignalType(str, Enum):
    TYPE_A = "breakout_pullback"       # Donchian breakout + Fib retracement
    TYPE_B = "liquidity_sweep"         # Liquidity sweep reversal
    TYPE_C = "fvg_mitigation"          # Fair Value Gap mitigation entry
    TYPE_D = "order_block_mitigation"  # Order Block mitigation entry
    COMPOSITE = "composite"            # Generic composite signal

# Priority: C > D > A > B (per spec §3)
_SIGNAL_PRIORITY: dict[SignalType, int] = {
    SignalType.TYPE_C: 4,
    SignalType.TYPE_D: 3,
    SignalType.TYPE_A: 2,
    SignalType.TYPE_B: 1,
    SignalType.COMPOSITE: 0,
}


# ── Session & Killzone Rules (§5) ────────────────────────────────────────────

@dataclass
class SessionRule:
    name: str
    start_utc: int           # hour (0-23)
    end_utc: int             # hour (0-23)
    allowed_types: set[SignalType] | None   # None = all allowed
    size_multiplier: float   # 1.0 = 100%, 0.5 = 50%, etc.
    no_trade: bool = False   # If True, block all signals

_SESSION_RULES: list[SessionRule] = [
    SessionRule("asia",           0,  8,  {SignalType.TYPE_B, SignalType.COMPOSITE}, 0.50),
    SessionRule("london_open",    8,  12, {SignalType.TYPE_A, SignalType.TYPE_B, SignalType.TYPE_C, SignalType.TYPE_D, SignalType.COMPOSITE}, 1.00),
    SessionRule("london_dead",    12, 13, None, 0.50),  # Reduced size during dead zone
    SessionRule("london_ny",      13, 17, None, 1.50),  # All types allowed, aggressive
    SessionRule("ny_only",        17, 22, {SignalType.TYPE_B, SignalType.TYPE_D, SignalType.COMPOSITE}, 0.75),
    SessionRule("low_liquidity",  22, 24, None, 0.25),  # Reduced size during low liquidity
]

# ICT Killzones (spec §5): 1.2x score multiplier, Type C & D only
_ICT_KILLZONES: list[tuple[int, int]] = [(13, 14), (15, 16)]


# ── Regime-specific risk percentages (§7) ────────────────────────────────────

_REGIME_RISK_PCT: dict[MarketRegime, float] = {
    MarketRegime.STRONG_TREND_UP:   0.02,   # 2% risk — trending
    MarketRegime.WEAK_TREND_UP:     0.02,   # 2% risk — trending
    MarketRegime.STRONG_TREND_DOWN: 0.02,   # 2% risk — trending
    MarketRegime.WEAK_TREND_DOWN:   0.02,   # 2% risk — trending
    MarketRegime.COMPRESSION:       0.025,  # 2.5% risk — breakout
    MarketRegime.RANGE_CHOP:        0.015,  # 1.5% risk — ranging
}

# Regime transition rules: reduce 50%, tighten SL 0.5x for 2 candles
_TRANSITION_CANDLE_WINDOW = 2
_TRANSITION_SIZE_MULT = 0.50
_TRANSITION_SL_MULT = 0.50


# ── Regime-adaptive weight profiles ──────────────────────────────────────────
# Each profile sums to 1.0.  Weights shift depending on detected regime so the
# signal generator emphasises the factors most predictive for that market state.
_REGIME_WEIGHT_PROFILES: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_TREND_UP: {
        "technical": 0.35, "ml": 0.30, "sentiment": 0.10,
        "macro": 0.05, "news": 0.10, "orderbook": 0.10,
    },
    MarketRegime.WEAK_TREND_UP: {
        "technical": 0.30, "ml": 0.30, "sentiment": 0.10,
        "macro": 0.05, "news": 0.15, "orderbook": 0.10,
    },
    MarketRegime.STRONG_TREND_DOWN: {
        "technical": 0.35, "ml": 0.30, "sentiment": 0.10,
        "macro": 0.05, "news": 0.10, "orderbook": 0.10,
    },
    MarketRegime.WEAK_TREND_DOWN: {
        "technical": 0.30, "ml": 0.30, "sentiment": 0.10,
        "macro": 0.05, "news": 0.15, "orderbook": 0.10,
    },
    MarketRegime.COMPRESSION: {
        "technical": 0.20, "ml": 0.20, "sentiment": 0.05,
        "macro": 0.05, "news": 0.10, "orderbook": 0.40,
    },
    MarketRegime.RANGE_CHOP: {
        "technical": 0.40, "ml": 0.15, "sentiment": 0.05,
        "macro": 0.05, "news": 0.10, "orderbook": 0.25,
    },
}

# Asymmetric direction threshold multipliers by regime.
# counter-trend trades require a higher score to fire.
_REGIME_DIRECTION_PENALTY: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_TREND_UP:   {"long": 1.0, "short": 1.5},
    MarketRegime.WEAK_TREND_UP:     {"long": 1.0, "short": 1.2},
    MarketRegime.STRONG_TREND_DOWN: {"long": 1.5, "short": 1.0},
    MarketRegime.WEAK_TREND_DOWN:   {"long": 1.2, "short": 1.0},
    MarketRegime.COMPRESSION:       {"long": 1.0, "short": 1.0},
    MarketRegime.RANGE_CHOP:        {"long": 1.0, "short": 1.0},
}

# Correlation groups — assets that tend to move together.
# Only one position per group is allowed to avoid concentrated risk.
_CORRELATION_GROUPS: list[set[str]] = [
    {"BTC/USDT:USDT", "BTC/USDT"},
    {"ETH/USDT:USDT", "ETH/USDT"},
    {"SOL/USDT:USDT", "SOL/USDT"},
    # Major crypto group: avoid same-direction on highly correlated assets
]
_CRYPTO_MAJOR_GROUP = {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                       "BTC/USDT", "ETH/USDT", "SOL/USDT"}


@dataclass
class TradingSignal:
    exchange: str
    symbol: str
    direction: str
    score: float
    technical_score: float
    ml_score: float
    sentiment_score: float
    macro_score: float
    news_score: float
    orderbook_score: float
    regime: str
    regime_confidence: float
    price: float
    atr: float
    stop_loss: float
    take_profit: float
    timestamp: int
    metadata: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    signal_type: str = SignalType.COMPOSITE.value
    quality_score: int = 0          # 0-100 quality score (§6)
    session_name: str = ""          # Active session name
    session_size_mult: float = 1.0  # Session-based position sizing
    regime_risk_pct: float = 0.01   # Regime-specific risk % (§7)
    mtf_agreement_count: int = 0    # Number of TFs in agreement
    mtf_weighted_score: float = 0.0 # Weighted MTF composite [-1, 1]
    size_multiplier: float = 1.0    # Combined position sizing multiplier

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def is_short(self) -> bool:
        return self.direction == "short"

    @property
    def risk_reward(self) -> float:
        if self.is_long:
            risk = self.price - self.stop_loss
            reward = self.take_profit - self.price
        else:
            risk = self.stop_loss - self.price
            reward = self.price - self.take_profit
        return reward / risk if risk > 0 else 0.0

    @property
    def factor_count(self) -> int:
        """Number of signal factors that have a non-zero contribution."""
        count = 0
        if self.technical_score != 0.0:
            count += 1
        if self.ml_score != 0.0:
            count += 1
        if self.sentiment_score != 0.0:
            count += 1
        if self.macro_score != 0.0:
            count += 1
        if self.news_score != 0.0:
            count += 1
        if self.orderbook_score != 0.0:
            count += 1
        return count


class TechnicalScorer:
    """L4 Momentum Convergence Matrix — 15 indicators, min 10 bullish for entry.

    Computes a bullish/bearish vote from each indicator and returns a
    normalised score in [-1, 1] plus human-readable reasons.
    """

    def score(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        if df is None or len(df) < 5:
            return 0.0, []
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last
        votes: list[float] = []   # each in {-1, -0.5, 0, 0.5, 1}
        reasons: list[str] = []

        # ── Trend Indicators (5) ─────────────────────────────────────────

        # 1. EMA Stack Alignment (9/21/50/200)
        ema9 = last.get("ema_9", 0); ema21 = last.get("ema_21", 0)
        ema50 = last.get("ema_50", 0); ema200 = last.get("ema_200", 0)
        if ema9 > ema21 > ema50 > ema200 and ema200 > 0:
            votes.append(1.0); reasons.append("EMA_stack_bullish")
        elif ema9 < ema21 < ema50 < ema200 and ema200 > 0:
            votes.append(-1.0); reasons.append("EMA_stack_bearish")
        else:
            votes.append(0.0)

        # 2. SuperTrend direction
        st_dir = last.get("supertrend_dir", 0)
        if st_dir == 1:
            votes.append(1.0); reasons.append("SuperTrend_bullish")
        elif st_dir == -1:
            votes.append(-1.0); reasons.append("SuperTrend_bearish")
        else:
            votes.append(0.0)

        # 3. Ichimoku Cloud position
        above_cloud = last.get("ichimoku_above_cloud", 0)
        below_cloud = last.get("ichimoku_below_cloud", 0)
        if above_cloud:
            votes.append(1.0); reasons.append("Ichimoku_above_cloud")
        elif below_cloud:
            votes.append(-1.0); reasons.append("Ichimoku_below_cloud")
        else:
            votes.append(0.0)

        # 4. Parabolic SAR placement
        psar_bull = last.get("psar_bullish", 0)
        if psar_bull == 1.0:
            votes.append(1.0); reasons.append("PSAR_bullish")
        elif psar_bull == 0.0:
            votes.append(-1.0); reasons.append("PSAR_bearish")
        else:
            votes.append(0.0)

        # 5. Aroon Up/Down (25-period)
        aroon_osc = last.get("aroon_osc", 0)
        if aroon_osc > 50:
            votes.append(1.0); reasons.append("Aroon_bullish")
        elif aroon_osc < -50:
            votes.append(-1.0); reasons.append("Aroon_bearish")
        else:
            votes.append(0.0)

        # ── Momentum Oscillators (5) ─────────────────────────────────────

        # 6. RSI (14) — 45-65 bullish zone per spec
        rsi = last.get("rsi_14", 50.0)
        if 45 <= rsi <= 65:
            votes.append(0.5); reasons.append("RSI_bullish_zone")
        elif rsi < 30:
            votes.append(1.0); reasons.append("RSI_oversold")
        elif rsi > 70:
            votes.append(-1.0); reasons.append("RSI_overbought")
        elif rsi < 45:
            votes.append(-0.5); reasons.append("RSI_low")
        else:
            votes.append(-0.5); reasons.append("RSI_high")

        # 7. MACD line/signal cross
        macd = last.get("macd", 0); macd_sig = last.get("macd_signal", 0)
        prev_macd = prev.get("macd", 0); prev_macd_sig = prev.get("macd_signal", 0)
        if prev_macd <= prev_macd_sig and macd > macd_sig:
            votes.append(1.0); reasons.append("MACD_bullish_cross")
        elif prev_macd >= prev_macd_sig and macd < macd_sig:
            votes.append(-1.0); reasons.append("MACD_bearish_cross")
        elif macd > macd_sig:
            votes.append(0.5)
        else:
            votes.append(-0.5)

        # 8. Money Flow Index (14) > 55 bullish
        mfi = last.get("mfi_14", 50.0)
        if mfi > 55:
            votes.append(1.0); reasons.append("MFI_bullish")
        elif mfi < 45:
            votes.append(-1.0); reasons.append("MFI_bearish")
        else:
            votes.append(0.0)

        # 9. Stochastic (14,3,3)
        stoch_k = last.get("stoch_k", 50); stoch_d = last.get("stoch_d", 50)
        prev_k = prev.get("stoch_k", 50); prev_d = prev.get("stoch_d", 50)
        if prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 80:
            votes.append(1.0); reasons.append("Stoch_bullish_cross")
        elif prev_k >= prev_d and stoch_k < stoch_d and stoch_k > 20:
            votes.append(-1.0); reasons.append("Stoch_bearish_cross")
        elif stoch_k < 20:
            votes.append(0.5); reasons.append("Stoch_oversold")
        elif stoch_k > 80:
            votes.append(-0.5); reasons.append("Stoch_overbought")
        else:
            votes.append(0.0)

        # 10. CCI (20)
        cci = last.get("cci_20", 0)
        if cci > 100:
            votes.append(1.0); reasons.append("CCI_bullish")
        elif cci < -100:
            votes.append(-1.0); reasons.append("CCI_bearish")
        else:
            votes.append(0.0)

        # ── Additional Confirmation (5) ──────────────────────────────────

        # 11. Williams %R (> -80 bullish per spec)
        wr = last.get("williams_r", -50)
        if wr > -20:
            votes.append(-0.5); reasons.append("WR_overbought")
        elif wr > -80:
            votes.append(0.5); reasons.append("WR_bullish")
        else:
            votes.append(-0.5); reasons.append("WR_oversold")

        # 12. Ultimate Oscillator rising
        ult = last.get("ult_osc", 50)
        prev_ult = prev.get("ult_osc", 50)
        if ult > prev_ult and ult > 50:
            votes.append(1.0); reasons.append("UltOsc_rising")
        elif ult < prev_ult and ult < 50:
            votes.append(-1.0); reasons.append("UltOsc_falling")
        else:
            votes.append(0.0)

        # 13. TRIX zero-line cross
        trix = last.get("trix", 0); prev_trix = prev.get("trix", 0)
        if prev_trix <= 0 and trix > 0:
            votes.append(1.0); reasons.append("TRIX_bullish_cross")
        elif prev_trix >= 0 and trix < 0:
            votes.append(-1.0); reasons.append("TRIX_bearish_cross")
        else:
            votes.append(0.0)

        # 14. Bollinger Bands %B > 0.5 bullish
        bb_pct = last.get("bb_pct", 0.5)
        if bb_pct > 0.5:
            votes.append(0.5); reasons.append("BB_pct_bullish")
        elif bb_pct < 0.5:
            votes.append(-0.5); reasons.append("BB_pct_bearish")
        else:
            votes.append(0.0)

        # 15. Keltner Channel position
        close = last.get("close", 0)
        kc_upper = last.get("kc_upper", 0); kc_lower = last.get("kc_lower", 0)
        if close > kc_upper:
            votes.append(1.0); reasons.append("KC_breakout_bullish")
        elif close < kc_lower:
            votes.append(-1.0); reasons.append("KC_breakout_bearish")
        else:
            votes.append(0.0)

        # ── ADX trend strength amplifier ──────────────────────────────────
        adx = last.get("adx", 0.0)
        plus_di = last.get("plus_di", 0.0)
        minus_di = last.get("minus_di", 0.0)
        if adx > 25:
            di_signal = 0.6 if plus_di > minus_di else -0.6
            votes.append(di_signal)
            reasons.append(f"ADX_strong_{adx:.0f}")

        # ── Volume confirmation amplifier ─────────────────────────────────
        vol_ratio = last.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            current_avg = np.mean(votes) if votes else 0.0
            vol_boost = 0.3 * (1.0 if current_avg >= 0 else -1.0)
            votes.append(vol_boost)
            reasons.append(f"vol_confirm_{vol_ratio:.1f}x")

        if not votes:
            return 0.0, reasons

        # Bullish vote count for logging
        bullish_count = sum(1 for v in votes if v > 0)
        bearish_count = sum(1 for v in votes if v < 0)

        return float(np.clip(np.mean(votes), -1.0, 1.0)), reasons


class MLScorer:

    def __init__(self, model_path: str = "ml_model.pkl") -> None:
        self._model = None
        self._feature_names = None
        self._model_loaded = False
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        import os
        import hashlib
        import hmac
        try:
            if os.path.exists(self._model_path):
                # P0: Verify model file integrity before loading to prevent
                # arbitrary code execution via pickle deserialization.
                expected_digest = os.getenv("ML_MODEL_SHA256", "").strip()
                if expected_digest:
                    with open(self._model_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if not hmac.compare_digest(file_hash, expected_digest):
                        logger.critical(
                            "Model file integrity check FAILED — refusing to load {}",
                            self._model_path,
                        )
                        return
                    logger.info("Model file SHA-256 verified")
                else:
                    logger.warning(
                        "ML_MODEL_SHA256 env var not set — model integrity not verified. "
                        "Set ML_MODEL_SHA256=<hex> in production.",
                    )
                import joblib
                model_data = joblib.load(self._model_path)
                self._model = model_data.get("model")
                self._feature_names = model_data.get("feature_names")
                self._model_loaded = True
                logger.info(f"MLScorer loaded model from {self._model_path}")
            else:
                logger.warning(f"MLScorer: model file {self._model_path} not found, using heuristics")
        except Exception as e:
            logger.error(f"MLScorer: failed to load model: {e}")
            self._model = None
            self._feature_names = None
            self._model_loaded = False

    def score(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 20:
            return 0.0
        if self._model is not None and self._feature_names is not None:
            try:
                last = df.iloc[-1]
                row = [float(last[f]) if f in df.columns else 0.0 for f in self._feature_names]
                X = np.array(row, dtype=float).reshape(1, -1)
                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba(X)
                    # Use probability of class 1 (up)
                    score = float(proba[0, 1]) * 2 - 1  # scale to [-1, 1]
                    return float(np.clip(score, -1.0, 1.0))
                elif hasattr(self._model, "predict"):
                    pred = self._model.predict(X)
                    return float(np.clip(pred[0], -1.0, 1.0))
            except Exception as e:
                logger.error(f"MLScorer: model prediction failed: {e}")
        # Fallback to heuristic
        return self._heuristic_score(df)

    def _heuristic_score(self, df: pd.DataFrame) -> float:
        """Multi-feature heuristic when no trained model is available.

        Combines momentum (z-score), mean-reversion (RSI), and volatility-adjusted
        trend strength instead of relying on a single z-score.
        """
        if df is None or len(df) < 20:
            return 0.0
        last = df.iloc[-1]

        # ── 1. Momentum: z-score of 5-bar returns ────────────────────────
        returns_5 = last.get("returns_5", 0.0)
        vol = last.get("vol_20", 0.3)
        momentum = 0.0
        if vol > 0:
            z_score = returns_5 / (vol / (252 ** 0.5))
            # Significance deadband: ignore noise below 0.5 standard deviations
            if abs(z_score) >= 0.5:
                momentum = float(np.clip(z_score * 0.15, -0.5, 0.5))

        # ── 2. Mean-reversion signal: RSI extremes ───────────────────────
        rsi = last.get("rsi_14", 50.0)
        reversion = 0.0
        if rsi < 30:
            reversion = 0.4   # Deeply oversold → expect bounce
        elif rsi < 40:
            reversion = 0.2
        elif rsi > 70:
            reversion = -0.4  # Deeply overbought → expect pullback
        elif rsi > 60:
            reversion = -0.2

        # ── 3. Trend strength: EMA alignment ─────────────────────────────
        ema12 = last.get("ema_12", 0.0)
        ema26 = last.get("ema_26", 0.0)
        trend = 0.0
        if ema12 > 0 and ema26 > 0:
            trend_pct = (ema12 - ema26) / ema26
            trend = float(np.clip(trend_pct * 10, -0.5, 0.5))

        # ── 4. Combine with adaptive weighting ───────────────────────────
        # In strong trends, weight momentum+trend; in ranges, weight reversion
        adx = last.get("adx", 20.0)
        if adx > 25:
            # Trending: momentum 40%, trend 40%, reversion 20%
            combined = momentum * 0.4 + trend * 0.4 + reversion * 0.2
        else:
            # Ranging: reversion 50%, momentum 20%, trend 30%
            combined = reversion * 0.5 + momentum * 0.2 + trend * 0.3

        return float(np.clip(combined, -1.0, 1.0))


class NewsScorer:
    """Converts news sentiment events into a directional score in [-1, 1].

    Maintains a decay-weighted rolling average so recent headlines
    dominate while stale news fades.
    """

    def __init__(self, decay_seconds: float = 3600.0) -> None:
        self._decay_seconds = decay_seconds
        self._events: list[tuple[float, float]] = []  # (timestamp, sentiment)

    def ingest(self, sentiment: float, ts: float | None = None) -> None:
        now = ts or time.time()
        self._events.append((now, float(np.clip(sentiment, -1.0, 1.0))))
        # prune events older than 4× decay window
        cutoff = now - self._decay_seconds * 4
        self._events = [(t, s) for t, s in self._events if t >= cutoff]

    def score(self) -> float:
        if not self._events:
            return 0.0
        now = time.time()
        weighted_sum = 0.0
        weight_sum = 0.0
        for ts, sent in self._events:
            age = max(0.0, now - ts)
            w = np.exp(-age / self._decay_seconds)
            weighted_sum += w * sent
            weight_sum += w
        if weight_sum == 0:
            return 0.0
        return float(np.clip(weighted_sum / weight_sum, -1.0, 1.0))


class OrderbookScorer:
    """Computes directional and liquidity signals from orderbook snapshots.

    Positive = buy-side dominance (bullish), negative = sell pressure.
    """

    def __init__(self) -> None:
        self._last_imbalance: float = 0.0
        self._last_imbalance_by_symbol: dict[str, float] = {}
        self._last_depth_ratio_by_symbol: dict[str, float] = {}

    def update(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        *,
        symbol: str | None = None,
        min_depth_usd: float = 0.0,
    ) -> float:
        """Accept orderbook levels as [(price, qty), …] for bids and asks."""
        bid_volume = sum(qty for _, qty in bids) if bids else 0.0
        ask_volume = sum(qty for _, qty in asks) if asks else 0.0
        total = bid_volume + ask_volume
        if total == 0:
            imbalance = 0.0
        else:
            imbalance = float(np.clip((bid_volume - ask_volume) / total, -1.0, 1.0))

        self._last_imbalance = imbalance
        if symbol:
            self._last_imbalance_by_symbol[symbol] = imbalance
            if min_depth_usd > 0:
                total_notional = (
                    sum(price * qty for price, qty in bids) +
                    sum(price * qty for price, qty in asks)
                )
                self._last_depth_ratio_by_symbol[symbol] = total_notional / min_depth_usd

        return imbalance

    def score(self, symbol: str | None = None) -> float:
        if symbol is not None:
            return self._last_imbalance_by_symbol.get(symbol, self._last_imbalance)
        return self._last_imbalance

    def depth_ratio(self, symbol: str | None = None) -> float:
        if symbol is None:
            if not self._last_depth_ratio_by_symbol:
                return 0.0
            return list(self._last_depth_ratio_by_symbol.values())[-1]
        return self._last_depth_ratio_by_symbol.get(symbol, 0.0)

    def depth_score(self, symbol: str | None = None) -> int:
        ratio = self.depth_ratio(symbol)
        if ratio >= 5.0:
            return 100
        if ratio >= 2.0:
            return 50
        return 0


class SignalGenerator:
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        data_manager: DataManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.data_manager = data_manager
        self._technical_scorer = TechnicalScorer()
        self._news_scorer = NewsScorer()
        self._orderbook_scorer = OrderbookScorer()
        self._strategy_selector = StrategySelector()
        self._smc_analyzer = SmartMoneyAnalyzer()
        self._volume_analyzer = VolumeProfileAnalyzer()
        self._running = False
        self._sentiment_score: float = 0.0
        self._sentiment_ts: float = 0.0
        self._macro_score: float = 0.0
        self._macro_ts: float = 0.0
        self._funding_by_symbol: dict[str, tuple[float, float]] = {}  # symbol → (rate, ts)
        self._score_ttl: float = 3600.0  # seconds before scores go stale
        self._min_factor_magnitude: float = 0.05  # minimum |score| to count as contributing
        self._risk_manager = None  # use set_risk_manager() to wire
        self._last_signal_time: dict[str, float] = {}
        self._min_signal_interval = 60.0
        self._auto_trading_enabled = False
        self._bot_config: dict = {
            "strategy": "ensemble",
            "sizing_mode": "risk_pct",
            "risk_per_trade": 2.0,
            "max_positions": 3,
            "max_drawdown": 10.0,
            "max_leverage": 5,
            "daily_loss_limit": 500.0,
            "trailing_mode": "none",
            "auto_sl_tp": False,
        }

        # ── Advanced: signal momentum tracker ─────────────────────────────
        # Tracks the last N composite scores per symbol to detect momentum
        self._score_history: dict[str, deque[float]] = {}
        self._score_history_len = 6   # track last 6 signals per symbol

        # ── Advanced: consecutive loss cooldown ───────────────────────────
        self._consecutive_losses: dict[str, int] = {}   # per-symbol loss streak
        self._max_loss_cooldown = 3   # after N losses, min_score *= 2

        # ── Advanced: correlation-aware position tracking ────────────────
        self._open_directions: dict[str, str] = {}  # symbol → last signal direction

        signals_cfg = config.get_value("signals") or {}
        self._ml_weight = float(signals_cfg.get("ml_weight", 0.25))
        self._tech_weight = float(signals_cfg.get("technical_weight", 0.30))
        self._sentiment_weight = float(signals_cfg.get("sentiment_weight", 0.10))
        self._macro_weight = float(signals_cfg.get("macro_weight", 0.10))
        self._news_weight = float(signals_cfg.get("news_weight", 0.15))
        self._orderbook_weight = float(signals_cfg.get("orderbook_weight", 0.10))
        self._min_score = float(signals_cfg.get("min_score_threshold", 0.65))
        self._min_factors = int(signals_cfg.get("min_contributing_factors", 3))
        self._primary_tf = signals_cfg.get("primary_timeframe", "15m")
        self._confirmation_tfs: list[str] = list(signals_cfg.get("confirmation_timeframes", ["1h", "4h"]))
        self._htf_threshold = float(signals_cfg.get("htf_threshold", 3.0))
        self._ml_model_path = str(signals_cfg.get("model_path", "ml_model.pkl"))
        self._ml_model_type = str(signals_cfg.get("ml_model_type", "lightgbm"))
        self._ml_bootstrap_min_rows = int(signals_cfg.get("ml_bootstrap_min_rows", 120))
        self._ml_bootstrap_retry_seconds = float(signals_cfg.get("ml_bootstrap_retry_seconds", 15.0))
        self._ml_retrain_interval_seconds = float(signals_cfg.get("ml_retrain_interval_seconds", 21600.0))
        self._model_trainer = ModelTrainer(model_type=self._ml_model_type)
        use_adaptive_ml = bool(signals_cfg.get("use_adaptive_ml", True))
        if use_adaptive_ml and _ADAPTIVE_ML_AVAILABLE:
            try:
                self._ml_scorer = AdaptiveMLScorer()
                self._adaptive_ml = True
                logger.info("AdaptiveMLScorer active — regime-aware ensemble + Kelly + drift detection")
            except Exception as exc:
                logger.warning("AdaptiveMLScorer init failed ({}), falling back to MLScorer", exc)
                self._ml_scorer = MLScorer(model_path=self._ml_model_path)
                self._adaptive_ml = False
        else:
            self._ml_scorer = MLScorer(model_path=self._ml_model_path)
            self._adaptive_ml = False
        self._session_filter = SessionFilter() if _SESSION_FILTER_AVAILABLE else None
        self._smc_selector = SMCStrategySelector() if _SMC_SELECTOR_AVAILABLE else None
        self._last_smc_signal_per_symbol: dict[str, dict[str, Any]] = {}
        self._last_master_score_per_symbol: dict[str, dict[str, Any]] = {}
        self._pair_registry: Any = PairRegistry() if _PAIR_REGISTRY_AVAILABLE else None
        if self._pair_registry is not None:
            try:
                universe = config.get_value("signal_generator.universe") or config.get_value("universe") or []
                if isinstance(universe, list):
                    for sym in universe:
                        if isinstance(sym, str):
                            self._pair_registry.register(sym, daily_volume_usd=1_000_000_000)
                        elif isinstance(sym, dict):
                            self._pair_registry.register(
                                sym.get("symbol", ""),
                                daily_volume_usd=float(sym.get("volume_usd", 1_000_000_000)),
                            )
            except Exception as exc:
                logger.debug("PairRegistry seed failed: {}", exc)
        self._estop: Any = EStopManager(self.event_bus) if _ESTOP_AVAILABLE else None
        self._ml_bootstrap_task: asyncio.Task | None = None
        self._last_ml_train_ts: float = 0.0
        risk_cfg = config.get_value("risk") or {}
        self._min_orderbook_depth_usd = float(
            risk_cfg.get("min_orderbook_depth_usd", risk_cfg.get("min_liquidity_usd", 0.0)) or 0.0
        )

        # ── Pipeline status tracking for dashboard API ────────────────────
        self._last_layer_status: dict[str, str] = {
            "session_filter": "PENDING",
            "session_filter_detail": "waiting for first evaluation",
            "htf_trend": "PENDING",
            "htf_trend_detail": "waiting for first evaluation",
            "technical_confluence": "PENDING",
            "technical_confluence_detail": "waiting for first evaluation",
            "smart_money_concepts": "PENDING",
            "smart_money_concepts_detail": "waiting for first evaluation",
            "volume_flow": "PENDING",
            "volume_flow_detail": "waiting for first evaluation",
            "regime_detection": "PENDING",
            "regime_detection_detail": "waiting for first evaluation",
            "ml_ensemble": "PENDING",
            "ml_ensemble_detail": "waiting for first evaluation",
            "signal_quality": "PENDING",
            "signal_quality_detail": "waiting for first evaluation",
            "risk_gate": "PENDING",
            "risk_gate_detail": "waiting for first evaluation",
        }
        self._last_quality_breakdown: dict[str, Any] = {
            "total": 0,
            "components": {
                "htf_alignment": 0, "signal_type_purity": 0,
                "volume_confirmation": 0, "liquidity_proximity": 0,
                "fvg_ob_overlap": 0, "session_alignment": 0,
                "sentiment_alignment": 0, "onchain_health": 0,
            },
        }
        self._ml_training_summary: dict[str, Any] = {
            "trained": bool(self._ml_scorer._model_loaded),
            "model_path": self._ml_model_path,
            "rows": 0,
            "feature_count": len(self._ml_scorer._feature_names or []),
            "last_train_ts": 0.0,
            "reason": "loaded_from_disk" if self._ml_scorer._model_loaded else "model_missing",
        }

        ai_cfg = config.get_value("ai_agent") or {}
        self._ai_agent = TradingAIAgent(
            enabled=bool(ai_cfg.get("enabled", True)),
            mode=str(ai_cfg.get("mode", "full")),
            min_confidence=float(ai_cfg.get("min_confidence", 0.55)),
            min_quality_score=int(ai_cfg.get("min_quality_score", self._get_quality_threshold())),
            min_risk_reward=float(ai_cfg.get("min_risk_reward", 1.0)),
            provider=str(ai_cfg.get("provider", "local")),
            model=str(ai_cfg.get("model", "claude-3-5-sonnet-latest")),
            api_key=str(ai_cfg.get("api_key", "")),
            timeout_seconds=float(ai_cfg.get("timeout_seconds", 8.0)),
            remote_weight=float(ai_cfg.get("remote_weight", 0.35)),
        )

    # ── Advanced helpers ──────────────────────────────────────────────────

    def _update_layer_status(self, **kwargs: int | str) -> None:
        """Update pipeline layer status dict for dashboard display.

        Maps pipeline layer scores (l0-l7) to dashboard layer names.
        Spec: L1=Session, L2=HTF Trend, L3=Technical, L4=SMC, L5=Volume, L6=Regime, L7=ML, L8=Quality, L9=Risk
        Dashboard: 1=Session, 2=HTF Trend, 3=Technical, 4=SMC, 5=Volume, 6=Regime, 7=ML, 8=Quality, 9=Risk
        """
        # Pipeline layer → dashboard key mapping (spec-compliant numbering)
        mapping = {
            "l0": "session_filter",       # L1 Session → Dashboard "Session Filter"
            "l1": "htf_trend",            # L2 HTF Trend → Dashboard "HTF Trend"
            "l2": "technical_confluence", # L3 Technical → Dashboard "Technical Confluence"
            "l3": "smart_money_concepts", # L4 SMC → Dashboard "Smart Money Concepts"
            "l4": "volume_flow",          # L5 Volume → Dashboard "Volume Flow"
            "l5": "regime_detection",     # L6 Regime → Dashboard "Regime Detection"
            "l6": "ml_ensemble",          # L7 ML → Dashboard "ML Ensemble"
            "l7": "signal_quality",       # L8 Quality → Dashboard "Signal Quality"
            "risk_gate": "risk_gate",     # L9 Risk gate → Dashboard "Risk Gate"
        }
        for key, dashboard_name in mapping.items():
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, (int, float)):
                    if val >= 70:
                        self._last_layer_status[dashboard_name] = "PASS"
                    elif val >= 40:
                        self._last_layer_status[dashboard_name] = "WEAK"
                    else:
                        self._last_layer_status[dashboard_name] = "FAIL"
                else:
                    self._last_layer_status[dashboard_name] = str(val)
                self._last_layer_status[f"{dashboard_name}_detail"] = f"score={val}"

    def _is_paper_mode(self) -> bool:
        """Return whether the current runtime is in paper/demo mode."""
        return bool(self.config.get_value("system", "paper_mode", default=getattr(self.config, "paper_mode", True)))

    def _get_quality_threshold(self) -> int:
        """Return the active quality gate for the current runtime mode."""
        return 30 if self._is_paper_mode() else 65

    def _get_technical_threshold(self) -> int:
        """Return the technical confluence threshold for the active mode."""
        return 30 if self._is_paper_mode() else 65

    def _get_volume_threshold(self) -> int:
        """Return the volume-flow threshold for the active mode."""
        return 20 if self._is_paper_mode() else 60

    def _get_smc_threshold(self) -> int:
        """Return the minimum SMC confluence for the active mode."""
        return 10 if self._is_paper_mode() else 20

    def get_ml_status(self) -> dict[str, Any]:
        """Return current ML model status and latest training summary."""
        return {
            "loaded": bool(self._ml_scorer._model_loaded),
            "model_path": self._ml_model_path,
            "model_type": self._ml_model_type,
            "feature_count": len(self._ml_scorer._feature_names or []),
            "last_train_ts": float(self._last_ml_train_ts),
            "training": dict(self._ml_training_summary),
        }

    def get_agent_status(self) -> dict[str, Any]:
        """Return status for the attached supervisory AI agent."""
        if self._ai_agent is None:
            return {"attached": False, "enabled": False, "mode": "off"}
        status = self._ai_agent.get_status()
        status["auto_trading_enabled"] = bool(self._auto_trading_enabled)
        return status

    def get_session_status(self) -> dict[str, Any]:
        """Return active ICT killzones / sessions for the dashboard."""
        if self._session_filter is None:
            return {"enabled": False, "active_killzones": [], "active_sessions": [], "score": 0.0}
        ctx = self._session_filter.session_context()
        allowed, reason = self._session_filter.should_trade()
        ctx["enabled"] = True
        ctx["allowed"] = allowed
        ctx["reason"] = reason
        return ctx

    def get_smc_signal(self, symbol: str | None = None) -> dict[str, Any]:
        """Return latest SMC strategy signal per symbol (or for a given symbol)."""
        if symbol:
            return self._last_smc_signal_per_symbol.get(symbol, {"symbol": symbol, "signal": None})
        return dict(self._last_smc_signal_per_symbol)

    def get_master_score(self, symbol: str | None = None) -> dict[str, Any]:
        """Return the latest V6 master score per symbol (or for a given symbol)."""
        if symbol:
            return self._last_master_score_per_symbol.get(symbol, {"symbol": symbol, "score": None})
        return dict(self._last_master_score_per_symbol)

    def get_pair_registry_snapshot(self) -> dict[str, Any]:
        """Return PairRegistry snapshot for the dashboard."""
        if self._pair_registry is None:
            return {"enabled": False, "pairs": {}}
        return {"enabled": True, "pairs": self._pair_registry.get_snapshot()}

    def get_estop_status(self) -> dict[str, Any]:
        """Return EStopManager status."""
        if self._estop is None:
            return {"enabled": False, "active": False}
        status = self._estop.get_status()
        status["enabled"] = True
        return status

    async def trigger_estop(self, reason: str = "manual", triggered_by: str = "dashboard") -> dict[str, Any]:
        if self._estop is None:
            return {"ok": False, "reason": "estop_unavailable"}
        await self._estop.trigger(reason, triggered_by=triggered_by)
        return {"ok": True, "status": self._estop.get_status()}

    async def release_estop(self, released_by: str = "dashboard") -> dict[str, Any]:
        if self._estop is None:
            return {"ok": False, "reason": "estop_unavailable"}
        ok = await self._estop.release(released_by=released_by)
        return {"ok": ok, "status": self._estop.get_status()}

    def configure_agent(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update AI agent runtime settings and return the new status."""
        payload = payload or {}
        if self._ai_agent is None:
            return {"attached": False, "enabled": False, "mode": "off"}
        self._ai_agent.configure(
            enabled=payload.get("enabled"),
            mode=payload.get("mode"),
            min_confidence=payload.get("min_confidence"),
            min_quality_score=payload.get("min_quality_score"),
            min_risk_reward=payload.get("min_risk_reward"),
            provider=payload.get("provider"),
            model=payload.get("model"),
            api_key=payload.get("api_key"),
            timeout_seconds=payload.get("timeout_seconds"),
            remote_weight=payload.get("remote_weight"),
        )
        return self.get_agent_status()

    def get_strategy_suggestion(self, symbol: str | None = None) -> dict[str, Any]:
        """Return a simple strategy recommendation from the current pipeline state."""
        target_symbol = str(symbol or "BTC/USDT:USDT")
        quality_total = int(self._last_quality_breakdown.get("total", 0) or 0)
        rejected_at = str(self._last_quality_breakdown.get("rejected_at", "") or "")
        layers = dict(self._last_layer_status)
        ml_loaded = bool(self._ml_scorer._model_loaded)

        strategy = "trend_following"
        action = "wait"
        reason = "Pipeline is still below execution quality thresholds."
        suggestions = [
            "Prefer high-quality setups above the live quality threshold.",
            "Keep auto trading aligned with the risk gate and session filter.",
        ]

        if quality_total >= self._get_quality_threshold() and layers.get("risk_gate") == "PASS":
            action = "consider_entry"
            reason = "Quality and risk conditions are aligned for a potential trade."
            suggestions = [
                "Use the trend-following path with strict stop loss.",
                "Let the AI supervisor confirm before full-size entries.",
            ]
        elif "Technical" in rejected_at or layers.get("technical_confluence") == "FAIL":
            strategy = "mean_reversion_watch"
            reason = "Technical confluence is weak, so patience is favored over entry."
            suggestions = [
                "Wait for RSI, EMA, and momentum alignment.",
                "Avoid forcing entries during weak confluence.",
            ]
        elif layers.get("volume_flow") == "FAIL":
            strategy = "liquidity_wait"
            reason = "Volume participation is too weak for a strong breakout trade."
            suggestions = [
                "Wait for stronger volume expansion.",
                "Reduce position size if a trade is still taken.",
            ]

        return {
            "symbol": target_symbol,
            "action": action,
            "strategy": strategy,
            "reason": reason,
            "quality_score": quality_total,
            "ml_loaded": ml_loaded,
            "auto_trading_enabled": bool(self._auto_trading_enabled),
            "suggestions": suggestions,
        }

    def get_quality_preview(self, symbol: str | None = None) -> dict[str, Any]:
        """Build a non-trading quality preview from the latest available market snapshot."""
        exchange_cfg = self.config.get_value("exchanges", "binance") or {}
        configured_symbols = list(exchange_cfg.get("symbols", []))
        target_symbol = str(symbol or (configured_symbols[0] if configured_symbols else "BTC/USDT:USDT"))

        try:
            df = self.data_manager.get_dataframe("binance", target_symbol, self._primary_tf)
        except TypeError:
            df = self.data_manager.get_dataframe("binance", target_symbol)
        except Exception:
            df = None

        if df is None or len(df) < 30:
            return {
                "total": 0,
                "components": {
                    "htf_trend": 0,
                    "technical_confluence": 0,
                    "smc_confluence": 0,
                    "volume_flow": 0,
                    "regime": 0,
                    "ml_confidence": 0,
                    "liquidity_depth": 0,
                },
                "preview": True,
                "symbol": target_symbol,
                "reason": "Waiting for enough live candles to evaluate the pipeline.",
            }

        now = time.time()
        session_rule = self._get_session_rule()
        regime_state = self.data_manager.get_regime("binance", target_symbol)
        sentiment = self._sentiment_score if (now - self._sentiment_ts) < self._score_ttl else 0.0

        smc_state = self._smc_analyzer.analyze(df)
        tech_score, _ = self._technical_scorer.score(df)
        ml_score = self._ml_scorer.score(df)
        vol_flow = self._volume_analyzer.analyze(df)

        last_row = df.iloc[-1]
        vol_ratio = float(last_row.get("volume_ratio", 1.0) or 1.0)
        orderbook_depth_ratio = self._orderbook_scorer.depth_ratio(target_symbol)
        liquidity_depth_score = self._orderbook_scorer.depth_score(target_symbol)

        direction_bias = 0.40 * tech_score + 0.30 * smc_state.smc_score + 0.15 * ml_score + 0.15 * vol_flow.flow_score
        direction = "long" if direction_bias >= 0 else "short"

        htf_ok, _, htf_weighted_score, _ = self._check_higher_timeframe_trend("binance", target_symbol, direction)
        if not htf_ok:
            alt_direction = "short" if direction == "long" else "long"
            alt_ok, _, alt_ws, _ = self._check_higher_timeframe_trend("binance", target_symbol, alt_direction)
            if alt_ok or abs(alt_ws) > abs(htf_weighted_score):
                direction = alt_direction
                htf_weighted_score = alt_ws

        htf_score_100 = int((htf_weighted_score + 1.0) / 2.0 * 100) if direction == "long" else int((1.0 - htf_weighted_score) / 2.0 * 100)
        tech_score_100 = int(max(0, min(100, (tech_score + 1.0) / 2.0 * 100)))
        smc_score_100 = int(max(0, min(100, (smc_state.smc_score + 1.0) / 2.0 * 100)))
        volume_score_100 = int(max(0, min(100, (vol_flow.flow_score + 1.0) / 2.0 * 100)))
        ml_confidence = int(max(0, min(100, (ml_score + 1.0) / 2.0 * 100)))
        regime_allows = bool(getattr(regime_state, "tradeable", True)) if regime_state is not None else True
        regime_score = 100 if regime_allows else 0
        signal_type = self._classify_signal_type(smc_state, df)

        total = self._compute_quality_score(
            htf_score=htf_weighted_score,
            signal_type=signal_type,
            vol_ratio=vol_ratio,
            smc_state=smc_state,
            vol_flow=vol_flow,
            session_rule=session_rule,
            sentiment=sentiment,
            regime_state=regime_state,
            direction=direction,
            tech_score_100=float(tech_score_100),
            smc_points=float(smc_score_100),
            volume_score_100=float(volume_score_100),
            regime_allows=regime_allows,
            ml_confidence=float(ml_confidence),
            orderbook_depth_ratio=orderbook_depth_ratio,
        )

        return {
            "total": int(total),
            "components": {
                "htf_trend": int(htf_score_100),
                "technical_confluence": int(tech_score_100),
                "smc_confluence": int(smc_score_100),
                "volume_flow": int(volume_score_100),
                "regime": int(regime_score),
                "ml_confidence": int(ml_confidence),
                "liquidity_depth": int(liquidity_depth_score),
            },
            "preview": True,
            "symbol": target_symbol,
            "signal_type": signal_type.value,
            "reason": "Live preview from the latest market snapshot.",
        }

    async def chat_with_agent(self, message: str) -> dict[str, Any]:
        """Interactive assistant endpoint for the dashboard AI panel."""
        prompt = str(message or "").strip()
        if not prompt:
            return {"success": False, "reply": "Please enter a message."}

        lower = prompt.lower()
        if any(key in lower for key in ["train model", "retrain", "start learning", "learn now"]):
            result = await self.retrain_model_now()
            trained = bool(result.get("trained", False))
            if trained:
                reply = (
                    f"Training completed using {result.get('rows', 0)} rows across {len(result.get('symbols', []))} symbols. "
                    f"Model features: {result.get('feature_count', 0)}."
                )
            else:
                reply = f"Training did not complete: {result.get('reason', 'unknown')}"
            return {"success": trained, "provider": "system", "reply": reply, "training": result}

        if any(key in lower for key in ["turn on auto", "enable auto", "start auto", "resume trading"]):
            self.set_auto_trading(True)
            return {"success": True, "provider": "system", "reply": "Auto trading is now ON."}

        if any(key in lower for key in ["turn off auto", "disable auto", "stop trading", "pause trading"]):
            self.set_auto_trading(False)
            return {"success": True, "provider": "system", "reply": "Auto trading is now OFF."}

        risk_positions = 0
        if self._risk_manager is not None:
            risk_positions = len(getattr(self._risk_manager, "positions", {}))

        context = {
            "auto_trading_enabled": bool(self._auto_trading_enabled),
            "open_positions": risk_positions,
            "ml_loaded": bool(self._ml_scorer._model_loaded),
            "ml_training": dict(self._ml_training_summary),
            "quality": dict(self._last_quality_breakdown),
            "layers": dict(self._last_layer_status),
        }
        if self._ai_agent is None:
            return {"success": True, "provider": "system", "reply": "AI agent is not attached."}
        return self._ai_agent.chat(prompt, context=context)

    def _collect_training_dataframe(self) -> tuple[pd.DataFrame, list[str]]:
        """Collect numeric historical features for ML training."""
        exchange_cfg = self.config.get_value("exchanges", "binance") or {}
        symbols = list(exchange_cfg.get("symbols", []))
        frames: list[pd.DataFrame] = []
        used_symbols: list[str] = []

        for idx, symbol in enumerate(symbols):
            try:
                df = self.data_manager.get_dataframe("binance", symbol, self._primary_tf)
            except TypeError:
                df = self.data_manager.get_dataframe("binance", symbol)
            except Exception:
                df = None

            if df is None or len(df) < self._ml_bootstrap_min_rows:
                continue

            frame = df.copy()
            frame["symbol_id"] = float(idx)
            numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            required_cols = {"open", "high", "low", "close", "volume", "symbol_id"}
            min_non_null = max(50, int(len(numeric) * 0.60))
            keep_cols = [
                col for col in numeric.columns
                if col in required_cols or int(numeric[col].notna().sum()) >= min_non_null
            ]
            if "close" not in keep_cols:
                continue

            frame = numeric[keep_cols].ffill().dropna()
            if len(frame) < self._ml_bootstrap_min_rows:
                continue

            frames.append(frame)
            used_symbols.append(symbol)

        if not frames:
            return pd.DataFrame(), []
        return pd.concat(frames, ignore_index=True), used_symbols

    def _bootstrap_ml_model(self, model_path: str | None = None) -> dict[str, Any]:
        """Train and load an ML model from the currently available history."""
        model_path = model_path or self._ml_model_path
        dataset, used_symbols = self._collect_training_dataframe()
        rows = int(len(dataset))

        if dataset.empty or rows < self._ml_bootstrap_min_rows:
            result = {
                "trained": False,
                "model_path": model_path,
                "rows": rows,
                "feature_count": 0,
                "symbols": used_symbols,
                "reason": "insufficient_data",
                "last_train_ts": float(self._last_ml_train_ts),
            }
            self._ml_training_summary = result
            return result

        try:
            model_id = f"live_bootstrap_{int(time.time())}"
            trained = self._model_trainer.train_model(
                dataset,
                target_col="returns",
                perform_cv=True,
                model_id=model_id,
            )
            if not self._model_trainer.save_model(model_id, model_path):
                raise RuntimeError("model_save_failed")

            self._ml_scorer = MLScorer(model_path=model_path)
            self._last_ml_train_ts = time.time()
            result = {
                "trained": bool(self._ml_scorer._model_loaded),
                "model_path": model_path,
                "rows": rows,
                "feature_count": len(trained.feature_names),
                "symbols": used_symbols,
                "metrics": {
                    "accuracy": float(trained.metrics.accuracy),
                    "precision": float(trained.metrics.precision),
                    "recall": float(trained.metrics.recall),
                    "f1": float(trained.metrics.f1),
                },
                "last_train_ts": float(self._last_ml_train_ts),
                "reason": "trained_from_history",
            }
            self._ml_training_summary = result
            logger.info(
                "ML bootstrap completed: rows={} features={} f1={:.3f}",
                rows,
                len(trained.feature_names),
                float(trained.metrics.f1),
            )
            return result
        except Exception as exc:
            logger.error("ML bootstrap training failed: {}", exc)
            result = {
                "trained": False,
                "model_path": model_path,
                "rows": rows,
                "feature_count": 0,
                "symbols": used_symbols,
                "reason": str(exc),
                "last_train_ts": float(self._last_ml_train_ts),
            }
            self._ml_training_summary = result
            return result

    async def retrain_model_now(self) -> dict[str, Any]:
        """Trigger model retraining in a worker thread."""
        return await asyncio.to_thread(self._bootstrap_ml_model)

    async def _bootstrap_ml_model_loop(self) -> None:
        """Background loop that trains a model once enough history is available."""
        if self._ml_scorer._model_loaded:
            return

        while self._running:
            result = await asyncio.to_thread(self._bootstrap_ml_model)
            if result.get("trained"):
                return
            await asyncio.sleep(self._ml_bootstrap_retry_seconds)

    def _get_regime_weights(self, regime: MarketRegime | None) -> dict[str, float]:
        """Return factor weights adapted to current market regime.

        When auto-trading overrides are active (e.g. paper mode), blend the
        regime profile with the configured instance weights so that overrides
        for tech/ml are respected.
        """
        configured = {
            "technical": self._tech_weight, "ml": self._ml_weight,
            "sentiment": self._sentiment_weight, "macro": self._macro_weight,
            "news": self._news_weight, "orderbook": self._orderbook_weight,
        }
        if regime and regime in _REGIME_WEIGHT_PROFILES:
            weights = dict(_REGIME_WEIGHT_PROFILES[regime])
            # Respect explicitly zeroed weights (e.g. sentiment=0 in paper mode)
            # and blend tech/ml with configured overrides
            for k in weights:
                cfg_val = configured.get(k)
                if cfg_val is not None and cfg_val == 0.0:
                    weights[k] = 0.0
                elif k in ("technical", "ml") and cfg_val is not None:
                    # Blend: average of profile and configured weight
                    weights[k] = (weights[k] + cfg_val) / 2.0
            return weights
        return configured

    def _get_direction_penalty(
        self, regime: MarketRegime | None, direction: str,
    ) -> float:
        """Multiplier on min_score for counter-trend trades."""
        if regime and regime in _REGIME_DIRECTION_PENALTY:
            return _REGIME_DIRECTION_PENALTY[regime].get(direction, 1.0)
        return 1.0

    def _update_score_momentum(self, key: str, composite: float) -> float:
        """Track composite scores over time, return momentum factor.

        Returns value in [-0.3, +0.3]:
        - Positive when scores are strengthening (getting more extreme)
        - Negative when scores are weakening (fading)
        """
        if key not in self._score_history:
            self._score_history[key] = deque(maxlen=self._score_history_len)
        history = self._score_history[key]
        history.append(composite)
        if len(history) < 3:
            return 0.0
        recent = list(history)
        # Linear regression slope over recent scores
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        return float(np.clip(slope * 2.0, -0.3, 0.3))

    def _check_correlation_block(
        self, symbol: str, direction: str,
    ) -> bool:
        """Block trades creating concentrated correlated exposure.

        Uses dynamic correlation from recent 15m closes (rolling 60 bars),
        falls back to the static major-group heuristic when data missing.
        Returns True if the trade should be BLOCKED.
        """
        CORR_THRESHOLD = 0.70
        MAX_CORRELATED = 2

        if not self._open_directions:
            return False

        same_dir_count_static = 0
        if symbol in _CRYPTO_MAJOR_GROUP:
            for sym, d in self._open_directions.items():
                if sym in _CRYPTO_MAJOR_GROUP and sym != symbol and d == direction:
                    same_dir_count_static += 1

        correlated_dyn = 0
        try:
            base_df = self.data_manager.get_dataframe("binance", symbol, self._primary_tf)
            base_ret = None
            if base_df is not None and len(base_df) >= 30:
                base_ret = base_df["close"].pct_change().dropna().tail(60)
            if base_ret is not None and len(base_ret) >= 20:
                base_arr = base_ret.values
                for sym, d in self._open_directions.items():
                    if sym == symbol or d != direction:
                        continue
                    other_df = self.data_manager.get_dataframe("binance", sym, self._primary_tf)
                    if other_df is None or len(other_df) < 30:
                        continue
                    other_ret = other_df["close"].pct_change().dropna().tail(60).values
                    n = min(len(base_arr), len(other_ret))
                    if n < 20:
                        continue
                    corr = float(np.corrcoef(base_arr[-n:], other_ret[-n:])[0, 1])
                    if not np.isfinite(corr):
                        continue
                    if corr >= CORR_THRESHOLD:
                        correlated_dyn += 1
        except Exception as exc:
            logger.debug("Correlation dyn calc error for {}: {}", symbol, exc)

        total = max(same_dir_count_static, correlated_dyn)
        if total >= MAX_CORRELATED:
            logger.debug(
                "Correlation block: {} {} — dyn={} static={}",
                symbol, direction, correlated_dyn, same_dir_count_static,
            )
            return True
        return False

    def _get_loss_cooldown_multiplier(self, key: str) -> float:
        """After consecutive losses, require higher score to re-enter."""
        streak = self._consecutive_losses.get(key, 0)
        if streak >= self._max_loss_cooldown:
            return 2.0  # Double the min_score requirement
        if streak >= 2:
            return 1.5  # 50% higher
        return 1.0

    def record_trade_result(self, symbol: str, is_win: bool) -> None:
        """Called externally when a paper/live trade closes.

        Updates consecutive loss tracker for cooldown.
        """
        if is_win:
            self._consecutive_losses.pop(symbol, None)
        else:
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        # Clean up direction tracking
        self._open_directions.pop(symbol, None)

    def set_auto_trading(self, enabled: bool) -> None:
        prev = self._auto_trading_enabled
        self._auto_trading_enabled = enabled
        if prev != enabled:
            logger.info("Auto-trading toggled: {} → {}", prev, enabled)

    def set_risk_manager(self, risk_manager: Any) -> None:
        """Wire risk manager for position counting. Use this instead of accessing _risk_manager directly."""
        self._risk_manager = risk_manager
        logger.debug("SignalGenerator: risk_manager wired")

    @property
    def auto_trading_enabled(self) -> bool:
        return self._auto_trading_enabled

    def _check_higher_timeframe_trend(
        self, exchange: str, symbol: str, direction: str,
    ) -> tuple[bool, str, float, int]:
        """L2 HTF Trend — Spec Layer 2: Higher Timeframe Weighted Agreement.

        Spec §4: Daily(3x) + 4H(2x) + 1H(1x) + 15m(1x) = 7 max.
        - Daily: Close > EMA50 AND BOS confirmed → +3 bullish
        - 4H: SuperTrend green → +2 bullish
        - 1H: Price > anchored VWAP → +1 bullish
        - 15m: EMA9 > EMA21 → +1 bullish

        Threshold: weighted_sum ≥ 5 for long, ≤ -5 for short (HARD GATE).
        Returns (ok, reason, normalized_score [-1,1], agreement_count).
        """
        weighted_sum = 0.0
        max_score = 7.0
        agreement_count = 0
        detail_parts: list[str] = []
        htf_frames_available = 0

        # ── Daily (3 points): Close > EMA50 + BOS ────────────────────────
        daily_df = self.data_manager.get_dataframe(exchange, symbol, "1d")
        if daily_df is not None and len(daily_df) >= 50:
            htf_frames_available += 1
            daily_last = daily_df.iloc[-1]
            daily_close = float(daily_last.get("close", 0))
            ema50 = float(daily_last.get("ema_50", 0))
            # BOS: last swing high broken (bullish) or swing low broken (bearish)
            bos_bullish = False
            bos_bearish = False
            if len(daily_df) >= 5:
                recent_high = float(daily_df["high"].iloc[-5:-1].max())
                recent_low = float(daily_df["low"].iloc[-5:-1].min())
                if daily_close > recent_high:
                    bos_bullish = True
                if daily_close < recent_low:
                    bos_bearish = True
            if daily_close > ema50 and bos_bullish:
                weighted_sum += 3.0
                detail_parts.append("1d:bull(3x)")
                if direction == "long":
                    agreement_count += 1
            elif daily_close < ema50 and bos_bearish:
                weighted_sum -= 3.0
                detail_parts.append("1d:bear(3x)")
                if direction == "short":
                    agreement_count += 1
            elif daily_close > ema50:
                weighted_sum += 1.5  # Partial: above EMA50 but no BOS
                detail_parts.append("1d:weak_bull(partial)")
                if direction == "long":
                    agreement_count += 1
            elif daily_close < ema50:
                weighted_sum -= 1.5
                detail_parts.append("1d:weak_bear(partial)")
                if direction == "short":
                    agreement_count += 1
            else:
                detail_parts.append("1d:neutral")
        else:
            detail_parts.append("1d:no_data")

        # ── 4H (2 points): SuperTrend ────────────────────────────────────
        four_h_df = self.data_manager.get_dataframe(exchange, symbol, "4h")
        if four_h_df is not None and len(four_h_df) >= 20:
            htf_frames_available += 1
            st_dir = float(four_h_df.iloc[-1].get("supertrend_dir", 0))
            if st_dir == 1:
                weighted_sum += 2.0
                detail_parts.append("4h:bull(2x)")
                if direction == "long":
                    agreement_count += 1
            elif st_dir == -1:
                weighted_sum -= 2.0
                detail_parts.append("4h:bear(2x)")
                if direction == "short":
                    agreement_count += 1
            else:
                detail_parts.append("4h:neutral")
        else:
            detail_parts.append("4h:no_data")

        # ── 1H (1 point): Price > VWAP ───────────────────────────────────
        one_h_df = self.data_manager.get_dataframe(exchange, symbol, "1h")
        if one_h_df is not None and len(one_h_df) >= 20:
            htf_frames_available += 1
            last_1h = one_h_df.iloc[-1]
            close_1h = float(last_1h.get("close", 0))
            vwap_1h = float(last_1h.get("vwap", 0))
            if vwap_1h > 0 and close_1h > vwap_1h:
                weighted_sum += 1.0
                detail_parts.append("1h:bull(1x)")
                if direction == "long":
                    agreement_count += 1
            elif vwap_1h > 0 and close_1h < vwap_1h:
                weighted_sum -= 1.0
                detail_parts.append("1h:bear(1x)")
                if direction == "short":
                    agreement_count += 1
            else:
                # Fallback: use EMA if VWAP not available
                ema_f = float(last_1h.get("ema_9", last_1h.get("ema_12", 0)))
                ema_s = float(last_1h.get("ema_21", last_1h.get("ema_26", 0)))
                if ema_f > ema_s:
                    weighted_sum += 1.0
                    detail_parts.append("1h:bull_ema(1x)")
                    if direction == "long":
                        agreement_count += 1
                elif ema_f < ema_s:
                    weighted_sum -= 1.0
                    detail_parts.append("1h:bear_ema(1x)")
                    if direction == "short":
                        agreement_count += 1
                else:
                    detail_parts.append("1h:neutral")
        else:
            detail_parts.append("1h:no_data")

        # ── 15m (1 point): EMA9 > EMA21 ──────────────────────────────────
        fifteen_df = self.data_manager.get_dataframe(exchange, symbol, "15m")
        if fifteen_df is not None and len(fifteen_df) >= 21:
            last_15m = fifteen_df.iloc[-1]
            ema9 = float(last_15m.get("ema_9", 0))
            ema21 = float(last_15m.get("ema_21", 0))
            if ema9 > ema21:
                weighted_sum += 1.0
                detail_parts.append("15m:bull(1x)")
                if direction == "long":
                    agreement_count += 1
            elif ema9 < ema21:
                weighted_sum -= 1.0
                detail_parts.append("15m:bear(1x)")
                if direction == "short":
                    agreement_count += 1
            else:
                detail_parts.append("15m:neutral")
        else:
            detail_parts.append("15m:no_data")

        # Normalize to [-1, 1]
        normalized_score = weighted_sum / max_score

        detail_str = f"weighted={weighted_sum:.1f}/7 [{', '.join(detail_parts)}]"

        if htf_frames_available == 0:
            logger.warning(
                "HTF confirmation bypassed for {}/{} — no higher timeframe data available ({})",
                exchange,
                symbol,
                detail_str,
            )
            return True, f"HTF_bypass_no_data {detail_str}", normalized_score, agreement_count

        # Spec: For long, require weighted_sum >= threshold; for short, <= -threshold
        htf_thresh = self._htf_threshold
        if direction == "long" and weighted_sum >= htf_thresh:
            return True, f"HTF_pass_long {detail_str}", normalized_score, agreement_count
        elif direction == "short" and weighted_sum <= -htf_thresh:
            return True, f"HTF_pass_short {detail_str}", normalized_score, agreement_count
        else:
            return False, f"HTF_fail {detail_str}", normalized_score, agreement_count

    # ── Session & Killzone enforcement (§5) ───────────────────────────────

    def _get_session_rule(self) -> SessionRule | None:
        """Return the active session rule based on current UTC hour."""
        utc_hour = datetime.datetime.utcnow().hour
        for rule in _SESSION_RULES:
            if rule.start_utc <= utc_hour < rule.end_utc:
                return rule
        return None

    def _is_ict_killzone(self) -> bool:
        """Check if current time is within an ICT killzone (§5)."""
        utc_hour = datetime.datetime.utcnow().hour
        return any(start <= utc_hour < end for start, end in _ICT_KILLZONES)

    def _classify_signal_type(self, smc_state: SMCState, df: pd.DataFrame) -> SignalType:
        """Classify the signal into Type A/B/C/D based on SMC state (§3).

        Spec-compliant classification:
        - Type C (HIGHEST priority): Price in 50-62% Fib retracement of FVG
        - Type D: Price in 38-62% of refined OB body (open-to-close)
        - Type B: Liquidity sweep with MSS
        - Type A: 20-period Donchian breakout + Fib 50-62% pullback
        """
        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr_val = float(last.get("atr_14", price * 0.01))

        # Type C: FVG Mitigation — price in 50-62% Fib retracement of gap
        if smc_state.active_fvgs:
            for fvg in smc_state.active_fvgs:
                fvg_high = getattr(fvg, "top", 0)
                fvg_low = getattr(fvg, "bottom", 0)
                gap_range = fvg_high - fvg_low
                if gap_range <= 0:
                    continue
                if getattr(fvg, "direction", "") == "bullish":
                    # Bullish FVG: price retraces DOWN into gap
                    fib_50 = fvg_high - 0.50 * gap_range
                    fib_62 = fvg_high - 0.62 * gap_range
                    if fib_62 <= price <= fib_50:
                        return SignalType.TYPE_C
                else:
                    # Bearish FVG: price retraces UP into gap
                    fib_50 = fvg_low + 0.50 * gap_range
                    fib_62 = fvg_low + 0.62 * gap_range
                    if fib_50 <= price <= fib_62:
                        return SignalType.TYPE_C

        # Type D: Order Block Mitigation — price in 38-62% of refined OB body
        if smc_state.active_order_blocks:
            for ob in smc_state.active_order_blocks:
                ob_high = getattr(ob, "high", 0)
                ob_low = getattr(ob, "low", 0)
                ob_range = ob_high - ob_low
                if ob_range <= 0:
                    continue
                fib_38 = ob_high - 0.38 * ob_range
                fib_62 = ob_high - 0.62 * ob_range
                if fib_62 <= price <= fib_38:
                    return SignalType.TYPE_D

        # Type B: Liquidity Sweep — recent sweep detected + MSS
        if any("liquidity_grab" in r for r in smc_state.reasons):
            return SignalType.TYPE_B

        # Type A: Breakout Pullback — 20-period Donchian breakout + Fib pullback
        if len(df) >= 22:
            donchian_high = float(df["high"].iloc[-21:-1].max())
            donchian_low = float(df["low"].iloc[-21:-1].min())
            # Check breakout candle (any of last 5)
            for i in range(-5, 0):
                candle = df.iloc[i]
                c_close = float(candle.get("close", 0))
                c_range = float(candle.get("high", 0)) - float(candle.get("low", 0))
                c_vol = float(candle.get("volume", 0))
                vol_avg = float(df["volume"].iloc[-21:-1].mean())
                is_breakout_up = c_close > donchian_high
                is_breakout_down = c_close < donchian_low
                strength_ok = c_range >= 1.5 * atr_val
                vol_ok = c_vol > 1.5 * vol_avg if vol_avg > 0 else False
                if (is_breakout_up or is_breakout_down) and strength_ok and vol_ok:
                    # Check current price is in 50-62% Fib retracement of breakout move
                    if is_breakout_up:
                        move_range = c_close - donchian_low
                        fib_50 = c_close - 0.50 * move_range
                        fib_62 = c_close - 0.618 * move_range
                        if fib_62 <= price <= fib_50:
                            return SignalType.TYPE_A
                    else:
                        move_range = donchian_high - c_close
                        fib_50 = c_close + 0.50 * move_range
                        fib_62 = c_close + 0.618 * move_range
                        if fib_50 <= price <= fib_62:
                            return SignalType.TYPE_A

        return SignalType.COMPOSITE

    def _compute_quality_score(
        self, *, htf_score: float, signal_type: SignalType, vol_ratio: float,
        smc_state: SMCState, vol_flow: VolumeFlowState,
        session_rule: SessionRule | None, sentiment: float,
        regime_state: RegimeState | None, direction: str,
        tech_score_100: float = 0.0, smc_points: float = 0.0,
        volume_score_100: float = 0.0, regime_allows: bool = True,
        ml_confidence: float = 0.0, orderbook_depth_ratio: float = 0.0,
        killzone_score: float = 0.0,
    ) -> int:
        """Compute 0-100 quality score — Spec Layer 8.

        Spec §8 weighted formula:
        1. HTF Trend (Layer 2)       20% — normalized weighted score
        2. Technical Confluence (L3)  20% — tech_score (0-100)
        3. SMC Confluence (Layer 4)   20% — smc_points (0-100)
        4. Volume Flow (Layer 5)      15% — volume_score (0-100)
        5. Regime (Layer 6)           10% — 100 if allowed, else 0
        6. ML Confidence (Layer 7)    10% — ml_confidence (0-100)
        7. Liquidity Depth            5%  — orderbook depth ratio
        """
        # 1. HTF Trend (20%)
        # Normalize [-1,1] htf_score to [0,100]
        if direction == "long":
            htf_norm = (htf_score + 1.0) / 2.0 * 100.0
        else:
            htf_norm = (1.0 - htf_score) / 2.0 * 100.0

        # 2. Technical Confluence (20%)
        tech_component = max(0.0, min(100.0, tech_score_100))

        # 3. SMC Confluence (20%)
        smc_component = max(0.0, min(100.0, smc_points))

        # 4. Volume Flow (15%)
        vol_component = max(0.0, min(100.0, volume_score_100))

        # 5. Regime (10%) — 100 if allowed, 0 if not
        regime_component = 100.0 if regime_allows else 0.0

        # 6. ML Confidence (10%)
        ml_component = max(0.0, min(100.0, ml_confidence))

        # 7. Liquidity Depth (5%)
        if orderbook_depth_ratio >= 5.0:
            liq_component = 100.0
        elif orderbook_depth_ratio >= 2.0:
            liq_component = 50.0
        else:
            liq_component = 0.0

        total = (
            htf_norm * 0.20
            + tech_component * 0.20
            + smc_component * 0.20
            + vol_component * 0.15
            + regime_component * 0.10
            + ml_component * 0.10
            + liq_component * 0.05
        )

        kz = max(0.0, min(1.0, killzone_score))
        total += kz * 5.0

        return min(100, max(0, int(total)))

    async def _handle_candle(self, payload: Any) -> None:
        """9-Layer Sequential Signal Confirmation Pipeline — V1.0 Professional Spec.

        L-1: On-Chain (placeholder — not yet integrated)
        L0: Session & Killzone Rules + Sentiment Pre-Filter
        L1: Market Regime Classification (ADX<25 = NO TRADE)
        L2: Market Structure (BOS/CHoCH via SMC)
        L3: Smart Money Concepts (FVG, Order Blocks, Liquidity)
        L4: Momentum Convergence Matrix (15 indicators)
        L5: Volume Profile & Order Flow Confirmation
        L6: Multi-Timeframe Weighted Alignment
        L7: Microstructure Confirmation (candle pattern + volume)
        """
        candle = payload
        if candle.timeframe != self._primary_tf:
            return

        # Skip signal evaluation during historical seeding (DataManager still stores candles)
        if getattr(self.event_bus, '_seeding', False):
            return

        key = f"{candle.exchange}:{candle.symbol}"
        now = time.time()
        if now - self._last_signal_time.get(key, 0) < self._min_signal_interval:
            logger.debug("DIAG {}: throttled (interval)", key)
            return

        # E-Stop gate — block all new signals when emergency stop is active
        if self._estop is not None and self._estop.is_active:
            logger.debug("DIAG {}: ESTOP active ({})", key, self._estop.reason)
            self._update_layer_status(l0="BLOCKED")
            self._last_layer_status["session_filter_detail"] = f"ESTOP:{self._estop.reason}"
            return

        # PairRegistry tier gate — skip Tier 4 pairs
        if self._pair_registry is not None:
            try:
                if self._pair_registry.get(candle.symbol) is None:
                    self._pair_registry.register(candle.symbol, daily_volume_usd=1_000_000_000)
                if not self._pair_registry.is_tradeable(candle.symbol):
                    logger.debug("DIAG {}: pair tier excluded", key)
                    self._update_layer_status(l0="BLOCKED")
                    return
            except Exception:
                pass

        df = self.data_manager.get_dataframe(candle.exchange, candle.symbol, self._primary_tf)
        if df is None or len(df) < 30:
            logger.debug("DIAG {}: df={}, need 30", key, len(df) if df is not None else None)
            return

        # ═══════════════════════════════════════════════════════════════════
        # L0-PRE: SESSION & KILLZONE RULES (§5)
        # ═══════════════════════════════════════════════════════════════════
        session_rule = self._get_session_rule()
        is_killzone = self._is_ict_killzone()

        # Weekend halt (Saturday/Sunday)
        utc_now = datetime.datetime.utcnow()
        if utc_now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            logger.debug("DIAG {}: weekend — no trade", key)
            self._update_layer_status(l0="BLOCKED")
            return

        # No-trade sessions (12-13 UTC, 22-00 UTC)
        if session_rule and session_rule.no_trade:
            logger.debug("DIAG {}: session {} — no trade", key, session_rule.name)
            self._update_layer_status(l0="BLOCKED")
            return

        # ═══════════════════════════════════════════════════════════════════
        # L0: SENTIMENT PRE-FILTER
        # ═══════════════════════════════════════════════════════════════════
        sentiment = self._sentiment_score if (now - self._sentiment_ts) < self._score_ttl else 0.0
        news_score = self._news_scorer.score()

        # ═══════════════════════════════════════════════════════════════════
        # L1: MARKET REGIME CLASSIFICATION
        # ═══════════════════════════════════════════════════════════════════
        regime_state = self.data_manager.get_regime(candle.exchange, candle.symbol)
        current_regime = regime_state.regime if regime_state else None

        # Regime gating moved to L6 inside the pipeline for proper layer tracking.
        # Pre-pipeline: allow all candles through; L6 will handle regime filtering.
        if not regime_state:
            current_regime = MarketRegime.UNKNOWN

        # ── Regime transition detection (§7) ─────────────────────────────
        in_transition = False
        if regime_state and hasattr(regime_state, 'candles_in_state'):
            if regime_state.candles_in_state <= _TRANSITION_CANDLE_WINDOW:
                in_transition = True
                logger.debug("DIAG {}: regime transition (candles_in_state={}), reducing size/tightening SL",
                             key, regime_state.candles_in_state)

        # Regime-gated strategy module evaluation
        strategy_signal: StrategySignal | None = None
        if regime_state and regime_state.regime != MarketRegime.UNKNOWN:
            strategy_signal = self._strategy_selector.select_and_evaluate(df, regime_state)

        # ═══════════════════════════════════════════════════════════════════
        # L2 + L3: MARKET STRUCTURE + SMART MONEY CONCEPTS
        # ═══════════════════════════════════════════════════════════════════
        smc_state: SMCState = self._smc_analyzer.analyze(df)
        smc_score = smc_state.smc_score  # [-1, 1]

        # ═══════════════════════════════════════════════════════════════════
        # L4: MOMENTUM CONVERGENCE MATRIX (15 indicators)
        # ═══════════════════════════════════════════════════════════════════
        tech_score, tech_reasons = self._technical_scorer.score(df)
        ml_score = self._ml_scorer.score(df)
        ob_score = self._orderbook_scorer.score(candle.symbol)

        # TTL: zero out stale macro scores
        macro = self._macro_score if (now - self._macro_ts) < self._score_ttl else 0.0

        # ═══════════════════════════════════════════════════════════════════
        # L5: VOLUME PROFILE & ORDER FLOW CONFIRMATION
        # ═══════════════════════════════════════════════════════════════════
        vol_flow: VolumeFlowState = self._volume_analyzer.analyze(df)
        flow_score = vol_flow.flow_score  # [-1, 1]

        # Normalize scores to 0-100 for layer computations
        ml_norm = (ml_score + 1) / 2 * 100  # [-1,1] → [0,100]
        tech_norm = (tech_score + 1) / 2 * 100
        vol_ratio = float(df.iloc[-1].get("volume_ratio", 1.0))
        orderbook_depth_ratio = self._orderbook_scorer.depth_ratio(candle.symbol)
        liquidity_depth_score = self._orderbook_scorer.depth_score(candle.symbol)

        # ═══════════════════════════════════════════════════════════════════
        # V6.0 SPEC-COMPLIANT 9-LAYER SEQUENTIAL HARD-GATE PIPELINE
        # Each layer is a HARD GATE per spec (reject immediately on fail).
        # Layer numbering matches spec: L1=Session, L2=HTF, L3=Tech, L4=SMC,
        # L5=Volume, L6=Regime, L7=ML, L8=Quality, L9=Risk
        # ═══════════════════════════════════════════════════════════════════

        # ── L1 score: Session (already passed above) ─────────────────────
        l1_session_score = 100 if (session_rule and session_rule.size_multiplier >= 1.0) else (
            70 if session_rule and not session_rule.no_trade else 0)

        # ── L1 Regime pre-check scoring ──────────────────────────────────
        last_row = df.iloc[-1]
        adx_val = float(last_row.get("adx_14", 0))
        atr_14 = float(last_row.get("atr_14", 0))
        close_price = float(last_row["close"])
        atr_ratio = atr_14 / close_price if close_price > 0 else 0
        bb_width = float(last_row.get("bb_width", 0))

        # ── L2: HTF TREND (HARD GATE per spec) ──────────────────────────
        # Try BOTH directions and pick the one HTF supports
        htf_ok_long, htf_reason_long, htf_ws_long, htf_ac_long = self._check_higher_timeframe_trend(
            candle.exchange, candle.symbol, "long",
        )
        htf_ok_short, htf_reason_short, htf_ws_short, htf_ac_short = self._check_higher_timeframe_trend(
            candle.exchange, candle.symbol, "short",
        )

        # Pick direction: prefer HTF-aligned direction, fallback to indicator direction
        direction_score = (
            0.25 * smc_score + 0.25 * tech_score + 0.15 * ml_score
            + 0.10 * flow_score + 0.10 * sentiment
        )
        if current_regime == MarketRegime.STRONG_TREND_UP:
            direction_score = max(direction_score, 0.15)
        elif current_regime == MarketRegime.STRONG_TREND_DOWN:
            direction_score = min(direction_score, -0.15)
        elif current_regime == MarketRegime.WEAK_TREND_UP:
            direction_score = max(direction_score, 0.05)
        elif current_regime == MarketRegime.WEAK_TREND_DOWN:
            direction_score = min(direction_score, -0.05)

        if htf_ok_long and not htf_ok_short:
            proposed_direction = "long"
        elif htf_ok_short and not htf_ok_long:
            proposed_direction = "short"
        elif htf_ok_long and htf_ok_short:
            proposed_direction = "long" if direction_score > 0 else "short"
        else:
            proposed_direction = "long" if direction_score > 0 else "short"

        if proposed_direction == "long":
            htf_ok, htf_reason, htf_weighted_score, htf_agreement_count = htf_ok_long, htf_reason_long, htf_ws_long, htf_ac_long
        else:
            htf_ok, htf_reason, htf_weighted_score, htf_agreement_count = htf_ok_short, htf_reason_short, htf_ws_short, htf_ac_short

        l2_htf_score = int((htf_weighted_score + 1.0) / 2.0 * 100) if proposed_direction == "long" else int((1.0 - htf_weighted_score) / 2.0 * 100)
        l2_htf_score = max(0, min(100, l2_htf_score))

        # HARD GATE: HTF must pass (weighted_sum >= 5/7)
        if not htf_ok:
            logger.info("DIAG {}: L2 HTF HARD GATE FAIL: {} (score={})", key, htf_reason, l2_htf_score)
            self._update_layer_status(
                l0=l1_session_score, l1=l2_htf_score, l2=0, l3=0,
                l4=0, l5=0, l6=0, l7=0,
            )
            self._last_layer_status["htf_trend"] = "FAIL"
            self._last_layer_status["htf_trend_detail"] = htf_reason
            self._last_quality_breakdown = {
                "total": 0,
                "components": {
                    "htf_trend": l2_htf_score, "technical_confluence": 0,
                    "smc_confluence": 0, "volume_flow": 0,
                    "regime": 0, "ml_confidence": 0, "liquidity_depth": 0,
                },
                "rejected_at": "L2_HTF_Trend",
            }
            return

        # ── L3: TECHNICAL CONFLUENCE (HARD GATE: ≥ 65/100) ──────────────
        # Spec: 12-indicator matrix → 3-group weighted avg
        # Trend group (35%), Momentum group (35%), Volatility group (30%)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        # Trend group (EMA Stack, SuperTrend, Ichimoku, SAR)
        ema9 = float(last.get("ema_9", 0)); ema21 = float(last.get("ema_21", 0))
        ema50 = float(last.get("ema_50", 0))
        if ema9 > ema21 > ema50 and ema50 > 0:
            ema_score = 100
        elif ema21 > ema50:
            ema_score = 60
        else:
            ema_score = 0

        st_dir = float(last.get("supertrend_dir", 0))
        supertrend_score = 100 if st_dir == 1 else 0

        ichimoku_score = 100 if last.get("ichimoku_above_cloud", 0) else 40
        sar_score = 100 if float(last.get("psar_bullish", 0)) == 1.0 else 0

        trend_avg = (ema_score + supertrend_score + ichimoku_score + sar_score) / 4.0

        # Momentum group (RSI, MACD, MFI, Stochastic)
        rsi = float(last.get("rsi_14", 50.0))
        if 55 <= rsi <= 75:
            rsi_score = 100
        elif 50 < rsi < 55 or 75 < rsi < 80:
            rsi_score = 50
        else:
            rsi_score = 0

        macd_val = float(last.get("macd", 0)); macd_sig = float(last.get("macd_signal", 0))
        macd_hist = float(last.get("macd_histogram", 0))
        prev_hist = float(prev.get("macd_histogram", 0))
        if macd_val > macd_sig and macd_hist > prev_hist:
            macd_score = 100
        else:
            macd_score = 30

        mfi = float(last.get("mfi_14", 50.0))
        mfi_score = 100 if mfi > 55 else 40

        stoch_k = float(last.get("stoch_k", 50)); stoch_d = float(last.get("stoch_d", 50))
        if stoch_k > stoch_d and 20 <= stoch_k <= 80:
            stoch_score = 100
        else:
            stoch_score = 50

        momentum_avg = (rsi_score + macd_score + mfi_score + stoch_score) / 4.0

        # Volatility group (Bollinger, ATR expansion, Keltner, Williams %R)
        bb_pct = float(last.get("bb_pct", 0.5))
        bb_upper = float(last.get("bb_upper", 0))
        bb_lower = float(last.get("bb_lower", 0))
        if close_price > bb_upper and bb_upper > 0:
            bb_score = 100
        elif bb_upper > 0 and bb_lower > 0 and bb_pct > 0.5:
            bb_score = 60  # Above BB midline
        else:
            bb_score = 20

        atr_20_mean = float(df["atr_14"].tail(20).mean()) if "atr_14" in df.columns and len(df) >= 20 else atr_14
        atr_expansion_ratio = atr_14 / atr_20_mean if atr_20_mean > 0 else 1.0
        atr_score = 100 if atr_expansion_ratio > 1.2 else (60 if atr_expansion_ratio > 0.8 else 20)

        kc_upper = float(last.get("kc_upper", 0))
        kc_lower = float(last.get("kc_lower", 0))
        kc_mid = (kc_upper + kc_lower) / 2 if kc_upper > 0 and kc_lower > 0 else 0
        if close_price > kc_upper and kc_upper > 0:
            keltner_score = 100
        elif kc_mid > 0 and close_price > kc_mid:
            keltner_score = 60
        else:
            keltner_score = 20

        wr = float(last.get("williams_r", -50.0))
        willr_score = 80 if -80 <= wr <= -20 else 30

        volatility_avg = (bb_score + atr_score + keltner_score + willr_score) / 4.0

        # Weighted total (spec: 35% trend, 35% momentum, 30% volatility)
        l3_tech_score = int(trend_avg * 0.35 + momentum_avg * 0.35 + volatility_avg * 0.30)
        l3_tech_score = max(0, min(100, l3_tech_score))

        technical_threshold = self._get_technical_threshold()
        if l3_tech_score < technical_threshold:
            logger.info("DIAG {}: L3 Technical HARD GATE FAIL: score={} < {} "
                        "(trend={:.0f} mom={:.0f} vol={:.0f})",
                        key, l3_tech_score, technical_threshold, trend_avg, momentum_avg, volatility_avg)
            self._update_layer_status(
                l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=0,
                l4=0, l5=0, l6=0, l7=0,
            )
            self._last_layer_status["technical_confluence"] = "FAIL"
            self._last_layer_status["technical_confluence_detail"] = f"score={l3_tech_score}<{technical_threshold}"
            self._last_quality_breakdown = {
                "total": 0,
                "components": {
                    "htf_trend": l2_htf_score, "technical_confluence": l3_tech_score,
                    "smc_confluence": 0, "volume_flow": 0,
                    "regime": 0, "ml_confidence": 0, "liquidity_depth": 0,
                },
                "rejected_at": "L3_Technical",
            }
            return

        # ── L4: SMC CONFLUENCE (HARD GATE: ≥ 40 points) ─────────────────
        # Spec: FVG retrace (20pts), OB retrace (20pts), Breaker (15pts),
        #        Liq sweep+MSS (25pts), London/NY overlap (20pts)
        # Relaxation: proximity scoring (10pts) + structural presence (5pts)
        l4_smc_points = 0.0

        # FVG in 50-62% retrace zone (20pts) or proximity (10pts)
        fvg_scored = False
        for fvg in smc_state.active_fvgs:
            fvg_high = getattr(fvg, "top", 0)
            fvg_low = getattr(fvg, "bottom", 0)
            gap_range = fvg_high - fvg_low
            if gap_range > 0:
                if getattr(fvg, "direction", "") == "bullish":
                    fib_50 = fvg_high - 0.50 * gap_range
                    fib_62 = fvg_high - 0.62 * gap_range
                    if fib_62 <= close_price <= fib_50:
                        l4_smc_points += 20.0
                        fvg_scored = True
                        break
                    # Proximity: price within the full FVG range
                    if fvg_low <= close_price <= fvg_high:
                        l4_smc_points += 10.0
                        fvg_scored = True
                        break
                else:
                    fib_50 = fvg_low + 0.50 * gap_range
                    fib_62 = fvg_low + 0.62 * gap_range
                    if fib_50 <= close_price <= fib_62:
                        l4_smc_points += 20.0
                        fvg_scored = True
                        break
                    if fvg_low <= close_price <= fvg_high:
                        l4_smc_points += 10.0
                        fvg_scored = True
                        break
        # Structural presence: active FVGs exist
        if not fvg_scored and smc_state.active_fvgs:
            l4_smc_points += 10.0

        # OB in 38-62% retrace zone (20pts) or price within OB (10pts)
        ob_scored = False
        for ob in smc_state.active_order_blocks:
            ob_high = getattr(ob, "high", 0)
            ob_low = getattr(ob, "low", 0)
            ob_range = ob_high - ob_low
            if ob_range > 0:
                fib_38 = ob_high - 0.38 * ob_range
                fib_62 = ob_high - 0.62 * ob_range
                if fib_62 <= close_price <= fib_38:
                    l4_smc_points += 20.0
                    ob_scored = True
                    break
                # Proximity: price within order block range
                if ob_low <= close_price <= ob_high:
                    l4_smc_points += 10.0
                    ob_scored = True
                    break
        # Structural presence: active OBs exist
        if not ob_scored and smc_state.active_order_blocks:
            l4_smc_points += 10.0

        # Breaker Block rejection
        if smc_state.active_breaker_blocks:
            l4_smc_points += 15.0

        # Liquidity sweep + MSS
        if smc_state.liquidity_grabs:
            # Check for MSS (CHoCH event)
            has_mss = len(smc_state.choch_events) > 0
            if has_mss:
                l4_smc_points += 25.0
            else:
                l4_smc_points += 10.0  # Sweep without MSS: partial credit

        # London/NY overlap (UTC 13-16)
        utc_hour = datetime.datetime.utcnow().hour
        if 13 <= utc_hour <= 16:
            l4_smc_points += 20.0

        # BOS/CHoCH confluence bonus
        if smc_state.bos_events:
            l4_smc_points += 10.0
        if smc_state.choch_events:
            l4_smc_points += 10.0

        l4_smc_score = int(min(100, l4_smc_points))

        smc_threshold = float(self._get_smc_threshold())
        if l4_smc_points < smc_threshold:
            logger.info("DIAG {}: L4 SMC LOW: points={:.0f} < {:.0f} "
                        "(FVGs={} OBs={} breakers={} grabs={})",
                        key, l4_smc_points, smc_threshold,
                        len(smc_state.active_fvgs), len(smc_state.active_order_blocks),
                        len(smc_state.active_breaker_blocks), len(smc_state.liquidity_grabs))
            if self._is_paper_mode():
                self._last_layer_status["smart_money_concepts"] = "WARN"
                self._last_layer_status["smart_money_concepts_detail"] = f"points={l4_smc_points:.0f}<{smc_threshold:.0f}"
            else:
                self._update_layer_status(
                    l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=l4_smc_score,
                    l4=0, l5=0, l6=0, l7=0,
                )
                self._last_layer_status["smart_money_concepts"] = "FAIL"
                self._last_layer_status["smart_money_concepts_detail"] = f"points={l4_smc_points:.0f}<{smc_threshold:.0f}"
                self._last_quality_breakdown = {
                    "total": 0,
                    "components": {
                        "htf_trend": l2_htf_score, "technical_confluence": l3_tech_score,
                        "smc_confluence": l4_smc_score, "volume_flow": 0,
                        "regime": 0, "ml_confidence": 0, "liquidity_depth": 0,
                    },
                    "rejected_at": "L4_SMC",
                }
                return

        # ── L5: VOLUME FLOW (HARD GATE: ≥ 60/100) ───────────────────────
        # Spec: VPFR LVN (25), Volume Delta (25), VWAP (20), CMF (15), OBV (15)
        l5_vol_points = 0.0

        # VPFR LVN: entry price in low-volume node
        for lvn in vol_flow.lvn_levels:
            if abs(close_price - lvn) / close_price < 0.005:
                l5_vol_points += 25.0
                break

        # Volume Delta positive and rising
        if vol_flow.delta_trend == "accumulating" and proposed_direction == "long":
            l5_vol_points += 25.0
        elif vol_flow.delta_trend == "distributing" and proposed_direction == "short":
            l5_vol_points += 25.0

        # VWAP alignment: price > VWAP AND slope rising
        vwap_val = float(last_row.get("vwap", 0))
        if vwap_val > 0:
            price_above_vwap = close_price > vwap_val if proposed_direction == "long" else close_price < vwap_val
            slope_ok = vol_flow.vwap_slope == "rising" if proposed_direction == "long" else vol_flow.vwap_slope == "falling"
            if price_above_vwap and slope_ok:
                l5_vol_points += 20.0
            elif price_above_vwap or slope_ok:
                l5_vol_points += 10.0  # Partial credit

        # CMF > +0.05 (accumulation)
        cmf = vol_flow.cmf_score
        if (proposed_direction == "long" and cmf > 0.05) or (proposed_direction == "short" and cmf < -0.05):
            l5_vol_points += 15.0

        # OBV rising (current > 20-period average)
        if vol_flow.obv_divergence == "bullish" and proposed_direction == "long":
            l5_vol_points += 15.0
        elif vol_flow.obv_divergence == "bearish" and proposed_direction == "short":
            l5_vol_points += 15.0

        l5_vol_score = int(min(100, l5_vol_points))

        volume_threshold = float(self._get_volume_threshold())
        if l5_vol_points < volume_threshold:
            logger.info("DIAG {}: L5 Volume HARD GATE FAIL: points={:.0f} < {:.0f} "
                        "(delta={} vwap={} cmf={:.3f} obv={} lvn={})",
                        key, l5_vol_points, volume_threshold, vol_flow.delta_trend,
                        vol_flow.vwap_slope, cmf, vol_flow.obv_divergence,
                        len(vol_flow.lvn_levels))
            self._update_layer_status(
                l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=l4_smc_score,
                l4=l5_vol_score, l5=0, l6=0, l7=0,
            )
            self._last_layer_status["volume_flow"] = "FAIL"
            self._last_layer_status["volume_flow_detail"] = f"points={l5_vol_points:.0f}<{volume_threshold:.0f}"
            self._last_quality_breakdown = {
                "total": 0,
                "components": {
                    "htf_trend": l2_htf_score, "technical_confluence": l3_tech_score,
                    "smc_confluence": l4_smc_score, "volume_flow": l5_vol_score,
                    "regime": 0, "ml_confidence": 0, "liquidity_depth": 0,
                },
                "rejected_at": "L5_Volume",
            }
            return

        # ── L6: REGIME DETECTION (HARD GATE) ────────────────────────────
        # Spec: ADX>30 + ATR ratio>1.2 + BB width>0.02 = TRENDING (A,C,D)
        #        ADX 25-30 + ATR 1.0-1.2 + BB 0.015-0.02 = MODERATE (C,D only)
        #        Otherwise = CHOPPY → NO TRADE
        atr_price_ratio = atr_14 / close_price if close_price > 0 else 0
        # Use 20-bar ATR rolling mean for ratio
        atr_20_avg = float(df["atr_14"].tail(20).mean()) if "atr_14" in df.columns and len(df) >= 20 else atr_14
        atr_expansion = atr_14 / atr_20_avg if atr_20_avg > 0 else 1.0

        if adx_val > 30 and atr_expansion > 1.2 and bb_width > 0.02:
            regime_class = "TRENDING"
            regime_size = 1.0
            regime_allowed_types = {SignalType.TYPE_A, SignalType.TYPE_C, SignalType.TYPE_D}
            l6_regime_score = 100
        elif 25 <= adx_val <= 30 and 1.0 <= atr_expansion <= 1.2 and 0.015 <= bb_width <= 0.02:
            regime_class = "MODERATE"
            regime_size = 0.75
            regime_allowed_types = {SignalType.TYPE_C, SignalType.TYPE_D}  # Type A blocked
            l6_regime_score = 70
        else:
            regime_class = "CHOPPY"
            regime_size = 0.0
            regime_allowed_types = set()
            l6_regime_score = 0

        # Signal type classification (needed for regime gating)
        signal_type = self._classify_signal_type(smc_state, df)

        # HARD GATE: Choppy regime blocks all signals
        if regime_class == "CHOPPY":
            # In live mode, always block choppy setups. Only paper mode may degrade gracefully.
            if (not self._is_paper_mode()) or not (regime_state and regime_state.tradeable):
                logger.info("DIAG {}: L6 Regime HARD GATE FAIL: CHOPPY "
                            "(ADX={:.1f} ATR_exp={:.2f} BB={:.4f})",
                            key, adx_val, atr_expansion, bb_width)
                self._update_layer_status(
                    l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=l4_smc_score,
                    l4=l5_vol_score, l5=l6_regime_score, l6=0, l7=0,
                )
                self._last_layer_status["regime_detection"] = "BLOCKED"
                self._last_quality_breakdown = {
                    "total": 0,
                    "components": {
                        "htf_trend": l2_htf_score, "technical_confluence": l3_tech_score,
                        "smc_confluence": l4_smc_score, "volume_flow": l5_vol_score,
                        "regime": 0, "ml_confidence": 0, "liquidity_depth": 0,
                    },
                    "rejected_at": "L6_Regime_CHOPPY",
                }
                return
            else:
                # Existing regime says tradeable — use its assessment but reduce score
                l6_regime_score = 40
                regime_size = max(regime_state.position_size_pct, 0.25) if regime_state else 0.5
                regime_allowed_types = {SignalType.TYPE_A, SignalType.TYPE_C, SignalType.TYPE_D, SignalType.TYPE_B, SignalType.COMPOSITE}

        # Regime signal type gating: MODERATE blocks Type A
        if signal_type not in regime_allowed_types and regime_allowed_types:
            logger.info("DIAG {}: L6 Signal type {} not allowed in {} regime (allowed: {})",
                        key, signal_type.value, regime_class,
                        [t.value for t in regime_allowed_types])
            self._update_layer_status(
                l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=l4_smc_score,
                l4=l5_vol_score, l5=l6_regime_score, l6=0, l7=0,
            )
            self._last_layer_status["regime_detection"] = "FAIL"
            self._last_layer_status["regime_detection_detail"] = f"{signal_type.value} blocked in {regime_class}"
            self._last_quality_breakdown = {
                "total": 0,
                "components": {
                    "htf_trend": l2_htf_score, "technical_confluence": l3_tech_score,
                    "smc_confluence": l4_smc_score, "volume_flow": l5_vol_score,
                    "regime": l6_regime_score, "ml_confidence": 0, "liquidity_depth": 0,
                },
                "rejected_at": f"L6_Regime_type_{signal_type.value}",
            }
            return

        # ── L7: ML ENSEMBLE ──────────────────────────────────────────────
        # Spec: ML confidence 0-100 (or 0 if drift/unavailable)
        ml_confidence_100 = ml_norm  # Already 0-100 from earlier
        l7_ml_score = int(max(0, min(100, ml_confidence_100)))
        # No hard gate on ML — it just feeds into quality score

        # ── L8: SIGNAL QUALITY (HARD GATE: ≥ 65) ────────────────────────
        kz_score = 0.0
        if self._session_filter is not None:
            try:
                kz_score = float(self._session_filter.killzone_score())
            except Exception:
                kz_score = 0.0
        quality_score = self._compute_quality_score(
            htf_score=htf_weighted_score, signal_type=signal_type,
            vol_ratio=vol_ratio, smc_state=smc_state, vol_flow=vol_flow,
            session_rule=session_rule, sentiment=sentiment,
            regime_state=regime_state, direction=proposed_direction,
            tech_score_100=float(l3_tech_score),
            smc_points=float(l4_smc_score),
            volume_score_100=float(l5_vol_score),
            regime_allows=(regime_class != "CHOPPY"),
            ml_confidence=float(l7_ml_score),
            orderbook_depth_ratio=orderbook_depth_ratio,
            killzone_score=kz_score,
        )

        # ── SMC Strategy Selector (ICT modules) ───────────────────────────
        if self._smc_selector is not None:
            try:
                smc_sig = self._smc_selector.select_and_evaluate(df, regime_state)
                if smc_sig is not None:
                    self._last_smc_signal_per_symbol[candle.symbol] = {
                        "type": getattr(smc_sig, "signal_type", smc_sig.__class__.__name__),
                        "direction": getattr(smc_sig, "direction", ""),
                        "score": float(getattr(smc_sig, "score", 0.0)),
                        "entry": float(getattr(smc_sig, "entry_price", 0.0) or 0.0),
                        "stop_loss": float(getattr(smc_sig, "stop_loss", 0.0) or 0.0),
                        "take_profit": float(getattr(smc_sig, "take_profit", 0.0) or 0.0),
                        "reason": getattr(smc_sig, "reason", ""),
                        "ts": time.time(),
                    }
            except Exception as exc:
                logger.debug("SMC selector error for {}: {}", candle.symbol, exc)

        # ── V6 MasterScorer composite (spec §11) ───────────────────────────
        if _MASTER_SCORER_AVAILABLE:
            try:
                exec_liq = compute_exec_liquidity_score(
                    orderbook_depth_usd=float(orderbook_depth_ratio) * 1000.0 if orderbook_depth_ratio else 0.0,
                    position_size_usd=1000.0,
                )
                drift = bool(getattr(self._ml_scorer, "drift_detected", False))
                master = compute_master_score(
                    layer3_score=float(l4_smc_score),
                    layer4_score=float(l3_tech_score),
                    layer5_score=float(l5_vol_score),
                    neural_confidence=float(l7_ml_score),
                    exec_liquidity_score=float(exec_liq),
                    drift_detected=drift,
                )
                self._last_master_score_per_symbol[candle.symbol] = {
                    "total": master.total_score,
                    "decision": master.decision,
                    "size_multiplier": master.size_multiplier,
                    "size_boost": master.size_boost,
                    "components": master.component_scores,
                    "explanation": master.explanation,
                    "ts": time.time(),
                }
            except Exception as exc:
                logger.debug("MasterScorer error for {}: {}", candle.symbol, exc)

        # Update all layer statuses for dashboard
        self._update_layer_status(
            l0=l1_session_score, l1=l2_htf_score, l2=l3_tech_score, l3=l4_smc_score,
            l4=l5_vol_score, l5=l6_regime_score, l6=l7_ml_score, l7=quality_score,
        )

        self._last_quality_breakdown = {
            "total": quality_score,
            "components": {
                "htf_trend": l2_htf_score,
                "technical_confluence": l3_tech_score,
                "smc_confluence": l4_smc_score,
                "volume_flow": l5_vol_score,
                "regime": l6_regime_score,
                "ml_confidence": l7_ml_score,
                "liquidity_depth": liquidity_depth_score,
            },
            "session": session_rule.name if session_rule else "none",
            "regime_class": regime_class,
            "signal_type": signal_type.value,
        }

        # HARD GATE: Quality score >= 65 (spec), relaxed to 30 for paper mode
        quality_threshold = self._get_quality_threshold()
        if quality_score < quality_threshold:
            logger.info(
                "DIAG {}: L8 Quality HARD GATE FAIL: score={} < {} | "
                "htf={} tech={} smc={} vol={} regime={} ml={} | "
                "type={} regime_class={}",
                key, quality_score, quality_threshold, l2_htf_score, l3_tech_score,
                l4_smc_score, l5_vol_score, l6_regime_score, l7_ml_score,
                signal_type.value, regime_class,
            )
            self._last_layer_status["signal_quality"] = "FAIL"
            self._last_layer_status["signal_quality_detail"] = f"quality={quality_score}<{quality_threshold}"
            self._last_layer_status["risk_gate"] = "FAIL"
            self._last_layer_status["risk_gate_detail"] = f"quality={quality_score}<{quality_threshold}"
            return

        # Funding divergence: ±5 quality points based on crowded/squeeze bias
        fund_div = self._funding_divergence_score(candle.symbol, df)
        prop_sign = 1.0 if proposed_direction == "long" else -1.0
        funding_bonus = int(round(fund_div * prop_sign * 5))  # favor when funding aligned w/ direction
        quality_score = max(0, min(100, int(quality_score) + funding_bonus))

        # Quality ≥ 90 → +25% size boost (spec)
        quality_size_boost = 1.25 if quality_score >= 90 else 1.0

        # ── L9: RISK GATE (handled by RiskManager on SIGNAL event) ───────
        # Composite score for backward compat
        master_score = float(quality_score)
        direction_sign = 1.0 if proposed_direction == "long" else -1.0
        composite = direction_sign * (master_score / 100.0)
        abs_score = abs(composite)

        # MTF position sizing
        mtf_size_mult = 1.0
        if htf_agreement_count >= 3:
            mtf_size_mult = 1.0
        elif htf_agreement_count == 2:
            mtf_size_mult = 0.75
        else:
            mtf_size_mult = 0.50

        # Regime position sizing
        regime_size_pct = regime_size if regime_class != "CHOPPY" else (regime_state.position_size_pct if regime_state else 1.0)

        # ── Session signal type filtering (§5) ────────────────────────────
        if session_rule and session_rule.allowed_types is not None:
            if signal_type not in session_rule.allowed_types:
                logger.debug("DIAG {}: signal_type {} not allowed in session {}",
                             key, signal_type.value, session_rule.name)
                return

        # ── Correlation filter ────────────────────────────────────────────
        if self._check_correlation_block(candle.symbol, proposed_direction):
            return

        # ── Auto-trading gate ─────────────────────────────────────────────
        if not self._auto_trading_enabled:
            logger.debug(
                "Signal suppressed for {} (auto_trading=off) — composite={:.3f}",
                key, composite,
            )
            return

        # ── Per-symbol duplicate guard ─────────────────────────────────────
        if candle.symbol in self._open_directions:
            logger.debug(
                "Signal suppressed for {}: already have {} position on {}",
                key, self._open_directions[candle.symbol], candle.symbol,
            )
            return

        # ── Max positions gate ────────────────────────────────────────────
        max_pos = int(self._bot_config.get("max_positions", 3))
        open_signal_count = len(self._open_directions)
        if self._risk_manager is not None and hasattr(self._risk_manager, 'positions'):
            rm_count = len(self._risk_manager.positions)
            open_signal_count = max(open_signal_count, rm_count)
        if open_signal_count >= max_pos:
            logger.debug(
                "Signal suppressed for {}: max_positions={} reached",
                key, max_pos,
            )
            return

        # ═══════════════════════════════════════════════════════════════════
        # POSITION MANAGEMENT: ATR-based SL + Tiered TP (1:1 / zone / trail)
        # ═══════════════════════════════════════════════════════════════════
        atr = atr_14
        price = close_price
        if atr <= 0:
            atr = price * 0.01

        risk_cfg = self.config.get_value("risk") or {}
        sl_atr_mult = float(risk_cfg.get("sl_atr_multiplier", 2.0))

        # Regime-adaptive SL
        if current_regime in (MarketRegime.RANGE_CHOP,):
            sl_atr_mult = min(sl_atr_mult, 1.5)
        elif current_regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            sl_atr_mult = max(sl_atr_mult, 2.5)

        # §7 Regime transition: tighten SL by 0.5x
        if in_transition:
            sl_atr_mult *= _TRANSITION_SL_MULT

        sl_dist = atr * sl_atr_mult

        # TP1: 1:1 R:R (50% position) — per spec
        tp1_dist = sl_dist * 1.0

        # TP2: Next liquidity zone (30% position) — SMC-driven
        liq_zone = self._smc_analyzer.get_nearest_liquidity_zone(
            df, proposed_direction, price,
        )
        if liq_zone is not None:
            tp2_dist = abs(liq_zone - price)
            tp2_dist = max(tp2_dist, tp1_dist * 1.5)
        else:
            tp2_dist = sl_dist * 2.5  # Fallback: 2.5R

        # TP3: SuperTrend trailing (20% position)
        supertrend = float(last.get("supertrend", 0))
        tp3_trail_level = supertrend if supertrend > 0 else 0

        # Primary TP = TP1 for the signal (TP2/TP3 managed by risk engine)
        direction = proposed_direction
        if direction == "long":
            sl = price - sl_dist
            tp = price + tp1_dist
        else:
            sl = price + sl_dist
            tp = price - tp1_dist

        # Override with bot config if auto_sl_tp is enabled
        if self._bot_config.get("auto_sl_tp"):
            bot_risk = float(self._bot_config.get("risk_per_trade", 2)) / 100.0
            sl_dist = min(sl_dist, price * bot_risk) if bot_risk > 0 else sl_dist

        # ═══════════════════════════════════════════════════════════════════
        # BUILD SIGNAL
        # ═══════════════════════════════════════════════════════════════════
        _LONG_REASONS = {"RSI_oversold", "RSI_bullish_zone", "RSI_low", "MACD_bullish_cross",
                         "EMA_stack_bullish", "EMA_golden_cross", "BB_pct_bullish",
                         "SuperTrend_bullish", "Ichimoku_above_cloud", "PSAR_bullish",
                         "Aroon_bullish", "MFI_bullish", "Stoch_bullish_cross", "Stoch_oversold",
                         "CCI_bullish", "WR_bullish", "UltOsc_rising", "TRIX_bullish_cross",
                         "KC_breakout_bullish", "ADX_strong_", "vol_confirm_",
                         "BOS_bullish", "CHoCH_bullish", "FVG_bullish_zone",
                         "OB_bullish_support", "liquidity_grab_bullish",
                         "OBV_bullish_div", "CMF_accumulation", "VWAP_rising",
                         "delta_accumulating", "above_value_area"}
        _SHORT_REASONS = {"RSI_overbought", "RSI_high", "MACD_bearish_cross",
                          "EMA_stack_bearish", "EMA_death_cross", "BB_pct_bearish",
                          "SuperTrend_bearish", "Ichimoku_below_cloud", "PSAR_bearish",
                          "Aroon_bearish", "MFI_bearish", "Stoch_bearish_cross", "Stoch_overbought",
                          "CCI_bearish", "WR_overbought", "UltOsc_falling", "TRIX_bearish_cross",
                          "KC_breakout_bearish", "ADX_strong_", "vol_confirm_",
                          "BOS_bearish", "CHoCH_bearish", "FVG_bearish_zone",
                          "OB_bearish_resistance", "liquidity_grab_bearish",
                          "OBV_bearish_div", "CMF_distribution", "VWAP_falling",
                          "delta_distributing", "below_value_area"}
        valid_set = _LONG_REASONS if direction == "long" else _SHORT_REASONS

        # Combine reasons from all layers
        all_raw_reasons = tech_reasons + smc_state.reasons + vol_flow.reasons
        reasons: list[str] = [
            r for r in all_raw_reasons
            if any(r.startswith(v) for v in valid_set)
        ]
        if strategy_signal is not None:
            reasons.extend(strategy_signal.reasons)
            reasons.append(f"strategy:{strategy_signal.strategy}")
        if ml_score * (1 if direction == "long" else -1) > 0 and abs(ml_score) > 0.1:
            reasons.append(f"ML_{'bullish' if ml_score > 0 else 'bearish'}")
        if sentiment * (1 if direction == "long" else -1) > 0 and abs(sentiment) > 0.2:
            reasons.append(f"sentiment_{'bullish' if sentiment > 0 else 'bearish'}")
        if current_regime:
            reasons.append(f"regime:{current_regime.value}")

        # ── Compute regime-specific risk % (§7) ──────────────────────────
        regime_risk_pct = _REGIME_RISK_PCT.get(current_regime, 0.01) if current_regime else 0.01

        # ── Session sizing multiplier (§5) ────────────────────────────────
        session_size_mult = session_rule.size_multiplier if session_rule else 1.0

        # ── Regime transition: reduce size by 50% (§7) ───────────────────
        transition_mult = _TRANSITION_SIZE_MULT if in_transition else 1.0

        # Factor count for metadata
        proposed_sign = 1.0 if composite > 0 else -1.0
        tmp_scores = {
            "technical": tech_score, "ml": ml_score, "sentiment": sentiment,
            "macro": macro, "news": news_score, "orderbook": ob_score,
            "smc": smc_score, "volume_flow": flow_score,
        }
        active_factors = sum(
            1 for v in tmp_scores.values()
            if abs(v) >= self._min_factor_magnitude and (v * proposed_sign) > 0
        )

        weights = self._get_regime_weights(current_regime)

        effective_size = max(0.10, session_size_mult * mtf_size_mult * transition_mult * quality_size_boost * max(regime_size, 0.25))

        signal = TradingSignal(
            exchange=candle.exchange,
            symbol=candle.symbol,
            direction=direction,
            score=abs_score,
            size_multiplier=effective_size,
            technical_score=tech_score,
            ml_score=ml_score,
            sentiment_score=sentiment,
            macro_score=macro,
            news_score=news_score,
            orderbook_score=ob_score,
            regime=regime_state.regime.value if regime_state else MarketRegime.UNKNOWN.value,
            regime_confidence=regime_state.confidence if regime_state else 0.0,
            price=price,
            atr=atr,
            stop_loss=sl,
            take_profit=tp,
            timestamp=int(now),
            reasons=reasons,
            signal_type=signal_type.value,
            quality_score=quality_score,
            session_name=session_rule.name if session_rule else "",
            session_size_mult=session_size_mult,
            regime_risk_pct=regime_risk_pct,
            mtf_agreement_count=htf_agreement_count,
            mtf_weighted_score=htf_weighted_score,
            metadata={
                "timeframe": self._primary_tf,
                "factors_active": active_factors,
                "strategy": strategy_signal.strategy if strategy_signal else "composite",
                "strategy_score": strategy_signal.score if strategy_signal else 0.0,
                "regime_weights": weights,
                "sl_atr_mult": sl_atr_mult,
                "risk_reward": tp1_dist / sl_dist if sl_dist > 0 else 0.0,
                "regime_size_pct": regime_size_pct,
                # SMC metadata
                "smc_score": smc_score,
                "smc_bias": smc_state.market_bias,
                "bos_count": len(smc_state.bos_events),
                "choch_count": len(smc_state.choch_events),
                "active_fvgs": len(smc_state.active_fvgs),
                "active_order_blocks": len(smc_state.active_order_blocks),
                # Volume flow metadata
                "volume_flow_score": flow_score,
                "poc_price": vol_flow.poc_price,
                "obv_divergence": vol_flow.obv_divergence,
                "cmf": vol_flow.cmf_score,
                "vwap_slope": vol_flow.vwap_slope,
                "delta_trend": vol_flow.delta_trend,
                # TP tiers
                "tp1_price": price + tp1_dist if direction == "long" else price - tp1_dist,
                "tp1_close_pct": 0.50,
                "tp2_price": price + tp2_dist if direction == "long" else price - tp2_dist,
                "tp2_close_pct": 0.30,
                "tp3_trail": True,
                "tp3_close_pct": 0.20,
                "supertrend_trail": tp3_trail_level,
                "liquidity_zone_tp": liq_zone,
                # §5/§6/§7 additions
                "session_name": session_rule.name if session_rule else "",
                "session_size_mult": session_size_mult,
                "mtf_size_mult": mtf_size_mult,
                "quality_score": quality_score,
                "quality_size_boost": quality_size_boost,
                "signal_type": signal_type.value,
                "regime_class": regime_class,
                "regime_risk_pct": regime_risk_pct,
                "in_transition": in_transition,
                "transition_mult": transition_mult,
                "effective_size_mult": effective_size,
                "is_killzone": is_killzone,
                # V6.0 spec-compliant layer scores
                "master_score": round(master_score, 1),
                "l1_session": l1_session_score,
                "l2_htf_trend": l2_htf_score,
                "l3_technical": l3_tech_score,
                "l4_smc": l4_smc_score,
                "l5_volume": l5_vol_score,
                "l6_regime": l6_regime_score,
                "l7_ml": l7_ml_score,
                "l8_quality": quality_score,
            },
        )

        agent_decision = self._ai_agent.review_signal(signal) if self._ai_agent is not None else None
        if agent_decision is not None:
            signal.metadata["ai_agent"] = agent_decision.to_dict()
            if not agent_decision.approved:
                self._last_layer_status["risk_gate"] = "FAIL"
                self._last_layer_status["risk_gate_detail"] = (
                    f"AI rejected {candle.symbol}: {agent_decision.reason} conf={agent_decision.confidence:.2f}"
                )
                logger.info(
                    "AI_AGENT rejected {}/{} {} conf={:.2f} reason={}",
                    candle.exchange,
                    candle.symbol,
                    direction.upper(),
                    agent_decision.confidence,
                    agent_decision.reason,
                )
                return
            if agent_decision.size_multiplier and agent_decision.size_multiplier != 1.0:
                signal.size_multiplier = max(0.10, signal.size_multiplier * agent_decision.size_multiplier)
                signal.metadata["effective_size_mult"] = signal.size_multiplier
                signal.metadata["ai_adjusted_size_mult"] = agent_decision.size_multiplier

        self._last_signal_time[key] = now
        self._open_directions[candle.symbol] = direction
        # Update risk gate status (signal passed all gates)
        self._last_layer_status["risk_gate"] = "PASS"
        self._last_layer_status["risk_gate_detail"] = f"Q={quality_score} type={signal_type.value} regime={regime_class}"
        await self.event_bus.publish("SIGNAL", signal)
        logger.info(
            "SIGNAL: {}/{} {} type={} Q={}/100 price={:.2f} sl={:.2f} "
            "tp1={:.2f} tp2={:.2f} R:R=1:{:.1f} regime_class={} risk={:.1%} session={} "
            "L2={} L3={} L4={} L5={} L6={} L7={} "
            "size_mult={:.2f}",
            signal.exchange, signal.symbol, signal.direction.upper(),
            signal_type.value, quality_score,
            signal.price, signal.stop_loss,
            signal.metadata["tp1_price"], signal.metadata["tp2_price"],
            tp2_dist / sl_dist if sl_dist > 0 else 0.0,
            regime_class, regime_risk_pct,
            session_rule.name if session_rule else "none",
            l2_htf_score, l3_tech_score, l4_smc_score, l5_vol_score,
            l6_regime_score, l7_ml_score,
            signal.metadata["effective_size_mult"],
        )

    async def _handle_sentiment(self, payload: Any) -> None:
        self._sentiment_score = float(payload.score) if hasattr(payload, "score") else 0.0
        self._sentiment_ts = time.time()

    async def _handle_funding(self, payload: Any) -> None:
        if not isinstance(payload, list) or not payload:
            return
        # Preserve sign: positive funding = longs pay (bearish), negative = shorts pay (bullish)
        rates = [r.rate for r in payload]
        avg = sum(rates) / len(rates) if rates else 0.0
        # Negative avg → bullish bias (positive macro_score), positive avg → bearish (negative)
        self._macro_score = float(max(min(-avg * 100, 1.0), -1.0))
        self._macro_ts = time.time()
        # Store per-symbol funding for divergence feature
        for r in payload:
            sym = getattr(r, "symbol", None)
            rate = float(getattr(r, "rate", 0.0) or 0.0)
            if sym:
                self._funding_by_symbol[sym] = (rate, time.time())

    def _funding_divergence_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Return [-1, 1] signed score based on funding vs. recent return divergence.

        Crowded long (pos funding + rising price) → short bias (negative).
        Crowded short (neg funding + falling price) → long bias (positive).
        Divergence (funding opposite to price) → squeeze bias (same dir as price).
        """
        fr = self._funding_by_symbol.get(symbol)
        if not fr or df is None or len(df) < 20:
            return 0.0
        rate, ts = fr
        if time.time() - ts > 7200:  # stale beyond 2h
            return 0.0
        try:
            ret = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1.0)
        except Exception:
            return 0.0
        # funding in raw units (e.g., 0.0001 = 1bp). Scale: ±0.05% is notable.
        f_norm = float(np.clip(rate / 0.0005, -1.0, 1.0))
        r_norm = float(np.clip(ret / 0.02, -1.0, 1.0))  # 2% = full
        # Crowded: same sign. Score opposite to crowd.
        crowded = f_norm * r_norm  # positive when crowded
        if crowded > 0:
            return float(np.clip(-f_norm, -1.0, 1.0))  # fade crowded side
        # Divergence: signs oppose. Favor price direction (squeeze momentum).
        return float(np.clip(r_norm, -1.0, 1.0)) * 0.5

    async def _handle_news_sentiment(self, payload: Any) -> None:
        """Consume NEWS_SENTIMENT events — {sentiment: float, timestamp: int}."""
        if isinstance(payload, dict):
            sent = float(payload.get("sentiment", 0.0))
            ts = float(payload.get("timestamp", time.time()))
        elif hasattr(payload, "sentiment"):
            sent = float(payload.sentiment)
            ts = float(getattr(payload, "timestamp", time.time()))
        else:
            return
        self._news_scorer.ingest(sent, ts)

    async def _handle_orderbook_update(self, payload: Any) -> None:
        """Consume ORDERBOOK_UPDATE events — {bids: [(p,q),...], asks: [(p,q),...]}."""
        symbol = None
        if isinstance(payload, dict):
            symbol = payload.get("symbol")
            bids = [(float(b[0]), float(b[1])) for b in payload.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in payload.get("asks", [])]
        elif hasattr(payload, "bids") and hasattr(payload, "asks"):
            symbol = getattr(payload, "symbol", None)
            bids = [(float(b[0]), float(b[1])) for b in payload.bids]
            asks = [(float(a[0]), float(a[1])) for a in payload.asks]
        else:
            return
        self._orderbook_scorer.update(
            bids,
            asks,
            symbol=symbol,
            min_depth_usd=self._min_orderbook_depth_usd,
        )

    async def run(self) -> None:
        self._running = True
        self.event_bus.subscribe("CANDLE", self._handle_candle)
        self.event_bus.subscribe("SENTIMENT", self._handle_sentiment)
        self.event_bus.subscribe("FUNDING_RATE", self._handle_funding)
        self.event_bus.subscribe("NEWS_SENTIMENT", self._handle_news_sentiment)
        self.event_bus.subscribe("ORDERBOOK_UPDATE", self._handle_orderbook_update)
        if self._ml_bootstrap_task is None or self._ml_bootstrap_task.done():
            self._ml_bootstrap_task = asyncio.create_task(self._bootstrap_ml_model_loop())
        logger.info(
            "SignalGenerator started (primary_tf={}, auto_trading={}, "
            "weights: tech={:.0%} ml={:.0%} sent={:.0%} macro={:.0%} news={:.0%} ob={:.0%}, "
            "min_factors={})",
            self._primary_tf, self._auto_trading_enabled,
            self._tech_weight, self._ml_weight, self._sentiment_weight,
            self._macro_weight, self._news_weight, self._orderbook_weight,
            self._min_factors,
        )
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._ml_bootstrap_task and not self._ml_bootstrap_task.done():
            self._ml_bootstrap_task.cancel()
        self.event_bus.unsubscribe("CANDLE", self._handle_candle)
        self.event_bus.unsubscribe("SENTIMENT", self._handle_sentiment)
        self.event_bus.unsubscribe("FUNDING_RATE", self._handle_funding)
        self.event_bus.unsubscribe("NEWS_SENTIMENT", self._handle_news_sentiment)
        self.event_bus.unsubscribe("ORDERBOOK_UPDATE", self._handle_orderbook_update)
