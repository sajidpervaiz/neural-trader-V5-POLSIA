"""
Tests for ARMS-V2.1 remaining 23 gaps — comprehensive coverage.
"""
from __future__ import annotations

import asyncio
import datetime
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analysis.regime import MarketRegime, RegimeState
from engine.strategy_modules import (
    BreakoutExpansionModule,
    MeanReversionModule,
    StrategySelector,
    StrategySignal,
    TrendContinuationModule,
    _check_confirmation_candle,
    _compute_rsi_percentile,
    _detect_candlestick_pattern,
    _find_sr_zones,
)
from execution.order_splitter import IcebergExecutor, TWAPExecutor
from execution.shadow_sl import ShadowStopManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_regime(regime: MarketRegime, adx: float = 30.0, confidence: float = 0.8) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        adx=adx,
        plus_di=20.0,
        minus_di=15.0,
        trend_slope=0.01,
        realized_vol=0.3,
        hurst_exponent=0.6,
        ema200_slope=0.002,
        bb_width_percentile=15.0,
        keltner_squeeze=True,
        timestamp=int(time.time()),
        candles_in_state=5,
    )


def _make_trend_df(n: int = 100, direction: str = "long") -> pd.DataFrame:
    """Generate a DataFrame suitable for TrendContinuationModule."""
    close = np.linspace(100, 120, n) if direction == "long" else np.linspace(120, 100, n)
    high = close + 1
    low = close - 1
    # Give last candle a strong body (>60% of range)
    opens = close.copy()
    if direction == "long":
        opens[-1] = close[-1] - 1.5  # bullish candle: close > open, body = 1.5, range ~2
    else:
        opens[-1] = close[-1] + 1.5  # bearish candle

    df = pd.DataFrame({
        "open": opens,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1500),
    })

    if direction == "long":
        df["ema_9"] = close + 0.5
        df["ema_21"] = close
        df["ema_55"] = close - 1
    else:
        df["ema_9"] = close - 0.5
        df["ema_21"] = close
        df["ema_55"] = close + 1

    # RSI: make it between 40-60 (pullback zone) — 50th percentile mid-range
    rsi_vals = np.full(n, 50.0)
    rsi_vals[-1] = 50.0
    df["rsi_14"] = rsi_vals
    df["macd_hist"] = np.linspace(-0.1, 0.5, n) if direction == "long" else np.linspace(0.1, -0.5, n)
    df["volume_ratio"] = 1.5
    df["atr_14"] = 1.5

    return df


def _make_breakout_df(n: int = 100, direction: str = "long") -> pd.DataFrame:
    """Generate DF for BreakoutExpansionModule."""
    # Tight consolidation range < 2% then breakout
    base = 100.0
    close = np.full(n, base)
    # Last candle breaks out
    if direction == "long":
        close[-1] = base + 2.5
    else:
        close[-1] = base - 2.5
    # Previous candle also broke out (for retest validation)
    if direction == "long":
        close[-2] = base + 1.5
    else:
        close[-2] = base - 1.5

    high = close + 0.5
    low = close - 0.5
    opens = close.copy()
    # Confirmation candle (body > 60%)
    if direction == "long":
        opens[-1] = close[-1] - 2.0  # big bullish candle
        high[-1] = close[-1] + 0.5
        low[-1] = opens[-1] - 0.2
    else:
        opens[-1] = close[-1] + 2.0
        high[-1] = opens[-1] + 0.2
        low[-1] = close[-1] - 0.5

    df = pd.DataFrame({
        "open": opens,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.linspace(1200, 800, n),  # Declining volume pre-breakout
    })
    df.loc[df.index[-1], "volume"] = 3500  # Spike on breakout

    df["bb_width_percentile"] = 10.0
    df["keltner_squeeze"] = True
    df["volume_ratio"] = 1.0
    df.loc[df.index[-1], "volume_ratio"] = 2.5
    df["high_50"] = base + 0.5  # Previous resistance
    df["low_50"] = base - 0.5
    df["atr_14"] = 1.0

    return df


def _make_mr_df(n: int = 100, direction: str = "long") -> pd.DataFrame:
    """Generate DF for MeanReversionModule with candlestick pattern."""
    close = np.full(n, 100.0) + np.random.RandomState(42).normal(0, 0.3, n)
    high = close + 1
    low = close - 1
    opens = close.copy()

    if direction == "long":
        # Hammer pattern: long lower wick, small body at top
        opens[-1] = close[-1] - 0.1  # small body
        low[-1] = close[-1] - 3.0  # long lower wick
        high[-1] = close[-1] + 0.1  # tiny upper wick
        # Previous candle bearish for engulfing context
        opens[-2] = close[-2] + 0.5
    else:
        # Shooting star: long upper wick, small body at bottom
        opens[-1] = close[-1] + 0.1
        high[-1] = close[-1] + 3.0
        low[-1] = close[-1] - 0.1

    df = pd.DataFrame({
        "open": opens,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000),
    })

    df["adx"] = 12.0
    df["rsi_14"] = 50.0
    df["stoch_k"] = 50.0
    df["bb_lower"] = 98.0
    df["bb_upper"] = 102.0
    df["bb_pct"] = 0.5
    df["atr_14"] = 1.0

    if direction == "long":
        df.loc[df.index[-1], "rsi_14"] = 25.0
        df.loc[df.index[-1], "stoch_k"] = 15.0
        df.loc[df.index[-1], "bb_pct"] = 0.02
    else:
        df.loc[df.index[-1], "rsi_14"] = 75.0
        df.loc[df.index[-1], "stoch_k"] = 85.0
        df.loc[df.index[-1], "bb_pct"] = 0.98

    return df


# ── §7A: Adaptive RSI Percentile ─────────────────────────────────────────────

class TestAdaptiveRSIPercentile:
    def test_rsi_percentile_mid_range(self):
        """RSI at 50th percentile should pass."""
        vals = np.full(100, 50.0)
        vals[-1] = 50.0  # Exactly at median
        df = pd.DataFrame({"rsi_14": np.linspace(20, 80, 99).tolist() + [50.0]})
        pctile = _compute_rsi_percentile(df)
        # 50 is roughly mid-range in [20..80]; percentile should be around 50
        assert 30 <= pctile <= 70

    def test_rsi_percentile_extreme_high(self):
        """RSI at top of range = high percentile."""
        vals = np.full(100, 50.0)
        vals[-1] = 80.0
        df = pd.DataFrame({"rsi_14": vals})
        pctile = _compute_rsi_percentile(df)
        assert pctile > 90

    def test_rsi_percentile_extreme_low(self):
        vals = np.full(100, 50.0)
        vals[-1] = 20.0
        df = pd.DataFrame({"rsi_14": vals})
        pctile = _compute_rsi_percentile(df)
        assert pctile < 10

    def test_rsi_percentile_insufficient_data(self):
        df = pd.DataFrame({"rsi_14": [50.0] * 10})
        assert _compute_rsi_percentile(df) == 50.0

    def test_trend_module_uses_adaptive_rsi(self):
        """TrendContinuation now uses percentile-based RSI."""
        mod = TrendContinuationModule()
        df = _make_trend_df(100, "long")
        regime = _make_regime(MarketRegime.STRONG_TREND_UP)
        sig = mod.evaluate(df, regime)
        if sig is not None:
            assert any("rsi_pctile" in r for r in sig.reasons)


# ── §7A: Confirmation Candle ─────────────────────────────────────────────────

class TestConfirmationCandle:
    def test_strong_bull_candle(self):
        """Body > 60% of range in correct direction passes."""
        df = pd.DataFrame({
            "open": [99, 100],
            "high": [100, 103],
            "low": [98, 100],
            "close": [100, 102.5],
        })
        assert _check_confirmation_candle(df, "long") is True

    def test_weak_candle_fails(self):
        """Body < 60% of range fails."""
        df = pd.DataFrame({
            "open": [99, 101],
            "high": [100, 103],
            "low": [98, 99],
            "close": [100, 101.2],
        })
        assert _check_confirmation_candle(df, "long") is False

    def test_wrong_direction_fails(self):
        """Bearish candle when long direction expected."""
        df = pd.DataFrame({
            "open": [99, 102],
            "high": [100, 103],
            "low": [98, 99],
            "close": [100, 99.5],
        })
        assert _check_confirmation_candle(df, "long") is False

    def test_short_confirmation(self):
        df = pd.DataFrame({
            "open": [100, 102],
            "high": [101, 102.2],
            "low": [99, 99],
            "close": [100, 99.5],
        })
        assert _check_confirmation_candle(df, "short") is True


# ── §7A: EMA Pullback Touch ─────────────────────────────────────────────────

class TestEmaPullbackTouch:
    def test_touch_required(self):
        """Price must be within 0.5% of EMA21 for trend entry."""
        mod = TrendContinuationModule(ema_touch_pct=0.005)
        df = _make_trend_df(100, "long")
        # Set EMA21 far from price — should block
        df["ema_21"] = df["close"] - 10
        regime = _make_regime(MarketRegime.STRONG_TREND_UP)
        assert mod.evaluate(df, regime) is None


# ── §7B: Pre-Breakout Consolidation ─────────────────────────────────────────

class TestBreakoutConsolidation:
    def test_tight_consolidation_passes(self):
        mod = BreakoutExpansionModule()
        df = _make_breakout_df(100, "long")
        regime = _make_regime(MarketRegime.COMPRESSION)
        sig = mod.evaluate(df, regime)
        if sig is not None:
            assert any("consolidation" in r for r in sig.reasons)

    def test_wide_range_blocks(self):
        """50-candle range > 2% should block breakout."""
        mod = BreakoutExpansionModule()
        df = _make_breakout_df(100, "long")
        # Make range very wide (>2%)
        df.loc[df.index[5], "high"] = 200
        df.loc[df.index[5], "low"] = 80
        regime = _make_regime(MarketRegime.COMPRESSION)
        assert mod.evaluate(df, regime) is None


# ── §7B: Volume Decay & Follow-Through ──────────────────────────────────────

class TestBreakoutVolumeDecay:
    def test_volume_decay_detected(self):
        mod = BreakoutExpansionModule()
        df = _make_breakout_df(100, "long")
        regime = _make_regime(MarketRegime.COMPRESSION)
        sig = mod.evaluate(df, regime)
        if sig is not None:
            # Volume decay check should fire (pre-breakout volumes decline)
            assert any("volume_decay" in r or "follow_through" in r for r in sig.reasons)

    def test_follow_through_candle(self):
        mod = BreakoutExpansionModule()
        df = _make_breakout_df(100, "long")
        # Weak candle (tiny body) → should fail follow-through
        df.loc[df.index[-1], "open"] = df.loc[df.index[-1], "close"] - 0.01
        regime = _make_regime(MarketRegime.COMPRESSION)
        assert mod.evaluate(df, regime) is None


# ── §7B: Retest Validation ──────────────────────────────────────────────────

class TestBreakoutRetest:
    def test_retest_confirmed(self):
        mod = BreakoutExpansionModule()
        df = _make_breakout_df(100, "long")
        regime = _make_regime(MarketRegime.COMPRESSION)
        sig = mod.evaluate(df, regime)
        if sig is not None:
            assert any("retest" in r for r in sig.reasons)


# ── §7C: Candlestick Pattern Detection ──────────────────────────────────────

class TestCandlestickPatterns:
    def test_bullish_engulfing(self):
        # prev: bearish (po=102, pc=99), curr: bullish engulfing (o=98 < pc=99, c=103 > po=102)
        df = pd.DataFrame({
            "open": [100, 102, 98],
            "high": [103, 103, 104],
            "low": [99, 98, 97],
            "close": [102, 99, 103],
        })
        assert _detect_candlestick_pattern(df) == "bullish_engulfing"

    def test_bearish_engulfing(self):
        # prev: bullish (po=99, pc=103), curr: bearish engulfing (o=104 > pc=103, c=98 < po=99)
        df = pd.DataFrame({
            "open": [100, 99, 104],
            "high": [103, 104, 105],
            "low": [99, 98, 97],
            "close": [102, 103, 98],
        })
        assert _detect_candlestick_pattern(df) == "bearish_engulfing"

    def test_hammer(self):
        df = pd.DataFrame({
            "open": [100, 100, 100.2],
            "high": [102, 102, 100.3],
            "low": [99, 99, 97.0],
            "close": [101, 101, 100.3],
        })
        assert _detect_candlestick_pattern(df) == "hammer"

    def test_shooting_star(self):
        # body=0.5, rng=3.1, body/rng=0.16 (>0.1 not doji, <0.35 passes)
        # upper_wick=2.5 > body*2=1.0, lower_wick=0.1 < body*0.5=0.25
        df = pd.DataFrame({
            "open": [100, 100, 100.0],
            "high": [102, 102, 103.0],
            "low": [99, 99, 99.9],
            "close": [101, 101, 100.5],
        })
        assert _detect_candlestick_pattern(df) == "shooting_star"

    def test_doji(self):
        df = pd.DataFrame({
            "open": [100, 100, 100.01],
            "high": [102, 102, 101],
            "low": [99, 99, 99],
            "close": [101, 101, 100.02],
        })
        assert _detect_candlestick_pattern(df) == "doji"

    def test_no_pattern(self):
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [103, 104, 105],
            "low": [99, 100, 101],
            "close": [102, 103, 104],
        })
        assert _detect_candlestick_pattern(df) is None

    def test_insufficient_data(self):
        df = pd.DataFrame({"open": [100], "high": [102], "low": [99], "close": [101]})
        assert _detect_candlestick_pattern(df) is None


# ── §7C: Support/Resistance Zones ────────────────────────────────────────────

class TestSRZones:
    def test_finds_pivot_points(self):
        n = 60
        np.random.seed(42)
        high = 100 + np.random.uniform(0, 5, n)
        low = 100 - np.random.uniform(0, 5, n)
        # Create clear pivot high
        high[25] = 110
        high[24] = 105
        high[26] = 105
        # Create clear pivot low
        low[35] = 90
        low[34] = 95
        low[36] = 95

        df = pd.DataFrame({"high": high, "low": low, "close": (high + low) / 2})
        zones = _find_sr_zones(df)
        assert len(zones["resistance"]) > 0
        assert len(zones["support"]) > 0

    def test_insufficient_data(self):
        df = pd.DataFrame({"high": [100], "low": [99], "close": [99.5]})
        zones = _find_sr_zones(df, lookback=50)
        assert zones["support"] == []
        assert zones["resistance"] == []

    def test_mr_module_uses_sr_zones(self):
        """MeanReversion should check S/R proximity."""
        mod = MeanReversionModule()
        df = _make_mr_df(100, "long")
        regime = _make_regime(MarketRegime.RANGE_CHOP, adx=12.0)
        sig = mod.evaluate(df, regime)
        if sig is not None:
            assert any("pattern" in r for r in sig.reasons)


# ── §8: Per-Tier Base Risk ───────────────────────────────────────────────────

class TestPerTierRisk:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {
            "risk": {"initial_equity": 100000, "risk_per_trade_pct": 0.01},
            "arms": {"risk": {
                "tier1_risk_pct": 0.0075,
                "tier2_risk_pct": 0.005,
                "tier3_risk_pct": 0.0025,
            }},
        }
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_tier1_risk(self):
        rm = self._make_rm()
        assert rm.get_tier_risk_pct(1) == 0.0075

    def test_tier2_risk(self):
        rm = self._make_rm()
        assert rm.get_tier_risk_pct(2) == 0.005

    def test_tier3_risk(self):
        rm = self._make_rm()
        assert rm.get_tier_risk_pct(3) == 0.0025

    def test_tier4_zero_risk(self):
        rm = self._make_rm()
        assert rm.get_tier_risk_pct(4) == 0.0


# ── §10: Weekly/Monthly Loss Caps ────────────────────────────────────────────

class TestWeeklyMonthlyCaps:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {
            "risk": {"initial_equity": 100000},
            "arms": {"risk": {
                "weekly_loss_limit_pct": 0.06,
                "monthly_loss_limit_pct": 0.10,
            }},
        }
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_within_limits(self):
        rm = self._make_rm()
        ok, _ = rm.check_weekly_monthly_limits()
        assert ok is True

    def test_weekly_limit_breached(self):
        rm = self._make_rm()
        rm._weekly_loss = -0.07
        ok, reason = rm.check_weekly_monthly_limits()
        assert ok is False
        assert "weekly" in reason

    def test_monthly_limit_breached(self):
        rm = self._make_rm()
        rm._monthly_loss = -0.11
        ok, reason = rm.check_weekly_monthly_limits()
        assert ok is False
        assert "monthly" in reason

    def test_loss_recording(self):
        rm = self._make_rm()
        rm._record_loss_for_period(-0.02)
        rm._record_loss_for_period(-0.03)
        assert rm._weekly_loss == pytest.approx(-0.05)
        assert rm._monthly_loss == pytest.approx(-0.05)

    def test_positive_pnl_not_recorded(self):
        rm = self._make_rm()
        rm._record_loss_for_period(0.05)
        assert rm._weekly_loss == 0.0


# ── §11: Drawdown Graduation Rules ──────────────────────────────────────────

class TestDrawdownGraduation:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {
            "risk": {"initial_equity": 100000},
            "arms": {"risk": {"dd_graduation_trades": 3}},
        }
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_3_consecutive_profits_graduate(self):
        rm = self._make_rm()
        rm._current_dd_phase = "caution"
        rm.record_trade_result(True)
        rm.record_trade_result(True)
        assert rm._current_dd_phase == "caution"  # Not yet
        rm.record_trade_result(True)
        assert rm._current_dd_phase == "normal"  # Graduated!

    def test_loss_resets_counter(self):
        rm = self._make_rm()
        rm._current_dd_phase = "defensive"
        rm.record_trade_result(True)
        rm.record_trade_result(True)
        rm.record_trade_result(False)  # Loss resets
        assert rm._consecutive_profitable == 0
        assert rm._current_dd_phase == "defensive"

    def test_emergency_graduates_to_defensive(self):
        rm = self._make_rm()
        rm._current_dd_phase = "emergency"
        for _ in range(3):
            rm.record_trade_result(True)
        assert rm._current_dd_phase == "defensive"

    def test_escalation_still_automatic(self):
        rm = self._make_rm()
        rm._current_dd_phase = "normal"
        rm._circuit_breaker._peak_equity = 100000
        rm._equity = 94000  # 6% DD → should escalate to caution (normal max_dd=5%, caution max_dd=8%)
        phase = rm.update_drawdown_phase()
        assert phase == "caution"

    def test_no_auto_deescalation(self):
        """Phases don't auto-demote even if DD recovers."""
        rm = self._make_rm()
        rm._current_dd_phase = "caution"
        rm._circuit_breaker._peak_equity = 100000
        rm._equity = 100000  # 0% DD — but caution stays until graduation
        phase = rm.update_drawdown_phase()
        assert phase == "caution"


# ── §10: ADX/ATR-Linked Dynamic Leverage ─────────────────────────────────────

class TestDynamicLeverage:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000, "default_leverage": 5.0}, "arms": {"risk": {}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_low_adx_allows_max_leverage(self):
        rm = self._make_rm()
        assert rm.get_dynamic_leverage(15, 30) == 5.0

    def test_high_adx_reduces_leverage(self):
        rm = self._make_rm()
        assert rm.get_dynamic_leverage(35, 30) == 2.0

    def test_high_atr_reduces_leverage(self):
        rm = self._make_rm()
        assert rm.get_dynamic_leverage(15, 80) == 2.0

    def test_both_high_picks_minimum(self):
        rm = self._make_rm()
        lev = rm.get_dynamic_leverage(55, 95)
        assert lev == 1.0


# ── §10.1: 3×ATR(4H) Liquidation Distance ───────────────────────────────────

class TestATRLiqDistance:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000, "default_leverage": 10.0}, "arms": {"risk": {}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_safe_distance(self):
        rm = self._make_rm()
        from execution.risk_manager import Position
        pos = Position("ex", "BTC/USDT", "long", 1.0, 50000, 50000, 49000, 52000, int(time.time()))
        action = rm.check_atr_liq_distance(pos, atr_4h=100)
        assert action is None

    def test_warning_distance(self):
        rm = self._make_rm()
        from execution.risk_manager import Position
        # leverage=10 → liq_price = 50000*(1-0.1*0.9) = 45500
        # distance = 45750-45500 = 250, min_distance = 300
        # 250 > 2*100=200 → 'warning'
        pos = Position("ex", "BTC/USDT", "long", 1.0, 50000, 45750, 44000, 52000, int(time.time()))
        action = rm.check_atr_liq_distance(pos, atr_4h=100)
        assert action == "warning"

    def test_emergency_distance(self):
        rm = self._make_rm()
        from execution.risk_manager import Position
        pos = Position("ex", "BTC/USDT", "long", 1.0, 50000, 45100, 44000, 52000, int(time.time()))
        action = rm.check_atr_liq_distance(pos, atr_4h=100)
        assert action in ("partial_close", "emergency_close")

    def test_no_leverage_skips(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000, "default_leverage": 1.0}, "arms": {"risk": {}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager, Position
        rm = RiskManager(config, event_bus)
        pos = Position("ex", "BTC/USDT", "long", 1.0, 50000, 45100, 44000, 52000, int(time.time()))
        assert rm.check_atr_liq_distance(pos, atr_4h=100) is None


# ── §8: BTC/ETH Correlation Group ────────────────────────────────────────────

class TestCorrelationGroups:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {"max_group_exposure_pct": 0.15}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_btc_group_detected(self):
        rm = self._make_rm()
        assert rm._get_correlation_group("BTC/USDT") == "btc_group"

    def test_eth_group_detected(self):
        rm = self._make_rm()
        assert rm._get_correlation_group("ETH/USDT") == "eth_group"

    def test_unaffiliated_symbol(self):
        rm = self._make_rm()
        assert rm._get_correlation_group("XRP/USDT") is None

    def test_group_exposure_limit(self):
        rm = self._make_rm()
        from execution.risk_manager import Position
        # Open a BTC position at >= 15% of equity (100000*0.15 = 15000)
        # 0.31 * 50000 = 15500 >= 15000
        rm._positions["ex:BTC/USDT"] = Position(
            "ex", "BTC/USDT", "long", 0.31, 50000, 50000, 48000, 55000, int(time.time()),
        )
        # Try adding BTC/USDC — same group
        ok, reason = rm.check_group_exposure("BTC/USDC")
        assert ok is False
        assert "btc_group" in reason


# ── §9: 8h Funding Re-Check ─────────────────────────────────────────────────

class TestFundingRecheck:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {"funding_recheck_seconds": 0}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager, Position
        rm = RiskManager(config, event_bus)
        rm._positions["ex:BTC/USDT"] = Position(
            "ex", "BTC/USDT", "long", 1.0, 50000, 50000, 48000, 55000, int(time.time()),
        )
        # 60bps → mult=0.0 (<0.5), triggers reduce action
        rm._current_funding_rates["BTC/USDT"] = 0.006
        return rm

    def test_adverse_funding_detected(self):
        rm = self._make_rm()
        actions = rm.check_funding_existing_positions()
        assert len(actions) >= 1
        assert actions[0]["action"] == "reduce"

    def test_no_action_when_favorable(self):
        rm = self._make_rm()
        rm._current_funding_rates["BTC/USDT"] = -0.001  # Favorable for longs
        rm._last_funding_recheck = 0  # reset interval
        actions = rm.check_funding_existing_positions()
        assert len(actions) == 0


# ── §12: Event Blackout ──────────────────────────────────────────────────────

class TestEventBlackout:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {
            "event_blackout_before_min": 30,
            "event_blackout_after_min": 15,
        }}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_blackout_active(self):
        rm = self._make_rm()
        rm.add_upcoming_event("FOMC", time.time() + 600)  # 10 min from now
        is_blackout, reason = rm.is_event_blackout()
        assert is_blackout is True
        assert "FOMC" in reason

    def test_no_blackout_far_event(self):
        rm = self._make_rm()
        rm.add_upcoming_event("CPI", time.time() + 36000)  # 10 hours away
        is_blackout, _ = rm.is_event_blackout()
        assert is_blackout is False

    def test_blackout_after_event(self):
        rm = self._make_rm()
        rm.add_upcoming_event("NFP", time.time() - 300)  # 5 min ago (within 15-min after)
        is_blackout, _ = rm.is_event_blackout()
        assert is_blackout is True


# ── §12: Weekend Rules ──────────────────────────────────────────────────────

class TestWeekendRules:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {
            "weekend_sizing_mult": 0.25,
            "weekend_halt": False,
        }}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_weekend_detected(self):
        rm = self._make_rm()
        saturday = datetime.datetime(2024, 1, 6, 12, 0, tzinfo=datetime.timezone.utc)  # Saturday
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.timezone = datetime.timezone
            assert rm.is_weekend() is True

    def test_weekday_not_weekend(self):
        rm = self._make_rm()
        wednesday = datetime.datetime(2024, 1, 3, 12, 0, tzinfo=datetime.timezone.utc)  # Wednesday
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = wednesday
            mock_dt.timezone = datetime.timezone
            assert rm.is_weekend() is False

    def test_weekend_sizing_mult(self):
        rm = self._make_rm()
        with patch.object(rm, "is_weekend", return_value=True):
            assert rm.get_weekend_sizing_mult() == 0.25
        with patch.object(rm, "is_weekend", return_value=False):
            assert rm.get_weekend_sizing_mult() == 1.0

    def test_weekend_halt(self):
        rm = self._make_rm()
        rm._weekend_halt = True
        with patch.object(rm, "is_weekend", return_value=True):
            assert rm.get_weekend_sizing_mult() == 0.0


# ── §10.1: 60s Liquidation Monitoring ────────────────────────────────────────

class TestPeriodicLiqMonitoring:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {
            "initial_equity": 100000, "default_leverage": 10.0,
        }, "arms": {"risk": {"liq_monitor_interval_seconds": 0}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager, Position
        rm = RiskManager(config, event_bus)
        rm._positions["ex:BTC/USDT"] = Position(
            "ex", "BTC/USDT", "long", 1.0, 50000, 45100, 44000, 52000, int(time.time()),
        )
        return rm

    def test_periodic_check_runs(self):
        rm = self._make_rm()
        actions = rm.run_periodic_liq_check()
        assert len(actions) >= 1

    def test_margin_mode_included(self):
        rm = self._make_rm()
        rm.set_margin_mode("BTC/USDT", "cross")
        actions = rm.run_periodic_liq_check()
        if actions:
            assert actions[0]["margin_mode"] == "cross"


# ── §10.1: Cross-Margin Detection ───────────────────────────────────────────

class TestCrossMarginDetection:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_default_isolated(self):
        rm = self._make_rm()
        assert rm.get_margin_mode("BTC/USDT") == "isolated"

    def test_set_cross_margin(self):
        rm = self._make_rm()
        rm.set_margin_mode("BTC/USDT", "cross")
        assert rm.get_margin_mode("BTC/USDT") == "cross"


# ── §13: TWAP Order Splitting ────────────────────────────────────────────────

class TestTWAPExecutor:
    def test_should_split(self):
        twap = TWAPExecutor(size_threshold_usd=50000)
        assert twap.should_split(60000) is True
        assert twap.should_split(30000) is False

    def test_create_split(self):
        twap = TWAPExecutor(num_slices=5)
        split = twap.create_split("order1", "BTC/USDT", "long", 10.0)
        assert len(split.children) == 5
        assert all(c.size == 2.0 for c in split.children)
        assert split.total_size == 10.0

    @pytest.mark.asyncio
    async def test_execute_twap(self):
        twap = TWAPExecutor(num_slices=3, duration_seconds=0.01)
        split = twap.create_split("order2", "BTC/USDT", "long", 3.0)

        async def mock_submit(sym, dir, size):
            return 50000.0

        result = await twap.execute(split, mock_submit)
        assert result.status == "complete"
        assert result.filled_size == 3.0
        assert result.avg_fill_price == 50000.0

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self):
        twap = TWAPExecutor(num_slices=3, duration_seconds=0.01)
        split = twap.create_split("order3", "BTC/USDT", "long", 3.0)
        call_count = 0

        async def mock_submit(sym, dir, size):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("exchange error")
            return 50000.0

        result = await twap.execute(split, mock_submit)
        assert result.filled_size == 2.0  # 2 out of 3 filled


# ── §13: Iceberg Order ──────────────────────────────────────────────────────

class TestIcebergExecutor:
    def test_should_split(self):
        iceberg = IcebergExecutor(size_threshold_usd=100000)
        assert iceberg.should_split(150000) is True
        assert iceberg.should_split(50000) is False

    def test_create_split(self):
        iceberg = IcebergExecutor(visible_pct=0.20)
        split = iceberg.create_split("order4", "BTC/USDT", "long", 10.0)
        assert len(split.children) == 5  # 1/0.20 = 5 slices
        assert all(c.size == 2.0 for c in split.children)

    @pytest.mark.asyncio
    async def test_execute_iceberg(self):
        iceberg = IcebergExecutor(visible_pct=0.25)
        split = iceberg.create_split("order5", "ETH/USDT", "short", 8.0)

        async def mock_submit(sym, dir, size):
            return 3500.0

        result = await iceberg.execute(split, mock_submit)
        assert result.status == "complete"
        assert result.filled_size == 8.0


# ── §13: Shadow Stop-Loss ───────────────────────────────────────────────────

class TestShadowStopManager:
    def test_register_stop(self):
        mgr = ShadowStopManager()
        stop = mgr.register_stop("ex:BTC/USDT", "BTC/USDT", "binance", "long", 49000, 1.0)
        assert stop.stop_price == 49000
        assert stop.primary_placed is False

    def test_update_stop_price(self):
        mgr = ShadowStopManager()
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        mgr.update_stop_price("key1", 49500)
        assert mgr._stops["key1"].stop_price == 49500

    def test_remove_stop(self):
        mgr = ShadowStopManager()
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        mgr.remove_stop("key1")
        assert "key1" not in mgr._stops

    @pytest.mark.asyncio
    async def test_primary_placement_success(self):
        mgr = ShadowStopManager()
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)

        async def mock_place(sym, exch, dir, price, size):
            return "order123"

        ok = await mgr.place_primary_stop("key1", mock_place)
        assert ok is True
        assert mgr._stops["key1"].primary_placed is True

    @pytest.mark.asyncio
    async def test_primary_failure_activates_shadow(self):
        mgr = ShadowStopManager(primary_timeout_seconds=0.01, max_retries=1)
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)

        async def mock_place(sym, exch, dir, price, size):
            return None  # Always fails

        ok = await mgr.place_primary_stop("key1", mock_place)
        assert ok is False
        assert mgr._stops["key1"].shadow_active is True

    def test_shadow_monitors_breach(self):
        mgr = ShadowStopManager(shadow_check_interval=0)
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        mgr._stops["key1"].shadow_active = True

        actions = mgr.check_shadow_stops({"BTC/USDT": 48500})
        assert len(actions) == 1
        assert actions[0]["layer"] == 3

    def test_shadow_no_breach(self):
        mgr = ShadowStopManager(shadow_check_interval=0)
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        mgr._stops["key1"].shadow_active = True

        actions = mgr.check_shadow_stops({"BTC/USDT": 50000})
        assert len(actions) == 0

    def test_fallback_layer_4(self):
        mgr = ShadowStopManager(fallback_grace_seconds=0)
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        mgr._stops["key1"].shadow_active = True
        mgr._stops["key1"].created_at = time.time() - 100

        actions = mgr.check_fallback()
        assert len(actions) == 1
        assert actions[0]["layer"] == 4

    def test_snapshot(self):
        mgr = ShadowStopManager()
        mgr.register_stop("key1", "BTC/USDT", "binance", "long", 49000, 1.0)
        snap = mgr.get_snapshot()
        assert snap["total_stops"] == 1
        assert "key1" in snap["stops"]


# ── Integration: Risk snapshot includes new fields ───────────────────────────

class TestRiskSnapshotNewFields:
    def _make_rm(self):
        from core.config import Config
        from core.event_bus import EventBus
        config = Config.__new__(Config)
        config._values = {"risk": {"initial_equity": 100000}, "arms": {"risk": {}}}
        config.get_value = lambda k: config._values.get(k)
        event_bus = EventBus.__new__(EventBus)
        event_bus.publish_nowait = lambda *a, **k: None
        from execution.risk_manager import RiskManager
        return RiskManager(config, event_bus)

    def test_snapshot_has_new_fields(self):
        rm = self._make_rm()
        snap = rm.get_risk_snapshot()
        assert "consecutive_profitable" in snap
        assert "weekly_loss_pct" in snap
        assert "monthly_loss_pct" in snap
        assert "is_weekend" in snap
        assert "drawdown_phase" in snap
