"""
SMC (Smart Money Concepts) Signal Modules — NeuralTrader V5.

Four discrete signal types as specified:

  Type A — BreakoutPullbackModule
      Donchian-55 channel breakout → Fibonacci 38.2–61.8 % retracement entry.
      Volume > 1.5× MA at retracement point confirms institutional interest.

  Type B — LiquiditySweepReversalModule
      Price sweeps a liquidity level (equal highs/lows pool), then closes
      back inside the range.  Entry on the reversal candle.

  Type C — FVGMitigationModule
      Price returns to fill a Fair Value Gap zone.  Entry when price is
      inside the gap and a confirmation candle fires.

  Type D — OrderBlockMitigationModule
      Price revisits an Order Block zone.  Entry with volume and candlestick
      confirmation.

Each module returns a ``StrategySignal`` (same dataclass used by ARMS-V2.1
strategy modules) or ``None`` when conditions are not met.

Dependencies (columns computed by TechnicalIndicators.compute_all):
  Type A: donchian_upper_55, donchian_lower_55, donchian_mid_55, volume_ratio, atr_14
  Type B: liq_swept_high, liq_swept_low, liq_high, liq_low
  Type C: fvg_bull_active, fvg_bear_active, fvg_bull_top, fvg_bull_bot,
          fvg_bear_top, fvg_bear_bot
  Type D: ob_bull_active, ob_bear_active, ob_bull_top, ob_bull_bot,
          ob_bear_top, ob_bear_bot
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from loguru import logger

from analysis.regime import MarketRegime, RegimeState
from engine.strategy_modules import StrategySignal, _check_confirmation_candle


# ── Fibonacci retracement helpers ────────────────────────────────────────────

_FIB_LEVELS = {
    "23.6": 0.236,
    "38.2": 0.382,
    "50.0": 0.500,
    "61.8": 0.618,
    "78.6": 0.786,
}

# Entry zone: 38.2 % – 61.8 % retracement
_FIB_ENTRY_LOW = _FIB_LEVELS["38.2"]
_FIB_ENTRY_HIGH = _FIB_LEVELS["61.8"]


def _fib_retracement_zone(
    swing_low: float, swing_high: float, direction: str,
) -> tuple[float, float]:
    """Return (zone_bottom, zone_top) for the 38.2–61.8 % retracement.

    For a bullish breakout (direction='long'), the swing is from low to high
    and a pullback retraces back toward the low.  Zone is between
    high - 0.618*(high-low) and high - 0.382*(high-low).
    """
    span = swing_high - swing_low
    if span <= 0:
        return swing_low, swing_high
    if direction == "long":
        top = swing_high - _FIB_ENTRY_LOW * span
        bot = swing_high - _FIB_ENTRY_HIGH * span
    else:
        bot = swing_low + _FIB_ENTRY_LOW * span
        top = swing_low + _FIB_ENTRY_HIGH * span
    return bot, top


# ── Type A: Breakout Pullback ─────────────────────────────────────────────────

class BreakoutPullbackModule:
    """Signal Type A: Donchian-55 channel breakout followed by
    Fibonacci 38.2–61.8 % retracement entry.

    Logic (long example):
    1. A candle in the recent lookback closed above donchian_upper_55
       (breakout confirmed).
    2. Price has since pulled back into the 38.2–61.8 % Fibonacci zone
       measured from the breakout swing.
    3. Volume ≥ volume_min_ratio × SMA(20) at the pullback entry bar.
    4. Confirmation candle (body > 60 % of range, bullish).
    5. Works in all regimes (breakout is regime-agnostic).
    """

    # No hard regime restriction — breakouts can occur from any regime.
    # But we require SOME trend direction (exclude RANGE_CHOP from strong filter).

    def __init__(
        self,
        breakout_lookback: int = 20,
        fib_tolerance_pct: float = 0.005,
        volume_min_ratio: float = 1.5,
    ) -> None:
        self._lookback = breakout_lookback
        self._fib_tol = fib_tolerance_pct
        self._vol_min = volume_min_ratio

    def evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        required = ["donchian_upper_55", "donchian_lower_55", "atr_14"]
        if any(c not in df.columns for c in required):
            return None
        if df is None or len(df) < 60:
            return None

        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        vol_ratio = float(last.get("volume_ratio", 1.0))
        reasons: list[str] = []

        # ── Detect recent breakout in lookback window ────────────────────
        window = df.tail(self._lookback + 5)
        dc_upper = window["donchian_upper_55"]
        dc_lower = window["donchian_lower_55"]
        closes = window["close"]
        highs = window["high"]
        lows = window["low"]

        # Find if a breakout occurred (close > dc_upper or close < dc_lower)
        # within the lookback, but NOT on the current bar (we want pullback)
        past = window.iloc[:-1]  # exclude current bar
        bull_broke = past["close"] > past["donchian_upper_55"]
        bear_broke = past["close"] < past["donchian_lower_55"]

        direction: str | None = None

        if bull_broke.any():
            # Bullish breakout detected
            breakout_idx = bull_broke.index[bull_broke][-1]  # most recent
            breakout_bar = df.loc[breakout_idx]
            swing_low = float(lows.loc[:breakout_idx].tail(20).min())
            swing_high = float(highs.loc[breakout_idx:].max())
            fib_bot, fib_top = _fib_retracement_zone(swing_low, swing_high, "long")
            # Price must be in Fib zone (with tolerance)
            tol = price * self._fib_tol
            if fib_bot - tol <= price <= fib_top + tol:
                direction = "long"
                reasons.append(f"donchian55_bull_breakout_fib_retrace_{fib_bot:.2f}-{fib_top:.2f}")

        if direction is None and bear_broke.any():
            breakout_idx = bear_broke.index[bear_broke][-1]
            breakout_bar = df.loc[breakout_idx]
            swing_high = float(highs.loc[:breakout_idx].tail(20).max())
            swing_low = float(lows.loc[breakout_idx:].min())
            fib_bot, fib_top = _fib_retracement_zone(swing_low, swing_high, "short")
            tol = price * self._fib_tol
            if fib_bot - tol <= price <= fib_top + tol:
                direction = "short"
                reasons.append(f"donchian55_bear_breakout_fib_retrace_{fib_bot:.2f}-{fib_top:.2f}")

        if direction is None:
            return None

        # ── Volume confirmation ──────────────────────────────────────────
        if vol_ratio < self._vol_min:
            return None
        reasons.append(f"volume_{vol_ratio:.1f}x")

        # ── Confirmation candle ──────────────────────────────────────────
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("confirmation_candle")

        # ── Score ────────────────────────────────────────────────────────
        regime_bonus = 0.0
        if regime is not None:
            if direction == "long" and regime.regime in {
                MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP
            }:
                regime_bonus = 0.15
            elif direction == "short" and regime.regime in {
                MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN
            }:
                regime_bonus = 0.15
        score = min(1.0, 0.55 + (vol_ratio - self._vol_min) * 0.05 + regime_bonus)

        return StrategySignal(
            strategy="breakout_pullback_A",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"type": "A", "signal": "Breakout Pullback"},
        )


# ── Type B: Liquidity Sweep Reversal ─────────────────────────────────────────

class LiquiditySweepReversalModule:
    """Signal Type B: Liquidity sweep followed by reversal.

    Logic (long example):
    1. A liquidity level (equal lows) exists below current price.
    2. On this bar, low sweeps below that level (liq_swept_low == 1).
    3. Close returns ABOVE the swept level → trap move, reversal entry.
    4. Confirmation candle (bullish body > 60 % of range).
    5. Works best in RANGE_CHOP and COMPRESSION regimes.
    """

    PREFERRED_REGIMES = {
        MarketRegime.RANGE_CHOP,
        MarketRegime.COMPRESSION,
        MarketRegime.WEAK_TREND_UP,
        MarketRegime.WEAK_TREND_DOWN,
    }

    def __init__(self, vol_min_ratio: float = 1.0) -> None:
        self._vol_min = vol_min_ratio

    def evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        required = ["liq_swept_high", "liq_swept_low", "liq_high", "liq_low"]
        if any(c not in df.columns for c in required):
            return None
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        vol_ratio = float(last.get("volume_ratio", 1.0))
        reasons: list[str] = []

        swept_high = float(last.get("liq_swept_high", 0))
        swept_low = float(last.get("liq_swept_low", 0))
        liq_high = last.get("liq_high", np.nan)
        liq_low = last.get("liq_low", np.nan)

        direction: str | None = None

        # Long: low swept equal-lows level, then closed back above it
        if swept_low > 0 and not np.isnan(liq_low):
            bar_low = float(last.get("low", price))
            if bar_low < float(liq_low) and price > float(liq_low):
                direction = "long"
                reasons.append(f"liq_sweep_low_{liq_low:.4f}_reversal")

        # Short: high swept equal-highs level, then closed back below it
        if direction is None and swept_high > 0 and not np.isnan(liq_high):
            bar_high = float(last.get("high", price))
            if bar_high > float(liq_high) and price < float(liq_high):
                direction = "short"
                reasons.append(f"liq_sweep_high_{liq_high:.4f}_reversal")

        if direction is None:
            return None

        # Volume filter
        if vol_ratio < self._vol_min:
            return None
        reasons.append(f"volume_{vol_ratio:.1f}x")

        # Confirmation candle
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("confirmation_candle")

        # Score — boosted when in preferred regime
        regime_bonus = 0.2 if (regime and regime.regime in self.PREFERRED_REGIMES) else 0.0
        score = min(1.0, 0.6 + regime_bonus)

        return StrategySignal(
            strategy="liquidity_sweep_reversal_B",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"type": "B", "signal": "Liquidity Sweep Reversal"},
        )


# ── Type C: FVG Mitigation ───────────────────────────────────────────────────

class FVGMitigationModule:
    """Signal Type C: Fair Value Gap mitigation entry.

    Logic (long example):
    1. A bullish FVG zone exists (gap formed when price rose impulsively).
    2. Price retraces back into the bullish FVG zone (fvg_bull_active == 1).
    3. Market structure is bullish (market_structure >= 0 or regime is trending).
    4. Confirmation candle inside the gap.
    5. Entry: price in gap, direction = long (back toward equilibrium).
    """

    def __init__(self, require_structure_alignment: bool = True) -> None:
        self._require_structure = require_structure_alignment

    def evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        required = ["fvg_bull_active", "fvg_bear_active",
                    "fvg_bull_top", "fvg_bull_bot",
                    "fvg_bear_top", "fvg_bear_bot"]
        if any(c not in df.columns for c in required):
            return None
        if df is None or len(df) < 10:
            return None

        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        reasons: list[str] = []

        fvg_bull = float(last.get("fvg_bull_active", 0))
        fvg_bear = float(last.get("fvg_bear_active", 0))
        market_struct = float(last.get("market_structure", 0))

        direction: str | None = None

        # Long: price inside bullish FVG and structure is bullish or neutral
        if fvg_bull > 0:
            if self._require_structure and market_struct < -0.5:
                pass  # bearish structure contradicts bullish FVG fill
            else:
                gap_top = float(last.get("fvg_bull_top", price))
                gap_bot = float(last.get("fvg_bull_bot", price))
                direction = "long"
                reasons.append(f"fvg_bull_mitigation_{gap_bot:.4f}-{gap_top:.4f}")

        # Short: price inside bearish FVG
        if direction is None and fvg_bear > 0:
            if self._require_structure and market_struct > 0.5:
                pass
            else:
                gap_top = float(last.get("fvg_bear_top", price))
                gap_bot = float(last.get("fvg_bear_bot", price))
                direction = "short"
                reasons.append(f"fvg_bear_mitigation_{gap_bot:.4f}-{gap_top:.4f}")

        if direction is None:
            return None

        # Confirmation candle
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("confirmation_candle")

        # BOS / CHoCH alignment bonus
        if "bos_bull" in df.columns and "bos_bear" in df.columns:
            bos_bull = float(last.get("bos_bull", 0))
            bos_bear = float(last.get("bos_bear", 0))
            choch_bull = float(last.get("choch_bull", 0))
            choch_bear = float(last.get("choch_bear", 0))
            if direction == "long" and (bos_bull > 0 or choch_bull > 0):
                reasons.append("bos_choch_bull_alignment")
            elif direction == "short" and (bos_bear > 0 or choch_bear > 0):
                reasons.append("bos_choch_bear_alignment")

        # Regime alignment bonus
        regime_bonus = 0.0
        if regime:
            if direction == "long" and regime.regime in {
                MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP
            }:
                regime_bonus = 0.15
            elif direction == "short" and regime.regime in {
                MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN
            }:
                regime_bonus = 0.15

        score = min(1.0, 0.55 + regime_bonus + (0.1 if "bos_choch" in " ".join(reasons) else 0))

        return StrategySignal(
            strategy="fvg_mitigation_C",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"type": "C", "signal": "FVG Mitigation",
                      "fvg_bull_active": fvg_bull, "fvg_bear_active": fvg_bear},
        )


# ── Type D: Order Block Mitigation ───────────────────────────────────────────

class OrderBlockMitigationModule:
    """Signal Type D: Order Block mitigation entry.

    Logic (long example):
    1. A bullish Order Block zone exists (last bearish candle before a
       bullish displacement).
    2. Price retraces back into that OB zone (ob_bull_active == 1).
    3. Volume and candlestick confirmation on the entry bar.
    4. Market structure / BOS alignment adds confidence.
    """

    def __init__(
        self,
        vol_min_ratio: float = 1.0,
        require_bos_alignment: bool = False,
    ) -> None:
        self._vol_min = vol_min_ratio
        self._require_bos = require_bos_alignment

    def evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        required = ["ob_bull_active", "ob_bear_active",
                    "ob_bull_top", "ob_bull_bot",
                    "ob_bear_top", "ob_bear_bot"]
        if any(c not in df.columns for c in required):
            return None
        if df is None or len(df) < 10:
            return None

        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        vol_ratio = float(last.get("volume_ratio", 1.0))
        reasons: list[str] = []

        ob_bull = float(last.get("ob_bull_active", 0))
        ob_bear = float(last.get("ob_bear_active", 0))

        direction: str | None = None

        if ob_bull > 0:
            ob_top = float(last.get("ob_bull_top", price))
            ob_bot = float(last.get("ob_bull_bot", price))

            # Optional BOS alignment requirement
            if self._require_bos:
                bos_bull = float(last.get("bos_bull", 0))
                if bos_bull == 0:
                    pass  # no BOS confirmation
                else:
                    direction = "long"
            else:
                direction = "long"

            if direction == "long":
                reasons.append(f"ob_bull_mitigation_{ob_bot:.4f}-{ob_top:.4f}")

        if direction is None and ob_bear > 0:
            ob_top = float(last.get("ob_bear_top", price))
            ob_bot = float(last.get("ob_bear_bot", price))

            if self._require_bos:
                bos_bear = float(last.get("bos_bear", 0))
                if bos_bear > 0:
                    direction = "short"
            else:
                direction = "short"

            if direction == "short":
                reasons.append(f"ob_bear_mitigation_{ob_bot:.4f}-{ob_top:.4f}")

        if direction is None:
            return None

        # Volume filter
        if vol_ratio < self._vol_min:
            return None
        reasons.append(f"volume_{vol_ratio:.1f}x")

        # Confirmation candle
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("confirmation_candle")

        # BOS / CHoCH bonus
        struct_bonus = 0.0
        if "bos_bull" in df.columns:
            bos_bull = float(last.get("bos_bull", 0))
            bos_bear = float(last.get("bos_bear", 0))
            choch_bull = float(last.get("choch_bull", 0))
            choch_bear = float(last.get("choch_bear", 0))
            if direction == "long" and (bos_bull > 0 or choch_bull > 0):
                reasons.append("bos_bull_aligned")
                struct_bonus = 0.1
            elif direction == "short" and (bos_bear > 0 or choch_bear > 0):
                reasons.append("bos_bear_aligned")
                struct_bonus = 0.1

        # Liquidity confluence: OB near a liquidity level
        liq_high = last.get("liq_high", np.nan)
        liq_low = last.get("liq_low", np.nan)
        if direction == "long" and not np.isnan(liq_low):
            ob_bot_val = float(last.get("ob_bull_bot", price))
            if abs(float(liq_low) - ob_bot_val) / price < 0.005:
                reasons.append("liq_level_ob_confluence")
                struct_bonus += 0.05
        elif direction == "short" and not np.isnan(liq_high):
            ob_top_val = float(last.get("ob_bear_top", price))
            if abs(float(liq_high) - ob_top_val) / price < 0.005:
                reasons.append("liq_level_ob_confluence")
                struct_bonus += 0.05

        # Regime bonus
        regime_bonus = 0.0
        if regime:
            if direction == "long" and regime.regime in {
                MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP
            }:
                regime_bonus = 0.1
            elif direction == "short" and regime.regime in {
                MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN
            }:
                regime_bonus = 0.1

        score = min(1.0, 0.55 + struct_bonus + regime_bonus)

        return StrategySignal(
            strategy="order_block_mitigation_D",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={"type": "D", "signal": "Order Block Mitigation",
                      "ob_bull_active": ob_bull, "ob_bear_active": ob_bear},
        )


# ── Type E: Breaker Block ─────────────────────────────────────────────────────

class BreakerBlockModule:
    """Breaker Block — an invalidated Order Block that flips polarity (spec §5.3).

    Formation sequence:
    1. A valid OB exists (ob_bull/ob_bear active zones tracked internally)
    2. Price breaks OB boundary with candle CLOSE beyond it
    3. Market forms BOS in opposite direction
    4. Price returns to the old OB zone → zone becomes a Breaker Block

    Confirmation requirements (spec §5.3.2):
    - Rejection candle: wick ≥ 2× body
    - Volume spike ≥ 180% of 20-period average

    State: Breaker zones are tracked as a list of dicts per direction.
    Call ``update(df)`` each bar BEFORE ``evaluate(df)``.
    """

    _MAX_ZONES = 10  # cap tracked breaker zones per side

    def __init__(self, vol_min_ratio: float = 1.8) -> None:
        self._vol_min = vol_min_ratio
        self._bull_breakers: list[dict] = []   # {'top', 'bot', 'bar_idx'}
        self._bear_breakers: list[dict] = []

    # ── State update ──────────────────────────────────────────────────────────

    def update(self, df: pd.DataFrame) -> None:
        """Detect new breaker block formations and update internal state.

        Must be called on each new closed bar before evaluate().
        """
        if df is None or len(df) < 10:
            return

        last = df.iloc[-1]
        close = float(last.get("close", 0))

        # ── Detect if an OB was just broken → creates a breaker zone ────────
        # Bullish OB broken downward → becomes a BEARISH Breaker Block
        if float(last.get("ob_bull_active", 0)) > 0:
            ob_bot = float(last.get("ob_bull_bot", np.nan))
            ob_top = float(last.get("ob_bull_top", np.nan))
            if not np.isnan(ob_bot) and close < ob_bot:
                # OB invalidated — create bearish breaker at this zone
                self._bear_breakers.append({
                    "top": ob_top,
                    "bot": ob_bot,
                    "bar_idx": len(df) - 1,
                })
                if len(self._bear_breakers) > self._MAX_ZONES:
                    self._bear_breakers.pop(0)

        # Bearish OB broken upward → becomes a BULLISH Breaker Block
        if float(last.get("ob_bear_active", 0)) > 0:
            ob_bot = float(last.get("ob_bear_bot", np.nan))
            ob_top = float(last.get("ob_bear_top", np.nan))
            if not np.isnan(ob_top) and close > ob_top:
                self._bull_breakers.append({
                    "top": ob_top,
                    "bot": ob_bot,
                    "bar_idx": len(df) - 1,
                })
                if len(self._bull_breakers) > self._MAX_ZONES:
                    self._bull_breakers.pop(0)

    def evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        """Check if price has returned to a breaker zone with confirmation."""
        if df is None or len(df) < 10 or not self._bull_breakers and not self._bear_breakers:
            return None

        last = df.iloc[-1]
        price = float(last.get("close", 0))
        atr = float(last.get("atr_14", price * 0.01))
        vol_ratio = float(last.get("volume_ratio", 1.0))
        reasons: list[str] = []

        if vol_ratio < self._vol_min:
            return None

        # Rejection candle: wick ≥ 2× body
        open_ = float(last.get("open", price))
        high = float(last.get("high", price))
        low = float(last.get("low", price))
        body = abs(price - open_)
        upper_wick = high - max(price, open_)
        lower_wick = min(price, open_) - low

        direction: str | None = None
        zone: dict | None = None

        # Check bullish breakers (price returning into former bearish OB zone)
        for br in reversed(self._bull_breakers):
            if br["bot"] <= price <= br["top"]:
                if lower_wick >= 2 * body and body > 0:
                    direction = "long"
                    zone = br
                    reasons.append(f"bull_breaker_{br['bot']:.4f}-{br['top']:.4f}")
                    break

        # Check bearish breakers
        if direction is None:
            for br in reversed(self._bear_breakers):
                if br["bot"] <= price <= br["top"]:
                    if upper_wick >= 2 * body and body > 0:
                        direction = "short"
                        zone = br
                        reasons.append(f"bear_breaker_{br['bot']:.4f}-{br['top']:.4f}")
                        break

        if direction is None or zone is None:
            return None

        reasons.append(f"volume_{vol_ratio:.1f}x")
        if not _check_confirmation_candle(df, direction):
            return None
        reasons.append("rejection_candle_confirmed")

        # Mark active for layer scorer
        active_key = "breaker_bull_active" if direction == "long" else "breaker_bear_active"

        regime_bonus = 0.0
        if regime:
            if direction == "long" and regime.regime in {
                MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP
            }:
                regime_bonus = 0.1
            elif direction == "short" and regime.regime in {
                MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN
            }:
                regime_bonus = 0.1

        score = min(1.0, 0.65 + regime_bonus)

        return StrategySignal(
            strategy="breaker_block_E",
            direction=direction,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons,
            metadata={
                "type": "E",
                "signal": "Breaker Block",
                active_key: 1,
                "breaker_top": zone["top"],
                "breaker_bot": zone["bot"],
            },
        )

    def inject_active_flags(self, df: pd.DataFrame, price: float) -> dict[str, float]:
        """Return dict of breaker_bull_active / breaker_bear_active flags for scoring."""
        bull_active = any(br["bot"] <= price <= br["top"] for br in self._bull_breakers)
        bear_active = any(br["bot"] <= price <= br["top"] for br in self._bear_breakers)
        return {
            "breaker_bull_active": float(bull_active),
            "breaker_bear_active": float(bear_active),
        }


# ── SMC Strategy Selector ────────────────────────────────────────────────────

class SMCStrategySelector:
    """Evaluates all 4 SMC signal type modules and returns the highest-scoring
    active signal, or None if no setup is present.

    Unlike the ARMS-V2.1 StrategySelector (which gates by regime), the SMC
    selector tries all modules on every bar — each module has its own internal
    conditions and may return None.
    """

    def __init__(
        self,
        breakout_pullback: BreakoutPullbackModule | None = None,
        liquidity_sweep: LiquiditySweepReversalModule | None = None,
        fvg_mitigation: FVGMitigationModule | None = None,
        order_block: OrderBlockMitigationModule | None = None,
        breaker_block: BreakerBlockModule | None = None,
    ) -> None:
        self.breakout_pullback = breakout_pullback or BreakoutPullbackModule()
        self.liquidity_sweep = liquidity_sweep or LiquiditySweepReversalModule()
        self.fvg_mitigation = fvg_mitigation or FVGMitigationModule()
        self.order_block = order_block or OrderBlockMitigationModule()
        self.breaker_block = breaker_block or BreakerBlockModule()

    def select_and_evaluate(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> StrategySignal | None:
        """Evaluate all SMC modules and return the highest-scoring signal."""
        # Update stateful modules first
        try:
            self.breaker_block.update(df)
        except Exception:
            pass

        candidates: list[StrategySignal] = []

        for module in (
            self.breaker_block,     # highest precision — evaluate first
            self.order_block,
            self.fvg_mitigation,
            self.liquidity_sweep,
            self.breakout_pullback,
        ):
            try:
                sig = module.evaluate(df, regime)
                if sig is not None:
                    candidates.append(sig)
            except Exception as exc:
                logger.debug("SMC module {} error: {}", module.__class__.__name__, exc)

        if not candidates:
            return None

        # Return highest-scoring candidate
        return max(candidates, key=lambda s: s.score)

    def evaluate_all(
        self, df: pd.DataFrame, regime: RegimeState | None = None,
    ) -> list[StrategySignal]:
        """Return all active SMC signals (may be multiple simultaneously)."""
        signals: list[StrategySignal] = []
        for module in (
            self.breaker_block,
            self.order_block,
            self.fvg_mitigation,
            self.liquidity_sweep,
            self.breakout_pullback,
        ):
            try:
                sig = module.evaluate(df, regime)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("SMC module {} error: {}", module.__class__.__name__, exc)
        return signals
