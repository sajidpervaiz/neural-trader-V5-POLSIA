"""Candlestick pattern recognition (REQ-IND extension).

Implements 14 institutional-grade candlestick patterns. Each detector is a
pure numpy/pandas function returning a Series of values:

  +100  → bullish pattern at this bar
  -100  → bearish pattern at this bar
     0  → no pattern

Matches the talib convention so the output can be drop-in-replaced if
talib is later installed (we don't depend on it — talib isn't available
in the bot's runtime).

Public surface:

  detect_patterns(open_, high, low, close)
    → returns (df_per_pattern, composite) where:
        df_per_pattern is a DataFrame with one column per pattern
        composite is a Series with the net pattern score:
          +N if bullish patterns dominate this bar (count × 100)
          -N if bearish patterns dominate
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# ── Building blocks ─────────────────────────────────────────────────────────

def _body(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_).abs()


def _range(high: pd.Series, low: pd.Series) -> pd.Series:
    return (high - low).abs()


def _upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def _lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([open_, close], axis=1).min(axis=1) - low


def _is_bullish(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close > open_


def _is_bearish(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close < open_


# ── Single-candle patterns ──────────────────────────────────────────────────

def doji(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
         body_pct: float = 0.05) -> pd.Series:
    """Doji — body <= body_pct of total range. Direction-neutral; emit +100."""
    rng = _range(high, low).replace(0, np.nan)
    body = _body(open_, close)
    is_doji = (body / rng) <= body_pct
    return pd.Series(np.where(is_doji.fillna(False), 100, 0), index=close.index)


def hammer(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Hammer — small body near top, long lower wick >= 2× body. Bullish."""
    body = _body(open_, close)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)
    cond = (lower >= 2 * body) & (upper <= body) & (body > 0)
    return pd.Series(np.where(cond, 100, 0), index=close.index)


def hanging_man(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                trend_lookback: int = 5) -> pd.Series:
    """Hanging Man — same shape as hammer, requires prior uptrend. Bearish."""
    body = _body(open_, close)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)
    shape = (lower >= 2 * body) & (upper <= body) & (body > 0)
    in_uptrend = close > close.shift(trend_lookback)
    cond = shape & in_uptrend
    return pd.Series(np.where(cond, -100, 0), index=close.index)


def shooting_star(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                  trend_lookback: int = 5) -> pd.Series:
    """Shooting Star — small body near bottom, long upper wick >= 2× body, in uptrend."""
    body = _body(open_, close)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)
    shape = (upper >= 2 * body) & (lower <= body) & (body > 0)
    in_uptrend = close > close.shift(trend_lookback)
    cond = shape & in_uptrend
    return pd.Series(np.where(cond, -100, 0), index=close.index)


def marubozu(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
             wick_pct: float = 0.03) -> pd.Series:
    """Marubozu — body fills the bar (wicks <= wick_pct of body). +100/-100 by direction."""
    body = _body(open_, close)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)
    no_wicks = (upper <= body * wick_pct) & (lower <= body * wick_pct) & (body > 0)
    bullish = no_wicks & _is_bullish(open_, close)
    bearish = no_wicks & _is_bearish(open_, close)
    out = pd.Series(0, index=close.index, dtype=int)
    out[bullish] = 100
    out[bearish] = -100
    return out


def spinning_top(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                 body_pct: float = 0.30) -> pd.Series:
    """Spinning top — small body (~30% of range) with both wicks visible. Indecision."""
    rng = _range(high, low).replace(0, np.nan)
    body = _body(open_, close)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)
    cond = (body / rng <= body_pct) & (upper > 0) & (lower > 0)
    return pd.Series(np.where(cond.fillna(False), 100, 0), index=close.index)


# ── Two-candle patterns ─────────────────────────────────────────────────────

def engulfing(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Engulfing — current body engulfs prior. Bullish if today bullish + prior bearish."""
    prev_bull = _is_bullish(open_.shift(1), close.shift(1))
    prev_bear = _is_bearish(open_.shift(1), close.shift(1))
    today_bull = _is_bullish(open_, close)
    today_bear = _is_bearish(open_, close)
    bullish = (
        today_bull & prev_bear
        & (open_ <= close.shift(1)) & (close >= open_.shift(1))
        & ((close - open_) > (open_.shift(1) - close.shift(1)))
    )
    bearish = (
        today_bear & prev_bull
        & (open_ >= close.shift(1)) & (close <= open_.shift(1))
        & ((open_ - close) > (close.shift(1) - open_.shift(1)))
    )
    out = pd.Series(0, index=close.index, dtype=int)
    out[bullish] = 100
    out[bearish] = -100
    return out


def harami(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Harami — small body INSIDE prior larger body."""
    prev_bull = _is_bullish(open_.shift(1), close.shift(1))
    prev_bear = _is_bearish(open_.shift(1), close.shift(1))
    today_body = _body(open_, close)
    prev_body = _body(open_.shift(1), close.shift(1))
    inside = (
        (open_.shift(1).where(prev_bull, close.shift(1)) >= pd.concat([open_, close], axis=1).max(axis=1))
        & (close.shift(1).where(prev_bull, open_.shift(1)) <= pd.concat([open_, close], axis=1).min(axis=1))
    )
    smaller = today_body < prev_body * 0.6
    bullish = inside & smaller & prev_bear & _is_bullish(open_, close)
    bearish = inside & smaller & prev_bull & _is_bearish(open_, close)
    out = pd.Series(0, index=close.index, dtype=int)
    out[bullish] = 100
    out[bearish] = -100
    return out


def piercing_pattern(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Piercing — bullish reversal: bear candle, then gap-down open + close >50% into prior body."""
    prev_bear = _is_bearish(open_.shift(1), close.shift(1))
    midpoint = (open_.shift(1) + close.shift(1)) / 2.0
    cond = (
        prev_bear
        & (open_ < close.shift(1))           # gap-down open
        & (close > midpoint)                 # closed above midpoint of prior body
        & (close < open_.shift(1))           # but not full engulf
        & _is_bullish(open_, close)
    )
    return pd.Series(np.where(cond, 100, 0), index=close.index)


def dark_cloud_cover(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Dark Cloud Cover — bearish reversal mirror of Piercing."""
    prev_bull = _is_bullish(open_.shift(1), close.shift(1))
    midpoint = (open_.shift(1) + close.shift(1)) / 2.0
    cond = (
        prev_bull
        & (open_ > close.shift(1))           # gap-up open
        & (close < midpoint)                 # closed below midpoint of prior body
        & (close > open_.shift(1))           # but not full engulf
        & _is_bearish(open_, close)
    )
    return pd.Series(np.where(cond, -100, 0), index=close.index)


# ── Three-candle patterns ───────────────────────────────────────────────────

def morning_star(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Morning Star — bear candle, small-body, then strong bull. Bullish reversal."""
    bar1_bear = _is_bearish(open_.shift(2), close.shift(2))
    bar1_body = _body(open_.shift(2), close.shift(2))
    bar2_body = _body(open_.shift(1), close.shift(1))
    bar3_bull = _is_bullish(open_, close)
    bar3_body = _body(open_, close)
    midpoint1 = (open_.shift(2) + close.shift(2)) / 2.0
    cond = (
        bar1_bear
        & (bar2_body < bar1_body * 0.5)
        & bar3_bull
        & (bar3_body > bar1_body * 0.5)
        & (close > midpoint1)
    )
    return pd.Series(np.where(cond, 100, 0), index=close.index)


def evening_star(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Evening Star — bull candle, small-body, then strong bear. Bearish reversal."""
    bar1_bull = _is_bullish(open_.shift(2), close.shift(2))
    bar1_body = _body(open_.shift(2), close.shift(2))
    bar2_body = _body(open_.shift(1), close.shift(1))
    bar3_bear = _is_bearish(open_, close)
    bar3_body = _body(open_, close)
    midpoint1 = (open_.shift(2) + close.shift(2)) / 2.0
    cond = (
        bar1_bull
        & (bar2_body < bar1_body * 0.5)
        & bar3_bear
        & (bar3_body > bar1_body * 0.5)
        & (close < midpoint1)
    )
    return pd.Series(np.where(cond, -100, 0), index=close.index)


def three_white_soldiers(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Three consecutive bullish candles, each closing near its high."""
    bull3 = _is_bullish(open_.shift(2), close.shift(2))
    bull2 = _is_bullish(open_.shift(1), close.shift(1))
    bull1 = _is_bullish(open_, close)
    rising = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    near_high = ((high - close) <= _body(open_, close) * 0.25) & (
        (high.shift(1) - close.shift(1)) <= _body(open_.shift(1), close.shift(1)) * 0.25
    ) & (
        (high.shift(2) - close.shift(2)) <= _body(open_.shift(2), close.shift(2)) * 0.25
    )
    cond = bull3 & bull2 & bull1 & rising & near_high
    return pd.Series(np.where(cond, 100, 0), index=close.index)


def three_black_crows(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Three consecutive bearish candles, each closing near its low."""
    bear3 = _is_bearish(open_.shift(2), close.shift(2))
    bear2 = _is_bearish(open_.shift(1), close.shift(1))
    bear1 = _is_bearish(open_, close)
    falling = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    near_low = ((close - low) <= _body(open_, close) * 0.25) & (
        (close.shift(1) - low.shift(1)) <= _body(open_.shift(1), close.shift(1)) * 0.25
    ) & (
        (close.shift(2) - low.shift(2)) <= _body(open_.shift(2), close.shift(2)) * 0.25
    )
    cond = bear3 & bear2 & bear1 & falling & near_low
    return pd.Series(np.where(cond, -100, 0), index=close.index)


# ── Public aggregator ───────────────────────────────────────────────────────

_PATTERN_FNS: dict[str, Callable[..., pd.Series]] = {
    "doji": doji,
    "hammer": hammer,
    "hanging_man": hanging_man,
    "shooting_star": shooting_star,
    "marubozu": marubozu,
    "spinning_top": spinning_top,
    "engulfing": engulfing,
    "harami": harami,
    "piercing": piercing_pattern,
    "dark_cloud": dark_cloud_cover,
    "morning_star": morning_star,
    "evening_star": evening_star,
    "three_white_soldiers": three_white_soldiers,
    "three_black_crows": three_black_crows,
}


def detect_patterns(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run every detector. Returns (per-pattern DataFrame, composite Series).

    Composite at each bar = sum of (sign of pattern × 1). +N when bullish
    detections outnumber bearish; -N when reversed.
    """
    cols: dict[str, pd.Series] = {}
    for name, fn in _PATTERN_FNS.items():
        cols[name] = fn(open_, high, low, close).fillna(0).astype(int)
    df = pd.DataFrame(cols, index=close.index)
    composite = df.apply(lambda row: int(np.sign(row).sum()), axis=1)
    return df, composite


__all__ = [
    "detect_patterns",
    "doji", "hammer", "hanging_man", "shooting_star", "marubozu", "spinning_top",
    "engulfing", "harami", "piercing_pattern", "dark_cloud_cover",
    "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
]
