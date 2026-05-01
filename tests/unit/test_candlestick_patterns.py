"""Candlestick pattern detection — sanity tests on hand-crafted bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.candlestick_patterns import (
    detect_patterns, doji, hammer, engulfing, morning_star, evening_star,
    three_white_soldiers, three_black_crows,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_doji_fires_on_balanced_bar() -> None:
    o = _series([100.0])
    h = _series([101.0])
    l = _series([99.0])
    c = _series([100.05])  # body = 0.05, range = 2.0 → 2.5% << 5% threshold
    out = doji(o, h, l, c)
    assert int(out.iloc[-1]) == 100


def test_hammer_fires_on_long_lower_wick() -> None:
    # Open 100, close 100.5 (small bullish body), low 96, high 100.6
    # body = 0.5, lower = min(100,100.5) - 96 = 4.0, upper = 100.6 - max = 0.1
    o = _series([100.0]); c = _series([100.5])
    h = _series([100.6]); l = _series([96.0])
    out = hammer(o, h, l, c)
    assert int(out.iloc[-1]) == 100


def test_bullish_engulfing() -> None:
    # Bar 0: bearish 102 → 99
    # Bar 1: bullish 98 → 103 (engulfs prior body 99..102)
    o = _series([102.0, 98.0])
    c = _series([99.0, 103.0])
    h = _series([102.5, 103.5])
    l = _series([98.5, 97.5])
    out = engulfing(o, h, l, c)
    assert int(out.iloc[-1]) == 100


def test_bearish_engulfing() -> None:
    o = _series([99.0, 103.0])
    c = _series([102.0, 98.0])
    h = _series([102.5, 103.5])
    l = _series([98.5, 97.5])
    out = engulfing(o, h, l, c)
    assert int(out.iloc[-1]) == -100


def test_morning_star_three_bar_pattern() -> None:
    # Bear bar, small body, strong bull
    o = _series([110.0, 100.5, 100.5])
    c = _series([100.0, 100.8, 108.0])
    h = _series([111.0, 101.2, 108.5])
    l = _series([99.5, 100.3, 100.0])
    out = morning_star(o, h, l, c)
    assert int(out.iloc[-1]) == 100


def test_evening_star_three_bar_pattern() -> None:
    o = _series([100.0, 109.5, 109.5])
    c = _series([110.0, 109.7, 102.0])
    h = _series([110.5, 110.0, 110.0])
    l = _series([99.5, 109.2, 101.5])
    out = evening_star(o, h, l, c)
    assert int(out.iloc[-1]) == -100


def test_three_white_soldiers() -> None:
    o = _series([100.0, 102.0, 104.0])
    c = _series([102.5, 104.5, 106.5])
    h = _series([102.6, 104.6, 106.6])
    l = _series([99.8, 101.8, 103.8])
    out = three_white_soldiers(o, h, l, c)
    assert int(out.iloc[-1]) == 100


def test_three_black_crows() -> None:
    o = _series([106.0, 104.0, 102.0])
    c = _series([103.5, 101.5, 99.5])
    h = _series([106.2, 104.2, 102.2])
    l = _series([103.4, 101.4, 99.4])
    out = three_black_crows(o, h, l, c)
    assert int(out.iloc[-1]) == -100


def test_detect_patterns_aggregates_all_columns() -> None:
    rng = np.random.default_rng(1)
    n = 50
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.3, n)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    patterns_df, composite = detect_patterns(df["open"], df["high"], df["low"], df["close"])
    assert set(patterns_df.columns) >= {
        "doji", "hammer", "hanging_man", "shooting_star",
        "marubozu", "spinning_top",
        "engulfing", "harami", "piercing", "dark_cloud",
        "morning_star", "evening_star",
        "three_white_soldiers", "three_black_crows",
    }
    assert len(composite) == n
    # Composite is bounded by # patterns
    assert composite.abs().max() <= len(patterns_df.columns)
