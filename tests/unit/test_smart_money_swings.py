"""Regression: SmartMoneyAnalyzer must not crash when the input DataFrame has
non-unique index labels.

When the candle DataFrame carries duplicate timestamps in its index, the
old swing-extraction path called ``df.index.get_loc(label)`` which can return
a ``slice`` (or boolean mask) instead of an int. The later ``swing_highs.sort()``
then raised:

    TypeError: '<' not supported between instances of 'slice' and 'int'

This test exercises the duplicate-timestamp path and asserts the analyzer
returns an SMCState without raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.smart_money import SmartMoneyAnalyzer


def _candles_with_duplicate_timestamps(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.3, n)
    vols = rng.uniform(100, 500, n)
    # Pre-computed swing markers — these trigger the get_loc() path
    swing_high = pd.Series(np.where(rng.random(n) < 0.1, highs, np.nan))
    swing_low = pd.Series(np.where(rng.random(n) < 0.1, lows, np.nan))
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
        "swing_high": swing_high, "swing_low": swing_low,
    })
    # Force duplicate index labels — this is what makes get_loc return a slice
    ts = pd.date_range("2026-04-28", periods=n // 2, freq="1min").repeat(2)
    df.index = ts[: len(df)]
    return df


def test_analyze_handles_duplicate_index_labels():
    df = _candles_with_duplicate_timestamps()
    analyzer = SmartMoneyAnalyzer()
    state = analyzer.analyze(df)
    # Just needs to return without raising; specific fields aren't asserted
    # because random input may produce empty lists.
    assert state is not None


def test_extract_swings_returns_int_positions():
    df = _candles_with_duplicate_timestamps()
    analyzer = SmartMoneyAnalyzer()
    highs, lows = analyzer._extract_swings(df)
    for idx, _ in highs + lows:
        assert isinstance(idx, int), f"expected int index, got {type(idx).__name__}={idx!r}"
