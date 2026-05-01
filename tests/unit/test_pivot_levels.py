"""Floor pivots + Fibonacci retracement helpers."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from analysis.pivot_levels import (
    classic_pivots, fibonacci_retracement,
    pivots_from_df, fib_from_df, nearest_level,
)


def test_classic_pivots_match_formula() -> None:
    p = classic_pivots(prev_high=110.0, prev_low=95.0, prev_close=105.0)
    expected_p = (110 + 95 + 105) / 3
    assert math.isclose(p["P"], expected_p)
    assert math.isclose(p["R1"], 2 * expected_p - 95.0)
    assert math.isclose(p["S1"], 2 * expected_p - 110.0)
    assert math.isclose(p["R2"], expected_p + (110.0 - 95.0))
    assert math.isclose(p["S2"], expected_p - (110.0 - 95.0))
    assert math.isclose(p["R3"], 110.0 + 2 * (expected_p - 95.0))
    assert math.isclose(p["S3"], 95.0 - 2 * (110.0 - expected_p))


def test_fib_levels_endpoints_and_618() -> None:
    f = fibonacci_retracement(swing_high=120.0, swing_low=100.0)
    assert f["level_0"] == 100.0
    assert f["level_1000"] == 120.0
    assert math.isclose(f["level_500"], 110.0)
    # 61.8% retracement: 100 + 0.618 * 20 = 112.36
    assert math.isclose(f["level_618"], 112.36)


def test_pivots_from_df_uses_prior_bar() -> None:
    df = pd.DataFrame({
        "high": [110.0, 115.0],
        "low":  [95.0, 99.0],
        "close": [105.0, 110.0],
    })
    p = pivots_from_df(df)  # default lookback_bar=-2 → row 0
    expected_p = (110 + 95 + 105) / 3
    assert math.isclose(p["P"], expected_p)


def test_fib_from_df_window_min_max() -> None:
    df = pd.DataFrame({
        "high": [105, 110, 120, 115, 118],
        "low":  [98, 95, 105, 102, 100],
        "close": [102, 108, 118, 110, 116],
    })
    f = fib_from_df(df, lookback=5)
    assert f["level_0"] == 95.0
    assert f["level_1000"] == 120.0


def test_nearest_level_returns_closest() -> None:
    levels = {"a": 100.0, "b": 110.0, "c": 120.0}
    out = nearest_level(price=113.0, levels=levels)
    assert out is not None
    name, lvl, dist = out
    assert name == "b"
    assert lvl == 110.0
    assert dist > 0


def test_nearest_level_handles_empty() -> None:
    assert nearest_level(price=100.0, levels={}) is None


def test_pivots_from_df_short_input() -> None:
    df = pd.DataFrame({"high": [100], "low": [99], "close": [99.5]})
    assert pivots_from_df(df) is None  # need >= 2 bars
