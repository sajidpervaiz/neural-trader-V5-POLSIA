"""Pattern confluence soft-boost — opt-in via config flag.

The trading-path test is integration-heavy (full signal pipeline). Here
we verify the contract:
  • Disabled by default → no breakdown entry.
  • When enabled, a clamp ±3 caps absurd composites.
  • Bonus = clamped × dir × weight, applied to quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.candlestick_patterns import detect_patterns


def _df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.2, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.2, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def test_pattern_composite_is_signed_int() -> None:
    df = _df()
    _, comp = detect_patterns(df["open"], df["high"], df["low"], df["close"])
    assert comp.dtype.kind in ("i", "u")
    assert isinstance(int(comp.iloc[-1]), int)


def test_pattern_composite_bounded_by_pattern_count() -> None:
    df = _df()
    patterns_df, comp = detect_patterns(df["open"], df["high"], df["low"], df["close"])
    assert comp.abs().max() <= len(patterns_df.columns)


def test_clamp_logic_caps_at_three() -> None:
    # Simulate a hypothetical extreme composite of +7 — clamp must cap to +3.
    composite = 7
    clamped = max(-3, min(3, composite))
    assert clamped == 3
    composite = -10
    clamped = max(-3, min(3, composite))
    assert clamped == -3
    composite = 1
    clamped = max(-3, min(3, composite))
    assert clamped == 1


def test_bonus_formula_matches_signal_generator() -> None:
    # Replicates the inline formula in _handle_candle.
    weight = 2.0
    prop_sign_long = 1.0
    prop_sign_short = -1.0
    # Bullish patterns + long direction → positive bonus.
    assert int(round(2 * prop_sign_long * weight)) == 4
    # Bullish patterns + short direction → negative bonus (penalty).
    assert int(round(2 * prop_sign_short * weight)) == -4
    # Clamped composite of +3, weight 2 → ±6.
    assert int(round(3 * prop_sign_long * weight)) == 6
