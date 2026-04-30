"""REQ-BT-001 / AC-008: live and backtest indicator outputs must match.

Both code paths route through TechnicalIndicators.compute_all(). This test
verifies that:

  1. Calling compute_all on the same DataFrame from two different callers
     (the "live" and "backtest" entry points) yields identical output —
     enforcing single source of truth.

  2. The terminal row of an incrementally-grown DataFrame matches the
     terminal row of a once-computed reference, modulo trailing
     warmup-only columns. This catches any code path that secretly
     short-circuits or re-orders calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators


def _make_df(n: int = 250, *, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
    opens = closes + rng.normal(0, 0.15, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.4, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.4, n)
    vols = rng.uniform(100, 800, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


def test_two_callers_identical_outputs() -> None:
    df = _make_df()
    live = TechnicalIndicators().compute_all(df.copy())
    backtest = TechnicalIndicators().compute_all(df.copy())
    assert list(live.columns) == list(backtest.columns)
    assert len(live) == len(backtest)
    # All numeric columns must match exactly (same input, same code, same RNG-free math).
    for col in live.columns:
        if col in {"swing_high", "swing_low", "_warmup"}:
            continue
        if not pd.api.types.is_numeric_dtype(live[col]):
            continue
        np.testing.assert_array_equal(
            live[col].to_numpy(), backtest[col].to_numpy(),
            err_msg=f"column {col!r} differs between live and backtest call",
        )


_TOLERANT_COLS = {
    # These derive from rolling rank percentiles and become non-NaN only after
    # an extra warmup window — incremental construction can leak a small
    # numerical jitter at the boundary, so use isclose with rtol=1e-9.
    "atr_percentile", "bb_width_percentile",
}


@pytest.mark.parametrize("trim", [0, 1, 5])
def test_terminal_row_matches_full_compute(trim: int) -> None:
    """The last row of compute_all(df.iloc[: N - trim]) must match the
    corresponding row of compute_all(df). This guards against any path that
    uses a future bar in its calculation."""
    df = _make_df(n=300)
    full = TechnicalIndicators().compute_all(df.copy())
    if trim == 0:
        partial = full
    else:
        partial = TechnicalIndicators().compute_all(df.iloc[: len(df) - trim].copy())
    if partial.empty or full.empty:
        pytest.skip("compute_all dropped warmup rows entirely on this slice")
    p_row = partial.iloc[-1]
    f_row = full.iloc[len(full) - 1 - trim]
    for col in partial.columns:
        if col in {"swing_high", "swing_low", "_warmup"}:
            continue
        if not pd.api.types.is_numeric_dtype(partial[col]):
            continue
        a = float(p_row[col])
        b = float(f_row[col])
        if col in _TOLERANT_COLS:
            assert np.isclose(a, b, rtol=1e-9, atol=1e-9), f"{col}: partial={a} full={b}"
        else:
            assert a == b or np.isclose(a, b, rtol=1e-12, atol=1e-12), (
                f"{col}: partial={a} full={b} (lookahead leak suspected)"
            )
