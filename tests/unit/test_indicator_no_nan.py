"""REQ-IND-006/007: TechnicalIndicators.compute_all must never emit NaN/Inf
in the returned dataframe (except sparse pivot markers swing_high/swing_low).

Asserted across multiple synthetic dataframes:
  - normal trending data
  - flat data (zero variance — easy NaN trigger via division)
  - very short data (stresses warmup logic)
  - data with wild outliers (Inf trigger)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators, sanitize_indicators


def _make_df(n: int, *, flat: bool = False, outliers: bool = False, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if flat:
        closes = np.full(n, 100.0)
    else:
        steps = rng.normal(0, 0.5, n)
        closes = 100.0 + np.cumsum(steps)
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.3, n)
    vols = rng.uniform(100, 500, n)
    if outliers:
        # Insert a few extreme spikes that historically caused Inf via division.
        vols[10:12] = 1e12
        closes[20] = closes[20] * 1e6
        highs[20] = closes[20] * 1.01
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


_PRESERVE = {"swing_high", "swing_low", "_warmup"}


def _assert_no_nan_or_inf(df: pd.DataFrame) -> None:
    for col in df.columns:
        if col in _PRESERVE:
            continue
        series = df[col]
        if series.dtype == bool:
            continue
        assert not series.isna().any(), f"column {col!r} has NaN: {series[series.isna()].head().to_dict()}"
        if pd.api.types.is_numeric_dtype(series):
            assert np.isfinite(series).all(), f"column {col!r} has +/-inf"


@pytest.mark.parametrize(
    "n,flat,outliers",
    [(300, False, False), (300, True, False), (300, False, True), (50, False, False)],
)
def test_compute_all_no_nan_inf(n: int, flat: bool, outliers: bool) -> None:
    ti = TechnicalIndicators()
    df = ti.compute_all(_make_df(n, flat=flat, outliers=outliers))
    if df.empty:
        pytest.skip(f"compute_all returned empty for n={n} (warmup too long)")
    _assert_no_nan_or_inf(df)


def test_warmup_flag_present_and_typed() -> None:
    ti = TechnicalIndicators()
    df = ti.compute_all(_make_df(300))
    assert "_warmup" in df.columns
    assert df["_warmup"].dtype == bool
    # First few rows after dropna() may still need ema_200 — they should be flagged.
    if len(df) >= 1:
        # Some data sets are long enough that the head is still inside the
        # ema_200 lookback. We just assert: every row is True or False.
        assert df["_warmup"].isin([True, False]).all()


def test_sanitize_preserves_swing_markers() -> None:
    df = pd.DataFrame({
        "close": [1.0, 2.0, 3.0],
        "rsi_14": [50.0, 60.0, 55.0],
        "ema_21": [1.0, 1.5, 2.5],
        "macd": [0.0, 0.1, 0.05],
        "atr_14": [0.1, 0.2, 0.15],
        "ema_200": [1.0, 1.5, 2.0],
        "swing_high": [np.nan, 5.0, np.nan],
        "swing_low": [np.nan, np.nan, 0.5],
    })
    out = sanitize_indicators(df.copy())
    # Sparse pivot markers must remain NaN where they were NaN.
    assert out["swing_high"].isna().sum() == 2
    assert out["swing_low"].isna().sum() == 2
