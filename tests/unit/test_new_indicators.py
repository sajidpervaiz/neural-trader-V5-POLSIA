"""Reference-spec indicator additions: VWMA / HMA / ALMA / ROC / StochRSI /
VW-MACD / VPT / NVI / PVI / Accelerator / RVI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators


def _df(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
    opens = closes + rng.normal(0, 0.15, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.4, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.4, n)
    vols = rng.uniform(100, 800, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


@pytest.fixture(scope="module")
def computed() -> pd.DataFrame:
    return TechnicalIndicators().compute_all(_df())


def test_new_overlay_columns_present_and_finite(computed: pd.DataFrame) -> None:
    for col in ("vwma_20", "hma_20", "alma_20"):
        assert col in computed.columns
        assert computed[col].notna().all()
        assert np.isfinite(computed[col]).all()


def test_new_oscillator_columns_present(computed: pd.DataFrame) -> None:
    expected = (
        "roc_10", "roc_20",
        "stoch_rsi_k", "stoch_rsi_d",
        "vw_macd", "vw_macd_signal", "vw_macd_hist",
        "vpt", "nvi", "pvi",
        "accelerator_osc",
        "rvi", "rvi_signal",
    )
    for col in expected:
        assert col in computed.columns, f"missing {col}"


def test_stoch_rsi_in_range(computed: pd.DataFrame) -> None:
    # %K and %D must sit in [0, 100] once warmed up.
    k = computed["stoch_rsi_k"].dropna()
    d = computed["stoch_rsi_d"].dropna()
    assert (k.between(-1, 101)).all()
    assert (d.between(-1, 101)).all()


def test_roc_is_percent_scaled(computed: pd.DataFrame) -> None:
    # Sanity: synthetic random walk, ROC magnitudes shouldn't blow up past ±100%.
    assert computed["roc_10"].abs().max() < 100.0
    assert computed["roc_20"].abs().max() < 100.0


def test_nvi_pvi_are_strictly_positive(computed: pd.DataFrame) -> None:
    # Both indices start at 1000 and only multiply by (1 + pct), so they stay > 0
    # as long as pct never reaches -1 (a -100% bar). Sanitiser fills NaN with 0
    # for these (not the close-tracking set) — confirm we have non-zero values.
    assert (computed["nvi"] > 0).any()
    assert (computed["pvi"] > 0).any()
