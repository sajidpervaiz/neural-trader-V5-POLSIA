"""REQ-IND-017: VolumeProfileAnalyzer must flag CVD output as degraded
when the candle-close aggressor proxy is inconclusive."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.volume_profile import VolumeProfileAnalyzer


def _doji_df(n: int = 60) -> pd.DataFrame:
    """All bars are doji — close is at the midpoint of the bar range."""
    rng = np.random.default_rng(1)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    opens = closes.copy()
    highs = closes + 1.0
    lows = closes - 1.0
    vols = rng.uniform(100, 500, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


def _trending_df(n: int = 60) -> pd.DataFrame:
    """All bars close near the high — strong directional aggressor inference."""
    rng = np.random.default_rng(2)
    base = np.linspace(100, 110, n)
    closes = base + rng.uniform(0, 0.05, n)
    highs = closes + rng.uniform(0, 0.05, n)
    lows = base - 1.0
    opens = base
    vols = rng.uniform(100, 500, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


def test_cvd_degraded_on_doji_window() -> None:
    state = VolumeProfileAnalyzer().analyze(_doji_df())
    assert state.cvd_degraded is True
    assert state.cvd_ambiguous_pct >= 0.5
    assert state.cvd_inference_method == "close_position_proxy"


def test_cvd_not_degraded_on_trending_window() -> None:
    state = VolumeProfileAnalyzer().analyze(_trending_df())
    assert state.cvd_degraded is False
    assert state.cvd_ambiguous_pct < 0.5


def test_cvd_degraded_does_not_contribute_to_flow_score() -> None:
    state = VolumeProfileAnalyzer().analyze(_doji_df())
    # When degraded, the delta_trend contribution is suppressed — verify
    # the reasons list mentions degradation but NOT the delta direction.
    assert any("cvd_degraded" in r for r in state.reasons)
    assert not any(r in ("delta_accumulating", "delta_distributing") for r in state.reasons)
