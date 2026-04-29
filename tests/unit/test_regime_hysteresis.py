"""REQ-REG-005: regime must persist for N consecutive candles before
switching, and hysteresis state must be inspectable via the public API."""
from __future__ import annotations

from analysis.regime import MarketRegime, RegimeDetector


def test_hysteresis_snapshot_initial_state() -> None:
    rd = RegimeDetector()
    snap = rd.get_hysteresis_snapshot()
    assert snap["current_regime"] == MarketRegime.UNKNOWN.value
    assert snap["candles_in_state"] == 0
    assert snap["pending_regime"] is None
    assert snap["pending_count"] == 0
    assert snap["confirmation_required"] >= 1
    assert snap["cooldown_remaining"] == 0
    assert snap["cooldown_total"] >= 0
    assert snap["recent_regimes"] == []


def test_pending_regime_requires_n_candles_to_commit() -> None:
    rd = RegimeDetector(confirmation_candles=3, cooldown_candles=0)
    # Force the state machine into a known starting regime by calling the
    # internal apply directly with the same regime several times to register it.
    rd._current_regime = MarketRegime.RANGE_CHOP

    # Apply two raw signals proposing a transition — should NOT switch yet.
    rd._apply_state_machine(MarketRegime.WEAK_TREND_UP)
    rd._apply_state_machine(MarketRegime.WEAK_TREND_UP)
    snap_pre = rd.get_hysteresis_snapshot()
    assert snap_pre["current_regime"] == MarketRegime.RANGE_CHOP.value
    assert snap_pre["pending_regime"] == MarketRegime.WEAK_TREND_UP.value
    assert snap_pre["pending_count"] == 2

    # Third apply commits the transition.
    rd._apply_state_machine(MarketRegime.WEAK_TREND_UP)
    snap_post = rd.get_hysteresis_snapshot()
    assert snap_post["current_regime"] == MarketRegime.WEAK_TREND_UP.value
    assert snap_post["pending_regime"] is None


def test_cooldown_blocks_further_transitions() -> None:
    rd = RegimeDetector(confirmation_candles=2, cooldown_candles=5)
    rd._current_regime = MarketRegime.RANGE_CHOP
    rd._apply_state_machine(MarketRegime.WEAK_TREND_UP)
    rd._apply_state_machine(MarketRegime.WEAK_TREND_UP)
    assert rd.get_hysteresis_snapshot()["current_regime"] == MarketRegime.WEAK_TREND_UP.value
    # Cooldown is now active — even repeated opposing signals shouldn't flip.
    for _ in range(4):
        rd._apply_state_machine(MarketRegime.WEAK_TREND_DOWN)
    snap = rd.get_hysteresis_snapshot()
    assert snap["current_regime"] == MarketRegime.WEAK_TREND_UP.value
    assert snap["cooldown_remaining"] >= 0
