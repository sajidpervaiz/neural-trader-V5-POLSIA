"""REQ-STATE-001..012: deterministic operational FSM."""
from __future__ import annotations

import pytest

from core.state_machine import (
    IllegalTransition,
    OperationalState,
    OperationalStateMachine,
    StateTransition,
)


def _walk_to(sm: OperationalStateMachine, target: OperationalState) -> None:
    """Helper: drive the FSM through the canonical boot path until target."""
    path = [
        OperationalState.CONFIG_VALIDATED,
        OperationalState.WARMUP,
        OperationalState.CONNECTING_MARKET_DATA,
        OperationalState.MARKET_DATA_LIVE,
        OperationalState.ORDER_BOOK_SYNCED,
        OperationalState.SIGNALING_ACTIVE,
        OperationalState.EXECUTION_ENABLED,
    ]
    for s in path:
        sm.transition_to(s, reason=f"test_walk_{s.value}")
        if s == target:
            return


def test_initial_state_is_init() -> None:
    sm = OperationalStateMachine()
    assert sm.current is OperationalState.INIT
    assert sm.is_blocking is False
    assert sm.is_trade_ready is False
    assert sm.snapshot()["current"] == "init"
    assert sm.history() == []


def test_canonical_boot_path_succeeds() -> None:
    sm = OperationalStateMachine()
    _walk_to(sm, OperationalState.EXECUTION_ENABLED)
    assert sm.current is OperationalState.EXECUTION_ENABLED
    assert sm.is_trade_ready is True
    # 7 transitions on the boot path
    assert len(sm.history()) == 7


def test_illegal_forward_skip_is_rejected() -> None:
    sm = OperationalStateMachine()
    # Cannot jump straight from INIT to EXECUTION_ENABLED
    with pytest.raises(IllegalTransition):
        sm.transition_to(OperationalState.EXECUTION_ENABLED, reason="cheat")
    assert sm.current is OperationalState.INIT


def test_safe_mode_reachable_from_any_state() -> None:
    sm = OperationalStateMachine()
    sm.transition_to(OperationalState.CONFIG_VALIDATED, reason="t")
    sm.transition_to(OperationalState.SAFE_MODE, reason="db_outage")
    assert sm.current is OperationalState.SAFE_MODE
    assert sm.is_blocking is True
    assert sm.is_trade_ready is False


def test_circuit_breaker_reachable_from_execution_enabled() -> None:
    sm = OperationalStateMachine()
    _walk_to(sm, OperationalState.EXECUTION_ENABLED)
    sm.transition_to(OperationalState.CIRCUIT_BREAKER_ACTIVE, reason="dd_breach")
    assert sm.is_blocking is True
    assert sm.is_trade_ready is False


def test_recovery_from_safe_mode_returns_to_execution() -> None:
    sm = OperationalStateMachine()
    _walk_to(sm, OperationalState.EXECUTION_ENABLED)
    sm.transition_to(OperationalState.SAFE_MODE, reason="stale_ws")
    sm.transition_to(OperationalState.EXECUTION_ENABLED, reason="ws_recovered")
    assert sm.current is OperationalState.EXECUTION_ENABLED
    assert sm.is_blocking is False


def test_in_position_round_trip() -> None:
    sm = OperationalStateMachine()
    _walk_to(sm, OperationalState.EXECUTION_ENABLED)
    sm.transition_to(OperationalState.IN_POSITION, reason="position_opened")
    assert sm.is_trade_ready is True
    sm.transition_to(OperationalState.EXECUTION_ENABLED, reason="position_closed")
    assert sm.current is OperationalState.EXECUTION_ENABLED


def test_shutdown_is_terminal() -> None:
    sm = OperationalStateMachine()
    sm.transition_to(OperationalState.SHUTDOWN, reason="bye")
    with pytest.raises(IllegalTransition):
        sm.transition_to(OperationalState.INIT, reason="back_again")


def test_no_op_transition_does_not_pollute_history() -> None:
    sm = OperationalStateMachine()
    sm.transition_to(OperationalState.CONFIG_VALIDATED, reason="cfg_ok")
    transitions_before = len(sm.history())
    out = sm.transition_to(OperationalState.CONFIG_VALIDATED, reason="redundant")
    assert sm.current is OperationalState.CONFIG_VALIDATED
    assert len(sm.history()) == transitions_before
    assert out.reason.startswith("noop:")


def test_emit_callback_fires_on_transition() -> None:
    fires: list[StateTransition] = []
    sm = OperationalStateMachine(emit_callback=fires.append)
    sm.transition_to(OperationalState.CONFIG_VALIDATED, reason="cfg_ok")
    sm.transition_to(OperationalState.WARMUP, reason="hydrate")
    assert len(fires) == 2
    assert fires[0].to_state is OperationalState.CONFIG_VALIDATED
    assert fires[1].from_state is OperationalState.CONFIG_VALIDATED


def test_force_to_bypasses_edge_check_but_records_forced_reason() -> None:
    sm = OperationalStateMachine()
    sm.force_to(OperationalState.IN_POSITION, reason="recovered_from_db")
    assert sm.current is OperationalState.IN_POSITION
    history = sm.history()
    assert len(history) == 1
    assert history[0].reason.startswith("forced:")


def test_snapshot_shape() -> None:
    sm = OperationalStateMachine()
    sm.transition_to(OperationalState.CONFIG_VALIDATED, reason="cfg_ok")
    snap = sm.snapshot()
    assert snap["current"] == "config_validated"
    assert snap["is_blocking"] is False
    assert snap["is_trade_ready"] is False
    assert "warmup" in snap["allowed_next"]
    assert isinstance(snap["history"], list)
    assert snap["history"][-1]["reason"] == "cfg_ok"
