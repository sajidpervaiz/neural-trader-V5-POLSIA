"""Operational State Machine — SRS REQ-STATE-001..012.

Single source of truth for "are we cleared to trade right now?". The 13
states mirror the SRS exactly. Transitions are restricted by an allowed-edges
map and recorded with their reason in a bounded history ring buffer.

Intentionally observational: the FSM does NOT drive trading. Existing
safety primitives (SafeModeManager, CircuitBreaker, kill switch) still
gate execution at their own layers; this module just records the global
state derived from them so the dashboard and tests have one place to ask.

The boot path drives transitions explicitly:

    INIT → CONFIG_VALIDATED → WARMUP → CONNECTING_MARKET_DATA →
    MARKET_DATA_LIVE → ORDER_BOOK_SYNCED → SIGNALING_ACTIVE →
    EXECUTION_ENABLED

From EXECUTION_ENABLED the runtime can move to IN_POSITION (when a
position opens), and any of {SAFE_MODE, CIRCUIT_BREAKER_ACTIVE,
MANUAL_KILL_ACTIVE} can be entered at any time. Recovery from those
returns to EXECUTION_ENABLED.

SHUTDOWN is terminal.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class OperationalState(str, Enum):
    INIT = "init"
    CONFIG_VALIDATED = "config_validated"
    WARMUP = "warmup"
    CONNECTING_MARKET_DATA = "connecting_market_data"
    MARKET_DATA_LIVE = "market_data_live"
    ORDER_BOOK_SYNCED = "order_book_synced"
    SIGNALING_ACTIVE = "signaling_active"
    EXECUTION_ENABLED = "execution_enabled"
    IN_POSITION = "in_position"
    SAFE_MODE = "safe_mode"
    CIRCUIT_BREAKER_ACTIVE = "circuit_breaker_active"
    MANUAL_KILL_ACTIVE = "manual_kill_active"
    SHUTDOWN = "shutdown"


_BLOCKING_STATES = frozenset({
    OperationalState.SAFE_MODE,
    OperationalState.CIRCUIT_BREAKER_ACTIVE,
    OperationalState.MANUAL_KILL_ACTIVE,
    OperationalState.SHUTDOWN,
})

_TRADE_READY_STATES = frozenset({
    OperationalState.EXECUTION_ENABLED,
    OperationalState.IN_POSITION,
})

# Forward boot-path edges. Any state may also enter {SAFE_MODE,
# CIRCUIT_BREAKER_ACTIVE, MANUAL_KILL_ACTIVE, SHUTDOWN}, so we union those
# in below.
_FORWARD_EDGES: dict[OperationalState, set[OperationalState]] = {
    OperationalState.INIT: {OperationalState.CONFIG_VALIDATED},
    OperationalState.CONFIG_VALIDATED: {OperationalState.WARMUP},
    OperationalState.WARMUP: {OperationalState.CONNECTING_MARKET_DATA},
    OperationalState.CONNECTING_MARKET_DATA: {OperationalState.MARKET_DATA_LIVE},
    OperationalState.MARKET_DATA_LIVE: {OperationalState.ORDER_BOOK_SYNCED, OperationalState.SIGNALING_ACTIVE},
    OperationalState.ORDER_BOOK_SYNCED: {OperationalState.SIGNALING_ACTIVE},
    OperationalState.SIGNALING_ACTIVE: {OperationalState.EXECUTION_ENABLED},
    OperationalState.EXECUTION_ENABLED: {OperationalState.IN_POSITION},
    OperationalState.IN_POSITION: {OperationalState.EXECUTION_ENABLED},
    OperationalState.SAFE_MODE: {OperationalState.EXECUTION_ENABLED, OperationalState.IN_POSITION},
    OperationalState.CIRCUIT_BREAKER_ACTIVE: {OperationalState.EXECUTION_ENABLED},
    OperationalState.MANUAL_KILL_ACTIVE: {OperationalState.EXECUTION_ENABLED},
    OperationalState.SHUTDOWN: set(),  # terminal
}

# All non-terminal states may enter the blocking states.
_ALWAYS_REACHABLE: set[OperationalState] = {
    OperationalState.SAFE_MODE,
    OperationalState.CIRCUIT_BREAKER_ACTIVE,
    OperationalState.MANUAL_KILL_ACTIVE,
    OperationalState.SHUTDOWN,
}


def _allowed_targets(src: OperationalState) -> set[OperationalState]:
    if src is OperationalState.SHUTDOWN:
        return set()
    return _FORWARD_EDGES.get(src, set()) | _ALWAYS_REACHABLE


@dataclass
class StateTransition:
    from_state: OperationalState
    to_state: OperationalState
    reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


class IllegalTransition(RuntimeError):
    """Raised when transition_to() is called with an edge not in the allowed map."""


class OperationalStateMachine:
    """Single-writer FSM. Construct once at startup; pass-by-reference everywhere.

    Not internally locked — callers must serialise transitions (the boot
    sequence is single-threaded; runtime transitions are emitted from
    SafeModeManager / CircuitBreaker / risk_manager hooks all of which run
    on the asyncio event loop).
    """

    def __init__(
        self,
        history_max: int = 200,
        emit_callback: Callable[[StateTransition], None] | None = None,
    ) -> None:
        self._state = OperationalState.INIT
        self._entered_at = time.time()
        self._history: deque[StateTransition] = deque(maxlen=int(history_max))
        self._emit = emit_callback

    # ── Read API ──────────────────────────────────────────────────────────
    @property
    def current(self) -> OperationalState:
        return self._state

    @property
    def is_blocking(self) -> bool:
        return self._state in _BLOCKING_STATES

    @property
    def is_trade_ready(self) -> bool:
        """True when execution_enabled or in_position — safe to send orders."""
        return self._state in _TRADE_READY_STATES

    def history(self) -> list[StateTransition]:
        return list(self._history)

    def snapshot(self) -> dict[str, Any]:
        """REQ-STATE-001: dashboard-friendly state summary."""
        now = time.time()
        return {
            "current": self._state.value,
            "is_blocking": self.is_blocking,
            "is_trade_ready": self.is_trade_ready,
            "entered_at": self._entered_at,
            "seconds_in_state": round(now - self._entered_at, 2),
            "allowed_next": sorted(s.value for s in _allowed_targets(self._state)),
            "history": [t.to_dict() for t in list(self._history)[-30:]],
        }

    # ── Write API ─────────────────────────────────────────────────────────
    def transition_to(
        self,
        target: OperationalState,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        if target == self._state:
            # No-op: same state. Don't pollute history.
            return StateTransition(
                from_state=self._state, to_state=self._state,
                reason=f"noop:{reason}", timestamp=time.time(),
                metadata=metadata or {},
            )
        allowed = _allowed_targets(self._state)
        if target not in allowed:
            raise IllegalTransition(
                f"{self._state.value} → {target.value} not in allowed set "
                f"{sorted(s.value for s in allowed)}"
            )
        now = time.time()
        transition = StateTransition(
            from_state=self._state,
            to_state=target,
            reason=reason,
            timestamp=now,
            metadata=metadata or {},
        )
        self._history.append(transition)
        prev = self._state
        self._state = target
        self._entered_at = now
        logger.info(
            "FSM: {} → {} (reason={})",
            prev.value, target.value, reason,
        )
        if self._emit is not None:
            try:
                self._emit(transition)
            except Exception as exc:
                logger.warning("FSM emit callback failed: {}", exc)
        return transition

    def force_to(
        self,
        target: OperationalState,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Bypass the allowed-edges check. Reserved for safety primitives
        (SafeMode/CircuitBreaker/Kill) that must be reachable from any
        state. Still records the transition with reason='forced:…'."""
        if target == self._state:
            return StateTransition(
                from_state=self._state, to_state=self._state,
                reason=f"noop:{reason}", timestamp=time.time(),
                metadata=metadata or {},
            )
        now = time.time()
        transition = StateTransition(
            from_state=self._state,
            to_state=target,
            reason=f"forced:{reason}",
            timestamp=now,
            metadata=metadata or {},
        )
        self._history.append(transition)
        prev = self._state
        self._state = target
        self._entered_at = now
        logger.warning(
            "FSM (forced): {} → {} (reason={})",
            prev.value, target.value, reason,
        )
        if self._emit is not None:
            try:
                self._emit(transition)
            except Exception as exc:
                logger.warning("FSM emit callback failed: {}", exc)
        return transition


__all__ = [
    "OperationalState",
    "OperationalStateMachine",
    "StateTransition",
    "IllegalTransition",
]
