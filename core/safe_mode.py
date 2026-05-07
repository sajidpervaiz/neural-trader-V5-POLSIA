"""
Centralized Safe Mode manager for disaster resilience.

When SafeMode is active:
- No new trades are opened.
- Existing positions remain protected by exchange-side SL/TP.
- The system logs the reason and timestamp of activation.
- SafeMode can only be cleared explicitly after the fault resolves.

Triggers:
- WebSocket disconnect (market data blind)
- USER_DATA stream failure (fill/cancel blind)
- Exchange API timeout
- DB outage
- Circuit breaker trip
- Manual activation
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class SafeModeReason(str, Enum):
    WEBSOCKET_DISCONNECT = "websocket_disconnect"
    USER_STREAM_LOST = "user_stream_lost"
    EXCHANGE_API_TIMEOUT = "exchange_api_timeout"
    DB_OUTAGE = "db_outage"
    CIRCUIT_BREAKER = "circuit_breaker"
    EXTREME_VOLATILITY = "extreme_volatility"
    EXTREME_SPREAD = "extreme_spread"
    ORDER_REJECTION_BURST = "order_rejection_burst"
    PARTIAL_FILL_STUCK = "partial_fill_stuck"
    MANUAL = "manual"


@dataclass
class SafeModeEvent:
    reason: SafeModeReason
    timestamp: float
    detail: str = ""


class SafeModeManager:
    """Thread-safe safe mode controller.  Multiple reasons can be active
    simultaneously — safe mode is ON as long as *any* reason remains."""

    def __init__(self, event_bus: Any = None) -> None:
        self._active_reasons: dict[SafeModeReason, SafeModeEvent] = {}
        self._history: list[dict[str, Any]] = []
        # Optional event bus for audit emission. Activate/deactivate/clear_all
        # publish ALERT_WARNING (or ALERT_CRITICAL on full activation) so the
        # existing audit pipeline persists them to the errors table.
        self._event_bus = event_bus

    # ── Queries ───────────────────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return len(self._active_reasons) > 0

    @property
    def active_reasons(self) -> list[SafeModeReason]:
        return list(self._active_reasons.keys())

    @property
    def active_events(self) -> list[SafeModeEvent]:
        return list(self._active_reasons.values())

    def _emit(self, alert_level: str, payload: dict[str, Any]) -> None:
        """Emit safe-mode state-change event onto the bus (if configured)."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish_nowait(alert_level, payload)
        except Exception as exc:
            logger.debug("SafeMode emit {} failed: {}", alert_level, exc)

    # ── Mutations ─────────────────────────────────────────────────────────
    def activate(self, reason: SafeModeReason, detail: str = "") -> None:
        """Activate safe mode for *reason*.  Idempotent — re-activation updates detail."""
        was_active = self.is_active
        event = SafeModeEvent(reason=reason, timestamp=time.time(), detail=detail)
        self._active_reasons[reason] = event
        self._history.append({
            "action": "activate",
            "reason": reason.value,
            "detail": detail,
            "timestamp": event.timestamp,
        })
        if not was_active:
            logger.critical("SAFE MODE ACTIVATED — reason: {} ({})", reason.value, detail)
            self._emit("ALERT_CRITICAL", {
                "type": f"safe_mode_activated_{reason.value}",
                "reason": reason.value,
                "detail": detail,
                "needs_ack": True,
                "ts": event.timestamp,
            })
        else:
            logger.warning("Safe mode reason added: {} ({})", reason.value, detail)
            self._emit("ALERT_WARNING", {
                "type": f"safe_mode_reason_added_{reason.value}",
                "reason": reason.value,
                "detail": detail,
                "ts": event.timestamp,
            })

    def deactivate(self, reason: SafeModeReason) -> bool:
        """Clear a single reason.  Returns True if safe mode is now fully clear."""
        removed = self._active_reasons.pop(reason, None)
        if removed:
            self._history.append({
                "action": "deactivate",
                "reason": reason.value,
                "timestamp": time.time(),
            })
            logger.info("Safe mode reason cleared: {}", reason.value)
            self._emit("ALERT_WARNING", {
                "type": f"safe_mode_reason_cleared_{reason.value}",
                "reason": reason.value,
                "ts": time.time(),
            })
        if not self.is_active:
            logger.info("SAFE MODE FULLY CLEARED — normal operation resumed")
            self._emit("ALERT_WARNING", {
                "type": "safe_mode_fully_cleared",
                "ts": time.time(),
            })
            return True
        return False

    def clear_all(self, operator: str = "system") -> None:
        """Force-clear all reasons (admin override)."""
        prior_reasons = [r.value for r in self._active_reasons.keys()]
        self._active_reasons.clear()
        self._history.append({
            "action": "clear_all",
            "operator": operator,
            "prior_reasons": prior_reasons,
            "timestamp": time.time(),
        })
        logger.warning("SAFE MODE force-cleared (all reasons) by {}; prior={}", operator, prior_reasons)
        self._emit("ALERT_CRITICAL", {
            "type": "safe_mode_force_cleared",
            "operator": operator,
            "prior_reasons": prior_reasons,
            "needs_ack": True,
            "ts": time.time(),
        })

    def get_status(self) -> dict[str, Any]:
        return {
            "safe_mode_active": self.is_active,
            "active_reasons": [
                {"reason": r.value, "detail": e.detail, "since": e.timestamp}
                for r, e in self._active_reasons.items()
            ],
            "history_count": len(self._history),
        }

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)
