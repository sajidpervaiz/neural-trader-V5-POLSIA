"""
Emergency Stop (E-Stop) System — V6 Specification §13.6

Triggered by:
  • Risk engine (daily loss, drawdown, manual command)
  • Flash crash detection (price move > 3% in 60 sec)
  • Manual API endpoint

Actions on E-Stop:
  1. Set global halt flag (blocks all new signal evaluation)
  2. Cancel all open orders via execution engine
  3. Close all open positions via market orders (aggressive)
  4. Halt signal generation
  5. Log immutable audit record

The EStop class is a singleton-style component that integrates with
EventBus and any executor implementing cancel_order / close_position.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class EStopEvent:
    trigger: str           # "manual", "daily_loss", "drawdown", "flash_crash", "consecutive_losses"
    triggered_at: float    # unix timestamp
    triggered_by: str      # component or username that triggered
    positions_closed: int = 0
    orders_cancelled: int = 0
    resolved_at: float | None = None
    resolved_by: str = ""
    notes: str = ""


class EStopManager:
    """Global emergency stop manager.

    Usage:
      estop = EStopManager(event_bus)
      await estop.trigger("manual", "dashboard_user")

      # In signal pipeline:
      if estop.is_active:
          return  # halt — do not generate signals
    """

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus
        self._active = False
        self._reason = ""
        self._triggered_at: float = 0.0
        self._history: list[EStopEvent] = []
        self._lock = asyncio.Lock()

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def triggered_at(self) -> float:
        return self._triggered_at

    # ── Trigger ───────────────────────────────────────────────────────────────

    async def trigger(
        self,
        trigger: str,
        triggered_by: str = "system",
        notes: str = "",
        executor: Any = None,
        positions: list[dict] | None = None,
    ) -> EStopEvent:
        """Activate the emergency stop.

        Args:
            trigger:      Reason string (e.g. "manual", "daily_loss")
            triggered_by: Who/what triggered this (component name or user)
            notes:        Optional extra context
            executor:     Executor instance with cancel_order / close_position
            positions:    List of open position dicts to close

        Returns:
            EStopEvent with the result of the stop sequence.
        """
        async with self._lock:
            if self._active:
                logger.warning("EStop: already active ({}), ignoring duplicate trigger: {}",
                               self._reason, trigger)
                return self._history[-1] if self._history else EStopEvent(trigger, time.time(), triggered_by)

            self._active = True
            self._reason = trigger
            self._triggered_at = time.time()

            event = EStopEvent(
                trigger=trigger,
                triggered_at=self._triggered_at,
                triggered_by=triggered_by,
                notes=notes,
            )

            logger.critical(
                "🛑 E-STOP TRIGGERED — trigger={} by={} notes={}",
                trigger, triggered_by, notes,
            )

            # Publish event for monitoring
            if self._event_bus is not None:
                try:
                    await self._event_bus.publish("ESTOP_TRIGGERED", {
                        "trigger": trigger,
                        "triggered_by": triggered_by,
                        "timestamp": self._triggered_at,
                        "notes": notes,
                    })
                except Exception as exc:
                    logger.error("EStop: event_bus publish error: {}", exc)

            # Execute stop sequence
            orders_cancelled = 0
            positions_closed = 0

            if executor is not None:
                # Cancel all open orders
                try:
                    orders_cancelled = await self._cancel_all_orders(executor)
                except Exception as exc:
                    logger.error("EStop: cancel_all_orders failed: {}", exc)

                # Close all positions via market orders
                try:
                    if positions:
                        positions_closed = await self._close_all_positions(executor, positions)
                except Exception as exc:
                    logger.error("EStop: close_all_positions failed: {}", exc)

            event.orders_cancelled = orders_cancelled
            event.positions_closed = positions_closed
            self._history.append(event)

            logger.critical(
                "🛑 E-STOP COMPLETE — cancelled={} orders, closed={} positions",
                orders_cancelled, positions_closed,
            )
            return event

    async def _cancel_all_orders(self, executor: Any) -> int:
        """Cancel all open orders via executor. Returns count cancelled."""
        count = 0
        # Try executor.get_open_orders if it exists
        get_orders = getattr(executor, "get_open_orders", None)
        cancel_order = getattr(executor, "cancel_order", None)
        if get_orders is None or cancel_order is None:
            return 0
        try:
            open_orders = await get_orders()
            for order in open_orders:
                oid = order.get("order_id") or order.get("id", "")
                sym = order.get("symbol", "")
                if oid and sym:
                    try:
                        await cancel_order(str(oid), sym)
                        count += 1
                    except Exception as exc:
                        logger.warning("EStop: cancel order {} failed: {}", oid, exc)
        except Exception as exc:
            logger.error("EStop: get_open_orders failed: {}", exc)
        return count

    async def _close_all_positions(self, executor: Any, positions: list[dict]) -> int:
        """Close all positions via market orders. Returns count closed."""
        count = 0
        close_pos = getattr(executor, "_close_position_live", None) or \
                    getattr(executor, "close_position", None)
        if close_pos is None:
            return 0
        for pos in positions:
            sym = pos.get("symbol", "")
            size = float(pos.get("size", 0))
            if sym and size > 0:
                try:
                    ok = await close_pos(sym, size)
                    if ok:
                        count += 1
                    else:
                        logger.warning("EStop: close_position {} returned False", sym)
                except Exception as exc:
                    logger.error("EStop: close_position {} failed: {}", sym, exc)
        return count

    # ── Release ───────────────────────────────────────────────────────────────

    async def release(self, released_by: str = "manual") -> bool:
        """Deactivate the E-Stop (manual override only).

        Returns True if was active and now released.
        """
        async with self._lock:
            if not self._active:
                return False
            self._active = False
            now = time.time()
            if self._history:
                self._history[-1].resolved_at = now
                self._history[-1].resolved_by = released_by

            logger.warning("EStop: RELEASED by={} after {:.0f}s",
                           released_by, now - self._triggered_at)

            if self._event_bus is not None:
                try:
                    await self._event_bus.publish("ESTOP_RELEASED", {
                        "released_by": released_by,
                        "timestamp": now,
                    })
                except Exception:
                    pass
            return True

    # ── Flash crash detector ───────────────────────────────────────────────────

    def check_flash_crash(
        self,
        price_history: list[tuple[float, float]],   # list of (timestamp, price)
        threshold_pct: float = 0.03,
        window_seconds: float = 60.0,
    ) -> bool:
        """Return True if a flash crash (>3% move in 60s) is detected.

        Intended to be called on each tick/candle update.
        Does NOT trigger E-Stop automatically — caller decides.
        """
        if len(price_history) < 2:
            return False
        now = time.time()
        cutoff = now - window_seconds
        recent = [(ts, px) for ts, px in price_history if ts >= cutoff]
        if len(recent) < 2:
            return False
        prices = [px for _, px in recent]
        pmin, pmax = min(prices), max(prices)
        if pmin <= 0:
            return False
        move_pct = (pmax - pmin) / pmin
        return move_pct >= threshold_pct

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "reason": self._reason,
            "triggered_at": self._triggered_at,
            "duration_seconds": time.time() - self._triggered_at if self._active else 0,
            "total_events": len(self._history),
            "history": [
                {
                    "trigger": e.trigger,
                    "triggered_at": e.triggered_at,
                    "triggered_by": e.triggered_by,
                    "positions_closed": e.positions_closed,
                    "orders_cancelled": e.orders_cancelled,
                    "resolved_at": e.resolved_at,
                    "resolved_by": e.resolved_by,
                    "notes": e.notes,
                }
                for e in self._history[-20:]  # last 20 events
            ],
        }
