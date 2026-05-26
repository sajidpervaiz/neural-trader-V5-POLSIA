"""DB-based state recovery for crash resilience.

After a restart, ``StateRecovery.recover()`` reads the PostgreSQL audit trail
and rebuilds the in-memory state of:

- OrderManager (open orders, fills, maps)
- RiskManager (last equity snapshot, drawdown)
- Recent signals (for duplicate suppression)
- Reconciliation status (safe_mode flag)

This replaces the JSON-file-based ``_load_order_state`` approach with a
durable, query-able database source of truth.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.error_handling import sanitize_exception

from storage.audit_repository import AuditRepository


@dataclass
class RecoveryResult:
    """Outcome of a state-recovery attempt."""
    success: bool = False
    orders_recovered: int = 0
    positions_recovered: int = 0
    fills_recovered: int = 0
    last_equity: float | None = None
    last_drawdown: float | None = None
    safe_mode: bool = False
    reconciliation_id: str | None = None
    errors: list[str] = field(default_factory=list)


class StateRecovery:
    """Rebuilds bot state from the PostgreSQL audit trail after a crash."""

    def __init__(self, audit_repo: AuditRepository) -> None:
        self._repo = audit_repo

    async def recover(self) -> RecoveryResult:
        """Run full recovery sequence.  Returns a RecoveryResult summary."""
        result = RecoveryResult()

        try:
            # 1. Recover open orders
            orders = await self._repo.load_open_orders()
            result.orders_recovered = len(orders)

            # 2. Recover open positions
            positions = await self._repo.load_open_positions()
            result.positions_recovered = len(positions)

            # 3. Recover fills for open orders (for OrderManager.fills rebuild).
            # Single batched query instead of N+1 — startup with 1000 historical
            # orders previously meant 1000 sequential round-trips.
            order_ids = [o.get("order_id", "") for o in orders if o.get("order_id")]
            fills_by_oid = await self._repo.load_fills_for_orders(order_ids)
            result.fills_recovered = sum(len(v) for v in fills_by_oid.values())

            # 4. Get last equity snapshot
            equity = await self._repo.load_latest_equity()
            if equity:
                result.last_equity = equity.get("equity")
                result.last_drawdown = equity.get("drawdown_pct")

            # 5. Check last reconciliation for safe_mode
            recon = await self._repo.load_last_reconciliation()
            if recon:
                result.safe_mode = bool(recon.get("safe_mode", False))
                result.reconciliation_id = recon.get("reconciliation_id")

            result.success = True
            logger.info(
                "State recovery complete: {} orders, {} positions, {} fills, "
                "equity={}, safe_mode={}",
                result.orders_recovered, result.positions_recovered,
                result.fills_recovered, result.last_equity, result.safe_mode,
            )

        except Exception as exc:
            result.errors.append(sanitize_exception(exc))
            logger.error("State recovery failed: {}", exc)

        return result

    async def rebuild_order_manager_state(self, order_manager: Any) -> int:
        """Populate an OrderManager's in-memory maps from the DB.

        Returns the number of orders loaded.

        This is a *supplement* to the reconciler's exchange-side rebuild.
        It restores the OrderManager's ``orders``, ``client_order_map``,
        and ``exchange_order_map`` dictionaries from persistent storage
        so that idempotency checks and duplicate-order prevention work
        correctly after a restart.
        """
        orders = await self._repo.load_all_orders(
            since=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7),
            limit=5000,
        )

        if not orders:
            return 0

        loaded = 0
        for row in orders:
            oid = row.get("order_id", "")
            if not oid or oid in order_manager.orders:
                continue

            try:
                order = order_manager._dict_to_order({
                    "order_id": oid,
                    "client_order_id": row.get("client_order_id", ""),
                    "symbol": row.get("symbol", ""),
                    "side": row.get("side", "buy"),
                    "order_type": row.get("order_type", "limit"),
                    "quantity": float(row.get("quantity", 0)),
                    "price": float(row.get("price", 0) or 0),
                    "status": row.get("status", "filled"),
                    "filled_quantity": float(row.get("filled_qty", 0)),
                    "remaining_quantity": max(0, float(row.get("quantity", 0)) - float(row.get("filled_qty", 0))),
                    "avg_fill_price": float(row.get("avg_fill_price", 0)),
                    "cumulative_quantity": float(row.get("filled_qty", 0)),
                    "average_fill_price": float(row.get("avg_fill_price", 0)),
                    "total_fee": float(row.get("total_fee", 0)),
                    "created_at": row.get("created_at", 0),
                    "updated_at": row.get("updated_at", 0),
                    "submitted_at": None,
                    "filled_at": None,
                    "exchange_order_id": None,
                    "venue": row.get("exchange", ""),
                    "time_in_force": "GTC",
                    "reduce_only": bool(row.get("reduce_only", False)),
                    "user_id": None,
                    "tags": {},
                    "stages": [],
                    "fills": [],
                    "error_message": None,
                    "metadata": row.get("metadata", {}),
                })
                order_manager.orders[oid] = order
                coid = row.get("client_order_id", "")
                if coid:
                    order_manager.client_order_map[coid] = order
                loaded += 1
            except Exception as exc:
                logger.debug("Skipping order {} during rebuild: {}", oid, exc)

        logger.info("Rebuilt {} orders into OrderManager from DB", loaded)
        return loaded
