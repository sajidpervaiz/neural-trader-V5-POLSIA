"""Async repository layer for the PostgreSQL audit trail.

All methods accept a raw ``asyncpg`` pool and execute idempotent
INSERT / UPSERT operations.  Every write includes an optional
``correlation_id`` so that every event in a signal-to-fill chain
can be traced end-to-end.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

from loguru import logger


_now = datetime.datetime.now
_UTC = datetime.UTC


class AuditRepository:
    """Thin async wrapper around asyncpg for the audit-trail tables."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _json(obj: Any) -> str:
        return json.dumps(obj, default=str) if obj else "{}"

    # ── signal_events ─────────────────────────────────────────────────────

    async def persist_signal(
        self,
        *,
        exchange: str,
        symbol: str,
        direction: str,
        score: float,
        technical_score: float = 0.0,
        ml_score: float = 0.0,
        sentiment_score: float = 0.0,
        macro_score: float = 0.0,
        news_score: float = 0.0,
        orderbook_score: float = 0.0,
        regime: str = "",
        regime_confidence: float = 0.0,
        price: float = 0.0,
        atr: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        factors_active: int = 0,
        acted_on: bool = False,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """Insert a signal event row. Returns the generated ``id``."""
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO signal_events (
                        time, correlation_id, exchange, symbol, direction, score,
                        technical_score, ml_score, sentiment_score, macro_score,
                        news_score, orderbook_score, regime, regime_confidence,
                        price, atr, stop_loss, take_profit,
                        factors_active, acted_on, metadata
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                        $15,$16,$17,$18,$19,$20,$21
                    ) RETURNING id
                    """,
                    _now(_UTC), correlation_id, exchange, symbol, direction, score,
                    technical_score, ml_score, sentiment_score, macro_score,
                    news_score, orderbook_score, regime, regime_confidence,
                    price, atr, stop_loss, take_profit,
                    factors_active, acted_on, self._json(metadata),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_signal failed: {}", exc)
            return None

    # ── risk_decisions ────────────────────────────────────────────────────

    async def persist_risk_decision(
        self,
        *,
        exchange: str,
        symbol: str,
        decision: str,
        reason: str,
        signal_score: float = 0.0,
        signal_direction: str = "",
        portfolio_heat: float = 0.0,
        drawdown_pct: float = 0.0,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO risk_decisions (
                        time, correlation_id, exchange, symbol, decision, reason,
                        signal_score, signal_direction, portfolio_heat, drawdown_pct, metadata
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    RETURNING id
                    """,
                    _now(_UTC), correlation_id, exchange, symbol, decision, reason,
                    signal_score, signal_direction, portfolio_heat, drawdown_pct,
                    self._json(metadata),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_risk_decision failed: {}", exc)
            return None

    # ── user_stream_events ────────────────────────────────────────────────

    async def persist_user_stream_event(
        self,
        *,
        event_type: str,
        raw_payload: dict,
        exchange: str = "binance",
        correlation_id: str | None = None,
    ) -> int | None:
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO user_stream_events (time, event_type, exchange, correlation_id, raw_payload)
                    VALUES ($1,$2,$3,$4,$5)
                    RETURNING id
                    """,
                    _now(_UTC), event_type, exchange, correlation_id,
                    self._json(raw_payload),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_user_stream_event failed: {}", exc)
            return None

    # ── reconciliation_events ─────────────────────────────────────────────

    async def persist_reconciliation(
        self,
        *,
        reconciliation_id: str,
        success: bool,
        safe_mode: bool = False,
        exchange_positions: int = 0,
        db_positions: int = 0,
        open_orders: int = 0,
        mismatches: list | None = None,
        positions_without_sl: list | None = None,
        actions_taken: list | None = None,
        balance: float = 0.0,
        leverage_settings: dict | None = None,
    ) -> int | None:
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO reconciliation_events (
                        reconciliation_id, time, success, safe_mode,
                        exchange_positions, db_positions, open_orders,
                        mismatches, positions_without_sl, actions_taken,
                        balance, leverage_settings
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    ON CONFLICT (reconciliation_id) DO NOTHING
                    RETURNING id
                    """,
                    reconciliation_id, _now(_UTC), success, safe_mode,
                    exchange_positions, db_positions, open_orders,
                    self._json(mismatches or []),
                    self._json(positions_without_sl or []),
                    self._json(actions_taken or []),
                    balance,
                    self._json(leverage_settings or {}),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_reconciliation failed: {}", exc)
            return None

    # ── pnl_snapshots ─────────────────────────────────────────────────────

    async def persist_pnl_snapshot(
        self,
        *,
        exchange: str,
        symbol: str | None = None,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        cumulative_pnl: float = 0.0,
        equity: float = 0.0,
        drawdown_pct: float = 0.0,
        metadata: dict | None = None,
    ) -> int | None:
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO pnl_snapshots (
                        time, exchange, symbol, realized_pnl, unrealized_pnl,
                        cumulative_pnl, equity, drawdown_pct, metadata
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    RETURNING id
                    """,
                    _now(_UTC), exchange, symbol, realized_pnl, unrealized_pnl,
                    cumulative_pnl, equity, drawdown_pct, self._json(metadata),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_pnl_snapshot failed: {}", exc)
            return None

    # ── errors (enhanced with stack_trace + correlation_id) ───────────────

    async def persist_error(
        self,
        *,
        component: str,
        error_type: str,
        message: str,
        correlation_id: str | None = None,
        stack_trace: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO errors (time, component, error_type, message,
                                        correlation_id, stack_trace, metadata)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    RETURNING id
                    """,
                    _now(_UTC), component, error_type, message,
                    correlation_id, stack_trace, self._json(metadata),
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error("persist_error (meta) failed: {}", exc)
            return None

    # ── Query helpers for recovery ────────────────────────────────────────

    async def load_open_orders(self) -> list[dict]:
        """Load all orders with active status for OrderManager rebuild."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM orders
                    WHERE status IN ('pending', 'submitted', 'open', 'partially_filled', 'new')
                    ORDER BY created_at
                    """
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("load_open_orders failed: {}", exc)
            return []

    async def load_all_orders(self, since: datetime.datetime | None = None, limit: int = 1000) -> list[dict]:
        """Load orders for state rebuild."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                if since:
                    rows = await conn.fetch(
                        "SELECT * FROM orders WHERE created_at >= $1 ORDER BY created_at LIMIT $2",
                        since, limit,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT * FROM orders ORDER BY created_at DESC LIMIT $1", limit,
                    )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("load_all_orders failed: {}", exc)
            return []

    async def load_fills_for_order(self, order_id: str) -> list[dict]:
        """Load all fills for a specific order."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM fills WHERE order_id = $1 ORDER BY trade_time", order_id,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("load_fills_for_order failed: {}", exc)
            return []

    async def load_fills_for_orders(self, order_ids: list[str]) -> dict[str, list[dict]]:
        """Batch-load fills for many orders in one query (closes the SR-1 N+1).

        Returns a dict {order_id: [fill_dict, ...]} so callers can group without
        a second pass. Empty input or no DB → empty dict.
        """
        if self._pool is None or not order_ids:
            return {}
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM fills WHERE order_id = ANY($1) ORDER BY trade_time",
                    list(order_ids),
                )
            grouped: dict[str, list[dict]] = {oid: [] for oid in order_ids}
            for r in rows:
                d = dict(r)
                oid = d.get("order_id", "")
                if oid in grouped:
                    grouped[oid].append(d)
            return grouped
        except Exception as exc:
            logger.error("load_fills_for_orders failed: {}", exc)
            return {}

    async def load_open_positions(self) -> list[dict]:
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE close_time IS NULL ORDER BY open_time"
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("load_open_positions failed: {}", exc)
            return []

    async def load_latest_equity(self) -> dict | None:
        """Load the most recent equity snapshot."""
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM equity_snapshots ORDER BY time DESC LIMIT 1"
                )
                return dict(row) if row else None
        except Exception as exc:
            logger.error("load_latest_equity failed: {}", exc)
            return None

    async def load_trace_by_correlation_id(self, correlation_id: str) -> dict[str, list[dict]]:
        """REQ-TR-001: assemble the end-to-end audit trail for a correlation_id.

        Returns a dict keyed by audit table with the rows that share the
        given correlation_id. Empty lists per key when no rows exist.
        """
        empty = {
            "signals": [],
            "risk_blocks": [],
            "orders": [],
            "fills": [],
            "user_stream_events": [],
            "pnl_snapshots": [],
            "reconciliation_events": [],
        }
        if self._pool is None or not correlation_id:
            return empty
        cid = str(correlation_id)
        result = dict(empty)
        queries = {
            "signals": "SELECT * FROM signal_events WHERE correlation_id = $1 ORDER BY time ASC",
            "risk_blocks": "SELECT * FROM risk_blocks WHERE correlation_id = $1 ORDER BY time ASC",
            "orders": "SELECT * FROM orders WHERE correlation_id = $1 ORDER BY time ASC",
            "fills": "SELECT * FROM fills WHERE correlation_id = $1 ORDER BY time ASC",
            "user_stream_events": "SELECT * FROM user_stream_events WHERE correlation_id = $1 ORDER BY time ASC",
            "pnl_snapshots": "SELECT * FROM pnl_snapshots WHERE correlation_id = $1 ORDER BY time ASC",
            "reconciliation_events": "SELECT * FROM reconciliation_events WHERE correlation_id = $1 ORDER BY time ASC",
        }
        try:
            async with self._pool.acquire() as conn:
                for key, sql in queries.items():
                    try:
                        rows = await conn.fetch(sql, cid)
                        result[key] = [dict(r) for r in rows]
                    except Exception as exc:
                        logger.debug("load_trace[{}] failed: {}", key, exc)
        except Exception as exc:
            logger.error("load_trace_by_correlation_id pool acquire failed: {}", exc)
        return result

    async def load_recent_signals(self, limit: int = 50) -> list[dict]:
        """Load recent signal events for dashboard / state rebuild."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM signal_events ORDER BY time DESC LIMIT $1", limit,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("load_recent_signals failed: {}", exc)
            return []

    async def load_last_reconciliation(self) -> dict | None:
        """Load the most recent reconciliation event."""
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM reconciliation_events ORDER BY time DESC LIMIT 1"
                )
                return dict(row) if row else None
        except Exception as exc:
            logger.error("load_last_reconciliation failed: {}", exc)
            return None
