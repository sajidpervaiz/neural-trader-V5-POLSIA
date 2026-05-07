"""Event-driven audit persistence — subscribes to EventBus and records
every trading-relevant event to PostgreSQL via AuditRepository.

This module is the glue between the EventBus (in-memory pub/sub) and
the durable audit trail in PostgreSQL.  It captures:

- SIGNAL events → signal_events table
- RISK_BLOCK / RISK_DECISION events → risk_decisions table
- USER_ORDER_UPDATE / USER_ACCOUNT_UPDATE → user_stream_events table
- RECONCILIATION_COMPLETE → reconciliation_events table
- EQUITY_SNAPSHOT → pnl_snapshots table
- ERROR events → errors table (enhanced with stack trace)
"""
from __future__ import annotations

import traceback
from typing import Any

from loguru import logger

from core.event_bus import EventBus
from storage.audit_repository import AuditRepository


class AuditEventPersistence:
    """Wires EventBus → AuditRepository for full audit trail."""

    def __init__(self, audit_repo: AuditRepository, event_bus: EventBus) -> None:
        self._repo = audit_repo
        self._event_bus = event_bus

    def subscribe_all(self) -> None:
        """Subscribe to all audit-relevant events."""
        self._event_bus.subscribe("SIGNAL", self._on_signal)
        self._event_bus.subscribe("RISK_BLOCK", self._on_risk_block)
        self._event_bus.subscribe("RISK_DECISION", self._on_risk_decision)
        self._event_bus.subscribe("USER_ORDER_UPDATE", self._on_user_stream_event)
        self._event_bus.subscribe("USER_ACCOUNT_UPDATE", self._on_user_stream_event)
        self._event_bus.subscribe("FILL_CONFIRMED", self._on_user_stream_event)
        self._event_bus.subscribe("RECONCILIATION_COMPLETE", self._on_reconciliation)
        self._event_bus.subscribe("EQUITY_SNAPSHOT", self._on_equity_snapshot)
        self._event_bus.subscribe("PNL_UPDATE", self._on_pnl_update)
        self._event_bus.subscribe("COMPONENT_ERROR", self._on_error)
        # Operator-visible alerts — were previously dead-letters (no subscriber).
        # Persisted into the `errors` table so they survive bot restarts.
        self._event_bus.subscribe("ALERT_WARNING", self._on_alert)
        self._event_bus.subscribe("ALERT_CRITICAL", self._on_alert)
        logger.info("AuditEventPersistence subscribed to all audit events")

    # ── Signal ────────────────────────────────────────────────────────────

    async def _on_signal(self, payload: Any) -> None:
        try:
            if hasattr(payload, "exchange"):
                # TradingSignal dataclass
                await self._repo.persist_signal(
                    exchange=payload.exchange,
                    symbol=payload.symbol,
                    direction=payload.direction,
                    score=payload.score,
                    technical_score=payload.technical_score,
                    ml_score=payload.ml_score,
                    sentiment_score=payload.sentiment_score,
                    macro_score=payload.macro_score,
                    news_score=payload.news_score,
                    orderbook_score=payload.orderbook_score,
                    regime=payload.regime,
                    regime_confidence=payload.regime_confidence,
                    price=payload.price,
                    atr=payload.atr,
                    stop_loss=payload.stop_loss,
                    take_profit=payload.take_profit,
                    factors_active=getattr(payload, "factor_count", 0),
                    correlation_id=payload.metadata.get("correlation_id") if hasattr(payload, "metadata") and isinstance(payload.metadata, dict) else None,
                    metadata=payload.metadata if hasattr(payload, "metadata") else None,
                )
            elif isinstance(payload, dict):
                await self._repo.persist_signal(
                    exchange=payload.get("exchange", ""),
                    symbol=payload.get("symbol", ""),
                    direction=payload.get("direction", ""),
                    score=float(payload.get("score", 0)),
                    price=float(payload.get("price", 0)),
                    correlation_id=payload.get("correlation_id"),
                )
        except Exception as exc:
            logger.debug("_on_signal audit error: {}", exc)

    # ── Risk decisions ────────────────────────────────────────────────────

    async def _on_risk_block(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                await self._repo.persist_risk_decision(
                    exchange=payload.get("exchange", ""),
                    symbol=payload.get("symbol", ""),
                    decision="blocked",
                    reason=payload.get("reason", "unknown"),
                    signal_score=float(payload.get("signal_score", 0)),
                    signal_direction=payload.get("signal_direction", ""),
                    portfolio_heat=float(payload.get("portfolio_heat", 0)),
                    drawdown_pct=float(payload.get("drawdown_pct", 0)),
                    correlation_id=payload.get("correlation_id"),
                    metadata=payload.get("metadata"),
                )
            elif hasattr(payload, "symbol"):
                await self._repo.persist_risk_decision(
                    exchange=getattr(payload, "exchange", ""),
                    symbol=payload.symbol,
                    decision="blocked",
                    reason=getattr(payload, "reason", "unknown"),
                    signal_score=float(getattr(payload, "signal_score", 0)),
                    signal_direction=getattr(payload, "signal_direction", ""),
                    correlation_id=getattr(payload, "correlation_id", None),
                )
        except Exception as exc:
            logger.debug("_on_risk_block audit error: {}", exc)

    async def _on_risk_decision(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                await self._repo.persist_risk_decision(
                    exchange=payload.get("exchange", ""),
                    symbol=payload.get("symbol", ""),
                    decision=payload.get("decision", "unknown"),
                    reason=payload.get("reason", ""),
                    signal_score=float(payload.get("signal_score", 0)),
                    signal_direction=payload.get("signal_direction", ""),
                    portfolio_heat=float(payload.get("portfolio_heat", 0)),
                    drawdown_pct=float(payload.get("drawdown_pct", 0)),
                    correlation_id=payload.get("correlation_id"),
                    metadata=payload.get("metadata"),
                )
        except Exception as exc:
            logger.debug("_on_risk_decision audit error: {}", exc)

    # ── User stream events ────────────────────────────────────────────────

    async def _on_user_stream_event(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                event_type = payload.get("event_type", payload.get("e", "unknown"))
                await self._repo.persist_user_stream_event(
                    event_type=str(event_type),
                    raw_payload=payload,
                    correlation_id=payload.get("correlation_id"),
                )
        except Exception as exc:
            logger.debug("_on_user_stream_event audit error: {}", exc)

    # ── Reconciliation ────────────────────────────────────────────────────

    async def _on_reconciliation(self, payload: Any) -> None:
        try:
            if hasattr(payload, "reconciliation_id"):
                await self._repo.persist_reconciliation(
                    reconciliation_id=payload.reconciliation_id,
                    success=payload.success,
                    safe_mode=payload.safe_mode,
                    exchange_positions=len(payload.exchange_positions),
                    db_positions=len(getattr(payload, "db_positions", [])),
                    open_orders=len(payload.exchange_open_orders),
                    mismatches=payload.mismatches,
                    positions_without_sl=payload.positions_without_sl,
                    actions_taken=payload.actions_taken,
                    balance=payload.balance,
                    leverage_settings=getattr(payload, "leverage_settings", {}),
                )
            elif isinstance(payload, dict):
                await self._repo.persist_reconciliation(
                    reconciliation_id=payload.get("reconciliation_id", ""),
                    success=payload.get("success", False),
                    safe_mode=payload.get("safe_mode", False),
                    mismatches=payload.get("mismatches", []),
                )
        except Exception as exc:
            logger.debug("_on_reconciliation audit error: {}", exc)

    # ── PnL / Equity ─────────────────────────────────────────────────────

    async def _on_equity_snapshot(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                await self._repo.persist_pnl_snapshot(
                    exchange=payload.get("exchange", "all"),
                    realized_pnl=float(payload.get("realized_pnl", 0)),
                    unrealized_pnl=float(payload.get("unrealized_pnl", 0)),
                    cumulative_pnl=float(payload.get("cumulative_pnl", 0)),
                    equity=float(payload.get("equity", 0)),
                    drawdown_pct=float(payload.get("drawdown_pct", 0)),
                )
        except Exception as exc:
            logger.debug("_on_equity_snapshot audit error: {}", exc)

    async def _on_pnl_update(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                await self._repo.persist_pnl_snapshot(
                    exchange=payload.get("exchange", "all"),
                    symbol=payload.get("symbol"),
                    realized_pnl=float(payload.get("realized_pnl", 0)),
                    unrealized_pnl=float(payload.get("unrealized_pnl", 0)),
                    equity=float(payload.get("equity", 0)),
                )
        except Exception as exc:
            logger.debug("_on_pnl_update audit error: {}", exc)

    # ── Errors ────────────────────────────────────────────────────────────

    async def _on_error(self, payload: Any) -> None:
        try:
            if isinstance(payload, dict):
                await self._repo.persist_error(
                    component=payload.get("component", "unknown"),
                    error_type=payload.get("error_type", "unknown"),
                    message=payload.get("message", ""),
                    correlation_id=payload.get("correlation_id"),
                    stack_trace=payload.get("stack_trace"),
                    metadata=payload.get("metadata"),
                )
            elif isinstance(payload, Exception):
                await self._repo.persist_error(
                    component="unknown",
                    error_type=type(payload).__name__,
                    message=str(payload),
                    stack_trace=traceback.format_exc(),
                )
        except Exception as exc:
            logger.debug("_on_error audit error: {}", exc)

    # ── Operator alerts (ALERT_WARNING / ALERT_CRITICAL) ─────────────────
    # Routed into the errors table so they're durable across bot restarts.
    # Distinguished from genuine COMPONENT_ERROR by error_type prefix "alert.".

    async def _on_alert(self, payload: Any) -> None:
        try:
            if not isinstance(payload, dict):
                return
            alert_type = str(payload.get("type", "unknown"))
            symbol = payload.get("symbol", "")
            error = payload.get("error", "")
            needs_ack = bool(payload.get("needs_ack", False))
            # Build a single-line message from the alert payload.
            parts = [f"[{alert_type}]"]
            if symbol: parts.append(f"symbol={symbol}")
            if error: parts.append(f"error={error}")
            if needs_ack: parts.append("NEEDS_ACK")
            message = " ".join(parts)
            await self._repo.persist_error(
                component="alert",
                error_type=f"alert.{alert_type}",
                message=message,
                metadata={k: v for k, v in payload.items() if k not in {"type", "symbol", "error"}},
            )
        except Exception as exc:
            logger.debug("_on_alert audit error: {}", exc)
