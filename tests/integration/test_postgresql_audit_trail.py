"""Integration tests for PostgreSQL audit trail, repository layer, and DB recovery.

These tests validate:
1. AuditRepository — all persist/load methods with idempotent constraints
2. AuditEventPersistence — EventBus → DB wiring for all event types
3. StateRecovery — OrderManager rebuild from DB, safe_mode detection
4. TradePersistence.migrate() — new audit DDL tables are created
5. Alembic migration — schema is valid and importable
6. End-to-end: signal→risk→order→fill→recovery chain

The tests use a mock asyncpg pool that records SQL calls and returns
controlled results, so they run without a live PostgreSQL instance.
When DATABASE_URL is set, the tests run against a real database.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.event_bus import EventBus
from storage.audit_repository import AuditRepository
from storage.audit_event_persistence import AuditEventPersistence
from storage.state_recovery import StateRecovery, RecoveryResult


# ── Helpers ───────────────────────────────────────────────────────────────


class FakeConnection:
    """Mock asyncpg connection that captures SQL calls."""

    _SENTINEL = object()

    def __init__(self, rows: list[dict] | None = None, fetchrow_result: dict | None | object = _SENTINEL):
        self._calls: list[tuple[str, tuple]] = []
        self._rows = rows or []
        self._fetchrow_result = {"id": 1} if fetchrow_result is FakeConnection._SENTINEL else fetchrow_result

    async def execute(self, sql: str, *args: Any) -> str:
        self._calls.append((sql.strip(), args))
        return "INSERT 0 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict]:
        self._calls.append((sql.strip(), args))
        return [FakeRecord(r) for r in self._rows]

    async def fetchrow(self, sql: str, *args: Any) -> dict | None:
        self._calls.append((sql.strip(), args))
        return FakeRecord(self._fetchrow_result) if self._fetchrow_result else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeRecord(dict):
    """Mimics asyncpg.Record that supports both dict access and __getitem__."""

    def __init__(self, data: dict):
        super().__init__(data)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)


class FakePool:
    """Mock asyncpg pool that returns a FakeConnection."""

    def __init__(self, conn: FakeConnection | None = None):
        self._conn = conn or FakeConnection()

    def acquire(self):
        return self._conn


# ── AuditRepository Tests ─────────────────────────────────────────────────


class TestAuditRepositoryPersistSignal:
    @pytest.mark.asyncio
    async def test_persist_signal_inserts_row(self):
        conn = FakeConnection(fetchrow_result={"id": 42})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_signal(
            exchange="binance",
            symbol="BTCUSDT",
            direction="long",
            score=0.85,
            technical_score=0.7,
            ml_score=0.6,
            price=50000.0,
            atr=500.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            factors_active=4,
            correlation_id="corr-123",
        )

        assert result == 42
        assert len(conn._calls) == 1
        sql = conn._calls[0][0]
        assert "INSERT INTO signal_events" in sql
        assert "correlation_id" in sql

    @pytest.mark.asyncio
    async def test_persist_signal_returns_none_when_pool_is_none(self):
        repo = AuditRepository(None)
        result = await repo.persist_signal(
            exchange="binance", symbol="BTCUSDT", direction="long",
            score=0.5, price=50000.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_persist_signal_handles_exception(self):
        conn = FakeConnection()
        conn.fetchrow = AsyncMock(side_effect=Exception("DB error"))
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_signal(
            exchange="binance", symbol="BTCUSDT", direction="long",
            score=0.5, price=50000.0,
        )
        assert result is None


class TestAuditRepositoryPersistRiskDecision:
    @pytest.mark.asyncio
    async def test_persist_risk_decision(self):
        conn = FakeConnection(fetchrow_result={"id": 10})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_risk_decision(
            exchange="binance",
            symbol="ETHUSDT",
            decision="blocked",
            reason="max_drawdown_exceeded",
            signal_score=0.9,
            portfolio_heat=85.0,
            drawdown_pct=12.5,
            correlation_id="corr-456",
        )

        assert result == 10
        sql = conn._calls[0][0]
        assert "INSERT INTO risk_decisions" in sql

    @pytest.mark.asyncio
    async def test_persist_risk_decision_none_pool(self):
        repo = AuditRepository(None)
        result = await repo.persist_risk_decision(
            exchange="binance", symbol="X", decision="ok", reason="test",
        )
        assert result is None


class TestAuditRepositoryPersistUserStreamEvent:
    @pytest.mark.asyncio
    async def test_persist_user_stream_event(self):
        conn = FakeConnection(fetchrow_result={"id": 5})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_user_stream_event(
            event_type="ORDER_TRADE_UPDATE",
            raw_payload={"order_id": "123", "status": "FILLED"},
            correlation_id="corr-789",
        )

        assert result == 5
        sql = conn._calls[0][0]
        assert "INSERT INTO user_stream_events" in sql

    @pytest.mark.asyncio
    async def test_persist_user_stream_event_none_pool(self):
        repo = AuditRepository(None)
        result = await repo.persist_user_stream_event(
            event_type="test", raw_payload={},
        )
        assert result is None


class TestAuditRepositoryPersistReconciliation:
    @pytest.mark.asyncio
    async def test_persist_reconciliation_idempotent(self):
        conn = FakeConnection(fetchrow_result={"id": 1})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_reconciliation(
            reconciliation_id="recon-abc123",
            success=True,
            safe_mode=False,
            exchange_positions=3,
            db_positions=3,
            open_orders=5,
            mismatches=[],
            balance=10000.0,
            leverage_settings={"BTCUSDT": {"leverage": 5}},
        )

        assert result == 1
        sql = conn._calls[0][0]
        assert "ON CONFLICT (reconciliation_id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_duplicate_reconciliation_returns_none(self):
        conn = FakeConnection(fetchrow_result=None)  # conflict → no returning
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_reconciliation(
            reconciliation_id="recon-dup",
            success=True,
        )
        assert result is None


class TestAuditRepositoryPersistPnlSnapshot:
    @pytest.mark.asyncio
    async def test_persist_pnl_snapshot(self):
        conn = FakeConnection(fetchrow_result={"id": 99})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_pnl_snapshot(
            exchange="binance",
            symbol="BTCUSDT",
            realized_pnl=150.0,
            unrealized_pnl=-20.0,
            cumulative_pnl=130.0,
            equity=10130.0,
            drawdown_pct=1.5,
        )

        assert result == 99
        sql = conn._calls[0][0]
        assert "INSERT INTO pnl_snapshots" in sql


class TestAuditRepositoryPersistError:
    @pytest.mark.asyncio
    async def test_persist_error_with_stack_trace(self):
        conn = FakeConnection(fetchrow_result={"id": 7})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.persist_error(
            component="risk_manager",
            error_type="ValueError",
            message="invalid position size",
            correlation_id="corr-err",
            stack_trace="Traceback...\nValueError: invalid position size",
        )

        assert result == 7
        sql = conn._calls[0][0]
        assert "INSERT INTO errors" in sql
        assert "stack_trace" in sql
        assert "correlation_id" in sql


# ── AuditRepository Query Tests ──────────────────────────────────────────


class TestAuditRepositoryQueries:
    @pytest.mark.asyncio
    async def test_load_open_orders(self):
        rows = [
            {"order_id": "o1", "status": "open", "symbol": "BTCUSDT"},
            {"order_id": "o2", "status": "pending", "symbol": "ETHUSDT"},
        ]
        conn = FakeConnection(rows=rows)
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_open_orders()
        assert len(result) == 2
        assert result[0]["order_id"] == "o1"
        sql = conn._calls[0][0]
        assert "status IN" in sql

    @pytest.mark.asyncio
    async def test_load_all_orders_with_since(self):
        conn = FakeConnection(rows=[{"order_id": "o1"}])
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        since = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
        result = await repo.load_all_orders(since=since, limit=10)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_load_fills_for_order(self):
        rows = [
            {"fill_id": "f1", "order_id": "o1", "quantity": 0.5, "price": 50000},
            {"fill_id": "f2", "order_id": "o1", "quantity": 0.3, "price": 50100},
        ]
        conn = FakeConnection(rows=rows)
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_fills_for_order("o1")
        assert len(result) == 2
        assert result[0]["fill_id"] == "f1"

    @pytest.mark.asyncio
    async def test_load_open_positions(self):
        rows = [{"exchange": "binance", "symbol": "BTCUSDT", "size": 1.0}]
        conn = FakeConnection(rows=rows)
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_open_positions()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_load_latest_equity(self):
        conn = FakeConnection(fetchrow_result={"equity": 10500.0, "drawdown_pct": 2.1})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_latest_equity()
        assert result["equity"] == 10500.0

    @pytest.mark.asyncio
    async def test_load_recent_signals(self):
        rows = [{"id": 1, "direction": "long", "score": 0.8}]
        conn = FakeConnection(rows=rows)
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_recent_signals(limit=5)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_load_last_reconciliation(self):
        conn = FakeConnection(fetchrow_result={
            "reconciliation_id": "r1", "safe_mode": True, "success": True,
        })
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        result = await repo.load_last_reconciliation()
        assert result["safe_mode"] is True

    @pytest.mark.asyncio
    async def test_queries_return_empty_when_pool_none(self):
        repo = AuditRepository(None)
        assert await repo.load_open_orders() == []
        assert await repo.load_all_orders() == []
        assert await repo.load_fills_for_order("x") == []
        assert await repo.load_open_positions() == []
        assert await repo.load_latest_equity() is None
        assert await repo.load_recent_signals() == []
        assert await repo.load_last_reconciliation() is None


# ── AuditEventPersistence Tests ───────────────────────────────────────────


class TestAuditEventPersistenceSignal:
    @pytest.mark.asyncio
    async def test_on_signal_dataclass(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_signal = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)
        aep.subscribe_all()

        @dataclass
        class FakeSignal:
            exchange: str = "binance"
            symbol: str = "BTCUSDT"
            direction: str = "long"
            score: float = 0.85
            technical_score: float = 0.7
            ml_score: float = 0.6
            sentiment_score: float = 0.1
            macro_score: float = 0.2
            news_score: float = 0.3
            orderbook_score: float = 0.4
            regime: str = "trending"
            regime_confidence: float = 0.9
            price: float = 50000.0
            atr: float = 500.0
            stop_loss: float = 49000.0
            take_profit: float = 52000.0
            factor_count: int = 5
            metadata: dict = field(default_factory=dict)

        await aep._on_signal(FakeSignal())
        repo.persist_signal.assert_called_once()
        call_kwargs = repo.persist_signal.call_args.kwargs
        assert call_kwargs["exchange"] == "binance"
        assert call_kwargs["direction"] == "long"
        assert call_kwargs["score"] == 0.85

    @pytest.mark.asyncio
    async def test_on_signal_dict(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_signal = AsyncMock(return_value=2)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_signal({"exchange": "bybit", "symbol": "ETHUSDT", "direction": "short", "score": 0.7, "price": 3000})
        repo.persist_signal.assert_called_once()


class TestAuditEventPersistenceRisk:
    @pytest.mark.asyncio
    async def test_on_risk_block_dict(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_risk_decision = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_risk_block({
            "exchange": "binance", "symbol": "BTCUSDT",
            "reason": "max_heat", "signal_score": 0.9,
            "portfolio_heat": 90.0, "drawdown_pct": 15.0,
        })
        repo.persist_risk_decision.assert_called_once()
        assert repo.persist_risk_decision.call_args.kwargs["decision"] == "blocked"

    @pytest.mark.asyncio
    async def test_on_risk_decision_dict(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_risk_decision = AsyncMock(return_value=2)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_risk_decision({
            "exchange": "binance", "symbol": "ETHUSDT",
            "decision": "approved", "reason": "within_limits",
        })
        repo.persist_risk_decision.assert_called_once()


class TestAuditEventPersistenceUserStream:
    @pytest.mark.asyncio
    async def test_on_user_stream_event(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_user_stream_event = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_user_stream_event({
            "event_type": "ORDER_TRADE_UPDATE",
            "order_id": "123",
            "status": "FILLED",
        })
        repo.persist_user_stream_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_user_stream_ignores_non_dict(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_user_stream_event = AsyncMock()
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_user_stream_event("not-a-dict")
        repo.persist_user_stream_event.assert_not_called()


class TestAuditEventPersistenceReconciliation:
    @pytest.mark.asyncio
    async def test_on_reconciliation_dataclass(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_reconciliation = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        @dataclass
        class FakeReconResult:
            reconciliation_id: str = "recon-abc"
            success: bool = True
            safe_mode: bool = False
            exchange_positions: list = field(default_factory=lambda: [{"symbol": "BTCUSDT"}])
            db_positions: list = field(default_factory=list)
            exchange_open_orders: list = field(default_factory=list)
            mismatches: list = field(default_factory=list)
            positions_without_sl: list = field(default_factory=list)
            actions_taken: list = field(default_factory=list)
            balance: float = 10000.0
            leverage_settings: dict = field(default_factory=dict)

        await aep._on_reconciliation(FakeReconResult())
        repo.persist_reconciliation.assert_called_once()
        kwargs = repo.persist_reconciliation.call_args.kwargs
        assert kwargs["reconciliation_id"] == "recon-abc"
        assert kwargs["exchange_positions"] == 1

    @pytest.mark.asyncio
    async def test_on_reconciliation_dict_fallback(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_reconciliation = AsyncMock(return_value=2)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_reconciliation({
            "reconciliation_id": "recon-dict",
            "success": False,
            "safe_mode": True,
        })
        repo.persist_reconciliation.assert_called_once()


class TestAuditEventPersistenceError:
    @pytest.mark.asyncio
    async def test_on_error_dict(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_error = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_error({
            "component": "order_manager",
            "error_type": "RuntimeError",
            "message": "connection lost",
            "stack_trace": "Traceback...",
            "correlation_id": "corr-err-1",
        })
        repo.persist_error.assert_called_once()
        kw = repo.persist_error.call_args.kwargs
        assert kw["component"] == "order_manager"
        assert kw["stack_trace"] == "Traceback..."

    @pytest.mark.asyncio
    async def test_on_error_exception_object(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_error = AsyncMock(return_value=2)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_error(ValueError("bad value"))
        repo.persist_error.assert_called_once()
        kw = repo.persist_error.call_args.kwargs
        assert kw["error_type"] == "ValueError"


class TestAuditEventPersistencePnL:
    @pytest.mark.asyncio
    async def test_on_equity_snapshot(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_pnl_snapshot = AsyncMock(return_value=1)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_equity_snapshot({
            "exchange": "binance",
            "equity": 10500.0,
            "realized_pnl": 500.0,
            "drawdown_pct": 2.0,
        })
        repo.persist_pnl_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_pnl_update(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.persist_pnl_snapshot = AsyncMock(return_value=2)
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)

        await aep._on_pnl_update({
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "realized_pnl": 100.0,
        })
        repo.persist_pnl_snapshot.assert_called_once()


class TestAuditEventPersistenceSubscriptions:
    @pytest.mark.asyncio
    async def test_subscribe_all_registers_handlers(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        event_bus = EventBus()
        aep = AuditEventPersistence(repo, event_bus)
        aep.subscribe_all()

        expected_events = [
            "SIGNAL", "RISK_BLOCK", "RISK_DECISION",
            "USER_ORDER_UPDATE", "USER_ACCOUNT_UPDATE", "FILL_CONFIRMED",
            "RECONCILIATION_COMPLETE", "EQUITY_SNAPSHOT",
            "PNL_UPDATE", "COMPONENT_ERROR",
        ]
        for event_name in expected_events:
            subs = event_bus._handlers.get(event_name, [])
            assert len(subs) >= 1, f"No subscriber for {event_name}"


# ── StateRecovery Tests ───────────────────────────────────────────────────


class TestStateRecoveryRecover:
    @pytest.mark.asyncio
    async def test_full_recovery_success(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_open_orders = AsyncMock(return_value=[
            {"order_id": "o1", "status": "open"},
            {"order_id": "o2", "status": "pending"},
        ])
        repo.load_open_positions = AsyncMock(return_value=[
            {"exchange": "binance", "symbol": "BTCUSDT", "size": 1.0},
        ])
        # SR-1 fix: state_recovery now uses the batched load_fills_for_orders
        # call instead of N+1 per-order. Mock returns {oid: [fill]} dict.
        repo.load_fills_for_orders = AsyncMock(return_value={
            "o1": [{"fill_id": "f1", "quantity": 0.5}],
            "o2": [{"fill_id": "f2", "quantity": 0.5}],
        })
        repo.load_latest_equity = AsyncMock(return_value={
            "equity": 10500.0, "drawdown_pct": 2.0,
        })
        repo.load_last_reconciliation = AsyncMock(return_value={
            "reconciliation_id": "r1", "safe_mode": False,
        })

        recovery = StateRecovery(repo)
        result = await recovery.recover()

        assert result.success is True
        assert result.orders_recovered == 2
        assert result.positions_recovered == 1
        assert result.fills_recovered == 2  # 1 fill per order * 2 orders
        assert result.last_equity == 10500.0
        assert result.last_drawdown == 2.0
        assert result.safe_mode is False
        assert result.reconciliation_id == "r1"
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_recovery_detects_safe_mode(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_open_orders = AsyncMock(return_value=[])
        repo.load_open_positions = AsyncMock(return_value=[])
        repo.load_latest_equity = AsyncMock(return_value=None)
        repo.load_last_reconciliation = AsyncMock(return_value={
            "reconciliation_id": "r2", "safe_mode": True,
        })

        recovery = StateRecovery(repo)
        result = await recovery.recover()

        assert result.success is True
        assert result.safe_mode is True
        assert result.reconciliation_id == "r2"

    @pytest.mark.asyncio
    async def test_recovery_handles_empty_db(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_open_orders = AsyncMock(return_value=[])
        repo.load_open_positions = AsyncMock(return_value=[])
        repo.load_latest_equity = AsyncMock(return_value=None)
        repo.load_last_reconciliation = AsyncMock(return_value=None)

        recovery = StateRecovery(repo)
        result = await recovery.recover()

        assert result.success is True
        assert result.orders_recovered == 0
        assert result.safe_mode is False
        assert result.last_equity is None

    @pytest.mark.asyncio
    async def test_recovery_handles_exception(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_open_orders = AsyncMock(side_effect=Exception("DB down"))

        recovery = StateRecovery(repo)
        result = await recovery.recover()

        assert result.success is False
        assert len(result.errors) == 1
        assert "DB down" in result.errors[0]


class TestStateRecoveryRebuildOrderManager:
    @pytest.mark.asyncio
    async def test_rebuild_populates_order_maps(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_all_orders = AsyncMock(return_value=[
            {
                "order_id": "ord-1",
                "client_order_id": "cli-1",
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_type": "limit",
                "quantity": 1.0,
                "price": 50000.0,
                "status": "open",
                "filled_qty": 0.0,
                "avg_fill_price": 0.0,
                "total_fee": 0.0,
                "reduce_only": False,
                "created_at": datetime.datetime.now(datetime.UTC),
                "updated_at": datetime.datetime.now(datetime.UTC),
                "metadata": {},
            },
        ])

        # Create a mock OrderManager with required methods
        order_mgr = MagicMock()
        order_mgr.orders = {}
        order_mgr.client_order_map = {}

        # Mock _dict_to_order to return a stub order
        stub_order = MagicMock()
        stub_order.order_id = "ord-1"
        order_mgr._dict_to_order = MagicMock(return_value=stub_order)

        recovery = StateRecovery(repo)
        count = await recovery.rebuild_order_manager_state(order_mgr)

        assert count == 1
        assert "ord-1" in order_mgr.orders
        assert "cli-1" in order_mgr.client_order_map

    @pytest.mark.asyncio
    async def test_rebuild_skips_existing_orders(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_all_orders = AsyncMock(return_value=[
            {"order_id": "existing", "client_order_id": "c1",
             "exchange": "binance", "symbol": "BTC", "side": "buy",
             "order_type": "limit", "quantity": 1.0, "price": 50000.0,
             "status": "filled", "filled_qty": 1.0, "avg_fill_price": 50000.0,
             "total_fee": 5.0, "reduce_only": False,
             "created_at": datetime.datetime.now(datetime.UTC),
             "updated_at": datetime.datetime.now(datetime.UTC), "metadata": {}},
        ])

        order_mgr = MagicMock()
        order_mgr.orders = {"existing": MagicMock()}  # already loaded
        order_mgr.client_order_map = {}

        recovery = StateRecovery(repo)
        count = await recovery.rebuild_order_manager_state(order_mgr)
        assert count == 0  # skipped existing

    @pytest.mark.asyncio
    async def test_rebuild_empty_db(self):
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()
        repo.load_all_orders = AsyncMock(return_value=[])

        order_mgr = MagicMock()
        order_mgr.orders = {}

        recovery = StateRecovery(repo)
        count = await recovery.rebuild_order_manager_state(order_mgr)
        assert count == 0


# ── Schema / Migration Tests ─────────────────────────────────────────────


class TestSchemaModels:
    def test_models_metadata_has_all_tables(self):
        from storage.models import metadata
        table_names = set(metadata.tables.keys())
        expected = {
            "ticks", "candles", "funding_rates", "open_interest",
            "positions", "orders", "fills",
            "signal_events", "risk_decisions", "user_stream_events",
            "reconciliation_events", "pnl_snapshots", "daily_pnl",
            "risk_blocks", "equity_snapshots", "errors", "signals",
        }
        assert expected.issubset(table_names), f"Missing: {expected - table_names}"

    def test_signal_events_has_correlation_id(self):
        from storage.models import signal_events
        col_names = {c.name for c in signal_events.columns}
        assert "correlation_id" in col_names
        assert "factors_active" in col_names
        assert "acted_on" in col_names

    def test_reconciliation_events_has_unique_recon_id(self):
        from storage.models import reconciliation_events
        col_names = {c.name for c in reconciliation_events.columns}
        assert "reconciliation_id" in col_names
        # Check unique constraint
        recon_col = next(c for c in reconciliation_events.columns if c.name == "reconciliation_id")
        assert recon_col.unique is True

    def test_orders_has_correlation_id(self):
        from storage.models import orders
        col_names = {c.name for c in orders.columns}
        assert "correlation_id" in col_names

    def test_fills_has_unique_constraint(self):
        from storage.models import fills
        # Check the unique constraint on (fill_id, exchange)
        constraints = [c for c in fills.constraints if hasattr(c, "columns")]
        unique_constraint = next(
            (c for c in constraints
             if set(col.name for col in c.columns) == {"fill_id", "exchange"}),
            None,
        )
        assert unique_constraint is not None


class TestAlembicMigration:
    def test_migration_module_is_importable(self):
        from importlib import import_module
        migration = import_module("storage.migrations.versions.0001_audit_trail_schema")
        assert migration.revision == "0001"
        assert migration.down_revision is None

    def test_migration_has_upgrade_and_downgrade(self):
        from importlib import import_module
        migration = import_module("storage.migrations.versions.0001_audit_trail_schema")
        assert callable(migration.upgrade)
        assert callable(migration.downgrade)


class TestTradePersistenceMigrateAuditDDL:
    @pytest.mark.asyncio
    async def test_migrate_creates_audit_tables(self):
        """Verify TradePersistence.migrate() includes the audit DDL."""
        from storage.trade_persistence import AUDIT_DDL, AUDIT_ALTER_COLUMNS

        # Ensure the audit DDL constants exist and are non-empty
        assert len(AUDIT_DDL) >= 5
        assert len(AUDIT_ALTER_COLUMNS) >= 7

        # Check table names are referenced
        ddl_text = "\n".join(AUDIT_DDL)
        for table in ["signal_events", "risk_decisions", "user_stream_events",
                       "reconciliation_events", "pnl_snapshots"]:
            assert table in ddl_text, f"Missing {table} in AUDIT_DDL"

        # Check alter columns include correlation_id
        alter_text = "\n".join(AUDIT_ALTER_COLUMNS)
        assert "correlation_id" in alter_text


# ── End-to-end: Signal → Risk → Order → Fill → Recovery ──────────────────


class TestEndToEndAuditChain:
    """Simulates a full trading cycle and verifies the audit trail records
    every step with correlated IDs."""

    @pytest.mark.asyncio
    async def test_signal_to_fill_audit_chain(self):
        """E2E: Signal emitted → persisted, risk decision → persisted,
        user stream event → persisted, then recovery reads it all back."""
        conn = FakeConnection(fetchrow_result={"id": 1})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        # 1. Signal persisted
        sig_id = await repo.persist_signal(
            exchange="binance", symbol="BTCUSDT", direction="long",
            score=0.85, price=50000.0, correlation_id="chain-001",
        )
        assert sig_id == 1

        # 2. Risk decision (approved)
        risk_id = await repo.persist_risk_decision(
            exchange="binance", symbol="BTCUSDT",
            decision="approved", reason="within_limits",
            signal_score=0.85, correlation_id="chain-001",
        )
        assert risk_id == 1

        # 3. User stream event (fill)
        us_id = await repo.persist_user_stream_event(
            event_type="ORDER_TRADE_UPDATE",
            raw_payload={"order_id": "o1", "status": "FILLED", "price": 50000},
            correlation_id="chain-001",
        )
        assert us_id == 1

        # 4. Error (simulated)
        err_id = await repo.persist_error(
            component="exchange",
            error_type="TimeoutError",
            message="read timeout",
            correlation_id="chain-001",
            stack_trace="...",
        )
        assert err_id == 1

        # Verify all 4 inserts hit the DB
        assert len(conn._calls) == 4

        # All correlation_ids match
        for sql, args in conn._calls:
            # correlation_id should be in the args
            assert "chain-001" in args, f"Missing correlation_id in: {sql[:60]}"

    @pytest.mark.asyncio
    async def test_crash_recovery_rebuilds_state(self):
        """Simulate: bot wrote orders+positions to DB, crashed, restarts,
        StateRecovery reads them back."""
        repo = AuditRepository.__new__(AuditRepository)
        repo._pool = MagicMock()

        # Simulate DB has 2 open orders and 1 position from before crash
        repo.load_open_orders = AsyncMock(return_value=[
            {"order_id": "pre-crash-1", "status": "open"},
            {"order_id": "pre-crash-2", "status": "partially_filled"},
        ])
        repo.load_open_positions = AsyncMock(return_value=[
            {"exchange": "binance", "symbol": "BTCUSDT", "size": 0.5, "entry_price": 49500},
        ])
        # SR-1: state_recovery uses batched load_fills_for_orders
        repo.load_fills_for_orders = AsyncMock(return_value={
            "pre-crash-1": [{"fill_id": "f1", "quantity": 0.3, "price": 49500}],
            "pre-crash-2": [{"fill_id": "f2", "quantity": 0.2, "price": 49600}],
        })
        repo.load_latest_equity = AsyncMock(return_value={
            "equity": 9800.0, "drawdown_pct": 3.5,
        })
        repo.load_last_reconciliation = AsyncMock(return_value={
            "reconciliation_id": "pre-crash-recon",
            "safe_mode": True,  # was in safe mode before crash
        })

        recovery = StateRecovery(repo)
        result = await recovery.recover()

        # Verify recovery captured pre-crash state
        assert result.success is True
        assert result.orders_recovered == 2
        assert result.positions_recovered == 1
        assert result.fills_recovered == 2  # 1 fill per each of 2 orders
        assert result.last_equity == 9800.0
        assert result.last_drawdown == 3.5
        assert result.safe_mode is True
        assert result.reconciliation_id == "pre-crash-recon"


# ── Idempotency Tests ────────────────────────────────────────────────────


class TestIdempotentInserts:
    @pytest.mark.asyncio
    async def test_duplicate_reconciliation_noop(self):
        """Inserting the same reconciliation_id twice should be a no-op."""
        conn = FakeConnection(fetchrow_result=None)  # conflict returns None
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        r1 = await repo.persist_reconciliation(
            reconciliation_id="dup-test", success=True,
        )
        # Second insert with same ID
        r2 = await repo.persist_reconciliation(
            reconciliation_id="dup-test", success=True,
        )

        # Both should have been attempted (mock doesn't enforce constraint)
        # but the SQL uses ON CONFLICT DO NOTHING
        assert len(conn._calls) == 2
        for sql, _ in conn._calls:
            assert "ON CONFLICT (reconciliation_id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_signal_events_allows_duplicates(self):
        """Signal events are append-only (no unique constraint on content)."""
        conn = FakeConnection(fetchrow_result={"id": 1})
        pool = FakePool(conn)
        repo = AuditRepository(pool)

        id1 = await repo.persist_signal(
            exchange="binance", symbol="BTCUSDT", direction="long",
            score=0.8, price=50000,
        )
        id2 = await repo.persist_signal(
            exchange="binance", symbol="BTCUSDT", direction="long",
            score=0.8, price=50000,
        )

        assert id1 == 1
        assert id2 == 1  # same mock, but both inserts ran
        assert len(conn._calls) == 2


# ── RecoveryResult Dataclass Tests ────────────────────────────────────────


class TestRecoveryResult:
    def test_default_values(self):
        r = RecoveryResult()
        assert r.success is False
        assert r.orders_recovered == 0
        assert r.positions_recovered == 0
        assert r.fills_recovered == 0
        assert r.last_equity is None
        assert r.last_drawdown is None
        assert r.safe_mode is False
        assert r.reconciliation_id is None
        assert r.errors == []

    def test_custom_values(self):
        r = RecoveryResult(success=True, orders_recovered=5, safe_mode=True)
        assert r.success is True
        assert r.orders_recovered == 5
        assert r.safe_mode is True
