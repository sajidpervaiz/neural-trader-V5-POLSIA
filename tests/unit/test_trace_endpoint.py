"""REQ-TR-001: end-to-end correlation_id traceability across audit tables."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from storage.audit_repository import AuditRepository


class _FakeConn:
    def __init__(self, table_rows: dict[str, list[dict]]) -> None:
        self.table_rows = table_rows

    async def fetch(self, sql: str, *args):
        # Parse table name out of the SQL text.
        for tname in self.table_rows:
            if f" FROM {tname} " in sql or sql.startswith(f"SELECT * FROM {tname}"):
                return [dict(r) for r in self.table_rows[tname]]
        return []


class _FakePool:
    def __init__(self, table_rows: dict[str, list[dict]]) -> None:
        self.conn = _FakeConn(table_rows)

    def acquire(self):
        # async context manager
        outer = self
        class _Ctx:
            async def __aenter__(self_inner):
                return outer.conn
            async def __aexit__(self_inner, *exc):
                return False
        return _Ctx()


@pytest.mark.asyncio
async def test_load_trace_empty_when_no_pool() -> None:
    repo = AuditRepository(pool=None)
    trace = await repo.load_trace_by_correlation_id("any-cid")
    assert trace["signals"] == []
    assert trace["orders"] == []


@pytest.mark.asyncio
async def test_load_trace_empty_when_no_correlation_id() -> None:
    repo = AuditRepository(pool=_FakePool({"signal_events": [{"x": 1}]}))
    trace = await repo.load_trace_by_correlation_id("")
    assert sum(len(v) for v in trace.values()) == 0


@pytest.mark.asyncio
async def test_load_trace_assembles_rows_across_tables() -> None:
    rows = {
        "signal_events": [{"id": 1, "correlation_id": "cid-A"}],
        "risk_blocks": [],
        "orders": [{"id": 7, "correlation_id": "cid-A"}],
        "fills": [
            {"id": 11, "correlation_id": "cid-A"},
            {"id": 12, "correlation_id": "cid-A"},
        ],
        "user_stream_events": [],
        "pnl_snapshots": [],
        "reconciliation_events": [],
    }
    repo = AuditRepository(pool=_FakePool(rows))
    trace = await repo.load_trace_by_correlation_id("cid-A")
    assert len(trace["signals"]) == 1
    assert len(trace["orders"]) == 1
    assert len(trace["fills"]) == 2
    assert trace["risk_blocks"] == []
    # All expected keys present even when empty.
    assert set(trace.keys()) == {
        "signals", "risk_blocks", "orders", "fills",
        "user_stream_events", "pnl_snapshots", "reconciliation_events",
    }
