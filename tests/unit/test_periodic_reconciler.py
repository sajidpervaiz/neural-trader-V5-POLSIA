"""REQ-POS-004 / REQ-FS-007: PeriodicReconciler must diff exchange vs
internal positions on a schedule and trip SafeMode on mismatch."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

from core.config import Config
from core.safe_mode import SafeModeManager, SafeModeReason
from execution.reconciliation import PeriodicReconciler


def _make_risk_manager(internal_positions: dict[str, float]) -> MagicMock:
    rm = MagicMock()
    rm.safe_mode = SafeModeManager()
    # Each position only needs a `.size` attribute.
    rm.positions = {
        sym: SimpleNamespace(size=qty)
        for sym, qty in internal_positions.items()
    }
    return rm


def _make_async_client(positions: list[dict]) -> MagicMock:
    client = MagicMock()
    client.fetch_positions = AsyncMock(return_value=positions)
    return client


@pytest.mark.asyncio
async def test_no_mismatch_when_internal_matches_exchange() -> None:
    rm = _make_risk_manager({"BTC/USDT:USDT": 0.5})
    client = _make_async_client([{"symbol": "BTC/USDT:USDT", "contracts": 0.5}])
    pr = PeriodicReconciler(Config(), rm, client, interval_seconds=60.0)
    result = await pr._check_once()
    assert result["mismatches"] == []
    assert result["safe_mode_triggered"] is False
    assert not rm.safe_mode.is_active


@pytest.mark.asyncio
async def test_quantity_mismatch_trips_safe_mode() -> None:
    rm = _make_risk_manager({"BTC/USDT:USDT": 0.5})
    client = _make_async_client([{"symbol": "BTC/USDT:USDT", "contracts": 0.7}])
    pr = PeriodicReconciler(Config(), rm, client, interval_seconds=60.0)
    result = await pr._check_once()
    assert any("BTC/USDT:USDT" in m for m in result["mismatches"])
    assert result["safe_mode_triggered"] is True
    assert rm.safe_mode.is_active


@pytest.mark.asyncio
async def test_position_only_on_exchange_is_mismatch() -> None:
    rm = _make_risk_manager({})
    client = _make_async_client([{"symbol": "ETH/USDT:USDT", "contracts": 1.0}])
    pr = PeriodicReconciler(Config(), rm, client, interval_seconds=60.0)
    result = await pr._check_once()
    assert any("ETH/USDT:USDT" in m for m in result["mismatches"])
    assert result["safe_mode_triggered"] is True


@pytest.mark.asyncio
async def test_position_only_in_internal_is_mismatch() -> None:
    rm = _make_risk_manager({"SOL/USDT:USDT": 2.0})
    client = _make_async_client([])
    pr = PeriodicReconciler(Config(), rm, client, interval_seconds=60.0)
    result = await pr._check_once()
    assert any("SOL/USDT:USDT" in m for m in result["mismatches"])
    assert result["safe_mode_triggered"] is True


@pytest.mark.asyncio
async def test_fetch_failure_recorded_but_does_not_crash() -> None:
    rm = _make_risk_manager({})
    client = MagicMock()
    client.fetch_positions = AsyncMock(side_effect=RuntimeError("boom"))
    pr = PeriodicReconciler(Config(), rm, client, interval_seconds=60.0)
    result = await pr._check_once()
    assert any("fetch_positions_failed" in m for m in result["mismatches"])
    # Fetch failure shouldn't trip SafeMode by itself — only divergence does.
    # (Exchange API outage is handled by other monitors.)
    assert result["safe_mode_triggered"] is False
