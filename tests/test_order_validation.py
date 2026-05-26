"""Regression tests for order input validation."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.circuit_breaker import CircuitBreaker
from core.config import Config
from core.event_bus import EventBus
from execution.order_manager import OrderManager, OrderSide, OrderType
from execution.order_validation import (
    validate_order_side,
    validate_order_type,
    validate_price,
    validate_quantity,
    validate_symbol,
)

CONFIG_PATH = "config/settings.yaml"


@pytest.mark.parametrize("symbol", ["BTC/USDT", "BTC/USDT:USDT", "BTC-USDT-SWAP", "PF_XBTUSD"])
def test_validate_symbol_accepts_supported_exchange_formats(symbol: str) -> None:
    validate_symbol(symbol)


@pytest.mark.parametrize("symbol", ["", "   ", None, "BTC USDT", "BTC/USDT;DROP"])
def test_validate_symbol_rejects_invalid_values(symbol) -> None:
    with pytest.raises(ValueError, match="symbol"):
        validate_symbol(symbol)


@pytest.mark.parametrize("quantity", [0.1, 1, "2.5"])
def test_validate_quantity_accepts_positive_values(quantity) -> None:
    validate_quantity(quantity)


@pytest.mark.parametrize("quantity", [0, -1, "0", "-2", None, math.nan, math.inf])
def test_validate_quantity_rejects_non_positive_or_non_finite_values(quantity) -> None:
    with pytest.raises(ValueError, match="quantity"):
        validate_quantity(quantity)


@pytest.mark.parametrize("price, order_type", [(42000.0, OrderType.LIMIT), (1, "post_only"), (None, OrderType.MARKET), (0, "market")])
def test_validate_price_accepts_valid_prices(price, order_type) -> None:
    validate_price(price, order_type)


@pytest.mark.parametrize("price, order_type", [(None, OrderType.LIMIT), (0, "limit"), (-1, "ioc"), (math.nan, "post_only"), (math.inf, "limit")])
def test_validate_price_rejects_invalid_prices(price, order_type) -> None:
    with pytest.raises(ValueError, match="price"):
        validate_price(price, order_type)


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL, "buy", "sell"])
def test_validate_order_side_accepts_valid_sides(side) -> None:
    validate_order_side(side)


@pytest.mark.parametrize("side", ["", "hold", None, SimpleNamespace(value="buy")])
def test_validate_order_side_rejects_invalid_sides(side) -> None:
    with pytest.raises(ValueError, match="side"):
        validate_order_side(side)


@pytest.mark.parametrize("order_type", [OrderType.LIMIT, OrderType.MARKET, OrderType.POST_ONLY, OrderType.IOC, "limit", "market", "post_only", "ioc"])
def test_validate_order_type_accepts_valid_types(order_type) -> None:
    validate_order_type(order_type)


@pytest.mark.parametrize("order_type", ["", "stop", None, SimpleNamespace(value="limit")])
def test_validate_order_type_rejects_invalid_types(order_type) -> None:
    with pytest.raises(ValueError, match="order_type"):
        validate_order_type(order_type)


def _manager(tmp_path) -> OrderManager:
    config = Config(config_path=CONFIG_PATH)
    event_bus = EventBus()
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    manager = OrderManager(
        config,
        event_bus,
        breaker,
        audit_log_path=str(tmp_path / "audit.jsonl"),
        order_state_path=str(tmp_path / "orders.json"),
    )
    manager.idempotency.records.clear()
    manager.idempotency.filepath = str(tmp_path / "idempotency.json")
    return manager


@pytest.mark.asyncio
async def test_place_order_rejects_invalid_side_before_creating_order(tmp_path) -> None:
    manager = _manager(tmp_path)

    success, order, reason = await manager.place_order(
        exchange="binance",
        symbol="BTC/USDT",
        side="hold",  # type: ignore[arg-type]
        quantity=0.1,
        price=42000.0,
        order_type=OrderType.LIMIT,
        client_order_id="invalid-side",
    )

    assert success is False
    assert order is None
    assert reason == "invalid_side"
    assert "invalid-side" not in manager.idempotency.records
    assert manager.orders == {}


@pytest.mark.asyncio
async def test_place_order_rejects_invalid_order_type_before_creating_order(tmp_path) -> None:
    manager = _manager(tmp_path)

    success, order, reason = await manager.place_order(
        exchange="binance",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=42000.0,
        order_type="stop",  # type: ignore[arg-type]
        client_order_id="invalid-type",
    )

    assert success is False
    assert order is None
    assert reason == "invalid_order_type"
    assert "invalid-type" not in manager.idempotency.records
    assert manager.orders == {}


@pytest.mark.asyncio
async def test_place_order_preserves_valid_order_creation(tmp_path) -> None:
    manager = _manager(tmp_path)

    success, order, reason = await manager.place_order(
        exchange="binance",
        symbol="BTC/USDT:USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=42000.0,
        order_type=OrderType.LIMIT,
        client_order_id="valid-order",
    )

    assert success is True
    assert order is not None
    assert reason == "created"
    assert order.symbol == "BTC/USDT:USDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
