import asyncio
from types import SimpleNamespace

import pytest

from core.circuit_breaker import CircuitBreaker
from core.config import Config
from core.event_bus import EventBus
from execution.order_manager import OrderManager, OrderSide, OrderType

CONFIG_PATH = "config/settings.yaml"


class _Venue:
    value = "binance"


class _SlowRouter:
    def __init__(self, delay: float = 0.20):
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def route_order(self, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return SimpleNamespace(
            recommended_venue=_Venue(),
            confidence=1.0,
            expected_avg_price=kwargs.get("quantity", 0.0),
            routes=[],
        )


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
async def test_concurrent_duplicate_client_order_id_creates_one_order(tmp_path):
    """Concurrent duplicate client_order_id calls must be atomic/idempotent."""
    manager = _manager(tmp_path)
    client_order_id = "race-duplicate-client-id"

    results = await asyncio.gather(
        *[
            manager.place_order(
                exchange="binance",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
                price=42000.0,
                order_type=OrderType.LIMIT,
                client_order_id=client_order_id,
            )
            for _ in range(25)
        ]
    )

    assert all(success for success, _order, _reason in results)
    returned_orders = [order for _success, order, _reason in results]
    assert all(order is returned_orders[0] for order in returned_orders)
    assert {reason for _success, _order, reason in results} == {"created", "idempotent_retry"}
    assert list(manager.client_order_map.keys()) == [client_order_id]
    assert len(manager.orders) == 1
    assert len(manager.audit_log) == 1


@pytest.mark.asyncio
async def test_exchange_routing_awaits_do_not_hold_order_idempotency_lock(tmp_path):
    """Slow external routing must happen before the order idempotency critical section."""
    manager = _manager(tmp_path)
    router = _SlowRouter(delay=0.20)
    manager.attach_router(router)

    results = await asyncio.gather(
        manager.place_order(
            exchange="auto",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
            order_type=OrderType.LIMIT,
            client_order_id="race-route-1",
        ),
        manager.place_order(
            exchange="auto",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=0.2,
            price=3000.0,
            order_type=OrderType.LIMIT,
            client_order_id="race-route-2",
        ),
    )
    assert all(success for success, _order, reason in results), results
    assert {reason for _success, _order, reason in results} == {"created"}
    assert len(manager.orders) == 2
    assert router.calls == 2
    assert router.max_active == 2, "routing serialized under order idempotency lock"


@pytest.mark.asyncio
async def test_invalid_order_inputs_fail_before_idempotency_record(tmp_path):
    """Input validation must reject bad requests before touching idempotency state."""
    manager = _manager(tmp_path)

    success, order, reason = await manager.place_order(
        exchange="binance",
        symbol="",
        side=OrderSide.BUY,
        quantity=0.1,
        price=42000.0,
        order_type=OrderType.LIMIT,
        client_order_id="invalid-symbol",
    )

    assert success is False
    assert order is None
    assert reason == "invalid_symbol"
    assert "invalid-symbol" not in manager.idempotency.records
    assert manager.orders == {}
