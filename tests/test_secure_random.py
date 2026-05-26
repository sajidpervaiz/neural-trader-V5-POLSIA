import asyncio
import inspect
import re
from types import SimpleNamespace

import pytest

from core.circuit_breaker import CircuitBreaker
from core.config import Config
from core.event_bus import EventBus
from execution.order_manager import OrderManager, OrderSide, OrderType
from interface.websocket_manager import WebsocketManager

CONFIG_PATH = "config/settings.yaml"


class _FakeWebSocket:
    async def accept(self):
        self.accepted = True

    async def close(self, code=None, reason=None):
        self.closed = True
        self.close_code = code
        self.close_reason = reason


class _DeterministicSecrets:
    def __init__(self):
        self.token_hex_calls = []
        self.token_urlsafe_calls = []

    def token_hex(self, nbytes):
        self.token_hex_calls.append(nbytes)
        return "a" * (nbytes * 2)

    def token_urlsafe(self, nbytes=None):
        self.token_urlsafe_calls.append(nbytes)
        return "securetoken"


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


def test_client_order_ids_use_secrets_token_hex(monkeypatch, tmp_path):
    secure = _DeterministicSecrets()
    import execution.order_manager as order_manager_module

    monkeypatch.setattr(order_manager_module, "secrets", secure)
    manager = _manager(tmp_path)

    client_order_id = manager.generate_client_order_id("binance", "BTC/USDT", OrderSide.BUY)

    assert secure.token_hex_calls == [8]
    assert client_order_id.endswith("-" + "a" * 16)
    assert client_order_id.startswith("binance-BTC/USDT-b-")


@pytest.mark.asyncio
async def test_order_ids_use_secrets_token_hex(monkeypatch, tmp_path):
    secure = _DeterministicSecrets()
    import execution.order_manager as order_manager_module

    monkeypatch.setattr(order_manager_module, "secrets", secure)
    manager = _manager(tmp_path)

    success, order, reason = await manager.place_order(
        exchange="binance",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=42000.0,
        order_type=OrderType.LIMIT,
    )

    assert success is True
    assert reason == "created"
    assert order is not None
    assert order.order_id == "ord_" + "a" * 32
    assert secure.token_hex_calls == [8, 16]


@pytest.mark.asyncio
async def test_websocket_client_tokens_use_secrets_token_urlsafe(monkeypatch):
    secure = _DeterministicSecrets()
    import interface.websocket_manager as websocket_manager_module

    monkeypatch.setattr(websocket_manager_module, "secrets", secure)
    manager = WebsocketManager()

    client_id = await manager.connect(_FakeWebSocket())

    assert secure.token_urlsafe_calls == [16]
    assert client_id == "ws_securetoken"


def test_sensitive_id_generators_do_not_use_random_or_uuid4():
    import execution.order_manager as order_manager_module
    import interface.websocket_manager as websocket_manager_module

    order_source = inspect.getsource(order_manager_module.OrderManager.generate_client_order_id)
    order_source += inspect.getsource(order_manager_module.OrderManager.place_order)
    ws_source = inspect.getsource(websocket_manager_module.WebsocketManager.connect)

    sensitive_source = order_source + "\n" + ws_source
    assert "random." not in sensitive_source
    assert "uuid.uuid4" not in sensitive_source
    assert "secrets.token_hex" in sensitive_source
    assert "secrets.token_urlsafe" in sensitive_source
