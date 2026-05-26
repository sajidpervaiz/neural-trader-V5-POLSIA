from __future__ import annotations

from unittest.mock import MagicMock

from execution.executor_contract import executor_contract_status
from execution.exchange_factory import create_all_executors, create_executor


class _ConfigStub:
    def __init__(self, data: dict) -> None:
        self._data = data

    def get_value(self, *keys, default=None):
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node


def test_factory_returns_runtime_compatible_executors_for_bybit_and_okx() -> None:
    config = _ConfigStub(
        {
            "exchanges": {
                "bybit": {"enabled": True, "api_key": "k", "api_secret": "s", "testnet": True},
                "okx": {"enabled": True, "api_key": "k", "api_secret": "s", "passphrase": "p", "testnet": True},
            }
        }
    )
    bus = MagicMock()
    risk_manager = MagicMock()

    executors = create_all_executors(config, bus, risk_manager)

    assert len(executors) == 2
    assert {executor.exchange_id for executor in executors} == {"bybit", "okx"}
    for executor in executors:
        assert callable(getattr(executor, "run", None))
        status = executor_contract_status(executor)
        assert status["contract_ok"], status


def test_factory_injects_order_manager_into_cex_executor_instances() -> None:
    config = _ConfigStub(
        {
            "exchanges": {
                "binance": {"enabled": True},
                "bybit": {"enabled": True, "api_key": "k", "api_secret": "s", "testnet": True},
            }
        }
    )
    bus = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()

    binance = create_executor("binance", config, bus, risk_manager, order_manager=order_manager)
    bybit = create_executor("bybit", config, bus, risk_manager, order_manager=order_manager)

    assert binance is not None
    assert bybit is not None
    assert getattr(binance, "_order_manager", None) is order_manager
    assert getattr(bybit, "_order_manager", None) is order_manager


def test_executor_contract_status_blocks_missing_safety_methods() -> None:
    class IncompleteExecutor:
        exchange_id = "incomplete"

        async def run(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def execute_signal(self, signal, size):
            return None

    status = executor_contract_status(IncompleteExecutor())

    assert not status["contract_ok"]
    assert "close_position" in status["blockers"]
    assert "cancel_order" in status["blockers"]
    assert "get_orderbook_snapshot" in status["blockers"]
