from __future__ import annotations

from unittest.mock import MagicMock

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
