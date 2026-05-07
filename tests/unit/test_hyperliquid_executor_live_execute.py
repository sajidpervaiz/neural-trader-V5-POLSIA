from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from execution.hyperliquid_executor import HyperliquidExecutor


class _HLExchangeStub:
    def market_open(self, coin: str, is_buy: bool, size: float, slippage: float):
        _ = (coin, is_buy, size, slippage)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"oid": 123, "avgPx": "101.5", "totalSz": "0.25"}}
                    ]
                }
            },
        }


@pytest.mark.asyncio
async def test_live_execute_syncs_reserved_position_and_publishes_fill_event() -> None:
    config = MagicMock()
    config.get_value.return_value = {}
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    risk_manager = MagicMock()
    risk_manager.open_position = AsyncMock()
    executor = HyperliquidExecutor(config, event_bus, risk_manager)
    executor._hl_exchange = _HLExchangeStub()

    signal = SimpleNamespace(direction="long", symbol="BTC/USDT:USDT", price=100.0)
    reserved_pos = SimpleNamespace(current_price=0.0, size=0.0)

    result = await executor._live_execute(signal, size=0.25, reserved_pos=reserved_pos)

    assert result is not None
    assert reserved_pos.current_price == pytest.approx(101.5)
    assert reserved_pos.size == pytest.approx(0.25)
    risk_manager.open_position.assert_not_awaited()
    event_bus.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_execute_opens_position_when_reserved_pos_missing() -> None:
    config = MagicMock()
    config.get_value.return_value = {}
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    risk_manager = MagicMock()
    risk_manager.open_position = AsyncMock()
    executor = HyperliquidExecutor(config, event_bus, risk_manager)
    executor._hl_exchange = _HLExchangeStub()

    signal = SimpleNamespace(direction="long", symbol="BTC/USDT:USDT", price=100.0)

    result = await executor._live_execute(signal, size=0.25, reserved_pos=None)

    assert result is not None
    risk_manager.open_position.assert_awaited_once()
    args = risk_manager.open_position.await_args.args
    assert args[0] is signal
    assert float(args[1]) == pytest.approx(25.375)
    event_bus.publish.assert_awaited_once()
