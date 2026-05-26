from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.exchange_factory import create_executor
from execution.order_manager import OrderManager, OrderStatus
from execution.risk_manager import RiskManager
from execution.simulated_exchange import SimulatedExchangeExecutor
from storage.sqlite_store import SQLiteStore


class _ConfigStub:
    paper_mode = True

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


class _CircuitStub:
    tripped = False


class _RiskStub:
    def __init__(self) -> None:
        self.rebases: list[tuple[float, float]] = []
        self.cancelled: list[tuple[str, str]] = []
        self._positions: dict[str, SimpleNamespace] = {}

    @property
    def positions(self) -> dict[str, SimpleNamespace]:
        return dict(self._positions)

    async def approve_and_open(self, signal, reserve_until_fill=False):
        pos = SimpleNamespace(
            exchange=signal.exchange,
            symbol=signal.symbol,
            direction=signal.direction,
            size=0.0,
            entry_price=signal.price,
            current_price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            pending_fill=reserve_until_fill,
            is_long=signal.direction == "long",
            pnl=0.0,
            pnl_pct=0.0,
        )
        self._positions[f"{signal.exchange}:{signal.symbol}"] = pos
        return True, "approved", 1_000.0, pos

    async def rebase_position_to_fill(self, exchange, symbol, fill_price, filled_quantity):
        self.rebases.append((fill_price, filled_quantity))
        pos = self._positions.get(f"{exchange}:{symbol}")
        if pos is None:
            return None
        pos.entry_price = fill_price
        pos.current_price = fill_price
        pos.size = filled_quantity
        pos.pending_fill = False
        return pos

    async def cancel_reserved_position(self, exchange, symbol):
        self.cancelled.append((exchange, symbol))
        return None

    async def close_position(self, exchange, symbol, price):
        pos = self._positions.pop(f"{exchange}:{symbol}", None)
        if pos is None:
            return None
        pos.current_price = price
        pos.pnl = (price - pos.entry_price) * pos.size if pos.is_long else (pos.entry_price - price) * pos.size
        pos.pnl_pct = pos.pnl / max(pos.entry_price * pos.size, 1e-9)
        return pos

    async def activate_kill_switch(self):
        return []


def _config(partial_probability: float = 0.0) -> _ConfigStub:
    return _ConfigStub(
        {
            "system": {"paper_mode": True},
            "exchanges": {"binance": {"enabled": True}},
            "backtest": {"slippage_pct": 0.0002, "commission_pct": 0.0004},
            "execution": {
                "simulated_exchange": {
                    "enabled": True,
                    "ack_latency_ms": 0,
                    "fill_latency_ms": 0,
                    "partial_fill_probability": partial_probability,
                    "partial_fill_ratio": 0.4,
                    "partial_fill_completion_ms": 0,
                    "reject_probability": 0,
                }
            },
            "risk": {
                "stop_loss_pct": 0.015,
                "rr_ratio": 2.0,
                "max_open_positions": 5,
            },
            "arms": {},
        }
    )


def _signal() -> TradingSignal:
    return TradingSignal(
        exchange="binance",
        symbol="BTC/USDT:USDT",
        direction="long",
        score=0.9,
        technical_score=0.8,
        ml_score=0.7,
        sentiment_score=0.1,
        macro_score=0.0,
        news_score=0.0,
        orderbook_score=0.2,
        regime="trend",
        regime_confidence=0.8,
        price=50_000.0,
        atr=500.0,
        stop_loss=49_000.0,
        take_profit=53_000.0,
        timestamp=int(time.time()),
        metadata={"spread_bps": 1.0, "atr_percentile": 50},
    )


@pytest.mark.asyncio
async def test_simulated_executor_records_order_manager_fill() -> None:
    cfg = _config()
    bus = EventBus()
    risk = _RiskStub()
    order_manager = OrderManager(cfg, bus, _CircuitStub(), audit_log_path="", order_state_path="")
    executor = SimulatedExchangeExecutor(cfg, bus, risk, "binance", order_manager=order_manager)

    result = await executor.execute_signal(_signal(), 1_000.0)

    assert result is not None
    assert result.status == "filled"
    filled = order_manager.get_filled_orders("binance")
    assert len(filled) == 1
    assert filled[0].status == OrderStatus.FILLED
    assert len(risk.rebases) == 1


@pytest.mark.asyncio
async def test_simulated_executor_can_complete_partial_fill() -> None:
    cfg = _config(partial_probability=1.0)
    bus = EventBus()
    risk = _RiskStub()
    order_manager = OrderManager(cfg, bus, _CircuitStub(), audit_log_path="", order_state_path="")
    executor = SimulatedExchangeExecutor(cfg, bus, risk, "binance", order_manager=order_manager)

    result = await executor.execute_signal(_signal(), 1_000.0)
    await asyncio.sleep(0.01)

    assert result is not None
    assert result.status == "partially_filled"
    filled = order_manager.get_filled_orders("binance")
    assert len(filled) == 1
    assert len(risk.rebases) == 2
    assert risk.rebases[0][1] < risk.rebases[1][1]


@pytest.mark.asyncio
async def test_simulated_executor_close_records_reduce_only_fill() -> None:
    cfg = _config()
    bus = EventBus()
    risk = _RiskStub()
    order_manager = OrderManager(cfg, bus, _CircuitStub(), audit_log_path="", order_state_path="")
    executor = SimulatedExchangeExecutor(cfg, bus, risk, "binance", order_manager=order_manager)

    entry = await executor.execute_signal(_signal(), 1_000.0)
    assert entry is not None

    close = await executor.close_position("BTC/USDT:USDT", entry.price, reason="unit_test_close")

    assert close is not None
    assert close.status == "closed"
    assert risk.positions == {}
    filled = order_manager.get_filled_orders("binance")
    assert len(filled) == 2
    assert filled[-1].metadata["reduce_only"] is True


@pytest.mark.asyncio
async def test_simulated_stop_loss_routes_through_reduce_only_order() -> None:
    cfg = _config()
    bus = EventBus()
    risk = _RiskStub()
    order_manager = OrderManager(cfg, bus, _CircuitStub(), audit_log_path="", order_state_path="")
    executor = SimulatedExchangeExecutor(cfg, bus, risk, "binance", order_manager=order_manager)

    entry = await executor.execute_signal(_signal(), 1_000.0)
    assert entry is not None

    await executor._handle_stop_loss({
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "price": entry.price * 0.99,
    })

    filled = order_manager.get_filled_orders("binance")
    assert len(filled) == 2
    assert filled[-1].metadata["close_reason"] == "simulated_stop_loss"
    assert filled[-1].metadata["reduce_only"] is True
    assert risk.positions == {}


@pytest.mark.asyncio
async def test_risk_manager_restores_sqlite_open_position(tmp_path) -> None:
    cfg = _config()
    bus = EventBus()
    store = SQLiteStore(tmp_path / "recovery.db")
    pos_id = store.insert_position(
        "binance",
        "BTC/USDT:USDT",
        "long",
        50_000.0,
        0.02,
        is_paper=True,
    )
    risk = RiskManager(cfg, bus, sqlite_store=store)

    result = await risk.restore_open_positions_from_sqlite()

    assert result["success"] is True
    assert result["restored"] == 1
    restored = risk.positions["binance:BTC/USDT:USDT"]
    assert restored.entry_price == 50_000.0
    assert restored.size == 0.02
    assert restored.pending_fill is False
    assert restored.stop_loss < restored.entry_price < restored.take_profit
    assert getattr(restored, "_db_id") == pos_id


def test_factory_uses_simulated_executor_in_paper_mode() -> None:
    executor = create_executor(
        "binance",
        _config(),
        EventBus(),
        MagicMock(),
        order_manager=MagicMock(),
    )

    assert isinstance(executor, SimulatedExchangeExecutor)


def test_sqlite_archive_refuses_live_and_closes_paper_duplicate(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "ledger_cleanup.db")
    paper_id = store.insert_position(
        "binance",
        "BTC/USDT:USDT",
        "long",
        50_000.0,
        0.01,
        is_paper=True,
    )
    live_id = store.insert_position(
        "binance",
        "ETH/USDT:USDT",
        "long",
        3_000.0,
        0.1,
        is_paper=False,
    )

    paper_result = store.archive_paper_open_position(paper_id, reason="unit_duplicate_cleanup")
    live_result = store.archive_paper_open_position(live_id, reason="unit_duplicate_cleanup")

    assert paper_result["success"] is True
    assert live_result["success"] is False
    assert live_result["reason"] == "live_row_refused"
    open_rows = store.get_open_positions()
    assert [row["id"] for row in open_rows] == [live_id]
