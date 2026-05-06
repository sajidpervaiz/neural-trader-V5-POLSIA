"""End-to-end smoke test of the auto-trading pipeline.

Validates the full SIGNAL → approve_and_open → paper_execute → ORDER_FILLED →
POSITION_OPENED chain works on a real EventBus without any mocks of the bus,
risk manager, or executor entry point.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.cex_executor import CEXExecutor
from execution.risk_manager import RiskManager


def _make_config(paper: bool = True) -> Any:
    cfg = MagicMock()
    cfg.paper_mode = paper

    risk = {
        "initial_equity": 10_000.0,
        "max_position_size_pct": 0.10,
        "risk_per_trade_pct": 0.01,
        "sizing_method": "risk_based",
        "max_open_positions": 3,
        "default_leverage": 1.0,
        "max_daily_loss_pct": 0.06,
        "max_drawdown_pct": 0.15,
        "stop_loss_pct": 0.015,
        "max_spread_bps": 50,
        "max_atr_pct": 0.05,
        "cooldown_seconds": 0,
        "session_start_utc": "00:00",
        "session_end_utc": "23:59",
        "atr_sl_multiplier": 1.5,
        "rr_ratio": 2.0,
        "min_signal_score": 0.0,
        "min_risk_reward": 0.0,
        "max_funding_rate_bps": 1000,
        "min_orderbook_depth_usd": 0,
        "max_order_size_usd": 100_000,
        "max_total_exposure_pct": 5.0,
        "max_exposure_per_symbol_pct": 1.0,
    }
    backtest = {"slippage_pct": 0.0002}

    def _get(*args, **kwargs):
        if not args:
            return kwargs.get("default")
        if args[0] == "risk":
            return risk if len(args) == 1 else risk.get(args[1], kwargs.get("default"))
        if args[0] == "backtest":
            return backtest if len(args) == 1 else backtest.get(args[1], kwargs.get("default"))
        if args[0] == "system" and len(args) >= 2 and args[1] == "paper_mode":
            return paper
        return kwargs.get("default", {})

    cfg.get_value = _get
    return cfg


def _make_signal(symbol: str = "BTC/USDT:USDT", direction: str = "long") -> TradingSignal:
    return TradingSignal(
        exchange="binance",
        symbol=symbol,
        direction=direction,
        score=0.85,
        technical_score=0.7,
        ml_score=0.8,
        sentiment_score=0.5,
        macro_score=0.5,
        news_score=0.5,
        orderbook_score=0.5,
        regime="strong_trend_up",
        regime_confidence=0.9,
        price=50_000.0,
        atr=500.0,
        stop_loss=49_250.0,
        take_profit=51_500.0,
        timestamp=int(time.time()),
    )


@pytest.mark.asyncio
async def test_paper_signal_flows_to_position():
    """A SIGNAL published on the bus must reach the executor and open a paper position."""
    bus = EventBus()
    config = _make_config(paper=True)
    risk = RiskManager(config, bus)
    executor = CEXExecutor(config, bus, risk, exchange_id="binance")

    bus_task = asyncio.create_task(bus.run())
    bus.subscribe("SIGNAL", executor._handle_signal)

    fills: list[Any] = []
    bus.subscribe("ORDER_FILLED", lambda payload: fills.append(payload))

    try:
        await bus.publish("SIGNAL", _make_signal())
        # let the bus drain
        for _ in range(30):
            await asyncio.sleep(0.05)
            if fills:
                break

        assert len(fills) == 1, f"expected 1 fill, got {len(fills)}"
        fill = fills[0]
        assert fill.is_paper is True
        assert fill.status == "filled"
        assert fill.symbol == "BTC/USDT:USDT"
        assert fill.direction == "long"

        # Risk manager has the position
        positions = risk.positions
        assert len(positions) == 1
        pos = next(iter(positions.values()))
        assert pos.symbol == "BTC/USDT:USDT"
        assert pos.direction == "long"
        assert pos.stop_loss < pos.entry_price < pos.take_profit
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_signal_for_other_exchange_is_ignored():
    """SIGNAL with a different exchange id must not be acted on."""
    bus = EventBus()
    config = _make_config(paper=True)
    risk = RiskManager(config, bus)
    executor = CEXExecutor(config, bus, risk, exchange_id="binance")

    bus_task = asyncio.create_task(bus.run())
    bus.subscribe("SIGNAL", executor._handle_signal)

    fills: list[Any] = []
    bus.subscribe("ORDER_FILLED", lambda payload: fills.append(payload))

    try:
        sig = _make_signal()
        sig.exchange = "kraken"
        await bus.publish("SIGNAL", sig)
        for _ in range(20):
            await asyncio.sleep(0.05)

        assert fills == []
        assert len(risk.positions) == 0
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_live_signal_routes_to_exchange_client():
    """Live mode: SIGNAL must hit ccxt.create_limit_order, fill, place SL/TP."""
    from unittest.mock import AsyncMock

    bus = EventBus()
    config = _make_config(paper=False)
    risk = RiskManager(config, bus)

    executor = CEXExecutor.__new__(CEXExecutor)
    executor.config = config
    executor.event_bus = bus
    executor.risk_manager = risk
    executor.exchange_id = "binance"
    executor._order_manager = None
    executor._running = True
    executor._rate_limiter = AsyncMock()
    executor._rate_limiter.acquire = AsyncMock()
    executor._maker_first = True
    executor._post_only = False
    executor._iceberg_threshold_usd = 0.0
    executor._iceberg_chunks = 1
    executor._order_placer = None  # exchange-side SL/TP path skipped in this test

    # Mock ccxt client
    mock_client = AsyncMock()
    mock_client.create_limit_order = AsyncMock(return_value={
        "id": "live-001",
        "status": "closed",
        "average": 50_000.0,
        "filled": 0.1,
    })
    mock_client.markets = {}
    executor._client = mock_client

    async def _wait_ok(signal, order, amount, **kw):
        return order
    executor._wait_for_fill = _wait_ok

    bus_task = asyncio.create_task(bus.run())
    bus.subscribe("SIGNAL", executor._handle_signal)

    fills: list[Any] = []
    bus.subscribe("ORDER_FILLED", lambda payload: fills.append(payload))

    try:
        await bus.publish("SIGNAL", _make_signal())
        for _ in range(40):
            await asyncio.sleep(0.05)
            if fills:
                break

        assert mock_client.create_limit_order.called
        assert len(fills) == 1
        fill = fills[0]
        assert fill.is_paper is False
        assert fill.status in ("filled", "closed")
        assert fill.order_id == "live-001"

        # Risk manager position registered
        assert len(risk.positions) == 1
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_live_exchange_failure_releases_reserved_slot():
    """If live execute returns None (exchange error), the reserved position is rolled back."""
    from unittest.mock import AsyncMock

    bus = EventBus()
    config = _make_config(paper=False)
    risk = RiskManager(config, bus)

    executor = CEXExecutor.__new__(CEXExecutor)
    executor.config = config
    executor.event_bus = bus
    executor.risk_manager = risk
    executor.exchange_id = "binance"
    executor._order_manager = None
    executor._running = True
    executor._rate_limiter = AsyncMock()
    executor._rate_limiter.acquire = AsyncMock()
    executor._maker_first = True
    executor._post_only = False
    executor._iceberg_threshold_usd = 0.0
    executor._iceberg_chunks = 1
    executor._order_placer = None
    executor._client = None  # forces _live_execute to return None

    bus_task = asyncio.create_task(bus.run())
    bus.subscribe("SIGNAL", executor._handle_signal)

    try:
        await bus.publish("SIGNAL", _make_signal())
        for _ in range(20):
            await asyncio.sleep(0.05)

        # Slot released
        assert len(risk.positions) == 0
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_max_positions_gate_blocks_4th_signal():
    """RiskManager max_open_positions must reject the 4th concurrent signal."""
    bus = EventBus()
    config = _make_config(paper=True)
    risk = RiskManager(config, bus)
    executor = CEXExecutor(config, bus, risk, exchange_id="binance")

    bus_task = asyncio.create_task(bus.run())
    bus.subscribe("SIGNAL", executor._handle_signal)

    fills: list[Any] = []
    bus.subscribe("ORDER_FILLED", lambda payload: fills.append(payload))

    try:
        for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"):
            await bus.publish("SIGNAL", _make_signal(symbol=sym))

        for _ in range(60):
            await asyncio.sleep(0.05)
            if len(fills) >= 3:
                break

        # First 3 fill, 4th rejected
        assert len(fills) == 3
        assert len(risk.positions) == 3
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass
