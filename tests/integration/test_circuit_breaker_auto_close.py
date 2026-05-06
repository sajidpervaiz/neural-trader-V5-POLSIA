"""Circuit-breaker auto-close: when daily loss trips, remaining positions
auto-close and CIRCUIT_BREAKER_TRIPPED + KILL_SWITCH_ACTIVATED events fire.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.risk_manager import RiskManager


def _config() -> Any:
    cfg = MagicMock()
    cfg.paper_mode = True
    risk = {
        "initial_equity": 10_000.0,
        "max_position_size_pct": 0.50,
        "risk_per_trade_pct": 0.50,
        "sizing_method": "risk_based",
        "max_open_positions": 5,
        "default_leverage": 1.0,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.15,
        "stop_loss_pct": 0.015,
        "max_spread_bps": 1000,
        "max_atr_pct": 1.0,
        "cooldown_seconds": 0,
        "session_start_utc": "00:00",
        "session_end_utc": "23:59",
        "atr_sl_multiplier": 1.5,
        "rr_ratio": 2.0,
        "min_signal_score": 0.0,
        "min_risk_reward": 0.0,
        "max_funding_rate_bps": 10000,
        "min_orderbook_depth_usd": 0,
        "max_order_size_usd": 1_000_000,
        "max_total_exposure_pct": 100.0,
        "max_exposure_per_symbol_pct": 100.0,
    }

    def _get(*args, **kwargs):
        if not args:
            return kwargs.get("default")
        if args[0] == "risk":
            return risk if len(args) == 1 else risk.get(args[1], kwargs.get("default"))
        if args[0] == "system" and len(args) >= 2 and args[1] == "paper_mode":
            return True
        return kwargs.get("default", {})

    cfg.get_value = _get
    return cfg


def _signal(symbol: str) -> TradingSignal:
    return TradingSignal(
        exchange="binance", symbol=symbol, direction="long",
        score=0.85, technical_score=0.7, ml_score=0.8,
        sentiment_score=0.5, macro_score=0.5, news_score=0.5, orderbook_score=0.5,
        regime="strong_trend_up", regime_confidence=0.9,
        price=50_000.0, atr=500.0, stop_loss=49_250.0, take_profit=51_500.0,
        timestamp=int(time.time()),
    )


@pytest.mark.asyncio
async def test_circuit_breaker_auto_closes_remaining_positions():
    bus = EventBus()
    risk = RiskManager(_config(), bus)

    bus_task = asyncio.create_task(bus.run())
    breaker_events: list[Any] = []
    kill_events: list[Any] = []
    bus.subscribe("CIRCUIT_BREAKER_TRIPPED", lambda p: breaker_events.append(p))
    bus.subscribe("KILL_SWITCH_ACTIVATED", lambda p: kill_events.append(p))

    try:
        # Open 3 positions
        for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
            sig = _signal(sym)
            ok, _, size_usd = risk.approve_signal(sig)
            assert ok, f"approve failed for {sym}"
            await risk.open_position(sig, size_usd)
        assert len(risk.positions) == 3

        # Close BTC at a -2.5% loss → trips daily-loss breaker (limit 2%)
        await risk.close_position("binance", "BTC/USDT:USDT", 48_750.0)

        # Background auto-close task drains
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not risk.positions and kill_events:
                break

        assert len(breaker_events) == 1
        assert "daily_loss" in breaker_events[0]["reason"]
        assert len(kill_events) == 1
        assert kill_events[0]["trigger"] == "circuit_breaker"
        assert risk.positions == {}
        assert risk._killed is True
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_midnight_reset_clears_daily_loss_trip():
    bus = EventBus()
    risk = RiskManager(_config(), bus)

    bus_task = asyncio.create_task(bus.run())
    try:
        sig = _signal("BTC/USDT:USDT")
        ok, _, size_usd = risk.approve_signal(sig)
        assert ok
        await risk.open_position(sig, size_usd)
        await risk.close_position("binance", "BTC/USDT:USDT", 48_750.0)

        for _ in range(20):
            await asyncio.sleep(0.05)
            if risk._killed and risk._circuit_breaker.tripped:
                break

        assert risk._circuit_breaker.tripped
        assert risk._killed

        risk.reset_daily_losses()

        assert not risk._circuit_breaker.tripped
        assert not risk._killed
    finally:
        await bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass
