"""
Integration tests for Startup Reconciliation — restart safety.

Simulates the bot restarting mid-position and verifies:
1. Exchange state is fetched and internal state rebuilt correctly
2. DB positions are cross-referenced with exchange truth
3. Stale DB positions are cleaned up
4. No duplicate positions are created after restart
5. SL/TP are placed for unprotected positions
6. Leverage/margin mismatches trigger safe mode
7. OrderManager state is rebuilt from exchange open orders
8. Double-reconcile is idempotent (no duplicate entries)
9. Safe mode blocks new entries while managing existing positions
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import Config
from core.event_bus import EventBus
from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
from execution.reconciliation import StartupReconciler, ReconciliationResult
from execution.risk_manager import RiskManager, Position
from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrders


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(overrides: dict | None = None) -> Config:
    base = {
        "risk": {
            "initial_equity": 100_000.0,
            "max_open_positions": 5,
            "stop_loss_pct": 0.015,
            "rr_ratio": 2.0,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.10,
        },
        "exchanges": {
            "binance": {
                "enabled": True,
                "leverage": 10,
                "margin_mode": "isolated",
                "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            },
        },
        "system": {"paper_mode": False},
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base:
                base[k].update(v)
            else:
                base[k] = v
    cfg = Config.__new__(Config)
    cfg._data = base
    return cfg


def _make_risk_manager(config: Config, bus: EventBus) -> RiskManager:
    return RiskManager(config=config, event_bus=bus)


def _make_exchange_client(
    positions: list[dict] | None = None,
    orders: list[dict] | None = None,
    balance: dict | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.fetch_balance = AsyncMock(return_value=balance or {
        "total": {"USDT": 100_000.0},
        "free": {"USDT": 90_000.0},
    })
    client.fetch_positions = AsyncMock(return_value=positions or [])
    client.fetch_open_orders = AsyncMock(return_value=orders or [])
    client.set_leverage = AsyncMock()
    client.set_margin_mode = AsyncMock()
    return client


def _make_exchange_position(
    symbol: str = "BTC/USDT:USDT",
    side: str = "long",
    contracts: float = 0.5,
    entry_price: float = 50_000.0,
    unrealized_pnl: float = 250.0,
    leverage: int = 10,
    margin_type: str = "isolated",
    liq_price: float = 45_000.0,
) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "notional": contracts * entry_price,
        "entryPrice": entry_price,
        "unrealizedPnl": unrealized_pnl,
        "leverage": leverage,
        "marginType": margin_type,
        "liquidationPrice": liq_price,
    }


def _make_exchange_order(
    order_id: str = "sl-1",
    symbol: str = "BTC/USDT:USDT",
    order_type: str = "stop_market",
    side: str = "sell",
    amount: float = 0.5,
    stop_price: float = 49_000.0,
    reduce_only: bool = True,
) -> dict:
    return {
        "id": order_id,
        "symbol": symbol,
        "type": order_type,
        "side": side,
        "amount": amount,
        "price": 0,
        "stopPrice": stop_price,
        "status": "open",
        "reduceOnly": reduce_only,
    }


def _make_trade_persistence(db_positions: list[dict] | None = None) -> AsyncMock:
    tp = AsyncMock()
    tp.load_open_positions = AsyncMock(return_value=db_positions or [])
    tp.persist_position_open = AsyncMock()
    tp.persist_position_close = AsyncMock()
    return tp


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RESTART MID-POSITION — Full lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestRestartMidPosition:
    """Simulate bot restart while positions are open on exchange."""

    @pytest.mark.asyncio
    async def test_restart_rebuilds_single_position(self) -> None:
        """Exchange has 1 BTC long → risk manager rebuilt with that position."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
                _make_exchange_order("tp-1", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 52000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.success is True
        assert len(result.exchange_positions) == 1
        key = "binance:BTC/USDT:USDT"
        assert key in rm._positions
        pos = rm._positions[key]
        assert pos.direction == "long"
        assert pos.size == 0.5
        assert pos.entry_price == 50_000.0
        assert pos.stop_loss == 49_000.0
        assert pos.take_profit == 52_000.0

    @pytest.mark.asyncio
    async def test_restart_rebuilds_multiple_positions(self) -> None:
        """Exchange has BTC long + ETH short → both rebuilt."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        positions = [
            _make_exchange_position("BTC/USDT:USDT", "long", 0.5, 50000),
            _make_exchange_position("ETH/USDT:USDT", "short", 5.0, 3000, leverage=10),
        ]
        orders = [
            _make_exchange_order("sl-btc", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            _make_exchange_order("tp-btc", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 52000),
            _make_exchange_order("sl-eth", "ETH/USDT:USDT", "stop_market", "buy", 5.0, 3100),
            _make_exchange_order("tp-eth", "ETH/USDT:USDT", "take_profit_market", "buy", 5.0, 2800),
        ]
        client = _make_exchange_client(positions=positions, orders=orders)

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.success is True
        assert len(rm._positions) == 2
        assert rm._positions["binance:BTC/USDT:USDT"].direction == "long"
        assert rm._positions["binance:ETH/USDT:USDT"].direction == "short"
        assert rm._positions["binance:ETH/USDT:USDT"].size == 5.0

    @pytest.mark.asyncio
    async def test_restart_preserves_existing_state(self) -> None:
        """If local state matches exchange, SL/TP/trailing are preserved."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        # Pre-populate local state (simulating prior run)
        rm._positions["binance:BTC/USDT:USDT"] = Position(
            exchange="binance", symbol="BTC/USDT:USDT", direction="long",
            size=0.5, entry_price=50000, current_price=51000,
            stop_loss=49000, take_profit=53000, open_time=int(time.time()) - 3600,
            trailing_active=True, breakeven_moved=True,
        )

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
                _make_exchange_order("tp-1", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 53000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.success is True
        assert len(result.mismatches) == 0  # No mismatch since position matches
        pos = rm._positions["binance:BTC/USDT:USDT"]
        # Trailing and breakeven should be preserved from old state
        assert pos.trailing_active is True
        assert pos.breakeven_moved is True
        assert "restored BTC/USDT:USDT from previous state" in result.actions_taken


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NO DUPLICATE POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoDuplicatePositions:
    """Ensure restart never creates duplicate position entries."""

    @pytest.mark.asyncio
    async def test_no_duplicate_after_restart(self) -> None:
        """Running reconcile twice yields exactly 1 position, not 2."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result1 = await reconciler.reconcile()
        assert result1.success is True
        assert len(rm._positions) == 1

        # Second call — idempotent guard
        result2 = await reconciler.reconcile()
        assert result2.success is True
        assert len(rm._positions) == 1  # Still exactly 1

    @pytest.mark.asyncio
    async def test_stale_local_position_cleared(self) -> None:
        """Local position with no exchange counterpart is removed and flagged."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        # Stale position in memory (exchange has nothing)
        rm._positions["binance:DOGE/USDT:USDT"] = Position(
            exchange="binance", symbol="DOGE/USDT:USDT", direction="long",
            size=1000, entry_price=0.10, current_price=0.10,
            stop_loss=0.09, take_profit=0.12, open_time=int(time.time()) - 7200,
        )

        client = _make_exchange_client(positions=[], orders=[])

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        # Stale position should be removed
        assert "binance:DOGE/USDT:USDT" not in rm._positions
        assert len(rm._positions) == 0
        # Should be flagged as mismatch
        assert any("stale_position" in m for m in result.mismatches)

    @pytest.mark.asyncio
    async def test_db_persist_on_rebuild(self) -> None:
        """Rebuilt positions are persisted to DB via trade_persistence."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        tp = _make_trade_persistence()

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client, trade_persistence=tp)
        result = await reconciler.reconcile()

        assert result.success is True
        # Position should have been persisted to DB
        tp.persist_position_open.assert_called_once_with(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            direction="long",
            entry_price=50_000.0,
            size=0.5,
        )

    @pytest.mark.asyncio
    async def test_stale_db_position_closed(self) -> None:
        """DB position not on exchange is closed in DB."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        db_positions = [
            {
                "exchange": "binance",
                "symbol": "DOGE/USDT:USDT",
                "direction": "long",
                "entry_price": 0.10,
                "size": 1000,
            },
        ]
        tp = _make_trade_persistence(db_positions=db_positions)

        # Exchange has no positions
        client = _make_exchange_client(positions=[], orders=[])

        reconciler = StartupReconciler(config, bus, rm, client, trade_persistence=tp)
        result = await reconciler.reconcile()

        assert result.success is True
        # Stale DB position should be closed
        tp.persist_position_close.assert_called_once_with(
            exchange="binance",
            symbol="DOGE/USDT:USDT",
            exit_price=0.10,
            pnl=0.0,
            pnl_pct=0.0,
        )
        assert any("closed_stale_db_position" in a for a in result.actions_taken)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAFE MODE — Mismatch handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeMode:
    """Verify safe mode activates on mismatches and blocks new entries."""

    @pytest.mark.asyncio
    async def test_equity_mismatch_triggers_safe_mode(self) -> None:
        """Large equity deviation (>10%) between config and exchange → safe mode."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        rm._equity = 200_000.0  # Local says 200k

        # Exchange says only 50k → >10% deviation
        client = _make_exchange_client(
            balance={"total": {"USDT": 50_000.0}, "free": {"USDT": 45_000.0}},
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.safe_mode is True
        assert reconciler.safe_mode is True
        assert any("equity_mismatch" in m for m in result.mismatches)
        assert rm._circuit_breaker.tripped is True

    @pytest.mark.asyncio
    async def test_unknown_position_triggers_safe_mode(self) -> None:
        """Position on exchange not in local state → mismatch → safe mode."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        # No local state at all

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.safe_mode is True
        assert any("unknown_position" in m for m in result.mismatches)
        # Position still rebuilt despite safe mode — manage existing
        assert "binance:BTC/USDT:USDT" in rm._positions

    @pytest.mark.asyncio
    async def test_safe_mode_blocks_new_signals(self) -> None:
        """After safe mode, approve_signal rejects all new entries."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        rm._equity = 200_000.0

        client = _make_exchange_client(
            balance={"total": {"USDT": 50_000.0}, "free": {"USDT": 45_000.0}},
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        await reconciler.reconcile()

        # Circuit breaker tripped → new signals rejected
        from engine.signal_generator import TradingSignal
        signal = TradingSignal(
            exchange="binance", symbol="ETH/USDT:USDT", direction="long",
            score=0.9, technical_score=0.8, ml_score=0.7, sentiment_score=0.5,
            macro_score=0.3, news_score=0.4, orderbook_score=0.3,
            regime="trending_up", regime_confidence=0.9,
            price=3000, atr=30, stop_loss=2950, take_profit=3100,
            timestamp=int(time.time()),
        )
        ok, reason, size = rm.approve_signal(signal)
        assert ok is False
        assert "circuit_breaker" in reason

    @pytest.mark.asyncio
    async def test_safe_mode_manages_existing_positions(self) -> None:
        """Safe mode still allows position management (SL/TP/trailing)."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        rm._equity = 200_000.0

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
            balance={"total": {"USDT": 50_000.0}, "free": {"USDT": 45_000.0}},
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.safe_mode is True
        # Existing position is still in risk manager and can be managed
        pos = rm._positions.get("binance:BTC/USDT:USDT")
        assert pos is not None
        # manage_position should still work (not blocked by safe mode)
        pos.update_price(51000)
        exit_reason = rm.manage_position(pos)
        # No exit trigger on slight price move — just verifying it runs
        assert exit_reason is None

    @pytest.mark.asyncio
    async def test_direction_mismatch_from_db(self) -> None:
        """DB says short, exchange says long → mismatch caught."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        db_positions = [
            {
                "exchange": "binance",
                "symbol": "BTC/USDT:USDT",
                "direction": "short",  # DB says short
                "entry_price": 50000,
                "size": 0.5,
            },
        ]
        tp = _make_trade_persistence(db_positions=db_positions)

        # Exchange says long
        client = _make_exchange_client(
            positions=[_make_exchange_position("BTC/USDT:USDT", "long", 0.5, 50000)],
            orders=[],
        )

        reconciler = StartupReconciler(config, bus, rm, client, trade_persistence=tp)
        result = await reconciler.reconcile()

        assert any("direction_mismatch" in m for m in result.mismatches)
        assert result.safe_mode is True

    @pytest.mark.asyncio
    async def test_size_mismatch_from_db(self) -> None:
        """DB says size=1.0, exchange says size=0.5 → mismatch caught."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        db_positions = [
            {
                "exchange": "binance",
                "symbol": "BTC/USDT:USDT",
                "direction": "long",
                "entry_price": 50000,
                "size": 1.0,  # DB says 1.0
            },
        ]
        tp = _make_trade_persistence(db_positions=db_positions)

        # Exchange says 0.5 (>5% deviation)
        client = _make_exchange_client(
            positions=[_make_exchange_position("BTC/USDT:USDT", "long", 0.5, 50000)],
            orders=[],
        )

        reconciler = StartupReconciler(config, bus, rm, client, trade_persistence=tp)
        result = await reconciler.reconcile()

        assert any("size_mismatch" in m for m in result.mismatches)
        assert result.safe_mode is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SL/TP PLACEMENT FOR UNPROTECTED POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestProtectiveOrderPlacement:
    """Verify SL/TP is placed immediately for unprotected positions."""

    @pytest.mark.asyncio
    async def test_missing_sl_placed_automatically(self) -> None:
        """Position without SL order → SL/TP placed via order_placer."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        order_placer = AsyncMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}
        order_placer.place_protective_orders = AsyncMock()

        # Position exists, no SL/TP orders on exchange
        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[],  # No protective orders!
        )

        reconciler = StartupReconciler(config, bus, rm, client, order_placer=order_placer)
        result = await reconciler.reconcile()

        assert "BTC/USDT:USDT" in result.positions_without_sl
        order_placer.place_protective_orders.assert_called_once()
        call_kwargs = order_placer.place_protective_orders.call_args
        assert call_kwargs[1]["symbol"] == "BTC/USDT:USDT"
        assert call_kwargs[1]["direction"] == "long"
        assert call_kwargs[1]["quantity"] == 0.5
        assert "placed SL/TP for BTC/USDT:USDT" in result.actions_taken

    @pytest.mark.asyncio
    async def test_existing_sl_tp_read_back(self) -> None:
        """Existing SL/TP on exchange → prices read back into position."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        order_placer = MagicMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 48500),
                _make_exchange_order("tp-1", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 53000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client, order_placer=order_placer)
        result = await reconciler.reconcile()

        pos = rm._positions["binance:BTC/USDT:USDT"]
        assert pos.stop_loss == 48_500.0
        assert pos.take_profit == 53_000.0
        # Protective orders registered in order_placer
        assert "BTC/USDT:USDT" in order_placer._protective
        prot = order_placer._protective["BTC/USDT:USDT"]
        assert prot.sl_order_id == "sl-1"
        assert prot.tp_order_id == "tp-1"

    @pytest.mark.asyncio
    async def test_sl_placement_failure_recorded(self) -> None:
        """If SL placement fails, it's recorded as a mismatch."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        order_placer = AsyncMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}
        order_placer.place_protective_orders = AsyncMock(side_effect=Exception("API error"))

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[],
        )

        reconciler = StartupReconciler(config, bus, rm, client, order_placer=order_placer)
        result = await reconciler.reconcile()

        assert any("sl_placement_failed" in m for m in result.mismatches)
        assert result.safe_mode is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LEVERAGE / MARGIN VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestLeverageMarginReconciliation:
    """Verify leverage/margin settings are checked and corrected."""

    @pytest.mark.asyncio
    async def test_leverage_mismatch_does_not_trigger_set_leverage_with_open_position(self) -> None:
        """Symbols with open positions: don't call set_leverage. Binance returns
        -4161 ('Leverage reduction not supported with open positions') for any
        leverage decrease while the position exists, which previously was caught
        and added to mismatches → safe_mode tripped on every restart whenever
        config leverage drifted. New behaviour: take exchange leverage as ground
        truth, record in actions_taken, do NOT trip safe_mode from this alone."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position(leverage=5)],  # Config expects 10
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        client.set_leverage.assert_not_called()
        assert any("leverage_mismatch_open_position" in a for a in result.actions_taken)
        assert not any("leverage_mismatch" in m for m in result.mismatches)
        # Exchange's actual leverage is recorded as ground truth
        assert result.leverage_settings["BTC/USDT:USDT"]["leverage"] == 5
        assert result.leverage_settings["BTC/USDT:USDT"]["expected_leverage"] == 10

    @pytest.mark.asyncio
    async def test_leverage_uses_risk_default_when_exchange_setting_missing(self) -> None:
        """If exchanges.binance.leverage is absent, expected_leverage falls back
        to risk.default_leverage. Same no-set_leverage rule applies for symbols
        with open positions."""
        config = _make_config({
            "risk": {"default_leverage": 5},
            "exchanges": {"binance": {"leverage": None}},
        })
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position(leverage=3)],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        # New behaviour: don't call set_leverage on open position
        client.set_leverage.assert_not_called()
        # expected_leverage came from risk.default_leverage = 5
        assert result.leverage_settings["BTC/USDT:USDT"]["expected_leverage"] == 5
        # actual exchange leverage (3) is recorded
        assert result.leverage_settings["BTC/USDT:USDT"]["leverage"] == 3

    @pytest.mark.asyncio
    async def test_margin_mode_mismatch_triggers_safe_mode(self) -> None:
        """Config says isolated, exchange says cross → mismatch (can't auto-correct)."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position(margin_type="cross")],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert any("margin_mode_mismatch" in m for m in result.mismatches)
        assert result.safe_mode is True

    @pytest.mark.asyncio
    async def test_leverage_mismatch_alone_does_not_trip_safe_mode(self) -> None:
        """Even if set_leverage WOULD have failed (e.g. -4161 with open position),
        the new code never calls it → no exception → leverage mismatch alone
        does not trip safe_mode. set_leverage AsyncMock is configured here only
        to verify it's never reached."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position(leverage=5)],
            orders=[
                _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            ],
        )
        client.set_leverage = AsyncMock(side_effect=Exception("Leverage change rejected"))

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        # Verify the broken set_leverage path is never taken
        client.set_leverage.assert_not_called()
        # Mismatch is now in actions_taken, not mismatches
        assert any("leverage_mismatch_open_position" in a for a in result.actions_taken)
        assert not any("leverage_mismatch" in m for m in result.mismatches)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORDER MANAGER REBUILD
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderManagerRebuild:
    """Verify OrderManager state rebuilt from exchange open orders."""

    @pytest.mark.asyncio
    async def test_open_orders_restored_to_order_manager(self) -> None:
        """Exchange open orders are registered in OrderManager."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        om = OrderManager(config, bus, rm._circuit_breaker)

        exchange_orders = [
            _make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            _make_exchange_order("tp-1", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 52000),
            _make_exchange_order("limit-1", "ETH/USDT:USDT", "limit", "buy", 5.0, 2900, reduce_only=False),
        ]
        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=exchange_orders,
        )

        reconciler = StartupReconciler(
            config, bus, rm, client, order_manager=om,
        )
        result = await reconciler.reconcile()

        assert "sl-1" in om.orders
        assert "tp-1" in om.orders
        assert "limit-1" in om.orders
        assert om.orders["sl-1"].status == OrderStatus.OPEN
        assert om.orders["sl-1"].reduce_only is True
        assert om.orders["limit-1"].reduce_only is False
        assert any("restored 3 open orders" in a for a in result.actions_taken)

    @pytest.mark.asyncio
    async def test_no_duplicate_orders_on_second_reconcile(self) -> None:
        """Second reconcile does not duplicate orders already in OrderManager."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        om = OrderManager(config, bus, rm._circuit_breaker)

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[_make_exchange_order("sl-1", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000)],
        )

        reconciler = StartupReconciler(config, bus, rm, client, order_manager=om)
        result1 = await reconciler.reconcile()
        assert "sl-1" in om.orders

        # Create a fresh reconciler (simulating separate startup)
        reconciler2 = StartupReconciler(config, bus, rm, client, order_manager=om)
        result2 = await reconciler2.reconcile()

        # Should still have exactly 1 order, not 2
        assert len([k for k in om.orders if k == "sl-1"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. IDEMPOTENCY AND EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestReconciliationIdempotency:
    """Verify reconciliation is idempotent and handles edge cases."""

    @pytest.mark.asyncio
    async def test_double_reconcile_idempotent(self) -> None:
        """Second reconcile on same reconciler is a no-op."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client(
            positions=[_make_exchange_position()],
            orders=[],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result1 = await reconciler.reconcile()
        assert result1.success is True
        assert len(rm._positions) == 1

        # Second call — guard prevents re-run
        result2 = await reconciler.reconcile()
        assert result2.success is True
        # Client should not be called again
        assert client.fetch_balance.call_count == 1

    @pytest.mark.asyncio
    async def test_paper_mode_skips_reconciliation(self) -> None:
        """No client → reconcile returns success immediately."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        reconciler = StartupReconciler(config, bus, rm, client=None)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.safe_mode is False
        assert len(result.exchange_positions) == 0

    @pytest.mark.asyncio
    async def test_exception_triggers_safe_mode(self) -> None:
        """If balance fetch fails, it's recorded as mismatch → safe mode."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = AsyncMock()
        client.fetch_balance = AsyncMock(side_effect=Exception("Network failure"))
        client.fetch_positions = AsyncMock(return_value=[])
        client.fetch_open_orders = AsyncMock(return_value=[])

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        # Balance failure is non-fatal but triggers safe mode
        assert result.success is True
        assert result.safe_mode is True
        assert rm._circuit_breaker.tripped is True
        assert any("balance_fetch_failed" in m for m in result.mismatches)

    @pytest.mark.asyncio
    async def test_total_failure_triggers_safe_mode(self) -> None:
        """Unhandled exception during reconciliation → safe mode, success=False."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = AsyncMock()
        # Make fetch_balance raise and propagate by patching _fetch_balance
        client.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 100_000.0}, "free": {"USDT": 90_000.0},
        })
        # Make _fetch_positions raise an unrecoverable error
        client.fetch_positions = AsyncMock(side_effect=RuntimeError("Total crash"))

        reconciler = StartupReconciler(config, bus, rm, client)
        # Patch to make it propagate
        original = reconciler._fetch_positions
        async def _explode(result):
            raise RuntimeError("Total crash")
        reconciler._fetch_positions = _explode

        result = await reconciler.reconcile()

        assert result.success is False
        assert result.safe_mode is True
        assert rm._circuit_breaker.tripped is True

    @pytest.mark.asyncio
    async def test_reconciliation_result_has_id(self) -> None:
        """Each reconciliation has a unique ID for audit trail."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client()

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert len(result.reconciliation_id) == 12
        assert result.reconciliation_id.isalnum()

    @pytest.mark.asyncio
    async def test_reconciliation_publishes_event(self) -> None:
        """RECONCILIATION_COMPLETE event published on success."""
        config = _make_config()
        bus = EventBus()
        bus.publish = AsyncMock()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client()

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        bus.publish.assert_called()
        # Find the RECONCILIATION_COMPLETE call
        calls = [c for c in bus.publish.call_args_list if c[0][0] == "RECONCILIATION_COMPLETE"]
        assert len(calls) == 1
        payload = calls[0][0][1]
        assert payload.reconciliation_id == result.reconciliation_id
        assert isinstance(payload.safe_mode, bool)

    @pytest.mark.asyncio
    async def test_zero_positions_clean_start(self) -> None:
        """No positions on exchange or locally → clean reconciliation."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = _make_exchange_client()

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.safe_mode is False
        assert len(rm._positions) == 0
        assert len(result.mismatches) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FULL RESTART SIMULATION — End-to-end
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullRestartSimulation:
    """End-to-end restart simulation proving no data loss or duplication."""

    @pytest.mark.asyncio
    async def test_restart_mid_position_full_lifecycle(self) -> None:
        """
        Simulate:
        1. Bot has BTC long + ETH short in memory and DB
        2. Bot crashes
        3. On restart, exchange still has both positions with SL/TP
        4. Reconciliation rebuilds state perfectly
        5. No duplicates, SL/TP preserved, equity correct
        """
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        om = OrderManager(config, bus, rm._circuit_breaker)

        # Simulate DB state from before crash
        db_positions = [
            {
                "exchange": "binance",
                "symbol": "BTC/USDT:USDT",
                "direction": "long",
                "entry_price": 50000,
                "size": 0.5,
            },
            {
                "exchange": "binance",
                "symbol": "ETH/USDT:USDT",
                "direction": "short",
                "entry_price": 3000,
                "size": 5.0,
            },
        ]
        tp = _make_trade_persistence(db_positions=db_positions)

        # Simulate pre-crash in-memory state
        rm._positions["binance:BTC/USDT:USDT"] = Position(
            exchange="binance", symbol="BTC/USDT:USDT", direction="long",
            size=0.5, entry_price=50000, current_price=50500,
            stop_loss=49000, take_profit=53000, open_time=int(time.time()) - 3600,
            trailing_active=True,
        )
        rm._positions["binance:ETH/USDT:USDT"] = Position(
            exchange="binance", symbol="ETH/USDT:USDT", direction="short",
            size=5.0, entry_price=3000, current_price=2950,
            stop_loss=3100, take_profit=2800, open_time=int(time.time()) - 1800,
        )

        # Exchange state (truth)
        exchange_positions = [
            _make_exchange_position("BTC/USDT:USDT", "long", 0.5, 50000),
            _make_exchange_position("ETH/USDT:USDT", "short", 5.0, 3000),
        ]
        exchange_orders = [
            _make_exchange_order("sl-btc", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
            _make_exchange_order("tp-btc", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 53000),
            _make_exchange_order("sl-eth", "ETH/USDT:USDT", "stop_market", "buy", 5.0, 3100),
            _make_exchange_order("tp-eth", "ETH/USDT:USDT", "take_profit_market", "buy", 5.0, 2800),
        ]
        client = _make_exchange_client(
            positions=exchange_positions,
            orders=exchange_orders,
        )

        order_placer = MagicMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}

        reconciler = StartupReconciler(
            config, bus, rm, client,
            order_placer=order_placer,
            trade_persistence=tp,
            order_manager=om,
        )
        result = await reconciler.reconcile()

        # ── Assertions ────────────────────────────────────────────────────
        assert result.success is True
        assert not result.safe_mode  # Everything matches
        assert len(result.mismatches) == 0

        # Exactly 2 positions, no duplicates
        assert len(rm._positions) == 2

        # BTC position preserved with trailing from old state
        btc = rm._positions["binance:BTC/USDT:USDT"]
        assert btc.direction == "long"
        assert btc.size == 0.5
        assert btc.stop_loss == 49_000.0
        assert btc.take_profit == 53_000.0
        assert btc.trailing_active is True

        # ETH position preserved
        eth = rm._positions["binance:ETH/USDT:USDT"]
        assert eth.direction == "short"
        assert eth.size == 5.0
        assert eth.stop_loss == 3_100.0
        assert eth.take_profit == 2_800.0

        # Equity updated from exchange
        assert rm._equity == 100_000.0

        # OrderManager has all 4 protective orders
        assert len(om.orders) == 4

        # Protective orders registered in order_placer
        assert "BTC/USDT:USDT" in order_placer._protective
        assert "ETH/USDT:USDT" in order_placer._protective

        # DB persisted
        assert tp.persist_position_open.call_count == 2

    @pytest.mark.asyncio
    async def test_restart_with_partial_crash_one_position_lost(self) -> None:
        """
        Simulate:
        1. Bot had BTC + ETH positions
        2. Bot crashes, ETH position was closed by SL during downtime
        3. On restart, only BTC exists on exchange, ETH doesn't
        4. Reconciliation cleans up ETH, keeps BTC, enters safe mode
        """
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        # Pre-crash state: both positions
        rm._positions["binance:BTC/USDT:USDT"] = Position(
            exchange="binance", symbol="BTC/USDT:USDT", direction="long",
            size=0.5, entry_price=50000, current_price=50500,
            stop_loss=49000, take_profit=53000, open_time=int(time.time()) - 3600,
        )
        rm._positions["binance:ETH/USDT:USDT"] = Position(
            exchange="binance", symbol="ETH/USDT:USDT", direction="short",
            size=5.0, entry_price=3000, current_price=2950,
            stop_loss=3100, take_profit=2800, open_time=int(time.time()) - 1800,
        )

        # Exchange: only BTC exists (ETH SL was hit during downtime)
        client = _make_exchange_client(
            positions=[_make_exchange_position("BTC/USDT:USDT", "long", 0.5, 50000)],
            orders=[
                _make_exchange_order("sl-btc", "BTC/USDT:USDT", "stop_market", "sell", 0.5, 49000),
                _make_exchange_order("tp-btc", "BTC/USDT:USDT", "take_profit_market", "sell", 0.5, 53000),
            ],
        )

        reconciler = StartupReconciler(config, bus, rm, client)
        result = await reconciler.reconcile()

        # ETH should be removed and flagged as stale
        assert "binance:ETH/USDT:USDT" not in rm._positions
        assert "binance:BTC/USDT:USDT" in rm._positions
        assert len(rm._positions) == 1

        # Stale position mismatch
        assert any("stale_position" in m and "ETH" in m for m in result.mismatches)
        assert result.safe_mode is True  # Mismatch triggers safe mode

        # BTC position preserved
        btc = rm._positions["binance:BTC/USDT:USDT"]
        assert btc.direction == "long"
        assert btc.stop_loss == 49_000.0

    @pytest.mark.asyncio
    async def test_restart_new_position_opened_while_down(self) -> None:
        """
        Someone opened a position via exchange UI while bot was down.
        Bot restarts → unknown position detected → safe mode.
        Position still rebuilt so it can be managed.
        """
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)
        # No pre-existing state

        # Exchange shows a position opened externally
        client = _make_exchange_client(
            positions=[
                _make_exchange_position("SOL/USDT:USDT", "long", 100, 150, leverage=5),
            ],
            orders=[],  # No SL/TP placed!
        )

        order_placer = AsyncMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}
        order_placer.place_protective_orders = AsyncMock()

        reconciler = StartupReconciler(
            config, bus, rm, client, order_placer=order_placer,
        )
        result = await reconciler.reconcile()

        # Unknown position → safe mode
        assert result.safe_mode is True
        assert any("unknown_position" in m and "SOL" in m for m in result.mismatches)

        # Position still rebuilt for management
        assert "binance:SOL/USDT:USDT" in rm._positions

        # SL/TP placed for the unprotected position
        order_placer.place_protective_orders.assert_called_once()
        assert "SOL/USDT:USDT" in result.positions_without_sl
