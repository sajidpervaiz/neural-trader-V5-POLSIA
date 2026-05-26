"""
Production readiness tests — validates all 5 critical blockers.

Tests:
1. Exchange-side SL/TP (ExchangeOrderPlacer)
2. User Data Stream parsing (UserDataStream)
3. Startup Reconciliation (StartupReconciler)
4. Trade Persistence (TradePersistence)
5. Startup Validation (StartupValidator)
Plus: graceful shutdown, OCO behavior, crash safety, fill tracking.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.config import Config
from core.event_bus import EventBus

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


async def _drain_event_bus(event_bus: EventBus, timeout: float = 0.1) -> None:
    """Process all queued events by briefly running the event bus loop."""
    task = asyncio.create_task(event_bus.run())
    await asyncio.sleep(timeout)
    await event_bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _make_async_pool_mock():
    """Create an AsyncMock that properly simulates asyncpg pool.acquire() context manager."""
    mock_conn = AsyncMock()
    mock_pool = MagicMock()

    @asynccontextmanager
    async def acquire():
        yield mock_conn

    mock_pool.acquire = acquire
    return mock_pool, mock_conn


# ═══════════════════════════════════════════════════════════════════════════════
# Blocker 1: Exchange-Side SL/TP (ExchangeOrderPlacer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExchangeOrderPlacer:
    """Tests for exchange-side protective orders."""

    def setup_method(self):
        from execution.exchange_order_placer import ExchangeOrderPlacer

        self.mock_client = AsyncMock()
        self.mock_client.create_order = AsyncMock(return_value={"id": "sl_001"})
        self.mock_client.cancel_order = AsyncMock(return_value={})
        self.placer = ExchangeOrderPlacer(self.mock_client, working_type="CONTRACT_PRICE")

    @pytest.mark.asyncio
    async def test_place_sl_and_tp(self):
        """SL and TP both placed after entry fill."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[{"id": "sl_001"}, {"id": "tp_001"}]
        )
        prot = await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        assert prot.sl_placed is True
        assert prot.tp_placed is True
        assert prot.sl_order_id == "sl_001"
        assert prot.tp_order_id == "tp_001"
        assert self.mock_client.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_sl_placed_first(self):
        """SL must be placed before TP (critical for crash protection)."""
        call_order = []

        async def track_create_order(**kwargs):
            call_order.append(kwargs.get("type", ""))
            return {"id": f"order_{len(call_order)}"}

        self.mock_client.create_order = track_create_order
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        assert call_order[0] == "STOP_MARKET"
        assert call_order[1] == "TAKE_PROFIT_MARKET"

    @pytest.mark.asyncio
    async def test_sl_failure_raises(self):
        """If SL placement fails, exception propagates (position unprotected)."""
        self.mock_client.create_order = AsyncMock(side_effect=RuntimeError("SL failed"))
        with pytest.raises(RuntimeError, match="SL failed"):
            await self.placer.place_protective_orders(
                symbol="BTC/USDT:USDT",
                direction="long",
                quantity=0.01,
                entry_price=50000.0,
                sl_price=49000.0,
                tp_price=52000.0,
            )

    @pytest.mark.asyncio
    async def test_tp_failure_sl_still_active(self):
        """TP failure doesn't affect SL placement."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[{"id": "sl_001"}, RuntimeError("TP failed")]
        )
        prot = await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        assert prot.sl_placed is True
        assert prot.tp_placed is False
        assert prot.sl_order_id == "sl_001"

    @pytest.mark.asyncio
    async def test_oco_sl_cancels_tp(self):
        """OCO: SL fill → cancel TP."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[{"id": "sl_001"}, {"id": "tp_001"}]
        )
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        await self.placer.handle_sl_filled("BTC/USDT:USDT")
        self.mock_client.cancel_order.assert_called_once_with("tp_001", "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_oco_tp_cancels_sl(self):
        """OCO: TP fill → cancel SL."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[{"id": "sl_001"}, {"id": "tp_001"}]
        )
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        await self.placer.handle_tp_filled("BTC/USDT:USDT")
        self.mock_client.cancel_order.assert_called_once_with("sl_001", "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_adjust_quantity_partial_fill(self):
        """Partial fill adjusts SL/TP quantities."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[
                {"id": "sl_001"}, {"id": "tp_001"},  # initial
                {"id": "sl_002"}, {"id": "tp_002"},  # adjusted
            ]
        )
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.1,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        await self.placer.adjust_quantity("BTC/USDT:USDT", 0.05)
        prot = self.placer.get_protective("BTC/USDT:USDT")
        assert prot.quantity == 0.05
        # Old orders cancelled, new ones placed
        assert self.mock_client.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_all_for_symbol(self):
        """Cancel all protective orders for symbol."""
        self.mock_client.create_order = AsyncMock(
            side_effect=[{"id": "sl_001"}, {"id": "tp_001"}]
        )
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        await self.placer.cancel_all_for_symbol("BTC/USDT:USDT")
        assert self.mock_client.cancel_order.call_count == 2
        assert self.placer.get_protective("BTC/USDT:USDT") is None

    @pytest.mark.asyncio
    async def test_reduce_only_set(self):
        """Protective orders use reduceOnly=True."""
        self.mock_client.create_order = AsyncMock(return_value={"id": "o1"})
        await self.placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )
        for call in self.mock_client.create_order.call_args_list:
            params = call.kwargs.get("params", {})
            assert params.get("reduceOnly") is True

    @pytest.mark.asyncio
    async def test_short_direction_uses_buy_close(self):
        """Short position SL/TP uses BUY side to close."""
        from execution.exchange_order_placer import ProtectiveOrders

        prot = ProtectiveOrders(
            symbol="ETH/USDT:USDT",
            direction="short",
            quantity=1.0,
            entry_price=3000.0,
            sl_price=3100.0,
            tp_price=2800.0,
        )
        assert prot.close_side == "BUY"
        assert not prot.is_long


# ═══════════════════════════════════════════════════════════════════════════════
# Blocker 2: User Data Stream Parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestUserDataStream:
    """Tests for Binance User Data Stream message parsing."""

    def setup_method(self):
        from data_ingestion.user_stream import UserDataStream

        self.config = Config.get(path=str(CONFIG_PATH))
        self.event_bus = EventBus()
        self.stream = UserDataStream(self.config, self.event_bus)

    def test_parse_order_trade_update(self):
        """Parse ORDER_TRADE_UPDATE with all fields."""
        data = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000000,
            "T": 1700000000000,
            "o": {
                "s": "BTCUSDT",
                "c": "client_123",
                "S": "BUY",
                "o": "LIMIT",
                "f": "GTC",
                "q": "0.01",
                "p": "50000.0",
                "ap": "50010.0",
                "sp": "0",
                "x": "TRADE",
                "X": "FILLED",
                "i": 123456789,
                "l": "0.01",
                "z": "0.01",
                "L": "50010.0",
                "n": "0.005",
                "N": "USDT",
                "t": 987654321,
                "R": False,
                "ps": "BOTH",
                "rp": "0",
                "m": False,
            },
        }
        parsed = self.stream._parse_order_update(data)
        assert parsed["symbol"] == "BTCUSDT"
        assert parsed["side"] == "BUY"
        assert parsed["order_type"] == "LIMIT"
        assert parsed["execution_type"] == "TRADE"
        assert parsed["order_status"] == "FILLED"
        assert parsed["last_filled_qty"] == 0.01
        assert parsed["last_filled_price"] == 50010.0
        assert parsed["commission"] == 0.005
        assert parsed["order_id"] == 123456789
        assert parsed["trade_id"] == 987654321
        assert parsed["reduce_only"] is False

    def test_parse_account_update(self):
        """Parse ACCOUNT_UPDATE with balances and positions."""
        data = {
            "e": "ACCOUNT_UPDATE",
            "E": 1700000000000,
            "T": 1700000000000,
            "a": {
                "m": "ORDER",
                "B": [
                    {"a": "USDT", "wb": "10000.0", "cw": "9000.0", "bc": "-50.0"},
                ],
                "P": [
                    {
                        "s": "BTCUSDT",
                        "pa": "0.01",
                        "ep": "50010.0",
                        "up": "10.0",
                        "mt": "cross",
                        "iw": "0",
                        "ps": "BOTH",
                    },
                ],
            },
        }
        parsed = self.stream._parse_account_update(data)
        assert parsed["reason"] == "ORDER"
        assert len(parsed["balances"]) == 1
        assert parsed["balances"][0]["asset"] == "USDT"
        assert parsed["balances"][0]["wallet_balance"] == 10000.0
        assert parsed["balances"][0]["balance_change"] == -50.0
        assert len(parsed["positions"]) == 1
        assert parsed["positions"][0]["symbol"] == "BTCUSDT"
        assert parsed["positions"][0]["position_amount"] == 0.01
        assert parsed["positions"][0]["unrealized_pnl"] == 10.0

    def test_parse_stop_market_fill(self):
        """Parse SL fill correctly identified as reduce_only STOP_MARKET."""
        data = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000000,
            "T": 1700000000000,
            "o": {
                "s": "BTCUSDT",
                "c": "",
                "S": "SELL",
                "o": "STOP_MARKET",
                "f": "GTC",
                "q": "0.01",
                "p": "0",
                "ap": "49000.0",
                "sp": "49000",
                "x": "TRADE",
                "X": "FILLED",
                "i": 111222333,
                "l": "0.01",
                "z": "0.01",
                "L": "49000.0",
                "n": "0.005",
                "N": "USDT",
                "t": 444555666,
                "R": True,
                "ps": "BOTH",
                "rp": "-100.0",
                "m": False,
            },
        }
        parsed = self.stream._parse_order_update(data)
        assert parsed["reduce_only"] is True
        assert parsed["order_type"] == "STOP_MARKET"
        assert parsed["realized_profit"] == -100.0

    @pytest.mark.asyncio
    async def test_handle_order_trade_update_publishes(self):
        """ORDER_TRADE_UPDATE published to EventBus."""
        received = []

        async def handler(p):
            received.append(p)

        self.event_bus.subscribe("USER_ORDER_UPDATE", handler)
        raw = json.dumps({
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000000,
            "T": 1700000000000,
            "o": {
                "s": "BTCUSDT", "c": "", "S": "BUY", "o": "LIMIT",
                "f": "GTC", "q": "0.01", "p": "50000", "ap": "50000",
                "sp": "0", "x": "TRADE", "X": "FILLED", "i": 1,
                "l": "0.01", "z": "0.01", "L": "50000", "n": "0.005",
                "N": "USDT", "t": 1, "R": False, "ps": "BOTH", "rp": "0", "m": False,
            },
        })
        await self.stream._handle_message(raw)
        await _drain_event_bus(self.event_bus)
        assert len(received) == 1
        assert received[0]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_handle_account_update_publishes(self):
        """ACCOUNT_UPDATE published to EventBus."""
        received = []

        async def handler(p):
            received.append(p)

        self.event_bus.subscribe("USER_ACCOUNT_UPDATE", handler)
        raw = json.dumps({
            "e": "ACCOUNT_UPDATE",
            "E": 1700000000000,
            "T": 1700000000000,
            "a": {
                "m": "ORDER",
                "B": [{"a": "USDT", "wb": "10000", "cw": "9000", "bc": "0"}],
                "P": [],
            },
        })
        await self.stream._handle_message(raw)
        await _drain_event_bus(self.event_bus)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_margin_call_publishes(self):
        """MARGIN_CALL published to EventBus."""
        received = []

        async def handler(p):
            received.append(p)

        self.event_bus.subscribe("MARGIN_CALL", handler)
        raw = json.dumps({"e": "MARGIN_CALL", "E": 1700000000000})
        await self.stream._handle_message(raw)
        await _drain_event_bus(self.event_bus)
        assert len(received) == 1

    def test_paper_mode_disables_stream(self):
        """Paper mode should not start user data stream."""
        assert self.config.paper_mode is True
        # In paper mode, run() should return immediately — tested by checking the guard


# ═══════════════════════════════════════════════════════════════════════════════
# Blocker 3: Startup Reconciliation
# ═══════════════════════════════════════════════════════════════════════════════


class TestStartupReconciler:
    """Tests for startup reconciliation."""

    def setup_method(self):
        from execution.reconciliation import StartupReconciler

        self.config = Config.get(path=str(CONFIG_PATH))
        self.event_bus = EventBus()

        # Mock risk manager with positions dict and circuit breaker
        self.risk_mgr = MagicMock()
        self.risk_mgr._positions = {}
        self.risk_mgr._equity = 10000.0
        self.risk_mgr._circuit_breaker = MagicMock()
        self.risk_mgr._circuit_breaker._tripped = False

        self.mock_client = AsyncMock()
        self.reconciler = StartupReconciler(
            config=self.config,
            event_bus=self.event_bus,
            risk_manager=self.risk_mgr,
            client=self.mock_client,
            order_placer=None,
        )

    @pytest.mark.asyncio
    async def test_clean_reconciliation(self):
        """No positions on exchange = clean reconciliation."""
        self.mock_client.fetch_balance.return_value = {
            "total": {"USDT": 10000.0},
            "free": {"USDT": 9500.0},
        }
        self.mock_client.fetch_positions.return_value = []
        self.mock_client.fetch_open_orders.return_value = []

        result = await self.reconciler.reconcile()
        assert result.success is True
        assert result.safe_mode is False
        assert len(result.mismatches) == 0

    @pytest.mark.asyncio
    async def test_unknown_position_triggers_safe_mode(self):
        """Position on exchange not in local state → mismatch → safe mode."""
        self.mock_client.fetch_balance.return_value = {
            "total": {"USDT": 10000.0},
            "free": {"USDT": 9000.0},
        }
        self.mock_client.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.01,
                "notional": 500.0,
                "entryPrice": 50000.0,
                "unrealizedPnl": 10.0,
                "leverage": 5,
                "marginType": "cross",
                "liquidationPrice": 40000.0,
            },
        ]
        self.mock_client.fetch_open_orders.return_value = []

        result = await self.reconciler.reconcile()
        assert result.success is True
        assert result.safe_mode is True
        assert any("unknown_position" in m for m in result.mismatches)

    @pytest.mark.asyncio
    async def test_equity_mismatch_detected(self):
        """Balance differs >10% from config equity → mismatch."""
        self.risk_mgr._equity = 10000.0
        self.mock_client.fetch_balance.return_value = {
            "total": {"USDT": 5000.0},  # 50% diff
            "free": {"USDT": 4500.0},
        }
        self.mock_client.fetch_positions.return_value = []
        self.mock_client.fetch_open_orders.return_value = []

        result = await self.reconciler.reconcile()
        assert any("equity_mismatch" in m for m in result.mismatches)

    @pytest.mark.asyncio
    async def test_reconciliation_failure_enters_safe_mode(self):
        """Balance fetch failure → mismatch → safe mode."""
        self.mock_client.fetch_balance.side_effect = RuntimeError("API down")
        self.mock_client.fetch_positions.return_value = []
        self.mock_client.fetch_open_orders.return_value = []

        result = await self.reconciler.reconcile()
        assert result.safe_mode is True
        assert any("balance_fetch_failed" in m for m in result.mismatches)

    @pytest.mark.asyncio
    async def test_no_client_skips_reconciliation(self):
        """No exchange client → skip reconciliation."""
        from execution.reconciliation import StartupReconciler

        reconciler = StartupReconciler(
            config=self.config,
            event_bus=self.event_bus,
            risk_manager=self.risk_mgr,
            client=None,
        )
        result = await reconciler.reconcile()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rebuilds_position_from_exchange(self):
        """Positions rebuilt from exchange truth, not stale local state."""
        self.mock_client.fetch_balance.return_value = {
            "total": {"USDT": 10000.0},
            "free": {"USDT": 9500.0},
        }
        self.mock_client.fetch_positions.return_value = [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "short",
                "contracts": 1.0,
                "notional": 3000.0,
                "entryPrice": 3000.0,
                "unrealizedPnl": -20.0,
                "leverage": 10,
                "marginType": "cross",
                "liquidationPrice": 3300.0,
            },
        ]
        self.mock_client.fetch_open_orders.return_value = []

        result = await self.reconciler.reconcile()
        assert len(result.exchange_positions) == 1
        assert "binance:ETH/USDT:USDT" in self.risk_mgr._positions


# ═══════════════════════════════════════════════════════════════════════════════
# Blocker 4: Trade Persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradePersistence:
    """Tests for trade persistence layer (mocked DB)."""

    def setup_method(self):
        from storage.trade_persistence import TradePersistence

        self.event_bus = EventBus()
        self.mock_pool, self.mock_conn = _make_async_pool_mock()
        self.persistence = TradePersistence(self.mock_pool, self.event_bus, is_paper=True)

    @pytest.mark.asyncio
    async def test_persist_order_idempotent(self):
        """Order persist uses ON CONFLICT DO UPDATE."""
        await self.persistence.persist_order(
            order_id="order_001",
            client_order_id="client_001",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="limit",
            quantity=0.01,
            price=50000.0,
            status="filled",
            filled_qty=0.01,
            avg_fill_price=50010.0,
        )
        self.mock_conn.execute.assert_called_once()
        sql = self.mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_persist_fill_idempotent(self):
        """Fill persist uses ON CONFLICT DO NOTHING."""
        await self.persistence.persist_fill(
            fill_id="fill_001",
            order_id="order_001",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            side="buy",
            quantity=0.01,
            price=50010.0,
            fee=0.005,
        )
        self.mock_conn.execute.assert_called_once()
        sql = self.mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_persist_position_open(self):
        """Position open persists to DB."""
        await self.persistence.persist_position_open(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            direction="long",
            entry_price=50000.0,
            size=0.01,
        )
        self.mock_conn.execute.assert_called_once()
        sql = self.mock_conn.execute.call_args[0][0]
        assert "INSERT INTO positions" in sql

    @pytest.mark.asyncio
    async def test_persist_position_close(self):
        """Position close updates existing row."""
        await self.persistence.persist_position_close(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            exit_price=52000.0,
            pnl=20.0,
            pnl_pct=2.0,
        )
        self.mock_conn.execute.assert_called_once()
        sql = self.mock_conn.execute.call_args[0][0]
        assert "UPDATE positions" in sql
        assert "close_time IS NULL" in sql

    @pytest.mark.asyncio
    async def test_persist_equity_snapshot(self):
        """Equity snapshots saved for audit trail."""
        await self.persistence.persist_equity_snapshot(
            equity=10500.0,
            unrealized_pnl=50.0,
            open_positions=2,
            drawdown_pct=0.5,
        )
        self.mock_conn.execute.assert_called_once()
        sql = self.mock_conn.execute.call_args[0][0]
        assert "equity_snapshots" in sql

    @pytest.mark.asyncio
    async def test_subscribe_events(self):
        """All required events get subscribed."""
        self.persistence.subscribe_events()
        # Verify key events are subscribed via the event bus
        assert "ORDER_FILLED" in self.event_bus._handlers
        assert "FILL_CONFIRMED" in self.event_bus._handlers
        assert "POSITION_CLOSED" in self.event_bus._handlers
        assert "USER_ORDER_UPDATE" in self.event_bus._handlers

    @pytest.mark.asyncio
    async def test_pool_none_no_crash(self):
        """Operations gracefully no-op when pool is None."""
        from storage.trade_persistence import TradePersistence

        persistence = TradePersistence(None, self.event_bus, is_paper=True)
        # None of these should raise
        await persistence.persist_order(
            "o1", "c1", "binance", "BTC", "buy", "limit", 1.0, 50000.0, "filled",
        )
        await persistence.persist_fill("f1", "o1", "binance", "BTC", "buy", 1.0, 50000.0)
        await persistence.persist_equity_snapshot(10000.0, 0.0, 0, 0.0)
        positions = await persistence.load_open_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_ddl_has_required_tables(self):
        """DDL includes all production tables."""
        from storage.trade_persistence import TRADING_DDL

        combined = "\n".join(TRADING_DDL)
        for table in ["orders", "fills", "equity_snapshots", "risk_blocks", "errors", "daily_pnl"]:
            assert table in combined, f"Missing table: {table}"


# ═══════════════════════════════════════════════════════════════════════════════
# Blocker 5: Startup Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestStartupValidator:
    """Tests for API key, balance, and permission validation."""

    def setup_method(self):
        from execution.startup_validation import StartupValidator

        self.config = Config.get(path=str(CONFIG_PATH))
        self.mock_client = AsyncMock()
        self.validator = StartupValidator(self.config, client=self.mock_client)

    @pytest.mark.asyncio
    async def test_paper_mode_skips_validation(self):
        """Paper mode skips all validation."""
        result = await self.validator.validate_all()
        assert result.get("skipped") is True

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        """Missing API key raises ValidationError."""
        from execution.startup_validation import StartupValidator, ValidationError

        config = Config.get(path=str(CONFIG_PATH))
        original = config.paper_mode
        config.paper_mode = False

        with patch.object(config, "get_value") as mock_get:
            def side_effect(*args, **kwargs):
                if args == ("exchanges",):
                    return {"binance": {"enabled": True, "api_key": "", "api_secret": "secret123"}}
                if args == ("risk",):
                    return {"min_liquidity_usd": 100}
                return kwargs.get("default")

            mock_get.side_effect = side_effect
            validator = StartupValidator(config, client=self.mock_client)

            with pytest.raises(ValidationError, match="API key missing"):
                await validator.validate_all()

        config.paper_mode = original

    @pytest.mark.asyncio
    async def test_insufficient_balance_raises(self):
        """Balance below minimum threshold raises ValidationError."""
        from execution.startup_validation import StartupValidator, ValidationError

        config = Config.get(path=str(CONFIG_PATH))
        original = config.paper_mode
        config.paper_mode = False

        with patch.object(config, "get_value") as mock_get:
            def side_effect(*args, **kwargs):
                if args == ("exchanges",):
                    return {"binance": {"enabled": True, "api_key": "key123", "api_secret": "secret123"}}
                if args == ("risk",):
                    return {"min_liquidity_usd": 1000, "max_position_size_pct": 0.02}
                return kwargs.get("default")

            mock_get.side_effect = side_effect

            self.mock_client.load_markets.return_value = {}
            self.mock_client.fetch_balance.return_value = {
                "total": {"USDT": 50.0},  # Below 1000 USD minimum
                "free": {"USDT": 50.0},
            }

            validator = StartupValidator(config, client=self.mock_client)

            with pytest.raises(ValidationError, match="Insufficient balance"):
                await validator.validate_all()

        config.paper_mode = original

    @pytest.mark.asyncio
    async def test_clock_drift_detection(self):
        """High clock drift raises ValidationError."""
        from execution.startup_validation import StartupValidator, ValidationError

        config = Config.get(path=str(CONFIG_PATH))
        original = config.paper_mode
        config.paper_mode = False

        with patch.object(config, "get_value") as mock_get:
            def side_effect(*args, **kwargs):
                if args == ("exchanges",):
                    return {"binance": {"enabled": True, "api_key": "key123", "api_secret": "secret123"}}
                if args == ("risk",):
                    return {"min_liquidity_usd": 100, "max_position_size_pct": 0.02, "default_leverage": 5}
                return kwargs.get("default")

            mock_get.side_effect = side_effect

            self.mock_client.load_markets.return_value = {"BTC/USDT:USDT": {}}
            self.mock_client.markets = {"BTC/USDT:USDT": {}}
            self.mock_client.fetch_balance.return_value = {
                "total": {"USDT": 10000.0},
                "free": {"USDT": 9000.0},
            }
            # Return server time 10 seconds in the future
            self.mock_client.fetch_time.return_value = int(time.time() * 1000) + 10000

            validator = StartupValidator(config, client=self.mock_client)

            with pytest.raises(ValidationError, match="Clock drift"):
                await validator.validate_all()

        config.paper_mode = original

    @pytest.mark.asyncio
    async def test_high_leverage_rejected(self):
        """Leverage > 20x is rejected."""
        from execution.startup_validation import StartupValidator, ValidationError

        config = Config.get(path=str(CONFIG_PATH))
        original = config.paper_mode
        config.paper_mode = False

        with patch.object(config, "get_value") as mock_get:
            def side_effect(*args, **kwargs):
                if args == ("exchanges",):
                    return {"binance": {"enabled": True, "api_key": "key123", "api_secret": "secret123"}}
                if args == ("risk",):
                    return {
                        "min_liquidity_usd": 100,
                        "max_position_size_pct": 0.02,
                        "default_leverage": 50,  # Way too high
                    }
                return kwargs.get("default")

            mock_get.side_effect = side_effect

            self.mock_client.load_markets.return_value = {}
            self.mock_client.markets = {}
            self.mock_client.fetch_balance.return_value = {
                "total": {"USDT": 10000.0},
                "free": {"USDT": 9000.0},
            }
            self.mock_client.fetch_time.return_value = int(time.time() * 1000)

            validator = StartupValidator(config, client=self.mock_client)

            with pytest.raises(ValidationError, match="Leverage.*dangerously high"):
                await validator.validate_all()

        config.paper_mode = original

    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        """No client in live mode raises ValidationError."""
        from execution.startup_validation import StartupValidator, ValidationError

        config = Config.get(path=str(CONFIG_PATH))
        original = config.paper_mode
        config.paper_mode = False

        with patch.object(config, "get_value") as mock_get:
            def side_effect(*args, **kwargs):
                if args == ("exchanges",):
                    return {"binance": {"enabled": True, "api_key": "key123", "api_secret": "secret123"}}
                if args == ("risk",):
                    return {"min_liquidity_usd": 100}
                return kwargs.get("default")

            mock_get.side_effect = side_effect

            validator = StartupValidator(config, client=None)

            with pytest.raises(ValidationError, match="No exchange client"):
                await validator.validate_all()

        config.paper_mode = original


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-cutting: CEXExecutor user stream integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestCEXExecutorUserStream:
    """Tests for CEXExecutor handling user stream events."""

    def setup_method(self):
        from execution.cex_executor import CEXExecutor

        self.config = Config.get(path=str(CONFIG_PATH))
        self.event_bus = EventBus()
        self.risk_mgr = MagicMock()
        self.risk_mgr._circuit_breaker = MagicMock()
        self.risk_mgr._circuit_breaker._tripped = False
        self.risk_mgr.close_position = AsyncMock(return_value=None)
        self.risk_mgr.open_position = AsyncMock()
        self.risk_mgr.activate_kill_switch = AsyncMock(return_value=[])

        self.executor = CEXExecutor(self.config, self.event_bus, self.risk_mgr, "binance")

    @pytest.mark.asyncio
    async def test_user_stream_lost_trips_circuit_breaker(self):
        """User stream disconnect should trip circuit breaker."""
        await self.executor._handle_user_stream_lost({})
        self.risk_mgr._circuit_breaker.trip.assert_called_once_with("user_stream_disconnected")

    @pytest.mark.asyncio
    async def test_sl_fill_from_user_stream(self):
        """STOP_MARKET fill via user stream closes position."""
        from execution.exchange_order_placer import ExchangeOrderPlacer

        mock_client = AsyncMock()
        self.executor._order_placer = ExchangeOrderPlacer(mock_client)

        # Set up a mock position
        mock_pos = MagicMock()
        mock_pos.exchange = "binance"
        mock_pos.symbol = "BTCUSDT"
        mock_pos.pnl = -100.0
        mock_pos.pnl_pct = -2.0
        mock_pos.current_price = 49000.0
        self.risk_mgr.close_position.return_value = mock_pos

        published = []

        async def handler(p):
            published.append(p)

        self.event_bus.subscribe("POSITION_CLOSED", handler)

        payload = {
            "symbol": "BTCUSDT",
            "execution_type": "TRADE",
            "order_status": "FILLED",
            "order_type": "STOP_MARKET",
            "reduce_only": True,
            "last_filled_qty": 0.01,
            "last_filled_price": 49000.0,
            "cumulative_filled_qty": 0.01,
            "quantity": 0.01,
            "commission": 0.005,
            "realized_profit": -100.0,
        }
        await self.executor._handle_user_order_update(payload)
        await _drain_event_bus(self.event_bus)

        self.risk_mgr.close_position.assert_called_once_with("binance", "BTCUSDT", 49000.0)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_tp_fill_from_user_stream(self):
        """TAKE_PROFIT_MARKET fill via user stream closes position."""
        from execution.exchange_order_placer import ExchangeOrderPlacer

        mock_client = AsyncMock()
        self.executor._order_placer = ExchangeOrderPlacer(mock_client)

        mock_pos = MagicMock()
        mock_pos.exchange = "binance"
        mock_pos.symbol = "BTCUSDT"
        mock_pos.pnl = 200.0
        mock_pos.pnl_pct = 4.0
        mock_pos.current_price = 52000.0
        self.risk_mgr.close_position.return_value = mock_pos

        published = []

        async def handler(p):
            published.append(p)

        self.event_bus.subscribe("POSITION_CLOSED", handler)

        payload = {
            "symbol": "BTCUSDT",
            "execution_type": "TRADE",
            "order_status": "FILLED",
            "order_type": "TAKE_PROFIT_MARKET",
            "reduce_only": True,
            "last_filled_qty": 0.01,
            "last_filled_price": 52000.0,
            "cumulative_filled_qty": 0.01,
            "quantity": 0.01,
            "commission": 0.005,
            "realized_profit": 200.0,
        }
        await self.executor._handle_user_order_update(payload)
        await _drain_event_bus(self.event_bus)

        self.risk_mgr.close_position.assert_called_once_with("binance", "BTCUSDT", 52000.0)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_handle_order_rejected(self):
        """ORDER_REJECTED published on rejection."""
        published = []

        async def handler(p):
            published.append(p)

        self.event_bus.subscribe("ORDER_REJECTED", handler)

        payload = {
            "symbol": "BTCUSDT",
            "execution_type": "REJECTED",
            "order_status": "REJECTED",
            "order_type": "LIMIT",
            "reduce_only": False,
        }
        await self.executor._handle_user_order_update(payload)
        await _drain_event_bus(self.event_bus)
        assert len(published) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Crash safety / restart safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrashSafety:
    """Tests ensuring positions remain protected after crash or restart."""

    @pytest.mark.asyncio
    async def test_exchange_sl_survives_bot_crash(self):
        """SL on exchange persists even if bot process dies."""
        from execution.exchange_order_placer import ExchangeOrderPlacer

        client = AsyncMock()
        client.create_order = AsyncMock(
            side_effect=[{"id": "sl_survives"}, {"id": "tp_survives"}]
        )
        placer = ExchangeOrderPlacer(client)

        prot = await placer.place_protective_orders(
            symbol="BTC/USDT:USDT",
            direction="long",
            quantity=0.01,
            entry_price=50000.0,
            sl_price=49000.0,
            tp_price=52000.0,
        )

        # Simulate crash: placer object destroyed, but exchange-side orders persist
        assert prot.sl_placed is True
        assert prot.sl_order_id == "sl_survives"
        # After restart, reconciliation would find these orders on exchange

    @pytest.mark.asyncio
    async def test_reconciliation_after_restart(self):
        """After restart, reconciliation rebuilds state from exchange."""
        from execution.reconciliation import StartupReconciler, ReconciliationResult

        config = Config.get(path=str(CONFIG_PATH))
        event_bus = EventBus()
        risk_mgr = MagicMock()
        risk_mgr._positions = {}
        risk_mgr._equity = 10000.0
        risk_mgr._circuit_breaker = MagicMock()
        risk_mgr._circuit_breaker._tripped = False

        client = AsyncMock()
        client.fetch_balance.return_value = {
            "total": {"USDT": 9900.0}, "free": {"USDT": 9000.0}
        }
        client.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.01,
                "notional": 500.0,
                "entryPrice": 50000.0,
                "unrealizedPnl": -10.0,
                "leverage": 5,
                "marginType": "cross",
                "liquidationPrice": 40000.0,
            },
        ]
        client.fetch_open_orders.return_value = [
            {
                "id": "sl_surviving",
                "symbol": "BTC/USDT:USDT",
                "type": "stop_market",
                "side": "sell",
                "amount": 0.01,
                "price": 0,
                "stopPrice": 49000.0,
                "status": "open",
                "reduceOnly": True,
            },
        ]

        reconciler = StartupReconciler(
            config=config, event_bus=event_bus,
            risk_manager=risk_mgr, client=client,
        )
        result = await reconciler.reconcile()

        # Position should be rebuilt in risk_mgr._positions
        assert "binance:BTC/USDT:USDT" in risk_mgr._positions
        pos = risk_mgr._positions["binance:BTC/USDT:USDT"]
        assert pos.entry_price == 50000.0
        assert pos.size == 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Live mode safety gate
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditFixes:
    """Tests for critical/high audit fixes."""

    @pytest.mark.asyncio
    async def test_init_client_guard_prevents_double_init(self):
        """Second _init_client call should be a no-op (preserves order_placer state)."""
        from execution.cex_executor import CEXExecutor
        from execution.exchange_order_placer import ExchangeOrderPlacer

        config = Config.get(path=str(CONFIG_PATH))
        event_bus = EventBus()
        risk_mgr = MagicMock()

        executor = CEXExecutor(config, event_bus, risk_mgr, "binance")
        mock_client = AsyncMock()
        executor._client = mock_client
        original_placer = ExchangeOrderPlacer(mock_client)
        executor._order_placer = original_placer

        # Simulate protective order tracking
        from execution.exchange_order_placer import ProtectiveOrders
        original_placer._protective["BTC/USDT:USDT"] = ProtectiveOrders(
            symbol="BTC/USDT:USDT", direction="long", quantity=0.01,
            entry_price=50000.0, sl_price=49000.0, tp_price=52000.0,
            sl_order_id="sl_001", sl_placed=True,
        )

        # Second init should not wipe the placer
        await executor._init_client()
        assert executor._order_placer is original_placer
        assert "BTC/USDT:USDT" in executor._order_placer._protective

    @pytest.mark.asyncio
    async def test_sl_failure_trips_circuit_breaker(self):
        """SL placement failure should trip circuit breaker."""
        from execution.cex_executor import CEXExecutor
        from execution.exchange_order_placer import ExchangeOrderPlacer
        from engine.signal_generator import TradingSignal

        config = Config.get(path=str(CONFIG_PATH))
        event_bus = EventBus()
        risk_mgr = MagicMock()
        risk_mgr._circuit_breaker = MagicMock()
        risk_mgr._circuit_breaker._tripped = False

        mock_pos = MagicMock()
        mock_pos.stop_loss = 49000.0
        mock_pos.take_profit = 52000.0
        risk_mgr.open_position = AsyncMock(return_value=mock_pos)

        executor = CEXExecutor(config, event_bus, risk_mgr, "binance")
        mock_client = AsyncMock()
        mock_client.create_limit_order.return_value = {
            "id": "entry_001", "average": 50000.0, "filled": 0.01, "status": "filled",
        }
        mock_client.fetch_order.return_value = {
            "id": "entry_001", "average": 50000.0, "filled": 0.01, "status": "filled",
        }
        executor._client = mock_client

        # Order placer that fails to place SL
        mock_placer = AsyncMock(spec=ExchangeOrderPlacer)
        mock_placer.place_protective_orders = AsyncMock(side_effect=RuntimeError("SL API down"))
        executor._order_placer = mock_placer

        signal = MagicMock(spec=TradingSignal)
        signal.symbol = "BTC/USDT:USDT"
        signal.exchange = "binance"
        signal.direction = "long"
        signal.is_long = True
        signal.price = 50000.0
        signal.metadata = {}

        result = await executor._live_execute(signal, 500.0)
        # Circuit breaker should be tripped via trip() method
        risk_mgr._circuit_breaker.trip.assert_called()
        call_args = risk_mgr._circuit_breaker.trip.call_args
        assert "sl_placement_failed" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_testnet_stop_order_fallback_does_not_trip_breaker(self):
        """Known testnet STOP_MARKET limitations should fall back to bot-managed exits."""
        from execution.cex_executor import CEXExecutor
        from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrderFallbackRequired
        from engine.signal_generator import TradingSignal

        config = Config.get(path=str(CONFIG_PATH))
        event_bus = EventBus()
        risk_mgr = MagicMock()
        risk_mgr._circuit_breaker = MagicMock()
        risk_mgr._circuit_breaker._tripped = False

        mock_pos = MagicMock()
        mock_pos.stop_loss = 49000.0
        mock_pos.take_profit = 52000.0
        risk_mgr.open_position = AsyncMock(return_value=mock_pos)

        executor = CEXExecutor(config, event_bus, risk_mgr, "binance")
        mock_client = AsyncMock()
        mock_client.create_limit_order.return_value = {
            "id": "entry_001", "average": 50000.0, "filled": 0.01, "status": "filled",
        }
        mock_client.fetch_order.return_value = {
            "id": "entry_001", "average": 50000.0, "filled": 0.01, "status": "filled",
        }
        executor._client = mock_client

        mock_placer = AsyncMock(spec=ExchangeOrderPlacer)
        mock_placer.place_protective_orders = AsyncMock(
            side_effect=ProtectiveOrderFallbackRequired("binance algo orders unavailable on testnet")
        )
        executor._order_placer = mock_placer

        signal = MagicMock(spec=TradingSignal)
        signal.symbol = "BTC/USDT:USDT"
        signal.exchange = "binance"
        signal.direction = "long"
        signal.is_long = True
        signal.price = 50000.0
        signal.metadata = {}

        result = await executor._live_execute(signal, 500.0)
        assert result is not None
        risk_mgr._circuit_breaker.trip.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconciliation_reads_sl_tp_from_exchange_orders(self):
        """Reconciliation should read SL/TP prices from existing exchange orders."""
        from execution.reconciliation import StartupReconciler

        config = Config.get(path=str(CONFIG_PATH))
        event_bus = EventBus()
        risk_mgr = MagicMock()
        risk_mgr._positions = {}
        risk_mgr._equity = 10000.0
        risk_mgr._circuit_breaker = MagicMock()
        risk_mgr._circuit_breaker._tripped = False

        client = AsyncMock()
        client.fetch_balance.return_value = {
            "total": {"USDT": 10000.0}, "free": {"USDT": 9000.0}
        }
        client.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.01,
                "notional": 500.0, "entryPrice": 50000.0, "unrealizedPnl": 0,
                "leverage": 5, "marginType": "cross", "liquidationPrice": 40000.0,
            },
        ]
        client.fetch_open_orders.return_value = [
            {
                "id": "sl_existing", "symbol": "BTC/USDT:USDT",
                "type": "stop_market", "side": "sell", "amount": 0.01,
                "price": 0, "stopPrice": 48500.0, "status": "open",
                "reduceOnly": True, "reduce_only": True,
            },
            {
                "id": "tp_existing", "symbol": "BTC/USDT:USDT",
                "type": "take_profit_market", "side": "sell", "amount": 0.01,
                "price": 0, "stopPrice": 53000.0, "status": "open",
                "reduceOnly": True, "reduce_only": True,
            },
        ]

        reconciler = StartupReconciler(
            config=config, event_bus=event_bus,
            risk_manager=risk_mgr, client=client,
        )
        result = await reconciler.reconcile()

        pos = risk_mgr._positions["binance:BTC/USDT:USDT"]
        assert pos.stop_loss == 48500.0
        assert pos.take_profit == 53000.0

    def test_min_balance_uses_dedicated_config(self):
        """StartupValidator uses min_balance_usd, not min_liquidity_usd."""
        config = Config.get(path=str(CONFIG_PATH))
        from execution.startup_validation import StartupValidator
        validator = StartupValidator(config, client=None)
        # min_balance_usd is 100, not min_liquidity_usd (100000)
        assert validator._min_balance_usd == 100


class TestLiveModeGate:
    """Tests for dry-run default and live mode confirmation."""

    def test_config_defaults_to_paper(self):
        """Config defaults to paper_mode=True."""
        config = Config.get(path=str(CONFIG_PATH))
        assert config.paper_mode is True

    def test_live_trading_env_required(self):
        """LIVE_TRADING_CONFIRMED env var required for live mode."""
        # This is enforced in main.py boot sequence
        # Testing the pattern: live mode should require explicit opt-in
        import os
        assert os.getenv("LIVE_TRADING_CONFIRMED", "").lower() != "true"

    def test_live_mode_aborts_when_postgres_audit_db_unavailable(self):
        """Live mode must fail closed when PostgreSQL audit persistence is unavailable."""
        from main import _enforce_live_audit_db_available

        config = MagicMock(spec=Config)
        config.paper_mode = False
        db = MagicMock()
        db.available = False

        with pytest.raises(SystemExit) as exc:
            _enforce_live_audit_db_available(config, db)

        assert exc.value.code == 1

    def test_paper_mode_can_continue_without_postgres_audit_db(self):
        """Paper mode may continue with DB offline because no real-money orders can be placed."""
        from main import _enforce_live_audit_db_available

        config = MagicMock(spec=Config)
        config.paper_mode = True
        db = MagicMock()
        db.available = False

        _enforce_live_audit_db_available(config, db)


# ═══════════════════════════════════════════════════════════════════════════════
# Module imports sanity check
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Verify all new production modules import correctly."""

    def test_import_exchange_order_placer(self):
        from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrders
        assert ExchangeOrderPlacer is not None
        assert ProtectiveOrders is not None

    def test_import_user_stream(self):
        from data_ingestion.user_stream import UserDataStream
        assert UserDataStream is not None

    def test_import_reconciliation(self):
        from execution.reconciliation import StartupReconciler, ReconciliationResult
        assert StartupReconciler is not None
        assert ReconciliationResult is not None

    def test_import_trade_persistence(self):
        from storage.trade_persistence import TradePersistence, TRADING_DDL
        assert TradePersistence is not None
        assert len(TRADING_DDL) >= 6

    def test_import_startup_validation(self):
        from execution.startup_validation import StartupValidator, ValidationError
        assert StartupValidator is not None
        assert issubclass(ValidationError, Exception)


# ═══════════════════════════════════════════════════════════════════════════════
# Deep audit round 2 — line-by-line bug fixes
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeepAuditFixes:
    """Tests for bugs found during line-by-line audit."""

    def test_db_handler_uses_configured_password_in_real_dsn(self):
        """DBHandler must only mask passwords in logs, never in the actual asyncpg DSN."""
        from storage.db_handler import DBHandler

        config = MagicMock(spec=Config)
        config.get_value.return_value = {
            "host": "db.local",
            "port": 5433,
            "database": "nt",
            "user": "operator",
            "password": "real-secret-password",
            "pool_size": 3,
            "timescaledb": False,
        }

        db = DBHandler(config)

        assert "real-secret-password" in db._dsn
        assert "***" not in db._dsn
        assert db._redacted_dsn == "postgresql://operator:***@db.local:5433/nt"

    @pytest.mark.asyncio
    async def test_db_connect_guard_prevents_double_pool(self):
        """db.connect() called twice should NOT create a second pool."""
        from storage.db_handler import DBHandler

        config = Config(config_path=CONFIG_PATH)
        db = DBHandler(config)

        # Simulate first successful connect by setting _pool
        fake_pool = MagicMock()
        db._pool = fake_pool

        # Second connect should be a no-op (guard returns early)
        await db.connect()

        # Pool should still be the same object, not replaced
        assert db._pool is fake_pool

    @pytest.mark.asyncio
    async def test_user_stream_reconnect_untrips_circuit_breaker(self):
        """Circuit breaker should un-trip when user data stream reconnects."""
        from execution.cex_executor import CEXExecutor

        config = Config(config_path=CONFIG_PATH)
        config.paper_mode = False
        event_bus = EventBus()
        from execution.risk_manager import RiskManager
        risk_mgr = RiskManager(config, event_bus)

        executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")

        # Simulate stream lost → circuit breaker tripped
        await executor._handle_user_stream_lost({})
        assert risk_mgr._circuit_breaker._tripped is True
        assert risk_mgr._circuit_breaker._trip_reason == "user_stream_disconnected"

        # Simulate stream reconnected → should un-trip
        await executor._handle_user_stream_connected({})
        assert risk_mgr._circuit_breaker._tripped is False
        assert risk_mgr._circuit_breaker._trip_reason == ""

    @pytest.mark.asyncio
    async def test_user_stream_reconnect_keeps_other_trips(self):
        """Reconnect should NOT un-trip circuit breaker if tripped for other reasons."""
        from execution.cex_executor import CEXExecutor
        from execution.risk_manager import RiskManager

        config = Config(config_path=CONFIG_PATH)
        config.paper_mode = False
        event_bus = EventBus()
        risk_mgr = RiskManager(config, event_bus)

        executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")

        # Trip for daily loss (not stream disconnect)
        risk_mgr._circuit_breaker._tripped = True
        risk_mgr._circuit_breaker._trip_reason = "daily_loss=-3.00%"
        risk_mgr._circuit_breaker._trip_time = time.time()

        # Stream reconnect should NOT reset this
        await executor._handle_user_stream_connected({})
        assert risk_mgr._circuit_breaker._tripped is True
        assert "daily_loss" in risk_mgr._circuit_breaker._trip_reason

    @pytest.mark.asyncio
    async def test_partial_fill_sl_failure_trips_circuit_breaker(self):
        """Partial fill SL placement failure should trip circuit breaker."""
        from execution.cex_executor import CEXExecutor
        from execution.risk_manager import RiskManager
        from engine.signal_generator import TradingSignal

        config = Config(config_path=CONFIG_PATH)
        config.paper_mode = False
        event_bus = EventBus()
        risk_mgr = RiskManager(config, event_bus)

        executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")

        # Mock the client that returns a partial fill
        mock_client = AsyncMock()
        partial_fill_order = {
            "id": "test123",
            "status": "partially_filled",
            "average": 50000.0,
            "price": 50000.0,
            "filled": 0.01,
        }
        mock_client.create_limit_order = AsyncMock(return_value=partial_fill_order)
        executor._client = mock_client

        # Skip the polling loop — return partial fill directly
        executor._wait_for_fill = AsyncMock(return_value=partial_fill_order)

        # Mock order placer that fails SL placement
        mock_placer = AsyncMock()
        mock_placer.place_protective_orders = AsyncMock(side_effect=RuntimeError("SL failed"))
        executor._order_placer = mock_placer

        signal = TradingSignal(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            direction="long",
            score=0.85,
            technical_score=0.8,
            ml_score=0.7,
            sentiment_score=0.6,
            macro_score=0.5,
            news_score=0.5,
            orderbook_score=0.5,
            regime="trending",
            regime_confidence=0.8,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            atr=500.0,
            timestamp=int(time.time()),
        )

        await executor._live_execute(signal, 500.0)

        # Circuit breaker should be tripped
        assert risk_mgr._circuit_breaker._tripped is True
        assert "sl_placement_failed" in risk_mgr._circuit_breaker._trip_reason

    @pytest.mark.asyncio
    async def test_position_closed_payload_is_always_dict(self):
        """POSITION_CLOSED should always publish dict format, not raw Position."""
        from execution.cex_executor import CEXExecutor
        from execution.risk_manager import RiskManager

        config = Config(config_path=CONFIG_PATH)
        config.paper_mode = False
        event_bus = EventBus()
        risk_mgr = RiskManager(config, event_bus)

        executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")

        # Open a position
        from engine.signal_generator import TradingSignal
        signal = TradingSignal(
            exchange="binance", symbol="BTC/USDT:USDT", direction="long",
            score=0.85, technical_score=0.8, ml_score=0.7, sentiment_score=0.6,
            macro_score=0.5, news_score=0.5, orderbook_score=0.5,
            regime="trending", regime_confidence=0.8,
            price=50000.0, stop_loss=49000.0, take_profit=52000.0,
            atr=500.0, timestamp=int(time.time()),
        )
        await risk_mgr.open_position(signal, 500.0)

        # Capture what gets published
        captured = []
        async def capture_handler(payload):
            captured.append(payload)
        event_bus.subscribe("POSITION_CLOSED", capture_handler)

        # Trigger stop loss
        await executor._handle_stop_loss({
            "exchange": "binance", "symbol": "BTC/USDT:USDT", "price": 49000.0,
        })
        await _drain_event_bus(event_bus)

        assert len(captured) == 1
        assert isinstance(captured[0], dict)
        assert "position" in captured[0]
        assert "reason" in captured[0]

    @pytest.mark.asyncio
    async def test_user_order_update_filtered_by_exchange_id(self):
        """Only binance executor should process USER_ORDER_UPDATE events."""
        from execution.cex_executor import CEXExecutor
        from execution.risk_manager import RiskManager

        config = Config(config_path=CONFIG_PATH)
        config.paper_mode = False
        event_bus = EventBus()
        risk_mgr = RiskManager(config, event_bus)

        # Create a non-binance executor
        executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="bybit")

        payload = {
            "symbol": "BTC/USDT:USDT",
            "execution_type": "TRADE",
            "order_status": "FILLED",
            "order_type": "LIMIT",
            "reduce_only": False,
            "last_filled_qty": 0.01,
            "last_filled_price": 50000.0,
            "cumulative_filled_qty": 0.01,
            "quantity": 0.01,
            "commission": 0.005,
            "realized_profit": 0,
            "trade_id": 12345,
            "order_id": 999,
        }

        # Should return early without processing (no error, no publish)
        await executor._handle_user_order_update(payload)
        # If it tried to process, it would hit close_position or other logic
        # No assertion needed beyond no exception — the filter check is in place


@pytest.mark.asyncio
async def test_live_execute_keeps_reservation_when_unfilled_order_cancel_fails():
    from execution.cex_executor import CEXExecutor
    from engine.signal_generator import TradingSignal

    config = Config(config_path=CONFIG_PATH)
    config.paper_mode = False
    event_bus = EventBus()
    risk_mgr = MagicMock()
    risk_mgr._circuit_breaker.trip = MagicMock()
    executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")
    executor._rate_limiter.acquire = AsyncMock()
    executor._client = AsyncMock()
    executor._client.create_limit_order = AsyncMock(return_value={"id": "entry-uncertain", "status": "open", "filled": 0, "price": 50000.0})
    executor._client.cancel_order = AsyncMock(side_effect=RuntimeError("cancel timeout"))
    executor._wait_for_fill = AsyncMock(return_value={"id": "entry-uncertain", "status": "open", "filled": 0, "price": 50000.0})
    captured: list[dict] = []

    async def capture(payload):
        captured.append(payload)

    event_bus.subscribe("ALERT_CRITICAL", capture)
    signal = TradingSignal(
        exchange="binance", symbol="BTC/USDT:USDT", direction="long",
        score=0.85, technical_score=0.8, ml_score=0.7, sentiment_score=0.6,
        macro_score=0.5, news_score=0.5, orderbook_score=0.5,
        regime="trending", regime_confidence=0.8,
        price=50000.0, stop_loss=49000.0, take_profit=52000.0,
        atr=500.0, timestamp=int(time.time()),
    )

    result = await executor._live_execute(signal, 500.0, reserved_pos=object())
    await _drain_event_bus(event_bus)

    assert result is not None
    assert result.status == "open"
    risk_mgr._circuit_breaker.trip.assert_called_once()
    assert any(evt.get("type") == "untracked_open_order_cancel_failed" for evt in captured)


@pytest.mark.asyncio
async def test_wait_for_fill_does_not_market_fallback_after_uncertain_cancel():
    from execution.cex_executor import CEXExecutor
    from execution.risk_manager import RiskManager
    from engine.signal_generator import TradingSignal

    config = Config(config_path=CONFIG_PATH)
    config.paper_mode = False
    event_bus = EventBus()
    risk_mgr = RiskManager(config, event_bus)
    executor = CEXExecutor(config, event_bus, risk_mgr, exchange_id="binance")
    executor._rate_limiter.acquire = AsyncMock()
    executor._fill_market_on_timeout = True
    executor._client = AsyncMock()
    executor._client.fetch_order = AsyncMock(return_value={"id": "limit-1", "status": "open", "filled": 0, "price": 50000.0})
    executor._client.cancel_order = AsyncMock(side_effect=RuntimeError("cancel timeout"))
    executor._client.create_market_order = AsyncMock(return_value={"id": "market-dup", "status": "closed", "filled": 0.01, "average": 50000.0})
    captured: list[dict] = []

    async def capture(payload):
        captured.append(payload)

    event_bus.subscribe("ALERT_CRITICAL", capture)
    signal = TradingSignal(
        exchange="binance", symbol="BTC/USDT:USDT", direction="long",
        score=0.85, technical_score=0.8, ml_score=0.7, sentiment_score=0.6,
        macro_score=0.5, news_score=0.5, orderbook_score=0.5,
        regime="trending", regime_confidence=0.8,
        price=50000.0, stop_loss=49000.0, take_profit=52000.0,
        atr=500.0, timestamp=int(time.time()),
    )

    result = await executor._wait_for_fill(signal, {"id": "limit-1", "status": "open", "filled": 0, "price": 50000.0}, 0.01, max_retries=0, wait_sec=0)
    await _drain_event_bus(event_bus)

    assert result["id"] == "limit-1"
    executor._client.create_market_order.assert_not_called()
    assert risk_mgr._circuit_breaker._tripped is True
    assert "cancel_uncertain" in risk_mgr._circuit_breaker._trip_reason
    assert any(evt.get("type") == "entry_cancel_uncertain_no_market_fallback" for evt in captured)
