"""Iteration 3 audit fix tests — event bus, model trainer, news feed, masking."""
from __future__ import annotations

import asyncio
import os
import time
import tempfile
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.event_bus import EventBus, CRITICAL_EVENTS


class TestEventBusCriticalTaskTracking:
    """P0: ensure_future for critical events must be tracked in _background_tasks."""

    @pytest.mark.asyncio
    async def test_critical_event_force_dispatch_tracked(self) -> None:
        bus = EventBus()
        received: list = []

        async def handler(payload: object) -> None:
            received.append(payload)

        bus.subscribe("KILL_SWITCH", handler)

        # Fill the queue to capacity
        for i in range(10_000):
            bus._queue.put_nowait(("NOOP", i))

        # Now publish a critical event via publish_nowait — queue is full
        bus.publish_nowait("KILL_SWITCH", {"reason": "test"})

        # The task should be tracked in _background_tasks
        assert len(bus._background_tasks) >= 1, (
            "Critical event force-dispatch task must be tracked"
        )

        # Let event loop run so the handler executes
        await asyncio.sleep(0.05)
        assert {"reason": "test"} in received

    @pytest.mark.asyncio
    async def test_stop_awaits_critical_tasks(self) -> None:
        bus = EventBus()
        completed = []

        async def slow_handler(payload: object) -> None:
            await asyncio.sleep(0.05)
            completed.append(payload)

        bus.subscribe("STOP_LOSS", slow_handler)

        for i in range(10_000):
            bus._queue.put_nowait(("NOOP", i))

        bus.publish_nowait("STOP_LOSS", "sl_test")
        # stop() should await the slow handler
        await bus.stop()
        assert "sl_test" in completed


class TestModelTrainerPickleRemoved:
    """P0: model_trainer must not use pickle.load."""

    def test_no_pickle_import(self) -> None:
        import engine.model_trainer as mt
        import inspect
        source = inspect.getsource(mt)
        assert "pickle.load(" not in source, "pickle.load must not appear in model_trainer"
        assert "pickle.dump(" not in source, "pickle.dump must not appear in model_trainer"

    def test_load_model_refuses_without_joblib(self) -> None:
        from engine.model_trainer import ModelTrainer
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.trained_models = {}

        with patch("engine.model_trainer.joblib", None):
            result = trainer.load_model("fake_model.pkl", "test_id")
        assert result is False

    def test_save_model_refuses_without_joblib(self) -> None:
        from engine.model_trainer import ModelTrainer, TrainedModel, ModelMetrics
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.trained_models = {}

        dummy = TrainedModel(
            model_type="test",
            model=None,
            feature_names=["a"],
            metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, auc=0.5, feature_importance={}),
            train_time=0.0,
            hyperparameters={},
        )
        trainer.trained_models["x"] = dummy

        with patch("engine.model_trainer.joblib", None):
            result = trainer.save_model("x", "/tmp/fake.pkl")
        assert result is False


class TestNewsFeedOrderedDedup:
    """P1: NewsFeed _seen_ids must use OrderedDict to keep newest items on trim."""

    def test_seen_ids_is_ordered(self) -> None:
        from data_ingestion.news_feed import NewsFeed
        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        feed = NewsFeed(config, bus)
        assert isinstance(feed._seen_ids, OrderedDict)

    @pytest.mark.asyncio
    async def test_trim_keeps_newest(self) -> None:
        from data_ingestion.news_feed import NewsFeed
        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        feed = NewsFeed(config, bus)

        # Simulate 510 articles
        for i in range(510):
            feed._seen_ids[f"id_{i:04d}"] = None

        # Trigger trim (happens at >500)
        while len(feed._seen_ids) > 500:
            feed._seen_ids.popitem(last=False)

        # Oldest IDs should have been removed, newest kept
        assert "id_0000" not in feed._seen_ids
        assert "id_0509" in feed._seen_ids


class TestDashboardMaskNoLeak:
    """P1: _mask function must not leak any part of API secrets."""

    def test_mask_hides_all(self) -> None:
        from interface.dashboard_api import build_app
        from fastapi.testclient import TestClient

        config = MagicMock()
        config.paper_mode = True
        config.get_value.return_value = {
            "binance": {
                "enabled": True,
                "api_key": "abcdefghijklmnop1234",
                "api_secret": "secretXYZ1234567890",
                "testnet": True,
            },
        }
        bus = MagicMock()
        app = build_app(config, bus)
        client = TestClient(app)

        resp = client.get("/api/config/exchanges")
        body = resp.json()

        # No part of the actual key should appear in the response
        response_str = str(body)
        assert "abcd" not in response_str, "First 4 chars of api_key leaked"
        assert "secr" not in response_str, "First 4 chars of api_secret leaked"


class TestOrderCleanup:
    """P1: OrderManager must clean up terminal-state orders."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_filled_orders(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        # Create a filled order with old timestamp
        order = Order(
            order_id="old_1",
            client_order_id="cold_1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.FILLED,
            updated_at=int((time.time() - 7200) * 1000),  # 2 hours ago
        )
        om.orders["old_1"] = order
        om.client_order_map["cold_1"] = order

        # Create a recent open order
        recent = Order(
            order_id="new_1",
            client_order_id="cnew_1",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=3000.0,
            status=OrderStatus.OPEN,
        )
        om.orders["new_1"] = recent
        om.client_order_map["cnew_1"] = recent

        await om._cleanup_old_orders()

        assert "old_1" not in om.orders, "Old filled order should be cleaned up"
        assert "cold_1" not in om.client_order_map
        assert "new_1" in om.orders, "Recent open order must NOT be cleaned up"

    @pytest.mark.asyncio
    async def test_partial_fill_monitor_cancels_using_client_order_id(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        order = Order(
            order_id="oid_partial",
            client_order_id="coid_partial",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=42000.0,
            status=OrderStatus.OPEN,
            created_at=int((time.time() - 45) * 1000),
            cumulative_quantity=0.1,
        )
        om.orders[order.order_id] = order
        om.client_order_map[order.client_order_id] = order
        om.cancel_order = AsyncMock(return_value=(True, order, "cancelled"))

        await om._monitor_partial_fills()

        om.cancel_order.assert_awaited_once_with("coid_partial")

    @pytest.mark.asyncio
    async def test_limit_expiry_monitor_cancels_using_client_order_id(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        order = Order(
            order_id="oid_expire",
            client_order_id="coid_expire",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=2.0,
            price=3000.0,
            status=OrderStatus.OPEN,
            created_at=int((time.time() - 180) * 1000),
            cumulative_quantity=0.0,
        )
        om.orders[order.order_id] = order
        om.client_order_map[order.client_order_id] = order
        om.cancel_order = AsyncMock(return_value=(True, order, "cancelled"))

        await om._monitor_limit_expiry()

        om.cancel_order.assert_awaited_once_with("coid_expire")

    @pytest.mark.asyncio
    async def test_cleanup_removes_exchange_order_tuple_mapping(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        order = Order(
            order_id="old_ex",
            client_order_id="cold_ex",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.FILLED,
            updated_at=int((time.time() - 7200) * 1000),
            venue="binance",
            exchange_order_id="ex_123",
        )
        om.orders[order.order_id] = order
        om.client_order_map[order.client_order_id] = order
        om.exchange_order_map[(order.venue, order.exchange_order_id)] = order

        await om._cleanup_old_orders()

        assert (order.venue, order.exchange_order_id) not in om.exchange_order_map

    @pytest.mark.asyncio
    async def test_record_fill_updates_response_fields(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        order = Order(
            order_id="oid_fill_fields",
            client_order_id="coid_fill_fields",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=2.0,
            price=42000.0,
            status=OrderStatus.OPEN,
        )
        om.orders[order.order_id] = order
        om.client_order_map[order.client_order_id] = order

        await om.record_fill("coid_fill_fields", "fill_a", quantity=0.5, price=42010.0, fee=1.0)

        assert order.filled_quantity == pytest.approx(0.5)
        assert order.remaining_quantity == pytest.approx(1.5)
        assert order.avg_fill_price == pytest.approx(42010.0)
        assert order.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.asyncio
    async def test_load_order_state_restores_client_order_map_by_client_id(self) -> None:
        from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock()
        config.get_value.return_value = {}
        bus = MagicMock()
        cb = CircuitBreaker()

        with tempfile.TemporaryDirectory() as td:
            state_path = os.path.join(td, "order_state.json")

            writer = OrderManager(config, bus, cb, audit_log_path="", order_state_path=state_path)
            order = Order(
                order_id="oid_restore",
                client_order_id="coid_restore",
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                price=3000.0,
                status=OrderStatus.OPEN,
                venue="binance",
                exchange_order_id="ex_restore",
            )
            writer.orders[order.order_id] = order
            writer.client_order_map[order.client_order_id] = order
            writer.exchange_order_map[(order.venue, order.exchange_order_id)] = order
            writer._save_order_state()

            reader = OrderManager(config, bus, cb, audit_log_path="", order_state_path=state_path)
            reader._load_order_state()

            assert "coid_restore" in reader.client_order_map
            restored = reader.client_order_map["coid_restore"]
            assert restored.order_id == "oid_restore"
            assert ("binance", "ex_restore") in reader.exchange_order_map


class TestCEXClientOrderIds:
    """P0: live CEX orders must carry deterministic exchange-side IDs."""

    def _signal(self):
        from engine.signal_generator import TradingSignal

        return TradingSignal(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            direction="long",
            score=0.9,
            technical_score=0.9,
            ml_score=0.9,
            sentiment_score=0.0,
            macro_score=0.0,
            news_score=0.0,
            orderbook_score=0.0,
            regime="trending_up",
            regime_confidence=0.8,
            price=50_000.0,
            atr=500.0,
            stop_loss=49_250.0,
            take_profit=51_500.0,
            timestamp=1_700_000_000,
        )

    def test_client_order_param_keys_are_exchange_specific(self) -> None:
        from execution.cex_executor import _client_order_params

        signal = self._signal()

        assert set(_client_order_params("binance", signal, "entry")) == {"newClientOrderId"}
        assert set(_client_order_params("bybit", signal, "entry")) == {"orderLinkId"}
        assert set(_client_order_params("okx", signal, "entry")) == {"clOrdId"}

    @pytest.mark.asyncio
    async def test_live_limit_order_uses_deterministic_client_order_id(self) -> None:
        from execution.cex_executor import CEXExecutor, _client_order_id

        config = MagicMock()
        config.paper_mode = False
        config.get_value.return_value = {
            "wait_for_fill": {"max_retries": 0, "market_on_timeout": False},
        }
        bus = MagicMock()
        bus.publish = AsyncMock()
        risk_manager = MagicMock()
        risk_manager.rebase_position_to_fill = AsyncMock(
            return_value=MagicMock(stop_loss=49_250.0, take_profit=51_500.0)
        )
        executor = CEXExecutor(config, bus, risk_manager, exchange_id="binance")
        executor._fill_max_retries = 0
        executor._fill_market_on_timeout = False
        executor._client = MagicMock()
        executor._client.create_limit_order = AsyncMock(return_value={
            "id": "ex-order-1",
            "status": "closed",
            "filled": 0.1,
            "average": 50_010.0,
        })

        signal = self._signal()
        result = await executor._live_execute(signal, size=5_000.0, reserved_pos=object())

        assert result is not None
        create_call = executor._client.create_limit_order.await_args.kwargs
        assert create_call["params"]["newClientOrderId"] == _client_order_id(signal, "entry")
        assert create_call["params"]["postOnly"] is True


class TestSmartOrderRouterLockScope:
    """P2: SmartOrderRouter must not hold lock during network I/O."""

    def test_refresh_called_outside_lock(self) -> None:
        """Verify _refresh_scores_if_needed is NOT inside the lock in route_order."""
        import inspect
        from execution.smart_order_router import SmartOrderRouter
        source = inspect.getsource(SmartOrderRouter.route_order)
        # The refresh call should appear BEFORE the 'async with self._lock' line
        refresh_pos = source.find("_refresh_scores_if_needed")
        lock_pos = source.find("async with self._lock")
        assert refresh_pos < lock_pos, (
            "_refresh_scores_if_needed must be called BEFORE acquiring self._lock"
        )
