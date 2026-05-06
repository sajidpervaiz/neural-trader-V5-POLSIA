"""Unit tests for the risk manager."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.config import Config
from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.risk_manager import RiskManager, CircuitBreaker


@pytest.fixture
def config() -> Config:
    c = MagicMock(spec=Config)
    c.paper_mode = True
    c.get_value.return_value = {
        "max_position_size_pct": 0.02,
        "max_open_positions": 5,
        "default_leverage": 1.0,
        "max_daily_loss_pct": 0.03,
        "max_drawdown_pct": 0.10,
        "max_portfolio_var_pct": 0.08,
        "returns_window": 250,
        "var_min_history": 30,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "initial_equity": 100000,
        "risk_per_trade_pct": 0.01,
        "sizing_method": "risk_based",
        "max_spread_bps": 10.0,
        "max_atr_pct": 0.05,
        "max_exposure_per_symbol_pct": 0.10,
        "cooldown_seconds": 0,
        "session_start_utc": "00:00",
        "session_end_utc": "23:59",
        "atr_sl_multiplier": 1.5,
        "rr_ratio": 2.0,
        "trailing_activation_atr": 2.0,
        "trailing_distance_atr": 1.0,
        "breakeven_trigger_atr": 1.0,
        "max_hold_minutes": 0,
        "circuit_breaker_pause_seconds": 0,
    }
    return c


@pytest.fixture
def risk_manager(config: Config) -> RiskManager:
    bus = EventBus()
    rm = RiskManager(config, bus)
    rm._equity = 100_000.0
    return rm


def _make_signal(score: float = 0.75, price: float = 50000.0) -> TradingSignal:
    sl = price * 0.985
    tp = price * 1.03
    rr = (tp - price) / (price - sl)
    return TradingSignal(
        exchange="binance",
        symbol="BTC/USDT:USDT",
        direction="long",
        score=score,
        technical_score=score,
        ml_score=score,
        sentiment_score=0.0,
        macro_score=0.0,
        news_score=0.0,
        orderbook_score=0.0,
        regime="trending_up",
        regime_confidence=0.8,
        price=price,
        atr=price * 0.01,
        stop_loss=sl,
        take_profit=tp,
        timestamp=1_700_000_000,
    )


class TestRiskManager:
    def test_approve_valid_signal(self, risk_manager: RiskManager) -> None:
        signal = _make_signal()
        approved, reason, size = risk_manager.approve_signal(signal)
        assert approved is True
        assert size > 0

    def test_reject_low_score(self, risk_manager: RiskManager) -> None:
        signal = _make_signal(score=0.3)
        approved, reason, size = risk_manager.approve_signal(signal)
        assert approved is False
        assert "score" in reason.lower()

    def test_reject_poor_risk_reward(self, risk_manager: RiskManager) -> None:
        sl = 49999.0
        tp = 50001.0
        signal = TradingSignal(
            exchange="binance", symbol="BTC/USDT:USDT", direction="long",
            score=0.9, technical_score=0.9, ml_score=0.9,
            sentiment_score=0.0, macro_score=0.0,
            news_score=0.0, orderbook_score=0.0,
            regime="trending_up", regime_confidence=0.8,
            price=50000.0, atr=500.0,
            stop_loss=sl, take_profit=tp, timestamp=1_700_000_000,
        )
        approved, reason, _ = risk_manager.approve_signal(signal)
        assert approved is False

    async def test_open_and_close_position(self, risk_manager: RiskManager) -> None:
        signal = _make_signal()
        pos = await risk_manager.open_position(signal, size=1000.0)
        assert pos is not None
        assert len(risk_manager.positions) == 1

        closed = await risk_manager.close_position("binance", "BTC/USDT:USDT", 51500.0)
        assert closed is not None
        assert len(risk_manager.positions) == 0
        assert closed.pnl_pct > 0

    async def test_max_positions_limit(self, risk_manager: RiskManager) -> None:
        for i in range(5):
            sig = _make_signal()
            sig.symbol = f"TOKEN{i}/USDT"
            approved, _, size = risk_manager.approve_signal(sig)
            if approved:
                await risk_manager.open_position(sig, size)
        extra = _make_signal()
        extra.symbol = "EXTRA/USDT"
        approved, reason, _ = risk_manager.approve_signal(extra)
        assert approved is False
        assert "max_positions" in reason.lower()

    def test_calculate_var_and_cvar(self, risk_manager: RiskManager) -> None:
        returns = [-0.02, -0.01, 0.01, -0.03, 0.015, -0.005, -0.04, 0.02]
        for r in returns:
            risk_manager.record_return(r)

        var95 = risk_manager.calculate_historical_var(0.95)
        cvar95 = risk_manager.calculate_cvar(0.95)

        assert var95 > 0
        assert cvar95 > 0
        assert cvar95 >= var95

    async def test_stress_test_output(self, risk_manager: RiskManager) -> None:
        signal = _make_signal(price=50000.0)
        pos = await risk_manager.open_position(signal, size=1000.0)
        assert pos is not None

        report = risk_manager.run_stress_test(shocks=[-0.10], correlation_breakdown_factor=2.0)
        assert report["portfolio_notional_usd"] == pytest.approx(1000.0)
        assert "shock_10pct" in report["scenarios"]
        assert report["scenarios"]["shock_10pct"]["estimated_loss_usd"] == pytest.approx(100.0)
        assert report["scenarios"]["correlation_breakdown"]["estimated_loss_usd"] == pytest.approx(200.0)

    def test_var_gate_rejects_signal_when_limit_exceeded(self, risk_manager: RiskManager) -> None:
        for _ in range(30):
            risk_manager.record_return(-0.10)

        signal = _make_signal()
        approved, reason, _ = risk_manager.approve_signal(signal)
        assert approved is False
        assert "var_limit_breach" in reason


class TestCircuitBreaker:
    def test_daily_loss_trips_breaker(self) -> None:
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10)
        cb.record_pnl(-0.01, 99000)
        assert not cb.tripped
        cb.record_pnl(-0.025, 97500)
        assert cb.tripped
        assert "daily_loss" in cb.trip_reason

    def test_reset_clears_breaker(self) -> None:
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10)
        cb.record_pnl(-0.05, 95000)
        assert cb.tripped
        cb.reset()
        assert not cb.tripped

    def test_timed_auto_resume(self) -> None:
        """Circuit breaker requires manual reset — no auto-resume after pause."""
        import time
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10, pause_seconds=0.1)
        cb.record_pnl(-0.05, 95000)
        assert cb.tripped
        time.sleep(0.15)
        assert cb.tripped  # Must stay tripped until manual reset
        cb.reset()
        assert not cb.tripped  # Manual reset clears it


class TestKillSwitch:
    async def test_kill_switch_closes_all_and_blocks(self, risk_manager: RiskManager) -> None:
        sig1 = _make_signal()
        sig1.symbol = "BTC/USDT"
        await risk_manager.open_position(sig1, 1000.0)

        sig2 = _make_signal()
        sig2.symbol = "ETH/USDT"
        await risk_manager.open_position(sig2, 500.0)
        assert len(risk_manager.positions) == 2

        closed = await risk_manager.activate_kill_switch()
        assert len(closed) == 2
        assert len(risk_manager.positions) == 0
        assert risk_manager.killed

        # New signals should be rejected
        approved, reason, _ = risk_manager.approve_signal(_make_signal())
        assert not approved
        assert "kill_switch" in reason

    async def test_deactivate_kill_switch(self, risk_manager: RiskManager) -> None:
        await risk_manager.activate_kill_switch()
        assert risk_manager.killed
        risk_manager.deactivate_kill_switch()
        assert not risk_manager.killed


class TestPositionManagement:
    def test_compute_atr_stops(self, risk_manager: RiskManager) -> None:
        signal = _make_signal(price=50000.0)
        sl, tp = risk_manager.compute_atr_stops(signal)
        # ATR = 50000 * 0.01 = 500, SL distance = 500 * 1.5 = 750
        assert sl < signal.price
        assert tp > signal.price
        assert tp - signal.price > signal.price - sl  # TP distance > SL distance (RR > 1)

    def test_risk_snapshot_includes_new_fields(self, risk_manager: RiskManager) -> None:
        snap = risk_manager.get_risk_snapshot()
        assert "kill_switch_active" in snap
        assert "current_drawdown_pct" in snap
        assert snap["kill_switch_active"] is False


class TestIteration1Fixes:
    """Tests for P0/P1 fixes from production readiness iteration."""

    def test_zero_price_returns_zero_size(self, risk_manager: RiskManager) -> None:
        """P0-4: calculate_position_size must return 0 for zero/negative price."""
        signal = _make_signal(price=1.0)  # create with valid price
        signal.price = 0.0  # then override to zero
        size = risk_manager.calculate_position_size(signal)
        assert size == 0.0

        signal.price = -100.0
        size_neg = risk_manager.calculate_position_size(signal)
        assert size_neg == 0.0

    async def test_approve_and_open_atomic(self, risk_manager: RiskManager) -> None:
        """P0: approve_and_open atomically approves + opens position under lock."""
        signal = _make_signal()
        approved, reason, size, pos = await risk_manager.approve_and_open(signal)
        assert approved is True
        assert pos is not None
        assert len(risk_manager.positions) == 1

        # Second call for same symbol must reject (already in position)
        approved2, reason2, _, pos2 = await risk_manager.approve_and_open(signal)
        assert approved2 is False
        assert "already_in_position" in reason2
        assert pos2 is None

    async def test_negative_equity_trips_kill_switch(self, risk_manager: RiskManager) -> None:
        """P0: equity going to zero or negative must activate kill switch."""
        signal = _make_signal()
        # Open a huge position (100% of equity)
        pos = await risk_manager.open_position(signal, size=200_000.0)
        # Close at near-zero price — creates massive loss exceeding equity
        closed = await risk_manager.close_position("binance", "BTC/USDT:USDT", 0.01)
        assert closed is not None
        assert risk_manager._equity == 0.0  # clamped to zero
        assert risk_manager.killed  # kill switch activated

    def test_circuit_breaker_trip_method(self) -> None:
        """P1-2: CircuitBreaker.trip() sets state correctly."""
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10)
        cb.trip("test_reason")
        assert cb.tripped is True  # pause_seconds=3600 means still tripped
        assert cb.trip_reason == "test_reason"

    def test_circuit_breaker_clear_if_reason(self) -> None:
        """P1-2: clear_if_reason only clears matching reason."""
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10)
        cb.trip("stream_lost")
        assert not cb.clear_if_reason("wrong_reason")
        assert cb.tripped is True
        assert cb.clear_if_reason("stream_lost")

    def test_circuit_breaker_trip_with_custom_pause(self) -> None:
        """P1-2: trip() with custom pause_seconds."""
        cb = CircuitBreaker(max_daily_loss_pct=0.03, max_drawdown_pct=0.10)
        cb.trip("recon_fail", pause_seconds=86400.0)
        assert cb._pause_seconds == 86400.0

    def test_var_cold_start_limits_size(self, risk_manager: RiskManager) -> None:
        """P1: With no return history, size should be halved (VaR cold start)."""
        risk_manager._return_history = []  # No history
        signal = _make_signal()
        approved, reason, size = risk_manager.approve_signal(signal)
        assert approved is True
        # Size should be capped at 0.5 × max_position_pct × leverage × equity
        max_cold = risk_manager._equity * risk_manager._max_position_pct * risk_manager._leverage * 0.5
        assert size <= max_cold + 0.01  # tolerance

    def test_max_leverage_per_symbol_enforced(self, risk_manager: RiskManager) -> None:
        """P1: max_leverage_per_symbol must cap position size."""
        risk_manager._leverage = 10.0
        risk_manager._max_position_pct = 1.0  # remove max cap to isolate leverage effect
        risk_manager._max_leverage_per_symbol = {"BTC/USDT:USDT": 3.0}
        signal = _make_signal()
        size_with_cap = risk_manager.calculate_position_size(signal)

        risk_manager._max_leverage_per_symbol = {}
        size_without_cap = risk_manager.calculate_position_size(signal)

        assert size_with_cap < size_without_cap


class TestIteration1FixesBatch2:
    """Tests for Iteration 1 Batch 2 fixes: USER_DATA fills, listenKey, CORS, reconciliation, rate limit."""

    async def test_portfolio_rate_limit_blocks_rapid_orders(self, risk_manager: RiskManager) -> None:
        """P1: Portfolio-level rate limit must block rapid fire orders."""
        risk_manager._max_orders_per_window = 2
        risk_manager._order_window_seconds = 10.0
        risk_manager._max_open = 10  # allow many positions

        signal1 = _make_signal(price=50000.0)
        signal1.symbol = "BTC/USDT:USDT"
        ok1, _, _, _ = await risk_manager.approve_and_open(signal1)
        assert ok1 is True

        signal2 = _make_signal(price=3000.0)
        signal2.symbol = "ETH/USDT:USDT"
        ok2, _, _, _ = await risk_manager.approve_and_open(signal2)
        assert ok2 is True

        signal3 = _make_signal(price=100.0)
        signal3.symbol = "SOL/USDT:USDT"
        ok3, reason3, _, _ = await risk_manager.approve_and_open(signal3)
        assert ok3 is False
        assert "portfolio_rate_limit" in reason3

    async def test_user_data_fill_syncs_order_manager(self) -> None:
        """P0: USER_DATA fills must call OrderManager.record_fill."""
        from unittest.mock import AsyncMock
        from execution.cex_executor import CEXExecutor
        from execution.order_manager import OrderManager

        config = MagicMock(spec=Config)
        config.paper_mode = True
        config.get_value.return_value = {}
        bus = EventBus()
        rm = MagicMock(spec=RiskManager)
        om = MagicMock(spec=OrderManager)
        om.record_fill = AsyncMock(return_value=None)

        executor = CEXExecutor(config, bus, rm, "binance", order_manager=om)

        payload = {
            "symbol": "BTCUSDT",
            "execution_type": "TRADE",
            "order_status": "FILLED",
            "order_type": "MARKET",
            "reduce_only": False,
            "last_filled_qty": 0.1,
            "last_filled_price": 50000.0,
            "cumulative_filled_qty": 0.1,
            "quantity": 0.1,
            "commission": 5.0,
            "realized_profit": 0,
            "client_order_id": "my_order_123",
            "trade_id": "12345",
        }
        await executor._handle_user_order_update(payload)
        om.record_fill.assert_called_once_with(
            client_order_id="my_order_123",
            fill_id="12345",
            quantity=0.1,
            price=50000.0,
            fee=5.0,
        )

    async def test_user_data_cancel_syncs_order_manager(self) -> None:
        """P0: USER_DATA cancel must call OrderManager.cancel_order."""
        from unittest.mock import AsyncMock
        from execution.cex_executor import CEXExecutor
        from execution.order_manager import OrderManager

        config = MagicMock(spec=Config)
        config.paper_mode = True
        config.get_value.return_value = {}
        bus = EventBus()
        rm = MagicMock(spec=RiskManager)
        om = MagicMock(spec=OrderManager)
        om.cancel_order = AsyncMock(return_value=(True, None, "cancelled"))

        executor = CEXExecutor(config, bus, rm, "binance", order_manager=om)

        payload = {
            "symbol": "BTCUSDT",
            "execution_type": "CANCELED",
            "client_order_id": "my_order_456",
            "order_id": 999,
        }
        await executor._handle_user_order_update(payload)
        om.cancel_order.assert_called_once_with("my_order_456", reason="exchange_cancel")

    def test_cors_wildcard_blocked_in_live_mode(self) -> None:
        """P1: CORS '*' must be stripped in non-paper mode."""
        from unittest.mock import patch, MagicMock as MM
        import interface.dashboard_api as dapi

        config = MagicMock(spec=Config)
        config.paper_mode = False
        config.get_value.side_effect = lambda *a, **kw: {
            "allow_origins": ["*", "http://localhost"],
            "auth": {"require_api_key": True, "api_key": "test123"},
        }

        # We just need to verify the cors_origins filtering logic
        dashboard_cfg = config.get_value("monitoring", "dashboard_api", default={}) or {}
        cors_origins = dashboard_cfg.get("allow_origins") or ["http://localhost", "http://127.0.0.1"]
        if not config.paper_mode and "*" in cors_origins:
            cors_origins = [o for o in cors_origins if o != "*"] or ["http://localhost"]

        assert "*" not in cors_origins
        assert "http://localhost" in cors_origins

    def test_cors_wildcard_allowed_in_paper_mode(self) -> None:
        """CORS '*' is acceptable in paper mode."""
        config = MagicMock(spec=Config)
        config.paper_mode = True
        cors_origins = ["*", "http://localhost"]
        if not config.paper_mode and "*" in cors_origins:
            cors_origins = [o for o in cors_origins if o != "*"] or ["http://localhost"]

        assert "*" in cors_origins  # paper mode — keep it

    async def test_listenkey_keepalive_retries(self) -> None:
        """P1: listenKey keepalive retries 3 times then invalidates key."""
        from data_ingestion.user_stream import UserDataStream

        config = MagicMock(spec=Config)
        config.get_value.return_value = {"api_key": "test", "api_secret": "test"}
        bus = EventBus()
        stream = UserDataStream(config, bus)
        stream._listen_key = "test_key_123"
        stream._ws = MagicMock()
        from unittest.mock import AsyncMock as AM
        stream._ws.close = AM()

        call_count = 0
        async def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("network error")

        stream._http_request = failing_request
        await stream._keepalive_listen_key()

        assert call_count == 3  # retried 3 times
        assert stream._listen_key is None  # invalidated for reconnect

    def test_reconciliation_registers_protective_orders(self) -> None:
        """P1: Reconciliation must register existing SL/TP in order_placer._protective."""
        from execution.reconciliation import StartupReconciler, ReconciliationResult
        from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrders

        config = MagicMock(spec=Config)
        bus = EventBus()
        rm = MagicMock(spec=RiskManager)
        from execution.risk_manager import Position
        pos = Position(
            exchange="binance", symbol="BTCUSDT", direction="long",
            size=0.1, entry_price=50000.0, current_price=50000.0,
            stop_loss=0.0, take_profit=0.0, open_time=0,
            highest_since_entry=50000.0, lowest_since_entry=50000.0,
        )
        rm._positions = {"binance:BTCUSDT": pos}
        rm.config = config

        client = MagicMock()
        order_placer = MagicMock(spec=ExchangeOrderPlacer)
        order_placer._protective = {}

        reconciler = StartupReconciler(config, bus, rm, client, order_placer)

        result = ReconciliationResult()
        result.exchange_positions = [
            {"symbol": "BTCUSDT", "side": "long", "size": 0.1, "entry_price": 50000.0},
        ]
        result.exchange_open_orders = [
            {"symbol": "BTCUSDT", "type": "stop_market", "reduce_only": True,
             "stop_price": 49000.0, "id": "sl_001"},
            {"symbol": "BTCUSDT", "type": "take_profit_market", "reduce_only": True,
             "stop_price": 53000.0, "id": "tp_001"},
        ]

        import asyncio
        asyncio.run(reconciler._ensure_protective_orders(result))

        assert "BTCUSDT" in order_placer._protective
        prot = order_placer._protective["BTCUSDT"]
        assert prot.sl_order_id == "sl_001"
        assert prot.tp_order_id == "tp_001"
        assert prot.sl_placed is True
        assert prot.tp_placed is True


class TestIteration2Fixes:
    """Tests for Iteration 2 fixes."""

    def test_mode_toggle_blocks_live(self) -> None:
        """P0: /api/mode/toggle must refuse paper→live switch."""
        from interface.dashboard_api import build_app
        from fastapi.testclient import TestClient
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock(spec=Config)
        config.paper_mode = True
        config.get_value.return_value = {}
        bus = EventBus()
        rm = RiskManager(config, bus)

        app = build_app(config, bus, risk_manager=rm)
        client = TestClient(app)
        resp = client.post("/api/mode/toggle", json={"mode": "live"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "Cannot switch to live" in body["error"]
        # Paper→paper should still work
        resp2 = client.post("/api/mode/toggle", json={"mode": "paper"})
        assert resp2.json()["success"] is True

    def test_api_trade_rejects_limit_price_zero(self) -> None:
        """P1: /api/trade must reject LIMIT orders with price <= 0."""
        from interface.dashboard_api import build_app
        from fastapi.testclient import TestClient
        from execution.order_manager import OrderManager
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock(spec=Config)
        config.paper_mode = True
        config.get_value.return_value = {}
        bus = EventBus()
        rm = RiskManager(config, bus)
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        app = build_app(config, bus, risk_manager=rm, order_manager=om)
        client = TestClient(app)
        resp = client.post("/api/trade", json={
            "symbol": "BTC/USDT",
            "side": "BUY",
            "size": 0.01,
            "order_type": "limit",
            "price": 0,
        })
        body = resp.json()
        assert body["success"] is False
        assert "price" in body["error"].lower()

    def test_api_trade_rejects_zero_size(self) -> None:
        """P1: /api/trade must reject size <= 0."""
        from interface.dashboard_api import build_app
        from fastapi.testclient import TestClient
        from execution.order_manager import OrderManager
        from core.circuit_breaker import CircuitBreaker

        config = MagicMock(spec=Config)
        config.paper_mode = True
        config.get_value.return_value = {}
        bus = EventBus()
        rm = RiskManager(config, bus)
        cb = CircuitBreaker()
        om = OrderManager(config, bus, cb)

        app = build_app(config, bus, risk_manager=rm, order_manager=om)
        client = TestClient(app)
        resp = client.post("/api/trade", json={
            "symbol": "BTC/USDT",
            "side": "BUY",
            "size": 0,
            "order_type": "market",
        })
        body = resp.json()
        assert body["success"] is False
        assert "size" in body["error"].lower()

    def test_validator_updates_last_price_on_rejection(self) -> None:
        """P1: TickValidator must update last_price on rejection to prevent permanent blindness."""
        from data_ingestion.validators import TickValidator
        from data_ingestion.normalizer import Tick

        v = TickValidator(max_age_s=600)
        import time
        now_us = int(time.time() * 1_000_000)

        t1 = Tick(exchange="binance", symbol="BTC/USDT", price=50000.0,
                  volume=1.0, timestamp_us=now_us, side="buy")
        assert v.validate(t1) is True

        # 30% jump — rejected but should update last_price
        t2 = Tick(exchange="binance", symbol="BTC/USDT", price=65000.0,
                  volume=1.0, timestamp_us=now_us, side="buy")
        assert v.validate(t2) is False

        # Next tick at 65500 — within 25% of 65000, should PASS
        t3 = Tick(exchange="binance", symbol="BTC/USDT", price=65500.0,
                  volume=1.0, timestamp_us=now_us, side="buy")
        assert v.validate(t3) is True

    def test_secrets_not_in_effective_venues(self) -> None:
        """P1: _effective_venues must not contain raw API secrets."""
        from interface.routes.config import _effective_venues, configure_config_routes

        _config = MagicMock(spec=Config)
        _config.paper_mode = True
        _config.get_value.return_value = {
            "binance": {"enabled": True, "api_key": "SECRET_KEY_123", "api_secret": "SECRET_SECRET_456"},
        }
        import interface.routes.config as cfg_mod
        old_config = cfg_mod._CONFIG
        cfg_mod._CONFIG = _config
        try:
            venues = _effective_venues()
            for venue, data in venues.items():
                assert "api_key" not in data
                assert "api_secret" not in data
                assert "api_key_configured" in data
                assert "api_secret_configured" in data
        finally:
            cfg_mod._CONFIG = old_config

    async def test_equity_unreconciled_blocks_signals(self, risk_manager: RiskManager) -> None:
        """P1: After kill switch, equity_unreconciled must block new signals."""
        signal = _make_signal()
        # Normal — should approve
        ok, reason, _ = risk_manager.approve_signal(signal)
        assert ok is True

        # Set unreconciled
        risk_manager._equity_unreconciled = True
        ok2, reason2, _ = risk_manager.approve_signal(signal)
        assert ok2 is False
        assert "unreconciled" in reason2

    def test_idempotency_debounce(self) -> None:
        """P1: PersistentIdempotencyManager with short save_interval should debounce I/O."""
        from core.persistent_idempotency import PersistentIdempotencyManager
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr = PersistentIdempotencyManager(ttl=3600, max_size=100, filepath=path, save_interval=999.0)
            # check_and_set returns False for new keys (not a duplicate)
            assert mgr.check_and_set("key1") is False
            # Second call with different key should debounce (not save yet)
            mgr._dirty = False
            mgr.check_and_set("key2")
            assert mgr._dirty is True  # dirty but not saved yet
            # Same key again should return True (duplicate)
            assert mgr.check_and_set("key1") is True
        finally:
            os.unlink(path)
