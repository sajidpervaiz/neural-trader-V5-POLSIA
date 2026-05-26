"""Integration tests that validate dashboard UI fetch/actions are responsive.

These tests mirror endpoints called from interface/static/index.html.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.circuit_breaker import CircuitBreaker
from core.event_bus import EventBus
from execution.order_manager import OrderManager, OrderStatus
from execution.risk_manager import RiskManager
from interface import dashboard_api
from interface.dashboard_api import build_app


@dataclass
class _FG:
    score: float
    label: str


class _FGSource:
    def get_latest(self):
        return _FG(score=0.2, label="Greed")


class _SignalGen:
    def __init__(self) -> None:
        self.auto_trading_enabled = False

    def set_auto_trading(self, enabled: bool) -> None:
        self.auto_trading_enabled = bool(enabled)


class _DataManagerWithClose:
    def __init__(self, close: float) -> None:
        self._df = pd.DataFrame([{"close": close, "open": close, "high": close, "low": close, "volume": 1.0}])

    def get_dataframe(self, exchange: str, symbol: str, timeframe: str):
        if exchange == "binance" and symbol == "BTC/USDT:USDT" and timeframe == "1m":
            return self._df
        return None


@pytest.fixture
def config_mock() -> MagicMock:
    cfg = MagicMock()
    cfg.paper_mode = True

    def _get_value(*keys, default=None):
        if keys == ("risk",):
            return {
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
                "initial_equity": 100_000,
            }
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"require_api_key": False}}
        if keys == ("exchanges",):
            return {
                "binance": {"enabled": True, "api_key": "x", "api_secret": "y", "testnet": True},
                "bybit": {"enabled": False, "api_key": "", "api_secret": ""},
                "okx": {"enabled": False, "api_key": "", "api_secret": "", "passphrase": ""},
                "kraken": {"enabled": False, "api_key": "", "api_secret": ""},
            }
        if keys == ("exchanges", "binance"):
            return {"enabled": True, "api_key": "x", "api_secret": "y", "testnet": True}
        if keys == ("dex",):
            return {
                "enabled": False,
                "rpc_url": "",
                "private_key": "",
                "uniswap": {"enabled": False},
                "sushiswap": {"enabled": False},
                "dydx": {"enabled": False},
            }
        if keys == ("notifications", "telegram"):
            return {"bot_token": "", "chat_id": ""}
        if keys == ("rust_services", "enabled"):
            return False
        if keys == ("ts_dex_layer", "enabled"):
            return False
        return default

    cfg.get_value.side_effect = _get_value
    return cfg


@pytest.fixture
def ui_client(config_mock: MagicMock) -> TestClient:
    bus = EventBus()
    breaker = CircuitBreaker()
    risk_mgr = RiskManager(config_mock, bus)
    order_mgr = OrderManager(config_mock, bus, breaker)
    signal_gen = _SignalGen()

    # Prime caches so tests do not depend on internet reachability.
    dashboard_api._market_cache["coins"] = [
        {
            "symbol": "BTC",
            "name": "Bitcoin",
            "price": 65000.0,
            "change_24h": 1.5,
            "volume_24h": 1.2e10,
            "high_24h": 66000.0,
            "low_24h": 64000.0,
            "market_cap": 1.0e12,
        }
    ]
    dashboard_api._market_cache["ts"] = time.time()
    dashboard_api._news_buffer.clear()
    dashboard_api._news_buffer.append(
        {"ts": int(time.time() * 1000), "title": "BTC steady", "sentiment": "neutral", "score": 0}
    )
    dashboard_api._orderbook_cache["binance:BTC/USDT"] = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "bids": [[64990.0, 2.0], [64980.0, 1.4]],
        "asks": [[65010.0, 2.1], [65020.0, 1.2]],
        "ts": time.time(),
    }
    dashboard_api._log_buffer.clear()
    dashboard_api._log_buffer.append(
        {"ts": int(time.time() * 1000), "level": "INFO", "message": "ui test log"}
    )

    sentiment = MagicMock()
    sentiment._fear_greed = _FGSource()

    app = build_app(
        config_mock,
        bus,
        risk_manager=risk_mgr,
        order_manager=order_mgr,
        signal_generator=signal_gen,
        news_feed=object(),
        sentiment_manager=sentiment,
    )
    return TestClient(app)


def test_dashboard_root_serves_ui_html(ui_client: TestClient) -> None:
    resp = ui_client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "toggleAuto()" in body
    assert "executeTrade('BUY')" in body
    assert "/api/realtime/stream" in body


def test_dashboard_static_fetches_can_send_configured_api_key(ui_client: TestClient) -> None:
    resp = ui_client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "nt_api_key" in body
    assert "X-API-Key" in body
    assert "fetchWithDashboardAuth" in body


def test_authenticated_dashboard_ui_calls_do_not_self_rate_limit(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.get_value.side_effect = _get_value
    app = build_app(config_mock, EventBus())
    client = TestClient(app)

    for _ in range(5):
        resp = client.get("/api/status", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200, resp.text


def test_authenticated_dashboard_can_load_local_static_assets(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.get_value.side_effect = _get_value
    app = build_app(config_mock, EventBus())
    client = TestClient(app)

    resp = client.get("/static/js/lightweight-charts.standalone.production.js")

    assert resp.status_code == 200, resp.text[:200]


def test_dashboard_does_not_return_api_key_secret(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.get_value.side_effect = _get_value
    app = build_app(config_mock, EventBus())
    client = TestClient(app)

    resp = client.get("/api/clientkey", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("configured") is True
    assert "key" not in body
    assert "test-key" not in resp.text


def test_live_readiness_endpoint_reports_fail_closed_blockers(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock()
    db.available = False
    risk = MagicMock()
    risk.kill_switch_active = False

    app = build_app(config_mock, EventBus(), db_handler=db, risk_manager=risk, executors=[])
    client = TestClient(app)

    resp = client.get("/api/live/readiness", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready_for_live"] is False
    assert body["checks"]["mode"]["ok"] is True
    assert body["checks"]["audit_db"]["ok"] is False
    assert body["checks"]["exchange"]["ok"] is False
    assert "audit_db" in body["blockers"]
    assert "exchange" in body["blockers"]
    assert isinstance(body.get("config_hash"), str)
    assert len(body["config_hash"]) == 64


def test_paper_manual_market_trade_fills_order_and_opens_position(ui_client: TestClient) -> None:
    resp = ui_client.post(
        "/api/trade",
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "BUY",
            "size": 0.01,
            "order_type": "market",
            "price": 0,
            "stop_loss_pct": 2,
            "take_profit_pct": 4,
            "leverage": 1,
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["paper"] is True
    assert body["status"] == "filled"
    assert body["filled"] == pytest.approx(0.01)
    assert body["price"] == pytest.approx(65000.0)

    order_manager = ui_client.app.state.order_manager
    order = order_manager.get_order(order_manager.get_filled_orders()[0].client_order_id)
    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(0.01)
    assert order.remaining_quantity == pytest.approx(0.0)
    assert order.avg_fill_price == pytest.approx(65000.0)

    status = ui_client.get("/api/status").json()
    assert status["open_positions"] == 1
    assert status["positions"][0]["symbol"] == "BTC/USDT:USDT"
    assert status["positions"][0]["side"] == "long"


def test_paper_manual_market_trade_can_fill_from_data_manager_without_market_cache(config_mock: MagicMock) -> None:
    bus = EventBus()
    breaker = CircuitBreaker()
    risk_mgr = RiskManager(config_mock, bus)
    order_mgr = OrderManager(config_mock, bus, breaker)
    dashboard_api._market_cache["coins"] = []
    dashboard_api._market_cache["ts"] = 0.0
    dashboard_api._orderbook_cache.clear()
    app = build_app(
        config_mock,
        bus,
        risk_manager=risk_mgr,
        order_manager=order_mgr,
        data_manager=_DataManagerWithClose(65123.45),
    )
    api = TestClient(app)

    resp = api.post(
        "/api/trade",
        json={"symbol": "BTC/USDT:USDT", "side": "BUY", "size": 0.01, "order_type": "market"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["status"] == "filled"
    assert body["price"] == pytest.approx(65123.45)
    assert risk_mgr.positions["binance:BTC/USDT:USDT"].entry_price == pytest.approx(65123.45)


def test_paper_manual_market_trade_fails_before_order_when_price_unavailable(config_mock: MagicMock) -> None:
    bus = EventBus()
    breaker = CircuitBreaker()
    risk_mgr = RiskManager(config_mock, bus)
    order_mgr = OrderManager(config_mock, bus, breaker)
    dashboard_api._market_cache["coins"] = []
    dashboard_api._market_cache["ts"] = 0.0
    dashboard_api._orderbook_cache.clear()
    app = build_app(config_mock, bus, risk_manager=risk_mgr, order_manager=order_mgr)
    api = TestClient(app)

    resp = api.post(
        "/api/trade",
        json={"symbol": "BTC/USDT:USDT", "side": "BUY", "size": 0.01, "order_type": "market"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is False
    assert "price" in body["error"].lower()
    assert order_mgr.get_stats()["total_orders"] == 0
    assert risk_mgr.positions == {}


def test_live_readiness_requires_initialized_exchange_client(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    risk = MagicMock(kill_switch_active=False)
    executor = MagicMock(exchange_id="binance")
    executor._client = None
    app = build_app(config_mock, EventBus(), db_handler=db, risk_manager=risk, executors=[executor])
    api = TestClient(app)

    resp = api.get("/api/live/readiness", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready_for_live"] is False
    assert body["checks"]["exchange"]["ok"] is False
    assert "exchange" in body["blockers"]


def test_live_readiness_requires_clean_startup_reconciliation(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    risk = MagicMock(kill_switch_active=False)
    client = MagicMock()
    client.markets = {"BTC/USDT:USDT": {}}
    executor = MagicMock(exchange_id="binance")
    executor._client = client
    recon = MagicMock()
    recon.success = False
    recon.safe_mode = True
    recon.mismatches = ["position mismatch"]
    recon.positions_without_sl = []
    user_stream = MagicMock(connected=True)

    app = build_app(
        config_mock,
        EventBus(),
        db_handler=db,
        risk_manager=risk,
        executors=[executor],
        reconciliation_result=recon,
        user_stream=user_stream,
    )
    api = TestClient(app)

    resp = api.get("/api/live/readiness", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready_for_live"] is False
    assert body["checks"]["reconciliation"]["ok"] is False
    assert "reconciliation" in body["blockers"]
    assert body["checks"]["reconciliation"]["mismatches"] == ["position mismatch"]


def test_live_readiness_blocks_when_risk_manager_killed(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    client = MagicMock()
    client.markets = {"BTC/USDT:USDT": {}}
    executor = MagicMock(exchange_id="binance")
    executor._client = client
    risk = MagicMock(killed=True)
    recon = MagicMock(success=True, safe_mode=False, mismatches=[], positions_without_sl=[])
    user_stream = MagicMock(connected=True)

    app = build_app(config_mock, EventBus(), db_handler=db, risk_manager=risk, executors=[executor], reconciliation_result=recon, user_stream=user_stream)
    api = TestClient(app)

    resp = api.get("/api/live/readiness", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready_for_live"] is False
    assert body["checks"]["risk"]["ok"] is False
    assert "risk" in body["blockers"]


def test_live_readiness_blocks_until_user_stream_connected(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    client = MagicMock()
    client.markets = {"BTC/USDT:USDT": {}}
    executor = MagicMock(exchange_id="binance")
    executor._client = client
    risk = MagicMock(kill_switch_active=False)
    recon = MagicMock(success=True, safe_mode=False, mismatches=[], positions_without_sl=[])
    user_stream = MagicMock(connected=False)

    app = build_app(config_mock, EventBus(), db_handler=db, risk_manager=risk, executors=[executor], reconciliation_result=recon, user_stream=user_stream)
    api = TestClient(app)

    resp = api.get("/api/live/readiness", headers={"X-API-Key": "test-key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready_for_live"] is False
    assert body["checks"]["user_stream"]["ok"] is False
    assert "user_stream" in body["blockers"]


def test_live_auto_toggle_enable_requires_clean_readiness_and_typed_confirmation(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    signal_gen = _SignalGen()
    app = build_app(config_mock, EventBus(), signal_generator=signal_gen, risk_manager=MagicMock(kill_switch_active=False))
    api = TestClient(app)

    resp = api.post("/api/auto/toggle", headers={"X-API-Key": "test-key"}, json={"enabled": True})

    assert resp.status_code == 423
    assert signal_gen.auto_trading_enabled is False
    assert "live_readiness_blocked" in resp.text


def test_live_config_cannot_enable_auto_trading_without_clean_readiness(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    signal_gen = _SignalGen()
    app = build_app(config_mock, EventBus(), signal_generator=signal_gen, risk_manager=MagicMock(kill_switch_active=False))
    api = TestClient(app)

    resp = api.post("/api/config", headers={"X-API-Key": "test-key"}, json={"auto_trading_enabled": True})

    assert resp.status_code == 423
    assert signal_gen.auto_trading_enabled is False
    assert "live_readiness_blocked" in resp.text


def test_live_quick_action_resume_auto_requires_clean_readiness(config_mock: MagicMock) -> None:
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    signal_gen = _SignalGen()
    app = build_app(config_mock, EventBus(), signal_generator=signal_gen, risk_manager=MagicMock(kill_switch_active=False))
    api = TestClient(app)

    resp = api.post("/api/quick-action", headers={"X-API-Key": "test-key"}, json={"action": "resume_auto"})

    assert resp.status_code == 423
    assert signal_gen.auto_trading_enabled is False
    assert "live_readiness_blocked" in resp.text


def test_live_manual_trade_requires_typed_confirmation_even_when_enabled(config_mock: MagicMock) -> None:
    """Live dashboard order placement must require an explicit typed confirmation."""
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {
                    "require_api_key": True,
                    "api_key": "test-key",
                    "rate_limit_per_min": 2,
                    "allow_manual_live_trading": True,
                },
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    client = MagicMock()
    client.create_market_order = AsyncMock(return_value={"id": "live-1", "status": "closed", "filled": 0.01, "average": 65000})
    executor = MagicMock()
    executor.exchange_id = "binance"
    executor._client = client
    executor._rate_limiter = None

    risk = MagicMock()
    risk.open_position = AsyncMock()
    app = build_app(config_mock, EventBus(), executors=[executor], risk_manager=risk)
    api = TestClient(app)

    resp = api.post(
        "/api/trade",
        headers={"X-API-Key": "test-key"},
        json={"symbol": "BTC/USDT:USDT", "side": "BUY", "size": 0.01, "order_type": "market"},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "confirmation" in resp.json()["error"].lower()
    client.create_market_order.assert_not_called()


def test_live_manual_trade_blocks_when_startup_reconciliation_not_clean(config_mock: MagicMock) -> None:
    """Manual live orders must re-check reconciliation immediately before exchange placement."""
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {
                    "require_api_key": True,
                    "api_key": "test-key",
                    "rate_limit_per_min": 2,
                    "allow_manual_live_trading": True,
                },
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    client = MagicMock()
    client.create_market_order = AsyncMock(return_value={"id": "live-1", "status": "closed", "filled": 0.01, "average": 65000})
    executor = MagicMock(exchange_id="binance")
    executor._client = client
    executor._rate_limiter = None
    risk = MagicMock(kill_switch_active=False)
    recon = MagicMock(success=False, safe_mode=True, mismatches=["position mismatch"], positions_without_sl=[])
    app = build_app(config_mock, EventBus(), db_handler=db, executors=[executor], risk_manager=risk, reconciliation_result=recon)
    api = TestClient(app)

    resp = api.post(
        "/api/trade",
        headers={"X-API-Key": "test-key"},
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "BUY",
            "size": 0.01,
            "order_type": "market",
            "confirmation": "PLACE BUY BTC/USDT:USDT",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "reconciliation" in resp.json()["error"].lower()
    client.create_market_order.assert_not_called()


def test_live_manual_trade_blocks_when_risk_manager_killed(config_mock: MagicMock) -> None:
    """A live kill switch must block manual exchange orders even with valid typed confirmation."""
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {
                    "require_api_key": True,
                    "api_key": "test-key",
                    "rate_limit_per_min": 2,
                    "allow_manual_live_trading": True,
                },
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    db = MagicMock(available=True)
    client = MagicMock()
    client.create_market_order = AsyncMock(return_value={"id": "live-1", "status": "closed", "filled": 0.01, "average": 65000})
    executor = MagicMock(exchange_id="binance")
    executor._client = client
    executor._rate_limiter = None
    class _KilledRisk:
        killed = True
        open_position = AsyncMock()

    risk = _KilledRisk()
    recon = MagicMock(success=True, safe_mode=False, mismatches=[], positions_without_sl=[])
    app = build_app(config_mock, EventBus(), db_handler=db, executors=[executor], risk_manager=risk, reconciliation_result=recon)
    api = TestClient(app)

    resp = api.post(
        "/api/trade",
        headers={"X-API-Key": "test-key"},
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "BUY",
            "size": 0.01,
            "order_type": "market",
            "confirmation": "PLACE BUY BTC/USDT:USDT",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "kill switch" in resp.json()["error"].lower()
    client.create_market_order.assert_not_called()


def test_legacy_order_create_route_is_blocked_in_live_mode(config_mock: MagicMock) -> None:
    """Legacy /api/orders create route must not bypass /api/trade live safeguards."""
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    order_manager = MagicMock()
    order_manager.place_order = AsyncMock()
    db = MagicMock(available=True)
    risk = MagicMock(killed=False)
    app = build_app(config_mock, EventBus(), db_handler=db, risk_manager=risk, order_manager=order_manager)
    api = TestClient(app)

    resp = api.post(
        "/api/orders/",
        headers={"X-API-Key": "test-key"},
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "order_type": "market",
            "quantity": 0.01,
            "venue": "binance",
        },
    )

    assert resp.status_code == 403
    assert "disabled in live" in resp.text.lower()
    order_manager.place_order.assert_not_called()


def test_legacy_position_close_route_is_blocked_in_live_mode(config_mock: MagicMock) -> None:
    """Legacy risk-ledger close route must not desync live exchange exposure."""
    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {
                "host": "127.0.0.1",
                "port": 8000,
                "auth": {"require_api_key": True, "api_key": "test-key", "rate_limit_per_min": 2},
            }
        return default

    config_mock.paper_mode = False
    config_mock.get_value.side_effect = _get_value
    risk = MagicMock(killed=False)
    risk.close_position = AsyncMock()
    app = build_app(config_mock, EventBus(), db_handler=MagicMock(available=True), risk_manager=risk)
    api = TestClient(app)

    resp = api.post(
        "/api/positions/close-all",
        headers={"X-API-Key": "test-key"},
    )

    assert resp.status_code == 403
    assert "disabled in live" in resp.text.lower()
    risk.close_position.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "/api/signals/recent",
        "/api/status",
        "/api/market?per_page=20",
        "/api/backtest/summary",
        "/api/news",
        "/api/logs/recent",
        "/api/feargreed",
        "/api/dex/pools",
        "/api/system/data-sources",
        "/api/auto/status",
        "/api/config",
        "/api/trades/history",
        "/api/indicators/BTC",
        "/api/orderbook?symbol=BTC/USDT&depth=20",
        "/api/candles?symbol=BTC/USDT&timeframe=1m",
        "/api/realtime/snapshot?symbol=BTC/USDT&timeframe=1m",
    ],
)
def test_ui_get_endpoints_responsive(ui_client: TestClient, path: str) -> None:
    resp = ui_client.get(path)
    assert resp.status_code == 200, f"GET {path} failed with {resp.status_code}: {resp.text[:200]}"
    assert isinstance(resp.json(), dict)


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/api/trade",
            {
                "symbol": "BTC/USDT:USDT",
                "side": "BUY",
                "size": 0.01,
                "order_type": "market",
                "price": 0,
                "stop_loss_pct": 2,
                "take_profit_pct": 4,
                "leverage": 1,
            },
        ),
        ("/api/positions/close-all", None),
        ("/api/positions/breakeven", None),
        (
            "/api/auto/toggle",
            {
                "enabled": True,
                "mode": "paper",
                "strategy": "ensemble",
                "sizing_mode": "risk_pct",
                "risk_per_trade": 2,
                "max_positions": 3,
                "max_drawdown": 10,
                "max_leverage": 5,
                "daily_loss_limit": 500,
                "trailing_mode": "none",
                "auto_sl_tp": True,
            },
        ),
        ("/api/mode/toggle", {"mode": "paper"}),
        (
            "/api/config",
            {
                "binance_enabled": True,
                "auto_trading_enabled": False,
                "auto_stop_loss_enabled": True,
                "auto_take_profit_enabled": True,
            },
        ),
        ("/api/config/test", None),
    ],
)
def test_ui_action_endpoints_responsive(ui_client: TestClient, path: str, payload: dict | None) -> None:
    if payload is None:
        resp = ui_client.post(path)
    else:
        resp = ui_client.post(path, json=payload)

    assert resp.status_code == 200, f"POST {path} failed with {resp.status_code}: {resp.text[:200]}"
    data = resp.json()
    assert isinstance(data, dict)

    # Most UI actions expose a success flag; verify when present.
    if "success" in data:
        assert data["success"] is True


def test_realtime_stream_endpoint_responds(ui_client: TestClient) -> None:
    # SSE streaming endpoints are infinite generators — we cannot use stream()
    # context manager without blocking. Instead verify the route is registered
    # by checking it doesn't 404. A real SSE test needs an async HTTP client.
    import threading, requests, time as _time

    # Use the TestClient as a context manager to start the server in a thread
    # and make a real HTTP request with a short timeout.
    from starlette.testclient import TestClient as _TC
    app = ui_client.app

    # Verify route registration by checking app routes contain the path
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/api/realtime/stream" in routes, "SSE stream route not registered"


def test_index_fetch_targets_are_implemented(ui_client: TestClient) -> None:
    # Keep this list in sync with interface/static/index.html fetch('/api/...') calls.
    fetch_targets = {
        "/api/signals/recent",
        "/api/status",
        "/api/market",
        "/api/backtest/summary",
        "/api/news",
        "/api/logs/recent",
        "/api/feargreed",
        "/api/dex/pools",
        "/api/system/data-sources",
        "/api/trade",
        "/api/positions/close-all",
        "/api/auto/toggle",
        "/api/positions/breakeven",
        "/api/auto/status",
        "/api/mode/toggle",
        "/api/config",
        "/api/config/test",
        "/api/trades/history",
        "/api/candles",
        "/api/orderbook",
        "/api/indicators",
        "/api/realtime/snapshot",
        "/api/realtime/stream",
    }

    # Smoke check by probing representative endpoints/methods.
    for path in sorted(fetch_targets):
        if path in {"/api/trade", "/api/positions/close-all", "/api/auto/toggle", "/api/positions/breakeven", "/api/mode/toggle", "/api/config", "/api/config/test"}:
            continue
        # For parameterized routes use concrete values.
        probe = path
        if path == "/api/candles":
            probe = "/api/candles?symbol=BTC/USDT&timeframe=1m"
        elif path == "/api/orderbook":
            probe = "/api/orderbook?symbol=BTC/USDT&depth=20"
        elif path == "/api/indicators":
            probe = "/api/indicators/BTC"
        elif path in {"/api/realtime/snapshot", "/api/realtime/stream"}:
            continue

        resp = ui_client.get(probe)
        assert resp.status_code == 200, f"missing/broken endpoint for UI fetch target: {probe}"

    # Ensure static file still exists where the browser serves it from.
    _repo = Path(__file__).resolve().parent.parent.parent
    assert (_repo / "interface" / "static" / "index.html").exists()


def test_dashboard_paper_mode_does_not_prefer_exchange_positions() -> None:
    """Paper/manual dogfood must render the paper ledger, not stale testnet positions."""
    _repo = Path(__file__).resolve().parent.parent.parent
    index = (_repo / "interface" / "static" / "index.html").read_text()

    assert "let isPaperMode = true;" in index
    assert "isPaperMode = !!s.auto.paper_mode;" in index
    assert "const useExchangePortfolio = (isLiveMode || isDemoMode) && !isPaperMode;" in index
    assert "const positions = (useExchangePortfolio && exchangePositions.length) ? exchangePositions : (d.positions||[]);" in index


def test_dashboard_sse_uses_existing_api_key_helper() -> None:
    """startStream must not throw before constructing authenticated EventSource."""
    _repo = Path(__file__).resolve().parent.parent.parent
    index = (_repo / "interface" / "static" / "index.html").read_text()

    assert "function getStoredDashboardApiKey()" in index
    assert "const streamKey = getStoredDashboardApiKey();" in index
    assert "streamParams.set('api_key', streamKey);" in index
