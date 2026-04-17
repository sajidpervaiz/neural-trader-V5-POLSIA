"""Integration tests for dashboard orders/risk API routes."""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from core.circuit_breaker import CircuitBreaker
from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.order_manager import OrderManager
from execution.risk_manager import RiskManager
import interface.dashboard_api as dashboard_api
from interface.dashboard_api import build_app


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
            }
        if keys == ("risk", "stop_loss_pct"):
            return 0.015
        if keys == ("risk", "take_profit_pct"):
            return 0.03
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {"enabled": False}
        if keys == ("dex", "enabled"):
            return False
        if keys == ("rust_services", "enabled"):
            return False
        if keys == ("ts_dex_layer", "enabled"):
            return False
        return default

    cfg.get_value.side_effect = _get_value
    return cfg


@pytest.fixture
def api_context(config_mock: MagicMock) -> dict:
    bus = EventBus()
    breaker = CircuitBreaker()
    risk_mgr = RiskManager(config_mock, bus)
    risk_mgr._equity = 100_000.0
    order_mgr = OrderManager(config_mock, bus, breaker)

    app = build_app(config_mock, bus, risk_manager=risk_mgr, order_manager=order_mgr)
    return {
        "client": TestClient(app),
        "risk_mgr": risk_mgr,
    }


@pytest.fixture
def api_client(api_context: dict) -> TestClient:
    return api_context["client"]


def test_dashboard_serves_interactive_tradingview_controls() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/')

    assert resp.status_code == 200
    html = resp.text
    assert 'runTvAction(' in html
    assert 'runChartTool(' in html
    assert 'tv-action-btn' in html


def test_dashboard_serves_extended_tradingview_workspace_features() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/')

    assert resp.status_code == 200
    html = resp.text
    assert 'openSymbolSearch(' in html
    assert 'handleHotkeys(' in html
    assert 'Search /' in html


def test_dashboard_serves_official_tradingview_widget_integration() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/')

    assert resp.status_code == 200
    html = resp.text
    assert 'tradingview-widget-container' in html
    assert 'initTradingViewWidget(' in html
    assert 's3.tradingview.com/external-embedding/embed-widget-advanced-chart.js' in html


def test_auto_status_exposes_live_binance_testnet_metadata() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("risk",):
            return {"default_leverage": 5.0}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)

    resp = client.get("/api/auto/status")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["mode"] == "live"
    assert payload["exchange"] == "binance"
    assert payload["testnet"] is True
    assert payload["label"] == "BINANCE DEMO"


def test_mode_toggle_switches_runtime_to_live_demo() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg._data = {
        "system": {"paper_mode": True},
        "exchanges": {
            "binance": {
                "enabled": True,
                "testnet": True,
                "api_key": "demo-key",
                "api_secret": "demo-secret",
            }
        },
    }

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return cfg._data.get("exchanges", {})
        if keys == ("risk",):
            return {"default_leverage": 5.0}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)

    resp = client.post("/api/mode/toggle", json={"mode": "live"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is False
    assert "Cannot switch to live" in payload["error"]
    assert cfg.paper_mode is True

    status_resp = client.get("/api/auto/status")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["mode"] == "paper"
    assert status_payload["testnet"] is False
    assert status_payload["label"] == "PAPER"


def test_config_exposes_ai_agent_settings() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "api_key": "abc", "api_secret": "def", "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {
                "enabled": True,
                "provider": "claude",
                "model": "claude-3-5-sonnet-latest",
                "api_key": "test-key",
                "timeout_seconds": 8.0,
                "remote_weight": 0.35,
            }
        return default

    cfg.get_value.side_effect = _get_value
    cfg._data = {}

    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/api/config')

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ai_agent_provider"] == "claude"
    assert payload["ai_agent_model"] == "claude-3-5-sonnet-latest"
    assert payload["ai_agent_api_key"] == "****"
    assert payload["ai_agent_enabled"] is True


def test_config_save_updates_ai_agent_runtime() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg.auto_trading_enabled = True

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.post('/api/config', json={
        "auto_trading_enabled": True,
        "ai_agent_enabled": True,
        "ai_agent_provider": "claude",
        "ai_agent_model": "claude-3-5-sonnet-latest",
        "ai_agent_api_key": "secret-key",
        "ai_agent_timeout_seconds": 9.0,
        "ai_agent_remote_weight": 0.4,
    })

    assert resp.status_code == 200
    sg.configure_agent.assert_called_once()
    payload = sg.configure_agent.call_args.kwargs.get("payload") or sg.configure_agent.call_args.args[0]
    assert payload["provider"] == "claude"
    assert payload["api_key"] == "secret-key"


def test_api_quality_returns_partial_total_when_components_exist() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg._last_quality_breakdown = {
        "total": 0,
        "components": {
            "htf_trend": 75,
            "technical_confluence": 26,
            "smc_confluence": 0,
            "volume_flow": 0,
            "regime": 0,
            "ml_confidence": 0,
            "liquidity_depth": 0,
        },
        "rejected_at": "L3_Technical",
    }

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.get('/api/quality')

    assert resp.status_code == 200
    payload = resp.json()
    assert payload['total'] > 0


def test_api_layers_coerces_zero_score_unknown_to_fail() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg._last_layer_status = {
        "smart_money_concepts": "UNKNOWN",
        "smart_money_concepts_detail": "score=0",
        "signal_quality": "UNKNOWN",
        "signal_quality_detail": "score=0",
    }
    sg._last_quality_breakdown = {
        "total": 0,
        "components": {
            "smc_confluence": 0,
            "volume_flow": 0,
            "regime": 0,
            "ml_confidence": 0,
        },
        "rejected_at": "L3_Technical",
    }

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.get('/api/layers')

    assert resp.status_code == 200
    layers = {item["name"]: item for item in resp.json()["layers"]}
    assert layers["Smart Money Concepts"]["status"] == "FAIL"
    assert layers["Signal Quality"]["status"] == "FAIL"


def test_agent_chat_endpoint_returns_response() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {"enabled": True, "provider": "claude"}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg.chat_with_agent.return_value = {"success": True, "reply": "Agent online", "provider": "local"}

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.post('/api/agent/chat', json={"message": "hello bot"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["reply"] == "Agent online"


def test_market_endpoint_supports_large_per_page() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/api/market', params={'per_page': 120})

    assert resp.status_code == 200
    payload = resp.json()
    assert 'coins' in payload
    assert len(payload['coins']) <= 120


def test_market_endpoint_refreshes_when_cached_result_is_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def json(self, content_type=None):
            return [
                {
                    "symbol": f"coin{i}",
                    "name": f"Coin {i}",
                    "current_price": float(i),
                    "price_change_percentage_24h": 0.0,
                    "total_volume": 1000.0,
                    "high_24h": float(i),
                    "low_24h": float(i),
                    "market_cap": 1000000.0 - i,
                }
                for i in range(120)
            ]

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def get(self, *args, **kwargs):
            return _FakeResponse()

    fake_aiohttp = types.SimpleNamespace(
        ClientSession=_FakeSession,
        ClientTimeout=lambda total: object(),
    )

    cfg.get_value.side_effect = _get_value
    monkeypatch.setitem(sys.modules, 'aiohttp', fake_aiohttp)
    dashboard_api._market_cache["coins"] = [{"symbol": "BTC"} for _ in range(20)]
    dashboard_api._market_cache["ts"] = __import__("time").time()

    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/api/market', params={'per_page': 120})

    assert resp.status_code == 200
    payload = resp.json()
    assert 'coins' in payload
    assert len(payload['coins']) == 120


def test_realtime_snapshot_uses_expanded_market_payload() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    dashboard_api._market_cache["coins"] = [{"symbol": f"C{i}"} for i in range(120)]
    dashboard_api._market_cache["ts"] = __import__("time").time()

    app = build_app(cfg, EventBus())
    client = TestClient(app)
    resp = client.get('/api/realtime/snapshot', params={'symbol': 'BTC/USDT', 'timeframe': '1m'})

    assert resp.status_code == 200
    payload = resp.json()
    assert 'market' in payload
    assert len(payload['market']['coins']) >= 100


def test_strategy_suggestion_endpoint_returns_payload() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg.get_strategy_suggestion.return_value = {
        "symbol": "BTC/USDT:USDT",
        "action": "wait",
        "strategy": "trend_following",
        "reason": "technical gate weak",
    }

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.get('/api/strategy/suggest', params={'symbol': 'BTC/USDT:USDT'})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload['symbol'] == 'BTC/USDT:USDT'
    assert payload['strategy'] == 'trend_following'


def test_quick_action_endpoint_controls_bot() -> None:
    cfg = MagicMock()
    cfg.paper_mode = False
    cfg._data = {}

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000, "auth": {"allow_unauthenticated_non_paper": True}}
        if keys == ("exchanges",):
            return {"binance": {"enabled": True, "testnet": True}}
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value
    sg = MagicMock()
    sg.chat_with_agent.return_value = {"success": True, "reply": "ok"}

    app = build_app(cfg, EventBus(), signal_generator=sg)
    client = TestClient(app)
    resp = client.post('/api/quick-action', json={'action': 'pause_auto'})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload['success'] is True
    assert 'action' in payload


def test_config_trading_mode_switch_updates_runtime_live_demo_state() -> None:
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg._data = {
        "exchanges": {
            "binance": {
                "enabled": True,
                "testnet": True,
                "api_key": "demo-key",
                "api_secret": "demo-secret",
            }
        }
    }

    def _get_value(*keys, default=None):
        if keys == ("monitoring", "dashboard_api"):
            return {"host": "127.0.0.1", "port": 8000}
        if keys == ("exchanges",):
            return cfg._data.get("exchanges", {})
        if keys == ("dex",):
            return {}
        if keys == ("notifications", "telegram"):
            return {}
        if keys == ("risk",):
            return {}
        if keys == ("ai_agent",):
            return {}
        return default

    cfg.get_value.side_effect = _get_value

    app = build_app(cfg, EventBus())
    client = TestClient(app)

    resp = client.post('/api/config/trading-mode?confirmation=true', json='live')

    assert resp.status_code == 200
    payload = resp.json()
    assert payload['mode'] == 'live'
    assert cfg.paper_mode is False

    auto_status = client.get('/api/auto/status')
    assert auto_status.status_code == 200
    auto_payload = auto_status.json()
    assert auto_payload['mode'] == 'live'
    assert auto_payload['testnet'] is True
    assert auto_payload['label'] == 'BINANCE DEMO'


def test_orders_create_list_and_cancel(api_client: TestClient) -> None:
    create_resp = api_client.post(
        "/orders/",
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.1,
            "price": 50000.0,
            "time_in_force": "GTC",
            "venue": "binance",
            "reduce_only": False,
            "client_order_id": "test-order-1",
        },
    )
    assert create_resp.status_code == 200
    payload = create_resp.json()
    order_id = payload["order_id"]
    assert payload["status"] == "PENDING"
    assert payload["symbol"] == "BTC/USDT:USDT"

    list_resp = api_client.get("/orders/")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert any(o["order_id"] == order_id for o in listed)

    open_resp = api_client.get("/orders/open")
    assert open_resp.status_code == 200
    open_orders = open_resp.json()
    assert any(o["order_id"] == order_id for o in open_orders)

    cancel_resp = api_client.delete(f"/orders/{order_id}", params={"venue": "binance"})
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


def test_risk_limits_check_and_stress(api_client: TestClient) -> None:
    limits_resp = api_client.get("/risk/limits")
    assert limits_resp.status_code == 200
    limits = limits_resp.json()
    assert "max_position_value" in limits
    assert "leverage_limit" in limits

    check_resp = api_client.post(
        "/risk/check",
        params={
            "symbol": "ETH/USDT:USDT",
            "side": "buy",
            "quantity": 1.5,
            "price": 2500.0,
            "user_id": "u1",
        },
    )
    assert check_resp.status_code == 200
    check_payload = check_resp.json()
    assert "passed" in check_payload
    assert "reason" in check_payload

    stress_resp = api_client.get("/risk/stress-test", params={"scenario": "flash_crash"})
    assert stress_resp.status_code == 200
    stress_payload = stress_resp.json()
    assert stress_payload["status"] == "completed"
    assert "report" in stress_payload


def test_risk_circuit_breaker_status_and_reset(api_client: TestClient) -> None:
    status_resp = api_client.get("/risk/circuit-breaker")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert "overall_status" in status_payload
    assert "venues" in status_payload

    reset_resp = api_client.post("/risk/circuit-breaker/reset", params={"venue": "global_risk"})
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "reset"


def test_orders_place_compat_alias(api_client: TestClient) -> None:
    resp = api_client.post(
        "/orders/place",
        json={
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.2,
            "price": 51000.0,
            "time_in_force": "GTC",
            "venue": "binance",
            "reduce_only": False,
            "client_order_id": "compat-alias-1",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "PENDING"
    assert payload["client_order_id"] == "compat-alias-1"


def test_positions_routes_are_manager_backed(api_context: dict) -> None:
    client: TestClient = api_context["client"]
    risk_mgr: RiskManager = api_context["risk_mgr"]

    signal = TradingSignal(
        exchange="binance",
        symbol="ETHUSDT",
        direction="long",
        score=0.8,
        technical_score=0.8,
        ml_score=0.8,
        sentiment_score=0.0,
        macro_score=0.0,
        news_score=0.0,
        orderbook_score=0.0,
        regime="integration",
        regime_confidence=1.0,
        price=3000.0,
        atr=30.0,
        stop_loss=2950.0,
        take_profit=3060.0,
        timestamp=1700000000,
    )
    asyncio.get_event_loop().run_until_complete(risk_mgr.open_position(signal, size=1500.0))

    list_resp = client.get("/positions/")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) == 1
    assert items[0]["symbol"] == "ETHUSDT"

    summary_resp = client.get("/positions/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["total_positions"] == 1

    get_resp = client.get("/positions/ETHUSDT", params={"venue": "binance"})
    assert get_resp.status_code == 200

    close_resp = client.post("/positions/ETHUSDT/close", params={"venue": "binance"})
    assert close_resp.status_code == 200
    assert close_resp.json()["status"] == "closed"
