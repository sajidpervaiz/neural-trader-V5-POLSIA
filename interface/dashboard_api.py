from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import json
import os
import re
import socket
import time
from collections import OrderedDict, deque
from typing import Any

from loguru import logger

from core.error_handling import sanitize_exception

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from sse_starlette.sse import EventSourceResponse  # type: ignore[import-untyped]
    import uvicorn
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

from pathlib import Path


from core.config import Config
from core.event_bus import EventBus
from execution.executor_contract import executor_contract_status
from interface.routes.config import router as config_router, configure_config_routes
from interface.routes.orders import router as orders_router, configure_order_routes
from interface.routes.positions import router as positions_router, configure_positions_routes
from interface.routes.risk import router as risk_router, configure_risk_routes

# Expose a top-level FastAPI app for uvicorn
def _default_config():
    # Minimal config for dashboard boot
    class DummyConfig:
        paper_mode = True
        def get_value(self, *args, **kwargs):
            return {}
    return DummyConfig()

def _default_event_bus():
    class DummyEventBus:
        def subscribe(self, *a, **k):
            pass
        async def publish(self, *a, **k):
            pass
    return DummyEventBus()


# Only create app if FastAPI is available (must be after build_app is defined)
_APP_ARGS = dict(
    config=_default_config(),
    event_bus=_default_event_bus(),
    risk_manager=None,
    data_manager=None,
    order_manager=None,
    db_handler=None,
    cache=None,
    signal_generator=None,
    news_feed=None,
    orderbook_feed=None,
    sentiment_manager=None,
    dex_feed=None,
)

# Defer app creation until after build_app is defined


# ── In-memory ring buffers for event-sourced data ────────────────────────────
_news_buffer: deque[dict[str, Any]] = deque(maxlen=50)
_orderbook_cache: dict[str, dict[str, Any]] = {}
_log_buffer: deque[dict[str, Any]] = deque(maxlen=200)
_market_cache: dict[str, Any] = {"coins": [], "ts": 0.0}  # TTL cache for market data

# ── TTL cache for exchange REST calls (avoids hammering exchange on every poll) ─
_exchange_cache: dict[str, tuple[float, Any]] = {}  # key -> (expiry_ts, data)
_EXCHANGE_CACHE_TTL = 5.0  # seconds

# ── /api/layers stale-while-revalidate cache ─────────────────────────────
# Cold get_quality_preview runs the full ML+SMC+HTF pipeline; in degraded
# environments (DNS timeouts saturating the threadpool) cold builds have been
# observed taking 30s+. The pattern below guarantees a sub-3s response in all
# conditions:
#   • Hit within TTL → return cached payload immediately.
#   • Hit past TTL but within STALE window → return stale, kick off a
#     background refresh so the next call is hot.
#   • Total miss → race the build against COLD_BUDGET. If we win, cache + return.
#     If we lose, fall back to the placeholder. The build keeps running and
#     populates the cache for the next caller.
_layers_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_LAYERS_CACHE_TTL = 30.0       # fresh window
_LAYERS_CACHE_STALE = 600.0    # serve-stale window (10 min)
_LAYERS_COLD_BUDGET = 3.0      # max we'll block a request waiting for a build
_layers_refresh_lock: asyncio.Lock | None = None  # lazy-created in api_layers

def _cache_get(key: str) -> Any | None:
    entry = _exchange_cache.get(key)
    if entry and entry[0] > time.monotonic():
        return entry[1]
    return None

def _cache_set(key: str, data: Any) -> None:
    _exchange_cache[key] = (time.monotonic() + _EXCHANGE_CACHE_TTL, data)

# Loguru sink that captures recent log lines for the /api/logs/recent endpoint

def _log_sink(message: Any) -> None:
    record = message.record
    _log_buffer.append({
        "ts": int(record["time"].timestamp() * 1000),
        "level": record["level"].name,
        "message": str(record["message"])[:300],
    })


# Place this after build_app is defined
# (MUST be after the build_app function definition)

# ── Input validation for query parameters ─────────────────────────────────
_VALID_SYMBOL = re.compile(r"^[A-Za-z0-9]{1,20}(/[A-Za-z0-9]{1,10})?(:[A-Za-z0-9]{1,10})?$")
_VALID_TIMEFRAME = re.compile(r"^[1-9][0-9]?[smhdwM]$")

def _validate_symbol(symbol: str) -> str:
    """Validate symbol format; raise ValueError on bad input."""
    if not _VALID_SYMBOL.match(symbol):
        raise ValueError(f"Invalid symbol format: {symbol!r}")
    return symbol

def _validate_timeframe(timeframe: str) -> str:
    """Validate timeframe format; raise ValueError on bad input."""
    if not _VALID_TIMEFRAME.match(timeframe):
        raise ValueError(f"Invalid timeframe format: {timeframe!r}")
    return timeframe


def _redact_config_value(value: Any) -> Any:
    """Return a stable JSON-safe config object with secrets removed before hashing."""
    secret_tokens = ("key", "secret", "password", "token", "passphrase", "private")
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if any(token in str(key).lower() for token in secret_tokens):
                redacted[str(key)] = "***"
            else:
                redacted[str(key)] = _redact_config_value(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_config_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _config_hash(config: Config) -> str:
    """Compute a deterministic redacted config fingerprint for audit/readiness."""
    snapshot: dict[str, Any] = {"paper_mode": bool(getattr(config, "paper_mode", True))}
    for section in (
        "system",
        "exchanges",
        "storage",
        "monitoring",
        "risk",
        "signals",
        "auto_trading",
    ):
        try:
            snapshot[section] = config.get_value(section, default={}) or {}
        except Exception:
            snapshot[section] = {}
    payload = json.dumps(_redact_config_value(snapshot), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_app(
    config: Config,
    event_bus: EventBus,
    risk_manager: Any = None,
    data_manager: Any = None,
    order_manager: Any = None,
    db_handler: Any = None,
    cache: Any = None,
    signal_generator: Any = None,
    *,
    news_feed: Any = None,
    orderbook_feed: Any = None,
    sentiment_manager: Any = None,
    dex_feed: Any = None,
    executors: list[Any] | None = None,
    user_stream: Any = None,
    reconciliation_result: Any = None,
    periodic_reconciler: Any = None,
    state_machine: Any = None,
    uptime_tracker: Any = None,
    sqlite_store: Any = None,
    audit_repo: Any = None,
    metrics: Any = None,
    geopolitical_feed: Any = None,
    market_data_integrity: Any = None,
) -> Any:
    if not _FASTAPI:
        return None

    app = FastAPI(
        title="NUERAL-TRADER-5",
        description="Hybrid Rust+TypeScript+Python trading engine",
        version="4.0.0",
    )
    app.state.started_at = int(time.time())
    app.state.config_hash = _config_hash(config)
    app.state.order_manager = order_manager
    app.state.risk_manager = risk_manager
    app.state.data_manager = data_manager
    app.state.market_data_integrity = market_data_integrity
    static_dir = Path(__file__).resolve().parent / "static"
    static_index = static_dir / "index.html"

    dashboard_cfg = config.get_value("monitoring", "dashboard_api", default={}) or {}
    cors_origins = dashboard_cfg.get("allow_origins") or ["http://localhost", "http://127.0.0.1"]
    # Wildcard CORS is never safe — even in paper mode it lets any site the user
    # visits issue state-changing calls against the local dashboard.
    if "*" in cors_origins:
        logger.warning("CORS wildcard '*' removed — falling back to localhost-only origins")
        cors_origins = [o for o in cors_origins if o != "*"] or ["http://localhost", "http://127.0.0.1"]
    auth_cfg = dashboard_cfg.get("auth", {}) if hasattr(dashboard_cfg, "get") else {}
    if not isinstance(auth_cfg, dict):
        auth_cfg = {}

    require_api_key = bool(auth_cfg.get("require_api_key", False))
    api_key = str(auth_cfg.get("api_key", "") or "").strip()
    rate_limit_per_min = int(auth_cfg.get("rate_limit_per_min", 120))
    allow_unauthenticated_non_paper = bool(auth_cfg.get("allow_unauthenticated_non_paper", False))
    trusted_proxy_hops = int(auth_cfg.get("trusted_proxy_hops", 0))

    # Secure-by-default posture: non-paper mode requires API auth unless explicitly overridden.
    if not config.paper_mode and not require_api_key and not allow_unauthenticated_non_paper:
        require_api_key = True
        logger.warning(
            "Enabling API key requirement automatically for non-paper mode; "
            "set monitoring.dashboard_api.auth.allow_unauthenticated_non_paper=true to override"
        )
    # /health leaks equity/positions/kill-switch state — gate it like everything else.
    # /livez is a plain liveness probe, safe to expose.
    exempt_paths = {"/", "/livez"}
    if config.paper_mode:
        exempt_paths.update({"/docs", "/openapi.json", "/redoc"})

    # LRU-bounded in-memory limiter — prevents unbounded growth from unique-IP floods.
    _MAX_RL_IPS = 10_000
    ip_rate_state: OrderedDict[str, dict[str, int]] = OrderedDict()

    def _client_ip(request: Request) -> str:
        if trusted_proxy_hops > 0:
            xff = request.headers.get("x-forwarded-for", "") or ""
            chain = [p.strip() for p in xff.split(",") if p.strip()]
            if len(chain) >= trusted_proxy_hops:
                return chain[-trusted_proxy_hops]
        return request.client.host if request.client else "unknown"

    def _risk_kill_switch_active() -> bool:
        """Return the canonical kill-switch state across RiskManager variants."""
        if risk_manager is None:
            return True
        for attr in ("kill_switch_active", "killed"):
            value = getattr(risk_manager, attr, None)
            if isinstance(value, bool):
                return value
        try:
            snap = risk_manager.get_risk_snapshot()
            if isinstance(snap, dict):
                return bool(snap.get("kill_switch_active", False))
        except Exception:
            logger.warning("Unable to read risk-manager kill-switch state; failing closed")
            return True
        return False

    def _live_readiness_blockers_for_activation() -> list[str]:
        """Fail-closed live activation gate shared by every auto-enable path."""
        if config.paper_mode:
            return []
        blockers: list[str] = []
        if not bool(getattr(db_handler, "available", False)):
            blockers.append("audit_db")
        if require_api_key and not api_key:
            blockers.append("dashboard_auth")
        exchange_count = 0
        live_clients = 0
        for ex in executors or []:
            exchange_count += 1
            client = getattr(ex, "_client", None)
            markets = getattr(client, "markets", None) if client is not None else None
            if client is not None and markets:
                live_clients += 1
        if exchange_count <= 0 or live_clients != exchange_count:
            blockers.append("exchange")
        if risk_manager is None or _risk_kill_switch_active():
            blockers.append("risk")
        if user_stream is None or not bool(getattr(user_stream, "connected", False)):
            blockers.append("user_stream")
        if reconciliation_result is None:
            blockers.append("reconciliation")
        else:
            recon_mismatches = list(getattr(reconciliation_result, "mismatches", []) or [])
            recon_positions_without_sl = list(getattr(reconciliation_result, "positions_without_sl", []) or [])
            if (
                not bool(getattr(reconciliation_result, "success", False))
                or bool(getattr(reconciliation_result, "safe_mode", False))
                or recon_mismatches
                or recon_positions_without_sl
            ):
                blockers.append("reconciliation")
        return blockers

    def _require_live_auto_activation_allowed(body: dict[str, Any]) -> None:
        """Block every live auto-trading activation unless readiness and typed confirmation are clean."""
        if config.paper_mode:
            return
        blockers = _live_readiness_blockers_for_activation()
        if blockers:
            raise HTTPException(
                status_code=423,
                detail={"error": "live_readiness_blocked", "blockers": blockers},
            )
        confirmation = str(body.get("confirm_live_auto_trading", "") or "").strip()
        if confirmation != "ENABLE LIVE AUTO TRADING":
            raise HTTPException(
                status_code=400,
                detail={"error": "typed_confirmation_required", "required": "ENABLE LIVE AUTO TRADING"},
            )

    @app.middleware("http")
    async def api_guard(request: Request, call_next):
        path = request.url.path
        if path in exempt_paths or path.startswith("/static/"):
            return await call_next(request)

        authenticated = False
        if require_api_key:
            if not api_key:
                return JSONResponse(status_code=503, content={"detail": "api_auth_misconfigured"})
            provided = request.headers.get("x-api-key") or ""
            if not provided and path == "/api/realtime/stream":
                # Browser EventSource cannot send custom headers. Permit the
                # dashboard to pass the same API key as a query parameter for
                # this SSE endpoint only; all normal fetch/XHR calls continue
                # to use X-API-Key / Authorization headers.
                provided = request.query_params.get("api_key", "") or ""
            if not provided:
                auth_header = request.headers.get("authorization", "") or ""
                if auth_header.lower().startswith("bearer "):
                    provided = auth_header[7:].strip()
            if not hmac.compare_digest(provided.encode("utf-8"), api_key.encode("utf-8")):
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
            authenticated = True

        # Do not throttle the authenticated operator UI against itself.  The
        # dashboard performs many parallel polling calls on load; counting those
        # before auth made the secured browser UI degrade into 429s and stale
        # zeros.  Unauthenticated/open deployments remain IP-rate-limited.
        if rate_limit_per_min > 0 and not authenticated:
            ip = _client_ip(request)
            now = int(time.time())
            state = ip_rate_state.get(ip) or {"window_start": now, "count": 0}
            if now - state["window_start"] >= 60:
                state["window_start"] = now
                state["count"] = 0
            state["count"] += 1
            ip_rate_state[ip] = state
            ip_rate_state.move_to_end(ip)
            if len(ip_rate_state) > _MAX_RL_IPS:
                ip_rate_state.popitem(last=False)
            if state["count"] > rate_limit_per_min:
                return JSONResponse(status_code=429, content={"detail": "rate_limit_exceeded"})

        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    configure_order_routes(order_manager, config=config, risk_manager=risk_manager, db_handler=db_handler)
    configure_risk_routes(risk_manager)
    configure_positions_routes(risk_manager, config=config, db_handler=db_handler)
    configure_config_routes(config, risk_manager, order_manager, executors=executors)
    # Mount existing routers under /api prefix so UI /api/* calls work
    app.include_router(config_router, prefix="/api")
    app.include_router(orders_router, prefix="/api")
    app.include_router(positions_router, prefix="/api")
    app.include_router(risk_router, prefix="/api")
    # Also keep original paths for backwards compatibility
    app.include_router(config_router)
    app.include_router(orders_router)
    app.include_router(positions_router)
    app.include_router(risk_router)

    # ── Subscribe to event bus for caching live data ──────────────────────
    async def _cache_news(payload: Any) -> None:
        _news_buffer.append({
            "ts": int(payload.get("timestamp", time.time()) * 1000),
            "title": payload.get("title", ""),
            "sentiment": "bullish" if payload.get("sentiment", 0) > 0.15 else ("bearish" if payload.get("sentiment", 0) < -0.15 else "neutral"),
            "score": payload.get("sentiment", 0),
        })

    async def _cache_orderbook(payload: Any) -> None:
        key = f"{payload.get('exchange', 'binance')}:{payload.get('symbol', '')}"
        _orderbook_cache[key] = {
            "exchange": payload.get("exchange", "binance"),
            "symbol": payload.get("symbol", ""),
            "bids": payload.get("bids", []),
            "asks": payload.get("asks", []),
            "ts": time.time(),
        }

    event_bus.subscribe("NEWS_SENTIMENT", _cache_news)
    event_bus.subscribe("ORDERBOOK_UPDATE", _cache_orderbook)

    @app.get("/")
    async def root() -> Any:
        if static_index.exists():
            return FileResponse(str(static_index))
        return {
            "message": "NUERAL-TRADER-5 Dashboard",
            "docs": "/docs",
            "health": "/health",
            "positions": "/positions",
            "config": "/config/summary",
        }

    @app.get("/livez")
    async def livez() -> dict[str, Any]:
        """Unauthenticated liveness probe — no sensitive data."""
        return {"ok": True, "uptime_seconds": int(time.time()) - app.state.started_at}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "paper_mode": config.paper_mode,
            "timestamp": int(time.time()),
            "uptime_seconds": int(time.time()) - app.state.started_at,
        }
        component_errors: list[str] = []

        # Risk snapshot
        if risk_manager is not None:
            try:
                rm = risk_manager
                result["risk"] = {
                    "equity": getattr(rm, "equity", 0.0),
                    "open_positions": len(getattr(rm, "positions", {})),
                    "circuit_breaker_active": getattr(rm, "circuit_breaker_active", False),
                    "kill_switch": getattr(rm, "kill_switch", False),
                    "daily_loss": getattr(rm, "daily_loss", 0.0),
                }
            except Exception as exc:
                logger.error("health: risk snapshot failed: {}", sanitize_exception(exc))
                logger.opt(exception=True).debug("health: risk snapshot stack trace")
                result["risk"] = {"error": "unavailable"}
                component_errors.append(f"risk:{type(exc).__name__}")

        # Safe mode
        try:
            from core.safe_mode import SafeModeManager
            sm = getattr(risk_manager, "safe_mode", None) if risk_manager else None
            if sm is not None and isinstance(sm, SafeModeManager):
                status = sm.get_status()
                result["safe_mode"] = {
                    "active": status.get("safe_mode_active", False),
                    "reasons": [r["reason"] for r in status.get("active_reasons", [])],
                }
        except Exception as exc:
            logger.error("health: safe_mode status failed: {}", sanitize_exception(exc))
            logger.opt(exception=True).debug("health: safe_mode status stack trace")
            component_errors.append(f"safe_mode:{type(exc).__name__}")

        # Alert manager
        try:
            from monitoring.alert_manager import AlertManager
            am = getattr(app.state, "alert_manager", None)
            if am is not None and isinstance(am, AlertManager):
                result["alerts"] = am.get_status()
        except Exception as exc:
            logger.error("health: alert_manager status failed: {}", sanitize_exception(exc))
            logger.opt(exception=True).debug("health: alert_manager status stack trace")
            component_errors.append(f"alerts:{type(exc).__name__}")

        # Component health (if a HealthChecker is attached)
        try:
            from monitoring.health_checks import HealthChecker
            hc = getattr(app.state, "health_checker", None)
            if hc is not None and isinstance(hc, HealthChecker):
                hcr = await hc.check_all_components()
                result["status"] = hcr.overall_status.value.lower()
                result["components"] = {
                    name: {
                        "status": comp.status.value,
                        "latency_ms": round(comp.latency_ms, 1),
                        "message": comp.message,
                    }
                    for name, comp in hcr.components.items()
                }
        except Exception as exc:
            logger.error("health: component checker failed: {}", sanitize_exception(exc))
            logger.opt(exception=True).debug("health: component checker stack trace")
            component_errors.append(f"components:{type(exc).__name__}")

        if component_errors:
            # Only downgrade if HealthChecker didn't already set a richer status.
            if result["status"] == "ok":
                result["status"] = "degraded"
            result["component_errors"] = component_errors

        return result

    @app.get("/positions")
    async def get_positions() -> dict[str, Any]:
        if risk_manager is None:
            return {"positions": [], "equity": 0.0}
        positions = risk_manager.positions
        return {
            "positions": [
                {
                    "exchange": p.exchange,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "pnl": p.pnl,
                    "pnl_pct": p.pnl_pct,
                }
                for p in positions.values()
            ],
            "equity": risk_manager.equity,
            "count": len(positions),
        }

    @app.get("/config/summary")
    async def config_summary() -> dict[str, Any]:
        return {
            "paper_mode": config.paper_mode,
            "enabled_exchanges": [
                k for k, v in (config.get_value("exchanges") or {}).items()
                if v.get("enabled", False)
            ],
            "dex_enabled": config.get_value("dex", "enabled") or False,
            "rust_enabled": config.get_value("rust_services", "enabled") or False,
            "ts_dex_enabled": config.get_value("ts_dex_layer", "enabled") or False,
        }

    @app.get("/features")
    async def features_status() -> dict[str, Any]:
        dex_config = config.get_value("dex", default={}) or {}
        macro_config = config.get_value("macro", default={}) or {}
        ts_dex_config = config.get_value("ts_dex_layer", default={}) or {}
        funding_cfg = macro_config.get("funding_rates", {}) if isinstance(macro_config, dict) else {}
        oi_cfg = macro_config.get("open_interest", {}) if isinstance(macro_config, dict) else {}
        vix_cfg = macro_config.get("vix_proxy", {}) if isinstance(macro_config, dict) else {}
        ts_dex_enabled = (
            bool(ts_dex_config.get("enabled", False))
            if isinstance(ts_dex_config, dict)
            else bool(ts_dex_config)
        )
        return {
            "enabled": {
                "cex_trading": True,
                "dex_trading": dex_config.get("enabled", False),
                "uniswap": dex_config.get("uniswap", {}).get("enabled", False),
                "sushiswap": dex_config.get("sushiswap", {}).get("enabled", False),
                "pancakeswap": dex_config.get("pancakeswap", {}).get("enabled", False),
                "dydx": dex_config.get("dydx", {}).get("enabled", False),
                "ts_dex_layer": ts_dex_enabled,
                "funding_rates": funding_cfg.get("enabled", False) if isinstance(funding_cfg, dict) else False,
                "open_interest": oi_cfg.get("enabled", False) if isinstance(oi_cfg, dict) else False,
                "vix_proxy": vix_cfg.get("enabled", False) if isinstance(vix_cfg, dict) else False,
            },
            "chains": ["ethereum", "bsc", "arbitrum", "dydx_chain"] if dex_config.get("enabled") else [],
            "protocols": ["uniswap_v3", "sushiswap", "pancakeswap_v3", "dydx_perpetuals"],
        }

    @app.get("/signals/recent")
    async def recent_signals() -> dict[str, Any]:
        signals: list[dict[str, Any]] = []
        if order_manager is not None:
            recent_orders = sorted(order_manager.orders.values(), key=lambda o: o.created_at, reverse=True)[:25]
            for order in recent_orders:
                signals.append(
                    {
                        "symbol": order.symbol,
                        "direction": "long" if str(order.side.value).lower() == "buy" else "short",
                        "score": float(order.metadata.get("score", 0.7)),
                        "confidence": float(order.metadata.get("confidence", 0.75)),
                        "timestamp": int(order.created_at),
                        "technical_score": float(order.metadata.get("technical_score", 0.7)),
                        "ml_score": float(order.metadata.get("ml_score", 0.7)),
                        "sentiment_score": float(order.metadata.get("sentiment_score", 0.5)),
                        "source": "order_flow",
                    }
                )
        return {
            "signals": signals[:20],
            "total_today": len(signals),
            "win_rate": None,
        }

    @app.get("/api/signals/recent")
    async def api_recent_signals() -> dict[str, Any]:
        return await recent_signals()

    @app.get("/performance")
    async def performance_metrics() -> dict[str, Any]:
        pnl_total = 0.0
        pnl_pct = 0.0
        trades_total = 0
        trades_closed = 0
        trades_open = 0
        total_fees = 0.0
        if risk_manager is not None:
            pnl_total = float(sum(p.pnl for p in risk_manager.positions.values()))
            equity = float(max(1.0, risk_manager.equity))
            pnl_pct = float((pnl_total / equity) * 100.0)
        if order_manager is not None:
            stats = order_manager.get_stats()
            trades_total = int(stats.get("total_orders", 0))
            trades_open = int(stats.get("open_orders", 0))
            trades_closed = int(stats.get("filled_orders", 0))
            total_fees = float(stats.get("total_fees", 0.0))
        return {
            "pnl_total": pnl_total,
            "pnl_pct": pnl_pct,
            "win_rate": None,
            "trades_total": trades_total,
            "trades_closed": trades_closed,
            "trades_open": trades_open,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "daily_pnl": pnl_total,
            "total_fees": total_fees,
        }

    @app.get("/system/stats")
    async def system_stats() -> dict[str, Any]:
        db_connected = bool(getattr(db_handler, "available", False))
        cache_connected = bool(getattr(cache, "available", False))
        now = int(time.time())
        started_at = int(getattr(app.state, "started_at", now))
        feeds_connected = 0
        if data_manager is not None:
            feeds_connected = len(getattr(data_manager, "_aggregators", {}))

        websockets_active = 0
        if event_bus is not None:
            websockets_active = int(getattr(event_bus, "_queue", None).qsize()) if hasattr(getattr(event_bus, "_queue", None), "qsize") else 0
        return {
            "uptime_seconds": max(0, now - started_at),
            "feeds_connected": feeds_connected,
            "websockets_active": websockets_active,
            "db_connected": db_connected,
            "cache_connected": cache_connected,
            "timestamp": now,
        }

    # ── Auto-trading control ──────────────────────────────────────────────
    @app.post("/auto/toggle")
    async def auto_toggle(request: Request) -> dict[str, Any]:
        return await api_auto_toggle(request)

    @app.get("/auto/status")
    async def auto_status() -> dict[str, Any]:
        return await api_auto_status()

    @app.get("/signals/weights")
    async def signal_weights() -> dict[str, Any]:
        """Return the current 6-factor signal weights and configuration."""
        if signal_generator is None:
            return {"weights": {}, "min_score": 0.0, "min_factors": 0}
        return {
            "weights": {
                "technical": getattr(signal_generator, "_tech_weight", 0),
                "ml": getattr(signal_generator, "_ml_weight", 0),
                "sentiment": getattr(signal_generator, "_sentiment_weight", 0),
                "macro": getattr(signal_generator, "_macro_weight", 0),
                "news": getattr(signal_generator, "_news_weight", 0),
                "orderbook": getattr(signal_generator, "_orderbook_weight", 0),
            },
            "min_score": getattr(signal_generator, "_min_score", 0),
            "min_factors": getattr(signal_generator, "_min_factors", 0),
            "auto_trading_enabled": getattr(signal_generator, "auto_trading_enabled", False),
        }

    @app.get("/api/signals/weights")
    async def api_signal_weights() -> dict[str, Any]:
        return await signal_weights()

    # ── Kill switch ───────────────────────────────────────────────────────
    @app.post("/v1/kill")
    async def kill_switch_activate() -> dict[str, Any]:
        """Emergency: close all positions, cancel all orders, block new signals."""
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        closed = await risk_manager.activate_kill_switch()
        # Directly cancel exchange-side orders (no event-bus race window)
        cancelled_total = 0
        for exc in (executors or []):
            client = getattr(exc, "_client", None)
            if client and not config.paper_mode:
                try:
                    open_orders = await client.fetch_open_orders()
                    for o in open_orders:
                        for _attempt in range(3):
                            try:
                                rl = getattr(exc, "_rate_limiter", None)
                                if rl:
                                    await rl.acquire()
                                await client.cancel_order(o["id"], o.get("symbol"))
                                cancelled_total += 1
                                break
                            except Exception:
                                if _attempt == 2:
                                    logger.error(
                                        "Kill switch: FAILED to cancel order {} on {}",
                                        o.get("id"), getattr(exc, "exchange_id", "?"),
                                    )
                except Exception as exc_err:
                    logger.warning("Kill switch order cancel failed: {}", exc_err)
            # Also clean up protective order tracking
            placer = getattr(exc, "_order_placer", None)
            if placer:
                for sym in list(getattr(placer, "protective_orders", {}).keys()):
                    try:
                        await placer.cancel_all_for_symbol(sym)
                    except Exception:
                        pass
        # Notify other subscribers (best-effort, non-critical path)
        if event_bus is not None:
            await event_bus.publish("KILL_SWITCH", {"source": "api"})
        return {
            "success": True,
            "positions_closed": len(closed),
            "orders_cancelled": cancelled_total,
            "kill_switch_active": True,
        }

    @app.post("/v1/kill/deactivate")
    async def kill_switch_deactivate(request: Request) -> dict[str, Any]:
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not config.paper_mode:
            if str(body.get("confirmation", "")).strip() != "DEACTIVATE KILL SWITCH":
                return {"success": False, "error": "typed confirmation required: DEACTIVATE KILL SWITCH"}
            if not bool(getattr(db_handler, "available", False)):
                return {"success": False, "error": "cannot deactivate kill switch: audit DB unavailable"}
        risk_manager.deactivate_kill_switch()
        return {"success": True, "kill_switch_active": False}

    @app.get("/v1/risk/snapshot")
    async def risk_snapshot() -> dict[str, Any]:
        """Full risk snapshot including drawdown, VaR, kill switch status."""
        if risk_manager is None:
            return {"error": "risk_manager not available"}
        return risk_manager.get_risk_snapshot()

    # ══════════════════════════════════════════════════════════════════════
    #  /api/* endpoints — UI fetches everything under this prefix
    # ══════════════════════════════════════════════════════════════════════

    # ── /api/status — portfolio status header bar ─────────────────────────
    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        equity = 0.0
        unrealized_pnl = 0.0
        drawdown_pct = 0.0
        portfolio_heat = 0.0
        daily_pnl = 0.0
        open_positions = 0
        positions_list: list[dict] = []
        win_rate = 0.0
        total_trades = 0

        if risk_manager is not None:
            equity = float(risk_manager.equity)
            positions = risk_manager.positions
            unrealized_pnl = float(sum(p.pnl for p in positions.values()))
            open_positions = len(positions)
            snap = risk_manager.get_risk_snapshot()
            drawdown_pct = float(snap.get("drawdown_pct", 0))
            portfolio_heat = float(snap.get("portfolio_heat", 0))
            positions_list = [
                {
                    "symbol": p.symbol,
                    "side": p.direction,
                    "size": p.size,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": p.pnl,
                    "liquidation": getattr(p, 'liquidation_price', 0.0),
                    "funding": getattr(p, 'funding_payment', 0.0),
                    "rpnl": getattr(p, 'realized_pnl', 0.0),
                }
                for p in positions.values()
            ]

        if order_manager is not None:
            stats = order_manager.get_stats()
            total_trades = int(stats.get("total_orders", 0))
            filled = int(stats.get("filled_orders", 0))
            win_rate = 0.0
            if filled > 0:
                total_pnl = unrealized_pnl
                win_rate = 100.0 if total_pnl >= 0 else 0.0

        daily_pnl = unrealized_pnl

        return {
            "equity": equity,
            "unrealized_pnl": unrealized_pnl,
            "drawdown_pct": drawdown_pct,
            "portfolio_heat": portfolio_heat,
            "daily_pnl": daily_pnl,
            "open_positions": open_positions,
            "positions": positions_list,
            "win_rate": win_rate,
            "total_trades": total_trades,
        }

    # ── /api/charts — REQ-IND-018..020: candles + aligned indicator series ─
    # Returns the chart-registry view per (symbol, timeframe): bar series,
    # indicator overlays aligned to the same index, and metadata. Lookback
    # capped to 500 bars to keep the payload reasonable.
    _CHART_OVERLAY_KEYS = (
        "ema_9", "ema_21", "ema_50", "ema_55", "ema_200",
        "sma_20", "sma_50",
        "vwma_20", "hma_20", "alma_20",
        "vwap",
        "bb_upper", "bb_mid", "bb_lower",
        "kc_upper", "kc_mid", "kc_lower",
        "donchian_upper", "donchian_lower", "donchian_mid",
        "supertrend",
        "psar",
        "tenkan_sen", "kijun_sen", "senkou_a", "senkou_b",
    )
    _CHART_OSCILLATOR_KEYS = (
        "rsi_14", "rsi_21",
        "stoch_rsi_k", "stoch_rsi_d",
        "macd", "macd_signal", "macd_hist",
        "vw_macd", "vw_macd_signal", "vw_macd_hist",
        "stoch_k", "stoch_d",
        "atr_14", "atr_pct",
        "adx", "plus_di", "minus_di",
        "mfi_14", "cci_20", "williams_r",
        "obv",
        "vpt", "nvi", "pvi",
        "vortex_diff", "awesome_osc", "accelerator_osc",
        "rvi", "rvi_signal",
        "roc_10", "roc_20",
    )

    @app.get("/api/charts/{symbol:path}/{timeframe}")
    async def api_charts(symbol: str, timeframe: str, lookback: int = 200) -> dict[str, Any]:
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        lookback = max(20, min(int(lookback), 500))

        df = None
        sym_used = symbol
        if data_manager is not None:
            clean = symbol.replace("/", "").replace(":USDT", "").upper()
            base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
            for sym_try in (f"{base}/USDT:USDT", symbol, f"{base}/USDT", clean):
                df = data_manager.get_dataframe("binance", sym_try, timeframe)
                if df is not None and len(df) > 0:
                    sym_used = sym_try
                    break
            else:
                df = None

        if df is None or len(df) == 0:
            return {
                "available": False,
                "symbol": symbol, "timeframe": timeframe,
                "reason": "no live dataframe — try /api/candles for historical fallback",
            }

        df = df.tail(lookback)

        def _series(col: str) -> list[float | None]:
            if col not in df.columns:
                return []
            out: list[float | None] = []
            for v in df[col].tolist():
                try:
                    fv = float(v)
                    out.append(fv if fv == fv else None)  # NaN → None
                except Exception:
                    out.append(None)
            return out

        timestamps = [
            (idx.isoformat() if hasattr(idx, "isoformat") else str(idx))
            for idx in df.index
        ]
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0),
                "close": float(row.get("close", 0) or 0),
                "volume": float(row.get("volume", 0) or 0),
            })
        overlays = {k: _series(k) for k in _CHART_OVERLAY_KEYS if k in df.columns}
        oscillators = {k: _series(k) for k in _CHART_OSCILLATOR_KEYS if k in df.columns}
        # Sparse pivot markers — keep NaN preserved as None (caller filters).
        pivots = {
            "swing_high": _series("swing_high"),
            "swing_low": _series("swing_low"),
        }
        warmup_count = int(df["_warmup"].sum()) if "_warmup" in df.columns else 0

        return {
            "available": True,
            "symbol": sym_used,
            "timeframe": timeframe,
            "bars": len(df),
            "warmup_bars": warmup_count,
            "timestamps": timestamps,
            "candles": candles,
            "overlays": overlays,
            "oscillators": oscillators,
            "pivots": pivots,
            "metadata": {
                "overlay_keys": list(overlays.keys()),
                "oscillator_keys": list(oscillators.keys()),
            },
        }

    # ── /api/candles — OHLCV data for chart ───────────────────────────────
    _candle_cache: dict[str, Any] = {"key": "", "ts": 0.0, "candles": []}

    @app.get("/api/candles")
    async def api_candles(
        symbol: str = Query("BTC/USDT"),
        timeframe: str = Query("1m"),
    ) -> dict[str, Any]:
        import time as _time
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})

        # Build all symbol variants to try
        clean_sym = symbol.replace("/", "").replace(":USDT", "").upper()
        base = clean_sym.replace("USDT", "") if clean_sym.endswith("USDT") else clean_sym
        sym_variants = [
            f"{base}/USDT:USDT",   # normalizer output format
            symbol,                 # as given
            f"{base}/USDT",         # ccxt-style
            clean_sym,              # raw "BTCUSDT"
        ]

        # Try DataManager first (live aggregated data)
        df = None
        if data_manager is not None:
            for sym_try in sym_variants:
                df = data_manager.get_dataframe("binance", sym_try, timeframe)
                if df is not None and len(df) > 0:
                    break
            else:
                df = None

        if df is not None and len(df) > 0:
            rows = []
            for idx, row in df.iterrows():
                rows.append({
                    "time": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                })
            return {"candles": rows}

        # Fallback: fetch historical klines from Binance
        cache_key = f"{clean_sym}:{timeframe}"
        now = _time.time()
        if _candle_cache["key"] == cache_key and (now - _candle_cache["ts"]) < 30:
            return {"candles": _candle_cache["candles"]}

        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        binance_tf = tf_map.get(timeframe, "1m")
        try:
            import aiohttp
            pair = f"{base}USDT"
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval={binance_tf}&limit=1000"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
                async with sess.get(url) as resp:
                    if resp.status == 200:
                        klines = await resp.json(content_type=None)
                        rows = []
                        for k in klines:
                            ts = int(k[0]) / 1000  # ms → s
                            from datetime import datetime, timezone
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            rows.append({
                                "time": dt.isoformat(),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                            })
                        _candle_cache["key"] = cache_key
                        _candle_cache["ts"] = now
                        _candle_cache["candles"] = rows
                        return {"candles": rows}
        except Exception as exc:
            logger.debug("Binance klines fallback error: {}", exc)

        return {"candles": []}

    # ── /api/market — watchlist / market data ─────────────────────────────
    @app.get("/api/market")
    async def api_market(per_page: int = Query(100, ge=1, le=250)) -> dict[str, Any]:
        import time as _time
        import aiohttp

        now = _time.time()
        requested = min(per_page, 250)
        cached_coins = _market_cache.get("coins") or []

        # Return cache only if it is both fresh and large enough for this request.
        if cached_coins and (now - _market_cache["ts"]) < 60 and len(cached_coins) >= requested:
            return {"coins": cached_coins[:requested]}

        coins: list[dict[str, Any]] = []
        seen_symbols: set[str] = set()

        def _append_coin(coin: dict[str, Any]) -> None:
            symbol = str(coin.get("symbol", "")).upper()
            if not symbol or symbol in seen_symbols:
                return
            coin["symbol"] = symbol
            seen_symbols.add(symbol)
            coins.append(coin)

        # Primary: CoinGecko
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as sess:
                async with sess.get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": requested,
                        "page": 1,
                        "sparkline": "false",
                    },
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        for c in data:
                            _append_coin({
                                "symbol": str(c.get("symbol", "")).upper(),
                                "name": c.get("name", ""),
                                "price": float(c.get("current_price") or 0),
                                "change_24h": float(c.get("price_change_percentage_24h") or 0),
                                "volume_24h": float(c.get("total_volume") or 0),
                                "high_24h": float(c.get("high_24h") or 0),
                                "low_24h": float(c.get("low_24h") or 0),
                                "market_cap": float(c.get("market_cap") or 0),
                            })
        except Exception as exc:
            logger.debug("CoinGecko market fetch error: {}", exc)

        # Fallback or backfill: Binance 24h ticker for top futures symbols
        if len(coins) < requested:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as sess:
                    async with sess.get(
                        "https://fapi.binance.com/fapi/v1/ticker/24hr"
                    ) as resp:
                        if resp.status == 200:
                            tickers = await resp.json(content_type=None)
                            sorted_tickers = sorted(
                                [t for t in tickers if isinstance(t, dict) and str(t.get("symbol", "")).endswith("USDT")],
                                key=lambda t: float(t.get("quoteVolume") or 0),
                                reverse=True,
                            )
                            for t in sorted_tickers:
                                sym = str(t.get("symbol", ""))
                                _append_coin({
                                    "symbol": sym.replace("USDT", ""),
                                    "name": sym.replace("USDT", ""),
                                    "price": float(t.get("lastPrice") or 0),
                                    "change_24h": float(t.get("priceChangePercent") or 0),
                                    "volume_24h": float(t.get("quoteVolume") or 0),
                                    "high_24h": float(t.get("highPrice") or 0),
                                    "low_24h": float(t.get("lowPrice") or 0),
                                    "market_cap": 0,
                                })
                                if len(coins) >= requested:
                                    break
            except Exception as exc:
                logger.debug("Binance ticker fallback error: {}", exc)

        if coins:
            _market_cache["coins"] = coins
            _market_cache["ts"] = now
        elif cached_coins:
            # Return stale cache rather than empty
            return {"coins": cached_coins[:requested]}

        return {"coins": coins[:requested]}

    # ── /api/exchange/balance — fetch REAL account balance (demo/live) ─────
    @app.get("/api/exchange/balance")
    async def api_exchange_balance() -> dict[str, Any]:
        """Fetch actual balance from the first connected exchange (testnet/mainnet).

        When trading-mode is DEMO or LIVE, this surfaces the real ccxt balance
        for the first enabled venue that has a live client attached. PAPER mode
        returns an empty response so the UI can fall back to simulated equity.
        """
        try:
            from interface.routes.config import get_active_trading_mode
            active_mode = get_active_trading_mode()
        except Exception:
            active_mode = "paper" if config.paper_mode else "live"

        if active_mode == "paper":
            return {"success": False, "mode": "paper", "error": "paper mode — no exchange balance"}

        for exc in (executors or []):
            client = getattr(exc, "_client", None)
            ex_id = getattr(exc, "exchange_id", "unknown")
            if client is None:
                try:
                    await exc._init_client()
                    client = getattr(exc, "_client", None)
                except Exception as init_err:
                    logger.warning("balance: {} client init failed: {}", ex_id, init_err)
                    continue
            if client is None:
                continue
            try:
                rl = getattr(exc, "_rate_limiter", None)
                if rl:
                    await rl.acquire()
                bal = await asyncio.wait_for(client.fetch_balance(), timeout=20.0)
            except asyncio.TimeoutError:
                return {"success": False, "exchange": ex_id, "error": "exchange timeout (8s)"}
            except Exception as fetch_err:
                logger.warning("fetch_balance failed for {}: {}", ex_id, fetch_err)
                return {"success": False, "exchange": ex_id, "error": f"{type(fetch_err).__name__}: {str(fetch_err)[:200]}"}

            totals = (bal or {}).get("total") or {}
            frees = (bal or {}).get("free") or {}
            used = (bal or {}).get("used") or {}
            rows: list[dict[str, Any]] = []
            for ccy, amt in sorted(totals.items(), key=lambda kv: -float(kv[1] or 0)):
                try:
                    amtf = float(amt)
                except Exception:
                    amtf = 0.0
                if amtf <= 0:
                    continue
                rows.append({
                    "asset": ccy,
                    "total": amtf,
                    "free": float(frees.get(ccy, 0) or 0),
                    "used": float(used.get(ccy, 0) or 0),
                })
            equity_usd = float(totals.get("USDT", totals.get("USD", 0)) or 0)
            venue_cfg = config.get_value("exchanges", ex_id) or {}
            return {
                "success": True,
                "mode": active_mode,
                "exchange": ex_id,
                "testnet": bool(venue_cfg.get("testnet", False)),
                "equity_usd": equity_usd,
                "balances": rows,
                "balance_count": len(rows),
            }
        return {"success": False, "mode": active_mode, "error": "no connected exchange client"}

    # ── /api/exchange/positions — fetch REAL exchange positions ────────────
    @app.get("/api/exchange/positions")
    async def api_exchange_positions() -> dict[str, Any]:
        """Fetch actual open positions from connected exchanges."""
        cached = _cache_get("positions")
        if cached is not None:
            return cached
        all_positions: list[dict] = []
        for exc in (executors or []):
            client = getattr(exc, "_client", None)
            ex_id = getattr(exc, "exchange_id", "unknown")
            if client is None:
                continue
            try:
                rl = getattr(exc, "_rate_limiter", None)
                if rl:
                    await rl.acquire()
                raw_positions = await client.fetch_positions()
                for p in raw_positions:
                    contracts = float(p.get("contracts", 0) or 0)
                    if contracts == 0:
                        continue
                    all_positions.append({
                        "exchange": ex_id,
                        "symbol": p.get("symbol", ""),
                        "side": p.get("side", ""),
                        "size": contracts,
                        "entry": float(p.get("entryPrice", 0) or 0),
                        "current": float(p.get("markPrice", 0) or 0),
                        "pnl": float(p.get("unrealizedPnl", 0) or 0),
                        "liquidation": float(p.get("liquidationPrice", 0) or 0),
                        "leverage": float(p.get("leverage", 1) or 1),
                        "margin": float(p.get("initialMargin", 0) or 0),
                        "notional": float(p.get("notional", 0) or 0),
                    })
            except Exception as exc_err:
                logger.warning("fetch_positions failed for {}: {}", ex_id, exc_err)
        result = {"positions": all_positions, "total": len(all_positions)}
        _cache_set("positions", result)
        return result

    # ── /api/exchange/orders — fetch REAL open orders from exchange ────────
    @app.get("/api/exchange/orders")
    async def api_exchange_orders() -> dict[str, Any]:
        """Fetch actual open orders from connected exchanges."""
        cached = _cache_get("orders")
        if cached is not None:
            return cached
        all_orders: list[dict] = []
        for exc in (executors or []):
            client = getattr(exc, "_client", None)
            ex_id = getattr(exc, "exchange_id", "unknown")
            if client is None:
                continue
            try:
                rl = getattr(exc, "_rate_limiter", None)
                if rl:
                    await rl.acquire()
                # Fetch per-symbol in parallel to reduce latency
                symbols = getattr(exc, "_symbols", None) or []
                if symbols:
                    async def _fetch_sym(sym: str):
                        try:
                            return await client.fetch_open_orders(sym)
                        except Exception:
                            return []
                    results = await asyncio.gather(*[_fetch_sym(s) for s in symbols])
                    raw_orders = [o for batch in results for o in batch]
                else:
                    raw_orders = await client.fetch_open_orders()
                for o in raw_orders:
                    all_orders.append({
                        "exchange": ex_id,
                        "id": o.get("id", ""),
                        "symbol": o.get("symbol", ""),
                        "type": o.get("type", ""),
                        "side": o.get("side", ""),
                        "amount": float(o.get("amount", 0) or 0),
                        "price": float(o.get("price", 0) or 0),
                        "filled": float(o.get("filled", 0) or 0),
                        "status": o.get("status", ""),
                        "timestamp": o.get("timestamp", 0),
                    })
            except Exception as exc_err:
                logger.warning("fetch_open_orders failed for {}: {}", ex_id, exc_err)
        result = {"orders": all_orders, "total": len(all_orders)}
        _cache_set("orders", result)
        return result

    # ── /api/latency — live latency statistics ────────────────────────────
    @app.get("/api/latency")
    async def api_latency() -> dict[str, Any]:
        """REQ-MON-002: latency stats with p50/p95/p99 percentiles."""
        stats: dict[str, Any] = {
            "feed_lag": {},
            "order_latency": {},
            "decision_latency": {},
            "pipeline_latency": {},
        }
        if metrics and hasattr(metrics, "get_latency_stats"):
            stats = metrics.get_latency_stats()
        return stats

    @app.get("/api/latency/percentiles")
    async def api_latency_percentiles() -> dict[str, Any]:
        """Compact p50/p95/p99 payload for the dashboard chip strip."""
        if metrics and hasattr(metrics, "get_latency_percentiles"):
            return metrics.get_latency_percentiles()
        return {"feed_lag": {}, "order_latency": {}, "decision_latency": {}, "pipeline_latency": {}}

    # ── /api/trade-history — closed trades + realized PnL ─────────────────
    @app.get("/api/trade-history")
    async def api_trade_history(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        """Return closed trade history and realized PnL summary."""
        trades: list[dict] = []
        # 1. In-memory closed trades from risk_manager (current session)
        if risk_manager and hasattr(risk_manager, "_closed_trades"):
            trades = list(risk_manager._closed_trades)
        # 2. Supplement from SQLite (previous sessions)
        if sqlite_store and sqlite_store.available:
            try:
                db_trades = sqlite_store.get_trade_history(limit=limit)
                # Merge: deduplicate by open_time+symbol
                seen = {(t["symbol"], t["open_time"]) for t in trades}
                for dt in db_trades:
                    key = (dt.get("symbol", ""), dt.get("open_time_ns", 0) // 10**9)
                    if key not in seen:
                        trades.append({
                            "exchange": dt.get("exchange", ""),
                            "symbol": dt.get("symbol", ""),
                            "direction": dt.get("direction", ""),
                            "entry_price": dt.get("entry_price", 0),
                            "exit_price": dt.get("exit_price", 0),
                            "size": dt.get("size", 0),
                            "pnl": dt.get("pnl", 0),
                            "pnl_pct": round((dt.get("pnl_pct", 0) or 0) * 100, 4),
                            "hold_seconds": 0,
                            "open_time": dt.get("open_time_ns", 0) // 10**9,
                            "close_time": dt.get("close_time_ns", 0) // 10**9,
                        })
            except Exception as e:
                logger.debug("SQLite trade history error: {}", e)
        # Sort by close_time desc
        trades.sort(key=lambda t: t.get("close_time", 0), reverse=True)
        trades = trades[:limit]
        # Summary
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
        return {
            "trades": trades,
            "total": len(trades),
            "total_pnl": round(total_pnl, 4),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        }

    # ── /api/exchange/currencies — all currencies from connected exchanges ─
    @app.get("/api/exchange/currencies")
    async def api_exchange_currencies(
        exchange: str | None = Query(None, description="Filter by exchange id"),
        quote: str | None = Query(None, description="Filter by quote currency (e.g. USDT)"),
        market_type: str | None = Query(None, alias="type", description="Filter by market type (spot, swap, future)"),
        active_only: bool = Query(True, description="Only return active markets"),
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict[str, Any]:
        if not executors:
            return {"exchanges": [], "total": 0, "error": "No executors configured"}
        result: dict[str, list[dict[str, Any]]] = {}
        for ex in executors:
            client = getattr(ex, "_client", None)
            if client is None:
                continue
            ex_id = getattr(ex, "exchange_id", str(type(ex).__name__))
            if exchange and ex_id != exchange:
                continue
            markets = getattr(client, "markets", None)
            if not markets:
                continue
            symbols: list[dict[str, Any]] = []
            for sym, info in markets.items():
                if active_only and not info.get("active", True):
                    continue
                if quote and info.get("quote", "").upper() != quote.upper():
                    continue
                mtype = info.get("type", "")
                if market_type and mtype != market_type:
                    continue
                symbols.append({
                    "symbol": sym,
                    "base": info.get("base", ""),
                    "quote": info.get("quote", ""),
                    "type": mtype,
                    "active": info.get("active", True),
                    "contractSize": info.get("contractSize"),
                    "precision": info.get("precision", {}),
                    "limits": info.get("limits", {}),
                })
                if len(symbols) >= limit:
                    break
            result[ex_id] = symbols
        total = sum(len(v) for v in result.values())
        return {"exchanges": result, "total": total}

    # ── /api/feargreed — fear & greed index ───────────────────────────────
    @app.get("/api/feargreed")
    async def api_feargreed() -> dict[str, Any]:
        if sentiment_manager is not None:
            fg = getattr(sentiment_manager, "_fear_greed", None)
            if fg is not None:
                latest = fg.get_latest()
                if latest is not None:
                    value = int(max(0, min(100, (latest.score + 1) * 50)))
                    return {
                        "value": value,
                        "classification": latest.label.capitalize(),
                    }
        # Fallback: direct fetch
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
                async with sess.get("https://api.alternative.me/fng/?limit=1&format=json") as resp:
                    data = await resp.json(content_type=None)
                    item = data.get("data", [{}])[0]
                    return {
                        "value": int(item.get("value", 50)),
                        "classification": item.get("value_classification", "Neutral"),
                    }
        except Exception:
            return {"value": 50, "classification": "Neutral"}

    # ── /api/orderbook — order book depth ─────────────────────────────────
    @app.get("/api/orderbook")
    async def api_orderbook(
        symbol: str = Query("BTC/USDT"),
        depth: int = Query(8),
    ) -> dict[str, Any]:
        try:
            symbol = _validate_symbol(symbol)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        depth = max(1, min(depth, 50))
        # Check event-bus cache first
        for key, cached in _orderbook_cache.items():
            if symbol.replace("/", "") in key.replace("/", "").replace(":", ""):
                bids_raw = cached.get("bids", [])
                asks_raw = cached.get("asks", [])
                bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in bids_raw[:depth]]
                asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in asks_raw[:depth]]
                mid = (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else 0
                spread = asks[0]["price"] - bids[0]["price"] if bids and asks else 0
                return {"bids": bids, "asks": asks, "spread": round(spread, 2), "mid_price": round(mid, 2)}

        # Fallback: fetch directly from Binance
        try:
            import aiohttp
            binance_sym = symbol.replace("/", "").replace("-", "").upper()
            # Always use mainnet for read-only market data (public endpoint, no auth needed)
            base_url = "https://fapi.binance.com/fapi/v1/depth"
            # Binance only accepts specific limit values
            valid_limits = [5, 10, 20, 50, 100, 500, 1000]
            binance_limit = min((v for v in valid_limits if v >= depth), default=20)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
                async with sess.get(base_url, params={"symbol": binance_sym, "limit": binance_limit}) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in data.get("bids", [])[:depth]]
                        asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in data.get("asks", [])[:depth]]
                        mid = (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else 0
                        spread = asks[0]["price"] - bids[0]["price"] if bids and asks else 0
                        return {"bids": bids, "asks": asks, "spread": round(spread, 2), "mid_price": round(mid, 2)}
        except Exception as exc:
            logger.debug("Orderbook direct fetch error: {}", exc)
        return {"bids": [], "asks": [], "spread": 0, "mid_price": 0}

    # ── /api/indicators/{symbol} — technical indicators ───────────────────
    @app.get("/api/indicators/{sym}")
    async def api_indicators(sym: str) -> dict[str, Any]:
        if data_manager is None:
            return _default_indicators()
        # Build all symbol format variants
        clean = sym.replace("/", "").replace(":USDT", "").upper()
        base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
        sym_variants = [
            f"{base}/USDT:USDT",   # normalizer output format
            f"{base}/USDT",         # ccxt-style
            f"{base}USDT",          # raw
            sym,                    # as given
        ]
        for sym_try in sym_variants:
            for tf_try in ["15m", "5m", "1m", "1h"]:
                df = data_manager.get_dataframe("binance", sym_try, tf_try)
                if df is not None and len(df) >= 5:
                    row = df.iloc[-1]
                    return {
                        "rsi": float(row.get("rsi", 50)),
                        "macd": float(row.get("macd", 0)),
                        "stoch_k": float(row.get("stoch_k", 50)),
                        "adx": float(row.get("adx", 20)),
                        "atr": float(row.get("atr", 0)),
                        "bb_width": float(row.get("bb_width", 0)),
                        "ema9": float(row.get("ema_9", 0)),
                        "ema21": float(row.get("ema_21", 0)),
                        "sma50": float(row.get("sma_50", 0)),
                        "ema_cross": float(row.get("ema_9", 0)) - float(row.get("ema_21", 0)),
                        "volume_ratio": float(row.get("volume", 0)) / max(1.0, float(df["volume"].rolling(20).mean().iloc[-1])) if "volume" in df.columns else 1.0,
                    }
        return _default_indicators()

    def _default_indicators() -> dict[str, Any]:
        return {
            "rsi": 50, "macd": 0, "stoch_k": 50, "adx": 20, "atr": 0,
            "bb_width": 0, "ema9": 0, "ema21": 0, "sma50": 0, "ema_cross": 0,
            "volume_ratio": 1.0,
        }

    # ── /api/dex/pools — DEX liquidity pools (cached with backoff) ──────
    _dex_cache: dict[str, Any] = {"data": {"pools": []}, "ts": 0.0, "fail_until": 0.0}

    @app.get("/api/dex/pools")
    async def api_dex_pools() -> dict[str, Any]:
        now = time.time()
        # Return cached data if fresh (60s) or in backoff window after failure
        if now - _dex_cache["ts"] < 60 or now < _dex_cache["fail_until"]:
            return _dex_cache["data"]
        try:
            import aiohttp
            query = '{ pools(first: 5, orderBy: totalValueLockedUSD, orderDirection: desc, where: { feeTier_in: [500, 3000] }) { token0 { symbol } token1 { symbol } totalValueLockedUSD } }'
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as sess:
                async with sess.post(
                    "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        pools = data.get("data", {}).get("pools", [])
                        result = {
                            "pools": [
                                {
                                    "pair": f"{p['token0']['symbol']}/{p['token1']['symbol']}",
                                    "tvl": float(p.get("totalValueLockedUSD", 0)),
                                }
                                for p in pools
                            ]
                        }
                        _dex_cache["data"] = result
                        _dex_cache["ts"] = now
                        _dex_cache["fail_until"] = 0.0
                        return result
        except Exception as exc:
            logger.debug("DEX pools fetch error: {}", exc)
        # On failure: backoff 120s before retrying
        _dex_cache["fail_until"] = now + 120
        return _dex_cache["data"]

    # ── /api/variational — Variational DEX status & positions ─────────────
    @app.get("/api/variational/status")
    async def api_variational_status() -> dict[str, Any]:
        var_cfg = config.get_value("variational") or {}
        enabled = var_cfg.get("enabled", False)
        if not enabled:
            return {"enabled": False, "connected": False, "testnet": True, "positions": [], "portfolio": {}}
        # Find Variational executor from executors list
        var_exec = None
        for ex in (executors or []):
            if hasattr(ex, "get_portfolio_summary"):
                var_exec = ex
                break
        if var_exec is None:
            return {"enabled": True, "connected": False, "testnet": var_cfg.get("testnet", True), "positions": [], "portfolio": {}}
        try:
            positions = await var_exec.get_positions()
            portfolio = await var_exec.get_portfolio_summary()
            return {
                "enabled": True,
                "connected": var_exec._client is not None,
                "testnet": var_cfg.get("testnet", True),
                "max_trade_usd": var_cfg.get("max_trade_usd", 50),
                "symbols": var_cfg.get("symbols", []),
                "positions": positions,
                "portfolio": portfolio,
            }
        except Exception as exc:
            logger.debug("Variational status error: {}", exc)
            return {"enabled": True, "connected": False, "error": sanitize_exception(exc), "positions": [], "portfolio": {}}

    # ── /api/news — live news feed ────────────────────────────────────────
    @app.get("/api/news")
    async def api_news() -> dict[str, Any]:
        items = list(_news_buffer)
        if items:
            return {"provider": "cryptocompare", "items": items[-20:][::-1]}
        # Fallback: fetch from CoinGecko trending + search/trending
        try:
            import aiohttp
            fetched_items: list[dict[str, Any]] = []
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as sess:
                # CoinGecko search/trending — returns trending coins as pseudo-news
                async with sess.get("https://api.coingecko.com/api/v3/search/trending") as resp:
                    if resp.status == 200:
                        from data_ingestion.news_feed import classify_sentiment
                        data = await resp.json(content_type=None)
                        coins = data.get("coins", [])
                        for c in coins[:10]:
                            item_data = c.get("item", {})
                            name = item_data.get("name", "")
                            sym = item_data.get("symbol", "")
                            score = item_data.get("score", 0)
                            price_chg = float(item_data.get("data", {}).get("price_change_percentage_24h", {}).get("usd", 0) or 0)
                            sentiment = "bullish" if price_chg > 2 else ("bearish" if price_chg < -2 else "neutral")
                            title = f"{name} ({sym}) trending — rank #{score + 1}, 24h: {price_chg:+.1f}%"
                            fetched_items.append({
                                "ts": int(time.time() * 1000),
                                "title": title,
                                "sentiment": sentiment,
                            })
                if fetched_items:
                    return {"provider": "coingecko_trending", "items": fetched_items}

                # Second fallback: CryptoCompare (may require API key)
                async with sess.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest") as resp:
                    if resp.status == 200:
                        from data_ingestion.news_feed import classify_sentiment
                        data = await resp.json(content_type=None)
                        articles = data.get("Data", data.get("data", []))
                        if isinstance(articles, list):
                            for a in articles[:10]:
                                title = a.get("title", "")
                                score = classify_sentiment(title + " " + a.get("body", "")[:200])
                                sentiment = "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")
                                fetched_items.append({
                                    "ts": int(a.get("published_on", time.time()) * 1000),
                                    "title": title[:200],
                                    "sentiment": sentiment,
                                })
                            if fetched_items:
                                return {"provider": "cryptocompare", "items": fetched_items}
        except Exception as exc:
            logger.debug("News direct fetch error: {}", exc)
        return {"provider": "unavailable", "items": []}

    # ── /api/geopolitical — Layer 10 RSS scorer + strategy details ────────
    @app.get("/api/geopolitical")
    async def api_geopolitical() -> dict[str, Any]:
        if geopolitical_feed is None:
            return {"available": False, "reason": "geopolitical feed not configured"}
        try:
            stats = geopolitical_feed.stats()
        except Exception as exc:
            return {"available": False, "error": sanitize_exception(exc)}

        # Strategy module exposes static config used by the dashboard's Geo tab
        # so the user can verify which keywords/feeds/hours the bot is actually
        # using, alongside live counters.
        from urllib.parse import urlparse as _urlparse
        try:
            from strategies.geo_political_strategy import (
                MARKET_CONFIGS as _MC,
                MIN_RELEVANCE as _MIN_REL,
                WINDOW_HOURS as _WIN_H,
                DIRECTION_TOKENS as _DT,
            )
        except Exception:
            _MC, _MIN_REL, _WIN_H, _DT = {}, 30, 6.0, {}

        feeds_meta = []
        for url in getattr(geopolitical_feed, "_feeds", []) or []:
            try:
                host = _urlparse(url).netloc
            except Exception:
                host = url
            feeds_meta.append({"url": url, "host": host})

        market_configs: dict[str, Any] = {}
        for sym, cfg in _MC.items():
            kws = cfg.get("keywords", {}) or {}
            top_kw = sorted(kws.items(), key=lambda kv: kv[1], reverse=True)[:10]
            dt = (_DT.get(sym) or {}) if _DT else {}
            bullish = sorted([k for k, v in dt.items() if v > 0])
            bearish = sorted([k for k, v in dt.items() if v < 0])
            market_configs[sym] = {
                "name": cfg.get("name") or sym,
                "trading_hours": cfg.get("trading_hours"),
                "keyword_count": len(kws),
                "top_keywords": [{"token": k, "weight": w} for k, w in top_kw],
                "bullish_tokens": bullish,
                "bearish_tokens": bearish,
            }

        recent_events: list[dict] = []
        try:
            scorer = getattr(geopolitical_feed, "_scorer", None)
            if scorer is not None and hasattr(scorer, "recent_events"):
                recent_events = scorer.recent_events(limit=20)
        except Exception as exc:
            logger.debug("recent_events lookup failed: {}", exc)

        return {
            "available": True,
            "polls": stats.get("polls", 0),
            "articles": stats.get("articles", 0),
            "events": stats.get("events", 0),
            "errors": {
                "dns": stats.get("errors_dns", 0),
                "http": stats.get("errors_http", 0),
                "parse": stats.get("errors_parse", 0),
                "other": stats.get("errors_other", 0),
            },
            "not_modified": stats.get("not_modified", 0),
            "symbols": stats.get("scorer", {}),
            "feeds": feeds_meta,
            "market_configs": market_configs,
            "constants": {
                "min_relevance": int(_MIN_REL),
                "window_hours": float(_WIN_H),
                "fetch_interval_seconds": float(getattr(geopolitical_feed, "_interval", 600.0)),
                "max_articles_per_feed": int(getattr(geopolitical_feed, "_max_articles_per_feed", 30)),
            },
            "recent_events": recent_events,
            # Pipeline stages from the strategy spec — surfaced so the UI can
            # render the canonical phase diagram without hardcoding it.
            "pipeline_phases": [
                {"id": "1A", "name": "RSS Fetch", "kind": "free", "summary": "Pull headlines from configured RSS feeds; dedupe by signal_uid"},
                {"id": "1B", "name": "Keyword Relevance", "kind": "free", "summary": f"Weight-sum match against per-market keywords; drop < {int(_MIN_REL)}"},
                {"id": "1C", "name": "LLM Sentiment", "kind": "paid", "summary": "Direction LONG/SHORT/SKIP + confidence; drop if conf < 60"},
                {"id": "GATE", "name": "Quality Gates", "kind": "free", "summary": "Hours filter + duplicate suppression + max-concurrent + 15m momentum"},
                {"id": "2", "name": "Price Confirmation", "kind": "paid", "summary": "Live ticker + OHLCV → CONFIRM / CONTRADICT / WEAK"},
                {"id": "3", "name": "Trade Planner", "kind": "paid", "summary": "Per-event SL/TP/timeout plan, clamped to safe rails"},
                {"id": "EXEC", "name": "Execute", "kind": "free", "summary": "Open paper position, append to ledger, send Telegram alert"},
                {"id": "MGMT", "name": "Manage", "kind": "free", "summary": "BE @ 50%, trail @ 75%, soft flat-timeout, hard 4h ceiling"},
            ],
        }

    # ── /api/system/data-sources — module source map ──────────────────────
    @app.get("/api/system/data-sources")
    async def api_data_sources() -> dict[str, Any]:
        return {
            "sentiment": {"source": "alternative.me/fng" if sentiment_manager else "unavailable"},
            "news": {"source": "cryptocompare" if news_feed else "unavailable"},
            "backtest": {"source": "fast_backtester"},
            "logs": {"source": "loguru_ringbuffer"},
            "auto": {"source": "signal_generator" if signal_generator else "unavailable"},
            "signals": {"source": "6_factor_composite" if signal_generator else "unavailable"},
        }

    # ── /api/reconciliation/status — startup + periodic reconciliation ────
    @app.get("/api/reconciliation/status")
    async def api_reconciliation_status() -> dict[str, Any]:
        # Periodic snapshot — populated by PeriodicReconciler.run() loop.
        periodic_payload: dict[str, Any] = {"available": False}
        if periodic_reconciler is not None:
            last = getattr(periodic_reconciler, "last_result", None) or {}
            periodic_payload = {
                "available": True,
                "ran": bool(last.get("ran", False)),
                "ts": float(last.get("ts", 0.0) or 0.0),
                "interval_seconds": float(last.get("interval_seconds", 300.0) or 300.0),
                "exchange_position_count": int(last.get("exchange_position_count", 0)),
                "internal_position_count": int(last.get("internal_position_count", 0)),
                "mismatches": list(last.get("mismatches", []) or []),
                "safe_mode_triggered": bool(last.get("safe_mode_triggered", False)),
            }

        if reconciliation_result is None:
            return {"available": False, "safe_mode": False, "periodic": periodic_payload}
        return {
            "available": True,
            "reconciliation_id": getattr(reconciliation_result, "reconciliation_id", ""),
            "success": bool(reconciliation_result.success),
            "safe_mode": bool(reconciliation_result.safe_mode),
            "exchange_positions": len(reconciliation_result.exchange_positions),
            "db_positions": len(getattr(reconciliation_result, "db_positions", [])),
            "open_orders": len(reconciliation_result.exchange_open_orders),
            "mismatches": list(reconciliation_result.mismatches),
            "positions_without_sl": list(reconciliation_result.positions_without_sl),
            "actions_taken": list(reconciliation_result.actions_taken),
            "balance": reconciliation_result.balance,
            "leverage_settings": getattr(reconciliation_result, "leverage_settings", {}),
            "periodic": periodic_payload,
        }

    # ── /api/levels/{symbol}/{tf} — pivots + Fibonacci retracement ───────
    @app.get("/api/levels/{symbol:path}/{timeframe}")
    async def api_levels(symbol: str, timeframe: str, fib_lookback: int = 100) -> dict[str, Any]:
        """Classic floor pivots (from prior bar) + Fibonacci retracement
        (from the highest high / lowest low of the last `fib_lookback` bars)."""
        from analysis.pivot_levels import pivots_from_df, fib_from_df, nearest_level
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        df = None
        if data_manager is not None:
            clean = symbol.replace("/", "").replace(":USDT", "").upper()
            base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
            for sym_try in (f"{base}/USDT:USDT", symbol, f"{base}/USDT", clean):
                df = data_manager.get_dataframe("binance", sym_try, timeframe)
                if df is not None and len(df) > 0:
                    break
            else:
                df = None
        if df is None or len(df) < 3:
            return {"available": False, "symbol": symbol, "timeframe": timeframe}
        pivots = pivots_from_df(df) or {}
        fib = fib_from_df(df, lookback=max(2, int(fib_lookback))) or {}
        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        return {
            "available": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "last_close": last_close,
            "pivots": pivots,
            "fibonacci": fib,
            "nearest_pivot": nearest_level(last_close, pivots) if pivots else None,
            "nearest_fib": nearest_level(last_close, fib) if fib else None,
        }

    # ── /api/patterns/{symbol}/{tf} — candlestick pattern recognition ────
    @app.get("/api/patterns/{symbol:path}/{timeframe}")
    async def api_patterns(symbol: str, timeframe: str, lookback: int = 50) -> dict[str, Any]:
        """Run the 14 candlestick detectors over the last `lookback` bars and
        return per-bar pattern hits + a composite score (sum of pattern signs)."""
        from analysis.candlestick_patterns import detect_patterns
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        lookback = max(5, min(int(lookback), 200))
        df = None
        if data_manager is not None:
            clean = symbol.replace("/", "").replace(":USDT", "").upper()
            base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
            for sym_try in (f"{base}/USDT:USDT", symbol, f"{base}/USDT", clean):
                df = data_manager.get_dataframe("binance", sym_try, timeframe)
                if df is not None and len(df) > 0:
                    break
            else:
                df = None
        if df is None or len(df) < 5:
            return {"available": False, "symbol": symbol, "timeframe": timeframe}
        df = df.tail(lookback)
        patterns_df, composite = detect_patterns(df["open"], df["high"], df["low"], df["close"])
        # Latest-bar hits + counts over the window.
        latest = patterns_df.iloc[-1].to_dict()
        latest_hits = {k: int(v) for k, v in latest.items() if v != 0}
        window_counts = {col: int((patterns_df[col] != 0).sum()) for col in patterns_df.columns}
        return {
            "available": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "latest_hits": latest_hits,
            "latest_composite": int(composite.iloc[-1]) if len(composite) else 0,
            "window_counts": window_counts,
            "composite_series": [int(x) for x in composite.tolist()],
        }

    # ── /api/slippage — pre-trade fill estimator (REQ-EXE-005) ────────────
    @app.get("/api/slippage")
    async def api_slippage(
        symbol: str = Query("BTC/USDT:USDT"),
        side: str = Query("buy"),
        qty: float = Query(0.01),
    ) -> dict[str, Any]:
        """Walk the latest cached order book to estimate the average fill
        price + slippage for a target qty. Returns 0-bps when no book is
        cached (the bot couldn't make a marketable decision in that case)."""
        from analysis.slippage import estimate_fill
        try:
            symbol = _validate_symbol(symbol)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        if signal_generator is None or not hasattr(signal_generator, "_orderbook_scorer"):
            return {"available": False, "reason": "orderbook_scorer not wired"}
        scorer = signal_generator._orderbook_scorer
        bids, asks = scorer.book_for(symbol)
        # Try alias variants for the cache key.
        if not bids and not asks:
            clean = symbol.replace("/", "").replace(":USDT", "").upper()
            base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
            for sym_try in (f"{base}/USDT:USDT", f"{base}/USDT", clean):
                bids, asks = scorer.book_for(sym_try)
                if bids or asks:
                    symbol = sym_try
                    break
        if not bids and not asks:
            return {"available": False, "symbol": symbol, "reason": "no cached book"}
        est = estimate_fill(side, qty, bids, asks)
        return {
            "available": True,
            "symbol": symbol,
            "book": {"bid_levels": len(bids), "ask_levels": len(asks)},
            **est.to_dict(),
        }

    # ── /api/state — operational FSM snapshot (REQ-STATE-001..012) ────────
    @app.get("/api/state")
    async def api_state() -> dict[str, Any]:
        """Return the current operational state, allowed-next states, and
        recent transition history. Single source of truth for "are we
        cleared to trade right now?"."""
        if state_machine is None:
            return {"available": False, "current": "unknown"}
        snap = state_machine.snapshot()
        return {"available": True, **snap}

    # ── /api/uptime — burn-in / AC-001 progress ───────────────────────────
    @app.get("/api/uptime")
    async def api_uptime() -> dict[str, Any]:
        """Return current session uptime, crash count, and AC-001 (7-day
        no-crash) burn-in progress."""
        if uptime_tracker is None:
            return {"available": False}
        return {"available": True, **uptime_tracker.snapshot()}

    # ── /api/eventbus/stats — backpressure metrics (REQ-ARC-003) ──────────
    @app.get("/api/eventbus/stats")
    async def api_eventbus_stats() -> dict[str, Any]:
        """Queue size, capacity %, dropped event count, subscriber counts."""
        if event_bus is None or not hasattr(event_bus, "stats"):
            return {"available": False}
        return {"available": True, **event_bus.stats()}

    # ── /api/trace/{cid} — REQ-TR-001 end-to-end audit chain ──────────────
    @app.get("/api/trace/{correlation_id}")
    async def api_trace(correlation_id: str) -> dict[str, Any]:
        """Walk every audit table for rows tagged with this correlation_id.
        Returns {signals, risk_blocks, orders, fills, user_stream_events,
        pnl_snapshots, reconciliation_events} so the dashboard can render
        the full decision → fill → PnL chain for a single signal."""
        if not correlation_id:
            return {"available": False, "reason": "missing correlation_id"}
        if audit_repo is None:
            return {"available": False, "reason": "audit_repo not configured (DB unavailable)"}
        try:
            trace = await audit_repo.load_trace_by_correlation_id(correlation_id)
        except Exception as exc:
            return {"available": False, "error": sanitize_exception(exc)}
        # Surface counts for quick rendering.
        counts = {k: len(v) for k, v in trace.items()}
        total = sum(counts.values())
        # Convert datetime keys to ISO strings for JSON serialisation.
        def _norm(rows: list[dict]) -> list[dict]:
            out: list[dict] = []
            for row in rows:
                clean = {}
                for k, v in row.items():
                    if hasattr(v, "isoformat"):
                        clean[k] = v.isoformat()
                    else:
                        clean[k] = v
                out.append(clean)
            return out
        return {
            "available": True,
            "correlation_id": correlation_id,
            "total_rows": total,
            "counts": counts,
            "trace": {k: _norm(v) for k, v in trace.items()},
        }

    # ── /api/user-stream/status — user data stream health ─────────────────
    @app.get("/api/user-stream/status")
    async def api_user_stream_status() -> dict[str, Any]:
        if user_stream is None:
            return {"available": False, "connected": False}
        metrics = getattr(user_stream, "metrics", {})
        return {
            "available": True,
            "connected": bool(getattr(user_stream, "connected", False)),
            "disconnect_duration": float(getattr(user_stream, "disconnect_duration", 0.0)),
            "fills_processed": int(metrics.get("fills_processed", 0)),
            "fills_deduped": int(metrics.get("fills_deduped", 0)),
            "state_transitions": int(metrics.get("state_transitions", 0)),
            "invalid_transitions": int(metrics.get("invalid_transitions", 0)),
            "reconnects": int(metrics.get("reconnects", 0)),
            "messages_received": int(metrics.get("messages_received", 0)),
        }

    # ── /api/backtest/summary — backtest metrics ──────────────────────────
    @app.get("/api/backtest/summary")
    async def api_backtest_summary() -> dict[str, Any]:
        gross_notional = 0.0
        win_rate_val = 0.0
        sharpe = 0.0
        max_dd = 0.0
        avg_trade = 0.0
        profit_factor = 0.0
        if order_manager is not None:
            stats = order_manager.get_stats()
            total = int(stats.get("total_orders", 0))
            filled = int(stats.get("filled_orders", 0))
            gross_notional = float(stats.get("total_fill_value", 0))
            # Compute win/loss from actual positions
            if risk_manager is not None:
                positions = list(risk_manager.positions.values()) if hasattr(risk_manager, 'positions') else []
                wins = sum(1 for p in positions if getattr(p, 'pnl', 0) > 0)
                losses = sum(1 for p in positions if getattr(p, 'pnl', 0) < 0)
                total_trades = wins + losses
                if total_trades > 0:
                    win_rate_val = (wins / total_trades) * 100
                # Compute Sharpe approximation from position PnLs
                pnls = [getattr(p, 'pnl', 0) for p in positions if getattr(p, 'pnl', 0) != 0]
                if len(pnls) >= 2:
                    import statistics
                    mean_pnl = statistics.mean(pnls)
                    std_pnl = statistics.stdev(pnls)
                    if std_pnl > 0:
                        sharpe = (mean_pnl / std_pnl) * (252 ** 0.5)  # Annualized
                # Profit factor
                gross_profit = sum(p for p in pnls if p > 0) if pnls else 0
                gross_loss = abs(sum(p for p in pnls if p < 0)) if pnls else 0
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    profit_factor = float('inf')
            if filled > 0 and filled > 0:
                avg_trade = gross_notional / filled
        if risk_manager is not None:
            snap = risk_manager.get_risk_snapshot()
            max_dd = float(snap.get("drawdown_pct", 0))
        return {
            "gross_notional": gross_notional,
            "win_rate": win_rate_val,
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": max_dd,
            "avg_trade_notional": round(avg_trade, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
        }

    # ── /api/logs/recent — recent log entries ─────────────────────────────
    @app.get("/api/logs/recent")
    async def api_logs_recent() -> dict[str, Any]:
        return {"logs": list(_log_buffer)[-50:][::-1]}

    def _resolve_paper_fill_price(symbol: str, fallback_price: float | None = None) -> float:
        """Resolve a deterministic paper fill price from in-memory market state.

        Never block the dashboard route on a public exchange call while placing a
        paper order; if cached market/orderbook data is unavailable, fail closed
        rather than creating an unpriced filled position.
        """
        if fallback_price and fallback_price > 0:
            return float(fallback_price)
        now = time.time()
        max_age = 60.0
        sym_key = symbol.replace("/", "").replace(":", "").replace("-", "").upper()
        for key, cached in _orderbook_cache.items():
            cache_ts = float(cached.get("ts", 0) or 0)
            if cache_ts and (now - cache_ts) > max_age:
                continue
            key_norm = key.replace("/", "").replace(":", "").replace("-", "").upper()
            if sym_key in key_norm or key_norm in sym_key:
                bids = cached.get("bids") or []
                asks = cached.get("asks") or []
                if bids and asks:
                    return (float(bids[0][0]) + float(asks[0][0])) / 2.0
                if bids:
                    return float(bids[0][0])
                if asks:
                    return float(asks[0][0])
        clean = symbol.replace("/", "").replace(":USDT", "").upper()
        base = clean.replace("USDT", "") if clean.endswith("USDT") else clean
        if data_manager is not None:
            for sym_try in (f"{base}/USDT:USDT", symbol, f"{base}/USDT", clean):
                for tf_try in ("1m", "5m", "15m", "1h"):
                    try:
                        df = data_manager.get_dataframe("binance", sym_try, tf_try)
                    except Exception:
                        df = None
                    if df is not None and len(df) > 0:
                        price = float(df.iloc[-1].get("close", 0) or 0)
                        if price > 0:
                            return price
        market_ts = float(_market_cache.get("ts", 0) or 0)
        if market_ts and (now - market_ts) <= max_age:
            for coin in _market_cache.get("coins", []) or []:
                if str(coin.get("symbol", "")).upper() == base:
                    price = float(coin.get("price") or 0)
                    if price > 0:
                        return price
        return 0.0

    # ── /api/trade — place a trade ────────────────────────────────────────
    def _default_probe_symbol() -> str:
        exchanges_cfg = config.get_value("exchanges", default={}) or {}
        if isinstance(exchanges_cfg, dict):
            for ex_cfg in exchanges_cfg.values():
                if isinstance(ex_cfg, dict):
                    symbols = ex_cfg.get("symbols", []) or []
                    if symbols:
                        return str(symbols[0])
        return "BTC/USDT:USDT"

    @app.get("/api/pipeline/probe/status")
    async def api_pipeline_probe_status() -> dict[str, Any]:
        payload = getattr(app.state, "last_pipeline_probe", None)
        if isinstance(payload, dict):
            return {"available": True, **payload}
        return {"available": False, "reason": "probe_not_run"}

    @app.post("/api/pipeline/probe")
    async def api_pipeline_probe(request: Request) -> dict[str, Any]:
        """Paper-only dry-run probe for signal -> risk -> executor wiring."""
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        if not config.paper_mode:
            return {
                "success": False,
                "available": True,
                "error": "pipeline_probe_is_paper_only",
                "mode": "live",
            }

        from engine.signal_generator import TradingSignal

        probe_started = time.perf_counter()
        symbol = str(body.get("symbol") or _default_probe_symbol())
        direction = str(body.get("direction") or "long").lower()
        if direction not in {"long", "short"}:
            direction = "long"
        price = _resolve_paper_fill_price(symbol, float(body.get("price", 0) or 0) or None)
        stages: list[dict[str, Any]] = []

        def _stage(name: str, ok: bool, detail: str, **extra: Any) -> None:
            payload = {"name": name, "ok": bool(ok), "detail": str(detail)}
            payload.update(extra)
            stages.append(payload)

        if price <= 0:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "dry_run": True,
                "symbol": symbol,
                "error": "paper_probe_price_unavailable",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_probe = result
            return result

        sl_pct = max(0.001, float(body.get("stop_loss_pct", 1.0) or 1.0) / 100.0)
        tp_pct = max(sl_pct * 1.5, float(body.get("take_profit_pct", 2.0) or 2.0) / 100.0)
        stop_loss = price * (1.0 - sl_pct) if direction == "long" else price * (1.0 + sl_pct)
        take_profit = price * (1.0 + tp_pct) if direction == "long" else price * (1.0 - tp_pct)
        signal = TradingSignal(
            exchange="binance",
            symbol=symbol,
            direction=direction,
            score=1.0,
            technical_score=0.75,
            ml_score=0.25,
            sentiment_score=0.0,
            macro_score=0.0,
            news_score=0.0,
            orderbook_score=0.25,
            regime="paper_probe",
            regime_confidence=1.0,
            price=price,
            atr=max(price * 0.01, 1e-8),
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=int(time.time()),
            quality_score=100,
            session_name="paper_probe",
            metadata={
                "source": "pipeline_probe",
                "paper": True,
                "atr_percentile": 50,
                "adx": 25,
            },
            reasons=["operator_probe"],
        )

        try:
            md = await api_market_data_health()
            md_ok = bool(md.get("healthy", False)) or str(md.get("status", "")).upper() == "OK"
            _stage("market_data", md_ok, f"{md.get('status', 'unknown')} feeds={md.get('healthy_count', 0)}/{md.get('feed_count', 0)}")
        except Exception as exc:
            md_ok = False
            _stage("market_data", False, sanitize_exception(exc))

        risk_ok = False
        risk_reason = "risk_manager_unavailable"
        approved_size = 0.0
        if risk_manager is not None and not _risk_kill_switch_active():
            risk_started = time.perf_counter()
            try:
                risk_ok, risk_reason, approved_size = risk_manager.approve_signal(signal)
                if metrics and hasattr(metrics, "record_pipeline_latency"):
                    metrics.record_pipeline_latency("paper_probe_risk_gate", "binance", symbol, time.perf_counter() - risk_started)
            except Exception as exc:
                risk_ok = False
                risk_reason = sanitize_exception(exc)
        _stage("risk_gate", risk_ok, risk_reason, approved_size=round(float(approved_size or 0.0), 6))

        paper_execs = [
            exc for exc in (executors or [])
            if bool(getattr(exc, "is_paper", False)) or "simulated" in type(exc).__name__.lower()
        ]
        contract_details = [
            executor_contract_status(exc, require_order_controls=True, require_market_data=True)
            for exc in paper_execs
        ]
        executor_ok = bool(paper_execs and all(item.get("contract_ok", False) for item in contract_details))
        missing = sorted({b for item in contract_details for b in item.get("blockers", [])})
        detail = f"{len(paper_execs)} paper executor(s)"
        if missing:
            detail = f"missing {', '.join(missing)}"
        _stage("executor_contract", executor_ok, detail, details=contract_details)

        om_ok = order_manager is not None and hasattr(order_manager, "get_stats")
        om_stats = order_manager.get_stats() if om_ok else {}
        _stage(
            "order_manager",
            om_ok,
            f"{int(om_stats.get('open_orders', 0) or 0)} open, {int(om_stats.get('filled_orders', 0) or 0)} filled",
        )

        total_latency = time.perf_counter() - probe_started
        if metrics and hasattr(metrics, "record_pipeline_latency"):
            metrics.record_pipeline_latency("paper_probe_total", "binance", symbol, total_latency)

        success = bool(md_ok and risk_ok and executor_ok and om_ok)
        result = {
            "success": success,
            "available": True,
            "mode": "paper",
            "dry_run": True,
            "symbol": symbol,
            "direction": direction,
            "price": round(float(price), 8),
            "approved": bool(risk_ok),
            "approved_size": round(float(approved_size or 0.0), 6),
            "risk_reason": risk_reason,
            "latency_ms": round(total_latency * 1000.0, 3),
            "stages": stages,
            "timestamp": int(time.time()),
        }
        app.state.last_pipeline_probe = result
        return result

    @app.get("/api/pipeline/probe/fill/status")
    async def api_pipeline_fill_probe_status() -> dict[str, Any]:
        payload = getattr(app.state, "last_pipeline_fill_probe", None)
        if isinstance(payload, dict):
            return {"available": True, **payload}
        return {"available": False, "reason": "fill_probe_not_run"}

    @app.post("/api/pipeline/probe/fill")
    async def api_pipeline_fill_probe(request: Request) -> dict[str, Any]:
        """Paper-only entry-fill-close lifecycle probe.

        This intentionally places and closes a simulated paper position so the
        operator can verify real order-manager fills, risk position state,
        protective SL/TP fields, SQLite close persistence, and post-close
        reconciliation without any live exchange client.
        """
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        if not config.paper_mode:
            return {
                "success": False,
                "available": True,
                "error": "fill_probe_is_paper_only",
                "mode": "live",
            }

        from engine.signal_generator import TradingSignal

        probe_started = time.perf_counter()
        probe_started_ms = int(time.time() * 1000)
        stages: list[dict[str, Any]] = []

        def _stage(name: str, ok: bool, detail: str, **extra: Any) -> None:
            payload = {"name": name, "ok": bool(ok), "detail": str(detail)}
            payload.update(extra)
            stages.append(payload)

        paper_execs = [
            exc for exc in (executors or [])
            if (
                bool(getattr(exc, "is_paper", False))
                or "simulated" in type(exc).__name__.lower()
            )
            and hasattr(exc, "execute_signal")
        ]
        executor = next((exc for exc in paper_execs if hasattr(exc, "close_position")), None)
        if executor is None:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "error": "paper_executor_close_contract_unavailable",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_fill_probe = result
            return result

        exchange_id = str(getattr(executor, "exchange_id", "binance") or "binance").lower()
        preferred_symbol = str(body.get("symbol") or "").strip()
        direction = str(body.get("direction") or "long").lower()
        if direction not in {"long", "short"}:
            direction = "long"

        symbols: list[str] = []
        if preferred_symbol:
            symbols.append(preferred_symbol)
        else:
            exchange_cfg = config.get_value("exchanges", exchange_id, default={}) or {}
            cfg_symbols = exchange_cfg.get("symbols", []) if isinstance(exchange_cfg, dict) else []
            symbols.extend(str(sym) for sym in (cfg_symbols or []))
            symbols.append(_default_probe_symbol())
        seen_symbols: set[str] = set()
        symbols = [sym for sym in symbols if sym and not (sym in seen_symbols or seen_symbols.add(sym))]

        positions_before = dict(getattr(risk_manager, "positions", {}) or {}) if risk_manager is not None else {}

        def _fill_probe_open_rows(symbol_value: str) -> int:
            if sqlite_store is None or not hasattr(sqlite_store, "query"):
                return 0
            try:
                rows = sqlite_store.query(
                    "SELECT COUNT(*) AS n FROM positions WHERE exchange=? AND symbol=? AND close_time_ns IS NULL",
                    (exchange_id, symbol_value),
                )
                return int(rows[0].get("n", 0)) if rows else 0
            except Exception:
                return 0

        def _probe_symbol_in_cooldown(symbol_value: str) -> bool:
            if risk_manager is None:
                return False
            last_trade = float((getattr(risk_manager, "_last_trade_time", {}) or {}).get(symbol_value, 0.0) or 0.0)
            cooldown = float(getattr(risk_manager, "_cooldown_seconds", 0.0) or 0.0)
            return cooldown > 0 and last_trade > 0 and (time.time() - last_trade) < cooldown

        symbol = ""
        for candidate in symbols:
            key = f"{exchange_id}:{candidate}"
            if (
                key not in positions_before
                and _fill_probe_open_rows(candidate) == 0
                and not _probe_symbol_in_cooldown(candidate)
            ):
                symbol = candidate
                break
        if not symbol and preferred_symbol:
            symbol = preferred_symbol

        if not symbol:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "error": "no_available_probe_symbol",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_fill_probe = result
            return result

        key = f"{exchange_id}:{symbol}"
        if key in positions_before:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "symbol": symbol,
                "error": "probe_symbol_already_has_open_position",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_fill_probe = result
            return result

        price = _resolve_paper_fill_price(symbol, float(body.get("price", 0) or 0) or None)
        if price <= 0:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "symbol": symbol,
                "error": "paper_probe_price_unavailable",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_fill_probe = result
            return result

        sl_pct = max(0.001, float(body.get("stop_loss_pct", 1.0) or 1.0) / 100.0)
        tp_pct = max(sl_pct * 1.5, float(body.get("take_profit_pct", 2.0) or 2.0) / 100.0)
        stop_loss = price * (1.0 - sl_pct) if direction == "long" else price * (1.0 + sl_pct)
        take_profit = price * (1.0 + tp_pct) if direction == "long" else price * (1.0 - tp_pct)
        signal = TradingSignal(
            exchange=exchange_id,
            symbol=symbol,
            direction=direction,
            score=1.0,
            technical_score=0.75,
            ml_score=0.25,
            sentiment_score=0.0,
            macro_score=0.0,
            news_score=0.0,
            orderbook_score=0.25,
            regime="paper_lifecycle_probe",
            regime_confidence=1.0,
            price=price,
            atr=max(price * 0.01, 1e-8),
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=int(time.time() * 1000),
            quality_score=100,
            session_name="paper_lifecycle_probe",
            metadata={
                "source": "pipeline_fill_probe",
                "paper": True,
                "atr_percentile": 50,
                "adx": 25,
            },
            reasons=["operator_lifecycle_probe"],
        )

        sqlite_closed_before = -1
        if sqlite_store is not None and hasattr(sqlite_store, "query"):
            try:
                rows = sqlite_store.query(
                    "SELECT COUNT(*) AS n FROM positions WHERE exchange=? AND symbol=? AND close_time_ns IS NOT NULL",
                    (exchange_id, symbol),
                )
                sqlite_closed_before = int(rows[0].get("n", 0)) if rows else 0
            except Exception:
                sqlite_closed_before = -1

        try:
            md = await api_market_data_health()
            md_ok = bool(md.get("healthy", False)) or str(md.get("status", "")).upper() == "OK"
            _stage("market_data", md_ok, f"{md.get('status', 'unknown')} feeds={md.get('healthy_count', 0)}/{md.get('feed_count', 0)}")
        except Exception as exc:
            md_ok = False
            _stage("market_data", False, sanitize_exception(exc))

        result_order = None
        entry_ok = False
        entry_detail = "not_submitted"
        if md_ok and risk_manager is not None and not _risk_kill_switch_active():
            try:
                result_order = await executor.execute_signal(signal, float(body.get("size", 0) or 0.0))
                if result_order is None:
                    entry_detail = "executor_returned_none"
                else:
                    entry_detail = str(getattr(result_order, "status", "unknown"))
                    entry_ok = entry_detail in {"filled", "partially_filled"}
            except Exception as exc:
                entry_detail = sanitize_exception(exc)
        _stage("entry_order", entry_ok, entry_detail)

        if result_order is not None and str(getattr(result_order, "status", "")) == "partially_filled":
            deadline = time.time() + 2.0
            while time.time() < deadline:
                await asyncio.sleep(0.05)
                orders_for_symbol = (
                    order_manager.get_orders_by_symbol(exchange_id, symbol)
                    if order_manager is not None and hasattr(order_manager, "get_orders_by_symbol")
                    else []
                )
                if any(str(getattr(o.status, "value", o.status)) == "filled" for o in orders_for_symbol):
                    break

        pos = (getattr(risk_manager, "positions", {}) or {}).get(key) if risk_manager is not None else None
        position_ok = bool(pos is not None and not bool(getattr(pos, "pending_fill", False)))
        _stage(
            "position_open",
            position_ok,
            "active" if position_ok else "missing_or_pending",
            size=round(float(getattr(pos, "size", 0.0) or 0.0), 8) if pos is not None else 0.0,
        )

        protective_ok = False
        protective_detail = "position_unavailable"
        if pos is not None:
            entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
            sl = float(getattr(pos, "stop_loss", 0.0) or 0.0)
            tp = float(getattr(pos, "take_profit", 0.0) or 0.0)
            if direction == "long":
                protective_ok = sl > 0 and tp > 0 and sl < entry < tp
            else:
                protective_ok = sl > 0 and tp > 0 and tp < entry < sl
            protective_detail = f"SL={sl:.4f} TP={tp:.4f} entry={entry:.4f}"
        _stage("protective_bounds", protective_ok, protective_detail)

        close_ok = False
        close_detail = "not_attempted"
        close_result = None
        if position_ok:
            close_price = float(getattr(pos, "current_price", 0.0) or getattr(pos, "entry_price", price) or price)
            try:
                close_result = await executor.close_position(
                    symbol,
                    close_price,
                    reason="paper_lifecycle_probe",
                )
                close_ok = close_result is not None and str(getattr(close_result, "status", "")) == "closed"
                close_detail = str(getattr(close_result, "status", "close_failed")) if close_result else "close_failed"
            except Exception as exc:
                close_detail = sanitize_exception(exc)
        _stage("exit_order", close_ok, close_detail)

        positions_after = dict(getattr(risk_manager, "positions", {}) or {}) if risk_manager is not None else {}
        no_orphan_position = key not in positions_after

        active_probe_orders = []
        filled_probe_orders = []
        if order_manager is not None and hasattr(order_manager, "get_orders_by_symbol"):
            for order in order_manager.get_orders_by_symbol(exchange_id, symbol):
                created_at = int(getattr(order, "created_at", 0) or 0)
                if created_at and created_at < probe_started_ms - 1000:
                    continue
                status = str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", "")))
                if status in {"pending", "submitted", "open", "partially_filled"}:
                    active_probe_orders.append(order)
                if status == "filled":
                    filled_probe_orders.append(order)
        orders_ok = len(filled_probe_orders) >= 2 and not active_probe_orders
        _stage(
            "order_manager_lifecycle",
            orders_ok,
            f"filled={len(filled_probe_orders)} active={len(active_probe_orders)}",
        )

        sqlite_ok = False
        sqlite_detail = "sqlite_store_unavailable"
        if sqlite_closed_before >= 0 and sqlite_store is not None and hasattr(sqlite_store, "query"):
            try:
                rows = sqlite_store.query(
                    "SELECT COUNT(*) AS n FROM positions WHERE exchange=? AND symbol=? AND close_time_ns IS NOT NULL",
                    (exchange_id, symbol),
                )
                sqlite_closed_after = int(rows[0].get("n", 0)) if rows else sqlite_closed_before
                sqlite_ok = sqlite_closed_after > sqlite_closed_before
                sqlite_detail = f"closed_rows_delta={sqlite_closed_after - sqlite_closed_before}"
            except Exception as exc:
                sqlite_ok = False
                sqlite_detail = sanitize_exception(exc)
        _stage("sqlite_close_audit", sqlite_ok, sqlite_detail)

        recon_ok = bool(no_orphan_position and orders_ok)
        _stage(
            "post_close_reconciliation",
            recon_ok,
            f"orphan_position={not no_orphan_position} active_orders={len(active_probe_orders)}",
        )

        if not no_orphan_position and risk_manager is not None:
            try:
                await risk_manager.close_position(exchange_id, symbol, price)
            except Exception:
                pass

        total_latency = time.perf_counter() - probe_started
        if metrics and hasattr(metrics, "record_pipeline_latency"):
            metrics.record_pipeline_latency("paper_fill_probe_total", exchange_id, symbol, total_latency)
        _exchange_cache.pop("positions", None)
        _exchange_cache.pop("orders", None)

        success = all(bool(stage.get("ok")) for stage in stages)
        result = {
            "success": success,
            "available": True,
            "mode": "paper",
            "dry_run": False,
            "symbol": symbol,
            "exchange": exchange_id,
            "direction": direction,
            "entry_order_id": str(getattr(result_order, "order_id", "")) if result_order is not None else "",
            "exit_order_id": str(getattr(close_result, "order_id", "")) if close_result is not None else "",
            "entry_price": round(float(getattr(result_order, "price", price) or price), 8) if result_order is not None else round(float(price), 8),
            "exit_price": round(float(getattr(close_result, "price", 0.0) or 0.0), 8) if close_result is not None else 0.0,
            "latency_ms": round(total_latency * 1000.0, 3),
            "stages": stages,
            "timestamp": int(time.time()),
        }
        app.state.last_pipeline_fill_probe = result
        return result

    def _risk_position_db_ids() -> dict[str, int]:
        result: dict[str, int] = {}
        positions = getattr(risk_manager, "positions", {}) or {}
        if not isinstance(positions, dict):
            return result
        for key, pos in positions.items():
            db_id = getattr(pos, "_db_id", None)
            if db_id is not None:
                try:
                    result[str(key)] = int(db_id)
                except Exception:
                    pass
        return result

    def _build_recovery_ledger_audit() -> dict[str, Any]:
        if sqlite_store is None or not hasattr(sqlite_store, "query"):
            return {"available": False, "reason": "sqlite_store_unavailable"}

        rows = sqlite_store.query(
            "SELECT * FROM positions WHERE close_time_ns IS NULL "
            "ORDER BY exchange, symbol, direction, open_time_ns, id"
        )
        mem_positions = getattr(risk_manager, "positions", {}) or {}
        if not isinstance(mem_positions, dict):
            mem_positions = {}
        mem_db_ids = _risk_position_db_ids()
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            exchange = str(row.get("exchange", "") or "").lower()
            symbol = str(row.get("symbol", "") or "")
            direction = str(row.get("direction", "") or "").lower()
            key = f"{exchange}:{symbol}"
            group_key = f"{exchange}:{symbol}:{direction}"
            group = groups.setdefault(group_key, {
                "key": group_key,
                "risk_key": key,
                "exchange": exchange,
                "symbol": symbol,
                "direction": direction,
                "in_memory": key in mem_positions,
                "canonical_id": None,
                "rows": [],
                "duplicate_ids": [],
                "orphan": False,
            })
            row_payload = {
                "id": int(row.get("id", 0) or 0),
                "exchange": exchange,
                "symbol": symbol,
                "direction": direction,
                "entry_price": float(row.get("entry_price", 0.0) or 0.0),
                "size": float(row.get("size", 0.0) or 0.0),
                "open_time_ns": int(row.get("open_time_ns", 0) or 0),
                "is_paper": bool(int(row.get("is_paper", 0) or 0)),
            }
            group["rows"].append(row_payload)

        duplicate_ids: list[int] = []
        duplicate_groups: list[dict[str, Any]] = []
        orphan_groups: list[dict[str, Any]] = []
        for group in groups.values():
            rows_for_group = list(group.get("rows", []) or [])
            risk_key = str(group.get("risk_key", ""))
            mem_db_id = mem_db_ids.get(risk_key)
            row_ids = [int(r.get("id", 0) or 0) for r in rows_for_group]
            canonical_id = mem_db_id if mem_db_id in row_ids else (row_ids[0] if row_ids else None)
            group["canonical_id"] = canonical_id
            safe_duplicate_ids = [
                int(r.get("id", 0) or 0)
                for r in rows_for_group
                if int(r.get("id", 0) or 0) != canonical_id and bool(r.get("is_paper", False))
            ]
            group["duplicate_ids"] = safe_duplicate_ids
            duplicate_ids.extend(safe_duplicate_ids)
            if len(rows_for_group) > 1:
                duplicate_groups.append(group)
            if not bool(group.get("in_memory", False)):
                group["orphan"] = True
                orphan_groups.append(group)

        phantom_positions = [
            {"key": str(key), "db_id": mem_db_ids.get(str(key))}
            for key in mem_positions.keys()
            if str(key) not in {str(g.get("risk_key")) for g in groups.values()}
        ]
        return {
            "available": True,
            "mode": "paper" if config.paper_mode else "live",
            "paper_only_cleanup": True,
            "open_rows": len(rows),
            "in_memory_open_positions": len(mem_positions),
            "groups": list(groups.values()),
            "duplicate_groups": duplicate_groups,
            "duplicate_ids": duplicate_ids,
            "duplicate_count": len(duplicate_ids),
            "orphan_groups": orphan_groups,
            "orphan_count": len(orphan_groups),
            "phantom_positions": phantom_positions,
            "phantom_count": len(phantom_positions),
            "status": "WARN" if duplicate_ids or orphan_groups or phantom_positions else "PASS",
            "timestamp": int(time.time()),
        }

    @app.get("/api/recovery/status")
    async def api_recovery_status() -> dict[str, Any]:
        status: dict[str, Any] = {}
        if risk_manager is not None and hasattr(risk_manager, "get_startup_recovery_status"):
            try:
                status = risk_manager.get_startup_recovery_status()
            except Exception as exc:
                status = {"source": "sqlite", "success": False, "error": sanitize_exception(exc)}
        elif hasattr(app.state, "sqlite_recovery_result"):
            status = dict(getattr(app.state, "sqlite_recovery_result") or {})
        else:
            status = {"source": "sqlite", "success": True, "attempted": 0, "restored": 0}

        ledger = _build_recovery_ledger_audit()
        sqlite_open_count = 0
        if sqlite_store is not None and hasattr(sqlite_store, "query"):
            try:
                rows = sqlite_store.query("SELECT COUNT(*) AS n FROM positions WHERE close_time_ns IS NULL")
                sqlite_open_count = int(rows[0].get("n", 0)) if rows else 0
            except Exception:
                sqlite_open_count = -1
        status["sqlite_open_positions"] = sqlite_open_count
        status["in_memory_open_positions"] = len(getattr(risk_manager, "positions", {}) or {}) if risk_manager is not None else 0
        status["ledger"] = ledger
        return status

    @app.get("/api/recovery/ledger")
    async def api_recovery_ledger() -> dict[str, Any]:
        return _build_recovery_ledger_audit()

    @app.post("/api/recovery/ledger/archive-duplicates")
    async def api_archive_duplicate_ledger_rows(request: Request) -> dict[str, Any]:
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        if not config.paper_mode:
            return {"success": False, "mode": "live", "error": "paper_only_cleanup"}
        if sqlite_store is None or not hasattr(sqlite_store, "archive_paper_open_position"):
            return {"success": False, "mode": "paper", "error": "sqlite_archive_unavailable"}

        audit_before = _build_recovery_ledger_audit()
        allowed_ids = {int(i) for i in audit_before.get("duplicate_ids", []) or []}
        requested_ids = body.get("ids")
        if isinstance(requested_ids, list) and requested_ids:
            try:
                target_ids = {int(i) for i in requested_ids}
            except Exception:
                return {"success": False, "mode": "paper", "error": "invalid_ids"}
            disallowed = sorted(target_ids - allowed_ids)
            if disallowed:
                return {
                    "success": False,
                    "mode": "paper",
                    "error": "ids_not_safe_duplicates",
                    "disallowed_ids": disallowed,
                    "allowed_ids": sorted(allowed_ids),
                }
            archive_ids = sorted(target_ids)
        else:
            archive_ids = sorted(allowed_ids)

        if not archive_ids:
            return {
                "success": True,
                "mode": "paper",
                "archived": 0,
                "results": [],
                "ledger": audit_before,
            }

        results = []
        for pos_id in archive_ids:
            results.append(sqlite_store.archive_paper_open_position(
                pos_id,
                reason=str(body.get("reason", "dashboard_duplicate_cleanup") or "dashboard_duplicate_cleanup"),
            ))

        audit_after = _build_recovery_ledger_audit()
        try:
            if hasattr(app.state, "last_pipeline_fill_probe"):
                app.state.last_pipeline_fill_probe = None
        except Exception:
            pass
        return {
            "success": all(bool(r.get("success")) for r in results),
            "mode": "paper",
            "archived": sum(1 for r in results if r.get("success")),
            "results": results,
            "ledger": audit_after,
        }

    @app.post("/api/pipeline/probe/recovery")
    async def api_pipeline_recovery_probe(request: Request) -> dict[str, Any]:
        """Paper-only cold-start recovery drill using the SQLite position ledger."""
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        if not config.paper_mode:
            return {
                "success": False,
                "available": True,
                "mode": "live",
                "error": "recovery_probe_is_paper_only",
            }
        if sqlite_store is None or not hasattr(sqlite_store, "query"):
            return {
                "success": False,
                "available": True,
                "mode": "paper",
                "error": "sqlite_store_unavailable",
            }

        from engine.signal_generator import TradingSignal
        from execution.risk_manager import RiskManager

        started = time.perf_counter()
        stages: list[dict[str, Any]] = []

        def _stage(name: str, ok: bool, detail: str, **extra: Any) -> None:
            payload = {"name": name, "ok": bool(ok), "detail": str(detail)}
            payload.update(extra)
            stages.append(payload)

        paper_execs = [
            exc for exc in (executors or [])
            if (
                bool(getattr(exc, "is_paper", False))
                or "simulated" in type(exc).__name__.lower()
            )
            and hasattr(exc, "execute_signal")
            and hasattr(exc, "close_position")
        ]
        executor = paper_execs[0] if paper_execs else None
        if executor is None:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "error": "paper_executor_close_contract_unavailable",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_recovery_probe = result
            return result

        exchange_id = str(getattr(executor, "exchange_id", "binance") or "binance").lower()
        direction = str(body.get("direction") or "long").lower()
        if direction not in {"long", "short"}:
            direction = "long"

        preferred_symbol = str(body.get("symbol") or "").strip()
        symbols: list[str] = [preferred_symbol] if preferred_symbol else []
        if not symbols:
            exchange_cfg = config.get_value("exchanges", exchange_id, default={}) or {}
            cfg_symbols = exchange_cfg.get("symbols", []) if isinstance(exchange_cfg, dict) else []
            symbols.extend(str(sym) for sym in (cfg_symbols or []))
            symbols.append(_default_probe_symbol())
        seen_symbols: set[str] = set()
        symbols = [sym for sym in symbols if sym and not (sym in seen_symbols or seen_symbols.add(sym))]

        current_positions = dict(getattr(risk_manager, "positions", {}) or {}) if risk_manager is not None else {}

        def _open_row_count(symbol_value: str) -> int:
            try:
                rows = sqlite_store.query(
                    "SELECT COUNT(*) AS n FROM positions WHERE exchange=? AND symbol=? AND close_time_ns IS NULL",
                    (exchange_id, symbol_value),
                )
                return int(rows[0].get("n", 0)) if rows else 0
            except Exception:
                return -1

        symbol = ""
        open_rows_before = 0
        for candidate in symbols:
            key = f"{exchange_id}:{candidate}"
            candidate_rows = _open_row_count(candidate)
            last_trade = 0.0
            cooldown = 0.0
            if risk_manager is not None:
                last_trade = float((getattr(risk_manager, "_last_trade_time", {}) or {}).get(candidate, 0.0) or 0.0)
                cooldown = float(getattr(risk_manager, "_cooldown_seconds", 0.0) or 0.0)
            in_cooldown = cooldown > 0 and last_trade > 0 and (time.time() - last_trade) < cooldown
            if key not in current_positions and candidate_rows == 0 and not in_cooldown:
                symbol = candidate
                open_rows_before = candidate_rows
                break
        if not symbol:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "error": "no_clean_probe_symbol",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_recovery_probe = result
            return result

        key = f"{exchange_id}:{symbol}"
        price = _resolve_paper_fill_price(symbol, float(body.get("price", 0) or 0) or None)
        if price <= 0:
            result = {
                "success": False,
                "available": True,
                "mode": "paper",
                "symbol": symbol,
                "error": "paper_probe_price_unavailable",
                "stages": stages,
                "timestamp": int(time.time()),
            }
            app.state.last_pipeline_recovery_probe = result
            return result

        signal = TradingSignal(
            exchange=exchange_id,
            symbol=symbol,
            direction=direction,
            score=1.0,
            technical_score=0.75,
            ml_score=0.25,
            sentiment_score=0.0,
            macro_score=0.0,
            news_score=0.0,
            orderbook_score=0.25,
            regime="paper_recovery_probe",
            regime_confidence=1.0,
            price=price,
            atr=max(price * 0.01, 1e-8),
            stop_loss=price * 0.985 if direction == "long" else price * 1.015,
            take_profit=price * 1.03 if direction == "long" else price * 0.97,
            timestamp=int(time.time() * 1000),
            quality_score=100,
            session_name="paper_recovery_probe",
            metadata={"source": "pipeline_recovery_probe", "paper": True, "atr_percentile": 50, "adx": 25},
            reasons=["operator_recovery_probe"],
        )

        entry = None
        try:
            entry = await executor.execute_signal(signal, float(body.get("size", 0) or 0.0))
            entry_ok = entry is not None and str(getattr(entry, "status", "")) in {"filled", "partially_filled"}
            _stage("entry_order", entry_ok, str(getattr(entry, "status", "not_submitted")) if entry else "not_submitted")
        except Exception as exc:
            entry_ok = False
            _stage("entry_order", False, sanitize_exception(exc))

        open_rows = []
        if entry_ok:
            try:
                open_rows = sqlite_store.query(
                    "SELECT * FROM positions WHERE exchange=? AND symbol=? AND close_time_ns IS NULL ORDER BY id DESC",
                    (exchange_id, symbol),
                )
                _stage(
                    "sqlite_open_audit",
                    len(open_rows) == open_rows_before + 1,
                    f"open_rows_delta={len(open_rows) - open_rows_before}",
                )
            except Exception as exc:
                _stage("sqlite_open_audit", False, sanitize_exception(exc))
        else:
            _stage("sqlite_open_audit", False, "entry_not_open")

        restored_positions: dict[str, Any] = {}
        recovery_result: dict[str, Any] = {}
        if open_rows:
            try:
                fresh_risk = RiskManager(config, event_bus, sqlite_store=sqlite_store)
                recovery_result = await fresh_risk.restore_open_positions_from_sqlite(open_rows)
                restored_positions = dict(getattr(fresh_risk, "positions", {}) or {})
                restored = restored_positions.get(key)
                restored_ok = restored is not None and not bool(getattr(restored, "pending_fill", False))
                _stage(
                    "restore_risk_state",
                    restored_ok,
                    f"restored={int(recovery_result.get('restored', 0) or 0)}",
                    recovery=recovery_result,
                )
                _stage(
                    "dedupe_guard",
                    len(restored_positions) == 1 and key in restored_positions,
                    f"in_memory_keys={len(restored_positions)}",
                )
            except Exception as exc:
                _stage("restore_risk_state", False, sanitize_exception(exc))
                _stage("dedupe_guard", False, "restore_failed")
        else:
            _stage("restore_risk_state", False, "sqlite_open_row_missing")
            _stage("dedupe_guard", False, "sqlite_open_row_missing")

        close_result = None
        if entry_ok:
            try:
                close_price = float(getattr(entry, "price", price) or price)
                close_result = await executor.close_position(symbol, close_price, reason="paper_recovery_probe_cleanup")
                _stage(
                    "cleanup_close",
                    close_result is not None and str(getattr(close_result, "status", "")) == "closed",
                    str(getattr(close_result, "status", "close_failed")) if close_result else "close_failed",
                )
            except Exception as exc:
                _stage("cleanup_close", False, sanitize_exception(exc))
        else:
            _stage("cleanup_close", True, "nothing_opened")

        open_rows_after = _open_row_count(symbol)
        no_orphan = key not in (getattr(risk_manager, "positions", {}) or {})
        cleanup_ok = open_rows_after == open_rows_before and no_orphan
        _stage(
            "post_cleanup",
            cleanup_ok,
            f"open_rows_after={open_rows_after} orphan_position={not no_orphan}",
        )
        if not cleanup_ok and risk_manager is not None:
            try:
                await risk_manager.close_position(exchange_id, symbol, price)
            except Exception:
                pass

        success = all(bool(stage.get("ok")) for stage in stages)
        result = {
            "success": success,
            "available": True,
            "mode": "paper",
            "symbol": symbol,
            "exchange": exchange_id,
            "direction": direction,
            "entry_order_id": str(getattr(entry, "order_id", "")) if entry is not None else "",
            "exit_order_id": str(getattr(close_result, "order_id", "")) if close_result is not None else "",
            "recovery": recovery_result,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "stages": stages,
            "timestamp": int(time.time()),
        }
        app.state.last_pipeline_recovery_probe = result
        return result

    @app.post("/api/trade")
    async def api_trade(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            sym = str(body.get("symbol", "BTC/USDT:USDT"))
            side_str = str(body.get("side", "BUY")).upper()
            size = float(body.get("size", 0))
            order_type_str = str(body.get("order_type", "market")).lower()
            price = float(body.get("price", 0)) if order_type_str == "limit" else None
            leverage = int(body.get("leverage", 1))

            if order_type_str == "limit" and (price is None or price <= 0):
                return {"success": False, "error": "price required and must be > 0 for LIMIT orders"}
            if size <= 0:
                return {"success": False, "error": "size must be > 0"}

            # Normalize symbol: BTC/USDT → BTC/USDT:USDT for futures
            if "/" in sym and ":USDT" not in sym and sym.endswith("USDT"):
                sym = sym + ":USDT"

            venue = str(body.get("venue", "binance"))

            if config.paper_mode:
                if order_manager is None:
                    return {"success": False, "error": "order_manager not available"}
                from execution.order_manager import OrderSide, OrderType
                side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
                ot = OrderType.MARKET if order_type_str == "market" else OrderType.LIMIT
                fill_price = 0.0
                if ot == OrderType.MARKET:
                    fill_price = _resolve_paper_fill_price(sym, price)
                    if fill_price <= 0:
                        return {"success": False, "error": "paper fill price unavailable"}
                success, order, reason = await order_manager.place_order(
                    exchange=venue, symbol=sym, side=side, quantity=size,
                    price=price or 0, order_type=ot,
                    metadata={"source": "ui", "paper": True,
                              "stop_loss_pct": body.get("stop_loss_pct", 2),
                              "take_profit_pct": body.get("take_profit_pct", 4),
                              "leverage": leverage},
                )
                if not success:
                    return {"success": False, "error": reason}

                # Paper market orders are execution intents, not passive order tickets:
                # simulate an immediate fill so the UI, order ledger, and risk ledger
                # converge.  Limit orders intentionally remain pending unless a
                # dedicated paper matching engine records a later fill.
                if ot != OrderType.MARKET:
                    return {
                        "success": True,
                        "order_id": getattr(order, "order_id", "unknown"),
                        "paper": True,
                        "status": getattr(getattr(order, "status", None), "value", "pending"),
                        "filled": 0.0,
                        "price": price or 0.0,
                    }

                client_order_id = getattr(order, "client_order_id", "")
                exchange_order_id = f"paper-{getattr(order, 'order_id', client_order_id)}"
                await order_manager.confirm_order_submission(client_order_id, exchange_order_id)
                filled_order = await order_manager.record_fill(
                    client_order_id=client_order_id,
                    fill_id=f"{exchange_order_id}-fill-1",
                    quantity=size,
                    price=fill_price,
                    fee=0.0,
                )

                if risk_manager is not None:
                    from engine.signal_generator import TradingSignal
                    direction = "long" if side_str == "BUY" else "short"
                    sl_pct = float(body.get("stop_loss_pct", 2)) / 100
                    tp_pct = float(body.get("take_profit_pct", 4)) / 100
                    sig = TradingSignal(
                        exchange=venue, symbol=sym, direction=direction,
                        price=fill_price, score=1.0, quality_score=100,
                        stop_loss=fill_price * (1 - sl_pct) if direction == "long" else fill_price * (1 + sl_pct),
                        take_profit=fill_price * (1 + tp_pct) if direction == "long" else fill_price * (1 - tp_pct),
                        technical_score=0.0, ml_score=0.0, sentiment_score=0.0,
                        macro_score=0.0, news_score=0.0, orderbook_score=0.0,
                        regime="paper_manual", regime_confidence=1.0,
                        atr=max(fill_price * 0.01, 1e-8),
                        timestamp=int(time.time()),
                        metadata={"source": "manual_ui", "paper": True, "order_id": getattr(order, "order_id", "")},
                    )
                    await risk_manager.open_position(sig, size * fill_price)

                _exchange_cache.pop("positions", None)
                _exchange_cache.pop("orders", None)

                return {
                    "success": True,
                    "order_id": getattr(order, "order_id", "unknown"),
                    "paper": True,
                    "status": getattr(getattr(filled_order or order, "status", None), "value", "filled"),
                    "filled": float(getattr(filled_order or order, "filled_quantity", size)),
                    "price": fill_price,
                }
            else:
                manual_live_allowed = bool(auth_cfg.get("allow_manual_live_trading", False))
                if not manual_live_allowed:
                    return {
                        "success": False,
                        "error": "manual live trading is disabled; set monitoring.dashboard_api.auth.allow_manual_live_trading=true only after live preflight and operator confirmation",
                    }

                required_confirmation = f"PLACE {side_str} {sym}"
                provided_confirmation = str(body.get("confirmation", "") or "").strip()
                if provided_confirmation != required_confirmation:
                    return {
                        "success": False,
                        "error": f"manual live trade requires typed confirmation: {required_confirmation}",
                    }

                if reconciliation_result is None:
                    return {"success": False, "error": "manual live trade blocked: startup reconciliation has not run"}
                recon_mismatches = list(getattr(reconciliation_result, "mismatches", []) or [])
                recon_positions_without_sl = list(getattr(reconciliation_result, "positions_without_sl", []) or [])
                if (
                    not bool(getattr(reconciliation_result, "success", False))
                    or bool(getattr(reconciliation_result, "safe_mode", False))
                    or recon_mismatches
                    or recon_positions_without_sl
                ):
                    return {"success": False, "error": "manual live trade blocked: reconciliation is not clean"}

                if not bool(getattr(db_handler, "available", False)):
                    return {"success": False, "error": "manual live trade blocked: audit DB unavailable"}
                if risk_manager is None or _risk_kill_switch_active():
                    return {"success": False, "error": "manual live trade blocked: risk manager unavailable or kill switch active"}

                # Live mode: send order directly to the exchange via executor client
                client = None
                rate_limiter = None
                for exc in (executors or []):
                    if getattr(exc, "exchange_id", "") == venue:
                        client = getattr(exc, "_client", None)
                        rate_limiter = getattr(exc, "_rate_limiter", None)
                        break
                if client is None:
                    return {"success": False, "error": f"No live client for {venue}"}

                if rate_limiter:
                    await rate_limiter.acquire()

                ccxt_side = "buy" if side_str == "BUY" else "sell"
                t_order_start = time.monotonic()
                if order_type_str == "market":
                    order = await client.create_market_order(
                        symbol=sym, side=ccxt_side, amount=size, params={},
                    )
                else:
                    order = await client.create_limit_order(
                        symbol=sym, side=ccxt_side, amount=size,
                        price=price, params={},
                    )
                order_latency = time.monotonic() - t_order_start

                if metrics and hasattr(metrics, "record_order_latency"):
                    metrics.record_order_latency(venue, order_type_str, order_latency)

                fill_price = float(order.get("average", order.get("price", price or 0)))
                filled = float(order.get("filled", size))
                status = order.get("status", "unknown")

                # Track in risk_manager if filled
                if status in ("closed", "filled") and filled > 0 and risk_manager:
                    from engine.signal_generator import TradingSignal
                    direction = "long" if side_str == "BUY" else "short"
                    sl_pct = float(body.get("stop_loss_pct", 2)) / 100
                    tp_pct = float(body.get("take_profit_pct", 4)) / 100
                    sig = TradingSignal(
                        exchange=venue, symbol=sym, direction=direction,
                        price=fill_price, score=1.0, quality_score=100,
                        stop_loss=fill_price * (1 - sl_pct) if direction == "long" else fill_price * (1 + sl_pct),
                        take_profit=fill_price * (1 + tp_pct) if direction == "long" else fill_price * (1 - tp_pct),
                        technical_score=0.0, ml_score=0.0, sentiment_score=0.0,
                        macro_score=0.0, news_score=0.0, orderbook_score=0.0,
                        regime="unknown", regime_confidence=0.0,
                        atr=fill_price * 0.01,
                        timestamp=int(time.time()),
                        metadata={"source": "manual_ui"},
                    )
                    await risk_manager.open_position(sig, filled * fill_price)
                # Invalidate cache after position change
                _exchange_cache.pop("positions", None)
                _exchange_cache.pop("orders", None)

                return {
                    "success": True,
                    "order_id": order.get("id", ""),
                    "paper": False,
                    "status": status,
                    "filled": filled,
                    "price": fill_price,
                }
        except Exception as exc:
            logger.error("Manual trade error: {}", exc)
            return {"success": False, "error": sanitize_exception(exc)}

    # ── /api/positions/close-all — close all positions ────────────────────
    @app.post("/api/positions/close-all")
    async def api_close_all() -> dict[str, Any]:
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        closed = await risk_manager.activate_kill_switch()
        # Paper/demo UI convenience may clear the switch after flattening. In
        # non-paper mode, never auto-deactivate the kill switch from this route;
        # an operator must perform an explicit readiness/reconciliation release.
        if config.paper_mode:
            risk_manager.deactivate_kill_switch()
        return {"success": True, "closed": len(closed), "kill_switch_active": bool(getattr(risk_manager, "kill_switch_active", False))}

    # ── /api/positions/close — close single position (frontend format) ───
    @app.post("/api/positions/close")
    async def api_close_position(request: Request) -> dict[str, Any]:
        """Close a single position. Body: {symbol: "BTC/USDT:USDT", venue: "binance"}."""
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        try:
            body = await request.json()
            symbol = str(body.get("symbol", ""))
            venue = str(body.get("venue", "binance"))
            if not symbol:
                return {"success": False, "error": "symbol required"}

            key = f"{venue}:{symbol}"
            pos = risk_manager.positions.get(key)
            if pos is None:
                return {"success": False, "error": f"Position not found: {symbol}"}

            close_price = float(pos.current_price)

            close_executor = None
            for exc in (executors or []):
                ex_id = str(getattr(exc, "exchange_id", "") or "").lower()
                if ex_id == venue.lower() and callable(getattr(exc, "close_position", None)):
                    close_executor = exc
                    break
            if close_executor is None:
                return {
                    "success": False,
                    "error": "executor close contract unavailable",
                    "symbol": symbol,
                    "venue": venue,
                }

            close_started = time.monotonic()
            close_result = await close_executor.close_position(
                symbol,
                close_price,
                reason="dashboard_close",
            )
            if close_result is None:
                return {"success": False, "error": "executor close failed", "symbol": symbol, "venue": venue}

            if metrics and hasattr(metrics, "record_order_latency"):
                metrics.record_order_latency(venue, "close", time.monotonic() - close_started)
            _exchange_cache.pop("positions", None)
            _exchange_cache.pop("orders", None)

            still_open = (
                f"{venue}:{symbol}" in risk_manager.positions
                or f"{venue.lower()}:{symbol}" in risk_manager.positions
            )
            closed_trades = list(getattr(risk_manager, "_closed_trades", []) or [])
            realized_pnl = 0.0
            if closed_trades:
                last_trade = closed_trades[-1]
                if (
                    last_trade.get("symbol") == symbol
                    and str(last_trade.get("exchange", "")).lower() == venue.lower()
                ):
                    realized_pnl = float(last_trade.get("pnl", 0.0) or 0.0)
            if still_open:
                return {
                    "success": False,
                    "error": "executor close sent but risk state still shows open; run reconciliation",
                    "symbol": symbol,
                    "venue": venue,
                    "order_id": str(getattr(close_result, "order_id", "")),
                }

            return {
                "success": True,
                "symbol": symbol,
                "venue": venue,
                "exit_price": round(float(getattr(close_result, "price", close_price) or close_price), 8),
                "realized_pnl": round(realized_pnl, 4),
                "order_id": str(getattr(close_result, "order_id", "")),
                "paper_order_id": str(getattr(close_result, "order_id", "")) if config.paper_mode else "",
            }
        except Exception as exc:
            logger.error("Close position error: {}", exc)
            return {"success": False, "error": sanitize_exception(exc)}

    # ── /api/positions/breakeven — set break-even on positions ────────────
    @app.post("/api/positions/breakeven")
    async def api_breakeven() -> dict[str, Any]:
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        count = 0
        async with risk_manager._lock:
            for pos in risk_manager._positions.values():
                if pos.pnl > 0:
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_moved = True
                    count += 1
        return {"success": True, "updated_positions": count}

    # ── Paper feed helpers (use the main event bus) ───────────────────────
    _main_paper_feed_task: asyncio.Task | None = None
    _main_paper_sltp_task: asyncio.Task | None = None
    _main_paper_exec_registered = False
    _paper_trades_main: list[dict] = []

    # Load persisted paper trades from SQLite on startup
    if sqlite_store:
        try:
            _saved = sqlite_store.get_paper_trades(limit=2000)
            if _saved:
                _paper_trades_main.extend(_saved)
                logger.info("Loaded {} paper trades from SQLite", len(_saved))
        except Exception as _le:
            logger.debug("Failed to load paper trades: {}", _le)

    async def _start_paper_feed_main(bus: EventBus) -> None:
        nonlocal _main_paper_feed_task, _main_paper_exec_registered, _paper_trades_main, _main_paper_sltp_task
        if _main_paper_feed_task is not None:
            return  # already running
        try:
            from data_ingestion.paper_feed import PaperFeed
            symbols_cfg = config.get_value("exchanges", "binance", "symbols") or ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
            feed = PaperFeed(
                event_bus=bus,
                symbols=symbols_cfg,
                timeframes=["1m", "15m", "1h", "4h", "1d"],
                poll_interval=30.0,
            )
            _main_paper_feed_task = asyncio.create_task(feed.run())
            logger.info("Paper feed started on main event bus for auto-trading")
        except Exception as exc:
            logger.error("Failed to start paper feed: {}", exc)
        # Register paper executor on main bus (once)
        if not _main_paper_exec_registered:
            async def _handle_signal_main(signal):
                import time as _time
                trade_id = f"paper_{int(_time.time()*1000)}"
                direction = getattr(signal, 'direction', 'unknown')
                sym = getattr(signal, 'symbol', '??')
                price = getattr(signal, 'price', 0)
                score = getattr(signal, 'score', 0)
                sl = getattr(signal, 'stop_loss', 0)
                tp = getattr(signal, 'take_profit', 0)
                meta = getattr(signal, 'metadata', {}) or {}

                # Reject signals with stale timestamps (from historical seeding)
                sig_ts = getattr(signal, 'timestamp', 0)
                if sig_ts > 0 and abs(_time.time() - sig_ts) > 120:
                    logger.debug("Paper executor: skipping stale signal (age={}s) for {}",
                                 int(abs(_time.time() - sig_ts)), sym)
                    return
                # Skip if already have an open trade on this symbol
                open_same = [t for t in _paper_trades_main
                             if t.get("symbol") == sym and t.get("status") == "OPEN"]
                if open_same:
                    logger.debug("Paper executor: skipping {} — already {} open trade(s)",
                                 sym, len(open_same))
                    return
                equity = 100000.0
                risk_pct = 0.02
                size_usd = equity * risk_pct
                qty = size_usd / price if price > 0 else 0
                paper_trade = {
                    "id": trade_id, "symbol": sym, "direction": direction,
                    "price": price, "quantity": round(qty, 6),
                    "notional": round(size_usd, 2), "score": round(score, 3),
                    "stop_loss": round(sl, 2), "take_profit": round(tp, 2),
                    "status": "OPEN", "timestamp": int(_time.time()),
                    "reasons": getattr(signal, 'reasons', []),
                    # Tiered TP tracking
                    "remaining_qty": round(qty, 6),
                    "tp_tiers": [
                        {"level": round(meta.get("tp1_price", tp), 2),
                         "close_pct": meta.get("tp1_close_pct", 0.50), "hit": False},
                        {"level": round(meta.get("tp2_price", 0), 2),
                         "close_pct": meta.get("tp2_close_pct", 0.30), "hit": False},
                    ],
                    "supertrend_trail": meta.get("supertrend_trail", 0),
                    "tp3_close_pct": meta.get("tp3_close_pct", 0.20),
                    "realized_pnl": 0.0,
                    "partial_fills": [],
                }
                _paper_trades_main.append(paper_trade)
                # Persist to SQLite
                if sqlite_store:
                    try:
                        sqlite_store.upsert_paper_trade(paper_trade)
                    except Exception as _pe:
                        logger.debug("Paper trade persist error: {}", _pe)
                logger.info(
                    "PAPER TRADE: {} {} {:.6f} @ ${:.2f} (score={:.2f}) SL={:.2f} TP1={:.2f} TP2={:.2f} [{}]",
                    direction.upper(), sym, qty, price, score, sl,
                    paper_trade["tp_tiers"][0]["level"],
                    paper_trade["tp_tiers"][1]["level"],
                    trade_id,
                )
            bus.subscribe("SIGNAL", _handle_signal_main)
            _main_paper_exec_registered = True
            logger.info("Paper executor subscribed to SIGNAL events on main bus")
        # Start SL/TP monitor
        if _main_paper_sltp_task is None:
            _main_paper_sltp_task = asyncio.create_task(_paper_sltp_monitor())
            logger.info("Paper SL/TP monitor started")

    async def _stop_paper_feed_main() -> None:
        nonlocal _main_paper_feed_task, _main_paper_sltp_task
        if _main_paper_feed_task is not None:
            _main_paper_feed_task.cancel()
            _main_paper_feed_task = None
            logger.info("Paper feed stopped")
        if _main_paper_sltp_task is not None:
            _main_paper_sltp_task.cancel()
            _main_paper_sltp_task = None
            logger.info("Paper SL/TP monitor stopped")

    async def _fetch_ticker_prices(symbols: list[str]) -> dict[str, float]:
        """Fetch current mark prices from Binance public API."""
        import httpx as _httpx
        prices: dict[str, float] = {}
        try:
            async with _httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://fapi.binance.com/fapi/v1/ticker/price")
                resp.raise_for_status()
                for item in resp.json():
                    prices[item["symbol"]] = float(item["price"])
        except Exception as exc:
            logger.debug("Paper SL/TP price fetch error: {}", exc)
        return prices

    def _symbol_to_binance(sym: str) -> str:
        """Convert BTC/USDT:USDT → BTCUSDT for Binance API lookup."""
        return sym.replace("/", "").replace(":USDT", "").upper()

    async def _paper_sltp_monitor() -> None:
        """Background task: check open paper trades against current prices for SL/TP."""
        import time as _time

        logger.info("Paper SL/TP monitor running — checking every 5s")
        try:
            while True:
                await asyncio.sleep(5)
                open_trades = [t for t in _paper_trades_main if t.get("status") == "OPEN"]
                if not open_trades:
                    continue

                # Fetch current prices
                needed_symbols = list({t["symbol"] for t in open_trades})
                prices = await _fetch_ticker_prices(needed_symbols)
                if not prices:
                    continue

                now = int(_time.time())
                for trade in open_trades:
                    bsym = _symbol_to_binance(trade["symbol"])
                    current_price = prices.get(bsym)
                    if current_price is None:
                        continue

                    direction = trade["direction"]
                    entry = trade["price"]
                    sl = trade["stop_loss"]
                    remaining = trade.get("remaining_qty", trade["quantity"])

                    # ── Check Stop Loss ──
                    sl_hit = False
                    if direction == "long" and current_price <= sl:
                        sl_hit = True
                    elif direction == "short" and current_price >= sl:
                        sl_hit = True

                    if sl_hit:
                        pnl_per_unit = (current_price - entry) if direction == "long" else (entry - current_price)
                        sl_pnl = pnl_per_unit * remaining
                        trade["realized_pnl"] = round(trade.get("realized_pnl", 0) + sl_pnl, 2)
                        trade["partial_fills"].append({
                            "type": "SL", "price": round(current_price, 2),
                            "qty": round(remaining, 6), "pnl": round(sl_pnl, 2),
                            "timestamp": now,
                        })
                        trade["remaining_qty"] = 0
                        trade["status"] = "CLOSED"
                        trade["close_price"] = round(current_price, 2)
                        trade["close_reason"] = "stop_loss"
                        trade["close_time"] = now
                        logger.info(
                            "PAPER SL HIT: {} {} @ ${:.2f} → ${:.2f} PnL=${:.2f} [{}]",
                            direction.upper(), trade["symbol"], entry,
                            current_price, trade["realized_pnl"], trade["id"],
                        )
                        if signal_generator is not None:
                            signal_generator.record_trade_result(
                                trade["symbol"], is_win=(trade["realized_pnl"] > 0),
                            )
                        if sqlite_store:
                            try:
                                sqlite_store.upsert_paper_trade(trade)
                            except Exception:
                                pass
                        continue

                    # ── Check Tiered Take Profits ──
                    tp_tiers = trade.get("tp_tiers", [])
                    original_qty = trade["quantity"]
                    for i, tier in enumerate(tp_tiers):
                        if tier.get("hit"):
                            continue
                        level = tier["level"]
                        if level <= 0:
                            continue

                        tp_hit = False
                        if direction == "long" and current_price >= level:
                            tp_hit = True
                        elif direction == "short" and current_price <= level:
                            tp_hit = True

                        if tp_hit:
                            close_qty = round(original_qty * tier["close_pct"], 6)
                            close_qty = min(close_qty, remaining)
                            if close_qty <= 0:
                                continue
                            pnl_per_unit = (current_price - entry) if direction == "long" else (entry - current_price)
                            tier_pnl = pnl_per_unit * close_qty
                            trade["realized_pnl"] = round(trade.get("realized_pnl", 0) + tier_pnl, 2)
                            remaining -= close_qty
                            trade["remaining_qty"] = round(remaining, 6)
                            tier["hit"] = True
                            tier_name = f"TP{i + 1}"
                            trade["partial_fills"].append({
                                "type": tier_name, "price": round(current_price, 2),
                                "qty": round(close_qty, 6), "pnl": round(tier_pnl, 2),
                                "timestamp": now,
                            })
                            logger.info(
                                "PAPER {} HIT: {} {} close {:.6f} @ ${:.2f} PnL=${:.2f} (remaining={:.6f}) [{}]",
                                tier_name, direction.upper(), trade["symbol"],
                                close_qty, current_price, tier_pnl, remaining, trade["id"],
                            )

                            # Move SL to breakeven after TP1
                            if i == 0:
                                trade["stop_loss"] = round(entry, 2)
                                logger.info(
                                    "PAPER SL→BE: {} {} SL moved to ${:.2f} [{}]",
                                    direction.upper(), trade["symbol"], entry, trade["id"],
                                )
                            if sqlite_store:
                                try:
                                    sqlite_store.upsert_paper_trade(trade)
                                except Exception:
                                    pass

                    # ── Check TP3 SuperTrend trailing stop for remaining qty ──
                    all_tiers_hit = all(t.get("hit") for t in tp_tiers)
                    trail_level = trade.get("supertrend_trail", 0)
                    if all_tiers_hit and remaining > 0 and trail_level > 0:
                        trail_hit = False
                        if direction == "long" and current_price <= trail_level:
                            trail_hit = True
                        elif direction == "short" and current_price >= trail_level:
                            trail_hit = True

                        if trail_hit:
                            pnl_per_unit = (current_price - entry) if direction == "long" else (entry - current_price)
                            trail_pnl = pnl_per_unit * remaining
                            trade["realized_pnl"] = round(trade.get("realized_pnl", 0) + trail_pnl, 2)
                            trade["partial_fills"].append({
                                "type": "TP3_trail", "price": round(current_price, 2),
                                "qty": round(remaining, 6), "pnl": round(trail_pnl, 2),
                                "timestamp": now,
                            })
                            trade["remaining_qty"] = 0
                            trade["status"] = "CLOSED"
                            trade["close_price"] = round(current_price, 2)
                            trade["close_reason"] = "tp3_trail"
                            trade["close_time"] = now
                            logger.info(
                                "PAPER TP3 TRAIL: {} {} @ ${:.2f} PnL=${:.2f} [{}]",
                                direction.upper(), trade["symbol"],
                                current_price, trade["realized_pnl"], trade["id"],
                            )
                            if signal_generator is not None:
                                signal_generator.record_trade_result(
                                    trade["symbol"], is_win=(trade["realized_pnl"] > 0),
                                )
                            if sqlite_store:
                                try:
                                    sqlite_store.upsert_paper_trade(trade)
                                except Exception:
                                    pass

                    # Close trade if no remaining qty
                    if trade.get("remaining_qty", 0) <= 0 and trade["status"] == "OPEN":
                        trade["status"] = "CLOSED"
                        trade["close_price"] = round(current_price, 2)
                        trade["close_reason"] = "fully_filled"
                        trade["close_time"] = now
                        if signal_generator is not None:
                            signal_generator.record_trade_result(
                                trade["symbol"], is_win=(trade.get("realized_pnl", 0) > 0),
                            )
                        if sqlite_store:
                            try:
                                sqlite_store.upsert_paper_trade(trade)
                            except Exception:
                                pass

        except asyncio.CancelledError:
            logger.info("Paper SL/TP monitor cancelled")
        except Exception as exc:
            logger.error("Paper SL/TP monitor error: {}", exc)

    # ── /api/auto/toggle — auto-trading toggle ────────────────────────────
    @app.post("/api/auto/toggle")
    async def api_auto_toggle(request: Request) -> dict[str, Any]:
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        if enabled:
            _require_live_auto_activation_allowed(body)
        if signal_generator is not None and hasattr(signal_generator, "set_auto_trading"):
            signal_generator.set_auto_trading(enabled)
            # Apply paper-mode-friendly settings when enabling auto trading
            if enabled and config.paper_mode:
                signal_generator._min_factors = 1
                signal_generator._min_score = 0.10
                signal_generator._min_factor_magnitude = 0.03
                signal_generator._tech_weight = 0.50
                signal_generator._ml_weight = 0.40
                signal_generator._sentiment_weight = 0.00
                signal_generator._macro_weight = 0.02
                signal_generator._news_weight = 0.04
                signal_generator._orderbook_weight = 0.02
                signal_generator._confirmation_tfs = []
                signal_generator._min_signal_interval = 30
        # Store bot config on signal_generator for reference
        if signal_generator is not None:
            signal_generator._bot_config = {
                "strategy": body.get("strategy", "ensemble"),
                "sizing_mode": body.get("sizing_mode", "risk_pct"),
                "risk_per_trade": float(body.get("risk_per_trade", 2)),
                "max_positions": int(body.get("max_positions", 3)),
                "max_drawdown": float(body.get("max_drawdown", 10)),
                "max_leverage": int(body.get("max_leverage", 5)),
                "daily_loss_limit": float(body.get("daily_loss_limit", 500)),
                "trailing_mode": body.get("trailing_mode", "none"),
                "auto_sl_tp": bool(body.get("auto_sl_tp", False)),
            }
        # Start/stop paper feed — use the main event_bus so candles reach the main signal generator
        if config.paper_mode:
            try:
                if enabled:
                    await _start_paper_feed_main(event_bus)
                else:
                    await _stop_paper_feed_main()
            except Exception as exc:
                logger.warning("Paper feed toggle error: {}", exc)
        return {"success": True, "auto_trading_enabled": enabled, "mode": "paper" if config.paper_mode else "live"}

    # ── /api/auto/status — auto-trading status ────────────────────────────
    @app.get("/api/auto/status")
    async def api_auto_status() -> dict[str, Any]:
        enabled = False
        if signal_generator is not None and hasattr(signal_generator, "auto_trading_enabled"):
            enabled = signal_generator.auto_trading_enabled

        exchanges_cfg = config.get_value("exchanges") or {}
        primary_exchange = "binance"
        primary_cfg: dict[str, Any] = {}
        if isinstance(exchanges_cfg, dict):
            for name, ex_cfg in exchanges_cfg.items():
                if isinstance(ex_cfg, dict) and ex_cfg.get("enabled"):
                    primary_exchange = str(name)
                    primary_cfg = ex_cfg
                    break
            if not primary_cfg:
                maybe_cfg = exchanges_cfg.get(primary_exchange, {})
                if isinstance(maybe_cfg, dict):
                    primary_cfg = maybe_cfg

        try:
            from interface.routes.config import get_active_trading_mode
            active_mode = get_active_trading_mode()
        except Exception:
            active_mode = "paper" if config.paper_mode else "live"

        if active_mode == "demo":
            testnet = True
            label = f"{primary_exchange.upper()} DEMO"
        elif active_mode == "live":
            testnet = bool(primary_cfg.get("testnet", False))
            label = f"{primary_exchange.upper()} {'DEMO' if testnet else 'LIVE'}"
        else:
            testnet = False
            label = "PAPER"

        return {
            "enabled": enabled,
            "mode": active_mode,
            "auto_trading_enabled": enabled,
            "paper_mode": config.paper_mode,
            "exchange": primary_exchange,
            "testnet": testnet,
            "label": label,
        }

    # ── /api/mode/toggle — switch between paper and live trading ────────
    @app.post("/api/mode/toggle")
    async def api_mode_toggle(request: Request) -> dict[str, Any]:
        body = await request.json()
        requested_mode = str(body.get("mode", "")).lower()
        if requested_mode not in ("paper", "live"):
            return {"success": False, "error": "mode must be 'paper' or 'live'"}

        if requested_mode == "live":
            if not config.paper_mode:
                return {
                    "success": True,
                    "mode": "live",
                    "paper_mode": config.paper_mode,
                }
            return {
                "success": False,
                "error": (
                    "Cannot switch to live mode via /api/mode/toggle. "
                    "Use /api/config/trading-mode for validated demo/testnet switching."
                ),
                "mode": "paper",
                "paper_mode": config.paper_mode,
            }

        config.paper_mode = True
        try:
            if signal_generator is not None and getattr(signal_generator, "auto_trading_enabled", False):
                await _start_paper_feed_main(event_bus)
        except Exception as exc:
            logger.warning("Paper feed start error during paper switch: {}", exc)
        try:
            config.persist_runtime_overrides()
        except Exception as exc:
            logger.warning("Failed to persist paper mode change: {}", exc)
        logger.info("Trading mode switched to: paper")
        return {
            "success": True,
            "mode": "paper",
            "paper_mode": config.paper_mode,
        }

    # ── /api/trades/history — closed trade history ────────────────────────
    @app.get("/api/trades/history")
    async def api_trades_history() -> dict[str, Any]:
        trades: list[dict[str, Any]] = []
        if order_manager is not None:
            filled = order_manager.get_filled_orders()
            for o in filled[-50:]:
                trades.append({
                    "time": o.filled_at or int(o.created_at * 1000),
                    "symbol": o.symbol,
                    "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                    "price": o.average_fill_price or o.price or 0,
                    "size": o.cumulative_quantity or o.quantity,
                    "pnl": float(o.metadata.get("pnl", 0)),
                })
        # Include paper trades from standalone paper executor
        if _FASTAPI and '_paper_trades' in globals():
            for pt in _paper_trades[-50:]:
                trades.append({
                    "time": pt.get("timestamp", 0) * 1000,
                    "symbol": pt.get("symbol", ""),
                    "side": pt.get("direction", ""),
                    "price": pt.get("price", 0),
                    "size": pt.get("quantity", 0),
                    "pnl": 0,
                    "paper": True,
                    "score": pt.get("score", 0),
                })
        return {"trades": trades}

    # ── /api/paper/trades — paper trade log ───────────────────────────────
    @app.get("/api/paper/trades")
    async def api_paper_trades() -> dict[str, Any]:
        all_trades = list(_paper_trades_main[-100:])
        if '_paper_trades' in globals():
            all_trades.extend(_paper_trades[-100:])
        all_trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
        open_trades = [t for t in _paper_trades_main if t.get("status") == "OPEN"]
        closed_trades = [t for t in _paper_trades_main if t.get("status") == "CLOSED"]
        total_pnl = sum(t.get("realized_pnl", 0) for t in _paper_trades_main)
        wins = sum(1 for t in closed_trades if t.get("realized_pnl", 0) > 0)
        losses = sum(1 for t in closed_trades if t.get("realized_pnl", 0) <= 0)
        return {
            "trades": all_trades[:100],
            "count": len(all_trades),
            "open_count": len(open_trades),
            "closed_count": len(closed_trades),
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
        }

    # ── /api/config — GET config for settings modal ───────────────────────
    @app.get("/api/config")
    async def api_config_get() -> dict[str, Any]:
        exchanges = config.get_value("exchanges") or {}
        dex_cfg = config.get_value("dex") or {}
        notifications_cfg = config.get_value("notifications", "telegram") or {}
        risk_cfg = config.get_value("risk") or {}
        ai_cfg = config.get_value("ai_agent") or {}
        auto_enabled = False
        if signal_generator and hasattr(signal_generator, "auto_trading_enabled"):
            auto_enabled = signal_generator.auto_trading_enabled

        agent_status: dict[str, Any] = {}
        if signal_generator and hasattr(signal_generator, "get_agent_status"):
            try:
                agent_status = signal_generator.get_agent_status() or {}
            except Exception:
                agent_status = {}

        # Build connection registry
        registry = []
        for name in ["binance", "bybit", "okx", "kraken"]:
            ex = exchanges.get(name, {})
            has_creds = bool(ex.get("api_key")) and bool(ex.get("api_secret"))
            registry.append({
                "exchange": name,
                "venue_type": "CEX",
                "connected": has_creds and ex.get("enabled", False),
                "status": "connected" if (has_creds and ex.get("enabled", False)) else "disconnected",
            })

        def _mask(val: str) -> str:
            s = str(val or "")
            return "****" if s else ""

        def _get_ai_key(cfg: dict[str, Any]) -> str:
            provider = str(cfg.get("provider", "local") or "local").strip().lower()
            env_key = {
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }.get(provider, "")
            return str(cfg.get("api_key", os.getenv(env_key, ""))) if env_key else str(cfg.get("api_key", ""))

        binance = exchanges.get("binance", {})
        bybit = exchanges.get("bybit", {})
        okx = exchanges.get("okx", {})
        kraken = exchanges.get("kraken", {})

        return {
            "binance_api_key": _mask(binance.get("api_key", "")),
            "binance_secret": _mask(binance.get("api_secret", "")),
            "bybit_api_key": _mask(bybit.get("api_key", "")),
            "bybit_secret": _mask(bybit.get("api_secret", "")),
            "okx_api_key": _mask(okx.get("api_key", "")),
            "okx_secret": _mask(okx.get("api_secret", "")),
            "okx_passphrase": _mask(okx.get("passphrase", "")),
            "kraken_api_key": _mask(kraken.get("api_key", "")),
            "kraken_secret": _mask(kraken.get("api_secret", "")),
            "dex_rpc_url": str(dex_cfg.get("rpc_url", "")),
            "dex_private_key": _mask(dex_cfg.get("private_key", "")),
            "telegram_bot_token": _mask(notifications_cfg.get("bot_token", "")),
            "telegram_chat_id": str(notifications_cfg.get("chat_id", "")),
            "binance_enabled": binance.get("enabled", False),
            "bybit_enabled": bybit.get("enabled", False),
            "okx_enabled": okx.get("enabled", False),
            "kraken_enabled": kraken.get("enabled", False),
            "uniswap_v3_enabled": dex_cfg.get("uniswap", {}).get("enabled", False),
            "sushiswap_enabled": dex_cfg.get("sushiswap", {}).get("enabled", False),
            "dydx_enabled": dex_cfg.get("dydx", {}).get("enabled", False),
            "trade_alerts_enabled": True,
            "risk_alerts_enabled": True,
            "daily_summary_enabled": False,
            "auto_trading_enabled": auto_enabled,
            "auto_stop_loss_enabled": bool(risk_cfg.get("stop_loss_pct")),
            "auto_take_profit_enabled": bool(risk_cfg.get("take_profit_pct")),
            "trailing_stops_enabled": bool(risk_cfg.get("trailing_stop", {}).get("enabled")),
            "atr_position_sizing_enabled": bool(risk_cfg.get("atr_stop", {}).get("enabled")),
            "ai_agent_enabled": bool(agent_status.get("enabled", ai_cfg.get("enabled", True))),
            "ai_agent_provider": str(agent_status.get("provider", ai_cfg.get("provider", "local"))),
            "ai_agent_model": str(agent_status.get("model", ai_cfg.get("model", "claude-sonnet-4-6"))),
            "ai_agent_api_key": _mask(_get_ai_key(ai_cfg)),
            "ai_agent_timeout_seconds": float(agent_status.get("timeout_seconds", ai_cfg.get("timeout_seconds", 8.0)) or 8.0),
            "ai_agent_remote_weight": float(ai_cfg.get("remote_weight", 0.35) or 0.35),
            "ai_agent_remote_enabled": bool(agent_status.get("remote_enabled", False)),
            "connection_registry": registry,
        }

    # ── /api/config POST — save settings ──────────────────────────────────
    @app.post("/api/config")
    async def api_config_post(request: Request) -> dict[str, Any]:
        body = await request.json()

        # ── Apply CEX API keys to runtime config ──
        exchanges = config._data.setdefault("exchanges", {})
        changed_venues: set[str] = set()
        for venue, key_field, sec_field in [
            ("binance", "binance_api_key", "binance_secret"),
            ("bybit", "bybit_api_key", "bybit_secret"),
            ("okx", "okx_api_key", "okx_secret"),
            ("kraken", "kraken_api_key", "kraken_secret"),
        ]:
            venue_cfg = exchanges.setdefault(venue, {})
            key_val = body.get(key_field, "")
            sec_val = body.get(sec_field, "")
            # Only overwrite if user provided a real value (not masked ****)
            if key_val and key_val != "****":
                venue_cfg["api_key"] = key_val
                os.environ[f"{venue.upper()}_API_KEY"] = str(key_val)
                changed_venues.add(venue)
            if sec_val and sec_val != "****":
                venue_cfg["api_secret"] = sec_val
                os.environ[f"{venue.upper()}_API_SECRET"] = str(sec_val)
                changed_venues.add(venue)
            # Venue enable/disable
            enabled_key = f"{venue}_enabled"
            if enabled_key in body:
                venue_cfg["enabled"] = bool(body[enabled_key])
                changed_venues.add(venue)

        if executors:
            executor_map = {getattr(executor, "exchange_id", ""): executor for executor in executors}
            for venue in changed_venues:
                executor = executor_map.get(venue)
                if executor is None:
                    continue
                client = getattr(executor, "_client", None)
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        pass
                executor._client = None
                executor._order_placer = None
                venue_cfg = exchanges.get(venue, {})
                if venue_cfg.get("enabled", False):
                    try:
                        await executor._init_client()
                    except Exception as exc:
                        logger.warning("{} client refresh failed after settings update: {}", venue, exc)

        # ── DEX config ──
        dex_cfg = config._data.setdefault("dex", {})
        rpc_val = body.get("dex_rpc_url", "")
        wallet_val = body.get("dex_private_key", "")
        if rpc_val:
            dex_cfg["rpc_url"] = rpc_val
        if wallet_val and wallet_val != "****":
            dex_cfg["private_key"] = wallet_val

        # ── Telegram notifications ──
        notif = config._data.setdefault("notifications", {}).setdefault("telegram", {})
        tg_token = body.get("telegram_bot_token", "")  # noqa: S105 — user-supplied, not hardcoded
        tg_chat = body.get("telegram_chat_id", "")
        if tg_token and tg_token != "****":  # noqa: S105 — sentinel mask, not a secret
            notif["bot_token"] = tg_token
        if tg_chat:
            notif["chat_id"] = tg_chat

        # ── AI agent config ──
        ai_cfg = config._data.setdefault("ai_agent", {})
        ai_cfg["enabled"] = bool(body.get("ai_agent_enabled", ai_cfg.get("enabled", True)))
        ai_cfg["provider"] = str(body.get("ai_agent_provider", ai_cfg.get("provider", "local")) or "local")
        default_model = "gpt-4o-mini" if ai_cfg["provider"] == "openai" else "claude-sonnet-4-6"
        ai_cfg["model"] = str(body.get("ai_agent_model", ai_cfg.get("model", default_model)) or default_model)
        key_val = body.get("ai_agent_api_key", "")
        if key_val and key_val != "****":
            ai_cfg["api_key"] = key_val
            env_key = {
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }.get(str(ai_cfg["provider"]).strip().lower(), "")
            if env_key:
                os.environ[env_key] = str(key_val)
        ai_cfg["timeout_seconds"] = float(body.get("ai_agent_timeout_seconds", ai_cfg.get("timeout_seconds", 8.0)) or 8.0)
        ai_cfg["remote_weight"] = float(body.get("ai_agent_remote_weight", ai_cfg.get("remote_weight", 0.35)) or 0.35)

        # ── Auto-trading toggle ──
        if signal_generator and hasattr(signal_generator, "set_auto_trading") and "auto_trading_enabled" in body:
            auto_enabled = bool(body.get("auto_trading_enabled"))
            currently_enabled = bool(getattr(signal_generator, "auto_trading_enabled", False))
            if auto_enabled and not currently_enabled:
                _require_live_auto_activation_allowed(body)
            signal_generator.set_auto_trading(auto_enabled)
        if signal_generator and hasattr(signal_generator, "configure_agent"):
            signal_generator.configure_agent(payload={
                "enabled": ai_cfg.get("enabled", True),
                "provider": ai_cfg.get("provider", "local"),
                "model": ai_cfg.get("model", "claude-sonnet-4-6"),
                "api_key": ai_cfg.get("api_key", ""),
                "timeout_seconds": ai_cfg.get("timeout_seconds", 8.0),
                "remote_weight": ai_cfg.get("remote_weight", 0.35),
            })

        # ── Build updated connection registry ──
        registry = []
        for name in ["binance", "bybit", "okx", "kraken"]:
            ex = exchanges.get(name, {})
            has_creds = bool(ex.get("api_key")) and bool(ex.get("api_secret"))
            registry.append({
                "exchange": name,
                "venue_type": "CEX",
                "connected": has_creds and ex.get("enabled", False),
                "status": "connected" if (has_creds and ex.get("enabled", False)) else "disconnected",
            })

        try:
            config.persist_runtime_overrides()
        except Exception as exc:
            logger.warning("Failed to persist runtime settings: {}", exc)

        logger.info("Settings saved to runtime config")
        return {
            "success": True,
            "message": "Settings applied to runtime",
            "connection_registry": registry,
            "ai_agent": {
                "enabled": ai_cfg.get("enabled", True),
                "provider": ai_cfg.get("provider", "local"),
                "model": ai_cfg.get("model", "claude-sonnet-4-6"),
                "remote_weight": ai_cfg.get("remote_weight", 0.35),
                "api_configured": bool(ai_cfg.get("api_key", "")),
            },
        }

    # ── /api/config/test — test exchange connections ──────────────────────
    @app.post("/api/config/test")
    async def api_config_test() -> dict[str, Any]:
        exchanges_cfg = config.get_value("exchanges") or {}
        dex_cfg = config.get_value("dex") or {}
        enabled = {}
        cex_creds = {}
        for name in ["binance", "bybit", "okx", "kraken"]:
            ex = exchanges_cfg.get(name, {})
            enabled[name] = ex.get("enabled", False)
            cex_creds[name] = bool(ex.get("api_key")) and bool(ex.get("api_secret"))

        registry = []
        for name in ["binance", "bybit", "okx", "kraken"]:
            ex = exchanges_cfg.get(name, {})
            has_creds = bool(ex.get("api_key")) and bool(ex.get("api_secret"))
            registry.append({
                "exchange": name,
                "venue_type": "CEX",
                "connected": has_creds and ex.get("enabled", False),
                "status": "connected" if (has_creds and ex.get("enabled", False)) else "disconnected",
            })

        return {
            "success": True,
            "connection_registry": registry,
            "checks": {
                "grpc": False,
                "clickhouse": False,
                "credentials_present": any(cex_creds.values()),
                "cex_credentials": cex_creds,
                "dex_credentials": {
                    "rpc_url": bool(dex_cfg.get("rpc_url")),
                    "private_key": bool(dex_cfg.get("private_key")),
                },
                "enabled": enabled,
            },
        }

    # ── /api/config/test-keys — preflight validation (real ccxt call) ────
    @app.post("/api/config/test-keys")
    async def api_config_test_keys(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        venue = str(body.get("venue") or body.get("exchange") or "binance").lower().strip()
        api_key = str(body.get("api_key") or "").strip()
        api_secret = str(body.get("api_secret") or "").strip()
        testnet = bool(body.get("testnet", False))
        # `type` selects the CCXT product line. For Binance "demo account" keys
        # (testnet.binancefuture.com) this MUST be "future" — those keys do
        # not exist on spot testnet and will return -2015 if probed there.
        # Default to whatever the venue's settings.yaml says, falling back
        # to "future" for binance/bybit/okx since that's what this bot trades.
        body_type = str(body.get("type") or body.get("market") or "").lower().strip()
        if not body_type:
            try:
                cfg_type = str((config.get_value("exchanges", venue) or {}).get("type", "") or "").lower()
            except Exception:
                cfg_type = ""
            body_type = cfg_type or ("future" if venue in {"binance", "bybit", "okx"} else "spot")

        if not api_key or not api_secret:
            return {"success": False, "error": "api_key and api_secret required"}

        try:
            import ccxt.async_support as ccxt_async  # type: ignore[import-not-found]
        except Exception as exc:
            return {"success": False, "error": f"ccxt not installed: {exc}"}

        cls = getattr(ccxt_async, venue, None)
        if cls is None:
            return {"success": False, "error": f"unknown venue: {venue}"}

        params: dict[str, Any] = {
            "apiKey": api_key, "secret": api_secret, "enableRateLimit": True,
            "options": {"defaultType": "future" if body_type in {"future", "futures", "perp", "swap"} else body_type or "spot"},
        }
        if venue == "okx":
            passphrase = body.get("passphrase") or body.get("api_password") or ""
            if passphrase:
                params["password"] = str(passphrase)
        client = cls(params)
        # Sandbox mode: two distinct binance environments — opt in via the request body.
        #   demo=true       → demo.binance.com keys, ccxt's enable_demo_trading()  ← CCXT announcement #92 path
        #   testnet=true    → testnet.binancefuture.com keys, classic urls['test']
        # If neither, talks to mainnet.
        demo = bool(body.get("demo", False))
        if venue == "binance" and demo:
            try:
                # Official ccxt path — swaps urls['api'] for urls['demo'] which
                # includes BOTH spot (demo-api.binance.com) and futures
                # (demo-fapi.binance.com). A manual fapi-only swap is insufficient
                # because ccxt's load_markets() / fetch_balance() also issues spot
                # (sapi) requests that need to be redirected.
                if hasattr(client, "enable_demo_trading"):
                    client.enable_demo_trading(True)
                else:
                    # Older ccxt that lacks the helper — fall back to manual swap
                    api_urls = client.urls.get("api", {})
                    demo_urls = client.urls.get("demo", {})
                    for k, v in demo_urls.items():
                        if k in api_urls:
                            api_urls[k] = v
                    client.urls["api"] = api_urls
            except Exception as exc:
                logger.debug("test-keys: demo mode setup failed: {}", exc)
        elif testnet:
            # CCXT removed set_sandbox_mode for binance futures (announcement #92).
            # Swap fapi/dapi entries from urls['test'] into urls['api']. Falls
            # back to set_sandbox_mode for other venues that still support it.
            try:
                test_urls = client.urls.get("test", {}) if isinstance(getattr(client, "urls", {}), dict) else {}
                api_urls = client.urls.get("api", {}) if isinstance(getattr(client, "urls", {}), dict) else {}
                swapped = 0
                for k, v in test_urls.items():
                    if k.startswith(("fapi", "dapi")) and k in api_urls:
                        api_urls[k] = v
                        swapped += 1
                if swapped == 0 and hasattr(client, "set_sandbox_mode"):
                    client.set_sandbox_mode(True)
            except Exception as exc:
                logger.debug("test-keys: testnet URL swap failed: {}", exc)
                if hasattr(client, "set_sandbox_mode"):
                    try: client.set_sandbox_mode(True)
                    except Exception: pass

        balances_summary: list[dict[str, Any]] = []
        try:
            bal = await asyncio.wait_for(client.fetch_balance(), timeout=20.0)
            totals = (bal or {}).get("total") or {}
            for ccy, amt in sorted(totals.items(), key=lambda kv: -float(kv[1] or 0))[:10]:
                try:
                    amtf = float(amt)
                except Exception:
                    amtf = 0.0
                if amtf > 0:
                    balances_summary.append({"asset": ccy, "total": amtf})
            logger.info(
                "test-keys OK venue={} key={}*** balances_nonzero={}",
                venue, api_key[:4], len(balances_summary),
            )
            return {"success": True, "venue": venue, "testnet": testnet,
                    "balances": balances_summary, "balance_count": len(balances_summary)}
        except asyncio.TimeoutError:
            return {"success": False, "error": "exchange timeout (8s)"}
        except Exception as exc:
            msg = sanitize_exception(exc)[:200]
            logger.warning("test-keys FAIL venue={} key={}***: {}", venue, api_key[:4], msg)
            return {"success": False, "error": f"{type(exc).__name__}: {msg}"}
        finally:
            try:
                await client.close()
            except Exception:
                pass

    # ── /api/realtime/snapshot — polling fallback ─────────────────────────
    @app.get("/api/realtime/snapshot")
    async def api_realtime_snapshot(
        symbol: str = Query("BTC/USDT"),
        timeframe: str = Query("1m"),
    ) -> dict[str, Any]:
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})
        # Assemble a full snapshot from all data sources
        status = (await api_status())
        fg = (await api_feargreed())
        ob = (await api_orderbook(symbol=symbol, depth=20))
        sym_base = symbol.split("/")[0] if "/" in symbol else symbol
        ind = (await api_indicators(sym_base))
        news = (await api_news())
        auto = (await api_auto_status())
        candles = (await api_candles(symbol=symbol, timeframe=timeframe))
        market = (await api_market(per_page=250))
        dex = (await api_dex_pools())
        recon = (await api_reconciliation_status())
        ustream = (await api_user_stream_status())
        return {
            "status": status,
            "feargreed": fg,
            "orderbook": ob,
            "indicators": ind,
            "news": news,
            "auto": auto,
            "candles": candles,
            "market": market,
            "dex": dex,
            "reconciliation": recon,
            "user_stream": ustream,
        }

    # ── /api/realtime/stream — SSE streaming ──────────────────────────────
    @app.get("/api/realtime/stream")
    async def api_realtime_stream(
        request: Request,
        symbol: str = Query("BTC/USDT"),
        timeframe: str = Query("1m"),
    ):
        try:
            symbol = _validate_symbol(symbol)
            timeframe = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": sanitize_exception(exc)})

        async def _event_generator():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snapshot = await api_realtime_snapshot(symbol=symbol, timeframe=timeframe)
                    yield {"event": "snapshot", "data": json.dumps(snapshot)}
                except Exception as exc:
                    logger.debug("SSE snapshot error: {}", exc)
                await asyncio.sleep(3)

        try:
            return EventSourceResponse(_event_generator())
        except Exception:
            # Fallback if sse_starlette is not installed
            from starlette.responses import StreamingResponse

            async def _sse_fallback():
                while True:
                    try:
                        snapshot = await api_realtime_snapshot(symbol=symbol, timeframe=timeframe)
                        yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
                    except Exception:
                        pass
                    await asyncio.sleep(3)

            return StreamingResponse(_sse_fallback(), media_type="text/event-stream")

    @app.get("/api/market-data/health")
    async def api_market_data_health() -> dict[str, Any]:
        """Return L0 market-data integrity status for dashboard and preflight."""
        monitor = market_data_integrity
        if monitor is not None and hasattr(monitor, "snapshot"):
            try:
                return monitor.snapshot()
            except Exception as exc:
                return {
                    "enabled": False,
                    "status": "ERROR",
                    "healthy": False,
                    "reason": sanitize_exception(exc),
                    "feeds": [],
                }
        if signal_generator is not None and hasattr(signal_generator, "get_l0_status"):
            payload = signal_generator.get_l0_status()
            if isinstance(payload, dict):
                return payload
        return {
            "enabled": False,
            "status": "UNAVAILABLE",
            "healthy": False,
            "reason": "market_data_integrity_monitor_not_wired",
            "feeds": [],
        }

    # ── §3 Spec: L0-L9 confirmation status plus optional L10 context ───────
    async def _build_layers_response() -> dict[str, Any]:
        """Heavy path — runs the full pipeline preview. Called only when
        the cache is cold or being refreshed; never inline on the request path
        without a wait_for timeout guarding it."""
        layers = _build_layers_skeleton()
        await _populate_layers_status(layers)
        return _finalize_layers_response(layers)

    async def _layers_background_refresh() -> None:
        """Fire-and-forget refresh — runs to completion regardless of how long
        get_quality_preview takes, so the next request gets hot data."""
        try:
            response = await _build_layers_response()
            _layers_cache["_"] = (time.monotonic(), response)
        except Exception as exc:
            logger.debug("layers background refresh failed: {}", exc)

    @app.get("/api/layers")
    async def api_layers() -> dict[str, Any]:
        """Return current state of the L0-L9/L10 confirmation pipeline.

        Uses stale-while-revalidate so the dashboard never blocks on a slow
        build. See the cache constants above for the budgets."""
        global _layers_refresh_lock
        if _layers_refresh_lock is None:
            _layers_refresh_lock = asyncio.Lock()

        cached = _layers_cache.get("_")
        now = time.monotonic()
        # Fresh hit
        if cached and (now - cached[0]) < _LAYERS_CACHE_TTL:
            return cached[1]
        # Stale hit — kick off a refresh in the background, return stale
        if cached and (now - cached[0]) < _LAYERS_CACHE_STALE:
            if not _layers_refresh_lock.locked():
                async def _bg():
                    async with _layers_refresh_lock:
                        await _layers_background_refresh()
                asyncio.create_task(_bg(), name="layers_bg_refresh")
            stale_payload = dict(cached[1])
            stale_payload["stale"] = True
            stale_payload["age_seconds"] = round(now - cached[0], 2)
            return stale_payload
        # Total miss — race the build against COLD_BUDGET. If we lose, return
        # placeholder; the build keeps running via fire-and-forget.
        if _layers_refresh_lock.locked():
            # Another caller is already building — wait briefly for them.
            try:
                await asyncio.wait_for(_layers_refresh_lock.acquire(), timeout=_LAYERS_COLD_BUDGET)
                _layers_refresh_lock.release()
                cached2 = _layers_cache.get("_")
                if cached2:
                    return cached2[1]
            except asyncio.TimeoutError:
                pass
            return _layers_placeholder_response("warming up — pipeline build in progress")
        # We're the first — try to build inline within the budget.
        async with _layers_refresh_lock:
            try:
                response = await asyncio.wait_for(_build_layers_response(), timeout=_LAYERS_COLD_BUDGET)
                _layers_cache["_"] = (time.monotonic(), response)
                return response
            except asyncio.TimeoutError:
                # Drop the inline wait, but keep building via background task
                # so the next caller gets hot data.
                asyncio.create_task(_layers_background_refresh(), name="layers_bg_refresh_fallback")
                return _layers_placeholder_response("pipeline build exceeded cold budget — refreshing in background")

    _LAYERS_DEFAULT_THRESHOLDS = {
        "market_data_integrity": 1.0,
        "session_filter": 1.0,
        "htf_trend": 3.0,
        "technical_confluence": 30.0,
        "smart_money_concepts": 0.0,
        "volume_flow": 20.0,
        "regime_detection": 1.0,
        "ml_ensemble": 0.5,
        "signal_quality": 65.0,
        "risk_gate": 1.0,
    }

    def _build_layers_skeleton() -> list[dict[str, Any]]:
        return [
            {"id": 0, "layer_index": "L0", "name": "Market Data Integrity", "description": "Freshness, sequence-gap, clock-drift, and orderbook sanity gate"},
            {"id": 1, "layer_index": "L1", "name": "Session Filter", "description": "Trading session & killzone enforcement"},
            {"id": 2, "layer_index": "L2", "name": "HTF Trend", "description": "Higher-timeframe weighted agreement"},
            {"id": 3, "layer_index": "L3", "name": "Technical Confluence", "description": "RSI, MACD, BB, EMA alignment"},
            {"id": 4, "layer_index": "L4", "name": "Smart Money Concepts", "description": "BOS/CHoCH + OB/FVG zones"},
            {"id": 5, "layer_index": "L5", "name": "Volume Flow", "description": "Delta, CVD, VWAP deviation"},
            {"id": 6, "layer_index": "L6", "name": "Regime Detection", "description": "Market regime (trending/ranging/breakout)"},
            {"id": 7, "layer_index": "L7", "name": "ML Ensemble", "description": "Model prediction confidence"},
            {"id": 8, "layer_index": "L8", "name": "Signal Quality", "description": "0-100 quality score gate (min 65)"},
            {"id": 9, "layer_index": "L9", "name": "Risk Gate", "description": "Position sizing, DD phase, circuit breaker"},
        ]

    def _build_preflight_status(reason: str | None = None) -> dict[str, Any]:
        if reason:
            raw_status = "PENDING"
            detail = reason
        elif signal_generator is None:
            raw_status = "UNKNOWN"
            detail = "signal generator unavailable"
        else:
            last = getattr(signal_generator, "_last_layer_status", {}) or {}
            raw_status = str(last.get("preflight", "PENDING") or "PENDING").upper()
            detail = str(last.get("preflight_detail", "") or "awaiting preflight")
            if raw_status in ("UNKNOWN", "PENDING"):
                estop = getattr(signal_generator, "_estop", None)
                if estop is not None and bool(getattr(estop, "is_active", False)):
                    raw_status = "FAIL"
                    detail = f"ESTOP:{getattr(estop, 'reason', 'active')}"
                else:
                    raw_status = "PASS"
                    detail = "runtime guards ready"

        if raw_status == "BLOCKED":
            raw_status = "FAIL"
        score = 100.0 if raw_status == "PASS" else 0.0 if raw_status == "FAIL" else None
        return {
            "id": "preflight",
            "layer_index": "PRE",
            "name": "Preflight",
            "description": "Runtime guards before L1-L10",
            "status": raw_status,
            "detail": detail,
            "score": score,
            "threshold": 1.0,
        }

    def _layers_score_to_status(score: float | int | None) -> str:
        try:
            numeric = float(score if score is not None else 0)
        except (TypeError, ValueError):
            return "UNKNOWN"
        if numeric >= 70:
            return "PASS"
        if numeric >= 40:
            return "WEAK"
        return "FAIL"

    def _layers_parse_score_threshold(detail: str) -> tuple[float | None, float | None]:
        """Extract numeric score + threshold from detail strings like
        'score=42<30', 'score=0.82', 'score=BLOCKED'."""
        import re as _re
        if not detail:
            return None, None
        m = _re.search(r"score\s*=\s*(-?[\d.]+)\s*(?:[<>]=?)?\s*(-?[\d.]+)?", detail)
        if not m:
            return None, None
        try:
            score = float(m.group(1))
        except (TypeError, ValueError):
            score = None
        thr: float | None = None
        if m.group(2) is not None:
            try:
                thr = float(m.group(2))
            except (TypeError, ValueError):
                thr = None
        return score, thr

    def _layers_is_stale_detail(detail: str) -> bool:
        text = str(detail or "").strip().lower()
        if not text:
            return True
        return any(token in text for token in (
            "waiting for first evaluation",
            "waiting for first market-data heartbeat",
            "awaiting evaluation",
            "pipeline build exceeded",
            "warming up",
        ))

    async def _populate_layers_status(layers: list[dict[str, Any]]) -> None:
        if signal_generator is None:
            for layer in layers:
                key = layer["name"].lower().replace(" ", "_")
                layer["status"] = "UNKNOWN"
                layer["detail"] = "signal generator unavailable"
                layer["score"] = None
                layer["threshold"] = _LAYERS_DEFAULT_THRESHOLDS.get(key)
            return

        last = getattr(signal_generator, "_last_layer_status", {}) or {}
        quality = getattr(signal_generator, "_last_quality_breakdown", {}) or {}
        if (int(quality.get("total", 0) or 0) <= 0) and hasattr(signal_generator, "get_quality_preview"):
            # Heavy call — only the build path reaches here; the request path
            # has already gated this with wait_for(_LAYERS_COLD_BUDGET).
            preview = await asyncio.to_thread(signal_generator.get_quality_preview)
            if isinstance(preview, dict) and preview.get("components"):
                quality = {**quality, **preview}
        components = quality.get("components", {}) or {}
        fallback_scores = {
            "session_filter": 100 if not (quality.get("rejected_at") == "L1_Session") else 0,
            "htf_trend": components.get("htf_alignment", components.get("htf_trend", 0)),
            "technical_confluence": components.get("technical_confluence", 0),
            "smart_money_concepts": components.get("smc_confluence", 0),
            "volume_flow": components.get("volume_flow", 0),
            "regime_detection": components.get("regime", 0),
            "ml_ensemble": components.get("ml_confidence", 0),
            "signal_quality": quality.get("total", 0),
        }
        for layer in layers:
            key = layer["name"].lower().replace(" ", "_")
            raw_status = str(last.get(key, "UNKNOWN") or "UNKNOWN").upper()
            detail = str(last.get(f"{key}_detail", "") or "")

            if key == "market_data_integrity":
                l0 = await api_market_data_health()
                l0_status = str(l0.get("status", "UNKNOWN") or "UNKNOWN").upper()
                if l0_status == "OK":
                    raw_status = "PASS"
                elif l0_status in {"DEGRADED", "WARMING"}:
                    raw_status = "WEAK" if l0_status == "DEGRADED" else "PENDING"
                elif l0_status in {"BLOCKED", "ERROR"}:
                    raw_status = "FAIL"
                elif raw_status == "UNKNOWN":
                    raw_status = "PENDING"
                feeds = int(l0.get("feed_count", 0) or 0)
                healthy = int(l0.get("healthy_count", 0) or 0)
                if _layers_is_stale_detail(detail):
                    detail = f"{l0_status.lower()} feeds={healthy}/{feeds}"
                layer["status"] = raw_status
                layer["detail"] = detail
                layer["score"] = 100.0 if raw_status == "PASS" else 50.0 if raw_status == "WEAK" else 0.0 if raw_status == "FAIL" else None
                layer["threshold"] = _LAYERS_DEFAULT_THRESHOLDS.get(key)
                layer["health"] = l0
                continue

            if raw_status in ("UNKNOWN", "PENDING") and key in fallback_scores:
                raw_status = _layers_score_to_status(fallback_scores.get(key))
                if key == "ml_ensemble" and raw_status == "FAIL" and float(fallback_scores.get(key, 0) or 0) > 0:
                    raw_status = "WEAK"
                if _layers_is_stale_detail(detail):
                    detail = f"score={fallback_scores.get(key, 0)}"

            if key == "risk_gate" and raw_status in ("UNKNOWN", "PENDING"):
                snap = risk_manager.get_risk_snapshot() if risk_manager is not None else {}
                blocked = bool(snap.get("kill_switch_active", False) or snap.get("circuit_breaker_tripped", False))
                raw_status = "FAIL" if blocked else "PASS"
                if _layers_is_stale_detail(detail):
                    detail = "risk controls ready" if not blocked else "risk controls blocking new trades"

            if raw_status == "BLOCKED":
                raw_status = "FAIL"
            if raw_status == "UNKNOWN" and "score=0" in detail:
                raw_status = "FAIL"

            score, thr = _layers_parse_score_threshold(detail)
            if score is None:
                fs = fallback_scores.get(key)
                if fs is not None:
                    try:
                        score = float(fs)
                    except (TypeError, ValueError):
                        score = None
            if thr is None:
                thr = _LAYERS_DEFAULT_THRESHOLDS.get(key)

            layer["status"] = raw_status
            layer["detail"] = detail or "awaiting evaluation"
            layer["score"] = score
            layer["threshold"] = thr

    def _finalize_layers_response(layers: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"pass": 0, "soft": 0, "fail": 0, "skip": 0, "other": 0}
        for lay in layers:
            st = str(lay.get("status", "")).upper()
            if st == "PASS":
                counts["pass"] += 1
            elif st in ("WEAK", "SOFT", "WARN"):
                counts["soft"] += 1
            elif st in ("FAIL", "BLOCKED"):
                counts["fail"] += 1
            elif st in ("SKIP", "PENDING", "UNKNOWN", ""):
                counts["skip"] += 1
            else:
                counts["other"] += 1
        return {
            "preflight": _build_preflight_status(),
            "layers": layers,
            "total": len(layers),
            "counts": counts,
            "paper_mode": bool(getattr(signal_generator, "_is_paper_mode", lambda: True)()) if signal_generator is not None else True,
        }

    def _layers_placeholder_response(reason: str) -> dict[str, Any]:
        """Fallback when no cache exists and the cold build is taking too long.
        Returns the layer skeleton with PENDING status so the UI still draws."""
        layers = _build_layers_skeleton()
        for layer in layers:
            key = layer["name"].lower().replace(" ", "_")
            layer["status"] = "PENDING"
            layer["detail"] = reason
            layer["score"] = None
            layer["threshold"] = _LAYERS_DEFAULT_THRESHOLDS.get(key)
        out = _finalize_layers_response(layers)
        out["preflight"] = _build_preflight_status(reason)
        out["pending"] = True
        out["reason"] = reason
        return out

    # ── §5 Spec: Session & killzone status ────────────────────────────────
    @app.get("/api/session")
    async def api_session() -> dict[str, Any]:
        """Return current trading session, killzone status, and session clock."""
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        hour = now.hour
        # Session rules (mirror signal_generator._SESSION_RULES)
        sessions = [
            {"name": "asia", "start": 0, "end": 8, "types": ["B"], "size_mult": 0.5},
            {"name": "london_open", "start": 8, "end": 12, "types": ["A", "C", "D"], "size_mult": 1.0},
            {"name": "london_dead", "start": 12, "end": 13, "types": [], "size_mult": 0.0, "no_trade": True},
            {"name": "london_ny_overlap", "start": 13, "end": 17, "types": ["A", "B", "C", "D"], "size_mult": 1.5},
            {"name": "ny_only", "start": 17, "end": 22, "types": ["B", "D"], "size_mult": 0.75},
            {"name": "low_liquidity", "start": 22, "end": 24, "types": [], "size_mult": 0.0, "no_trade": True},
        ]
        active_session = None
        for s in sessions:
            if s["start"] <= hour < s["end"]:
                active_session = s
                break
        # ICT killzones
        ict_killzones = [{"start": 13, "end": 14}, {"start": 15, "end": 16}]
        in_killzone = any(kz["start"] <= hour < kz["end"] for kz in ict_killzones)
        is_weekend = now.weekday() >= 5
        return {
            "utc_hour": hour,
            "utc_time": now.strftime("%H:%M:%S"),
            "active_session": active_session,
            "in_killzone": in_killzone,
            "is_weekend": is_weekend,
            "sessions": sessions,
            "ict_killzones": ict_killzones,
        }

    # ── ML status & training ───────────────────────────────────────────────
    @app.get("/api/ml/status")
    async def api_ml_status() -> dict[str, Any]:
        """Return ML model load/training status for the live signal engine."""
        if signal_generator is not None and hasattr(signal_generator, "get_ml_status"):
            return signal_generator.get_ml_status()
        return {
            "loaded": False,
            "model_path": "ml_model.pkl",
            "model_type": "lightgbm",
            "feature_count": 0,
            "last_train_ts": 0.0,
            "training": {"trained": False, "reason": "signal_generator_unavailable"},
        }

    @app.post("/api/ml/train")
    async def api_ml_train() -> dict[str, Any]:
        """Trigger immediate ML retraining from the currently cached historical data."""
        if signal_generator is not None and hasattr(signal_generator, "retrain_model_now"):
            return await signal_generator.retrain_model_now()
        return {"trained": False, "reason": "signal_generator_unavailable"}

    @app.get("/api/agent/status")
    async def api_agent_status() -> dict[str, Any]:
        """Return the attached AI agent mode and recent decision state."""
        if signal_generator is not None and hasattr(signal_generator, "get_agent_status"):
            return signal_generator.get_agent_status()
        return {"attached": False, "enabled": False, "mode": "off"}

    @app.post("/api/agent/config")
    async def api_agent_config(request: Request) -> dict[str, Any]:
        """Update AI agent runtime settings."""
        body = await request.json()
        if signal_generator is not None and hasattr(signal_generator, "configure_agent"):
            return signal_generator.configure_agent(body)
        return {"attached": False, "enabled": False, "mode": "off"}

    @app.post("/api/agent/test")
    async def api_agent_test() -> dict[str, Any]:
        """Live-test the configured AI provider with a tiny round-trip prompt."""
        if signal_generator is None:
            return {"success": False, "reason": "agent_unavailable", "message": "AI agent is not attached."}
        try:
            if hasattr(signal_generator, "test_agent_connection"):
                return await signal_generator.test_agent_connection()
            if hasattr(signal_generator, "_ai_agent") and signal_generator._ai_agent is not None:
                return await signal_generator._ai_agent.test_connection()
            return {"success": False, "reason": "agent_unavailable", "message": "AI agent is not attached."}
        except Exception as exc:
            return {"success": False, "reason": "exception", "message": sanitize_exception(exc)[:240]}

    @app.post("/api/agent/chat")
    async def api_agent_chat(request: Request) -> dict[str, Any]:
        """Chat with the trading agent and trigger safe bot interactions."""
        body = await request.json()
        message = str(body.get("message", "") or "").strip()
        if not message:
            return {"success": False, "reply": "Please enter a message."}
        if signal_generator is not None and hasattr(signal_generator, "chat_with_agent"):
            result = signal_generator.chat_with_agent(message)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        return {"success": True, "provider": "system", "reply": "Agent is unavailable right now."}

    @app.get("/api/strategy/suggest")
    async def api_strategy_suggest(symbol: str = Query("BTC/USDT:USDT")) -> dict[str, Any]:
        """Return a live strategy suggestion based on current pipeline state."""
        if signal_generator is not None and hasattr(signal_generator, "get_strategy_suggestion"):
            return signal_generator.get_strategy_suggestion(symbol)
        return {
            "symbol": symbol,
            "action": "wait",
            "strategy": "unavailable",
            "reason": "signal_generator_unavailable",
            "suggestions": [],
        }

    @app.post("/api/quick-action")
    async def api_quick_action(request: Request) -> dict[str, Any]:
        """Execute a safe predefined bot action from the dashboard."""
        body = await request.json()
        action = str(body.get("action", "") or "").strip().lower()
        symbol = str(body.get("symbol", "BTC/USDT:USDT") or "BTC/USDT:USDT")

        if signal_generator is None:
            return {"success": False, "action": action, "reply": "signal_generator_unavailable"}

        if action == "pause_auto" and hasattr(signal_generator, "set_auto_trading"):
            signal_generator.set_auto_trading(False)
            return {"success": True, "action": action, "reply": "Auto trading paused."}
        if action == "resume_auto" and hasattr(signal_generator, "set_auto_trading"):
            _require_live_auto_activation_allowed(body)
            signal_generator.set_auto_trading(True)
            return {"success": True, "action": action, "reply": "Auto trading resumed."}
        if action == "train_model" and hasattr(signal_generator, "retrain_model_now"):
            result = await signal_generator.retrain_model_now()
            return {"success": bool(result.get("trained", False)), "action": action, "reply": "Model retraining completed." if result.get("trained", False) else f"Training did not complete: {result.get('reason', 'unknown')}", "training": result}
        if action == "strategy_suggest" and hasattr(signal_generator, "get_strategy_suggestion"):
            result = signal_generator.get_strategy_suggestion(symbol)
            return {"success": True, "action": action, "reply": result.get("reason", "Strategy suggestion ready."), "suggestion": result}
        if action in {"status", "risk"} and hasattr(signal_generator, "chat_with_agent"):
            result = signal_generator.chat_with_agent(action)
            if asyncio.iscoroutine(result):
                result = await result
            result["action"] = action
            return result

        return {"success": False, "action": action, "reply": "Unknown quick action."}

    # ── §6 Spec: Quality score breakdown ──────────────────────────────────
    @app.get("/api/quality")
    async def api_quality() -> dict[str, Any]:
        """Return last signal quality score breakdown."""
        if signal_generator is not None:
            last_q = getattr(signal_generator, '_last_quality_breakdown', None)
            if last_q:
                payload = dict(last_q)
                components = dict(payload.get("components") or {})
                total = int(payload.get("total", 0) or 0)
                rejected_at = str(payload.get("rejected_at", "") or "")
                if total <= 0 and components:
                    stage_keys = [
                        "htf_trend",
                        "technical_confluence",
                        "smc_confluence",
                        "volume_flow",
                        "regime",
                        "ml_confidence",
                        "liquidity_depth",
                    ]
                    cutoff_map = {
                        "L2": 1,
                        "L3": 2,
                        "L4": 3,
                        "L5": 4,
                        "L6": 5,
                        "L7": 6,
                        "L8": 7,
                    }
                    cutoff = next((v for k, v in cutoff_map.items() if rejected_at.startswith(k)), len(stage_keys))
                    selected = stage_keys[:cutoff]
                    values = [max(0.0, float(components.get(key, 0) or 0)) for key in selected]
                    if values and (rejected_at or any(v > 0 for v in values)):
                        payload["total"] = int(round(sum(values) / len(values)))
                        payload["partial"] = True
                if int(payload.get("total", 0) or 0) <= 0 and hasattr(signal_generator, "get_quality_preview"):
                    preview = signal_generator.get_quality_preview()
                    if isinstance(preview, dict) and (int(preview.get("total", 0) or 0) > 0 or preview.get("reason")):
                        payload = {**payload, **preview}
                # REQ-SIG-009: classify the master score into spec bands so the
                # dashboard can render the 4-band gradient.
                if hasattr(signal_generator, "_master_score_bands") and hasattr(signal_generator, "classify_master_score"):
                    payload["bands"] = signal_generator._master_score_bands()
                    payload["band"] = signal_generator.classify_master_score(payload.get("total", 0) or 0)
                return payload
        return {
            "total": 0,
            "components": {
                "htf_trend": 0,
                "technical_confluence": 0,
                "smc_confluence": 0,
                "volume_flow": 0,
                "regime": 0,
                "ml_confidence": 0,
                "liquidity_depth": 0,
            },
            "min_threshold": 65,
            "boost_threshold": 90,
        }

    # ── §7 Spec: Regime state & transition info ───────────────────────────
    @app.get("/api/regime")
    async def api_regime() -> dict[str, Any]:
        """Return current regime state and transition info per symbol."""
        regimes: dict[str, Any] = {}
        # Regime data lives on data_manager._regimes (keyed as "exchange:symbol")
        if data_manager is not None:
            raw_regimes: dict = getattr(data_manager, '_regimes', {})
            # Map MarketRegime enum values to frontend categories
            _regime_category = {
                "strong_trend_up": "trending", "weak_trend_up": "trending",
                "strong_trend_down": "trending", "weak_trend_down": "trending",
                "compression": "breakout", "range_chop": "ranging",
                "unknown": "unknown",
            }
            _regime_risk = {
                "trending": 0.02, "breakout": 0.025, "ranging": 0.015, "unknown": 0.02,
            }
            for key, state in raw_regimes.items():
                # key is "exchange:symbol" — extract symbol part for display
                sym = key.split(":", 1)[1] if ":" in key else key
                regime_val = str(state.regime.value) if hasattr(state.regime, 'value') else str(state.regime)
                category = _regime_category.get(regime_val, "unknown")
                regimes[sym] = {
                    "regime": category,
                    "regime_raw": regime_val,
                    "confidence": float(state.confidence),
                    "candles_in_state": int(state.candles_in_state),
                    "trend_strength": float(state.trend_slope),
                    "adx": float(state.adx),
                    "in_transition": state.candles_in_state <= 2,
                    "risk_pct": _regime_risk.get(category, 0.02),
                    "tradeable": state.tradeable,
                }
        return {"regimes": regimes}

    # ── §9 Spec: Risk guardrails status ───────────────────────────────────
    @app.get("/api/guardrails")
    async def api_guardrails() -> dict[str, Any]:
        """Return risk guardrail status: circuit breaker, consecutive losses, flash crash, Kelly."""
        result: dict[str, Any] = {
            "circuit_breaker_tripped": False,
            "circuit_breaker_reason": "",
            "consecutive_losses": 0,
            "max_consecutive_losses": 3,
            "flash_crash_tripped": False,
            "kelly_enabled": False,
            "kelly_win_rate": 0.5,
            "kelly_avg_win": 0.0,
            "kelly_avg_loss": 0.0,
        }
        if risk_manager is not None:
            cb = getattr(risk_manager, '_circuit_breaker', None)
            if cb:
                result["circuit_breaker_tripped"] = cb.tripped
                result["circuit_breaker_reason"] = cb.trip_reason
            result["consecutive_losses"] = getattr(risk_manager, '_consecutive_losses', 0)
            result["max_consecutive_losses"] = getattr(risk_manager, '_max_consecutive_losses', 3)
            result["flash_crash_tripped"] = getattr(risk_manager, '_flash_crash_tripped', False)
            result["kelly_enabled"] = getattr(risk_manager, '_kelly_enabled', False)
            result["kelly_win_rate"] = getattr(risk_manager, '_kelly_win_rate', 0.5)
            result["kelly_avg_win"] = getattr(risk_manager, '_kelly_avg_win', 0.0)
            result["kelly_avg_loss"] = getattr(risk_manager, '_kelly_avg_loss', 0.0)
        return result

    # ── /api/arms/* — ARMS tab 3×3 panel data ──────────────────────────────
    def _default_stress_scenarios() -> list[dict[str, Any]]:
        return [
            {"id": "btc_flash_crash_10", "label": "-10% BTC flash crash", "type": "price_shock",
             "symbol_filter": ["BTC/USDT", "BTC/USDT:USDT"], "price_shock_pct": -0.10},
            {"id": "market_correction_25", "label": "-25% market correction", "type": "price_shock",
             "symbol_filter": "*", "price_shock_pct": -0.25},
            {"id": "funding_spike_05", "label": "Funding spike +0.5%", "type": "funding_spike",
             "funding_pct": 0.005, "duration_h": 8},
            {"id": "exchange_outage_1h", "label": "Exchange outage 1h", "type": "outage",
             "duration_h": 1, "outage_slippage_pct": 0.02},
        ]

    def _stress_status(loss_pct: float) -> str:
        a = abs(loss_pct)
        if a < 5.0:
            return "ok"
        if a < 15.0:
            return "warn"
        return "danger"

    def _daily_return_usd(equity: float) -> float:
        """Best-effort rolling-return estimate (USD/day) for recovery calc."""
        if risk_manager is not None and hasattr(risk_manager, "_closed_trades"):
            trades = list(getattr(risk_manager, "_closed_trades", []))[-30:]
            if trades:
                total_pnl = sum(float(t.get("pnl", 0.0)) for t in trades)
                days = max(1.0, len(trades) / 3.0)
                avg = total_pnl / days
                if avg > 0:
                    return avg
        return max(1.0, equity * 0.01)

    @app.get("/api/arms/stress-test")
    async def api_arms_stress_test() -> dict[str, Any]:
        cfg_scenarios = []
        try:
            arms_cfg = config.get_value("arms") or {}
            cfg_scenarios = list(arms_cfg.get("stress_scenarios") or [])
        except Exception:
            cfg_scenarios = []
        scenarios_def = cfg_scenarios if cfg_scenarios else _default_stress_scenarios()

        equity = 0.0
        positions: list[Any] = []
        if risk_manager is not None:
            equity = float(getattr(risk_manager, "_equity", 0.0) or 0.0)
            positions = list(getattr(risk_manager, "_positions", {}).values())

        daily_usd = _daily_return_usd(equity if equity > 0 else 10000.0)

        def _symbol_match(pos_sym: str, filt: Any) -> bool:
            if filt == "*" or filt is None:
                return True
            if isinstance(filt, str):
                return pos_sym == filt or pos_sym.startswith(filt)
            if isinstance(filt, (list, tuple)):
                return any(pos_sym == f or pos_sym.startswith(f) for f in filt)
            return False

        out: list[dict[str, Any]] = []
        for sc in scenarios_def:
            stype = str(sc.get("type", "price_shock"))
            loss_usd = 0.0
            worst_sym = ""
            worst_side = ""
            worst_loss = 0.0

            if stype == "price_shock":
                shock = float(sc.get("price_shock_pct", 0.0))
                for pos in positions:
                    if not _symbol_match(pos.symbol, sc.get("symbol_filter", "*")):
                        continue
                    notional = abs(float(pos.size) * float(pos.current_price or pos.entry_price))
                    direction_mult = 1.0 if pos.direction == "long" else -1.0
                    pos_loss = notional * shock * direction_mult
                    loss_usd += pos_loss
                    if pos_loss < worst_loss:
                        worst_loss = pos_loss
                        worst_sym = pos.symbol
                        worst_side = pos.direction
            elif stype == "funding_spike":
                funding = float(sc.get("funding_pct", 0.0))
                dur_h = float(sc.get("duration_h", 8.0))
                for pos in positions:
                    if ":" not in pos.symbol:  # spot has no funding
                        continue
                    notional = abs(float(pos.size) * float(pos.current_price or pos.entry_price))
                    pos_loss = -notional * funding * (dur_h / 8.0)
                    if pos.direction == "short":
                        pos_loss = -pos_loss
                    loss_usd += pos_loss
                    if pos_loss < worst_loss:
                        worst_loss = pos_loss
                        worst_sym = pos.symbol
                        worst_side = pos.direction
            elif stype == "outage":
                slip = float(sc.get("outage_slippage_pct", 0.02))
                total_notional = sum(
                    abs(float(p.size) * float(p.current_price or p.entry_price)) for p in positions
                )
                loss_usd = -total_notional * slip
                worst_sym = "All positions" if positions else ""
                worst_side = "—"
                worst_loss = loss_usd

            loss_pct = (loss_usd / equity * 100.0) if equity > 0 else 0.0
            recovery_h = 24.0
            if loss_usd < 0 and daily_usd > 0:
                recovery_h = min(720.0, max(1.0, abs(loss_usd) / daily_usd * 24.0))

            out.append({
                "id": sc.get("id", ""),
                "label": sc.get("label", ""),
                "loss_pct": round(loss_pct, 2),
                "loss_usd": round(loss_usd, 2),
                "worst_symbol": worst_sym or ("—" if not positions else ""),
                "worst_side": worst_side.upper() if worst_side else "",
                "recovery_hours": round(recovery_h, 1),
                "status": _stress_status(loss_pct) if positions else "ok",
                "note": "" if positions else "no open positions",
            })

        return {
            "scenarios": out,
            "scenarios_count": len(out),
            "equity": equity,
            "has_positions": len(positions) > 0,
        }

    @app.get("/api/arms/execution")
    async def api_arms_execution() -> dict[str, Any]:
        twap = {"active": 0, "orders": [], "avg_latency_ms": 0, "status": "idle"}
        iceberg = {"active": 0, "orders": [], "reveal_pct": 0.20, "status": "idle"}
        shadow = {"watching": 0, "status": "idle", "stops": []}
        prewarm_on = False

        if order_manager is not None:
            try:
                twap_snap = order_manager.get_twap_snapshot()
                twap["active"] = int(twap_snap.get("active_count", 0))
                twap["orders"] = list(twap_snap.get("orders", {}).values())
                twap["status"] = "active" if twap["active"] > 0 else "armed"
            except Exception as e:
                logger.debug("twap snapshot error: {}", e)
            try:
                ice_snap = order_manager.get_iceberg_snapshot()
                iceberg["active"] = int(ice_snap.get("active_count", 0))
                iceberg["orders"] = list(ice_snap.get("orders", {}).values())
                iceberg["status"] = "active" if iceberg["active"] > 0 else "armed"
            except Exception as e:
                logger.debug("iceberg snapshot error: {}", e)
            try:
                sh_snap = order_manager.get_shadow_sl_snapshot()
                shadow["watching"] = int(sh_snap.get("active_stops", 0) or sh_snap.get("watching", 0) or 0)
                shadow["stops"] = list(sh_snap.get("stops", []) or [])
                shadow["status"] = "monitoring" if shadow["watching"] > 0 else "armed"
            except Exception as e:
                logger.debug("shadow snapshot error: {}", e)

        if metrics and hasattr(metrics, "get_latency_stats"):
            try:
                lat = metrics.get_latency_stats()
                ol = lat.get("order_latency", {}) if isinstance(lat, dict) else {}
                twap["avg_latency_ms"] = int(ol.get("avg_ms", 0) or 0)
                prewarm_on = bool(ol.get("avg_ms", 0) and ol.get("avg_ms", 0) < 500)
            except Exception:
                pass

        try:
            slicing_cfg = config.get_value("execution", "slicing") or {}
            reveal_pct = float((slicing_cfg.get("iceberg") or {}).get("default_reveal_pct", 0.20))
            iceberg["reveal_pct"] = reveal_pct
        except Exception:
            pass

        return {"twap": twap, "iceberg": iceberg, "shadow_sl": shadow, "prewarm_on": prewarm_on}

    @app.get("/api/arms/weights")
    async def api_arms_weights() -> dict[str, Any]:
        weights = {"technical": 0.0, "ml": 0.0, "sentiment": 0.0,
                   "macro": 0.0, "news": 0.0, "orderbook": 0.0}
        profile = "unavailable"
        regime_hint = "UNKNOWN"

        if signal_generator is not None:
            try:
                current_regime = getattr(signal_generator, "_current_regime", None)
                if current_regime is None and data_manager is not None:
                    raw = getattr(data_manager, "_regimes", {}) or {}
                    for _, state in raw.items():
                        current_regime = getattr(state, "regime", None)
                        break
                if current_regime is not None:
                    regime_hint = str(getattr(current_regime, "value", current_regime)).upper()
                if hasattr(signal_generator, "_get_regime_weights"):
                    w = signal_generator._get_regime_weights(current_regime)
                    if isinstance(w, dict):
                        for k in weights:
                            weights[k] = float(w.get(k, weights[k]))
                        profile = f"adaptive_{regime_hint.lower()}"
            except Exception as e:
                logger.debug("weights fetch error: {}", e)

        return {"weights": weights, "profile_name": profile, "regime_hint": regime_hint}

    @app.get("/api/arms/prewarm")
    async def api_arms_prewarm() -> dict[str, Any]:
        result = {
            "ws_feed_lag_ms": 0,
            "order_exec_avg_ms": 0,
            "order_exec_p95_ms": 0,
            "last_order_ms": 0,
            "cache_hit_rate": 0.0,
            "cache_age_sec": 0.0,
            "prewarm_active": False,
            "prewarm_before_ms": 0,
            "prewarm_after_ms": 0,
            "status": "normal",
        }
        if metrics and hasattr(metrics, "get_latency_stats"):
            try:
                lat = metrics.get_latency_stats() or {}
                fl = lat.get("feed_lag", {}) if isinstance(lat, dict) else {}
                ol = lat.get("order_latency", {}) if isinstance(lat, dict) else {}
                result["ws_feed_lag_ms"] = int(fl.get("avg_ms", 0) or 0)
                result["order_exec_avg_ms"] = int(ol.get("avg_ms", 0) or 0)
                result["order_exec_p95_ms"] = int(ol.get("p95_ms", 0) or 0)
                result["last_order_ms"] = int(ol.get("last_ms", 0) or 0)
            except Exception as e:
                logger.debug("prewarm latency fetch error: {}", e)

        if cache is not None:
            try:
                st = cache.stats() if hasattr(cache, "stats") else {}
                hits = float(st.get("hits", 0) or 0)
                misses = float(st.get("misses", 0) or 0)
                total = hits + misses
                result["cache_hit_rate"] = round(hits / total, 3) if total > 0 else 0.0
                result["cache_age_sec"] = float(st.get("avg_age_sec", 0.0) or 0.0)
            except Exception:
                pass

        result["prewarm_after_ms"] = result["order_exec_avg_ms"]
        result["prewarm_before_ms"] = int(result["order_exec_avg_ms"] * 10) if result["order_exec_avg_ms"] else 0
        result["prewarm_active"] = bool(result["order_exec_avg_ms"] and result["order_exec_avg_ms"] < 500)

        p95 = result["order_exec_p95_ms"] or result["order_exec_avg_ms"]
        if p95 >= 1000:
            result["status"] = "degraded"
        elif p95 >= 500:
            result["status"] = "elevated"
        else:
            result["status"] = "normal"

        return result

    @app.get("/api/clientkey")
    async def api_clientkey() -> dict[str, Any]:
        """Return dashboard auth metadata without echoing the shared secret."""
        return {"configured": bool(api_key), "auth_required": require_api_key}

    @app.get("/api/live/readiness")
    async def api_live_readiness() -> dict[str, Any]:
        """Fail-closed live readiness checklist for operators and automation."""
        checks: dict[str, dict[str, Any]] = {}

        checks["mode"] = {
            "ok": not bool(config.paper_mode),
            "value": "paper" if config.paper_mode else "live",
            "message": "live mode active" if not config.paper_mode else "paper mode active",
        }

        db_ok = bool(getattr(db_handler, "available", False))
        sqlite_personal = bool(config.get_value("storage", "personal_sqlite_mode", default=False))
        sqlite_ok = bool(sqlite_store is not None and getattr(sqlite_store, "available", False))
        audit_ok = db_ok or (sqlite_personal and sqlite_ok)
        checks["audit_db"] = {
            "ok": audit_ok,
            "mode": "postgresql" if db_ok else ("sqlite_personal" if sqlite_personal and sqlite_ok else "unavailable"),
            "message": (
                "PostgreSQL audit persistence available"
                if db_ok
                else (
                    "Personal SQLite audit persistence available"
                    if sqlite_personal and sqlite_ok
                    else "Audit persistence unavailable"
                )
            ),
        }

        dash_ok = bool(require_api_key and api_key)
        checks["dashboard_auth"] = {
            "ok": dash_ok,
            "message": "dashboard API key auth enabled" if dash_ok else "dashboard API key auth is not fully configured",
        }

        exchange_count = len(executors or [])
        exchange_details: list[dict[str, Any]] = []
        contract_details: list[dict[str, Any]] = []
        live_clients = 0
        for exc in (executors or []):
            ex_id = str(getattr(exc, "exchange_id", type(exc).__name__))
            contract = executor_contract_status(exc, require_order_controls=True, require_market_data=True)
            contract_details.append(contract)
            client = getattr(exc, "_client", None)
            client_ok = client is not None
            markets = getattr(client, "markets", None) if client is not None else None
            # Markets are populated by ccxt.load_markets during executor init;
            # requiring either loaded markets or an explicit initialized marker
            # prevents a mere executor object from passing live readiness.
            initialized = bool(client_ok and (markets or getattr(exc, "_running", False)))
            if initialized:
                live_clients += 1
            exchange_details.append({
                "exchange": ex_id,
                "client": client_ok,
                "initialized": initialized,
                "markets_loaded": bool(markets),
                "contract_ok": bool(contract.get("contract_ok", False)),
                "contract_blockers": contract.get("blockers", []),
            })
        exchange_ok = exchange_count > 0 and live_clients == exchange_count
        checks["exchange"] = {
            "ok": exchange_ok,
            "count": exchange_count,
            "initialized_clients": live_clients,
            "details": exchange_details,
            "message": "exchange clients initialized" if exchange_ok else "one or more exchange clients are unavailable or uninitialized",
        }
        missing_contract = sorted({b for item in contract_details for b in item.get("blockers", [])})
        contract_ok = exchange_count > 0 and all(item.get("contract_ok", False) for item in contract_details)
        checks["executor_contract"] = {
            "ok": contract_ok,
            "details": contract_details,
            "missing": missing_contract,
            "message": (
                "executor close/cancel/orderbook contract available"
                if contract_ok
                else f"executor contract missing: {', '.join(missing_contract) or 'executor'}"
            ),
        }

        risk_snap = risk_manager.get_risk_snapshot() if risk_manager is not None and hasattr(risk_manager, "get_risk_snapshot") else {}
        trading_state = str(risk_snap.get("trading_state", "UNKNOWN") or "UNKNOWN")
        risk_ok = (
            risk_manager is not None
            and not _risk_kill_switch_active()
            and bool(risk_snap.get("can_open_new_positions", True))
        )
        checks["risk"] = {
            "ok": risk_ok,
            "trading_state": trading_state,
            "blockers": risk_snap.get("trading_state_blockers", []),
            "message": "risk manager ready" if risk_ok else f"risk manager blocked ({trading_state.lower()})",
        }

        user_stream_ok = True
        user_stream_connected = bool(getattr(user_stream, "connected", False)) if user_stream is not None else False
        user_stream_message = "user data stream not required in paper mode"
        if not config.paper_mode:
            user_stream_ok = user_stream is not None and user_stream_connected
            user_stream_message = "user data stream connected" if user_stream_ok else "user data stream is not connected"
        checks["user_stream"] = {
            "ok": user_stream_ok,
            "connected": user_stream_connected,
            "message": user_stream_message,
        }

        recon_ok = True
        recon_message = "startup reconciliation clean"
        recon_mismatches: list[Any] = []
        recon_positions_without_sl: list[Any] = []
        if not config.paper_mode:
            if reconciliation_result is None:
                recon_ok = False
                recon_message = "startup reconciliation has not run"
            else:
                recon_success = bool(getattr(reconciliation_result, "success", False))
                recon_safe_mode = bool(getattr(reconciliation_result, "safe_mode", False))
                recon_mismatches = list(getattr(reconciliation_result, "mismatches", []) or [])
                recon_positions_without_sl = list(getattr(reconciliation_result, "positions_without_sl", []) or [])
                recon_ok = recon_success and not recon_safe_mode and not recon_mismatches and not recon_positions_without_sl
                if not recon_ok:
                    recon_message = "startup reconciliation failed, safe-mode is active, or protective-order gaps remain"
        checks["reconciliation"] = {
            "ok": recon_ok,
            "message": recon_message,
            "mismatches": recon_mismatches,
            "positions_without_sl": recon_positions_without_sl,
        }

        blockers = [name for name, check in checks.items() if not check.get("ok")]
        return {
            "ready_for_live": not blockers,
            "blockers": blockers,
            "checks": checks,
            "config_hash": str(getattr(app.state, "config_hash", "")),
            "timestamp": int(time.time()),
        }

    @app.get("/api/entry-eligibility")
    async def api_entry_eligibility(symbol: str | None = None, exchange: str | None = None) -> dict[str, Any]:
        """Current entry/no-trade decision with a receipt-style audit payload."""
        if signal_generator is None or not hasattr(signal_generator, "get_entry_eligibility"):
            return {
                "available": False,
                "allowed": False,
                "decision": "NO_ENTRY",
                "reason": "signal_generator_unavailable",
                "blockers": ["signal_generator_unavailable"],
                "warnings": [],
                "timestamp": int(time.time()),
            }
        try:
            payload = signal_generator.get_entry_eligibility(symbol=symbol, exchange=exchange)
            if isinstance(payload, dict):
                payload.setdefault("available", True)
                payload.setdefault("timestamp", int(time.time()))
                return payload
        except Exception as exc:
            return {
                "available": False,
                "allowed": False,
                "decision": "NO_ENTRY",
                "reason": sanitize_exception(exc),
                "blockers": ["entry_eligibility_error"],
                "warnings": [],
                "timestamp": int(time.time()),
            }
        return {
            "available": False,
            "allowed": False,
            "decision": "NO_ENTRY",
            "reason": "entry_eligibility_unavailable",
            "blockers": ["entry_eligibility_unavailable"],
            "warnings": [],
            "timestamp": int(time.time()),
        }

    @app.get("/api/pipeline/readiness")
    async def api_pipeline_readiness() -> dict[str, Any]:
        """Unified paper/live readiness gate for the full trading pipeline."""
        paper_mode = bool(config.paper_mode)
        mode = "paper" if paper_mode else "live"
        stages: list[dict[str, Any]] = []
        entry_payload: dict[str, Any] = {
            "available": False,
            "allowed": False,
            "decision": "NO_ENTRY",
            "reason": "not_evaluated",
            "blockers": [],
            "warnings": [],
        }

        def _status(value: str) -> str:
            s = str(value or "UNKNOWN").upper()
            if s in {"OK", "READY", "HEALTHY", "PASS"}:
                return "PASS"
            if s in {"WARN", "WARNING", "WARMING", "DEGRADED", "SOFT"}:
                return "WARN"
            if s in {"BLOCK", "BLOCKED", "FAIL", "FAILED", "ERROR", "UNAVAILABLE"}:
                return "BLOCK"
            return "UNKNOWN"

        def _add_stage(
            stage_id: str,
            name: str,
            status: str,
            detail: str,
            *,
            critical: bool = True,
            metrics_payload: dict[str, Any] | None = None,
        ) -> None:
            stages.append({
                "id": stage_id,
                "name": name,
                "status": _status(status),
                "detail": str(detail or ""),
                "critical": bool(critical),
                "metrics": metrics_payload or {},
            })

        def _max_p95(bucket: Any, *, ignore_prefixes: tuple[str, ...] = ()) -> float:
            if not isinstance(bucket, dict):
                return 0.0
            value = 0.0
            for key, item in bucket.items():
                if ignore_prefixes and str(key).startswith(ignore_prefixes):
                    continue
                if isinstance(item, dict):
                    value = max(value, float(item.get("p95_ms", 0.0) or 0.0))
            return value

        # L0 market-data integrity.
        try:
            md = await api_market_data_health()
            md_status = str(md.get("status", "UNKNOWN")).upper()
            feeds = md.get("feeds", [])
            feed_count = len(feeds) if isinstance(feeds, list) else int(md.get("feed_count", 0) or 0)
            md_ok = bool(md.get("healthy", False)) or md_status in {"OK", "READY", "HEALTHY", "PASS"}
            if md_ok:
                stage_status = "PASS"
            elif md_status in {"ERROR", "BLOCKED", "FAIL", "FAILED"}:
                stage_status = "BLOCK"
            else:
                stage_status = "WARN"
            _add_stage(
                "market_data",
                "Market Data L0",
                stage_status,
                f"{md_status.lower()} with {feed_count} feed(s)",
                metrics_payload={
                    "status": md_status,
                    "feed_count": feed_count,
                    "healthy": bool(md.get("healthy", False)),
                    "reason": md.get("reason"),
                },
            )
        except Exception as exc:
            _add_stage("market_data", "Market Data L0", "BLOCK", sanitize_exception(exc))

        # Event bus pressure and sequencing backbone.
        try:
            eb = await api_eventbus_stats()
            if not eb.get("available", False):
                _add_stage("event_bus", "Event Bus", "BLOCK", "event bus unavailable")
            else:
                queue_pct = float(eb.get("queue_pct", 0.0) or 0.0)
                dropped = int(eb.get("dropped_count", 0) or 0)
                running = bool(eb.get("running", True))
                if not running or queue_pct >= 90.0:
                    stage_status = "BLOCK"
                elif queue_pct >= 70.0 or dropped > 0:
                    stage_status = "WARN"
                else:
                    stage_status = "PASS"
                _add_stage(
                    "event_bus",
                    "Event Bus",
                    stage_status,
                    f"queue {queue_pct:.1f}%, dropped {dropped}, topics {int(eb.get('subscribed_topics', 0) or 0)}",
                    metrics_payload=eb,
                )
        except Exception as exc:
            _add_stage("event_bus", "Event Bus", "BLOCK", sanitize_exception(exc))

        # Candle/indicator manager.
        try:
            if data_manager is None:
                _add_stage("data_manager", "Data Manager", "BLOCK", "data manager unavailable")
            else:
                running = bool(getattr(data_manager, "_running", False))
                history = getattr(data_manager, "_candle_history", {}) or {}
                series = sum(len(tf_map) for tf_map in history.values()) if isinstance(history, dict) else 0
                _add_stage(
                    "data_manager",
                    "Data Manager",
                    "PASS" if running else "BLOCK",
                    f"{'running' if running else 'stopped'}, {series} candle series",
                    metrics_payload={"running": running, "series": series},
                )
        except Exception as exc:
            _add_stage("data_manager", "Data Manager", "BLOCK", sanitize_exception(exc))

        # Strategy layer preview from the stale-while-revalidate cache only.
        try:
            cached_layers = _layers_cache.get("_")
            if not cached_layers and hasattr(signal_generator, "get_quality_preview"):
                preview_layers = _build_layers_skeleton()
                await _populate_layers_status(preview_layers)
                preview_payload = _finalize_layers_response(preview_layers)
                _layers_cache["_"] = (time.monotonic(), preview_payload)
                cached_layers = _layers_cache.get("_")
            if cached_layers:
                layer_payload = cached_layers[1] or {}
                layer_list = list(layer_payload.get("layers", []) or [])
                layer_stale = bool(layer_payload.get("stale", False))
                layer_age = float(layer_payload.get("age_seconds", 0.0) or 0.0)
                pending = [
                    str(item.get("layer_index", item.get("id", "")))
                    for item in layer_list
                    if (
                        str(item.get("status", "")).upper() in {"PENDING", "UNKNOWN", ""}
                        or _layers_is_stale_detail(str(item.get("detail", "") or ""))
                    )
                ]
                weak = [
                    str(item.get("layer_index", item.get("id", "")))
                    for item in layer_list
                    if str(item.get("status", "")).upper() in {"FAIL", "BLOCK", "BLOCKED", "WARN"}
                ]
                if layer_stale:
                    stage_status = "WARN"
                    detail = f"{len(layer_list)} cached layer(s), stale {layer_age:.0f}s; entry gate uses fresh preview"
                elif pending:
                    stage_status = "WARN"
                    detail = f"{len(layer_list)} cached layer(s), pending: {', '.join(pending[:4])}"
                else:
                    stage_status = "PASS"
                    detail = f"{len(layer_list)} layer(s) evaluated"
                    if weak:
                        detail += f", no-trade: {', '.join(weak[:4])}"
                _add_stage(
                    "strategy_layers",
                    "Strategy Layers",
                    stage_status,
                    detail,
                    critical=False,
                    metrics_payload={
                        "layers": len(layer_list),
                        "pending": pending[:8],
                        "weak": weak[:8],
                        "stale": layer_stale,
                        "age_seconds": layer_age,
                    },
                )
            else:
                _add_stage(
                    "strategy_layers",
                    "Strategy Layers",
                    "WARN",
                    "layer preview cache warming",
                    critical=False,
                )
        except Exception as exc:
            _add_stage("strategy_layers", "Strategy Layers", "WARN", sanitize_exception(exc), critical=False)

        # Signal loop and auto-trading state.
        try:
            auto = await api_auto_status()
            if signal_generator is None:
                _add_stage("signal_engine", "Signal Engine", "BLOCK", "signal generator unavailable")
            else:
                running_attr = getattr(signal_generator, "_running", None)
                running = True if running_attr is None else bool(running_attr)
                auto_enabled = bool(auto.get("enabled", auto.get("auto_trading_enabled", False)))
                _add_stage(
                    "signal_engine",
                    "Signal Engine",
                    "PASS" if running else "BLOCK",
                    f"{'running' if running else 'stopped'}, auto {'on' if auto_enabled else 'off'}",
                    metrics_payload={"running": running, "auto_trading_enabled": auto_enabled, "mode": auto.get("mode", mode)},
                )
        except Exception as exc:
            _add_stage("signal_engine", "Signal Engine", "BLOCK", sanitize_exception(exc))

        # Predictive model state.
        try:
            ml = await api_ml_status()
            loaded = bool(ml.get("loaded", False))
            feature_count = int(ml.get("feature_count", 0) or 0)
            _add_stage(
                "model",
                "Model",
                "PASS" if loaded else "WARN",
                f"{ml.get('model_type', 'model')} {'loaded' if loaded else 'not loaded'}, {feature_count} feature(s)",
                critical=False,
                metrics_payload=ml,
            )
        except Exception as exc:
            _add_stage("model", "Model", "WARN", sanitize_exception(exc), critical=False)

        # AI/agent decision provider. Advisory is safe by default; direct mode is paper-gated.
        try:
            agent = await api_agent_status()
            requested = str(agent.get("requested_mode", agent.get("mode", "off")) or "off").lower()
            effective = str(agent.get("effective_mode", agent.get("mode", "off")) or "off").lower()
            direct_requested = requested in {"direct", "full"} or effective == "direct"
            attached = bool(agent.get("attached", False))
            enabled = bool(agent.get("enabled", False))
            if direct_requested and not paper_mode:
                _add_stage(
                    "ai_decision",
                    "AI Decision",
                    "BLOCK",
                    "AI-direct requested outside paper mode",
                    critical=True,
                    metrics_payload=agent,
                )
            elif attached and enabled:
                _add_stage(
                    "ai_decision",
                    "AI Decision",
                    "PASS",
                    f"{effective or requested} mode",
                    critical=False,
                    metrics_payload=agent,
                )
            else:
                _add_stage(
                    "ai_decision",
                    "AI Decision",
                    "WARN",
                    "agent advisory unavailable; core strategy still active",
                    critical=False,
                    metrics_payload=agent,
                )
        except Exception as exc:
            _add_stage("ai_decision", "AI Decision", "WARN", sanitize_exception(exc), critical=False)

        # Risk gate.
        try:
            if risk_manager is None:
                _add_stage("risk", "Risk Gate", "BLOCK", "risk manager unavailable")
            else:
                snap = risk_manager.get_risk_snapshot() if hasattr(risk_manager, "get_risk_snapshot") else {}
                kill = _risk_kill_switch_active()
                circuit = bool(snap.get("circuit_breaker_tripped", False)) if isinstance(snap, dict) else False
                safe_mode = bool(getattr(getattr(risk_manager, "safe_mode", None), "is_active", False))
                trading_state = str(snap.get("trading_state", "UNKNOWN") or "UNKNOWN") if isinstance(snap, dict) else "UNKNOWN"
                can_open = bool(snap.get("can_open_new_positions", True)) if isinstance(snap, dict) else False
                blocked = kill or circuit or safe_mode or not can_open
                detail = "ready"
                if kill:
                    detail = "kill switch active"
                elif circuit:
                    detail = f"circuit breaker: {snap.get('circuit_breaker_reason', 'active')}"
                elif safe_mode:
                    detail = "safe mode active"
                elif not can_open:
                    detail = f"trading state {trading_state.lower()}"
                _add_stage(
                    "risk",
                    "Risk Gate",
                    "BLOCK" if blocked else "PASS",
                    detail,
                    metrics_payload=snap if isinstance(snap, dict) else {},
                )
        except Exception as exc:
            _add_stage("risk", "Risk Gate", "BLOCK", sanitize_exception(exc))

        # Entry eligibility is the current no-trade decision, not infrastructure readiness.
        try:
            entry_payload = await api_entry_eligibility()
            entry_allowed = bool(entry_payload.get("allowed", False))
            entry_reason = str(entry_payload.get("reason", "") or ("entry_allowed" if entry_allowed else "no_entry"))
            entry_blockers = list(entry_payload.get("blockers", []) or [])
            detail = "entry allowed" if entry_allowed else entry_reason
            if entry_blockers and not entry_allowed:
                detail = f"{entry_reason}; blockers {', '.join(str(item) for item in entry_blockers[:3])}"
            _add_stage(
                "entry_eligibility",
                "Entry Eligibility",
                "PASS" if entry_allowed else "WARN",
                detail,
                critical=False,
                metrics_payload=entry_payload if isinstance(entry_payload, dict) else {},
            )
        except Exception as exc:
            entry_payload = {
                "available": False,
                "allowed": False,
                "decision": "NO_ENTRY",
                "reason": sanitize_exception(exc),
                "blockers": ["entry_eligibility_error"],
                "warnings": [],
            }
            _add_stage("entry_eligibility", "Entry Eligibility", "WARN", sanitize_exception(exc), critical=False)

        # Exchange executor contract.
        try:
            execs = list(executors or [])
            details: list[dict[str, Any]] = []
            live_clients = 0
            paper_clients = 0
            for exc in execs:
                ex_id = str(getattr(exc, "exchange_id", type(exc).__name__))
                contract = executor_contract_status(exc, require_order_controls=True, require_market_data=True)
                client = getattr(exc, "_client", None)
                markets = getattr(client, "markets", None) if client is not None else None
                is_paper = bool(getattr(exc, "is_paper", False)) or "simulated" in type(exc).__name__.lower()
                initialized = bool(client is not None and (markets or getattr(exc, "_running", False)))
                if getattr(exc, "_hl_exchange", None) is not None:
                    initialized = True
                live_clients += 1 if initialized else 0
                paper_clients += 1 if is_paper else 0
                details.append({
                    "exchange": ex_id,
                    "paper": is_paper,
                    "client": client is not None,
                    "initialized": initialized,
                    "contract": contract,
                })
            contract_ok = bool(execs and all(item.get("contract", {}).get("contract_ok", False) for item in details))
            missing_contract = sorted({
                blocker
                for item in details
                for blocker in item.get("contract", {}).get("blockers", [])
            })
            if not execs:
                _add_stage("executor", "Executor", "BLOCK", "no exchange executor configured")
            elif paper_mode and paper_clients < len(execs):
                _add_stage(
                    "executor",
                    "Executor",
                    "BLOCK",
                    "paper mode has non-paper executor(s)",
                    metrics_payload={"count": len(execs), "details": details},
                )
            elif not contract_ok:
                _add_stage(
                    "executor",
                    "Executor",
                    "BLOCK",
                    f"contract missing {', '.join(missing_contract) or 'executor method'}",
                    metrics_payload={"count": len(execs), "details": details},
                )
            elif paper_mode:
                _add_stage(
                    "executor",
                    "Executor",
                    "PASS",
                    f"{paper_clients} simulated executor(s), contract ok",
                    metrics_payload={"count": len(execs), "details": details},
                )
            elif live_clients == len(execs):
                _add_stage(
                    "executor",
                    "Executor",
                    "PASS",
                    f"{live_clients}/{len(execs)} live client(s), contract ok",
                    metrics_payload={"count": len(execs), "details": details},
                )
            else:
                _add_stage(
                    "executor",
                    "Executor",
                    "BLOCK",
                    f"{live_clients}/{len(execs)} live client(s) initialized",
                    metrics_payload={"count": len(execs), "details": details},
                )
        except Exception as exc:
            _add_stage("executor", "Executor", "BLOCK", sanitize_exception(exc))

        # Order manager and idempotency ledger.
        try:
            if order_manager is None:
                _add_stage("order_manager", "Order Manager", "BLOCK", "order manager unavailable")
            else:
                stats = order_manager.get_stats() if hasattr(order_manager, "get_stats") else {}
                _add_stage(
                    "order_manager",
                    "Order Manager",
                    "PASS",
                    f"{int(stats.get('open_orders', 0) or 0)} open, {int(stats.get('filled_orders', 0) or 0)} filled",
                    metrics_payload=stats,
                )
        except Exception as exc:
            _add_stage("order_manager", "Order Manager", "BLOCK", sanitize_exception(exc))

        # Durable audit trail.
        try:
            db_ok = bool(getattr(db_handler, "available", False))
            sqlite_personal = bool(config.get_value("storage", "personal_sqlite_mode", default=False))
            sqlite_ok = bool(sqlite_store is not None and getattr(sqlite_store, "available", False))
            audit_ok = db_ok or (sqlite_personal and sqlite_ok)
            if audit_ok:
                persistence_status = "PASS"
            else:
                persistence_status = "BLOCK" if not paper_mode else "WARN"
            detail = "postgresql audit db" if db_ok else ("personal sqlite audit db" if sqlite_personal and sqlite_ok else "audit persistence unavailable")
            _add_stage(
                "persistence",
                "Persistence",
                persistence_status,
                detail,
                critical=not paper_mode,
                metrics_payload={"postgresql": db_ok, "sqlite_personal": sqlite_personal, "sqlite": sqlite_ok},
            )
        except Exception as exc:
            _add_stage("persistence", "Persistence", "BLOCK" if not paper_mode else "WARN", sanitize_exception(exc), critical=not paper_mode)

        # Local startup recovery guard for SQLite personal mode.
        try:
            recovery = {}
            if risk_manager is not None and hasattr(risk_manager, "get_startup_recovery_status"):
                recovery = risk_manager.get_startup_recovery_status()
            elif hasattr(app.state, "sqlite_recovery_result"):
                recovery = dict(getattr(app.state, "sqlite_recovery_result") or {})
            errors = list(recovery.get("errors", []) or [])
            ledger = _build_recovery_ledger_audit()
            ledger_issues = (
                int(ledger.get("duplicate_count", 0) or 0)
                + int(ledger.get("orphan_count", 0) or 0)
                + int(ledger.get("phantom_count", 0) or 0)
            ) if ledger.get("available", False) else 0
            restored = int(recovery.get("restored", 0) or 0)
            attempted = int(recovery.get("attempted", 0) or 0)
            skipped = int(recovery.get("skipped", 0) or 0)
            disabled = bool(recovery.get("disabled", False))
            over_limit = bool(recovery.get("over_limit", False))
            if ledger_issues:
                stage_status = "WARN"
                detail = (
                    f"ledger dup={int(ledger.get('duplicate_count', 0) or 0)} "
                    f"orphan={int(ledger.get('orphan_count', 0) or 0)} "
                    f"phantom={int(ledger.get('phantom_count', 0) or 0)}"
                )
            elif errors:
                stage_status = "BLOCK" if not paper_mode else "WARN"
                detail = f"sqlite recovery error(s): {len(errors)}"
            elif over_limit:
                stage_status = "WARN"
                detail = f"restored {restored}/{attempted}, above max position limit"
            elif skipped:
                stage_status = "PASS"
                detail = f"restored {restored}/{attempted}; ledger clean"
            elif disabled:
                stage_status = "WARN" if not paper_mode else "PASS"
                detail = "position recovery disabled"
            else:
                stage_status = "PASS"
                detail = f"restored {restored}/{attempted} open position(s)"
            _add_stage(
                "startup_recovery",
                "Startup Recovery",
                stage_status,
                detail,
                critical=not paper_mode,
                metrics_payload={**recovery, "ledger": ledger, "skipped_at_startup": skipped},
            )
        except Exception as exc:
            _add_stage("startup_recovery", "Startup Recovery", "WARN", sanitize_exception(exc), critical=False)

        # User stream and reconciliation are live-critical, informational in paper.
        try:
            stream_connected = bool(getattr(user_stream, "connected", False)) if user_stream is not None else False
            if paper_mode:
                _add_stage("user_stream", "User Stream", "PASS", "not required in paper mode", critical=False)
            else:
                _add_stage(
                    "user_stream",
                    "User Stream",
                    "PASS" if stream_connected else "BLOCK",
                    "connected" if stream_connected else "not connected",
                    metrics_payload={"connected": stream_connected},
                )
        except Exception as exc:
            _add_stage("user_stream", "User Stream", "BLOCK" if not paper_mode else "WARN", sanitize_exception(exc), critical=not paper_mode)

        try:
            if paper_mode:
                _add_stage("reconciliation", "Reconciliation", "PASS", "not required in paper mode", critical=False)
            elif reconciliation_result is None:
                _add_stage("reconciliation", "Reconciliation", "BLOCK", "startup reconciliation has not run")
            else:
                mismatches = list(getattr(reconciliation_result, "mismatches", []) or [])
                missing_sl = list(getattr(reconciliation_result, "positions_without_sl", []) or [])
                recon_ok = (
                    bool(getattr(reconciliation_result, "success", False))
                    and not bool(getattr(reconciliation_result, "safe_mode", False))
                    and not mismatches
                    and not missing_sl
                )
                _add_stage(
                    "reconciliation",
                    "Reconciliation",
                    "PASS" if recon_ok else "BLOCK",
                    "clean" if recon_ok else "mismatch or protective-order gap",
                    metrics_payload={"mismatches": len(mismatches), "positions_without_sl": len(missing_sl)},
                )
        except Exception as exc:
            _add_stage("reconciliation", "Reconciliation", "BLOCK" if not paper_mode else "WARN", sanitize_exception(exc), critical=not paper_mode)

        # Alerts and recent logs.
        try:
            am = getattr(app.state, "alert_manager", None)
            if am is None:
                _add_stage("alerts", "Alerts", "WARN", "alert manager unavailable", critical=False)
            else:
                alert_status = am.get_status() if hasattr(am, "get_status") else {}
                channels = int(alert_status.get("channels", 0) or 0)
                _add_stage(
                    "alerts",
                    "Alerts",
                    "PASS" if channels > 0 else "WARN",
                    f"{channels} channel(s)",
                    critical=False,
                    metrics_payload=alert_status,
                )
        except Exception as exc:
            _add_stage("alerts", "Alerts", "WARN", sanitize_exception(exc), critical=False)

        try:
            recent = list(_log_buffer)[-50:]
            errors = sum(1 for row in recent if str(row.get("level", "")).upper() in {"ERROR", "CRITICAL"})
            _add_stage(
                "logs",
                "Logs",
                "WARN" if errors else "PASS",
                f"{errors} recent error(s)",
                critical=False,
                metrics_payload={"recent": len(recent), "errors": errors},
            )
        except Exception as exc:
            _add_stage("logs", "Logs", "WARN", sanitize_exception(exc), critical=False)

        # Latency telemetry across feed, decision, risk/execution and order ack paths.
        try:
            lat = metrics.get_latency_percentiles() if metrics and hasattr(metrics, "get_latency_percentiles") else {}
            feed_p95 = _max_p95(lat.get("feed_lag", {})) if isinstance(lat, dict) else 0.0
            order_p95 = _max_p95(lat.get("order_latency", {})) if isinstance(lat, dict) else 0.0
            decision_p95 = _max_p95(lat.get("decision_latency", {})) if isinstance(lat, dict) else 0.0
            pipeline_bucket = lat.get("pipeline_latency", {}) if isinstance(lat, dict) else {}
            pipeline_p95 = _max_p95(
                pipeline_bucket,
                ignore_prefixes=("paper_fill_probe",),
            ) if isinstance(lat, dict) else 0.0
            sample_count = 0
            if isinstance(lat, dict):
                for bucket in ("feed_lag", "order_latency", "decision_latency"):
                    for item in (lat.get(bucket, {}) or {}).values():
                        if isinstance(item, dict):
                            sample_count += int(item.get("count", 0) or 0)
                for key, item in (pipeline_bucket or {}).items():
                    if str(key).startswith("paper_fill_probe"):
                        continue
                    if isinstance(item, dict):
                        sample_count += int(item.get("count", 0) or 0)
            has_samples = sample_count > 0
            worst = max(feed_p95, order_p95, decision_p95, pipeline_p95)
            if not has_samples:
                stage_status = "WARN"
            elif order_p95 >= 2500.0 or feed_p95 >= 5000.0 or pipeline_p95 >= 1000.0:
                stage_status = "BLOCK" if not paper_mode else "WARN"
            elif order_p95 >= 750.0 or feed_p95 >= 1000.0 or pipeline_p95 >= 200.0:
                stage_status = "WARN"
            else:
                stage_status = "PASS"
            _add_stage(
                "latency",
                "Latency",
                stage_status,
                "warming" if not has_samples else f"worst p95 {worst:.0f}ms",
                critical=not paper_mode,
                metrics_payload={
                    "sample_count": sample_count,
                    "feed_p95_ms": feed_p95,
                    "order_p95_ms": order_p95,
                    "decision_p95_ms": decision_p95,
                    "pipeline_p95_ms": pipeline_p95,
                },
            )
        except Exception as exc:
            _add_stage("latency", "Latency", "WARN", sanitize_exception(exc), critical=False)

        # Final live gate mirrors /api/live/readiness but does not block paper mode.
        live_payload: dict[str, Any] = {}
        try:
            live_payload = await api_live_readiness()
            live_ready = bool(live_payload.get("ready_for_live", False))
            if paper_mode:
                _add_stage("live_gate", "Live Gate", "PASS", "paper mode active", critical=False, metrics_payload=live_payload)
            else:
                _add_stage(
                    "live_gate",
                    "Live Gate",
                    "PASS" if live_ready else "BLOCK",
                    "ready for live" if live_ready else f"blocked: {', '.join(live_payload.get('blockers', []) or [])}",
                    metrics_payload=live_payload,
                )
        except Exception as exc:
            live_payload = {"ready_for_live": False, "error": sanitize_exception(exc)}
            _add_stage("live_gate", "Live Gate", "BLOCK" if not paper_mode else "WARN", sanitize_exception(exc), critical=not paper_mode)

        summary = {"pass": 0, "warn": 0, "block": 0, "unknown": 0, "total": len(stages)}
        for stage in stages:
            key = str(stage.get("status", "UNKNOWN")).lower()
            if key in summary:
                summary[key] += 1
            else:
                summary["unknown"] += 1

        critical_blockers = [stage["id"] for stage in stages if stage.get("critical", True) and stage.get("status") == "BLOCK"]
        warnings = [stage["id"] for stage in stages if stage.get("status") in {"WARN", "UNKNOWN"}]
        if critical_blockers:
            overall = "BLOCKED"
        elif warnings:
            overall = "WARMING"
        else:
            overall = "READY"

        can_trade_paper = not critical_blockers
        can_trade_live = (not paper_mode) and can_trade_paper and bool(live_payload.get("ready_for_live", False))
        entry_allowed_now = bool(entry_payload.get("allowed", False)) and can_trade_paper
        can_enter_paper = can_trade_paper and entry_allowed_now
        can_enter_live = can_trade_live and entry_allowed_now
        return {
            "available": True,
            "mode": mode,
            "status": overall,
            "engine_ready_paper": can_trade_paper,
            "engine_ready_live": can_trade_live,
            "can_trade_paper": can_trade_paper,
            "can_trade_live": can_trade_live,
            "entry_allowed_now": entry_allowed_now,
            "can_enter_paper": can_enter_paper,
            "can_enter_live": can_enter_live,
            "entry_blockers": list(entry_payload.get("blockers", []) or []),
            "entry_receipt": entry_payload,
            "blockers": critical_blockers,
            "warnings": warnings,
            "summary": summary,
            "stages": stages,
            "live_readiness": live_payload,
            "config_hash": str(getattr(app.state, "config_hash", "")),
            "timestamp": int(time.time()),
        }

    @app.get("/api/health/detailed")
    async def api_health_detailed() -> dict[str, Any]:
        hc = getattr(app.state, "health_checker", None)
        if hc is None:
            return {"overall": "unknown", "components": {}, "timestamp": int(time.time())}
        try:
            result = await hc.check_all_components()
            return {
                "overall": result.overall_status.value.lower(),
                "uptime_seconds": round(result.uptime_seconds),
                "components": {
                    name: {
                        "status": comp.status.value,
                        "latency_ms": round(comp.latency_ms, 1),
                        "message": comp.message,
                        "last_check": comp.last_check,
                        "details": comp.details,
                    }
                    for name, comp in result.components.items()
                },
                "timestamp": int(time.time()),
            }
        except Exception as exc:
            logger.debug("health/detailed error: {}", exc)
            return {"overall": "error", "components": {}, "error": sanitize_exception(exc)}

    @app.get("/api/alerts/status")
    async def api_alerts_status() -> dict[str, Any]:
        am = getattr(app.state, "alert_manager", None)
        if am is None:
            return {"enabled": False}
        status = am.get_status()
        status["enabled"] = True
        return status

    @app.get("/api/alerts/history")
    async def api_alerts_history() -> dict[str, Any]:
        am = getattr(app.state, "alert_manager", None)
        if am is None:
            return {"alerts": [], "enabled": False}
        history = [a.to_dict() for a in am.history[-50:]]
        return {"alerts": history, "total": len(am.history), "enabled": True}

    # ── V6 spec endpoints: SessionFilter, SMC, MasterScorer, Retrainer, Pairs, EStop ──
    def _sg():
        return signal_generator

    @app.get("/api/v6/killzone")
    async def api_killzone() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_session_status"):
            return {"enabled": False}
        return sg.get_session_status()

    @app.get("/api/smc/{symbol:path}")
    async def api_smc_symbol(symbol: str) -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_smc_signal"):
            return {"symbol": symbol, "signal": None}
        return {"symbol": symbol, "signal": sg.get_smc_signal(symbol)}

    @app.get("/api/smc")
    async def api_smc_all() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_smc_signal"):
            return {"signals": {}}
        return {"signals": sg.get_smc_signal()}

    @app.get("/api/v6/master/{symbol:path}")
    async def api_master_score_symbol(symbol: str) -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_master_score"):
            return {"symbol": symbol, "score": None}
        return {"symbol": symbol, "score": sg.get_master_score(symbol)}

    @app.get("/api/v6/master")
    async def api_master_score_all() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_master_score"):
            return {"scores": {}}
        return {"scores": sg.get_master_score()}

    @app.get("/api/pairs/tiers")
    async def api_pairs_tiers() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_pair_registry_snapshot"):
            return {"enabled": False, "pairs": {}}
        return sg.get_pair_registry_snapshot()

    @app.get("/api/estop/status")
    async def api_estop_status() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_estop_status"):
            return {"enabled": False, "active": False}
        return sg.get_estop_status()

    @app.post("/api/estop/trigger")
    async def api_estop_trigger(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "trigger_estop"):
            return {"ok": False, "reason": "unavailable"}
        payload = payload or {}
        return await sg.trigger_estop(
            reason=str(payload.get("reason", "manual")),
            triggered_by=str(payload.get("triggered_by", "dashboard")),
        )

    @app.post("/api/estop/release")
    async def api_estop_release(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "release_estop"):
            return {"ok": False, "reason": "unavailable"}
        payload = payload or {}
        return await sg.release_estop(released_by=str(payload.get("released_by", "dashboard")))

    @app.get("/api/efficiency/status")
    async def api_efficiency_status() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_efficiency_status"):
            return {"enabled": False}
        data = sg.get_efficiency_status()
        data["execution"] = sg.get_execution_config()
        data["enabled"] = True
        return data

    @app.get("/api/efficiency/correlation")
    async def api_efficiency_correlation() -> dict[str, Any]:
        sg = _sg()
        if sg is None or not hasattr(sg, "get_correlation_matrix"):
            return {"symbols": [], "matrix": {}}
        return sg.get_correlation_matrix()

    @app.get("/api/ml/retrain/status")
    async def api_ml_retrain_status() -> dict[str, Any]:
        mr = getattr(app.state, "model_retrainer", None)
        if mr is None:
            return {"enabled": False}
        return mr.get_status()

    @app.post("/api/ml/retrain/trigger")
    async def api_ml_retrain_trigger() -> dict[str, Any]:
        mr = getattr(app.state, "model_retrainer", None)
        if mr is None:
            return {"ok": False, "reason": "unavailable"}
        try:
            result = await mr.trigger_now()
            if result is None:
                return {"ok": False, "reason": "skipped_insufficient_data"}
            return {
                "ok": bool(result.success),
                "auc": result.auc,
                "rows": result.rows,
                "duration_s": result.duration_s,
                "reason": result.reason,
            }
        except Exception as exc:
            return {"ok": False, "reason": sanitize_exception(exc)}

    return app

# Top-level FastAPI app for uvicorn import
def _build_standalone_managers():
    """Create lightweight managers + paper trading stack for standalone dashboard."""
    cfg = _default_config()
    # Use real EventBus so CANDLE → SignalGenerator → SIGNAL pipeline works
    from core.event_bus import EventBus as RealEventBus
    bus = RealEventBus()
    om = None
    rm = None
    sg = None
    dm = None
    try:
        from core.circuit_breaker import CircuitBreaker
        from execution.order_manager import OrderManager
        from execution.risk_manager import RiskManager
        from analysis.data_manager import DataManager
        from engine.signal_generator import SignalGenerator
        cb = CircuitBreaker()
        om = OrderManager(config=cfg, event_bus=bus, circuit_breaker=cb)
        rm = RiskManager(config=cfg, event_bus=bus)
        dm = DataManager(config=cfg, event_bus=bus)
        sg = SignalGenerator(config=cfg, event_bus=bus, data_manager=dm)
        sg.set_auto_trading(False)
        # Paper mode: lower thresholds so signals can fire with limited data feeds
        if cfg.paper_mode:
            sg._min_factors = 2
            sg._min_score = 0.15
            sg._min_factor_magnitude = 0.05
            # Redistribute weights: boost tech+ML since sentiment/macro/news/orderbook are 0
            sg._tech_weight = 0.50
            sg._ml_weight = 0.40
            sg._sentiment_weight = 0.00
            sg._macro_weight = 0.02
            sg._news_weight = 0.04
            sg._orderbook_weight = 0.02
            # Disable HTF confirmation — too restrictive with limited seeded data
            sg._confirmation_tfs = []
            sg._min_signal_interval = 30  # Allow faster signals in paper mode
    except Exception as exc:
        logger.warning("Standalone manager init partial: {}", exc)
    return cfg, bus, om, rm, sg, dm

if _FASTAPI:
    _sa_cfg, _sa_bus, _sa_om, _sa_rm, _sa_sg, _sa_dm = _build_standalone_managers()
    # Attach loguru sink for /api/logs/recent
    logger.add(_log_sink, level="DEBUG", format="{message}")

    # Start DataManager + SignalGenerator event subscriptions
    async def _start_paper_stack():
        """Start paper trading components in background."""
        try:
            if _sa_dm is not None:
                await _sa_dm.run()
            if _sa_sg is not None:
                await _sa_sg.run()
        except Exception as exc:
            logger.error("Paper stack startup error: {}", exc)

    # Paper feed management
    _paper_feed_task = None
    _paper_stack_task = None
    _paper_sg_task = None
    _paper_bus_task = None
    _paper_exec_task = None
    _paper_trades: list[dict] = []

    async def _ensure_paper_stack():
        """Ensure EventBus, DataManager and SignalGenerator are running (idempotent)."""
        global _paper_stack_task, _paper_sg_task, _paper_bus_task, _paper_exec_task
        # EventBus must be running to dispatch events
        if _paper_bus_task is None and _sa_bus is not None:
            async def _run_bus():
                try:
                    await _sa_bus.run()
                except Exception as exc:
                    logger.error("EventBus run error: {}", exc)
            _paper_bus_task = asyncio.create_task(_run_bus())
        if _paper_stack_task is None and _sa_dm is not None:
            async def _run_dm():
                try:
                    await _sa_dm.run()
                except Exception as exc:
                    logger.error("DataManager run error: {}", exc)
            _paper_stack_task = asyncio.create_task(_run_dm())
        if _paper_sg_task is None and _sa_sg is not None:
            async def _run_sg():
                try:
                    await _sa_sg.run()
                except Exception as exc:
                    logger.error("SignalGenerator run error: {}", exc)
            _paper_sg_task = asyncio.create_task(_run_sg())
        # Paper executor: subscribe to SIGNAL events and log paper trades
        if _paper_exec_task is None and _sa_bus is not None:
            async def _handle_signal(signal):
                """Paper executor: simulates trade execution for signals."""
                import time as _time
                trade_id = f"paper_{int(_time.time()*1000)}"
                direction = getattr(signal, 'direction', 'unknown')
                symbol = getattr(signal, 'symbol', '??')
                price = getattr(signal, 'price', 0)
                score = getattr(signal, 'score', 0)
                sl = getattr(signal, 'stop_loss', 0)
                tp = getattr(signal, 'take_profit', 0)
                # Calculate position size (risk-based from config)
                equity = 100000.0
                risk_pct = 0.02
                size_usd = equity * risk_pct
                qty = size_usd / price if price > 0 else 0
                paper_trade = {
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "price": price,
                    "quantity": round(qty, 6),
                    "notional": round(size_usd, 2),
                    "score": round(score, 3),
                    "stop_loss": round(sl, 2),
                    "take_profit": round(tp, 2),
                    "status": "FILLED",
                    "timestamp": int(_time.time()),
                    "reasons": getattr(signal, 'reasons', []),
                }
                _paper_trades.append(paper_trade)
                logger.info(
                    "📄 PAPER TRADE: {} {} {:.6f} @ ${:.2f} (score={:.2f}, sl={:.2f}, tp={:.2f}) [{}]",
                    direction.upper(), symbol, qty, price, score, sl, tp, trade_id,
                )
                await _sa_bus.publish("ORDER_FILLED", paper_trade)
            _sa_bus.subscribe("SIGNAL", _handle_signal)
            _paper_exec_task = True  # sentinel — handler is registered, not a task
            logger.info("Paper executor subscribed to SIGNAL events")

    async def start_paper_feed():
        """Start PaperFeed to emit CANDLE events from Binance public API."""
        global _paper_feed_task
        if _paper_feed_task is not None:
            return  # already running
        try:
            from data_ingestion.paper_feed import PaperFeed
            await _ensure_paper_stack()
            symbols_cfg = _sa_cfg.get_value("exchanges", "binance", "symbols") or ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
            feed = PaperFeed(
                event_bus=_sa_bus,
                symbols=symbols_cfg,
                timeframes=["1m", "15m", "1h", "4h", "1d"],
                poll_interval=30.0,
            )
            _paper_feed_task = asyncio.create_task(feed.run())
            logger.info("Paper feed started for auto-trading")
        except Exception as exc:
            logger.error("Failed to start paper feed: {}", exc)

    async def stop_paper_feed():
        """Stop PaperFeed."""
        global _paper_feed_task
        if _paper_feed_task is not None:
            _paper_feed_task.cancel()
            _paper_feed_task = None
            logger.info("Paper feed stopped")

    app = build_app(
        config=_sa_cfg,
        event_bus=_sa_bus,
        risk_manager=_sa_rm,
        data_manager=_sa_dm,
        order_manager=_sa_om,
        db_handler=None,
        cache=None,
        signal_generator=_sa_sg,
        news_feed=None,
        orderbook_feed=None,
        sentiment_manager=None,
        dex_feed=None,
    )


async def run_dashboard(config: Config, app: Any) -> None:
    if not _FASTAPI or app is None:
        return
    api_cfg = config.get_value("monitoring", "dashboard_api") or {}
    host = api_cfg.get("host", "0.0.0.0")  # noqa: S104  # nosec B104 — auth-gated in live mode
    port = int(api_cfg.get("port", 8000))
    # Pre-bind the listening socket so we can surface a friendly error before
    # handing it to uvicorn, which otherwise logs OSError internally and
    # silently returns from serve().
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        if exc.errno == errno.EADDRINUSE:
            logger.critical(
                "Dashboard port {}:{} is already in use — another bot instance is likely running. "
                "Stop it (e.g., `pkill -f 'python3 main.py'`) and retry.",
                host, port,
            )
            raise SystemExit(1) from exc
        raise
    sock.listen(2048)
    sock.setblocking(False)

    server_config = uvicorn.Config(app, log_level="warning")
    server = uvicorn.Server(server_config)
    logger.info("Dashboard API starting on http://{}:{}", host, port)
    try:
        await server.serve(sockets=[sock])
    finally:
        sock.close()
