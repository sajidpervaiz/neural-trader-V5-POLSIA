from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from collections import defaultdict, deque
from typing import Any

from loguru import logger

try:
    from fastapi import FastAPI, Query, Request
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
    sqlite_store: Any = None,
    metrics: Any = None,
) -> Any:
    if not _FASTAPI:
        return None

    app = FastAPI(
        title="NUERAL-TRADER-5",
        description="Hybrid Rust+TypeScript+Python trading engine",
        version="4.0.0",
    )
    app.state.started_at = int(time.time())
    static_dir = Path(__file__).resolve().parent / "static"
    static_index = static_dir / "index.html"

    dashboard_cfg = config.get_value("monitoring", "dashboard_api", default={}) or {}
    cors_origins = dashboard_cfg.get("allow_origins") or ["http://localhost", "http://127.0.0.1"]
    # In paper mode, allow all origins for dev convenience (Codespace proxies, etc.)
    if config.paper_mode:
        cors_origins = ["*"]
    # Block wildcard CORS in non-paper (live) mode
    if not config.paper_mode and "*" in cors_origins:
        logger.warning(
            "CORS wildcard '*' blocked in live mode — restricting to localhost only"
        )
        cors_origins = [o for o in cors_origins if o != "*"] or ["http://localhost"]
    auth_cfg = dashboard_cfg.get("auth", {}) if hasattr(dashboard_cfg, "get") else {}
    if not isinstance(auth_cfg, dict):
        auth_cfg = {}

    require_api_key = bool(auth_cfg.get("require_api_key", False))
    api_key = str(auth_cfg.get("api_key", "") or "").strip()
    rate_limit_per_min = int(auth_cfg.get("rate_limit_per_min", 120))
    allow_unauthenticated_non_paper = bool(auth_cfg.get("allow_unauthenticated_non_paper", False))

    # Secure-by-default posture: non-paper mode requires API auth unless explicitly overridden.
    if not config.paper_mode and not require_api_key and not allow_unauthenticated_non_paper:
        require_api_key = True
        logger.warning(
            "Enabling API key requirement automatically for non-paper mode; "
            "set monitoring.dashboard_api.auth.allow_unauthenticated_non_paper=true to override"
        )
    exempt_paths = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # In-memory limiter is sufficient for single-instance Tier 0 deployments.
    ip_rate_state: dict[str, dict[str, int]] = defaultdict(lambda: {"window_start": 0, "count": 0})

    @app.middleware("http")
    async def api_guard(request: Request, call_next):
        path = request.url.path
        if path not in exempt_paths:
            if rate_limit_per_min > 0:
                ip = request.client.host if request.client else "unknown"
                now = int(time.time())
                state = ip_rate_state[ip]
                if now - state["window_start"] >= 60:
                    state["window_start"] = now
                    state["count"] = 0
                state["count"] += 1
                if state["count"] > rate_limit_per_min:
                    return JSONResponse(status_code=429, content={"detail": "rate_limit_exceeded"})

            if require_api_key and api_key:
                provided = request.headers.get("x-api-key")
                if not provided:
                    auth_header = request.headers.get("authorization", "")
                    if auth_header.lower().startswith("bearer "):
                        provided = auth_header[7:].strip()
                if not hmac.compare_digest(provided, api_key):
                    return JSONResponse(status_code=401, content={"detail": "unauthorized"})
            elif require_api_key and not api_key:
                return JSONResponse(status_code=503, content={"detail": "api_auth_misconfigured"})

        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    configure_order_routes(order_manager)
    configure_risk_routes(risk_manager)
    configure_positions_routes(risk_manager)
    configure_config_routes(config, risk_manager, order_manager)
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

    @app.get("/health")
    async def health() -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "paper_mode": config.paper_mode,
            "timestamp": int(time.time()),
            "uptime_seconds": int(time.time()) - app.state.started_at,
        }

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
            except Exception:
                result["risk"] = {"error": "unavailable"}

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
        except Exception:
            pass

        # Alert manager
        try:
            from monitoring.alert_manager import AlertManager
            am = getattr(app.state, "alert_manager", None)
            if am is not None and isinstance(am, AlertManager):
                result["alerts"] = am.get_status()
        except Exception:
            pass

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
        except Exception:
            pass

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
        dex_config = config.get_value("dex") or {}
        return {
            "enabled": {
                "cex_trading": True,
                "dex_trading": dex_config.get("enabled", False),
                "uniswap": dex_config.get("uniswap", {}).get("enabled", False),
                "sushiswap": dex_config.get("sushiswap", {}).get("enabled", False),
                "pancakeswap": dex_config.get("pancakeswap", {}).get("enabled", False),
                "dydx": dex_config.get("dydx", {}).get("enabled", False),
                "ts_dex_layer": config.get_value("ts_dex_layer", "enabled", False),
                "funding_rates": config.get_value("macro", "funding_rates", {}).get("enabled", False),
                "open_interest": config.get_value("macro", "open_interest", {}).get("enabled", False),
                "vix_proxy": config.get_value("macro", "vix_proxy", {}).get("enabled", False),
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
    async def kill_switch_deactivate() -> dict[str, Any]:
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
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
            return JSONResponse(status_code=400, content={"detail": str(exc)})

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
        """Return current latency statistics: feed lag, order execution times."""
        stats: dict[str, Any] = {"feed_lag": {}, "order_latency": {}}
        if metrics and hasattr(metrics, "get_latency_stats"):
            stats = metrics.get_latency_stats()
        return stats

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
            return JSONResponse(status_code=400, content={"detail": str(exc)})
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
            return {"enabled": True, "connected": False, "error": str(exc), "positions": [], "portfolio": {}}

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

    # ── /api/reconciliation/status — startup reconciliation state ─────────
    @app.get("/api/reconciliation/status")
    async def api_reconciliation_status() -> dict[str, Any]:
        if reconciliation_result is None:
            return {"available": False, "safe_mode": False}
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

    # ── /api/trade — place a trade ────────────────────────────────────────
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
                return {"success": True, "order_id": getattr(order, "order_id", "unknown"), "paper": True}
            else:
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
            return {"success": False, "error": str(exc)}

    # ── /api/positions/close-all — close all positions ────────────────────
    @app.post("/api/positions/close-all")
    async def api_close_all() -> dict[str, Any]:
        if risk_manager is None:
            return {"success": False, "error": "risk_manager not available"}
        closed = await risk_manager.activate_kill_switch()
        risk_manager.deactivate_kill_switch()
        return {"success": True, "closed": len(closed)}

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

            # In live mode: close on exchange first
            if not config.paper_mode:
                for exc in (executors or []):
                    ex_id = getattr(exc, "exchange_id", "")
                    if ex_id != venue:
                        continue
                    client = getattr(exc, "_client", None)
                    if client is None:
                        continue
                    try:
                        rl = getattr(exc, "_rate_limiter", None)
                        if rl:
                            await rl.acquire()
                        close_side = "sell" if pos.direction == "long" else "buy"
                        t_close_start = time.monotonic()
                        # Run close order + cancel protective orders concurrently
                        close_coro = client.create_market_order(
                            symbol=symbol, side=close_side, amount=abs(pos.size),
                            params={"reduceOnly": True},
                        )
                        placer = getattr(exc, "_order_placer", None)
                        if placer:
                            async def _cancel_protective():
                                try:
                                    await placer.cancel_all_for_symbol(symbol)
                                    placer.remove_tracking(symbol)
                                except Exception:
                                    pass
                            await asyncio.gather(close_coro, _cancel_protective())
                        else:
                            await close_coro
                        close_latency = time.monotonic() - t_close_start
                        if metrics and hasattr(metrics, "record_order_latency"):
                            metrics.record_order_latency(ex_id, "close", close_latency)
                        # Invalidate cache after position change
                        _exchange_cache.pop("positions", None)
                        _exchange_cache.pop("orders", None)
                        logger.info("Exchange-side close for {} {} qty={}", venue, symbol, pos.size)
                    except Exception as exc_err:
                        logger.error("Exchange close failed for {}: {}", symbol, exc_err)
                        return {"success": False, "error": f"Exchange close failed: {exc_err}"}

            closed = await risk_manager.close_position(venue, symbol, close_price)
            if closed is None:
                return {"success": False, "error": "Position not found in risk manager"}

            return {
                "success": True,
                "symbol": symbol,
                "venue": venue,
                "exit_price": close_price,
                "realized_pnl": round(float(closed.pnl), 4),
            }
        except Exception as exc:
            logger.error("Close position error: {}", exc)
            return {"success": False, "error": str(exc)}

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

        testnet = bool(primary_cfg.get("testnet", False)) if not config.paper_mode else False
        label = "PAPER" if config.paper_mode else f"{primary_exchange.upper()} {'DEMO' if testnet else 'LIVE'}"

        return {
            "enabled": enabled,
            "mode": "paper" if config.paper_mode else "live",
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

        exchanges_cfg = config.get_value("exchanges") or {}
        enabled_venues = [
            (str(name), ex_cfg)
            for name, ex_cfg in exchanges_cfg.items()
            if isinstance(ex_cfg, dict) and ex_cfg.get("enabled", False)
        ]

        if requested_mode == "live":
            if not enabled_venues:
                return {"success": False, "error": "Enable a demo exchange before switching to live mode."}
            for venue, ex_cfg in enabled_venues:
                api_key = str(ex_cfg.get("api_key", "") or "").strip()
                api_secret = str(ex_cfg.get("api_secret", "") or "").strip()
                if not api_key or api_key.startswith("${") or not api_secret or api_secret.startswith("${"):
                    return {"success": False, "error": f"Add the {venue} demo API key and secret first."}
                if not bool(ex_cfg.get("testnet", False)):
                    return {"success": False, "error": "Runtime switch is only supported for demo/testnet venues."}

            config.paper_mode = False
            os.environ["LIVE_TRADING_CONFIRMED"] = "true"
            try:
                await _stop_paper_feed_main()
            except Exception as exc:
                logger.warning("Paper feed stop error during live switch: {}", exc)
            try:
                config.persist_runtime_overrides()
            except Exception as exc:
                logger.warning("Failed to persist live mode change: {}", exc)
            logger.info("Trading mode switched to: live demo")
            return {
                "success": True,
                "mode": "live",
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
            "ai_agent_model": str(agent_status.get("model", ai_cfg.get("model", "claude-3-5-sonnet-latest"))),
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
        tg_token = body.get("telegram_bot_token", "")
        tg_chat = body.get("telegram_chat_id", "")
        if tg_token and tg_token != "****":
            notif["bot_token"] = tg_token
        if tg_chat:
            notif["chat_id"] = tg_chat

        # ── AI agent config ──
        ai_cfg = config._data.setdefault("ai_agent", {})
        ai_cfg["enabled"] = bool(body.get("ai_agent_enabled", ai_cfg.get("enabled", True)))
        ai_cfg["provider"] = str(body.get("ai_agent_provider", ai_cfg.get("provider", "local")) or "local")
        default_model = "gpt-4o-mini" if ai_cfg["provider"] == "openai" else "claude-3-5-sonnet-latest"
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
            signal_generator.set_auto_trading(bool(body.get("auto_trading_enabled")))
        if signal_generator and hasattr(signal_generator, "configure_agent"):
            signal_generator.configure_agent(payload={
                "enabled": ai_cfg.get("enabled", True),
                "provider": ai_cfg.get("provider", "local"),
                "model": ai_cfg.get("model", "claude-3-5-sonnet-latest"),
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
                "model": ai_cfg.get("model", "claude-3-5-sonnet-latest"),
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
            return JSONResponse(status_code=400, content={"detail": str(exc)})
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
            return JSONResponse(status_code=400, content={"detail": str(exc)})

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

    # ── §3 Spec: 9-layer confirmation status ──────────────────────────────
    @app.get("/api/layers")
    async def api_layers() -> dict[str, Any]:
        """Return current state of the 9-layer confirmation pipeline."""
        layers = [
            {"id": 1, "name": "Session Filter", "description": "Trading session & killzone enforcement"},
            {"id": 2, "name": "HTF Trend", "description": "Higher-timeframe weighted agreement"},
            {"id": 3, "name": "Technical Confluence", "description": "RSI, MACD, BB, EMA alignment"},
            {"id": 4, "name": "Smart Money Concepts", "description": "BOS/CHoCH + OB/FVG zones"},
            {"id": 5, "name": "Volume Flow", "description": "Delta, CVD, VWAP deviation"},
            {"id": 6, "name": "Regime Detection", "description": "Market regime (trending/ranging/breakout)"},
            {"id": 7, "name": "ML Ensemble", "description": "Model prediction confidence"},
            {"id": 8, "name": "Signal Quality", "description": "0-100 quality score gate (min 65)"},
            {"id": 9, "name": "Risk Gate", "description": "Position sizing, DD phase, circuit breaker"},
        ]
        def _score_to_status(score: float | int | None) -> str:
            try:
                numeric = float(score if score is not None else 0)
            except (TypeError, ValueError):
                return "UNKNOWN"
            if numeric >= 70:
                return "PASS"
            if numeric >= 40:
                return "WEAK"
            return "FAIL"

        # Populate layer status from signal_generator if available
        if signal_generator is not None:
            last = getattr(signal_generator, '_last_layer_status', {}) or {}
            quality = getattr(signal_generator, '_last_quality_breakdown', {}) or {}
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

                if raw_status == "UNKNOWN" and key in fallback_scores:
                    raw_status = _score_to_status(fallback_scores.get(key))
                    if not detail:
                        detail = f"score={fallback_scores.get(key, 0)}"

                if key == "risk_gate" and raw_status == "UNKNOWN":
                    snap = risk_manager.get_risk_snapshot() if risk_manager is not None else {}
                    blocked = bool(snap.get("kill_switch_active", False) or snap.get("circuit_breaker_tripped", False))
                    raw_status = "FAIL" if blocked else "PASS"
                    if not detail:
                        detail = "risk controls ready" if not blocked else "risk controls blocking new trades"

                if raw_status == "BLOCKED":
                    raw_status = "FAIL"
                if raw_status == "UNKNOWN" and "score=0" in detail:
                    raw_status = "FAIL"

                layer["status"] = raw_status
                layer["detail"] = detail or "awaiting evaluation"
        return {"layers": layers, "total": 9}

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

    @app.get("/api/clientkey")
    async def api_clientkey() -> dict[str, Any]:
        return {"key": api_key if api_key else ""}

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
            return {"overall": "error", "components": {}, "error": str(exc)}

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
    host = api_cfg.get("host", "0.0.0.0")
    port = int(api_cfg.get("port", 8000))
    server_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    logger.info("Dashboard API starting on http://{}:{}", host, port)
    await server.serve()
