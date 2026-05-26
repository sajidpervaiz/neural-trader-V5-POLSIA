#!/usr/bin/env python3
"""NUERAL-TRADER-5 — Hybrid Rust + TypeScript + Python trading engine."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Load .env (gitignored secrets) before any module reads os.environ.
# python-dotenv is in requirements.txt; if it's missing, fall back to a
# tiny inline parser so the bot still boots.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from loguru import logger

from core.logging_utils import configure_sensitive_logging_redaction

configure_sensitive_logging_redaction()

from core.config import Config
from core.event_bus import EventBus
from core.dispatcher import Dispatcher

from data_ingestion.cex_websocket import CEXWebSocketManager
from data_ingestion.dex_rpc import DEXRPCFeed
from data_ingestion.funding_feed import FundingRateFeed
from data_ingestion.news_feed import NewsFeed
from data_ingestion.geopolitical_news import GeoPoliticalNewsFeed
from data_ingestion.oi_feed import OpenInterestFeed
from data_ingestion.orderbook_feed import OrderbookFeed
from data_ingestion.market_data_integrity import MarketDataIntegrityMonitor
from data_ingestion.vix_proxy import VIXProxy
from data_ingestion.user_stream import UserDataStream

from analysis.data_manager import DataManager
from analysis.sentiment import SentimentManager

from engine.geopolitical_scorer import GeoPoliticalScorer
from engine.signal_generator import SignalGenerator

from execution.risk_manager import RiskManager
from execution.order_manager import OrderManager
from execution.exchange_factory import create_all_executors, create_variational_executor
from execution.smart_order_router import SmartOrderRouter
from execution.startup_validation import StartupValidator, ValidationError
from execution.reconciliation import StartupReconciler, PeriodicReconciler

from storage.db_handler import DBHandler
from storage.cache import Cache
from storage.trade_persistence import TradePersistence
from storage.audit_repository import AuditRepository
from storage.audit_event_persistence import AuditEventPersistence
from storage.state_recovery import StateRecovery
from storage.sqlite_store import SQLiteStore

from monitoring.metrics import Metrics
from monitoring.health_checks import HealthChecker
from monitoring.alert_manager import build_alert_manager_from_config, AlertDispatcher

from interface.dashboard_api import build_app, run_dashboard
from interface.telegram_bot import TelegramNotifier


def _configure_event_loop() -> None:
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("uvloop event loop policy activated")
    except ImportError:
        logger.debug("uvloop not available — using default asyncio event loop")


def _requires_live_confirmation(config: Config) -> bool:
    if config.paper_mode:
        return False

    exchanges = config.get_value("exchanges", default={}) or {}
    enabled_venues = [
        venue_cfg
        for venue_cfg in exchanges.values()
        if isinstance(venue_cfg, dict) and venue_cfg.get("enabled", False)
    ]
    if not enabled_venues:
        return True

    return not all(bool(venue_cfg.get("testnet", False)) for venue_cfg in enabled_venues)


def _enforce_live_audit_db_available(
    config: Config,
    db: DBHandler,
    sqlite_store: SQLiteStore | None = None,
) -> None:
    """Fail closed in live mode unless durable audit persistence is explicit."""
    if config.paper_mode:
        return
    if getattr(db, "available", False):
        return
    sqlite_personal = bool(config.get_value("storage", "personal_sqlite_mode", default=False))
    sqlite_ok = bool(sqlite_store is not None and getattr(sqlite_store, "available", False))
    if sqlite_personal and sqlite_ok:
        logger.warning(
            "PostgreSQL audit DB unavailable; explicit personal SQLite mode is active at {}. "
            "This is durable local persistence for personal use, not clustered HA storage.",
            getattr(sqlite_store, "path", "data/neural_trader.db"),
        )
        return
    logger.critical(
        "LIVE TRADING BLOCKED: PostgreSQL audit database is unavailable. "
        "Set storage.personal_sqlite_mode=true only if you intentionally accept local SQLite persistence."
    )
    sys.exit(1)


def _setup_logging(config: Config) -> None:
    log_dir = Path(config.get_value("system", "log_dir") or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    configure_sensitive_logging_redaction()
    logger.add(
        sys.stdout,
        level=config.log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
    )
    logger.add(
        log_dir / "neural_trader_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="gz",
    )


async def _run_bot(components: dict[str, Any]) -> None:
    config = Config.get(path=os.getenv("NT_CONFIG_PATH"))
    _setup_logging(config)
    components["paper_mode"] = config.paper_mode

    logger.info("=" * 60)
    logger.info("  NUERAL-TRADER-5  |  paper_mode={}", config.paper_mode)
    logger.info("=" * 60)

    # ── Live mode safety gate ─────────────────────────────────────────────
    if not config.paper_mode:
        if _requires_live_confirmation(config):
            live_confirm = os.getenv("LIVE_TRADING_CONFIRMED", "").lower()
            if live_confirm != "true":
                logger.critical(
                    "LIVE TRADING requires LIVE_TRADING_CONFIRMED=true env var. "
                    "Set it explicitly to acknowledge real-money risk."
                )
                sys.exit(1)
        else:
            logger.warning("Starting in demo or testnet live mode — real-money confirmation not required")

    event_bus = EventBus()
    # Pass event_bus so DB connection failures emit ALERT_CRITICAL into the
    # audit pipeline instead of silently turning off persistence.
    db = DBHandler(config, event_bus=event_bus)
    components["db"] = db
    cache = Cache(config)
    storage_cfg = config.get_value("storage", default={}) or {}
    sqlite_cfg = storage_cfg.get("sqlite", {}) if isinstance(storage_cfg, dict) else {}
    sqlite_store = SQLiteStore(sqlite_cfg.get("path", "data/neural_trader.db"))
    metrics = Metrics(config, event_bus)
    health_checker = HealthChecker(check_interval=30)
    components["health_checker"] = health_checker
    alert_manager = build_alert_manager_from_config(config)
    alert_dispatcher = AlertDispatcher(event_bus, alert_manager)
    components["alert_dispatcher"] = alert_dispatcher

    # ── Persist candles to SQLite on each CANDLE event ────────────────────
    async def _persist_candle(candle: Any) -> None:
        try:
            sqlite_store.insert_candle(
                exchange=getattr(candle, "exchange", "binance"),
                symbol=getattr(candle, "symbol", ""),
                timeframe=getattr(candle, "timeframe", ""),
                o=getattr(candle, "open", 0),
                h=getattr(candle, "high", 0),
                l=getattr(candle, "low", 0),
                c=getattr(candle, "close", 0),
                v=getattr(candle, "volume", 0),
                ts_ns=int(getattr(candle, "timestamp", 0) * 1e9),
            )
        except Exception as exc:
            logger.error("Failed to persist candle: {}", exc)
    event_bus.subscribe("CANDLE", _persist_candle)
    # ── Database ──────────────────────────────────────────────────────────
    await db.connect()
    _enforce_live_audit_db_available(config, db, sqlite_store)

    # ── Trade persistence (production audit trail) ────────────────────────
    trade_persistence: TradePersistence | None = None
    audit_repo: AuditRepository | None = None
    if db.available:
        trade_persistence = TradePersistence(db._pool, event_bus, is_paper=config.paper_mode)
        await trade_persistence.migrate()
        trade_persistence.subscribe_events()

        # Audit repository + event wiring (signals, risk, user stream, recon, errors)
        audit_repo = AuditRepository(db._pool)
        audit_events = AuditEventPersistence(audit_repo, event_bus)
        audit_events.subscribe_all()

        # DB state recovery — rebuild OrderManager state from DB
        recovery = StateRecovery(audit_repo)
        recovery_result = await recovery.recover()
        if recovery_result.safe_mode:
            logger.warning("Recovery detected safe_mode from last reconciliation")

        logger.info(
            "Audit trail initialized — recovered {} orders, {} positions",
            recovery_result.orders_recovered, recovery_result.positions_recovered,
        )

    ws_manager = CEXWebSocketManager(config, event_bus)
    components["ws_manager"] = ws_manager
    dex_feed = DEXRPCFeed(config, event_bus)
    components["dex_feed"] = dex_feed
    funding_feed = FundingRateFeed(config, event_bus)
    components["funding_feed"] = funding_feed
    oi_feed = OpenInterestFeed(config, event_bus)
    components["oi_feed"] = oi_feed
    vix_proxy = VIXProxy(config, event_bus)
    components["vix_proxy"] = vix_proxy
    sentiment = SentimentManager(config, event_bus)
    components["sentiment"] = sentiment
    news_feed = NewsFeed(config, event_bus)
    components["news_feed"] = news_feed
    orderbook_feed = OrderbookFeed(config, event_bus)
    components["orderbook_feed"] = orderbook_feed
    market_data_integrity = MarketDataIntegrityMonitor(config, event_bus)
    components["market_data_integrity"] = market_data_integrity

    # Geopolitical scorer + RSS feed (Layer 10 additive contributor — opt-in via
    # signals.geopolitical_weight; defaults to 0.0 so existing flows are unchanged).
    geopolitical_scorer = GeoPoliticalScorer()
    # Map every traded-symbol variant the bot may see (spot BTC/USDT, perp BTCUSDT,
    # exchange-prefixed binance:BTC/USDT:USDT, etc.) to its canonical config key.
    _geo_canonicals = list(geopolitical_scorer.symbols)
    for canonical in _geo_canonicals:
        base = canonical.split("/")[0]
        for alias in {
            canonical.replace(":USDT", ""),  # BTC/USDT
            canonical.replace("/", "").replace(":USDT", ""),  # BTCUSDT
            f"{base}/USDT",
            f"{base}USDT",
            f"binance:{canonical}",
            f"bybit:{canonical}",
        }:
            geopolitical_scorer.add_alias(alias, canonical)
    components["geopolitical_scorer"] = geopolitical_scorer
    geopolitical_feed = GeoPoliticalNewsFeed(config, event_bus, geopolitical_scorer)
    components["geopolitical_feed"] = geopolitical_feed

    data_manager = DataManager(config, event_bus)
    signal_gen = SignalGenerator(config, event_bus, data_manager)
    signal_gen.set_geopolitical_scorer(geopolitical_scorer)
    # Live mode: require explicit operator opt-in via /api/auto/toggle.
    # Paper mode: auto-enable for instant signal flow during dev/testing.
    signal_gen.set_auto_trading(bool(config.paper_mode))
    if not config.paper_mode:
        logger.warning(
            "LIVE MODE: auto-trading disabled at boot. "
            "POST /api/auto/toggle {{\"enabled\": true}} to start trading.",
        )
    risk_mgr = RiskManager(config, event_bus, sqlite_store=sqlite_store)
    sqlite_recovery_result: dict[str, Any] | None = None
    restore_positions = bool(
        config.get_value("execution", "restore_positions_on_startup", default=True)
    )
    if restore_positions and getattr(sqlite_store, "available", False):
        try:
            sqlite_recovery_result = await risk_mgr.restore_open_positions_from_sqlite()
            components["sqlite_recovery_result"] = sqlite_recovery_result
            logger.info(
                "SQLite position recovery: restored={} skipped={} errors={}",
                sqlite_recovery_result.get("restored", 0),
                sqlite_recovery_result.get("skipped", 0),
                len(sqlite_recovery_result.get("errors", []) or []),
            )
        except Exception as exc:
            sqlite_recovery_result = {
                "source": "sqlite",
                "success": False,
                "error": str(exc),
                "attempted": 0,
                "restored": 0,
            }
            components["sqlite_recovery_result"] = sqlite_recovery_result
            logger.error("SQLite position recovery failed: {}", exc)
    else:
        sqlite_recovery_result = {
            "source": "sqlite",
            "success": True,
            "disabled": True,
            "attempted": 0,
            "restored": 0,
        }
        components["sqlite_recovery_result"] = sqlite_recovery_result
    signal_gen.set_risk_manager(risk_mgr)  # wire for accurate position counting
    signal_gen.set_market_data_integrity(market_data_integrity)
    order_mgr = OrderManager(config, event_bus, risk_mgr._circuit_breaker)
    components["order_mgr"] = order_mgr

    executors = create_all_executors(config, event_bus, risk_mgr, order_manager=order_mgr)
    components["executors"] = executors

    # Variational DEX executor (perpetual futures via RFQ)
    variational_executor = create_variational_executor(config, event_bus, risk_mgr)
    if variational_executor is not None:
        components["variational_executor"] = variational_executor

    by_exchange = {getattr(executor, "exchange_id", ""): executor for executor in executors}
    smart_router = SmartOrderRouter(
        binance_executor=by_exchange.get("binance"),
        bybit_executor=by_exchange.get("bybit"),
        okx_executor=by_exchange.get("okx"),
    )
    order_mgr.attach_router(smart_router)

    telegram = TelegramNotifier(config, event_bus)
    components["telegram"] = telegram

    # ── User Data Stream (live mode only) ─────────────────────────────────
    user_stream = UserDataStream(config, event_bus)
    components["user_stream"] = user_stream

    # ── Pre-trade validation (live mode only) ─────────────────────────────
    recon_result = None
    if not config.paper_mode:
        binance_executor = by_exchange.get("binance")
        client = getattr(binance_executor, "_client", None) if binance_executor else None

        # Initialize client early for validation
        if binance_executor and client is None:
            await binance_executor._init_client()
            client = getattr(binance_executor, "_client", None)

        # Step 1: Startup validation (API keys, balance, clock, permissions,
        #         leverage, margin mode, symbol specs, order feasibility)
        try:
            validator = StartupValidator(config, client=client)
            validation_result = await validator.validate_all()
            logger.info("Startup validation: {}", validation_result.get("checks", {}))
            for w in validation_result.get("warnings", []):
                logger.warning("Startup warning: {}", w)
        except ValidationError as exc:
            logger.critical("STARTUP VALIDATION FAILED: {}", exc)
            logger.critical("Bot cannot start in live mode. Fix the issue and restart.")
            sys.exit(1)

        # Step 2: Reconciliation (sync state with exchange)
        if client:
            order_placer = getattr(binance_executor, "_order_placer", None)
            reconciler = StartupReconciler(
                config=config,
                event_bus=event_bus,
                risk_manager=risk_mgr,
                client=client,
                order_placer=order_placer,
                trade_persistence=trade_persistence,
                order_manager=order_mgr,
            )
            recon_result = await reconciler.reconcile()
            if recon_result.safe_mode:
                logger.critical(
                    "SAFE MODE: {} mismatch(es) detected — no new entries until resolved",
                    len(recon_result.mismatches),
                )
            logger.info("Reconciliation result: {}", recon_result)

    # REQ-POS-004 / REQ-FS-007: lightweight periodic re-check during the run.
    # Distinct from StartupReconciler — it just diffs exchange vs internal
    # positions every 5 min and trips SafeMode on mismatch (no state rebuild).
    periodic_reconciler: PeriodicReconciler | None = None
    # `client` is only defined inside the live-mode init block above; paper
    # mode never gets a real exchange client and there's nothing to reconcile.
    _maybe_client = locals().get("client")
    if _maybe_client:
        recon_cfg = config.get_value("monitoring", "reconciliation") or {}
        periodic_reconciler = PeriodicReconciler(
            config=config,
            risk_manager=risk_mgr,
            client=_maybe_client,
            interval_seconds=float(recon_cfg.get("interval_seconds", 300.0)),
        )
        components["periodic_reconciler"] = periodic_reconciler

    app = build_app(
        config, event_bus, risk_mgr, data_manager, order_mgr, db, cache, signal_gen,
        news_feed=news_feed,
        orderbook_feed=orderbook_feed,
        sentiment_manager=sentiment,
        dex_feed=dex_feed,
        executors=executors,
        user_stream=user_stream,
        reconciliation_result=recon_result,
        sqlite_store=sqlite_store,
        metrics=metrics,
        geopolitical_feed=geopolitical_feed,
        periodic_reconciler=periodic_reconciler,
        market_data_integrity=market_data_integrity,
    )

    if app is not None:
        app.state.health_checker = health_checker
        app.state.alert_manager = alert_manager
        app.state.sqlite_recovery_result = sqlite_recovery_result

    # ── ModelRetrainer (periodic background retrain of the ML model) ──────
    model_retrainer = None
    try:
        from engine.model_retrainer import ModelRetrainer
        model_retrainer = ModelRetrainer(
            ml_scorer=signal_gen._ml_scorer,
            alert_manager=alert_manager,
            config=config,
        )
        if app is not None:
            app.state.model_retrainer = model_retrainer
    except Exception as exc:
        logger.warning("ModelRetrainer init failed: {}", exc)

    if app is not None:
        app.state.signal_generator = signal_gen

    # Re-add the dashboard log sink (logger.remove() in _setup_logging wipes it)
    from interface.dashboard_api import _log_sink
    logger.add(_log_sink, level="INFO", format="{message}")

    dispatcher = Dispatcher(
        config=config,
        event_bus=event_bus,
        data_manager=data_manager,
        signal_generator=signal_gen,
        risk_manager=risk_mgr,
        db_handler=db,
        cache=cache,
        metrics=metrics,
    )
    components["dispatcher"] = dispatcher

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows event loops do not implement add_signal_handler().
            signal.signal(sig, _signal_handler)

    # ── ARMS-V2.1: Periodic risk tasks ────────────────────────────────────
    async def _periodic_liq_check(rm: RiskManager, stop_ev: asyncio.Event) -> None:
        """Run liquidation distance check every 60s."""
        while not stop_ev.is_set():
            try:
                actions = rm.run_periodic_liq_check()
                for action in actions:
                    logger.warning("Liq check action: {}", action)
                    await event_bus.publish("LIQ_CHECK_ACTION", action)
            except Exception as exc:
                logger.error("Periodic liq check failed: {}", exc)
            try:
                await asyncio.wait_for(stop_ev.wait(), timeout=60.0)
                break
            except asyncio.TimeoutError:
                pass

    async def _midnight_daily_reset(rm: RiskManager, stop_ev: asyncio.Event) -> None:
        """Reset daily-loss counter at 00:00 UTC. Industry-standard safety pattern."""
        import datetime as _dt
        while not stop_ev.is_set():
            now = _dt.datetime.now(_dt.timezone.utc)
            tomorrow = (now + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_s = max(60.0, (tomorrow - now).total_seconds())
            try:
                await asyncio.wait_for(stop_ev.wait(), timeout=sleep_s)
                break
            except asyncio.TimeoutError:
                pass
            try:
                rm.reset_daily_losses()
            except Exception as exc:
                logger.error("Midnight daily reset failed: {}", exc)

    async def _periodic_funding_check(rm: RiskManager, stop_ev: asyncio.Event) -> None:
        """Run funding re-check for existing positions every 8h."""
        while not stop_ev.is_set():
            try:
                actions = rm.check_funding_existing_positions()
                for action in actions:
                    logger.warning("Funding recheck action: {}", action)
                    await event_bus.publish("FUNDING_RECHECK_ACTION", action)
            except Exception as exc:
                logger.error("Periodic funding check failed: {}", exc)
            try:
                await asyncio.wait_for(stop_ev.wait(), timeout=8 * 3600.0)
                break
            except asyncio.TimeoutError:
                pass

    # ── Startup historical data seed (populates regimes, indicators, etc.) ──
    # In paper mode, PaperFeed seeds + polls continuously (no live WS feed).
    # In live mode, PaperFeed seeds only — live data comes from CEX WebSocket.
    from data_ingestion.paper_feed import PaperFeed
    paper_feed = PaperFeed(
        event_bus=event_bus,
        symbols=config.get_value("exchanges", "binance", "symbols") or ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
        data_manager=data_manager,
        market_data_integrity=market_data_integrity,
    )
    components["paper_feed"] = paper_feed
    _seed_only = not config.paper_mode  # live mode: seed only, WS provides data

    # ── Health check component registrations ──────────────────────────────
    async def _check_event_bus() -> bool:
        return not getattr(event_bus, "_stopped", False)

    async def _check_signal_generator() -> dict:
        return {
            "status": "HEALTHY",
            "running": bool(getattr(signal_gen, "_running", False)),
            "auto_trading": bool(getattr(signal_gen, "_auto_trading_enabled", False)),
        }

    async def _check_risk_manager() -> dict:
        cb = getattr(risk_mgr, "_circuit_breaker", None)
        return {
            "status": "HEALTHY",
            "equity": float(getattr(risk_mgr, "equity", 0)),
            "open_positions": len(getattr(risk_mgr, "_positions", {}) or {}),
            "circuit_breaker_open": bool(getattr(cb, "is_open", False)) if cb else False,
        }

    async def _check_data_manager() -> dict:
        aggs = getattr(data_manager, "_aggregators", {}) or {}
        return {"status": "HEALTHY", "symbols_tracked": len(aggs)}

    await health_checker.register_component("event_bus", _check_event_bus)
    await health_checker.register_component("signal_generator", _check_signal_generator)
    await health_checker.register_component("risk_manager", _check_risk_manager)
    await health_checker.register_component("data_manager", _check_data_manager)

    tasks = [
        asyncio.create_task(paper_feed.run(seed_only=_seed_only), name="paper_feed"),
        asyncio.create_task(dispatcher.start(), name="dispatcher"),
        asyncio.create_task(ws_manager.run(), name="cex_ws"),
        asyncio.create_task(dex_feed.run(), name="dex_rpc"),
        asyncio.create_task(funding_feed.run(), name="funding"),
        asyncio.create_task(oi_feed.run(), name="oi"),
        asyncio.create_task(vix_proxy.run(), name="vix"),
        asyncio.create_task(sentiment.run(), name="sentiment"),
        asyncio.create_task(news_feed.run(), name="news_feed"),
        asyncio.create_task(geopolitical_feed.run(), name="geopolitical_feed"),
        asyncio.create_task(orderbook_feed.run(), name="orderbook_feed"),
        asyncio.create_task(market_data_integrity.run(), name="market_data_integrity"),
        asyncio.create_task(telegram.run(), name="telegram"),
        asyncio.create_task(order_mgr.run(), name="order_manager"),
        asyncio.create_task(metrics.run(), name="metrics"),
        asyncio.create_task(user_stream.run(), name="user_data_stream"),
        asyncio.create_task(_periodic_liq_check(risk_mgr, stop_event), name="liq_check"),
        asyncio.create_task(_periodic_funding_check(risk_mgr, stop_event), name="funding_recheck"),
        asyncio.create_task(_midnight_daily_reset(risk_mgr, stop_event), name="midnight_reset"),
        asyncio.create_task(health_checker.start_periodic_checks(), name="health_checker"),
        asyncio.create_task(alert_dispatcher.run(), name="alert_dispatcher"),
    ]

    if model_retrainer is not None:
        tasks.append(asyncio.create_task(model_retrainer.run(), name="model_retrainer"))

    if periodic_reconciler is not None:
        tasks.append(asyncio.create_task(periodic_reconciler.run(), name="periodic_reconciler"))

    for executor in executors:
        tasks.append(asyncio.create_task(executor.run(), name=f"exec_{executor.exchange_id}"))

    if variational_executor is not None:
        tasks.append(asyncio.create_task(variational_executor.run(), name="exec_variational"))

    if app is not None:
        tasks.append(asyncio.create_task(run_dashboard(config, app), name="dashboard"))

    components["tasks"] = tasks

    await stop_event.wait()
    logger.info("Initiating graceful shutdown…")
    # Component teardown is performed in main()'s finally block (_final_cleanup),
    # which also runs on partial-state startup crashes so aiohttp ClientSessions
    # are always closed.


async def _safe_call(label: str, fn: Any) -> None:
    if fn is None:
        return
    try:
        result = fn()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        logger.warning("Shutdown {} failed: {}", label, exc)


async def _final_cleanup(components: dict[str, Any]) -> None:
    """Idempotent teardown — runs on graceful shutdown AND startup-crash paths."""
    paper_mode = bool(components.get("paper_mode", True))
    tasks = components.get("tasks") or []
    executors = components.get("executors") or []

    # 1. Cancel open exchange orders (live mode only).
    if not paper_mode:
        for executor in executors:
            client = getattr(executor, "_client", None)
            if not client:
                continue
            try:
                open_orders = await client.fetch_open_orders()
                for o in open_orders:
                    try:
                        await client.cancel_order(o["id"], o.get("symbol"))
                    except Exception:
                        pass
                if open_orders:
                    logger.info(
                        "Shutdown: cancelled {} open orders on {}",
                        len(open_orders), getattr(executor, "exchange_id", "?"),
                    )
            except Exception as exc:
                logger.warning(
                    "Shutdown order cancel failed on {}: {}",
                    getattr(executor, "exchange_id", "?"), exc,
                )

    # 2. Close executors (closes ccxt client + aiohttp session).
    for executor in executors:
        await _safe_call(f"exec_{getattr(executor, 'exchange_id', '?')}_close",
                         getattr(executor, "close", None))
    variational = components.get("variational_executor")
    if variational is not None:
        await _safe_call("variational_executor_close", getattr(variational, "close", None))

    # 3. Stop user data stream before cancelling tasks (releases listenKey).
    user_stream = components.get("user_stream")
    if user_stream is not None:
        await _safe_call("user_stream", getattr(user_stream, "stop", None))

    # 4. Cancel background tasks and await their completion.
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            logger.warning("Shutdown task gather failed: {}", exc)

    # 5. Stop remaining components in the same order as the previous graceful path.
    cleanup_order = [
        "dispatcher", "ws_manager", "funding_feed", "oi_feed", "sentiment",
        "news_feed", "geopolitical_feed", "orderbook_feed", "market_data_integrity", "dex_feed", "vix_proxy", "telegram",
        "order_mgr", "alert_dispatcher", "paper_feed",
    ]
    for name in cleanup_order:
        comp = components.get(name)
        if comp is None:
            continue
        await _safe_call(name, getattr(comp, "stop", None))

    # 6. Health checker uses a different stop method name.
    hc = components.get("health_checker")
    if hc is not None:
        await _safe_call("health_checker", getattr(hc, "stop_periodic_checks", None))

    # 7. DB last so persistence calls during teardown still succeed.
    db = components.get("db")
    if db is not None:
        await _safe_call("db", getattr(db, "close", None))


async def main() -> None:
    components: dict[str, Any] = {}
    try:
        await _run_bot(components)
    finally:
        await _final_cleanup(components)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    _configure_event_loop()
    asyncio.run(main())
