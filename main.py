#!/usr/bin/env python3
"""NUERAL-TRADER-5 — Hybrid Rust + TypeScript + Python trading engine."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from core.dispatcher import Dispatcher
from core.state_machine import OperationalState, OperationalStateMachine

from data_ingestion.cex_websocket import CEXWebSocketManager
from data_ingestion.dex_rpc import DEXRPCFeed
from data_ingestion.funding_feed import FundingRateFeed
from data_ingestion.news_feed import NewsFeed
from data_ingestion.oi_feed import OpenInterestFeed
from data_ingestion.orderbook_feed import OrderbookFeed
from data_ingestion.vix_proxy import VIXProxy
from data_ingestion.user_stream import UserDataStream

from analysis.data_manager import DataManager
from analysis.sentiment import SentimentManager

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
from monitoring.uptime_tracker import UptimeTracker

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


def _setup_logging(config: Config) -> None:
    log_dir = Path(config.get_value("system", "log_dir") or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
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


async def main() -> None:
    # REQ-STATE-001..012: operational FSM.
    # Constructed first so every subsequent stage can drive transitions
    # through the canonical boot path. Single-writer; passed by reference.
    state_machine = OperationalStateMachine()

    config = Config.get(path=os.getenv("NT_CONFIG_PATH"))
    _setup_logging(config)
    state_machine.transition_to(OperationalState.CONFIG_VALIDATED, reason="config_loaded")

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
    db = DBHandler(config)
    cache = Cache(config)
    sqlite_store = SQLiteStore()
    metrics = Metrics(config, event_bus)
    health_checker = HealthChecker(check_interval=30)
    uptime_tracker = UptimeTracker(paper_mode=config.paper_mode)
    alert_manager = build_alert_manager_from_config(config)
    alert_dispatcher = AlertDispatcher(event_bus, alert_manager)

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
    dex_feed = DEXRPCFeed(config, event_bus)
    funding_feed = FundingRateFeed(config, event_bus)
    oi_feed = OpenInterestFeed(config, event_bus)
    vix_proxy = VIXProxy(config, event_bus)
    sentiment = SentimentManager(config, event_bus)
    news_feed = NewsFeed(config, event_bus)
    orderbook_feed = OrderbookFeed(config, event_bus)

    data_manager = DataManager(config, event_bus)
    signal_gen = SignalGenerator(config, event_bus, data_manager)
    # Auto-trading defaults to ON in paper mode for immediate signal generation
    signal_gen.set_auto_trading(True)
    risk_mgr = RiskManager(config, event_bus, sqlite_store=sqlite_store)
    signal_gen.set_risk_manager(risk_mgr)  # wire for accurate position counting
    order_mgr = OrderManager(config, event_bus, risk_mgr._circuit_breaker)

    executors = create_all_executors(config, event_bus, risk_mgr)

    # Variational DEX executor (perpetual futures via RFQ)
    variational_executor = create_variational_executor(config, event_bus, risk_mgr)

    by_exchange = {getattr(executor, "exchange_id", ""): executor for executor in executors}
    smart_router = SmartOrderRouter(
        binance_executor=by_exchange.get("binance"),
        bybit_executor=by_exchange.get("bybit"),
        okx_executor=by_exchange.get("okx"),
    )
    order_mgr.attach_router(smart_router)

    telegram = TelegramNotifier(config, event_bus)

    # ── User Data Stream (live mode only) ─────────────────────────────────
    user_stream = UserDataStream(config, event_bus)

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
    # `client` is only defined inside the live-mode init block above; paper
    # mode never gets a real exchange client and there's nothing to reconcile.
    periodic_reconciler: PeriodicReconciler | None = None
    _maybe_client = locals().get("client")
    if _maybe_client:
        recon_cfg = config.get_value("monitoring", "reconciliation") or {}
        periodic_reconciler = PeriodicReconciler(
            config=config,
            risk_manager=risk_mgr,
            client=_maybe_client,
            interval_seconds=float(recon_cfg.get("interval_seconds", 300.0)),
        )

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
        audit_repo=audit_repo,
        metrics=metrics,
        periodic_reconciler=periodic_reconciler,
        state_machine=state_machine,
        uptime_tracker=uptime_tracker,
    )

    if app is not None:
        app.state.health_checker = health_checker
        app.state.alert_manager = alert_manager

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

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

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
    )
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

    # ── FSM: boot path → SIGNALING_ACTIVE → EXECUTION_ENABLED ─────────────
    # WARMUP / market-data / order-book stages are visited synchronously here
    # before background tasks run. The actual WS connect+sync happens in the
    # paper_feed/orderbook_feed tasks below; advancing to MARKET_DATA_LIVE
    # there is a future enhancement. For now we record the structural
    # progression that we know is true at this point.
    state_machine.transition_to(OperationalState.WARMUP, reason="hydrating_indicators")
    state_machine.transition_to(OperationalState.CONNECTING_MARKET_DATA, reason="ws_subscribe")
    state_machine.transition_to(OperationalState.MARKET_DATA_LIVE, reason="paper_feed_ready")
    state_machine.transition_to(OperationalState.ORDER_BOOK_SYNCED, reason="orderbook_feed_ready")
    state_machine.transition_to(OperationalState.SIGNALING_ACTIVE, reason="signal_generator_active")
    # Reflect the active safety primitives now — execution is ENABLED unless
    # the kill switch / safe mode / circuit breaker says otherwise.
    sm = getattr(risk_mgr, "safe_mode", None)
    cb_open = bool(getattr(getattr(risk_mgr, "_circuit_breaker", None), "is_open", False))
    if sm is not None and getattr(sm, "is_active", False):
        state_machine.transition_to(OperationalState.SAFE_MODE, reason="safe_mode_active_at_boot")
    elif cb_open:
        state_machine.transition_to(OperationalState.CIRCUIT_BREAKER_ACTIVE, reason="cb_open_at_boot")
    else:
        state_machine.transition_to(OperationalState.EXECUTION_ENABLED, reason="all_gates_clear")

    # Bridge SafeModeManager into the FSM. SafeMode is multi-reason; we map
    # ANY active reason → SAFE_MODE, all-clear → back to EXECUTION_ENABLED.
    if sm is not None:
        _orig_activate = sm.activate
        _orig_deactivate = sm.deactivate

        def _bridge_activate(reason, detail=""):  # type: ignore[no-untyped-def]
            result = _orig_activate(reason, detail)
            try:
                if state_machine.current is not OperationalState.SAFE_MODE:
                    state_machine.force_to(
                        OperationalState.SAFE_MODE,
                        reason=f"safe_mode:{getattr(reason, 'value', reason)}",
                    )
            except Exception as exc:
                logger.warning("FSM bridge SafeMode.activate failed: {}", exc)
            return result

        def _bridge_deactivate(reason):  # type: ignore[no-untyped-def]
            result = _orig_deactivate(reason)
            try:
                if not sm.is_active and state_machine.current is OperationalState.SAFE_MODE:
                    state_machine.transition_to(
                        OperationalState.EXECUTION_ENABLED,
                        reason=f"safe_mode_cleared:{getattr(reason, 'value', reason)}",
                    )
            except Exception as exc:
                logger.warning("FSM bridge SafeMode.deactivate failed: {}", exc)
            return result

        sm.activate = _bridge_activate  # type: ignore[method-assign]
        sm.deactivate = _bridge_deactivate  # type: ignore[method-assign]

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
        asyncio.create_task(orderbook_feed.run(), name="orderbook_feed"),
        asyncio.create_task(telegram.run(), name="telegram"),
        asyncio.create_task(order_mgr.run(), name="order_manager"),
        asyncio.create_task(user_stream.run(), name="user_data_stream"),
        asyncio.create_task(_periodic_liq_check(risk_mgr, stop_event), name="liq_check"),
        asyncio.create_task(_periodic_funding_check(risk_mgr, stop_event), name="funding_recheck"),
        asyncio.create_task(health_checker.start_periodic_checks(), name="health_checker"),
        asyncio.create_task(alert_dispatcher.run(), name="alert_dispatcher"),
    ]

    if model_retrainer is not None:
        tasks.append(asyncio.create_task(model_retrainer.run(), name="model_retrainer"))

    tasks.append(asyncio.create_task(uptime_tracker.run(), name="uptime_tracker"))

    if periodic_reconciler is not None:
        tasks.append(asyncio.create_task(periodic_reconciler.run(), name="periodic_reconciler"))

    for executor in executors:
        tasks.append(asyncio.create_task(executor.run(), name=f"exec_{executor.exchange_id}"))

    if variational_executor is not None:
        tasks.append(asyncio.create_task(variational_executor.run(), name="exec_variational"))

    if app is not None:
        tasks.append(asyncio.create_task(run_dashboard(config, app), name="dashboard"))

    await stop_event.wait()
    logger.info("Initiating graceful shutdown…")

    # ── Graceful shutdown: cancel open orders on exchange ──────────────────
    for executor in executors:
        if not config.paper_mode and hasattr(executor, "_client") and executor._client:
            try:
                open_orders = await executor._client.fetch_open_orders()
                for o in open_orders:
                    try:
                        await executor._client.cancel_order(o["id"], o.get("symbol"))
                    except Exception:
                        pass
                if open_orders:
                    logger.info(
                        "Shutdown: cancelled {} open orders on {}",
                        len(open_orders), executor.exchange_id,
                    )
            except Exception as exc:
                logger.warning("Shutdown order cancel failed on {}: {}", executor.exchange_id, exc)

    # Close executors (closes ccxt client)
    for executor in executors:
        await executor.close()

    # Stop user data stream
    await user_stream.stop()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    await dispatcher.stop()
    await ws_manager.stop()
    await funding_feed.stop()
    await oi_feed.stop()
    await sentiment.stop()
    await news_feed.stop()
    await orderbook_feed.stop()
    await telegram.stop()
    await order_mgr.stop()
    await alert_dispatcher.stop()
    await health_checker.stop_periodic_checks()
    await db.close()

    uptime_tracker.stop(clean=True)
    state_machine.force_to(OperationalState.SHUTDOWN, reason="graceful_shutdown")
    logger.info("Shutdown complete")


if __name__ == "__main__":
    _configure_event_loop()
    asyncio.run(main())
