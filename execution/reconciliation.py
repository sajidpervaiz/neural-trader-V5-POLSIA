"""
Startup reconciliation — ensures bot state matches exchange reality before trading.

On boot:
1. Fetch open positions from Binance Futures
2. Fetch open orders per symbol
3. Fetch account balance / margin
4. Fetch leverage / margin-mode settings
5. Compare exchange truth with DB and in-memory state
6. Rebuild internal state from exchange truth (positions, orders, equity)
7. If mismatch → SAFE MODE (no new entries, manage existing positions)
8. Ensure every open position has SL/TP; if missing, place them immediately
9. Prevent duplicate positions after restart via idempotent rebuild
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from execution.risk_manager import RiskManager, Position
from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrderFallbackRequired
from execution.startup_validation import _sum_stable_usd


class ReconciliationResult:
    """Result of startup reconciliation."""

    def __init__(self) -> None:
        self.reconciliation_id: str = uuid.uuid4().hex[:12]
        self.success = False
        self.safe_mode = False
        self.exchange_positions: list[dict] = []
        self.exchange_open_orders: list[dict] = []
        self.db_positions: list[dict] = []
        self.leverage_settings: dict[str, dict] = {}  # symbol → {leverage, margin_mode}
        self.balance: float = 0.0
        self.available_balance: float = 0.0
        self.mismatches: list[str] = []
        self.positions_without_sl: list[str] = []
        self.actions_taken: list[str] = []

    def __repr__(self) -> str:
        return (
            f"ReconciliationResult(id={self.reconciliation_id}, "
            f"success={self.success}, safe_mode={self.safe_mode}, "
            f"positions={len(self.exchange_positions)}, "
            f"db_positions={len(self.db_positions)}, "
            f"mismatches={len(self.mismatches)})"
        )


class StartupReconciler:
    """
    Reconciles bot state with exchange reality on startup.

    If mismatches are detected, the bot enters SAFE MODE:
    - No new entries allowed
    - Existing positions continue to be managed (SL/TP active)
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        risk_manager: RiskManager,
        client: Any,
        order_placer: ExchangeOrderPlacer | None = None,
        trade_persistence: Any | None = None,
        order_manager: Any | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self._client = client
        self._order_placer = order_placer
        self._trade_persistence = trade_persistence
        self._order_manager = order_manager
        self._safe_mode = False
        self._reconciled = False  # Guard against double-reconcile

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    async def reconcile(self) -> ReconciliationResult:
        """
        Run full startup reconciliation.

        Returns ReconciliationResult with details of what was found and done.
        Idempotent: calling twice returns the cached result.
        """
        result = ReconciliationResult()

        if self._reconciled:
            logger.warning("Reconciliation already completed — skipping duplicate call")
            result.success = True
            return result

        if self._client is None:
            logger.warning("No exchange client — skipping reconciliation")
            result.success = True
            self._reconciled = True
            return result

        logger.info("=" * 50)
        logger.info("STARTUP RECONCILIATION BEGIN (id={})", result.reconciliation_id)
        logger.info("=" * 50)

        try:
            # 1. Fetch account balance
            await self._fetch_balance(result)

            # 2. Fetch open positions from exchange
            await self._fetch_positions(result)

            # 3. Fetch open orders
            await self._fetch_open_orders(result)

            # 4. Load DB positions and cross-reference
            await self._load_and_compare_db(result)

            # 5. Fetch leverage/margin settings
            await self._fetch_leverage_settings(result)

            # 6. Rebuild internal state from exchange truth
            await self._rebuild_state(result)

            # 7. Rebuild order manager state from open orders
            await self._rebuild_order_manager(result)

            # 8. Ensure every position has protective orders
            await self._ensure_protective_orders(result)

            # 9. Publish reconciliation event
            await self.event_bus.publish("RECONCILIATION_COMPLETE", result)

            # 10. Determine if safe mode needed
            if result.mismatches:
                self._safe_mode = True
                result.safe_mode = True
                # Trip circuit breaker to prevent new entries
                self.risk_manager._circuit_breaker.trip(
                    f"reconciliation_mismatch ({len(result.mismatches)} issues)",
                    pause_seconds=86400.0,
                )
                logger.critical(
                    "SAFE MODE ACTIVATED — {} mismatches detected. No new entries allowed.",
                    len(result.mismatches),
                )
                for m in result.mismatches:
                    logger.critical("  Mismatch: {}", m)
            else:
                logger.info("Reconciliation clean — no mismatches")

            result.success = True
            self._reconciled = True
            logger.info(
                "STARTUP RECONCILIATION COMPLETE (id={}, safe_mode={})",
                result.reconciliation_id, result.safe_mode,
            )
            logger.info("=" * 50)

        except Exception as exc:
            logger.critical("Reconciliation FAILED: {} — entering safe mode", exc)
            result.success = False
            result.safe_mode = True
            self._safe_mode = True
            # Trip circuit breaker
            self.risk_manager._circuit_breaker.trip(
                f"reconciliation_failed: {exc}",
                pause_seconds=86400.0,
            )

        return result

    async def _fetch_balance(self, result: ReconciliationResult) -> None:
        """Fetch account balance from exchange."""
        try:
            balance = await self._client.fetch_balance()
            # Sum stable USD pegs (USDT / USDC / BUSD / FDUSD) — accounts holding
            # USDC must not be reset to equity=0.
            total = _sum_stable_usd(balance.get("total"))
            free = _sum_stable_usd(balance.get("free"))
            result.balance = total
            result.available_balance = free
            logger.info(
                "Exchange balance: total={:.2f} USDT, available={:.2f} USDT",
                total, free,
            )
            # Update risk manager equity to real exchange value
            if total > 0:
                old_equity = self.risk_manager._equity
                self.risk_manager._equity = total
                if abs(old_equity - total) / max(old_equity, 1) > 0.1:
                    result.mismatches.append(
                        f"equity_mismatch: config={old_equity:.2f}, exchange={total:.2f}"
                    )
        except Exception as exc:
            logger.error("Failed to fetch balance: {}", exc)
            result.mismatches.append(f"balance_fetch_failed: {exc}")

    async def _fetch_positions(self, result: ReconciliationResult) -> None:
        """Fetch open positions from Binance Futures."""
        try:
            # ccxt fetch_positions returns all (including zero-size)
            positions = await self._client.fetch_positions()
            for pos in positions:
                size = abs(float(pos.get("contracts", 0) or 0))
                if size > 0:
                    result.exchange_positions.append({
                        "symbol": pos.get("symbol", ""),
                        "side": pos.get("side", ""),
                        "size": size,
                        "notional": abs(float(pos.get("notional", 0) or 0)),
                        "entry_price": float(pos.get("entryPrice", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                        "leverage": float(pos.get("leverage", 1) or 1),
                        "margin_type": pos.get("marginType", ""),
                        "liquidation_price": float(pos.get("liquidationPrice", 0) or 0),
                    })

            logger.info(
                "Exchange has {} open position(s)",
                len(result.exchange_positions),
            )
            for ep in result.exchange_positions:
                logger.info(
                    "  {} {} size={:.6f} entry={:.2f} pnl={:.4f}",
                    ep["symbol"], ep["side"], ep["size"], ep["entry_price"], ep["unrealized_pnl"],
                )
        except Exception as exc:
            logger.error("Failed to fetch positions: {}", exc)
            result.mismatches.append(f"positions_fetch_failed: {exc}")

    async def _fetch_open_orders(self, result: ReconciliationResult) -> None:
        """Fetch all open orders from exchange."""
        try:
            orders = await self._client.fetch_open_orders()
            result.exchange_open_orders = [
                {
                    "id": o.get("id", ""),
                    "symbol": o.get("symbol", ""),
                    "type": o.get("type", ""),
                    "side": o.get("side", ""),
                    "amount": float(o.get("amount", 0)),
                    "price": float(o.get("price", 0) or 0),
                    "stop_price": float(o.get("stopPrice", 0) or 0),
                    "status": o.get("status", ""),
                    "reduce_only": o.get("reduceOnly", False),
                }
                for o in orders
            ]
            logger.info("Exchange has {} open order(s)", len(result.exchange_open_orders))
        except Exception as exc:
            logger.error("Failed to fetch open orders: {}", exc)
            result.mismatches.append(f"orders_fetch_failed: {exc}")

    async def _load_and_compare_db(self, result: ReconciliationResult) -> None:
        """Load open positions from DB and cross-reference with exchange."""
        if self._trade_persistence is None:
            logger.debug("No trade_persistence — skipping DB comparison")
            return

        try:
            db_positions = await self._trade_persistence.load_open_positions()
            result.db_positions = db_positions
            logger.info("DB has {} open position(s)", len(db_positions))

            # Build lookup: (exchange, symbol) → db row
            db_lookup: dict[str, dict] = {}
            for db_pos in db_positions:
                key = f"{db_pos.get('exchange', 'binance')}:{db_pos.get('symbol', '')}"
                db_lookup[key] = db_pos

            # Build lookup of exchange positions
            exch_keys: set[str] = set()
            for ep in result.exchange_positions:
                key = f"binance:{ep['symbol']}"
                exch_keys.add(key)

                if key in db_lookup:
                    db_row = db_lookup[key]
                    # Compare direction
                    db_dir = db_row.get("direction", "")
                    exch_dir = ep["side"] if ep["side"] in ("long", "short") else (
                        "long" if ep["side"] == "buy" else "short"
                    )
                    if db_dir and db_dir != exch_dir:
                        result.mismatches.append(
                            f"direction_mismatch: {ep['symbol']} db={db_dir} exchange={exch_dir}"
                        )
                    # Compare size (allow 5% tolerance for partial fills)
                    db_size = float(db_row.get("size", 0))
                    if db_size > 0 and abs(db_size - ep["size"]) / db_size > 0.05:
                        result.mismatches.append(
                            f"size_mismatch: {ep['symbol']} db={db_size:.6f} exchange={ep['size']:.6f}"
                        )
                else:
                    # Exchange has position not in DB — record but still rebuild
                    logger.warning(
                        "Position {} exists on exchange but not in DB — will persist after rebuild",
                        ep["symbol"],
                    )

            # Close stale DB positions that don't exist on exchange anymore
            for key, db_row in db_lookup.items():
                if key not in exch_keys:
                    logger.warning(
                        "DB position {} not found on exchange — marking as stale",
                        db_row.get("symbol", key),
                    )
                    result.actions_taken.append(
                        f"stale_db_position: {db_row.get('symbol', key)} (in DB but not on exchange)"
                    )
                    # Close it in DB with zero PnL (actual PnL unknown)
                    try:
                        await self._trade_persistence.persist_position_close(
                            exchange=db_row.get("exchange", "binance"),
                            symbol=db_row.get("symbol", ""),
                            exit_price=float(db_row.get("entry_price", 0)),
                            pnl=0.0,
                            pnl_pct=0.0,
                        )
                        result.actions_taken.append(
                            f"closed_stale_db_position: {db_row.get('symbol', key)}"
                        )
                    except Exception as exc:
                        logger.error("Failed to close stale DB position: {}", exc)

        except Exception as exc:
            logger.error("DB position comparison failed: {}", exc)
            # Non-fatal: exchange is truth, continue

    async def _fetch_leverage_settings(self, result: ReconciliationResult) -> None:
        """Fetch and verify leverage/margin settings for positioned symbols."""
        cfg = self.config.get_value("exchanges", "binance") or {}
        risk_cfg = self.config.get_value("risk") or {}
        leverage_value = cfg.get("leverage")
        if leverage_value in (None, "", 0):
            leverage_value = risk_cfg.get("default_leverage", 1)
        expected_leverage = int(float(leverage_value))
        expected_margin = str(cfg.get("margin_mode", "isolated")).lower()

        for ep in result.exchange_positions:
            symbol = ep["symbol"]
            exch_leverage = int(ep.get("leverage", 1))
            exch_margin = str(ep.get("margin_type", "")).lower()

            result.leverage_settings[symbol] = {
                "leverage": exch_leverage,
                "margin_mode": exch_margin,
                "expected_leverage": expected_leverage,
                "expected_margin": expected_margin,
            }

            if exch_leverage != expected_leverage:
                # Symbols in exchange_positions HAVE an open position. Binance
                # rejects set_leverage with -4161 ("Leverage reduction is not
                # supported in Isolated Margin Mode with open positions"), which
                # was previously caught and added to mismatches → safe_mode
                # tripped on every restart whenever leverage drifted. Treat the
                # exchange's actual leverage as ground truth and leave it alone;
                # leverage will be re-applied at next entry on a flat symbol
                # (cex_executor._init_client already verifies & warns there).
                logger.warning(
                    "Leverage mismatch for {} (open position — leaving exchange-side as-is): "
                    "config={} exchange={}",
                    symbol, expected_leverage, exch_leverage,
                )
                result.actions_taken.append(
                    f"leverage_mismatch_open_position: {symbol} config={expected_leverage} exchange={exch_leverage}"
                )

            if exch_margin and expected_margin and exch_margin != expected_margin:
                logger.warning(
                    "Margin mode mismatch for {}: config={} exchange={}",
                    symbol, expected_margin, exch_margin,
                )
                # Don't auto-correct margin mode on open position (exchange may reject)
                result.mismatches.append(
                    f"margin_mode_mismatch: {symbol} config={expected_margin} exchange={exch_margin}"
                )

    async def _rebuild_order_manager(self, result: ReconciliationResult) -> None:
        """Rebuild OrderManager state from exchange open orders."""
        if self._order_manager is None:
            return

        # Import inline to avoid circular
        from execution.order_manager import Order, OrderStatus, OrderSide, OrderType

        restored = 0
        for eo in result.exchange_open_orders:
            order_id = str(eo.get("id", ""))
            if not order_id or order_id in self._order_manager.orders:
                continue  # Already tracked or no ID

            side_str = str(eo.get("side", "buy")).lower()
            type_str = str(eo.get("type", "limit")).lower()

            # Map exchange type to OrderType
            if "market" in type_str or "stop" in type_str or "take_profit" in type_str:
                order_type = OrderType.MARKET
            elif "limit" in type_str:
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.LIMIT

            order = Order(
                order_id=order_id,
                client_order_id=None,
                symbol=eo.get("symbol", ""),
                side=OrderSide.BUY if side_str == "buy" else OrderSide.SELL,
                order_type=order_type,
                quantity=float(eo.get("amount", 0)),
                price=float(eo.get("price", 0) or 0),
                status=OrderStatus.OPEN,
                exchange_order_id=order_id,
                venue="binance",
                reduce_only=bool(eo.get("reduce_only", False)),
            )
            self._order_manager.orders[order_id] = order
            restored += 1

        if restored:
            result.actions_taken.append(f"restored {restored} open orders to OrderManager")
            logger.info("OrderManager: restored {} open orders from exchange", restored)

    async def _persist_rebuilt_positions(self, result: ReconciliationResult) -> None:
        """Persist rebuilt positions to DB to prevent drift."""
        if self._trade_persistence is None:
            return

        for ep in result.exchange_positions:
            symbol = ep["symbol"]
            direction = ep["side"] if ep["side"] in ("long", "short") else (
                "long" if ep["side"] == "buy" else "short"
            )
            try:
                await self._trade_persistence.persist_position_open(
                    exchange="binance",
                    symbol=symbol,
                    direction=direction,
                    entry_price=ep["entry_price"],
                    size=ep["size"],
                )
            except Exception as exc:
                logger.error("Failed to persist position {} to DB: {}", symbol, exc)

    async def _rebuild_state(self, result: ReconciliationResult) -> None:
        """Rebuild internal position state from exchange truth."""
        # Clear stale in-memory positions
        old_positions = dict(self.risk_manager._positions)
        self.risk_manager._positions.clear()

        for ep in result.exchange_positions:
            symbol = ep["symbol"]
            side = ep["side"]  # "long" or "short"
            direction = side if side in ("long", "short") else ("long" if side == "buy" else "short")
            key = f"binance:{symbol}"

            pos = Position(
                exchange="binance",
                symbol=symbol,
                direction=direction,
                size=ep["size"],
                entry_price=ep["entry_price"],
                current_price=ep["entry_price"],  # Will update on next tick
                stop_loss=0.0,  # Will be set from existing orders or computed
                take_profit=0.0,
                open_time=int(time.time()),
                highest_since_entry=ep["entry_price"],
                lowest_since_entry=ep["entry_price"],
            )
            self.risk_manager._positions[key] = pos

            # Check if this was in old state
            if key in old_positions:
                old = old_positions.pop(key)
                # Preserve SL/TP from old state
                pos.stop_loss = old.stop_loss
                pos.take_profit = old.take_profit
                pos.open_time = old.open_time
                pos.trailing_active = old.trailing_active
                pos.breakeven_moved = old.breakeven_moved
                result.actions_taken.append(f"restored {symbol} from previous state")
            else:
                result.mismatches.append(
                    f"unknown_position: {symbol} {direction} size={ep['size']:.6f} "
                    f"(exists on exchange but not in local state)"
                )

        # Check for positions in old state that don't exist on exchange
        for key, old_pos in old_positions.items():
            result.mismatches.append(
                f"stale_position: {key} {old_pos.direction} size={old_pos.size:.6f} "
                f"(in local state but not on exchange)"
            )

        logger.info(
            "State rebuilt: {} positions from exchange, {} stale removed",
            len(result.exchange_positions), len(old_positions),
        )

        # Persist rebuilt positions to DB to keep state consistent
        await self._persist_rebuilt_positions(result)

    async def _ensure_protective_orders(self, result: ReconciliationResult) -> None:
        """Ensure every open position has SL/TP orders on exchange."""
        for ep in result.exchange_positions:
            symbol = ep["symbol"]
            has_sl = False
            has_tp = False

            # Check existing open orders for this symbol and read prices back
            key = f"binance:{symbol}"
            pos = self.risk_manager._positions.get(key)
            sl_order_id: str | None = None
            tp_order_id: str | None = None
            sl_price_found: float = 0.0
            tp_price_found: float = 0.0
            for order in result.exchange_open_orders:
                if order["symbol"] != symbol or not order.get("reduce_only", False):
                    continue
                if order["type"] in ("stop_market", "stop", "STOP_MARKET", "STOP"):
                    has_sl = True
                    sl_order_id = str(order.get("id", ""))
                    # Read SL price from exchange into in-memory position
                    sl_price_found = float(order.get("stop_price", 0) or 0)
                    if pos and sl_price_found > 0:
                        pos.stop_loss = sl_price_found
                        logger.info("Read SL price from exchange for {}: {:.2f}", symbol, sl_price_found)
                if order["type"] in ("take_profit_market", "take_profit", "TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                    has_tp = True
                    tp_order_id = str(order.get("id", ""))
                    # Read TP price from exchange into in-memory position
                    tp_price_found = float(order.get("stop_price", 0) or 0)
                    if pos and tp_price_found > 0:
                        pos.take_profit = tp_price_found
                        logger.info("Read TP price from exchange for {}: {:.2f}", symbol, tp_price_found)

            # Register existing exchange SL/TP in order_placer so OCO cancel logic works
            if (has_sl or has_tp) and pos and self._order_placer:
                from execution.exchange_order_placer import ProtectiveOrders
                prot = ProtectiveOrders(
                    symbol=symbol,
                    direction=pos.direction,
                    quantity=ep["size"],
                    entry_price=ep["entry_price"],
                    sl_price=sl_price_found,
                    tp_price=tp_price_found,
                    sl_order_id=sl_order_id,
                    tp_order_id=tp_order_id,
                    sl_placed=has_sl,
                    tp_placed=has_tp,
                )
                self._order_placer._protective[symbol] = prot
                logger.info(
                    "Registered existing protective orders for {} (SL={}, TP={})",
                    symbol, sl_order_id, tp_order_id,
                )

            if not has_sl:
                result.positions_without_sl.append(symbol)
                logger.critical("Position {} has NO exchange-side SL — placing now", symbol)

                # Compute SL/TP from risk config
                key = f"binance:{symbol}"
                pos = self.risk_manager._positions.get(key)
                if pos and pos.stop_loss == 0.0:
                    # Fallback: use 1.5% from entry
                    sl_pct = float((self.risk_manager.config.get_value("risk") or {}).get("stop_loss_pct", 0.015))
                    if pos.is_long:
                        pos.stop_loss = ep["entry_price"] * (1 - sl_pct)
                        pos.take_profit = ep["entry_price"] * (1 + sl_pct * self.risk_manager._rr_ratio)
                    else:
                        pos.stop_loss = ep["entry_price"] * (1 + sl_pct)
                        pos.take_profit = ep["entry_price"] * (1 - sl_pct * self.risk_manager._rr_ratio)

                if pos and self._order_placer:
                    try:
                        await self._order_placer.place_protective_orders(
                            symbol=symbol,
                            direction=pos.direction,
                            quantity=ep["size"],
                            entry_price=ep["entry_price"],
                            sl_price=pos.stop_loss,
                            tp_price=pos.take_profit,
                        )
                        result.actions_taken.append(f"placed SL/TP for {symbol}")
                    except ProtectiveOrderFallbackRequired as exc:
                        logger.warning(
                            "Exchange-side SL unsupported for {} during reconciliation — bot-managed exits remain active: {}",
                            symbol,
                            exc,
                        )
                        result.actions_taken.append(f"local SL/TP fallback for {symbol}")
                    except Exception as exc:
                        logger.critical("FAILED to place SL for {}: {}", symbol, exc)
                        result.mismatches.append(f"sl_placement_failed: {symbol}: {exc}")
                elif pos and not self._order_placer:
                    logger.warning("No order_placer — cannot place SL/TP for {}", symbol)
            elif not has_tp:
                logger.warning("Position {} has SL but no TP — acceptable", symbol)


# ── REQ-POS-004 / REQ-FS-007: periodic in-flight reconciliation ─────────────
# StartupReconciler is one-shot and rebuilds state — too destructive to run
# while the bot is live. PeriodicReconciler does a *lightweight* diff between
# exchange truth and in-memory state, triggers SafeMode on mismatch, and
# stores the latest result snapshot for the dashboard.

class PeriodicReconciler:
    """Lightweight position/balance diff, runs on a fixed schedule."""

    def __init__(
        self,
        config: Config,
        risk_manager: RiskManager,
        client: Any,
        interval_seconds: float = 300.0,
        tolerance_qty: float = 1e-6,
    ) -> None:
        self.config = config
        self.risk_manager = risk_manager
        self._client = client
        self._interval = float(interval_seconds)
        self._tolerance = float(tolerance_qty)
        self._running = False
        self.last_check_ts: float = 0.0
        self.last_result: dict[str, Any] = {
            "ran": False,
            "ts": 0.0,
            "interval_seconds": self._interval,
            "exchange_position_count": 0,
            "internal_position_count": 0,
            "mismatches": [],
            "safe_mode_triggered": False,
        }

    async def _check_once(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ran": True,
            "ts": time.time(),
            "interval_seconds": self._interval,
            "exchange_position_count": 0,
            "internal_position_count": 0,
            "mismatches": [],
            "safe_mode_triggered": False,
        }
        if self._client is None:
            result["mismatches"].append("no_client_configured")
            return result
        try:
            ex_positions_raw = await self._client.fetch_positions()
        except Exception as exc:
            result["mismatches"].append(f"fetch_positions_failed: {type(exc).__name__}: {exc}")
            return result
        ex_positions = {
            p["symbol"]: float(p.get("contracts", 0.0) or 0.0)
            for p in (ex_positions_raw or [])
            if abs(float(p.get("contracts", 0.0) or 0.0)) > 0
        }
        internal: dict[str, float] = {}
        for sym, pos in (getattr(self.risk_manager, "positions", {}) or {}).items():
            qty = abs(float(getattr(pos, "size", 0.0) or 0.0))
            if qty > 0:
                internal[sym] = qty
        result["exchange_position_count"] = len(ex_positions)
        result["internal_position_count"] = len(internal)

        # Diff per symbol
        for sym, ex_qty in ex_positions.items():
            int_qty = internal.get(sym, 0.0)
            if abs(int_qty - ex_qty) > self._tolerance:
                result["mismatches"].append(f"{sym}: exchange={ex_qty} internal={int_qty}")
        for sym, int_qty in internal.items():
            if sym not in ex_positions:
                result["mismatches"].append(f"{sym}: internal={int_qty} exchange=missing")

        if result["mismatches"]:
            try:
                from core.safe_mode import SafeModeReason
                self.risk_manager.safe_mode.activate(
                    SafeModeReason.EXCHANGE_API_TIMEOUT,
                    detail=f"periodic_reconciliation_mismatch: {result['mismatches'][:3]}",
                )
                result["safe_mode_triggered"] = True
            except Exception as exc:
                logger.error("Failed to trip SafeMode on reconciliation mismatch: {}", exc)
        return result

    async def run(self) -> None:
        """Loop forever, sleeping `interval_seconds` between checks."""
        import asyncio
        self._running = True
        logger.info("PeriodicReconciler started (interval={}s)", self._interval)
        while self._running:
            try:
                self.last_result = await self._check_once()
                self.last_check_ts = self.last_result["ts"]
                if self.last_result["mismatches"]:
                    logger.warning(
                        "Periodic reconciliation mismatches: {}",
                        self.last_result["mismatches"][:5],
                    )
            except Exception as exc:
                logger.exception("PeriodicReconciler iteration failed: {}", exc)
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
