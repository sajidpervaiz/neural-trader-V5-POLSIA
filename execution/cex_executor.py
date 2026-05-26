from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger

from core.error_handling import sanitize_exception


def _client_order_id(signal: "TradingSignal", suffix: str = "") -> str:
    """Deterministic, exchange-safe client order ID.

    Binance accepts up to 36 chars of [A-Za-z0-9_-]. We hash the signal's
    stable fields so a retry from the same signal hashes to the same ID and
    the exchange rejects duplicate submissions on its side. The suffix
    distinguishes iceberg children / re-submits.
    """
    sig_time = int(getattr(signal, "timestamp", 0) or 0)
    raw = (
        f"{signal.exchange}|{signal.symbol}|{signal.direction}|"
        f"{sig_time}|{round(float(signal.price), 8)}|{suffix}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"nt5-{digest}"


def _client_order_params(exchange_id: str, signal: "TradingSignal", suffix: str = "") -> dict[str, str]:
    client_id = _client_order_id(signal, suffix)
    if exchange_id == "binance":
        return {"newClientOrderId": client_id}
    if exchange_id == "bybit":
        return {"orderLinkId": client_id}
    if exchange_id == "okx":
        return {"clOrdId": client_id}
    return {"clientOrderId": client_id}

from core.config import Config
from core.event_bus import EventBus
from core.safe_mode import SafeModeReason
from engine.signal_generator import TradingSignal
from execution.legacy_cex_common import L2DepthLevel, OrderbookSnapshot
from execution.order_manager import OrderManager, OrderSide, OrderType
from execution.rate_limiter import RateLimiter
from execution.risk_manager import RiskManager
from execution.exchange_order_placer import ExchangeOrderPlacer, ProtectiveOrderFallbackRequired


@dataclass
class OrderResult:
    order_id: str
    exchange: str
    symbol: str
    direction: str
    price: float
    quantity: float
    status: str
    is_paper: bool
    timestamp: int
    raw: dict[str, Any] | None = None


class CEXExecutor:
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        risk_manager: RiskManager,
        exchange_id: str,
        order_manager: OrderManager | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self.exchange_id = exchange_id
        self._order_manager = order_manager
        self._client: Any = None
        self._order_placer: ExchangeOrderPlacer | None = None
        self._running = False
        # Binance allows 1200 req/min; cap at 600/min (10/sec) for safety margin
        self._rate_limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        exec_cfg = (config.get_value("execution") or {})
        self._maker_first = bool(exec_cfg.get("maker_first", True))
        self._post_only = bool(exec_cfg.get("post_only", True))
        self._iceberg_threshold_usd = float(exec_cfg.get("iceberg_threshold_usd", 10000.0))
        self._iceberg_chunks = int(exec_cfg.get("iceberg_chunks", 4))
        # Limit-fill polling behaviour (used by _wait_for_fill).
        # Defaults preserve previous behaviour: 3 polls × 5s = 15s, then market.
        # Set market_on_timeout: false to disable auto-conversion.
        fill_cfg = exec_cfg.get("wait_for_fill", {}) or {}
        self._fill_max_retries = int(fill_cfg.get("max_retries", 3))
        self._fill_wait_seconds = float(fill_cfg.get("wait_seconds", 5.0))
        self._fill_market_on_timeout = bool(fill_cfg.get("market_on_timeout", True))
        # Verified exchange-side leverage per symbol — populated by _init_client.
        # If a set_leverage call fails or echoes a different value, the actual
        # leverage as reported by the exchange is recorded here.
        self._actual_leverage: dict[str, int] = {}

    @staticmethod
    def _apply_fill_to_position(pos: Any, signal: TradingSignal, fill_price: float, filled_qty: float) -> None:
        old_entry = getattr(pos, "entry_price", signal.price)
        old_sl = getattr(pos, "stop_loss", None)
        old_tp = getattr(pos, "take_profit", None)
        if all(isinstance(v, (int, float)) for v in (old_entry, old_sl, old_tp)):
            sl_distance = abs(float(old_entry) - float(old_sl))
            tp_distance = abs(float(old_tp) - float(old_entry))
            if signal.is_long:
                pos.stop_loss = fill_price - sl_distance
                pos.take_profit = fill_price + tp_distance
            else:
                pos.stop_loss = fill_price + sl_distance
                pos.take_profit = fill_price - tp_distance
        pos.entry_price = fill_price
        pos.current_price = fill_price
        pos.highest_since_entry = fill_price
        pos.lowest_since_entry = fill_price
        pos.size = filled_qty
        if hasattr(pos, "pending_fill"):
            pos.pending_fill = False

    def _publish_pipeline_latency(self, signal: TradingSignal, stage: str, latency_s: float) -> None:
        try:
            payload = {
                "stage": str(stage),
                "exchange": getattr(signal, "exchange", self.exchange_id),
                "symbol": getattr(signal, "symbol", "unknown"),
                "latency_s": max(0.0, float(latency_s)),
                "ts": time.time(),
            }
            publish_nowait = getattr(self.event_bus, "publish_nowait", None)
            if callable(publish_nowait):
                publish_nowait("PIPELINE_LATENCY", payload)
            else:
                asyncio.create_task(self.event_bus.publish("PIPELINE_LATENCY", payload))
        except Exception as exc:
            logger.debug("{} pipeline latency publish failed: {}", self.exchange_id, exc)

    def _exchange_symbol(self, symbol: str) -> str:
        normalizer = getattr(self, "_normalize_symbol", None)
        if callable(normalizer):
            try:
                return str(normalizer(symbol))
            except Exception:
                return symbol
        return symbol

    async def _init_client(self) -> None:
        if self._client is not None:
            return  # Already initialized — avoid wiping order_placer state
        cfg = self.config.get_value("exchanges", self.exchange_id) or {}
        if not cfg.get("enabled", False):
            return
        cls = getattr(ccxt, self.exchange_id, None)
        if cls is None:
            logger.warning("Unknown exchange: {}", self.exchange_id)
            return
        params: dict[str, Any] = {
            "apiKey": cfg.get("api_key", ""),
            "secret": cfg.get("api_secret", ""),
            "enableRateLimit": True,
        }
        passphrase = cfg.get("passphrase")
        if passphrase:
            params["password"] = passphrase
        # ccxt options apply equally to demo, testnet, and live — futures-only API
        # keys require defaultType=future and disable spot/sapi calls during
        # load_markets / fetch_balance. Skipping these for the demo branch (as the
        # previous code did) made stray sapi calls hit mainnet api.binance.com,
        # where demo keys don't exist → -2008.
        raw_type = cfg.get("type", "future")
        if raw_type == "futures":
            raw_type = "future"
        params["options"] = {
            "defaultType": raw_type,
            "fetchCurrencies": False,
            "fetchMargins": False,
            "warnOnFetchOpenOrdersWithoutSymbol": False,
        }
        if raw_type == "future":
            # Restrict market loading to linear/inverse only — without this, ccxt
            # tries to also fetch spot markets (sapi) which futures-only keys reject.
            params["options"]["fetchMarkets"] = ["linear", "inverse"]
        try:
            self._client = cls(params)
            # Two distinct binance sandboxes:
            #   demo: true     → demo.binance.com keys (ccxt.enable_demo_trading)
            #   testnet: true  → testnet.binancefuture.com keys, classic urls['test'] swap
            # If both are set, demo wins (it's the CCXT-supported direction per
            # announcement #92 — testnet/sandbox no longer supported for futures).
            if cfg.get("demo") and self.exchange_id == "binance":
                if hasattr(self._client, "enable_demo_trading"):
                    self._client.enable_demo_trading(True)
                else:
                    api_urls = self._client.urls.get("api", {})
                    for k, v in self._client.urls.get("demo", {}).items():
                        if k in api_urls:
                            api_urls[k] = v
                    self._client.urls["api"] = api_urls
            elif cfg.get("testnet"):
                # Proper testnet setup: manually swap only futures URLs
                # instead of set_sandbox_mode() which triggers ccxt's
                # deprecation error for Binance futures testnet.
                testnet_urls = self._client.urls.get("test", {})
                for key, url in testnet_urls.items():
                    if key.startswith(("fapi", "dapi")) and key in self._client.urls["api"]:
                        self._client.urls["api"][key] = url
            await self._client.load_markets()
            # Detect hedge mode (dual position side)
            hedge_mode = bool(cfg.get("hedge_mode", False))
            if not hedge_mode:
                try:
                    pm = await self._client.fapiPrivateGetPositionSideDual()
                    hedge_mode = bool(pm.get("dualSidePosition", False))
                except Exception:
                    pass  # default to one-way
            # Create exchange-side order placer for SL/TP
            working_type = str(cfg.get("working_type", "CONTRACT_PRICE"))
            self._order_placer = ExchangeOrderPlacer(
                self._client,
                working_type=working_type,
                rate_limiter=self._rate_limiter,
                hedge_mode=hedge_mode,
            )

            # ── Set leverage and margin mode for all configured symbols ───
            leverage = int(cfg.get("leverage", self.risk_manager._leverage if hasattr(self.risk_manager, '_leverage') else 1))
            margin_mode = str(cfg.get("margin_mode", "isolated")).lower()
            symbols = cfg.get("symbols", [])
            self._symbols = symbols
            for sym in symbols:
                try:
                    await self._client.set_margin_mode(margin_mode, sym)
                except Exception as e:
                    # set_margin_mode often errors benignly (-4046 "no need to change",
                    # -4067 "open orders"). Logged at debug; safe.
                    logger.debug("{} set_margin_mode({}, {}): {}", self.exchange_id, margin_mode, sym, e)
                # Apply leverage and VERIFY the exchange actually accepted it.
                # A silent failure here would leave the symbol on whatever leverage
                # was set in a prior session — including potentially 50x.
                applied = leverage
                try:
                    resp = await self._client.set_leverage(leverage, sym)
                    if isinstance(resp, dict) and resp.get("leverage") is not None:
                        applied = int(float(resp.get("leverage")))
                except Exception as e:
                    logger.warning(
                        "{} set_leverage({}, {}) failed ({}) — fetching actual leverage from exchange",
                        self.exchange_id, leverage, sym, e,
                    )
                    try:
                        positions = await self._client.fetch_positions([sym])
                        if positions:
                            applied = int(float(positions[0].get("leverage") or leverage))
                    except Exception as fe:
                        logger.warning("{} could not verify {} leverage: {}", self.exchange_id, sym, fe)
                self._actual_leverage[sym] = applied
                if applied != leverage:
                    logger.warning(
                        "{} {} LEVERAGE MISMATCH: requested={} applied={} — exchange-side leverage is {}x",
                        self.exchange_id, sym, leverage, applied, applied,
                    )
                    try:
                        await self.event_bus.publish("ALERT_WARNING", {
                            "type": "leverage_mismatch",
                            "exchange": self.exchange_id,
                            "symbol": sym,
                            "requested": leverage,
                            "applied": applied,
                            "ts": int(time.time()),
                        })
                    except Exception:
                        pass
            logger.info("{} CEX client initialized (leverage={}, margin={})", self.exchange_id, leverage, margin_mode)
            # Pre-warm HTTP connection pool to eliminate cold-start latency
            try:
                await self._client.fetch_ticker(symbols[0] if symbols else "BTC/USDT:USDT")
            except Exception:
                pass
        except Exception as exc:
            logger.warning("{} client init failed: {}", self.exchange_id, exc)
            self._client = None

    async def execute_signal(self, signal: TradingSignal, size: float) -> OrderResult | None:
        if self.config.paper_mode:
            return await self._paper_execute(signal, size)
        return await self._live_execute(signal, size)

    async def _paper_execute_with_pos(self, signal: TradingSignal, size: float, pos: Any) -> OrderResult:
        """Paper execute with pre-opened position (from approve_and_open).

        Applies BOTH slippage + commission to the fill so paper PnL matches
        real-world taker costs (~5bps slippage + 5bps Binance USDM taker fee
        per side ≈ 20bps round-trip). Without the commission, paper-mode
        backtests showed fictional profits on strategies that lose money live.
        """
        submit_started = time.perf_counter()
        bt = self.config.get_value("backtest") or {}
        slippage = float(bt.get("slippage_pct", 0.0002))
        commission = float(bt.get("commission_pct", 0.0005))  # taker default
        cost = slippage + commission
        fill_price = signal.price * (1 + cost if signal.is_long else 1 - cost)
        result = OrderResult(
            order_id=f"paper_{int(time.time()*1000)}",
            exchange=signal.exchange,
            symbol=signal.symbol,
            direction=signal.direction,
            price=fill_price,
            quantity=size / fill_price if fill_price > 0 else 0,
            status="filled",
            is_paper=True,
            timestamp=int(time.time()),
        )
        await self._record_paper_order_in_manager(signal, fill_price, result.quantity)
        await self.event_bus.publish("ORDER_FILLED", result)
        self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
        logger.info("Paper order filled: {} {}/{} @ {:.2f}", signal.direction.upper(), signal.exchange, signal.symbol, fill_price)
        return result

    async def _paper_execute(self, signal: TradingSignal, size: float) -> OrderResult:
        submit_started = time.perf_counter()
        bt = self.config.get_value("backtest") or {}
        slippage = float(bt.get("slippage_pct", 0.0002))
        commission = float(bt.get("commission_pct", 0.0005))
        cost = slippage + commission
        fill_price = signal.price * (1 + cost if signal.is_long else 1 - cost)
        # Note: open_position is called below and uses signal.price as entry.
        # We overwrite pos.entry_price = fill_price afterwards so the position's
        # cost basis reflects the entry-side slippage + commission.
        result = OrderResult(
            order_id=f"paper_{int(time.time()*1000)}",
            exchange=signal.exchange,
            symbol=signal.symbol,
            direction=signal.direction,
            price=fill_price,
            quantity=size / fill_price if fill_price > 0 else 0,
            status="filled",
            is_paper=True,
            timestamp=int(time.time()),
        )
        pos = await self.risk_manager.open_position(signal, size)
        # Overwrite cost basis to the cost-adjusted fill_price so paper PnL
        # reflects entry-side slippage + commission. Without this, only exit-side
        # cost was charged in close_position → round-trip cost ≈ 7bps instead of
        # the realistic 14bps.
        if pos is not None:
            pos.entry_price = fill_price
            pos.current_price = fill_price
            pos.highest_since_entry = fill_price
            pos.lowest_since_entry = fill_price
        await self._record_paper_order_in_manager(signal, fill_price, result.quantity)
        await self.event_bus.publish("ORDER_FILLED", result)
        self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
        logger.info("Paper order filled: {} {}/{} @ {:.2f}", signal.direction.upper(), signal.exchange, signal.symbol, fill_price)
        return result

    async def _record_paper_order_in_manager(
        self, signal: TradingSignal, fill_price: float, filled_qty: float,
    ) -> None:
        """Register a paper fill through OrderManager so the lifecycle
        (idempotency, self-trade prevention, audit, /api/exchange/orders) is
        identical to live. Without this, paper mode silently bypasses
        OrderManager and the live path is exercised first time in production.
        """
        if self._order_manager is None or filled_qty <= 0:
            return
        try:
            side = OrderSide.BUY if signal.is_long else OrderSide.SELL
            success, order, reason = await self._order_manager.place_order(
                exchange=signal.exchange,
                symbol=signal.symbol,
                side=side,
                quantity=filled_qty,
                price=fill_price,
                order_type=OrderType.MARKET,
                metadata={"paper": True, "signal_score": float(getattr(signal, "score", 0.0) or 0.0)},
            )
            if not success or order is None:
                logger.debug("paper order not registered in OrderManager: {}", reason)
                return
            await self._order_manager.confirm_order_submission(
                client_order_id=order.client_order_id,
                exchange_order_id=order.order_id,  # synthetic exchange id for paper
            )
            await self._order_manager.record_fill(
                client_order_id=order.client_order_id,
                fill_id=f"paper_fill_{int(time.time() * 1_000_000)}",
                quantity=filled_qty,
                price=fill_price,
                fee=0.0,
            )
        except Exception as exc:
            logger.debug("paper OrderManager registration failed: {}", exc)

    async def _emergency_market_close(
        self, signal: TradingSignal, filled_qty: float, fill_price: float, *, reason: str,
    ) -> bool:
        """Flatten an unprotected position with a reduceOnly market order.

        Called when exchange-side SL placement raises a non-fallback exception:
        the position is open but has no protective stop, so the safer move is to
        exit immediately rather than leave it bleeding while the operator
        responds to the alert. Returns True iff the exchange close succeeded.
        """
        if self._client is None or filled_qty <= 0:
            return False
        close_side = "sell" if signal.is_long else "buy"
        try:
            await self._rate_limiter.acquire()
            await self._client.create_market_order(
                symbol=signal.symbol, side=close_side, amount=filled_qty,
                params={"reduceOnly": True},
            )
            try:
                await self.risk_manager.close_position(signal.exchange, signal.symbol, fill_price)
            except Exception as rm_exc:
                logger.warning("Emergency close: risk_manager.close_position failed: {}", rm_exc)
            logger.warning(
                "EMERGENCY market close completed for {} ({}): qty={} reason={}",
                signal.symbol, signal.direction, filled_qty, reason,
            )
            return True
        except Exception as close_exc:
            logger.critical(
                "EMERGENCY market close FAILED for {}: {} — position UNPROTECTED",
                signal.symbol, close_exc,
            )
            return False

    async def _live_execute(self, signal: TradingSignal, size: float, reserved_pos: Any = None) -> OrderResult | None:
        """Execute using LIMIT order for entries; MARKET only for emergency exits.
        Places exchange-side SL/TP after fill for crash protection.
        If reserved_pos is provided, position was already opened atomically — skip re-opening."""
        if self._client is None:
            logger.error("No live client for {} — cannot execute", self.exchange_id)
            return None
        try:
            side = "buy" if signal.is_long else "sell"
            amount = size / signal.price
            is_emergency = signal.metadata.get("emergency_exit", False)

            if is_emergency:
                await self._rate_limiter.acquire()
                submit_started = time.perf_counter()
                order = await self._client.create_market_order(
                    symbol=signal.symbol,
                    side=side,
                    amount=amount,
                    params=_client_order_params(self.exchange_id, signal, "emergency"),
                )
                self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
            else:
                limit_params: dict[str, Any] = _client_order_params(
                    self.exchange_id,
                    signal,
                    "entry",
                )
                if self._post_only:
                    limit_params["postOnly"] = True
                notional = size
                if (
                    self._iceberg_threshold_usd > 0
                    and notional > self._iceberg_threshold_usd
                    and self._iceberg_chunks > 1
                ):
                    order = await self._place_iceberg(
                        signal, side, amount, limit_params,
                    )
                else:
                    await self._rate_limiter.acquire()
                    submit_started = time.perf_counter()
                    order = await self._client.create_limit_order(
                        symbol=signal.symbol, side=side, amount=amount,
                        price=signal.price, params=limit_params,
                    )
                    self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
                    order = await self._wait_for_fill(signal, order, amount)

            fill_price = float(order.get("average", order.get("price", signal.price)))
            filled_qty = float(order.get("filled", amount))

            if signal.price > 0 and fill_price > 0:
                slippage_bps = ((fill_price - signal.price) / signal.price) * 10_000
                slip_against = slippage_bps if signal.direction == "long" else -slippage_bps
                if abs(slip_against) >= 5.0:
                    logger.warning(
                        "Slippage {} {}: expected {:.2f} got {:.2f} ({:+.1f}bps against)",
                        signal.symbol, signal.direction, signal.price, fill_price, slip_against,
                    )
                else:
                    logger.debug(
                        "Slippage {} {}: {:+.1f}bps", signal.symbol, signal.direction, slip_against,
                    )
                try:
                    await self.event_bus.publish("SLIPPAGE_OBSERVED", {
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "expected_price": signal.price,
                        "fill_price": fill_price,
                        "slippage_bps": slippage_bps,
                        "slippage_against_bps": slip_against,
                        "timestamp": int(time.time()),
                    })
                except Exception:
                    pass

            result = OrderResult(
                order_id=order.get("id", ""),
                exchange=signal.exchange,
                symbol=signal.symbol,
                direction=signal.direction,
                price=fill_price,
                quantity=filled_qty,
                status=str(order.get("status", "unknown") or "unknown").lower(),
                is_paper=False,
                timestamp=int(time.time()),
                raw=order,
            )
            if result.status in ("filled", "closed"):
                # Commit the reservation to the real fill price — this updates
                # entry_price, slides SL/TP by the strategy's intended distance,
                # persists to SQLite, and flips pending_fill off.
                if reserved_pos is not None:
                    pos = await self.risk_manager.rebase_position_to_fill(
                        signal.exchange, signal.symbol, fill_price, filled_qty,
                    )
                    if pos is None:
                        # Reservation vanished (e.g. kill switch). Bail.
                        logger.error(
                            "Reservation missing on rebase for {}/{} — "
                            "aborting protective-order placement",
                            signal.exchange, signal.symbol,
                        )
                        return result
                else:
                    pos = await self.risk_manager.open_position(signal, filled_qty * fill_price)
                    if pos is not None:
                        self._apply_fill_to_position(pos, signal, fill_price, filled_qty)
                await self.event_bus.publish("ORDER_FILLED", result)

                # ── Place exchange-side SL/TP (crash protection) ──────────
                if self._order_placer and not is_emergency:
                    try:
                        await self._order_placer.place_protective_orders(
                            symbol=signal.symbol,
                            direction=signal.direction,
                            quantity=filled_qty,
                            entry_price=fill_price,
                            sl_price=pos.stop_loss,
                            tp_price=pos.take_profit,
                        )
                        logger.info(
                            "Exchange-side SL/TP placed for {} SL={:.2f} TP={:.2f}",
                            signal.symbol, pos.stop_loss, pos.take_profit,
                        )
                    except ProtectiveOrderFallbackRequired as exc:
                        # ESCALATED to CRITICAL: the position has no exchange-side
                        # stop. If the bot crashes before a bot-managed exit fires,
                        # the position is fully unprotected. Operator must ack.
                        logger.critical(
                            "Exchange-side SL/TP unavailable for {} — falling back to bot-managed exits "
                            "(bot crash leaves position unprotected); OPERATOR ACK REQUIRED: {}",
                            signal.symbol, exc,
                        )
                        await self.event_bus.publish("ALERT_WARNING", {
                            "type": "sl_fallback_local",
                            "symbol": signal.symbol,
                            "error": sanitize_exception(exc),
                            "needs_ack": True,
                            "exchange": signal.exchange,
                            "direction": signal.direction,
                            "quantity": filled_qty,
                            "fill_price": fill_price,
                            "ts": int(time.time()),
                        })
                    except Exception as exc:
                        logger.critical(
                            "FAILED to place exchange-side SL for {} — attempting emergency market close: {}",
                            signal.symbol, exc,
                        )
                        # Try to flatten the now-unprotected position before halting trading.
                        closed_ok = await self._emergency_market_close(
                            signal, filled_qty, fill_price, reason="sl_placement_failed",
                        )
                        # Trip the breaker either way — new entries are paused until operator reviews.
                        self.risk_manager._circuit_breaker.trip(
                            f"sl_placement_failed:{signal.symbol}"
                        )
                        await self.event_bus.publish("ALERT_CRITICAL", {
                            "type": "sl_placement_failed",
                            "symbol": signal.symbol,
                            "error": sanitize_exception(exc),
                            "emergency_close_ok": closed_ok,
                        })
            elif filled_qty > 0:
                if reserved_pos is not None:
                    pos = await self.risk_manager.rebase_position_to_fill(
                        signal.exchange, signal.symbol, fill_price, filled_qty,
                    )
                    if pos is None:
                        logger.error(
                            "Reservation missing on partial rebase for {}/{} — "
                            "aborting protective-order placement",
                            signal.exchange, signal.symbol,
                        )
                        return result
                else:
                    pos = await self.risk_manager.open_position(signal, filled_qty * fill_price)
                    if pos is not None:
                        self._apply_fill_to_position(pos, signal, fill_price, filled_qty)
                await self.event_bus.publish("ORDER_PARTIALLY_FILLED", result)
                # Place protective orders for partial fill qty
                if self._order_placer:
                    try:
                        await self._order_placer.place_protective_orders(
                            symbol=signal.symbol,
                            direction=signal.direction,
                            quantity=filled_qty,
                            entry_price=fill_price,
                            sl_price=pos.stop_loss,
                            tp_price=pos.take_profit,
                        )
                    except ProtectiveOrderFallbackRequired as exc:
                        logger.critical(
                            "Exchange-side SL/TP unavailable for PARTIAL fill {} — falling back to bot-managed exits "
                            "(bot crash leaves partial position unprotected); OPERATOR ACK REQUIRED: {}",
                            signal.symbol, exc,
                        )
                        await self.event_bus.publish("ALERT_WARNING", {
                            "type": "sl_fallback_local_partial",
                            "symbol": signal.symbol,
                            "error": sanitize_exception(exc),
                            "needs_ack": True,
                            "exchange": signal.exchange,
                            "direction": signal.direction,
                            "quantity": filled_qty,
                            "fill_price": fill_price,
                            "ts": int(time.time()),
                        })
                    except Exception as exc:
                        logger.critical(
                            "SL placement failed for partial fill {} — attempting emergency market close: {}",
                            signal.symbol, exc,
                        )
                        closed_ok = await self._emergency_market_close(
                            signal, filled_qty, fill_price, reason="sl_placement_failed_partial",
                        )
                        self.risk_manager._circuit_breaker.trip(
                            f"sl_placement_failed:{signal.symbol}"
                        )
                        await self.event_bus.publish("ALERT_CRITICAL", {
                            "type": "sl_placement_failed_partial",
                            "symbol": signal.symbol,
                            "error": sanitize_exception(exc),
                            "emergency_close_ok": closed_ok,
                        })
            else:
                logger.warning("{} order not filled: status={}", self.exchange_id, result.status)
                cancel_uncertain = False
                if reserved_pos is not None and result.status in ("open", "new") and self._client is not None:
                    try:
                        await self._rate_limiter.acquire()
                        await self._client.cancel_order(result.order_id, signal.symbol)
                        logger.info(
                            "{} cancelled unfilled live order {} before rolling back reservation",
                            self.exchange_id, result.order_id,
                        )
                    except Exception as exc:
                        cancel_uncertain = True
                        logger.critical(
                            "{} could not cancel unfilled live order {} for {} before reservation rollback: {}",
                            self.exchange_id, result.order_id, signal.symbol, exc,
                        )
                        try:
                            self.risk_manager._circuit_breaker.trip(
                                f"untracked_open_order_cancel_failed:{signal.symbol}"
                            )
                        except Exception:
                            pass
                        await self.event_bus.publish("ALERT_CRITICAL", {
                            "type": "untracked_open_order_cancel_failed",
                            "exchange": self.exchange_id,
                            "symbol": signal.symbol,
                            "order_id": result.order_id,
                            "status": result.status,
                            "error": sanitize_exception(exc),
                            "needs_ack": True,
                            "ts": int(time.time()),
                        })
                await self.event_bus.publish("ORDER_FAILED", result)
                if reserved_pos is not None:
                    if cancel_uncertain:
                        # A real exchange order may still be live. Keep the
                        # pending reservation so _handle_signal does not drop
                        # tracking; reconciliation/operator intervention must
                        # decide final state.
                        return result
                    # Status is e.g. cancelled/rejected and cancellation is
                    # verified or unnecessary — allow _handle_signal to roll
                    # the reservation back through cancel_reserved_position.
                    return None
            return result
        except Exception as exc:
            logger.error("{} live order failed: {}", self.exchange_id, sanitize_exception(exc))
            logger.opt(exception=True).debug("{} live order stack trace", self.exchange_id)
            return None

    async def _place_iceberg(
        self,
        signal: TradingSignal,
        side: str,
        amount: float,
        limit_params: dict,
    ) -> dict:
        """Split a large order into N child limit orders at the same price.

        Returns a synthetic aggregate dict compatible with the downstream fill handling.
        """
        chunks = max(2, int(self._iceberg_chunks))
        chunk_amt = amount / chunks
        logger.info(
            "Iceberg: {} {} x {} chunks of {:.6f} (post_only={})",
            self.exchange_id, signal.symbol, chunks, chunk_amt, self._post_only,
        )
        child_orders: list[dict] = []
        filled_total = 0.0
        avg_price_accum = 0.0
        for i in range(chunks):
            try:
                child_params = {
                    **limit_params,
                    **_client_order_params(self.exchange_id, signal, f"iceberg-{i}"),
                }
                await self._rate_limiter.acquire()
                submit_started = time.perf_counter()
                child = await self._client.create_limit_order(
                    symbol=signal.symbol, side=side, amount=chunk_amt,
                    price=signal.price, params=child_params,
                )
                self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
                child = await self._wait_for_fill(signal, child, chunk_amt)
                child_orders.append(child)
                f = float(child.get("filled", 0) or 0)
                p = float(child.get("average", child.get("price", signal.price)) or signal.price)
                filled_total += f
                avg_price_accum += f * p
            except Exception as exc:
                logger.warning("Iceberg child {} failed: {}", i, exc)
        avg_price = avg_price_accum / filled_total if filled_total > 0 else signal.price
        agg_status = "filled" if filled_total >= amount * 0.98 else (
            "partially_filled" if filled_total > 0 else "cancelled"
        )
        return {
            "id": "iceberg_" + str(int(time.time() * 1000)),
            "status": agg_status,
            "filled": filled_total,
            "average": avg_price,
            "price": signal.price,
            "children": child_orders,
        }

    async def _wait_for_fill(
        self, signal: TradingSignal, order: dict, amount: float,
        max_retries: int | None = None, wait_sec: float | None = None,
    ) -> dict:
        """Poll for fill; on timeout cancel and (if configured) replace at market.

        Polling cadence + market-conversion behaviour are config-driven via
        execution.wait_for_fill.{max_retries,wait_seconds,market_on_timeout}.
        Explicit kwargs override config (used for tests / iceberg tuning).
        """
        if max_retries is None:
            max_retries = self._fill_max_retries
        if wait_sec is None:
            wait_sec = self._fill_wait_seconds
        order_id = order.get("id", "")
        for attempt in range(max_retries):
            await asyncio.sleep(wait_sec)
            if self._client is None:
                logger.warning("{} client gone during fill wait — returning last order state", self.exchange_id)
                return order
            try:
                await self._rate_limiter.acquire()
                fetched = await self._client.fetch_order(order_id, signal.symbol)
            except Exception:
                continue
            status = fetched.get("status", "")
            if status in ("closed", "filled"):
                return fetched
            filled_qty = float(fetched.get("filled", 0))
            if filled_qty > 0:
                return fetched  # partial fill — accept it
            logger.debug("{} order {} not filled after {}s, attempt {}/{}",
                         self.exchange_id, order_id, (attempt + 1) * wait_sec, attempt + 1, max_retries)

        if self._client is None:
            return order

        # Operator opt-out: if market_on_timeout is False, return the unfilled
        # limit order as-is. Caller decides whether to wait longer / cancel.
        if not self._fill_market_on_timeout:
            logger.info(
                "{} limit {} unfilled after {}×{}s — market_on_timeout=false, leaving open",
                self.exchange_id, order_id, max_retries, wait_sec,
            )
            return order

        # Cancel the limit order and fall back to market only after cancellation
        # is confirmed. If cancellation is uncertain, a fallback market order can
        # double exposure when the original limit later fills.
        try:
            await self._rate_limiter.acquire()
            await self._client.cancel_order(order_id, signal.symbol)
            logger.info("{} cancelled unfilled limit order {}, placing market order",
                        self.exchange_id, order_id)
        except Exception as exc:
            logger.critical(
                "{} cancel of unfilled limit order {} is uncertain; blocking market fallback: {}",
                self.exchange_id, order_id, exc,
            )
            try:
                self.risk_manager._circuit_breaker.trip(
                    f"entry_cancel_uncertain:{signal.symbol}"
                )
            except Exception:
                pass
            await self.event_bus.publish("ALERT_CRITICAL", {
                "type": "entry_cancel_uncertain_no_market_fallback",
                "exchange": self.exchange_id,
                "symbol": signal.symbol,
                "order_id": order_id,
                "error": sanitize_exception(exc),
                "needs_ack": True,
                "ts": int(time.time()),
            })
            try:
                await self._rate_limiter.acquire()
                final_state = await self._client.fetch_order(order_id, signal.symbol)
                return final_state
            except Exception:
                return order

        if self._client is None:
            return order

        # P0: Re-fetch order to check what filled during cancel race
        already_filled = 0.0
        try:
            await self._rate_limiter.acquire()
            final_state = await self._client.fetch_order(order_id, signal.symbol)
            already_filled = float(final_state.get("filled", 0))
            if final_state.get("status") in ("closed", "filled"):
                return final_state  # fully filled during cancel — no market needed
        except Exception:
            already_filled = 0.0

        remaining = amount - already_filled
        if remaining <= 0:
            return final_state

        if self._client is None:
            return order

        side = "buy" if signal.is_long else "sell"
        await self._rate_limiter.acquire()
        fallback_params = _client_order_params(self.exchange_id, signal, suffix="timeout-market")
        submit_started = time.perf_counter()
        market_order = await self._client.create_market_order(
            symbol=signal.symbol, side=side, amount=remaining, params=fallback_params,
        )
        self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)
        return market_order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an exchange order through the same runtime executor contract."""
        if bool(getattr(self.config, "paper_mode", False)):
            logger.info("[PAPER] {} cancel order {} {}", self.exchange_id, order_id, symbol)
            return True
        if self._client is None:
            logger.error("{} cancel_order failed: client unavailable", self.exchange_id)
            return False
        try:
            await self._rate_limiter.acquire()
            await self._client.cancel_order(order_id, self._exchange_symbol(symbol))
            logger.info("{} cancelled order {} {}", self.exchange_id, order_id, symbol)
            return True
        except Exception as exc:
            logger.error("{} cancel_order failed for {} {}: {}", self.exchange_id, symbol, order_id, sanitize_exception(exc))
            return False

    async def get_orderbook_snapshot(self, symbol: str, depth: int = 10) -> OrderbookSnapshot:
        """Fetch an L2 snapshot using the shared router/contract shape."""
        if self._client is None:
            await self._init_client()
        if self._client is None:
            raise RuntimeError(f"{self.exchange_id} client unavailable")
        await self._rate_limiter.acquire()
        orderbook = await self._client.fetch_order_book(self._exchange_symbol(symbol), limit=depth)
        bids = [
            L2DepthLevel(price=float(level[0]), quantity=float(level[1]))
            for level in orderbook.get("bids", [])[:depth]
            if len(level) >= 2
        ]
        asks = [
            L2DepthLevel(price=float(level[0]), quantity=float(level[1]))
            for level in orderbook.get("asks", [])[:depth]
            if len(level) >= 2
        ]
        return OrderbookSnapshot(
            symbol=symbol,
            timestamp_ms=int(time.time() * 1000),
            bids=bids,
            asks=asks,
            sequence=int(orderbook.get("timestamp") or 0),
        )

    async def close_position(
        self,
        symbol: str,
        price: float,
        *,
        reason: str = "manual_close",
    ) -> OrderResult | None:
        """Close one tracked position with a reduce-only exchange order first.

        The dashboard, kill-switch helpers, and paper simulator should all use
        this contract instead of reaching into executor._client directly.
        """
        positions = getattr(self.risk_manager, "positions", {}) or {}
        pos = positions.get(f"{self.exchange_id}:{symbol}")
        if pos is None:
            logger.warning("{} close_position skipped: no tracked position for {}", self.exchange_id, symbol)
            return None

        requested_price = float(price or 0.0)
        if requested_price <= 0:
            requested_price = float(getattr(pos, "current_price", 0.0) or getattr(pos, "entry_price", 0.0) or 0.0)
        quantity = abs(float(getattr(pos, "size", 0.0) or 0.0))
        if quantity <= 0 or requested_price <= 0:
            logger.warning("{} close_position invalid qty/price for {} qty={} price={}", self.exchange_id, symbol, quantity, requested_price)
            return None

        direction = str(getattr(pos, "direction", "long") or "long").lower()
        close_side = "sell" if direction == "long" else "buy"

        if bool(getattr(self.config, "paper_mode", False)):
            closed = await self.risk_manager.close_position(self.exchange_id, symbol, requested_price)
            if closed is None:
                return None
            result = OrderResult(
                order_id=f"paper_close_{self.exchange_id}_{int(time.time() * 1000)}",
                exchange=self.exchange_id,
                symbol=symbol,
                direction=direction,
                price=float(getattr(closed, "current_price", requested_price) or requested_price),
                quantity=quantity,
                status="filled",
                is_paper=True,
                timestamp=int(time.time()),
                raw={"reason": reason, "contract_close": True},
            )
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": closed,
                "reason": reason,
                "price": result.price,
                "order_id": result.order_id,
            })
            return result

        if self._client is None:
            logger.error("{} close_position failed: client unavailable", self.exchange_id)
            return None

        exchange_symbol = self._exchange_symbol(symbol)
        try:
            async def _cancel_protective_orders() -> None:
                placer = getattr(self, "_order_placer", None)
                if placer is None:
                    return
                try:
                    await placer.cancel_all_for_symbol(symbol)
                    if exchange_symbol != symbol:
                        await placer.cancel_all_for_symbol(exchange_symbol)
                    placer.remove_tracking(symbol)
                    if exchange_symbol != symbol:
                        placer.remove_tracking(exchange_symbol)
                except Exception as exc:
                    logger.warning("{} protective-order cleanup failed for {}: {}", self.exchange_id, symbol, sanitize_exception(exc))

            await self._rate_limiter.acquire()
            close_started = time.perf_counter()
            close_coro = self._client.create_market_order(
                symbol=exchange_symbol,
                side=close_side,
                amount=quantity,
                params={"reduceOnly": True},
            )
            if getattr(self, "_order_placer", None) is not None:
                order, _ = await asyncio.gather(close_coro, _cancel_protective_orders())
            else:
                order = await close_coro

            fill_price = float(order.get("average") or order.get("price") or requested_price)
            filled_qty = float(order.get("filled") or quantity)
            status = str(order.get("status", "unknown") or "unknown").lower()
            result = OrderResult(
                order_id=str(order.get("id", f"close_{int(time.time() * 1000)}")),
                exchange=self.exchange_id,
                symbol=symbol,
                direction=direction,
                price=fill_price,
                quantity=filled_qty,
                status=status,
                is_paper=False,
                timestamp=int(time.time()),
                raw={
                    **order,
                    "reason": reason,
                    "contract_close": True,
                    "close_latency_s": time.perf_counter() - close_started,
                },
            )

            closed = await self.risk_manager.close_position(self.exchange_id, symbol, fill_price)
            if closed is None:
                logger.critical(
                    "{} exchange close succeeded for {} but risk state was not closed",
                    self.exchange_id,
                    symbol,
                )
                await self.event_bus.publish("ALERT_CRITICAL", {
                    "type": "executor_close_risk_sync_failed",
                    "exchange": self.exchange_id,
                    "symbol": symbol,
                    "order_id": result.order_id,
                    "reason": reason,
                    "ts": int(time.time()),
                })
            else:
                await self.event_bus.publish("POSITION_CLOSED", {
                    "position": closed,
                    "reason": reason,
                    "price": fill_price,
                    "order_id": result.order_id,
                })

            logger.info("{} close_position {} {} qty={} @ {}", self.exchange_id, close_side, symbol, filled_qty, fill_price)
            return result
        except Exception as exc:
            logger.error("{} close_position failed for {}: {}", self.exchange_id, symbol, sanitize_exception(exc))
            logger.opt(exception=True).debug("{} close_position stack trace", self.exchange_id)
            return None

    async def _handle_signal(self, payload: Any) -> None:
        signal: TradingSignal = payload
        if signal.exchange != self.exchange_id:
            return

        if self.config.paper_mode:
            # Atomic approve + open under lock (no race window)
            approved, reason, size, pos = await self.risk_manager.approve_and_open(signal)
            if not approved:
                logger.debug("Signal rejected for {}/{}: {}", signal.exchange, signal.symbol, reason)
                return
            await self._paper_execute_with_pos(signal, size, pos)
        else:
            # For live: approve + reserve slot atomically, then execute on exchange
            approved, reason, size, pos = await self.risk_manager.approve_and_open(
                signal,
                reserve_until_fill=True,
            )
            if not approved:
                logger.debug("Signal rejected for {}/{}: {}", signal.exchange, signal.symbol, reason)
                return
            result = await self._live_execute(signal, size, pos)
            if result is None:
                # Exchange order never reached a fill — drop the reservation
                # WITHOUT logging a fake closed trade or touching equity.
                await self.risk_manager.cancel_reserved_position(
                    signal.exchange, signal.symbol,
                )

    async def _handle_stop_loss(self, payload: Any) -> None:
        exchange = payload.get("exchange", "")
        symbol = payload.get("symbol", "")
        price = float(payload.get("price", 0))
        if exchange != self.exchange_id:
            return
        pos = await self.risk_manager.close_position(exchange, symbol, price)
        if pos:
            # OCO: SL triggered → cancel TP on exchange
            if self._order_placer:
                await self._order_placer.handle_sl_filled(symbol)
                self._order_placer.remove_tracking(symbol)
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos, "reason": "stop_loss", "price": price,
            })

    async def _handle_take_profit(self, payload: Any) -> None:
        exchange = payload.get("exchange", "")
        symbol = payload.get("symbol", "")
        price = float(payload.get("price", 0))
        if exchange != self.exchange_id:
            return
        pos = await self.risk_manager.close_position(exchange, symbol, price)
        if pos:
            # OCO: TP triggered → cancel SL on exchange
            if self._order_placer:
                await self._order_placer.handle_tp_filled(symbol)
                self._order_placer.remove_tracking(symbol)
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos, "reason": "take_profit", "price": price,
            })

    async def _handle_kill_switch(self, payload: Any) -> None:
        """Emergency: cancel all open orders, close all positions at market."""
        logger.critical("KILL SWITCH received on {} executor", self.exchange_id)
        closed = await self.risk_manager.activate_kill_switch()
        # Cancel all protective orders tracking
        if self._order_placer:
            for symbol in list(self._order_placer.protective_orders.keys()):
                await self._order_placer.cancel_all_for_symbol(symbol)
        # Cancel all open orders on exchange
        if self._client:
            try:
                open_orders = await self._client.fetch_open_orders()
                for o in open_orders:
                    for _attempt in range(3):
                        try:
                            await self._rate_limiter.acquire()
                            await self._client.cancel_order(o["id"], o.get("symbol"))
                            break
                        except Exception as cancel_exc:
                            if _attempt == 2:
                                logger.error(
                                    "Kill switch: FAILED to cancel order {} after 3 attempts: {}",
                                    o.get('id'), cancel_exc,
                                )
                logger.info("{} cancelled {} open orders", self.exchange_id, len(open_orders))
            except Exception as exc:
                logger.warning("{} failed to cancel orders: {}", self.exchange_id, exc)
        # Close remaining positions at market
        for pos in closed:
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos, "reason": "kill_switch", "price": 0,
            })

    async def _handle_user_order_update(self, payload: Any) -> None:
        """Handle fill/cancel/reject from Binance User Data Stream."""
        # Only the executor for the stream's exchange should process these
        if self.exchange_id != "binance":
            return
        symbol = payload.get("symbol", "")
        exec_type = payload.get("execution_type", "")
        order_status = payload.get("order_status", "")
        order_type = payload.get("order_type", "")
        reduce_only = payload.get("reduce_only", False)

        if exec_type == "TRADE":
            # A fill occurred
            filled_qty = float(payload.get("last_filled_qty", 0))
            filled_price = float(payload.get("last_filled_price", 0))
            cum_qty = float(payload.get("cumulative_filled_qty", 0))
            total_qty = float(payload.get("quantity", 0))
            commission = float(payload.get("commission", 0))
            realized_pnl = float(payload.get("realized_profit", 0))

            logger.info(
                "Fill via user stream: {} qty={:.6f}@{:.2f} cum={:.6f}/{:.6f} pnl={:.4f}",
                symbol, filled_qty, filled_price, cum_qty, total_qty, realized_pnl,
            )

            # ── Sync fill into OrderManager ───────────────────────────────
            client_oid = str(payload.get("client_order_id", ""))
            trade_id = str(payload.get("trade_id", ""))
            if self._order_manager and client_oid:
                await self._order_manager.record_fill(
                    client_order_id=client_oid,
                    fill_id=trade_id or f"fill_{int(time.time()*1000)}",
                    quantity=filled_qty,
                    price=filled_price,
                    fee=commission,
                )

            # Detect if this is a SL or TP fill (reduce_only protective order)
            if reduce_only and order_type in ("STOP_MARKET", "STOP"):
                await self._on_exchange_sl_filled(symbol, filled_price)
            elif reduce_only and order_type in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                await self._on_exchange_tp_filled(symbol, filled_price)
            else:
                # Entry order fill — adjust protective orders if partial
                if order_status == "PARTIALLY_FILLED" and self._order_placer:
                    await self._order_placer.adjust_quantity(symbol, cum_qty)

            await self.event_bus.publish("FILL_CONFIRMED", payload)

        elif exec_type == "CANCELED":
            client_oid = str(payload.get("client_order_id", ""))
            if self._order_manager and client_oid:
                await self._order_manager.cancel_order(client_oid, reason="exchange_cancel")
            logger.info("Order cancelled via user stream: {} orderId={}", symbol, payload.get("order_id"))
            await self.event_bus.publish("ORDER_CANCELLED_EXCHANGE", payload)

        elif exec_type == "REJECTED":
            logger.error("Order REJECTED by exchange: {} reason={}", symbol, payload)
            await self.event_bus.publish("ORDER_REJECTED", payload)

        elif exec_type == "EXPIRED":
            logger.info("Order expired: {} orderId={}", symbol, payload.get("order_id"))

    async def _on_exchange_sl_filled(self, symbol: str, price: float) -> None:
        """Exchange-side SL triggered — sync internal state."""
        logger.warning("Exchange SL triggered for {} @ {:.2f}", symbol, price)
        pos = await self.risk_manager.close_position(self.exchange_id, symbol, price)
        if pos:
            if self._order_placer:
                await self._order_placer.handle_sl_filled(symbol)
                self._order_placer.remove_tracking(symbol)
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos, "reason": "exchange_stop_loss", "price": price,
            })

    async def _on_exchange_tp_filled(self, symbol: str, price: float) -> None:
        """Exchange-side TP triggered — sync internal state."""
        logger.info("Exchange TP triggered for {} @ {:.2f}", symbol, price)
        pos = await self.risk_manager.close_position(self.exchange_id, symbol, price)
        if pos:
            if self._order_placer:
                await self._order_placer.handle_tp_filled(symbol)
                self._order_placer.remove_tracking(symbol)
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos, "reason": "exchange_take_profit", "price": price,
            })

    async def _handle_user_stream_lost(self, payload: Any) -> None:
        """User data stream disconnected — enter safety mode."""
        logger.critical("User data stream LOST — entering safety mode, blocking new trades")
        # Trip the circuit breaker to prevent new entries while stream is down
        self.risk_manager._circuit_breaker.trip("user_stream_disconnected")
        # Activate safe mode
        self.risk_manager.safe_mode.activate(
            SafeModeReason.USER_STREAM_LOST,
            detail=f"exchange={self.exchange_id}",
        )

    async def _handle_user_stream_connected(self, payload: Any) -> None:
        """User data stream reconnected — clear transient safety trips and resume normal operation."""
        logger.info("User data stream reconnected — resuming normal operation")
        cb = self.risk_manager._circuit_breaker
        if cb.clear_if_reason("user_stream_disconnected"):
            logger.info("Circuit breaker reset after user stream reconnection")
        elif cb.tripped and cb.trip_reason.startswith("reconciliation_mismatch"):
            cb.reset()
            logger.info("Circuit breaker reset after reconciliation recovery")
        self.risk_manager.safe_mode.deactivate(SafeModeReason.USER_STREAM_LOST)

    async def run(self) -> None:
        self._running = True
        await self._init_client()
        self.event_bus.subscribe("SIGNAL", self._handle_signal)
        self.event_bus.subscribe("STOP_LOSS", self._handle_stop_loss)
        self.event_bus.subscribe("TAKE_PROFIT", self._handle_take_profit)
        self.event_bus.subscribe("KILL_SWITCH", self._handle_kill_switch)
        self.event_bus.subscribe("USER_ORDER_UPDATE", self._handle_user_order_update)
        self.event_bus.subscribe("USER_STREAM_LOST", self._handle_user_stream_lost)
        self.event_bus.subscribe("USER_STREAM_CONNECTED", self._handle_user_stream_connected)
        logger.info("{} CEX executor started (paper_mode={})", self.exchange_id, self.config.paper_mode)
        while self._running:
            await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        for event, handler in [
            ("SIGNAL", self._handle_signal),
            ("STOP_LOSS", self._handle_stop_loss),
            ("TAKE_PROFIT", self._handle_take_profit),
            ("KILL_SWITCH", self._handle_kill_switch),
            ("USER_ORDER_UPDATE", self._handle_user_order_update),
            ("USER_STREAM_LOST", self._handle_user_stream_lost),
            ("USER_STREAM_CONNECTED", self._handle_user_stream_connected),
        ]:
            self.event_bus.unsubscribe(event, handler)
        if self._client:
            try:
                await self._client.close()
            except Exception as exc:
                logger.warning("{} client close failed: {}", self.exchange_id, exc)
            finally:
                self._client = None

    async def close(self) -> None:
        """Compatibility alias used by main shutdown sequence."""
        await self.stop()
