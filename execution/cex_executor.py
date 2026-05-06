from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from core.safe_mode import SafeModeReason
from engine.signal_generator import TradingSignal
from execution.order_manager import OrderManager
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
        if cfg.get("testnet"):
            # Normalize type: ccxt expects 'future' not 'futures'
            raw_type = cfg.get("type", "future")
            if raw_type == "futures":
                raw_type = "future"
            params["options"] = {
                "defaultType": raw_type,
                # Skip sapi calls (spot auth) — testnet keys only work on futures endpoints
                "fetchCurrencies": False,
                "fetchMargins": False,
                "warnOnFetchOpenOrdersWithoutSymbol": False,
                # Restrict load_markets/fetch_balance to futures-only — without this,
                # ccxt issues signed sapi calls to api.binance.com (mainnet) for spot
                # margin/capital lookup, which fail with -2008 under testnet keys.
                "fetchMarkets": ["linear", "inverse"],
            }
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
                    logger.debug("{} set_margin_mode({}, {}): {}", self.exchange_id, margin_mode, sym, e)
                try:
                    await self._client.set_leverage(leverage, sym)
                except Exception as e:
                    logger.debug("{} set_leverage({}, {}): {}", self.exchange_id, leverage, sym, e)
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
        """Paper execute with pre-opened position (from approve_and_open)."""
        slippage = self.config.get_value("backtest", "slippage_pct") or 0.0002
        fill_price = signal.price * (1 + slippage if signal.is_long else 1 - slippage)
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
        await self.event_bus.publish("ORDER_FILLED", result)
        logger.info("Paper order filled: {} {}/{} @ {:.2f}", signal.direction.upper(), signal.exchange, signal.symbol, fill_price)
        return result

    async def _paper_execute(self, signal: TradingSignal, size: float) -> OrderResult:
        slippage = self.config.get_value("backtest", "slippage_pct") or 0.0002
        fill_price = signal.price * (1 + slippage if signal.is_long else 1 - slippage)
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
        await self.event_bus.publish("ORDER_FILLED", result)
        logger.info("Paper order filled: {} {}/{} @ {:.2f}", signal.direction.upper(), signal.exchange, signal.symbol, fill_price)
        return result

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
                order = await self._client.create_market_order(
                    symbol=signal.symbol, side=side, amount=amount, params={},
                )
            else:
                limit_params: dict[str, Any] = {}
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
                    order = await self._client.create_limit_order(
                        symbol=signal.symbol, side=side, amount=amount,
                        price=signal.price, params=limit_params,
                    )
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
                status=order.get("status", "unknown"),
                is_paper=False,
                timestamp=int(time.time()),
                raw=order,
            )
            if result.status in ("filled", "closed"):
                # Use pre-reserved position if available, otherwise open fresh
                if reserved_pos is not None:
                    pos = reserved_pos
                    pos.current_price = fill_price
                    pos.size = filled_qty
                else:
                    pos = await self.risk_manager.open_position(signal, filled_qty * fill_price)
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
                        logger.warning(
                            "Exchange-side SL/TP unavailable for {} — using bot-managed exits only: {}",
                            signal.symbol,
                            exc,
                        )
                        await self.event_bus.publish("ALERT_WARNING", {
                            "type": "sl_fallback_local",
                            "symbol": signal.symbol,
                            "error": str(exc),
                        })
                    except Exception as exc:
                        logger.critical(
                            "FAILED to place exchange-side SL for {} — tripping circuit breaker: {}",
                            signal.symbol, exc,
                        )
                        # Position is unprotected — trip circuit breaker to prevent more entries
                        self.risk_manager._circuit_breaker.trip(
                            f"sl_placement_failed:{signal.symbol}"
                        )
                        await self.event_bus.publish("ALERT_CRITICAL", {
                            "type": "sl_placement_failed",
                            "symbol": signal.symbol,
                            "error": str(exc),
                        })
            elif result.status == "partially_filled" and filled_qty > 0:
                if reserved_pos is not None:
                    pos = reserved_pos
                    pos.current_price = fill_price
                    pos.size = filled_qty
                else:
                    pos = await self.risk_manager.open_position(signal, filled_qty * fill_price)
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
                        logger.warning(
                            "Exchange-side SL/TP unavailable for partial fill {} — using bot-managed exits only: {}",
                            signal.symbol,
                            exc,
                        )
                    except Exception as exc:
                        logger.critical("SL placement failed for partial fill {}: {}", signal.symbol, exc)
                        self.risk_manager._circuit_breaker.trip(
                            f"sl_placement_failed:{signal.symbol}"
                        )
            else:
                logger.warning("{} order not filled: status={}", self.exchange_id, result.status)
                await self.event_bus.publish("ORDER_FAILED", result)
            return result
        except Exception as exc:
            logger.exception("{} live order failed: {}", self.exchange_id, exc)
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
                await self._rate_limiter.acquire()
                child = await self._client.create_limit_order(
                    symbol=signal.symbol, side=side, amount=chunk_amt,
                    price=signal.price, params=limit_params,
                )
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
        max_retries: int = 3, wait_sec: float = 5.0,
    ) -> dict:
        """Poll for fill; cancel & replace at market after max_retries."""
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

        # Cancel the limit order and fall back to market
        try:
            await self._rate_limiter.acquire()
            await self._client.cancel_order(order_id, signal.symbol)
            logger.info("{} cancelled unfilled limit order {}, placing market order",
                        self.exchange_id, order_id)
        except Exception as exc:
            logger.warning("{} cancel failed (may already be filled): {}", self.exchange_id, exc)

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
        market_order = await self._client.create_market_order(
            symbol=signal.symbol, side=side, amount=remaining, params={},
        )
        return market_order

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
            approved, reason, size, pos = await self.risk_manager.approve_and_open(signal)
            if not approved:
                logger.debug("Signal rejected for {}/{}: {}", signal.exchange, signal.symbol, reason)
                return
            result = await self._live_execute(signal, size, pos)
            if result is None:
                # Exchange failed — release the reserved position
                await self.risk_manager.close_position(signal.exchange, signal.symbol, signal.price)

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
