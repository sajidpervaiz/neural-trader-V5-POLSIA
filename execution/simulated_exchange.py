from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.cex_executor import OrderResult
from execution.order_manager import Order, OrderManager, OrderSide, OrderStatus, OrderType
from execution.risk_manager import RiskManager


@dataclass
class SimulatedDepthLevel:
    price: float
    quantity: float


@dataclass
class SimulatedOrderbookSnapshot:
    bids: list[SimulatedDepthLevel]
    asks: list[SimulatedDepthLevel]


def _cfg(config: Config, *keys: str, default: Any = None) -> Any:
    getter = getattr(config, "get_value", None)
    if not callable(getter):
        return default
    try:
        return getter(*keys, default=default)
    except TypeError:
        try:
            return getter(*keys)
        except Exception:
            return default
    except Exception:
        return default


def _client_order_id(signal: TradingSignal, exchange_id: str) -> str:
    sig_time = int(getattr(signal, "timestamp", 0) or 0)
    raw = (
        f"paper|{exchange_id}|{signal.symbol}|{signal.direction}|"
        f"{sig_time}|{round(float(signal.price), 8)}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"nt5-paper-{digest}"


def _close_client_order_id(exchange_id: str, symbol: str, direction: str) -> str:
    raw = f"paper-close|{exchange_id}|{symbol}|{direction}|{time.time_ns()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"nt5-paper-close-{digest}"


class SimulatedExchangeExecutor:
    """Paper-mode exchange that follows the live submit/ack/fill lifecycle.

    It is intentionally an exchange executor, not a backtester shortcut. Signals
    still pass through RiskManager reservations, OrderManager idempotency, fill
    accounting, and event-bus telemetry before a position becomes active.
    """

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
        self.exchange_id = exchange_id.lower()
        self._order_manager = order_manager
        self._running = False
        self._completion_tasks: set[asyncio.Task] = set()
        self._last_prices: dict[str, float] = {}

        exec_cfg = _cfg(config, "execution", default={}) or {}
        sim_cfg = exec_cfg.get("simulated_exchange", {}) if isinstance(exec_cfg, dict) else {}
        self._ack_latency_s = max(0.0, float(sim_cfg.get("ack_latency_ms", 15.0))) / 1000.0
        self._fill_latency_s = max(0.0, float(sim_cfg.get("fill_latency_ms", 35.0))) / 1000.0
        self._partial_probability = min(1.0, max(0.0, float(sim_cfg.get("partial_fill_probability", 0.0))))
        self._partial_ratio = min(1.0, max(0.01, float(sim_cfg.get("partial_fill_ratio", 0.5))))
        self._partial_completion_s = max(0.0, float(sim_cfg.get("partial_fill_completion_ms", 250.0))) / 1000.0
        self._reject_probability = min(1.0, max(0.0, float(sim_cfg.get("reject_probability", 0.0))))
        self._max_slippage_bps = max(0.0, float(sim_cfg.get("max_slippage_bps", 50.0)))
        self._synthetic_spread_bps = max(0.0, float(sim_cfg.get("synthetic_spread_bps", 2.0)))
        self._synthetic_depth_usd = max(1.0, float(sim_cfg.get("synthetic_orderbook_depth_usd", 1_000_000.0)))
        self._default_price = max(1.0, float(sim_cfg.get("default_price", 50_000.0)))

    @property
    def is_paper(self) -> bool:
        return True

    def _publish_pipeline_latency(self, signal: TradingSignal, stage: str, latency_s: float) -> None:
        try:
            payload = {
                "stage": str(stage),
                "exchange": self.exchange_id,
                "symbol": getattr(signal, "symbol", "unknown"),
                "latency_s": max(0.0, float(latency_s)),
                "ts": time.time(),
                "paper_simulated": True,
            }
            publish_nowait = getattr(self.event_bus, "publish_nowait", None)
            if callable(publish_nowait):
                publish_nowait("PIPELINE_LATENCY", payload)
            else:
                asyncio.create_task(self.event_bus.publish("PIPELINE_LATENCY", payload))
        except Exception as exc:
            logger.debug("{} simulated latency publish failed: {}", self.exchange_id, exc)

    def _effective_fill_price(self, signal: TradingSignal) -> tuple[float, float, float]:
        bt = _cfg(self.config, "backtest", default={}) or {}
        slippage = max(0.0, float(bt.get("slippage_pct", 0.0002)))
        commission = max(0.0, float(bt.get("commission_pct", 0.0004)))
        if self._max_slippage_bps > 0:
            slippage = min(slippage, self._max_slippage_bps / 10_000.0)
        effective_cost = slippage + commission
        price = float(signal.price)
        fill_price = price * (1 + effective_cost if signal.is_long else 1 - effective_cost)
        return fill_price, slippage, commission

    def _effective_exit_price(self, pos: Any, requested_price: float) -> tuple[float, float, float]:
        bt = _cfg(self.config, "backtest", default={}) or {}
        slippage = max(0.0, float(bt.get("slippage_pct", 0.0002)))
        commission = max(0.0, float(bt.get("commission_pct", 0.0005)))
        effective_cost = slippage + commission
        is_long = bool(getattr(pos, "is_long", str(getattr(pos, "direction", "")).lower() == "long"))
        fill_price = requested_price * (1 - effective_cost if is_long else 1 + effective_cost)
        return fill_price, slippage, commission

    async def _create_order(
        self,
        signal: TradingSignal,
        quantity: float,
        fill_price: float,
    ) -> tuple[Order | None, str, bool]:
        exchange_order_id = f"sim_{self.exchange_id}_{int(time.time() * 1_000_000)}"
        if self._order_manager is None:
            return None, exchange_order_id, True

        side = OrderSide.BUY if signal.is_long else OrderSide.SELL
        success, order, reason = await self._order_manager.place_order(
            exchange=self.exchange_id,
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            order_type=OrderType.MARKET,
            client_order_id=_client_order_id(signal, self.exchange_id),
            metadata={
                "paper": True,
                "simulated_exchange": True,
                "signal_score": float(getattr(signal, "score", 0.0) or 0.0),
                "signal_timestamp": int(getattr(signal, "timestamp", 0) or 0),
            },
        )
        if not success or order is None:
            logger.warning(
                "{} simulated order rejected by OrderManager: {} {}/{} reason={}",
                self.exchange_id, signal.direction, signal.symbol, quantity, reason,
            )
            return None, exchange_order_id, False

        if reason == "idempotent_retry":
            return order, order.exchange_order_id or exchange_order_id, False

        confirmed = await self._order_manager.confirm_order_submission(
            client_order_id=order.client_order_id or "",
            exchange_order_id=exchange_order_id,
        )
        return confirmed or order, exchange_order_id, True

    async def _record_fill(
        self,
        order: Order | None,
        quantity: float,
        fill_price: float,
        *,
        fill_tag: str,
    ) -> None:
        if self._order_manager is None or order is None or not order.client_order_id:
            return
        await self._order_manager.record_fill(
            client_order_id=order.client_order_id,
            fill_id=f"{fill_tag}_{int(time.time() * 1_000_000)}",
            quantity=quantity,
            price=fill_price,
            fee=0.0,
        )

    def _order_result(
        self,
        signal: TradingSignal,
        exchange_order_id: str,
        fill_price: float,
        quantity: float,
        status: str,
        raw: dict[str, Any] | None = None,
    ) -> OrderResult:
        return OrderResult(
            order_id=exchange_order_id,
            exchange=self.exchange_id,
            symbol=signal.symbol,
            direction=signal.direction,
            price=fill_price,
            quantity=quantity,
            status=status,
            is_paper=True,
            timestamp=int(time.time()),
            raw=raw or {},
        )

    async def execute_signal(self, signal: TradingSignal, size: float) -> OrderResult | None:
        signal.exchange = self.exchange_id
        approved, reason, approved_size, pos = await self.risk_manager.approve_and_open(
            signal,
            reserve_until_fill=True,
        )
        if not approved:
            logger.debug("Simulated signal rejected for {}/{}: {}", signal.exchange, signal.symbol, reason)
            return None
        return await self._simulate_entry_fill(signal, approved_size or size, pos)

    async def _simulate_entry_fill(
        self,
        signal: TradingSignal,
        size: float,
        reserved_pos: Any,
    ) -> OrderResult | None:
        submit_started = time.perf_counter()
        fill_price, slippage, commission = self._effective_fill_price(signal)
        quantity = size / fill_price if fill_price > 0 else 0.0
        if quantity <= 0:
            await self.risk_manager.cancel_reserved_position(signal.exchange, signal.symbol)
            return None

        order, exchange_order_id, is_new_order = await self._create_order(signal, quantity, fill_price)
        if order is None and self._order_manager is not None:
            await self.risk_manager.cancel_reserved_position(signal.exchange, signal.symbol)
            return None

        if not is_new_order and order is not None and order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }:
            await self.risk_manager.cancel_reserved_position(signal.exchange, signal.symbol)
            status = order.status.value
            return self._order_result(
                signal,
                exchange_order_id,
                order.average_fill_price or fill_price,
                order.cumulative_quantity,
                status,
                {"idempotent_retry": True, "simulated_exchange": True},
            )

        if self._ack_latency_s > 0:
            await asyncio.sleep(self._ack_latency_s)
        self._publish_pipeline_latency(signal, "order_submit_ack", time.perf_counter() - submit_started)

        if self._reject_probability > 0 and random.random() < self._reject_probability:
            await self.risk_manager.cancel_reserved_position(signal.exchange, signal.symbol)
            if self._order_manager is not None and order is not None and order.client_order_id:
                await self._order_manager.cancel_order(order.client_order_id, reason="simulated_exchange_reject")
            result = self._order_result(
                signal,
                exchange_order_id,
                fill_price,
                0.0,
                "rejected",
                {"simulated_exchange": True, "reject_probability": self._reject_probability},
            )
            await self.event_bus.publish("ORDER_REJECTED", result)
            return result

        if self._fill_latency_s > 0:
            await asyncio.sleep(self._fill_latency_s)

        raw = {
            "simulated_exchange": True,
            "slippage_pct": slippage,
            "commission_pct_charged_in_effective_price": commission,
            "estimated_entry_fee_usd": size * commission,
            "signal_price": float(signal.price),
        }

        should_partial = self._partial_probability > 0 and random.random() < self._partial_probability
        if should_partial and self._partial_ratio < 0.999:
            partial_qty = max(0.0, min(quantity, quantity * self._partial_ratio))
            remaining_qty = max(0.0, quantity - partial_qty)
            await self._record_fill(order, partial_qty, fill_price, fill_tag="sim_partial")
            await self.risk_manager.rebase_position_to_fill(
                self.exchange_id,
                signal.symbol,
                fill_price,
                partial_qty,
            )
            partial_result = self._order_result(
                signal,
                exchange_order_id,
                fill_price,
                partial_qty,
                "partially_filled",
                {**raw, "partial_fill_ratio": self._partial_ratio},
            )
            await self.event_bus.publish("ORDER_PARTIALLY_FILLED", partial_result)

            if remaining_qty > 0:
                task = asyncio.create_task(
                    self._complete_partial_fill(signal, order, exchange_order_id, fill_price, partial_qty, remaining_qty, raw)
                )
                self._completion_tasks.add(task)
                task.add_done_callback(self._completion_tasks.discard)
            logger.info(
                "{} simulated partial fill: {} {}/{} qty={:.6f}/{:.6f} @ {:.4f}",
                self.exchange_id, signal.direction, signal.exchange, signal.symbol,
                partial_qty, quantity, fill_price,
            )
            return partial_result

        await self._record_fill(order, quantity, fill_price, fill_tag="sim_fill")
        await self.risk_manager.rebase_position_to_fill(
            self.exchange_id,
            signal.symbol,
            fill_price,
            quantity,
        )
        result = self._order_result(signal, exchange_order_id, fill_price, quantity, "filled", raw)
        await self.event_bus.publish("ORDER_FILLED", result)
        logger.info(
            "{} simulated fill: {} {}/{} qty={:.6f} @ {:.4f}",
            self.exchange_id, signal.direction, signal.exchange, signal.symbol,
            quantity, fill_price,
        )
        return result

    async def _complete_partial_fill(
        self,
        signal: TradingSignal,
        order: Order | None,
        exchange_order_id: str,
        fill_price: float,
        partial_qty: float,
        remaining_qty: float,
        raw: dict[str, Any],
    ) -> None:
        try:
            if self._partial_completion_s > 0:
                await asyncio.sleep(self._partial_completion_s)
            await self._record_fill(order, remaining_qty, fill_price, fill_tag="sim_fill")
            total_qty = partial_qty + remaining_qty
            await self.risk_manager.rebase_position_to_fill(
                self.exchange_id,
                signal.symbol,
                fill_price,
                total_qty,
            )
            result = self._order_result(
                signal,
                exchange_order_id,
                fill_price,
                total_qty,
                "filled",
                {**raw, "completed_partial_fill": True},
            )
            await self.event_bus.publish("ORDER_FILLED", result)
        except Exception as exc:
            logger.warning("{} simulated partial completion failed: {}", self.exchange_id, exc)

    async def close_position(
        self,
        symbol: str,
        price: float,
        *,
        reason: str = "simulated_close",
    ) -> OrderResult | None:
        """Close a paper position through an exchange-style reduce-only order."""
        positions = getattr(self.risk_manager, "positions", {})
        pos = positions.get(f"{self.exchange_id}:{symbol}") if isinstance(positions, dict) else None
        if pos is None:
            return None

        quantity = abs(float(getattr(pos, "size", 0.0) or 0.0))
        requested_price = float(price or getattr(pos, "current_price", 0.0) or getattr(pos, "entry_price", 0.0) or 0.0)
        if quantity <= 0 or requested_price <= 0:
            return None

        close_started = time.perf_counter()
        fill_price, slippage, commission = self._effective_exit_price(pos, requested_price)
        direction = str(getattr(pos, "direction", "long") or "long").lower()
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        exchange_order_id = f"sim_{self.exchange_id}_close_{int(time.time() * 1_000_000)}"
        order: Order | None = None

        if self._order_manager is not None:
            success, order, om_reason = await self._order_manager.place_order(
                exchange=self.exchange_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=fill_price,
                order_type=OrderType.MARKET,
                client_order_id=_close_client_order_id(self.exchange_id, symbol, direction),
                metadata={
                    "paper": True,
                    "simulated_exchange": True,
                    "reduce_only": True,
                    "close_reason": reason,
                },
            )
            if not success or order is None:
                logger.warning(
                    "{} simulated close rejected by OrderManager: {} qty={} reason={}",
                    self.exchange_id, symbol, quantity, om_reason,
                )
                return None
            if om_reason != "idempotent_retry":
                order = await self._order_manager.confirm_order_submission(
                    client_order_id=order.client_order_id or "",
                    exchange_order_id=exchange_order_id,
                ) or order

        if self._ack_latency_s > 0:
            await asyncio.sleep(self._ack_latency_s)
        self._publish_pipeline_latency(
            TradingSignal(
                exchange=self.exchange_id,
                symbol=symbol,
                direction=direction,
                score=0.0,
                technical_score=0.0,
                ml_score=0.0,
                sentiment_score=0.0,
                macro_score=0.0,
                news_score=0.0,
                orderbook_score=0.0,
                regime="paper_close",
                regime_confidence=0.0,
                price=requested_price,
                atr=max(requested_price * 0.01, 1e-8),
                stop_loss=0.0,
                take_profit=0.0,
                timestamp=int(time.time() * 1000),
                metadata={"source": reason, "paper": True, "reduce_only": True},
            ),
            "order_close_ack",
            time.perf_counter() - close_started,
        )
        if self._fill_latency_s > 0:
            await asyncio.sleep(self._fill_latency_s)

        await self._record_fill(order, quantity, fill_price, fill_tag="sim_close")
        closed = await self.risk_manager.close_position(self.exchange_id, symbol, requested_price)
        if closed is None:
            return None

        result = OrderResult(
            order_id=exchange_order_id,
            exchange=self.exchange_id,
            symbol=symbol,
            direction=direction,
            price=fill_price,
            quantity=quantity,
            status="closed",
            is_paper=True,
            timestamp=int(time.time()),
            raw={
                "simulated_exchange": True,
                "reduce_only": True,
                "close_reason": reason,
                "slippage_pct": slippage,
                "commission_pct_charged_in_effective_price": commission,
                "requested_price": requested_price,
            },
        )
        await self.event_bus.publish("POSITION_CLOSED", {
            "position": closed,
            "reason": reason,
            "price": fill_price,
            "exchange": self.exchange_id,
            "symbol": symbol,
            "order_id": exchange_order_id,
            "paper": True,
            "simulated_exchange": True,
        })
        logger.info(
            "{} simulated close: {} {} qty={:.6f} @ {:.4f} reason={}",
            self.exchange_id, side.value, symbol, quantity, fill_price, reason,
        )
        return result

    async def _handle_signal(self, payload: Any) -> None:
        signal: TradingSignal = payload
        if getattr(signal, "exchange", "").lower() != self.exchange_id:
            return
        signal.exchange = self.exchange_id
        approved, reason, size, pos = await self.risk_manager.approve_and_open(
            signal,
            reserve_until_fill=True,
        )
        if not approved:
            logger.debug("Simulated signal rejected for {}/{}: {}", signal.exchange, signal.symbol, reason)
            return
        result = await self._simulate_entry_fill(signal, size, pos)
        if result is None:
            await self.risk_manager.cancel_reserved_position(signal.exchange, signal.symbol)

    async def _handle_stop_loss(self, payload: Any) -> None:
        exchange = str(payload.get("exchange", "")).lower()
        symbol = str(payload.get("symbol", ""))
        price = float(payload.get("price", 0.0) or 0.0)
        if exchange != self.exchange_id:
            return
        await self.close_position(symbol, price, reason="simulated_stop_loss")

    async def _handle_take_profit(self, payload: Any) -> None:
        exchange = str(payload.get("exchange", "")).lower()
        symbol = str(payload.get("symbol", ""))
        price = float(payload.get("price", 0.0) or 0.0)
        if exchange != self.exchange_id:
            return
        await self.close_position(symbol, price, reason="simulated_take_profit")

    async def _handle_kill_switch(self, payload: Any) -> None:
        logger.critical("KILL SWITCH received on {} simulated executor", self.exchange_id)
        positions = list((getattr(self.risk_manager, "positions", {}) or {}).values())
        for pos in positions:
            if str(getattr(pos, "exchange", "")).lower() != self.exchange_id:
                continue
            symbol = str(getattr(pos, "symbol", "") or "")
            price = float(getattr(pos, "current_price", 0.0) or getattr(pos, "entry_price", 0.0) or 0.0)
            if symbol and price > 0:
                await self.close_position(symbol, price, reason="simulated_kill_switch")
        closed = await self.risk_manager.activate_kill_switch()
        for pos in closed:
            if getattr(pos, "exchange", "").lower() != self.exchange_id:
                continue
            await self.event_bus.publish("POSITION_CLOSED", {
                "position": pos,
                "reason": "simulated_kill_switch",
                "price": 0,
                "exchange": self.exchange_id,
                "symbol": getattr(pos, "symbol", ""),
            })

    async def _handle_market_price(self, payload: Any) -> None:
        symbol = ""
        price = 0.0
        exchange = ""
        if isinstance(payload, dict):
            symbol = str(payload.get("symbol", ""))
            exchange = str(payload.get("exchange", "")).lower()
            price = float(payload.get("price") or payload.get("close") or 0.0)
        else:
            symbol = str(getattr(payload, "symbol", ""))
            exchange = str(getattr(payload, "exchange", "")).lower()
            price = float(getattr(payload, "price", 0.0) or getattr(payload, "close", 0.0) or 0.0)
        if price > 0 and symbol and (not exchange or exchange == self.exchange_id):
            self._last_prices[symbol] = price

    async def get_orderbook_snapshot(self, symbol: str, depth: int = 10) -> SimulatedOrderbookSnapshot:
        depth = max(1, int(depth or 1))
        mid = float(self._last_prices.get(symbol) or self._default_price)
        spread = mid * self._synthetic_spread_bps / 10_000.0
        step = max(mid * 0.5 / 10_000.0, spread / 2.0, 0.01)
        usd_per_level = self._synthetic_depth_usd / depth
        qty_per_level = max(0.000001, usd_per_level / mid)
        bids = [
            SimulatedDepthLevel(price=mid - spread / 2.0 - i * step, quantity=qty_per_level)
            for i in range(depth)
        ]
        asks = [
            SimulatedDepthLevel(price=mid + spread / 2.0 + i * step, quantity=qty_per_level)
            for i in range(depth)
        ]
        return SimulatedOrderbookSnapshot(bids=bids, asks=asks)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._order_manager is None:
            return False
        order = self._order_manager.client_order_map.get(order_id)
        if order is None:
            order = self._order_manager.exchange_order_map.get((self.exchange_id, order_id))
        if order is None:
            for candidate in self._order_manager.get_open_orders(self.exchange_id):
                if candidate.symbol == symbol and candidate.exchange_order_id == order_id:
                    order = candidate
                    break
        if order is None or not order.client_order_id:
            return False
        ok, _order, _reason = await self._order_manager.cancel_order(
            order.client_order_id,
            reason="simulated_cancel",
        )
        if ok:
            await self.event_bus.publish("ORDER_CANCELLED_EXCHANGE", {
                "exchange": self.exchange_id,
                "symbol": symbol,
                "order_id": order_id,
                "paper": True,
                "simulated_exchange": True,
            })
        return bool(ok)

    async def run(self) -> None:
        self._running = True
        self.event_bus.subscribe("SIGNAL", self._handle_signal)
        self.event_bus.subscribe("STOP_LOSS", self._handle_stop_loss)
        self.event_bus.subscribe("TAKE_PROFIT", self._handle_take_profit)
        self.event_bus.subscribe("KILL_SWITCH", self._handle_kill_switch)
        self.event_bus.subscribe("TICK", self._handle_market_price)
        self.event_bus.subscribe("CANDLE", self._handle_market_price)
        logger.info("{} simulated exchange executor started (paper mode)", self.exchange_id)
        while self._running:
            await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        for event, handler in [
            ("SIGNAL", self._handle_signal),
            ("STOP_LOSS", self._handle_stop_loss),
            ("TAKE_PROFIT", self._handle_take_profit),
            ("KILL_SWITCH", self._handle_kill_switch),
            ("TICK", self._handle_market_price),
            ("CANDLE", self._handle_market_price),
        ]:
            self.event_bus.unsubscribe(event, handler)
        if self._completion_tasks:
            await asyncio.gather(*list(self._completion_tasks), return_exceptions=True)

    async def close(self) -> None:
        await self.stop()
