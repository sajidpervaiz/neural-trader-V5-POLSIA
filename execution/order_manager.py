"""
Production-grade Order Manager with idempotency, lifecycle tracking,
and self-trade prevention.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from core.circuit_breaker import CircuitBreaker
from core.persistent_idempotency import PersistentIdempotencyManager
from core.retry import RetryPolicy
from execution.order_splitter import TWAPExecutor, IcebergExecutor
from execution.shadow_sl import ShadowStopManager


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY = "post_only"
    IOC = "ioc"


class OrderLifecycleStage(Enum):
    VALIDATION = "VALIDATION"
    RISK_CHECK = "RISK_CHECK"
    SUBMISSION = "SUBMISSION"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    PARTIAL_FILL = "PARTIAL_FILL"
    COMPLETE_FILL = "COMPLETE_FILL"
    CANCELLATION = "CANCELLATION"
    COMPLETION = "COMPLETION"


@dataclass
class OrderStage:
    stage: OrderLifecycleStage
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class OrderFill:
    """Represents a partial or complete fill"""
    fill_id: str
    timestamp: int
    quantity: float
    price: float
    fee: float = 0.0
    fee_currency: str = "USDT"

    @property
    def total_value(self) -> float:
        """Total value of fill (quantity * price)"""
        return self.quantity * self.price


@dataclass
class Order:
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: float = 0.0
    cumulative_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fee: float = 0.0
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    submitted_at: Optional[int] = None
    filled_at: Optional[int] = None
    exchange_order_id: Optional[str] = None
    venue: Optional[str] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    stages: List[OrderStage] = field(default_factory=list)
    fills: List[OrderFill] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity

    def add_stage(self, stage: OrderLifecycleStage, metadata: Optional[Dict] = None):
        """Record a lifecycle stage."""
        self.stages.append(OrderStage(
            stage=stage,
            timestamp=time.time(),
            metadata=metadata or {},
        ))
        self.updated_at = int(time.time() * 1000)

    # NOTE: Fill tracking is handled exclusively by OrderManager.record_fill()
    # which manages deduplication, audit logging, and lifecycle stages.


class OrderManager:
    """
    Production-grade order manager with:
    - Idempotent order placement
    - Lifecycle tracking
    - Self-trade prevention
    - Order amend/cancel with retry logic
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        circuit_breaker: CircuitBreaker,
        audit_log_path: str = "order_audit_log.jsonl",
        order_state_path: str = "order_state.json",
    ):
        self.config = config
        self.event_bus = event_bus
        self.circuit_breaker = circuit_breaker
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.client_order_map: Dict[str, Order] = {}
        self.exchange_order_map: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.user_orders: Dict[str, Dict[str, Order]] = {}
        # Idempotency manager (persistent)
        self.idempotency = PersistentIdempotencyManager(ttl=86400, max_size=50000, filepath="idempotency_store.json")
        # Retry policy
        self.retry_policy = RetryPolicy(max_attempts=3, base_delay=0.1)
        # Self-trade prevention window
        self.self_trade_window_ms = 60000
        # Audit trail
        self.audit_log: List[Dict] = []
        self.max_audit_log_size = 100000
        self.smart_router: Any = None

        # ── ARMS-V2.1: Order splitting (§13) ────────────────────────────────
        arms_cfg = config.get_value("arms") or {}
        splitting_cfg = arms_cfg.get("order_splitting", {})
        self.twap_executor = TWAPExecutor(
            num_slices=int(splitting_cfg.get("twap_slices", 5)),
            duration_seconds=float(splitting_cfg.get("twap_duration_seconds", 300)),
            size_threshold_usd=float(splitting_cfg.get("twap_threshold_usd", 50_000)),
        )
        self.iceberg_executor = IcebergExecutor(
            visible_pct=float(splitting_cfg.get("iceberg_visible_pct", 0.20)),
            size_threshold_usd=float(splitting_cfg.get("iceberg_threshold_usd", 100_000)),
        )

        # ── ARMS-V2.1: Shadow stop-loss (§13) ────────────────────────────────
        shadow_cfg = arms_cfg.get("shadow_sl", {})
        self.shadow_sl = ShadowStopManager(
            primary_timeout_seconds=float(shadow_cfg.get("primary_timeout", 5.0)),
            max_retries=int(shadow_cfg.get("max_retries", 3)),
            shadow_check_interval=float(shadow_cfg.get("check_interval", 1.0)),
            fallback_grace_seconds=float(shadow_cfg.get("fallback_grace", 10.0)),
        )

        self._lock = asyncio.Lock()
        self._running = False
        self._audit_log_path = audit_log_path
        self._order_state_path = order_state_path
        self._restore_state_on_startup = bool(
            self.config.get_value("execution", "restore_orders_on_startup", default=False)
        )
        if self._restore_state_on_startup:
            self._load_audit_log()
            self._load_order_state()

    def get_split_strategy(self, notional_usd: float) -> Optional[str]:
        """Return 'iceberg', 'twap', or None based on order notional size."""
        if self.iceberg_executor.should_split(notional_usd):
            return "iceberg"
        if self.twap_executor.should_split(notional_usd):
            return "twap"
        return None

    def get_twap_snapshot(self) -> Dict:
        """Active TWAP orders snapshot."""
        active = self.twap_executor.get_active_orders()
        return {
            "active_count": len(active),
            "orders": {
                oid: {
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "total_size": s.total_size,
                    "filled_size": s.filled_size,
                    "remaining_size": s.remaining_size,
                    "children": len(s.children),
                    "status": s.status,
                }
                for oid, s in active.items()
            },
        }

    def get_iceberg_snapshot(self) -> Dict:
        """Active Iceberg orders snapshot."""
        active = self.iceberg_executor.get_active_orders()
        return {
            "active_count": len(active),
            "orders": {
                oid: {
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "total_size": s.total_size,
                    "filled_size": s.filled_size,
                    "remaining_size": s.remaining_size,
                    "children": len(s.children),
                    "status": s.status,
                }
                for oid, s in active.items()
            },
        }

    def get_shadow_sl_snapshot(self) -> Dict:
        """Shadow stop-loss status snapshot."""
        return self.shadow_sl.get_snapshot()

    def _save_audit_log(self):
        import json
        if not self._audit_log_path:
            return
        try:
            with open(self._audit_log_path, "w") as f:
                for entry in self.audit_log:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")

    def _append_audit_entry(self, entry: dict) -> None:
        """Append a single audit entry (O(1) I/O instead of rewriting entire log)."""
        import json
        if not self._audit_log_path:
            return
        try:
            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append audit entry: {e}")

    def _load_audit_log(self):
        import json
        import os
        if not self._audit_log_path:
            return
        if os.path.exists(self._audit_log_path):
            try:
                with open(self._audit_log_path, "r") as f:
                    self.audit_log = [json.loads(line) for line in f]
            except Exception as e:
                logger.error(f"Failed to load audit log: {e}")

    def _save_order_state(self):
        import json
        if not self._order_state_path:
            return
        try:
            # Map exchange order keys to their order_id for proper restoration
            exchange_map_save = {
                f"{k[0]}|{k[1]}": v.order_id
                for k, v in self.exchange_order_map.items()
            }
            state = {
                "orders": {k: self._order_to_dict(v) for k, v in self.orders.items()},
                "client_order_map": list(self.client_order_map.keys()),
                "exchange_order_map": exchange_map_save,
            }
            with open(self._order_state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save order state: {e}")

    def _load_order_state(self):
        import json
        import os
        if not self._order_state_path:
            return
        if os.path.exists(self._order_state_path):
            try:
                with open(self._order_state_path, "r") as f:
                    state = json.load(f)
                self.orders = {k: self._dict_to_order(v) for k, v in state.get("orders", {}).items()}
                self.client_order_map = {k: self.orders[k] for k in state.get("client_order_map", []) if k in self.orders}
                # Restore exchange_order_map with proper (venue, exchange_id) → Order mapping
                exchange_map_raw = state.get("exchange_order_map", {})
                if isinstance(exchange_map_raw, dict):
                    for composite_key, order_id in exchange_map_raw.items():
                        if order_id in self.orders and "|" in composite_key:
                            venue, exch_id = composite_key.split("|", 1)
                            self.exchange_order_map[(venue, exch_id)] = self.orders[order_id]
                else:
                    # Legacy format: list of [venue, exchange_order_id] — best-effort skip
                    logger.warning("Legacy exchange_order_map format detected, skipping restore")
            except Exception as e:
                logger.error(f"Failed to load order state: {e}")

    def _order_to_dict(self, order: 'Order') -> dict:
        from dataclasses import asdict
        d = asdict(order)
        d["side"] = order.side.value
        d["order_type"] = order.order_type.value
        d["status"] = order.status.value
        d["stages"] = [asdict(s) for s in order.stages]
        d["fills"] = [asdict(f) for f in order.fills]
        return d

    def _dict_to_order(self, d: dict) -> 'Order':
        # Reconstruct enums and dataclasses
        d = dict(d)
        d["side"] = OrderSide(d["side"])
        d["order_type"] = OrderType(d["order_type"])
        d["status"] = OrderStatus(d["status"])
        d["stages"] = [OrderStage(OrderLifecycleStage(s["stage"]), s["timestamp"], s.get("metadata", {})) for s in d.get("stages", [])]
        d["fills"] = [OrderFill(**f) for f in d.get("fills", [])]
        return Order(**d)

    def attach_router(self, router: Any) -> None:
        """Attach an optional smart order router for auto venue selection."""
        self.smart_router = router

    def _default_exchange(self) -> str:
        exchanges = self.config.get_value("exchanges") or {}
        for exchange_id, cfg in exchanges.items():
            if isinstance(cfg, dict) and cfg.get("enabled", False):
                return str(exchange_id)
        return "binance"

    async def _resolve_exchange(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        metadata: Dict,
    ) -> tuple[str, Dict]:
        requested = (exchange or "").lower().strip()
        if requested not in {"auto", "best", "router"}:
            return exchange, metadata

        if self.smart_router is None:
            fallback = self._default_exchange()
            metadata["routing_mode"] = "fallback_no_router"
            metadata["requested_exchange"] = exchange
            return fallback, metadata

        try:
            from execution.binance_executor import OrderSide as RouterOrderSide

            router_side = RouterOrderSide.BUY if side == OrderSide.BUY else RouterOrderSide.SELL
            decision = await self.smart_router.route_order(
                symbol=symbol,
                side=router_side,
                quantity=quantity,
                max_venues=3,
                min_score_threshold=0.1,
            )
            if decision is None:
                fallback = self._default_exchange()
                metadata["routing_mode"] = "fallback_no_decision"
                metadata["requested_exchange"] = exchange
                return fallback, metadata

            selected_exchange = str(getattr(decision.recommended_venue, "value", "") or "")
            if not selected_exchange:
                fallback = self._default_exchange()
                metadata["routing_mode"] = "fallback_invalid_decision"
                metadata["requested_exchange"] = exchange
                return fallback, metadata

            metadata["routing_mode"] = "smart_router"
            metadata["requested_exchange"] = exchange
            metadata["routing_decision"] = {
                "recommended_venue": selected_exchange,
                "confidence": float(getattr(decision, "confidence", 0.0)),
                "expected_avg_price": float(getattr(decision, "expected_avg_price", 0.0)),
                "routes": [
                    {
                        "venue": str(getattr(getattr(route, "venue", None), "value", "")),
                        "quantity": float(getattr(route, "quantity", 0.0)),
                        "expected_fill_price": float(getattr(route, "expected_fill_price", 0.0)),
                        "score": float(getattr(route, "score", 0.0)),
                    }
                    for route in list(getattr(decision, "routes", []))
                ],
            }
            return selected_exchange, metadata
        except Exception as exc:
            fallback = self._default_exchange()
            logger.warning("Smart routing failed: {} — falling back to {}", exc, fallback)
            metadata["routing_mode"] = "fallback_error"
            metadata["routing_error"] = str(exc)
            metadata["requested_exchange"] = exchange
            return fallback, metadata

    def generate_client_order_id(self, exchange: str, symbol: str, side: OrderSide) -> str:
        """Generate unique client order ID"""
        timestamp_ms = int(time.time() * 1000)
        unique_part = str(uuid.uuid4())[:8]
        return f"{exchange}-{symbol}-{side.value[0]}-{timestamp_ms}-{unique_part}"

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.LIMIT,
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[Order], str]:
        """
        Place order with idempotency and self-trade prevention.
        
        Returns:
            (success: bool, order: Order, reason: str)
        """
        async with self._lock:
            # Check circuit breaker
            cb_open = False
            if hasattr(self.circuit_breaker, 'get_state'):
                cb_open = self.circuit_breaker.get_state().value == "OPEN"
            elif hasattr(self.circuit_breaker, 'tripped'):
                cb_open = self.circuit_breaker.tripped
            if cb_open:
                return False, None, "circuit_breaker_open"

            # Use caller-provided client order ID when supplied; otherwise generate one.
            if not client_order_id:
                client_order_id = self.generate_client_order_id(exchange, symbol, side)
            
            # Check idempotency
            if self.idempotency.check_and_set(client_order_id):
                cached_order = self.client_order_map.get(client_order_id)
                if cached_order:
                    logger.info(f"Order {client_order_id} already placed (idempotent retry)")
                    return True, cached_order, "idempotent_retry"

            # Self-trade prevention
            metadata = dict(metadata or {})
            exchange, metadata = await self._resolve_exchange(exchange, symbol, side, quantity, metadata)
            self_trade_reason = self._check_self_trade(exchange, symbol, side)
            if self_trade_reason:
                logger.warning(f"Self-trade prevented for {symbol}: {self_trade_reason}")
                return False, None, self_trade_reason

            # ARMS-V2.1: Annotate split strategy recommendation
            notional_usd = quantity * price if price else 0.0
            split_strategy = self.get_split_strategy(notional_usd)
            if split_strategy:
                metadata["split_strategy"] = split_strategy
                logger.info(
                    "Large order detected: {} {} notional=${:.0f} — recommending {}",
                    symbol, side.value, notional_usd, split_strategy,
                )

            # Create order
            order_id = f"ord_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"
            order = Order(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                exchange_order_id=None,
                venue=exchange,
                metadata=metadata,
            )

            # Store order
            self.orders[order_id] = order
            self.client_order_map[client_order_id] = order
            self.pending_orders[order_id] = order
            self.idempotency.set_result(client_order_id, order)
            self._record_audit("ORDER_CREATED", order)

            logger.info(
                f"Order created: {client_order_id} {side.value} {quantity} {symbol} @ {price}"
            )
            
            return True, order, "created"

    def _check_self_trade(self, exchange: str, symbol: str, side: OrderSide) -> Optional[str]:
        """Check for self-trade risk"""
        open_orders = [
            o for o in self.orders.values()
            if o.venue == exchange 
            and o.symbol == symbol
            and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]

        for order in open_orders:
            if order.side != side:
                age_ms = int(time.time() * 1000) - order.created_at
                if age_ms < self.self_trade_window_ms:
                    return f"opposing_order_open ({order.client_order_id})"

        return None

    async def confirm_order_submission(
        self,
        client_order_id: str,
        exchange_order_id: str,
    ) -> Optional[Order]:
        """Confirm order submitted to exchange"""
        async with self._lock:
            order = self.client_order_map.get(client_order_id)
            if not order:
                logger.error(f"Order {client_order_id} not found for confirmation")
                return None

            order.exchange_order_id = exchange_order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = int(time.time() * 1000)
            
            self.exchange_order_map[(order.venue, exchange_order_id)] = order
            self._record_audit("ORDER_SUBMITTED", order)
            logger.debug(f"Order {client_order_id} submitted")

            return order

    async def record_fill(
        self,
        client_order_id: str,
        fill_id: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
    ) -> Optional[Order]:
        """Record a fill for an order"""
        async with self._lock:
            order = self.client_order_map.get(client_order_id)
            if not order:
                logger.error(f"Order {client_order_id} not found for fill")
                return None

            # P0-1: Guard against duplicate fill processing
            if any(f.fill_id == fill_id for f in order.fills):
                logger.warning(f"Duplicate fill_id {fill_id} for {client_order_id} — skipping")
                return order

            fill = OrderFill(
                fill_id=fill_id,
                timestamp=int(time.time() * 1000),
                quantity=quantity,
                price=price,
                fee=fee,
            )

            order.fills.append(fill)
            order.cumulative_quantity += quantity
            order.total_fee += fee
            
            total_value = sum(f.quantity * f.price for f in order.fills)
            order.average_fill_price = total_value / order.cumulative_quantity if order.cumulative_quantity > 0 else 0.0
            
            if abs(order.cumulative_quantity - order.quantity) < 1e-8:
                order.status = OrderStatus.FILLED
                order.filled_at = fill.timestamp
            elif order.cumulative_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED

            self._record_audit("ORDER_FILL", order)
            logger.info(
                f"Fill recorded for {client_order_id}: {quantity} @ {price} (fee={fee})"
            )

            return order

    async def cancel_order(
        self,
        client_order_id: str,
        reason: str = "",
    ) -> Tuple[bool, Optional[Order], str]:
        """Cancel an open order"""
        async with self._lock:
            order = self.client_order_map.get(client_order_id)
            if not order:
                return False, None, "order_not_found"

            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                return False, order, f"order_not_active ({order.status.value})"

            order.status = OrderStatus.CANCELLED
            self._record_audit("ORDER_CANCELLED", order)

            logger.info(f"Order {client_order_id} cancelled: {reason}")

            return True, order, "cancelled"

    def _record_audit(self, event_type: str, order: Order) -> None:
        """Record audit trail entry"""
        entry = {
            "timestamp": int(time.time() * 1000),
            "event_type": event_type,
            "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "cumulative_quantity": order.cumulative_quantity,
            "average_fill_price": order.average_fill_price,
        }

        self.audit_log.append(entry)

        # Enforce size limit
        if len(self.audit_log) > self.max_audit_log_size:
            self.audit_log = self.audit_log[-self.max_audit_log_size:]
        # Append single entry instead of rewriting entire file on every op
        self._append_audit_entry(entry)
    def save_state(self):
        """Persist audit log and order state to disk."""
        self._save_audit_log()
        self._save_order_state()

    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client ID"""
        return self.client_order_map.get(client_order_id)

    def get_orders_by_symbol(self, exchange: str, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [
            o for o in self.orders.values()
            if o.venue == exchange and o.symbol == symbol
        ]

    def get_open_orders(self, exchange: Optional[str] = None) -> List[Order]:
        """Get all open/active orders"""
        orders = [
            o for o in self.orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]
        if exchange:
            orders = [o for o in orders if o.venue == exchange]
        return orders

    def get_filled_orders(self, exchange: Optional[str] = None) -> List[Order]:
        """Get all filled orders"""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.FILLED]
        if exchange:
            orders = [o for o in orders if o.venue == exchange]
        return orders

    def get_stats(self) -> Dict:
        """Get order manager statistics"""
        open_orders = self.get_open_orders()
        filled_orders = self.get_filled_orders()

        total_value_filled = sum(
            o.cumulative_quantity * o.average_fill_price
            for o in filled_orders
        )

        return {
            "total_orders": len(self.orders),
            "open_orders": len(open_orders),
            "filled_orders": len(filled_orders),
            "total_fill_value": total_value_filled,
            "total_fees": sum(o.total_fee for o in filled_orders),
            "idempotency_records": self.idempotency.get_stats()["total_records"],
            "audit_log_size": len(self.audit_log),
        }

    _TERMINAL_STATUSES = frozenset({
        OrderStatus.FILLED, OrderStatus.CANCELLED,
        OrderStatus.REJECTED, OrderStatus.EXPIRED,
    })
    _ORDER_TTL_SEC = 3600  # keep terminal-state orders for 1 hour

    async def run(self) -> None:
        """Start order manager with periodic cleanup and V6.0 fill monitoring."""
        self._running = True
        logger.info("OrderManager started")
        while self._running:
            await asyncio.sleep(5)  # Check every 5s for time-sensitive monitors
            await self._monitor_partial_fills()
            await self._monitor_limit_expiry()
            # Full cleanup less frequently
            if int(time.time()) % 60 < 6:
                await self._cleanup_old_orders()

    # ── V6.0: Partial fill timeout (cancel if <70% filled after 30s) ─────
    async def _monitor_partial_fills(self) -> None:
        """Cancel orders that are <70% filled after 30 seconds."""
        now_ms = int(time.time() * 1000)
        threshold_ms = 30_000  # 30 seconds
        fill_threshold = 0.70

        orders_to_cancel = []
        async with self._lock:
            for oid, order in self.orders.items():
                if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.SUBMITTED):
                    continue
                age_ms = now_ms - order.created_at
                if age_ms < threshold_ms:
                    continue
                fill_ratio = order.cumulative_quantity / order.quantity if order.quantity > 0 else 0
                if fill_ratio < fill_threshold:
                    orders_to_cancel.append(oid)

        for oid in orders_to_cancel:
            try:
                await self.cancel_order(oid)
                logger.info("V6.0 partial fill timeout: cancelled order {} (<70% filled after 30s)", oid)
            except Exception as exc:
                logger.debug("Failed to cancel order {}: {}", oid, exc)

    # ── V6.0: Limit order expiry (cancel after 120s if unfilled) ─────────
    async def _monitor_limit_expiry(self) -> None:
        """Expire limit orders that haven't filled within 120 seconds."""
        now_ms = int(time.time() * 1000)
        expiry_ms = 120_000  # 120 seconds

        orders_to_expire = []
        async with self._lock:
            for oid, order in self.orders.items():
                if order.order_type != OrderType.LIMIT:
                    continue
                if order.status not in (OrderStatus.OPEN, OrderStatus.SUBMITTED):
                    continue
                age_ms = now_ms - order.created_at
                if age_ms >= expiry_ms and order.cumulative_quantity == 0:
                    orders_to_expire.append(oid)

        for oid in orders_to_expire:
            try:
                await self.cancel_order(oid)
                order = self.orders.get(oid)
                if order:
                    order.status = OrderStatus.EXPIRED
                logger.info("V6.0 limit order expired: {} (unfilled after 120s)", oid)
            except Exception as exc:
                logger.debug("Failed to expire order {}: {}", oid, exc)

    async def _cleanup_old_orders(self) -> None:
        """Remove orders in terminal states older than _ORDER_TTL_SEC."""
        now = time.time() * 1000  # updated_at is milliseconds
        cutoff = now - self._ORDER_TTL_SEC * 1000
        stale_ids = [
            oid for oid, order in self.orders.items()
            if order.status in self._TERMINAL_STATUSES
            and order.updated_at < cutoff
        ]
        if not stale_ids:
            return
        async with self._lock:
            for oid in stale_ids:
                order = self.orders.pop(oid, None)
                if order is None:
                    continue
                self.client_order_map.pop(order.client_order_id, None)
                if order.exchange_order_id:
                    self.exchange_order_map.pop(order.exchange_order_id, None)
                self.pending_orders.pop(oid, None)
                if order.user_id and order.user_id in self.user_orders:
                    self.user_orders[order.user_id].pop(oid, None)
            logger.info("Cleaned up {} terminal-state orders", len(stale_ids))

    async def stop(self) -> None:
        """Stop order manager and persist state"""
        self._running = False
        self.save_state()
        logger.info("OrderManager stopped and state saved")
