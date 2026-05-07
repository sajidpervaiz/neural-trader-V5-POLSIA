"""
Orders API routes for FastAPI dashboard.
"""

import time
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger

from execution.order_manager import (
    OrderManager,
    OrderSide as OMSide,
    OrderType as OMType,
)

router = APIRouter(prefix="/orders", tags=["orders"])
_ORDER_MANAGER: Optional[OrderManager] = None


def configure_order_routes(order_manager: Optional[OrderManager]) -> None:
    global _ORDER_MANAGER
    _ORDER_MANAGER = order_manager


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    time_in_force: TimeInForce = TimeInForce.GTC
    venue: str = Field(..., description="Venue (binance, bybit, okx, etc.)")
    reduce_only: bool = False
    client_order_id: Optional[str] = None


class OrderResponse(BaseModel):
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    status: str
    venue: str
    created_at: int
    updated_at: int


def _require_order_manager() -> OrderManager:
    if _ORDER_MANAGER is None:
        raise HTTPException(status_code=503, detail="order_manager_unavailable")
    return _ORDER_MANAGER


def _map_order_type(order_type: OrderType) -> OMType:
    if order_type == OrderType.MARKET:
        return OMType.MARKET
    if order_type == OrderType.LIMIT:
        return OMType.LIMIT
    # Fallback for unsupported STOP type in current order manager.
    return OMType.LIMIT


def _to_response(order: Any) -> OrderResponse:
    side = OrderSide.BUY if str(order.side.value).lower() == "buy" else OrderSide.SELL
    order_type_map = {
        "market": OrderType.MARKET,
        "limit": OrderType.LIMIT,
        "post_only": OrderType.LIMIT,
        "ioc": OrderType.LIMIT,
    }
    order_type = order_type_map.get(str(order.order_type.value).lower(), OrderType.LIMIT)
    return OrderResponse(
        order_id=order.order_id,
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=side,
        order_type=order_type,
        quantity=float(order.quantity),
        price=float(order.price) if order.price is not None else None,
        filled_quantity=float(order.filled_quantity),
        remaining_quantity=float(order.remaining_quantity),
        avg_fill_price=float(order.avg_fill_price),
        status=str(order.status.value).upper(),
        venue=str(order.venue),
        created_at=int(order.created_at),
        updated_at=int(order.updated_at),
    )


@router.post("/", response_model=OrderResponse)
async def create_order(
    request: OrderRequest,
    idempotency_key: Optional[str] = Query(None),
):
    """
    Create a new order.
    """
    try:
        manager = _require_order_manager()
        side = OMSide.BUY if request.side == OrderSide.BUY else OMSide.SELL
        om_type = _map_order_type(request.order_type)
        price = float(request.price or 0.0)
        success, order, reason = await manager.place_order(
            exchange=request.venue,
            symbol=request.symbol,
            side=side,
            quantity=float(request.quantity),
            price=price,
            order_type=om_type,
            client_order_id=request.client_order_id,
            metadata={
                "time_in_force": request.time_in_force.value,
                "reduce_only": request.reduce_only,
                "client_order_id": request.client_order_id,
                "idempotency_key": idempotency_key,
                "api_created_at": int(time.time() * 1000),
            },
        )
        if not success or order is None:
            raise HTTPException(status_code=400, detail=reason)
        return _to_response(order)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place", response_model=OrderResponse)
async def place_order_compat(request: OrderRequest):
    """
    Backward-compatible alias for older deployment scripts/docs.
    """
    return await create_order(request)


@router.get("/", response_model=List[OrderResponse])
async def get_orders(
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
):
    """
    Get orders with optional filtering.
    """
    try:
        manager = _require_order_manager()
        orders = list(manager.orders.values())
        if venue:
            orders = [o for o in orders if str(o.venue) == venue]
        if symbol:
            orders = [o for o in orders if str(o.symbol) == symbol]
        if status:
            normalized = status.lower()
            orders = [o for o in orders if str(o.status.value).lower() == normalized]
        orders = sorted(orders, key=lambda o: o.created_at, reverse=True)[:limit]
        return [_to_response(o) for o in orders]

    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def create_batch_orders(
    orders: List[OrderRequest] = Body(...),
):
    """
    Create multiple orders in a batch.
    """
    try:
        results = []
        for order in orders:
            result = await create_order(order)
            results.append(result)

        return {
            "total": len(orders),
            "successful": len(results),
            "orders": results,
        }

    except Exception as e:
        logger.error(f"Error creating batch orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/open", response_model=List[OrderResponse])
async def get_open_orders(
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """
    Get all open orders.
    """
    try:
        manager = _require_order_manager()
        open_orders = manager.get_open_orders(exchange=venue)
        if symbol:
            open_orders = [o for o in open_orders if str(o.symbol) == symbol]
        return [_to_response(o) for o in open_orders]

    except Exception as e:
        logger.error(f"Error fetching open orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/open")
async def cancel_all_open_orders(
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """
    Cancel all open orders, optionally filtered.
    """
    try:
        manager = _require_order_manager()
        open_orders = manager.get_open_orders(exchange=venue)
        if symbol:
            open_orders = [o for o in open_orders if str(o.symbol) == symbol]
        cancelled = 0
        for order in open_orders:
            if not order.client_order_id:
                continue
            success, _, _ = await manager.cancel_order(order.client_order_id, reason="api_cancel_all")
            if success:
                cancelled += 1
        return {
            "status": "cancelled",
            "cancelled_count": cancelled,
            "venue_filter": venue,
            "symbol_filter": symbol,
        }

    except Exception as e:
        logger.error(f"Error cancelling open orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════
#  ARMS-V2.1: Order splitting & shadow SL status
#  (Must be above /{order_id} catch-all)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/twap-status")
async def twap_status():
    """Active TWAP split orders."""
    try:
        manager = _require_order_manager()
        return manager.get_twap_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/iceberg-status")
async def iceberg_status():
    """Active Iceberg split orders."""
    try:
        manager = _require_order_manager()
        return manager.get_iceberg_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shadow-sl")
async def shadow_sl_status():
    """Shadow stop-loss 4-layer redundancy status."""
    try:
        manager = _require_order_manager()
        return manager.get_shadow_sl_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """
    Get order by ID.
    """
    try:
        manager = _require_order_manager()
        order = next((o for o in manager.orders.values() if str(o.order_id) == order_id), None)
        if order is None:
            raise HTTPException(status_code=404, detail="Order not found")
        return _to_response(order)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{order_id}")
async def cancel_order(order_id: str, venue: str = Query(...)):
    """
    Cancel an order.
    """
    try:
        manager = _require_order_manager()
        order = next(
            (
                o for o in manager.orders.values()
                if str(o.order_id) == order_id and str(o.venue) == venue
            ),
            None,
        )
        if order is None or not order.client_order_id:
            raise HTTPException(status_code=404, detail="Order not found")
        success, _, reason = await manager.cancel_order(order.client_order_id, reason="api_cancel")
        if not success:
            raise HTTPException(status_code=400, detail=reason)
        return {
            "order_id": order_id,
            "venue": venue,
            "status": "cancelled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
