"""
Positions API routes for FastAPI dashboard.
"""

import time
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger

from core.error_handling import sanitize_exception

from execution.risk_manager import RiskManager, Position

router = APIRouter(prefix="/positions", tags=["positions"])
_RISK_MANAGER: Optional[RiskManager] = None
_CONFIG = None
_DB_HANDLER = None


def configure_positions_routes(risk_manager: Optional[RiskManager], *, config=None, db_handler=None) -> None:
    global _RISK_MANAGER, _CONFIG, _DB_HANDLER
    _RISK_MANAGER = risk_manager
    _CONFIG = config
    _DB_HANDLER = db_handler


def _paper_mode() -> bool:
    return bool(getattr(_CONFIG, "paper_mode", True))


def _risk_kill_switch_active() -> bool:
    if _RISK_MANAGER is None:
        return True
    for attr in ("kill_switch_active", "killed"):
        value = getattr(_RISK_MANAGER, attr, None)
        if isinstance(value, bool):
            return value
    try:
        snap = _RISK_MANAGER.get_risk_snapshot()
        if isinstance(snap, dict):
            return bool(snap.get("kill_switch_active", False))
    except Exception:
        return True
    return False


def _require_legacy_position_mutation_allowed() -> None:
    if _paper_mode():
        return
    if not bool(getattr(_DB_HANDLER, "available", False)):
        raise HTTPException(status_code=503, detail="live_position_routes_require_audit_db")
    if _risk_kill_switch_active():
        raise HTTPException(status_code=423, detail="live_position_routes_blocked_by_kill_switch")
    raise HTTPException(status_code=403, detail="legacy position mutation routes disabled in live mode; use exchange-close API with typed confirmation")


class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    venue: str
    entry_time: int


class PositionSummary(BaseModel):
    total_positions: int
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_margin_used: float
    open_symbols: List[str]
    by_venue: dict


def _require_risk_manager() -> RiskManager:
    if _RISK_MANAGER is None:
        raise HTTPException(status_code=503, detail="risk_manager_unavailable")
    return _RISK_MANAGER


def _to_response(pos: Position, leverage: float) -> PositionResponse:
    return PositionResponse(
        symbol=str(pos.symbol),
        side=str(pos.direction),
        quantity=float(pos.size),
        avg_price=float(pos.entry_price),
        unrealized_pnl=float(pos.pnl),
        realized_pnl=0.0,
        leverage=int(max(1, round(leverage))),
        venue=str(pos.exchange),
        entry_time=int(pos.open_time),
    )


@router.get("/", response_model=List[PositionResponse])
async def get_positions(
    user_id: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """
    Get all open positions with optional filtering.
    """
    try:
        manager = _require_risk_manager()
        positions = list(manager.positions.values())
        if venue:
            positions = [p for p in positions if str(p.exchange) == venue]
        if symbol:
            positions = [p for p in positions if str(p.symbol) == symbol]
        return [_to_response(p, manager._leverage) for p in positions]

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/summary", response_model=PositionSummary)
async def get_position_summary(
    user_id: Optional[str] = Query(None),
):
    """
    Get position summary across all venues.
    """
    try:
        manager = _require_risk_manager()
        positions = list(manager.positions.values())
        by_venue: dict[str, float] = {}
        for pos in positions:
            exposure = abs(float(pos.size) * float(pos.current_price))
            by_venue[pos.exchange] = by_venue.get(pos.exchange, 0.0) + exposure

        return PositionSummary(
            total_positions=len(positions),
            total_unrealized_pnl=float(sum(float(p.pnl) for p in positions)),
            total_realized_pnl=0.0,
            total_margin_used=float(manager.portfolio_notional() / max(1.0, manager._leverage)),
            open_symbols=sorted({str(p.symbol) for p in positions}),
            by_venue=by_venue,
        )

    except Exception as e:
        logger.error(f"Error fetching position summary: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/{symbol}", response_model=PositionResponse)
async def get_position(
    symbol: str,
    venue: str = Query(..., description="Venue (binance, bybit, okx, etc.)"),
):
    """
    Get position for specific symbol on a venue.
    """
    try:
        manager = _require_risk_manager()
        key = f"{venue}:{symbol}"
        pos = manager.positions.get(key)
        if pos is None:
            raise HTTPException(status_code=404, detail="Position not found")
        return _to_response(pos, manager._leverage)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error fetching position {symbol} on {venue}: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/{symbol}/close")
async def close_position(
    symbol: str,
    venue: str = Query(...),
    quantity: Optional[float] = Query(None, description="Quantity to close (default: all)"),
    price: Optional[float] = Query(None, gt=0, description="Exit price override"),
):
    """
    Close position for specific symbol.
    """
    try:
        _require_legacy_position_mutation_allowed()
        manager = _require_risk_manager()
        key = f"{venue}:{symbol}"
        pos = manager.positions.get(key)
        if pos is None:
            raise HTTPException(status_code=404, detail="Position not found")

        # Current implementation is full close. Quantity is accepted for API compatibility.
        close_price = float(price if price is not None else pos.current_price)
        closed = await manager.close_position(venue, symbol, close_price)
        if closed is None:
            raise HTTPException(status_code=404, detail="Position not found")

        return {
            "symbol": symbol,
            "venue": venue,
            "status": "closed",
            "quantity": quantity,
            "closed_quantity": float(closed.size),
            "exit_price": close_price,
            "realized_pnl": float(closed.pnl),
            "closed_at": int(time.time()),
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/close-all")
async def close_all_positions(
    venue: Optional[str] = Query(None),
):
    """
    Close all positions, optionally filtered by venue.
    """
    try:
        _require_legacy_position_mutation_allowed()
        manager = _require_risk_manager()
        positions = list(manager.positions.values())
        if venue:
            positions = [p for p in positions if str(p.exchange) == venue]

        closed_count = 0
        realized_total = 0.0
        for pos in positions:
            closed = await manager.close_position(pos.exchange, pos.symbol, pos.current_price)
            if closed is not None:
                closed_count += 1
                realized_total += float(closed.pnl)

        return {
            "status": "closed",
            "venue_filter": venue,
            "closed_count": closed_count,
            "realized_pnl_total": realized_total,
            "closed_at": int(time.time()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))
