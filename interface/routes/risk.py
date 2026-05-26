"""
Risk API routes for FastAPI dashboard.
"""

import time
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger

from core.error_handling import sanitize_exception

from engine.signal_generator import TradingSignal
from execution.risk_manager import RiskManager

router = APIRouter(prefix="/risk", tags=["risk"])
_RISK_MANAGER: Optional[RiskManager] = None


def configure_risk_routes(risk_manager: Optional[RiskManager]) -> None:
    global _RISK_MANAGER
    _RISK_MANAGER = risk_manager


class RiskLimitType(str, Enum):
    POSITION_VALUE = "position_value"
    ORDER_SIZE = "order_size"
    ORDERS_PER_SEC = "orders_per_sec"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"


class RiskLimits(BaseModel):
    max_position_value: float = Field(default=1_000_000.0, gt=0)
    max_order_size: float = Field(default=10_000.0, gt=0)
    max_orders_per_sec: int = Field(default=100, gt=0)
    max_concentration: float = Field(default=0.3, ge=0, le=1.0)
    leverage_limit: float = Field(default=10.0, gt=0)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1.0)


class MarginInfo(BaseModel):
    account_balance: float
    margin_available: float
    margin_used: float
    total_exposure: float
    leverage_used: float
    liquidation_price: Optional[float]
    health_score: float


class RiskCheck(BaseModel):
    passed: bool
    reason: str
    margin_required: Optional[float]
    exposure_after: Optional[float]


def _require_risk_manager() -> RiskManager:
    if _RISK_MANAGER is None:
        raise HTTPException(status_code=503, detail="risk_manager_unavailable")
    return _RISK_MANAGER


@router.get("/trading-state")
async def get_trading_state():
    """Return the effective trading state gate."""
    try:
        manager = _require_risk_manager()
        return manager.get_trading_state()
    except Exception as e:
        logger.error(f"Error fetching trading state: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/trading-state")
async def set_trading_state(payload: dict = Body(...)):
    """Set ACTIVE, REDUCING, or HALTED for operator-controlled risk gating."""
    try:
        manager = _require_risk_manager()
        state = str(payload.get("state", "") or "")
        reason = str(payload.get("reason", "operator_request") or "operator_request")
        operator = str(payload.get("operator", "dashboard") or "dashboard")
        return manager.set_trading_state(state, reason=reason, operator=operator)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_exception(e))
    except Exception as e:
        logger.error(f"Error setting trading state: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/limits", response_model=RiskLimits)
async def get_risk_limits():
    """Get current risk limits."""
    try:
        manager = _require_risk_manager()
        snap = manager.get_risk_snapshot()
        max_position_value = snap["equity"] * manager._max_position_pct * manager._leverage
        return RiskLimits(
            max_position_value=max(1.0, float(max_position_value)),
            max_order_size=max(1.0, float(max_position_value)),
            max_orders_per_sec=100,
            max_concentration=0.3,
            leverage_limit=float(manager._leverage),
            stop_loss_pct=float(manager.config.get_value("risk", "stop_loss_pct", default=0.015)),
        )

    except Exception as e:
        logger.error(f"Error fetching risk limits: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/limits", response_model=dict)
async def set_risk_limits(
    limits: RiskLimits = Body(...),
    user_id: Optional[str] = Query(None),
):
    """Update risk limits."""
    try:
        manager = _require_risk_manager()
        manager._leverage = float(max(0.1, limits.leverage_limit))
        # Convert max position value back into a position percentage of equity.
        if manager.equity > 0:
            manager._max_position_pct = float(min(1.0, max(0.0001, limits.max_position_value / manager.equity)))
        return {
            "status": "updated",
            "limits": limits.dict(),
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Error setting risk limits: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/margin", response_model=MarginInfo)
async def get_margin_info(
    user_id: Optional[str] = Query(None),
):
    """Get margin information."""
    try:
        manager = _require_risk_manager()
        snap = manager.get_risk_snapshot()
        margin_used = float(snap["portfolio_notional_usd"]) / max(1.0, manager._leverage)
        margin_available = max(0.0, float(snap["equity"]) - margin_used)
        leverage_used = float(snap["portfolio_notional_usd"]) / max(1.0, float(snap["equity"]))
        health_score = max(0.0, min(1.0, 1.0 - snap["var_95"]))
        return MarginInfo(
            account_balance=float(snap["equity"]),
            margin_available=float(margin_available),
            margin_used=float(margin_used),
            total_exposure=float(snap["portfolio_notional_usd"]),
            leverage_used=float(leverage_used),
            liquidation_price=None,
            health_score=float(health_score),
        )

    except Exception as e:
        logger.error(f"Error fetching margin info: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/check", response_model=RiskCheck)
async def check_risk(
    symbol: str = Query(...),
    side: str = Query(..., pattern="^(buy|sell)$"),
    quantity: float = Query(..., gt=0),
    price: float = Query(..., gt=0),
    user_id: Optional[str] = Query(None),
):
    """
    Check if order passes risk controls.
    """
    try:
        manager = _require_risk_manager()
        side = side.lower()
        sl_pct = float(manager.config.get_value("risk", "stop_loss_pct", default=0.015))
        tp_pct = float(manager.config.get_value("risk", "take_profit_pct", default=0.03))
        is_buy = side == "buy"
        stop_loss = price * (1 - sl_pct) if is_buy else price * (1 + sl_pct)
        take_profit = price * (1 + tp_pct) if is_buy else price * (1 - tp_pct)

        signal = TradingSignal(
            exchange="api",
            symbol=symbol,
            direction="long" if is_buy else "short",
            score=0.8,
            technical_score=0.8,
            ml_score=0.8,
            sentiment_score=0.0,
            macro_score=0.0,
            news_score=0.0,
            orderbook_score=0.0,
            regime="api",
            regime_confidence=1.0,
            price=price,
            atr=price * 0.01,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=int(time.time()),
            metadata={"user_id": user_id, "quantity": quantity},
        )

        passed, reason, approved_notional = manager.approve_signal(signal)
        margin_required = approved_notional / max(1.0, manager._leverage)
        exposure_after = manager.portfolio_notional() + approved_notional

        return RiskCheck(
            passed=bool(passed),
            reason=reason,
            margin_required=float(margin_required),
            exposure_after=float(exposure_after),
        )

    except Exception as e:
        logger.error(f"Error checking risk: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/exposure")
async def get_exposure_breakdown(
    user_id: Optional[str] = Query(None),
):
    """Get exposure breakdown by symbol and venue."""
    try:
        manager = _require_risk_manager()
        by_symbol: dict[str, float] = {}
        by_venue: dict[str, float] = {}
        by_side: dict[str, float] = {"long": 0.0, "short": 0.0}

        for pos in manager.positions.values():
            exposure = abs(float(pos.size) * float(pos.current_price))
            by_symbol[pos.symbol] = by_symbol.get(pos.symbol, 0.0) + exposure
            by_venue[pos.exchange] = by_venue.get(pos.exchange, 0.0) + exposure
            by_side[pos.direction] = by_side.get(pos.direction, 0.0) + exposure

        total_exposure = float(sum(by_symbol.values()))
        return {
            "total_exposure": total_exposure,
            "by_symbol": by_symbol,
            "by_venue": by_venue,
            "by_side": by_side,
        }

    except Exception as e:
        logger.error(f"Error fetching exposure breakdown: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/circuit-breaker")
async def get_circuit_breaker_status():
    """Get circuit breaker status across venues."""
    try:
        manager = _require_risk_manager()
        cb = manager._circuit_breaker
        state = "OPEN" if cb.tripped else "CLOSED"
        return {
            "venues": {
                "global_risk": {
                    "state": state,
                    "failure_count": 0,
                    "last_failure_time": None,
                    "reason": cb.trip_reason,
                }
            },
            "overall_status": "HEALTHY" if state == "CLOSED" else "TRIPPED",
        }

    except Exception as e:
        logger.error(f"Error fetching circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(venue: str = Query(...)):
    """Reset circuit breaker for a venue."""
    try:
        manager = _require_risk_manager()
        manager._circuit_breaker.reset()
        return {
            "venue": venue,
            "status": "reset",
        }

    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/stress-test")
async def run_stress_test(
    scenario: str = Query(..., description="Stress test scenario"),
):
    """
    Run a stress test against risk controls.

    Scenarios: flash_crash, liquidity_crisis, correlation_breakdown, extreme_volatility
    """
    try:
        manager = _require_risk_manager()
        scenario_map = {
            "flash_crash": [-0.20],
            "liquidity_crisis": [-0.10],
            "correlation_breakdown": [-0.15],
            "extreme_volatility": [-0.05, -0.10, -0.20],
        }
        shocks = scenario_map.get(scenario, [-0.10])
        report = manager.run_stress_test(shocks=shocks, correlation_breakdown_factor=1.5)
        return {
            "scenario": scenario,
            "status": "completed",
            "report": report,
        }

    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


# ══════════════════════════════════════════════════════════════════════════
#  ARMS-V2.1 endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.get("/arms/snapshot")
async def arms_snapshot():
    """Full ARMS-V2.1 risk snapshot — tier risk, drawdown, blackout, leverage, etc."""
    try:
        manager = _require_risk_manager()
        base = manager.get_risk_snapshot()

        # Tier risk percentages
        tier_risk = {}
        for tier in range(1, 4):
            tier_risk[tier] = manager.get_tier_risk_pct(tier)

        # Drawdown phase details
        dd_phase = manager.update_drawdown_phase()
        dd_mult = manager.get_drawdown_size_multiplier()
        dd_max_pos = manager.get_drawdown_max_positions()

        # Event blackout
        blackout_active, blackout_reason = manager.is_event_blackout()

        # Weekend
        weekend_mult = manager.get_weekend_sizing_mult()

        # Weekly/monthly limits
        wm_ok, wm_reason = manager.check_weekly_monthly_limits()

        # Group exposure summary
        groups: dict[str, dict] = {}
        for group_name, symbols in getattr(manager, "_correlation_groups", {}).items():
            notional = 0.0
            for pos in manager.positions.values():
                if pos.symbol in symbols:
                    notional += abs(pos.size * pos.current_price)
            max_notional = manager.equity * getattr(manager, "_max_group_exposure_pct", 1.0)
            groups[group_name] = {
                "symbols": list(symbols),
                "notional": notional,
                "max_notional": max_notional,
                "utilization_pct": (notional / max_notional * 100) if max_notional > 0 else 0,
            }

        return {
            **base,
            "tier_risk": tier_risk,
            "drawdown_phase": dd_phase,
            "drawdown_size_multiplier": dd_mult,
            "drawdown_max_positions": dd_max_pos,
            "blackout_active": blackout_active,
            "blackout_reason": blackout_reason,
            "weekend_sizing_mult": weekend_mult,
            "weekly_monthly_ok": wm_ok,
            "weekly_monthly_reason": wm_reason,
            "correlation_groups": groups,
        }

    except Exception as e:
        logger.error(f"ARMS snapshot error: {e}")
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/arms/tier-risk")
async def get_tier_risk():
    """Per-tier risk-per-trade percentages."""
    try:
        manager = _require_risk_manager()
        return {
            "tiers": {tier: manager.get_tier_risk_pct(tier) for tier in range(1, 4)},
            "default_risk_per_trade": float(getattr(manager, "_risk_per_trade", 0.01)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/arms/group-exposure")
async def get_group_exposure():
    """Correlation group exposure breakdown."""
    try:
        manager = _require_risk_manager()
        groups: dict[str, dict] = {}
        for group_name, symbols in getattr(manager, "_correlation_groups", {}).items():
            notional = 0.0
            for pos in manager.positions.values():
                if pos.symbol in symbols:
                    notional += abs(pos.size * pos.current_price)
            max_notional = manager.equity * getattr(manager, "_max_group_exposure_pct", 1.0)
            groups[group_name] = {
                "symbols": list(symbols),
                "notional": notional,
                "max_notional": max_notional,
                "utilization_pct": (notional / max_notional * 100) if max_notional > 0 else 0,
            }
        return {"groups": groups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/arms/blackout")
async def get_blackout_status():
    """Event blackout status."""
    try:
        manager = _require_risk_manager()
        active, reason = manager.is_event_blackout()
        events = getattr(manager, "_upcoming_events", [])
        return {
            "blackout_active": active,
            "reason": reason,
            "upcoming_events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/arms/dynamic-leverage")
async def get_dynamic_leverage(
    adx: float = Query(25.0, description="ADX value"),
    atr_percentile: float = Query(50.0, description="ATR percentile"),
):
    """ADX/ATR-linked dynamic leverage calculation."""
    try:
        manager = _require_risk_manager()
        lev = manager.get_dynamic_leverage(adx, atr_percentile)
        return {
            "adx": adx,
            "atr_percentile": atr_percentile,
            "dynamic_leverage": lev,
            "max_leverage": float(getattr(manager, "_leverage", 1.0)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.get("/arms/margin-modes")
async def get_margin_modes():
    """Margin mode for each open position."""
    try:
        manager = _require_risk_manager()
        modes = {}
        for key, pos in manager.positions.items():
            modes[key] = {
                "symbol": pos.symbol,
                "margin_mode": manager.get_margin_mode(pos.symbol),
            }
        return {"positions": modes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/arms/liq-check")
async def trigger_liq_check():
    """Manually trigger liquidation distance check."""
    try:
        manager = _require_risk_manager()
        actions = manager.run_periodic_liq_check()
        return {"actions": actions, "checked_positions": len(manager.positions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))


@router.post("/arms/funding-recheck")
async def trigger_funding_recheck():
    """Manually trigger funding recheck for existing positions."""
    try:
        manager = _require_risk_manager()
        actions = manager.check_funding_existing_positions()
        return {"actions": actions, "checked_positions": len(manager.positions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_exception(e))
