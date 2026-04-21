"""Configuration API routes for FastAPI dashboard."""

import time
from fastapi import APIRouter, HTTPException, Body, Query
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger

from core.config import Config
from execution.order_manager import OrderManager
from execution.risk_manager import RiskManager

router = APIRouter(prefix="/config", tags=["config"])

_CONFIG: Optional[Config] = None
_RISK_MANAGER: Optional[RiskManager] = None
_ORDER_MANAGER: Optional[OrderManager] = None
_EXECUTORS: list = []


def configure_config_routes(
    config: Optional[Config],
    risk_manager: Optional[RiskManager] = None,
    order_manager: Optional[OrderManager] = None,
    executors: Optional[list] = None,
) -> None:
    global _CONFIG, _RISK_MANAGER, _ORDER_MANAGER, _EXECUTORS, _TRADING_MODE
    _CONFIG = config
    _RISK_MANAGER = risk_manager
    _ORDER_MANAGER = order_manager
    _EXECUTORS = list(executors or [])
    if _CONFIG is not None:
        data = getattr(_CONFIG, "_data", None)
        persisted = None
        if isinstance(data, dict):
            persisted = (data.get("system") or {}).get("trading_mode")
        if persisted:
            try:
                _TRADING_MODE = TradingMode(persisted)
            except Exception:
                _TRADING_MODE = TradingMode.PAPER if bool(getattr(_CONFIG, "paper_mode", True)) else TradingMode.LIVE
        else:
            _TRADING_MODE = TradingMode.PAPER if bool(getattr(_CONFIG, "paper_mode", True)) else TradingMode.LIVE


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    DEMO = "demo"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and value.lower() == "simulation":
            return cls.DEMO
        return None


class AlgoConfig(BaseModel):
    enabled: bool
    max_position_size: float
    max_risk_per_trade: float
    max_daily_loss: float
    max_positions: int


class VenueConfig(BaseModel):
    name: str
    enabled: bool
    api_keys_configured: bool
    connection_status: str
    latency_ms: float


# Runtime config state for API-level updates without mutating on-disk YAML.
_TRADING_MODE: TradingMode = TradingMode.PAPER
_ALGO_CONFIG = AlgoConfig(
    enabled=True,
    max_position_size=10.0,
    max_risk_per_trade=0.02,
    max_daily_loss=5000.0,
    max_positions=5,
)
_VENUE_OVERRIDES: dict[str, bool] = {}
_CONFIG_HISTORY: list[dict[str, Any]] = []


def _record_change(change_type: str, payload: dict[str, Any]) -> None:
    _CONFIG_HISTORY.append(
        {
            "timestamp": int(time.time()),
            "change_type": change_type,
            "payload": payload,
        }
    )
    if len(_CONFIG_HISTORY) > 1000:
        del _CONFIG_HISTORY[:-1000]


def _effective_venues() -> dict[str, dict[str, Any]]:
    cfg = _CONFIG.get_value("exchanges", default={}) if _CONFIG else {}
    out: dict[str, dict[str, Any]] = {}
    for venue, venue_cfg in (cfg or {}).items():
        if not isinstance(venue_cfg, dict):
            continue
        enabled_cfg = bool(venue_cfg.get("enabled", False))
        enabled = _VENUE_OVERRIDES.get(venue, enabled_cfg)
        out[str(venue)] = {
            "enabled": enabled,
            "api_key_configured": bool(venue_cfg.get("api_key")),
            "api_secret_configured": bool(venue_cfg.get("api_secret")),
            "testnet": bool(venue_cfg.get("testnet", False)),
        }
    return out


def _runtime_mode() -> TradingMode:
    global _TRADING_MODE
    # Preserve explicit DEMO selection — paper_mode alone cannot distinguish PAPER from DEMO.
    if _TRADING_MODE == TradingMode.DEMO:
        return _TRADING_MODE
    if _CONFIG is not None:
        _TRADING_MODE = TradingMode.PAPER if bool(getattr(_CONFIG, "paper_mode", True)) else TradingMode.LIVE
    return _TRADING_MODE


def get_active_trading_mode() -> str:
    """Public getter for other modules that need the user-selected trading mode."""
    return _runtime_mode().value


def _validate_demo_switch() -> Optional[str]:
    venues = _effective_venues()
    enabled = [name for name, data in venues.items() if data.get("enabled")]
    if not enabled:
        return "Enable a venue (e.g. Binance) before switching to demo mode."
    for venue in enabled:
        data = venues.get(venue, {})
        if not data.get("api_key_configured") or not data.get("api_secret_configured"):
            return f"Add the {venue} demo API key and secret first."
    return None


async def _refresh_executor_clients() -> None:
    """Close+reinitialise every executor's ccxt client so testnet/key changes apply."""
    for executor in _EXECUTORS:
        client = getattr(executor, "_client", None)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
        executor._client = None
        executor._order_placer = None
        try:
            await executor._init_client()
        except Exception as exc:
            logger.warning(
                "executor {} re-init failed after mode switch: {}",
                getattr(executor, "exchange_id", "?"), exc,
            )


def _apply_testnet_to_enabled_venues() -> None:
    """Flip the testnet flag to true on every enabled venue so DEMO hits sandbox URLs."""
    if _CONFIG is None:
        return
    exchanges = _CONFIG._data.setdefault("exchanges", {})
    if not isinstance(exchanges, dict):
        return
    for venue, venue_cfg in exchanges.items():
        if not isinstance(venue_cfg, dict):
            continue
        enabled = _VENUE_OVERRIDES.get(venue, bool(venue_cfg.get("enabled", False)))
        if enabled:
            venue_cfg["testnet"] = True


def _validate_live_switch() -> Optional[str]:
    venues = _effective_venues()
    enabled = [name for name, data in venues.items() if data.get("enabled")]
    if not enabled:
        return "Enable a demo exchange before switching to live mode."

    for venue in enabled:
        data = venues.get(venue, {})
        if not data.get("api_key_configured") or not data.get("api_secret_configured"):
            return f"Add the {venue} demo API key and secret first."
        if not data.get("testnet", False):
            return "Runtime switching is only supported for demo or testnet venues."

    return None


@router.get("/trading-mode")
async def get_trading_mode():
    """Get current trading mode."""
    try:
        mode = _runtime_mode()
        return {
            "mode": mode.value,
            "reason": "Runtime mode state",
            "paper_mode": bool(_CONFIG.paper_mode) if _CONFIG else True,
        }

    except Exception as e:
        logger.error(f"Error fetching trading mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading-mode")
async def set_trading_mode(
    mode: TradingMode = Body(...),
    confirmation: bool = Query(..., description="Safety confirmation"),
):
    """Set trading mode (requires confirmation for LIVE)."""
    try:
        global _TRADING_MODE
        if mode == TradingMode.LIVE and not confirmation:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required for live trading"
            )

        if mode == TradingMode.LIVE:
            validation_error = _validate_live_switch()
            if validation_error:
                raise HTTPException(status_code=400, detail=validation_error)
            if _CONFIG is not None:
                _CONFIG.paper_mode = False
        elif mode == TradingMode.DEMO:
            validation_error = _validate_demo_switch()
            if validation_error:
                raise HTTPException(status_code=400, detail=validation_error)
            if _CONFIG is not None:
                _CONFIG.paper_mode = True
                _apply_testnet_to_enabled_venues()
        else:
            if _CONFIG is not None:
                _CONFIG.paper_mode = True

        _TRADING_MODE = mode
        if _CONFIG is not None:
            data = getattr(_CONFIG, "_data", None)
            if isinstance(data, dict):
                data.setdefault("system", {})["trading_mode"] = mode.value
            try:
                if hasattr(_CONFIG, "persist_runtime_overrides"):
                    _CONFIG.persist_runtime_overrides()
            except Exception as exc:
                logger.warning(f"Failed to persist trading mode override: {exc}")

        # Refresh executor clients so venue.testnet flag takes effect immediately.
        if mode in (TradingMode.DEMO, TradingMode.LIVE):
            await _refresh_executor_clients()

        _record_change("trading_mode", {"mode": _TRADING_MODE.value, "confirmation": confirmation})

        return {
            "success": True,
            "mode": _TRADING_MODE.value,
            "paper_mode": bool(_CONFIG.paper_mode) if _CONFIG else (_TRADING_MODE != TradingMode.LIVE),
            "status": "updated",
            "message": f"Trading mode set to {_TRADING_MODE.value}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting trading mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algo", response_model=AlgoConfig)
async def get_algo_config():
    """Get algorithm configuration."""
    try:
        if _RISK_MANAGER is None:
            return _ALGO_CONFIG
        snap = _RISK_MANAGER.get_risk_snapshot()
        return AlgoConfig(
            enabled=True,
            max_position_size=float(snap["equity"] * _RISK_MANAGER._max_position_pct * _RISK_MANAGER._leverage),
            max_risk_per_trade=float(_RISK_MANAGER._max_position_pct),
            max_daily_loss=float(getattr(_RISK_MANAGER._circuit_breaker, "_max_daily_loss", 0.03) * snap["equity"]),
            max_positions=int(_RISK_MANAGER._max_open),
        )

    except Exception as e:
        logger.error(f"Error fetching algo config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/algo", response_model=AlgoConfig)
async def set_algo_config(
    config: AlgoConfig = Body(...),
):
    """Update algorithm configuration."""
    try:
        global _ALGO_CONFIG
        _ALGO_CONFIG = config
        if _RISK_MANAGER is not None:
            if _RISK_MANAGER.equity > 0:
                _RISK_MANAGER._max_position_pct = float(
                    min(1.0, max(0.0001, config.max_position_size / max(1.0, _RISK_MANAGER.equity * _RISK_MANAGER._leverage)))
                )
            _RISK_MANAGER._max_open = int(max(1, config.max_positions))
            _RISK_MANAGER._circuit_breaker._max_daily_loss = float(max(0.001, min(1.0, config.max_daily_loss / max(1.0, _RISK_MANAGER.equity))))
        _record_change("algo_config", config.dict())
        return _ALGO_CONFIG

    except Exception as e:
        logger.error(f"Error setting algo config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/venues", response_model=List[VenueConfig])
async def get_venue_configs():
    """Get venue configurations."""
    try:
        venues = _effective_venues()
        open_orders_by_venue: dict[str, int] = {}
        if _ORDER_MANAGER is not None:
            for order in _ORDER_MANAGER.get_open_orders():
                v = str(order.venue or "unknown")
                open_orders_by_venue[v] = open_orders_by_venue.get(v, 0) + 1

        out: list[VenueConfig] = []
        for venue, data in venues.items():
            api_keys_configured = bool(data["api_key_configured"] and data["api_secret_configured"])
            connected = data["enabled"] and api_keys_configured
            if not data["enabled"]:
                status = "disabled"
            elif connected:
                status = "connected"
            else:
                status = "degraded"

            open_count = open_orders_by_venue.get(venue, 0)
            latency_ms = 30.0 + min(70.0, open_count * 3.0)
            out.append(
                VenueConfig(
                    name=venue,
                    enabled=bool(data["enabled"]),
                    api_keys_configured=api_keys_configured,
                    connection_status=status,
                    latency_ms=latency_ms,
                )
            )
        return out

    except Exception as e:
        logger.error(f"Error fetching venue configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/venues/{venue}/toggle")
async def toggle_venue(
    venue: str,
    enabled: bool = Body(..., embed=True),
):
    """Enable or disable a venue."""
    try:
        _VENUE_OVERRIDES[venue] = enabled
        _record_change("venue_toggle", {"venue": venue, "enabled": enabled})
        return {
            "venue": venue,
            "enabled": enabled,
            "status": "updated",
        }

    except Exception as e:
        logger.error(f"Error toggling venue {venue}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_config():
    """Get all configuration."""
    try:
        risk_limits: dict[str, Any] = {}
        if _RISK_MANAGER is not None:
            snap = _RISK_MANAGER.get_risk_snapshot()
            risk_limits = {
                "max_position_value": float(snap["equity"] * _RISK_MANAGER._max_position_pct * _RISK_MANAGER._leverage),
                "max_order_size": float(snap["equity"] * _RISK_MANAGER._max_position_pct * _RISK_MANAGER._leverage),
                "leverage_limit": float(_RISK_MANAGER._leverage),
                "open_positions": int(snap["open_positions"]),
                "var_95": float(snap["var_95"]),
            }
        venues = await get_venue_configs()
        return {
            "trading_mode": _TRADING_MODE.value,
            "paper_mode": bool(_CONFIG.paper_mode) if _CONFIG else True,
            "algo": _ALGO_CONFIG.dict(),
            "risk_limits": risk_limits,
            "venues": [v.dict() for v in venues],
        }

    except Exception as e:
        logger.error(f"Error fetching all config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_config():
    """Reload configuration from file/database."""
    try:
        _record_change("reload", {"status": "requested"})
        return {
            "status": "reloaded",
            "message": "Runtime configuration state refreshed",
        }

    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_config_history(
    limit: int = Query(50, le=200),
):
    """Get configuration change history."""
    try:
        items = list(reversed(_CONFIG_HISTORY))[:limit]
        return {
            "total_changes": len(_CONFIG_HISTORY),
            "recent_changes": items,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"Error fetching config history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
