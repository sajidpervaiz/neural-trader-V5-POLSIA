from __future__ import annotations

from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from execution.cex_executor import CEXExecutor
from execution.bybit_executor import BybitExecutor
from execution.okx_executor import OKXExecutor
from execution.kraken_executor import KrakenExecutor
from execution.hyperliquid_executor import HyperliquidExecutor
from execution.risk_manager import RiskManager
from execution.variational_executor import VariationalExecutor


_EXECUTOR_MAP: dict[str, type[CEXExecutor]] = {
    "binance": CEXExecutor,
    "bybit": BybitExecutor,
    "okx": OKXExecutor,
    "kraken": KrakenExecutor,
    "hyperliquid": HyperliquidExecutor,
}


def create_executor(
    exchange_id: str,
    config: Config,
    event_bus: EventBus,
    risk_manager: RiskManager,
) -> CEXExecutor | None:
    cls = _EXECUTOR_MAP.get(exchange_id.lower())
    if cls is None:
        logger.warning("No executor class for exchange '{}'", exchange_id)
        return None

    cfg = config.get_value("exchanges", exchange_id) or {}
    if not cfg.get("enabled", False):
        logger.debug("Exchange '{}' is disabled — skipping executor creation", exchange_id)
        return None

    if exchange_id == "binance":
        executor = CEXExecutor(config, event_bus, risk_manager, exchange_id="binance")
    elif exchange_id == "bybit":
        executor = BybitExecutor(
            api_key=cfg.get("api_key", ""),
            api_secret=cfg.get("api_secret", ""),
            testnet=cfg.get("testnet", True),
            enable_paper_trading=cfg.get("paper_trading", True),
        )
    elif exchange_id == "okx":
        executor = OKXExecutor(
            api_key=cfg.get("api_key", ""),
            api_secret=cfg.get("api_secret", ""),
            passphrase=cfg.get("passphrase", ""),
            testnet=cfg.get("testnet", True),
            enable_paper_trading=cfg.get("paper_trading", True),
        )
    else:
        executor = cls(config, event_bus, risk_manager)

    logger.info("Created executor for '{}'", exchange_id)
    return executor


def create_all_executors(
    config: Config,
    event_bus: EventBus,
    risk_manager: RiskManager,
) -> list[CEXExecutor]:
    exchanges_cfg = config.get_value("exchanges") or {}
    executors = []
    for exchange_id in exchanges_cfg:
        executor = create_executor(exchange_id, config, event_bus, risk_manager)
        if executor is not None:
            executors.append(executor)
    logger.info("Created {} CEX executor(s)", len(executors))
    return executors


def create_variational_executor(
    config: Config,
    event_bus: EventBus,
    risk_manager: RiskManager,
) -> VariationalExecutor | None:
    """Create a Variational DEX executor if enabled in config."""
    var_cfg = config.get_value("variational") or {}
    if not var_cfg.get("enabled", False):
        logger.debug("Variational DEX disabled — skipping executor creation")
        return None
    executor = VariationalExecutor(config, event_bus, risk_manager)
    logger.info("Created Variational DEX executor")
    return executor
