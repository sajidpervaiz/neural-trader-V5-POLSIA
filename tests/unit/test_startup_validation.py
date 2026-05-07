"""
Comprehensive tests for API key + balance validation (Prompt #6).

Covers:
- Missing / empty / unresolved API keys
- Insufficient balance, infeasible risk_per_trade, infeasible position size
- Invalid leverage (zero, negative, exceeds exchange max, >20x)
- Invalid margin mode
- Missing / invalid symbols
- Tick size, step size, min notional validation
- Order size feasibility against min notional
- Clock drift
- Exchange unreachable / no client
- Config schema enforcement (risk coherence, live-mode guards)
- Paper mode skip
- Full happy-path validation
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import Config
from execution.startup_validation import (
    StartupValidator,
    SymbolSpec,
    ValidationError,
    ValidationResult,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(
    paper_mode: bool = False,
    exchanges: dict | None = None,
    risk: dict | None = None,
) -> Config:
    """Create a Config with mocked get_value for test control."""
    config = Config.get(path=str(CONFIG_PATH))
    original_paper = config.paper_mode
    config.paper_mode = paper_mode

    default_exchanges = {
        "binance": {
            "enabled": True,
            "api_key": "test_key_123",
            "api_secret": "test_secret_456",
            "testnet": True,
            "type": "futures",
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "margin_mode": "isolated",
            "leverage": 5,
        }
    }
    default_risk = {
        "min_balance_usd": 100,
        "max_order_size_usd": 500_000,
        "risk_per_trade_pct": 0.01,
        "max_position_size_pct": 0.02,
        "default_leverage": 5.0,
        "max_leverage_per_symbol": {},
    }

    exc_cfg = exchanges if exchanges is not None else default_exchanges
    rsk_cfg = risk if risk is not None else default_risk

    patcher = patch.object(config, "get_value")
    mock_get = patcher.start()

    def side_effect(*args, **kwargs):
        if args == ("exchanges",):
            return exc_cfg
        if len(args) >= 2 and args[0] == "exchanges":
            return exc_cfg.get(args[1], kwargs.get("default"))
        if args == ("risk",):
            return rsk_cfg
        return kwargs.get("default")

    mock_get.side_effect = side_effect

    # Store patcher and original for cleanup
    config._test_patcher = patcher
    config._test_original_paper = original_paper
    return config


def _cleanup_config(config: Config) -> None:
    """Restore config after test."""
    config.paper_mode = config._test_original_paper
    config._test_patcher.stop()


def _make_client(
    balance: dict | None = None,
    markets: dict | None = None,
    server_time_ms: int | None = None,
    exchange_id: str = "binance",
) -> AsyncMock:
    """Create a mock CCXT exchange client."""
    client = AsyncMock()
    client.id = exchange_id

    if balance is None:
        balance = {
            "total": {"USDT": 10_000.0},
            "free": {"USDT": 9_000.0},
        }
    client.fetch_balance.return_value = balance

    if markets is None:
        markets = {
            "BTC/USDT:USDT": {
                "precision": {"price": 0.01, "amount": 0.001},
                "limits": {"cost": {"min": 5.0}},
                "info": {
                    "maxLeverage": "125",
                    "filters": [
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                    ],
                },
                "contractSize": 1,
            },
            "ETH/USDT:USDT": {
                "precision": {"price": 0.01, "amount": 0.01},
                "limits": {"cost": {"min": 5.0}},
                "info": {
                    "maxLeverage": "50",
                    "filters": [],
                },
                "contractSize": 1,
            },
        }
    client.markets = markets
    client.load_markets.return_value = markets

    if server_time_ms is None:
        server_time_ms = int(time.time() * 1000)
    client.fetch_time.return_value = server_time_ms

    return client


# ═══════════════════════════════════════════════════════════════════════════════
# Paper mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperModeSkip:
    @pytest.mark.asyncio
    async def test_paper_mode_skips_all_validation(self):
        config = _make_config(paper_mode=True)
        try:
            validator = StartupValidator(config, client=None)
            result = await validator.validate_all()
            assert result["mode"] == "paper"
            assert result["skipped"] is True
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# API Key Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAPIKeyValidation:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        config = _make_config(exchanges={
            "binance": {"enabled": True, "api_key": "", "api_secret": "secret123"},
        })
        try:
            validator = StartupValidator(config, client=_make_client())
            with pytest.raises(ValidationError, match="API key missing"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_missing_api_secret_raises(self):
        config = _make_config(exchanges={
            "binance": {"enabled": True, "api_key": "key123", "api_secret": ""},
        })
        try:
            validator = StartupValidator(config, client=_make_client())
            with pytest.raises(ValidationError, match="API secret missing"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_unresolved_env_var_raises(self):
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "${BINANCE_API_KEY}",
                "api_secret": "secret123",
            },
        })
        try:
            validator = StartupValidator(config, client=_make_client())
            with pytest.raises(ValidationError, match="API key missing"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_whitespace_only_key_raises(self):
        config = _make_config(exchanges={
            "binance": {"enabled": True, "api_key": "   ", "api_secret": "secret"},
        })
        try:
            validator = StartupValidator(config, client=_make_client())
            with pytest.raises(ValidationError, match="API key missing"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_disabled_exchange_skipped(self):
        config = _make_config(exchanges={
            "binance": {
                "enabled": False,
                "api_key": "",
                "api_secret": "",
            },
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            # Should not raise for disabled exchange, but will raise for no-client
            # since no enabled exchange has keys
            result = await validator.validate_all()
            assert "binance_api_keys" not in result["checks"]
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_valid_api_keys_pass(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["binance_api_keys"] == "ok"
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Connectivity
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectivity:
    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        config = _make_config()
        try:
            validator = StartupValidator(config, client=None)
            with pytest.raises(ValidationError, match="No exchange client"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_load_markets_failure_raises(self):
        config = _make_config()
        try:
            client = _make_client()
            client.load_markets.side_effect = Exception("Connection refused")
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Exchange unreachable"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Permissions
# ═══════════════════════════════════════════════════════════════════════════════


class TestPermissions:
    @pytest.mark.asyncio
    async def test_empty_balance_response_raises(self):
        config = _make_config()
        try:
            client = _make_client()
            client.fetch_balance.return_value = {}
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Empty balance.*permissions"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_permission_denied_raises(self):
        config = _make_config()
        try:
            client = _make_client()
            client.fetch_balance.side_effect = Exception("API key has no permission")
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="API permission check failed"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Balance Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestBalanceValidation:
    @pytest.mark.asyncio
    async def test_insufficient_total_balance_raises(self):
        config = _make_config(risk={"min_balance_usd": 1000, "risk_per_trade_pct": 0.01,
                                    "max_position_size_pct": 0.02, "default_leverage": 5.0,
                                    "max_order_size_usd": 500_000, "max_leverage_per_symbol": {}})
        try:
            client = _make_client(balance={
                "total": {"USDT": 50.0},
                "free": {"USDT": 50.0},
            })
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Insufficient balance"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_infeasible_risk_per_trade_raises(self):
        """Balance × risk_per_trade < $1."""
        config = _make_config(risk={"min_balance_usd": 10, "risk_per_trade_pct": 0.001,
                                    "max_position_size_pct": 0.5, "default_leverage": 1.0,
                                    "max_order_size_usd": 500_000, "max_leverage_per_symbol": {}})
        try:
            client = _make_client(balance={
                "total": {"USDT": 50.0},
                "free": {"USDT": 50.0},
            })
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="risk_per_trade infeasible"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_infeasible_position_size_raises(self):
        """Balance × max_position_size_pct < $5."""
        config = _make_config(risk={"min_balance_usd": 10, "risk_per_trade_pct": 0.1,
                                    "max_position_size_pct": 0.001, "default_leverage": 1.0,
                                    "max_order_size_usd": 500_000, "max_leverage_per_symbol": {}})
        try:
            client = _make_client(balance={
                "total": {"USDT": 100.0},
                "free": {"USDT": 100.0},
            })
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Balance too low"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_max_order_exceeds_margin_warns(self):
        """max_order_size > free × leverage triggers warning."""
        config = _make_config(risk={"min_balance_usd": 100, "risk_per_trade_pct": 0.01,
                                    "max_position_size_pct": 0.02, "default_leverage": 2.0,
                                    "max_order_size_usd": 500_000, "max_leverage_per_symbol": {}})
        try:
            client = _make_client(balance={
                "total": {"USDT": 10_000.0},
                "free": {"USDT": 9_000.0},
            })
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert any("max_order_size_usd" in w for w in result["warnings"])
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_adequate_balance_passes(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["balance"]["total_usdt"] == 10_000.0
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Symbol Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymbolValidation:
    @pytest.mark.asyncio
    async def test_missing_symbol_raises(self):
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT", "FAKE/USDT:USDT"],
                "margin_mode": "isolated",
            },
        })
        try:
            client = _make_client(markets={
                "BTC/USDT:USDT": {
                    "precision": {"price": 0.01, "amount": 0.001},
                    "limits": {"cost": {"min": 5.0}},
                    "info": {"maxLeverage": "125"},
                    "contractSize": 1,
                },
            })
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="FAKE/USDT:USDT"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_tick_size_warning(self):
        """Zero tick_size emits a warning."""
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "isolated",
            },
        })
        try:
            client = _make_client(markets={
                "BTC/USDT:USDT": {
                    "precision": {"price": 0, "amount": 0.001},
                    "limits": {"cost": {"min": 5.0}},
                    "info": {"maxLeverage": "125"},
                    "contractSize": 1,
                },
            })
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert any("tick_size" in w for w in result["warnings"])
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_step_size_warning(self):
        """Zero step_size emits a warning."""
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "isolated",
            },
        })
        try:
            client = _make_client(markets={
                "BTC/USDT:USDT": {
                    "precision": {"price": 0.01, "amount": 0},
                    "limits": {"cost": {"min": 5.0}},
                    "info": {"maxLeverage": "125"},
                    "contractSize": 1,
                },
            })
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert any("step_size" in w for w in result["warnings"])
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_min_notional_extracted_from_filters(self):
        """Min notional extracted from Binance-style filters."""
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "isolated",
            },
        })
        try:
            client = _make_client(markets={
                "BTC/USDT:USDT": {
                    "precision": {"price": 0.01, "amount": 0.001},
                    "limits": {"cost": {}},  # No min in limits
                    "info": {
                        "maxLeverage": "125",
                        "filters": [
                            {"filterType": "MIN_NOTIONAL", "notional": "10.0"},
                        ],
                    },
                    "contractSize": 1,
                },
            })
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            # Access internal result for symbol_specs verification
            assert result["checks"]["binance_symbols"] == "ok"
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_all_symbols_valid_passes(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["binance_symbols"] == "ok"
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Leverage Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestLeverageValidation:
    @pytest.mark.asyncio
    async def test_leverage_over_20_raises(self):
        config = _make_config(risk={
            "min_balance_usd": 100, "risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.02, "default_leverage": 50.0,
            "max_order_size_usd": 500_000, "max_leverage_per_symbol": {},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="dangerously high"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_leverage_zero_raises(self):
        config = _make_config(risk={
            "min_balance_usd": 100, "risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.02, "default_leverage": 0.0,
            "max_order_size_usd": 500_000, "max_leverage_per_symbol": {},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Invalid default_leverage"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_per_symbol_leverage_exceeds_exchange_max_raises(self):
        """ETH max on exchange is 50x, but config asks for 100x."""
        config = _make_config(risk={
            "min_balance_usd": 100, "risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.02, "default_leverage": 5.0,
            "max_order_size_usd": 500_000,
            "max_leverage_per_symbol": {"ETH/USDT:USDT": 100.0},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="exceeds exchange maximum"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_per_symbol_negative_leverage_raises(self):
        config = _make_config(risk={
            "min_balance_usd": 100, "risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.02, "default_leverage": 5.0,
            "max_order_size_usd": 500_000,
            "max_leverage_per_symbol": {"BTC/USDT:USDT": -1.0},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Invalid leverage"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_leverage_15x_requires_acknowledgement(self):
        """Issue #23: 10x-20x range now requires explicit
        ACKNOWLEDGE_HIGH_LEVERAGE=true env var to start. Without it,
        ValidationError. With it, warning row recorded + critical log."""
        import os as _os
        config = _make_config(risk={
            "min_balance_usd": 100, "risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.02, "default_leverage": 15.0,
            "max_order_size_usd": 500_000, "max_leverage_per_symbol": {},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            # Without env var → ValidationError refuses to start
            _os.environ.pop("ACKNOWLEDGE_HIGH_LEVERAGE", None)
            with pytest.raises(ValidationError, match="ACKNOWLEDGE_HIGH_LEVERAGE"):
                await validator.validate_all()
            # With env var → starts, but emits warning row
            _os.environ["ACKNOWLEDGE_HIGH_LEVERAGE"] = "true"
            try:
                result = await validator.validate_all()
                assert any("HIGH LEVERAGE" in w for w in result["warnings"])
            finally:
                _os.environ.pop("ACKNOWLEDGE_HIGH_LEVERAGE", None)
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_valid_leverage_passes(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["leverage"]["status"] == "ok"
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Margin Mode Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarginModeValidation:
    @pytest.mark.asyncio
    async def test_invalid_margin_mode_raises(self):
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "hedged",
            },
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Invalid margin_mode"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_isolated_mode_passes(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["margin_mode"]["mode"] == "isolated"
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_cross_mode_passes(self):
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "cross",
            },
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["margin_mode"]["mode"] == "cross"
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Order Size Feasibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderSizeFeasibility:
    @pytest.mark.asyncio
    async def test_min_notional_too_high_raises(self):
        """Position notional < exchange min_notional."""
        config = _make_config(
            exchanges={
                "binance": {
                    "enabled": True,
                    "api_key": "key",
                    "api_secret": "secret",
                    "symbols": ["BTC/USDT:USDT"],
                    "margin_mode": "isolated",
                },
            },
            risk={
                "min_balance_usd": 100,
                "risk_per_trade_pct": 0.01,
                "max_position_size_pct": 0.02,
                "default_leverage": 1.0,
                "max_order_size_usd": 500_000,
                "max_leverage_per_symbol": {},
            },
        )
        try:
            # Balance is 1000; max pos = 1000 * 0.02 * 1x = 20 USDT < 500 min_notional
            client = _make_client(
                balance={"total": {"USDT": 1_000.0}, "free": {"USDT": 1_000.0}},
                markets={
                    "BTC/USDT:USDT": {
                        "precision": {"price": 0.01, "amount": 0.001},
                        "limits": {"cost": {"min": 500.0}},
                        "info": {"maxLeverage": "125"},
                        "contractSize": 1,
                    },
                },
            )
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="min notional"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_feasible_order_size_passes(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["order_size_feasibility"] == "ok"
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Clock Sync
# ═══════════════════════════════════════════════════════════════════════════════


class TestClockSync:
    @pytest.mark.asyncio
    async def test_large_clock_drift_raises(self):
        config = _make_config()
        try:
            client = _make_client(
                server_time_ms=int(time.time() * 1000) + 10_000,
            )
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="Clock drift"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_moderate_drift_warns(self):
        config = _make_config()
        try:
            client = _make_client(
                server_time_ms=int(time.time() * 1000) + 2_000,
            )
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert any("Clock drift" in w for w in result["warnings"])
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_small_drift_passes_clean(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            drift = result["checks"].get("clock_drift_ms", -1)
            assert drift >= 0
            assert drift < 5000
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_clock_check_failure_non_fatal(self):
        """fetch_time exception is logged but does not raise."""
        config = _make_config()
        try:
            client = _make_client()
            client.fetch_time.side_effect = Exception("timeout")
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["checks"]["clock_drift_ms"] == -1
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Happy Path
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullValidation:
    @pytest.mark.asyncio
    async def test_all_checks_pass(self):
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert result["mode"] == "live"
            checks = result["checks"]
            assert checks["binance_api_keys"] == "ok"
            assert checks["connectivity"] == "ok"
            assert checks["permissions"] == "ok"
            assert checks["balance"]["total_usdt"] == 10_000.0
            assert checks["binance_symbols"] == "ok"
            assert checks["leverage"]["status"] == "ok"
            assert checks["margin_mode"]["status"] == "ok"
            assert checks["order_size_feasibility"] == "ok"
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_balance_cached_across_checks(self):
        """fetch_balance should be called once (cached between permissions and balance)."""
        config = _make_config()
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            await validator.validate_all()
            # Once in _check_permissions (cached), not again in _check_balance
            assert client.fetch_balance.call_count == 1
        finally:
            _cleanup_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Data model tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymbolSpec:
    def test_symbol_spec_defaults(self):
        spec = SymbolSpec(symbol="BTC/USDT")
        assert spec.tick_size == 0.0
        assert spec.step_size == 0.0
        assert spec.min_notional == 0.0
        assert spec.max_leverage == 125.0

    def test_symbol_spec_custom(self):
        spec = SymbolSpec(
            symbol="ETH/USDT", tick_size=0.01, step_size=0.001,
            min_notional=5.0, max_leverage=50.0, contract_size=1.0,
        )
        assert spec.max_leverage == 50.0


class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult()
        assert r.passed is True
        assert r.mode == "live"
        assert r.checks == {}
        assert r.warnings == []


# ═══════════════════════════════════════════════════════════════════════════════
# Config Schema Enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigSchemaEnforcement:
    def test_risk_sl_gte_tp_rejected(self):
        """stop_loss_pct >= take_profit_pct should fail."""
        from core.config_schema import RiskConfig
        with pytest.raises(Exception, match="stop_loss_pct"):
            RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.03)

    def test_risk_drawdown_lt_daily_loss_rejected(self):
        """max_drawdown_pct < max_daily_loss_pct should fail."""
        from core.config_schema import RiskConfig
        with pytest.raises(Exception, match="max_drawdown_pct"):
            RiskConfig(max_drawdown_pct=0.02, max_daily_loss_pct=0.03)

    def test_per_symbol_leverage_negative_rejected(self):
        """Negative leverage per symbol should fail."""
        from core.config_schema import RiskConfig
        with pytest.raises(Exception, match="leverage.*must be > 0"):
            RiskConfig(max_leverage_per_symbol={"BTC/USDT": -5})

    def test_per_symbol_leverage_over_125_rejected(self):
        from core.config_schema import RiskConfig
        with pytest.raises(Exception, match="exceeds exchange max"):
            RiskConfig(max_leverage_per_symbol={"BTC/USDT": 200})

    def test_leverage_over_125_rejected(self):
        from core.config_schema import RiskConfig
        with pytest.raises(Exception):
            RiskConfig(default_leverage=200.0)

    def test_valid_risk_config_passes(self):
        from core.config_schema import RiskConfig
        cfg = RiskConfig()
        assert cfg.default_leverage == 1.0
        assert cfg.min_balance_usd == 100

    def test_sizing_method_invalid_rejected(self):
        from core.config_schema import RiskConfig
        with pytest.raises(Exception, match="sizing_method"):
            RiskConfig(sizing_method="yolo")

    def test_live_mode_no_symbols_rejected(self):
        """Enabled exchange with no symbols in live mode fails."""
        from core.config_schema import AppConfig, ExchangeConfig, SystemConfig
        with pytest.raises(Exception, match="no symbols"):
            AppConfig(
                system=SystemConfig(paper_mode=False),
                exchanges={"binance": ExchangeConfig(enabled=True, symbols=[],
                                                      api_key="k", api_secret="s")},
            )

    def test_live_mode_zero_min_balance_rejected(self):
        from core.config_schema import AppConfig, ExchangeConfig, SystemConfig, RiskConfig
        with pytest.raises(Exception, match="min_balance_usd"):
            AppConfig(
                system=SystemConfig(paper_mode=False),
                exchanges={"binance": ExchangeConfig(
                    enabled=True, symbols=["BTC/USDT"],
                    api_key="k", api_secret="s",
                )},
                risk=RiskConfig(min_balance_usd=0),
            )

    def test_live_mode_leverage_over_20_rejected(self):
        from core.config_schema import AppConfig, ExchangeConfig, SystemConfig, RiskConfig
        with pytest.raises(Exception, match="default_leverage.*dangerous"):
            AppConfig(
                system=SystemConfig(paper_mode=False),
                exchanges={"binance": ExchangeConfig(
                    enabled=True, symbols=["BTC/USDT"],
                    api_key="k", api_secret="s",
                )},
                risk=RiskConfig(default_leverage=25.0),
            )

    def test_paper_mode_allows_high_leverage(self):
        """Paper mode allows >20x leverage (for backtesting)."""
        from core.config_schema import AppConfig, SystemConfig, RiskConfig
        cfg = AppConfig(
            system=SystemConfig(paper_mode=True),
            risk=RiskConfig(default_leverage=25.0),
        )
        assert cfg.risk.default_leverage == 25.0


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_multiple_exchanges_validates_all_keys(self):
        """When second exchange has missing key, it raises."""
        config = _make_config(exchanges={
            "binance": {
                "enabled": True,
                "api_key": "key1",
                "api_secret": "secret1",
                "symbols": ["BTC/USDT:USDT"],
                "margin_mode": "isolated",
            },
            "bybit": {
                "enabled": True,
                "api_key": "",
                "api_secret": "secret2",
                "symbols": ["BTC/USDT:USDT"],
            },
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            with pytest.raises(ValidationError, match="API key missing for bybit"):
                await validator.validate_all()
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_no_enabled_exchange_passes_api_check(self):
        """If zero exchanges enabled, API key check passes (nothing to check)."""
        config = _make_config(exchanges={
            "binance": {"enabled": False, "api_key": "", "api_secret": ""},
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert "binance_api_keys" not in result["checks"]
        finally:
            _cleanup_config(config)

    @pytest.mark.asyncio
    async def test_exchange_cfg_not_dict_skipped(self):
        """Non-dict exchange config entries are skipped."""
        config = _make_config(exchanges={
            "binance": "invalid_config_string",
        })
        try:
            client = _make_client()
            validator = StartupValidator(config, client=client)
            result = await validator.validate_all()
            assert "binance_api_keys" not in result["checks"]
        finally:
            _cleanup_config(config)
