"""
Startup validation — fail-fast before trading with real money.

Validates:
- API keys present and non-empty
- Exchange reachability and permissions (read + trade enabled)
- Account equity and available margin above minimum
- max_order_size_usd and risk_per_trade feasible given real balance
- Leverage within exchange limits per symbol
- Margin mode (isolated/cross) matches exchange
- Symbol availability, tick size, step size, min notional
- Clock sync with exchange
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.config import Config


class ValidationError(Exception):
    """Fatal validation error — bot must not start."""
    pass


# Stable USD-pegged assets that count 1:1 toward USDT-equivalent balance.
# Add more pegs here when needed (e.g. USDD, TUSD). BTC/ETH/etc. require a
# price feed and are intentionally excluded.
_STABLE_USD_ASSETS: tuple[str, ...] = ("USDT", "USDC", "BUSD", "FDUSD")


def _sum_stable_usd(balance_field: dict[str, Any] | None) -> float:
    """Sum the configured stable USD pegs from a ccxt balance['total'] / ['free'] dict."""
    if not isinstance(balance_field, dict):
        return 0.0
    total = 0.0
    for asset in _STABLE_USD_ASSETS:
        try:
            total += float(balance_field.get(asset, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


@dataclass
class SymbolSpec:
    """Extracted market spec for a single symbol."""
    symbol: str
    tick_size: float = 0.0
    step_size: float = 0.0
    min_notional: float = 0.0
    max_leverage: float = 125.0
    contract_size: float = 1.0


@dataclass
class ValidationResult:
    """Structured result of all startup checks."""
    mode: str = "live"
    passed: bool = True
    checks: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    symbol_specs: dict[str, SymbolSpec] = field(default_factory=dict)


class StartupValidator:
    """
    Validates all prerequisites before live trading.

    If any check fails, raises ValidationError which prevents bot startup.
    Only runs in non-paper mode.
    """

    def __init__(self, config: Config, client: Any = None) -> None:
        self.config = config
        self._client = client
        risk_cfg = config.get_value("risk") or {}
        self._min_balance_usd = float(risk_cfg.get("min_balance_usd", 100))
        self._max_order_size_usd = float(risk_cfg.get("max_order_size_usd", 500_000))
        self._risk_per_trade_pct = float(risk_cfg.get("risk_per_trade_pct", 0.01))
        self._max_position_size_pct = float(risk_cfg.get("max_position_size_pct", 0.02))
        self._default_leverage = float(risk_cfg.get("default_leverage", 1.0))
        self._max_leverage_per_symbol: dict[str, float] = risk_cfg.get(
            "max_leverage_per_symbol", {}
        )
        # Cached balance from permissions check — avoid duplicate fetch
        self._cached_balance: dict[str, Any] | None = None

    async def validate_all(self) -> dict[str, Any]:
        """
        Run all validations. Returns summary dict.
        Raises ValidationError on any critical failure.
        """
        if self.config.paper_mode:
            logger.info("Paper mode — skipping live startup validation")
            return {"mode": "paper", "skipped": True}

        result = ValidationResult()

        # 1. API keys (config-level, no network)
        self._check_api_keys(result)

        # 2. Exchange connectivity
        await self._check_connectivity(result)

        # 3. Exchange permissions (read + trade)
        await self._check_permissions(result)

        # 4. Account balance + feasibility
        await self._check_balance(result)

        # 5. Symbol availability + market specs (tick/step/min notional)
        await self._check_symbols(result)

        # 6. Leverage validation per symbol against exchange limits
        await self._check_leverage(result)

        # 7. Margin mode verification (config-side schema)
        await self._check_margin_mode(result)

        # 7b. Exchange-side leverage + margin verification (live truth)
        await self._check_exchange_side_leverage_margin(result)

        # 8. Order size feasibility
        self._check_order_size_feasibility(result)

        # 9. Clock sync
        await self._check_clock_sync(result)

        if result.warnings:
            for w in result.warnings:
                logger.warning("Startup warning: {}", w)

        logger.info("Startup validation PASSED — all checks OK")
        return {"mode": result.mode, "checks": result.checks, "warnings": result.warnings}

    # ── 1. API keys ───────────────────────────────────────────────────────

    def _check_api_keys(self, result: ValidationResult) -> None:
        """Validate API keys are present and non-empty for all enabled exchanges."""
        exchanges_cfg = self.config.get_value("exchanges") or {}
        for exchange_id, cfg in exchanges_cfg.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                continue
            api_key = str(cfg.get("api_key", "")).strip()
            api_secret = str(cfg.get("api_secret", "")).strip()

            if not api_key or api_key.startswith("${"):
                raise ValidationError(
                    f"API key missing for {exchange_id}. "
                    f"Set {exchange_id.upper()}_API_KEY environment variable."
                )
            if not api_secret or api_secret.startswith("${"):
                raise ValidationError(
                    f"API secret missing for {exchange_id}. "
                    f"Set {exchange_id.upper()}_API_SECRET environment variable."
                )

            result.checks[f"{exchange_id}_api_keys"] = "ok"
            logger.info("{} API keys present", exchange_id)

    # ── 2. Connectivity ──────────────────────────────────────────────────

    async def _check_connectivity(self, result: ValidationResult) -> None:
        """Verify exchange client is connected and functional."""
        if self._client is None:
            raise ValidationError("No exchange client — cannot trade live")
        try:
            await self._client.load_markets()
            result.checks["connectivity"] = "ok"
            logger.info("Exchange connectivity OK")
        except Exception as exc:
            raise ValidationError(f"Exchange unreachable: {exc}") from exc

    # ── 3. Permissions ───────────────────────────────────────────────────

    async def _check_permissions(self, result: ValidationResult) -> None:
        """Check that API key has trading permissions."""
        if self._client is None:
            return
        import asyncio
        last_exc = None
        for attempt in range(3):
            try:
                if attempt > 0:
                    await asyncio.sleep(2)
                balance = await self._client.fetch_balance()
                if not balance:
                    raise ValidationError("Empty balance response — check API permissions")
                # Cache for subsequent checks
                self._cached_balance = balance
                result.checks["permissions"] = "ok"
                logger.info("Exchange permissions OK (read+trade)")
                return
            except ValidationError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning("Balance fetch attempt {}/3 failed: {}", attempt + 1, exc)
        raise ValidationError(
            f"API permission check failed: {last_exc}. "
            "Ensure API key has 'Enable Futures' and 'Enable Reading' permissions."
        )

    # ── 4. Balance + feasibility ─────────────────────────────────────────

    async def _check_balance(self, result: ValidationResult) -> None:
        """Verify account equity and available margin meet requirements."""
        if self._client is None:
            return
        try:
            balance = self._cached_balance
            if balance is None:
                balance = await self._client.fetch_balance()
                self._cached_balance = balance

            # Sum stable USD pegs (USDT / USDC / BUSD / FDUSD) — accounts holding
            # USDC must not be rejected for "0 USDT".
            total_usdt = _sum_stable_usd(balance.get("total"))
            free_usdt = _sum_stable_usd(balance.get("free"))

            result.checks["balance"] = {
                "total_usdt": total_usdt,
                "free_usdt": free_usdt,
                "min_required": self._min_balance_usd,
            }

            # Hard minimum
            if total_usdt < self._min_balance_usd:
                raise ValidationError(
                    f"Insufficient balance: {total_usdt:.2f} USDT < "
                    f"minimum {self._min_balance_usd:.2f} USDT"
                )

            # risk_per_trade feasibility: equity × risk_per_trade must be > $1
            risk_amount = total_usdt * self._risk_per_trade_pct
            if risk_amount < 1.0:
                raise ValidationError(
                    f"risk_per_trade infeasible: {total_usdt:.2f} × "
                    f"{self._risk_per_trade_pct} = {risk_amount:.2f} USDT (need >= $1)"
                )

            # max_position_size_pct feasibility
            min_trade_size = total_usdt * self._max_position_size_pct
            if min_trade_size < 5:
                raise ValidationError(
                    f"Balance too low for configured position size: "
                    f"{total_usdt:.2f} × {self._max_position_size_pct} = "
                    f"{min_trade_size:.2f} USDT (need >= $5)"
                )

            # max_order_size_usd feasibility — can't exceed free margin × leverage
            max_notional = free_usdt * self._default_leverage
            if self._max_order_size_usd > max_notional:
                result.warnings.append(
                    f"max_order_size_usd ({self._max_order_size_usd:.0f}) exceeds "
                    f"available margin × leverage ({max_notional:.0f}). "
                    "Orders may be rejected by exchange."
                )

            logger.info(
                "Balance OK: total={:.2f} USDT, free={:.2f} USDT, "
                "risk/trade={:.2f} USDT",
                total_usdt, free_usdt, risk_amount,
            )
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Balance check failed: {exc}") from exc

    # ── 5. Symbol availability + market specs ────────────────────────────

    async def _check_symbols(self, result: ValidationResult) -> None:
        """Verify configured symbols exist and extract tick/step/min notional."""
        if self._client is None:
            return
        markets = self._client.markets or {}
        exchanges_cfg = self.config.get_value("exchanges") or {}
        client_exchange = getattr(self._client, "id", "")
        cfg = exchanges_cfg.get(client_exchange, {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return

        symbols = cfg.get("symbols", [])
        missing: list[str] = []

        for sym in symbols:
            market = markets.get(sym)
            if market is None:
                missing.append(sym)
                continue

            # Extract precision / limits from CCXT market structure
            precision = market.get("precision", {})
            limits = market.get("limits", {})
            info = market.get("info", {})

            tick_size = float(precision.get("price", 0))
            step_size = float(precision.get("amount", 0))

            # min notional: CCXT stores under limits.cost.min or filters
            cost_limits = limits.get("cost", {})
            min_notional = float(cost_limits.get("min", 0) if cost_limits else 0)

            # Some exchanges provide via info.filters (Binance)
            if min_notional == 0 and isinstance(info, dict):
                for f in info.get("filters", []):
                    if f.get("filterType") == "MIN_NOTIONAL":
                        min_notional = float(f.get("notional", 0))
                        break

            max_leverage = float(info.get("maxLeverage", 125)) if isinstance(info, dict) else 125.0
            contract_size = float(market.get("contractSize", 1) or 1)

            spec = SymbolSpec(
                symbol=sym,
                tick_size=tick_size,
                step_size=step_size,
                min_notional=min_notional,
                max_leverage=max_leverage,
                contract_size=contract_size,
            )
            result.symbol_specs[sym] = spec

            # Validate tick_size and step_size are set
            if tick_size <= 0:
                result.warnings.append(
                    f"{sym}: tick_size not resolved ({tick_size}) — "
                    "price rounding may fail"
                )
            if step_size <= 0:
                result.warnings.append(
                    f"{sym}: step_size not resolved ({step_size}) — "
                    "quantity rounding may fail"
                )

            logger.debug(
                "{}: tick={}, step={}, min_notional={}, max_lev={}",
                sym, tick_size, step_size, min_notional, max_leverage,
            )

        if missing:
            raise ValidationError(
                f"Symbols not available on {client_exchange}: {missing}. "
                "Check exchange market listing or fix config."
            )

        result.checks[f"{client_exchange}_symbols"] = "ok"
        logger.info("{} symbols validated: {}", client_exchange, symbols)

    # ── 6. Leverage ──────────────────────────────────────────────────────

    async def _check_leverage(self, result: ValidationResult) -> None:
        """Verify leverage settings are within exchange limits per symbol."""
        risk_cfg = self.config.get_value("risk") or {}
        default_lev = float(risk_cfg.get("default_leverage", 1.0))

        if default_lev <= 0:
            raise ValidationError(
                f"Invalid default_leverage: {default_lev}. Must be >= 1."
            )
        if default_lev > 20:
            raise ValidationError(
                f"Leverage {default_lev}x is dangerously high. "
                "Maximum recommended: 20x. Reduce risk.default_leverage."
            )
        if default_lev > 10:
            result.warnings.append(
                f"High leverage: {default_lev}x — ensure adequate margin"
            )

        # Per-symbol leverage checks against exchange max
        for sym, spec in result.symbol_specs.items():
            sym_lev = float(self._max_leverage_per_symbol.get(sym, default_lev))
            if sym_lev > spec.max_leverage:
                raise ValidationError(
                    f"Configured leverage for {sym} ({sym_lev}x) exceeds "
                    f"exchange maximum ({spec.max_leverage}x). "
                    f"Reduce leverage for {sym}."
                )
            if sym_lev <= 0:
                raise ValidationError(
                    f"Invalid leverage for {sym}: {sym_lev}. Must be >= 1."
                )

        result.checks["leverage"] = {"configured": default_lev, "status": "ok"}
        logger.info("Leverage check OK: default={}x", default_lev)

    # ── 7. Margin mode ───────────────────────────────────────────────────

    async def _check_margin_mode(self, result: ValidationResult) -> None:
        """Verify margin mode is valid and consistent."""
        exchanges_cfg = self.config.get_value("exchanges") or {}
        client_exchange = getattr(self._client, "id", "") if self._client else ""
        cfg = exchanges_cfg.get(client_exchange, {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return

        margin_mode = str(cfg.get("margin_mode", "isolated")).lower()
        if margin_mode not in ("isolated", "cross"):
            raise ValidationError(
                f"Invalid margin_mode '{margin_mode}' for {client_exchange}. "
                "Must be 'isolated' or 'cross'."
            )

        result.checks["margin_mode"] = {"mode": margin_mode, "status": "ok"}
        logger.info("Margin mode OK: {}", margin_mode)

    # ── 7b. Exchange-side leverage + margin verification ─────────────────

    async def _check_exchange_side_leverage_margin(self, result: ValidationResult) -> None:
        """Verify exchange's actual leverage + margin mode match config per symbol.

        Previously validation only checked config strings against schema. The
        exchange might carry stale 50x / cross from a prior session; the bot
        would happily start with 1x in config but trade at 50x in reality.
        Mismatches surface as warnings (operator-visible at boot).
        """
        if self._client is None:
            return
        exchanges_cfg = self.config.get_value("exchanges") or {}
        client_exchange = getattr(self._client, "id", "")
        cfg = exchanges_cfg.get(client_exchange, {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return
        symbols = cfg.get("symbols", [])
        if not symbols:
            return
        expected_margin = str(cfg.get("margin_mode", "isolated")).lower()
        expected_lev_default = float(
            cfg.get("leverage", self._default_leverage) or self._default_leverage
        )
        try:
            positions = await self._client.fetch_positions(symbols)
        except Exception as exc:
            result.warnings.append(f"could_not_verify_exchange_leverage: {exc}")
            return
        mismatches: list[str] = []
        for p in positions or []:
            sym = p.get("symbol", "")
            if sym not in symbols:
                continue
            actual_lev = int(float(p.get("leverage", 0) or 0))
            actual_margin = str(p.get("marginMode") or p.get("marginType") or "").lower()
            expected_lev = int(
                float(self._max_leverage_per_symbol.get(sym, expected_lev_default) or 0)
            )
            if actual_lev > 0 and expected_lev > 0 and actual_lev != expected_lev:
                mismatches.append(
                    f"{sym}: exchange leverage {actual_lev}x != config {expected_lev}x"
                )
            if actual_margin and expected_margin and actual_margin != expected_margin:
                mismatches.append(
                    f"{sym}: exchange margin '{actual_margin}' != config '{expected_margin}'"
                )
        if mismatches:
            for m in mismatches:
                result.warnings.append(m)
            logger.warning("Exchange-side leverage/margin mismatch detected: {}", mismatches)
        result.checks["exchange_side_leverage_margin"] = (
            "ok" if not mismatches else f"{len(mismatches)} mismatch(es) — see warnings"
        )

    # ── 8. Order size feasibility ────────────────────────────────────────

    def _check_order_size_feasibility(self, result: ValidationResult) -> None:
        """Verify max_order_size and risk_per_trade are feasible for each symbol."""
        balance_info = result.checks.get("balance", {})
        total_usdt = float(balance_info.get("total_usdt", 0))

        if total_usdt <= 0:
            return  # Balance check already failed

        for sym, spec in result.symbol_specs.items():
            # min notional check: ensure risk_per_trade can produce an order
            # above exchange minimum
            if spec.min_notional > 0:
                risk_amount = total_usdt * self._risk_per_trade_pct
                # With leverage, the notional = risk_amount × leverage / SL_distance
                # As a rough check, ensure equity × max_position_pct × leverage > min_notional
                sym_lev = float(
                    self._max_leverage_per_symbol.get(sym, self._default_leverage)
                )
                max_notional = total_usdt * self._max_position_size_pct * sym_lev
                if max_notional < spec.min_notional:
                    raise ValidationError(
                        f"{sym}: max position notional ({max_notional:.2f} USDT) < "
                        f"exchange min notional ({spec.min_notional:.2f} USDT). "
                        "Increase balance, leverage, or position size."
                    )

        result.checks["order_size_feasibility"] = "ok"

    # ── 9. Clock sync ────────────────────────────────────────────────────

    async def _check_clock_sync(self, result: ValidationResult) -> None:
        """Check clock drift with exchange server."""
        if self._client is None:
            return
        try:
            server_time = await self._client.fetch_time()
            local_time = int(time.time() * 1000)
            drift_ms = abs(server_time - local_time)
            result.checks["clock_drift_ms"] = drift_ms

            if drift_ms > 5000:
                raise ValidationError(
                    f"Clock drift too high: {drift_ms}ms. "
                    "Sync system clock with NTP."
                )
            elif drift_ms > 1000:
                result.warnings.append(f"Clock drift: {drift_ms}ms — consider NTP sync")
            else:
                logger.info("Clock sync OK (drift={}ms)", drift_ms)
        except ValidationError:
            raise
        except Exception as exc:
            logger.warning("Clock sync check failed (non-fatal): {}", exc)
            result.checks["clock_drift_ms"] = -1
