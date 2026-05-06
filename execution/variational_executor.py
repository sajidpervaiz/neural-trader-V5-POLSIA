"""Variational DEX executor — perpetual futures via RFQ protocol.

Integrates with https://omni.variational.io/ using the official `variational` SDK.
Supports both testnet and mainnet.  Follows the same signal→execute pattern as CEXExecutor.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.cex_executor import OrderResult
from execution.rate_limiter import RateLimiter
from execution.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Lazy-import variational SDK so the rest of the app doesn't break when the
# package isn't installed (e.g. during tests).
# ---------------------------------------------------------------------------
try:
    from variational import (
        Client as VariationalClient,
        TESTNET as VAR_TESTNET,
        MAINNET as VAR_MAINNET,
        PerpetualFuture,
        Structure,
        Leg,
        LegQuote,
        TradeSide,
        UseExistingPool,
        CreateNewPool,
        InstrumentType,
        PollingHelper,
        ClearingStatus,
    )
    HAS_VARIATIONAL = True
except ImportError:
    HAS_VARIATIONAL = False


# Symbol mapping: our internal symbol → Variational underlying + settlement
_SYMBOL_MAP: dict[str, tuple[str, str]] = {
    "BTC/USDT:USDT": ("BTC", "USDT"),
    "ETH/USDT:USDT": ("ETH", "USDT"),
    "SOL/USDT:USDT": ("SOL", "USDT"),
    "BTC/USDT":      ("BTC", "USDT"),
    "ETH/USDT":      ("ETH", "USDT"),
    "SOL/USDT":      ("SOL", "USDT"),
}


class VariationalExecutor:
    """Trade perpetual futures on Variational DEX via RFQ flow."""

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self._client: Any = None
        self._poller: Any = None
        self._running = False
        self._rate_limiter = RateLimiter(max_calls=5, period_seconds=1.0)
        # Cached pool location for reuse across trades
        self._pool_location: str | None = None
        self._counterparty: str | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _init_client(self) -> None:
        if self._client is not None:
            return

        if not HAS_VARIATIONAL:
            logger.error("variational SDK not installed — run: pip install variational")
            return

        var_cfg = self.config.get_value("variational") or {}
        if not var_cfg.get("enabled", False):
            logger.info("Variational executor disabled in config")
            return

        api_key = var_cfg.get("api_key", "")
        api_secret = var_cfg.get("api_secret", "")
        if not api_key or not api_secret:
            logger.error("Variational API key/secret not configured")
            return

        testnet = var_cfg.get("testnet", True)
        base_url = VAR_TESTNET if testnet else VAR_MAINNET

        try:
            self._client = VariationalClient(
                key=api_key,
                secret=api_secret,
                base_url=base_url,
            )
            self._poller = PollingHelper(self._client)

            # Verify connection
            me = self._client.get_me()
            logger.info(
                "Variational connected: {} ({})",
                me.data.get("company", "unknown") if me.data else "unknown",
                "testnet" if testnet else "MAINNET",
            )

            # Cache counterparty (Omni market-maker) and pool if configured
            self._counterparty = var_cfg.get("counterparty", "")
            self._pool_location = var_cfg.get("pool_location", "")

        except Exception as exc:
            logger.error("Variational client init failed: {}", exc)
            self._client = None

    # ------------------------------------------------------------------
    # Core execution flow: Signal → RFQ → Quote → Accept → Trade
    # ------------------------------------------------------------------

    async def execute_signal(self, signal: TradingSignal, size: float) -> OrderResult | None:
        """Execute a trading signal on Variational.

        Flow: create RFQ → wait for quotes → accept best → wait for clearing.
        """
        if self._client is None:
            await self._init_client()
            if self._client is None:
                return None

        var_cfg = self.config.get_value("variational") or {}
        underlying, settlement = _SYMBOL_MAP.get(signal.symbol, (None, None))
        if not underlying:
            logger.warning("Variational: unsupported symbol {}", signal.symbol)
            return None

        try:
            # 1. Build the perpetual future instrument
            instrument = PerpetualFuture(
                instrument_type=InstrumentType.PERPETUAL_FUTURE,
                underlying=underlying,
                settlement_asset=settlement,
                funding_interval_s=3600,
                dex_token_details=None,
            )

            # 2. Build the structure (single-leg perp)
            side = TradeSide.BUY if signal.is_long else TradeSide.SELL
            structure = Structure(
                legs=[Leg(side=side, ratio=1, instrument=instrument)]
            )

            # 3. Calculate quantity in base units
            qty = size / signal.price if signal.price > 0 else 0.0
            if qty <= 0:
                logger.warning("Variational: invalid qty for {} — size={} price={}", signal.symbol, size, signal.price)
                return None

            # 4. Create RFQ
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
            target_companies = [self._counterparty] if self._counterparty else []

            rfq_resp = self._client.create_rfq(
                structure=structure,
                qty=str(round(qty, 8)),
                expires_at=expires_at,
                target_companies=target_companies,
            )

            if not rfq_resp.data:
                logger.error("Variational: RFQ creation failed — no data returned")
                return None

            rfq_id = rfq_resp.data["rfq_id"]
            logger.info(
                "Variational RFQ created: {} {} {} qty={:.6f} rfq_id={}",
                side.value.upper(), underlying, settlement, qty, rfq_id,
            )

            # 5. Poll for quotes (wait up to 30s)
            best_quote = await self._wait_for_best_quote(rfq_id, side, timeout=30)
            if best_quote is None:
                logger.warning("Variational: no quotes received for RFQ {}", rfq_id)
                self._client.cancel_rfq(rfq_id)
                return None

            quote_id = best_quote["parent_quote_id"]
            quote_price = float(best_quote.get("price", signal.price))

            # 6. Accept the best quote
            accept_resp = self._client.accept_quote(
                rfq_id=rfq_id,
                parent_quote_id=quote_id,
                side=side,
            )

            if not accept_resp.data:
                logger.error("Variational: quote accept failed for RFQ {}", rfq_id)
                return None

            logger.info(
                "Variational quote accepted: rfq={} quote={} price={:.2f}",
                rfq_id, quote_id, quote_price,
            )

            # 7. Wait for clearing
            try:
                cleared = self._poller.wait_for_clearing_status(
                    parent_quote_id=quote_id,
                    status=ClearingStatus.CONFIRMED,
                )
                logger.info("Variational trade cleared: quote={}", quote_id)
            except Exception as ce:
                logger.warning("Variational: clearing wait failed: {} — trade may still settle", ce)

            # 8. Build result
            result = OrderResult(
                order_id=f"var_{rfq_id}_{quote_id}",
                exchange="variational",
                symbol=signal.symbol,
                direction=signal.direction,
                price=quote_price,
                quantity=qty,
                status="filled",
                is_paper=False,
                timestamp=int(time.time()),
                raw={"rfq_id": rfq_id, "quote_id": quote_id},
            )

            await self.event_bus.publish("ORDER_FILLED", result)
            logger.info(
                "Variational {} filled: {} {:.6f} @ {:.2f} [{}]",
                signal.direction.upper(), signal.symbol, qty, quote_price, result.order_id,
            )
            return result

        except Exception as exc:
            logger.error("Variational execute_signal failed: {}", exc)
            return None

    async def _wait_for_best_quote(
        self, rfq_id: str, side: Any, timeout: int = 30,
    ) -> dict | None:
        """Poll for incoming quotes on an RFQ, return best price."""
        deadline = time.time() + timeout
        best: dict | None = None
        best_price = float("inf") if side == TradeSide.BUY else 0.0

        while time.time() < deadline:
            try:
                rfqs = self._client.get_rfqs_sent(id=rfq_id, price=True)
                if rfqs.data and len(rfqs.data) > 0:
                    rfq = rfqs.data[0]
                    quotes = rfq.get("bids", []) if side == TradeSide.BUY else rfq.get("asks", [])
                    for q in quotes:
                        leg_quotes = q.get("leg_quotes", [])
                        if not leg_quotes:
                            continue
                        # For single-leg, price is from the first leg
                        lq = leg_quotes[0]
                        price_str = lq.get("bid") if side == TradeSide.BUY else lq.get("ask")
                        if not price_str:
                            continue
                        price = float(price_str)
                        # Best = lowest ask for BUY, highest bid for SELL
                        if side == TradeSide.BUY and price < best_price:
                            best_price = price
                            best = {**q, "price": price, "parent_quote_id": q.get("parent_quote_id", "")}
                        elif side == TradeSide.SELL and price > best_price:
                            best_price = price
                            best = {**q, "price": price, "parent_quote_id": q.get("parent_quote_id", "")}
                    if best:
                        return best
            except Exception as e:
                logger.debug("Variational quote poll error: {}", e)

            await asyncio.sleep(2)

        return best

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[dict]:
        """Fetch open positions from Variational."""
        if self._client is None:
            return []
        try:
            positions = self._client.get_portfolio_positions()
            result = []
            for p in positions.data:
                qty = float(p.get("qty", 0))
                if abs(qty) < 1e-12:
                    continue
                result.append({
                    "symbol": p.get("instrument", {}).get("underlying", ""),
                    "side": "long" if qty > 0 else "short",
                    "qty": abs(qty),
                    "entry_price": float(p.get("avg_entry_price", 0)),
                    "pool": p.get("pool_location", ""),
                    "counterparty": p.get("counterparty", ""),
                })
            return result
        except Exception as e:
            logger.error("Variational get_positions failed: {}", e)
            return []

    async def get_portfolio_summary(self) -> dict:
        """Fetch portfolio summary (balance, PnL, delta)."""
        if self._client is None:
            return {}
        try:
            summary = self._client.get_portfolio_summary()
            if summary.data:
                return {
                    "balance": float(summary.data.get("sum_balance", 0)),
                    "unrealized_pnl": float(summary.data.get("sum_upnl", 0)),
                    "notional": float(summary.data.get("sum_notional", 0)),
                    "delta": float(summary.data.get("sum_delta", 0)),
                }
            return {}
        except Exception as e:
            logger.error("Variational get_portfolio_summary failed: {}", e)
            return {}

    # ------------------------------------------------------------------
    # Signal handler (matches CEXExecutor pattern)
    # ------------------------------------------------------------------

    async def _handle_signal(self, payload: Any) -> None:
        """Handle SIGNAL events from the event bus."""
        signal: TradingSignal = payload
        if signal.exchange != "variational":
            return

        approved, reason, size, pos = await self.risk_manager.approve_and_open(signal)
        if not approved:
            logger.debug("Variational signal rejected for {}: {}", signal.symbol, reason)
            return

        result = await self.execute_signal(signal, size)
        if result is None:
            # Execution failed — release the reserved position
            await self.risk_manager.close_position("variational", signal.symbol, signal.price)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._init_client()
        if self._client is None:
            logger.warning("Variational executor not started — client unavailable")
            return

        self._running = True
        self.event_bus.subscribe("SIGNAL", self._handle_signal)
        logger.info("Variational DEX executor started")

    async def stop(self) -> None:
        self._running = False
        logger.info("Variational DEX executor stopped")
