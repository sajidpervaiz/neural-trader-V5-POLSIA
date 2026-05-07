from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from engine.signal_generator import TradingSignal
from execution.cex_executor import CEXExecutor, OrderResult
from execution.order_manager import OrderManager
from execution.risk_manager import RiskManager


class HyperliquidExecutor(CEXExecutor):
    """Hyperliquid perpetuals executor using the official hyperliquid-python-sdk.

    Config block (config/settings.yaml):
        exchanges:
          hyperliquid:
            enabled: true
            private_key: "0x..."          # EVM private key
            wallet_address: "0x..."       # (optional) use a different account address
            vault_address: "0x..."        # (optional) trade from a vault
            testnet: false
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        risk_manager: RiskManager,
        order_manager: OrderManager | None = None,
    ) -> None:
        super().__init__(
            config,
            event_bus,
            risk_manager,
            exchange_id="hyperliquid",
            order_manager=order_manager,
        )
        self._hl_info: Any = None
        self._hl_exchange: Any = None
        self._wallet_address: str = ""

    # ── Initialisation ────────────────────────────────────────────────────

    async def _init_client(self) -> None:
        """Build Info + Exchange from the SDK using the configured private key."""
        if self._hl_exchange is not None:
            return

        cfg = self.config.get_value("exchanges", "hyperliquid") or {}
        if not cfg.get("enabled", False):
            logger.info("Hyperliquid exchange disabled in config — skipping init")
            return

        private_key: str = str(cfg.get("private_key", "")).strip()
        if not private_key:
            logger.warning("Hyperliquid: no private_key configured — executor inactive")
            return

        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            wallet = Account.from_key(private_key)
            base_url = (
                constants.TESTNET_API_URL if cfg.get("testnet") else constants.MAINNET_API_URL
            )

            vault_address: str | None = cfg.get("vault_address") or None
            account_address: str | None = cfg.get("wallet_address") or None

            self._hl_info = Info(base_url, skip_ws=True)
            self._hl_exchange = Exchange(
                wallet,
                base_url,
                vault_address=vault_address,
                account_address=account_address,
            )
            self._wallet_address = account_address or wallet.address

            logger.info(
                "HyperliquidExecutor ready — wallet={} testnet={}",
                self._wallet_address[:10] + "...",
                bool(cfg.get("testnet")),
            )
        except Exception as exc:
            logger.error("HyperliquidExecutor init failed: {}", exc)

    # ── Symbol helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_hl_coin(symbol: str) -> str:
        """Convert 'BTC/USDT:USDT' → 'BTC'."""
        return symbol.split("/")[0]

    # ── Live execution ────────────────────────────────────────────────────

    async def _live_execute(
        self, signal: TradingSignal, size: float, reserved_pos: Any = None
    ) -> OrderResult | None:
        if self._hl_exchange is None:
            logger.warning("HyperliquidExecutor: client not initialised")
            return None

        coin = self._to_hl_coin(signal.symbol)
        is_buy = signal.direction == "long"

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_exchange.market_open(
                    coin,
                    is_buy,
                    round(size, 6),
                    slippage=0.01,  # 1 % max slippage
                ),
            )

            status = result.get("status", "unknown")
            if status != "ok":
                logger.warning("Hyperliquid order returned non-ok status: {}", result)
                return None

            # Parse the filled response
            data = result.get("response", {}).get("data", {})
            statuses = data.get("statuses", [{}])
            first = statuses[0] if statuses else {}
            filled = first.get("filled", {})
            order_id = str(filled.get("oid", int(time.time() * 1000)))
            fill_px = float(filled.get("avgPx", signal.price or 0))
            fill_sz = float(filled.get("totalSz", size))

            logger.info(
                "Hyperliquid {} {} {} @ {:.4f} — oid={}",
                signal.direction.upper(),
                coin,
                fill_sz,
                fill_px,
                order_id,
            )

            result = OrderResult(
                order_id=order_id,
                exchange="hyperliquid",
                symbol=signal.symbol,
                direction=signal.direction,
                price=fill_px,
                quantity=fill_sz,
                status="filled",
                is_paper=False,
                timestamp=int(time.time() * 1000),
            )
            if reserved_pos is not None:
                reserved_pos.current_price = fill_px
                reserved_pos.size = fill_sz
            else:
                await self.risk_manager.open_position(signal, fill_sz * fill_px)
            await self.event_bus.publish("ORDER_FILLED", result)
            return result

        except Exception as exc:
            logger.error("Hyperliquid live execute error: {}", exc)
            return None

    # ── Position close ────────────────────────────────────────────────────

    async def _close_position_live(self, symbol: str, size: float) -> bool:
        if self._hl_exchange is None:
            return False
        coin = self._to_hl_coin(symbol)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_exchange.market_close(coin, sz=size, slippage=0.01),
            )
            ok = result.get("status") == "ok"
            if ok:
                logger.info("Hyperliquid closed {} size={}", coin, size)
            else:
                logger.warning("Hyperliquid close non-ok: {}", result)
            return ok
        except Exception as exc:
            logger.error("Hyperliquid close error: {}", exc)
            return False

    # ── Cancel order ──────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._hl_exchange is None:
            return False
        coin = self._to_hl_coin(symbol)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_exchange.cancel(coin, int(order_id)),
            )
            return result.get("status") == "ok"
        except Exception as exc:
            logger.error("Hyperliquid cancel error: {}", exc)
            return False

    # ── Balance / positions ───────────────────────────────────────────────

    async def get_balance(self) -> dict[str, float]:
        if self._hl_info is None:
            return {}
        try:
            state = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_info.user_state(self._wallet_address),
            )
            margin = state.get("marginSummary", {})
            return {
                "total": float(margin.get("accountValue", 0)),
                "available": float(margin.get("withdrawable", 0)),
            }
        except Exception as exc:
            logger.error("Hyperliquid get_balance error: {}", exc)
            return {}

    async def get_open_positions(self) -> list[dict[str, Any]]:
        if self._hl_info is None:
            return []
        try:
            state = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_info.user_state(self._wallet_address),
            )
            out = []
            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                szi = float(p.get("szi", 0))
                if szi == 0:
                    continue
                out.append({
                    "symbol": p.get("coin", "") + "/USDT:USDT",
                    "side": "long" if szi > 0 else "short",
                    "size": abs(szi),
                    "entry": float(p.get("entryPx", 0)),
                    "pnl": float(p.get("unrealizedPnl", 0)),
                    "leverage": float(p.get("leverage", {}).get("value", 1)),
                })
            return out
        except Exception as exc:
            logger.error("Hyperliquid get_open_positions error: {}", exc)
            return []

    # ── Leverage ──────────────────────────────────────────────────────────

    async def set_leverage(self, symbol: str, leverage: int, *, cross_margin: bool = False) -> bool:
        """Set leverage for a symbol on Hyperliquid."""
        if self._hl_exchange is None:
            logger.warning("HyperliquidExecutor: client not initialised — cannot set leverage")
            return False
        coin = self._to_hl_coin(symbol)
        is_cross = cross_margin
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_exchange.update_leverage(leverage, coin, is_cross),
            )
            ok = result.get("status") == "ok"
            if ok:
                logger.info("Hyperliquid leverage set {} × {} ({})", coin, leverage,
                            "cross" if is_cross else "isolated")
            else:
                logger.warning("Hyperliquid set_leverage non-ok: {}", result)
            return ok
        except Exception as exc:
            logger.error("Hyperliquid set_leverage error: {}", exc)
            return False

    # ── Funding rate ──────────────────────────────────────────────────────

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Return current funding rate info for a symbol."""
        if self._hl_info is None:
            return {}
        coin = self._to_hl_coin(symbol)
        try:
            meta = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_info.meta(),
            )
            universe = meta.get("universe", [])
            for asset in universe:
                if asset.get("name") == coin:
                    funding = asset.get("funding", 0.0)
                    open_interest = asset.get("openInterest", 0.0)
                    return {
                        "symbol": symbol,
                        "coin": coin,
                        "funding_rate": float(funding),
                        "funding_rate_pct": float(funding) * 100,
                        "open_interest": float(open_interest),
                    }
            return {"symbol": symbol, "coin": coin, "funding_rate": 0.0}
        except Exception as exc:
            logger.error("Hyperliquid get_funding_rate error: {}", exc)
            return {}

    # ── Orderbook ─────────────────────────────────────────────────────────

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        if self._hl_info is None:
            return {}
        coin = self._to_hl_coin(symbol)
        try:
            snap = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._hl_info.l2_snapshot(coin),
            )
            levels = snap.get("levels", [[], []])
            bids = [{"price": float(l["px"]), "quantity": float(l["sz"])} for l in levels[0][:depth]]
            asks = [{"price": float(l["px"]), "quantity": float(l["sz"])} for l in levels[1][:depth]]
            mid = (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else 0
            return {
                "bids": bids,
                "asks": asks,
                "mid_price": round(mid, 4),
                "spread": round(asks[0]["price"] - bids[0]["price"], 4) if bids and asks else 0,
            }
        except Exception as exc:
            logger.error("Hyperliquid orderbook error: {}", exc)
            return {}
