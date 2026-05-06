"""
Integration tests — exchange-side SL/TP protection.

Proves:
1. After entry fill, STOP_MARKET + TAKE_PROFIT_MARKET exist exchange-side
2. reduceOnly=True in one-way mode; positionSide=LONG/SHORT in hedge mode
3. Tick-size & step-size rounding
4. OCO: SL fill cancels TP; TP fill cancels SL
5. Partial fill adjusts SL/TP quantity
6. Crash-safety: after entry + bot termination, SL survives in exchange state
7. Reconciliation on restart detects SL/TP and re-registers them
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import Config
from core.event_bus import EventBus
from execution.exchange_order_placer import (
    ExchangeOrderPlacer,
    ProtectiveOrders,
    _round_to_precision,
)
from execution.reconciliation import StartupReconciler
from execution.risk_manager import RiskManager, Position


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config() -> Config:
    cfg = Config.__new__(Config)
    cfg._data = {"risk": {"initial_equity": 100_000}, "exchanges": {}, "system": {"paper_mode": True}}
    return cfg


def _make_risk_manager(config: Config | None = None, bus: EventBus | None = None) -> RiskManager:
    return RiskManager(config or _make_config(), bus or EventBus())


def _mock_client_with_markets() -> AsyncMock:
    """Build a mock ccxt client with realistic market info for BTC/USDT:USDT."""
    client = AsyncMock()
    client.markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "precision": {
                "price": 0.10,       # tick size = $0.10
                "amount": 0.001,     # step size = 0.001 BTC
            },
            "limits": {
                "amount": {"min": 0.001},
                "price": {"min": 0.10},
            },
        },
        "ETH/USDT:USDT": {
            "id": "ETHUSDT",
            "symbol": "ETH/USDT:USDT",
            "precision": {
                "price": 0.01,       # tick size = $0.01
                "amount": 0.01,      # step size = 0.01 ETH
            },
        },
    }
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TICK SIZE & STEP SIZE ROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrecisionRounding:
    """Verify _round_to_precision and placer rounding helpers."""

    def test_round_price_to_tick(self) -> None:
        assert _round_to_precision(50123.456, 0.10) == 50123.4
        assert _round_to_precision(3456.789, 0.01) == 3456.78
        assert _round_to_precision(1.23456, 0.001) == 1.234

    def test_round_quantity_to_step(self) -> None:
        assert _round_to_precision(0.12345, 0.001) == 0.123
        assert _round_to_precision(1.999, 0.01) == 1.99
        assert _round_to_precision(100.0, 1.0) == 100.0

    def test_floor_rounding_never_exceeds_input(self) -> None:
        """Floor rounding must never return a value larger than input."""
        for val, prec in [(0.999, 0.01), (50000.999, 0.10), (0.1234, 0.001)]:
            rounded = _round_to_precision(val, prec)
            assert rounded <= val, f"rounded={rounded} > original={val}"

    def test_placer_round_price_uses_market_info(self) -> None:
        client = _mock_client_with_markets()
        placer = ExchangeOrderPlacer(client=client)
        assert placer.round_price("BTC/USDT:USDT", 49123.456) == 49123.4
        assert placer.round_price("ETH/USDT:USDT", 3456.789) == 3456.78

    def test_placer_round_quantity_uses_market_info(self) -> None:
        client = _mock_client_with_markets()
        placer = ExchangeOrderPlacer(client=client)
        assert placer.round_quantity("BTC/USDT:USDT", 0.12345) == 0.123
        assert placer.round_quantity("ETH/USDT:USDT", 1.999) == 1.99

    def test_placer_round_quantity_handles_decimal_precision_digits(self) -> None:
        client = AsyncMock()
        client.markets = {
            "ETH/USDT:USDT": {
                "id": "ETHUSDT",
                "symbol": "ETH/USDT:USDT",
                "precision": {
                    "price": 2,
                    "amount": 3,
                },
                "limits": {
                    "amount": {"min": 0.001},
                },
            },
        }
        placer = ExchangeOrderPlacer(client=client)
        assert placer.round_quantity("ETH/USDT:USDT", 0.3183) == 0.318

    def test_unknown_symbol_returns_raw_value(self) -> None:
        client = _mock_client_with_markets()
        placer = ExchangeOrderPlacer(client=client)
        assert placer.round_price("UNKNOWN/USDT", 123.456789) == 123.456789
        assert placer.round_quantity("UNKNOWN/USDT", 0.123456) == 0.123456


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONE-WAY MODE: reduceOnly=True on SL/TP
# ═══════════════════════════════════════════════════════════════════════════════

class TestOneWayModeReduceOnly:
    """In one-way (non-hedge) mode, SL and TP must carry reduceOnly=True."""

    @pytest.mark.asyncio
    async def test_sl_has_reduce_only_true(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(return_value={"id": "sl-1", "status": "NEW"})
        placer = ExchangeOrderPlacer(client=client, hedge_mode=False)

        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        # First create_order call = SL
        sl_call = client.create_order.call_args_list[0]
        params = sl_call.kwargs.get("params", sl_call[1].get("params", {}))
        assert params.get("reduceOnly") is True
        assert "positionSide" not in params
        assert "type" not in params

    @pytest.mark.asyncio
    async def test_tp_has_reduce_only_true(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-1", "status": "NEW"},
            {"id": "tp-1", "status": "NEW"},
        ])
        placer = ExchangeOrderPlacer(client=client, hedge_mode=False)

        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        # Second create_order call = TP
        tp_call = client.create_order.call_args_list[1]
        params = tp_call.kwargs.get("params", tp_call[1].get("params", {}))
        assert params.get("reduceOnly") is True
        assert "positionSide" not in params


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HEDGE MODE: positionSide=LONG/SHORT on SL/TP
# ═══════════════════════════════════════════════════════════════════════════════

class TestHedgeModePositionSide:
    """In hedge mode, SL/TP must carry positionSide=LONG|SHORT instead of reduceOnly."""

    @pytest.mark.asyncio
    async def test_long_position_sl_tp_carry_position_side_long(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-h1", "status": "NEW"},
            {"id": "tp-h1", "status": "NEW"},
        ])
        placer = ExchangeOrderPlacer(client=client, hedge_mode=True)

        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        for call in client.create_order.call_args_list:
            params = call.kwargs.get("params", call[1].get("params", {}))
            assert params.get("positionSide") == "LONG"
            assert "reduceOnly" not in params

    @pytest.mark.asyncio
    async def test_short_position_sl_tp_carry_position_side_short(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-h2", "status": "NEW"},
            {"id": "tp-h2", "status": "NEW"},
        ])
        placer = ExchangeOrderPlacer(client=client, hedge_mode=True)

        await placer.place_protective_orders(
            symbol="ETH/USDT:USDT", direction="short",
            quantity=1.0, entry_price=3000, sl_price=3100, tp_price=2800,
        )

        for call in client.create_order.call_args_list:
            params = call.kwargs.get("params", call[1].get("params", {}))
            assert params.get("positionSide") == "SHORT"
            assert "reduceOnly" not in params

    @pytest.mark.asyncio
    async def test_hedge_mode_adjust_quantity_carries_position_side(self) -> None:
        """When adjusting SL/TP for partial fills in hedge mode, positionSide is preserved."""
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-adj", "status": "NEW"},
            {"id": "tp-adj", "status": "NEW"},
            {"id": "sl-adj2", "status": "NEW"},  # replacement SL
            {"id": "tp-adj2", "status": "NEW"},  # replacement TP
        ])
        client.cancel_order = AsyncMock()

        placer = ExchangeOrderPlacer(client=client, hedge_mode=True)
        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        await placer.adjust_quantity("BTC/USDT:USDT", 0.05)

        # The replacement orders (calls 2,3) should also have positionSide=LONG
        for call in client.create_order.call_args_list[2:]:
            params = call.kwargs.get("params", call[1].get("params", {}))
            assert params.get("positionSide") == "LONG"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ROUNDED PRICES SENT TO EXCHANGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoundedPricesSentToExchange:
    """Verify that the actual stopPrice values sent to the exchange are rounded."""

    @pytest.mark.asyncio
    async def test_sl_tp_prices_are_rounded_to_tick_size(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-r", "status": "NEW"},
            {"id": "tp-r", "status": "NEW"},
        ])
        placer = ExchangeOrderPlacer(client=client)

        # Intentionally pass prices with too many decimals
        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.12345,        # should round to 0.123 (step=0.001)
            entry_price=50000,
            sl_price=49123.456,      # should round to 49123.4 (tick=0.10)
            tp_price=53456.789,      # should round to 53456.7 (tick=0.10)
        )

        sl_call = client.create_order.call_args_list[0]
        tp_call = client.create_order.call_args_list[1]

        sl_params = sl_call.kwargs.get("params", sl_call[1].get("params", {}))
        tp_params = tp_call.kwargs.get("params", tp_call[1].get("params", {}))

        assert sl_params["stopPrice"] == 49123.4
        assert tp_params["stopPrice"] == 53456.7

        # Quantity rounded to step size
        assert sl_call.kwargs.get("amount", sl_call[1].get("amount")) == 0.123
        assert tp_call.kwargs.get("amount", tp_call[1].get("amount")) == 0.123


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OCO BEHAVIOUR — full flow from fill to cancellation
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCOFullFlow:
    """Integration: entry fill → SL/TP → one triggers → other cancelled."""

    @pytest.mark.asyncio
    async def test_full_flow_sl_triggers_then_tp_cancelled(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-f1", "status": "NEW"},
            {"id": "tp-f1", "status": "NEW"},
        ])
        client.cancel_order = AsyncMock()

        placer = ExchangeOrderPlacer(client=client)

        # Step 1: Entry filled → place SL/TP
        prot = await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )
        assert prot.sl_placed and prot.tp_placed

        # Step 2: SL triggers on exchange
        await placer.handle_sl_filled("BTC/USDT:USDT")

        # Step 3: TP should be cancelled
        client.cancel_order.assert_called_once_with("tp-f1", "BTC/USDT:USDT")

        # Verify state
        p = placer.get_protective("BTC/USDT:USDT")
        assert p is not None
        assert p.sl_filled is True

    @pytest.mark.asyncio
    async def test_full_flow_tp_triggers_then_sl_cancelled(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-f2", "status": "NEW"},
            {"id": "tp-f2", "status": "NEW"},
        ])
        client.cancel_order = AsyncMock()

        placer = ExchangeOrderPlacer(client=client)

        prot = await placer.place_protective_orders(
            symbol="ETH/USDT:USDT", direction="short",
            quantity=1.0, entry_price=3000, sl_price=3100, tp_price=2800,
        )

        await placer.handle_tp_filled("ETH/USDT:USDT")
        client.cancel_order.assert_called_once_with("sl-f2", "ETH/USDT:USDT")

    @pytest.mark.asyncio
    async def test_partial_fill_adjusts_sl_tp_quantity(self) -> None:
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-p1", "status": "NEW"},
            {"id": "tp-p1", "status": "NEW"},
            {"id": "sl-p2", "status": "NEW"},  # adjusted SL
            {"id": "tp-p2", "status": "NEW"},  # adjusted TP
        ])
        client.cancel_order = AsyncMock()

        placer = ExchangeOrderPlacer(client=client)

        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        # Partial fill: now 0.15 BTC filled
        await placer.adjust_quantity("BTC/USDT:USDT", 0.15)

        # Old orders cancelled, new ones created
        assert client.cancel_order.call_count == 2
        assert client.create_order.call_count == 4  # 2 original + 2 replacement

        p = placer.get_protective("BTC/USDT:USDT")
        assert p is not None
        assert p.quantity == 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CRASH SAFETY — SL survives bot termination
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrashSafety:
    """Prove: after entry fill + SL/TP placement, if the bot dies, the stop
    orders remain on the exchange (they are exchange-side, not local)."""

    @pytest.mark.asyncio
    async def test_sl_survives_bot_crash(self) -> None:
        """Simulate: place SL/TP, then destroy the placer.
        The exchange-side orders still exist (verified by checking the mock's
        call log — no cancel was issued)."""
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-crash", "status": "NEW"},
            {"id": "tp-crash", "status": "NEW"},
        ])
        client.cancel_order = AsyncMock()

        placer = ExchangeOrderPlacer(client=client)
        prot = await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        # Bot crashes — placer object destroyed
        del placer

        # Exchange state: SL and TP were placed, never cancelled
        assert client.create_order.call_count == 2
        assert client.cancel_order.call_count == 0
        # The orders exist on exchange with these IDs
        assert prot.sl_order_id == "sl-crash"
        assert prot.tp_order_id == "tp-crash"
        assert prot.sl_placed is True
        assert prot.tp_placed is True

    @pytest.mark.asyncio
    async def test_sl_tp_are_exchange_side_not_local_timers(self) -> None:
        """Prove SL/TP are STOP_MARKET orders, not in-process timers."""
        client = _mock_client_with_markets()
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-type", "status": "NEW"},
            {"id": "tp-type", "status": "NEW"},
        ])
        placer = ExchangeOrderPlacer(client=client)

        await placer.place_protective_orders(
            symbol="BTC/USDT:USDT", direction="long",
            quantity=0.1, entry_price=50000, sl_price=49000, tp_price=53000,
        )

        sl_call = client.create_order.call_args_list[0]
        tp_call = client.create_order.call_args_list[1]

        # Verify exchange-side order types
        assert sl_call.kwargs.get("type") == "STOP_MARKET"
        assert tp_call.kwargs.get("type") == "TAKE_PROFIT_MARKET"

        # Verify they are server-side (sent to client.create_order on exchange)
        assert client.create_order.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RESTART RECONCILIATION — re-registers surviving SL/TP
# ═══════════════════════════════════════════════════════════════════════════════

class TestReconciliationReregistersProtection:
    """After a crash+restart, reconciliation finds existing exchange SL/TP
    and re-registers them in the order placer so OCO logic resumes."""

    @pytest.mark.asyncio
    async def test_reconciliation_finds_existing_sl_tp(self) -> None:
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = AsyncMock()
        client.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 50_000.0}, "free": {"USDT": 45_000.0},
        })
        client.fetch_positions = AsyncMock(return_value=[{
            "symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1,
            "notional": 5000, "entryPrice": 50000, "unrealizedPnl": 100,
            "leverage": 10, "marginType": "cross", "liquidationPrice": 45000,
        }])
        client.fetch_open_orders = AsyncMock(return_value=[
            {
                "id": "sl-recon", "symbol": "BTC/USDT:USDT",
                "type": "stop_market", "side": "sell",
                "reduceOnly": True, "stopPrice": 49000,
                "amount": 0.1, "price": 0, "status": "open",
            },
            {
                "id": "tp-recon", "symbol": "BTC/USDT:USDT",
                "type": "take_profit_market", "side": "sell",
                "reduceOnly": True, "stopPrice": 53000,
                "amount": 0.1, "price": 0, "status": "open",
            },
        ])

        placer = ExchangeOrderPlacer(client=client)
        reconciler = StartupReconciler(config, bus, rm, client, order_placer=placer)
        result = await reconciler.reconcile()

        assert result.success is True
        # Placer should have the SL/TP registered (recovered from exchange)
        prot = placer.get_protective("BTC/USDT:USDT")
        assert prot is not None
        assert prot.sl_order_id == "sl-recon"
        assert prot.tp_order_id == "tp-recon"
        assert prot.sl_placed is True
        assert prot.tp_placed is True
        # No positions flagged as unprotected
        assert len(result.positions_without_sl) == 0

    @pytest.mark.asyncio
    async def test_reconciliation_places_missing_sl(self) -> None:
        """If a position has NO SL on exchange, reconciliation places one."""
        config = _make_config()
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        client = MagicMock()
        client.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 50_000.0}, "free": {"USDT": 45_000.0},
        })
        client.fetch_positions = AsyncMock(return_value=[{
            "symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1,
            "notional": 5000, "entryPrice": 50000, "unrealizedPnl": 0,
            "leverage": 10, "marginType": "cross", "liquidationPrice": 45000,
        }])
        # No open orders → no SL/TP on exchange
        client.fetch_open_orders = AsyncMock(return_value=[])
        # Mock create_order for placing SL/TP
        client.create_order = AsyncMock(side_effect=[
            {"id": "sl-new", "status": "NEW"},
            {"id": "tp-new", "status": "NEW"},
        ])
        client.markets = {}

        placer = ExchangeOrderPlacer(client=client)
        reconciler = StartupReconciler(config, bus, rm, client, order_placer=placer)
        result = await reconciler.reconcile()

        assert result.success is True
        assert "BTC/USDT:USDT" in result.positions_without_sl
        assert any("placed SL/TP" in a for a in result.actions_taken)
        # Placer should now have the newly placed orders
        prot = placer.get_protective("BTC/USDT:USDT")
        assert prot is not None
        assert prot.sl_placed is True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ENTRY → SL/TP INTEGRATION (CEXExecutor flow)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEXExecutorEntryToSLTP:
    """Integration: CEXExecutor._live_execute → entry fill → SL/TP placed."""

    @pytest.mark.asyncio
    async def test_live_execute_places_sl_tp_after_fill(self) -> None:
        from execution.cex_executor import CEXExecutor
        from engine.signal_generator import TradingSignal

        config = _make_config()
        config._data["system"]["paper_mode"] = False
        bus = EventBus()
        rm = _make_risk_manager(config, bus)

        # Build executor with mocks
        executor = CEXExecutor.__new__(CEXExecutor)
        executor.config = config
        executor.event_bus = bus
        executor.risk_manager = rm
        executor.exchange_id = "binance"
        executor._order_manager = None
        executor._running = True
        executor._rate_limiter = AsyncMock()
        executor._rate_limiter.acquire = AsyncMock()
        executor._maker_first = True
        executor._post_only = False
        executor._iceberg_threshold_usd = 0.0
        executor._iceberg_chunks = 1

        # Mock ccxt client
        mock_client = AsyncMock()
        mock_client.create_limit_order = AsyncMock(return_value={
            "id": "entry-001", "status": "closed", "average": 50000.0, "filled": 0.1,
        })
        mock_client.create_order = AsyncMock(side_effect=[
            {"id": "sl-exe", "status": "NEW"},
            {"id": "tp-exe", "status": "NEW"},
        ])
        mock_client.markets = {
            "BTC/USDT:USDT": {
                "precision": {"price": 0.10, "amount": 0.001},
            }
        }
        executor._client = mock_client

        # Mock order placer
        placer = ExchangeOrderPlacer(
            client=mock_client, working_type="CONTRACT_PRICE",
        )
        executor._order_placer = placer

        # Patch _wait_for_fill to return the order immediately (already closed)
        async def _fake_wait(signal, order, amount, **kw):
            return order
        executor._wait_for_fill = _fake_wait

        # Create signal
        sig = TradingSignal(
            exchange="binance", symbol="BTC/USDT:USDT",
            direction="long", score=0.9,
            technical_score=0.8, ml_score=0.7, sentiment_score=0.5,
            macro_score=0.5, news_score=0.5, orderbook_score=0.5,
            regime="trending", regime_confidence=0.9,
            price=50000.0, atr=500.0,
            stop_loss=49000.0, take_profit=53000.0,
            timestamp=int(time.time()),
        )

        result = await executor._live_execute(sig, 5000.0)

        assert result is not None
        assert result.status in ("filled", "closed")

        # SL/TP should have been placed
        prot = placer.get_protective("BTC/USDT:USDT")
        assert prot is not None
        assert prot.sl_placed is True
        assert prot.tp_placed is True
        assert prot.sl_order_id == "sl-exe"
        assert prot.tp_order_id == "tp-exe"
