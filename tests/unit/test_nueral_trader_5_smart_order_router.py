from __future__ import annotations

from dataclasses import dataclass

import pytest

from execution.smart_order_router import SmartOrderRouter, Venue
from execution.binance_executor import OrderSide


@dataclass
class _Level:
    price: float
    quantity: float


@dataclass
class _Snapshot:
    bids: list[_Level]
    asks: list[_Level]


class _DummyExecutor:
    def __init__(self, bid: float, ask: float, qty: float) -> None:
        self.bid = bid
        self.ask = ask
        self.qty = qty

    async def get_orderbook_snapshot(self, symbol: str, depth: int = 10) -> _Snapshot:
        _ = (symbol, depth)
        return _Snapshot(
            bids=[_Level(self.bid, self.qty) for _ in range(5)],
            asks=[_Level(self.ask, self.qty) for _ in range(5)],
        )


class _CexClientStub:
    async def fetch_order_book(self, symbol: str, limit: int = 10):
        _ = (symbol, limit)
        return {
            "bids": [[100.0, 25.0], [99.9, 20.0]],
            "asks": [[100.1, 24.0], [100.2, 21.0]],
        }


class _CexStyleExecutor:
    def __init__(self) -> None:
        self._client = None

    async def _init_client(self) -> None:
        self._client = _CexClientStub()


@pytest.mark.asyncio
async def test_route_order_refreshes_scores_lazily() -> None:
    router = SmartOrderRouter(
        binance_executor=_DummyExecutor(bid=100.0, ask=100.1, qty=20.0),
        bybit_executor=None,
        okx_executor=None,
    )

    assert router.venue_scores == {}
    decision = await router.route_order(
        "BTC/USDT",
        OrderSide.BUY,
        quantity=5.0,
        min_score_threshold=0.1,
    )

    assert decision is not None
    assert decision.total_quantity > 0
    assert Venue.BINANCE in router.venue_scores


@pytest.mark.asyncio
async def test_estimated_cost_uses_expected_price_not_liquidity() -> None:
    router = SmartOrderRouter(
        binance_executor=_DummyExecutor(bid=100.0, ask=100.2, qty=100.0),
        bybit_executor=_DummyExecutor(bid=99.9, ask=100.3, qty=100.0),
        okx_executor=None,
    )

    decision = await router.route_order("BTC/USDT", OrderSide.BUY, quantity=2.0, max_venues=1)

    assert decision is not None
    route = decision.routes[0]
    score = router.venue_scores[route.venue]

    expected_slippage = (score.spread / 2) * route.quantity
    expected_fee = route.expected_fill_price * route.quantity * score.fee
    assert route.estimated_cost == pytest.approx(expected_slippage + expected_fee, rel=1e-9)


@pytest.mark.asyncio
async def test_route_order_supports_cex_runtime_executor_without_legacy_snapshot_api() -> None:
    router = SmartOrderRouter(
        binance_executor=_CexStyleExecutor(),
        bybit_executor=None,
        okx_executor=None,
    )

    decision = await router.route_order(
        "BTC/USDT:USDT",
        OrderSide.BUY,
        quantity=3.0,
        min_score_threshold=0.1,
    )

    assert decision is not None
    assert decision.recommended_venue == Venue.BINANCE
