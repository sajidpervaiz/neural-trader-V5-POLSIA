"""REQ-EXE-005: pre-trade spread + slippage estimator.

Pure function tests covering: walk-up on a deep book (zero slippage),
walk through multiple levels (positive bps), exhausted book (partial fill
flagged), empty book (graceful), invalid side, and spread sanity.
"""
from __future__ import annotations

import math

import pytest

from analysis.slippage import estimate_fill


# Synthetic book: best ask 100.0 size 1, then 100.5 size 2, then 101 size 5.
ASKS = [(100.0, 1.0), (100.5, 2.0), (101.0, 5.0)]
BIDS = [(99.5, 1.0), (99.0, 2.0), (98.0, 5.0)]


def test_buy_within_top_level_no_slippage() -> None:
    est = estimate_fill("buy", 0.5, BIDS, ASKS)
    assert est.filled_qty == 0.5
    assert est.avg_fill_price == 100.0
    assert est.slippage_bps == 0.0
    assert est.exhausted is False
    assert est.levels_walked == 1


def test_buy_walks_three_levels_slippage_positive() -> None:
    # Take 1 + 2 + 1 = 4 → cost = 100*1 + 100.5*2 + 101*1 = 302
    # avg = 302/4 = 75.5? wait 302/4 = 75.5 ... let me redo
    # 100 + 201 + 101 = 402, / 4 = 100.5
    est = estimate_fill("buy", 4.0, BIDS, ASKS)
    assert est.filled_qty == 4.0
    assert math.isclose(est.avg_fill_price, 100.5, rel_tol=1e-9)
    # slippage = (100.5 - 100) / 100 * 10000 = 50 bps
    assert math.isclose(est.slippage_bps, 50.0, rel_tol=1e-9)
    assert est.exhausted is False
    assert est.levels_walked == 3


def test_sell_walks_bids_slippage_positive() -> None:
    # 1 + 2 + 1 = 4 → 99.5*1 + 99*2 + 98*1 = 99.5 + 198 + 98 = 395.5; /4 = 98.875
    est = estimate_fill("sell", 4.0, BIDS, ASKS)
    assert est.filled_qty == 4.0
    assert math.isclose(est.avg_fill_price, 98.875, rel_tol=1e-9)
    # ref = best bid 99.5 → slip = (99.5 - 98.875)/99.5 * 10_000 ≈ 62.81 bps
    assert math.isclose(est.slippage_bps, (99.5 - 98.875) / 99.5 * 10_000, rel_tol=1e-9)


def test_exhausted_book_flagged() -> None:
    est = estimate_fill("buy", 100.0, BIDS, ASKS)
    # Total ask qty = 1 + 2 + 5 = 8 — anything beyond = exhausted
    assert est.exhausted is True
    assert est.filled_qty == 8.0
    assert est.levels_walked == 3


def test_empty_book() -> None:
    est = estimate_fill("buy", 1.0, [], [])
    assert est.filled_qty == 0.0
    assert est.exhausted is True
    assert est.slippage_bps == 0.0
    assert est.spread_bps == 0.0


def test_zero_qty_returns_no_walk() -> None:
    est = estimate_fill("buy", 0.0, BIDS, ASKS)
    assert est.filled_qty == 0.0
    assert est.levels_walked == 0
    assert est.exhausted is False
    assert est.avg_fill_price == 100.0  # reference == best ask


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError):
        estimate_fill("long", 1.0, BIDS, ASKS)


def test_spread_bps_computed() -> None:
    est = estimate_fill("buy", 0.1, BIDS, ASKS)
    # mid = (99.5 + 100) / 2 = 99.75; spread bps = (100 - 99.5)/99.75 * 10_000 ≈ 50.13
    assert math.isclose(est.spread_bps, 0.5 / 99.75 * 10_000, rel_tol=1e-9)


def test_unsorted_book_normalised() -> None:
    # Pass the book intentionally out of order — function should sort it.
    asks_unsorted = [(101.0, 5.0), (100.0, 1.0), (100.5, 2.0)]
    est = estimate_fill("buy", 1.0, BIDS, asks_unsorted)
    assert est.avg_fill_price == 100.0
    assert est.slippage_bps == 0.0
