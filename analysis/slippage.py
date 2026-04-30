"""Pre-trade spread + slippage estimator (REQ-EXE-005).

Given an order-book snapshot (sorted bids descending, asks ascending) and a
target fill quantity, walks the book to compute:

  • avg_fill_price — VWAP across consumed levels
  • slippage_bps   — (avg_fill - reference_price) / reference_price × 10_000
                     (signed: positive = adverse for the taker direction)
  • exhausted      — True when the book did not contain enough qty
  • levels_walked  — how many price levels were consumed

`reference_price` is the best opposite-side quote (the price the order
would print at if there were no slippage). `slippage_bps` is signed so
that a buy that walks UP through asks comes out positive, a sell that
walks DOWN through bids comes out positive — both adverse for the taker.

Pure function — no state, no IO. Safe to call from the trading hot path
or from a /api/slippage endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SlippageEstimate:
    side: str  # "buy" / "sell"
    target_qty: float
    filled_qty: float
    avg_fill_price: float
    reference_price: float
    spread_bps: float
    slippage_bps: float
    exhausted: bool
    levels_walked: int

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "target_qty": self.target_qty,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "reference_price": self.reference_price,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "exhausted": self.exhausted,
            "levels_walked": self.levels_walked,
        }


def _spread_bps(best_bid: float, best_ask: float) -> float:
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return 0.0
    return (best_ask - best_bid) / mid * 10_000.0


def estimate_fill(
    side: str,
    target_qty: float,
    bids: Iterable[tuple[float, float]] | None,
    asks: Iterable[tuple[float, float]] | None,
) -> SlippageEstimate:
    """Walk the order book to estimate the average fill price.

    Args:
      side: "buy" walks up the asks; "sell" walks down the bids.
      target_qty: positive quantity in base units.
      bids: best→worst (descending price).
      asks: best→worst (ascending price).
    """
    side_lc = side.lower().strip()
    if side_lc not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    qty = max(0.0, float(target_qty))
    bid_list = sorted(((float(p), float(q)) for p, q in (bids or []) if q > 0), key=lambda x: -x[0])
    ask_list = sorted(((float(p), float(q)) for p, q in (asks or []) if q > 0), key=lambda x: x[0])
    best_bid = bid_list[0][0] if bid_list else 0.0
    best_ask = ask_list[0][0] if ask_list else 0.0
    spread_bps = _spread_bps(best_bid, best_ask)
    levels = ask_list if side_lc == "buy" else bid_list
    reference = best_ask if side_lc == "buy" else best_bid

    if qty == 0 or not levels or reference <= 0:
        return SlippageEstimate(
            side=side_lc, target_qty=qty, filled_qty=0.0,
            avg_fill_price=reference, reference_price=reference,
            spread_bps=spread_bps, slippage_bps=0.0,
            exhausted=(qty > 0 and not levels),
            levels_walked=0,
        )

    remaining = qty
    notional = 0.0
    walked = 0
    for price, available in levels:
        if remaining <= 0:
            break
        take = min(remaining, available)
        notional += price * take
        remaining -= take
        walked += 1
    filled = qty - remaining
    if filled <= 0:
        return SlippageEstimate(
            side=side_lc, target_qty=qty, filled_qty=0.0,
            avg_fill_price=reference, reference_price=reference,
            spread_bps=spread_bps, slippage_bps=0.0,
            exhausted=True, levels_walked=0,
        )
    avg = notional / filled
    if side_lc == "buy":
        slip_bps = (avg - reference) / reference * 10_000.0
    else:
        slip_bps = (reference - avg) / reference * 10_000.0
    return SlippageEstimate(
        side=side_lc, target_qty=qty, filled_qty=filled,
        avg_fill_price=avg, reference_price=reference,
        spread_bps=spread_bps, slippage_bps=slip_bps,
        exhausted=remaining > 1e-12, levels_walked=walked,
    )


__all__ = ["SlippageEstimate", "estimate_fill"]
