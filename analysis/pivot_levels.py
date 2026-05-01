"""Floor pivots + Fibonacci retracement levels (REQ-IND extension).

Two pure functions used by the L4/L5 layers and by the dashboard:

  classic_pivots(prev_high, prev_low, prev_close)
    → {'P', 'R1','R2','R3', 'S1','S2','S3'}

  fibonacci_retracement(swing_high, swing_low)
    → {'level_0','level_236','level_382','level_500','level_618',
       'level_786','level_1000'}

Plus convenience wrappers that take a DataFrame and emit the latest set.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def classic_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Floor pivots from the *previous* bar's H/L/C.

    P  = (H + L + C) / 3
    R1 = 2·P − L          S1 = 2·P − H
    R2 = P + (H − L)      S2 = P − (H − L)
    R3 = H + 2·(P − L)    S3 = L − 2·(H − P)
    """
    h, low, c = float(prev_high), float(prev_low), float(prev_close)
    p = (h + low + c) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - h
    r2 = p + (h - low)
    s2 = p - (h - low)
    r3 = h + 2 * (p - low)
    s3 = low - 2 * (h - p)
    return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def fibonacci_retracement(swing_high: float, swing_low: float) -> dict[str, float]:
    """Standard fib retracement levels between a swing low and high.

    level_0    = swing_low
    level_236  = swing_low + 0.236·(swing_high − swing_low)
    …
    level_1000 = swing_high
    """
    lo, hi = float(swing_low), float(swing_high)
    diff = hi - lo
    return {
        "level_0": lo,
        "level_236": lo + 0.236 * diff,
        "level_382": lo + 0.382 * diff,
        "level_500": lo + 0.500 * diff,
        "level_618": lo + 0.618 * diff,
        "level_786": lo + 0.786 * diff,
        "level_1000": hi,
    }


def pivots_from_df(df: pd.DataFrame, lookback_bar: int = -2) -> dict[str, float] | None:
    """Compute pivots from a DataFrame using the bar at `lookback_bar` (default
    is the bar BEFORE the latest, i.e. the closed prior bar)."""
    if df is None or len(df) < abs(lookback_bar):
        return None
    bar = df.iloc[lookback_bar]
    return classic_pivots(
        prev_high=bar.get("high", 0.0),
        prev_low=bar.get("low", 0.0),
        prev_close=bar.get("close", 0.0),
    )


def fib_from_df(df: pd.DataFrame, lookback: int = 100) -> dict[str, float] | None:
    """Compute Fibonacci retracement using the highest high / lowest low of the
    last `lookback` bars."""
    if df is None or len(df) < 2:
        return None
    seg = df.tail(lookback)
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    if hi <= lo:
        return None
    return fibonacci_retracement(hi, lo)


def nearest_level(price: float, levels: dict[str, float]) -> tuple[str, float, float] | None:
    """Find the level closest to a given price. Returns (name, level_price, distance_pct)."""
    if not levels:
        return None
    name, lvl = min(levels.items(), key=lambda kv: abs(kv[1] - price))
    if price <= 0:
        return (name, lvl, 0.0)
    return (name, lvl, abs(lvl - price) / price * 100.0)


__all__ = [
    "classic_pivots",
    "fibonacci_retracement",
    "pivots_from_df",
    "fib_from_df",
    "nearest_level",
]
