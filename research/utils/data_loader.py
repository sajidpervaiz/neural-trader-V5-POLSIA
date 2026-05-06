"""Research utilities — database data loading for Jupyter notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import asyncpg
    import asyncio
    _ASYNCPG = True
except ImportError:
    _ASYNCPG = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _get_dsn() -> str:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.config import Config
    cfg = Config.get()
    pg = cfg.get_value("storage", "postgres") or {}
    return (
        f"postgresql://{pg.get('user', 'trader')}:{pg.get('password', '')}@"
        f"{pg.get('host', 'localhost')}:{pg.get('port', 5432)}/"
        f"{pg.get('database', 'neural_trader')}"
    )


async def _async_query(sql: str, *args: Any) -> list[Any]:
    if not _ASYNCPG:
        return []
    conn = await asyncpg.connect(dsn=_get_dsn())
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


def _run(coro: Any) -> Any:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def load_candles(
    exchange: str,
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    """Load OHLCV candles from the database into a DataFrame."""
    conditions = ["exchange = $1", "symbol = $2", "timeframe = $3"]
    params: list[Any] = [exchange, symbol, timeframe]
    idx = 4
    if start:
        conditions.append(f"time >= ${idx}")
        params.append(start)
        idx += 1
    if end:
        conditions.append(f"time <= ${idx}")
        params.append(end)
        idx += 1

    where = " AND ".join(conditions)
    # nosec B608 — `where` built from hardcoded "col = $N" fragments; limit int-coerced; values bound via $N
    sql = f"""
    SELECT time, open, high, low, close, volume, num_trades
    FROM candles
    WHERE {where}
    ORDER BY time DESC
    LIMIT {int(limit)}
    """  # nosec B608
    rows = _run(_async_query(sql, *params))
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "num_trades"])
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "num_trades"])
    df = df.set_index("time").sort_index()
    return df


def load_funding_rates(
    symbol: str | None = None,
    exchange: str | None = None,
    limit: int = 1_000,
) -> pd.DataFrame:
    """Load funding rate history from the database."""
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1
    if exchange:
        conditions.append(f"exchange = ${idx}")
        params.append(exchange)
        idx += 1
    if symbol:
        conditions.append(f"symbol = ${idx}")
        params.append(symbol)
        idx += 1
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # nosec B608 — `where` built from hardcoded "col = $N" fragments; limit int-coerced; values bound via $N
    sql = f"""
    SELECT time, exchange, symbol, funding_rate, predicted_rate
    FROM funding_rates
    {where}
    ORDER BY time DESC
    LIMIT {int(limit)}
    """  # nosec B608
    rows = _run(_async_query(sql, *params))
    if not rows:
        return pd.DataFrame(columns=["time", "exchange", "symbol", "funding_rate", "predicted_rate"])
    return pd.DataFrame(rows, columns=["time", "exchange", "symbol", "funding_rate", "predicted_rate"]).set_index("time").sort_index()


def load_open_interest(
    symbol: str | None = None,
    exchange: str | None = None,
    limit: int = 1_000,
) -> pd.DataFrame:
    """Load open interest history from the database."""
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1
    if exchange:
        conditions.append(f"exchange = ${idx}")
        params.append(exchange)
        idx += 1
    if symbol:
        conditions.append(f"symbol = ${idx}")
        params.append(symbol)
        idx += 1
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # nosec B608 — `where` built from hardcoded "col = $N" fragments; limit int-coerced; values bound via $N
    sql = f"""
    SELECT time, exchange, symbol, oi_usd, oi_contracts, oi_change_24h
    FROM open_interest
    {where}
    ORDER BY time DESC
    LIMIT {int(limit)}
    """  # nosec B608
    rows = _run(_async_query(sql, *params))
    if not rows:
        return pd.DataFrame(columns=["time", "exchange", "symbol", "oi_usd", "oi_contracts", "oi_change_24h"])
    return pd.DataFrame(rows, columns=["time", "exchange", "symbol", "oi_usd", "oi_contracts", "oi_change_24h"]).set_index("time").sort_index()
