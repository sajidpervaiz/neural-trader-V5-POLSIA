from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    import asyncpg
    _ASYNCPG = True
except ImportError:
    _ASYNCPG = False

from core.config import Config


CREATE_TICKS = """
CREATE TABLE IF NOT EXISTS ticks (
    time        TIMESTAMPTZ NOT NULL,
    exchange    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    side        TEXT
);
"""

CREATE_CANDLES = """
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ NOT NULL,
    exchange    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    num_trades  INTEGER DEFAULT 0
);
"""

CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    time            TIMESTAMPTZ NOT NULL,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    score           DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    regime          TEXT,
    is_paper        BOOLEAN DEFAULT TRUE
);
"""

CREATE_FUNDING_RATES = """
CREATE TABLE IF NOT EXISTS funding_rates (
    time            TIMESTAMPTZ NOT NULL,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    funding_rate    DOUBLE PRECISION NOT NULL,
    predicted_rate  DOUBLE PRECISION
);
"""

CREATE_OPEN_INTEREST = """
CREATE TABLE IF NOT EXISTS open_interest (
    time            TIMESTAMPTZ NOT NULL,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    oi_usd          DOUBLE PRECISION NOT NULL,
    oi_contracts    DOUBLE PRECISION NOT NULL,
    oi_change_24h   DOUBLE PRECISION DEFAULT 0
);
"""

CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    id              SERIAL PRIMARY KEY,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    size            DOUBLE PRECISION NOT NULL,
    pnl             DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    open_time       TIMESTAMPTZ NOT NULL,
    close_time      TIMESTAMPTZ,
    is_paper        BOOLEAN DEFAULT TRUE
);
"""

# Partial unique index: only one open position per exchange+symbol at a time
CREATE_POSITIONS_UNIQUE_IDX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_open_unique
    ON positions (exchange, symbol) WHERE close_time IS NULL;
"""

TIMESCALE_HYPERTABLES = [
    "SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);",
    "SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);",
    "SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE);",
    "SELECT create_hypertable('open_interest', 'time', if_not_exists => TRUE);",
]

ALL_DDL = [
    CREATE_TICKS,
    CREATE_CANDLES,
    CREATE_SIGNALS,
    CREATE_FUNDING_RATES,
    CREATE_OPEN_INTEREST,
    CREATE_POSITIONS,
    CREATE_POSITIONS_UNIQUE_IDX,
]


class DBHandler:
    def __init__(self, config: Config, event_bus: Any = None) -> None:
        self.config = config
        self._pool: Any = None
        # Optional event bus for ALERT_CRITICAL on connection loss; if absent,
        # only loguru sees the failure (legacy behaviour).
        self._event_bus = event_bus
        pg = config.get_value("storage", "postgres") or {}
        self._dsn = (
            f"postgresql://{pg.get('user', 'trader')}:{pg.get('password', '')}@"
            f"{pg.get('host', 'localhost')}:{pg.get('port', 5432)}/"
            f"{pg.get('database', 'neural_trader')}"
        )
        self._pool_size = int(pg.get("pool_size", 10))
        self._timescale = bool(pg.get("timescaledb", True))
        self._connected_once = False  # for distinguishing initial fail vs reconnect fail

    async def connect(self, timeout: float = 10.0) -> None:
        if self._pool is not None:
            return  # Already connected — avoid creating a second pool
        if not _ASYNCPG:
            logger.warning("asyncpg not installed — DB disabled")
            return
        try:
            self._pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=2,
                    max_size=self._pool_size,
                    command_timeout=30,
                ),
                timeout=timeout,
            )
            await self._migrate()
            logger.info("Database connected (pool_size={})", self._pool_size)
            if not self._connected_once:
                self._connected_once = True
            else:
                # Reconnect after an outage — surface for audit
                self._emit_alert("ALERT_WARNING", "db_reconnected",
                                 "Database connection RESTORED after outage")
        except Exception as exc:
            # Loud-and-loud: was previously a single warning. Now CRITICAL +
            # ALERT_CRITICAL event so monitoring and audit (when bus is wired)
            # see it. The bot still continues — but the operator knows audit
            # is offline for the duration.
            logger.critical(
                "Database connection FAILED: {} — running without DB. "
                "Audit log offline; trades during this window will NOT be "
                "persisted to PostgreSQL.", exc,
            )
            self._pool = None
            self._emit_alert(
                "ALERT_CRITICAL",
                "db_unavailable" if not self._connected_once else "db_disconnected",
                f"PostgreSQL unreachable: {str(exc)[:160]}",
            )

    def _emit_alert(self, level: str, alert_type: str, message: str) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish_nowait(level, {
                "type": alert_type,
                "error": message,
                "needs_ack": (level == "ALERT_CRITICAL"),
                "ts": int(__import__("time").time()),
            })
        except Exception as exc:
            logger.debug("DBHandler alert emit failed: {}", exc)

    async def _migrate(self) -> None:
        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            for ddl in ALL_DDL:
                await conn.execute(ddl)
            if self._timescale:
                for stmt in TIMESCALE_HYPERTABLES:
                    try:
                        await conn.execute(stmt)
                    except Exception:
                        pass
        logger.info("Database schema up to date")

    async def insert_tick(self, exchange: str, symbol: str, price: float, volume: float, side: str = "") -> None:
        if self._pool is None:
            return
        try:
            import datetime
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO ticks(time, exchange, symbol, price, volume, side) VALUES($1, $2, $3, $4, $5, $6)",
                    datetime.datetime.now(datetime.UTC), exchange, symbol, price, volume, side,
                )
        except Exception as exc:
            logger.debug("insert_tick error: {}", exc)

    async def insert_funding_rate(
        self, exchange: str, symbol: str, rate: float, predicted: float | None = None
    ) -> None:
        if self._pool is None:
            return
        try:
            import datetime
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO funding_rates(time, exchange, symbol, funding_rate, predicted_rate) VALUES($1,$2,$3,$4,$5)",
                    datetime.datetime.now(datetime.UTC), exchange, symbol, rate, predicted,
                )
        except Exception as exc:
            logger.debug("insert_funding_rate error: {}", exc)

    async def insert_open_interest(
        self, exchange: str, symbol: str, oi_usd: float, oi_contracts: float, oi_change: float = 0.0
    ) -> None:
        if self._pool is None:
            return
        try:
            import datetime
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO open_interest(time, exchange, symbol, oi_usd, oi_contracts, oi_change_24h) VALUES($1,$2,$3,$4,$5,$6)",
                    datetime.datetime.now(datetime.UTC), exchange, symbol, oi_usd, oi_contracts, oi_change,
                )
        except Exception as exc:
            logger.debug("insert_open_interest error: {}", exc)

    async def query(self, sql: str, *args: Any) -> list[Any]:
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetch(sql, *args)
        except Exception as exc:
            logger.debug("query error: {}", exc)
            return []

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @property
    def available(self) -> bool:
        return self._pool is not None
