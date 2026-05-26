"""SQLite persistence layer — zero-dependency local fallback when PostgreSQL is unavailable."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    time_ns     INTEGER NOT NULL,
    exchange    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    price       REAL NOT NULL,
    volume      REAL NOT NULL,
    side        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS candles (
    time_ns     INTEGER NOT NULL,
    exchange    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    num_trades  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    time_ns         INTEGER NOT NULL,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    score           REAL NOT NULL,
    price           REAL NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    regime          TEXT,
    is_paper        INTEGER DEFAULT 1,
    reasons         TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    size            REAL NOT NULL,
    pnl             REAL,
    pnl_pct         REAL,
    open_time_ns    INTEGER NOT NULL,
    close_time_ns   INTEGER,
    is_paper        INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    client_order_id TEXT,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    quantity        REAL NOT NULL,
    price           REAL,
    status          TEXT NOT NULL,
    filled_quantity REAL DEFAULT 0,
    avg_fill_price  REAL DEFAULT 0,
    created_ns      INTEGER NOT NULL,
    updated_ns      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    time_ns     INTEGER NOT NULL,
    equity      REAL NOT NULL,
    drawdown    REAL DEFAULT 0,
    daily_pnl   REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_ns  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id              TEXT PRIMARY KEY,
    exchange        TEXT NOT NULL DEFAULT 'binance',
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    price           REAL NOT NULL,
    quantity        REAL NOT NULL,
    notional        REAL NOT NULL DEFAULT 0,
    score           REAL NOT NULL DEFAULT 0,
    stop_loss       REAL DEFAULT 0,
    take_profit     REAL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'OPEN',
    timestamp       INTEGER NOT NULL,
    realized_pnl    REAL DEFAULT 0,
    remaining_qty   REAL DEFAULT 0,
    reasons         TEXT DEFAULT '[]',
    metadata        TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_ns  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_time ON ticks(time_ns);
CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(time_ns, exchange, symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(time_ns);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(close_time_ns);
CREATE INDEX IF NOT EXISTS idx_equity_time ON equity_curve(time_ns);
CREATE INDEX IF NOT EXISTS idx_paper_trades_ts ON paper_trades(timestamp);
"""

CURRENT_SCHEMA_VERSION = 2


class SQLiteStore:
    """Thread-safe SQLite persistence for local/dev/backtest usage."""

    def __init__(self, db_path: str | Path = "data/neural_trader.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._connect()
        self._migrate()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info("SQLite store opened: {}", self._path)

    def _migrate(self) -> None:
        assert self._conn is not None
        with self._lock:
            self._conn.executescript(_SCHEMA)
            cur = self._conn.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            current = row[0] if row and row[0] else 0
            if current < CURRENT_SCHEMA_VERSION:
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version(version, applied_ns) VALUES(?, ?)",
                    (CURRENT_SCHEMA_VERSION, time.time_ns()),
                )
                self._conn.commit()
                logger.info("SQLite schema at version {}", CURRENT_SCHEMA_VERSION)

    def _now_ns(self) -> int:
        return time.time_ns()

    # ── Tick persistence ──────────────────────────────────────────────────
    def insert_tick(self, exchange: str, symbol: str, price: float, volume: float, side: str = "") -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO ticks(time_ns, exchange, symbol, price, volume, side) VALUES(?, ?, ?, ?, ?, ?)",
                (self._now_ns(), exchange, symbol, price, volume, side),
            )
            self._conn.commit()

    # ── Candle persistence ────────────────────────────────────────────────
    def insert_candle(
        self, exchange: str, symbol: str, timeframe: str,
        o: float, h: float, l: float, c: float, v: float, ts_ns: int = 0,
    ) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO candles(time_ns, exchange, symbol, timeframe, open, high, low, close, volume) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts_ns or self._now_ns(), exchange, symbol, timeframe, o, h, l, c, v),
            )
            self._conn.commit()

    # ── Signal persistence ────────────────────────────────────────────────
    def insert_signal(
        self, exchange: str, symbol: str, direction: str, score: float,
        price: float, sl: float, tp: float, regime: str = "",
        is_paper: bool = True, reasons: list[str] | None = None,
    ) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO signals(time_ns, exchange, symbol, direction, score, price, "
                "stop_loss, take_profit, regime, is_paper, reasons) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._now_ns(), exchange, symbol, direction, score, price, sl, tp,
                 regime, int(is_paper), json.dumps(reasons or [])),
            )
            self._conn.commit()

    # ── Position persistence ──────────────────────────────────────────────
    def insert_position(
        self, exchange: str, symbol: str, direction: str,
        entry_price: float, size: float, is_paper: bool = True,
    ) -> int:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "INSERT INTO positions(exchange, symbol, direction, entry_price, size, "
                "open_time_ns, is_paper) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (exchange, symbol, direction, entry_price, size, self._now_ns(), int(is_paper)),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def close_position(self, pos_id: int, exit_price: float, pnl: float, pnl_pct: float) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "UPDATE positions SET exit_price=?, pnl=?, pnl_pct=?, close_time_ns=? WHERE id=?",
                (exit_price, pnl, pnl_pct, self._now_ns(), pos_id),
            )
            self._conn.commit()

    def archive_paper_open_position(self, pos_id: int, reason: str = "manual_ledger_cleanup") -> dict[str, Any]:
        """Mark a paper-only open ledger row closed without deleting audit data."""
        with self._lock:
            assert self._conn is not None
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute("SELECT * FROM positions WHERE id=?", (int(pos_id),))
            row_obj = cur.fetchone()
            self._conn.row_factory = None
            if row_obj is None:
                return {"success": False, "id": int(pos_id), "reason": "position_not_found"}

            row = dict(row_obj)
            if row.get("close_time_ns") is not None:
                return {"success": False, "id": int(pos_id), "reason": "position_already_closed"}
            if int(row.get("is_paper", 0) or 0) != 1:
                return {"success": False, "id": int(pos_id), "reason": "live_row_refused"}

            now_ns = self._now_ns()
            exit_price = float(row.get("entry_price", 0.0) or 0.0)
            self._conn.execute(
                "UPDATE positions SET exit_price=?, pnl=?, pnl_pct=?, close_time_ns=? "
                "WHERE id=? AND is_paper=1 AND close_time_ns IS NULL",
                (exit_price, 0.0, 0.0, now_ns, int(pos_id)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO risk_state(key, value, updated_ns) VALUES(?, ?, ?)",
                (
                    f"ledger_cleanup:{int(pos_id)}:{now_ns}",
                    json.dumps({
                        "action": "archive_paper_open_position",
                        "reason": reason,
                        "position_id": int(pos_id),
                        "exchange": row.get("exchange"),
                        "symbol": row.get("symbol"),
                        "direction": row.get("direction"),
                        "entry_price": exit_price,
                    }),
                    now_ns,
                ),
            )
            self._conn.commit()
            return {
                "success": True,
                "id": int(pos_id),
                "reason": reason,
                "exchange": row.get("exchange"),
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "exit_price": exit_price,
                "close_time_ns": now_ns,
            }

    # ── Equity curve ──────────────────────────────────────────────────────
    def record_equity(self, equity: float, drawdown: float = 0.0, daily_pnl: float = 0.0) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO equity_curve(time_ns, equity, drawdown, daily_pnl) VALUES(?, ?, ?, ?)",
                (self._now_ns(), equity, drawdown, daily_pnl),
            )
            self._conn.commit()

    # ── Risk state ────────────────────────────────────────────────────────
    def save_risk_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            assert self._conn is not None
            now = self._now_ns()
            for key, value in state.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO risk_state(key, value, updated_ns) VALUES(?, ?, ?)",
                    (key, json.dumps(value), now),
                )
            self._conn.commit()

    def load_risk_state(self) -> dict[str, Any]:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute("SELECT key, value FROM risk_state")
            return {row[0]: json.loads(row[1]) for row in cur.fetchall()}

    # ── Order persistence ─────────────────────────────────────────────────
    def upsert_order(self, order: dict[str, Any]) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR REPLACE INTO orders(order_id, client_order_id, exchange, symbol, "
                "side, order_type, quantity, price, status, filled_quantity, avg_fill_price, "
                "created_ns, updated_ns) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order["order_id"], order.get("client_order_id", ""),
                    order["exchange"], order["symbol"], order["side"], order["order_type"],
                    order["quantity"], order.get("price", 0.0), order["status"],
                    order.get("filled_quantity", 0.0), order.get("avg_fill_price", 0.0),
                    order.get("created_ns", self._now_ns()), self._now_ns(),
                ),
            )
            self._conn.commit()

    # ── Paper trade persistence ─────────────────────────────────────────
    def upsert_paper_trade(self, trade: dict[str, Any]) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR REPLACE INTO paper_trades(id, exchange, symbol, direction, price, "
                "quantity, notional, score, stop_loss, take_profit, status, timestamp, "
                "realized_pnl, remaining_qty, reasons, metadata) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade["id"], trade.get("exchange", "binance"), trade["symbol"],
                    trade["direction"], trade["price"], trade["quantity"],
                    trade.get("notional", 0), trade.get("score", 0),
                    trade.get("stop_loss", 0), trade.get("take_profit", 0),
                    trade.get("status", "OPEN"), trade.get("timestamp", int(time.time())),
                    trade.get("realized_pnl", 0), trade.get("remaining_qty", trade["quantity"]),
                    json.dumps(trade.get("reasons", [])),
                    json.dumps({k: v for k, v in trade.items()
                                if k in ("tp_tiers", "supertrend_trail", "tp3_close_pct", "partial_fills")}),
                ),
            )
            self._conn.commit()

    def get_paper_trades(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT ?", (limit,),
        )
        for r in rows:
            r["reasons"] = json.loads(r.get("reasons", "[]"))
            meta = json.loads(r.get("metadata", "{}"))
            r.update(meta)
        return rows

    def get_open_paper_trades(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM paper_trades WHERE status = 'OPEN'")
        for r in rows:
            r["reasons"] = json.loads(r.get("reasons", "[]"))
            meta = json.loads(r.get("metadata", "{}"))
            r.update(meta)
        return rows

    # ── Candle batch insert ───────────────────────────────────────────────
    def insert_candles_batch(
        self, exchange: str, symbol: str, timeframe: str,
        candles: list[dict[str, Any]],
    ) -> int:
        """Batch insert candles. Each candle dict has: time, open, high, low, close, volume."""
        if not candles:
            return 0
        with self._lock:
            assert self._conn is not None
            rows = []
            for c in candles:
                ts = c.get("time_ns", 0)
                if not ts and "time" in c:
                    # Convert ISO or epoch to nanoseconds
                    t = c["time"]
                    if isinstance(t, str):
                        import datetime
                        dt = datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))
                        ts = int(dt.timestamp() * 1e9)
                    elif isinstance(t, (int, float)):
                        ts = int(t * 1e9) if t < 1e12 else int(t * 1e6) if t < 1e15 else int(t)
                rows.append((ts, exchange, symbol, timeframe,
                             c["open"], c["high"], c["low"], c["close"], c.get("volume", 0)))
            self._conn.executemany(
                "INSERT OR IGNORE INTO candles(time_ns, exchange, symbol, timeframe, open, high, low, close, volume) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", rows,
            )
            self._conn.commit()
            return len(rows)

    def get_candles(
        self, exchange: str, symbol: str, timeframe: str, limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.query(
            "SELECT time_ns, open, high, low, close, volume FROM candles "
            "WHERE exchange=? AND symbol=? AND timeframe=? ORDER BY time_ns DESC LIMIT ?",
            (exchange, symbol, timeframe, limit),
        )

    # ── Queries ───────────────────────────────────────────────────────────
    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            assert self._conn is not None
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute(sql, params)
            rows = [dict(row) for row in cur.fetchall()]
            self._conn.row_factory = None
            return rows

    def get_equity_curve(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.query(
            "SELECT time_ns, equity, drawdown, daily_pnl FROM equity_curve "
            "ORDER BY time_ns DESC LIMIT ?", (limit,),
        )

    def get_open_positions(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM positions WHERE close_time_ns IS NULL")

    def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM positions WHERE close_time_ns IS NOT NULL "
            "ORDER BY close_time_ns DESC LIMIT ?", (limit,),
        )

    # ── Retention ─────────────────────────────────────────────────────────
    def apply_retention(self, max_age_days: int = 30) -> int:
        """Delete tick/candle data older than max_age_days."""
        cutoff_ns = self._now_ns() - (max_age_days * 86400 * 10**9)
        deleted = 0
        with self._lock:
            assert self._conn is not None
            for table in ("ticks", "candles"):
                # Table names are a hardcoded literal tuple, not user input
                cur = self._conn.execute(f"DELETE FROM {table} WHERE time_ns < ?", (cutoff_ns,))  # noqa: S608  # nosec B608
                deleted += cur.rowcount
            self._conn.commit()
        if deleted:
            logger.info("Retention: deleted {} old rows", deleted)
        return deleted

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite store closed")

    @property
    def available(self) -> bool:
        return self._conn is not None

    @property
    def path(self) -> str:
        return str(self._path)
