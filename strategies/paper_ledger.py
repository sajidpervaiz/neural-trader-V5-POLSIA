"""Paper trade ledger — append-only JSONL with full exit logic."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PaperTrade:
    trade_id: str
    signal_uid: str
    opened_at: str
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    sl_pct: float
    tp_pct: float
    timeout_minutes: int
    margin_usd: float
    leverage: int
    notional_usd: float
    qty: float
    headline: str
    source: str
    news_confidence: int
    phase2_verdict: str
    phase2_score: int
    phase2_reasoning: str
    plan_confidence: int
    plan_reasoning: str
    strategy: str = "v3"
    status: str = "OPEN"
    be_activated: bool = False
    trailing_activated: bool = False
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: str | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    @property
    def opened_ts(self) -> float:
        try:
            return datetime.fromisoformat(self.opened_at).timestamp()
        except (ValueError, TypeError):
            return 0.0

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.opened_ts) / 60.0


# ── Exit logic constants ─────────────────────────────────────────────────────

MAX_HOLD_MIN = 240
FLAT_TIMEOUT_PCT = 0.30
EARLY_EXIT_ENABLED = False
EARLY_EXIT_ADVERSE_PCT = 0.20
BE_ACTIVATION_PCT = 50.0
TRAILING_ACTIVATION_PCT = 75.0
TRAILING_LOCK_RATIO = 0.50


def check_exits(trade: PaperTrade, current_price: float, move_5m_pct: float | None = None) -> tuple[str | None, float | None]:
    """Run all exit checks. Returns (exit_reason, exit_price) or (None, None)."""
    if trade.status != "OPEN":
        return None, None

    age = trade.age_minutes
    entry = trade.entry_price
    sl = trade.sl_price
    tp = trade.tp_price
    is_long = trade.is_long

    # (v) SL/TP hit
    if is_long:
        if current_price <= sl:
            reason = "BE" if trade.be_activated else "SL"
            return reason, current_price
        if current_price >= tp:
            return "TP", current_price
    else:
        if current_price >= sl:
            reason = "BE" if trade.be_activated else "SL"
            return reason, current_price
        if current_price <= tp:
            return "TP", current_price

    # (i) Hard timeout
    if age >= MAX_HOLD_MIN:
        return "HARD-TIMEOUT", current_price

    # (ii) Soft timeout (flat check)
    if age >= trade.timeout_minutes:
        move_pct = abs(current_price - entry) / entry * 100
        if move_pct <= FLAT_TIMEOUT_PCT:
            return "FLAT-TIMEOUT", current_price

    # (iii) Early exit momentum reversal (opt-in)
    if EARLY_EXIT_ENABLED and move_5m_pct is not None and age <= 15:
        in_profit = (is_long and current_price > entry) or (not is_long and current_price < entry)
        adverse = (
            (is_long and move_5m_pct <= -EARLY_EXIT_ADVERSE_PCT)
            or (not is_long and move_5m_pct >= EARLY_EXIT_ADVERSE_PCT)
        )
        if adverse and not in_profit:
            return "EARLY-EXIT-MOMENTUM", current_price

    return None, None


def update_protective_stops(trade: PaperTrade, current_price: float) -> None:
    """Update breakeven + trailing stop in-place."""
    entry = trade.entry_price
    tp = trade.tp_price
    is_long = trade.is_long

    tp_distance = abs(tp - entry)
    if tp_distance == 0:
        return

    if is_long:
        profit_distance = current_price - entry
    else:
        profit_distance = entry - current_price

    profit_pct_of_tp = (profit_distance / tp_distance) * 100

    # Breakeven at 50% of TP distance
    if profit_pct_of_tp >= BE_ACTIVATION_PCT and not trade.be_activated:
        buffer = tp_distance * 0.05
        if is_long:
            new_sl = entry + buffer
            if new_sl > trade.sl_price:
                trade.sl_price = new_sl
                trade.be_activated = True
                logger.info("Trade {} BE activated at {:.2f}", trade.trade_id, new_sl)
        else:
            new_sl = entry - buffer
            if new_sl < trade.sl_price:
                trade.sl_price = new_sl
                trade.be_activated = True
                logger.info("Trade {} BE activated at {:.2f}", trade.trade_id, new_sl)

    # Trailing stop at 75% — lock 50% of profit
    if profit_pct_of_tp >= TRAILING_ACTIVATION_PCT and trade.be_activated:
        lock_distance = profit_distance * TRAILING_LOCK_RATIO
        if is_long:
            new_sl = entry + lock_distance
            if new_sl > trade.sl_price:
                trade.sl_price = new_sl
                trade.trailing_activated = True
                logger.info("Trade {} trailing stop at {:.2f}", trade.trade_id, new_sl)
        else:
            new_sl = entry - lock_distance
            if new_sl < trade.sl_price:
                trade.sl_price = new_sl
                trade.trailing_activated = True
                logger.info("Trade {} trailing stop at {:.2f}", trade.trade_id, new_sl)


class PaperLedger:
    """Append-only JSONL ledger for paper trades."""

    def __init__(self, path: str = "data/geo_paper_trades.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.trades: dict[str, PaperTrade] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text().strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj.get("trade_id", "")
                if not tid:
                    continue
                self.trades[tid] = PaperTrade(**{
                    k: v for k, v in obj.items()
                    if k in PaperTrade.__dataclass_fields__
                })
        except Exception as exc:
            logger.warning("Failed to load paper ledger: {}", exc)

    def _append(self, trade: PaperTrade) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(trade), default=str) + "\n")

    def open_trade(self, trade: PaperTrade) -> None:
        self.trades[trade.trade_id] = trade
        self._append(trade)
        logger.info(
            "PAPER OPEN {} {} {} @ {:.2f} SL={:.2f} TP={:.2f}",
            trade.trade_id, trade.direction, trade.symbol,
            trade.entry_price, trade.sl_price, trade.tp_price,
        )

    def close_trade(
        self, trade_id: str, exit_price: float, exit_reason: str,
    ) -> PaperTrade | None:
        trade = self.trades.get(trade_id)
        if not trade or trade.status != "OPEN":
            return None

        trade.status = "CLOSED"
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.closed_at = datetime.now(tz=timezone.utc).isoformat()

        if trade.is_long:
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

        trade.pnl_pct = round(pnl_pct, 4)
        trade.pnl_usd = round(trade.notional_usd * pnl_pct / 100, 4)

        self._append(trade)
        logger.info(
            "PAPER CLOSE {} reason={} pnl={:+.2f}% (${:+.2f})",
            trade.trade_id, exit_reason, trade.pnl_pct, trade.pnl_usd,
        )
        return trade

    def get_open_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades.values() if t.status == "OPEN"]

    def get_open_count(self) -> int:
        return sum(1 for t in self.trades.values() if t.status == "OPEN")

    def has_open_for(self, symbol: str, direction: str) -> bool:
        return any(
            t.symbol == symbol and t.direction == direction
            for t in self.trades.values() if t.status == "OPEN"
        )

    def was_recently_traded(self, signal_uid: str, hours: float = 6.0) -> bool:
        cutoff = time.time() - hours * 3600
        return any(
            t.signal_uid == signal_uid and t.opened_ts > cutoff
            for t in self.trades.values()
        )

    @staticmethod
    def make_trade_id() -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%H%M%S")
        return f"PT-{uuid.uuid4().hex[:8]}-{ts}"

    def get_stats(self) -> dict[str, Any]:
        closed = [t for t in self.trades.values() if t.status == "CLOSED"]
        if not closed:
            return {"total_trades": 0, "win_rate": 0, "avg_pnl_pct": 0}
        wins = sum(1 for t in closed if (t.pnl_pct or 0) > 0)
        exit_reasons: dict[str, int] = {}
        for t in closed:
            r = t.exit_reason or "UNKNOWN"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        return {
            "total_trades": len(closed),
            "open_trades": self.get_open_count(),
            "win_rate": round(wins / len(closed) * 100, 1),
            "avg_pnl_pct": round(sum(t.pnl_pct or 0 for t in closed) / len(closed), 3),
            "total_pnl_usd": round(sum(t.pnl_usd or 0 for t in closed), 2),
            "exit_reasons": exit_reasons,
        }
