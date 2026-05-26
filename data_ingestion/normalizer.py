from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
from loguru import logger


@dataclass
class Tick:
    exchange: str
    symbol: str
    timestamp_us: int
    price: float
    volume: float
    side: str = ""
    trade_id: str = ""
    receive_time_us: int = 0
    sequence: int = 0
    sequence_gap: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tick":
        return cls(
            exchange=d["exchange"],
            symbol=d["symbol"],
            timestamp_us=int(d.get("timestamp_us", time.time_ns() // 1000)),
            price=float(d["price"]),
            volume=float(d.get("volume", 0.0)),
            side=str(d.get("side", "")),
            trade_id=str(d.get("trade_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp_us": self.timestamp_us,
            "price": self.price,
            "volume": self.volume,
            "side": self.side,
            "trade_id": self.trade_id,
            "receive_time_us": self.receive_time_us,
            "sequence": self.sequence,
            "sequence_gap": self.sequence_gap,
        }


@dataclass
class Candle:
    exchange: str
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int = 0
    receive_time_us: int = 0
    source_tick_timestamp_us: int = 0


@dataclass
class OrderBookSnapshot:
    exchange: str
    symbol: str
    timestamp_us: int
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return ((self.best_ask - self.best_bid) / mid) * 10_000


class Normalizer:
    """Normalises raw exchange messages into unified Tick / Candle objects."""

    EXCHANGE_SYMBOL_MAP: dict[str, dict[str, str]] = {
        "binance": {},
        "bybit": {},
        "okx": {},
        "kraken": {},
    }

    def normalize_tick(self, exchange: str, raw: dict[str, Any]) -> Tick | None:
        ticks = self.normalize_tick_batch(exchange, raw)
        return ticks[0] if ticks else None

    def normalize_tick_batch(self, exchange: str, raw: dict[str, Any]) -> list[Tick]:
        """Return all ticks from a single websocket message (Kraken sends batches)."""
        try:
            if exchange == "binance":
                return [Tick(
                    exchange=exchange,
                    symbol=self._unify_symbol(exchange, raw.get("s", "")),
                    timestamp_us=int(raw.get("T", time.time_ns() // 1_000_000)) * 1000,
                    price=float(raw.get("p", 0)),
                    volume=float(raw.get("q", 0)),
                    side="buy" if raw.get("m") is False else "sell",
                    trade_id=str(raw.get("t", "")),
                )]
            if exchange == "bybit":
                data = raw.get("data", [{}])
                if isinstance(data, list):
                    data = data[0] if data else {}
                return [Tick(
                    exchange=exchange,
                    symbol=self._unify_symbol(exchange, data.get("s", "")),
                    timestamp_us=int(data.get("T", time.time_ns() // 1_000_000)) * 1000,
                    price=float(data.get("p", 0)),
                    volume=float(data.get("v", 0)),
                    side=data.get("S", "").lower(),
                    trade_id=str(data.get("i", "")),
                )]
            if exchange == "okx":
                data = raw.get("data", [{}])
                if isinstance(data, list):
                    data = data[0] if data else {}
                return [Tick(
                    exchange=exchange,
                    symbol=self._unify_symbol(exchange, data.get("instId", "")),
                    timestamp_us=int(data.get("ts", time.time_ns() // 1_000_000)) * 1000,
                    price=float(data.get("px", 0)),
                    volume=float(data.get("sz", 0)),
                    side=data.get("side", "").lower(),
                    trade_id=str(data.get("tradeId", "")),
                )]
            if exchange == "kraken":
                # Canonical Kraken trade payload shape:
                # [channel_id, [[price, volume, time, side, order_type, misc], ...], "trade", "XBT/USD"]
                if isinstance(raw, list) and len(raw) >= 2:
                    symbol_raw = str(raw[-1]) if len(raw) >= 4 else ""
                    trades = raw[1] if isinstance(raw[1], list) else []
                    results: list[Tick] = []

                    for entry in trades:
                        if isinstance(entry, list):
                            t = entry
                        else:
                            t = raw
                        results.append(Tick(
                            exchange=exchange,
                            symbol=self._unify_symbol(exchange, symbol_raw),
                            timestamp_us=int(float(t[2]) * 1_000_000) if len(t) > 2 else int(time.time_ns() // 1000),
                            price=float(t[0]) if len(t) > 0 else 0.0,
                            volume=float(t[1]) if len(t) > 1 else 0.0,
                            side="buy" if len(t) > 3 and t[3] == "b" else "sell",
                            trade_id=str(t[5]) if len(t) > 5 else "",
                        ))
                    return results
        except (KeyError, ValueError, IndexError) as exc:
            logger.debug("normalize_tick failed for exchange {}: {}", exchange, exc)
        return []

    def _unify_symbol(self, exchange: str, symbol: str) -> str:
        mapping = self.EXCHANGE_SYMBOL_MAP.get(exchange, {})
        if symbol in mapping:
            return mapping[symbol]

        if exchange == "okx" and symbol.endswith("-USDT-SWAP"):
            base = symbol[: -len("-USDT-SWAP")]
            return f"{base}/USDT:USDT"

        if exchange in {"binance", "bybit"} and symbol.endswith("USDT") and "/" not in symbol:
            base = symbol[: -len("USDT")]
            if base:
                return f"{base}/USDT:USDT"

        return symbol
