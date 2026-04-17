from __future__ import annotations

import time
from typing import Any

from loguru import logger

from data_ingestion.normalizer import Tick, Candle


try:
    import neural_trader_rust as _rust
    _RUST_AVAILABLE = True
    logger.info("Rust tick processor loaded (neural_trader_rust)")
except ImportError:
    _RUST_AVAILABLE = False
    logger.info("Rust module unavailable — using Python tick processor")


class CandleAggregator:
    """Aggregates ticks into candles. Uses Rust if available, else pure Python."""

    def __init__(self, exchange: str, symbol: str, timeframe_seconds: int) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe_seconds = timeframe_seconds
        self._current: dict[str, Any] | None = None
        self._completed: list[Candle] = []
        self._start_bucket: int | None = None

    def _current_bucket(self, ts_us: int) -> int:
        """Anchor buckets to the first observed tick for deterministic per-stream candles."""
        ts_s = ts_us // 1_000_000
        if self._start_bucket is None:
            self._start_bucket = ts_s
        elapsed = max(0, ts_s - self._start_bucket)
        return self._start_bucket + (elapsed // self.timeframe_seconds) * self.timeframe_seconds

    def add_tick(self, tick: Tick) -> Candle | None:
        # Keep candle state instance-local. The shared Rust module is still used for
        # batch parsing, but candle aggregation must remain deterministic per stream.
        return self._add_tick_python(tick)

    def _add_tick_python(self, tick: Tick) -> Candle | None:
        bucket = self._current_bucket(tick.timestamp_us)
        completed: Candle | None = None

        if self._current is None:
            self._current = {
                "bucket": bucket,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.volume,
                "num_trades": 1,
            }
        elif self._current["bucket"] != bucket:
            c = self._current
            completed = Candle(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe=f"{self.timeframe_seconds}s",
                timestamp=c["bucket"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
                num_trades=c["num_trades"],
            )
            self._current = {
                "bucket": bucket,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.volume,
                "num_trades": 1,
            }
        else:
            self._current["high"] = max(self._current["high"], tick.price)
            self._current["low"] = min(self._current["low"], tick.price)
            self._current["close"] = tick.price
            self._current["volume"] += tick.volume
            self._current["num_trades"] += 1

        return completed

    def _add_tick_rust(self, tick: Tick) -> Candle | None:
        try:
            result = _rust.TickParser.add_tick(
                self.exchange,
                self.symbol,
                self.timeframe_seconds,
                tick.timestamp_us,
                tick.price,
                tick.volume,
            )
            if result:
                return Candle(
                    exchange=result["exchange"],
                    symbol=result["symbol"],
                    timeframe=result["timeframe"],
                    timestamp=result["timestamp"],
                    open=result["open"],
                    high=result["high"],
                    low=result["low"],
                    close=result["close"],
                    volume=result["volume"],
                    num_trades=result["num_trades"],
                )
        except Exception as exc:
            logger.debug("Rust tick processor error (falling back): {}", exc)
            return self._add_tick_python(tick)
        return None


class TickBatchParser:
    """Parses batches of raw tick dicts efficiently."""

    def __init__(self) -> None:
        self.parse_errors = 0

    def parse_batch(self, exchange: str, raw_ticks: list[dict]) -> list[Tick]:
        if _RUST_AVAILABLE:
            try:
                return [
                    Tick(**t)
                    for t in _rust.TickParser.parse_batch(exchange, raw_ticks)
                ]
            except Exception as exc:
                logger.debug("Rust batch parse error: {}", exc)

        results = []
        now_us = time.time_ns() // 1000
        for raw in raw_ticks:
            try:
                results.append(Tick(
                    exchange=exchange,
                    symbol=raw["symbol"],
                    timestamp_us=int(raw.get("timestamp_us", now_us)),
                    price=float(raw["price"]),
                    volume=float(raw.get("volume", 0.0)),
                    side=str(raw.get("side", "")),
                    trade_id=str(raw.get("trade_id", "")),
                ))
            except (KeyError, ValueError):
                self.parse_errors += 1
                continue
        return results
