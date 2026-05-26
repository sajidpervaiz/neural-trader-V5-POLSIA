from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

import pandas as pd
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from data_ingestion.normalizer import Tick, Candle
from data_ingestion.tick_processor import CandleAggregator
from analysis.technical import TechnicalIndicators
from analysis.regime import RegimeDetector, RegimeState


TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

MAX_CANDLES_PER_SERIES = 1000


class DataManager:
    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._indicators = TechnicalIndicators()
        self._regime_detectors: dict[str, RegimeDetector] = {}
        self._aggregators: dict[str, dict[str, CandleAggregator]] = defaultdict(dict)
        self._candle_history: dict[str, dict[str, deque[Candle]]] = defaultdict(dict)
        self._dataframes: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self._regimes: dict[str, RegimeState] = {}
        self._running = False

        timeframes_cfg = config.get_value("signals", "timeframes") or ["1m", "5m", "15m", "1h"]
        self._timeframes = [tf for tf in timeframes_cfg if tf in TIMEFRAME_SECONDS]

    def _get_aggregator(self, exchange: str, symbol: str, timeframe: str) -> CandleAggregator:
        key = f"{exchange}:{symbol}"
        if timeframe not in self._aggregators[key]:
            self._aggregators[key][timeframe] = CandleAggregator(
                exchange, symbol, TIMEFRAME_SECONDS[timeframe]
            )
        return self._aggregators[key][timeframe]

    def _store_candle(self, exchange: str, symbol: str, timeframe: str, candle: Candle, compute: bool = True) -> None:
        key = f"{exchange}:{symbol}"
        if timeframe not in self._candle_history[key]:
            self._candle_history[key][timeframe] = deque(maxlen=MAX_CANDLES_PER_SERIES)
        self._candle_history[key][timeframe].append(candle)

        if not compute:
            return

        candles = list(self._candle_history[key][timeframe])
        if len(candles) < 5:
            return

        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in candles])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp").sort_index()

        try:
            df = self._indicators.compute_all(df)
        except Exception as exc:
            logger.debug("Indicator computation error for {}/{}/{}: {}", exchange, symbol, timeframe, exc)
            return

        self._dataframes[key][timeframe] = df
        self._update_regime(key, timeframe, df)

    def recompute_all(self) -> None:
        """Recompute indicators and regimes for all stored candle series. Call after bulk seeding."""
        for key, tf_map in self._candle_history.items():
            for timeframe, candle_deque in tf_map.items():
                candles = list(candle_deque)
                if len(candles) < 5:
                    continue
                df = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                } for c in candles])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df = df.set_index("timestamp").sort_index()
                try:
                    df = self._indicators.compute_all(df)
                except Exception as exc:
                    logger.debug("Recompute error for {}/{}: {}", key, timeframe, exc)
                    continue
                self._dataframes[key][timeframe] = df
                self._update_regime(key, timeframe, df, bootstrap=True)
        logger.info("DataManager: recomputed indicators for {} series",
                     sum(len(tfs) for tfs in self._candle_history.values()))

    def _update_regime(self, key: str, timeframe: str, df: pd.DataFrame, bootstrap: bool = False) -> None:
        if timeframe != "4h":
            return
        if key not in self._regime_detectors:
            self._regime_detectors[key] = RegimeDetector()
        try:
            detector = self._regime_detectors[key]
            if bootstrap and len(df) >= 30:
                # Bootstrap: feed last N windows to advance the state machine
                # past the confirmation threshold (default 3 candles).
                n_boot = min(6, len(df) - 29)  # at least 30 rows per window
                for i in range(n_boot, 0, -1):
                    window = df.iloc[:-i] if i > 0 else df
                    if len(window) >= 30:
                        detector.detect(window)
            state = detector.detect(df)
            self._regimes[key] = state
        except Exception as exc:
            logger.debug("Regime detection error for {}/{}: {}", key, timeframe, exc)

    async def _handle_tick(self, payload: Any) -> None:
        tick: Tick = payload
        for timeframe in self._timeframes:
            agg = self._get_aggregator(tick.exchange, tick.symbol, timeframe)
            candle = agg.add_tick(tick)
            if candle is not None:
                candle.timeframe = timeframe  # Normalize "900s" → "15m" etc.
                candle.receive_time_us = getattr(tick, "receive_time_us", 0)
                candle.source_tick_timestamp_us = getattr(tick, "timestamp_us", 0)
                self._store_candle(tick.exchange, tick.symbol, timeframe, candle)
                await self.event_bus.publish("CANDLE", candle)

    async def _handle_candle(self, payload: Any) -> None:
        candle: Candle = payload
        self._store_candle(candle.exchange, candle.symbol, candle.timeframe, candle)

    def get_dataframe(self, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
        key = f"{exchange}:{symbol}"
        return self._dataframes.get(key, {}).get(timeframe)

    def get_regime(self, exchange: str, symbol: str) -> RegimeState | None:
        return self._regimes.get(f"{exchange}:{symbol}")

    async def run(self) -> None:
        self._running = True
        self.event_bus.subscribe("TICK", self._handle_tick)
        self.event_bus.subscribe("CANDLE", self._handle_candle)
        logger.info("DataManager started — tracking timeframes: {}", self._timeframes)
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        self.event_bus.unsubscribe("TICK", self._handle_tick)
        self.event_bus.unsubscribe("CANDLE", self._handle_candle)
