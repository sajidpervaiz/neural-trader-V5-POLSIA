"""Paper-mode market data feed.

Streams Binance public kline data over WebSocket (combined-stream) and emits
CANDLE events into the EventBus. Falls back to REST polling if the WebSocket
fails (or when websockets package is unavailable).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from loguru import logger

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]
    _WS_AVAILABLE = False

from core.event_bus import EventBus
from data_ingestion.normalizer import Candle

# Binance futures public klines endpoint (no auth needed)
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
# Combined-stream WebSocket: one connection for many symbol/timeframe pairs
WS_BASE = "wss://fstream.binance.com/stream"

TF_MAP = {
    "1m": ("1m", 60),
    "5m": ("5m", 300),
    "15m": ("15m", 900),
    "1h": ("1h", 3600),
    "4h": ("4h", 14400),
    "1d": ("1d", 86400),
}


class PaperFeed:
    """Fetches candles from Binance public API and publishes CANDLE events."""

    def __init__(
        self,
        event_bus: EventBus,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        poll_interval: float = 30.0,
        data_manager: Any = None,
        market_data_integrity: Any = None,
        use_websocket: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.timeframes = timeframes or ["1m", "15m", "1h", "4h"]
        self.poll_interval = poll_interval
        self._data_manager = data_manager
        self._market_data_integrity = market_data_integrity
        self._use_websocket = use_websocket and _WS_AVAILABLE
        self._running = False
        self._seeding_complete = False
        self._client: httpx.AsyncClient | None = None
        self._last_candle_time: dict[str, int] = {}
        self._ws_reconnect_delay = 5.0

    def _binance_symbol(self, sym: str) -> str:
        """Normalize symbol to Binance format: BTC/USDT:USDT -> BTCUSDT"""
        return sym.replace("/", "").replace(":USDT", "").upper()

    def _internal_symbol(self, binance_sym: str) -> str:
        """Convert BTCUSDT -> BTC/USDT:USDT for internal use."""
        for suffix in ("USDT", "BUSD"):
            if binance_sym.endswith(suffix):
                base = binance_sym[: -len(suffix)]
                return f"{base}/{suffix}:{suffix}"
        return binance_sym

    async def _fetch_klines(
        self, symbol: str, timeframe: str, limit: int = 100,
    ) -> list[Candle]:
        """Fetch klines from Binance public API."""
        if self._client is None:
            return []
        binance_tf = TF_MAP.get(timeframe, (timeframe, 60))[0]
        try:
            resp = await self._client.get(
                KLINES_URL,
                params={
                    "symbol": self._binance_symbol(symbol),
                    "interval": binance_tf,
                    "limit": limit,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("PaperFeed klines error {}/{}: {}", symbol, timeframe, exc)
            return []

        candles = []
        internal_sym = self._internal_symbol(self._binance_symbol(symbol))
        receive_time_us = time.time_ns() // 1000
        now_ms = int(time.time() * 1000)
        for k in data:
            close_time_ms = int(k[6]) if len(k) > 6 else int(k[0])
            if close_time_ms > now_ms:
                continue
            candles.append(Candle(
                exchange="binance",
                symbol=internal_sym,
                timeframe=timeframe,
                timestamp=int(k[0]) // 1000,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                num_trades=int(k[8]) if len(k) > 8 else 0,
                receive_time_us=receive_time_us,
                source_tick_timestamp_us=close_time_ms * 1000,
            ))
        return candles

    async def _poll_once(self) -> int:
        """Poll all symbols/timeframes and emit new candles. Returns count emitted."""
        emitted = 0
        for sym in self.symbols:
            for tf in self.timeframes:
                candles = await self._fetch_klines(sym, tf, limit=100)
                key = f"{sym}:{tf}"
                last_ts = self._last_candle_time.get(key, 0)

                for c in candles:
                    if c.timestamp > last_ts:
                        await self.event_bus.publish("CANDLE", c)
                        emitted += 1

                if candles:
                    self._last_candle_time[key] = candles[-1].timestamp
        return emitted

    async def _poll_loop(self) -> None:
        """REST watchdog used both as fallback and WebSocket gap filler."""
        while self._running:
            await asyncio.sleep(self.poll_interval)
            try:
                count = await self._poll_once()
                if count > 0:
                    logger.debug("PaperFeed: emitted {} new REST candle(s)", count)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("PaperFeed REST watchdog error: {}", exc)

    async def seed_history(self) -> None:
        """Seed DataManager with historical candles on startup.
        
        If data_manager is available, seeds directly (fast, no event bus overhead).
        Otherwise falls back to publishing through event bus.
        """
        logger.info("PaperFeed: seeding historical candles...")
        self.event_bus._seeding = True  # Signal pipeline skips during seeding
        total = 0
        for sym in self.symbols:
            for tf in self.timeframes:
                candles = await self._fetch_klines(sym, tf, limit=200)
                if self._data_manager is not None:
                    # Direct inject — bypasses EventBus queue, skip indicator compute per-candle
                    for c in candles:
                        self._data_manager._store_candle(c.exchange, c.symbol, c.timeframe, c, compute=False)
                        if (
                            self._market_data_integrity is not None
                            and hasattr(self._market_data_integrity, "observe_candle")
                        ):
                            self._market_data_integrity.observe_candle(c)
                        total += 1
                else:
                    for c in candles:
                        await self.event_bus.publish("CANDLE", c)
                        total += 1
                        if total % 50 == 0:
                            await asyncio.sleep(0)
                if candles:
                    key = f"{sym}:{tf}"
                    self._last_candle_time[key] = candles[-1].timestamp
                # Yield to event loop between symbol/tf combos so HTTP stays responsive
                await asyncio.sleep(0)

        # Bulk recompute indicators once after all candles are loaded
        if self._data_manager is not None:
            self._data_manager.recompute_all()

        self.event_bus._seeding = False
        logger.info("PaperFeed: seeded {} candles across {} symbols × {} timeframes",
                     total, len(self.symbols), len(self.timeframes))
        self._seeding_complete = True

    def _build_ws_url(self) -> str:
        """Build a combined-stream URL with one kline stream per symbol×timeframe."""
        streams = []
        for sym in self.symbols:
            bsym = self._binance_symbol(sym).lower()
            for tf in self.timeframes:
                binance_tf = TF_MAP.get(tf, (tf, 60))[0]
                streams.append(f"{bsym}@kline_{binance_tf}")
        return f"{WS_BASE}?streams={'/'.join(streams)}"

    def _parse_kline_message(self, message: dict) -> Candle | None:
        """Parse a Binance combined-stream kline payload into a Candle."""
        data = message.get("data") or {}
        kline = data.get("k") or {}
        if not kline.get("x"):  # only emit closed candles to avoid lookahead
            return None
        binance_sym = (data.get("s") or "").upper()
        if not binance_sym:
            return None
        # Map binance interval back to internal timeframe
        binance_tf = kline.get("i", "")
        tf = next((k for k, v in TF_MAP.items() if v[0] == binance_tf), binance_tf)
        receive_time_us = time.time_ns() // 1000
        return Candle(
            exchange="binance",
            symbol=self._internal_symbol(binance_sym),
            timeframe=tf,
            timestamp=int(kline.get("t", 0)) // 1000,
            open=float(kline.get("o", 0)),
            high=float(kline.get("h", 0)),
            low=float(kline.get("l", 0)),
            close=float(kline.get("c", 0)),
            volume=float(kline.get("v", 0)),
            num_trades=int(kline.get("n", 0)),
            receive_time_us=receive_time_us,
            source_tick_timestamp_us=int(kline.get("T", kline.get("t", 0)) or 0) * 1000,
        )

    async def _ws_loop(self) -> None:
        """WebSocket consumer loop with auto-reconnect."""
        url = self._build_ws_url()
        logger.info("PaperFeed: connecting WebSocket to {} streams", len(self.symbols) * len(self.timeframes))
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("PaperFeed: WebSocket connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            message = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        candle = self._parse_kline_message(message)
                        if candle is None:
                            continue
                        await self.event_bus.publish("CANDLE", candle)
                        key = f"{self._binance_symbol(candle.symbol)}:{candle.timeframe}"
                        self._last_candle_time[key] = candle.timestamp
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("PaperFeed WS error, reconnecting in {}s: {}", self._ws_reconnect_delay, exc)
                await asyncio.sleep(self._ws_reconnect_delay)

    async def run(self, seed_only: bool = False) -> None:
        """Main loop: seed history, then stream live candles.

        Uses WebSocket when available, falls back to REST polling otherwise.

        Args:
            seed_only: If True, seed history and return (used in live mode where WS provides data).
        """
        self._running = True
        self._client = httpx.AsyncClient()
        try:
            await self.seed_history()
            if seed_only:
                logger.info("PaperFeed: seed-only mode — streaming disabled (live WS provides data)")
                return
            if self._use_websocket:
                logger.info("PaperFeed started — WebSocket streaming for {} symbols × {} timeframes",
                            len(self.symbols), len(self.timeframes))
                logger.info("PaperFeed REST watchdog active every {}s", self.poll_interval)
                await asyncio.gather(self._ws_loop(), self._poll_loop())
            else:
                logger.info("PaperFeed started — REST polling every {}s for {}",
                            self.poll_interval, self.symbols)
                await self._poll_loop()
        except asyncio.CancelledError:
            pass
        finally:
            if self._client:
                await self._client.aclose()
                self._client = None
            self._running = False
            logger.info("PaperFeed stopped")

    async def stop(self) -> None:
        self._running = False
