from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger

from core.error_handling import sanitize_exception

from core.config import Config
from core.event_bus import EventBus


@dataclass
class OpenInterest:
    exchange: str
    symbol: str
    oi_usd: float
    oi_contracts: float
    timestamp: int
    oi_change_24h: float = 0.0


OI_URLS: dict[str, str] = {
    "binance": "https://fapi.binance.com/fapi/v1/openInterest",
    "bybit": "https://api.bybit.com/v5/market/open-interest",
    "okx": "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
}


class OpenInterestFeed:
    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, OpenInterest] = {}
        self._history: dict[str, list[OpenInterest]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def _fetch_binance(self, symbols: list[str]) -> list[OpenInterest]:
        session = await self._get_session()
        results = []
        for symbol in symbols:
            clean = symbol.replace("/USDT:USDT", "USDT").replace("/", "")
            try:
                async with session.get(
                    OI_URLS["binance"], params={"symbol": clean}
                ) as resp:
                    data = await resp.json()
                    oi = OpenInterest(
                        exchange="binance",
                        symbol=symbol,
                        oi_contracts=float(data.get("openInterest", 0)),
                        oi_usd=float(data.get("openInterestNotional", 0)),
                        timestamp=int(time.time()),
                    )
                    results.append(oi)
            except Exception as exc:
                logger.debug("Binance OI fetch error for {}: {}", symbol, exc)
        return results

    async def _fetch_bybit(self, symbols: list[str]) -> list[OpenInterest]:
        session = await self._get_session()
        results = []
        for symbol in symbols:
            clean = symbol.replace("/USDT:USDT", "USDT")
            try:
                async with session.get(
                    OI_URLS["bybit"],
                    params={"category": "linear", "symbol": clean, "intervalTime": "5min", "limit": 2},
                ) as resp:
                    data = await resp.json()
                    items = data.get("result", {}).get("list", [])
                    if items:
                        latest = items[0]
                        prev = items[1] if len(items) > 1 else None
                        oi_val = float(latest.get("openInterest", 0))
                        prev_val = float(prev.get("openInterest", 0)) if prev else oi_val
                        results.append(OpenInterest(
                            exchange="bybit",
                            symbol=symbol,
                            oi_contracts=oi_val,
                            oi_usd=oi_val,
                            timestamp=int(time.time()),
                            oi_change_24h=(oi_val - prev_val) / prev_val if prev_val else 0.0,
                        ))
            except Exception as exc:
                logger.debug("Bybit OI fetch error for {}: {}", symbol, exc)
        return results

    def get_latest(self, exchange: str, symbol: str) -> OpenInterest | None:
        return self._cache.get(f"{exchange}:{symbol}")

    def get_aggregate(self, symbol: str) -> dict[str, Any]:
        total_oi = 0.0
        breakdown: dict[str, float] = {}
        for key, oi in self._cache.items():
            if oi.symbol == symbol:
                total_oi += oi.oi_usd
                breakdown[oi.exchange] = oi.oi_usd
        return {"total_usd": total_oi, "by_exchange": breakdown}

    async def run(self) -> None:
        self._running = True
        macro_cfg = self.config.get_value("macro", "open_interest") or {}
        if not macro_cfg.get("enabled", False):
            logger.info("Open interest feed disabled — skipping")
            while self._running:
                await asyncio.sleep(10)
            return

        interval = float(macro_cfg.get("fetch_interval_seconds", 300))
        exchanges_cfg = self.config.get_value("exchanges") or {}
        logger.info("Open interest feed started (interval={}s)", interval)

        while self._running:
            try:
                all_results: list[OpenInterest] = []
                binance_syms = exchanges_cfg.get("binance", {}).get("symbols", [])
                if binance_syms:
                    all_results.extend(await self._fetch_binance(binance_syms))
                if exchanges_cfg.get("bybit", {}).get("enabled"):
                    bybit_syms = exchanges_cfg.get("bybit", {}).get("symbols", [])
                    all_results.extend(await self._fetch_bybit(bybit_syms))

                for oi in all_results:
                    key = f"{oi.exchange}:{oi.symbol}"
                    self._cache[key] = oi
                    if key not in self._history:
                        self._history[key] = []
                    self._history[key].append(oi)
                    if len(self._history[key]) > 288:
                        self._history[key] = self._history[key][-288:]

                await self.event_bus.publish("OPEN_INTEREST", all_results)
                logger.debug("Fetched OI for {} instruments", len(all_results))
            except Exception as exc:
                logger.error("OI feed error: {}", sanitize_exception(exc))
                logger.opt(exception=True).debug("OI feed stack trace")
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
