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
class FundingRate:
    exchange: str
    symbol: str
    rate: float
    predicted_rate: float | None
    next_funding_time: int
    timestamp: int


@dataclass
class FundingArbitrageOpportunity:
    symbol: str
    long_exchange: str
    short_exchange: str
    long_rate: float
    short_rate: float
    spread_bps: float
    timestamp: int


FUNDING_URLS: dict[str, str] = {
    "binance": "https://fapi.binance.com/fapi/v1/premiumIndex",
    "bybit": "https://api.bybit.com/v5/market/funding/history",
    "okx": "https://www.okx.com/api/v5/public/funding-rate",
}


class FundingRateFeed:
    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, FundingRate] = {}

    def detect_arbitrage_opportunities(
        self,
        rates: list[FundingRate],
        min_spread_bps: float = 2.0,
    ) -> list[FundingArbitrageOpportunity]:
        """Find cross-venue funding spread opportunities for same symbol."""
        if not rates:
            return []

        by_symbol: dict[str, list[FundingRate]] = {}
        for r in rates:
            key = self._normalize_symbol(r.symbol)
            by_symbol.setdefault(key, []).append(r)

        opportunities: list[FundingArbitrageOpportunity] = []
        now_ts = int(time.time())

        for symbol, symbol_rates in by_symbol.items():
            if len(symbol_rates) < 2:
                continue

            min_rate = min(symbol_rates, key=lambda x: x.rate)
            max_rate = max(symbol_rates, key=lambda x: x.rate)

            spread_bps = (max_rate.rate - min_rate.rate) * 10000.0
            if spread_bps < min_spread_bps:
                continue

            opportunities.append(
                FundingArbitrageOpportunity(
                    symbol=symbol,
                    long_exchange=min_rate.exchange,
                    short_exchange=max_rate.exchange,
                    long_rate=min_rate.rate,
                    short_rate=max_rate.rate,
                    spread_bps=spread_bps,
                    timestamp=now_ts,
                )
            )

        opportunities.sort(key=lambda x: x.spread_bps, reverse=True)
        return opportunities

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def _fetch_binance(self, symbols: list[str]) -> list[FundingRate]:
        session = await self._get_session()
        results = []
        try:
            async with session.get(FUNDING_URLS["binance"]) as resp:
                if resp.status != 200:
                    logger.debug("Binance funding HTTP {}", resp.status)
                    return results
                data = await resp.json()
                if not isinstance(data, list):
                    logger.debug("Binance funding unexpected response type: {}", type(data).__name__)
                    return results
                for item in data:
                    symbol = item.get("symbol", "")
                    if not symbol:
                        continue
                    results.append(FundingRate(
                        exchange="binance",
                        symbol=symbol,
                        rate=float(item.get("lastFundingRate", 0)),
                        predicted_rate=float(item.get("interestRate", 0)),
                        next_funding_time=int(item.get("nextFundingTime", 0)),
                        timestamp=int(time.time()),
                    ))
        except Exception as exc:
            logger.debug("Binance funding fetch error: {}", exc)
        return results

    async def _fetch_bybit(self, symbols: list[str]) -> list[FundingRate]:
        session = await self._get_session()
        results = []
        for symbol in symbols:
            try:
                clean = symbol.replace("/USDT:USDT", "USDT")
                async with session.get(
                    FUNDING_URLS["bybit"],
                    params={"category": "linear", "symbol": clean, "limit": 1},
                ) as resp:
                    data = await resp.json()
                    items = data.get("result", {}).get("list", [])
                    for item in items:
                        results.append(FundingRate(
                            exchange="bybit",
                            symbol=item.get("symbol", symbol),
                            rate=float(item.get("fundingRate", 0)),
                            predicted_rate=None,
                            next_funding_time=int(item.get("fundingRateTimestamp", 0)) // 1000,
                            timestamp=int(time.time()),
                        ))
            except Exception as exc:
                logger.debug("Bybit funding fetch error for {}: {}", symbol, exc)
        return results

    async def _fetch_okx(self, symbols: list[str]) -> list[FundingRate]:
        session = await self._get_session()
        results = []
        for symbol in symbols:
            try:
                inst_id = symbol.replace("/USDT:USDT", "-USDT-SWAP")
                async with session.get(
                    FUNDING_URLS["okx"],
                    params={"instId": inst_id},
                ) as resp:
                    data = await resp.json()
                    for item in data.get("data", []):
                        results.append(FundingRate(
                            exchange="okx",
                            symbol=item.get("instId", symbol),
                            rate=float(item.get("fundingRate", 0)),
                            predicted_rate=float(item.get("nextFundingRate", 0)),
                            next_funding_time=int(item.get("nextFundingTime", 0)) // 1000,
                            timestamp=int(time.time()),
                        ))
            except Exception as exc:
                logger.debug("OKX funding fetch error for {}: {}", symbol, exc)
        return results

    async def _fetch_all(self) -> list[FundingRate]:
        macro_cfg = self.config.get_value("macro", "funding_rates") or {}
        sources = macro_cfg.get("sources", ["binance"])
        exchanges_cfg = self.config.get_value("exchanges") or {}

        tasks: list[Any] = []
        if "binance" in sources:
            syms = exchanges_cfg.get("binance", {}).get("symbols", [])
            tasks.append(self._fetch_binance(syms))
        if "bybit" in sources and exchanges_cfg.get("bybit", {}).get("enabled"):
            syms = exchanges_cfg.get("bybit", {}).get("symbols", [])
            tasks.append(self._fetch_bybit(syms))
        if "okx" in sources and exchanges_cfg.get("okx", {}).get("enabled"):
            syms = exchanges_cfg.get("okx", {}).get("symbols", [])
            tasks.append(self._fetch_okx(syms))

        all_results: list[FundingRate] = []
        if not tasks:
            return all_results

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.debug("Funding source task failed: {}", r)
                continue
            all_results.extend(r)
        return all_results

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = symbol.upper().replace("/", "").replace(":", "").replace("-", "")
        # Normalize common perp suffixes to canonical spot-like key for cross-venue compare.
        for suffix in ("USDTUSDT", "USDTSWAP", "PERP", "SWAP"):
            if s.endswith(suffix):
                s = s[: -len(suffix)] + "USDT"
                break
        return s

    def get_latest(self, exchange: str, symbol: str) -> FundingRate | None:
        return self._cache.get(f"{exchange}:{symbol}")

    def get_all_latest(self) -> dict[str, FundingRate]:
        return dict(self._cache)

    async def run(self) -> None:
        self._running = True
        macro_cfg = self.config.get_value("macro", "funding_rates") or {}
        if not macro_cfg.get("enabled", False):
            logger.info("Funding rate feed disabled — skipping")
            while self._running:
                await asyncio.sleep(10)
            return

        interval = float(macro_cfg.get("fetch_interval_seconds", 300))
        logger.info("Funding rate feed started (interval={}s)", interval)
        while self._running:
            try:
                rates = await self._fetch_all()
                for rate in rates:
                    key = f"{rate.exchange}:{rate.symbol}"
                    self._cache[key] = rate
                await self.event_bus.publish("FUNDING_RATE", rates)
                macro_cfg = self.config.get_value("macro", "funding_rates") or {}
                min_spread_bps = float(macro_cfg.get("arbitrage_min_spread_bps", 2.0))
                opportunities = self.detect_arbitrage_opportunities(rates, min_spread_bps=min_spread_bps)
                if opportunities:
                    await self.event_bus.publish("FUNDING_ARBITRAGE", opportunities)
                    logger.info("Detected {} funding arbitrage opportunities", len(opportunities))
                logger.debug("Fetched {} funding rates", len(rates))
            except Exception as exc:
                logger.error("Funding rate feed error: {}", sanitize_exception(exc))
                logger.opt(exception=True).debug("Funding rate feed stack trace")
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
