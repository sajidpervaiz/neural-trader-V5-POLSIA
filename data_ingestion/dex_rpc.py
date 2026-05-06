from __future__ import annotations

import asyncio
import time

import aiohttp
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from data_ingestion.normalizer import Tick


UNISWAP_V3_GRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"


class DEXRPCFeed:
    """Polls DEX price data via JSON-RPC / The Graph when the TS layer is unavailable."""

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def _query_uniswap_v3_prices(self) -> list[Tick]:
        query = """
        {
          bundles(first: 1) { ethPriceUSD }
          pools(
            first: 5
            orderBy: totalValueLockedUSD
            orderDirection: desc
            where: { feeTier_in: [500, 3000] }
          ) {
            token0 { symbol }
            token1 { symbol }
            token0Price
            token1Price
          }
        }
        """
        session = await self._get_session()
        try:
            async with session.post(
                UNISWAP_V3_GRAPH,
                json={"query": query},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    logger.debug("Uniswap V3 graph returned status {}", resp.status)
                    return list()
                data = await resp.json()
                ticks = []
                pools = data.get("data", {}).get("pools", [])
                for pool in pools:
                    symbol = f"{pool['token0']['symbol']}/{pool['token1']['symbol']}"
                    price = float(pool.get("token0Price", 0))
                    if price > 0:
                        ticks.append(Tick(
                            exchange="uniswap_v3",
                            symbol=symbol,
                            timestamp_us=time.time_ns() // 1000,
                            price=price,
                            volume=0.0,
                        ))
                return ticks
        except Exception as exc:
            logger.debug("Uniswap V3 graph query failed: {}", exc)
            return list()

    async def run(self) -> None:
        self._running = True
        dex_cfg = self.config.get_value("dex") or {}
        if not dex_cfg.get("enabled", False):
            logger.info("DEX RPC feed disabled — skipping")
            while self._running:
                await asyncio.sleep(10)
            return

        logger.info("DEX RPC feed started")
        interval = 5.0
        while self._running:
            try:
                ticks = await self._query_uniswap_v3_prices()
                for tick in ticks:
                    await self.event_bus.publish("TICK", tick)
            except Exception as exc:
                logger.warning("DEX RPC feed error: {}", exc)
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
