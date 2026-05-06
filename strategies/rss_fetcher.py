"""Phase 1A: RSS headline fetcher with deduplication."""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from xml.etree import ElementTree

import aiohttp
from loguru import logger

RSS_FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Reuters": "https://feeds.reuters.com/reuters/worldNews",
    "Guardian": "https://www.theguardian.com/world/middleeast/rss",
    "NPR": "https://feeds.npr.org/1004/rss.xml",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "OilPrice": "https://oilprice.com/rss/main",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Bitcoin Mag": "https://bitcoinmagazine.com/.rss/full/",
    "Kitco": "https://www.kitco.com/rss/gold.xml",
}

WINDOW_HOURS = 6


@dataclass
class RSSItem:
    title: str
    url: str
    summary: str
    source: str
    published_ts: float
    signal_uid: str = ""

    def __post_init__(self) -> None:
        if not self.signal_uid:
            raw = f"{self.title}{self.url}"
            self.signal_uid = hashlib.sha1(raw.encode()).hexdigest()[:12]


class RSSFetcher:
    def __init__(self, feeds: dict[str, str] | None = None) -> None:
        self._feeds = feeds or RSS_FEEDS
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._session: aiohttp.ClientSession | None = None
        self._max_seen = 5000

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "NeuralTrader/5.0 RSS"},
            )
        return self._session

    async def _fetch_feed(self, source: str, url: str) -> list[RSSItem]:
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
        except Exception as exc:
            logger.debug("RSS fetch failed for {}: {}", source, exc)
            return []

        items: list[RSSItem] = []
        try:
            root = ElementTree.fromstring(text)
            for item_el in root.iter("item"):
                title_el = item_el.find("title")
                link_el = item_el.find("link")
                desc_el = item_el.find("description")
                title = (title_el.text or "").strip() if title_el is not None else ""
                link = (link_el.text or "").strip() if link_el is not None else ""
                desc = (desc_el.text or "").strip() if desc_el is not None else ""
                if not title:
                    continue
                items.append(RSSItem(
                    title=title[:500],
                    url=link[:500],
                    summary=desc[:500],
                    source=source,
                    published_ts=time.time(),
                ))
        except ElementTree.ParseError:
            logger.debug("XML parse error for {}", source)
        return items

    async def fetch_all(self) -> list[RSSItem]:
        await self._get_session()
        tasks = [self._fetch_feed(src, url) for src, url in self._feeds.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cutoff = time.time() - WINDOW_HOURS * 3600
        new_items: list[RSSItem] = []

        for result in results:
            if isinstance(result, Exception):
                continue
            for item in result:
                if item.signal_uid in self._seen:
                    continue
                if item.published_ts < cutoff:
                    continue
                self._seen[item.signal_uid] = item.published_ts
                new_items.append(item)

        while len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)

        return new_items

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
