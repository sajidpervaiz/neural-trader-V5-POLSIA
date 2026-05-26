"""Geopolitical RSS poller — fetches the spec's 13 feeds, scores per-symbol, emits events.

For each new article (deduped by GUID/link), the configured GeoPoliticalScorer
is asked to produce per-symbol ScoredEvents. Each surviving event is published
to the EventBus as a GEOPOLITICAL_EVENT and ingested into the scorer.

Independent from the existing crypto NewsFeed — this runs in parallel and does
not displace NEWS_SENTIMENT events, only adds to them.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse

import aiohttp
from loguru import logger

# Hardened XML parser — disables external entity resolution, DTDs, and the
# billion-laughs amplification vector. Falls back to stdlib ElementTree if
# defusedxml is not installed (RSS hosts are trusted-but-public so this is
# defence in depth, not a primary control).
try:
    from defusedxml import ElementTree as ET  # type: ignore[import-untyped]
    _ET_DEFUSED = True
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    _ET_DEFUSED = False
    logger.warning(
        "defusedxml not installed — geopolitical RSS parser using stdlib ElementTree "
        "(install defusedxml to harden against XXE / billion-laughs attacks)",
    )

from core.config import Config
from core.event_bus import EventBus
from engine.geopolitical_scorer import GeoPoliticalScorer, ScoredEvent
from strategies.geo_political_strategy import RSS_FEEDS


_USER_AGENT = "neural-trader-v5/geopolitical-feed (+https://github.com)"
_FEED_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=8)

# aiohttp ≥3.10 added ClientConnectorDNSError; older versions fold DNS failures
# into ClientConnectorError. Resolve at import time so we can catch precisely
# without breaking on older releases.
_DNS_ERROR_CLS: type[BaseException] = getattr(
    aiohttp, "ClientConnectorDNSError", aiohttp.ClientConnectorError,
)


def _parse_rss(xml_bytes: bytes) -> list[dict[str, str]]:
    """Tolerant RSS / Atom parser — returns dicts with id/title/description/link."""
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return items

    # RSS 2.0
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        desc = (it.findtext("description") or "").strip()
        link = (it.findtext("link") or "").strip()
        guid = (it.findtext("guid") or link or title).strip()
        if title:
            items.append({"id": guid, "title": title, "description": desc, "link": link})

    if items:
        return items

    # Atom fallback (e.g. Bitcoin Magazine)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        if not summary:
            summary = (entry.findtext("a:content", default="", namespaces=ns) or "").strip()
        link_el = entry.find("a:link", namespaces=ns)
        link = link_el.get("href", "") if link_el is not None else ""
        guid = (entry.findtext("a:id", default="", namespaces=ns) or link or title).strip()
        if title:
            items.append({"id": guid, "title": title, "description": summary, "link": link})
    return items


class GeoPoliticalNewsFeed:
    """Periodic RSS poller for the geopolitical strategy."""

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        scorer: GeoPoliticalScorer,
        feeds: list[str] | None = None,
        fetch_interval: float | None = None,
        max_articles_per_feed: int = 30,
        max_seen: int = 4000,
        max_concurrent_fetches: int = 4,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self._scorer = scorer
        cfg = config.get_value("geopolitical") or {}
        self._feeds: list[str] = list(feeds or cfg.get("feeds") or RSS_FEEDS)
        default_interval = float(cfg.get("fetch_interval", 600.0))
        self._interval: float = float(fetch_interval if fetch_interval is not None else default_interval)
        self._max_articles_per_feed = int(max_articles_per_feed)
        self._seen_ids: OrderedDict[str, None] = OrderedDict()
        self._max_seen = int(max_seen)
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        # Per-host concurrency cap — gentler on RSS hosts, avoids self-DDoS.
        self._fetch_semaphore = asyncio.Semaphore(int(max_concurrent_fetches))
        # Per-URL conditional-GET cache: {url: {"etag": str, "last_modified": str}}.
        # Saves ~6× bandwidth/day vs. unconditional re-fetch every 600s.
        self._conditional_cache: dict[str, dict[str, str]] = {}
        self._stats = {
            "polls": 0, "articles": 0, "events": 0,
            "errors_dns": 0, "errors_http": 0, "errors_parse": 0, "errors_other": 0,
            "not_modified": 0,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=_FEED_TIMEOUT,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
                # Avoid aiohttp/aiodns "exception in shielded future" stderr
                # tracebacks when one public RSS host has flaky DNS.
                connector=aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver()),
            )
        return self._session

    async def _fetch_feed(self, url: str) -> list[dict[str, str]]:
        session = await self._get_session()
        # Build conditional-GET headers from previous response (304 short-circuit).
        cached = self._conditional_cache.get(url, {})
        headers: dict[str, str] = {}
        if cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        if cached.get("last_modified"):
            headers["If-Modified-Since"] = cached["last_modified"]

        async with self._fetch_semaphore:
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 304:
                        self._stats["not_modified"] += 1
                        return []
                    if resp.status != 200:
                        self._stats["errors_http"] += 1
                        logger.debug(
                            "GeoPolitical feed [{}] HTTP {}",
                            urlparse(url).netloc, resp.status,
                        )
                        return []
                    body = await resp.read()
                    new_cache: dict[str, str] = {}
                    if etag := resp.headers.get("ETag"):
                        new_cache["etag"] = etag
                    if lm := resp.headers.get("Last-Modified"):
                        new_cache["last_modified"] = lm
                    if new_cache:
                        self._conditional_cache[url] = new_cache
            except _DNS_ERROR_CLS:
                self._stats["errors_dns"] += 1
                logger.debug("GeoPolitical feed [{}] DNS failure", urlparse(url).netloc)
                return []
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self._stats["errors_http"] += 1
                logger.debug(
                    "GeoPolitical feed [{}] connect/timeout: {}",
                    urlparse(url).netloc, exc,
                )
                return []
            except Exception as exc:
                self._stats["errors_other"] += 1
                logger.debug("GeoPolitical feed [{}] error: {}", urlparse(url).netloc, exc)
                return []

        try:
            return _parse_rss(body)[: self._max_articles_per_feed]
        except Exception as exc:
            self._stats["errors_parse"] += 1
            logger.debug("GeoPolitical feed [{}] parse error: {}", urlparse(url).netloc, exc)
            return []

    def _remember(self, article_id: str) -> bool:
        """Returns True if this is a new article (not seen before)."""
        if article_id in self._seen_ids:
            return False
        self._seen_ids[article_id] = None
        while len(self._seen_ids) > self._max_seen:
            self._seen_ids.popitem(last=False)
        return True

    async def _publish_event(self, event: ScoredEvent) -> None:
        payload = dict(asdict(event))
        await self.event_bus.publish("GEOPOLITICAL_EVENT", payload)

    async def _poll_once(self) -> None:
        results = await asyncio.gather(
            *[self._fetch_feed(url) for url in self._feeds],
            return_exceptions=False,
        )
        new_events = 0
        new_articles = 0
        for url, articles in zip(self._feeds, results):
            source = urlparse(url).netloc
            for art in articles:
                aid = art.get("id") or art.get("link") or art.get("title", "")
                if not aid or not self._remember(aid):
                    continue
                new_articles += 1
                events = self._scorer.score_article(
                    title=art.get("title", ""),
                    body=art.get("description", ""),
                    source=source,
                    timestamp=time.time(),
                )
                for ev in events:
                    self._scorer.ingest(ev)
                    await self._publish_event(ev)
                    new_events += 1
        self._stats["polls"] += 1
        self._stats["articles"] += new_articles
        self._stats["events"] += new_events
        if new_articles or new_events:
            logger.info(
                "GeoPolitical poll: {} new articles, {} scored events (totals: {} arts, {} events)",
                new_articles, new_events, self._stats["articles"], self._stats["events"],
            )
        else:
            logger.debug("GeoPolitical poll: no new articles")

    async def run(self) -> None:
        self._running = True
        logger.info(
            "GeoPoliticalNewsFeed started (interval={}s, feeds={})",
            self._interval, len(self._feeds),
        )
        # First poll immediately so the scorer has data within seconds of startup.
        try:
            await self._poll_once()
        except Exception as exc:
            logger.warning("GeoPolitical initial poll failed: {}", exc)
        while self._running:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            try:
                await self._poll_once()
            except Exception as exc:
                logger.warning("GeoPolitical poll error: {}", exc)
                self._stats["errors_other"] += 1

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        err_total = (
            self._stats["errors_dns"] + self._stats["errors_http"]
            + self._stats["errors_parse"] + self._stats["errors_other"]
        )
        logger.info(
            "GeoPoliticalNewsFeed stopped (polls={} articles={} events={} errors={} "
            "[dns={} http={} parse={} other={}] not_modified={})",
            self._stats["polls"], self._stats["articles"], self._stats["events"], err_total,
            self._stats["errors_dns"], self._stats["errors_http"],
            self._stats["errors_parse"], self._stats["errors_other"],
            self._stats["not_modified"],
        )

    def stats(self) -> dict[str, Any]:
        snap = dict(self._stats)
        snap["scorer"] = self._scorer.snapshot()
        return snap


__all__ = ["GeoPoliticalNewsFeed"]
