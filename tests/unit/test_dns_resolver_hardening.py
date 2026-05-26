from __future__ import annotations

from unittest.mock import MagicMock

import aiohttp
import pytest

from data_ingestion.dex_rpc import DEXRPCFeed
from data_ingestion.geopolitical_news import GeoPoliticalNewsFeed


class _DummySession:
    closed = False


@pytest.mark.asyncio
async def test_dex_rpc_uses_threaded_dns_resolver(monkeypatch):
    captured = {}

    def fake_client_session(**kwargs):
        captured.update(kwargs)
        return _DummySession()

    monkeypatch.setattr(aiohttp, "ClientSession", fake_client_session)
    feed = DEXRPCFeed(MagicMock(), MagicMock())

    session = await feed._get_session()

    assert session is feed._session
    connector = captured.get("connector")
    assert isinstance(connector, aiohttp.TCPConnector)
    assert isinstance(connector._resolver, aiohttp.ThreadedResolver)
    await connector.close()


@pytest.mark.asyncio
async def test_geopolitical_feed_uses_threaded_dns_resolver(monkeypatch):
    captured = {}

    def fake_client_session(**kwargs):
        captured.update(kwargs)
        return _DummySession()

    monkeypatch.setattr(aiohttp, "ClientSession", fake_client_session)
    feed = GeoPoliticalNewsFeed(MagicMock(), MagicMock(), MagicMock(), feeds=[])

    session = await feed._get_session()

    assert session is feed._session
    connector = captured.get("connector")
    assert isinstance(connector, aiohttp.TCPConnector)
    assert isinstance(connector._resolver, aiohttp.ThreadedResolver)
    await connector.close()
