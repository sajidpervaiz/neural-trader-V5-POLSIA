"""Geo-Political News-Driven Trading Strategy — main orchestrator.

Runs a 2-minute scan loop:
  Phase 1A: RSS fetch → Phase 1B: keyword relevance → Phase 1C: LLM sentiment
  → V1 gates (dedup, max concurrent, momentum) → Phase 2: LLM confirmation
  → Phase 3: LLM trade plan → Execute paper trade → Manage open trades
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from strategies.market_configs import (
    ALL_MARKETS, MIN_RELEVANCE, MarketConfig,
)
from strategies.rss_fetcher import RSSFetcher, RSSItem
from strategies.llm_client import LLMClient, SentimentResult, ConfirmationResult, TradePlan
from strategies.paper_ledger import (
    PaperLedger, PaperTrade, check_exits, update_protective_stops,
)

try:
    import ccxt.async_support as ccxt_async
    _CCXT = True
except ImportError:
    ccxt_async = None
    _CCXT = False


# ── Configuration defaults ────────────────────────────────────────────────────

SCAN_INTERVAL_SEC = 120
MAX_CONCURRENT_TRADES = 3
MARGIN_USD = 5.0
LEVERAGE = 5
MIN_PHASE1_CONFIDENCE = 60
MIN_PHASE2_SCORE = 60

# Momentum gate thresholds by Phase-1 confidence tier
MOMENTUM_THRESHOLDS = {
    "low": 0.30,   # conf < 60
    "mid": 0.20,   # 60 <= conf < 80
    "high": 0.10,  # conf >= 80
}


@dataclass
class ScoredItem:
    item: RSSItem
    market: MarketConfig
    relevance: int


@dataclass
class MarketSnapshot:
    price: float
    pct_1h: float
    pct_24h: float
    pct_15m: float
    pct_5m: float
    volume_24h: float
    source: str = "ccxt"


class GeoPolStrategy:
    """Main strategy loop. Call run() to start, stop() to stop."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        event_bus: Any = None,
        ledger_path: str = "data/geo_paper_trades.jsonl",
    ) -> None:
        cfg = config or {}
        self._scan_interval = cfg.get("scan_interval_sec", SCAN_INTERVAL_SEC)
        self._max_concurrent = cfg.get("max_concurrent_trades", MAX_CONCURRENT_TRADES)
        self._margin_usd = cfg.get("margin_usd", MARGIN_USD)
        self._leverage = cfg.get("leverage", LEVERAGE)
        self._min_p1_conf = cfg.get("min_phase1_confidence", MIN_PHASE1_CONFIDENCE)
        self._min_p2_score = cfg.get("min_phase2_score", MIN_PHASE2_SCORE)
        self._dry_run = cfg.get("dry_run", False)

        self._event_bus = event_bus
        self._running = False

        self._rss = RSSFetcher()
        self._llm = LLMClient(
            api_key=cfg.get("llm_api_key", ""),
            base_url=cfg.get("llm_base_url", ""),
            model=cfg.get("llm_model", ""),
        )
        self._ledger = PaperLedger(path=ledger_path)
        self._exchange: Any = None

    # ── Market data via ccxt ──────────────────────────────────────────────

    async def _get_exchange(self) -> Any:
        if self._exchange is None and _CCXT:
            self._exchange = ccxt_async.binance({"enableRateLimit": True})
        return self._exchange

    async def _fetch_snapshot(self, symbol: str) -> MarketSnapshot | None:
        exchange = await self._get_exchange()
        if not exchange:
            return None
        try:
            ticker = await exchange.fetch_ticker(symbol)
            ohlcv = await exchange.fetch_ohlcv(symbol, "1m", limit=20)

            price = float(ticker.get("last", 0))
            pct_24h = float(ticker.get("percentage", 0) or 0)
            volume = float(ticker.get("quoteVolume", 0) or 0)

            pct_1h = 0.0
            pct_15m = 0.0
            pct_5m = 0.0

            if ohlcv and len(ohlcv) >= 2:
                closes = [c[4] for c in ohlcv if c[4]]
                if len(closes) >= 15:
                    pct_15m = (closes[-1] - closes[-15]) / closes[-15] * 100
                if len(closes) >= 5:
                    pct_5m = (closes[-1] - closes[-5]) / closes[-5] * 100
                if len(closes) >= 2:
                    pct_1h = pct_15m * 4  # approximate from 15m

            return MarketSnapshot(
                price=price, pct_1h=pct_1h, pct_24h=pct_24h,
                pct_15m=pct_15m, pct_5m=pct_5m, volume_24h=volume,
            )
        except Exception as exc:
            logger.warning("Failed to fetch snapshot for {}: {}", symbol, exc)
            return None

    # ── Phase 1B: keyword scoring + market assignment ─────────────────────

    def _score_items(self, items: list[RSSItem]) -> list[ScoredItem]:
        scored: list[ScoredItem] = []
        for item in items:
            text = f"{item.title} {item.summary}"
            best_market: MarketConfig | None = None
            best_score = 0
            for market in ALL_MARKETS:
                s = market.relevance_score(text)
                if s > best_score:
                    best_score = s
                    best_market = market
            if best_market and best_score >= MIN_RELEVANCE:
                scored.append(ScoredItem(item=item, market=best_market, relevance=best_score))
        scored.sort(key=lambda s: s.relevance, reverse=True)
        return scored

    # ── V1 Quality Gates ──────────────────────────────────────────────────

    def _gate_duplicate(self, item: RSSItem, direction: str, market: MarketConfig) -> bool:
        if self._ledger.was_recently_traded(item.signal_uid):
            return False
        if self._ledger.has_open_for(market.symbol, direction):
            return False
        return True

    def _gate_max_concurrent(self) -> bool:
        return self._ledger.get_open_count() < self._max_concurrent

    async def _gate_momentum(
        self, market: MarketConfig, direction: str, confidence: int,
    ) -> bool:
        snap = await self._fetch_snapshot(market.symbol)
        if snap is None:
            return True  # fail-open

        if confidence >= 80:
            threshold = MOMENTUM_THRESHOLDS["high"]
        elif confidence >= 60:
            threshold = MOMENTUM_THRESHOLDS["mid"]
        else:
            threshold = MOMENTUM_THRESHOLDS["low"]

        if direction == "LONG":
            return snap.pct_15m >= threshold
        else:
            return snap.pct_15m <= -threshold

    # ── Full pipeline for one scan cycle ──────────────────────────────────

    async def _scan_once(self) -> None:
        items = await self._rss.fetch_all()
        if not items:
            return

        scored = self._score_items(items)
        logger.debug("Geo scan: {} RSS items, {} relevant", len(items), len(scored))

        for si in scored:
            market = si.market
            item = si.item

            if not market.is_market_open():
                logger.debug("Market closed for {}: {}", market.name, item.title[:60])
                continue

            # Phase 1C: LLM sentiment
            sentiment = await self._llm.phase1_sentiment(
                title=item.title,
                summary=item.summary,
                source=item.source,
                asset_context=market.name,
                sentiment_rules=market.sentiment_rules,
            )
            if not sentiment:
                continue
            if sentiment.direction == "SKIP" or sentiment.confidence < self._min_p1_conf:
                logger.debug(
                    "Phase1 skip: {} conf={} dir={}",
                    item.title[:50], sentiment.confidence, sentiment.direction,
                )
                continue

            direction = sentiment.direction

            # V1 gates
            if not self._gate_duplicate(item, direction, market):
                logger.debug("Gate: duplicate/open for {}", item.title[:50])
                continue
            if not self._gate_max_concurrent():
                logger.debug("Gate: max concurrent reached")
                continue
            if not await self._gate_momentum(market, direction, sentiment.confidence):
                logger.debug("Gate: momentum failed for {}", item.title[:50])
                continue

            # Phase 2: LLM price confirmation
            snap = await self._fetch_snapshot(market.symbol)
            if snap is None:
                continue

            confirm = await self._llm.phase2_confirm(
                direction=direction,
                asset_name=market.name,
                asset_context=market.name,
                title=item.title,
                phase1_reasoning=sentiment.reasoning,
                price=snap.price,
                pct_1h=snap.pct_1h,
                pct_24h=snap.pct_24h,
                pct_15m=snap.pct_15m,
                pct_5m=snap.pct_5m,
                volume=snap.volume_24h,
            )
            if not confirm or confirm.verdict != "CONFIRM" or confirm.score < self._min_p2_score:
                logger.debug(
                    "Phase2 reject: {} verdict={} score={}",
                    item.title[:50],
                    confirm.verdict if confirm else "none",
                    confirm.score if confirm else 0,
                )
                continue

            # Phase 3: LLM trade planner
            plan = await self._llm.phase3_plan(
                asset_context=market.name,
                title=item.title,
                direction=direction,
                conf_1=sentiment.confidence,
                conf_2=confirm.score,
                price=snap.price,
                pct_15m=snap.pct_15m,
                pct_24h=snap.pct_24h,
                volume=snap.volume_24h,
            )
            if not plan or plan.skip:
                logger.debug("Phase3 skip: {}", item.title[:50])
                continue

            # Execute
            if self._dry_run:
                logger.info(
                    "[DRY-RUN] Would open {} {} @ {:.2f} SL={:.1f}% TP={:.1f}%",
                    direction, market.symbol, snap.price, plan.sl_pct, plan.tp_pct,
                )
                continue

            self._open_paper_trade(
                market=market, item=item, direction=direction,
                sentiment=sentiment, confirm=confirm, plan=plan,
                price=snap.price,
            )

    def _open_paper_trade(
        self, market: MarketConfig, item: RSSItem, direction: str,
        sentiment: SentimentResult, confirm: ConfirmationResult,
        plan: TradePlan, price: float,
    ) -> PaperTrade:
        notional = self._margin_usd * self._leverage
        qty = notional / price

        if direction == "LONG":
            sl = price * (1 - plan.sl_pct / 100)
            tp = price * (1 + plan.tp_pct / 100)
        else:
            sl = price * (1 + plan.sl_pct / 100)
            tp = price * (1 - plan.tp_pct / 100)

        trade = PaperTrade(
            trade_id=PaperLedger.make_trade_id(),
            signal_uid=item.signal_uid,
            opened_at=datetime.now(tz=timezone.utc).isoformat(),
            symbol=market.symbol,
            direction=direction,
            entry_price=price,
            sl_price=round(sl, 6),
            tp_price=round(tp, 6),
            sl_pct=plan.sl_pct,
            tp_pct=plan.tp_pct,
            timeout_minutes=plan.timeout_minutes,
            margin_usd=self._margin_usd,
            leverage=self._leverage,
            notional_usd=notional,
            qty=round(qty, 6),
            headline=item.title[:300],
            source=item.source,
            news_confidence=sentiment.confidence,
            phase2_verdict=confirm.verdict,
            phase2_score=confirm.score,
            phase2_reasoning=confirm.reasoning[:300],
            plan_confidence=plan.confidence,
            plan_reasoning=plan.reasoning[:300],
        )
        self._ledger.open_trade(trade)

        if self._event_bus:
            try:
                self._event_bus.publish_nowait("GEO_TRADE_OPENED", {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "price": trade.entry_price,
                    "headline": trade.headline[:100],
                })
            except Exception:
                pass

        return trade

    # ── Manage open trades ────────────────────────────────────────────────

    async def _manage_open_trades(self) -> None:
        open_trades = self._ledger.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            snap = await self._fetch_snapshot(trade.symbol)
            if snap is None:
                continue

            update_protective_stops(trade, snap.price)

            exit_reason, exit_price = check_exits(
                trade, snap.price, move_5m_pct=snap.pct_5m,
            )
            if exit_reason and exit_price is not None:
                closed = self._ledger.close_trade(trade.trade_id, exit_price, exit_reason)
                if closed and self._event_bus:
                    try:
                        self._event_bus.publish_nowait("GEO_TRADE_CLOSED", {
                            "trade_id": closed.trade_id,
                            "symbol": closed.symbol,
                            "exit_reason": closed.exit_reason,
                            "pnl_pct": closed.pnl_pct,
                            "pnl_usd": closed.pnl_usd,
                        })
                    except Exception:
                        pass

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        logger.info(
            "GeoPolStrategy started — scan_interval={}s, max_concurrent={}, margin=${}",
            self._scan_interval, self._max_concurrent, self._margin_usd,
        )

        while self._running:
            try:
                await self._scan_once()
            except Exception as exc:
                logger.error("Geo scan error: {}", exc)

            try:
                await self._manage_open_trades()
            except Exception as exc:
                logger.error("Geo trade management error: {}", exc)

            await asyncio.sleep(self._scan_interval)

    async def stop(self) -> None:
        self._running = False
        await self._rss.close()
        await self._llm.close()
        if self._exchange:
            await self._exchange.close()
        logger.info("GeoPolStrategy stopped. Stats: {}", self._ledger.get_stats())
