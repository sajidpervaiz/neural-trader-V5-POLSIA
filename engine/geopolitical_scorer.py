"""Geo-political news scorer — keyword relevance + direction inference per symbol.

Consumes RSS articles via score_article() and accumulates time-decayed events
per symbol. score(symbol) returns a [-1, 1] composite suitable for additive
contribution to the multi-factor signal generator.

Direction inference uses DIRECTION_TOKENS from strategies.geo_political_strategy.
Articles below MIN_RELEVANCE are dropped. The composite decays linearly over
WINDOW_HOURS so a single hot headline does not dominate indefinitely.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Iterable

from loguru import logger

from strategies.geo_political_strategy import (
    DIRECTION_TOKENS,
    MARKET_CONFIGS,
    MIN_RELEVANCE,
    WINDOW_HOURS,
)


@dataclass(frozen=True)
class ScoredEvent:
    symbol: str
    direction: int        # +1 long, -1 short, 0 ambiguous
    relevance: int        # 0–100+
    confidence: float     # 0.0–1.0
    title: str
    source: str
    timestamp: float


def _normalize(text: str) -> str:
    return text.lower()


def _trading_hours_open(rule: dict | None, now: datetime | None = None) -> bool:
    """Return True if symbol's trading window is currently open (None = always open)."""
    if not rule:
        return True
    now = now or datetime.now(timezone.utc)
    start = dtime.fromisoformat(rule["start"])
    end = dtime.fromisoformat(rule["end"])
    weekday = now.strftime("%a").lower()[:3]
    if weekday not in {d.lower()[:3] for d in rule["days"]}:
        return False
    cur = now.timetz().replace(tzinfo=None)
    if start <= end:
        return start <= cur <= end
    # Wraps midnight (oil 23:00 → 22:00 next day)
    return cur >= start or cur <= end


class GeoPoliticalScorer:
    """Per-symbol keyword/direction scorer with time-decayed event window."""

    def __init__(
        self,
        symbols: Iterable[str] | None = None,
        window_hours: float = WINDOW_HOURS,
        min_relevance: int = MIN_RELEVANCE,
        max_events_per_symbol: int = 200,
    ) -> None:
        self._symbols: list[str] = list(symbols) if symbols else list(MARKET_CONFIGS.keys())
        self._window_seconds = float(window_hours) * 3600.0
        self._min_relevance = int(min_relevance)
        self._events: dict[str, deque[ScoredEvent]] = {
            sym: deque(maxlen=max_events_per_symbol) for sym in self._symbols
        }
        # Optional alias map so CL/USDT:USDT-style configs also score against
        # raw "BTC/USDT:USDT" or "BTCUSDT" symbols the bot already trades.
        self._aliases: dict[str, str] = {}

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def add_alias(self, alias: str, canonical: str) -> None:
        """Map a bot-side symbol (e.g. 'BTC/USDT:USDT') to a config key."""
        if canonical in self._symbols:
            self._aliases[alias] = canonical

    def _resolve(self, symbol: str) -> str | None:
        if symbol in self._symbols:
            return symbol
        return self._aliases.get(symbol)

    @staticmethod
    def _relevance(text_lc: str, keywords: dict[str, int]) -> int:
        score = 0
        for token, weight in keywords.items():
            if token in text_lc:
                score += weight
        return score

    @staticmethod
    def _direction_vote(text_lc: str, tokens: dict[str, int]) -> int:
        vote = 0
        for token, sign in tokens.items():
            if token in text_lc:
                vote += sign
        if vote > 0:
            return 1
        if vote < 0:
            return -1
        return 0

    def score_article(
        self,
        title: str,
        body: str = "",
        source: str = "",
        timestamp: float | None = None,
    ) -> list[ScoredEvent]:
        """Score one article against all configured symbols.

        Returns the per-symbol ScoredEvent list (only for symbols that pass
        relevance + trading-hours gates). Caller is expected to ingest() each.
        """
        text_lc = _normalize(f"{title} {body}")
        ts = timestamp if timestamp is not None else time.time()
        out: list[ScoredEvent] = []
        for sym in self._symbols:
            cfg = MARKET_CONFIGS[sym]
            if not _trading_hours_open(cfg.get("trading_hours")):
                continue
            relevance = self._relevance(text_lc, cfg["keywords"])
            if relevance < self._min_relevance:
                continue
            direction = self._direction_vote(text_lc, DIRECTION_TOKENS.get(sym, {}))
            confidence = min(1.0, relevance / 100.0)
            out.append(ScoredEvent(
                symbol=sym,
                direction=direction,
                relevance=relevance,
                confidence=confidence,
                title=title[:200],
                source=source,
                timestamp=ts,
            ))
        return out

    def ingest(self, event: ScoredEvent) -> None:
        if event.symbol not in self._events:
            return
        if event.direction == 0:
            return  # ambiguous events don't push the score in either direction
        self._events[event.symbol].append(event)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        for sym, dq in self._events.items():
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    def score(self, symbol: str, now: float | None = None) -> float:
        """Composite [-1, 1] for the given symbol.

        Each event contributes (direction × confidence × decay_weight) where
        decay_weight is linear in (1 - age/window). Result is normalised by the
        sum of decay weights so a single fresh event yields ~direction*confidence
        rather than being divided by the symbol's max_events capacity.
        """
        canonical = self._resolve(symbol)
        if canonical is None:
            return 0.0
        now = now if now is not None else time.time()
        self._prune(now)
        dq = self._events.get(canonical)
        if not dq:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for ev in dq:
            age = now - ev.timestamp
            decay = max(0.0, 1.0 - age / self._window_seconds)
            if decay <= 0:
                continue
            w = decay * ev.confidence
            total += ev.direction * w
            weight_sum += w
        if weight_sum <= 0:
            return 0.0
        composite = total / weight_sum
        return max(-1.0, min(1.0, composite))

    def event_count(self, symbol: str) -> int:
        canonical = self._resolve(symbol)
        if canonical is None:
            return 0
        return len(self._events.get(canonical, ()))

    def snapshot(self) -> dict[str, dict]:
        """Diagnostic — current per-symbol score + event count."""
        now = time.time()
        out: dict[str, dict] = {}
        for sym in self._symbols:
            out[sym] = {
                "score": self.score(sym, now=now),
                "events": self.event_count(sym),
                "trading_open": _trading_hours_open(MARKET_CONFIGS[sym].get("trading_hours")),
            }
        return out

    def recent_events(self, limit: int = 20) -> list[dict]:
        """Return the most recent scored events across all symbols, newest first."""
        all_events: list[ScoredEvent] = []
        for dq in self._events.values():
            all_events.extend(dq)
        all_events.sort(key=lambda e: e.timestamp, reverse=True)
        return [
            {
                "symbol": ev.symbol,
                "title": ev.title,
                "source": ev.source,
                "direction": ev.direction,
                "relevance": ev.relevance,
                "confidence": ev.confidence,
                "timestamp": ev.timestamp,
            }
            for ev in all_events[: int(max(1, limit))]
        ]


__all__ = ["GeoPoliticalScorer", "ScoredEvent"]
