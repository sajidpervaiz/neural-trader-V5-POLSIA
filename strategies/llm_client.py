"""LLM client for Phases 1C, 2, and 3 with robust JSON/natural-language parsing."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
import aiohttp
from loguru import logger


@dataclass
class SentimentResult:
    direction: str   # LONG / SHORT / SKIP
    confidence: int  # 0-100
    reasoning: str


@dataclass
class ConfirmationResult:
    verdict: str  # CONFIRM / CONTRADICT / WEAK
    score: int    # 0-100
    reasoning: str


@dataclass
class TradePlan:
    skip: bool
    sl_pct: float
    tp_pct: float
    timeout_minutes: int
    confidence: int
    reasoning: str


# ── Robust JSON / NL parser ───────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_RULE_RESTATE_RE = re.compile(r"\bif\b.*?(?:then|→|=|means)\b[^.]*\.?", re.IGNORECASE)

_VERDICT_KEYWORDS = {"CONFIRM", "CONTRADICT", "WEAK"}
_DIRECTION_KEYWORDS = {"LONG", "SHORT", "SKIP"}

_NL_VERDICT_RE = re.compile(
    r"(?:market\s+|verdict[:\s]+|therefore\s+|=\s*|conclusion[:\s]+)"
    r"(CONFIRM|CONTRADICT|WEAK)",
    re.IGNORECASE,
)
_NL_DIRECTION_RE = re.compile(
    r"(?:direction[:\s]+|therefore\s+|=\s*|conclusion[:\s]+)"
    r"(LONG|SHORT|SKIP)",
    re.IGNORECASE,
)


def _extract_json(text: str) -> dict | None:
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    for m in _JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_nl_verdict(text: str) -> str | None:
    cleaned = _RULE_RESTATE_RE.sub("", text)
    tail = cleaned[-300:] if len(cleaned) > 300 else cleaned
    m = _NL_VERDICT_RE.search(tail)
    return m.group(1).upper() if m else None


def _extract_nl_direction(text: str) -> str | None:
    cleaned = _RULE_RESTATE_RE.sub("", text)
    tail = cleaned[-300:] if len(cleaned) > 300 else cleaned
    m = _NL_DIRECTION_RE.search(tail)
    return m.group(1).upper() if m else None


def parse_sentiment(raw: str) -> SentimentResult | None:
    obj = _extract_json(raw)
    if obj:
        d = str(obj.get("direction", "")).upper()
        if d in _DIRECTION_KEYWORDS:
            return SentimentResult(
                direction=d,
                confidence=int(obj.get("confidence", 0)),
                reasoning=str(obj.get("reasoning", "")),
            )
    d = _extract_nl_direction(raw)
    if d:
        return SentimentResult(direction=d, confidence=50, reasoning="NL-parsed")
    return None


def parse_confirmation(raw: str) -> ConfirmationResult | None:
    obj = _extract_json(raw)
    if obj:
        v = str(obj.get("verdict", "")).upper()
        if v in _VERDICT_KEYWORDS:
            return ConfirmationResult(
                verdict=v,
                score=int(obj.get("score", 0)),
                reasoning=str(obj.get("reasoning", "")),
            )
    v = _extract_nl_verdict(raw)
    if v:
        return ConfirmationResult(verdict=v, score=60, reasoning="NL-parsed")
    return None


def parse_trade_plan(raw: str) -> TradePlan | None:
    obj = _extract_json(raw)
    if not obj:
        return None
    return TradePlan(
        skip=bool(obj.get("skip", True)),
        sl_pct=_clamp(float(obj.get("sl_pct", 1.5)), 0.5, 3.0),
        tp_pct=_clamp(float(obj.get("tp_pct", 3.0)), 0.8, 6.0),
        timeout_minutes=int(_clamp(float(obj.get("timeout_minutes", 30)), 5, 60)),
        confidence=int(obj.get("confidence", 0)),
        reasoning=str(obj.get("reasoning", "")),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── LLM API client ───────────────────────────────────────────────────────────

_PHASE1_TEMPLATE = """You are an expert geopolitical analyst for {asset_context}.

Headline: {title}
Summary:  {summary}
Source:   {source}

{sentiment_rules}

Rate this headline for trading impact:
1. direction: LONG / SHORT / SKIP
2. confidence: 0-100 (how sure are you?)
3. reasoning: ONE sentence, specific to this headline.

Output ONLY valid JSON on a single line:
{{"direction": "LONG", "confidence": 78, "reasoning": "..."}}"""

_PHASE2_TEMPLATE = """You are an expert trader evaluating whether live market action confirms a
news-driven signal on {asset_context}.

Signal: {direction} {asset_name}
News headline: {title}
News reasoning: {phase1_reasoning}

Live market data:
  Current price: ${price}
  1-hour change: {pct_1h:+.2f}%
  24-hour change: {pct_24h:+.2f}%
  15-min change: {pct_15m:+.2f}%
  5-min change:  {pct_5m:+.2f}%
  24h volume:    ${volume:,.0f}

Rules:
  - If direction is LONG and price is RISING (1h > +0.2%): market CONFIRMS
  - If direction is LONG and price is FALLING (< -0.2%): market CONTRADICTS
  - If direction is SHORT and price is FALLING: market CONFIRMS
  - If direction is SHORT and price is RISING: market CONTRADICTS
  - Volume below $50M: market is WEAK (low liquidity)
  - Volume above $50M with aligned price: strong CONFIRM
  - Price flat (±0.2%): market is WEAK

OUTPUT FORMAT (STRICT): Line 1 MUST be a compact JSON object:
{{"verdict": "CONFIRM", "score": 75, "reasoning": "..."}}"""

_PHASE3_TEMPLATE = """You are a risk manager producing a trade plan for a geopolitical news event
on {asset_context}.

Event: {title}
Direction: {direction}
Phase-1 confidence: {conf_1}
Phase-2 score: {conf_2}
Current price: ${price}
15m move: {pct_15m:+.2f}%    24h move: {pct_24h:+.2f}%    Volume: ${volume:,.0f}

Produce a JSON plan:
{{
  "skip": false,
  "sl_pct": 2.2,
  "tp_pct": 4.8,
  "timeout_minutes": 40,
  "confidence": 82,
  "reasoning": "One sentence specific to THIS event."
}}

Examples:
  Hard supply-cut + trend aligned → {{"skip":false, "sl_pct":2.2, "tp_pct":4.8, "timeout_minutes":40, "confidence":82}}
  Soft commentary + counter-trend → {{"skip":true,  "sl_pct":1.0, "tp_pct":1.5, "timeout_minutes":10, "confidence":25}}
  Duplicate / already priced-in  → {{"skip":true,  "sl_pct":0.8, "tp_pct":1.2, "timeout_minutes":8,  "confidence":30}}
  Hard escalation + trending      → {{"skip":false, "sl_pct":2.6, "tp_pct":5.5, "timeout_minutes":50, "confidence":88}}"""


class LLMClient:
    """Calls an OpenAI-compatible API (works with Claude, GPT, MiniMax, etc.)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key or os.getenv("GEO_LLM_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("GEO_LLM_BASE_URL", "")
            or "https://api.openai.com/v1"
        )
        self.model = model or os.getenv("GEO_LLM_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _call(self, prompt: str) -> str:
        if not self.api_key:
            logger.warning("No LLM API key — returning empty response")
            return ""
        session = await self._get_session()
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("LLM API error {}: {}", resp.status, body[:200])
                    return ""
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return choices[0].get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("LLM call failed: {}", exc)
            return ""

    async def phase1_sentiment(
        self, title: str, summary: str, source: str,
        asset_context: str, sentiment_rules: str,
    ) -> SentimentResult | None:
        prompt = _PHASE1_TEMPLATE.format(
            asset_context=asset_context, title=title, summary=summary,
            source=source, sentiment_rules=sentiment_rules,
        )
        raw = await self._call(prompt)
        if not raw:
            return None
        return parse_sentiment(raw)

    async def phase2_confirm(
        self, direction: str, asset_name: str, asset_context: str,
        title: str, phase1_reasoning: str,
        price: float, pct_1h: float, pct_24h: float,
        pct_15m: float, pct_5m: float, volume: float,
    ) -> ConfirmationResult | None:
        prompt = _PHASE2_TEMPLATE.format(
            asset_context=asset_context, direction=direction,
            asset_name=asset_name, title=title,
            phase1_reasoning=phase1_reasoning,
            price=price, pct_1h=pct_1h, pct_24h=pct_24h,
            pct_15m=pct_15m, pct_5m=pct_5m, volume=volume,
        )
        raw = await self._call(prompt)
        if not raw:
            return None
        return parse_confirmation(raw)

    async def phase3_plan(
        self, asset_context: str, title: str, direction: str,
        conf_1: int, conf_2: int, price: float,
        pct_15m: float, pct_24h: float, volume: float,
    ) -> TradePlan | None:
        prompt = _PHASE3_TEMPLATE.format(
            asset_context=asset_context, title=title, direction=direction,
            conf_1=conf_1, conf_2=conf_2, price=price,
            pct_15m=pct_15m, pct_24h=pct_24h, volume=volume,
        )
        raw = await self._call(prompt)
        if not raw:
            return None
        return parse_trade_plan(raw)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
