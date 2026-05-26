from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from loguru import logger

from core.error_handling import sanitize_exception


@dataclass
class AgentDecision:
    approved: bool
    action: str
    confidence: float
    size_multiplier: float = 1.0
    reason: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradingAIAgent:
    """Supervisory AI layer for trade review.

    The existing risk engine remains the final authority. Advisory mode is the
    default; direct/enforced decisions are gated by SignalGenerator so they can
    only affect execution in risk-gated paper mode.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "advisory",
        min_confidence: float = 0.55,
        min_quality_score: int = 60,
        min_risk_reward: float = 1.0,
        provider: str = "local",
        model: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout_seconds: float = 8.0,
        remote_weight: float = 0.35,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = self._normalize_mode(mode)
        self.min_confidence = float(min_confidence)
        self.min_quality_score = int(min_quality_score)
        self.min_risk_reward = float(min_risk_reward)
        self.provider = self._normalize_provider(provider)
        self.model = str(model or self._default_model_for_provider(self.provider)).strip()
        self.api_key = self._resolve_api_key(self.provider, api_key)
        self.api_url = self._resolve_api_url(self.provider, api_url, self.model)
        self.timeout_seconds = max(1.0, float(timeout_seconds or 8.0))
        self.remote_weight = max(0.0, min(1.0, float(remote_weight or 0.35)))
        self._decision_count = 0
        self._approved_count = 0
        self._rejected_count = 0
        self._last_decision_source = "local"
        self._last_error = ""
        self._cache_ttl_seconds = 15.0
        self._decision_cache: dict[str, tuple[float, AgentDecision]] = {}
        self._last_decision: dict[str, Any] = {
            "approved": True,
            "action": "approve",
            "confidence": 1.0,
            "size_multiplier": 1.0,
            "reason": "agent_initialized",
            "summary": "AI agent attached",
        }

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        raw = str(mode or "advisory").strip().lower().replace("-", "_")
        if raw in {"semi", "semi_auto"}:
            return "semi_auto"
        if raw in {"full", "direct", "ai_direct", "enforced"}:
            return "direct"
        if raw in {"advisory", "semi_auto", "direct"}:
            return raw
        return "advisory"

    @staticmethod
    def _normalize_provider(provider: str | None) -> str:
        raw = str(provider or "local").strip().lower().replace("-", "_")
        if raw in {"anthropic", "claude"}:
            return "claude"
        if raw in {"openai", "chatgpt", "gpt"}:
            return "openai"
        if raw in {"gemini", "google", "google_ai"}:
            return "gemini"
        if raw in {"local", "offline", "none"}:
            return "local"
        return raw or "local"

    @staticmethod
    def _default_model_for_provider(provider: str) -> str:
        defaults = {
            "claude": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            "local": "local-rule-engine",
        }
        return str(defaults.get(provider, defaults["local"]))

    @staticmethod
    def _resolve_api_key(provider: str, api_key: str | None) -> str:
        raw = str(api_key or "").strip()
        if raw and not (raw.startswith("${") and raw.endswith("}")):
            return raw
        env_key = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(provider, "")
        return str(os.getenv(env_key, "")).strip() if env_key else ""

    @classmethod
    def _resolve_api_url(cls, provider: str, api_url: str | None, model: str) -> str:
        if api_url is not None and str(api_url).strip():
            return str(api_url).strip()
        if provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        if provider == "gemini":
            model_name = str(model or cls._default_model_for_provider("gemini")).strip()
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        return "https://api.anthropic.com/v1/messages"

    def _provider_display_name(self) -> str:
        return {
            "claude": "Claude",
            "openai": "OpenAI",
            "gemini": "Gemini",
            "local": "Local",
        }.get(self.provider, self.provider.replace("_", " ").title())

    def configure(
        self,
        *,
        enabled: bool | None = None,
        mode: str | None = None,
        min_confidence: float | None = None,
        min_quality_score: int | None = None,
        min_risk_reward: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        remote_weight: float | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if mode is not None:
            self.mode = self._normalize_mode(mode)
        if min_confidence is not None:
            self.min_confidence = float(min_confidence)
        if min_quality_score is not None:
            self.min_quality_score = int(min_quality_score)
        if min_risk_reward is not None:
            self.min_risk_reward = float(min_risk_reward)
        provider_changed = False
        if provider is not None:
            self.provider = self._normalize_provider(provider)
            provider_changed = True
        if model is not None:
            self.model = str(model).strip() or self._default_model_for_provider(self.provider)
        elif provider_changed:
            self.model = self._default_model_for_provider(self.provider)
        if api_key is not None or provider_changed:
            self.api_key = self._resolve_api_key(self.provider, api_key if api_key is not None else self.api_key)
        if timeout_seconds is not None:
            self.timeout_seconds = max(1.0, float(timeout_seconds))
        if remote_weight is not None:
            self.remote_weight = max(0.0, min(1.0, float(remote_weight)))
        self.api_url = self._resolve_api_url(self.provider, None, self.model)

    def get_status(self) -> dict[str, Any]:
        return {
            "attached": True,
            "enabled": self.enabled,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "api_configured": bool(self.api_key),
            "remote_enabled": self._remote_enabled(),
            "remote_reason": self._remote_reason(),
            "timeout_seconds": self.timeout_seconds,
            "min_confidence": self.min_confidence,
            "min_quality_score": self.min_quality_score,
            "min_risk_reward": self.min_risk_reward,
            "decision_count": self._decision_count,
            "approved_count": self._approved_count,
            "rejected_count": self._rejected_count,
            "last_decision_source": self._last_decision_source,
            "last_error": self._last_error,
            "last_decision": dict(self._last_decision),
        }

    async def review_signal(self, signal: Any) -> AgentDecision:
        if not self.enabled:
            return self._finalize(
                AgentDecision(
                    approved=True,
                    action="approve",
                    confidence=1.0,
                    size_multiplier=1.0,
                    reason="agent_disabled",
                    summary="AI review bypassed",
                ),
                source="local",
            )

        local_decision = self._local_review(signal)
        if not local_decision.approved or not self._remote_enabled():
            return self._finalize(local_decision, source="local")

        cache_key = self._make_cache_key(signal, local_decision)
        cached = self._cache_get(cache_key)
        if cached is not None:
            merged = self._merge_decisions(local_decision, cached)
            return self._finalize(merged, source=f"{self.provider}_cache")

        remote_decision: AgentDecision | None = None
        try:
            remote_decision = await self._review_with_remote(signal, local_decision)
        except Exception as exc:
            self._last_error = sanitize_exception(exc)[:240]
            logger.warning("{} supervisory review failed — falling back to local agent: {}", self._provider_display_name(), exc)

        if remote_decision is None:
            return self._finalize(local_decision, source="local")

        self._cache_set(cache_key, remote_decision)
        merged = self._merge_decisions(local_decision, remote_decision)
        return self._finalize(merged, source=self.provider)

    def _local_review(self, signal: Any) -> AgentDecision:
        quality_score = float(getattr(signal, "quality_score", 0.0) or 0.0)
        score_strength = min(1.0, max(0.0, abs(float(getattr(signal, "score", 0.0) or 0.0))))
        risk_reward = float(
            getattr(signal, "metadata", {}).get("risk_reward", getattr(signal, "risk_reward", 0.0)) or 0.0
        )
        technical = float(getattr(signal, "technical_score", 0.0) or 0.0)
        ml_score = float(getattr(signal, "ml_score", 0.0) or 0.0)
        direction = str(getattr(signal, "direction", "long") or "long").lower()
        reasons = list(getattr(signal, "reasons", []) or [])
        volume_flow = float(getattr(signal, "metadata", {}).get("volume_flow_score", 0.0) or 0.0)
        smc_score = float(getattr(signal, "metadata", {}).get("smc_score", 0.0) or 0.0)

        direction_sign = 1.0 if direction == "long" else -1.0
        alignment_raw = ((technical * direction_sign) + (ml_score * direction_sign)) / 2.0
        alignment_score = max(0.0, min(1.0, (alignment_raw + 1.0) / 2.0))
        rr_score = max(0.0, min(1.0, risk_reward / 2.0))
        reason_score = max(0.0, min(1.0, len(reasons) / 5.0))
        structure_score = max(0.0, min(1.0, (max(volume_flow, 0.0) + max(smc_score, 0.0)) / 200.0))

        confidence = (
            (quality_score / 100.0) * 0.40
            + score_strength * 0.20
            + alignment_score * 0.20
            + rr_score * 0.10
            + reason_score * 0.05
            + structure_score * 0.05
        )
        confidence = max(0.0, min(1.0, confidence))

        if (
            quality_score < max(45, self.min_quality_score - 15)
            or risk_reward < 0.9
            or confidence < (self.min_confidence * 0.85)
        ):
            return AgentDecision(
                approved=False,
                action="reject",
                confidence=confidence,
                size_multiplier=0.0,
                reason="weak_setup",
                summary=f"Rejected: Q={quality_score:.0f}, RR={risk_reward:.2f}, conf={confidence:.2f}",
            )

        if quality_score < self.min_quality_score or risk_reward < self.min_risk_reward or confidence < self.min_confidence:
            action = "reduce_size" if self.mode == "direct" else "approve"
            return AgentDecision(
                approved=True,
                action=action,
                confidence=confidence,
                size_multiplier=0.5 if self.mode == "direct" else 1.0,
                reason="borderline_setup",
                summary=f"Borderline but tradable: Q={quality_score:.0f}, RR={risk_reward:.2f}, conf={confidence:.2f}",
            )

        return AgentDecision(
            approved=True,
            action="approve",
            confidence=confidence,
            size_multiplier=1.0,
            reason="approved",
            summary=f"Approved: Q={quality_score:.0f}, RR={risk_reward:.2f}, conf={confidence:.2f}",
        )

    def _remote_enabled(self) -> bool:
        return self.enabled and self.provider in {"claude", "openai", "gemini"} and bool(self.api_key)

    def _remote_reason(self) -> str:
        if not self.enabled:
            return "AI agent is disabled."
        if self.provider == "local":
            return "Local provider selected."
        provider_name = self._provider_display_name()
        if not self.api_key:
            return f"Add your {provider_name} API key in Settings to enable live remote supervision."
        return f"{provider_name} remote supervision is active."

    def _make_cache_key(self, signal: Any, local_decision: AgentDecision) -> str:
        symbol = str(getattr(signal, "symbol", ""))
        direction = str(getattr(signal, "direction", ""))
        score = round(float(getattr(signal, "score", 0.0) or 0.0), 4)
        return f"{symbol}|{direction}|{score}|{local_decision.reason}"

    def _cache_get(self, key: str) -> AgentDecision | None:
        entry = self._decision_cache.get(key)
        if not entry:
            return None
        expires_at, decision = entry
        if expires_at < time.monotonic():
            self._decision_cache.pop(key, None)
            return None
        return decision

    def _cache_set(self, key: str, decision: AgentDecision) -> None:
        self._decision_cache[key] = (time.monotonic() + self._cache_ttl_seconds, decision)

    def _merge_decisions(self, local: AgentDecision, remote: AgentDecision) -> AgentDecision:
        approved = bool(local.approved and remote.approved)
        if not approved:
            return AgentDecision(
                approved=False,
                action="reject",
                confidence=min(local.confidence, remote.confidence),
                size_multiplier=0.0,
                reason=remote.reason or local.reason or "remote_reject",
                summary=remote.summary or local.summary or "Rejected by supervisory AI",
            )

        size_multiplier = min(max(0.1, local.size_multiplier), max(0.1, remote.size_multiplier))
        action = "reduce_size" if size_multiplier < 0.999 or remote.action == "reduce_size" or local.action == "reduce_size" else "approve"
        confidence = max(0.0, min(1.0, (local.confidence * (1.0 - self.remote_weight)) + (remote.confidence * self.remote_weight)))
        return AgentDecision(
            approved=True,
            action=action,
            confidence=confidence,
            size_multiplier=size_multiplier,
            reason=remote.reason or local.reason or "approved",
            summary=remote.summary or local.summary or "Approved by AI supervision",
        )

    async def chat(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Interactive assistant reply for dashboard chat and bot supervision."""
        prompt = str(message or "").strip()
        if not prompt:
            return {"success": False, "provider": self.provider, "reply": "Please enter a message."}

        context = context or {}
        if self._remote_enabled():
            try:
                reply = await self._chat_with_remote(prompt, context)
                self._last_decision_source = f"{self.provider}_chat"
                return {"success": True, "provider": self.provider, "reply": reply}
            except Exception as exc:
                self._last_error = sanitize_exception(exc)[:240]
                logger.warning("{} chat failed — falling back to local assistant: {}", self._provider_display_name(), exc)

        self._last_decision_source = "local"
        return {"success": True, "provider": "local", "reply": self._local_chat_reply(prompt, context)}

    async def test_connection(self) -> dict[str, Any]:
        """Ping the configured remote provider with a tiny prompt to verify the API key works."""
        provider_name = self._provider_display_name()
        if not self.enabled:
            return {"success": False, "provider": self.provider, "reason": "agent_disabled", "message": "AI agent is disabled."}
        if self.provider == "local":
            return {"success": True, "provider": "local", "reason": "local_ok", "message": "Local rule engine is always available."}
        if not self.api_key:
            return {"success": False, "provider": self.provider, "reason": "missing_api_key", "message": f"{provider_name} API key is not configured."}
        try:
            reply = await self._chat_with_remote("Reply with the single word: OK.", {"healthcheck": True})
        except Exception as exc:
            self._last_error = sanitize_exception(exc)[:240]
            return {"success": False, "provider": self.provider, "reason": "remote_error", "message": f"{provider_name} error: {exc}"[:240]}
        ok = bool(reply and "OK" in reply.upper())
        return {
            "success": ok,
            "provider": self.provider,
            "model": self.model,
            "reason": "remote_ok" if ok else "unexpected_reply",
            "message": (reply or "")[:240] if ok else f"Reply did not contain OK: {(reply or '')[:120]}",
        }

    def _local_chat_reply(self, message: str, context: dict[str, Any]) -> str:
        lower = message.lower()
        auto_on = bool(context.get("auto_trading_enabled", False))
        open_positions = int(context.get("open_positions", 0) or 0)
        ml_loaded = bool(context.get("ml_loaded", False))
        provider = str(self.provider).upper()
        quality_total = int((context.get("quality") or {}).get("total", 0) or 0)

        if any(word in lower for word in ["status", "summary", "health"]):
            return (
                f"Agent is {('enabled' if self.enabled else 'disabled')} in {self.mode} mode. "
                f"Provider is {provider}. Auto trading is {('ON' if auto_on else 'OFF')}, "
                f"open positions: {open_positions}, ML loaded: {ml_loaded}, latest quality score: {quality_total}."
            )
        if "risk" in lower:
            return (
                "Risk-first mode is active. Weak setups are rejected, borderline setups are reduced in size, "
                "and the risk engine still has final authority over execution."
            )
        if any(word in lower for word in ["train", "learning", "learn"]):
            return (
                "Learning is enabled through the model trainer and historical data bootstrap. "
                "Use the Train button to retrain immediately from the latest market history."
            )
        if any(word in lower for word in ["help", "what can you do", "commands"]):
            return (
                "You can ask for bot status, risk summary, learning status, or request training. "
                "You can also use the dashboard settings to connect Claude, OpenAI, or stay in local mode."
            )
        return (
            "I can help monitor the bot, explain signals, summarize risk state, and guide learning and auto-trading settings. "
            "For deeper interaction, connect a supported remote AI provider in Settings."
        )

    async def _review_with_remote(self, signal: Any, local_decision: AgentDecision) -> AgentDecision | None:
        if self.provider == "claude":
            return await self._review_with_claude(signal, local_decision)
        if self.provider == "openai":
            return await self._review_with_openai(signal, local_decision)
        if self.provider == "gemini":
            return await self._review_with_gemini(signal, local_decision)
        return None

    async def _chat_with_remote(self, message: str, context: dict[str, Any]) -> str:
        if self.provider == "claude":
            return await self._chat_with_claude(message, context)
        if self.provider == "openai":
            return await self._chat_with_openai(message, context)
        if self.provider == "gemini":
            return await self._chat_with_gemini(message, context)
        return self._local_chat_reply(message, context)

    def _parse_remote_decision(self, text: str, *, reason_prefix: str, summary_prefix: str) -> AgentDecision:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Remote response did not contain JSON")
        parsed = json.loads(text[start : end + 1])

        approved = bool(parsed.get("approved", True))
        action = str(parsed.get("action", "approve")).strip().lower()
        if action not in {"approve", "reduce_size", "reject"}:
            action = "reject" if not approved else "approve"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5) or 0.5)))
        size_multiplier = float(parsed.get("size_multiplier", 1.0) or 1.0)
        size_multiplier = 0.0 if not approved else max(0.1, min(1.0, size_multiplier))

        return AgentDecision(
            approved=approved,
            action=action,
            confidence=confidence,
            size_multiplier=size_multiplier,
            reason=str(parsed.get("reason", reason_prefix) or reason_prefix)[:80],
            summary=str(parsed.get("summary", summary_prefix) or summary_prefix)[:240],
        )

    def _review_prompt_payload(self, signal: Any, local_decision: AgentDecision) -> dict[str, Any]:
        return {
            "symbol": str(getattr(signal, "symbol", "")),
            "direction": str(getattr(signal, "direction", "")),
            "score": float(getattr(signal, "score", 0.0) or 0.0),
            "quality_score": float(getattr(signal, "quality_score", 0.0) or 0.0),
            "technical_score": float(getattr(signal, "technical_score", 0.0) or 0.0),
            "ml_score": float(getattr(signal, "ml_score", 0.0) or 0.0),
            "risk_reward": float(getattr(signal, "metadata", {}).get("risk_reward", getattr(signal, "risk_reward", 0.0)) or 0.0),
            "volume_flow_score": float(getattr(signal, "metadata", {}).get("volume_flow_score", 0.0) or 0.0),
            "smc_score": float(getattr(signal, "metadata", {}).get("smc_score", 0.0) or 0.0),
            "reasons": list(getattr(signal, "reasons", []) or [])[:8],
            "local_decision": local_decision.to_dict(),
        }

    async def _review_with_claude(self, signal: Any, local_decision: AgentDecision) -> AgentDecision | None:
        prompt = self._review_prompt_payload(signal, local_decision)
        payload = await self._call_claude(
            system=(
                "You are a risk-first crypto trading supervisor. Return JSON only. "
                "Never increase size above 1.0. Use one of approve, reduce_size, reject."
            ),
            user_prompt=(
                "Review this trade candidate. Respond with strict JSON containing "
                "approved, action, confidence, size_multiplier, reason, summary.\n"
                + json.dumps(prompt, separators=(",", ":"))
            ),
            max_tokens=180,
        )

        text = self._extract_text_content(payload)
        if not text:
            return None
        return self._parse_remote_decision(text, reason_prefix="claude_review", summary_prefix="Claude supervisory review")

    async def _review_with_openai(self, signal: Any, local_decision: AgentDecision) -> AgentDecision | None:
        prompt = self._review_prompt_payload(signal, local_decision)
        payload = await self._call_openai(
            system=(
                "You are a risk-first crypto trading supervisor. Respond with JSON only. "
                "Never increase size above 1.0. Use one of approve, reduce_size, reject."
            ),
            user_prompt=(
                "Review this trade candidate. Respond with strict JSON containing "
                "approved, action, confidence, size_multiplier, reason, summary.\n"
                + json.dumps(prompt, separators=(",", ":"))
            ),
            max_tokens=180,
        )
        text = self._extract_openai_text_content(payload)
        if not text:
            return None
        return self._parse_remote_decision(text, reason_prefix="openai_review", summary_prefix="OpenAI supervisory review")

    async def _review_with_gemini(self, signal: Any, local_decision: AgentDecision) -> AgentDecision | None:
        prompt = self._review_prompt_payload(signal, local_decision)
        payload = await self._call_gemini(
            system=(
                "You are a risk-first crypto trading supervisor. Respond with JSON only. "
                "Never increase size above 1.0. Use one of approve, reduce_size, reject."
            ),
            user_prompt=(
                "Review this trade candidate. Respond with strict JSON containing "
                "approved, action, confidence, size_multiplier, reason, summary.\n"
                + json.dumps(prompt, separators=(",", ":"))
            ),
            max_tokens=180,
        )
        text = self._extract_gemini_text_content(payload)
        if not text:
            return None
        return self._parse_remote_decision(text, reason_prefix="gemini_review", summary_prefix="Gemini supervisory review")

    async def _chat_with_claude(self, message: str, context: dict[str, Any]) -> str:
        payload = await self._call_claude(
            system=(
                "You are an embedded trading-bot assistant. Be concise, risk-first, and practical. "
                "Do not promise profits. Use the supplied runtime context."
            ),
            user_prompt=(
                "User message:\n"
                + message
                + "\n\nBot runtime context:\n"
                + json.dumps(context, default=str, separators=(",", ":"))
            ),
            max_tokens=220,
        )
        text = self._extract_text_content(payload)
        return text or "Claude did not return a reply."

    async def _chat_with_openai(self, message: str, context: dict[str, Any]) -> str:
        payload = await self._call_openai(
            system=(
                "You are an embedded trading-bot assistant. Be concise, risk-first, and practical. "
                "Do not promise profits. Use the supplied runtime context."
            ),
            user_prompt=(
                "User message:\n"
                + message
                + "\n\nBot runtime context:\n"
                + json.dumps(context, default=str, separators=(",", ":"))
            ),
            max_tokens=220,
        )
        text = self._extract_openai_text_content(payload)
        return text or "OpenAI did not return a reply."

    async def _chat_with_gemini(self, message: str, context: dict[str, Any]) -> str:
        payload = await self._call_gemini(
            system=(
                "You are an embedded trading-bot assistant. Be concise, risk-first, and practical. "
                "Do not promise profits. Use the supplied runtime context."
            ),
            user_prompt=(
                "User message:\n"
                + message
                + "\n\nBot runtime context:\n"
                + json.dumps(context, default=str, separators=(",", ":"))
            ),
            max_tokens=220,
        )
        text = self._extract_gemini_text_content(payload)
        return text or "Gemini did not return a reply."

    async def _call_claude(self, *, system: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    async def _call_openai(self, *, system: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    async def _call_gemini(self, *, system: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        # Gemini auths via ?key= query string and folds the system prompt into systemInstruction.
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": int(max_tokens)},
        }
        params = {"key": self.api_key}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.api_url, params=params, json=body)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _extract_text_content(payload: dict[str, Any]) -> str:
        content = payload.get("content", [])
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        ).strip()

    @staticmethod
    def _extract_openai_text_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
        return str(content or "").strip()

    @staticmethod
    def _extract_gemini_text_content(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        return "\n".join(
            str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")
        ).strip()

    def _finalize(self, decision: AgentDecision, *, source: str) -> AgentDecision:
        self._last_decision_source = source
        self._remember(decision)
        return decision

    def _remember(self, decision: AgentDecision) -> None:
        self._decision_count += 1
        if decision.approved:
            self._approved_count += 1
        else:
            self._rejected_count += 1
        self._last_decision = decision.to_dict()
