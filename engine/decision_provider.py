from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engine.ai_agent import AgentDecision, TradingAIAgent


@runtime_checkable
class TradingDecisionProvider(Protocol):
    """Common contract for AI, ML, RL, or rules-based trade decision providers."""

    name: str

    async def review_signal(self, signal: Any) -> AgentDecision:
        """Return the final supervisory decision for a generated trade signal."""
        ...

    def configure(self, **kwargs: Any) -> None:
        """Update provider settings at runtime."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Return provider health and recent decision state for the dashboard/API."""
        ...


class AIAgentDecisionProvider:
    """Adapter that exposes TradingAIAgent through the provider contract."""

    name = "ai_agent"

    def __init__(self, agent: TradingAIAgent) -> None:
        self.agent = agent

    async def review_signal(self, signal: Any) -> AgentDecision:
        return await self.agent.review_signal(signal)

    def configure(self, **kwargs: Any) -> None:
        self.agent.configure(**kwargs)

    def get_status(self) -> dict[str, Any]:
        status = self.agent.get_status()
        status["decision_provider"] = self.name
        status["decision_contract"] = "TradingDecisionProvider"
        return status


class LocalPassThroughDecisionProvider:
    """Minimal provider for smoke tests and emergency paper-mode bypasses."""

    name = "local_passthrough"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._decision_count = 0
        self._last_decision: dict[str, Any] = {}

    async def review_signal(self, signal: Any) -> AgentDecision:
        self._decision_count += 1
        decision = AgentDecision(
            approved=bool(self.enabled),
            action="approve" if self.enabled else "reject",
            confidence=1.0 if self.enabled else 0.0,
            size_multiplier=1.0 if self.enabled else 0.0,
            reason="passthrough_enabled" if self.enabled else "passthrough_disabled",
            summary="Local pass-through provider decision",
        )
        self._last_decision = decision.to_dict()
        return decision

    def configure(self, **kwargs: Any) -> None:
        if "enabled" in kwargs and kwargs["enabled"] is not None:
            self.enabled = bool(kwargs["enabled"])

    def get_status(self) -> dict[str, Any]:
        return {
            "attached": True,
            "enabled": self.enabled,
            "mode": "passthrough",
            "provider": "local",
            "decision_provider": self.name,
            "decision_contract": "TradingDecisionProvider",
            "decision_count": self._decision_count,
            "last_decision": dict(self._last_decision),
        }
