from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


BLOCKING_STATUSES = {"BLOCK", "BLOCKED", "FAIL", "FAILED", "ERROR", "UNAVAILABLE"}
WARNING_STATUSES = {"WARN", "WARNING", "WARMING", "WEAK", "SOFT", "PENDING", "UNKNOWN", ""}

DEFAULT_ENTRY_LAYERS = (
    "market_data_integrity",
    "preflight",
    "session_filter",
    "htf_trend",
    "technical_confluence",
    "smart_money_concepts",
    "volume_flow",
    "regime_detection",
    "signal_quality",
)


@dataclass(slots=True)
class TradeDecisionReceipt:
    """Audit record explaining why a candidate entry is allowed or blocked."""

    receipt_id: str
    ts: int
    mode: str
    allowed: bool
    decision: str
    reason: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, Any] = field(default_factory=dict)
    signal: dict[str, Any] = field(default_factory=dict)
    ai: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "ts": self.ts,
            "mode": self.mode,
            "allowed": self.allowed,
            "decision": self.decision,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "thresholds": dict(self.thresholds),
            "quality": dict(self.quality),
            "layers": dict(self.layers),
            "signal": dict(self.signal),
            "ai": dict(self.ai),
            "risk": dict(self.risk),
            "latency": dict(self.latency),
            "provenance": dict(self.provenance),
        }


class EntryEligibilityGate:
    """Single entry gate used by signal generation, dashboard readiness, and AI-direct mode."""

    def __init__(self, hard_layers: tuple[str, ...] = DEFAULT_ENTRY_LAYERS) -> None:
        self.hard_layers = tuple(hard_layers)

    def evaluate(
        self,
        *,
        mode: str,
        symbol: str | None = None,
        exchange: str | None = None,
        direction: str | None = None,
        layers: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        signal: Any | None = None,
        ai: dict[str, Any] | None = None,
        risk: dict[str, Any] | None = None,
        latency: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        strict_layers: bool = True,
    ) -> TradeDecisionReceipt:
        mode_name = str(mode or "paper").lower()
        layer_status = self._normalize_layers(layers or {})
        threshold_payload = dict(thresholds or {})
        quality_payload = dict(quality or {})
        signal_payload = self._signal_snapshot(signal, symbol=symbol, exchange=exchange, direction=direction)
        ai_payload = dict(ai or {})
        risk_payload = dict(risk or {})
        latency_payload = dict(latency or {})
        provenance_payload = dict(provenance or {})
        provenance_payload["strict_layers"] = bool(strict_layers)
        provenance_payload.setdefault("contract", "EntryEligibilityGate")

        blockers: list[str] = []
        warnings: list[str] = []

        quality_total = self._as_float(
            quality_payload.get("total", signal_payload.get("quality_score", signal_payload.get("l8_quality", 0)))
        )
        quality_threshold = self._as_float(threshold_payload.get("quality", 65))
        quality_payload.setdefault("total", quality_total)
        if quality_total < quality_threshold:
            blockers.append(f"quality:{quality_total:.0f}<{quality_threshold:.0f}")

        for layer_name in self.hard_layers:
            status = str(layer_status.get(layer_name, "UNKNOWN") or "UNKNOWN").upper()
            if status in BLOCKING_STATUSES:
                if strict_layers or status in {"BLOCK", "BLOCKED", "ERROR", "UNAVAILABLE"}:
                    blockers.append(f"layer:{layer_name}:{status}")
                else:
                    warnings.append(f"layer:{layer_name}:{status}")
            elif status in WARNING_STATUSES:
                if strict_layers and status in {"PENDING", "UNKNOWN", ""}:
                    blockers.append(f"layer:{layer_name}:{status or 'UNKNOWN'}")
                else:
                    warnings.append(f"layer:{layer_name}:{status or 'UNKNOWN'}")

        self._evaluate_risk(risk_payload, blockers, warnings)
        self._evaluate_ai(ai_payload, blockers, warnings)

        allowed = not blockers
        decision = "ENTRY_ALLOWED" if allowed else "NO_ENTRY"
        reason = "entry_allowed" if allowed else blockers[0]
        ts = int(time.time())
        receipt_seed = {
            "ts": ts,
            "mode": mode_name,
            "symbol": signal_payload.get("symbol"),
            "exchange": signal_payload.get("exchange"),
            "direction": signal_payload.get("direction"),
            "decision": decision,
            "reason": reason,
            "blockers": blockers,
            "warnings": warnings,
            "quality": quality_payload,
            "layers": layer_status,
            "risk_state": risk_payload.get("trading_state"),
            "ai_mode": ai_payload.get("effective_mode", ai_payload.get("mode")),
        }
        receipt_id = hashlib.sha256(
            json.dumps(receipt_seed, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

        return TradeDecisionReceipt(
            receipt_id=receipt_id,
            ts=ts,
            mode=mode_name,
            allowed=allowed,
            decision=decision,
            reason=reason,
            blockers=blockers,
            warnings=warnings,
            thresholds=threshold_payload,
            quality=quality_payload,
            layers=layer_status,
            signal=signal_payload,
            ai=ai_payload,
            risk=risk_payload,
            latency=latency_payload,
            provenance=provenance_payload,
        )

    @staticmethod
    def _normalize_layers(layers: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in layers.items():
            key_name = str(key)
            if key_name.endswith("_detail"):
                continue
            normalized[key_name] = str(value or "UNKNOWN").upper()
            detail_key = f"{key_name}_detail"
            if detail_key in layers:
                normalized[detail_key] = str(layers.get(detail_key) or "")
        return normalized

    @staticmethod
    def _signal_snapshot(
        signal: Any | None,
        *,
        symbol: str | None,
        exchange: str | None,
        direction: str | None,
    ) -> dict[str, Any]:
        if signal is None:
            return {
                "symbol": symbol or "",
                "exchange": exchange or "",
                "direction": direction or "",
                "candidate": False,
            }
        metadata = getattr(signal, "metadata", {}) or {}
        snapshot = {
            "symbol": getattr(signal, "symbol", symbol or ""),
            "exchange": getattr(signal, "exchange", exchange or ""),
            "direction": getattr(signal, "direction", direction or ""),
            "price": getattr(signal, "price", None),
            "stop_loss": getattr(signal, "stop_loss", None),
            "take_profit": getattr(signal, "take_profit", None),
            "confidence": getattr(signal, "confidence", None),
            "size_multiplier": getattr(signal, "size_multiplier", None),
            "quality_score": metadata.get("quality_score", metadata.get("l8_quality")),
            "signal_type": metadata.get("signal_type"),
            "regime_class": metadata.get("regime_class"),
            "candidate": True,
        }
        return {key: value for key, value in snapshot.items() if value is not None}

    @staticmethod
    def _evaluate_risk(risk: dict[str, Any], blockers: list[str], warnings: list[str]) -> None:
        if not risk:
            warnings.append("risk:unavailable")
            return
        trading_state = str(risk.get("trading_state", "UNKNOWN") or "UNKNOWN").upper()
        if trading_state in {"HALTED", "REDUCING", "BLOCKED"}:
            blockers.append(f"risk:trading_state:{trading_state}")
        if risk.get("can_open_new_positions") is False:
            blockers.append("risk:cannot_open_new_positions")
        if bool(risk.get("kill_switch_active", False) or risk.get("killed", False)):
            blockers.append("risk:kill_switch")
        if bool(risk.get("circuit_breaker_tripped", False)):
            blockers.append("risk:circuit_breaker")
        if bool(risk.get("safe_mode_active", False) or risk.get("safe_mode", False)):
            blockers.append("risk:safe_mode")

    @staticmethod
    def _evaluate_ai(ai: dict[str, Any], blockers: list[str], warnings: list[str]) -> None:
        if not ai:
            warnings.append("ai:unavailable")
            return
        effective_mode = str(ai.get("effective_mode", ai.get("mode", "advisory")) or "advisory").lower()
        approved = ai.get("approved")
        if effective_mode in {"direct", "full"} and approved is False:
            blockers.append("ai:direct_rejected")
        elif approved is False:
            warnings.append("ai:advisory_rejected")

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
