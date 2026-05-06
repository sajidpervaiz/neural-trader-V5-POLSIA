"""Alerting module — Prompt 10.

Centralized alert manager that routes critical trading events to
Telegram, Discord, and/or generic webhooks with per-alert throttling
to prevent spam.

Event-bus integration:  AlertDispatcher subscribes to events from the
EventBus and converts them into Alert objects that flow through
AlertManager → channels.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from loguru import logger

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False


# ═══════════════════════════════════════════════════════════════════════════
# Alert domain
# ═══════════════════════════════════════════════════════════════════════════

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_BREAKER = "drawdown_breaker"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    USER_STREAM_DISCONNECT = "user_stream_disconnect"
    DB_FAILURE = "db_failure"
    ORDER_REJECTION = "order_rejection"
    KILL_SWITCH = "kill_switch"
    SAFE_MODE_ACTIVATED = "safe_mode_activated"
    SAFE_MODE_CLEARED = "safe_mode_cleared"
    SL_PLACEMENT_FAILED = "sl_placement_failed"
    POSITION_CLOSED = "position_closed"
    CUSTOM = "custom"


@dataclass
class Alert:
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_telegram_html(self) -> str:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
            self.severity.value, "📢"
        )
        lines = [
            f"{icon} <b>{self.title}</b>",
            f"Severity: {self.severity.value.upper()}",
            self.message,
        ]
        if self.metadata:
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_discord_embed(self) -> dict[str, Any]:
        color = {"info": 0x3498DB, "warning": 0xF39C12, "critical": 0xE74C3C}.get(
            self.severity.value, 0x95A5A6
        )
        fields = [{"name": k, "value": str(v), "inline": True}
                  for k, v in self.metadata.items()]
        return {
            "embeds": [{
                "title": self.title,
                "description": self.message,
                "color": color,
                "fields": fields,
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
                ),
            }]
        }


# ═══════════════════════════════════════════════════════════════════════════
# Alert channels (protocol + implementations)
# ═══════════════════════════════════════════════════════════════════════════

class AlertChannel(Protocol):
    async def send(self, alert: Alert) -> bool: ...


class TelegramAlertChannel:
    """Send alerts to Telegram via Bot API."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = chat_id
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"

    async def send(self, alert: Alert) -> bool:
        if not _HTTPX:
            logger.debug("httpx not available — Telegram alert skipped")
            return False
        payload = {
            "chat_id": self._chat_id,
            "text": alert.to_telegram_html(),
            "parse_mode": "HTML",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._url, json=payload)
                if resp.status_code == 200:
                    return True
                logger.warning("Telegram alert failed: {} {}", resp.status_code, resp.text[:200])
                return False
        except Exception as exc:
            logger.warning("Telegram alert error: {}", exc)
            return False


class DiscordAlertChannel:
    """Send alerts to a Discord webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def send(self, alert: Alert) -> bool:
        if not _HTTPX:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._url, json=alert.to_discord_embed())
                if resp.status_code in (200, 204):
                    return True
                logger.warning("Discord alert failed: {} {}", resp.status_code, resp.text[:200])
                return False
        except Exception as exc:
            logger.warning("Discord alert error: {}", exc)
            return False


class WebhookAlertChannel:
    """Send alerts to a generic webhook as JSON POST."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}

    async def send(self, alert: Alert) -> bool:
        if not _HTTPX:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._url, json=alert.to_dict(), headers=self._headers,
                )
                return resp.status_code in range(200, 300)
        except Exception as exc:
            logger.warning("Webhook alert error: {}", exc)
            return False


class LogAlertChannel:
    """Log-based alert channel (always available, used as fallback)."""

    async def send(self, alert: Alert) -> bool:
        level = {"info": "info", "warning": "warning", "critical": "critical"}.get(
            alert.severity.value, "info"
        )
        getattr(logger, level)(
            "[ALERT] {} | {} | {}", alert.title, alert.severity.value, alert.message,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════
# AlertManager — core routing + throttling
# ═══════════════════════════════════════════════════════════════════════════

class AlertManager:
    """Routes alerts to configured channels with per-type throttling.

    Throttle window prevents the same AlertType from spamming channels
    more frequently than ``default_throttle_seconds``.
    """

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        default_throttle_seconds: float = 300.0,
        throttle_overrides: dict[AlertType, float] | None = None,
    ) -> None:
        self._channels: list[AlertChannel] = channels or [LogAlertChannel()]
        self._default_throttle = default_throttle_seconds
        self._throttle_overrides = throttle_overrides or {}
        self._last_sent: dict[AlertType, float] = {}
        self._history: list[Alert] = []
        self._suppressed_count: int = 0

    @property
    def channels(self) -> list[AlertChannel]:
        return list(self._channels)

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)

    def _is_throttled(self, alert_type: AlertType) -> bool:
        window = self._throttle_overrides.get(alert_type, self._default_throttle)
        last = self._last_sent.get(alert_type, 0.0)
        return (time.time() - last) < window

    async def send(self, alert: Alert) -> bool:
        """Send alert through all channels.  Returns True if at least one succeeded.
        Returns False (without sending) if throttled."""
        if self._is_throttled(alert.alert_type):
            self._suppressed_count += 1
            logger.debug(
                "Alert throttled: {} (suppressed={})",
                alert.alert_type.value, self._suppressed_count,
            )
            return False

        self._last_sent[alert.alert_type] = time.time()
        self._history.append(alert)
        # Keep bounded history
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        any_ok = False
        for ch in self._channels:
            try:
                ok = await ch.send(alert)
                if ok:
                    any_ok = True
            except Exception as exc:
                logger.warning("Alert channel {} failed: {}", type(ch).__name__, exc)
        return any_ok

    def reset_throttle(self, alert_type: AlertType | None = None) -> None:
        """Clear throttle for a specific type or all types."""
        if alert_type:
            self._last_sent.pop(alert_type, None)
        else:
            self._last_sent.clear()

    def get_status(self) -> dict[str, Any]:
        return {
            "channels": len(self._channels),
            "channel_types": [type(c).__name__ for c in self._channels],
            "history_count": len(self._history),
            "suppressed_count": self._suppressed_count,
            "throttle_seconds": self._default_throttle,
            "active_throttles": {
                k.value: round(time.time() - v, 1)
                for k, v in self._last_sent.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — wires EventBus events to AlertManager
# ═══════════════════════════════════════════════════════════════════════════

class AlertDispatcher:
    """Subscribes to critical EventBus events and generates alerts."""

    def __init__(self, event_bus: Any, alert_manager: AlertManager) -> None:
        self.event_bus = event_bus
        self.alert_manager = alert_manager
        self._running = False

    async def _on_stop_loss(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        symbol = data.get("symbol", "?")
        price = data.get("price", 0)
        reason = data.get("reason", "stop_loss")
        await self.alert_manager.send(Alert(
            alert_type=AlertType.STOP_LOSS_HIT,
            severity=AlertSeverity.WARNING,
            title=f"Stop Loss Hit — {symbol}",
            message=f"{reason} triggered at {price:.2f}" if price else f"{reason} triggered",
            metadata={"symbol": symbol, "price": price, "reason": reason},
        ))

    async def _on_take_profit(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        symbol = data.get("symbol", "?")
        price = data.get("price", 0)
        await self.alert_manager.send(Alert(
            alert_type=AlertType.TAKE_PROFIT_HIT,
            severity=AlertSeverity.INFO,
            title=f"Take Profit Hit — {symbol}",
            message=f"TP triggered at {price:.2f}" if price else "TP triggered",
            metadata={"symbol": symbol, "price": price},
        ))

    async def _on_kill_switch(self, payload: Any) -> None:
        await self.alert_manager.send(Alert(
            alert_type=AlertType.KILL_SWITCH,
            severity=AlertSeverity.CRITICAL,
            title="KILL SWITCH ACTIVATED",
            message="Emergency: all positions being closed, new signals blocked.",
            metadata={"payload": str(payload)[:200] if payload else ""},
        ))

    async def _on_alert_critical(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        alert_type_str = data.get("type", "unknown")
        # Map known ALERT_CRITICAL sub-types
        if "sl_placement" in alert_type_str:
            at = AlertType.SL_PLACEMENT_FAILED
        elif "reconciliation" in alert_type_str:
            at = AlertType.RECONCILIATION_MISMATCH
        else:
            at = AlertType.CUSTOM
        await self.alert_manager.send(Alert(
            alert_type=at,
            severity=AlertSeverity.CRITICAL,
            title=f"Critical Alert — {alert_type_str}",
            message=data.get("error", str(data)),
            metadata=data,
        ))

    async def _on_user_stream_lost(self, payload: Any) -> None:
        await self.alert_manager.send(Alert(
            alert_type=AlertType.USER_STREAM_DISCONNECT,
            severity=AlertSeverity.CRITICAL,
            title="User Data Stream Disconnected",
            message="Fill/cancel blind — safe mode activated, no new trades.",
            metadata={"payload": str(payload)[:200] if payload else ""},
        ))

    async def _on_user_stream_connected(self, payload: Any) -> None:
        self.alert_manager.reset_throttle(AlertType.USER_STREAM_DISCONNECT)
        await self.alert_manager.send(Alert(
            alert_type=AlertType.SAFE_MODE_CLEARED,
            severity=AlertSeverity.INFO,
            title="User Data Stream Reconnected",
            message="Stream restored, normal operation resuming.",
        ))

    async def _on_order_rejected(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        await self.alert_manager.send(Alert(
            alert_type=AlertType.ORDER_REJECTION,
            severity=AlertSeverity.WARNING,
            title=f"Order Rejected — {data.get('symbol', '?')}",
            message=f"Exchange rejected order: {data.get('reason', 'unknown')}",
            metadata=data,
        ))

    async def _on_position_closed(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        reason = data.get("reason", "unknown")
        pos = data.get("position")
        symbol = getattr(pos, "symbol", "?") if pos else "?"
        pnl = getattr(pos, "pnl", 0.0) if pos else 0.0
        severity = AlertSeverity.WARNING if reason in ("stop_loss", "exchange_stop_loss", "kill_switch") else AlertSeverity.INFO
        await self.alert_manager.send(Alert(
            alert_type=AlertType.POSITION_CLOSED,
            severity=severity,
            title=f"Position Closed — {symbol}",
            message=f"Reason: {reason}, PnL: {pnl:.2f}",
            metadata={"symbol": symbol, "reason": reason, "pnl": pnl},
        ))

    def subscribe_all(self) -> None:
        """Subscribe to all critical events on the event bus."""
        self.event_bus.subscribe("STOP_LOSS", self._on_stop_loss)
        self.event_bus.subscribe("TAKE_PROFIT", self._on_take_profit)
        self.event_bus.subscribe("KILL_SWITCH", self._on_kill_switch)
        self.event_bus.subscribe("ALERT_CRITICAL", self._on_alert_critical)
        self.event_bus.subscribe("USER_STREAM_LOST", self._on_user_stream_lost)
        self.event_bus.subscribe("USER_STREAM_CONNECTED", self._on_user_stream_connected)
        self.event_bus.subscribe("ORDER_REJECTED", self._on_order_rejected)
        self.event_bus.subscribe("POSITION_CLOSED", self._on_position_closed)
        logger.info("AlertDispatcher subscribed to 8 event types")

    def unsubscribe_all(self) -> None:
        for ev, handler in [
            ("STOP_LOSS", self._on_stop_loss),
            ("TAKE_PROFIT", self._on_take_profit),
            ("KILL_SWITCH", self._on_kill_switch),
            ("ALERT_CRITICAL", self._on_alert_critical),
            ("USER_STREAM_LOST", self._on_user_stream_lost),
            ("USER_STREAM_CONNECTED", self._on_user_stream_connected),
            ("ORDER_REJECTED", self._on_order_rejected),
            ("POSITION_CLOSED", self._on_position_closed),
        ]:
            self.event_bus.unsubscribe(ev, handler)

    async def run(self) -> None:
        self._running = True
        self.subscribe_all()
        logger.info("AlertDispatcher started")
        while self._running:
            await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        self.unsubscribe_all()


# ═══════════════════════════════════════════════════════════════════════════
# Factory: build AlertManager from config
# ═══════════════════════════════════════════════════════════════════════════

def build_alert_manager_from_config(config: Any) -> AlertManager:
    """Build AlertManager with channels based on settings.yaml."""
    channels: list[AlertChannel] = [LogAlertChannel()]

    alert_cfg = config.get_value("monitoring", "alerts") or {}

    # Telegram — try alerts.telegram first, then top-level monitoring.telegram
    tg_cfg = alert_cfg.get("telegram") or config.get_value("monitoring", "telegram") or {}
    if tg_cfg.get("enabled") and tg_cfg.get("token") and tg_cfg.get("chat_id"):
        channels.append(TelegramAlertChannel(
            token=tg_cfg["token"], chat_id=str(tg_cfg["chat_id"]),
        ))
        logger.info("Telegram alert channel configured")

    # Discord
    discord_cfg = alert_cfg.get("discord") or {}
    discord_url = discord_cfg.get("webhook_url") or alert_cfg.get("discord_webhook_url")
    if discord_url:
        channels.append(DiscordAlertChannel(webhook_url=discord_url))
        logger.info("Discord alert channel configured")

    # Generic webhook
    webhook_cfg = alert_cfg.get("webhook") or {}
    webhook_url = webhook_cfg.get("url") or alert_cfg.get("webhook_url")
    if webhook_url:
        channels.append(WebhookAlertChannel(url=webhook_url))
        logger.info("Webhook alert channel configured")

    throttle = float(alert_cfg.get("throttle_seconds", 300))
    overrides: dict[AlertType, float] = {}
    # Critical alerts have shorter throttle
    for at in (AlertType.KILL_SWITCH, AlertType.SL_PLACEMENT_FAILED):
        overrides[at] = min(throttle, 60.0)

    return AlertManager(
        channels=channels,
        default_throttle_seconds=throttle,
        throttle_overrides=overrides,
    )
