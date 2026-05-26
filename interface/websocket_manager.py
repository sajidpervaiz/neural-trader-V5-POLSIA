"""
Websocket manager for real-time data broadcasting to dashboard clients.
"""

import asyncio
import json
import secrets
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class Channel(str, Enum):
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    ORDERS = "orders"
    POSITIONS = "positions"
    RISK = "risk"
    SIGNALS = "signals"
    ALERTS = "alerts"
    ALL = "all"


@dataclass
class ClientSubscription:
    client_id: str
    websocket: WebSocket
    channels: Set[Channel] = field(default_factory=set)
    subscribed_at: float = field(default_factory=lambda: __import__('time').time())


class WebsocketManager:
    """
    WebSocket connection manager for real-time dashboard updates.

    Features:
    - Connection pooling
    - Channel-based subscriptions
    - Broadcast to subscribers
    - Heartbeat monitoring
    - Automatic reconnection handling
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        max_connections: int = 100,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.max_connections = max_connections

        self.active_connections: Dict[str, ClientSubscription] = {}
        self.channel_subscribers: Dict[Channel, Set[str]] = defaultdict(set)

        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(
        self,
        websocket: WebSocket,
        channels: Optional[Set[Channel]] = None,
    ) -> str:
        """Accept new WebSocket connection."""
        await websocket.accept()

        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Max connections exceeded")
            raise Exception("Max connections exceeded")

        client_id = f"ws_{secrets.token_urlsafe(16)}"

        subscription = ClientSubscription(
            client_id=client_id,
            websocket=websocket,
            channels=channels or set(),
        )

        self.active_connections[client_id] = subscription

        for channel in subscription.channels:
            self.channel_subscribers[channel].add(client_id)

        logger.info(
            f"WebSocket connected: {client_id}, channels: {[c.value for c in subscription.channels]}"
        )

        return client_id

    async def disconnect(
        self,
        client_id: str,
        code: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Disconnect WebSocket client."""
        if client_id not in self.active_connections:
            return

        subscription = self.active_connections[client_id]

        for channel in subscription.channels:
            self.channel_subscribers[channel].discard(client_id)

        del self.active_connections[client_id]

        logger.info(
            f"WebSocket disconnected: {client_id}, "
            f"code: {code}, reason: {reason}"
        )

    async def subscribe(
        self,
        client_id: str,
        channels: Set[Channel],
    ) -> bool:
        """Subscribe client to additional channels."""
        if client_id not in self.active_connections:
            return False

        subscription = self.active_connections[client_id]

        for channel in channels:
            subscription.channels.add(channel)
            self.channel_subscribers[channel].add(client_id)

        logger.info(
            f"Client {client_id} subscribed to: {[c.value for c in channels]}"
        )

        return True

    async def unsubscribe(
        self,
        client_id: str,
        channels: Set[Channel],
    ) -> bool:
        """Unsubscribe client from channels."""
        if client_id not in self.active_connections:
            return False

        subscription = self.active_connections[client_id]

        for channel in channels:
            subscription.channels.discard(channel)
            self.channel_subscribers[channel].discard(client_id)

        logger.info(
            f"Client {client_id} unsubscribed from: {[c.value for c in channels]}"
        )

        return True

    async def send_personal_message(
        self,
        client_id: str,
        message: Any,
    ) -> bool:
        """Send message to specific client."""
        if client_id not in self.active_connections:
            return False

        subscription = self.active_connections[client_id]

        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)

            await subscription.websocket.send_text(message)
            return True

        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            await self.disconnect(client_id)
            return False

    async def broadcast(
        self,
        channel: Channel,
        message: Any,
    ) -> int:
        """Broadcast message to all subscribers of a channel."""
        subscriber_ids = self.channel_subscribers[channel].copy()
        sent_count = 0

        for client_id in subscriber_ids:
            success = await self.send_personal_message(client_id, {
                "channel": channel.value,
                "message": message,
                "timestamp": __import__('time').time(),
            })

            if success:
                sent_count += 1

        return sent_count

    async def broadcast_to_all(self, message: Any) -> int:
        """Broadcast message to all connected clients."""
        sent_count = 0

        for client_id in list(self.active_connections.keys()):
            success = await self.send_personal_message(client_id, message)
            if success:
                sent_count += 1

        return sent_count

    async def start_heartbeat(self) -> None:
        """Start heartbeat monitoring."""
        self._running = True

        async def _heartbeat_loop():
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)

                for client_id in list(self.active_connections.keys()):
                    try:
                        await self.send_personal_message(client_id, {
                            "type": "heartbeat",
                            "timestamp": __import__('time').time(),
                        })
                    except Exception:
                        pass

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info("WebSocket heartbeat started")

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("WebSocket heartbeat stopped")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_channel_subscribers(self, channel: Channel) -> int:
        """Get number of subscribers for a channel."""
        return len(self.channel_subscribers[channel])

    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """Get information about a client."""
        if client_id not in self.active_connections:
            return None

        sub = self.active_connections[client_id]
        return {
            "client_id": client_id,
            "channels": [c.value for c in sub.channels],
            "subscribed_at": sub.subscribed_at,
        }


# Global websocket manager instance
_ws_manager: Optional[WebsocketManager] = None


def init_websocket_manager(
    heartbeat_interval: int = 30,
    max_connections: int = 100,
) -> WebsocketManager:
    """Initialize global WebSocket manager."""
    global _ws_manager

    if _ws_manager is None:
        _ws_manager = WebsocketManager(heartbeat_interval, max_connections)

    return _ws_manager


def get_websocket_manager() -> Optional[WebsocketManager]:
    """Get global WebSocket manager."""
    return _ws_manager
