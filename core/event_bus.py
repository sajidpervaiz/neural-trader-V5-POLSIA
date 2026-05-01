from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from loguru import logger


Handler = Callable[..., Coroutine[Any, Any, None]]

# Events that MUST NOT be silently dropped — they protect capital
CRITICAL_EVENTS = frozenset({
    "STOP_LOSS", "TAKE_PROFIT", "KILL_SWITCH", "ALERT_CRITICAL",
    "LIQUIDATION", "MARGIN_CALL",
})

# Events that require serial (ordered) handler execution
SERIAL_EVENTS = frozenset({
    "SIGNAL",
    "STOP_LOSS", "TAKE_PROFIT", "KILL_SWITCH", "LIQUIDATION",
    "MARGIN_CALL", "ORDER_FILLED", "ORDER_PARTIALLY_FILLED",
    "FILL_CONFIRMED", "POSITION_CLOSED",
})


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=10_000)
        self._running = False
        self._background_tasks: set[asyncio.Task] = set()
        self._dropped_count: int = 0
        self._backpressure_warned: bool = False

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("EventBus: {} subscribed to '{}'", handler.__qualname__, event_type)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    async def publish(self, event_type: str, payload: Any = None) -> None:
        qsize = self._queue.qsize()
        if qsize > 8_000 and not self._backpressure_warned:
            logger.warning("EventBus queue at {}% capacity ({}/10000) — backpressure risk", qsize // 100, qsize)
            self._backpressure_warned = True
        elif qsize < 5_000:
            self._backpressure_warned = False
        await self._queue.put((event_type, payload))

    def publish_nowait(self, event_type: str, payload: Any = None) -> None:
        try:
            self._queue.put_nowait((event_type, payload))
        except asyncio.QueueFull:
            if event_type in CRITICAL_EVENTS:
                logger.critical(
                    "EventBus queue full — CRITICAL event '{}' BLOCKED. "
                    "Scheduling direct dispatch.",
                    event_type,
                )
                # Force-dispatch critical events directly bypassing the queue.
                # Track task so stop() awaits it before shutting down.
                task = asyncio.ensure_future(self._dispatch(event_type, payload))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                self._dropped_count += 1
                logger.warning("EventBus queue full — dropping event '{}' (total dropped: {})", event_type, self._dropped_count)

    async def _dispatch(self, event_type: str, payload: Any) -> None:
        """Dispatch an event to its handlers."""
        handlers = self._handlers.get(event_type, [])
        if event_type in SERIAL_EVENTS:
            # Serial execution for order-sensitive events
            for handler in handlers:
                try:
                    await handler(payload)
                except Exception as exc:
                    logger.exception("EventBus handler error in '{}': {}", handler.__qualname__, exc)
        else:
            # Concurrent execution for non-critical events (TICK, etc.)
            for handler in handlers:
                task = asyncio.create_task(self._safe_call(handler, payload))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    async def _safe_call(handler: Handler, payload: Any) -> None:
        try:
            await handler(payload)
        except Exception as exc:
            logger.exception("EventBus handler error in '{}': {}", handler.__qualname__, exc)

    async def run(self) -> None:
        self._running = True
        logger.info("EventBus started")
        while self._running:
            try:
                event_type, payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._dispatch(event_type, payload)

    async def stop(self) -> None:
        self._running = False
        # Wait for in-flight background handler tasks
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info("EventBus stopped")

    def stats(self) -> dict[str, Any]:
        """REQ-ARC-003 / REQ-MON-001: backpressure observability."""
        capacity = int(self._queue.maxsize) if self._queue.maxsize > 0 else 0
        size = int(self._queue.qsize())
        pct = (size / capacity * 100.0) if capacity > 0 else 0.0
        return {
            "queue_size": size,
            "queue_capacity": capacity,
            "queue_pct": round(pct, 1),
            "dropped_count": int(self._dropped_count),
            "backpressure_warned": bool(self._backpressure_warned),
            "subscribed_topics": len(self._handlers),
            "subscriber_counts": {topic: len(hs) for topic, hs in self._handlers.items()},
            "background_tasks": len(self._background_tasks),
            "running": bool(self._running),
        }
