from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.config import Config
from core.event_bus import EventBus


class Dispatcher:
    """Central orchestrator that wires all subsystems together."""

    def __init__(
        self,
        config: Config | None = None,
        event_bus: EventBus | None = None,
        data_manager: Any = None,
        signal_generator: Any = None,
        risk_manager: Any = None,
        execution_engine: Any = None,
        db_handler: Any = None,
        cache: Any = None,
        metrics: Any = None,
    ) -> None:
        self.config = config or Config.get()
        self.event_bus = event_bus or EventBus()
        self.data_manager = data_manager
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.db_handler = db_handler
        self.cache = cache
        self.metrics = metrics
        self._tasks: list[asyncio.Task] = []

    async def _start_component(self, name: str, coro: Any) -> None:
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        logger.info("Dispatcher: started component '{}'", name)

    async def start(self) -> None:
        logger.info("Dispatcher starting — paper_mode={}", self.config.paper_mode)

        if self.db_handler:
            await self.db_handler.connect()

        if self.cache:
            await self.cache.connect()

        await self._start_component("event_bus", self.event_bus.run())

        if self.data_manager:
            await self._start_component("data_manager", self.data_manager.run())

        if self.signal_generator:
            await self._start_component("signal_generator", self.signal_generator.run())

        if self.risk_manager:
            await self._start_component("risk_manager", self.risk_manager.run())

        if self.execution_engine:
            await self._start_component("execution_engine", self.execution_engine.run())

        if self.metrics:
            await self._start_component("metrics", self.metrics.run())

        logger.info("Dispatcher fully started with {} components", len(self._tasks))

    async def stop(self) -> None:
        logger.info("Dispatcher stopping…")
        await self.event_bus.stop()
        for task in self._tasks:
            task.cancel()
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                logger.warning("Component error on shutdown: {}", r)

        if self.db_handler:
            await self.db_handler.close()
        if self.cache:
            await self.cache.close()
        logger.info("Dispatcher stopped cleanly")

    async def run_forever(self) -> None:
        await self.start()
        try:
            pending: set[asyncio.Task] = set(self._tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                fatal_error: Exception | None = None
                for task in done:
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        continue
                    if exc is not None and not isinstance(exc, asyncio.CancelledError):
                        logger.error(
                            "Dispatcher detected fatal component crash in '{}': {}",
                            task.get_name(),
                            exc,
                        )
                        try:
                            self.event_bus.publish_nowait("COMPONENT_ERROR", {
                                "component": task.get_name(),
                                "error": repr(exc)[:500],
                                "error_type": type(exc).__name__,
                            })
                        except Exception:
                            pass
                        fatal_error = exc
                        break

                if fatal_error is not None:
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    raise RuntimeError("Dispatcher stopped due to component failure") from fatal_error
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
