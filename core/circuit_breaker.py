"""
Circuit Breaker with half-open state testing for resilience.
"""

import asyncio
import inspect
import time
from enum import Enum
from typing import Callable, Optional, Type
from loguru import logger


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Production-grade circuit breaker with:
    - Closed, Open, and Half-Open states
    - Failure counting
    - Automatic recovery with half-open testing
    - Thread-safe state transitions
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Optional[Type[Exception]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception or Exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        self.half_open_success_threshold = 2

        self._lock = asyncio.Lock()
        self._call_stats = {"total": 0, "success": 0, "failure": 0}

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioned to HALF_OPEN for recovery testing")
                else:
                    raise Exception("Circuit breaker is OPEN - rejecting call")

        try:
            result = await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
            async with self._lock:
                self.record_success()
                self._call_stats["total"] += 1
                self._call_stats["success"] += 1
            return result

        except Exception as e:
            async with self._lock:
                if isinstance(e, self.expected_exception):
                    self.record_failure()
                self._call_stats["total"] += 1
                self._call_stats["failure"] += 1
            raise

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Transition immediately to keep behavior deterministic for callers/tests.
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning("Circuit breaker OPENED from HALF_OPEN after failure")
        elif self.failure_count >= self.failure_threshold:
            # Transition immediately to keep behavior deterministic for callers/tests.
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker OPENED: {self.failure_count} failures")

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        async with self._lock:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker OPENED: {self.failure_count} failures")

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioned to HALF_OPEN for recovery testing")

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")

    def get_state(self) -> CircuitState:
        """Get current state."""
        return self.state

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "call_stats": self._call_stats,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self._call_stats = {"total": 0, "success": 0, "failure": 0}
        logger.info("Circuit breaker reset")
