"""
Retry Policy with exponential backoff and jitter.
"""

import asyncio
import functools
import inspect
import random
import time
from typing import Callable, Optional, Type, List
from loguru import logger


# Sensible default for transient/retryable failures. Anything outside this
# list (auth errors, validation errors, business-logic errors, FileNotFound,
# PermissionError, etc.) should fail fast — retrying them just burns rate
# budget and confuses operators with 3× the same trace.
#
# Note: deliberately NOT including OSError — its subclasses include
# PermissionError and FileNotFoundError which are NOT transient.
# ConnectionError is a sibling that already covers BrokenPipeError,
# ConnectionResetError, etc. — the genuine transient cases.
#
# Callers that want "retry every Exception" must opt in via
# retryable_exceptions=[Exception].
_DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


class RetryPolicy:
    """
    Production-grade retry policy with:
    - Exponential backoff with jitter
    - Max retry attempts
    - Configurable base and max delay
    - Exception filtering
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        initial_delay: Optional[float] = None,
        max_delay: float = 10.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        dead_letter_queue_size: int = 100,
    ):
        self.max_attempts = max_attempts
        # Backward compatibility: accept `initial_delay` as alias for `base_delay`.
        self.base_delay = initial_delay if initial_delay is not None else base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        # Default narrowed from [Exception] to transient network errors —
        # auth / business-logic errors no longer waste 3 retries before failing.
        # Pass retryable_exceptions=[Exception] explicitly to restore old behavior.
        self.retryable_exceptions = retryable_exceptions or list(_DEFAULT_RETRYABLE_EXCEPTIONS)
        self.dead_letter_queue_size = dead_letter_queue_size
        self._dead_letter_queue: List[dict] = []

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay

        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
        )

    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs,
    ):
        """Execute an async operation with retry semantics and DLQ fallback."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if inspect.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.is_retryable(e):
                    logger.warning(f"Non-retryable exception in {operation_name}: {e}")
                    self._add_to_dlq(operation_name, e)
                    raise

                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt}/{self.max_attempts} failed in {operation_name}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts failed in {operation_name}. Last error: {e}"
                    )
                    self._add_to_dlq(operation_name, e)

        raise last_exception

    def _add_to_dlq(self, operation_name: str, exception: Exception) -> None:
        entry = {
            "timestamp": time.time(),
            "operation_name": operation_name,
            "error": str(exception),
            "exception_type": type(exception).__name__,
        }
        self._dead_letter_queue.append(entry)
        if len(self._dead_letter_queue) > self.dead_letter_queue_size:
            self._dead_letter_queue = self._dead_letter_queue[-self.dead_letter_queue_size:]

    def get_dlq(self) -> List[dict]:
        """Return dead-letter queue entries."""
        return list(self._dead_letter_queue)


def with_retry(policy: Optional[RetryPolicy] = None):
    """Decorator for retry logic."""
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not policy.is_retryable(e):
                        logger.warning(f"Non-retryable exception: {e}")
                        raise

                    if attempt < policy.max_attempts:
                        delay = policy._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt}/{policy.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {policy.max_attempts} attempts failed. Last error: {e}"
                        )

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not policy.is_retryable(e):
                        logger.warning(f"Non-retryable exception: {e}")
                        raise

                    if attempt < policy.max_attempts:
                        delay = policy._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt}/{policy.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {policy.max_attempts} attempts failed. Last error: {e}"
                        )

            raise last_exception

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
