"""
Health checks for service monitoring.
"""

from typing import Awaitable, Callable, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from loguru import logger

from core.error_handling import sanitize_exception


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComponentHealth:
    component: str
    status: HealthStatus
    latency_ms: float
    message: str
    last_check: float
    details: Optional[Dict] = None


@dataclass
class HealthCheckResult:
    overall_status: HealthStatus
    timestamp: float
    components: Dict[str, ComponentHealth]
    uptime_seconds: float
    version: str


class HealthChecker:
    """
    Health check system for all components.

    Monitors:
    - Database connectivity
    - Redis connectivity
    - Venue APIs
    - Message queues
    - Internal services
    """

    def __init__(
        self,
        component_timeout: float = 5.0,
        check_interval: int = 30,
    ):
        self.component_timeout = component_timeout
        self.check_interval = check_interval

        self.components: Dict[str, ComponentHealth] = {}
        self._component_checks: Dict[str, Callable[[], Awaitable[bool | dict]]] = {}
        self.start_time = time.time()
        self.version = "4.0.0"

        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    async def register_component(
        self,
        name: str,
        check_func: Callable[[], Awaitable[bool | dict]],
    ) -> None:
        """Register a component for health checking."""
        self.components[name] = ComponentHealth(
            component=name,
            status=HealthStatus.UNKNOWN,
            latency_ms=0.0,
            message="Not yet checked",
            last_check=0.0,
        )
        self._component_checks[name] = check_func

        logger.info(f"Registered health check component: {name}")

    async def check_component_health(
        self,
        name: str,
        check_func: Callable[[], Awaitable[bool | dict]],
    ) -> ComponentHealth:
        """Check health of a specific component."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check_func(),
                timeout=self.component_timeout,
            )

            latency_ms = (time.time() - start_time) * 1000

            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Failed"
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "HEALTHY"))
                message = result.get("message", "OK")
            else:
                status = HealthStatus.HEALTHY
                message = str(result)

            return ComponentHealth(
                component=name,
                status=status,
                latency_ms=latency_ms,
                message=message,
                last_check=time.time(),
                details=result if isinstance(result, dict) else None,
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                component=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.component_timeout * 1000,
                message=f"Timeout after {self.component_timeout}s",
                last_check=time.time(),
            )

        except Exception as e:
            return ComponentHealth(
                component=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=sanitize_exception(e),
                last_check=time.time(),
            )

    async def check_all_components(self) -> HealthCheckResult:
        """Check health of all registered components."""
        component_checks = []

        for name, check_func in self._component_checks.items():
            check_task = asyncio.create_task(
                self.check_component_health(name, check_func)
            )
            component_checks.append(check_task)

        results = await asyncio.gather(*component_checks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check error: {result}")
            elif isinstance(result, ComponentHealth):
                self.components[result.component] = result

        # Determine overall status
        statuses = [c.status for c in self.components.values()]

        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        return HealthCheckResult(
            overall_status=overall_status,
            timestamp=time.time(),
            components=self.components.copy(),
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
        )

    async def start_periodic_checks(self) -> None:
        """Start periodic health checks."""
        self._running = True

        async def _check_loop():
            while self._running:
                result = await self.check_all_components()
                logger.info(
                    f"Health check: {result.overall_status.value} - "
                    f"{len(result.components)} components checked"
                )
                await asyncio.sleep(self.check_interval)

        self._check_task = asyncio.create_task(_check_loop())
        logger.info("Periodic health checks started")

    async def stop_periodic_checks(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Periodic health checks stopped")

    async def get_health_status(self) -> HealthCheckResult:
        """Get current health status."""
        if not self.components:
            return HealthCheckResult(
                overall_status=HealthStatus.UNKNOWN,
                timestamp=time.time(),
                components={},
                uptime_seconds=0.0,
                version=self.version,
            )

        # Use last known results if recent
        now = time.time()
        stale_threshold = self.check_interval * 2

        any_stale = any(
            now - c.last_check > stale_threshold
            for c in self.components.values()
        )

        if any_stale:
            return await self.check_all_components()

        return HealthCheckResult(
            overall_status=self._determine_overall_status(),
            timestamp=now,
            components=self.components.copy(),
            uptime_seconds=now - self.start_time,
            version=self.version,
        )

    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall status from components."""
        statuses = [c.status for c in self.components.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def register_builtin_checks(self) -> None:
        """Register standard health checks."""
        # Database
        async def check_database():
            try:
                import asyncpg  # type: ignore
            except Exception:
                return {
                    "status": "DEGRADED",
                    "message": "asyncpg not installed",
                }

            import os
            host = os.environ.get("POSTGRES_HOST", "localhost")
            port = int(os.environ.get("POSTGRES_PORT", "5432"))
            user = os.environ.get("POSTGRES_USER", "trader")
            password = os.environ.get("POSTGRES_PASSWORD", "")
            database = os.environ.get("POSTGRES_DB", "neural_trader")

            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                timeout=2.0,
            )
            await conn.close()
            return {
                "status": "HEALTHY",
                "message": "database reachable",
            }

        # Redis
        async def check_redis():
            try:
                import redis.asyncio as redis  # type: ignore
            except Exception:
                return {
                    "status": "DEGRADED",
                    "message": "redis client not installed",
                }

            import os
            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            password = os.environ.get("REDIS_PASSWORD", "")

            client = redis.Redis(
                host=host,
                port=port,
                password=password or None,
                socket_timeout=2.0,
                decode_responses=True,
            )
            pong = await client.ping()
            await client.close()
            return {
                "status": "HEALTHY" if pong else "UNHEALTHY",
                "message": "redis reachable" if pong else "redis ping failed",
            }

        # Trading engine
        async def check_trading_engine():
            await asyncio.sleep(0)
            return {
                "status": "HEALTHY",
                "message": "event loop responsive",
            }

        self.components["database"] = ComponentHealth(
            component="database",
            status=HealthStatus.UNKNOWN,
            latency_ms=0.0,
            message="Not yet checked",
            last_check=0.0,
        )
        self._component_checks["database"] = check_database

        self.components["redis"] = ComponentHealth(
            component="redis",
            status=HealthStatus.UNKNOWN,
            latency_ms=0.0,
            message="Not yet checked",
            last_check=0.0,
        )
        self._component_checks["redis"] = check_redis

        self.components["trading_engine"] = ComponentHealth(
            component="trading_engine",
            status=HealthStatus.UNKNOWN,
            latency_ms=0.0,
            message="Not yet checked",
            last_check=0.0,
        )
        self._component_checks["trading_engine"] = check_trading_engine


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def init_health_checker(
    component_timeout: float = 5.0,
    check_interval: int = 30,
) -> HealthChecker:
    """Initialize global health checker."""
    global _health_checker

    if _health_checker is None:
        _health_checker = HealthChecker(component_timeout, check_interval)
        _health_checker.register_builtin_checks()

    return _health_checker


def get_health_checker() -> Optional[HealthChecker]:
    """Get global health checker."""
    return _health_checker
