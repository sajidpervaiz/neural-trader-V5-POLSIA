"""
OpenTelemetry tracing for distributed tracing across Python/Rust/TS.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from contextvars import ContextVar
from typing import Optional, Dict
import inspect
import secrets
import time
from loguru import logger

from core.error_handling import sanitize_exception

# Context for correlation ID
correlation_id_context: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def get_correlation_id() -> str:
    """Get or generate correlation ID for the current request."""
    cid = correlation_id_context.get()
    if not cid:
        cid = secrets.token_hex(16)
        correlation_id_context.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for the current request."""
    correlation_id_context.set(cid)


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "NUERAL-TRADER-5",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.enabled = enabled


class DistributedTracer:
    """
    Distributed tracer with OpenTelemetry integration.

    Provides automatic instrumentation for:
    - FastAPI endpoints
    - HTTP clients
    - Redis operations
    - Custom spans
    """

    def __init__(self, config: TracingConfig):
        self.config = config

        if config.enabled:
            self._setup_tracing()
        else:
            logger.warning("Tracing is disabled")
            self.tracer = trace.get_tracer(__name__)

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing with Jaeger exporter."""
        try:
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                "service.version": "4.0.0",
                "deployment.environment": "production",
            })

            trace.set_tracer_provider(TracerProvider(resource=resource))

            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_host,
                agent_port=self.config.jaeger_port,
            )

            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

            self.tracer = trace.get_tracer(__name__)

            logger.info(f"Tracing initialized: {self.config.service_name} -> Jaeger")

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.tracer = trace.get_tracer(__name__)

    def instrument_fastapi(self, app) -> None:
        """Instrument FastAPI application."""
        if self.config.enabled:
            try:
                FastAPIInstrumentor.instrument_app(app)
                logger.info("FastAPI instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument FastAPI: {e}")

    def instrument_httpx(self, client) -> None:
        """Instrument HTTPX client."""
        if self.config.enabled:
            try:
                HTTPXClientInstrumentor().instrument()
                logger.info("HTTPX instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument HTTPX: {e}")

    def instrument_redis(self) -> None:
        """Instrument Redis client."""
        if self.config.enabled:
            try:
                RedisInstrumentor().instrument()
                logger.info("Redis instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument Redis: {e}")

    def start_span(self, name: str, attributes: Optional[Dict] = None):
        """Start a new span."""
        if not self.config.enabled:
            return None

        correlation_id = get_correlation_id()
        span_attrs = {
            "correlation.id": correlation_id,
        }
        if attributes:
            span_attrs.update(attributes)

        return self.tracer.start_span(name, attributes=span_attrs)

    def trace_function(self, name: str):
        """Decorator for tracing function execution."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                span = self.start_span(name)
                start_time = time.time()

                try:
                    if span:
                        span.set_attribute("function.name", func.__name__)

                    result = await func(*args, **kwargs)

                    if span:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration.ms", duration_ms)
                        span.set_status(
                            trace.Status(
                                code=trace.StatusCode.OK,
                                description="Function completed successfully"
                            )
                        )
                        span.end()

                    return result

                except Exception as e:
                    if span:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration.ms", duration_ms)
                        span.set_status(
                            trace.Status(
                                code=trace.StatusCode.ERROR,
                                description=sanitize_exception(e)
                            )
                        )
                        span.record_exception(e)
                        span.end()

                    raise

            def sync_wrapper(*args, **kwargs):
                span = self.start_span(name)
                start_time = time.time()

                try:
                    if span:
                        span.set_attribute("function.name", func.__name__)

                    result = func(*args, **kwargs)

                    if span:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration.ms", duration_ms)
                        span.set_status(
                            trace.Status(
                                code=trace.StatusCode.OK,
                                description="Function completed successfully"
                            )
                        )
                        span.end()

                    return result

                except Exception as e:
                    if span:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration.ms", duration_ms)
                        span.set_status(
                            trace.Status(
                                code=trace.StatusCode.ERROR,
                                description=sanitize_exception(e)
                            )
                        )
                        span.record_exception(e)
                        span.end()

                    raise

            if hasattr(func, '__call__') and inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


# Global tracer instance
_tracer: Optional[DistributedTracer] = None


def init_tracing(config: Optional[TracingConfig] = None) -> DistributedTracer:
    """Initialize global tracer instance."""
    global _tracer

    if _tracer is None:
        if config is None:
            config = TracingConfig()

        _tracer = DistributedTracer(config)

    return _tracer


def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance."""
    return _tracer


def add_span_attributes(attributes: Dict[str, any]) -> None:
    """Add attributes to current span."""
    current_span = trace.get_current_span()
    if current_span:
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict] = None) -> None:
    """Add event to current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.add_event(name, attributes or {})
