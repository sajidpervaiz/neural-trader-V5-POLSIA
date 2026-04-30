from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        REGISTRY,
    )
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

from core.config import Config
from core.event_bus import EventBus


def _noop(*a: Any, **kw: Any) -> Any:
    class _Fake:
        def labels(self, **kw: Any) -> "_Fake":
            return self
        def inc(self, *a: Any) -> None: pass
        def dec(self, *a: Any) -> None: pass
        def set(self, *a: Any) -> None: pass
        def observe(self, *a: Any) -> None: pass
    return _Fake()


class Metrics:
    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._feed_lag_samples: list[tuple[str, float]] = []  # (exchange, lag_s)
        self._order_latency_samples: list[tuple[str, str, float]] = []  # (exchange, type, lat_s)
        self._max_samples = 500

        if _PROMETHEUS:
            self.ticks_total = Counter(
                "neuraltrader_ticks_total",
                "Total number of ticks received",
                ["exchange", "symbol"],
            )
            self.signals_total = Counter(
                "neuraltrader_signals_total",
                "Total number of trading signals generated",
                ["exchange", "symbol", "direction"],
            )
            self.orders_total = Counter(
                "neuraltrader_orders_total",
                "Total number of orders submitted",
                ["exchange", "symbol", "is_paper"],
            )
            self.open_positions = Gauge(
                "neuraltrader_open_positions",
                "Number of currently open positions",
                ["exchange"],
            )
            self.equity = Gauge(
                "neuraltrader_equity_usd",
                "Current account equity in USD",
            )
            self.daily_pnl = Gauge(
                "neuraltrader_daily_pnl_usd",
                "Daily profit/loss in USD",
            )
            self.funding_rate = Gauge(
                "neuraltrader_funding_rate",
                "Latest funding rate",
                ["exchange", "symbol"],
            )
            self.open_interest_usd = Gauge(
                "neuraltrader_open_interest_usd",
                "Open interest in USD",
                ["exchange", "symbol"],
            )
            self.vix_proxy = Gauge(
                "neuraltrader_vix_proxy",
                "VIX proxy annualized volatility",
                ["symbol", "lookback_days"],
            )
            self.rust_tick_latency = Histogram(
                "neuraltrader_rust_tick_latency_seconds",
                "Rust tick parsing latency",
                buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            )
            self.dex_quote_latency = Histogram(
                "neuraltrader_dex_quote_latency_seconds",
                "DEX quote generation latency",
                buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            )
            self.ws_feed_lag = Histogram(
                "neuraltrader_ws_feed_lag_seconds",
                "WebSocket feed lag (receive_time - exchange_timestamp)",
                ["exchange"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            )
            self.order_execution_latency = Histogram(
                "neuraltrader_order_execution_seconds",
                "Order execution round-trip latency",
                ["exchange", "order_type"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            )
            self.clock_drift = Gauge(
                "neuraltrader_clock_drift_ms",
                "Clock drift between bot and exchange in ms",
                ["exchange"],
            )
            self.service_health = Gauge(
                "neuraltrader_service_health",
                "Health of external services (1=up, 0=down)",
                ["service"],
            )
            self.alerts_total = Counter(
                "neuraltrader_alerts_total",
                "Total alerts fired by type and severity",
                ["alert_type", "severity"],
            )
            self.alerts_suppressed_total = Counter(
                "neuraltrader_alerts_suppressed_total",
                "Total alerts suppressed by throttling",
            )
            self.circuit_breaker_trips_total = Counter(
                "neuraltrader_circuit_breaker_trips_total",
                "Total circuit breaker trips",
                ["reason"],
            )
            self.safe_mode_active = Gauge(
                "neuraltrader_safe_mode_active",
                "Whether safe mode is currently active (1=yes, 0=no)",
            )
            self.kill_switch_active = Gauge(
                "neuraltrader_kill_switch_active",
                "Whether kill switch is active (1=yes, 0=no)",
            )
            self.stop_loss_hits_total = Counter(
                "neuraltrader_stop_loss_hits_total",
                "Total stop loss events",
                ["exchange", "symbol"],
            )
            self.take_profit_hits_total = Counter(
                "neuraltrader_take_profit_hits_total",
                "Total take profit events",
                ["exchange", "symbol"],
            )
        else:
            for name in [
                "ticks_total", "signals_total", "orders_total",
                "open_positions", "equity", "daily_pnl",
                "funding_rate", "open_interest_usd", "vix_proxy",
                "rust_tick_latency", "dex_quote_latency",
                "ws_feed_lag", "order_execution_latency", "clock_drift",
                "service_health",
                "alerts_total", "alerts_suppressed_total",
                "circuit_breaker_trips_total", "safe_mode_active",
                "kill_switch_active", "stop_loss_hits_total",
                "take_profit_hits_total",
            ]:
                setattr(self, name, _noop())

    async def _handle_tick(self, payload: Any) -> None:
        tick = payload
        if hasattr(tick, "exchange") and hasattr(tick, "symbol"):
            self.ticks_total.labels(exchange=tick.exchange, symbol=tick.symbol).inc()
            # Track WebSocket feed lag
            recv = getattr(tick, "receive_time_us", 0)
            ts = getattr(tick, "timestamp_us", 0)
            if recv > 0 and ts > 0:
                lag_s = (recv - ts) / 1_000_000.0
                if -1.0 < lag_s < 10.0:  # sanity bounds
                    self.ws_feed_lag.labels(exchange=tick.exchange).observe(lag_s)
                    # Keep rolling stats for /api/latency
                    self._feed_lag_samples.append((tick.exchange, lag_s))
            # Trim rolling buffer
            if len(self._feed_lag_samples) > self._max_samples:
                self._feed_lag_samples = self._feed_lag_samples[-self._max_samples:]

    def record_order_latency(self, exchange: str, order_type: str, latency_s: float) -> None:
        """Record order execution latency (called from trade endpoints)."""
        self.order_execution_latency.labels(exchange=exchange, order_type=order_type).observe(latency_s)
        self._order_latency_samples.append((exchange, order_type, latency_s))
        if len(self._order_latency_samples) > self._max_samples:
            self._order_latency_samples = self._order_latency_samples[-self._max_samples:]

    @staticmethod
    def _percentile_ms(values: list[float], pct: float) -> float:
        """Safe percentile in ms — handles 0/1-element arrays without IndexError."""
        if not values:
            return 0.0
        if len(values) == 1:
            return round(values[0] * 1000, 1)
        s = sorted(values)
        # Nearest-rank method, clamped to last index.
        idx = min(len(s) - 1, max(0, int(round(pct / 100.0 * (len(s) - 1)))))
        return round(s[idx] * 1000, 1)

    def get_latency_stats(self) -> dict[str, Any]:
        """REQ-MON-002: per-bucket latency stats with p50/p95/p99 percentiles."""
        import statistics
        result: dict[str, Any] = {"feed_lag": {}, "order_latency": {}}

        by_exchange: dict[str, list[float]] = {}
        for ex, lag in self._feed_lag_samples:
            by_exchange.setdefault(ex, []).append(lag)
        for ex, lags in by_exchange.items():
            result["feed_lag"][ex] = {
                "count": len(lags),
                "avg_ms": round(statistics.mean(lags) * 1000, 1) if lags else 0,
                "p50_ms": self._percentile_ms(lags, 50),
                "p95_ms": self._percentile_ms(lags, 95),
                "p99_ms": self._percentile_ms(lags, 99),
                "max_ms": round(max(lags) * 1000, 1) if lags else 0,
            }

        by_type: dict[str, list[float]] = {}
        for ex, otype, lat in self._order_latency_samples:
            key = f"{ex}_{otype}"
            by_type.setdefault(key, []).append(lat)
        for key, lats in by_type.items():
            result["order_latency"][key] = {
                "count": len(lats),
                "avg_ms": round(statistics.mean(lats) * 1000, 0) if lats else 0,
                "p50_ms": self._percentile_ms(lats, 50),
                "p95_ms": self._percentile_ms(lats, 95),
                "p99_ms": self._percentile_ms(lats, 99),
                "max_ms": round(max(lats) * 1000, 0) if lats else 0,
            }
        return result

    def get_latency_percentiles(self) -> dict[str, Any]:
        """Compact percentiles-only payload for the dashboard chip strip."""
        full = self.get_latency_stats()
        compact: dict[str, Any] = {"feed_lag": {}, "order_latency": {}}
        for ex, stats in full["feed_lag"].items():
            compact["feed_lag"][ex] = {
                "p50_ms": stats["p50_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "count": stats["count"],
            }
        for key, stats in full["order_latency"].items():
            compact["order_latency"][key] = {
                "p50_ms": stats["p50_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "count": stats["count"],
            }
        return compact

    async def _handle_signal(self, payload: Any) -> None:
        signal = payload
        if hasattr(signal, "exchange"):
            self.signals_total.labels(
                exchange=signal.exchange,
                symbol=signal.symbol,
                direction=signal.direction,
            ).inc()

    async def _handle_order(self, payload: Any) -> None:
        order = payload
        if hasattr(order, "exchange"):
            self.orders_total.labels(
                exchange=order.exchange,
                symbol=order.symbol,
                is_paper=str(order.is_paper),
            ).inc()

    async def _handle_funding(self, payload: Any) -> None:
        if isinstance(payload, list):
            for rate in payload:
                self.funding_rate.labels(exchange=rate.exchange, symbol=rate.symbol).set(rate.rate)

    async def _handle_oi(self, payload: Any) -> None:
        if isinstance(payload, list):
            for oi in payload:
                self.open_interest_usd.labels(exchange=oi.exchange, symbol=oi.symbol).set(oi.oi_usd)

    async def _handle_vix(self, payload: Any) -> None:
        if isinstance(payload, list):
            for v in payload:
                self.vix_proxy.labels(symbol=v.symbol, lookback_days=v.lookback_days).set(v.annualized)

    async def _handle_stop_loss(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        exchange = data.get("exchange", "unknown")
        symbol = data.get("symbol", "unknown")
        self.stop_loss_hits_total.labels(exchange=exchange, symbol=symbol).inc()

    async def _handle_take_profit(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        exchange = data.get("exchange", "unknown")
        symbol = data.get("symbol", "unknown")
        self.take_profit_hits_total.labels(exchange=exchange, symbol=symbol).inc()

    async def _handle_kill_switch(self, payload: Any) -> None:
        self.kill_switch_active.set(1)

    async def run(self) -> None:
        self._running = True
        mon_cfg = self.config.get_value("monitoring", "prometheus") or {}
        if mon_cfg.get("enabled", True) and _PROMETHEUS:
            port = int(mon_cfg.get("port", 9090))
            try:
                start_http_server(port)
                logger.info("Prometheus metrics server started on :{}", port)
            except Exception as exc:
                logger.warning("Prometheus server failed: {}", exc)

        self.event_bus.subscribe("TICK", self._handle_tick)
        self.event_bus.subscribe("SIGNAL", self._handle_signal)
        self.event_bus.subscribe("ORDER_FILLED", self._handle_order)
        self.event_bus.subscribe("FUNDING_RATE", self._handle_funding)
        self.event_bus.subscribe("OPEN_INTEREST", self._handle_oi)
        self.event_bus.subscribe("VOLATILITY_INDEX", self._handle_vix)
        self.event_bus.subscribe("STOP_LOSS", self._handle_stop_loss)
        self.event_bus.subscribe("TAKE_PROFIT", self._handle_take_profit)
        self.event_bus.subscribe("KILL_SWITCH", self._handle_kill_switch)

        while self._running:
            await asyncio.sleep(15)

    async def stop(self) -> None:
        self._running = False
        self.event_bus.unsubscribe("TICK", self._handle_tick)
        self.event_bus.unsubscribe("SIGNAL", self._handle_signal)
        self.event_bus.unsubscribe("ORDER_FILLED", self._handle_order)
        self.event_bus.unsubscribe("FUNDING_RATE", self._handle_funding)
        self.event_bus.unsubscribe("OPEN_INTEREST", self._handle_oi)
        self.event_bus.unsubscribe("VOLATILITY_INDEX", self._handle_vix)
        self.event_bus.unsubscribe("STOP_LOSS", self._handle_stop_loss)
        self.event_bus.unsubscribe("TAKE_PROFIT", self._handle_take_profit)
        self.event_bus.unsubscribe("KILL_SWITCH", self._handle_kill_switch)
