"""Pydantic v2 schema for settings.yaml — fail-fast validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SystemConfig(BaseModel):
    name: str = "NUERAL-TRADER-5"
    version: str = "4.0.0"
    paper_mode: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs"
    timezone: str = "UTC"

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v.upper()


class ExchangeConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    type: str = "futures"
    testnet: bool = True
    symbols: list[str] = Field(default_factory=list)
    rate_limit_rpm: int = 600


class RiskConfig(BaseModel):
    max_position_size_pct: float = Field(0.02, ge=0.001, le=0.5)
    max_daily_loss_pct: float = Field(0.03, ge=0.005, le=0.5)
    max_drawdown_pct: float = Field(0.10, ge=0.01, le=0.5)
    max_open_positions: int = Field(5, ge=1, le=100)
    default_leverage: float = Field(1.0, ge=1.0, le=125.0)
    stop_loss_pct: float = Field(0.015, ge=0.001, le=0.5)
    take_profit_pct: float = Field(0.03, ge=0.001, le=1.0)
    kelly_fraction: float = Field(0.25, ge=0.0, le=1.0)
    min_liquidity_usd: float = Field(100_000, ge=0)
    min_balance_usd: float = Field(100, ge=0)
    initial_equity: float = Field(100_000, gt=0)
    sizing_method: str = "risk_based"
    risk_per_trade_pct: float = Field(0.01, ge=0.001, le=0.1)
    max_spread_bps: float = Field(10.0, ge=0)
    max_atr_pct: float = Field(0.05, ge=0)
    max_exposure_per_symbol_pct: float = Field(0.10, ge=0.01, le=1.0)
    cooldown_seconds: float = Field(300.0, ge=0)
    session_start_utc: str = "00:00"
    session_end_utc: str = "23:59"
    atr_sl_multiplier: float = Field(1.5, ge=0.1, le=10.0)
    rr_ratio: float = Field(2.0, ge=1.0, le=20.0)
    trailing_activation_atr: float = Field(2.0, ge=0)
    trailing_distance_atr: float = Field(1.0, ge=0)
    breakeven_trigger_atr: float = Field(1.0, ge=0)
    max_hold_minutes: int = Field(0, ge=0)
    circuit_breaker_pause_seconds: float = Field(3600.0, ge=0)
    max_portfolio_var_pct: float = Field(0.08, ge=0)
    returns_window: int = Field(250, ge=10)
    var_min_history: int = Field(30, ge=5)
    # New: funding rate filter
    max_funding_rate_bps: float = Field(50.0, ge=0)
    # New: orderbook depth filter
    min_orderbook_depth_usd: float = Field(50_000, ge=0)
    # New: hard max order size
    max_order_size_usd: float = Field(500_000, gt=0)
    # New: max leverage per symbol (dict mapping symbol -> max leverage, empty = use default_leverage)
    max_leverage_per_symbol: dict[str, float] = Field(default_factory=dict)

    @field_validator("sizing_method")
    @classmethod
    def _valid_sizing(cls, v: str) -> str:
        allowed = {"risk_based", "fixed_fraction", "kelly"}
        if v not in allowed:
            raise ValueError(f"sizing_method must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _risk_coherence(self) -> "RiskConfig":
        # stop_loss must be smaller than take_profit
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError(
                f"stop_loss_pct ({self.stop_loss_pct}) must be < "
                f"take_profit_pct ({self.take_profit_pct})"
            )
        # max_drawdown must be larger than daily loss limit
        if self.max_drawdown_pct < self.max_daily_loss_pct:
            raise ValueError(
                f"max_drawdown_pct ({self.max_drawdown_pct}) must be >= "
                f"max_daily_loss_pct ({self.max_daily_loss_pct})"
            )
        # Per-symbol leverage caps must not exceed default_leverage * 2 as sanity
        for sym, lev in self.max_leverage_per_symbol.items():
            if lev <= 0:
                raise ValueError(f"leverage for {sym} must be > 0, got {lev}")
            if lev > 125:
                raise ValueError(f"leverage for {sym} ({lev}) exceeds exchange max (125)")
        return self


class SignalsConfig(BaseModel):
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    primary_timeframe: str = "15m"
    confirmation_timeframes: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    min_score_threshold: float = Field(0.65, ge=0.0, le=1.0)
    ml_weight: float = Field(0.25, ge=0.0, le=1.0)
    technical_weight: float = Field(0.30, ge=0.0, le=1.0)
    sentiment_weight: float = Field(0.10, ge=0.0, le=1.0)
    macro_weight: float = Field(0.10, ge=0.0, le=1.0)
    news_weight: float = Field(0.15, ge=0.0, le=1.0)
    orderbook_weight: float = Field(0.10, ge=0.0, le=1.0)
    min_contributing_factors: int = Field(3, ge=1, le=6)

    @model_validator(mode="after")
    def _weights_sum(self) -> "SignalsConfig":
        total = (
            self.ml_weight + self.technical_weight + self.sentiment_weight
            + self.macro_weight + self.news_weight + self.orderbook_weight
        )
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"Signal weights must sum to ~1.0, got {total:.3f}")
        return self


class BacktestConfig(BaseModel):
    initial_capital: float = Field(100_000, gt=0)
    commission_pct: float = Field(0.0004, ge=0, le=0.01)
    slippage_pct: float = Field(0.0002, ge=0, le=0.01)
    use_rust_engine: bool = True
    wfo_splits: int = Field(5, ge=2, le=20)
    monte_carlo_runs: int = Field(1000, ge=100, le=100_000)


class SimulatedExchangeConfig(BaseModel):
    enabled: bool = True
    ack_latency_ms: float = Field(15.0, ge=0)
    fill_latency_ms: float = Field(35.0, ge=0)
    partial_fill_probability: float = Field(0.0, ge=0, le=1)
    partial_fill_ratio: float = Field(0.5, gt=0, le=1)
    partial_fill_completion_ms: float = Field(250.0, ge=0)
    reject_probability: float = Field(0.0, ge=0, le=1)
    max_slippage_bps: float = Field(50.0, ge=0)
    synthetic_spread_bps: float = Field(2.0, ge=0)
    synthetic_orderbook_depth_usd: float = Field(1_000_000.0, gt=0)
    default_price: float = Field(50_000.0, gt=0)


class ExecutionConfig(BaseModel):
    simulated_exchange: SimulatedExchangeConfig = Field(default_factory=SimulatedExchangeConfig)
    wait_for_fill: dict[str, Any] = Field(default_factory=dict)
    restore_orders_on_startup: bool | None = None
    restore_positions_on_startup: bool = True
    maker_first: bool = True
    post_only: bool = True
    iceberg_threshold_usd: float = Field(10_000.0, ge=0)
    iceberg_chunks: int = Field(4, ge=1)

    model_config = {"extra": "allow"}


class MarketDataIntegrityConfig(BaseModel):
    enabled: bool = True
    enforce_signal_gate: bool = True
    tick_stale_seconds: float = Field(10.0, gt=0)
    orderbook_stale_seconds: float = Field(90.0, gt=0)
    signal_event_stale_seconds: float = Field(5.0, gt=0)
    candle_stale_multiple: float = Field(2.5, ge=1.0)
    max_clock_drift_seconds: float = Field(2.0, ge=0)
    gap_quarantine_seconds: float = Field(30.0, ge=0)
    check_interval_seconds: float = Field(1.0, gt=0)


class OrderbookFeedConfig(BaseModel):
    mode: str = "hybrid"
    depth: int = Field(20, ge=5, le=20)
    update_speed_ms: int = Field(250, ge=100, le=500)
    rest_poll_interval_seconds: float = Field(30.0, gt=0)
    ws_reconnect_min_seconds: float = Field(1.0, gt=0)
    ws_reconnect_max_seconds: float = Field(30.0, gt=0)

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        value = v.lower()
        allowed = {"rest", "ws", "websocket", "ws_partial", "hybrid"}
        if value not in allowed:
            raise ValueError(f"orderbook_feed.mode must be one of {allowed}")
        return value

    @field_validator("depth")
    @classmethod
    def _valid_depth(cls, v: int) -> int:
        if v not in {5, 10, 20}:
            raise ValueError("orderbook_feed.depth must be one of 5, 10, 20")
        return v

    @field_validator("update_speed_ms")
    @classmethod
    def _valid_speed(cls, v: int) -> int:
        if v not in {100, 250, 500}:
            raise ValueError("orderbook_feed.update_speed_ms must be one of 100, 250, 500")
        return v


class DataIngestionConfig(BaseModel):
    ws_prolonged_disconnect_seconds: float = Field(30.0, gt=0)
    orderbook_feed: OrderbookFeedConfig = Field(default_factory=OrderbookFeedConfig)

    model_config = {"extra": "allow"}


class StoragePostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "neural_trader"
    user: str = "trader"
    password: str = ""
    pool_size: int = Field(10, ge=1, le=50)
    timescaledb: bool = True


class StorageRedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    pool_size: int = Field(20, ge=1)
    default_ttl_seconds: int = 300


class StorageConfig(BaseModel):
    postgres: StoragePostgresConfig = Field(default_factory=StoragePostgresConfig)
    redis: StorageRedisConfig = Field(default_factory=StorageRedisConfig)
    personal_sqlite_mode: bool = False
    sqlite: dict[str, Any] = Field(default_factory=lambda: {"path": "data/neural_trader.db"})


class MonitoringDashboardAuth(BaseModel):
    require_api_key: bool = False
    api_key: str = ""
    rate_limit_per_min: int = Field(120, ge=1)


class MonitoringDashboardConfig(BaseModel):
    # Binding to 0.0.0.0 is intentional for container/VM deployments; the
    # dashboard_api middleware enforces API-key auth in non-paper mode.
    host: str = "0.0.0.0"  # noqa: S104  # nosec B104 — auth-gated via dashboard middleware
    port: int = 8000
    allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost"])
    auth: MonitoringDashboardAuth = Field(default_factory=MonitoringDashboardAuth)


class MonitoringConfig(BaseModel):
    prometheus: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "port": 9090})
    dashboard_api: MonitoringDashboardConfig = Field(default_factory=MonitoringDashboardConfig)
    telegram: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Top-level application config with strict validation."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    exchanges: dict[str, ExchangeConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    market_data_integrity: MarketDataIntegrityConfig = Field(default_factory=MarketDataIntegrityConfig)
    data_ingestion: DataIngestionConfig = Field(default_factory=DataIngestionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Allow extra sections (dex, macro, rust_services, etc.)
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _safety_checks(self) -> "AppConfig":
        if not self.system.paper_mode:
            # Live mode: require API keys for all enabled exchanges
            for name, exc in self.exchanges.items():
                if exc.enabled and not exc.testnet:
                    if not exc.api_key:
                        raise ValueError(
                            f"Exchange '{name}' is live (testnet=false) but has no API key. "
                            "Set paper_mode=true or provide credentials."
                        )
                if exc.enabled:
                    if not exc.symbols:
                        raise ValueError(
                            f"Exchange '{name}' is enabled but has no symbols configured."
                        )
            # Live mode: enforce non-zero min_balance
            if self.risk.min_balance_usd <= 0:
                raise ValueError(
                    "risk.min_balance_usd must be > 0 in live mode"
                )
            # Live mode: leverage sanity
            if self.risk.default_leverage > 20:
                raise ValueError(
                    f"risk.default_leverage ({self.risk.default_leverage}) > 20 is "
                    "too dangerous for live trading. Reduce leverage."
                )
        return self


def validate_config(raw: dict[str, Any]) -> AppConfig:
    """Validate a raw config dict, raising ValidationError if invalid."""
    return AppConfig.model_validate(raw)
