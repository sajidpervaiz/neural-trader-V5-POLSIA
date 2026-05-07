from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from core.config import Config
from core.event_bus import EventBus
from core.safe_mode import SafeModeManager
from engine.signal_generator import TradingSignal


@dataclass
class Position:
    exchange: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    open_time: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trailing_active: bool = False
    trailing_stop: float = 0.0
    breakeven_moved: bool = False
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.is_long:
            self.pnl_pct = (price - self.entry_price) / self.entry_price
            self.highest_since_entry = max(self.highest_since_entry, price)
        else:
            self.pnl_pct = (self.entry_price - price) / self.entry_price
            self.lowest_since_entry = min(self.lowest_since_entry, price)
        self.pnl = self.pnl_pct * self.size * self.entry_price

    @property
    def hold_seconds(self) -> int:
        return int(time.time()) - self.open_time


class CircuitBreaker:
    def __init__(
        self,
        max_daily_loss_pct: float,
        max_drawdown_pct: float,
        pause_seconds: float = 3600.0,
    ) -> None:
        self._max_daily_loss = max_daily_loss_pct
        self._max_drawdown = max_drawdown_pct
        self._pause_seconds = pause_seconds
        self._daily_loss = 0.0
        self._peak_equity = 0.0
        self._tripped = False
        self._trip_reason = ""
        self._trip_time: float = 0.0

    def reset_daily(self) -> None:
        self._daily_loss = 0.0

    def record_pnl(self, pnl_pct: float, equity: float) -> bool:
        self._daily_loss += min(0, pnl_pct)
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if abs(self._daily_loss) >= self._max_daily_loss:
            self._tripped = True
            self._trip_reason = f"daily_loss={self._daily_loss:.2%}"
            self._trip_time = time.time()
        if drawdown >= self._max_drawdown:
            self._tripped = True
            self._trip_reason = f"drawdown={drawdown:.2%}"
            self._trip_time = time.time()
        return self._tripped

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_elapsed(self) -> float:
        """Seconds since the circuit breaker was tripped."""
        if not self._tripped:
            return 0.0
        return time.time() - self._trip_time

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def reset(self) -> None:
        self._tripped = False
        self._trip_reason = ""
        self._daily_loss = 0.0

    def trip(self, reason: str, pause_seconds: float | None = None) -> None:
        """Trip the circuit breaker with a reason. Use this instead of direct attribute mutation."""
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = time.time()
        if pause_seconds is not None:
            self._pause_seconds = pause_seconds
        logger.warning("Circuit breaker TRIPPED: {}", reason)

    def clear_if_reason(self, reason: str) -> bool:
        """Clear the trip only if current reason matches. Returns True if cleared."""
        if self._tripped and self._trip_reason == reason:
            self._tripped = False
            self._trip_reason = ""
            return True
        return False


class RiskManager:
    def __init__(self, config: Config, event_bus: EventBus, safe_mode: SafeModeManager | None = None, sqlite_store: Any | None = None) -> None:
        self.config = config
        self.event_bus = event_bus
        self.safe_mode = safe_mode or SafeModeManager()
        self._sqlite_store = sqlite_store
        self._closed_trades: list[dict] = []  # in-memory closed trade log
        risk_cfg = config.get_value("risk") or {}

        # ── Position sizing (V6.0: 1.5% hard cap) ─────────────────────
        self._max_position_pct = float(risk_cfg.get("max_position_size_pct", 0.015))
        self._risk_per_trade = float(risk_cfg.get("risk_per_trade_pct", 0.01))
        self._sizing_method = str(risk_cfg.get("sizing_method", "risk_based"))  # risk_based | fixed_fraction
        self._kelly_fraction = float(risk_cfg.get("kelly_fraction", 0.25))
        self._max_open = int(risk_cfg.get("max_open_positions", 5))
        self._leverage = float(risk_cfg.get("default_leverage", 1.0))

        # ── Pre-trade filters ─────────────────────────────────────────────
        self._max_spread_bps = float(risk_cfg.get("max_spread_bps", 10.0))
        self._max_atr_pct = float(risk_cfg.get("max_atr_pct", 0.05))
        self._max_exposure_per_symbol = float(risk_cfg.get("max_exposure_per_symbol_pct", 0.10))  # Match settings.yaml default
        # V6.0: total portfolio exposure hard limit
        self._max_total_exposure_pct = float(risk_cfg.get("max_total_exposure_pct", 0.06))
        self._cooldown_seconds = float(risk_cfg.get("cooldown_seconds", 300.0))
        self._session_start_utc = str(risk_cfg.get("session_start_utc", "00:00"))
        self._session_end_utc = str(risk_cfg.get("session_end_utc", "23:59"))

        # ── Position management ───────────────────────────────────────────
        self._atr_sl_multiplier = float(risk_cfg.get("atr_sl_multiplier", 1.5))
        self._rr_ratio = float(risk_cfg.get("rr_ratio", 2.0))
        self._trailing_activation_atr = float(risk_cfg.get("trailing_activation_atr", 2.0))
        self._trailing_distance_atr = float(risk_cfg.get("trailing_distance_atr", 1.0))
        self._breakeven_trigger_atr = float(risk_cfg.get("breakeven_trigger_atr", 1.0))
        self._max_hold_minutes = int(risk_cfg.get("max_hold_minutes", 0))  # 0 = disabled

        # ── VaR ───────────────────────────────────────────────────────────
        self._max_portfolio_var_pct = float(risk_cfg.get("max_portfolio_var_pct", 0.08))
        self._returns_window = int(risk_cfg.get("returns_window", 250))
        self._var_min_history = int(risk_cfg.get("var_min_history", 30))

        # ── New risk filters (V6.0: funding rate 2bps) ─────────────────────
        self._max_funding_rate_bps = float(risk_cfg.get("max_funding_rate_bps", 2.0))
        self._min_orderbook_depth_usd = float(risk_cfg.get("min_orderbook_depth_usd", 50_000.0))
        self._max_order_size_usd = float(risk_cfg.get("max_order_size_usd", 500_000.0))
        self._max_leverage_per_symbol: dict[str, float] = risk_cfg.get("max_leverage_per_symbol", {})
        self._min_signal_score = float(risk_cfg.get("min_signal_score", 0.5))
        self._min_risk_reward = float(risk_cfg.get("min_risk_reward", 1.5))

        # ── State ─────────────────────────────────────────────────────────
        self._positions: dict[str, Position] = {}
        self._equity = float(risk_cfg.get("initial_equity", 100_000.0))
        self._return_history: list[float] = []
        self._last_trade_time: dict[str, float] = {}
        self._killed = False
        self._current_funding_rates: dict[str, float] = {}  # symbol → rate (decimal)
        self._current_orderbook_depth: dict[str, float] = {}  # symbol → top-of-book USD
        self._kill_switch_file = Path("data/.kill_switch")
        self._kill_switch_cache: bool = False
        self._kill_switch_last_check: float = 0.0
        self._kill_switch_check_interval: float = 5.0  # seconds
        self._circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.03)),
            max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.12)),
            pause_seconds=float(risk_cfg.get("circuit_breaker_pause_seconds", 86400)),  # §9: 24h halt on daily loss
        )
        self._running = False
        self._lock = asyncio.Lock()
        self._equity_unreconciled = False
        # Portfolio-level order rate limit
        self._max_orders_per_window = int(risk_cfg.get("max_orders_per_window", 5))
        self._order_window_seconds = float(risk_cfg.get("order_window_seconds", 10.0))
        self._recent_order_times: list[float] = []

        # ── ARMS-V2.1: Drawdown recovery protocol (4 phases) ─────────────
        self._drawdown_phases = {
            "normal": {"max_dd": 0.05, "size_mult": 1.0, "max_positions": self._max_open},
            "caution": {"max_dd": 0.08, "size_mult": 0.5, "max_positions": max(1, self._max_open - 2)},
            "defensive": {"max_dd": 0.12, "size_mult": 0.25, "max_positions": 1},
            "emergency": {"max_dd": 1.0, "size_mult": 0.0, "max_positions": 0},
        }
        self._current_dd_phase = "normal"

        # ── ARMS-V2.1: ATR percentile-based volatility scaling ────────────
        self._atr_pctile_bands = [
            (25, 1.25),   # <25th percentile → +25%
            (50, 1.0),    # 25-50th → normal
            (75, 0.75),   # 50-75th → -25%
            (90, 0.50),   # 75-90th → -50%
            (100, 0.25),  # >90th → -75%
        ]

        # ── ARMS-V2.1: Adaptive SL multiplier by ATR percentile ──────────
        self._atr_sl_bands = [
            (25, 1.2),    # tight markets → tight stop
            (50, 1.5),    # normal
            (75, 2.0),    # volatile → wider
            (100, 2.5),   # very volatile → very wide
        ]

        # ── V1.0 Spec: Tiered take-profit (TP1: 1:1 50%, TP2: zone 30%, TP3: trail 20%) ──
        self._tp_tiers = [
            {"r_multiple": 1.0, "close_pct": 0.50},  # TP1: 1:1 R:R, close 50%
            {"r_multiple": 2.5, "close_pct": 0.30},  # TP2: next liquidity zone / 2.5R fallback, close 30%
            # TP3: remaining 20% — SuperTrend trailing
        ]

        # ── ARMS-V2.1: Session time filters ──────────────────────────────
        self._sessions = {
            "asia": ("00:00", "08:00"),
            "london": ("07:00", "16:00"),
            "new_york": ("13:00", "22:00"),
            "asia_london_overlap": ("07:00", "08:00"),
            "london_ny_overlap": ("13:00", "16:00"),
        }

        # ── ARMS-V2.1: Correlation tracking ──────────────────────────────
        self._price_history: dict[str, list[float]] = {}
        self._correlation_window = 30  # days
        self._max_correlated_exposure = 0.15  # 15% of equity

        # ── ARMS-V2.1: Funding rate directional bias (graduated) ─────────
        self._funding_bias_tiers = [
            (10.0, 1.0),    # < 10bps — no reduction
            (25.0, 0.75),   # 10-25bps — 25% reduction
            (50.0, 0.50),   # 25-50bps — 50% reduction
            (100.0, 0.0),   # > 50bps — block
        ]

        # ── ARMS-V2.1: Liquidation safety layer ──────────────────────────
        self._liq_warning_distance_pct = 0.15   # 15% from liq → warning
        self._liq_partial_close_pct = 0.10      # 10% from liq → partial close
        self._liq_emergency_close_pct = 0.05    # 5% from liq → full close

        # ── ARMS-V2.1: Per-tier base risk (§8) ───────────────────────────
        arms_cfg = (config.get_value("arms") or {}).get("risk", {})
        self._tier_risk_pct: dict[int, float] = {
            1: float(arms_cfg.get("tier1_risk_pct", 0.0075)),   # 0.75%
            2: float(arms_cfg.get("tier2_risk_pct", 0.005)),    # 0.50%
            3: float(arms_cfg.get("tier3_risk_pct", 0.0025)),   # 0.25%
            4: 0.0,
        }

        # ── ARMS-V2.1: Weekly/monthly loss caps (§10) ────────────────────
        self._weekly_loss: float = 0.0
        self._monthly_loss: float = 0.0
        self._weekly_loss_limit = float(arms_cfg.get("weekly_loss_limit_pct", 0.08))  # §9: 8% weekly DD limit
        self._monthly_loss_limit = float(arms_cfg.get("monthly_loss_limit_pct", 0.10))  # 10%
        self._last_weekly_reset: float = time.time()
        self._last_monthly_reset: float = time.time()

        # ── ARMS-V2.1: Drawdown graduation (§11) — 3 consecutive profits ─
        self._consecutive_profitable: int = 0
        self._dd_graduation_threshold: int = int(arms_cfg.get("dd_graduation_trades", 3))

        # ── ARMS-V2.1: Correlation groups (§8) ───────────────────────────
        default_groups = {
            "btc_group": ["BTC/USDT", "BTC/USDC", "BTC/USD", "BTCUSDT", "BTCUSDC", "BTC/USDT:USDT"],
            "eth_group": ["ETH/USDT", "ETH/USDC", "ETH/USD", "ETHUSDT", "ETHUSDC", "ETH/USDT:USDT"],
            "sol_group": ["SOL/USDT", "SOL/USDC", "SOLUSDT", "SOL/USDT:USDT"],
        }
        self._correlation_groups: dict[str, list[str]] = arms_cfg.get("correlation_groups", default_groups)
        self._max_group_exposure_pct = float(arms_cfg.get("max_group_exposure_pct", 0.15))

        # ── ARMS-V2.1: 8h funding re-check (§9) ─────────────────────────
        self._funding_recheck_interval = float(arms_cfg.get("funding_recheck_seconds", 28800))  # 8h
        self._last_funding_recheck: float = 0.0

        # ── ARMS-V2.1: ADX/ATR-linked max leverage (§10) ────────────────
        self._adx_leverage_bands = [
            (20, 5.0),   # Low ADX → max 5x
            (30, 3.0),   # Medium ADX → max 3x
            (50, 2.0),   # High ADX → max 2x
            (100, 1.0),  # Very high → 1x only
        ]
        self._atr_leverage_bands = [
            (50, 5.0),   # Low ATR pctile → max 5x
            (75, 3.0),   # Moderate → max 3x
            (90, 2.0),   # High → max 2x
            (100, 1.0),  # Extreme → 1x only
        ]

        # ── ARMS-V2.1: Event blackout (§12) ──────────────────────────────
        self._event_blackout_keywords = [
            "FOMC", "CPI", "NFP", "Non-Farm", "Fed Rate",
            "ECB Rate", "BOJ Rate", "GDP",
        ]
        self._event_blackout_minutes_before = int(arms_cfg.get("event_blackout_before_min", 30))

        # ── §13 Guardrails: Consecutive loss circuit breaker (V6.0: 5) ──
        self._consecutive_losses: int = 0
        self._max_consecutive_losses: int = int(risk_cfg.get("max_consecutive_losses", 5))

        # ── §13 Guardrails: Win-rate circuit breaker (V6.0: <55% over 20) ─
        self._win_rate_lookback = int(risk_cfg.get("win_rate_circuit_lookback", 20))
        self._win_rate_min = float(risk_cfg.get("win_rate_circuit_min", 0.55))
        self._win_rate_breaker_tripped = False

        # ── §13 Guardrails: Flash crash detection ─────────────────────────
        self._flash_crash_threshold_pct = float(risk_cfg.get("flash_crash_threshold_pct", 0.05))  # 5% in 5min
        self._flash_crash_window_seconds = float(risk_cfg.get("flash_crash_window_seconds", 300.0))
        self._flash_crash_tripped = False
        self._price_snapshots: list[tuple[float, float]] = []  # (timestamp, price)

        # ── §9 Risk: Kelly Criterion activation ──────────────────────────
        self._kelly_enabled = bool(risk_cfg.get("kelly_enabled", True))
        self._kelly_win_rate: float = 0.5
        self._kelly_avg_win: float = 0.02
        self._kelly_avg_loss: float = 0.01
        self._trade_results: list[float] = []  # PnL history for rolling Kelly
        self._event_blackout_minutes_after = int(arms_cfg.get("event_blackout_after_min", 15))
        self._upcoming_events: list[dict[str, Any]] = []  # [{name, timestamp}]

        # ── ARMS-V2.1: Weekend rules (§12) ───────────────────────────────
        self._weekend_sizing_mult = float(arms_cfg.get("weekend_sizing_mult", 0.25))
        self._weekend_halt = bool(arms_cfg.get("weekend_halt", False))

        # ── ARMS-V2.1: Margin mode tracking (§10.1) ─────────────────────
        self._margin_modes: dict[str, str] = {}  # symbol → "isolated" | "cross"

        # ── ARMS-V2.1: 60s liquidation monitoring (§10.1) ────────────────
        self._liq_monitor_interval = float(arms_cfg.get("liq_monitor_interval_seconds", 60.0))
        self._last_liq_check: float = 0.0

    # ── Risk-based position sizing ────────────────────────────────────────
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """size = (equity × risk_per_trade) / |entry − stop|"""
        if signal.price <= 0:
            return 0.0
        risk_distance = abs(signal.price - signal.stop_loss)
        if risk_distance == 0:
            return 0.0

        if self._sizing_method == "risk_based":
            notional = (self._equity * self._risk_per_trade) / (risk_distance / signal.price)
        else:
            # Fixed-fraction fallback
            notional = self._equity * self._max_position_pct

        notional *= self._leverage

        # Enforce per-symbol leverage cap if configured
        if self._max_leverage_per_symbol and hasattr(signal, 'symbol'):
            sym_max_lev = self._max_leverage_per_symbol.get(signal.symbol)
            if sym_max_lev is not None and self._leverage > sym_max_lev:
                notional = notional * (sym_max_lev / self._leverage)

        # Cap at max position pct of equity
        max_notional = self._equity * self._max_position_pct * self._leverage
        notional = min(notional, max_notional)
        return max(0.0, notional)

    # ── ARMS-V2.1: ATR percentile volatility scaling ─────────────────────
    def get_atr_volatility_multiplier(self, atr_percentile: float) -> float:
        """Returns sizing multiplier based on ATR percentile band."""
        for threshold, mult in self._atr_pctile_bands:
            if atr_percentile <= threshold:
                return mult
        return 0.25

    # ── ARMS-V2.1: Adaptive SL multiplier ────────────────────────────────
    def get_adaptive_sl_multiplier(self, atr_percentile: float) -> float:
        """Returns SL multiplier adjusted for volatility regime."""
        for threshold, mult in self._atr_sl_bands:
            if atr_percentile <= threshold:
                return mult
        return 2.5

    # ── ARMS-V2.1: Drawdown recovery phase ───────────────────────────────
    def update_drawdown_phase(self) -> str:
        """Update and return current drawdown recovery phase.

        Phases escalate automatically by DD threshold but only de-escalate
        via consecutive profitable trades (graduation rule §11).
        """
        dd = self.current_drawdown_pct()
        old_phase = self._current_dd_phase
        phase_order = ["normal", "caution", "defensive", "emergency"]

        # Determine the phase dictated by current drawdown level
        if dd < self._drawdown_phases["normal"]["max_dd"]:
            dd_phase = "normal"
        elif dd < self._drawdown_phases["caution"]["max_dd"]:
            dd_phase = "caution"
        elif dd < self._drawdown_phases["defensive"]["max_dd"]:
            dd_phase = "defensive"
        else:
            dd_phase = "emergency"

        # Only escalate automatically; de-escalation requires graduation
        old_idx = phase_order.index(old_phase)
        new_idx = phase_order.index(dd_phase)
        if new_idx > old_idx:
            # Escalate
            self._current_dd_phase = dd_phase
            self._consecutive_profitable = 0  # Reset on escalation
        # De-escalation handled by record_trade_result → _graduate_dd_phase

        if old_phase != self._current_dd_phase:
            logger.warning("Drawdown phase: {} → {} (dd={:.2%})", old_phase, self._current_dd_phase, dd)
        return self._current_dd_phase

    def get_drawdown_size_multiplier(self) -> float:
        phase = self.update_drawdown_phase()
        return float(self._drawdown_phases[phase]["size_mult"])

    def get_drawdown_max_positions(self) -> int:
        phase = self.update_drawdown_phase()
        return int(self._drawdown_phases[phase]["max_positions"])

    # ── ARMS-V2.1: Tiered take-profit levels ─────────────────────────────
    def compute_tiered_tp(
        self, entry_price: float, sl_distance: float, is_long: bool,
        liquidity_zone: float | None = None,
        supertrend_level: float | None = None,
    ) -> list[dict[str, float]]:
        """Compute TP1, TP2, TP3 (trail) levels from risk distance.

        V1.0 Spec:
        - TP1: 1:1 Risk/Reward (50% position closed)
        - TP2: Next FVG/Liquidity zone (30% closed), fallback 2.5R
        - TP3: SuperTrend trailing stop (20% remaining)
        """
        tps: list[dict[str, float]] = []
        for tier in self._tp_tiers:
            r_dist = sl_distance * tier["r_multiple"]
            if is_long:
                tp_price = entry_price + r_dist
            else:
                tp_price = entry_price - r_dist
            tps.append({"price": tp_price, "close_pct": tier["close_pct"]})

        # TP2: override with liquidity zone if available and better than default
        if liquidity_zone is not None and len(tps) >= 2:
            if is_long and liquidity_zone > tps[0]["price"]:
                tps[1]["price"] = liquidity_zone
            elif not is_long and liquidity_zone < tps[0]["price"]:
                tps[1]["price"] = liquidity_zone

        # TP3: SuperTrend trailing (remaining 20%)
        remaining = 1.0 - sum(t["close_pct"] for t in tps)
        trail_entry: dict[str, Any] = {
            "price": supertrend_level if supertrend_level else 0.0,
            "close_pct": remaining,
            "trail": True,
        }
        tps.append(trail_entry)
        return tps

    # ── ARMS-V2.1: Session filter ────────────────────────────────────────
    def get_active_sessions(self) -> list[str]:
        """Returns list of currently active trading sessions."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        current_time = now.strftime("%H:%M")
        active = []
        for name, (start, end) in self._sessions.items():
            if start <= end:
                if start <= current_time <= end:
                    active.append(name)
            else:  # wraps midnight
                if current_time >= start or current_time <= end:
                    active.append(name)
        return active

    # ── ARMS-V2.1: Funding rate directional bias ─────────────────────────
    def get_funding_size_multiplier(self, symbol: str, direction: str) -> float:
        """Graduated funding rate penalty. Returns sizing multiplier 0.0–1.0."""
        rate = self._current_funding_rates.get(symbol, 0.0)
        rate_bps = abs(rate) * 10000

        # Direction check: funding positive = longs pay → penalize longs
        if rate > 0 and direction == "long":
            pass  # adverse
        elif rate < 0 and direction == "short":
            pass  # adverse
        else:
            return 1.0  # favorable or neutral

        for threshold, mult in self._funding_bias_tiers:
            if rate_bps <= threshold:
                return mult
        return 0.0

    # ── ARMS-V2.1: Correlation-adjusted exposure ─────────────────────────
    def update_price_history(self, symbol: str, price: float) -> None:
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)
        # Keep only correlation_window days worth (assuming ~24 data points per day)
        max_len = self._correlation_window * 24
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

    def compute_correlation(self, sym_a: str, sym_b: str) -> float:
        """30-day Pearson correlation between two symbols."""
        hist_a = self._price_history.get(sym_a, [])
        hist_b = self._price_history.get(sym_b, [])
        if len(hist_a) < 30 or len(hist_b) < 30:
            return 0.0
        min_len = min(len(hist_a), len(hist_b))
        a = np.array(hist_a[-min_len:])
        b = np.array(hist_b[-min_len:])
        ret_a = np.diff(a) / a[:-1]
        ret_b = np.diff(b) / b[:-1]
        if len(ret_a) < 10:
            return 0.0
        cc = np.corrcoef(ret_a, ret_b)
        val = float(cc[0, 1])
        return val if not np.isnan(val) else 0.0

    def check_correlation_exposure(self, symbol: str, direction: str) -> tuple[bool, str]:
        """Check if adding this position would create excessive correlated exposure."""
        for key, pos in self._positions.items():
            other_sym = pos.symbol
            if other_sym == symbol:
                continue
            corr = self.compute_correlation(symbol, other_sym)
            # High positive correlation and same direction → concentrated risk
            if corr > 0.7 and pos.direction == direction:
                combined_exposure = (
                    abs(pos.size * pos.current_price)
                    + self._equity * self._max_position_pct
                )
                if combined_exposure / self._equity > self._max_correlated_exposure:
                    return False, f"correlated_with_{other_sym}(r={corr:.2f})"
        return True, ""

    # ── ARMS-V2.1: Liquidation safety layer ──────────────────────────────
    def check_liquidation_distance(self, pos: Position) -> str | None:
        """Returns action needed based on distance to estimated liquidation price.
        Returns: None, 'warning', 'partial_close', 'emergency_close'
        """
        if self._leverage <= 1.0:
            return None

        # Estimate liquidation price
        margin_ratio = 1.0 / self._leverage
        if pos.is_long:
            liq_price = pos.entry_price * (1 - margin_ratio * 0.9)
            distance_pct = (pos.current_price - liq_price) / pos.current_price if pos.current_price > 0 else 1.0
        else:
            liq_price = pos.entry_price * (1 + margin_ratio * 0.9)
            distance_pct = (liq_price - pos.current_price) / pos.current_price if pos.current_price > 0 else 1.0

        if distance_pct <= self._liq_emergency_close_pct:
            return "emergency_close"
        if distance_pct <= self._liq_partial_close_pct:
            return "partial_close"
        if distance_pct <= self._liq_warning_distance_pct:
            return "warning"
        return None

    # ── ARMS-V2.1: Per-tier risk sizing (§8) ─────────────────────────────
    def get_tier_risk_pct(self, tier: int) -> float:
        """Returns risk-per-trade % for the given pair tier."""
        return self._tier_risk_pct.get(tier, self._risk_per_trade)

    # ── ARMS-V2.1: Weekly/monthly loss checks (§10) ──────────────────────
    def check_weekly_monthly_limits(self) -> tuple[bool, str]:
        """Check if weekly or monthly loss limits have been breached."""
        now = time.time()
        # Auto-reset weekly (every 7 days)
        if now - self._last_weekly_reset >= 7 * 86400:
            self._weekly_loss = 0.0
            self._last_weekly_reset = now
        # Auto-reset monthly (every 30 days)
        if now - self._last_monthly_reset >= 30 * 86400:
            self._monthly_loss = 0.0
            self._last_monthly_reset = now

        if abs(self._weekly_loss) >= self._weekly_loss_limit:
            return False, f"weekly_loss_limit ({self._weekly_loss:.2%} >= {self._weekly_loss_limit:.2%})"
        if abs(self._monthly_loss) >= self._monthly_loss_limit:
            return False, f"monthly_loss_limit ({self._monthly_loss:.2%} >= {self._monthly_loss_limit:.2%})"
        return True, ""

    def _record_loss_for_period(self, pnl_pct: float) -> None:
        """Accumulate losses into weekly/monthly trackers."""
        if pnl_pct < 0:
            self._weekly_loss += pnl_pct
            self._monthly_loss += pnl_pct

    # ── ARMS-V2.1: Drawdown graduation via consecutive profits (§11) ────
    def record_trade_result(self, profitable: bool) -> None:
        """Track consecutive profitable trades for DD phase graduation."""
        if profitable:
            self._consecutive_profitable += 1
            if (
                self._consecutive_profitable >= self._dd_graduation_threshold
                and self._current_dd_phase != "normal"
            ):
                self._graduate_dd_phase()
                self._consecutive_profitable = 0
        else:
            self._consecutive_profitable = 0

    def _graduate_dd_phase(self) -> None:
        """Step back one drawdown phase after consecutive profitable trades."""
        phase_order = ["emergency", "defensive", "caution", "normal"]
        try:
            idx = phase_order.index(self._current_dd_phase)
        except ValueError:
            return
        if idx < len(phase_order) - 1:
            new_phase = phase_order[idx + 1]
            logger.info(
                "DD graduation: {} → {} after {} consecutive profits",
                self._current_dd_phase, new_phase, self._dd_graduation_threshold,
            )
            self._current_dd_phase = new_phase

    # ── ARMS-V2.1: Correlation group check (§8) ──────────────────────────
    def _get_correlation_group(self, symbol: str) -> str | None:
        """Returns the group name if symbol belongs to a correlation group."""
        for group_name, symbols in self._correlation_groups.items():
            if symbol in symbols:
                return group_name
        return None

    def check_group_exposure(self, symbol: str) -> tuple[bool, str]:
        """Check BTC/ETH correlation group hard limit (§8)."""
        group = self._get_correlation_group(symbol)
        if group is None:
            return True, ""
        # Sum notional of all open positions in same group
        group_symbols = set(self._correlation_groups[group])
        group_notional = 0.0
        for key, pos in self._positions.items():
            if pos.symbol in group_symbols:
                group_notional += abs(pos.size * pos.current_price)
        max_notional = self._equity * self._max_group_exposure_pct
        if group_notional >= max_notional:
            return False, f"{group}_exposure_limit ({group_notional:.0f} >= {max_notional:.0f})"
        return True, ""

    # ── ARMS-V2.1: 8h funding re-check for existing positions (§9) ──────
    def check_funding_existing_positions(self) -> list[dict[str, Any]]:
        """Periodic 8h check: reduce positions if funding now adverse."""
        now = time.time()
        if now - self._last_funding_recheck < self._funding_recheck_interval:
            return []
        self._last_funding_recheck = now
        actions: list[dict[str, Any]] = []
        for key, pos in self._positions.items():
            mult = self.get_funding_size_multiplier(pos.symbol, pos.direction)
            if mult < 0.5:
                actions.append({
                    "symbol": pos.symbol,
                    "action": "reduce",
                    "funding_mult": mult,
                    "reason": f"funding_adverse_for_{pos.direction}",
                })
                logger.warning(
                    "8h funding re-check: {} {} — mult={:.2f}, suggesting reduction",
                    pos.symbol, pos.direction, mult,
                )
        return actions

    # ── ARMS-V2.1: ADX/ATR-linked dynamic leverage (§10) ────────────────
    def get_dynamic_leverage(self, adx: float, atr_percentile: float) -> float:
        """Returns max allowed leverage based on ADX and ATR conditions."""
        adx_max_lev = self._leverage
        for threshold, max_lev in self._adx_leverage_bands:
            if adx <= threshold:
                adx_max_lev = max_lev
                break

        atr_max_lev = self._leverage
        for threshold, max_lev in self._atr_leverage_bands:
            if atr_percentile <= threshold:
                atr_max_lev = max_lev
                break

        return min(adx_max_lev, atr_max_lev, self._leverage)

    # ── ARMS-V2.1: 3×ATR(4H) liquidation distance (§10.1) ──────────────
    def check_atr_liq_distance(self, pos: Position, atr_4h: float) -> str | None:
        """Check if position is within 3×ATR(4H) of estimated liquidation price."""
        if self._leverage <= 1.0 or atr_4h <= 0:
            return None
        margin_ratio = 1.0 / self._leverage
        if pos.is_long:
            liq_price = pos.entry_price * (1 - margin_ratio * 0.9)
            distance = pos.current_price - liq_price
        else:
            liq_price = pos.entry_price * (1 + margin_ratio * 0.9)
            distance = liq_price - pos.current_price

        min_distance = 3.0 * atr_4h
        if distance < min_distance:
            if distance < atr_4h:
                return "emergency_close"
            elif distance < 2 * atr_4h:
                return "partial_close"
            else:
                return "warning"
        return None

    # ── ARMS-V2.1: Event blackout (§12) ──────────────────────────────────
    def add_upcoming_event(self, name: str, timestamp: float) -> None:
        """Register an upcoming high-impact event for blackout checking."""
        self._upcoming_events.append({"name": name, "timestamp": timestamp})
        # Prune past events
        now = time.time()
        self._upcoming_events = [
            e for e in self._upcoming_events
            if e["timestamp"] > now - self._event_blackout_minutes_after * 60
        ]

    def is_event_blackout(self) -> tuple[bool, str]:
        """Check if we're within an event blackout window."""
        now = time.time()
        for event in self._upcoming_events:
            before_start = event["timestamp"] - self._event_blackout_minutes_before * 60
            after_end = event["timestamp"] + self._event_blackout_minutes_after * 60
            if before_start <= now <= after_end:
                return True, f"event_blackout:{event['name']}"
        return False, ""

    # ── ARMS-V2.1: Weekend rules (§12) ───────────────────────────────────
    def is_weekend(self) -> bool:
        """Check if current day is Saturday or Sunday (UTC)."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return now.weekday() >= 5  # 5=Saturday, 6=Sunday

    def get_weekend_sizing_mult(self) -> float:
        """Returns weekend sizing multiplier (1.0 on weekdays)."""
        if self.is_weekend():
            if self._weekend_halt:
                return 0.0
            return self._weekend_sizing_mult
        return 1.0

    # ── ARMS-V2.1: Margin mode (§10.1) ──────────────────────────────────
    def set_margin_mode(self, symbol: str, mode: str) -> None:
        """Set margin mode for symbol: 'isolated' or 'cross'."""
        self._margin_modes[symbol] = mode

    def get_margin_mode(self, symbol: str) -> str:
        """Returns margin mode for symbol (default: 'isolated')."""
        return self._margin_modes.get(symbol, "isolated")

    # ── ARMS-V2.1: Periodic liquidation monitoring (§10.1) ──────────────
    def run_periodic_liq_check(self) -> list[dict[str, Any]]:
        """60-second interval liquidation check across all positions."""
        now = time.time()
        if now - self._last_liq_check < self._liq_monitor_interval:
            return []
        self._last_liq_check = now
        actions: list[dict[str, Any]] = []
        for key, pos in self._positions.items():
            action = self.check_liquidation_distance(pos)
            if action:
                actions.append({
                    "key": key,
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "action": action,
                    "margin_mode": self.get_margin_mode(pos.symbol),
                })
        return actions

    # ── Pre-trade gate checks ─────────────────────────────────────────────
    def _check_session_window(self) -> tuple[bool, str]:
        """Check if current UTC time is within trading session."""
        if self._session_start_utc == "00:00" and self._session_end_utc == "23:59":
            return True, ""
        now = time.gmtime()
        current_minutes = now.tm_hour * 60 + now.tm_min
        start_parts = self._session_start_utc.split(":")
        end_parts = self._session_end_utc.split(":")
        start_min = int(start_parts[0]) * 60 + int(start_parts[1])
        end_min = int(end_parts[0]) * 60 + int(end_parts[1])
        if start_min <= end_min:
            in_session = start_min <= current_minutes <= end_min
        else:
            in_session = current_minutes >= start_min or current_minutes <= end_min
        if not in_session:
            return False, f"outside_session ({self._session_start_utc}-{self._session_end_utc})"
        return True, ""

    def _check_cooldown(self, symbol: str) -> tuple[bool, str]:
        """Per-symbol post-trade cooldown."""
        last = self._last_trade_time.get(symbol, 0.0)
        elapsed = time.time() - last
        if elapsed < self._cooldown_seconds:
            remaining = self._cooldown_seconds - elapsed
            return False, f"cooldown ({remaining:.0f}s remaining for {symbol})"
        return True, ""

    def _check_spread(self, signal: TradingSignal) -> tuple[bool, str]:
        """Spread filter: reject if spread > max_spread_bps."""
        spread_bps = float(signal.metadata.get("spread_bps", 0.0))
        if spread_bps > 0 and spread_bps > self._max_spread_bps:
            return False, f"spread_too_wide ({spread_bps:.1f} > {self._max_spread_bps:.1f} bps)"
        return True, ""

    def _check_volatility(self, signal: TradingSignal) -> tuple[bool, str]:
        """Volatility filter: ATR(14)/price < max_atr_pct."""
        if signal.price <= 0:
            return True, ""
        atr_val = signal.atr if signal.atr and signal.atr > 0 else 0.0
        if atr_val != atr_val:  # NaN guard
            atr_val = 0.0
        if atr_val <= 0:
            return True, ""
        atr_pct = atr_val / signal.price
        if atr_pct > self._max_atr_pct:
            return False, f"volatility_too_high (atr_pct={atr_pct:.4f} > {self._max_atr_pct:.4f})"
        return True, ""

    def _check_symbol_exposure(self, signal: TradingSignal, proposed_size: float) -> tuple[bool, str]:
        """Max exposure per symbol."""
        max_exposure = self._equity * self._max_exposure_per_symbol
        if proposed_size > max_exposure:
            return False, f"symbol_exposure_exceeded ({proposed_size:.0f} > {max_exposure:.0f})"
        return True, ""

    def _check_funding_rate(self, signal: TradingSignal) -> tuple[bool, str]:
        """Funding rate filter: skip trading when funding is extreme."""
        rate = self._current_funding_rates.get(signal.symbol, 0.0)
        rate_bps = abs(rate) * 10_000
        if rate_bps > self._max_funding_rate_bps:
            return False, f"funding_rate_extreme ({rate_bps:.1f} > {self._max_funding_rate_bps:.1f} bps)"
        return True, ""

    def _check_orderbook_depth(self, signal: TradingSignal) -> tuple[bool, str]:
        """Order book depth filter: reject if top-of-book liquidity insufficient."""
        depth_usd = self._current_orderbook_depth.get(signal.symbol, float("inf"))
        if depth_usd < self._min_orderbook_depth_usd:
            return False, f"orderbook_depth_insufficient ({depth_usd:.0f} < {self._min_orderbook_depth_usd:.0f} USD)"
        return True, ""

    def _check_max_order_size(self, proposed_size: float) -> tuple[bool, str]:
        """Hard max order size cap per trade."""
        if proposed_size > self._max_order_size_usd:
            return False, f"max_order_size_exceeded ({proposed_size:.0f} > {self._max_order_size_usd:.0f} USD)"
        return True, ""

    def _check_liquidation_risk(self, signal: TradingSignal) -> tuple[bool, str]:
        """Refuse trades that place liquidation price too close to entry."""
        if self._leverage <= 1.0:
            return True, ""  # No leverage = no liquidation risk
        # Liquidation price ≈ entry × (1 - 1/leverage) for longs
        # Reject if stop_loss is beyond estimated liquidation price
        liq_margin = 1.0 / self._leverage
        if signal.is_long:
            est_liq = signal.price * (1 - liq_margin * 0.9)  # 90% of margin = danger zone
            if signal.stop_loss <= est_liq:
                return False, f"liquidation_risk (SL={signal.stop_loss:.2f} <= est_liq={est_liq:.2f})"
        else:
            est_liq = signal.price * (1 + liq_margin * 0.9)
            if signal.stop_loss >= est_liq:
                return False, f"liquidation_risk (SL={signal.stop_loss:.2f} >= est_liq={est_liq:.2f})"
        return True, ""

    def _publish_risk_decision(self, signal: TradingSignal, *, approved: bool, reason: str, size: float) -> None:
        """Emit RISK_DECISION (always) + RISK_BLOCK (when rejected) for the audit log."""
        try:
            payload = {
                "exchange": getattr(signal, "exchange", ""),
                "symbol": getattr(signal, "symbol", ""),
                "direction": getattr(signal, "direction", ""),
                "score": float(getattr(signal, "score", 0.0) or 0.0),
                "approved": bool(approved),
                "reason": str(reason or ""),
                "size_usd": float(size or 0.0),
                "ts": time.time(),
            }
            self.event_bus.publish_nowait("RISK_DECISION", payload)
            if not approved:
                self.event_bus.publish_nowait("RISK_BLOCK", payload)
        except Exception as exc:
            logger.debug("RISK_DECISION publish failed: {}", exc)

    def update_funding_rate(self, symbol: str, rate: float) -> None:
        """Update current funding rate for a symbol (in decimal, e.g., 0.0001)."""
        self._current_funding_rates[symbol] = rate

    def update_orderbook_depth(self, symbol: str, depth_usd: float) -> None:
        """Update current top-of-book depth in USD for a symbol."""
        self._current_orderbook_depth[symbol] = depth_usd

    def approve_signal(self, signal: TradingSignal) -> tuple[bool, str, float]:
        if self._killed:
            return False, "kill_switch_active", 0.0

        # ── Safe mode gate — never open new trades while blind ────────────
        if self.safe_mode.is_active:
            reasons = ", ".join(r.value for r in self.safe_mode.active_reasons)
            return False, f"safe_mode_active ({reasons})", 0.0

        # File-based kill switch — manual override (cached to avoid hot-path I/O)
        now = time.time()
        if now - self._kill_switch_last_check >= self._kill_switch_check_interval:
            self._kill_switch_cache = self._kill_switch_file.exists()
            self._kill_switch_last_check = now
        if self._kill_switch_cache:
            logger.critical("Kill switch FILE detected: {}", self._kill_switch_file)
            self._killed = True
            return False, "kill_switch_file_detected", 0.0

        if self._circuit_breaker.tripped:
            return False, f"circuit_breaker: {self._circuit_breaker.trip_reason}", 0.0

        # V6.0: Win-rate circuit breaker — halt if <55% over last 20 trades
        if len(self._trade_results) >= self._win_rate_lookback:
            recent = self._trade_results[-self._win_rate_lookback:]
            win_count = sum(1 for r in recent if r > 0)
            win_rate = win_count / len(recent)
            if win_rate < self._win_rate_min:
                self._win_rate_breaker_tripped = True
                return False, f"win_rate_breaker ({win_rate:.0%} < {self._win_rate_min:.0%} over {self._win_rate_lookback})", 0.0
            else:
                self._win_rate_breaker_tripped = False

        if self._equity_unreconciled:
            return False, "equity_unreconciled (post kill-switch, awaiting exchange sync)", 0.0

        key = f"{signal.exchange}:{signal.symbol}"
        if key in self._positions:
            return False, "already_in_position", 0.0

        if len(self._positions) >= self._max_open:
            return False, f"max_positions_reached ({self._max_open})", 0.0

        if signal.score < self._min_signal_score:
            return False, f"score_too_low ({signal.score:.2f} < {self._min_signal_score})", 0.0

        if signal.risk_reward < self._min_risk_reward:
            return False, f"poor_risk_reward ({signal.risk_reward:.2f} < {self._min_risk_reward})", 0.0

        # ── New pre-trade filters ─────────────────────────────────────────
        ok, reason = self._check_session_window()
        if not ok:
            return False, reason, 0.0

        # ARMS-V2.1: Weekend rules (§12)
        weekend_mult = self.get_weekend_sizing_mult()
        if weekend_mult <= 0:
            return False, "weekend_halt", 0.0

        # ARMS-V2.1: Event blackout (§12)
        blackout, reason = self.is_event_blackout()
        if blackout:
            return False, reason, 0.0

        # ARMS-V2.1: Weekly/monthly loss caps (§10)
        ok, reason = self.check_weekly_monthly_limits()
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_cooldown(signal.symbol)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_spread(signal)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_volatility(signal)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_funding_rate(signal)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_orderbook_depth(signal)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_liquidation_risk(signal)
        if not ok:
            return False, reason, 0.0

        # ── Position sizing ───────────────────────────────────────────────
        size = self.calculate_position_size(signal)
        if size <= 0:
            return False, "position_size_zero", 0.0

        # ARMS-V2.1: Per-tier risk adjustment (§8)
        tier = int(signal.metadata.get("pair_tier", 0)) if signal.metadata else 0
        if tier > 0:
            tier_risk = self.get_tier_risk_pct(tier)
            if tier_risk > 0 and self._risk_per_trade > 0:
                tier_mult = tier_risk / self._risk_per_trade
                size *= tier_mult

        # Regime-aware Kelly scaling: use signal.regime_confidence and tier
        try:
            regime_conf = float(getattr(signal, "regime_confidence", 1.0) or 1.0)
            tier_sizemult = {1: 1.0, 2: 0.6, 3: 0.3, 4: 0.0}.get(tier, 1.0) if tier else 1.0
            kelly_adj = self.compute_kelly_size(
                base_risk_pct=self._risk_per_trade,
                regime_confidence=regime_conf,
                tier_multiplier=tier_sizemult,
            )
            if self._risk_per_trade > 0:
                size *= (kelly_adj / self._risk_per_trade)
        except Exception as exc:
            logger.debug("Regime-Kelly scaling error: {}", exc)

        ok, reason = self._check_symbol_exposure(signal, size)
        if not ok:
            return False, reason, 0.0

        ok, reason = self._check_max_order_size(size)
        if not ok:
            return False, reason, 0.0

        # ARMS-V2.1: Correlation group check (§8)
        ok, reason = self.check_group_exposure(signal.symbol)
        if not ok:
            return False, reason, 0.0

        # ── VaR check ────────────────────────────────────────────────────
        if len(self._return_history) >= self._var_min_history and self._equity > 0:
            var95 = self.calculate_historical_var(0.95)
            projected_position_impact = size / self._equity
            projected_var = var95 + projected_position_impact
            if projected_var > self._max_portfolio_var_pct:
                return False, (
                    f"var_limit_breach (projected={projected_var:.2%}, "
                    f"limit={self._max_portfolio_var_pct:.2%})"
                ), 0.0
        elif len(self._return_history) < self._var_min_history and self._equity > 0:
            # Insufficient history for VaR — apply conservative half-size cap
            size = min(size, self._equity * self._max_position_pct * self._leverage * 0.5)
            if size <= 0:
                return False, "position_size_zero (VaR cold start)", 0.0

        # ── ARMS-V2.1: Drawdown recovery phase gate ──────────────────────
        dd_phase = self.update_drawdown_phase()
        dd_max_pos = self.get_drawdown_max_positions()
        if len(self._positions) >= dd_max_pos:
            return False, f"drawdown_phase_{dd_phase}_max_positions ({dd_max_pos})", 0.0
        dd_mult = self.get_drawdown_size_multiplier()
        if dd_mult <= 0:
            return False, f"drawdown_phase_{dd_phase}_blocks_trading", 0.0
        size *= dd_mult

        # ── ARMS-V2.1: ATR percentile volatility scaling ─────────────────
        atr_pctile = float(signal.metadata.get("atr_percentile", 50)) if signal.metadata else 50.0
        vol_mult = self.get_atr_volatility_multiplier(atr_pctile)
        size *= vol_mult

        # ── ARMS-V2.1: Funding rate directional bias ─────────────────────
        funding_mult = self.get_funding_size_multiplier(signal.symbol, signal.direction)
        if funding_mult <= 0:
            return False, f"funding_rate_blocks_{signal.direction}", 0.0
        size *= funding_mult

        # ── ARMS-V2.1: Weekend sizing reduction (§12) ────────────────────
        if weekend_mult < 1.0:
            size *= weekend_mult

        # ── ARMS-V2.1: Dynamic leverage cap from ADX/ATR (§10) ───────────
        adx_val = float(signal.metadata.get("adx", 25)) if signal.metadata else 25.0
        atr_pctile_val = float(signal.metadata.get("atr_percentile", 50)) if signal.metadata else 50.0
        dyn_lev = self.get_dynamic_leverage(adx_val, atr_pctile_val)
        if dyn_lev < self._leverage and self._leverage > 0:
            lev_adjustment = dyn_lev / self._leverage
            size *= lev_adjustment

        # ── ARMS-V2.1: Correlation check ─────────────────────────────────
        ok, reason = self.check_correlation_exposure(signal.symbol, signal.direction)
        if not ok:
            return False, reason, 0.0

        if size <= 0:
            return False, "position_size_zero (after ARMS adjustments)", 0.0

        # V6.0: Total portfolio exposure check (after all sizing adjustments)
        if self._equity > 0:
            total_exposure = sum(
                abs(p.entry_price * p.size)
                for p in self._positions.values()
            ) + abs(size)  # size is already notional USD
            total_exposure_pct = total_exposure / self._equity
            if total_exposure_pct > self._max_total_exposure_pct:
                return False, (
                    f"total_exposure_breach ({total_exposure_pct:.2%} > "
                    f"{self._max_total_exposure_pct:.2%})"
                ), 0.0

        return True, "approved", size

    async def approve_and_open(self, signal: TradingSignal) -> tuple[bool, str, float, Position | None]:
        """Atomically approve signal + open position under lock.
        Prevents race where two signals for same symbol both pass approve_signal
        before either calls open_position.
        Returns (approved, reason, size, position_or_None).
        """
        async with self._lock:
            # Portfolio-level order rate limit
            now = time.time()
            cutoff = now - self._order_window_seconds
            self._recent_order_times = [t for t in self._recent_order_times if t > cutoff]
            if len(self._recent_order_times) >= self._max_orders_per_window:
                reason = f"portfolio_rate_limit ({self._max_orders_per_window} orders in {self._order_window_seconds}s)"
                self._publish_risk_decision(signal, approved=False, reason=reason, size=0.0)
                return False, reason, 0.0, None

            approved, reason, size = self.approve_signal(signal)
            self._publish_risk_decision(signal, approved=approved, reason=reason, size=size)
            if not approved:
                return False, reason, 0.0, None
            # Open position inline (already holding lock, skip open_position's lock)
            sl, tp = self.compute_atr_stops(signal)
            pos = Position(
                exchange=signal.exchange,
                symbol=signal.symbol,
                direction=signal.direction,
                size=size / signal.price if signal.price > 0 else 0.0,
                entry_price=signal.price,
                current_price=signal.price,
                stop_loss=sl,
                take_profit=tp,
                open_time=int(time.time()),
                highest_since_entry=signal.price,
                lowest_since_entry=signal.price,
            )
            key = f"{signal.exchange}:{signal.symbol}"
            self._positions[key] = pos
            self._last_trade_time[signal.symbol] = time.time()
            self._recent_order_times.append(time.time())
            logger.info(
                "Position opened: {}/{} {} size={:.4f} @ {:.2f} SL={:.2f} TP={:.2f} (ATR-based)",
                signal.exchange, signal.symbol, signal.direction,
                pos.size, pos.entry_price, pos.stop_loss, pos.take_profit,
            )
            return True, "approved", size, pos

    def compute_atr_stops(
        self, signal: TradingSignal
    ) -> tuple[float, float]:
        """Compute ATR-based SL (with adaptive multiplier) and tiered TP."""
        raw_atr = signal.atr if signal.atr and signal.atr > 0 else 0.0
        # Guard against NaN
        if raw_atr != raw_atr:  # NaN check
            raw_atr = 0.0
        atr = raw_atr if raw_atr > 0 else signal.price * 0.01

        # ARMS-V2.1: Adaptive SL multiplier based on ATR percentile
        atr_pctile = float(signal.metadata.get("atr_percentile", 50)) if signal.metadata else 50.0
        sl_mult = self.get_adaptive_sl_multiplier(atr_pctile)
        sl_distance = atr * sl_mult

        if signal.is_long:
            sl = signal.price - sl_distance
            tp = signal.price + sl_distance * self._rr_ratio
        else:
            sl = signal.price + sl_distance
            tp = signal.price - sl_distance * self._rr_ratio

        # Fallback: ensure SL is at least min percent away from entry
        fallback_pct = float((self.config.get_value("risk") or {}).get("stop_loss_pct", 0.015))
        if signal.is_long:
            min_sl = signal.price * (1 - fallback_pct)
            sl = min(sl, min_sl)  # For longs, move SL further DOWN (lower = wider)
        else:
            max_sl = signal.price * (1 + fallback_pct)
            sl = max(sl, max_sl)  # For shorts, move SL further UP (higher = wider)

        return sl, tp

    async def open_position(self, signal: TradingSignal, size: float) -> Position:
        async with self._lock:
            sl, tp = self.compute_atr_stops(signal)

            pos = Position(
                exchange=signal.exchange,
                symbol=signal.symbol,
                direction=signal.direction,
                size=size / signal.price if signal.price > 0 else 0.0,
                entry_price=signal.price,
                current_price=signal.price,
                stop_loss=sl,
                take_profit=tp,
                open_time=int(time.time()),
                highest_since_entry=signal.price,
                lowest_since_entry=signal.price,
            )
            key = f"{signal.exchange}:{signal.symbol}"
            self._positions[key] = pos
            self._last_trade_time[signal.symbol] = time.time()
            # Persist to SQLite
            if self._sqlite_store:
                try:
                    db_id = self._sqlite_store.insert_position(
                        signal.exchange, signal.symbol, signal.direction,
                        signal.price, pos.size, is_paper=self.config.paper_mode,
                    )
                    pos._db_id = db_id
                except Exception as e:
                    logger.debug("SQLite insert_position failed: {}", e)
            logger.info(
                "Position opened: {}/{} {} size={:.4f} @ {:.2f} SL={:.2f} TP={:.2f} (ATR-based)",
                signal.exchange, signal.symbol, signal.direction,
                pos.size, pos.entry_price, pos.stop_loss, pos.take_profit,
            )
            return pos

    async def close_position(self, exchange: str, symbol: str, exit_price: float) -> Position | None:
        async with self._lock:
            key = f"{exchange}:{symbol}"
            pos = self._positions.pop(key, None)
            if pos is None:
                return None
            pos.update_price(exit_price)
            pnl_dollar = pos.pnl
            self._equity += pnl_dollar
            # Guard: equity must never go below zero — trip kill switch
            if self._equity <= 0:
                logger.critical(
                    "EQUITY ZERO OR NEGATIVE ({:.2f}) — activating kill switch", self._equity
                )
                self._equity = 0.0
                self._killed = True
            self.record_return(pos.pnl_pct)
            self._record_loss_for_period(pos.pnl_pct)
            self.record_trade_result(pos.pnl_pct > 0)
            was_tripped = self._circuit_breaker.tripped
            self._circuit_breaker.record_pnl(pos.pnl_pct, self._equity)
            self._last_trade_time[symbol] = time.time()
            logger.info(
                "Position closed: {}/{} pnl={:.2f} ({:.2%}) equity={:.2f} held={:.0f}s",
                exchange, symbol, pnl_dollar, pos.pnl_pct, self._equity, pos.hold_seconds,
            )
            if not was_tripped and self._circuit_breaker.tripped:
                logger.critical(
                    "🛑 CIRCUIT BREAKER TRIPPED: {} — auto-closing remaining {} positions",
                    self._circuit_breaker.trip_reason, len(self._positions),
                )
                self._killed = True
                try:
                    asyncio.create_task(self._auto_close_on_trip())
                except RuntimeError:
                    pass
                try:
                    await self.event_bus.publish("CIRCUIT_BREAKER_TRIPPED", {
                        "reason": self._circuit_breaker.trip_reason,
                        "equity": self._equity,
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass
            # Persist to SQLite
            db_id = getattr(pos, '_db_id', None)
            if self._sqlite_store and db_id:
                try:
                    self._sqlite_store.close_position(db_id, exit_price, pnl_dollar, pos.pnl_pct)
                except Exception as e:
                    logger.debug("SQLite close_position failed: {}", e)
            # Keep in-memory closed trade log
            self._closed_trades.append({
                "exchange": exchange, "symbol": symbol, "direction": pos.direction,
                "entry_price": pos.entry_price, "exit_price": exit_price,
                "size": pos.size, "pnl": round(pnl_dollar, 4),
                "pnl_pct": round(pos.pnl_pct * 100, 4),
                "hold_seconds": pos.hold_seconds,
                "open_time": pos.open_time, "close_time": int(time.time()),
            })
            if len(self._closed_trades) > 500:
                self._closed_trades = self._closed_trades[-500:]
            try:
                self.event_bus.publish_nowait("PNL_UPDATE", {
                    "exchange": exchange, "symbol": symbol, "direction": pos.direction,
                    "entry_price": pos.entry_price, "exit_price": exit_price,
                    "size": pos.size, "pnl_usd": round(pnl_dollar, 4),
                    "pnl_pct": round(pos.pnl_pct, 6),
                    "hold_seconds": pos.hold_seconds,
                    "equity_after": round(self._equity, 2),
                    "ts": time.time(),
                })
            except Exception as exc:
                logger.debug("PNL_UPDATE publish failed: {}", exc)
            return pos

    # ── Trailing stop / breakeven / time exit ─────────────────────────────
    def manage_position(self, pos: Position) -> str | None:
        """Returns exit reason string if position should be closed, else None."""
        atr = pos.entry_price * 0.01  # fallback; real ATR should be on position

        # Time-based exit
        if self._max_hold_minutes > 0:
            if pos.hold_seconds >= self._max_hold_minutes * 60:
                return "time_exit"

        profit_in_atr = 0.0
        if pos.is_long:
            profit_in_atr = (pos.current_price - pos.entry_price) / atr if atr > 0 else 0
        else:
            profit_in_atr = (pos.entry_price - pos.current_price) / atr if atr > 0 else 0

        # Breakeven move: after +1×ATR profit, move SL to entry
        if not pos.breakeven_moved and profit_in_atr >= self._breakeven_trigger_atr:
            pos.stop_loss = pos.entry_price
            pos.breakeven_moved = True
            logger.info("Breakeven moved for {}/{}", pos.exchange, pos.symbol)

        # Trailing stop activation: after trailing_activation_atr × ATR profit
        if profit_in_atr >= self._trailing_activation_atr:
            pos.trailing_active = True
            if pos.is_long:
                new_trail = pos.highest_since_entry - atr * self._trailing_distance_atr
                if new_trail > pos.stop_loss:
                    pos.stop_loss = new_trail
                    pos.trailing_stop = new_trail
            else:
                new_trail = pos.lowest_since_entry + atr * self._trailing_distance_atr
                if new_trail < pos.stop_loss:
                    pos.stop_loss = new_trail
                    pos.trailing_stop = new_trail

        return None

    # ── Auto-trip handler ────────────────────────────────────────────────
    async def _auto_close_on_trip(self) -> None:
        """Background close-all triggered by circuit breaker trip.

        Runs OUTSIDE the lock held by close_position() to avoid deadlock —
        activate_kill_switch acquires the same lock.
        """
        try:
            closed = await self.activate_kill_switch()
            try:
                await self.event_bus.publish("KILL_SWITCH_ACTIVATED", {
                    "trigger": "circuit_breaker",
                    "positions_closed": len(closed),
                    "reason": self._circuit_breaker.trip_reason,
                })
            except Exception:
                pass
        except Exception as exc:
            logger.error("auto_close_on_trip failed: {}", exc)

    # ── Daily reset (midnight UTC) ────────────────────────────────────────
    def reset_daily_losses(self) -> None:
        """Reset daily loss counter and clear daily-loss circuit-breaker trip.

        Drawdown-based trips are NOT reset — only daily-loss trips.
        Designed to be called at 00:00 UTC by a scheduler.
        """
        self._circuit_breaker.reset_daily()
        if self._circuit_breaker.tripped and self._circuit_breaker.trip_reason.startswith("daily_loss"):
            self._circuit_breaker.clear_if_reason(self._circuit_breaker.trip_reason)
            self._killed = False
            logger.warning("Daily-loss reset at UTC midnight — circuit breaker cleared, kill switch released")
        else:
            logger.info("Daily-loss counter reset at UTC midnight")

    # ── Kill switch ───────────────────────────────────────────────────────
    async def activate_kill_switch(self) -> list[Position]:
        """Emergency: close all positions and block new signals."""
        async with self._lock:
            self._killed = True
            closed: list[Position] = []
            for key in list(self._positions.keys()):
                pos = self._positions[key]
                # close_position acquires lock too, so call inner logic directly
                pos.update_price(pos.current_price)
                pnl_dollar = pos.pnl
                self._equity += pnl_dollar
                if self._equity < 0:
                    self._equity = 0.0
                self.record_return(pos.pnl_pct)
                self._record_loss_for_period(pos.pnl_pct)
                self.record_trade_result(pos.pnl_pct > 0)
                self._circuit_breaker.record_pnl(pos.pnl_pct, self._equity)
                self._last_trade_time[pos.symbol] = time.time()
                self._positions.pop(key, None)
                closed.append(pos)
            self._equity_unreconciled = True
            logger.critical(
                "KILL SWITCH activated — closed {} positions (equity={:.2f} APPROXIMATE)",
                len(closed), self._equity,
            )
            return closed

    def deactivate_kill_switch(self, operator: str = "system") -> None:
        """Release the kill switch and clear the file-based trigger.

        Without removing the kill-switch file and busting the 5s cache, the
        next call to approve_signal would re-detect the file and re-set
        self._killed = True — meaning the operator's deactivation would be
        silently reverted. This method also resets the cached file-state
        check so the next approve_signal sees fresh ground truth.
        """
        self._killed = False
        self._equity_unreconciled = False
        # Remove the kill-switch file if present, so cache re-check doesn't re-trip.
        try:
            if self._kill_switch_file.exists():
                self._kill_switch_file.unlink()
        except Exception as exc:
            logger.warning("Could not remove kill-switch file {}: {}", self._kill_switch_file, exc)
        # Bust the cache so the next approve_signal re-stats the file path immediately.
        self._kill_switch_cache = False
        self._kill_switch_last_check = 0.0
        logger.warning("Kill switch DEACTIVATED by {}", operator)
        try:
            self.event_bus.publish_nowait("ESTOP_RELEASED", {
                "operator": operator,
                "ts": time.time(),
            })
        except Exception:
            pass

    @property
    def killed(self) -> bool:
        return self._killed

    def record_return(self, pnl_pct: float) -> None:
        self._return_history.append(float(pnl_pct))
        if len(self._return_history) > self._returns_window:
            self._return_history = self._return_history[-self._returns_window:]

    def calculate_historical_var(self, confidence: float = 0.95) -> float:
        if not self._return_history:
            return 0.0
        arr = np.array(self._return_history, dtype=float)
        q = np.percentile(arr, (1.0 - confidence) * 100.0)
        return float(max(0.0, -q))

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        if not self._return_history:
            return 0.0
        arr = np.array(self._return_history, dtype=float)
        q = np.percentile(arr, (1.0 - confidence) * 100.0)
        tail = arr[arr <= q]
        if tail.size == 0:
            return 0.0
        return float(max(0.0, -tail.mean()))

    def portfolio_notional(self) -> float:
        return float(sum(abs(p.size * p.current_price) for p in self._positions.values()))

    def current_drawdown_pct(self) -> float:
        peak = self._circuit_breaker._peak_equity
        if peak <= 0:
            return 0.0
        return float((peak - self._equity) / peak)

    def run_stress_test(
        self,
        shocks: list[float] | None = None,
        correlation_breakdown_factor: float = 1.5,
    ) -> dict[str, Any]:
        if shocks is None:
            shocks = [-0.05, -0.10, -0.20]

        notional = self.portfolio_notional()
        scenarios: dict[str, Any] = {}
        worst_loss = 0.0

        for shock in shocks:
            loss = abs(notional * shock)
            worst_loss = max(worst_loss, loss)
            scenarios[f"shock_{int(abs(shock) * 100)}pct"] = {
                "shock_pct": float(shock),
                "estimated_loss_usd": float(loss),
                "loss_pct_equity": float(loss / self._equity) if self._equity > 0 else 0.0,
            }

        corr_loss = worst_loss * correlation_breakdown_factor
        scenarios["correlation_breakdown"] = {
            "factor": float(correlation_breakdown_factor),
            "estimated_loss_usd": float(corr_loss),
            "loss_pct_equity": float(corr_loss / self._equity) if self._equity > 0 else 0.0,
        }

        return {
            "portfolio_notional_usd": notional,
            "var_95": self.calculate_historical_var(0.95),
            "var_99": self.calculate_historical_var(0.99),
            "cvar_95": self.calculate_cvar(0.95),
            "cvar_99": self.calculate_cvar(0.99),
            "scenarios": scenarios,
            "max_portfolio_var_pct": self._max_portfolio_var_pct,
        }

    def get_risk_snapshot(self) -> dict[str, Any]:
        return {
            "equity": float(self._equity),
            "open_positions": len(self._positions),
            "portfolio_notional_usd": self.portfolio_notional(),
            "var_95": self.calculate_historical_var(0.95),
            "var_99": self.calculate_historical_var(0.99),
            "cvar_95": self.calculate_cvar(0.95),
            "cvar_99": self.calculate_cvar(0.99),
            "return_history_size": len(self._return_history),
            "var_min_history": self._var_min_history,
            "max_portfolio_var_pct": self._max_portfolio_var_pct,
            "circuit_breaker_tripped": self._circuit_breaker.tripped,
            "circuit_breaker_reason": self._circuit_breaker.trip_reason,
            "kill_switch_active": self._killed,
            "current_drawdown_pct": self.current_drawdown_pct(),
            "drawdown_phase": self._current_dd_phase,
            "consecutive_profitable": self._consecutive_profitable,
            "weekly_loss_pct": self._weekly_loss,
            "monthly_loss_pct": self._monthly_loss,
            "is_weekend": self.is_weekend(),
            "active_sessions": self.get_active_sessions(),
        }

    async def update_prices(self, exchange: str, symbol: str, price: float) -> None:
        async with self._lock:
            key = f"{exchange}:{symbol}"
            pos = self._positions.get(key)
            if pos is None:
                return
            pos.update_price(price)

            # ARMS-V2.1: Liquidation safety layer
            liq_action = self.check_liquidation_distance(pos)
            if liq_action == "emergency_close":
                logger.critical("LIQUIDATION EMERGENCY for {}/{} — closing position", exchange, symbol)
                self.event_bus.publish_nowait("STOP_LOSS", {
                    "exchange": exchange, "symbol": symbol, "price": price,
                    "reason": "liquidation_emergency",
                })
                return
            elif liq_action == "partial_close":
                logger.warning("Liquidation danger for {}/{} — partial close triggered", exchange, symbol)
                self.event_bus.publish_nowait("PARTIAL_CLOSE", {
                    "exchange": exchange, "symbol": symbol, "price": price,
                    "reason": "liquidation_partial", "close_pct": 0.5,
                })
            elif liq_action == "warning":
                logger.warning("Liquidation warning for {}/{} — distance shrinking", exchange, symbol)

            # ARMS-V2.1: Tiered take-profit check
            tp_tiers = self.compute_tiered_tp(
                pos.entry_price,
                abs(pos.entry_price - pos.stop_loss),
                pos.is_long,
            )
            for i, tp_tier in enumerate(tp_tiers):
                if tp_tier.get("trail"):
                    continue  # Trail handled by trailing stop logic
                tp_price = tp_tier["price"]
                if pos.is_long and price >= tp_price:
                    close_pct = tp_tier["close_pct"]
                    logger.info("TP{} hit for {}/{} at {:.2f} — close {:.0%}",
                                i + 1, exchange, symbol, price, close_pct)
                    self.event_bus.publish_nowait("PARTIAL_CLOSE", {
                        "exchange": exchange, "symbol": symbol, "price": price,
                        "reason": f"tp{i+1}", "close_pct": close_pct,
                    })
                    break
                elif not pos.is_long and price <= tp_price:
                    close_pct = tp_tier["close_pct"]
                    logger.info("TP{} hit for {}/{} at {:.2f} — close {:.0%}",
                                i + 1, exchange, symbol, price, close_pct)
                    self.event_bus.publish_nowait("PARTIAL_CLOSE", {
                        "exchange": exchange, "symbol": symbol, "price": price,
                        "reason": f"tp{i+1}", "close_pct": close_pct,
                    })
                    break

            # Run position management (trailing stop, breakeven, time exit)
            exit_reason = self.manage_position(pos)
            if exit_reason:
                logger.info("{} exit triggered for {}/{}", exit_reason, exchange, symbol)
                self.event_bus.publish_nowait("STOP_LOSS", {
                    "exchange": exchange, "symbol": symbol, "price": price, "reason": exit_reason,
                })
                return

            if (pos.is_long and price <= pos.stop_loss) or (not pos.is_long and price >= pos.stop_loss):
                reason = "trailing_stop" if pos.trailing_active else "stop_loss"
                logger.warning("{} triggered for {}/{} at {:.2f}", reason, exchange, symbol, price)
                self.event_bus.publish_nowait("STOP_LOSS", {
                    "exchange": exchange, "symbol": symbol, "price": price, "reason": reason,
                })
            elif (pos.is_long and price >= pos.take_profit) or (not pos.is_long and price <= pos.take_profit):
                logger.info("Take profit triggered for {}/{} at {:.2f}", exchange, symbol, price)
                self.event_bus.publish_nowait("TAKE_PROFIT", {
                    "exchange": exchange, "symbol": symbol, "price": price, "reason": "take_profit",
                })

    # ── §13 Guardrails: Flash crash detection ──────────────────────────
    def check_flash_crash(self, symbol: str, price: float) -> bool:
        """Detect flash crash: >=5% drop within 5-minute window. Returns True if tripped."""
        now = time.time()
        self._price_snapshots.append((now, price))
        # Prune snapshots older than window
        cutoff = now - self._flash_crash_window_seconds
        self._price_snapshots = [(t, p) for t, p in self._price_snapshots if t >= cutoff]
        if len(self._price_snapshots) < 2:
            return self._flash_crash_tripped
        oldest_price = self._price_snapshots[0][1]
        if oldest_price > 0:
            drop_pct = (oldest_price - price) / oldest_price
            if drop_pct >= self._flash_crash_threshold_pct:
                if not self._flash_crash_tripped:
                    self._flash_crash_tripped = True
                    self._circuit_breaker.trip("flash_crash", pause_seconds=600)
                    logger.warning(
                        "Flash crash detected for {} — {:.2f}% drop in {:.0f}s, halting 10min",
                        symbol, drop_pct * 100, self._flash_crash_window_seconds,
                    )
            else:
                self._flash_crash_tripped = False
        return self._flash_crash_tripped

    # ── §13 Guardrails: Consecutive loss tracking ────────────────────────
    def record_trade_pnl(self, pnl: float) -> None:
        """Track consecutive losses and trip breaker after N consecutive."""
        # Update Kelly stats
        self._trade_results.append(pnl)
        if len(self._trade_results) > 100:
            self._trade_results = self._trade_results[-100:]
        wins = [r for r in self._trade_results if r > 0]
        losses = [r for r in self._trade_results if r <= 0]
        if self._trade_results:
            self._kelly_win_rate = len(wins) / len(self._trade_results)
        if wins:
            self._kelly_avg_win = sum(wins) / len(wins)
        if losses:
            self._kelly_avg_loss = abs(sum(losses) / len(losses))

        # Consecutive loss tracking
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._max_consecutive_losses:
                self._circuit_breaker.trip("consecutive_losses", pause_seconds=3600)
                logger.warning(
                    "Consecutive loss breaker tripped — {} losses in a row, halting 1h",
                    self._consecutive_losses,
                )
        else:
            self._consecutive_losses = 0

    # ── §9 Risk: Kelly Criterion position sizing ─────────────────────────
    def compute_kelly_size(
        self,
        base_risk_pct: float,
        regime_confidence: float = 1.0,
        tier_multiplier: float = 1.0,
    ) -> float:
        """Apply Kelly Criterion, scaled by regime confidence and pair tier.

        regime_confidence : 0.0–1.0 — trending strong = 1.0, choppy = 0.3
        tier_multiplier   : TIER_1=1.0, TIER_2=0.6, TIER_3=0.3 (from PairRegistry)
        """
        if not self._kelly_enabled or len(self._trade_results) < 20:
            # No Kelly history — still apply regime/tier multipliers
            return base_risk_pct * max(0.25, regime_confidence) * tier_multiplier
        w = self._kelly_win_rate
        if self._kelly_avg_loss <= 0:
            return base_risk_pct * max(0.25, regime_confidence) * tier_multiplier
        r = self._kelly_avg_win / self._kelly_avg_loss
        if r <= 0:
            return base_risk_pct * max(0.25, regime_confidence) * tier_multiplier
        kelly_f = w - (1.0 - w) / r
        kelly_f = max(0.0, kelly_f) * 0.5  # Half-Kelly
        # Scale base risk by kelly_f (25% → 2x band), then by regime + tier
        adjusted = base_risk_pct * min(2.0, max(0.25, kelly_f * 50.0))
        adjusted *= max(0.25, min(1.0, regime_confidence))
        adjusted *= max(0.0, min(1.0, tier_multiplier))
        logger.debug("Kelly: W={:.2f} R={:.2f} f*={:.4f} regime={:.2f} tier={:.2f} adj={:.4f}",
                      w, r, kelly_f * 2, regime_confidence, tier_multiplier, adjusted)
        return adjusted

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def equity(self) -> float:
        return self._equity

    async def _handle_tick(self, payload: Any) -> None:
        tick = payload
        await self.update_prices(tick.exchange, tick.symbol, tick.price)

    async def run(self) -> None:
        self._running = True
        self.event_bus.subscribe("TICK", self._handle_tick)
        logger.info("RiskManager started — equity={:.2f}, sizing={}", self._equity, self._sizing_method)
        snapshot_interval = 60.0
        last_snapshot = 0.0
        while self._running:
            await asyncio.sleep(5)
            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                last_snapshot = now
                try:
                    unrealized = sum(p.pnl for p in self._positions.values())
                    self.event_bus.publish_nowait("EQUITY_SNAPSHOT", {
                        "equity": round(self._equity, 2),
                        "open_positions": len(self._positions),
                        "unrealized_pnl": round(unrealized, 4),
                        "circuit_breaker_tripped": bool(self._circuit_breaker.tripped),
                        "killed": bool(self._killed),
                        "ts": now,
                    })
                except Exception as exc:
                    logger.debug("EQUITY_SNAPSHOT publish failed: {}", exc)

    async def stop(self) -> None:
        self._running = False
        self.event_bus.unsubscribe("TICK", self._handle_tick)
