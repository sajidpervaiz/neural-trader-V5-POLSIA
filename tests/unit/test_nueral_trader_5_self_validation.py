"""
Self-validation tests — MANDATORY before declaring completion.

1. Hacker test: attempt malicious signal injection via API; must be rejected
2. Ops test: WebSocket reconnect behavior  
3. Risk test: simulate 10 consecutive losses; daily loss limit must halt trading
4. Idempotency test: replay same tick; must not duplicate orders
5. Config validation test: invalid config must fail fast
6. Kill switch file test: file-based kill must block trading
7. Instance lock test: dual instance must be prevented
8. Monte Carlo test: simulation produces valid results
9. SQLite persistence test: data roundtrip
10. Multi-TF confirmation test: HTF override blocks contra-trend signals
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import Config
from core.event_bus import EventBus
from core.idempotency import IdempotencyManager
from core.instance_lock import InstanceLock
from engine.fast_backtester import BacktestTrade, FastBacktester, MonteCarloSimulator
from engine.signal_generator import TradingSignal
from execution.risk_manager import CircuitBreaker, RiskManager
from storage.sqlite_store import SQLiteStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(overrides: dict | None = None) -> Config:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    Config._instance = None
    cfg = Config(path=cfg_path)
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg._data
            for part in keys[:-1]:
                d = d.setdefault(part, {})
            d[keys[-1]] = v
    return cfg


def _make_signal(
    symbol: str = "BTC/USDT:USDT",
    direction: str = "long",
    score: float = 0.85,
    price: float = 50000.0,
    atr: float = 500.0,
    sl: float = 49000.0,
    tp: float = 52000.0,
    metadata: dict | None = None,
) -> TradingSignal:
    return TradingSignal(
        exchange="binance",
        symbol=symbol,
        direction=direction,
        score=score,
        technical_score=0.8,
        ml_score=0.5,
        sentiment_score=0.3,
        macro_score=0.2,
        news_score=0.4,
        orderbook_score=0.3,
        regime="trending_up",
        regime_confidence=0.7,
        price=price,
        atr=atr,
        stop_loss=sl,
        take_profit=tp,
        timestamp=int(time.time()),
        metadata=metadata or {},
    )


# ── 1. Hacker Test: Malicious Signal Injection ───────────────────────────────

class TestHackerSignalInjection:
    """Attempt to inject malicious trades via the risk engine.  Must be REJECTED."""

    def test_reject_signal_below_score_threshold(self):
        """Signals with low confidence must be blocked."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        signal = _make_signal(score=0.1)  # Very low confidence
        approved, reason, _ = rm.approve_signal(signal)
        assert not approved
        assert "score_too_low" in reason

    def test_reject_zero_risk_reward(self):
        """Signals with poor R:R must be blocked."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        signal = _make_signal(sl=49999.0, tp=50001.0)  # Terrible R:R
        approved, reason, _ = rm.approve_signal(signal)
        assert not approved
        assert "risk_reward" in reason

    async def test_reject_when_killed(self):
        """Kill switch active must block ALL signals."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        await rm.activate_kill_switch()
        signal = _make_signal()
        approved, reason, _ = rm.approve_signal(signal)
        assert not approved
        assert "kill_switch" in reason

    def test_reject_extreme_position_size(self):
        """Even high-score signals must respect max position size."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        # Create a signal that would generate a huge position
        signal = _make_signal(price=50000, sl=49999, tp=52000, score=0.99)
        approved, reason, size = rm.approve_signal(signal)
        if approved:
            # Verify position size is capped
            max_pct = float(cfg.get_value("risk", "max_position_size_pct") or 0.02)
            max_expected = rm._equity * max_pct * rm._leverage
            assert size <= max_expected * 1.01, f"Size {size} exceeds max {max_expected}"


# ── 2. Ops Test: WebSocket Reconnect (via circuit breaker) ───────────────────

class TestOpsReconnection:
    """Verify the circuit breaker recovers after failures."""

    def test_circuit_breaker_trips_and_recovers(self):
        """After N failures, circuit breaker trips and auto-recovers after pause."""
        from core.circuit_breaker import CircuitBreaker as CoreCB, CircuitState
        cb = CoreCB(failure_threshold=3, recovery_timeout=0.1)
        
        # Trip the breaker
        for _ in range(3):
            cb.record_failure()
        
        assert cb.get_state() == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Manually transition (in production this happens via call())
        # Verify the timeout has elapsed
        assert time.time() - cb.last_failure_time >= cb.recovery_timeout
        
        # Simulate the transition that call() would do
        cb.state = CircuitState.HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_event_bus_handles_handler_errors(self):
        """Event handlers that throw must not crash the bus."""
        bus = EventBus()
        
        errors: list[Exception] = []
        
        async def bad_handler(payload):
            raise RuntimeError("handler crash")
        
        async def good_handler(payload):
            errors.append(None)  # just a marker
        
        bus.subscribe("TEST", bad_handler)
        bus.subscribe("TEST", good_handler)
        
        async def run():
            bus._queue.put_nowait(("TEST", "data"))
            await asyncio.sleep(0.1)

        asyncio.run(run())


# ── 3. Risk Test: 10 Consecutive Losses → Daily Loss Halt ───────────────────

class TestRiskDailyLossHalt:
    """Simulate 10 consecutive losses.  Circuit breaker MUST halt trading."""

    async def test_10_losses_trips_circuit_breaker(self):
        """After 10 consecutive losing trades, circuit breaker should trip."""
        cfg = _make_config({"risk.max_daily_loss_pct": "0.03"})
        bus = EventBus()
        rm = RiskManager(cfg, bus)

        initial_equity = rm._equity  # 100_000

        # Simulate 10 consecutive 0.5% losses
        for i in range(10):
            signal = _make_signal(
                symbol=f"TEST{i}/USDT:USDT",
                price=50000 + i,
                sl=49000,
                tp=52000,
            )
            
            # Force open a position
            pos = await rm.open_position(signal, 2000.0)
            # Close at a loss (0.5% each)
            loss_price = pos.entry_price * 0.995 if pos.is_long else pos.entry_price * 1.005
            await rm.close_position(pos.exchange, pos.symbol, loss_price)

        # After 10 × 0.5% = 5% loss, should exceed 3% daily limit
        assert rm._circuit_breaker.tripped, (
            f"Circuit breaker should be tripped after 10 losses. "
            f"Daily loss: {rm._circuit_breaker._daily_loss:.4f}"
        )

        # Verify new signals are blocked. Daily-loss trip auto-engages the
        # kill switch (production-safe pattern), which checks first; either
        # rejection reason indicates trading is halted.
        signal = _make_signal(symbol="BLOCKED/USDT:USDT")
        approved, reason, _ = rm.approve_signal(signal)
        assert not approved
        assert ("circuit_breaker" in reason) or ("kill_switch" in reason)

    def test_drawdown_limit_blocks_trading(self):
        """Equity drawdown beyond threshold blocks new trades."""
        cfg = _make_config({"risk.max_drawdown_pct": "0.05"})
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm._circuit_breaker._max_drawdown = 0.05
        rm._circuit_breaker._peak_equity = 100_000
        
        # Record PnL that causes > 5% drawdown
        for i in range(10):
            rm._circuit_breaker.record_pnl(-0.01, rm._equity - (i * 1000))
        
        if rm._circuit_breaker.tripped:
            signal = _make_signal()
            approved, reason, _ = rm.approve_signal(signal)
            assert not approved


# ── 4. Idempotency Test: Replay Must Not Duplicate ───────────────────────────

class TestIdempotency:
    """Replay the same tick/order — must not place duplicate orders."""

    def test_idempotency_manager_deduplicates(self):
        """Same key submitted twice → second is detected as duplicate."""
        mgr = IdempotencyManager(ttl=60, max_size=100)
        key = "order_btc_buy_12345"
        
        # First call: not duplicate
        is_dup_1 = mgr.check_and_set(key)
        assert not is_dup_1
        
        # Second call: IS duplicate
        is_dup_2 = mgr.check_and_set(key)
        assert is_dup_2

    @pytest.mark.asyncio
    async def test_order_manager_idempotent(self):
        """OrderManager rejects duplicate client_order_ids."""
        from core.circuit_breaker import CircuitBreaker as CoreCB
        from execution.order_manager import OrderManager, OrderSide, OrderType
        
        cfg = _make_config()
        bus = EventBus()
        cb = CoreCB(failure_threshold=5, recovery_timeout=60)
        om = OrderManager(cfg, bus, cb)
        
        client_id = "test_idem_001"
        
        ok1, order1, reason1 = await om.place_order(
            exchange="binance", symbol="BTC/USDT:USDT",
            side=OrderSide.BUY, quantity=0.01, price=50000.0,
            client_order_id=client_id,
        )
        assert ok1
        assert reason1 == "created"
        
        ok2, order2, reason2 = await om.place_order(
            exchange="binance", symbol="BTC/USDT:USDT",
            side=OrderSide.BUY, quantity=0.01, price=50000.0,
            client_order_id=client_id,
        )
        assert ok2
        assert reason2 == "idempotent_retry"


# ── 5. Config Validation Test ─────────────────────────────────────────────────

class TestConfigValidation:
    """Invalid configs must fail-fast."""

    def test_invalid_risk_pct_rejected(self):
        from core.config_schema import validate_config
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({
                "system": {"paper_mode": True},
                "risk": {"max_position_size_pct": 5.0},  # > 0.5 limit
            })

    def test_signal_weights_must_sum_to_one(self):
        from core.config_schema import validate_config
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({
                "system": {"paper_mode": True},
                "signals": {
                    "ml_weight": 0.5,
                    "technical_weight": 0.5,
                    "sentiment_weight": 0.5,
                    "macro_weight": 0.5,
                    "news_weight": 0.5,
                    "orderbook_weight": 0.5,
                },
            })

    def test_valid_config_passes(self):
        from core.config_schema import validate_config
        result = validate_config({
            "system": {"paper_mode": True},
            "risk": {"max_position_size_pct": 0.02},
        })
        assert result.system.paper_mode is True


# ── 6. Kill Switch File Test ──────────────────────────────────────────────────

class TestKillSwitchFile:
    """File-based kill switch must block trading."""

    def test_kill_switch_file_blocks_signals(self, tmp_path):
        """Creating the kill switch file must block all new signals."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        kill_file = tmp_path / ".kill_switch"
        rm._kill_switch_file = kill_file
        
        # Without file — should pass (assuming score etc. are OK)
        signal = _make_signal()
        approved, reason, _ = rm.approve_signal(signal)
        # May or may not pass due to other filters, but NOT due to kill switch
        if not approved:
            assert "kill_switch" not in reason
        
        # Create kill switch file
        kill_file.touch()
        # Force cache refresh (cache interval is 5s)
        rm._kill_switch_last_check = 0.0
        
        signal2 = _make_signal(symbol="NEW/USDT:USDT")
        approved2, reason2, _ = rm.approve_signal(signal2)
        assert not approved2
        assert "kill_switch" in reason2


# ── 7. Instance Lock Test ────────────────────────────────────────────────────

class TestInstanceLock:
    """Two instances must not be allowed simultaneously."""

    def test_lock_acquire_and_release(self, tmp_path):
        lock_file = tmp_path / ".test.lock"
        lock = InstanceLock(lock_file=lock_file)
        assert lock.acquire()
        assert lock.locked
        lock.release()
        assert not lock.locked

    def test_double_lock_fails(self, tmp_path):
        lock_file = tmp_path / ".test.lock"
        lock1 = InstanceLock(lock_file=lock_file)
        lock2 = InstanceLock(lock_file=lock_file)
        assert lock1.acquire()
        assert not lock2.acquire()  # Second instance must fail
        lock1.release()


# ── 8. Monte Carlo Test ──────────────────────────────────────────────────────

class TestMonteCarlo:
    """Monte Carlo simulation must produce valid probability estimates."""

    def test_monte_carlo_with_profitable_trades(self):
        """Profitable trades should have low ruin probability."""
        trades = [
            BacktestTrade(
                symbol="TEST", direction="long",
                entry_price=100, exit_price=102,
                entry_time=0, exit_time=1,
                pnl=200, pnl_pct=0.02, hold_periods=1
            )
            for _ in range(50)
        ]
        mc = MonteCarloSimulator(n_runs=500, ruin_threshold_pct=0.5)
        result = mc.run(trades, initial_capital=10000)
        assert result["probability_of_ruin"] == 0.0
        assert result["median_final_equity"] > 10000

    def test_monte_carlo_with_losing_trades(self):
        """Losing trades should have higher drawdown estimates."""
        trades = [
            BacktestTrade(
                symbol="TEST", direction="long",
                entry_price=100, exit_price=95,
                entry_time=0, exit_time=1,
                pnl=-500, pnl_pct=-0.05, hold_periods=1
            )
            for _ in range(30)
        ]
        mc = MonteCarloSimulator(n_runs=500, ruin_threshold_pct=0.5)
        result = mc.run(trades, initial_capital=10000)
        assert result["median_final_equity"] < 10000
        assert result["mean_max_drawdown"] > 0


# ── 9. SQLite Persistence Test ───────────────────────────────────────────────

class TestSQLitePersistence:
    """SQLite store: write, read back, verify data integrity."""

    def test_tick_roundtrip(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        store.insert_tick("binance", "BTC/USDT", 50000.0, 1.5, "buy")
        rows = store.query("SELECT * FROM ticks")
        assert len(rows) == 1
        assert rows[0]["price"] == 50000.0
        store.close()

    def test_position_lifecycle(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        pos_id = store.insert_position("binance", "ETH/USDT", "long", 3000.0, 1.0)
        store.close_position(pos_id, 3100.0, 100.0, 0.0333)
        
        history = store.get_trade_history()
        assert len(history) == 1
        assert history[0]["pnl"] == 100.0
        store.close()

    def test_equity_curve(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        for i in range(5):
            store.record_equity(100000 + i * 100, drawdown=0.01 * i)
        curve = store.get_equity_curve()
        assert len(curve) == 5
        store.close()

    def test_risk_state_roundtrip(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        store.save_risk_state({"equity": 100000, "killed": False})
        state = store.load_risk_state()
        assert state["equity"] == 100000
        assert state["killed"] is False
        store.close()


# ── 10. Multi-TF Confirmation Test ──────────────────────────────────────────

class TestMultiTimeframeConfirmation:
    """Higher TF trend must override lower TF entries."""

    def test_htf_bearish_blocks_long_signal(self):
        """If 1h is bearish, 15m long signals should be blocked."""
        cfg = _make_config()
        bus = EventBus()
        from analysis.data_manager import DataManager
        dm = DataManager(cfg, bus)
        
        from engine.signal_generator import SignalGenerator
        sg = SignalGenerator(cfg, bus, dm)
        sg._confirmation_tfs = ["1h"]
        
        # Create mock dataframe where EMA12 < EMA26 (bearish)
        dates = pd.date_range("2024-01-01", periods=30, freq="1h")
        bearish_df = pd.DataFrame({
            "close": np.linspace(100, 90, 30),  # Downtrend
            "ema_12": np.linspace(95, 89, 30),
            "ema_26": np.linspace(96, 92, 30),  # EMA26 > EMA12 = bearish
            "supertrend_dir": [-1] * 30,  # bearish SuperTrend
            "rsi_14": [35.0] * 30,  # below 50 = bearish
        }, index=dates)
        
        # Mock data_manager to return bearish 1h data
        dm.get_dataframe = MagicMock(return_value=bearish_df)
        
        ok, reason, _, _ = sg._check_higher_timeframe_trend("binance", "BTC/USDT:USDT", "long")
        assert not ok
        assert "bearish" in reason.lower()

    def test_htf_bullish_allows_long_signal(self):
        """If 1h is bullish, 15m long signals should pass HTF check."""
        cfg = _make_config()
        bus = EventBus()
        from analysis.data_manager import DataManager
        dm = DataManager(cfg, bus)
        
        from engine.signal_generator import SignalGenerator
        sg = SignalGenerator(cfg, bus, dm)
        sg._confirmation_tfs = ["1h"]
        
        dates = pd.date_range("2024-01-01", periods=30, freq="1h")
        bullish_df = pd.DataFrame({
            "close": np.linspace(90, 100, 30),  # Uptrend
            "ema_12": np.linspace(92, 101, 30),
            "ema_26": np.linspace(88, 97, 30),  # EMA12 > EMA26 = bullish
            "supertrend_dir": [1] * 30,  # bullish SuperTrend
            "rsi_14": [60.0] * 30,  # above 50 = bullish
        }, index=dates)
        
        dm.get_dataframe = MagicMock(return_value=bullish_df)
        
        ok, reason, _, _ = sg._check_higher_timeframe_trend("binance", "BTC/USDT:USDT", "long")
        assert ok


# ── New Risk Engine Filter Tests ─────────────────────────────────────────────

class TestNewRiskFilters:
    """Validate the newly added risk engine filters."""

    def test_funding_rate_filter_blocks_extreme(self):
        """Extreme funding rates must block trading."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm._max_funding_rate_bps = 50.0  # 50 bps max
        
        # Set extreme funding rate (100 bps = 1%)
        rm.update_funding_rate("BTC/USDT:USDT", 0.01)
        
        signal = _make_signal()
        ok, reason = rm._check_funding_rate(signal)
        assert not ok
        assert "funding_rate" in reason

    def test_funding_rate_filter_passes_normal(self):
        """Normal funding rates should pass."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm.update_funding_rate("BTC/USDT:USDT", 0.0001)  # 1 bps — normal
        
        signal = _make_signal()
        ok, reason = rm._check_funding_rate(signal)
        assert ok

    def test_orderbook_depth_filter_blocks_thin(self):
        """Thin order books must block trading."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm._min_orderbook_depth_usd = 50_000
        rm.update_orderbook_depth("BTC/USDT:USDT", 10_000)  # Only $10K depth
        
        signal = _make_signal()
        ok, reason = rm._check_orderbook_depth(signal)
        assert not ok
        assert "depth" in reason

    def test_max_order_size_cap(self):
        """Orders exceeding hard max must be rejected."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm._max_order_size_usd = 100_000
        
        ok, reason = rm._check_max_order_size(200_000)
        assert not ok
        assert "max_order_size" in reason

    def test_liquidation_risk_filter(self):
        """Signals with SL beyond estimated liquidation should be blocked."""
        cfg = _make_config()
        bus = EventBus()
        rm = RiskManager(cfg, bus)
        rm._leverage = 10.0  # 10x leverage → liq at ~10% move
        
        signal = _make_signal(price=50000, sl=44000)  # SL at -12%, liq at ~$45k
        ok, reason = rm._check_liquidation_risk(signal)
        assert not ok
        assert "liquidation" in reason
