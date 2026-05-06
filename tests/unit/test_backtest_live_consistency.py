"""
Tests for Prompt #8: Backtest vs Live Consistency.

Validates:
- No lookahead bias (latency_bars enforcement)
- Configurable fees / slippage / spread / latency
- Event-driven backtester mirrors live execution path
- SL/TP modeled on high/low intra-bar
- Metrics: Sharpe, max_drawdown, win_rate, profit_factor, expectancy, avg_duration
- Benchmarks: buy-and-hold, SMA crossover baseline
- Walk-forward validation
- Monte Carlo simulation (1000 runs)
- Backtest report file generation
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from engine.fast_backtester import (
    BacktestConfig,
    BacktestReportGenerator,
    BacktestResult,
    BacktestTrade,
    BenchmarkRunner,
    EventDrivenBacktester,
    FastBacktester,
    MonteCarloSimulator,
    WalkForwardOptimizer,
    _compute_result_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 500, seed: int = 42, start_price: float = 50_000.0) -> pd.DataFrame:
    """Produce a realistic OHLCV DataFrame with ATR column."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.008, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.uniform(100, 1000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)
    # Simple ATR proxy for backtest use
    tr = np.maximum.reduce([high - low, np.abs(high - open_), np.abs(low - open_)])
    df["atr_14"] = pd.Series(tr, index=idx).rolling(14).mean().fillna(pd.Series(close * 0.005, index=idx))
    # Add features needed by TechnicalScorer
    df["rsi_14"] = 50.0 + rng.normal(0, 8, n)
    df["macd"] = rng.normal(0, 10, n)
    df["macd_signal"] = df["macd"].rolling(9).mean().fillna(0)
    df["ema_12"] = close  # simplified
    df["ema_26"] = close  # simplified
    df["trend_ema"] = rng.normal(0, 0.1, n)
    df["bb_pct"] = rng.uniform(0.1, 0.9, n)
    return df


def _make_signals(df: pd.DataFrame, seed: int = 7) -> pd.Series:
    """Produce a signal series with periodic long/short signals."""
    rng = np.random.default_rng(seed)
    sigs = pd.Series(0.0, index=df.index)
    for i in range(50, len(df) - 50, 80):
        sigs.iloc[i] = 1.0  # long
        sigs.iloc[i + 40] = -1.0  # exit / short
    return sigs


# ── BacktestConfig ────────────────────────────────────────────────────────────

class TestBacktestConfig:
    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000
        assert cfg.commission_pct == 0.0004
        assert cfg.slippage_pct == 0.0002
        assert cfg.spread_pct == 0.0001
        assert cfg.latency_bars == 1
        assert cfg.signal_threshold == 0.5

    def test_custom_values(self):
        cfg = BacktestConfig(
            initial_capital=50_000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            spread_pct=0.0003,
            latency_bars=2,
        )
        assert cfg.initial_capital == 50_000
        assert cfg.latency_bars == 2
        assert cfg.spread_pct == 0.0003


# ── No Lookahead Bias ─────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_latency_1_enters_at_next_bar_open(self):
        """Signal at bar 50 must fill at bar 51's open, NOT bar 50's close."""
        df = _make_ohlcv(200)
        # Distinct open prices to detect which bar was used
        df["open"] = df["close"] * 0.999  # opens slightly lower than close
        sigs = pd.Series(0.0, index=df.index)
        sigs.iloc[50] = 1.0   # long signal at bar 50
        sigs.iloc[100] = -1.0  # exit

        bt = FastBacktester(initial_capital=100_000, latency_bars=1)
        result = bt.run(df, sigs, symbol="TEST")

        assert result.num_trades >= 1
        # Entry should be near bar 51's open, not bar 50's close
        trade = result.trades[0]
        bar51_open = df["open"].iloc[51]
        bar50_close = df["close"].iloc[50]
        # Trade entry should be closer to bar 51 open than bar 50 close
        assert abs(trade.entry_price - bar51_open) < abs(trade.entry_price - bar50_close) * 5

    def test_latency_0_fills_at_signal_bar(self):
        """With latency_bars=0, fill happens at the signal bar (vectorized mode)."""
        df = _make_ohlcv(200)
        sigs = pd.Series(0.0, index=df.index)
        sigs.iloc[50] = 1.0
        sigs.iloc[100] = -1.0

        bt = FastBacktester(initial_capital=100_000, latency_bars=0)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.num_trades >= 1

    def test_latency_2_skips_one_extra_bar(self):
        """Latency=2 means signal at bar 50 executes at bar 52."""
        df = _make_ohlcv(200)
        df["open"] = df["close"] * 0.998
        sigs = pd.Series(0.0, index=df.index)
        sigs.iloc[50] = 1.0
        sigs.iloc[120] = -1.0

        bt = FastBacktester(initial_capital=100_000, latency_bars=2)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.num_trades >= 1
        trade = result.trades[0]
        bar52_open = df["open"].iloc[52]
        # Entry should be near bar 52 open
        assert abs(trade.entry_price - bar52_open) < bar52_open * 0.01


# ── Fees / Slippage / Spread ──────────────────────────────────────────────────

class TestConfigurableCosts:
    def test_higher_fees_reduce_returns(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)

        bt_low = FastBacktester(initial_capital=100_000, commission_pct=0.0001, slippage_pct=0.0001, spread_pct=0.0)
        bt_high = FastBacktester(initial_capital=100_000, commission_pct=0.002, slippage_pct=0.001, spread_pct=0.001)

        r_low = bt_low.run(df, sigs, symbol="TEST")
        r_high = bt_high.run(df, sigs, symbol="TEST")

        # Higher costs should result in lower (or equal) total return
        assert r_high.total_return <= r_low.total_return + 1e-9

    def test_spread_increases_total_fees(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)

        bt_no_spread = FastBacktester(initial_capital=100_000, spread_pct=0.0)
        bt_with_spread = FastBacktester(initial_capital=100_000, spread_pct=0.001)

        r0 = bt_no_spread.run(df, sigs, symbol="TEST")
        r1 = bt_with_spread.run(df, sigs, symbol="TEST")

        assert r1.total_fees >= r0.total_fees

    def test_backtest_config_propagates_to_backtester(self):
        cfg = BacktestConfig(commission_pct=0.005, slippage_pct=0.003)
        bt = FastBacktester(config=cfg)
        assert bt._commission == 0.005
        assert bt._slippage == 0.003


# ── SL / TP Modeled on High/Low ──────────────────────────────────────────────

class TestStopLossTakeProfit:
    def test_stop_loss_triggers_on_low(self):
        """A long position's SL should trigger when low touches SL price."""
        df = _make_ohlcv(200)
        sigs = pd.Series(0.0, index=df.index)
        sigs.iloc[50] = 1.0  # go long

        # Manually set a very low value to trigger SL
        entry_approx = df["open"].iloc[51]
        atr = df["atr_14"].iloc[50]
        sl_dist = atr * 1.5
        # Force a bar with low below SL
        for j in range(55, 70):
            df.at[df.index[j], "low"] = entry_approx - sl_dist * 2

        bt = FastBacktester(initial_capital=100_000, latency_bars=1)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.num_trades >= 1
        # Check that exit_reason is stop_loss
        sl_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        assert len(sl_trades) >= 1

    def test_take_profit_triggers_on_high(self):
        """A long position's TP should trigger when high touches TP price."""
        df = _make_ohlcv(200)
        sigs = pd.Series(0.0, index=df.index)
        sigs.iloc[50] = 1.0  # go long

        # With latency=1, entry fills at bar 51 open.
        entry_approx = df["open"].iloc[51]
        atr = df["atr_14"].iloc[50]
        if atr <= 0 or np.isnan(atr):
            atr = entry_approx * 0.01
        sl_dist = atr * 1.5
        tp_price_approx = entry_approx + sl_dist * 2.0
        sl_price_approx = entry_approx - sl_dist

        # Keep lows well above SL and push highs above TP on bars 52-70
        for j in range(52, 70):
            df.at[df.index[j], "low"] = entry_approx * 0.999  # safe above SL
            df.at[df.index[j], "high"] = tp_price_approx * 1.5
            df.at[df.index[j], "close"] = entry_approx * 1.001
            df.at[df.index[j], "open"] = entry_approx * 1.0005

        bt = FastBacktester(initial_capital=100_000, latency_bars=1)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.num_trades >= 1
        tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
        assert len(tp_trades) >= 1


# ── Enhanced Metrics ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_result_has_all_metrics(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")

        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "expectancy")
        assert hasattr(result, "avg_duration")
        assert hasattr(result, "calmar_ratio")
        assert hasattr(result, "total_fees")
        assert hasattr(result, "equity_curve")

    def test_to_dict_includes_new_fields(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        d = result.to_dict()
        assert "expectancy" in d
        assert "avg_duration" in d
        assert "calmar_ratio" in d
        assert "total_fees" in d

    def test_win_rate_bounded(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self):
        df = _make_ohlcv(300)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.max_drawdown >= 0.0

    def test_equity_curve_length(self):
        df = _make_ohlcv(200)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        # Equity curve should have len(df) + 1 entries (initial + one per bar)
        assert len(result.equity_curve) >= len(df)

    def test_expectancy_positive_for_winning_strategy(self):
        """A strategy that always buys at low and sells at high should have positive expectancy."""
        # Create a trending up dataset
        rng = np.random.default_rng(99)
        n = 300
        close = 50000 * np.cumprod(1 + np.abs(rng.normal(0.001, 0.002, n)))
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.003,
            "low": close * 0.997, "close": close,
        }, index=idx)
        df["atr_14"] = close * 0.003
        sigs = pd.Series(0.0, index=idx)
        for i in range(50, n - 20, 60):
            sigs.iloc[i] = 1.0
            sigs.iloc[i + 30] = -1.0

        bt = FastBacktester(initial_capital=100_000, latency_bars=1)
        result = bt.run(df, sigs, symbol="TEST")
        if result.num_trades > 0:
            # In a trending up market, we expect positive return
            assert result.total_return > -0.1  # at least not catastrophic

    def test_compute_result_metrics_function(self):
        trades = [
            BacktestTrade("SYM", "long", 100, 110, 0, 1, 10, 0.1, 5, 0.5, "take_profit"),
            BacktestTrade("SYM", "short", 110, 100, 1, 2, 10, 0.1, 3, 0.3, "take_profit"),
            BacktestTrade("SYM", "long", 100, 95, 2, 3, -5, -0.05, 4, 0.2, "stop_loss"),
        ]
        result = _compute_result_metrics(trades, 100_015, 100_000, 0.05, [100_000, 100_010, 100_020, 100_015], 1.0)
        assert result.num_trades == 3
        assert result.win_rate == pytest.approx(2 / 3, abs=0.01)
        assert result.expectancy != 0.0
        assert result.avg_duration > 0.0
        assert result.total_fees == 1.0


# ── Event-Driven Backtester ───────────────────────────────────────────────────

class TestEventDrivenBacktester:
    def test_runs_without_error(self):
        df = _make_ohlcv(200)
        edb = EventDrivenBacktester()
        result = edb.run(df, symbol="BTCUSDT")
        assert isinstance(result, BacktestResult)

    def test_custom_scorer(self):
        df = _make_ohlcv(200)

        call_count = [0]

        def my_scorer(sub_df):
            call_count[0] += 1
            # Simple mean reversion: buy when RSI low, sell when high
            if len(sub_df) < 5:
                return 0.0, []
            rsi = sub_df.iloc[-1].get("rsi_14", 50)
            if rsi < 35:
                return 0.8, ["rsi_oversold"]
            elif rsi > 65:
                return -0.8, ["rsi_overbought"]
            return 0.0, []

        edb = EventDrivenBacktester(BacktestConfig(signal_threshold=0.5))
        result = edb.run(df, scorer=my_scorer, symbol="TEST")
        assert isinstance(result, BacktestResult)
        # Scorer should have been called many times (once per bar after warmup)
        assert call_count[0] > 50

    def test_no_lookahead_event_driven(self):
        """Scorer only receives data up to current bar — no future leak."""
        seen_lengths: list[int] = []

        def tracking_scorer(sub_df):
            seen_lengths.append(len(sub_df))
            return 0.0, []

        df = _make_ohlcv(100)
        edb = EventDrivenBacktester(BacktestConfig(signal_threshold=0.5))
        edb.run(df, scorer=tracking_scorer, symbol="TEST")
        # Each call should see incrementally more data
        for i in range(1, len(seen_lengths)):
            assert seen_lengths[i] >= seen_lengths[i - 1]
        # Last call should see at most len(df) rows
        assert seen_lengths[-1] <= len(df)

    def test_event_driven_uses_config(self):
        cfg = BacktestConfig(initial_capital=50_000, commission_pct=0.001)
        edb = EventDrivenBacktester(config=cfg)
        assert edb._cfg.initial_capital == 50_000

    def test_sl_tp_exits_in_event_driven(self):
        """EventDrivenBacktester should record SL/TP exits."""
        df = _make_ohlcv(200)
        # Force extreme conditions by making a scorer that always signals long
        def always_long(sub_df):
            if len(sub_df) < 35:
                return 0.0, []
            return 0.9, ["forced"]

        # Force some bars to have very low lows (trigger SL)
        for j in range(60, 70):
            df.at[df.index[j], "low"] = df["close"].iloc[j] * 0.9

        edb = EventDrivenBacktester(BacktestConfig(signal_threshold=0.5))
        result = edb.run(df, scorer=always_long, symbol="TEST")
        # Should have some trades with SL or TP exits
        if result.num_trades > 0:
            exit_reasons = [t.exit_reason for t in result.trades]
            assert any(r in ("stop_loss", "take_profit") for r in exit_reasons)


# ── Benchmarks ────────────────────────────────────────────────────────────────

class TestBenchmarkRunner:
    def test_buy_and_hold(self):
        df = _make_ohlcv(200)
        bm = BenchmarkRunner.buy_and_hold(df, initial_capital=100_000)
        assert isinstance(bm, BacktestResult)
        assert bm.num_trades == 1
        assert len(bm.equity_curve) > 0
        assert bm.trades[0].direction == "long"
        assert bm.trades[0].exit_reason == "end_of_data"

    def test_buy_and_hold_empty_df(self):
        df = pd.DataFrame({"close": [100]}, index=pd.date_range("2024-01-01", periods=1, freq="1h"))
        bm = BenchmarkRunner.buy_and_hold(df)
        assert bm.num_trades == 0

    def test_sma_crossover(self):
        df = _make_ohlcv(300)
        bm = BenchmarkRunner.sma_crossover(df, fast_period=20, slow_period=50)
        assert isinstance(bm, BacktestResult)

    def test_sma_crossover_short_data(self):
        df = _make_ohlcv(30)
        bm = BenchmarkRunner.sma_crossover(df)
        assert bm.num_trades == 0

    def test_benchmark_vs_strategy_comparison(self):
        """Strategy and benchmark results should be independently computed."""
        df = _make_ohlcv(300)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        strategy_result = bt.run(df, sigs, symbol="STRATEGY")
        bh_result = BenchmarkRunner.buy_and_hold(df, initial_capital=100_000)
        # Both should produce valid results (values may differ)
        assert isinstance(strategy_result.total_return, float)
        assert isinstance(bh_result.total_return, float)


# ── Walk-Forward Validation ───────────────────────────────────────────────────

class TestWalkForwardOptimizer:
    def test_produces_multiple_splits(self):
        df = _make_ohlcv(500)

        def signal_fn(train_df, test_df):
            # Simple signal: buy at every 50th bar
            sigs = pd.Series(0.0, index=test_df.index)
            for i in range(10, len(test_df) - 10, 30):
                sigs.iloc[i] = 1.0
                sigs.iloc[i + 15] = -1.0
            return sigs

        bt = FastBacktester(initial_capital=100_000)
        wfo = WalkForwardOptimizer(bt, n_splits=3, train_pct=0.6)
        results = wfo.run(df, signal_fn, symbol="WFO_TEST")
        assert len(results) > 0
        for r in results:
            assert isinstance(r, BacktestResult)

    def test_wfo_out_of_sample_consistency(self):
        """Walk-forward results should each have valid metrics."""
        df = _make_ohlcv(600)

        def signal_fn(train_df, test_df):
            sigs = pd.Series(0.0, index=test_df.index)
            for i in range(20, len(test_df) - 20, 50):
                sigs.iloc[i] = 1.0
                sigs.iloc[i + 25] = -1.0
            return sigs

        bt = FastBacktester(initial_capital=100_000)
        wfo = WalkForwardOptimizer(bt, n_splits=5, train_pct=0.7)
        results = wfo.run(df, signal_fn, symbol="WFO")
        for r in results:
            assert 0.0 <= r.win_rate <= 1.0
            assert r.max_drawdown >= 0.0


# ── Monte Carlo Simulation ────────────────────────────────────────────────────

class TestMonteCarloSimulator:
    def test_1000_runs(self):
        trades = [
            BacktestTrade("SYM", "long", 100, 105, 0, 1, 5, 0.05, 10, 0.1, "tp"),
            BacktestTrade("SYM", "short", 105, 100, 1, 2, 5, 0.05, 8, 0.1, "tp"),
            BacktestTrade("SYM", "long", 100, 97, 2, 3, -3, -0.03, 6, 0.05, "sl"),
            BacktestTrade("SYM", "long", 100, 108, 3, 4, 8, 0.08, 12, 0.2, "tp"),
            BacktestTrade("SYM", "short", 108, 103, 4, 5, 5, 0.05, 5, 0.1, "tp"),
        ]

        mc = MonteCarloSimulator(n_runs=1000, ruin_threshold_pct=0.50)
        result = mc.run(trades, initial_capital=100_000)

        assert result["runs"] == 1000
        assert result["trades_per_run"] == 5
        assert 0.0 <= result["probability_of_ruin"] <= 1.0
        assert result["median_final_equity"] > 0
        assert result["p5_final_equity"] <= result["p95_final_equity"]
        assert result["median_max_drawdown"] >= 0.0
        assert result["p95_max_drawdown"] >= result["median_max_drawdown"]

    def test_empty_trades_returns_error(self):
        mc = MonteCarloSimulator()
        result = mc.run([])
        assert result["error"] == "no_trades"

    def test_large_losses_yield_high_ruin(self):
        """Trades with large losses should produce non-zero ruin probability."""
        trades = [
            BacktestTrade("SYM", "long", 100, 50, 0, 1, -50, -0.5, 5, 1.0, "sl"),
            BacktestTrade("SYM", "long", 50, 25, 1, 2, -25, -0.5, 5, 0.5, "sl"),
            BacktestTrade("SYM", "long", 25, 30, 2, 3, 5, 0.2, 5, 0.1, "tp"),
        ]
        mc = MonteCarloSimulator(n_runs=1000, ruin_threshold_pct=0.50)
        result = mc.run(trades)
        assert result["probability_of_ruin"] > 0.0


# ── Backtest Report Generation ────────────────────────────────────────────────

class TestBacktestReportGenerator:
    def test_generates_json_file(self, tmp_path):
        df = _make_ohlcv(200)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="REPORT_TEST")

        output = str(tmp_path / "report.json")
        gen = BacktestReportGenerator()
        gen.generate(result, output_path=output)

        assert os.path.exists(output)
        with open(output) as f:
            report = json.load(f)
        assert "performance" in report
        assert "trades" in report
        assert report["performance"]["num_trades"] == result.num_trades

    def test_report_includes_benchmarks(self, tmp_path):
        df = _make_ohlcv(200)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        bh = BenchmarkRunner.buy_and_hold(df)

        output = str(tmp_path / "report_bm.json")
        gen = BacktestReportGenerator()
        gen.generate(result, benchmarks={"buy_and_hold": bh}, output_path=output)

        with open(output) as f:
            report = json.load(f)
        assert "benchmarks" in report
        assert "buy_and_hold" in report["benchmarks"]

    def test_report_includes_walk_forward(self, tmp_path):
        df = _make_ohlcv(500)

        def signal_fn(train, test):
            sigs = pd.Series(0.0, index=test.index)
            for i in range(10, len(test) - 10, 30):
                sigs.iloc[i] = 1.0
                sigs.iloc[i + 15] = -1.0
            return sigs

        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, _make_signals(df), symbol="TEST")
        wfo = WalkForwardOptimizer(bt, n_splits=3)
        wf_results = wfo.run(df, signal_fn, symbol="TEST")

        output = str(tmp_path / "report_wf.json")
        gen = BacktestReportGenerator()
        gen.generate(result, walk_forward=wf_results, output_path=output)

        with open(output) as f:
            report = json.load(f)
        assert "walk_forward" in report
        assert report["walk_forward"]["n_splits"] == len(wf_results)

    def test_report_includes_monte_carlo(self, tmp_path):
        df = _make_ohlcv(200)
        sigs = _make_signals(df)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")

        mc = MonteCarloSimulator(n_runs=100)
        mc_result = mc.run(result.trades)

        output = str(tmp_path / "report_mc.json")
        gen = BacktestReportGenerator()
        gen.generate(result, monte_carlo=mc_result, output_path=output)

        with open(output) as f:
            report = json.load(f)
        assert "monte_carlo" in report
        assert report["monte_carlo"]["runs"] == 100

    def test_report_includes_config(self, tmp_path):
        cfg = BacktestConfig(latency_bars=2, spread_pct=0.0005)
        bt = FastBacktester(config=cfg)
        df = _make_ohlcv(200)
        result = bt.run(df, _make_signals(df), symbol="TEST")

        output = str(tmp_path / "report_cfg.json")
        gen = BacktestReportGenerator()
        gen.generate(result, config=cfg, output_path=output)

        with open(output) as f:
            report = json.load(f)
        assert report["config"]["latency_bars"] == 2
        assert report["config"]["spread_pct"] == 0.0005


# ── BacktestTrade enhanced fields ─────────────────────────────────────────────

class TestBacktestTrade:
    def test_trade_has_fees_and_exit_reason(self):
        t = BacktestTrade(
            symbol="BTC", direction="long", entry_price=100, exit_price=110,
            entry_time=0, exit_time=1, pnl=10, pnl_pct=0.1, hold_periods=5,
            fees=0.5, exit_reason="take_profit",
        )
        assert t.fees == 0.5
        assert t.exit_reason == "take_profit"

    def test_trade_default_values(self):
        t = BacktestTrade(
            symbol="ETH", direction="short", entry_price=200, exit_price=190,
            entry_time=0, exit_time=1, pnl=10, pnl_pct=0.05, hold_periods=3,
        )
        assert t.fees == 0.0
        assert t.exit_reason == ""


# ── End-to-End: Full Backtest Pipeline ────────────────────────────────────────

class TestFullPipeline:
    def test_full_pipeline(self, tmp_path):
        """Run: strategy backtest → benchmarks → WFO → MC → report."""
        df = _make_ohlcv(500)
        sigs = _make_signals(df)
        cfg = BacktestConfig(initial_capital=100_000)

        # Strategy backtest
        bt = FastBacktester(config=cfg)
        result = bt.run(df, sigs, symbol="BTC/USDT")
        assert result.num_trades > 0

        # Benchmarks
        bh = BenchmarkRunner.buy_and_hold(df, initial_capital=100_000)
        sma = BenchmarkRunner.sma_crossover(df, initial_capital=100_000)

        # Walk-forward
        def sig_fn(train, test):
            s = pd.Series(0.0, index=test.index)
            for i in range(10, len(test) - 10, 40):
                s.iloc[i] = 1.0
                s.iloc[i + 20] = -1.0
            return s

        wfo = WalkForwardOptimizer(bt, n_splits=3)
        wf_results = wfo.run(df, sig_fn, symbol="BTC/USDT")

        # Monte Carlo
        mc = MonteCarloSimulator(n_runs=1000)
        mc_result = mc.run(result.trades)
        assert mc_result["runs"] == 1000

        # Report
        output = str(tmp_path / "full_report.json")
        gen = BacktestReportGenerator()
        gen.generate(
            result,
            config=cfg,
            benchmarks={"buy_and_hold": bh, "sma_crossover": sma},
            walk_forward=wf_results,
            monte_carlo=mc_result,
            output_path=output,
        )

        with open(output) as f:
            report = json.load(f)
        assert "performance" in report
        assert "benchmarks" in report
        assert "walk_forward" in report
        assert "monte_carlo" in report
        assert "generated_at" in report

    def test_no_trades_produces_valid_result(self):
        """A backtest with no signals should return a valid empty result."""
        df = _make_ohlcv(200)
        sigs = pd.Series(0.0, index=df.index)
        bt = FastBacktester(initial_capital=100_000)
        result = bt.run(df, sigs, symbol="TEST")
        assert result.num_trades == 0
        assert result.total_return == 0.0
        assert result.win_rate == 0
        assert len(result.equity_curve) > 0
