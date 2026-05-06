from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

try:
    import neural_trader_rust as _rust
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configurable backtest assumptions — MUST mirror live trading parameters
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Single source of truth for backtest cost/execution assumptions.

    Every parameter here has a direct live-mode counterpart.  Changing
    a value in one place (config/settings.yaml → ``backtest:`` section)
    propagates uniformly to both backtest and paper-trade paths.
    """
    initial_capital: float = 100_000
    commission_pct: float = 0.0004       # per side (same as exchange tier)
    slippage_pct: float = 0.0002         # market-impact estimate
    spread_pct: float = 0.0001           # half-spread per side
    latency_bars: int = 1                # 1 = execute on NEXT bar open (no lookahead)
    position_size_pct: float = 0.02      # fixed-fraction fallback
    risk_per_trade_pct: float = 0.01     # risk-based sizing fraction
    max_open_positions: int = 5
    stop_loss_atr_mult: float = 1.5
    take_profit_rr: float = 2.0
    signal_threshold: float = 0.5
    use_risk_sizing: bool = True         # True = risk-based, False = fixed fraction


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: int
    exit_time: int
    pnl: float
    pnl_pct: float
    hold_periods: int
    fees: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_pnl: float
    expectancy: float = 0.0
    avg_duration: float = 0.0
    calmar_ratio: float = 0.0
    total_fees: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_pnl": self.avg_pnl,
            "expectancy": self.expectancy,
            "avg_duration": self.avg_duration,
            "calmar_ratio": self.calmar_ratio,
            "total_fees": self.total_fees,
        }


class FastBacktester:
    """Backtester with optional Rust-accelerated inner loop.

    Enforces no-lookahead via ``latency_bars`` (default 1): a signal at
    bar *i* triggers execution at bar *i + latency_bars* open, NOT at bar
    *i* close.  Commission, slippage **and** spread are applied per-side.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_pct: float = 0.0004,
        slippage_pct: float = 0.0002,
        spread_pct: float = 0.0001,
        latency_bars: int = 1,
        config: BacktestConfig | None = None,
    ) -> None:
        if config is not None:
            self._capital = config.initial_capital
            self._commission = config.commission_pct
            self._slippage = config.slippage_pct
            self._spread = config.spread_pct
            self._latency = config.latency_bars
            self._sl_atr_mult = config.stop_loss_atr_mult
            self._tp_rr = config.take_profit_rr
            self._size_pct = config.position_size_pct
        else:
            self._capital = initial_capital
            self._commission = commission_pct
            self._slippage = slippage_pct
            self._spread = spread_pct
            self._latency = latency_bars
            self._sl_atr_mult = 1.5
            self._tp_rr = 2.0
            self._size_pct = 0.02

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        # Python path is feature-complete (latency, SL/TP, trades, equity_curve).
        # Rust path is a coarse approximation and returns a degraded result, so
        # only use it when the caller explicitly opts in via _run_rust().
        return self._run_python(df, signals, symbol)

    def _run_python(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
    ) -> BacktestResult:
        equity = self._capital
        peak_equity = equity
        max_dd = 0.0
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [equity]
        total_fees = 0.0
        position: dict[str, Any] | None = None
        pending_signal: dict[str, Any] | None = None  # for latency delay

        close_prices = df["close"].values
        open_prices = df["open"].values if "open" in df.columns else close_prices
        high_prices = df["high"].values if "high" in df.columns else close_prices
        low_prices = df["low"].values if "low" in df.columns else close_prices
        atr_values = df["atr_14"].values if "atr_14" in df.columns else np.full(len(df), 0.0)
        signal_values = signals.reindex(df.index).fillna(0).values
        timestamps = df.index.astype(np.int64) // 10**9 if hasattr(df.index, "astype") else list(range(len(df)))

        cost_per_side = self._commission + self._slippage + self._spread

        for i in range(len(close_prices)):
            price = close_prices[i]
            sig = signal_values[i]
            ts = timestamps[i]

            # ── Execute pending signal (latency delay: no lookahead) ──────
            if pending_signal is not None and i >= pending_signal["exec_bar"]:
                direction = pending_signal["direction"]
                fill_price = open_prices[i] * (1 + cost_per_side if direction == "long" else 1 - cost_per_side)
                atr = atr_values[min(pending_signal["signal_bar"], len(atr_values) - 1)]
                if atr <= 0 or atr != atr:
                    atr = fill_price * 0.01
                sl_dist = atr * self._sl_atr_mult
                if direction == "long":
                    sl_price = fill_price - sl_dist
                    tp_price = fill_price + sl_dist * self._tp_rr
                else:
                    sl_price = fill_price + sl_dist
                    tp_price = fill_price - sl_dist * self._tp_rr
                entry_fee = equity * self._size_pct * cost_per_side
                total_fees += entry_fee
                equity -= entry_fee
                position = {
                    "direction": direction,
                    "entry_price": fill_price,
                    "entry_time": ts,
                    "entry_idx": i,
                    "size_pct": self._size_pct,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                }
                pending_signal = None

            # ── Check position SL/TP/exit ─────────────────────────────────
            if position is not None:
                should_exit = False
                exit_price = price
                exit_reason = "signal"

                # Check SL/TP using high/low to detect intra-bar hits
                if position["direction"] == "long":
                    if low_prices[i] <= position["stop_loss"]:
                        should_exit = True
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif high_prices[i] >= position["take_profit"]:
                        should_exit = True
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"
                    elif sig < 0:
                        should_exit = True
                        exit_price = price * (1 - cost_per_side)
                else:
                    if high_prices[i] >= position["stop_loss"]:
                        should_exit = True
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif low_prices[i] <= position["take_profit"]:
                        should_exit = True
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"
                    elif sig > 0:
                        should_exit = True
                        exit_price = price * (1 + cost_per_side)

                if should_exit:
                    entry_price = position["entry_price"]
                    if position["direction"] == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    # Commission on exit side
                    exit_fee = equity * position["size_pct"] * cost_per_side
                    total_fees += exit_fee
                    pnl_pct -= cost_per_side  # exit costs
                    trade_pnl = equity * position["size_pct"] * pnl_pct
                    equity += trade_pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                    max_dd = max(max_dd, dd)

                    trades.append(BacktestTrade(
                        symbol=symbol,
                        direction=position["direction"],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=int(position["entry_time"]),
                        exit_time=int(ts),
                        pnl=trade_pnl,
                        pnl_pct=pnl_pct,
                        hold_periods=i - position["entry_idx"],
                        fees=entry_fee + exit_fee if exit_reason != "signal" else exit_fee,
                        exit_reason=exit_reason,
                    ))
                    position = None

            # ── Queue new signal with latency delay ───────────────────────
            if position is None and pending_signal is None and abs(sig) > 0.5:
                direction = "long" if sig > 0 else "short"
                exec_bar = i + self._latency  # execute on future bar
                if exec_bar < len(close_prices):
                    pending_signal = {
                        "direction": direction,
                        "signal_bar": i,
                        "exec_bar": exec_bar,
                    }

            equity_curve.append(equity)

        if not trades:
            return BacktestResult(
                0, 0, 0, 0, 0, 0, 0, 0,
                equity_curve=equity_curve,
                total_fees=total_fees,
            )

        return _compute_result_metrics(trades, equity, self._capital, max_dd, equity_curve, total_fees)

    def _run_rust(self, df: pd.DataFrame, signals: pd.Series, symbol: str) -> BacktestResult:
        prices = df["close"].tolist()
        sigs = signals.reindex(df.index).fillna(0).tolist()
        result = _rust.FastBacktester.run(
            prices=prices,
            signals=sigs,
            initial_capital=self._capital,
            commission_pct=self._commission,
            slippage_pct=self._slippage,
        )
        return BacktestResult(
            total_return=result["total_return"],
            sharpe_ratio=result["sharpe_ratio"],
            sortino_ratio=result["sortino_ratio"],
            max_drawdown=result["max_drawdown"],
            win_rate=result["win_rate"],
            profit_factor=result["profit_factor"],
            num_trades=result["num_trades"],
            avg_pnl=result["avg_pnl"],
        )


class WalkForwardOptimizer:
    def __init__(
        self,
        backtester: FastBacktester,
        n_splits: int = 5,
        train_pct: float = 0.7,
    ) -> None:
        self._bt = backtester
        self._n_splits = n_splits
        self._train_pct = train_pct

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Any,
        symbol: str = "UNKNOWN",
    ) -> list[BacktestResult]:
        total_len = len(df)
        split_size = total_len // self._n_splits
        results = []

        for i in range(self._n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self._n_splits - 1 else total_len
            split_df = df.iloc[start:end].copy()
            train_len = int(len(split_df) * self._train_pct)
            test_df = split_df.iloc[train_len:]

            if len(test_df) < 5:
                continue

            try:
                signals = signal_fn(split_df.iloc[:train_len], test_df)
                result = self._bt.run(test_df, signals, symbol=symbol)
                results.append(result)
                logger.debug(
                    "WFO split {}/{}: return={:.2%} sharpe={:.2f}",
                    i + 1, self._n_splits, result.total_return, result.sharpe_ratio
                )
            except Exception as exc:
                logger.warning("WFO split {} error: {}", i + 1, exc)

        return results


class MonteCarloSimulator:
    """Monte Carlo simulation over trade sequences to estimate probability of ruin."""

    def __init__(self, n_runs: int = 1000, ruin_threshold_pct: float = 0.50) -> None:
        self._n_runs = n_runs
        self._ruin_threshold = ruin_threshold_pct

    def run(self, trades: list[BacktestTrade], initial_capital: float = 100_000) -> dict[str, Any]:
        if not trades:
            return {"error": "no_trades", "runs": 0}

        pnl_pcts = np.array([t.pnl_pct for t in trades])
        n_trades = len(pnl_pcts)
        ruin_count = 0
        final_equities: list[float] = []
        max_drawdowns: list[float] = []
        rng = np.random.default_rng(42)

        for _ in range(self._n_runs):
            # Shuffle trade order
            shuffled = rng.permutation(pnl_pcts)
            equity = initial_capital
            peak = equity
            max_dd = 0.0

            for pnl_pct in shuffled:
                equity *= (1.0 + pnl_pct)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

                if dd >= self._ruin_threshold:
                    ruin_count += 1
                    break

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        final_arr = np.array(final_equities)
        dd_arr = np.array(max_drawdowns)

        return {
            "runs": self._n_runs,
            "trades_per_run": n_trades,
            "probability_of_ruin": ruin_count / self._n_runs,
            "ruin_threshold_pct": self._ruin_threshold,
            "ruin_count": ruin_count,
            "median_final_equity": float(np.median(final_arr)),
            "mean_final_equity": float(np.mean(final_arr)),
            "p5_final_equity": float(np.percentile(final_arr, 5)),
            "p95_final_equity": float(np.percentile(final_arr, 95)),
            "median_max_drawdown": float(np.median(dd_arr)),
            "p95_max_drawdown": float(np.percentile(dd_arr, 95)),
            "mean_max_drawdown": float(np.mean(dd_arr)),
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_result_metrics(
    trades: list[BacktestTrade],
    final_equity: float,
    initial_capital: float,
    max_dd: float,
    equity_curve: list[float],
    total_fees: float,
) -> BacktestResult:
    """Compute all performance metrics from a list of completed trades."""
    total_return = (final_equity - initial_capital) / initial_capital
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_loss = abs(sum(losses))
    profit_factor = abs(sum(wins)) / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy = avg_win * win_rate - avg_loss * loss_rate
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean([abs(l) for l in losses])) if losses else 0.0
    loss_rate = 1.0 - win_rate
    expectancy = avg_win * win_rate - avg_loss * loss_rate

    # Average trade duration
    durations = [t.hold_periods for t in trades]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    # Sharpe / Sortino
    daily_returns = pd.Series(pnl_pcts)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    downside = daily_returns[daily_returns < 0].std()
    sortino = float(daily_returns.mean() / downside * np.sqrt(252)) if (downside is not None and downside > 0) else 0.0

    # Calmar ratio = annual_return / max_drawdown
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        expectancy=expectancy,
        avg_duration=avg_duration,
        calmar_ratio=calmar,
        total_fees=total_fees,
        trades=trades,
        equity_curve=equity_curve,
    )


# ---------------------------------------------------------------------------
# Event-Driven Backtester — mirrors the live execution path exactly
# ---------------------------------------------------------------------------

class EventDrivenBacktester:
    """Bar-by-bar backtester that reuses the live signal scorer and
    risk checks.  Eliminates lookahead bias structurally:

    1. Signal computed using data up to and including bar *i*.
    2. Entry executes at bar *i + 1* open (configurable latency).
    3. Fees = commission + slippage + spread per side.
    4. SL/TP checked on high/low each bar (same logic as live trailing).
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._cfg = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        scorer: Callable[[pd.DataFrame], tuple[float, list[str]]] | None = None,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run event-driven backtest.

        ``scorer`` receives df[:i+1] and returns (score_in_[-1,1], reasons).
        If None, uses TechnicalScorer from signal_generator.
        """
        from engine.signal_generator import TechnicalScorer

        if scorer is None:
            _ts = TechnicalScorer()
            scorer = lambda sub_df: _ts.score(sub_df)

        cfg = self._cfg
        equity = cfg.initial_capital
        peak_equity = equity
        max_dd = 0.0
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [equity]
        total_fees = 0.0
        position: dict[str, Any] | None = None
        pending_entry: dict[str, Any] | None = None
        cost_per_side = cfg.commission_pct + cfg.slippage_pct + cfg.spread_pct

        close_prices = df["close"].values
        open_prices = df["open"].values if "open" in df.columns else close_prices
        high_prices = df["high"].values if "high" in df.columns else close_prices
        low_prices = df["low"].values if "low" in df.columns else close_prices
        atr_values = df["atr_14"].values if "atr_14" in df.columns else np.full(len(df), 0.0)
        timestamps = (
            df.index.astype(np.int64) // 10**9
            if hasattr(df.index, "astype")
            else np.arange(len(df))
        )

        for i in range(len(close_prices)):
            ts = int(timestamps[i])

            # ── Fill pending entry at this bar's open ─────────────────────
            if pending_entry is not None and i >= pending_entry["exec_bar"]:
                direction = pending_entry["direction"]
                fill_price = open_prices[i] * (
                    1 + cost_per_side if direction == "long" else 1 - cost_per_side
                )
                atr = atr_values[min(pending_entry["signal_bar"], len(atr_values) - 1)]
                if atr <= 0 or atr != atr:
                    atr = fill_price * 0.01
                sl_dist = atr * cfg.stop_loss_atr_mult
                if direction == "long":
                    sl = fill_price - sl_dist
                    tp = fill_price + sl_dist * cfg.take_profit_rr
                else:
                    sl = fill_price + sl_dist
                    tp = fill_price - sl_dist * cfg.take_profit_rr
                entry_fee = equity * cfg.position_size_pct * cost_per_side
                total_fees += entry_fee
                equity -= entry_fee
                position = {
                    "direction": direction,
                    "entry_price": fill_price,
                    "entry_time": ts,
                    "entry_idx": i,
                    "stop_loss": sl,
                    "take_profit": tp,
                }
                pending_entry = None

            # ── Check SL/TP/signal-exit on open position ──────────────────
            if position is not None:
                should_exit = False
                exit_price = close_prices[i]
                exit_reason = ""

                if position["direction"] == "long":
                    if low_prices[i] <= position["stop_loss"]:
                        should_exit, exit_price, exit_reason = True, position["stop_loss"], "stop_loss"
                    elif high_prices[i] >= position["take_profit"]:
                        should_exit, exit_price, exit_reason = True, position["take_profit"], "take_profit"
                else:
                    if high_prices[i] >= position["stop_loss"]:
                        should_exit, exit_price, exit_reason = True, position["stop_loss"], "stop_loss"
                    elif low_prices[i] <= position["take_profit"]:
                        should_exit, exit_price, exit_reason = True, position["take_profit"], "take_profit"

                if should_exit:
                    entry_price = position["entry_price"]
                    if position["direction"] == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    exit_fee = equity * cfg.position_size_pct * cost_per_side
                    total_fees += exit_fee
                    pnl_pct -= cost_per_side
                    trade_pnl = equity * cfg.position_size_pct * pnl_pct
                    equity += trade_pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                    max_dd = max(max_dd, dd)
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        direction=position["direction"],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        pnl=trade_pnl,
                        pnl_pct=pnl_pct,
                        hold_periods=i - position["entry_idx"],
                        fees=exit_fee,
                        exit_reason=exit_reason,
                    ))
                    position = None

            # ── Score current bar (only sees data up to bar i) ────────────
            if position is None and pending_entry is None and i >= 30:
                sub_df = df.iloc[: i + 1]
                score, _reasons = scorer(sub_df)
                if abs(score) >= cfg.signal_threshold:
                    exec_bar = i + cfg.latency_bars
                    if exec_bar < len(close_prices):
                        pending_entry = {
                            "direction": "long" if score > 0 else "short",
                            "signal_bar": i,
                            "exec_bar": exec_bar,
                        }

            equity_curve.append(equity)

        if not trades:
            return BacktestResult(
                0, 0, 0, 0, 0, 0, 0, 0,
                equity_curve=equity_curve,
                total_fees=total_fees,
            )
        return _compute_result_metrics(trades, equity, cfg.initial_capital, max_dd, equity_curve, total_fees)


# ---------------------------------------------------------------------------
# Benchmarks — buy-and-hold & SMA crossover baselines
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Generate benchmark equity curves for comparison."""

    @staticmethod
    def buy_and_hold(
        df: pd.DataFrame,
        initial_capital: float = 100_000,
        commission_pct: float = 0.0004,
    ) -> BacktestResult:
        """Simple buy at first bar, sell at last bar."""
        if df is None or len(df) < 2:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0)
        close = df["close"].values
        entry = close[0] * (1 + commission_pct)
        exit_price = close[-1] * (1 - commission_pct)
        pnl_pct = (exit_price - entry) / entry
        total_fees = initial_capital * commission_pct * 2

        equity_curve = [initial_capital]
        peak = initial_capital
        max_dd = 0.0
        for p in close[1:]:
            eq = initial_capital * (1 + (p - entry) / entry)
            equity_curve.append(eq)
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        timestamps = (
            df.index.astype(np.int64) // 10**9
            if hasattr(df.index, "astype")
            else list(range(len(df)))
        )
        trade = BacktestTrade(
            symbol="BUY_AND_HOLD",
            direction="long",
            entry_price=entry,
            exit_price=exit_price,
            entry_time=int(timestamps[0]),
            exit_time=int(timestamps[-1]),
            pnl=initial_capital * pnl_pct,
            pnl_pct=pnl_pct,
            hold_periods=len(df) - 1,
            fees=total_fees,
            exit_reason="end_of_data",
        )
        return BacktestResult(
            total_return=pnl_pct,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=max_dd,
            win_rate=1.0 if pnl_pct > 0 else 0.0,
            profit_factor=float("inf") if pnl_pct > 0 else 0.0,
            num_trades=1,
            avg_pnl=initial_capital * pnl_pct,
            trades=[trade],
            equity_curve=equity_curve,
            total_fees=total_fees,
        )

    @staticmethod
    def sma_crossover(
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        initial_capital: float = 100_000,
        commission_pct: float = 0.0004,
        slippage_pct: float = 0.0002,
    ) -> BacktestResult:
        """SMA crossover baseline: long when SMA_fast > SMA_slow, else flat."""
        if df is None or len(df) < slow_period + 5:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0)
        close = df["close"]
        sma_fast = close.rolling(fast_period).mean()
        sma_slow = close.rolling(slow_period).mean()

        signals = pd.Series(0.0, index=df.index)
        above = sma_fast > sma_slow
        # Go long on cross-above, flat on cross-below
        signals[above & ~above.shift(1).fillna(False)] = 1.0
        signals[~above & above.shift(1).fillna(False)] = -1.0

        bt = FastBacktester(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            latency_bars=1,
        )
        return bt.run(df, signals, symbol="SMA_CROSSOVER")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class BacktestReportGenerator:
    """Produce a JSON backtest report file with metrics, trades,
    benchmark comparison, walk-forward, and Monte Carlo results."""

    def generate(
        self,
        result: BacktestResult,
        *,
        config: BacktestConfig | None = None,
        benchmarks: dict[str, BacktestResult] | None = None,
        walk_forward: list[BacktestResult] | None = None,
        monte_carlo: dict[str, Any] | None = None,
        output_path: str = "backtest_report.json",
    ) -> str:
        report: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "initial_capital": config.initial_capital if config else 100_000,
                "commission_pct": config.commission_pct if config else 0.0004,
                "slippage_pct": config.slippage_pct if config else 0.0002,
                "spread_pct": config.spread_pct if config else 0.0001,
                "latency_bars": config.latency_bars if config else 1,
            } if config else {},
            "performance": result.to_dict(),
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "hold_periods": t.hold_periods,
                    "fees": t.fees,
                    "exit_reason": t.exit_reason,
                }
                for t in result.trades
            ],
            "equity_curve_length": len(result.equity_curve),
        }

        if benchmarks:
            report["benchmarks"] = {
                name: bm.to_dict() for name, bm in benchmarks.items()
            }

        if walk_forward:
            report["walk_forward"] = {
                "n_splits": len(walk_forward),
                "splits": [wf.to_dict() for wf in walk_forward],
                "avg_sharpe": float(np.mean([wf.sharpe_ratio for wf in walk_forward])) if walk_forward else 0.0,
                "avg_return": float(np.mean([wf.total_return for wf in walk_forward])) if walk_forward else 0.0,
            }

        if monte_carlo:
            report["monte_carlo"] = monte_carlo

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Backtest report written to {}", output_path)
        return output_path
