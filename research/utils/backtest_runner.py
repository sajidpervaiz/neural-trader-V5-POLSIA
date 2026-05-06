"""Backtest runner utilities for research notebooks."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.fast_backtester import FastBacktester, BacktestResult, WalkForwardOptimizer
from engine.strategy_registry import registry


def run_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 100_000,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0002,
    symbol: str = "UNKNOWN",
) -> BacktestResult:
    """Run a named strategy from the registry against a price DataFrame."""
    signal_fn = registry.get(strategy_name)
    if signal_fn is None:
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {registry.list()}")
    signals = signal_fn(df)
    bt = FastBacktester(initial_capital, commission_pct, slippage_pct)
    return bt.run(df, signals, symbol=symbol)


def parameter_sweep(
    df: pd.DataFrame,
    strategy_fn: Callable[..., pd.Series],
    param_grid: dict[str, list[Any]],
    metric: str = "sharpe_ratio",
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """Run a strategy across a parameter grid and return results sorted by metric."""
    import itertools
    bt = FastBacktester(initial_capital)
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            signals = strategy_fn(df, **params)
            result = bt.run(df, signals)
            row = {**params, **result.to_dict()}
            results.append(row)
        except Exception:
            pass
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(metric, ascending=False).reset_index(drop=True)


def walk_forward_analysis(
    df: pd.DataFrame,
    strategy_name: str,
    n_splits: int = 5,
    train_pct: float = 0.7,
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """Run walk-forward optimization and return per-split results."""
    signal_fn = registry.get(strategy_name)
    if signal_fn is None:
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {registry.list()}")

    def wfo_signal_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        return signal_fn(test_df)

    bt = FastBacktester(initial_capital)
    optimizer = WalkForwardOptimizer(bt, n_splits=n_splits, train_pct=train_pct)
    split_results = optimizer.run(df, wfo_signal_fn, symbol="WFO")
    if not split_results:
        return pd.DataFrame()
    rows = [r.to_dict() for r in split_results]
    return pd.DataFrame(rows)


def monte_carlo_simulation(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_periods: int = 252,
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """Run Monte Carlo simulation on historical returns."""
    import numpy as np
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    results = []
    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        sim_returns = rng.normal(mean_ret, std_ret, n_periods)
        equity = initial_capital * (1 + sim_returns).cumprod()
        final = equity[-1]
        dd = float((1 - equity / equity.cummax()).max())
        results.append({"final_equity": final, "total_return": (final - initial_capital) / initial_capital, "max_drawdown": dd})
    return pd.DataFrame(results)
