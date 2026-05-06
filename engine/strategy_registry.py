from __future__ import annotations

from typing import Callable

import pandas as pd
from loguru import logger


SignalFn = Callable[[pd.DataFrame], pd.Series]


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, SignalFn] = {}

    def register(self, name: str, fn: SignalFn) -> None:
        self._strategies[name] = fn
        logger.debug("Registered strategy '{}'", name)

    def get(self, name: str) -> SignalFn | None:
        return self._strategies.get(name)

    def list(self) -> list[str]:
        return list(self._strategies.keys())

    def __call__(self, name: str) -> Callable[[SignalFn], SignalFn]:
        def decorator(fn: SignalFn) -> SignalFn:
            self.register(name, fn)
            return fn
        return decorator


registry = StrategyRegistry()


@registry("ema_crossover")
def ema_crossover_signals(df: pd.DataFrame) -> pd.Series:
    if "ema_9" not in df.columns or "ema_21" not in df.columns:
        return pd.Series(0.0, index=df.index)
    cross_up = (df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
    cross_down = (df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[cross_up] = 1.0
    signal[cross_down] = -1.0
    return signal.fillna(0)


@registry("rsi_mean_reversion")
def rsi_mean_reversion_signals(df: pd.DataFrame) -> pd.Series:
    if "rsi_14" not in df.columns:
        return pd.Series(0.0, index=df.index)
    signal = pd.Series(0.0, index=df.index)
    signal[df["rsi_14"] < 30] = 1.0
    signal[df["rsi_14"] > 70] = -1.0
    return signal.fillna(0)


@registry("macd_signal")
def macd_signals(df: pd.DataFrame) -> pd.Series:
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        return pd.Series(0.0, index=df.index)
    above = df["macd"] > df["macd_signal"]
    cross_up = above & ~above.shift(1).fillna(False)
    cross_down = ~above & above.shift(1).fillna(False)
    signal = pd.Series(0.0, index=df.index)
    signal[cross_up] = 1.0
    signal[cross_down] = -1.0
    return signal.fillna(0)


@registry("bb_reversion")
def bb_reversion_signals(df: pd.DataFrame) -> pd.Series:
    if "bb_pct" not in df.columns:
        return pd.Series(0.0, index=df.index)
    signal = pd.Series(0.0, index=df.index)
    signal[df["bb_pct"] < 0.05] = 1.0
    signal[df["bb_pct"] > 0.95] = -1.0
    return signal.fillna(0)
