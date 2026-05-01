from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import talib as ta
    _TALIB = True
except ImportError:
    _TALIB = False
    try:
        import pandas_ta as pta
        _PANDAS_TA = True
    except ImportError:
        _PANDAS_TA = False


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _TALIB:
        return pd.Series(ta.RSI(series.values, timeperiod=period), index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if _TALIB:
        m, s, h = ta.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        idx = series.index
        return pd.Series(m, index=idx), pd.Series(s, index=idx), pd.Series(h, index=idx)
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if _TALIB:
        upper, middle, lower = ta.BBANDS(series.values, timeperiod=period, nbdevup=num_std, nbdevdn=num_std)
        idx = series.index
        return pd.Series(upper, index=idx), pd.Series(middle, index=idx), pd.Series(lower, index=idx)
    mid = _sma(series, period)
    std = series.rolling(window=period).std()
    return mid + num_std * std, mid, mid - num_std * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if _TALIB:
        return pd.Series(ta.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    if _TALIB:
        k, d = ta.STOCH(high.values, low.values, close.values, fastk_period=k_period, slowd_period=d_period)
        return pd.Series(k, index=close.index), pd.Series(d, index=close.index)
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    d = _sma(k, d_period)
    return k, d


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical * volume).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _keltner_channels(
    close: pd.Series, high: pd.Series, low: pd.Series,
    period: int = 20, multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: EMA ± multiplier * ATR."""
    ema = _ema(close, period)
    atr_s = _atr(high, low, close, period)
    upper = ema + multiplier * atr_s
    lower = ema - multiplier * atr_s
    return upper, ema, lower


def _supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """SuperTrend indicator. Returns (supertrend_line, direction) where
    direction = 1 (bullish / price above) or -1 (bearish / price below)."""
    atr_s = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_s
    lower_band = hl2 - multiplier * atr_s

    st = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(1, index=close.index, dtype=int)

    for i in range(1, len(close)):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            st.iloc[i] = st.iloc[i - 1] if not np.isnan(st.iloc[i - 1]) else close.iloc[i]
            direction.iloc[i] = direction.iloc[i - 1]
            continue

        if close.iloc[i] > upper_band.iloc[i - 1] if not np.isnan(upper_band.iloc[i - 1]) else False:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1] if not np.isnan(lower_band.iloc[i - 1]) else False:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return st, direction.astype(float)


def _ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> dict[str, pd.Series]:
    """Ichimoku Cloud components."""
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou_span": chikou,
    }


def _parabolic_sar(
    high: pd.Series, low: pd.Series,
    af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.20,
) -> pd.Series:
    """Parabolic SAR."""
    length = len(high)
    sar = pd.Series(np.nan, index=high.index, dtype=float)
    if length < 2:
        return sar
    bull = True
    af = af_start
    ep = high.iloc[0]
    sar.iloc[0] = low.iloc[0]
    for i in range(1, length):
        prev_sar = sar.iloc[i - 1] if not np.isnan(sar.iloc[i - 1]) else low.iloc[i]
        sar.iloc[i] = prev_sar + af * (ep - prev_sar)
        if bull:
            if low.iloc[i] < sar.iloc[i]:
                bull = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            if high.iloc[i] > sar.iloc[i]:
                bull = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
    return sar


def _aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up and Aroon Down."""
    aroon_up = high.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    return aroon_up, aroon_down


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    direction = typical_price.diff()
    pos_mf = pd.Series(np.where(direction > 0, raw_mf, 0.0), index=close.index)
    neg_mf = pd.Series(np.where(direction < 0, raw_mf, 0.0), index=close.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series,
    p1: int = 7, p2: int = 14, p3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator (multi-period weighted)."""
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum().replace(0, np.nan)
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum().replace(0, np.nan)
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum().replace(0, np.nan)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def _trix(close: pd.Series, period: int = 15) -> pd.Series:
    """TRIX: triple EMA rate of change."""
    ema1 = _ema(close, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return ema3.pct_change() * 100


def _cmf(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (linear weighting, newest bar heaviest)."""
    weights = np.arange(1, period + 1, dtype=float)
    weights_sum = weights.sum()
    return series.rolling(period).apply(
        lambda x: float(np.dot(x, weights) / weights_sum), raw=True,
    )


def _vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average. VWMA above SMA = buying pressure."""
    pv = close * volume
    return pv.rolling(period, min_periods=1).sum() / volume.rolling(period, min_periods=1).sum().replace(0, np.nan)


def _hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average. HMA = WMA(2*WMA(n/2) − WMA(n), sqrt(n))."""
    half = max(2, int(period / 2))
    sqrt_n = max(2, int(np.sqrt(period)))
    raw = 2 * _wma(series, half) - _wma(series, period)
    return _wma(raw, sqrt_n)


def _alma(series: pd.Series, period: int = 20, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average. Gaussian-weighted, low lag."""
    if period < 2:
        return series.copy()
    m = offset * (period - 1)
    s = period / float(sigma)
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s ** 2))
    weights = weights / weights.sum()
    return series.rolling(period).apply(
        lambda x: float(np.dot(x, weights)), raw=True,
    )


def _roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change as percent: (close − close[N]) / close[N] * 100."""
    return close.pct_change(period) * 100.0


def _stoch_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI on a 0..100 scale. Returns (%K, %D)."""
    rsi = _rsi(close, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    raw_k = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100.0
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def _vw_macd(
    close: pd.Series, volume: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Volume-weighted MACD. Same shape as MACD; uses VWMA legs."""
    fast_leg = _vwma(close, volume, fast)
    slow_leg = _vwma(close, volume, slow)
    macd_line = fast_leg - slow_leg
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Price Trend. Cumulative volume * pct-change-of-price."""
    pct = close.pct_change().fillna(0.0)
    return (volume * pct).cumsum()


def _nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Negative Volume Index — tracks price moves on lower-volume bars."""
    nvi = pd.Series(1000.0, index=close.index)
    pct = close.pct_change().fillna(0.0)
    vol_diff = volume.diff().fillna(0.0)
    for i in range(1, len(close)):
        if vol_diff.iat[i] < 0:
            nvi.iat[i] = nvi.iat[i - 1] * (1.0 + pct.iat[i])
        else:
            nvi.iat[i] = nvi.iat[i - 1]
    return nvi


def _pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index — tracks price moves on higher-volume bars."""
    pvi = pd.Series(1000.0, index=close.index)
    pct = close.pct_change().fillna(0.0)
    vol_diff = volume.diff().fillna(0.0)
    for i in range(1, len(close)):
        if vol_diff.iat[i] > 0:
            pvi.iat[i] = pvi.iat[i - 1] * (1.0 + pct.iat[i])
        else:
            pvi.iat[i] = pvi.iat[i - 1]
    return pvi


def _accelerator_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bill Williams Accelerator. AC = AO − SMA(AO, 5)."""
    median = (high + low) / 2.0
    ao = _sma(median, 5) - _sma(median, 34)
    return ao - _sma(ao, 5)


def _rvi(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """Relative Vigor Index — close-open momentum vs high-low range.

    Returns (RVI line, Signal line). Both ~[-1, +1] after settling.
    """
    co = close - open_
    hl = (high - low).replace(0, np.nan)
    # Symmetric weighted numerator/denominator (1, 2, 2, 1) per the original formula.
    weights = np.array([1.0, 2.0, 2.0, 1.0])
    num = co.rolling(4).apply(lambda x: float(np.dot(x, weights)), raw=True)
    den = hl.rolling(4).apply(lambda x: float(np.dot(x, weights)), raw=True)
    rvi = (num / den).rolling(period).mean()
    signal = rvi.rolling(4).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)
    return rvi, signal


def _adx_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, +DI, -DI as full Series."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    dm_plus = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)
    mask = dm_plus <= dm_minus
    dm_plus = dm_plus.copy()
    dm_plus[mask] = 0
    dm_minus = dm_minus.copy()
    mask2 = dm_minus <= dm_plus
    dm_minus[mask2] = 0

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx, di_plus, di_minus


# ── Neutral defaults for sanitisation (REQ-IND-006/007) ────────────────────
# Columns whose neutral fill is the mid-band of an oscillator (0–100 scale).
_FILL_50: frozenset[str] = frozenset({
    "rsi_14", "rsi_21", "stoch_k", "stoch_d",
    "mfi_14", "williams_r", "ult_osc",
    "stoch_rsi_k", "stoch_rsi_d",
})
# Columns whose neutral fill is the bar's close price (price-tracking lines).
_FILL_CLOSE: frozenset[str] = frozenset({
    "ema_9", "ema_12", "ema_21", "ema_26", "ema_50", "ema_55", "ema_200",
    "sma_20", "sma_50",
    "vwap",
    "bb_upper", "bb_mid", "bb_lower",
    "kc_upper", "kc_mid", "kc_lower",
    "donchian_upper", "donchian_lower", "donchian_mid",
    "tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou_span",
    "psar", "supertrend",
    "vwma_20", "hma_20", "alma_20",
})
# Sparse markers — NaN at non-pivot rows is the contract; downstream code
# iterates with .dropna(). Do NOT fill these.
_PRESERVE_NAN: frozenset[str] = frozenset({"swing_high", "swing_low", "_warmup"})


def sanitize_indicators(df: "pd.DataFrame") -> "pd.DataFrame":
    """REQ-IND-006: never emit NaN/Inf from the indicator engine.
    REQ-IND-007: surface warmup status when an indicator's lookback isn't met.

    Replaces +/-inf with NaN, fills remaining NaN with column-appropriate
    neutral defaults, and tags rows where any of the key long-lookback
    indicators is still warming up.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    key_warmup_cols = [c for c in ("rsi_14", "ema_21", "macd", "atr_14", "ema_200") if c in df.columns]
    if key_warmup_cols:
        df["_warmup"] = df[key_warmup_cols].isna().any(axis=1)
    else:
        df["_warmup"] = False
    close = df.get("close")
    for col in df.columns:
        if col in _PRESERVE_NAN:
            continue
        series = df[col]
        if not series.isna().any():
            continue
        if col in _FILL_50:
            df[col] = series.fillna(50.0)
        elif col in _FILL_CLOSE and close is not None:
            df[col] = series.fillna(close).fillna(0.0)
        else:
            df[col] = series.fillna(0.0)
    return df


class TechnicalIndicators:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(1.0, index=df.index))

        df["rsi_14"] = _rsi(close, 14)
        df["rsi_21"] = _rsi(close, 21)

        df["ema_9"] = _ema(close, 9)
        df["ema_21"] = _ema(close, 21)
        df["ema_50"] = _ema(close, 50)
        df["ema_55"] = _ema(close, 55)
        df["ema_200"] = _ema(close, 200)
        df["sma_20"] = _sma(close, 20)
        df["sma_50"] = _sma(close, 50)

        macd_line, macd_signal, macd_hist = _macd(close)
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
        df["bb_upper"] = bb_upper
        df["bb_mid"] = bb_mid
        df["bb_lower"] = bb_lower
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        # ARMS-V2.1: BB width percentile (200-candle rolling rank)
        df["bb_width_percentile"] = df["bb_width"].rolling(200, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False,
        )

        df["atr_14"] = _atr(high, low, close, 14)
        df["atr_pct"] = df["atr_14"] / close.replace(0, np.nan)

        # ARMS-V2.1: ATR percentile (200-candle rolling rank)
        df["atr_percentile"] = df["atr_14"].rolling(200, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False,
        )

        stoch_k, stoch_d = _stochastic(high, low, close)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d

        df["vwap"] = _vwap(high, low, close, volume)

        # ARMS-V2.1: ADX, +DI, -DI as columns
        adx_s, di_plus_s, di_minus_s = _adx_di(high, low, close)
        df["adx"] = adx_s
        df["plus_di"] = di_plus_s
        df["minus_di"] = di_minus_s

        # ARMS-V2.1: Keltner Channels
        kc_upper, kc_mid, kc_lower = _keltner_channels(close, high, low)
        df["kc_upper"] = kc_upper
        df["kc_mid"] = kc_mid
        df["kc_lower"] = kc_lower

        # ARMS-V2.1: Keltner squeeze (BB inside KC)
        df["keltner_squeeze"] = (df["bb_upper"] < kc_upper) & (df["bb_lower"] > kc_lower)

        # ARMS-V2.1: EMA200 slope (10-bar percentage change)
        df["ema200_slope"] = df["ema_200"].pct_change(10)

        # ARMS-V2.1: Volume SMA(20)
        df["volume_sma_20"] = _sma(volume, 20)
        df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

        df["returns_1"] = close.pct_change(1)
        df["returns_5"] = close.pct_change(5)
        df["returns_20"] = close.pct_change(20)

        df["vol_20"] = df["returns_1"].rolling(20).std() * (252 ** 0.5)

        df["trend_ema"] = np.where(df["ema_21"] > df["ema_50"], 1.0, -1.0)
        df["momentum_rsi"] = np.where(df["rsi_14"] > 50, 1.0, -1.0)

        # ARMS-V2.1: 50-candle high/low for breakout detection
        df["high_50"] = high.rolling(50, min_periods=10).max()
        df["low_50"] = low.rolling(50, min_periods=10).min()

        # ── V1.0 Spec: 15 additional indicators ──────────────────────────

        # SuperTrend (ATR 10, Multiplier 3)
        st_line, st_dir = _supertrend(high, low, close, period=10, multiplier=3.0)
        df["supertrend"] = st_line
        df["supertrend_dir"] = st_dir  # 1 = bullish, -1 = bearish

        # Ichimoku Cloud
        ichimoku = _ichimoku(high, low, close)
        df["tenkan_sen"] = ichimoku["tenkan_sen"]
        df["kijun_sen"] = ichimoku["kijun_sen"]
        df["senkou_a"] = ichimoku["senkou_a"]
        df["senkou_b"] = ichimoku["senkou_b"]
        df["chikou_span"] = ichimoku["chikou_span"]
        # Ichimoku cloud status: price above cloud = bullish
        cloud_top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
        cloud_bot = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)
        df["ichimoku_above_cloud"] = (close > cloud_top).astype(float)
        df["ichimoku_below_cloud"] = (close < cloud_bot).astype(float)

        # Parabolic SAR
        df["psar"] = _parabolic_sar(high, low)
        df["psar_bullish"] = (close > df["psar"]).astype(float)

        # Aroon Up/Down (25-period)
        aroon_up, aroon_down = _aroon(high, low, period=25)
        df["aroon_up"] = aroon_up
        df["aroon_down"] = aroon_down
        df["aroon_osc"] = aroon_up - aroon_down

        # Money Flow Index (14)
        df["mfi_14"] = _mfi(high, low, close, volume, period=14)

        # Commodity Channel Index (20)
        df["cci_20"] = _cci(high, low, close, period=20)

        # Williams %R (14)
        df["williams_r"] = _williams_r(high, low, close, period=14)

        # Ultimate Oscillator (7, 14, 28)
        df["ult_osc"] = _ultimate_oscillator(high, low, close)

        # TRIX (15)
        df["trix"] = _trix(close, period=15)

        # Chaikin Money Flow (20)
        df["cmf"] = _cmf(high, low, close, volume, period=20)

        # On-Balance Volume
        df["obv"] = _obv(close, volume)
        df["obv_sma_20"] = _sma(df["obv"], 20)

        # EMA 12 / 26 (explicit columns for cross detection)
        df["ema_12"] = _ema(close, 12)
        df["ema_26"] = _ema(close, 26)

        # ── Swing structure for BOS/CHoCH ─────────────────────────────────
        # 5-bar swing pivot detection
        pivot_len = 5
        df["swing_high"] = high.rolling(2 * pivot_len + 1, center=True).apply(
            lambda x: x.iloc[pivot_len] if x.iloc[pivot_len] == x.max() else np.nan, raw=False,
        )
        df["swing_low"] = low.rolling(2 * pivot_len + 1, center=True).apply(
            lambda x: x.iloc[pivot_len] if x.iloc[pivot_len] == x.min() else np.nan, raw=False,
        )

        # ── §3 Spec: Donchian Channels (20-period) ───────────────────────
        donchian_period = 20
        df["donchian_upper"] = high.rolling(donchian_period, min_periods=1).max()
        df["donchian_lower"] = low.rolling(donchian_period, min_periods=1).min()
        df["donchian_mid"] = (df["donchian_upper"] + df["donchian_lower"]) / 2.0
        df["donchian_width"] = (df["donchian_upper"] - df["donchian_lower"]) / df["donchian_mid"].replace(0, np.nan)

        # ── §3 Spec: Vortex Indicator (14-period) ────────────────────────
        vortex_period = 14
        vm_plus = (high - low.shift(1)).abs()
        vm_minus = (low - high.shift(1)).abs()
        tr_vi = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        vm_plus_sum = vm_plus.rolling(vortex_period, min_periods=1).sum()
        vm_minus_sum = vm_minus.rolling(vortex_period, min_periods=1).sum()
        tr_sum = tr_vi.rolling(vortex_period, min_periods=1).sum()
        df["vortex_plus"] = vm_plus_sum / tr_sum.replace(0, np.nan)
        df["vortex_minus"] = vm_minus_sum / tr_sum.replace(0, np.nan)
        df["vortex_diff"] = df["vortex_plus"] - df["vortex_minus"]

        # ── §3 Spec: Awesome Oscillator ──────────────────────────────────
        midpoint = (high + low) / 2.0
        df["awesome_osc"] = _sma(midpoint, 5) - _sma(midpoint, 34)

        # ── Reference-spec indicators ────────────────────────────────────
        # Volume-weighted, hull, ALMA moving averages (low-lag alternatives).
        df["vwma_20"] = _vwma(close, volume, 20)
        df["hma_20"] = _hma(close, 20)
        df["alma_20"] = _alma(close, 20)
        # Rate of Change (percentage momentum).
        df["roc_10"] = _roc(close, 10)
        df["roc_20"] = _roc(close, 20)
        # Stochastic RSI (%K, %D both on 0..100 scale).
        stoch_rsi_k, stoch_rsi_d = _stoch_rsi(close, 14, 3, 3)
        df["stoch_rsi_k"] = stoch_rsi_k
        df["stoch_rsi_d"] = stoch_rsi_d
        # Volume-weighted MACD.
        vw_m, vw_s, vw_h = _vw_macd(close, volume)
        df["vw_macd"] = vw_m
        df["vw_macd_signal"] = vw_s
        df["vw_macd_hist"] = vw_h
        # Volume Price Trend / NVI / PVI (accumulation-distribution family).
        df["vpt"] = _vpt(close, volume)
        df["nvi"] = _nvi(close, volume)
        df["pvi"] = _pvi(close, volume)
        # Accelerator Oscillator and Relative Vigor Index.
        df["accelerator_osc"] = _accelerator_oscillator(high, low)
        rvi_line, rvi_signal = _rvi(df["open"], high, low, close, 10)
        df["rvi"] = rvi_line
        df["rvi_signal"] = rvi_signal

        # REQ-IND-006/007: drop the unusable warmup head, then sanitise the
        # remainder so downstream consumers never see NaN/Inf.
        df = df.dropna(subset=["rsi_14", "ema_21", "macd"])
        return sanitize_indicators(df)
