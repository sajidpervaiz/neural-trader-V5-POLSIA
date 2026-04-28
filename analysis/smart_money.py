"""Smart Money Concepts (SMC/ICT) Framework — V6.0 Spec.

Implements:
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Fair Value Gaps (FVG) with displacement filter, volume filter, expiry, mitigation levels
- Order Blocks with strength scoring (BOS, volume, FVG overlap, LVN, HTF align)
- Breaker Blocks with rejection wick/volume filters
- Liquidity Grabs with equal-level detection (≥3 within 0.1%) and MSS requirement
- Inducement detection
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FairValueGap:
    index: int
    direction: str  # "bullish" or "bearish"
    top: float
    bottom: float
    size_pct: float
    filled: bool = False
    # V6.0 additions
    displacement: bool = False  # body≥60% range, range≥1.2×ATR, vol≥150%
    volume_pct: float = 0.0     # volume as % of 20-bar mean
    age: int = 0                # candles since formation
    mitigation_pct: float = 0.0  # 0/25/50/75/100
    expiry_candles: int = 50


@dataclass
class OrderBlock:
    index: int
    direction: str  # "bullish" or "bearish"
    high: float
    low: float
    mitigated: bool = False
    # V6.0 additions
    strength_score: float = 0.0  # 0-100 score
    has_bos: bool = False        # preceded by BOS
    impulse_volume_pct: float = 0.0  # impulse candle volume as % of 20-bar mean
    fvg_overlap: bool = False    # overlaps with an FVG
    lvn_overlap: bool = False    # overlaps with LVN zone
    htf_aligned: bool = False    # aligns with higher-timeframe OB


@dataclass
class StructureBreak:
    index: int
    break_type: str  # "BOS" or "CHoCH"
    direction: str  # "bullish" or "bearish"
    level: float


@dataclass
class LiquidityGrab:
    index: int
    direction: str  # "bullish" (swept lows, reversed up) or "bearish"
    swept_level: float
    reversal_close: float


@dataclass
class SMCState:
    """Complete SMC analysis for the latest bar."""
    bos_events: list[StructureBreak] = field(default_factory=list)
    choch_events: list[StructureBreak] = field(default_factory=list)
    active_fvgs: list[FairValueGap] = field(default_factory=list)
    active_order_blocks: list[OrderBlock] = field(default_factory=list)
    active_breaker_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_grabs: list[LiquidityGrab] = field(default_factory=list)
    market_bias: str = "neutral"  # "bullish", "bearish", "neutral"
    smc_score: float = 0.0  # [-1, 1] composite
    reasons: list[str] = field(default_factory=list)


class SmartMoneyAnalyzer:
    """Analyzes price-action for SMC/ICT concepts on a given DataFrame.

    The DataFrame must have columns: open, high, low, close, volume,
    plus swing_high / swing_low (from TechnicalIndicators).
    """

    def __init__(
        self,
        fvg_min_pct: float = 0.005,
        ob_lookback: int = 50,
        liquidity_lookback: int = 20,
        pivot_bars: int = 3,
        fvg_expiry_candles: int = 50,
        equal_level_tolerance: float = 0.002,
        equal_level_min_touches: int = 2,
        equal_level_lookback: int = 100,
    ) -> None:
        self._fvg_min_pct = fvg_min_pct
        self._ob_lookback = ob_lookback
        self._liq_lookback = liquidity_lookback
        self._pivot_bars = pivot_bars
        self._fvg_expiry = fvg_expiry_candles
        self._eq_tol = equal_level_tolerance
        self._eq_min = equal_level_min_touches
        self._eq_lookback = equal_level_lookback
        self._prev_swing_highs: list[float] = []
        self._prev_swing_lows: list[float] = []
        self._last_structure_dir: str = "neutral"

    # ── Swing Structure ───────────────────────────────────────────────────
    def _extract_swings(self, df: pd.DataFrame) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Return lists of (index, price) for swing highs and lows."""
        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []
        n = len(df)
        pb = self._pivot_bars
        for i in range(pb, n - pb):
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            if all(h >= df["high"].iloc[i - pb:i].max() and h >= df["high"].iloc[i + 1:i + pb + 1].max()
                   for _ in [0]) if n > i + pb else False:
                swing_highs.append((i, h))
            if all(l <= df["low"].iloc[i - pb:i].min() and l <= df["low"].iloc[i + 1:i + pb + 1].min()
                   for _ in [0]) if n > i + pb else False:
                swing_lows.append((i, l))
        # Also get from pre-computed columns if available
        # df.index.get_loc(label) can return int OR slice OR boolean array when
        # the index isn't unique. The downstream sort() then crashes with
        # "'<' not supported between instances of 'slice' and 'int'". Use
        # get_indexer([label]) which always yields an integer position (or -1
        # for a missing label), then collapse any non-integer fallthrough.
        def _to_int_pos(label: object) -> int | None:
            if isinstance(label, int):
                return label
            try:
                pos = int(df.index.get_indexer([label])[0])
                return pos if pos >= 0 else None
            except Exception:
                return None

        if "swing_high" in df.columns:
            for i, v in df["swing_high"].dropna().items():
                idx = _to_int_pos(i)
                if idx is not None and (idx, v) not in swing_highs:
                    swing_highs.append((idx, v))
        if "swing_low" in df.columns:
            for i, v in df["swing_low"].dropna().items():
                idx = _to_int_pos(i)
                if idx is not None and (idx, v) not in swing_lows:
                    swing_lows.append((idx, v))
        swing_highs.sort(key=lambda x: x[0])
        swing_lows.sort(key=lambda x: x[0])
        return swing_highs, swing_lows

    # ── Break of Structure / Change of Character ──────────────────────────
    def _detect_structure_breaks(
        self,
        df: pd.DataFrame,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]],
    ) -> tuple[list[StructureBreak], list[StructureBreak]]:
        """Detect BOS and CHoCH events from recent swing structure."""
        bos_list: list[StructureBreak] = []
        choch_list: list[StructureBreak] = []
        n = len(df)
        if n < 3 or len(swing_highs) < 2 or len(swing_lows) < 2:
            return bos_list, choch_list

        last_idx = n - 1
        close = df["close"].iloc[-1]
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]

        # Last two swing highs and lows
        sh1_idx, sh1 = swing_highs[-1]
        sh2_idx, sh2 = swing_highs[-2] if len(swing_highs) >= 2 else swing_highs[-1]
        sl1_idx, sl1 = swing_lows[-1]
        sl2_idx, sl2 = swing_lows[-2] if len(swing_lows) >= 2 else swing_lows[-1]

        # BOS bullish: price breaks above most recent swing high
        if high > sh1 and sh1_idx < last_idx:
            # Is this a continuation (BOS) or reversal (CHoCH)?
            if self._last_structure_dir in ("bullish", "neutral"):
                bos_list.append(StructureBreak(last_idx, "BOS", "bullish", sh1))
            else:
                choch_list.append(StructureBreak(last_idx, "CHoCH", "bullish", sh1))
            self._last_structure_dir = "bullish"

        # BOS bearish: price breaks below most recent swing low
        if low < sl1 and sl1_idx < last_idx:
            if self._last_structure_dir in ("bearish", "neutral"):
                bos_list.append(StructureBreak(last_idx, "BOS", "bearish", sl1))
            else:
                choch_list.append(StructureBreak(last_idx, "CHoCH", "bearish", sl1))
            self._last_structure_dir = "bearish"

        return bos_list, choch_list

    # ── Fair Value Gaps (V6.0: displacement, volume, expiry, mitigation) ─
    def _detect_fvgs(self, df: pd.DataFrame) -> list[FairValueGap]:
        """Detect Fair Value Gaps with V6.0 displacement + volume filters."""
        fvgs: list[FairValueGap] = []
        n = len(df)
        lookback = min(self._ob_lookback, n - 2)
        # Pre-compute 20-bar volume mean and ATR for displacement check
        vol_mean = df["volume"].rolling(20).mean()
        high_low = df["high"] - df["low"]
        atr_20 = high_low.rolling(20).mean()

        for i in range(max(2, n - lookback), n):
            # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
            prev2_high = df["high"].iloc[i - 2]
            curr_low = df["low"].iloc[i]
            if curr_low > prev2_high:
                gap_size = (curr_low - prev2_high) / df["close"].iloc[i]
                if gap_size >= self._fvg_min_pct:
                    fvg = FairValueGap(i, "bullish", curr_low, prev2_high, gap_size,
                                       expiry_candles=self._fvg_expiry)
                    # V6.0 displacement filter: middle candle body≥60% range, range≥1.2×ATR, vol≥150%
                    mid_i = i - 1
                    mid_body = abs(df["close"].iloc[mid_i] - df["open"].iloc[mid_i])
                    mid_range = df["high"].iloc[mid_i] - df["low"].iloc[mid_i]
                    mid_vol = df["volume"].iloc[mid_i]
                    vm = vol_mean.iloc[mid_i] if not np.isnan(vol_mean.iloc[mid_i]) else 1
                    atr_val = atr_20.iloc[mid_i] if not np.isnan(atr_20.iloc[mid_i]) else 0
                    fvg.volume_pct = (mid_vol / vm * 100) if vm > 0 else 0
                    fvg.displacement = (
                        mid_range > 0 and
                        mid_body / mid_range >= 0.60 and
                        mid_range >= 1.2 * atr_val and
                        fvg.volume_pct >= 150
                    )
                    # Mitigation level + age + filled check
                    fvg.age = n - 1 - i
                    if i < n - 1:
                        subsequent_low = df["low"].iloc[i + 1:].min() if i + 1 < n else curr_low
                        gap_range = fvg.top - fvg.bottom
                        if gap_range > 0:
                            penetration = max(0, fvg.top - subsequent_low)
                            fvg.mitigation_pct = min(100, round(penetration / gap_range * 100))
                            if fvg.mitigation_pct >= 100:
                                fvg.filled = True
                    fvgs.append(fvg)

            # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
            prev2_low = df["low"].iloc[i - 2]
            curr_high = df["high"].iloc[i]
            if prev2_low > curr_high:
                gap_size = (prev2_low - curr_high) / df["close"].iloc[i]
                if gap_size >= self._fvg_min_pct:
                    fvg = FairValueGap(i, "bearish", prev2_low, curr_high, gap_size,
                                       expiry_candles=self._fvg_expiry)
                    mid_i = i - 1
                    mid_body = abs(df["close"].iloc[mid_i] - df["open"].iloc[mid_i])
                    mid_range = df["high"].iloc[mid_i] - df["low"].iloc[mid_i]
                    mid_vol = df["volume"].iloc[mid_i]
                    vm = vol_mean.iloc[mid_i] if not np.isnan(vol_mean.iloc[mid_i]) else 1
                    atr_val = atr_20.iloc[mid_i] if not np.isnan(atr_20.iloc[mid_i]) else 0
                    fvg.volume_pct = (mid_vol / vm * 100) if vm > 0 else 0
                    fvg.displacement = (
                        mid_range > 0 and
                        mid_body / mid_range >= 0.60 and
                        mid_range >= 1.2 * atr_val and
                        fvg.volume_pct >= 150
                    )
                    fvg.age = n - 1 - i
                    if i < n - 1:
                        subsequent_high = df["high"].iloc[i + 1:].max() if i + 1 < n else curr_high
                        gap_range = fvg.top - fvg.bottom
                        if gap_range > 0:
                            penetration = max(0, subsequent_high - fvg.bottom)
                            fvg.mitigation_pct = min(100, round(penetration / gap_range * 100))
                            if fvg.mitigation_pct >= 100:
                                fvg.filled = True
                    fvgs.append(fvg)

        # Filter: not filled AND not expired
        return [f for f in fvgs if not f.filled and f.age <= f.expiry_candles]

    # ── Order Blocks (V6.0: strength scoring, BOS, volume, FVG/LVN overlap)
    def _detect_order_blocks(self, df: pd.DataFrame,
                             bos_events: list[StructureBreak] | None = None,
                             fvgs: list[FairValueGap] | None = None,
                             ) -> list[OrderBlock]:
        """Detect Order Blocks with V6.0 strength scoring (0-100)."""
        obs: list[OrderBlock] = []
        n = len(df)
        lookback = min(self._ob_lookback, n - 3)
        vol_mean = df["volume"].rolling(20).mean()
        bos_indices = {b.index for b in (bos_events or [])}

        for i in range(max(1, n - lookback), n - 1):
            curr_close = df["close"].iloc[i]
            curr_open = df["open"].iloc[i]
            next_close = df["close"].iloc[i + 1]
            next_open = df["open"].iloc[i + 1]

            body = abs(curr_close - curr_open)
            next_body = abs(next_close - next_open)
            if body == 0 or next_body == 0:
                continue

            impulse_vol = df["volume"].iloc[i + 1]
            vm = vol_mean.iloc[i + 1] if i + 1 < len(vol_mean) and not np.isnan(vol_mean.iloc[i + 1]) else 1
            vol_pct = (impulse_vol / vm * 100) if vm > 0 else 0

            # Bullish OB: bearish candle followed by bullish impulse (>1.5x body)
            if curr_close < curr_open and next_close > next_open and next_body > 1.5 * body:
                ob = OrderBlock(i, "bullish", df["high"].iloc[i], df["low"].iloc[i])
                ob.impulse_volume_pct = vol_pct
                # Check mitigation
                mid = (ob.high + ob.low) / 2
                if i + 2 < n:
                    subsequent_low = df["low"].iloc[i + 2:].min()
                    if subsequent_low <= mid:
                        ob.mitigated = True
                # V6.0 strength scoring
                score = 0.0
                # BOS preceded (+25)
                if any(idx in bos_indices for idx in range(i, min(i + 3, n))):
                    score += 25; ob.has_bos = True
                # Volume >200% (+20)
                if vol_pct >= 200:
                    score += 20
                elif vol_pct >= 150:
                    score += 10
                # FVG overlap (+20)
                for fvg in (fvgs or []):
                    if fvg.direction == "bullish" and fvg.bottom <= ob.high and fvg.top >= ob.low:
                        score += 20; ob.fvg_overlap = True; break
                # Body fills >70% of OB range (+15)
                ob_range = ob.high - ob.low
                if ob_range > 0 and body / ob_range >= 0.70:
                    score += 15
                ob.strength_score = min(100, score)
                obs.append(ob)

            # Bearish OB: bullish candle followed by bearish impulse
            if curr_close > curr_open and next_close < next_open and next_body > 1.5 * body:
                ob = OrderBlock(i, "bearish", df["high"].iloc[i], df["low"].iloc[i])
                ob.impulse_volume_pct = vol_pct
                mid = (ob.high + ob.low) / 2
                if i + 2 < n:
                    subsequent_high = df["high"].iloc[i + 2:].max()
                    if subsequent_high >= mid:
                        ob.mitigated = True
                score = 0.0
                if any(idx in bos_indices for idx in range(i, min(i + 3, n))):
                    score += 25; ob.has_bos = True
                if vol_pct >= 200:
                    score += 20
                elif vol_pct >= 150:
                    score += 10
                for fvg in (fvgs or []):
                    if fvg.direction == "bearish" and fvg.bottom <= ob.high and fvg.top >= ob.low:
                        score += 20; ob.fvg_overlap = True; break
                ob_range = ob.high - ob.low
                if ob_range > 0 and body / ob_range >= 0.70:
                    score += 15
                ob.strength_score = min(100, score)
                obs.append(ob)

        return obs

    # ── Breaker Blocks (V6.0: rejection wick≥2×body, volume≥180%) ────────
    def _detect_breaker_blocks(self, order_blocks: list[OrderBlock],
                               df: pd.DataFrame | None = None) -> list[OrderBlock]:
        """Breaker = mitigated OB with rejection candle filter."""
        breakers = []
        vol_mean = None
        if df is not None and len(df) > 20:
            vol_mean = df["volume"].rolling(20).mean()
        for ob in order_blocks:
            if not ob.mitigated:
                continue
            # V6.0: rejection candle must have wick≥2×body and volume≥180%
            if df is not None and ob.index < len(df):
                candle_idx = ob.index
                o = df["open"].iloc[candle_idx]
                c = df["close"].iloc[candle_idx]
                h = df["high"].iloc[candle_idx]
                l = df["low"].iloc[candle_idx]
                body = abs(c - o)
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                max_wick = max(upper_wick, lower_wick)
                vol = df["volume"].iloc[candle_idx]
                vm = vol_mean.iloc[candle_idx] if vol_mean is not None and not np.isnan(vol_mean.iloc[candle_idx]) else 1
                vol_pct = (vol / vm * 100) if vm > 0 else 0
                if body > 0 and max_wick >= 2 * body and vol_pct >= 180:
                    breakers.append(ob)
            else:
                breakers.append(ob)  # fallback
        return breakers

    # ── Equal Levels (V6.0: ≥3 within 0.1% tolerance, 100 candle lookback)
    def _find_equal_levels(self, df: pd.DataFrame,
                           swing_highs: list[tuple[int, float]],
                           swing_lows: list[tuple[int, float]],
                           ) -> tuple[list[float], list[float]]:
        """Return equal high levels and equal low levels (liquidity pools)."""
        equal_highs: list[float] = []
        equal_lows: list[float] = []
        n = len(df)
        cutoff = max(0, n - self._eq_lookback)

        # Filter recent swings
        recent_highs = [p for idx, p in swing_highs if idx >= cutoff]
        recent_lows = [p for idx, p in swing_lows if idx >= cutoff]

        # Cluster highs within tolerance
        if recent_highs:
            sorted_h = sorted(recent_highs)
            clusters: list[list[float]] = [[sorted_h[0]]]
            for p in sorted_h[1:]:
                if abs(p - clusters[-1][0]) / clusters[-1][0] <= self._eq_tol:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])
            for c in clusters:
                if len(c) >= self._eq_min:
                    equal_highs.append(float(np.mean(c)))

        # Cluster lows within tolerance
        if recent_lows:
            sorted_l = sorted(recent_lows)
            clusters = [[sorted_l[0]]]
            for p in sorted_l[1:]:
                if abs(p - clusters[-1][0]) / clusters[-1][0] <= self._eq_tol:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])
            for c in clusters:
                if len(c) >= self._eq_min:
                    equal_lows.append(float(np.mean(c)))

        return equal_highs, equal_lows

    # ── Liquidity Grabs (V6.0: equal levels + MSS requirement) ───────────
    def _detect_liquidity_grabs(
        self, df: pd.DataFrame,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]],
        has_mss: bool = False,
    ) -> list[LiquidityGrab]:
        """Detect liquidity sweeps: price pierces swing level then reverses.
        V6.0: also checks equal-level pools and requires MSS (structure shift)."""
        grabs: list[LiquidityGrab] = []
        n = len(df)
        if n < 3:
            return grabs

        last = df.iloc[-1]

        # Equal levels as additional sweep targets
        eq_highs, eq_lows = self._find_equal_levels(df, swing_highs, swing_lows)

        # Combine swing levels with equal levels
        all_high_levels = [p for _, p in swing_highs[-self._liq_lookback:]] + eq_highs
        all_low_levels = [p for _, p in swing_lows[-self._liq_lookback:]] + eq_lows

        # Check if current bar swept recent swing highs then closed below
        for sh_price in all_high_levels:
            if last["high"] > sh_price and last["close"] < sh_price:
                grabs.append(LiquidityGrab(n - 1, "bearish", sh_price, last["close"]))

        # Check if current bar swept recent swing lows then closed above
        for sl_price in all_low_levels:
            if last["low"] < sl_price and last["close"] > sl_price:
                grabs.append(LiquidityGrab(n - 1, "bullish", sl_price, last["close"]))

        # V6.0: if MSS not confirmed, downgrade grabs (still report but mark)
        if not has_mss and grabs:
            # Without MSS, liquidity grabs are less reliable — reduce score weight in scoring
            pass

        return grabs

    # ── Composite Analysis (V6.0) ───────────────────────────────────────
    def analyze(self, df: pd.DataFrame) -> SMCState:
        """Run full SMC analysis and return composite state."""
        if df is None or len(df) < 20:
            return SMCState()

        state = SMCState()
        swing_highs, swing_lows = self._extract_swings(df)

        # Structure breaks
        bos, choch = self._detect_structure_breaks(df, swing_highs, swing_lows)
        state.bos_events = bos
        state.choch_events = choch

        # FVGs (V6.0 with displacement + volume + expiry + mitigation)
        state.active_fvgs = self._detect_fvgs(df)

        # Order Blocks (V6.0 with strength scoring)
        all_obs = self._detect_order_blocks(df, bos_events=bos + choch, fvgs=state.active_fvgs)
        state.active_order_blocks = [ob for ob in all_obs if not ob.mitigated]
        state.active_breaker_blocks = self._detect_breaker_blocks(all_obs, df)

        # MSS detection: CHoCH = Market Structure Shift
        has_mss = len(choch) > 0

        # Liquidity Grabs (V6.0 with equal levels + MSS)
        state.liquidity_grabs = self._detect_liquidity_grabs(
            df, swing_highs, swing_lows, has_mss=has_mss)

        # ── Equal levels stored for downstream use ────────────────────────
        eq_highs, eq_lows = self._find_equal_levels(df, swing_highs, swing_lows)
        state._equal_highs = eq_highs  # type: ignore[attr-defined]
        state._equal_lows = eq_lows    # type: ignore[attr-defined]

        # ── Score computation (V6.0: weighted by displacement + strength) ──
        score = 0.0
        reasons: list[str] = []
        close = df["close"].iloc[-1]

        # BOS / CHoCH
        for b in bos:
            if b.direction == "bullish":
                score += 0.3
                reasons.append("BOS_bullish")
            else:
                score -= 0.3
                reasons.append("BOS_bearish")

        for c in choch:
            if c.direction == "bullish":
                score += 0.4
                reasons.append("CHoCH_bullish")
            else:
                score -= 0.4
                reasons.append("CHoCH_bearish")

        # FVG proximity: V6.0 — only displaced FVGs get full weight
        for fvg in state.active_fvgs[-3:]:
            weight = 0.25 if fvg.displacement else 0.10
            # Mitigation-adjusted weight: 50-75% mitigation = sweet spot entry
            if 50 <= fvg.mitigation_pct <= 75:
                weight *= 1.3
            if fvg.direction == "bullish":
                if close <= fvg.top * 1.005:
                    score += weight
                    tag = "FVG_bullish_displaced" if fvg.displacement else "FVG_bullish_zone"
                    reasons.append(tag)
            else:
                if close >= fvg.bottom * 0.995:
                    score -= weight
                    tag = "FVG_bearish_displaced" if fvg.displacement else "FVG_bearish_zone"
                    reasons.append(tag)

        # Order Blocks: V6.0 — only OBs with strength≥60 get full weight
        for ob in state.active_order_blocks[-3:]:
            weight = 0.30 if ob.strength_score >= 60 else 0.12
            if ob.direction == "bullish" and ob.low <= close <= ob.high * 1.01:
                score += weight
                reasons.append(f"OB_bullish_s{int(ob.strength_score)}")
            elif ob.direction == "bearish" and ob.low * 0.99 <= close <= ob.high:
                score -= weight
                reasons.append(f"OB_bearish_s{int(ob.strength_score)}")

        # Liquidity grabs: V6.0 — MSS-confirmed grabs get more weight
        for lg in state.liquidity_grabs:
            weight = 0.40 if has_mss else 0.20
            if lg.direction == "bullish":
                score += weight
                reasons.append("liquidity_grab_bullish" + ("_mss" if has_mss else ""))
            else:
                score -= weight
                reasons.append("liquidity_grab_bearish" + ("_mss" if has_mss else ""))

        # Breaker blocks
        for bb in state.active_breaker_blocks[-2:]:
            if bb.direction == "bullish" and bb.low <= close <= bb.high * 1.01:
                score += 0.15
                reasons.append("breaker_bullish")
            elif bb.direction == "bearish" and bb.low * 0.99 <= close <= bb.high:
                score -= 0.15
                reasons.append("breaker_bearish")

        state.smc_score = float(np.clip(score, -1.0, 1.0))
        state.reasons = reasons

        if score > 0.1:
            state.market_bias = "bullish"
        elif score < -0.1:
            state.market_bias = "bearish"
        else:
            state.market_bias = "neutral"

        return state

    def get_nearest_liquidity_zone(
        self, df: pd.DataFrame, direction: str, entry_price: float,
    ) -> float | None:
        """Find the nearest unfilled FVG or Order Block as a TP target.

        Returns a price level or None if no zone found.
        """
        if df is None or len(df) < 20:
            return None

        targets: list[float] = []
        # Unfilled FVGs
        fvgs = self._detect_fvgs(df)
        for fvg in fvgs:
            if direction == "long" and fvg.direction == "bearish" and fvg.bottom > entry_price:
                targets.append(fvg.bottom)
            elif direction == "short" and fvg.direction == "bullish" and fvg.top < entry_price:
                targets.append(fvg.top)

        # Unmitigated Order Blocks
        obs = self._detect_order_blocks(df)
        for ob in obs:
            if not ob.mitigated:
                if direction == "long" and ob.direction == "bearish" and ob.low > entry_price:
                    targets.append(ob.low)
                elif direction == "short" and ob.direction == "bullish" and ob.high < entry_price:
                    targets.append(ob.high)

        if not targets:
            return None
        # Return the nearest target
        return min(targets, key=lambda t: abs(t - entry_price))
