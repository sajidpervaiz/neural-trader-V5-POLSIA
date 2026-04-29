"""REQ-SIG-010..012: layer-conflict penalty + N-candle execution block.

These tests target the SignalGenerator's conflict-streak counters directly
without spinning up the full pipeline — that path is exercised by
integration tests; here we just verify the counter / accessor contract.
"""
from __future__ import annotations

from engine.signal_generator import SignalGenerator
from core.config import Config
from core.event_bus import EventBus
from analysis.data_manager import DataManager


def _make_sg() -> SignalGenerator:
    config = Config()
    bus = EventBus()
    dm = DataManager(config, bus)
    return SignalGenerator(config, bus, dm)


def test_conflict_state_starts_clean() -> None:
    sg = _make_sg()
    state = sg.get_conflict_state()
    assert state["streak_per_symbol"] == {}
    assert state["blocked_symbols"] == {}
    assert state["max_streak"] >= 1
    assert state["penalty_per_candle"] >= 0


def test_conflict_streak_increments_and_resets() -> None:
    sg = _make_sg()
    sym = "BTC/USDT:USDT"
    sg._conflict_streak[sym] = 0

    # Simulate three consecutive evaluations with conflict
    sg._conflict_streak[sym] += 1
    sg._conflict_streak[sym] += 1
    sg._conflict_streak[sym] += 1
    assert sg.get_conflict_state()["streak_per_symbol"][sym] == 3

    # Resolution clears the streak
    sg._conflict_streak[sym] = 0
    sg._last_conflict_block_reason.pop(sym, None)
    assert sg.get_conflict_state()["streak_per_symbol"][sym] == 0
    assert sym not in sg.get_conflict_state()["blocked_symbols"]


def test_conflict_block_reason_recorded() -> None:
    sg = _make_sg()
    sym = "ETH/USDT:USDT"
    sg._conflict_streak[sym] = 5
    sg._last_conflict_block_reason[sym] = "conflict_streak=5>=3"
    snap = sg.get_conflict_state()
    assert snap["blocked_symbols"][sym] == "conflict_streak=5>=3"
