"""REQ-SIG-009: master-score bands must match spec defaults
(85 strong / 70 normal / 30 no-trade) and respect config overrides."""
from __future__ import annotations

from analysis.data_manager import DataManager
from core.config import Config
from core.event_bus import EventBus
from engine.signal_generator import SignalGenerator


def _make_sg() -> SignalGenerator:
    return SignalGenerator(Config(), EventBus(), DataManager(Config(), EventBus()))


def test_default_bands_match_spec() -> None:
    sg = _make_sg()
    bands = sg._master_score_bands()
    assert bands == {"strong": 85, "normal": 70, "no_trade": 30}


def test_band_classification_boundary_values() -> None:
    sg = _make_sg()
    # Boundary values fall into the upper band (>= comparison).
    assert sg.classify_master_score(85) == "STRONG"
    assert sg.classify_master_score(84.99) == "NORMAL"
    assert sg.classify_master_score(70) == "NORMAL"
    assert sg.classify_master_score(69.99) == "NO_TRADE"
    assert sg.classify_master_score(30) == "NO_TRADE"
    assert sg.classify_master_score(29.99) == "OPPOSITE"
    assert sg.classify_master_score(0) == "OPPOSITE"
    assert sg.classify_master_score(100) == "STRONG"


def test_band_classification_invalid_input_returns_no_trade() -> None:
    sg = _make_sg()
    assert sg.classify_master_score(None) == "NO_TRADE"  # type: ignore[arg-type]
    assert sg.classify_master_score("abc") == "NO_TRADE"  # type: ignore[arg-type]
