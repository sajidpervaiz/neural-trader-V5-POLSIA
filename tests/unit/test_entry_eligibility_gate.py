from engine.entry_eligibility import DEFAULT_ENTRY_LAYERS, EntryEligibilityGate


def _passing_layers() -> dict[str, str]:
    return {layer: "PASS" for layer in DEFAULT_ENTRY_LAYERS}


def test_entry_gate_allows_clean_candidate() -> None:
    gate = EntryEligibilityGate()

    receipt = gate.evaluate(
        mode="paper",
        layers=_passing_layers(),
        quality={"total": 72},
        thresholds={"quality": 65},
        risk={"trading_state": "ACTIVE", "can_open_new_positions": True},
        ai={"effective_mode": "advisory", "approved": True},
    )

    assert receipt.allowed is True
    assert receipt.decision == "ENTRY_ALLOWED"
    assert receipt.blockers == []


def test_entry_gate_blocks_no_trade_quality() -> None:
    gate = EntryEligibilityGate()

    receipt = gate.evaluate(
        mode="paper",
        layers=_passing_layers(),
        quality={"total": 29},
        thresholds={"quality": 30},
        risk={"trading_state": "ACTIVE", "can_open_new_positions": True},
        ai={"effective_mode": "advisory", "approved": True},
    )

    assert receipt.allowed is False
    assert receipt.blockers[0] == "quality:29<30"


def test_entry_gate_blocks_reducing_risk_state() -> None:
    gate = EntryEligibilityGate()

    receipt = gate.evaluate(
        mode="paper",
        layers=_passing_layers(),
        quality={"total": 85},
        thresholds={"quality": 30},
        risk={"trading_state": "REDUCING", "can_open_new_positions": False},
        ai={"effective_mode": "advisory", "approved": True},
    )

    assert receipt.allowed is False
    assert "risk:trading_state:REDUCING" in receipt.blockers
    assert "risk:cannot_open_new_positions" in receipt.blockers


def test_entry_gate_blocks_ai_direct_rejection() -> None:
    gate = EntryEligibilityGate()

    receipt = gate.evaluate(
        mode="paper",
        layers=_passing_layers(),
        quality={"total": 85},
        thresholds={"quality": 30},
        risk={"trading_state": "ACTIVE", "can_open_new_positions": True},
        ai={"effective_mode": "direct", "approved": False},
    )

    assert receipt.allowed is False
    assert "ai:direct_rejected" in receipt.blockers
