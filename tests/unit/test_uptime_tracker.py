"""REQ AC-001: burn-in tracker captures session uptime + crash detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoring.uptime_tracker import UptimeTracker


def test_first_session_creates_state_file(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    UptimeTracker(path=p, heartbeat_seconds=60.0, paper_mode=True)
    state = json.loads(p.read_text())
    assert state["current_session"] is not None
    assert state["current_session"]["paper_mode"] is True
    assert state["history"] == []


def test_clean_shutdown_appends_history(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    t = UptimeTracker(path=p, paper_mode=True)
    t.close_session(clean=True)
    state = json.loads(p.read_text())
    assert state["current_session"] is None
    assert len(state["history"]) == 1
    assert state["history"][0]["clean_shutdown"] is True


def test_unclean_prior_session_recorded_as_crash(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    # Seed a "prior session that didn't finalise"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "current_session": {
            "start_ts": 1000.0,
            "last_heartbeat_ts": 1500.0,
            "clean_shutdown": False,
            "paper_mode": True,
        },
        "history": [],
    }))
    UptimeTracker(path=p, paper_mode=True)
    state = json.loads(p.read_text())
    # The prior unclean session moved into history.
    assert len(state["history"]) == 1
    assert state["history"][0]["clean_shutdown"] is False
    assert state["current_session"] is not None


def test_snapshot_shape(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    t = UptimeTracker(path=p, paper_mode=True)
    snap = t.snapshot()
    assert "current_session" in snap and "burn_in" in snap and "history" in snap
    assert snap["burn_in"]["ac001_target_days"] == 7.0
    assert 0.0 <= snap["burn_in"]["ac001_progress_pct"] <= 100.0
    assert isinstance(snap["burn_in"]["ac001_passed"], bool)


def test_heartbeat_persists_timestamp(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    t = UptimeTracker(path=p)
    state_before = json.loads(p.read_text())
    initial_hb = state_before["current_session"]["last_heartbeat_ts"]
    t.heartbeat()
    state_after = json.loads(p.read_text())
    assert state_after["current_session"]["last_heartbeat_ts"] >= initial_hb


def test_burn_in_progress_with_recorded_crash(tmp_path: Path) -> None:
    p = tmp_path / "uptime.json"
    import time as _time
    # Seed: a clean session a long time ago, then a crash 2 days ago, then clean.
    two_days_ago = _time.time() - 2 * 86400
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "current_session": None,
        "history": [
            {"start_ts": two_days_ago - 100, "end_ts": two_days_ago,
             "duration_s": 100.0, "clean_shutdown": False, "paper_mode": True},
        ],
    }))
    t = UptimeTracker(path=p)
    snap = t.snapshot()
    # Burn-in is roughly 2 days
    assert 1.5 < snap["burn_in"]["days"] < 2.5
    assert snap["burn_in"]["ac001_passed"] is False
    assert snap["history"]["crashes"] >= 1
