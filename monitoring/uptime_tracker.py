"""Burn-in / uptime tracker — REQ AC-001.

Tracks paper-mode burn-in evidence so we can prove "7 consecutive days
without crash" (the SRS acceptance criterion). Persists session history
to data/uptime.json. Cheap heartbeat (every 30s) is the crash detector:
if the bot starts and the previous heartbeat is fresh, the previous
session crashed before clean shutdown.

State file shape:
    {
      "current_session": {"start_ts": ..., "last_heartbeat_ts": ...,
                          "clean_shutdown": false, "paper_mode": true},
      "history": [
          {"start_ts": ..., "end_ts": ..., "duration_s": ...,
           "clean_shutdown": true | false, "paper_mode": true | false},
          ...
      ]
    }

History capped at 200 entries (~6 months at one session/day).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


_DEFAULT_PATH = Path("data/uptime.json")
_HEARTBEAT_SECONDS = 30.0
_CRASH_GRACE_SECONDS = 90.0   # if prior heartbeat < this old, treat as crash
_HISTORY_MAX = 200


class UptimeTracker:
    def __init__(
        self,
        path: Path | str = _DEFAULT_PATH,
        heartbeat_seconds: float = _HEARTBEAT_SECONDS,
        paper_mode: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._interval = float(heartbeat_seconds)
        self._paper_mode = bool(paper_mode)
        self._running = False
        self._task: asyncio.Task | None = None
        self._state = self._load()
        self._open_session()
        self._save()

    # ── persistence ─────────────────────────────────────────────────────
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"current_session": None, "history": []}
        try:
            return json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("UptimeTracker: failed to load {} — starting fresh ({})", self.path, exc)
            return {"current_session": None, "history": []}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._state, indent=2))
        except Exception as exc:
            logger.warning("UptimeTracker: save failed: {}", exc)

    # ── lifecycle ───────────────────────────────────────────────────────
    def _open_session(self) -> None:
        now = time.time()
        prev = self._state.get("current_session")
        if prev is not None:
            # Previous session never finalised → either crash or unclean exit.
            last_hb = float(prev.get("last_heartbeat_ts", prev.get("start_ts", 0)) or 0)
            duration = max(0.0, last_hb - float(prev.get("start_ts", last_hb) or 0))
            crashed = (now - last_hb) <= _CRASH_GRACE_SECONDS or duration > 0
            entry = {
                "start_ts": prev.get("start_ts", last_hb),
                "end_ts": last_hb,
                "duration_s": round(duration, 2),
                "clean_shutdown": False,
                "paper_mode": prev.get("paper_mode", True),
                "crashed": bool(crashed),
            }
            history = self._state.setdefault("history", [])
            history.append(entry)
            self._state["history"] = history[-_HISTORY_MAX:]
            logger.warning(
                "UptimeTracker: previous session ended uncleanly "
                "(duration={:.0f}s, last heartbeat {:.0f}s ago)",
                duration, now - last_hb,
            )
        self._state["current_session"] = {
            "start_ts": now,
            "last_heartbeat_ts": now,
            "clean_shutdown": False,
            "paper_mode": self._paper_mode,
        }

    def heartbeat(self) -> None:
        cs = self._state.get("current_session")
        if cs is None:
            return
        cs["last_heartbeat_ts"] = time.time()
        self._save()

    def close_session(self, *, clean: bool = True) -> None:
        cs = self._state.get("current_session")
        if cs is None:
            return
        now = time.time()
        history = self._state.setdefault("history", [])
        history.append({
            "start_ts": cs["start_ts"],
            "end_ts": now,
            "duration_s": round(now - float(cs["start_ts"] or now), 2),
            "clean_shutdown": bool(clean),
            "paper_mode": bool(cs.get("paper_mode", True)),
            "crashed": False,
        })
        self._state["history"] = history[-_HISTORY_MAX:]
        self._state["current_session"] = None
        self._save()

    async def run(self) -> None:
        """Background heartbeat loop — call from main.py task list."""
        self._running = True
        logger.info("UptimeTracker: heartbeat loop started (every {:.0f}s)", self._interval)
        while self._running:
            try:
                self.heartbeat()
            except Exception as exc:
                logger.debug("UptimeTracker heartbeat error: {}", exc)
            await asyncio.sleep(self._interval)

    def stop(self, *, clean: bool = True) -> None:
        self._running = False
        self.close_session(clean=clean)

    # ── reporting ───────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """REQ AC-001 evidence: session uptime, crash count, days since last crash."""
        now = time.time()
        cs = self._state.get("current_session") or {}
        session_start = float(cs.get("start_ts", now) or now)
        session_uptime = round(now - session_start, 1)

        history = list(self._state.get("history", []) or [])
        total = len(history)
        crashes = sum(1 for h in history if not h.get("clean_shutdown", True))
        last_crash_ts = max(
            (float(h.get("end_ts", 0) or 0) for h in history if not h.get("clean_shutdown", True)),
            default=0.0,
        )
        days_since_crash = (now - last_crash_ts) / 86400.0 if last_crash_ts > 0 else None
        # Burn-in coverage: continuous time since last crash + current session
        if last_crash_ts > 0:
            burn_in_seconds = (now - last_crash_ts)
        else:
            # No recorded crash — count from oldest session start, or this one.
            oldest_start = min(
                (float(h.get("start_ts", now) or now) for h in history),
                default=session_start,
            )
            burn_in_seconds = now - oldest_start

        return {
            "current_session": {
                "start_ts": session_start,
                "uptime_seconds": session_uptime,
                "uptime_hours": round(session_uptime / 3600.0, 2),
                "paper_mode": bool(cs.get("paper_mode", self._paper_mode)),
            },
            "history": {
                "sessions_total": total,
                "clean_shutdowns": total - crashes,
                "crashes": crashes,
                "last_crash_ts": last_crash_ts if last_crash_ts > 0 else None,
                "days_since_last_crash": (round(days_since_crash, 3) if days_since_crash is not None else None),
            },
            "burn_in": {
                "seconds": round(burn_in_seconds, 1),
                "hours": round(burn_in_seconds / 3600.0, 2),
                "days": round(burn_in_seconds / 86400.0, 3),
                "ac001_target_days": 7.0,
                "ac001_progress_pct": round(min(100.0, burn_in_seconds / (7 * 86400.0) * 100.0), 1),
                "ac001_passed": burn_in_seconds >= 7 * 86400.0,
            },
        }


__all__ = ["UptimeTracker"]
