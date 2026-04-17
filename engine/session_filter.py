"""
Session / Killzone filter for NeuralTrader V5.

ICT Killzones (UTC):
  Asian Open     : 00:00 – 03:00
  London Open    : 07:00 – 10:00  ← highest probability
  New York Open  : 13:00 – 16:00  ← highest probability
  London Close   : 15:00 – 17:00
  London-NY Overlap: 13:00 – 16:00

Usage:
    sf = SessionFilter()
    if sf.should_trade():
        ...
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KillzoneInfo:
    name: str
    start_hour: int   # UTC inclusive
    end_hour: int     # UTC exclusive
    priority: int     # 1=low, 2=medium, 3=high
    description: str


KILLZONES: list[KillzoneInfo] = [
    KillzoneInfo("asian_open",       0,  3,  2, "Asian session open — moderate liquidity"),
    KillzoneInfo("london_open",      7, 10,  3, "London open — prime killzone, highest prob"),
    KillzoneInfo("new_york_open",   13, 16,  3, "New York open — prime killzone, highest prob"),
    KillzoneInfo("london_close",    15, 17,  2, "London close — stop hunts, reversals"),
    KillzoneInfo("asia_london_gap",  5,  7,  1, "Pre-London — gap fill setups"),
]

# Sessions (broader windows)
SESSIONS: dict[str, tuple[int, int]] = {
    "asian":    (0, 8),    # 00:00 – 08:00 UTC
    "london":   (7, 16),   # 07:00 – 16:00 UTC
    "new_york": (13, 22),  # 13:00 – 22:00 UTC
}


class SessionFilter:
    """Determine whether the current UTC time falls within a tradeable
    killzone or session window.

    Parameters
    ----------
    min_priority : int
        Minimum killzone priority required for ``should_trade`` to return True.
        1 = trade all windows, 2 = skip low-priority, 3 = prime windows only.
    require_killzone : bool
        If True, ``should_trade`` returns False when no killzone is active.
        If False, trades are allowed at any time but killzone bonus is 0.
    """

    def __init__(
        self,
        min_priority: int = 1,
        require_killzone: bool = False,
    ) -> None:
        self._min_priority = min_priority
        self._require_killzone = require_killzone

    # ── Public API ────────────────────────────────────────────────────────

    def active_killzones(
        self, dt: Optional[datetime.datetime] = None,
    ) -> list[KillzoneInfo]:
        """Return all killzones currently active at the given UTC datetime."""
        now = _utc_now(dt)
        hour = now.hour + now.minute / 60.0
        return [kz for kz in KILLZONES if kz.start_hour <= hour < kz.end_hour]

    def active_sessions(
        self, dt: Optional[datetime.datetime] = None,
    ) -> list[str]:
        """Return names of all trading sessions currently active."""
        now = _utc_now(dt)
        hour = now.hour + now.minute / 60.0
        return [
            name for name, (start, end) in SESSIONS.items()
            if start <= hour < end
        ]

    def is_in_killzone(
        self, dt: Optional[datetime.datetime] = None,
    ) -> tuple[bool, Optional[KillzoneInfo]]:
        """Return (is_active, best_killzone) where best_killzone is the
        highest-priority active zone (or None)."""
        active = self.active_killzones(dt)
        if not active:
            return False, None
        best = max(active, key=lambda kz: kz.priority)
        return True, best

    def should_trade(
        self, dt: Optional[datetime.datetime] = None,
    ) -> tuple[bool, str]:
        """Return (allowed, reason).

        If require_killzone=True, blocks trades outside killzones.
        Respects min_priority threshold.
        """
        active = self.active_killzones(dt)

        if not active:
            if self._require_killzone:
                return False, "no_killzone_active"
            return True, "outside_killzone_permitted"

        best = max(active, key=lambda kz: kz.priority)
        if best.priority < self._min_priority:
            if self._require_killzone:
                return False, f"killzone_priority_{best.priority}_below_min_{self._min_priority}"
            return True, f"low_priority_killzone_{best.name}"

        return True, f"killzone_{best.name}_priority_{best.priority}"

    def killzone_score(
        self, dt: Optional[datetime.datetime] = None,
    ) -> float:
        """Return a 0–1 timing score for use in quality scoring.

        0.0 = no killzone active
        0.5 = low-priority killzone
        1.0 = prime killzone (London Open or NY Open)
        """
        active = self.active_killzones(dt)
        if not active:
            return 0.0
        best = max(active, key=lambda kz: kz.priority)
        return {1: 0.3, 2: 0.6, 3: 1.0}.get(best.priority, 0.0)

    def session_context(
        self, dt: Optional[datetime.datetime] = None,
    ) -> dict:
        """Return a summary dict suitable for signal metadata."""
        now = _utc_now(dt)
        active_kz = self.active_killzones(dt)
        best_kz = max(active_kz, key=lambda kz: kz.priority) if active_kz else None
        return {
            "utc_hour": now.hour,
            "active_sessions": self.active_sessions(dt),
            "active_killzones": [kz.name for kz in active_kz],
            "best_killzone": best_kz.name if best_kz else None,
            "killzone_priority": best_kz.priority if best_kz else 0,
            "killzone_score": self.killzone_score(dt),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now(dt: Optional[datetime.datetime]) -> datetime.datetime:
    if dt is not None:
        return dt if dt.tzinfo is None else dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return datetime.datetime.utcnow()
