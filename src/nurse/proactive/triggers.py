"""Trigger sources for proactive engagement.

Each trigger inspects some state and, if it is "due", returns an Engagement describing
what Aria should raise. The scheduler applies quiet-hours and priority across them.

Triggers are pure with respect to the clock: `due(now)` takes the current time so they
can be unit-tested with a fake clock and fixture files — no audio, no real time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from nurse.config import get_config, resolve

logger = logging.getLogger(__name__)


@dataclass
class Engagement:
    """One thing Aria wants to proactively raise with the patient."""
    kind: str                       # "reminder" | "check_in" | "follow_up" | "vitals_alert"
    detail: str = ""                # filled into the persona template ({detail})
    overrides_quiet_hours: bool = False
    priority: int = 0               # higher wins when several fire on the same tick
    on_fire: Any = field(default=None, repr=False)  # optional callback after engaging


class Trigger(Protocol):
    def due(self, now: datetime) -> Engagement | None: ...


def _profiles_dir() -> Path:
    return resolve("data/patient_profiles")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed line in %s", path.name)
    return out


# ── Reminders ───────────────────────────────────────────────────────────────────

class DueReminders:
    """Fires medication/appointment reminders written by the set_reminder tool.

    Consumes {patient}_reminders.jsonl (written but never read until now). A reminder
    with an HH:MM scheduled_time whose time has passed today and that is not yet
    acknowledged fires once; firing marks it acknowledged so it won't repeat.
    Relative times like "in 2 hours" are not deterministically schedulable here and
    are skipped (left for the conversational path).
    """
    overrides_quiet_hours = True

    def __init__(self, patient_id: str) -> None:
        self.path = _profiles_dir() / f"{patient_id}_reminders.jsonl"

    def due(self, now: datetime) -> Engagement | None:
        reminders = _read_jsonl(self.path)
        changed = False
        fired: Engagement | None = None
        for r in reminders:
            if r.get("acknowledged"):
                continue
            sched = str(r.get("scheduled_time", ""))
            when = self._parse_hhmm(sched, now)
            if when is None or now < when:
                continue
            # Due. Mark acknowledged immediately so it fires exactly once.
            r["acknowledged"] = True
            changed = True
            fired = Engagement(
                kind="reminder",
                detail=r.get("reason", "reminder"),
                overrides_quiet_hours=True,
                priority=80,
            )
            break  # one reminder per tick
        if changed:
            self.path.write_text("\n".join(json.dumps(r) for r in reminders) + "\n")
        return fired

    @staticmethod
    def _parse_hhmm(sched: str, now: datetime) -> datetime | None:
        try:
            h, m = sched.split(":")
            return now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        except (ValueError, AttributeError):
            return None


# ── Interval check-in ─────────────────────────────────────────────────────────

class IntervalCheckIn:
    """Fires a gentle check-in when enough time has passed since the last interaction.

    `last_interaction()` is a callable returning the most recent datetime Aria and the
    patient interacted (reactive turn or prior engagement), or None if never.
    """
    overrides_quiet_hours = False

    def __init__(self, last_interaction) -> None:
        self._last_interaction = last_interaction
        self.interval = timedelta(
            minutes=get_config()["proactive"]["check_in_interval_minutes"]
        )

    def due(self, now: datetime) -> Engagement | None:
        last = self._last_interaction()
        if last is None or (now - last) >= self.interval:
            return Engagement(kind="check_in", priority=20)
        return None


# ── Memory follow-up ────────────────────────────────────────────────────────────

class MemoryFollowUp:
    """Follows up on the last session's topic, at most N times per day.

    `long_term` is a LongTermMemory; `latest()` supplies the topic. `last_interaction`
    is a callable returning the most recent interaction time. The follow-up only fires
    after a stretch of no interaction (the same idle gap as a check-in) so it never
    interrupts right after the greeting or mid-conversation. Dedup state is in-memory
    (last fire date) — good enough for a single always-on process.
    """
    overrides_quiet_hours = False

    def __init__(self, long_term, last_interaction) -> None:
        self.long_term = long_term
        self._last_interaction = last_interaction
        self.max_per_day = get_config()["proactive"]["follow_up_max_per_day"]
        self.idle = timedelta(
            minutes=get_config()["proactive"]["check_in_interval_minutes"]
        )
        self._fires_on: dict[str, int] = {}  # date iso -> count

    def due(self, now: datetime) -> Engagement | None:
        # Don't follow up until the patient has been idle for a while.
        last = self._last_interaction()
        if last is not None and (now - last) < self.idle:
            return None
        day = now.date().isoformat()
        if self._fires_on.get(day, 0) >= self.max_per_day:
            return None
        latest = self.long_term.latest()
        if not latest:
            return None
        topic = (latest.get("clinical") or "").strip()
        if not topic:
            return None

        def _mark() -> None:
            self._fires_on[day] = self._fires_on.get(day, 0) + 1

        return Engagement(kind="follow_up", detail=topic, priority=10, on_fire=_mark)


# ── Vitals threshold ──────────────────────────────────────────────────────────

class VitalsThreshold:
    """Fires when the newest reading in the vitals feed is outside its configured band.

    Reads {patient}_vitals_feed.jsonl — lines of {"type","value","timestamp"} appended
    by an external sensor/process (stubbed for now). Only the most recent reading per
    type is evaluated; a crossing fires an immediate, quiet-hours-overriding alert. A
    per-type marker prevents re-firing on the same reading timestamp.
    """
    overrides_quiet_hours = True

    def __init__(self, patient_id: str) -> None:
        self.path = _profiles_dir() / f"{patient_id}_vitals_feed.jsonl"
        self.thresholds = get_config()["proactive"]["vitals_thresholds"]
        self._last_seen: dict[str, str] = {}  # type -> timestamp already alerted

    def due(self, now: datetime) -> Engagement | None:
        readings = _read_jsonl(self.path)
        if not readings:
            return None
        # Most recent reading per type.
        latest_by_type: dict[str, dict] = {}
        for r in readings:
            t = r.get("type")
            if t:
                latest_by_type[t] = r

        for vtype, bounds in self.thresholds.items():
            r = latest_by_type.get(vtype)
            if not r:
                continue
            ts = str(r.get("timestamp", ""))
            if self._last_seen.get(vtype) == ts:
                continue  # already alerted on this exact reading
            value = self._to_float(r.get("value"))
            if value is None:
                continue
            if value < bounds["min"] or value > bounds["max"]:
                self._last_seen[vtype] = ts
                pretty = vtype.replace("_", " ")
                return Engagement(
                    kind="vitals_alert",
                    detail=f"{pretty} reading of {r.get('value')} is outside the normal range",
                    overrides_quiet_hours=True,
                    priority=100,  # highest — clinical
                )
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            # Tolerate values like "118" or "118 bpm" or "98.6 F"
            return float(str(value).split()[0])
        except (ValueError, IndexError):
            return None
