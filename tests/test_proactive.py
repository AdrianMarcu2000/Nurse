"""Unit tests for proactive logic — quiet hours, triggers, engage classifier, scheduler.

Pure logic only: fake clock, fixture JSONL files, no audio. Mirrors the path-insert
style of test_safety.py.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Redirect data/patient_profiles to tmp_path and clear config cache."""
    import nurse.config as cfg_module
    original_resolve = cfg_module.resolve

    def patched_resolve(rel):
        if rel.startswith("data/patient_profiles"):
            return tmp_path
        return original_resolve(rel)

    monkeypatch.setattr(cfg_module, "resolve", patched_resolve)
    # triggers.py did `from nurse.config import resolve`, binding its own reference at
    # import time — patch that name too so trigger file paths land in tmp_path.
    import nurse.proactive.triggers as trig_module
    monkeypatch.setattr(trig_module, "resolve", patched_resolve)
    return tmp_path


# ── QuietHours ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hhmm,expected", [
    ("23:30", True),    # inside wrap-around 22:00–07:00
    ("06:59", True),
    ("07:00", False),   # end is exclusive
    ("21:59", False),
    ("22:00", True),    # start is inclusive
    ("12:00", False),
])
def test_quiet_hours_wraparound(profiles, hhmm, expected):
    from nurse.proactive.quiet_hours import QuietHours
    h, m = map(int, hhmm.split(":"))
    now = datetime(2026, 5, 31, h, m)
    assert QuietHours().active(now) is expected


# ── DueReminders ──────────────────────────────────────────────────────────────

def test_due_reminder_fires_and_acknowledges(profiles):
    from nurse.proactive.triggers import DueReminders
    path = profiles / "default_reminders.jsonl"
    path.write_text(json.dumps({
        "reason": "metformin 500mg", "scheduled_time": "08:00", "acknowledged": False,
    }) + "\n")

    now = datetime(2026, 5, 31, 8, 5)
    trig = DueReminders("default")
    eng = trig.due(now)
    assert eng is not None and eng.kind == "reminder"
    assert "metformin" in eng.detail
    assert eng.overrides_quiet_hours is True
    # Marked acknowledged on disk → does not fire again.
    assert trig.due(now) is None
    assert json.loads(path.read_text().strip())["acknowledged"] is True


def test_reminder_not_yet_due(profiles):
    from nurse.proactive.triggers import DueReminders
    (profiles / "default_reminders.jsonl").write_text(json.dumps({
        "reason": "lunch", "scheduled_time": "12:00", "acknowledged": False,
    }) + "\n")
    assert DueReminders("default").due(datetime(2026, 5, 31, 8, 0)) is None


# ── IntervalCheckIn ─────────────────────────────────────────────────────────────

def test_interval_check_in(profiles):
    from nurse.proactive.triggers import IntervalCheckIn
    now = datetime(2026, 5, 31, 14, 0)

    # Never interacted → due.
    assert IntervalCheckIn(lambda: None).due(now).kind == "check_in"
    # Just interacted → not due.
    assert IntervalCheckIn(lambda: now - timedelta(minutes=5)).due(now) is None
    # Long ago → due (default interval 120 min).
    assert IntervalCheckIn(lambda: now - timedelta(hours=3)).due(now) is not None


# ── MemoryFollowUp ──────────────────────────────────────────────────────────────

def test_memory_follow_up_waits_for_idle(profiles):
    """The bug: follow-up fired right after the greeting. It must wait for idle time."""
    from nurse.proactive.triggers import MemoryFollowUp
    now = datetime(2026, 5, 31, 14, 0)
    lt = type("LT", (), {"latest": staticmethod(lambda: {"clinical": "headache"})})()

    # Just greeted / just interacted → must NOT fire (this was the reported bug).
    assert MemoryFollowUp(lt, lambda: now - timedelta(minutes=1)).due(now) is None
    # Idle long enough → fires.
    eng = MemoryFollowUp(lt, lambda: now - timedelta(hours=3)).due(now)
    assert eng is not None and eng.kind == "follow_up"
    # Per-day cap respected after firing.
    trig = MemoryFollowUp(lt, lambda: now - timedelta(hours=3))
    trig.due(now).on_fire()
    assert trig.due(now) is None


# ── VitalsThreshold ─────────────────────────────────────────────────────────────

def test_vitals_threshold_out_of_range(profiles):
    from nurse.proactive.triggers import VitalsThreshold
    path = profiles / "default_vitals_feed.jsonl"
    path.write_text(
        json.dumps({"type": "heart_rate", "value": "135 bpm", "timestamp": "t1"}) + "\n"
    )
    trig = VitalsThreshold("default")
    eng = trig.due(datetime(2026, 5, 31, 3, 0))
    assert eng is not None and eng.kind == "vitals_alert"
    assert eng.overrides_quiet_hours is True
    # Same reading does not re-fire.
    assert trig.due(datetime(2026, 5, 31, 3, 1)) is None


def test_vitals_in_range_no_fire(profiles):
    from nurse.proactive.triggers import VitalsThreshold
    (profiles / "default_vitals_feed.jsonl").write_text(
        json.dumps({"type": "heart_rate", "value": "72", "timestamp": "t1"}) + "\n"
    )
    assert VitalsThreshold("default").due(datetime(2026, 5, 31, 12, 0)) is None


# ── Engage classifier ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("yes please", "yes"),
    ("sure, go ahead", "yes"),
    ("no", "no"),
    ("not now", "no"),
    ("go away", "no"),
    ("", "unclear"),
    ("what is the weather", "unclear"),
])
def test_engage_classifier(profiles, text, expected):
    from nurse.proactive.engage import classify_engage_reply
    assert classify_engage_reply(text) == expected


# ── Scheduler priority + quiet hours (dry run) ──────────────────────────────────

class _FakePipeline:
    def __init__(self, profiles):
        self.patient_id = "default"
        self.long_term = type("LT", (), {"latest": staticmethod(lambda: None)})()
    def last_interaction(self):
        return None  # forces check-in to be due


def test_scheduler_quiet_hours_suppresses_checkin(profiles):
    from nurse.proactive.scheduler import ProactiveScheduler
    sched = ProactiveScheduler(_FakePipeline(profiles), dry_run=True)
    # 03:00 is quiet; check-in is due (no last interaction) but must be suppressed.
    assert sched.tick(datetime(2026, 5, 31, 3, 0)) is False
    # 14:00 is waking hours; check-in fires.
    assert sched.tick(datetime(2026, 5, 31, 14, 0)) is True


def test_scheduler_vitals_beats_checkin(profiles):
    from nurse.proactive.scheduler import ProactiveScheduler
    (profiles / "default_vitals_feed.jsonl").write_text(
        json.dumps({"type": "oxygen_saturation", "value": "88", "timestamp": "t1"}) + "\n"
    )
    sched = ProactiveScheduler(_FakePipeline(profiles), dry_run=True)
    eng = sched._pick(datetime(2026, 5, 31, 14, 0))
    assert eng.kind == "vitals_alert"  # priority 100 beats check-in's 20


class _ScriptedPipeline(_FakePipeline):
    """Pipeline whose engage() returns a scripted sequence of results."""
    def __init__(self, profiles, results):
        super().__init__(profiles)
        self._results = list(results)
        self.engaged = []  # kinds engaged, in order

    def engage(self, engagement):
        self.engaged.append(engagement.kind)
        return self._results.pop(0) if self._results else "no_response"


def test_no_response_rearms_and_retries_after_delay(profiles):
    from nurse.proactive.scheduler import ProactiveScheduler
    # First engage gets no response, second (the retry) is delivered.
    pipe = _ScriptedPipeline(profiles, ["no_response", "delivered"])
    sched = ProactiveScheduler(pipe, dry_run=False)
    retry_min = sched.retry_after.total_seconds() / 60

    t0 = datetime(2026, 5, 31, 14, 0)
    # check-in fires (no last interaction), gets no response → re-armed.
    assert sched.tick(t0) is False
    assert sched._pending_retry is not None
    assert pipe.engaged == ["check_in"]

    # Before the retry window: not retried again.
    sched.tick(t0 + timedelta(minutes=retry_min - 1))
    assert pipe.engaged == ["check_in"]  # unchanged

    # After the retry window: retried and delivered, pending cleared.
    assert sched.tick(t0 + timedelta(minutes=retry_min + 1)) is True
    assert pipe.engaged == ["check_in", "check_in"]
    assert sched._pending_retry is None


def test_retry_waits_out_quiet_hours_for_nonurgent(profiles):
    from nurse.proactive.scheduler import ProactiveScheduler
    pipe = _ScriptedPipeline(profiles, ["no_response", "delivered"])
    sched = ProactiveScheduler(pipe, dry_run=False)
    retry_min = sched.retry_after.total_seconds() / 60

    # Fire a check-in at 21:50 (waking), no response → re-armed ~30 min later (22:20,
    # inside quiet hours). The non-urgent retry must wait, not fire during quiet hours.
    t0 = datetime(2026, 5, 31, 21, 50)
    sched.tick(t0)
    assert sched._pending_retry is not None
    # 22:20 is quiet → retry suppressed, still pending, engage not called again.
    sched.tick(datetime(2026, 5, 31, 22, 20))
    assert pipe.engaged == ["check_in"]
    assert sched._pending_retry is not None
