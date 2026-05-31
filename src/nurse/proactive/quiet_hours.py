"""Fixed quiet-hours window — suppresses non-urgent proactive speech."""
from __future__ import annotations

from datetime import datetime, time

from nurse.config import get_config


def _parse(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


class QuietHours:
    """A wrap-around daily window (e.g. 22:00–07:00) during which Aria stays quiet.

    Reads proactive.quiet_hours from config. `active(now)` is robust to windows that
    cross midnight. Urgent triggers (med reminders, vitals alerts) consult
    `overrides_quiet_hours` on the engagement and bypass this gate at the scheduler.
    """

    def __init__(self) -> None:
        qh = get_config()["proactive"]["quiet_hours"]
        self.start = _parse(qh["start"])
        self.end = _parse(qh["end"])

    def active(self, now: datetime | None = None) -> bool:
        t = (now or datetime.now()).time()
        if self.start == self.end:
            return False  # zero-length window → never quiet
        if self.start < self.end:
            # Same-day window, e.g. 01:00–05:00
            return self.start <= t < self.end
        # Wrap-around window, e.g. 22:00–07:00
        return t >= self.start or t < self.end
