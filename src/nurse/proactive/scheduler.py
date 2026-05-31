"""ProactiveScheduler — the loop that decides when Aria initiates.

Runs as an asyncio coroutine beside the reactive mic loop. Every tick it polls the
triggers, applies the quiet-hours gate (which urgent triggers bypass), picks the
highest-priority due engagement, and hands it to pipeline.engage().
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from nurse.config import get_config
from nurse.proactive.quiet_hours import QuietHours
from nurse.proactive.triggers import (
    DueReminders,
    IntervalCheckIn,
    MemoryFollowUp,
    VitalsThreshold,
)

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    def __init__(self, pipeline, *, dry_run: bool = False) -> None:
        self.pipeline = pipeline
        self.dry_run = dry_run
        cfg = get_config()["proactive"]
        self.enabled: bool = cfg["enabled"]
        self.tick_seconds: int = cfg["tick_seconds"]
        self.retry_after = timedelta(minutes=cfg["retry_after_minutes"])
        self.quiet = QuietHours()
        # An engagement that got no response, to retry once its not-before time passes.
        self._pending_retry = None          # the Engagement
        self._retry_not_before: datetime | None = None
        self.triggers = [
            VitalsThreshold(pipeline.patient_id),
            DueReminders(pipeline.patient_id),
            IntervalCheckIn(pipeline.last_interaction),
            MemoryFollowUp(pipeline.long_term, pipeline.last_interaction),
        ]

    def _pick(self, now: datetime):
        """Return the highest-priority due engagement allowed right now, or None."""
        quiet = self.quiet.active(now)
        candidates = []
        for trig in self.triggers:
            try:
                eng = trig.due(now)
            except Exception as e:  # a flaky trigger must not kill the loop
                logger.warning("Trigger %s failed: %s", type(trig).__name__, e)
                continue
            if eng is None:
                continue
            if quiet and not eng.overrides_quiet_hours:
                logger.debug("Suppressed %s — quiet hours", eng.kind)
                continue
            candidates.append(eng)
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.priority)

    def tick(self, now: datetime | None = None) -> bool:
        """One scheduling decision. Returns True if an engagement was delivered.

        A pending retry (an engagement that previously got no response) takes
        precedence once its not-before time passes; otherwise the triggers are polled.

        Pure-ish: with dry_run it never speaks, only logs — used for bench testing the
        scheduling logic with a fake clock.
        """
        now = now or datetime.now()

        # A retry that is now due preempts fresh polling — but still respects quiet
        # hours for non-urgent engagements.
        eng = None
        if self._pending_retry is not None and now >= self._retry_not_before:
            cand = self._pending_retry
            if self.quiet.active(now) and not cand.overrides_quiet_hours:
                # Keep waiting — push the retry to the end of quiet hours implicitly by
                # leaving it pending; it'll be eligible again next tick outside quiet.
                pass
            else:
                eng = cand
                self._pending_retry = None
                self._retry_not_before = None
        if eng is None:
            eng = self._pick(now)
            # While a retry is pending, don't let the same kind re-fire through the
            # normal trigger path before its retry window — that would defeat the
            # back-off. (A different, higher-priority kind may still fire.)
            if (eng is not None and self._pending_retry is not None
                    and eng.kind == self._pending_retry.kind):
                return False
        if eng is None:
            return False

        logger.info("Proactive trigger due: %s (priority %d) detail=%r",
                    eng.kind, eng.priority, eng.detail)
        if self.dry_run:
            return True

        result = self.pipeline.engage(eng)
        if result == "delivered":
            if eng.on_fire:
                eng.on_fire()  # e.g. mark follow-up as used for the day
            return True
        if result == "no_response":
            # Re-arm: try this same engagement again after the configured delay.
            self._pending_retry = eng
            self._retry_not_before = now + self.retry_after
            logger.info("Engagement %s re-armed for %s", eng.kind, self._retry_not_before)
        # "declined" and "busy" → drop / let normal triggers handle later.
        return False

    async def run(self) -> None:
        """Tick forever until cancelled."""
        if not self.enabled:
            logger.info("Proactive scheduler disabled in config.")
            return
        logger.info("Proactive scheduler started (tick=%ds, dry_run=%s)",
                    self.tick_seconds, self.dry_run)
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    # tick() blocks (ASR/LLM/TTS during an engagement); run it off the
                    # event loop so reactive turns and the sleep timer stay responsive.
                    await loop.run_in_executor(None, self.tick)
                except Exception as e:
                    logger.warning("Scheduler tick error: %s", e)
                await asyncio.sleep(self.tick_seconds)
        except asyncio.CancelledError:
            logger.info("Proactive scheduler stopped.")
            raise
