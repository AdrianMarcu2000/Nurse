"""SpeechArbiter — the single owner of the TTS path.

Every occasion to speak (Front-Voice turn, safety escalation, proactive surfacing) is
submitted as a `SpeechIntent`. One consumer thread pulls the highest-priority intent and
speaks it via the injected `speak_fn`. The arbiter — not a bare lock — decides *what*
speaks and *when*: priority ordering, staleness drop, and (Step 8) interruption.

The arbiter is deterministic control logic with **no model in it** — it can be trusted
to gate. It is the only thing that drives speech output.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Priority levels (higher wins).
PRIORITY_SAFETY = 100      # escalation — preempts, never dropped, not interruptible
PRIORITY_REACTIVE = 50     # the patient is owed a reply to what they just said
PRIORITY_PROACTIVE = 20    # a surfaced finding / check-in
PRIORITY_IDLE = 10         # idle small-talk


@dataclass(order=True)
class SpeechIntent:
    # order=True + sort_index makes the heap/sort compare on (−priority, created_at).
    sort_index: tuple = field(init=False, repr=False)
    priority: int
    source: str = field(compare=False)
    # text_or_stream is a str or an iterator/generator of text tokens.
    text_or_stream: Any = field(default="", compare=False)
    created_at: float = field(default_factory=time.monotonic, compare=False)
    ttl: float | None = field(default=None, compare=False)         # seconds; None = no expiry
    still_relevant: Callable[[], bool] = field(default=lambda: True, compare=False)
    interruptible: bool = field(default=True, compare=False)

    def __post_init__(self) -> None:
        # Lower sort_index = served first: highest priority, then oldest.
        self.sort_index = (-self.priority, self.created_at)

    def is_stale(self, now: float) -> bool:
        if self.ttl is not None and (now - self.created_at) > self.ttl:
            return True
        try:
            return not self.still_relevant()
        except Exception:
            return False


class SpeechArbiter:
    """Single-consumer priority arbiter over one `speak_fn`.

    `speak_fn(text_or_stream, stop_event)` performs the actual synth+playback; it should
    consult `stop_event` between chunks so the arbiter can interrupt (Step 8). For Step 2
    a simple blocking speak_fn that ignores stop_event is sufficient.
    """

    def __init__(self, speak_fn: Callable[[Any, threading.Event], None]) -> None:
        self._speak_fn = speak_fn
        self._queue: list[SpeechIntent] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop_current = threading.Event()
        self._current: SpeechIntent | None = None
        self._consumer: threading.Thread | None = None
        self._running = False

    # ── submission ──────────────────────────────────────────────────────────────

    def submit(self, intent: SpeechIntent) -> None:
        with self._wake:
            self._queue.append(intent)
            self._queue.sort()                 # small queue; sort is fine and stable
            # If a higher-priority intent arrives and the current one is interruptible,
            # signal preemption (Step 8 honors it; Step 2 speak_fn may ignore it).
            if (self._current is not None and self._current.interruptible
                    and intent.priority > self._current.priority):
                self._stop_current.set()
            self._wake.notify()

    def stop(self) -> None:
        """Interrupt the current (interruptible) utterance — used by barge-in (Step 8)."""
        with self._wake:
            if self._current is not None and self._current.interruptible:
                self._stop_current.set()

    # ── consumer loop ─────────────────────────────────────────────────────────────

    def _next_intent(self) -> SpeechIntent | None:
        """Pop the best non-stale intent; drop stale ones. Caller holds the lock."""
        now = time.monotonic()
        while self._queue:
            intent = self._queue.pop(0)
            if intent.is_stale(now):
                logger.info("Dropping stale speech intent from %s", intent.source)
                continue
            return intent
        return None

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._queue:
                    self._wake.wait()
                if not self._running:
                    return
                intent = self._next_intent()
                if intent is None:
                    continue
                self._current = intent
                self._stop_current.clear()
            # Speak outside the lock so new intents can be queued/preempt concurrently.
            try:
                self._speak_fn(intent.text_or_stream, self._stop_current)
            except Exception as e:
                logger.warning("speak_fn error for %s: %s", intent.source, e)
            finally:
                with self._wake:
                    self._current = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consumer = threading.Thread(target=self._run, daemon=True, name="speech-arbiter")
        self._consumer.start()

    def stop_loop(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify_all()

    # ── synchronous helper (for tests / simple callers) ──────────────────────────

    def speak_now(self, intent: SpeechIntent) -> bool:
        """Synchronously serve one intent without the consumer thread: skip if stale,
        else speak. Returns True if spoken. Useful for tests and simple in-line use."""
        if intent.is_stale(time.monotonic()):
            logger.info("speak_now: dropping stale intent from %s", intent.source)
            return False
        self._current = intent
        self._stop_current.clear()
        try:
            self._speak_fn(intent.text_or_stream, self._stop_current)
            return True
        finally:
            self._current = None
