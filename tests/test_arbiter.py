"""Unit tests for the SpeechArbiter — priority, staleness, preemption signalling."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from nurse.speech.arbiter import (
    PRIORITY_PROACTIVE,
    PRIORITY_REACTIVE,
    PRIORITY_SAFETY,
    SpeechArbiter,
    SpeechIntent,
)


def _recorder():
    """A fake speak_fn that records what it was asked to speak."""
    spoken = []
    def speak(text_or_stream, stop_event):
        spoken.append(text_or_stream)
    return spoken, speak


def test_speak_now_speaks_fresh_intent():
    spoken, speak = _recorder()
    arb = SpeechArbiter(speak)
    assert arb.speak_now(SpeechIntent(priority=PRIORITY_REACTIVE, source="r", text_or_stream="hi"))
    assert spoken == ["hi"]


def test_speak_now_drops_stale_intent():
    spoken, speak = _recorder()
    arb = SpeechArbiter(speak)
    intent = SpeechIntent(priority=PRIORITY_PROACTIVE, source="p", text_or_stream="late",
                          still_relevant=lambda: False)
    assert arb.speak_now(intent) is False
    assert spoken == []


def test_speak_now_drops_expired_ttl():
    spoken, speak = _recorder()
    arb = SpeechArbiter(speak)
    intent = SpeechIntent(priority=PRIORITY_PROACTIVE, source="p", text_or_stream="old", ttl=0.0)
    time.sleep(0.01)
    assert arb.speak_now(intent) is False
    assert spoken == []


def test_intent_priority_ordering():
    # Lower sort_index = served first. Safety < reactive < proactive in sort order.
    safety = SpeechIntent(priority=PRIORITY_SAFETY, source="s")
    reactive = SpeechIntent(priority=PRIORITY_REACTIVE, source="r")
    proactive = SpeechIntent(priority=PRIORITY_PROACTIVE, source="p")
    ordered = sorted([proactive, safety, reactive])
    assert [i.source for i in ordered] == ["s", "r", "p"]


def test_queue_consumer_serves_highest_priority_first():
    order = []
    done = threading.Event()
    def speak(text, stop_event):
        order.append(text)
        if len(order) == 2:
            done.set()
    arb = SpeechArbiter(speak)
    # Submit low then high BEFORE starting the consumer, so both are queued.
    arb.submit(SpeechIntent(priority=PRIORITY_PROACTIVE, source="p", text_or_stream="low"))
    arb.submit(SpeechIntent(priority=PRIORITY_REACTIVE, source="r", text_or_stream="high"))
    arb.start()
    assert done.wait(timeout=2.0)
    arb.stop_loop()
    assert order == ["high", "low"]


def test_higher_priority_signals_preemption_of_interruptible_current():
    started = threading.Event()
    release = threading.Event()
    def speak(text, stop_event):
        started.set()
        # Block until the test releases, checking the stop flag.
        release.wait(timeout=2.0)
    arb = SpeechArbiter(speak)
    arb.submit(SpeechIntent(priority=PRIORITY_REACTIVE, source="r", text_or_stream="talking"))
    arb.start()
    assert started.wait(timeout=2.0)
    # A safety intent arrives mid-utterance → stop flag set on the current intent.
    arb.submit(SpeechIntent(priority=PRIORITY_SAFETY, source="s", text_or_stream="ALERT"))
    assert arb._stop_current.wait(timeout=1.0)
    release.set()
    arb.stop_loop()


def test_safety_intent_not_interruptible():
    # A non-interruptible current intent must NOT be preempted even by higher priority.
    arb = SpeechArbiter(lambda t, e: None)
    arb._current = SpeechIntent(priority=PRIORITY_SAFETY, source="s", interruptible=False)
    arb.submit(SpeechIntent(priority=PRIORITY_SAFETY + 1, source="x"))
    assert not arb._stop_current.is_set()


# ── Step 8: interruptible streaming + partial-draft semantics ──────────────────

def test_stream_and_speak_interrupts_and_reports_spoken():
    """_stream_and_speak halts when stop_event sets mid-stream and reports what was
    actually spoken vs the full draft. Exercised on a minimal object that binds the real
    method, with TTS stubbed (no audio)."""
    import types
    from nurse.pipeline import NursePipeline

    obj = types.SimpleNamespace()
    spoken_segments = []
    stop = threading.Event()

    def fake_speak_text(text, stop_event=None):
        # Simulate the patient interrupting after the first spoken sentence.
        spoken_segments.append(text)
        stop.set()
        return 1.0
    obj._speak_text = fake_speak_text

    def stream():
        # Two sentences; the second should never be spoken (interrupted after the first).
        for tok in ["Hello", " there.", " Second", " sentence."]:
            yield tok

    full, timing = NursePipeline._stream_and_speak(obj, stream(), stop_event=stop)
    assert timing["interrupted"] is True
    assert spoken_segments == ["Hello there."]          # only first sentence voiced
    assert timing["spoken"] == "Hello there."
    assert "Second" in full                              # full draft still captured


def test_speak_text_stops_between_segments(monkeypatch):
    """_speak_text checks stop_event between synthesized segments and calls speaker.stop()."""
    import types
    from nurse.pipeline import NursePipeline
    import nurse.pipeline as pl

    played, stopped = [], []
    stop = threading.Event()
    stop.set()                                          # already requested → stop immediately

    obj = types.SimpleNamespace()
    obj._mic = None
    obj.speaker = types.SimpleNamespace(
        play=lambda audio, sample_rate=None: played.append(1),
        stop=lambda: stopped.append(1),
    )
    monkeypatch.setattr(pl, "synthesize_sentences",
                        lambda text: iter([("audioA", "A"), ("audioB", "B")]))

    NursePipeline._speak_text(obj, "A. B.", stop_event=stop)
    assert played == [] and stopped == [1]              # nothing played; stop() called
