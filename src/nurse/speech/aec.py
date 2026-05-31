"""Acoustic Echo Cancellation (AEC) — let Aria listen while speaking, for barge-in.

To accept barge-in we must keep the mic open during playback without Aria hearing her
own voice. AEC subtracts the known speaker signal (reference) from the mic input. This is
the heaviest, most hardware/mic-dependent piece; it cannot be meaningfully unit-tested
(echo behavior is physical), so it's gated behind config and degrades gracefully:

- If a real AEC backend (speexdsp) is installed and `barge_in.enabled`, VAD runs on the
  echo-cancelled stream and sustained patient speech triggers a barge-in.
- Otherwise `available()` is False and the caller keeps the existing mic-mute behavior
  (no barge-in) — never broken, just no talk-over.
"""
from __future__ import annotations

import logging

import numpy as np

from nurse.config import get_config

logger = logging.getLogger(__name__)


class EchoCanceller:
    """Thin wrapper over an optional speexdsp echo canceller, fed the TTS output as the
    reference signal. `process(mic_frame, ref_frame)` returns the echo-cancelled frame."""

    def __init__(self, frame_size: int, sample_rate: int, filter_len: int = 2048) -> None:
        self._ec = None
        try:
            from speexdsp import EchoCanceller as _Speex  # optional heavy dep
            self._ec = _Speex.create(frame_size, filter_len, sample_rate)
            logger.info("AEC: speexdsp echo canceller active")
        except Exception as e:  # not installed / unsupported platform
            logger.warning("AEC unavailable (%s) — barge-in disabled, using mic-mute", e)

    def available(self) -> bool:
        return self._ec is not None

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        if self._ec is None:
            return mic_frame
        # speexdsp works on int16 PCM bytes; convert, cancel, convert back.
        mic_i16 = (np.clip(mic_frame, -1, 1) * 32767).astype(np.int16).tobytes()
        ref_i16 = (np.clip(ref_frame, -1, 1) * 32767).astype(np.int16).tobytes()
        out = self._ec.process(mic_i16, ref_i16)
        return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767.0


def barge_in_enabled() -> bool:
    return bool(get_config().get("barge_in", {}).get("enabled", False))
