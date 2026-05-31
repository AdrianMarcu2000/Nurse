"""Plays audio arrays through the speaker, blocking until playback completes."""
from __future__ import annotations

import numpy as np
import sounddevice as sd

from nurse.config import get_config


class Speaker:
    def __init__(self) -> None:
        cfg = get_config()["audio"]
        self.sample_rate: int = cfg["sample_rate"]
        self.output_device = cfg.get("output_device")

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Block until the audio has finished playing."""
        sr = sample_rate or self.sample_rate
        sd.play(audio, samplerate=sr, device=self.output_device)
        sd.wait()

    def stop(self) -> None:
        """Immediately stop any in-progress playback (used for barge-in)."""
        sd.stop()

    def play_chunks(self, chunks: list[np.ndarray], sample_rate: int | None = None) -> None:
        """Play a sequence of audio chunks in order."""
        if not chunks:
            return
        combined = np.concatenate(chunks)
        self.play(combined, sample_rate)
