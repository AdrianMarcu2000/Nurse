"""Microphone capture with Silero VAD — yields complete utterances as numpy arrays."""
from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from nurse.config import get_config


class MicrophoneStream:
    """
    Captures audio from the mic and uses Silero VAD to detect when the user
    has finished speaking. Each complete utterance is yielded as a float32
    numpy array at 16 kHz.

    Usage:
        async for audio in MicrophoneStream():
            # audio is a float32 np.ndarray at 16 kHz
    """

    def __init__(self) -> None:
        cfg = get_config()["audio"]
        self.sample_rate: int = cfg["sample_rate"]
        self.channels: int = cfg["channels"]
        self.input_device = cfg.get("input_device")
        self.vad_silence_duration: float = cfg["vad_silence_duration"]
        self.vad_threshold: float = cfg["vad_threshold"]
        self.chunk_samples: int = int(self.sample_rate * cfg["vad_chunk_ms"] / 1000)
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        # When set, the VAD processing thread discards all audio and resets state.
        self._muted = threading.Event()

    def mute(self) -> None:
        """Stop accepting audio — called while the robot is speaking."""
        self._muted.set()

    def unmute(self) -> None:
        """Resume accepting audio — called after the robot finishes speaking."""
        # Drain any audio that arrived during mute (speaker echo)
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._muted.clear()

    def _load_vad(self):
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_ts, *_) = utils
        return model, get_speech_ts

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        vad_model, _ = await loop.run_in_executor(None, self._load_vad)

        import torch

        # Ring buffer for rolling chunk processing
        buffer: list[np.ndarray] = []
        speech_buffer: list[np.ndarray] = []
        in_speech = False
        silence_chunks = 0
        silence_threshold_chunks = int(
            self.vad_silence_duration * self.sample_rate / self.chunk_samples
        )

        def callback(indata: np.ndarray, frames: int, time, status):
            self._audio_q.put(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.input_device,
            callback=callback,
        )

        result_q: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def process():
            nonlocal in_speech, silence_chunks, speech_buffer
            with stream:
                while True:
                    chunk = self._audio_q.get()
                    if chunk is None:
                        break

                    # While muted (robot is speaking), discard audio and reset VAD state
                    if self._muted.is_set():
                        speech_buffer = []
                        in_speech = False
                        silence_chunks = 0
                        continue

                    mono = chunk[:, 0] if chunk.ndim > 1 else chunk
                    tensor = torch.from_numpy(mono).float()
                    confidence = vad_model(tensor, self.sample_rate).item()
                    is_speech = confidence >= self.vad_threshold

                    if is_speech:
                        in_speech = True
                        silence_chunks = 0
                        speech_buffer.append(mono)
                    elif in_speech:
                        speech_buffer.append(mono)
                        silence_chunks += 1
                        if silence_chunks >= silence_threshold_chunks:
                            utterance = np.concatenate(speech_buffer)
                            loop.call_soon_threadsafe(result_q.put_nowait, utterance)
                            speech_buffer = []
                            in_speech = False
                            silence_chunks = 0

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

        try:
            while True:
                utterance = await result_q.get()
                yield utterance
        finally:
            self._audio_q.put(None)
