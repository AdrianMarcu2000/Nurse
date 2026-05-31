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
        # Completed utterances land here from the VAD thread. Both the async iterator
        # (reactive loop) and listen_once() (proactive engagement) read from it, so
        # there is a single source of truth for "the user said something".
        self._utterance_q: queue.Queue[np.ndarray] = queue.Queue()
        # When set, the VAD processing thread discards all audio and resets state.
        self._muted = threading.Event()
        self._vad_started = False
        # Barge-in: when enabled AND echo cancellation is available, the mic keeps
        # listening (on the echo-cancelled stream) during playback and fires this callback
        # on sustained patient speech, instead of discarding muted audio.
        self._barge_in_cb = None
        self._aec = None
        self._barge_in_chunks = 0
        bcfg = get_config().get("barge_in", {})
        self._barge_in_enabled = bool(bcfg.get("enabled", False))
        self._barge_in_trigger = int(bcfg.get("trigger_chunks", 4))

    def set_barge_in_callback(self, callback) -> None:
        """Register a callback fired when the patient speaks while Aria is talking.
        Only active if barge_in is enabled and an AEC backend is available."""
        self._barge_in_cb = callback
        if self._barge_in_enabled:
            from nurse.speech.aec import EchoCanceller
            ec = EchoCanceller(self.chunk_samples, self.sample_rate)
            self._aec = ec if ec.available() else None

    def _barge_in_active(self) -> bool:
        return self._barge_in_enabled and self._aec is not None and self._barge_in_cb is not None

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

    def _start_vad(self) -> None:
        """Open the input stream and start the VAD thread that fills _utterance_q.

        Idempotent — safe to call from both the async loop and listen_once().
        """
        if self._vad_started:
            return
        self._vad_started = True

        vad_model, _ = self._load_vad()
        import torch

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

        def process():
            speech_buffer: list[np.ndarray] = []
            in_speech = False
            silence_chunks = 0
            with stream:
                while True:
                    chunk = self._audio_q.get()
                    if chunk is None:
                        break

                    # While muted (robot is speaking): normally discard audio. But if
                    # barge-in is active, run VAD on the echo-cancelled mic and fire the
                    # callback on sustained patient speech (talk-over interruption).
                    if self._muted.is_set():
                        speech_buffer = []
                        in_speech = False
                        silence_chunks = 0
                        if self._barge_in_active():
                            mono = chunk[:, 0] if chunk.ndim > 1 else chunk
                            # No true reference frame wired yet → AEC passthrough; the real
                            # reference is the speaker output (wired with the audio backend
                            # on-device). VAD still gates on the (cancelled) stream.
                            clean = self._aec.process(mono, np.zeros_like(mono))
                            conf = vad_model(torch.from_numpy(clean).float(), self.sample_rate).item()
                            if conf >= self.vad_threshold:
                                self._barge_in_chunks += 1
                                if self._barge_in_chunks >= self._barge_in_trigger:
                                    self._barge_in_chunks = 0
                                    try:
                                        self._barge_in_cb()
                                    except Exception:
                                        pass
                            else:
                                self._barge_in_chunks = 0
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
                            self._utterance_q.put(np.concatenate(speech_buffer))
                            speech_buffer = []
                            in_speech = False
                            silence_chunks = 0

        self._thread = threading.Thread(target=process, daemon=True)
        self._thread.start()

    def listen_once(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until the next complete utterance, or None if `timeout` elapses.

        Used by proactive engagements to capture a single reply. Starts the VAD if it
        is not already running.
        """
        self._start_vad()
        try:
            return self._utterance_q.get(timeout=timeout)
        except queue.Empty:
            return None

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._start_vad)
        try:
            while True:
                # Pull from the shared utterance queue without blocking the event loop.
                utterance = await loop.run_in_executor(None, self._utterance_q.get)
                yield utterance
        finally:
            self._audio_q.put(None)
