"""Faster-Whisper ASR wrapper — transcribes a numpy audio array."""
from __future__ import annotations

import logging
import time
from functools import lru_cache

import numpy as np

from nurse.config import get_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel

    cfg = get_config()["asr"]
    device = cfg.get("device", "auto")
    if device == "auto":
        # Apple Silicon uses cpu with float16 via CoreML; CUDA on Jetson
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "cpu"           # faster-whisper uses CPU on MPS systems
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    return WhisperModel(
        cfg["model_size"],
        device=device,
        compute_type=cfg.get("compute_type", "int8"),
    )


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a float32 mono 16 kHz numpy array.
    Returns the transcript string (empty string if nothing detected).
    """
    cfg = get_config()["asr"]
    model = _load_model()

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        language=cfg.get("language", "en"),
        beam_size=cfg.get("beam_size", 1),
        vad_filter=cfg.get("vad_filter", True),
    )
    text = " ".join(seg.text for seg in segments).strip()
    elapsed = (time.perf_counter() - t0) * 1000
    if text:
        # Log the patient's words so the full exchange is captured in logs/nurse.log
        # (the console print alone bypassed the logger / file log).
        logger.info("User said (%.0fms ASR): %s", elapsed, text)
        from rich.console import Console
        Console().print(f"[dim]ASR {elapsed:.0f}ms:[/dim] {text}")
    return text
