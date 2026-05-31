"""Kokoro TTS — converts text to speech, sentence-by-sentence for low latency."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from nurse.config import get_config

# Sentence boundary pattern — split on . ! ? followed by whitespace or end
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
# Clause boundary pattern — split on , ; : followed by whitespace. Used only to
# break up a long opening segment so the first audio comes back faster; the
# punctuation is kept on the left segment so prosody stays natural.
_CLAUSE_RE = re.compile(r'(?<=[,;:])\s+')

# A segment shorter than this (chars) is synthesized whole — splitting it would
# only add choppiness and per-call overhead for no latency win. Segments longer
# than this are broken at the first clause boundary so playback can start sooner.
_CLAUSE_SPLIT_MIN_CHARS = 60


def _segments(text: str) -> list[str]:
    """
    Split text into synthesis segments tuned for low first-audio latency.

    Sentences are the primary unit. A long sentence is additionally split at its
    first clause boundary (comma/semicolon/colon) so the opening segment is short
    and reaches the speaker fast; the remainder is kept together for prosody.
    """
    segments: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE_RE.split(text)):
        if not sentence:
            continue
        if len(sentence) <= _CLAUSE_SPLIT_MIN_CHARS:
            segments.append(sentence)
            continue
        # Break off the first clause; keep the rest as one segment.
        parts = _CLAUSE_RE.split(sentence, maxsplit=1)
        segments.extend(p.strip() for p in parts if p.strip())
    return segments


@lru_cache(maxsize=1)
def _load_pipeline():
    from kokoro import KPipeline
    cfg = get_config()["tts"]
    # Kokoro language code: 'a' = American English
    return KPipeline(lang_code="a")


def synthesize(text: str) -> np.ndarray:
    """Synthesize a full text string to a float32 audio array at 24 kHz."""
    pipeline = _load_pipeline()
    cfg = get_config()["tts"]
    voice = cfg.get("voice", "af_heart")
    speed = cfg.get("speed", 1.1)

    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(np.array(audio, dtype=np.float32))

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def synthesize_sentences(text: str):
    """
    Generator that splits text into sentences and yields (audio, sentence) pairs
    as each sentence is synthesized. Allows the speaker to start playing before
    the full response is generated.
    """
    cfg = get_config()["tts"]
    if not cfg.get("sentence_split", True):
        yield synthesize(text), text
        return

    segments = _segments(text)
    if not segments:
        return

    pipeline = _load_pipeline()
    voice = cfg.get("voice", "af_heart")
    speed = cfg.get("speed", 1.1)

    for segment in segments:
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(segment, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                chunks.append(np.array(audio, dtype=np.float32))
        if chunks:
            yield np.concatenate(chunks), segment


# Kokoro outputs at 24 kHz
SAMPLE_RATE = 24000
