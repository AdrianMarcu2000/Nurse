"""
LLM client factory — selects the right backend automatically:
  - Apple Silicon (mlx available) → MLXBackend  (faster on M-series)
  - Everything else (Jetson CUDA, x86) → LlamaBackend  (llama.cpp)

All callers use stream_response() regardless of which backend is active.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from nurse.llm.backends.base import LLMBackend
from nurse.llm.tools import ToolDispatcher

logger = logging.getLogger(__name__)


def _build_backend(dispatcher: ToolDispatcher) -> LLMBackend:
    try:
        import mlx.core  # noqa: F401 — present only on Apple Silicon
        from nurse.llm.backends.mlx_backend import MLXBackend
        logger.info("LLM backend: MLX (Apple Silicon)")
        return MLXBackend(dispatcher)
    except ImportError:
        from nurse.llm.backends.llama_backend import LlamaBackend
        logger.info("LLM backend: llama.cpp")
        return LlamaBackend(dispatcher)


class LLMClient:
    """Thin wrapper so the rest of the codebase never imports a backend directly."""

    def __init__(self, dispatcher: ToolDispatcher) -> None:
        self._backend = _build_backend(dispatcher)

    def warmup(self) -> None:
        """Pre-load weights and prime prompt caches before the first turn."""
        self._backend.warmup()

    def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        return self._backend.stream_response(messages)
