"""Abstract base class for LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class LLMBackend(ABC):
    @abstractmethod
    def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Stream the assistant reply token by token, dispatching any tool calls."""
        ...

    def warmup(self) -> None:
        """Load weights and pre-fill any cacheable prompt prefix ahead of the
        first turn. Optional — defaults to a no-op for backends without caching."""
        return None
