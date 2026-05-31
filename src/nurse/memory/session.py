"""Short-term conversation memory — rolling window of recent turns."""
from __future__ import annotations

from typing import Any

from nurse.config import get_config


class SessionMemory:
    """Keeps the last N conversation turns for injection into the LLM context."""

    def __init__(self) -> None:
        self._max_turns: int = get_config()["llm"]["context_turns"]
        self._history: list[dict[str, Any]] = []

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str, tool_calls: list | None = None) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._history.append(msg)
        self._trim()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._history.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def _trim(self) -> None:
        # Each turn = 2 messages (user + assistant); keep last _max_turns pairs
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear(self) -> None:
        self._history.clear()
