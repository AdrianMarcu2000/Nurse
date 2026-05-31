"""Assembles the message list sent to the LLM each turn."""
from __future__ import annotations

import json
from typing import Any

from nurse.config import get_persona


def build_system_prompt(patient_summary: str = "", rag_context: str = "") -> str:
    persona = get_persona()
    parts = [persona["system_prompt"]]
    if patient_summary:
        # Framed as reference, not conversation. Without this a small model tends to
        # *continue* the prior-visit narrative instead of answering what the patient
        # just said.
        parts.append(
            "\n## Background from prior visits (reference only)\n"
            "The following is what you remember about this patient from earlier "
            "sessions. Use it for context, but respond to the patient's CURRENT "
            "message — do not resume or repeat past topics unless the patient does.\n"
            f"{patient_summary}"
        )
    if rag_context:
        parts.append(f"\n## Relevant care protocols\n{rag_context}")
    return "\n".join(parts)


def build_messages(
    history: list[dict[str, Any]],
    user_text: str,
    patient_summary: str = "",
    rag_context: str = "",
) -> list[dict[str, Any]]:
    """
    Returns the full message list: system + history + new user turn.
    history entries are dicts with keys: role, content, and optionally tool_calls / tool_call_id.
    """
    system = build_system_prompt(patient_summary, rag_context)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
