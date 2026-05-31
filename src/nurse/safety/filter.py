"""
Safety filter — checks user input for emergency keywords before the LLM runs.
If triggered, injects a forced escalate_to_human tool call into the message list
so the LLM will speak the escalation message instead of attempting to help.
"""
from __future__ import annotations

import json
import logging
import re

from nurse.config import get_config

logger = logging.getLogger(__name__)


def _build_pattern():
    keywords = get_config()["safety"]["escalation_keywords"]
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile("|".join(escaped), re.IGNORECASE)


_PATTERN: re.Pattern | None = None


def _get_pattern() -> re.Pattern:
    global _PATTERN
    if _PATTERN is None:
        _PATTERN = _build_pattern()
    return _PATTERN


def check_and_escalate(user_text: str, messages: list[dict]) -> tuple[bool, list[dict]]:
    """
    Returns (triggered, messages).
    If an emergency keyword is found, prepends a forced tool-call assistant message
    so the LLM is primed to deliver the escalation response.
    The caller is still responsible for actually dispatching the tool.
    """
    match = _get_pattern().search(user_text)
    if not match:
        return False, messages

    keyword = match.group(0)
    logger.warning("Safety filter triggered by keyword: '%s'", keyword)

    # Inject a pre-formed tool call so the LLM knows escalation already happened
    forced_tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "forced_escalation",
                "type": "function",
                "function": {
                    "name": "escalate_to_human",
                    "arguments": json.dumps({
                        "reason": f"Patient reported: {keyword}",
                        "urgency": "immediate",
                    }),
                },
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "forced_escalation",
        "content": json.dumps({"status": "nursing_team_alerted", "urgency": "immediate"}),
    }

    modified = list(messages) + [forced_tool_call, tool_result]
    return True, modified
