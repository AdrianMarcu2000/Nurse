"""Unit tests for the safety escalation filter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from nurse.safety.filter import check_and_escalate


@pytest.mark.parametrize("text,should_trigger", [
    ("I have chest pain and can't breathe", True),
    ("My chest hurts a lot", True),
    ("I want to die", True),
    ("I feel suicidal", True),
    ("I fell and there's severe bleeding", True),
    ("I have a slight headache", False),
    ("Can you remind me to take my pills?", False),
    ("What time is lunch?", False),
    ("My temperature is 99.5", False),
    ("CHEST PAIN started 5 minutes ago", True),   # case-insensitive
])
def test_safety_filter(text, should_trigger):
    messages = [{"role": "user", "content": text}]
    triggered, result_messages = check_and_escalate(text, messages)
    assert triggered == should_trigger
    if should_trigger:
        # Injected messages: original + assistant tool-call + tool result
        assert len(result_messages) == 3
        tool_call = result_messages[1]
        assert tool_call["role"] == "assistant"
        assert tool_call["tool_calls"][0]["function"]["name"] == "escalate_to_human"
