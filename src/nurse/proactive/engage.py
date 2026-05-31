"""Ask-to-engage classifier — decide if the patient agreed to talk.

Deterministic keyword matching (same philosophy as safety/filter.py): we never let a
model decide whether consent was given. Returns "yes", "no", or "unclear"; the
scheduler treats "unclear" and silence as a decline and backs off.
"""
from __future__ import annotations

import re

from nurse.config import get_config


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def classify_engage_reply(text: str) -> str:
    """Return 'yes', 'no', or 'unclear' for the patient's reply to the engage prompt."""
    cfg = get_config()["proactive"]
    if not text or not text.strip():
        return "unclear"
    lowered = text.lower()
    words = _words(text)

    # Phrase-level "no" cues take precedence (e.g. "not now", "go away").
    no_phrases = [p for p in cfg["engage_no"] if " " in p]
    for phrase in no_phrases:
        if phrase in lowered:
            return "no"
    yes_phrases = [p for p in cfg["engage_yes"] if " " in p]
    for phrase in yes_phrases:
        if phrase in lowered:
            return "yes"

    # Single-word cues.
    no_words = {p for p in cfg["engage_no"] if " " not in p}
    yes_words = {p for p in cfg["engage_yes"] if " " not in p}
    hit_no = words & no_words
    hit_yes = words & yes_words
    if hit_no and not hit_yes:
        return "no"
    if hit_yes and not hit_no:
        return "yes"
    return "unclear"
