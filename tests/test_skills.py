"""Unit tests for the skill platform base types, registry, and router."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from nurse.skills.base import Context, FrontVoice, Skill, SkillFinding
from nurse.skills.registry import SkillRegistry
from nurse.skills.router import SkillRouter


# ── test doubles ────────────────────────────────────────────────────────────────

class _Cardiac(Skill):
    name = "cardiac"
    domain_keywords = ["heart", "chest", "palpitations"]
    def run(self, context):
        return SkillFinding(source="cardiac", summary="HR looks high")


class _Vision(Skill):
    name = "vision"
    domain_keywords = []              # ambient sensor — always relevant
    def run(self, context):
        return SkillFinding(source="vision", summary="patient is seated")


class _Unavailable(Skill):
    name = "broken"
    def available(self):
        return False
    def run(self, context):
        return None


class _Voice(FrontVoice):
    def respond(self, context):
        yield "hello"


# ── Context / findings recency ────────────────────────────────────────────────

def test_context_findings_ordered_newest_first():
    ctx = Context(user_text="hi")
    old = SkillFinding(source="a", summary="old", observed_at=datetime(2026, 1, 1))
    new = SkillFinding(source="b", summary="new", observed_at=datetime(2026, 6, 1))
    ctx.add_finding(old)
    ctx.add_finding(new)
    text = ctx.findings_text()
    assert text.index("new") < text.index("old")   # newest first
    assert Context(user_text="x").findings_text() == ""


# ── Registry: availability + enabled filtering ─────────────────────────────────

def test_registry_drops_unavailable_skills():
    reg = SkillRegistry(_Voice(), [_Cardiac(), _Unavailable()])
    names = [s.name for s in reg.enrichment_skills()]
    assert "cardiac" in names and "broken" not in names


# ── Router: domain scoring ──────────────────────────────────────────────────────

def test_router_selects_on_domain_hit():
    router = SkillRouter([_Cardiac(), _Vision()])
    # Cardiac-flavored input → cardiac selected; vision always (ambient).
    sel = {s.name for s in router.select(Context(user_text="my chest hurts"))}
    assert sel == {"cardiac", "vision"}


def test_router_skips_specialist_without_hit():
    router = SkillRouter([_Cardiac(), _Vision()])
    # Chit-chat → cardiac NOT selected; vision still (ambient).
    sel = {s.name for s in router.select(Context(user_text="tell me about Rome"))}
    assert sel == {"vision"}


def test_router_score_counts_keyword_hits():
    router = SkillRouter([_Cardiac()])
    assert router.score(_Cardiac(), Context(user_text="my heart and chest")) == 2
    assert router.score(_Cardiac(), Context(user_text="nice weather")) == 0


# ── Front Voice parity: builds the same messages the old pipeline did ──────────

class _FakeLLM:
    def __init__(self):
        self.seen_messages = None
    def stream_response(self, messages):
        self.seen_messages = messages
        yield "ok"


def test_front_voice_builds_same_messages_as_build_messages():
    from nurse.llm.prompt import build_messages
    from nurse.skills.front_voice import LLMFrontVoice

    history = [{"role": "user", "content": "earlier"}]
    fake = _FakeLLM()
    fv = LLMFrontVoice(fake)
    ctx = Context(user_text="how are you", history=history,
                  patient_summary="summary text", rag_context="rag text")
    out = "".join(fv.respond(ctx))

    expected = build_messages(history, "how are you",
                              patient_summary="summary text", rag_context="rag text")
    assert out == "ok"
    assert fake.seen_messages == expected   # identical prompt to the old path


def test_front_voice_weaves_findings_into_context():
    from nurse.skills.front_voice import LLMFrontVoice
    fake = _FakeLLM()
    ctx = Context(user_text="hi", rag_context="")
    ctx.add_finding(SkillFinding(source="vision", summary="patient on the floor"))
    "".join(LLMFrontVoice(fake).respond(ctx))
    system = fake.seen_messages[0]["content"]
    assert "patient on the floor" in system   # finding reached the prompt


# ── Step 4: companion register + clinical hard-rules both present ──────────────

def test_persona_allows_companion_and_keeps_clinical_rules():
    from nurse.llm.prompt import build_system_prompt
    sp = build_system_prompt().lower()
    assert "companion" in sp                     # may chat about non-clinical topics
    assert "not a doctor" in sp                  # clinical hard-rule intact
    assert "escalate" in sp                      # escalation rule intact
    assert "never discuss topics unrelated" not in sp   # blanket ban removed
