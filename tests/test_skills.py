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


# ── Step 5: open-the-turn race ──────────────────────────────────────────────────

import time as _time


class _SlowSkill(Skill):
    name = "cardiac"
    domain_keywords = ["heart"]
    def __init__(self, delay, summary="HR high"):
        self._delay = delay
        self._summary = summary
    def run(self, context):
        _time.sleep(self._delay)
        return SkillFinding(source="cardiac", summary=self._summary)


class _RecordingVoice(FrontVoice):
    """Records how many findings were in context when it was asked to respond."""
    def __init__(self):
        self.findings_at_respond = []
    def respond(self, context):
        self.findings_at_respond.append(len(context.findings))
        yield "spoken"


def _orch(skills, voice, deadline):
    from nurse.skills.orchestrator import Orchestrator
    from nurse.skills.registry import SkillRegistry
    from nurse.skills.router import SkillRouter
    reg = SkillRegistry(voice, skills)
    o = Orchestrator(reg, router=SkillRouter(skills))
    o.open_deadline_s = deadline
    return o


def test_fast_specialist_opens_the_turn():
    # Specialist returns well before the deadline → its finding is in context when the
    # Front Voice opens (the Front Voice relays the better answer).
    voice = _RecordingVoice()
    o = _orch([_SlowSkill(delay=0.01)], voice, deadline=0.5)
    ctx = o.build_context("my heart is racing", [], "", "")
    list(o.respond(ctx))
    assert voice.findings_at_respond[0] == 1      # finding present at open
    assert len(ctx.findings) == 1


def test_slow_specialist_does_not_delay_opener_then_enriches():
    # Specialist misses the deadline → Front Voice opens with NO finding; the finding is
    # then collected afterwards for the continuation.
    voice = _RecordingVoice()
    o = _orch([_SlowSkill(delay=0.3)], voice, deadline=0.05)
    ctx = o.build_context("my heart is racing", [], "", "")
    list(o.respond(ctx))
    assert voice.findings_at_respond[0] == 0      # opener had no finding (not delayed)
    late = o.collect_pending(ctx, timeout=2.0)
    assert len(late) == 1 and len(ctx.findings) == 1   # enriched after the opener


def test_no_specialist_selected_is_plain_front_voice():
    voice = _RecordingVoice()
    o = _orch([_SlowSkill(delay=0.01)], voice, deadline=0.5)
    # Chit-chat: cardiac (keyword "heart") not matched → no skills run.
    ctx = o.build_context("tell me about Rome", [], "", "")
    list(o.respond(ctx))
    assert voice.findings_at_respond[0] == 0
    assert o.collect_pending(ctx) == []


# ── Step 6: sensing skills, recency fusion, promotion ──────────────────────────

@pytest.fixture
def sense_dir(tmp_path, monkeypatch):
    """Redirect data/sense (and config resolve) to tmp_path; clear config cache."""
    import nurse.config as cfg_module
    orig = cfg_module.resolve
    def patched(rel):
        if rel == "data/sense":
            return tmp_path
        return orig(rel)
    monkeypatch.setattr(cfg_module, "resolve", patched)
    import nurse.skills.sensing.vision as vmod
    monkeypatch.setattr(vmod, "resolve", patched)
    return tmp_path


def test_vision_skill_reads_json_sidecar(sense_dir, monkeypatch):
    import json
    from nurse.skills.sensing.vision import VisionSkill
    (sense_dir / "last_image.json").write_text(json.dumps(
        {"summary": "patient is on the floor", "observed_at": "2026-05-31T14:30:00"}))
    skill = VisionSkill(model_id="some-vlm")     # model_id set → available
    assert skill.available()
    finding = skill.run(Context(user_text="anything"))
    assert finding.source == "vision"
    assert "on the floor" in finding.summary
    assert finding.keep is True
    assert finding.observed_at.hour == 14


def test_vision_skill_disabled_without_model_or_input(sense_dir, monkeypatch):
    import nurse.skills.sensing.vision as vmod
    monkeypatch.setattr(vmod, "resolve", lambda rel: sense_dir)
    from nurse.skills.sensing.vision import VisionSkill
    assert VisionSkill(model_id=None).available() is False        # no model
    assert VisionSkill(model_id="vlm").available() is False       # model but no input


def test_recency_window_drops_stale_finding():
    now = datetime(2026, 6, 1, 12, 0, 0)
    ctx = Context(user_text="x")
    ctx.add_finding(SkillFinding(source="vision", summary="fresh",
                                 observed_at=now - timedelta(seconds=10)))
    ctx.add_finding(SkillFinding(source="vision", summary="stale",
                                 observed_at=now - timedelta(seconds=600)))
    fresh = ctx.fresh_findings(max_age_s=120, now=now)
    summaries = [f.summary for f in fresh]
    assert summaries == ["fresh"]                 # stale (10 min) dropped, only fresh kept


# ── Step 7: cloud skill (async, opt-in, mocked HTTP) ───────────────────────────

def test_cloud_skill_off_by_default(monkeypatch):
    from nurse.skills.sensing.external import CloudSkill
    # No config / no endpoint → unavailable (offline-first default).
    assert CloudSkill("vision", endpoint=None).available() is False


def test_cloud_skill_returns_finding_with_mocked_client():
    from nurse.skills.sensing.external import CloudSkill
    skill = CloudSkill("vision", endpoint="https://example/api")
    skill.enabled = True
    # Mock the HTTP client — no real network.
    class _FakeClient:
        def post(self, payload):
            return {"summary": "cloud saw a cluttered floor"}
    skill._client = _FakeClient()
    assert skill.available()
    finding = skill.run(Context(user_text="what do you see"))
    assert finding.source == "vision" and "cluttered floor" in finding.summary
    assert finding.keep is True


# ── Cloud (Anthropic) specialist ────────────────────────────────────────────────

def test_anthropic_client_key_from_env_only(monkeypatch):
    """No key in env → complete() returns '' (graceful), never raises, never reads
    anything but the env var."""
    from nurse.skills.anthropic_client import AnthropicClient, api_key_present
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert api_key_present() is False
    assert AnthropicClient("sonnet").complete("sys", "user") == ""


def test_anthropic_alias_resolution():
    from nurse.skills.anthropic_client import resolve_model
    assert resolve_model("sonnet").startswith("claude-sonnet")
    assert resolve_model("haiku").startswith("claude-haiku")
    assert resolve_model("claude-custom-id") == "claude-custom-id"


def test_cardiac_cloud_uses_anthropic_with_mock(monkeypatch):
    """Cloud cardiac calls Anthropic (mocked) and returns a finding — no real network."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import nurse.config as cfg
    cfg.set_overrides({"skills": {"registry": {"cardiac": {
        "enabled": True, "location": "cloud", "provider": "anthropic", "model": "sonnet"}}}})
    try:
        from nurse.skills.cardiac import CardiacSkill
        import nurse.skills.anthropic_client as ac
        monkeypatch.setattr(ac.AnthropicClient, "complete",
                            lambda self, system, user: "Possible tachycardia; monitor and escalate if it persists.")
        skill = CardiacSkill()
        assert skill._is_cloud() and skill.available()
        finding = skill.run(Context(user_text="my heart is racing"))
        assert finding.source == "cardiac" and "tachycardia" in finding.summary
    finally:
        cfg.set_overrides({"skills": {"registry": {"cardiac": {
            "location": "on_device", "provider": None}}}})
        cfg.get_config.cache_clear()
