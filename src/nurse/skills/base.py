"""Core types for the skill platform: Context (blackboard), SkillFinding, and the
Skill / FrontVoice ABCs.

Design (docs/skill-platform-plan.md):
- The **Front Voice** is the one model that ever speaks; it yields plain text tokens.
- **Skills** are background enrichment (sensing OR domain specialists); they NEVER speak
  — `run()` returns a `SkillFinding` that the orchestrator feeds back into the Front
  Voice's context so the Front Voice continues in its own words.
- `Context` is the per-turn blackboard; keep-worthy findings are later promoted to memory.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SkillFinding:
    """One piece of enrichment produced by a skill.

    `observed_at` is when the observation *refers to* (not when inserted) — it drives
    recency ranking so a late-returning async finding can't masquerade as "now".
    `keep` marks whether this should be promoted to long-term memory after the turn.
    """
    source: str                       # skill name, e.g. "vision", "cardiac"
    summary: str                      # short natural-language summary for the Front Voice
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    observed_at: datetime = field(default_factory=datetime.now)
    keep: bool = False


@dataclass
class Context:
    """Per-turn blackboard. Built fresh each turn, fused, used, discarded. Cross-turn
    persistence is memory's job (via promotion), not the Context's."""
    user_text: str
    history: list[dict[str, Any]] = field(default_factory=list)
    patient_summary: str = ""
    rag_context: str = ""
    findings: list[SkillFinding] = field(default_factory=list)

    def add_finding(self, finding: SkillFinding) -> None:
        self.findings.append(finding)

    def findings_text(self) -> str:
        """Render current findings for injection into the Front Voice's prompt, newest
        first. Empty string when there are none."""
        if not self.findings:
            return ""
        ordered = sorted(self.findings, key=lambda f: f.observed_at, reverse=True)
        return "\n".join(f"- ({f.source}) {f.summary}" for f in ordered)


class Skill(ABC):
    """A background enrichment skill — sensing (vision/audio) or a domain specialist.

    Never speaks. Declares where it runs and when its result is usable. The orchestrator
    decides whether to run it (domain scoring) and how to use its finding.
    """

    name: str = "skill"
    location: str = "on_device"       # "on_device" | "cloud"
    mode: str = "sync"                # "sync" (may complete in-turn) | "async" (later)
    domain_keywords: list[str] = []
    model_id: str | None = None

    @abstractmethod
    def run(self, context: Context) -> SkillFinding | None:
        """Produce a finding for this turn (or None). May block (runs off the voice
        path). Must set `observed_at` appropriately on the finding."""
        ...

    def available(self) -> bool:
        """Whether this skill can run (model/endpoint present). A skill that returns
        False is disabled gracefully — sensing/enrichment is never required for a turn."""
        return True


class FrontVoice(ABC):
    """The single speaking model. Yields plain text tokens consumed by the one
    sentence-buffer → TTS path (the TTS feed contract). Never calls TTS directly."""

    @abstractmethod
    def respond(self, context: Context) -> Iterator[str]:
        """Stream the spoken reply as text tokens, weaving in any findings in `context`.
        Can be re-invoked to *continue* once more findings land."""
        ...
