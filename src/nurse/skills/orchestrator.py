"""Orchestrator — the GP. Conducts a turn: build Context, run the Front Voice, and
(Step 5+) dispatch enrichment skills whose findings flow back into the Front Voice.

For Step 3 it is intentionally thin: it builds the Context and returns the Front Voice's
token stream. The pipeline still owns ASR, the safety gate, and speech (via the arbiter).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from nurse.skills.base import Context
from nurse.skills.registry import SkillRegistry
from nurse.skills.router import SkillRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, registry: SkillRegistry, router: SkillRouter | None = None) -> None:
        self.registry = registry
        self.router = router or SkillRouter(registry.enrichment_skills())

    def build_context(self, user_text: str, history, patient_summary: str,
                      rag_context: str) -> Context:
        return Context(
            user_text=user_text,
            history=history,
            patient_summary=patient_summary,
            rag_context=rag_context,
        )

    def respond(self, context: Context) -> Iterator[str]:
        """Return the Front Voice's token stream for this turn.

        Step 3: Front Voice only. Step 5 adds: run selected enrichment skills, feed
        findings into the Context, and run the open-the-turn race / continuation.
        """
        return self.registry.front_voice.respond(context)
