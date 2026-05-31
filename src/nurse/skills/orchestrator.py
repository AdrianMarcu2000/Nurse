"""Orchestrator — the GP. Conducts a turn: build Context, run enrichment skills in the
background, run the open-the-turn race, and stream the Front Voice (the only speaker).

Open-the-turn race (docs/skill-platform-plan.md):
- Front Voice + selected specialists start in parallel.
- Wait up to `skills.open_deadline_s`. If a specialist finished by then, its finding is
  put into the Context and the Front Voice opens by relaying it (better answer in hand).
  Otherwise the Front Voice opens from what it knows; specialists that finish later are
  added to the Context for the continuation.
- Either way ONE voice speaks: the Front Voice. Specialists never speak.
"""
from __future__ import annotations

import concurrent.futures
import logging

from nurse.config import get_config
from nurse.skills.base import Context
from nurse.skills.registry import SkillRegistry
from nurse.skills.router import SkillRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, registry: SkillRegistry, router: SkillRouter | None = None,
                 executor: concurrent.futures.ThreadPoolExecutor | None = None) -> None:
        self.registry = registry
        self.router = router or SkillRouter(registry.enrichment_skills())
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.open_deadline_s: float = (
            get_config().get("skills", {}).get("open_deadline_s", 0.8)
        )

    def build_context(self, user_text: str, history, patient_summary: str,
                      rag_context: str) -> Context:
        return Context(
            user_text=user_text,
            history=history,
            patient_summary=patient_summary,
            rag_context=rag_context,
        )

    def respond(self, context: Context):
        """Stream the Front Voice's tokens for this turn, running the open-the-turn race.

        Returns an iterator of text tokens (the TTS feed contract). Specialist findings
        are folded into `context` so the Front Voice references them in its own words.
        """
        skills = self.router.select(context)
        if not skills:
            return self.registry.front_voice.respond(context)

        # Dispatch every selected skill in the background, off the voice path.
        futures = {self._executor.submit(self._safe_run, s, context): s for s in skills}

        # Open-the-turn race: wait up to the deadline for ANY specialist to return.
        done, _pending = concurrent.futures.wait(
            futures, timeout=self.open_deadline_s,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        # Fold whatever finished in time into the context (newest observation wins later).
        for fut in done:
            finding = fut.result()
            if finding is not None:
                context.add_finding(finding)

        # Whether or not a specialist opened, the Front Voice speaks — with any findings
        # already in context, it opens with the better answer; otherwise from its own
        # knowledge. Remaining specialists keep running; their findings are collected for
        # promotion / continuation by collect_pending().
        self._pending_futures = [f for f in futures if f not in done]
        self._all_futures = futures
        return self.registry.front_voice.respond(context)

    def collect_pending(self, context: Context, timeout: float = 5.0) -> list:
        """After the opener, gather any specialist findings that hadn't returned yet and
        add them to the context (for the Front Voice's continuation / promotion). Returns
        the newly added findings."""
        pending = getattr(self, "_pending_futures", [])
        added = []
        if not pending:
            return added
        done, _ = concurrent.futures.wait(pending, timeout=timeout)
        for fut in done:
            try:
                finding = fut.result()
            except Exception:
                finding = None
            if finding is not None:
                context.add_finding(finding)
                added.append(finding)
        return added

    @staticmethod
    def _safe_run(skill, context: Context):
        try:
            return skill.run(context)
        except Exception as e:
            logger.warning("Skill %s failed: %s", getattr(skill, "name", "?"), e)
            return None
