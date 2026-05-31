"""SkillRouter — domain scoring to pick which enrichment skills run this turn.

Deterministic-first (same philosophy as safety/filter.py): score each skill by how many
of its `domain_keywords` appear in the user text (plus a small boost if RAG fired). The
Front Voice ALWAYS runs; the router only decides which skills *add* enrichment. An
optional tiny-LLM tiebreak is left as a hook for ambiguous cases (off the hot path).
"""
from __future__ import annotations

import logging
import re

from nurse.skills.base import Context, Skill

logger = logging.getLogger(__name__)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


class SkillRouter:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills

    def score(self, skill: Skill, context: Context) -> int:
        words = _words(context.user_text)
        kws = [k.lower() for k in (skill.domain_keywords or [])]
        # Count keyword hits (phrase keywords matched as substrings, single words by token).
        hits = 0
        lowered = context.user_text.lower()
        for kw in kws:
            if " " in kw:
                if kw in lowered:
                    hits += 1
            elif kw in words:
                hits += 1
        return hits

    def select(self, context: Context) -> list[Skill]:
        """Return the enrichment skills worth running this turn (possibly empty).

        A skill is selected if it scores at least one domain hit. Sensing skills with no
        `domain_keywords` are treated as always-relevant ambient sensors and included.
        """
        selected: list[Skill] = []
        for skill in self.skills:
            if not skill.domain_keywords:
                selected.append(skill)            # ambient sensor (e.g. vision/audio)
            elif self.score(skill, context) > 0:
                selected.append(skill)
        if selected:
            logger.debug("Router selected skills: %s", [s.name for s in selected])
        return selected
