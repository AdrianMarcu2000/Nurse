"""SkillRegistry — builds the Front Voice + enrichment skills from config.

Mirrors how the proactive scheduler holds its trigger list. On-device models are
loaded at startup; skills whose model/endpoint is unavailable are disabled gracefully
(skipped, not fatal — enrichment is never required for a turn).
"""
from __future__ import annotations

import logging

from nurse.config import get_config
from nurse.skills.base import FrontVoice, Skill

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, front_voice: FrontVoice, skills: list[Skill]) -> None:
        self.front_voice = front_voice
        # Only keep skills that are actually usable on this deployment.
        self.skills: list[Skill] = []
        for s in skills:
            try:
                if s.available():
                    self.skills.append(s)
                else:
                    logger.warning("Skill %s disabled — model/endpoint unavailable", s.name)
            except Exception as e:
                logger.warning("Skill %s failed availability check (%s) — disabled", s.name, e)

    def enrichment_skills(self) -> list[Skill]:
        return list(self.skills)

    @classmethod
    def from_config(cls, front_voice: FrontVoice, skill_objects: list[Skill]) -> "SkillRegistry":
        """Filter the provided skill objects by their config `enabled` flag, then build.

        Skill *construction* (which model id, etc.) is the caller's job; this just honors
        the per-skill enable flags in config.skills.registry.
        """
        cfg = get_config().get("skills", {}).get("registry", {})
        enabled = [
            s for s in skill_objects
            if cfg.get(s.name, {}).get("enabled", True)
        ]
        return cls(front_voice, enabled)
