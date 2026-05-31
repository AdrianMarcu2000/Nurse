"""CardiacSkill — a worked example domain specialist.

A specialist is a background `Skill` (NOT a speaker): `run()` consults its own model with
a cardiac-focused prompt and returns a `SkillFinding` (its analysis). The orchestrator
feeds that finding back to the Front Voice, which speaks it in one voice. Adding further
specialists (cold/flu, etc.) is the same shape with a different prompt + model + keywords.
"""
from __future__ import annotations

import logging

from nurse.config import get_config
from nurse.llm.client import LLMClient
from nurse.llm.tools import ToolDispatcher
from nurse.skills.base import Context, Skill, SkillFinding

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a cardiac-care assistant supporting a nurse. Given what the patient said, "
    "give a brief, factual note for the nurse about possible cardiac considerations and "
    "what to watch for. You are NOT a doctor: never diagnose, prescribe, or recommend "
    "medication changes; if it sounds like an emergency, say it should be escalated. "
    "Two sentences max, plain language."
)


class CardiacSkill(Skill):
    name = "cardiac"
    location = "on_device"
    mode = "sync"
    domain_keywords = [
        "heart", "chest", "palpitation", "palpitations", "racing", "pulse",
        "heartbeat", "cardiac", "blood pressure",
    ]

    def __init__(self, model_id: str | None = None, dispatcher: ToolDispatcher | None = None) -> None:
        cfg = get_config().get("skills", {}).get("registry", {}).get("cardiac", {})
        self.model_id = model_id or cfg.get("model_id")
        self.location = cfg.get("location", self.location)
        self._dispatcher = dispatcher or ToolDispatcher("default")
        self._llm: LLMClient | None = None

    def available(self) -> bool:
        # Usable if we have a model id (a concrete model). Without one it's disabled.
        return bool(self.model_id)

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self._dispatcher, model_id=self.model_id)
        return self._llm

    def run(self, context: Context) -> SkillFinding | None:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": context.user_text},
        ]
        text = "".join(self._client().stream_response(messages)).strip()
        if not text:
            return None
        logger.info("CardiacSkill finding: %s", text)
        return SkillFinding(source="cardiac", summary=text, confidence=0.8, keep=True)
