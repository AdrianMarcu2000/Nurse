"""AudioSceneSkill — describe background sound via an audio-understanding model.

Real model call (per-skill configurable `skills.registry.audio.model_id`). Disabled
gracefully when no model/clip is present. Input is a clip path under `data/sense/`; a
json sidecar is supported for tests and for an upstream classifier that already labelled
the scene ("a smoke alarm is beeping", "someone is crying").
"""
from __future__ import annotations

import logging
from datetime import datetime

from nurse.config import get_config, resolve
from nurse.skills.base import Context, Skill, SkillFinding

logger = logging.getLogger(__name__)


class AudioSceneSkill(Skill):
    name = "audio"
    location = "on_device"
    mode = "sync"
    domain_keywords = []          # ambient sensor

    def __init__(self, model_id: str | None = None) -> None:
        cfg = get_config().get("skills", {}).get("registry", {}).get("audio", {})
        self.model_id = model_id or cfg.get("model_id")
        self.location = cfg.get("location", self.location)
        self._clip_path = resolve("data/sense") / "last_sound.wav"
        self._json_path = resolve("data/sense") / "last_sound.json"

    def available(self) -> bool:
        return bool(self.model_id) and (self._clip_path.exists() or self._json_path.exists())

    def run(self, context: Context) -> SkillFinding | None:
        observed_at = datetime.now()
        if self._json_path.exists():
            import json
            try:
                data = json.loads(self._json_path.read_text())
            except Exception:
                return None
            summary = str(data.get("summary", "")).strip()
            ts = data.get("observed_at")
            if ts:
                try:
                    observed_at = datetime.fromisoformat(ts)
                except ValueError:
                    pass
            if not summary:
                return None
            return SkillFinding(source="audio", summary=summary, observed_at=observed_at, keep=True)

        try:
            summary = self._classify(self._clip_path)
        except Exception as e:
            logger.warning("AudioSceneSkill model call failed: %s", e)
            return None
        return SkillFinding(source="audio", summary=summary, observed_at=observed_at,
                            keep=True) if summary else None

    def _classify(self, clip_path) -> str:
        """Call the configured audio model. Isolated/lazy so the heavy dep is optional.
        Concrete integration (an audio-event model / audio-LLM) is the deferred piece."""
        raise NotImplementedError("audio model integration is deferred (Step 10)")
