"""VisionSkill — describe what the camera sees via a vision-language model.

Real model call (the model is per-skill configurable, `skills.registry.vision.model_id`,
defaulting to a small local VLM). Disabled gracefully when no model is configured or no
image is available, so the platform runs fine without it. Image source is a path under
`data/sense/` (a frame grab written by the camera layer); tests inject a fixed image.
"""
from __future__ import annotations

import logging
from datetime import datetime

from nurse.config import get_config, resolve
from nurse.skills.base import Context, Skill, SkillFinding

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are the eyes of a care assistant. Briefly describe what is happening in this "
    "image that a nurse should know (the patient's posture, safety, distress). One "
    "sentence, factual; if nothing notable, say so."
)


class VisionSkill(Skill):
    name = "vision"
    location = "on_device"
    mode = "sync"
    domain_keywords = []          # ambient sensor — relevant every turn

    def __init__(self, model_id: str | None = None) -> None:
        cfg = get_config().get("skills", {}).get("registry", {}).get("vision", {})
        self.model_id = model_id or cfg.get("model_id")
        self.location = cfg.get("location", self.location)
        self._image_path = resolve("data/sense") / "last_image.jpg"
        self._json_path = resolve("data/sense") / "last_image.json"

    def available(self) -> bool:
        # Need a configured model AND some input present (image or a mock json).
        return bool(self.model_id) and (self._image_path.exists() or self._json_path.exists())

    def run(self, context: Context) -> SkillFinding | None:
        observed_at = datetime.now()
        # Mock/offline path: a json sidecar describing the scene (used in tests and when
        # an upstream process has already captured a description).
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
            return SkillFinding(source="vision", summary=summary, observed_at=observed_at, keep=True)

        # Real VLM path: call the configured multimodal model on the captured frame.
        try:
            summary = self._describe_image(self._image_path)
        except Exception as e:
            logger.warning("VisionSkill model call failed: %s", e)
            return None
        if not summary:
            return None
        return SkillFinding(source="vision", summary=summary, observed_at=observed_at, keep=True)

    def _describe_image(self, image_path) -> str:
        """Call the configured VLM. Kept isolated so the heavy multimodal backend is only
        imported when a real image+model are actually present."""
        from nurse.llm.vision_client import VisionClient  # lazy: optional heavy dep
        client = VisionClient(self.model_id)
        return client.describe(str(image_path), _PROMPT).strip()
