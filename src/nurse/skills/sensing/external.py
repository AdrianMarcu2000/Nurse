"""Cloud (external) skills — opt-in, off by default, ALWAYS async.

The project is offline-first; a cloud skill sends data off-device only when a deployer
consciously enables it. Cloud calls are slow and may finish after the turn, so they never
sit on the voice path: they run in the background and, when an important result returns,
it is queued for proactive surfacing (the Front Voice brings it up) via the scheduler's
FindingTrigger.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime

from nurse.config import get_config
from nurse.skills.base import Context, Skill, SkillFinding

logger = logging.getLogger(__name__)


class CloudClient:
    """Minimal JSON-over-HTTPS client. Isolated so it can be mocked in tests (no real
    network in CI). Real providers (cloud vision, etc.) call through this."""

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))


class CloudSkill(Skill):
    """Base for `location: cloud`, `mode: async` skills. Subclasses implement
    `build_payload` and `parse_finding`; this handles availability + the call."""

    location = "cloud"
    mode = "async"

    def __init__(self, name: str, endpoint: str | None = None) -> None:
        self.name = name
        cfg = get_config().get("skills", {}).get("registry", {}).get(name, {})
        self.endpoint = endpoint or cfg.get("endpoint")
        self.enabled = cfg.get("enabled", False)   # cloud is OFF by default
        self._client = CloudClient(self.endpoint) if self.endpoint else None

    def available(self) -> bool:
        return bool(self.enabled and self.endpoint)

    def build_payload(self, context: Context) -> dict:
        return {"text": context.user_text}

    def parse_finding(self, response: dict) -> SkillFinding | None:
        summary = str(response.get("summary", "")).strip()
        if not summary:
            return None
        return SkillFinding(source=self.name, summary=summary,
                            observed_at=datetime.now(), keep=True)

    def run(self, context: Context) -> SkillFinding | None:
        if self._client is None:
            return None
        try:
            response = self._client.post(self.build_payload(context))
        except Exception as e:
            logger.warning("CloudSkill %s call failed: %s", self.name, e)
            return None
        return self.parse_finding(response)
