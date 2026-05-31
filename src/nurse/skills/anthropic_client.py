"""Anthropic (Claude) API client for cloud-backed specialists.

Opt-in cloud path: a specialist can run on Claude (haiku/sonnet) instead of a local
model. The API key is read ONLY from the ANTHROPIC_API_KEY environment variable — never
from code, config, or the repo. Uses stdlib urllib (no new dependency); isolated so it
can be mocked in tests with no real network.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

# Friendly aliases selectable at startup; raw model ids also pass through.
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class AnthropicClient:
    """Minimal Claude Messages API client. Key from env only."""

    def __init__(self, model: str, max_tokens: int = 512, timeout: float = 30.0) -> None:
        self.model = resolve_model(model)
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        """Single-shot completion. Returns the text, or "" on any failure (cloud is
        enrichment — a failed call must never break the turn)."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("ANTHROPIC_API_KEY not set — cloud specialist disabled")
            return ""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
        )
        # Make the network call unmistakable in the log: the device is reaching off-box.
        logger.info("CLOUD CALL → Anthropic %s | input: %r", self.model, user)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            # Messages API returns {"content": [{"type":"text","text": "..."}], ...}
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            text = "".join(parts).strip()
            logger.info("CLOUD REPLY ← Anthropic %s (%.0fms): %s",
                        self.model, (time.perf_counter() - t0) * 1000, text)
            return text
        except Exception as e:
            logger.warning("CLOUD CALL FAILED ← Anthropic %s: %s", self.model, e)
            return ""
