"""Long-term per-patient memory — a structured, per-session timeline.

Replaces the old single ever-growing summary blob (which polluted every turn and
re-summarized its own output across sessions). The store now holds:
  - patient_facts: stable, slow-changing facts (preferred name, recurring concerns)
  - sessions: a dated timeline, each entry summarizing ONE session's clinical events
    and rapport notes. Newest last. Old entries age out.

Two readers serve different needs:
  - load_for_prompt(): compact background injected into the system prompt, framed as
    prior-visit reference (NOT the current conversation) so a small model doesn't
    continue the old story instead of answering the patient.
  - latest(): the most recent session entry, used to make the greeting history-aware.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from nurse.config import get_config, resolve

logger = logging.getLogger(__name__)

# Keep the timeline bounded so the prompt stays small and "latest" stays meaningful.
_MAX_SESSIONS = 8
# Sessions injected into the system prompt as background (most recent few).
_PROMPT_SESSIONS = 3


class LongTermMemory:
    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        self._path = resolve("data/patient_profiles") / f"{patient_id}_memory.json"

    # ── load ──────────────────────────────────────────────────────────────────

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"patient_facts": {}, "sessions": []}
        try:
            data = json.loads(self._path.read_text())
            data.setdefault("patient_facts", {})
            data.setdefault("sessions", [])
            return data
        except Exception as e:
            logger.warning("Failed to load long-term memory: %s", e)
            return {"patient_facts": {}, "sessions": []}

    def latest(self) -> dict[str, Any] | None:
        """The most recent session entry, or None if no history yet."""
        sessions = self._read()["sessions"]
        return sessions[-1] if sessions else None

    def patient_facts(self) -> dict[str, Any]:
        return self._read()["patient_facts"]

    def load_for_prompt(self) -> str:
        """Compact background string for the system prompt.

        Returns "" when there is no history (first-ever session). The text is framed
        as reference about prior visits so the model treats it as context, not as the
        conversation to continue.
        """
        data = self._read()
        facts, sessions = data["patient_facts"], data["sessions"]
        if not facts and not sessions:
            return ""

        parts: list[str] = []
        if facts:
            fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            parts.append(f"What you know about this patient:\n{fact_lines}")
        for entry in sessions[-_PROMPT_SESSIONS:]:
            date = entry.get("date", "a previous visit")
            clinical = entry.get("clinical", "").strip()
            rapport = entry.get("rapport", "").strip()
            line = f"[{date}] {clinical}"
            if rapport:
                line += f" (Personal: {rapport})"
            parts.append(line)
        return "\n".join(parts)

    # ── update ────────────────────────────────────────────────────────────────

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))
        logger.info("Long-term memory saved for patient %s", self.patient_id)

    def update_from_session(self, session_transcript: str, llm_client, date: str) -> None:
        """Summarize ONLY this session's transcript into a new dated timeline entry.

        Critically, we summarize the transcript alone — never the prior summary — so
        the store does not re-summarize its own output and accrete hallucinations
        across sessions. The model returns JSON we parse into clinical/rapport/topics.
        """
        prompt = (
            "Summarize this single nurse-patient conversation as JSON with keys "
            "'clinical' (1-2 sentences: symptoms reported, vitals logged, reminders "
            "set, concerns), 'rapport' (one short phrase of non-clinical/personal "
            "detail worth remembering, or empty), and 'topics' (a short list of "
            "keywords). Be factual; do not invent anything not in the transcript. "
            "Respond with ONLY the JSON object.\n\n"
            f"Conversation:\n{session_transcript}"
        )
        messages = [
            {"role": "system", "content": "You are a clinical note summarizer. Output JSON only."},
            {"role": "user", "content": prompt},
        ]
        raw = "".join(llm_client.stream_response(messages)).strip()
        entry = self._parse_entry(raw, date)
        if entry is None:
            logger.warning("Could not parse session summary; skipping save.")
            return

        data = self._read()
        data["sessions"].append(entry)
        data["sessions"] = data["sessions"][-_MAX_SESSIONS:]
        self._write(data)

    @staticmethod
    def _parse_entry(raw: str, date: str) -> dict[str, Any] | None:
        """Extract the JSON object from model output; fall back to a plain entry."""
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                return {
                    "date": date,
                    "clinical": str(obj.get("clinical", "")).strip(),
                    "rapport": str(obj.get("rapport", "")).strip(),
                    "topics": obj.get("topics", []) or [],
                }
            except json.JSONDecodeError:
                pass
        # Fallback: store the raw text as clinical so nothing is lost.
        text = raw.strip()
        return {"date": date, "clinical": text, "rapport": "", "topics": []} if text else None
