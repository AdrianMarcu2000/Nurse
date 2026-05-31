"""Tool definitions and dispatch for the nurse LLM."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nurse.config import get_config, resolve

logger = logging.getLogger(__name__)

# ── Tool schemas (OpenAI function-calling format, supported by Qwen2.5) ───────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "log_vital",
            "description": "Log a patient vital sign measurement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Vital type: temperature | blood_pressure | heart_rate | oxygen_saturation | blood_glucose | weight | pain_level",
                    },
                    "value": {"type": "string", "description": "The measured value with units, e.g. '98.6 F' or '120/80 mmHg'"},
                    "note": {"type": "string", "description": "Optional context note"},
                },
                "required": ["type", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "What the reminder is for, e.g. 'Take metformin 500mg'"},
                    "time": {"type": "string", "description": "Time in HH:MM 24h format or relative like 'in 2 hours'"},
                },
                "required": ["reason", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Alert the nursing team immediately. Use for any emergency, distress, or situation you cannot handle safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Brief description of the issue"},
                    "urgency": {"type": "string", "enum": ["urgent", "immediate"], "description": "immediate = life-threatening"},
                },
                "required": ["reason", "urgency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_patient_profile",
            "description": "Retrieve the current patient's care plan, medications, allergies, or other profile details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "What to look up: medications | allergies | care_plan | all"},
                },
                "required": ["field"],
            },
        },
    },
]


# ── Dispatcher ────────────────────────────────────────────────────────────────

class ToolDispatcher:
    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        self._log_path = resolve("data/patient_profiles") / f"{patient_id}_vitals.jsonl"
        self._reminders_path = resolve("data/patient_profiles") / f"{patient_id}_reminders.jsonl"

    def dispatch(self, name: str, arguments: str | dict) -> str:
        args: dict = json.loads(arguments) if isinstance(arguments, str) else arguments
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return json.dumps(handler(**args))
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return json.dumps({"error": str(e)})

    def _tool_log_vital(self, type: str, value: str, note: str = "") -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": self.patient_id,
            "type": type,
            "value": value,
            "note": note,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Logged vital: %s", entry)
        return {"status": "logged", "entry": entry}

    def _tool_set_reminder(self, reason: str, time: str) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": self.patient_id,
            "reason": reason,
            "scheduled_time": time,
            "acknowledged": False,
        }
        self._reminders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._reminders_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Set reminder: %s", entry)
        return {"status": "reminder_set", "entry": entry}

    def _tool_escalate_to_human(self, reason: str, urgency: str) -> dict:
        # In production: call nurse-call API, buzz pager, etc.
        # For now: log loudly and return so the LLM can speak the escalation message.
        entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": self.patient_id,
            "reason": reason,
            "urgency": urgency,
        }
        alert_path = resolve("data/patient_profiles") / f"{self.patient_id}_alerts.jsonl"
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alert_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.warning("ESCALATION %s: %s — %s", urgency.upper(), self.patient_id, reason)
        return {"status": "nursing_team_alerted", "urgency": urgency}

    def _tool_lookup_patient_profile(self, field: str) -> dict:
        profile_path = resolve("data/patient_profiles") / f"{self.patient_id}.json"
        if not profile_path.exists():
            return {"error": "No profile found for this patient"}
        with open(profile_path) as f:
            profile = json.load(f)
        if field == "all":
            return profile
        return {field: profile.get(field, "Not found")}
