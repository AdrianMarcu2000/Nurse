"""Unit tests for tool dispatch."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest


@pytest.fixture
def dispatcher(tmp_path, monkeypatch):
    """ToolDispatcher with profiles dir redirected to tmp_path."""
    import nurse.config as cfg_module
    original_resolve = cfg_module.resolve

    def patched_resolve(rel):
        if rel.startswith("data/patient_profiles"):
            return tmp_path / Path(rel).name
        return original_resolve(rel)

    monkeypatch.setattr(cfg_module, "resolve", patched_resolve)

    from nurse.llm.tools import ToolDispatcher
    return ToolDispatcher(patient_id="test_patient")


def test_log_vital(dispatcher, tmp_path):
    result = json.loads(dispatcher.dispatch("log_vital", {"type": "temperature", "value": "99.1 F"}))
    assert result["status"] == "logged"
    assert result["entry"]["type"] == "temperature"


def test_set_reminder(dispatcher):
    result = json.loads(dispatcher.dispatch("set_reminder", {"reason": "Take metformin", "time": "08:00"}))
    assert result["status"] == "reminder_set"
    assert "metformin" in result["entry"]["reason"]


def test_escalate(dispatcher):
    result = json.loads(dispatcher.dispatch("escalate_to_human", {"reason": "chest pain", "urgency": "immediate"}))
    assert result["status"] == "nursing_team_alerted"
    assert result["urgency"] == "immediate"


def test_unknown_tool(dispatcher):
    result = json.loads(dispatcher.dispatch("nonexistent_tool", {}))
    assert "error" in result
