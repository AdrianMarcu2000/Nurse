#!/usr/bin/env python3
"""
Offline evaluation — runs each scenario through the LLM (no audio, no TTS)
and scores against expected behaviors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

console = Console()


def score_scenario(scenario: dict, llm_response: str, tool_calls_made: list[str]) -> tuple[bool, str]:
    """Returns (passed, reason)."""
    # Check expected tool call
    expected_tool = scenario.get("expect_tool")
    if expected_tool and expected_tool not in tool_calls_made:
        return False, f"Expected tool '{expected_tool}' not called (got {tool_calls_made})"

    # Check escalation
    if scenario.get("expect_escalation") and "escalate_to_human" not in tool_calls_made:
        return False, "Expected escalation but none happened"

    # Check response must contain
    for phrase in scenario.get("response_must_contain_one_of", []):
        if phrase.lower() in llm_response.lower():
            break
    else:
        required = scenario.get("response_must_contain_one_of", [])
        if required:
            return False, f"Response missing expected phrases: {required}"

    # Check must not contain
    for phrase in scenario.get("must_not_contain", []):
        if phrase.lower() in llm_response.lower():
            return False, f"Response contains forbidden phrase: '{phrase}'"

    return True, "ok"


def run_eval() -> None:
    scenarios_path = Path(__file__).parent / "scenarios.yaml"
    scenarios = yaml.safe_load(scenarios_path.read_text())["scenarios"]

    from nurse.llm.tools import ToolDispatcher, TOOLS
    from nurse.llm.prompt import build_messages
    from nurse.llm.client import LLMClient
    from nurse.memory.longterm import LongTermMemory
    from nurse.rag.retrieve import retrieve

    dispatcher = ToolDispatcher("eval_patient")
    llm = LLMClient(dispatcher)

    results = []
    table = Table(title="Eval Results", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Input", max_width=40)
    table.add_column("Response", max_width=50)
    table.add_column("Pass", justify="center")
    table.add_column("Reason")

    for scenario in scenarios:
        sid = scenario["id"]
        user_text = scenario["input"]
        console.print(f"[dim]Running: {sid}…[/dim]")

        rag_ctx = retrieve(user_text)
        messages = build_messages([], user_text, rag_context=rag_ctx)

        # Track tool calls by monkey-patching dispatcher
        calls_made: list[str] = []
        original_dispatch = dispatcher.dispatch

        def tracking_dispatch(name, args, _orig=original_dispatch):
            calls_made.append(name)
            return _orig(name, args)

        dispatcher.dispatch = tracking_dispatch  # type: ignore

        tokens = list(llm.stream_response(messages))
        dispatcher.dispatch = original_dispatch  # type: ignore

        response = "".join(tokens).strip()
        passed, reason = score_scenario(scenario, response, calls_made)

        style = "green" if passed else "red"
        table.add_row(
            sid,
            user_text[:40],
            response[:50],
            f"[{style}]{'✓' if passed else '✗'}[/{style}]",
            reason,
        )
        results.append({"id": sid, "passed": passed, "reason": reason})

    console.print(table)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    console.print(f"\n[bold]Score: {passed}/{total} ({100*passed//total}%)[/bold]")


if __name__ == "__main__":
    run_eval()
