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


def run_eval(model: str | None = None) -> None:
    scenarios_path = Path(__file__).parent / "scenarios.yaml"
    scenarios = yaml.safe_load(scenarios_path.read_text())["scenarios"]

    # Honor --model so the eval runs (and reports) the model you intend — a silently
    # ignored flag previously made scores ambiguous. Reuses nurse run's alias mapping.
    from nurse.main import _apply_model_override
    _apply_model_override(model)

    from nurse.config import get_config
    effective_model = get_config()["llm"]["mlx_model"]
    console.print(f"[bold]Eval model:[/bold] {effective_model}")

    from nurse.pipeline import NursePipeline

    # Drive the REAL pipeline (safety gate → orchestrator → Front Voice → tool dispatch)
    # via process_text, with the speaker stubbed so nothing is played and we can capture
    # what Aria would say. This tests what actually ships, not the bare model.
    pipeline = NursePipeline(patient_id="eval_patient")

    spoken_chunks: list[str] = []
    pipeline.speaker.play = lambda *a, **k: None          # don't actually play audio
    # Capture spoken text at the synth boundary (covers greeting, reply, escalation).
    import nurse.pipeline as _pl
    _orig_synth = _pl.synthesize_sentences
    def _capture_synth(text):
        spoken_chunks.append(text)
        # Yield a single dummy (audio, sentence) so the play loop runs without real TTS.
        import numpy as _np
        yield (_np.zeros(1, dtype=_np.float32), text)
    _pl.synthesize_sentences = _capture_synth

    # Track tool dispatches (incl. escalation, which the pipeline dispatches itself).
    calls_made: list[str] = []
    _orig_dispatch = pipeline.dispatcher.dispatch
    def _tracking_dispatch(name, args, _orig=_orig_dispatch):
        calls_made.append(name)
        return _orig(name, args)
    pipeline.dispatcher.dispatch = _tracking_dispatch     # type: ignore

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

        # Reset capture for this scenario and run the full pipeline turn.
        spoken_chunks.clear()
        calls_made.clear()
        pipeline.process_text(user_text)

        response = " ".join(c.strip() for c in spoken_chunks).strip()
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
    import argparse

    parser = argparse.ArgumentParser(description="Run the offline nurse eval.")
    parser.add_argument("--model", "-m", default=None,
                        help="Front Voice model: 1.5b | 3b | 7b (or a raw model id). "
                             "Defaults to the configured model.")
    # parse_args() errors loudly on unknown flags — no more silently-ignored --model.
    args = parser.parse_args()
    run_eval(model=args.model)
