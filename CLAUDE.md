# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, fully-offline voice assistant for an AI nurse robot ("Aria"): voice in → ASR → safety/RAG/LLM → TTS → voice out. Runs on Apple Silicon (MLX) for development and on a Jetson Orin NX (llama.cpp + CUDA) in production. No network calls at runtime — all models run locally.

## Commands

Install (Mac dev): `pip install -e ".[mac]"` — the `mac` extra adds `mlx-lm`, which switches the LLM to the MLX backend automatically.

Download model weights (once, before first run): `bash scripts/download_models.sh`, then build the RAG index: `python scripts/build_rag_index.py`.

Run the assistant: `nurse run` (or `nurse run -p <patient_id> -v --no-rag`). Other CLI commands: `nurse devices` (list audio I/O devices), `nurse bench` (end-to-end latency without a mic).

Tests: `pytest tests/` — run one with `pytest tests/test_safety.py::test_safety_filter`. The behavioral eval is a separate script, not a pytest target: `python tests/eval/run_eval.py` (loads the real LLM, scores scenarios in `tests/eval/scenarios.yaml`).

Jetson deploy: `deploy/jetson/install.sh` builds `llama-cpp-python` with `-DGGML_CUDA=on`, installs a `nurse.service` systemd unit, and downloads models.

No linter/formatter/type-checker is configured. There is no `[tool.*]` config in `pyproject.toml`.

## Architecture

**Turn pipeline** (`src/nurse/pipeline.py`) is the spine. One user utterance flows: ASR → RAG retrieval + long-term memory load (run in parallel via a `ThreadPoolExecutor`) → `build_messages` → **safety check** → LLM stream → sentence-buffered TTS → playback. `main.py` drives it: `MicrophoneStream` yields complete utterances (Silero VAD detects end-of-turn), each is fed to `pipeline.process_audio`. On SIGINT, `end_session()` asks the LLM to summarize the transcript into long-term memory.

**Latency is the central design constraint.** Three patterns recur and should be preserved:
- Streaming all the way down: LLM tokens are accumulated into *sentences* (`_stream_llm_and_speak`), and each completed sentence is sent to TTS and played immediately, so the patient hears audio before the full response is generated.
- The mic is **muted while the robot speaks** (`pipeline._speak_text` → `mic.mute()/unmute()`) to prevent the speaker output from being captured as input. `unmute()` also drains the queue to discard echo.
- Heavy model loads are `@lru_cache`'d and lazily imported (see ASR, TTS, RAG, both LLM backends). `main.py` defers imports so CLI startup is fast.

**LLM backend selection is automatic** (`src/nurse/llm/client.py`). If `mlx.core` imports, use `MLXBackend`; otherwise `LlamaBackend` (llama.cpp). All callers go through `LLMClient.stream_response` and never import a backend directly. Both backends implement the same abstract `LLMBackend` (`backends/base.py`). The model is Qwen2.5-Instruct in both cases (different sizes/quantizations per platform — see `config/default.yaml`).

**Tool calling is parsed from text, not native.** `backends/tools.py` defines OpenAI-format schemas, but the MLX backend extracts calls by regex from `<tool_call>…</tool_call>` tags (or raw JSON) the model emits, dispatches them, appends results as `role: tool` messages, and re-enters generation (up to 3 rounds). When streaming, `_split_safe` holds back any trailing text that might be the start of a `<tool_call>` tag so partial tags never reach TTS. The four tools (`log_vital`, `set_reminder`, `escalate_to_human`, `lookup_patient_profile`) are dispatched by `ToolDispatcher`, which appends to per-patient JSONL files in `data/patient_profiles/`.

**Safety is a deterministic pre-LLM gate, not LLM judgment** (`src/nurse/safety/filter.py`). User text is regex-matched against `safety.escalation_keywords` in config *before* the LLM runs. On a hit, `check_and_escalate` injects a pre-formed `escalate_to_human` tool-call + result into the message list AND the pipeline dispatches the escalation itself and speaks a canned message — bypassing the LLM entirely for emergencies. When editing safety, keep the keyword list authoritative; do not move emergency detection into the model.

**Config** (`src/nurse/config.py`): `config/default.yaml` is the base; `config/local.yaml` (gitignored) is deep-merged over it. `config/persona.yaml` holds the system prompt, greeting, and escalation message (loaded via `get_persona()`). Both loaders are `@lru_cache`'d. Use `resolve(rel_path)` to get paths relative to project root.

**RAG** (`src/nurse/rag/`): protocol markdown in `data/protocols/` is chunked and embedded (BGE) into a LanceDB table. `retrieve()` returns a formatted string injected into the system prompt; it fails *soft* (returns `""`) if the index is missing or no chunk clears the score threshold. `build_index()` is also called at startup unless `--no-rag`.

**Memory**: `SessionMemory` is in-process conversation history; `LongTermMemory` is a per-patient `*_summary.json` regenerated by the LLM at session end and injected into future system prompts.

## Conventions

- All runtime/generated patient data (`*_vitals.jsonl`, `*_reminders.jsonl`, `*_alerts.jsonl`, `*_summary.json`), the LanceDB dir, model weights, and `config/local.yaml` are gitignored.
- Modules use `from __future__ import annotations` and standard `logging` (configured centrally in `main.py`).
- Tests insert `src/` onto `sys.path` directly (the package isn't assumed installed in the test env).
