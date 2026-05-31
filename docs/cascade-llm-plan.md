# Plan: Cascade (fast-opener + routing) LLM, with Orin/TensorRT-LLM production path

## Context

`nurse-brain` is a fully-offline voice assistant ("Aria"). It runs on Mac (MLX) for
dev and is intended for a **Jetson AGX Orin 64GB** in production. The user wants two
things, in order:

1. **A multi-model "cascade"**: a small/fast model speaks an immediate opener and
   handles simple turns; a larger model produces the substantive answer for complex
   turns. The router decides *up front* (before any clinical content is spoken), not
   mid-sentence — spoken audio is irreversible, and a self-correcting nurse voice is
   worse than a half-second more latency.
2. **A production port to the Orin** using TensorRT-LLM for the engine speedup and
   cross-turn KV-cache reuse, while keeping llama.cpp/GGUF as a portable fallback.

The design must drop in behind the existing `LLMBackend` seam
([src/nurse/llm/backends/base.py](../src/nurse/llm/backends/base.py)) so that
`NursePipeline` ([src/nurse/pipeline.py](../src/nurse/pipeline.py)) is unchanged: it
already streams tokens → sentence-buffer → TTS, mutes the mic while speaking, and
bypasses the LLM entirely for safety escalations
(pipeline.py:82-94, [src/nurse/safety/filter.py](../src/nurse/safety/filter.py)). The
cascade only ever sees non-emergency turns.

## Goals / non-goals

- **Goal**: opener + up-front routing + big-model continuation, validated on Mac with
  two MLX models, then portable to Orin.
- **Goal**: cheap Orin wins (warmup, power mode) shipped first, independent of cascade.
- **Non-goal**: swapping models *mid-sentence* on text already streamed/spoken. The
  router commits before substantive content. (Confirmed with user across the design
  discussion.)
- **Non-goal**: moving safety into the model — it stays a deterministic pre-LLM gate.

## Build order (each step independently shippable)

### Step 0 — Free Orin/runtime wins (no cascade)
These are correctness/perf gaps that exist today regardless of the cascade.

- **`LlamaBackend.warmup()`** — currently `LLMClient.warmup()` is called in
  main.py:63 but `LlamaBackend` inherits the no-op `warmup()` from base.py:17 (only
  `MLXBackend` overrides it at mlx_backend.py:138). So on the Orin the patient's
  **first turn** pays full model-load + prefill. Add a `warmup()` to
  [src/nurse/llm/backends/llama_backend.py](../src/nurse/llm/backends/llama_backend.py)
  that calls `_load_model()` and runs a 1-token `create_chat_completion` to force
  weight load + CUDA graph warm.
- **Power mode** — add `nvpmodel -m 0` + `jetson_clocks` to Orin startup. Put it in an
  `ExecStartPre=` in [deploy/jetson/systemd/nurse.service](../deploy/jetson/systemd/nurse.service)
  (needs root; either run those lines as a oneshot pre-unit or document the manual step).
  This is the single biggest Orin latency lever and is currently absent.

### Step 1 — `CascadeBackend` (the core feature), Mac-first
New file: `src/nurse/llm/backends/cascade.py`, implementing `LLMBackend`.

Shape (`stream_response(messages)` generator):
1. **Opener**: yield a short opener immediately so TTS starts (~0.6s to first audio).
2. **Route in parallel**: while the opener plays, a deterministic `Router` inspects the
   turn (see Step 1b) and the chosen backend prefills.
3. **Continuation**: yield tokens from the selected backend (small if simple, big if
   complex) as the rest of the turn. Tool-call parsing/dispatch is unchanged — the
   inner backends already do it (mlx_backend.py:107-222, llama_backend.py:45-114); the
   cascade just forwards the generator.

`CascadeBackend.__init__(small: LLMBackend, big: LLMBackend, router: Router)`.
`warmup()` warms **both** inner backends.

Use the existing `ThreadPoolExecutor` pattern already proven in pipeline.py:45 /
pipeline.py:69 for running the router + big-model prefill concurrently with the opener.

#### Step 1a — opener strategy (resolved: canned templated opener)
Generate the opener from a small set of warm, persona-consistent templates (e.g.
"Let me check that for you…", "Okay — one moment…") chosen by the router intent, NOT
by calling the small model. Rationale: zero generation latency, fully deterministic,
no risk of the small model saying something clinical that the big model contradicts.
Store templates next to the persona ([config/persona.yaml](../config/persona.yaml)) so
they stay tonally consistent with the greeting/escalation messages already there.
(If a turn is routed *simple*, skip the opener and let the small model just answer —
openers are only worth it when we're about to wait on the big model.)

#### Step 1b — `Router` (deterministic, mirrors safety/filter.py)
New: `src/nurse/llm/router.py`. A `Router.route(user_text, rag_context, messages) ->
"simple" | "complex"`. Start with deterministic heuristics, modeled on the existing
regex gate in [src/nurse/safety/filter.py](../src/nurse/safety/filter.py):
- complex if: RAG returned context (clinical-protocol question), input length over a
  threshold, presence of "why/how/explain/should I/what does", multi-clause questions.
- simple otherwise (greetings, vitals logging, reminders, acknowledgements).
Thresholds/keywords live in a new `routing:` block in
[config/default.yaml](../config/default.yaml), loaded via the existing `get_config()`
([src/nurse/config.py](../src/nurse/config.py)). Keep it auditable and testable —
no model call in the default router.

#### Step 1c — wire through the factory
In [src/nurse/llm/client.py](../src/nurse/llm/client.py), extend `_build_backend` so
that when `llm.cascade.enabled` is true it constructs two inner backends (reusing the
exact same MLX-vs-llama.cpp selection logic already there) at the configured small/big
model IDs and wraps them in `CascadeBackend`. When disabled, behavior is identical to
today (single backend). This keeps `LLMClient`/pipeline callers untouched.

Config additions to [config/default.yaml](../config/default.yaml) under `llm:`:
```
llm:
  cascade:
    enabled: false          # off by default; opt-in
    small_model_mlx: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    big_model_mlx:   "mlx-community/Qwen2.5-7B-Instruct-4bit"
    small_model_path: "models/qwen2.5-3b-instruct-q4_k_m.gguf"   # Orin GGUF fallback
    big_model_path:   "models/<bigger>-q4_k_m.gguf"
  routing:
    complex_keywords: ["why", "how", "explain", "should i", "what does"]
    complex_min_chars: 80
    route_complex_on_rag_hit: true
```
Both MLX models fit in Mac unified memory; both fit trivially in Orin 64GB, so escalation
never pays a load stall (models stay resident).

> **Note**: `_load_model()` in both backends is currently `@lru_cache(maxsize=1)` with
> no arguments (mlx_backend.py:26, llama_backend.py:18), so two models can't coexist.
> The backends must be parameterized by model id (constructor arg → cache key) so small
> and big load independently. This is the main non-trivial edit to the existing backends.

### Step 2 — Evals for the cascade
Extend [tests/eval/run_eval.py](../tests/eval/run_eval.py) and
[tests/eval/scenarios.yaml](../tests/eval/scenarios.yaml):
- add an expected `route: simple|complex` field per scenario and assert the router's
  decision, so routing is locked before the Orin port.
- confirm safety still bypasses the cascade entirely (the chest-pain scenario must
  never reach either model — assert via the existing dispatcher monkey-patch).
- add a unit test `tests/test_router.py` (pure, no model load) mirroring
  [tests/test_safety.py](../tests/test_safety.py)'s parametrized style.

### Step 3 — `TensorRTLLMBackend` (Orin production engine)
New: `src/nurse/llm/backends/trtllm_backend.py`, implementing `LLMBackend`
(`stream_response` + real `warmup`). Mirrors `LlamaBackend`'s tool-call loop structure
but drives a TensorRT-LLM engine. Selected in `client.py` on the Orin (e.g. a config
flag `llm.engine: "trtllm" | "llama" | "auto"`), with `LlamaBackend`/GGUF kept as the
fallback if the engine isn't built. Add the engine-build step to
[deploy/jetson/install.sh](../deploy/jetson/install.sh) and document the per-model build.
The cascade and router from Steps 1–2 are unchanged — only the inner backend type differs.

### Step 4 — Point Orin cascade at the two TRT-LLM engines
Config: Orin's `local.yaml` sets `cascade.enabled: true` + `engine: trtllm` with the
two compiled engines as small/big. Keep GGUF fallback. Production.

## Critical files
- **New**: `src/nurse/llm/backends/cascade.py`, `src/nurse/llm/router.py`,
  `src/nurse/llm/backends/trtllm_backend.py`, `tests/test_router.py`.
- **Modified**: `src/nurse/llm/client.py` (factory),
  `src/nurse/llm/backends/llama_backend.py` (+warmup, +model-id param),
  `src/nurse/llm/backends/mlx_backend.py` (+model-id param to the cached loaders),
  `config/default.yaml` (+`cascade`/`routing`), `config/persona.yaml` (+opener templates),
  `deploy/jetson/systemd/nurse.service` (+power mode), `deploy/jetson/install.sh`
  (+TRT-LLM build), `tests/eval/run_eval.py` + `tests/eval/scenarios.yaml` (+route assertions).
- **Unchanged**: `src/nurse/pipeline.py`, audio, ASR, TTS, RAG, memory, safety —
  the whole point of using the `LLMBackend` seam.

## Reuse (don't reinvent)
- `LLMBackend` ABC + `LLMClient` factory pattern — extend, don't replace.
- The deterministic-regex approach in `safety/filter.py` — the router copies it.
- `ThreadPoolExecutor` parallelism already in `pipeline.py` — same pattern in cascade.
- Existing tool-call parse/dispatch in both backends — cascade forwards, doesn't redo.
- The eval harness + dispatcher monkey-patch in `run_eval.py` — extend for routing.

## Verification
1. **Unit**: `pytest tests/test_router.py tests/test_safety.py tests/test_tools.py` —
   router decisions + safety still deterministic.
2. **Eval (Mac, real models)**: `python tests/eval/run_eval.py` with
   `cascade.enabled: true` in `config/local.yaml` — every scenario routes as expected,
   chest-pain still bypasses both models, response quality unchanged or better.
3. **Live (Mac)**: `nurse run -v` — confirm you *hear* the opener within ~0.6s on a
   complex turn ("why is my blood pressure high?") and a snappy direct answer on a
   simple turn ("remind me to take metformin at 8pm"). Verbose logs show which model
   handled the continuation and the per-turn latency line already emitted at pipeline.py:101.
4. **Bench**: `nurse bench -t "..."` for LLM/TTS timings without a mic (main.py:98) —
   compare cascade vs single-model.
5. **Orin (Steps 3–4)**: after engine build, `nvpmodel -m 0 && jetson_clocks`, then
   `nurse run` — confirm first turn no longer stalls (warmup), and time-to-first-audio
   on complex turns is lower than the llama.cpp baseline. Re-run the eval on-device.

## Caveats (stated, not hidden)
- The performance multipliers (TRT-LLM ~1.5–2.5× over llama.cpp on Orin; "4–8B is smart
  enough") are reasoned from model families + Orin architecture, **not** benchmarked on
  the user's unit/JetPack. Step 2's eval + Step 5's on-device run are what convert these
  assumptions into evidence before committing the production port.
- TRT-LLM requires a per-model/per-quant/per-JetPack engine build — less portable than
  dropping in a GGUF. The GGUF/`LlamaBackend` fallback is retained for that reason.
