# Plan: Aria Skill Platform — collaborating skills (companion, vision, audio-scene) on a safety-first orchestrator

## Context

Aria today is a single-purpose reactive+proactive **nurse** voice assistant. The user
wants to evolve it into a **companion platform**: it should chat about non-clinical
topics like a companion, and host multiple specialized **skills** — interpreting
images, recognizing background noises/audio scenes, and fusing those with speech to
produce richer responses. Skills should be able to **sense in parallel**, while Aria
still **speaks one thing at a time**.

Existing systems this builds on:
- **LLM layer** (small-opener + big/specialist-continuation behind the `LLMBackend`
  seam) is the generation engine the **nurse** and **companion** skills call. Built as
  Steps L0–L2 below; responders reach it through the existing `LLMClient`
  ([llm/client.py](../src/nurse/llm/client.py)), so the
  platform ships whether the client is a single model or the layered opener+specialist
  setup — no ordering dependency between the two.
- The **proactive scheduler** ([src/nurse/proactive/](../src/nurse/proactive/))
  is the precedent for background/parallel work and the **busy-lock** that already
  serializes speech ([pipeline.py](../src/nurse/pipeline.py)).
- The **deterministic safety gate** ([safety/filter.py](../src/nurse/safety/filter.py))
  stays a pre-everything gate — it runs on every input before any skill, so an emergency
  phrase escalates even mid-casual-chat.

Decisions confirmed with the user:
- **Architecture**: central **Orchestrator + Skill registry** (in-process skills).
- **Vision/audio**: **real modality-specialized model calls** (a VLM for images, an audio
  model for sound), each a per-skill configurable model — not canned stubs. A skill whose
  model isn't installed is disabled gracefully (not a crash). Companion-chat built fully.
- **Safety**: deterministic safety gate runs first, always, regardless of skill.
- **Parallelism**: **parallel sensing, serial speech** — skills analyze concurrently,
  findings are fused, exactly one skill produces the spoken response.
- **Doc**: this plan is saved to `docs/skill-platform-plan.md`.

## Architecture: blackboard-via-orchestrator

There is **one Front Voice** the patient ever hears — a fast on-device conversational
model that owns every turn. It is ready to answer from what it knows almost instantly and
usually opens the turn (within a short ~0.8s window a faster specialist may open instead —
see Turn sequencing), then keeps the conversation going. **Specialists run in the
background** and feed detail *back into the Front Voice's context*; the Front Voice then
continues in its own words. Specialists never speak directly, so a specialist being
consulted is **invisible to the user — one continuous voice.**

```
  audio in ─▶ ASR ─▶ ┌──────────── SAFETY GATE (always first, unconditional) ──────┐
                     │ emergency → escalate NOW (alert team + canned message),      │
                     │ THEN hand to Front Voice to calm the patient (never instead) │
                     └───────────────────────┬──────────────────────────────────────┘
                                             │ (every turn)
                                             ▼
                   ┌──────────── Front Voice (on-device convo model) ───────────────┐
                   │ • ready instantly from what it knows; usually opens the turn    │
                   │   (incl. comfort/chit-chat: history, geography, reassurance)    │
                   │ • submits its stream to the SpeechArbiter                       │
                   └───────────────────────┬──────────────────────────────────────────┘
        in parallel, NOT on the voice path: │
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Orchestrator dispatches background work, results flow BACK to Front  │
   │  • sensing skills (vision/audio) → findings                          │
   │  • specialist skills (cardiac/cold-flu/nurse-detail) → richer detail │
   │  • each writes to shared Context/memory with observed_at             │
   │  → Front Voice continues in its own words ("…looking closer at that")│
   └──────────────────────────────────────────────────────────────────────┘
                                             ▼
              SpeechArbiter (priority queue, deterministic referee)
                       → sentence-buffer → TTS → speaker  (ONE voice)
```

**Front Voice = the conversational responder; specialists = background enrichment.**
- The Front Voice is the fast on-device model (companion-class). For most turns —
  including comfort talk to calm a distressed patient — it simply *is* the answer.
- When a turn needs domain depth, a **specialist works in the background** and its finding
  is **fed back into the Front Voice's context**; the Front Voice continues seamlessly in
  its own words. The specialist's text is never spoken directly — only the Front Voice
  speaks. (Resolved: "front voice continues with the detail.")
- The **SpeechArbiter is the deterministic referee, NOT the Front Voice** — it is control
  logic (priority/staleness/preemption), generates nothing, and must contain no model so
  it can be trusted to gate. The Front Voice is one of its *producers*. They are
  deliberately separate: a referee that is also a talker can't referee.
- The opener model and the companion/general responder are **the same model** — one
  "first responder" hat, not a throwaway filler model.

**Safety stays unconditional and first.** Emergency detection still fires the
deterministic escalation immediately (alert the team + speak the canned escalation) — the
Front Voice's calming conversation happens *after and in addition*, never *instead*.
"Comfort the patient during a stroke" is allowed **only** because escalation has already
fired; the calming path can never gate, soften, or delay the alert.

### Turn sequencing (open-the-turn race, then Front Voice + background enrichment)

The orchestrator conducts; only the Front Voice ever speaks. Within
`orchestrator.handle()`:
1. **Both start in parallel, nothing speaks yet**: the Front Voice begins generating its
   first answer from what it knows (history, memory, persona), and — off the voice path —
   the orchestrator dispatches relevant sensing + **specialist** skills (router domain
   scoring) via the pipeline `ThreadPoolExecutor`.
2. **Open-the-turn race with a ~0.8s deadline** (`skills.open_deadline_s`). The first
   sentence to be ready *opens the turn*:
   - If a **specialist** has its answer ready by the deadline (it beat or tied the Front
     Voice), the **specialist's content opens the turn** — rendered in the **one voice**
     (the Front Voice phrases/relays it, or it's spoken directly through the same TTS), and
     the Front Voice's now-redundant generic draft is discarded. Why: no point speaking a
     vaguer on-device answer when the better specialist answer is already in hand.
   - Else (deadline passes with no specialist ready) the **Front Voice opens**, exactly as
     before. This caps the wait at ~0.8s; the patient never waits longer for an opener.
   - Safety escalations are never subject to this race — they preempt via the arbiter.
3. **After the opener, enrichment continues into the Front Voice's context** (never to the
   speaker directly). A specialist/sensing result that returns *later* is written to shared
   Context/memory and the Front Voice **continues its own stream** ("…and looking more
   closely, your heart rate does look high — I've let the nurse know"). One voice, always.
4. **Returns after the turn ends** surface via the proactive path (`FindingTrigger` →
   Front Voice brings it up) — same mechanism as late sensing/cloud results.

> The race is a *pre-audio* decision: it resolves before anything is spoken (within the
> 0.8s window), so there is never a mid-speech cut-over — consistent with "spoken audio is
> irreversible." Once the opener is chosen and starts playing, the loser is discarded.

### On-device vs cloud (where the work runs — voice stays local)
- The **Front Voice is always on-device** (instant, offline, owns the speaker).
- **On-device specialists** typically return fast enough to be woven into the same turn.
- **Cloud specialists/sensing** (`location: cloud`, opt-in) take seconds and are **always
  async**: the Front Voice carries the conversation now; the cloud result, when it
  returns, is fed back and surfaced by the Front Voice (in-turn if quick, else via
  `FindingTrigger`). The cloud never sits on the voice path. This unifies cloud responders
  and cloud sensing under one async-enrichment mechanism.

### TTS feed contract (how text reaches the speaker — one path)

How TTS decides what to speak today, and the rule the platform preserves:
- **The single driver is `_stream_llm_and_speak`** ([pipeline.py:262](../src/nurse/pipeline.py#L262)):
  it consumes a **stream of text tokens**, accumulates them in a buffer, and emits each
  chunk to `_speak_text` (Kokoro synth + blocking playback) when a **sentence boundary**
  (`[.!?]` + whitespace) completes — not per token, not at the end. The partial tail is
  held back. That sentence-boundary trigger is "how TTS knows when to speak."
- **Contract (confirmed): only the Front Voice yields a plain text-token stream** to TTS.
  The existing sentence-buffer is the **sole** thing that drives TTS; the Front Voice
  never calls TTS directly and emits no structured/event output. One code path, unchanged.
- **Skill findings (sensing OR specialist) are never spoken directly.** A finding reaches
  speech only by being **injected into the Front Voice's prompt/Context**, so the Front
  Voice *generates* it as ordinary tokens ("I can see…", "your heart rate does look
  high"). This is what makes specialists invisible and keeps one voice — no second
  producer races for the speaker.
- **In-turn, the Front Voice's stream is the only feed**; the SpeechArbiter serializes it
  against any safety escalation or proactive surfacing. The blocking `sd.wait()` in
  `_speak_text` is the physical serialization underneath.
- **Late async results are a separate, later utterance** — not the in-turn loop; they
  surface via the proactive path (Front Voice brings them up) when they return.
- Preserve the existing **tool-tag guard** (MLX backend withholds a trailing fragment
  that could begin a `<tool_call>` tag) so partial tool syntax never reaches TTS — this
  lives in the LLM layer and is inherited unchanged.

- **Per-skill models with an explicit location**: every skill (responder *or* sensing)
  declares **`location: on_device | cloud`** and a `model_id` — a deliberate, declared
  routing of that skill to either a local model or a cloud API. This is a first-class
  property, not a loading optimization: e.g. the nurse + opener run **on-device** (offline
  guarantee), a heavy vision skill may run in the **cloud**, a cardiac specialist
  on-device. Two client types behind one interface:
  - `on_device` → the existing `LLMClient` / model-id backend loaders (MLX / llama.cpp /
    TRT-LLM), loaded at startup.
  - `cloud` → a `CloudClient` making an API call; opt-in, off by default, **never on the
    voice path** — cloud sensing and cloud specialists are async enrichment: the Front
    Voice talks now, and the cloud result is fed back / surfaced via the `FindingTrigger`
    → Front Voice path when it returns (see Turn sequencing above).
- **Memory budget = sum of the on-device models only.** Cloud-located skills cost no
  local RAM. On-device models are all loaded at startup (no lazy loading — behavior is
  predictable); the deployer keeps the on-device set within the hardware budget (ample on
  Orin 64GB; pick a smaller on-device set on a 16–32GB Mac, push heavy skills to cloud).
- **Serial speech**: all speech goes through the **SpeechArbiter** (below), which owns
  the single TTS path; the responder still streams through `_stream_llm_and_speak`, but
  the arbiter — not a bare busy-lock — decides *what* speaks and *when*.

## SpeechArbiter — one mouth, consistent voice (resolved)

Only the Front Voice generates speech, but several *occasions* to speak still compete: a
reactive Front-Voice turn, a safety escalation, and proactive surfacing of a returned
finding can all want the speaker. The busy-lock would serialize them but **not** decide
which wins or whether a queued one is still worth saying. So **a single `SpeechArbiter`
owns all speech**, and every occasion is submitted as a **speech intent**; the busy-lock
becomes an internal detail.

`src/nurse/speech/arbiter.py` (new):
- **One queue, one consumer.** Every occasion to speak (Front-Voice turn, safety
  escalation, proactive `FindingTrigger`, cloud-result surfacing) submits a `SpeechIntent
  { text_or_stream, priority, source, created_at, ttl, still_relevant() }` instead of
  calling TTS. A single arbiter loop pulls the highest-priority intent and drives
  `_stream_llm_and_speak`/`_speak_text`. This is the *only* code that touches the speaker.
  (Specialists do NOT submit intents — they enrich the Front Voice, which speaks.)
- **Priority ordering**: safety escalation > reactive Front-Voice turn (patient is owed a
  reply) > proactive finding > idle small-talk. Ties broken by `created_at`.
- **Staleness / superseding**: before speaking a queued intent the arbiter calls
  `still_relevant()` and checks `ttl` against the shared transcript — e.g. an async
  finding queued behind a 20s conversation that the patient already moved past is
  **dropped**, not spoken late. This is the core fix for the async-finding race.
- **Urgent preemption at sentence boundary**: a clinical/safety-grade intent preempts the
  in-progress utterance **at the next sentence boundary** (never mid-word — the
  sentence-buffer already chunks there, so preemption checks the queue between chunks).
  Everything non-urgent waits its turn. *Truly* emergency input still uses the existing
  deterministic safety escalation, which the arbiter treats as top priority.
- **Consistency via shared transcript + neutral opener**: every spoken line is appended to
  the one session transcript/memory, and each producer reads it before generating, so a
  follow-up continues from what was actually said. The **opener stays strictly
  content-free** (commits to nothing — no "I'll answer that" before triage decides), so
  opener → responder → any follow-up always read as one Aria. A proactive finding that
  would clash with the current thread is deferred or dropped by `still_relevant()`.

This replaces the bare busy-lock as the speech guarantor; the lock (or the arbiter's
single-consumer loop) still physically serializes playback underneath.

- **Interruptible playback (for barge-in)**: the arbiter exposes `stop()`. `_speak_text`
  becomes chunked + cancellable — instead of one blocking `sd.play()`/`sd.wait()`, it
  plays short audio chunks and checks a stop flag between them, calling `sd.stop()` on
  interrupt. This is the mechanism barge-in (below) uses to cut Aria off mid-utterance.

## Interruption / barge-in (the patient can talk over Aria)

Today the patient **cannot** interrupt: `_speak_text` mutes the mic while Aria speaks
(echo prevention) and blocks until the sentence finishes — so Aria is deaf mid-utterance.
True voice barge-in (resolved) requires undoing both, carefully:

- **Acoustic Echo Cancellation (AEC)** replaces mic-muting. To listen *while* speaking
  without hearing herself, the input path runs AEC: subtract the known speaker signal from
  the mic input (a reference-signal echo canceller — a real new audio component, e.g. a
  WebRTC/`speexdsp` AEC, fed the TTS output as the reference). VAD then runs on the
  echo-cancelled stream, so only the *patient's* voice triggers. The current
  `mic.mute()/unmute()` ([input.py](../src/nurse/audio/input.py))
  becomes a fallback for when AEC is unavailable (degrade to no-barge-in, not broken).
- **Barge-in detection → arbiter stop**: when VAD-on-AEC detects sustained patient speech
  during playback, the pipeline calls `arbiter.stop()`; chunked playback halts at the next
  chunk (sub-second), the mic captures the new utterance, and it's handled as a **fresh
  turn**.
- **On interrupt — keep the draft, mark it undelivered** (resolved): the in-progress LLM
  response (spoken part + unspoken remainder) is **retained in chat history, flagged
  `delivered: partial`** with what was actually voiced. Aria does **not** re-speak it, but
  the model sees it next turn as "drafted, not fully communicated," giving continuity
  ("you cut in while I was explaining X") without parroting. Any in-flight specialist for
  that turn is cancelled or demoted to a background finding.
- **Safety carve-out**: an in-progress **safety escalation is not interruptible** — it must
  finish alerting. Barge-in stops normal speech only.
- **Effort note**: AEC is the single biggest new piece and is hardware/mic-dependent; the
  arbiter `stop()` + chunked playback + draft-marking are straightforward and can ship
  first (interruptible by a button), with AEC-driven voice barge-in dropped in after.

## Blackboard vs. memory, and recency (resolved)

These are distinct and we keep both, with no new subsystem:
- **The per-turn `Context` IS the blackboard** — an ephemeral scratchpad holding the
  user utterance, history, RAG, and the `SkillFinding`s gathered for *this* turn. Built,
  fused, used by the responder, discarded. No standalone blackboard store.
- **Memory is the durable record** — `SessionMemory`
  ([memory/session.py](../src/nurse/memory/session.py)) for
  the rolling conversation, `LongTermMemory`
  ([memory/longterm.py](../src/nurse/memory/longterm.py))
  for the dated per-session narrative.
- **Promotion bridges them**: after a turn, keep-worthy findings (e.g. "noticed Octavian
  on the floor 14:30", an out-of-range vital) are **promoted** from Context into
  LongTermMemory; transient ones (ambient TV noise) are used and dropped. A small,
  explicit `promote()` step — not a new system.

**Recency** is enforced by giving every `SkillFinding` an **`observed_at` timestamp**
(when the thing happened, not when it was inserted — critical for async/external findings
that return late):
- *Within a turn*: the fuser ranks/weights findings by `observed_at` and drops anything
  older than a configurable staleness window, so the newest observation wins.
- *Across memory*: `load_for_prompt()` already injects only the last few sessions
  (recency-by-truncation); promoted findings carry `observed_at` so a late external
  result is ordered by when it was *observed*, never allowed to masquerade as "now".

## External / async skills (opt-in, off by default)

The project's core rule is **no network calls at runtime**. External-API skills (e.g.
cloud video processing) are therefore an **explicit per-skill opt-in, disabled by
default**, with a privacy note (PHI leaves the device only when a deployer consciously
enables it). The device must remain fully functional offline.

Because external calls are slow and may finish *after* the triggering turn, they are
**asynchronous background sensing**, never on the speech path:
- They run in the background (same pattern as the proactive scheduler), and on
  completion write a timestamped `SkillFinding` to memory.
- **If the finding is important, it becomes a proactive trigger** — reusing the existing
  scheduler ([src/nurse/proactive/](../src/nurse/proactive/)):
  a new `FindingTrigger` fires an engagement, and the **Front Voice** brings it up ("By
  the way, the video review flagged…"). This is why late findings need no synchronous
  fusion — they surface through the proactive path, which already handles ask-to-engage,
  quiet hours, and the speech arbiter.
- Local/fast sensing still fuses synchronously into the current turn's Context.

## New package: `src/nurse/skills/`

- **`base.py`** — small ABCs:
  - `Skill.run(context) -> SkillFinding | None` — a background enrichment skill (sensing
    OR domain specialist). Runs off the voice path; **never speaks** — it returns a
    finding that flows back into the Front Voice's context. Declares `location`
    (`on_device`/`cloud`), `mode` (`sync` = may complete in-turn, `async` = surfaces
    later), `domain_keywords` (so the orchestrator knows when it's worth running), and a
    `model_id`. A `cardiac` specialist and a `vision` sensor are both just `Skill`s — the
    only difference is the model and the prompt.
  - `SkillFinding` dataclass: `source`, `summary`, `data`, `confidence`, **`observed_at`**
    (timestamp the observation refers to — drives recency), `keep` (promote to
    LongTermMemory?).
  - `FrontVoice` — the **one speaking model**. `respond(context) -> Iterator[str]`
    yielding **plain text tokens** (the TTS feed contract): owns an on-device `LLMClient`,
    builds messages from Context (persona + history + memory + any findings so far),
    streams text, dispatches the nurse tools. It is the only thing whose output reaches
    the SpeechArbiter. It can be re-invoked to *continue* when a finding returns.
- **`registry.py`** — `SkillRegistry` holding the Front Voice + the enrichment skills,
  built from config so deployments add/remove specialists/sensors without code. Loads
  every `on_device` model at startup, wires `cloud` skills to a `CloudClient`; a skill
  whose model/endpoint is unavailable is disabled gracefully.
- **`orchestrator.py`** — the **GP**. `handle(user_text)`: (1) start the **Front Voice**
  responding immediately (submitting its stream to the SpeechArbiter); (2) in parallel,
  select relevant enrichment skills by domain scoring and run them off the voice path
  (pipeline `ThreadPoolExecutor`, [pipeline.py:45](../src/nurse/pipeline.py#L45));
  (3) as findings return, feed them into the Front Voice's context so it **continues in
  its own words** — in-turn if quick, else via `FindingTrigger`. After the turn,
  `promote()` keep-worthy findings into LongTermMemory.
- **`speech/arbiter.py`** — the single owner of all speech (see SpeechArbiter section):
  priority queue of `SpeechIntent`s, staleness/`still_relevant()` checks, urgent
  sentence-boundary preemption, one consumer driving `_stream_llm_and_speak`/`_speak_text`.
- **`sensing/external.py`** — base for `cloud`/`async` skills: opt-in, network, runs in
  the background executor, timestamps findings with `observed_at`, routes important
  results back to the Front Voice (in-turn or via `FindingTrigger`). Concrete impls (cloud
  video) make a real API call; tested against a mocked HTTP layer so CI needs no network.
- **`proactive/triggers.py`** (extend) — add `FindingTrigger`: fires an engagement from a
  queued async result (late sensing finding or cloud specialist answer); the **Front
  Voice** brings it up when it returns.
- **`router.py`** — `SkillRouter.select(context) -> list[Skill]`, **domain scoring**: pick
  which enrichment skills are worth running this turn (the Front Voice always runs).
  **Deterministic-first with an optional tiny LLM fallback**: score skills by
  `domain_keywords` ∩ input + RAG/finding signals (mirrors
  [safety/filter.py](../src/nurse/safety/filter.py)); pick
  the clear winners (zero, one, or several skills may be worth running). The LLM tiebreak
  is off the path for clear cases. Keywords/weights live in config — adding a "cardiac" or
  "cold/flu" specialist is a config edit, not a router rewrite. The Front Voice always
  runs regardless; skills only *add* enrichment.
- **`front_voice.py`** (`FrontVoice`) — the one speaking model. Persona spans both nurse
  *and* companion registers: clinical care (RAG + long-term memory + the four existing
  tools, reusing today's [llm/prompt.py](../src/nurse/llm/prompt.py))
  **and** warm non-clinical conversation (history, geography, reassurance — the rapport
  memory) so it can comfort a distressed patient. It's the refactor of today's
  `process_audio` body behind the `respond()` interface, plus the ability to be re-invoked
  to *continue* once a skill finding lands. On-device, fast.
- **Domain specialists** (e.g. `skills/cardiac.py`, `skills/cold_flu.py`) — thin `Skill`s
  (NOT speakers): a domain prompt, a domain `model_id`, domain keywords, domain RAG
  protocols; `run()` returns a `SkillFinding` (its analysis) that the Front Voice weaves
  into speech. **Built as one worked example now** (cardiac OR cold/flu) to prove the
  enrichment pattern end-to-end; the rest are config/prompt additions. All sit behind the
  one safety gate.
- **`sensing/vision.py`** (`VisionSkill`) — makes a **real model call** to a vision
  model chosen for the task, returning a timestamped `SkillFinding`. The model is
  per-skill configurable (`vision_model`), defaulting to a small local **VLM**
  (e.g. Qwen2.5-VL / Moondream) for offline use; a deployer may instead point it at a
  `cloud` vision API (opt-in). It calls through the same model abstraction the Front Voice
  uses, extended for image input (a `VisionClient` analogous to `LLMClient`, or a
  multimodal-capable backend). Image source is a frame grab / a path under `data/sense/`;
  tests inject a fixed image so the call is real but deterministic. `run()` returns a
  finding the Front Voice speaks ("I can see…").
- **`sensing/audio_scene.py`** (`AudioSceneSkill`) — makes a **real model call** to an
  audio-understanding model (`audio_model`) to describe background sound ("a smoke alarm
  is beeping", "someone is crying"), returning a timestamped `SkillFinding`. Offline
  default is a local audio-event/audio-LLM model (e.g. an audio classifier or an
  audio-capable multimodal SLM); cloud is an opt-in alternative. Same per-skill model
  abstraction; tests feed a fixed audio clip.

> **Why real calls, not canned stubs**: each sensing skill is a genuine
> modality-specialized model behind the `SensingSkill` interface, picked per use case —
> a VLM for images, an audio model for sound — exactly mirroring the per-skill-model
> design of the responders. "Pick the right model for each problem" applies to senses as
> well as responders. What stays mock-able is the *input source* and the *network*
> (cloud variants), not the model call itself.

## Integration into the pipeline (minimal, reuse-heavy)

- **`pipeline.py`**: `_process_audio_locked` keeps doing ASR + the **safety gate first**
  (unchanged, [pipeline.py:196-](../src/nurse/pipeline.py#L196)).
  On non-emergency, instead of building messages inline, it calls
  `self.orchestrator.handle(user_text)`, which runs the Front Voice (→ SpeechArbiter) and
  background skills. The Front Voice contains the logic that used to live here, so behavior
  is preserved when no enrichment skills are registered.
- **LLM reuse**: the Front Voice and every skill use the existing `LLMClient`
  ([llm/client.py](../src/nurse/llm/client.py)); once the
  LLM layer (L0–L2) lands, the Front Voice's client *is* the on-device opener/companion
  model. No coupling — the skill platform and the LLM layer compose cleanly.
- **Config** ([config/default.yaml](../config/default.yaml)):
  a `front_voice:` entry + a `skills:` block + `skills.open_deadline_s` (the open-the-turn
  race window, default 0.8). The Front Voice and every skill declare
  **`location: on_device | cloud`** and a `model_id`/endpoint. Each skill also: `enabled`,
  `domain_keywords`, `mode` (`sync` = may complete in-turn, `async` = surfaces later).
  Plus skill timeouts. Adding a skill is a config entry (+ a prompt), not code. Loaded via
  existing `get_config()`. Example:
  ```
  front_voice: {location: on_device, model_id: "...qwen-1.5b...", enabled: true}
  skills:
    open_deadline_s: 0.8        # wait this long for a specialist to open the turn
    registry:
      cardiac: {location: on_device, model_id: "...cardiac-slm...", mode: sync,
                domain_keywords: [heart, chest, palpitations, ...]}
      vision:  {location: cloud, endpoint: "...", mode: async, enabled: false}
      audio:   {location: on_device, model_id: "...audio...", mode: sync}
  ```

## Build order (each step independently shippable)

### LLM layer (multi-model loading + Orin engine; composes with skills)
The Front Voice already provides the "fast first response that then continues" behavior
the old cascade aimed at, so this layer's job narrows to: load multiple models at once,
warm them, and speed them up on Orin.
- **L0 — Free runtime wins**: add `LlamaBackend.warmup()`
  ([llama_backend.py](../src/nurse/llm/backends/llama_backend.py)
  inherits the no-op `warmup()` today) and the Orin `nvpmodel -m 0` + `jetson_clocks`
  in [deploy/jetson/systemd/nurse.service](../deploy/jetson/systemd/nurse.service).
- **L1 — Per-model-id loading** (the change the platform actually needs): parameterize
  `_load_model()` in both backends by model id so the Front Voice + several skill models
  coexist (today both are `@lru_cache(maxsize=1)` with no args, so only one model loads).
  Optional `CascadeBackend` (`src/nurse/llm/backends/cascade.py`) for an *intra-model*
  small→big speedup within a single client, behind `llm.cascade.enabled` — useful but no
  longer the headline, since cross-model "speak now, enrich later" is the Front Voice's
  job. The small classifier the `SkillRouter` LLM-tiebreak uses reuses the Front Voice
  model. All on-device models fit Mac unified memory (Front Voice + 1–2 skills) and
  trivially in Orin 64GB.
- **L2 — `TensorRTLLMBackend`** (`src/nurse/llm/backends/trtllm_backend.py`) for the Orin
  production engine, selected by an `llm.engine` flag, GGUF/`LlamaBackend` as fallback;
  engine-build step added to [deploy/jetson/install.sh](../deploy/jetson/install.sh).
- Cascade eval: extend [tests/eval/scenarios.yaml](../tests/eval/scenarios.yaml)
  with expected `route: simple|complex` and add `tests/test_router.py`.
- **Skills are agnostic to all of L0–L2** — they call `LLMClient`, which *is* the cascade
  once enabled. The skill platform (below) can be built before, after, or alongside.

### Skill platform
1. **Interfaces + registry + Context/SkillFinding** + `Skill` and `FrontVoice` ABCs
   (per-model `LLMClient` by model id) — pure code, unit-tested, no behavior change
   (nothing registered yet).
2. **SpeechArbiter** — the single speech owner: `SpeechIntent`, priority queue, one
   consumer over `_stream_llm_and_speak`/`_speak_text`, staleness/`still_relevant()`
   drop, urgent sentence-boundary preemption. Route the *existing* reactive turn,
   greeting, and proactive engagement through it first (no skills yet) — behavior parity
   with today, but everything now speaks via one arbiter. Unit-tested with a fake clock +
   fake TTS (priority order, stale-drop, preemption point). Foundational.
3. **Refactor current behavior into the `FrontVoice`** behind the orchestrator (which
   submits to the arbiter); no enrichment skills yet. **Goal: identical behavior to
   today** — proven by the existing eval
   ([tests/eval/run_eval.py](../tests/eval/run_eval.py))
   still passing. Riskiest step (moves the spine) so it ships alone.
4. **Front Voice carries general + companion conversation** — extend the FrontVoice
   persona to handle non-clinical/comfort talk (history, geography, reassurance) using the
   rapport memory, not just clinical replies. New persona content + tests that casual
   input gets a warm conversational reply and clinical input still does the clinical thing.
5. **One worked domain specialist as a background `Skill`** (e.g. cardiac OR cold/flu) —
   domain prompt + `location: on_device` + `model_id` + keywords + RAG; `run()` returns a
   finding the Front Voice weaves in. Router selects it on a domain-flavored input.
   Implement the **open-the-turn race** here: with a fast specialist, assert its content
   **opens** the turn (Front Voice draft discarded); with a slow one, the Front Voice opens
   at the deadline and the specialist enriches later — both in **one voice** (the
   specialist never speaks). Confirm the on-device model set fits the hardware budget.
6. **Sensing skills with real models + recency-aware fusion + promotion** — build
   `VisionSkill` (local VLM) and `AudioSceneSkill` (local audio model) as `sync` `Skill`s;
   parallel `run()`, `observed_at` on findings, fuser drops stale ones, findings fed to
   the Front Voice, keep-worthy ones promoted to LongTermMemory. Graceful-disable when a
   model isn't installed. Tests inject a fixed image/clip so the call is real but
   deterministic; CI `skipif` the weight is absent.
7. **Async cloud skills → proactive surfacing** — `CloudClient` + cloud `Skill` base
   (opt-in, off by default), background dispatch, completion writes a timestamped finding;
   add `FindingTrigger` so an important late result (cloud sensing or cloud specialist)
   makes the **Front Voice** bring it up. Real API calls exercised against a mocked HTTP
   layer (no real network in CI). This is where cloud video lands.
8. **Barge-in (a): interruptible playback + arbiter stop** — make `_speak_text` chunked +
   cancellable, add `arbiter.stop()`, retain the interrupted draft in history flagged
   `delivered: partial`, treat new input as a fresh turn, safety escalation
   non-interruptible. Wire a simple trigger first (e.g. a stop key) so the whole stop path
   is testable without AEC. Unit-tested: stop mid-stream halts within one chunk, draft is
   retained-not-respoken, safety intent ignores stop.
9. **Barge-in (b): voice barge-in via AEC** — add `speech/aec.py`, feed TTS output as the
   echo reference, run VAD on the echo-cancelled mic so Aria can listen while speaking;
   sustained patient speech → `arbiter.stop()`. Falls back to mic-mute (no barge-in) when
   AEC is unavailable. Hardware/mic-dependent — the heaviest single piece; lands after the
   stop path proves out. Live-test only (echo behavior can't be unit-tested meaningfully).
10. **(Deferred) heavier/production models** — larger VLMs, a dedicated audio-event model,
    or a specific cloud vision provider, plus per-platform tuning. Separate task;
    hardware/network-dependent.

## Critical files
- **New (skills + speech)**: `src/nurse/skills/{base,registry,orchestrator,router}.py`,
  `src/nurse/skills/front_voice.py`, `src/nurse/skills/{cardiac_or_cold_flu}.py`,
  `src/nurse/skills/sensing/{vision,audio_scene}.py`, `src/nurse/speech/arbiter.py`,
  `src/nurse/speech/aec.py` (echo cancellation for barge-in),
  `tests/test_skills.py`, `tests/test_arbiter.py`.
- **New (LLM layer)**: `src/nurse/llm/backends/cascade.py`, `src/nurse/llm/router.py`,
  `src/nurse/llm/backends/trtllm_backend.py`, `tests/test_router.py`.
- **Modified**: `src/nurse/pipeline.py` (call orchestrator after safety gate; barge-in
  detection → `arbiter.stop()`); `src/nurse/audio/output.py` + `input.py` (chunked
  cancellable playback; AEC-fed VAD-during-playback, mute as fallback);
  `src/nurse/llm/client.py` (LLM-layer factory); `llama_backend.py`/`mlx_backend.py`
  (+warmup, +model-id param); `config/default.yaml` (+`skills`, +`barge_in`,
  +`llm.cascade`/`routing`); `config/persona.yaml` (+companion/comfort register); Jetson
  deploy.
- **Docs**: write `docs/skill-platform-plan.md` (this plan); remove the now-superseded
  `docs/cascade-llm-plan.md`.
- **Unchanged**: ASR, safety filter, memory, proactive scheduler — reused through their
  existing seams.

## Reuse (don't reinvent)
- Deterministic-gate pattern (`safety/filter.py`) → `SkillRouter` skill scoring.
- The model-level `Router` (L1) and the skill-level `SkillRouter` share the same
  deterministic-scoring approach at two altitudes (model selection vs skill selection).
- `ThreadPoolExecutor` on the pipeline → parallel background skills.
- Busy-lock + `_speak_text` → the *physical* serialization the SpeechArbiter sits on top
  of; `_speak_text`'s blocking `sd.wait()` becomes chunked+cancellable for barge-in.
- `mic.mute()/unmute()` → kept as the **no-AEC fallback** (no barge-in) when echo
  cancellation isn't available; AEC replaces it where present.
- Per-session `rapport` memory field → Front Voice comfort/companion register.
- `ToolDispatcher` + four tools → invoked by the Front Voice.
- `LLMClient` + the per-model-id backend loaders → the Front Voice's AND every skill's
  generation engine (extended for image/audio input where needed).

## Verification
1. **Unit** (`tests/test_skills.py`, mirrors test_safety.py style): registry add/lookup;
   `SkillRouter.select` picks the cardiac skill on cardiac-flavored input and nothing on
   chit-chat (Front Voice handles it alone); safety gate still fires before anything
   (emergency input never reaches a skill or the Front Voice's normal path); a specialist
   finding fed to the Front Voice produces a single enriched stream (the specialist never
   speaks). **Open-the-turn race** (fake clock): a specialist ready before the deadline
   opens the turn and the Front Voice draft is discarded; a specialist that misses the
   deadline → Front Voice opens, specialist enriches after; the race resolves *before* any
   audio (no mid-speech cut-over). Sensing skills tested with a **real model call on a
   fixed image/clip** (asserting finding content), `skipif` the weight is absent; a
   graceful-disable test confirms a missing-model skill is skipped, not fatal.
2. **SpeechArbiter** (`tests/test_arbiter.py`, fake clock + fake TTS): higher-priority
   intent speaks before lower; a stale intent (TTL elapsed / `still_relevant()` false) is
   **dropped, not spoken late**; an urgent (safety) intent preempts at the next sentence
   boundary (asserted: between chunks, never mid-chunk); two intents never play
   concurrently. **Barge-in stop**: `arbiter.stop()` halts within one chunk; the
   interrupted draft is retained in history as `delivered: partial` and NOT re-spoken next
   turn; a safety intent ignores `stop()`. (Voice/AEC barge-in itself is live-tested.)
3. **Regression — behavior parity after Step 3**: existing `pytest tests/` stays green
   and `python tests/eval/run_eval.py` scores the same as before the refactor (proves the
   Front Voice == old pipeline; all speech now via the arbiter).
4. **Live (Mac)** `nurse run -v`: clinical input → clinical reply + tools; casual input
   ("tell me about your day") → warm conversational reply — **same voice**; a domain input
   ("my heart is racing") → the Front Voice answers and *continues* with the cardiac
   skill's detail in one voice (the specialist's `model_id` shows in `logs/nurse.log` as
   having run, but never as a separate speaker); an emergency phrase mid-chat → escalation
   fires first, then the Front Voice keeps the patient calm. With mock
   `data/sense/last_image.json`, the Front Voice references the scene ("I can see…").
   Confirm one voice at a time via the log. **Barge-in**: talk over Aria mid-sentence →
   she stops within ~a sentence and listens; confirm she does NOT re-speak the cut-off
   answer next turn but does acknowledge the new input; confirm she does NOT interrupt
   herself (AEC working — no self-trigger from her own voice).
5. **Async/cloud + recency** (Step 7): a cloud skill (real call, mocked HTTP) that
   "completes" after the turn writes a timestamped finding; assert it does NOT alter the
   past reply, that an important one enqueues a `FindingTrigger`, and that the Front Voice
   surfaces it on the next proactive tick. Assert the fuser drops a finding older than the
   staleness window and orders by `observed_at`, so a late result can't masquerade as now.

## Caveats / risks (stated, not hidden)
- **Steps 2–3 move the spine.** All speech moves behind the SpeechArbiter (Step 2) and the
  current logic relocates into the Front Voice (Step 3); the parity eval (Verification #3)
  is the guardrail. Ship these alone, in order.
- **The SpeechArbiter is now the consistency-critical component.** A bug there (wrong
  priority, failing to drop a stale intent, preempting mid-word) is heard by the patient.
  It's small and deterministic by design, and `tests/test_arbiter.py` is the guard — but
  it carries the coherence guarantees the bare busy-lock did not.
- **Barge-in trades the echo guarantee for interruptibility.** Listening while speaking
  removes the mic-mute that prevented Aria hearing herself; **AEC is what stands in for
  it**, and it's hardware/mic-dependent — poor AEC means either self-interruption or
  missed barge-in. That's why it's last, behind a fallback (mic-mute = no barge-in), and
  the only feature that's *purely* live-tested. The stop-path (Step 8) is safe and useful
  on its own (button-triggered) even if AEC (Step 9) is never enabled.
- **External skills break the offline guarantee — deliberately, opt-in.** Disabled by
  default; enabling sends data (possibly PHI) off-device, a conscious deployer decision
  with a privacy note. Core must stay fully functional with the network down.
- **Recency depends on `observed_at` being set correctly.** Async/external findings that
  mis-stamp their observation time could wrongly dominate; the fuser orders by
  `observed_at` and enforces a staleness window, but a skill that lies about its
  timestamp defeats it — so `observed_at` is the skill's responsibility, tested per skill.
- **Blackboard is intentionally ephemeral.** Cross-turn persistence is memory's job via
  `promote()`; if a finding isn't promoted it's gone after the turn (by design).
- **Sensing skills make real model calls — heavier than the SLMs.** A VLM / audio model
  adds load time, RAM, and latency; each sensing skill is gated by its own model being
  installed and **disables gracefully** if absent (sensing is enrichment, never required
  for a turn). Model choice/size is per-platform and Orin-hardware-dependent; what's
  mock-able in tests is the *input* (fixed image/clip) and the *network* (cloud variants),
  not the model call. CI skips a sensing test when its weight isn't present.
- **Router is deterministic first.** Skill misroutes are possible on ambiguous input;
  keywords/weights are tunable in config, and safety is independent of the router so a
  misroute is never a *safety* risk. A misroute to the wrong *specialist* is a quality
  issue (general-nurse fallback covers it), not a safety one.
- **On-device models cost RAM; cloud models cost privacy/availability.** The on-device
  set (opener + nurse + each `on_device` specialist/sensing model) is all loaded at
  startup and must fit the hardware (ample on Orin 64GB; pick a smaller on-device set on a
  16–32GB Mac). The explicit `location` per skill is the lever: push heavy or non-PHI
  skills to `cloud`, keep PHI-bearing and latency-critical skills `on_device`. No lazy
  loading — what's enabled on-device is resident and predictable.
- **Domain SLMs give domain-flavored medical talk — higher stakes than companion chat.**
  Specialists must keep the same hard rules as the nurse persona (never diagnose/
  prescribe; escalate emergencies). The deterministic safety gate still runs first for
  *every* skill, but specialist prompts must inherit the nurse guardrails, and a small
  domain SLM may be *more* prone to overstepping — eval scenarios must cover each
  specialist, not just the general nurse.
- **No live audio testing by me** — fusion and routing logic are unit-tested; the spoken
  experience needs a live run on the user's hardware.
```
