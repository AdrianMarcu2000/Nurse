# Plan: Proactive, always-on Aria (triggers + intervals, quiet-hours aware)

## Context

Today Aria is purely **reactive**: [main.py](../src/nurse/main.py) runs
`async for audio in mic: pipeline.process_audio(audio)` — nothing happens until the
patient (Octavian) speaks. The goal is an **always-on assistant that also initiates**:
it speaks to Octavian on triggers or at intervals, but **not while he's asleep**, and
it **asks permission before engaging** ("Octavian, is now a good time?").

This is an additive control loop beside the reactive one. The turn pipeline
(`process_audio`, `_speak_text`, the LLM-spoken-message pattern in `greet()`) is
**reused unchanged** — the new code only decides *when* Aria should speak and routes a
prompt into the existing pipeline.

## Decisions (confirmed with user)

- **Triggers (all four)**: scheduled reminders, interval check-ins, memory-driven
  follow-ups, sensor/vitals thresholds.
- **Sleep**: fixed quiet-hours window in config (deterministic, no hardware).
- **Style**: ask-to-engage — Aria asks if it's a good time, proceeds only on a yes.
- **Med override**: medication reminders fire even during quiet hours; check-ins and
  follow-ups are suppressed in the window.
- **Vitals source**: stub `{patient}_vitals_feed.jsonl` + configurable thresholds now;
  real sensor drops into the same seam later.

## Architecture

```
main.py: asyncio runs TWO coroutines concurrently
  • mic loop      → pipeline.process_audio()        (reactive, exists)
  • scheduler.run() → pipeline.engage(prompt, ctx)  (proactive, NEW)

ProactiveScheduler (NEW) ticks every ~30–60s:
  1. QuietHours.active(now)? → skip non-override triggers
  2. poll trigger sources, collect any that are "due"
  3. highest-priority due trigger → pipeline.engage():
       a. acquire the single Aria-busy lock (mutually exclusive with reactive turns)
       b. ask "Octavian, is now a good time?", open mic, classify yes/no
       c. yes → speak the trigger's message and run a normal listening turn
          no/silence → back off; re-arm the trigger for later
```

### Concurrency (the crux)
Reactive turns and proactive engagements must never overlap (Aria can't talk over
herself or hear her own voice). Add **one `threading.Lock` ("Aria busy")** on the
pipeline; both `process_audio` and `engage` acquire it (non-blocking try — if busy,
the proactive attempt is deferred to the next tick). Reuse the existing
`mic.mute()/unmute()` discipline ([input.py:38-50](../src/nurse/audio/input.py#L38-L50)).

## New files

- `src/nurse/proactive/scheduler.py` — `ProactiveScheduler`: async `run()` tick loop,
  polls triggers, applies quiet hours + priority, calls `pipeline.engage()`.
- `src/nurse/proactive/triggers.py` — a `Trigger` protocol (`due(now) -> Engagement |
  None`) and four implementations:
  - `DueReminders` — reads `{patient}_reminders.jsonl` (written today by
    [tools.py:116-128](../src/nurse/llm/tools.py#L116-L128) but **never read**), fires
    when `scheduled_time` ≤ now and not yet `acknowledged`. Marks acknowledged after.
    `overrides_quiet_hours = True`.
  - `IntervalCheckIn` — fires when `now - last_interaction ≥ interval`. Suppressed in
    quiet hours.
  - `MemoryFollowUp` — uses `long_term.latest()` ([longterm.py](../src/nurse/memory/longterm.py))
    to follow up on the last session's topic; rate-limited to once/day. Suppressed in
    quiet hours.
  - `VitalsThreshold` — polls `{patient}_vitals_feed.jsonl`; fires when a reading
    crosses a configured bound. `overrides_quiet_hours = True` (clinical).
- `src/nurse/proactive/quiet_hours.py` — `QuietHours.active(now)` for a wrap-around
  window (e.g. 22:00–07:00).
- `src/nurse/proactive/engage.py` (or a method on the pipeline) — the ask-to-engage
  flow + a deterministic yes/no classifier modeled on
  [safety/filter.py](../src/nurse/safety/filter.py)'s keyword approach.

## Modified files

- `src/nurse/pipeline.py` — add `engage(engagement)`: acquire busy-lock, ask-to-engage,
  on yes speak the message and run a listening turn (reusing `_stream_llm_and_speak` /
  `process_audio` internals). Track `last_interaction` timestamp. Add the busy-lock and
  have `process_audio` acquire it too.
- `src/nurse/main.py` — run the scheduler coroutine alongside the mic loop in the
  existing event loop; pass the pipeline in. Graceful shutdown of both on SIGINT.
- `config/default.yaml` — new `proactive:` block:
  ```
  proactive:
    enabled: true
    tick_seconds: 45
    quiet_hours: { start: "22:00", end: "07:00" }
    check_in_interval_minutes: 120
    follow_up_max_per_day: 1
    engage_prompt: "Octavian, is now a good time to talk?"
    vitals_thresholds:
      heart_rate:        { min: 45, max: 120 }
      oxygen_saturation: { min: 92, max: 100 }
      blood_glucose:     { min: 70, max: 250 }
  ```
- `config/persona.yaml` — add proactive message templates (check-in opener,
  follow-up framing) consistent with the existing greeting/escalation tone.

## Reuse (don't reinvent)
- The reminders JSONL already exists and is write-only today — `DueReminders` finally
  consumes it.
- `_speak_text` + the LLM-spoken-message pattern from the new `greet()` — proactive
  messages are generated/spoken the same way.
- `mic.mute()/unmute()` for talk/listen arbitration.
- Deterministic keyword matching (safety/filter.py) for the yes/no engage classifier.
- `long_term.latest()` for memory follow-ups.

## Verification
1. **Unit (no audio)**: `tests/test_proactive.py` —
   - `QuietHours.active()` across the wrap-around boundary and the med-override path.
   - each `Trigger.due()` with fixture JSONL files (a due vs not-yet reminder; an
     out-of-range vs normal vitals reading; interval elapsed vs not).
   - yes/no classifier on "yes/sure/okay" vs "no/not now/later" vs silence.
2. **Scheduler dry-run**: a `--proactive-dry-run` mode that logs which trigger *would*
   fire each tick (with a fake clock) without speaking — confirms scheduling/priority/
   quiet-hours logic on the bench.
3. **Live**: `nurse run` with a reminder seeded for ~1 min out and a vitals_feed line
   out of range → confirm Aria asks to engage, respects "not now", and that a check-in
   during quiet hours stays silent while a med reminder overrides. Read `logs/nurse.log`.

## Open risks / caveats
- **Audio device contention** is the real hazard; the single busy-lock + try-acquire is
  the mitigation, but live testing (which I can't do) is needed to confirm no overlap.
- **last_interaction** must update on BOTH reactive turns and proactive engagements, or
  check-ins will mis-fire.
- Quiet-hours uses local system time; no timezone handling beyond that.
