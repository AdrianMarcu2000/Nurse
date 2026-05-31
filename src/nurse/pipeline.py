"""
Core conversation pipeline — one turn at a time.

Flow per turn:
  1. Transcribe audio (ASR)
  2. Safety check
  3. RAG retrieval
  4. Build messages (system + history + user)
  5. Stream LLM response (with tool dispatch)
  6. Buffer complete sentences → stream to TTS → play audio
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from datetime import datetime

import numpy as np

from nurse.audio.output import Speaker
from nurse.asr.whisper_stream import transcribe
from nurse.config import get_config, get_persona
from nurse.llm.client import LLMClient
from nurse.llm.prompt import build_messages
from nurse.llm.tools import ToolDispatcher
from nurse.memory.longterm import LongTermMemory
from nurse.memory.session import SessionMemory
from nurse.rag.retrieve import retrieve
from nurse.safety.filter import check_and_escalate
from nurse.tts.kokoro_stream import synthesize_sentences, SAMPLE_RATE

logger = logging.getLogger(__name__)


class NursePipeline:
    def __init__(self, patient_id: str = "default") -> None:
        self.patient_id = patient_id
        self.speaker = Speaker()
        self.session = SessionMemory()
        self.long_term = LongTermMemory(patient_id)
        self.dispatcher = ToolDispatcher(patient_id)
        self.llm = LLMClient(self.dispatcher)
        self._transcript_log: list[str] = []
        self._mic = None  # set via set_mic() after MicrophoneStream is created
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # Aria can only do one thing at a time — a reactive turn OR a proactive
        # engagement. Both acquire this lock so she never talks over herself or hears
        # her own voice. Proactive attempts use a non-blocking acquire and defer if busy.
        self._busy = threading.Lock()
        # Most recent time Aria and the patient interacted, for interval check-ins.
        self._last_interaction: datetime | None = None

    def set_mic(self, mic) -> None:
        """Wire the mic so the pipeline can mute it while speaking."""
        self._mic = mic

    def greet(self) -> None:
        """Generate and speak a history-aware greeting at session start.

        References the most recent prior session (if any) so Aria opens by picking up
        where she left off — e.g. "Good morning, John. Yesterday your cough was
        bothering you — how are you feeling?". Falls back to the static persona
        greeting only when there is no history (first-ever session) or generation
        yields nothing.
        """
        persona = get_persona()
        name = get_config()["patient"]["name"]
        latest = self.long_term.latest()
        facts = self.long_term.patient_facts()

        # Always address the patient by name. Static fallback is used on a first-ever
        # session (no history) or if generation yields nothing.
        fallback = f"Hello, {name}! I'm Aria, your nurse assistant. How are you feeling today?"

        if not latest and not facts:
            # No history at all — nothing to personalize from beyond the name.
            self._speak_text(fallback)
            self._transcript_log.append(f"Aria: {fallback}")
            self._last_interaction = datetime.now()
            return

        ctx_parts = []
        if facts:
            ctx_parts.append("Known facts: " + ", ".join(f"{k}={v}" for k, v in facts.items()))
        if latest:
            ctx_parts.append(
                f"Last session ({latest.get('date', 'recently')}): "
                f"{latest.get('clinical', '')} {latest.get('rapport', '')}".strip()
            )
        context = "\n".join(ctx_parts)

        # Put the actual name inline in the instruction (never as a labeled "name:" slot)
        # and show the literal opening — small models otherwise tend to parrot the slot
        # label and say "Hello, patient name!" instead of resolving it.
        messages = [
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user", "content": (
                f'Write a greeting that begins exactly with "Hello, {name}!" then, in '
                "one or two warm spoken sentences, refers naturally to ONE relevant "
                "thing from the last session and asks how they are. Use the literal "
                f"name {name} — never the words \"patient name\". Do not give medical "
                "advice or invent anything.\n\n"
                f"Context about {name}:\n{context}"
            )},
        ]
        greeting = "".join(self.llm.stream_response(messages)).strip()
        if not greeting or self._leaked_placeholder(greeting):
            logger.warning("Greeting leaked a placeholder or was empty; using fallback. Got: %r", greeting)
            greeting = fallback
        logger.info("Greeting: %s", greeting)
        self._speak_text(greeting)
        self._transcript_log.append(f"Aria: {greeting}")
        self._last_interaction = datetime.now()

    @staticmethod
    def _leaked_placeholder(text: str) -> bool:
        """True if the model left an unresolved name slot, e.g. '[Patient Name]',
        '{name}', '<name>', or the literal phrase 'patient name'. We never want to
        speak any of these aloud."""
        import re
        lowered = text.lower()
        if "patient name" in lowered or "preferred name" in lowered:
            return True
        # Any bracketed/braced placeholder token.
        return bool(re.search(r"[\[{<][^\]}>]*\b(name|patient)\b[^\]}>]*[\]}>]", lowered))

    def last_interaction(self) -> datetime | None:
        """When Aria and the patient last interacted — used by interval check-ins."""
        return self._last_interaction

    def engage(self, engagement) -> str:
        """Proactively raise something with the patient, ask-to-engage style.

        Returns one of:
          "delivered"   — patient agreed and the engagement ran
          "declined"    — patient said no / not now
          "no_response" — no reply within the engage timeout (scheduler should retry)
          "busy"        — Aria was mid reactive turn; deferred to a later tick

        Acquires the busy-lock non-blocking so a reactive turn always wins.
        """
        from nurse.proactive.engage import classify_engage_reply

        if not self._busy.acquire(blocking=False):
            logger.debug("Engage skipped — Aria is busy")
            return "busy"
        try:
            persona = get_persona()
            pro = persona["proactive"]
            timeout = get_config()["proactive"]["engage_response_timeout_seconds"]

            # 1. Ask permission.
            self._speak_text(get_config()["proactive"]["engage_prompt"])

            # 2. Listen for one reply, bounded by the engage timeout. Silence (None)
            #    is a no-response, distinct from a spoken decline.
            reply_audio = self._mic.listen_once(timeout=timeout) if self._mic else None
            if reply_audio is None:
                logger.info("Engage (%s) — no response within %ss", engagement.kind, timeout)
                return "no_response"

            reply = transcribe(reply_audio)
            verdict = classify_engage_reply(reply)
            logger.info("Engage (%s) reply=%r → %s", engagement.kind, reply, verdict)

            if verdict != "yes":
                self._speak_text(pro["declined"])
                self._last_interaction = datetime.now()
                return "declined"

            # 3. Deliver the proactive message, then run a normal listening turn so the
            #    patient can respond and Aria can help / log / escalate as usual. The
            #    follow-up turn blocks normally — he's already engaged.
            message = pro[engagement.kind].format(detail=engagement.detail)
            self._speak_text(message)
            self._transcript_log.append(f"Aria (proactive {engagement.kind}): {message}")
            self._last_interaction = datetime.now()

            turn_audio = self._mic.listen_once() if self._mic else None
            if turn_audio is not None:
                # Reuse the full reactive pipeline (already holding the lock).
                self._process_audio_locked(turn_audio)
            return "delivered"
        finally:
            self._busy.release()

    def process_audio(self, audio: np.ndarray) -> None:
        """Process one user utterance end-to-end."""
        # A reactive turn takes priority; block until any in-flight engagement frees
        # the device, so we never run two conversations at once.
        with self._busy:
            self._process_audio_locked(audio)

    def _process_audio_locked(self, audio: np.ndarray) -> None:
        t_start = time.perf_counter()

        # 1. ASR
        user_text = transcribe(audio)
        if not user_text:
            return

        t_asr = time.perf_counter()
        self._last_interaction = datetime.now()
        self._transcript_log.append(f"Patient: {user_text}")

        # 2. Fire RAG + long-term memory load in parallel while safety check runs
        rag_future = self._executor.submit(retrieve, user_text)
        summary_future = self._executor.submit(self.long_term.load_for_prompt)

        patient_summary = summary_future.result()
        rag_context = rag_future.result()
        t_rag = time.perf_counter()
        messages = build_messages(
            self.session.get_history(),
            user_text,
            patient_summary=patient_summary,
            rag_context=rag_context,
        )
        escalated, messages = check_and_escalate(user_text, messages)

        if escalated:
            # Dispatch the escalation tool ourselves too (writes the alert log)
            self.dispatcher.dispatch("escalate_to_human", {
                "reason": f"Emergency keyword in: {user_text}",
                "urgency": "immediate",
            })
            # Deliver the canned escalation message immediately
            escalation_msg = get_persona()["escalation_message"]
            self._speak_text(escalation_msg)
            self.session.add_user(user_text)
            self.session.add_assistant(escalation_msg)
            self._transcript_log.append(f"Aria: {escalation_msg}")
            return

        # 3. LLM stream + TTS
        t_llm_start = time.perf_counter()
        full_response, timing = self._stream_llm_and_speak(messages)
        t_end = time.perf_counter()

        # Break the turn into stages so the bottleneck is unambiguous. LLM and TTS
        # interleave (each sentence is spoken as it completes), so these are the
        # cumulative wall-clock spent generating vs synthesizing+playing, plus the
        # all-important time-to-first-audio the patient actually perceives.
        logger.info(
            "Turn latency — ASR: %.0fms | RAG+mem: %.0fms | "
            "first-audio: %.0fms | gen: %.0fms | TTS: %.0fms | total: %.0fms",
            (t_asr - t_start) * 1000,
            (t_rag - t_asr) * 1000,
            (timing["first_audio"] - t_llm_start) * 1000 if timing["first_audio"] else -1,
            timing["gen_ms"],
            timing["tts_ms"],
            (t_end - t_start) * 1000,
        )

        # 4. Update session memory
        self.session.add_user(user_text)
        self.session.add_assistant(full_response)
        self._transcript_log.append(f"Aria: {full_response}")

    def _stream_llm_and_speak(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Stream tokens from LLM. Accumulate into sentences.
        As each sentence completes, synthesize and play immediately.
        Returns (full_response, timing) where timing has keys:
          first_audio — perf_counter when the first sentence began speaking (or None)
          gen_ms      — cumulative ms spent waiting on LLM tokens
          tts_ms      — cumulative ms spent synthesizing + playing audio
        """
        sentence_buffer = ""
        full_response = ""
        timing = {"first_audio": None, "gen_ms": 0.0, "tts_ms": 0.0}

        import re
        sentence_end = re.compile(r'[.!?]\s')

        def speak(text: str) -> None:
            if timing["first_audio"] is None:
                timing["first_audio"] = time.perf_counter()
            timing["tts_ms"] += self._speak_text(text)

        t_tok = time.perf_counter()
        for token in self.llm.stream_response(messages):
            timing["gen_ms"] += (time.perf_counter() - t_tok) * 1000
            sentence_buffer += token
            full_response += token

            # Check if we have at least one complete sentence
            if sentence_end.search(sentence_buffer) or len(sentence_buffer) > 200:
                # Find the last sentence boundary to keep partial tail
                parts = sentence_end.split(sentence_buffer)
                if len(parts) > 1:
                    # Everything except the last partial fragment
                    to_speak = sentence_end.sub(
                        lambda m: m.group(0),
                        sentence_buffer[:sentence_buffer.rfind(parts[-1])]
                    ).strip()
                    sentence_buffer = parts[-1]

                    if to_speak:
                        speak(to_speak)
            t_tok = time.perf_counter()

        # Speak any remaining buffer
        if sentence_buffer.strip():
            speak(sentence_buffer.strip())

        return full_response.strip(), timing

    def _speak_text(self, text: str) -> float:
        """Synthesize text and play it through the speaker. Mutes mic during playback.
        Returns the wall-clock ms spent synthesizing + playing."""
        t0 = time.perf_counter()
        if self._mic:
            self._mic.mute()
        try:
            for audio_chunk, sentence in synthesize_sentences(text):
                logger.debug("Speaking: %s", sentence)
                self.speaker.play(audio_chunk, sample_rate=SAMPLE_RATE)
        finally:
            if self._mic:
                self._mic.unmute()
        return (time.perf_counter() - t0) * 1000

    def end_session(self) -> None:
        """Update long-term memory with a dated summary of this session."""
        # Need at least one patient turn — a transcript with only Aria's greeting
        # has nothing worth remembering.
        if not any(line.startswith("Patient:") for line in self._transcript_log):
            return
        from datetime import date
        transcript = "\n".join(self._transcript_log)
        try:
            self.long_term.update_from_session(transcript, self.llm, date.today().isoformat())
            logger.info("Session summary saved.")
        except Exception as e:
            logger.warning("Failed to save session summary: %s", e)
