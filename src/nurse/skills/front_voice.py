"""The Front Voice — the one model that ever speaks.

For Step 3 this is exactly today's behavior behind the `FrontVoice` interface: build the
clinical messages from the Context (system prompt + history + patient summary + RAG) and
stream tokens from the LLM. Step 4 extends the persona to carry companion/comfort talk;
later steps weave skill findings into `Context.findings` so the Front Voice references
them in its own words.
"""
from __future__ import annotations

from collections.abc import Iterator

from nurse.config import get_config
from nurse.llm.client import LLMClient
from nurse.llm.prompt import build_messages
from nurse.skills.base import Context, FrontVoice


class LLMFrontVoice(FrontVoice):
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._finding_max_age_s = (
            get_config().get("skills", {}).get("finding_max_age_s", 120.0)
        )

    def respond(self, context: Context) -> Iterator[str]:
        messages = build_messages(
            context.history,
            context.user_text,
            patient_summary=context.patient_summary,
            rag_context=self._with_findings(context),
        )
        yield from self.llm.stream_response(messages)

    def continue_with(self, context, already_said, new_findings):
        """A brief spoken follow-up adding only the NEW info from late findings. Framed as
        a continuation of what was already said, so it never re-answers the question."""
        if not new_findings:
            return
        findings_text = "\n".join(f"- ({f.source}) {f.summary}" for f in new_findings)
        instruction = (
            "You just told the patient:\n"
            f'"{already_said}"\n\n'
            "A moment later, more detail came in:\n"
            f"{findings_text}\n\n"
            "Add a short, natural spoken follow-up (one or two sentences) that shares ONLY "
            "what's genuinely new or important from that detail. Continue the conversation "
            "as if you'd just remembered to mention it (e.g. \"Oh — and …\"). Do NOT repeat "
            "what you already said. If the detail adds nothing new, say nothing."
        )
        messages = build_messages(
            context.history,
            instruction,
            patient_summary=context.patient_summary,
            rag_context=context.rag_context,
        )
        yield from self.llm.stream_response(messages)

    def _with_findings(self, context: Context) -> str:
        """Fold fresh skill findings into the RAG-context slot so the Front Voice can
        reference them ('I can see…'). Stale findings (older than the window) are dropped
        so a late async result can't masquerade as current. Empty → just the RAG context."""
        findings = context.findings_text(max_age_s=self._finding_max_age_s)
        if not findings:
            return context.rag_context
        block = f"## What you are currently observing\n{findings}"
        return f"{context.rag_context}\n\n{block}" if context.rag_context else block
