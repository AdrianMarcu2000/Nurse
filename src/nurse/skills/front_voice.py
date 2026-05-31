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

    def _with_findings(self, context: Context) -> str:
        """Fold fresh skill findings into the RAG-context slot so the Front Voice can
        reference them ('I can see…'). Stale findings (older than the window) are dropped
        so a late async result can't masquerade as current. Empty → just the RAG context."""
        findings = context.findings_text(max_age_s=self._finding_max_age_s)
        if not findings:
            return context.rag_context
        block = f"## What you are currently observing\n{findings}"
        return f"{context.rag_context}\n\n{block}" if context.rag_context else block
