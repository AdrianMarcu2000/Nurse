"""The Front Voice — the one model that ever speaks.

For Step 3 this is exactly today's behavior behind the `FrontVoice` interface: build the
clinical messages from the Context (system prompt + history + patient summary + RAG) and
stream tokens from the LLM. Step 4 extends the persona to carry companion/comfort talk;
later steps weave skill findings into `Context.findings` so the Front Voice references
them in its own words.
"""
from __future__ import annotations

from collections.abc import Iterator

from nurse.llm.client import LLMClient
from nurse.llm.prompt import build_messages
from nurse.skills.base import Context, FrontVoice


class LLMFrontVoice(FrontVoice):
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def respond(self, context: Context) -> Iterator[str]:
        messages = build_messages(
            context.history,
            context.user_text,
            patient_summary=context.patient_summary,
            rag_context=self._with_findings(context),
        )
        yield from self.llm.stream_response(messages)

    @staticmethod
    def _with_findings(context: Context) -> str:
        """Fold any skill findings into the RAG-context slot so the Front Voice can
        reference them ('I can see…'). Empty findings → just the RAG context."""
        findings = context.findings_text()
        if not findings:
            return context.rag_context
        block = f"## What you are currently observing\n{findings}"
        return f"{context.rag_context}\n\n{block}" if context.rag_context else block
