"""MLX backend — Apple Silicon only, significantly faster than llama.cpp on M-series."""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from collections.abc import Generator
from functools import lru_cache
from typing import Any

from nurse.config import get_config
from nurse.llm.backends.base import LLMBackend
from nurse.llm.prompt import build_system_prompt
from nurse.llm.tools import TOOLS, ToolDispatcher

logger = logging.getLogger(__name__)

# Qwen2.5-Instruct emits tool calls inside these tags when using MLX
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Also handle raw JSON function call format some versions emit
_JSON_CALL_RE = re.compile(r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\}', re.DOTALL)


def _default_model_id() -> str:
    return get_config()["llm"].get("mlx_model", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")


@lru_cache(maxsize=None)
def _load_model(model_id: str):
    """Load (model, tokenizer) for a given model id. Cached per id so several models
    (Front Voice + specialists) can stay resident at once."""
    from mlx_lm import load

    logger.info("Loading LLM (MLX) — %s …", model_id)
    return load(model_id)  # returns (model, tokenizer)


@lru_cache(maxsize=None)
def _primed_prefix(model_id: str):
    """
    Pre-fill a KV cache with the static prompt prefix (persona system prompt +
    tool schemas) that is identical on every turn. This is the bulk of the
    ~900-token prompt; prefilling it once collapses per-turn prefill from
    ~1000ms to ~150ms. Returns (prefix_token_ids, master_cache). Cached per model id.

    The master is never mutated during generation — each turn deep-copies it
    (and trims the copy to the longest matching prefix) so generation state
    never corrupts the cached prefix.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = _load_model(model_id)
    # Render only the static part: persona system prompt (no per-turn summary /
    # RAG) + tool schemas, with no generation prompt appended.
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": build_system_prompt()}],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix_ids = tokenizer.encode(prefix_text)

    master = make_prompt_cache(model)
    t0 = time.perf_counter()
    model(mx.array(prefix_ids)[None], cache=master)
    mx.eval([c.state for c in master])
    logger.info(
        "MLX prefix cache primed: %d tokens in %.0fms",
        len(prefix_ids), (time.perf_counter() - t0) * 1000,
    )
    return prefix_ids, master


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _format_prompt(messages: list[dict[str, Any]], tokenizer) -> str:
    """Apply the model's chat template, injecting tool schemas."""
    # Some chat templates (e.g. Qwen2.5) concatenate message content directly and crash
    # on a None content. Normalize any None content to "" before rendering.
    messages = [
        {**m, "content": ("" if m.get("content") is None else m["content"])}
        for m in messages
    ]
    try:
        # Newer transformers / mlx-lm support tools= directly
        return tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # Fallback: inject tool definitions into the system message manually
        tool_text = (
            "\n\nYou have access to these tools (call them as JSON inside "
            "<tool_call>…</tool_call> tags):\n"
            + json.dumps(TOOLS, indent=2)
        )
        patched = list(messages)
        if patched and patched[0]["role"] == "system":
            patched[0] = {**patched[0], "content": patched[0]["content"] + tool_text}
        return tokenizer.apply_chat_template(
            patched, tokenize=False, add_generation_prompt=True
        )


def _loads_lenient(raw: str) -> dict | None:
    """Parse a tool-call JSON body, tolerating common small-model quirks. Smaller models
    (e.g. Qwen2.5-3B) sometimes echo the chat template's Jinja braces and emit doubled
    braces — `{{"name": ...}}` — which is invalid JSON. Repair and retry before giving up."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Collapse doubled braces, then retry.
    repaired = raw.replace("{{", "{").replace("}}", "}")
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool call dicts from the model output."""
    calls = []
    for raw in _TOOL_CALL_RE.findall(text):
        obj = _loads_lenient(raw)
        if obj is not None:
            calls.append(obj)
    if not calls:
        for name, args_str in _JSON_CALL_RE.findall(text):
            obj = _loads_lenient(args_str)
            if obj is not None:
                calls.append({"name": name, "arguments": obj})
    return calls


def _strip_tool_tags(text: str) -> str:
    """Remove tool_call XML tags from text before yielding to the user."""
    text = _TOOL_CALL_RE.sub("", text)
    text = _JSON_CALL_RE.sub("", text)
    return text.strip()


class MLXBackend(LLMBackend):
    def __init__(self, dispatcher: ToolDispatcher, model_id: str | None = None) -> None:
        self.dispatcher = dispatcher
        cfg = get_config()["llm"]
        self.model_id = model_id or _default_model_id()
        self.max_tokens: int = cfg["max_tokens"]
        self.temperature: float = cfg["temperature"]

    def warmup(self) -> None:
        """Load the model and prime the prefix KV cache so the first turn does
        not pay the ~3.6s cold-start cost on the patient's first utterance."""
        _load_model(self.model_id)
        _primed_prefix(self.model_id)

    def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = _load_model(self.model_id)
        prefix_ids, master = _primed_prefix(self.model_id)
        sampler = make_sampler(temp=self.temperature)
        working_messages = list(messages)

        for _round in range(3):
            full_prompt = _format_prompt(working_messages, tokenizer)
            ids = tokenizer.encode(full_prompt)

            # Reuse the primed prefix for whatever leading tokens still match
            # (the persona + tools block). Copy the master so generation never
            # mutates it, then trim the copy down to the common-prefix length
            # and feed only the remaining suffix to the model.
            common = _common_prefix_len(prefix_ids, ids)
            cache = [copy.deepcopy(c) for c in master]
            extra = len(prefix_ids) - common
            if extra > 0:
                trim_prompt_cache(cache, extra)
            prompt = mx.array(ids[common:])
            t0 = time.perf_counter()

            accumulated = ""
            pending = ""  # chars held back in case they start a <tool_call> tag

            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                prompt_cache=cache,
            ):
                token_text = response.text  # GenerationResponse.text in mlx-lm 0.21+
                accumulated += token_text
                pending += token_text

                safe, pending = _split_safe(pending)
                if safe:
                    yield safe

            # Flush remaining buffer
            if pending:
                tool_calls = _parse_tool_calls(pending)
                if not tool_calls:
                    yield _strip_tool_tags(pending)
                pending = ""

            logger.debug(
                "MLX round %d: %.0fms", _round, (time.perf_counter() - t0) * 1000
            )

            # Check the full output for tool calls
            tool_calls = _parse_tool_calls(accumulated)
            if not tool_calls:
                break

            # Dispatch each tool call, build result messages, re-enter.
            # NB: content must be a string, never None — some chat templates (Qwen2.5)
            # concatenate message content directly and crash on None.
            logger.info("MLX tool calls: %s", [tc["name"] for tc in tool_calls])
            working_messages.append({
                "role": "assistant",
                "content": _strip_tool_tags(accumulated),
            })
            for i, tc in enumerate(tool_calls):
                call_id = f"mlx_tool_{_round}_{i}"
                result = self.dispatcher.dispatch(tc["name"], tc.get("arguments", {}))
                # Append as a tool result in OpenAI format for the next round
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })


def _split_safe(text: str) -> tuple[str, str]:
    """
    Split text into (safe_to_yield, hold_back).
    Hold back the tail if it could be the start of a <tool_call> tag.
    """
    tag = "<tool_call>"
    # Find the last '<' that could begin a tag
    idx = text.rfind("<")
    if idx == -1:
        return text, ""
    tail = text[idx:]
    # If the tail matches a prefix of <tool_call>, hold it back
    if tag.startswith(tail) or tail.startswith("<tool_call>"):
        return text[:idx], tail
    return text, ""
