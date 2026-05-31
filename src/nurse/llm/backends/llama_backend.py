"""llama-cpp-python backend — used on Jetson (CUDA) and as Mac fallback."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from functools import lru_cache
from typing import Any

from nurse.config import get_config, resolve
from nurse.llm.backends.base import LLMBackend
from nurse.llm.tools import TOOLS, ToolDispatcher

logger = logging.getLogger(__name__)


def _default_model_id() -> str:
    """For llama.cpp the 'model id' is the GGUF path (relative to project root)."""
    return get_config()["llm"]["model_path"]


@lru_cache(maxsize=None)
def _load_model(model_id: str):
    """Load a llama.cpp model by GGUF path. Cached per id so several models
    (Front Voice + specialists) can stay resident at once."""
    from llama_cpp import Llama

    cfg = get_config()["llm"]
    model_path = resolve(model_id)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LLM model not found at {model_path}. Run scripts/download_models.sh first."
        )
    logger.info("Loading LLM (llama.cpp) from %s …", model_path)
    return Llama(
        model_path=str(model_path),
        n_ctx=cfg["n_ctx"],
        n_gpu_layers=cfg["n_gpu_layers"],
        n_threads=cfg["n_threads"],
        verbose=False,
    )


class LlamaBackend(LLMBackend):
    def __init__(self, dispatcher: ToolDispatcher, model_id: str | None = None) -> None:
        self.dispatcher = dispatcher
        cfg = get_config()["llm"]
        self.model_id = model_id or _default_model_id()
        self.temperature: float = cfg["temperature"]
        self.max_tokens: int = cfg["max_tokens"]

    def warmup(self) -> None:
        """Load weights and run a 1-token generation so the patient's first turn does
        not pay the full model-load + prefill cost (the base class's no-op left the
        first turn cold on the Jetson)."""
        llm = _load_model(self.model_id)
        t0 = time.perf_counter()
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        logger.info("llama.cpp warmup: %.0fms", (time.perf_counter() - t0) * 1000)

    def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        llm = _load_model(self.model_id)
        working_messages = list(messages)

        for _round in range(3):
            t0 = time.perf_counter()
            response = llm.create_chat_completion(
                messages=working_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            full_content = ""
            tool_calls: list[dict] = []

            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})

                if "content" in delta and delta["content"]:
                    token = delta["content"]
                    full_content += token
                    yield token

                if "tool_calls" in delta and delta["tool_calls"]:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        if tc.get("id"):
                            tool_calls[idx]["id"] = tc["id"]
                        if tc.get("function"):
                            fn = tc["function"]
                            if fn.get("name"):
                                tool_calls[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += fn["arguments"]

            logger.debug(
                "llama round %d: %.0fms, tool_calls=%d",
                _round, (time.perf_counter() - t0) * 1000, len(tool_calls),
            )

            if not tool_calls:
                break

            working_messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                result = self.dispatcher.dispatch(
                    tc["function"]["name"], tc["function"]["arguments"]
                )
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
