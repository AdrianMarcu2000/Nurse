"""VisionClient — minimal multimodal (image→text) client, analogous to LLMClient.

Kept small and lazily importing the heavy backend so the rest of the platform runs
without a VLM installed. The concrete VLM integration (mlx-vlm on Mac / a TRT/llama.cpp
multimodal backend on Orin) is the deferred heavy piece; this is the seam it drops into.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VisionClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def describe(self, image_path: str, prompt: str) -> str:
        """Return a textual description of `image_path` given `prompt`.

        Tries mlx-vlm (Apple Silicon). If unavailable, raises so the caller
        (VisionSkill) disables/skips gracefully — sensing is never required for a turn.
        """
        from mlx_vlm import generate, load           # heavy, optional
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        model, processor = load(self.model_id)
        config = load_config(self.model_id)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        out = generate(model, processor, formatted, [image_path], verbose=False)
        return out if isinstance(out, str) else str(out)
