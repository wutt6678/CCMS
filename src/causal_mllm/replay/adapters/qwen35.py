"""Qwen3.5 adapter (Iteration 11).

Generalizes the frozen ``HFLocalBackend`` to any Qwen3.5 checkpoint
(2B / 4B / 9B) selected by ``model_spec.model_id``.  Checkpoint size is
never inferred from the response — it comes from the registry's
declared ``size_metadata``.

Thinking mode is driven by the frozen registry entry
(``thinking_mode: false`` for every Iteration 11 target), reproducing
the Iteration 8-10 ``enable_thinking=False`` setting.  Everything else
(tokenization, greedy decoding, EOS probing, ``image_token_id``
accounting, revision resolution) is inherited unchanged from
``HFAdapterBase``, which mirrors ``HFLocalBackend``.
"""

from __future__ import annotations

from causal_mllm.replay.adapters.base import HFAdapterBase


class Qwen35Adapter(HFAdapterBase):
    adapter_name = "qwen35"
    adapter_version = "1.0.0"

    def chat_template_kwargs(self) -> dict:
        # Qwen3.5 exposes a thinking switch; Iterations 8-10 froze it
        # OFF and Iteration 11 must not silently change it.
        return {"enable_thinking": bool(self.model_spec.thinking_mode)}

    def extra_runtime_metadata(self) -> dict:
        return {
            "enable_thinking": bool(self.model_spec.thinking_mode),
            "declared_parameters": self.model_spec.size_metadata.get(
                "language_parameters"),
        }
