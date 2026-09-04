"""Iteration 11 model adapters.

``build_adapter`` maps a resolved model spec to the thin family adapter.
Non-Qwen adapters are imported lazily so this package stays importable
before their modules land (11.3 / 11.4).
"""

from __future__ import annotations

from causal_mllm.replay.adapters.base import (
    HFAdapterBase, TargetModelAdapter)
from causal_mllm.replay.adapters.qwen35 import Qwen35Adapter
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import ResolvedModel

__all__ = [
    "TargetModelAdapter", "HFAdapterBase", "Qwen35Adapter", "build_adapter",
]


def build_adapter(model_spec: ResolvedModel, config,
                  device: str | None = None) -> TargetModelAdapter:
    """Instantiate the adapter named by ``model_spec.adapter``."""
    kind = model_spec.adapter
    if kind == "qwen35":
        return Qwen35Adapter(config, model_spec, device=device)
    if kind == "ministral3":
        from causal_mllm.replay.adapters.ministral3 import Ministral3Adapter
        return Ministral3Adapter(config, model_spec, device=device)
    if kind == "phi4_multimodal":
        from causal_mllm.replay.adapters.phi4_multimodal import (
            Phi4MultimodalAdapter)
        return Phi4MultimodalAdapter(config, model_spec, device=device)
    if kind == "gemma3":
        from causal_mllm.replay.adapters.gemma3 import Gemma3Adapter
        return Gemma3Adapter(config, model_spec, device=device)
    raise ReplayError(f"no adapter registered for kind {kind!r}")
