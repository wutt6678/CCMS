"""Ministral-3 adapter (Iteration 11).

``mistralai/Ministral-3-3B-Instruct-2512-BF16`` is ``model_type:
mistral3`` / ``Mistral3ForConditionalGeneration``, natively supported by
transformers 5.14.1, so no remote code is executed (the registry pins
``trust_remote_code: false``).

Three family differences from Qwen3.5 matter and are handled here:

1. **No thinking switch.** The official chat template does not accept
   ``enable_thinking``; passing it would raise. ``chat_template_kwargs``
   is therefore empty.
2. **Image-token accounting.** ``PixtralProcessor`` exposes no
   ``image_token_id`` and ``config.image_token_id`` is ``null``; the
   placeholder id lives at ``config.image_token_index`` (``[IMG]`` = 10).
   The generic path would silently report 0 image tokens, so this adapter
   resolves the id explicitly.
3. **A vendor default system prompt.** The template injects a 2,406-char
   Mistral/Le Chat default *when ``messages[0]['role'] != 'system'``*.
   The frozen CCMS system prompt is always ``messages[0]``, which
   suppresses it — but that is VERIFIED per generation rather than
   assumed, because a leaked vendor prompt would mean Ministral was
   evaluated under different instructions than the Qwen arm and would
   invalidate the cross-family comparison.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from causal_mllm.replay.adapters.base import HFAdapterBase

#: Distinctive fragments of the vendor default system prompt shipped in
#: the checkpoint's chat_template.jinja / SYSTEM_PROMPT.txt. Their
#: presence in a rendered prompt means the default was injected.
VENDOR_DEFAULT_MARKERS = (
    "created by Mistral AI",
    "You power an AI assistant called Le Chat",
    "WEB BROWSING INSTRUCTIONS",
)

IMAGE_TOKEN = "[IMG]"
IMAGE_BREAK_TOKEN = "[IMG_BREAK]"
IMAGE_END_TOKEN = "[IMG_END]"


class Ministral3Adapter(HFAdapterBase):
    adapter_name = "ministral3"
    adapter_version = "1.0.0"

    def chat_template_kwargs(self) -> dict:
        # No thinking switch exists for this family; the frozen protocol
        # requires thinking off everywhere, which is the template's only
        # behaviour here.
        return {}

    # --- token accounting ---------------------------------------------
    def _image_token_id(self) -> int | None:
        """Resolve the ``[IMG]`` placeholder id for this checkpoint."""
        config = getattr(self.model, "config", None)
        for attr in ("image_token_index", "image_token_id"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value
        processor_value = getattr(self.processor, "image_token_id", None)
        if isinstance(processor_value, int):
            return processor_value
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            except Exception:
                return None
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        return None

    def _marker_token_id(self, token: str) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None
        return token_id if isinstance(token_id, int) and token_id >= 0 else None

    def count_image_tokens(self, inputs) -> int:
        image_token_id = self._image_token_id()
        if image_token_id is None:
            return 0
        return int((inputs["input_ids"][0] == image_token_id).sum())

    def _marker_counts(self, inputs) -> dict:
        counts = {}
        for name, token in (("image", IMAGE_TOKEN),
                            ("image_break", IMAGE_BREAK_TOKEN),
                            ("image_end", IMAGE_END_TOKEN)):
            token_id = (self._image_token_id() if name == "image"
                        else self._marker_token_id(token))
            counts[f"{name}_token_count"] = (
                int((inputs["input_ids"][0] == token_id).sum())
                if token_id is not None else None)
        return counts

    def _eos_token_ids(self) -> set:
        # generation_config carries eos (</s> = 2); the composite config
        # does not expose it at the top level.
        ids = set(super()._eos_token_ids())
        generation_config = getattr(self.model, "generation_config", None)
        candidate = getattr(generation_config, "eos_token_id", None)
        if isinstance(candidate, int):
            ids.add(candidate)
        elif isinstance(candidate, (list, tuple)):
            ids.update(int(x) for x in candidate if isinstance(x, int))
        return ids

    # --- prompt-integrity diagnostics ---------------------------------
    def _vendor_system_prompt_sha256(self) -> str | None:
        """Hash of the vendor default we deliberately suppress.

        Resolved through the hub snapshot rather than
        ``config._name_or_path``, which holds the repo id (not a local
        path) when the model is loaded with ``device_map``.
        """
        from causal_mllm.replay.checkpoint_size import resolve_snapshot_dir
        try:
            snapshot = resolve_snapshot_dir(
                self.model_spec.model_id,
                revision=self.model_spec.revision or self._revision)
        except Exception:
            return None
        candidate = Path(snapshot) / "SYSTEM_PROMPT.txt"
        try:
            if candidate.exists():
                return hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            return None
        return None

    def adapter_diagnostics(self, text: str, inputs) -> dict:
        leaked = [marker for marker in VENDOR_DEFAULT_MARKERS
                  if marker in text]
        tokenizer = getattr(self.processor, "tokenizer", None)
        diagnostics = {
            "vendor_default_system_prompt_injected": bool(leaked),
            "vendor_default_markers_found": leaked,
            "vendor_default_system_prompt_sha256":
                self._vendor_system_prompt_sha256(),
            "frozen_system_prompt_present_verbatim":
                self.config.system_prompt.strip() in text,
            "image_token_id": self._image_token_id(),
            # transformers warns that Mistral tokenizers may need
            # fix_mistral_regex=True. For this pinned revision the flag
            # provably changes nothing (identical token ids for
            # None/True/False), so it is not set; the observed value is
            # recorded so the claim stays auditable.
            "tokenizer_fix_mistral_regex": getattr(
                tokenizer, "fix_mistral_regex", None),
        }
        diagnostics.update(self._marker_counts(inputs))
        return diagnostics

    def extra_runtime_metadata(self) -> dict:
        return {
            "thinking_switch_available": False,
            "vendor_default_system_prompt_sha256":
                self._vendor_system_prompt_sha256(),
            # The registry carries the specification's declared
            # approximations; the MEASURED counts are in the preflight
            # report's size_metadata and are authoritative.
            "registry_declared_parameters": {
                "language": self.model_spec.size_metadata.get(
                    "language_parameters"),
                "vision": self.model_spec.size_metadata.get(
                    "vision_parameters"),
            },
        }
