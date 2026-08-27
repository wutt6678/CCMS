"""Replay backends (Iteration 8).

A backend turns one replay chat (system prompt + stored history +
terminal q*) into a raw response. Backends are SWAPPABLE: the initial
target is a local HuggingFace Qwen3.5-9B, but the runner only depends
on the ``generate`` contract, so model/config changes never touch the
stage logic. Tests use ``CallableBackend``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from causal_mllm.replay.config import ReplayConfig


class ReplayBackend(Protocol):
    def generate(self, chat_messages: list[dict]) -> dict:
        """Return {"response": str, "input_token_count": int,
        "image_token_count": int}."""

    def model_revision(self) -> str | None:
        """Resolved content revision of the loaded weights."""


class CallableBackend:
    """Backend from a plain function (tests, dry runs)."""

    def __init__(self, fn: Callable[[list[dict]], dict | str], *,
                 model_name: str = "callable-stub",
                 model_revision: str | None = "stub"):
        self._fn = fn
        self._model_name = model_name
        self._model_revision = model_revision

    def generate(self, chat_messages: list[dict]) -> dict:
        result = self._fn(chat_messages)
        if isinstance(result, str):
            result = {"response": result}
        result.setdefault("input_token_count", 0)
        result.setdefault("image_token_count", 0)
        result.setdefault("output_token_count", 0)
        result.setdefault("finish_reason", "eos")
        result.setdefault("hit_max_new_tokens", False)
        return result

    def model_revision(self) -> str | None:
        return self._model_revision

    def model_name(self) -> str:
        return self._model_name


class HFLocalBackend:
    """Local HuggingFace VLM backend (initial target: Qwen3.5-9B).

    Loads lazily via :meth:`load`; ``generate`` applies the chat
    template with ``enable_thinking`` from the config, decodes
    greedy/sample per config, and reports token counts including the
    number of image pad tokens (visual-token metadata) and OUTPUT
    diagnostics (output_token_count, finish_reason,
    hit_max_new_tokens) so truncation is visible per condition.

    Reproducibility: if ``config.model_revision`` is set it is passed
    to BOTH ``from_pretrained`` calls, so the recorded revision is the
    revision actually loaded. The config seed is applied via
    ``transformers.set_seed`` at load time — honest provenance even if
    sampling is ever enabled.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._revision = config.model_revision

    def _pretrained_kwargs(self) -> dict:
        """Pin the recorded revision into the actual load."""
        kwargs = {}
        if self.config.model_revision is not None:
            kwargs["revision"] = self.config.model_revision
        return kwargs

    def load(self) -> "HFLocalBackend":
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            set_seed,
        )

        kwargs = self._pretrained_kwargs()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, **kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device,
            **kwargs,
        )
        self.model.eval()
        # Apply the recorded seed so the provenance is never a lie
        # (no-op for greedy decoding today, meaningful if sampling is
        # ever enabled).
        set_seed(self.config.seed)
        if self._revision is None:
            self._revision = self._resolve_revision()
        return self

    def _resolve_revision(self) -> str:
        """Content-addressed revision of the loaded weights.

        Prefers the hub cache metadata (refs/main -> commit hash) for
        ``org/name`` model ids; falls back to the snapshot directory
        actually loaded from; finally the model name itself.
        """
        import os

        cache_root = os.environ.get("HF_HOME")
        hub_dir = Path(cache_root) / "hub" if cache_root else \
            Path.home() / ".cache" / "huggingface" / "hub"
        if "/" in self.config.model_name:
            repo_dir = hub_dir / (
                "models--" + self.config.model_name.replace("/", "--"))
            refs_dir = repo_dir / "refs"
            if refs_dir.exists():
                for ref_file in sorted(refs_dir.glob("*")):
                    commit = ref_file.read_text(encoding="utf-8").strip()
                    if commit:
                        return commit
        name_or_path = getattr(self.model.config, "_name_or_path", "")
        model_path = Path(getattr(self.model, "name_or_path", "")
                          or name_or_path)
        if "snapshots" in model_path.parts:
            return model_path.parts[
                model_path.parts.index("snapshots") + 1]
        return Path(name_or_path).name or self.config.model_name

    def model_revision(self) -> str | None:
        return self._revision

    def _eos_token_ids(self) -> set:
        """All EOS ids of the loaded stack, defensively collected.

        Composite VLM configs may not expose ``eos_token_id`` at the
        top level (Qwen3.5 keeps it in ``text_config``), so probe
        every plausible location and tolerate missing attributes.
        """
        eos_ids: set = set()

        def add(value):
            if isinstance(value, int):
                eos_ids.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, int):
                        eos_ids.add(item)

        config = self.model.config
        add(getattr(config, "eos_token_id", None))
        add(getattr(getattr(config, "text_config", None),
                    "eos_token_id", None))
        add(getattr(self.processor, "eos_token_id", None))
        add(getattr(getattr(self.processor, "tokenizer", None),
                    "eos_token_id", None))
        return eos_ids

    def model_name(self) -> str:
        return self.config.model_name

    def generate(self, chat_messages: list[dict]) -> dict:
        if self.model is None:
            raise RuntimeError("HFLocalBackend.load() was not called")
        import torch
        from PIL import Image

        config = self.config
        images = []
        for message in chat_messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        images.append(Image.open(part["image"]).convert("RGB"))

        text = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )
        inputs = self.processor(
            text=[text], images=images or None, padding=True,
            return_tensors="pt",
        ).to(self.config.device)

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True)[0].strip()

        # OUTPUT diagnostics: truncation is not condition-independent
        # (refusals are short, compliant answers are long), so every
        # record must say how generation stopped.
        output_count = int(new_tokens.shape[1])
        hit_cap = output_count >= config.max_new_tokens
        if hit_cap:
            finish_reason = "length"
        else:
            last = int(new_tokens[0, -1])
            finish_reason = "eos" if last in self._eos_token_ids() \
                else "stop"

        input_ids = inputs["input_ids"][0]
        image_token_id = getattr(self.processor, "image_token_id", None)
        image_tokens = (
            int((input_ids == image_token_id).sum())
            if image_token_id is not None else 0
        )
        return {
            "response": response,
            "input_token_count": int(input_ids.shape[0]),
            "image_token_count": image_tokens,
            "output_token_count": output_count,
            "finish_reason": finish_reason,
            "hit_max_new_tokens": hit_cap,
        }
