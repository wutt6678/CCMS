"""Iteration 11 target-model adapter contract + shared HF base.

One shared generation pipeline (``HFAdapterBase``) with thin
model-family adapters that override only what genuinely differs:
chat-template kwargs, model class, checkpoint loader, and runtime
metadata.

``HFAdapterBase`` deliberately mirrors the frozen
:class:`~causal_mllm.replay.backend.HFLocalBackend` flow line for line
(PIL ``Image.open(...).convert("RGB")``, ``padding=True``,
``device_map``, ``transformers.set_seed``, ``image_token_id`` counting,
defensive EOS probing across ``model.config`` / ``text_config`` /
processor / tokenizer, greedy ``finish_reason`` derivation, and NO
exception wrapping so the runner's ``classify_error`` still sees the
original failure).  With a Qwen3.5 spec this reproduces the
Iterations 8-10 behaviour exactly; the frozen ``HFLocalBackend`` itself
is left untouched and remains the path for legacy single-model runs.

Adapters satisfy the existing ``ReplayBackend`` protocol so they plug
into ``run_replay_stage`` via ``backend=``, and additionally expose the
richer ``TargetModelAdapter`` surface (serialize / decode / token
accounting / runtime metadata) required by the Iteration 11 record
schema.
"""

from __future__ import annotations

import abc
import hashlib
import os
from pathlib import Path
from typing import Any

from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import ResolvedModel
from causal_mllm.seeds import sha256_text

SUPPORTED_QUANTIZATION = {"none", None}


def _file_sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def semantic_prompt_hash(chat_messages: list[dict]) -> str:
    """Hash of the text-only semantic content (model-independent).

    Covers the system prompt, every turn's text and the terminal
    question, independent of how a given tokenizer renders them, so two
    families can be compared before family-specific serialization.
    """
    parts: list[str] = []
    for message in chat_messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            parts.append(f"{role}:{content}")
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(f"{role}:{part.get('text', '')}")
            elif part.get("type") == "image":
                parts.append(f"{role}:<image>")
    return sha256_text("\n".join(parts))


def ordered_image_hashes(chat_messages: list[dict]) -> list[str]:
    """SHA-256 of every referenced image, in message order."""
    hashes: list[str] = []
    for message in chat_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                digest = _file_sha256(part.get("image", ""))
                if digest is not None:
                    hashes.append(digest)
    return hashes


class TargetModelAdapter(abc.ABC):
    """Contract every Iteration 11 model adapter must satisfy."""

    adapter_name: str = "abstract"
    adapter_version: str = "1.0.0"

    @abc.abstractmethod
    def load(self) -> Any:
        """Load weights + processor; resolve immutable revisions."""

    @abc.abstractmethod
    def serialize_messages(self, chat_messages: list[dict]) -> Any:
        """Render semantic messages into model-specific inputs."""

    @abc.abstractmethod
    def generate(self, chat_messages: list[dict]) -> dict:
        """One greedy generation + full per-record diagnostics."""

    @abc.abstractmethod
    def decode_new_tokens(self, new_tokens: Any) -> str:
        """Decode ONLY the generated continuation."""

    @abc.abstractmethod
    def count_input_tokens(self, inputs: Any) -> int: ...

    @abc.abstractmethod
    def count_output_tokens(self, new_tokens: Any) -> int: ...

    @abc.abstractmethod
    def runtime_metadata(self) -> dict:
        """Versions, hardware, seed and determinism actually in force."""

    # --- ReplayBackend protocol (metadata) -----------------------------
    @abc.abstractmethod
    def model_name(self) -> str: ...

    @abc.abstractmethod
    def model_revision(self) -> str | None: ...

    @abc.abstractmethod
    def processor_revision(self) -> str | None: ...

    @abc.abstractmethod
    def transformers_version(self) -> str | None: ...

    @abc.abstractmethod
    def torch_version(self) -> str | None: ...

    @abc.abstractmethod
    def cuda_version(self) -> str | None: ...


class HFAdapterBase(TargetModelAdapter):
    """Shared HuggingFace generation pipeline with thin family hooks."""

    adapter_name = "hf_base"
    adapter_version = "1.0.0"

    def __init__(self, config, model_spec: ResolvedModel,
                 device: str | None = None, **extra_pretrained_kwargs):
        if model_spec.quantization not in SUPPORTED_QUANTIZATION:
            # Fail closed rather than silently comparing a quantized
            # checkpoint against the bf16 reference panel.
            raise ReplayError(
                f"{model_spec.model_key}: quantization "
                f"{model_spec.quantization!r} is out of scope; the frozen "
                f"protocol requires full-precision bf16 checkpoints")
        self.config = config
        self.model_spec = model_spec
        self.device = device or config.device
        self._extra_pretrained_kwargs = extra_pretrained_kwargs
        self.model = None
        self.processor = None
        self._revision = model_spec.revision
        self._processor_revision: str | None = None
        self._transformers_version: str | None = None
        self._torch_version: str | None = None
        self._cuda_version: str | None = None

    # --- family hooks --------------------------------------------------
    def chat_template_kwargs(self) -> dict:
        """Extra apply_chat_template kwargs (Qwen: enable_thinking)."""
        return {}

    def select_model_class(self):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText

    def load_model(self, model_cls, model_id: str, kwargs: dict):
        """Override for families needing a non-standard checkpoint load."""
        return model_cls.from_pretrained(model_id, **kwargs)

    def extra_runtime_metadata(self) -> dict:
        return {}

    # --- loading -------------------------------------------------------
    def _pretrained_kwargs(self) -> dict:
        """Pin the recorded revision into the actual load."""
        kwargs: dict[str, Any] = {}
        if self.model_spec.revision is not None:
            kwargs["revision"] = self.model_spec.revision
        if self.model_spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        kwargs.update(self._extra_pretrained_kwargs)
        return kwargs

    def _model_kwargs(self, torch, kwargs: dict) -> dict:
        model_kwargs = {
            "torch_dtype": getattr(torch, self.model_spec.dtype),
            "device_map": self.device,
        }
        model_kwargs.update(kwargs)
        return model_kwargs

    def load(self) -> "HFAdapterBase":
        import torch
        import transformers
        from transformers import AutoProcessor, set_seed

        self._transformers_version = transformers.__version__
        self._torch_version = torch.__version__
        self._cuda_version = (torch.version.cuda
                              if torch.cuda.is_available() else None)
        kwargs = self._pretrained_kwargs()
        self.processor = AutoProcessor.from_pretrained(
            self.model_spec.model_id, **kwargs)
        # When the revision is explicitly pinned, use it directly for the
        # processor revision too — reading the first cached ref could
        # report a later main revision if the cache changes.
        if self.model_spec.revision is not None:
            self._processor_revision = self.model_spec.revision
        else:
            self._processor_revision = self._resolve_revision_from_cache()
        self.model = self.load_model(
            self.select_model_class(), self.model_spec.model_id,
            self._model_kwargs(torch, kwargs))
        self.model.eval()
        # Apply the recorded seed so the provenance is never a lie
        # (no-op for greedy decoding today, meaningful if sampling is
        # ever enabled).
        set_seed(self.config.seed)
        if self._revision is None:
            self._revision = self._resolve_revision()
        return self

    def _resolve_revision(self) -> str:
        """Content-addressed revision of the loaded weights."""
        return self._resolve_revision_from_cache() or \
            self._resolve_revision_from_model()

    def _resolve_revision_from_cache(self) -> str | None:
        """Read refs/main from the HF hub cache for the model repo."""
        cache_root = os.environ.get("HF_HOME")
        hub_dir = Path(cache_root) / "hub" if cache_root else \
            Path.home() / ".cache" / "huggingface" / "hub"
        model_id = self.model_spec.model_id
        if "/" in model_id:
            repo_dir = hub_dir / ("models--" + model_id.replace("/", "--"))
            refs_dir = repo_dir / "refs"
            if refs_dir.exists():
                for ref_file in sorted(refs_dir.glob("*")):
                    commit = ref_file.read_text(encoding="utf-8").strip()
                    if commit:
                        return commit
        return None

    def _resolve_revision_from_model(self) -> str:
        """Fallback: extract the revision from the loaded snapshot path."""
        name_or_path = getattr(self.model.config, "_name_or_path", "")
        model_path = Path(getattr(self.model, "name_or_path", "")
                          or name_or_path)
        if "snapshots" in model_path.parts:
            return model_path.parts[
                model_path.parts.index("snapshots") + 1]
        return Path(name_or_path).name or self.model_spec.model_id

    def _eos_token_ids(self) -> set:
        """All EOS ids of the loaded stack, defensively collected.

        Composite VLM configs may not expose ``eos_token_id`` at the top
        level (Qwen3.5 keeps it in ``text_config``), so probe every
        plausible location and tolerate missing attributes.
        """
        eos_ids: set = set()

        def add(value):
            if isinstance(value, int):
                eos_ids.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, int):
                        eos_ids.add(item)

        config = getattr(self.model, "config", None)
        add(getattr(config, "eos_token_id", None))
        add(getattr(getattr(config, "text_config", None),
                    "eos_token_id", None))
        add(getattr(self.processor, "eos_token_id", None))
        add(getattr(getattr(self.processor, "tokenizer", None),
                    "eos_token_id", None))
        return eos_ids

    # --- generation ----------------------------------------------------
    def serialize_messages(self, chat_messages: list[dict]):
        """Semantic messages -> (rendered text, PIL images, inputs)."""
        from PIL import Image

        images = []
        for message in chat_messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        images.append(
                            Image.open(part["image"]).convert("RGB"))
        text = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True,
            **self.chat_template_kwargs())
        inputs = self.processor(
            text=[text], images=images or None, padding=True,
            return_tensors="pt",
        ).to(self.device)
        return text, images, inputs

    def decode_new_tokens(self, new_tokens) -> str:
        return self.processor.batch_decode(
            new_tokens, skip_special_tokens=True)[0].strip()

    def count_input_tokens(self, inputs) -> int:
        return int(inputs["input_ids"][0].shape[0])

    def count_output_tokens(self, new_tokens) -> int:
        return int(new_tokens.shape[1])

    def count_image_tokens(self, inputs) -> int:
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            return 0
        return int((inputs["input_ids"][0] == image_token_id).sum())

    def effective_decoding(self) -> dict:
        """The decoding actually in force (greedy => sampling knobs off)."""
        return {
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample
            else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
            "top_k": None,
            "num_beams": 1,
            "max_new_tokens": self.config.max_new_tokens,
        }

    def generate(self, chat_messages: list[dict]) -> dict:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.load() was not called")
        import torch

        config = self.config
        text, images, inputs = self.serialize_messages(chat_messages)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[:, prompt_len:]
        response = self.decode_new_tokens(new_tokens)

        # OUTPUT diagnostics: truncation is not condition-independent
        # (refusals are short, compliant answers are long), so every
        # record must say how generation stopped.
        output_count = self.count_output_tokens(new_tokens)
        hit_cap = output_count >= config.max_new_tokens
        if hit_cap:
            finish_reason = "length"
        else:
            last = int(new_tokens[0, -1]) if output_count else -1
            finish_reason = "eos" if last in self._eos_token_ids() \
                else "stop"

        return {
            "response": response,
            "input_token_count": self.count_input_tokens(inputs),
            "image_token_count": self.count_image_tokens(inputs),
            "output_token_count": output_count,
            "finish_reason": finish_reason,
            "hit_max_new_tokens": hit_cap,
            # --- Iteration 11 per-record provenance --------------------
            "semantic_prompt_hash": semantic_prompt_hash(chat_messages),
            "serialized_prompt_hash": sha256_text(text),
            "ordered_image_hashes": ordered_image_hashes(chat_messages),
            "effective_decoding": self.effective_decoding(),
        }

    # --- metadata ------------------------------------------------------
    def runtime_metadata(self) -> dict:
        import torch
        import transformers

        meta: dict[str, Any] = {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "model_key": self.model_spec.model_key,
            "model_id": self.model_spec.model_id,
            "dtype": self.model_spec.dtype,
            "quantization": self.model_spec.quantization,
            "trust_remote_code": self.model_spec.trust_remote_code,
            "thinking_mode": self.model_spec.thinking_mode,
            "device": self.device,
            "transformers_version": self._transformers_version
            or transformers.__version__,
            "torch_version": self._torch_version or torch.__version__,
            "cuda_version": self._cuda_version,
            "requested_seed": self.config.seed,
            "effective_seed": self.config.seed,
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()),
            "hardware": self._hardware_metadata(torch),
        }
        meta.update(self.extra_runtime_metadata())
        return meta

    def _hardware_metadata(self, torch) -> dict | None:
        if not torch.cuda.is_available():
            return None
        try:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return {
                "device_index": index,
                "gpu_name": props.name,
                "total_memory_mb": int(props.total_memory / (1024 ** 2)),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        except Exception:
            return None

    def model_name(self) -> str:
        return self.model_spec.model_id

    def model_revision(self) -> str | None:
        return self._revision

    def processor_revision(self) -> str | None:
        return self._processor_revision

    def transformers_version(self) -> str | None:
        return self._transformers_version

    def torch_version(self) -> str | None:
        return self._torch_version

    def cuda_version(self) -> str | None:
        return self._cuda_version
