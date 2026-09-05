"""Phi-4-multimodal adapter (Iteration 11.4).

The frozen protocol's ``phi4_load_strategy`` decision is
``shim_in_shared_env``: ``microsoft/Phi-4-multimodal-instruct`` is a
transformers-4.x remote-code checkpoint (vendored ``modeling_phi4mm.py``,
bundled PEFT vision/speech LoRA adapters, flash-attention-2 hard-coded in
``config.json``) that has to run on the shared ``midp-qwen35``
environment's transformers 5.14.1.

Four incompatibilities are repaired, exactly the four the frozen protocol
pre-declared — items 1, 2, 3 and 8 below — plus four more discovered by
actually running the load (items 4-7):

1. ``config._attn_implementation`` is stored as ``flash_attention_2``;
   FA2 is unavailable for this model here, so sdpa is forced on the
   config and every nested sub-config.  ``Phi4MMSdpaAttention`` is a real
   vendor class, so this selects a supported attention path rather than
   monkey-patching one into existence.
2. transformers 5.x always meta-initialises and ignores
   ``low_cpu_mem_usage``; the bundled speech-conformer encoder calls
   ``.item()`` on a meta tensor during ``__init__``, so the checkpoint is
   constructed directly and its bf16 safetensors are loaded explicitly.
3. ``peft`` reads ``base_model.prepare_inputs_for_generation`` off the
   INNER ``Phi4MMModel``, which no longer has it now that transformers 5.x
   split ``GenerationMixin`` out of ``PreTrainedModel``.  The ``PeftModel``
   that peft builds is discarded by the vendor ``__init__`` (bound to a
   local, never to ``self.model``), so the binding only has to exist.
4. The vendor ships the 4.x list form ``_tied_weights_keys =
   ["lm_head.weight"]``; transformers 5.x requires a ``{target: source}``
   dict.
5. ``generation_config.json`` declares ``eos_token_id = [200020,
   199999]`` while ``config.json`` declares only ``199999``.  Direct
   construction derives the generation config from the model config and
   would therefore drop ``<|end|>`` (200020) — the token the chat template
   uses to terminate every message.  Losing it would make every response
   run to the 1536-token cap, so the shipped generation config is loaded
   explicitly.
6. ``generate`` only sets ``logits_to_keep`` when ``forward`` advertises
   that exact parameter name; this model names it ``num_logits_to_keep``,
   and the vendor's ``prepare_inputs_for_generation`` defaults it to
   ``None``, which would reach ``hidden_states[:, -None:, :]``.  Passing
   ``num_logits_to_keep=1`` both avoids that and matches what the other
   families get automatically — greedy decoding only ever consumes the
   last position's logits, so the sampled tokens are unchanged while the
   prefill logits tensor shrinks from the full sequence to one position.
7. transformers 5.x removed ``Cache.get_usable_length``, which the vendored
   attention calls on every step.  It is restored with its exact 4.x
   semantics (see :func:`shim_cache_api`).
8. The frozen protocol's fourth fix (gradient checkpointing / the custom
   SigLIP tower): checkpointing flags are disabled, and the vendored
   vision tower's own ``_flash_attention_forward`` hook — which does not
   consult ``config._attn_implementation`` — is redirected to sdpa.

The four numbered shims beyond the frozen list were each found by running
the load, not predicted; every one is recorded per run in
``runtime_metadata()["phi4_shims"]`` so the evidence states exactly what
was patched.

Because the checkpoint is loaded outside ``from_pretrained``, none of
transformers' safety nets apply.  The load is therefore verified
explicitly and fails closed: an untied ``lm_head`` or a parameter left
uninitialised would emit fluent-looking garbage while every superficial
check still passed.

The frozen protocol also records ``audio_tower_initialized: false``.  That
is not what the checkpoint does: ``Phi4MMImageAudioEmbedding`` builds the
audio tower unconditionally and the checkpoint ships its weights, and the
VISION path routes through ``audio_embed.audio_projection.vision``.  The
audio tower is therefore fully initialised; what is false is that any
audio INPUT is supplied.  This deviation is reported per record in
``runtime_metadata()["phi4_audio_tower"]`` rather than silently
contradicting the frozen artifact.
"""

from __future__ import annotations

import json
import re

from causal_mllm.replay.adapters.base import HFAdapterBase
from causal_mllm.replay.errors import ReplayError

#: Placeholder the vendor template/processor accept (``<\|image_\d+\|>``
#: is regex-normalised to ``<|endoftext10|>`` and then expanded to the
#: image's token count).  One placeholder per image, in order — the
#: processor asserts the counts match.
IMAGE_PLACEHOLDER = "<|image_{index}|>"
IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
END_SPECIAL_TOKEN = "<|end|>"
IMAGE_SPECIAL_TOKEN_ID = 200010
END_SPECIAL_TOKEN_ID = 200020

#: ``lm_head.weight`` is absent from the checkpoint because
#: ``tie_word_embeddings`` is true; 5.x wants ``{target: source}``.
TIED_TARGET = "lm_head.weight"
TIED_SOURCE = "model.embed_tokens.weight"
TIED_WEIGHTS_KEYS = {TIED_TARGET: TIED_SOURCE}

#: InputMode values from the vendored ``processing_phi4mm.py``.
INPUT_MODE_LANGUAGE = 0
INPUT_MODE_VISION = 1


def force_sdpa(config, _depth: int = 0, _seen: set | None = None) -> list:
    """Force ``sdpa`` on a config and every nested sub-config.

    Returns the names of the configs that were changed, so the shim is
    recorded rather than assumed.
    """
    from transformers.configuration_utils import PretrainedConfig

    seen = set() if _seen is None else _seen
    if id(config) in seen or _depth > 4:
        return []
    seen.add(id(config))
    changed = []
    before = getattr(config, "_attn_implementation", None)
    try:
        config._attn_implementation = "sdpa"
        config._attn_implementation_internal = "sdpa"
    except Exception:  # pragma: no cover - defensive
        return changed
    if before != "sdpa":
        changed.append(f"{type(config).__name__}: {before} -> sdpa")
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            changed.extend(force_sdpa(value, _depth + 1, seen))
        elif isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, PretrainedConfig):
                    changed.extend(force_sdpa(sub, _depth + 1, seen))
    return changed


def normalize_tied_weights_keys(model_cls) -> str | None:
    """Rewrite the 4.x list form of ``_tied_weights_keys`` to the 5.x dict."""
    current = getattr(model_cls, "_tied_weights_keys", None)
    if isinstance(current, dict) or current is None:
        return None
    model_cls._tied_weights_keys = dict(TIED_WEIGHTS_KEYS)
    return f"_tied_weights_keys {current!r} -> {TIED_WEIGHTS_KEYS!r}"


def bind_inner_prepare_inputs(model_cls) -> str | None:
    """Give the inner ``Phi4MMModel`` a ``prepare_inputs_for_generation``.

    peft 0.20 reads that attribute while injecting LoRA.  The generic
    transformers implementation is bound (rather than a sentinel) so the
    attribute is honest if anything ever does call it.
    """
    import importlib

    from transformers.generation.utils import GenerationMixin

    module = importlib.import_module(model_cls.__module__)
    inner = getattr(module, "Phi4MMModel", None)
    if inner is None or hasattr(inner, "prepare_inputs_for_generation"):
        return None
    inner.prepare_inputs_for_generation = \
        GenerationMixin.prepare_inputs_for_generation
    return f"bound GenerationMixin.prepare_inputs_for_generation onto " \
        f"{inner.__name__} (peft 0.20 compatibility; the peft wrapper the " \
        f"vendor builds is discarded)"


def cache_get_usable_length(cache, new_seq_length: int,
                            layer_idx: int = 0) -> int:
    """The transformers-4.x ``Cache.get_usable_length`` semantics.

    Kept as a standalone function (rather than a closure inside the shim)
    so the restored arithmetic is unit-testable on its own, with no
    transformers import and no mutation of global class state.
    """
    max_length = cache.get_max_length()
    previous = cache.get_seq_length(layer_idx)
    if max_length is not None and previous + new_seq_length > max_length:
        return max_length - new_seq_length
    return previous


def shim_cache_api() -> list[str]:
    """Restore the one ``Cache`` method this checkpoint still calls.

    transformers 5.x removed ``Cache.get_usable_length``, which the vendored
    attention calls on every step (eager / FA2 / sdpa at lines 1137 / 1232 /
    1354 of ``modeling_phi4mm.py``).  The 4.x semantics are restored
    exactly: the length already cached for that layer, clipped when the
    cache declares a maximum.  ``DynamicCache`` declares none, so in
    practice this is the plain cached length — the same value the 4.x
    stack produced.

    The patch is ADDITIVE (it defines a method that no longer exists and
    overrides nothing).  ``to_legacy_cache`` / ``from_legacy_cache`` were
    removed too, but they sit behind ``return_legacy_cache``, which is only
    set when ``past_key_values`` is not a ``Cache``; ``generate`` always
    passes a ``DynamicCache``, so those paths are unreachable and are
    deliberately left unpatched rather than papered over.
    """
    from transformers.cache_utils import Cache

    if hasattr(Cache, "get_usable_length"):
        return []
    Cache.get_usable_length = cache_get_usable_length
    return [
        "restored Cache.get_usable_length (removed in transformers 5.x) "
        "with its 4.x semantics — cached length for the layer, clipped to "
        "get_max_length() when the cache declares one",
    ]


def patch_vision_attention() -> int:
    """Redirect the vendored SigLIP tower's FA2 hook to sdpa.

    The custom ``vision_siglip_navit`` module implements its own
    ``_flash_attention_forward`` and does not consult
    ``config._attn_implementation``, so forcing sdpa on the config does not
    reach it.  Returns the number of classes patched.
    """
    import sys

    import torch
    import torch.nn.functional as F

    patched = 0
    for module_name, module in list(sys.modules.items()):
        if module is None or "vision_siglip_navit" not in module_name:
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, type):
                continue
            if not hasattr(attr, "_flash_attention_forward"):
                continue

            def _sdpa_vision(self, query_states, key_states, value_states,
                             attention_mask, query_length, dropout=0.0,
                             softmax_scale=None):
                causal = bool(self.is_causal) and query_length != 1
                q = query_states.transpose(1, 2)
                k = key_states.transpose(1, 2)
                v = value_states.transpose(1, 2)
                mask = attention_mask
                if mask is not None and mask.dim() >= 2:
                    if mask.dim() == 2:
                        mask = mask[:, None, None, :]
                    elif mask.dim() == 3:
                        mask = mask[:, None, :, :]
                    if mask.dtype != torch.bool:
                        mask = _to_bool_mask(mask)
                    # An explicit mask already encodes what may be attended
                    # to; combining it with is_causal would double-mask.
                    causal = False
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout,
                    is_causal=causal, scale=softmax_scale)
                return out.transpose(1, 2)

            attr._flash_attention_forward = _sdpa_vision
            patched += 1
    return patched


def _to_bool_mask(mask):
    """Normalise the mask formats the vendored tower can hand us."""
    import torch

    if mask.dtype == torch.bool:
        return mask
    if not torch.is_floating_point(mask):
        return mask != 0
    if mask.min() >= 0 and mask.max() <= 1:
        return mask > 0
    return mask > -1e4


def disable_gradient_checkpointing(model) -> int:
    """Turn off checkpointing in the vendored towers.

    transformers 5.x removed ``_gradient_checkpointing_func``, which the
    custom SigLIP/conformer code still calls.  Inference is ``eval()``, and
    ``SiglipEncoder`` only invokes it when ``gradient_checkpointing and
    self.training``, so disabling the flags is sufficient.
    """
    disabled = 0
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            module.gradient_checkpointing = False
            disabled += 1
    return disabled


class Phi4MultimodalAdapter(HFAdapterBase):
    """Phi-4-multimodal via the protocol's ``shim_in_shared_env`` route."""

    adapter_name = "phi4_multimodal"
    adapter_version = "1.0.0"

    def __init__(self, config, model_spec, device=None,
                 **extra_pretrained_kwargs):
        super().__init__(config, model_spec, device=device,
                         **extra_pretrained_kwargs)
        self.phi4_shims: list[str] = []
        self.phi4_load_report: dict = {}
        self._checkpoint_summary: dict = {}
        self._image_token_id: int | None = None
        self._last_image_count: int | None = None

    # --- family hooks --------------------------------------------------
    def chat_template_kwargs(self) -> dict:
        # The vendored template only reads role/content/tools.
        return {}

    def template_messages(self, chat_messages: list[dict]) -> list[dict]:
        """Flatten semantic parts into the string content the template needs.

        The template does ``'<|' + role + '|>' + content + '<|end|>'``, so a
        list content raises ``TypeError``.  Images become one
        ``<|image_k|>`` placeholder each, numbered in message order, and the
        text follows — the vendor's own documented form.  The processor
        asserts one placeholder per supplied image.
        """
        flattened: list[dict] = []
        index = 0
        for message in chat_messages:
            content = message.get("content")
            if isinstance(content, str):
                flattened.append(
                    {"role": message.get("role"), "content": content})
                continue
            parts: list[str] = []
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image":
                    index += 1
                    parts.append(IMAGE_PLACEHOLDER.format(index=index))
                elif part.get("type") == "text":
                    parts.append(part.get("text", ""))
            flattened.append(
                {"role": message.get("role"), "content": "".join(parts)})
        return flattened

    def extra_generation_kwargs(self) -> dict:
        # See module docstring item 6: without this the vendor's
        # prepare_inputs_for_generation forwards num_logits_to_keep=None
        # into a `-None` slice, and prefill materialises full-sequence
        # logits over a 200064-token vocabulary.
        return {"num_logits_to_keep": 1}

    def select_model_class(self):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Resolving the class IS the act of running the pinned remote code,
        # so trust_remote_code is not a parameter here; the revision pin is
        # what makes that safe.
        return get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMForCausalLM", self.model_spec.model_id,
            revision=self._pretrained_kwargs().get("revision"))

    # --- loading -------------------------------------------------------
    def load_model(self, model_cls, model_id: str, kwargs: dict):
        """Direct bf16 safetensors load (frozen fix #2), verified.

        ``kwargs`` (the base's ``torch_dtype`` / ``device_map``) is
        deliberately ignored: ``from_pretrained`` is never called, so dtype
        and placement are applied explicitly at the end instead. The
        checkpoint is already uniformly bf16, so the cast is a no-op that
        is asserted rather than relied upon.
        """
        import torch
        from transformers import AutoConfig, GenerationConfig

        from causal_mllm.replay.checkpoint_size import resolve_snapshot_dir

        pretrained = self._pretrained_kwargs()
        config = AutoConfig.from_pretrained(model_id, **pretrained)
        self.phi4_shims.extend(force_sdpa(config))

        tied = normalize_tied_weights_keys(model_cls)
        if tied:
            self.phi4_shims.append(tied)
        bound = bind_inner_prepare_inputs(model_cls)
        if bound:
            self.phi4_shims.append(bound)
        self.phi4_shims.extend(shim_cache_api())

        model = model_cls(config)
        snapshot = resolve_snapshot_dir(model_id,
                                        revision=pretrained.get("revision"))
        state, dtype_histogram = self._read_checkpoint(snapshot)

        result = model.load_state_dict(state, strict=False, assign=True)
        checkpoint_keys = set(state.keys())
        del state
        model.tie_weights()

        # The shipped generation config carries the <|end|> stop token that
        # config.json does not (module docstring item 5).
        generation_config = GenerationConfig.from_pretrained(
            model_id, **pretrained)
        model.generation_config = generation_config

        report = dict(self._checkpoint_summary)
        report.update(
            self._verify_load(model, result, dtype_histogram, checkpoint_keys))
        report["generation_config_eos_token_id"] = _as_id_list(
            generation_config.eos_token_id)
        report["generation_config_source"] = "generation_config.json"
        self.phi4_load_report = report

        dtype = getattr(torch, self.model_spec.dtype)
        model = model.to(dtype).to(self.device)
        report["device"] = str(next(model.parameters()).device)
        report["dtype"] = str(next(model.parameters()).dtype)

        vision_patched = patch_vision_attention()
        self.phi4_shims.append(
            f"vision tower _flash_attention_forward -> sdpa "
            f"({vision_patched} class(es))")
        report["vision_attention_classes_patched"] = vision_patched
        report["gradient_checkpointing_modules_disabled"] = \
            disable_gradient_checkpointing(model)
        report["attention_classes_instantiated"] = sorted(
            {type(m).__name__ for m in model.modules()
             if "Attention" in type(m).__name__})
        self._image_token_id = self._resolve_image_token_id()
        return model

    def _read_checkpoint(self, snapshot) -> tuple[dict, dict]:
        """Load every index-referenced shard; report the dtype histogram.

        Parameter counts come from the safetensors HEADERS (dtype + shape),
        never from the response and never from the constructed model, so
        the declared size stays independent of what the shim happened to
        build.
        """
        from safetensors.torch import load_file

        from causal_mllm.replay.checkpoint_size import _shard_files

        shards = _shard_files(snapshot)
        state: dict = {}
        histogram: dict = {}
        parameter_count = 0
        for shard in shards:
            with open(shard, "rb") as handle:
                length = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(length))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                histogram[meta["dtype"]] = histogram.get(meta["dtype"], 0) + 1
                shape = meta.get("shape") or []
                count = 1
                for dimension in shape:
                    count *= dimension
                parameter_count += count
            state.update(load_file(str(shard), device="cpu"))
        if not state:
            raise ReplayError(
                f"{snapshot}: shards contained no tensors — refusing to "
                f"construct a randomly initialised model")
        self._checkpoint_summary = {
            "shards": [shard.name for shard in shards],
            "checkpoint_tensors": len(state),
            "checkpoint_dtype_histogram": histogram,
            "checkpoint_parameter_count": parameter_count,
        }
        return state, histogram

    def _verify_load(self, model, result, dtype_histogram,
                     checkpoint_keys=()) -> dict:
        """Fail closed on anything that would silently corrupt outputs."""
        missing = sorted(result.missing_keys)
        unexpected = sorted(result.unexpected_keys)
        # remove_duplicate=False is REQUIRED here: named_parameters()
        # dedups by tensor identity, so a *successfully* tied lm_head.weight
        # would vanish from the mapping and read as "not tied".
        params = dict(model.named_parameters(remove_duplicate=False))

        head = params.get(TIED_TARGET)
        source = params.get(TIED_SOURCE)
        tied = bool(head is not None and source is not None
                    and head.data_ptr() == source.data_ptr())
        meta_left = sorted(n for n, p in params.items()
                           if p.device.type == "meta")
        # A missing key is only acceptable if it is the tied head (which
        # tie_weights() has just populated) or a non-persistent buffer.
        buffers = {n for n, _ in model.named_buffers()}
        unexplained = [k for k in missing
                       if k != TIED_TARGET and k not in buffers]

        problems = []
        if head is None or source is None:
            problems.append(
                f"tied pair not found on the constructed model "
                f"({TIED_TARGET}={head is not None}, "
                f"{TIED_SOURCE}={source is not None})")
        elif not tied:
            problems.append(
                f"{TIED_TARGET} is not tied to {TIED_SOURCE} — the output "
                f"projection would be randomly initialised")
        if meta_left:
            problems.append(
                f"{len(meta_left)} parameter(s) still on the meta device: "
                f"{meta_left[:5]}")
        if unexplained:
            problems.append(
                f"{len(unexplained)} checkpoint key(s) never materialised: "
                f"{unexplained[:5]}")
        if unexpected:
            problems.append(
                f"{len(unexpected)} checkpoint tensor(s) matched no "
                f"parameter: {unexpected[:5]}")
        if set(dtype_histogram) - {"BF16"}:
            problems.append(
                f"checkpoint is not uniformly bf16: {dtype_histogram}")
        if problems:
            raise ReplayError(
                "Phi-4 direct checkpoint load failed verification: "
                + "; ".join(problems))

        audio_names = [n for n in model.state_dict() if "audio" in n.lower()]
        buffer_tensors = dict(model.named_buffers())
        # Exact reconciliation of the two counts. A few checkpoint tensors
        # are registered as BUFFERS by the model rather than parameters, so
        # the header pass counts them and `parameters()` does not. Naming
        # them keeps the difference auditable instead of leaving the two
        # totals to differ by an unexplained constant.
        stored_as_buffers = sorted(
            key for key in checkpoint_keys if key in buffer_tensors)
        return {
            # Tied weights are counted once here (``parameters()`` dedups by
            # tensor identity) and the tied pair is named explicitly below.
            "constructed_parameter_count":
                sum(p.numel() for p in model.parameters()),
            "buffer_count": len(buffer_tensors),
            "buffer_numel": sum(b.numel() for b in buffer_tensors.values()),
            "checkpoint_tensors_registered_as_buffers": stored_as_buffers,
            "registered_as_buffer_numel":
                sum(buffer_tensors[k].numel() for k in stored_as_buffers),
            "lm_head_tied_to_embed_tokens": tied,
            "tied_weight_names": [TIED_TARGET, TIED_SOURCE],
            "tied_weight_shape":
                list(head.shape) if head is not None else None,
            "tied_weight_numel":
                int(head.numel()) if head is not None else None,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "parameters_left_on_meta": len(meta_left),
            "attn_implementation": getattr(
                model.config, "_attn_implementation", None),
            # longrope: past this position the vendor switches rope factors
            # AND invalidates the KV cache mid-generation, which would
            # perturb outputs in a way the other families do not share.
            # Recorded so the preflight can assert the cap never reaches it.
            "rope_scaling_type": (
                (getattr(model.config, "rope_scaling", None) or {}).get(
                    "type")),
            "rope_switch_position": getattr(
                model.config, "original_max_position_embeddings", None),
            "max_position_embeddings": getattr(
                model.config, "max_position_embeddings", None),
            "audio_tower_present": bool(audio_names),
            "audio_tensor_count": len(audio_names),
        }

    def _resolve_image_token_id(self) -> int | None:
        """The expanded image-embedding token id, never guessed.

        ``Phi4MMProcessor`` exposes no ``image_token_id``, so the inherited
        counting path would report 0 image tokens for every vision variant.
        The vendor's own accessors are tried in order; each is guarded
        separately because ``special_image_token_id`` is a property that
        calls the tokenizer and can raise.
        """
        processor = self.processor
        tokenizer = getattr(processor, "tokenizer", None)
        accessors = (
            lambda: getattr(processor, "image_token_id", None),
            lambda: processor.get_special_image_token_id(),
            lambda: processor.special_image_token_id,
            lambda: tokenizer.convert_tokens_to_ids(IMAGE_SPECIAL_TOKEN),
        )
        for accessor in accessors:
            try:
                candidate = accessor()
            except Exception:
                continue
            if isinstance(candidate, int) and candidate >= 0:
                return candidate
        return None

    # --- accounting ----------------------------------------------------
    def count_image_tokens(self, inputs) -> int:
        # Phi4MMProcessor exposes no `image_token_id`, so the generic base
        # path would report 0 for every image.
        if self._image_token_id is None:
            return 0
        return int((inputs["input_ids"][0] == self._image_token_id).sum())

    def _eos_token_ids(self) -> set:
        # The base probe reads config/tokenizer only and would miss
        # <|end|> (200020), which is how this model actually stops.
        ids = set(super()._eos_token_ids())
        ids.update(_as_id_list(getattr(
            getattr(self.model, "generation_config", None),
            "eos_token_id", None)))
        return ids

    def _active_lora(self) -> dict:
        """Which bundled LoRA was in force for the generation just run."""
        try:
            from peft.tuners.lora.layer import LoraLayer
        except Exception:  # pragma: no cover - peft is a hard dep here
            return {"available": False}
        active, disabled = set(), None
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                adapter = getattr(module, "active_adapter", None)
                if isinstance(adapter, (list, tuple)):
                    active.update(str(a) for a in adapter)
                elif adapter is not None:
                    active.add(str(adapter))
                disabled = bool(getattr(module, "_disable_adapters", False))
                break
        return {"available": True, "active_adapters": sorted(active),
                "adapters_disabled": disabled}

    def adapter_diagnostics(self, text: str, inputs) -> dict:
        placeholder_count = len(re.findall(r"<\|image_\d+\|>", text))
        input_mode = None
        if "input_mode" in inputs:
            raw = inputs["input_mode"]
            input_mode = int(raw.flatten()[0]) if hasattr(raw, "flatten") \
                else int(raw)
        return {
            "image_special_token_id": self._image_token_id,
            "image_placeholders_in_rendered_text": placeholder_count,
            "images_supplied": self._last_image_count,
            # The processor asserts one placeholder per image; recording it
            # per generation means a silent mismatch is visible in evidence.
            "placeholders_match_supplied_images":
                None if self._last_image_count is None
                else placeholder_count == self._last_image_count,
            "image_token_count_after_expansion":
                self.count_image_tokens(inputs),
            "audio_special_token_present": AUDIO_SPECIAL_TOKEN in text,
            "input_mode": input_mode,
            "input_mode_is_vision_or_language":
                None if input_mode is None
                else input_mode in (INPUT_MODE_LANGUAGE, INPUT_MODE_VISION),
            "end_token_present": END_SPECIAL_TOKEN in text,
            "frozen_system_prompt_present_verbatim":
                self.config.system_prompt.strip() in text,
            "active_lora": self._active_lora(),
        }

    def serialize_messages(self, chat_messages: list[dict]):
        text, images, inputs = super().serialize_messages(chat_messages)
        # Recorded so the placeholder/image-count invariant is checkable
        # per generation rather than assumed.
        self._last_image_count = len(images)
        return text, images, inputs

    def extra_runtime_metadata(self) -> dict:
        return {
            "thinking_switch_available": False,
            "load_strategy": "shim_in_shared_env",
            "phi4_shims": list(self.phi4_shims),
            "phi4_load_report": dict(self.phi4_load_report),
            # Explicit deviation from the frozen artifact, not a silent one.
            "phi4_audio_tower": {
                "frozen_protocol_claim": "audio_tower_initialized: false",
                "observed": bool(self.phi4_load_report.get(
                    "audio_tower_present")),
                "audio_input_supplied": False,
                "explanation": (
                    "Phi4MMImageAudioEmbedding builds the audio tower "
                    "unconditionally and the checkpoint ships its weights; "
                    "the vision path itself routes through "
                    "audio_embed.audio_projection.vision. The tower is "
                    "therefore initialised. No audio input is supplied "
                    "(input_mode is VISION or LANGUAGE only)."),
            },
            "registry_declared_parameters":
                self.model_spec.size_metadata.get(
                    "architectural_parameters"),
        }


def _as_id_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value if isinstance(v, int)]
    return []
