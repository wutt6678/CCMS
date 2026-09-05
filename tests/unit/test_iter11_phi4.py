"""Iteration 11.4: Phi-4-multimodal adapter (shared-env shim).

CI-safe: no torch, no transformers, no peft and no checkpoint. The pure
shim semantics, the message flattening and the fail-closed load
verification are exercised directly with fakes; the eligibility gates in
the preflight script are exercised on plain dicts; and the real-stack
behaviour (what the shims actually did to a 5.6B checkpoint on
transformers 5.14.1) is pinned from the committed preflight artifact.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from causal_mllm.replay import ReplayConfig, ReplayError, build_adapter
from causal_mllm.replay.adapters.phi4_multimodal import (
    AUDIO_SPECIAL_TOKEN,
    END_SPECIAL_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_SPECIAL_TOKEN,
    IMAGE_SPECIAL_TOKEN_ID,
    INPUT_MODE_LANGUAGE,
    INPUT_MODE_VISION,
    TIED_SOURCE,
    TIED_TARGET,
    TIED_WEIGHTS_KEYS,
    Phi4MultimodalAdapter,
    cache_get_usable_length,
    normalize_tied_weights_keys,
)
from causal_mllm.replay.registry import (
    DEFAULT_LOCK,
    is_immutable_revision,
    load_lock,
    resolve_model,
)

ROOT = Path(__file__).resolve().parents[2]
PHI4_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PREFLIGHT = (ROOT / "outputs" / "iteration_11" / "preflight" / "phi4_mm"
             / "preflight.json")
PROTOCOL = ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"

# Header-derived counts from the committed preflight artifact.
CHECKPOINT_PARAMS = 5574460384
CONSTRUCTED_PARAMS = 5574460224
ROPE_SWITCH = 4096
FROZEN_CAP = 1536


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preflight = _load_script("iter11_model_preflight")


@pytest.fixture(autouse=True)
def _block_peft(monkeypatch):
    """Keep this module torch-free.

    ``_active_lora`` imports ``peft.tuners.lora.layer``, which pulls in
    torch; no unit test in this repo does. The import is blocked so the
    adapter takes its documented graceful-degradation branch, and the REAL
    LoRA state is pinned from the committed preflight artifact instead —
    which is stronger evidence than a fake module object would be. The
    preflight's LoRA *gates* are still unit-tested directly on plain dicts
    in ``TestPreflightGates``.
    """
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.layer", None)


# --- fakes -------------------------------------------------------------
class _FakeDevice:
    def __init__(self, name="cpu"):
        self.type = name.split(":")[0]


class _FakeTensor:
    def __init__(self, ptr, numel=1000, shape=(20, 50), device="cpu"):
        self._ptr = ptr
        self._numel = numel
        self.shape = shape
        self.device = _FakeDevice(device)

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._numel


class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeResult:
    def __init__(self, missing=(), unexpected=()):
        self.missing_keys = list(missing)
        self.unexpected_keys = list(unexpected)


class _FakeModel:
    """Mimics the parts of ``nn.Module`` that ``_verify_load`` reads.

    ``named_parameters(remove_duplicate=True)`` drops tensors already
    yielded, exactly like torch — which is what makes a *successfully*
    tied ``lm_head.weight`` disappear and what the regression test below
    pins.
    """

    def __init__(self, *, tied=True, meta=(), buffers=None, audio_keys=(),
                 attn="sdpa", numel=1000):
        embed = _FakeTensor(ptr=111, numel=numel)
        # Tying shares ONE parameter object, exactly as torch does; that is
        # what makes the deduplicating named_parameters() view drop the
        # head, which the regression test below pins.
        head = embed if tied else _FakeTensor(ptr=222, numel=numel)
        params = [(TIED_SOURCE, embed), (TIED_TARGET, head)]
        for name in meta:
            params.append((name, _FakeTensor(ptr=333, device="meta")))
        self._params = params
        self._buffers = {n: _FakeTensor(ptr=900 + i, numel=80)
                         for i, n in enumerate(buffers or [])}
        self._state_dict_keys = [n for n, _ in params] + list(audio_keys)
        self.config = _FakeConfig(_attn_implementation=attn)
        self.generation_config = _FakeConfig(eos_token_id=None)

    def named_parameters(self, remove_duplicate=True):
        seen = set()
        for name, tensor in self._params:
            if remove_duplicate and id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield name, tensor

    def parameters(self):
        seen, out = set(), []
        for _, tensor in self._params:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            out.append(tensor)
        return iter(out)

    def named_buffers(self):
        return iter(self._buffers.items())

    def modules(self):
        return iter(())

    def state_dict(self):
        return {k: None for k in self._state_dict_keys}


class _WorkingTokenizer:
    """A tokenizer that resolves the image special token."""

    def convert_tokens_to_ids(self, token):
        assert token == IMAGE_SPECIAL_TOKEN
        return IMAGE_SPECIAL_TOKEN_ID


class _FakeCache:
    def __init__(self, previous, max_length=None):
        self._previous = previous
        self._max = max_length
        self.requested_layer = None

    def get_max_length(self):
        return self._max

    def get_seq_length(self, layer_idx=0):
        self.requested_layer = layer_idx
        return self._previous


class _FakeBoolVec:
    def __init__(self, flags):
        self.flags = flags

    def sum(self):
        return sum(self.flags)


class _FakeIdRow:
    def __init__(self, ids):
        self.ids = ids

    def __eq__(self, other):
        return _FakeBoolVec([i == other for i in self.ids])


class _FakeMode:
    def __init__(self, value):
        self.value = [value]

    def flatten(self):
        return self.value


class _FakeInputs:
    def __init__(self, ids, input_mode=None):
        self._data = {"input_ids": [_FakeIdRow(ids)]}
        if input_mode is not None:
            self._data["input_mode"] = _FakeMode(input_mode)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


def _adapter(**spec_kwargs) -> Phi4MultimodalAdapter:
    spec = resolve_model("phi4_mm")
    if spec_kwargs:
        spec = dataclasses.replace(spec, **spec_kwargs)
    return Phi4MultimodalAdapter(ReplayConfig(), spec)


def _messages():
    """The shape ``build_chat_messages`` emits: parts list, images first."""
    return [
        {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
        {"role": "user", "content": [
            {"type": "image", "image": "a.png"},
            {"type": "text", "text": "first turn"}]},
        {"role": "user", "content": [
            {"type": "image", "image": "b.png"},
            {"type": "text", "text": "q*"}]},
    ]


# ---------------------------------------------------------------------
# Registry dispatch and frozen-protocol conformance
# ---------------------------------------------------------------------
class TestRegistryDispatch:
    def test_registry_dispatches_to_the_phi4_adapter(self):
        adapter = build_adapter(resolve_model("phi4_mm"), ReplayConfig())
        assert isinstance(adapter, Phi4MultimodalAdapter)
        assert adapter.adapter_name == "phi4_multimodal"

    def test_frozen_registry_entry_is_remote_code_bf16_unquantized(self):
        spec = resolve_model("phi4_mm")
        assert spec.trust_remote_code is True
        assert spec.dtype == "bfloat16"
        assert spec.quantization == "none"
        assert spec.thinking_mode is False
        assert spec.license == "MIT"

    def test_quantized_phi4_fails_closed(self):
        with pytest.raises(ReplayError, match="out of scope"):
            _adapter(quantization="int4")

    def test_no_thinking_switch_is_passed(self):
        # The vendored template reads only role/content/tools.
        assert _adapter().chat_template_kwargs() == {}
        assert _adapter().extra_runtime_metadata()[
            "thinking_switch_available"] is False

    def test_logit_slicing_kwarg_is_explicit(self):
        # transformers only auto-sets `logits_to_keep` when forward
        # advertises that exact name; this model names it
        # `num_logits_to_keep`, whose vendor default of None would reach a
        # `-None` slice.
        assert _adapter().extra_generation_kwargs() == {
            "num_logits_to_keep": 1}

    def test_load_strategy_matches_the_frozen_protocol(self):
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert _adapter().extra_runtime_metadata()["load_strategy"] == \
            frozen["phi4_load_strategy"]["decision"] == "shim_in_shared_env"

    def test_audio_tower_deviation_is_declared_not_silent(self):
        # The frozen artifact claims audio_tower_initialized: false. The
        # checkpoint disagrees, so the adapter must record BOTH rather
        # than quietly contradicting a frozen protocol.
        block = _adapter().extra_runtime_metadata()["phi4_audio_tower"]
        assert block["frozen_protocol_claim"] == \
            "audio_tower_initialized: false"
        assert block["audio_input_supplied"] is False
        assert "audio_projection.vision" in block["explanation"]


# ---------------------------------------------------------------------
# Message flattening: the template concatenates content as a string
# ---------------------------------------------------------------------
class TestMessageFlattening:
    def test_list_content_becomes_a_string(self):
        flat = _adapter().template_messages(_messages())
        assert all(isinstance(m["content"], str) for m in flat)

    def test_roles_and_turn_count_are_preserved(self):
        source = _messages()
        flat = _adapter().template_messages(source)
        assert [m["role"] for m in flat] == [m["role"] for m in source]

    def test_images_are_numbered_sequentially_across_messages(self):
        flat = _adapter().template_messages(_messages())
        assert flat[1]["content"] == "<|image_1|>first turn"
        assert flat[2]["content"] == "<|image_2|>q*"

    def test_placeholder_precedes_text_matching_part_order(self):
        # build_chat_messages appends images before the turn text, so the
        # flattened string must keep that order.
        flat = _adapter().template_messages(_messages())
        assert flat[1]["content"].startswith("<|image_")
        assert flat[1]["content"].endswith("first turn")

    def test_system_prompt_survives_verbatim(self):
        flat = _adapter().template_messages(_messages())
        assert flat[0]["content"] == "SYS"

    def test_string_content_passes_through_untouched(self):
        source = [{"role": "user", "content": "already a string"}]
        assert _adapter().template_messages(source) == source

    def test_textless_image_turn_yields_only_the_placeholder(self):
        flat = _adapter().template_messages(
            [{"role": "user", "content": [{"type": "image", "image": "a"}]}])
        assert flat[0]["content"] == "<|image_1|>"

    def test_unknown_and_malformed_parts_are_ignored(self):
        flat = _adapter().template_messages([{"role": "user", "content": [
            "not-a-dict", {"type": "audio"}, {"type": "text"}]}])
        assert flat[0]["content"] == ""

    def test_flattening_does_not_mutate_the_semantic_messages(self):
        # The semantic hash is computed from the ORIGINAL messages, so
        # flattening must not touch them.
        source = _messages()
        before = json.dumps(source, sort_keys=True)
        _adapter().template_messages(source)
        assert json.dumps(source, sort_keys=True) == before

    def test_no_audio_placeholder_is_ever_emitted(self):
        flat = _adapter().template_messages(_messages())
        assert not any(AUDIO_SPECIAL_TOKEN in m["content"] for m in flat)


# ---------------------------------------------------------------------
# Shim semantics (pure; no transformers import)
# ---------------------------------------------------------------------
class TestCacheShimSemantics:
    def test_unbounded_cache_returns_the_cached_length(self):
        # DynamicCache.get_max_length() is None, so this is the value the
        # 4.x stack produced on every decode step.
        assert cache_get_usable_length(_FakeCache(17), 1) == 17

    def test_layer_index_is_forwarded(self):
        cache = _FakeCache(5)
        cache_get_usable_length(cache, 1, layer_idx=31)
        assert cache.requested_layer == 31

    def test_clips_when_the_new_tokens_would_overflow(self):
        cache = _FakeCache(previous=90, max_length=100)
        assert cache_get_usable_length(cache, 20) == 80

    def test_no_clipping_below_the_maximum(self):
        cache = _FakeCache(previous=10, max_length=100)
        assert cache_get_usable_length(cache, 20) == 10

    def test_exactly_at_the_maximum_is_not_clipped(self):
        cache = _FakeCache(previous=80, max_length=100)
        assert cache_get_usable_length(cache, 20) == 80


class TestTiedWeightsNormalization:
    def test_vendor_list_form_becomes_the_5x_dict(self):
        class Vendor:
            _tied_weights_keys = ["lm_head.weight"]

        assert normalize_tied_weights_keys(Vendor) is not None
        assert Vendor._tied_weights_keys == TIED_WEIGHTS_KEYS == {
            TIED_TARGET: TIED_SOURCE}

    def test_dict_form_is_left_alone(self):
        class Already:
            _tied_weights_keys = {"a.weight": "b.weight"}

        assert normalize_tied_weights_keys(Already) is None
        assert Already._tied_weights_keys == {"a.weight": "b.weight"}

    def test_absent_mapping_is_left_alone(self):
        class None_:
            _tied_weights_keys = None

        assert normalize_tied_weights_keys(None_) is None

    def test_normalization_is_idempotent(self):
        class Vendor:
            _tied_weights_keys = ["lm_head.weight"]

        normalize_tied_weights_keys(Vendor)
        assert normalize_tied_weights_keys(Vendor) is None


# ---------------------------------------------------------------------
# The fail-closed load verification
# ---------------------------------------------------------------------
class TestLoadVerification:
    def _verify(self, model, result=None, histogram=None,
                checkpoint_keys=()):
        adapter = _adapter()
        return adapter._verify_load(
            model, result or _FakeResult(missing=[TIED_TARGET]),
            histogram if histogram is not None else {"BF16": 2047},
            checkpoint_keys)

    def test_a_clean_load_passes(self):
        report = self._verify(_FakeModel())
        assert report["lm_head_tied_to_embed_tokens"] is True
        assert report["parameters_left_on_meta"] == 0
        assert report["attn_implementation"] == "sdpa"

    def test_the_tied_head_is_not_read_as_missing_because_of_dedup(self):
        # REGRESSION: named_parameters() dedups by tensor identity, so a
        # SUCCESSFULLY tied lm_head.weight vanishes from the mapping. An
        # earlier version read that as "not tied" and rejected a good
        # load; the converse bug (accepting an untied head) is worse.
        model = _FakeModel(tied=True)
        deduped = dict(model.named_parameters())
        assert TIED_TARGET not in deduped  # the trap
        assert self._verify(model)["lm_head_tied_to_embed_tokens"] is True

    def test_an_untied_output_projection_is_rejected(self):
        with pytest.raises(ReplayError, match="not tied"):
            self._verify(_FakeModel(tied=False))

    def test_a_parameter_left_on_meta_is_rejected(self):
        with pytest.raises(ReplayError, match="meta device"):
            self._verify(_FakeModel(meta=["model.layers.0.mlp.weight"]))

    def test_unmatched_checkpoint_tensors_are_rejected(self):
        with pytest.raises(ReplayError, match="matched no parameter"):
            self._verify(_FakeModel(), _FakeResult(
                missing=[TIED_TARGET], unexpected=["stray.weight"]))

    def test_an_unexplained_missing_key_is_rejected(self):
        with pytest.raises(ReplayError, match="never materialised"):
            self._verify(_FakeModel(), _FakeResult(
                missing=[TIED_TARGET, "model.layers.7.self_attn.q_proj"]))

    def test_a_missing_buffer_is_tolerated(self):
        # Non-persistent buffers (rotary inv_freq) are legitimately absent
        # from the checkpoint and must not fail the load.
        model = _FakeModel(buffers=["model.layers.0.rotary_emb.inv_freq"])
        report = self._verify(model, _FakeResult(
            missing=[TIED_TARGET, "model.layers.0.rotary_emb.inv_freq"]))
        assert report["buffer_count"] == 1

    def test_a_mixed_precision_checkpoint_is_rejected(self):
        with pytest.raises(ReplayError, match="not uniformly bf16"):
            self._verify(_FakeModel(), histogram={"BF16": 2000, "F32": 47})

    def test_the_two_parameter_counts_reconcile_exactly(self):
        # A few checkpoint tensors are registered as BUFFERS by the model,
        # so the header pass counts them and `parameters()` does not.
        # Naming them makes the difference auditable rather than leaving
        # the two totals to differ by an unexplained constant.
        report = self._verify(
            _FakeModel(buffers=["global_mean", "global_invstd"], numel=1000),
            checkpoint_keys=["global_mean", "global_invstd",
                             "model.embed_tokens.weight"])
        assert report["constructed_parameter_count"] == 1000
        assert report["checkpoint_tensors_registered_as_buffers"] == [
            "global_invstd", "global_mean"]
        assert report["registered_as_buffer_numel"] == 160
        assert report["buffer_count"] == 2
        assert report["buffer_numel"] == 160

    def test_tied_pair_identity_is_recorded(self):
        report = self._verify(_FakeModel())
        assert report["tied_weight_names"] == [TIED_TARGET, TIED_SOURCE]
        assert report["tied_weight_numel"] == 1000

    def test_audio_tower_presence_is_reported(self):
        report = self._verify(_FakeModel(audio_keys=[
            "model.embed_tokens_extend.audio_embed.encoder.x"]))
        assert report["audio_tower_present"] is True
        assert report["audio_tensor_count"] == 1


# ---------------------------------------------------------------------
# Token accounting and stop tokens
# ---------------------------------------------------------------------
class TestVendorTokenConstants:
    """Pin the ids/strings against the vendored processing_phi4mm.py."""

    def test_special_tokens_match_the_vendored_processor(self):
        assert IMAGE_SPECIAL_TOKEN == "<|endoftext10|>"
        assert AUDIO_SPECIAL_TOKEN == "<|endoftext11|>"
        assert END_SPECIAL_TOKEN == "<|end|>"
        assert IMAGE_SPECIAL_TOKEN_ID == 200010

    def test_input_modes_match_the_vendored_enum(self):
        assert INPUT_MODE_LANGUAGE == 0
        assert INPUT_MODE_VISION == 1

    def test_placeholder_is_the_backward_compatible_form(self):
        # The processor regex-normalises `<|image_\d+|>` to the internal
        # image token, then expands it per image.
        assert IMAGE_PLACEHOLDER.format(index=3) == "<|image_3|>"

    def test_tying_maps_the_head_onto_the_input_embedding(self):
        assert TIED_WEIGHTS_KEYS == {TIED_TARGET: TIED_SOURCE} == {
            "lm_head.weight": "model.embed_tokens.weight"}


class TestTokenAccounting:
    def test_counts_the_expanded_image_token(self):
        adapter = _adapter()
        adapter._image_token_id = IMAGE_SPECIAL_TOKEN_ID
        ids = [1, 200010, 200010, 200010, 5]
        assert adapter.count_image_tokens(_FakeInputs(ids)) == 3

    def test_an_unresolved_image_token_reports_zero_not_a_guess(self):
        adapter = _adapter()
        assert adapter._image_token_id is None
        assert adapter.count_image_tokens(
            _FakeInputs([200010, 200010])) == 0

    def test_the_base_processor_attribute_is_absent_for_phi4(self):
        # Phi4MMProcessor exposes no `image_token_id`, so the inherited
        # counting path would report 0 for every image; the override
        # resolves the id from the processor's own accessors instead.
        adapter = _adapter()
        adapter.processor = _FakeConfig(tokenizer=None)
        assert getattr(adapter.processor, "image_token_id", None) is None
        assert adapter._resolve_image_token_id() is None

    def test_the_id_is_taken_from_the_vendor_accessor(self):
        adapter = _adapter()
        adapter.processor = _FakeConfig(
            tokenizer=None,
            get_special_image_token_id=lambda: IMAGE_SPECIAL_TOKEN_ID)
        assert adapter._resolve_image_token_id() == IMAGE_SPECIAL_TOKEN_ID

    def test_the_id_falls_back_to_the_tokenizer(self):
        adapter = _adapter()
        adapter.processor = _FakeConfig(tokenizer=_WorkingTokenizer())
        assert adapter._resolve_image_token_id() == IMAGE_SPECIAL_TOKEN_ID

    def test_a_raising_property_does_not_abort_the_resolution(self):
        # `special_image_token_id` is a property that calls the tokenizer,
        # so it can raise; each accessor is guarded separately and the
        # tokenizer fallback still resolves the id.
        class _Processor:
            tokenizer = _WorkingTokenizer()

            @property
            def special_image_token_id(self):
                raise RuntimeError("tokenizer exploded")

            def get_special_image_token_id(self):
                raise RuntimeError("tokenizer exploded")

        adapter = _adapter()
        adapter.processor = _Processor()
        assert adapter._resolve_image_token_id() == IMAGE_SPECIAL_TOKEN_ID

    def test_eos_includes_the_end_token_the_config_omits(self):
        # config.json declares only 199999; generation_config.json adds
        # 200020 (<|end|>), which is how this model actually stops.
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter.model.config = _FakeConfig(eos_token_id=199999)
        adapter.model.generation_config = _FakeConfig(
            eos_token_id=[200020, 199999])
        adapter.processor = _FakeConfig(tokenizer=None)
        ids = adapter._eos_token_ids()
        assert 200020 in ids and 199999 in ids

    def test_eos_survives_a_scalar_generation_config(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter.model.config = _FakeConfig(eos_token_id=199999)
        adapter.model.generation_config = _FakeConfig(eos_token_id=200020)
        adapter.processor = _FakeConfig(tokenizer=None)
        assert 200020 in adapter._eos_token_ids()

    def test_eos_survives_a_missing_generation_config(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter.model.config = _FakeConfig(eos_token_id=199999)
        adapter.model.generation_config = None
        adapter.processor = _FakeConfig(tokenizer=None)
        assert isinstance(adapter._eos_token_ids(), set)

    def test_lora_reporting_degrades_gracefully_without_peft(self,
                                                            monkeypatch):
        monkeypatch.setitem(sys.modules, "peft.tuners.lora.layer", None)
        adapter = _adapter()
        adapter.model = _FakeModel()
        assert adapter._active_lora() == {"available": False}


# ---------------------------------------------------------------------
# Per-generation diagnostics
# ---------------------------------------------------------------------
class TestAdapterDiagnostics:
    def _diag(self, *, placeholders, images, input_mode, ids=()):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter._image_token_id = 200010
        adapter._last_image_count = images
        text = "<|system|>SYS<|end|>" + "".join(
            f"<|image_{i}|>" for i in range(1, placeholders + 1))
        return adapter.adapter_diagnostics(
            text, _FakeInputs(list(ids), input_mode=input_mode))

    def test_vision_arm_reports_a_consistent_image_accounting(self):
        diag = self._diag(placeholders=1, images=1,
                          input_mode=INPUT_MODE_VISION,
                          ids=[200010] * 545 + [1, 2])
        assert diag["image_placeholders_in_rendered_text"] == 1
        assert diag["images_supplied"] == 1
        assert diag["placeholders_match_supplied_images"] is True
        assert diag["image_token_count_after_expansion"] == 545
        assert diag["input_mode"] == INPUT_MODE_VISION
        assert diag["input_mode_is_vision_or_language"] is True
        assert diag["end_token_present"] is True
        assert diag["frozen_system_prompt_present_verbatim"] is False

    def test_language_arm_reports_no_image_at_all(self):
        diag = self._diag(placeholders=0, images=0,
                          input_mode=INPUT_MODE_LANGUAGE)
        assert diag["image_placeholders_in_rendered_text"] == 0
        assert diag["image_token_count_after_expansion"] == 0
        assert diag["input_mode"] == INPUT_MODE_LANGUAGE
        assert diag["placeholders_match_supplied_images"] is True

    def test_a_placeholder_image_mismatch_is_visible(self):
        diag = self._diag(placeholders=2, images=1,
                          input_mode=INPUT_MODE_VISION)
        assert diag["placeholders_match_supplied_images"] is False

    def test_before_any_serialization_the_check_is_none_not_true(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter._image_token_id = 200010
        diag = adapter.adapter_diagnostics(
            "<|end|>", _FakeInputs([], input_mode=INPUT_MODE_LANGUAGE))
        assert diag["images_supplied"] is None
        assert diag["placeholders_match_supplied_images"] is None

    def test_a_speech_input_mode_is_flagged(self):
        diag = self._diag(placeholders=0, images=0, input_mode=3)
        assert diag["input_mode_is_vision_or_language"] is False

    def test_a_missing_input_mode_is_not_guessed(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter._image_token_id = 200010
        diag = adapter.adapter_diagnostics("<|end|>", _FakeInputs([]))
        assert diag["input_mode"] is None
        assert diag["input_mode_is_vision_or_language"] is None

    def test_an_audio_placeholder_would_be_reported(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter._image_token_id = 200010
        diag = adapter.adapter_diagnostics(
            f"<|audio_1|>{AUDIO_SPECIAL_TOKEN}<|end|>", _FakeInputs([]))
        assert diag["audio_special_token_present"] is True

    def test_the_frozen_prompt_is_checked_verbatim(self):
        adapter = _adapter()
        adapter.model = _FakeModel()
        adapter._image_token_id = 200010
        frozen = adapter.config.system_prompt.strip()
        present = adapter.adapter_diagnostics(
            f"<|system|>{frozen}<|end|>", _FakeInputs([]))
        absent = adapter.adapter_diagnostics(
            "<|system|>something else<|end|>", _FakeInputs([]))
        assert present["frozen_system_prompt_present_verbatim"] is True
        assert absent["frozen_system_prompt_present_verbatim"] is False

    def test_serialize_records_the_supplied_image_count(self, monkeypatch):
        # The placeholder/image invariant is only checkable per generation
        # if serialize_messages records how many images it opened.
        from causal_mllm.replay.adapters.base import HFAdapterBase

        adapter = _adapter()
        assert adapter._last_image_count is None
        monkeypatch.setattr(
            HFAdapterBase, "serialize_messages",
            lambda self, chat: ("rendered", ["a.png", "b.png"], "inputs"))
        assert adapter.serialize_messages([]) == \
            ("rendered", ["a.png", "b.png"], "inputs")
        assert adapter._last_image_count == 2


# ---------------------------------------------------------------------
# Preflight eligibility gates (pure dicts; no GPU)
# ---------------------------------------------------------------------
def _attempt(**overrides):
    base = {
        "response_sha256": "a" * 64, "response_chars": 120,
        "response_head": "x", "input_token_count": 709,
        "image_token_count": 545, "output_token_count": 40,
        "finish_reason": "eos", "hit_max_new_tokens": False,
        "serialized_prompt_hash": "b" * 64,
        "semantic_prompt_hash": "c" * 64,
        "ordered_image_hashes": ["d" * 64],
        "adapter_diagnostics": {
            "placeholders_match_supplied_images": True,
            "audio_special_token_present": False,
            "input_mode_is_vision_or_language": True,
            "end_token_present": True,
            "frozen_system_prompt_present_verbatim": True,
            "input_mode": INPUT_MODE_VISION,
            "active_lora": {"available": True,
                            "active_adapters": ["vision"],
                            "adapters_disabled": False},
        },
    }
    base.update(overrides)
    return base


def _smoke(text_attempt=None):
    text = text_attempt or _attempt(
        input_token_count=178, image_token_count=0, ordered_image_hashes=[],
        adapter_diagnostics={
            "placeholders_match_supplied_images": True,
            "audio_special_token_present": False,
            "input_mode_is_vision_or_language": True,
            "end_token_present": True,
            "frozen_system_prompt_present_verbatim": True,
            "input_mode": INPUT_MODE_LANGUAGE,
            "active_lora": {"available": True,
                            "active_adapters": ["vision"],
                            "adapters_disabled": True},
        })
    return [{"variant": preflight.VISION_VARIANT, "repeats": 2,
             "attempts": [_attempt(), _attempt()], "deterministic": True,
             "n_distinct_responses": 1},
            {"variant": preflight.TEXT_VARIANT, "repeats": 2,
             "attempts": [text, text], "deterministic": True,
             "n_distinct_responses": 1}]


SIZE_META = {"checkpoint_parameter_count": CHECKPOINT_PARAMS,
             "vision_parameters": 415000000, "unclassified_parameters": 0}
LOAD_REPORT = {"lm_head_tied_to_embed_tokens": True,
               "parameters_left_on_meta": 0, "unexpected_keys": [],
               "missing_keys": [TIED_TARGET], "attn_implementation": "sdpa",
               "checkpoint_dtype_histogram": {"BF16": 2047},
               "checkpoint_parameter_count": CHECKPOINT_PARAMS,
               "rope_switch_position": ROPE_SWITCH}


class TestPreflightGates:
    def _problems(self, smoke=None, meta=None, cap=FROZEN_CAP):
        return preflight._check_smoke(
            smoke or _smoke(), meta or dict(SIZE_META),
            {"phi4_load_report": dict(LOAD_REPORT)}, cap)

    def test_a_conformant_phi4_smoke_passes(self):
        assert self._problems() == []

    def test_a_mismatched_image_placeholder_count_fails(self):
        smoke = _smoke()
        smoke[0]["attempts"][0]["adapter_diagnostics"][
            "placeholders_match_supplied_images"] = False
        assert any("placeholders" in p for p in self._problems(smoke=smoke))

    def test_an_audio_placeholder_fails(self):
        smoke = _smoke()
        smoke[0]["attempts"][0]["adapter_diagnostics"][
            "audio_special_token_present"] = True
        assert any("vision-only" in p for p in self._problems(smoke=smoke))

    def test_a_missing_end_terminator_fails(self):
        smoke = _smoke()
        smoke[0]["attempts"][0]["adapter_diagnostics"][
            "end_token_present"] = False
        assert any("<|end|>" in p for p in self._problems(smoke=smoke))

    def test_a_cross_modal_prompt_that_degraded_to_language_fails(self):
        smoke = _smoke()
        diag = smoke[0]["attempts"][0]["adapter_diagnostics"]
        diag["input_mode"] = INPUT_MODE_LANGUAGE
        assert any("input_mode" in p for p in self._problems(smoke=smoke))

    def test_an_inactive_vision_lora_fails(self):
        smoke = _smoke()
        smoke[0]["attempts"][0]["adapter_diagnostics"]["active_lora"] = {
            "available": True, "active_adapters": ["speech"],
            "adapters_disabled": False}
        assert any("vision LoRA" in p for p in self._problems(smoke=smoke))

    def test_a_live_lora_on_the_text_arm_fails(self):
        smoke = _smoke()
        smoke[1]["attempts"][0]["adapter_diagnostics"]["active_lora"] = {
            "available": True, "active_adapters": ["vision"],
            "adapters_disabled": False}
        assert any("language-only" in p for p in self._problems(smoke=smoke))

    def test_an_untied_head_fails(self):
        report = dict(LOAD_REPORT, lm_head_tied_to_embed_tokens=False)
        problems = preflight._check_smoke(
            _smoke(), dict(SIZE_META), {"phi4_load_report": report},
            FROZEN_CAP)
        assert any("not tied" in p for p in problems)

    def test_a_parameter_left_on_meta_fails(self):
        report = dict(LOAD_REPORT, parameters_left_on_meta=3)
        problems = preflight._check_smoke(
            _smoke(), dict(SIZE_META), {"phi4_load_report": report},
            FROZEN_CAP)
        assert any("meta" in p for p in problems)

    def test_a_non_sdpa_attention_path_fails(self):
        report = dict(LOAD_REPORT, attn_implementation="flash_attention_2")
        problems = preflight._check_smoke(
            _smoke(), dict(SIZE_META), {"phi4_load_report": report},
            FROZEN_CAP)
        assert any("expected 'sdpa'" in p for p in problems)

    def test_disagreement_between_the_two_size_counts_fails(self):
        meta = dict(SIZE_META, checkpoint_parameter_count=CHECKPOINT_PARAMS + 1)
        problems = preflight._check_smoke(
            _smoke(), meta, {"phi4_load_report": dict(LOAD_REPORT)},
            FROZEN_CAP)
        assert any("declared-size pass" in p for p in problems)

    def test_generation_crossing_the_longrope_switch_fails(self):
        # 2600-token prompt + the frozen 1536 cap passes 4096, where the
        # vendor swaps rope factors and drops the KV cache mid-sequence.
        smoke = _smoke()
        smoke[0]["attempts"][0]["input_token_count"] = 2600
        problems = self._problems(smoke=smoke)
        assert any("longrope" in p for p in problems)

    def test_generation_under_the_longrope_switch_passes(self):
        assert self._problems(cap=ROPE_SWITCH - 709) == []

    def test_families_without_a_load_report_are_not_checked(self):
        assert preflight._check_load_report({}, SIZE_META) == []
        assert preflight._check_load_report(None, SIZE_META) == []

    def test_families_without_a_switch_point_are_not_checked(self):
        report = dict(LOAD_REPORT)
        report.pop("rope_switch_position")
        assert preflight._check_rope_headroom(
            _smoke(), {"phi4_load_report": report}, FROZEN_CAP) == []

    def test_phi4_parity_notes_are_recorded(self):
        notes = preflight.parity_notes("phi4_multimodal")
        assert len(notes) > len(preflight.PARITY_NOTES)
        joined = " ".join(notes)
        for marker in ("shim_in_shared_env", "phi4_shims", "<|end|>",
                       "audio_tower_initialized", "num_logits_to_keep"):
            assert marker in joined, marker


# ---------------------------------------------------------------------
# Committed 11.4 preflight evidence (GPU3, full 1536 cap)
# ---------------------------------------------------------------------
PHI4_SIZE = {
    "total": 5574460384,
    "language": 4666493952,
    "vision": 441550016,
    "auxiliary": 466416416,
    "tensors": 2047,
    "shards": 3,
}
CROSS_MODAL = {"in": 709, "img": 545, "out": 92}
TEXT_ONLY = {"in": 178, "img": 0, "out": 67}
TIED_SHAPE = [200064, 3072]
TIED_NUMEL = 614596608
BUFFER_RECONCILIATION = 160


@pytest.fixture(scope="module")
def phi4_report():
    if not PREFLIGHT.exists():
        pytest.skip("phi4_mm preflight artifact not committed")
    return json.loads(PREFLIGHT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def phi4_load(phi4_report):
    return phi4_report["runtime_metadata"]["phi4_load_report"]


@pytest.fixture(scope="module")
def phi4_smoke(phi4_report):
    return {entry["variant"]: entry for entry in phi4_report["gpu_smoke"]}


class TestPhi4PreflightEvidence:
    def test_passed_with_no_problems(self, phi4_report):
        assert phi4_report["status"] == "PASS"
        assert phi4_report["problems"] == []
        assert phi4_report["model_key"] == "phi4_mm"

    def test_revision_is_immutable_and_locked(self, phi4_report):
        assert phi4_report["resolved_revision"] == PHI4_REVISION
        assert is_immutable_revision(PHI4_REVISION)
        assert phi4_report["processor_revision"] == PHI4_REVISION
        assert phi4_report["registry_revision"] == PHI4_REVISION
        if DEFAULT_LOCK.exists():
            assert load_lock(DEFAULT_LOCK)["phi4_mm"]["revision"] == \
                PHI4_REVISION
            assert is_immutable_revision(
                resolve_model("phi4_mm", confirmatory=True).revision)

    def test_measured_size_matches(self, phi4_report):
        size = phi4_report["size_metadata"]
        assert size["checkpoint_parameter_count"] == PHI4_SIZE["total"]
        assert size["language_parameters"] == PHI4_SIZE["language"]
        assert size["vision_parameters"] == PHI4_SIZE["vision"]
        assert size["auxiliary_parameters"] == PHI4_SIZE["auxiliary"]
        assert size["unclassified_parameters"] == 0
        assert size["n_tensors"] == PHI4_SIZE["tensors"]
        assert size["n_shards"] == PHI4_SIZE["shards"]
        assert size["revision_used"] == PHI4_REVISION
        assert size["inferred_from_response"] is False
        # The split must sum to the total, or a bucket is double-counting.
        assert size["language_parameters"] + size["vision_parameters"] \
            + size["auxiliary_parameters"] == PHI4_SIZE["total"]

    def test_vision_parameters_are_now_attributed(self, phi4_report):
        # Regression: `image_embed.*` was being swallowed by the
        # `embed_tokens` LANGUAGE marker, reporting vision=0 and failing
        # the multimodal eligibility gate.
        assert phi4_report["size_metadata"]["vision_parameters"] > 0

    def test_the_direct_load_received_every_weight(self, phi4_load):
        assert phi4_load["lm_head_tied_to_embed_tokens"] is True
        assert phi4_load["missing_keys"] == [TIED_TARGET]
        assert phi4_load["unexpected_keys"] == []
        assert phi4_load["parameters_left_on_meta"] == 0
        assert phi4_load["checkpoint_tensors"] == PHI4_SIZE["tensors"]
        assert phi4_load["checkpoint_dtype_histogram"] == {
            "BF16": PHI4_SIZE["tensors"]}

    def test_the_tied_pair_is_the_real_output_projection(self, phi4_load):
        assert phi4_load["tied_weight_names"] == [TIED_TARGET, TIED_SOURCE]
        assert phi4_load["tied_weight_shape"] == TIED_SHAPE
        assert phi4_load["tied_weight_numel"] == TIED_NUMEL
        assert TIED_SHAPE[0] * TIED_SHAPE[1] == TIED_NUMEL

    def test_the_two_parameter_counts_reconcile_exactly(self, phi4_load):
        delta = (phi4_load["checkpoint_parameter_count"]
                 - phi4_load["constructed_parameter_count"])
        assert delta == phi4_load["registered_as_buffer_numel"] == \
            BUFFER_RECONCILIATION
        # Named, so the delta is auditable rather than a mystery constant:
        # the conformer encoder's global_mean/global_invstd are stored as
        # tensors and registered as buffers.
        assert phi4_load["checkpoint_tensors_registered_as_buffers"] == [
            "model.embed_tokens_extend.audio_embed.encoder."
            "encoder_embedding.global_invstd",
            "model.embed_tokens_extend.audio_embed.encoder."
            "encoder_embedding.global_mean",
        ]

    def test_sdpa_was_selected_not_flash_attention(self, phi4_load):
        assert phi4_load["attn_implementation"] == "sdpa"
        assert "Phi4MMSdpaAttention" in \
            phi4_load["attention_classes_instantiated"]
        assert phi4_load["vision_attention_classes_patched"] == 1
        assert phi4_load["gradient_checkpointing_modules_disabled"] == 1

    def test_the_end_stop_token_was_loaded(self, phi4_load):
        # config.json declares only 199999; without generation_config.json
        # the model's real terminator <|end|> (200020) would be missing.
        assert phi4_load["generation_config_eos_token_id"] == [200020, 199999]
        assert phi4_load["generation_config_source"] == "generation_config.json"

    def test_longrope_switch_point_is_beyond_the_frozen_cap(self, phi4_load,
                                                            phi4_smoke):
        assert phi4_load["rope_scaling_type"] == "longrope"
        assert phi4_load["rope_switch_position"] == ROPE_SWITCH
        for entry in phi4_smoke.values():
            for attempt in entry["attempts"]:
                assert attempt["input_token_count"] + FROZEN_CAP <= ROPE_SWITCH

    def test_the_audio_tower_deviation_is_recorded(self, phi4_report,
                                                   phi4_load):
        # The frozen protocol claims audio_tower_initialized: false. The
        # checkpoint disagrees, so BOTH are recorded rather than the frozen
        # artifact being silently contradicted.
        assert phi4_load["audio_tower_present"] is True
        assert phi4_load["audio_tensor_count"] == 887
        block = phi4_report["runtime_metadata"]["phi4_audio_tower"]
        assert block["frozen_protocol_claim"] == \
            "audio_tower_initialized: false"
        assert block["observed"] is True
        assert block["audio_input_supplied"] is False

    def test_every_applied_shim_is_recorded(self, phi4_report):
        joined = " ".join(phi4_report["runtime_metadata"]["phi4_shims"])
        for marker in ("flash_attention_2 -> sdpa", "_tied_weights_keys",
                       "prepare_inputs_for_generation",
                       "Cache.get_usable_length",
                       "vision tower _flash_attention_forward"):
            assert marker in joined, marker

    def test_gpu_slot_is_recorded_from_the_weights_not_the_process(
            self, phi4_report, phi4_load):
        hardware = phi4_report["runtime_metadata"]["hardware"]
        assert hardware["requested_device"] == "cuda:3"
        assert hardware["device_index"] == 3
        assert phi4_load["device"] == "cuda:3"
        assert phi4_load["dtype"] == "torch.bfloat16"

    def test_vision_arm_ran_the_vision_adapter(self, phi4_smoke):
        attempt = phi4_smoke[preflight.VISION_VARIANT]["attempts"][0]
        assert attempt["input_token_count"] == CROSS_MODAL["in"]
        assert attempt["image_token_count"] == CROSS_MODAL["img"]
        diag = attempt["adapter_diagnostics"]
        assert diag["input_mode"] == INPUT_MODE_VISION
        assert diag["image_placeholders_in_rendered_text"] == 1
        assert diag["images_supplied"] == 1
        assert diag["placeholders_match_supplied_images"] is True
        assert diag["image_token_count_after_expansion"] == \
            CROSS_MODAL["img"]
        assert diag["active_lora"]["active_adapters"] == ["vision"]
        assert diag["active_lora"]["adapters_disabled"] is False
        assert diag["audio_special_token_present"] is False
        assert diag["frozen_system_prompt_present_verbatim"] is True

    def test_language_arm_ran_no_adapter(self, phi4_smoke):
        attempt = phi4_smoke[preflight.TEXT_VARIANT]["attempts"][0]
        assert attempt["input_token_count"] == TEXT_ONLY["in"]
        assert attempt["image_token_count"] == TEXT_ONLY["img"]
        assert attempt["ordered_image_hashes"] == []
        diag = attempt["adapter_diagnostics"]
        assert diag["input_mode"] == INPUT_MODE_LANGUAGE
        assert diag["image_placeholders_in_rendered_text"] == 0
        assert diag["image_token_count_after_expansion"] == 0
        assert diag["active_lora"]["adapters_disabled"] is True
        assert diag["frozen_system_prompt_present_verbatim"] is True

    def test_both_arms_stopped_on_eos_not_the_cap(self, phi4_smoke):
        # The decisive evidence for the generation_config fix: had <|end|>
        # (200020) been dropped, both arms would have run to 1536 and
        # reported finish_reason "length".
        for variant, expected in ((preflight.VISION_VARIANT,
                                   CROSS_MODAL["out"]),
                                  (preflight.TEXT_VARIANT, TEXT_ONLY["out"])):
            for attempt in phi4_smoke[variant]["attempts"]:
                assert attempt["finish_reason"] == "eos"
                assert attempt["hit_max_new_tokens"] is False
                assert attempt["output_token_count"] == expected
                assert attempt["output_token_count"] < FROZEN_CAP

    def test_greedy_decoding_is_repeat_stable(self, phi4_report, phi4_smoke):
        assert phi4_report["determinism"]["all_variants_repeat_stable"] is True
        assert phi4_report["determinism"]["greedy_decoding"] is True
        for entry in phi4_smoke.values():
            assert entry["deterministic"] is True
            assert entry["n_distinct_responses"] == 1
            hashes = {a["response_sha256"] for a in entry["attempts"]}
            assert len(hashes) == 1

    def test_the_two_arms_differ_so_the_contrast_is_real(self, phi4_smoke):
        vision = phi4_smoke[preflight.VISION_VARIANT]["attempts"][0]
        text = phi4_smoke[preflight.TEXT_VARIANT]["attempts"][0]
        assert vision["response_sha256"] != text["response_sha256"]
        assert vision["semantic_prompt_hash"] != text["semantic_prompt_hash"]
        assert vision["serialized_prompt_hash"] != \
            text["serialized_prompt_hash"]
        # Same image hash accounting as the other families: the vision arm
        # references exactly one image, the text arm none.
        assert len(vision["ordered_image_hashes"]) == 1

