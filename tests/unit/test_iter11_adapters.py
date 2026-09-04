"""Iteration 11.1: model registry, adapter contract, per-record
provenance and resume — all without downloading a single checkpoint.

The frozen Iteration 8-10 single-model path must be provably untouched,
so the regression anchors here are read from
``outputs/iteration_11/protocol/frozen_9b_reference.json`` (the
immutable Iteration 10 Qwen3.5-9B run) rather than restated inline.

Opt-in GPU integration tests live in ``tests/gpu``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.replay import (
    CallableBackend,
    HFAdapterBase,
    HFLocalBackend,
    Qwen35Adapter,
    ReplayConfig,
    ReplayError,
    ResolvedModel,
    assert_confirmatory_revision,
    build_adapter,
    build_chat_messages,
    fingerprint_hardware,
    is_immutable_revision,
    iteration11_run_fingerprint,
    load_registry,
    resolve_model,
    resolved_fingerprint,
    run_replay_stage,
)
from causal_mllm.replay.adapters.base import (
    ordered_image_hashes, semantic_prompt_hash)
from causal_mllm.replay.registry import DEFAULT_LOCK
from causal_mllm.seeds import sha256_text
from tests.unit.test_grounding import CLEAN_Q, _built_family

PROTOCOL_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "iteration_11" / "protocol")
FROZEN_9B_REF = json.loads(
    (PROTOCOL_DIR / "frozen_9b_reference.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads(
    (PROTOCOL_DIR / "iteration_11_protocol.json").read_text(encoding="utf-8"))

FROZEN_9B_REVISION = FROZEN_9B_REF["resolved_model_revision"]
# Recovered by reproduction: this is the only device string whose
# ReplayConfig fingerprint equals the frozen config_sha256, i.e. the
# device the Iteration 10 Scale-C panel was generated on.
FROZEN_RUN_DEVICE = "cuda:3"

TARGET_KEYS = ("qwen35_2b", "qwen35_4b", "qwen35_9b",
               "ministral3_3b", "phi4_mm")
NEW_TARGET_KEYS = ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm")

LEGACY_RECORD_KEYS = {
    "run_id", "family_id", "source_id", "variant", "model",
    "requested_model_revision", "resolved_model_revision", "revision_pinned",
    "model_revision", "prompt_template_revision", "system_prompt_sha256",
    "generation_config", "terminal_sha256", "n_images", "input_token_count",
    "image_token_count", "output_token_count", "finish_reason",
    "hit_max_new_tokens", "response", "error",
}
ITER11_RECORD_KEYS = {
    "model_key", "model_id", "adapter", "dtype", "quantization", "sample_id",
    "variant_id", "code_commit", "dataset_manifest_hash",
    "resolved_run_fingerprint", "semantic_prompt_hash",
    "serialized_prompt_hash", "ordered_image_hashes", "requested_seed",
    "effective_seed", "deterministic_algorithms", "runtime_versions",
    "hardware", "truncated", "effective_decoding",
}


def _write_validated(tmp_path, *families):
    write_jsonl(tmp_path / "validated_families.jsonl",
                [f.to_dict() for f in families])


def _families(tmp_path, n=2):
    families = [_built_family(tmp_path, CLEAN_Q, fill_grounding=True)
                for _ in range(n)]
    for i, family in enumerate(families):
        family.family_id = f"fam{i:03d}"
    _write_validated(tmp_path, *families)
    return families


def _frozen_9b_config(device: str = FROZEN_RUN_DEVICE) -> ReplayConfig:
    """The Iteration 10 Qwen3.5-9B semantic config, reconstructed."""
    return ReplayConfig(
        model_name=FROZEN_9B_REF["model"],
        model_revision=FROZEN_9B_REVISION,
        max_new_tokens=FROZEN_9B_REF["generation_config"]["max_new_tokens"],
        device=device,
        enable_thinking=FROZEN_9B_REF["enable_thinking"])


class _StubAdapter:
    """Satisfies ReplayBackend + runtime_metadata without touching a GPU.

    ``fail_after=N`` makes every generate() call after the Nth raise, so
    resume behaviour can be tested deterministically: families are
    replayed in order, so the first N calls belong to the leading
    families.
    """

    def __init__(self, model_spec: ResolvedModel, fail_after=None,
                 finish_reason="eos", hardware=None):
        self.model_spec = model_spec
        self.n_calls = 0
        self.fail_after = fail_after
        self.finish_reason = finish_reason
        self._hardware = hardware if hardware is not None else {
            "device_index": 0, "gpu_name": "stub-gpu",
            "total_memory_mb": 1, "compute_capability": "8.9"}

    def generate(self, chat_messages):
        self.n_calls += 1
        if self.fail_after is not None and self.n_calls > self.fail_after:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        n_images = sum(
            1 for m in chat_messages if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image")
        hit = self.finish_reason == "length"
        return {
            "response": f"stub-{self.n_calls}",
            "input_token_count": 10 + n_images,
            "image_token_count": 4 * n_images,
            "output_token_count": 1536 if hit else 12,
            "finish_reason": self.finish_reason,
            "hit_max_new_tokens": hit,
            "semantic_prompt_hash": "a" * 64,
            "serialized_prompt_hash": "b" * 64,
            "ordered_image_hashes": ["c" * 64] * n_images,
            "effective_decoding": {"do_sample": False, "num_beams": 1},
        }

    def runtime_metadata(self):
        return {
            "adapter_name": "stub", "adapter_version": "0.0.0",
            "model_key": self.model_spec.model_key,
            "hardware": self._hardware,
            "deterministic_algorithms": True,
            "requested_seed": 42, "effective_seed": 42,
        }

    def model_name(self):
        return self.model_spec.model_id

    def model_revision(self):
        return FROZEN_9B_REVISION

    def processor_revision(self):
        return FROZEN_9B_REVISION

    def transformers_version(self):
        return "5.14.1"

    def torch_version(self):
        return "2.8.0+cu128"

    def cuda_version(self):
        return "12.8"


def _spec(model_key="qwen35_9b") -> ResolvedModel:
    return resolve_model(model_key)


def _run(tmp_path, backend, model_spec=None, **kwargs):
    return run_replay_stage(
        tmp_path, tmp_path / "runs", backend=backend, run_id="iter11-run",
        model_spec=model_spec, **kwargs)


# ---------------------------------------------------------------------
# Revision policy (fail closed)
# ---------------------------------------------------------------------
class TestImmutableRevisionPolicy:
    @pytest.mark.parametrize("rev", [
        FROZEN_9B_REVISION,
        "0" * 40,
        "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
    ])
    def test_accepts_full_hex_sha(self, rev):
        assert is_immutable_revision(rev) is True

    @pytest.mark.parametrize("rev", [
        None, "", "main", "MAIN", "master", "head", "latest", "none",
        "0" * 39, "0" * 41, "z" * 40, "v1.0", True, False, 12345, 0,
    ])
    def test_rejects_floating_and_malformed(self, rev):
        assert is_immutable_revision(rev) is False

    def test_bool_is_not_a_revision(self):
        # True == 1 in Python; a revision must never be a bool.
        assert is_immutable_revision(True) is False

    def test_assert_confirmatory_accepts_pinned(self):
        assert_confirmatory_revision("qwen35_9b", FROZEN_9B_REVISION)

    @pytest.mark.parametrize("rev", [None, "main", "0" * 39])
    def test_assert_confirmatory_rejects_floating(self, rev):
        with pytest.raises(ReplayError, match="immutable 40-hex"):
            assert_confirmatory_revision("qwen35_2b", rev)


# ---------------------------------------------------------------------
# Registry contents + resolution
# ---------------------------------------------------------------------
class TestRegistry:
    def test_registry_has_exactly_the_frozen_matrix(self):
        assert set(load_registry()["models"]) == set(TARGET_KEYS)

    def test_adapter_mapping(self):
        expected = {"qwen35_2b": "qwen35", "qwen35_4b": "qwen35",
                    "qwen35_9b": "qwen35", "ministral3_3b": "ministral3",
                    "phi4_mm": "phi4_multimodal"}
        for key, adapter in expected.items():
            assert resolve_model(key).adapter == adapter, key

    def test_every_target_is_full_precision_bf16_thinking_off(self):
        for key in TARGET_KEYS:
            spec = resolve_model(key)
            assert spec.dtype == "bfloat16", key
            assert spec.quantization == "none", key
            assert spec.thinking_mode is False, key

    def test_9b_is_pinned_to_the_frozen_iteration_10_revision(self):
        spec = resolve_model("qwen35_9b")
        assert spec.revision == FROZEN_9B_REVISION
        assert spec.revision == FROZEN_9B_REF["resolved_model_revision"]
        assert is_immutable_revision(spec.revision)
        assert spec.revision_source == "registry"

    def test_new_targets_are_unpinned_in_the_frozen_registry(self):
        # The FROZEN registry itself still carries null revisions for the
        # new targets; the immutable values resolved at preflight live in
        # the lock file, which resolve_model() merges in.
        raw = yaml.safe_load(
            (PROTOCOL_DIR / "model_registry.yaml").read_text(
                encoding="utf-8"))["models"]
        for key in NEW_TARGET_KEYS:
            assert raw[key]["revision"] is None, key
        assert raw["qwen35_9b"]["revision"] == FROZEN_9B_REVISION

    def test_only_phi4_needs_remote_code(self):
        for key in TARGET_KEYS:
            expected = key == "phi4_mm"
            assert resolve_model(key).trust_remote_code is expected, key

    def test_confirmatory_rejects_a_target_with_no_locked_revision(
            self, tmp_path):
        # Isolated from the committed preflight lock: with an empty lock
        # every new target is still unpinned and must be rejected.
        empty = tmp_path / "empty-lock.yaml"
        empty.write_text("models: {}\n", encoding="utf-8")
        for key in NEW_TARGET_KEYS:
            with pytest.raises(ReplayError, match="immutable 40-hex"):
                resolve_model(key, confirmatory=True, lock_path=empty)

    def test_committed_preflight_lock_enables_confirmatory_resolution(self):
        if not DEFAULT_LOCK.exists():
            pytest.skip("no preflight lock committed yet")
        locked = yaml.safe_load(
            DEFAULT_LOCK.read_text(encoding="utf-8")).get("models", {})
        assert locked, "lock file has no models"
        for key in locked:
            spec = resolve_model(key, confirmatory=True)
            assert is_immutable_revision(spec.revision), key
            assert spec.revision_source == "lock", key
            assert spec.revision == locked[key]["revision"], key

    def test_confirmatory_accepts_pinned_9b(self):
        assert resolve_model("qwen35_9b",
                             confirmatory=True).revision == \
            FROZEN_9B_REVISION

    def test_lock_supplies_the_confirmatory_revision(self, tmp_path):
        locked = "ab" * 20
        lock = tmp_path / "resolved_models.lock.yaml"
        lock.write_text(yaml.safe_dump(
            {"models": {"qwen35_2b": {"revision": locked}}}),
            encoding="utf-8")
        spec = resolve_model("qwen35_2b", confirmatory=True,
                             lock_path=lock)
        assert spec.revision == locked
        assert spec.revision_source == "lock"

    def test_lock_cannot_rescue_a_floating_revision(self, tmp_path):
        lock = tmp_path / "lock.yaml"
        lock.write_text(yaml.safe_dump(
            {"models": {"qwen35_2b": {"revision": "main"}}}),
            encoding="utf-8")
        with pytest.raises(ReplayError, match="immutable 40-hex"):
            resolve_model("qwen35_2b", confirmatory=True, lock_path=lock)

    def test_floating_non_null_revision_rejected_even_in_preflight(self):
        reg = {"models": {"m": {"model_id": "org/name",
                                "revision": "main", "adapter": "qwen35"}}}
        with pytest.raises(ReplayError, match="floating"):
            resolve_model("m", registry=reg)

    def test_null_revision_allowed_in_preflight(self):
        reg = {"models": {"m": {"model_id": "org/name",
                                "revision": None, "adapter": "qwen35"}}}
        assert resolve_model("m", registry=reg).revision is None

    def test_unknown_model_key_rejected(self):
        with pytest.raises(ReplayError, match="unknown model_key"):
            resolve_model("qwen35_70b")

    def test_unknown_adapter_rejected(self):
        reg = {"models": {"m": {"model_id": "org/name", "revision": None,
                                "adapter": "bogus"}}}
        with pytest.raises(ReplayError, match="not in"):
            resolve_model("m", registry=reg)

    def test_size_metadata_is_declared_not_inferred(self):
        # Ministral-3 carries declared language/vision parameter counts;
        # Qwen sizes stay null until read from the checkpoint config at
        # preflight (never inferred from a response).
        reg = load_registry()["models"]
        assert reg["ministral3_3b"]["size_metadata"][
            "language_parameters"] == 3400000000
        assert isinstance(reg["ministral3_3b"]["size_metadata"][
            "language_parameters"], int)
        assert reg["qwen35_2b"]["size_metadata"][
            "language_parameters"] is None

    def test_excluded_models_are_not_qwen_backbones(self):
        excluded = {e["model_id"]: e["reason"]
                    for e in load_registry()["excluded_models"]}
        assert "InternVL3.5-4B" in excluded and "Molmo2-4B" in excluded
        for reason in excluded.values():
            assert "Qwen3" in reason

    def test_fallback_is_technical_gate_only(self):
        fb = load_registry()["fallback"]
        assert fb["model_key"] == "gemma3_4b"
        assert fb["replaces"] == "phi4_mm"
        assert fb["allowed_reason"] == "technical_eligibility_gate_only"
        assert "NEVER for low effect size" in fb["rule"]


# ---------------------------------------------------------------------
# Adapter contract (no weights loaded)
# ---------------------------------------------------------------------
class TestAdapterContract:
    def test_build_adapter_dispatches_qwen(self):
        adapter = build_adapter(_spec(), ReplayConfig())
        assert isinstance(adapter, Qwen35Adapter)
        assert isinstance(adapter, HFAdapterBase)
        assert adapter.adapter_name == "qwen35"

    def test_build_adapter_rejects_unknown_kind(self):
        spec = ResolvedModel(model_key="x", model_id="org/x",
                             adapter="bogus", revision=None)
        with pytest.raises(ReplayError, match="no adapter registered"):
            build_adapter(spec, ReplayConfig())

    def test_thinking_disabled_from_frozen_registry(self):
        adapter = build_adapter(_spec(), _frozen_9b_config())
        assert adapter.chat_template_kwargs() == {"enable_thinking": False}

    def test_pretrained_kwargs_pin_the_revision(self):
        adapter = build_adapter(_spec(), _frozen_9b_config())
        assert adapter._pretrained_kwargs() == {
            "revision": FROZEN_9B_REVISION}

    def test_unpinned_spec_passes_no_revision(self):
        # Hand-built rather than resolved, so the assertion stays valid
        # once preflight locks the real targets.
        spec = ResolvedModel(model_key="unpinned", model_id="org/x",
                             adapter="qwen35", revision=None)
        adapter = build_adapter(spec, ReplayConfig())
        assert adapter._pretrained_kwargs() == {}

    def test_trust_remote_code_forwarded(self):
        spec = ResolvedModel(model_key="phi4_mm",
                             model_id="microsoft/Phi-4-multimodal-instruct",
                             adapter="phi4_multimodal", revision=None,
                             trust_remote_code=True)
        adapter = Qwen35Adapter(ReplayConfig(), spec)
        assert adapter._pretrained_kwargs() == {"trust_remote_code": True}

    def test_quantized_checkpoint_fails_closed(self):
        spec = ResolvedModel(model_key="q", model_id="org/q",
                             adapter="qwen35", revision=None,
                             quantization="int4")
        with pytest.raises(ReplayError, match="out of scope"):
            build_adapter(spec, ReplayConfig())

    def test_generate_before_load_raises(self):
        adapter = build_adapter(_spec(), ReplayConfig())
        with pytest.raises(RuntimeError, match="load\\(\\) was not called"):
            adapter.generate([])

    def test_model_name_comes_from_the_spec(self):
        adapter = build_adapter(_spec(), ReplayConfig())
        assert adapter.model_name() == FROZEN_9B_REF["model"]

    def test_effective_decoding_is_greedy(self):
        adapter = build_adapter(_spec(), _frozen_9b_config())
        assert adapter.effective_decoding() == \
            PROTOCOL["frozen_inputs"]["effective_decoding"]


class TestPromptHashes:
    def test_semantic_hash_is_stable_and_hex(self, tmp_path):
        family = _families(tmp_path, 1)[0]
        config = _frozen_9b_config()
        chat = build_chat_messages(family, "cross_modal", config)
        digest = semantic_prompt_hash(chat)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        assert semantic_prompt_hash(chat) == digest

    def test_semantic_hash_is_model_independent(self, tmp_path):
        # The same semantic messages hash identically under two different
        # family adapters: the hash must not depend on a tokenizer.
        family = _families(tmp_path, 1)[0]
        config = _frozen_9b_config()
        chat = build_chat_messages(family, "cross_modal", config)
        qwen = build_adapter(_spec("qwen35_9b"), config)
        other = build_adapter(_spec("qwen35_2b"), config)
        assert semantic_prompt_hash(chat) == semantic_prompt_hash(chat)
        assert qwen.adapter_name == other.adapter_name == "qwen35"

    def test_semantic_hash_separates_text_from_image_bearing(self, tmp_path):
        family = _families(tmp_path, 1)[0]
        config = _frozen_9b_config()
        text_only = semantic_prompt_hash(
            build_chat_messages(family, "text_only", config))
        cross = semantic_prompt_hash(
            build_chat_messages(family, "cross_modal", config))
        assert text_only != cross

    def test_ordered_image_hashes_match_the_media_bytes(self, tmp_path):
        family = _families(tmp_path, 1)[0]
        config = _frozen_9b_config()
        chat = build_chat_messages(family, "cross_modal", config)
        hashes = ordered_image_hashes(chat)
        assert len(hashes) == 1
        expected_path = next(
            media["path"] for atom in family.semantic_atoms
            for media in atom.source_media)
        assert hashes[0] == hashlib.sha256(
            Path(expected_path).read_bytes()).hexdigest()

    def test_text_only_variant_has_no_image_hashes(self, tmp_path):
        family = _families(tmp_path, 1)[0]
        chat = build_chat_messages(family, "text_only", _frozen_9b_config())
        assert ordered_image_hashes(chat) == []


# ---------------------------------------------------------------------
# Runner: Iteration 11 per-record provenance
# ---------------------------------------------------------------------
class TestRunnerIteration11Provenance:
    def _outputs(self, tmp_path, **kwargs):
        return read_jsonl(
            tmp_path / "runs" / "iter11-run" / "replay_outputs.jsonl")

    def test_records_carry_the_full_iteration_11_schema(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        report = _run(tmp_path, stub, spec)
        outputs = self._outputs(tmp_path)
        assert len(outputs) == 6 and report["n_failed"] == 0
        for record in outputs:
            missing = ITER11_RECORD_KEYS - set(record)
            assert not missing, missing
            assert LEGACY_RECORD_KEYS <= set(record)

    def test_run_level_provenance_is_identical_across_records(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        _run(tmp_path, stub, spec)
        outputs = self._outputs(tmp_path)
        for key in ("model_key", "model_id", "adapter", "dtype",
                    "quantization", "code_commit", "dataset_manifest_hash",
                    "resolved_run_fingerprint", "runtime_versions",
                    "hardware", "requested_seed", "effective_seed",
                    "deterministic_algorithms"):
            assert len({json.dumps(r[key], sort_keys=True)
                        for r in outputs}) == 1, key
        assert outputs[0]["model_key"] == "qwen35_9b"
        assert outputs[0]["adapter"] == "qwen35"
        assert outputs[0]["runtime_versions"] == {
            "transformers": "5.14.1", "torch": "2.8.0+cu128",
            "cuda": "12.8"}

    def test_resolved_run_fingerprint_is_recomputable(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=stub, run_id="iter11-run", model_spec=spec)
        expected = iteration11_run_fingerprint(
            stub, config, tmp_path, spec,
            stub.runtime_metadata()["hardware"])
        outputs = self._outputs(tmp_path)
        assert {r["resolved_run_fingerprint"] for r in outputs} == {expected}

    def test_iteration_11_fingerprint_separates_model_key_and_adapter(
            self, tmp_path):
        # Three specs over the SAME weights (identical model_id and
        # revision) differing only in registry key / adapter: the legacy
        # fingerprint structurally cannot separate them, the Iteration 11
        # fingerprint must.
        _families(tmp_path, 1)
        config = ReplayConfig(max_new_tokens=1536)
        specs = [
            ResolvedModel(model_key="k1", model_id="org/same",
                          adapter="qwen35", revision=FROZEN_9B_REVISION),
            ResolvedModel(model_key="k2", model_id="org/same",
                          adapter="qwen35", revision=FROZEN_9B_REVISION),
            ResolvedModel(model_key="k1", model_id="org/same",
                          adapter="ministral3",
                          revision=FROZEN_9B_REVISION),
        ]
        stubs = [_StubAdapter(spec) for spec in specs]
        hw = stubs[0].runtime_metadata()["hardware"]
        legacy = {resolved_fingerprint(stub, config, tmp_path)
                  for stub in stubs}
        assert len(legacy) == 1, "legacy fingerprint became model-aware"
        iter11 = {
            iteration11_run_fingerprint(stub, config, tmp_path, spec, hw)
            for stub, spec in zip(stubs, specs)}
        assert len(iter11) == 3

    def test_legacy_fingerprint_invariant_under_the_model_dimension(
            self, tmp_path):
        # Running the SAME backend with and without a model_spec must not
        # perturb the frozen resolved_sha256; Iteration 11 only ADDS a
        # separate resolved_run_fingerprint.
        _families(tmp_path, 1)
        config = ReplayConfig(max_new_tokens=1536)
        spec = _spec()
        stub = _StubAdapter(spec)
        legacy = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=stub,
            run_id="legacy-run")
        iter11 = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=stub,
            run_id="iter11-dim", model_spec=spec)
        assert legacy["provenance"]["resolved_sha256"] == \
            iter11["provenance"]["resolved_sha256"]
        assert "resolved_run_fingerprint" not in legacy["provenance"]
        assert iter11["provenance"]["resolved_run_fingerprint"] != \
            iter11["provenance"]["resolved_sha256"]

    def test_dataset_manifest_hash_binds_the_validated_panel(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        _run(tmp_path, _StubAdapter(spec), spec)
        expected = hashlib.sha256(
            (tmp_path / "validated_families.jsonl").read_bytes()).hexdigest()
        assert self._outputs(tmp_path)[0]["dataset_manifest_hash"] == expected

    def test_sample_and_variant_identifiers_present(self, tmp_path):
        families = _families(tmp_path, 1)
        spec = _spec()
        _run(tmp_path, _StubAdapter(spec), spec)
        outputs = self._outputs(tmp_path)
        assert {r["variant_id"] for r in outputs} == {r["variant"]
                                                      for r in outputs}
        assert {r["sample_id"] for r in outputs} == {
            families[0].source.get("source_id")}
        assert len({r["family_id"] for r in outputs}) == 1

    def test_truncated_flag_follows_finish_reason(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        _run(tmp_path, _StubAdapter(spec, finish_reason="length"), spec)
        outputs = self._outputs(tmp_path)
        assert all(r["truncated"] is True for r in outputs)
        assert all(r["hit_max_new_tokens"] is True for r in outputs)

    def test_report_gains_the_iteration_11_dimension(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        report = _run(tmp_path, _StubAdapter(spec), spec)
        assert report["iteration"] == "11"
        assert report["model_key"] == "qwen35_9b"
        assert report["adapter"] == "qwen35"
        assert report["model_spec"]["model_id"] == FROZEN_9B_REF["model"]
        assert report["resume"] == {"enabled": False, "n_pairs_resumed": 0}
        prov = report["provenance"]
        assert prov["resolved_run_fingerprint"]
        assert prov["model_key"] == "qwen35_9b"
        assert prov["hardware"]["gpu_name"] == "stub-gpu"
        assert prov["deterministic_algorithms"] is True
        # Legacy provenance is still emitted alongside.
        assert prov["resolved_sha256"] and prov["config_sha256"]

    def test_failure_records_keep_the_iteration_11_schema(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        report = _run(tmp_path, _StubAdapter(spec, fail_after=0), spec)
        failures = read_jsonl(
            tmp_path / "runs" / "iter11-run" / "replay_failures.jsonl")
        assert report["n_failed"] == 6 and len(failures) == 6
        for record in failures:
            assert record["model_key"] == "qwen35_9b"
            assert record["resolved_run_fingerprint"]
            assert record["error"]["category"] == "oom"
            assert record["response"] is None


# ---------------------------------------------------------------------
# Hardware identity: the scheduling slot is not a scientific variable
# ---------------------------------------------------------------------
class TestHardwareFingerprinting:
    def test_scheduling_slot_keys_are_excluded(self):
        hardware = {"device_index": 3, "requested_device": "cuda:3",
                    "gpu_name": "NVIDIA RTX 6000 Ada",
                    "compute_capability": "8.9", "total_memory_mb": 48508}
        bound = fingerprint_hardware(hardware)
        assert "device_index" not in bound
        assert "requested_device" not in bound
        assert bound == {"compute_capability": "8.9",
                         "gpu_name": "NVIDIA RTX 6000 Ada",
                         "total_memory_mb": 48508}

    @pytest.mark.parametrize("value", [None, "cuda:3", 3, []])
    def test_non_dict_hardware_is_not_bound(self, value):
        assert fingerprint_hardware(value) is None

    def test_fingerprint_invariant_to_the_gpu_slot(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(max_new_tokens=1536)
        stub = _StubAdapter(spec)
        base = stub.runtime_metadata()["hardware"]
        slots = [dict(base, device_index=i) for i in (0, 1, 3)]
        fingerprints = {
            iteration11_run_fingerprint(stub, config, tmp_path, spec, hw)
            for hw in slots}
        assert len(fingerprints) == 1

    def test_fingerprint_sensitive_to_the_hardware_class(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(max_new_tokens=1536)
        stub = _StubAdapter(spec)
        base = stub.runtime_metadata()["hardware"]
        ada = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, base)
        other = iteration11_run_fingerprint(
            stub, config, tmp_path, spec,
            dict(base, gpu_name="NVIDIA A100-SXM4-80GB"))
        assert ada != other

    def test_fingerprint_invariant_to_config_device(self, tmp_path):
        # config.fingerprint() serializes `device`, so it must NOT feed
        # the Iteration 11 run fingerprint.
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        hw = stub.runtime_metadata()["hardware"]
        fingerprints = {
            iteration11_run_fingerprint(
                stub, ReplayConfig(max_new_tokens=1536, device=device),
                tmp_path, spec, hw)
            for device in ("cuda:0", "cuda:1", "cuda:3")}
        assert len(fingerprints) == 1

    def test_resume_survives_a_gpu_slot_change(self, tmp_path):
        _families(tmp_path, 2)
        spec = _spec()
        first = ReplayConfig(model_name=spec.model_id, device="cuda:0")
        run_replay_stage(tmp_path, tmp_path / "runs", config=first,
                         backend=_StubAdapter(spec, fail_after=6),
                         run_id="iter11-run", model_spec=spec)
        # Interrupted, then resumed on a DIFFERENT slot of the same GPU
        # class: this must continue rather than report a mismatch.
        resumed = ReplayConfig(model_name=spec.model_id, device="cuda:3")
        stub = _StubAdapter(spec)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=resumed, backend=stub,
            run_id="iter11-run", model_spec=spec, resume=True)
        assert stub.n_calls == 6
        assert report["n_succeeded"] == 12 and report["n_failed"] == 0


class _FakeDevice:
    def __init__(self, type_, index):
        self.type = type_
        self.index = index


class _FakeParam:
    def __init__(self, type_, index):
        self.device = _FakeDevice(type_, index)


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        # The trap: loading with device_map="cuda:3" leaves the CURRENT
        # device at 0.
        return 0


class _FakeTorch:
    cuda = _FakeCuda


class TestActiveDeviceIndex:
    """Regression: the recorded hardware must be the slot the WEIGHTS
    occupy, not ``torch.cuda.current_device()``."""

    def test_reads_the_device_of_the_loaded_weights(self):
        adapter = build_adapter(_spec(), ReplayConfig(device="cuda:3"))
        adapter.model = _FakeModel([_FakeParam("cuda", 3)])
        assert adapter._active_device_index(_FakeTorch) == 3

    def test_ignores_a_misleading_current_device(self):
        adapter = build_adapter(_spec(), ReplayConfig(device="cuda:3"))
        adapter.model = _FakeModel([_FakeParam("cuda", 3)])
        assert _FakeTorch.cuda.current_device() == 0
        assert adapter._active_device_index(_FakeTorch) != 0

    def test_falls_back_to_the_configured_device_string(self):
        adapter = build_adapter(_spec(), ReplayConfig(device="cuda:3"))
        adapter.model = _FakeModel([])
        assert adapter._active_device_index(_FakeTorch) == 3

    def test_cpu_weights_fall_back_to_the_configured_device(self):
        adapter = build_adapter(_spec(), ReplayConfig(device="cuda:3"))
        adapter.model = _FakeModel([_FakeParam("cpu", None)])
        assert adapter._active_device_index(_FakeTorch) == 3

    def test_bare_cuda_device_uses_the_current_device(self):
        adapter = build_adapter(_spec(), ReplayConfig(device="cuda"))
        adapter.model = _FakeModel([])
        assert adapter._active_device_index(_FakeTorch) == 0


# ---------------------------------------------------------------------
# Runner: legacy single-model path must be untouched
# ---------------------------------------------------------------------
class TestLegacyPathUnchanged:
    def test_legacy_record_schema_is_exact(self, tmp_path):
        _families(tmp_path, 1)
        run_replay_stage(tmp_path, tmp_path / "runs",
                         backend=CallableBackend(lambda c: "ok"),
                         run_id="legacy")
        outputs = read_jsonl(tmp_path / "runs" / "legacy" /
                             "replay_outputs.jsonl")
        assert len(outputs) == 6
        for record in outputs:
            assert set(record) == LEGACY_RECORD_KEYS, (
                set(record) ^ LEGACY_RECORD_KEYS)

    def test_legacy_report_has_no_iteration_11_dimension(self, tmp_path):
        _families(tmp_path, 1)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs",
            backend=CallableBackend(lambda c: "ok"), run_id="legacy")
        assert report["iteration"] == "8"
        assert "model_key" not in report
        assert "adapter" not in report
        assert "model_spec" not in report
        assert "resume" not in report
        assert "resolved_run_fingerprint" not in report["provenance"]

    def test_legacy_generation_config_unchanged(self, tmp_path):
        _families(tmp_path, 1)
        run_replay_stage(tmp_path, tmp_path / "runs",
                         backend=CallableBackend(lambda c: "ok"),
                         run_id="legacy")
        for record in read_jsonl(tmp_path / "runs" / "legacy" /
                                 "replay_outputs.jsonl"):
            assert record["generation_config"] == {
                "temperature": 0.0, "top_p": 1.0, "do_sample": False,
                "max_new_tokens": 256, "seed": 42}


# ---------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------
class TestResume:
    def _outputs(self, tmp_path):
        return read_jsonl(
            tmp_path / "runs" / "iter11-run" / "replay_outputs.jsonl")

    def test_resume_completes_only_the_missing_pairs(self, tmp_path):
        _families(tmp_path, 2)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        # First attempt: fam000's six variants succeed, fam001 OOMs.
        first = _StubAdapter(spec, fail_after=6)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=first,
            run_id="iter11-run", model_spec=spec)
        assert report["n_succeeded"] == 6 or report["n_failed"] == 6
        assert len(self._outputs(tmp_path)) == 6

        # Resume: fam000 must be skipped entirely, fam001 regenerated.
        second = _StubAdapter(spec)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=second,
            run_id="iter11-run", model_spec=spec, resume=True)
        assert second.n_calls == 6, "completed family was regenerated"
        assert report["n_succeeded"] == 12 and report["n_failed"] == 0
        assert report["missing_variants"] == []
        assert report["resume"] == {"enabled": True, "n_pairs_resumed": 6}
        outputs = self._outputs(tmp_path)
        assert len(outputs) == 12
        pairs = {(r["family_id"], r["variant"]) for r in outputs}
        assert len(pairs) == 12

    def test_resume_of_a_complete_run_regenerates_nothing(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        again = _StubAdapter(spec)
        # No overwrite flag: resume must bypass the evidence guard.
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=again,
            run_id="iter11-run", model_spec=spec, resume=True)
        assert again.n_calls == 0
        assert report["n_succeeded"] == 6
        assert report["resume"]["n_pairs_resumed"] == 6

    def test_resume_retries_failed_variants(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec, fail_after=0),
                         run_id="iter11-run", model_spec=spec)
        assert self._outputs(tmp_path) == []
        retried = _StubAdapter(spec)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=retried,
            run_id="iter11-run", model_spec=spec, resume=True)
        assert retried.n_calls == 6
        assert report["n_succeeded"] == 6 and report["n_failed"] == 0

    def _tamper(self, tmp_path, mutate):
        path = tmp_path / "runs" / "iter11-run" / "replay_outputs.jsonl"
        records = read_jsonl(path)
        mutate(records)
        write_jsonl(path, records)

    def test_resume_rejects_a_different_run_fingerprint(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        self._tamper(tmp_path, lambda recs: recs[0].__setitem__(
            "resolved_run_fingerprint", "f" * 64))
        with pytest.raises(ReplayError, match="resume mismatch"):
            run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                             backend=_StubAdapter(spec), run_id="iter11-run",
                             model_spec=spec, resume=True)

    def test_resume_rejects_a_different_model_key(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        self._tamper(tmp_path, lambda recs: recs[0].__setitem__(
            "model_key", "qwen35_2b"))
        with pytest.raises(ReplayError, match="resume mismatch"):
            run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                             backend=_StubAdapter(spec), run_id="iter11-run",
                             model_spec=spec, resume=True)

    def test_resume_rejects_duplicate_stored_records(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        self._tamper(tmp_path, lambda recs: recs.append(dict(recs[0])))
        with pytest.raises(ReplayError, match="duplicate stored record"):
            run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                             backend=_StubAdapter(spec), run_id="iter11-run",
                             model_spec=spec, resume=True)


# ---------------------------------------------------------------------
# Qwen3.5-9B regression: the frozen semantic config is reproducible
# ---------------------------------------------------------------------
class TestQwen9BFrozenConfigRegression:
    def test_config_fingerprint_reproduces_the_frozen_run(self):
        # Strongest available anchor: ReplayConfig serialization is
        # untouched, so the frozen Iteration 10 config_sha256 is
        # reproduced bit-for-bit from the same semantic settings.
        assert _frozen_9b_config().fingerprint() == \
            FROZEN_9B_REF["config_sha256"]

    def test_system_prompt_sha_matches_the_frozen_run(self):
        assert sha256_text(_frozen_9b_config().system_prompt) == \
            FROZEN_9B_REF["system_prompt_sha256"]
        assert sha256_text(ReplayConfig.system_prompt) == \
            PROTOCOL["frozen_inputs"]["system_prompt_sha256"]

    def test_generation_config_matches_the_frozen_run(self):
        assert _frozen_9b_config().generation_settings() == \
            FROZEN_9B_REF["generation_config"]
        assert _frozen_9b_config().generation_settings() == \
            PROTOCOL["frozen_inputs"][
                "iteration_10_generation_config_verbatim"]

    def test_effective_decoding_matches_the_frozen_protocol(self):
        adapter = build_adapter(_spec(), _frozen_9b_config())
        assert adapter.effective_decoding() == \
            PROTOCOL["frozen_inputs"]["effective_decoding"]
        assert adapter.effective_decoding()["do_sample"] is False
        assert adapter.effective_decoding()["max_new_tokens"] == 1536

    def test_adapter_and_frozen_backend_resolve_the_same_target(self):
        config = _frozen_9b_config()
        adapter = build_adapter(_spec(), config)
        frozen = HFLocalBackend(config)
        assert adapter.model_name() == frozen.model_name() == \
            FROZEN_9B_REF["model"]
        assert adapter.model_revision() == frozen.model_revision() == \
            FROZEN_9B_REVISION
        assert adapter._pretrained_kwargs() == frozen._pretrained_kwargs()

    def test_dtype_and_thinking_match_the_frozen_run(self):
        spec = _spec()
        config = _frozen_9b_config()
        assert spec.dtype == FROZEN_9B_REF["torch_dtype"] == "bfloat16"
        assert config.torch_dtype == FROZEN_9B_REF["torch_dtype"]
        assert build_adapter(spec, config).chat_template_kwargs() == {
            "enable_thinking": FROZEN_9B_REF["enable_thinking"]}
        assert config.enable_thinking is False

    def test_prompt_template_revision_matches(self):
        assert _frozen_9b_config().prompt_template_revision == \
            FROZEN_9B_REF["prompt_template_revision"] == "v1"

    def test_9b_reference_is_never_regenerated_by_iteration_11(self):
        # The 9B arm is a REFERENCE to immutable Iteration 10 evidence.
        ref_run = (Path(__file__).resolve().parents[2]
                   / FROZEN_9B_REF["run_dir"])
        assert ref_run.exists()
        report = json.loads(
            (ref_run / "replay_report.json").read_text(encoding="utf-8"))
        assert report["provenance"]["config_sha256"] == \
            FROZEN_9B_REF["config_sha256"]
        assert report["provenance"]["resolved_sha256"] == \
            FROZEN_9B_REF["resolved_sha256"]
        assert report["n_succeeded"] == FROZEN_9B_REF["n_succeeded"] == 600
