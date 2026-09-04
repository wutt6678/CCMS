"""Iteration 11.2: declared checkpoint size metadata.

These tests are CI-safe: they parse SYNTHETIC safetensors headers
(8-byte little-endian length + JSON), so no checkpoint is downloaded and
no GPU is touched. The real measured values for the cached Qwen3.5
targets are pinned from the committed preflight artifacts.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from causal_mllm.replay.checkpoint_size import (
    _safetensors_header,
    _shard_files,
    checkpoint_size_metadata,
    classify_tensor_key,
    resolve_snapshot_dir,
)
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import is_immutable_revision, resolve_model

PREFLIGHT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "iteration_11" / "preflight")

# Measured from the checkpoints during 11.2 preflight (header shapes).
PINNED_SIZES = {
    "qwen35_2b": {
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "total": 2274069824,
        "language": 1881825088,
        "vision": 331416576,
        "auxiliary": 60828160,
    },
    "qwen35_4b": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "total": 4659865088,
        "language": 4205751296,
        "vision": 333514240,
        "auxiliary": 120599552,
    },
}


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a header-only stand-in for a safetensors shard."""
    header = {
        key: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        for key, (dtype, shape) in tensors.items()
    }
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _synthetic_snapshot(tmp_path, *, indexed=True, n_shards=1):
    """A fake checkpoint: language + vision + auxiliary tensors."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    specs = [
        ("model.language_model.layers.0.self_attn.q_proj.weight",
         "BF16", [8, 4]),          # 32
        ("model.language_model.embed_tokens.weight", "BF16", [10, 2]),  # 20
        ("model.visual.blocks.0.attn.qkv.weight", "BF16", [3, 3]),     # 9
        ("model.visual.merger.mlp.0.weight", "F32", [2, 2]),           # 4
        ("mtp.layers.0.fc.weight", "BF16", [5, 1]),                    # 5
    ]
    for index, (key, dtype, shape) in enumerate(specs):
        shard = f"model-0000{index % n_shards + 1}-of-0000{n_shards}" \
                ".safetensors"
        path = snapshot / shard
        existing = {}
        if path.exists():
            existing = {
                k: (v["dtype"], v["shape"])
                for k, v in _safetensors_header(path).items()}
        existing[key] = (dtype, shape)
        _write_safetensors(path, existing)
        weight_map[key] = shard
    if indexed:
        # Real byte size of the synthetic tensors: BF16 (32+20+9+5=66
        # params x 2 bytes = 132) + F32 (4 params x 4 bytes = 16) = 148.
        # Note 148 // 2 = 74 != 70 parameters: the mixed dtype population
        # is exactly why the byte-size shortcut cannot be used.
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 148},
                        "weight_map": weight_map}), encoding="utf-8")
    return snapshot


class TestTensorKeyClassification:
    @pytest.mark.parametrize("key", [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.31.self_attn.o_proj.weight",
        "model.norm.weight",
    ])
    def test_language_keys(self, key):
        assert classify_tensor_key(key) == "language"

    @pytest.mark.parametrize("key", [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_tower.blocks.2.mlp.fc1.weight",
        "model.multi_modal_projector.linear.weight",
    ])
    def test_vision_keys(self, key):
        assert classify_tensor_key(key) == "vision"

    @pytest.mark.parametrize("key", [
        "mtp.layers.0.fc.weight",
        "mtp.norm.weight",
        "model.multi_token_prediction.head.weight",
    ])
    def test_auxiliary_heads_are_not_folded_into_language(self, key):
        assert classify_tensor_key(key) == "auxiliary"

    def test_unrecognised_keys_are_visible_not_silently_language(self):
        assert classify_tensor_key("something.entirely.different") == "other"


class TestSafetensorsHeaderParsing:
    def test_header_round_trip(self, tmp_path):
        path = tmp_path / "shard.safetensors"
        _write_safetensors(path, {"a.weight": ("BF16", [2, 3])})
        header = _safetensors_header(path)
        assert header["a.weight"]["shape"] == [2, 3]
        assert header["a.weight"]["dtype"] == "BF16"

    def test_truncated_header_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 500) + b"{}")
        with pytest.raises(ReplayError, match="truncated safetensors header"):
            _safetensors_header(path)

    def test_unparseable_header_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        blob = b"not json at all"
        path.write_bytes(struct.pack("<Q", len(blob)) + blob)
        with pytest.raises(ReplayError, match="unreadable safetensors"):
            _safetensors_header(path)

    def test_too_short_for_a_length_prefix(self, tmp_path):
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x01\x02")
        with pytest.raises(ReplayError, match="truncated safetensors header"):
            _safetensors_header(path)


class TestShardDiscovery:
    def test_index_weight_map_is_deduped_and_sorted(self, tmp_path):
        snapshot = _synthetic_snapshot(tmp_path, n_shards=2)
        shards = _shard_files(snapshot)
        assert [p.name for p in shards] == sorted(
            {p.name for p in shards})
        assert len(shards) == 2

    def test_glob_fallback_without_an_index(self, tmp_path):
        snapshot = _synthetic_snapshot(tmp_path, indexed=False)
        (snapshot / "model.safetensors.index.json").unlink(missing_ok=True)
        shards = _shard_files(snapshot)
        assert len(shards) == 1

    def test_no_shards_fails_loudly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ReplayError, match="no safetensors shards"):
            _shard_files(empty)

    def test_empty_weight_map_fails_loudly(self, tmp_path):
        snapshot = tmp_path / "s"
        snapshot.mkdir()
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}), encoding="utf-8")
        with pytest.raises(ReplayError, match="empty weight_map"):
            _shard_files(snapshot)


class TestCheckpointSizeMetadata:
    def test_counts_are_exact_and_component_split(self, tmp_path, monkeypatch):
        snapshot = _synthetic_snapshot(tmp_path)
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        meta = checkpoint_size_metadata("org/model")
        # language 32 + 20, vision 9 + 4, auxiliary 5
        assert meta["checkpoint_parameter_count"] == 70
        assert meta["language_parameters"] == 52
        assert meta["vision_parameters"] == 13
        assert meta["auxiliary_parameters"] == 5
        assert meta["unclassified_parameters"] == 0
        assert meta["unclassified_prefixes"] == {}
        assert meta["n_tensors"] == 5
        assert meta["stored_dtype_histogram"] == {"BF16": 4, "F32": 1}

    def test_provenance_of_the_number_is_declared(self, tmp_path, monkeypatch):
        snapshot = _synthetic_snapshot(tmp_path)
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        meta = checkpoint_size_metadata("org/model", revision="abc")
        assert meta["size_source"] == "safetensors_header_shapes"
        assert meta["inferred_from_response"] is False
        assert meta["model_id"] == "org/model"
        assert meta["revision_requested"] == "abc"
        assert meta["index_total_size_bytes"] == 148
        assert meta["shard_bytes"] > 0

    def test_byte_size_shortcut_is_not_used(self, tmp_path, monkeypatch):
        # The mixed BF16/F32 population means total_size / 2 would give a
        # WRONG parameter count; the header shapes are authoritative.
        snapshot = _synthetic_snapshot(tmp_path)
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        meta = checkpoint_size_metadata("org/model")
        assert meta["index_total_size_bytes"] // 2 != \
            meta["checkpoint_parameter_count"]

    def test_unclassified_parameters_are_surfaced(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        _write_safetensors(snapshot / "model.safetensors",
                           {"weird.block.weight": ("BF16", [4, 4])})
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        meta = checkpoint_size_metadata("org/model")
        assert meta["unclassified_parameters"] == 16
        assert meta["unclassified_prefixes"] == {"weird.block.weight": 16}

    def test_missing_shard_listed_in_index_fails(self, tmp_path, monkeypatch):
        snapshot = _synthetic_snapshot(tmp_path)
        for shard in snapshot.glob("*.safetensors"):
            shard.unlink()
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        with pytest.raises(ReplayError, match="shard listed in index"):
            checkpoint_size_metadata("org/model")

    def test_uncached_checkpoint_points_at_preflight(self):
        with pytest.raises(ReplayError, match="download it during preflight"):
            resolve_snapshot_dir(
                "org/definitely-not-cached-anywhere-12345",
                local_files_only=True)


class TestMeasuredQwenPreflightEvidence:
    """Pins the 11.2 preflight artifacts committed for the cached
    Qwen3.5 targets (skipped where those artifacts are absent)."""

    @pytest.fixture(params=sorted(PINNED_SIZES))
    def preflight(self, request):
        path = PREFLIGHT_ROOT / request.param / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not committed")
        return request.param, json.loads(path.read_text(encoding="utf-8"))

    def test_status_is_pass_with_no_problems(self, preflight):
        key, report = preflight
        assert report["status"] == "PASS", report["problems"]
        assert report["problems"] == []
        assert report["model_key"] == key

    def test_resolved_revision_is_immutable(self, preflight):
        key, report = preflight
        assert is_immutable_revision(report["resolved_revision"])
        assert report["resolved_revision"] == PINNED_SIZES[key]["revision"]
        assert report["revision_is_immutable"] is False, \
            "the registry revision is still null before the 11.5 lock"

    def test_declared_sizes_match_the_measurement(self, preflight):
        key, report = preflight
        size = report["size_metadata"]
        pinned = PINNED_SIZES[key]
        assert size["checkpoint_parameter_count"] == pinned["total"]
        assert size["language_parameters"] == pinned["language"]
        assert size["vision_parameters"] == pinned["vision"]
        assert size["auxiliary_parameters"] == pinned["auxiliary"]
        assert size["unclassified_parameters"] == 0
        assert size["inferred_from_response"] is False

    def test_smoke_covers_an_image_and_a_text_variant(self, preflight):
        _, report = preflight
        variants = {entry["variant"] for entry in report["gpu_smoke"]}
        assert variants == {"cross_modal", "text_only"}
        by_variant = {e["variant"]: e["attempts"][0] for e in report["gpu_smoke"]}
        assert by_variant["cross_modal"]["image_token_count"] > 0
        assert by_variant["text_only"]["image_token_count"] == 0
        assert by_variant["cross_modal"]["output_token_count"] > 0
        for attempt in by_variant.values():
            assert attempt["finish_reason"] in {"eos", "stop", "length"}
            assert attempt["input_token_count"] > 0

    def test_greedy_generation_is_repeat_stable(self, preflight):
        _, report = preflight
        det = report["determinism"]
        assert det["greedy_decoding"] is True
        assert det["batch_size"] == 1
        assert det["requested_seed"] == 42
        assert det["all_variants_repeat_stable"] is True
        assert all(entry["deterministic"] for entry in report["gpu_smoke"])

    def test_recorded_hardware_is_the_slot_the_weights_ran_on(
            self, preflight):
        # Regression pin: device_map="cuda:3" leaves
        # torch.cuda.current_device() at 0, so the index must be read
        # from the weights. A recorded 0 here means the bug is back.
        _, report = preflight
        hardware = report["runtime_metadata"]["hardware"]
        assert report["device"] == "cuda:3"
        assert hardware["requested_device"] == "cuda:3"
        assert hardware["device_index"] == 3
        assert hardware["gpu_name"] == "NVIDIA RTX 6000 Ada Generation"
        assert hardware["compute_capability"] == "8.9"

    def test_frozen_prompt_and_cap_are_in_force(self, preflight):
        _, report = preflight
        assert report["system_prompt_sha256"] == \
            "e51b41e6a82264406aa184050eb0552cce8653ff097db9225e775a20b1bf7d9c"
        assert report["generation_config"]["max_new_tokens"] == 1536
        assert report["generation_config"]["do_sample"] is False
        assert report["effective_decoding"]["num_beams"] == 1
        assert report["model_spec"]["thinking_mode"] is False
        assert report["runtime_metadata"]["enable_thinking"] is False

    def test_smoke_used_the_frozen_scale_c_panel(self, preflight):
        _, report = preflight
        assert report["dataset"]["validated_families_sha256"]
        assert report["dataset"]["n_families_smoked"] == 1
        assert "families_panel" in report["dataset"]["input_dir"]

    def test_scale_arm_renders_an_identical_prompt(self):
        """The 2B/4B scale arm must differ in WEIGHTS ONLY: identical
        serialized prompts and image bytes, different responses."""
        paths = [PREFLIGHT_ROOT / key / "preflight.json"
                 for key in ("qwen35_2b", "qwen35_4b")]
        if not all(p.exists() for p in paths):
            pytest.skip("both Qwen preflight artifacts are required")
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        small, large = reports
        assert small["resolved_revision"] != large["resolved_revision"]
        for variant in ("cross_modal", "text_only"):
            a = next(e for e in small["gpu_smoke"]
                     if e["variant"] == variant)["attempts"][0]
            b = next(e for e in large["gpu_smoke"]
                     if e["variant"] == variant)["attempts"][0]
            assert a["serialized_prompt_hash"] == b["serialized_prompt_hash"]
            assert a["semantic_prompt_hash"] == b["semantic_prompt_hash"]
            assert a["ordered_image_hashes"] == b["ordered_image_hashes"]
            assert a["input_token_count"] == b["input_token_count"]
            assert a["response_sha256"] != b["response_sha256"]

    def test_4b_is_the_matched_scale_anchor(self):
        # Ministral-3-3B declares 3.4B language parameters; the matched
        # comparison uses the Qwen checkpoint of comparable size (4B),
        # not the 2B or 9B arm.
        registry_spec = resolve_model("ministral3_3b")
        declared = registry_spec.size_metadata["language_parameters"]
        path = PREFLIGHT_ROOT / "qwen35_4b" / "preflight.json"
        if not path.exists():
            pytest.skip("qwen35_4b preflight artifact not committed")
        measured = json.loads(path.read_text(encoding="utf-8"))[
            "size_metadata"]["language_parameters"]
        assert declared == 3400000000
        # Same order of magnitude: within a factor of two.
        assert 0.5 * measured < declared < 2.0 * measured


class TestCrossSlotReproducibility:
    """The GPU scheduling slot is not a scientific variable.

    The same checkpoint, prompt and greedy decoding were run on cuda:3
    (primary) and cuda:0 (control) and produced byte-identical
    responses. This is the empirical basis for excluding the slot from
    ``iteration11_run_fingerprint`` while retaining the hardware class.
    """

    PRIMARY = PREFLIGHT_ROOT / "qwen35_2b" / "preflight.json"
    CONTROL = PREFLIGHT_ROOT / "qwen35_2b" / "preflight_cross_slot_check.json"

    @pytest.fixture
    def pair(self):
        if not (self.PRIMARY.exists() and self.CONTROL.exists()):
            pytest.skip("cross-slot control artifact not committed")
        return (json.loads(self.PRIMARY.read_text(encoding="utf-8")),
                json.loads(self.CONTROL.read_text(encoding="utf-8")))

    def test_the_two_runs_used_different_slots(self, pair):
        primary, control = pair
        assert primary["device"] == "cuda:3"
        assert control["device"] == "cuda:0"
        assert primary["runtime_metadata"]["hardware"]["device_index"] == 3
        assert control["runtime_metadata"]["hardware"]["device_index"] == 0

    def test_same_checkpoint_and_same_gpu_class(self, pair):
        primary, control = pair
        assert primary["resolved_revision"] == control["resolved_revision"]
        assert is_immutable_revision(primary["resolved_revision"])
        assert primary["runtime_metadata"]["hardware"]["gpu_name"] == \
            control["runtime_metadata"]["hardware"]["gpu_name"]
        assert primary["runtime_metadata"]["hardware"][
            "compute_capability"] == control["runtime_metadata"]["hardware"][
                "compute_capability"]

    def test_responses_are_byte_identical_across_slots(self, pair):
        primary, control = pair
        for variant in ("cross_modal", "text_only"):
            a = [x["response_sha256"] for x in next(
                e for e in primary["gpu_smoke"]
                if e["variant"] == variant)["attempts"]]
            b = [x["response_sha256"] for x in next(
                e for e in control["gpu_smoke"]
                if e["variant"] == variant)["attempts"]]
            assert a == b, f"{variant} diverged across GPU slots"
            assert len(a) == len(b) == 2

    def test_token_accounting_is_identical_across_slots(self, pair):
        primary, control = pair
        for variant in ("cross_modal", "text_only"):
            a = next(e for e in primary["gpu_smoke"]
                     if e["variant"] == variant)["attempts"][0]
            b = next(e for e in control["gpu_smoke"]
                     if e["variant"] == variant)["attempts"][0]
            assert a["input_token_count"] == b["input_token_count"]
            assert a["image_token_count"] == b["image_token_count"]
            assert a["output_token_count"] == b["output_token_count"]
            assert a["serialized_prompt_hash"] == b["serialized_prompt_hash"]
