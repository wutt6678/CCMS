"""Iteration 11.3: Ministral-3 adapter + preflight lock file.

CI-safe: the adapter is exercised with fake model/processor/tokenizer
objects and fake id vectors, so no checkpoint is downloaded and torch is
never imported. The measured values are pinned from the committed
preflight artifact and lock file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from causal_mllm.replay import ReplayConfig, ReplayError, build_adapter
from causal_mllm.replay.adapters.ministral3 import (
    IMAGE_TOKEN, Ministral3Adapter, VENDOR_DEFAULT_MARKERS)
from causal_mllm.replay.registry import (
    DEFAULT_LOCK, dependency_lock_sha256, is_immutable_revision,
    load_lock, resolve_model, update_lock)

PREFLIGHT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "iteration_11" / "preflight")
MINISTRAL_PREFLIGHT = PREFLIGHT_ROOT / "ministral3_3b" / "preflight.json"

MINISTRAL_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
MINISTRAL_SIZE = {
    "total": 3849090048,
    "language": 3429006336,
    "vision": 420083712,
    "auxiliary": 0,
    "tensors": 458,
    "shards": 2,
}
IMAGE_TOKEN_ID = 10
IMAGE_BREAK_TOKEN_ID = 12
IMAGE_END_TOKEN_ID = 13
EOS_TOKEN_ID = 2
VENDOR_PROMPT_SHA = (
    "331b249682cd52226c50e533f59825184997f3b31f9a62b2c3fea940db6999c5")


# --- fakes -------------------------------------------------------------
class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModel:
    def __init__(self, config=None, eos=None):
        self.config = config if config is not None else _FakeConfig()
        self.generation_config = _FakeConfig(eos_token_id=eos)


class _FakeTokenizer:
    def __init__(self, ids=None):
        self._ids = ids or {}

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token)


class _FakeProcessor:
    def __init__(self, tokenizer=None, image_token_id=None):
        self.tokenizer = tokenizer
        if image_token_id is not None:
            self.image_token_id = image_token_id


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


class _FakeInputs:
    def __init__(self, ids):
        self._data = {"input_ids": [_FakeIdRow(ids)]}

    def __getitem__(self, key):
        return self._data[key]


def _adapter(**spec_kwargs) -> Ministral3Adapter:
    spec = resolve_model("ministral3_3b")
    return Ministral3Adapter(ReplayConfig(), spec, **spec_kwargs)


def _wire(adapter, *, image_token_index=IMAGE_TOKEN_ID,
          processor_image_token_id=None, tokenizer_ids=None, eos=EOS_TOKEN_ID,
          ids=()):
    adapter.model = _FakeModel(
        config=_FakeConfig(image_token_index=image_token_index), eos=eos)
    adapter.processor = _FakeProcessor(
        tokenizer=_FakeTokenizer(tokenizer_ids),
        image_token_id=processor_image_token_id)
    return _FakeInputs(list(ids))


# ---------------------------------------------------------------------
# Adapter dispatch and family differences
# ---------------------------------------------------------------------
class TestMinistral3Adapter:
    def test_registry_dispatches_to_the_ministral_adapter(self):
        adapter = build_adapter(resolve_model("ministral3_3b"),
                                ReplayConfig())
        assert isinstance(adapter, Ministral3Adapter)
        assert adapter.adapter_name == "ministral3"

    def test_no_thinking_switch_is_passed(self):
        # The official template rejects enable_thinking; the frozen
        # protocol's "thinking off" is the template's only behaviour.
        assert _adapter().chat_template_kwargs() == {}
        assert _adapter().extra_runtime_metadata()[
            "thinking_switch_available"] is False

    def test_no_remote_code_is_executed(self):
        assert resolve_model("ministral3_3b").trust_remote_code is False
        assert _adapter()._pretrained_kwargs() in (
            {}, {"revision": MINISTRAL_REVISION})

    def test_image_token_id_prefers_config_image_token_index(self):
        adapter = _adapter()
        _wire(adapter, image_token_index=IMAGE_TOKEN_ID)
        assert adapter._image_token_id() == IMAGE_TOKEN_ID

    def test_image_token_id_falls_back_to_the_processor(self):
        adapter = _adapter()
        _wire(adapter, image_token_index=None,
              processor_image_token_id=77)
        assert adapter._image_token_id() == 77

    def test_image_token_id_falls_back_to_the_tokenizer(self):
        adapter = _adapter()
        _wire(adapter, image_token_index=None,
              tokenizer_ids={IMAGE_TOKEN: IMAGE_TOKEN_ID})
        assert adapter._image_token_id() == IMAGE_TOKEN_ID

    def test_unresolvable_image_token_id_is_none_not_zero_guessed(self):
        adapter = _adapter()
        _wire(adapter, image_token_index=None, tokenizer_ids={})
        assert adapter._image_token_id() is None
        assert adapter.count_image_tokens(_FakeInputs([1, 2, 3])) == 0

    def test_counts_image_placeholder_tokens(self):
        # PixtralProcessor exposes no image_token_id and config
        # .image_token_id is null, so the generic path would report 0.
        adapter = _adapter()
        ids = [1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5, IMAGE_BREAK_TOKEN_ID,
               IMAGE_TOKEN_ID, IMAGE_END_TOKEN_ID]
        inputs = _wire(adapter, ids=ids)
        assert adapter.count_image_tokens(inputs) == 3

    def test_marker_counts_are_reported_separately(self):
        adapter = _adapter()
        ids = [IMAGE_TOKEN_ID] * 4 + [IMAGE_BREAK_TOKEN_ID] * 2 + \
              [IMAGE_END_TOKEN_ID]
        inputs = _wire(adapter,
                       tokenizer_ids={"[IMG_BREAK]": IMAGE_BREAK_TOKEN_ID,
                                      "[IMG_END]": IMAGE_END_TOKEN_ID},
                       ids=ids)
        counts = adapter._marker_counts(inputs)
        assert counts["image_token_count"] == 4
        assert counts["image_break_token_count"] == 2
        assert counts["image_end_token_count"] == 1

    def test_eos_includes_the_generation_config_value(self):
        adapter = _adapter()
        _wire(adapter, eos=EOS_TOKEN_ID)
        assert EOS_TOKEN_ID in adapter._eos_token_ids()

    def test_eos_survives_a_missing_generation_config_value(self):
        adapter = _adapter()
        _wire(adapter, eos=None)
        assert isinstance(adapter._eos_token_ids(), set)


# ---------------------------------------------------------------------
# Prompt integrity: the vendor default must never reach the model
# ---------------------------------------------------------------------
class TestVendorSystemPromptSuppression:
    def test_leak_is_detected_for_every_marker(self):
        adapter = _adapter()
        _wire(adapter)
        for marker in VENDOR_DEFAULT_MARKERS:
            diagnostics = adapter.adapter_diagnostics(
                f"[SYSTEM_PROMPT]{marker}[/SYSTEM_PROMPT]hello",
                _FakeInputs([]))
            assert diagnostics[
                "vendor_default_system_prompt_injected"] is True, marker
            assert marker in diagnostics["vendor_default_markers_found"]

    def test_clean_prompt_reports_no_leak(self):
        adapter = _adapter()
        _wire(adapter)
        diagnostics = adapter.adapter_diagnostics(
            "[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT]"
            "[INST]hi[/INST]", _FakeInputs([]))
        assert diagnostics["vendor_default_system_prompt_injected"] is False
        assert diagnostics["vendor_default_markers_found"] == []

    def test_frozen_prompt_presence_is_verified(self):
        adapter = _adapter()
        _wire(adapter)
        frozen = adapter.config.system_prompt.strip()
        present = adapter.adapter_diagnostics(
            f"[SYSTEM_PROMPT]{frozen}[/SYSTEM_PROMPT]x", _FakeInputs([]))
        absent = adapter.adapter_diagnostics(
            "[SYSTEM_PROMPT]something else[/SYSTEM_PROMPT]x",
            _FakeInputs([]))
        assert present["frozen_system_prompt_present_verbatim"] is True
        assert absent["frozen_system_prompt_present_verbatim"] is False

    def test_tokenizer_regex_flag_state_is_recorded(self):
        adapter = _adapter()
        _wire(adapter)
        diagnostics = adapter.adapter_diagnostics("x", _FakeInputs([]))
        assert "tokenizer_fix_mistral_regex" in diagnostics

    def test_vendor_prompt_is_identified_by_hash(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "SYSTEM_PROMPT.txt").write_text(
            "You are Ministral, created by Mistral AI.", encoding="utf-8")
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        adapter = _adapter()
        _wire(adapter)
        expected = hashlib.sha256(
            (snapshot / "SYSTEM_PROMPT.txt").read_bytes()).hexdigest()
        assert adapter._vendor_system_prompt_sha256() == expected
        assert adapter.adapter_diagnostics("x", _FakeInputs([]))[
            "vendor_default_system_prompt_sha256"] == expected

    def test_missing_vendor_prompt_is_none_not_a_guess(self, tmp_path,
                                                       monkeypatch):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        monkeypatch.setattr(
            "causal_mllm.replay.checkpoint_size.resolve_snapshot_dir",
            lambda *a, **k: snapshot)
        adapter = _adapter()
        _wire(adapter)
        assert adapter._vendor_system_prompt_sha256() is None


# ---------------------------------------------------------------------
# Lock file semantics
# ---------------------------------------------------------------------
class TestPreflightLock:
    def test_lock_records_an_immutable_revision(self, tmp_path):
        path = tmp_path / "lock.yaml"
        update_lock("m1", revision="a" * 40, lock_path=path)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["models"]["m1"]["revision"] == "a" * 40
        assert loaded["models"]["m1"]["processor_revision"] == "a" * 40

    def test_lock_refuses_a_floating_revision(self, tmp_path):
        with pytest.raises(ReplayError, match="refusing to lock"):
            update_lock("m1", revision="main", lock_path=tmp_path / "l.yaml")

    def test_lock_refuses_to_move_a_pinned_revision(self, tmp_path):
        path = tmp_path / "lock.yaml"
        update_lock("m1", revision="a" * 40, lock_path=path)
        with pytest.raises(ReplayError, match="refusing to move it"):
            update_lock("m1", revision="b" * 40, lock_path=path)
        # The original pin survived the refused attempt.
        assert load_lock(path)["m1"]["revision"] == "a" * 40

    def test_explicit_repin_preserves_the_superseded_value(self, tmp_path):
        path = tmp_path / "lock.yaml"
        update_lock("m1", revision="a" * 40, lock_path=path)
        update_lock("m1", revision="b" * 40, allow_change=True,
                    lock_path=path)
        entry = load_lock(path)["m1"]
        assert entry["revision"] == "b" * 40
        assert entry["superseded_revisions"] == ["a" * 40]

    def test_lock_merges_models_and_keeps_the_dependency_lock(self, tmp_path):
        path = tmp_path / "lock.yaml"
        dep = {"pip_freeze_sha256": "d" * 64, "n_packages": 3}
        update_lock("m1", revision="a" * 40, dependency_lock=dep,
                    lock_path=path)
        update_lock("m2", revision="b" * 40, lock_path=path)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(loaded["models"]) == {"m1", "m2"}
        assert loaded["dependency_lock"] == dep

    def test_measured_size_is_recorded(self, tmp_path):
        path = tmp_path / "lock.yaml"
        update_lock("m1", revision="a" * 40,
                    measured_size={"checkpoint_parameter_count": 42},
                    lock_path=path)
        assert load_lock(path)["m1"]["measured_size"][
            "checkpoint_parameter_count"] == 42

    def test_dependency_lock_hash_is_none_when_absent(self, tmp_path):
        assert dependency_lock_sha256(tmp_path / "missing.yaml") is None

    def test_dependency_lock_hash_is_stable(self, tmp_path):
        path = tmp_path / "lock.yaml"
        dep = {"pip_freeze_sha256": "d" * 64, "n_packages": 3}
        update_lock("m1", revision="a" * 40, dependency_lock=dep,
                    lock_path=path)
        first = dependency_lock_sha256(path)
        update_lock("m2", revision="b" * 40, lock_path=path)
        assert dependency_lock_sha256(path) == first
        assert is_immutable_revision(first) is False  # it is a plain hash
        assert len(first) == 64

    def test_dependency_lock_hash_changes_with_the_environment(self,
                                                              tmp_path):
        path = tmp_path / "lock.yaml"
        update_lock("m1", revision="a" * 40,
                    dependency_lock={"pip_freeze_sha256": "d" * 64},
                    lock_path=path)
        before = dependency_lock_sha256(path)
        update_lock("m1", revision="a" * 40,
                    dependency_lock={"pip_freeze_sha256": "e" * 64},
                    lock_path=path)
        assert dependency_lock_sha256(path) != before


# ---------------------------------------------------------------------
# Committed 11.3 preflight evidence
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def ministral_report():
    if not MINISTRAL_PREFLIGHT.exists():
        pytest.skip("ministral3_3b preflight artifact not committed")
    return json.loads(MINISTRAL_PREFLIGHT.read_text(encoding="utf-8"))


class TestMinistralPreflightEvidence:
    def test_passed_with_no_problems(self, ministral_report):
        assert ministral_report["status"] == "PASS"
        assert ministral_report["problems"] == []
        assert ministral_report["model_key"] == "ministral3_3b"

    def test_revision_is_immutable_and_locked(self, ministral_report):
        assert ministral_report["resolved_revision"] == MINISTRAL_REVISION
        assert is_immutable_revision(MINISTRAL_REVISION)
        assert ministral_report["processor_revision"] == MINISTRAL_REVISION
        if DEFAULT_LOCK.exists():
            assert load_lock(DEFAULT_LOCK)["ministral3_3b"]["revision"] == \
                MINISTRAL_REVISION
            # With the lock present the target is confirmatory-eligible.
            assert is_immutable_revision(
                resolve_model("ministral3_3b", confirmatory=True).revision)

    def test_measured_size_matches(self, ministral_report):
        size = ministral_report["size_metadata"]
        assert size["checkpoint_parameter_count"] == MINISTRAL_SIZE["total"]
        assert size["language_parameters"] == MINISTRAL_SIZE["language"]
        assert size["vision_parameters"] == MINISTRAL_SIZE["vision"]
        assert size["auxiliary_parameters"] == MINISTRAL_SIZE["auxiliary"]
        assert size["unclassified_parameters"] == 0
        assert size["n_tensors"] == MINISTRAL_SIZE["tensors"]
        assert size["n_shards"] == MINISTRAL_SIZE["shards"]
        assert size["revision_used"] == MINISTRAL_REVISION
        assert size["stored_dtype_histogram"] == {"BF16": 458}

    def test_measured_size_supersedes_the_registry_approximation(
            self, ministral_report):
        # The frozen registry carries the specification's round numbers;
        # the measurement is authoritative and close to them.
        declared = resolve_model("ministral3_3b").size_metadata
        measured = ministral_report["size_metadata"]
        assert declared["language_parameters"] == 3400000000
        assert measured["language_parameters"] == 3429006336
        assert abs(measured["language_parameters"]
                   - declared["language_parameters"]) / declared[
                       "language_parameters"] < 0.05

    def test_vendor_prompt_suppressed_and_frozen_prompt_verbatim(
            self, ministral_report):
        for entry in ministral_report["gpu_smoke"]:
            for attempt in entry["attempts"]:
                diag = attempt["adapter_diagnostics"]
                assert diag[
                    "vendor_default_system_prompt_injected"] is False
                assert diag["vendor_default_markers_found"] == []
                assert diag[
                    "frozen_system_prompt_present_verbatim"] is True
                assert diag["vendor_default_system_prompt_sha256"] == \
                    VENDOR_PROMPT_SHA

    def test_image_tokens_are_accounted_for(self, ministral_report):
        by_variant = {e["variant"]: e["attempts"][0]
                      for e in ministral_report["gpu_smoke"]}
        vision = by_variant["cross_modal"]
        assert vision["image_token_count"] == 121
        assert vision["adapter_diagnostics"]["image_token_id"] == \
            IMAGE_TOKEN_ID
        assert vision["adapter_diagnostics"]["image_token_count"] == 121
        assert vision["adapter_diagnostics"]["image_break_token_count"] == 10
        assert vision["adapter_diagnostics"]["image_end_token_count"] == 1
        assert by_variant["text_only"]["image_token_count"] == 0

    def test_generation_is_repeat_stable_and_complete(
            self, ministral_report):
        det = ministral_report["determinism"]
        assert det["greedy_decoding"] is True
        assert det["all_variants_repeat_stable"] is True
        for entry in ministral_report["gpu_smoke"]:
            assert entry["deterministic"] is True
            assert entry["repeats"] == 2
            attempt = entry["attempts"][0]
            assert attempt["finish_reason"] == "eos"
            assert attempt["output_token_count"] > 0
            assert attempt["hit_max_new_tokens"] is False

    def test_frozen_prompt_and_cap_are_in_force(self, ministral_report):
        assert ministral_report["system_prompt_sha256"] == \
            "e51b41e6a82264406aa184050eb0552cce8653ff097db9225e775a20b1bf7d9c"
        assert ministral_report["generation_config"]["max_new_tokens"] == 1536
        assert ministral_report["generation_config"]["do_sample"] is False
        assert ministral_report["model_spec"]["thinking_mode"] is False

    def test_family_parity_notes_are_recorded(self, ministral_report):
        notes = " ".join(ministral_report["parity_notes"])
        assert "consolidated.safetensors" in notes
        assert "fix_mistral_regex" in notes
        assert "SYSTEM_PROMPT.txt" in notes
