"""Unit tests for Iteration 11.0 protocol freeze (cross-model).

Portable per the CI-portability lesson: repo root is derived from
__file__; checks that need the full git history (sealed-blob verify) or
that would touch large/committed-evidence are skip-guarded. The
protocol artifacts themselves are committed, so structural checks run
in CI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


freeze = _load_script("iter11_freeze_protocol")
closeout = _load_script("scale_c_closeout_manifest")

PROTOCOL_DIR = ROOT / "outputs" / "iteration_11" / "protocol"
REG_PATH = PROTOCOL_DIR / "model_registry.yaml"
PROTO_PATH = PROTOCOL_DIR / "iteration_11_protocol.json"

NEW_KEYS = ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm")
ALL_KEYS = NEW_KEYS + ("qwen35_9b",)


class TestImmutableRevision:
    def test_accepts_full_sha(self):
        assert freeze.is_immutable_revision(
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a")

    @pytest.mark.parametrize("bad", [
        None, "", "main", "master", "HEAD", "latest", "Main",
        "c202236",                     # short sha
        "branch/feature",
        "z" * 40,                      # non-hex
        12345,                         # wrong type
    ])
    def test_rejects_floating_or_invalid(self, bad):
        assert not freeze.is_immutable_revision(bad)

    def test_confirmatory_rejects_null_and_branch(self):
        for bad in (None, "main", "master", "HEAD"):
            with pytest.raises(ValueError):
                freeze.assert_confirmatory_revision("m", bad)

    def test_confirmatory_accepts_sha(self):
        freeze.assert_confirmatory_revision(
            "m", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


class TestRegistryValidation:
    def test_committed_registry_validates(self):
        if not REG_PATH.exists():
            pytest.skip("registry not generated yet")
        reg = yaml.safe_load(REG_PATH.read_text(encoding="utf-8"))
        assert freeze.validate_registry(reg) == []

    def test_built_registry_validates(self):
        assert freeze.validate_registry(freeze.build_registry()) == []

    def test_all_keys_resolve_once(self):
        reg = freeze.build_registry()
        assert sorted(reg["models"]) == sorted(ALL_KEYS)
        assert len(set(reg["models"])) == len(ALL_KEYS)

    def test_new_models_null_revision_9b_pinned(self):
        reg = freeze.build_registry()
        for k in NEW_KEYS:
            assert reg["models"][k]["revision"] is None
        assert freeze.is_immutable_revision(
            reg["models"]["qwen35_9b"]["revision"])

    def test_rejects_bad_dtype(self):
        reg = freeze.build_registry()
        reg["models"]["qwen35_2b"]["dtype"] = "float16"
        assert any("bfloat16" in i for i in freeze.validate_registry(reg))

    def test_rejects_quantization(self):
        reg = freeze.build_registry()
        reg["models"]["phi4_mm"]["quantization"] = "int8"
        assert any("quantization" in i for i in freeze.validate_registry(reg))

    def test_rejects_thinking_mode_true(self):
        reg = freeze.build_registry()
        reg["models"]["ministral3_3b"]["thinking_mode"] = True
        assert any("thinking_mode" in i
                   for i in freeze.validate_registry(reg))

    def test_rejects_missing_fields(self):
        reg = freeze.build_registry()
        del reg["models"]["qwen35_4b"]["adapter"]
        assert any("missing fields" in i for i in freeze.validate_registry(reg))

    def test_rejects_missing_fallback_and_excluded(self):
        reg = freeze.build_registry()
        del reg["fallback"]
        del reg["excluded_models"]
        issues = freeze.validate_registry(reg)
        assert any("fallback" in i for i in issues)
        assert any("excluded_models" in i for i in issues)

    def test_rejects_empty_models(self):
        assert freeze.validate_registry({"models": {}}) != []


class TestProtocolContent:
    @pytest.fixture()
    def proto(self):
        if not PROTO_PATH.exists():
            pytest.skip("protocol not generated yet")
        return json.loads(PROTO_PATH.read_text(encoding="utf-8"))

    def test_required_sections_present(self, proto):
        for sec in ("protocol", "frozen_inputs", "model_matrix", "fallback",
                    "excluded_models", "phi4_load_strategy",
                    "uniform_cap_rule", "hypotheses", "multiplicity",
                    "provenance_schema_per_record", "resume_key",
                    "stop_conditions"):
            assert sec in proto, f"missing section {sec}"

    def test_frozen_system_prompt_sha(self, proto):
        assert proto["frozen_inputs"]["system_prompt_sha256"] == \
            freeze.FROZEN_SYSTEM_PROMPT_SHA

    def test_iter10_generation_config_verbatim(self, proto):
        assert proto["frozen_inputs"][
            "iteration_10_generation_config_verbatim"] == \
            freeze.ITER10_GENERATION_CONFIG

    def test_effective_decoding_is_greedy_normalized(self, proto):
        d = proto["frozen_inputs"]["effective_decoding"]
        assert d["do_sample"] is False
        assert d["num_beams"] == 1
        assert d["max_new_tokens"] == 1536
        # inert sampling values normalized to omitted
        assert d["temperature"] is None
        assert d["top_p"] is None
        assert d["top_k"] is None

    def test_model_matrix_keys_match_registry(self, proto):
        mm_keys = {m["model_key"] for m in proto["model_matrix"]}
        assert mm_keys == set(ALL_KEYS)

    def test_phi4_shim_strategy_recorded(self, proto):
        s = proto["phi4_load_strategy"]
        assert s["decision"] == "shim_in_shared_env"
        assert s["dtype"] == "bfloat16" and s["quantization"] == "none"
        assert any("sdpa" in f for f in s["fixes"])
        assert any("prepare_inputs_for_generation" in f for f in s["fixes"])

    def test_fallback_and_exclusions_recorded(self, proto):
        assert proto["fallback"]["model_key"] == "gemma3_4b"
        assert proto["fallback"]["replaces"] == "phi4_mm"
        excluded = {m["model_id"] for m in proto["excluded_models"]}
        assert {"InternVL3.5-4B", "Molmo2-4B"} <= excluded

    def test_hypotheses_and_multiplicity(self, proto):
        for h in ("H1", "H2", "H3", "H4", "H5"):
            assert h in proto["hypotheses"]
        assert proto["multiplicity"]["family_wise_correction"] == \
            "Holm-Bonferroni"
        assert proto["multiplicity"]["n_confirmatory_model_tests"] == 4

    def test_resume_key(self, proto):
        assert proto["resume_key"] == freeze.RESUME_KEY


class TestImageManifest:
    def test_aggregate_is_deterministic_and_bound_to_panel(self):
        if not freeze.PANEL.exists():
            pytest.skip("panel not present")
        a = freeze.image_manifest()
        b = freeze.image_manifest()
        assert a["aggregate_manifest_sha256"] == b["aggregate_manifest_sha256"]
        assert a["n_unique_images"] == 100
        assert a["ci_reproducible"] is True


class TestVerifyGate:
    def test_verify_gate_passes(self):
        if not (REG_PATH.exists() and PROTO_PATH.exists()):
            pytest.skip("protocol artifacts not generated yet")
        if not closeout._commit_exists(closeout.SEALED_MANIFEST_COMMIT):
            pytest.skip("shallow clone: Iteration 10 sealed blob unavailable")
        assert freeze.verify_gate() == 0
