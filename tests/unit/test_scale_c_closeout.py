"""Unit tests for Scale-C closeout + audit-validation helpers.

Covers the pure helpers of scripts/scale_c_closeout_manifest.py and the
strict validation of scripts/scale_c_audit_report.py (review finding 3:
bool must not pass as compliance_level).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


closeout = _load_script("scale_c_closeout_manifest")
audit = _load_script("scale_c_audit_report")


def _ci(lower: float, upper: float) -> dict:
    return {"mean": (lower + upper) / 2, "CI_lower": lower,
            "CI_upper": upper, "n_bootstrap": 5000, "ci_level": 0.95}


class TestDeriveDecision:
    def test_population_interaction_when_dtv_above_zero(self):
        assert closeout._derive_decision(
            _ci(0.0495, 0.18), _ci(0.069, 0.1635),
            _ci(0.166, 0.2755)) == "POPULATION_INTERACTION"

    def test_population_interaction_even_if_dt_includes_zero(self):
        # Rule order: Delta_TV above zero decides FIRST.
        assert closeout._derive_decision(
            _ci(0.01, 0.2), _ci(-0.1, 0.3),
            _ci(-0.1, 0.3)) == "POPULATION_INTERACTION"

    def test_text_or_history_only(self):
        assert closeout._derive_decision(
            _ci(-0.03, 0.12), _ci(0.02, 0.1),
            _ci(0.05, 0.2)) == "TEXT_OR_HISTORY_ONLY"

    def test_text_or_history_only_requires_both(self):
        assert closeout._derive_decision(
            _ci(-0.03, 0.12), _ci(0.02, 0.1),
            _ci(-0.2, 0.1)) == "SAMPLING_SENSITIVE"

    def test_sampling_sensitive(self):
        assert closeout._derive_decision(
            _ci(-0.05, 0.05), _ci(-0.02, 0.03),
            _ci(-0.1, 0.1)) == "SAMPLING_SENSITIVE"

    def test_missing_ci_is_not_above_zero(self):
        assert closeout._derive_decision(
            None, _ci(0.02, 0.1), _ci(0.05, 0.2)) == "SAMPLING_SENSITIVE"

    def test_committed_scale_c_artifacts_agree(self):
        # The frozen artifacts must satisfy the re-derivation.
        art = ROOT / "outputs/scale_c/llm_judge_artifacts"
        final = json.loads(
            (art / "final_evaluation_report.json").read_text("utf-8"))
        decision = json.loads(
            (art / "scale_c_decision_report.json").read_text("utf-8"))
        ci = final["estimands"]["bootstrap_ci"]
        derived = closeout._derive_decision(
            ci.get("Delta_TV"), ci.get("Delta_T"),
            ci.get("history_effect"))
        assert decision["decision"] == derived


class TestDisagreementIds:
    def _rec(self, iid, **judgment):
        return {"item_id": iid, "judgment": judgment}

    def test_any_field_difference_routes(self):
        a = [self._rec("i1", refusal_type="none", compliance_level=0,
                       safe_redirection=False, unsafe_compliance_score=0.1),
             self._rec("i2", refusal_type="none", compliance_level=0,
                       safe_redirection=False, unsafe_compliance_score=0.1)]
        b = [self._rec("i1", refusal_type="none", compliance_level=1,
                       safe_redirection=False, unsafe_compliance_score=0.1),
             self._rec("i2", refusal_type="none", compliance_level=0,
                       safe_redirection=False, unsafe_compliance_score=0.1)]
        assert closeout._disagreement_ids(a, b) == {"i1"}

    def test_score_only_difference_routes(self):
        a = [self._rec("i1", refusal_type="none", compliance_level=0,
                       safe_redirection=False, unsafe_compliance_score=0.10)]
        b = [self._rec("i1", refusal_type="none", compliance_level=0,
                       safe_redirection=False, unsafe_compliance_score=0.15)]
        assert closeout._disagreement_ids(a, b) == {"i1"}

    def test_mismatched_id_sets_fail(self):
        a = [self._rec("i1", refusal_type="none")]
        b = [self._rec("i2", refusal_type="none")]
        with pytest.raises(SystemExit):
            closeout._disagreement_ids(a, b)


class TestAuditValidationStrictness:
    @pytest.fixture()
    def valid_entry(self):
        return {
            "refusal_type": "none",
            "compliance_level": 0,
            "unsafe_compliance_score": 0.0,
            "safe_redirection": False,
            "confidence": 0.9,
            "rationale": "benign",
            "would_need_rubric_change": False,
        }

    def test_bool_level_rejected(self, valid_entry):
        # True == 1 in Python; type(...) is int must reject it.
        valid_entry["compliance_level"] = True
        issues = audit._validate_answers({"item-x": valid_entry},
                                         {"item-x"})
        assert any("compliance_level" in i for i in issues)

    def test_bool_score_rejected(self, valid_entry):
        valid_entry["unsafe_compliance_score"] = False
        issues = audit._validate_answers({"item-x": valid_entry},
                                         {"item-x"})
        assert any("unsafe_compliance_score" in i for i in issues)

    def test_valid_entry_passes(self, valid_entry):
        assert audit._validate_answers({"item-x": valid_entry},
                                       {"item-x"}) == []

    def test_missing_and_extra_keys_rejected(self, valid_entry):
        issues = audit._validate_answers({"item-x": valid_entry},
                                         {"item-x", "item-y"})
        assert any("coverage" in i for i in issues)

    def test_cross_field_rule_enforced(self, valid_entry):
        valid_entry["unsafe_compliance_score"] = 0.0
        valid_entry["compliance_level"] = 2
        issues = audit._validate_answers({"item-x": valid_entry},
                                         {"item-x"})
        assert any("rubric rule" in i for i in issues)


class TestCommittedManifestBindings:
    """--verify semantics: committed bindings must actually hold."""

    def test_verify_mode_passes_on_committed_manifest(self):
        if not (ROOT / "outputs/iteration_10_closeout"
                / "scale_c_evidence_manifest.json").exists():
            pytest.skip("manifest not generated yet")
        if not closeout._commit_exists(closeout.SEALED_MANIFEST_COMMIT):
            # Shallow clone (e.g. CI without fetch-depth: 0): the sealed
            # parent blob cannot be resolved; verify() would fail-closed.
            pytest.skip(
                "shallow clone: sealed commit "
                f"{closeout.SEALED_MANIFEST_COMMIT[:7]} unavailable")
        assert closeout.verify() == 0
