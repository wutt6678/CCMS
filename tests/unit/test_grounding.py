"""Unit tests for canonical-q grounding flags.

Covers the repair gate: flag cases where q* contains unresolved image
references in text-only/reset conditions, and pin the committed
human-reviewed Scale-A build as flag-free research evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_mllm.construction.grounding import flag_grounding_issues
from causal_mllm.construction.harmonize import (
    CallableHarmonizer,
    ManualHarmonizer,
    apply_terminal_harmonization,
)
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.validation import (
    validate_factorial_relations,
    validate_factorial_semantic_eligibility,
)
from tests.unit.test_variants import _full_annotator, _skeleton

SMOKE_FAMILIES = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "families" / "scale_a_smoke" / "families.jsonl"
)
SCALE_B_FAMILIES = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "families" / "scale_b_smoke" / "families.jsonl"
)
SCALE_B_VALIDATED = SCALE_B_FAMILIES.with_name("validated_families.jsonl")
SCALE_B_REPORT = SCALE_B_FAMILIES.with_name("validation_report.json")

DEICTIC_Q = ("If helmets needed to be removed rapidly during an "
             "emergency drill, which visible features on these helmets "
             "would likely allow the quickest doffing?")
CLEAN_Q = ("Which helmet features would typically allow the quickest "
           "removal during an emergency drill?")


def _built_family(tmp_path, canonical_q: str, *, fill_grounding: bool):
    skeleton = _skeleton(tmp_path)
    annotated = _full_annotator().annotate_family(skeleton)
    key = skeleton.source["source_id"]
    if fill_grounding:
        path = tmp_path / "h.json"
        path.write_text(json.dumps({key: {
            "canonical_q": canonical_q,
            "canonical_q_grounding_valid": True,
            "canonical_q_no_unintended_modality_dependency": True,
            "canonical_q_semantically_preserves_mm_source": True,
            "canonical_q_semantically_preserves_text_source": True,
        }}))
        harmonized = apply_terminal_harmonization(
            annotated, ManualHarmonizer(path))
    else:
        harmonized = apply_terminal_harmonization(
            annotated,
            CallableHarmonizer(lambda fk, mm, tx: canonical_q,
                               model_name="test-llm"))
    from causal_mllm.construction.variants import build_family_variants
    return build_family_variants(harmonized)


class TestGroundingFlags:
    def test_deictic_canonical_q_flagged_everywhere(self, tmp_path):
        family = _built_family(tmp_path, DEICTIC_Q, fill_grounding=True)
        flags = flag_grounding_issues(family)
        scopes = {f["scope"] for f in flags}
        # The q* itself and every text-only condition that repeats it
        assert "canonical_q" in scopes
        assert "history_reset" in scopes
        assert "text_only" in scopes
        assert "neutral" in scopes

    def test_clean_canonical_q_no_deictic_flags(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        flags = flag_grounding_issues(family)
        assert flags == []

    def test_unresolved_grounding_targets_flagged(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=False)
        flags = flag_grounding_issues(family)
        assert [f for f in flags if f["scope"] == "grounding_targets"]


class TestCommittedScaleAEvidence:
    """Pins the human-reviewed Scale-A build: zero grounding flags.

    This reads the COMMITTED research evidence (not placeholder
    output); if the smoke set is regenerated, the review must be
    redone and this test re-validated.
    """

    def test_human_reviewed_build_is_flag_free(self):
        assert SMOKE_FAMILIES.exists(), "committed smoke build missing"
        records = [json.loads(line)
                   for line in SMOKE_FAMILIES.open(encoding="utf-8")]
        assert len(records) == 5
        for record in records:
            family = CausalFamily.from_dict(record)
            assert set(family.variants) == {
                "neutral", "text_only", "vision_only",
                "cross_modal", "shuffle", "history_reset"}
            flags = flag_grounding_issues(family)
            assert flags == [], (family.family_id, flags)
            block = family.validation["terminal_harmonization"]
            assert block["validation"] == "human"
            assert block["method"] == "manual"

    def test_negative_controls_stay_out_of_research_set(self):
        controls_path = SMOKE_FAMILIES.with_name("negative_controls.jsonl")
        controls = [json.loads(line)
                    for line in controls_path.open(encoding="utf-8")]
        assert len(controls) == 5
        built_ids = {
            json.loads(line)["source"]["source_id"]
            for line in SMOKE_FAMILIES.open(encoding="utf-8")
        }
        control_ids = {c["source_id"] for c in controls}
        assert built_ids.isdisjoint(control_ids)


class TestCommittedScaleBEvidence:
    """Pins the Iteration-7 Scale-B research smoke set: 20 families,
    120 trajectories, flag-free, with 41 decided negative controls."""

    def test_scale_b_20_families_120_trajectories_flag_free(self):
        assert SCALE_B_FAMILIES.exists(), "committed Scale-B build missing"
        records = [json.loads(line)
                   for line in SCALE_B_FAMILIES.open(encoding="utf-8")]
        assert len(records) == 20
        for record in records:
            family = CausalFamily.from_dict(record)
            assert set(family.variants) == {
                "neutral", "text_only", "vision_only",
                "cross_modal", "shuffle", "history_reset"}
            assert flag_grounding_issues(family) == [], family.family_id
            block = family.validation["terminal_harmonization"]
            assert block["validation"] == "human"
            assert block["method"] == "manual"

    def test_scale_b_negative_controls_are_decided_not_pending(self):
        controls_path = SCALE_B_FAMILIES.with_name("negative_controls.jsonl")
        controls = [json.loads(line)
                    for line in controls_path.open(encoding="utf-8")]
        assert len(controls) == 41
        for control in controls:
            assert any("NOT equivalent" in r for r in control["reasons"])
        built_ids = {
            json.loads(line)["source"]["source_id"]
            for line in SCALE_B_FAMILIES.open(encoding="utf-8")
        }
        assert built_ids.isdisjoint({c["source_id"] for c in controls})

    def test_scale_b_factorial_relations_hold_on_committed_artifact(self):
        """The firewall re-derived from the artifact alone must pass.

        Media-file checks are skipped: the (git-ignored) media store
        is absent in the offline CI unit job. Structural relations —
        image placement, identical vision hashes across H01/H11/
        shuffle, H11-vs-shuffle multiset + permutation, terminal-hash
        invariant — hold regardless.
        """
        for line in SCALE_B_FAMILIES.open(encoding="utf-8"):
            family = CausalFamily.from_dict(json.loads(line))
            errors = validate_factorial_relations(
                family, check_media_files=False)
            assert errors == [], (family.family_id, errors)
            # The persisted annotations must still carry the POSITIVE
            # Iteration-5 evidence that justified the built variants.
            semantic = validate_factorial_semantic_eligibility(family)
            assert semantic == [], (family.family_id, semantic)

    def test_scale_b_validation_artifacts_pinned(self):
        """Pins the validation artifacts themselves, not just inputs."""
        built_ids = {
            json.loads(line)["source"]["source_id"]
            for line in SCALE_B_FAMILIES.open(encoding="utf-8")
        }
        validated = [json.loads(line)
                     for line in SCALE_B_VALIDATED.open(encoding="utf-8")]
        assert len(validated) == 20
        validated_ids = {r["source"]["source_id"] for r in validated}
        assert validated_ids == built_ids

        report = json.loads(SCALE_B_REPORT.read_text(encoding="utf-8"))
        assert report["n_input"] == 20
        assert report["n_validated"] == 20
        assert report["n_excluded"] == 0
        assert report["judge"] is None
        assert report["strict_causal_subset"] == []
        assert len(report["families"]) == 20
        for entry in report["families"]:
            assert entry["status"] == "validated"
            assert entry["automatic_errors"] == []
            assert entry["factorial_cells"] == {
                "neutral": [0, 0], "text_only": [1, 0],
                "vision_only": [0, 1], "cross_modal": [1, 1],
            }
