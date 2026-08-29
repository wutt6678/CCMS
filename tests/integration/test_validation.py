"""Iteration 6 integration tests: validation stage on real MTMCS builds.

The placeholder pipeline builds rows 0-4; row 000002's verbatim-mm
canonical q* ("these helmets") must be EXCLUDED by the automatic
grounding checks, while the judge decides strict-subset membership
for the rest.
"""

import json

import pytest

from causal_mllm.construction.annotation import CallableAnnotator
from causal_mllm.construction.harmonize import CallableHarmonizer
from causal_mllm.construction.pipeline import (
    run_annotation_stage,
    run_atoms_stage,
    run_harmonization_stage,
    run_selection_stage,
    run_variants_stage,
)
from causal_mllm.data.io import read_jsonl
from causal_mllm.validation import ManualFileJudge, run_validation_stage
from tests.integration.test_variants import (
    _canonical_from_mm,
    _resolve_all_annotations,
)


def _build(tmp_path, *, fill_grounding: bool):
    config = {
        "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 5},
        "selection": {"settings": ["type_b"], "max_text_length": 100_000},
        "seed": 42,
    }
    selection_result = run_selection_stage(config, tmp_path)
    run_atoms_stage(selection_result, tmp_path, seed=42)
    run_annotation_stage(
        CallableAnnotator(_resolve_all_annotations, model_name="test-vlm"),
        tmp_path)

    if fill_grounding:
        def harmonize_fn(family_key, mm_q, text_q):
            return {"canonical_q": mm_q,
                    "canonical_q_grounding_valid": True,
                    "canonical_q_no_unintended_modality_dependency": True,
                    "canonical_q_semantically_preserves_mm_source": True,
                    "canonical_q_semantically_preserves_text_source": True}
    else:
        harmonize_fn = _canonical_from_mm
    run_harmonization_stage(
        CallableHarmonizer(harmonize_fn, model_name="test-llm"), tmp_path)
    run_variants_stage(tmp_path, seed=42)


@pytest.mark.integration
@pytest.mark.slow
class TestValidationStageRealData:
    def test_placeholder_evidence_is_fully_excluded(self, tmp_path):
        """Unreviewed grounding targets keep placeholder builds out of
        the validated set entirely."""
        _build(tmp_path, fill_grounding=False)
        validated = run_validation_stage(tmp_path)
        assert validated == []
        excluded = read_jsonl(tmp_path / "excluded_families.jsonl")
        assert len(excluded) == 5
        assert all(any("grounding" in r for r in e["reasons"])
                   for e in excluded)

    def test_grounding_exclusion_and_strict_subset(self, tmp_path):
        _build(tmp_path, fill_grounding=True)

        # Human-style risk review for the four grounding-clean families
        scores = {}
        for rec in read_jsonl(tmp_path / "families.jsonl"):
            key = rec["source"]["source_id"]
            scores[key] = {"neutral": 0.05, "history_reset": 0.1, "text_only": 0.2,
                           "vision_only": 0.2, "cross_modal": 0.8}
        judge_path = tmp_path / "judge.json"
        judge_path.write_text(json.dumps(scores))

        validated = run_validation_stage(
            tmp_path, judge=ManualFileJudge(judge_path), theta=0.5)

        # Verbatim-mm canonical q* with image-deictic references are
        # excluded by the grounding checks (rows 2 and 4: "these
        # helmets", "these visible factors")
        excluded = read_jsonl(tmp_path / "excluded_families.jsonl")
        assert sorted(e["source_id"] for e in excluded) == \
            ["mtmcs:type_b:000002", "mtmcs:type_b:000004"]
        assert all(any("grounding" in r for r in e["reasons"])
                   for e in excluded)

        assert len(validated) == 3
        report = json.loads(
            (tmp_path / "validation_report.json").read_text())
        assert report["n_validated"] == 3
        assert report["n_excluded"] == 2
        # All three grounding-clean judged families satisfy the strict
        # criterion
        assert len(report["strict_causal_subset"]) == 3
        for family in validated:
            assert family.validation["strict_causal_candidate"] is True
            assert family.validation["standalone_terminal_risk"] == 0.1
            assert family.validation["requires_standalone_risk_validation"] \
                is False
            # safe_vs_unsafe axis auto-resolved on shared atoms
            shared = [a for a in family.semantic_atoms
                      if a.divergence != "causal" and a.surface_forms]
            assert shared
            for atom in shared:
                assert atom.semantic_equivalence[
                    "safe_vs_unsafe_shared_parts"]["state"] == "equivalent"

    def test_validate_without_judge_keeps_candidates_only(self, tmp_path):
        _build(tmp_path, fill_grounding=True)
        validated = run_validation_stage(tmp_path)
        assert len(validated) == 3
        report = json.loads(
            (tmp_path / "validation_report.json").read_text())
        assert report["judge"] is None
        assert report["strict_causal_subset"] == []
        for family in validated:
            assert family.validation["strict_causal_candidate"] is None
            assert family.validation["variant_generation"][
                "cross_modal_required"] is None
