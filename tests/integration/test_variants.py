"""Iteration 5 integration tests: Scale-A smoke build on real MTMCS data.

Acceptance gate coverage:
  * annotation application integrated into the pipeline
  * annotation provenance persisted
  * unresolved required semantics fail loudly (gates)
  * canonical q* harmonizer implemented; original mm/text q* retained
  * exact canonical-q hash invariant across all six variants
  * all six variants generated independently with provenance
  * source skeletons immutable
  * 5-family real MTMCS smoke build -> 30 trajectories passing checks

NOTE: the annotations and harmonizations used here are PROGRAMMATIC
PLACEHOLDERS produced by test backends. The production route for the
smoke sets is human review (ManualFileAnnotator / ManualHarmonizer) or
a real LLM/VLM backend via CallableAnnotator / CallableHarmonizer.
"""

import json
import subprocess
import sys

import pytest
import yaml

from causal_mllm.construction.annotation import CallableAnnotator
from causal_mllm.construction.harmonize import (
    CallableHarmonizer,
    apply_terminal_harmonization,
    canonical_terminal,
)
from causal_mllm.construction.pipeline import (
    run_annotation_stage,
    run_atoms_stage,
    run_harmonization_stage,
    run_selection_stage,
    run_variants_stage,
)
from causal_mllm.construction.readiness import VariantPrerequisiteError
from causal_mllm.construction.variants import (
    VARIANT_GENERATORS,
    validate_variant_trajectory,
)
from causal_mllm.data.io import read_jsonl
from causal_mllm.seeds import sha256_text

ALL_VARIANTS = ("neutral", "text_only", "vision_only",
                "cross_modal", "shuffle", "history_reset")


def _resolve_all_annotations(family_key: str, atom: dict):
    """Test annotation backend resolving everything generators need."""
    payload = None
    if atom["divergence"] == "causal":
        payload = {"semantic_type": "relation",
                   "semantic_description": "opening-turn framing divergence"}
    elif atom.get("structural_role") == "shared_image":
        payload = {"semantic_type": "entity_or_scene",
                   "risk_relevance": "relevant",
                   "required_for_joint_interpretation": True}
    forms = atom.get("surface_forms") or {}
    if any(k.startswith("multimodal_") for k in forms) \
            and any(k.startswith("unimodal_") for k in forms):
        payload = dict(payload or {"semantic_type": "reference"})
        payload["semantic_equivalence"] = {
            "multimodal_vs_unimodal":
                {"state": "equivalent", "confidence": 0.9},
        }
    return payload


def _canonical_from_mm(family_key: str, source_mm_q: str, source_text_q):
    """Placeholder harmonization: adopt the multimodal form as canonical.

    Real harmonization is a human/LLM decision; this keeps the smoke
    build deterministic without claiming any linguistic judgment.
    """
    return source_mm_q


@pytest.mark.integration
@pytest.mark.slow
class TestScaleASmokeBuild:
    """5 real Type-B families -> 30 trajectories (Scale A)."""

    def _run_pipeline(self, tmp_path):
        config = {
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 5},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
            "seed": 42,
        }
        selection_result = run_selection_stage(config, tmp_path)
        run_atoms_stage(selection_result, tmp_path, seed=42)
        annotator = CallableAnnotator(_resolve_all_annotations,
                                      model_name="test-vlm",
                                      prompt_version="v1")
        run_annotation_stage(annotator, tmp_path)
        harmonizer = CallableHarmonizer(_canonical_from_mm,
                                        model_name="test-llm",
                                        prompt_version="v1")
        run_harmonization_stage(harmonizer, tmp_path)
        complete = run_variants_stage(tmp_path, seed=42)
        return complete, tmp_path

    def test_five_families_produce_thirty_trajectories(self, tmp_path):
        complete, out = self._run_pipeline(tmp_path)
        assert len(complete) == 5
        assert sum(len(f.variants) for f in complete) == 30
        for family in complete:
            assert set(family.variants) == set(ALL_VARIANTS)

    def test_exact_canonical_q_hash_invariant_on_real_data(self, tmp_path):
        complete, _ = self._run_pipeline(tmp_path)
        for family in complete:
            q, sha = canonical_terminal(family)
            for name, variant in family.variants.items():
                last = variant.messages[-1]
                assert last.text == q, (family.family_id, name)
                assert sha256_text(last.text) == sha, (family.family_id, name)
                assert validate_variant_trajectory(family, variant) == [], \
                    (family.family_id, name)

    def test_real_families_required_harmonization(self, tmp_path):
        """The 752-row diagnostic says 0% cross-modality terminal match;
        every real family's harmonization block must say required=True."""
        complete, _ = self._run_pipeline(tmp_path)
        for family in complete:
            block = family.validation["terminal_harmonization"]
            assert block["required"] is True, family.family_id
            assert block["source_mm_q"] and block["source_text_q"]
            assert block["source_mm_q"] != block["source_text_q"]
            assert block["method"] == "llm"
            # Original skeleton terminal retained untouched
            assert family.terminal_query.text == block["source_mm_q"]

    def test_annotation_provenance_persisted(self, tmp_path):
        complete, out = self._run_pipeline(tmp_path)
        # Check the on-disk artifact, not just the in-memory objects
        records = read_jsonl(out / "annotated_skeletons.jsonl")
        assert len(records) == 5
        seen_llm_prov = False
        for rec in records:
            for atom in rec["semantic_atoms"]:
                if atom["semantic_validation"] == "llm":
                    prov = atom["annotation_provenance"]
                    assert prov["backend"] == "llm"
                    assert prov["model"] == "test-vlm"
                    seen_llm_prov = True
        assert seen_llm_prov

    def test_unresolved_semantics_fail_loudly_on_real_data(self, tmp_path):
        """Skipping annotation must block semantic variants, loudly."""
        config = {
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 1},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
            "seed": 42,
        }
        selection_result = run_selection_stage(config, tmp_path)
        skeletons = run_atoms_stage(selection_result, tmp_path, seed=42)
        harmonizer = CallableHarmonizer(_canonical_from_mm,
                                        model_name="test-llm")
        harmonized = [apply_terminal_harmonization(s, harmonizer)
                      for s in skeletons]

        family = harmonized[0]
        # Image-bearing variants need equivalence/relevance evidence
        for name in ("vision_only", "cross_modal", "shuffle"):
            with pytest.raises(VariantPrerequisiteError):
                VARIANT_GENERATORS[name](family)
        # Text-only conditions never cross modalities: constructible
        for name in ("neutral", "text_only", "history_reset"):
            assert VARIANT_GENERATORS[name](family).messages

    def test_negative_control_routing_on_decided_negative_annotation(
            self, tmp_path):
        """A family annotated irrelevant is routed to negative controls,
        not built — the causal subset stays clean."""

        def irrelevant_annotator(family_key: str, atom: dict):
            payload = _resolve_all_annotations(family_key, atom)
            if atom.get("structural_role") == "shared_image":
                payload = dict(payload)
                payload["risk_relevance"] = "irrelevant"
            return payload

        config = {
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 1},
            "selection": {"settings": ["type_b"],
                          "max_text_length": 100_000},
            "seed": 42,
        }
        selection_result = run_selection_stage(config, tmp_path)
        run_atoms_stage(selection_result, tmp_path, seed=42)
        run_annotation_stage(
            CallableAnnotator(irrelevant_annotator, model_name="test-vlm"),
            tmp_path)
        run_harmonization_stage(
            CallableHarmonizer(_canonical_from_mm, model_name="test-llm"),
            tmp_path)
        complete = run_variants_stage(tmp_path, seed=42)
        assert complete == []
        controls = read_jsonl(tmp_path / "negative_controls.jsonl")
        assert len(controls) == 1
        assert any("annotated irrelevant" in r for r in controls[0]["reasons"])
        report = json.loads((tmp_path / "variants_report.json").read_text())
        assert report["n_families"] == 0
        assert report["n_negative_controls"] == 1

    def test_stage_artifacts_and_reports(self, tmp_path):
        complete, out = self._run_pipeline(tmp_path)
        for artifact in ("candidates.jsonl", "family_review_flags.jsonl",
                         "family_skeletons.jsonl", "annotation_report.json",
                         "annotated_skeletons.jsonl",
                         "harmonized_families.jsonl", "families.jsonl",
                         "variants_report.json"):
            assert (out / artifact).exists(), artifact

        report = json.loads((out / "variants_report.json").read_text())
        assert report["n_families"] == 5
        assert report["n_trajectories"] == 30
        assert report["cross_modal_status"] == "candidate"

        ann_report = json.loads((out / "annotation_report.json").read_text())
        assert ann_report["n_families"] == 5
        assert ann_report["n_l1_semantic"] == 5


# ---------------------------------------------------------------------------
# CLI end-to-end: full chain with manual annotation/harmonization files
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestBuildFamiliesFullChainCli:
    def test_cli_select_to_variants(self, tmp_path):
        config = {
            "experiment_name": "iter5_cli_test",
            "seed": 42,
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 2},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config, f)
        output_dir = tmp_path / "out"

        def run_cli(*extra):
            return subprocess.run(
                [sys.executable, "-m", "causal_mllm.cli.build_families",
                 "--config", str(config_path),
                 "--output-dir", str(output_dir), *extra],
                capture_output=True, text=True, timeout=900,
            )

        # Stage 1: build skeletons
        result = run_cli("--stage", "atoms")
        assert result.returncode == 0, result.stderr

        # Derive manual-style annotation + harmonization files from the
        # skeletons (placeholders standing in for human review)
        skeletons = read_jsonl(output_dir / "family_skeletons.jsonl")
        annotations, harmonization = {}, {}
        for rec in skeletons:
            fkey = rec["source"]["source_id"]
            annotations[fkey] = {}
            for atom in rec["semantic_atoms"]:
                payload = None
                if atom["divergence"] == "causal":
                    payload = {"semantic_type": "relation"}
                elif atom.get("structural_role") == "shared_image":
                    payload = {"semantic_type": "entity_or_scene",
                               "risk_relevance": "relevant",
                               "required_for_joint_interpretation": True}
                forms = atom.get("surface_forms") or {}
                if any(k.startswith("multimodal_") for k in forms) \
                        and any(k.startswith("unimodal_") for k in forms):
                    payload = dict(payload or {"semantic_type": "reference"})
                    payload["semantic_equivalence"] = {
                        "multimodal_vs_unimodal": "equivalent"}
                if payload:
                    annotations[fkey][atom["atom_id"]] = payload
            harmonization[fkey] = rec["terminal_query"]["text"]

        ann_path = tmp_path / "annotations.json"
        har_path = tmp_path / "harmonization.json"
        ann_path.write_text(json.dumps(annotations))
        har_path.write_text(json.dumps(harmonization))

        # Stage 2: annotate -> harmonize -> variants
        result = run_cli("--stage", "variants",
                         "--annotations", str(ann_path),
                         "--harmonization", str(har_path))
        assert result.returncode == 0, result.stderr

        families = read_jsonl(output_dir / "families.jsonl")
        assert len(families) == 2
        for rec in families:
            assert set(rec["variants"]) == set(ALL_VARIANTS)
            block = rec["validation"]["terminal_harmonization"]
            canonical_sha = block["canonical_sha256"]
            for variant in rec["variants"].values():
                last = variant["messages"][-1]
                assert sha256_text(last["text"]) == canonical_sha

    def test_cli_variants_without_inputs_fails_clearly(self, tmp_path):
        config = {
            "experiment_name": "iter5_cli_missing",
            "seed": 42,
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 1},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config, f)
        result = subprocess.run(
            [sys.executable, "-m", "causal_mllm.cli.build_families",
             "--config", str(config_path),
             "--stage", "annotate",
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, timeout=900,
        )
        assert result.returncode != 0
        assert "--annotations" in (result.stderr + result.stdout)
