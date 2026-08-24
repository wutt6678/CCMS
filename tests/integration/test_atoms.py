"""Iteration 4 integration tests: comparative atom extraction on real data.

Covers:
  * Family skeletons built from freshly normalized real MTMCS rows
  * The comparative invariant on real data: exactly one causal divergence
    per type_b family, at the opening turn, with q* shared
  * build_families --stage atoms CLI end-to-end
"""

import json
import subprocess
import sys

import pytest
import yaml

from causal_mllm.construction.families import build_family_skeletons
from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionResult,
    select_candidates,
)
from causal_mllm.data.io import read_jsonl
from causal_mllm.data.validate_schema import validate_family_skeleton


@pytest.mark.integration
@pytest.mark.slow
class TestAtomsOnRealMTMCS:
    def _load_type_b_candidates(self, max_rows: int = 3):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter

        adapter = MTMCSAdapter()
        records = adapter.load_and_normalize(split="type_b", max_rows=max_rows)
        config = SelectionConfig(settings=frozenset({"type_b"}),
                                 max_text_length=100_000)
        accepted, rejections = select_candidates(records, config)
        assert rejections == []
        return accepted

    def test_real_type_b_families_decompose_comparatively(self):
        accepted = self._load_type_b_candidates(max_rows=3)
        result = SelectionResult(accepted=accepted)
        from causal_mllm.construction.select import group_into_family_units
        units = group_into_family_units(result.accepted)
        assert len(units) == 3

        skeletons = build_family_skeletons(units, seed=42)
        for skeleton in skeletons:
            errors = validate_family_skeleton(skeleton.to_dict())
            assert errors == [], f"{skeleton.family_id}: {errors}"

            # Comparative core: exactly one causal atom, at the opening turn
            causal = [a for a in skeleton.semantic_atoms
                      if a.divergence == "causal"]
            assert len(causal) == 1, skeleton.family_id
            assert causal[0].source_turns == [0]
            assert causal[0].safe_text != causal[0].unsafe_text
            # Four-condition surface forms present and consistent
            forms = causal[0].surface_forms
            assert set(forms) == {
                "multimodal_safe", "multimodal_unsafe",
                "unimodal_safe", "unimodal_unsafe",
            }
            assert forms["multimodal_safe"]["text"] == causal[0].safe_text
            # Structure known, meaning pending annotation
            assert causal[0].structural_role == "divergent_history_turn"
            assert causal[0].semantic_type == "unknown"

            # q* shared within conditions; cross-modality alignment measured
            assert skeleton.ground_truth["shared_terminal_query"] is True
            assert skeleton.ground_truth["divergent_turns"] == [0]
            ta = skeleton.ground_truth["terminal_alignment"]
            assert ta["mm_safe_vs_mm_unsafe"] is True
            assert ta["text_safe_vs_text_unsafe"] is True
            assert "multimodal_vs_unimodal" in ta
            assert skeleton.ground_truth["requires_terminal_harmonization"] \
                == (not ta["multimodal_vs_unimodal"])

            # Vision atom present, shared, with hashed media references
            vision = [a for a in skeleton.semantic_atoms
                      if "vision" in a.source_modalities]
            assert vision and all(a.divergence == "shared" for a in vision)
            for atom in vision:
                assert atom.source_media, "vision atom missing source_media"
                for ref in atom.source_media:
                    assert ref["path"]
                    # Real adapter-saved media must be hashable
                    assert ref["sha256"], f"missing hash for {ref['path']}"

    def test_family_ids_deterministic_across_runs(self):
        accepted = self._load_type_b_candidates(max_rows=2)
        from causal_mllm.construction.select import group_into_family_units
        units1 = group_into_family_units(accepted)
        units2 = group_into_family_units(list(accepted))
        ids1 = [s.family_id for s in build_family_skeletons(units1, seed=42)]
        ids2 = [s.family_id for s in build_family_skeletons(units2, seed=42)]
        assert ids1 == ids2

    def test_atoms_stage_writes_artifacts(self, tmp_path):
        from causal_mllm.construction.pipeline import (
            run_atoms_stage,
            run_selection_stage,
        )

        config = {
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 3},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
            "seed": 42,
        }
        selection_result = run_selection_stage(config, tmp_path)
        skeletons = run_atoms_stage(selection_result, tmp_path, seed=42)

        skeleton_records = read_jsonl(tmp_path / "family_skeletons.jsonl")
        atoms_report = json.loads((tmp_path / "atoms_report.json").read_text())

        assert len(skeleton_records) == len(skeletons) == 3
        for rec in skeleton_records:
            assert validate_family_skeleton(rec) == []
        assert atoms_report["n_families"] == 3
        assert atoms_report["n_causal_atoms"] == 3  # one per type_b family
        assert atoms_report["extraction_backend"] == "rule"


# ---------------------------------------------------------------------------
# Type-B cross-modality diagnostics on real data
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestTypeBDiagnosticsOnRealData:
    def test_diagnostic_structure_on_real_rows(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        from causal_mllm.construction.diagnostics import diagnose_type_b_rows

        adapter = MTMCSAdapter()
        rows = []
        for row in adapter.load("type_b"):
            rows.append(row)
            if len(rows) >= 20:
                break

        report = diagnose_type_b_rows(rows)
        assert report["n_type_b"] == 20
        assert report["n_rows_complete"] == 20
        term = report["terminal_query_cross_modality"]
        assert 0.0 <= term["fraction_exact_match"] <= 1.0
        assert 0.0 <= term["fraction_normalized_match"] <= 1.0
        assert set(report["per_turn_alignment"]) == {
            "safe_r1", "unsafe_r1", "r2", "r3",
        }
        usable = report["directly_usable"]
        assert (usable["n_exact"] + usable["n_requiring_rewrite_exact"]
                == report["n_rows_complete"])

    def test_real_type_b_requires_terminal_harmonization(self):
        """Real-data finding: mm/text dialogues are separately written.

        The 752-row diagnostic (outputs/diagnostics/type_b_alignment.json)
        found 0% cross-modality terminal equality; every extracted family
        must therefore carry requires_terminal_harmonization=True.
        """
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        from causal_mllm.construction.atoms import extract_family_atoms
        from causal_mllm.construction.select import group_into_family_units

        adapter = MTMCSAdapter()
        records = adapter.load_and_normalize(split="type_b", max_rows=3)
        for _, unit_records in group_into_family_units(records):
            extraction = extract_family_atoms("CMST_diag", unit_records)
            assert extraction.terminal_alignment["mm_safe_vs_mm_unsafe"]
            assert extraction.terminal_alignment["text_safe_vs_text_unsafe"]
            assert extraction.requires_terminal_harmonization is True


# ---------------------------------------------------------------------------
# CLI end-to-end: --stage atoms runs select + comparative extraction
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestBuildFamiliesAtomsCli:
    def test_cli_stage_atoms_end_to_end(self, tmp_path):
        config = {
            "experiment_name": "iter4_cli_test",
            "seed": 42,
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 2},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "-m", "causal_mllm.cli.build_families",
             "--config", str(config_path),
             "--stage", "atoms",
             "--output-dir", str(output_dir)],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # Selection artifacts still produced
        assert (output_dir / "candidates.jsonl").exists()
        assert (output_dir / "family_review_flags.jsonl").exists()
        # Atoms artifacts
        assert (output_dir / "family_skeletons.jsonl").exists()
        assert (output_dir / "atoms_report.json").exists()

        skeletons = read_jsonl(output_dir / "family_skeletons.jsonl")
        assert len(skeletons) == 2
        for rec in skeletons:
            assert validate_family_skeleton(rec) == []
            causal = [a for a in rec["semantic_atoms"]
                      if a["divergence"] == "causal"]
            assert len(causal) == 1
            # Risk-validation placeholders survive into the skeleton
            assert rec["validation"]["requires_standalone_risk_validation"] is True
            assert rec["validation"]["standalone_terminal_risk"] is None
