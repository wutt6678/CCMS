"""Iteration 3 integration tests: candidate selection on real data.

Covers:
  * Selection over MTMCS golden fixtures (offline, pinned data)
  * Selection over freshly normalized real MTMCS rows
  * run_selection_stage artifact persistence
  * build_families --stage select CLI end-to-end
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from causal_mllm.construction.select import (
    SelectionConfig,
    run_selection,
    select_candidates,
)
from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CanonicalSourceExample

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _load_golden_examples() -> list[CanonicalSourceExample]:
    records = read_jsonl(FIXTURES_DIR / "mtmcs_golden.jsonl")
    return [CanonicalSourceExample.from_dict(r) for r in records]


# ---------------------------------------------------------------------------
# Selection over golden fixtures (offline)
# ---------------------------------------------------------------------------

class TestSelectionOnGoldenFixtures:
    def test_all_golden_groups_accepted_with_permissive_config(self):
        """Golden fixtures are curated: a permissive config accepts all."""
        examples = _load_golden_examples()
        config = SelectionConfig(require_images=True, max_text_length=100_000)
        accepted, rejections = select_candidates(examples, config)
        assert len(accepted) == len(examples) == 28
        assert rejections == []

    def test_accounting_holds(self):
        examples = _load_golden_examples()
        config = SelectionConfig(settings=frozenset({"type_b"}))
        accepted, rejections = select_candidates(examples, config)
        assert len(accepted) + len(rejections) == len(examples)

    def test_settings_filter_keeps_only_type_b(self):
        examples = _load_golden_examples()
        config = SelectionConfig(settings=frozenset({"type_b"}),
                                 max_text_length=100_000)
        accepted, rejections = select_candidates(examples, config)
        assert all(ex.source_setting == "type_b" for ex in accepted)
        assert all(r.reason == "setting_excluded" for r in rejections)
        # 5 type_b rows x 4 accepted; 2 type_a rows x 4 rejected
        assert len(accepted) == 20
        assert len(rejections) == 8

    def test_accepted_groups_are_complete(self):
        """Every accepted pair_id must retain all 4 conditions."""
        examples = _load_golden_examples()
        config = SelectionConfig(max_text_length=100_000)
        accepted, _ = select_candidates(examples, config)
        groups: dict[str, set[str]] = {}
        for ex in accepted:
            pid = ex.metadata["pair_id"]
            groups.setdefault(pid, set()).add(
                f"{ex.metadata['modality']}:{ex.metadata['safety']}"
            )
        for pid, conditions in groups.items():
            assert conditions == {
                "multimodal:safe", "multimodal:unsafe",
                "unimodal:safe", "unimodal:unsafe",
            }, f"{pid}: incomplete accepted group {conditions}"

    def test_accepted_records_stay_user_turns_only(self):
        """Pass-through: selection never injects assistant responses."""
        examples = _load_golden_examples()
        config = SelectionConfig(max_text_length=100_000)
        accepted, _ = select_candidates(examples, config)
        for ex in accepted:
            assert all(m.role == "user" for m in ex.messages)

    def test_report_structure(self):
        examples = _load_golden_examples()
        config = SelectionConfig(settings=frozenset({"type_b"}),
                                 max_text_length=100_000)
        result = run_selection(examples, config)
        report = result.report
        assert report["iteration"] == 3
        assert report["accounting_ok"] is True
        assert report["n_input"] == 28
        assert report["n_families_accepted"] == 5
        assert report["n_families_rejected"] == 2
        assert report["accepted_by_dataset"] == {"mtmcs": 20}
        # Record-level: 2 rejected rows x 4 records; family-level: 2 pairs
        assert report["rejected_records_by_reason"] == {"setting_excluded": 8}
        assert report["rejected_families_by_reason"] == {"setting_excluded": 2}
        # All golden MTMCS pairs are safe+unsafe mixed -> balanced by design
        assert report["families_by_safety"] == {"mixed": 5}
        assert all("safety" not in w for w in report["balance_warnings"])
        assert report["config_hash"] and report["timestamp"]


# ---------------------------------------------------------------------------
# Selection on freshly normalized real MTMCS rows
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSelectionOnRealMTMCS:
    def test_real_type_b_rows_selected_cleanly(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter

        adapter = MTMCSAdapter()
        records = adapter.load_and_normalize(split="type_b", max_rows=5)
        config = SelectionConfig(settings=frozenset({"type_b"}),
                                 max_text_length=100_000)
        accepted, rejections = select_candidates(records, config)
        # Real type_b data must satisfy the shared-terminal invariant
        assert len(accepted) == 20
        assert rejections == []

    def test_real_type_a_rows_selected_cleanly(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter

        adapter = MTMCSAdapter()
        records = adapter.load_and_normalize(split="type_a", max_rows=5)
        config = SelectionConfig(settings=frozenset({"type_a"}),
                                 max_text_length=100_000)
        accepted, rejections = select_candidates(records, config)
        assert len(accepted) == 20
        assert rejections == []

    def test_selection_stage_writes_artifacts(self, tmp_path):
        from causal_mllm.construction.pipeline import run_selection_stage

        config = {
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 3},
            "selection": {
                "settings": ["type_b"],
                "max_text_length": 100_000,
                "max_families": 2,
                "seed": 42,
            },
        }
        result = run_selection_stage(config, tmp_path)

        candidates = read_jsonl(tmp_path / "candidates.jsonl")
        sel_rejections = read_jsonl(tmp_path / "selection_rejections.jsonl")
        norm_rejections = read_jsonl(tmp_path / "normalization_rejections.jsonl")
        report = json.loads((tmp_path / "selection_report.json").read_text())

        # 3 rows x 4 = 12 input records; 2 families kept, 1 not sampled
        assert len(candidates) == 8
        assert len(sel_rejections) == 4
        assert all(r["reason"] == "not_sampled" for r in sel_rejections)
        assert norm_rejections == []
        assert report["n_input"] == 12
        assert report["n_families_accepted"] == 2
        assert len(result.accepted) == 8

        # Candidates roundtrip through the canonical schema
        for rec in candidates:
            CanonicalSourceExample.from_dict(rec)


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestBuildFamiliesSelectCli:
    def test_cli_stage_select_end_to_end(self, tmp_path):
        config = {
            "experiment_name": "iter3_cli_test",
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
             "--stage", "select",
             "--output-dir", str(output_dir)],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        assert (output_dir / "candidates.jsonl").exists()
        assert (output_dir / "selection_rejections.jsonl").exists()
        assert (output_dir / "normalization_rejections.jsonl").exists()
        assert (output_dir / "selection_report.json").exists()

        candidates = read_jsonl(output_dir / "candidates.jsonl")
        assert len(candidates) == 8  # 2 rows x 4, all eligible

        report = json.loads((output_dir / "selection_report.json").read_text())
        assert report["accounting_ok"] is True
        assert report["n_input"] == 8
        assert report["n_accepted"] == 8

    def test_cli_max_families_override(self, tmp_path):
        config = {
            "experiment_name": "iter3_cli_sample",
            "seed": 42,
            "source": {"dataset": "mtmcs", "split": "type_b", "max_rows": 4},
            "selection": {"settings": ["type_b"], "max_text_length": 100_000},
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "-m", "causal_mllm.cli.build_families",
             "--config", str(config_path),
             "--stage", "select",
             "--max-families", "2",
             "--output-dir", str(output_dir)],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, result.stderr

        candidates = read_jsonl(output_dir / "candidates.jsonl")
        assert len(candidates) == 8  # 2 families x 4 records
        pair_ids = {rec["metadata"]["pair_id"] for rec in candidates}
        assert len(pair_ids) == 2
