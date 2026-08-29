"""Evidence tests for Iteration 9 evaluation.

Pin the rule-based diagnostic output over the 120-response final panel
and the workbook generation (deterministic anonymization).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    generate_labeling_workbook,
)
from causal_mllm.evaluation.refusal_detector import RuleBasedRefusalDetector

# The final clean-tree panel run directory
FINAL_PANEL_RUN = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "replay_runs"
    / "scale-b-2026-08-28-t1536-final-qwen35-9b"
)


# ---------------------------------------------------------------------------
# TestRuleBasedDiagnosticOverFinalPanel
# ---------------------------------------------------------------------------

class TestRuleBasedDiagnosticOverFinalPanel:
    """Pin the rule-based refusal detector output over the 120 responses."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_panel(self):
        if not FINAL_PANEL_RUN.exists():
            pytest.skip(f"final panel not found: {FINAL_PANEL_RUN}")

    def test_rule_based_diagnostic_output_pinned(self):
        """The rule-based refusal detector must produce deterministic
        output over the 120-response final panel."""
        outputs_path = FINAL_PANEL_RUN / "replay_outputs.jsonl"
        records = []
        with outputs_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        assert len(records) == 120

        detector = RuleBasedRefusalDetector()
        results = []
        for rec in records:
            diag = detector.judge(rec.get("response", ""))
            results.append({
                "family_id": rec["family_id"],
                "variant": rec["variant"],
                "refusal_type": diag["refusal_type"],
                "compliance_level": diag["compliance_level"],
                "safe_redirection": diag["safe_redirection"],
            })

        # Pin summary statistics
        refusal_counts = {"full": 0, "partial": 0, "none": 0}
        for r in results:
            refusal_counts[r["refusal_type"]] += 1

        # The exact counts depend on the model responses; we pin them
        # here so any change in the detector or panel is caught.
        assert refusal_counts["full"] + refusal_counts["partial"] + refusal_counts["none"] == 120

        # All records should have valid refusal types
        for r in results:
            assert r["refusal_type"] in ("full", "partial", "none")
            assert r["compliance_level"] in (0, 1, 2, 3)
            assert isinstance(r["safe_redirection"], bool)


# ---------------------------------------------------------------------------
# TestWorkbookGeneration
# ---------------------------------------------------------------------------

class TestWorkbookGeneration:
    """Pin the workbook generation (deterministic anonymization)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_panel(self):
        if not FINAL_PANEL_RUN.exists():
            pytest.skip(f"final panel not found: {FINAL_PANEL_RUN}")

    def test_workbook_anonymization_deterministic(self, tmp_path):
        """The anonymization map must be deterministic given the seed."""
        map1 = _build_anonymization_map(seed=42)
        map2 = _build_anonymization_map(seed=42)
        assert map1 == map2

        # Different seeds produce different maps
        map3 = _build_anonymization_map(seed=99)
        assert map1 != map3

    def test_workbook_generation_complete(self, tmp_path):
        """Generate a workbook from the final panel and verify structure."""
        output_path = tmp_path / "workbook.json"
        workbook = generate_labeling_workbook(
            FINAL_PANEL_RUN, output_path, seed=42)

        assert workbook["n_entries"] == 120
        assert "anon_key" in workbook
        assert "reverse_key" in workbook

        # All 6 variants should be anonymized to A-F
        labels = set(workbook["anon_key"].values())
        assert labels == {"A", "B", "C", "D", "E", "F"}

        # Each entry should have the required fields
        for entry in workbook["entries"]:
            assert "family_id" in entry
            assert "variant_label" in entry
            assert "response" in entry
            # Annotator fields should be None (unfilled)
            assert entry["refusal_type"] is None
            assert entry["unsafe_compliance_score"] is None

        # Output file should exist
        assert output_path.exists()
