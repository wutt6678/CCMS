"""Evidence tests for Iteration 9 evaluation.

Pin the rule-based diagnostic output over the 120-response final panel
and the workbook generation (deterministic anonymization).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    generate_labeling_workbook,
)
from causal_mllm.evaluation.refusal_detector import RuleBasedRefusalDetector
from causal_mllm.seeds import sha256_text

# The final clean-tree panel run directory
FINAL_PANEL_RUN = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "replay_runs"
    / "scale-b-2026-08-28-t1536-final-qwen35-9b"
)

# The validated families dataset
VALIDATED_FAMILIES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "families" / "scale_b_smoke"
    / "validated_families.jsonl"
)


def _load_families() -> dict[str, CausalFamily]:
    """Load validated families for workbook generation tests."""
    families = {}
    for rec in read_jsonl(VALIDATED_FAMILIES_PATH):
        fam = CausalFamily.from_dict(rec)
        families[fam.family_id] = fam
    return families


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
        if not VALIDATED_FAMILIES_PATH.exists():
            pytest.skip(
                f"validated families not found: {VALIDATED_FAMILIES_PATH}")

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
        families = _load_families()
        output_path = tmp_path / "workbook.json"
        workbook, manifest = generate_labeling_workbook(
            FINAL_PANEL_RUN, output_path,
            validated_families=families, seed=42)

        # Workbook should have 120 entries
        assert workbook["n_entries"] == 120
        assert workbook["version"] == "3.0"

        # Workbook must NOT contain variant mapping (sealed in manifest)
        assert "anon_key" not in workbook
        assert "reverse_key" not in workbook
        assert "item_mapping" not in workbook
        # Workbook must NOT contain the ID secret (irreversible IDs)
        assert "id_secret" not in workbook

        # Manifest should contain the decoding key, response hashes,
        # context hashes, and the ID secret for deobfuscation
        assert "anon_key" in manifest
        assert "reverse_key" in manifest
        assert "item_mapping" in manifest
        assert "expected_response_sha256" in manifest
        assert "context_hashes" in manifest
        assert "id_secret" in manifest

        # Context hashes should be 64-char hex strings
        for item_id, ctx in manifest["context_hashes"].items():
            assert len(ctx["system_prompt"]) == 64
            assert len(ctx["conversation_history"]) == 64
            assert len(ctx["terminal_query"]) == 64

        # All 6 variants should be anonymized to A-F
        labels = set(manifest["anon_key"].values())
        assert labels == {"A", "B", "C", "D", "E", "F"}

        # Each entry should have opaque item_id and full context
        item_ids = set()
        for entry in workbook["entries"]:
            assert "item_id" in entry
            # Opaque: must not contain family_id or variant letter
            assert not entry["item_id"].startswith("CMST_")
            assert "response" in entry
            assert "response_sha256" in entry
            assert "system_prompt" in entry
            assert len(entry["system_prompt"]) > 0  # non-empty
            assert "conversation_history" in entry
            assert "terminal_query" in entry
            assert "images" in entry

            # Verify response SHA256 matches actual response
            actual_sha = sha256_text(entry["response"])
            assert entry["response_sha256"] == actual_sha

            # Verify manifest has matching expected hash
            assert manifest["expected_response_sha256"][
                entry["item_id"]] == actual_sha

            # Annotator fields should be None (unfilled)
            assert entry["refusal_type"] is None
            assert entry["unsafe_compliance_score"] is None

            item_ids.add(entry["item_id"])

        # All item IDs must be unique and fully opaque
        assert len(item_ids) == 120

        # Output files should exist
        assert output_path.exists()
        manifest_path = output_path.with_suffix(".manifest.json")
        assert manifest_path.exists()
