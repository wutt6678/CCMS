"""Iteration 2: Golden fixture tests and schema-change guards.

These tests pin the output of adapters against real source data.
Any adapter change that alters normalization output will fail here.

Golden fixtures are stored in tests/fixtures/ and checked into git.
Schema guards define expected column sets per source dataset.
"""

import json
from pathlib import Path

import pytest

from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CanonicalSourceExample
from causal_mllm.data.validate_schema import validate_canonical_example
from causal_mllm.seeds import sha256_text

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _load_golden(name: str) -> list[dict]:
    """Load a golden fixture JSONL file."""
    return read_jsonl(FIXTURES_DIR / f"{name}_golden.jsonl")


def _load_schema_guards() -> dict:
    """Load the schema guards configuration."""
    with open(FIXTURES_DIR / "schema_guards.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Golden fixture structural invariants
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSGoldenFixtures:
    """Verify MTMCS golden fixtures match adapter output."""

    def test_fixtures_exist(self):
        fixtures = _load_golden("mtmcs")
        assert len(fixtures) == 28  # 5 type_b rows × 4 + 2 type_a rows × 4

    def test_fixtures_pass_canonical_validation(self):
        """Every golden fixture must pass the enhanced canonical validator."""
        fixtures = _load_golden("mtmcs")
        for rec in fixtures:
            example = CanonicalSourceExample.from_dict(rec)
            errors = validate_canonical_example(example)
            assert errors == [], f"{rec['source_id']}: {errors}"

    def test_type_b_terminal_query_shared(self):
        """Type B golden fixtures: terminal query must be shared across safe/unsafe."""
        fixtures = _load_golden("mtmcs")
        # Group by pair_id
        groups: dict[str, list[dict]] = {}
        for rec in fixtures:
            pid = rec["metadata"]["pair_id"]
            groups.setdefault(pid, []).append(rec)

        for pid, records in groups.items():
            if "type_b" not in pid:
                continue
            # All records in the group must have the same terminal_query_hash
            mm_hashes = set()
            for r in records:
                if r["metadata"]["modality"] == "multimodal":
                    mm_hashes.add(r["terminal_query_hash"])
            assert len(mm_hashes) == 1, (
                f"{pid}: type_b multimodal terminal hashes differ: {mm_hashes}"
            )

    def test_type_a_terminal_query_differs(self):
        """Type A golden fixtures: safe/unsafe terminal queries must differ."""
        fixtures = _load_golden("mtmcs")
        groups: dict[str, list[dict]] = {}
        for rec in fixtures:
            pid = rec["metadata"]["pair_id"]
            groups.setdefault(pid, []).append(rec)

        for pid, records in groups.items():
            if "type_a" not in pid:
                continue
            mm_safe_hashes = set()
            mm_unsafe_hashes = set()
            for r in records:
                if r["metadata"]["modality"] == "multimodal":
                    if r["metadata"]["safety"] == "safe":
                        mm_safe_hashes.add(r["terminal_query_hash"])
                    else:
                        mm_unsafe_hashes.add(r["terminal_query_hash"])
            assert mm_safe_hashes != mm_unsafe_hashes, (
                f"{pid}: type_a safe/unsafe terminal hashes unexpectedly identical"
            )

    def test_adapter_output_matches_golden(self):
        """Re-normalize source rows and compare structural properties to golden."""
        from causal_mllm.adapters.mtmcs import MTMCSAdapter

        adapter = MTMCSAdapter()
        golden = _load_golden("mtmcs")

        # Re-normalize the first type_b row
        type_b_groups = adapter.load_and_normalize_grouped(
            split="type_b", max_rows=1
        )
        assert len(type_b_groups) == 1
        fresh_records = [r.to_dict() for r in type_b_groups[0]]

        # Find matching golden records (first 4 = first type_b row)
        golden_first_row = [
            r for r in golden
            if r["metadata"]["pair_id"] == fresh_records[0]["metadata"]["pair_id"]
        ]
        assert len(golden_first_row) == 4

        for fresh, gold in zip(
            sorted(fresh_records, key=lambda r: r["source_id"]),
            sorted(golden_first_row, key=lambda r: r["source_id"]),
        ):
            assert fresh["source_id"] == gold["source_id"]
            assert fresh["source_split"] == gold["source_split"]
            assert fresh["source_setting"] == gold["source_setting"]
            assert fresh["label"] == gold["label"]
            assert len(fresh["messages"]) == len(gold["messages"])
            # Terminal query hash must match
            assert sha256_text(fresh["terminal_query"]) == gold["terminal_query_hash"]


@pytest.mark.integration
@pytest.mark.slow
class TestCoSafeGoldenFixtures:
    """Verify CoSafe golden fixtures."""

    def test_fixtures_exist(self):
        fixtures = _load_golden("cosafe")
        assert len(fixtures) == 5

    def test_fixtures_pass_canonical_validation(self):
        fixtures = _load_golden("cosafe")
        for rec in fixtures:
            example = CanonicalSourceExample.from_dict(rec)
            errors = validate_canonical_example(example)
            assert errors == [], f"{rec['source_id']}: {errors}"

    def test_all_records_have_user_and_assistant(self):
        fixtures = _load_golden("cosafe")
        for rec in fixtures:
            roles = {m["role"] for m in rec["messages"]}
            assert "user" in roles, f"{rec['source_id']}: no user role"
            assert "assistant" in roles, f"{rec['source_id']}: no assistant role"

    def test_no_images(self):
        fixtures = _load_golden("cosafe")
        for rec in fixtures:
            for msg in rec["messages"]:
                assert msg["images"] == [], (
                    f"{rec['source_id']}: unexpected image in CoSafe record"
                )


@pytest.mark.integration
@pytest.mark.slow
class TestMTIDGoldenFixtures:
    """Verify MTID golden fixtures."""

    def test_fixtures_exist(self):
        fixtures = _load_golden("mtid")
        assert len(fixtures) == 5

    def test_fixtures_pass_canonical_validation(self):
        fixtures = _load_golden("mtid")
        for rec in fixtures:
            example = CanonicalSourceExample.from_dict(rec)
            errors = validate_canonical_example(example)
            assert errors == [], f"{rec['source_id']}: {errors}"

    def test_labels_are_safe_or_unsafe(self):
        fixtures = _load_golden("mtid")
        for rec in fixtures:
            assert rec["label"] in ("safe", "unsafe"), (
                f"{rec['source_id']}: unexpected label '{rec['label']}'"
            )

    def test_metadata_has_target_turn(self):
        fixtures = _load_golden("mtid")
        for rec in fixtures:
            assert "target_turn" in rec["metadata"], (
                f"{rec['source_id']}: missing target_turn in metadata"
            )
            assert "meta_intent" in rec["metadata"], (
                f"{rec['source_id']}: missing meta_intent in metadata"
            )


# ---------------------------------------------------------------------------
# Schema-change guards
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSchemaGuards:
    """Detect source dataset schema changes.

    If a source dataset adds/removes columns or changes structure,
    these tests will fail loudly, preventing silent data loss.
    """

    def test_mtmcs_columns_unchanged(self):
        """MTMCS HF dataset columns must match the pinned schema."""
        from datasets import load_dataset

        guards = _load_schema_guards()
        expected = set(guards["mtmcs"]["huggingface_columns"])

        for split in guards["mtmcs"]["splits"]:
            ds = load_dataset("ND-25/MCS-bench", split=split)
            actual = set(ds.column_names)
            assert actual == expected, (
                f"MTMCS {split} columns changed!\n"
                f"  Expected: {sorted(expected)}\n"
                f"  Actual:   {sorted(actual)}\n"
                f"  New:      {sorted(actual - expected)}\n"
                f"  Missing:  {sorted(expected - actual)}"
            )

    def test_mtmcs_dialogue_keys_unchanged(self):
        """MTMCS dialogue dict keys must match the pinned schema."""
        from datasets import load_dataset

        guards = _load_schema_guards()
        expected = set(guards["mtmcs"]["dialogue_keys"])

        ds = load_dataset("ND-25/MCS-bench", split="type_a")
        row = ds[0]
        for dlg_field in ("multimodal_dialogue", "unimodal_dialogue"):
            actual = set(row[dlg_field].keys())
            assert actual == expected, (
                f"MTMCS {dlg_field} keys changed!\n"
                f"  Expected: {sorted(expected)}\n"
                f"  Actual:   {sorted(actual)}"
            )

    def test_mtmcs_row_count_stable(self):
        """MTMCS row count per split must not silently change."""
        from datasets import load_dataset

        guards = _load_schema_guards()
        expected = guards["mtmcs"]["rows_per_split"]

        for split in guards["mtmcs"]["splits"]:
            ds = load_dataset("ND-25/MCS-bench", split=split)
            assert len(ds) == expected, (
                f"MTMCS {split} row count changed: "
                f"expected {expected}, got {len(ds)}"
            )

    def test_cosafe_message_keys_unchanged(self):
        """CoSafe message dict keys must match the pinned schema."""
        guards = _load_schema_guards()
        expected = set(guards["cosafe"]["message_keys"])

        cosafe_dir = Path("data/raw/cosafe/CoSafe-Dataset/CoSafe datasets")
        if not cosafe_dir.exists():
            pytest.skip("CoSafe data not available")

        first_file = sorted(cosafe_dir.glob("*.json"))[0]
        with first_file.open() as f:
            first_line = json.loads(f.readline())

        assert isinstance(first_line, list) and len(first_line) > 0
        actual = set(first_line[0].keys())
        assert actual == expected, (
            f"CoSafe message keys changed!\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Actual:   {sorted(actual)}"
        )

    def test_mtid_keys_unchanged(self):
        """MTID JSONL keys must match the pinned schema."""
        from huggingface_hub import hf_hub_download

        guards = _load_schema_guards()
        expected_harmful = set(guards["mtid"]["harmful_keys"])
        expected_benign = set(guards["mtid"]["benign_keys"])

        # Check harmful test
        path = hf_hub_download("Graph-COM/MTID", "harmful_test.jsonl",
                               repo_type="dataset")
        with open(path) as f:
            first_record = json.loads(f.readline())
        actual_harmful = set(first_record.keys())
        assert actual_harmful == expected_harmful, (
            f"MTID harmful keys changed!\n"
            f"  Expected: {sorted(expected_harmful)}\n"
            f"  Actual:   {sorted(actual_harmful)}"
        )

        # Check benign test
        path = hf_hub_download("Graph-COM/MTID", "benign_test.jsonl",
                               repo_type="dataset")
        with open(path) as f:
            first_record = json.loads(f.readline())
        actual_benign = set(first_record.keys())
        assert actual_benign == expected_benign, (
            f"MTID benign keys changed!\n"
            f"  Expected: {sorted(expected_benign)}\n"
            f"  Actual:   {sorted(actual_benign)}"
        )


# ---------------------------------------------------------------------------
# Enhanced canonical validator tests
# ---------------------------------------------------------------------------

class TestCanonicalValidator:
    """Unit tests for validate_canonical_example()."""

    def test_valid_mtmcs_record(self):
        """A valid MTMCS record must pass with no errors."""
        fixtures = _load_golden("mtmcs")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        errors = validate_canonical_example(example)
        assert errors == []

    def test_valid_cosafe_record(self):
        """A valid CoSafe record must pass with no errors."""
        fixtures = _load_golden("cosafe")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        errors = validate_canonical_example(example)
        assert errors == []

    def test_valid_mtid_record(self):
        """A valid MTID record must pass with no errors."""
        fixtures = _load_golden("mtid")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        errors = validate_canonical_example(example)
        assert errors == []

    def test_detects_empty_source_id(self):
        """Empty source_id must be caught."""
        example = CanonicalSourceExample(
            source_dataset="mtmcs",
            source_id="",
            source_split="type_b",
            source_category=None,
            source_setting="type_b",
            label="safe",
            messages=[],
            terminal_turn_index=0,
            terminal_query="test",
        )
        errors = validate_canonical_example(example)
        assert any("source_id" in e for e in errors)

    def test_detects_invalid_label(self):
        """Invalid label must be caught."""
        fixtures = _load_golden("mtmcs")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        example.label = "INVALID"
        errors = validate_canonical_example(example)
        assert any("label" in e for e in errors)

    def test_detects_mtmcs_non_user_turn(self):
        """MTMCS record with assistant turn must be caught."""
        from causal_mllm.data.schemas import Message
        fixtures = _load_golden("mtmcs")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        # Corrupt one message role
        example.messages[0] = Message(
            turn_index=0, role="assistant", text="oops", images=[]
        )
        errors = validate_canonical_example(example)
        assert any("user turns" in e for e in errors)

    def test_detects_terminal_mismatch(self):
        """Terminal query not matching last user message must be caught."""
        fixtures = _load_golden("cosafe")
        example = CanonicalSourceExample.from_dict(fixtures[0])
        example.terminal_query = "WRONG TERMINAL QUERY"
        errors = validate_canonical_example(example)
        assert any("terminal_query" in e for e in errors)
