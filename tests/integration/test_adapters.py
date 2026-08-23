"""Integration tests for source dataset adapters.

These tests require network access and are marked as 'integration'.
They verify that real source data loads and normalizes correctly.

MTMCS semantic golden tests are in test_mtmcs_golden.py.
"""

import pytest

from causal_mllm.data.validate_schema import validate_source_example


# ---------------------------------------------------------------------------
# MTMCS-Bench adapter tests (basic — golden tests are separate)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSAdapter:
    """Basic integration tests for MTMCS-Bench adapter."""

    def test_load_and_normalize_produces_4_per_row(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(split="type_a", max_examples=8)
        # 2 rows × 4 records = 8
        assert len(examples) == 8

    def test_source_id_format(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(split="type_b", max_examples=4)
        for ex in examples:
            assert ex.source_id.startswith("mtmcs:type_b:")
            assert ex.source_id.endswith((":safe", ":unsafe"))

    def test_terminal_query_non_empty(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        for split in ("type_a", "type_b"):
            examples = adapter.load_and_normalize(split=split, max_examples=4)
            for ex in examples:
                assert len(ex.terminal_query.strip()) > 0

    def test_schema_validation(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(split="type_b", max_examples=4)
        for ex in examples:
            errors = validate_source_example(ex.to_dict())
            assert errors == [], f"Schema errors for {ex.source_id}: {errors}"


# ---------------------------------------------------------------------------
# CoSafe adapter tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestCoSafeAdapter:
    """Integration tests for CoSafe adapter."""

    def test_load_and_normalize_5_examples(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        assert len(examples) == 5

    def test_source_id_retained(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert ex.source_dataset == "cosafe"
            assert ex.source_id is not None

    def test_message_order_retained(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            turn_indices = [m.turn_index for m in ex.messages]
            assert turn_indices == sorted(turn_indices)

    def test_terminal_query_non_empty(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert len(ex.terminal_query.strip()) > 0

    def test_labels_survive_normalization(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert ex.label in ("safe", "unsafe", "unknown")

    def test_schema_validation(self):
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            errors = validate_source_example(ex.to_dict())
            assert errors == [], f"Schema errors for {ex.source_id}: {errors}"


# ---------------------------------------------------------------------------
# MTID adapter tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTIDAdapter:
    """Integration tests for MTID adapter."""

    def test_load_and_normalize_5_examples(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        assert len(examples) == 5

    def test_source_id_retained(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        for ex in examples:
            assert ex.source_dataset == "mtid"
            assert ex.source_id is not None

    def test_message_order_retained(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        for ex in examples:
            turn_indices = [m.turn_index for m in ex.messages]
            assert turn_indices == sorted(turn_indices)

    def test_terminal_query_non_empty(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        for ex in examples:
            assert len(ex.terminal_query.strip()) > 0

    def test_labels_survive_normalization(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        for ex in examples:
            assert ex.label in ("safe", "unsafe", "unknown")

    def test_schema_validation(self):
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter()
        examples = adapter.load_and_normalize(split="test", max_examples=5)
        for ex in examples:
            errors = validate_source_example(ex.to_dict())
            assert errors == [], f"Schema errors for {ex.source_id}: {errors}"
