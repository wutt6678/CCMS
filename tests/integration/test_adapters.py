"""Integration tests for source dataset adapters.

These tests require network access and are marked as 'integration'.
They verify that real source data loads and normalizes correctly.
"""

import pytest

from causal_mllm.data.schemas import CanonicalSourceExample
from causal_mllm.data.validate_schema import validate_source_example


# ---------------------------------------------------------------------------
# MTMCS-Bench adapter tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSAdapter:
    """Integration tests for MTMCS-Bench adapter."""

    def test_load_and_normalize_5_examples(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        assert len(examples) == 5

    def test_source_id_retained(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert ex.source_dataset == "mtmcs"
            assert ex.source_id is not None
            assert len(ex.source_id) > 0

    def test_message_order_retained(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            turn_indices = [m.turn_index for m in ex.messages]
            assert turn_indices == sorted(turn_indices), "Message order not preserved"

    def test_terminal_query_non_empty(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert len(ex.terminal_query.strip()) > 0

    def test_image_references_resolve(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
        for ex in examples:
            assert ex.has_images, f"No images found for MTMCS id={ex.source_id}"

    def test_schema_validation(self):
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter()
        examples = adapter.load_and_normalize(max_examples=5)
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
