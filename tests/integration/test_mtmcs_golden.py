"""Semantic golden tests for the MTMCS adapter.

These tests verify the *meaning* of the dialogue reconstruction, not
just syntactic validity. They would have caught the original P0 bug
immediately.

Verified against the upstream MTMCS inference code semantics:
  Type A:  unsafe = r1 → r2 → unsafe_r3
           safe   = r1 → r2 → safe_r3
  Type B:  unsafe = unsafe_r1 → r2 → r3
           safe   = safe_r1   → r2 → r3
"""

import pytest

from causal_mllm.adapters.mtmcs import MTMCSAdapter
from causal_mllm.data.media import MediaLoadError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_split_records(split: str, n: int = 5):
    """Load n raw HF rows for a given split."""
    from datasets import load_dataset
    ds = load_dataset("ND-25/MCS-bench", split=split)
    raws = []
    for i in range(min(n, len(ds))):
        row = dict(ds[i])
        row["_split"] = split
        raws.append(row)
    return raws


def _normalize_one(adapter: MTMCSAdapter, raw: dict) -> dict[str, any]:
    """Normalize one row and return a dict keyed by source_id."""
    records = adapter.normalize_row(raw)
    return {r.source_id: r for r in records}


# ---------------------------------------------------------------------------
# Type A golden tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSTypeAGolden:
    """Type A: divergence at the terminal turn.

    unsafe: user(r1) → user(r2) → user(unsafe_r3)
    safe:   user(r1) → user(r2) → user(safe_r3)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adapter = MTMCSAdapter()
        self.raws = _load_split_records("type_a", n=5)

    def test_type_a_unsafe_terminal_equals_unsafe_r3(self):
        """unsafe terminal query must be exactly raw.multimodal_dialogue.unsafe_r3."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_unsafe_id = f"mtmcs:type_a:{raw['id']:06d}:mm:unsafe"
            assert mm_unsafe_id in recs, f"Missing {mm_unsafe_id}"
            rec = recs[mm_unsafe_id]
            expected = raw["multimodal_dialogue"]["unsafe_r3"]
            assert rec.terminal_query == expected, (
                f"id={raw['id']}: type_a mm_unsafe terminal != unsafe_r3\n"
                f"  got:      {rec.terminal_query[:80]!r}\n"
                f"  expected: {expected[:80]!r}"
            )

    def test_type_a_safe_terminal_equals_safe_r3(self):
        """safe terminal query must be exactly raw.multimodal_dialogue.safe_r3."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe_id = f"mtmcs:type_a:{raw['id']:06d}:mm:safe"
            rec = recs[mm_safe_id]
            expected = raw["multimodal_dialogue"]["safe_r3"]
            assert rec.terminal_query == expected, (
                f"id={raw['id']}: type_a mm_safe terminal != safe_r3"
            )

    def test_type_a_terminal_queries_differ(self):
        """safe_r3 and unsafe_r3 should normally differ in type_a."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe_id = f"mtmcs:type_a:{raw['id']:06d}:mm:safe"
            mm_unsafe_id = f"mtmcs:type_a:{raw['id']:06d}:mm:unsafe"
            safe_q = recs[mm_safe_id].terminal_query
            unsafe_q = recs[mm_unsafe_id].terminal_query
            # They should normally differ (this is the nature of type_a)
            assert safe_q != unsafe_q, (
                f"id={raw['id']}: type_a safe/unsafe terminals are unexpectedly identical"
            )

    def test_type_a_shared_turns_are_identical(self):
        """r1 and r2 must be identical across safe and unsafe trajectories."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe = recs[f"mtmcs:type_a:{raw['id']:06d}:mm:safe"]
            mm_unsafe = recs[f"mtmcs:type_a:{raw['id']:06d}:mm:unsafe"]
            # First two turns should be identical (r1, r2)
            assert mm_safe.messages[0].text == mm_unsafe.messages[0].text, (
                f"id={raw['id']}: type_a r1 differs between safe/unsafe"
            )
            assert mm_safe.messages[1].text == mm_unsafe.messages[1].text, (
                f"id={raw['id']}: type_a r2 differs between safe/unsafe"
            )

    def test_type_a_all_messages_are_user_role(self):
        """MTMCS source data has NO assistant turns — all are user turns."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            for sid, rec in recs.items():
                for msg in rec.messages:
                    assert msg.role == "user", (
                        f"{sid}: message turn {msg.turn_index} has role={msg.role!r}, "
                        f"expected 'user' (MTMCS stores only user turns)"
                    )

    def test_type_a_exactly_three_turns(self):
        """Each type_a trajectory should have exactly 3 user turns."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            for sid, rec in recs.items():
                assert len(rec.messages) == 3, (
                    f"{sid}: expected 3 turns, got {len(rec.messages)}"
                )


# ---------------------------------------------------------------------------
# Type B golden tests — THE most critical for the causal experiment
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSTypeBGolden:
    """Type B: divergence at the opening turn.

    unsafe: user(unsafe_r1) → user(r2) → user(r3)
    safe:   user(safe_r1)   → user(r2) → user(r3)

    Terminal query r3 is IDENTICAL across safe/unsafe.
    This is the causal experiment's gold standard.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adapter = MTMCSAdapter()
        self.raws = _load_split_records("type_b", n=5)

    def test_type_b_unsafe_first_equals_unsafe_r1(self):
        """unsafe opening turn must be exactly raw.multimodal_dialogue.unsafe_r1."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_unsafe_id = f"mtmcs:type_b:{raw['id']:06d}:mm:unsafe"
            rec = recs[mm_unsafe_id]
            expected = raw["multimodal_dialogue"]["unsafe_r1"]
            assert rec.messages[0].text == expected, (
                f"id={raw['id']}: type_b mm_unsafe first turn != unsafe_r1\n"
                f"  got:      {rec.messages[0].text[:80]!r}\n"
                f"  expected: {expected[:80]!r}"
            )

    def test_type_b_safe_first_equals_safe_r1(self):
        """safe opening turn must be exactly raw.multimodal_dialogue.safe_r1."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe_id = f"mtmcs:type_b:{raw['id']:06d}:mm:safe"
            rec = recs[mm_safe_id]
            expected = raw["multimodal_dialogue"]["safe_r1"]
            assert rec.messages[0].text == expected, (
                f"id={raw['id']}: type_b mm_safe first turn != safe_r1"
            )

    def test_type_b_terminal_query_shared(self):
        """CRITICAL: type_b terminal query r3 must be IDENTICAL across safe/unsafe.

        This is the fundamental invariant for the causal experiment:
        (H_safe, q*) vs (H_unsafe, q*)
        """
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe_id = f"mtmcs:type_b:{raw['id']:06d}:mm:safe"
            mm_unsafe_id = f"mtmcs:type_b:{raw['id']:06d}:mm:unsafe"
            safe_terminal = recs[mm_safe_id].terminal_query
            unsafe_terminal = recs[mm_unsafe_id].terminal_query
            assert safe_terminal == unsafe_terminal, (
                f"id={raw['id']}: CRITICAL — type_b terminal query differs!\n"
                f"  safe:   {safe_terminal[:80]!r}\n"
                f"  unsafe: {unsafe_terminal[:80]!r}"
            )

    def test_type_b_terminal_equals_raw_r3(self):
        """Terminal queries must equal the r3 from the corresponding dialogue dict."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            # Multimodal records use multimodal_dialogue.r3;
            # unimodal records use unimodal_dialogue.r3
            for suffix, dlg_key in [
                ("mm:safe", "multimodal_dialogue"),
                ("mm:unsafe", "multimodal_dialogue"),
                ("text:safe", "unimodal_dialogue"),
                ("text:unsafe", "unimodal_dialogue"),
            ]:
                sid = f"mtmcs:type_b:{raw['id']:06d}:{suffix}"
                rec = recs[sid]
                expected_r3 = raw[dlg_key]["r3"]
                assert rec.terminal_query == expected_r3, (
                    f"{sid}: terminal != {dlg_key}.r3\n"
                    f"  got:      {rec.terminal_query[:80]!r}\n"
                    f"  expected: {expected_r3[:80]!r}"
                )

    def test_type_b_shared_turns_r2_identical(self):
        """r2 must be identical across safe and unsafe trajectories."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe = recs[f"mtmcs:type_b:{raw['id']:06d}:mm:safe"]
            mm_unsafe = recs[f"mtmcs:type_b:{raw['id']:06d}:mm:unsafe"]
            # Turn 1 (r2) should be identical
            assert mm_safe.messages[1].text == mm_unsafe.messages[1].text, (
                f"id={raw['id']}: type_b r2 differs between safe/unsafe"
            )

    def test_type_b_opening_turns_differ(self):
        """unsafe_r1 and safe_r1 should normally differ in type_b."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            mm_safe = recs[f"mtmcs:type_b:{raw['id']:06d}:mm:safe"]
            mm_unsafe = recs[f"mtmcs:type_b:{raw['id']:06d}:mm:unsafe"]
            assert mm_safe.messages[0].text != mm_unsafe.messages[0].text, (
                f"id={raw['id']}: type_b safe/unsafe openings are unexpectedly identical"
            )

    def test_type_b_all_messages_are_user_role(self):
        """All MTMCS source messages must be user turns."""
        for raw in self.raws:
            recs = _normalize_one(self.adapter, raw)
            for sid, rec in recs.items():
                for msg in rec.messages:
                    assert msg.role == "user", (
                        f"{sid}: role={msg.role!r}, expected 'user'"
                    )


# ---------------------------------------------------------------------------
# Source ID and pair_id structure tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSStructure:
    """Verify source IDs, pair_ids, labels, and splits."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adapter = MTMCSAdapter()

    def test_source_id_format(self):
        """Source IDs must follow mtmcs:{split}:{id}:{modality}:{safety}."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                recs = _normalize_one(self.adapter, raw)
                expected_ids = {
                    f"mtmcs:{split}:{raw['id']:06d}:mm:safe",
                    f"mtmcs:{split}:{raw['id']:06d}:mm:unsafe",
                    f"mtmcs:{split}:{raw['id']:06d}:text:safe",
                    f"mtmcs:{split}:{raw['id']:06d}:text:unsafe",
                }
                assert set(recs.keys()) == expected_ids, (
                    f"split={split} id={raw['id']}: "
                    f"got {set(recs.keys())}, expected {expected_ids}"
                )

    def test_pair_id_shared(self):
        """All 4 records from one row must share the same pair_id."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                recs = _normalize_one(self.adapter, raw)
                pair_ids = {r.metadata["pair_id"] for r in recs.values()}
                assert len(pair_ids) == 1, (
                    f"id={raw['id']}: multiple pair_ids: {pair_ids}"
                )
                expected = f"mtmcs:{split}:{raw['id']:06d}"
                assert pair_ids.pop() == expected

    def test_labels_correct(self):
        """Safe records labeled 'safe', unsafe records labeled 'unsafe'."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                recs = _normalize_one(self.adapter, raw)
                for sid, rec in recs.items():
                    if sid.endswith(":safe"):
                        assert rec.label == "safe", f"{sid}: label={rec.label!r}"
                    elif sid.endswith(":unsafe"):
                        assert rec.label == "unsafe", f"{sid}: label={rec.label!r}"

    def test_source_split_preserved(self):
        """source_split must be 'type_a' or 'type_b', not None."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                recs = _normalize_one(self.adapter, raw)
                for sid, rec in recs.items():
                    assert rec.source_split == split, (
                        f"{sid}: source_split={rec.source_split!r}, expected {split!r}"
                    )

    def test_source_setting_matches_split(self):
        """source_setting must be 'type_a' or 'type_b', not hardcoded 'escalation'."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                recs = _normalize_one(self.adapter, raw)
                for sid, rec in recs.items():
                    assert rec.source_setting == split, (
                        f"{sid}: source_setting={rec.source_setting!r}, expected {split!r}"
                    )

    def test_four_records_per_row(self):
        """Each HF row must produce exactly 4 canonical records."""
        for split in ("type_a", "type_b"):
            raws = _load_split_records(split, n=2)
            for raw in raws:
                records = self.adapter.normalize_row(raw)
                assert len(records) == 4, (
                    f"split={split} id={raw['id']}: got {len(records)} records, expected 4"
                )

    def test_multimodal_has_image_unimodal_does_not(self):
        """Multimodal records should have images; unimodal should not."""
        raws = _load_split_records("type_a", n=2)
        for raw in raws:
            recs = _normalize_one(self.adapter, raw)
            for sid, rec in recs.items():
                if ":mm:" in sid:
                    assert rec.has_images, f"{sid}: multimodal record has no images"
                elif ":text:" in sid:
                    assert not rec.has_images, f"{sid}: unimodal record has images"

    def test_variant_images_preserved_in_metadata(self):
        """variant_image_paths must be present in metadata."""
        raws = _load_split_records("type_a", n=2)
        for raw in raws:
            recs = _normalize_one(self.adapter, raw)
            for sid, rec in recs.items():
                assert "variant_image_paths" in rec.metadata, (
                    f"{sid}: missing variant_image_paths in metadata"
                )


# ---------------------------------------------------------------------------
# Image failure handling
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSMediaFailure:
    """Image failures must raise MediaLoadError, never return None silently."""

    def test_none_image_raises_media_error(self):
        """Passing None as image must raise MediaLoadError."""
        adapter = MTMCSAdapter()
        raws = _load_split_records("type_a", n=1)
        raw = raws[0]
        raw["image"] = None  # simulate missing image

        with pytest.raises(MediaLoadError):
            adapter.normalize_row(raw)

    def test_corrupt_image_raises_media_error(self, tmp_path):
        """A corrupt PIL image must raise MediaLoadError."""
        adapter = MTMCSAdapter(media_dir=str(tmp_path))
        raws = _load_split_records("type_a", n=1)
        raw = raws[0]

        # Replace image with a non-image object
        class FakeImage:
            def save(self, path):
                with open(path, "wb") as f:
                    f.write(b"not a real image")

        raw["image"] = FakeImage()

        with pytest.raises(MediaLoadError):
            adapter.normalize_row(raw)


# ---------------------------------------------------------------------------
# Atomic grouping and error-handling tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestMTMCSAtomicGrouping:
    """Verify that load_and_normalize never truncates within a 4-record group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adapter = MTMCSAdapter()

    def test_max_rows_returns_exact_multiples(self):
        """max_rows=N must return exactly N*4 records."""
        for n in (1, 2, 3):
            examples = self.adapter.load_and_normalize(
                split="type_b", max_rows=n
            )
            assert len(examples) == n * 4, (
                f"max_rows={n}: expected {n*4}, got {len(examples)}"
            )

    def test_grouped_returns_list_of_4_tuples(self):
        """load_and_normalize_grouped must return list[list[4]]."""
        groups = self.adapter.load_and_normalize_grouped(
            split="type_b", max_rows=2
        )
        assert len(groups) == 2
        for group in groups:
            assert len(group) == 4

    def test_grouped_conditions_complete(self):
        """Each group must have all 4 conditions."""
        groups = self.adapter.load_and_normalize_grouped(
            split="type_b", max_rows=2
        )
        for i, group in enumerate(groups):
            conditions = {
                f"{r.metadata['modality']}:{r.metadata['safety']}"
                for r in group
            }
            assert conditions == {"multimodal:safe", "multimodal:unsafe", "unimodal:safe", "unimodal:unsafe"}, (
                f"group {i}: incomplete conditions: {conditions}"
            )

    def test_grouped_pair_ids_unique(self):
        """Each group must have a unique pair_id."""
        groups = self.adapter.load_and_normalize_grouped(
            split="type_a", max_rows=3
        )
        pair_ids = [g[0].metadata["pair_id"] for g in groups]
        assert len(pair_ids) == len(set(pair_ids)), "Duplicate pair_ids found"

    def test_on_error_raise_propagates(self):
        """Default on_error='raise' must not silently skip bad rows."""
        import unittest.mock as mock

        # Make normalize_row raise MediaLoadError for any input
        with mock.patch.object(
            MTMCSAdapter, "normalize_row",
            side_effect=MediaLoadError("mock_path", "mock image failure"),
        ):
            with pytest.raises(MediaLoadError):
                self.adapter.load_and_normalize(split="type_a", max_rows=1)

    def test_on_error_record_accounts_for_rejections(self):
        """on_error='record' must produce rejection entries for bad rows."""
        from causal_mllm.adapters.base import NormalizationRejection

        rejections: list[NormalizationRejection] = []
        examples = self.adapter.load_and_normalize(
            split="type_b",
            max_rows=3,
            on_error="record",
            rejections=rejections,
        )
        # All rows should succeed for real data (no rejections expected)
        assert len(rejections) == 0
        assert len(examples) == 3 * 4

    def test_on_error_record_requires_rejections_list(self):
        """on_error='record' without rejections list must raise ValueError."""
        with pytest.raises(ValueError, match="rejections list"):
            self.adapter.load_and_normalize(
                split="type_b", max_rows=1, on_error="record"
            )

    def test_no_silent_disappearance(self):
        """Every source row must produce either 4 records or 1 rejection.

        (processed_rows * 4) == len(results) + (len(rejections) * 4)
        """
        from causal_mllm.adapters.base import NormalizationRejection

        rejections: list[NormalizationRejection] = []
        examples = self.adapter.load_and_normalize(
            split="type_b",
            max_rows=5,
            on_error="record",
            rejections=rejections,
        )
        # For clean data: 5 rows * 4 = 20 records, 0 rejections
        n_rows = 5
        assert len(examples) + len(rejections) * 4 == n_rows * 4, (
            f"Accounting error: {len(examples)} records + "
            f"{len(rejections)} rejections × 4 ≠ {n_rows} × 4"
        )
