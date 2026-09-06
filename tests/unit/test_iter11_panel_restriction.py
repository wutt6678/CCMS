"""Iteration 11.7: a refused cell costs a family, and the report says so.

Aliyun's input moderation refuses 2 of the 600 frozen cells for judge A --
family ``CMST_795308`` in its ``cross_modal`` and ``shuffle`` variants. Those
cells are excluded from every arm, so no arm has a label for them. But the
frozen estimator needs all six variants of a family to produce ANY of its five
estimands, and the paired bootstrap resamples FAMILIES while evaluating all
five on the same resample. So the exclusion does not cost 2 cells: it costs the
whole family, and the analysis runs on 594 records over 99 families.

Three numbers are true at once, about three different stages, and conflating
any two of them would misreport the panel:

    the REPLAY generated 600 cells over 100 families (the panel gate certifies
    this and still passes -- restricting afterwards is what keeps the two
    statements separate);
    the JUDGE labelled 598 of them;
    the ANALYSIS uses 594, over the 99 families that stayed complete.

CI-safe: pure functions over synthetic records, no network and no fixtures.
"""

from __future__ import annotations

import pytest

from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.estimands import (
    REQUIRED_VARIANTS,
    compute_family_estimands,
    incomplete_families,
)
from causal_mllm.evaluation.runner import restrict_panel_to_labels

VARIANTS = sorted(REQUIRED_VARIANTS)

# The exclusion as the production run recorded it, restated rather than read
# from the artifact so that a change to the panel is a test failure to read.
REFUSED_FAMILY = "CMST_795308"
REFUSED_CELLS = ((REFUSED_FAMILY, "cross_modal"), (REFUSED_FAMILY, "shuffle"))


def _family_ids(n_families: int = 100) -> list[str]:
    """``n_families`` distinct ids, always including the refused one.

    The real panel's ids are not sequential, and a synthetic panel that omitted
    ``CMST_795308`` would make every test of the actual exclusion vacuous --
    the restriction would raise "not in this panel" instead of restricting.
    """
    others = [f"CMST_{fi:06d}" for fi in range(n_families - 1)]
    return sorted({REFUSED_FAMILY, *others})


def _panel(n_families: int = 100) -> list[dict]:
    """The frozen panel's shape: every family, all six variants."""
    ids = _family_ids(n_families)
    assert REFUSED_FAMILY in ids, "the refused family must be in the panel"
    assert len(ids) == n_families
    return [{"family_id": fid, "variant": variant,
             "response": f"{fid}-{variant}"}
            for fid in ids for variant in VARIANTS]


class TestIncompleteFamilies:
    def test_a_complete_panel_has_none(self):
        assert incomplete_families(_panel()) == []

    def test_a_family_missing_one_variant_is_named(self):
        records = [r for r in _panel(3)
                   if not (r["family_id"] == "CMST_000001"
                           and r["variant"] == "shuffle")]
        assert incomplete_families(records) == ["CMST_000001"]

    def test_it_names_every_broken_family_not_just_the_first(self):
        records = [r for r in _panel(5)
                   if r["variant"] != "neutral"
                   or r["family_id"] not in ("CMST_000001", "CMST_000003")]
        assert incomplete_families(records) == ["CMST_000001", "CMST_000003"]

    def test_the_estimator_still_raises_where_the_helper_only_reports(self):
        # The split is the point: compute_family_estimands stays strict so a
        # silently truncated panel cannot pass, and incomplete_families is how
        # a caller that has ALREADY lost cells restricts on purpose.
        records = [r for r in _panel(2)
                   if not (r["family_id"] == "CMST_000000"
                           and r["variant"] == "cross_modal")]
        assert incomplete_families(records) == ["CMST_000000"]
        with pytest.raises(EvaluationError, match="missing variants"):
            compute_family_estimands(records)

    def test_a_record_without_a_family_or_variant_is_not_a_broken_family(self):
        assert incomplete_families([{"variant": "shuffle"}, {"family_id": "X"}
                                    ]) == []


class TestRestrictPanelToLabels:
    def test_no_exclusions_changes_nothing_and_records_nothing(self):
        records = _panel()
        kept, restriction = restrict_panel_to_labels(records, [])
        assert kept == records
        assert restriction == {}

    def test_the_iteration_11_exclusion_costs_a_whole_family(self):
        kept, restriction = restrict_panel_to_labels(_panel(), REFUSED_CELLS)
        assert restriction["n_records_in_panel"] == 600
        assert restriction["n_records_analysed"] == 594
        assert restriction["n_families_in_panel"] == 100
        assert restriction["n_families_analysed"] == 99
        assert restriction["n_excluded_cells"] == 2
        assert restriction["cells_dropped_with_their_family"] == 4, \
            "2 excluded cells plus the 4 survivors of the same family"
        assert restriction["families_dropped_incomplete"] == [REFUSED_FAMILY]

    def test_no_surviving_record_belongs_to_the_broken_family(self):
        kept, _ = restrict_panel_to_labels(_panel(), REFUSED_CELLS)
        assert not [r for r in kept if r["family_id"] == REFUSED_FAMILY]
        assert len(kept) == 594

    def test_the_survivors_satisfy_the_frozen_estimator(self):
        # The whole reason for the restriction: what comes out must be
        # consumable by the estimator that raised in the first place.
        kept, _ = restrict_panel_to_labels(_panel(), REFUSED_CELLS)
        scored = [{**r, "unsafe_compliance_score": 0.25} for r in kept]
        estimands = compute_family_estimands(scored)
        assert len(estimands) == 99
        assert REFUSED_FAMILY not in estimands

    def test_the_restriction_states_why_and_that_it_was_not_chosen(self):
        _, restriction = restrict_panel_to_labels(_panel(), REFUSED_CELLS)
        assert "all six variants" in restriction["rule"]
        assert "before any label existed" in restriction["outcome_independent"]

    def test_the_excluded_cells_are_listed_by_name(self):
        _, restriction = restrict_panel_to_labels(_panel(), REFUSED_CELLS)
        assert restriction["excluded_cells"] == [
            "CMST_795308/cross_modal", "CMST_795308/shuffle"]


class TestRestrictionFailsClosed:
    def test_an_exclusion_naming_an_absent_cell_raises(self):
        # A mistyped family id would otherwise shrink nothing while the report
        # claimed a restriction, which is worse than either outcome alone.
        with pytest.raises(EvaluationError, match="not in this panel"):
            restrict_panel_to_labels(
                _panel(), [("CMST_999999", "cross_modal")])

    def test_one_matched_and_one_unmatched_exclusion_still_raises(self):
        with pytest.raises(EvaluationError, match="CMST_999999/cross_modal"):
            restrict_panel_to_labels(
                _panel(), [REFUSED_CELLS[0], ("CMST_999999", "cross_modal")])

    @pytest.mark.parametrize("bad", [
        "CMST_795308/cross_modal",
        ("CMST_795308",),
        ("CMST_795308", "cross_modal", "extra"),
        ("", "cross_modal"),
        ("CMST_795308", None),
        42,
    ])
    def test_a_malformed_exclusion_raises(self, bad):
        with pytest.raises(EvaluationError, match=r"\(family_id, variant\)"):
            restrict_panel_to_labels(_panel(), [bad])

    def test_excluding_a_whole_family_is_accepted(self):
        # Legal, if unusual: every cell of one family gone leaves 99 complete
        # families and no partial one to drop.
        cells = [("CMST_000000", v) for v in VARIANTS]
        kept, restriction = restrict_panel_to_labels(_panel(), cells)
        assert restriction["n_records_analysed"] == 594
        assert restriction["n_families_analysed"] == 99
        assert restriction["cells_dropped_with_their_family"] == 0
        assert len(kept) == 594

    def test_a_restriction_that_leaves_nothing_raises(self):
        ids = _family_ids(2)
        cells = [(fid, v) for fid in ids for v in VARIANTS]
        with pytest.raises(EvaluationError, match="left no complete family"):
            restrict_panel_to_labels(_panel(2), cells)

    def test_a_list_cell_is_accepted_as_well_as_a_tuple(self):
        kept, _ = restrict_panel_to_labels(
            _panel(), [list(cell) for cell in REFUSED_CELLS])
        assert len(kept) == 594
