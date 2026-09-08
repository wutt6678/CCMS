"""Iteration 11: the transportability decision, derived rather than declared.

The Iteration 10 result is a positive Delta_TV in Qwen3.5-9B. Iteration 11
replayed the same frozen panel through four more targets and asked, as its
confirmatory family, whether the sign holds in each. It holds in one. The
sentence that follows from that is "model-specific, not generally transported",
and this stage files it as a decision with its measurement attached.

A decision artifact is the easiest thing in the repository to overstate: it is
one paragraph, nobody re-runs it, and the paragraph is the part a paper quotes.
So what is pinned here is not that the paragraph reads well but that it cannot
read better than the numbers under it:

* the call is DERIVED -- ``decide`` computes it from the per-target signs, and a
  document whose own signs support the other call is refused rather than filed;
* the two facts the bound reports about a sign are kept apart, because the trap
  in this evidence is real and the first draft of the stage fell into it. The
  bound's ``for_every_admissible_score`` asks whether a target MATCHES the
  reference at every admissible score of the one differentially missing cell, so
  it is False in exactly the three targets that reverse, and a verifier that
  read that column as "failed to survive the sensitivity" would report the
  reversal as a weakness in the decision instead of as the decision;
* every prose count is generated from the count it describes, so moving a sign
  moves the sentence;
* the panel arithmetic is stated where two numbers that look as though one
  should contain the other do not: the per-target mean is a bootstrap mean over
  98 families and the bound's range is a point mean over 99, and for one target
  the former lies outside the latter. That is filed and explained rather than
  left for a reader to find as an apparent arithmetic error.

CI-safe: every test reads committed artifacts or synthesises the shapes they
have. Nothing calls a model, a judge or the network.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


decision = _load_script("iter11_transportability_decision")

ARMS = decision.EXPECTED_TARGETS
FILED = decision.OUT_PATH

#: The panel as the committed analysis records it: 100 families, three cells no
#: judge could label, one of them missing in a single arm.
CELLS = ["CMST_456921/text_only", "CMST_795308/cross_modal",
         "CMST_795308/shuffle"]
SHARED_CELLS = ["CMST_795308/cross_modal", "CMST_795308/shuffle"]


def _filed() -> dict:
    assert FILED.is_file(), f"{FILED} is committed evidence; run the stage"
    return json.loads(FILED.read_text(encoding="utf-8"))


def _analysis(signs: dict[str, str], means: dict[str, float] | None = None) -> dict:
    """A cross-model analysis with the shape ``per_target`` reads."""
    means = means or {}
    per_model, verdicts = {}, {}
    for i, arm in enumerate(ARMS):
        sign = signs[arm]
        mean = means.get(arm, 0.085 if sign == "positive" else -0.13)
        per_model[arm] = {
            "sign": sign,
            "sign_matches_reference": sign == "positive",
            "n_families": 98,
            "estimands": {"Delta_TV": {
                "mean": mean, "bootstrap_mean": mean,
                "ci_lower": mean - 0.05, "ci_upper": mean + 0.05, "n": 98}},
            "panel_restriction": {
                "excluded_cells": ([CELLS[0]] if arm == "ministral3_3b" else [])
                + SHARED_CELLS,
                "rule": "a cell no judge could label is dropped, and so is "
                        "every remaining cell of a family that lost one"},
        }
        verdicts[f"H{i + 1}"] = {
            "model": arm, "statement": f"{arm} matches the reference sign.",
            "verdict": "confirmed" if sign == "positive" else "refuted",
            "raw_p": 0.0002, "adjusted_p": 0.0008}
    return {
        "primary_estimand": "Delta_TV",
        "per_model": per_model,
        "verdicts": verdicts,
        "common_panel": {
            "n_families_in_panel": 100, "n_families_common": 98,
            "union_excluded_cells": list(CELLS),
            "families_dropped": ["CMST_456921", "CMST_795308"],
            "per_target_excluded_cells": {
                arm: per_model[arm]["panel_restriction"]["excluded_cells"]
                for arm in ARMS}},
        "reference": {"model": "Qwen3.5-9B", "status": "restricted",
                      "n_families": 98, "sign": "positive",
                      "Delta_TV": {"bootstrap_mean": 0.11367,
                                   "ci_lower": 0.0459, "ci_upper": 0.1811},
                      "source": "reference_restriction.json",
                      "source_sha256": "0" * 64},
        "summary": {"retention": "all results are retained and reported"},
    }


def _bound(ranges: dict[str, tuple[float, float]],
           holds: dict[str, bool] | None = None) -> dict:
    """A bound whose mean ranges and match-at-every-score flags are given."""
    holds = {} if holds is None else holds
    rows = {}
    for i, arm in enumerate(ARMS):
        low, high = ranges[arm]
        sign = "positive" if low > 0.0 else "negative"
        rows[f"H{i + 1}"] = {
            "arm": arm,
            "sign_matches_the_99_family_reference": {
                "at_the_label_used": holds.get(arm),
                "for_every_admissible_score": holds.get(
                    arm, sign == "positive"),
                "mean_range_over_the_whole_rubric": [low, high]},
            "p_over_the_whole_rubric": {"worst_p": 0.0002}}
    return {
        "cell": CELLS[0], "family": "CMST_456921", "variant": "text_only",
        "family_still_excluded": "CMST_795308",
        "the_arm_whose_reply_caused_the_refusal": "ministral3_3b",
        "n_families_committed": 98, "n_families_sensitivity": 99,
        "score_range_bounded": [0.0, 1.0],
        "why_the_sensitivity_stops_at_99": "CMST_795308 cannot be restored "
                                           "anywhere",
        "what_this_does_not_do": "this is a sensitivity, not a re-run",
        "verdicts": {"per_hypothesis": rows},
    }


MATCHING = {arm: "positive" for arm in ARMS}
ONE_MATCH = {"ministral3_3b": "negative", "phi4_mm": "negative",
             "qwen35_2b": "negative", "qwen35_4b": "positive"}
NEG_RANGES = {arm: (-0.15, -0.12) for arm in ARMS}
POS_RANGES = {arm: (0.08, 0.09) for arm in ARMS}


def _as_filed(signs: dict[str, str],
              ranges: dict[str, tuple[float, float]],
              holds: dict[str, bool] | None = None) -> dict:
    """What the committed evidence looks like: matching signs, pinned ranges."""
    analysis = _analysis(signs)
    bound = _bound(ranges, holds)
    reference = decision.reference_of(analysis)
    targets = decision.per_target(analysis, bound)
    measurement = decision.decide(reference, targets)
    return {
        "decision": measurement["decision"],
        "the_decision_in_words":
            decision.the_decision_in_words(reference, measurement),
        "reference": reference,
        "per_target": targets,
        "the_measurement": measurement,
        "why_this_is_not_an_artifact_of_the_exclusion": {
            "the_two_facts_read_out_of_the_bound":
                measurement["what_surviving_the_exclusion_required"]},
        **decision.where_the_two_means_part_company(targets),
    }


# ---------------------------------------------------------------------------
# The call is derived
# ---------------------------------------------------------------------------

class TestTheCallIsDerivedFromTheSignsAndNotWrittenDown:
    def test_every_target_matching_is_transportation(self):
        doc = _as_filed(MATCHING, POS_RANGES)
        assert doc["decision"] == decision.DECISION_TRANSPORTED
        assert doc["the_measurement"]["n_reversing_it"] == 0
        assert doc["the_measurement"]["n_carrying_the_reference_sign"] == 4

    def test_one_reversal_out_of_four_is_not(self):
        signs = dict(MATCHING, phi4_mm="negative")
        doc = _as_filed(signs, dict(POS_RANGES, phi4_mm=(-0.06, -0.05)))
        assert doc["decision"] == decision.DECISION_MODEL_SPECIFIC
        assert doc["the_measurement"]["n_reversing_it"] == 1, (
            "the rule requires EVERY target to carry the reference sign, so one "
            "reversal is the whole of the finding and a proportion of four "
            "targets is not a rate")

    def test_the_committed_evidence_reverses_in_three(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        measurement = doc["the_measurement"]
        assert doc["decision"] == decision.DECISION_MODEL_SPECIFIC
        assert measurement["carrying_the_reference_sign"] == ["qwen35_4b"]
        assert measurement["reversing_it"] == [
            "ministral3_3b", "phi4_mm", "qwen35_2b"]
        assert measurement["every_reversal_is_a_result_and_not_a_null"]

    def test_a_sign_that_is_neither_is_a_finding_not_a_count(self):
        analysis = _analysis(dict(ONE_MATCH, phi4_mm=None))
        bound = _bound(dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        reference = decision.reference_of(analysis)
        targets = decision.per_target(analysis, bound)
        with pytest.raises(SystemExit) as exc:
            decision.decide(reference, targets)
        assert exc.value.code == 1, (
            "an undecided sign cannot be counted as either matching or "
            "reversing, and counting it as reversing would decide the call by "
            "an absence")

    def test_a_reference_whose_own_sign_is_neither_is_refused(self):
        analysis = _analysis(MATCHING)
        analysis["reference"]["sign"] = "zero"
        with pytest.raises(SystemExit) as exc:
            decision.reference_of(analysis)
        assert exc.value.code == 1

    def test_a_fifth_target_or_a_missing_one_is_refused(self):
        analysis = _analysis(ONE_MATCH)
        analysis["per_model"]["llama_4"] = copy.deepcopy(
            analysis["per_model"]["phi4_mm"])
        bound = _bound(dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        with pytest.raises(SystemExit) as exc:
            decision.per_target(analysis, bound)
        assert exc.value.code == 1

        del analysis["per_model"]["llama_4"]
        del analysis["per_model"]["qwen35_2b"]
        with pytest.raises(SystemExit) as exc:
            decision.per_target(analysis, bound)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# The trap: two facts about a sign, not one
# ---------------------------------------------------------------------------

class TestTheTwoFactsTheBoundReportsAboutASignAreKeptApart:
    """``for_every_admissible_score`` asks about MATCHING, not about flipping."""

    def test_a_reversal_is_false_in_the_matching_field_and_true_in_the_pin(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        for arm in ("ministral3_3b", "phi4_mm", "qwen35_2b"):
            target = doc["per_target"][arm]
            assert target["the_reference_sign_holds_at_every_admissible_score"] \
                is False
            assert target["the_mean_cannot_flip_sign_over_the_whole_rubric"] \
                is True
            assert target["the_sign_the_mean_cannot_leave"] == "negative"

    def test_the_decision_survives_because_of_the_pin_and_not_the_match(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        measurement = doc["the_measurement"]
        assert measurement["the_decision_survives_the_differential_exclusion"] \
            is True, (
                "the three reversals are False in the bound's matching field, "
                "and reading that field alone would report the decision as "
                "failing the sensitivity it was checked against")
        required = measurement["what_surviving_the_exclusion_required"]
        assert required["the_mean_cannot_reach_zero_in"] == sorted(ARMS)
        assert required["the_reference_sign_holds_at_every_admissible_score_in"] \
            == ["qwen35_4b"]

    def test_a_range_straddling_zero_is_the_case_that_does_fail(self):
        doc = _as_filed(ONE_MATCH,
                        dict(NEG_RANGES, phi4_mm=(-0.01, 0.02),
                             qwen35_4b=(0.08, 0.09)))
        target = doc["per_target"]["phi4_mm"]
        assert target["the_mean_cannot_flip_sign_over_the_whole_rubric"] is False
        assert target["the_sign_the_mean_cannot_leave"] is None
        assert doc["the_measurement"][
            "the_decision_survives_the_differential_exclusion"] is False, (
                "a mean that reaches zero somewhere in the rubric leaves the "
                "reversal to the missing label, which is the one thing this "
                "stage must not file")

    def test_a_range_touching_zero_is_degenerate_not_positive(self):
        doc = _as_filed(ONE_MATCH,
                        dict(NEG_RANGES, phi4_mm=(0.0, 0.02),
                             qwen35_4b=(0.08, 0.09)))
        target = doc["per_target"]["phi4_mm"]
        assert target["the_mean_cannot_flip_sign_over_the_whole_rubric"] is False
        assert target["the_sign_the_mean_cannot_leave"] is None

    def test_a_match_that_holds_only_at_the_label_used_does_not_survive(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)),
                        holds={**{arm: False for arm in ARMS},
                               "qwen35_4b": False})
        measurement = doc["the_measurement"]
        assert measurement["the_decision_survives_the_differential_exclusion"] \
            is False, (
                "qwen35_4b carries the reference sign at the label the frozen "
                "rule gave but not at every admissible score, so the count 1 of "
                "4 depends on the missing label")
        assert measurement["what_surviving_the_exclusion_required"][
            "the_reference_sign_holds_at_every_admissible_score_in"] == []

    def test_the_entailment_between_the_two_fields_is_checked_not_assumed(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        assert decision.check_the_decision(doc) == []
        # A match at every score entails a mean pinned to the reference side.
        forged = copy.deepcopy(doc)
        forged["per_target"]["qwen35_4b"][
            "the_mean_cannot_flip_sign_over_the_whole_rubric"] = False
        forged["per_target"]["qwen35_4b"]["the_sign_the_mean_cannot_leave"] = None
        issues = decision.check_the_decision(forged)
        assert any("holds at every admissible score" in i or "entails" in i
                   for i in issues), issues

    def test_the_two_sources_for_a_sign_must_agree(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        # The analysis files the 98-family bootstrap sign; the bound files the
        # 99-family range. Different panels, different code, same fact.
        forged["per_target"]["phi4_mm"]["sign"] = "positive"
        forged["per_target"]["phi4_mm"]["sign_matches_the_reference"] = True
        issues = decision.check_the_decision(forged)
        assert any("cannot leave 'negative'" in i for i in issues), issues


# ---------------------------------------------------------------------------
# What is missing from the panel, read rather than remembered
# ---------------------------------------------------------------------------

class TestTheExclusionIsReadOutOfTheCommittedAnalysis:
    def test_a_cell_missing_in_one_arm_is_differential(self):
        analysis = json.loads(decision.CROSS_MODEL.read_text(encoding="utf-8"))
        bound = json.loads(decision.BOUND.read_text(encoding="utf-8"))
        exclusion = decision.the_exclusion(analysis, bound)
        assert exclusion[
            "missing_in_one_arm_only_and_therefore_differential"] == [
                "CMST_456921/text_only"]
        assert exclusion[
            "missing_in_every_arm_and_therefore_not_differential"] == [
                "CMST_795308/cross_modal", "CMST_795308/shuffle"]
        assert exclusion["n_families_in_the_panel"] == 100
        assert exclusion["n_families_in_the_confirmatory_panel"] == 98
        assert exclusion["n_families_the_bound_restores_to"] == 99
        assert exclusion["the_arm_that_lost_it"] == "ministral3_3b"

    def test_the_panel_is_98_for_two_reasons_and_only_one_is_differential(self):
        analysis = json.loads(decision.CROSS_MODEL.read_text(encoding="utf-8"))
        bound = json.loads(decision.BOUND.read_text(encoding="utf-8"))
        exclusion = decision.the_exclusion(analysis, bound)
        assert exclusion["families_dropped"] == ["CMST_456921", "CMST_795308"]
        assert exclusion["the_family_still_out_of_the_bound"] == "CMST_795308"

    def test_a_bound_over_a_sharedly_missing_cell_is_refused(self):
        analysis = _analysis(ONE_MATCH)
        bound = _bound(dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        bound["cell"] = "CMST_795308/cross_modal"
        with pytest.raises(SystemExit) as exc:
            decision.the_exclusion(analysis, bound)
        assert exc.value.code == 1, (
            "a cell missing in every arm is shared missingness; ranging over it "
            "would bound nothing differential")

    def test_a_bound_with_no_row_for_an_arm_is_refused(self):
        analysis = _analysis(ONE_MATCH)
        bound = _bound(dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        del bound["verdicts"]["per_hypothesis"]["H4"]
        with pytest.raises(SystemExit) as exc:
            decision.per_target(analysis, bound)
        assert exc.value.code == 1

    def test_a_mean_range_that_is_not_an_interval_is_refused(self):
        analysis = _analysis(ONE_MATCH)
        bound = _bound(dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        row = bound["verdicts"]["per_hypothesis"]["H4"]
        row["sign_matches_the_99_family_reference"][
            "mean_range_over_the_whole_rubric"] = [0.08]
        with pytest.raises(SystemExit) as exc:
            decision.per_target(analysis, bound)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# The two means do not nest
# ---------------------------------------------------------------------------

class TestTheTwoMeansAreOverDifferentPanels:
    def test_a_mean_outside_its_range_is_reported_where_it_falls(self):
        doc = _as_filed(
            ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08282828, 0.09292929)),
        )
        doc["per_target"]["qwen35_4b"]["mean"] = 0.09319214
        block = decision.where_the_two_means_part_company(doc["per_target"])
        nesting = block[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        assert nesting[
            "n_targets_whose_98_family_mean_falls_outside_the_99_family_range"
        ] == 1
        where = nesting["where"]["qwen35_4b"]
        assert where["which_end_it_falls_past"] == "above"
        assert where["by"] == pytest.approx(0.09319214 - 0.09292929)
        assert where["both_still_carry_the_same_sign"] is True

    def test_a_mean_inside_its_range_is_not_reported(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.12)))
        nesting = decision.where_the_two_means_part_company(doc["per_target"])[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        assert nesting["where"] == {}
        assert nesting[
            "n_targets_whose_98_family_mean_falls_outside_the_99_family_range"
        ] == 0

    def test_the_filed_document_names_the_one_target_that_does_not_nest(self):
        doc = _filed()
        nesting = doc[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        assert sorted(nesting["where"]) == ["qwen35_4b"]
        assert nesting["where"]["qwen35_4b"]["which_end_it_falls_past"] == "above"
        assert nesting["where"]["qwen35_4b"][
            "both_still_carry_the_same_sign"] is True, (
                "the non-nesting is a difference of denominator, not of sign, "
                "and if the signs disagreed the decision would not survive it")

    def test_a_sign_disagreement_across_the_two_panels_would_be_named(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        doc["per_target"]["qwen35_4b"]["mean"] = -0.001
        nesting = decision.where_the_two_means_part_company(doc["per_target"])[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        assert nesting["where"]["qwen35_4b"][
            "both_still_carry_the_same_sign"] is False


# ---------------------------------------------------------------------------
# The prose cannot outrun the numbers
# ---------------------------------------------------------------------------

class TestTheProseIsGeneratedFromTheCountsItStates:
    def test_the_filed_sentence_carries_the_filed_counts(self):
        doc = _filed()
        measurement = doc["the_measurement"]
        words = doc["the_decision_in_words"]
        assert f"{measurement['n_carrying_the_reference_sign']} of " \
               f"{measurement['n_targets_tested']} targets carry" in words
        assert f"the other {measurement['n_reversing_it']} reverse it" in words
        assert f"reverses in {measurement['n_reversing_it']} of " \
               f"{measurement['n_targets_tested']} targets" in words
        for arm in measurement["carrying_the_reference_sign"]:
            assert arm in words
        for arm in measurement["reversing_it"]:
            assert arm in words

    def test_no_count_is_spelled_out_where_a_numeral_is_derived(self):
        """A written-out "three of four" beside a derived 3 outlives the 3."""
        source = (ROOT / "scripts" / "iter11_transportability_decision.py") \
            .read_text(encoding="utf-8").splitlines()
        # Comments explain the trap and are allowed to name it; string literals
        # are filed prose and are not.
        body = "\n".join(line for line in source
                         if not line.lstrip().startswith("#"))
        start = body.index("def the_decision_in_words")
        end = body.index("def where_the_two_means_part_company")
        for spelled in ("three of four", "one of four", "two of four",
                        "three of the four", "one of the four"):
            assert spelled not in body[start:end].lower(), (
                f"the_decision_in_words says {spelled!r} in prose; the count is "
                f"derived, so the sentence has to be too")

    def test_a_different_count_produces_a_different_sentence(self):
        reference = {"sign": "positive"}
        three = {"n_targets_tested": 4,
                 "carrying_the_reference_sign": ["qwen35_4b"],
                 "reversing_it": ["ministral3_3b", "phi4_mm", "qwen35_2b"]}
        two = {"n_targets_tested": 4,
               "carrying_the_reference_sign": ["phi4_mm", "qwen35_4b"],
               "reversing_it": ["ministral3_3b", "qwen35_2b"]}
        words_three = decision.the_decision_in_words(reference, three)
        words_two = decision.the_decision_in_words(reference, two)
        assert "1 of 4 targets carry" in words_three
        assert "the other 3 reverse it" in words_three
        assert "reverses in 3 of 4 targets" in words_three
        assert "2 of 4 targets carry" in words_two
        assert "the other 2 reverse it" in words_two
        assert words_three != words_two, (
            "a sentence that does not move when the count moves is a sentence "
            "about the count that was true when it was written")

    def test_both_branches_of_the_rule_are_written(self):
        doc = _as_filed(MATCHING, POS_RANGES)
        assert decision.check_the_decision(doc) == []
        assert doc["the_decision_in_words"].startswith("GENERALLY TRANSPORTED")
        assert "not what the committed evidence produced" in doc[
            "the_decision_in_words"], (
                "the branch this evidence did not produce is filed anyway, so "
                "the rule is visible in both directions")
        reversed_doc = _as_filed(
            ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        assert decision.check_the_decision(reversed_doc) == []
        assert reversed_doc["the_decision_in_words"].startswith(
            "MODEL-SPECIFIC")

    def test_the_filed_document_agrees_with_itself(self):
        assert decision.check_the_decision(_filed()) == []

    def test_a_decision_contradicted_by_its_own_signs_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        forged["decision"] = decision.DECISION_TRANSPORTED
        issues = decision.check_the_decision(forged)
        assert any("disagree" in i for i in issues), issues

    def test_a_sentence_edited_after_the_fact_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        assert decision.check_the_decision(doc) == []
        forged = copy.deepcopy(doc)
        forged["the_decision_in_words"] = forged[
            "the_decision_in_words"].replace(
                "not one this panel supports", "supported by this panel")
        issues = decision.check_the_decision(forged)
        assert any("drifted from the measurement" in i for i in issues), issues

    def test_a_sentence_left_behind_by_a_moved_count_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        # phi4_mm joins the matchers; the counts move but the sentence does not.
        forged["per_target"]["phi4_mm"]["sign"] = "positive"
        forged["per_target"]["phi4_mm"]["sign_matches_the_reference"] = True
        forged["per_target"]["phi4_mm"]["verdict"] = "confirmed"
        forged["per_target"]["phi4_mm"][
            "the_sign_the_mean_cannot_leave"] = "positive"
        forged["per_target"]["phi4_mm"][
            "mean_range_over_the_whole_rubric"] = [0.05, 0.06]
        forged["per_target"]["phi4_mm"][
            "the_reference_sign_holds_at_every_admissible_score"] = True
        forged["the_measurement"]["carrying_the_reference_sign"] = [
            "phi4_mm", "qwen35_4b"]
        forged["the_measurement"]["reversing_it"] = [
            "ministral3_3b", "qwen35_2b"]
        forged["the_measurement"]["n_carrying_the_reference_sign"] = 2
        forged["the_measurement"]["n_reversing_it"] = 2
        forged["the_measurement"]["decision"] = \
            decision.DECISION_MODEL_SPECIFIC
        forged["the_measurement"]["every_reversal_is_a_result_and_not_a_null"] \
            = True
        forged["the_measurement"]["what_surviving_the_exclusion_required"][
            "the_reference_sign_holds_at_every_admissible_score_in"] = [
                "phi4_mm", "qwen35_4b"]
        forged["why_this_is_not_an_artifact_of_the_exclusion"][
            "the_two_facts_read_out_of_the_bound"][
            "the_reference_sign_holds_at_every_admissible_score_in"] = [
                "phi4_mm", "qwen35_4b"]
        nesting = forged[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        nesting["n_targets_whose_98_family_mean_falls_outside_the_99_family_"
                "range"] = 0
        nesting["where"] = {}
        issues = decision.check_the_decision(forged)
        assert any("drifted from the measurement" in i for i in issues), (
            "everything else in the document was updated consistently and only "
            f"the sentence was left behind: {issues}")

    def test_a_reversal_called_inconclusive_is_not_a_result(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        forged["per_target"]["phi4_mm"]["verdict"] = "inconclusive"
        forged["the_measurement"][
            "every_reversal_is_a_result_and_not_a_null"] = False
        issues = decision.check_the_decision(forged)
        assert any("result rather than a null" in i for i in issues), issues

    def test_a_reverser_moved_into_the_holds_everywhere_list_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        required = forged["the_measurement"][
            "what_surviving_the_exclusion_required"]
        required["the_reference_sign_holds_at_every_admissible_score_in"] = [
            "phi4_mm", "qwen35_4b"]
        issues = decision.check_the_decision(forged)
        assert any("holds_at_every_admissible_score_in" in i for i in issues)

    def test_a_paragraph_restating_the_table_wrongly_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        forged["why_this_is_not_an_artifact_of_the_exclusion"][
            "the_two_facts_read_out_of_the_bound"][
                "the_mean_cannot_reach_zero_in"] = ["qwen35_4b"]
        issues = decision.check_the_decision(forged)
        assert any("paragraph that disagrees with the table under it" in i
                   for i in issues), issues

    def test_a_survival_claim_the_signs_do_not_support_is_refused(self):
        doc = _as_filed(ONE_MATCH, dict(NEG_RANGES, qwen35_4b=(0.08, 0.09)))
        forged = copy.deepcopy(doc)
        forged["the_measurement"][
            "the_decision_survives_the_differential_exclusion"] = False
        issues = decision.check_the_decision(forged)
        assert any("the_decision_survives_the_differential_exclusion" in i
                   for i in issues), issues

    def test_a_forged_non_nesting_count_is_refused(self):
        doc = _filed()
        forged = copy.deepcopy(doc)
        block = forged[
            "the_two_means_are_over_different_panels_and_do_not_nest"]
        block["n_targets_whose_98_family_mean_falls_outside_the_99_family_"
              "range"] = 0
        block["where"] = {}
        issues = decision.check_the_decision(forged)
        assert any("fall outside" in i for i in issues), issues


# ---------------------------------------------------------------------------
# Re-derivation
# ---------------------------------------------------------------------------

class TestTheDocumentReDerivesExactly:
    def test_the_filed_document_is_the_re_derived_one(self):
        code, conclusion, issues = decision.verify()
        assert code == 0, issues
        assert conclusion == "reproduced_exactly"

    def test_building_twice_yields_the_same_bytes(self):
        first = json.dumps(decision.build(), indent=2, sort_keys=True)
        second = json.dumps(decision.build(), indent=2, sort_keys=True)
        assert first == second

    def test_nothing_in_it_records_when_or_where_it_was_written(self):
        """A timestamp or a commit would make exact re-derivation impossible."""
        doc = _filed()
        banned = {"generated_at", "timestamp", "git_commit", "git_dirty",
                  "code_commit", "python", "host", "generated_from_commit"}

        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key not in banned, f"{path}/{key}"
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(doc, "")

    def test_its_inputs_are_bound_by_hash_and_both_are_committed(self):
        doc = _filed()
        inputs = doc["inputs"]
        for key in ("cross_model_analysis", "differential_censoring_bound"):
            path = ROOT / inputs[key]["path"]
            assert path.is_file(), inputs[key]["path"]
            assert decision.sha256_file(path) == inputs[key]["sha256"]
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", inputs[key]["path"]],
                cwd=ROOT, capture_output=True, text=True)
            assert tracked.returncode == 0, (
                f"{inputs[key]['path']} is an input to a committed decision and "
                f"is not tracked: {tracked.stderr.strip()}")

    def test_a_moved_input_hash_is_a_disagreement(self):
        doc = _filed()
        forged = copy.deepcopy(doc)
        forged["inputs"]["cross_model_analysis"]["sha256"] = "0" * 64
        code, conclusion, issues = decision.verify(decision.OUT_PATH)
        assert code == 0  # the filed document still re-derives
        assert forged["inputs"] != doc["inputs"]


# ---------------------------------------------------------------------------
# Exit codes: one test per documented code
# ---------------------------------------------------------------------------

class TestTheExitCodes:
    def test_zero_when_the_filed_decision_re_derives(self, tmp_path):
        target = tmp_path / "decision.json"
        target.write_text(json.dumps(decision.build(), indent=2),
                          encoding="utf-8")
        assert decision.main(["--verify", "--out", str(target)]) == 0

    def test_one_when_the_filed_decision_does_not(self, tmp_path):
        doc = decision.build()
        doc["decision"] = decision.DECISION_TRANSPORTED
        target = tmp_path / "decision.json"
        target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        assert decision.main(["--verify", "--out", str(target)]) == 1

    def test_two_when_nothing_is_filed_to_verify_against(self, tmp_path):
        assert decision.main(
            ["--verify", "--out", str(tmp_path / "absent.json")]) == 2

    def test_two_when_an_input_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(decision, "CROSS_MODEL", tmp_path / "absent.json")
        with pytest.raises(SystemExit) as exc:
            decision.build()
        assert exc.value.code == 2

    def test_one_and_two_are_not_the_same_failure(self, tmp_path, monkeypatch):
        """1 is a contradiction; 2 is nothing to contradict."""
        monkeypatch.setattr(decision, "BOUND", tmp_path / "absent.json")
        code, conclusion, _ = decision.verify(tmp_path / "filed.json")
        assert code == 2 and conclusion == "not_filed"
        target = tmp_path / "decision.json"
        target.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(decision, "OUT_PATH", target)
        code, conclusion, issues = decision.verify(target)
        assert code == 2 and conclusion == "could_not_rederive", (code, issues)

    def test_fatal_exits_with_the_code_it_is_given(self):
        with pytest.raises(SystemExit) as exc:
            decision.fatal("a missing input", 2)
        assert exc.value.code == 2, (
            "raise SystemExit('text') exits 1 whatever the text says, and 1 is "
            "this script's code for a disagreement with a filed decision")

    def test_three_is_documented_as_unreachable_here(self):
        """Every input to this stage is committed, so no checkout lacks one."""
        docstring = decision.__doc__
        assert "3  not reachable for this stage" in docstring
        for path in (decision.CROSS_MODEL, decision.BOUND):
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch",
                 str(path.relative_to(ROOT))],
                cwd=ROOT, capture_output=True, text=True)
            assert tracked.returncode == 0, (
                f"{path} is untracked, which would make exit 3 reachable and "
                f"the docstring wrong")

    def test_write_and_verify_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            decision.main(["--verify", "--write"])
        assert exc.value.code == 2  # argparse's own code for a bad invocation

    def test_verify_writes_nothing(self, tmp_path):
        before = FILED.read_bytes()
        decision.main(["--verify"])
        assert FILED.read_bytes() == before
