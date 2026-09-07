"""Iteration 11.8: the confirmatory tests, and the correction over them.

The protocol froze the correction and the estimand but not a p-value, because
Iteration 10 published only means and percentile intervals. These tests pin the
two numbers 11.8 does compute and the one decision rule it applies:

* the bootstrap p-value comes from the SAME resamples as the frozen CI, so the
  two cannot disagree about whether zero is inside the interval. That is not a
  convenience -- the protocol characterises the reference sign by its interval,
  so a test that could contradict the interval it is reported beside would be
  testing something else.
* the family-level sign test is exact and distribution-free, and it answers a
  different question. It is a sensitivity, never the confirmatory verdict.
* Holm-Bonferroni is step-down: it STOPS at the first failure. A procedure
  that kept going would be per-test Bonferroni with extra steps and would not
  control the family-wise rate the protocol pre-declared.
"""

from __future__ import annotations

import math

import pytest

from causal_mllm.evaluation.bootstrap import (
    ESTIMAND_NAMES,
    _percentile,
    bootstrap_two_sided_p,
    paired_bootstrap_ci,
    paired_bootstrap_samples,
)
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.hypotheses import (
    family_sign_test,
    holm_bonferroni,
)


def _families(n: int, delta_tv=None) -> dict[str, dict]:
    """Family estimands over ``n`` families.

    ``delta_tv`` maps family index to a value; the other four estimands are
    filled deterministically so the whole five-estimand matrix is exercised.
    """
    out = {}
    for i in range(n):
        value = 0.1 * (i % 5 - 2) if delta_tv is None else delta_tv(i)
        out[f"CMST_{i:06d}"] = {
            "Delta_T": value / 2.0, "Delta_V": value / 3.0,
            "Delta_TV": value, "order_effect": value / 5.0,
            "history_effect": -value / 7.0}
    return out


class TestTheResamplesAndTheIntervalAreOneDistribution:
    def test_summarising_the_samples_reproduces_the_ci_exactly(self):
        # The refactor that exposed the resamples moved the loop, so this is
        # the assertion that it moved nothing: same rng consumption order,
        # same floats, all five estimands.
        family_estimands = _families(12)
        ci = paired_bootstrap_ci(family_estimands, n_bootstrap=500, seed=42)
        samples = paired_bootstrap_samples(
            family_estimands, n_bootstrap=500, seed=42)
        assert list(samples) == list(ESTIMAND_NAMES)
        for name in ESTIMAND_NAMES:
            ordered = sorted(samples[name])
            assert len(ordered) == 500
            assert ci[name]["mean"] == sum(ordered) / len(ordered)
            assert ci[name]["CI_lower"] == _percentile(ordered, 2.5)
            assert ci[name]["CI_upper"] == _percentile(ordered, 97.5)

    def test_the_seed_still_decides_the_resamples(self):
        family_estimands = _families(12)
        assert paired_bootstrap_samples(family_estimands, 100, seed=42) != \
            paired_bootstrap_samples(family_estimands, 100, seed=43)
        assert paired_bootstrap_samples(family_estimands, 100, seed=42) == \
            paired_bootstrap_samples(family_estimands, 100, seed=42)

    def test_no_families_is_an_error_in_both(self):
        with pytest.raises(EvaluationError, match="no family estimands"):
            paired_bootstrap_samples({}, 100)
        with pytest.raises(EvaluationError, match="no family estimands"):
            paired_bootstrap_ci({}, 100)


class TestTheBootstrapPValue:
    def test_a_distribution_entirely_above_zero_is_floored_not_zero(self):
        # 5000 draws cannot resolve a tail smaller than one draw, and an exact
        # 0.0 would claim a certainty the resamples do not have.
        assert bootstrap_two_sided_p([0.1] * 1000) == pytest.approx(1 / 1000)

    def test_a_distribution_straddling_zero_is_not_significant(self):
        samples = [-0.1] * 500 + [0.1] * 500
        assert bootstrap_two_sided_p(samples) == pytest.approx(1.0)

    def test_the_p_value_and_the_ci_agree_at_the_boundary(self):
        # The property the whole design rests on: p < alpha exactly when the
        # percentile interval at 1 - alpha excludes the null.
        family_estimands = _families(20, delta_tv=lambda i: 0.3 + 0.01 * i)
        samples = paired_bootstrap_samples(family_estimands, 2000, seed=42)
        ci = paired_bootstrap_ci(family_estimands, 2000, 0.95, 42)
        ordered = sorted(samples["Delta_TV"])
        excludes_zero = ci["Delta_TV"]["CI_lower"] > 0.0
        significant = bootstrap_two_sided_p(ordered) < 0.05
        assert excludes_zero == significant

    def test_a_null_panel_is_not_significant_and_says_which_side_it_tips(
            self):
        family_estimands = _families(20, delta_tv=lambda i: 0.05 * (i % 3 - 1))
        samples = paired_bootstrap_samples(family_estimands, 2000, seed=42)
        p = bootstrap_two_sided_p(sorted(samples["Delta_TV"]))
        assert p > 0.05

    def test_the_null_can_be_somewhere_other_than_zero(self):
        # Every resample sits exactly AT the null, so both tails hold all of
        # them and the two-sided doubling has to be capped rather than
        # reported as 2.0.
        assert bootstrap_two_sided_p([0.5] * 100, null=0.5) == 1.0
        assert bootstrap_two_sided_p([0.5] * 100, null=0.0) == \
            pytest.approx(1 / 100)

    def test_no_samples_is_an_error(self):
        with pytest.raises(EvaluationError, match="no bootstrap samples"):
            bootstrap_two_sided_p([])


class TestTheFamilySignTest:
    def test_the_exact_two_sided_value_matches_the_binomial_by_hand(self):
        values = {f"f{i}": 1.0 for i in range(9)}
        values.update({f"g{i}": -1.0 for i in range(1)})
        result = family_sign_test(values)
        # n = 10, k = 9: 2 * P(X >= 9) = 2 * (10 + 1) / 2**10
        assert result["p_value"] == pytest.approx(2 * 11 / 2 ** 10)
        assert result["n_positive"] == 9
        assert result["n_negative"] == 1
        assert result["n_tested"] == 10
        assert result["majority_sign"] == "positive"

    def test_an_even_split_is_maximally_uninformative(self):
        values = {f"f{i}": 1.0 for i in range(5)}
        values.update({f"g{i}": -1.0 for i in range(5)})
        result = family_sign_test(values)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["majority_sign"] == "tied"

    def test_ties_are_dropped_and_counted(self):
        # A family exactly at zero carries no direction. Counting it as
        # negative would turn a tie into evidence, and silently dropping it
        # would hide how much of the panel that was.
        values = {"a": 1.0, "b": 1.0, "c": -1.0, "d": 0.0, "e": 0.0}
        result = family_sign_test(values)
        assert result["n_families"] == 5
        assert result["n_ties_dropped"] == 2
        assert result["tie_families"] == ["d", "e"]
        assert result["n_tested"] == 3
        assert result["p_value"] == pytest.approx(1.0)

    def test_a_panel_that_is_all_ties_has_nothing_to_test(self):
        result = family_sign_test({"a": 0.0, "b": 0.0})
        assert result["p_value"] == 1.0
        assert result["n_tested"] == 0
        assert "no direction to test" in result["note"]

    def test_no_families_is_an_error(self):
        with pytest.raises(EvaluationError, match="no family-level values"):
            family_sign_test({})

    def test_the_sign_test_and_the_mean_test_can_disagree(self):
        # The reason both are reported. One large family carries the mean while
        # a majority of families point the other way: the bootstrap says the
        # panel mean is positive, the sign test says most families are not.
        values = {f"f{i}": -0.01 for i in range(9)}
        values["big"] = 5.0
        mean = sum(values.values()) / len(values)
        assert mean > 0
        assert family_sign_test(values)["majority_sign"] == "negative"
        assert family_sign_test(values)["p_value"] == pytest.approx(
            2 * sum(math.comb(10, i) for i in range(0, 2)) / 2 ** 10)


class TestHolmBonferroni:
    def test_the_step_down_stops_at_the_first_failure(self):
        # C's own comparison would pass (1 * 0.045 <= 0.05) but B's fails, and
        # Holm retains everything after the first failure. Continuing anyway
        # would be per-test Bonferroni and would not control the family-wise
        # rate the protocol pre-declared.
        result = holm_bonferroni({"A": 0.01, "B": 0.04, "C": 0.045}, 0.05)
        assert result["order"] == ["A", "B", "C"]
        assert result["rejected"] == ["A"]
        assert result["retained"] == ["B", "C"]
        assert result["critical_values"] == {"A": 0.05 / 3, "B": 0.05 / 2,
                                             "C": 0.05}

    def test_the_textbook_four_test_adjustment(self):
        result = holm_bonferroni(
            {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04}, 0.05)
        assert result["n_tests"] == 4
        assert result["adjusted_p"]["A"] == pytest.approx(0.04)
        assert result["adjusted_p"]["B"] == pytest.approx(0.06)
        # Monotone enforcement: C and D inherit B's larger adjusted value
        # rather than falling back to their own smaller product.
        assert result["adjusted_p"]["C"] == pytest.approx(0.06)
        assert result["adjusted_p"]["D"] == pytest.approx(0.06)
        assert result["rejected"] == ["A"]

    def test_adjusted_p_selects_exactly_the_rejected_set(self):
        p_values = {"H1": 0.0002, "H2": 0.03, "H3": 0.2, "H4": 0.9}
        result = holm_bonferroni(p_values, 0.05)
        assert sorted(n for n in p_values
                      if result["adjusted_p"][n] <= 0.05) == \
            result["rejected"]

    def test_all_four_can_be_rejected(self):
        result = holm_bonferroni(
            {"H1": 0.001, "H2": 0.002, "H3": 0.003, "H4": 0.004}, 0.05)
        assert result["n_rejected"] == 4
        assert result["retained"] == []
        # The monotone enforcement binds: H4's own product is 1 * 0.004, but
        # it inherits the largest one seen so far, H2's 3 * 0.002.
        assert result["adjusted_p"]["H4"] == pytest.approx(0.006)

    def test_a_null_model_does_not_shield_the_others_from_the_correction(self):
        # m stays 4 because the protocol counts four confirmatory tests, not
        # four significant ones. Dropping the null model would shrink every
        # critical value and make the other three easier to reject -- the
        # retroactive selection the multiplicity clause forbids.
        result = holm_bonferroni(
            {"H1": 0.001, "H2": 0.002, "H3": 0.003, "H4": 0.9}, 0.05)
        assert result["n_tests"] == 4
        assert result["rejected"] == ["H1", "H2", "H3"]
        assert result["critical_values"]["H1"] == pytest.approx(0.05 / 4)
        assert result["adjusted_p"]["H4"] == pytest.approx(0.9)

    def test_the_ordering_does_not_depend_on_insertion_order(self):
        one = holm_bonferroni({"H1": 0.04, "H2": 0.01, "H3": 0.02}, 0.05)
        two = holm_bonferroni({"H3": 0.02, "H1": 0.04, "H2": 0.01}, 0.05)
        assert one == two
        assert one["order"] == ["H2", "H3", "H1"]

    def test_ties_are_broken_by_name(self):
        result = holm_bonferroni({"B": 0.01, "A": 0.01}, 0.05)
        assert result["order"] == ["A", "B"]

    def test_adjusted_p_is_capped_at_one(self):
        result = holm_bonferroni({"A": 0.6, "B": 0.9}, 0.05)
        assert result["adjusted_p"]["A"] == 1.0
        assert result["adjusted_p"]["B"] == 1.0
        assert result["rejected"] == []

    def test_a_single_test_is_its_own_family(self):
        result = holm_bonferroni({"H1": 0.03}, 0.05)
        assert result["adjusted_p"]["H1"] == pytest.approx(0.03)
        assert result["rejected"] == ["H1"]

    def test_no_p_values_is_an_error(self):
        with pytest.raises(EvaluationError, match="no p-values"):
            holm_bonferroni({})

    def test_an_alpha_outside_the_unit_interval_is_an_error(self):
        for alpha in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(EvaluationError, match="alpha must be"):
                holm_bonferroni({"A": 0.01}, alpha)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "0.01", None])
    def test_a_p_value_outside_the_unit_interval_is_an_error(self, bad):
        with pytest.raises(EvaluationError, match="must be a number"):
            holm_bonferroni({"A": bad})
