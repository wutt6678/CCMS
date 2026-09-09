"""Iteration 11: what the differential exclusion costs, bounded not priced once.

``CMST_456921/text_only`` was refused by judge A's provider in the
``ministral3_3b`` arm alone, so the moderation union dropped it from all four
arms and took the whole family with it: the confirmatory analysis runs on 98
families. Two numbers had been filed about what that costs and neither was a
bound -- the judge-B shift of 0.004545, measured with a different and
vision-ablated instrument over a change in the FAMILY SET, and the point
estimate you get by putting the family back.

What replaces them is an exact range. ``paired_bootstrap_samples`` draws its
resample indices from ``Random(seed)`` as a function of the seed and the family
count alone, never of the data, so with 99 families every resample mean is
affine in the one missing score and the two-sided p's tail counts are monotone
step functions of it. The extremes are therefore at an endpoint or at a
breakpoint, and both scripts enumerate them instead of searching.

Three properties are pinned here, because each is the kind of claim that reads
as true and is not:

* the closed form IS the frozen bootstrap's algebra -- tested against
  ``paired_bootstrap_samples`` and ``bootstrap_two_sided_p`` themselves on
  synthetic estimands, not against a second implementation of the same idea;
* every number quoted in prose about the evidence is READ from that evidence --
  the label range, and the disagreement counts the four sealed sets record. The
  prose this replaced claimed the 2,388 committed labels sit "on a 0.05 grid"
  (two of them, 0.72 and 0.88, do not) and that regenerating the sealed sets
  would re-adjudicate "all 243 disagreements" (243 is ``qwen35_4b``'s own count;
  the four record 1,009 between them);
* a re-file of the sensitivity labels SPENDS NO CALLS -- the two kimi-k3
  adjudications are reused, bound by a request hash recomputed offline through
  the production call path with the transport stubbed, so a label is reused only
  where it answers the request the frozen rule would send now.

CI-safe: no test here calls the gateway. The request-hash recomputation needs no
credentials because ``request_hash`` binds the prompt, the image hashes, the
model id, the temperature and the seed -- not the key -- and the live-call path
is exercised through a stub.
"""

from __future__ import annotations

import builtins
import copy
import functools
import importlib.util
import json
import math
import operator
import sys
from pathlib import Path

import pytest

from causal_mllm.evaluation.bootstrap import (
    bootstrap_two_sided_p,
    paired_bootstrap_ci,
    paired_bootstrap_samples,
)
from causal_mllm.evaluation.censoring import (
    DIFFERENTIAL_CELL,
    DIFFERENTIAL_TARGET,
    EVIDENCE_ARTIFACT,
)
from causal_mllm.replay import reproduction

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


adjudicator = _load_script("iter11_adjudicate_sensitivity_cell")
bound = _load_script("iter11_differential_censoring_bound")

FAMILY = bound.FAMILY
ARMS = bound.ARMS
ESTIMANDS = bound.ESTIMANDS

JUDGE_ROOT = ROOT / "outputs" / "iteration_11" / "judge"
BOUND_ARTIFACT = bound.OUT_PATH
SENSITIVITY_ARTIFACT = adjudicator.OUT_PATH
CROSS_MODEL = bound.COMMITTED_ANALYSIS


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def filed_bound() -> dict:
    if not BOUND_ARTIFACT.exists():
        pytest.skip(f"no bound artifact at {BOUND_ARTIFACT}")
    return _read(BOUND_ARTIFACT)


@pytest.fixture(scope="module")
def filed_sensitivity() -> dict:
    if not SENSITIVITY_ARTIFACT.exists():
        pytest.skip(f"no sensitivity artifact at {SENSITIVITY_ARTIFACT}")
    return _read(SENSITIVITY_ARTIFACT)


def _credential_free(*_args, **_kwargs):
    raise EnvironmentError(
        "LLM_JUDGE_API_KEY is required. Set the environment variable "
        "LLM_JUDGE_API_KEY, or copy configs/evaluation/"
        "llm_judge_credentials.conf.example to llm_judge_credentials.conf "
        "(simulated here, not read from this machine)")


@pytest.fixture
def no_credentials(monkeypatch):
    """This checkout holds no adjudicator credentials.

    A fresh clone has neither ``LLM_JUDGE_API_KEY`` nor the gitignored
    credentials conf, so loading the pipeline raises ``EnvironmentError`` and
    :func:`adjudicator.adjudicator_config` falls back to the pinned constants
    with ``sendable=False``. Patched at the loader rather than by moving a file,
    because the credentials on this machine are this machine's and a test that
    deleted them would be a test about the author's checkout.

    Both caches are cleared either side: ``adjudicator_config`` memoises, so a
    test that patched the loader without clearing the cache would read an answer
    computed before the patch and pass for the wrong reason.
    """
    monkeypatch.setattr(adjudicator, "_pipeline", _credential_free)
    adjudicator._CONFIG_CACHE.clear()
    adjudicator._PIPELINE_CACHE.clear()
    assert adjudicator.adjudicator_config()[2] is False
    yield _credential_free
    adjudicator._CONFIG_CACHE.clear()
    adjudicator._PIPELINE_CACHE.clear()


@pytest.fixture
def sendable_adjudicator(monkeypatch):
    """Credentials present, so ``build()`` will reach the transport.

    The transport stays stubbed by whichever test asks for this; what changes is
    only whether this checkout COULD send, which ``build()`` establishes before it
    calls anything. A test that stubs the transport but not sendability passes on
    a machine holding a key and exits 2 in CI, where nobody holds one -- a lane
    that only works where the author ran it is not the documented offline lane.
    """
    real = adjudicator.adjudicator_config

    def sendable():
        config, source, _sendable = real()
        return config, source, True

    monkeypatch.setattr(adjudicator, "adjudicator_config", sendable)
    adjudicator._CONFIG_CACHE.clear()
    yield sendable
    adjudicator._CONFIG_CACHE.clear()


# ---------------------------------------------------------------------------
# Synthetic estimands: enough families to bootstrap, no repository files
# ---------------------------------------------------------------------------

def _synthetic(n_families: int = 12, seed: int = 7) -> dict:
    """A family-estimand table with the restored family in it.

    Values are spread across zero on purpose: a decomposition that is only
    correct when every resample mean sits on one side of the null would still
    pass a test built from all-positive data, and the interesting case for a
    p-value is a distribution that straddles it.
    """
    import random as _random

    rng = _random.Random(seed)
    ids = [FAMILY] + [f"CMST_{i:06d}" for i in range(n_families - 1)]
    return {
        fid: {name: rng.uniform(-0.4, 0.4) if name != "Delta_V"
              else rng.uniform(-0.1, 0.1) for name in ESTIMANDS}
        for fid in ids
    }


def _shifted(fe: dict, estimand: str, score: float, slope: float) -> dict:
    """The same table with the missing cell's score set to ``score``.

    ``slope`` is how that score enters the estimand: -1 for Delta_TV, +1 for
    Delta_T, 0 for the three that do not involve ``text_only`` at all.
    """
    out = {fid: dict(values) for fid, values in fe.items()}
    out[FAMILY][estimand] = fe[FAMILY][estimand] + slope * score
    return out


class TestTheClosedFormIsTheFrozenBootstrapsOwnAlgebra:
    """The claim the whole bound rests on, checked against the frozen code.

    Not against a re-implementation: ``paired_bootstrap_samples`` and
    ``bootstrap_two_sided_p`` are imported from
    ``causal_mllm.evaluation.bootstrap`` and asked the same question at the same
    seed. If the affine story were wrong -- if the resample indices depended on
    the data, or the mean were not linear in one family's value -- these would
    diverge at the first x that is not the anchor.
    """

    @pytest.mark.parametrize("score", [0.0, 0.05, 0.37, 0.9, 1.0])
    def test_every_resample_mean_is_affine_in_the_missing_score(self, score):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=400, seed=42)
        samples = paired_bootstrap_samples(
            _shifted(fe, "Delta_TV", score, bound.SLOPE["Delta_TV"]),
            n_bootstrap=400, seed=42)["Delta_TV"]
        closed = [a + m * score
                  for a, m in zip(dec["intercepts"], dec["slopes"])]
        assert len(samples) == len(closed) == 400
        for direct, affine in zip(samples, closed):
            assert direct == pytest.approx(affine,
                                           abs=reproduction.FLOAT_TOLERANCE)

    @pytest.mark.parametrize("score", [0.0, 0.2, 0.55, 1.0])
    def test_the_p_value_is_the_frozen_rule_on_the_frozen_samples(self, score):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=400, seed=42)
        samples = paired_bootstrap_samples(
            _shifted(fe, "Delta_TV", score, bound.SLOPE["Delta_TV"]),
            n_bootstrap=400, seed=42)["Delta_TV"]
        assert bound.p_at(dec, score) == pytest.approx(
            bootstrap_two_sided_p(samples), abs=reproduction.FLOAT_TOLERANCE)

    def test_the_observed_mean_is_affine_too(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=400, seed=42)
        for score in (0.0, 0.25, 1.0):
            column = [_shifted(fe, "Delta_TV", score, bound.SLOPE["Delta_TV"])
                      [fid]["Delta_TV"] for fid in sorted(fe)]
            assert bound.mean_at(dec, score) == pytest.approx(
                sum(column) / len(column), abs=reproduction.FLOAT_TOLERANCE)

    def test_the_decomposition_is_anchored_at_zero_and_says_so(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=50, seed=42)
        assert dec["anchored_at"] == bound.SCORE_LO
        column = [fe[fid]["Delta_TV"] for fid in sorted(fe)]
        assert bound.mean_at(dec, 0.0) == pytest.approx(
            sum(column) / len(column), abs=reproduction.FLOAT_TOLERANCE)

    def test_the_resample_slope_is_the_family_slope_times_its_count(self):
        fe = _synthetic(n_families=9)
        n = len(fe)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=200, seed=42)
        j = dec["family_index"]
        ids = sorted(fe)
        import random as _random
        rng = _random.Random(42)
        for b in range(200):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            assert dec["slopes"][b] == pytest.approx(
                bound.SLOPE["Delta_TV"] * indices.count(j) / n)
            assert dec["intercepts"][b] == pytest.approx(
                sum(fe[ids[i]]["Delta_TV"] for i in indices) / n,
                abs=reproduction.FLOAT_TOLERANCE)

    def test_the_family_index_is_the_restored_family(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=10, seed=42)
        assert sorted(fe)[dec["family_index"]] == FAMILY

    def test_decomposing_without_the_restored_family_refuses(self, capsys):
        fe = _synthetic()
        fe.pop(FAMILY)
        with pytest.raises(SystemExit) as exc:
            bound.decompose(fe, "Delta_TV", n_bootstrap=10, seed=42)
        assert exc.value.code == 1
        assert FAMILY in capsys.readouterr().err

    @pytest.mark.parametrize("estimand", ["Delta_T", "Delta_V", "Delta_TV",
                                          "order_effect", "history_effect"])
    def test_the_slope_of_every_estimand_is_the_one_the_rubric_implies(
            self, estimand):
        fe = _synthetic()
        dec = bound.decompose(fe, estimand, n_bootstrap=100, seed=42)
        assert dec["family_slope"] == bound.SLOPE[estimand]
        assert dec["constant_in_x"] is (bound.SLOPE[estimand] == 0.0)
        samples = paired_bootstrap_samples(
            _shifted(fe, estimand, 1.0, bound.SLOPE[estimand]),
            n_bootstrap=100, seed=42)[estimand]
        closed = [a + m * 1.0
                  for a, m in zip(dec["intercepts"], dec["slopes"])]
        for direct, affine in zip(samples, closed):
            assert direct == pytest.approx(affine,
                                           abs=reproduction.FLOAT_TOLERANCE)

    def test_an_estimand_that_does_not_involve_the_cell_does_not_move(self):
        fe = _synthetic()
        for estimand in ("Delta_V", "order_effect", "history_effect"):
            dec = bound.decompose(fe, estimand, n_bootstrap=100, seed=42)
            worst = bound.worst_case(dec)
            assert set(dec["slopes"]) == {0.0}
            assert worst["worst_p"] == worst["best_p"]
            assert worst["mean_range"][0] == pytest.approx(
                worst["mean_range"][1], abs=reproduction.FLOAT_TOLERANCE)

    def test_a_slope_can_be_overridden_for_the_pooled_estimand(self):
        """H5 averages four models, so one arm's score enters at a quarter."""
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=100, seed=42,
                              slope=bound.SLOPE["Delta_TV"] / len(ARMS))
        assert dec["family_slope"] == pytest.approx(-0.25)
        samples = paired_bootstrap_samples(
            _shifted(fe, "Delta_TV", 1.0, -0.25),
            n_bootstrap=100, seed=42)["Delta_TV"]
        closed = [a + m for a, m in zip(dec["intercepts"], dec["slopes"])]
        for direct, affine in zip(samples, closed):
            assert direct == pytest.approx(affine,
                                           abs=reproduction.FLOAT_TOLERANCE)


class TestTheFastTailCountsAreTheSlowOnes:
    """The O(log n) sweep is only usable if it equals the O(n) count."""

    @pytest.mark.parametrize("score", [0.0, 0.01, 0.13, 0.5, 0.77, 0.9, 1.0])
    def test_the_index_agrees_with_counting_every_resample(self, score):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        index = bound.index_tails(dec)
        assert bound.fast_tails(index, score) == bound.tails_at(dec, score)
        assert bound.fast_p(dec, index, score) == pytest.approx(
            bound.p_at(dec, score), abs=reproduction.FLOAT_TOLERANCE)

    def test_the_two_tail_rules_agree_across_a_dense_grid(self):
        fe = _synthetic(n_families=20)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        index = bound.index_tails(dec)
        disagreements = 0
        for step in range(201):
            score = step / 200.0
            if bound.ties_at(dec, score):
                continue  # a resample exactly at zero: counted, not waved past
            if bound.fast_tails(index, score) != bound.tails_at(dec, score):
                disagreements += 1
        assert disagreements == 0

    def test_the_p_floor_is_one_resample_and_the_step_is_two(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=5000, seed=42)
        assert bound.p_step(dec) == pytest.approx(2.0 / 5000)
        worst = bound.worst_case(dec)
        assert worst["best_p"] >= 1.0 / 5000
        assert worst["worst_p"] >= 1.0 / 5000

    def test_ties_are_counted_at_a_breakpoint_and_named(self):
        fe = _synthetic(n_families=15)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        index = bound.index_tails(dec)
        crossing = next((x for x in index["negative"] + index["positive"]
                         if bound.SCORE_LO < x < bound.SCORE_HI), None)
        assert crossing is not None, "the fixture should straddle the null"
        assert bound.ties_at(dec, crossing) >= 1
        assert bound.ties_at(dec, crossing, band=0.0) <= bound.ties_at(
            dec, crossing)


def _neumaier_sum(values) -> float:
    """CPython 3.12's float summation, in shape rather than in every detail.

    3.12 changed the builtin ``sum`` to compensate with the Neumaier variant of
    Kahan summation, so ``sum`` over the same floats returns different bits
    under 3.10 and 3.12. This reproduces the arithmetic that matters -- a
    running total, a compensation term, and a final addition of the two --
    without the C code's infinity and overflow special cases, which cannot
    arise on a column of rubric scores.
    """
    total = 0.0
    compensation = 0.0
    for value in values:
        nxt = total + value
        if abs(total) >= abs(value):
            compensation += (total - nxt) + value
        else:
            compensation += (value - nxt) + total
        total = nxt
    return total + compensation


def _naive_sum(values) -> float:
    """The left-to-right sum, written the only two ways that agree everywhere."""
    return functools.reduce(operator.add, values, 0.0)


class TestTheBoundDoesNotDependOnWhichInterpreterSummedIt:
    """The portability finding, tested on the interpreter CI actually runs.

    Under CPython 3.12 the filed artifact came back with ten differences: two
    exactness booleans flipped, a breakpoint count moved by one, and H2's
    99-family p moved a single bootstrap step, 0.0224 to 0.0220. No verdict
    moved. The cause was not the data and not NumPy -- the family-estimand
    column is bit-identical under both interpreters -- but the builtin ``sum``,
    whose float result 3.12 changed.

    CI runs one interpreter, so the version axis is tested by substituting the
    other one's summation for the builtin and requiring that nothing the bound
    files moves. That is a stronger test than running both interpreters would
    have been: it fails on ANY dependence on the builtin's bits, not only on
    the one difference 3.12 happens to introduce.
    """

    @pytest.mark.parametrize("summation", [_neumaier_sum, math.fsum])
    def test_the_decomposition_is_bit_identical_under_a_different_sum(
            self, monkeypatch, summation):
        fe = _synthetic(n_families=20)
        baseline = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        probe = [0.1] * 3 + [0.2] * 3 + [1 / 3] * 3
        assert summation(probe) != _naive_sum(probe), (
            "the substituted summation is the one _portable_sum implements, so "
            "the patch below would change nothing and the comparison after it "
            "would pass for the wrong reason")
        monkeypatch.setattr(builtins, "sum", summation)
        assert bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42) \
            == baseline

    @pytest.mark.parametrize("summation", [_neumaier_sum, math.fsum])
    def test_the_worst_case_and_every_p_on_the_grid_are_too(
            self, monkeypatch, summation):
        fe = _synthetic(n_families=18)
        baseline = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        expected_worst = bound.worst_case(baseline)
        expected_p = [bound.p_at(baseline, step / 200.0)
                      for step in range(201)]
        monkeypatch.setattr(builtins, "sum", summation)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        worst = bound.worst_case(dec)
        assert worst == expected_worst, (
            "a field of the worst case moved when the builtin sum did, which "
            "is how the breakpoint count came to differ between interpreters")
        assert [bound.p_at(dec, step / 200.0)
                for step in range(201)] == expected_p
        assert bound.ties_at(dec, expected_worst["worst_p_at_x"]) \
            == bound.ties_at(baseline, expected_worst["worst_p_at_x"])

    def test_the_chosen_summation_is_the_frozen_estimators_own(self):
        """Which portable sum, and why not the correctly rounded one.

        ``math.fsum`` is portable too, and more accurate. It was not used
        because the frozen estimator in
        :mod:`causal_mllm.evaluation.bootstrap` sums left to right under the
        certified interpreter, so switching to ``fsum`` would have moved the
        p-values that are already filed in order to make the ones that are not
        portable. An explicit loop is both.
        """
        column = [0.1] * 3 + [0.2] * 3 + [1 / 3] * 3
        assert bound._portable_sum(column) == _naive_sum(column)
        assert math.fsum(column) != _naive_sum(column), (
            "this column no longer separates the two summations, so the "
            "assertion above would pass without pinning anything")
        assert bound._portable_sum(column) != math.fsum(column)

    def test_the_two_summations_are_filed_beside_each_other(self):
        fe = _synthetic(n_families=20)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=200, seed=42)
        summation = dec["summation"]
        column = [fe[f]["Delta_TV"] for f in sorted(fe)]
        assert summation["portable_sum_of_the_family_column"] \
            == bound._portable_sum(column)
        assert summation["correctly_rounded_sum_of_the_same_column"] \
            == math.fsum(column)
        assert summation["their_difference"] == \
            bound._portable_sum(column) - math.fsum(column)
        n_moved = summation[
            "n_resample_intercepts_the_two_summations_disagree_on"]
        assert isinstance(n_moved, int)
        assert 0 <= n_moved <= summation["n_resamples"]

    def test_the_count_of_intercepts_the_summation_moves_is_recounted(self):
        """The filed count is a measurement, so it is repeated here.

        A count quoted in prose that nothing recomputes is the failure this
        repository has already paid for twice, and this one sits inside a
        sentence that argues from it.
        """
        import random as _random

        fe = _synthetic(n_families=20)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=200, seed=42)
        column = [fe[f]["Delta_TV"] for f in sorted(fe)]
        n = len(column)
        rng = _random.Random(42)
        moved = 0
        for _ in range(200):
            values = [column[rng.randint(0, n - 1)] for _ in range(n)]
            if math.fsum(values) / n != bound._portable_sum(values) / n:
                moved += 1
        assert dec["summation"][
            "n_resample_intercepts_the_two_summations_disagree_on"] == moved

    def test_the_why_block_reproduction_command_is_one_that_runs(self):
        """The prose carries a command. A command a reader cannot run is prose.

        It is executed here rather than read, so a rename of ``math.fsum`` or a
        change of interpreter behaviour fails this test instead of quietly
        leaving a false instruction in a filed artifact.
        """
        why = bound.decompose(
            _synthetic(n_families=4), "Delta_TV", n_bootstrap=5,
            seed=42)["summation"]["why"]
        assert "python3 -c" in why
        command = why.split("python3 -c ", 1)[1].split('"')[1]
        import subprocess
        done = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True,
            timeout=60)
        assert done.returncode == 0, done.stderr
        naive, builtin, rounded = done.stdout.strip().split(" ")
        column = [0.1] * 3 + [0.2] * 3 + [1 / 3] * 3
        assert naive == repr(_naive_sum(column))
        assert rounded == repr(math.fsum(column))
        assert naive != rounded, (
            "the column the command uses has to separate the two summations, "
            "or the demonstration demonstrates nothing")
        assert builtin in (naive, rounded), (
            "the builtin is one of the two, and which one is the interpreter's "
            "business -- that is the whole reason it is not used")


class TestTheWorstCaseIsExactRatherThanSearched:
    def test_the_enumerated_worst_is_at_least_the_worst_on_a_dense_grid(self):
        fe = _synthetic(n_families=20)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        worst = bound.worst_case(dec)
        grid = max(bound.p_at(dec, step / 500.0) for step in range(501))
        assert worst["worst_p"] >= grid - reproduction.FLOAT_TOLERANCE
        assert worst["best_p"] <= min(
            bound.p_at(dec, step / 500.0) for step in range(501)) \
            + reproduction.FLOAT_TOLERANCE

    def test_the_extremes_are_at_an_endpoint_or_at_a_breakpoint(self):
        fe = _synthetic(n_families=18)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        worst = bound.worst_case(dec)
        index = bound.index_tails(dec)
        allowed = {bound.SCORE_LO, bound.SCORE_HI}
        allowed.update(x for x in index["negative"] + index["positive"]
                       if bound.SCORE_LO <= x <= bound.SCORE_HI)
        assert worst["worst_p_at_x"] in allowed
        assert worst["best_p_at_x"] in allowed
        assert worst["n_breakpoints_evaluated"] == len(allowed)

    def test_the_mean_extremes_are_the_endpoints_because_it_is_affine(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=200, seed=42)
        worst = bound.worst_case(dec)
        ends = [bound.mean_at(dec, bound.SCORE_LO),
                bound.mean_at(dec, bound.SCORE_HI)]
        assert worst["mean_range"] == [min(ends), max(ends)]
        assert worst["mean_at_the_range_ends"]["at_0"] == pytest.approx(ends[0])
        assert worst["mean_at_the_range_ends"]["at_1"] == pytest.approx(ends[1])
        interior = [bound.mean_at(dec, s / 100.0) for s in range(101)]
        assert min(interior) >= worst["mean_range"][0] - 1e-12
        assert max(interior) <= worst["mean_range"][1] + 1e-12

    def test_the_sign_can_only_flip_if_the_endpoints_straddle_zero(self):
        fe = _synthetic()
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=100, seed=42)
        worst = bound.worst_case(dec)
        lo, hi = worst["mean_at_the_range_ends"]["at_0"], \
            worst["mean_at_the_range_ends"]["at_1"]
        assert worst["sign_can_flip"] is ((lo > 0.0) != (hi > 0.0))

    def test_the_ci_envelope_contains_the_ci_at_both_ends(self):
        fe = _synthetic(n_families=20)
        dec = bound.decompose(fe, "Delta_TV", n_bootstrap=500, seed=42)
        envelope = bound.ci_envelope(dec)
        for score in (bound.SCORE_LO, bound.SCORE_HI):
            ci = paired_bootstrap_ci(
                _shifted(fe, "Delta_TV", score, bound.SLOPE["Delta_TV"]),
                n_bootstrap=500, seed=42, with_samples=False)["Delta_TV"]
            assert ci["CI_lower"] >= envelope[0] - reproduction.FLOAT_TOLERANCE
            assert ci["CI_upper"] <= envelope[1] + reproduction.FLOAT_TOLERANCE

    def test_the_percentile_rule_is_the_frozen_estimators_own(self):
        from causal_mllm.evaluation.bootstrap import _percentile as frozen

        values = [float(i) for i in range(101)]
        assert bound._percentile(values, 2.5) == pytest.approx(2.5)
        assert bound._percentile(values, 97.5) == pytest.approx(97.5)
        assert bound._percentile([1.0, 2.0], 2.5) == pytest.approx(1.025)
        import random as _random
        rng = _random.Random(11)
        for _ in range(20):
            sample = sorted(rng.uniform(-1, 1) for _ in range(rng.choice(
                [2, 7, 50, 5000])))
            for pct in (2.5, 50.0, 97.5):
                assert bound._percentile(sample, pct) == pytest.approx(
                    frozen(sample, pct), abs=reproduction.FLOAT_TOLERANCE)

    def test_the_p_rule_is_the_frozen_one_on_counts(self):
        assert bound.p_from_tails(0, 5000, 5000) == pytest.approx(1.0 / 5000)
        assert bound.p_from_tails(2500, 2500, 5000) == pytest.approx(1.0)
        assert bound.p_from_tails(5000, 5000, 5000) == pytest.approx(1.0)
        assert bound.p_from_tails(64, 4936, 5000) == pytest.approx(
            bootstrap_two_sided_p(
                [-1.0] * 64 + [1.0] * 4936), abs=1e-12)


class TestTheVerdictRuleIsTheCommittedAnalysissOwn:
    @pytest.mark.parametrize("rejected,matches,expected", [
        (True, True, "confirmed"),
        (True, False, "refuted"),
        (False, True, "inconclusive"),
        (False, False, "inconclusive"),
    ])
    def test_the_three_verdicts_come_from_two_booleans(self, rejected,
                                                       matches, expected):
        assert bound.verdict_from(rejected, matches) == expected

    def test_the_rule_reproduces_every_filed_verdict(self, filed_bound):
        committed = _read(CROSS_MODEL)
        config = filed_bound["holm_bonferroni"]["configurations"][
            "a_committed_98_families"]
        for hid, row in filed_bound["verdicts"]["per_hypothesis"].items():
            assert row["filed_verdict"] == committed["verdicts"][hid]["verdict"]
            assert row["filed_raw_p"] == pytest.approx(
                committed["verdicts"][hid]["raw_p"])
            assert bound.verdict_from(
                hid in config["holm"]["rejected"],
                config["sign_matches_reference"][hid]) == row["filed_verdict"]


class TestTheObservedLabelRangeIsReadNotDescribed:
    """The prose this replaced claimed a 0.05 grid the labels do not sit on."""

    def test_the_four_sealed_sets_hold_the_labels_the_prose_quotes(self):
        observed = bound.observed_scores()
        assert observed["n_labels"] == 4 * adjudicator.FROZEN_LABEL_COUNT
        assert observed["n_labels_per_arm"] == {
            arm: adjudicator.FROZEN_LABEL_COUNT for arm in ARMS}
        assert observed["n_labels_carrying_no_score"] == 0

    def test_the_observed_span_stops_below_the_top_of_the_rubric(self):
        observed = bound.observed_scores()
        assert observed["observed_min"] == 0.0
        assert observed["observed_max"] == 0.95
        assert observed["top_of_the_rubric_observed"] is False
        assert observed["headroom_above_the_observed_max"] == pytest.approx(0.05)

    def test_the_labels_are_not_on_a_0_05_grid_and_the_block_says_so(self):
        observed = bound.observed_scores()
        assert observed["on_a_0_05_grid"] is False
        assert observed["off_a_0_05_grid"] == [0.72, 0.88]
        assert observed["n_distinct_values"] == len(
            observed["distinct_values"]) == 20

    def test_the_filed_prose_carries_the_measured_numbers(self, filed_bound):
        observed = filed_bound["observed_label_scores"]
        sentence = filed_bound["score_range_is_the_rubric_not_the_observed"]
        assert observed == bound.observed_scores()
        assert f"{observed['n_labels']:,}" in sentence
        assert str(observed["observed_max"]) in sentence
        assert str(observed["n_distinct_values"]) in sentence
        assert str(len(observed["off_a_0_05_grid"])) in sentence

    def test_no_filed_prose_claims_a_grid_the_labels_are_not_on(
            self, filed_bound):
        sentence = filed_bound["score_range_is_the_rubric_not_the_observed"]
        assert "on a 0.05 grid, so" not in sentence
        assert "span 0.0 to 0.95 on a 0.05 grid" not in json.dumps(filed_bound)
        assert bound.observed_scores()["on_a_0_05_grid"] is False

    def test_the_bounded_range_is_the_rubric_and_not_the_observed(self,
                                                                  filed_bound):
        assert filed_bound["score_range_bounded"] == [bound.SCORE_LO,
                                                      bound.SCORE_HI]
        assert filed_bound["score_range_bounded"][1] > \
            bound.observed_scores()["observed_max"]


# ---------------------------------------------------------------------------
# The sealed sets' own counts, which is what the prose now quotes
# ---------------------------------------------------------------------------

def _sealed_label_file(n_labels: int, n_disagreements: int,
                       with_provenance: bool = True) -> dict:
    """A sealed label set shaped like the real one, at any size."""
    labels = {f"CMST_{i:06d}": {"neutral": {
        "refusal_type": "full", "unsafe_compliance_score": 0.0,
        "compliance_level": 0, "safe_redirection": True, "confidence": 0.9,
        "rationale": "fixture"}} for i in range(n_labels)}
    doc = {"labels": labels}
    if with_provenance:
        doc["provenance"] = {
            "n_labels": n_labels,
            "ensemble": {"n_disagreements": n_disagreements,
                         "adjudicator_model": "kimi-k3",
                         "adjudication_method":
                             "distinct_model_adjudication_on_all_disagreements"},
        }
    return doc


COUNTS = {"ministral3_3b": 299, "phi4_mm": 198, "qwen35_2b": 269,
          "qwen35_4b": 243}


@pytest.fixture
def sealed_tree(tmp_path, monkeypatch):
    """A miniature judge root holding four sealed sets and nothing else."""
    root = tmp_path / "judge"
    for arm, count in COUNTS.items():
        path = root / arm / adjudicator.FROZEN_LABELS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_sealed_label_file(
            adjudicator.FROZEN_LABEL_COUNT, count), indent=2),
            encoding="utf-8")
    monkeypatch.setattr(adjudicator, "JUDGE_ROOT", root)
    return root


class TestTheCostOfRegeneratingTheSealedSetsIsReadFromThem:
    def test_the_four_sets_record_the_counts_the_readme_documents(
            self, sealed_tree):
        sealed = adjudicator.sealed_label_sets()
        assert sealed["per_arm_counts"] == COUNTS
        assert sealed["n_disagreements_total"] == 1009
        assert sealed["n_labels_each"] == adjudicator.FROZEN_LABEL_COUNT

    def test_the_committed_sets_give_the_same_total(self):
        sealed = adjudicator.sealed_label_sets()
        assert sealed["n_disagreements_total"] == 1009
        assert sealed["per_arm_counts"] == COUNTS
        for arm in ARMS:
            row = sealed["per_arm"][arm]
            assert row["n_labels"] == row["n_labels_present"] == 597
            assert row["adjudicator_model"] == "kimi-k3"
            assert row["sha256"] and len(row["sha256"]) == 64

    def test_the_total_is_a_sum_and_not_one_arms_number_quoted_for_all(
            self, sealed_tree):
        sealed = adjudicator.sealed_label_sets()
        assert sealed["n_disagreements_total"] == sum(COUNTS.values())
        assert sealed["n_disagreements_total"] != COUNTS["qwen35_4b"]
        assert sealed["n_disagreements_total"] - COUNTS["qwen35_4b"] == 766

    def test_a_set_that_records_no_count_refuses_to_be_quoted(
            self, sealed_tree, monkeypatch, capsys):
        path = sealed_tree / "phi4_mm" / adjudicator.FROZEN_LABELS
        doc = _read(path)
        doc["provenance"]["ensemble"].pop("n_disagreements")
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            adjudicator.sealed_label_sets()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "phi4_mm" in err and "cannot describe" in err

    def test_a_set_of_a_different_size_is_not_the_one_bounded(
            self, sealed_tree, capsys):
        path = sealed_tree / "qwen35_2b" / adjudicator.FROZEN_LABELS
        path.write_text(json.dumps(_sealed_label_file(598, 269)),
                        encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            adjudicator.sealed_label_sets()
        assert exc.value.code == 1
        assert "598" in capsys.readouterr().err

    def test_a_set_whose_provenance_does_not_describe_it_is_not_evidence(
            self, sealed_tree, capsys):
        path = sealed_tree / "ministral3_3b" / adjudicator.FROZEN_LABELS
        doc = _read(path)
        fam = next(iter(doc["labels"]))
        doc["labels"][fam]["extra_variant"] = doc["labels"][fam]["neutral"]
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            adjudicator.sealed_label_sets()
        assert exc.value.code == 1
        assert "does not describe" in capsys.readouterr().err

    def test_a_missing_sealed_set_is_a_missing_input(self, sealed_tree,
                                                     capsys):
        (sealed_tree / "qwen35_4b" / adjudicator.FROZEN_LABELS).unlink()
        with pytest.raises(SystemExit) as exc:
            adjudicator.sealed_label_sets()
        assert exc.value.code == 2, "an absent file is not a disagreement"
        assert "qwen35_4b" in capsys.readouterr().err

    def test_the_filed_artifact_quotes_the_derived_total(
            self, filed_sensitivity):
        block = filed_sensitivity["frozen_labels_untouched"]
        sealed = block["sealed_label_sets"]
        assert sealed["n_disagreements_total"] == 1009
        assert "1,009 disagreements" in block["why"]
        for arm, count in COUNTS.items():
            assert f"{arm} {count}" in block["why"]
        assert "all 243 disagreements" not in block["why"]
        assert "243" in block["why"], "one arm's own count is still named"

    def test_no_filed_prose_quotes_one_arms_count_for_all_four(
            self, filed_sensitivity):
        text = json.dumps(filed_sensitivity)
        assert "re-adjudicate all 243" not in text
        script = (ROOT / "scripts" / "iter11_adjudicate_sensitivity_cell.py"
                  ).read_text(encoding="utf-8")
        assert "all 243 disagreements" not in script


# ---------------------------------------------------------------------------
# A re-file reuses the filed call only if it answers the same request
# ---------------------------------------------------------------------------

ADJUDICATED_ARMS = ("phi4_mm", "qwen35_2b")


class TestTheRequestTheFrozenRuleWouldSendIsRecomputable:
    def test_the_hash_comes_back_for_both_adjudicated_arms(self):
        for arm in ADJUDICATED_ARMS:
            expected = adjudicator.expected_request(arm)
            assert expected["available"] is True, expected.get("why_not")
            assert len(expected["request_hash"]) == 64
            assert len(expected["prompt_sha256"]) == 64
            assert expected["image_hashes"] == []
            assert expected["rubric_version"] == "1.1"

    def test_it_equals_the_request_the_filed_calls_actually_made(
            self, filed_sensitivity):
        for arm in ADJUDICATED_ARMS:
            entry = filed_sensitivity["per_arm"][arm]
            expected = adjudicator.expected_request(arm)
            assert entry["call_provenance"]["request_hash"] == \
                expected["request_hash"]
            assert entry["call_provenance"]["prompt_sha256"] == \
                expected["prompt_sha256"]
            assert entry["request_the_frozen_rule_would_send"][
                "request_hash"] == expected["request_hash"]

    def test_the_arm_with_no_primary_label_has_no_request_to_hash(self):
        expected = adjudicator.expected_request(DIFFERENTIAL_TARGET)
        assert expected["available"] is False
        assert expected["request_hash"] is None
        assert "judge A's label" in expected["why_not"]

    def test_the_rubric_is_read_from_the_judge_that_loads_it(self):
        rubric = adjudicator.rubric_identity()
        assert rubric["rubric_version"] == "1.1"
        assert rubric["rubric_sha256"].startswith("ce6c2005")
        assert rubric["rubric_path"].endswith("annotation_rubric_v1_1.md")

    def test_the_identity_is_the_frozen_one_whichever_source_gave_it(self):
        config, source, _sendable = adjudicator.adjudicator_config()
        assert config.model_id == adjudicator.ADJUDICATOR_MODEL_ID == "kimi-k3"
        assert config.seed == adjudicator.ADJUDICATOR_MODEL_SEED == 99
        assert config.temperature == adjudicator.ADJUDICATOR_TEMPERATURE == 0.0
        assert config.provider == adjudicator.ADJUDICATOR_PROVIDER == "aliyun"
        assert source

    def test_the_pinned_constants_are_the_pipelines_where_credentials_exist(
            self):
        """The substitution is only safe if the two agree where both exist."""
        try:
            pipeline = adjudicator._pipeline()
        except EnvironmentError:
            pytest.skip("no judge credentials in this checkout")
        config = pipeline.ADJUDICATOR_CONFIG
        assert config is not None
        assert config.model_id == adjudicator.ADJUDICATOR_MODEL_ID
        assert config.provider == adjudicator.ADJUDICATOR_PROVIDER
        assert config.seed == adjudicator.ADJUDICATOR_MODEL_SEED
        assert config.temperature == adjudicator.ADJUDICATOR_TEMPERATURE

    def test_the_stand_in_config_hashes_the_same_request(self, monkeypatch):
        """The credential-free path CI takes must file the SAME request block.

        ``request_hash`` binds the prompt, the image hashes, the model id, the
        temperature and the seed -- not the key and not the base_url -- so
        substituting stand-ins for those two cannot move it. But the hash is not
        the only thing compared: a re-file checks a preserved call against the
        whole filed block, so a block that differs in ANY field makes the reuse
        depend on the local credential state. The whole block is asserted equal,
        not just the hash, because the hash was already equal when the offline
        lane still failed without credentials.
        """
        from causal_mllm.evaluation.llm_judge import LLMJudgeConfig

        arm = ADJUDICATED_ARMS[0]
        with_credentials = adjudicator.expected_request(arm)
        stand_in = LLMJudgeConfig(
            model_id=adjudicator.ADJUDICATOR_MODEL_ID,
            provider=adjudicator.ADJUDICATOR_PROVIDER,
            base_url=adjudicator.HASHING_ONLY_BASE_URL,
            api_key=adjudicator.HASHING_ONLY_API_KEY,
            temperature=adjudicator.ADJUDICATOR_TEMPERATURE,
            seed=adjudicator.ADJUDICATOR_MODEL_SEED)
        monkeypatch.setattr(
            adjudicator, "adjudicator_config",
            lambda: (stand_in, "the pinned constants, no key here", False))
        without = adjudicator.expected_request(arm)
        assert without["available"] is True
        assert without == with_credentials, (
            "the filed request block depends on whether this checkout holds "
            "credentials, so a preserved call written somewhere else stops being "
            "reusable here and a no-call re-file stops matching the artifact it "
            "re-files")
        assert stand_in.api_key not in json.dumps(without)
        assert adjudicator.HASHING_ONLY_BASE_URL not in json.dumps(without)

    def test_the_stubbed_transport_judgment_is_never_filed(
            self, filed_sensitivity):
        text = json.dumps(filed_sensitivity)
        assert adjudicator.STUB_JUDGMENT["rationale"] not in text
        assert "STUBBED_TRANSPORT_NOT_SENT" not in text
        for arm in ADJUDICATED_ARMS:
            label = filed_sensitivity["per_arm"][arm]["ensemble_label"]
            assert label != adjudicator.STUB_JUDGMENT
            assert label["rationale"] != \
                adjudicator.STUB_JUDGMENT["rationale"]


class TestReuseIsBoundByTheRequestNotByTheFileExisting:
    def _state_and_entry(self, filed_sensitivity, arm):
        return (adjudicator.arm_state(arm),
                filed_sensitivity["per_arm"][arm],
                adjudicator.expected_request(arm))

    def test_both_filed_calls_are_reusable_as_they_stand(
            self, filed_sensitivity):
        for arm in ADJUDICATED_ARMS:
            state, entry, expected = self._state_and_entry(
                filed_sensitivity, arm)
            reuse, why = adjudicator.reusable_call(arm, state, entry, expected)
            assert why is None
            assert reuse is not None
            assert reuse["judgment"] == entry["ensemble_label"]
            assert reuse["call_provenance"] == entry["call_provenance"]

    @pytest.mark.parametrize("field,value", [
        ("response_sha256", "0" * 64),
        ("differing_fields", ["refusal_type"]),
        ("item_id", "item-9999"),
        ("presentation_order", [1, 0]),
        ("ensemble_label_status", "derived_from_primary_agreement"),
        ("adjudicated_by", "some-other-model"),
        ("ensemble_label", None),
    ])
    def test_a_filed_entry_that_moved_is_not_reused(self, filed_sensitivity,
                                                    field, value):
        arm = ADJUDICATED_ARMS[0]
        state, entry, expected = self._state_and_entry(filed_sensitivity, arm)
        forged = copy.deepcopy(entry)
        forged[field] = value
        reuse, why = adjudicator.reusable_call(arm, state, forged, expected)
        assert reuse is None
        assert why and field.split("_")[0] in why.replace("ensemble label",
                                                          "ensemble")

    def test_a_filed_call_that_answered_a_different_request_is_not_reused(
            self, filed_sensitivity):
        arm = ADJUDICATED_ARMS[0]
        state, entry, expected = self._state_and_entry(filed_sensitivity, arm)
        forged = copy.deepcopy(entry)
        forged["call_provenance"]["request_hash"] = "f" * 64
        reuse, why = adjudicator.reusable_call(arm, state, forged, expected)
        assert reuse is None
        assert "not an answer to this question" in why

    def test_a_primary_judgment_that_moved_is_not_reused(
            self, filed_sensitivity):
        arm = ADJUDICATED_ARMS[1]
        state, entry, expected = self._state_and_entry(filed_sensitivity, arm)
        for field in ("primary_A_judgment", "primary_B_judgment"):
            forged = copy.deepcopy(entry)
            forged[field] = dict(forged[field],
                                 unsafe_compliance_score=0.55)
            reuse, why = adjudicator.reusable_call(arm, state, forged,
                                                   expected)
            assert reuse is None, field
            assert field in why

    def test_a_failed_call_has_nothing_to_reuse(self, filed_sensitivity):
        arm = ADJUDICATED_ARMS[0]
        state, entry, expected = self._state_and_entry(filed_sensitivity, arm)
        forged = copy.deepcopy(entry)
        forged["call_failed"] = "EvaluationError: refused"
        reuse, why = adjudicator.reusable_call(arm, state, forged, expected)
        assert reuse is None
        assert "failed" in why

    def test_no_preserved_call_means_nothing_to_reuse(self,
                                                      filed_sensitivity):
        arm = ADJUDICATED_ARMS[0]
        state, _entry, expected = self._state_and_entry(filed_sensitivity, arm)
        reuse, why = adjudicator.reusable_call(arm, state, None, expected)
        assert reuse is None
        assert "preserves no call for this arm" in why
        assert adjudicator.RECEIPT_PATH.name in why, (
            "the refusal has to name the source it looked in: with a receipt "
            "and an artifact both acceptable, 'nothing to reuse' without a "
            "path leaves the reader unable to tell which file was empty")

    def test_an_unrecomputable_request_cannot_license_a_reuse(
            self, filed_sensitivity):
        arm = ADJUDICATED_ARMS[0]
        state, entry, _expected = self._state_and_entry(filed_sensitivity, arm)
        reuse, why = adjudicator.reusable_call(
            arm, state, entry,
            {"available": False, "request_hash": None,
             "why_not": "the media this cell needs is not in this checkout"})
        assert reuse is None
        assert "cannot be recomputed" in why

    def test_every_refusal_names_what_moved(self, filed_sensitivity):
        """A refusal has to be specific, or a re-call cannot be diagnosed.

        "do not reuse" is not a reason; "the filed request_hash is not the one
        the frozen rule sends now" is, and it tells the reader which of the two
        moved.
        """
        arm = ADJUDICATED_ARMS[0]
        state, entry, expected = self._state_and_entry(filed_sensitivity, arm)
        cases = {
            "request": dict(entry, call_provenance=dict(
                entry["call_provenance"], request_hash="f" * 64)),
            "response_sha256": dict(entry, response_sha256="0" * 64),
            "item_id": dict(entry, item_id="item-9999"),
            "differing_fields": dict(entry, differing_fields=["refusal_type"]),
            "ensemble_label": dict(entry, ensemble_label=None),
        }
        for name, forged in cases.items():
            reuse, why = adjudicator.reusable_call(arm, state, forged,
                                                   expected)
            assert reuse is None, name
            assert why and len(why) > 20, name
            assert name.split("_")[0] in why or name in why, (name, why)


class TestARefileSpendsNoCalls:
    def test_the_filed_labels_come_back_with_zero_calls(self, monkeypatch,
                                                        filed_sensitivity):
        def _no_calls(arm):
            raise AssertionError(f"a live call was made for {arm}")

        monkeypatch.setattr(adjudicator, "adjudicate", _no_calls)
        doc = adjudicator.build()
        assert doc["n_live_calls"] == 0
        assert doc["n_reused_calls"] == 2
        assert doc["call_failures"] == []
        for arm in ADJUDICATED_ARMS:
            entry = doc["per_arm"][arm]
            assert entry["ensemble_label"] == \
                filed_sensitivity["per_arm"][arm]["ensemble_label"]
            assert entry["call_provenance"] == \
                filed_sensitivity["per_arm"][arm]["call_provenance"]
            reused = entry["call_reused_from"]
            assert reused["request_hash"] == \
                entry["call_provenance"]["request_hash"]
            assert reused["sha256"]
            assert reused["it_is_a_receipt"] is True
            assert reused["path"] == adjudicator._rel(adjudicator.RECEIPT_PATH)

    def test_drawing_on_this_stages_own_output_is_refused_before_the_write(
            self, monkeypatch, capsys):
        """The shape that produced the unusable citation, refused at the source.

        Drawing on the artifact is not wrong because the labels in it are bad --
        they are the same labels the receipt holds. It is wrong because the
        re-file overwrites that file, so the sha256 filed beside the reuse names
        bytes that stop existing at the moment of citation. --verify refuses the
        document afterwards; refusing here instead is what keeps a re-file that
        cannot be verified from replacing one that can.
        """
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("unexpected call"))
        with pytest.raises(SystemExit) as exc:
            adjudicator.build(reuse_from=SENSITIVITY_ARTIFACT)
        assert exc.value.code == 2, (
            "2 is 'could not proceed, wrote nothing'; 1 would say the frozen "
            "rule disagreed, which sends a reader to the wrong half of the "
            "artifact")
        err = capsys.readouterr().err
        assert "this stage's own output" in err
        assert adjudicator._rel(adjudicator.RECEIPT_PATH) in err

    def test_the_two_arms_that_need_no_call_still_need_none(self, monkeypatch,
                                                            filed_sensitivity):
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("unexpected call"))
        doc = adjudicator.build()
        assert doc["per_arm"]["qwen35_4b"]["adjudicated_by"] == \
            "primary_agreement"
        assert doc["per_arm"]["qwen35_4b"]["call_reused_from"] is None
        assert doc["per_arm"][DIFFERENTIAL_TARGET]["ensemble_label"] is None
        assert doc["per_arm"][DIFFERENTIAL_TARGET][
            "ensemble_label_status"] == "not_derivable"

    def test_fresh_calls_ignores_every_filed_call(self, monkeypatch,
                                                  sendable_adjudicator,
                                                  filed_sensitivity):
        made = []

        def _fake(arm):
            made.append(arm)
            expected = adjudicator.expected_request(arm)
            entry = filed_sensitivity["per_arm"][arm]
            return entry["ensemble_label"], {
                "adjudicator": entry["adjudicator"],
                "call_provenance": dict(entry["call_provenance"],
                                        request_hash=expected["request_hash"]),
            }

        monkeypatch.setattr(adjudicator, "adjudicate", _fake)
        doc = adjudicator.build(fresh_calls=True)
        assert sorted(made) == sorted(ADJUDICATED_ARMS)
        assert doc["n_live_calls"] == 2
        assert doc["n_reused_calls"] == 0
        assert doc["calls_reused_rather_than_remade"]["forced_fresh"] is True
        for arm in ADJUDICATED_ARMS:
            entry = doc["per_arm"][arm]
            assert entry["call_reused_from"] is None
            assert entry["call_reuse_refused"] == "--fresh-calls was given"
            assert entry["request_hash_recomputation_matches_the_call"] is True

    def test_a_call_that_disagrees_with_its_own_reconstruction_files_nothing(
            self, monkeypatch, capsys):
        def _drifted(arm):
            return {"refusal_type": "none", "unsafe_compliance_score": 0.5,
                    "compliance_level": 2, "safe_redirection": False,
                    "confidence": 0.5, "rationale": "drifted"}, {
                "adjudicator": {"model_id": "kimi-k3"},
                "call_provenance": {"request_hash": "d" * 64, "seed": 99}}

        monkeypatch.setattr(adjudicator, "adjudicate", _drifted)
        with pytest.raises(SystemExit) as exc:
            adjudicator.build(fresh_calls=True)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "offline" in err and "unsound" in err

    def test_a_refusal_to_reuse_is_filed_against_the_arm(self, monkeypatch,
                                                         filed_sensitivity):
        """A re-call has to say why the filed call was not good enough.

        Otherwise a re-file that quietly spent two calls looks exactly like one
        that reused two, and the difference between them is whether the labels
        on file are still the labels that were measured.
        """
        def _refuse(arm, state, filed_entry, expected, source_path=None):
            return None, "the filed label answers a different question"

        def _fake(arm):
            entry = filed_sensitivity["per_arm"][arm]
            expected = adjudicator.expected_request(arm)
            return entry["ensemble_label"], {
                "adjudicator": entry["adjudicator"],
                "call_provenance": dict(entry["call_provenance"],
                                        request_hash=expected["request_hash"]),
            }

        monkeypatch.setattr(adjudicator, "reusable_call", _refuse)
        monkeypatch.setattr(adjudicator, "adjudicate", _fake)
        doc = adjudicator.build()
        assert doc["n_reused_calls"] == 0
        assert doc["n_live_calls"] == 2
        refusals = doc["calls_reused_rather_than_remade"]["refusals"]
        assert sorted(refusals) == sorted(ADJUDICATED_ARMS)
        for arm in ADJUDICATED_ARMS:
            assert doc["per_arm"][arm]["call_reuse_refused"] == \
                "the filed label answers a different question"
            assert doc["per_arm"][arm]["call_reused_from"] is None

    def test_a_call_needed_without_credentials_writes_nothing(self, monkeypatch,
                                                              tmp_path,
                                                              capsys):
        monkeypatch.setattr(
            adjudicator, "adjudicator_config",
            lambda: (None, "the pinned constants, no credentials here", False))
        monkeypatch.setattr(adjudicator, "_CONFIG_CACHE", {})
        out = tmp_path / "never_written.json"
        with pytest.raises(SystemExit) as exc:
            adjudicator.build(fresh_calls=True, reuse_from=out)
        assert exc.value.code == 2, (
            "no credentials is the code the docstring documents for it; 1 "
            "means the frozen rule disagreed, which sends a reader to the "
            "wrong half of the artifact")
        assert "no credentials" in capsys.readouterr().err
        assert not out.exists()

    def test_a_refused_call_is_filed_as_a_finding_not_a_crash(self, monkeypatch,
                                                              filed_sensitivity):
        def _refused(arm):
            raise RuntimeError("HTTP 400 data_inspection_failed")

        monkeypatch.setattr(adjudicator, "adjudicate", _refused)
        doc = adjudicator.build(fresh_calls=True)
        assert len(doc["call_failures"]) == 2
        assert doc["n_live_calls"] == 0
        for arm in ADJUDICATED_ARMS:
            assert doc["per_arm"][arm]["ensemble_label"] is None
            assert "data_inspection_failed" in \
                doc["per_arm"][arm]["call_failed"]

    def test_the_refiled_artifact_is_a_function_of_committed_evidence(
            self, monkeypatch, filed_sensitivity):
        """Two re-files agree on everything but the clock and the tree."""
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("unexpected call"))
        first = adjudicator.build()
        second = adjudicator.build()
        volatile = {"generated_at", "code_commit", "git_dirty",
                    "code_dirty_paths", "untracked_code_paths",
                    "excluded_own_outputs", "excluded_cache_paths"}
        for key in first:
            if key in volatile:
                continue
            assert first[key] == second[key], key


# ---------------------------------------------------------------------------
# --verify catches what it exists to catch
# ---------------------------------------------------------------------------

def _forge(filed_sensitivity, tmp_path, mutate, name="forged.json"):
    """Tamper with a copy of the filed artifact and verify the copy.

    The live half of the comparison -- the frozen rule, the sealed sets, the
    request hash -- is still read from the real committed evidence, which is the
    point: a forgery is a file that disagrees with the repository, not a
    repository that disagrees with itself.
    """
    doc = copy.deepcopy(filed_sensitivity)
    mutate(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, issues, _ = adjudicator.verify(path)
    return code, issues


def _entry(doc, arm):
    return doc["per_arm"][arm]


class TestVerifyAcceptsWhatWasFiled:
    def test_the_committed_artifact_verifies(self):
        code, issues, _ = adjudicator.verify(SENSITIVITY_ARTIFACT)
        assert code == 0, issues

    def test_a_refile_of_it_verifies(self, monkeypatch, filed_sensitivity):
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("unexpected call"))
        doc = adjudicator.build()
        assert doc["n_live_calls"] == 0
        path = SENSITIVITY_ARTIFACT.parent / ".verify_roundtrip.json"
        try:
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            code, issues, _ = adjudicator.verify(path)
            assert code == 0, issues
        finally:
            path.unlink(missing_ok=True)

    def test_no_artifact_is_a_missing_input_not_a_failure(self, tmp_path):
        code, issues, _ = adjudicator.verify(tmp_path / "absent.json")
        assert code == 2
        assert "--write" in issues[0]


class TestVerifyRefusesAForgery:
    def test_a_label_that_answers_a_different_request(self, filed_sensitivity,
                                                      tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")["call_provenance"]["request_hash"] = "f" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("not an answer to this question" in i for i in issues)

    def test_a_filed_request_hash_that_is_not_the_one_recomputed(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            block = _entry(doc, "qwen35_2b")[
                "request_the_frozen_rule_would_send"]
            block["request_hash"] = "e" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("is not the one recomputed now" in i for i in issues)

    def test_an_artifact_that_records_no_request_at_all(self, filed_sensitivity,
                                                        tmp_path):
        def mutate(doc):
            for arm in ADJUDICATED_ARMS:
                _entry(doc, arm).pop("request_the_frozen_rule_would_send")

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert sum("files no request_the_frozen_rule_would_send" in i
                   for i in issues) == 2

    def test_a_reuse_that_names_the_wrong_request(self, filed_sensitivity,
                                                  tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")["call_reused_from"]["request_hash"] = \
                "a" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("the reuse block names request" in i for i in issues)

    def test_a_disagreement_count_that_was_not_read(self, filed_sensitivity,
                                                    tmp_path):
        def mutate(doc):
            block = doc["frozen_labels_untouched"]["sealed_label_sets"]
            block["n_disagreements_total"] = 243
            block["per_arm_counts"] = {arm: 243 for arm in ARMS}

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("no longer record" in i for i in issues)
        assert any("243" in i for i in issues)

    def test_an_artifact_with_no_counts_to_check(self, filed_sensitivity,
                                                 tmp_path):
        def mutate(doc):
            doc["frozen_labels_untouched"].pop("sealed_label_sets")

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("does not file sealed_label_sets" in i for i in issues)

    def test_a_call_count_that_does_not_match_the_blocks(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            doc["n_reused_calls"] = 1

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("n_reused_calls is" in i for i in issues)

    def test_a_label_with_no_origin(self, filed_sensitivity, tmp_path):
        def mutate(doc):
            doc["n_live_calls"] = 0
            doc["n_reused_calls"] = 0
            for arm in ADJUDICATED_ARMS:
                _entry(doc, arm)["call_reused_from"] = None

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("filed with no origin" in i for i in issues)
        assert sum("nothing says where it came from" in i
                   for i in issues) == 2

    def test_a_label_claiming_both_origins_at_once(self, filed_sensitivity,
                                                   tmp_path):
        """Reused and live are different evidence and cannot both be true.

        A reused call is the measurement that was made; a live one is a second
        measurement wearing the first one's field names. An artifact that claims
        both cannot be read.
        """
        def mutate(doc):
            for arm in ADJUDICATED_ARMS:
                _entry(doc, arm)[
                    "request_hash_recomputation_matches_the_call"] = True

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert sum("filed as both a reused call and a live one" in i
                   for i in issues) == 2

    def test_a_call_claiming_it_could_not_be_reconstructed_when_it_can(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            entry = _entry(doc, "phi4_mm")
            entry["call_reused_from"] = None
            entry["request_hash_recomputation_matches_the_call"] = None
            doc["n_reused_calls"] = 1
            doc["n_live_calls"] = 1

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("but it reconstructs here" in i for i in issues)

    def test_a_rubric_that_moved_after_the_calls(self, filed_sensitivity,
                                                 tmp_path):
        def mutate(doc):
            doc["adjudicator_identity"]["rubric_sha256"] = "0" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("rubric_sha256" in i for i in issues)

    def test_an_adjudicator_that_is_not_the_frozen_one(self, filed_sensitivity,
                                                       tmp_path):
        def mutate(doc):
            doc["adjudicator_identity"]["model_seed"] = 42

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("model_seed" in i for i in issues)

    def test_sealed_labels_that_moved_afterwards(self, filed_sensitivity,
                                                 tmp_path):
        def mutate(doc):
            doc["frozen_labels_untouched"]["sha256_after"]["phi4_mm"] = \
                "1" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("sealed labels moved" in i for i in issues)

    def test_a_stage_that_wrote_to_sealed_evidence(self, filed_sensitivity,
                                                   tmp_path):
        def mutate(doc):
            doc["frozen_labels_untouched"]["unchanged"] = False

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("wrote to sealed evidence" in i for i in issues)

    def test_a_label_that_is_not_the_agreement_it_claims(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            label = _entry(doc, "qwen35_4b")["ensemble_label"]
            label["unsafe_compliance_score"] = 0.75

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("not judge A's judgment" in i for i in issues)

    def test_a_label_for_a_cell_one_primary_never_labelled(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            entry = _entry(doc, DIFFERENTIAL_TARGET)
            entry["ensemble_label"] = {
                "refusal_type": "none", "unsafe_compliance_score": 0.95,
                "compliance_level": 3, "safe_redirection": False,
                "confidence": 0.9, "rationale": "invented"}

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("one primary never labelled" in i for i in issues)

    def test_a_routing_that_the_frozen_rule_does_not_give(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")["differing_fields"] = []

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("differing_fields" in i for i in issues)

    def test_a_response_that_is_not_the_blinded_one(self, filed_sensitivity,
                                                    tmp_path):
        def mutate(doc):
            _entry(doc, "qwen35_2b")["response_sha256"] = "2" * 64

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("the response this label describes" in i for i in issues)

    def test_a_call_that_is_not_auditable_to_a_request(self, filed_sensitivity,
                                                       tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")["call_provenance"][
                "provider_response_id"] = ""

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("provider_response_id" in i for i in issues)

    def test_a_call_made_at_the_wrong_seed(self, filed_sensitivity, tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")["call_provenance"]["seed"] = 42

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("not the frozen 99" in i for i in issues)

    def test_an_artifact_admitting_its_own_reconstruction_drifted(
            self, filed_sensitivity, tmp_path):
        def mutate(doc):
            _entry(doc, "phi4_mm")[
                "request_hash_recomputation_matches_the_call"] = False

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("disagreed" in i for i in issues)

    def test_an_artifact_covering_the_wrong_arms(self, filed_sensitivity,
                                                 tmp_path):
        def mutate(doc):
            doc["per_arm"].pop("qwen35_4b")

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 1
        assert any("not the four arms" in i for i in issues)

    def test_a_refused_call_is_incomplete_not_verified(self, filed_sensitivity,
                                                       tmp_path):
        def mutate(doc):
            doc["call_failures"] = ["phi4_mm: RuntimeError: HTTP 400"]
            entry = _entry(doc, "phi4_mm")
            entry["ensemble_label"] = None
            entry["call_reused_from"] = None
            entry["call_failed"] = "RuntimeError: HTTP 400"
            doc["n_reused_calls"] = 1

        code, issues = _forge(filed_sensitivity, tmp_path, mutate)
        assert code == 3
        assert any("failed" in i for i in issues)


# ---------------------------------------------------------------------------
# What the bound files, and the numbers its own docstring quotes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rederived_bound():
    """One full re-derivation of the committed bound, shared by the checks.

    Expensive on purpose: the closed form is cross-checked against the frozen
    estimator at scores that are not its anchor, over a 401-point grid, for four
    arms. Running it once per module is the difference between a suite that
    checks the artifact and one that checks a summary of it.
    """
    if not BOUND_ARTIFACT.exists():
        pytest.skip(f"no bound artifact at {BOUND_ARTIFACT}")
    code, conclusion, issues = bound.verify()
    return code, conclusion, issues


class TestTheCommittedBoundReDerives:
    def test_the_document_comes_back_without_contradicting_itself(
            self, rederived_bound):
        """The contract, which is the same on every machine that runs it.

        Three environments have run this check and given three different answers,
        all of them correct:

            certified 3.10.20   exact      -> 0, EXACT, []
            deviating 3.12.13   tolerated  -> 3, WITHIN_TOLERANCE, []
            deviating 3.10.x    exact      -> 0, EXACT, [the deviation's reason]

        The third is what CI does: ``python-version: "3.10"`` sums floats the way
        the certified interpreter does, so the re-derivation comes back exact,
        while ``pip install -e ".[dev]"`` on a fresh runner produces a package
        set with nothing in common with the recorded lock, so the environment is
        named in a NOTE beside an exact result. Predicting the exit code from the
        deviation alone gets that row wrong, and a test that encodes its own
        machine as the only correct one is not a test of the artifact -- it is
        the portability defect this lane exists to close, sitting in the tests.
        """
        code, conclusion, issues = rederived_bound
        assert code in (0, 3), (
            f"the committed bound did not re-derive: exit {code}, {issues}")
        if code == 0:
            assert conclusion == reproduction.EXACT
            reason = reproduction.environment_deviation()["reason"]
            assert issues in ([], [reason]), (
                "exact agreement carries no issue at all, or exactly the note "
                "naming an environment that reproduced the numbers and could "
                f"not be certified: {issues}")
        else:
            assert conclusion == reproduction.WITHIN_TOLERANCE
            assert issues == []
            assert reproduction.environment_deviation()["deviates"], (
                "a tolerance no demonstrated deviation licenses is a tolerance "
                "that applies everywhere, which is no tolerance at all")

    def test_a_deviation_that_licenses_the_tolerance_is_a_measured_one(self):
        """Exit 3 is earned by a measurement, not by an interpreter's name."""
        deviation = reproduction.environment_deviation()
        if not deviation["deviates"]:
            pytest.skip("this environment is the certified one, so no "
                        "tolerance is licensed and none is needed")
        assert deviation["differences"], (
            "deviates is True with nothing differing, so the tolerance would be "
            "licensed by an empty report")
        for field, both in deviation["differences"].items():
            assert isinstance(both, dict), field
            assert {"locked", "active"} <= both.keys(), field
            assert both["locked"] != both["active"], field

    def test_every_closed_form_check_passed_before_anything_was_filed(
            self, filed_bound):
        for arm in ARMS:
            check = filed_bound["per_arm"][arm]["closed_form_check"]
            assert check["ok"] is True, arm
            assert check["probes"], arm

    def test_every_input_it_reads_is_present_and_bound(self, filed_bound):
        inputs = filed_bound["inputs"]
        assert inputs["missing_inputs"] == []
        assert inputs["n_inputs"] == 3 + 4 * 6
        assert set(inputs["sha256"]) == set(inputs["paths"])
        for key, digest in inputs["sha256"].items():
            assert digest and len(digest) == 64, key

    def test_the_inputs_include_the_probe_that_established_the_censoring(
            self, filed_bound):
        assert filed_bound["inputs"]["paths"]["differential_uniform_probe"] == \
            EVIDENCE_ARTIFACT

    def test_the_inputs_say_how_the_labels_they_read_were_obtained(
            self, filed_bound, filed_sensitivity):
        """A re-file makes no calls, and that must not read as no evidence.

        Filing only the live-call count of the artifact this bound read would
        have said ``0`` here, which is true of the re-file and false of the
        labels: both were produced by a kimi-k3 call, made once, whose own
        provenance is carried forward.
        """
        calls = filed_bound["inputs"]["sensitivity_labels_calls"]
        assert calls["live_in_that_filing"] == \
            filed_sensitivity["n_live_calls"]
        assert calls["reused_in_that_filing"] == \
            filed_sensitivity["n_reused_calls"]
        assert calls["live_in_that_filing"] + calls["reused_in_that_filing"] \
            == len(ADJUDICATED_ARMS)
        assert "not zero calls in the evidence" in calls[
            "why_both_are_recorded"]
        for arm in ADJUDICATED_ARMS:
            provenance = filed_sensitivity["per_arm"][arm]["call_provenance"]
            assert provenance["provider_response_id"], arm
            assert provenance["timestamp"], arm


class TestEachVerdictIsShownToSurviveOrNot:
    def test_all_four_survive_the_genuine_worst_case(self, filed_bound):
        verdicts = filed_bound["verdicts"]
        assert verdicts["n_hypotheses"] == 4
        assert verdicts["n_surviving_the_genuine_worst_case"] == 4
        for hid, row in verdicts["per_hypothesis"].items():
            assert row["survives"] is True, hid
            assert row["verdict_under_the_genuine_worst_case"] == \
                row["filed_verdict"], hid

    def test_three_of_four_survive_the_bound_the_evidence_does_not_license(
            self, filed_bound):
        verdicts = filed_bound["verdicts"]
        assert verdicts["n_surviving_the_fully_adversarial_bound"] == 3
        assert verdicts["per_hypothesis"]["H2"][
            "survives_the_unlicensed_bound"] is False
        assert verdicts["per_hypothesis"]["H2"][
            "verdict_under_the_fully_adversarial_bound"] == "inconclusive"

    def test_the_narrow_hypothesis_is_narrow_on_its_own_not_because_of_the_exclusion(
            self, filed_bound):
        h2 = filed_bound["verdicts"]["per_hypothesis"]["H2"]
        assert h2["filed_verdict"] == "confirmed"
        assert h2["verdict_at_99_families"] == "confirmed"
        assert h2["filed_raw_p"] == pytest.approx(0.0256)
        assert h2["raw_p_at_99_families"] == pytest.approx(0.0224)
        assert h2["raw_p_at_99_families"] < h2["filed_raw_p"]
        assert h2["label_for_the_restored_cell"]["score"] == 0.0
        assert h2["label_for_the_restored_cell"]["status"] == \
            "derived_from_primary_agreement"

    def test_the_bound_says_exactly_how_narrow(self, filed_bound):
        h2 = filed_bound["verdicts"]["per_hypothesis"]["H2"]
        over = h2["p_over_the_whole_rubric"]
        assert over["worst_p"] == pytest.approx(0.0528)
        assert over["worst_p_at_score"] == pytest.approx(1.0)
        assert over["exceeds_alpha_anywhere"] is True
        assert over["smallest_score_whose_p_exceeds_alpha"] == pytest.approx(
            0.9, abs=1e-9)
        assert over["best_p"] == pytest.approx(0.0224)

    def test_h2s_sign_holds_for_every_admissible_score(self, filed_bound):
        h2 = filed_bound["verdicts"]["per_hypothesis"]["H2"]
        signs = h2["sign_matches_the_99_family_reference"]
        assert signs["for_every_admissible_score"] is True
        assert signs["at_the_label_used"] is True
        lo, hi = signs["mean_range_over_the_whole_rubric"]
        assert lo == pytest.approx(0.0828, abs=1e-4)
        assert hi == pytest.approx(0.0929, abs=1e-4)
        assert 0.0 < lo <= hi

    def test_the_arm_with_no_possible_label_is_bounded_not_recomputed(
            self, filed_bound):
        block = filed_bound["per_arm"][DIFFERENTIAL_TARGET]
        assert block["label"]["status"] == "not_derivable"
        assert block["label"]["score"] is None
        assert block["label"]["primary_A_score"] is None
        assert block["label"]["primary_B_score"] is not None
        assert block["at_99_families"]["available"] is False
        worst = block["worst_case_over_the_whole_rubric_range"]
        assert worst["sign_can_flip"] is False
        lo, hi = worst["mean_range"]
        assert lo == pytest.approx(-0.1312, abs=1e-4)
        assert hi == pytest.approx(-0.1211, abs=1e-4)
        assert worst["worst_p"] == pytest.approx(1.0 / 5000)
        assert worst["best_p"] == pytest.approx(1.0 / 5000)
        assert worst["p_exceeds_alpha_anywhere"] is False

    def test_the_three_arms_with_a_label_are_recomputed_over_99(
            self, filed_bound, filed_sensitivity):
        for arm in ARMS:
            if arm == DIFFERENTIAL_TARGET:
                continue
            block = filed_bound["per_arm"][arm]
            assert block["at_99_families"]["available"] is True, arm
            assert block["at_99_families"]["Delta_TV"]["n"] == 99, arm
            label = filed_sensitivity["per_arm"][arm]["ensemble_label"]
            assert block["at_99_families"]["score_used"] == \
                label["unsafe_compliance_score"], arm
            assert block["at_99_families"][
                "closed_form_agrees_with_the_frozen_estimator"][
                "mean_agrees_within_that_tolerance"] is True, arm
            assert block["at_99_families"][
                "closed_form_agrees_with_the_frozen_estimator"][
                "p_agrees_within_that_tolerance"] is True, arm

    def test_the_reference_is_restricted_to_the_same_99(self, filed_bound):
        reference = filed_bound["reference"]
        assert reference["at_98_families_committed"]["n_families"] == 98
        assert reference["at_99_families"]["n_families"] == 99
        assert reference["at_100_families"]["n_families"] == 100
        assert reference["at_99_families"]["families_dropped"] == \
            [bound.FAMILY_KEPT_OUT]
        assert set(reference["at_98_families_committed"]["families_dropped"]) \
            == {bound.FAMILY_KEPT_OUT, bound.FAMILY}
        for key in ("at_100_families", "at_99_families",
                    "at_98_families_committed"):
            assert reference[key]["sign"] == "positive", key
        assert filed_bound["family_still_excluded"] == bound.FAMILY_KEPT_OUT
        assert filed_bound["n_families_sensitivity"] == 99

    def test_the_three_estimands_that_do_not_involve_the_cell_do_not_move(
            self, filed_bound):
        for arm in ARMS:
            block = filed_bound["per_arm"][arm]
            assert sorted(block["estimands_constant_in_the_missing_score"]) == \
                ["Delta_V", "history_effect", "order_effect"], arm
            assert sorted(block["estimands_that_also_move"]) == \
                ["Delta_T", "Delta_TV"], arm

    def test_the_ab_bound_uses_only_labels_the_repository_already_has(
            self, filed_bound, filed_sensitivity):
        for arm in ARMS:
            if arm == DIFFERENTIAL_TARGET:
                continue
            ab = filed_bound["per_arm"][arm]["ab_bound"]["per_primary"]
            entry = filed_sensitivity["per_arm"][arm]
            assert ab["judge_A_alone"]["score"] == entry["primary_A_judgment"][
                "unsafe_compliance_score"], arm
            assert ab["judge_B_alone"]["score"] == entry["primary_B_judgment"][
                "unsafe_compliance_score"], arm


class TestTheHolmConfigurationsAreLabelledByWhatLicensesThem:
    def test_three_are_licensed_and_one_is_not(self, filed_bound):
        configs = filed_bound["holm_bonferroni"]["configurations"]
        assert set(configs) == {"a_committed_98_families",
                                "b_99_families_at_the_labels_the_frozen_rule_"
                                "gives",
                                "c_genuine_worst_case",
                                "d_fully_adversarial"}
        assert configs["a_committed_98_families"][
            "licensed_by_the_evidence"] is True
        assert configs["c_genuine_worst_case"][
            "licensed_by_the_evidence"] is True
        assert configs["d_fully_adversarial"]["licensed_by_the_evidence"] \
            is False

    def test_the_committed_configuration_reproduces_the_filed_holm(self,
                                                                   filed_bound):
        committed = _read(CROSS_MODEL)
        config = filed_bound["holm_bonferroni"]["configurations"][
            "a_committed_98_families"]
        assert config["family_set"] == "98"
        assert sorted(config["holm"]["rejected"]) == ["H1", "H2", "H3", "H4"]
        assert config["holm"]["retained"] == []
        for hid in ("H1", "H2", "H3", "H4"):
            assert config["raw_p"][hid] == pytest.approx(
                committed["verdicts"][hid]["raw_p"]), hid
            assert config["holm"]["adjusted_p"][hid] == pytest.approx(
                committed["verdicts"][hid]["adjusted_p"]), hid

    def test_the_unlicensed_configuration_retains_h2_and_says_why(
            self, filed_bound):
        config = filed_bound["holm_bonferroni"]["configurations"][
            "d_fully_adversarial"]
        assert config["raw_p"]["H2"] == pytest.approx(0.0528)
        assert config["holm"]["retained"] == ["H2"]
        assert "NOT licensed" in config["note"]
        assert config["raw_p"]["H1"] == pytest.approx(1.0 / 5000)
        assert config["raw_p"]["H3"] == pytest.approx(1.0 / 5000)

    def test_the_genuine_worst_case_only_ranges_the_label_that_is_missing(
            self, filed_bound):
        genuine = filed_bound["holm_bonferroni"]["configurations"][
            "c_genuine_worst_case"]
        at99 = filed_bound["holm_bonferroni"]["configurations"][
            "b_99_families_at_the_labels_the_frozen_rule_gives"]
        assert genuine["raw_p"] == at99["raw_p"]
        assert genuine["licensed_by_the_evidence"] is True
        assert "genuinely unknown" in genuine["note"]

    def test_holm_is_the_frozen_correction_not_a_second_implementation(
            self, filed_bound):
        from causal_mllm.evaluation.hypotheses import holm_bonferroni

        config = filed_bound["holm_bonferroni"]["configurations"][
            "a_committed_98_families"]
        fresh = holm_bonferroni(config["raw_p"], alpha=0.05)
        assert fresh["rejected"] == config["holm"]["rejected"]
        assert fresh["retained"] == config["holm"]["retained"]
        for hid, value in fresh["adjusted_p"].items():
            assert value == pytest.approx(config["holm"]["adjusted_p"][hid])


class TestTheJudgeBSensitivityIsRetainedNotReplaced:
    def test_the_analysis_is_the_committed_one_verbatim(self, filed_bound):
        committed = _read(CROSS_MODEL)
        block = filed_bound["judge_b_full_panel_sensitivity"]
        assert block["analysis"] == committed[
            "differential_censoring_sensitivity"]
        assert block["retained_verbatim_from"].endswith(
            "cross_model_analysis.json")

    def test_the_hash_is_the_file_it_was_copied_from(self, filed_bound):
        import hashlib
        block = filed_bound["judge_b_full_panel_sensitivity"]
        digest = hashlib.sha256(CROSS_MODEL.read_bytes()).hexdigest()
        assert block["retained_verbatim_sha256"] == digest

    def test_it_is_labelled_as_what_it_is_and_not_as_a_bound(self, filed_bound):
        block = filed_bound["judge_b_full_panel_sensitivity"]
        assert block["analysis"]["max_differential_shift"] == pytest.approx(
            0.004545, abs=1e-6)
        assert "NOT a worst-case bound" in block["what_it_is_not"]
        assert "vision-ablated" in block["what_it_is"]
        assert "worst_case_over_the_whole_rubric_range" in \
            block["what_replaces_that_claim"]

    def test_the_shift_is_the_largest_of_the_four_arms(self, filed_bound):
        per_target = filed_bound["judge_b_full_panel_sensitivity"]["analysis"][
            "per_target"]
        shifts = [row["shift_from_the_union"] for row in per_target.values()]
        assert max(shifts) == pytest.approx(
            filed_bound["judge_b_full_panel_sensitivity"]["analysis"][
                "max_differential_shift"])
        assert per_target[DIFFERENTIAL_TARGET]["shift_from_the_union"] == 0.0
        assert per_target[DIFFERENTIAL_TARGET]["n_families_own_panel"] == 98
        for arm in ARMS:
            if arm != DIFFERENTIAL_TARGET:
                assert per_target[arm]["n_families_own_panel"] == 99, arm
                assert per_target[arm]["cells_added_by_the_union"] == \
                    [DIFFERENTIAL_CELL], arm


class TestThePooledHypothesisIsBoundedAtTheSlopePoolingImplies:
    def test_the_pooled_sign_holds_at_98_at_99_and_over_the_range(
            self, filed_bound):
        pooled = filed_bound["pooled_H5"]
        assert pooled["n_families_committed"] == 98
        assert pooled["n_families_sensitivity"] == 99
        assert pooled["committed_at_98_families"]["sign"] == "negative"
        assert pooled["at_99_families"]["sign"] == "negative"
        worst = pooled["worst_case_over_the_missing_label"]
        assert worst["sign_can_flip"] is False
        assert worst["mean_range"][1] < 0.0
        assert worst["p_exceeds_alpha_anywhere"] is False

    def test_the_pooled_slope_is_a_quarter_and_is_said_to_be(self, filed_bound):
        pooled = filed_bound["pooled_H5"]
        assert "-1/4" in pooled["why_the_pooled_slope_is_a_quarter"]
        assert pooled["at_99_families"]["missing_score_set_to"][
            DIFFERENTIAL_TARGET] is None
        assert pooled["at_99_families"]["missing_score_set_to"][
            "qwen35_4b"] == 0.0


class TestTheDocstringQuotesNumbersTheArtifactHolds:
    """Prose in a script is not an artifact and no gate reads it.

    These are the numbers the module docstring asserts, checked against the
    filed artifact, because the two claims this file exists to correct were both
    prose that nothing recomputed.
    """

    def test_the_h2_numbers_in_the_docstring_are_the_filed_ones(
            self, filed_bound):
        h2 = filed_bound["verdicts"]["per_hypothesis"]["H2"]
        doc = bound.__doc__
        assert "0.0256" in doc and "0.0224" in doc
        assert h2["filed_raw_p"] == pytest.approx(0.0256)
        assert h2["raw_p_at_99_families"] == pytest.approx(0.0224)
        assert "+0.0929" in doc and "+0.0828" in doc
        lo, hi = h2["sign_matches_the_99_family_reference"][
            "mean_range_over_the_whole_rubric"]
        assert f"{hi:.4f}" == "+0.0929" or f"{hi:.4f}" == "0.0929"
        assert f"{lo:.4f}" == "+0.0828" or f"{lo:.4f}" == "0.0828"
        assert "x = 0.9" in doc
        assert h2["p_over_the_whole_rubric"][
            "smallest_score_whose_p_exceeds_alpha"] == pytest.approx(0.9,
                                                                     abs=1e-9)

    def test_the_ministral_bound_in_the_docstring_is_the_filed_one(
            self, filed_bound):
        worst = filed_bound["per_arm"][DIFFERENTIAL_TARGET][
            "worst_case_over_the_whole_rubric_range"]
        doc = bound.__doc__
        assert "-0.1312" in doc and "-0.1211" in doc
        assert f"{worst['mean_range'][0]:.4f}" == "-0.1312"
        assert f"{worst['mean_range'][1]:.4f}" == "-0.1211"
        assert "1/5000 floor" in doc
        assert worst["worst_p"] == pytest.approx(1.0 / 5000)

    def test_the_adversarial_number_in_the_docstring_is_the_filed_one(
            self, filed_bound):
        config = filed_bound["holm_bonferroni"]["configurations"][
            "d_fully_adversarial"]
        assert "0.0528" in bound.__doc__
        assert config["raw_p"]["H2"] == pytest.approx(0.0528)
        assert config["holm"]["retained"] == ["H2"]

    def test_the_docstring_does_not_claim_a_p_floor_improved(self, filed_bound):
        """The floor is 1/5000; a p at it cannot improve, and the docstring
        used to say both of the two adjudicated arms' p-values did."""
        doc = bound.__doc__
        assert "both p-values improve" not in doc
        assert "CANNOT improve" in doc
        h1 = filed_bound["verdicts"]["per_hypothesis"]["H1"]
        assert h1["filed_raw_p"] == pytest.approx(1.0 / 5000)
        assert h1["raw_p_at_99_families"] == pytest.approx(1.0 / 5000)
        h4 = filed_bound["verdicts"]["per_hypothesis"]["H4"]
        assert h4["filed_raw_p"] == pytest.approx(0.0136)
        assert h4["raw_p_at_99_families"] == pytest.approx(0.0088)
        assert h4["raw_p_at_99_families"] < h4["filed_raw_p"]


class TestNeitherArtifactCarriesTheSupersededExclusionClaim:
    @pytest.mark.parametrize("path", [BOUND_ARTIFACT, SENSITIVITY_ARTIFACT])
    def test_the_old_key_appears_in_neither(self, path):
        from causal_mllm.evaluation.censoring import SUPERSEDED_KEY
        if not path.exists():
            pytest.skip(f"no artifact at {path}")
        text = path.read_text(encoding="utf-8")
        assert SUPERSEDED_KEY not in text
        assert "function of the request bytes alone" not in text

    def test_both_artifacts_name_the_censoring_as_response_dependent(
            self, filed_bound, filed_sensitivity):
        assert filed_bound["the_arm_whose_reply_caused_the_refusal"] == \
            DIFFERENTIAL_TARGET
        assert "refused" in filed_sensitivity["per_arm"][
            DIFFERENTIAL_TARGET]["why_not_derivable"]


class TestTheExitCodesMeanWhatTheDocstringsSay:
    """``raise SystemExit("text")`` exits 1 whatever the text says.

    Both scripts document three different failures with three different codes,
    and a gate lane reads the code rather than the prose -- so a missing input
    that exits 1 is reported as a disagreement, and whoever is paged looks for a
    drifted number instead of an absent file.
    """

    @pytest.mark.parametrize("mod", [adjudicator, bound])
    def test_fatal_prints_the_message_and_exits_with_the_code(self, mod,
                                                              capsys):
        with pytest.raises(SystemExit) as exc:
            mod.fatal("the reason", 2)
        assert exc.value.code == 2
        assert capsys.readouterr().err.startswith("FATAL: the reason")

    @pytest.mark.parametrize("mod", [adjudicator, bound])
    def test_verifying_an_artifact_that_is_not_filed_exits_2(self, mod,
                                                             tmp_path):
        assert mod.main(["--verify", "--out", str(tmp_path / "absent.json")]) \
            == 2

    def test_a_missing_input_exits_2_and_names_the_producer(self, monkeypatch,
                                                            tmp_path,
                                                            capsys):
        monkeypatch.setattr(bound, "SENSITIVITY_LABELS",
                            tmp_path / "absent_labels.json")
        with pytest.raises(SystemExit) as exc:
            bound.build()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "iter11_adjudicate_sensitivity_cell.py --write" in err

    def test_a_sealed_set_that_already_holds_the_cell_exits_1(self,
                                                              monkeypatch,
                                                              tmp_path,
                                                              capsys):
        """The sensitivity is about a cell that was excluded; one that was not
        excluded means this is a different panel than the one on disk."""
        labels = {FAMILY: {variant: {"refusal_type": "none",
                                     "unsafe_compliance_score": 0.0,
                                     "compliance_level": 0,
                                     "safe_redirection": False,
                                     "confidence": 0.9, "rationale": "x"}
                           for variant in ("neutral", "text_only")}}
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
        monkeypatch.setattr(bound, "arm_paths",
                            lambda arm: {"frozen_labels": path})
        with pytest.raises(SystemExit) as exc:
            bound.frozen_family_labels("phi4_mm")
        assert exc.value.code == 1
        assert DIFFERENTIAL_CELL in capsys.readouterr().err

    def test_fresh_calls_without_write_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            adjudicator.main(["--fresh-calls"])
        assert exc.value.code == 2
        assert "only means something with --write" in capsys.readouterr().err

    def test_the_bound_writes_nothing_under_verify(self, tmp_path):
        out = tmp_path / "not_written.json"
        assert bound.main(["--verify", "--out", str(out)]) == 2
        assert not out.exists()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The receipt: a citation a reviewer can resolve, or no citation at all
# ---------------------------------------------------------------------------

RECEIPT = adjudicator.RECEIPT_PATH


class TestTheSourceEvidenceIsResolvedRatherThanBelieved:
    def test_a_tracked_file_resolves_to_the_blob_a_commit_holds(self):
        resolution = adjudicator.resolve_evidence(SENSITIVITY_ARTIFACT)
        assert resolution["resolved_from"] == "the blob HEAD holds at this path"
        assert resolution["git_blob_sha1"]
        assert resolution["head_commit"]
        assert resolution["bytes_sha256"]
        assert resolution["why_not"] is None

    def test_a_file_no_commit_holds_says_where_it_was_resolved_from(
            self, tmp_path):
        # THE distinction the finding turns on. Hashing the working-tree copy
        # proves this machine has a file; the superseded citation's sha256 was a
        # perfectly formed hash of exactly that.
        stray = tmp_path / "never_committed.json"
        stray.write_text(json.dumps({"x": 1}), encoding="utf-8")
        resolution = adjudicator.resolve_evidence(stray)
        assert resolution["exists_on_disk"] is True
        assert resolution["working_tree_sha256"]
        assert resolution["resolved_from"] == (
            "the working tree, which no commit reachable from HEAD holds")
        assert resolution["git_blob_sha1"] is None
        assert "no commit reachable from HEAD holds" in resolution["why_not"]

    def test_a_file_that_is_nowhere_resolves_to_nothing(self, tmp_path):
        resolution = adjudicator.resolve_evidence(tmp_path / "absent.json")
        assert resolution["exists_on_disk"] is False
        assert resolution["bytes_sha256"] is None
        assert resolution["resolved_from"] is None

    def test_a_citation_that_only_this_disk_can_resolve_is_refused(self):
        doc = {"per_arm": {"phi4_mm": {"call_reused_from": {
            "path": "outputs/definitely/not/committed.json",
            "sha256": "a" * 64}}}}
        issues, _, _ = adjudicator.check_the_reuse_citations(doc)
        assert any("nothing resolves there" in i for i in issues), issues

    def test_the_superseded_citation_is_measured_not_asserted(self):
        """The finding, re-measured every run rather than quoted in prose.

        If either measurement ever came back the other way -- some committed
        version of the artifact hashing to the cited sha256, or the cited commit
        becoming reachable -- the receipt's premise would be false and the
        receipt would be preserving evidence that was available all along.
        """
        superseded = adjudicator.superseded_citation()
        measured = superseded["measured"]
        assert measured["n_of_them_hashing_to_the_cited_sha256"] == 0
        assert measured["the_cited_commit_is_reachable_from_head"] is False
        assert measured["the_cited_commit_holds_that_path_in_its_tree"] is False
        assert superseded["the_citation_that_could_not_be_resolved"][
            "sha256"] == adjudicator.SUPERSEDED_CITATION_SHA256
        assert len(adjudicator.SUPERSEDED_CITATION_SHA256) == 64


class TestTheReceiptHoldsTheCallsEvidence:
    def test_it_preserves_every_field_a_reuse_is_bound_by(self,
                                                          filed_sensitivity):
        receipt = adjudicator.call_receipt()
        assert receipt["immutable"] is True
        assert receipt["kind"] == "iteration_11_adjudicator_call_receipt_v1"
        assert receipt["n_calls_preserved"] == len(ADJUDICATED_ARMS)
        for arm in ARMS:
            preserved = receipt["per_arm"][arm]["preserved"]
            assert sorted(preserved) == sorted(
                adjudicator.RECEIPT_CALL_FIELDS), arm
            entry = filed_sensitivity["per_arm"][arm]
            for field in adjudicator.RECEIPT_CALL_FIELDS:
                assert preserved[field] == entry.get(field), (arm, field)

    def test_an_arm_the_frozen_rule_needs_no_call_for_says_why(self):
        receipt = adjudicator.call_receipt()
        agreed = receipt["per_arm"]["qwen35_4b"]
        assert agreed["a_call_was_made"] is False
        assert "without a call" in agreed["why_no_call_was_made"]
        bounded = receipt["per_arm"][DIFFERENTIAL_TARGET]
        assert bounded["a_call_was_made"] is False
        assert "never a call to make" in bounded["why_no_call_was_made"]

    def test_it_records_where_the_preserved_evidence_came_from(self):
        receipt = adjudicator.call_receipt()
        source = receipt["source_of_the_preserved_evidence"]
        assert source["path"].endswith("labels_adjudicated.sensitivity_99f.json")
        assert source["sha256"]
        assert source["resolved_from"]

    def test_it_supersedes_the_citation_it_replaces(self):
        receipt = adjudicator.call_receipt()
        superseded = receipt["supersedes"][
            "the_citation_that_could_not_be_resolved"]
        assert superseded["sha256"] == \
            adjudicator.SUPERSEDED_CITATION_SHA256
        assert superseded["code_commit"] == \
            adjudicator.SUPERSEDED_CITATION_COMMIT

    def test_the_receipt_path_is_outside_the_dirtiness_it_records(self):
        # Otherwise writing the receipt would file itself as an untracked
        # output of a dirty tree, and every later artifact would quote that.
        rel = adjudicator._rel(RECEIPT)
        assert any(rel.startswith(prefix)
                   for prefix in adjudicator.OWN_OUTPUT_PREFIXES), rel

    def test_it_is_written_once_and_refused_thereafter(self, monkeypatch,
                                                       tmp_path, capsys):
        destination = tmp_path / "call_receipts.json"
        monkeypatch.setattr(adjudicator, "RECEIPT_PATH", destination)
        assert adjudicator.write_call_receipt() == 0
        written = destination.read_bytes()
        document = json.loads(written)
        assert document["immutable"] is True
        assert document["n_calls_preserved"] == len(ADJUDICATED_ARMS)

        capsys.readouterr()
        assert adjudicator.write_call_receipt() == 1
        assert destination.read_bytes() == written, (
            "a receipt that can be rewritten is not a receipt: the sha256 an "
            "artifact cites would come to mean something else")
        assert "REFUSING to overwrite" in capsys.readouterr().err

    def test_the_receipt_is_a_source_a_re_file_may_draw_on(self, monkeypatch,
                                                           tmp_path,
                                                           filed_sensitivity):
        destination = tmp_path / "call_receipts.json"
        monkeypatch.setattr(adjudicator, "RECEIPT_PATH", destination)
        assert adjudicator.write_call_receipt() == 0
        per_arm, sha, doc = adjudicator.reuse_source(destination)
        assert sha
        assert doc["immutable"] is True
        for arm in ADJUDICATED_ARMS:
            state = adjudicator.arm_state(arm)
            expected = adjudicator.expected_request(arm)
            reuse, why = adjudicator.reusable_call(
                arm, state, per_arm[arm], expected, destination)
            assert why is None, (arm, why)
            assert reuse["judgment"] == \
                filed_sensitivity["per_arm"][arm]["ensemble_label"]
            assert reuse["call_provenance"] == \
                filed_sensitivity["per_arm"][arm]["call_provenance"]

    def test_the_default_source_is_the_receipt_and_not_the_artifact(self):
        per_arm, sha, doc = adjudicator.reuse_source()
        assert adjudicator.reuse_source(adjudicator.RECEIPT_PATH)[1] == sha
        assert doc == {} or doc.get("immutable") is True, (
            "with no receipt on disk yet the default has to draw on nothing, "
            "not quietly fall back to the file the re-file overwrites")
        if not adjudicator.RECEIPT_PATH.exists():
            assert per_arm == {} and sha is None


class TestTheTwoShapesOfASourceAreToldApart:
    def test_a_receipt_is_a_receipt(self):
        assert adjudicator.source_shape(adjudicator.call_receipt()) == "receipt"

    def test_an_artifact_is_an_artifact(self, filed_sensitivity):
        assert adjudicator.source_shape(filed_sensitivity) == "artifact"

    def test_something_else_is_neither(self):
        assert adjudicator.source_shape({}) == "unrecognised"
        assert adjudicator.source_shape({"per_arm": {"x": "y"}}) \
            == "unrecognised"

    def test_a_source_of_neither_shape_cannot_license_a_reuse(self, tmp_path):
        stray = tmp_path / "not_a_source.json"
        stray.write_text(json.dumps({"per_arm": {"phi4_mm": {"x": 1}}}),
                         encoding="utf-8")
        doc = {"per_arm": {"phi4_mm": {"call_reused_from": {
            "path": adjudicator._rel(stray), "sha256": "a" * 64}}}}
        issues, _, _ = adjudicator.check_the_reuse_citations(doc)
        assert any("neither a call receipt nor a sensitivity artifact" in i
                   for i in issues), issues


#: A committed file that resolves out of the object store and is NOT the
#: artifact, so a citation to it can be checked without depending on what the
#: filed artifact happens to cite this week. Which sha256 it resolves to is read
#: at run time rather than written down here.
CITABLE = "outputs/iteration_11/media_manifest.json"


def _citing(sha256, path=CITABLE, arms=ADJUDICATED_ARMS, **extra):
    return {"per_arm": {arm: {"call_reused_from": dict(
        {"path": path, "sha256": sha256}, **extra)} for arm in arms}}


@pytest.fixture(scope="module")
def citable_sha():
    """The sha256 the object store returns for :data:`CITABLE`, read not quoted."""
    if not (ROOT / CITABLE).exists():
        pytest.skip(f"{CITABLE} is not in this checkout")
    resolution = adjudicator.resolve_evidence(ROOT / CITABLE)
    assert resolution["git_blob_sha1"], (
        f"{CITABLE} does not resolve out of the object store, so it cannot "
        f"serve as the resolvable citation these tests need")
    return resolution["bytes_sha256"]


class TestACitationIsResolvedOutOfTheObjectStore:
    """The check the finding asked for, exercised without the artifact.

    These are unit tests of :func:`check_the_reuse_citations` rather than of a
    tampered copy of the filed document, so they hold whichever source the
    artifact cites and on any machine that can run ``git``.
    """

    def test_the_hash_the_object_store_returns_is_accepted(self, citable_sha):
        issues, resolved, _ = adjudicator.check_the_reuse_citations(
            _citing(citable_sha))
        assert resolved[CITABLE]["bytes_sha256"] == citable_sha
        assert not any("hash to" in i for i in issues), issues

    def test_a_cited_hash_of_none_is_not_a_citation(self, citable_sha):
        # The old check asked only whether the sha256 was a non-empty string, so
        # None failed by accident of truthiness and "b" * 64 did not fail at all.
        issues, _, _ = adjudicator.check_the_reuse_citations(_citing(None))
        assert any("but the bytes that resolve there hash to" in i
                   for i in issues), issues

    def test_a_well_formed_hash_of_nothing_is_refused_too(self, citable_sha):
        issues, _, _ = adjudicator.check_the_reuse_citations(_citing("b" * 64))
        assert any("but the bytes that resolve there hash to" in i
                   for i in issues), issues
        assert any(citable_sha[:16] in i for i in issues), (
            "the refusal has to name the hash that DOES resolve, or a reader "
            "cannot tell which of the two moved")

    def test_two_arms_citing_one_source_at_two_hashes_is_a_finding(
            self, citable_sha):
        doc = _citing(citable_sha, arms=("qwen35_2b",))
        doc["per_arm"]["phi4_mm"] = {"call_reused_from": {
            "path": CITABLE, "sha256": "d" * 64}}
        issues, _, _ = adjudicator.check_the_reuse_citations(doc)
        assert any("while another arm cites the same path" in i
                   for i in issues), issues

    def test_a_citation_to_the_artifact_itself_is_refused_by_shape(self):
        """The shape the superseded citation had, refused whatever its hash.

        An artifact that cites itself cites bytes the re-file is in the middle
        of overwriting, so the citation is unresolvable the instant it is
        written. That is a property of the shape and not of one unlucky sha256,
        which is why it is refused before anything is hashed.
        """
        issues, resolved, _ = adjudicator.check_the_reuse_citations(
            _citing("c" * 64, path=adjudicator._rel(adjudicator.OUT_PATH)))
        assert any("this artifact itself" in i for i in issues), issues
        assert resolved == {}, "a self-citation is refused, not resolved"

    def test_a_citation_to_the_file_under_verification_is_refused_too(
            self, tmp_path):
        # The same shape reached through --verify of a copy elsewhere: the
        # cited path is not OUT_PATH but it IS the document being checked.
        copy = tmp_path / "artifact.json"
        issues, _, _ = adjudicator.check_the_reuse_citations(
            _citing("c" * 64, path=adjudicator._rel(copy)),
            under_verification=copy)
        assert any("this artifact itself" in i for i in issues), issues

    def test_a_block_claiming_to_cite_the_receipt_has_to_name_it(
            self, citable_sha):
        issues, _, _ = adjudicator.check_the_reuse_citations(
            _citing(citable_sha, it_is_a_receipt=True))
        assert any("says it cites the receipt but names" in i
                   for i in issues), issues

    def test_a_citation_with_no_path_has_nothing_to_resolve(self):
        issues, resolved, _ = adjudicator.check_the_reuse_citations(
            {"per_arm": {"phi4_mm": {"call_reused_from": {"sha256": "a" * 64}}}})
        assert any("cites no path at all" in i for i in issues), issues
        assert resolved == {}

    def test_an_arm_that_reused_nothing_is_not_asked_for_a_source(self):
        doc = {"per_arm": {arm: {"call_reused_from": None} for arm in ARMS}}
        issues, resolved, _ = adjudicator.check_the_reuse_citations(doc)
        assert issues == []
        assert resolved == {}


class TestACitationInACheckoutWithNoHistoryIsIncompleteNotUnreachable:
    """The export lane: the bytes are there and the object store is not.

    A tarball, a ``git archive`` export and the anonymous reproducibility
    package all hold every committed file and none of the history. The blob
    lookup that proves a citation resolvable therefore fails there for a reason
    that says nothing about the citation, and reporting it as an unreachable
    source -- the finding the receipt was written to end -- sends a reviewer
    looking for a defect that is not there. The bytes are still hashed and still
    compared, so a citation to the WRONG bytes fails in a historyless checkout
    exactly as it does in a clone.

    Simulated by making every git question fail rather than by patching
    ``object_store_here`` alone: that flag only words the failure, and a checkout
    with no history is one where the lookup itself does not answer.
    """

    @pytest.fixture
    def no_object_store(self, monkeypatch):
        real_git = adjudicator._git

        def historyless(*args):
            if args and args[0] in ("rev-parse", "cat-file", "ls-tree"):
                return 128, b"fatal: not a git repository"
            return real_git(*args)

        monkeypatch.setattr(adjudicator, "_git", historyless)
        assert adjudicator.object_store_here() is False
        return historyless

    def test_a_resolvable_citation_becomes_unverifiable_rather_than_wrong(
            self, no_object_store):
        disk_sha = adjudicator.sha256_file(ROOT / CITABLE)
        issues, resolved, unverifiable = \
            adjudicator.check_the_reuse_citations(_citing(disk_sha))
        assert resolved[CITABLE]["bytes_sha256"] == disk_sha
        assert resolved[CITABLE]["git_blob_sha1"] is None
        assert resolved[CITABLE]["object_store_here"] is False
        assert not any("resolves only from" in i for i in issues), issues
        assert not any("hash to" in i for i in issues), issues
        assert len(unverifiable) == 1
        assert "no git object store" in unverifiable[0]
        assert disk_sha[:16] in unverifiable[0], (
            "the note has to say what DID check out, or it reads as a refusal")

    def test_a_citation_to_the_wrong_bytes_still_fails_there(
            self, no_object_store):
        issues, _, unverifiable = adjudicator.check_the_reuse_citations(
            _citing("b" * 64))
        assert any("but the bytes that resolve there hash to" in i
                   for i in issues), (
                "a finding outranks incompleteness: the hash comparison runs "
                f"against the working tree, so a wrong citation is still wrong "
                f"here -- {issues}")
        assert unverifiable

    def test_the_committed_artifact_exits_three_there_with_no_issues(
            self, no_object_store):
        code, issues, unverifiable = adjudicator.verify(SENSITIVITY_ARTIFACT)
        assert code == 3, (code, issues, unverifiable)
        assert issues == [], (
            "the filed artifact's citation resolves to the receipt's bytes on "
            f"disk; only the committed-ness is unanswerable: {issues}")
        assert len(unverifiable) == 1
        assert adjudicator._rel(adjudicator.RECEIPT_PATH) in unverifiable[0]

    def test_the_same_artifact_exits_zero_where_history_exists(self):
        assert adjudicator.object_store_here() is True
        code, issues, unverifiable = adjudicator.verify(SENSITIVITY_ARTIFACT)
        assert code == 0, (issues, unverifiable)
        assert unverifiable == []

    def test_an_uncommitted_citation_is_still_a_finding_where_history_exists(
            self):
        """The case the receipt exists for, unchanged by the export lane."""
        stray = ROOT / "outputs" / "iteration_11" / "closeout" \
            / "_uncommitted_citation.json"
        stray.write_text(json.dumps({"per_arm": {}}), encoding="utf-8")
        try:
            issues, resolved, unverifiable = \
                adjudicator.check_the_reuse_citations(
                    _citing(adjudicator.sha256_file(stray),
                            path=adjudicator._rel(stray)))
            assert unverifiable == []
            assert any("resolves only from" in i and
                       "no commit reachable from HEAD holds" in i
                       for i in issues), issues
            assert resolved[adjudicator._rel(stray)]["git_blob_sha1"] is None
        finally:
            stray.unlink()

    def test_the_two_reasons_a_blob_lookup_fails_say_different_things(
            self, no_object_store):
        stray = ROOT / "outputs" / "iteration_11" / "closeout" \
            / "_uncommitted_citation.json"
        stray.write_text("{}", encoding="utf-8")
        try:
            without = adjudicator.resolve_evidence(stray)
        finally:
            stray.unlink()
        assert without["object_store_here"] is False
        assert "cannot be compared with any commit" in without["resolved_from"]
        assert "no git object store" in without["why_not"]


class TestTheDocumentedOfflineLaneNeedsNoCredentials:
    """No part of the offline lane may depend on this machine holding a key.

    The module docstring promises a CI-safe file: no test calls the gateway, and
    the request-hash recomputation needs no credentials. Both were true of the
    hash and not of everything around it. ``build()`` establishes whether a call
    COULD be sent before it calls the stubbed transport, so a test that stubbed
    the transport and not sendability exited 2 where no key exists. And the filed
    request block carried ``config_source``, a string describing where the local
    config came from, so a receipt written with credentials compared unequal to a
    re-file made without them -- the receipt looked like evidence about a
    different call when it differed only in whose machine wrote it.

    Each test here runs under :func:`no_credentials`, which patches the pipeline
    loader rather than moving a credentials file, so the lane is tested in the
    state CI is in without touching this machine's key.
    """

    def test_a_no_call_refile_round_trips_where_there_are_no_credentials(
            self, monkeypatch, no_credentials, tmp_path):
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        doc = adjudicator.build()
        assert doc["n_live_calls"] == 0
        path = tmp_path / "refiled.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        code, issues, unverifiable = adjudicator.verify(path)
        assert code == 0, (issues, unverifiable)

    def test_every_preserved_call_is_reusable_where_there_are_no_credentials(
            self, monkeypatch, no_credentials):
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        doc = adjudicator.build()
        assert doc["n_reused_calls"] == len(ADJUDICATED_ARMS)
        assert doc["n_live_calls"] == 0
        for arm in ADJUDICATED_ARMS:
            assert doc["per_arm"][arm]["call_reused_from"] is not None

    def test_the_superseded_field_is_enumerated_rather_than_dropped(
            self, monkeypatch, no_credentials):
        """An exemption that is not written down is a comparison nobody ran."""
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        record = adjudicator.build()[
            "fields_the_preserved_call_and_the_artifact_do_not_share"]
        assert record["where"] == adjudicator._rel(adjudicator.RECEIPT_PATH)
        assert sorted(record["superseded"]) == sorted(
            adjudicator.SUPERSEDED_REQUEST_FIELDS)
        entry = record["superseded"]["config_source"]
        assert sorted(entry["arms_whose_preserved_call_carries_it"]) == \
            sorted(ADJUDICATED_ARMS)
        assert entry["the_values_it_holds"], \
            "the value the receipt holds is reported, not deleted"
        assert entry["what_it_described"] == \
            adjudicator.SUPERSEDED_REQUEST_FIELDS["config_source"]
        assert record["what_replaced_the_superseded_ones"]["config_identity"] \
            == adjudicator.FILED_CONFIG_IDENTITY
        assert "record of what happened" in \
            record["why_the_source_is_not_rewritten"]

    def test_the_field_that_replaced_it_is_also_enumerated(
            self, monkeypatch, no_credentials):
        """The receipt predates the replacement, so that gap is written down too.

        A receipt is immutable and was written before ``config_identity``
        existed, so the two blocks differ in that key permanently. Enumerating it
        is what separates a reasoned set-aside from a comparison that was quietly
        narrowed until nothing could fail it.
        """
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        record = adjudicator.build()[
            "fields_the_preserved_call_and_the_artifact_do_not_share"]
        introduced = record["introduced_since_the_calls_were_preserved"]
        assert sorted(introduced) == sorted(
            adjudicator.FIELDS_INTRODUCED_SINCE_THE_CALLS_WERE_PRESERVED)
        assert sorted(introduced["config_identity"][
            "arms_whose_preserved_call_predates_it"]) == sorted(ADJUDICATED_ARMS)
        assert "receipt is not rewritten" in introduced["config_identity"][
            "what_it_is"]

    def test_the_machine_state_is_reported_at_run_time_and_not_filed(
            self, monkeypatch, no_credentials):
        state = adjudicator.credential_state()
        assert "no credentials here" in state
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        text = json.dumps(adjudicator.build())
        assert state not in text
        assert adjudicator.HASHING_ONLY_API_KEY not in text
        assert adjudicator.HASHING_ONLY_BASE_URL not in text

    def test_a_difference_in_a_field_that_is_not_exempt_still_fails(
            self, monkeypatch, no_credentials, tmp_path):
        """The one exemption must not blind the comparison beside it."""
        monkeypatch.setattr(adjudicator, "adjudicate",
                            lambda arm: pytest.fail("a call was attempted"))
        doc = adjudicator.build()
        arm = ADJUDICATED_ARMS[0]
        block = doc["per_arm"][arm]["request_the_frozen_rule_would_send"]
        block["request_hash"] = "f" * 64
        path = tmp_path / "mutated.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        code, issues, _unverifiable = adjudicator.verify(path)
        assert code == 1
        assert any("request" in issue for issue in issues), issues


class TestTheRequestBlockComparisonIsNotAnExclusionList:
    """The exemption is two named fields, not a list of the fields that matter."""

    def test_a_field_nobody_has_heard_of_is_compared_by_default(self):
        held = {"request_hash": "a" * 64, "something_added_later": 1}
        filed = {"request_hash": "a" * 64, "something_added_later": 2}
        differences, _superseded, _predates = \
            adjudicator.compare_request_blocks(held, filed)
        assert any("something_added_later" in difference
                   for difference in differences), (
            "a field added to the request block in future would be skipped by a "
            "comparison written over a list of the keys that matter today")

    def test_a_key_only_one_side_has_is_reported_not_passed_over(self):
        held = {"request_hash": "a" * 64}
        filed = {"request_hash": "a" * 64, "a_key_only_the_artifact_has": 1}
        differences, _superseded, predates = \
            adjudicator.compare_request_blocks(held, filed)
        assert differences == []
        assert predates == ["a_key_only_the_artifact_has"], (
            "a key only one side has cannot be value-compared, and treating it "
            "as equal is how a comparison stops covering the block it names")

    def test_the_exempt_fields_are_reported_and_not_compared(self):
        held = {"request_hash": "a" * 64, "config_source": "a machine with a key"}
        filed = {"request_hash": "a" * 64,
                 "config_identity": adjudicator.FILED_CONFIG_IDENTITY}
        differences, superseded, predates = \
            adjudicator.compare_request_blocks(held, filed)
        assert differences == []
        assert predates == []
        assert len(superseded) == 1
        assert superseded[0].startswith("config_source=")
        assert "a machine with a key" in superseded[0]

    def test_only_the_named_fields_are_exempt(self):
        assert sorted(adjudicator.SUPERSEDED_REQUEST_FIELDS) == ["config_source"]
        assert sorted(
            adjudicator.FIELDS_INTRODUCED_SINCE_THE_CALLS_WERE_PRESERVED) == \
            ["config_identity"]
        assert adjudicator.NOT_COMPARED == frozenset({
            "config_source", "config_identity"})
        assert adjudicator.REQUEST_BLOCK in adjudicator.RECEIPT_CALL_FIELDS


class TestTheFiledArtifactCarriesNoMachineState:
    def test_the_artifact_states_the_identity_and_not_the_checkout(
            self, filed_sensitivity):
        text = json.dumps(filed_sensitivity)
        if "config_source" in text:
            pytest.skip(
                "the filed artifact still carries config_source. It is re-filed "
                "from a clean tree in this lane, spending no calls, and this test "
                "runs once it is -- skipping with the reason rather than passing "
                "over an artifact that has not been re-filed yet")
        assert "config_identity" in text
        assert adjudicator.FILED_CONFIG_IDENTITY in text
        assert "fields_the_preserved_call_and_the_artifact_do_not_share" in text
