#!/usr/bin/env python3
"""Bound what the differential exclusion costs, instead of pricing it once.

``CMST_456921/text_only`` was refused by judge A's provider in the
``ministral3_3b`` arm alone, so the union dropped it from all four and took the
whole family with it: the confirmatory analysis runs on 98 families. Two numbers
have been filed about what that costs, and neither is a bound.

The first is the judge-B shift in ``cross_model_analysis.json``
(``max_differential_shift`` = 0.004545). It is an empirical sensitivity measured
with a DIFFERENT instrument -- judge B alone, which is vision-ablated, so its
absolute level is not the confirmatory quantity -- and it prices the change in
the FAMILY SET, not the missing ensemble outcome. It is retained here verbatim
and labelled as what it is.

The second is the point estimate you get by putting the family back. That is
better, but it is still one number where the honest question is a range: what
could the missing label have been, and does any admissible value change a
verdict?

This script answers the range. It is exact rather than searched, and the reason
is a property of the frozen bootstrap: ``paired_bootstrap_samples`` draws its
resample indices from ``Random(seed)`` as a function of the SEED and the FAMILY
COUNT alone, never of the data. With 99 families fixed, every one of the 5000
resample means is therefore an AFFINE function of the single missing score x,

    sample_b(x) = a_b + m_b * x,      m_b = -k_b / 99   for Delta_TV

where ``k_b`` is how often the restored family appears in resample b. The
observed mean is affine too. So the two-sided p-value's tail counts are
monotone step functions of x that jump only at ``x*_b = -a_b / m_b``, and the
worst case over an interval is attained at an endpoint or at one of those
breakpoints -- enumerated exactly, then cross-checked against a dense grid and
against the frozen estimator itself.

The answers this files, for the primary estimand Delta_TV:

  * ``qwen35_4b`` -- A and B agree exactly, so the frozen routing rule pins the
    ensemble label at 0.0 without a call. H2's SIGN matches the reference for
    every x in [0, 1] (the mean stays between +0.0929 and +0.0828), and its raw
    p IMPROVES from 0.0256 at 98 families to 0.0224 at 99. The verdict is not
    narrow because of the exclusion; it is narrow on its own, and the bound says
    exactly how narrow: p crosses alpha = 0.05 at x = 0.9, which is 0.9 of a
    rubric away from the label both primaries gave.
  * ``ministral3_3b`` -- judge A has no label, so no ensemble label exists. Over
    the WHOLE admissible score range its mean stays in [-0.1312, -0.1211], the
    sign cannot flip, and the p-value sits at the 1/5000 floor for every x. H3's
    refutation survives every value the missing label could have taken. That is
    the genuine worst-case bound the judge-B shift was never able to be.
  * ``qwen35_2b`` and ``phi4_mm`` -- one fresh kimi-k3 adjudication each, filed
    by ``iter11_adjudicate_sensitivity_cell.py``. Both signs hold.
    ``phi4_mm``'s raw p improves from 0.0136 to 0.0088; ``qwen35_2b``'s is
    already at the 1/5000 bootstrap floor at 98 families (0.0002) and stays
    there at 99, which is a p-value that CANNOT improve rather than one that
    did not -- the floor is ``bootstrap_two_sided_p``'s own, and a claim that
    both improved would have been a claim about the floor.

A fourth configuration is filed alongside the three above and is NOT licensed by
the evidence: if every arm's cell were ranged over the whole rubric, ignoring
that three of the four labels are known, H2's raw p reaches 0.0528 and Holm
retains it. It is recorded because a bound that only reports the comfortable
configuration is not a bound, and because the distance between the two is the
distance between "the label is pinned by primary agreement" and "the label is
unknown" -- which is precisely the difference the differential exclusion makes
in ONE arm out of four.

Usage:
    python3 scripts/iter11_differential_censoring_bound.py            # write
    python3 scripts/iter11_differential_censoring_bound.py --verify   # no write

Exit codes: 0 written / reproduced exactly; 1 a recomputed field disagrees with
the filed one, or the closed form and the frozen estimator disagree; 2 a required
input is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.adjudication import enforce_coherence  # noqa: E402
from causal_mllm.evaluation.censoring import (  # noqa: E402
    DIFFERENTIAL_CELL,
    DIFFERENTIAL_TARGET,
    EVIDENCE_ARTIFACT,
)
from causal_mllm.evaluation.estimands import (  # noqa: E402
    compute_family_estimands,
)
from causal_mllm.evaluation.hypotheses import holm_bonferroni  # noqa: E402
from causal_mllm.evaluation.schema import make_judge_record  # noqa: E402
from causal_mllm.replay import reproduction  # noqa: E402

FAMILY, VARIANT = DIFFERENTIAL_CELL.split("/")
ARMS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")
HYPOTHESES = {"H1": "qwen35_2b", "H2": "qwen35_4b", "H3": "ministral3_3b",
              "H4": "phi4_mm"}
PRIMARY = "Delta_TV"
ESTIMANDS = ("Delta_T", "Delta_V", "Delta_TV", "order_effect", "history_effect")

#: How the missing cell's score enters each estimand. Delta_TV is
#: ``Y_cross_modal - Y_text_only - Y_vision_only + Y_neutral`` so the missing
#: ``text_only`` score carries coefficient -1; Delta_T carries +1; the other
#: three do not involve ``text_only`` at all and are therefore CONSTANT in x,
#: which is itself worth stating rather than leaving to be noticed.
SLOPE = {"Delta_T": 1.0, "Delta_V": 0.0, "Delta_TV": -1.0,
         "order_effect": 0.0, "history_effect": 0.0}

#: The rubric's own range for ``unsafe_compliance_score``. The worst case is
#: taken over the whole of it rather than over the values observed in this
#: dataset, because a bound fitted to the observed labels would be a bound on
#: what the judges happened to say and not on what the missing one could say.
SCORE_LO, SCORE_HI = 0.0, 1.0

#: The grid the closed form is cross-checked against. Coarse enough to run in
#: seconds, fine enough that a closed form wrong anywhere but a razor edge would
#: show: the p-value is a step function whose steps are 2/5000 = 0.0004 wide.
CHECK_GRID_STEPS = 401

ALPHA = 0.05
N_FAMILIES_COMMITTED = 98
N_FAMILIES_SENSITIVITY = 99
FAMILY_KEPT_OUT = "CMST_795308"

JUDGE_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "judge"
GENERATIONS_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "generations"
OUT_DIR = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "differential_censoring"
OUT_PATH = OUT_DIR / "differential_censoring_bound.json"
SENSITIVITY_LABELS = OUT_DIR / "labels_adjudicated.sensitivity_99f.json"
COMMITTED_ANALYSIS = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "cross_model" / "cross_model_analysis.json"
FROZEN_LABELS = "llm_labels_adjudicated.json"

OWN_OUTPUT_PREFIXES = (
    "outputs/iteration_11/analysis/differential_censoring/",)


def _load_script(name: str):
    """The 11.8 analysis module, for its estimator rather than a copy of it.

    Loaded by path because ``scripts/`` is not a package. Reusing ``estimate``,
    ``load_arm`` and ``_reference_family_estimands`` is what makes the numbers
    here the frozen estimator's numbers over a different family set, instead of
    a second implementation that agrees today.
    """
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_differential_bound",
        REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


analysis = _load_script("iter11_cross_model_analysis")


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    ``raise SystemExit("text")`` exits 1 and prints the text, and 1 is the code
    this script documents for a recomputed field disagreeing with the filed one.
    A missing input reported as a disagreement sends whoever reads the exit code
    looking for a drifted number instead of an absent file, so the message goes
    to stderr and the code goes to ``SystemExit`` on its own: 2 for an input
    that is not there, 1 for evidence that is there and does not agree.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require(path: Path, what: str):
    if not path.exists():
        fatal(f"no {what} at {_rel(path)}. This stage recomputes from "
              f"committed evidence and invents nothing, so a missing input is "
              f"a missing input rather than something to estimate around", 2)
    return path


def inputs() -> dict:
    """Every committed artifact this recomputation reads, by hash.

    This is the provenance, and it is stronger than naming the commit that
    produced the file: the bound is a pure function of these inputs, so binding
    them says exactly what would have to change for the numbers to change. It is
    also what makes the artifact deterministic -- no timestamp, no commit, no
    tree state -- so ``--verify`` can compare the WHOLE document exactly rather
    than a list of fields someone remembered to exclude. The 11.8 analysis it
    sits beside is built the same way for the same reason.
    """
    labels = sensitivity_labels()
    paths = {
        "sensitivity_labels": SENSITIVITY_LABELS,
        "committed_cross_model_analysis": COMMITTED_ANALYSIS,
        "differential_uniform_probe": REPO_ROOT / EVIDENCE_ARTIFACT,
    }
    for arm in ARMS:
        arm_paths_ = arm_paths(arm)
        for name in ("cells", "report", "frozen_labels", "judge_A", "judge_B",
                     "replay"):
            paths[f"{arm}.{name}"] = arm_paths_[name]
    missing = sorted(key for key, path in paths.items()
                     if not path.exists())
    return {
        "sha256": {key: _sha256(path)
                   for key, path in sorted(paths.items())},
        "paths": {key: _rel(path) for key, path in sorted(paths.items())},
        "n_inputs": len(paths),
        "missing_inputs": missing,
        "sensitivity_labels_calls": {
            "live_in_that_filing": labels.get("n_live_calls"),
            "reused_in_that_filing": labels.get("n_reused_calls"),
            "why_both_are_recorded": (
                "the labels come from two kimi-k3 calls made once, and a "
                "re-file reuses them bound by the request hash each one "
                "answered, so the filing this bound read may itself have made "
                "none. Zero live calls in a re-file is not zero calls in the "
                "evidence: the call provenance each entry carries is the "
                "original call's, timestamp and provider response id "
                "included"),
        },
        "why_bound_by_hash": (
            "the bound is a pure function of these files: no live call, no "
            "clock and no tree state enters it. Binding them is what lets "
            "--verify compare the whole artifact exactly instead of comparing "
            "a chosen subset and calling the rest provenance"),
    }


def sensitivity_labels() -> dict:
    """The ensemble labels for the restored cell, from the separate artifact."""
    _require(SENSITIVITY_LABELS,
             "sensitivity label artifact (run "
             "scripts/iter11_adjudicate_sensitivity_cell.py --write)")
    return _load_json(SENSITIVITY_LABELS)


# ---------------------------------------------------------------------------
# Reconstructing the 99th family
# ---------------------------------------------------------------------------

def arm_paths(arm: str) -> dict[str, Path]:
    root = JUDGE_ROOT / arm
    generations = sorted(GENERATIONS_ROOT.joinpath(arm).glob(
        "*/replay_outputs.jsonl"))
    return {
        "cells": root / "evaluation_results" / "evaluation_outputs.jsonl",
        "report": root / "evaluation_results" / "evaluation_report.json",
        "frozen_labels": root / FROZEN_LABELS,
        "judge_A": root / "llm_labels_judge_A.json",
        "judge_B": root / "llm_labels_judge_B.json",
        "replay": generations[0] if generations else
                  GENERATIONS_ROOT / arm / "MISSING_replay_outputs.jsonl",
    }


def arm_config(arm: str) -> dict:
    return _load_json(_require(arm_paths(arm)["report"],
                               f"{arm}'s evaluation report"))["config"]


def replay_family(arm: str) -> dict[str, dict]:
    """The six replay records for the dropped family, by variant.

    Read from the confirmatory run rather than regenerated: the sensitivity has
    to describe the responses that were actually judged, and a re-generation
    would be a different panel wearing the same name.
    """
    path = _require(arm_paths(arm)["replay"], f"{arm}'s replay outputs")
    by_variant = {r["variant"]: r for r in _load_jsonl(path)
                  if r["family_id"] == FAMILY}
    if len(by_variant) != 6:
        fatal(f"{arm}'s replay outputs hold {len(by_variant)} variant(s) of "
              f"{FAMILY}, not the six the frozen estimator needs to produce "
              f"any estimand for it", 1)
    return by_variant


def frozen_family_labels(arm: str) -> dict[str, dict]:
    """The five labels this family already has in the sealed ensemble set."""
    path = _require(arm_paths(arm)["frozen_labels"], f"{arm}'s frozen labels")
    labels = dict(_load_json(path)["labels"][FAMILY])
    if VARIANT in labels:
        fatal(f"{arm}'s sealed {FROZEN_LABELS} already holds "
              f"{DIFFERENTIAL_CELL}, so the cell was not excluded from that "
              f"arm and this whole sensitivity is about a different panel "
              f"than the one on disk", 1)
    if len(labels) != 5:
        fatal(f"{arm}'s frozen labels hold {len(labels)} of {FAMILY}'s "
              f"variants; expected the five the union left behind", 1)
    return labels


def observed_scores() -> dict:
    """What the four sealed label sets actually scored, measured not described.

    The bound is taken over the rubric's whole range rather than over the values
    these judges happened to produce, and saying so is only honest if those
    values are read. The sentence this feeds once claimed the 2,388 committed
    labels sit "on a 0.05 grid"; two of them, 0.72 and 0.88, do not, and nothing
    in the script asked the labels whether they did. A claim about evidence that
    the evidence is never asked about is a claim waiting to be wrong.
    """
    counts: dict[float, int] = {}
    per_arm: dict[str, int] = {}
    missing_score = 0
    for arm in ARMS:
        path = _require(arm_paths(arm)["frozen_labels"],
                        f"{arm}'s frozen labels")
        labels = _load_json(path)["labels"]
        n = 0
        for by_variant in labels.values():
            for judgment in by_variant.values():
                n += 1
                score = judgment.get("unsafe_compliance_score")
                if score is None:
                    missing_score += 1
                    continue
                counts[float(score)] = counts.get(float(score), 0) + 1
        per_arm[arm] = n
    distinct = sorted(counts)
    off_grid = sorted(value for value in distinct
                      if abs(value / 0.05 - round(value / 0.05)) > 1e-9)
    return {
        "n_labels": sum(per_arm.values()),
        "n_labels_per_arm": per_arm,
        "n_labels_carrying_no_score": missing_score,
        "n_distinct_values": len(distinct),
        "distinct_values": distinct,
        "observed_min": distinct[0],
        "observed_max": distinct[-1],
        "off_a_0_05_grid": off_grid,
        "on_a_0_05_grid": not off_grid,
        "top_of_the_rubric_observed": SCORE_HI in counts,
        "headroom_above_the_observed_max": SCORE_HI - distinct[-1],
        "read_from": "the unsafe_compliance_score of every label in the four "
                     "sealed llm_labels_adjudicated.json files",
    }


def judgment_at(score: float) -> dict:
    """A coherent judgment carrying only the score the bound varies.

    Coherence is enforced rather than assumed: the estimand reads the score and
    nothing else, but a bound computed over judgments the rubric could not
    produce would be a bound over a wider space than the one that exists.
    """
    return enforce_coherence({
        "refusal_type": "none", "unsafe_compliance_score": float(score),
        "compliance_level": 0, "safe_redirection": False, "confidence": 0.9,
        "rationale": "worst-case sensitivity: score varied over the rubric "
                     "range, other fields set coherently"})


def family_records(arm: str, text_only_judgment: dict) -> list[dict]:
    """The restored family's six records, through the production schema."""
    replay = replay_family(arm)
    labels = frozen_family_labels(arm)
    labels[VARIANT] = text_only_judgment
    prov = {"backend": "llm_ensemble",
            "annotator_id": "llm_ensemble",
            "adjudicated": True,
            "sensitivity": "the five other variants are the sealed ensemble "
                           "labels; text_only is supplied by "
                           "iter11_adjudicate_sensitivity_cell.py or by the "
                           "worst-case sweep"}
    return [make_judge_record(replay[variant], labels[variant], prov)
            for variant in sorted(replay)]


def records_at(arm: str, score: float) -> list[dict]:
    paths = arm_paths(arm)
    _require(paths["cells"], f"{arm}'s evaluated cells")
    return (_load_jsonl(paths["cells"])
            + family_records(arm, judgment_at(score)))


def family_estimands_at(arm: str, score: float) -> dict:
    config = arm_config(arm)
    return compute_family_estimands(
        records_at(arm, score),
        outcome=config.get("primary_outcome", "unsafe_compliance_score"))


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

def _portable_sum(values) -> float:
    """The left-to-right sum, spelled out because the builtin one is not.

    CPython 3.12 sums floats with Neumaier compensation; 3.10 and 3.11 do not.
    The same column of the same committed floats therefore returns
    ``1.8999999999999997`` from ``sum`` under 3.10.20 and ``1.9`` under 3.12.13,
    while this loop returns the first under both and :func:`math.fsum` returns
    the second under both. An explicit loop is the one summation whose result is
    a property of the numbers rather than of the interpreter, and it is also
    what the frozen estimator computes under the certified interpreter -- so a
    closed form built on it bounds the estimator that produced the sealed
    analysis instead of bounding a slightly different arithmetic.
    """
    total = 0.0
    for value in values:
        total += value
    return total


def decompose(family_estimands_at_zero: dict, estimand: str,
              n_bootstrap: int, seed: int, slope: float | None = None) -> dict:
    """The affine decomposition of every resample mean in ``estimand``.

    The family estimands MUST have been built with the missing cell's score at
    ``SCORE_LO``, which is 0.0. Anchoring there is what makes ``mean_at(dec, x)``
    the mean AT x rather than the mean at an offset from some other origin: the
    decomposition carries no origin term, so an anchor that was not zero would
    silently shift every bound by a constant and nothing downstream would notice.

    Every sum in here goes through :func:`_portable_sum`, never through the
    builtin ``sum``, so an intercept, a breakpoint, a tail count, a p-value and
    the count of evaluated breakpoints are bit-identical under CPython 3.10 and
    3.12. That is the difference between a bound that verifies anywhere and one
    that verifies only where it was filed: measured on this evidence and filed in
    ``summation`` below, the builtin moves most of the 5,000 resample intercepts
    of the qwen35_4b column between the two interpreters, by up to a couple of
    units in the last place, which is enough to carry one resample across the
    null, one step of p, and two candidates out of the breakpoint count.

    The frozen estimator in :mod:`causal_mllm.evaluation.bootstrap` keeps its own
    builtin ``sum`` and is NOT changed here: the sealed 98-family analysis was
    produced by it, so making it portable would mean re-deriving sealed evidence.
    Every comparison against it below is therefore tolerance-based, and the
    correctly rounded alternative to this summation is filed beside it so a
    reader who recomputes with :func:`math.fsum` or NumPy can see the distance.
    """
    ids = sorted(family_estimands_at_zero)
    if FAMILY not in ids:
        fatal(f"{FAMILY} is not among the {len(ids)} families, so there is "
              f"nothing to decompose", 1)
    j, n = ids.index(FAMILY), len(ids)
    column = [family_estimands_at_zero[f][estimand] for f in ids]

    # The same construction paired_bootstrap_samples uses, in the same order,
    # so ``a`` and ``k`` are the frozen bootstrap's own intercepts and slopes
    # rather than a re-implementation of them.
    rng = random.Random(seed)
    intercepts, counts = [], []
    moved_by_the_summation = 0
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        values = [column[i] for i in indices]
        mean = _portable_sum(values) / n
        if math.fsum(values) / n != mean:
            moved_by_the_summation += 1
        intercepts.append(mean)
        counts.append(indices.count(j))
    slope = SLOPE[estimand] if slope is None else slope
    portable = _portable_sum(column)
    correctly_rounded = math.fsum(column)
    return {
        "estimand": estimand, "n_families": n, "family_index": j,
        "intercepts": intercepts,
        "slopes": [slope * k / n for k in counts],
        "family_slope": slope,
        "observed_slope": slope / n,
        "observed_intercept": portable / n,
        "n_resamples": n_bootstrap, "seed": seed,
        "constant_in_x": slope == 0.0,
        "anchored_at": SCORE_LO,
        "summation": {
            "used": "an explicit left-to-right loop (_portable_sum)",
            "portable_sum_of_the_family_column": portable,
            "correctly_rounded_sum_of_the_same_column": correctly_rounded,
            "their_difference": portable - correctly_rounded,
            "n_resample_intercepts_the_two_summations_disagree_on":
                moved_by_the_summation,
            "n_resamples": n_bootstrap,
            "why": (
                "the two sums above are of one committed column of "
                f"{n} floats and they are not the same float. The builtin "
                "sum() returns the first under CPython 3.10 and 3.11 and the "
                "second under CPython 3.12 and later, because 3.12 sums "
                "floats with Neumaier compensation, so the builtin is not "
                "portable in EITHER direction and is not used. The explicit "
                "loop is portable and is also what the frozen estimator "
                "computes under the certified interpreter, which is why this "
                f"decomposition uses it: {moved_by_the_summation} of the "
                f"{n_bootstrap} resample intercepts would be a different "
                "float under the correctly rounded summation, and at a zero "
                "crossing one such resample is one step of p. Reproduce the "
                "interpreter difference on any column of floats with: "
                "python3 -c \"import functools, math, operator; "
                "v=[0.1]*3+[0.2]*3+[1/3]*3; print(repr(functools.reduce("
                "operator.add, v, 0.0)), repr(sum(v)), repr(math.fsum(v)))\""),
        },
    }


def mean_at(dec: dict, x: float) -> float:
    return dec["observed_intercept"] + dec["observed_slope"] * x


def tails_at(dec: dict, x: float) -> tuple[int, int]:
    """The two tail counts directly. O(n) and used to check the O(log n) form."""
    below = above = 0
    for a_b, m_b in zip(dec["intercepts"], dec["slopes"]):
        value = a_b + m_b * x
        below += value <= 0.0
        above += value >= 0.0
    return below, above


def p_from_tails(below: int, above: int, n: int) -> float:
    """``bootstrap_two_sided_p``'s rule, on counts rather than on a sample list."""
    return min(1.0, max(2.0 * min(below, above) / n, 1.0 / n))


def p_at(dec: dict, x: float) -> float:
    return p_from_tails(*tails_at(dec, x), dec["n_resamples"])


def index_tails(dec: dict) -> dict:
    """Precomputed breakpoint arrays, so a sweep is O(log n) per point.

    Split by the sign of the slope because the two signs flip in opposite
    directions: for ``m_b < 0`` the resample mean is non-increasing in x, so it
    is at or below zero exactly when x has passed its breakpoint.
    """
    negative, positive, zero_below, zero_above = [], [], 0, 0
    for a_b, m_b in zip(dec["intercepts"], dec["slopes"]):
        if m_b == 0.0:
            zero_below += a_b <= 0.0
            zero_above += a_b >= 0.0
        elif m_b < 0.0:
            negative.append(-a_b / m_b)
        else:
            positive.append(-a_b / m_b)
    negative.sort()
    positive.sort()
    return {"negative": negative, "positive": positive,
            "zero_slope_below": zero_below, "zero_slope_above": zero_above}


def fast_tails(index: dict, x: float) -> tuple[int, int]:
    below = index["zero_slope_below"]
    above = index["zero_slope_above"]
    negative, positive = index["negative"], index["positive"]
    below += bisect_right(negative, x)
    above += len(negative) - bisect_left(negative, x)
    below += len(positive) - bisect_left(positive, x)
    above += bisect_right(positive, x)
    return below, above


def fast_p(dec: dict, index: dict, x: float) -> float:
    return p_from_tails(*fast_tails(index, x), dec["n_resamples"])


def ci_envelope(dec: dict) -> list[float]:
    """An interval guaranteed to contain the percentile CI for every x.

    Each resample mean is affine, so over [lo, hi] it stays inside the interval
    spanned by its two endpoint values. The 2.5th percentile of 5000 numbers
    each confined to its own interval is confined by the 2.5th percentiles of
    the lower and upper envelopes -- a bound, and reported as one rather than as
    the CI, because the ordering of the resamples changes with x and the
    percentile itself is not affine.
    """
    lo, hi = SCORE_LO, SCORE_HI
    lower, upper = [], []
    for a_b, m_b in zip(dec["intercepts"], dec["slopes"]):
        first, second = a_b + m_b * lo, a_b + m_b * hi
        lower.append(min(first, second))
        upper.append(max(first, second))
    return [_percentile(sorted(lower), 2.5),
            _percentile(sorted(upper), 97.5)]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """The frozen estimator's own percentile rule."""
    position = (len(sorted_values) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) \
        * (position - low)


def worst_case(dec: dict) -> dict:
    """The exact extremes of the mean and of the p-value over the score range.

    Exact, not searched: the mean is affine so its extremes are the endpoints,
    and the p-value's tail counts are monotone step functions whose only
    breakpoints are the ``x*_b``, so its extremes are at an endpoint or at a
    breakpoint. Every breakpoint inside the range is evaluated.

    The COUNT of evaluated breakpoints is filed, and so are the two margins that
    decide it, because that count is the one field here whose value can rest on a
    last bit: a breakpoint is inside the range or it is not, and two breakpoints
    are one candidate or two, and both questions are answered by comparing
    floats. On this evidence both margins are far narrower than
    :data:`reproduction.FLOAT_TOLERANCE` -- they are filed in
    ``breakpoint_margins`` rather than quoted here, so the numbers a reader sees
    are the ones this checkout measured. The counts are therefore exactly
    reproducible, because every input to them is one portable summation of a
    committed column, but they are not ROBUST: they would move by one if that
    column moved. What a verdict reads is not one of them, and that is also
    measured rather than asserted -- the p at a breakpoint that hugs an endpoint
    is filed beside the p at that endpoint, in resample steps.
    """
    index = index_tails(dec)
    n = dec["n_resamples"]
    inside = sorted(x for x in index["negative"] + index["positive"]
                    if SCORE_LO <= x <= SCORE_HI)
    candidates = {SCORE_LO, SCORE_HI}
    candidates.update(inside)
    rows = []
    for x in sorted(candidates):
        below, above = fast_tails(index, x)
        rows.append({"x": x, "mean": mean_at(dec, x),
                     "p": p_from_tails(below, above, n),
                     "below": below, "above": above})
    worst = max(rows, key=lambda r: (r["p"], r["x"]))
    best = min(rows, key=lambda r: (r["p"], -r["x"]))
    over_alpha = [r for r in rows if r["p"] > ALPHA]
    means = [mean_at(dec, SCORE_LO), mean_at(dec, SCORE_HI)]

    distinct = sorted(set(inside))
    distinct_gaps = [b - a for a, b in zip(distinct, distinct[1:])]
    tolerance = reproduction.FLOAT_TOLERANCE
    if inside:
        nearest = min(inside, key=lambda x: min(x - SCORE_LO, SCORE_HI - x))
        nearest_endpoint = (SCORE_LO if nearest - SCORE_LO <= SCORE_HI - nearest
                            else SCORE_HI)
        nearest_distance = abs(nearest - nearest_endpoint)
        nearest_p = p_from_tails(*fast_tails(index, nearest), n)
        endpoint_p = p_from_tails(*fast_tails(index, nearest_endpoint), n)
    else:
        nearest = nearest_endpoint = nearest_distance = None
        nearest_p = endpoint_p = None
    smallest_gap = min(distinct_gaps) if distinct_gaps else None
    return {
        "score_range": [SCORE_LO, SCORE_HI],
        "n_breakpoints_evaluated": len(rows),
        "breakpoint_margins": {
            "n_breakpoints_inside_the_range": len(inside),
            "n_distinct_breakpoints_inside_the_range": len(distinct),
            "n_exact_duplicates": len(inside) - len(distinct),
            "smallest_gap_between_distinct_breakpoints": smallest_gap,
            "nearest_breakpoint_to_a_range_endpoint": nearest,
            "the_endpoint_it_is_nearest_to": nearest_endpoint,
            "its_distance_from_that_endpoint": nearest_distance,
            "float_tolerance": tolerance,
            "the_count_rests_on_a_margin_narrower_than_the_float_tolerance":
                bool(inside) and (
                    (nearest_distance is not None
                     and nearest_distance <= tolerance)
                    or (smallest_gap is not None
                        and smallest_gap <= tolerance)),
            "nearest_breakpoint_p": nearest_p,
            "nearest_endpoint_p": endpoint_p,
            "their_difference_in_resample_steps":
                None if nearest_p is None or endpoint_p is None
                else abs(nearest_p - endpoint_p) / p_step(dec),
            "why_exact_duplicates_are_not_a_margin": (
                "two breakpoints can be the SAME float, which happens when two "
                "resamples draw the same multiset of families or two quotients "
                "round alike. A duplicate is not a last-bit hazard: it is one "
                "candidate under every interpreter, so it is counted separately "
                "from the smallest gap between DISTINCT breakpoints, which is "
                "the margin a rounding difference could actually close"),
            "what_it_means": (
                "a breakpoint this close to an endpoint, or this close to "
                "another breakpoint, makes the count of evaluated candidates a "
                "last-bit question, so the count is reproducible only because "
                "every float behind it is one portable summation of a committed "
                "column, and it is filed with its margins rather than alone. "
                "The p-value is not last-bit sensitive in the same way: the two "
                "p-values above differ by the filed number of resample steps, "
                "which is inside the documented p tolerance, so no verdict "
                "reads the membership decision"),
        },
        "mean_range": [min(means), max(means)],
        "mean_at_the_range_ends": {"at_0": means[0], "at_1": means[1]},
        "sign_can_flip": (means[0] > 0.0) != (means[1] > 0.0),
        "worst_p": worst["p"],
        "worst_p_at_x": worst["x"],
        "best_p": best["p"],
        "best_p_at_x": best["x"],
        "p_exceeds_alpha_anywhere": bool(over_alpha),
        "smallest_x_whose_p_exceeds_alpha":
            min((r["x"] for r in over_alpha), default=None),
        "n_x_evaluated_whose_p_exceeds_alpha": len(over_alpha),
        "ci_envelope": ci_envelope(dec),
        "ci_envelope_is_a_bound_not_the_ci": True,
    }


def ties_at(dec: dict, x: float, band: float | None = None) -> int:
    """Resample means that are zero at ``x`` to within the float tolerance.

    These are the only points where the closed form and the frozen estimator can
    disagree, and they are counted rather than tolerated blindly. A resample at
    zero is counted in BOTH tails by ``bootstrap_two_sided_p`` (``<= null`` and
    ``>= null``), so which tail a value that is zero up to rounding lands in is
    decided by the last bit -- and there are THREE float paths to that value:
    the estimator sums 99 per-family numbers and divides, the closed form
    evaluates ``a_b + m_b * x``, and the breakpoint index compares x against
    ``-a_b / m_b``. All three are the same number in exact arithmetic and can be
    one resample apart near a crossing.

    The band is ``reproduction.FLOAT_TOLERANCE``, the tolerance this repository
    already documents for a re-derived continuous field, rather than a new
    constant chosen to make this check pass. Measured as a residual rather than
    as equality with a stored breakpoint: a breakpoint is a quotient, so asking
    whether x EQUALS one is a stricter question than asking whether the resample
    mean is zero here, and it is the second question that decides the tail.
    """
    band = reproduction.FLOAT_TOLERANCE if band is None else band
    return sum(1 for a_b, m_b in zip(dec["intercepts"], dec["slopes"])
               if m_b != 0.0 and abs(a_b + m_b * x) <= band)


def p_step(dec: dict) -> float:
    """One resample step of the two-sided p-value."""
    return 2.0 / dec["n_resamples"]


def check_the_closed_form(dec: dict) -> dict:
    """Does the affine form agree with the frozen estimator, and with itself?

    Three separate agreements, because a bound is only as good as the claim
    that it bounds the right quantity:

      * the affine mean and p must equal what ``estimate`` produces from
        records built the production way, at the score this arm actually files
        AND at a score that is not the decomposition's own anchor;
      * the O(log n) tail counts must equal the O(n) ones across a dense grid,
        except at exact ties, which are counted and named rather than waved
        through;
      * the worst case over breakpoints must be at least as large as the worst
        p on that grid, so the enumeration is not missing a breakpoint.

    The p-value comparison uses the tolerance ``causal_mllm.replay.reproduction``
    already documents for a re-derivation, widened only as far as the exact tie
    count at that point requires and no further. Both numbers are recorded, so a
    reader can see whether the agreement was exact, tie-sized, or tolerance-sized.
    """
    index = index_tails(dec)
    documented = reproduction.p_value_tolerance(dec["n_resamples"])
    probes = []
    issues: list[str] = []
    for x in sorted({SCORE_LO, SCORE_HI, 0.35, dec["filed_score"]}
                    - {None}):
        estimated = analysis.estimate(
            family_estimands_at_from(dec, x), dec["config"])[dec["estimand"]]
        ties = ties_at(dec, x)
        mean_difference = abs(estimated["mean"] - mean_at(dec, x))
        p_difference = abs(estimated["bootstrap_p_two_sided"] - p_at(dec, x))
        # Resample means at zero license one step each; anything beyond the
        # documented tolerance is a defect in the decomposition, not in the
        # arithmetic.
        licensed = max(documented, ties * p_step(dec))
        mean_ok = mean_difference <= reproduction.FLOAT_TOLERANCE
        p_ok = p_difference <= licensed
        if not mean_ok:
            issues.append(
                f"{dec['estimand']}: the affine mean at x={x} is "
                f"{mean_at(dec, x)!r} but the frozen estimator over records "
                f"built the production way gives {estimated['mean']!r} "
                f"(difference {mean_difference:.3g})")
        if not p_ok:
            issues.append(
                f"{dec['estimand']}: the closed-form p at x={x} is "
                f"{p_at(dec, x)!r} but the frozen estimator gives "
                f"{estimated['bootstrap_p_two_sided']!r} -- a difference of "
                f"{p_difference / p_step(dec):.2f} resample step(s) with only "
                f"{ties} resample mean(s) at zero within "
                f"{reproduction.FLOAT_TOLERANCE:g} of it, so this is not "
                f"rounding at a crossing and the decomposition is wrong")
        probes.append({
            "x": x, "n_resample_means_at_zero": ties,
            "frozen_estimator_mean": estimated["mean"],
            "closed_form_mean": mean_at(dec, x),
            "frozen_estimator_p": estimated["bootstrap_p_two_sided"],
            "closed_form_p": p_at(dec, x),
            "mean_difference": mean_difference,
            "p_difference": p_difference,
            "p_difference_in_resample_steps": p_difference / p_step(dec),
            "mean_tolerance": reproduction.FLOAT_TOLERANCE,
            "mean_agrees_within_that_tolerance": mean_ok,
            "p_tolerance_licensed": licensed,
            "p_tolerance_documented": documented,
            "p_agrees_within_the_licensed_tolerance": p_ok,
            "ok": mean_ok and p_ok,
        })

    step = (SCORE_HI - SCORE_LO) / (CHECK_GRID_STEPS - 1)
    grid_worst = 0.0
    disagreements = tied_disagreements = 0
    for i in range(CHECK_GRID_STEPS):
        x = SCORE_LO + i * step
        direct, fast = tails_at(dec, x), fast_tails(index, x)
        if direct != fast:
            disagreements += 1
            # Counted only where a disagreement actually occurred: this is O(n)
            # per point and the grid is not, and a tie band that is only ever
            # consulted when the two methods differ is still consulted exactly
            # when it matters.
            ties = ties_at(dec, x)
            if ties and abs(direct[0] - fast[0]) <= ties \
                    and abs(direct[1] - fast[1]) <= ties:
                tied_disagreements += 1
            else:
                issues.append(
                    f"{dec['estimand']}: the indexed tail counts disagree with "
                    f"the direct ones at x={x!r} -- {direct} vs {fast} with "
                    f"{ties} resample mean(s) at zero within "
                    f"{reproduction.FLOAT_TOLERANCE:g}, so the disagreement is "
                    f"not a rounding tie at a crossing")
                break
        grid_worst = max(grid_worst, fast_p(dec, index, x))

    bound = worst_case(dec)
    if grid_worst > bound["worst_p"]:
        issues.append(
            f"{dec['estimand']}: a grid point reaches p={grid_worst}, above the "
            f"closed-form worst case {bound['worst_p']}, so the breakpoint "
            f"enumeration is missing one")
    return {
        "probes": probes,
        "no_exactness_booleans": {
            "worst_mean_difference_over_these_probes": max(
                (p["mean_difference"] for p in probes), default=0.0),
            "mean_tolerance": reproduction.FLOAT_TOLERANCE,
            "worst_p_difference_over_these_probes_in_resample_steps": max(
                (p["p_difference_in_resample_steps"] for p in probes),
                default=0.0),
            "p_tolerance_in_resample_steps":
                reproduction.P_VALUE_TOLERANCE_STEPS,
            "why": (
                "these probes used to file mean_exact and p_exact, and both "
                "flipped between CPython 3.10.20 and 3.12.13 on this evidence "
                "while every verdict, sign and count a hypothesis test reads "
                "stayed put. An exactness boolean has no headroom at all, so "
                "the last bit of a float decides it and the interpreter that "
                "summed the float decides the last bit. A tolerance boolean is "
                "decided by the arithmetic instead: the worst difference over "
                "these probes is filed beside a tolerance orders of magnitude "
                "wider, so the answer is the same under any interpreter that "
                "reads IEEE doubles. Asking the weaker question hides nothing, "
                "because both the difference and the tolerance are filed -- it "
                "is the only one of the two questions that has a stable "
                "answer"),
        },
        "grid": {
            "n_points": CHECK_GRID_STEPS,
            "step": step,
            "worst_p_on_the_grid": grid_worst,
            "closed_form_worst_p": bound["worst_p"],
            "grid_never_exceeds_the_closed_form":
                grid_worst <= bound["worst_p"],
            "n_tail_count_disagreements": disagreements,
            "n_of_those_at_a_zero_crossing": tied_disagreements,
            "tie_band": reproduction.FLOAT_TOLERANCE,
        },
        "issues": issues,
        "ok": not issues,
    }


def family_estimands_at_from(dec: dict, score: float) -> dict:
    """Rebuild the family estimands at ``score``, for the cross-check only."""
    arm = dec["arm"]
    return compute_family_estimands(
        records_at(arm, score),
        outcome=dec["config"].get("primary_outcome", "unsafe_compliance_score"))


def build_decomposition(arm: str, estimand: str = PRIMARY,
                        filed_score: float | None = None) -> dict:
    """One arm's decomposition, anchored at the bottom of the rubric range.

    ``filed_score`` is the score this arm's ensemble label actually carries, and
    it is kept on the decomposition so the closed-form check probes the point
    that matters rather than only points chosen for convenience.
    """
    config = arm_config(arm)
    dec = decompose(family_estimands_at(arm, SCORE_LO), estimand,
                    int(config["n_bootstrap"]), int(config["seed"]))
    dec["arm"] = arm
    dec["config"] = config
    dec["filed_score"] = filed_score
    return dec


# ---------------------------------------------------------------------------
# The reference over the same families
# ---------------------------------------------------------------------------

def reference_block() -> dict:
    """The sealed Iteration 10 reference, recomputed over each family set.

    A sign comparison between two different family sets is not a comparison, so
    the 99-family sensitivity has to be read against the reference restricted to
    the SAME 99 families. Recomputed from the sealed per-cell scores through the
    frozen estimator -- the same route ``iter11_reference_restriction.py`` uses
    for the 98-family number -- rather than read off a sentence.
    """
    family_estimands, config = analysis._reference_family_estimands()
    out = {}
    for label, dropped in (
            ("at_100_families", ()),
            ("at_99_families", (FAMILY_KEPT_OUT,)),
            ("at_98_families_committed", (FAMILY_KEPT_OUT, FAMILY))):
        subset = {fid: value for fid, value in family_estimands.items()
                  if fid not in dropped}
        estimated = analysis.estimate(subset, config)
        out[label] = {
            "n_families": estimated["n_families"],
            "families_dropped": sorted(dropped),
            PRIMARY: estimated[PRIMARY],
            "sign": analysis._sign(estimated[PRIMARY]["bootstrap_mean"]),
        }
    out["why_recomputed"] = (
        "the arms' sensitivity runs on 99 families, so the reference sign has to "
        "come from the same 99; comparing a 99-family arm against a 98-family "
        "reference would attribute the difference between the two family sets to "
        "the model")
    return out


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def label_source(labels: dict, arm: str) -> dict:
    entry = labels["per_arm"][arm]
    status = entry["ensemble_label_status"]
    source = {
        "derived_from_primary_agreement":
            "judge A and judge B agree exactly, so the frozen routing rule "
            "(adjudicate_pairwise_with_model: items with full agreement keep the "
            "agreed label) determines the ensemble label with no call",
        "requires_adjudication":
            f"a fresh {ADJUDICATOR_NAME} call on this cell alone, at the frozen "
            f"seed, filed by iter11_adjudicate_sensitivity_cell.py",
        "not_derivable":
            "judge A has no label for this cell in this arm, and "
            "compute_pairwise_agreement requires full mutual coverage, so no "
            "ensemble label exists or can be produced from the committed "
            "evidence",
    }[status]
    judgment = entry.get("ensemble_label")
    return {
        "status": status,
        "how_it_was_obtained": source,
        "score": None if judgment is None
                 else judgment["unsafe_compliance_score"],
        "adjudicated_by": entry.get("adjudicated_by"),
        "primary_A_score": (entry.get("primary_A_judgment") or {}).get(
            "unsafe_compliance_score"),
        "primary_B_score": (entry.get("primary_B_judgment") or {}).get(
            "unsafe_compliance_score"),
        "differing_fields": entry.get("differing_fields"),
    }


ADJUDICATOR_NAME = "kimi-k3"


def arm_block(arm: str, labels: dict, reference: dict) -> dict:
    """One arm at 98 families, at 99, and over the whole admissible range."""
    source = label_source(labels, arm)
    committed_arm = analysis.load_arm(arm)
    committed = committed_arm["recomputed"]
    dec = build_decomposition(arm, filed_score=source["score"])
    closed_form = check_the_closed_form(dec)
    bound = worst_case(dec)
    ref99 = reference["at_99_families"]
    ref98 = reference["at_98_families_committed"]

    block = {
        "label": source,
        "n_families_committed": committed["n_families"],
        "n_families_sensitivity": dec["n_families"],
        "closed_form_check": closed_form,
        "arithmetic_of_the_decomposition": dec["summation"],
        "committed_at_98_families": {
            "reference_sign": ref98["sign"],
            PRIMARY: committed[PRIMARY],
            "sign": analysis._sign(committed[PRIMARY]["bootstrap_mean"]),
            "sign_matches_reference":
                analysis._sign(committed[PRIMARY]["bootstrap_mean"])
                == ref98["sign"],
        },
        "worst_case_over_the_whole_rubric_range": bound,
        "estimands_constant_in_the_missing_score": sorted(
            name for name, value in SLOPE.items() if value == 0.0),
        "estimands_that_also_move": sorted(
            name for name, value in SLOPE.items() if value != 0.0),
    }

    if source["score"] is None:
        block["at_99_families"] = {
            "available": False,
            "why": source["how_it_was_obtained"],
            "what_is_reported_instead": (
                "the whole admissible range, which is a stronger statement than "
                "a point estimate would have been: every value the missing "
                "ensemble label could take is covered"),
        }
    else:
        score = source["score"]
        estimated = analysis.estimate(family_estimands_at(arm, score),
                                      dec["config"])
        sign = analysis._sign(estimated[PRIMARY]["bootstrap_mean"])
        block["at_99_families"] = {
            "available": True,
            "score_used": score,
            "reference_sign": ref99["sign"],
            PRIMARY: estimated[PRIMARY],
            "sign": sign,
            "sign_matches_reference": sign == ref99["sign"],
            "all_five_estimands": {name: estimated[name]
                                   for name in ESTIMANDS},
            "closed_form_agrees_with_the_frozen_estimator": {
                "closed_form_mean": mean_at(dec, score),
                "frozen_estimator_mean": estimated[PRIMARY]["mean"],
                "mean_difference": abs(estimated[PRIMARY]["mean"]
                                       - mean_at(dec, score)),
                "mean_tolerance": reproduction.FLOAT_TOLERANCE,
                "mean_agrees_within_that_tolerance":
                    abs(estimated[PRIMARY]["mean"] - mean_at(dec, score))
                    <= reproduction.FLOAT_TOLERANCE,
                "closed_form_p": p_at(dec, score),
                "frozen_estimator_p":
                    estimated[PRIMARY]["bootstrap_p_two_sided"],
                "p_difference": abs(
                    estimated[PRIMARY]["bootstrap_p_two_sided"]
                    - p_at(dec, score)),
                "p_tolerance": reproduction.p_value_tolerance(
                    int(dec["config"]["n_bootstrap"])),
                "p_agrees_within_that_tolerance": abs(
                    estimated[PRIMARY]["bootstrap_p_two_sided"]
                    - p_at(dec, score)) <= reproduction.p_value_tolerance(
                    int(dec["config"]["n_bootstrap"])),
                "why_not_exactness": (
                    "the two fields here used to be mean_matches and "
                    "p_matches, both compared with <= 1e-12. For a mean that "
                    "is a float tolerance. For a p-value it is not a tolerance "
                    "at all: a bootstrap p can only move in steps of "
                    "2/n_resamples, which at 5,000 resamples is 0.0004, so "
                    "that comparison demanded an agreement eight orders of "
                    "magnitude tighter than the quantity's own granularity and "
                    "could pass only where the two float paths to the same "
                    "number happened to land on the same side of every zero "
                    "crossing. Each quantity is now held to the tolerance "
                    "documented for its kind, with the difference and the "
                    "tolerance both filed"),
            },
        }

    # The A/B bound: what the estimate would be on either primary's own label
    # instead of the ensemble's. Narrower than the whole-range bound and asked
    # for separately, because it is the bound that uses only labels this
    # repository already has.
    ab = {}
    for name, key in (("judge_A_alone", "primary_A_score"),
                      ("judge_B_alone", "primary_B_score")):
        score = source[key]
        if score is None:
            ab[name] = {"available": False,
                        "why": f"{key.split('_')[1]} has no label for this cell "
                               f"in this arm"}
            continue
        estimated = analysis.estimate(family_estimands_at(arm, score),
                                      dec["config"])
        ab[name] = {
            "available": True, "score": score,
            PRIMARY: estimated[PRIMARY],
            "sign": analysis._sign(estimated[PRIMARY]["bootstrap_mean"]),
            "sign_matches_reference_at_99":
                analysis._sign(estimated[PRIMARY]["bootstrap_mean"])
                == ref99["sign"],
            "closed_form_p": p_at(dec, score),
        }
    scores = [v["score"] for v in ab.values() if v["available"]]
    if source["score"] is not None:
        scores.append(source["score"])
    block["ab_bound"] = {
        "per_primary": ab,
        "scores_considered": sorted(set(scores)),
        "sign_matches_reference_at_every_score": all(
            analysis._sign(mean_at(dec, s)) == ref99["sign"] for s in scores),
        "p_range_over_those_scores": [
            min((p_at(dec, s) for s in scores), default=None),
            max((p_at(dec, s) for s in scores), default=None)],
        "what_it_is": (
            "the estimate on each primary's OWN label for the cell, instead of "
            "the ensemble's. It is a bound over labels this repository already "
            "holds, and it is narrower than the whole-range bound because it "
            "does not ask what a label nobody produced could have been"),
    }
    return block


# ---------------------------------------------------------------------------
# Holm-Bonferroni over the four confirmatory tests
# ---------------------------------------------------------------------------

def raw_p(block: dict, at_worst_case: bool) -> float:
    if at_worst_case:
        return block["worst_case_over_the_whole_rubric_range"]["worst_p"]
    at99 = block.get("at_99_families") or {}
    if at99.get("available"):
        return at99[PRIMARY]["bootstrap_p_two_sided"]
    # No 99-family point estimate exists, so the confirmatory number stands.
    return block["committed_at_98_families"][PRIMARY]["bootstrap_p_two_sided"]


def holm_configurations(blocks: dict, reference: dict) -> dict:
    """Four Holm runs, from the committed analysis to the unlicensed worst case.

    Configurations (c) and (d) differ in ONE respect and the difference is the
    finding: (c) ranges only the label that is genuinely missing, while (d)
    ranges all four as though none were known. Holm's adjusted p is monotone in
    every raw p, so maximising each arm's p simultaneously is the least
    favourable configuration for every hypothesis at once -- which is what makes
    (d) a bound and not a selection.
    """
    committed = {hid: blocks[arm]["committed_at_98_families"][PRIMARY][
        "bootstrap_p_two_sided"] for hid, arm in HYPOTHESES.items()}
    at99 = {hid: raw_p(blocks[arm], at_worst_case=False)
            for hid, arm in HYPOTHESES.items()}
    genuine = dict(at99)
    genuine["H3"] = blocks[DIFFERENTIAL_TARGET][
        "worst_case_over_the_whole_rubric_range"]["worst_p"]
    adversarial = {hid: raw_p(blocks[arm], at_worst_case=True)
                   for hid, arm in HYPOTHESES.items()}

    def run(raw: dict, label: str, panel: str, licensed: bool,
            note: str) -> dict:
        corrected = holm_bonferroni(raw, alpha=ALPHA)
        signs = {}
        for hid, arm in HYPOTHESES.items():
            block = blocks[arm]
            if panel == "98":
                signs[hid] = block["committed_at_98_families"][
                    "sign_matches_reference"]
            else:
                bound = block["worst_case_over_the_whole_rubric_range"]
                at99 = block.get("at_99_families") or {}
                signs[hid] = (
                    at99["sign_matches_reference"] if at99.get("available")
                    else not bound["sign_can_flip"]
                    and analysis._sign(bound["mean_range"][0])
                    == reference["at_99_families"]["sign"])
        return {
            "label": label, "family_set": panel, "raw_p": raw,
            "holm": corrected, "sign_matches_reference": signs,
            "licensed_by_the_evidence": licensed, "note": note,
            "all_four_signs_match_or_all_four_mismatch_as_committed": None,
        }

    return {
        "alpha": ALPHA, "n_tests": len(HYPOTHESES),
        "hypothesis_to_arm": dict(sorted(HYPOTHESES.items())),
        "configurations": {
            "a_committed_98_families": run(
                committed, "the confirmatory analysis as filed", "98", True,
                "the pre-registered comparison, over the common panel every arm "
                "actually judged"),
            "b_99_families_at_the_labels_the_frozen_rule_gives": run(
                at99, "the family restored where a label exists", "99", True,
                "H3 stays at 98 families because no ensemble label exists for "
                "that cell in that arm; the other three are recomputed over 99 "
                "against the reference restricted to the same 99"),
            "c_genuine_worst_case": run(
                genuine, "the missing Ministral label ranged over the whole "
                         "rubric", "99", True,
                "the only label that is genuinely unknown is allowed to be "
                "anything the rubric permits; the other three are pinned by "
                "primary agreement or by a filed adjudication"),
            "d_fully_adversarial": run(
                adversarial, "every arm's cell ranged over the whole rubric",
                "99", False,
                "NOT licensed by the evidence: three of the four labels are "
                "known, one by exact primary agreement and two by a filed "
                "adjudication. Filed because a bound that reports only the "
                "comfortable configuration is not a bound, and because the "
                "distance between this and (c) is exactly the distance between "
                "'the label is known' and 'the label is missing'"),
        },
    }


# ---------------------------------------------------------------------------
# Pooled H5, and the judge-B sensitivity that is retained rather than replaced
# ---------------------------------------------------------------------------

def pooled_block(labels: dict) -> dict:
    """H5 at 98 as filed, at 99 with the restored family, and over the range."""
    arms = {arm: analysis.load_arm(arm) for arm in ARMS}
    pooled98 = analysis.estimate(analysis.pooled_family_estimands(arms),
                                 arms[ARMS[0]]["config"])

    scores = {arm: label_source(labels, arm)["score"] for arm in ARMS}
    per_arm_99 = {arm: family_estimands_at(arm, scores[arm] or SCORE_LO)
                  for arm in ARMS}
    # The builtin sum() over these four floats is interpreter-dependent in its
    # last bit for exactly the reason decompose() documents, and the pooled
    # decomposition inherits every bit of it, so the portable sum is used here
    # too: a four-term sum is not too short to matter, it is just shorter to
    # overlook.
    pooled99 = {fid: {name: _portable_sum(
                          [per_arm_99[a][fid][name] for a in ARMS])
                      / len(ARMS) for name in ESTIMANDS}
                for fid in sorted(per_arm_99[ARMS[0]])}
    config = arms[ARMS[0]]["config"]
    estimated99 = analysis.estimate(pooled99, config)

    # Only the differential target's label is unknown, and pooling averages four
    # models, so the pooled family value carries slope -1/4 in that score.
    dec = decompose(pooled99, PRIMARY, int(config["n_bootstrap"]),
                    int(config["seed"]), slope=SLOPE[PRIMARY] / len(ARMS))
    bound = worst_case(dec)
    return {
        "n_families_committed": pooled98["n_families"],
        "n_families_sensitivity": estimated99["n_families"],
        "committed_at_98_families": {
            PRIMARY: pooled98[PRIMARY],
            "sign": analysis._sign(pooled98[PRIMARY]["bootstrap_mean"])},
        "at_99_families": {
            "missing_score_set_to": {arm: scores[arm] for arm in ARMS},
            PRIMARY: estimated99[PRIMARY],
            "sign": analysis._sign(estimated99[PRIMARY]["bootstrap_mean"]),
            "all_five_estimands": {name: estimated99[name]
                                   for name in ESTIMANDS}},
        "worst_case_over_the_missing_label": bound,
        "arithmetic_of_the_decomposition": dec["summation"],
        "why_the_pooled_slope_is_a_quarter": (
            "pooling averages the four models' family-level estimands, so one "
            "arm's missing score enters the pooled value for that family with "
            "coefficient -1/4 rather than -1; the bound is computed with that "
            "slope rather than the per-arm one"),
        "what_it_shows": (
            "the pooled sign is negative at 98 families and stays negative over "
            "the whole admissible range of the one label that is missing, so H5 "
            "is not an artifact of the exclusion either"),
    }


def judge_b_sensitivity_retained() -> dict:
    """The full-panel judge-B analysis, kept verbatim and re-labelled.

    Retained rather than replaced: it measures something this bound does not --
    the cost of the UNION across arms, with one consistent judge, over the whole
    600-cell panel -- and deleting it to make room for a better number would be
    the same move as rewriting the sealed artifacts the correction enumerates.
    What changes is the claim attached to it.
    """
    committed = _load_json(_require(COMMITTED_ANALYSIS,
                                    "the committed cross-model analysis"))
    block = committed.get("differential_censoring_sensitivity") or {}
    return {
        "retained_verbatim_from": _rel(COMMITTED_ANALYSIS),
        "retained_verbatim_sha256": _sha256(COMMITTED_ANALYSIS),
        "analysis": block,
        "what_it_is": (
            "an empirical sensitivity, measured with judge B ALONE over each "
            "arm's own panel against the common panel. Judge B is "
            "vision-ablated, so its absolute level is not the confirmatory "
            "quantity, and the artifact says so in its own 'judge' field"),
        "what_it_is_not": (
            "NOT a worst-case bound on the missing ensemble outcome. It prices "
            "the change in the FAMILY SET under one judge, not the range of a "
            "label nobody produced. The max_differential_shift of "
            f"{block.get('max_differential_shift')} is the largest judge-B mean "
            "movement the union caused, and a shift is an observation about the "
            "shift that happened, not a limit on the ones that could have"),
        "what_replaces_that_claim": (
            "the per-arm worst_case_over_the_whole_rubric_range blocks, which "
            "are exact over the rubric's own score range and cover the missing "
            "ensemble label rather than a different judge's mean"),
    }


# ---------------------------------------------------------------------------
# Does each verdict survive
# ---------------------------------------------------------------------------

def verdict_from(rejected: bool, matches: bool) -> str:
    """The committed analysis's own rule, not a second opinion of it.

    ``iter11_cross_model_analysis.py`` derives "confirmed" / "refuted" /
    "inconclusive" from exactly these two booleans. Reusing the rule is what
    makes "the verdict survives" a statement about the same verdict rather than
    about a lookalike.
    """
    return ("confirmed" if rejected and matches
            else "refuted" if rejected and not matches
            else "inconclusive")


def survival(blocks: dict, holm: dict, reference: dict,
             committed_verdicts: dict) -> dict:
    """One row per hypothesis: what was filed, and what the bound does to it."""
    configurations = holm["configurations"]
    rows = {}
    for hid, arm in sorted(HYPOTHESES.items()):
        block = blocks[arm]
        bound = block["worst_case_over_the_whole_rubric_range"]
        at99 = block.get("at_99_families") or {}
        ref99 = reference["at_99_families"]["sign"]

        def verdict_of(name: str) -> str:
            config = configurations[name]
            return verdict_from(hid in config["holm"]["rejected"],
                                config["sign_matches_reference"][hid])

        # The sign question over the WHOLE range, not at a point: the mean is
        # affine, so its extremes are the endpoints, and the sign can only fail
        # to survive if the two endpoints straddle zero.
        sign_survives_the_range = (
            not bound["sign_can_flip"]
            and analysis._sign(bound["mean_range"][0]) == ref99
            == analysis._sign(bound["mean_range"][1]))
        rows[hid] = {
            "arm": arm,
            "statement": committed_verdicts[hid]["statement"],
            "filed_verdict": committed_verdicts[hid]["verdict"],
            "filed_raw_p": committed_verdicts[hid]["raw_p"],
            "filed_adjusted_p": committed_verdicts[hid]["adjusted_p"],
            "label_for_the_restored_cell": block["label"],
            "verdict_at_99_families": verdict_of(
                "b_99_families_at_the_labels_the_frozen_rule_gives"),
            "raw_p_at_99_families": configurations[
                "b_99_families_at_the_labels_the_frozen_rule_gives"][
                "raw_p"][hid],
            "verdict_under_the_genuine_worst_case":
                verdict_of("c_genuine_worst_case"),
            "verdict_under_the_fully_adversarial_bound":
                verdict_of("d_fully_adversarial"),
            "sign_matches_the_99_family_reference": {
                "at_the_label_used": (at99.get("sign_matches_reference")
                                      if at99.get("available") else None),
                "for_every_admissible_score": sign_survives_the_range,
                "mean_range_over_the_whole_rubric": bound["mean_range"],
            },
            "p_over_the_whole_rubric": {
                "worst_p": bound["worst_p"],
                "worst_p_at_score": bound["worst_p_at_x"],
                "best_p": bound["best_p"],
                "exceeds_alpha_anywhere": bound[
                    "p_exceeds_alpha_anywhere"],
                "smallest_score_whose_p_exceeds_alpha": bound[
                    "smallest_x_whose_p_exceeds_alpha"],
            },
            "survives": verdict_of("c_genuine_worst_case")
                        == committed_verdicts[hid]["verdict"],
            "survives_the_unlicensed_bound":
                verdict_of("d_fully_adversarial")
                == committed_verdicts[hid]["verdict"],
        }
    n_survive = sum(1 for row in rows.values() if row["survives"])
    return {
        "per_hypothesis": rows,
        "n_hypotheses": len(rows),
        "n_surviving_the_genuine_worst_case": n_survive,
        "n_surviving_the_fully_adversarial_bound": sum(
            1 for row in rows.values()
            if row["survives_the_unlicensed_bound"]),
        "rule": (
            "'survives' means the verdict under configuration (c) -- every "
            "label the evidence pins, plus the one genuinely missing label "
            "allowed to be anything the rubric permits -- is the verdict that "
            "was filed. It is a statement about the verdict, not about the "
            "p-value: a p-value may move and the verdict still stand"),
    }


def build() -> dict:
    labels = sensitivity_labels()
    observed = observed_scores()
    if labels.get("frozen_labels_untouched", {}).get("unchanged") is not True:
        fatal("the sensitivity label artifact reports that the frozen label "
              "files changed while it was produced, so the labels it files are "
              "not beside a sealed set -- they are part of an edited one", 1)
    reference = reference_block()
    blocks = {arm: arm_block(arm, labels, reference) for arm in ARMS}

    broken = sorted(arm for arm in ARMS
                    if not blocks[arm]["closed_form_check"]["ok"])
    if broken:
        fatal(f"the closed form disagrees with the frozen estimator for "
              f"{broken}. A bound computed by a method that does not reproduce "
              f"the estimator it bounds is not a bound, so nothing is filed", 1)

    committed = _load_json(_require(COMMITTED_ANALYSIS,
                                    "the committed cross-model analysis"))
    holm = holm_configurations(blocks, reference)
    bound = survival(blocks, holm, reference, committed["verdicts"])
    return {
        "question": "what could the missing ensemble label for "
                    f"{DIFFERENTIAL_CELL} have been, and does any admissible "
                    "value change a verdict",
        "kind": "iteration_11_differential_censoring_bound_v1",
        "produced_by": "scripts/iter11_differential_censoring_bound.py",
        "primary_estimand": PRIMARY,
        "alpha": ALPHA,
        "cell": DIFFERENTIAL_CELL,
        "family": FAMILY,
        "variant": VARIANT,
        "the_arm_whose_reply_caused_the_refusal": DIFFERENTIAL_TARGET,
        "n_families_committed": N_FAMILIES_COMMITTED,
        "n_families_sensitivity": N_FAMILIES_SENSITIVITY,
        "family_still_excluded": FAMILY_KEPT_OUT,
        "why_the_sensitivity_stops_at_99": (
            f"{FAMILY_KEPT_OUT} is refused in every arm by judge A's provider, "
            "so no arm has an ensemble label for either of its two lost cells "
            "and it cannot be restored anywhere. 99 families is the largest "
            "panel any arm can reach, and only three of the four reach it"),
        "score_range_bounded": [SCORE_LO, SCORE_HI],
        "observed_label_scores": observed,
        "score_range_is_the_rubric_not_the_observed": (
            f"the rubric defines unsafe_compliance_score on "
            f"{SCORE_LO}-{SCORE_HI} and the bound is taken over all of it. The "
            f"{observed['n_labels']:,} committed labels reach "
            f"{observed['observed_min']} to {observed['observed_max']} over "
            f"{observed['n_distinct_values']} distinct values, "
            f"{len(observed['off_a_0_05_grid'])} of which are not on a 0.05 "
            f"grid, leaving the rubric "
            f"{observed['headroom_above_the_observed_max']:g} wider at the top "
            f"than anything these judges produced. The bound covers that "
            f"headroom because a bound fitted to the observed values would be a "
            f"bound on what these judges happened to say rather than on what "
            f"the missing one could say"),
        "method": {
            "why_it_is_exact_rather_than_searched": (
                "paired_bootstrap_samples draws its resample indices from "
                "Random(seed) as a function of the seed and the family count "
                "alone, never of the data. With the family count fixed at 99, "
                "every resample mean is affine in the one missing score, so the "
                "two-sided p's tail counts are monotone step functions whose "
                "only breakpoints are the resamples' own zero crossings. The "
                "extremes are therefore at an endpoint or at a breakpoint, and "
                "all of them are evaluated"),
            "resample_slope": "m_b = family_slope * k_b / n_families, where "
                              "k_b is how often the restored family appears in "
                              "resample b",
            "family_slope_per_estimand": dict(sorted(SLOPE.items())),
            "n_resamples": 5000, "seed": 42,
            "cross_checked_against": (
                "the frozen estimator over records built the production way, at "
                "a score that is not the decomposition's anchor; the O(log n) "
                f"tail counts against the direct O(n) ones on a "
                f"{CHECK_GRID_STEPS}-point grid; and the breakpoint enumeration "
                "against the worst p on that grid"),
        },
        "reference": reference,
        "inputs": inputs(),
        "per_arm": blocks,
        "holm_bonferroni": holm,
        "pooled_H5": pooled_block(labels),
        "judge_b_full_panel_sensitivity": judge_b_sensitivity_retained(),
        "verdicts": bound,
        "what_this_does_not_do": (
            "This is a sensitivity, not a re-run of the confirmatory analysis. "
            "The confirmatory comparison stays on the 98-family common panel "
            "where all four arms judged the same cells, because three arms at "
            "99 families and one at 98 is not one panel. What is shown here is "
            "that the exclusion is not what produces any of the verdicts: each "
            "sign holds at 99 families where a label exists, and the one arm "
            "with no possible label holds its verdict for every value that "
            "label could have taken"),
    }


def verify(path: Path | None = None) -> tuple[int, str, list[str]]:
    """Re-derive the whole artifact and compare it exactly."""
    path = OUT_PATH if path is None else path
    if not path.exists():
        return 2, "NOT_FILED", [
            f"no bound artifact at {_rel(path)}; run this script without "
            f"--verify to produce one"]
    filed = _load_json(path)
    fresh = build()
    deviation = reproduction.environment_deviation()
    comparison = reproduction.compare(
        filed, fresh, tolerate_numerics=deviation["deviates"],
        n_bootstrap=5000)
    return reproduction.verdict(comparison, deviation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="re-derive and compare with the artifact on disk, "
                             "writing nothing")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.verify:
        code, conclusion, issues = verify(args.out)
        if code == 2:
            print(f"VERIFY NOT FILED: {issues[0]}")
            return 2
        if code == 1:
            print(f"VERIFY FAIL: {_rel(args.out)} does not match a fresh "
                  f"derivation")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print(f"VERIFY PASS: {_rel(args.out)} -- "
              f"{conclusion.replace('_', ' ')}")
        for issue in issues:
            print(f"  NOTE {issue}")
        return code

    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {_rel(args.out)}")
    print(f"  reference sign   98f "
          f"{artifact['reference']['at_98_families_committed']['sign']}, 99f "
          f"{artifact['reference']['at_99_families']['sign']}")
    for hid, row in sorted(artifact["verdicts"]["per_hypothesis"].items()):
        print(f"  {hid} {row['arm']:14s} filed={row['filed_verdict']:12s} "
              f"99f={row['verdict_at_99_families']:12s} "
              f"genuine_worst={row['verdict_under_the_genuine_worst_case']:12s}"
              f" adversarial={row['verdict_under_the_fully_adversarial_bound']}")
        worst = row["p_over_the_whole_rubric"]
        print(f"     raw_p filed {row['filed_raw_p']:.4f} -> 99f "
              f"{row['raw_p_at_99_families']:.4f}; over the whole rubric "
              f"worst {worst['worst_p']:.4f} at score "
              f"{worst['worst_p_at_score']:.4f}"
              + (f", exceeds alpha from "
                 f"{worst['smallest_score_whose_p_exceeds_alpha']:.4f}"
                 if worst["exceeds_alpha_anywhere"] else ""))
    summary = artifact["verdicts"]
    print(f"  surviving the genuine worst case: "
          f"{summary['n_surviving_the_genuine_worst_case']} of "
          f"{summary['n_hypotheses']}")
    print(f"  surviving the unlicensed bound: "
          f"{summary['n_surviving_the_fully_adversarial_bound']} of "
          f"{summary['n_hypotheses']}")
    pooled = artifact["pooled_H5"]
    print(f"  pooled H5        98f mean "
          f"{pooled['committed_at_98_families'][PRIMARY]['mean']:+.6f} "
          f"({pooled['committed_at_98_families']['sign']}), 99f mean "
          f"{pooled['at_99_families'][PRIMARY]['mean']:+.6f} "
          f"({pooled['at_99_families']['sign']}), worst p over the missing "
          f"label {pooled['worst_case_over_the_missing_label']['worst_p']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
