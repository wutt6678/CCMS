"""The confirmatory tests Iteration 11.8 pre-declared.

The frozen protocol (``outputs/iteration_11/protocol/iteration_11_protocol.json``)
fixes the correction and the estimand::

    "multiplicity": {
        "primary_estimand_per_new_model": "Delta_TV sign (frozen estimator)",
        "family_wise_correction": "Holm-Bonferroni",
        "alpha": 0.05,
        "n_confirmatory_model_tests": 4,
        "raw_ci_preserved": true
    }

and H1-H4 each state that one new model's ``Delta_TV`` SIGN matches the
Iteration 10 reference sign. What it does not fix is a p-value, because none
exists in this repository: Iteration 10 published means and percentile
intervals, and a family-wise correction needs one number per test.

Two are computed here, deliberately, with different jobs:

* the CONFIRMATORY one is the two-sided bootstrap p-value from the same seed-42
  resamples that produce the frozen percentile CI
  (:func:`causal_mllm.evaluation.bootstrap.bootstrap_two_sided_p`). It tests
  the same quantity the CI describes, so the sign statement and the interval
  cannot disagree -- and the protocol characterises the reference sign BY its
  interval, ``CI [0.0495, 0.1800]``, so the new models have to be tested on the
  same footing.
* the SENSITIVITY one is an exact family-level sign test
  (:func:`family_sign_test`), which is what "Delta_TV sign" reads like taken
  literally and answers a different question: whether MORE THAN HALF the
  families move in the reference direction. A model can have a positive mean
  with a minority of families positive, and if that happens here the two
  numbers are reported side by side rather than one being quietly preferred.

:func:`holm_bonferroni` applies the pre-declared correction to the four
confirmatory p-values. Nothing here selects which models or metrics are
reported: the protocol's retention clause requires every result, favourable or
not, and a correction that is applied to a subset is not a correction.
"""

from __future__ import annotations

import math

from causal_mllm.evaluation.errors import EvaluationError


def _binomial_tail(n: int, k: int) -> tuple[float, float]:
    """``(P(X <= k), P(X >= k))`` for ``X ~ Binomial(n, 0.5)``."""
    total = 0.0
    lower = 0.0
    for i in range(n + 1):
        term = math.comb(n, i) * 0.5 ** n
        total += term
        if i <= k:
            lower += term
    return lower, total - lower + math.comb(n, k) * 0.5 ** n


def family_sign_test(values: dict[str, float], null: float = 0.0) -> dict:
    """Exact two-sided binomial sign test over family-level estimands.

    ``values`` maps family_id to that family's estimand. Families exactly at
    ``null`` are dropped, as a sign test must: they carry no direction, and
    counting them as negative would turn a tie into evidence. How many were
    dropped is reported, because a panel where that number is large is not
    testing what a panel where it is zero tests.

    Distribution-free and independent of the bootstrap: it uses the COUNT of
    families on each side of zero and nothing about their magnitudes, so it can
    disagree with a mean-based test precisely when a few large families carry
    the average. That disagreement is the reason to report both.
    """
    if not values:
        raise EvaluationError("no family-level values to sign-test")
    positive = sorted(fid for fid, v in values.items() if v > null)
    negative = sorted(fid for fid, v in values.items() if v < null)
    ties = sorted(fid for fid, v in values.items() if v == null)
    n = len(positive) + len(negative)
    result = {
        "test": "exact two-sided binomial sign test, H0: P(> null) = 0.5",
        "null": null,
        "n_families": len(values),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "n_ties_dropped": len(ties),
        "tie_families": ties,
        "n_tested": n,
        "majority_sign": ("positive" if len(positive) > len(negative)
                          else "negative" if len(negative) > len(positive)
                          else "tied"),
    }
    if n == 0:
        result["p_value"] = 1.0
        result["note"] = "every family sits exactly at the null; there is no " \
                         "direction to test"
        return result
    k = len(positive)
    lower, upper = _binomial_tail(n, k)
    result["p_value"] = min(1.0, 2.0 * min(lower, upper))
    return result


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over a named set of p-values.

    Step-down: order ascending, compare the i-th smallest against
    ``alpha / (m - i + 1)``, and STOP at the first comparison that fails --
    every later hypothesis is retained with it, which is what makes the
    procedure control the family-wise rate rather than the per-test rate.
    Adjusted p-values are the running maximum of ``(m - i + 1) * p`` capped at
    1, so they are monotone in the ordering and ``adjusted <= alpha`` picks out
    exactly the rejected set.

    Ties are broken by name, so the result does not depend on dict ordering and
    a ``--verify`` can reproduce it byte for byte.
    """
    if not p_values:
        raise EvaluationError("no p-values to correct")
    if not 0.0 < alpha < 1.0:
        raise EvaluationError(f"alpha must be in (0, 1), got {alpha}")
    for name, p in p_values.items():
        if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise EvaluationError(
                f"p-value for {name} must be a number in [0, 1], got {p!r}")

    m = len(p_values)
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered, start=1):
        running = max(running, (m - rank + 1) * p_values[name])
        adjusted[name] = min(running, 1.0)

    n_rejected = 0
    for rank, name in enumerate(ordered, start=1):
        if (m - rank + 1) * p_values[name] <= alpha:
            n_rejected = rank
        else:
            break
    rejected = ordered[:n_rejected]

    return {
        "method": "Holm-Bonferroni",
        "alpha": alpha,
        "n_tests": m,
        "order": ordered,
        "raw_p": {name: p_values[name] for name in ordered},
        "adjusted_p": {name: adjusted[name] for name in ordered},
        "critical_values": {
            name: alpha / (m - rank + 1)
            for rank, name in enumerate(ordered, start=1)},
        "rejected": sorted(rejected),
        "retained": sorted(name for name in ordered if name not in rejected),
        "n_rejected": n_rejected,
        "rule": (
            "step-down over the ascending raw p-values against "
            "alpha / (m - i + 1), stopping at the first failure; adjusted "
            "p-values are the running maximum of (m - i + 1) * p capped at 1"),
    }
