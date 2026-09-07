"""Paired family-level bootstrap CIs (Iteration 9).

Family-level resampling: resample families WITH replacement, recompute
means for each estimand, and report the empirical percentile CI.

This is the PRIMARY inference method for the causal estimands.  The
unit of resampling is the FAMILY (not the response), preserving the
within-family factorial structure.

Pure Python implementation (no numpy dependency).
"""

from __future__ import annotations

import random

from causal_mllm.evaluation.errors import EvaluationError


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute the percentile of a sorted list using linear interpolation.

    Args:
        sorted_vals: Sorted list of numeric values.
        pct: Percentile in [0, 100].

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]

    # Linear interpolation method
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


ESTIMAND_NAMES = ("Delta_T", "Delta_V", "Delta_TV",
                  "order_effect", "history_effect")


def paired_bootstrap_samples(
    family_estimands: dict[str, dict],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """The resample distribution behind :func:`paired_bootstrap_ci`.

    Same loop, same ``rng`` consumption order, same seed -- the percentile CI
    is a summary of exactly these values, so a p-value computed from them and
    a CI computed by :func:`paired_bootstrap_ci` describe one distribution and
    cannot disagree about whether it straddles zero.

    Exposed because Iteration 11.8 pre-declares Holm-Bonferroni over four
    confirmatory model tests, and a family-wise correction needs a p-value per
    test while Iteration 10 published only means and intervals.

    Returns:
        ``{estimand_name: [resample mean, ...]}`` in ``ESTIMAND_NAMES`` order,
        each list of length ``n_bootstrap``.
    """
    if not family_estimands:
        raise EvaluationError("no family estimands to bootstrap")

    family_ids = sorted(family_estimands.keys())
    n_families = len(family_ids)

    # Build matrix: rows = families, cols = estimands
    data = [
        [family_estimands[fid][name] for name in ESTIMAND_NAMES]
        for fid in family_ids
    ]

    rng = random.Random(seed)
    samples: list[list[float]] = [[] for _ in range(len(ESTIMAND_NAMES))]
    for _b in range(n_bootstrap):
        indices = [rng.randint(0, n_families - 1) for _ in range(n_families)]
        for e_idx in range(len(ESTIMAND_NAMES)):
            vals = [data[i][e_idx] for i in indices]
            samples[e_idx].append(sum(vals) / len(vals))
    return {name: samples[e_idx] for e_idx, name in enumerate(ESTIMAND_NAMES)}


def bootstrap_two_sided_p(samples: list[float], null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for ``H0: estimand == null``.

    ``2 * min(P(resample <= null), P(resample >= null))``, capped at 1 and
    floored at ``1 / len(samples)``: a resample distribution cannot resolve a
    smaller tail than one draw out of the number actually taken, and reporting
    an exact zero would claim a certainty 5000 draws do not have. The floor is
    what makes the p-value and the percentile CI agree at the boundary rather
    than one saying "0.0" and the other saying "excludes zero". The cap is
    needed because both tails include the resamples exactly AT the null, so
    their sum can exceed 1.
    """
    n = len(samples)
    if n == 0:
        raise EvaluationError("no bootstrap samples to test")
    below = sum(1 for s in samples if s <= null)
    above = sum(1 for s in samples if s >= null)
    return min(1.0, max(2.0 * min(below, above) / n, 1.0 / n))


def paired_bootstrap_ci(
    family_estimands: dict[str, dict],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
    with_samples: bool = False,
) -> dict:
    """Compute paired bootstrap CIs for each estimand.

    Args:
        family_estimands: Dict keyed by family_id, each value is a dict
            with Delta_T, Delta_V, Delta_TV, order_effect, history_effect.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        with_samples: Also return the resample distribution each interval
            summarises. Opt-in and additive: Iteration 11.8 needs a p-value
            per confirmatory test and re-drawing the resamples to get one
            would run the loop twice for the same numbers. The default output
            is unchanged, key for key, because the sealed Iteration 9 and 10
            reports carry it.

    Returns:
        Dict with mean, CI_lower, CI_upper for each estimand.
    """
    bootstrap = paired_bootstrap_samples(
        family_estimands, n_bootstrap=n_bootstrap, seed=seed)

    # Percentile CI
    alpha = 1.0 - ci_level
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    result: dict[str, dict] = {}
    for name in ESTIMAND_NAMES:
        samples = sorted(bootstrap[name])
        result[name] = {
            "mean": sum(samples) / len(samples),
            "CI_lower": _percentile(samples, lower_pct),
            "CI_upper": _percentile(samples, upper_pct),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }
        if with_samples:
            result[name]["bootstrap_samples"] = samples

    return result
