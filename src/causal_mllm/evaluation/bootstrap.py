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


def paired_bootstrap_ci(
    family_estimands: dict[str, dict],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute paired bootstrap CIs for each estimand.

    Args:
        family_estimands: Dict keyed by family_id, each value is a dict
            with Delta_T, Delta_V, Delta_TV, order_effect, history_effect.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean, CI_lower, CI_upper for each estimand.
    """
    if not family_estimands:
        raise EvaluationError("no family estimands to bootstrap")

    estimand_names = ("Delta_T", "Delta_V", "Delta_TV",
                      "order_effect", "history_effect")
    family_ids = sorted(family_estimands.keys())
    n_families = len(family_ids)
    n_estimands = len(estimand_names)

    # Build matrix: rows = families, cols = estimands
    data = [
        [family_estimands[fid][name] for name in estimand_names]
        for fid in family_ids
    ]

    rng = random.Random(seed)
    # Store bootstrap means per estimand
    bootstrap_samples: list[list[float]] = [[] for _ in range(n_estimands)]

    for _b in range(n_bootstrap):
        indices = [rng.randint(0, n_families - 1) for _ in range(n_families)]
        for e_idx in range(n_estimands):
            vals = [data[i][e_idx] for i in indices]
            bootstrap_samples[e_idx].append(sum(vals) / len(vals))

    # Percentile CI
    alpha = 1.0 - ci_level
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    result: dict[str, dict] = {}
    for e_idx, name in enumerate(estimand_names):
        samples = sorted(bootstrap_samples[e_idx])
        result[name] = {
            "mean": sum(samples) / len(samples),
            "CI_lower": _percentile(samples, lower_pct),
            "CI_upper": _percentile(samples, upper_pct),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }

    return result
