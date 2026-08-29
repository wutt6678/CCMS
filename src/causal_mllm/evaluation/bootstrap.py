"""Paired family-level bootstrap CIs (Iteration 9).

Family-level resampling: resample families WITH replacement, recompute
means for each estimand, and report the empirical percentile CI.

This is the PRIMARY inference method for the causal estimands.  The
unit of resampling is the FAMILY (not the response), preserving the
within-family factorial structure.
"""

from __future__ import annotations

import numpy as np

from causal_mllm.evaluation.errors import EvaluationError


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

    # Build matrix: rows = families, cols = estimands
    data = np.array([
        [family_estimands[fid][name] for name in estimand_names]
        for fid in family_ids
    ])

    rng = np.random.RandomState(seed)
    bootstrap_samples = np.zeros((n_bootstrap, len(estimand_names)))

    for b in range(n_bootstrap):
        indices = rng.randint(0, n_families, size=n_families)
        resampled = data[indices]
        bootstrap_samples[b] = resampled.mean(axis=0)

    # Percentile CI
    alpha = 1.0 - ci_level
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    result: dict[str, dict] = {}
    for i, name in enumerate(estimand_names):
        samples = bootstrap_samples[:, i]
        result[name] = {
            "mean": float(np.mean(samples)),
            "CI_lower": float(np.percentile(samples, lower_pct)),
            "CI_upper": float(np.percentile(samples, upper_pct)),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }

    return result
