"""Inter-judge agreement metrics for LLM judges.

Computes agreement between multiple LLM judges:
- Cohen's κ for refusal type (categorical)
- Weighted κ for compliance level (ordinal)
- MAE for unsafe compliance score (continuous)
- ICC or rank correlation for scores (continuous)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from causal_mllm.evaluation.errors import EvaluationError


def _cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's κ for two label sequences.

    Args:
        labels_a: First annotator's labels.
        labels_b: Second annotator's labels.

    Returns:
        Cohen's κ coefficient.
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))

    # Observed agreement
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p_a = sum(1 for x in labels_a if x == cat) / n
        p_b = sum(1 for x in labels_b if x == cat) / n
        pe += p_a * p_b

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1.0 - pe)


def _weighted_kappa(labels_a: list, labels_b: list, weights: str = "quadratic") -> float:
    """Compute weighted Cohen's κ for ordinal labels.

    Args:
        labels_a: First annotator's labels.
        labels_b: Second annotator's labels.
        weights: Weight type ("quadratic" or "linear").

    Returns:
        Weighted κ coefficient.
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    k = len(categories)

    # Build confusion matrix
    obs = np.zeros((k, k))
    for a, b in zip(labels_a, labels_b):
        i = categories.index(a)
        j = categories.index(b)
        obs[i, j] += 1

    # Normalize to proportions
    obs = obs / n

    # Compute expected matrix
    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    exp = np.outer(row_sums, col_sums)

    # Build weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "quadratic":
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
            else:  # linear
                w[i, j] = abs(i - j) / (k - 1)

    # Compute weighted κ
    numerator = (w * obs).sum()
    denominator = (w * exp).sum()

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return 1.0 - numerator / denominator


def _icc(scores_list: list[list[float]], icc_type: str = "ICC3k") -> float:
    """Compute Intraclass Correlation Coefficient.

    Args:
        scores_list: List of score arrays, one per judge.
        icc_type: ICC type ("ICC3k" for consistency, "ICC2k" for agreement).

    Returns:
        ICC value.
    """
    # Convert to numpy array (subjects × raters)
    data = np.array(scores_list).T  # Shape: (n_subjects, n_raters)
    n_subjects, n_raters = data.shape

    if n_subjects < 2 or n_raters < 2:
        return 0.0

    # Compute means
    subject_means = data.mean(axis=1)
    grand_mean = data.mean()

    # Between-subjects variance
    ss_between = n_raters * ((subject_means - grand_mean) ** 2).sum()
    df_between = n_subjects - 1
    ms_between = ss_between / df_between if df_between > 0 else 0

    # Within-subjects variance
    ss_within = ((data - subject_means[:, None]) ** 2).sum()
    df_within = n_subjects * (n_raters - 1)
    ms_within = ss_within / df_within if df_within > 0 else 0

    # ICC(3,k) for consistency
    if icc_type == "ICC3k":
        if ms_within == 0:
            return 1.0
        return (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)

    # ICC(2,k) for agreement (simplified)
    return (ms_between - ms_within) / ms_between if ms_between > 0 else 0.0


def compute_judge_agreement(
    judgments_a: list[dict],
    judgments_b: list[dict],
    judgments_c: list[dict],
) -> dict:
    """Compute inter-judge agreement metrics.

    Args:
        judgments_a, judgments_b, judgments_c: Lists of judgment records
            from three judges. Each record has "item_id" and "judgment" keys.

    Returns:
        Dict with agreement metrics:
        - kappa_refusal: Cohen's κ for refusal_type (A-B, A-C, B-C, mean)
        - kappa_compliance_weighted: Weighted κ for compliance_level
        - mae_score: Mean absolute error for unsafe_compliance_score
        - icc_score: ICC for unsafe_compliance_score
        - spearman_rho: Spearman rank correlation for scores
    """
    # Build lookups by item_id
    lookup_a = {j["item_id"]: j["judgment"] for j in judgments_a}
    lookup_b = {j["item_id"]: j["judgment"] for j in judgments_b}
    lookup_c = {j["item_id"]: j["judgment"] for j in judgments_c}

    # Align on common item_ids
    common_ids = sorted(set(lookup_a) & set(lookup_b) & set(lookup_c))

    if len(common_ids) != 120:
        raise EvaluationError(
            f"judge agreement requires exactly 120 common items, "
            f"got {len(common_ids)}")

    # Extract labels
    refusal_a = [lookup_a[i]["refusal_type"] for i in common_ids]
    refusal_b = [lookup_b[i]["refusal_type"] for i in common_ids]
    refusal_c = [lookup_c[i]["refusal_type"] for i in common_ids]

    compliance_a = [lookup_a[i]["compliance_level"] for i in common_ids]
    compliance_b = [lookup_b[i]["compliance_level"] for i in common_ids]
    compliance_c = [lookup_c[i]["compliance_level"] for i in common_ids]

    scores_a = [lookup_a[i]["unsafe_compliance_score"] for i in common_ids]
    scores_b = [lookup_b[i]["unsafe_compliance_score"] for i in common_ids]
    scores_c = [lookup_c[i]["unsafe_compliance_score"] for i in common_ids]

    # Cohen's κ for refusal_type (pairwise)
    kappa_ab = _cohens_kappa(refusal_a, refusal_b)
    kappa_ac = _cohens_kappa(refusal_a, refusal_c)
    kappa_bc = _cohens_kappa(refusal_b, refusal_c)
    kappa_refusal_mean = (kappa_ab + kappa_ac + kappa_bc) / 3

    # Weighted κ for compliance_level (pairwise)
    wkappa_ab = _weighted_kappa(compliance_a, compliance_b)
    wkappa_ac = _weighted_kappa(compliance_a, compliance_c)
    wkappa_bc = _weighted_kappa(compliance_b, compliance_c)
    wkappa_compliance_mean = (wkappa_ab + wkappa_ac + wkappa_bc) / 3

    # MAE for scores (pairwise)
    mae_ab = np.mean([abs(a - b) for a, b in zip(scores_a, scores_b)])
    mae_ac = np.mean([abs(a - c) for a, c in zip(scores_a, scores_c)])
    mae_bc = np.mean([abs(b - c) for b, c in zip(scores_b, scores_c)])
    mae_mean = (mae_ab + mae_ac + mae_bc) / 3

    # ICC for scores
    icc = _icc([scores_a, scores_b, scores_c])

    # Spearman rank correlation (mean of pairwise)
    rho_ab, _ = stats.spearmanr(scores_a, scores_b)
    rho_ac, _ = stats.spearmanr(scores_a, scores_c)
    rho_bc, _ = stats.spearmanr(scores_b, scores_c)
    spearman_mean = (rho_ab + rho_ac + rho_bc) / 3

    return {
        "kappa_refusal": {
            "A_B": kappa_ab,
            "A_C": kappa_ac,
            "B_C": kappa_bc,
            "mean": kappa_refusal_mean,
        },
        "kappa_compliance_weighted": {
            "A_B": wkappa_ab,
            "A_C": wkappa_ac,
            "B_C": wkappa_bc,
            "mean": wkappa_compliance_mean,
        },
        "mae_score": {
            "A_B": float(mae_ab),
            "A_C": float(mae_ac),
            "B_C": float(mae_bc),
            "mean": float(mae_mean),
        },
        "icc_score": float(icc),
        "spearman_rho": {
            "A_B": float(rho_ab),
            "A_C": float(rho_ac),
            "B_C": float(rho_bc),
            "mean": float(spearman_mean),
        },
        "n_items": len(common_ids),
    }
