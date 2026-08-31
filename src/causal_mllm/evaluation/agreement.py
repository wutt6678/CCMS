"""Inter-judge agreement metrics for LLM judges.

Computes agreement between multiple LLM judges:
- Cohen's κ for refusal type (categorical)
- Weighted κ for compliance level (ordinal)
- MAE for unsafe compliance score (continuous)
- ICC or rank correlation for scores (continuous)

Pure Python implementation (no numpy/scipy dependency).
"""

from __future__ import annotations

import math

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

    # Build confusion matrix (list of lists)
    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        i = categories.index(a)
        j = categories.index(b)
        obs[i][j] += 1

    # Normalize to proportions
    obs = [[obs[i][j] / n for j in range(k)] for i in range(k)]

    # Compute expected matrix
    row_sums = [sum(obs[i]) for i in range(k)]
    col_sums = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    exp = [[row_sums[i] * col_sums[j] for j in range(k)] for i in range(k)]

    # Build weight matrix
    w = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if weights == "quadratic":
                w[i][j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
            else:  # linear
                w[i][j] = abs(i - j) / (k - 1) if k > 1 else 0.0

    # Compute weighted κ
    numerator = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    denominator = sum(w[i][j] * exp[i][j] for i in range(k) for j in range(k))

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return 1.0 - numerator / denominator


def _mean(vals: list[float]) -> float:
    """Compute mean of a list of values."""
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _icc(scores_list: list[list[float]], icc_type: str = "ICC(3,k)") -> dict:
    """Compute consistency Intraclass Correlation via two-way ANOVA.

    Implements Shrout & Fleiss (1979) Case 3 (consistency), forms (3,1)
    and (3,k). This is a two-way mixed-effects model where raters are
    fixed. The residual error term removes BOTH subject (row) and rater
    (column) effects, so additive per-rater offsets do not reduce the
    consistency ICC.

    Args:
        scores_list: List of score arrays, one per judge (k raters).
        icc_type: Which form to mark as "requested".

    Returns:
        Dict with "ICC(3,1)", "ICC(3,k)", and "requested". Values are
        NOT clamped; negative ICCs are reported as-is (they indicate
        agreement worse than chance).

    Decomposition (n subjects, k raters, X[i][j] = subject i, rater j):
        SS_rows  = k * sum_i (row_mean_i  - grand_mean)^2   df = n-1
        SS_cols  = n * sum_j (col_mean_j  - grand_mean)^2   df = k-1
        SS_error = SS_total - SS_rows - SS_cols             df = (n-1)(k-1)

        ICC(3,1) = (MS_rows - MS_error) / (MS_rows + (k-1)*MS_error)
        ICC(3,k) = (MS_rows - MS_error) / MS_rows
    """
    n_raters = len(scores_list)
    n_subjects = len(scores_list[0])

    if n_subjects < 2 or n_raters < 2:
        return {"ICC(3,1)": 0.0, "ICC(3,k)": 0.0, "requested": 0.0}

    # data[subject][rater]
    data = [[scores_list[r][s] for r in range(n_raters)]
            for s in range(n_subjects)]

    grand_mean = _mean([data[s][r]
                        for s in range(n_subjects)
                        for r in range(n_raters)])
    row_means = [_mean(data[s]) for s in range(n_subjects)]
    col_means = [_mean([data[s][r] for s in range(n_subjects)])
                 for r in range(n_raters)]

    # Sum of squares
    ss_total = sum((data[s][r] - grand_mean) ** 2
                   for s in range(n_subjects)
                   for r in range(n_raters))
    ss_rows = n_raters * sum((row_means[s] - grand_mean) ** 2
                             for s in range(n_subjects))
    ss_cols = n_subjects * sum((col_means[r] - grand_mean) ** 2
                               for r in range(n_raters))
    ss_error = ss_total - ss_rows - ss_cols
    # Guard tiny negative from floating point
    if ss_error < 0 and ss_error > -1e-9:
        ss_error = 0.0

    # Degrees of freedom
    df_rows = n_subjects - 1
    df_error = (n_subjects - 1) * (n_raters - 1)

    ms_rows = ss_rows / df_rows if df_rows > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # ICC(3,1): single-measure consistency (NOT clamped)
    denom_31 = ms_rows + (n_raters - 1) * ms_error
    icc_3_1 = (ms_rows - ms_error) / denom_31 if denom_31 != 0 else 0.0

    # ICC(3,k): average-measure consistency (NOT clamped)
    icc_3_k = (ms_rows - ms_error) / ms_rows if ms_rows != 0 else 0.0

    requested = icc_3_k if icc_type == "ICC(3,k)" else icc_3_1

    return {
        "ICC(3,1)": icc_3_1,
        "ICC(3,k)": icc_3_k,
        "requested": requested,
    }


def _rank_data(vals: list[float]) -> list[float]:
    """Compute ranks for a list of values (average rank for ties)."""
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all items with the same value
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        # Average rank
        avg_rank = (i + j) / 2.0 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        x, y: Two lists of numeric values.

    Returns:
        Spearman's ρ coefficient.
    """
    n = len(x)
    if n < 2:
        return 0.0

    rx = _rank_data(x)
    ry = _rank_data(y)

    # Pearson correlation on ranks
    mean_rx = _mean(rx)
    mean_ry = _mean(ry)

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def compute_pairwise_agreement(
    judgments_a: list[dict],
    judgments_b: list[dict],
    n_items_expected: int = 120,
) -> dict:
    """Compute agreement between TWO judges (cross-model, A-B).

    Used for the 2-primary-judge architecture where a distinct adjudicator
    reviews disagreements. Reports Cohen's kappa (refusal), weighted kappa
    (compliance), MAE and Spearman (score), and ICC(3,1)/ICC(3,k) with k=2.

    Args:
        judgments_a, judgments_b: Judgment records from the two judges.
        n_items_expected: Expected number of common items.

    Returns:
        Dict with pairwise agreement metrics.
    """
    lookup_a = {j["item_id"]: j["judgment"] for j in judgments_a}
    lookup_b = {j["item_id"]: j["judgment"] for j in judgments_b}
    common_ids = sorted(set(lookup_a) & set(lookup_b))

    if len(common_ids) != n_items_expected:
        raise EvaluationError(
            f"pairwise agreement requires exactly {n_items_expected} "
            f"common items, got {len(common_ids)}")

    refusal_a = [lookup_a[i]["refusal_type"] for i in common_ids]
    refusal_b = [lookup_b[i]["refusal_type"] for i in common_ids]
    compliance_a = [lookup_a[i]["compliance_level"] for i in common_ids]
    compliance_b = [lookup_b[i]["compliance_level"] for i in common_ids]
    scores_a = [lookup_a[i]["unsafe_compliance_score"] for i in common_ids]
    scores_b = [lookup_b[i]["unsafe_compliance_score"] for i in common_ids]

    icc_result = _icc([scores_a, scores_b])

    return {
        "kappa_refusal": _cohens_kappa(refusal_a, refusal_b),
        "kappa_compliance_weighted": _weighted_kappa(compliance_a, compliance_b),
        "mae_score": _mean([abs(a - b) for a, b in zip(scores_a, scores_b)]),
        "icc_score": {
            "ICC(3,1)": icc_result["ICC(3,1)"],
            "ICC(3,k)": icc_result["ICC(3,k)"],
        },
        "spearman_rho": _spearman_rho(scores_a, scores_b),
        "n_items": len(common_ids),
        "note": "Pairwise cross-model agreement between the two primary "
                "judges (A-B).",
    }


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
    mae_ab = _mean([abs(a - b) for a, b in zip(scores_a, scores_b)])
    mae_ac = _mean([abs(a - c) for a, c in zip(scores_a, scores_c)])
    mae_bc = _mean([abs(b - c) for b, c in zip(scores_b, scores_c)])
    mae_mean = (mae_ab + mae_ac + mae_bc) / 3

    # ICC for scores (now returns dict with both ICC(3,1) and ICC(3,k))
    icc_result = _icc([scores_a, scores_b, scores_c])

    # Spearman rank correlation (mean of pairwise)
    rho_ab = _spearman_rho(scores_a, scores_b)
    rho_ac = _spearman_rho(scores_a, scores_c)
    rho_bc = _spearman_rho(scores_b, scores_c)
    spearman_mean = (rho_ab + rho_ac + rho_bc) / 3

    return {
        "kappa_refusal": {
            "A_B": kappa_ab,
            "A_C": kappa_ac,
            "B_C": kappa_bc,
            # Cross-model agreement excludes the within-model A-C pair.
            "cross_model_mean": (kappa_ab + kappa_bc) / 2,
            "within_model_AC": kappa_ac,
            "mean": kappa_refusal_mean,
            "note": "A_C measures within-model repeatability (same model, "
                    "different seeds). A_B and B_C measure cross-model "
                    "agreement. Use cross_model_mean for inter-model "
                    "agreement; do not interpret 'mean' as general "
                    "inter-judge agreement since it includes A_C.",
        },
        "kappa_compliance_weighted": {
            "A_B": wkappa_ab,
            "A_C": wkappa_ac,
            "B_C": wkappa_bc,
            "cross_model_mean": (wkappa_ab + wkappa_bc) / 2,
            "within_model_AC": wkappa_ac,
            "mean": wkappa_compliance_mean,
        },
        "mae_score": {
            "A_B": mae_ab,
            "A_C": mae_ac,
            "B_C": mae_bc,
            "cross_model_mean": (mae_ab + mae_bc) / 2,
            "within_model_AC": mae_ac,
            "mean": mae_mean,
        },
        "icc_score": {
            "ICC(3,1)": icc_result["ICC(3,1)"],
            "ICC(3,k)": icc_result["ICC(3,k)"],
            "note": "Two-way ANOVA consistency ICC (Shrout & Fleiss 1979, "
                    "Case 3). Residual removes both subject and rater "
                    "effects. Values are unclamped; negatives indicate "
                    "worse-than-chance agreement. ICC(3,1) is single-"
                    "measure, ICC(3,k) is average-measure.",
        },
        "spearman_rho": {
            "A_B": rho_ab,
            "A_C": rho_ac,
            "B_C": rho_bc,
            "mean": spearman_mean,
        },
        "n_items": len(common_ids),
    }
