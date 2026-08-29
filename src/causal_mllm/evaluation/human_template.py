"""Human labeling template (Iteration 9).

Generate a JSON workbook for human annotators to label the 120
replay responses.  Variant names are replaced with anonymous labels
(A, B, C, D, E, F) in random order to prevent bias.

The completed workbook is parsed back into judge records for the
causal estimand computation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.schema import validate_judgment


def _build_anonymization_map(seed: int = 42) -> dict[str, str]:
    """Map variant names to anonymous labels A–F in random order."""
    rng = random.Random(seed)
    variants = list(ALL_VARIANT_NAMES)
    rng.shuffle(variants)
    labels = [chr(ord("A") + i) for i in range(len(variants))]
    return dict(zip(variants, labels))


def _reverse_map(anon_map: dict[str, str]) -> dict[str, str]:
    """Invert {variant: label} -> {label: variant}."""
    return {v: k for k, v in anon_map.items()}


def generate_labeling_workbook(
    run_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict:
    """Create a JSON workbook for human labeling.

    Each entry contains:
      - family_id, source_id, anonymous variant label
      - response text, terminal query (from replay record)
      - Empty fields for the annotator to fill in

    Args:
        run_dir: Path to the replay run directory.
        output_path: Where to write the workbook JSON.
        seed: Random seed for anonymization.

    Returns:
        The workbook dict (also written to output_path).
    """
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "replay_outputs.jsonl")

    anon_map = _build_anonymization_map(seed)
    entries: list[dict] = []

    for rec in records:
        variant = rec.get("variant")
        if variant not in anon_map:
            raise EvaluationError(
                f"unknown variant '{variant}' in replay record")
        entry = {
            "family_id": rec.get("family_id"),
            "source_id": rec.get("source_id"),
            "variant_label": anon_map[variant],
            "variant_anon_key": anon_map,
            # Context for the annotator
            "response": rec.get("response", ""),
            "response_sha256": rec.get("terminal_sha256", ""),
            # Annotator fills these in:
            "refusal_type": None,
            "unsafe_compliance_score": None,
            "compliance_level": None,
            "safe_redirection": None,
            "confidence": None,
            "rationale": "",
        }
        entries.append(entry)

    workbook = {
        "version": "1.0",
        "seed": seed,
        "anon_key": anon_map,
        "reverse_key": _reverse_map(anon_map),
        "n_entries": len(entries),
        "entries": entries,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(workbook, f, indent=2, ensure_ascii=False)

    return workbook


def parse_completed_workbook(path: str | Path) -> list[dict]:
    """Convert a completed labeling workbook to judge records.

    Args:
        path: Path to the completed workbook JSON.

    Returns:
        List of validated judgment dicts, one per entry.

    Raises:
        EvaluationError: If any entry has unfilled required fields.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        workbook = json.load(f)

    reverse_key = _reverse_map(workbook.get("anon_key", {}))
    records: list[dict] = []

    for i, entry in enumerate(workbook.get("entries", [])):
        label = entry.get("variant_label")
        variant = reverse_key.get(label)
        if variant is None:
            raise EvaluationError(
                f"entry {i}: unknown variant_label '{label}'")

        # Check required fields are filled
        for field in ("refusal_type", "unsafe_compliance_score",
                       "compliance_level", "confidence"):
            if entry.get(field) is None:
                raise EvaluationError(
                    f"entry {i} ({entry.get('family_id')}/{variant}): "
                    f"field '{field}' is not filled")

        judgment = validate_judgment({
            "refusal_type": entry["refusal_type"],
            "unsafe_compliance_score": entry["unsafe_compliance_score"],
            "compliance_level": entry["compliance_level"],
            "safe_redirection": entry.get("safe_redirection", False),
            "confidence": entry["confidence"],
            "rationale": entry.get("rationale", ""),
        })
        judgment["family_id"] = entry.get("family_id")
        judgment["variant"] = variant
        records.append(judgment)

    return records


def agreement_stats(labels_a: list[dict],
                     labels_b: list[dict]) -> dict:
    """Compute inter-annotator agreement between two label sets.

    Args:
        labels_a, labels_b: Lists of judgment dicts from two annotators.
            Must be aligned (same order, same family_id/variant).

    Returns:
        Dict with Cohen's kappa (for refusal_type, compliance_level),
        exact agreement rate, and mean absolute difference for scores.
    """
    if len(labels_a) != len(labels_b):
        raise EvaluationError(
            f"label sets must have equal length: "
            f"{len(labels_a)} vs {len(labels_b)}")

    n = len(labels_a)
    if n == 0:
        return {"kappa_refusal": None, "kappa_compliance": None,
                "exact_agreement_rate": None,
                "mean_abs_score_diff": None}

    # Exact agreement
    n_exact = sum(
        1 for a, b in zip(labels_a, labels_b)
        if (a.get("refusal_type") == b.get("refusal_type")
            and a.get("compliance_level") == b.get("compliance_level"))
    )

    # Cohen's kappa for refusal_type
    kappa_refusal = _cohens_kappa(
        [a.get("refusal_type") for a in labels_a],
        [b.get("refusal_type") for b in labels_b],
    )

    # Cohen's kappa for compliance_level
    kappa_compliance = _cohens_kappa(
        [a.get("compliance_level") for a in labels_a],
        [b.get("compliance_level") for b in labels_b],
    )

    # Mean absolute score difference
    score_diffs = [
        abs(float(a.get("unsafe_compliance_score", 0))
            - float(b.get("unsafe_compliance_score", 0)))
        for a, b in zip(labels_a, labels_b)
    ]

    return {
        "kappa_refusal": kappa_refusal,
        "kappa_compliance": kappa_compliance,
        "exact_agreement_rate": n_exact / n,
        "mean_abs_score_diff": sum(score_diffs) / n,
        "n_compared": n,
    }


def _cohens_kappa(labels_a: list, labels_b: list) -> float | None:
    """Compute Cohen's kappa for two label sequences."""
    n = len(labels_a)
    if n == 0:
        return None
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
