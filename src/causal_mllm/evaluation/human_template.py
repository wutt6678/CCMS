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
) -> tuple[dict, dict]:
    """Create a JSON workbook for human labeling.

    Each entry contains:
      - family_id, source_id, anonymous variant label
      - response text, terminal query (from replay record)
      - Empty fields for the annotator to fill in

    The variant mapping is stored in a separate sealed manifest file
    (output_path with .manifest.json extension) to prevent bias.

    Args:
        run_dir: Path to the replay run directory.
        output_path: Where to write the workbook JSON.
        seed: Random seed for anonymization.

    Returns:
        (workbook, manifest) tuple. The workbook contains only opaque
        item IDs; the manifest contains the decoding key.
    """
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "replay_outputs.jsonl")

    anon_map = _build_anonymization_map(seed)
    entries: list[dict] = []

    # Randomize item order to prevent pattern recognition
    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    for rec in shuffled_records:
        variant = rec.get("variant")
        if variant not in anon_map:
            raise EvaluationError(
                f"unknown variant '{variant}' in replay record")
        # Generate opaque item ID
        item_id = f"{rec.get('family_id')}_{anon_map[variant]}"
        entry = {
            "item_id": item_id,
            # Context for the annotator
            "response": rec.get("response", ""),
            "response_sha256": rec.get("response_sha256", rec.get("terminal_sha256", "")),
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
        "version": "2.0",
        "seed": seed,
        "n_entries": len(entries),
        "entries": entries,
    }

    # Create sealed manifest with decoding key
    manifest = {
        "version": "2.0",
        "seed": seed,
        "anon_key": anon_map,
        "reverse_key": _reverse_map(anon_map),
        "item_mapping": {
            f"{rec.get('family_id')}_{anon_map[rec.get('variant')]}": {
                "family_id": rec.get("family_id"),
                "variant": rec.get("variant"),
                "source_id": rec.get("source_id"),
            }
            for rec in records
        },
    }

    # Write workbook
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(workbook, f, indent=2, ensure_ascii=False)

    # Write sealed manifest
    manifest_path = output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return workbook, manifest


def parse_completed_workbook(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    """Convert a completed labeling workbook to judge records.

    Args:
        path: Path to the completed workbook JSON.
        manifest_path: Path to the sealed manifest JSON. If None, looks
            for a .manifest.json file alongside the workbook.

    Returns:
        List of validated judgment dicts, one per entry. Each dict includes
        family_id, variant, source_id, and response_sha256 for verification.

    Raises:
        EvaluationError: If any entry has unfilled required fields or if
            the manifest is missing/corrupted.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        workbook = json.load(f)

    # Load manifest for decoding
    if manifest_path is None:
        manifest_path = path.with_suffix(".manifest.json")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise EvaluationError(
            f"sealed manifest not found: {manifest_path} — "
            f"cannot decode item_id to family_id/variant")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    item_mapping = manifest.get("item_mapping", {})
    records: list[dict] = []

    for i, entry in enumerate(workbook.get("entries", [])):
        item_id = entry.get("item_id")
        if item_id is None:
            raise EvaluationError(
                f"entry {i}: missing item_id")

        # Decode item_id to family_id/variant using manifest
        mapping = item_mapping.get(item_id)
        if mapping is None:
            raise EvaluationError(
                f"entry {i}: item_id '{item_id}' not found in manifest")

        family_id = mapping["family_id"]
        variant = mapping["variant"]
        source_id = mapping.get("source_id")

        # Check required fields are filled
        for field in ("refusal_type", "unsafe_compliance_score",
                       "compliance_level", "confidence"):
            if entry.get(field) is None:
                raise EvaluationError(
                    f"entry {i} ({item_id} = {family_id}/{variant}): "
                    f"field '{field}' is not filled")

        judgment = validate_judgment({
            "refusal_type": entry["refusal_type"],
            "unsafe_compliance_score": entry["unsafe_compliance_score"],
            "compliance_level": entry["compliance_level"],
            "safe_redirection": entry.get("safe_redirection", False),
            "confidence": entry["confidence"],
            "rationale": entry.get("rationale", ""),
        })
        judgment["family_id"] = family_id
        judgment["variant"] = variant
        judgment["source_id"] = source_id
        judgment["item_id"] = item_id
        judgment["response_sha256"] = entry.get("response_sha256", "")
        records.append(judgment)

    return records


def workbook_to_human_labels(
    parsed_records: list[dict],
    rubric_version: str = "1.0",
    annotator_id: str = "unknown",
    adjudicated: bool = False,
) -> dict:
    """Convert parsed workbook records to HumanLabelJudge format.

    Args:
        parsed_records: List of judgment dicts from parse_completed_workbook.
        rubric_version: Version of the labeling rubric used.
        annotator_id: Identifier for the annotator.
        adjudicated: Whether the labels have been adjudicated.

    Returns:
        Dict in the format expected by HumanLabelJudge:
        {family_id: {variant: {label_dict}}}
    """
    labels: dict[str, dict[str, dict]] = {}
    for rec in parsed_records:
        family_id = rec["family_id"]
        variant = rec["variant"]
        if family_id not in labels:
            labels[family_id] = {}
        labels[family_id][variant] = {
            "refusal_type": rec["refusal_type"],
            "unsafe_compliance_score": rec["unsafe_compliance_score"],
            "compliance_level": rec["compliance_level"],
            "safe_redirection": rec.get("safe_redirection", False),
            "confidence": rec["confidence"],
            "rationale": rec.get("rationale", ""),
            # Provenance metadata
            "response_sha256": rec.get("response_sha256", ""),
            "rubric_version": rubric_version,
            "annotator_id": annotator_id,
            "adjudicated": adjudicated,
            "item_id": rec.get("item_id", ""),
        }
    return labels


def save_human_labels(
    labels: dict,
    output_path: str | Path,
    label_file_sha256: str | None = None,
) -> None:
    """Save human labels to JSON with provenance metadata.

    Args:
        labels: Dict from workbook_to_human_labels.
        output_path: Where to write the labels JSON.
        label_file_sha256: Optional SHA256 of the source workbook file.
    """
    import hashlib
    output_path = Path(output_path)

    # Compute SHA256 of the labels content for provenance
    labels_json = json.dumps(labels, sort_keys=True, ensure_ascii=False)
    labels_sha256 = hashlib.sha256(labels_json.encode("utf-8")).hexdigest()

    output = {
        "labels": labels,
        "provenance": {
            "labels_sha256": labels_sha256,
            "label_file_sha256": label_file_sha256,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


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
