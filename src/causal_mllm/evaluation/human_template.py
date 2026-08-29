"""Human labeling template (Iteration 9).

Generate a JSON workbook for human annotators to label the 120
replay responses.  Variant names are replaced with anonymous labels
in random order to prevent bias.  Each entry includes the full
conversation context (system prompt, history, terminal query, images)
so annotators can judge unsafe compliance.

The completed workbook is parsed back into judge records for the
causal estimand computation.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from pathlib import Path

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.schema import validate_judgment
from causal_mllm.seeds import sha256_text

# ---------------------------------------------------------------------------
# Anonymization helpers
# ---------------------------------------------------------------------------

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


def _build_item_namespace(seed: int) -> uuid.UUID:
    """Derive a deterministic UUID namespace from the manifest seed."""
    return uuid.UUID(hashlib.sha256(
        f"iter9-workbook-seed-{seed}".encode()
    ).hexdigest()[:32])


def _make_opaque_item_id(family_id: str, variant: str,
                         seed: int) -> str:
    """Generate a fully opaque deterministic item ID.

    The ID reveals nothing about family membership or variant condition.
    """
    ns = _build_item_namespace(seed)
    return f"item-{uuid.uuid5(ns, f'{family_id}:{variant}')}"


def _extract_conversation_context(
    family: CausalFamily,
    variant_name: str,
) -> tuple[str, list[dict], str]:
    """Extract system_prompt, history_messages, terminal_query from family.

    Returns:
        (system_prompt, history_messages, terminal_query)
    """
    variant = family.variants.get(variant_name)
    if variant is None:
        raise EvaluationError(
            f"variant '{variant_name}' not found in family {family.family_id}")

    messages = []
    for msg in variant.messages:
        content_parts = []
        for img in msg.images:
            content_parts.append({"type": "image", "image": img})
        if msg.text is not None:
            content_parts.append({"type": "text", "text": msg.text})
        messages.append({
            "role": msg.role,
            "content": content_parts,
        })

    if not messages:
        raise EvaluationError(
            f"no messages in variant {variant_name} "
            f"of family {family.family_id}")

    terminal_msg = messages[-1]
    terminal_query = ""
    for part in terminal_msg.get("content", []):
        if part.get("type") == "text":
            terminal_query = part.get("text", "")
            break

    history_messages = messages[:-1]
    return "", history_messages, terminal_query


# ---------------------------------------------------------------------------
# Workbook generation
# ---------------------------------------------------------------------------

def generate_labeling_workbook(
    run_dir: str | Path,
    output_path: str | Path,
    validated_families: dict[str, CausalFamily] | None = None,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Create a JSON workbook for human labeling.

    Each entry includes full conversation context:
      - system_prompt, conversation_history, terminal_query, images
      - response text and its SHA256
      - Empty fields for the annotator to fill in

    The variant mapping and expected response hashes are stored in a
    separate sealed manifest file to prevent bias.

    Args:
        run_dir: Path to the replay run directory.
        output_path: Where to write the workbook JSON.
        validated_families: Dict of family_id -> CausalFamily for
            conversation context reconstruction.
        seed: Random seed for anonymization.

    Returns:
        (workbook, manifest) tuple. The workbook contains only opaque
        item IDs and full context; the manifest contains the decoding
        key and expected response hashes.

    Raises:
        EvaluationError: If validated_families is None or a family is
            missing.
    """
    if validated_families is None:
        raise EvaluationError(
            "validated_families is required for workbook generation — "
            "pass the loaded families dict from the runner")

    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "replay_outputs.jsonl")

    anon_map = _build_anonymization_map(seed)
    entries: list[dict] = []
    item_mapping: dict[str, dict] = {}
    expected_response_shas: dict[str, str] = {}

    # Randomize item order to prevent pattern recognition
    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    for rec in shuffled_records:
        variant = rec.get("variant")
        family_id = rec.get("family_id")
        if variant not in anon_map:
            raise EvaluationError(
                f"unknown variant '{variant}' in replay record")

        # Fully opaque item ID
        item_id = _make_opaque_item_id(family_id, variant, seed)

        # Compute actual response hash
        response_text = rec.get("response", "")
        response_sha256 = sha256_text(response_text)

        # Reconstruct conversation context from validated families
        family = validated_families.get(family_id)
        if family is None:
            raise EvaluationError(
                f"family {family_id} not found in validated_families")
        _sys, history_msgs, terminal_q = _extract_conversation_context(
            family, variant)

        # Collect image references from history and terminal
        images: list[str] = []
        for msg in history_msgs:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    images.append(part["image"])
        # Terminal query images (from the last message of the variant)
        vdata = family.variants[variant]
        if vdata.messages:
            for img in vdata.messages[-1].images:
                images.append(img)

        entry = {
            "item_id": item_id,
            # Full conversation context for the annotator
            "system_prompt": rec.get("_system_prompt", ""),
            "conversation_history": json.dumps(
                history_msgs, ensure_ascii=False),
            "terminal_query": terminal_q,
            "images": images,
            # Response and verification hash
            "response": response_text,
            "response_sha256": response_sha256,
            # Annotator fills these in:
            "refusal_type": None,
            "unsafe_compliance_score": None,
            "compliance_level": None,
            "safe_redirection": None,
            "confidence": None,
            "rationale": "",
        }
        entries.append(entry)

        # Manifest mappings
        item_mapping[item_id] = {
            "family_id": family_id,
            "variant": variant,
            "source_id": rec.get("source_id"),
        }
        expected_response_shas[item_id] = response_sha256

    workbook = {
        "version": "3.0",
        "seed": seed,
        "n_entries": len(entries),
        "entries": entries,
    }

    # Sealed manifest with decoding key and expected hashes
    manifest = {
        "version": "3.0",
        "seed": seed,
        "anon_key": anon_map,
        "reverse_key": _reverse_map(anon_map),
        "item_mapping": item_mapping,
        "expected_response_sha256": expected_response_shas,
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


# ---------------------------------------------------------------------------
# Workbook parsing
# ---------------------------------------------------------------------------

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
        EvaluationError: If any entry has unfilled required fields, if
            the manifest is missing/corrupted, or if response hashes
            don't match.
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
    expected_shas = manifest.get("expected_response_sha256", {})
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

        # Verify response hash against manifest
        response_text = entry.get("response", "")
        actual_sha = sha256_text(response_text)
        expected_sha = expected_shas.get(item_id, "")
        if expected_sha and actual_sha != expected_sha:
            raise EvaluationError(
                f"entry {i} ({item_id} = {family_id}/{variant}): "
                f"response SHA256 mismatch — "
                f"expected {expected_sha}, got {actual_sha}")

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
        judgment["response_sha256"] = actual_sha
        records.append(judgment)

    return records


# ---------------------------------------------------------------------------
# Human-label conversion and persistence
# ---------------------------------------------------------------------------

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
    rubric_version: str = "1.0",
    annotator_id: str = "unknown",
    adjudicated: bool = False,
) -> None:
    """Save human labels to JSON with provenance metadata.

    The output schema is::

        {
          "labels": { "<family_id>": { "<variant>": {...} } },
          "provenance": {
            "labels_sha256": "...",
            "label_file_sha256": "...",
            "rubric_version": "...",
            "annotator_id": "...",
            "adjudicated": false,
            "n_families": 20,
            "n_labels": 120
          }
        }

    Args:
        labels: Dict from workbook_to_human_labels.
        output_path: Where to write the labels JSON.
        label_file_sha256: Optional SHA256 of the source workbook file.
        rubric_version: Version of the labeling rubric used.
        annotator_id: Identifier for the annotator.
        adjudicated: Whether the labels have been adjudicated.
    """
    output_path = Path(output_path)

    # Compute SHA256 of the labels content for provenance
    labels_json = json.dumps(labels, sort_keys=True, ensure_ascii=False)
    labels_sha256 = hashlib.sha256(
        labels_json.encode("utf-8")).hexdigest()

    # Count total labels
    n_labels = sum(len(v) for v in labels.values())

    output = {
        "labels": labels,
        "provenance": {
            "labels_sha256": labels_sha256,
            "label_file_sha256": label_file_sha256,
            "rubric_version": rubric_version,
            "annotator_id": annotator_id,
            "adjudicated": adjudicated,
            "n_families": len(labels),
            "n_labels": n_labels,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------

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
