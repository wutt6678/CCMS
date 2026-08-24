"""Construction pipeline stage: candidate selection (Iteration 3).

End-to-end "select" stage:

    adapter -> load_and_normalize(on_error="record") -> select_candidates
            -> candidates.jsonl + rejection manifests + report

Fail-closed by construction:

  * Normalization errors are recorded (never silently skipped) via the
    adapter's rejection manifest.
  * Selection rejections carry machine-readable reasons.
  * The accounting invariant (every input record is either accepted or
    rejected) is asserted before anything is written.
  * Selection is pass-through: accepted records are written back exactly
    as normalized — no synthetic assistant responses, no edits. Source
    trajectory != experimental frozen trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_mllm.adapters import get_adapter
from causal_mllm.construction.families import build_family_skeletons
from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionResult,
    build_family_review_flags,
    group_into_family_units,
    run_selection,
)
from causal_mllm.data.io import write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import NormalizationRejection
from causal_mllm.data.validate_schema import (
    SchemaValidationError,
    validate_family_skeleton,
)

log = get_logger("causal_mllm.construction.pipeline")

# Output file names produced by run_selection_stage()
CANDIDATES_FILE = "candidates.jsonl"
NORMALIZATION_REJECTIONS_FILE = "normalization_rejections.jsonl"
SELECTION_REJECTIONS_FILE = "selection_rejections.jsonl"
SELECTION_REPORT_FILE = "selection_report.json"
FAMILY_REVIEW_FLAGS_FILE = "family_review_flags.jsonl"

# Output file names produced by run_atoms_stage()
FAMILY_SKELETONS_FILE = "family_skeletons.jsonl"
ATOMS_REPORT_FILE = "atoms_report.json"


def _load_normalized(config: dict, *, max_rows: int | None,
                     max_examples: int | None) -> tuple[list, list[NormalizationRejection]]:
    """Load + normalize source records with rejection recording."""
    source_cfg = config.get("source", {})
    dataset_name = source_cfg.get("dataset")
    if not dataset_name:
        raise ValueError("config must define source.dataset")

    split = source_cfg.get("split")
    adapter = get_adapter(dataset_name)
    rejections: list[NormalizationRejection] = []

    if dataset_name.lower() == "mtmcs":
        # Atomic grouped loading: max_rows limits SOURCE ROWS (x4 records).
        # split=None -> load both type_a and type_b.
        effective_max_rows = max_rows if max_rows is not None \
            else source_cfg.get("max_rows")
        records = []
        splits = [split] if split else ["type_a", "type_b"]
        for s in splits:
            records.extend(adapter.load_and_normalize(
                split=s,
                max_rows=effective_max_rows,
                on_error="record",
                rejections=rejections,
            ))
    else:
        records = adapter.load_and_normalize(
            split=split,
            max_examples=max_examples,
            on_error="record",
            rejections=rejections,
        )

    return records, rejections


def run_selection_stage(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    max_rows: int | None = None,
    max_examples: int | None = None,
) -> SelectionResult:
    """Run the full candidate-selection stage and persist artifacts.

    Args:
        config: Generation config (must contain ``source.dataset``;
            optionally ``source.split`` and a ``selection`` section).
        output_dir: Directory for candidates, rejection manifests, and
            the selection report.
        max_rows: MTMCS-only limit on source rows (atomic groups).
        max_examples: Record-level limit for non-MTMCS adapters.

    Returns:
        SelectionResult with accepted records, rejections, and report.

    Raises:
        AssertionError: If any accounting invariant is violated.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, norm_rejections = _load_normalized(
        config, max_rows=max_rows, max_examples=max_examples,
    )
    log.info("Normalized %d records (%d normalization rejections)",
             len(records), len(norm_rejections))

    sel_config = SelectionConfig.from_config(config.get("selection", {}))
    result = run_selection(records, sel_config)
    log.info("Selected %d candidates (%d selection rejections)",
             len(result.accepted), len(result.rejections))

    # ---- Persist artifacts ----
    write_jsonl(output_dir / CANDIDATES_FILE,
                [ex.to_dict() for ex in result.accepted])
    write_jsonl(output_dir / NORMALIZATION_REJECTIONS_FILE,
                [r.to_dict() for r in norm_rejections])
    write_jsonl(output_dir / SELECTION_REJECTIONS_FILE,
                [r.to_dict() for r in result.rejections])
    # Standalone terminal-risk review flags: placeholders until a judge
    # estimates Risk(q*) in Iteration 6. No family is treated as a strict
    # causal candidate before that validation.
    write_jsonl(output_dir / FAMILY_REVIEW_FLAGS_FILE,
                build_family_review_flags(result))
    with (output_dir / SELECTION_REPORT_FILE).open("w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=2, ensure_ascii=False)

    log.info("Wrote selection artifacts to %s", output_dir)
    return result


def run_atoms_stage(
    selection_result: SelectionResult,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> list:
    """Run the family-level comparative atom extraction stage.

    Groups accepted candidates into family units, decomposes each unit
    comparatively (H_safe vs H_unsafe for MTMCS), builds family skeletons,
    validates every skeleton fail-loud, and persists them.

    Args:
        selection_result: Output of run_selection_stage().
        output_dir: Artifact directory (same as the selection stage).
        seed: Experiment seed for deterministic family IDs.

    Returns:
        List of validated CausalFamily skeletons (variants empty).

    Raises:
        SchemaValidationError: If any skeleton fails validation.
    """
    import datetime

    from causal_mllm.seeds import get_git_commit

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    units = group_into_family_units(selection_result.accepted)
    skeletons = build_family_skeletons(units, seed=seed)

    # Fail loudly: no skeleton may persist if it violates the schema
    for skeleton in skeletons:
        errors = validate_family_skeleton(skeleton.to_dict())
        if errors:
            raise SchemaValidationError(
                [f"{skeleton.family_id}: {e}" for e in errors]
            )

    write_jsonl(output_dir / FAMILY_SKELETONS_FILE,
                [s.to_dict() for s in skeletons])

    n_atoms = sum(len(s.semantic_atoms) for s in skeletons)
    n_causal = sum(
        1 for s in skeletons for a in s.semantic_atoms
        if a.divergence == "causal"
    )
    report = {
        "iteration": 4,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "seed": seed,
        "extraction_backend": "rule",
        "n_families": len(skeletons),
        "n_atoms": n_atoms,
        "n_causal_atoms": n_causal,
        "family_ids": [s.family_id for s in skeletons],
    }
    with (output_dir / ATOMS_REPORT_FILE).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Built %d family skeletons (%d atoms, %d causal) -> %s",
             len(skeletons), n_atoms, n_causal, output_dir)
    return skeletons
