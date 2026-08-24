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
from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionResult,
    run_selection,
)
from causal_mllm.data.io import write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import NormalizationRejection

log = get_logger("causal_mllm.construction.pipeline")

# Output file names produced by run_selection_stage()
CANDIDATES_FILE = "candidates.jsonl"
NORMALIZATION_REJECTIONS_FILE = "normalization_rejections.jsonl"
SELECTION_REJECTIONS_FILE = "selection_rejections.jsonl"
SELECTION_REPORT_FILE = "selection_report.json"


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
    with (output_dir / SELECTION_REPORT_FILE).open("w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=2, ensure_ascii=False)

    log.info("Wrote selection artifacts to %s", output_dir)
    return result
