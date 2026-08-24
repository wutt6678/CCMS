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
from causal_mllm.construction.annotation import AtomAnnotator
from causal_mllm.construction.families import build_family_skeletons
from causal_mllm.construction.harmonize import (
    TerminalHarmonizer,
    apply_terminal_harmonization,
)
from causal_mllm.construction.readiness import family_readiness
from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionResult,
    build_family_review_flags,
    group_into_family_units,
    run_selection,
)
from causal_mllm.construction.variants import build_family_variants
from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import CausalFamily, NormalizationRejection
from causal_mllm.data.validate_schema import (
    SchemaValidationError,
    validate_causal_family,
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

# Output file names produced by the Iteration-5 stages
ANNOTATED_SKELETONS_FILE = "annotated_skeletons.jsonl"
ANNOTATION_REPORT_FILE = "annotation_report.json"
HARMONIZED_FAMILIES_FILE = "harmonized_families.jsonl"
FAMILIES_FILE = "families.jsonl"
VARIANTS_REPORT_FILE = "variants_report.json"


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


def _read_families(path: Path) -> list[CausalFamily]:
    return [CausalFamily.from_dict(rec) for rec in read_jsonl(path)]


def run_annotation_stage(
    annotator: AtomAnnotator,
    output_dir: str | Path,
) -> list:
    """Iteration 5A: apply semantic annotations to family skeletons.

    Reads family_skeletons.jsonl, applies the annotator (copy-only —
    skeletons on disk are regenerated from the annotated copies), and
    reports per-family readiness so unresolved semantics are visible
    BEFORE variant generation.

    Returns:
        List of annotated CausalFamily skeletons.
    """
    import datetime

    output_dir = Path(output_dir)
    skeletons_path = output_dir / FAMILY_SKELETONS_FILE
    if not skeletons_path.exists():
        raise FileNotFoundError(
            f"{skeletons_path} not found — run the atoms stage first"
        )
    skeletons = _read_families(skeletons_path)

    annotated = [annotator.annotate_family(s) for s in skeletons]

    readiness = {
        s.family_id: family_readiness(a)
        for s, a in zip(skeletons, annotated)
    }
    write_jsonl(output_dir / ANNOTATED_SKELETONS_FILE,
                [a.to_dict() for a in annotated])
    report = {
        "iteration": "5A",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "annotator": type(annotator).__name__,
        "n_families": len(annotated),
        "n_l1_semantic": sum(
            1 for r in readiness.values()
            if r["level"] in ("L1_semantic", "L2_variant_ready")
        ),
        "readiness": {
            fid: {"level": r["level"],
                  "L1_semantic_gaps": r["L1_semantic"]}
            for fid, r in readiness.items()
        },
    }
    with (output_dir / ANNOTATION_REPORT_FILE).open(
            "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Annotated %d skeletons (%d at L1+) -> %s",
             len(annotated), report["n_l1_semantic"], output_dir)
    return annotated


def run_harmonization_stage(
    harmonizer: TerminalHarmonizer,
    output_dir: str | Path,
) -> list:
    """Iteration 5B: construct one canonical q* per family.

    Reads annotated_skeletons.jsonl (falling back to family_skeletons),
    applies the harmonizer, and persists harmonized_families.jsonl.
    Original skeleton terminal queries are preserved untouched; the
    canonical query lives in validation.terminal_harmonization.
    Missing required harmonizations fail loudly.
    """
    output_dir = Path(output_dir)
    source_path = output_dir / ANNOTATED_SKELETONS_FILE
    if not source_path.exists():
        source_path = output_dir / FAMILY_SKELETONS_FILE
    if not source_path.exists():
        raise FileNotFoundError(
            f"No skeletons found in {output_dir} — run earlier stages first"
        )
    families = _read_families(source_path)

    harmonized = [apply_terminal_harmonization(f, harmonizer)
                  for f in families]
    write_jsonl(output_dir / HARMONIZED_FAMILIES_FILE,
                [h.to_dict() for h in harmonized])
    log.info("Harmonized %d families (backend=%s) -> %s",
             len(harmonized), harmonizer.method, output_dir)
    return harmonized


def run_variants_stage(
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> list:
    """Iteration 5C: generate all six variants per family, gated.

    Reads harmonized_families.jsonl, builds variants (each generator
    asserts its own prerequisites — unresolved semantics raise
    VariantPrerequisiteError), validates the full family schema, and
    persists families.jsonl + variants_report.json.
    """
    import datetime

    output_dir = Path(output_dir)
    source_path = output_dir / HARMONIZED_FAMILIES_FILE
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found — run the harmonize stage first"
        )
    harmonized = _read_families(source_path)

    complete = [build_family_variants(f, seed=seed) for f in harmonized]
    for family in complete:
        errors = validate_causal_family(family.to_dict())
        if errors:
            raise SchemaValidationError(
                [f"{family.family_id}: {e}" for e in errors]
            )

    write_jsonl(output_dir / FAMILIES_FILE,
                [f.to_dict() for f in complete])
    report = {
        "iteration": "5C",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "n_families": len(complete),
        "n_trajectories": sum(len(f.variants) for f in complete),
        "variant_names": sorted(
            {name for f in complete for name in f.variants}),
        "family_ids": [f.family_id for f in complete],
        "cross_modal_status": "candidate",  # causality = Iteration 6+
    }
    with (output_dir / VARIANTS_REPORT_FILE).open(
            "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Generated %d families x 6 variants (%d trajectories) -> %s",
             len(complete), report["n_trajectories"], output_dir)
    return complete
