"""CLI: Build causal families from source datasets.

Stage chain (each stage runs all predecessors):
    select -> atoms -> annotate -> harmonize -> variants -> validate

Usage:
    python -m causal_mllm.cli.build_families \
        --config configs/generation/mvp.yaml --max-families 5

    python -m causal_mllm.cli.build_families \
        --config configs/generation/mvp.yaml --stage variants \
        --annotations data/families/annotations.json \
        --harmonization data/families/harmonization.json
"""

from __future__ import annotations

import argparse

from causal_mllm.data.io import load_config
from causal_mllm.data.logging import get_logger, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_families",
        description="Build causal counterfactual families from normalized source examples.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generation config YAML.",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=None,
        help="Override selection.max_families from config.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="MTMCS only: limit source ROWS loaded (atomic x4 groups).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["select", "atoms", "annotate", "harmonize", "variants",
                 "validate", "all"],
        help="Run the pipeline up to this stage (default: all).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Manual annotation JSON ({family_key: {atom_id: payload}}) "
             "for the annotate stage.",
    )
    parser.add_argument(
        "--harmonization",
        type=str,
        default=None,
        help="Manual harmonization JSON ({family_key: canonical_q}) "
             "for the harmonize stage.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Manual risk-judge JSON ({family_key: {variant: score}}) "
             "for the validate stage; without it behavioral fields "
             "stay null (candidates only).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Strict cross-modal causal-subset threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    log = get_logger("causal_mllm.cli.build_families")

    config = load_config(args.config)
    selection_cfg = dict(config.get("selection", {}))
    max_families = args.max_families or selection_cfg.get("max_families") \
        or config.get("source", {}).get("max_families")
    stage = args.stage or "all"

    log.info("Building families: config=%s, max_families=%s, stage=%s",
             args.config, max_families, stage)

    output_dir = args.output_dir or config.get("output", {}).get(
        "families_dir", "data/families/draft")
    seed = int(config.get("seed", 42))

    # The stage chain: every stage runs all of its predecessors.
    chain = ["select", "atoms", "annotate", "harmonize", "variants",
             "validate"]
    target = chain.index("validate" if stage == "all" else stage)
    run_through = chain[:target + 1]

    selection_result = None

    if "select" in run_through:
        from causal_mllm.construction.pipeline import run_selection_stage

        if max_families is not None:
            selection_cfg["max_families"] = int(max_families)
        config["selection"] = selection_cfg

        selection_result = run_selection_stage(
            config, output_dir, max_rows=args.max_rows,
        )
        report = selection_result.report
        log.info(
            "Selection done: %d/%d records accepted (%d families), "
            "%d rejected. Rejected families by reason: %s",
            report["n_accepted"], report["n_input"],
            report["n_families_accepted"], report["n_rejected"],
            report["rejected_families_by_reason"],
        )
        log.info("Accepted families by source intent: %s | by safety: %s",
                 report["families_by_source_intent"], report["families_by_safety"])
        for warning in report["balance_warnings"]:
            log.warning("Balance check: %s", warning)

    if "atoms" in run_through:
        from causal_mllm.construction.pipeline import run_atoms_stage

        skeletons = run_atoms_stage(selection_result, output_dir, seed=seed)
        log.info("Atoms stage done: %d family skeletons "
                 "(comparative H_safe-vs-H_unsafe decomposition)",
                 len(skeletons))

    if "annotate" in run_through:
        from causal_mllm.construction.annotation import ManualFileAnnotator
        from causal_mllm.construction.pipeline import run_annotation_stage

        if not args.annotations:
            raise SystemExit(
                "The annotate stage requires --annotations "
                "(manual annotation JSON; LLM backends are wired via the "
                "CallableAnnotator API)."
            )
        annotator = ManualFileAnnotator(args.annotations)
        annotated = run_annotation_stage(annotator, output_dir)
        log.info("Annotation stage done: %d families", len(annotated))

    if "harmonize" in run_through:
        from causal_mllm.construction.harmonize import ManualHarmonizer
        from causal_mllm.construction.pipeline import run_harmonization_stage

        if not args.harmonization:
            raise SystemExit(
                "The harmonize stage requires --harmonization "
                "(manual canonical-q JSON; LLM backends are wired via the "
                "CallableHarmonizer API)."
            )
        harmonizer = ManualHarmonizer(args.harmonization)
        harmonized = run_harmonization_stage(harmonizer, output_dir)
        log.info("Harmonization stage done: %d families", len(harmonized))

    if "variants" in run_through:
        from causal_mllm.construction.pipeline import run_variants_stage

        complete = run_variants_stage(output_dir, seed=seed)
        log.info("Variant stage done: %d families x 6 variants "
                 "(%d trajectories; cross_modal = CANDIDATE until "
                 "behavioral validation in Iteration 6+)",
                 len(complete), 6 * len(complete))

    if "validate" in run_through:
        from causal_mllm.validation import (
            ManualFileJudge,
            run_validation_stage,
        )

        judge = ManualFileJudge(args.judge) if args.judge else None
        validated = run_validation_stage(
            output_dir, judge=judge, theta=args.theta)
        log.info("Validation stage done: %d validated families "
                 "(judge=%s, theta=%.2f)",
                 len(validated), "manual" if judge else "none", args.theta)

    log.info("Done.")


if __name__ == "__main__":
    main()
