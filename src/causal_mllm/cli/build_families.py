"""CLI: Build causal families from source datasets.

Usage:
    python -m causal_mllm.cli.build_families \
        --config configs/generation/mvp.yaml --max-families 5
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
        choices=["select", "atoms", "variants", "all"],
        help="Run only a specific stage (default: all).",
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

    if stage in ("select", "all"):
        from causal_mllm.construction.pipeline import run_selection_stage

        if max_families is not None:
            selection_cfg["max_families"] = int(max_families)
        config["selection"] = selection_cfg

        output_dir = args.output_dir or config.get("output", {}).get(
            "families_dir", "data/families/draft")
        result = run_selection_stage(
            config, output_dir, max_rows=args.max_rows,
        )
        report = result.report
        log.info(
            "Selection done: %d/%d records accepted (%d families), "
            "%d rejected. Report: %s",
            report["n_accepted"], report["n_input"],
            report["n_families_accepted"], report["n_rejected"],
            report["reason_counts"],
        )

    if stage in ("atoms", "variants") or stage == "all":
        # TODO: Implement remaining stages (Iteration 4+)
        log.warning("Stages beyond 'select' are stubs. "
                    "Atom extraction begins at Iteration 4.")

    log.info("Done.")


if __name__ == "__main__":
    main()
