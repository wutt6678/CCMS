"""CLI: Build causal families from source datasets.

Usage:
    python -m causal_mllm.cli.build_families \
        --config configs/generation/mvp.yaml --max-families 5
"""

from __future__ import annotations

import argparse
import sys

from causal_mllm.data.io import load_config
from causal_mllm.data.logging import setup_logging, get_logger


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
        help="Override max_families from config.",
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
    logger = setup_logging()
    log = get_logger("causal_mllm.cli.build_families")

    config = load_config(args.config)
    max_families = args.max_families or config.get("source", {}).get("max_families", 5)

    log.info("Building families: config=%s, max_families=%d, stage=%s",
             args.config, max_families, args.stage or "all")

    # TODO: Implement stages (Iteration 3+)
    log.warning("build_families is a stub. Implementation begins at Iteration 3.")
    log.info("Done (no-op).")


if __name__ == "__main__":
    main()
