"""CLI: Evaluate inference results and compute causal metrics.

Usage:
    python -m causal_mllm.cli.evaluate \
        --families data/families/validated \
        --responses outputs/inference/mvp/responses.jsonl \
        --config configs/evaluation/default.yaml \
        --output outputs/evaluation/mvp
"""

from __future__ import annotations

import argparse

from causal_mllm.data.io import load_config
from causal_mllm.data.logging import setup_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate inference results and compute causal metrics.",
    )
    parser.add_argument(
        "--families",
        type=str,
        required=True,
        help="Directory containing validated family JSONL files.",
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to inference responses JSONL.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for evaluation results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    log = get_logger("causal_mllm.cli.evaluate")

    config = load_config(args.config)

    log.info("Evaluating: families=%s, responses=%s, output=%s",
             args.families, args.responses, args.output)

    # TODO: Implement evaluation pipeline (Iteration 9)
    log.warning("evaluate is a stub. Implementation begins at Iteration 9.")
    log.info("Done (no-op).")


if __name__ == "__main__":
    main()
