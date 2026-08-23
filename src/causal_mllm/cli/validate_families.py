"""CLI: Validate causal families.

Usage:
    python -m causal_mllm.cli.validate_families \
        --input data/families/draft --output data/families/validated
"""

from __future__ import annotations

import argparse

from causal_mllm.data.logging import setup_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_families",
        description="Run validation checks on causal families.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing draft family JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write validated families.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Reject families that fail any soft check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    log = get_logger("causal_mllm.cli.validate_families")

    log.info("Validating families: input=%s, output=%s, strict=%s",
             args.input, args.output, args.strict)

    # TODO: Implement validation pipeline (Iteration 6)
    log.warning("validate_families is a stub. Implementation begins at Iteration 6.")
    log.info("Done (no-op).")


if __name__ == "__main__":
    main()
