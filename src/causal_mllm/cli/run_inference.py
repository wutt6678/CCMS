"""CLI: Run model inference on causal families.

Usage:
    python -m causal_mllm.cli.run_inference \
        --families data/families/validated \
        --model-config configs/models/qwen_mllm.yaml \
        --output outputs/inference/mvp
"""

from __future__ import annotations

import argparse

from causal_mllm.data.io import load_config
from causal_mllm.data.logging import setup_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_inference",
        description="Run frozen replay inference on causal families.",
    )
    parser.add_argument(
        "--families",
        type=str,
        required=True,
        help="Directory containing validated family JSONL files.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for inference results.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID (default: auto-generated).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    log = get_logger("causal_mllm.cli.run_inference")

    model_config = load_config(args.model_config)

    log.info("Running inference: families=%s, model=%s, output=%s",
             args.families, model_config.get("model_name_or_path", "unknown"), args.output)

    # TODO: Implement frozen replay (Iteration 8)
    log.warning("run_inference is a stub. Implementation begins at Iteration 8.")
    log.info("Done (no-op).")


if __name__ == "__main__":
    main()
