"""CLI: Inspect source dataset schemas.

Usage:
    python -m causal_mllm.cli.inspect_source \
        --dataset mtmcs --split test --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causal_mllm.data.io import load_config
from causal_mllm.data.logging import setup_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inspect_source",
        description="Inspect and report the schema of a source dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mtmcs", "cosafe", "mtid"],
        help="Source dataset to inspect.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to inspect (default: dataset-specific preferred split).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of rows to inspect (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/schema",
        help="Directory to write schema reports (default: outputs/schema).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Hugging Face cache directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    log = get_logger("causal_mllm.cli.inspect_source")

    log.info("Inspecting dataset: %s (split=%s, n=%d)", args.dataset, args.split, args.n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import the correct adapter
    if args.dataset == "mtmcs":
        from causal_mllm.adapters.mtmcs import MTMCSAdapter
        adapter = MTMCSAdapter(cache_dir=args.cache_dir)
    elif args.dataset == "cosafe":
        from causal_mllm.adapters.cosafe import CoSafeAdapter
        adapter = CoSafeAdapter(data_dir=args.cache_dir)
    elif args.dataset == "mtid":
        from causal_mllm.adapters.mtid import MTIDAdapter
        adapter = MTIDAdapter(cache_dir=args.cache_dir)
    else:
        log.error("Unknown dataset: %s", args.dataset)
        sys.exit(1)

    # Run schema inspection
    log.info("Running schema inspection...")
    try:
        report = adapter.inspect_schema(n=args.n)
    except Exception as e:
        log.error("Schema inspection failed for %s: %s", args.dataset, e)
        sys.exit(1)

    # Write schema report
    report_path = output_dir / f"{args.dataset}_schema.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info("Schema report written to %s", report_path)

    # Write example rows
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples_path = examples_dir / f"{args.dataset}_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(report.get("example_values", []), f, indent=2, ensure_ascii=False, default=str)
    log.info("Example rows written to %s", examples_path)

    # Summary
    log.info("Schema inspection complete for %s", args.dataset)
    log.info("  Total rows: %s", report.get("total_rows", "unknown"))
    log.info("  Columns: %s", report.get("column_names", []))


if __name__ == "__main__":
    main()
