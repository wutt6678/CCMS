"""CLI: Type-B cross-modality alignment diagnostics.

Measures how much of MTMCS Type-B is directly usable for the 2x2
factorial construction (one q* across modalities) versus how much
requires rewriting.

Usage:
    python -m causal_mllm.cli.diagnose_type_b \
        --output outputs/diagnostics/type_b_alignment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_mllm.data.logging import get_logger, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="diagnose_type_b",
        description="Diagnose MTMCS Type-B cross-modality terminal-query "
                    "and per-turn alignment across all source rows.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/diagnostics/type_b_alignment.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect the first N rows (smoke mode).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    log = get_logger("causal_mllm.cli.diagnose_type_b")

    from causal_mllm.adapters.mtmcs import MTMCSAdapter
    from causal_mllm.construction.diagnostics import diagnose_type_b_rows

    adapter = MTMCSAdapter()
    rows = []
    for row in adapter.load("type_b"):
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    report = diagnose_type_b_rows(rows)
    report["split"] = "type_b"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    term = report["terminal_query_cross_modality"]
    usable = report["directly_usable"]
    log.info("Type-B diagnostics over %d rows (%d complete)",
             report["n_type_b"], report["n_rows_complete"])
    log.info("Terminal q* mm-vs-text: exact %d (%.1f%%), normalized %d (%.1f%%)",
             term["n_exact_match"], 100 * term["fraction_exact_match"],
             term["n_normalized_match"], 100 * term["fraction_normalized_match"])
    log.info("Directly usable: %d exact / %d normalized; rewrite needed: "
             "%d exact / %d normalized",
             usable["n_exact"], usable["n_normalized"],
             usable["n_requiring_rewrite_exact"],
             usable["n_requiring_rewrite_normalized"])
    log.info("Report written to %s", out_path)


if __name__ == "__main__":
    main()
