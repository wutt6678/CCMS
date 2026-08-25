"""Frozen replay runner (Iteration 8): trajectory -> raw response.

Consumes validated_families.jsonl ONLY and stores raw model responses
with full provenance, separate from the dataset artifacts. No judging:
the safety judge and the causal estimands arrive in Iteration 9.

Usage::

    python -m causal_mllm.cli.replay \
        --input-dir outputs/families/scale_b_smoke \
        --max-families 5          # smoke; omit for the full run
"""

from __future__ import annotations

import argparse

from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.runner import run_replay_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen replay of validated families (Iteration 8)")
    parser.add_argument("--input-dir", required=True,
                        help="Dataset dir with validated_families.jsonl")
    parser.add_argument("--output-root", default="outputs/replay_runs")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Limit families (smoke runs)")
    parser.add_argument("--model-name", default=ReplayConfig.model_name)
    parser.add_argument("--device", default=ReplayConfig.device)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = ReplayConfig(model_name=args.model_name, device=args.device)
    report = run_replay_stage(
        args.input_dir, args.output_root, config=config,
        max_families=args.max_families, run_id=args.run_id)
    print(f"run_id: {report['run_id']}")
    print(f"attempted: {report['n_attempted']}/{report['expected_attempts']}"
          f" succeeded: {report['n_succeeded']} failed: {report['n_failed']}")


if __name__ == "__main__":
    main()
