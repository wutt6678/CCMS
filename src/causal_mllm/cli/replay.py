"""Frozen replay runner (Iteration 8) + cross-model targets (Iteration 11).

Consumes validated_families.jsonl ONLY and stores raw model responses
with full provenance, separate from the dataset artifacts. No judging:
the safety judge and the causal estimands arrive in Iteration 9.

Without ``--model-key`` this is the frozen single-model (Qwen3.5-9B)
path, byte-for-byte as in Iterations 8-10.  With ``--model-key`` the
target is resolved exactly once from the frozen Iteration 11 model
registry and dispatched to its thin family adapter; outputs land in a
model-separated directory.

Usage::

    python -m causal_mllm.cli.replay \
        --input-dir outputs/families/scale_b_smoke \
        --max-families 5          # smoke; omit for the full run

    # Iteration 11 target (preflight resolves + records the revision)
    python -m causal_mllm.cli.replay \
        --input-dir outputs/families/scale_c \
        --model-key qwen35_4b --preflight --max-families 1
"""

from __future__ import annotations

import argparse

from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.registry import resolve_model
from causal_mllm.replay.runner import run_replay_stage

LEGACY_OUTPUT_ROOT = "outputs/replay_runs"
ITER11_OUTPUT_ROOT = "outputs/iteration_11/generations"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen replay of validated families "
                    "(Iteration 8 / Iteration 11 targets)")
    parser.add_argument("--input-dir", required=True,
                        help="Dataset dir with validated_families.jsonl")
    parser.add_argument("--output-root", default=None,
                        help=f"Defaults to {LEGACY_OUTPUT_ROOT}, or "
                             f"{ITER11_OUTPUT_ROOT}/<model_key> with "
                             f"--model-key")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Limit families (smoke runs)")
    parser.add_argument("--model-name", default=ReplayConfig.model_name)
    parser.add_argument("--model-revision", default=None,
                        help="Pin the weights revision actually loaded")
    parser.add_argument("--max-new-tokens", type=int,
                        default=ReplayConfig.max_new_tokens)
    parser.add_argument("--device", default=ReplayConfig.device)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Allow overwriting an existing run directory")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Continue an interrupted run, skipping "
                             "(family, variant) pairs already recorded "
                             "under the same resolved run fingerprint")
    # --- Iteration 11 cross-model target selection ---------------------
    parser.add_argument("--model-key", default=None,
                        help="Registry key (e.g. qwen35_4b, ministral3_3b, "
                             "phi4_mm). Selects the frozen target model + "
                             "family adapter from the Iteration 11 registry")
    parser.add_argument("--registry", default=None,
                        help="Model registry path (default: frozen "
                             "outputs/iteration_11/protocol/"
                             "model_registry.yaml)")
    parser.add_argument("--lock", default=None,
                        help="resolved_models.lock.yaml supplying immutable "
                             "revisions for confirmatory runs")
    parser.add_argument("--preflight", action="store_true", default=False,
                        help="Technical preflight only: allow a null "
                             "revision and resolve it at load time. "
                             "Confirmatory runs (the default) reject "
                             "null/branch/'main'/'latest'")
    args = parser.parse_args()

    model_spec = None
    model_name = args.model_name
    model_revision = args.model_revision
    if args.model_key:
        model_spec = resolve_model(
            args.model_key,
            confirmatory=not args.preflight,
            registry_path=args.registry,
            lock_path=args.lock,
        )
        model_name = model_spec.model_id
        # The registry/lock revision wins; --model-revision may not
        # silently override a frozen target.
        model_revision = model_spec.revision
        if args.model_revision and args.model_revision != model_revision:
            parser.error(
                f"--model-revision {args.model_revision!r} conflicts with "
                f"the registry-resolved revision {model_revision!r} for "
                f"{args.model_key}")

    config = ReplayConfig(model_name=model_name,
                          model_revision=model_revision,
                          max_new_tokens=args.max_new_tokens,
                          device=args.device,
                          enable_thinking=(model_spec.thinking_mode
                                           if model_spec else
                                           ReplayConfig.enable_thinking))

    if args.output_root:
        output_root = args.output_root
    elif model_spec is not None:
        output_root = f"{ITER11_OUTPUT_ROOT}/{model_spec.model_key}"
    else:
        output_root = LEGACY_OUTPUT_ROOT

    report = run_replay_stage(
        args.input_dir, output_root, config=config,
        max_families=args.max_families, run_id=args.run_id,
        overwrite=args.overwrite, model_spec=model_spec,
        resume=args.resume)
    print(f"run_id: {report['run_id']}")
    if model_spec is not None:
        prov = report["provenance"]
        print(f"model_key: {report['model_key']} "
              f"({model_spec.model_id} adapter={report['adapter']})")
        print(f"requested_revision: {prov['requested_model_revision']}")
        print(f"resolved_revision: {prov['resolved_model_revision']}")
        print(f"resolved_run_fingerprint: "
              f"{prov.get('resolved_run_fingerprint')}")
    print(f"attempted: {report['n_attempted']}/{report['expected_attempts']}"
          f" succeeded: {report['n_succeeded']} failed: {report['n_failed']}")


if __name__ == "__main__":
    main()
