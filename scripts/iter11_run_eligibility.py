#!/usr/bin/env python3
"""Iteration 11.5 — the eligibility replay and its signed report.

For one registry ``--model-key`` this replays the PRE-REGISTERED 12-family
selection through all six frozen variants (72 generations), then decides
TECHNICAL eligibility: can this target represent the same semantic role and
image structure, and produce complete, non-truncated, deterministic
responses for every variant?

Eligibility is never performance-based. Nothing here reads a response for
its content, and no candidate-target information entered the selection of
families — see ``causal_mllm/replay/selection.py``.

Writes:
  outputs/iteration_11/eligibility/generations/<model_key>/<run_id>/
      replay_outputs.jsonl   replay_failures.jsonl   replay_report.json
  outputs/iteration_11/eligibility/<model_key>/preflight_report.json

The report is the artifact ``enforce_confirmatory_protocol`` requires before
11.6 may generate, and it is validated against
``ELIGIBILITY_REQUIRED_FIELDS`` and ``ELIGIBILITY_REQUIRED_GATES`` — so this
script builds it by calling the SAME validator before writing it. A report
this script could not satisfy is not written as PASS.

Everything the frozen protocol fixes is enforced BEFORE any generation, by
``enforce_eligibility_protocol``: the raw-byte frozen panel hash, all 100
families carrying exactly the six variants, the uniform 1536 cap, greedy
decoding with thinking suppressed, a clean tree (untracked files count),
immutable model AND processor revisions agreeing with the lock, no
quantization, a live dependency environment matching the lock with no
third-party editable install, the canonical output root, no --overwrite, and
a family subset equal to the selection re-derived from frozen Iteration 10
evidence.

Usage::

    python3 scripts/iter11_run_eligibility.py --model-key qwen35_2b
    python3 scripts/iter11_run_eligibility.py --model-key qwen35_2b --resume
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES  # noqa: E402
from causal_mllm.data.io import read_jsonl  # noqa: E402
from causal_mllm.data.schemas import CausalFamily  # noqa: E402
from causal_mllm.replay.config import ReplayConfig  # noqa: E402
from causal_mllm.replay.confirmatory import (  # noqa: E402
    ELIGIBILITY_GENERATIONS_ROOT,
    ELIGIBILITY_N_ATTEMPTS,
    ELIGIBILITY_N_FAMILIES,
    ELIGIBILITY_OWN_OUTPUT_PREFIXES,
    eligibility_report_path,
    enforce_eligibility_protocol,
    protocol_sha256,
    validate_eligibility_report,
)
from causal_mllm.replay.environment import certify_environment  # noqa: E402
from causal_mllm.replay.errors import ReplayError  # noqa: E402
from causal_mllm.replay.registry import (  # noqa: E402
    dependency_lock_sha256,
    resolve_model,
)
from causal_mllm.replay.runner import (  # noqa: E402
    REPLAY_OUTPUTS_FILE,
    build_chat_messages,
    run_replay_stage,
)
from causal_mllm.replay.selection import derive_frozen_selection  # noqa: E402
from causal_mllm.seeds import code_tree_status, sha256_text  # noqa: E402

FROZEN_PANEL_DIR = REPO_ROOT / "outputs" / "scale_c" / "families_panel"
FROZEN_PROTOCOL = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"
ELIGIBILITY_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "eligibility"
DEFAULT_LOCK = REPO_ROOT / "outputs" / "iteration_11" / "preflight" \
    / "resolved_models.lock.yaml"

#: The frozen uniform cap. Read from the protocol below, never from here —
#: this constant only exists so the run id can name it.
FROZEN_CAP = 1536

#: Greedy determinism observations per cell, counting the journaled
#: response as the first. The eligibility gate requires at least two: one
#: observation cannot show a difference.
DETERMINISM_OBSERVATIONS = 2


# --------------------------------------------------------------------- #
# Evidence -> gate
# --------------------------------------------------------------------- #
def panel_canonical_queries(panel_path: Path) -> dict:
    """``{family_id: canonical terminal query text}`` from the frozen panel.

    The CANONICAL (harmonized) q* is what every variant carries as its
    terminal message, not the skeleton's original terminal field — the same
    definition ``scripts/scale_c_replay_checks.py`` uses, so the two checks
    cannot disagree about what the invariant is.
    """
    canonical: dict = {}
    for line in panel_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        family = json.loads(line)
        harmonization = (family.get("validation") or {}).get(
            "terminal_harmonization") or {}
        text = harmonization.get("canonical_q")
        if not text:
            query = family["terminal_query"]
            text = query["text"] if isinstance(query, dict) else query
        canonical[family["family_id"]] = text
    return canonical


def terminal_query_evidence(records: list[dict], canonical: dict) -> dict:
    """One terminal query per family, identical across all six variants.

    This is what makes the six variants comparable WITHIN a family: if
    variant construction had altered the terminal message, any difference
    between variants would be confounded with a difference in the question.
    """
    by_family: dict = defaultdict(set)
    for record in records:
        by_family[record["family_id"]].add(record.get("terminal_sha256"))
    mismatched = []
    for family_id, hashes in sorted(by_family.items()):
        if len(hashes) != 1:
            mismatched.append(
                f"{family_id}: {len(hashes)} distinct terminal hashes across "
                f"variants")
            continue
        expected_text = canonical.get(family_id)
        if expected_text is None:
            mismatched.append(f"{family_id}: not in the frozen panel")
            continue
        expected = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
        if next(iter(hashes)) != expected:
            mismatched.append(
                f"{family_id}: terminal hash != the panel canonical q*")
    return {
        "passed": not mismatched,
        "n_families_checked": len(by_family),
        "n_mismatched": len(mismatched),
        "mismatched": mismatched[:20],
        "definition": ("sha256 of the terminal message text actually "
                       "replayed, required identical across all six variants "
                       "of a family and equal to sha256 of the panel's "
                       "canonical_q*"),
    }


def vision_path_evidence(records: list[dict]) -> dict:
    """The image path actually engaged on every image-bearing cell.

    A target that silently degraded a cross-modal prompt to text-only would
    still produce fluent, complete responses, and every superficial check
    would pass while the vision arm of the contrast measured nothing. An
    image-bearing cell reporting zero image tokens is that failure.
    """
    bearing = [r for r in records if (r.get("n_images") or 0) > 0]
    counts = [r.get("image_token_count") for r in bearing]
    unmeasured = [f"{r['family_id']}:{r['variant']}" for r in bearing
                  if r.get("image_token_count") is None]
    # The minimum over EVERY measured image-bearing cell, zeros included:
    # reporting the minimum over only the non-zero ones would understate the
    # very failure this gate exists to catch, and would disagree with
    # ``passed``.
    measured = [c for c in counts if c is not None]
    zero = [f"{r['family_id']}:{r['variant']}" for r in bearing
            if r.get("image_token_count") == 0]
    return {
        "passed": bool(bearing) and not zero and not unmeasured,
        "n_image_bearing_cells": len(bearing),
        "min_image_token_count": min(measured) if measured else 0,
        "max_image_token_count": max(measured) if measured else 0,
        "n_cells_with_zero_image_tokens": len(zero),
        "cells_with_zero_image_tokens": zero[:20],
        "n_cells_without_a_measurement": len(unmeasured),
        "variants_observed": sorted({r["variant"] for r in bearing}),
    }


def truncation_evidence(records: list[dict]) -> dict:
    """Truncation overall and per variant.

    ANY truncation is a protocol-level STOP rather than an eligibility
    warning: raising the cap would require a uniform five-model replay
    including the frozen 9B reference, so a target that cannot finish
    inside 1536 tokens is not eligible on these terms.
    """
    by_variant: dict = {}
    for variant in ALL_VARIANT_NAMES:
        cells = [r for r in records if r["variant"] == variant]
        truncated = [r for r in cells if r.get("hit_max_new_tokens") is True]
        by_variant[variant] = {
            "n": len(cells),
            "n_truncated": len(truncated),
            "truncation_rate": (len(truncated) / len(cells) if cells else 0.0),
            "truncated_cells": [f"{r['family_id']}:{r['variant']}"
                                for r in truncated][:20],
            "max_output_tokens": max(
                (r["output_token_count"] for r in cells
                 if r.get("output_token_count") is not None), default=None),
        }
    rates = [v["truncation_rate"] for v in by_variant.values()]
    n_truncated = sum(v["n_truncated"] for v in by_variant.values())
    return {
        "passed": n_truncated == 0,
        "n_truncated": n_truncated,
        "truncation_rate": (n_truncated / len(records) if records else 0.0),
        # The largest per-variant rate difference. Zero when nothing
        # truncated; the value the 11.6 completion gate bounds at 0.05.
        "max_variant_spread": (max(rates) - min(rates)) if rates else 0.0,
        "by_variant": by_variant,
    }


def determinism_evidence(adapter, families: list, config: ReplayConfig,
                         records: list[dict],
                         observations: int) -> dict:
    """Re-generate every cell and compare against the journaled response.

    The journaled response counts as the first observation, so
    ``observations=2`` costs one extra generation per cell. Greedy decoding
    must be byte-identical: a target whose output moves between identical
    requests would make every downstream contrast noise.
    """
    by_id = {f.family_id: f for f in families}
    per_cell = []
    unstable = []
    for record in sorted(records, key=lambda r: (r["family_id"], r["variant"])):
        family = by_id[record["family_id"]]
        chat = build_chat_messages(family, record["variant"], config)
        hashes = {sha256_text(record["response"] or "")}
        for _ in range(max(0, observations - 1)):
            result = adapter.generate(chat)
            hashes.add(sha256_text(result["response"]))
        entry = {
            "family_id": record["family_id"],
            "variant": record["variant"],
            "n_observations": observations,
            "n_distinct_responses": len(hashes),
            "deterministic": len(hashes) == 1,
        }
        per_cell.append(entry)
        if len(hashes) != 1:
            unstable.append(f"{record['family_id']}:{record['variant']}")
    return {
        "passed": bool(per_cell) and not unstable,
        "n_repeats": observations,
        # The most distinct responses seen for ANY one cell: 1 means every
        # cell was byte-stable across its observations.
        "n_distinct_responses": max(
            (c["n_distinct_responses"] for c in per_cell), default=0),
        "n_cells_repeated": len(per_cell),
        "n_unstable_cells": len(unstable),
        "unstable_cells": unstable[:20],
        "per_cell": per_cell,
        "comparison": ("sha256 of the response text, against the response "
                       "journaled by the eligibility run itself"),
    }


# --------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------- #
def build_report(*, spec, config, run_report, run_dir, gate, environment,
                 selection, records, protocol_path, lock_path,
                 determinism, device) -> dict:
    """The 11.5 report, assembled from measured evidence only.

    Every gate is computed from the run's own records; none is asserted.
    ``status`` is derived from the gates and from the run's failure count,
    so a report cannot claim PASS while its own evidence says otherwise.
    """
    canonical = panel_canonical_queries(
        FROZEN_PANEL_DIR / "validated_families.jsonl")
    truncation = truncation_evidence(records)
    vision = vision_path_evidence(records)
    terminal = terminal_query_evidence(records, canonical)
    provenance = run_report["provenance"]
    gates = {
        "generations_complete": {
            "passed": (run_report["n_attempted"] == ELIGIBILITY_N_ATTEMPTS
                       and run_report["n_succeeded"] == ELIGIBILITY_N_ATTEMPTS
                       and run_report["n_failed"] == 0),
            "n_attempts": run_report["n_attempted"],
            "n_succeeded": run_report["n_succeeded"],
            "n_failed": run_report["n_failed"],
            "n_failure_attempts": run_report["n_failure_attempts"],
            "failed_cells": run_report["failed_cells"][:20],
        },
        "truncation_reviewed": truncation,
        "vision_path_engaged": vision,
        "terminal_query_invariant": terminal,
        "revision_pinned": {
            "passed": (
                provenance["resolved_model_revision"] == spec.revision
                and provenance["processor_revision"]
                == _locked_processor_revision(spec.model_key, lock_path)),
            "model_revision": provenance["resolved_model_revision"],
            "processor_revision": provenance["processor_revision"],
            "revision_pinned": provenance["revision_pinned"],
        },
        "determinism": determinism,
    }
    # Computed BEFORE the report is written: writing it is what makes the
    # tree dirty under this stage's own output prefix, and the value has to
    # describe the tree the run executed in.
    tree = code_tree_status(exclude_prefixes=ELIGIBILITY_OWN_OUTPUT_PREFIXES)
    problems = [f"gate {name!r} did not pass"
                for name, entry in sorted(gates.items())
                if entry.get("passed") is not True]
    problems.extend(f"environment: {p}" for p in environment["problems"])
    problems.extend(f"run: {p}" for p in run_report.get("missing_variants"))
    eligible = not problems
    return {
        "iteration": "11",
        "stage": "eligibility_preflight",
        "substage": "11.5",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if eligible else "FAIL",
        "eligible": eligible,
        "model_key": spec.model_key,
        "model_id": spec.model_id,
        "adapter": spec.adapter,
        "model_revision": provenance["resolved_model_revision"],
        "processor_revision": provenance["processor_revision"],
        "code_commit": provenance["code_commit"],
        # The reconstruction-relevant answer (modified OR untracked, minus
        # cache paths and this stage's own output tree), not the
        # tracked-only one: a report claiming a clean tree while an
        # untracked module changed what executed is the defect the
        # confirmatory gate now refuses.
        #
        # Both answers are recorded because they answer different questions.
        # The gate already established a whole-tree clean state at LAUNCH, so
        # by the time this report is written the process has imported its code
        # and code_commit pins it — a file appearing under outputs/ cannot
        # retroactively change what was generated, while a file appearing under
        # src/ can, because Python imports lazily and the determinism pass runs
        # after the generations. git_dirty_code_paths is the subset that can
        # still invalidate this run, and it is what the report is failed on;
        # the whole-tree answer stays recorded so the non-fatal changes are
        # visible rather than silently forgiven.
        "git_dirty": tree["dirty"],
        "git_dirty_paths": tree["dirty_paths"],
        "git_dirty_code_paths": tree["code_dirty_paths"],
        "git_untracked_paths": tree["untracked_paths"],
        "git_dirty_excluded_own_outputs": [
            p for p in tree["excluded_own_outputs"]],
        "protocol_sha256": protocol_sha256(protocol_path),
        "protocol_path": str(protocol_path),
        "dependency_lock_sha256": dependency_lock_sha256(lock_path),
        "selected_family_ids": list(selection["selected_family_ids"]),
        "selected_families_sha256": selection["selected_families_sha256"],
        "n_selected_families": selection["n_selected_families"],
        "selection": selection,
        "variants": list(ALL_VARIANT_NAMES),
        "n_expected_attempts": ELIGIBILITY_N_ATTEMPTS,
        "n_attempts": run_report["n_attempted"],
        "n_succeeded": run_report["n_succeeded"],
        "n_failed": run_report["n_failed"],
        "n_failure_attempts": run_report["n_failure_attempts"],
        "failed_cells": run_report["failed_cells"],
        "missing_variants": run_report["missing_variants"],
        "truncation_by_variant": {
            variant: {
                "n": entry["n"],
                "n_truncated": entry["n_truncated"],
                "truncation_rate": entry["truncation_rate"],
            } for variant, entry in truncation["by_variant"].items()},
        "truncation": truncation,
        "gates": gates,
        "run_id": run_report["run_id"],
        "run_dir": str(run_dir),
        "run_report": str(Path(run_dir) / "replay_report.json"),
        "resolved_run_fingerprint": provenance["resolved_run_fingerprint"],
        "device": device,
        "generation_config": config.generation_settings(),
        "environment": environment["values"],
        "eligibility_gate": gate,
        "token_stats": run_report["token_stats"],
        "problems": problems,
    }


def _locked_processor_revision(model_key: str, lock_path: Path) -> str | None:
    """The processor revision the lock resolved for this target."""
    import yaml
    document = yaml.safe_load(Path(lock_path).read_text(encoding="utf-8")) or {}
    entry = (document.get("models") or {}).get(model_key)
    return entry.get("processor_revision") if isinstance(entry, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--device", default="cuda:3",
                        help="GPU slot (standing instruction: cuda:3). The "
                             "slot is NOT part of the run fingerprint; the "
                             "hardware class is.")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--run-id", default=None,
                        help="Override the run id. The default is stable per "
                             "model_key so --resume continues the same run.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Continue an interrupted eligibility run.")
    parser.add_argument("--observations", type=int,
                        default=DETERMINISM_OBSERVATIONS,
                        help="Determinism observations per cell, counting "
                             "the journaled response as the first.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Enforce the gate and report what would run, "
                             "without loading a checkpoint.")
    args = parser.parse_args()

    # Every relative path this stage and the gate resolve is repo-root
    # relative, so the launch directory must not be able to change what is
    # read or where evidence lands.
    os.chdir(REPO_ROOT)

    if not FROZEN_PROTOCOL.exists():
        print(f"FAIL: frozen protocol not found: {FROZEN_PROTOCOL}",
              file=sys.stderr)
        return 2
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))

    # Environment first: unlike a dirty tree, an environment holding a
    # third-party editable install cannot produce certifiable evidence at
    # all, so there is nothing to gain by loading a checkpoint.
    values, environment_problems = certify_environment(protocol)
    environment = {"values": values, "problems": environment_problems}
    if environment_problems:
        for problem in environment_problems:
            print(f"FAIL {args.model_key}: {problem}", file=sys.stderr)
        return 2

    selection = derive_frozen_selection(REPO_ROOT)
    family_ids = list(selection["selected_family_ids"])
    print(f"pre-registered selection: {selection['n_selected_families']} "
          f"families, sha256={selection['selected_families_sha256']}")

    lock_path = Path(args.lock)
    spec = resolve_model(args.model_key, confirmatory=True,
                         lock_path=lock_path)
    cap = protocol["uniform_cap_rule"]["initial_cap"]
    config = ReplayConfig(
        model_name=spec.model_id,
        model_revision=spec.revision,
        max_new_tokens=cap,
        device=args.device,
        enable_thinking=spec.thinking_mode)
    output_root = f"{ELIGIBILITY_GENERATIONS_ROOT}/{spec.model_key}"
    run_id = args.run_id or (
        f"eligibility-{ELIGIBILITY_N_FAMILIES}f-t{cap}-{spec.model_key}")

    gate = enforce_eligibility_protocol(
        input_dir=FROZEN_PANEL_DIR,
        config=config,
        model_spec=spec,
        family_ids=family_ids,
        max_families=None,
        output_root=output_root,
        lock_path=lock_path,
        protocol_path=FROZEN_PROTOCOL,
        expected_selection=selection,
        repo_root=REPO_ROOT,
    )
    print(f"eligibility gate: PASS ({len(gate['checks'])} checks, "
          f"panel={gate['checks']['panel_sha256'][:16]}…, cap="
          f"{gate['checks']['max_new_tokens']}, "
          f"code_commit={str(gate['checks']['code_commit'])[:12]}, "
          f"selection={gate['checks']['expected_selection_sha256'][:16]}…)")
    print(f"would replay {len(family_ids)} families x "
          f"{len(ALL_VARIANT_NAMES)} variants = {ELIGIBILITY_N_ATTEMPTS} "
          f"generations on {args.device}, plus "
          f"{ELIGIBILITY_N_ATTEMPTS * max(0, args.observations - 1)} "
          f"determinism re-generation(s)")
    if args.dry_run:
        print("--dry-run: no checkpoint loaded, nothing generated")
        return 0

    from causal_mllm.replay.adapters import build_adapter
    adapter = build_adapter(spec, config, device=args.device)
    adapter.load()

    run_report = run_replay_stage(
        FROZEN_PANEL_DIR, output_root, config=config, backend=adapter,
        family_ids=family_ids, run_id=run_id, overwrite=False,
        model_spec=spec, resume=args.resume, lock_path=lock_path,
        gate_evidence=gate,
        # This stage's own tree, which includes the report it is about to
        # write and the generations of any sibling target already run. The
        # confirmatory stage uses a narrower list, because there this
        # stage's report is an INPUT rather than a product.
        own_output_prefixes=ELIGIBILITY_OWN_OUTPUT_PREFIXES)
    run_dir = Path(output_root) / run_report["run_id"]
    records = read_jsonl(run_dir / REPLAY_OUTPUTS_FILE)
    print(f"run {run_report['run_id']}: {run_report['n_succeeded']}/"
          f"{run_report['n_attempted']} succeeded, "
          f"{run_report['n_failed']} failed -> {run_dir}")

    families = [CausalFamily.from_dict(rec)
                for rec in read_jsonl(
                    FROZEN_PANEL_DIR / "validated_families.jsonl")
                if rec["family_id"] in set(family_ids)]
    print(f"determinism: re-generating {len(records)} cell(s) "
          f"({args.observations} observations each)")
    determinism = determinism_evidence(adapter, families, config, records,
                                       args.observations)
    print(f"determinism: {determinism['n_unstable_cells']} unstable cell(s) "
          f"of {determinism['n_cells_repeated']}")

    report = build_report(
        spec=spec, config=config, run_report=run_report, run_dir=run_dir,
        gate=gate, environment=environment, selection=selection,
        records=records, protocol_path=FROZEN_PROTOCOL,
        lock_path=lock_path, determinism=determinism, device=args.device)

    # Validate against the SAME contract the confirmatory gate will apply,
    # before writing: a report this script could not satisfy must not be
    # filed as PASS and discovered to be invalid at 11.6 launch time.
    violations = validate_eligibility_report(
        report, model_spec=spec,
        expected_protocol_sha=protocol_sha256(FROZEN_PROTOCOL),
        expected_lock_sha=dependency_lock_sha256(lock_path),
        expected_processor_revision=_locked_processor_revision(
            spec.model_key, lock_path),
        expected_family_ids=family_ids,
        expected_selection_sha256=selection["selected_families_sha256"],
        panel_family_ids=set(panel_canonical_queries(
            FROZEN_PANEL_DIR / "validated_families.jsonl")))
    report["self_validation"] = {
        "n_violations": len(violations), "violations": violations}
    if violations:
        report["status"] = "FAIL"
        report["eligible"] = False
        report["problems"].extend(f"self-validation: {v}" for v in violations)

    path = eligibility_report_path(spec.model_key, ELIGIBILITY_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"status: {report['status']}  eligible={report['eligible']}")
    for name in sorted(report["gates"]):
        entry = report["gates"][name]
        print(f"  gate {name}: passed={entry.get('passed')}")
    for problem in report["problems"]:
        print(f"  PROBLEM {problem}", file=sys.stderr)
    print(f"wrote {path}")
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReplayError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(2)
