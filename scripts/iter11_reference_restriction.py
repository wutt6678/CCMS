#!/usr/bin/env python3
"""Iteration 11.8: the frozen reference, restricted to the families in play.

The cross-model comparison is between the four Iteration 11 arms and the frozen
Qwen3.5-9B reference. Those have to be estimated over the SAME families, and
they are not by default: aliyun's input moderation refuses 2 of the 600 cells
for judge A, the exclusion drops them from every arm, and the frozen estimator
needs all six variants of a family to produce any of its five estimands -- so
family CMST_795308 goes whole and the arms analyse 99 families while the
reference was published over 100. See
``outputs/iteration_11/diagnostics/judge_moderation/``.

This script recomputes the reference over whatever family set the arms actually
analysed. It reads the sealed reference's own per-cell scores and re-runs the
frozen estimator, so nothing under ``outputs/scale_c/`` is touched or rewritten.

WHY IT FIRST PROVES IT CAN REPRODUCE THE PUBLISHED NUMBERS
A restricted estimate is only meaningful if the unrestricted one is the frozen
one. So before reporting anything over a subset, this recomputes all five
estimands over ALL of the reference's families and compares them to the CI
published in the sealed report. If that fails, the estimator or the inputs are
not the frozen ones, and a number computed over 99 families would be a
reconstruction wearing the reference's name. The check fails closed.

The comparison uses a tolerance rather than equality. The two agree to about
3e-16 -- the last bits of a floating-point sum -- because the sealed report was
written by a different interpreter than the one reading it, and summation order
is not guaranteed across versions. A tolerance of 1e-12 is twelve orders of
magnitude above that noise and still nine orders below any difference that
would mean something.

Usage:
    python3 scripts/iter11_reference_restriction.py
    python3 scripts/iter11_reference_restriction.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.bootstrap import paired_bootstrap_ci  # noqa: E402
from causal_mllm.evaluation.estimands import (  # noqa: E402
    aggregate_estimands,
    compute_family_estimands,
    incomplete_families,
)

#: The sealed Iteration 10 / Scale-C reference. Read only, never written.
REFERENCE_ROOT = REPO_ROOT / "outputs" / "scale_c" / "llm_judge_artifacts"
REFERENCE_REPORT = (REFERENCE_ROOT / "evaluation_results"
                    / "evaluation_report.json")
REFERENCE_CELLS = (REFERENCE_ROOT / "evaluation_results"
                   / "evaluation_outputs.jsonl")

#: Where the Iteration 11 arms' own evaluation reports land, one per target.
JUDGE_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "judge"
TARGETS = ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm")

OUT_PATH = (REPO_ROOT / "outputs" / "iteration_11" / "analysis"
            / "cross_model" / "reference_restriction.json")

ESTIMANDS = ("Delta_T", "Delta_V", "Delta_TV", "order_effect",
             "history_effect")

#: Agreement tolerance for the reproduction self-check. See the module
#: docstring: the residual is floating-point summation order, about 3e-16.
TOLERANCE = 1e-12


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rel(path: Path) -> str:
    """The repo-relative form when there is one, else the absolute path.

    ``relative_to`` raises rather than returning something usable, and a path
    outside the repo is a legitimate thing to be handed -- a test fixture, or
    a reference relocated for a re-run. Recording the absolute path in that
    case keeps the artifact honest about what it read instead of crashing.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_reference() -> tuple[dict, list[dict], dict]:
    """The sealed report, its per-cell scores, and its frozen eval config.

    The config is read out of the sealed report rather than defaulted here, so
    the recomputation uses the theta, bootstrap count, CI level and seed the
    reference was actually published with.
    """
    for path in (REFERENCE_REPORT, REFERENCE_CELLS):
        if not path.exists():
            raise SystemExit(f"FATAL: sealed reference artifact absent: {path}")
    report = _load_json(REFERENCE_REPORT)
    cells = _load_jsonl(REFERENCE_CELLS)
    config = report.get("config") or {}
    for key in ("n_bootstrap", "ci_level", "seed", "primary_outcome"):
        if key not in config:
            raise SystemExit(
                f"FATAL: the sealed report's config carries no {key!r}, so "
                f"the frozen estimator settings cannot be read from it")
    return report, cells, config


def estimate(family_estimands: dict, config: dict) -> dict:
    """Per-estimand mean/std/n plus the paired bootstrap CI, frozen settings."""
    aggregated = aggregate_estimands(family_estimands)
    ci = paired_bootstrap_ci(
        family_estimands,
        n_bootstrap=int(config["n_bootstrap"]),
        ci_level=float(config["ci_level"]),
        seed=int(config["seed"]),
    )
    out = {}
    for name in ESTIMANDS:
        agg = aggregated["estimands"][name]
        out[name] = {
            "mean": agg["mean"],
            "std": agg["std"],
            "n": agg["n"],
            "bootstrap_mean": ci[name]["mean"],
            "ci_lower": ci[name]["CI_lower"],
            "ci_upper": ci[name]["CI_upper"],
        }
    out["n_families"] = aggregated["n_families"]
    return out


def check_reproduction(recomputed: dict, published: dict) -> list[str]:
    """Does the unrestricted recomputation match the sealed published CI?

    Compares all five estimands, not just Delta_TV: the point is to establish
    that the whole estimator path is the frozen one, and a single estimand
    could agree by coincidence.
    """
    issues = []
    bootstrap_ci = (published.get("estimands") or {}).get("bootstrap_ci") or {}
    aggregated = (published.get("estimands") or {}).get("aggregated") or {}
    for name in ESTIMANDS:
        got, want = recomputed.get(name), bootstrap_ci.get(name)
        if want is None:
            issues.append(f"the sealed report publishes no CI for {name}")
            continue
        for label, key in (("bootstrap_mean", "mean"),
                           ("ci_lower", "CI_lower"),
                           ("ci_upper", "CI_upper")):
            diff = abs(got[label] - want[key])
            if diff > TOLERANCE:
                issues.append(
                    f"{name}.{label} recomputes to {got[label]!r} but the "
                    f"sealed report publishes {want[key]!r} (difference "
                    f"{diff:.3g} exceeds the {TOLERANCE:g} tolerance), so "
                    f"this is not the frozen estimator over the frozen inputs")
        published_agg = aggregated.get(name) or {}
        if "mean" in published_agg:
            diff = abs(recomputed[name]["mean"] - published_agg["mean"])
            if diff > TOLERANCE:
                issues.append(
                    f"{name}.mean recomputes to {recomputed[name]['mean']!r} "
                    f"but the sealed report aggregates {published_agg['mean']!r}"
                    f" (difference {diff:.3g})")
        if "n" in published_agg and published_agg["n"] != recomputed[name]["n"]:
            issues.append(
                f"{name} covers {recomputed[name]['n']} families but the "
                f"sealed report covers {published_agg['n']}")
    return issues


def analysed_family_set() -> tuple[set | None, dict, list[str]]:
    """The families every Iteration 11 arm actually analysed.

    Read from each target's own ``panel_restriction`` rather than restated
    here, and INTERSECTED across targets. The intersection matters because the
    provider's moderation verdict is not stable over time and the four arms
    reach a given cell minutes apart: one target could lose a family another
    kept, and comparing a 99-family arm with a 98-family arm is not a
    cross-model comparison over one panel.

    Returns ``(family_ids or None, per_target, issues)``. ``None`` means the
    arms have not been evaluated yet, which is a state to report rather than
    an error to raise.
    """
    per_target, issues, sets = {}, [], []
    for key in TARGETS:
        report_path = (JUDGE_ROOT / key / "evaluation_results"
                       / "evaluation_report.json")
        if not report_path.exists():
            issues.append(f"{key}: no evaluation_report.json yet "
                          f"(phase 2 has not finalized this target)")
            per_target[key] = {"status": "pending"}
            continue
        report = _load_json(report_path)
        restriction = report.get("panel_restriction")
        cells_path = JUDGE_ROOT / key / "llm_labels_adjudicated.json"
        labels_sha = (hashlib.sha256(cells_path.read_bytes()).hexdigest()
                      if cells_path.exists() else None)
        if restriction is None:
            # No restriction recorded means the arm labelled every cell, so it
            # analysed every family the reference has. That is a legitimate
            # outcome -- moderation may not refuse the cell on a later run --
            # and it has to be distinguishable from "not finalized yet".
            n_families = (report.get("estimands") or {}).get("n_families")
            per_target[key] = {
                "status": "unrestricted",
                "n_families_analysed": n_families,
                "labels_sha256": labels_sha,
            }
            sets.append(None)
            continue
        dropped = set(restriction.get("families_dropped_incomplete") or [])
        n_analysed = restriction.get("n_families_analysed")
        per_target[key] = {
            "status": "restricted",
            "n_families_analysed": n_analysed,
            "n_families_in_panel": restriction.get("n_families_in_panel"),
            "n_records_analysed": restriction.get("n_records_analysed"),
            "excluded_cells": restriction.get("excluded_cells"),
            "families_dropped_incomplete": sorted(dropped),
            "labels_sha256": labels_sha,
        }
        sets.append(dropped)

    if issues:
        # Either some target has not been evaluated yet, or the ones that have
        # disagree about the panel. Both are states to report, and in the
        # second case the union below is what makes the comparison safe.
        return None, per_target, issues

    dropped_union: set = set()
    for dropped in sets:
        if dropped:
            dropped_union |= dropped
    # The intersection of the ANALYSED sets is the complement of the UNION of
    # the dropped ones: a family any arm lost has to be lost by all of them.
    divergent = {key: sorted(entry.get("families_dropped_incomplete") or [])
                 for key, entry in per_target.items()
                 if entry.get("status") == "restricted"}
    distinct = {tuple(v) for v in divergent.values()}
    if len(distinct) > 1:
        issues.append(
            f"the arms did not lose the same families: {divergent}; the "
            f"reference is restricted to the union so that every arm is "
            f"compared over families all of them analysed")
    return dropped_union, per_target, issues


def build() -> dict:
    report, cells, config = load_reference()

    all_families = compute_family_estimands(cells)
    unrestricted = estimate(all_families, config)
    reproduction_issues = check_reproduction(
        unrestricted, report)
    if reproduction_issues:
        for issue in reproduction_issues:
            print(f"  FAIL {issue}", file=sys.stderr)
        raise SystemExit(
            f"FATAL: the unrestricted recomputation does not reproduce the "
            f"sealed reference ({len(reproduction_issues)} difference(s)); "
            f"a restricted estimate derived from it would mean nothing")

    dropped, per_target, target_issues = analysed_family_set()

    restricted = None
    if dropped is not None:
        kept = {fid: est for fid, est in all_families.items()
                if fid not in dropped}
        if not kept:
            raise SystemExit("FATAL: restricting left no families")
        still_incomplete = incomplete_families(
            [c for c in cells if c["family_id"] in kept])
        if still_incomplete:
            raise SystemExit(
                f"FATAL: after restriction these families are still "
                f"incomplete: {still_incomplete}")
        restricted = estimate(kept, config)
        restricted["families_dropped"] = sorted(dropped)

    return {
        "produced_by": "scripts/iter11_reference_restriction.py",
        "reference": {
            "report": _rel(REFERENCE_REPORT),
            "cells": _rel(REFERENCE_CELLS),
            "run_id": (report.get("panel_gate") or {}).get(
                "panel", {}).get("run_id"),
            "config": config,
            "labels_sha256": (report.get("judge_provenance") or {}).get(
                "labels_sha256"),
        },
        "reproduction_check": {
            "status": "PASS",
            "tolerance": TOLERANCE,
            "n_estimands_compared": len(ESTIMANDS),
            "note": (
                "the unrestricted recomputation reproduces every published "
                "estimand and CI bound of the sealed reference within "
                f"{TOLERANCE:g}; the residual is floating-point summation "
                "order across interpreter versions, about 3e-16, not a "
                "modelling difference"),
        },
        "unrestricted": unrestricted,
        "targets": per_target,
        "status": ("restricted" if restricted is not None
                   else "pending_target_reports"),
        "target_issues": target_issues,
        "restricted": restricted,
        "rule": (
            "the reference is restricted to the families EVERY arm analysed, "
            "which is the complement of the union of the families the arms "
            "lost, so the comparison is over one panel"),
    }


def verify() -> int:
    if not OUT_PATH.exists():
        print(f"VERIFY FAIL: no artifact at {OUT_PATH}", file=sys.stderr)
        return 1
    committed = _load_json(OUT_PATH)
    fresh = build()
    if fresh != committed:
        print("VERIFY FAIL: a fresh derivation differs from the committed "
              "artifact", file=sys.stderr)
        for key in sorted(set(fresh) | set(committed)):
            if fresh.get(key) != committed.get(key):
                print(f"  differs: {key}", file=sys.stderr)
        return 1
    print(f"VERIFY PASS: {OUT_PATH.relative_to(REPO_ROOT)} matches a fresh "
          f"derivation")
    _print_summary(fresh)
    return 0


def _print_summary(artifact: dict) -> None:
    unre = artifact["unrestricted"]
    print(f"\nreference          {artifact['reference']['run_id']}")
    print(f"reproduction check {artifact['reproduction_check']['status']} "
          f"over {artifact['reproduction_check']['n_estimands_compared']} "
          f"estimands, tolerance "
          f"{artifact['reproduction_check']['tolerance']:g}")
    print(f"\n  unrestricted  n={unre['n_families']}  "
          f"Delta_TV mean={unre['Delta_TV']['mean']:.6f}  "
          f"CI [{unre['Delta_TV']['ci_lower']:.4f}, "
          f"{unre['Delta_TV']['ci_upper']:.4f}]")
    for key in TARGETS:
        entry = artifact["targets"].get(key) or {}
        print(f"  {key:16s} {entry.get('status')}  "
              f"n={entry.get('n_families_analysed')}")
    restricted = artifact.get("restricted")
    if restricted is None:
        print(f"\nstatus {artifact['status']}")
        for issue in artifact["target_issues"]:
            print(f"  - {issue}")
        return
    print(f"  restricted    n={restricted['n_families']}  "
          f"Delta_TV mean={restricted['Delta_TV']['mean']:.6f}  "
          f"CI [{restricted['Delta_TV']['ci_lower']:.4f}, "
          f"{restricted['Delta_TV']['ci_upper']:.4f}]")
    print(f"  dropped       {restricted['families_dropped']}")
    delta = restricted["Delta_TV"]["mean"] - unre["Delta_TV"]["mean"]
    print(f"\n  restriction moves the reference Delta_TV by {delta:+.6f}")
    for issue in artifact["target_issues"]:
        print(f"  NOTE {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="re-derive and compare to the committed artifact; "
                             "write nothing")
    parser.add_argument("--json-out", default=str(OUT_PATH))
    args = parser.parse_args()

    if args.verify:
        return verify()

    artifact = build()
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    _print_summary(artifact)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
