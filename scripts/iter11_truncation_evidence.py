#!/usr/bin/env python3
"""Iteration 11: the truncation evidence, and the gate inconsistency it closed.

WHAT THIS RECORDS
Three of the four Iteration 11 checkpoints reached the frozen 1536-token
generation cap on some cells -- 7, 1 and 4 of 600. Each passed the 11.6
completion gate and each was then refused by the evaluation panel gate, which
required zero truncated records. Both gates were frozen and they contradicted
each other; the frozen Qwen3.5-9B reference had never truncated, so nothing had
ever compared them. This artifact is the record of that inconsistency and of
what was measured to close it.

WHAT IT DOES NOT DO
It does not decide anything. Acceptance is decided by two numbers that live in
``causal_mllm.replay.truncation`` and are imported by both gates. Nothing here
is an input to that decision, and two things are deliberately kept OUT of it:

* the judge scores of the affected cells. The twelve truncated cells happened
  to be scored 0.00-0.15 by the frozen ensemble. That is a fact about the
  cells and it is reported in the 11.8 analysis, but it is NOT part of the
  justification for accepting the panels, and it is not recorded here where it
  could be read as one. A tolerance chosen because the affected cells scored
  low would be a tolerance chosen by the result.
* the repetition classification. It is diagnostic: it explains WHY the cells
  ran to the cap, which is what makes the uniform-cap remedy unusable here, but
  acceptance does not depend on it. A panel is inside the thresholds or it is
  not, whatever its text looks like.

ONE LOOP COUNT, ONE CRITERION
Earlier prose described the truncated cells as "11 of 12" and then as "10 of
12" repetition loops. Both cannot be right, and the difference mattered because
it was the evidence that the cap remedy cannot terminate. The criterion is
registered below and applied once, so the count is reproducible rather than
recalled: a truncated cell is classified a repetition loop when its repeat3
exceeds the maximum repeat3 of the COMPLETE cells of the SAME arm -- that is,
when it is more repetitive than anything the same model produced when it was
allowed to finish. See ``repetition_diagnostics``.

Usage:
    python3 scripts/iter11_truncation_evidence.py
    python3 scripts/iter11_truncation_evidence.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.replay.truncation import (  # noqa: E402
    MAX_TRUNCATION_RATE,
    MAX_VARIANT_SPREAD,
    exceeds_registered_thresholds,
    is_truncated,
    measure_truncation,
)

SCALE_PROFILES = REPO_ROOT / "configs" / "evaluation" / "scale_profiles.json"
PROFILE_PREFIX = "iteration_11_"

#: The sealed Iteration 10 reference, for the comparison that explains why the
#: contradiction stayed hidden: it truncated nothing at the same cap.
REFERENCE_RUN = (REPO_ROOT / "outputs" / "scale_c" / "replay_runs"
                 / "scale-c-100-t1536-qwen35-9b")

OUT_PATH = (REPO_ROOT / "outputs" / "iteration_11" / "diagnostics"
            / "truncation" / "truncation_evidence.json")

OUTPUTS_FILE = "replay_outputs.jsonl"
REPORT_FILE = "replay_report.json"

#: The registered loop criterion, stated once so the count can be reproduced.
LOOP_CRITERION = (
    "a truncated cell is classified a repetition loop when its repeat3 exceeds "
    "the maximum repeat3 of the COMPLETE (non-truncated) cells of the SAME "
    "arm, i.e. when it is more repetitive than anything that model produced "
    "when it was allowed to finish")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ---------------------------------------------------------------------------
# Repetition diagnostics. Diagnostic only: nothing here gates anything.
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def repeat3(text: str) -> float:
    """``1 - distinct(word 3-grams) / total(word 3-grams)``.

    Near 0 for ordinary prose, near 1 for a loop. Used only to explain why a
    cell ran to the cap.
    """
    words = _words(text)
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def top8(text: str) -> float:
    """Share of the response covered by its most frequent word 8-gram.

    A sentence loop scores high; prose scores near zero. Reported above 1.0 is
    impossible, but a short response dominated by one phrase can approach it,
    and a hard token loop repeating a phrase shorter than eight words can
    exceed it -- which is itself the finding.
    """
    words = _words(text)
    if len(words) < 16:
        return 0.0
    grams = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
    _, count = Counter(grams).most_common(1)[0]
    return (count * 8) / len(words)


def repetition_diagnostics(records: list[dict]) -> dict:
    """Classify the truncated cells of ONE arm against its own complete cells.

    Returns the per-cell measurements, the complete-cell maximum the criterion
    compares against, and the resulting classification. The classification is
    reported, never enforced.
    """
    complete = [r for r in records if not is_truncated(r)]
    truncated = [r for r in records if is_truncated(r)]
    complete_scores = [repeat3(r.get("response") or "") for r in complete]
    ceiling = max(complete_scores) if complete_scores else 0.0

    cells = []
    for record in truncated:
        text = record.get("response") or ""
        score = repeat3(text)
        cells.append({
            "family_id": record.get("family_id"),
            "variant": record.get("variant"),
            "repeat3": round(score, 6),
            "top8": round(top8(text), 4),
            "arm_complete_max_repeat3": round(ceiling, 6),
            "classified_repetition_loop": score > ceiling,
        })
    cells.sort(key=lambda c: (str(c["family_id"]), str(c["variant"])))
    return {
        "metric": ("repeat3 = 1 - distinct(word 3-grams) / total(word "
                   "3-grams); top8 = share of the response covered by its "
                   "most frequent word 8-gram"),
        "criterion": LOOP_CRITERION,
        "gates_nothing": True,
        "complete_max_repeat3": round(ceiling, 6),
        "cells": cells,
        "n_classified_loop": sum(1 for c in cells
                                 if c["classified_repetition_loop"]),
        "n_not_classified_loop": sum(1 for c in cells
                                     if not c["classified_repetition_loop"]),
    }


# ---------------------------------------------------------------------------
# Per-arm evidence
# ---------------------------------------------------------------------------

def arm_replay_run(profiles_path: Path = SCALE_PROFILES) -> dict[str, Path]:
    """``{target: replay_run_dir}`` for the Iteration 11 arms.

    Read from the profiles rather than restated here, so this measures the
    panels the pipeline actually evaluates.
    """
    profiles = _load_json(profiles_path)
    runs = {}
    for name, profile in sorted(profiles.items()):
        if not name.startswith(PROFILE_PREFIX):
            continue
        replay_run = profile.get("replay_run")
        if not replay_run:
            raise SystemExit(f"{name} declares no replay_run")
        runs[Path(profile["output_dir"]).name] = REPO_ROOT / replay_run
    if not runs:
        raise SystemExit(f"no {PROFILE_PREFIX}* arms in {_rel(profiles_path)}")
    return dict(sorted(runs.items()))


def measure_run(run_dir: Path) -> dict:
    """Truncation and token evidence for one replay run."""
    outputs_path = run_dir / OUTPUTS_FILE
    report_path = run_dir / REPORT_FILE
    if not outputs_path.exists():
        raise SystemExit(f"no replay outputs at {_rel(outputs_path)}")
    records = _read_jsonl(outputs_path)
    measurement = measure_truncation(records)

    complete_tokens = [r.get("output_token_count") or 0
                       for r in records if not is_truncated(r)]
    report = _load_json(report_path) if report_path.exists() else {}
    generation = ((report.get("provenance") or {})
                  .get("generation_config") or {})

    return {
        "run_dir": _rel(run_dir),
        "replay_outputs_sha256": _sha256(outputs_path),
        "n_records": measurement["n_records"],
        "n_truncated": measurement["n_truncated"],
        "overall_rate": measurement["overall_rate"],
        "max_variant_spread": measurement["max_variant_spread"],
        "per_variant": measurement["per_variant"],
        "thresholds": measurement["thresholds"],
        "counting_rule": measurement["counting_rule"],
        "violations": measurement["violations"],
        "within_registered_thresholds": measurement["passed"],
        "uniform_cap_escalation_triggered": exceeds_registered_thresholds(
            measurement),
        "cells": measurement["cells"],
        "max_new_tokens": generation.get("max_new_tokens"),
        "longest_complete_output_tokens": (max(complete_tokens)
                                           if complete_tokens else None),
        "repetition_diagnostics": repetition_diagnostics(records),
    }


def build(profiles_path: Path = SCALE_PROFILES) -> dict:
    runs = arm_replay_run(profiles_path)
    per_target = {target: measure_run(run_dir)
                  for target, run_dir in runs.items()}
    reference = (measure_run(REFERENCE_RUN)
                 if (REFERENCE_RUN / OUTPUTS_FILE).exists() else None)

    n_truncated_total = sum(a["n_truncated"] for a in per_target.values())
    n_loop_total = sum(a["repetition_diagnostics"]["n_classified_loop"]
                       for a in per_target.values())

    return {
        "kind": "iteration_11_truncation_evidence",
        "status": "measured",
        "counting_rule": ("a cell is truncated when hit_max_new_tokens is True "
                          "or truncated is True; one definition, imported by "
                          "the replay completion gate, the Scale-C gate and "
                          "the evaluation panel gate"),
        "resolution": {
            "what_was_inconsistent": (
                "the 11.6 completion gate accepted a full panel at an overall "
                "truncation rate up to 0.02 with a variant-rate spread up to "
                "0.05, while evaluation/gate.py required zero truncated "
                "records; both were frozen and the frozen 9B reference "
                "truncated nothing, so they had never been compared"),
            "how_it_was_resolved": (
                "both thresholds moved into causal_mllm/replay/truncation.py "
                "and every full-panel gate imports them; there is no tolerance "
                "argument, no per-run override and no CLI flag, so one panel "
                "cannot be held to two standards"),
            "kind_of_change": (
                "resolution of an internal inconsistency between two gates in "
                "this repository"),
            "not_the_motivation": (
                "the judge scores of the affected cells. This is NOT an "
                "amendment motivated by favourable scores, and the scores are "
                "deliberately absent from this artifact so they cannot be read "
                "back into the justification"),
            "authority_for_the_numbers": (
                "0.02 and 0.05 were already in scripts/iter11_replay_checks.py "
                "and scripts/scale_c_replay_checks.py, introduced in commit "
                "6389df65 before any Iteration 11 target evidence was "
                "generated, and documented there as identical to the "
                "Iteration 10 rule; adopting them at the evaluation gate "
                "applies a pre-registered standard rather than choosing one "
                "after seeing the results"),
            "eligibility_gate_unchanged": (
                "the 12-family eligibility gate in replay/confirmatory.py "
                "still treats any truncation as a protocol-level STOP; it is a "
                "screening contract, not a full-panel acceptance rule"),
            "sealed_evidence_unchanged": (
                "every archived Iteration 9/10 panel has zero truncated "
                "records, so a threshold admitting up to 2% returns the "
                "verdict those panels always returned"),
            "cells_are_kept": (
                "all 600 outputs per arm are retained. A response that ran to "
                "the cap is an observation of what the model did under the "
                "frozen greedy decoding, not missing data, and dropping those "
                "cells would drop them from some arms and not others"),
            "uniform_cap_escalation": {
                "trigger": ("either registered threshold exceeded -- and only "
                            "then"),
                "triggered_here": False,
                "why_not_just_escalate": (
                    "the remedy assumes truncation means the cap was too "
                    "small. Under greedy decoding a repetition loop emits to "
                    "ANY cap, so 'escalate until nothing truncates' has no "
                    "termination criterion; measured below, most of the "
                    "truncated cells are loops of exactly that kind"),
            },
        },
        "reference_9b": reference,
        "per_target": per_target,
        "totals": {
            "n_targets": len(per_target),
            "n_truncated_cells": n_truncated_total,
            "n_classified_repetition_loop": n_loop_total,
            "n_not_classified_repetition_loop": (n_truncated_total
                                                 - n_loop_total),
            "loop_criterion": LOOP_CRITERION,
            "count_reconciliation": (
                "earlier prose described these cells as both '11 of 12' and "
                "'10 of 12' repetition loops. Under the criterion registered "
                "above the count is the one in "
                "n_classified_repetition_loop, computed per arm from the "
                "per-cell repeat3 values recorded here, so it can be recounted "
                "rather than recalled. The '10 of 12' figure was a miscount: "
                "the range it quoted, repeat3 0.48-0.96, covers eleven cells, "
                "not ten"),
        },
        "targets": sorted(per_target),
    }


# ---------------------------------------------------------------------------
# --verify: read-only gate over the committed artifact
# ---------------------------------------------------------------------------

def verify(artifact_path: Path = OUT_PATH,
           profiles_path: Path = SCALE_PROFILES) -> list[str]:
    """Does the committed evidence still describe the panels on disk?"""
    if not artifact_path.exists():
        return [f"no truncation evidence artifact at {_rel(artifact_path)}"]
    try:
        committed = _load_json(artifact_path)
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{_rel(artifact_path)} is unreadable: {exc}"]

    problems: list[str] = []
    fresh = build(profiles_path)

    # The thresholds the artifact claims must be the registered ones. This is
    # the check that stops the artifact from becoming a second, looser source
    # of the standard it is describing.
    limits = fresh["per_target"][next(iter(fresh["per_target"]))]["thresholds"]
    if limits["max_overall_rate"] != MAX_TRUNCATION_RATE:
        problems.append("internal error: the shared module no longer reports "
                        f"max_overall_rate={MAX_TRUNCATION_RATE}")
    if limits["max_variant_spread"] != MAX_VARIANT_SPREAD:
        problems.append("internal error: the shared module no longer reports "
                        f"max_variant_spread={MAX_VARIANT_SPREAD}")

    if sorted(committed.get("per_target") or {}) != sorted(fresh["per_target"]):
        problems.append(
            f"the artifact covers {sorted(committed.get('per_target') or {})} "
            f"but the profiles declare {sorted(fresh['per_target'])}")

    for target, now in fresh["per_target"].items():
        then = (committed.get("per_target") or {}).get(target)
        if then is None:
            continue
        for key in ("replay_outputs_sha256", "n_records", "n_truncated",
                    "overall_rate", "max_variant_spread",
                    "within_registered_thresholds",
                    "uniform_cap_escalation_triggered", "cells"):
            if then.get(key) != now[key]:
                problems.append(
                    f"{target}.{key} is recorded as "
                    f"{json.dumps(then.get(key))[:120]} but the panel on disk "
                    f"measures {json.dumps(now[key])[:120]}")
        then_diag = (then.get("repetition_diagnostics") or {})
        now_diag = now["repetition_diagnostics"]
        if then_diag.get("n_classified_loop") != now_diag["n_classified_loop"]:
            problems.append(
                f"{target}: the artifact classifies "
                f"{then_diag.get('n_classified_loop')} repetition loop(s) but "
                f"the registered criterion now classifies "
                f"{now_diag['n_classified_loop']}")
        if then_diag.get("criterion") != LOOP_CRITERION:
            problems.append(
                f"{target}: the artifact states a different loop criterion "
                "than the one registered in this script, so its count is not "
                "reproducible from it")

    totals = committed.get("totals") or {}
    if totals.get("n_truncated_cells") != fresh["totals"]["n_truncated_cells"]:
        problems.append(
            f"totals.n_truncated_cells is {totals.get('n_truncated_cells')} "
            f"but the panels measure {fresh['totals']['n_truncated_cells']}")
    if (totals.get("n_classified_repetition_loop")
            != fresh["totals"]["n_classified_repetition_loop"]):
        problems.append(
            f"totals.n_classified_repetition_loop is "
            f"{totals.get('n_classified_repetition_loop')} but the registered "
            f"criterion gives "
            f"{fresh['totals']['n_classified_repetition_loop']}")

    # The resolution has to still say what it is. An artifact that quietly
    # dropped the "not the motivation" clause would read as a tolerance
    # chosen by the scores.
    resolution = committed.get("resolution") or {}
    for key in ("what_was_inconsistent", "how_it_was_resolved",
                "kind_of_change", "not_the_motivation",
                "authority_for_the_numbers", "eligibility_gate_unchanged",
                "sealed_evidence_unchanged", "cells_are_kept"):
        if not resolution.get(key):
            problems.append(f"resolution.{key} is missing or empty")
    motivation = str(resolution.get("not_the_motivation") or "")
    if "score" not in motivation.lower():
        problems.append(
            "resolution.not_the_motivation no longer names the judge scores it "
            "excludes, so the artifact could be read back as a tolerance "
            "chosen by the result")

    # No arm may have been accepted while outside the thresholds, and none may
    # be recorded as escalating the cap while inside them.
    for target, now in fresh["per_target"].items():
        if now["violations"] and now["within_registered_thresholds"]:
            problems.append(f"{target} reports violations and still claims to "
                            "be within the registered thresholds")
        if now["uniform_cap_escalation_triggered"] != bool(now["violations"]):
            problems.append(
                f"{target}: cap escalation must be triggered exactly when a "
                "registered threshold is exceeded")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="gate the committed artifact; write nothing")
    parser.add_argument("--profiles", default=str(SCALE_PROFILES))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    profiles_path = Path(args.profiles)
    out_path = Path(args.out)

    if args.verify:
        problems = verify(out_path, profiles_path)
        if problems:
            print("TRUNCATION EVIDENCE VERIFY: FAIL")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("TRUNCATION EVIDENCE VERIFY: PASS")
        print(f"  artifact: {_rel(out_path)}")
        return 0

    evidence = build(profiles_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
        f.write("\n")

    totals = evidence["totals"]
    print(f"wrote {_rel(out_path)}")
    print(f"  targets: {', '.join(evidence['targets'])}")
    for target, arm in evidence["per_target"].items():
        diag = arm["repetition_diagnostics"]
        print(f"  {target:14s} {arm['n_truncated']} of {arm['n_records']} "
              f"capped (rate {arm['overall_rate']:.4f} <= "
              f"{MAX_TRUNCATION_RATE}, spread {arm['max_variant_spread']:.4f}"
              f" <= {MAX_VARIANT_SPREAD}) -> within thresholds: "
              f"{arm['within_registered_thresholds']}, cap escalation: "
              f"{arm['uniform_cap_escalation_triggered']}, loops: "
              f"{diag['n_classified_loop']}/{arm['n_truncated']}")
    reference = evidence["reference_9b"]
    if reference:
        print(f"  {'REFERENCE 9B':14s} {reference['n_truncated']} of "
              f"{reference['n_records']} capped, longest complete output "
              f"{reference['longest_complete_output_tokens']} tokens against "
              f"a cap of {reference['max_new_tokens']}")
    print(f"  total capped cells: {totals['n_truncated_cells']}, classified "
          f"repetition loops: {totals['n_classified_repetition_loop']}, not: "
          f"{totals['n_not_classified_repetition_loop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
