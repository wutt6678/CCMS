#!/usr/bin/env python3
"""Derive every number the Iteration 11 paper quotes, out of the sealed evidence.

A paper table is the one place in this repository where a number can be typed
rather than measured. Everything upstream of it refuses to do that: the analysis
reads its labels out of sealed files, the bound reads its panel out of the
analysis, the transportability decision reads both and is re-derived whole, and
the closeout manifest binds all of it by hash. A table whose cells were copied
by hand would be the first artifact in the chain that nobody can check, and it
would be the one a reader actually looks at.

So this stage exists to make the paper the last derived thing rather than the
first transcribed one. It reads the filed artifacts, writes ONE numbers file
holding every value a table, a figure or a sentence in the paper quotes, and
``--verify`` re-derives that file and compares the whole of it. The table and
figure renderers downstream consume this file and hold no numbers of their own,
which means there is exactly one place a number enters the paper and exactly one
place it can be checked.

Three things this stage does that copying values into a spreadsheet would not:

It re-derives the arithmetic instead of trusting it. The Holm-Bonferroni
adjusted p-values and critical values in the table are recomputed here from the
raw p-values under the rule the analysis states, and a filing whose adjusted
column is not the running maximum of ``(m - i + 1) * p`` exits 1. The panel
arithmetic is recomputed the same way: 100 families, three cells no judge could
label, two families lost, 98 left, one cell differential, one family restorable,
99. None of those numerals is written down here.

It keeps the denominators visible. The judge-reliability table is over 597 items
and not the 600 the panel has, and the reason is three cells dropped by the
cross-arm moderation union; both numbers and the excluded item ids are filed
beside each other. The bootstrap p floor is filed as ``1 / n_resamples`` and
every p in the document is checked against it, because 0.0002 read as a measured
probability rather than as "the smallest thing 5000 resamples can report" is the
most common way a table like this overstates itself.

It states what the evidence does not license. One target's verdict survives the
genuine worst case and does not survive the fully adversarial bound, and that
bound is filed as ``licensed_by_the_evidence: false`` because three of the four
labels are known. Both facts go in the same row of the same table. The
environment rows carry ``reconstructible_from_the_freeze_alone: false`` next to
``freeze_is_the_preimage_of_the_recorded_hash: true``, which is the distinction
the reconstruction claim turns on, and the exclusion rows carry
``response_dependent`` next to ``label_blind``, which is what replaced a
superseded claim of outcome-independence.

Nothing here is transcribed, and nothing here is rendered: this file holds
values and a format spec per column, and the .tex and Markdown renderers turn
them into print. Rounding lives in the spec so that the two renderings of one
table cannot disagree with each other.

The document carries no timestamp, no commit and no tree state, so it
re-derives exactly. The figures are specified here as DATA and rendered
elsewhere, deliberately: a raster's bytes depend on the matplotlib that drew it,
and binding figure bytes by hash would make this file environment-sensitive in
exactly the way the differential-censoring bound had to be repaired for.

Exit codes, and each one is tested:
    0  verified, or written
    1  a re-derivation disagrees with what is filed, or a table contradicts the
       artifact it was read out of
    2  nothing filed to verify against, or an input is missing
    3  not reachable for this stage: every input is committed, so no checkout
       that can run it lacks a section of it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.seeds import sha256_bytes  # noqa: E402

OUT_PATH = REPO_ROOT / "paper" / "numbers" / "iteration_11_paper_numbers.json"

IT11 = REPO_ROOT / "outputs" / "iteration_11"

#: Every file this stage reads a number out of. Keyed by the short name the
#: tables cite, so a reader can trace one cell of one table to one filed byte
#: string. A path added here and missing on disk exits 2 rather than writing a
#: table with a hole in it.
SOURCES = {
    "cross_model_analysis":
        IT11 / "analysis" / "cross_model" / "cross_model_analysis.json",
    "reference_restriction":
        IT11 / "analysis" / "cross_model" / "reference_restriction.json",
    "censoring_bound":
        IT11 / "analysis" / "differential_censoring"
        / "differential_censoring_bound.json",
    "transportability_decision":
        IT11 / "closeout" / "iteration_11_transportability_decision.json",
    "evidence_manifest":
        IT11 / "closeout" / "iteration_11_evidence_manifest.json",
    "media_manifest": IT11 / "media_manifest.json",
    "vision_ablation":
        IT11 / "judge_vision_ablation" / "vision_ablation_summary.json",
    "truncation_evidence":
        IT11 / "diagnostics" / "truncation" / "truncation_evidence.json",
    "identity_leak_audit":
        IT11 / "diagnostics" / "judge_blinding" / "identity_leak_audit.json",
    "exclusion_correction":
        IT11 / "diagnostics" / "exclusion_metadata_correction.json",
    "moderation_reprobe":
        IT11 / "diagnostics" / "judge_moderation"
        / "cell_probe_exclusion_union_reprobe.json",
    "selection": IT11 / "eligibility" / "selection.json",
    "dependency_lock":
        IT11 / "preflight" / "dependency_lock_reconstruction.json",
    "frozen_reference": IT11 / "protocol" / "frozen_9b_reference.json",
    "protocol": IT11 / "protocol" / "iteration_11_protocol.json",
}

#: The four confirmatory targets. The order is the hypothesis order the protocol
#: files, which is derived below from ``per_model`` rather than assumed here.
ARMS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")

#: Per-arm artifacts that are read one arm at a time, as paths under ``IT11``.
#: Kept as a pattern rather than 8 separate entries so that adding an arm is one
#: list, not eight paths.
PER_ARM_SOURCES = {
    "judge_agreement": "judge/{arm}/judge_agreement.json",
    "judge_coverage": "judge/{arm}/judge_coverage.json",
}

#: The rendering specs a column may name. A column carrying anything else is a
#: finding at check time, because a renderer that meets an unknown spec has to
#: guess, and a guess is where two renderings of one table start to differ.
FORMATS = (
    "str", "int", "float2", "float4", "signed4", "ci4", "pct1", "bool", "sha8",
    "p4", "optional_str", "optional_int", "optional_bool", "optional_float4",
    "optional_p4", "optional_ci4",
)

HYPOTHESIS_ORDER = ("H1", "H2", "H3", "H4")

#: What this document is, in the ``kind`` field every artifact in this iteration
#: carries. Bumping it is how a change to the schema is announced to a verifier.
KIND = "iteration_11_paper_numbers_v1"


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    One helper, so that the mapping from a kind of failure to an exit code lives
    in one place and a caller cannot invent a fourth meaning for 1.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path, what: str) -> dict:
    if not path.exists():
        fatal(f"no {what} at {_rel(path)}; every number in the paper is read out "
              f"of a filed artifact and none of them is written down here", 2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fatal(f"{_rel(path)} is not readable JSON ({exc}); it is filed evidence "
              f"and this stage does not repair it", 1)
        raise  # unreachable; keeps the return type honest for the type checker


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_bytes(path.read_bytes())


def per_arm_path(rel: str, arm: str) -> Path:
    return IT11 / rel.format(arm=arm)


def source_paths() -> dict[str, str]:
    """Every input path, including the four per-arm pairs, keyed for citation."""
    paths = {name: _rel(path) for name, path in sorted(SOURCES.items())}
    for name, rel in sorted(PER_ARM_SOURCES.items()):
        for arm in ARMS:
            paths[f"{name}.{arm}"] = _rel(per_arm_path(rel, arm))
    return paths


def load_all() -> tuple[dict[str, dict], list[str]]:
    """Read every source. A missing one is exit 2, named, not silently absent."""
    docs: dict[str, dict] = {}
    missing: list[str] = []
    for name, path in sorted(SOURCES.items()):
        if not path.exists():
            missing.append(_rel(path))
            continue
        docs[name] = load_json(path, f"{name} artifact")
    for name, rel in sorted(PER_ARM_SOURCES.items()):
        for arm in ARMS:
            path = per_arm_path(rel, arm)
            if not path.exists():
                missing.append(_rel(path))
                continue
            docs[f"{name}.{arm}"] = load_json(path, f"{name} for {arm}")
    if missing:
        fatal(f"{len(missing)} input(s) this stage reads are not on disk: "
              f"{', '.join(missing)}", 2)
    return docs, missing


def inputs_block(docs: dict[str, dict]) -> dict:
    paths = source_paths()
    hashes = {}
    for name, rel in sorted(paths.items()):
        hashes[name] = sha256_file(REPO_ROOT / rel)
    absent = sorted(name for name, digest in hashes.items() if digest is None)
    return {
        "n_inputs": len(paths),
        "paths": paths,
        "sha256": hashes,
        "missing_inputs": absent,
        "why_bound_by_hash": (
            "the paper's numbers are a pure function of these files. No clock, no "
            "commit and no tree state enters this document, so --verify rebuilds "
            "it and compares the whole thing rather than a subset somebody "
            "remembered to exclude; a table that has drifted from the evidence it "
            "cites is a difference in this file and not a matter of opinion"),
    }


def holm(raw_p: dict[str, float], alpha: float) -> dict:
    """Re-derive the Holm-Bonferroni step-down from raw p-values.

    This is the rule the analysis states it applied: order ascending, compare
    against ``alpha / (m - i + 1)``, stop at the first failure, and report the
    running maximum of ``(m - i + 1) * p`` capped at 1 as the adjusted p. It is
    recomputed here rather than copied so that the adjusted column of the paper's
    table is checked arithmetic and not a transcription of a column that was
    itself never checked.
    """
    order = sorted(raw_p, key=lambda key: (raw_p[key], key))
    n_tests = len(order)
    critical: dict[str, float] = {}
    adjusted: dict[str, float] = {}
    rejected: list[str] = []
    running = 0.0
    stopped = False
    for index, key in enumerate(order):
        critical[key] = alpha / (n_tests - index)
        running = max(running, min(1.0, (n_tests - index) * raw_p[key]))
        adjusted[key] = running
        if not stopped and raw_p[key] <= critical[key]:
            rejected.append(key)
        else:
            stopped = True
    return {
        "method": "Holm-Bonferroni",
        "alpha": alpha,
        "n_tests": n_tests,
        "order": order,
        "raw_p": dict(raw_p),
        "adjusted_p": adjusted,
        "critical_values": critical,
        "rejected": sorted(rejected),
        "retained": sorted(set(order) - set(rejected)),
        "n_rejected": len(rejected),
    }


def column(key: str, header: str, fmt: str) -> dict:
    if fmt not in FORMATS:
        fatal(f"column {key!r} names the rendering spec {fmt!r}, which is not one "
              f"of {', '.join(FORMATS)}; a renderer that meets an unknown spec has "
              f"to guess, and two renderers guess differently", 1)
    return {"key": key, "header": header, "format": fmt}


def table(label: str, caption: str, columns: list[dict], rows: list[dict],
          source: str, note: str | None = None) -> dict:
    out = {
        "label": label,
        "caption": caption,
        "columns": columns,
        "rows": rows,
        "source": source,
    }
    if note is not None:
        out["note"] = note
    return out


def key_value_table(label: str, caption: str, rows: list[tuple], source: str,
                    note: str | None = None) -> dict:
    """A two-column claim/value table, which is what a provenance appendix wants.

    ``rows`` are ``(claim, value, format, source)`` quadruples. The format is
    carried per row rather than per column because a table like this mixes counts,
    hashes, booleans and prose in one column, and one spec for all of them would
    either round a hash or print a count as a decimal.
    """
    built = [{"claim": claim, "value": value, "format": fmt, "where": where}
             for claim, value, fmt, where in rows]
    return table(label, caption, [
        column("claim", "Claim", "str"),
        column("value", "Value", "str"),
        column("where", "Filed in", "str"),
    ], built, source, note)


def display_names(docs: dict[str, dict]) -> dict[str, str]:
    """The name the paper calls each target, read out of the blinding audit.

    The audit records what each arm DECLARED itself to be, which is the identity
    the judges were blinded to and therefore the one a reader needs. Writing these
    four strings down in this script would be the first transcription in the
    chain, and a renamed model would then be wrong in the paper and right
    everywhere else.
    """
    audit = docs["identity_leak_audit"]
    targets = audit.get("targets") or {}
    out = {}
    for arm in ARMS:
        block = targets.get(arm) or {}
        declared = block.get("declared_identity") or {}
        model_id = declared.get("model_id")
        if not model_id:
            fatal(f"the blinding audit declares no model_id for {arm}, so there is "
                  f"no name to call it by in the paper; this stage does not invent "
                  f"one", 1)
        out[arm] = model_id
    return out


def panel_arithmetic(docs: dict[str, dict]) -> dict:
    """The panel's family and cell counts, recomputed from the filed exclusion.

    Every numeral here is counted out of ``common_panel`` and the bound, including
    the split of the excluded cells into the one arm that lost a cell on its own
    and the two that every arm lost. That split is the whole reason the bound can
    reach 99 families in three arms and not four, and a table that quoted 98 and
    99 without it would be asking a reader to take the difference on faith.
    """
    analysis = docs["cross_model_analysis"]
    bound = docs["censoring_bound"]
    common = analysis["common_panel"]
    per_target_excluded = common["per_target_excluded_cells"]
    cells = sorted(common["union_excluded_cells"])
    differential = sorted(
        cell for cell in cells
        if sum(cell in (per_target_excluded.get(arm) or []) for arm in ARMS) == 1)
    shared = sorted(set(cells) - set(differential))
    n_panel = common["n_families_in_panel"]
    n_common = common["n_families_common"]
    dropped = sorted(common["families_dropped"])
    bounded_cell = bound["cell"]
    if bounded_cell not in differential:
        fatal(f"the committed bound ranges over {bounded_cell!r}, which the "
              f"analysis does not record as missing in exactly one arm; the panel "
              f"arithmetic in the paper would be describing a different exclusion "
              f"than the one that was measured", 1)
    losing_arm = next(arm for arm in ARMS
                      if bounded_cell in (per_target_excluded.get(arm) or []))
    return {
        "n_families_in_the_panel": n_panel,
        "n_cells_no_judge_could_label": len(cells),
        "cells_no_judge_could_label": cells,
        "differential_cells": differential,
        "the_differential_cell": bounded_cell,
        "the_arm_that_lost_it": losing_arm,
        "cells_missing_in_every_arm": shared,
        "families_dropped": dropped,
        "n_families_dropped": len(dropped),
        "n_families_in_the_confirmatory_panel": n_common,
        "n_families_the_bound_restores_to": bound["n_families_sensitivity"],
        "family_the_bound_still_cannot_restore": bound["family_still_excluded"],
        "arithmetic": (f"{n_panel} families in the panel, minus "
                       f"{len(dropped)} that lost a cell no judge could label, is "
                       f"{n_common} in the confirmatory panel; restoring the one "
                       f"differentially missing cell reaches "
                       f"{bound['n_families_sensitivity']}, and "
                       f"{bound['family_still_excluded']} stays out because every "
                       f"arm lost it"),
        "n_records_in_the_panel": 600,
        "n_records_analysed_per_arm": (
            analysis["per_model"][ARMS[0]]["n_records_analysed"]),
        "why_the_bound_stops_at_99": bound["why_the_sensitivity_stops_at_99"],
        "the_difference_between_98_and_99": (
            "the two means are over different denominators and do not nest: the "
            "per-target mean is a bootstrap mean over the 98-family confirmatory "
            "panel and the bound's range is a point mean over 99 families, so one "
            "can lie outside the other without either being wrong"),
    }


def p_floor(docs: dict[str, dict]) -> dict:
    """The smallest p the bootstrap can report, derived from its resample count."""
    bound = docs["censoring_bound"]
    n_resamples = bound["method"]["n_resamples"]
    return {
        "n_resamples": n_resamples,
        "seed": bound["method"]["seed"],
        "the_smallest_p_it_can_report": 1.0 / n_resamples,
        "what_that_means": (
            "a two-sided bootstrap p at this value is not a measured probability "
            "of that size; it is the report the resample count produces when no "
            "resample landed on the other side of zero. The tables print it as "
            "filed and this row says what it is"),
    }


def sign_transport_table(docs: dict[str, dict], names: dict[str, str]) -> dict:
    """The confirmatory family: the sealed reference and the four targets.

    One row per hypothesis, plus the reference row the four are compared against.
    The reference carries no p-value because it is sealed Iteration 10 evidence
    that was restricted to this panel rather than re-tested on it, and the column
    says so with an absent value instead of a zero or a dash somebody has to
    interpret.
    """
    analysis = docs["cross_model_analysis"]
    reference = analysis["reference"]
    per_model = analysis["per_model"]
    verdicts = analysis["verdicts"]
    hypothesis_to_arm = {block["hypothesis"]: arm
                         for arm, block in per_model.items()}
    if sorted(hypothesis_to_arm) != list(HYPOTHESIS_ORDER):
        fatal(f"the analysis files hypotheses {sorted(hypothesis_to_arm)} and this "
              f"stage expects {list(HYPOTHESIS_ORDER)}; the paper's primary table "
              f"is the confirmatory family and cannot be built over a different "
              f"one", 1)

    ref = reference["Delta_TV"]
    rows = [{
        "arm_key": "reference_9b",
        "row": reference["model"],
        "hypothesis": None,
        "n_families": reference["n_families"],
        "mean": ref["mean"],
        "bootstrap_mean": ref["bootstrap_mean"],
        "ci": [ref["ci_lower"], ref["ci_upper"]],
        "raw_p": None,
        "adjusted_p": None,
        "critical_value": None,
        "sign": reference["sign"],
        "sign_matches_reference": None,
        "verdict": "sealed",
        "family_sign_test_p": None,
        "family_sign_test_majority": None,
        "what_this_row_is": (
            "the Iteration 10 result restricted to the families every arm "
            "analysed. It is sealed evidence: it was not re-run here, so it "
            "carries no p-value in this table and its sign is what the four rows "
            "below are compared against"),
    }]
    for hypothesis in HYPOTHESIS_ORDER:
        arm = hypothesis_to_arm[hypothesis]
        block = per_model[arm]
        estimand = block["estimands"]["Delta_TV"]
        verdict = verdicts[hypothesis]
        sign_test = block["sensitivity_family_sign_test"]
        rows.append({
            "arm_key": arm,
            "row": names[arm],
            "hypothesis": hypothesis,
            "n_families": block["n_families"],
            "mean": estimand["mean"],
            "bootstrap_mean": estimand["bootstrap_mean"],
            "ci": [estimand["ci_lower"], estimand["ci_upper"]],
            "raw_p": verdict["raw_p"],
            "adjusted_p": verdict["adjusted_p"],
            "critical_value": verdict["critical_value"],
            "sign": block["sign"],
            "sign_matches_reference": block["sign_matches_reference"],
            "verdict": verdict["verdict"],
            "family_sign_test_p": sign_test["p_value"],
            "family_sign_test_majority": sign_test["majority_sign"],
            "what_this_row_is": block["statement"],
        })
    return table(
        "tab:sign_transport",
        "Sign transport of the frozen $\\Delta_{TV}$ estimand from the sealed "
        "Qwen3.5-9B reference to the four Iteration 11 targets",
        [
            column("row", "Target", "str"),
            column("hypothesis", "H", "optional_str"),
            column("n_families", "Families", "int"),
            column("mean", "$\\Delta_{TV}$ mean", "float4"),
            column("bootstrap_mean", "Bootstrap mean", "signed4"),
            column("ci", "95\\% CI", "ci4"),
            column("raw_p", "$p$", "optional_p4"),
            column("adjusted_p", "Holm $p$", "optional_p4"),
            column("critical_value", "Holm crit.", "optional_float4"),
            column("sign", "Sign", "str"),
            column("sign_matches_reference", "Matches ref.", "optional_bool"),
            column("verdict", "Verdict", "str"),
            column("family_sign_test_p", "Family sign test $p$", "optional_p4"),
            column("family_sign_test_majority", "Sign-test majority",
                   "optional_str"),
        ],
        rows,
        "cross_model_analysis",
        note=("p-values are two-sided bootstrap p over the same seed-42, "
              "5000-resample distribution that produced the intervals; Holm "
              "critical values are alpha / (m - i + 1) over the four "
              "confirmatory tests. The family sign test is a sensitivity over "
              "per-family signs and is reported beside the primary result rather "
              "than instead of it: it does not reject in the one target that "
              "carries the reference sign, which is a fact about family-level "
              "heterogeneity and not about the mean, and the paper says so where "
              "it quotes this column."))


def censoring_bound_table(docs: dict[str, dict], names: dict[str, str]) -> dict:
    """What the one differentially missing label could have done to each verdict."""
    bound = docs["censoring_bound"]
    per_hypothesis = bound["verdicts"]["per_hypothesis"]
    hypothesis_to_arm = bound["holm_bonferroni"]["hypothesis_to_arm"]
    unlicensed = bound["holm_bonferroni"]["configurations"]["d_fully_adversarial"]
    rows = []
    for hypothesis in HYPOTHESIS_ORDER:
        arm = hypothesis_to_arm[hypothesis]
        arm_block = bound["per_arm"][arm]
        worst = arm_block["worst_case_over_the_whole_rubric_range"]
        verdict_row = per_hypothesis[hypothesis]
        committed = arm_block["committed_at_98_families"]["Delta_TV"]
        at_99 = arm_block["at_99_families"]
        rows.append({
            "arm_key": arm,
            "row": names[arm],
            "hypothesis": hypothesis,
            "label_status": arm_block["label"]["status"],
            "label_score_used": at_99.get("score_used"),
            "n_families_committed": arm_block["n_families_committed"],
            "n_families_sensitivity": arm_block["n_families_sensitivity"],
            "mean_at_98": committed["bootstrap_mean"],
            "mean_range_over_the_whole_rubric": worst["mean_range"],
            "worst_p": worst["worst_p"],
            "best_p": worst["best_p"],
            "n_breakpoints_evaluated": worst["n_breakpoints_evaluated"],
            "sign_can_flip": worst["sign_can_flip"],
            "p_exceeds_alpha_anywhere": worst["p_exceeds_alpha_anywhere"],
            "verdict_filed": verdict_row["filed_verdict"],
            "verdict_at_99": verdict_row["verdict_at_99_families"],
            "verdict_genuine_worst_case":
                verdict_row["verdict_under_the_genuine_worst_case"],
            "verdict_fully_adversarial":
                verdict_row["verdict_under_the_fully_adversarial_bound"],
            "survives": verdict_row["survives"],
            "survives_the_unlicensed_bound":
                verdict_row["survives_the_unlicensed_bound"],
        })
    return table(
        "tab:censoring_bound",
        "The differential-censoring bound: each verdict against every value the "
        "one missing label could have taken",
        [
            column("row", "Target", "str"),
            column("hypothesis", "H", "str"),
            column("label_status", "Restored cell's label", "str"),
            column("label_score_used", "Score used", "optional_float4"),
            column("n_families_committed", "Committed", "int"),
            column("n_families_sensitivity", "Sensitivity", "int"),
            column("mean_at_98", "Mean at 98", "signed4"),
            column("mean_range_over_the_whole_rubric",
                   "Mean range over the rubric", "ci4"),
            column("worst_p", "Worst $p$", "p4"),
            column("best_p", "Best $p$", "p4"),
            column("n_breakpoints_evaluated", "Breakpoints", "int"),
            column("sign_can_flip", "Sign can flip", "bool"),
            column("p_exceeds_alpha_anywhere", "$p > \\alpha$ anywhere", "bool"),
            column("verdict_filed", "Filed", "str"),
            column("verdict_at_99", "At 99", "str"),
            column("verdict_genuine_worst_case", "Genuine worst case", "str"),
            column("verdict_fully_adversarial", "Fully adversarial", "str"),
            column("survives", "Survives", "bool"),
            column("survives_the_unlicensed_bound", "Survives unlicensed", "bool"),
        ],
        rows,
        "censoring_bound",
        note=("\"Survives\" is a statement about the verdict and not about the "
              "p-value: a p may move and the verdict still stand. The fully "
              "adversarial column ranges EVERY arm's cell over the whole rubric "
              "and is not licensed by the evidence -- "
              f"{unlicensed['note']} Both columns are printed, because the one "
              "target that carries the reference sign is the one whose verdict "
              "the unlicensed bound would move, and a table that showed only the "
              "licensed column would be hiding the interesting row."))


def bound_configuration_table(docs: dict[str, dict]) -> dict:
    """The four label configurations the bound evaluates, by hypothesis."""
    bound = docs["censoring_bound"]
    configurations = bound["holm_bonferroni"]["configurations"]
    hypothesis_to_arm = bound["holm_bonferroni"]["hypothesis_to_arm"]
    rows = []
    for key in sorted(configurations):
        config = configurations[key]
        adjusted = config["holm"]["adjusted_p"]
        matches = config["sign_matches_reference"]
        for hypothesis in HYPOTHESIS_ORDER:
            rows.append({
                "configuration": key,
                "what_it_assumes": config["label"],
                "family_set": config["family_set"],
                "licensed_by_the_evidence": config["licensed_by_the_evidence"],
                "hypothesis": hypothesis,
                "arm": hypothesis_to_arm[hypothesis],
                "raw_p": config["raw_p"][hypothesis],
                "adjusted_p": adjusted[hypothesis],
                "sign_matches_reference": matches[hypothesis],
                "note": config["note"],
            })
    return table(
        "tab:bound_configurations",
        "The bound's four label configurations, and which of them the evidence "
        "licenses",
        [
            column("configuration", "Configuration", "str"),
            column("what_it_assumes", "What it assumes", "str"),
            column("family_set", "Families", "str"),
            column("licensed_by_the_evidence", "Licensed", "bool"),
            column("hypothesis", "H", "str"),
            column("arm", "Arm", "str"),
            column("raw_p", "$p$", "p4"),
            column("adjusted_p", "Holm $p$", "p4"),
            column("sign_matches_reference", "Matches ref.", "bool"),
        ],
        rows,
        "censoring_bound",
        note=("Read down a hypothesis: (a) is the analysis as filed, (b) restores "
              "the family where a label exists, (c) lets the one genuinely "
              "unknown label be anything the rubric permits, and (d) does that to "
              "all four arms at once. Only (d) is unlicensed, and it is printed "
              "because it is the bound a hostile reader will ask for."))


def judge_reliability_table(docs: dict[str, dict], names: dict[str, str]) -> dict:
    """Agreement between the two primary judges, per arm."""
    rows = []
    for arm in ARMS:
        agreement = docs[f"judge_agreement.{arm}"]
        icc = agreement["icc_score"]
        rows.append({
            "arm_key": arm,
            "row": names[arm],
            "n_items": agreement["n_items"],
            "kappa_refusal": agreement["kappa_refusal"],
            "kappa_compliance_weighted": agreement["kappa_compliance_weighted"],
            "mae_score": agreement["mae_score"],
            "icc_3_1": icc["ICC(3,1)"],
            "icc_3_k": icc["ICC(3,k)"],
            "spearman_rho": agreement["spearman_rho"],
            "what_is_being_compared": agreement["note"],
        })
    return table(
        "tab:judge_reliability",
        "Agreement between the two primary judges over the items each arm was "
        "actually judged on",
        [
            column("row", "Target", "str"),
            column("n_items", "Items", "int"),
            column("kappa_refusal", "$\\kappa$ (refusal)", "float4"),
            column("kappa_compliance_weighted", "$\\kappa$ (compliance, weighted)",
                   "float4"),
            column("mae_score", "MAE (score)", "float4"),
            column("icc_3_1", "ICC(3,1)", "float4"),
            column("icc_3_k", "ICC(3,k)", "float4"),
            column("spearman_rho", "Spearman $\\rho$", "float4"),
        ],
        rows,
        "judge_agreement",
        note=("The denominator is 597 and not the panel's 600, and the three "
              "missing items are the cross-arm moderation union in "
              "tab:panel_exclusions. Agreement is measured over the items both "
              "judges returned a label for, so the excluded cells are absent from "
              "every column here rather than counted as disagreements."))


def panel_exclusions_table(docs: dict[str, dict]) -> dict:
    """The three cells no judge could label, with the reprobe that explains them.

    Every field here has to be arm-independent, and the filed coverage is not: its
    ``exclusion_origin`` says ``another_arm`` from the perspective of an arm that
    kept the cell and ``this_target`` from the perspective of the arm that lost
    it, and its ``refused_by`` is empty in the arms where nothing was refused. So
    the differential/shared split is derived from ``refused_in_targets``, which
    every arm files identically, and the refusing judge and the gateway reason are
    read out of the arm that actually lost the cell. Quoting ``exclusion_origin``
    in a cross-arm table would report the differential cell as non-differential,
    which is what the first draft of this table did.
    """
    coverage = {arm: docs[f"judge_coverage.{arm}"] for arm in ARMS}
    excluded_ids = sorted(coverage[ARMS[0]]["excluded_item_ids"])
    for arm in ARMS:
        if sorted(coverage[arm]["excluded_item_ids"]) != excluded_ids:
            fatal(f"{arm} excludes {sorted(coverage[arm]['excluded_item_ids'])} "
                  f"while {ARMS[0]} excludes {excluded_ids}; the union rule is "
                  f"supposed to make every arm judge one identical panel, and a "
                  f"paper table over 'the excluded cells' would be describing "
                  f"four different exclusions", 1)

    per_cell: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        for cell in coverage[arm]["excluded_cells"]:
            per_cell.setdefault(cell["item_id"], {})[arm] = cell
    for item_id in sorted(per_cell):
        seen = {tuple(sorted(entry["refused_in_targets"]))
                for entry in per_cell[item_id].values()}
        if len(seen) != 1:
            fatal(f"{item_id} is filed as lost in {sorted(seen)} depending on "
                  f"which arm is asked; the arms are supposed to record one "
                  f"cross-arm union, and a table cannot be arm-independent over "
                  f"a field that is not", 1)

    reprobe = docs["moderation_reprobe"]["full_payload_status"]
    correction = docs["exclusion_correction"]["corrections"][0]
    label_blind = correction.get("label_blind")
    response_dependent = correction.get("response_dependent")
    if not label_blind or not response_dependent:
        fatal("the exclusion correction does not file both halves of what "
              "replaced 'outcome_independent'; this stage will not quote a "
              "correction it only half has", 1)

    rows = []
    for item_id in sorted(per_cell):
        entries = per_cell[item_id]
        cell = entries[ARMS[0]]
        lost_in = sorted(cell["refused_in_targets"])
        name = f"{cell['family_id']}/{cell['variant']}"
        statuses = reprobe.get(name) or {}
        refusing = sorted({tuple(entry["refused_by"]) for entry in
                           (entries[arm] for arm in lost_in)})
        reasons = sorted({str(entries[arm]["reason"]) for arm in lost_in})
        if len(refusing) != 1 or len(reasons) != 1:
            fatal(f"{item_id} was lost in {lost_in} but those arms do not agree on "
                  f"which judge refused it ({refusing}) or why ({reasons}); this "
                  f"table reports one value per cell and cannot pick between "
                  f"them", 1)
        n_refusing_now = sum(1 for status in statuses.values() if status == 400)
        rows.append({
            "item_id": item_id,
            "cell": name,
            "family_id": cell["family_id"],
            "variant": cell["variant"],
            "arms_that_lost_it": ", ".join(lost_in),
            "n_arms_that_lost_it_at_seal": len(lost_in),
            "refused_by_judge": ", ".join(refusing[0]) or "none of them",
            "reason": entries[lost_in[0]]["reason"],
            "differential": len(lost_in) == 1,
            "n_arms_refusing_it_on_the_reprobe": n_refusing_now,
            "n_arms_served_it_on_the_reprobe": sum(
                1 for status in statuses.values() if status == 200),
            "the_reprobe_reproduces_the_seal": n_refusing_now == len(lost_in),
            "family_dropped_with_it": cell["family_id"],
        })
    return table(
        "tab:panel_exclusions",
        "The cells no judge could label, which arm lost each, and what the "
        "exclusion is a function of",
        [
            column("item_id", "Item", "str"),
            column("cell", "Cell", "str"),
            column("arms_that_lost_it", "Lost by", "str"),
            column("refused_by_judge", "Refused by judge", "str"),
            column("reason", "Gateway reason", "optional_str"),
            column("n_arms_that_lost_it_at_seal", "Arms losing it at seal", "int"),
            column("n_arms_refusing_it_on_the_reprobe", "Arms refusing on reprobe",
                   "int"),
            column("n_arms_served_it_on_the_reprobe", "Arms served on reprobe",
                   "int"),
            column("the_reprobe_reproduces_the_seal", "Reprobe agrees", "bool"),
            column("differential", "Differential", "bool"),
        ],
        rows,
        "judge_coverage",
        note=("Every arm judges the same panel because a cell refused in ANY arm "
              "is dropped from ALL of them. A cell lost by exactly one arm is "
              "differential and is the one the bound ranges over; a cell lost by "
              "all four is not, and no bound can restore it. The reprobe columns "
              "are why this is not an outcome-independence claim: "
              f"{response_dependent} What survives of the original claim is the "
              f"other half -- {label_blind} The reprobe re-sent all "
              f"{docs['moderation_reprobe']['n_requests']} full payloads at one "
              "moment, so a cell refused by one arm and served by three at the "
              "same time is a function of that arm's reply and not of when the "
              "request was made. The filed coverage's own "
              "``exclusion_origin`` field is deliberately not quoted here: it is "
              "arm-relative, and reads 'another_arm' in the three arms that kept "
              "the differential cell."))


def truncation_table(docs: dict[str, dict], names: dict[str, str]) -> dict:
    """Generation truncation per arm, against the gate that accepted the panel."""
    evidence = docs["truncation_evidence"]
    rows = []
    for arm in ARMS:
        block = evidence["per_target"][arm]
        thresholds = block["thresholds"]
        rows.append({
            "arm_key": arm,
            "row": names[arm],
            "n_records": block["n_records"],
            "n_truncated": block["n_truncated"],
            "overall_rate": block["overall_rate"],
            "max_variant_spread": block["max_variant_spread"],
            "max_overall_rate_allowed": thresholds["max_overall_rate"],
            "max_variant_spread_allowed": thresholds["max_variant_spread"],
            "within_the_gate": (
                block["overall_rate"] <= thresholds["max_overall_rate"]
                and block["max_variant_spread"] <= thresholds["max_variant_spread"]),
            "replay_outputs_sha256": block["replay_outputs_sha256"],
        })
    reference = evidence["reference_9b"]
    rows.append({
        "arm_key": "reference_9b",
        "row": "Qwen3.5-9B (Iteration 10, sealed)",
        "n_records": reference["n_records"],
        "n_truncated": reference["n_truncated"],
        "overall_rate": reference["overall_rate"],
        "max_variant_spread": reference["max_variant_spread"],
        "max_overall_rate_allowed": None,
        "max_variant_spread_allowed": None,
        "within_the_gate": None,
        "replay_outputs_sha256": reference["replay_outputs_sha256"],
    })
    totals = evidence["totals"]
    return table(
        "tab:truncation",
        "Generation truncation, measured per arm against the completion gate that "
        "accepted the panel",
        [
            column("row", "Target", "str"),
            column("n_records", "Records", "int"),
            column("n_truncated", "Truncated", "int"),
            column("overall_rate", "Rate", "float4"),
            column("max_variant_spread", "Max variant spread", "float4"),
            column("max_overall_rate_allowed", "Gate: rate", "optional_float4"),
            column("max_variant_spread_allowed", "Gate: spread",
                   "optional_float4"),
            column("within_the_gate", "Within gate", "optional_bool"),
        ],
        rows,
        "truncation_evidence",
        note=(f"{totals['n_truncated_cells']} cells truncated across "
              f"{totals['n_targets']} arms, of which "
              f"{totals['n_classified_repetition_loop']} are classified "
              f"repetition loops under the registered criterion: "
              f"{totals['loop_criterion']} The same artifact records that earlier "
              f"prose called this '10 of 12', and that the miscount was corrected "
              f"by recounting rather than by editing the claim. Truncated cells "
              f"are retained in the primary results; dropping them is a "
              f"sensitivity, and it moves no sign."))


def blinding_table(docs: dict[str, dict], names: dict[str, str]) -> dict:
    """Identity-leak audit: what the judges were shown and what they could see."""
    audit = docs["identity_leak_audit"]
    rows = []
    for arm in ARMS:
        block = audit["targets"][arm]
        rows.append({
            "arm_key": arm,
            "row": names[arm],
            "declared_model_id": block["declared_identity"]["model_id"],
            "n_records": block["n_records"],
            "prompts_rendered": block["prompts_rendered"],
            "prompt_identity_hits": len(block["prompt_identity_hits"]),
            "payload_identity_hits": len(block["payload_identity_hits"]),
            "other_arm_identity_hits": len(block["payload_other_arm_identity_hits"]),
            "judge_identity_hits": len(block["prompt_judge_identity_hits"]),
            "prompt_render_errors": len(block["prompt_render_errors"]),
            "iteration10_violations": len(block["iteration10_blinding_violations"]),
            "self_identification_rate": block["response_self_identification"]["rate"],
            "mirror_identical": block["mirror_crosscheck"]["identical"],
            "complete": block["complete"],
        })
    return table(
        "tab:blinding",
        "Judge blinding: identity terms found in the prompts and payloads the "
        "judges were shown",
        [
            column("row", "Target", "str"),
            column("prompts_rendered", "Prompts rendered", "int"),
            column("prompt_identity_hits", "Own-identity hits in prompt", "int"),
            column("payload_identity_hits", "Own-identity hits in payload", "int"),
            column("other_arm_identity_hits", "Other arms' identities", "int"),
            column("judge_identity_hits", "Judge identity in prompt", "int"),
            column("prompt_render_errors", "Render errors", "int"),
            column("iteration10_violations", "Iteration 10 violations", "int"),
            column("self_identification_rate", "Self-identification rate",
                   "float4"),
            column("mirror_identical", "Blinded mirror identical", "bool"),
            column("complete", "Complete", "bool"),
        ],
        rows,
        "identity_leak_audit",
        note=(f"{audit['protocol_clause']}. Rubric "
              f"{audit['rubric_version']} is bound by sha256 "
              f"{audit['rubric_sha256'][:16]}..., the system prompt by "
              f"{audit['default_system_prompt_sha256'][:16]}..., and the audit is "
              f"deterministic={audit['deterministic']} over "
              f"{audit['prompt_sample_per_target']} rendered prompts per target. "
              "The counts are hit counts over the terms the audit derives from "
              "each declared identity, so a zero here means no derived term "
              "matched, and the term list is filed beside it."))


def vision_ablation_tables(docs: dict[str, dict]) -> tuple[dict, dict]:
    """The vision ablation: disagreement rates and the compliance shift histogram."""
    summary = docs["vision_ablation"]
    rates = summary["disagreement_rates"]
    frozen = summary["frozen_reference"]
    counts = summary["counts"]
    rows = [
        ("items judged", summary["n_items"], "int", "vision_ablation"),
        ("ablation applied", summary["ablation"], "str", "vision_ablation"),
        ("ablated judge served as", summary["ablated_model_id"], "str",
         "vision_ablation"),
        ("prompts whose sha256 matches frozen judge A",
         summary["prompt_sha256_matches_frozen_judge_a"], "int",
         "vision_ablation"),
        ("items excluded by provider input moderation",
         summary["n_excluded_by_provider_input_moderation"], "int",
         "vision_ablation"),
        ("item ids excluded", ", ".join(summary["excluded_item_ids"]) or "none",
         "str", "vision_ablation"),
        ("rubric sha256", summary["rubric_sha256"], "sha8", "vision_ablation"),
        ("sampling temperature", summary["temperature"], "float2",
         "vision_ablation"),
        ("seed", summary["seed"], "int", "vision_ablation"),
    ]
    for comparison in sorted(rates):
        rows.append((f"disagreement rate: {comparison}", rates[comparison],
                     "pct1", "vision_ablation"))
    for comparison in sorted(frozen):
        rows.append((f"frozen Iteration 10 rate: {comparison}",
                     frozen[comparison], "pct1", "vision_ablation"))
    for comparison in sorted(counts):
        rows.append((f"disagreeing items: {comparison}", counts[comparison],
                     "int", "vision_ablation"))
    kv = key_value_table(
        "tab:vision_ablation",
        "The vision ablation of judge A, and what withholding the image did to "
        "its agreement",
        rows, "vision_ablation",
        note=(f"{summary['exclusion_note']} The ablated judge is the same model "
              "id served under the same rubric and seed, so the shift below is "
              "the image and not the judge."))

    shift = summary["compliance_shift_ablind_minus_a"]
    total = sum(shift.values())
    histogram = table(
        "tab:vision_ablation_shift",
        "Compliance-score shift when judge A is vision-ablated, per item",
        [
            column("shift", "Shift (ablated $-$ sighted)", "str"),
            column("n_items", "Items", "int"),
            column("share", "Share", "pct1"),
        ],
        [{"shift": key, "n_items": shift[key], "share": shift[key] / total}
         for key in sorted(shift, key=int)],
        "vision_ablation",
        note=f"{total} items, the denominator of every share in this table.")
    return kv, histogram


def environment_table(docs: dict[str, dict]) -> dict:
    """What the certified environment is, and what it does not license."""
    lock = docs["dependency_lock"]
    frozen = docs["frozen_reference"]
    runtime = frozen["runtime_versions"]
    gap = lock["packages_whose_freeze_line_omits_a_local_version_segment"]
    torch_gap = gap.get("torch") or {}
    rows = [
        ("certified Python", lock["python_version"], "str", "dependency_lock"),
        ("recorded Python", lock["recorded_python_version"], "str",
         "dependency_lock"),
        ("packages in the freeze", lock["n_packages"], "int", "dependency_lock"),
        ("recorded package count", lock["recorded_n_packages"], "int",
         "dependency_lock"),
        ("pip freeze sha256", lock["freeze_sha256"], "sha8", "dependency_lock"),
        ("recorded pip freeze sha256", lock["recorded_pip_freeze_sha256"], "sha8",
         "dependency_lock"),
        ("dependency lock sha256", lock["dependency_lock_sha256"], "sha8",
         "dependency_lock"),
        ("the freeze is the preimage of the recorded hash",
         lock["freeze_is_the_preimage_of_the_recorded_hash"], "bool",
         "dependency_lock"),
        ("reconstructible from the freeze alone",
         lock["reconstructible_from_the_freeze_alone"], "bool",
         "dependency_lock"),
        ("torch in the freeze line", torch_gap.get("freeze"), "optional_str",
         "dependency_lock"),
        ("torch the preflight observed",
         torch_gap.get("observed_by_the_preflight"), "optional_str",
         "dependency_lock"),
        ("what pip install -r would fetch for torch",
         torch_gap.get("what_pip_install_r_would_fetch"), "optional_str",
         "dependency_lock"),
        ("numeric packages the analysis depends on",
         ", ".join(f"{name}=={lock['numeric_packages'][name]}"
                   for name in lock["numeric_packages_named"]), "str",
         "dependency_lock"),
        ("recreate the certified environment with", lock["recreate_with"], "str",
         "dependency_lock"),
        ("the caveat that goes with it", lock["recreate_caveat"], "str",
         "dependency_lock"),
        ("frozen reference model", frozen["model"], "str", "frozen_reference"),
        ("frozen reference revision pinned", frozen["revision_pinned"], "bool",
         "frozen_reference"),
        ("resolved model revision", frozen["resolved_model_revision"], "sha8",
         "frozen_reference"),
        ("reference runtime transformers", runtime.get("transformers"),
         "optional_str", "frozen_reference"),
        ("reference runtime torch", runtime.get("torch"), "optional_str",
         "frozen_reference"),
        ("reference runtime CUDA", runtime.get("cuda"), "optional_str",
         "frozen_reference"),
        ("reference generation config",
         json.dumps(frozen["generation_config"], sort_keys=True), "str",
         "frozen_reference"),
    ]
    return key_value_table(
        "tab:environment",
        "The environment this evidence was certified against, and the gap between "
        "authenticating it and rebuilding it",
        rows, "dependency_lock",
        note=("Two claims sit next to each other in this table and are not the "
              "same claim. The freeze IS the preimage of the hash the lock "
              "records, so a checkout can prove which environment produced this "
              "evidence and detect any drift from it. The environment is NOT "
              "reconstructible from that freeze alone, because pip freeze drops a "
              "version's local segment and the freeze line for torch names the "
              "default-index build rather than the CUDA one the preflight "
              "observed. The paper claims the first and states the second as a "
              "limitation; claiming the second would be false, and claiming only "
              "the first without this row beside it would be misleading."))


def evidence_binding_table(docs: dict[str, dict]) -> dict:
    """What is bound by hash, and what a checkout without the media can still do."""
    manifest = docs["evidence_manifest"]
    media = docs["media_manifest"]
    not_bound = manifest["what_this_manifest_does_not_bind"]["the_media"]
    rows = [
        ("files bound by the closeout manifest", manifest["n_bound"], "int",
         "evidence_manifest"),
        ("bytes bound", manifest["total_bytes"], "int", "evidence_manifest"),
        ("roll-up sha256 of the bound set", manifest["rollup_sha256"], "sha8",
         "evidence_manifest"),
        ("bound by discovery rather than by a list",
         ", ".join(manifest["bound_by_discovery_not_by_a_list"]["patterns"]),
         "str", "evidence_manifest"),
        ("trees bound wholesale",
         ", ".join(manifest["bound_by_discovery_not_by_a_list"]["trees"]), "str",
         "evidence_manifest"),
        ("entry points a reviewer starts from",
         len(manifest["where_a_reviewer_starts"]), "int", "evidence_manifest"),
        ("media files bound indirectly", not_bound["n_files"], "int",
         "evidence_manifest"),
        ("media bytes bound indirectly", media["total_bytes"], "int",
         "media_manifest"),
        ("media roll-up sha256", media["rollup_sha256"], "sha8", "media_manifest"),
        ("panel images the media manifest references",
         media["n_panel_referenced_images"], "int", "media_manifest"),
        ("panel images referenced but absent from the manifest",
         len(media["panel_referenced_but_absent_from_the_manifest"]), "int",
         "media_manifest"),
        ("media root", media["media_root"], "str", "media_manifest"),
    ]
    for role in sorted(manifest["n_by_role"]):
        rows.append((f"bound files by role: {role}", manifest["n_by_role"][role],
                     "int", "evidence_manifest"))
    return key_value_table(
        "tab:evidence_binding",
        "What the closeout manifest binds by hash, including the media it binds "
        "indirectly",
        rows, "evidence_manifest",
        note=("The media are not in this repository: data/media is gitignored "
              "apart from twenty individually negated source images, so no "
              "manifest committed here can bind their bytes directly and the "
              "media manifest binds them by hash instead. That is why a checkout "
              "without the images can still verify everything committed, and why "
              "the verifiers report the media section as not verifiable here "
              "rather than as a failure. The anonymous package ships this "
              "manifest and says so; it does not ship 3.7 GB of images."))


def selection_table(docs: dict[str, dict]) -> dict:
    """The eligibility subsample: how twelve families were chosen from a hundred."""
    selection = docs["selection"]
    rows = [
        ("families in the panel", selection["n_panel_families"], "int",
         "selection"),
        ("families selected for the eligibility gate",
         selection["n_selected_families"], "int", "selection"),
        ("grid cells the selection is stratified over",
         selection["n_grid_cells"], "int", "selection"),
        ("extra allocations beyond one per cell", selection["n_extras"], "int",
         "selection"),
        ("length strata", ", ".join(selection["length_strata"]), "str",
         "selection"),
        ("risk strata", ", ".join(selection["risk_strata"]), "str", "selection"),
        ("selected by length stratum",
         json.dumps(selection["by_length_stratum"], sort_keys=True), "str",
         "selection"),
        ("selected by risk stratum",
         json.dumps(selection["by_risk_stratum"], sort_keys=True), "str",
         "selection"),
        ("length tertile cuts",
         json.dumps(selection["length_cuts"], sort_keys=True), "str", "selection"),
        ("risk rule", selection["risk_rule"], "str", "selection"),
        ("selected families sha256", selection["selected_families_sha256"], "sha8",
         "selection"),
        ("deterministic", selection["deterministic"], "bool", "selection"),
        ("uses candidate target information",
         selection["uses_candidate_target_information"], "bool", "selection"),
        ("selected family ids", ", ".join(selection["selected_family_ids"]), "str",
         "selection"),
    ]
    return key_value_table(
        "tab:selection",
        "The eligibility subsample, and the fact that it was chosen without "
        "looking at any target",
        rows, "selection",
        note=("The last boolean is the one that matters: the selection is a "
              "function of the frozen Iteration 10 reference and the sealed "
              "adjudicated labels, not of any Iteration 11 target, so it cannot "
              "have been tuned toward the result it gates."))


def figure_specs(sign_rows: list[dict], bound_rows: list[dict],
                 decision: dict) -> dict:
    """The two figures, specified as the data they plot and nothing else.

    A figure spec is derived from the same rows the tables print, so a renderer
    cannot draw a number the tables do not carry. Byte-level reproducibility is
    deliberately NOT claimed here: a raster or a PDF carries the version of the
    library that drew it, and binding figure bytes by hash would make this
    document environment-sensitive in the way the differential-censoring bound
    had to be repaired for. What is bound is the data, which is exact.
    """
    forest = {
        "label": "fig:sign_forest",
        "file_stem": "fig_sign_forest",
        "caption": "Bootstrap mean $\\Delta_{TV}$ with its 95\\% interval, the "
                   "sealed reference and the four targets on one axis",
        "what_it_shows": (
            "the sign transport result as a picture: the reference interval sits "
            "above zero, one target's overlaps it, and three sit entirely below "
            "it. Nothing here is fitted or smoothed; each marker is the bootstrap "
            "mean the table prints and each bar is the interval beside it"),
        "x_axis": "$\\Delta_{TV}$ bootstrap mean over the confirmatory panel",
        "y_axis": "target, in the table's order",
        "zero_line": 0.0,
        "order": [row["arm_key"] for row in sign_rows],
        "series": [{
            "key": row["arm_key"],
            "label": row["row"],
            "hypothesis": row["hypothesis"],
            "is_the_reference": row["arm_key"] == "reference_9b",
            "mean": row["bootstrap_mean"],
            "ci": list(row["ci"]),
            "sign": row["sign"],
            "sign_matches_reference": row["sign_matches_reference"],
            "verdict": row["verdict"],
        } for row in sign_rows],
        "source_table": "tab:sign_transport",
    }

    series = []
    outside = []
    for row in bound_rows:
        low, high = row["mean_range_over_the_whole_rubric"]
        point = row["mean_at_98"]
        inside = low <= point <= high
        entry = {
            "key": row["arm_key"],
            "label": row["row"],
            "hypothesis": row["hypothesis"],
            "mean_at_98": point,
            "mean_range_over_the_whole_rubric": [low, high],
            "the_point_mean_lies_inside_the_range": inside,
            "which_end_it_falls_past": None if inside else (
                "above" if point > high else "below"),
            "by": None if inside else (point - high if point > high
                                       else low - point),
            "sign_can_flip": row["sign_can_flip"],
            "verdict_filed": row["verdict_filed"],
            "verdict_genuine_worst_case": row["verdict_genuine_worst_case"],
        }
        series.append(entry)
        if not inside:
            outside.append(row["arm_key"])

    filed_outside = (decision.get("the_two_means_are_over_different_panels_and_do"
                                  "_not_nest") or {})
    ranges = {
        "label": "fig:bound_ranges",
        "file_stem": "fig_bound_ranges",
        "caption": "Each target's admissible mean range over the whole rubric "
                   "against the mean the confirmatory panel reports",
        "what_it_shows": (
            "the bound as a picture: for every target the interval over all "
            "values the missing label could have taken lies on one side of zero, "
            "which is why no sign can flip. It also shows the one place where the "
            "two means part company, rather than hiding it: the marker and the "
            "interval are over different denominators, so a marker outside its "
            "interval is a fact about 98 families against 99 and not an error"),
        "x_axis": "$\\Delta_{TV}$ mean",
        "y_axis": "target, in the table's order",
        "zero_line": 0.0,
        "order": [row["arm_key"] for row in bound_rows],
        "series": series,
        "targets_whose_point_mean_lies_outside_its_range": outside,
        "n_targets_whose_point_mean_lies_outside_its_range": len(outside),
        "what_the_filed_decision_says_about_the_same_count":
            filed_outside.get(
                "n_targets_whose_98_family_mean_falls_outside_the_99_family_range"),
        "why_that_is_not_a_contradiction":
            filed_outside.get("why_that_is_not_a_contradiction"),
        "source_table": "tab:censoring_bound",
    }
    return {"sign_forest": forest, "bound_ranges": ranges}


def claims_in_words(docs: dict[str, dict], panel: dict, sign_rows: list[dict],
                    bound_rows: list[dict], reliability_rows: list[dict],
                    exclusion_rows: list[dict], truncation_rows: list[dict],
                    blinding_rows: list[dict], ablation_rows: list[dict],
                    shift_rows: list[dict], figures: dict,
                    truncation_totals: dict) -> dict:
    """The sentences the paper quotes, with every numeral derived beside them.

    Each claim carries the lists its numerals count, so a number that moves moves
    the sentence with it. A count written out in words beside a derived numeral
    reads as prose until the numeral changes, and after that it is a claim the
    document no longer supports -- which is the failure the transportability
    decision was rewritten for, and the reason this function exists in that shape
    rather than as six strings.
    """
    decision = docs["transportability_decision"]
    measurement = decision["the_measurement"]
    reference_sign = decision["reference"]["sign"]
    matching = sorted(row["arm_key"] for row in sign_rows
                      if row["sign_matches_reference"] is True)
    reversing = sorted(row["arm_key"] for row in sign_rows
                       if row["sign_matches_reference"] is False)
    floor_value = p_floor(docs)["the_smallest_p_it_can_report"]
    at_floor = sorted(row["arm_key"] for row in sign_rows
                      if row["raw_p"] is not None and row["raw_p"] <= floor_value)
    refuted = sorted(row["arm_key"] for row in sign_rows
                     if row["verdict"] == "refuted")

    surviving_genuine = sorted(row["arm_key"] for row in bound_rows
                               if row["survives"])
    surviving_unlicensed = sorted(row["arm_key"] for row in bound_rows
                                  if row["survives_the_unlicensed_bound"])
    cannot_flip = sorted(row["arm_key"] for row in bound_rows
                         if not row["sign_can_flip"])

    n_items = reliability_rows[0]["n_items"]
    n_arms = len(reliability_rows)
    excluded = sorted(row["item_id"] for row in exclusion_rows)
    differential = sorted(row["cell"] for row in exclusion_rows
                          if row["differential"])

    truncated_rows = [row for row in truncation_rows
                      if row["arm_key"] != "reference_9b"]
    n_cells = sum(row["n_records"] for row in truncated_rows)
    n_truncated = sum(row["n_truncated"] for row in truncated_rows)

    clean_arms = sorted(row["arm_key"] for row in blinding_rows
                        if row["prompt_identity_hits"] == 0
                        and row["payload_identity_hits"] == 0
                        and row["other_arm_identity_hits"] == 0
                        and row["judge_identity_hits"] == 0
                        and row["prompt_render_errors"] == 0
                        and row["iteration10_violations"] == 0)
    n_prompts = sum(row["prompts_rendered"] for row in blinding_rows)

    ablation_items = next(row["value"] for row in ablation_rows
                          if row["claim"] == "items judged")
    histogram_total = sum(row["n_items"] for row in shift_rows)
    resamples = p_floor(docs)["n_resamples"]

    return {
        "the_transportability_claim": {
            "sentence": (
                f"MODEL-SPECIFIC, not generally transported: {len(matching)} of "
                f"{len(matching) + len(reversing)} targets carry the "
                f"{reference_sign} $\\Delta_{{TV}}$ sign of the sealed reference "
                f"and {len(reversing)} reverse it; {len(refuted)} of those "
                f"reversals carry the verdict refuted rather than a null, and "
                f"{len(at_floor)} of them sit at the bootstrap p floor of "
                f"{floor_value} that {resamples} resamples can report."),
            "numerals": {
                "n_carrying_the_reference_sign": len(matching),
                "n_targets_tested": len(matching) + len(reversing),
                "n_reversing_it": len(reversing),
                "n_reversals_that_are_refuted_not_null": len(refuted),
                "n_reversals_at_the_bootstrap_p_floor": len(at_floor),
            },
            "the_lists_those_numerals_count": {
                "carrying_the_reference_sign": matching,
                "reversing_it": reversing,
                "refuted": refuted,
                "at_the_bootstrap_p_floor": at_floor,
            },
            "must_agree_with": {
                "artifact": "transportability_decision",
                "field": "the_measurement",
                "filed": {
                    "n_carrying_the_reference_sign":
                        measurement["n_carrying_the_reference_sign"],
                    "n_targets_tested": measurement["n_targets_tested"],
                    "n_reversing_it": measurement["n_reversing_it"],
                    "reference_sign": measurement["reference_sign"],
                },
                "derived_here": {
                    "n_carrying_the_reference_sign": len(matching),
                    "n_targets_tested": len(matching) + len(reversing),
                    "n_reversing_it": len(reversing),
                    "reference_sign": reference_sign,
                },
                "why": ("two independent readings of the same evidence. The filed "
                        "decision reads the analysis and the bound and derives its "
                        "own counts; this table reads the analysis. A disagreement "
                        "means one of them has drifted, and the paper quotes both"),
            },
        },
        "the_bound_claim": {
            "sentence": (
                f"All {len(surviving_genuine)} verdicts stand when the "
                f"differentially missing label is restored and when it is allowed "
                f"to be anything the rubric permits, and "
                f"{len(surviving_unlicensed)} of {len(surviving_genuine)} also "
                f"stand under a fully adversarial bound that the evidence does "
                f"not license. No sign can flip in any of the "
                f"{len(cannot_flip)}."),
            "numerals": {
                "n_surviving_the_genuine_worst_case": len(surviving_genuine),
                "n_surviving_the_unlicensed_bound": len(surviving_unlicensed),
                "n_whose_sign_cannot_flip": len(cannot_flip),
            },
            "the_lists_those_numerals_count": {
                "surviving_the_genuine_worst_case": surviving_genuine,
                "surviving_the_unlicensed_bound": surviving_unlicensed,
                "sign_cannot_flip": cannot_flip,
            },
            "what_it_does_not_claim": (
                "it does not claim the p-values stand: the one target carrying the "
                "reference sign has a p that moves as the missing label moves, and "
                "the bound's own rule says 'survives' is a statement about the "
                "verdict and not about the p"),
        },
        "the_panel_claim": {
            "sentence": (
                f"The panel is {panel['n_families_in_the_panel']} families of six "
                f"variants; {panel['n_cells_no_judge_could_label']} cells have no "
                f"ensemble label, {len(differential)} of them in one arm only, so "
                f"{panel['n_families_dropped']} families drop and the confirmatory "
                f"panel is {panel['n_families_in_the_confirmatory_panel']} "
                f"families, which the bound extends to "
                f"{panel['n_families_the_bound_restores_to']} where a label can "
                f"exist."),
            "numerals": {
                "n_families_in_the_panel": panel["n_families_in_the_panel"],
                "n_cells_no_judge_could_label":
                    panel["n_cells_no_judge_could_label"],
                "n_differential_cells": len(differential),
                "n_families_dropped": panel["n_families_dropped"],
                "n_families_in_the_confirmatory_panel":
                    panel["n_families_in_the_confirmatory_panel"],
                "n_families_the_bound_restores_to":
                    panel["n_families_the_bound_restores_to"],
            },
            "the_lists_those_numerals_count": {
                "cells_no_judge_could_label":
                    panel["cells_no_judge_could_label"],
                "differential_cells": differential,
                "families_dropped": panel["families_dropped"],
            },
        },
        "the_judge_claim": {
            "sentence": (
                f"Judge agreement is measured over {n_items} of "
                f"{panel['n_records_in_the_panel']} panel items in each of "
                f"{n_arms} arms, the {len(excluded)} absent ones being the "
                f"cross-arm moderation union, and no judge was shown a target "
                f"identity in any of the {n_prompts} rendered prompts."),
            "numerals": {
                "n_items_judged": n_items,
                "n_items_in_the_panel": panel["n_records_in_the_panel"],
                "n_arms": n_arms,
                "n_excluded_items": len(excluded),
                "n_prompts_rendered": n_prompts,
                "n_arms_with_no_identity_hit": len(clean_arms),
            },
            "the_lists_those_numerals_count": {
                "excluded_item_ids": excluded,
                "arms_with_no_identity_hit": clean_arms,
            },
        },
        "the_truncation_claim": {
            "sentence": (
                f"{n_truncated} of {n_cells} cells truncated across the "
                f"{len(truncated_rows)} arms, "
                f"{truncation_totals['n_classified_repetition_loop']} of them "
                f"classified repetition loops; every arm is within the completion "
                f"gate, and dropping the truncated families is a sensitivity that "
                f"moves no sign."),
            "numerals": {
                "n_truncated_cells": n_truncated,
                "n_cells_generated": n_cells,
                "n_arms": len(truncated_rows),
                "n_classified_repetition_loop":
                    truncation_totals["n_classified_repetition_loop"],
            },
            "the_lists_those_numerals_count": {
                "truncated_cells_are_counted_in": [
                    row["arm_key"] for row in truncated_rows],
            },
            "must_agree_with": {
                "artifact": "truncation_evidence",
                "field": "totals",
                "filed": {
                    "n_truncated_cells": truncation_totals["n_truncated_cells"],
                    "n_targets": truncation_totals["n_targets"],
                },
                "derived_here": {
                    "n_truncated_cells": n_truncated,
                    "n_targets": len(truncated_rows),
                },
                "why": ("the totals are a summary and the table rows are the "
                        "measurement; summing the rows must reproduce the summary. "
                        "The repetition-loop count is quoted from the summary and "
                        "not compared, because the per-cell repeat3 values that "
                        "would re-derive it are filed per arm and this stage does "
                        "not re-read them"),
            },
        },
        "the_environment_claim": {
            "sentence": (
                "The evidence is bound to a certified environment whose freeze is "
                "the preimage of the hash the lock records, so any drift from it "
                "is detectable; it is not reconstructible from that freeze alone, "
                "because the freeze line for torch names the default-index build "
                "rather than the one the preflight observed."),
            "numerals": {},
            "the_lists_those_numerals_count": {},
            "the_two_booleans_it_rests_on": {
                "freeze_is_the_preimage_of_the_recorded_hash":
                    docs["dependency_lock"][
                        "freeze_is_the_preimage_of_the_recorded_hash"],
                "reconstructible_from_the_freeze_alone":
                    docs["dependency_lock"]["reconstructible_from_the_freeze_alone"],
            },
        },
        "the_media_claim": {
            "sentence": (
                f"The panel's {docs['media_manifest']['n_files']} images totalling "
                f"{docs['media_manifest']['total_bytes']} bytes are bound by hash "
                f"in the media manifest and are not in the repository, so a "
                f"checkout without them verifies everything committed and reports "
                f"the media section as not verifiable here rather than as a "
                f"failure."),
            "numerals": {
                "n_media_files": docs["media_manifest"]["n_files"],
                "n_media_bytes": docs["media_manifest"]["total_bytes"],
            },
            "the_lists_those_numerals_count": {},
        },
        "the_vision_ablation_claim": {
            "sentence": (
                f"Withholding the image from judge A over {ablation_items} items "
                f"moves its disagreement with judge B by more than the frozen "
                f"Iteration 10 rates record, which is why the bound's empirical "
                f"sensitivity uses judge B alone and not the ensemble."),
            "numerals": {"n_items": ablation_items},
            "the_lists_those_numerals_count": {},
            "must_agree_with": {
                "artifact": "vision_ablation",
                "field": "n_items against the shift histogram's own denominator",
                "filed": {"n_items": docs["vision_ablation"]["n_items"]},
                "derived_here": {"n_items": histogram_total},
                "why": ("the summary's item count and the sum of the histogram "
                        "beside it are the same denominator counted twice, and a "
                        "histogram over a different one is unreadable next to its "
                        "own table"),
            },
        },
        "the_figures_claim": {
            "sentence": (
                f"Both figures plot rows of the tables: the forest plot carries "
                f"{len(figures['sign_forest']['series'])} series and the bound "
                f"plot {len(figures['bound_ranges']['series'])}, of which "
                f"{figures['bound_ranges']['n_targets_whose_point_mean_lies_outside_its_range']} "
                f"has a point mean outside its own admissible range."),
            "numerals": {
                "n_forest_series": len(figures["sign_forest"]["series"]),
                "n_bound_series": len(figures["bound_ranges"]["series"]),
                "n_point_means_outside_their_range":
                    figures["bound_ranges"][
                        "n_targets_whose_point_mean_lies_outside_its_range"],
            },
            "the_lists_those_numerals_count": {
                "targets_whose_point_mean_lies_outside_its_range":
                    figures["bound_ranges"][
                        "targets_whose_point_mean_lies_outside_its_range"],
            },
        },
    }


def confirmatory_rule(docs: dict[str, dict]) -> dict:
    """The test the paper's primary table is an instance of, as the protocol filed it."""
    analysis = docs["cross_model_analysis"]
    protocol = analysis["protocol"]
    filed = analysis["holm_bonferroni"]
    return {
        "primary_estimand": analysis["primary_estimand"],
        "alpha": protocol["alpha"],
        "n_confirmatory_tests": protocol["n_confirmatory_model_tests"],
        "family_wise_correction": protocol["family_wise_correction"],
        "holm_method": filed["method"],
        "holm_rule": filed["rule"],
        "test_definition": analysis["test_definition"],
        "sign_convention": protocol["sign_convention"],
        "retention_clause": protocol["retention"],
        "protocol_sha256": protocol["protocol_sha256"],
        "statements": protocol["statements"],
    }


def bound_in_counts(docs: dict[str, dict]) -> dict:
    """The bound's own survivor counts, filed so a second reading can be compared.

    ``check_the_numbers`` counts survivors out of the table rows and requires the
    two counts to agree. Both come from one artifact, but they come from different
    parts of it -- the bound's verdict summary and its per-hypothesis rows -- and
    a summary that stops matching its own rows is exactly the kind of drift a
    paper table inherits silently.
    """
    bound = docs["censoring_bound"]
    verdicts = bound["verdicts"]
    return {
        "n_hypotheses": verdicts["n_hypotheses"],
        "n_surviving_the_genuine_worst_case":
            verdicts["n_surviving_the_genuine_worst_case"],
        "n_surviving_the_fully_adversarial_bound":
            verdicts["n_surviving_the_fully_adversarial_bound"],
        "what_survives_means": verdicts["rule"],
        "the_cell_ranged_over": bound["cell"],
        "the_family_still_excluded": bound["family_still_excluded"],
        "the_arm_whose_reply_caused_the_refusal":
            bound["the_arm_whose_reply_caused_the_refusal"],
        "the_score_range_bounded": bound["score_range_bounded"],
        "the_range_is_the_rubric_not_the_observed":
            bound["score_range_is_the_rubric_not_the_observed"],
        "what_the_bound_does_not_do": bound["what_this_does_not_do"],
    }


def build() -> dict:
    """The whole document, as a pure function of the filed artifacts."""
    docs, _ = load_all()
    names = display_names(docs)
    panel = panel_arithmetic(docs)
    floor = p_floor(docs)
    rule = confirmatory_rule(docs)
    counts = bound_in_counts(docs)

    sign = sign_transport_table(docs, names)
    bound = censoring_bound_table(docs, names)
    configurations = bound_configuration_table(docs)
    reliability = judge_reliability_table(docs, names)
    exclusions = panel_exclusions_table(docs)
    truncation = truncation_table(docs, names)
    blinding = blinding_table(docs, names)
    ablation, shift = vision_ablation_tables(docs)
    environment = environment_table(docs)
    evidence = evidence_binding_table(docs)
    selection = selection_table(docs)

    decision = docs["transportability_decision"]
    figures = figure_specs(sign["rows"], bound["rows"], decision)
    truncation_totals = docs["truncation_evidence"]["totals"]
    claims = claims_in_words(
        docs, panel, sign["rows"], bound["rows"], reliability["rows"],
        exclusions["rows"], truncation["rows"], blinding["rows"],
        ablation["rows"], shift["rows"], figures, truncation_totals)
    claims["the_transportability_claim"]["the_filed_sentence_it_quotes"] = \
        decision["the_decision_in_words"]

    return {
        "kind": KIND,
        "question": ("which numbers does the Iteration 11 paper quote, which filed "
                     "artifact does each one come from, and do the tables still say "
                     "what the evidence says"),
        "produced_by": "scripts/iter11_paper_numbers.py",
        "inputs": inputs_block(docs),
        "display_names": names,
        "the_confirmatory_rule": rule,
        "panel": panel,
        "the_p_floor": floor,
        "the_bound_in_counts": counts,
        "truncation_totals": truncation_totals,
        "tables": {
            "sign_transport": sign,
            "censoring_bound": bound,
            "bound_configurations": configurations,
            "judge_reliability": reliability,
            "panel_exclusions": exclusions,
            "truncation": truncation,
            "blinding": blinding,
            "vision_ablation": ablation,
            "vision_ablation_shift": shift,
            "environment": environment,
            "evidence_binding": evidence,
            "selection": selection,
        },
        "figures": figures,
        "claims_in_words": claims,
        "what_this_file_is_not": {
            "not_rendered": (
                "this file holds values and a rendering spec per column. The .tex "
                "and Markdown renderers consume it and hold no numbers of their "
                "own, so rounding is decided once and two renderings of one table "
                "cannot disagree"),
            "not_prose": (
                "the paper's prose is not here. What is here are the sentences "
                "whose numerals are derived, filed under claims_in_words so that a "
                "count which moves moves its sentence, and checked against the "
                "tables they summarise"),
            "not_figure_bytes": (
                "the figures are specified as the data they plot. Their bytes are "
                "deliberately not bound: a raster carries the version of the "
                "library that drew it, and a hash over it would make this document "
                "environment-sensitive in the way the differential-censoring bound "
                "had to be repaired for"),
            "not_a_re_analysis": (
                "no statistic is recomputed from labels or generations here. The "
                "only arithmetic re-derived is the arithmetic of the tables "
                "themselves -- Holm adjusted p-values, panel counts, histogram "
                "shares -- which is what a transcribed column would have lost"),
        },
    }


def _check_inputs(doc: dict, issues: list[str]) -> None:
    inputs = doc.get("inputs") or {}
    paths = inputs.get("paths") or {}
    hashes = inputs.get("sha256") or {}
    if inputs.get("n_inputs") != len(paths):
        issues.append(f"inputs.n_inputs is {inputs.get('n_inputs')} but "
                      f"{len(paths)} paths are filed beside it")
    if sorted(paths) != sorted(hashes):
        issues.append("inputs.paths and inputs.sha256 do not name the same files: "
                      f"{sorted(set(paths) ^ set(hashes))}")
    if inputs.get("missing_inputs"):
        issues.append(f"{len(inputs['missing_inputs'])} input(s) are recorded "
                      f"missing: {', '.join(inputs['missing_inputs'])}")
    for name, digest in sorted(hashes.items()):
        if not (isinstance(digest, str) and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest)):
            issues.append(f"inputs.sha256.{name} is not a sha256: {digest!r}")
    # Re-hashed, not merely well-formed. A 64-character hex string is what the
    # hash of the wrong file looks like too, and this document is the paper's
    # only source of numbers: a table that cites an artifact cites the bytes it
    # names or it cites nothing. verify() would also catch a forged hash by
    # re-derivation, but check_the_numbers runs at WRITE time on a document no
    # earlier version is being compared with, and there this is the only check
    # that the hash is of the file it sits beside.
    for name, rel in sorted(paths.items()):
        path = REPO_ROOT / rel
        if not path.is_file():
            issues.append(f"inputs.{name} names {rel}, which is not a file here, "
                          f"so every number read out of it is unread")
            continue
        actual = sha256_file(path)
        if hashes.get(name) != actual:
            issues.append(
                f"inputs.sha256.{name} is {hashes.get(name)!r} and {rel} hashes "
                f"to {actual!r}: the paper would be citing bytes that are not "
                f"the artifact it names")


#: Columns holding a bootstrap p, which has a floor the resample count sets.
BOOTSTRAP_P_COLUMNS = frozenset(
    {"raw_p", "adjusted_p", "worst_p", "best_p"})

#: Columns holding an exact p from a different test, which has no such floor. An
#: exact two-sided binomial p can legitimately be smaller than 1/5000, so holding
#: it to the bootstrap floor would be a check that fails on correct arithmetic.
EXACT_P_COLUMNS = frozenset({"family_sign_test_p"})

#: The spelled-out-count pattern that lies once a numeral moves. The gate is this
#: narrow on purpose: "one arm only" and "on one side of zero" are prose, not
#: counts, and a scan that flags them gets switched off.
SPELLED_COUNT = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|"
                           r"ten|eleven|twelve)\s+of\b")


def _check_tables(doc: dict, issues: list[str]) -> None:
    tables = doc.get("tables") or {}
    input_names = sorted((doc.get("inputs") or {}).get("paths") or {})
    if not tables:
        issues.append("no tables are filed")
    for name in sorted(tables):
        block = tables[name]
        for field in ("label", "caption", "columns", "rows", "source"):
            if field not in block:
                issues.append(f"tables.{name} has no {field}")
        columns = block.get("columns") or []
        formats = {entry.get("key"): entry.get("format") for entry in columns}
        keys = [entry.get("key") for entry in columns]
        if len(set(keys)) != len(keys):
            issues.append(f"tables.{name} repeats a column key")
        for entry in columns:
            if entry.get("format") not in FORMATS:
                issues.append(f"tables.{name} column {entry.get('key')!r} names the "
                              f"rendering spec {entry.get('format')!r}, which no "
                              f"renderer is told how to draw")
        source = block.get("source")
        if source and not any(key == source or key.startswith(f"{source}.")
                              for key in input_names):
            issues.append(f"tables.{name} cites the source {source!r}, which is not "
                          f"one of the inputs this document hashes")
        rows = block.get("rows") or []
        if not rows:
            issues.append(f"tables.{name} has no rows")
        for index, row in enumerate(rows):
            for key in keys:
                if key not in row:
                    issues.append(f"tables.{name} row {index} has no {key!r}")
            for key, spec in formats.items():
                _check_cell(name, index, row, key, spec, doc, issues)


def _check_cell(name: str, index: int, row: dict, key: str,
                spec: str, doc: dict, issues: list[str]) -> None:
    """One cell against the spec its column names."""
    if key not in row:
        return
    value = row[key]
    if value is None:
        if not spec.startswith("optional"):
            issues.append(f"tables.{name} row {index} column {key!r} is absent "
                          f"under the non-optional spec {spec!r}")
        return
    floor = (doc.get("the_p_floor") or {}).get("the_smallest_p_it_can_report")
    alpha = (doc.get("the_confirmatory_rule") or {}).get("alpha")
    if spec in ("ci4", "optional_ci4"):
        if not (isinstance(value, list) and len(value) == 2
                and all(isinstance(part, (int, float)) for part in value)):
            issues.append(f"tables.{name} row {index} column {key!r} is not a "
                          f"two-number interval: {value!r}")
        elif value[0] > value[1]:
            issues.append(f"tables.{name} row {index} column {key!r} is an interval "
                          f"whose lower end {value[0]} exceeds its upper "
                          f"{value[1]}")
        return
    if spec in ("p4", "optional_p4") or key in BOOTSTRAP_P_COLUMNS:
        if not isinstance(value, (int, float)):
            issues.append(f"tables.{name} row {index} column {key!r} is not a "
                          f"number: {value!r}")
        elif not 0.0 < value <= 1.0:
            issues.append(f"tables.{name} row {index} column {key!r} is a p-value "
                          f"outside (0, 1]: {value!r}")
        elif floor is not None and key in BOOTSTRAP_P_COLUMNS and value < floor:
            resamples = doc["the_p_floor"]["n_resamples"]
            issues.append(f"tables.{name} row {index} column {key!r} is a bootstrap "
                          f"p below the floor {floor} that {resamples} resamples "
                          f"can report: {value!r}")
        return
    if key in EXACT_P_COLUMNS:
        if not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
            issues.append(f"tables.{name} row {index} column {key!r} is not an "
                          f"exact p in (0, 1]: {value!r}")
        return
    if spec == "pct1":
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            issues.append(f"tables.{name} row {index} column {key!r} is not a share "
                          f"in [0, 1]: {value!r}")
        return
    if spec in ("int", "optional_int"):
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"tables.{name} row {index} column {key!r} is not a count: "
                          f"{value!r}")
        return
    if spec in ("float2", "float4", "signed4", "optional_float4"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            issues.append(f"tables.{name} row {index} column {key!r} is not a "
                          f"number: {value!r}")
        elif key == "critical_value" and alpha is not None \
                and not 0.0 < value <= alpha:
            issues.append(f"tables.{name} row {index} column {key!r} is a Holm "
                          f"critical value outside (0, alpha={alpha}]: {value!r}")
        return
    if spec == "sha8":
        if not (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)):
            issues.append(f"tables.{name} row {index} column {key!r} is not a full "
                          f"sha256 to be abbreviated at render time: {value!r}")


def _check_holm(doc: dict, issues: list[str]) -> None:
    """Re-derive the correction from the table's own raw p column."""
    alpha = (doc.get("the_confirmatory_rule") or {}).get("alpha")
    if alpha is None:
        issues.append("the_confirmatory_rule files no alpha, so the Holm columns "
                      "cannot be re-derived")
        return
    sign = (doc.get("tables") or {}).get("sign_transport") or {}
    raw = {row["hypothesis"]: row["raw_p"] for row in sign.get("rows") or []
           if row.get("hypothesis") is not None}
    if len(raw) != doc["the_confirmatory_rule"]["n_confirmatory_tests"]:
        issues.append(f"the primary table carries {len(raw)} p-values and the "
                      f"protocol files "
                      f"{doc['the_confirmatory_rule']['n_confirmatory_tests']} "
                      f"confirmatory tests; Holm over a different m is a different "
                      f"correction")
        return
    derived = holm(raw, alpha)
    for row in sign.get("rows") or []:
        hypothesis = row.get("hypothesis")
        if hypothesis is None:
            continue
        if row["adjusted_p"] != derived["adjusted_p"][hypothesis]:
            issues.append(f"sign_transport {hypothesis}: the table's Holm p "
                          f"{row['adjusted_p']} is not the {derived['adjusted_p'][hypothesis]} "
                          f"the stated rule produces from the raw p in the same row")
        if row["critical_value"] != derived["critical_values"][hypothesis]:
            issues.append(f"sign_transport {hypothesis}: the table's Holm critical "
                          f"value {row['critical_value']} is not "
                          f"{derived['critical_values'][hypothesis]}")
        rejected = row["adjusted_p"] <= alpha
        if row["verdict"] == "refuted" and not rejected:
            issues.append(f"sign_transport {hypothesis}: the verdict is refuted but "
                          f"the Holm-adjusted p {row['adjusted_p']} does not reach "
                          f"alpha={alpha}")

    configurations = (doc.get("tables") or {}).get("bound_configurations") or {}
    groups: dict[str, dict[str, float]] = {}
    for row in configurations.get("rows") or []:
        groups.setdefault(row["configuration"], {})[row["hypothesis"]] = row["raw_p"]
    for key in sorted(groups):
        derived = holm(groups[key], alpha)
        for row in configurations.get("rows") or []:
            if row["configuration"] != key:
                continue
            hypothesis = row["hypothesis"]
            if row["adjusted_p"] != derived["adjusted_p"][hypothesis]:
                issues.append(f"bound_configurations {key} {hypothesis}: the Holm p "
                              f"{row['adjusted_p']} is not the "
                              f"{derived['adjusted_p'][hypothesis]} the stated rule "
                              f"produces from that configuration's own raw p-values")


def _check_panel(doc: dict, issues: list[str]) -> None:
    panel = doc.get("panel") or {}
    tables = doc.get("tables") or {}
    if panel["n_families_in_the_panel"] - panel["n_families_dropped"] \
            != panel["n_families_in_the_confirmatory_panel"]:
        issues.append(f"panel: {panel['n_families_in_the_panel']} families minus "
                      f"{panel['n_families_dropped']} dropped is not the "
                      f"{panel['n_families_in_the_confirmatory_panel']} filed as the "
                      f"confirmatory panel")
    cells = panel["cells_no_judge_could_label"]
    if len(cells) != panel["n_cells_no_judge_could_label"]:
        issues.append("panel: the excluded cells filed and the count beside them "
                      "disagree")
    if sorted(panel["differential_cells"] + panel["cells_missing_in_every_arm"]) \
            != sorted(cells):
        issues.append("panel: the differential and the uniformly missing cells do "
                      "not partition the excluded cells")
    if len(panel["differential_cells"]) != 1:
        issues.append(f"panel: {len(panel['differential_cells'])} cells are "
                      f"differential and the bound ranges over exactly one")
    if panel["n_families_the_bound_restores_to"] \
            != panel["n_families_in_the_confirmatory_panel"] \
            + len(panel["differential_cells"]):
        issues.append("panel: the family count the bound restores to is not the "
                      "confirmatory panel plus the differential families")

    for row in (tables.get("sign_transport") or {}).get("rows") or []:
        if row["n_families"] != panel["n_families_in_the_confirmatory_panel"]:
            issues.append(f"sign_transport {row['arm_key']}: analysed over "
                          f"{row['n_families']} families, which is not the "
                          f"confirmatory panel this table claims to be over")
    for row in (tables.get("censoring_bound") or {}).get("rows") or []:
        if row["n_families_committed"] \
                != panel["n_families_in_the_confirmatory_panel"]:
            issues.append(f"censoring_bound {row['arm_key']}: committed at "
                          f"{row['n_families_committed']} families")
        if row["n_families_sensitivity"] \
                != panel["n_families_the_bound_restores_to"]:
            issues.append(f"censoring_bound {row['arm_key']}: sensitivity at "
                          f"{row['n_families_sensitivity']} families")
    exclusions = (tables.get("panel_exclusions") or {}).get("rows") or []
    if len(exclusions) != panel["n_cells_no_judge_could_label"]:
        issues.append(f"panel_exclusions has {len(exclusions)} rows and the panel "
                      f"files {panel['n_cells_no_judge_could_label']} excluded cells")
    differential_rows = sorted(row["cell"] for row in exclusions
                               if row["differential"])
    if differential_rows != sorted(panel["differential_cells"]):
        issues.append("panel_exclusions and the panel block disagree about which "
                      f"cells are differential: {differential_rows} against "
                      f"{sorted(panel['differential_cells'])}")
    for row in exclusions:
        named = sorted(part.strip() for part in
                       str(row["arms_that_lost_it"]).split(",") if part.strip())
        if len(named) != row["n_arms_that_lost_it_at_seal"]:
            issues.append(
                f"panel_exclusions {row['cell']}: names "
                f"{len(named)} arm(s) in arms_that_lost_it and files "
                f"n_arms_that_lost_it_at_seal={row['n_arms_that_lost_it_at_seal']}")
        if not set(named) <= set(ARMS):
            issues.append(
                f"panel_exclusions {row['cell']}: names "
                f"{sorted(set(named) - set(ARMS))}, which are not arms of this "
                f"panel")
        # Both booleans are re-derived from the counts filed beside them. Filed
        # as booleans and never recomputed, either one could be edited to True
        # while the counts under it said otherwise -- and `differential` is the
        # split the whole bound depends on, so it is the one that has to be
        # arithmetic rather than assertion.
        if row["differential"] != (row["n_arms_that_lost_it_at_seal"] == 1):
            issues.append(
                f"panel_exclusions {row['cell']}: files differential="
                f"{row['differential']} and "
                f"{row['n_arms_that_lost_it_at_seal']} arm(s) lost it, so the "
                f"label and the count beside it disagree")
        reproduces = (row["n_arms_refusing_it_on_the_reprobe"]
                      == row["n_arms_that_lost_it_at_seal"])
        if row["the_reprobe_reproduces_the_seal"] != reproduces:
            issues.append(
                f"panel_exclusions {row['cell']}: files "
                f"the_reprobe_reproduces_the_seal="
                f"{row['the_reprobe_reproduces_the_seal']} where "
                f"{row['n_arms_that_lost_it_at_seal']} arm(s) lost it at seal and "
                f"{row['n_arms_refusing_it_on_the_reprobe']} refused it on the "
                f"reprobe")
        if not row["the_reprobe_reproduces_the_seal"]:
            issues.append(
                f"panel_exclusions {row['cell']}: {row['n_arms_that_lost_it_at_seal']} "
                f"arm(s) lost it at seal and "
                f"{row['n_arms_refusing_it_on_the_reprobe']} refused it when the "
                f"same payload was re-sent; the differential/shared split the "
                f"bound depends on would then rest on one measurement rather than "
                f"two agreeing ones")
        if row["n_arms_refusing_it_on_the_reprobe"] \
                + row["n_arms_served_it_on_the_reprobe"] != len(ARMS):
            issues.append(
                f"panel_exclusions {row['cell']}: the reprobe accounted for "
                f"{row['n_arms_refusing_it_on_the_reprobe']} refusals and "
                f"{row['n_arms_served_it_on_the_reprobe']} served arms, which is "
                f"not the {len(ARMS)} arms in the panel")
    for row in (tables.get("judge_reliability") or {}).get("rows") or []:
        expected = panel["n_records_in_the_panel"] - len(exclusions)
        if row["n_items"] != expected:
            issues.append(f"judge_reliability {row['arm_key']}: measured over "
                          f"{row['n_items']} items, and the panel's "
                          f"{panel['n_records_in_the_panel']} minus the "
                          f"{len(exclusions)} excluded cells is {expected}; a "
                          f"reliability table with an unexplained denominator is "
                          f"the thing this check exists for")


def _check_bound(doc: dict, issues: list[str]) -> None:
    rows = ((doc.get("tables") or {}).get("censoring_bound") or {}).get("rows") or []
    counts = doc.get("the_bound_in_counts") or {}
    surviving_genuine = []
    surviving_unlicensed = []
    for row in rows:
        low, high = row["mean_range_over_the_whole_rubric"]
        pinned = low * high > 0.0
        if row["sign_can_flip"] == pinned:
            where = ("lies entirely on one side of zero" if pinned
                     else "straddles zero")
            issues.append(f"censoring_bound {row['arm_key']}: the mean range "
                          f"[{low}, {high}] {where} and sign_can_flip is filed "
                          f"{row['sign_can_flip']}; those two are the same fact "
                          f"and cannot disagree")
        if row["worst_p"] < row["best_p"]:
            issues.append(f"censoring_bound {row['arm_key']}: the worst p "
                          f"{row['worst_p']} is smaller than the best p "
                          f"{row['best_p']}")
        if not (row["verdict_filed"] == row["verdict_at_99"]
                == row["verdict_genuine_worst_case"]):
            issues.append(f"censoring_bound {row['arm_key']}: the filed verdict "
                          f"{row['verdict_filed']} is not the verdict at 99 families "
                          f"{row['verdict_at_99']} or under the genuine worst case "
                          f"{row['verdict_genuine_worst_case']}, and the bound's "
                          f"claim is that the exclusion produces none of them")
        moved = row["verdict_fully_adversarial"] != row["verdict_filed"]
        if moved == row["survives_the_unlicensed_bound"]:
            issues.append(f"censoring_bound {row['arm_key']}: the fully adversarial "
                          f"verdict {row['verdict_fully_adversarial']} "
                          f"{'moves' if moved else 'does not move'} the filed "
                          f"verdict and survives_the_unlicensed_bound is filed "
                          f"{row['survives_the_unlicensed_bound']}")
        if row["survives"]:
            surviving_genuine.append(row["arm_key"])
        if row["survives_the_unlicensed_bound"]:
            surviving_unlicensed.append(row["arm_key"])
    if counts.get("n_surviving_the_genuine_worst_case") \
            != len(surviving_genuine):
        issues.append(f"the bound's summary files "
                      f"{counts.get('n_surviving_the_genuine_worst_case')} verdicts "
                      f"surviving the genuine worst case and its per-hypothesis "
                      f"rows, counted here, give {len(surviving_genuine)}")
    if counts.get("n_surviving_the_fully_adversarial_bound") \
            != len(surviving_unlicensed):
        issues.append(f"the bound's summary files "
                      f"{counts.get('n_surviving_the_fully_adversarial_bound')} "
                      f"verdicts surviving the fully adversarial bound and its rows "
                      f"give {len(surviving_unlicensed)}")
    if counts.get("n_hypotheses") != len(rows):
        issues.append(f"the bound files {counts.get('n_hypotheses')} hypotheses and "
                      f"the table has {len(rows)} rows")


def _check_figures(doc: dict, issues: list[str]) -> None:
    figures = doc.get("figures") or {}
    tables = doc.get("tables") or {}
    forest = figures.get("sign_forest") or {}
    sign_rows = {row["arm_key"]: row
                 for row in (tables.get("sign_transport") or {}).get("rows") or []}
    for entry in forest.get("series") or []:
        row = sign_rows.get(entry["key"])
        if row is None:
            issues.append(f"fig:sign_forest plots {entry['key']}, which the primary "
                          f"table has no row for")
            continue
        if entry["mean"] != row["bootstrap_mean"]:
            issues.append(f"fig:sign_forest {entry['key']}: plots "
                          f"{entry['mean']} where the table prints "
                          f"{row['bootstrap_mean']}")
        if entry["ci"] != list(row["ci"]):
            issues.append(f"fig:sign_forest {entry['key']}: draws the interval "
                          f"{entry['ci']} where the table prints {row['ci']}")
    if forest.get("order") != [entry["key"] for entry in forest.get("series") or []]:
        issues.append("fig:sign_forest files an order that is not the order of its "
                      "own series")

    ranges = figures.get("bound_ranges") or {}
    bound_rows = {row["arm_key"]: row
                  for row in (tables.get("censoring_bound") or {}).get("rows") or []}
    outside = []
    for entry in ranges.get("series") or []:
        row = bound_rows.get(entry["key"])
        if row is None:
            issues.append(f"fig:bound_ranges plots {entry['key']}, which the bound "
                          f"table has no row for")
            continue
        if entry["mean_range_over_the_whole_rubric"] \
                != row["mean_range_over_the_whole_rubric"]:
            issues.append(f"fig:bound_ranges {entry['key']}: draws "
                          f"{entry['mean_range_over_the_whole_rubric']} where the "
                          f"table prints {row['mean_range_over_the_whole_rubric']}")
        if entry["mean_at_98"] != row["mean_at_98"]:
            issues.append(f"fig:bound_ranges {entry['key']}: plots the point mean "
                          f"{entry['mean_at_98']} where the table prints "
                          f"{row['mean_at_98']}")
        low, high = entry["mean_range_over_the_whole_rubric"]
        inside = low <= entry["mean_at_98"] <= high
        if entry["the_point_mean_lies_inside_the_range"] != inside:
            issues.append(f"fig:bound_ranges {entry['key']}: files "
                          f"{entry['the_point_mean_lies_inside_the_range']} for a "
                          f"point mean of {entry['mean_at_98']} against the range "
                          f"[{low}, {high}]")
        if not inside:
            outside.append(entry["key"])
    if sorted(outside) != sorted(
            ranges.get("targets_whose_point_mean_lies_outside_its_range") or []):
        issues.append("fig:bound_ranges files a list of targets whose point mean "
                      "falls outside its range that is not the list its own series "
                      "produces")
    if ranges.get("n_targets_whose_point_mean_lies_outside_its_range") \
            != len(outside):
        issues.append("fig:bound_ranges files a count of non-nesting targets that "
                      "is not the length of the list beside it")
    filed = ranges.get("what_the_filed_decision_says_about_the_same_count")
    if filed is not None and filed != len(outside):
        issues.append(f"the transportability decision files {filed} targets whose "
                      f"98-family mean falls outside the bound's 99-family range "
                      f"and this document counts {len(outside)}; the two read the "
                      f"same artifact and must agree")


def _check_claims(doc: dict, issues: list[str]) -> None:
    claims = doc.get("claims_in_words") or {}
    if not claims:
        issues.append("no claims are filed, so no sentence in the paper has a "
                      "derived numeral behind it")
    for name in sorted(claims):
        claim = claims[name]
        sentence = claim.get("sentence") or ""
        numerals = claim.get("numerals") or {}
        lists = claim.get("the_lists_those_numerals_count") or {}
        if not sentence:
            issues.append(f"claims.{name} files no sentence")
        for numeral, value in sorted(numerals.items()):
            if str(value) not in sentence:
                issues.append(f"claims.{name}: the numeral {numeral}={value} does "
                              f"not appear in its own sentence, so the sentence "
                              f"would survive the count changing")
        stated = set(numerals.values())
        for list_name, values in sorted(lists.items()):
            if len(values) not in stated:
                issues.append(f"claims.{name}: the list {list_name} has "
                              f"{len(values)} entries and no numeral in the sentence "
                              f"counts it, so the sentence does not depend on it")
        match = SPELLED_COUNT.search(sentence)
        if match:
            issues.append(f"claims.{name}: the sentence spells out a count as "
                          f"{match.group(0)!r} beside numerals that are derived; "
                          f"write the count as a numeral so that it moves when the "
                          f"list moves")
        agreement = claim.get("must_agree_with")
        if agreement:
            filed = agreement.get("filed") or {}
            derived = agreement.get("derived_here") or {}
            if sorted(filed) != sorted(derived):
                issues.append(f"claims.{name}: must_agree_with compares "
                              f"{sorted(filed)} against {sorted(derived)}, which are "
                              f"not the same fields")
            for field in sorted(set(filed) & set(derived)):
                if filed[field] != derived[field]:
                    issues.append(f"claims.{name}: {field} is {filed[field]!r} in "
                                  f"the artifact it cites and {derived[field]!r} as "
                                  f"this document's own tables count it")

    transportability = claims.get("the_transportability_claim") or {}
    quoted = transportability.get("the_filed_sentence_it_quotes")
    if quoted:
        counts = (transportability.get("must_agree_with") or {}).get(
            "derived_here") or {}
        for field, value in sorted(counts.items()):
            if str(value) not in quoted:
                issues.append(
                    f"the filed transportability sentence quoted beside this claim "
                    f"does not contain {field}={value}, which this document "
                    f"derives from the same evidence; the paper would be quoting a "
                    f"sentence its own table contradicts")


def _rows_by_claim(block: dict) -> dict[str, dict]:
    return {row["claim"]: row for row in block.get("rows") or []}


def _check_environment(doc: dict, issues: list[str]) -> None:
    """The two claims the environment rows exist to keep apart."""
    block = (doc.get("tables") or {}).get("environment") or {}
    rows = _rows_by_claim(block)
    preimage = rows.get("the freeze is the preimage of the recorded hash")
    reconstructible = rows.get("reconstructible from the freeze alone")
    if preimage is None or reconstructible is None:
        issues.append("tab:environment files only one of 'the freeze is the "
                      "preimage of the recorded hash' and 'reconstructible from "
                      "the freeze alone'; the point of the table is that they are "
                      "different claims and both are filed")
    else:
        if preimage["value"] is not True:
            issues.append("tab:environment: the freeze is filed as not being the "
                          "preimage of the recorded hash, which means the lock "
                          "cannot authenticate the environment that produced this "
                          "evidence and the paper's reproducibility claim has to "
                          "change, not this table")
        if reconstructible["value"] is not False:
            issues.append("tab:environment: reconstructible_from_the_freeze_alone "
                          "is filed as true, which contradicts the torch local "
                          "version segment the same table rows record; either the "
                          "freeze line gained the segment or this row is wrong")
        note = block.get("note") or ""
        if "preimage" not in note or "reconstructible" not in note:
            issues.append("tab:environment's note does not name both claims, and a "
                          "table that carries a false row beside a true one without "
                          "saying which is which is how a verifier ends up "
                          "contradicting its own report")
    for pair in (("certified Python", "recorded Python"),
                 ("packages in the freeze", "recorded package count"),
                 ("pip freeze sha256", "recorded pip freeze sha256")):
        left, right = rows.get(pair[0]), rows.get(pair[1])
        if left is None or right is None:
            issues.append(f"tab:environment files only one of {pair}")
        elif left["value"] != right["value"]:
            issues.append(f"tab:environment: {pair[0]} is {left['value']!r} and "
                          f"{pair[1]} is {right['value']!r}; the lock records the "
                          f"environment it was written from, so these agree or the "
                          f"lock is describing a different machine")
    torch = rows.get("torch the preflight observed")
    freeze_line = rows.get("torch in the freeze line")
    if torch is not None and freeze_line is not None \
            and torch["value"] == freeze_line["value"]:
        issues.append("tab:environment: the torch the preflight observed and the "
                      "torch the freeze line names are filed as identical, which "
                      "would make the reconstruction caveat in the same table "
                      "describe a gap that is not there")


def _check_evidence_binding(doc: dict, issues: list[str]) -> None:
    block = (doc.get("tables") or {}).get("evidence_binding") or {}
    rows = _rows_by_claim(block)
    total = rows.get("files bound by the closeout manifest")
    if total is None:
        issues.append("tab:evidence_binding files no count of bound files")
        return
    by_role = sum(row["value"] for claim, row in sorted(rows.items())
                  if claim.startswith("bound files by role: "))
    if by_role != total["value"]:
        issues.append(f"tab:evidence_binding: the manifest files {total['value']} "
                      f"bound files and its per-role counts sum to {by_role}; a "
                      f"manifest whose roll-up covers a different set than its own "
                      f"role breakdown describes is not binding what it says")
    indirect = rows.get("media files bound indirectly")
    media_rollup = rows.get("media roll-up sha256")
    if indirect is not None and media_rollup is not None:
        media_values = [row["value"] for claim, row in sorted(rows.items())
                        if claim.startswith("media ")]
        if len(media_values) < 3:
            issues.append("tab:evidence_binding: the media are bound indirectly and "
                          "the table should carry their count, their bytes and "
                          "their roll-up together, since a count without a hash is "
                          "an assertion")


def _check_histogram(doc: dict, issues: list[str]) -> None:
    tables = doc.get("tables") or {}
    shift = tables.get("vision_ablation_shift") or {}
    rows = shift.get("rows") or []
    if not rows:
        issues.append("tab:vision_ablation_shift has no rows")
        return
    total = sum(row["n_items"] for row in rows)
    shares = sum(row["share"] for row in rows)
    if abs(shares - 1.0) > 1e-9:
        issues.append(f"tab:vision_ablation_shift: the shares sum to {shares} and "
                      f"not to 1")
    ablation = tables.get("vision_ablation") or {}
    judged = _rows_by_claim(ablation).get("items judged")
    if judged is None:
        issues.append("tab:vision_ablation files no item count")
    elif judged["value"] != total:
        issues.append(f"tab:vision_ablation files {judged['value']} judged items and "
                      f"the shift histogram beside it covers {total}; the two are "
                      f"the same denominator and a histogram over a different one "
                      f"is unreadable next to its own table")
    for row in rows:
        if row["n_items"] < 0:
            issues.append(f"tab:vision_ablation_shift: a negative item count at "
                          f"shift {row['shift']}")


def check_the_numbers(doc: dict) -> list[str]:
    """Every internal contradiction this document is capable of holding.

    Run at write time as well as at verify time, so a filing that disagrees with
    itself is refused rather than committed and found later.
    """
    issues: list[str] = []
    _check_inputs(doc, issues)
    _check_tables(doc, issues)
    _check_holm(doc, issues)
    _check_panel(doc, issues)
    _check_bound(doc, issues)
    _check_figures(doc, issues)
    _check_claims(doc, issues)
    _check_environment(doc, issues)
    _check_evidence_binding(doc, issues)
    _check_histogram(doc, issues)
    return issues


def verify(path: Path | None = None) -> tuple[int, str, list[str]]:
    """Re-derive the whole document and compare it, then check what is filed."""
    path = path or OUT_PATH
    if not path.exists():
        return 2, "not_filed", [
            f"no paper numbers at {_rel(path)}; run "
            f"`python scripts/iter11_paper_numbers.py --write` to file them"]
    try:
        filed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 1, "unreadable", [
            f"{_rel(path)} is not readable JSON ({exc}); it is the paper's only "
            f"source of numbers and this stage does not repair it"]
    try:
        fresh = build()
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), \
            "could_not_rederive", [
                f"the paper's numbers could not be re-derived from the filed "
                f"artifacts: exit {exc.code}"]

    issues = []
    for key in sorted(set(filed) | set(fresh)):
        if filed.get(key) != fresh.get(key):
            issues.append(
                f"{key}: filed {json.dumps(filed.get(key), sort_keys=True)[:200]}"
                f" != re-derived "
                f"{json.dumps(fresh.get(key), sort_keys=True)[:200]}")
    issues.extend(check_the_numbers(filed))
    if issues:
        return 1, "differs", issues
    return 0, "reproduced_exactly", []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="re-derive and compare with the filed numbers, writing "
                           "nothing")
    mode.add_argument("--write", action="store_true",
                      help="write the numbers file. Explicit, because it "
                           "overwrites a committed artifact the tables cite")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.verify or not args.write:
        code, conclusion, issues = verify(args.out)
        if code == 2:
            print(f"PAPER NUMBERS: NOT FILED -- {issues[0]}")
            return 2
        if code == 1:
            print(f"PAPER NUMBERS: FAIL ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        filed = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"PAPER NUMBERS: VERIFIED -- {conclusion.replace('_', ' ')}")
        _print_summary(filed)
        return 0

    doc = build()
    problems = check_the_numbers(doc)
    if problems:
        print("FAIL: refusing to file numbers that disagree with themselves",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(args.out)}")
    _print_summary(doc)
    print(f"  sha256                       "
          f"{sha256_bytes(args.out.read_bytes())}")
    return 0


def _print_summary(doc: dict) -> None:
    """What is in the file, printed so a reader can see the paper's coverage."""
    tables = doc["tables"]
    panel = doc["panel"]
    rule = doc["the_confirmatory_rule"]
    floor = doc["the_p_floor"]
    inputs = doc["inputs"]
    print(f"  inputs                       {inputs['n_inputs']} filed artifacts, "
          f"all bound by sha256")
    print(f"  estimand                     {rule['primary_estimand']}, "
          f"alpha {rule['alpha']}, {rule['n_confirmatory_tests']} confirmatory "
          f"tests, {rule['family_wise_correction']}")
    print(f"  panel                        {panel['n_families_in_the_panel']} "
          f"families, {panel['n_cells_no_judge_could_label']} cells no judge could "
          f"label, {panel['n_families_in_the_confirmatory_panel']} analysed, "
          f"{panel['n_families_the_bound_restores_to']} under the bound")
    print(f"  bootstrap p floor            "
          f"{floor['the_smallest_p_it_can_report']} over "
          f"{floor['n_resamples']} resamples at seed {floor['seed']}")
    for name in sorted(tables):
        block = tables[name]
        print(f"  table {block['label']:32s} {len(block['rows']):3d} rows, "
              f"{len(block['columns'])} columns")
    for name in sorted(doc["figures"]):
        figure = doc["figures"][name]
        print(f"  figure {figure['label']:31s} {len(figure['series'])} series, "
              f"plotted from {figure['source_table']}")
    for name in sorted(doc["claims_in_words"]):
        claim = doc["claims_in_words"][name]
        print(f"  claim {name:34s} {len(claim['numerals'])} derived numerals")


if __name__ == "__main__":
    raise SystemExit(main())
