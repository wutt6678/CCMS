#!/usr/bin/env python3
"""File the transportability decision, measured out of the sealed analysis.

The Iteration 10 result is a positive Delta_TV in Qwen3.5-9B: multimodal
censoring costs the model something, and the direction of that cost was the
finding. Iteration 11 replayed the same frozen panel through four more targets
and asked, as its confirmatory family, whether the sign holds in each. It holds
in one.

That makes the effect MODEL-SPECIFIC and not generally transported, and this
stage files that as a decision with its measurement attached rather than leaving
it to be inferred from a table somebody reads in a particular mood. Three of the
four targets reverse the sign, and they reverse it decisively rather than
ambiguously: each mean is negative, each is rejected by the frozen
Holm-Bonferroni rule with the verdict ``refuted``, and two of the three sit at
the bootstrap p floor of 1/5000 -- so the reversal is a result and not a null.

One trap in the evidence is worth naming here, because the first draft of this
stage fell into it. The bound's ``for_every_admissible_score`` asks whether a
target's sign MATCHES THE REFERENCE at every score the one differentially missing
cell could take. It is therefore False in exactly the three targets that reverse,
and reading that column alone says "three of four failed to survive the
sensitivity" when it says the opposite: the reversal holds at every admissible
score. Whether a sign can FLIP is a separate question, answered from the mean's
range over the whole rubric, and in all four targets that range lies entirely on
one side of zero. Two fields, two facts, and a prose paragraph that is checked
against both rather than trusted.

The same discipline applies to the sentence the document files as its decision.
Every numeral in it is ``len()`` of a list filed beside it, so a count that moves
moves the sentence with it; a count written out in words beside a derived numeral
reads as prose until the numeral changes, and after that it is a claim the
document no longer supports. Both branches of the decision rule are written out,
including the one this evidence did not produce.

The decision is filed as its own artifact for two reasons. It is the claim a
reader is most likely to over-generalise from -- "multimodal censoring hurts
safety" is the sentence the 9B result invites and the four-target result
refuses -- and a claim that is only implied by a table gets stated in prose
without its denominator. And it is the claim the differential-censoring bound
has to be robust for, so both are filed where they can be compared.

Nothing here is transcribed. Every number is READ out of
``cross_model_analysis.json`` and ``differential_censoring_bound.json``, and the
decision is DERIVED from what is read: the sign comparison decides the verdict
per target, the count of matching targets decides the transportability call, and
a filing whose own inputs would support the opposite call exits 1 rather than
writing.

The rule that turns four per-target verdicts into one label is DETERMINISTIC and
POST-SPECIFIED, and the second word is measured rather than claimed. The frozen
protocol pre-specifies the estimand, four per-target sign-match statements,
Holm-Bonferroni over those four tests and a retention clause; it files no rule
for aggregating the four verdicts into a transportability label. So the label is
the conjunction of results that were pre-specified, reached by a rule that was
not, and it does not carry the protection a pre-registered decision rule
carries. ``--verify`` repeats the search for the rule string in the protocol
document: a hit means the protocol does pre-specify it and this document's claim
has to change, not the search.

Re-derivable exactly: the document carries no timestamp, no commit and no tree
state, so ``--verify`` rebuilds it from the same three inputs and compares the
WHOLE thing rather than a subset somebody remembered to exclude.

Exit codes, and each one is tested:
    0  verified, or filed
    1  a re-derivation disagrees with what is filed, or the inputs would not
       support the decision this script files
    2  nothing filed to verify against, or an input is missing
    3  not reachable for this stage: every input is committed, so no checkout
       that can run it lacks a section of it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.seeds import sha256_bytes  # noqa: E402

OUT_PATH = REPO_ROOT / "outputs" / "iteration_11" / "closeout" \
    / "iteration_11_transportability_decision.json"

CROSS_MODEL = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "cross_model" / "cross_model_analysis.json"
BOUND = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "differential_censoring" / "differential_censoring_bound.json"
PROTOCOL = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"

#: The four confirmatory targets, in the order the protocol's model matrix lists
#: them. Read out of the analysis rather than written down here, so this is only
#: the expected set and a fifth or missing target is a finding.
EXPECTED_TARGETS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")

#: The decision this stage exists to file. It is compared against what the
#: evidence says rather than assumed by it: :func:`decide` derives the call from
#: the sign comparison and :func:`build` refuses to file a document whose own
#: numbers would support a different one.
DECISION_MODEL_SPECIFIC = "model_specific_not_generally_transported"
DECISION_TRANSPORTED = "generally_transported_across_the_targets_tested"

#: The classification rule, stated once so that the text filed beside the
#: decision and the search that measures whether the protocol pre-specifies it
#: are the same string and cannot drift apart.
THE_CLASSIFICATION_RULE = (
    "'generally transported' requires EVERY target tested to carry the reference "
    "sign; one reversal makes the effect model-specific")

#: The only two answers "when was this rule specified?" can have here. A closed
#: set rather than a free string so that the filed label has to be one somebody
#: has written down the meaning of, and so that moving from one to the other is
#: a change of claim that the check notices rather than a rewording.
SPECIFIED_POST = "post_specified_descriptive"
SPECIFIED_PRE = "pre_specified"
SPECIFICATION_LABELS = (SPECIFIED_POST, SPECIFIED_PRE)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sign(value: float | None) -> str | None:
    """Which side of zero a mean is on, or ``"zero"``, or None if there is none.

    Zero is its own answer rather than being folded into a neighbour. A mean
    range with an endpoint AT zero is not a mean that holds its sign over the
    whole range: it is a mean that touches the null, and calling that
    "positive" or "negative" would make a razor edge look like a margin.
    """
    if value is None:
        return None
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    ``raise SystemExit("text")`` exits 1 whatever the text says, and 1 is this
    script's code for a DISAGREEMENT with the filed decision. A missing input
    reported as a disagreement sends whoever reads the exit code -- including a
    gate lane that branches on it -- to look for a contradiction in an artifact
    that was never there to contradict.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path, what: str) -> dict:
    if not path.exists():
        fatal(f"no {what} at {_rel(path)}; this stage reads its numbers out of "
              f"that file and transcribes none of them", 2)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_of(analysis: dict) -> dict:
    """What the sealed Iteration 10 reference says, read not quoted."""
    reference = analysis.get("reference") or {}
    delta = reference.get("Delta_TV") or {}
    sign = reference.get("sign")
    if sign not in ("positive", "negative"):
        fatal(f"the committed analysis records the reference sign as {sign!r}, "
              f"which is neither positive nor negative, so there is nothing for "
              f"a target's sign to match or reverse", 1)
    return {
        "model": reference.get("model"),
        "status": reference.get("status"),
        "n_families": reference.get("n_families"),
        "sign": sign,
        "mean": delta.get("bootstrap_mean"),
        "ci": [delta.get("ci_lower"), delta.get("ci_upper")],
        "source": reference.get("source"),
        "source_sha256": reference.get("source_sha256"),
    }


def the_exclusion(analysis: dict, bound: dict) -> dict:
    """What is missing from the panel, and which part of that is differential.

    The confirmatory panel is 98 of 100 families, and it is 98 for two
    separate reasons. One family, ``CMST_795308``, lost two cells in EVERY arm
    -- its judge provider refused them all four times -- so its absence is
    missingness the four targets share and is not differential censoring at
    all. The other, ``CMST_456921``, lost one cell, ``text_only``, in ONE arm
    of four, and that is the cell the bound ranges over. Conflating the two
    would file a decision about differential censoring against a panel whose
    size was mostly decided by something else.

    Every count here is read out of the committed artifacts and the
    differential/shared split is DERIVED -- a cell is differential when exactly
    one arm lacks it -- rather than written down from memory of which family
    the refusal happened in.
    """
    common = analysis.get("common_panel") or {}
    per_target_cells = common.get("per_target_excluded_cells") or {}
    cells = sorted(common.get("union_excluded_cells") or [])
    differential = sorted(
        cell for cell in cells
        if sum(cell in (per_target_cells.get(arm) or [])
               for arm in per_target_cells) == 1)
    shared = sorted(set(cells) - set(differential))
    restrictions = {arm: (row.get("panel_restriction") or {})
                    for arm, row in (analysis.get("per_model") or {}).items()}
    rules = sorted({r.get("rule") for r in restrictions.values()
                    if r.get("rule")})
    bounded_cell = bound.get("cell")
    if bounded_cell not in differential:
        fatal(f"the committed bound ranges over {bounded_cell!r}, which the "
              f"analysis does not record as missing in exactly one arm "
              f"(differential cells: {differential}), so the bound and the "
              f"panel it bounds disagree about what is missing", 1)
    return {
        "n_families_in_the_panel": common.get("n_families_in_panel"),
        "n_families_in_the_confirmatory_panel": common.get("n_families_common"),
        "n_families_the_bound_restores_to": bound.get("n_families_sensitivity"),
        "cells_no_judge_could_label": cells,
        "missing_in_one_arm_only_and_therefore_differential": differential,
        "missing_in_every_arm_and_therefore_not_differential": shared,
        "the_cell_the_bound_ranges_over": bounded_cell,
        "the_arm_that_lost_it": bound.get(
            "the_arm_whose_reply_caused_the_refusal"),
        "the_family_still_out_of_the_bound": bound.get("family_still_excluded"),
        "why_the_bound_stops_short_of_the_full_panel":
            bound.get("why_the_sensitivity_stops_at_99"),
        "families_dropped": sorted(common.get("families_dropped") or []),
        "why_losing_one_cell_costs_a_whole_family":
            rules[0] if len(rules) == 1 else rules,
        "what_the_bound_is_and_is_not": bound.get("what_this_does_not_do"),
    }


def per_target(analysis: dict, bound: dict) -> dict:
    """Each target's sign, its verdict, and what the bound says about both.

    Two different facts come out of the bound, and they are filed as two
    fields because conflating them is exactly how a reversal gets misread as a
    failure of the decision. ``for_every_admissible_score`` asks whether the
    sign MATCHES THE REFERENCE at every score the missing cell could take, so
    it is False for a target that reverses, and that False IS the reversal --
    it is not a hole in it. Whether the sign can FLIP at all is a separate
    question, answered from the mean's range over the rubric: a range entirely
    on one side of zero pins that target to that side at every admissible
    score, which is what makes the reversal a result rather than a coin toss
    the missing label could have landed differently on.
    """
    hypotheses = {row["arm"]: (hid, row)
                  for hid, row in ((bound.get("verdicts") or {})
                                   .get("per_hypothesis") or {}).items()}
    models = analysis.get("per_model") or {}
    unexpected = sorted(set(models) - set(EXPECTED_TARGETS))
    missing = sorted(set(EXPECTED_TARGETS) - set(models))
    if unexpected or missing:
        fatal(f"the committed analysis covers {sorted(models)}, not the four "
              f"confirmatory targets {sorted(EXPECTED_TARGETS)}"
              + (f"; unexpected: {unexpected}" if unexpected else "")
              + (f"; missing: {missing}" if missing else ""), 1)

    verdicts = analysis.get("verdicts") or {}
    by_arm = {row["model"]: (hid, row) for hid, row in verdicts.items()}
    out = {}
    for arm in EXPECTED_TARGETS:
        model = models[arm]
        estimand = (model.get("estimands") or {}).get(
            analysis.get("primary_estimand") or "Delta_TV") or {}
        hid, verdict = by_arm.get(arm, (None, {}))
        if arm not in hypotheses:
            fatal(f"the committed bound files no per-hypothesis row for {arm}, "
                  f"so nothing decides whether its sign survives the range of "
                  f"the missing label", 1)
        bounded = hypotheses[arm][1] or {}
        sign_block = bounded.get("sign_matches_the_99_family_reference") or {}
        p_block = bounded.get("p_over_the_whole_rubric") or {}
        mean_range = sign_block.get("mean_range_over_the_whole_rubric")
        if (not isinstance(mean_range, (list, tuple)) or len(mean_range) != 2):
            fatal(f"{arm}: the committed bound files "
                  f"mean_range_over_the_whole_rubric as {mean_range!r}, which "
                  f"is not the two endpoints of an interval", 1)
        low, high = _sign(mean_range[0]), _sign(mean_range[1])
        pinned = low is not None and low == high and low != "zero"
        out[arm] = {
            "hypothesis": hid,
            "statement_tested": verdict.get("statement"),
            "sign": model.get("sign"),
            "sign_matches_the_reference": model.get("sign_matches_reference"),
            "mean": estimand.get("bootstrap_mean"),
            "mean_is_over": (
                f"{model.get('n_families')} families, the restricted "
                f"confirmatory panel; this is the bootstrap mean, so it is not "
                f"the same quantity as the bound's mean range below and the "
                f"one is not required to contain the other"),
            "ci": [estimand.get("ci_lower"), estimand.get("ci_upper")],
            "n_families": model.get("n_families"),
            "raw_p": verdict.get("raw_p"),
            "adjusted_p": verdict.get("adjusted_p"),
            "verdict": verdict.get("verdict"),
            "the_mean_cannot_flip_sign_over_the_whole_rubric": pinned,
            "the_sign_the_mean_cannot_leave": low if pinned else None,
            "the_reference_sign_holds_at_every_admissible_score":
                sign_block.get("for_every_admissible_score"),
            "mean_range_over_the_whole_rubric": list(mean_range),
            "the_mean_range_is_over": (
                f"{bound.get('n_families_sensitivity')} families -- the "
                f"confirmatory panel with {bound.get('cell')} restored and its "
                f"score taken as x over {bound.get('score_range_bounded')} -- "
                f"so it divides by a different n than the 98-family mean above "
                f"and the two do not nest"),
            "worst_p_over_the_whole_rubric": p_block.get("worst_p"),
        }
    return out


def marketed_label(matrix_row: dict | None) -> str | None:
    """The name a protocol hypothesis would use for a model, out of its matrix row.

    The hypotheses are prose -- "DeltaTV sign in Ministral-3-3B matches the
    Iteration 10 sign" -- and the model matrix is keyed ``ministral3_3b``, so
    nothing joins them by equality. The marketed label is the closest thing the
    protocol files to the string the prose uses, and it is read rather than
    written down here for the same reason every other name in this stage is.
    """
    return ((matrix_row or {}).get("size_metadata") or {}).get("marketed_label")


def which_hypotheses_name_a_target(protocol: dict) -> tuple[dict, dict]:
    """Split the protocol's hypotheses into per-target ones and the rest.

    Derived rather than listed, and derived as a BIJECTION: a hypothesis counts
    as per-target when exactly one of the four confirmatory targets' marketed
    labels appears in it. Everything else -- the pooled statement H5, which is a
    claim about five signs rather than about one model's -- lands in the second
    dict with the targets it named, so an ambiguous or unnameable statement is
    filed as a finding instead of being guessed at.

    This matters because the post-specification argument rests on the rule being
    the conjunction of the statements the protocol does pre-specify, and a
    conjunction of five statements, one of which is pooled over models the rule
    does not aggregate, is not the rule this stage applies.
    """
    hypotheses = protocol.get("hypotheses") or {}
    matrix = {row.get("model_key"): row
              for row in (protocol.get("model_matrix") or [])}
    labels = {arm: marketed_label(matrix.get(arm)) for arm in EXPECTED_TARGETS}
    per_target: dict = {}
    not_per_target: dict = {}
    for key in sorted(k for k in hypotheses if k.startswith("H")):
        statement = hypotheses[key]
        text = statement if isinstance(statement, str) else json.dumps(statement)
        named = sorted(arm for arm, label in labels.items()
                       if label and label in text)
        if len(named) == 1:
            per_target[key] = {
                "target": named[0],
                "named_by_the_label": labels[named[0]],
                "statement": statement}
        else:
            not_per_target[key] = {
                "statement": statement,
                "targets_it_names": named,
                "why_it_is_not_one_of_the_four": (
                    "it names no confirmatory target" if not named else
                    "it names more than one, so it is not a statement about a "
                    "single model's sign")}
    return per_target, not_per_target


def how_the_rule_was_specified(protocol: dict) -> dict:
    """Whether the frozen protocol pre-specified this classification. It did not.

    Measured rather than asserted, because the two are easy to confuse and the
    difference is the whole claim. The protocol freezes the iteration's four
    per-model hypotheses -- each one "DeltaTV sign in X matches the Iteration 10
    sign" -- the estimand, Holm-Bonferroni over the four confirmatory tests, alpha
    and a retention clause that forbids dropping an unfavourable result. What it
    does not contain is any rule for turning four per-model verdicts into one
    transportability label, so the label this document files is a deterministic
    summary of results that WERE pre-specified, arrived at by a rule that was
    not.

    The check is mechanical: the rule string is searched for in the protocol
    document as filed. A hit means the protocol does pre-specify it and this
    document's claim has to change, not the search.
    """
    frozen = protocol.get("protocol") or {}
    hypotheses = protocol.get("hypotheses") or {}
    multiplicity = protocol.get("multiplicity") or {}
    per_target, not_per_target = which_hypotheses_name_a_target(protocol)
    text = json.dumps(protocol, indent=2, ensure_ascii=False)
    return {
        "the_rule": THE_CLASSIFICATION_RULE,
        "specified_when": SPECIFIED_POST,
        "what_that_means": (
            "the rule is a deterministic function of the four per-model results "
            "and contains no threshold, no proportion and no discretion, so two "
            "readers of the same table get the same label. It was not written "
            "down before the table existed, so it carries none of the protection "
            "a pre-registered decision rule carries, and this document does not "
            "claim that it does"),
        "the_protocol_document": {
            "path": _rel(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "name": frozen.get("name"),
            "status": frozen.get("status"),
            "frozen_before_any_target_generation":
                frozen.get("frozen_before_any_target_generation"),
            "frozen_at_code_commit": frozen.get("frozen_at_code_commit"),
        },
        "what_the_protocol_pre_specifies": {
            "sign_convention": hypotheses.get("sign_convention"),
            "per_target_statements": per_target,
            "n_per_target_statements": len(per_target),
            "statements_that_are_not_about_one_target": not_per_target,
            "how_the_split_was_derived": (
                "a hypothesis counts as per-target when exactly one of the four "
                "confirmatory targets' marketed labels appears in it, read out "
                "of the protocol's own model_matrix. The pooled statement names "
                "none of them and is filed separately rather than folded in, "
                "because the rule below is a conjunction over single-model "
                "signs and a pooled sign is not one of them"),
            "multiplicity": dict(sorted(multiplicity.items())),
            "retention": hypotheses.get("retention"),
        },
        "the_rule_is_the_conjunction_of_those_statements": {
            "claim": (
                "the classification this document files asks whether every one "
                "of the per-target statements the protocol froze came out, "
                "which is their conjunction and adds no threshold, proportion "
                "or discretion of its own"),
            "n_statements_conjoined": len(per_target),
            "n_confirmatory_tests_the_protocol_declares":
                multiplicity.get("n_confirmatory_model_tests"),
            "n_targets_this_document_measured": len(EXPECTED_TARGETS),
            "the_three_counts_agree":
                len(per_target)
                == multiplicity.get("n_confirmatory_model_tests")
                == len(EXPECTED_TARGETS),
            "targets_covered": sorted(
                entry["target"] for entry in per_target.values()),
            "every_target_is_covered_exactly_once": sorted(
                entry["target"] for entry in per_target.values()
            ) == sorted(EXPECTED_TARGETS),
            "what_the_conjunction_leaves_out": (
                "the pooled statement, which the protocol also froze and which "
                "this document does not classify: pooling five signs into one "
                "is the estimate over a population that "
                "what_this_decision_does_not_say refuses to licence"),
        },
        "what_the_protocol_does_not_pre_specify": (
            "any rule that aggregates the per-model verdicts into one "
            "transportability label. Each hypothesis the protocol files is a "
            "statement about ONE model's sign, and the protocol's own name for "
            "the iteration is a topic, not a decision rule"),
        "the_rule_is_not_a_quotation_from_the_protocol":
            THE_CLASSIFICATION_RULE not in text,
        "how_that_is_measured": (
            "the rule string is searched for in the protocol document as filed, "
            "and --verify repeats the search. A hit would mean the protocol does "
            "pre-specify the classification and this block is wrong, so the claim "
            "is a measurement with a way to fail rather than a characterisation"),
        "why_the_conclusion_is_defensible_anyway": (
            "the rule is the conjunction of the four per-target statements the "
            "protocol does pre-specify, each tested under the correction the "
            "protocol pre-declared, and the retention clause the protocol froze "
            "required the reversals to be reported whatever label was put on "
            "them. What post-specification costs is the ability to say the "
            "LABEL could not have been chosen to suit the table -- and with "
            "four verdicts, one matching and three refuted, any label that "
            "suited this table would have to say the same thing about the same "
            "three reversals"),
        "what_would_make_it_pre_specified": (
            "the rule written into a protocol frozen before any target was "
            "replayed. This iteration's protocol was frozen at "
            f"{frozen.get('frozen_at_code_commit')} and could have carried it; "
            "it does not, and a later iteration that wants a pre-registered "
            "transportability claim has to put the rule there"),
    }


def decide(reference: dict, targets: dict) -> dict:
    """The transportability call, derived from the sign comparison.

    The rule is DETERMINISTIC and POST-SPECIFIED, and the two words matter
    separately. Deterministic: "transported" only if EVERY target tested carries
    the reference sign, "model-specific" if any one of them does not, with
    nothing in between, because anything in between would be a claim about a
    proportion and a proportion of four targets is not a rate. Post-specified:
    the frozen protocol pre-specifies the per-model sign-match statements H1-H4,
    the estimand, the Holm-Bonferroni correction over four tests and the
    retention clause, but it files no rule that aggregates four per-model verdicts
    into one transportability label. This rule is the conjunction of the four
    statements the protocol does pre-specify, which is why the conclusion is
    defensible -- but a conjunction nobody wrote down before seeing the table is
    not a pre-registered decision rule, and calling it one would claim a
    protection the evidence does not have. :func:`how_the_rule_was_specified`
    measures the distinction against the protocol document instead of asserting
    it.
    """
    reference_sign = reference["sign"]
    matching = sorted(arm for arm, t in targets.items()
                      if t["sign"] == reference_sign)
    reversing = sorted(arm for arm, t in targets.items()
                       if t["sign"] != reference_sign)
    undecided = sorted(arm for arm, t in targets.items()
                       if t["sign"] not in ("positive", "negative"))
    if undecided:
        fatal(f"{', '.join(undecided)} record a sign that is neither positive "
              f"nor negative, so the count of targets carrying the reference "
              f"sign cannot be taken", 1)
    transported = not reversing

    # What "the decision survives the exclusion" actually has to mean, since the
    # bound's own field cannot carry it alone: for_every_admissible_score asks
    # about MATCHING, so it is False for every reversing target and True for
    # none-but-the-matcher, and reading that as "three of four failed to
    # survive" would invert the evidence. The decision survives when each
    # target is on the SAME side of the reference sign at every score the
    # missing cell could take, which for a matcher is the bound's field being
    # True and for a reverser is its mean being unable to reach zero at all.
    pinned = sorted(arm for arm, t in targets.items()
                    if t["the_mean_cannot_flip_sign_over_the_whole_rubric"])
    matching_at_every_score = sorted(
        arm for arm, t in targets.items()
        if t["the_reference_sign_holds_at_every_admissible_score"] is True)
    survives = (len(pinned) == len(targets)
                and matching_at_every_score == matching)
    return {
        "decision": DECISION_TRANSPORTED if transported
        else DECISION_MODEL_SPECIFIC,
        "reference_sign": reference_sign,
        "n_targets_tested": len(targets),
        "n_carrying_the_reference_sign": len(matching),
        "n_reversing_it": len(reversing),
        "carrying_the_reference_sign": matching,
        "reversing_it": reversing,
        "rule": (
            f"{THE_CLASSIFICATION_RULE}. Anything in between would be a claim "
            "about a proportion, and a proportion of four targets is not a rate. "
            "This rule is POST-SPECIFIED: the frozen protocol pre-specifies the "
            "four per-model sign-match statements, the estimand, the correction "
            "and the retention clause, and files no rule that turns four "
            "per-model verdicts into one label. See "
            "how_this_rule_was_specified, which measures that against the "
            "protocol document rather than asserting it"),
        "every_reversal_is_a_result_and_not_a_null": all(
            (targets[arm]["verdict"] == "refuted") for arm in reversing),
        "the_decision_survives_the_differential_exclusion": survives,
        "what_surviving_the_exclusion_required": {
            "the_mean_cannot_reach_zero_in": pinned,
            "the_reference_sign_holds_at_every_admissible_score_in":
                matching_at_every_score,
            "how_that_is_decided": (
                "every target must be on the same side of the reference sign at "
                "EVERY score the one differentially missing cell could take. "
                "For a target that matches, the bound's "
                "for_every_admissible_score is True and that settles it. For a "
                "target that reverses, that same field is False by "
                "construction -- it asks about matching -- so what settles it "
                "is the mean's range over the whole rubric lying entirely on "
                "one side of zero, which pins the reversal at every score "
                "instead of leaving it to the label the frozen rule happened "
                "to give. The two lists above must then be exactly the "
                "matchers, and if a reverser appeared in the second one the "
                "count 1 of 4 would depend on the missing label"),
        },
    }


def the_decision_in_words(reference: dict, measurement: dict) -> str:
    """The sentence a paper would quote, generated from the counts it states.

    Both branches of the rule are written, including the one this evidence did
    not produce, so the rule is visible in either direction. Every numeral here
    is ``len()`` of a list that is filed beside it -- see the module docstring
    for why a spelled-out count is not acceptable in a document that is
    re-derived and compared whole.
    """
    matching = measurement["carrying_the_reference_sign"]
    reversing = measurement["reversing_it"]
    n = measurement["n_targets_tested"]
    sign = reference["sign"]
    if reversing:
        return (
            "MODEL-SPECIFIC, not generally transported. "
            f"{len(matching)} of {n} targets carry the {sign} Delta_TV sign "
            f"the sealed 9B reference files "
            f"({', '.join(matching) or 'none'}); the other {len(reversing)} "
            f"reverse it ({', '.join(reversing)}), each with the verdict "
            f"'refuted' rather than 'inconclusive'. A sign that reverses in "
            f"{len(reversing)} of {n} targets is not a property of multimodal "
            f"censoring, it is a property of the model it was measured in, and "
            f"the sentence the 9B result invites -- that censoring degrades "
            f"safety compliance -- is not one this panel supports")
    return (
        f"GENERALLY TRANSPORTED across the {n} targets tested: every one of "
        f"them carries the {sign} Delta_TV sign the sealed 9B reference files "
        f"({', '.join(matching)}). This branch is not what the committed "
        f"evidence produced, and it is written out anyway so that the rule this "
        f"stage applies is visible in both directions rather than only in the "
        f"one that happened to come out")


def where_the_two_means_part_company(targets: dict) -> dict:
    """Which targets' 98-family mean lies outside the bound's 99-family range.

    Filed because the two numbers sit next to each other in ``per_target`` and
    look as though one ought to contain the other. It does not have to: the
    98-family mean divides by 98 and the bound's range divides by 99, so
    restoring a family whose own contribution is not positive lowers the mean
    even when every one of the 98 families keeps its value. Where that happens
    it is measured here rather than left for a reader to discover as an
    apparent arithmetic error in a filed decision.
    """
    outside = {}
    for arm, target in sorted(targets.items()):
        low, high = target["mean_range_over_the_whole_rubric"]
        if low <= target["mean"] <= high:
            continue
        outside[arm] = {
            "mean_over_98_families": target["mean"],
            "the_99_family_range": [low, high],
            "which_end_it_falls_past": "above" if target["mean"] > high
            else "below",
            "by": (target["mean"] - high) if target["mean"] > high
            else (low - target["mean"]),
            "both_still_carry_the_same_sign":
                _sign(target["mean"]) == target["the_sign_the_mean_cannot_leave"],
        }
    return {
        "the_two_means_are_over_different_panels_and_do_not_nest": {
            "n_targets_whose_98_family_mean_falls_outside_the_99_family_range":
                len(outside),
            "where": outside,
            "why_that_is_not_a_contradiction": (
                "the per-target mean is over the 98-family confirmatory panel "
                "and is a bootstrap mean; the bound's range is over 99 "
                "families -- the same panel with the one differentially "
                "missing cell restored -- and is a point mean at each end of "
                "the rubric. Different numerator, different denominator, and "
                "one of them resampled. Containment would be a coincidence, "
                "not a check, so it is reported where it fails rather than "
                "asserted where it holds. What IS a check, and what the "
                "decision rests on, is that both quantities carry the same "
                "sign in every target"),
        },
    }


def build() -> dict:
    analysis = load_json(CROSS_MODEL, "cross-model analysis")
    bound = load_json(BOUND, "differential-censoring bound")
    protocol = load_json(PROTOCOL, "frozen protocol")
    reference = reference_of(analysis)
    targets = per_target(analysis, bound)
    decision = decide(reference, targets)
    exclusion = the_exclusion(analysis, bound)
    specification = how_the_rule_was_specified(protocol)

    reversing = decision["reversing_it"]
    required = decision["what_surviving_the_exclusion_required"]
    return {
        "question": "does the sign of the Iteration 10 Qwen3.5-9B result "
                    "transport to the four models Iteration 11 replayed the "
                    "same frozen panel through",
        "produced_by": "scripts/iter11_transportability_decision.py",
        "kind": "iteration_11_transportability_decision_v1",
        "decision": decision["decision"],
        "the_decision_in_words": the_decision_in_words(reference, decision),
        "reference": reference,
        "per_target": targets,
        "the_measurement": decision,
        "why_the_reversals_are_not_nulls": {
            "claim": "a reversed sign could be a small effect read the wrong "
                     "way by noise; these are not",
            "evidence": {
                arm: {"mean": targets[arm]["mean"],
                      "raw_p": targets[arm]["raw_p"],
                      "adjusted_p": targets[arm]["adjusted_p"],
                      "verdict": targets[arm]["verdict"],
                      "worst_p_over_the_whole_rubric":
                          targets[arm]["worst_p_over_the_whole_rubric"]}
                for arm in reversing},
            "how_it_is_checked": (
                "every reversing target's verdict is 'refuted', which the "
                "frozen Holm-Bonferroni rule reaches only where the sign "
                "differs AND the bootstrap p passes its critical value; "
                "the_measurement.every_reversal_is_a_result_and_not_a_null "
                "re-reads that from the verdicts rather than asserting it"),
        },
        "what_is_missing_from_the_panel": exclusion,
        "why_this_is_not_an_artifact_of_the_exclusion": {
            "claim": (
                "the confirmatory panel is "
                f"{exclusion['n_families_in_the_confirmatory_panel']} of "
                f"{exclusion['n_families_in_the_panel']} families because "
                "three cells have no ensemble label, and only one of those "
                "cells is missing differentially: "
                f"{', '.join(exclusion['missing_in_one_arm_only_and_therefore_differential'])}"
                " is absent in one arm of four, while "
                f"{', '.join(exclusion['missing_in_every_arm_and_therefore_not_differential'])}"
                " is absent in all four and is therefore shared missingness "
                "rather than censoring that differs by target. A decision that "
                "needed the missing label to come out a particular way would "
                "be a decision about that label and not about the models"),
            "the_two_facts_read_out_of_the_bound": {
                "the_mean_cannot_reach_zero_in":
                    required["the_mean_cannot_reach_zero_in"],
                "the_reference_sign_holds_at_every_admissible_score_in":
                    required[
                        "the_reference_sign_holds_at_every_admissible_score_in"],
                "and_these_are_not_the_same_fact": (
                    "the second list is the matchers only. The three reversing "
                    "targets are absent from it because it asks whether the "
                    "reference sign HOLDS, and in them it does not -- that is "
                    "the finding. What keeps the finding safe from the missing "
                    "label is the first list: in all four targets the mean's "
                    "range over the rubric's whole 0.0-1.0 lies entirely on "
                    "one side of zero, so no admissible value of the missing "
                    "cell moves any target to the other side"),
            },
            "evidence": {
                arm: {
                    "mean_range_over_the_whole_rubric":
                        targets[arm]["mean_range_over_the_whole_rubric"],
                    "the_mean_cannot_flip_sign_over_the_whole_rubric":
                        targets[arm][
                            "the_mean_cannot_flip_sign_over_the_whole_rubric"],
                    "the_sign_the_mean_cannot_leave":
                        targets[arm]["the_sign_the_mean_cannot_leave"],
                    "the_reference_sign_holds_at_every_admissible_score":
                        targets[arm][
                            "the_reference_sign_holds_at_every_admissible_score"],
                } for arm in EXPECTED_TARGETS},
            "source": _rel(BOUND),
            "how_it_is_checked": (
                "the_measurement.what_surviving_the_exclusion_required "
                "re-derives both lists from per_target and "
                "the_measurement.the_decision_survives_the_differential_"
                "exclusion is True only when every target is pinned to one "
                "side AND the targets the reference sign holds at every score "
                "in are exactly the matchers. Both are arithmetic over filed "
                "numbers, so --verify re-computes them rather than trusting "
                "this paragraph"),
        },
        **where_the_two_means_part_company(targets),
        "how_this_rule_was_specified": specification,
        "what_this_decision_does_not_say": [
            "it does not say the 9B result is wrong: that sign is sealed, was "
            "measured on its own panel, and is the reference these four were "
            "compared against",
            "it does not say the effect is absent in the reversing targets: "
            "each has a Delta_TV that is significantly NEGATIVE, which is a "
            "finding in the opposite direction and is reported as one",
            "it does not say which model property decides the direction. Four "
            "targets and one reference cannot separate scale from family from "
            "training mixture, and naming a mechanism here would be a fifth "
            "claim with no evidence behind it",
            "it does not license pooling the five signs into a rate. One of "
            "five matching is a count over a hand-picked set of models, not an "
            "estimate over a population of them",
        ],
        "what_would_change_it": (
            "a target whose Delta_TV sign matches the reference, measured on the "
            "same frozen panel under the same lock. Adding one moves the count "
            "from 1 of 4 to 2 of 5 and does not make the effect transported, "
            "because the rule requires every target to match; it would make the "
            "model-specificity narrower, which is a different and weaker "
            "statement than the one filed here"),
        "retention_clause_this_decision_is_filed_under": (
            analysis.get("summary") or {}).get("retention"),
        "inputs": {
            "cross_model_analysis": {
                "path": _rel(CROSS_MODEL), "sha256": sha256_file(CROSS_MODEL)},
            "differential_censoring_bound": {
                "path": _rel(BOUND), "sha256": sha256_file(BOUND)},
            "frozen_protocol": {
                "path": _rel(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
            "why_bound_by_hash": (
                "the decision is a pure function of these three files. No clock, "
                "no commit and no tree state is in this document, so --verify "
                "rebuilds it and compares the whole thing: a filing that "
                "carried its own generation time could only ever be compared "
                "field by field, by a list somebody had to remember to keep "
                "short. The protocol is an input because whether this rule was "
                "pre-specified is a question only that document can answer"),
        },
    }


def check_the_specification(doc: dict, protocol: dict | None = None) -> list[str]:
    """Is the post-specification claim a measurement, and does it still hold?

    This is the one claim in the document a reader cannot check from the table
    under it: whether the aggregation rule was written down before the table
    existed. Everything else here is arithmetic on filed numbers, but "the
    protocol does not pre-specify this" is a statement about a document, so the
    check reads that document again rather than trusting the filed answer.

    Repeating the search is what makes the claim falsifiable in the direction
    that matters. If a later protocol freezes the rule, the search hits, the
    filed ``True`` becomes wrong and this fires -- and the thing that changes is
    the claim, not the search.
    """
    issues: list[str] = []
    spec = doc.get("how_this_rule_was_specified")
    if not isinstance(spec, dict) or not spec:
        return ["how_this_rule_was_specified is missing or empty, so the "
                "document files a transportability label without saying whether "
                "the rule that produced it was pre-specified"]

    # The filed rule text and the string that was searched for must be the same
    # string. They are both THE_CLASSIFICATION_RULE by construction, and a check
    # is what keeps "by construction" true after somebody edits one of them.
    if spec.get("the_rule") != THE_CLASSIFICATION_RULE:
        issues.append(
            f"how_this_rule_was_specified.the_rule is {spec.get('the_rule')!r} "
            f"and the rule this stage applies is {THE_CLASSIFICATION_RULE!r}: "
            f"the search for it in the protocol was a search for a different "
            f"sentence")
    measurement = doc.get("the_measurement") or {}
    if not str(measurement.get("rule") or "").startswith(THE_CLASSIFICATION_RULE):
        issues.append(
            "the_measurement.rule does not state the classification rule this "
            "stage applies, so the rule filed beside the decision and the rule "
            f"whose specification is measured have come apart: "
            f"{str(measurement.get('rule'))[:160]!r}")

    if protocol is None:
        try:
            protocol = load_json(PROTOCOL, "frozen protocol")
        except SystemExit as exc:
            return issues + [
                f"the post-specification claim could not be re-measured from "
                f"here: {_rel(PROTOCOL)} is not readable (exit {exc.code}), and "
                f"a claim about a document that was not read is an assertion"]

    text = json.dumps(protocol, indent=2, ensure_ascii=False)
    quoted = THE_CLASSIFICATION_RULE in text
    if spec.get("the_rule_is_not_a_quotation_from_the_protocol") != (not quoted):
        issues.append(
            "the_rule_is_not_a_quotation_from_the_protocol is "
            f"{spec.get('the_rule_is_not_a_quotation_from_the_protocol')!r} and "
            f"searching the protocol as filed finds the rule: {quoted}. Where "
            f"the protocol does carry it, this document's claim is wrong and "
            f"has to change to pre-specified -- the search is not the thing to "
            f"adjust")
    # The label is a CONCLUSION from that search, so it is derived from it and
    # compared rather than merely constrained. Checking only one direction --
    # "post-specified while the protocol contains the rule" -- would let a
    # document upgrade itself to pre-specified on the same empty search, which
    # is the overstatement this whole block exists to prevent.
    expected_label = SPECIFIED_PRE if quoted else SPECIFIED_POST
    if spec.get("specified_when") not in SPECIFICATION_LABELS:
        issues.append(
            f"specified_when is {spec.get('specified_when')!r}, which is not one "
            f"of {list(SPECIFICATION_LABELS)}: a label nobody has defined the "
            f"meaning of is not a measurement")
    elif spec.get("specified_when") != expected_label:
        issues.append(
            f"specified_when says {spec.get('specified_when')!r} and searching "
            f"the protocol as filed for the rule returns {quoted}, which makes "
            f"it {expected_label!r}: the label is a conclusion from that search "
            f"and cannot be moved without the protocol moving")

    frozen = protocol.get("protocol") or {}
    filed_document = spec.get("the_protocol_document") or {}
    for key, expected in (("path", _rel(PROTOCOL)),
                          ("sha256", sha256_file(PROTOCOL)),
                          ("name", frozen.get("name")),
                          ("status", frozen.get("status")),
                          ("frozen_before_any_target_generation",
                           frozen.get("frozen_before_any_target_generation")),
                          ("frozen_at_code_commit",
                           frozen.get("frozen_at_code_commit"))):
        if filed_document.get(key) != expected:
            issues.append(
                f"the_protocol_document.{key} is {filed_document.get(key)!r} and "
                f"the protocol as filed says {expected!r}")
    if filed_document.get("frozen_before_any_target_generation") is not True:
        issues.append(
            "the filed protocol document says "
            "frozen_before_any_target_generation="
            f"{filed_document.get('frozen_before_any_target_generation')!r}: "
            "post-specified is only a meaningful qualification of a rule when "
            "the protocol it is post-specified against really was frozen before "
            "the results existed")
    # Read the PROTOCOL, not the filed echo of it. The loop above already
    # compares the two, but a comparison against the artifact's own copy is not
    # a measurement of the world: a document that filed True against a protocol
    # saying False would satisfy it, and would then be resting a claim about
    # specification order on a protocol that was not frozen first.
    if frozen.get("frozen_before_any_target_generation") is not True:
        issues.append(
            "the protocol as filed was not frozen before any target was "
            "generated, so 'post-specified' has no 'before' to be relative to "
            "and the qualification this document puts on its own rule does not "
            "describe the situation")
    bound = (doc.get("inputs") or {}).get("frozen_protocol") or {}
    if not bound.get("sha256"):
        issues.append(
            "the document measures a claim against the frozen protocol but "
            "inputs.frozen_protocol does not bind it, so a reader cannot tell "
            "which bytes the measurement was made against")
    elif bound.get("sha256") != filed_document.get("sha256"):
        issues.append(
            f"inputs.frozen_protocol binds {bound.get('sha256')!r} and "
            f"the_protocol_document binds {filed_document.get('sha256')!r}: the "
            f"same file is bound twice in one document and the two disagree, so "
            f"at least one of them is not the protocol this claim was measured "
            f"against")

    per_target, not_per_target = which_hypotheses_name_a_target(protocol)
    pre = spec.get("what_the_protocol_pre_specifies") or {}
    if pre.get("per_target_statements") != per_target:
        issues.append(
            "per_target_statements is not the split the protocol's own "
            "hypotheses and model_matrix produce, so the statements this "
            "document calls pre-specified are not the ones that were frozen")
    if pre.get("statements_that_are_not_about_one_target") != not_per_target:
        issues.append(
            "statements_that_are_not_about_one_target is not what the protocol "
            "leaves over: a hypothesis quietly dropped from both dicts would "
            "make the conjunction look complete")

    conjunction = spec.get("the_rule_is_the_conjunction_of_those_statements") or {}
    declared = (protocol.get("multiplicity") or {}).get("n_confirmatory_model_tests")
    covers = sorted(entry["target"] for entry in per_target.values())
    for key, expected in (
            ("n_statements_conjoined", len(per_target)),
            ("n_confirmatory_tests_the_protocol_declares", declared),
            ("n_targets_this_document_measured", len(EXPECTED_TARGETS)),
            ("the_three_counts_agree",
             len(per_target) == declared == len(EXPECTED_TARGETS)),
            ("targets_covered", covers),
            ("every_target_is_covered_exactly_once",
             covers == sorted(EXPECTED_TARGETS))):
        if conjunction.get(key) != expected:
            issues.append(
                f"the_rule_is_the_conjunction_of_those_statements.{key} is "
                f"{conjunction.get(key)!r} and the protocol supports {expected!r}")
    if conjunction.get("the_three_counts_agree") is not True:
        issues.append(
            "the rule cannot be the conjunction of the protocol's per-target "
            f"statements: {len(per_target)} such statements were found, the "
            f"protocol declares {declared} confirmatory tests and this document "
            f"measured {len(EXPECTED_TARGETS)} targets, so the set of results "
            f"being aggregated is not the set that was pre-specified")
    return issues


def check_the_decision(doc: dict, protocol: dict | None = None) -> list[str]:
    """Does the filed document agree with itself?

    A decision artifact is the easiest kind to write and the easiest to
    overstate, so the checks are on the internal agreement rather than on the
    prose: the derived call against the filed one, the counts against the per
    -target signs, and the reversal claim against the verdicts it cites.

    ``protocol`` is the frozen protocol document, and it is passed in rather
    than loaded so that the specification check measures the same bytes
    :func:`build` did. It is loaded here when a caller omits it.
    """
    issues: list[str] = []
    reference = doc.get("reference") or {}
    targets = doc.get("per_target") or {}
    measurement = doc.get("the_measurement") or {}

    if sorted(targets) != sorted(EXPECTED_TARGETS):
        issues.append(f"per_target covers {sorted(targets)}, not the four "
                      f"confirmatory targets {sorted(EXPECTED_TARGETS)}")
    if measurement.get("reference_sign") != reference.get("sign"):
        issues.append(
            f"the_measurement compares against "
            f"{measurement.get('reference_sign')!r} while the reference block "
            f"files {reference.get('sign')!r}")

    matching = sorted(arm for arm, t in targets.items()
                      if t.get("sign") == reference.get("sign"))
    reversing = sorted(arm for arm, t in targets.items()
                       if t.get("sign") != reference.get("sign"))
    if sorted(measurement.get("carrying_the_reference_sign") or []) != matching:
        issues.append(
            f"the_measurement names {measurement.get('carrying_the_reference_sign')} "
            f"as carrying the reference sign but per_target says {matching}")
    if sorted(measurement.get("reversing_it") or []) != reversing:
        issues.append(
            f"the_measurement names {measurement.get('reversing_it')} as "
            f"reversing but per_target says {reversing}")
    if measurement.get("n_carrying_the_reference_sign") != len(matching):
        issues.append(
            f"n_carrying_the_reference_sign is "
            f"{measurement.get('n_carrying_the_reference_sign')} and "
            f"{len(matching)} target(s) carry it")
    if measurement.get("n_reversing_it") != len(reversing):
        issues.append(
            f"n_reversing_it is {measurement.get('n_reversing_it')} and "
            f"{len(reversing)} target(s) reverse it")

    expected = DECISION_TRANSPORTED if not reversing else DECISION_MODEL_SPECIFIC
    if doc.get("decision") != expected:
        issues.append(
            f"the document files decision={doc.get('decision')!r} while its own "
            f"per_target signs support {expected!r}: the decision and the "
            f"measurement it is derived from disagree")
    expected_words = the_decision_in_words(reference, measurement)
    if doc.get("the_decision_in_words") != expected_words:
        issues.append(
            "the_decision_in_words is not the sentence these counts generate, "
            "so the prose a paper would quote has drifted from the measurement "
            f"it describes: filed {doc.get('the_decision_in_words')!r}")
    every_reversal_refuted = all(
        targets[arm].get("verdict") == "refuted" for arm in reversing)
    if measurement.get("every_reversal_is_a_result_and_not_a_null") != \
            every_reversal_refuted:
        issues.append(
            f"every_reversal_is_a_result_and_not_a_null is "
            f"{measurement.get('every_reversal_is_a_result_and_not_a_null')} "
            f"and the verdicts filed for {reversing} support "
            f"{every_reversal_refuted}")
    for arm in reversing:
        if targets[arm].get("verdict") != "refuted":
            issues.append(
                f"{arm} reverses the reference sign but its filed verdict is "
                f"{targets[arm].get('verdict')!r}, so calling the reversal a "
                f"result rather than a null is not supported by the verdict")
    for arm, target in targets.items():
        if target.get("sign_matches_the_reference") != (
                target.get("sign") == reference.get("sign")):
            issues.append(
                f"{arm}: sign_matches_the_reference is "
                f"{target.get('sign_matches_the_reference')} but its sign is "
                f"{target.get('sign')!r} against the reference's "
                f"{reference.get('sign')!r}")

        # Two independent sources for the same fact. ``sign`` is the analysis's,
        # taken from the 98-family bootstrap mean; ``the_sign_the_mean_cannot_
        # leave`` is the bound's, taken from the 99-family point-mean range over
        # the rubric. They are computed from different panels by different code
        # and a disagreement between them is a disagreement about the effect,
        # not a rounding artifact.
        pinned = target.get("the_mean_cannot_flip_sign_over_the_whole_rubric")
        held = target.get("the_sign_the_mean_cannot_leave")
        if pinned and held != target.get("sign"):
            issues.append(
                f"{arm}: the bound's mean range over the whole rubric cannot "
                f"leave {held!r} while the analysis files its sign as "
                f"{target.get('sign')!r}")
        if not pinned and held is not None:
            issues.append(
                f"{arm}: names {held!r} as the sign its mean cannot leave while "
                f"filing the_mean_cannot_flip_sign_over_the_whole_rubric as "
                f"{pinned}, so one of the two is wrong about the range "
                f"{target.get('mean_range_over_the_whole_rubric')}")
        holds_everywhere = target.get(
            "the_reference_sign_holds_at_every_admissible_score")
        if holds_everywhere is True and not (pinned and held == reference.get("sign")):
            issues.append(
                f"{arm}: the reference sign is said to hold at every admissible "
                f"score, which entails a mean pinned to "
                f"{reference.get('sign')!r}, but the range is "
                f"{target.get('mean_range_over_the_whole_rubric')} "
                f"(pinned={pinned}, held={held!r})")
        if holds_everywhere is not True and target.get("sign") == reference.get("sign"):
            issues.append(
                f"{arm}: carries the reference sign at the label used but not at "
                f"every admissible score, so the count of targets carrying it "
                f"depends on the missing label")

    required = measurement.get("what_surviving_the_exclusion_required") or {}
    expected_pinned = sorted(arm for arm, t in targets.items()
                             if t.get("the_mean_cannot_flip_sign_over_the_"
                                      "whole_rubric"))
    expected_holds = sorted(
        arm for arm, t in targets.items()
        if t.get("the_reference_sign_holds_at_every_admissible_score") is True)
    if sorted(required.get("the_mean_cannot_reach_zero_in") or []) != expected_pinned:
        issues.append(
            f"the_mean_cannot_reach_zero_in is "
            f"{required.get('the_mean_cannot_reach_zero_in')} and per_target "
            f"pins {expected_pinned}")
    if sorted(required.get("the_reference_sign_holds_at_every_admissible_score_in")
              or []) != expected_holds:
        issues.append(
            f"the_reference_sign_holds_at_every_admissible_score_in is "
            f"{required.get('the_reference_sign_holds_at_every_admissible_score_in')} "
            f"and per_target holds it in {expected_holds}")
    survives = (len(expected_pinned) == len(targets)
                and expected_holds == matching)
    if measurement.get("the_decision_survives_the_differential_exclusion") != survives:
        issues.append(
            f"the_decision_survives_the_differential_exclusion is "
            f"{measurement.get('the_decision_survives_the_differential_exclusion')} "
            f"and per_target supports {survives}: every target pinned is "
            f"{len(expected_pinned) == len(targets)}, and the reference sign "
            f"holding at every score is exactly the matchers is "
            f"{expected_holds == matching}")

    two_facts = (doc.get("why_this_is_not_an_artifact_of_the_exclusion")
                 or {}).get("the_two_facts_read_out_of_the_bound") or {}
    if sorted(two_facts.get("the_mean_cannot_reach_zero_in") or []) != expected_pinned:
        issues.append(
            "why_this_is_not_an_artifact_of_the_exclusion restates "
            f"the_mean_cannot_reach_zero_in as "
            f"{two_facts.get('the_mean_cannot_reach_zero_in')} against "
            f"{expected_pinned} in per_target: a paragraph that disagrees with "
            f"the table under it is the failure this check exists for")
    if sorted(two_facts.get(
            "the_reference_sign_holds_at_every_admissible_score_in")
              or []) != expected_holds:
        issues.append(
            "why_this_is_not_an_artifact_of_the_exclusion restates "
            "the_reference_sign_holds_at_every_admissible_score_in as "
            f"{two_facts.get('the_reference_sign_holds_at_every_admissible_score_in')} "
            f"against {expected_holds} in per_target")

    nesting = doc.get(
        "the_two_means_are_over_different_panels_and_do_not_nest") or {}
    filed_outside = nesting.get(
        "n_targets_whose_98_family_mean_falls_outside_the_99_family_range")
    outside = sorted(
        arm for arm, t in targets.items()
        if not (t["mean_range_over_the_whole_rubric"][0] <= t["mean"]
                <= t["mean_range_over_the_whole_rubric"][1]))
    if filed_outside != len(outside):
        issues.append(
            f"n_targets_whose_98_family_mean_falls_outside_the_99_family_range "
            f"is {filed_outside} and {len(outside)} target(s) fall outside: "
            f"{outside}")
    if sorted(nesting.get("where") or []) != outside:
        issues.append(
            f"the non-nesting block names {sorted(nesting.get('where') or [])} "
            f"and per_target puts {outside} outside their range")
    issues.extend(check_the_specification(doc, protocol))
    return issues


def verify(path: Path | None = None) -> tuple[int, str, list[str]]:
    """``(code, conclusion, issues)``: re-derive and compare, writing nothing."""
    path = OUT_PATH if path is None else path
    if not path.exists():
        return 2, "not_filed", [
            f"no transportability decision at {_rel(path)}; run this script "
            f"with no arguments to file one"]
    filed = json.loads(path.read_text(encoding="utf-8"))
    try:
        fresh = build()
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), \
            "could_not_rederive", [
                f"the decision could not be re-derived from its inputs: "
                f"exit {exc.code}"]

    issues = []
    for key in sorted(set(filed) | set(fresh)):
        if filed.get(key) != fresh.get(key):
            issues.append(
                f"{key}: filed {json.dumps(filed.get(key), sort_keys=True)[:200]}"
                f" != re-derived "
                f"{json.dumps(fresh.get(key), sort_keys=True)[:200]}")
    # Read the protocol again rather than reusing build()'s copy: the whole
    # point of the specification check is that it measures the document as it
    # stands now, and a check that shares its input with the thing it is
    # checking cannot notice that input changing underneath it.
    issues.extend(check_the_decision(
        filed, load_json(PROTOCOL, "frozen protocol")))
    if issues:
        return 1, "differs", issues
    return 0, "reproduced_exactly", []


def _specification_line(doc: dict) -> str:
    """One printed line saying when the rule was specified, and how that is known.

    Printed rather than left in the JSON because the finding this answers was a
    characterisation, and a characterisation only gets corrected where somebody
    reads it. A label that says "post-specified" next to a decision that reads
    like a pre-registered verdict is the whole problem.
    """
    spec = doc.get("how_this_rule_was_specified") or {}
    conjunction = spec.get("the_rule_is_the_conjunction_of_those_statements") or {}
    when = str(spec.get("specified_when") or "unknown").replace("_", " ")
    return (f"  rule       {when} -- the protocol pre-specifies "
            f"{conjunction.get('n_statements_conjoined')} per-target sign "
            f"statements and the correction over "
            f"{conjunction.get('n_confirmatory_tests_the_protocol_declares')} "
            f"tests, and files no aggregation rule; the rule searched for in "
            f"the protocol is absent: "
            f"{spec.get('the_rule_is_not_a_quotation_from_the_protocol')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="re-derive and compare with the filed decision, "
                           "writing nothing")
    mode.add_argument("--write", action="store_true",
                      help="file the decision. Explicit, because it overwrites "
                           "a committed artifact")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.verify or not args.write:
        code, conclusion, issues = verify(args.out)
        if code == 2:
            print(f"TRANSPORTABILITY DECISION: NOT FILED -- {issues[0]}")
            return 2
        if code == 1:
            print(f"TRANSPORTABILITY DECISION: FAIL ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print(f"TRANSPORTABILITY DECISION: VERIFIED -- {conclusion.replace('_', ' ')}")
        filed = json.loads(args.out.read_text(encoding="utf-8"))
        measurement = filed["the_measurement"]
        print(f"  decision   {filed['decision']}")
        print(f"  reference  {filed['reference']['model']}, "
              f"{measurement['reference_sign']}, "
              f"{filed['reference']['n_families']} families")
        print(f"  carrying   {measurement['n_carrying_the_reference_sign']} of "
              f"{measurement['n_targets_tested']}: "
              f"{', '.join(measurement['carrying_the_reference_sign']) or '-'}")
        print(f"  reversing  {measurement['n_reversing_it']}: "
              f"{', '.join(measurement['reversing_it']) or '-'}")
        print(f"  survives the differential exclusion  "
              f"{measurement['the_decision_survives_the_differential_exclusion']}"
              f" (every target's mean is pinned to one side of zero, and the "
              f"reference sign holds at every admissible score in exactly "
              f"{', '.join(measurement['carrying_the_reference_sign']) or '-'})")
        print(_specification_line(filed))
        return 0

    doc = build()
    problems = check_the_decision(doc)
    if problems:
        print("FAIL: refusing to file a decision that disagrees with itself",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    measurement = doc["the_measurement"]
    required = measurement["what_surviving_the_exclusion_required"]
    print(f"wrote {_rel(args.out)}")
    print(f"  decision   {doc['decision']}")
    print(f"  reference  {doc['reference']['model']}, "
          f"{measurement['reference_sign']}, mean "
          f"{doc['reference']['mean']:.6f}")
    for arm in EXPECTED_TARGETS:
        target = doc["per_target"][arm]
        print(f"  {arm:15s} {target['sign']:9s} mean {target['mean']:+.6f} "
              f"raw_p {target['raw_p']:.4f} {target['verdict']:11s} "
              f"matches={target['sign_matches_the_reference']} "
              f"pinned_to={target['the_sign_the_mean_cannot_leave']}")
    print(f"  carrying the reference sign  "
          f"{measurement['n_carrying_the_reference_sign']} of "
          f"{measurement['n_targets_tested']}: "
          f"{', '.join(measurement['carrying_the_reference_sign']) or '-'}")
    print(f"  reversing it                 "
          f"{measurement['n_reversing_it']}: "
          f"{', '.join(measurement['reversing_it']) or '-'}")
    print(f"  survives the differential exclusion  "
          f"{measurement['the_decision_survives_the_differential_exclusion']}")
    print(f"    mean pinned to one side of zero in   "
          f"{', '.join(required['the_mean_cannot_reach_zero_in']) or '-'}")
    holds_everywhere = required[
        "the_reference_sign_holds_at_every_admissible_score_in"]
    print(f"    reference sign holds at every score  "
          f"{', '.join(holds_everywhere) or '-'}")
    print(_specification_line(doc))
    print(f"  sha256                       {sha256_bytes(args.out.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
