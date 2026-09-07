"""How much truncation a full evaluation panel may carry, in ONE place.

Three gates look at truncation, and until Iteration 11 two of them had never
been compared, because until Iteration 11 the difference could not show:

* the 11.5 ELIGIBILITY gate (``replay.confirmatory``, gate
  ``truncation_reviewed``) requires ``n_truncated == 0`` over the 12-family
  screening panel. That is unchanged and deliberately separate: eligibility
  screens a checkpoint before any confirmatory spend, so it is allowed to be
  stricter than the panel it is screening for.
* the 11.6 COMPLETION gate (``scripts/iter11_replay_checks.py``) accepted a
  full panel at an overall rate up to ``MAX_TRUNCATION_RATE`` with a
  variant-rate spread up to ``MAX_VARIANT_SPREAD``.
* the EVALUATION panel gate (``evaluation/gate.py``) required zero.

The frozen Qwen3.5-9B reference satisfied both full-panel standards at once
-- it truncated nothing, its longest response being 1291 tokens against a
1536 cap -- so the two full-panel gates agreed on every panel that had ever
been evaluated and the disagreement was invisible. Iteration 11 replayed four
SMALLER checkpoints under the same frozen cap and greedy decoding. Three of
them truncate: 7, 1 and 4 cells of 600. Each passed 11.6 and each was then
refused by the evaluation gate, after roughly nine hours of judging had
already been spent per arm.

That is an inconsistency between two gates in this repository, not a property
of the models, and this module is its resolution: ONE pair of thresholds,
defined once, imported by every gate that accepts a FULL panel.

WHY THESE NUMBERS ARE NOT NEW
``scripts/iter11_replay_checks.py`` already carried 0.02 and 0.05, and its
comment already said they were "identical to the Iteration 10 rule". They were
introduced in commit 6389df65, BEFORE any Iteration 11 target evidence was
generated, so adopting them at the evaluation gate applies a pre-registered
standard rather than choosing one after seeing the results. What changes is
that the evaluation gate now reads them instead of restating a stricter rule
of its own.

WHY NOT SIMPLY RAISE THE CAP
The protocol's ``uniform_cap_rule`` says to choose a new uniform cap and rerun
all five checkpoints. That remedy assumes truncation means the cap was too
small. Under greedy decoding a small model can enter a repetition loop and
emit to ANY cap, so escalating cannot reach zero and the remedy has no
termination criterion. Escalation is therefore defined here as firing ONLY
when a registered threshold is exceeded -- see
:func:`exceeds_registered_thresholds` -- and not merely because some cell
reached the cap.

WHY A CAPPED LOOP IS KEPT RATHER THAN DROPPED
A response that ran to the cap is an observation of what the model did under
the frozen decoding configuration. It is not missing data, and removing it
would remove cells from some arms and not others, changing the quantity the
cross-model comparison estimates. All 600 outputs per arm are therefore kept.
Acceptance depends on the two registered thresholds and on nothing else: not
on how repetitive the text is, and not on what any judge scored it. Repetition
diagnostics are reported for interpretation only and gate nothing.

Sealed Iteration 9 and 10 artifacts are unaffected: every archived panel in
``outputs/`` has zero truncated records, so a threshold that admits up to 2%
returns the same verdict it always did on all of them.
"""

from __future__ import annotations

from collections import defaultdict

#: Maximum share of a full panel that may have reached the generation cap.
#: Inclusive: exactly this rate passes.
MAX_TRUNCATION_RATE = 0.02

#: Maximum difference between the highest and lowest per-variant truncation
#: rate. Inclusive. This is the check that matters for the estimands: a global
#: rate can hide condition-dependent truncation, and the causal quantities are
#: differences BETWEEN variants, so truncation concentrated in one variant
#: biases them in a way the same number of truncated cells spread evenly does
#: not.
MAX_VARIANT_SPREAD = 0.05

#: ``finish_reason`` a capped generation carries. Every other reason besides
#: the natural terminations is a defect and is not admitted by any threshold.
TRUNCATED_FINISH_REASON = "length"

#: finish reasons that mean the model stopped by itself.
NATURAL_FINISH_REASONS = frozenset({"eos", "stop"})


def is_truncated(record: dict) -> bool:
    """Is this one cell truncated? The single definition, used everywhere.

    Both flags are honoured because both are written: the replay backend
    records ``hit_max_new_tokens``, and the per-record provenance schema the
    Iteration 11 protocol registers names ``truncated``. Counting one in one
    gate and the other in another is how two gates came to disagree about the
    same file. Across all 3948 archived replay records the two never disagree,
    so unifying them changes no existing count.

    Identity against ``True`` rather than truthiness: a provenance field that
    is missing, ``None``, ``0`` or ``"false"`` must not silently become a
    truncation, and must not silently become one in a gate that counts and
    not in another.
    """
    return (record.get("hit_max_new_tokens") is True
            or record.get("truncated") is True)


def _cap_of(record: dict):
    """The generation cap this record was produced under, if it says.

    The per-record provenance nests it under ``generation_config``; a caller
    building records by hand may put it at the top level. Recording a null cap
    next to a cell that reached it would leave the reader unable to see how
    close to the boundary the rest of the panel ran.
    """
    config = record.get("generation_config")
    if isinstance(config, dict) and config.get("max_new_tokens") is not None:
        return config["max_new_tokens"]
    return record.get("max_new_tokens")


def measure_truncation(records: list[dict]) -> dict:
    """Everything a report needs to state its truncation, measured once.

    Args:
        records: Replay output records for one panel.

    Returns:
        A JSON-serialisable dict carrying the counts, the overall rate, the
        per-variant rates, the spread, the registered thresholds, the affected
        cells, whether the panel passes and why not when it does not. Callers
        record this verbatim rather than recomputing any part of it, so the
        numbers in a report are the numbers the gate used.
    """
    n_records = len(records)
    by_variant_n: dict[str, int] = defaultdict(int)
    by_variant_trunc: dict[str, int] = defaultdict(int)
    cells: list[dict] = []

    for record in records:
        variant = record.get("variant")
        by_variant_n[variant] += 1
        if not is_truncated(record):
            continue
        by_variant_trunc[variant] += 1
        cells.append({
            "family_id": record.get("family_id"),
            "variant": variant,
            "finish_reason": record.get("finish_reason"),
            "output_token_count": record.get("output_token_count"),
            "max_new_tokens": _cap_of(record),
        })

    n_truncated = len(cells)
    overall_rate = n_truncated / n_records if n_records else 0.0

    # Rates only for variants actually present, so an absent variant cannot
    # contribute a zero that widens the spread.
    per_variant = {
        variant: {
            "n": by_variant_n[variant],
            "n_truncated": by_variant_trunc.get(variant, 0),
            "rate": (by_variant_trunc.get(variant, 0) / by_variant_n[variant]
                     if by_variant_n[variant] else 0.0),
        }
        for variant in sorted(by_variant_n)
    }
    rates = [entry["rate"] for entry in per_variant.values()]
    spread = (max(rates) - min(rates)) if rates else 0.0

    cells.sort(key=lambda c: (str(c["family_id"]), str(c["variant"])))
    measurement = {
        "n_records": n_records,
        "n_truncated": n_truncated,
        "overall_rate": overall_rate,
        "max_variant_spread": spread,
        "per_variant": per_variant,
        "cells": cells,
        "thresholds": {
            "max_overall_rate": MAX_TRUNCATION_RATE,
            "max_variant_spread": MAX_VARIANT_SPREAD,
            "inclusive": True,
        },
        "counting_rule": ("hit_max_new_tokens is True or truncated is True"),
    }
    measurement["violations"] = truncation_violations(measurement)
    measurement["passed"] = not measurement["violations"]
    return measurement


def truncation_violations(measurement: dict) -> list[str]:
    """Why this measurement fails the registered thresholds, if it does.

    Both thresholds must hold, inclusively. Exceeding either is enough to
    fail, and the message names the number and the bound so a report can be
    read without reopening this module.
    """
    violations: list[str] = []
    rate = measurement["overall_rate"]
    if rate > MAX_TRUNCATION_RATE:
        violations.append(
            f"truncation rate {rate:.6f} exceeds the registered maximum "
            f"{MAX_TRUNCATION_RATE} ({measurement['n_truncated']} of "
            f"{measurement['n_records']} records reached the generation cap)")
    spread = measurement["max_variant_spread"]
    if spread > MAX_VARIANT_SPREAD:
        violations.append(
            f"truncation spread across variants {spread:.6f} exceeds the "
            f"registered maximum {MAX_VARIANT_SPREAD}: truncation is "
            f"concentrated in some variants rather than spread across them, "
            f"which biases estimands that are differences between variants")
    return violations


def exceeds_registered_thresholds(measurement: dict) -> bool:
    """Does the uniform-cap remedy fire?

    Only here. A cell reaching the cap is not by itself a reason to escalate:
    under greedy decoding a repetition loop runs to any cap, so "some cell was
    truncated" has no termination criterion and would escalate forever. The
    registered thresholds are the trigger, which makes escalation a decision
    about the PANEL's truncation profile rather than about one response.
    """
    return bool(measurement.get("violations"))


def finish_reason_violations(records: list[dict]) -> list[str]:
    """Finish reasons that no threshold admits.

    A natural termination always passes. ``length`` passes only on a record
    this module counts as truncated, so a response cannot claim to have
    stopped for length while also claiming not to have reached the cap -- the
    two flags contradicting each other is a defect in the evidence, not a
    truncation to be budgeted. Any other reason is a failure outright.
    """
    bad: list[str] = []
    inconsistent: list[str] = []
    for record in records:
        reason = record.get("finish_reason")
        if reason in NATURAL_FINISH_REASONS:
            continue
        if reason == TRUNCATED_FINISH_REASON:
            if is_truncated(record):
                continue
            inconsistent.append(
                f"{record.get('family_id')}/{record.get('variant')}")
            continue
        bad.append(f"{record.get('family_id')}/{record.get('variant')}: "
                   f"{reason!r}")
    violations = []
    if bad:
        violations.append(
            f"finish_reason must be in "
            f"{sorted(NATURAL_FINISH_REASONS)} or "
            f"{TRUNCATED_FINISH_REASON!r} on a truncated record, got "
            f"{len(bad)} record(s) otherwise: {bad[:10]}")
    if inconsistent:
        violations.append(
            f"{len(inconsistent)} record(s) report finish_reason "
            f"{TRUNCATED_FINISH_REASON!r} without being marked truncated, "
            f"which is self-contradictory provenance: {inconsistent[:10]}")
    return violations
