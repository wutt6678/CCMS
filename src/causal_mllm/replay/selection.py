"""Iteration 11.5 — the pre-registered eligibility selection.

This module is the *contract* for which 12 families constitute the 11.5
eligibility subset. It lives apart from both the gate and the producer so
that neither can define the selection in terms of the other:

* :mod:`causal_mllm.replay.confirmatory` imports it to RE-DERIVE the
  expected selection and reject a report that names different families;
* the 11.5 producer imports it to decide what to replay.

Both therefore depend on one recipe, and the recipe depends only on frozen,
committed Iteration 10 evidence.

Why re-derivation rather than a stored constant: a report that records both
its family ids and their digest is self-consistent no matter which families
it names, so checking the digest against the ids in the same document proves
nothing about WHICH families were chosen. Deriving the selection from the
frozen 9B reference run and the frozen adjudicated labels — both tracked in
git and covered by ``scale_c_closeout_manifest.py --verify`` — makes the
expected value something a report cannot influence.

Eligibility is TECHNICAL, never performance-based: the question is whether a
target can represent the same semantic role and image structure and produce
complete, non-truncated responses for all six variants. Stratifying on the
REFERENCE model's properties is what keeps the selection independent of how
any candidate performs, which is the selection freedom the frozen protocol
exists to remove.

Stratification dimensions:

* **Length** — tertiles of each family's median 9B output-token count across
  its six variants. This is what stresses the generation cap and makes the
  truncation diagnostics informative.
* **Risk** — the family's maximum adjudicated ``compliance_level``, collapsed
  to compliant (0), partial (1), noncompliant (2-3). This is what makes the
  subset span the behavioural range the cross-model comparison measures.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.replay.errors import ReplayError

#: Declared order matters: it is the final tie-break in both allocation
#: rounds, so it is part of the pre-registered recipe.
LENGTH_STRATA = ("short", "median", "long")
RISK_STRATA = ("compliant", "partial", "noncompliant")

#: 12 families x 6 variants = 72 generations per target, 288 across four.
N_ELIGIBILITY_FAMILIES = 12
N_ELIGIBILITY_VARIANTS = len(ALL_VARIANT_NAMES)
N_ELIGIBILITY_ATTEMPTS = N_ELIGIBILITY_FAMILIES * N_ELIGIBILITY_VARIANTS

#: One family per grid cell guarantees both dimensions are covered; the
#: remainder goes to the largest cells so the subset stays representative
#: rather than uniformly thin.
N_CELLS = len(LENGTH_STRATA) * len(RISK_STRATA)
N_EXTRAS = N_ELIGIBILITY_FAMILIES - N_CELLS

#: Pointer files, repo-root relative. Every input path is READ from these
#: rather than hardcoded, so the selection follows the frozen artifacts.
FROZEN_9B_REFERENCE = ("outputs/iteration_11/protocol/"
                       "frozen_9b_reference.json")
SCALE_PROFILES = "configs/evaluation/scale_profiles.json"
SCALE_PROFILE = "scale_c"
ADJUDICATED_LABELS_FILE = "llm_labels_adjudicated.json"
PANEL_FILE = "validated_families.jsonl"
REFERENCE_OUTPUTS_FILE = "replay_outputs.jsonl"

#: Where 11.5 writes the human-auditable selection artifact. Its contents
#: must match a fresh re-derivation, so tampering with the committed copy is
#: detected rather than trusted.
SELECTION_ARTIFACT = "outputs/iteration_11/eligibility/selection.json"

SELECTION_RECIPE = (
    "Length stratum = tertile of the family's median Qwen3.5-9B output-token "
    "count across its six variants, taken from the frozen Iteration 10 "
    "reference run named by outputs/iteration_11/protocol/"
    "frozen_9b_reference.json; cut points are the sorted per-family medians "
    "at n//3 and 2n//3, assigned as short < t1 <= median <= t2 < long. Risk "
    "stratum = the family's maximum adjudicated compliance_level from the "
    "frozen Iteration 10 ensemble labels, collapsed to compliant=0, "
    "partial=1, noncompliant=2 or 3. Allocation: round 1 takes one family "
    "from each of the nine cells; round 2 gives one further family to each "
    "of the three largest cells not yet granted an extra, breaking ties in "
    "favour of the risk stratum with the fewest families selected so far and "
    "then by the declared cell order. "
    "Within a cell, families are ranked by absolute distance from that "
    "cell's median length, ties broken by family_id, so the most "
    "representative member is chosen. The selection digest is sha256 over "
    "the sorted ids joined by newlines. No randomness, and no "
    "candidate-target information, is used at any step."
)


def selected_families_sha256(family_ids: Iterable) -> str:
    """Canonical digest of a family selection.

    Sorted and newline-joined so the hash depends on the SET selected and
    not on the order a report happened to list it in. Making the recipe
    explicit is what lets the gate recompute and verify a recorded value
    instead of merely checking that one is present.
    """
    canonical = "\n".join(sorted(str(f) for f in family_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- #
# Frozen reference measurements
# --------------------------------------------------------------------- #
def reference_family_lengths(reference_outputs: str | Path) -> dict:
    """``{family_id: median 9B output tokens}`` from the frozen run.

    The median over a family's six variants rather than the mean, so one
    unusually long variant cannot move a family across a tertile boundary.
    """
    path = Path(reference_outputs)
    if not path.exists():
        raise ReplayError(
            f"frozen 9B reference outputs not found at {path}; the 11.5 "
            f"stratification is defined on that run and cannot proceed "
            f"without it")
    per_family: dict = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tokens = record.get("output_token_count")
        if tokens is None:
            raise ReplayError(
                f"frozen 9B reference record for family "
                f"{record.get('family_id')!r} variant "
                f"{record.get('variant')!r} has no output_token_count; "
                f"length strata cannot be derived")
        per_family[record["family_id"]].append(tokens)
    if not per_family:
        raise ReplayError(f"no records read from {path}")
    return {family: statistics.median(values)
            for family, values in per_family.items()}


def reference_family_risk(labels_path: str | Path) -> dict:
    """``{family_id: max adjudicated compliance_level}`` per family.

    The MAXIMUM across variants is the family's risk level: a family is
    noncompliant if the reference model complied under any of the six
    conditions, which is the behaviour the cross-model comparison is about.
    """
    path = Path(labels_path)
    if not path.exists():
        raise ReplayError(
            f"frozen adjudicated labels not found at {path}; risk strata "
            f"cannot be derived")
    document = json.loads(path.read_text(encoding="utf-8"))
    labels = document.get("labels") if isinstance(document, dict) else None
    if not isinstance(labels, Mapping):
        raise ReplayError(
            f"{path} does not contain a 'labels' mapping keyed by family_id")
    risks: dict = {}
    for family, variants in labels.items():
        if not isinstance(variants, Mapping) or not variants:
            raise ReplayError(
                f"family {family!r} has no per-variant labels in {path}")
        levels = []
        for variant, label in variants.items():
            if not isinstance(label, Mapping) \
                    or "compliance_level" not in label:
                raise ReplayError(
                    f"family {family!r} variant {variant!r} has no "
                    f"compliance_level in {path}")
            levels.append(int(label["compliance_level"]))
        risks[family] = max(levels)
    return risks


def panel_family_ids(panel_path: str | Path) -> list:
    """The frozen panel's family ids, in file order."""
    path = Path(panel_path)
    if not path.exists():
        raise ReplayError(f"frozen panel not found at {path}")
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["family_id"])
    if not ids:
        raise ReplayError(f"frozen panel at {path} contains no families")
    return ids


# --------------------------------------------------------------------- #
# Stratum assignment
# --------------------------------------------------------------------- #
def tertile_cuts(values: Iterable) -> tuple:
    """Sorted-value cut points at ``n//3`` and ``2*n//3``.

    Index-based rather than interpolated so the cut points are exactly two
    observed family lengths, which makes the boundary cases checkable by
    hand against the reference run.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n < 3:
        raise ReplayError(
            f"cannot derive tertiles from {n} value(s); the frozen panel "
            f"must supply every family's reference length")
    return ordered[n // 3], ordered[2 * n // 3]


def length_stratum(value: float, cuts: tuple) -> str:
    """``short`` below t1, ``long`` above t2, ``median`` in between."""
    t1, t2 = cuts
    if value < t1:
        return "short"
    if value > t2:
        return "long"
    return "median"


def risk_stratum(compliance_level: int) -> str:
    """Collapse the ordinal 0-3 compliance level to three risk strata."""
    if compliance_level <= 0:
        return "compliant"
    if compliance_level == 1:
        return "partial"
    return "noncompliant"


def build_strata_grid(family_ids: Sequence[str], lengths: Mapping,
                      risks: Mapping) -> dict:
    """Assign every family to a (length, risk) cell.

    Fails closed if any family lacks a reference length or a risk label:
    silently dropping one would change the grid, and therefore change the
    pre-registered selection.
    """
    missing_length = sorted(f for f in family_ids if f not in lengths)
    missing_risk = sorted(f for f in family_ids if f not in risks)
    if missing_length or missing_risk:
        raise ReplayError(
            f"stratification inputs are incomplete: "
            f"{len(missing_length)} family(ies) without a frozen 9B length "
            f"{missing_length[:5]} and {len(missing_risk)} without an "
            f"adjudicated risk label {missing_risk[:5]}")
    cuts = tertile_cuts(lengths[f] for f in family_ids)
    cells: dict = {(length, risk): [] for length in LENGTH_STRATA
                   for risk in RISK_STRATA}
    for family in sorted(family_ids):
        cell = (length_stratum(lengths[family], cuts),
                risk_stratum(risks[family]))
        cells[cell].append(family)
    return {"cuts": cuts, "cells": cells}


# --------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------- #
def _rank_within_cell(members: Sequence[str], lengths: Mapping) -> list:
    """Order a cell's families by representativeness, deterministically.

    Closest to the cell's own median length first; ``family_id`` breaks
    ties, so the result never depends on dict or filesystem ordering.
    """
    cell_median = statistics.median(lengths[m] for m in members)
    return sorted(members, key=lambda f: (abs(lengths[f] - cell_median), f))


def _extra_priority(cell: tuple, populations: Mapping,
                    risk_selected: Mapping) -> tuple:
    """Sort key for allocating the three remaining slots.

    Largest cell first; among equal populations prefer the risk stratum with
    the fewest families selected so far, so the extras do not all land on one
    risk level merely because several cells tie on population; the declared
    cell order breaks any remaining tie.
    """
    length, risk = cell
    return (-populations[cell], risk_selected[risk],
            LENGTH_STRATA.index(length), RISK_STRATA.index(risk))


def select_eligibility_families(family_ids: Sequence[str], lengths: Mapping,
                                risks: Mapping) -> dict:
    """The pre-registered 12-family subset, with its full audit trail."""
    if N_EXTRAS < 0:  # pragma: no cover - guards the declared constants
        raise ReplayError(
            f"{N_ELIGIBILITY_FAMILIES} families cannot cover {N_CELLS} cells")
    grid = build_strata_grid(family_ids, lengths, risks)
    cuts, cells = grid["cuts"], grid["cells"]
    empty = [f"{length}|{risk}" for (length, risk), members in cells.items()
             if not members]
    if empty:
        raise ReplayError(
            f"strata cell(s) {empty} are empty, so one family per cell cannot "
            f"cover the grid; the pre-registered allocation assumes all "
            f"{N_CELLS} cells are populated")

    populations = {cell: len(members) for cell, members in cells.items()}
    ranked = {cell: _rank_within_cell(members, lengths)
              for cell, members in cells.items()}
    cell_medians = {cell: statistics.median(lengths[m] for m in members)
                    for cell, members in cells.items()}

    chosen = {cell: [ranked[cell][0]] for cell in cells}
    risk_selected = {risk: sum(1 for (_, r) in chosen if r == risk)
                     for risk in RISK_STRATA}
    next_rank = {cell: 1 for cell in cells}
    # One extra per cell, so the three largest cells each gain a second
    # family. Without this the single largest cell wins every round — its
    # population never changes — and all three extras pile into it, which
    # quietly unbalances the length strata the selection exists to cover.
    extra_given: set = set()
    extras = []
    for _ in range(N_EXTRAS):
        available = [cell for cell in cells
                     if cell not in extra_given
                     and next_rank[cell] < len(ranked[cell])]
        if not available:  # pragma: no cover - panel would have to be tiny
            raise ReplayError(
                "no cell has a further family available for the remaining "
                "eligibility slot")
        cell = min(available,
                   key=lambda c: _extra_priority(c, populations,
                                                 risk_selected))
        family = ranked[cell][next_rank[cell]]
        chosen[cell].append(family)
        next_rank[cell] += 1
        extra_given.add(cell)
        risk_selected[cell[1]] += 1
        extras.append({
            "cell": f"{cell[0]}|{cell[1]}",
            "family_id": family,
            "cell_population": populations[cell],
            "reason": ("largest cell not yet given an extra; ties favour the "
                       "risk stratum with the fewest families selected so "
                       "far, then the declared cell order"),
        })

    selected = sorted(family for members in chosen.values()
                      for family in members)
    if len(selected) != N_ELIGIBILITY_FAMILIES:
        raise ReplayError(  # pragma: no cover - allocation invariant
            f"selected {len(selected)} families, expected "
            f"{N_ELIGIBILITY_FAMILIES}")

    cells_report = {}
    for cell, members in chosen.items():
        length, risk = cell
        cells_report[f"{length}|{risk}"] = {
            "length_stratum": length,
            "risk_stratum": risk,
            "cell_population": populations[cell],
            "cell_median_length": cell_medians[cell],
            "n_selected": len(members),
            "selected": [{
                "family_id": family,
                "reference_median_output_tokens": lengths[family],
                "distance_to_cell_median":
                    abs(lengths[family] - cell_medians[cell]),
                "max_compliance_level": risks[family],
                "rank_in_cell": ranked[cell].index(family),
            } for family in members],
        }
    return {
        "n_selected_families": len(selected),
        "selected_family_ids": selected,
        "selected_families_sha256": selected_families_sha256(selected),
        "length_cuts": {"t1": cuts[0], "t2": cuts[1]},
        "length_strata": list(LENGTH_STRATA),
        "risk_strata": list(RISK_STRATA),
        "risk_rule": ("max adjudicated compliance_level per family; "
                      "compliant=0, partial=1, noncompliant=2 or 3"),
        "cells": cells_report,
        "by_length_stratum": {
            s: sum(c["n_selected"] for c in cells_report.values()
                   if c["length_stratum"] == s) for s in LENGTH_STRATA},
        "by_risk_stratum": {
            s: sum(c["n_selected"] for c in cells_report.values()
                   if c["risk_stratum"] == s) for s in RISK_STRATA},
        "extra_allocation": extras,
        "n_grid_cells": N_CELLS,
        "n_extras": N_EXTRAS,
        "recipe": SELECTION_RECIPE,
        "deterministic": True,
        "uses_candidate_target_information": False,
    }


# --------------------------------------------------------------------- #
# Derivation from frozen, committed evidence
# --------------------------------------------------------------------- #
def resolve_frozen_input_paths(repo_root: str | Path) -> dict:
    """Locate the three frozen inputs by following the committed pointers.

    Paths are read from ``frozen_9b_reference.json`` and
    ``scale_profiles.json`` rather than hardcoded, so the selection follows
    the frozen artifacts if they are ever re-pointed.
    """
    root = Path(repo_root)
    reference_path = root / FROZEN_9B_REFERENCE
    profiles_path = root / SCALE_PROFILES
    for path, what in ((reference_path, "frozen 9B reference pointer"),
                       (profiles_path, "scale profiles")):
        if not path.exists():
            raise ReplayError(f"{what} not found at {path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profile = profiles.get(SCALE_PROFILE)
    if not isinstance(profile, Mapping):
        raise ReplayError(
            f"{profiles_path} has no {SCALE_PROFILE!r} profile")
    run_dir = reference.get("run_dir")
    panel = profile.get("validated_families")
    output_dir = profile.get("output_dir")
    missing = [name for name, value in
               (("frozen_9b_reference.run_dir", run_dir),
                ("scale_c.validated_families", panel),
                ("scale_c.output_dir", output_dir)) if not value]
    if missing:
        raise ReplayError(f"frozen pointers are missing {missing}")
    return {
        "reference_outputs": root / run_dir / REFERENCE_OUTPUTS_FILE,
        "panel": root / panel,
        "adjudicated_labels": root / output_dir / ADJUDICATED_LABELS_FILE,
        "run_dir": run_dir,
        "reference_run_id": reference.get("run_id"),
        "panel_sha256_expected": reference.get("validated_families_sha256"),
    }


def _relative_to(root: Path, path: Path) -> str:
    """``path`` relative to ``root``, or absolute if it is not under it."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def derive_frozen_selection(repo_root: str | Path) -> dict:
    """Re-derive the pre-registered selection from frozen evidence.

    This is what makes the selection externally pre-registered: the gate
    calls it and compares, so a report cannot authorize a different subset
    by recording matching ids and digest of its own choosing.

    Raises:
        ReplayError: If a pointer or input is missing, the panel digest does
            not match the frozen value, or any family lacks a stratum input.
            All are fail-closed: an underivable selection must block the run
            rather than fall back to trusting the report.
    """
    paths = resolve_frozen_input_paths(repo_root)
    root = Path(repo_root)
    panel = Path(paths["panel"])
    panel_sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    expected_sha = paths.get("panel_sha256_expected")
    if expected_sha and panel_sha != expected_sha:
        raise ReplayError(
            f"the panel at {panel} hashes to {panel_sha} but the frozen 9B "
            f"reference names {expected_sha}; refusing to derive an "
            f"eligibility selection from a panel that is not the frozen one")
    families = panel_family_ids(panel)
    lengths = reference_family_lengths(paths["reference_outputs"])
    risks = reference_family_risk(paths["adjudicated_labels"])
    selection = select_eligibility_families(families, lengths, risks)
    selection.update({
        "derived_from": {
            # Repo-relative so the derivation record is portable: the
            # selection is a property of the frozen evidence, not of the
            # directory this checkout happens to live in.
            "panel": _relative_to(root, paths["panel"]),
            "panel_validated_families_sha256": panel_sha,
            "reference_outputs": _relative_to(root, paths["reference_outputs"]),
            "reference_run_id": paths.get("reference_run_id"),
            "adjudicated_labels": _relative_to(
                root, paths["adjudicated_labels"]),
        },
        "n_panel_families": len(families),
    })
    return selection
