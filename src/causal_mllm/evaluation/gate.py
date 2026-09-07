"""Fail-closed panel validation (Iteration 9; scale-agnostic in 10).

Before any judging begins, the replay panel must pass a strict gate:
exactly N families × 6 variants records (N declared by the caller —
Scale-B 20, Scale-C 100), zero failures, truncation within the registered
panel tolerance, pinned revision, and every finish reason either a natural
termination or the cap on a record that says it reached the cap.

A panel that fails the gate is NEVER judged — EvaluationError halts
the evaluation stage.

TRUNCATION: ONE STANDARD, DEFINED ELSEWHERE
This gate used to require zero truncated records while the Iteration 11
completion gate accepted up to 2% overall with a 5-point variant spread. The
frozen Qwen3.5-9B reference truncated nothing, so both standards had always
agreed and nothing compared them. Iteration 11 replayed four smaller
checkpoints under the same frozen cap: three passed the completion gate and
were then refused here, after the judging budget had been spent.

Both thresholds now live in :mod:`causal_mllm.replay.truncation` and this gate
imports them rather than restating its own, so a panel is held to the same
standard from the moment it is replayed to the moment it is evaluated. There
is deliberately no argument here to loosen it: no tolerance parameter, no
per-run override, no CLI flag. The 12-family ELIGIBILITY gate stays at zero
truncation and is a separate screening contract in ``replay.confirmatory``.

Every report this gate returns carries what it measured -- thresholds, counts,
rates, spread and the affected cells -- so a reader can see the truncation a
panel was accepted with instead of inferring that it had none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.replay.truncation import (
    MAX_TRUNCATION_RATE,
    MAX_VARIANT_SPREAD,
    NATURAL_FINISH_REASONS,
    finish_reason_violations,
    measure_truncation,
)

REPLAY_OUTPUTS_FILE = "replay_outputs.jsonl"
REPLAY_FAILURES_FILE = "replay_failures.jsonl"
REPLAY_REPORT_FILE = "replay_report.json"

# Scale-B defaults retained for backwards compatibility; callers for other
# panels (e.g. Scale-C 100 families) must pass expected_n_families.
EXPECTED_N_FAMILIES = 20
EXPECTED_N_VARIANTS = len(ALL_VARIANT_NAMES)  # 6
EXPECTED_N_RECORDS = EXPECTED_N_FAMILIES * EXPECTED_N_VARIANTS  # 120

#: Derived, not restated: a natural termination is defined once, in the module
#: that also defines what a truncated record is.
VALID_FINISH_REASONS = set(NATURAL_FINISH_REASONS)


@dataclass(frozen=True)
class PanelReport:
    """Run metadata returned when the panel gate passes."""

    run_id: str
    run_dir: str
    n_families: int
    n_records: int
    provenance: dict = field(default_factory=dict)
    #: What the gate measured about truncation, thresholds included. Present
    #: even when nothing was truncated, because "the check ran and found none"
    #: and "the check did not run" have to be distinguishable in a report.
    truncation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "n_families": self.n_families,
            "n_records": self.n_records,
            "provenance": dict(self.provenance),
            "truncation": dict(self.truncation),
        }


def validate_panel(
    run_dir: str | Path,
    expected_n_families: int | None = None,
) -> tuple[PanelReport, list[dict]]:
    """Validate the replay panel; fail-closed.

    Args:
        run_dir: Replay run directory carrying the outputs/failures/report.
        expected_n_families: Declared panel size (from the validated
            families file). None = legacy Scale-B default (20).

    Returns:
        (PanelReport, records) — the report carries run metadata;
        records is the list of replay output dicts.

    Raises:
        EvaluationError: On ANY gate violation.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []
    exp_families = (
        expected_n_families
        if expected_n_families is not None else EXPECTED_N_FAMILIES)
    exp_records = exp_families * EXPECTED_N_VARIANTS

    # --- Load report ---
    report_path = run_dir / REPLAY_REPORT_FILE
    if not report_path.exists():
        raise EvaluationError(
            f"replay report not found: {report_path}")
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    # --- Load failures ---
    failures_path = run_dir / REPLAY_FAILURES_FILE
    if not failures_path.exists():
        raise EvaluationError(
            f"replay failures file not found: {failures_path}")
    failures = read_jsonl(failures_path)
    if failures:
        errors.append(
            f"zero failures required, got {len(failures)} failure(s)")

    # --- Load outputs ---
    outputs_path = run_dir / REPLAY_OUTPUTS_FILE
    if not outputs_path.exists():
        raise EvaluationError(
            f"replay outputs file not found: {outputs_path}")
    records = read_jsonl(outputs_path)

    # --- Record count ---
    if len(records) != exp_records:
        errors.append(
            f"expected {exp_records} records "
            f"({exp_families} families × {EXPECTED_N_VARIANTS} "
            f"variants), got {len(records)}")

    # --- Family × variant coverage ---
    unique_families = {r.get("family_id") for r in records}
    if len(unique_families) != exp_families:
        errors.append(
            f"expected {exp_families} families, "
            f"got {len(unique_families)}")
    for family_id in sorted(unique_families):
        family_variants = {
            r.get("variant") for r in records
            if r.get("family_id") == family_id
        }
        for v in ALL_VARIANT_NAMES:
            if v not in family_variants:
                errors.append(
                    f"{family_id}: missing variant '{v}'")

    # --- Output diagnostics completeness ---
    for i, rec in enumerate(records):
        if "finish_reason" not in rec:
            errors.append(f"record {i}: missing finish_reason")
        if "hit_max_new_tokens" not in rec:
            errors.append(f"record {i}: missing hit_max_new_tokens")

    # --- Truncation, measured once and recorded pass or fail ---
    # The measurement is taken here and reused below, so the count a report
    # states is the count the gate decided on. Zero truncation still passes:
    # the thresholds admit up to MAX_TRUNCATION_RATE overall with a variant
    # spread up to MAX_VARIANT_SPREAD, and a panel with none is inside both.
    truncation = measure_truncation(records)
    errors.extend(truncation["violations"])
    if (truncation["thresholds"]["max_overall_rate"] != MAX_TRUNCATION_RATE
            or truncation["thresholds"]["max_variant_spread"]
            != MAX_VARIANT_SPREAD):
        # Fail closed on our own wiring rather than publish a report whose
        # advertised tolerance is not the one that was applied.
        errors.append(
            f"internal error: the truncation measurement applied thresholds "
            f"{truncation['thresholds']} but this gate imported "
            f"{MAX_TRUNCATION_RATE}/{MAX_VARIANT_SPREAD}, so one panel would "
            f"be judged against two different standards")

    # --- Pinned revision ---
    provenance = report.get("provenance", {})
    if not provenance.get("revision_pinned"):
        errors.append("revision_pinned must be True")

    # --- Clean-tree acceptance criteria (Iteration 8 hardening) ---
    # git_dirty must be explicitly False (not missing, not True)
    git_dirty = provenance.get("git_dirty")
    if git_dirty is not False:
        errors.append(
            f"git_dirty must be False for clean-tree provenance, "
            f"got {git_dirty!r}")

    # requested and resolved revisions must be nonempty and equal
    requested_rev = provenance.get("requested_model_revision")
    resolved_rev = provenance.get("resolved_model_revision")
    if not requested_rev:
        errors.append("requested_model_revision must be nonempty")
    if not resolved_rev:
        errors.append("resolved_model_revision must be nonempty")
    if requested_rev and resolved_rev and requested_rev != resolved_rev:
        errors.append(
            f"requested_model_revision ({requested_rev}) must equal "
            f"resolved_model_revision ({resolved_rev})")

    # Report/record run-ID consistency
    report_run_id = report.get("run_id")
    record_run_ids = {r.get("run_id") for r in records}
    if not report_run_id:
        errors.append("report run_id is missing")
    if len(record_run_ids) != 1:
        errors.append(
            f"records must have exactly one run_id, got {record_run_ids}")
    if report_run_id and len(record_run_ids) == 1:
        if report_run_id != list(record_run_ids)[0]:
            errors.append(
                f"report run_id ({report_run_id}) must match record run_id "
                f"({list(record_run_ids)[0]})")

    # Revision consistency between report and records
    record_revisions = {r.get("model_revision") for r in records}
    if len(record_revisions) != 1:
        errors.append(
            f"records must have exactly one model_revision, got {record_revisions}")
    if resolved_rev and len(record_revisions) == 1:
        if resolved_rev != list(record_revisions)[0]:
            errors.append(
                f"report resolved_model_revision ({resolved_rev}) must match "
                f"record model_revision ({list(record_revisions)[0]})")

    # N×6 unique family/variant pairs
    family_variant_pairs = {
        (r.get("family_id"), r.get("variant")) for r in records
    }
    if len(family_variant_pairs) != exp_records:
        errors.append(
            f"expected {exp_records} unique (family_id, variant) pairs, "
            f"got {len(family_variant_pairs)}")

    # Nonempty responses
    empty_responses = [
        i for i, r in enumerate(records)
        if not r.get("response")
    ]
    if empty_responses:
        errors.append(
            f"all responses must be nonempty, got {len(empty_responses)} "
            f"empty response(s)")

    # --- Finish reasons ---
    # A natural termination always passes; the cap passes only on a record
    # counted as truncated above, so the two provenance flags cannot
    # contradict each other without the panel being refused.
    errors.extend(finish_reason_violations(records))

    # --- Fail-closed ---
    if errors:
        msg = "panel gate FAILED:\n  " + "\n  ".join(errors)
        raise EvaluationError(msg)

    panel = PanelReport(
        run_id=report.get("run_id", "unknown"),
        run_dir=str(run_dir),
        n_families=len(unique_families),
        n_records=len(records),
        provenance=provenance,
        truncation=truncation,
    )
    return panel, records
