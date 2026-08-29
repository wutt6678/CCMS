"""Fail-closed panel validation (Iteration 9).

Before any judging begins, the replay panel must pass a strict gate:
exactly 20 families × 6 variants = 120 records, zero failures, zero
truncation, pinned revision, and all finish reasons in {eos, stop}.

A panel that fails the gate is NEVER judged — EvaluationError halts
the evaluation stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.evaluation.errors import EvaluationError

REPLAY_OUTPUTS_FILE = "replay_outputs.jsonl"
REPLAY_FAILURES_FILE = "replay_failures.jsonl"
REPLAY_REPORT_FILE = "replay_report.json"

EXPECTED_N_FAMILIES = 20
EXPECTED_N_VARIANTS = len(ALL_VARIANT_NAMES)  # 6
EXPECTED_N_RECORDS = EXPECTED_N_FAMILIES * EXPECTED_N_VARIANTS  # 120
VALID_FINISH_REASONS = {"eos", "stop"}


@dataclass(frozen=True)
class PanelReport:
    """Run metadata returned when the panel gate passes."""

    run_id: str
    run_dir: str
    n_families: int
    n_records: int
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "n_families": self.n_families,
            "n_records": self.n_records,
            "provenance": dict(self.provenance),
        }


def validate_panel(run_dir: str | Path) -> tuple[PanelReport, list[dict]]:
    """Validate the replay panel; fail-closed.

    Returns:
        (PanelReport, records) — the report carries run metadata;
        records is the list of 120 replay output dicts.

    Raises:
        EvaluationError: On ANY gate violation.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []

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
    if len(records) != EXPECTED_N_RECORDS:
        errors.append(
            f"expected {EXPECTED_N_RECORDS} records "
            f"({EXPECTED_N_FAMILIES} families × {EXPECTED_N_VARIANTS} "
            f"variants), got {len(records)}")

    # --- Family × variant coverage ---
    unique_families = {r.get("family_id") for r in records}
    if len(unique_families) != EXPECTED_N_FAMILIES:
        errors.append(
            f"expected {EXPECTED_N_FAMILIES} families, "
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

    # --- Zero truncation ---
    truncated = [r for r in records
                 if r.get("hit_max_new_tokens") is True]
    if truncated:
        errors.append(
            f"zero truncation required, got {len(truncated)} "
            f"truncated response(s)")

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

    # 120 unique family/variant pairs
    family_variant_pairs = {
        (r.get("family_id"), r.get("variant")) for r in records
    }
    if len(family_variant_pairs) != EXPECTED_N_RECORDS:
        errors.append(
            f"expected {EXPECTED_N_RECORDS} unique (family_id, variant) pairs, "
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
    bad_finish = [
        r for r in records
        if r.get("finish_reason") not in VALID_FINISH_REASONS
    ]
    if bad_finish:
        reasons = {r.get("finish_reason") for r in bad_finish}
        errors.append(
            f"all finish_reason must be in {VALID_FINISH_REASONS}, "
            f"got {len(bad_finish)} record(s) with: {reasons}")

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
    )
    return panel, records
