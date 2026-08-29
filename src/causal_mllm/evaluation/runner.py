"""Evaluation orchestrator (Iteration 9).

The evaluation stage consumes a frozen replay panel and produces
causal estimands with bootstrap CIs.  The pipeline:

  1. Gate: validate panel (fail-closed)
  2. Judge: run judge over all 120 responses (variant-blind)
  3. Diagnostic: run rule-based refusal detector
  4. Estimands: compute family-level causal estimands
  5. Bootstrap: compute paired bootstrap CIs
  6. Sensitivity: threshold sweep
  7. Persist: evaluation_outputs.jsonl, evaluation_report.json

The primary outcome is unsafe_compliance_score.  Refusal rates are
reported separately as diagnostics.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.bootstrap import paired_bootstrap_ci
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.estimands import (
    aggregate_estimands,
    benign_over_refusal,
    compute_family_estimands,
)
from causal_mllm.evaluation.gate import validate_panel
from causal_mllm.evaluation.judge import HumanLabelJudge, ResponseJudge
from causal_mllm.evaluation.refusal_detector import RuleBasedRefusalDetector
from causal_mllm.evaluation.schema import make_judge_record
from causal_mllm.evaluation.sensitivity import threshold_sweep
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

log = get_logger(__name__)

EVALUATION_OUTPUTS_FILE = "evaluation_outputs.jsonl"
EVALUATION_REPORT_FILE = "evaluation_report.json"
REPLAY_REPORT_FILE = "replay_report.json"


def _file_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_replay_provenance(
    run_dir: Path,
) -> dict:
    """Load provenance from the replay report.

    Returns the provenance dict, which includes system_prompt_sha256
    and validated_families_sha256.
    """
    report_path = run_dir / REPLAY_REPORT_FILE
    if not report_path.exists():
        raise EvaluationError(
            f"replay report not found: {report_path}")
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    return report.get("provenance", {})


def _load_validated_families(
    families_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, CausalFamily]:
    """Load validated families from an explicit path.

    Args:
        families_path: Path to validated_families.jsonl.
        expected_sha256: If provided, verify the file SHA256 matches.

    Raises:
        EvaluationError: If the file is missing or SHA256 mismatches.
    """
    if not families_path.exists():
        raise EvaluationError(
            f"validated_families.jsonl not found: {families_path} — "
            f"pass --validated-families to the CLI or "
            f"validated_families_path to run_evaluation_stage()")

    # Verify SHA256 against provenance
    if expected_sha256:
        actual_sha = _file_sha256(families_path)
        if actual_sha != expected_sha256:
            raise EvaluationError(
                f"validated_families.jsonl SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha}")

    families = {}
    for rec in read_jsonl(families_path):
        fam = CausalFamily.from_dict(rec)
        families[fam.family_id] = fam
    return families


def _reconstruct_conversation_context(
    family: CausalFamily,
    variant_name: str,
    system_prompt_sha256: str,
    terminal_sha256: str,
) -> tuple[str, list[dict], str]:
    """Reconstruct system_prompt, history_messages, terminal_query from family.

    The system prompt is imported from the frozen replay configuration
    (DEFAULT_SYSTEM_PROMPT) and verified against the replay record's
    system_prompt_sha256.

    Verifies terminal query hash against the replay record.

    Returns:
        (system_prompt, history_messages, terminal_query)

    Raises:
        EvaluationError: If hashes don't match.
    """
    variant = family.variants.get(variant_name)
    if variant is None:
        raise EvaluationError(
            f"variant '{variant_name}' not found in family {family.family_id}")

    # Verify system prompt against the frozen replay config
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if system_prompt_sha256:
        actual_sha = sha256_text(system_prompt)
        if actual_sha != system_prompt_sha256:
            raise EvaluationError(
                f"system prompt SHA256 mismatch for "
                f"{family.family_id}/{variant_name}: "
                f"DEFAULT_SYSTEM_PROMPT hashes to {actual_sha}, "
                f"expected {system_prompt_sha256}")

    # Extract messages
    messages = []
    for msg in variant.messages:
        content_parts = []
        for img in msg.images:
            content_parts.append({"type": "image", "image": img})
        if msg.text is not None:
            content_parts.append({"type": "text", "text": msg.text})
        messages.append({
            "role": msg.role,
            "content": content_parts,
        })

    # The last message is the terminal query
    if not messages:
        raise EvaluationError(
            f"no messages in variant {variant_name} "
            f"of family {family.family_id}")

    terminal_msg = messages[-1]
    terminal_query = ""
    for part in terminal_msg.get("content", []):
        if part.get("type") == "text":
            terminal_query = part.get("text", "")
            break

    # Verify terminal hash
    if terminal_sha256:
        if sha256_text(terminal_query) != terminal_sha256:
            raise EvaluationError(
                f"terminal query hash mismatch for "
                f"{family.family_id}/{variant_name}: "
                f"expected {terminal_sha256}, "
                f"got {sha256_text(terminal_query)}")

    # History is all messages except the last
    history_messages = messages[:-1]

    return system_prompt, history_messages, terminal_query


def run_evaluation_stage(
    run_dir: str | Path,
    judge: ResponseJudge | HumanLabelJudge,
    config: EvalConfig | None = None,
    output_root: str | Path | None = None,
    validated_families_path: str | Path | None = None,
) -> dict:
    """Run the full evaluation stage.

    Args:
        run_dir: Path to the replay run to evaluate.
        judge: Response judge backend (variant-blind).
        config: Evaluation settings (defaults: theta=0.5, 5000 bootstrap).
        output_root: Where to write evaluation outputs (defaults to run_dir).
        validated_families_path: Explicit path to validated_families.jsonl.
            Required — the runner no longer guesses the location.

    Returns:
        The evaluation report dict.

    Raises:
        EvaluationError: On any gate violation or judge failure.
    """
    config = config or EvalConfig()
    run_dir = Path(run_dir)
    output_root = Path(output_root) if output_root else run_dir

    log.info("Evaluation: starting stage on %s", run_dir)

    # 1. Gate: validate panel (fail-closed)
    panel, records = validate_panel(run_dir)
    log.info("Evaluation: panel gate passed (%d records, %d families)",
             panel.n_records, panel.n_families)

    # Load replay provenance for SHA verification
    provenance = _load_replay_provenance(run_dir)
    expected_families_sha = provenance.get("validated_families_sha256")

    # 2. Load validated families from explicit path
    if validated_families_path is None:
        raise EvaluationError(
            "validated_families_path is required — "
            "pass the path to validated_families.jsonl explicitly")
    families_path = Path(validated_families_path)
    families = _load_validated_families(families_path, expected_families_sha)
    log.info(
        "Evaluation: loaded %d validated families from %s "
        "(SHA256 verified)",
        len(families), families_path)

    # 3. Judge: run judge over all 120 responses (variant-blind)
    judged_records: list[dict] = []
    judge_provenance = judge.provenance()

    # For HumanLabelJudge, verify response SHA256 against replay outputs
    if isinstance(judge, HumanLabelJudge):
        expected_response_shas = {
            (rec["family_id"], rec["variant"]): sha256_text(
                rec.get("response", ""))
            for rec in records
        }
        judge.verify_response_shas(expected_response_shas)
        log.info("Evaluation: human label response SHA256 verified "
                 "against %d replay responses", len(expected_response_shas))

    for rec in records:
        family_id = rec["family_id"]
        variant = rec["variant"]

        # For HumanLabelJudge, use judge_for() with family_id + variant
        if isinstance(judge, HumanLabelJudge):
            judgment = judge.judge_for(family_id, variant)
        else:
            # Reconstruct conversation context from validated families
            # and verify against replay hashes
            family = families.get(family_id)
            if family is None:
                raise EvaluationError(
                    f"family {family_id} not found in "
                    f"validated_families.jsonl")

            system_prompt, history_messages, terminal_query = (
                _reconstruct_conversation_context(
                    family, variant,
                    rec.get("system_prompt_sha256", ""),
                    rec.get("terminal_sha256", ""),
                )
            )

            # Variant-blind: the judge sees the conversation context
            # and the response, but NOT the variant name
            judgment = judge.judge(
                system_prompt=system_prompt,
                history_messages=history_messages,
                terminal_query=terminal_query,
                response=rec.get("response", ""),
            )
        judge_rec = make_judge_record(rec, judgment, judge_provenance)
        judged_records.append(judge_rec)

    log.info("Evaluation: judged %d responses", len(judged_records))

    # 4. Diagnostic: run rule-based refusal detector
    refusal_detector = RuleBasedRefusalDetector()
    diagnostic_refusals: list[dict] = []
    for rec in judged_records:
        diag = refusal_detector.judge(rec.get("response", ""))
        diag["family_id"] = rec.get("family_id")
        diag["variant"] = rec.get("variant")
        diagnostic_refusals.append(diag)

    # 5. Estimands: compute family-level causal estimands
    family_estimands = compute_family_estimands(
        judged_records, outcome=config.primary_outcome)
    aggregated = aggregate_estimands(family_estimands)

    # 6. Bootstrap: compute paired bootstrap CIs
    bootstrap_ci = paired_bootstrap_ci(
        family_estimands,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
        seed=config.seed,
    )

    # 7. Sensitivity: threshold sweep
    sensitivity = threshold_sweep(
        family_estimands, thetas=config.theta_grid)

    # 8. Benign over-refusal rate
    bor = benign_over_refusal(judged_records)

    # 9. Persist outputs
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / EVALUATION_OUTPUTS_FILE, judged_records)

    report = {
        "iteration": "9",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "panel_gate": {
            "status": "passed",
            "panel": panel.to_dict(),
        },
        "judge_provenance": judge_provenance,
        "validated_families": {
            "path": str(families_path),
            "sha256": expected_families_sha,
        },
        "estimands": {
            "aggregated": {
                k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in aggregated["estimands"].items()
            },
            "bootstrap_ci": bootstrap_ci,
            "n_families": aggregated["n_families"],
        },
        "threshold_sensitivity": sensitivity,
        "diagnostic_refusal_detector": {
            "provenance": refusal_detector.provenance(),
            "summary": _summarize_diagnostic_refusals(diagnostic_refusals),
        },
        "benign_over_refusal": bor,
    }

    with (output_root / EVALUATION_REPORT_FILE).open(
            "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Evaluation: report written to %s", output_root)
    return report


def _summarize_diagnostic_refusals(diagnostic_refusals: list[dict]) -> dict:
    """Summarize the rule-based refusal detector output."""
    by_variant: dict[str, dict] = {}
    for rec in diagnostic_refusals:
        variant = rec.get("variant")
        if variant not in by_variant:
            by_variant[variant] = {"full": 0, "partial": 0, "none": 0, "total": 0}
        by_variant[variant][rec.get("refusal_type", "none")] += 1
        by_variant[variant]["total"] += 1

    return {
        "by_variant": by_variant,
        "n_total": len(diagnostic_refusals),
    }
