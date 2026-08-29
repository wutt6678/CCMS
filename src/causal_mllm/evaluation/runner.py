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
import json
from pathlib import Path

from causal_mllm.data.io import write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.evaluation.bootstrap import paired_bootstrap_ci
from causal_mllm.evaluation.config import EvalConfig
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

log = get_logger(__name__)

EVALUATION_OUTPUTS_FILE = "evaluation_outputs.jsonl"
EVALUATION_REPORT_FILE = "evaluation_report.json"


def run_evaluation_stage(
    run_dir: str | Path,
    judge: ResponseJudge | HumanLabelJudge,
    config: EvalConfig | None = None,
    output_root: str | Path | None = None,
) -> dict:
    """Run the full evaluation stage.

    Args:
        run_dir: Path to the replay run to evaluate.
        judge: Response judge backend (variant-blind).
        config: Evaluation settings (defaults: theta=0.5, 5000 bootstrap).
        output_root: Where to write evaluation outputs (defaults to run_dir).

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

    # 2. Judge: run judge over all 120 responses (variant-blind)
    judged_records: list[dict] = []
    provenance = judge.provenance()

    for rec in records:
        # For HumanLabelJudge, use judge_for() with family_id + variant
        if isinstance(judge, HumanLabelJudge):
            judgment = judge.judge_for(rec["family_id"], rec["variant"])
        else:
            # Variant-blind: the judge sees only the conversation context
            # and the response.  We extract system_prompt, history_messages,
            # terminal_query from the replay record (not stored in the
            # replay output, so we pass empty placeholders for now).
            # In practice, the replay record carries the response only;
            # the full conversation would need to be reconstructed from
            # the validated families.  For Iteration 9, we assume the
            # judge has access to the response text only.
            judgment = judge.judge(
                system_prompt="",  # not stored in replay output
                history_messages=[],  # not stored in replay output
                terminal_query="",  # not stored in replay output
                response=rec.get("response", ""),
            )
        judge_rec = make_judge_record(rec, judgment, provenance)
        judged_records.append(judge_rec)

    log.info("Evaluation: judged %d responses", len(judged_records))

    # 3. Diagnostic: run rule-based refusal detector
    refusal_detector = RuleBasedRefusalDetector()
    diagnostic_refusals: list[dict] = []
    for rec in judged_records:
        diag = refusal_detector.judge(rec.get("response", ""))
        diag["family_id"] = rec.get("family_id")
        diag["variant"] = rec.get("variant")
        diagnostic_refusals.append(diag)

    # 4. Estimands: compute family-level causal estimands
    family_estimands = compute_family_estimands(
        judged_records, outcome=config.primary_outcome)
    aggregated = aggregate_estimands(family_estimands)

    # 5. Bootstrap: compute paired bootstrap CIs
    bootstrap_ci = paired_bootstrap_ci(
        family_estimands,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
        seed=config.seed,
    )

    # 6. Sensitivity: threshold sweep
    sensitivity = threshold_sweep(
        family_estimands, thetas=config.theta_grid)

    # 7. Benign over-refusal rate
    bor = benign_over_refusal(judged_records)

    # 8. Persist outputs
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
        "judge_provenance": provenance,
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
