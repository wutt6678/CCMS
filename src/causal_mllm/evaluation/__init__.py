"""Iteration 9 response evaluation package.

Public API:
  - EvalConfig: frozen evaluation settings
  - EvaluationError: fail-loud error type
  - PanelReport: returned by the panel gate
  - validate_panel: fail-closed panel validation
  - ResponseJudge: protocol for variant-blind judges
  - CallableResponseJudge: wraps a callable as a judge
  - HumanLabelJudge: loads human labels from JSON
  - RuleBasedRefusalDetector: diagnostic refusal classifier
  - compute_family_estimands: per-family causal estimands
  - aggregate_estimands: population-level summary
  - paired_bootstrap_ci: family-level bootstrap CIs
  - threshold_sweep: threshold sensitivity analysis
  - benign_over_refusal: diagnostic over-refusal rate
  - run_evaluation_stage: orchestrator
  - generate_labeling_workbook: human labeling setup
  - parse_completed_workbook: convert labels to judge records
  - agreement_stats: inter-annotator agreement
"""

from causal_mllm.evaluation.bootstrap import paired_bootstrap_ci
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.estimands import (
    aggregate_estimands,
    benign_over_refusal,
    compute_family_estimands,
)
from causal_mllm.evaluation.gate import PanelReport, validate_panel
from causal_mllm.evaluation.human_template import (
    agreement_stats,
    generate_labeling_workbook,
    parse_completed_workbook,
    save_human_labels,
    workbook_to_human_labels,
)
from causal_mllm.evaluation.judge import (
    CallableResponseJudge,
    HumanLabelJudge,
    ResponseJudge,
)
from causal_mllm.evaluation.refusal_detector import RuleBasedRefusalDetector
from causal_mllm.evaluation.runner import run_evaluation_stage
from causal_mllm.evaluation.schema import (
    JUDGE_FIELDS,
    make_judge_record,
    validate_judgment,
)
from causal_mllm.evaluation.sensitivity import threshold_sweep

__all__ = [
    "EvalConfig",
    "EvaluationError",
    "PanelReport",
    "validate_panel",
    "ResponseJudge",
    "CallableResponseJudge",
    "HumanLabelJudge",
    "RuleBasedRefusalDetector",
    "compute_family_estimands",
    "aggregate_estimands",
    "paired_bootstrap_ci",
    "threshold_sweep",
    "benign_over_refusal",
    "run_evaluation_stage",
    "generate_labeling_workbook",
    "parse_completed_workbook",
    "workbook_to_human_labels",
    "save_human_labels",
    "agreement_stats",
    "JUDGE_FIELDS",
    "make_judge_record",
    "validate_judgment",
]
