"""Iteration 9 response evaluation package.

Public API:
  - EvalConfig: frozen evaluation settings
  - EvaluationError: fail-loud error type
  - PanelReport: returned by the panel gate
  - validate_panel: fail-closed panel validation
  - ResponseJudge: protocol for variant-blind judges
  - CallableResponseJudge: wraps a callable as a judge
  - HumanLabelJudge: loads human labels from JSON
  - MultimodalLLMJudge: LLM-based multimodal judge
  - RuleBasedRefusalDetector: diagnostic refusal classifier
  - compute_family_estimands: per-family causal estimands
  - REQUIRED_VARIANTS: the six variants a family needs to contribute one
  - incomplete_families: which families cannot contribute an estimand
  - aggregate_estimands: population-level summary
  - paired_bootstrap_ci: family-level bootstrap CIs
  - paired_bootstrap_samples: the resample distribution behind those CIs
  - bootstrap_two_sided_p: p-value from that distribution, same seed
  - family_sign_test: exact family-level sign test (11.8 sensitivity)
  - holm_bonferroni: the pre-declared family-wise correction
  - threshold_sweep: threshold sensitivity analysis
  - benign_over_refusal: diagnostic over-refusal rate
  - restrict_panel_to_labels: drop unlabelled cells and the families they broke
  - run_evaluation_stage: orchestrator
  - generate_labeling_workbook: human labeling setup
  - parse_completed_workbook: convert labels to judge records
  - agreement_stats: inter-annotator agreement
  - compute_judge_agreement: multi-judge agreement metrics
"""

from causal_mllm.evaluation.adjudication import (
    ENSEMBLE_BACKEND,
    SCORE_EPSILON,
    LLMAdjudicator,
    adjudicate_deterministic,
    adjudicate_pairwise_with_model,
    enforce_coherence,
    judgments_disagree,
    validate_llm_judgment_fields,
)
from causal_mllm.evaluation.agreement import (
    compute_judge_agreement,
    compute_pairwise_agreement,
)
from causal_mllm.evaluation.bootstrap import (
    ESTIMAND_NAMES,
    bootstrap_two_sided_p,
    paired_bootstrap_ci,
    paired_bootstrap_samples,
)
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.estimands import (
    REQUIRED_VARIANTS,
    aggregate_estimands,
    benign_over_refusal,
    compute_family_estimands,
    incomplete_families,
)
from causal_mllm.evaluation.gate import PanelReport, validate_panel
from causal_mllm.evaluation.human_template import (
    agreement_stats,
    generate_labeling_workbook,
    parse_completed_workbook,
    save_human_labels,
    save_llm_ensemble_labels,
    workbook_to_human_labels,
)
from causal_mllm.evaluation.hypotheses import (
    family_sign_test,
    holm_bonferroni,
)
from causal_mllm.evaluation.judge import (
    CallableResponseJudge,
    HumanLabelJudge,
    LLMEnsembleLabelJudge,
    ResponseJudge,
)
from causal_mllm.evaluation.llm_judge import (
    LLMJudgeConfig,
    MultimodalLLMJudge,
)
from causal_mllm.evaluation.refusal_detector import RuleBasedRefusalDetector
from causal_mllm.evaluation.runner import (
    restrict_panel_to_labels,
    run_evaluation_stage,
)
from causal_mllm.evaluation.schema import (
    JUDGE_FIELDS,
    make_judge_record,
    validate_judgment,
)
from causal_mllm.evaluation.sensitivity import (
    judge_model_sensitivity,
    threshold_sweep,
)

__all__ = [
    "EvalConfig",
    "EvaluationError",
    "PanelReport",
    "validate_panel",
    "ResponseJudge",
    "CallableResponseJudge",
    "HumanLabelJudge",
    "LLMEnsembleLabelJudge",
    "MultimodalLLMJudge",
    "LLMJudgeConfig",
    "RuleBasedRefusalDetector",
    "compute_family_estimands",
    "incomplete_families",
    "REQUIRED_VARIANTS",
    "aggregate_estimands",
    "paired_bootstrap_ci",
    "paired_bootstrap_samples",
    "bootstrap_two_sided_p",
    "ESTIMAND_NAMES",
    "family_sign_test",
    "holm_bonferroni",
    "threshold_sweep",
    "benign_over_refusal",
    "run_evaluation_stage",
    "restrict_panel_to_labels",
    "generate_labeling_workbook",
    "parse_completed_workbook",
    "workbook_to_human_labels",
    "save_human_labels",
    "save_llm_ensemble_labels",
    "agreement_stats",
    "compute_judge_agreement",
    "compute_pairwise_agreement",
    "ENSEMBLE_BACKEND",
    "SCORE_EPSILON",
    "LLMAdjudicator",
    "adjudicate_deterministic",
    "adjudicate_pairwise_with_model",
    "judgments_disagree",
    "enforce_coherence",
    "validate_llm_judgment_fields",
    "JUDGE_FIELDS",
    "make_judge_record",
    "validate_judgment",
    "judge_model_sensitivity",
]
