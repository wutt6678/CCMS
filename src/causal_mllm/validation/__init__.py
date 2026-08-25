"""Automatic validation layer (Iteration 6).

Judges (behavioral risk backends) and the validation stage that
decides research validity of built families.
"""

from causal_mllm.validation.judges import (
    JUDGED_VARIANTS,
    CallableJudge,
    JudgeError,
    ManualFileJudge,
    RiskJudge,
)
from causal_mllm.validation.relations import (
    FACTORIAL_CELLS,
    validate_factorial_relations,
)
from causal_mllm.validation.stage import (
    DEFAULT_THETA,
    automatic_family_checks,
    run_validation_stage,
    strict_causal_decision,
)

__all__ = [
    "JUDGED_VARIANTS",
    "CallableJudge",
    "DEFAULT_THETA",
    "FACTORIAL_CELLS",
    "JudgeError",
    "ManualFileJudge",
    "RiskJudge",
    "automatic_family_checks",
    "run_validation_stage",
    "strict_causal_decision",
    "validate_factorial_relations",
]
