"""Causal family construction pipeline.

Stages:
  select    (Iteration 3) — candidate selection with rejection reasons
  atoms     (Iteration 4) — family-level comparative atom extraction
  annotate  (Iteration 5A) — semantic annotation resolution
  harmonize (Iteration 5B) — canonical terminal query q* construction
  variants  (Iteration 5C) — six gated, provenance-tracked generators
"""

from causal_mllm.construction.annotation import (
    AnnotationError,
    AtomAnnotator,
    CallableAnnotator,
    ManualFileAnnotator,
    apply_annotations,
)
from causal_mllm.construction.atoms import (
    AtomExtraction,
    AtomExtractionError,
    extract_family_atoms,
)
from causal_mllm.construction.families import (
    build_family_skeleton,
    build_family_skeletons,
)
from causal_mllm.construction.grounding import flag_grounding_issues
from causal_mllm.construction.harmonize import (
    CallableHarmonizer,
    ManualHarmonizer,
    TerminalHarmonizationError,
    apply_terminal_harmonization,
    canonical_terminal,
)
from causal_mllm.construction.readiness import (
    L0_STRUCTURAL,
    L1_SEMANTIC,
    L2_VARIANT_READY,
    VariantPrerequisiteError,
    assert_variant_ready,
    family_readiness,
    semantic_gaps,
)
from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionRejection,
    SelectionResult,
    assert_canonical,
    build_family_review_flags,
    build_selection_report,
    group_into_family_units,
    run_selection,
    select_candidates,
)
from causal_mllm.construction.variants import (
    VARIANT_GENERATORS,
    VariantConstructionError,
    build_family_variants,
    validate_variant_trajectory,
)

__all__ = [
    "AnnotationError",
    "AtomAnnotator",
    "AtomExtraction",
    "AtomExtractionError",
    "CallableAnnotator",
    "CallableHarmonizer",
    "L0_STRUCTURAL",
    "L1_SEMANTIC",
    "L2_VARIANT_READY",
    "ManualFileAnnotator",
    "ManualHarmonizer",
    "SelectionConfig",
    "SelectionRejection",
    "SelectionResult",
    "TerminalHarmonizationError",
    "VARIANT_GENERATORS",
    "VariantConstructionError",
    "VariantPrerequisiteError",
    "apply_annotations",
    "apply_terminal_harmonization",
    "assert_canonical",
    "assert_variant_ready",
    "build_family_review_flags",
    "build_family_skeleton",
    "build_family_skeletons",
    "build_family_variants",
    "build_selection_report",
    "canonical_terminal",
    "extract_family_atoms",
    "family_readiness",
    "flag_grounding_issues",
    "group_into_family_units",
    "run_selection",
    "select_candidates",
    "semantic_gaps",
    "validate_variant_trajectory",
]
