"""Causal family construction pipeline.

Stages:
  select   (Iteration 3) — candidate selection with rejection reasons
  atoms    (Iteration 4) — family-level comparative atom extraction
  variants (Iteration 5) — six variant generators          [stub]
"""

from causal_mllm.construction.atoms import (
    AtomExtraction,
    AtomExtractionError,
    extract_family_atoms,
)
from causal_mllm.construction.families import (
    build_family_skeleton,
    build_family_skeletons,
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

__all__ = [
    "AtomExtraction",
    "AtomExtractionError",
    "SelectionConfig",
    "SelectionRejection",
    "SelectionResult",
    "assert_canonical",
    "build_family_review_flags",
    "build_family_skeleton",
    "build_family_skeletons",
    "build_selection_report",
    "extract_family_atoms",
    "group_into_family_units",
    "run_selection",
    "select_candidates",
]
