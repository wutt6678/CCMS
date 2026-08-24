"""Causal family construction pipeline.

Stages:
  select   (Iteration 3) — candidate selection with rejection reasons
  atoms    (Iteration 4) — semantic atom extraction        [stub]
  variants (Iteration 5) — six variant generators          [stub]
"""

from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionRejection,
    SelectionResult,
    assert_canonical,
    build_selection_report,
    run_selection,
    select_candidates,
)

__all__ = [
    "SelectionConfig",
    "SelectionRejection",
    "SelectionResult",
    "assert_canonical",
    "build_selection_report",
    "run_selection",
    "select_candidates",
]
