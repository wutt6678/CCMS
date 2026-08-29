"""Fail-loud error types for Iteration 9 response evaluation.

Evaluation errors are NEVER silently absorbed into scores: a
malformed replay panel, a missing judge record, or a broken
factorial relation must raise EvaluationError and halt the stage.
"""

from __future__ import annotations


class EvaluationError(RuntimeError):
    """Base class for evaluation-stage failures."""
