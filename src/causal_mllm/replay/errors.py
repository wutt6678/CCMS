"""Fail-loud error types for the frozen replay runner (Iteration 8).

Errors are CLASSIFIED, never converted into responses: an OOM, a
missing/corrupt media file, or a context-overflow must show up in
``replay_failures.jsonl`` with an error category — never as a
safe/refusal label in Iteration 9.
"""

from __future__ import annotations


class ReplayError(RuntimeError):
    """Base class; carries a stable machine-readable category."""

    category = "replay"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ReplayMediaError(ReplayError):
    """Referenced media missing, unrecorded, or hash-mismatched."""

    category = "media"


class ReplayGenerationError(ReplayError):
    """The backend failed to produce a response."""

    category = "generation"


def classify_error(exc: Exception) -> dict:
    """Stable error record for replay_failures.jsonl."""
    message = str(exc)
    lowered = message.lower()
    try:
        import torch
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
    except ImportError:  # pragma: no cover - torch always present
        is_oom = False
    if isinstance(exc, ReplayMediaError):
        category = "media"
    elif is_oom or "out of memory" in lowered:
        # torch wraps OOM in a plain RuntimeError on some paths; never
        # let one become a response.
        category = "oom"
    elif "context length" in lowered or "maximum context" in lowered:
        category = "context_length"
    else:
        category = "generation"
    return {
        "category": category,
        "type": type(exc).__name__,
        "message": message[:1000],
    }
