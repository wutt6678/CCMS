"""Safety-risk judge backends (Iteration 6).

The automatic validation layer needs behavioral risk scores to decide
strict cross-modal causal-subset membership:

    Risk(T) < θ,  Risk(V) < θ,  Risk(T,V) >= θ

plus the standalone terminal risk Risk(q* alone). These are BEHAVIORAL
claims — they cannot be produced by annotator belief. This module
provides the judge interface; concrete backends:

  * ManualFileJudge — JSON {family_key: {variant_name: score}} for the
    smoke sets (human risk review of the constructed trajectories)
  * CallableJudge   — any LLM/VLM/model callable returning a score in
    [0, 1]; model identity is mandatory provenance

Scores are validated fail-loud: a score outside [0, 1] or a missing
required score raises JudgeError. The real frozen-replay model judge
lands with the inference phase (Iteration 8+) and plugs in through
CallableJudge without changing the validation stage.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional


class JudgeError(ValueError):
    """Raised on malformed or missing risk judgments."""


# Variants whose risk scores the strict-subset decision consumes
JUDGED_VARIANTS = (
    "neutral",        # Risk(H00): benign baseline (Iteration 9)
    "history_reset",  # Risk(q* alone): standalone terminal risk
    "text_only",      # Risk(T)
    "vision_only",    # Risk(V)
    "cross_modal",    # Risk(T,V)
)


def _validate_score(value, family_key: str, variant: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeError(
            f"risk score for {family_key}/{variant} must be a number, "
            f"got {value!r}"
        )
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise JudgeError(
            f"risk score for {family_key}/{variant} must lie in [0, 1], "
            f"got {score}"
        )
    return score


class RiskJudge(ABC):
    """Backend interface: score one constructed trajectory."""

    validation_label: str = "pending"

    @abstractmethod
    def score(self, family_key: str, variant: str,
              messages: list[dict]) -> Optional[float]:
        """Return a risk score in [0, 1], or None if unavailable."""
        ...

    def provenance(self) -> dict:
        return {"backend": self.validation_label}


class ManualFileJudge(RiskJudge):
    """Risk scores from a JSON file: {family_key: {variant: score}}.

    The human-review route for the smoke sets: a reviewer reads each
    constructed trajectory and records an unsafe-compliance risk
    estimate per judged variant.
    """

    validation_label = "human"

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Judge file not found: {self._path}")
        with self._path.open(encoding="utf-8") as f:
            self._data: dict = json.load(f)

    def score(self, family_key: str, variant: str,
              messages: list[dict]) -> Optional[float]:
        value = (self._data.get(family_key) or {}).get(variant)
        if value is None:
            return None
        return _validate_score(value, family_key, variant)

    def provenance(self) -> dict:
        return {"backend": "manual", "file": str(self._path)}


class CallableJudge(RiskJudge):
    """Wraps any callable (LLM/VLM or model wrapper) as a risk judge.

    The callable receives (family_key, variant, messages) where
    messages is a list of turn dicts, and returns a score in [0, 1]
    (or None). Model identity is mandatory provenance.
    """

    validation_label = "llm"

    def __init__(self, fn: Callable[[str, str, list], Optional[float]],
                 *, model_name: str, model_revision: Optional[str] = None,
                 prompt_version: Optional[str] = None,
                 temperature: float = 0.0, seed: Optional[int] = None):
        if not model_name:
            raise JudgeError("CallableJudge requires model_name for "
                             "provenance")
        self._fn = fn
        self._provenance = {
            "backend": "llm",
            "model": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "seed": seed,
        }

    def score(self, family_key: str, variant: str,
              messages: list[dict]) -> Optional[float]:
        value = self._fn(family_key, variant, messages)
        if value is None:
            return None
        return _validate_score(value, family_key, variant)

    def provenance(self) -> dict:
        return dict(self._provenance)
