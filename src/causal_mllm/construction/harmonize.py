"""Terminal-query harmonization (Iteration 5B).

All 752 Type-B families require harmonization: the multimodal and
unimodal source dialogues are separately written paraphrases (0%
cross-modality terminal equality measured on the full split), so a
single canonical q*_i must be CONSTRUCTED per family.

The harmonized block is additive provenance — the original skeleton
terminal query is NEVER overwritten:

    validation.terminal_harmonization = {
        "required": true,
        "canonical_q": "...",
        "canonical_sha256": "...",
        "source_mm_q": "...",
        "source_text_q": "...",
        "method": "manual|llm",
        "validation": "human|llm",
        "provenance": {...},
    }

Every variant condition must then reference exactly this string/hash:
q*_00 = q*_10 = q*_01 = q*_11 = q*_shuffle = q*_reset.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from causal_mllm.data.schemas import CausalFamily
from causal_mllm.seeds import sha256_text


class TerminalHarmonizationError(ValueError):
    """Raised when a required harmonization cannot be produced."""


def _source_text_q(family: CausalFamily) -> Optional[str]:
    """Recover the unimodal terminal form from the terminal atom."""
    for atom in family.semantic_atoms:
        if atom.structural_role == "terminal_query":
            form = (atom.surface_forms or {}).get("unimodal_safe")
            if form and form.get("text"):
                return form["text"]
    return None


class TerminalHarmonizer(ABC):
    """Backend interface: produce the canonical q* for one family."""

    method: str = "unknown"
    validation_label: str = "pending"

    @abstractmethod
    def harmonize(self, family_key: str, source_mm_q: str,
                  source_text_q: Optional[str]) -> Optional[str]:
        """Return the canonical query, or None if unavailable."""
        ...

    def provenance(self) -> dict:
        return {"backend": self.validation_label}


class ManualHarmonizer(TerminalHarmonizer):
    """Canonical queries from a JSON file: {family_key: canonical_q}.

    The intended route for the smoke sets: a human reads the mm/text
    terminal pair and writes one canonical query per family.
    """

    method = "manual"
    validation_label = "human"

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Harmonization file not found: {self._path}"
            )
        with self._path.open(encoding="utf-8") as f:
            self._data: dict = json.load(f)

    def harmonize(self, family_key: str, source_mm_q: str,
                  source_text_q: Optional[str]) -> Optional[str]:
        return self._data.get(family_key)

    def provenance(self) -> dict:
        return {"backend": "manual", "file": str(self._path)}


class CallableHarmonizer(TerminalHarmonizer):
    """Wraps any callable (e.g. an LLM) as the harmonization backend.

    The callable receives (family_key, source_mm_q, source_text_q) and
    returns the canonical query string. Model identity is mandatory
    provenance.
    """

    method = "llm"
    validation_label = "llm"

    def __init__(self, fn: Callable[[str, str, Optional[str]], Optional[str]],
                 *, model_name: str, model_revision: Optional[str] = None,
                 prompt_version: Optional[str] = None,
                 temperature: float = 0.0, seed: Optional[int] = None):
        if not model_name:
            raise TerminalHarmonizationError(
                "CallableHarmonizer requires model_name for provenance"
            )
        self._fn = fn
        self._provenance = {
            "backend": "llm",
            "model": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "seed": seed,
        }

    def harmonize(self, family_key: str, source_mm_q: str,
                  source_text_q: Optional[str]) -> Optional[str]:
        return self._fn(family_key, source_mm_q, source_text_q)

    def provenance(self) -> dict:
        return dict(self._provenance)


def apply_terminal_harmonization(
    family: CausalFamily,
    harmonizer: TerminalHarmonizer,
) -> CausalFamily:
    """Return a harmonized COPY of the family; the input is not mutated.

    The skeleton's terminal_query field is preserved as-is; the
    canonical query lives in validation.terminal_harmonization.

    Raises:
        TerminalHarmonizationError: If harmonization is required but no
            canonical query could be produced, or the result is empty.
    """
    harmonized = copy.deepcopy(family)
    family_key = str(harmonized.source.get("source_id"))
    required = bool(harmonized.ground_truth.get(
        "requires_terminal_harmonization", True))

    source_mm_q = harmonized.terminal_query.text
    source_text_q = _source_text_q(harmonized)

    canonical = harmonizer.harmonize(family_key, source_mm_q, source_text_q)
    if canonical is not None and not str(canonical).strip():
        canonical = None
    if canonical is None and required:
        raise TerminalHarmonizationError(
            f"Family {family_key} requires terminal harmonization but "
            f"the '{harmonizer.method}' backend produced no canonical q*"
        )

    block = {
        "required": required,
        "canonical_q": canonical,
        "canonical_sha256": sha256_text(canonical) if canonical else None,
        "source_mm_q": source_mm_q,
        "source_text_q": source_text_q,
        "method": harmonizer.method,
        "validation": harmonizer.validation_label,
        "provenance": harmonizer.provenance(),
    }
    if harmonized.validation is None:
        harmonized.validation = {}
    harmonized.validation["terminal_harmonization"] = block
    return harmonized


def canonical_terminal(family: CausalFamily) -> tuple[str, str]:
    """Return (canonical_q, canonical_sha256); raise if not established."""
    block = (family.validation or {}).get("terminal_harmonization")
    if not isinstance(block, dict) or not block.get("canonical_q"):
        raise TerminalHarmonizationError(
            f"Family {family.family_id} has no canonical q* — run "
            f"harmonization before variant generation"
        )
    return block["canonical_q"], block["canonical_sha256"]
