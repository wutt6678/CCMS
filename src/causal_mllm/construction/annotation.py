"""Semantic annotation of extracted atoms (Iteration 4, P0-1/P0-2).

Extraction identifies STRUCTURE (which turn/modality differs, what is
shared, where the image lives). Iteration 5 needs MEANING to construct
valid interventions: which semantic content is preserved, removed, or
transferred between H_10 / H_11 conditions.

This module provides the annotation scaffolding:

  * ``ManualFileAnnotator`` — JSON file keyed by family_key -> atom_id.
    Intended for the manual checking of the first 10-20 families.
  * ``CallableAnnotator`` — wraps ANY callable (e.g. an LLM/VLM client)
    mapping (family_key, atom_dict) -> annotation dict; recorded with
    semantic_validation='llm'.

Annotation fields established here:

  semantic_type / semantic_description — what the turn contributes
  semantic_equivalence — validated statement that differently worded
    surface forms express the SAME contribution (S(T_mm) ~ S(T_text));
    required before treating them as modality counterfactuals
  risk_relevance — for visual atoms: 'present' vs 'supplies information
    required to interpret the risky trajectory'

Annotation never invents structure; it only fills semantic fields on
already-extracted atoms, keeping ``atom_type`` in sync as the exact
alias of ``semantic_type``.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from causal_mllm.data.schemas import (
    EQUIVALENCE_AXES,
    EQUIVALENCE_STATES,
    RISK_RELEVANCE_STATES,
    SEMANTIC_VALIDATION_STATES,
    AtomType,
    CausalFamily,
    SemanticAtom,
)


class AnnotationError(ValueError):
    """Raised when an annotation payload is invalid."""


def _validate_annotation_payload(payload: dict,
                                 default_validation: str = "llm") -> None:
    """Fail loudly on malformed annotations — no silent bad labels.

    ``default_validation`` is the annotator backend's label, used when
    the payload does not state one (manual payloads default to 'human',
    callable backends to 'llm').
    """
    sem_type = payload.get("semantic_type")
    valid_types = {v.value for v in AtomType}
    if sem_type not in valid_types:
        raise AnnotationError(
            f"semantic_type '{sem_type}' not in {sorted(valid_types)}"
        )
    validation = payload.get("semantic_validation", default_validation)
    if validation not in SEMANTIC_VALIDATION_STATES or validation == "pending":
        raise AnnotationError(
            f"semantic_validation must be a resolved state, got '{validation}'"
        )
    # P0 for real LLM-generated annotations: 'an LLM did it' is not
    # provenance. The exact pipeline must be recoverable later.
    if validation == "llm":
        prov = payload.get("annotation_provenance")
        if not isinstance(prov, dict) or not prov.get("backend") \
                or not prov.get("model"):
            raise AnnotationError(
                "llm-backed annotations require annotation_provenance "
                "with at least backend and model"
            )
    equivalence = payload.get("semantic_equivalence", {})
    for axis, value in equivalence.items():
        if axis not in EQUIVALENCE_AXES:
            raise AnnotationError(f"Unknown equivalence axis '{axis}'")
        state = value.get("state", value) if isinstance(value, dict) else value
        if state not in EQUIVALENCE_STATES:
            raise AnnotationError(
                f"Equivalence state '{state}' invalid for axis '{axis}'"
            )
        if isinstance(value, dict):
            confidence = value.get("confidence")
            if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
                raise AnnotationError(
                    f"confidence {confidence} out of [0, 1] for axis '{axis}'"
                )
    risk = payload.get("risk_relevance")
    if risk is not None and risk not in RISK_RELEVANCE_STATES:
        raise AnnotationError(f"risk_relevance '{risk}' invalid")


class AtomAnnotator(ABC):
    """Backend interface: returns an annotation payload for one atom.

    Return None to leave the atom unannotated (fields stay pending).
    """

    #: semantic_validation value recorded by this backend
    validation_label: str = "pending"

    @abstractmethod
    def annotate_atom(self, family_key: str, atom: SemanticAtom) -> Optional[dict]:
        ...

    def annotate_family(self, family: CausalFamily) -> CausalFamily:
        """Return an ANNOTATED COPY; the input skeleton is never mutated."""
        annotated = copy.deepcopy(family)
        family_key = str(annotated.source.get("source_id"))
        for atom in annotated.semantic_atoms:
            payload = self.annotate_atom(family_key, atom)
            if payload is None:
                continue
            _validate_annotation_payload(payload, self.validation_label)
            atom.set_semantic_annotation(
                semantic_type=payload["semantic_type"],
                semantic_description=payload.get("semantic_description"),
                semantic_validation=payload.get(
                    "semantic_validation", self.validation_label),
                semantic_equivalence=_normalized_equivalence(
                    payload.get("semantic_equivalence")),
                risk_relevance=payload.get("risk_relevance"),
                required_for_joint_interpretation=payload.get(
                    "required_for_joint_interpretation"),
                annotation_provenance=payload.get("annotation_provenance",
                                                  self.default_provenance()),
            )
        return annotated

    def default_provenance(self) -> dict:
        """Provenance recorded when the payload does not supply one."""
        return {"backend": self.validation_label}


def _normalized_equivalence(raw: Optional[dict]) -> Optional[dict]:
    """Fill missing axes with 'pending'; normalize bare states to dicts."""
    if raw is None:
        return None
    normalized: dict = {}
    for axis in EQUIVALENCE_AXES:
        value = raw.get(axis, "pending")
        if isinstance(value, dict):
            normalized[axis] = value
        else:
            normalized[axis] = {"state": value}
    return normalized


class ManualFileAnnotator(AtomAnnotator):
    """Annotations from a JSON file: {family_key: {atom_id: payload}}.

    The intended workflow for the first 10-20 families: review skeleton
    atoms, write payloads by hand, then apply. Payloads use the same
    keys as ``set_semantic_annotation``.
    """

    validation_label = "human"

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self._path}")
        with self._path.open(encoding="utf-8") as f:
            self._data: dict = json.load(f)

    def annotate_atom(self, family_key: str, atom: SemanticAtom) -> Optional[dict]:
        return self._data.get(family_key, {}).get(atom.atom_id)


class CallableAnnotator(AtomAnnotator):
    """Wraps any callable as an LLM/VLM-assisted annotator.

    The callable receives (family_key, atom.to_dict()) and must return
    an annotation payload dict (or None). This keeps the pipeline
    model-agnostic: any client — local VLM, hosted API — can be plugged
    in without changing the construction code.

    Provenance is mandatory for llm annotations: model identity,
    revision, prompt version, temperature, and seed are recorded on
    every annotated atom so the producing pipeline is recoverable.
    """

    validation_label = "llm"

    def __init__(self, fn: Callable[[str, dict], Optional[dict]],
                 *, model_name: str, model_revision: Optional[str] = None,
                 prompt_version: Optional[str] = None,
                 temperature: float = 0.0, seed: Optional[int] = None):
        if not model_name:
            raise AnnotationError(
                "CallableAnnotator requires model_name for provenance"
            )
        self._fn = fn
        self.model_name = model_name
        self._provenance = {
            "backend": "llm",
            "model": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "seed": seed,
        }

    def default_provenance(self) -> dict:
        return dict(self._provenance)

    def annotate_atom(self, family_key: str, atom: SemanticAtom) -> Optional[dict]:
        payload = self._fn(family_key, atom.to_dict())
        if payload is not None and "annotation_provenance" not in payload:
            payload = {**payload, "annotation_provenance": self.default_provenance()}
        return payload


def apply_annotations(
    family: CausalFamily,
    annotator: AtomAnnotator,
) -> CausalFamily:
    """Convenience wrapper: annotated copy of one family skeleton."""
    return annotator.annotate_family(family)
