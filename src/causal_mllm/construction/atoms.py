"""Family-level comparative semantic atom extraction (Iteration 4).

Architectural decision: atoms are extracted at the FAMILY level, not
independently from each record. A Type-B family is decomposed
comparatively — H_safe vs H_unsafe with shared q* — so the extractor
identifies which semantic content differs CAUSALLY between the
histories, rather than merely summarizing each trajectory separately.

Rule-based and deterministic (no LLM):

  * turns whose text is identical across safe/unsafe  -> shared atoms
  * turns whose text differs                          -> causal atoms
    (carry both surface forms: safe_text / unsafe_text)
  * shared image turns                                -> vision atoms
  * the shared terminal turn                          -> intent atom (q*)

Integrity is enforced loudly:

  * divergent turn sets must agree between the multimodal and the
    text-only pair (both are built from the same source fields);
  * image paths must be identical across conditions;
  * message alignment must match 1:1 by turn_index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from causal_mllm.data.schemas import (
    AtomType,
    CanonicalSourceExample,
    Message,
    SemanticAtom,
)

# Valid divergence states for an atom
DIVERGENCE_SHARED = "shared"
DIVERGENCE_CAUSAL = "causal"
DIVERGENCE_NOT_APPLICABLE = "not_applicable"
ALL_DIVERGENCES = frozenset({
    DIVERGENCE_SHARED, DIVERGENCE_CAUSAL, DIVERGENCE_NOT_APPLICABLE,
})


class AtomExtractionError(ValueError):
    """Raised when a family's records cannot be decomposed consistently."""


@dataclass
class AtomExtraction:
    """Result of family-level comparative atom extraction."""
    atoms: list[SemanticAtom]
    divergent_turns: list[int]
    shared_terminal_query: bool
    backend: str = "rule"
    meta: dict = field(default_factory=dict)

    @property
    def causal_atoms(self) -> list[SemanticAtom]:
        return [a for a in self.atoms if a.divergence == DIVERGENCE_CAUSAL]


def _norm(text: Optional[str]) -> str:
    """Whitespace normalization consistent with seeds.sha256_text."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _condition(records: list[CanonicalSourceExample],
               modality: str, safety: str) -> CanonicalSourceExample:
    for r in records:
        if (r.metadata.get("modality") == modality
                and r.metadata.get("safety") == safety):
            return r
    raise AtomExtractionError(
        f"Family is missing condition {modality}:{safety}; "
        f"cannot run comparative extraction"
    )


def _aligned_messages(safe: CanonicalSourceExample,
                      unsafe: CanonicalSourceExample) -> list[tuple[Message, Message]]:
    """Pair messages 1:1 by turn_index; fail loudly on misalignment."""
    safe_by_turn = {m.turn_index: m for m in safe.messages}
    unsafe_by_turn = {m.turn_index: m for m in unsafe.messages}
    if set(safe_by_turn) != set(unsafe_by_turn):
        raise AtomExtractionError(
            f"Turn misalignment between {safe.source_id} and "
            f"{unsafe.source_id}: {sorted(safe_by_turn)} vs "
            f"{sorted(unsafe_by_turn)}"
        )
    return [(safe_by_turn[t], unsafe_by_turn[t]) for t in sorted(safe_by_turn)]


def _divergent_turns(safe: CanonicalSourceExample,
                     unsafe: CanonicalSourceExample) -> set[int]:
    return {
        ms.turn_index
        for ms, mu in _aligned_messages(safe, unsafe)
        if _norm(ms.text) != _norm(mu.text)
    }


def _modality_labels(msg: Message) -> list[str]:
    modalities = []
    if msg.images:
        modalities.append("vision")
    if _norm(msg.text):
        modalities.append("text")
    return modalities or ["text"]


def _extract_mtmcs(family_id: str,
                   records: list[CanonicalSourceExample]) -> AtomExtraction:
    """Comparative decomposition of one MTMCS family (4 conditions)."""
    mm_safe = _condition(records, "multimodal", "safe")
    mm_unsafe = _condition(records, "multimodal", "unsafe")
    text_safe = _condition(records, "unimodal", "safe")
    text_unsafe = _condition(records, "unimodal", "unsafe")

    # ---- Integrity: divergence must agree across modality pairs ----
    div_mm = _divergent_turns(mm_safe, mm_unsafe)
    div_text = _divergent_turns(text_safe, text_unsafe)
    if div_mm != div_text:
        raise AtomExtractionError(
            f"Divergent turn sets disagree between multimodal and text "
            f"pairs ({sorted(div_mm)} vs {sorted(div_text)}). The adapter "
            f"builds both from the same source fields — this indicates "
            f"data corruption."
        )

    # ---- Integrity: shared terminal for type_b ----
    setting = mm_safe.source_setting
    shared_terminal = _norm(mm_safe.terminal_query) == _norm(mm_unsafe.terminal_query)
    if setting == "type_b" and not shared_terminal:
        raise AtomExtractionError(
            f"type_b family {family_id} lost its shared terminal query "
            f"during extraction"
        )

    terminal_turn = mm_safe.terminal_turn_index
    atoms: list[SemanticAtom] = []

    for ms, mu in _aligned_messages(mm_safe, mm_unsafe):
        turn = ms.turn_index

        # Image consistency across conditions (same image by construction)
        if ms.images or mu.images:
            if list(ms.images) != list(mu.images):
                raise AtomExtractionError(
                    f"turn {turn}: image paths differ across safe/unsafe "
                    f"conditions ({ms.images} vs {mu.images})"
                )
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:vision",
                description=(
                    f"shared visual content at turn {turn} "
                    f"({len(ms.images)} image(s))"
                ),
                source_turns=[turn],
                source_modalities=["vision"],
                atom_type=AtomType.ENTITY_OR_SCENE.value,
                divergence=DIVERGENCE_SHARED,
            ))

        # Text content: shared or causally divergent
        if turn in div_mm:
            atom_type = (AtomType.INTENT.value if turn == terminal_turn
                         else AtomType.ATTRIBUTE_OR_STATE.value)
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:causal",
                description=(
                    f"causal divergence at turn {turn}: safe and unsafe "
                    f"histories carry different user content here"
                ),
                source_turns=[turn],
                source_modalities=["text"],
                atom_type=atom_type,
                divergence=DIVERGENCE_CAUSAL,
                safe_text=ms.text,
                unsafe_text=mu.text,
            ))
        elif _norm(ms.text):
            if turn == terminal_turn:
                atom_type = AtomType.INTENT.value
                desc = f"shared terminal query q* at turn {turn}"
            else:
                atom_type = AtomType.CONTEXTUAL_DISAMBIGUATOR.value
                desc = f"shared context at turn {turn}"
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:text",
                description=desc,
                source_turns=[turn],
                source_modalities=["text"],
                atom_type=atom_type,
                divergence=DIVERGENCE_SHARED,
            ))

    return AtomExtraction(
        atoms=atoms,
        divergent_turns=sorted(div_mm),
        shared_terminal_query=shared_terminal,
        meta={
            "setting": setting,
            "reference_source_id": mm_safe.source_id,
            "n_conditions": len(records),
        },
    )


def _extract_singleton(family_id: str,
                       record: CanonicalSourceExample) -> AtomExtraction:
    """Structural decomposition for records without a safe/unsafe pair.

    No comparative signal exists here; atoms are marked not_applicable
    for divergence so downstream stages never mistake them for causal.
    """
    atoms: list[SemanticAtom] = []
    terminal_turn = record.terminal_turn_index

    for msg in record.messages:
        turn = msg.turn_index
        if msg.images:
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:vision",
                description=f"visual content at turn {turn}",
                source_turns=[turn],
                source_modalities=["vision"],
                atom_type=AtomType.ENTITY_OR_SCENE.value,
                divergence=DIVERGENCE_NOT_APPLICABLE,
            ))
        if not _norm(msg.text):
            continue
        if turn == terminal_turn:
            atom_type, desc = AtomType.INTENT.value, f"terminal query at turn {turn}"
        elif msg.role == "assistant":
            atom_type, desc = AtomType.REFERENCE.value, f"assistant context at turn {turn}"
        else:
            atom_type, desc = (AtomType.CONTEXTUAL_DISAMBIGUATOR.value,
                               f"user context at turn {turn}")
        atoms.append(SemanticAtom(
            atom_id=f"{family_id}:t{turn}:text",
            description=desc,
            source_turns=[turn],
            source_modalities=["text"],
            atom_type=atom_type,
            divergence=DIVERGENCE_NOT_APPLICABLE,
        ))

    return AtomExtraction(
        atoms=atoms,
        divergent_turns=[],
        shared_terminal_query=False,
        meta={"setting": record.source_setting,
              "reference_source_id": record.source_id,
              "n_conditions": 1},
    )


def extract_family_atoms(
    family_id: str,
    records: list[CanonicalSourceExample],
) -> AtomExtraction:
    """Extract semantic atoms for one family unit.

    MTMCS groups (4 conditions) are decomposed comparatively
    (H_safe vs H_unsafe). Singletons are decomposed structurally.

    Args:
        family_id: Deterministic family ID (used for atom IDs).
        records: The family's canonical records (one group or singleton).

    Raises:
        AtomExtractionError: On any cross-condition inconsistency.
    """
    if not records:
        raise AtomExtractionError("Cannot extract atoms from an empty family")
    if records[0].source_dataset == "mtmcs" and len(records) > 1:
        return _extract_mtmcs(family_id, records)
    if len(records) != 1:
        raise AtomExtractionError(
            f"Non-MTMCS family must be a singleton, got {len(records)} records"
        )
    return _extract_singleton(family_id, records[0])
