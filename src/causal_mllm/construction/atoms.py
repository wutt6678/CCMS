"""Family-level comparative semantic atom extraction (Iteration 4).

Architectural decision: atoms are extracted at the FAMILY level, not
independently from each record. A Type-B family is decomposed
comparatively — H_safe vs H_unsafe with shared q* — so the extractor
identifies which semantic content differs CAUSALLY between the
histories, rather than merely summarizing each trajectory separately.

Guarantees added in the Iteration-4 review:

  1. FOUR-CONDITION SURFACE FORMS — every turn atom records its content
     in all four conditions (multimodal_safe, multimodal_unsafe,
     unimodal_safe, unimodal_unsafe) as {text, images}. The MTMCS
     multimodal and unimodal dialogues are SEPARATELY WRITTEN source
     fields; recording all four forms makes their (non-)equivalence
     explicit instead of assuming interchangeability.
  2. CROSS-MODALITY TERMINAL ALIGNMENT — ``terminal_alignment`` reports
     whether q*_mm equals q*_text (the factorial experiment needs one q*
     across neutral / text_only / vision_only / cross_modal), and
     ``requires_terminal_harmonization`` flags families that need
     rewriting before Iteration 5.
  3. STRUCTURE vs MEANING — atoms carry ``structural_role`` (observable
     fact) and ``semantic_type`` (meaning, 'unknown' until annotated by
     metadata / LLM / human). Semantic type is NEVER inferred from turn
     position: an opening divergence may encode intent, relation,
     constraint, reference, attribute/state, scene framing, or a mix.
  4. EXPLICIT MEDIA REFERENCES — vision atoms carry ``source_media``
     ({path, sha256}) so downstream stages never infer which image an
     atom refers to.

Rule-based and deterministic (no LLM). Integrity is enforced loudly:
divergent-turn sets must agree across modality pairs, image paths must
match across conditions, and turns must align 1:1 by turn_index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from causal_mllm.data.schemas import (
    MTMCS_CONDITIONS,
    AtomType,
    CanonicalSourceExample,
    Message,
    SemanticAtom,
)
from causal_mllm.seeds import sha256_bytes

# Valid divergence states for an atom
DIVERGENCE_SHARED = "shared"
DIVERGENCE_CAUSAL = "causal"
DIVERGENCE_NOT_APPLICABLE = "not_applicable"
ALL_DIVERGENCES = frozenset({
    DIVERGENCE_SHARED, DIVERGENCE_CAUSAL, DIVERGENCE_NOT_APPLICABLE,
})

# Condition key helper: ("multimodal", "safe") -> "multimodal_safe"
_CONDITION_KEY = {
    (modality, safety): f"{modality}_{safety}"
    for modality in ("multimodal", "unimodal")
    for safety in ("safe", "unsafe")
}


class AtomExtractionError(ValueError):
    """Raised when a family's records cannot be decomposed consistently."""


@dataclass
class AtomExtraction:
    """Result of family-level comparative atom extraction."""
    atoms: list[SemanticAtom]
    divergent_turns: list[int]
    shared_terminal_query: bool
    terminal_alignment: dict = field(default_factory=dict)
    backend: str = "rule"
    meta: dict = field(default_factory=dict)

    @property
    def requires_terminal_harmonization(self) -> bool:
        """True when q*_mm differs from q*_text and the family cannot be
        used in the 2x2 factorial design without rewriting the query."""
        return not self.terminal_alignment.get("multimodal_vs_unimodal", True)

    @property
    def causal_atoms(self) -> list[SemanticAtom]:
        return [a for a in self.atoms if a.divergence == DIVERGENCE_CAUSAL]


def _norm(text: Optional[str]) -> str:
    """Whitespace normalization consistent with seeds.sha256_text."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _media_ref(path: str) -> dict:
    """Explicit media reference with content hash when the file exists.

    Missing files get sha256=None with a logged warning — real MTMCS
    media always exists (the adapter saves and verifies it), so a None
    hash in production artifacts is a red flag, not a normal state.
    """
    ref = {"path": path, "sha256": None}
    p = Path(path)
    if p.is_file():
        ref["sha256"] = sha256_bytes(p.read_bytes())
    else:
        from causal_mllm.data.logging import get_logger
        get_logger("causal_mllm.construction.atoms").warning(
            "Media file not found for atom reference: %s", path,
        )
    return ref


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


def _messages_by_turn(record: CanonicalSourceExample) -> dict[int, Message]:
    return {m.turn_index: m for m in record.messages}


def _aligned_messages(safe: CanonicalSourceExample,
                      unsafe: CanonicalSourceExample) -> list[tuple[Message, Message]]:
    """Pair messages 1:1 by turn_index; fail loudly on misalignment."""
    safe_by_turn = _messages_by_turn(safe)
    unsafe_by_turn = _messages_by_turn(unsafe)
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


def _surface_forms(turn_msgs: dict[str, Message]) -> dict[str, dict]:
    """All four condition forms for one turn: {condition: {text, images}}."""
    forms = {}
    for cond_key in MTMCS_CONDITIONS:
        msg = turn_msgs[cond_key]
        forms[cond_key] = {
            "text": msg.text,
            "images": list(msg.images),
        }
    return forms


def _extract_mtmcs(family_id: str,
                   records: list[CanonicalSourceExample]) -> AtomExtraction:
    """Comparative decomposition of one MTMCS family (4 conditions)."""
    conditions = {
        _CONDITION_KEY[(modality, safety)]: _condition(records, modality, safety)
        for modality in ("multimodal", "unimodal")
        for safety in ("safe", "unsafe")
    }
    mm_safe = conditions["multimodal_safe"]
    mm_unsafe = conditions["multimodal_unsafe"]
    text_safe = conditions["unimodal_safe"]
    text_unsafe = conditions["unimodal_unsafe"]

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

    # ---- Cross-modality terminal-query alignment (P0 diagnostic) ----
    # The factorial experiment needs ONE q* across neutral / text_only /
    # vision_only / cross_modal, so q*_mm must equal q*_text.
    terminal_alignment = {
        "mm_safe_vs_mm_unsafe":
            _norm(mm_safe.terminal_query) == _norm(mm_unsafe.terminal_query),
        "text_safe_vs_text_unsafe":
            _norm(text_safe.terminal_query) == _norm(text_unsafe.terminal_query),
        "multimodal_vs_unimodal":
            _norm(mm_safe.terminal_query) == _norm(text_safe.terminal_query),
    }

    setting = mm_safe.source_setting
    shared_terminal = terminal_alignment["mm_safe_vs_mm_unsafe"]
    if setting == "type_b" and not shared_terminal:
        raise AtomExtractionError(
            f"type_b family {family_id} lost its shared terminal query "
            f"during extraction"
        )

    # ---- Align all four conditions by turn ----
    by_turn = {key: _messages_by_turn(rec)
               for key, rec in conditions.items()}
    turn_sets = {key: set(mapping) for key, mapping in by_turn.items()}
    reference_turns = turn_sets["multimodal_safe"]
    if any(ts != reference_turns for ts in turn_sets.values()):
        raise AtomExtractionError(
            f"Turn sets differ across conditions: {turn_sets}"
        )

    terminal_turn = mm_safe.terminal_turn_index
    atoms: list[SemanticAtom] = []

    for turn in sorted(reference_turns):
        turn_msgs = {key: by_turn[key][turn] for key in MTMCS_CONDITIONS}
        ms = turn_msgs["multimodal_safe"]
        mu = turn_msgs["multimodal_unsafe"]
        surface_forms = _surface_forms(turn_msgs)

        # ---- Vision atom: shared image with explicit media references ----
        if ms.images or mu.images:
            if list(ms.images) != list(mu.images):
                raise AtomExtractionError(
                    f"turn {turn}: image paths differ across safe/unsafe "
                    f"conditions ({ms.images} vs {mu.images})"
                )
            # Structural fact: an image is present and shared. Whether it
            # is entity/scene content (semantic_type) and whether it is
            # RISK-RELEVANT (I -> Risk) are annotation questions, never
            # extraction conclusions.
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:vision",
                description=(
                    f"shared visual content at turn {turn} "
                    f"({len(ms.images)} image(s))"
                ),
                source_turns=[turn],
                source_modalities=["vision"],
                atom_type=AtomType.UNKNOWN.value,  # alias of semantic_type
                divergence=DIVERGENCE_SHARED,
                structural_role="shared_image",
                semantic_type="unknown",
                semantic_validation="pending",
                risk_relevance="pending",
                required_for_joint_interpretation=None,
                surface_forms=surface_forms,
                source_media=[_media_ref(p) for p in ms.images],
            ))

        # ---- Text atom: shared or causally divergent ----
        if turn in div_mm:
            # Structural fact: this turn differs between H_safe/H_unsafe.
            # Semantic meaning (intent? relation? constraint? framing?)
            # is NOT inferred here — it stays unknown until annotated.
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:causal",
                description=(
                    f"causal divergence at turn {turn}: safe and unsafe "
                    f"histories carry different user content here"
                ),
                source_turns=[turn],
                source_modalities=["text"],
                atom_type=AtomType.UNKNOWN.value,
                divergence=DIVERGENCE_CAUSAL,
                structural_role="divergent_history_turn",
                semantic_type="unknown",
                semantic_description=None,
                semantic_validation="pending",
                safe_text=ms.text,
                unsafe_text=mu.text,
                surface_forms=surface_forms,
            ))
        elif _norm(ms.text):
            if turn == terminal_turn:
                role = "terminal_query"
                desc = f"shared terminal query q* at turn {turn}"
            else:
                role = "shared_history_turn"
                desc = f"shared history content at turn {turn}"
            # Structural role only; semantic type stays unknown (the alias
            # atom_type mirrors it) until annotation.
            atoms.append(SemanticAtom(
                atom_id=f"{family_id}:t{turn}:text",
                description=desc,
                source_turns=[turn],
                source_modalities=["text"],
                atom_type=AtomType.UNKNOWN.value,
                divergence=DIVERGENCE_SHARED,
                structural_role=role,
                semantic_type="unknown",
                semantic_validation="pending",
                surface_forms=surface_forms,
            ))

    return AtomExtraction(
        atoms=atoms,
        divergent_turns=sorted(div_mm),
        shared_terminal_query=shared_terminal,
        terminal_alignment=terminal_alignment,
        meta={
            "setting": setting,
            "reference_source_id": mm_safe.source_id,
            "n_conditions": len(records),
            "requires_terminal_harmonization":
                not terminal_alignment["multimodal_vs_unimodal"],
        },
    )


def _extract_singleton(family_id: str,
                       record: CanonicalSourceExample) -> AtomExtraction:
    """Structural decomposition for records without a safe/unsafe pair.

    No comparative signal exists here; atoms are marked not_applicable
    for divergence so downstream stages never mistake them for causal.
    Semantic types stay unknown — role labels are structural only.
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
                atom_type=AtomType.UNKNOWN.value,
                divergence=DIVERGENCE_NOT_APPLICABLE,
                structural_role="shared_image",
                semantic_type="unknown",
                semantic_validation="pending",
                risk_relevance="pending",
                source_media=[_media_ref(p) for p in msg.images],
            ))
        if not _norm(msg.text):
            continue
        if turn == terminal_turn:
            role, desc = "terminal_query", f"terminal query at turn {turn}"
        elif msg.role == "assistant":
            role, desc = "assistant_context", f"assistant context at turn {turn}"
        else:
            role, desc = "shared_history_turn", f"user context at turn {turn}"
        atoms.append(SemanticAtom(
            atom_id=f"{family_id}:t{turn}:text",
            description=desc,
            source_turns=[turn],
            source_modalities=["text"],
            atom_type=AtomType.UNKNOWN.value,
            divergence=DIVERGENCE_NOT_APPLICABLE,
            structural_role=role,
            semantic_type="unknown",
            semantic_validation="pending",
        ))

    return AtomExtraction(
        atoms=atoms,
        divergent_turns=[],
        shared_terminal_query=False,
        terminal_alignment={},
        meta={"setting": record.source_setting,
              "reference_source_id": record.source_id,
              "n_conditions": 1,
              "requires_terminal_harmonization": False},
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
