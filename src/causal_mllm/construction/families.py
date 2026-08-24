"""Family skeleton construction (Iteration 4).

A family skeleton binds one selection unit to:
  * a deterministic family_id,
  * its comparatively extracted semantic atoms (see atoms.py),
  * the invariant terminal query q* (with sha256),
  * ground-truth provenance (condition labels, divergent turns),
  * review placeholders for standalone terminal-risk validation.

Skeletons carry an EMPTY ``variants`` dict — the six controlled variants
are generated in Iteration 5. Source records remain untouched
(pass-through): the skeleton references them, it does not modify them.
"""

from __future__ import annotations

from causal_mllm.construction.atoms import (
    AtomExtraction,
    extract_family_atoms,
)
from causal_mllm.data.schemas import (
    CanonicalSourceExample,
    CausalFamily,
    TerminalQuery,
)
from causal_mllm.seeds import deterministic_family_id


def _family_key(records: list[CanonicalSourceExample]) -> str:
    ref = records[0]
    pair_id = ref.metadata.get("pair_id")
    if ref.source_dataset == "mtmcs" and pair_id:
        return str(pair_id)
    return ref.source_id


def _reference_record(records: list[CanonicalSourceExample]) -> CanonicalSourceExample:
    """Reference condition for terminal query and provenance.

    For MTMCS families the multimodal safe record carries q* (shared
    across safe/unsafe for type_b; the safe variant's terminal is the
    reference for type_a).
    """
    for r in records:
        if (r.metadata.get("modality") == "multimodal"
                and r.metadata.get("safety") == "safe"):
            return r
    return records[0]


def build_family_skeleton(
    records: list[CanonicalSourceExample],
    *,
    seed: int = 42,
) -> CausalFamily:
    """Build one family skeleton from a selection unit's records.

    Args:
        records: All canonical records of one family unit (4 for MTMCS
            groups, 1 for singletons).
        seed: Global experiment seed (family_id determinism).

    Returns:
        A CausalFamily with atoms populated and variants empty.

    Raises:
        AtomExtractionError: On cross-condition inconsistency.
        ValueError: On empty input.
    """
    if not records:
        raise ValueError("Cannot build a family skeleton from no records")

    key = _family_key(records)
    ref = _reference_record(records)
    family_id = deterministic_family_id(ref.source_dataset, key, seed)

    extraction: AtomExtraction = extract_family_atoms(family_id, records)

    labels_by_condition = {
        f"{r.metadata.get('modality')}:{r.metadata.get('safety')}": r.label
        for r in records if r.metadata.get("modality")
    } or {ref.source_id: ref.label}

    condition_source_ids = sorted(r.source_id for r in records)

    return CausalFamily(
        family_id=family_id,
        source={
            "dataset": ref.source_dataset,
            "source_id": key,
            "setting": ref.source_setting,
            "condition_source_ids": condition_source_ids,
        },
        category=ref.source_category,
        setting=ref.source_setting,
        terminal_query=TerminalQuery.create(ref.terminal_query),
        semantic_atoms=extraction.atoms,
        ground_truth={
            "labels_by_condition": labels_by_condition,
            "unsafe_intent": ref.source_category,
            "divergent_turns": extraction.divergent_turns,
            "shared_terminal_query": extraction.shared_terminal_query,
            # Cross-modality q* diagnostics: the factorial design needs one
            # q* across neutral/text_only/vision_only/cross_modal. When
            # multimodal_vs_unimodal is False the family needs rewriting
            # (requires_terminal_harmonization) before Iteration 5.
            "terminal_alignment": extraction.terminal_alignment,
            "requires_terminal_harmonization":
                extraction.requires_terminal_harmonization,
            "causal_atom_ids": [a.atom_id for a in extraction.causal_atoms],
            "extraction_backend": extraction.backend,
        },
        variants={},  # Iteration 5
        validation={
            # Placeholders until Iteration 6 estimates Risk(q*).
            "requires_standalone_risk_validation": True,
            "standalone_terminal_risk": None,
            "strict_causal_candidate": None,
            "atoms_meta": extraction.meta,
        },
    )


def build_family_skeletons(
    units: list[tuple[str, list[CanonicalSourceExample]]],
    *,
    seed: int = 42,
) -> list[CausalFamily]:
    """Build skeletons for all family units; IDs must be unique."""
    skeletons = [build_family_skeleton(records, seed=seed)
                 for _, records in units]
    ids = [s.family_id for s in skeletons]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(
            f"Family ID collision after deterministic hashing: {duplicates}. "
            f"Resolve before dataset construction (Iteration 7)."
        )
    return skeletons
