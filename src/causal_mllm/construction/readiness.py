"""Variant readiness levels and hard prerequisites (Iteration 5).

Three explicit readiness levels make variant construction debuggable:

    L0_STRUCTURAL   family extracted correctly (valid skeleton)
    L1_SEMANTIC     the annotations that variant generators ACTUALLY use
                    are resolved (not every atom in every trajectory)
    L2_VARIANT_READY  canonical q* established via harmonization

Variant generation NEVER silently continues on unresolved evidence:
``assert_variant_ready`` raises ``VariantPrerequisiteError`` listing
exactly why a family cannot yet produce a given condition.

Prerequisite strength differs by variant — ``history_reset`` is mostly
structural, ``cross_modal`` requires substantially stronger evidence:

    neutral / history_reset / shuffle : L2 only
    text_only    : L2 + resolved cross-modal equivalence (uses text forms)
    vision_only  : L2 + resolved visual risk relevance
    cross_modal  : L2 + resolved equivalence + resolved risk relevance

A family is a cross_modal_CANDIDATE once constructible; whether it
belongs to the strict cross-modal causal subset (Risk(T)<θ, Risk(V)<θ,
Risk(T,V)>=θ) requires BEHAVIORAL evidence and is decided in
Iteration 6+, never by annotator belief.
"""

from __future__ import annotations

from causal_mllm.data.schemas import CausalFamily, SemanticAtom
from causal_mllm.data.validate_schema import validate_family_skeleton

L0_STRUCTURAL = "L0_structural"
L1_SEMANTIC = "L1_semantic"
L2_VARIANT_READY = "L2_variant_ready"

ALL_VARIANT_NAMES = (
    "neutral", "text_only", "vision_only",
    "cross_modal", "shuffle", "history_reset",
)

# Resolved decisions (not 'pending' and not 'uncertain')
_RESOLVED_EQUIVALENCE = {"equivalent", "not_equivalent"}
_RESOLVED_RISK = {"relevant", "irrelevant"}


class VariantPrerequisiteError(RuntimeError):
    """Raised when a variant cannot be built from the current evidence."""

    def __init__(self, family_id: str, variant: str, reasons: list[str]):
        self.family_id = family_id
        self.variant = variant
        self.reasons = reasons
        super().__init__(
            f"Family {family_id} is not ready for variant '{variant}':\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )


def _equivalence_state(atom: SemanticAtom, axis: str) -> str:
    value = atom.semantic_equivalence.get(axis, "pending")
    if isinstance(value, dict):
        return value.get("state", "pending")
    return value


def structural_gaps(family: CausalFamily) -> list[str]:
    """L0: is the family a valid extracted skeleton?"""
    return validate_family_skeleton(family.to_dict())


def semantic_gaps(family: CausalFamily) -> list[str]:
    """L1: minimum annotations the variant generators actually use.

    Deliberately NOT exhaustive: only what interventions depend on.
      * causal atoms: semantic_type + validation resolved
      * shared image atoms: risk_relevance decision +
        required_for_joint_interpretation evidenced
      * atoms with mm+text surface forms: multimodal_vs_unimodal decided
    """
    gaps: list[str] = []
    for atom in family.semantic_atoms:
        if atom.divergence == "causal":
            if atom.semantic_type == "unknown":
                gaps.append(
                    f"{atom.atom_id}: causal atom semantic_type is unknown"
                )
            if atom.semantic_validation == "pending":
                gaps.append(
                    f"{atom.atom_id}: causal atom semantic_validation "
                    f"is pending"
                )
        if atom.structural_role == "shared_image":
            if atom.risk_relevance not in _RESOLVED_RISK:
                gaps.append(
                    f"{atom.atom_id}: risk_relevance is "
                    f"'{atom.risk_relevance}' (need relevant/irrelevant)"
                )
            if atom.required_for_joint_interpretation is None:
                gaps.append(
                    f"{atom.atom_id}: required_for_joint_interpretation "
                    f"is null"
                )
        forms = atom.surface_forms or {}
        has_mm = any(k.startswith("multimodal_") for k in forms)
        has_text = any(k.startswith("unimodal_") for k in forms)
        if has_mm and has_text:
            state = _equivalence_state(atom, "multimodal_vs_unimodal")
            if state not in _RESOLVED_EQUIVALENCE:
                gaps.append(
                    f"{atom.atom_id}: multimodal_vs_unimodal equivalence "
                    f"is '{state}'"
                )
    return gaps


def harmonization_gaps(family: CausalFamily) -> list[str]:
    """L2: is a canonical q* established for this family?"""
    block = (family.validation or {}).get("terminal_harmonization")
    if not isinstance(block, dict):
        return ["terminal_harmonization block missing"]
    gaps = []
    if not block.get("canonical_q"):
        gaps.append("canonical_q missing")
    if not block.get("canonical_sha256"):
        gaps.append("canonical_sha256 missing")
    return gaps


def family_readiness(family: CausalFamily) -> dict:
    """Report achieved readiness level with all outstanding reasons."""
    l0 = structural_gaps(family)
    l1 = l0 or semantic_gaps(family)
    l2 = l1 or harmonization_gaps(family)
    if not l2:
        level = L2_VARIANT_READY
    elif not l1:
        level = L1_SEMANTIC
    elif not l0:
        level = L0_STRUCTURAL
    else:
        level = None
    return {
        "level": level,
        L0_STRUCTURAL: l0,
        L1_SEMANTIC: semantic_gaps(family) if not l0 else l0,
        L2_VARIANT_READY: harmonization_gaps(family) if not l1 else l1,
    }


# Per-variant semantic evidence requirements beyond L2
_VARIANT_REQUIREMENTS = {
    "neutral": (),
    "history_reset": (),
    "shuffle": (),
    "text_only": ("equivalence",),
    "vision_only": ("risk_relevance",),
    "cross_modal": ("equivalence", "risk_relevance"),
}


def _variant_semantic_reasons(family: CausalFamily,
                              requirements: tuple) -> list[str]:
    reasons: list[str] = []
    if "equivalence" in requirements:
        for atom in family.semantic_atoms:
            forms = atom.surface_forms or {}
            has_mm = any(k.startswith("multimodal_") for k in forms)
            has_text = any(k.startswith("unimodal_") for k in forms)
            if has_mm and has_text:
                state = _equivalence_state(atom, "multimodal_vs_unimodal")
                if state not in _RESOLVED_EQUIVALENCE:
                    reasons.append(
                        f"cross-modal equivalence unresolved on "
                        f"{atom.atom_id} ('{state}') — modality and "
                        f"wording would be confounded"
                    )
    if "risk_relevance" in requirements:
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                if atom.risk_relevance not in _RESOLVED_RISK:
                    reasons.append(
                        f"visual atom {atom.atom_id} risk_relevance is "
                        f"'{atom.risk_relevance}' — image presence must "
                        f"not be confused with risk relevance"
                    )
                if atom.required_for_joint_interpretation is None:
                    reasons.append(
                        f"visual atom {atom.atom_id} "
                        f"required_for_joint_interpretation is null"
                    )
    return reasons


def assert_variant_ready(family: CausalFamily, variant: str) -> None:
    """Hard precondition gate for one variant. Fails loudly with reasons.

    Gates are PER-VARIANT: structural conditions (neutral, history_reset,
    shuffle) need only L0 + a canonical q*, while text_only / vision_only /
    cross_modal additionally require the specific semantic evidence they
    consume. The full L1 gap list remains available via family_readiness
    for reporting, but a pending annotation that no generator uses never
    blocks construction.

    Raises:
        ValueError: On unknown variant name.
        VariantPrerequisiteError: If any prerequisite is unresolved.
    """
    if variant not in ALL_VARIANT_NAMES:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected one of {ALL_VARIANT_NAMES}"
        )

    reasons: list[str] = []
    reasons.extend(f"L0: {r}" for r in structural_gaps(family))
    if not reasons:
        reasons.extend(f"L2: {r}" for r in harmonization_gaps(family))
    if not reasons:
        reasons.extend(
            _variant_semantic_reasons(
                family, _VARIANT_REQUIREMENTS[variant])
        )

    if reasons:
        raise VariantPrerequisiteError(family.family_id, variant, reasons)
