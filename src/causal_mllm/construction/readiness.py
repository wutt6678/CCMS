"""Variant readiness levels and hard prerequisites (Iteration 5).

Three explicit readiness levels make variant construction debuggable:

    L0_STRUCTURAL   family extracted correctly (valid skeleton)
    L1_SEMANTIC     the annotations that variant generators ACTUALLY use
                    are resolved (not every atom in every trajectory)
    L2_VARIANT_READY  canonical q* established via harmonization

Variant generation NEVER silently continues on unresolved evidence:
``assert_variant_ready`` raises ``VariantPrerequisiteError`` listing
exactly why a family cannot yet produce a given condition.

Two DIFFERENT notions of a semantic judgment are kept apart:

  * ANNOTATED (L1 completeness): a decision was made. ``not_equivalent``
    and ``irrelevant`` are complete annotations — family_readiness()
    legitimately reports L1 for them.
  * FACTORIAL ELIGIBILITY: the decision must be the RIGHT one for the
    comparison. ``S(T_mm) !~ S(T_text)`` cannot be constructed as a
    modality counterfactual, and an ``irrelevant`` image cannot ground
    a causal visual condition. Eligibility therefore requires
    ``equivalent`` / ``relevant`` / ``required_for_joint_interpretation
    == True`` — a decided-but-negative annotation REJECTS the family
    from the causal subset (it belongs with the negative controls).

Per-variant prerequisites (beyond L0 + canonical q*):

    neutral       : none
    text_only     : none (both forms are unimodal; no modality crossing)
    vision_only   : equivalence + visual relevance == relevant
    cross_modal   : equivalence + visual relevance == relevant +
                    required_for_joint_interpretation == True
    shuffle       : cross_modal-ready (same content, permuted order)
    history_reset : none

vision_only needs the equivalence gate because the H00->H01 contrast
(unimodal_safe -> multimodal_safe + image) changes BOTH image presence
and text wording unless the two text histories are established as
semantically equivalent.

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

# ANNOTATED (L1 completeness): a DECISION exists, whichever way it went.
# not_equivalent / irrelevant are complete annotations — the family is
# simply ineligible for the causal subset (negative control material).
ANNOTATED_EQUIVALENCE = {"equivalent", "not_equivalent"}
ANNOTATED_RISK_RELEVANCE = {"relevant", "irrelevant"}

# FACTORIAL ELIGIBILITY: the decision must support the comparison.
# Constructing a modality counterfactual from an explicitly
# not_equivalent annotation would defeat the whole point of annotating.
FACTORIAL_EQUIVALENCE = {"equivalent"}
FACTORIAL_RISK_RELEVANCE = {"relevant"}


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
            if atom.risk_relevance not in ANNOTATED_RISK_RELEVANCE:
                gaps.append(
                    f"{atom.atom_id}: risk_relevance is "
                    f"'{atom.risk_relevance}' (need a decision: "
                    f"relevant/irrelevant)"
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
            if state not in ANNOTATED_EQUIVALENCE:
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


# Per-variant factorial eligibility requirements beyond L0 + canonical q*.
# text_only is NOT gated by equivalence: both neutral and text_only draw
# unimodal text forms, so the contrast never crosses modalities.
_VARIANT_REQUIREMENTS = {
    "neutral": (),
    "text_only": (),
    "history_reset": (),
    "vision_only": ("equivalence", "visual_relevance"),
    "cross_modal": ("equivalence", "visual_relevance",
                    "joint_interpretation"),
    "shuffle": ("equivalence", "visual_relevance", "joint_interpretation"),
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
                if state not in FACTORIAL_EQUIVALENCE:
                    if state in ANNOTATED_EQUIVALENCE:
                        reasons.append(
                            f"{atom.atom_id}: annotated NOT equivalent "
                            f"across modalities — cannot construct a "
                            f"modality counterfactual (family belongs "
                            f"with negative controls)"
                        )
                    else:
                        reasons.append(
                            f"cross-modal equivalence unresolved on "
                            f"{atom.atom_id} ('{state}') — modality and "
                            f"wording would be confounded"
                        )
    if "visual_relevance" in requirements:
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                if atom.risk_relevance not in FACTORIAL_RISK_RELEVANCE:
                    if atom.risk_relevance in ANNOTATED_RISK_RELEVANCE:
                        reasons.append(
                            f"visual atom {atom.atom_id}: annotated "
                            f"irrelevant — an irrelevant image cannot "
                            f"ground a causal visual condition (family "
                            f"belongs with negative controls)"
                        )
                    else:
                        reasons.append(
                            f"visual atom {atom.atom_id} risk_relevance "
                            f"is '{atom.risk_relevance}' — image presence "
                            f"must not be confused with risk relevance"
                        )
    if "joint_interpretation" in requirements:
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                if atom.required_for_joint_interpretation is None:
                    reasons.append(
                        f"visual atom {atom.atom_id} "
                        f"required_for_joint_interpretation is null"
                    )
                elif atom.required_for_joint_interpretation is not True:
                    reasons.append(
                        f"visual atom {atom.atom_id}: annotated "
                        f"required_for_joint_interpretation=False — "
                        f"cross_modal is not a causal candidate"
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
