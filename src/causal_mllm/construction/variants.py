"""Six independent variant generators (Iteration 5C).

Generators consume RESOLVED semantic atoms and the canonical q* —
never raw MTMCS field names. Each generator:

  1. asserts its variant-specific prerequisites (readiness gates)
  2. builds messages from atom surface forms
  3. attaches full provenance to every transformation
  4. leaves the source family immutable

Factorial mapping (2x2: history risk T x vision V), with one canonical
q* shared by ALL six conditions (hash-invariant):

    neutral       H_00  safe history text,        no image   + q*
    text_only     H_10  unsafe history text,      no image   + q*
    vision_only   H_01  safe history text,        + image    + q*
    cross_modal   H_11  unsafe history text,      + image    + q*
    shuffle             cross_modal history, permuted order  + q*
    history_reset       q* alone (minimal context)

NAMING NOTE: the variant names are convenience aliases for the
factorial cells H_ij. 'vision_only' does NOT mean 'no text semantics'
— it is H_01 (T=0, V=1): a SAFE textual history plus the shared image.
T=1 means 'unsafe/risk-bearing textual history', not 'any text exists'.
Canonical cells: H00 = safe_text + no_image, H10 = unsafe_text +
no_image, H01 = safe_text + image, H11 = unsafe_text + image.

Prerequisite strength differs per variant (see readiness.py): text-only
conditions are structural + canonical q*; image-bearing conditions
additionally require ANNOTATED-POSITIVE evidence — equivalence ==
'equivalent', risk_relevance == 'relevant' (not merely decided), and
for cross_modal/shuffle required_for_joint_interpretation == True.
A decided-but-negative annotation (not_equivalent / irrelevant /
False) REJECTS the family from the causal subset.

Cross-modal CAUSALITY (the strict subset) is NOT claimed here: a built
cross_modal variant is a cross_modal_CANDIDATE until Iteration 6
behavioral evidence establishes Risk(T)<θ, Risk(V)<θ, Risk(T,V)>=θ.
"""

from __future__ import annotations

import datetime
import random

from causal_mllm.construction.harmonize import canonical_terminal
from causal_mllm.construction.readiness import (
    ALL_VARIANT_NAMES,
    assert_variant_ready,
)
from causal_mllm.data.schemas import (
    CausalFamily,
    GeneratorProvenance,
    Message,
    SemanticAtom,
    VariantData,
)
from causal_mllm.seeds import get_git_commit, sha256_text

# Which surface-form condition each factorial condition draws from
_VISION_VARIANTS = {"vision_only", "cross_modal", "shuffle"}
_CONDITION_KEYS = {
    "neutral": "unimodal_safe",
    "text_only": "unimodal_unsafe",
    "vision_only": "multimodal_safe",
    "cross_modal": "multimodal_unsafe",
    "shuffle": "multimodal_unsafe",
}

_HISTORY_ROLES = {"divergent_history_turn", "shared_history_turn"}


class VariantConstructionError(RuntimeError):
    """Raised when a variant cannot be assembled from resolved atoms."""


def _provenance(transformations: list[str], seed: int) -> GeneratorProvenance:
    return GeneratorProvenance(
        type="rule",
        model=None,
        prompt_version="v1",
        seed=seed,
        parent_variant="source",
        transformations=transformations,
        creation_timestamp=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        config_hash=None,
        source_revision=None,
    )


def _history_atoms(family: CausalFamily) -> dict[int, dict[str, SemanticAtom]]:
    """Group history atoms by turn index: {turn: {'text': atom|None,
    'vision': atom|None}}."""
    by_turn: dict[int, dict[str, SemanticAtom]] = {}
    for atom in family.semantic_atoms:
        if atom.structural_role in _HISTORY_ROLES:
            turn = atom.source_turns[0]
            by_turn.setdefault(turn, {"text": None, "vision": None})
            by_turn[turn]["text"] = atom
        elif atom.structural_role == "shared_image":
            turn = atom.source_turns[0]
            by_turn.setdefault(turn, {"text": None, "vision": None})
            by_turn[turn]["vision"] = atom
    return by_turn


def _history_messages(family: CausalFamily, variant: str) -> list[Message]:
    """Assemble history messages from atom surface forms."""
    condition_key = _CONDITION_KEYS[variant]
    include_vision = variant in _VISION_VARIANTS
    messages: list[Message] = []

    for turn in sorted(_history_atoms(family)):
        atoms = _history_atoms(family)[turn]
        text = None
        images: list[str] = []

        text_atom = atoms["text"]
        if text_atom is not None:
            form = (text_atom.surface_forms or {}).get(condition_key)
            if form is None:
                raise VariantConstructionError(
                    f"Atom {text_atom.atom_id} lacks surface form "
                    f"'{condition_key}' needed by variant '{variant}'"
                )
            text = form.get("text")

        vision_atom = atoms["vision"]
        if vision_atom is not None and include_vision:
            # Explicit media references — no inferring the image asset
            images = [m["path"] for m in vision_atom.source_media]
            if not images:
                raise VariantConstructionError(
                    f"Vision atom {vision_atom.atom_id} has no source_media"
                )

        if text or images:
            messages.append(Message(
                turn_index=len(messages), role="user",
                text=text, images=images,
            ))
    return messages


def _terminal_message(family: CausalFamily, turn_index: int) -> Message:
    q, _ = canonical_terminal(family)
    return Message(turn_index=turn_index, role="user", text=q, images=[])


def _finish(family: CausalFamily, variant: str, messages: list[Message],
            transformations: list[str], seed: int,
            permutation: list[int] | None = None) -> VariantData:
    """Reindex, append the canonical q*, attach provenance, validate."""
    # Sequential turn indices regardless of source/permutation order
    for i, msg in enumerate(messages):
        msg.turn_index = i
    terminal = _terminal_message(family, len(messages))
    variant_data = VariantData(
        name=variant,
        messages=[*messages, terminal],
        provenance=_provenance(transformations, seed),
        shuffle_permutation=permutation,
    )
    errors = validate_variant_trajectory(family, variant_data)
    if errors:
        raise VariantConstructionError(
            f"Variant '{variant}' for {family.family_id} failed structural "
            f"checks:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return variant_data


# ---------------------------------------------------------------------------
# The six generators — independent, each gated by its own prerequisites
# ---------------------------------------------------------------------------

def build_neutral(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """H_00: topic-matched neutral (safe) history, text only, + q*."""
    assert_variant_ready(family, "neutral")
    history = _history_messages(family, "neutral")
    return _finish(
        family, "neutral", history,
        ["history:safe_text_form", "modality:drop_images",
         "terminal:canonical_q"], seed,
    )


def build_text_only(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """H_10: unsafe history with all relevant semantics in text, + q*."""
    assert_variant_ready(family, "text_only")
    history = _history_messages(family, "text_only")
    return _finish(
        family, "text_only", history,
        ["history:unsafe_text_form", "modality:drop_images",
         "terminal:canonical_q"], seed,
    )


def build_vision_only(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """H_01: neutral text history + relevant visual evidence, + q*."""
    assert_variant_ready(family, "vision_only")
    history = _history_messages(family, "vision_only")
    if not any(m.images for m in history):
        raise VariantConstructionError(
            f"vision_only for {family.family_id} has no image-bearing "
            f"history turn"
        )
    return _finish(
        family, "vision_only", history,
        ["history:safe_text_form", "modality:attach_shared_image",
         "terminal:canonical_q"], seed,
    )


def build_cross_modal(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """H_11: complementary text + vision (main treatment), + q*.

    Produces a cross_modal_CANDIDATE. Strict cross-modal causality is a
    behavioral claim established in Iteration 6+, not here.
    """
    assert_variant_ready(family, "cross_modal")
    history = _history_messages(family, "cross_modal")
    if not any(m.images for m in history):
        raise VariantConstructionError(
            f"cross_modal for {family.family_id} has no image-bearing "
            f"history turn"
        )
    return _finish(
        family, "cross_modal", history,
        ["history:unsafe_text_form", "modality:attach_shared_image",
         "terminal:canonical_q"], seed,
    )


def build_shuffle(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """Same content as cross_modal, permuted temporal order, + q*.

    The permutation is deterministic (seeded by family + global seed)
    and recorded in shuffle_permutation.
    """
    assert_variant_ready(family, "shuffle")
    history = _history_messages(family, "shuffle")
    if len(history) < 2:
        raise VariantConstructionError(
            f"shuffle for {family.family_id} needs >= 2 history turns, "
            f"got {len(history)}"
        )
    order = list(range(len(history)))
    rng = random.Random(f"{family.family_id}:shuffle:{seed}")
    rng.shuffle(order)
    if order == sorted(order):  # avoid the identity permutation
        order = order[1:] + order[:1]
    permuted = [history[i] for i in order]
    return _finish(
        family, "shuffle", permuted,
        ["history:unsafe_text_form", "modality:attach_shared_image",
         "order:deterministic_permutation", "terminal:canonical_q"],
        seed, permutation=order,
    )


def build_history_reset(family: CausalFamily, *, seed: int = 42) -> VariantData:
    """Terminal query with minimal context: q* alone."""
    assert_variant_ready(family, "history_reset")
    return _finish(
        family, "history_reset", [],
        ["history:removed", "terminal:canonical_q"], seed,
    )


VARIANT_GENERATORS = {
    "neutral": build_neutral,
    "text_only": build_text_only,
    "vision_only": build_vision_only,
    "cross_modal": build_cross_modal,
    "shuffle": build_shuffle,
    "history_reset": build_history_reset,
}


# ---------------------------------------------------------------------------
# Structural validation of a generated trajectory
# ---------------------------------------------------------------------------

def validate_variant_trajectory(family: CausalFamily,
                                variant: VariantData) -> list[str]:
    """Structural checks for one generated trajectory."""
    errors: list[str] = []
    msgs = variant.messages

    if not msgs:
        return ["variant has no messages"]

    # Sequential turn indices
    if [m.turn_index for m in msgs] != list(range(len(msgs))):
        errors.append(f"turn indices not sequential: "
                      f"{[m.turn_index for m in msgs]}")

    # Canonical-q hash invariant: the LAST user turn must be exactly q*
    try:
        q, sha = canonical_terminal(family)
        last = msgs[-1]
        if last.role != "user":
            errors.append("terminal turn is not a user message")
        if last.text != q:
            errors.append("terminal text is not the canonical q*")
        elif sha256_text(last.text) != sha:
            errors.append("terminal sha256 differs from canonical hash")
        if last.images:
            errors.append("terminal turn must be text-only")
    except Exception as e:  # canonical q* missing
        errors.append(f"canonical q* unavailable: {e}")

    # Images only where the condition allows them
    if variant.name not in _VISION_VARIANTS:
        for m in msgs:
            if m.images:
                errors.append(
                    f"text-only condition '{variant.name}' carries images "
                    f"at turn {m.turn_index}"
                )

    if variant.name == "history_reset" and len(msgs) != 1:
        errors.append("history_reset must contain exactly the terminal turn")

    if variant.name == "shuffle" and variant.shuffle_permutation is None:
        errors.append("shuffle variant missing shuffle_permutation")

    return errors


def build_family_variants(family: CausalFamily, *,
                          seed: int = 42) -> CausalFamily:
    """Build all six variants for one harmonized, annotated family.

    Returns a COPY with variants populated; the input family is never
    mutated. Every prerequisite failure raises loudly — no silent
    fallback, no guessing.
    """
    result = family.__class__.from_dict(family.to_dict())
    for name in ALL_VARIANT_NAMES:
        result.variants[name] = VARIANT_GENERATORS[name](family, seed=seed)
    if result.validation is None:
        result.validation = {}
    result.validation["variant_generation"] = {
        # True is justified here: construction only succeeds when the
        # annotations are POSITIVE (equivalent / relevant /
        # joint-interpretation True). Decided-but-negative families
        # never reach this point — they are negative controls.
        "cross_modal_candidate": True,   # constructible; NOT yet causal
        "cross_modal_required": None,    # behavioral evidence (Iter 6+)
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "seed": seed,
    }
    return result
