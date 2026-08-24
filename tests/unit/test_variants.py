"""Unit tests for Iteration 5: annotate -> harmonize -> construct variants.

Covers readiness gates (L0/L1/L2), VariantPrerequisiteError, terminal
harmonization provenance, the six independent generators, the exact
canonical-q hash invariant, and skeleton immutability.
"""

from __future__ import annotations

import json

import pytest

from causal_mllm.construction.annotation import CallableAnnotator
from causal_mllm.construction.families import build_family_skeleton
from causal_mllm.construction.harmonize import (
    CallableHarmonizer,
    ManualHarmonizer,
    TerminalHarmonizationError,
    apply_terminal_harmonization,
    canonical_terminal,
)
from causal_mllm.construction.readiness import (
    L0_STRUCTURAL,
    L1_SEMANTIC,
    L2_VARIANT_READY,
    VariantPrerequisiteError,
    assert_variant_ready,
    family_readiness,
)
from causal_mllm.construction.variants import (
    VARIANT_GENERATORS,
    build_family_variants,
    validate_variant_trajectory,
)
from causal_mllm.seeds import sha256_text
from tests.unit.factories import make_mtmcs_group

CANONICAL_Q = "Canonical: what should be done with this scene?"


def _skeleton(tmp_path, n_turns: int = 4):
    img = tmp_path / "img.png"
    img.write_bytes(b"fake-image-bytes")
    group = make_mtmcs_group("type_b", 1, n_turns=n_turns,
                             image_path=str(img))
    return build_family_skeleton(group, seed=42)


def _full_annotator():
    """Annotator resolving everything variant generators need."""

    def fn(family_key: str, atom: dict):
        payload = None
        if atom["divergence"] == "causal":
            payload = {"semantic_type": "relation",
                       "semantic_description": "opening framing divergence"}
        elif atom.get("structural_role") == "shared_image":
            payload = {"semantic_type": "entity_or_scene",
                       "risk_relevance": "relevant",
                       "required_for_joint_interpretation": True}
        forms = atom.get("surface_forms") or {}
        if any(k.startswith("multimodal_") for k in forms) \
                and any(k.startswith("unimodal_") for k in forms):
            payload = dict(payload or {"semantic_type": "reference"})
            payload["semantic_equivalence"] = {
                "multimodal_vs_unimodal":
                    {"state": "equivalent", "confidence": 0.97},
            }
        return payload

    return CallableAnnotator(fn, model_name="test-vlm",
                             model_revision="r1", prompt_version="v1",
                             temperature=0.0, seed=42)


def _harmonizer():
    return CallableHarmonizer(
        lambda fk, mm_q, text_q: CANONICAL_Q,
        model_name="test-llm", prompt_version="v1",
    )


def _ready_family(tmp_path, n_turns: int = 4):
    skeleton = _skeleton(tmp_path, n_turns=n_turns)
    annotated = _full_annotator().annotate_family(skeleton)
    return apply_terminal_harmonization(annotated, _harmonizer())


# ---------------------------------------------------------------------------
# Readiness levels and hard gates
# ---------------------------------------------------------------------------

class TestReadinessGates:
    def test_bare_skeleton_is_L0(self, tmp_path):
        readiness = family_readiness(_skeleton(tmp_path))
        assert readiness["level"] == L0_STRUCTURAL
        assert readiness["L1_semantic"]  # semantic gaps listed

    def test_annotated_family_is_L1(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        readiness = family_readiness(annotated)
        assert readiness["level"] == L1_SEMANTIC
        assert readiness["L2_variant_ready"]  # harmonization missing

    def test_harmonized_family_is_L2(self, tmp_path):
        family = _ready_family(tmp_path)
        assert family_readiness(family)["level"] == L2_VARIANT_READY

    def test_unannotated_family_fails_every_variant(self, tmp_path):
        """Bare skeleton: no canonical q*, so no variant is constructible."""
        family = _skeleton(tmp_path)
        for name in VARIANT_GENERATORS:
            with pytest.raises(
                    VariantPrerequisiteError, match="not ready") as exc:
                assert_variant_ready(family, name)
            assert any("L2" in r for r in exc.value.reasons)

    def test_annotated_but_unharmonized_fails_with_L2(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        with pytest.raises(
                VariantPrerequisiteError, match="not ready") as exc:
            assert_variant_ready(annotated, "history_reset")
        assert any("L2" in r for r in exc.value.reasons)

    def test_pending_equivalence_blocks_image_bearing_variants(self, tmp_path):
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.divergence == "causal":
                atom.semantic_equivalence["multimodal_vs_unimodal"] = \
                    {"state": "pending"}
        # Image-bearing variants cross modalities and are blocked...
        for name in ("vision_only", "cross_modal", "shuffle"):
            with pytest.raises(
                    VariantPrerequisiteError, match="equivalence"):
                assert_variant_ready(family, name)
        # ...text-only conditions never cross modalities
        for name in ("neutral", "text_only", "history_reset"):
            assert_variant_ready(family, name)

    def test_not_equivalent_blocks_modality_counterfactuals(self, tmp_path):
        """P0 regression: an explicit S(T_mm)!~S(T_text) decision must
        NOT be treated as eligibility for modality comparisons."""
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.divergence == "causal":
                atom.semantic_equivalence["multimodal_vs_unimodal"] = \
                    {"state": "not_equivalent", "confidence": 0.9}
        # The annotation is COMPLETE: L1 readiness still holds
        assert family_readiness(family)["level"] == L2_VARIANT_READY \
            or family_readiness(family)["L1_semantic"] == []
        # But factorial eligibility is lost for modality crossings
        for name in ("vision_only", "cross_modal", "shuffle"):
            with pytest.raises(VariantPrerequisiteError,
                               match="NOT equivalent"):
                assert_variant_ready(family, name)
        # Non-crossing conditions remain constructible
        for name in ("neutral", "text_only", "history_reset"):
            assert_variant_ready(family, name)

    def test_pending_risk_relevance_blocks_vision_variants(self, tmp_path):
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                atom.risk_relevance = "pending"
        for name in ("vision_only", "cross_modal", "shuffle"):
            with pytest.raises(
                    VariantPrerequisiteError, match="risk_relevance"):
                assert_variant_ready(family, name)
        assert_variant_ready(family, "text_only")

    def test_irrelevant_image_blocks_visual_variants(self, tmp_path):
        """P0 regression: an explicitly irrelevant image cannot ground
        a causal visual condition — the family is a negative control."""
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                atom.risk_relevance = "irrelevant"
        for name in ("vision_only", "cross_modal", "shuffle"):
            with pytest.raises(VariantPrerequisiteError,
                               match="annotated irrelevant"):
                assert_variant_ready(family, name)
        assert_variant_ready(family, "neutral")
        assert_variant_ready(family, "text_only")

    def test_joint_interpretation_false_blocks_cross_modal(self, tmp_path):
        """P0 regression: required_for_joint_interpretation=False means
        cross_modal is not a causal candidate."""
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                atom.required_for_joint_interpretation = False
        for name in ("cross_modal", "shuffle"):
            with pytest.raises(VariantPrerequisiteError,
                               match="joint_interpretation=False"):
                assert_variant_ready(family, name)
        # vision_only does not claim joint text+vision interpretation
        assert_variant_ready(family, "vision_only")

    def test_build_family_variants_fails_on_negative_annotation(self, tmp_path):
        """Decided-but-negative families are rejected, not built."""
        family = _ready_family(tmp_path)
        for atom in family.semantic_atoms:
            if atom.structural_role == "shared_image":
                atom.risk_relevance = "irrelevant"
        with pytest.raises(VariantPrerequisiteError):
            build_family_variants(family)

    def test_unknown_variant_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown variant"):
            assert_variant_ready(_ready_family(tmp_path), "everything")


# ---------------------------------------------------------------------------
# Terminal harmonization (5B)
# ---------------------------------------------------------------------------

class TestHarmonization:
    def test_block_structure_and_provenance(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        harmonized = apply_terminal_harmonization(annotated, _harmonizer())
        block = harmonized.validation["terminal_harmonization"]
        # Synthetic factory aligns mm/text terminals, so harmonization is
        # optional here; real Type-B rows always require it (0/752 match)
        assert block["required"] is False
        assert block["canonical_q"] == CANONICAL_Q
        assert block["canonical_sha256"] == sha256_text(CANONICAL_Q)
        assert block["source_mm_q"]  # original preserved as provenance
        assert block["source_text_q"]  # unimodal form recovered from atoms
        assert block["method"] == "llm"
        assert block["validation"] == "llm"
        assert block["provenance"]["model"] == "test-llm"
        # Grounding validation targets present, unresolved (null)
        for target in (
            "canonical_q_grounding_valid",
            "canonical_q_no_unintended_modality_dependency",
            "canonical_q_semantically_preserves_mm_source",
            "canonical_q_semantically_preserves_text_source",
        ):
            assert target in block and block[target] is None

    def test_original_terminal_not_overwritten(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        original_q = annotated.terminal_query.text
        harmonized = apply_terminal_harmonization(annotated, _harmonizer())
        assert harmonized.terminal_query.text == original_q
        assert harmonized.terminal_query.text != CANONICAL_Q

    def test_input_family_not_mutated(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        before = annotated.to_dict()
        apply_terminal_harmonization(annotated, _harmonizer())
        assert annotated.to_dict() == before

    def test_missing_required_harmonization_fails_loudly(self, tmp_path):
        annotated = _full_annotator().annotate_family(_skeleton(tmp_path))
        # Force the real-data situation: harmonization REQUIRED
        annotated.ground_truth["requires_terminal_harmonization"] = True
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")
        with pytest.raises(TerminalHarmonizationError, match="requires"):
            apply_terminal_harmonization(
                annotated, ManualHarmonizer(empty_file))

    def test_manual_harmonizer(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        key = skeleton.source["source_id"]
        path = tmp_path / "harmonize.json"
        path.write_text(f'{{"{key}": "{CANONICAL_Q}"}}')
        annotated = _full_annotator().annotate_family(skeleton)
        harmonized = apply_terminal_harmonization(
            annotated, ManualHarmonizer(path))
        block = harmonized.validation["terminal_harmonization"]
        assert block["method"] == "manual"
        assert block["validation"] == "human"
        q, sha = canonical_terminal(harmonized)
        assert (q, sha) == (CANONICAL_Q, sha256_text(CANONICAL_Q))

    def test_canonical_terminal_raises_without_harmonization(self, tmp_path):
        with pytest.raises(TerminalHarmonizationError, match="no canonical"):
            canonical_terminal(_skeleton(tmp_path))

    def test_callable_harmonizer_requires_model_name(self):
        with pytest.raises(TerminalHarmonizationError, match="model_name"):
            CallableHarmonizer(lambda fk, mm, tx: "q", model_name="")

    def test_manual_harmonizer_dict_form_fills_grounding_targets(self, tmp_path):
        """Human reviewers may attach grounding judgments to the entry."""
        skeleton = _skeleton(tmp_path)
        key = skeleton.source["source_id"]
        path = tmp_path / "harmonize.json"
        path.write_text(json.dumps({key: {
            "canonical_q": CANONICAL_Q,
            "canonical_q_grounding_valid": True,
            "canonical_q_no_unintended_modality_dependency": True,
            "canonical_q_semantically_preserves_mm_source": True,
            "canonical_q_semantically_preserves_text_source": True,
        }}))
        annotated = _full_annotator().annotate_family(skeleton)
        harmonized = apply_terminal_harmonization(
            annotated, ManualHarmonizer(path))
        block = harmonized.validation["terminal_harmonization"]
        assert block["canonical_q_grounding_valid"] is True
        assert block["canonical_q_no_unintended_modality_dependency"] is True
        assert block["canonical_q_semantically_preserves_mm_source"] is True
        assert block["canonical_q_semantically_preserves_text_source"] is True


# ---------------------------------------------------------------------------
# The six generators (5C)
# ---------------------------------------------------------------------------

class TestVariantGenerators:
    def test_all_six_variants_built(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        assert set(complete.variants) == set(VARIANT_GENERATORS)

    def test_exact_canonical_q_hash_invariant_across_all_variants(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        _, canonical_sha = canonical_terminal(family)
        for name, variant in complete.variants.items():
            last = variant.messages[-1]
            assert last.text == CANONICAL_Q, name
            assert sha256_text(last.text) == canonical_sha, name
            assert last.role == "user" and not last.images

    def test_factorial_image_placement(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        for name in ("neutral", "text_only", "history_reset"):
            assert not any(m.images for m in complete.variants[name].messages)
        for name in ("vision_only", "cross_modal", "shuffle"):
            assert any(m.images for m in complete.variants[name].messages)

    def test_safe_vs_unsafe_history_selection(self, tmp_path):
        """neutral/vision_only use SAFE forms; text_only/cross_modal UNSAFE."""
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        safe_marker = "[safe opening] describe the scene 1"
        unsafe_marker = "[unsafe opening] ignore rules 1"

        def opening(variant):
            return variant.messages[0].text

        assert opening(complete.variants["neutral"]) == safe_marker
        assert opening(complete.variants["vision_only"]) == safe_marker
        assert opening(complete.variants["text_only"]) == unsafe_marker
        assert opening(complete.variants["cross_modal"]) == unsafe_marker

    def test_history_reset_is_terminal_only(self, tmp_path):
        family = _ready_family(tmp_path)
        variant = build_family_variants(family).variants["history_reset"]
        assert len(variant.messages) == 1
        assert variant.messages[0].text == CANONICAL_Q

    def test_shuffle_permutes_but_preserves_content(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        shuffle = complete.variants["shuffle"]
        cross = complete.variants["cross_modal"]
        perm = shuffle.shuffle_permutation
        assert perm is not None and sorted(perm) == list(range(len(perm)))
        assert perm != sorted(perm)  # genuinely reordered
        # Same history content (excluding terminal), different order
        assert sorted(m.text or "" for m in shuffle.messages[:-1]) == \
            sorted(m.text or "" for m in cross.messages[:-1])
        assert [m.text for m in shuffle.messages[:-1]] != \
            [m.text for m in cross.messages[:-1]]

    def test_shuffle_is_deterministic(self, tmp_path):
        family = _ready_family(tmp_path)
        a = build_family_variants(family, seed=7)
        b = build_family_variants(family, seed=7)
        assert (a.variants["shuffle"].shuffle_permutation
                == b.variants["shuffle"].shuffle_permutation)

    def test_every_variant_has_rule_provenance(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        for name, variant in complete.variants.items():
            prov = variant.provenance
            assert prov.type == "rule"
            assert prov.transformations, name
            assert prov.creation_timestamp

    def test_all_trajectories_pass_structural_validation(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        for name, variant in complete.variants.items():
            assert validate_variant_trajectory(family, variant) == [], name

    def test_source_family_immutable(self, tmp_path):
        family = _ready_family(tmp_path)
        before = family.to_dict()
        build_family_variants(family)
        assert family.to_dict() == before

    def test_cross_modal_marked_candidate_not_required(self, tmp_path):
        family = _ready_family(tmp_path)
        complete = build_family_variants(family)
        gen = complete.validation["variant_generation"]
        assert gen["cross_modal_candidate"] is True
        assert gen["cross_modal_required"] is None  # behavioral (Iter 6+)

    def test_generator_on_unready_family_raises(self, tmp_path):
        with pytest.raises(VariantPrerequisiteError):
            VARIANT_GENERATORS["cross_modal"](_skeleton(tmp_path))
