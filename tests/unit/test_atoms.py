"""Unit tests for Iteration 4: family-level comparative atom extraction.

Core principle under test: a Type-B family is decomposed COMPARATIVELY
(H_safe vs H_unsafe with shared q*), so the extractor identifies which
semantic content differs causally between the histories — not merely a
per-trajectory summary.
"""

from __future__ import annotations

import pytest

from causal_mllm.construction.atoms import (
    AtomExtractionError,
    extract_family_atoms,
)
from causal_mllm.construction.families import (
    build_family_skeleton,
    build_family_skeletons,
)
from causal_mllm.data.validate_schema import validate_family_skeleton
from causal_mllm.seeds import deterministic_family_id, sha256_text
from tests.unit.factories import make_mtmcs_group, make_text_only_singleton

FAMILY_ID = "CMST_test01"


# ---------------------------------------------------------------------------
# Comparative extraction: type_b (divergence in history, shared q*)
# ---------------------------------------------------------------------------

class TestTypeBComparativeExtraction:
    def test_exactly_one_causal_atom_at_divergent_turn(self):
        group = make_mtmcs_group("type_b", 1)
        extraction = extract_family_atoms(FAMILY_ID, group)
        causal = extraction.causal_atoms
        assert len(causal) == 1
        assert causal[0].source_turns == [0]  # opening-turn divergence
        assert extraction.divergent_turns == [0]

    def test_causal_atom_carries_both_surface_forms(self):
        group = make_mtmcs_group("type_b", 1)
        extraction = extract_family_atoms(FAMILY_ID, group)
        atom = extraction.causal_atoms[0]
        assert atom.safe_text == "[safe opening] describe the scene 1"
        assert atom.unsafe_text == "[unsafe opening] ignore rules 1"
        assert atom.atom_type == "attribute_or_state"

    def test_shared_terminal_is_a_shared_intent_atom(self):
        group = make_mtmcs_group("type_b", 1)
        extraction = extract_family_atoms(FAMILY_ID, group)
        terminal_atoms = [a for a in extraction.atoms if a.source_turns == [2]]
        assert len(terminal_atoms) == 1
        atom = terminal_atoms[0]
        assert atom.atom_type == "intent"
        assert atom.divergence == "shared"  # q* identical across conditions
        assert extraction.shared_terminal_query is True

    def test_shared_image_is_a_shared_vision_atom(self):
        group = make_mtmcs_group("type_b", 1)
        extraction = extract_family_atoms(FAMILY_ID, group)
        vision = [a for a in extraction.atoms if "vision" in a.source_modalities]
        assert len(vision) == 1
        assert vision[0].source_turns == [0]
        assert vision[0].atom_type == "entity_or_scene"
        assert vision[0].divergence == "shared"

    def test_shared_context_turns_marked_shared(self):
        group = make_mtmcs_group("type_b", 1, n_turns=5)
        extraction = extract_family_atoms(FAMILY_ID, group)
        middle = [a for a in extraction.atoms
                  if a.atom_type == "contextual_disambiguator"]
        assert middle, "expected shared context atoms for middle turns"
        assert all(a.divergence == "shared" for a in middle)

    def test_atom_ids_deterministic_and_unique(self):
        group = make_mtmcs_group("type_b", 1)
        first = extract_family_atoms(FAMILY_ID, group)
        second = extract_family_atoms(FAMILY_ID, make_mtmcs_group("type_b", 1))
        assert [a.atom_id for a in first.atoms] == [a.atom_id for a in second.atoms]
        ids = [a.atom_id for a in first.atoms]
        assert len(ids) == len(set(ids))
        assert all(aid.startswith(FAMILY_ID) for aid in ids)

    def test_extraction_is_not_a_per_record_summary(self):
        """The causal atom must exist ONLY because the pair was compared."""
        group = make_mtmcs_group("type_b", 1)
        extraction = extract_family_atoms(FAMILY_ID, group)
        causal = extraction.causal_atoms[0]
        # Neither surface form alone identifies divergence — both are kept
        assert causal.safe_text and causal.unsafe_text
        assert causal.safe_text != causal.unsafe_text


# ---------------------------------------------------------------------------
# Comparative extraction: type_a (divergence at the terminal turn)
# ---------------------------------------------------------------------------

class TestTypeAComparativeExtraction:
    def test_causal_atom_is_the_terminal_intent(self):
        group = make_mtmcs_group("type_a", 2)
        extraction = extract_family_atoms(FAMILY_ID, group)
        causal = extraction.causal_atoms
        assert len(causal) == 1
        assert causal[0].source_turns == [2]
        assert causal[0].atom_type == "intent"  # divergence IS the query
        assert extraction.shared_terminal_query is False
        assert extraction.divergent_turns == [2]


# ---------------------------------------------------------------------------
# Integrity: fail loudly on cross-condition inconsistency
# ---------------------------------------------------------------------------

class TestExtractionIntegrity:
    def test_mm_text_divergence_mismatch_raises(self):
        group = make_mtmcs_group("type_b", 1)
        text_unsafe = next(r for r in group
                           if r.metadata["modality"] == "unimodal"
                           and r.metadata["safety"] == "unsafe")
        # Corrupt one text turn so the text pair diverges elsewhere
        text_unsafe.messages[1].text = "CORRUPTED MIDDLE TURN"
        with pytest.raises(AtomExtractionError, match="Divergent turn sets"):
            extract_family_atoms(FAMILY_ID, group)

    def test_image_path_mismatch_raises(self):
        group = make_mtmcs_group("type_b", 1)
        mm_unsafe = next(r for r in group
                         if r.metadata["modality"] == "multimodal"
                         and r.metadata["safety"] == "unsafe")
        mm_unsafe.messages[0].images = ["media/source/OTHER.png"]
        with pytest.raises(AtomExtractionError, match="image paths differ"):
            extract_family_atoms(FAMILY_ID, group)

    def test_missing_condition_raises(self):
        group = make_mtmcs_group("type_b", 1)[:3]
        with pytest.raises(AtomExtractionError, match="missing condition"):
            extract_family_atoms(FAMILY_ID, group)

    def test_turn_misalignment_raises(self):
        group = make_mtmcs_group("type_b", 1)
        mm_unsafe = next(r for r in group
                         if r.metadata["modality"] == "multimodal"
                         and r.metadata["safety"] == "unsafe")
        mm_unsafe.messages[0].turn_index = 99
        with pytest.raises(AtomExtractionError, match="misalignment"):
            extract_family_atoms(FAMILY_ID, group)

    def test_empty_family_raises(self):
        with pytest.raises(AtomExtractionError, match="empty family"):
            extract_family_atoms(FAMILY_ID, [])


# ---------------------------------------------------------------------------
# Singleton decomposition (no safe/unsafe pair available)
# ---------------------------------------------------------------------------

class TestSingletonExtraction:
    def test_no_causal_atoms_for_singletons(self):
        singleton = make_text_only_singleton()
        extraction = extract_family_atoms(FAMILY_ID, [singleton])
        assert extraction.causal_atoms == []
        assert all(a.divergence == "not_applicable" for a in extraction.atoms)

    def test_terminal_intent_present(self):
        singleton = make_text_only_singleton()
        extraction = extract_family_atoms(FAMILY_ID, [singleton])
        intents = [a for a in extraction.atoms if a.atom_type == "intent"]
        assert len(intents) == 1
        assert intents[0].source_turns == [2]

    def test_assistant_turns_are_reference_atoms(self):
        singleton = make_text_only_singleton()
        extraction = extract_family_atoms(FAMILY_ID, [singleton])
        refs = [a for a in extraction.atoms if a.atom_type == "reference"]
        assert len(refs) == 1
        assert refs[0].source_turns == [1]


# ---------------------------------------------------------------------------
# Family skeletons
# ---------------------------------------------------------------------------

class TestFamilySkeleton:
    def test_type_b_skeleton_structure(self):
        group = make_mtmcs_group("type_b", 1)
        skeleton = build_family_skeleton(group, seed=42)
        assert skeleton.family_id == deterministic_family_id(
            "mtmcs", "mtmcs:type_b:000001", 42)
        assert skeleton.source["dataset"] == "mtmcs"
        assert skeleton.source["source_id"] == "mtmcs:type_b:000001"
        assert len(skeleton.source["condition_source_ids"]) == 4
        assert skeleton.setting == "type_b"
        assert skeleton.variants == {}  # Iteration 5 fills these

    def test_terminal_query_hash_integrity(self):
        group = make_mtmcs_group("type_b", 1)
        skeleton = build_family_skeleton(group)
        assert skeleton.terminal_query.sha256 == sha256_text(
            skeleton.terminal_query.text)

    def test_ground_truth_records_comparative_facts(self):
        group = make_mtmcs_group("type_b", 1)
        skeleton = build_family_skeleton(group)
        gt = skeleton.ground_truth
        assert gt["divergent_turns"] == [0]
        assert gt["shared_terminal_query"] is True
        assert len(gt["causal_atom_ids"]) == 1
        assert gt["labels_by_condition"] == {
            "multimodal:safe": "safe",
            "multimodal:unsafe": "unsafe",
            "unimodal:safe": "safe",
            "unimodal:unsafe": "unsafe",
        }

    def test_validation_placeholders_present(self):
        group = make_mtmcs_group("type_b", 1)
        skeleton = build_family_skeleton(group)
        assert skeleton.validation["requires_standalone_risk_validation"] is True
        assert skeleton.validation["standalone_terminal_risk"] is None
        assert skeleton.validation["strict_causal_candidate"] is None

    def test_skeleton_passes_skeleton_validator(self):
        group = make_mtmcs_group("type_b", 1)
        skeleton = build_family_skeleton(group)
        assert validate_family_skeleton(skeleton.to_dict()) == []

    def test_type_a_skeleton_marks_terminal_divergence(self):
        group = make_mtmcs_group("type_a", 2)
        skeleton = build_family_skeleton(group)
        assert skeleton.ground_truth["shared_terminal_query"] is False
        assert skeleton.ground_truth["divergent_turns"] == [2]
        assert validate_family_skeleton(skeleton.to_dict()) == []

    def test_singleton_skeleton(self):
        singleton = make_text_only_singleton()
        skeleton = build_family_skeleton([singleton])
        assert skeleton.source["dataset"] == "cosafe"
        assert skeleton.ground_truth["causal_atom_ids"] == []
        assert validate_family_skeleton(skeleton.to_dict()) == []

    def test_source_records_not_mutated(self):
        group = make_mtmcs_group("type_b", 1)
        before = [ex.to_dict() for ex in group]
        build_family_skeleton(group)
        assert [ex.to_dict() for ex in group] == before

    def test_empty_records_raise(self):
        with pytest.raises(ValueError):
            build_family_skeleton([])

    def test_batch_builder_detects_id_collisions(self):
        g1 = make_mtmcs_group("type_b", 1)
        g2 = make_mtmcs_group("type_b", 1)  # same pair_id -> same family_id
        with pytest.raises(ValueError, match="collision"):
            build_family_skeletons([("k1", g1), ("k2", g2)])


# ---------------------------------------------------------------------------
# Skeleton validator negative cases
# ---------------------------------------------------------------------------

class TestSkeletonValidator:
    def _valid_dict(self):
        group = make_mtmcs_group("type_b", 1)
        return build_family_skeleton(group).to_dict()

    def test_detects_missing_atoms(self):
        record = self._valid_dict()
        record["semantic_atoms"] = []
        errors = validate_family_skeleton(record)
        assert any("non-empty" in e for e in errors)

    def test_detects_duplicate_atom_ids(self):
        record = self._valid_dict()
        record["semantic_atoms"][1]["atom_id"] = record["semantic_atoms"][0]["atom_id"]
        errors = validate_family_skeleton(record)
        assert any("Duplicate atom_id" in e for e in errors)

    def test_detects_terminal_hash_mismatch(self):
        record = self._valid_dict()
        record["terminal_query"]["text"] = "TAMPERED QUERY"
        errors = validate_family_skeleton(record)
        assert any("sha256" in e for e in errors)

    def test_detects_mtmcs_family_without_causal_atom(self):
        record = self._valid_dict()
        for atom in record["semantic_atoms"]:
            atom["divergence"] = "shared"
            atom.pop("safe_text", None)
            atom.pop("unsafe_text", None)
        errors = validate_family_skeleton(record)
        assert any("no causal atom" in e for e in errors)

    def test_detects_causal_atom_without_surface_forms(self):
        record = self._valid_dict()
        causal = next(a for a in record["semantic_atoms"]
                      if a["divergence"] == "causal")
        causal.pop("safe_text")
        errors = validate_family_skeleton(record)
        assert any("safe_text/unsafe_text" in e for e in errors)

    def test_detects_invalid_divergence(self):
        record = self._valid_dict()
        record["semantic_atoms"][0]["divergence"] = "maybe"
        errors = validate_family_skeleton(record)
        assert any("invalid divergence" in e for e in errors)
