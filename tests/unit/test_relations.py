"""Mutation tests for the factorial-relations firewall (Iter-6 hardening).

Iteration 6 is the FINAL integrity firewall over the persisted
artifact: it must catch a corrupted families.jsonl even when the
variant generators themselves are correct. Every mutation below
deliberately corrupts one property of an otherwise valid built family
and asserts EXCLUSION (errors produced, validation stage excludes).
"""

from __future__ import annotations

import json

from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.validation import (
    FACTORIAL_CELLS,
    automatic_family_checks,
    run_validation_stage,
    validate_factorial_relations,
    validate_factorial_semantic_eligibility,
)
from tests.unit.test_grounding import CLEAN_Q, _built_family


def _mutate(family, fn):
    """Apply fn to a deep dict copy and rebuild the family."""
    from causal_mllm.data.schemas import CausalFamily
    record = family.to_dict()
    fn(record)
    return CausalFamily.from_dict(record)


class TestCleanFamilyPasses:
    def test_no_errors_on_intact_family(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        assert validate_factorial_relations(family) == []
        assert validate_factorial_semantic_eligibility(family) == []
        errors, _ = automatic_family_checks(family)
        assert errors == []


class TestMutations:
    def _family(self, tmp_path):
        return _built_family(tmp_path, CLEAN_Q, fill_grounding=True)

    def test_changed_h11_image_detected(self, tmp_path):
        family = self._family(tmp_path)
        other = tmp_path / "other.png"
        other.write_bytes(b"\x89PNG\r\n\x1a\nother")

        def mutate(record):
            for message in record["variants"]["cross_modal"]["messages"]:
                if message["images"]:
                    message["images"][0] = str(other)
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("image hashes differ" in e or "not recorded" in e
                   for e in errors)

    def test_deleted_shuffle_message_detected(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            record["variants"]["shuffle"]["messages"].pop(0)

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert errors

    def test_altered_shuffle_message_detected(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            messages = record["variants"]["shuffle"]["messages"]
            for message in messages[:-1]:
                if message["text"]:
                    message["text"] += " (tampered)"
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("shuffle history content differs" in e or
                   "does not match" in e for e in errors)

    def test_corrupted_media_hash_detected(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                for media in atom["source_media"]:
                    media["sha256"] = "0" * 64
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("content hash differs" in e for e in errors)

    def test_identity_permutation_detected(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            n = len(record["variants"]["shuffle"]["messages"]) - 1
            record["variants"]["shuffle"]["shuffle_permutation"] = \
                list(range(n))
            # Restore the H11 order so ONLY the permutation is wrong
            record["variants"]["shuffle"]["messages"][:-1] = \
                record["variants"]["cross_modal"]["messages"][:-1]

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("identity" in e for e in errors)

    def test_image_in_text_only_condition_detected(self, tmp_path):
        family = self._family(tmp_path)
        img = next(
            m.images[0]
            for m in family.variants["cross_modal"].messages
            if m.images
        )

        def mutate(record):
            record["variants"]["text_only"]["messages"][0]["images"] = [img]

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("text-only condition" in e for e in errors)

    def test_divergent_terminal_hash_detected(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            block = record["validation"]["terminal_harmonization"]
            block["canonical_sha256"] = "f" * 64

        mutated = _mutate(family, mutate)
        errors = validate_factorial_relations(mutated)
        assert any("canonical q* hash" in e for e in errors)

    def test_media_file_replacement_detected(self, tmp_path):
        family = self._family(tmp_path)
        media_path = next(
            media["path"]
            for atom in family.semantic_atoms
            for media in atom.source_media
        )
        from pathlib import Path
        Path(media_path).write_bytes(b"\x89PNG\r\n\x1a\nreplaced content")
        errors = validate_factorial_relations(family)
        assert any("content hash differs" in e for e in errors)


class TestSemanticEligibilityMutations:
    """Flip the POSITIVE Iteration-5 annotations on an already-built
    family: the firewall must re-derive eligibility from the persisted
    artifact and EXCLUDE it, before any scaling beyond reviewed sets.
    """

    def _family(self, tmp_path):
        return _built_family(tmp_path, CLEAN_Q, fill_grounding=True)

    def test_equivalent_flipped_to_not_equivalent(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                axis = atom["semantic_equivalence"].get(
                    "multimodal_vs_unimodal")
                if isinstance(axis, dict) and axis.get("state") == \
                        "equivalent":
                    axis["state"] = "not_equivalent"
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_semantic_eligibility(mutated)
        assert any("NOT equivalent" in e for e in errors)
        check_errors, _ = automatic_family_checks(mutated)
        assert any("semantic eligibility" in e for e in check_errors)

    def test_relevant_flipped_to_irrelevant(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                if atom["structural_role"] == "shared_image":
                    atom["risk_relevance"] = "irrelevant"
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_semantic_eligibility(mutated)
        assert any("annotated irrelevant" in e for e in errors)

    def test_joint_interpretation_flipped_to_false(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                if atom["structural_role"] == "shared_image":
                    atom["required_for_joint_interpretation"] = False
                    return

        mutated = _mutate(family, mutate)
        errors = validate_factorial_semantic_eligibility(mutated)
        assert any("required_for_joint_interpretation=False" in e
                   for e in errors)

    def test_semantic_mutations_cause_stage_exclusion(self, tmp_path):
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                if atom["structural_role"] == "shared_image":
                    atom["risk_relevance"] = "irrelevant"
                    return

        mutated = _mutate(family, mutate)
        write_jsonl(tmp_path / "families.jsonl", [mutated.to_dict()])
        assert run_validation_stage(tmp_path) == []
        excluded = read_jsonl(tmp_path / "excluded_families.jsonl")
        assert len(excluded) == 1
        assert any("semantic eligibility" in r
                   for r in excluded[0]["reasons"])

    def test_negative_control_never_passes_as_eligible(self, tmp_path):
        """A decided-but-negative family is structurally buildable yet
        semantically ineligible: the firewall must reject it."""
        family = self._family(tmp_path)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                axis = atom["semantic_equivalence"].get(
                    "multimodal_vs_unimodal")
                if isinstance(axis, dict):
                    axis["state"] = "not_equivalent"

        mutated = _mutate(family, mutate)
        errors = validate_factorial_semantic_eligibility(mutated)
        # Every gated variant (vision_only, cross_modal, shuffle)
        # reports the negative annotation.
        assert any(e.startswith("vision_only:") for e in errors)
        assert any(e.startswith("cross_modal:") for e in errors)
        assert any(e.startswith("shuffle:") for e in errors)


class TestMutationsCauseExclusion:
    """The stage must EXCLUDE a corrupted artifact, not merely warn."""

    def test_deleted_shuffle_message_excluded(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)

        def mutate(record):
            record["variants"]["shuffle"]["messages"].pop(0)

        mutated = _mutate(family, mutate)
        write_jsonl(tmp_path / "families.jsonl", [mutated.to_dict()])
        validated = run_validation_stage(tmp_path)
        assert validated == []
        excluded = read_jsonl(tmp_path / "excluded_families.jsonl")
        assert len(excluded) == 1
        assert excluded[0]["reasons"]

    def test_corrupted_media_hash_excluded(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)

        def mutate(record):
            for atom in record["semantic_atoms"]:
                for media in atom["source_media"]:
                    media["sha256"] = "0" * 64
                    return

        mutated = _mutate(family, mutate)
        write_jsonl(tmp_path / "families.jsonl", [mutated.to_dict()])
        assert run_validation_stage(tmp_path) == []
        excluded = read_jsonl(tmp_path / "excluded_families.jsonl")
        assert any("content hash differs" in r
                   for r in excluded[0]["reasons"])


class TestFactorialCellsRecorded:
    def test_report_records_explicit_cells(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        write_jsonl(tmp_path / "families.jsonl", [family.to_dict()])
        run_validation_stage(tmp_path)
        report = json.loads(
            (tmp_path / "validation_report.json").read_text())
        entry = report["families"][0]
        assert entry["factorial_cells"] == {
            name: list(cell) for name, cell in FACTORIAL_CELLS.items()
        }
        assert entry["factorial_cells"] == {
            "neutral": [0, 0], "text_only": [1, 0],
            "vision_only": [0, 1], "cross_modal": [1, 1],
        }
