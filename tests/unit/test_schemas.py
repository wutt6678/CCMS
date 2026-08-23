"""Unit tests for schema validation."""

import pytest

from causal_mllm.data.schemas import (
    CanonicalSourceExample,
    CausalFamily,
    Message,
    SemanticAtom,
    TerminalQuery,
    VariantName,
)
from causal_mllm.data.validate_schema import (
    SchemaValidationError,
    validate_causal_family,
    validate_family_strict,
    validate_source_example,
    validate_source_strict,
)


class TestSourceExampleValidation:
    """Source example validation must catch errors correctly."""

    def test_valid_example(self, synthetic_source_dict):
        errors = validate_source_example(synthetic_source_dict)
        assert errors == []

    def test_missing_source_dataset(self, synthetic_source_dict):
        del synthetic_source_dict["source_dataset"]
        errors = validate_source_example(synthetic_source_dict)
        assert any("source_dataset" in e for e in errors)

    def test_missing_messages(self, synthetic_source_dict):
        del synthetic_source_dict["messages"]
        errors = validate_source_example(synthetic_source_dict)
        assert any("messages" in e for e in errors)

    def test_empty_messages(self, synthetic_source_dict):
        synthetic_source_dict["messages"] = []
        errors = validate_source_example(synthetic_source_dict)
        assert any("empty" in e for e in errors)

    def test_invalid_role(self, synthetic_source_dict):
        synthetic_source_dict["messages"][0]["role"] = "bot"
        errors = validate_source_example(synthetic_source_dict)
        assert any("role" in e for e in errors)

    def test_terminal_turn_index_mismatch(self, synthetic_source_dict):
        synthetic_source_dict["terminal_turn_index"] = 99
        errors = validate_source_example(synthetic_source_dict)
        assert any("terminal_turn_index" in e for e in errors)

    def test_validate_strict_valid(self, synthetic_source_dict):
        example = validate_source_strict(synthetic_source_dict)
        assert isinstance(example, CanonicalSourceExample)
        assert example.source_id == "synth_001"

    def test_validate_strict_raises(self, synthetic_source_dict):
        del synthetic_source_dict["terminal_query"]
        with pytest.raises(SchemaValidationError):
            validate_source_strict(synthetic_source_dict)


class TestCausalFamilyValidation:
    """Causal family validation must catch structural errors."""

    def test_valid_family(self, synthetic_family_dict):
        errors = validate_causal_family(synthetic_family_dict)
        assert errors == []

    def test_missing_family_id(self, synthetic_family_dict):
        del synthetic_family_dict["family_id"]
        errors = validate_causal_family(synthetic_family_dict)
        assert any("family_id" in e for e in errors)

    def test_missing_variants(self, synthetic_family_dict):
        del synthetic_family_dict["variants"]["neutral"]
        errors = validate_causal_family(synthetic_family_dict)
        assert any("neutral" in e for e in errors)

    def test_duplicate_atom_ids(self, synthetic_family_dict):
        synthetic_family_dict["semantic_atoms"][1]["atom_id"] = "A"
        errors = validate_causal_family(synthetic_family_dict)
        assert any("Duplicate" in e for e in errors)

    def test_validate_family_strict(self, synthetic_family_dict):
        family = validate_family_strict(synthetic_family_dict)
        assert isinstance(family, CausalFamily)
        assert family.family_id == "CMST_999999"


class TestSchemaClasses:
    """Data class serialization roundtrips."""

    def test_message_roundtrip(self):
        msg = Message(turn_index=0, role="user", text="hello", images=["img.jpg"])
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.turn_index == msg.turn_index
        assert msg2.text == msg.text
        assert msg2.images == msg.images

    def test_terminal_query_create(self):
        tq = TerminalQuery.create("What is this?")
        assert tq.text == "What is this?"
        assert len(tq.sha256) == 64
        assert tq.invariant_required is True

    def test_semantic_atom_roundtrip(self):
        atom = SemanticAtom(
            atom_id="A",
            description="test atom",
            source_turns=[0, 1],
            source_modalities=["vision"],
            atom_type="entity_or_scene",
        )
        d = atom.to_dict()
        atom2 = SemanticAtom.from_dict(d)
        assert atom2.atom_id == "A"
        assert atom2.source_turns == [0, 1]

    def test_canonical_example_properties(self, synthetic_canonical):
        ex = synthetic_canonical
        assert ex.num_turns == 5
        assert ex.has_images is True
        assert ex.image_count == 1

    def test_variant_name_values(self):
        names = [v.value for v in VariantName]
        assert "neutral" in names
        assert "cross_modal" in names
        assert "history_reset" in names
        assert len(names) == 6
