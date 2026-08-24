"""Unit tests for semantic annotation (P0-1 equivalence, P0-2 risk relevance).

Annotation fills MEANING on already-extracted structural atoms. It must:
  * keep atom_type as the exact alias of semantic_type
  * normalize equivalence axes (missing axes -> pending)
  * fail loudly on invalid payloads
  * never mutate the input skeleton
"""

from __future__ import annotations

import json

import pytest

from causal_mllm.construction.annotation import (
    AnnotationError,
    CallableAnnotator,
    ManualFileAnnotator,
    apply_annotations,
)
from causal_mllm.construction.families import build_family_skeleton
from tests.unit.factories import make_mtmcs_group


def _skeleton(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fake-image-bytes")
    group = make_mtmcs_group("type_b", 1, image_path=str(img))
    return build_family_skeleton(group, seed=42)


def _causal_atom_id(skeleton) -> str:
    return next(a.atom_id for a in skeleton.semantic_atoms
                if a.divergence == "causal")


def _vision_atom_id(skeleton) -> str:
    return next(a.atom_id for a in skeleton.semantic_atoms
                if a.structural_role == "shared_image")


class TestManualFileAnnotator:
    def test_applies_semantic_annotation_with_alias_sync(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        aid = _causal_atom_id(skeleton)
        annotations = {
            "mtmcs:type_b:000001": {
                aid: {
                    "semantic_type": "relation",
                    "semantic_description":
                        "safe/unsafe framing of the opening request",
                    "semantic_equivalence": {
                        "multimodal_vs_unimodal":
                            {"state": "equivalent", "confidence": 0.94},
                    },
                },
            },
        }
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations))

        annotated = apply_annotations(
            skeleton, ManualFileAnnotator(ann_path))
        atom = next(a for a in annotated.semantic_atoms if a.atom_id == aid)
        assert atom.semantic_type == "relation"
        assert atom.atom_type == "relation"  # exact alias, always in sync
        assert atom.semantic_description.startswith("safe/unsafe framing")
        assert atom.semantic_validation == "human"
        # Equivalence normalized: provided axis + pending fallback
        assert atom.semantic_equivalence["multimodal_vs_unimodal"] == {
            "state": "equivalent", "confidence": 0.94,
        }
        assert atom.semantic_equivalence["safe_vs_unsafe_shared_parts"] == {
            "state": "pending",
        }

    def test_unannotated_atoms_stay_pending(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text("{}")
        annotated = apply_annotations(
            skeleton, ManualFileAnnotator(ann_path))
        assert all(a.semantic_type == "unknown"
                   for a in annotated.semantic_atoms)
        assert all(a.semantic_validation == "pending"
                   for a in annotated.semantic_atoms)

    def test_manual_annotation_records_human_backend(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        aid = _causal_atom_id(skeleton)
        annotations = {"mtmcs:type_b:000001": {aid: {
            "semantic_type": "constraint",
            "semantic_description": "explicit constraint",
        }}}
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations))
        annotated = apply_annotations(
            skeleton, ManualFileAnnotator(ann_path))
        atom = next(a for a in annotated.semantic_atoms if a.atom_id == aid)
        assert atom.annotation_provenance == {"backend": "human"}

    def test_input_skeleton_not_mutated(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        aid = _causal_atom_id(skeleton)
        before = skeleton.to_dict()
        annotations = {"mtmcs:type_b:000001": {aid: {
            "semantic_type": "intent",
            "semantic_description": "x",
        }}}
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations))
        apply_annotations(skeleton, ManualFileAnnotator(ann_path))
        assert skeleton.to_dict() == before

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ManualFileAnnotator(tmp_path / "nope.json")


class TestCallableAnnotator:
    def test_llm_backend_records_validation_label(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        vid = _vision_atom_id(skeleton)

        def fake_llm(family_key: str, atom_dict: dict):
            if atom_dict["atom_id"] == vid:
                return {
                    "semantic_type": "entity_or_scene",
                    "semantic_description": "batter in the photo",
                    "risk_relevance": "relevant",
                    "required_for_joint_interpretation": True,
                }
            return None

        annotated = apply_annotations(
            skeleton, CallableAnnotator(fake_llm, model_name="fake-vlm"))
        vision = next(a for a in annotated.semantic_atoms if a.atom_id == vid)
        assert vision.semantic_type == "entity_or_scene"
        assert vision.atom_type == "entity_or_scene"
        assert vision.semantic_validation == "llm"
        # P0-2: image supplies information required for the risky reading
        assert vision.risk_relevance == "relevant"
        assert vision.required_for_joint_interpretation is True
        # Provenance must identify the exact producing pipeline
        prov = vision.annotation_provenance
        assert prov["backend"] == "llm"
        assert prov["model"] == "fake-vlm"
        assert "prompt_version" in prov and "temperature" in prov
        # Non-vision atoms untouched
        causal = next(a for a in annotated.semantic_atoms
                      if a.divergence == "causal")
        assert causal.semantic_validation == "pending"
        assert causal.annotation_provenance is None

    def test_callable_annotator_requires_model_name(self):
        with pytest.raises(AnnotationError, match="model_name"):
            CallableAnnotator(lambda fk, ad: None, model_name="")


class TestAnnotationValidation:
    def _apply_payload(self, tmp_path, payload):
        skeleton = _skeleton(tmp_path)
        aid = _causal_atom_id(skeleton)
        fn = lambda fk, ad: payload if ad["atom_id"] == aid else None  # noqa: E731
        return apply_annotations(
            skeleton, CallableAnnotator(fn, model_name="test-model"))

    def test_rejects_unknown_semantic_type(self, tmp_path):
        with pytest.raises(AnnotationError, match="semantic_type"):
            self._apply_payload(tmp_path, {"semantic_type": "vibes"})

    def test_rejects_pending_validation_state(self, tmp_path):
        with pytest.raises(AnnotationError, match="semantic_validation"):
            self._apply_payload(tmp_path, {
                "semantic_type": "intent",
                "semantic_validation": "pending",
            })

    def test_rejects_llm_payload_without_provenance(self, tmp_path):
        """'An LLM did it' is not provenance."""
        skeleton = _skeleton(tmp_path)
        aid = _causal_atom_id(skeleton)
        # Payload explicitly claims llm validation but strips provenance
        fn = lambda fk, ad: ({  # noqa: E731
            "semantic_type": "intent",
            "semantic_validation": "llm",
            "annotation_provenance": {"backend": "llm"},
        } if ad["atom_id"] == aid else None)
        with pytest.raises(AnnotationError, match="annotation_provenance"):
            apply_annotations(
                skeleton,
                CallableAnnotator(fn, model_name="test-model"))

    def test_rejects_bad_equivalence_state(self, tmp_path):
        with pytest.raises(AnnotationError, match="Equivalence state"):
            self._apply_payload(tmp_path, {
                "semantic_type": "intent",
                "semantic_equivalence": {"multimodal_vs_unimodal": "same_ish"},
            })

    def test_rejects_bad_confidence(self, tmp_path):
        with pytest.raises(AnnotationError, match="confidence"):
            self._apply_payload(tmp_path, {
                "semantic_type": "intent",
                "semantic_equivalence": {
                    "multimodal_vs_unimodal":
                        {"state": "equivalent", "confidence": 2.0},
                },
            })

    def test_rejects_bad_risk_relevance(self, tmp_path):
        skeleton = _skeleton(tmp_path)
        vid = _vision_atom_id(skeleton)
        fn = lambda fk, ad: (  # noqa: E731
            {"semantic_type": "entity_or_scene", "risk_relevance": "maybe"}
            if ad["atom_id"] == vid else None
        )
        with pytest.raises(AnnotationError, match="risk_relevance"):
            apply_annotations(
                skeleton, CallableAnnotator(fn, model_name="test-model"))
