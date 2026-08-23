"""Typed dataset schemas for canonical source examples and causal families."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceDataset(str, Enum):
    MTMCS = "mtmcs"
    COSAFE = "cosafe"
    MTID = "mtid"


class SafetyLabel(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class SourceSetting(str, Enum):
    ESCALATION = "escalation"
    CONTEXT_SWITCH = "context_switch"
    COREFERENCE = "coreference"
    OTHER = "other"


class VariantName(str, Enum):
    NEUTRAL = "neutral"
    TEXT_ONLY = "text_only"
    VISION_ONLY = "vision_only"
    CROSS_MODAL = "cross_modal"
    SHUFFLE = "shuffle"
    HISTORY_RESET = "history_reset"


ALL_VARIANTS = list(VariantName)


class AtomType(str, Enum):
    ENTITY_OR_SCENE = "entity_or_scene"
    ATTRIBUTE_OR_STATE = "attribute_or_state"
    RELATION = "relation"
    INTENT = "intent"
    CONSTRAINT = "constraint"
    REFERENCE = "reference"
    CONTEXTUAL_DISAMBIGUATOR = "contextual_disambiguator"


class GeneratorType(str, Enum):
    RULE = "rule"
    LLM = "llm"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Canonical source example
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversational turn message."""
    turn_index: int
    role: str  # "user" | "assistant" | "system"
    text: Optional[str] = None
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "text": self.text,
            "images": list(self.images),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(
            turn_index=d["turn_index"],
            role=d["role"],
            text=d.get("text"),
            images=d.get("images", []),
        )


@dataclass
class CanonicalSourceExample:
    """Normalized representation of one source example across all datasets."""
    source_dataset: str
    source_id: str
    source_split: Optional[str]
    source_category: Optional[str]
    source_setting: str  # escalation | context_switch | other
    label: str  # safe | unsafe | unknown
    messages: list[Message]
    terminal_turn_index: int
    terminal_query: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_dataset": self.source_dataset,
            "source_id": self.source_id,
            "source_split": self.source_split,
            "source_category": self.source_category,
            "source_setting": self.source_setting,
            "label": self.label,
            "messages": [m.to_dict() for m in self.messages],
            "terminal_turn_index": self.terminal_turn_index,
            "terminal_query": self.terminal_query,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CanonicalSourceExample:
        return cls(
            source_dataset=d["source_dataset"],
            source_id=d["source_id"],
            source_split=d.get("source_split"),
            source_category=d.get("source_category"),
            source_setting=d.get("source_setting", "other"),
            label=d["label"],
            messages=[Message.from_dict(m) for m in d["messages"]],
            terminal_turn_index=d["terminal_turn_index"],
            terminal_query=d["terminal_query"],
            metadata=d.get("metadata", {}),
        )

    @property
    def num_turns(self) -> int:
        return len(self.messages)

    @property
    def has_images(self) -> bool:
        return any(len(m.images) > 0 for m in self.messages)

    @property
    def image_count(self) -> int:
        return sum(len(m.images) for m in self.messages)


# ---------------------------------------------------------------------------
# Causal family schemas
# ---------------------------------------------------------------------------

@dataclass
class SemanticAtom:
    """An abstract semantic contribution from one or more turns."""
    atom_id: str
    description: str
    source_turns: list[int]
    source_modalities: list[str]  # "text" | "vision"
    atom_type: str = "entity_or_scene"  # AtomType value

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "description": self.description,
            "source_turns": list(self.source_turns),
            "source_modalities": list(self.source_modalities),
            "type": self.atom_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SemanticAtom:
        return cls(
            atom_id=d["atom_id"],
            description=d["description"],
            source_turns=d["source_turns"],
            source_modalities=d["source_modalities"],
            atom_type=d.get("type", "entity_or_scene"),
        )


@dataclass
class GeneratorProvenance:
    """Provenance for a generated or transformed field."""
    type: str  # GeneratorType value
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    seed: Optional[int] = None
    parent_variant: Optional[str] = None
    transformations: list[str] = field(default_factory=list)
    creation_timestamp: Optional[str] = None
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    source_revision: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "generator": {
                "type": self.type,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "seed": self.seed,
            },
            "parent_variant": self.parent_variant,
            "transformations": list(self.transformations),
            "creation_timestamp": self.creation_timestamp,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "source_revision": self.source_revision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneratorProvenance:
        gen = d.get("generator", {})
        return cls(
            type=gen.get("type", "rule"),
            model=gen.get("model"),
            prompt_version=gen.get("prompt_version"),
            seed=gen.get("seed"),
            parent_variant=d.get("parent_variant"),
            transformations=d.get("transformations", []),
            creation_timestamp=d.get("creation_timestamp"),
            git_commit=d.get("git_commit"),
            config_hash=d.get("config_hash"),
            source_revision=d.get("source_revision"),
        )


@dataclass
class VariantData:
    """One variant within a causal family."""
    name: str  # VariantName value
    messages: list[Message]
    provenance: GeneratorProvenance
    shuffle_permutation: Optional[list[int]] = None  # only for shuffle variant

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "name": self.name,
            "messages": [m.to_dict() for m in self.messages],
            "provenance": self.provenance.to_dict(),
        }
        if self.shuffle_permutation is not None:
            result["shuffle_permutation"] = self.shuffle_permutation
        return result

    @classmethod
    def from_dict(cls, d: dict) -> VariantData:
        return cls(
            name=d["name"],
            messages=[Message.from_dict(m) for m in d["messages"]],
            provenance=GeneratorProvenance.from_dict(d.get("provenance", {})),
            shuffle_permutation=d.get("shuffle_permutation"),
        )


@dataclass
class TerminalQuery:
    """Terminal query with invariant enforcement."""
    text: str
    sha256: str
    invariant_required: bool = True

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sha256": self.sha256,
            "invariant_required": self.invariant_required,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TerminalQuery:
        return cls(
            text=d["text"],
            sha256=d["sha256"],
            invariant_required=d.get("invariant_required", True),
        )

    @classmethod
    def create(cls, text: str) -> TerminalQuery:
        from causal_mllm.seeds import sha256_text
        return cls(text=text, sha256=sha256_text(text), invariant_required=True)


@dataclass
class CausalFamily:
    """The atomic research unit: one source example + six controlled variants."""
    family_id: str
    source: dict  # {"dataset": "...", "source_id": "..."}
    category: Optional[str]
    setting: str
    terminal_query: TerminalQuery
    semantic_atoms: list[SemanticAtom]
    ground_truth: dict[str, Any] = field(default_factory=dict)
    variants: dict[str, VariantData] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "source": self.source,
            "category": self.category,
            "setting": self.setting,
            "terminal_query": self.terminal_query.to_dict(),
            "semantic_atoms": [a.to_dict() for a in self.semantic_atoms],
            "ground_truth": self.ground_truth,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "validation": self.validation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CausalFamily:
        return cls(
            family_id=d["family_id"],
            source=d["source"],
            category=d.get("category"),
            setting=d.get("setting", "other"),
            terminal_query=TerminalQuery.from_dict(d["terminal_query"]),
            semantic_atoms=[SemanticAtom.from_dict(a) for a in d.get("semantic_atoms", [])],
            ground_truth=d.get("ground_truth", {}),
            variants={k: VariantData.from_dict(v) for k, v in d.get("variants", {}).items()},
            validation=d.get("validation", {}),
        )


# ---------------------------------------------------------------------------
# Inference output schema
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """Stored model response, kept separate from the immutable dataset."""
    run_id: str
    family_id: str
    variant: str
    model: str
    generation_config: dict[str, Any]
    response: str
    latency: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # oom | media_load | context_overflow | other

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "family_id": self.family_id,
            "variant": self.variant,
            "model": self.model,
            "generation_config": self.generation_config,
            "response": self.response,
            "latency": self.latency,
            "error": self.error,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InferenceOutput:
        return cls(
            run_id=d["run_id"],
            family_id=d["family_id"],
            variant=d["variant"],
            model=d["model"],
            generation_config=d.get("generation_config", {}),
            response=d.get("response", ""),
            latency=d.get("latency"),
            error=d.get("error"),
            error_type=d.get("error_type"),
        )


# ---------------------------------------------------------------------------
# Evaluation label schema
# ---------------------------------------------------------------------------

@dataclass
class SafetyJudgeLabel:
    """Structured output from the safety judge."""
    safety_label: str  # safe | borderline | unsafe
    compliance_level: int
    refusal: bool
    helpfulness_on_benign: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "safety_label": self.safety_label,
            "compliance_level": self.compliance_level,
            "refusal": self.refusal,
            "helpfulness_on_benign": self.helpfulness_on_benign,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SafetyJudgeLabel:
        return cls(
            safety_label=d["safety_label"],
            compliance_level=d["compliance_level"],
            refusal=d["refusal"],
            helpfulness_on_benign=d.get("helpfulness_on_benign"),
            confidence=d.get("confidence", 0.0),
        )


# ---------------------------------------------------------------------------
# Normalization rejection tracking
# ---------------------------------------------------------------------------

@dataclass
class NormalizationRejection:
    """Record of a rejected source row during normalization.

    No row should ever disappear from the dataset without an
    explicit rejection record.
    """
    source_id: str
    stage: str  # "normalization" | "media" | "schema"
    error_type: str  # e.g. "MediaLoadError", "ValueError", "KeyError"
    reason: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "stage": self.stage,
            "error_type": self.error_type,
            "reason": self.reason,
        }
