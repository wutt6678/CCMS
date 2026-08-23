"""Data subpackage: schemas, I/O, validation, logging, and media."""

from causal_mllm.data.io import (
    append_jsonl,
    load_config,
    read_jsonl,
    read_jsonl_iter,
    save_config,
    write_jsonl,
)
from causal_mllm.data.logging import get_logger, setup_logging
from causal_mllm.data.media import MediaLoadError
from causal_mllm.data.schemas import (
    ALL_VARIANTS,
    CanonicalSourceExample,
    CausalFamily,
    GeneratorProvenance,
    InferenceOutput,
    Message,
    NormalizationRejection,
    SafetyJudgeLabel,
    SafetyLabel,
    SemanticAtom,
    SourceDataset,
    SourceSetting,
    TerminalQuery,
    VariantData,
    VariantName,
)
from causal_mllm.data.validate_schema import (
    SchemaValidationError,
    validate_causal_family,
    validate_family_strict,
    validate_source_example,
    validate_source_strict,
)

__all__ = [
    "ALL_VARIANTS",
    "CanonicalSourceExample",
    "CausalFamily",
    "GeneratorProvenance",
    "InferenceOutput",
    "MediaLoadError",
    "Message",
    "NormalizationRejection",
    "SafetyJudgeLabel",
    "SafetyLabel",
    "SchemaValidationError",
    "SemanticAtom",
    "SourceDataset",
    "SourceSetting",
    "TerminalQuery",
    "VariantData",
    "VariantName",
    "append_jsonl",
    "get_logger",
    "load_config",
    "read_jsonl",
    "read_jsonl_iter",
    "save_config",
    "setup_logging",
    "validate_causal_family",
    "validate_family_strict",
    "validate_source_example",
    "validate_source_strict",
    "write_jsonl",
]
