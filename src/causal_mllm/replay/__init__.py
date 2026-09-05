"""Frozen replay runner (Iteration 8) + cross-model targets (Iteration 11).

Trajectory -> raw model response with full provenance. No judging:
Delta_T / Delta_V / Delta_TV and the safety judge belong to
Iteration 9.

Iteration 11 adds a model registry (``resolve_model``) and thin
model-family adapters (``build_adapter``) that plug into the SAME
``run_replay_stage`` pipeline.  The legacy single-model path
(``model_spec=None``) is unchanged.
"""

from causal_mllm.replay.adapters import (
    HFAdapterBase,
    Qwen35Adapter,
    TargetModelAdapter,
    build_adapter,
)
from causal_mllm.replay.backend import (
    CallableBackend,
    HFLocalBackend,
    ReplayBackend,
)
from causal_mllm.replay.config import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_TEMPLATE_REVISION,
    ReplayConfig,
)
from causal_mllm.replay.confirmatory import (
    eligibility_report_path,
    enforce_confirmatory_protocol,
    load_eligibility_report,
    protocol_sha256,
)
from causal_mllm.replay.errors import (
    ReplayError,
    ReplayGenerationError,
    ReplayMediaError,
    classify_error,
)
from causal_mllm.replay.registry import (
    ResolvedModel,
    assert_confirmatory_revision,
    dependency_lock_sha256,
    dependency_lock_snapshot,
    is_immutable_revision,
    load_dependency_lock,
    load_lock,
    load_registry,
    resolve_all,
    resolve_model,
    update_lock,
    verify_active_dependency_lock,
)
from causal_mllm.replay.runner import (
    append_journal,
    build_chat_messages,
    fingerprint_hardware,
    iteration11_run_fingerprint,
    resolved_fingerprint,
    run_replay_stage,
    validate_journal,
    verify_family_media,
)

__all__ = [
    "CallableBackend",
    "DEFAULT_SYSTEM_PROMPT",
    "HFAdapterBase",
    "HFLocalBackend",
    "PROMPT_TEMPLATE_REVISION",
    "Qwen35Adapter",
    "ReplayBackend",
    "ReplayConfig",
    "ReplayError",
    "ReplayGenerationError",
    "ReplayMediaError",
    "ResolvedModel",
    "TargetModelAdapter",
    "append_journal",
    "assert_confirmatory_revision",
    "build_adapter",
    "build_chat_messages",
    "classify_error",
    "dependency_lock_sha256",
    "dependency_lock_snapshot",
    "eligibility_report_path",
    "enforce_confirmatory_protocol",
    "fingerprint_hardware",
    "is_immutable_revision",
    "iteration11_run_fingerprint",
    "load_dependency_lock",
    "load_eligibility_report",
    "load_lock",
    "load_registry",
    "protocol_sha256",
    "resolve_all",
    "resolve_model",
    "resolved_fingerprint",
    "run_replay_stage",
    "update_lock",
    "validate_journal",
    "verify_active_dependency_lock",
    "verify_family_media",
]
