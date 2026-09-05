"""Frozen replay runner (Iteration 8) + cross-model targets (Iteration 11).

Trajectory -> raw model response with full provenance. No judging:
Delta_T / Delta_V / Delta_TV and the safety judge belong to
Iteration 9.

Iteration 11 adds a model registry (``resolve_model``) and thin
model-family adapters (``build_adapter``) that plug into the SAME
``run_replay_stage`` pipeline.  The legacy single-model path
(``model_spec=None``) is unchanged.

``confirmatory`` holds the gate that enforces the frozen Iteration 11
protocol on a run, and ``selection`` holds the pre-registered 11.5
eligibility selection the gate re-derives.  They are separate modules so
that neither the gate nor the 11.5 producer can define the selection in
terms of the other.
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
    ELIGIBILITY_GENERATIONS_ROOT,
    ELIGIBILITY_REQUIRED_GATES,
    eligibility_report_path,
    enforce_confirmatory_protocol,
    enforce_eligibility_protocol,
    load_eligibility_report,
    protocol_sha256,
    validate_eligibility_report,
    validate_gate_entry,
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
    editable_installs,
    editable_vcs_revisions,
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
from causal_mllm.replay.selection import (
    SELECTION_ARTIFACT,
    derive_frozen_selection,
    select_eligibility_families,
    selected_families_sha256,
)

__all__ = [
    "CallableBackend",
    "DEFAULT_SYSTEM_PROMPT",
    "ELIGIBILITY_GENERATIONS_ROOT",
    "ELIGIBILITY_REQUIRED_GATES",
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
    "SELECTION_ARTIFACT",
    "TargetModelAdapter",
    "append_journal",
    "assert_confirmatory_revision",
    "build_adapter",
    "build_chat_messages",
    "classify_error",
    "dependency_lock_sha256",
    "dependency_lock_snapshot",
    "derive_frozen_selection",
    "editable_installs",
    "editable_vcs_revisions",
    "eligibility_report_path",
    "enforce_confirmatory_protocol",
    "enforce_eligibility_protocol",
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
    "select_eligibility_families",
    "selected_families_sha256",
    "update_lock",
    "validate_eligibility_report",
    "validate_gate_entry",
    "validate_journal",
    "verify_active_dependency_lock",
    "verify_family_media",
]
