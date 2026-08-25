"""Frozen replay runner (Iteration 8).

Trajectory -> raw model response with full provenance. No judging:
Delta_T / Delta_V / Delta_TV and the safety judge belong to
Iteration 9.
"""

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
from causal_mllm.replay.errors import (
    ReplayError,
    ReplayGenerationError,
    ReplayMediaError,
    classify_error,
)
from causal_mllm.replay.runner import (
    build_chat_messages,
    run_replay_stage,
    verify_family_media,
)

__all__ = [
    "CallableBackend",
    "DEFAULT_SYSTEM_PROMPT",
    "HFLocalBackend",
    "PROMPT_TEMPLATE_REVISION",
    "ReplayBackend",
    "ReplayConfig",
    "ReplayError",
    "ReplayGenerationError",
    "ReplayMediaError",
    "build_chat_messages",
    "classify_error",
    "run_replay_stage",
    "verify_family_media",
]
