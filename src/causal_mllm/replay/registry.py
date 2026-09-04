"""Iteration 11 model registry + revision resolution.

Loads the frozen ``model_registry.yaml`` (written by
``scripts/iter11_freeze_protocol.py``) and an optional
``resolved_models.lock.yaml`` (written at preflight, 11.5) that carries
the immutable revisions resolved for confirmatory runs.

Fail-closed revision policy (spec Section 6/12):
  * preflight / adapter smoke: ``revision`` may be null and is resolved
    at load time (requested vs resolved recorded separately);
  * confirmatory: ``revision`` MUST be an immutable 40-hex SHA —
    null / branch / 'main' / 'latest' are rejected.

This module never loads models; it only resolves declarative specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from causal_mllm.replay.errors import ReplayError

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "outputs" / "iteration_11" / "protocol" / "model_registry.yaml")
DEFAULT_LOCK = (
    Path(__file__).resolve().parents[3]
    / "outputs" / "iteration_11" / "protocol" / "resolved_models.lock.yaml")

IMMUTABLE_REV_HEXLEN = 40
_FLOATING = {"", "main", "master", "head", "latest", "none"}

VALID_ADAPTERS = {"qwen35", "ministral3", "phi4_multimodal", "gemma3"}


@dataclass(frozen=True)
class ResolvedModel:
    """A fully-resolved target-model specification."""

    model_key: str
    model_id: str
    adapter: str
    revision: str | None
    dtype: str = "bfloat16"
    quantization: str = "none"
    trust_remote_code: bool = False
    thinking_mode: bool = False
    role: str = ""
    license: str = ""
    size_metadata: dict = field(default_factory=dict)
    revision_source: str = "registry"  # registry | lock | resolved_at_load

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key, "model_id": self.model_id,
            "adapter": self.adapter, "revision": self.revision,
            "dtype": self.dtype, "quantization": self.quantization,
            "trust_remote_code": self.trust_remote_code,
            "thinking_mode": self.thinking_mode, "role": self.role,
            "license": self.license, "size_metadata": dict(self.size_metadata),
            "revision_source": self.revision_source,
        }


def is_immutable_revision(rev) -> bool:
    """True only for a full 40-hex commit SHA (not a branch/tag/null)."""
    if not isinstance(rev, str):
        return False
    if rev.strip().lower() in _FLOATING:
        return False
    if len(rev) != IMMUTABLE_REV_HEXLEN:
        return False
    return all(c in "0123456789abcdef" for c in rev.lower())


def assert_confirmatory_revision(model_key: str, rev) -> None:
    """Fail-closed: a confirmatory run must pin an immutable SHA."""
    if not is_immutable_revision(rev):
        raise ReplayError(
            f"{model_key}: confirmatory runs require an immutable 40-hex "
            f"revision, got {rev!r}. Floating revisions (null/branch/"
            f"'main'/'latest') are rejected outside preflight; resolve and "
            f"lock the revision first (resolved_models.lock.yaml).")


def load_registry(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_REGISTRY
    if not p.exists():
        raise ReplayError(f"model registry not found: {p}")
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(reg, dict) or "models" not in reg:
        raise ReplayError(f"malformed model registry (no 'models'): {p}")
    return reg


def load_lock(path: str | Path | None = None) -> dict:
    """Optional lock file: {model_key: {revision, processor_revision}}."""
    p = Path(path) if path else DEFAULT_LOCK
    if not p.exists():
        return {}
    lock = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return lock.get("models", lock) if isinstance(lock, dict) else {}


def resolve_model(
    model_key: str,
    *,
    registry: dict | None = None,
    lock: dict | None = None,
    confirmatory: bool = False,
    registry_path: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> ResolvedModel:
    """Resolve one model_key exactly once into a ResolvedModel.

    Args:
        confirmatory: when True, require an immutable revision (from the
            lock or registry) and reject floating values. When False
            (preflight/adapter smoke), a null revision is allowed and
            resolved at load time.
    """
    reg = registry if registry is not None else load_registry(registry_path)
    models = reg["models"]
    if model_key not in models:
        raise ReplayError(
            f"unknown model_key {model_key!r}; registry has "
            f"{sorted(models)}")
    m = models[model_key]
    adapter = m.get("adapter")
    if adapter not in VALID_ADAPTERS:
        raise ReplayError(
            f"{model_key}: adapter {adapter!r} not in {sorted(VALID_ADAPTERS)}")

    lk = (lock if lock is not None else load_lock(lock_path)) or {}
    revision = m.get("revision")
    revision_source = "registry"
    if model_key in lk and isinstance(lk[model_key], dict):
        locked_rev = lk[model_key].get("revision")
        if locked_rev is not None:
            revision = locked_rev
            revision_source = "lock"

    if confirmatory:
        assert_confirmatory_revision(model_key, revision)
    elif revision is not None and not is_immutable_revision(revision):
        # A non-null but floating value (branch/tag) is never acceptable,
        # even in preflight: either null (resolve at load) or an immutable
        # SHA.
        raise ReplayError(
            f"{model_key}: revision {revision!r} is floating; use null "
            f"(resolve at load) or an immutable 40-hex SHA")

    return ResolvedModel(
        model_key=model_key,
        model_id=m["model_id"],
        adapter=adapter,
        revision=revision,
        dtype=m.get("dtype", "bfloat16"),
        quantization=m.get("quantization", "none"),
        trust_remote_code=bool(m.get("trust_remote_code", False)),
        thinking_mode=bool(m.get("thinking_mode", False)),
        role=m.get("role", ""),
        license=m.get("license", ""),
        size_metadata=dict(m.get("size_metadata", {})),
        revision_source=revision_source,
    )


def resolve_all(*, confirmatory: bool = False,
                registry_path: str | Path | None = None,
                lock_path: str | Path | None = None) -> dict[str, ResolvedModel]:
    reg = load_registry(registry_path)
    return {
        key: resolve_model(key, registry=reg, confirmatory=confirmatory,
                           lock_path=lock_path)
        for key in reg["models"]
    }
