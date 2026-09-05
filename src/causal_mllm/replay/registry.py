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

import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from causal_mllm.replay.errors import ReplayError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "model_registry.yaml"
# The lock is a PREFLIGHT OUTPUT, not a frozen artifact, so it lives with
# the other preflight evidence rather than in the immutable protocol dir.
DEFAULT_LOCK = REPO_ROOT / "outputs" / "iteration_11" / "preflight" \
    / "resolved_models.lock.yaml"

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


# ---------------------------------------------------------------------
# Lock file: preflight output binding resolved revisions + dependencies
# ---------------------------------------------------------------------
def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


#: This project's own distribution, excluded from the dependency-lock hash.
#:
#: ``pip freeze`` reports an editable self-install in one of two ways
#: depending on how the process was invoked — either
#: ``-e git+<url>@<revision>#egg=causal_mllm`` or ``causal-mllm==0.1.0`` —
#: and the first form embeds the repository's LIVE git HEAD. The same
#: working tree therefore hashed to four different values across four
#: invocations, and the value moved on every commit. Since
#: ``iteration11_run_fingerprint`` binds this hash and uses it to decide
#: whether a run may be resumed, an unstable value would reject a
#: legitimate resume and makes the recorded lock useless as evidence.
#:
#: Nothing is lost by excluding it: the code identity is already bound
#: separately and more precisely via ``code_commit`` / ``git_dirty``. The
#: exclusion is still reported — by distribution NAME, deliberately, since
#: recording the raw editable line would embed the live git HEAD inside the
#: hashed lock block and reintroduce the same instability one level up.
#: Third-party editable installs (e.g. the MIDP prior-art package) are KEPT —
#: a change there is a real dependency change.
SELF_DISTRIBUTIONS = ("causal_mllm", "causal-mllm")


def _self_distribution_name(line: str) -> str | None:
    """Which of :data:`SELF_DISTRIBUTIONS` a freeze line refers to, if any."""
    lowered = line.lower()
    for name in SELF_DISTRIBUTIONS:
        if f"#egg={name}" in lowered:
            return name
        if lowered.startswith(f"{name}==") or lowered.startswith(f"{name} @"):
            return name
    return None


def _is_self_distribution_line(line: str) -> bool:
    return _self_distribution_name(line) is not None


def dependency_lock_snapshot() -> dict:
    """Hashed snapshot of the reference environment.

    The frozen protocol requires a complete pip-freeze lock hash to be
    captured at preflight and bound into each resolved run fingerprint.
    The project's own editable install is excluded (see
    :data:`SELF_DISTRIBUTIONS`) so the hash depends only on third-party
    packages and is reproducible across invocations of the same tree.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False)
    all_lines = sorted(line.strip() for line in completed.stdout.splitlines()
                       if line.strip() and not line.startswith("#"))
    excluded = sorted({name for name in
                       (_self_distribution_name(line) for line in all_lines)
                       if name})
    lines = [line for line in all_lines if not _is_self_distribution_line(line)]
    freeze_text = "\n".join(lines)
    pyproject = REPO_ROOT / "pyproject.toml"
    return {
        "pip_freeze_sha256": hashlib.sha256(
            freeze_text.encode("utf-8")).hexdigest(),
        "n_packages": len(lines),
        "excluded_self_distributions": excluded,
        "pyproject_sha256": _file_sha256(pyproject),
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
    }


def dependency_lock_sha256(lock_path: str | Path | None = None) -> str | None:
    """Hash of the recorded dependency lock (None if not yet captured)."""
    path = Path(lock_path) if lock_path else DEFAULT_LOCK
    if not path.exists():
        return None
    try:
        lock = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    dependency = lock.get("dependency_lock")
    if not isinstance(dependency, dict):
        return None
    blob = yaml.safe_dump(dependency, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def update_lock(model_key: str, *, revision: str,
                processor_revision: str | None = None,
                evidence: str | None = None,
                resolved_at: str | None = None,
                measured_size: dict | None = None,
                dependency_lock: dict | None = None,
                allow_change: bool = False,
                lock_path: str | Path | None = None) -> Path:
    """Record a resolved immutable revision for one model_key.

    Fail-closed: once a revision is locked, re-locking to a DIFFERENT
    revision raises unless ``allow_change`` is explicit, so an upstream
    commit can never silently move a confirmatory target.
    """
    if not is_immutable_revision(revision):
        raise ReplayError(
            f"{model_key}: refusing to lock non-immutable revision "
            f"{revision!r}")
    path = Path(lock_path) if lock_path else DEFAULT_LOCK
    lock: dict = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ReplayError(f"{path}: malformed lock file")
        lock = loaded
    models = lock.setdefault("models", {})
    entry = models.get(model_key)
    if not isinstance(entry, dict):
        entry = {}
    prior = entry.get("revision")
    if prior and prior != revision and not allow_change:
        raise ReplayError(
            f"{model_key}: revision already locked to {prior!r}; refusing "
            f"to move it to {revision!r} implicitly. Pass allow_change "
            f"(--force-lock) to re-pin deliberately.")
    if prior and prior != revision:
        history = entry.setdefault("superseded_revisions", [])
        history.append(prior)
    entry["revision"] = revision
    entry["processor_revision"] = processor_revision or revision
    if evidence:
        entry["evidence"] = evidence
    if resolved_at:
        entry["resolved_at"] = resolved_at
    if measured_size:
        # Measured from the checkpoint headers; supersedes the registry's
        # declared approximations for analysis.
        entry["measured_size"] = measured_size
    entry.setdefault("resolved_by", "scripts/iter11_model_preflight.py")
    models[model_key] = entry
    if dependency_lock:
        lock["dependency_lock"] = dependency_lock
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Iteration 11 preflight output (NOT a frozen artifact).\n"
        "# Immutable revisions resolved for confirmatory runs, plus the\n"
        "# hashed dependency lock bound into every resolved_run_fingerprint.\n"
        + yaml.safe_dump(lock, sort_keys=True),
        encoding="utf-8")
    return path

