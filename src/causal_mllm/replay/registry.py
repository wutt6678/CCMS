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
import re
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


#: The revision of an editable VCS install, matched as the LAST ``@``-
#: separated hex segment before any ``#egg=``/``#subdirectory=`` fragment.
#: The URL itself may contain ``@`` (``ssh://git@host``), so the match is
#: anchored on a hex revision followed by end-of-line or a fragment.
_EDITABLE_VCS_REVISION = re.compile(
    r"^(?P<head>.*@)(?P<rev>[0-9a-fA-F]{7,40})(?P<tail>(#.*)?)$")

#: ``#egg=<name>`` of an editable VCS line, used to key the recorded
#: revisions by something readable.
_EGG_NAME = re.compile(r"#egg=([^&\s]+)")


def _editable_vcs_name(line: str) -> str:
    """A readable key for an editable VCS freeze line."""
    match = _EGG_NAME.search(line)
    if match:
        return match.group(1)
    return line.split()[0] if line.split() else line


def editable_vcs_revisions(lines: list[str]) -> dict:
    """``{distribution: revision}`` for editable VCS installs in ``lines``.

    DETECTION only — the revision stays in the hashed freeze text. An
    editable install's revision is part of dependency identity: normalizing
    it away would let the source of a dependency change while the certified
    lock hash stood still, which is the opposite of what a reproducible
    lock is for.

    The real protection is that a third-party editable install is not
    certifiable AT ALL (see :func:`verify_active_dependency_lock`), because
    ``pip freeze`` reports such an install by the sibling repository's
    committed HEAD and is blind to its uncommitted working-tree changes. No
    hash of freeze output can capture those, so the only sound answer is to
    refuse to run confirmatory work in an environment that has one.
    """
    revisions: dict = {}
    for line in lines:
        match = _EDITABLE_VCS_REVISION.match(line)
        if match:
            revisions[_editable_vcs_name(line)] = match.group("rev")
    return revisions


def dependency_lock_snapshot() -> dict:
    """Hashed snapshot of the reference environment.

    The frozen protocol requires a complete pip-freeze lock hash to be
    captured at preflight and bound into each resolved run fingerprint.
    Only this project's own editable install is excluded (see
    :data:`SELF_DISTRIBUTIONS`), because ``pip freeze`` renders it in one of
    two forms depending on invocation and one of them embeds THIS
    repository's live HEAD — which ``code_commit`` already binds more
    precisely. Third-party editable installs are kept verbatim, revision
    included, and are additionally reported so their presence can be
    refused (see :func:`editable_vcs_revisions`).

    Fail-closed: a non-zero ``pip freeze`` exit is an error, not an empty
    snapshot. A partial or empty freeze still hashes to a STABLE value, so
    it would silently certify an environment that was never observed.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ReplayError(
            f"`{sys.executable} -m pip freeze` exited "
            f"{completed.returncode}; refusing to record a dependency lock "
            f"from a partial snapshot. stderr: "
            f"{(completed.stderr or '').strip()[:400]}")
    all_lines = sorted(line.strip() for line in completed.stdout.splitlines()
                       if line.strip() and not line.startswith("#"))
    excluded = sorted({name for name in
                       (_self_distribution_name(line) for line in all_lines)
                       if name})
    lines = [line for line in all_lines
             if not _is_self_distribution_line(line)]
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
        "editable_vcs_revisions": editable_vcs_revisions(lines),
    }


#: Fields that constitute PORTABLE dependency identity.
#:
#: ``executable`` is deliberately excluded: it is an absolute interpreter
#: path that differs between hosts and virtualenvs even when the installed
#: package set is byte-identical, so hashing it would make the lock
#: un-transferable while describing no dependency change at all. It is
#: still RECORDED in the snapshot as operational metadata (useful when
#: debugging which interpreter produced an artifact), it simply is not
#: part of what the hash certifies.
LOCK_IDENTITY_FIELDS = (
    "pip_freeze_sha256",
    "n_packages",
    "excluded_self_distributions",
    "pyproject_sha256",
    "python_version",
)

#: Operational (non-identity) metadata recorded alongside the lock:
#: ``executable`` is where the interpreter happens to live, and
#: ``editable_vcs_revisions`` are the live HEADs of SIBLING repositories
#: that are editable-installed into the same environment. Both are recorded
#: and reported, but neither is part of what the lock hash certifies.
LOCK_OPERATIONAL_FIELDS = ("executable", "editable_vcs_revisions")


def load_dependency_lock(lock_path: str | Path | None = None) -> dict | None:
    """The recorded ``dependency_lock`` block, or None if not captured.

    None means "no lock has been captured yet" (absent file, or the file
    has no ``dependency_lock`` block). A CORRUPT lock is not None: a YAML
    parse failure raises, because reading a corrupt lock as "nothing
    recorded" would let a run proceed with no dependency evidence.
    """
    path = Path(lock_path) if lock_path else DEFAULT_LOCK
    if not path.exists():
        return None
    try:
        lock = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ReplayError(f"{path}: unreadable dependency lock: {exc}") from exc
    dependency = lock.get("dependency_lock") if isinstance(lock, dict) else None
    return dependency if isinstance(dependency, dict) else None


def dependency_lock_sha256(lock_path: str | Path | None = None) -> str | None:
    """Hash of the recorded dependency lock (None if not yet captured).

    Hashes :data:`LOCK_IDENTITY_FIELDS` only, so the value is a statement
    about the installed package set and not about where the interpreter
    happens to live. A lock block that is present but missing an identity
    field is an error rather than a weaker hash.
    """
    dependency = load_dependency_lock(lock_path)
    if dependency is None:
        return None
    missing = [f for f in LOCK_IDENTITY_FIELDS if f not in dependency]
    if missing:
        raise ReplayError(
            f"{Path(lock_path) if lock_path else DEFAULT_LOCK}: recorded "
            f"dependency_lock is missing identity fields {missing}; "
            f"refusing to hash a partial dependency identity")
    identity = {f: dependency[f] for f in LOCK_IDENTITY_FIELDS}
    blob = yaml.safe_dump(identity, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_active_dependency_lock(
    lock_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> dict:
    """Compare the environment RUNNING NOW against the recorded lock.

    Recording a lock hash is not the same as enforcing it: without this
    check a run could execute under a different transformers/PEFT install
    while still reporting the old recorded hash, because the hash is read
    from the lock FILE rather than measured from the live environment.

    Args:
        strict: raise on any difference (confirmatory/eligibility runs).
            When False the differences are only reported, for diagnostics.

    Returns:
        A dict recording both snapshots' identity fields, the per-field
        differences, and the operational ``executable`` paths.
    """
    resolved_path = Path(lock_path) if lock_path else DEFAULT_LOCK
    locked = load_dependency_lock(resolved_path)
    if locked is None:
        if strict:
            raise ReplayError(
                f"{resolved_path}: no dependency lock recorded; a "
                f"confirmatory run cannot certify its environment. Run the "
                f"model preflight to capture one.")
        return {
            "verified": False,
            "reason": "no_dependency_lock_recorded",
            "lock_path": str(resolved_path),
            "checked_fields": list(LOCK_IDENTITY_FIELDS),
            "differences": {},
        }
    active = dependency_lock_snapshot()
    # A third-party editable install is not certifiable at all, whatever
    # the lock says: pip freeze identifies it by the sibling repository's
    # COMMITTED HEAD and is blind to that repository's uncommitted
    # working-tree changes, so no hash of freeze output can prove which
    # dependency source would execute. Refusing is the only sound answer.
    offenders = dict(active.get("editable_vcs_revisions") or {})
    differences = {
        f: {"locked": locked.get(f), "active": active.get(f)}
        for f in LOCK_IDENTITY_FIELDS
        if locked.get(f) != active.get(f)
    }
    # Operational drift is RECORDED but not by itself fatal: a different
    # interpreter path does not change what this project has installed.
    informational = {
        f: {"locked": locked.get(f), "active": active.get(f)}
        for f in LOCK_OPERATIONAL_FIELDS
        if locked.get(f) != active.get(f)
    }
    result = {
        "verified": not differences and not offenders,
        "lock_path": str(resolved_path),
        "checked_fields": list(LOCK_IDENTITY_FIELDS),
        "differences": differences,
        "informational_differences": informational,
        "third_party_editable_vcs": offenders,
        "locked_identity": {f: locked.get(f) for f in LOCK_IDENTITY_FIELDS},
        "active_identity": {f: active.get(f) for f in LOCK_IDENTITY_FIELDS},
        # Operational only: differing interpreters with identical identity
        # fields is expected across hosts and is NOT a failure.
        "locked_executable": locked.get("executable"),
        "active_executable": active.get("executable"),
        "locked_editable_vcs_revisions": locked.get("editable_vcs_revisions"),
        "active_editable_vcs_revisions": active.get("editable_vcs_revisions"),
        "dependency_lock_sha256": dependency_lock_sha256(resolved_path),
    }
    if offenders and strict:
        raise ReplayError(
            f"the active environment contains third-party editable VCS "
            f"install(s) {sorted(offenders)} at revisions "
            f"{offenders}. An editable dependency's source can change "
            f"without its recorded revision moving, so the environment "
            f"cannot be certified reproducible. Run confirmatory and "
            f"eligibility work in a dedicated Iteration 11 environment with "
            f"no third-party editable installs.")
    if differences and strict:
        detail = "; ".join(
            f"{f}: locked={v['locked']!r} active={v['active']!r}"
            for f, v in sorted(differences.items()))
        raise ReplayError(
            f"active environment does not match the recorded dependency "
            f"lock at {resolved_path} — {detail}. Inference would run under "
            f"a different dependency set than the one certified at "
            f"preflight; re-run the preflight to re-lock deliberately.")
    return result


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

