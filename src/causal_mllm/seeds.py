"""Deterministic seed utilities for reproducible experiments."""

import hashlib
import os
import random
from typing import Optional


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for all relevant RNGs.

    Args:
        seed: Integer seed for reproducibility.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def deterministic_family_id(source_dataset: str, source_id: str, seed: int = 42) -> str:
    """Generate a deterministic family ID from source provenance.

    The ID is stable across runs given the same inputs and seed.

    Args:
        source_dataset: Name of the source dataset (e.g., 'mtmcs').
        source_id: Original ID from the source dataset.
        seed: Global experiment seed.

    Returns:
        A string like 'CMST_000001' derived from a hash of inputs.
    """
    raw = f"{source_dataset}:{source_id}:{seed}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # Use the first 8 hex chars as a numeric index, modulo a large range
    numeric = int(digest[:8], 16)
    return f"CMST_{numeric % 1_000_000:06d}"


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string after normalization.

    Normalization strips leading/trailing whitespace and collapses
    internal whitespace runs to a single space, matching the spec's
    normalize_bytes requirement.
    """
    import re
    normalized = re.sub(r"\s+", " ", text.strip())
    return sha256_bytes(normalized.encode("utf-8"))


def config_hash(config_dict: dict) -> str:
    """Compute a deterministic hash of a configuration dictionary.

    Args:
        config_dict: Arbitrary configuration dictionary (must be JSON-serializable).

    Returns:
        SHA-256 hex digest of the canonical JSON representation.
    """
    import json
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return sha256_bytes(canonical.encode("utf-8"))


#: Repository root, derived from this file rather than from the process
#: cwd so the answer does not depend on where a script was launched.
#: Every provenance question below is anchored here: ``git_commit`` and
#: ``git_dirty`` must describe the repository whose code executed, not
#: whatever directory the process happened to be started in. Launching a
#: stage from outside the repo used to record ``code_commit: None`` (and so
#: an unresolvable provenance) while the tree determination, which was
#: already anchored, described the repo correctly — two answers to one
#: question. For a non-editable install this reports the installed source's
#: own repository, or None; that is the honest answer, and strictly better
#: than recording the commit of an unrelated repository the caller stood in.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if not in a git repo.

    Anchored on :data:`_REPO_ROOT`, not on the process cwd.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def git_working_tree_paths() -> Optional[dict]:
    """Modified-tracked and untracked paths, repo-root relative.

    ``--untracked-files=all`` expands untracked DIRECTORIES into their
    individual files, so an untracked ``src/.../new_module.py`` is visible
    rather than hidden behind a directory entry. Files matched by
    ``.gitignore`` are not reported by git at all, which is why the cache
    filter in :func:`code_tree_status` is a belt-and-braces measure rather
    than the primary defence.

    Returns:
        ``{"modified": [...], "untracked": [...]}``, or None if git is
        unavailable or this is not a repository (provenance unknown, which
        callers must treat as "cannot certify").
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    modified: list = []
    untracked: list = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # porcelain v1: "XY <path>", or "XY <orig> -> <path>" for renames.
        code = line[:2]
        entry = line[3:].strip()
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip('"')
        if code == "??":
            untracked.append(entry)
        else:
            modified.append(entry)
    return {"modified": modified, "untracked": untracked}


def dirty_tracked_files() -> Optional[list]:
    """Tracked files with uncommitted changes (None if git is unavailable).

    Retained for callers that want the narrow, tracked-only answer; the
    provenance gate uses :func:`code_tree_status`, which also considers
    untracked files.
    """
    paths = git_working_tree_paths()
    return None if paths is None else paths["modified"]


#: Path components that are caches or transient build state wherever they
#: appear. ``.gitignore`` already excludes these in this repository, so the
#: explicit list only matters if that ever changes — it is deliberately NOT
#: a general escape hatch for untracked files.
CACHE_PATH_COMPONENTS = frozenset({
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".cache", ".conda", ".ci-venv", "node_modules",
})

#: Suffixes that are never execution-relevant source.
CACHE_SUFFIXES = (".pyc", ".pyo", ".log")


def is_cache_path(path: str) -> bool:
    """True for cache/transient paths, which never affect what executes."""
    if any(component in CACHE_PATH_COMPONENTS
           for component in path.split("/")):
        return True
    return path.endswith(CACHE_SUFFIXES)


def code_tree_status(exclude_prefixes=()) -> dict:
    """Could ``code_commit`` reconstruct the code that is about to run?

    Considers modified tracked files AND untracked files. Untracked source
    is the dangerous case, and ignoring it was a real hole: an untracked
    ``sitecustomize.py`` or ``conftest.py``, a top-level module shadowing an
    installed package, or a new module that tracked code imports all change
    what executes while leaving ``code_commit`` pointing at a tree that
    cannot reproduce it — and the artifact would still record
    ``git_dirty: false``.

    A stage's own output artifacts are the one legitimate exception, because
    a stage that regenerates its own committed evidence would otherwise
    deadlock itself: the first target's artifact would make the tree dirty
    for every later target with no code change at all. That exclusion is
    narrow, per-stage and REPORTED, never a blanket ignore of ``outputs/``.

    Args:
        exclude_prefixes: repo-root-relative prefixes belonging to the
            calling stage's own outputs.

    Returns:
        ``{"dirty": bool | None, "dirty_paths": [...], "untracked_paths":
        [...], "excluded_own_outputs": [...], "excluded_cache_paths":
        [...]}``. ``dirty`` is None when git status is unavailable.
    """
    paths = git_working_tree_paths()
    if paths is None:
        return {"dirty": None, "dirty_paths": [], "untracked_paths": [],
                "excluded_own_outputs": [], "excluded_cache_paths": []}
    prefixes = tuple(exclude_prefixes)
    all_paths = list(paths["modified"]) + list(paths["untracked"])
    own_outputs, caches, dirty = [], [], []
    for path in all_paths:
        if any(path.startswith(prefix) for prefix in prefixes):
            own_outputs.append(path)
        elif is_cache_path(path):
            caches.append(path)
        else:
            dirty.append(path)
    untracked_dirty = [p for p in dirty if p in set(paths["untracked"])]
    return {
        "dirty": bool(dirty),
        "dirty_paths": sorted(dirty),
        "untracked_paths": sorted(untracked_dirty),
        "excluded_own_outputs": sorted(own_outputs),
        "excluded_cache_paths": sorted(caches),
    }


def is_git_dirty() -> Optional[bool]:
    """Return True if the git working tree has uncommitted changes to
    tracked files.  Untracked files (e.g. newly created replay output
    directories) are NOT counted — they are normal side effects of
    running the pipeline and become evidence once committed.

    This is the narrow, tracked-only answer, and it is what provenance
    RECORDS as ``git_dirty``. It is not sufficient to GATE on: use
    :func:`code_tree_status`, which also considers untracked files.

    Anchored on :data:`_REPO_ROOT`, not on the process cwd.

    Returns None if not inside a git repo (provenance unknown).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
