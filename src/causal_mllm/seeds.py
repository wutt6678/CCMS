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


#: Repo-root-relative prefixes whose contents can change what a run computes.
CODE_PATH_PREFIXES = ("src/", "scripts/", "configs/", "tests/", ".github/")

#: Frozen INPUTS that live under ``outputs/``. They are not code, but a run
#: reads them: the 11.5 selection is derived from the frozen Iteration-10
#: panel, reference run and adjudicated labels, and the gate reads the frozen
#: protocol and the resolved-model lock. A mid-run change to one of these
#: changes what the run computes just as surely as a change under ``src/``.
RUNTIME_INPUT_PREFIXES = (
    "outputs/iteration_11/protocol/",
    "outputs/iteration_11/preflight/",
    "outputs/scale_c/",
)

#: Generated evidence, media and prose: cannot change what a run computes.
#: Deliberately a short, explicit list — see the fail-closed default below.
NON_EXECUTION_PREFIXES = ("outputs/", "docs/", "data/media/", "figures/",
                          "reports/")
NON_EXECUTION_SUFFIXES = (".md", ".txt", ".rst", ".png", ".jpg", ".jpeg",
                          ".pdf", ".svg", ".jsonl", ".log", ".csv")
NON_EXECUTION_NAMES = frozenset({
    "README.md", "LICENSE", "CHANGELOG.md", ".gitignore", ".gitattributes",
})

#: Root-level or specially-named files that can change what a run computes
#: wherever they appear: packaging metadata, tool configuration, and the files
#: Python's import machinery picks up implicitly.
CODE_PATH_NAMES = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "tox.ini",
    "ruff.toml", ".flake8", "Makefile", "conftest.py", "sitecustomize.py",
    "usercustomize.py",
})


def is_execution_relevant_path(path: str) -> bool:
    """Could a change at ``path`` alter what a run computes?

    Used to tell apart two kinds of dirty tree at the END of a run. The tree
    is checked whole at LAUNCH, so by the time a report is written the process
    has already imported its code and ``code_commit`` pins it: a file appearing
    among the generated evidence cannot retroactively change what was produced.
    A file appearing under ``src/`` can, because Python imports lazily and a
    later stage of the same run may import something it has not imported yet —
    and so can a change to a frozen input the run still has to read.

    FAIL-CLOSED: relevance is the default and only the declared evidence,
    media and prose shapes are forgiven. A directory nobody anticipated is
    therefore treated as able to change the result rather than as harmless, so
    this scoping cannot become a way for an unrecognised file to excuse itself.

    Args:
        path: a repo-root-relative path from :func:`git_working_tree_paths`.

    Returns:
        True unless the path is recognisably generated evidence, media or
        documentation.
    """
    if any(path.startswith(prefix) for prefix in RUNTIME_INPUT_PREFIXES):
        return True
    if any(path.startswith(prefix) for prefix in CODE_PATH_PREFIXES):
        return True
    name = path.rsplit("/", 1)[-1]
    if name in CODE_PATH_NAMES:
        return True
    # A stray top-level module can shadow an installed package or be picked up
    # by site/conftest machinery, which is the case code_tree_status already
    # treats as dangerous.
    if "/" not in path and name.endswith(".py"):
        return True
    if any(path.startswith(prefix) for prefix in NON_EXECUTION_PREFIXES):
        return False
    if name in NON_EXECUTION_NAMES or name.endswith(NON_EXECUTION_SUFFIXES):
        return False
    return True


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
        [...], "code_dirty_paths": [...], "excluded_own_outputs": [...],
        "excluded_cache_paths": [...]}``. ``dirty`` is None when git status is
        unavailable. ``code_dirty_paths`` is the subset of ``dirty_paths`` that
        :func:`is_execution_relevant_path` considers able to change what a run
        computes; a stage that has already imported its code gates on that
        subset at the END of a run while still recording the whole tree.
    """
    paths = git_working_tree_paths()
    if paths is None:
        return {"dirty": None, "dirty_paths": [], "untracked_paths": [],
                "code_dirty_paths": [],
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
        "code_dirty_paths": sorted(p for p in dirty
                                   if is_execution_relevant_path(p)),
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
