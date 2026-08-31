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


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def is_git_dirty() -> Optional[bool]:
    """Return True if the git working tree has uncommitted changes to
    tracked files.  Untracked files (e.g. newly created replay output
    directories) are NOT counted — they are normal side effects of
    running the pipeline and become evidence once committed.

    Returns None if not inside a git repo (provenance unknown).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
