"""JSONL read/write helpers and configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed JSON objects.

    Blank lines are silently skipped. Each non-blank line must be valid JSON.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at {path}:{lineno}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return records


def read_jsonl_iter(path: str | Path) -> Iterator[dict]:
    """Lazily iterate over JSONL records without loading all into memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: str | Path, records: list[dict], *, append: bool = False) -> int:
    """Write a list of dictionaries to a JSONL file.

    Args:
        path: Destination path.
        records: Dictionaries to serialize.
        append: If True, append to existing file instead of overwriting.

    Returns:
        Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# YAML configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
