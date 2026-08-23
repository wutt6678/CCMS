"""CoSafe dataset adapter.

Source: https://github.com/ErxinYu/CoSafe-Dataset
Paper: Yu et al. (2024), EMNLP 2024.

Schema discovered programmatically during Iteration 1:
  - Format: JSONL files organized by safety category
  - Each record is a list of message dicts: [{role, content}, ...]
  - Roles: "user" and "assistant"
  - Terminal query: last user message
  - Text-only (no images)
  - 14 category files, ~100 records each

Used primarily as a structural template for coreference-based risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from causal_mllm.adapters.base import DatasetAdapter
from causal_mllm.data.schemas import CanonicalSourceExample, Message


class CoSafeAdapter(DatasetAdapter):
    """Adapter for CoSafe dataset (cloned from GitHub)."""

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = "data/raw/cosafe/CoSafe-Dataset/CoSafe datasets"
        self._data_dir = Path(data_dir)
        self._schema_report: dict[str, Any] | None = None

    @property
    def source_name(self) -> str:
        return "cosafe"

    def load(self, split: str | None = None) -> Iterator[dict]:
        """Load CoSafe examples from cloned repository JSONL files."""
        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"CoSafe data directory not found: {self._data_dir}. "
                "Clone https://github.com/ErxinYu/CoSafe-Dataset first."
            )

        for jf in sorted(self._data_dir.glob("*.json")):
            category = jf.stem
            with jf.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    messages = json.loads(stripped)
                    yield {
                        "source_file": jf.name,
                        "category": category,
                        "line_number": lineno,
                        "messages": messages,
                    }

    def normalize(self, raw_example: dict) -> CanonicalSourceExample:
        """Normalize a raw CoSafe example into the canonical schema."""
        messages_raw = raw_example["messages"]
        category = raw_example.get("category", "unknown")
        source_file = raw_example.get("source_file", "unknown")
        line_number = raw_example.get("line_number", 0)

        # Build canonical messages
        messages: list[Message] = []
        for i, msg in enumerate(messages_raw):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            messages.append(Message(
                turn_index=i,
                role=role,
                text=content,
                images=[],
            ))

        if not messages:
            raise ValueError(f"Empty conversation in {source_file}:{line_number}")

        # Terminal query is the last user message
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            raise ValueError(f"No user messages in {source_file}:{line_number}")

        terminal_msg = user_messages[-1]
        terminal_turn_index = terminal_msg.turn_index
        terminal_query = terminal_msg.text

        source_id = f"{source_file}:{line_number}"

        return CanonicalSourceExample(
            source_dataset="cosafe",
            source_id=source_id,
            source_split=None,
            source_category=category,
            source_setting="coreference",
            label="unsafe",  # CoSafe scenarios represent unsafe contexts
            messages=messages,
            terminal_turn_index=terminal_turn_index,
            terminal_query=terminal_query,
            metadata={
                "source_file": source_file,
                "category": category,
                "num_messages": len(messages),
            },
        )

    def inspect_schema(self, n: int = 20) -> dict[str, Any]:
        """Inspect CoSafe schema from cloned repository."""
        from causal_mllm.adapters.inspect_datasets import inspect_cosafe
        self._schema_report = inspect_cosafe(data_dir=str(self._data_dir), n=n)
        return self._schema_report
