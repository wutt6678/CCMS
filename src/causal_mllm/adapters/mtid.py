"""MTID (Multi-Turn Intervention Dataset) adapter.

Source: https://huggingface.co/datasets/Graph-COM/MTID
Repository: https://github.com/Graph-COM/TurnGate

Schema discovered programmatically during Iteration 1:
  - Format: JSONL files (harmful/benign x train/test/valid)
  - Keys: sample_index, rollout_id, dataset_key, target_turn,
    target_confidence, target_reasoning, meta_intent, conversation,
    asr_classification
  - conversation: list of {turn_id, role, content, hidden_rationale}
  - Text-only (no images)
  - 800 unique samples, 20 rollouts each, ~16000 trajectories

Note: The HF datasets library cannot load this dataset due to a 'Json'
feature type incompatibility. We use direct JSONL download via huggingface_hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from causal_mllm.adapters.base import DatasetAdapter
from causal_mllm.data.schemas import CanonicalSourceExample, Message


class MTIDAdapter(DatasetAdapter):
    """Adapter for MTID (Graph-COM/MTID)."""

    def __init__(self, cache_dir: str | None = None, data_dir: str | None = None):
        self._cache_dir = cache_dir
        self._data_dir = Path(data_dir) if data_dir else Path("data/raw/mtid")
        self._schema_report: dict[str, Any] | None = None
        self._downloaded = False

    @property
    def source_name(self) -> str:
        return "mtid"

    def _ensure_downloaded(self) -> None:
        """Download JSONL files from HuggingFace Hub if not already present."""
        if self._downloaded:
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        for fname in [
            "harmful_test.jsonl", "harmful_train.jsonl", "harmful_valid.jsonl",
            "benign_test.jsonl", "benign_train.jsonl", "benign_valid.jsonl",
        ]:
            dest = self._data_dir / fname
            if not dest.exists():
                path = hf_hub_download("Graph-COM/MTID", fname, repo_type="dataset")
                import shutil
                shutil.copy2(path, str(dest))

        self._downloaded = True

    def load(self, split: str | None = None) -> Iterator[dict]:
        """Load raw examples from downloaded MTID JSONL files."""
        self._ensure_downloaded()

        if split and split in ("test", "train", "valid"):
            # Load specific split (both harmful and benign)
            for prefix in ["harmful", "benign"]:
                fname = f"{prefix}_{split}.jsonl"
                fpath = self._data_dir / fname
                if fpath.exists():
                    yield from self._load_jsonl(fpath, f"{prefix}_{split}")
        else:
            # Load all files
            for fpath in sorted(self._data_dir.glob("*.jsonl")):
                yield from self._load_jsonl(fpath, fpath.stem)

    def _load_jsonl(self, fpath: Path, source_file: str) -> Iterator[dict]:
        """Load records from a single JSONL file."""
        with fpath.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                record["_source_file"] = source_file
                record["_line_number"] = lineno
                yield record

    def normalize(self, raw_example: dict) -> CanonicalSourceExample:
        """Normalize a raw MTID example into the canonical schema."""
        sample_index = raw_example.get("sample_index", 0)
        rollout_id = raw_example.get("rollout_id", 0)
        source_file = raw_example.get("_source_file", "unknown")
        conversation = raw_example.get("conversation", [])

        # Build canonical messages
        messages: list[Message] = []
        for turn in conversation:
            messages.append(Message(
                turn_index=turn.get("turn_id", len(messages)),
                role=turn.get("role", "unknown"),
                text=turn.get("content", ""),
                images=[],
            ))

        if not messages:
            raise ValueError(f"Empty conversation in {source_file}:sample_{sample_index}")

        # Terminal query is the last user message
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            raise ValueError(f"No user messages in {source_file}:sample_{sample_index}")

        terminal_msg = user_messages[-1]
        terminal_turn_index = terminal_msg.turn_index
        terminal_query = terminal_msg.text

        # Determine label
        is_harmful = "harmful" in source_file
        source_id = f"mtid_{sample_index}_rollout{rollout_id}"

        # Determine setting
        target_turn = raw_example.get("target_turn")

        return CanonicalSourceExample(
            source_dataset="mtid",
            source_id=source_id,
            source_split=source_file,
            source_category=raw_example.get("dataset_key"),
            source_setting="other",
            label="unsafe" if is_harmful else "safe",
            messages=messages,
            terminal_turn_index=terminal_turn_index,
            terminal_query=terminal_query,
            metadata={
                "sample_index": sample_index,
                "rollout_id": rollout_id,
                "target_turn": target_turn,
                "target_confidence": raw_example.get("target_confidence"),
                "target_reasoning": raw_example.get("target_reasoning"),
                "meta_intent": raw_example.get("meta_intent"),
                "asr_classification": raw_example.get("asr_classification"),
            },
        )

    def inspect_schema(self, n: int = 20) -> dict[str, Any]:
        """Inspect MTID schema via direct JSONL download."""
        from causal_mllm.adapters.inspect_datasets import inspect_mtid
        self._schema_report = inspect_mtid(cache_dir=self._cache_dir, n=n)
        return self._schema_report
