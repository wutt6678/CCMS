"""MTMCS-Bench dataset adapter.

Source: https://huggingface.co/datasets/ND-25/MCS-bench
Paper: Liu et al. (2026), Findings of ACL 2026.

Schema discovered programmatically during Iteration 1.

Schema summary:
  - id: int32 (unique scenario identifier)
  - image: PIL.Image (main scenario image, real photograph)
  - variant_images: List[PIL.Image] (3 derived images)
  - unsafe_intent: str (abstract description of harmful goal)
  - multimodal_dialogue: dict {r1, unsafe_r1, safe_r1, r2, r3, unsafe_r3, safe_r3}
  - unimodal_dialogue: dict (same structure, text-only variant)
  - multimodal_mcq: List[{question, question_type, options, ground_truth}]
  - multimodal_tf: List[{question, ground_truth(bool)}]
  - unimodal_mcq, unimodal_tf: same as above

Dialogue structure:
  - r1: Round 1 user query (benign, image-referencing)
  - r2: Round 2 user query (escalation, narrowing)
  - r3: Round 3 query field (usually empty)
  - safe_r3 / unsafe_r3: encode the terminal query variant + response

For normalization, we construct the canonical message list from the
multimodal_dialogue fields, using r1 and r2 as historical user turns,
and safe_r3 as the basis for the terminal query.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from causal_mllm.adapters.base import DatasetAdapter
from causal_mllm.data.schemas import CanonicalSourceExample, Message


class MTMCSAdapter(DatasetAdapter):
    """Adapter for MTMCS-Bench (ND-25/MCS-bench)."""

    def __init__(self, cache_dir: str | None = None, media_dir: str | None = None):
        self._cache_dir = cache_dir
        self._media_dir = Path(media_dir) if media_dir else Path("data/media/source")
        self._schema_report: dict[str, Any] | None = None

    @property
    def source_name(self) -> str:
        return "mtmcs"

    def load(self, split: str | None = None) -> Iterator[dict]:
        """Load raw examples from MTMCS-Bench via Hugging Face datasets."""
        from datasets import load_dataset

        actual_split = split or "type_a"
        ds = load_dataset("ND-25/MCS-bench", split=actual_split, cache_dir=self._cache_dir)
        for row in ds:
            yield dict(row)

    def normalize(self, raw_example: dict) -> CanonicalSourceExample:
        """Normalize a raw MTMCS example into the canonical schema.

        Constructs a multi-turn conversation from the multimodal_dialogue
        fields. The terminal query is extracted from safe_r3.
        """
        source_id = str(raw_example["id"])
        md = raw_example.get("multimodal_dialogue", {})

        # Build message list from dialogue fields
        messages: list[Message] = []
        turn_index = 0

        # Round 1: user query with image
        r1 = md.get("r1", "")
        if r1:
            img_path = self._save_image(raw_example.get("image"), source_id, "main")
            messages.append(Message(
                turn_index=turn_index,
                role="user",
                text=r1,
                images=[img_path] if img_path else [],
            ))
            turn_index += 1

        # Round 1: assistant response (safe)
        safe_r1 = md.get("safe_r1", "")
        if safe_r1:
            messages.append(Message(
                turn_index=turn_index,
                role="assistant",
                text=safe_r1,
                images=[],
            ))
            turn_index += 1

        # Round 2: user query (escalation)
        r2 = md.get("r2", "")
        if r2:
            messages.append(Message(
                turn_index=turn_index,
                role="user",
                text=r2,
                images=[],
            ))
            turn_index += 1

        # Round 2: assistant response (safe)
        # We don't have a separate safe_r2, so we skip

        # Round 3: terminal query extracted from safe_r3
        # The safe_r3 field encodes both the question and the safe response.
        # We extract the question framing as the terminal query.
        safe_r3 = md.get("safe_r3", "")
        unsafe_r3 = md.get("unsafe_r3", "")

        # Use safe_r3 as the terminal query (it frames the question in a
        # way that can be answered safely or unsafely)
        terminal_query = safe_r3 if safe_r3 else unsafe_r3
        if not terminal_query:
            raise ValueError(f"No terminal query found for MTMCS id={source_id}")

        messages.append(Message(
            turn_index=turn_index,
            role="user",
            text=terminal_query,
            images=[],
        ))

        # Determine setting
        setting = "escalation"  # MTMCS uses escalation-based risk

        # Build metadata
        metadata = {
            "unsafe_intent": raw_example.get("unsafe_intent", ""),
            "has_variant_images": len(raw_example.get("variant_images", [])) > 0,
            "variant_image_count": len(raw_example.get("variant_images", [])),
            "multimodal_mcq_count": len(raw_example.get("multimodal_mcq", [])),
            "multimodal_tf_count": len(raw_example.get("multimodal_tf", [])),
            "unimodal_dialogue_available": bool(raw_example.get("unimodal_dialogue")),
            "raw_safe_r3": safe_r3,
            "raw_unsafe_r3": unsafe_r3,
        }

        return CanonicalSourceExample(
            source_dataset="mtmcs",
            source_id=source_id,
            source_split=None,
            source_category=raw_example.get("unsafe_intent", "")[:50] if raw_example.get("unsafe_intent") else None,
            source_setting=setting,
            label="unsafe",  # All MTMCS scenarios represent unsafe contexts
            messages=messages,
            terminal_turn_index=turn_index,
            terminal_query=terminal_query,
            metadata=metadata,
        )

    def inspect_schema(self, n: int = 20) -> dict[str, Any]:
        """Inspect MTMCS-Bench schema by loading real rows."""
        from causal_mllm.adapters.inspect_datasets import inspect_mtmcs
        self._schema_report = inspect_mtmcs(cache_dir=self._cache_dir, n=n)
        return self._schema_report

    def _save_image(self, image, source_id: str, suffix: str) -> str | None:
        """Save a PIL image to the media directory and return the relative path."""
        if image is None:
            return None
        try:
            self._media_dir.mkdir(parents=True, exist_ok=True)
            filename = f"mtmcs_{source_id}_{suffix}.png"
            path = self._media_dir / filename
            if not path.exists():
                image.save(str(path))
            return str(path)
        except Exception:
            return None
