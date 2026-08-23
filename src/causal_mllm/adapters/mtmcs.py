"""MTMCS-Bench dataset adapter.

Source: https://huggingface.co/datasets/ND-25/MCS-bench
Paper: Liu et al. (2026), Findings of ACL 2026.
Official code: https://github.com/franciscoliu/MTMCS-Bench

====================================================================
CRITICAL: Dialogue semantics (verified against upstream inference code)
====================================================================

All r1/unsafe_r1/safe_r1/r2/r3/unsafe_r3/safe_r3 are USER turns.
The upstream evaluation generates assistant responses DYNAMICALLY
after each user turn. They are NOT pre-written assistant responses.

TYPE A  (divergence at the TERMINAL turn):

    unsafe trajectory:  user(r1) → user(r2) → user(unsafe_r3)
    safe   trajectory:  user(r1) → user(r2) → user(safe_r3)

    • r1, r2 are shared between safe and unsafe.
    • unsafe_r3 ≠ safe_r3: the terminal query itself changes.
    • Useful for studying escalation effects, but NOT naturally a
      fixed-terminal-query causal pair.

TYPE B  (divergence at the OPENING turn):

    unsafe trajectory:  user(unsafe_r1) → user(r2) → user(r3)
    safe   trajectory:  user(safe_r1)   → user(r2) → user(r3)

    • r2, r3 are shared between safe and unsafe.
    • unsafe_r1 ≠ safe_r1: only the opening history differs.
    • r3 is the terminal query — IDENTICAL across safe/unsafe.
    • This is almost exactly the causal experiment we want:
        (H_safe, q*) vs (H_unsafe, q*)

Each row also has a parallel unimodal_dialogue with the same structure
but text-only (no image context).

Each row has:
  • image: PIL.Image (main scenario photograph)
  • variant_images: List[PIL.Image] (3 derived/variant images)
  • unsafe_intent: str (abstract description of the harmful goal)
  • multimodal_mcq / multimodal_tf: evaluation questions
  • unimodal_mcq / unimodal_tf: text-only evaluation questions

Normalization explodes each row into FOUR canonical records:
  multimodal_safe, multimodal_unsafe, unimodal_safe, unimodal_unsafe
connected by a shared pair_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from causal_mllm.adapters.base import DatasetAdapter
from causal_mllm.data.media import MediaLoadError
from causal_mllm.data.schemas import CanonicalSourceExample, Message


class MTMCSAdapter(DatasetAdapter):
    """Adapter for MTMCS-Bench (ND-25/MCS-bench).

    Each HF row is exploded into 4 canonical records (mm_safe, mm_unsafe,
    text_safe, text_unsafe) connected by a shared pair_id.
    """

    def __init__(self, cache_dir: str | None = None, media_dir: str | None = None):
        self._cache_dir = cache_dir
        self._media_dir = Path(media_dir) if media_dir else Path("data/media/source")
        self._schema_report: dict[str, Any] | None = None

    @property
    def source_name(self) -> str:
        return "mtmcs"

    def load(self, split: str | None = None) -> Iterator[dict]:
        """Load raw rows from MTMCS-Bench. Each row yields one dict."""
        from datasets import load_dataset

        actual_split = split or "type_a"
        if actual_split not in ("type_a", "type_b"):
            raise ValueError(f"MTMCS split must be 'type_a' or 'type_b', got '{actual_split}'")
        ds = load_dataset("ND-25/MCS-bench", split=actual_split, cache_dir=self._cache_dir)
        for row in ds:
            d = dict(row)
            d["_split"] = actual_split
            yield d

    def normalize(self, raw_example: dict) -> CanonicalSourceExample:
        """Normalize is not used directly — use normalize_row() instead.

        The MTMCS adapter explodes each row into 4 records, so the
        base class's single-record normalize() is not appropriate.
        """
        raise NotImplementedError(
            "MTMCSAdapter.normalize() is not used directly. "
            "Use MTMCSAdapter.normalize_row() which returns 4 records per HF row."
        )

    def normalize_row(self, raw: dict) -> list[CanonicalSourceExample]:
        """Normalize one MTMCS HF row into four canonical records.

        Returns:
            A list of 4 CanonicalSourceExample instances:
            [multimodal_safe, multimodal_unsafe, unimodal_safe, unimodal_unsafe]
        """
        split = raw.get("_split", "type_a")
        row_id = raw["id"]
        unsafe_intent = raw.get("unsafe_intent", "")

        # Build pair_id — shared across all 4 records from this row
        pair_id = f"mtmcs:{split}:{row_id:06d}"

        # Save main image and verify it
        image_path = self._save_and_verify_image(raw.get("image"), split, row_id, "main")

        # Save variant images as metadata
        variant_paths = []
        for vi, vimg in enumerate(raw.get("variant_images", []) or []):
            vp = self._save_and_verify_image(vimg, split, row_id, f"variant{vi}")
            variant_paths.append(vp)

        results = []

        # --- Multimodal records ---
        for safety in ("safe", "unsafe"):
            messages = self._build_messages(raw, split, safety, modality="multimodal",
                                            image_path=image_path)
            terminal_query = self._get_terminal_query(raw, split, safety,
                                                      modality="multimodal")
            source_id = f"mtmcs:{split}:{row_id:06d}:mm:{safety}"
            label = "safe" if safety == "safe" else "unsafe"
            setting = split  # "type_a" or "type_b"

            results.append(CanonicalSourceExample(
                source_dataset="mtmcs",
                source_id=source_id,
                source_split=split,
                source_category=unsafe_intent[:80] if unsafe_intent else None,
                source_setting=setting,
                label=label,
                messages=messages,
                terminal_turn_index=messages[-1].turn_index,
                terminal_query=terminal_query,
                metadata={
                    "pair_id": pair_id,
                    "safety": safety,
                    "modality": "multimodal",
                    "unsafe_intent": unsafe_intent,
                    "image_path": image_path,
                    "variant_image_paths": variant_paths,
                    "mcq_count": len(raw.get("multimodal_mcq", []) or []),
                    "tf_count": len(raw.get("multimodal_tf", []) or []),
                },
            ))

        # --- Unimodal records (no image) ---
        for safety in ("safe", "unsafe"):
            messages = self._build_messages(raw, split, safety, modality="unimodal",
                                            image_path=None)
            terminal_query = self._get_terminal_query(raw, split, safety,
                                                      modality="unimodal")
            source_id = f"mtmcs:{split}:{row_id:06d}:text:{safety}"
            label = "safe" if safety == "safe" else "unsafe"
            setting = split

            results.append(CanonicalSourceExample(
                source_dataset="mtmcs",
                source_id=source_id,
                source_split=split,
                source_category=unsafe_intent[:80] if unsafe_intent else None,
                source_setting=setting,
                label=label,
                messages=messages,
                terminal_turn_index=messages[-1].turn_index,
                terminal_query=terminal_query,
                metadata={
                    "pair_id": pair_id,
                    "safety": safety,
                    "modality": "unimodal",
                    "unsafe_intent": unsafe_intent,
                    "image_path": None,
                    "variant_image_paths": variant_paths,
                    "mcq_count": len(raw.get("unimodal_mcq", []) or []),
                    "tf_count": len(raw.get("unimodal_tf", []) or []),
                },
            ))

        return results

    # ------------------------------------------------------------------
    # Dialogue reconstruction (mirrors upstream build_dialogue_turns_variant)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        raw: dict,
        split: str,
        safety: str,
        modality: str,
        image_path: str | None,
    ) -> list[Message]:
        """Reconstruct user trajectory per official MTMCS semantics.

        All fields are USER turns. No assistant responses are stored
        in the source data — the upstream evaluation generates them
        dynamically.

        Type A:
            unsafe: user(r1), user(r2), user(unsafe_r3)
            safe:   user(r1), user(r2), user(safe_r3)

        Type B:
            unsafe: user(unsafe_r1), user(r2), user(r3)
            safe:   user(safe_r1),   user(r2), user(r3)
        """
        dlg_key = "multimodal_dialogue" if modality == "multimodal" else "unimodal_dialogue"
        dlg = raw[dlg_key]
        messages: list[Message] = []
        turn = 0

        if split == "type_a":
            # Turn 0: r1 (shared) — with image for multimodal
            r1_text = dlg["r1"]
            if not r1_text or not r1_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: r1 is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=r1_text,
                images=[image_path] if (image_path and modality == "multimodal") else [],
            ))
            turn += 1

            # Turn 1: r2 (shared)
            r2_text = dlg["r2"]
            if not r2_text or not r2_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: r2 is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=r2_text,
                images=[],
            ))
            turn += 1

            # Turn 2: safe_r3 or unsafe_r3 (divergent terminal query)
            terminal_key = "safe_r3" if safety == "safe" else "unsafe_r3"
            terminal_text = dlg[terminal_key]
            if not terminal_text or not terminal_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: {terminal_key} is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=terminal_text,
                images=[],
            ))

        elif split == "type_b":
            # Turn 0: safe_r1 or unsafe_r1 (divergent opening) — with image
            opening_key = "safe_r1" if safety == "safe" else "unsafe_r1"
            opening_text = dlg[opening_key]
            if not opening_text or not opening_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: {opening_key} is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=opening_text,
                images=[image_path] if (image_path and modality == "multimodal") else [],
            ))
            turn += 1

            # Turn 1: r2 (shared)
            r2_text = dlg["r2"]
            if not r2_text or not r2_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: r2 is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=r2_text,
                images=[],
            ))
            turn += 1

            # Turn 2: r3 (shared terminal query)
            r3_text = dlg["r3"]
            if not r3_text or not r3_text.strip():
                raise ValueError(
                    f"MTMCS {split} id={raw['id']}: r3 is empty but required"
                )
            messages.append(Message(
                turn_index=turn,
                role="user",
                text=r3_text,
                images=[],
            ))

        else:
            raise ValueError(f"Unknown MTMCS split: '{split}'")

        return messages

    def _get_terminal_query(
        self, raw: dict, split: str, safety: str, modality: str
    ) -> str:
        """Extract the terminal query text for a given condition."""
        dlg_key = "multimodal_dialogue" if modality == "multimodal" else "unimodal_dialogue"
        dlg = raw[dlg_key]

        if split == "type_a":
            key = "safe_r3" if safety == "safe" else "unsafe_r3"
        elif split == "type_b":
            key = "r3"  # shared across safe/unsafe
        else:
            raise ValueError(f"Unknown split: {split}")

        text = dlg[key]
        if not text or not text.strip():
            raise ValueError(
                f"MTMCS {split} id={raw['id']}: terminal key '{key}' is empty"
            )
        return text

    # ------------------------------------------------------------------
    # Convenience: load and normalize all 4 records per row
    # ------------------------------------------------------------------

    def load_and_normalize(
        self, split: str | None = None, max_examples: int | None = None
    ) -> list[CanonicalSourceExample]:
        """Load and explode each row into 4 canonical records.

        Args:
            split: 'type_a' or 'type_b' (default: 'type_a').
            max_examples: Stop after this many TOTAL records (not rows).

        Returns:
            List of CanonicalSourceExample (4 per source row).
        """
        results: list[CanonicalSourceExample] = []
        for raw in self.load(split):
            try:
                records = self.normalize_row(raw)
                results.extend(records)
            except (ValueError, MediaLoadError) as e:
                from causal_mllm.data.logging import get_logger
                get_logger("causal_mllm.adapters").warning(
                    "Skipping MTMCS row %s: %s", raw.get("id", "?"), e
                )
            if max_examples and len(results) >= max_examples:
                break
        return results[:max_examples] if max_examples else results

    # ------------------------------------------------------------------
    # Schema inspection
    # ------------------------------------------------------------------

    def inspect_schema(self, n: int = 20) -> dict[str, Any]:
        """Inspect MTMCS-Bench schema for BOTH type_a and type_b."""
        from causal_mllm.adapters.inspect_datasets import inspect_mtmcs
        self._schema_report = inspect_mtmcs(cache_dir=self._cache_dir, n=n)
        return self._schema_report

    # ------------------------------------------------------------------
    # Image handling — fail loudly
    # ------------------------------------------------------------------

    def _save_and_verify_image(
        self, image: Any, split: str, row_id: int, suffix: str
    ) -> str:
        """Save a PIL image and verify it can be decoded.

        Raises MediaLoadError on any failure. Never returns None silently.
        """
        if image is None:
            raise MediaLoadError(
                f"mtmcs:{split}:{row_id}:{suffix}",
                "Image is None — expected a PIL Image object",
            )

        self._media_dir.mkdir(parents=True, exist_ok=True)
        filename = f"mtmcs_{split}_{row_id}_{suffix}.png"
        path = self._media_dir / filename

        try:
            if not path.exists():
                image.save(str(path))

            # Verify the saved file can be decoded by PIL
            from PIL import Image
            with Image.open(str(path)) as img:
                img.verify()

        except Exception as e:
            raise MediaLoadError(str(path), f"PIL save/verify failed: {e}") from e

        # Re-open after verify() to confirm (verify closes the image)
        try:
            with Image.open(str(path)) as img:
                img.load()
        except Exception as e:
            raise MediaLoadError(str(path), f"PIL load-after-verify failed: {e}") from e

        return str(path)
