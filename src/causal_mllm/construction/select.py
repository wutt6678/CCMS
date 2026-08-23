"""Candidate selection for causal family construction.

Iteration 2: Type guard ensuring the family builder consumes only
CanonicalSourceExample instances. This prevents raw dicts from
bypassing normalization.

Iteration 3 will implement the full selection pipeline here.
"""

from __future__ import annotations

from typing import Sequence

from causal_mllm.data.schemas import CanonicalSourceExample


def assert_canonical(examples: Sequence[CanonicalSourceExample]) -> None:
    """Type guard: verify that all inputs are CanonicalSourceExample.

    The family builder pipeline must consume ONLY CanonicalSourceExample.
    This function raises TypeError if any element is not the correct type,
    preventing raw dicts from silently bypassing normalization.

    Args:
        examples: Sequence of normalized examples.

    Raises:
        TypeError: If any element is not a CanonicalSourceExample.
    """
    for i, ex in enumerate(examples):
        if not isinstance(ex, CanonicalSourceExample):
            raise TypeError(
                f"examples[{i}] is {type(ex).__name__}, "
                f"expected CanonicalSourceExample. "
                f"Raw dicts must be normalized before entering the pipeline."
            )


def select_candidates(
    examples: Sequence[CanonicalSourceExample],
    *,
    min_turns: int = 3,
    max_turns: int = 8,
    require_images: bool = True,
    max_text_length: int = 5000,
) -> tuple[list[CanonicalSourceExample], list[dict]]:
    """Conservative candidate selector for causal family construction.

    Applies inclusion/exclusion criteria from the experiment plan.
    Returns (accepted, rejections) where each rejection has a reason.

    This is a stub for Iteration 2. Full implementation in Iteration 3.

    Args:
        examples: Normalized source examples.
        min_turns: Minimum conversational turns.
        max_turns: Maximum conversational turns.
        require_images: Whether to require at least one image.
        max_text_length: Maximum total text length.

    Returns:
        Tuple of (accepted examples, rejection records).
    """
    assert_canonical(examples)

    accepted: list[CanonicalSourceExample] = []
    rejections: list[dict] = []

    for ex in examples:
        reason = None

        # Turn count
        if ex.num_turns < min_turns:
            reason = f"too few turns ({ex.num_turns} < {min_turns})"
        elif ex.num_turns > max_turns:
            reason = f"too many turns ({ex.num_turns} > {max_turns})"

        # Image requirement
        if require_images and not ex.has_images:
            reason = "no images"

        # Text length
        if reason is None:
            total_text = sum(len(m.text or "") for m in ex.messages)
            if total_text > max_text_length:
                reason = f"text too long ({total_text} > {max_text_length})"

        if reason:
            rejections.append({
                "source_id": ex.source_id,
                "source_dataset": ex.source_dataset,
                "reason": reason,
            })
        else:
            accepted.append(ex)

    return accepted, rejections
