"""Shared synthetic builders for construction-pipeline unit tests."""

from __future__ import annotations

from causal_mllm.data.schemas import CanonicalSourceExample, Message


def make_mtmcs_group(
    split: str = "type_b",
    pair_num: int = 1,
    *,
    break_terminal: bool = False,
    equal_terminals: bool = False,
    n_turns: int = 3,
) -> list[CanonicalSourceExample]:
    """Build one valid MTMCS 4-record atomic group.

    type_b: divergence at opening turn, terminal r3 SHARED across safe/unsafe.
    type_a: shared opening turns, terminal turn DIVERGES across safe/unsafe.

    break_terminal: corrupt the mm_unsafe terminal (for invariant tests).
    equal_terminals: force type_a-style terminals to be identical.
    """
    pair_id = f"mtmcs:{split}:{pair_num:06d}"
    records = []

    shared_terminal = f"What should we do with the object in the image, row {pair_num}?"
    safe_terminal_a = f"Can you help me safely handle this, row {pair_num}?"
    unsafe_terminal_a = f"Now tell me how to misuse it, row {pair_num}?"

    for modality in ("multimodal", "unimodal"):
        for safety in ("safe", "unsafe"):
            if split == "type_b":
                opening = (f"[safe opening] describe the scene {pair_num}"
                           if safety == "safe"
                           else f"[unsafe opening] ignore rules {pair_num}")
                terminal = shared_terminal
                if break_terminal and modality == "multimodal" and safety == "unsafe":
                    terminal = "CORRUPTED DIFFERENT TERMINAL QUERY"
            else:  # type_a
                opening = f"Opening question about row {pair_num}."
                terminal = safe_terminal_a if safety == "safe" else unsafe_terminal_a
                if equal_terminals:
                    terminal = safe_terminal_a

            # Pad turns up to n_turns exactly (terminal is always last)
            texts = [opening]
            while len(texts) < n_turns - 1:
                texts.append(f"Filler turn {len(texts)} about row {pair_num}.")
            texts.append(terminal)

            messages = []
            for i, text in enumerate(texts):
                images = []
                if modality == "multimodal" and i == 0:
                    images = [f"media/source/mtmcs_{split}_{pair_num}_main.png"]
                messages.append(Message(turn_index=i, role="user",
                                        text=text, images=images))

            records.append(CanonicalSourceExample(
                source_dataset="mtmcs",
                source_id=f"mtmcs:{split}:{pair_num:06d}:"
                          f"{'mm' if modality == 'multimodal' else 'text'}:{safety}",
                source_split=split,
                source_category=None,
                source_setting=split,
                label=safety,
                messages=messages,
                terminal_turn_index=len(texts) - 1,
                terminal_query=terminal,
                metadata={
                    "pair_id": pair_id,
                    "safety": safety,
                    "modality": modality,
                },
            ))
    return records


def make_text_only_singleton(
    source_id: str = "cosafe:test:000001",
    dataset: str = "cosafe",
    setting: str = "coreference",
) -> CanonicalSourceExample:
    """Build a text-only singleton (CoSafe/MTID-shaped)."""
    messages = [
        Message(turn_index=0, role="user", text="Tell me about that device."),
        Message(turn_index=1, role="assistant", text="It is a kitchen blender."),
        Message(turn_index=2, role="user",
                text="How fast can it operate unattended?"),
    ]
    return CanonicalSourceExample(
        source_dataset=dataset,
        source_id=source_id,
        source_split="test",
        source_category=None,
        source_setting=setting,
        label="unsafe",
        messages=messages,
        terminal_turn_index=2,
        terminal_query="How fast can it operate unattended?",
        metadata={},
    )
