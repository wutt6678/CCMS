"""Type-B cross-modality alignment diagnostics.

The factorial experiment (neutral / text_only / vision_only /
cross_modal) needs ONE terminal query q* across modalities, and the
MTMCS multimodal and unimodal dialogues are SEPARATELY WRITTEN source
fields — their equivalence must be measured, never assumed.

This module measures, across all Type-B source rows:

  * safe/unsafe fixed-q* validity (within-condition invariants)
  * mm/text terminal-query equality (exact AND normalized whitespace)
  * mm/text per-turn textual alignment rates
  * rows directly usable for 2x2 construction vs rows needing rewriting

The heavy lifting is a pure function over raw rows
(`diagnose_type_b_rows`) so the logic is unit-testable without data.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# The four dialogue fields of a Type-B row, in conversation order
TYPE_B_FIELDS = ("safe_r1", "unsafe_r1", "r2", "r3")

# The field that becomes the terminal query q*
TERMINAL_FIELD = "r3"


def _norm(text: Any) -> str:
    """Whitespace normalization consistent with seeds.sha256_text."""
    return re.sub(r"\s+", " ", str(text or "").strip())


def _row_complete(row: dict) -> bool:
    """All eight dialogue fields present and non-empty."""
    for dlg_field in ("multimodal_dialogue", "unimodal_dialogue"):
        dlg = row.get(dlg_field) or {}
        for field in TYPE_B_FIELDS:
            if not _norm(dlg.get(field)):
                return False
    return True


def _row_within_condition_valid(row: dict) -> dict:
    """Fixed-q* validity within each condition.

    Type-B rows carry a SINGLE r3 field per modality, so both safe and
    unsafe variants necessarily share it — but we verify the field is
    non-empty rather than trusting the structure silently.
    """
    mm = row.get("multimodal_dialogue") or {}
    text = row.get("unimodal_dialogue") or {}
    return {
        "mm_safe_vs_mm_unsafe": bool(_norm(mm.get(TERMINAL_FIELD))),
        "text_safe_vs_text_unsafe": bool(_norm(text.get(TERMINAL_FIELD))),
    }


def diagnose_type_b_rows(raw_rows: Iterable[dict]) -> dict:
    """Pure diagnostic over raw Type-B HF rows.

    Args:
        raw_rows: Raw MTMCS rows (dicts with multimodal_dialogue and
            unimodal_dialogue).

    Returns:
        Report dict with exact/normalized cross-modality match rates,
        per-turn alignment, and directly-usable vs rewrite-needed counts.
    """
    rows = list(raw_rows)
    n_total = len(rows)
    complete = [r for r in rows if _row_complete(r)]
    n_complete = len(complete)

    # ---- Within-condition fixed-q* validity ----
    n_within_valid = sum(
        1 for r in complete if all(_row_within_condition_valid(r).values())
    )

    # ---- Terminal query cross-modality equality ----
    n_terminal_exact = 0
    n_terminal_normalized = 0
    for row in complete:
        q_mm = str(row["multimodal_dialogue"][TERMINAL_FIELD])
        q_text = str(row["unimodal_dialogue"][TERMINAL_FIELD])
        if q_mm == q_text:
            n_terminal_exact += 1
        if _norm(q_mm) == _norm(q_text):
            n_terminal_normalized += 1

    # ---- Per-turn mm/text alignment ----
    per_turn: dict[str, dict] = {}
    for field in TYPE_B_FIELDS:
        n_exact = n_norm = 0
        for row in complete:
            mm_val = str(row["multimodal_dialogue"][field])
            text_val = str(row["unimodal_dialogue"][field])
            if mm_val == text_val:
                n_exact += 1
            if _norm(mm_val) == _norm(text_val):
                n_norm += 1
        per_turn[field] = {
            "n_exact": n_exact,
            "fraction_exact": (n_exact / n_complete) if n_complete else 0.0,
            "n_normalized": n_norm,
            "fraction_normalized": (n_norm / n_complete) if n_complete else 0.0,
        }

    # ---- Directly usable for 2x2 construction ----
    n_usable_exact = 0
    n_usable_normalized = 0
    for row in complete:
        mm = row["multimodal_dialogue"]
        text = row["unimodal_dialogue"]
        exact_all = all(str(mm[f]) == str(text[f]) for f in TYPE_B_FIELDS)
        norm_all = all(_norm(mm[f]) == _norm(text[f]) for f in TYPE_B_FIELDS)
        n_usable_exact += int(exact_all)
        n_usable_normalized += int(norm_all)

    return {
        "n_type_b": n_total,
        "n_rows_complete": n_complete,
        "within_condition_fixed_q": {
            "n_valid": n_within_valid,
            "note": (
                "Type-B rows carry one r3 field per modality, so safe/unsafe "
                "share q* by construction; non-emptiness verified instead."
            ),
        },
        "terminal_query_cross_modality": {
            "n_exact_match": n_terminal_exact,
            "fraction_exact_match":
                (n_terminal_exact / n_complete) if n_complete else 0.0,
            "n_normalized_match": n_terminal_normalized,
            "fraction_normalized_match":
                (n_terminal_normalized / n_complete) if n_complete else 0.0,
        },
        "per_turn_alignment": per_turn,
        "directly_usable": {
            "n_exact": n_usable_exact,
            "n_normalized": n_usable_normalized,
            "n_requiring_rewrite_exact": n_complete - n_usable_exact,
            "n_requiring_rewrite_normalized": n_complete - n_usable_normalized,
        },
    }


def run_type_b_diagnostics(split: str = "type_b") -> dict:
    """Load raw MTMCS rows and run the diagnostics (no image I/O)."""
    from causal_mllm.adapters.mtmcs import MTMCSAdapter

    adapter = MTMCSAdapter()
    raw_rows = list(adapter.load(split))
    report = diagnose_type_b_rows(raw_rows)
    report["split"] = split
    return report
