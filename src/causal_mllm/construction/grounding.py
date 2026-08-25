"""Canonical-q grounding flags (Iteration-5 repair).

Exact-string invariance of q* across the six variants is NECESSARY but
NOT SUFFICIENT: the same string can receive different interpretations
across conditions. A q* referring to "these helmets" is under-specified
in history_reset (which has no image), introducing the confound

    image treatment -> reference resolvability -> Y

instead of the intended

    image semantic evidence -> safety state -> Y.

``flag_grounding_issues`` scans a built family for image-deictic
references ("these", "this photo", "shown here", ...) in:

  * the canonical q* itself, and
  * every message of the text-only conditions (neutral, text_only,
    history_reset), where no image is present to resolve them.

It also flags built families whose grounding validation targets are
still unresolved (null). The human review of the Scale-A smoke set
must produce zero flags; the placeholder harmonization (adopting the
multimodal terminal verbatim) is expected to flag, which is exactly
why it is not research-valid evidence.
"""

from __future__ import annotations

import re

from causal_mllm.construction.harmonize import GROUNDING_VALIDATION_TARGETS
from causal_mllm.data.schemas import CausalFamily

# Conservative image-deictic patterns: references that can only be
# resolved when the shared image is present.
DEICTIC_IMAGE_PATTERNS = (
    (re.compile(r"\bthese\b", re.I), "'these' (plural deictic)"),
    (re.compile(r"\bthis (photo|image|picture)\b", re.I),
     "'this photo/image/picture'"),
    (re.compile(r"\bin this (photo|image|picture|scene)\b", re.I),
     "'in this photo/image/scene'"),
    (re.compile(r"\bshown here\b", re.I), "'shown here'"),
    (re.compile(r"\bvisible here\b", re.I), "'visible here'"),
    (re.compile(r"\bthe photo\b", re.I), "'the photo'"),
)

# Conditions that carry NO image: any deictic reference there is an
# unresolved grounding issue.
_TEXT_ONLY_CONDITIONS = ("neutral", "text_only", "history_reset")


def _deictic_hits(text: str | None) -> list[str]:
    if not text:
        return []
    return [label for pattern, label in DEICTIC_IMAGE_PATTERNS
            if pattern.search(text)]


def flag_grounding_issues(family: CausalFamily) -> list[dict]:
    """Return grounding flags for one (built) family; empty == clean.

    Each flag: {"scope": ..., "detail": ...} where scope is
    'canonical_q', a text-only variant name, or 'grounding_targets'.
    """
    flags: list[dict] = []

    block = (family.validation or {}).get("terminal_harmonization") or {}
    canonical_q = block.get("canonical_q")
    for hit in _deictic_hits(canonical_q):
        flags.append({
            "scope": "canonical_q",
            "detail": f"canonical q* contains unresolved image "
                      f"reference {hit}",
        })

    for name in _TEXT_ONLY_CONDITIONS:
        variant = family.variants.get(name)
        if variant is None:
            continue
        for message in variant.messages:
            for hit in _deictic_hits(message.text):
                flags.append({
                    "scope": name,
                    "detail": f"turn {message.turn_index} contains "
                              f"unresolved image reference {hit} in a "
                              f"condition without images",
                })

    # Built families must not carry unresolved grounding judgments.
    if family.variants:
        unresolved = [t for t in GROUNDING_VALIDATION_TARGETS
                      if block.get(t) is None]
        if unresolved:
            flags.append({
                "scope": "grounding_targets",
                "detail": f"grounding validation targets unresolved: "
                          f"{unresolved}",
            })

    return flags
