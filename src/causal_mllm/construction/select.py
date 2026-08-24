"""Candidate selection for causal family construction (Iteration 3).

Selection is a PURE, PASS-THROUGH filter over normalized source records:

  * It never mutates a record. Accepted examples are the same objects
    that were passed in — no synthetic assistant responses, no edits.
    Source trajectory != experimental frozen trajectory; interventions
    happen only in later variant-generation iterations.
  * It never loses a record silently. The accounting invariant is
    ``n_input == n_accepted + n_rejected`` and is asserted before any
    result is returned.
  * MTMCS records are selected at the atomic 4-record group level
    (keyed by ``metadata['pair_id']``). A group is accepted only if
    ALL four conditions pass; otherwise all four are rejected with
    machine-readable reasons.

Rejection reason codes (stable, machine-readable):

  dataset_excluded                     source dataset not in config.datasets
  setting_excluded                     source_setting not in config.settings
  too_few_turns                        num_turns < min_turns
  too_many_turns                       num_turns > max_turns
  text_too_long                        total text chars > max_text_length
  terminal_query_too_short             terminal query below char minimum
  no_images                            vision required but record has none
  group_incomplete                     MTMCS group missing conditions
  terminal_query_invariant_violated    type_b safe/unsafe terminals differ
  terminal_query_not_divergent         type_a safe/unsafe terminals equal
  not_sampled                          eligible but not drawn by seeded sample
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from causal_mllm.data.schemas import CanonicalSourceExample
from causal_mllm.seeds import config_hash, get_git_commit, sha256_text

# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Selection configuration and rejection manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionConfig:
    """Inclusion/exclusion criteria for candidate selection.

    Attributes:
        min_turns: Minimum number of messages per record.
        max_turns: Maximum number of messages per record.
        require_images: Require vision support for multimodal candidates.
            MTMCS unimodal (text) records satisfy this via their multimodal
            sibling in the same pair. Records from text-only datasets
            (cosafe, mtid) are rejected under this flag.
        max_text_length: Maximum total text characters per record.
        min_terminal_query_chars: Minimum terminal query length; rejects
            degenerate/empty terminal queries.
        datasets: Allowed source_dataset values (None = all).
        settings: Allowed source_setting values (None = all), e.g.
            ``frozenset({"type_b"})`` to keep only the causal gold standard.
        seed: Seed for deterministic family sampling.
        max_families: Maximum number of family units to keep after
            filtering (None = keep all). MTMCS groups count as one unit.
    """
    min_turns: int = 3
    max_turns: int = 8
    require_images: bool = True
    max_text_length: int = 5000
    min_terminal_query_chars: int = 10
    datasets: Optional[frozenset[str]] = None
    settings: Optional[frozenset[str]] = None
    seed: int = 42
    max_families: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "require_images": self.require_images,
            "max_text_length": self.max_text_length,
            "min_terminal_query_chars": self.min_terminal_query_chars,
            "datasets": sorted(self.datasets) if self.datasets else None,
            "settings": sorted(self.settings) if self.settings else None,
            "seed": self.seed,
            "max_families": self.max_families,
        }

    @classmethod
    def from_config(cls, config: dict) -> "SelectionConfig":
        """Build from the ``selection`` section of a generation YAML."""
        fields = {f for f in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {}
        for key, value in (config or {}).items():
            if key not in fields:
                raise ValueError(f"Unknown selection config key: '{key}'")
            if key in ("datasets", "settings") and value is not None:
                value = frozenset(value)
            kwargs[key] = value
        return cls(**kwargs)


@dataclass
class SelectionRejection:
    """Machine-readable rejection record. No record disappears silently."""
    source_id: str
    reason: str
    stage: str = "selection"
    detail: str = ""
    pair_id: Optional[str] = None
    source_dataset: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "stage": self.stage,
            "reason": self.reason,
            "detail": self.detail,
            "pair_id": self.pair_id,
            "source_dataset": self.source_dataset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelectionRejection":
        return cls(
            source_id=d["source_id"],
            reason=d["reason"],
            stage=d.get("stage", "selection"),
            detail=d.get("detail", ""),
            pair_id=d.get("pair_id"),
            source_dataset=d.get("source_dataset"),
        )


@dataclass
class SelectionResult:
    """Output of candidate selection with full accounting."""
    accepted: list[CanonicalSourceExample] = field(default_factory=list)
    rejections: list[SelectionRejection] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def verify_accounting(self, n_input: int) -> None:
        """Assert that no record disappeared without a rejection record."""
        n_accounted = len(self.accepted) + len(self.rejections)
        if n_accounted != n_input:
            raise AssertionError(
                f"Selection accounting mismatch: input={n_input}, "
                f"accepted={len(self.accepted)}, "
                f"rejected={len(self.rejections)} "
                f"(accounted={n_accounted})"
            )


# ---------------------------------------------------------------------------
# Family units: MTMCS records are grouped, everything else is singleton
# ---------------------------------------------------------------------------

@dataclass
class _FamilyUnit:
    """One selection unit: an MTMCS 4-record group or a singleton."""
    key: str  # pair_id for MTMCS groups, source_id for singletons
    records: list[CanonicalSourceExample]
    is_group: bool


def _group_key(ex: CanonicalSourceExample) -> Optional[str]:
    """Return the MTMCS pair_id if this record belongs to a group."""
    if ex.source_dataset == "mtmcs":
        pair_id = ex.metadata.get("pair_id")
        if pair_id:
            return str(pair_id)
    return None


def _build_units(examples: Sequence[CanonicalSourceExample]) -> list[_FamilyUnit]:
    """Partition examples into family units, preserving input order."""
    units: list[_FamilyUnit] = []
    group_index: dict[str, int] = {}
    for ex in examples:
        key = _group_key(ex)
        if key is None:
            units.append(_FamilyUnit(key=ex.source_id, records=[ex], is_group=False))
        elif key in group_index:
            units[group_index[key]].records.append(ex)
        else:
            group_index[key] = len(units)
            units.append(_FamilyUnit(key=key, records=[ex], is_group=True))
    return units


def group_into_family_units(
    examples: Sequence[CanonicalSourceExample],
) -> list[tuple[str, list[CanonicalSourceExample]]]:
    """Public grouping helper: (family_key, records) per family unit.

    MTMCS records are grouped by ``metadata['pair_id']``; every other
    record forms its own singleton unit. Used by downstream construction
    stages (atom extraction, family skeletons).
    """
    return [(unit.key, list(unit.records)) for unit in _build_units(examples)]


# ---------------------------------------------------------------------------
# Per-record filters
# ---------------------------------------------------------------------------

def _record_reasons(ex: CanonicalSourceExample, config: SelectionConfig) -> list[str]:
    """Evaluate per-record inclusion criteria. Returns reason codes."""
    reasons: list[str] = []

    if config.datasets is not None and ex.source_dataset not in config.datasets:
        reasons.append("dataset_excluded")
    if config.settings is not None and ex.source_setting not in config.settings:
        reasons.append("setting_excluded")

    if ex.num_turns < config.min_turns:
        reasons.append(f"too_few_turns:{ex.num_turns}<{config.min_turns}")
    if ex.num_turns > config.max_turns:
        reasons.append(f"too_many_turns:{ex.num_turns}>{config.max_turns}")

    total_text = sum(len(m.text or "") for m in ex.messages)
    if total_text > config.max_text_length:
        reasons.append(f"text_too_long:{total_text}>{config.max_text_length}")

    if len((ex.terminal_query or "").strip()) < config.min_terminal_query_chars:
        reasons.append(
            f"terminal_query_too_short:{len((ex.terminal_query or '').strip())}"
            f"<{config.min_terminal_query_chars}"
        )

    return reasons


def _reason_code(reason: str) -> str:
    """Strip the numeric detail suffix from a reason for counting."""
    return reason.split(":", 1)[0]


def _image_reason(
    ex: CanonicalSourceExample,
    config: SelectionConfig,
    unit: _FamilyUnit,
) -> Optional[str]:
    """Vision-requirement check.

    MTMCS unimodal records inherit vision support from their multimodal
    sibling in the same pair. Records from text-only datasets carry no
    vision at all and are rejected when images are required.
    """
    if not config.require_images:
        return None
    if ex.has_images:
        return None
    if unit.is_group:
        if any(r.has_images for r in unit.records):
            return None  # sibling carries the image
        return "no_images"
    return "no_images"


# ---------------------------------------------------------------------------
# Group-level causal invariants (MTMCS)
# ---------------------------------------------------------------------------

_GROUP_CONDITIONS = frozenset({
    "multimodal:safe",
    "multimodal:unsafe",
    "unimodal:safe",
    "unimodal:unsafe",
})


def _group_invariant_reasons(records: list[CanonicalSourceExample]) -> list[str]:
    """Verify MTMCS group-level invariants.

    Homogeneity guards (defensive — the adapter constructs groups this way,
    but selection must not trust it silently). All members must share:
    pair_id, source_dataset, source_split, source_setting; source IDs must
    be unique. Only after these hold does the first record's setting safely
    represent the whole group.

    Causal invariants:
    - Group must contain all four conditions exactly once.
    - type_b: safe/unsafe terminal queries must be IDENTICAL per modality
      (shared terminal query = the causal experiment gold standard).
    - type_a: safe/unsafe terminal queries must DIFFER (divergence at the
      terminal turn is what defines type_a).

    NOTE: a shared type_b terminal query is NOT assumed to be
    history-dependent. Standalone terminal-risk validation happens later
    (see build_family_review_flags / Iteration 6).
    """
    reasons: list[str] = []

    # ---- Homogeneity guards ----
    datasets = {r.source_dataset for r in records}
    if len(datasets) > 1:
        reasons.append(f"group_inconsistent:dataset:{sorted(datasets)}")
    splits = {r.source_split for r in records}
    if len(splits) > 1:
        reasons.append(f"group_inconsistent:split:{sorted(map(str, splits))}")
    settings = {r.source_setting for r in records}
    if len(settings) > 1:
        reasons.append(f"group_inconsistent:setting:{sorted(settings)}")
    pair_ids = {str(r.metadata.get("pair_id")) for r in records}
    if len(pair_ids) > 1:
        reasons.append(f"group_inconsistent:pair_id:{sorted(pair_ids)}")
    source_ids = [r.source_id for r in records]
    if len(source_ids) != len(set(source_ids)):
        reasons.append("duplicate_source_ids")
    if reasons:
        return reasons  # heterogeneous group: causal invariants undefined

    # ---- Completeness ----
    conditions = {}
    for r in records:
        cond = f"{r.metadata.get('modality')}:{r.metadata.get('safety')}"
        conditions[cond] = r
    if set(conditions) != _GROUP_CONDITIONS or len(records) != len(_GROUP_CONDITIONS):
        reasons.append(f"group_incomplete:{sorted(conditions)}")
        return reasons  # cannot evaluate invariants on a partial group

    # ---- Terminal-query causal invariants ----
    setting = records[0].source_setting  # safe: homogeneity verified above
    for modality, field_name in (("multimodal", "mm"), ("unimodal", "text")):
        safe_q = conditions[f"{modality}:safe"].terminal_query
        unsafe_q = conditions[f"{modality}:unsafe"].terminal_query
        same = sha256_text(safe_q) == sha256_text(unsafe_q)
        if setting == "type_b" and not same:
            reasons.append(f"terminal_query_invariant_violated:{field_name}")
        elif setting == "type_a" and same:
            reasons.append(f"terminal_query_not_divergent:{field_name}")

    return reasons


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_candidates(
    examples: Sequence[CanonicalSourceExample],
    config: SelectionConfig | None = None,
    *,
    min_turns: int | None = None,
    max_turns: int | None = None,
    require_images: bool | None = None,
    max_text_length: int | None = None,
) -> tuple[list[CanonicalSourceExample], list[SelectionRejection]]:
    """Conservative candidate selector for causal family construction.

    Applies inclusion/exclusion criteria and returns ``(accepted,
    rejections)``. Every input record appears in exactly one of the two
    outputs; nothing is silently dropped. Accepted records are the exact
    input objects (pass-through — selection never mutates source data).

    MTMCS records are selected atomically per ``pair_id`` group: either
    all four conditions are accepted or all four are rejected.

    Args:
        examples: Normalized source examples (CanonicalSourceExample only).
        config: Full selection configuration. If None, a default config is
            built from the keyword overrides below.
        min_turns: Override config.min_turns (keyword convenience).
        max_turns: Override config.max_turns.
        require_images: Override config.require_images.
        max_text_length: Override config.max_text_length.

    Returns:
        Tuple of (accepted examples, rejection records).

    Raises:
        TypeError: If any input is not a CanonicalSourceExample.
        AssertionError: If the accounting invariant is violated.
    """
    assert_canonical(examples)

    if config is None:
        config = SelectionConfig()
    if min_turns is not None:
        config = _replace(config, min_turns=min_turns)
    if max_turns is not None:
        config = _replace(config, max_turns=max_turns)
    if require_images is not None:
        config = _replace(config, require_images=require_images)
    if max_text_length is not None:
        config = _replace(config, max_text_length=max_text_length)

    units = _build_units(examples)

    # ---- Phase 1: criterion filtering (whole-unit decisions) ----
    eligible: list[_FamilyUnit] = []
    rejections: list[SelectionRejection] = []
    for unit in units:
        reasons = _evaluate_unit(unit, config)
        if reasons:
            for r in unit.records:
                rejections.append(_make_rejection(r, reasons))
        else:
            eligible.append(unit)

    # ---- Phase 2: deterministic family sampling ----
    if config.max_families is not None and len(eligible) > config.max_families:
        eligible = sorted(eligible, key=lambda u: u.key)
        rng = random.Random(config.seed)
        sampled = set(rng.sample([u.key for u in eligible], config.max_families))
        kept: list[_FamilyUnit] = []
        for unit in eligible:
            if unit.key in sampled:
                kept.append(unit)
            else:
                for r in unit.records:
                    rejections.append(_make_rejection(r, ["not_sampled"]))
        eligible = kept

    # ---- Assemble result; preserve original input order ----
    accepted_ids = {id(r) for unit in eligible for r in unit.records}
    accepted = [ex for ex in examples if id(ex) in accepted_ids]

    result = SelectionResult(accepted=accepted, rejections=rejections)
    result.verify_accounting(len(examples))
    return result.accepted, result.rejections


def _replace(config: SelectionConfig, **kwargs) -> SelectionConfig:
    """Frozen-dataclass copy helper."""
    current = {f: getattr(config, f) for f in config.__dataclass_fields__}
    current.update(kwargs)
    return SelectionConfig(**current)


def _evaluate_unit(unit: _FamilyUnit, config: SelectionConfig) -> list[str]:
    """Return rejection reasons for a whole unit (empty = eligible)."""
    if unit.is_group:
        reasons = _group_invariant_reasons(unit.records)
        if reasons:
            return reasons

    reasons: list[str] = []
    for r in unit.records:
        reasons.extend(_record_reasons(r, config))
        image_reason = _image_reason(r, config, unit)
        if image_reason:
            reasons.append(image_reason)
    # De-duplicate while preserving order (same reason on several records)
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return unique


def _make_rejection(
    ex: CanonicalSourceExample, reasons: list[str]
) -> SelectionRejection:
    return SelectionRejection(
        source_id=ex.source_id,
        reason=";".join(_reason_code(r) for r in reasons),
        detail="; ".join(reasons),
        pair_id=_group_key(ex),
        source_dataset=ex.source_dataset,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# A family set is flagged when a single category/label covers at least this
# share of accepted families (and there are enough families to matter).
# Deliberately simple: report + warn, no stratified resampling yet.
BALANCE_CONCENTRATION_THRESHOLD = 0.8
BALANCE_MIN_FAMILIES = 5


def _family_key_of(ex: CanonicalSourceExample) -> str:
    """Family key used for family-level aggregation."""
    return str(ex.metadata.get("pair_id") or ex.source_id)


def _concentration_warnings(counts: dict[str, int], dimension: str) -> list[str]:
    """Warn when one value dominates a distribution."""
    total = sum(counts.values())
    if total < BALANCE_MIN_FAMILIES or not counts:
        return []
    warnings: list[str] = []
    for value, n in sorted(counts.items()):
        share = n / total
        if share >= BALANCE_CONCENTRATION_THRESHOLD:
            warnings.append(
                f"concentration:{dimension} '{value}' covers "
                f"{n}/{total} families ({share:.0%})"
            )
    return warnings


def build_selection_report(
    n_input: int,
    result: SelectionResult,
    config: SelectionConfig,
) -> dict:
    """Build a machine-readable selection report with full provenance.

    Rejection counts are reported at BOTH granularities:
      * ``rejected_records_by_reason`` — one count per rejection row. A
        rejected MTMCS family contributes 4 (one per condition record).
      * ``rejected_families_by_reason`` — one count per family unit
        (pair_id for MTMCS groups, source_id for singletons). A rejected
        MTMCS family contributes exactly 1.
    """
    import datetime

    result.verify_accounting(n_input)

    # Record-level counts (one per rejection row)
    record_counts: dict[str, int] = {}
    for rej in result.rejections:
        for code in rej.reason.split(";"):
            record_counts[code] = record_counts.get(code, 0) + 1

    # Family-level counts (one per family unit, de-duplicated per reason)
    family_reasons: dict[str, set[str]] = {}
    for rej in result.rejections:
        key = rej.pair_id or rej.source_id
        family_reasons.setdefault(key, set()).update(rej.reason.split(";"))
    family_counts: dict[str, int] = {}
    for codes in family_reasons.values():
        for code in codes:
            family_counts[code] = family_counts.get(code, 0) + 1

    dataset_counts: dict[str, int] = {}
    setting_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    family_intents: dict[str, str] = {}
    family_labels: dict[str, set] = {}
    for ex in result.accepted:
        dataset_counts[ex.source_dataset] = dataset_counts.get(ex.source_dataset, 0) + 1
        setting_counts[ex.source_setting] = setting_counts.get(ex.source_setting, 0) + 1
        # NOTE: source_category currently derives from MTMCS unsafe_intent,
        # i.e. it is a scenario/intent description, NOT a stable safety
        # taxonomy. Reported as *_by_source_intent to avoid overstating
        # category diversity. A normalized taxonomy (cyber, privacy,
        # physical_harm, fraud, ...) arrives with the annotation work.
        intent = ex.source_category or "(none)"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
        # Family-level distributions (for the 20-family balance check)
        fkey = _family_key_of(ex)
        family_intents[fkey] = intent
        family_labels.setdefault(fkey, set()).add(ex.label)

    family_intent_counts: dict[str, int] = {}
    for intent in family_intents.values():
        family_intent_counts[intent] = family_intent_counts.get(intent, 0) + 1
    # A family's safety mix: balanced (has both labels), or its single label
    family_safety_counts: dict[str, int] = {}
    for labels in family_labels.values():
        key = "mixed" if len(labels) > 1 else next(iter(labels))
        family_safety_counts[key] = family_safety_counts.get(key, 0) + 1

    n_families = len({_family_key_of(ex) for ex in result.accepted})

    balance_warnings = (
        _concentration_warnings(family_intent_counts, "source_intent")
        # 'mixed' families contain both safe and unsafe records and are
        # balanced by construction — only single-label families can
        # concentrate a safety category.
        + _concentration_warnings(
            {k: v for k, v in family_safety_counts.items() if k != "mixed"},
            "safety",
        )
    )

    return {
        "iteration": 3,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "config": config.to_dict(),
        "config_hash": config_hash(config.to_dict()),
        "n_input": n_input,
        "n_accepted": len(result.accepted),
        "n_rejected": len(result.rejections),
        "n_families_accepted": n_families,
        "n_families_rejected": len(family_reasons),
        # Every accepted family still needs standalone terminal-risk
        # validation before it may be called a strict causal candidate.
        "n_families_pending_risk_validation": n_families,
        "accounting_ok": True,
        "rejected_records_by_reason": dict(sorted(record_counts.items())),
        "rejected_families_by_reason": dict(sorted(family_counts.items())),
        "accepted_by_dataset": dict(sorted(dataset_counts.items())),
        "accepted_by_setting": dict(sorted(setting_counts.items())),
        "accepted_by_source_intent": dict(sorted(intent_counts.items())),
        "accepted_by_label": dict(sorted(label_counts.items())),
        "families_by_source_intent": dict(sorted(family_intent_counts.items())),
        "families_by_safety": dict(sorted(family_safety_counts.items())),
        "balance_warnings": balance_warnings,
    }


def build_family_review_flags(result: SelectionResult) -> list[dict]:
    """Per-family review flags for standalone terminal-risk validation.

    A terminal query that is identical across a type_b pair is NOT
    automatically history-dependent: Risk(q*) alone may already be high
    (an obviously unsafe query needs no history). Until a judge estimates
    P(Y_unsafe | q*) — Iteration 6 — every family carries:

        standalone_terminal_risk: null
        strict_causal_candidate: null
        requires_standalone_risk_validation: true

    The strict causal subset later requires Risk(q*) low/ambiguous while
    P(Y_unsafe | H_unsafe, q*) changes materially.
    """
    flags: list[dict] = []
    for unit in _build_units(result.accepted):
        ref = unit.records[0]
        # Prefer the multimodal safe record's terminal when available
        terminal = next(
            (r.terminal_query for r in unit.records
             if r.metadata.get("modality") == "multimodal"
             and r.metadata.get("safety") == "safe"),
            ref.terminal_query,
        )
        flags.append({
            "family_key": unit.key,
            "source_dataset": ref.source_dataset,
            "source_setting": ref.source_setting,
            "n_records": len(unit.records),
            "terminal_query_sha256": sha256_text(terminal),
            "standalone_terminal_risk": None,
            "strict_causal_candidate": None,
            "requires_standalone_risk_validation": True,
        })
    return flags


def run_selection(
    examples: Sequence[CanonicalSourceExample],
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Full selection pass returning a SelectionResult with the report."""
    accepted, rejections = select_candidates(examples, config)
    result = SelectionResult(accepted=accepted, rejections=rejections)
    result.report = build_selection_report(len(examples), result,
                                           config or SelectionConfig())
    return result
