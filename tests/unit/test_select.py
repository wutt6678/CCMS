"""Unit tests for Iteration 3: candidate selection with rejection reasons.

Covers:
  * Per-record inclusion criteria and reason codes
  * MTMCS group atomicity (all-4-or-none)
  * Causal invariants (type_b shared terminal, type_a divergent terminal)
  * Accounting invariant (no silent disappearance)
  * Deterministic seeded family sampling
  * Pass-through guarantee (selection never mutates source records)
"""

from __future__ import annotations

import pytest

from causal_mllm.construction.select import (
    SelectionConfig,
    SelectionRejection,
    assert_canonical,
    build_family_review_flags,
    run_selection,
    select_candidates,
)
from causal_mllm.data.schemas import CanonicalSourceExample, Message
from causal_mllm.seeds import sha256_text

# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------

class TestTypeGuard:
    def test_raw_dict_rejected(self):
        with pytest.raises(TypeError, match="dict"):
            assert_canonical([{"source_id": "x"}])

    def test_select_candidates_rejects_raw_dicts(self):
        with pytest.raises(TypeError):
            select_candidates([{"source_id": "x"}])


# ---------------------------------------------------------------------------
# Per-record criteria
# ---------------------------------------------------------------------------

class TestPerRecordCriteria:
    def test_valid_group_accepted_by_default(self):
        group = make_mtmcs_group("type_b", 1)
        accepted, rejections = select_candidates(group)
        assert len(accepted) == 4
        assert rejections == []

    def test_too_few_turns(self):
        group = make_mtmcs_group("type_b", 1, n_turns=2)
        accepted, rejections = select_candidates(group, min_turns=3)
        assert accepted == []
        assert len(rejections) == 4
        assert all("too_few_turns" in r.reason for r in rejections)

    def test_too_many_turns(self):
        group = make_mtmcs_group("type_b", 1, n_turns=6)
        accepted, rejections = select_candidates(group, max_turns=5)
        assert accepted == []
        assert len(rejections) == 4
        assert all("too_many_turns" in r.reason for r in rejections)

    def test_text_too_long(self):
        group = make_mtmcs_group("type_b", 1)
        accepted, rejections = select_candidates(group, max_text_length=50)
        assert accepted == []
        assert all("text_too_long" in r.reason for r in rejections)

    def test_terminal_query_too_short(self):
        group = make_mtmcs_group("type_b", 1)
        config = SelectionConfig(min_terminal_query_chars=10_000)
        accepted, rejections = select_candidates(group, config)
        assert accepted == []
        assert all("terminal_query_too_short" in r.reason for r in rejections)

    def test_dataset_excluded(self):
        group = make_mtmcs_group("type_b", 1)
        config = SelectionConfig(datasets=frozenset({"cosafe"}))
        accepted, rejections = select_candidates(group, config)
        assert accepted == []
        assert all(r.reason == "dataset_excluded" for r in rejections)

    def test_setting_excluded(self):
        group = make_mtmcs_group("type_b", 1)
        config = SelectionConfig(settings=frozenset({"type_a"}))
        accepted, rejections = select_candidates(group, config)
        assert accepted == []
        assert all(r.reason == "setting_excluded" for r in rejections)

    def test_text_only_singleton_rejected_when_images_required(self):
        singleton = make_text_only_singleton()
        accepted, rejections = select_candidates([singleton])  # require_images=True
        assert accepted == []
        assert len(rejections) == 1
        assert rejections[0].reason == "no_images"

    def test_text_only_singleton_accepted_when_images_optional(self):
        singleton = make_text_only_singleton()
        accepted, rejections = select_candidates([singleton], require_images=False)
        assert len(accepted) == 1
        assert rejections == []


# ---------------------------------------------------------------------------
# MTMCS group atomicity + causal invariants
# ---------------------------------------------------------------------------

class TestGroupAtomicity:
    def test_broken_type_b_terminal_rejects_entire_group(self):
        group = make_mtmcs_group("type_b", 1, break_terminal=True)
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert len(rejections) == 4  # ALL four rejected atomically
        pair_ids = {r.pair_id for r in rejections}
        assert pair_ids == {"mtmcs:type_b:000001"}
        assert all("terminal_query_invariant_violated" in r.reason
                   for r in rejections)

    def test_type_a_equal_terminals_rejected(self):
        group = make_mtmcs_group("type_a", 2, equal_terminals=True)
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert all("terminal_query_not_divergent" in r.reason
                   for r in rejections)

    def test_valid_type_a_group_accepted(self):
        group = make_mtmcs_group("type_a", 2)
        accepted, rejections = select_candidates(group)
        assert len(accepted) == 4
        assert rejections == []

    def test_incomplete_group_rejected(self):
        group = make_mtmcs_group("type_b", 1)[:3]  # drop one condition
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert len(rejections) == 3
        assert all("group_incomplete" in r.reason for r in rejections)

    def test_one_bad_record_rejects_whole_group(self):
        """A per-record failure on one member rejects all 4."""
        group = make_mtmcs_group("type_b", 1)
        group[0].messages = group[0].messages[:2]  # mm_safe has 2 turns
        config = SelectionConfig(min_turns=2)
        accepted, rejections = select_candidates(group, config)
        assert len(accepted) == 4  # 2 turns >= min_turns=2 passes
        config_strict = SelectionConfig(min_turns=3)
        accepted, rejections = select_candidates(group, config_strict)
        assert accepted == []
        assert len(rejections) == 4

    def test_mixed_input_group_atomicity(self):
        """Rejecting one group must not affect other groups."""
        good = make_mtmcs_group("type_b", 1)
        bad = make_mtmcs_group("type_b", 2, break_terminal=True)
        accepted, rejections = select_candidates(good + bad)
        assert len(accepted) == 4
        assert {r.pair_id for r in rejections} == {"mtmcs:type_b:000002"}


# ---------------------------------------------------------------------------
# Accounting invariant
# ---------------------------------------------------------------------------

class TestAccounting:
    def test_no_silent_disappearance(self):
        inputs = (make_mtmcs_group("type_b", 1)
                  + make_mtmcs_group("type_b", 2, break_terminal=True)
                  + [make_text_only_singleton()])
        accepted, rejections = select_candidates(inputs)
        assert len(accepted) + len(rejections) == len(inputs)

    def test_rejection_record_fields(self):
        singleton = make_text_only_singleton()
        _, rejections = select_candidates([singleton])
        d = rejections[0].to_dict()
        assert d["stage"] == "selection"
        assert d["source_id"] == singleton.source_id
        assert d["source_dataset"] == "cosafe"
        # Roundtrip
        restored = SelectionRejection.from_dict(d)
        assert restored.to_dict() == d

    def test_report_accounting(self):
        inputs = make_mtmcs_group("type_b", 1) + [make_text_only_singleton()]
        result = run_selection(inputs)
        assert result.report["accounting_ok"] is True
        assert result.report["n_input"] == len(inputs)
        assert (result.report["n_accepted"] + result.report["n_rejected"]
                == result.report["n_input"])
        assert result.report["rejected_records_by_reason"] == {"no_images": 1}
        assert result.report["rejected_families_by_reason"] == {"no_images": 1}
        assert result.report["n_families_rejected"] == 1
        assert result.report["n_families_accepted"] == 1
        assert result.report["config_hash"]


# ---------------------------------------------------------------------------
# Report granularity: record-level vs family-level rejection counts
# ---------------------------------------------------------------------------

class TestReportGranularity:
    def test_one_rejected_family_counts_4_records_but_1_family(self):
        good = make_mtmcs_group("type_b", 1)
        bad = make_mtmcs_group("type_b", 2, break_terminal=True)
        report = run_selection(good + bad).report
        assert report["rejected_records_by_reason"] == {
            "terminal_query_invariant_violated": 4,
        }
        assert report["rejected_families_by_reason"] == {
            "terminal_query_invariant_violated": 1,
        }
        assert report["n_families_rejected"] == 1

    def test_not_sampled_counts_once_per_family(self):
        inputs = [r for i in range(1, 8) for r in make_mtmcs_group("type_b", i)]
        report = run_selection(
            inputs, SelectionConfig(max_families=2, seed=42)
        ).report
        assert report["rejected_records_by_reason"] == {"not_sampled": 20}
        assert report["rejected_families_by_reason"] == {"not_sampled": 5}
        assert report["n_families_rejected"] == 5

    def test_singleton_family_counts_match(self):
        singleton = make_text_only_singleton()
        report = run_selection([singleton]).report  # require_images=True
        assert report["rejected_records_by_reason"] == {"no_images": 1}
        assert report["rejected_families_by_reason"] == {"no_images": 1}


# ---------------------------------------------------------------------------
# Balance reporting: category/safety distributions + concentration warnings
# ---------------------------------------------------------------------------

class TestBalanceReporting:
    def test_distributions_reported(self):
        group = make_mtmcs_group("type_b", 1)
        report = run_selection(group).report
        assert report["accepted_by_label"] == {"safe": 2, "unsafe": 2}
        assert report["families_by_safety"] == {"mixed": 1}
        assert report["families_by_source_intent"] == {"(none)": 1}
        assert report["accepted_by_source_intent"] == {"(none)": 4}
        # Every accepted family is pending standalone-risk validation
        assert report["n_families_pending_risk_validation"] == 1
        # Below BALANCE_MIN_FAMILIES -> never warn
        assert report["balance_warnings"] == []

    def test_concentrated_intents_and_safety_warned(self):
        singletons = [
            make_text_only_singleton(source_id=f"cosafe:test:{i:06d}")
            for i in range(6)
        ]
        report = run_selection(
            singletons, SelectionConfig(require_images=False)
        ).report
        assert report["families_by_source_intent"] == {"(none)": 6}
        assert report["families_by_safety"] == {"unsafe": 6}
        assert any("source_intent" in w for w in report["balance_warnings"])
        assert any("safety" in w for w in report["balance_warnings"])

    def test_mixed_families_do_not_trigger_safety_warning(self):
        """MTMCS pairs contain safe+unsafe records: balanced by construction."""
        groups = [r for i in range(1, 7) for r in make_mtmcs_group("type_b", i)]
        report = run_selection(groups).report
        assert report["families_by_safety"] == {"mixed": 6}
        assert all("safety" not in w for w in report["balance_warnings"])

    def test_balanced_intents_no_warning(self):
        inputs = []
        for i in range(1, 7):
            group = make_mtmcs_group("type_b", i)
            for ex in group:
                ex.source_category = f"intent_{i % 2}"
            inputs.extend(group)
        report = run_selection(inputs).report
        assert report["families_by_source_intent"] == {"intent_0": 3, "intent_1": 3}
        assert report["balance_warnings"] == []


# ---------------------------------------------------------------------------
# Group-homogeneity guards (defensive validation)
# ---------------------------------------------------------------------------

class TestGroupHomogeneityGuards:
    def test_mixed_settings_rejected(self):
        group = make_mtmcs_group("type_b", 1)
        group[1].source_setting = "type_a"
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert len(rejections) == 4
        assert all("group_inconsistent" in r.reason for r in rejections)
        assert all("setting" in r.detail for r in rejections)

    def test_mixed_datasets_rejected(self):
        """Direct guard check: grouping itself is keyed on mtmcs, so the
        dataset guard is verified at the invariant-function level."""
        from causal_mllm.construction.select import _group_invariant_reasons
        group = make_mtmcs_group("type_b", 1)
        group[2].source_dataset = "cosafe"
        reasons = _group_invariant_reasons(group)
        assert any(r.startswith("group_inconsistent:dataset") for r in reasons)

    def test_mixed_splits_rejected(self):
        group = make_mtmcs_group("type_b", 1)
        group[3].source_split = "type_a"
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert all("group_inconsistent" in r.reason for r in rejections)
        assert all("split" in r.detail for r in rejections)

    def test_mixed_pair_ids_rejected(self):
        """Direct guard check for pair_id homogeneity."""
        from causal_mllm.construction.select import _group_invariant_reasons
        group = make_mtmcs_group("type_b", 1)
        group[0].metadata["pair_id"] = "mtmcs:type_b:999999"
        # select_candidates re-groups by pair_id, so test the guard directly
        reasons = _group_invariant_reasons(group)
        assert any(r.startswith("group_inconsistent:pair_id") for r in reasons)

    def test_duplicate_source_ids_rejected(self):
        group = make_mtmcs_group("type_b", 1)
        group[1].source_id = group[0].source_id
        accepted, rejections = select_candidates(group)
        assert accepted == []
        assert all("duplicate_source_ids" in r.reason for r in rejections)

    def test_homogeneous_group_still_passes(self):
        group = make_mtmcs_group("type_b", 1)
        accepted, rejections = select_candidates(group)
        assert len(accepted) == 4
        assert rejections == []


# ---------------------------------------------------------------------------
# Standalone terminal-risk review flags (placeholders for Iteration 6)
# ---------------------------------------------------------------------------

class TestFamilyReviewFlags:
    def test_flags_are_placeholders(self):
        group = make_mtmcs_group("type_b", 1)
        result = run_selection(group)
        flags = build_family_review_flags(result)
        assert len(flags) == 1
        flag = flags[0]
        assert flag["family_key"] == "mtmcs:type_b:000001"
        assert flag["standalone_terminal_risk"] is None
        assert flag["strict_causal_candidate"] is None
        assert flag["requires_standalone_risk_validation"] is True
        assert flag["n_records"] == 4

    def test_type_b_flag_uses_shared_terminal(self):
        """Flag hash must equal the shared type_b terminal query hash."""
        group = make_mtmcs_group("type_b", 1)
        result = run_selection(group)
        flags = build_family_review_flags(result)
        mm_safe = next(ex for ex in group
                       if ex.metadata["modality"] == "multimodal"
                       and ex.metadata["safety"] == "safe")
        mm_unsafe = next(ex for ex in group
                         if ex.metadata["modality"] == "multimodal"
                         and ex.metadata["safety"] == "unsafe")
        assert flags[0]["terminal_query_sha256"] == sha256_text(mm_safe.terminal_query)
        assert sha256_text(mm_safe.terminal_query) == sha256_text(mm_unsafe.terminal_query)

    def test_singleton_flags(self):
        singleton = make_text_only_singleton()
        result = run_selection([singleton], SelectionConfig(require_images=False))
        flags = build_family_review_flags(result)
        assert len(flags) == 1
        assert flags[0]["family_key"] == singleton.source_id
        assert flags[0]["requires_standalone_risk_validation"] is True


# ---------------------------------------------------------------------------
# Deterministic family sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_max_families_limits_units(self):
        inputs = [r for i in range(1, 6) for r in make_mtmcs_group("type_b", i)]
        config = SelectionConfig(max_families=2, seed=42)
        accepted, rejections = select_candidates(inputs, config)
        # 2 groups x 4 records
        assert len(accepted) == 8
        assert len(rejections) == 12
        pair_ids = {ex.metadata["pair_id"] for ex in accepted}
        assert len(pair_ids) == 2
        assert all(r.reason == "not_sampled" for r in rejections)

    def test_sampling_is_deterministic(self):
        inputs = [r for i in range(1, 11) for r in make_mtmcs_group("type_b", i)]
        config = SelectionConfig(max_families=3, seed=123)
        acc1, _ = select_candidates(list(inputs), config)
        acc2, _ = select_candidates(list(inputs), config)
        assert [ex.source_id for ex in acc1] == [ex.source_id for ex in acc2]

    def test_different_seed_can_differ(self):
        inputs = [r for i in range(1, 21) for r in make_mtmcs_group("type_b", i)]
        acc1, _ = select_candidates(list(inputs), SelectionConfig(max_families=3, seed=1))
        acc2, _ = select_candidates(list(inputs), SelectionConfig(max_families=3, seed=2))
        ids1 = {ex.metadata["pair_id"] for ex in acc1}
        ids2 = {ex.metadata["pair_id"] for ex in acc2}
        assert ids1 != ids2  # 20 groups, two seeds, collision extremely unlikely

    def test_sampling_accounts_every_record(self):
        inputs = [r for i in range(1, 6) for r in make_mtmcs_group("type_b", i)]
        config = SelectionConfig(max_families=2, seed=42)
        accepted, rejections = select_candidates(inputs, config)
        assert len(accepted) + len(rejections) == len(inputs)

    def test_max_families_larger_than_eligible_keeps_all(self):
        inputs = make_mtmcs_group("type_b", 1)
        config = SelectionConfig(max_families=10)
        accepted, rejections = select_candidates(inputs, config)
        assert len(accepted) == 4
        assert rejections == []


# ---------------------------------------------------------------------------
# Pass-through guarantee: source trajectory != frozen trajectory
# ---------------------------------------------------------------------------

class TestPassThrough:
    def test_accepted_records_are_identical_objects(self):
        """Selection must never copy or mutate — accepted ARE the inputs."""
        group = make_mtmcs_group("type_b", 1)
        accepted, _ = select_candidates(group)
        for ex in accepted:
            assert any(ex is original for original in group)

    def test_records_not_mutated_by_selection(self):
        group = make_mtmcs_group("type_b", 1)
        before = [ex.to_dict() for ex in group]
        select_candidates(group)
        after = [ex.to_dict() for ex in group]
        assert before == after

    def test_no_assistant_turns_injected(self):
        """Canonical MTMCS records stay user-turns-only after selection."""
        group = make_mtmcs_group("type_b", 1)
        accepted, _ = select_candidates(group)
        for ex in accepted:
            assert all(m.role == "user" for m in ex.messages)


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------

class TestSelectionConfig:
    def test_from_config_parses_yaml_shape(self):
        config = SelectionConfig.from_config({
            "min_turns": 2,
            "settings": ["type_b"],
            "datasets": ["mtmcs", "cosafe"],
        })
        assert config.min_turns == 2
        assert config.settings == frozenset({"type_b"})
        assert config.datasets == frozenset({"mtmcs", "cosafe"})

    def test_from_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown selection config key"):
            SelectionConfig.from_config({"bogus_option": True})

    def test_to_dict_roundtrip(self):
        config = SelectionConfig(settings=frozenset({"type_b"}))
        d = config.to_dict()
        assert d["settings"] == ["type_b"]
        restored = SelectionConfig.from_config(d)
        assert restored == config
