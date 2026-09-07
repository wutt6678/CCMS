"""One truncation standard, defined once, imported by every full-panel gate.

Iteration 11 exposed a contradiction between two gates in this repository. The
completion gate (``scripts/iter11_replay_checks.py``) accepted a full panel at
an overall truncation rate up to 0.02 with a variant-rate spread up to 0.05;
the evaluation panel gate (``evaluation/gate.py``) required zero. The frozen
Qwen3.5-9B reference truncated nothing, so both had always agreed and nothing
ever compared them. Three of the four Iteration 11 checkpoints truncate -- 7, 1
and 4 cells of 600 -- and each passed the completion gate only to be refused by
the evaluation gate, after roughly nine hours of judging had been spent per arm.

Both now import ``causal_mllm.replay.truncation``. These tests pin:

* the boundary is inclusive, and failing it by ONE cell fails the panel;
* low overall truncation concentrated in one variant still fails, because the
  estimands are differences BETWEEN variants;
* truncation is counted identically everywhere;
* no caller, script or CLI flag can supply its own threshold;
* the measurement is recorded in every validation report;
* the 12-family ELIGIBILITY gate still stops at zero truncation -- it is a
  different screening contract and was deliberately not relaxed;
* the sealed Iteration 9/10 panels return the verdict they always returned.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.gate import validate_panel
from causal_mllm.replay import truncation as shared
from causal_mllm.replay.confirmatory import validate_gate_entry
from causal_mllm.replay.truncation import (
    MAX_TRUNCATION_RATE,
    MAX_VARIANT_SPREAD,
    exceeds_registered_thresholds,
    is_truncated,
    measure_truncation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 100 families x 6 variants = 600 records, the Iteration 11 panel size. At
#: this size the registered rate boundary is exactly 12 records and each
#: variant's rate moves in steps of 0.01, so a boundary can be hit exactly and
#: exceeded by the smallest amount one cell allows.
N_FAMILIES = 100
N_VARIANTS = len(ALL_VARIANT_NAMES)
N_RECORDS = N_FAMILIES * N_VARIANTS


def _panel(tmp_path, per_variant, *, n_families=N_FAMILIES,
           run_id="test_run", flag="hit_max_new_tokens"):
    """A replay run whose truncation is an exact per-variant pattern.

    Args:
        per_variant: variant name -> how many of that variant's cells reached
            the cap. The first ``k`` families of the variant are the ones
            truncated, so the pattern is deterministic.
        flag: which provenance flag carries the truncation, so the two flags
            the codebase writes can be exercised separately.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = list(ALL_VARIANT_NAMES)[:N_VARIANTS]

    records = []
    for fam_idx in range(n_families):
        family_id = f"CMST_{fam_idx:06d}"
        for variant in variants:
            truncated = fam_idx < per_variant.get(variant, 0)
            rec = {
                "run_id": run_id,
                "family_id": family_id,
                "variant": variant,
                "model_revision": "abc123",
                "finish_reason": "length" if truncated else "eos",
                "response": f"Response for {family_id}/{variant}",
                "output_token_count": 1536 if truncated else 400,
                "max_new_tokens": 1536,
                "hit_max_new_tokens": False,
                "truncated": False,
            }
            rec[flag] = truncated
            records.append(rec)

    with (run_dir / "replay_outputs.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    (run_dir / "replay_failures.jsonl").write_text("", encoding="utf-8")
    report = {
        "run_id": run_id,
        "n_families": n_families,
        "provenance": {
            "revision_pinned": True,
            "git_dirty": False,
            "requested_model_revision": "abc123",
            "resolved_model_revision": "abc123",
        },
    }
    with (run_dir / "replay_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f)
    return run_dir


def _spread(per_variant) -> float:
    """The variant-rate spread, over ALL six variants.

    A pattern naming only the variants that truncated is the natural way to
    write one down, and every variant it omits is clean, so the omitted ones
    count as zero rather than dropping out of the min. Without that, a single
    variant truncated in five of its cells would look like a spread of zero.
    """
    rates = [per_variant.get(v, 0) / N_FAMILIES for v in ALL_VARIANT_NAMES]
    return max(rates) - min(rates)


def _total(per_variant) -> int:
    return sum(per_variant.values())


# ---------------------------------------------------------------------------
# The counting rule: one definition, honoured by every gate
# ---------------------------------------------------------------------------

class TestOneCountingRule:
    def test_a_cell_that_reached_the_cap_is_truncated(self):
        assert is_truncated({"hit_max_new_tokens": True}) is True

    def test_the_provenance_flag_alone_is_enough(self):
        # The Iteration 11 per-record provenance schema names `truncated`; the
        # replay backend writes `hit_max_new_tokens`. Counting one in one gate
        # and the other in another is how two gates came to disagree about the
        # same file.
        assert is_truncated({"truncated": True}) is True
        assert is_truncated({"hit_max_new_tokens": False,
                             "truncated": True}) is True

    def test_neither_flag_is_not_a_truncation(self):
        assert is_truncated({}) is False
        assert is_truncated({"hit_max_new_tokens": False,
                             "truncated": False}) is False

    def test_a_truthy_non_boolean_does_not_count(self):
        # Identity against True, not truthiness: a provenance field carrying 1
        # or "false" must not become a truncation in one gate and not in
        # another, so it is not allowed to become one at all.
        for value in (1, "true", None, 0):
            assert is_truncated({"hit_max_new_tokens": value}) is False
            assert is_truncated({"truncated": value}) is False

    def test_the_measurement_counts_both_flags_once(self):
        records = [
            {"variant": "neutral", "hit_max_new_tokens": True},
            {"variant": "neutral", "truncated": True},
            {"variant": "neutral", "hit_max_new_tokens": True,
             "truncated": True},
            {"variant": "neutral"},
        ]
        assert measure_truncation(records)["n_truncated"] == 3


# ---------------------------------------------------------------------------
# The boundaries: inclusive, and one cell is enough to fail
# ---------------------------------------------------------------------------

class TestRegisteredBoundaries:
    def test_zero_truncation_passes(self, tmp_path):
        run_dir = _panel(tmp_path, {})
        panel, records = validate_panel(run_dir,
                                        expected_n_families=N_FAMILIES)
        assert panel.n_records == N_RECORDS
        assert len(records) == N_RECORDS
        assert panel.truncation["n_truncated"] == 0
        assert panel.truncation["passed"] is True

    def test_exactly_two_percent_overall_passes(self, tmp_path):
        # 12 of 600 = 0.02 exactly, spread flat, so the inclusive boundary is
        # on the passing side.
        per_variant = {v: 2 for v in ALL_VARIANT_NAMES}
        assert _total(per_variant) == 12
        assert _total(per_variant) / N_RECORDS == MAX_TRUNCATION_RATE
        assert _spread(per_variant) == 0.0
        panel, _ = validate_panel(_panel(tmp_path, per_variant),
                                  expected_n_families=N_FAMILIES)
        assert panel.truncation["overall_rate"] == MAX_TRUNCATION_RATE
        assert panel.truncation["passed"] is True

    def test_exactly_five_point_spread_passes(self, tmp_path):
        # One variant at 5/100 = 0.05, the rest clean: spread exactly at the
        # boundary, overall 5/600 well inside the rate boundary.
        per_variant = {ALL_VARIANT_NAMES[0]: 5}
        assert _spread(per_variant) == MAX_VARIANT_SPREAD
        assert _total(per_variant) / N_RECORDS < MAX_TRUNCATION_RATE
        panel, _ = validate_panel(_panel(tmp_path, per_variant),
                                  expected_n_families=N_FAMILIES)
        assert panel.truncation["max_variant_spread"] == MAX_VARIANT_SPREAD
        assert panel.truncation["passed"] is True

    def test_both_boundaries_at_once_pass(self, tmp_path):
        # 12 truncated overall AND a 5-point spread, both exactly at the
        # registered maximum: [5, 0, 2, 2, 2, 1].
        per_variant = dict(zip(ALL_VARIANT_NAMES, [5, 0, 2, 2, 2, 1]))
        assert _total(per_variant) == 12
        assert _total(per_variant) / N_RECORDS == MAX_TRUNCATION_RATE
        assert _spread(per_variant) == MAX_VARIANT_SPREAD
        panel, _ = validate_panel(_panel(tmp_path, per_variant),
                                  expected_n_families=N_FAMILIES)
        assert panel.truncation["passed"] is True
        assert exceeds_registered_thresholds(panel.truncation) is False

    def test_one_cell_past_the_rate_boundary_fails(self, tmp_path):
        # The same spread, one more cell: 13/600 > 0.02. This is the smallest
        # amount by which the rate boundary can be exceeded on this panel.
        per_variant = dict(zip(ALL_VARIANT_NAMES, [5, 0, 2, 2, 2, 2]))
        assert _total(per_variant) == 13
        assert _spread(per_variant) == MAX_VARIANT_SPREAD
        with pytest.raises(EvaluationError) as exc:
            validate_panel(_panel(tmp_path, per_variant),
                           expected_n_families=N_FAMILIES)
        assert "exceeds the registered maximum 0.02" in str(exc.value)
        assert "13 of 600" in str(exc.value)

    def test_concentration_past_the_spread_boundary_fails(self, tmp_path):
        # The SAME total as the passing case above -- 12 cells, overall rate
        # exactly 0.02 -- but moved into one variant: [6, 0, 2, 2, 1, 1].
        # Overall truncation is inside the tolerance and the panel still fails,
        # because the estimands are differences between variants and truncation
        # concentrated in one variant biases them.
        per_variant = dict(zip(ALL_VARIANT_NAMES, [6, 0, 2, 2, 1, 1]))
        assert _total(per_variant) == 12
        assert _total(per_variant) / N_RECORDS == MAX_TRUNCATION_RATE
        assert _spread(per_variant) > MAX_VARIANT_SPREAD
        with pytest.raises(EvaluationError) as exc:
            validate_panel(_panel(tmp_path, per_variant),
                           expected_n_families=N_FAMILIES)
        message = str(exc.value)
        assert "spread across variants" in message
        assert "concentrated in some variants" in message
        # The rate boundary was NOT the problem, and the report must not say
        # it was.
        assert "exceeds the registered maximum 0.02" not in message

    def test_exceeding_both_boundaries_names_both(self, tmp_path):
        per_variant = {ALL_VARIANT_NAMES[0]: 40}
        assert _total(per_variant) / N_RECORDS > MAX_TRUNCATION_RATE
        assert _spread(per_variant) > MAX_VARIANT_SPREAD
        with pytest.raises(EvaluationError) as exc:
            validate_panel(_panel(tmp_path, per_variant),
                           expected_n_families=N_FAMILIES)
        message = str(exc.value)
        assert "exceeds the registered maximum 0.02" in message
        assert "spread across variants" in message

    def test_the_escalation_trigger_follows_the_thresholds(self):
        # Escalating the cap is the remedy for a panel OUTSIDE the registered
        # thresholds, and for nothing else: under greedy decoding a repetition
        # loop runs to any cap, so "escalate until no cell truncates" has no
        # termination criterion.
        inside = measure_truncation(
            [{"variant": "neutral", "hit_max_new_tokens": True}]
            + [{"variant": "neutral"}] * 99)
        assert inside["n_truncated"] == 1
        assert inside["passed"] is True
        assert exceeds_registered_thresholds(inside) is False

        outside = measure_truncation(
            [{"variant": "neutral", "hit_max_new_tokens": True}]
            + [{"variant": "neutral"}] * 9)
        assert outside["overall_rate"] > MAX_TRUNCATION_RATE
        assert outside["passed"] is False
        assert exceeds_registered_thresholds(outside) is True

    def test_a_wholly_truncated_panel_still_fails(self, tmp_path):
        # The pre-existing fixture shape: every cell truncated. Under the old
        # zero-truncation rule and under the registered thresholds alike this
        # is refused; only the message changed.
        per_variant = {v: N_FAMILIES for v in ALL_VARIANT_NAMES}
        with pytest.raises(EvaluationError, match="panel gate FAILED"):
            validate_panel(_panel(tmp_path, per_variant),
                           expected_n_families=N_FAMILIES)


# ---------------------------------------------------------------------------
# One source: neither gate may restate a threshold
# ---------------------------------------------------------------------------

def _load_script(name):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestOneSourceForTheThresholds:
    def test_the_replay_checker_imports_the_shared_objects(self):
        checks = _load_script("iter11_replay_checks")
        assert checks.MAX_TRUNCATION_RATE is MAX_TRUNCATION_RATE
        assert checks.MAX_VARIANT_SPREAD is MAX_VARIANT_SPREAD

    def test_the_scale_c_checker_imports_the_shared_objects(self):
        checks = _load_script("scale_c_replay_checks")
        assert checks.MAX_TRUNCATION_RATE is MAX_TRUNCATION_RATE
        assert checks.MAX_VARIANT_SPREAD is MAX_VARIANT_SPREAD

    def test_the_evaluation_gate_imports_the_shared_objects(self):
        from causal_mllm.evaluation import gate
        assert gate.MAX_TRUNCATION_RATE is MAX_TRUNCATION_RATE
        assert gate.MAX_VARIANT_SPREAD is MAX_VARIANT_SPREAD

    def test_no_script_restates_the_literals(self):
        # Importing is not enough: a file that also defines its own copy can
        # drift from it. Neither checker may assign the numbers locally.
        for name in ("iter11_replay_checks", "scale_c_replay_checks"):
            source = (REPO_ROOT / "scripts" / f"{name}.py").read_text(
                encoding="utf-8")
            for literal in ("MAX_TRUNCATION_RATE = ", "MAX_VARIANT_SPREAD = "):
                assert literal not in source, \
                    f"{name}.py restates {literal.strip()} instead of " \
                    f"importing it from causal_mllm.replay.truncation"

    def test_the_evaluation_gate_restates_no_literal(self):
        source = (REPO_ROOT / "src" / "causal_mllm" / "evaluation"
                  / "gate.py").read_text(encoding="utf-8")
        for literal in ("MAX_TRUNCATION_RATE = ", "MAX_VARIANT_SPREAD = ",
                        "= 0.02", "= 0.05"):
            assert literal not in source, \
                f"evaluation/gate.py carries its own {literal.strip()!r}"

    def test_the_three_gates_agree_on_one_panel(self, tmp_path):
        # The Iteration 11 arms: 7, 1 and 4 truncated cells of 600. The
        # evaluation gate and the completion gate must reach the same verdict,
        # which is the whole point of sharing the constants.
        for per_variant in (
                dict(zip(ALL_VARIANT_NAMES, [2, 0, 1, 0, 3, 1])),   # 7, 2b
                {ALL_VARIANT_NAMES[0]: 1},                          # 1, ministral
                dict(zip(ALL_VARIANT_NAMES, [0, 0, 3, 1, 0, 0]))):  # 4, phi4
            total = _total(per_variant)
            assert total / N_RECORDS <= MAX_TRUNCATION_RATE
            assert _spread(per_variant) <= MAX_VARIANT_SPREAD
            panel, _ = validate_panel(_panel(tmp_path / str(total),
                                             per_variant),
                                      expected_n_families=N_FAMILIES)
            assert panel.truncation["n_truncated"] == total
            assert panel.truncation["passed"] is True
            assert exceeds_registered_thresholds(panel.truncation) is False


# ---------------------------------------------------------------------------
# No override: not per run, not per call, not from the CLI
# ---------------------------------------------------------------------------

class TestNoThresholdOverride:
    @pytest.mark.parametrize("target", [
        ("causal_mllm.evaluation.gate", "validate_panel"),
        ("causal_mllm.evaluation.runner", "run_evaluation_stage"),
    ])
    def test_no_function_accepts_a_threshold(self, target):
        module_name, func_name = target
        func = getattr(importlib.import_module(module_name), func_name)
        offenders = [
            name for name in inspect.signature(func).parameters
            if any(word in name.lower() for word in
                   ("trunc", "toler", "threshold", "spread", "max_rate"))
        ]
        assert not offenders, \
            f"{func_name} accepts {offenders}, which would let one caller " \
            f"evaluate a panel against a standard no other caller uses"

    def test_the_cli_offers_no_threshold_flag(self):
        from causal_mllm.cli import evaluate_responses
        source = Path(evaluate_responses.__file__).read_text(encoding="utf-8")
        for word in ("trunc", "toler", "spread"):
            assert word not in source.lower(), \
                f"the evaluate_responses CLI mentions {word!r}; the " \
                f"truncation standard is not a command-line choice"

    def test_an_unrecognised_threshold_flag_is_refused(self):
        from causal_mllm.cli.evaluate_responses import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--run-dir", "x", "--validated-families", "y",
                        "--max-truncation-rate", "1.0"])


# ---------------------------------------------------------------------------
# The measurement is recorded, whether or not anything was truncated
# ---------------------------------------------------------------------------

class TestRecordedEvidence:
    def test_the_report_carries_thresholds_counts_rates_spread_and_cells(
            self, tmp_path):
        per_variant = dict(zip(ALL_VARIANT_NAMES, [2, 0, 1, 0, 3, 1]))
        panel, _ = validate_panel(_panel(tmp_path, per_variant),
                                  expected_n_families=N_FAMILIES)
        measured = panel.truncation
        assert measured["thresholds"] == {
            "max_overall_rate": MAX_TRUNCATION_RATE,
            "max_variant_spread": MAX_VARIANT_SPREAD,
            "inclusive": True,
        }
        assert measured["n_records"] == N_RECORDS
        assert measured["n_truncated"] == 7
        assert measured["overall_rate"] == pytest.approx(7 / N_RECORDS)
        assert measured["max_variant_spread"] == pytest.approx(0.03)
        assert set(measured["per_variant"]) == set(ALL_VARIANT_NAMES)
        assert measured["per_variant"]["shuffle"]["n_truncated"] == 3

    def test_the_affected_cells_are_named_with_their_finish_reason(self,
                                                                   tmp_path):
        per_variant = {ALL_VARIANT_NAMES[0]: 2}
        panel, _ = validate_panel(_panel(tmp_path, per_variant),
                                  expected_n_families=N_FAMILIES)
        cells = panel.truncation["cells"]
        assert len(cells) == 2
        for cell in cells:
            assert cell["variant"] == ALL_VARIANT_NAMES[0]
            assert cell["family_id"].startswith("CMST_")
            assert cell["finish_reason"] == "length"
            assert cell["output_token_count"] == 1536

    def test_the_measurement_is_bound_into_the_report_dict(self, tmp_path):
        # run_evaluation_stage embeds PanelReport.to_dict() as
        # report["panel_gate"]["panel"], so recording it on the panel is what
        # binds the thresholds into the published evidence.
        panel, _ = validate_panel(_panel(tmp_path, {}),
                                  expected_n_families=N_FAMILIES)
        assert panel.to_dict()["truncation"]["thresholds"][
            "max_overall_rate"] == MAX_TRUNCATION_RATE

    def test_a_clean_panel_still_records_that_the_check_ran(self, tmp_path):
        # "The check ran and found nothing" and "the check did not run" have
        # to be distinguishable in a report.
        panel, _ = validate_panel(_panel(tmp_path, {}),
                                  expected_n_families=N_FAMILIES)
        measured = panel.truncation
        assert measured["n_truncated"] == 0
        assert measured["cells"] == []
        assert measured["thresholds"]["max_overall_rate"] == \
            MAX_TRUNCATION_RATE
        assert measured["violations"] == []

    def test_the_truncated_flag_alone_is_counted_and_recorded(self, tmp_path):
        run_dir = _panel(tmp_path, {ALL_VARIANT_NAMES[1]: 3},
                         flag="truncated")
        panel, _ = validate_panel(run_dir, expected_n_families=N_FAMILIES)
        assert panel.truncation["n_truncated"] == 3
        assert all(c["variant"] == ALL_VARIANT_NAMES[1]
                   for c in panel.truncation["cells"])


# ---------------------------------------------------------------------------
# Finish reasons: the cap is admitted only on a record that admits reaching it
# ---------------------------------------------------------------------------

class TestFinishReasons:
    def test_the_cap_is_accepted_on_a_truncated_record(self, tmp_path):
        # _panel already pairs finish_reason="length" with a truncation flag,
        # and the passing boundary tests above exercise it.
        panel, _ = validate_panel(
            _panel(tmp_path, {ALL_VARIANT_NAMES[0]: 1}),
            expected_n_families=N_FAMILIES)
        assert panel.truncation["cells"][0]["finish_reason"] == "length"

    def test_a_capped_record_that_denies_truncating_fails(self, tmp_path):
        run_dir = _panel(tmp_path, {})
        path = run_dir / "replay_outputs.jsonl"
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines()]
        records[0]["finish_reason"] = "length"
        path.write_text("".join(json.dumps(r) + "\n" for r in records),
                        encoding="utf-8")
        with pytest.raises(EvaluationError) as exc:
            validate_panel(run_dir, expected_n_families=N_FAMILIES)
        assert "self-contradictory provenance" in str(exc.value)

    def test_an_unknown_finish_reason_fails(self, tmp_path):
        run_dir = _panel(tmp_path, {})
        path = run_dir / "replay_outputs.jsonl"
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines()]
        records[0]["finish_reason"] = "content_filter"
        path.write_text("".join(json.dumps(r) + "\n" for r in records),
                        encoding="utf-8")
        with pytest.raises(EvaluationError) as exc:
            validate_panel(run_dir, expected_n_families=N_FAMILIES)
        assert "content_filter" in str(exc.value)

    def test_natural_terminations_are_the_shared_definition(self):
        from causal_mllm.evaluation.gate import VALID_FINISH_REASONS
        assert VALID_FINISH_REASONS == set(shared.NATURAL_FINISH_REASONS)
        assert VALID_FINISH_REASONS == {"eos", "stop"}


# ---------------------------------------------------------------------------
# The 12-family eligibility gate was NOT relaxed
# ---------------------------------------------------------------------------

class TestEligibilityGateStaysAtZero:
    def test_one_truncated_cell_still_stops_eligibility(self):
        problems = validate_gate_entry("truncation_reviewed", {
            "passed": True,
            "n_truncated": 1,
            "truncation_rate": 1 / 72,
            "max_variant_spread": 0.0,
        })
        assert any("any truncation is a protocol-level" in p
                   for p in problems), problems

    def test_a_clean_eligibility_panel_still_passes(self):
        assert validate_gate_entry("truncation_reviewed", {
            "passed": True,
            "n_truncated": 0,
            "truncation_rate": 0.0,
            "max_variant_spread": 0.0,
        }) == []

    def test_the_screening_gate_is_stricter_than_the_panel_gate(self, tmp_path):
        # ONE truncated cell. Over a 600-cell confirmatory panel that is
        # 0.17% and the panel is accepted. Over the 72-cell eligibility screen
        # the same single cell is a protocol-level STOP. That asymmetry is
        # deliberate -- eligibility screens a checkpoint before any
        # confirmatory spend, so it is allowed to be stricter than the panel
        # it is screening for -- and these two assertions are what keep it from
        # being "fixed" by accident.
        panel, _ = validate_panel(
            _panel(tmp_path, {ALL_VARIANT_NAMES[0]: 1}),
            expected_n_families=N_FAMILIES)
        assert panel.truncation["n_truncated"] == 1
        assert panel.truncation["passed"] is True

        problems = validate_gate_entry("truncation_reviewed", {
            "passed": True, "n_truncated": 1,
            "truncation_rate": 1 / 72, "max_variant_spread": 0.0})
        assert any("any truncation is a protocol-level" in p
                   for p in problems), problems


# ---------------------------------------------------------------------------
# Sealed Iteration 9/10 evidence returns the verdict it always returned
# ---------------------------------------------------------------------------

SEALED_PANELS = [
    ("outputs/scale_c/replay_runs/scale-c-100-t1536-qwen35-9b", 100),
    ("outputs/replay_runs/scale-b-2026-08-28-t1536-final-qwen35-9b", 20),
]


class TestSealedEvidenceUnchanged:
    @pytest.mark.parametrize("relative,n_families", SEALED_PANELS)
    def test_a_sealed_panel_still_passes_with_zero_recorded(self,
                                                            relative,
                                                            n_families):
        run_dir = REPO_ROOT / relative
        if not (run_dir / "replay_outputs.jsonl").exists():
            pytest.skip(f"sealed panel not present in this checkout: "
                        f"{relative}")
        panel, records = validate_panel(run_dir,
                                        expected_n_families=n_families)
        assert panel.n_families == n_families
        assert len(records) == n_families * N_VARIANTS
        assert panel.truncation["n_truncated"] == 0
        assert panel.truncation["overall_rate"] == 0.0
        assert panel.truncation["max_variant_spread"] == 0.0
        assert panel.truncation["passed"] is True

    def test_every_archived_panel_counts_zero_truncation(self):
        # The tolerance admits up to 2%, and no archived panel comes anywhere
        # near it, so adopting it cannot have changed a historical verdict.
        archived = sorted((REPO_ROOT / "outputs").rglob("replay_outputs.jsonl"))
        if not archived:
            pytest.skip("no archived replay panels in this checkout")
        for path in archived:
            records = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            measured = measure_truncation(records)
            assert measured["passed"] is True, path
            if path.parent.parent.parent.name == "scale_c":
                assert measured["n_truncated"] == 0, path
