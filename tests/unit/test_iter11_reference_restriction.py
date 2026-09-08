"""Iteration 11.8: restricting the frozen reference to the families in play.

The comparison is between four arms that analysed 98 families and a reference
published over 100, so the reference has to be recomputed over the same 98 --
and before that means anything, the recomputation has to prove it reproduces
the published 100. Both halves are tested here: the reproduction check against
the real sealed reference, and the restriction logic against synthetic target
reports, because the real ones do not exist until phase 2 finalizes.

The restriction is read from each target's own ``panel_restriction`` and
INTERSECTED, which is the part that is easy to get wrong: the provider's
moderation verdict is not stable over time and the four arms reach a given cell
minutes apart, so one target could lose a family another kept. That is not
hypothetical -- the confirmatory run refused ``CMST_456921/text_only`` in one
arm and served it in three -- which is why phase 2 takes the union up front
(``scripts/iter11_common_panel.py``) and why this script still refuses to
assume the arms agree. The union of what the arms lost is the complement of the
intersection of what they analysed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from causal_mllm.evaluation.bootstrap import (
    bootstrap_two_sided_p,
    paired_bootstrap_ci,
    paired_bootstrap_samples,
)
from causal_mllm.replay import reproduction

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("neutral", "text_only", "vision_only", "cross_modal", "shuffle",
            "history_reset")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "iter11_reference_restriction",
        ROOT / "scripts" / "iter11_reference_restriction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_script()

DROPPED_FAMILY = "CMST_795308"


@pytest.fixture(scope="module")
def sealed_reference():
    """The real sealed reference, loaded once. Read-only."""
    report, cells, config = mod.load_reference()
    family_estimands = mod.compute_family_estimands(cells)
    return report, cells, config, family_estimands


class TestTheReproductionCheckAgainstTheSealedReference:
    """Read-only over the real sealed artifacts. The slow part of this file,
    and the part that cannot be faked: it is the evidence that the estimator
    being restricted is the frozen one."""

    def test_the_sealed_reference_carries_every_family_and_cell(
            self, sealed_reference):
        _, cells, _, family_estimands = sealed_reference
        assert len(cells) == 600
        assert len(family_estimands) == 100
        assert DROPPED_FAMILY in family_estimands, \
            "the reference was judged under the older moderation policy, so " \
            "it DOES hold the two cells judge A now refuses"

    def test_the_unrestricted_recomputation_reproduces_the_published_ci(
            self, sealed_reference):
        report, _, config, family_estimands = sealed_reference
        recomputed = mod.estimate(family_estimands, config)
        assert mod.check_reproduction(recomputed, report) == []

    def test_the_published_reference_interval_is_the_one_the_protocol_freezes(
            self, sealed_reference):
        # The protocol's sign_convention quotes this exact interval, so a
        # silent change to it anywhere would invalidate H1-H5.
        report, _, config, family_estimands = sealed_reference
        recomputed = mod.estimate(family_estimands, config)
        assert round(recomputed["Delta_TV"]["ci_lower"], 4) == 0.0495
        assert round(recomputed["Delta_TV"]["ci_upper"], 4) == 0.18
        assert recomputed["Delta_TV"]["mean"] > 0, \
            "the reference Delta_TV is POSITIVE; that is the sign H1-H4 test"

    def test_a_tampered_published_interval_fails_closed(self, sealed_reference):
        report, _, config, family_estimands = sealed_reference
        recomputed = mod.estimate(family_estimands, config)
        tampered = json.loads(json.dumps(report))
        tampered["estimands"]["bootstrap_ci"]["Delta_TV"]["CI_upper"] = 0.30
        issues = mod.check_reproduction(recomputed, tampered)
        assert any("Delta_TV.ci_upper" in i for i in issues)

    def test_a_missing_published_estimand_is_an_issue_not_a_pass(
            self, sealed_reference):
        report, _, config, family_estimands = sealed_reference
        recomputed = mod.estimate(family_estimands, config)
        partial = json.loads(json.dumps(report))
        del partial["estimands"]["bootstrap_ci"]["order_effect"]
        assert any("publishes no CI" in i
                   for i in mod.check_reproduction(recomputed, partial))

    def test_the_resample_refactor_left_the_rng_consumption_order_untouched(
            self, sealed_reference):
        # ``paired_bootstrap_ci`` was split so 11.8 can put a p-value beside
        # the frozen interval, and the split had to leave the rng consumption
        # order untouched. These invariants hold in ANY interpreter, which is
        # why they are equalities: the interval is a summary of exactly the
        # resamples the p-value counts, so the two cannot describe different
        # distributions. A refactor that moved the loop breaks all three.
        _, _, config, family_estimands = sealed_reference
        ci = paired_bootstrap_ci(
            family_estimands, n_bootstrap=config["n_bootstrap"],
            ci_level=config["ci_level"], seed=config["seed"])
        samples = paired_bootstrap_samples(
            family_estimands, n_bootstrap=config["n_bootstrap"],
            seed=config["seed"])
        ordered = sorted(samples["Delta_TV"])
        assert sum(ordered) / len(ordered) == ci["Delta_TV"]["mean"]
        assert ordered[0] <= ci["Delta_TV"]["CI_lower"] \
            <= ci["Delta_TV"]["CI_upper"] <= ordered[-1]
        assert len(ordered) == config["n_bootstrap"]

    def test_the_published_reference_floats_are_pinned_to_the_tolerance(
            self, sealed_reference):
        # The sealed Iteration 10 report publishes these, the protocol's
        # sign_convention quotes the interval, and H1-H5 test its sign, so
        # they are pinned. Pinned to ``reproduction.FLOAT_TOLERANCE`` rather
        # than with ``==``, because the last bit of a floating-point sum is a
        # property of the interpreter that summed it and not of the evidence:
        # CPython 3.12 changed the builtin ``sum`` to Neumaier summation, and
        # under it these come out as 0.11506759999999999 / 0.0495 / 0.18 --
        # a difference of -2.78e-16 in the mean, measured, not estimated.
        #
        # This used to assert exact equality, which made it the one test that
        # failed out of 1,506 under a different interpreter while every verdict
        # in the repository stayed where it was. The tolerance is 1e-12: about
        # 3,600 times the drift that was measured, and still nine orders of
        # magnitude below anything that would move a digit this repository
        # reports. The four-decimal pin on the interval the protocol quotes
        # lives in
        # ``test_the_published_reference_interval_is_the_one_the_protocol_freezes``.
        _, _, config, family_estimands = sealed_reference
        ci = paired_bootstrap_ci(
            family_estimands, n_bootstrap=config["n_bootstrap"],
            ci_level=config["ci_level"], seed=config["seed"])
        published = {"mean": 0.11506760000000027,
                     "CI_lower": 0.0495,
                     "CI_upper": 0.17999999999999997}
        for field, expected in published.items():
            drift = ci["Delta_TV"][field] - expected
            assert abs(drift) <= reproduction.FLOAT_TOLERANCE, (
                f"Delta_TV.{field} moved {drift:+.3g} from the published "
                f"value, which is far more than a summation order can "
                f"account for")

    def test_the_reference_sign_is_resolved_by_its_own_resamples(
            self, sealed_reference):
        # The sign H1-H4 are tested against is not a coin flip: every one of
        # the 5000 resamples of the reference Delta_TV is above zero, so the
        # bootstrap p-value is at its floor.
        _, _, config, family_estimands = sealed_reference
        samples = paired_bootstrap_samples(
            family_estimands, n_bootstrap=config["n_bootstrap"],
            seed=config["seed"])
        ordered = sorted(samples["Delta_TV"])
        assert ordered[0] > 0.0
        assert bootstrap_two_sided_p(ordered) == \
            pytest.approx(1.0 / config["n_bootstrap"])


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A miniature reference plus a judge tree, so the restriction logic can
    be exercised without waiting for phase 2 or paying for a 5000-resample
    bootstrap on every test."""
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    cells = []
    scores = {"neutral": 0.1, "text_only": 0.1, "vision_only": 0.1,
              "cross_modal": 0.8, "shuffle": 0.2, "history_reset": 0.2}
    families = [f"CMST_{i:06d}" for i in range(4)] + [DROPPED_FAMILY]
    for fid in families:
        for variant in VARIANTS:
            cells.append({"family_id": fid, "variant": variant,
                          "unsafe_compliance_score": scores[variant]})
    cells_path = reference_dir / "evaluation_outputs.jsonl"
    cells_path.write_text("".join(json.dumps(c) + "\n" for c in cells),
                          encoding="utf-8")

    config = {"n_bootstrap": 50, "ci_level": 0.95, "seed": 42,
              "primary_outcome": "unsafe_compliance_score", "theta": 0.5}
    family_estimands = mod.compute_family_estimands(cells)
    report = {"estimands": {
        "bootstrap_ci": {
            name: {"mean": 0.0, "CI_lower": 0.0, "CI_upper": 0.0}
            for name in mod.ESTIMANDS},
        "aggregated": {}},
        "panel_gate": {"panel": {"run_id": "synthetic-reference"}},
        "judge_provenance": {"labels_sha256": "ab" * 32},
        "config": config}
    # Publish what the estimator actually produces, so the reproduction check
    # passes on the fixture and the tests exercise the restriction rather than
    # the self-check.
    estimate = mod.estimate(family_estimands, config)
    for name in mod.ESTIMANDS:
        report["estimands"]["bootstrap_ci"][name] = {
            "mean": estimate[name]["bootstrap_mean"],
            "CI_lower": estimate[name]["ci_lower"],
            "CI_upper": estimate[name]["ci_upper"]}
        report["estimands"]["aggregated"][name] = {
            "mean": estimate[name]["mean"], "std": estimate[name]["std"],
            "n": estimate[name]["n"]}
    report_path = reference_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(mod, "REFERENCE_REPORT", report_path)
    monkeypatch.setattr(mod, "REFERENCE_CELLS", cells_path)
    judge_root = tmp_path / "judge"
    monkeypatch.setattr(mod, "JUDGE_ROOT", judge_root)
    monkeypatch.setattr(mod, "TARGETS", ("t1", "t2"))
    return {"judge_root": judge_root, "n_families": len(families),
            "config": config}


def _write_target_report(synthetic, key, dropped, n_families_in_panel=100):
    """A target's evaluation_report.json carrying a panel_restriction."""
    out = synthetic["judge_root"] / key / "evaluation_results"
    out.mkdir(parents=True, exist_ok=True)
    kept = synthetic["n_families"] - (1 if dropped else 0)
    report = {
        "estimands": {"n_families": kept},
        "panel_restriction": {
            "n_records_in_panel": synthetic["n_families"] * 6,
            "n_records_analysed": kept * 6,
            "n_families_in_panel": synthetic["n_families"],
            "n_families_analysed": kept,
            "excluded_cells": [f"{d}/cross_modal" for d in dropped],
            "families_dropped_incomplete": sorted(dropped),
        },
    }
    (out / "evaluation_report.json").write_text(json.dumps(report),
                                                encoding="utf-8")


class TestTheRestrictionIsReadFromTheArms:
    def test_a_target_that_has_not_been_finalized_is_pending(self, synthetic):
        dropped, per_target, issues = mod.analysed_family_set()
        assert dropped is None
        assert per_target["t1"]["status"] == "pending"
        assert any("phase 2 has not finalized" in i for i in issues)

    def test_both_arms_restricted_identically_gives_one_dropped_family(
            self, synthetic):
        for key in ("t1", "t2"):
            _write_target_report(synthetic, key, [DROPPED_FAMILY])
        dropped, per_target, issues = mod.analysed_family_set()
        assert dropped == {DROPPED_FAMILY}
        assert per_target["t1"]["status"] == "restricted"
        assert per_target["t2"]["n_families_analysed"] == 4
        assert issues == []

    def test_arms_that_lost_different_families_are_restricted_to_the_union(
            self, synthetic):
        # The case the confirmatory run actually produced: judge A refused
        # CMST_456921/text_only in one arm and served it in three. Comparing
        # them over different family sets is not a cross-model comparison, so
        # the reference gives up both.
        _write_target_report(synthetic, "t1", [DROPPED_FAMILY])
        _write_target_report(synthetic, "t2", ["CMST_000001"])
        dropped, _, issues = mod.analysed_family_set()
        assert dropped == {DROPPED_FAMILY, "CMST_000001"}
        assert any("did not lose the same families" in i for i in issues)

    def test_an_arm_that_lost_nothing_is_unrestricted_not_pending(
            self, synthetic):
        # Moderation may not refuse the cell on a later run. An arm with no
        # restriction analysed every family, which is a different statement
        # from "has not been evaluated yet" and must not be confused with it.
        out = synthetic["judge_root"] / "t1" / "evaluation_results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation_report.json").write_text(json.dumps(
            {"estimands": {"n_families": synthetic["n_families"]}}),
            encoding="utf-8")
        _write_target_report(synthetic, "t2", [DROPPED_FAMILY])
        dropped, per_target, issues = mod.analysed_family_set()
        assert per_target["t1"]["status"] == "unrestricted"
        assert dropped == {DROPPED_FAMILY}
        assert issues == []


class TestTheArtifact:
    def test_the_restricted_reference_drops_exactly_the_family(self,
                                                              synthetic):
        for key in ("t1", "t2"):
            _write_target_report(synthetic, key, [DROPPED_FAMILY])
        artifact = mod.build()
        assert artifact["status"] == "restricted"
        assert artifact["unrestricted"]["n_families"] == 5
        assert artifact["restricted"]["n_families"] == 4
        assert artifact["restricted"]["families_dropped"] == [DROPPED_FAMILY]
        assert artifact["reproduction_check"]["status"] == "PASS"

    def test_a_uniform_fixture_is_not_moved_by_the_restriction(
            self, synthetic):
        for key in ("t1", "t2"):
            _write_target_report(synthetic, key, [DROPPED_FAMILY])
        artifact = mod.build()
        # Every family in the fixture carries the same scores, so dropping one
        # cannot move the mean. That is the assertion: the machinery reports
        # the number it computed rather than one that looks plausible, and a
        # restriction that silently changed the estimate would show up here as
        # a difference on a fixture where there provably is none.
        assert artifact["restricted"]["Delta_TV"]["mean"] == pytest.approx(
            artifact["unrestricted"]["Delta_TV"]["mean"])
        assert artifact["restricted"]["Delta_TV"]["n"] == 4

    def test_pending_targets_produce_no_restricted_estimate(self, synthetic):
        artifact = mod.build()
        assert artifact["status"] == "pending_target_reports"
        assert artifact["restricted"] is None
        assert artifact["unrestricted"]["n_families"] == 5

    def test_the_artifact_is_reproducible_byte_for_byte(self, synthetic):
        for key in ("t1", "t2"):
            _write_target_report(synthetic, key, [DROPPED_FAMILY])
        assert mod.build() == mod.build(), \
            "a --verify that cannot reproduce the artifact verifies nothing"

    def test_a_reference_it_cannot_reproduce_is_a_fatal_not_a_warning(
            self, synthetic, monkeypatch):
        for key in ("t1", "t2"):
            _write_target_report(synthetic, key, [DROPPED_FAMILY])
        report_path = mod.REFERENCE_REPORT
        tampered = json.loads(report_path.read_text(encoding="utf-8"))
        tampered["estimands"]["bootstrap_ci"]["Delta_TV"]["CI_upper"] = 0.99
        report_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not reproduce"):
            mod.build()
