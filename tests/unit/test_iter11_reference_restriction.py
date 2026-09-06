"""Iteration 11.8: restricting the frozen reference to the families in play.

The comparison is between four arms that analysed 99 families and a reference
published over 100, so the reference has to be recomputed over the same 99 --
and before that means anything, the recomputation has to prove it reproduces
the published 100. Both halves are tested here: the reproduction check against
the real sealed reference, and the restriction logic against synthetic target
reports, because the real ones do not exist until phase 2 finalizes.

The restriction is read from each target's own ``panel_restriction`` and
INTERSECTED, which is the part that is easy to get wrong: the provider's
moderation verdict is not stable over time and the four arms reach a given cell
minutes apart, so one target could lose a family another kept. The union of
what the arms lost is the complement of the intersection of what they analysed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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
        # The case the time-varying moderation verdict makes possible: t1
        # refused a cell t2 was served. Comparing them over different family
        # sets is not a cross-model comparison, so the reference gives up both.
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
