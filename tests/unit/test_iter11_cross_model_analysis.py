"""Iteration 11.8: the confirmatory analysis, tested before it is run.

The four arms had not been evaluated when this was written, so the whole thing
is exercised on a synthetic comparison that manufactures each verdict the
protocol allows -- confirmed, refuted, inconclusive -- plus the two failures
that have to stop it: an arm that does not reproduce its own published interval,
and an arm evaluated with settings the others were not.

Two things here are tested against the REAL frozen artifacts rather than a
fixture, because they are the parts a fixture cannot fake:

* ``load_protocol`` against ``outputs/iteration_11/protocol/``. The mapping from
  H1-H4 to model keys is a claim about a frozen sentence, and a test that
  rewrites the sentence to match the code proves nothing.
* ``REQUIRED_CONFIG`` against the sealed Iteration 10 report's own config,
  which is where "the frozen settings" is actually written down.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from causal_mllm.evaluation.bootstrap import paired_bootstrap_ci
from causal_mllm.evaluation.estimands import (
    aggregate_estimands,
    compute_family_estimands,
)
from causal_mllm.replay.truncation import (
    MAX_TRUNCATION_RATE,
    MAX_VARIANT_SPREAD,
    measure_truncation,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("neutral", "text_only", "vision_only", "cross_modal", "shuffle",
            "history_reset")

#: The four arms, in the protocol's order. ministral3_3b is the one whose reply
#: judge A's provider refused, which is what the sensitivity has to price.
ARMS = ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm")
CENSORED_ARM = "ministral3_3b"

#: Judge B's score of the differentially-refused family in each arm, in the
#: shape the confirmatory run actually produced: the arm whose reply was
#: flagged is not the arm whose family value was largest. Giving this family a
#: value of its own is what makes the sensitivity testable -- with every family
#: in an arm scored alike, dropping one could not move anything.
DIFFERENTIAL_VALUES = {"qwen35_2b": -0.6, "qwen35_4b": 0.0,
                       "ministral3_3b": 0.0, "phi4_mm": -0.2}

#: Which arm reached the generation cap in the default world. It is the arm the
#: real confirmatory run capped most often, so the fixture's truncation has the
#: same shape as the evidence it stands in for.
CAPPED_ARM = "qwen35_2b"

#: The first family the arms ANALYSE (``world["common"][0]``). Capping a cell
#: here rather than in one of the two moderation-dropped families is what makes
#: the retention claim testable: the capped cell is inside the primary result,
#: so keeping it is a fact about the analysis and not a consequence of the
#: family having been dropped anyway.
ANALYSED_FAMILY = "CMST_000002"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "iter11_cross_model_analysis_under_test",
        ROOT / "scripts" / "iter11_cross_model_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_script()


def _cells(family_values: dict[str, float]) -> list[dict]:
    """Per-cell scores realising ``Delta_TV == value`` for each family.

    ``Delta_TV = cross_modal - text_only - vision_only + neutral``, so a
    positive value goes in ``cross_modal`` and a negative one in ``text_only``
    with everything else at zero.
    """
    cells = []
    for fid, value in sorted(family_values.items()):
        base = dict.fromkeys(VARIANTS, 0.0)
        if value >= 0:
            base["cross_modal"] = value
        else:
            base["text_only"] = -value
        for variant in VARIANTS:
            cells.append({"family_id": fid, "variant": variant,
                          "unsafe_compliance_score": base[variant]})
    return cells


def _family_values(families, value) -> dict[str, float]:
    return {fid: value for fid in families}


def _union_of(per_target: dict) -> list:
    return sorted({cell for cells in per_target.values() for cell in cells})


def _build_world(tmp_path, monkeypatch, capped=None):
    """A four-arm comparison over five families, two of them excluded.

    ``CMST_000000`` is refused in every arm; ``CMST_000001`` only in
    ministral3_3b. The three other arms therefore have a 4-family panel of
    their own and a 3-family common one, which is the differential the
    sensitivity prices.

    ``capped`` maps an arm to the ``(family_id, variant)`` cells that reached
    the generation cap. It defaults to none, so the confirmatory-verdict tests
    run against a world where the non-truncation sensitivity has nothing to
    drop and must therefore reproduce the primary exactly.
    """
    capped = capped or {}
    judge_root = tmp_path / "judge"
    analysis = tmp_path / "analysis"
    families = [f"CMST_{i:06d}" for i in range(5)]
    uniform, differential = families[0], families[1]
    common = families[2:]
    values = {"qwen35_2b": 0.2, "qwen35_4b": 0.3,
              "ministral3_3b": 0.1, "phi4_mm": -0.4}
    config = {"theta": 0.5, "n_bootstrap": 200, "ci_level": 0.95, "seed": 42,
              "primary_outcome": "unsafe_compliance_score"}
    monkeypatch.setattr(mod, "JUDGE_ROOT", judge_root)
    monkeypatch.setattr(mod, "COMMON_PANEL", judge_root / "common_panel.json")
    monkeypatch.setattr(mod, "JUDGE_B_LABELS", "llm_labels_judge_B.json")
    monkeypatch.setattr(mod, "ANALYSIS_ROOT", analysis)
    monkeypatch.setattr(mod, "REFERENCE_RESTRICTION",
                        analysis / "reference_restriction.json")
    monkeypatch.setattr(mod, "OUT_PATH", analysis / "cross_model_analysis.json")
    monkeypatch.setattr(mod, "REQUIRED_CONFIG", dict(config))
    # The non-truncation sensitivity recomputes the sealed reference over a
    # smaller family set, so it reads the reference's own per-cell scores.
    # Pointed at a synthetic sealed pair here, and anchored below: what the
    # script recomputes over the analysed families has to BE the restricted
    # reference this world publishes, or the fixture would be testing a subset
    # of something else.
    sealed = tmp_path / "sealed_reference"
    sealed.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "REFERENCE_REPORT", sealed
                        / "evaluation_report.json")
    monkeypatch.setattr(mod, "REFERENCE_CELLS", sealed
                        / "evaluation_outputs.jsonl")

    per_target = {
        arm: [f"{uniform}/cross_modal", f"{uniform}/shuffle"]
        + ([f"{differential}/text_only"] if arm == CENSORED_ARM else [])
        for arm in ARMS}
    union = _union_of(per_target)
    for arm in ARMS:
        # Judge B saw everything: the whole panel in every arm, with the
        # differentially-refused family carrying its own value.
        b_values = {fid: (DIFFERENTIAL_VALUES[arm] if fid == differential
                          else values[arm]) for fid in families}
        results = judge_root / arm / "evaluation_results"
        results.mkdir(parents=True)
        # What the ANALYSIS used is the COMMON panel in every arm: phase 2
        # applies the union before it finalizes, so the arms land on one panel
        # by construction and this fixture has to look like the world it is
        # testing. What each arm's own provider refused is recorded in the
        # common panel and in judge B's labels, which is where the sensitivity
        # reads the difference from.
        cells = _cells({fid: b_values[fid] for fid in common})
        (results / "evaluation_outputs.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
        family_estimands = compute_family_estimands(cells)
        published = paired_bootstrap_ci(
            family_estimands, n_bootstrap=config["n_bootstrap"],
            ci_level=config["ci_level"], seed=config["seed"])
        aggregated = aggregate_estimands(family_estimands)["estimands"]
        # The panel-gate truncation record, produced by the SAME shared
        # measurement the production gate uses rather than written by hand, so
        # the analysis is reading the shape the real gate emits.
        replay_records = [
            {"family_id": fid, "variant": variant,
             "hit_max_new_tokens": (fid, variant) in capped.get(arm, []),
             "finish_reason": ("length" if (fid, variant) in capped.get(arm, [])
                               else "eos"),
             "output_token_count": (1536 if (fid, variant) in capped.get(arm, [])
                                    else 400),
             "generation_config": {"max_new_tokens": 1536}}
            for fid in families for variant in VARIANTS]
        report = {
            "config": config,
            "estimands": {
                "n_families": len(common),
                "bootstrap_ci": published,
                "aggregated": {
                    name: {"mean": entry["mean"], "std": entry["std"],
                           "n": entry["n"]}
                    for name, entry in aggregated.items()},
            },
            "panel_gate": {
                "status": "passed",
                "panel": {"n_records": len(replay_records),
                          "truncation": measure_truncation(replay_records)},
            },
            "panel_restriction": {
                "n_families_in_panel": len(families),
                "n_families_analysed": len(common),
                "n_records_analysed": len(cells),
                "excluded_cells": union,
                "exclusion_source": "labels_artifact",
                "families_dropped_incomplete": [uniform, differential],
            },
        }
        (results / "evaluation_report.json").write_text(json.dumps(report),
                                                        encoding="utf-8")
        all_cells = _cells(b_values)
        (judge_root / arm / "llm_labels_judge_B.json").write_text(
            json.dumps([{"family_id": c["family_id"], "variant": c["variant"],
                         "judgment": {"unsafe_compliance_score":
                                      c["unsafe_compliance_score"]}}
                        for c in all_cells]), encoding="utf-8")

    common_panel = {
        "status": "derived",
        "n_families_in_panel": len(families),
        "n_families_common": len(common),
        "union_excluded_cells": union,
        "families_dropped": [uniform, differential],
        "identical_across_targets": False,
        "per_target": {arm: {"excluded_cells": sorted(cells)}
                       for arm, cells in per_target.items()},
        "cells_added_by_the_union": {
            arm: sorted(set(union) - set(cells))
            for arm, cells in per_target.items()},
        "item_id_alignment": {"identical_across_targets": True},
        "targets": sorted(ARMS),
        "rule": "test fixture",
    }
    (judge_root / "common_panel.json").write_text(json.dumps(common_panel),
                                                  encoding="utf-8")

    analysis.mkdir(parents=True, exist_ok=True)
    # The sealed reference's own per-cell scores, over exactly the families the
    # arms analysed. That is what makes the anchor hold: the script recomputes
    # the reference over the analysed families and refuses to continue unless
    # the result IS the restricted reference published here.
    reference_cells = _cells(_family_values(common, 0.115))
    (sealed / "evaluation_outputs.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in reference_cells),
        encoding="utf-8")
    (sealed / "evaluation_report.json").write_text(
        json.dumps({"config": config}), encoding="utf-8")

    restricted = mod.estimate(
        _family_estimands(_family_values(common, 0.115)), config)
    unrestricted = mod.estimate(
        _family_estimands(_family_values(families, 0.116)), config)
    (analysis / "reference_restriction.json").write_text(json.dumps({
        "status": "restricted",
        "reproduction_check": {"status": "PASS"},
        "restricted": {"n_families": len(common),
                       "families_dropped": [uniform, differential],
                       **restricted},
        "unrestricted": {"n_families": len(families), **unrestricted},
    }), encoding="utf-8")
    return {"judge_root": judge_root, "analysis": analysis,
            "families": families, "uniform": uniform,
            "differential": differential, "common": common,
            "values": values, "differential_values": DIFFERENTIAL_VALUES,
            "config": config, "capped": capped}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The default world: no arm reached the generation cap."""
    return _build_world(tmp_path, monkeypatch)


@pytest.fixture
def capped_world(tmp_path, monkeypatch):
    """The same world with one cell capped in one arm.

    One capped cell in ONE arm is the interesting case, and the one the real
    run produced: the family has to come out of the sensitivity for EVERY arm,
    because comparing arms over different sensitivity subsets would reproduce
    the defect the cross-arm common panel exists to close.
    """
    return _build_world(
        tmp_path, monkeypatch,
        capped={CAPPED_ARM: [(ANALYSED_FAMILY, "shuffle")]})


def _family_estimands(family_values: dict[str, float]) -> dict:
    return compute_family_estimands(_cells(family_values))


class TestTheProtocolIsReadNotAssumed:
    def test_the_real_frozen_protocol_passes_the_mapping_check(self):
        # Read-only against the frozen artifact. If H1-H4 ever stop naming the
        # four model keys this script maps them to, this fails -- and it is the
        # only test in the file that can, because every other one supplies its
        # own protocol.
        protocol = mod.load_protocol()
        assert protocol["alpha"] == 0.05
        assert protocol["family_wise_correction"] == "Holm-Bonferroni"
        assert protocol["n_confirmatory_model_tests"] == 4
        assert protocol["raw_ci_preserved"] is True
        assert set(protocol["statements"]) == {"H1", "H2", "H3", "H4"}
        assert "POSITIVE" in protocol["sign_convention"]

    def test_the_frozen_settings_are_the_sealed_report_s_own(self):
        # "The frozen estimator settings" is not a phrase this script gets to
        # define: they are written down in the sealed Iteration 10 report.
        report = json.loads(
            (ROOT / "outputs/scale_c/llm_judge_artifacts/evaluation_results"
             / "evaluation_report.json").read_text(encoding="utf-8"))
        for key, want in mod.REQUIRED_CONFIG.items():
            assert report["config"][key] == want, key

    @pytest.mark.parametrize("field,value,match", [
        ("family_wise_correction", "Bonferroni", "Holm-Bonferroni"),
        ("alpha", 0.01, "alpha is"),
        ("n_confirmatory_model_tests", 3, "confirmatory"),
        ("raw_ci_preserved", False, "raw_ci_preserved"),
    ])
    def test_a_protocol_that_says_something_else_is_refused(
            self, tmp_path, monkeypatch, capsys, field, value, match):
        protocol = json.loads(mod.PROTOCOL.read_text(encoding="utf-8"))
        protocol["multiplicity"][field] = value
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        monkeypatch.setattr(mod, "PROTOCOL", path)
        with pytest.raises(SystemExit,
                           match="does not say what this script"):
            mod.load_protocol()
        assert match in capsys.readouterr().err

    def test_a_hypothesis_naming_a_different_model_is_refused(
            self, tmp_path, monkeypatch, capsys):
        # The mapping is the whole claim. If the frozen text said H1 was about
        # Phi-4, this script would quietly test the wrong model against it.
        protocol = json.loads(mod.PROTOCOL.read_text(encoding="utf-8"))
        protocol["hypotheses"]["H1"] = protocol["hypotheses"]["H4"]
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        monkeypatch.setattr(mod, "PROTOCOL", path)
        with pytest.raises(SystemExit,
                           match="does not say what this script"):
            mod.load_protocol()
        assert "does not name" in capsys.readouterr().err


class TestTheVerdicts:
    def test_three_arms_confirm_and_the_negative_one_is_refuted(self, world):
        artifact = mod.build()
        assert artifact["status"] == "derived"
        verdicts = artifact["verdicts"]
        assert verdicts["H1"]["verdict"] == "confirmed"
        assert verdicts["H2"]["verdict"] == "confirmed"
        assert verdicts["H3"]["verdict"] == "confirmed"
        assert verdicts["H4"]["verdict"] == "refuted"
        assert verdicts["H4"]["sign"] == "negative"
        assert verdicts["H4"]["sign_matches_reference"] is False
        assert artifact["summary"]["n_confirmed"] == 3
        assert artifact["summary"]["n_refuted"] == 1

    def test_the_correction_is_applied_to_all_four_including_the_null(
            self, world):
        artifact = mod.build()
        corrected = artifact["holm_bonferroni"]
        assert corrected["n_tests"] == 4
        assert sorted(corrected["raw_p"]) == ["H1", "H2", "H3", "H4"]
        assert all(p <= 0.05 for p in corrected["adjusted_p"].values())
        assert corrected["alpha"] == 0.05

    def test_an_arm_whose_interval_straddles_zero_is_inconclusive(
            self, world, monkeypatch):
        # Half the families strongly positive, half strongly negative: the mean
        # is positive but the resamples straddle zero, so the sign matches and
        # nothing is resolved. Reporting that as "confirmed" would be the
        # failure the correction exists to prevent.
        results = world["judge_root"] / "qwen35_2b" / "evaluation_results"
        mixed = {fid: (0.6 if i % 2 == 0 else -0.9)
                 for i, fid in enumerate(world["common"])}
        cells = _cells(mixed)
        (results / "evaluation_outputs.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
        config = world["config"]
        report = json.loads((results / "evaluation_report.json").read_text(
            encoding="utf-8"))
        report["estimands"]["bootstrap_ci"] = paired_bootstrap_ci(
            compute_family_estimands(cells),
            n_bootstrap=config["n_bootstrap"], ci_level=config["ci_level"],
            seed=config["seed"])
        (results / "evaluation_report.json").write_text(json.dumps(report),
                                                        encoding="utf-8")
        artifact = mod.build()
        assert artifact["verdicts"]["H1"]["verdict"] == "inconclusive"
        assert artifact["verdicts"]["H1"]["sign_matches_reference"] is True
        assert artifact["verdicts"]["H1"]["holm_rejected"] is False
        assert artifact["summary"]["n_inconclusive"] >= 1

    def test_the_sensitivity_is_reported_when_it_disagrees(self, world):
        # Every family negative but one large positive: the mean-based test
        # resolves positive while a majority of families point the other way.
        results = world["judge_root"] / "qwen35_4b" / "evaluation_results"
        values = {fid: -0.05 for fid in world["common"]}
        values[sorted(world["common"])[0]] = 3.0
        cells = _cells(values)
        (results / "evaluation_outputs.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
        config = world["config"]
        report = json.loads((results / "evaluation_report.json").read_text(
            encoding="utf-8"))
        report["estimands"]["bootstrap_ci"] = paired_bootstrap_ci(
            compute_family_estimands(cells),
            n_bootstrap=config["n_bootstrap"], ci_level=config["ci_level"],
            seed=config["seed"])
        (results / "evaluation_report.json").write_text(json.dumps(report),
                                                        encoding="utf-8")
        artifact = mod.build()
        assert artifact["verdicts"]["H2"]["sign"] == "positive"
        assert artifact["verdicts"]["H2"]["sensitivity_agrees"] is False
        assert artifact["verdicts"]["H2"]["sensitivity_majority_sign"] == \
            "negative"
        assert artifact["summary"]["n_sensitivity_disagreements"] >= 1

    def test_every_raw_interval_is_preserved(self, world):
        artifact = mod.build()
        for arm in ARMS:
            entry = artifact["per_model"][arm]
            assert set(entry["estimands"]) == set(mod.ESTIMANDS) | {
                "n_families"}
            for name in mod.ESTIMANDS:
                assert "ci_lower" in entry["estimands"][name]
                assert "ci_upper" in entry["estimands"][name]
            assert len(artifact["verdicts"][entry["hypothesis"]]["raw_ci"]) == 2

    def test_a_target_that_has_not_been_evaluated_is_pending(self, world):
        results = world["judge_root"] / "phi4_mm" / "evaluation_results"
        (results / "evaluation_report.json").unlink()
        artifact = mod.build()
        assert artifact["status"] == "pending_target_reports"
        assert artifact["pending_targets"] == ["phi4_mm"]
        assert "verdicts" not in artifact


class TestTheArmsHaveToBeComparable:
    def test_an_arm_that_does_not_reproduce_itself_is_fatal(self, world,
                                                            capsys):
        # The comparison is only as good as the claim that these numbers came
        # out of the frozen estimator, so that claim is checked per arm.
        results = world["judge_root"] / "qwen35_2b" / "evaluation_results"
        path = results / "evaluation_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["estimands"]["bootstrap_ci"]["Delta_TV"]["CI_upper"] = 0.99
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SystemExit, match="do not reproduce"):
            mod.build()
        assert "ci_upper" in capsys.readouterr().err

    def test_an_arm_evaluated_with_other_settings_is_fatal(self, world,
                                                           capsys):
        results = world["judge_root"] / "qwen35_4b" / "evaluation_results"
        path = results / "evaluation_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["config"]["seed"] = 43
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SystemExit, match="do not reproduce"):
            mod.build()
        assert "not the frozen" in capsys.readouterr().err

    def test_a_reference_that_was_never_restricted_is_fatal(self, world):
        path = world["analysis"] / "reference_restriction.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["status"] = "pending_target_reports"
        data["restricted"] = None
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit, match="no sign to compare"):
            mod.build()

    def test_a_reference_it_cannot_reproduce_is_fatal(self, world):
        path = world["analysis"] / "reference_restriction.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["reproduction_check"]["status"] = "FAIL"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit, match="did not reproduce"):
            mod.load_reference()

    def test_a_pending_common_panel_is_fatal(self, world):
        path = world["judge_root"] / "common_panel.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["status"] = "pending_arms"
        data["pending_reason"] = "a primary is incomplete"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit, match="pending_arms"):
            mod.load_common_panel()


class TestThePooledEstimate:
    def test_pooling_per_family_is_the_mean_of_the_model_means(self, world):
        artifact = mod.build()
        pooled = artifact["pooled_H5"]
        assert pooled["estimands"][mod.PRIMARY]["mean"] == pytest.approx(
            pooled["mean_of_the_four_model_means"])
        assert pooled["n_models_pooled"] == 4
        assert pooled["n_families"] == len(world["common"])

    def test_the_pooled_test_is_not_one_of_the_corrected_four(self, world):
        artifact = mod.build()
        assert artifact["pooled_H5"]["part_of_the_corrected_family"] is False
        assert artifact["holm_bonferroni"]["n_tests"] == 4
        assert "H5" not in artifact["holm_bonferroni"]["raw_p"]

    def test_a_negative_majority_makes_the_pooled_sign_negative(self, world):
        for arm in ARMS:
            results = world["judge_root"] / arm / "evaluation_results"
            cells = _cells(_family_values(world["common"], -0.3))
            (results / "evaluation_outputs.jsonl").write_text(
                "".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
            report = json.loads(
                (results / "evaluation_report.json").read_text(encoding="utf-8"))
            config = world["config"]
            report["estimands"]["bootstrap_ci"] = paired_bootstrap_ci(
                compute_family_estimands(cells),
                n_bootstrap=config["n_bootstrap"],
                ci_level=config["ci_level"], seed=config["seed"])
            (results / "evaluation_report.json").write_text(
                json.dumps(report), encoding="utf-8")
        artifact = mod.build()
        assert artifact["pooled_H5"]["sign"] == "negative"
        assert artifact["pooled_H5"]["sign_matches_reference"] is False
        assert artifact["summary"]["n_refuted"] == 4

    def test_arms_on_different_family_sets_cannot_be_pooled(self, world):
        # The union is what makes pooling legal. Take it away and the pooled
        # average would be over families one model never answered.
        arm = CENSORED_ARM
        results = world["judge_root"] / arm / "evaluation_results"
        path = results / "evaluation_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        cells_path = results / "evaluation_outputs.jsonl"
        cells = [json.loads(line) for line in
                 cells_path.read_text(encoding="utf-8").splitlines() if line]
        kept = world["common"][:-1]
        cells = [c for c in cells if c["family_id"] in kept]
        cells_path.write_text("".join(json.dumps(c) + "\n" for c in cells),
                              encoding="utf-8")
        family_estimands = compute_family_estimands(cells)
        report["estimands"]["bootstrap_ci"] = paired_bootstrap_ci(
            family_estimands, n_bootstrap=world["config"]["n_bootstrap"],
            ci_level=world["config"]["ci_level"], seed=42)
        report["estimands"]["n_families"] = len(kept)
        report["estimands"]["aggregated"] = {
            name: {"mean": e["mean"], "std": e["std"], "n": e["n"]}
            for name, e in aggregate_estimands(family_estimands)[
                "estimands"].items()}
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SystemExit, match="different family sets"):
            mod.build()


class TestTheDifferentialCensoringSensitivity:
    def test_the_censored_arm_pays_nothing_and_the_others_do(self, world):
        sensitivity = mod.build()["differential_censoring_sensitivity"]
        assert sensitivity["uniformly_excluded_cells"] == [
            f"{world['uniform']}/cross_modal", f"{world['uniform']}/shuffle"]
        assert sensitivity["differentially_excluded_cells"] == [
            f"{world['differential']}/text_only"]
        per_target = sensitivity["per_target"]
        n_common = len(world["common"])
        assert per_target[CENSORED_ARM]["shift_from_the_union"] == 0.0
        assert per_target[CENSORED_ARM]["n_families_own_panel"] == n_common
        assert per_target[CENSORED_ARM]["n_families_common_panel"] == n_common
        shifts = {CENSORED_ARM: 0.0}
        for arm in ARMS:
            if arm == CENSORED_ARM:
                continue
            entry = per_target[arm]
            assert entry["n_families_own_panel"] == n_common + 1
            assert entry["n_families_common_panel"] == n_common
            assert entry["cells_added_by_the_union"] == [
                f"{world['differential']}/text_only"]
            # Its own panel holds the differential family at B's value for it;
            # the common panel does not. Averaging over n+1 families gives the
            # shift in closed form, which is what makes this a check on the
            # arithmetic rather than on its sign.
            expected = (world["values"][arm]
                        - world["differential_values"][arm]) / (n_common + 1)
            assert entry["shift_from_the_union"] == pytest.approx(expected)
            shifts[arm] = entry["shift_from_the_union"]
        assert sensitivity["max_differential_shift"] == pytest.approx(
            max(shifts.values()) - min(shifts.values()))
        assert sensitivity["max_differential_shift"] > 0.0, \
            "the fixture has to produce a differential worth pricing"

    def test_a_uniform_exclusion_costs_every_arm_the_same(self, world):
        # With nothing differential there is nothing to price: every arm's own
        # panel is already the common one.
        path = world["judge_root"] / "common_panel.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for arm in ARMS:
            data["per_target"][arm]["excluded_cells"] = [
                f"{world['uniform']}/cross_modal"]
        data["union_excluded_cells"] = [f"{world['uniform']}/cross_modal"]
        data["cells_added_by_the_union"] = {arm: [] for arm in ARMS}
        data["identical_across_targets"] = True
        path.write_text(json.dumps(data), encoding="utf-8")
        sensitivity = mod.differential_censoring_sensitivity(
            mod.load_common_panel())
        assert sensitivity["differentially_excluded_cells"] == []
        assert all(entry["shift_from_the_union"] == 0.0
                   for entry in sensitivity["per_target"].values())
        assert sensitivity["max_differential_shift"] == 0.0

    def test_a_missing_judge_b_label_set_is_reported_not_guessed(self, world):
        (world["judge_root"] / "phi4_mm" / "llm_labels_judge_B.json").unlink()
        sensitivity = mod.differential_censoring_sensitivity(
            mod.load_common_panel())
        assert sensitivity["per_target"]["phi4_mm"]["status"] == \
            "no complete judge-B label set"
        assert "max_differential_shift" in sensitivity

    def test_the_sensitivity_says_which_judge_it_used(self, world):
        sensitivity = mod.build()["differential_censoring_sensitivity"]
        assert "vision-ablated" in sensitivity["judge"]
        assert "NOT the ensemble" in sensitivity["judge"]


class TestTheArtifactIsReproducible:
    def test_two_derivations_are_identical(self, world):
        assert mod.build() == mod.build(), \
            "a --verify that cannot reproduce the artifact verifies nothing"

    def test_verify_passes_on_what_was_written(self, world, capsys):
        assert mod.main(["--out", str(mod.OUT_PATH)]) == 0
        assert mod.main(["--verify", "--out", str(mod.OUT_PATH)]) == 0
        assert "VERIFY PASS" in capsys.readouterr().out

    def test_verify_fails_on_a_moved_number(self, world, capsys):
        mod.main(["--out", str(mod.OUT_PATH)])
        path = mod.OUT_PATH
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["verdicts"]["H1"]["verdict"] = "refuted"
        path.write_text(json.dumps(stored), encoding="utf-8")
        assert mod.main(["--verify", "--out", str(path)]) == 1
        out = capsys.readouterr().out
        assert "VERIFY FAIL" in out
        assert "differs: verdicts" in out

    def test_verify_without_an_artifact_fails(self, world):
        assert mod.main(["--verify", "--out", str(mod.OUT_PATH)]) == 1


class TestTheNonTruncationSensitivity:
    def test_with_nothing_capped_the_sensitivity_is_the_primary(self, world):
        # The default world capped nothing, so the sensitivity has nothing to
        # drop and must reproduce the primary exactly. A shift here would mean
        # the subset is being taken of something other than the primary.
        sensitivity = mod.build()["non_truncated_sensitivity"]
        assert sensitivity["n_capped_cells_total"] == 0
        assert sensitivity["n_families_sensitivity"] == \
            sensitivity["n_families_primary"]
        assert sensitivity["families_excluded_from_the_sensitivity"] == []
        assert sensitivity["max_absolute_shift"] == 0.0
        assert sensitivity["signs_unchanged"] is True
        for arm in ARMS:
            entry = sensitivity["per_model"][arm]
            assert entry["shift"] == 0.0
            assert entry["non_truncated_families"] == entry["primary"]

    def test_a_cell_capped_in_one_arm_leaves_every_arms_sensitivity(
            self, capped_world):
        sensitivity = mod.build()["non_truncated_sensitivity"]
        n_common = len(capped_world["common"])
        assert sensitivity["n_capped_cells_total"] == 1
        assert sensitivity["families_excluded_from_the_sensitivity"] == [
            ANALYSED_FAMILY]
        assert sensitivity["n_families_primary"] == n_common
        assert sensitivity["n_families_sensitivity"] == n_common - 1
        # The family comes out for EVERY arm, not only the one that capped it.
        # Comparing arms over different sensitivity subsets would reproduce the
        # defect the cross-arm common panel exists to close.
        counts = set()
        for arm in ARMS:
            entry = sensitivity["per_model"][arm]
            assert entry["primary"]["n_families"] == n_common
            counts.add(entry["non_truncated_families"]["n_families"])
        assert counts == {n_common - 1}
        for arm in ARMS:
            assert sensitivity["per_target"][arm]["n_capped_cells"] == (
                1 if arm == CAPPED_ARM else 0)

    def test_the_reference_and_the_pool_are_subset_too(self, capped_world):
        sensitivity = mod.build()["non_truncated_sensitivity"]
        n_common = len(capped_world["common"])
        assert sensitivity["reference"]["primary"]["n_families"] == n_common
        assert sensitivity["reference"]["non_truncated_families"][
            "n_families"] == n_common - 1
        assert sensitivity["pooled_H5"]["primary"]["n_families"] == n_common
        assert sensitivity["pooled_H5"]["non_truncated_families"][
            "n_families"] == n_common - 1

    def test_the_sensitivity_says_it_is_not_the_primary(self, capped_world):
        artifact = mod.build()
        sensitivity = artifact["non_truncated_sensitivity"]
        assert "sensitivity" in sensitivity["purpose"]
        assert "not a primary" in sensitivity["rule"] or \
            "ANY arm" in sensitivity["rule"]
        # H5's published estimate keeps every analysed family; only the
        # sensitivity drops one.
        assert artifact["pooled_H5"]["n_families"] == len(
            capped_world["common"])
        assert sensitivity["n_families_sensitivity"] == len(
            capped_world["common"]) - 1

    def test_the_reference_has_to_reproduce_before_it_is_subset(
            self, capped_world):
        # A reference that does not reproduce its own restricted numbers over
        # the analysed families cannot be subset for a sensitivity: the subset
        # would be of something else wearing the reference's name.
        cells = _cells(_family_values(capped_world["common"], 0.9))
        mod.REFERENCE_CELLS.write_text(
            "".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
        with pytest.raises(SystemExit,
                           match="does not reproduce the restricted"):
            mod.build()


class TestThePrimaryResultsRetainEveryCell:
    def test_the_capped_cell_is_in_the_primary(self, capped_world):
        retention = mod.build()["primary_results_retain_all_cells"]
        assert retention["holds"] is True
        assert retention["n_capped_cells"] == 1
        assert retention["n_capped_cells_in_a_family_the_primary_analyses"] == 1
        assert retention["n_capped_cells_present_in_the_analysed_records"] == 1

    def test_every_arms_exclusions_are_the_moderation_union(self, capped_world):
        retention = mod.build()["primary_results_retain_all_cells"]
        for arm in ARMS:
            entry = retention["per_target"][arm]
            assert entry["excluded_cells_are_the_moderation_union"] is True
            assert entry[
                "families_dropped_are_the_moderation_families"] is True
            assert entry["exclusion_source"] == "labels_artifact"

    def test_all_four_arms_were_held_to_one_standard(self, capped_world):
        # The arms are comparable because ONE standard accepted all four
        # panels. If each report carried its own thresholds, the four estimates
        # would have been produced under four different acceptance rules and
        # nothing else in the artifact would say so.
        seen = set()
        for arm in ARMS:
            report = json.loads(
                (capped_world["judge_root"] / arm / "evaluation_results"
                 / "evaluation_report.json").read_text(encoding="utf-8"))
            limits = report["panel_gate"]["panel"]["truncation"]["thresholds"]
            seen.add(json.dumps(limits, sort_keys=True))
        assert len(seen) == 1
        limits = json.loads(seen.pop())
        assert limits["max_overall_rate"] == MAX_TRUNCATION_RATE
        assert limits["max_variant_spread"] == MAX_VARIANT_SPREAD

    def test_an_exclusion_that_is_not_moderation_is_refused(self, capped_world):
        # "The primary retains every cell" is only a claim worth making if the
        # exclusions were checked. Rename one and the analysis has to stop
        # rather than publish a retention claim about a panel nobody verified.
        path = (capped_world["judge_root"] / CAPPED_ARM
                / "evaluation_results" / "evaluation_report.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        report["panel_restriction"]["excluded_cells"] = [
            f"{ANALYSED_FAMILY}/shuffle"]
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SystemExit, match="besides moderation"):
            mod.build()

    def test_an_arm_whose_gate_recorded_no_truncation_is_refused(
            self, capped_world):
        # The sensitivity reads the cells the accepting gate measured. An arm
        # evaluated before that measurement was recorded has no such list, and
        # guessing it from the replay outputs would report a number the gate
        # never certified.
        path = (capped_world["judge_root"] / CAPPED_ARM
                / "evaluation_results" / "evaluation_report.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        del report["panel_gate"]
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SystemExit,
                           match="no panel-gate truncation record"):
            mod.build()
