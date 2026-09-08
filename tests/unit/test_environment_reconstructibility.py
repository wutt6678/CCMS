"""The environment is authenticated, and now it is also reconstructible.

``dependency_lock_snapshot()`` recorded ``pip_freeze_sha256`` and
``n_packages``. That is a hash with no committed preimage: another machine could
find out whether it happened to possess the certified environment and could not
create it. So every exact-equality verifier in the repository was usable in
exactly one place, and the reason it mattered is measurable.

Under CPython 3.12.3, against artifacts written by the certified 3.10.20, the
11.8 derivation differs in 166 leaves and in NOTHING else: 155 continuous
quantities with a worst absolute difference of 2.61e-15, and 11 p-values moving
in whole steps of 2/5000 because a bootstrap p-value counts resamples on one
side of zero. H2's raw p reads 0.0248 instead of 0.0256 and H2 is still
CONFIRMED at the same adjusted 0.0272. The cause is not NumPy -- the frozen
bootstrap is pure Python and touches no array library -- it is that CPython 3.12
changed the builtin ``sum`` to Neumaier summation, and the bootstrap averages
every resample with ``sum``.

The two halves of the fix are pinned here. The committed freeze makes the
certified environment rebuildable from a checkout, and
``causal_mllm.replay.reproduction`` decides what a re-derivation has to match:
every non-numeric leaf exactly in every environment, every integer exactly
because there is no summation order in a count, and floats within a documented
tolerance that is licensed ONLY by a demonstrated deviation from the lock.
Inside the certified environment the comparison stays exact, because there the
last bit is reproducible and a difference is a defect.

CI-safe: the comparison rules and the producer are exercised against miniature
artifacts under ``tmp_path``, and the committed-evidence class reads files, so
none of it needs the media, a GPU, or the certified interpreter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from causal_mllm.replay import reproduction
from causal_mllm.replay.registry import freeze_text, verify_committed_freeze

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


producer = _load_script("iter11_write_dependency_lock")

WORLD_LOCK = ROOT / "outputs" / "iteration_11" / "preflight" \
    / "resolved_models.lock.yaml"
WORLD_FREEZE = ROOT / "outputs" / "iteration_11" / "preflight" \
    / "dependency_freeze.lock.txt"
WORLD_REPORT = ROOT / "outputs" / "iteration_11" / "preflight" \
    / "dependency_lock_reconstruction.json"

#: Measured, not estimated: CPython 3.10.20 against 3.12.3 on the two 11.8
#: artifacts and on the sealed Iteration 10 reference.
MEASURED_WORST_CONTINUOUS = 2.60902e-15
MEASURED_WORST_P_VALUE = 0.0044
MEASURED_REFERENCE_MEAN_DRIFT = -2.7755575615628914e-16
MEASURED_N_DIFFERING_LEAVES_11_8 = 166
MEASURED_N_DIFFERING_LEAVES_REFERENCE = 34

#: The distance from each confirmatory p-value to the boundary that decides it,
#: read out of the committed artifact. Holm-Bonferroni compares the k-th
#: smallest raw p to alpha/(m-k+1) with m=4 and alpha=0.05; the adjusted p is
#: compared to alpha.
NEAREST_RAW_P_TO_ITS_HOLM_CRITICAL_VALUE = 0.025 - 0.0136
NEAREST_ADJUSTED_P_TO_ALPHA = 0.05 - 0.0272

FROZEN_LINES = ["numpy==2.2.6", "pandas==2.3.3", "scipy==1.15.3",
                "torch==2.8.0", "yarl==1.24.5"]


def _lock_document(lines=None, python_version="3.10.20") -> dict:
    lines = FROZEN_LINES if lines is None else lines
    return {"dependency_lock": {
        "pip_freeze_sha256": hashlib.sha256(
            freeze_text(lines).encode("utf-8")).hexdigest(),
        "n_packages": len(lines),
        "excluded_self_distributions": ["causal-mllm"],
        "pyproject_sha256": "d" * 64,
        "python_version": python_version,
        "executable": "/somewhere/bin/python",
        "editable_vcs_revisions": {},
        "editable_installs": {},
    }}


@pytest.fixture
def lock_world(tmp_path, monkeypatch):
    """A miniature lock, and the two artifacts this stage would file."""
    lock = tmp_path / "resolved_models.lock.yaml"
    lock.write_text(yaml.safe_dump(_lock_document()), encoding="utf-8")
    monkeypatch.setattr(producer, "FREEZE_PATH", tmp_path / "freeze.lock.txt")
    monkeypatch.setattr(producer, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(producer, "normalized_freeze_lines",
                        lambda: (list(FROZEN_LINES), ["causal-mllm"]))
    monkeypatch.setattr(producer, "dependency_lock_snapshot", lambda: {
        "pip_freeze_sha256": hashlib.sha256(
            freeze_text(FROZEN_LINES).encode("utf-8")).hexdigest(),
        "python_version": "3.10.20",
        "pyproject_sha256": "d" * 64,
    })
    monkeypatch.setattr(producer, "code_tree_status",
                        lambda exclude_prefixes=(): {
                            "dirty": False, "code_dirty_paths": [],
                            "untracked_paths": [],
                            "excluded_own_outputs": list(exclude_prefixes),
                            "excluded_cache_paths": []})
    monkeypatch.setattr(producer, "get_git_commit", lambda: "0" * 40)
    monkeypatch.setattr(producer, "observed_versions", lambda: {})
    return lock


class TestTheCommittedFreezeIsTheHashsPreimage:
    def test_the_producer_files_the_hashed_text_and_nothing_else(
            self, lock_world):
        assert producer.main(["--lock", str(lock_world)]) == 0
        raw = producer.FREEZE_PATH.read_bytes()
        assert raw == freeze_text(FROZEN_LINES).encode("utf-8")
        assert not raw.endswith(b"\n"), \
            "a trailing newline would make the file's own bytes hash to " \
            "something other than what it lists"

    def test_what_it_files_verifies_clean(self, lock_world):
        producer.main(["--lock", str(lock_world)])
        assert producer.verify(lock_world) == 0

    def test_the_report_names_its_producer_and_binds_the_same_hash(
            self, lock_world):
        producer.main(["--lock", str(lock_world)])
        report = json.loads(producer.REPORT_PATH.read_text(encoding="utf-8"))
        assert report["produced_by"] == \
            "scripts/iter11_write_dependency_lock.py"
        assert report["freeze_sha256"] == \
            report["recorded_pip_freeze_sha256"]
        assert report["freeze_is_the_preimage_of_the_recorded_hash"] is True
        assert report["n_packages"] == len(FROZEN_LINES)
        assert report["python_version"] == "3.10.20"
        assert report["recreate_with"].startswith("conda create")
        assert "pip install -r" in report["recreate_with"]
        assert "3.10.20" in report["recreate_with"]

    def test_the_report_names_the_packages_that_move_a_float(self, lock_world):
        producer.main(["--lock", str(lock_world)])
        report = json.loads(producer.REPORT_PATH.read_text(encoding="utf-8"))
        assert report["numeric_packages"] == {
            "numpy": "2.2.6", "scipy": "1.15.3", "torch": "2.8.0",
            "pandas": "2.3.3"}

    def test_a_tampered_freeze_no_longer_hashes_to_the_lock(self, lock_world):
        producer.main(["--lock", str(lock_world)])
        text = producer.FREEZE_PATH.read_text(encoding="utf-8")
        producer.FREEZE_PATH.write_text(
            text.replace("numpy==2.2.6", "numpy==2.3.0"), encoding="utf-8")
        result = verify_committed_freeze(producer.FREEZE_PATH, lock_world)
        assert result["matches_recorded_hash"] is False
        assert result["reconstructs_the_certified_environment"] is False
        assert any("is not the environment the artifacts were certified "
                   "against" in i for i in result["issues"])
        assert producer.verify(lock_world) == 1

    def test_a_freeze_missing_a_package_is_caught_by_its_count(self,
                                                              lock_world):
        producer.main(["--lock", str(lock_world)])
        lines = producer.FREEZE_PATH.read_text(encoding="utf-8").split("\n")
        producer.FREEZE_PATH.write_text("\n".join(lines[:-1]),
                                        encoding="utf-8")
        result = verify_committed_freeze(producer.FREEZE_PATH, lock_world)
        assert result["matches_recorded_count"] is False
        assert any("lists 4 packages" in i for i in result["issues"])

    def test_a_freeze_that_omits_numpy_says_so(self, lock_world):
        producer.main(["--lock", str(lock_world)])
        lines = [line for line in FROZEN_LINES
                 if not line.startswith("numpy")]
        producer.FREEZE_PATH.write_text(freeze_text(lines), encoding="utf-8")
        result = verify_committed_freeze(producer.FREEZE_PATH, lock_world)
        assert result["numeric_packages"]["numpy"] is None
        assert any("pins no numpy" in i for i in result["issues"])

    def test_nothing_committed_yet_is_its_own_answer(self, lock_world):
        assert producer.verify(lock_world) == 2
        result = verify_committed_freeze(producer.FREEZE_PATH, lock_world)
        assert result["exists"] is False
        assert any("cannot be recreated from a checkout" in i
                   for i in result["issues"])

    def test_a_freeze_without_its_report_is_incomplete(self, lock_world):
        producer.main(["--lock", str(lock_world)])
        producer.REPORT_PATH.unlink()
        assert producer.verify(lock_world) == 1

    def test_it_refuses_to_file_an_environment_that_is_not_the_certified_one(
            self, lock_world, monkeypatch):
        # Filing a different package list beside a hash that does not describe
        # it would put two contradictory statements in the same directory.
        moved = [line.replace("2.2.6", "2.3.0") if "numpy" in line else line
                 for line in FROZEN_LINES]
        monkeypatch.setattr(producer, "normalized_freeze_lines",
                            lambda: (moved, ["causal-mllm"]))
        monkeypatch.setattr(producer, "dependency_lock_snapshot", lambda: {
            "pip_freeze_sha256": hashlib.sha256(
                freeze_text(moved).encode("utf-8")).hexdigest(),
            "python_version": "3.10.20", "pyproject_sha256": "d" * 64})
        assert producer.main(["--lock", str(lock_world)]) == 1
        assert not producer.FREEZE_PATH.exists()
        assert not producer.REPORT_PATH.exists()

    def test_it_refuses_to_file_a_different_interpreter(self, lock_world,
                                                        monkeypatch):
        monkeypatch.setattr(producer, "dependency_lock_snapshot", lambda: {
            "pip_freeze_sha256": hashlib.sha256(
                freeze_text(FROZEN_LINES).encode("utf-8")).hexdigest(),
            "python_version": "3.12.3", "pyproject_sha256": "d" * 64})
        built = producer.build(lock_world)
        assert any("the interpreter is part of what decides the last bit"
                   in issue for issue in built["issues"])

    def test_no_lock_recorded_is_a_refusal_not_an_empty_file(self, tmp_path):
        empty = tmp_path / "no-lock.yaml"
        empty.write_text(yaml.safe_dump({"models": {}}), encoding="utf-8")
        built = producer.build(empty)
        assert built.get("freeze_text") is None
        assert any("no certified environment to file" in i
                   for i in built["issues"])


class TestTheFreezeCannotSayEverything:
    def test_a_local_version_segment_is_a_gap_in_what_pip_can_reinstall(self):
        # pip freeze reports torch 2.8.0+cu128 as torch==2.8.0, so the line that
        # is committed is not the line that would put the CUDA build back.
        gaps = producer.local_version_gaps(
            {"torch": "2.8.0", "numpy": "2.2.6"},
            {"torch": "2.8.0+cu128", "numpy": "2.2.6"})
        assert sorted(gaps) == ["torch"]
        assert gaps["torch"]["freeze"] == "2.8.0"
        assert gaps["torch"]["observed_by_the_preflight"] == "2.8.0+cu128"
        assert "default-index build" in \
            gaps["torch"]["what_pip_install_r_would_fetch"]

    def test_a_version_that_merely_differs_is_not_reported_as_a_local_segment(
            self):
        assert producer.local_version_gaps(
            {"torch": "2.8.0"}, {"torch": "2.9.1"}) == {}
        assert producer.local_version_gaps({"torch": "2.8.0"}, {}) == {}

    def test_the_report_says_when_the_freeze_alone_is_not_enough(
            self, lock_world, monkeypatch):
        monkeypatch.setattr(producer, "observed_versions",
                            lambda: {"torch": "2.8.0+cu128"})
        producer.main(["--lock", str(lock_world)])
        report = json.loads(producer.REPORT_PATH.read_text(encoding="utf-8"))
        assert report["reconstructible_from_the_freeze_alone"] is False
        assert "torch" in report[
            "packages_whose_freeze_line_omits_a_local_version_segment"]
        assert "pip freeze drops a version's local segment" \
            in report["recreate_caveat"]

    def test_a_freeze_with_no_gap_promises_nothing_it_cannot_deliver(
            self, lock_world):
        producer.main(["--lock", str(lock_world)])
        report = json.loads(producer.REPORT_PATH.read_text(encoding="utf-8"))
        assert report["reconstructible_from_the_freeze_alone"] is True
        assert report["recreate_caveat"] is None


class TestNonNumericFieldsAreExactInEveryEnvironment:
    def test_a_moved_verdict_is_never_tolerated(self):
        filed = {"verdicts": {"H2": {"verdict": "confirmed", "raw_p": 0.0256}}}
        fresh = {"verdicts": {"H2": {"verdict": "refuted", "raw_p": 0.0256}}}
        comparison = reproduction.compare(filed, fresh, tolerate_numerics=True)
        assert comparison["n_nonnumeric_differences"] == 1
        code, conclusion, issues = reproduction.verdict(
            comparison, {"deviates": True, "certified": False, "reason": "x",
                         "differences": {}})
        assert code == 1
        assert conclusion == reproduction.DIFFERS
        assert any("NON-NUMERIC" in i and "verdict" in i for i in issues)

    def test_a_count_is_exact_even_while_tolerating_floats(self):
        comparison = reproduction.compare(
            {"common_panel": {"n_families_common": 98}},
            {"common_panel": {"n_families_common": 97}},
            tolerate_numerics=True)
        assert comparison["n_count_differences"] == 1
        assert comparison["numeric_differences_outside_tolerance"][0]["kind"] \
            == "count"
        assert reproduction.verdict(
            comparison, {"deviates": True, "certified": False, "reason": "x",
                         "differences": {}})[0] == 1

    def test_a_count_that_arrives_as_a_float_is_a_type_change(self):
        comparison = reproduction.compare(
            {"n": 98}, {"n": 98.0}, tolerate_numerics=True)
        assert comparison["n_nonnumeric_differences"] == 1

    def test_a_boolean_is_not_a_number(self):
        comparison = reproduction.compare(
            {"holds": True}, {"holds": 1}, tolerate_numerics=True)
        assert comparison["n_nonnumeric_differences"] == 1

    def test_a_changed_list_length_is_not_a_numeric_difference(self):
        comparison = reproduction.compare(
            {"union_excluded_cells": ["a/x"]},
            {"union_excluded_cells": ["a/x", "b/y"]},
            tolerate_numerics=True)
        assert comparison["n_nonnumeric_differences"] == 1

    def test_a_moved_family_id_is_not_a_numeric_difference(self):
        comparison = reproduction.compare(
            {"excluded": ["CMST_000001/cross_modal"]},
            {"excluded": ["CMST_000002/shuffle"]},
            tolerate_numerics=True)
        assert comparison["n_nonnumeric_differences"] == 1
        assert comparison["n_numeric_differences"] == 0


class TestTheNumericToleranceIsMeasuredAndBounded:
    def test_the_measured_worst_cases_are_inside_the_documented_tolerance(
            self):
        assert MEASURED_WORST_CONTINUOUS < reproduction.FLOAT_TOLERANCE
        assert MEASURED_REFERENCE_MEAN_DRIFT < 0
        assert abs(MEASURED_REFERENCE_MEAN_DRIFT) < reproduction.FLOAT_TOLERANCE
        assert MEASURED_WORST_P_VALUE \
            < reproduction.p_value_tolerance(5000)

    def test_the_tolerance_cannot_reach_a_decision_boundary(self):
        # Why 16 resample steps and not 160: the bound has to be smaller than
        # the distance from any confirmatory p-value to the threshold that
        # decides it, so that a p-value moving inside the tolerance cannot be
        # the reason a verdict moved. Non-numeric exactness already guarantees
        # that; this is the arithmetic that makes the guarantee unnecessary.
        tolerance = reproduction.p_value_tolerance(5000)
        assert tolerance == 0.0064
        assert tolerance < NEAREST_RAW_P_TO_ITS_HOLM_CRITICAL_VALUE
        assert tolerance < NEAREST_ADJUSTED_P_TO_ALPHA

    def test_the_p_value_tolerance_scales_with_the_resample_count(self):
        assert reproduction.p_value_tolerance(5000) == 16 * 2 / 5000
        assert reproduction.p_value_tolerance(1000) == 16 * 2 / 1000
        with pytest.raises(ValueError):
            reproduction.p_value_tolerance(0)

    def test_a_p_value_is_held_to_resample_steps_not_to_float_epsilon(self):
        comparison = reproduction.compare(
            {"verdicts": {"H2": {"raw_p": 0.0256}}},
            {"verdicts": {"H2": {"raw_p": 0.0248}}},
            tolerate_numerics=True)
        assert comparison["n_numeric_differences"] == 1
        assert comparison["numeric_differences_outside_tolerance"] == []
        assert comparison["worst_p_value_movement_in_resample_steps"] == 2

    def test_a_p_value_in_a_map_keyed_by_hypothesis_is_still_a_p_value(self):
        # The classification bug this pins: ``holm_bonferroni.raw_p.H2`` has a
        # hypothesis id as its leaf, so a leaf-name rule read it as a
        # continuous quantity and held a count of resamples to 1e-12.
        comparison = reproduction.compare(
            {"holm_bonferroni": {"raw_p": {"H2": 0.0256}}},
            {"holm_bonferroni": {"raw_p": {"H2": 0.0248}}},
            tolerate_numerics=True)
        assert comparison["n_numeric_differences"] == 1
        assert comparison["numeric_differences_outside_tolerance"] == []
        assert reproduction.is_p_value_path("$.holm_bonferroni.raw_p.H2")
        assert reproduction.is_p_value_path("$.verdicts.H2.raw_p")
        assert reproduction.is_p_value_path(
            "$.per_model.phi4_mm.estimands.order_effect"
            ".bootstrap_p_two_sided")
        assert not reproduction.is_p_value_path("$.verdicts.H2.raw_ci[0]")
        assert not reproduction.is_p_value_path(
            "$.per_model.qwen35_2b.estimands.Delta_TV.bootstrap_mean")

    def test_a_p_value_that_moves_further_than_the_bound_is_a_disagreement(
            self):
        comparison = reproduction.compare(
            {"verdicts": {"H2": {"raw_p": 0.0256}}},
            {"verdicts": {"H2": {"raw_p": 0.05}}},
            tolerate_numerics=True)
        assert comparison["numeric_differences_outside_tolerance"]

    def test_a_continuous_difference_inside_the_tolerance_is_not_one(self):
        comparison = reproduction.compare(
            {"mean": 0.11506760000000027},
            {"mean": 0.11506759999999999},
            tolerate_numerics=True)
        assert comparison["n_numeric_differences"] == 1
        assert comparison["numeric_differences_outside_tolerance"] == []
        assert comparison["worst_continuous_absolute_difference"] \
            == abs(MEASURED_REFERENCE_MEAN_DRIFT)

    def test_a_continuous_difference_outside_the_tolerance_is_one(self):
        comparison = reproduction.compare(
            {"mean": 0.1150676}, {"mean": 0.1150686}, tolerate_numerics=True)
        assert comparison["numeric_differences_outside_tolerance"]


class TestTheToleranceIsLicensedByADemonstratedDeviationOnly:
    DEVIATING = {"deviates": True, "certified": False,
                 "reason": "the active interpreter is 3.12.3 and the lock "
                           "records 3.10.20",
                 "differences": {"python_version": {"locked": "3.10.20",
                                                    "active": "3.12.3"}}}
    CERTIFIED = {"deviates": False, "certified": True,
                 "reason": "the active environment matches", "differences": {}}

    def test_inside_the_certified_environment_a_float_difference_is_a_defect(
            self):
        comparison = reproduction.compare(
            {"mean": 0.11506760000000027}, {"mean": 0.11506759999999999},
            tolerate_numerics=False)
        code, conclusion, issues = reproduction.verdict(comparison,
                                                        self.CERTIFIED)
        assert code == 1
        assert conclusion == reproduction.DIFFERS
        assert any("the last bit IS reproducible here" in i for i in issues)

    def test_under_a_demonstrated_deviation_the_same_difference_is_not(self):
        comparison = reproduction.compare(
            {"mean": 0.11506760000000027}, {"mean": 0.11506759999999999},
            tolerate_numerics=True)
        code, conclusion, issues = reproduction.verdict(comparison,
                                                        self.DEVIATING)
        assert code == 3
        assert conclusion == reproduction.WITHIN_TOLERANCE
        assert issues == []

    def test_exact_reproduction_is_a_different_answer_from_tolerated(self):
        same = {"mean": 0.11506760000000027}
        comparison = reproduction.compare(same, dict(same),
                                          tolerate_numerics=True)
        assert comparison["exactly_equal"] is True
        code, conclusion, _ = reproduction.verdict(comparison, self.DEVIATING)
        assert code == 0
        assert conclusion == reproduction.EXACT

    def test_an_uncertifiable_environment_licenses_nothing(self):
        # Not being able to tell which environment you are in is not evidence
        # that it deviates.
        unknown = {"deviates": False, "certified": False,
                   "reason": "no dependency lock is recorded", "differences": {}}
        comparison = reproduction.compare(
            {"mean": 0.1}, {"mean": 0.1 + 1e-15}, tolerate_numerics=False)
        assert reproduction.verdict(comparison, unknown)[0] == 1

    def test_the_deviation_report_separates_the_interpreter_from_the_packages(
            self):
        # A minimal interpreter has no pip at all, and denying the tolerance
        # there would deny it to exactly the machine that needs it.
        deviation = reproduction.environment_deviation(
            ROOT / "outputs" / "iteration_11" / "preflight"
            / "definitely-not-a-lock.yaml")
        assert deviation["certified"] is False
        assert deviation["deviates"] is False
        assert deviation["package_set_comparable"] is False
        assert "not licensed" in deviation["reason"]

    def test_the_recorded_lock_is_read_without_measuring_the_environment(self):
        deviation = reproduction.environment_deviation(WORLD_LOCK)
        assert deviation["recorded_python_version"] == "3.10.20"
        assert isinstance(deviation["deviates"], bool)
        assert isinstance(deviation["package_set_comparable"], bool)


class TestOneStandardGovernsEveryComparison:
    def test_no_script_restates_the_tolerance_literal(self):
        for name in ("iter11_cross_model_analysis",
                     "iter11_reference_restriction"):
            source = (ROOT / "scripts" / f"{name}.py").read_text(
                encoding="utf-8")
            assert "TOLERANCE = 1e-12" not in source, \
                f"{name}.py restates the tolerance instead of importing it " \
                f"from causal_mllm.replay.reproduction"

    def test_both_verifiers_import_the_shared_object(self):
        for name in ("iter11_cross_model_analysis",
                     "iter11_reference_restriction"):
            mod = _load_script(name)
            assert mod.TOLERANCE is reproduction.FLOAT_TOLERANCE

    def test_the_tolerance_is_the_one_the_artifacts_already_record(self):
        # iter11_reference_restriction writes its tolerance into the artifact it
        # files, so the artifact says which standard it was checked with.
        artifact = json.loads(
            (ROOT / "outputs" / "iteration_11" / "analysis" / "cross_model"
             / "reference_restriction.json").read_text(encoding="utf-8"))
        assert artifact["reproduction_check"]["tolerance"] \
            == reproduction.FLOAT_TOLERANCE



# ---------------------------------------------------------------------------
# The committed environment
# ---------------------------------------------------------------------------

RECORDED_FREEZE_SHA256 = (
    "c03a5800ca95b02003c97f35c25db744c357f6d35b216c4836b8cb32e9f91014")
RECORDED_N_PACKAGES = 100
RECORDED_PYTHON = "3.10.20"
RECORDED_NUMERIC_PACKAGES = {"numpy": "2.2.6", "scipy": "1.15.3",
                             "torch": "2.8.0", "pandas": "2.3.3"}
#: What the filed preflights observed, and what ``pip freeze`` could say about
#: it. The gap is the reason the reconstruction report exists separately from
#: the freeze text.
TORCH_LOCAL_SEGMENT = "2.8.0+cu128"
MODEL_KEYS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")


def _freeze_lines() -> list[str]:
    assert WORLD_FREEZE.exists(), \
        f"{WORLD_FREEZE} is the committed preimage of the lock's " \
        f"pip_freeze_sha256; without it the certified environment can be " \
        f"recognised but not rebuilt"
    raw = WORLD_FREEZE.read_bytes()
    assert not raw.endswith(b"\n"), \
        "a trailing newline would make the file's own bytes hash to something " \
        "other than what it lists"
    return raw.decode("utf-8").split("\n")


def _reconstruction_report() -> dict:
    assert WORLD_REPORT.exists(), \
        f"{WORLD_REPORT} is what says which script filed the freeze and how to " \
        f"recreate the environment from it"
    return json.loads(WORLD_REPORT.read_text(encoding="utf-8"))


class TestTheCommittedFreezeIsTheEnvironmentTheEvidenceWasMadeIn:
    def test_its_bytes_hash_to_the_value_every_artifact_already_binds(self):
        lines = _freeze_lines()
        assert len(lines) == RECORDED_N_PACKAGES
        assert hashlib.sha256(
            freeze_text(lines).encode("utf-8")).hexdigest() \
            == RECORDED_FREEZE_SHA256
        lock = yaml.safe_load(WORLD_LOCK.read_text(encoding="utf-8"))
        assert lock["dependency_lock"]["pip_freeze_sha256"] \
            == RECORDED_FREEZE_SHA256
        assert lock["dependency_lock"]["n_packages"] == RECORDED_N_PACKAGES
        assert lock["dependency_lock"]["python_version"] == RECORDED_PYTHON

    def test_the_four_preflight_artifacts_bind_the_same_environment(self):
        # The point of filing the preimage: it has to be the environment the
        # EVIDENCE was certified in, not merely the one on this machine today.
        for model_key in MODEL_KEYS:
            artifact = json.loads(
                (ROOT / "outputs" / "iteration_11" / "preflight" / model_key
                 / "preflight.json").read_text(encoding="utf-8"))
            assert artifact["environment"]["pip_freeze_sha256"] \
                == RECORDED_FREEZE_SHA256, model_key
            assert artifact["environment"]["n_packages"] \
                == RECORDED_N_PACKAGES, model_key
            assert artifact["lock"]["dependency_lock"]["pip_freeze_sha256"] \
                == RECORDED_FREEZE_SHA256, model_key

    def test_it_is_installable_and_carries_no_line_for_this_repository(self):
        for line in _freeze_lines():
            name, sep, version = line.partition("==")
            assert sep and name and version and " " not in line, \
                f"{line!r} is not a name==version line, so pip install -r " \
                f"would not accept this file"
            assert "causal-mllm" not in line, \
                f"{line!r} is this project's own distribution, whose freeze " \
                f"line embeds this repository's live HEAD and would make the " \
                f"committed file move on every commit"

    def test_it_names_the_packages_that_move_a_floating_point_result(self):
        lines = _freeze_lines()
        assert dict(line.split("==") for line in lines
                    if line.split("==")[0] in RECORDED_NUMERIC_PACKAGES) \
            == RECORDED_NUMERIC_PACKAGES

    def test_the_verifier_accepts_the_committed_pair(self):
        result = verify_committed_freeze(WORLD_FREEZE, WORLD_LOCK)
        assert result["issues"] == []
        assert result["matches_recorded_hash"] is True
        assert result["matches_recorded_count"] is True
        assert result["reconstructs_the_certified_environment"] is True
        assert producer.verify(WORLD_LOCK) == 0


class TestTheReconstructionReportSaysWhatTheTextCannot:
    def test_it_binds_the_same_hash_and_names_its_producer(self):
        report = _reconstruction_report()
        assert report["produced_by"] == \
            "scripts/iter11_write_dependency_lock.py"
        assert report["freeze_sha256"] == RECORDED_FREEZE_SHA256
        assert report["recorded_pip_freeze_sha256"] == RECORDED_FREEZE_SHA256
        assert report["freeze_is_the_preimage_of_the_recorded_hash"] is True
        assert report["n_packages"] == RECORDED_N_PACKAGES
        assert report["python_version"] == RECORDED_PYTHON
        assert report["numeric_packages"] == RECORDED_NUMERIC_PACKAGES

    def test_it_was_filed_from_a_clean_tree_at_an_immutable_commit(self):
        from causal_mllm.replay.registry import is_immutable_revision
        report = _reconstruction_report()
        assert is_immutable_revision(report["code_commit"]), \
            f"code_commit {report['code_commit']!r} is not a 40-hex SHA"
        assert report["git_dirty"] is False
        assert report["code_dirty_paths"] == []
        assert report["untracked_code_paths"] == []

    def test_it_binds_the_lock_digest_the_gates_compare_against(self):
        from causal_mllm.replay.registry import dependency_lock_sha256
        report = _reconstruction_report()
        assert report["dependency_lock_sha256"] \
            == dependency_lock_sha256(WORLD_LOCK)

    def test_it_records_the_one_thing_a_freeze_line_cannot_say(self):
        # pip freeze reports torch 2.8.0+cu128 as torch==2.8.0. Following the
        # recreate command literally would install the default-index build, so
        # the report has to say so rather than let the command look complete.
        report = _reconstruction_report()
        gaps = report[
            "packages_whose_freeze_line_omits_a_local_version_segment"]
        assert sorted(gaps) == ["torch"]
        assert gaps["torch"]["freeze"] == RECORDED_NUMERIC_PACKAGES["torch"]
        assert gaps["torch"]["observed_by_the_preflight"] \
            == TORCH_LOCAL_SEGMENT
        assert report["reconstructible_from_the_freeze_alone"] is False
        assert "pip freeze drops a version's local segment" \
            in report["recreate_caveat"]

    def test_the_preflight_artifacts_are_where_the_gap_was_measured(self):
        for model_key in MODEL_KEYS:
            artifact = json.loads(
                (ROOT / "outputs" / "iteration_11" / "preflight" / model_key
                 / "preflight.json").read_text(encoding="utf-8"))
            assert artifact["environment"]["observed_versions"]["torch"] \
                == TORCH_LOCAL_SEGMENT, model_key

    def test_the_recreate_command_names_the_interpreter_and_the_file(self):
        report = _reconstruction_report()
        assert RECORDED_PYTHON in report["recreate_with"]
        assert "pip install -r" in report["recreate_with"]
        assert report["freeze_path"] in report["recreate_with"]
