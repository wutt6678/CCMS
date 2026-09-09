"""Iteration 11: the paper's numbers, derived from filed artifacts and pinned.

A paper transcribes. Somebody reads a number out of a JSON file, types it into a
table, rounds it by eye, and from that moment nothing in the repository would
notice if the artifact moved underneath it. This stage exists so that does not
happen: every number the paper quotes is READ out of a filed artifact into one
document, with the artifact it came from named beside it, and ``--verify``
rebuilds the whole document and compares it.

What is pinned here is therefore not that the numbers are the ones anybody
wanted, but that they cannot be anything other than the ones the evidence says:

* the document is a PURE function of the filed artifacts -- no timestamp, no
  commit, no host -- so ``--verify`` compares the whole thing rather than a
  subset somebody remembered to exclude;
* the arithmetic a transcription would have lost is RE-DERIVED: the Holm-Bonferroni
  step-down from the table's own raw p column, the panel counts from the
  exclusions each arm filed, the histogram shares from the counts they summarise;
* every sentence whose numerals are derived carries the lists those numerals
  count, so a count that moves moves the sentence with it, and a count written
  out in words beside a derived numeral is refused;
* the figures are specified as the DATA they plot and checked against the tables
  they duplicate, and their bytes are deliberately not bound, because a raster
  carries the version of the library that drew it;
* each input is named, hashed and tracked, so a table that cites an artifact
  cites one a reviewer can resolve.

CI-safe: every test reads committed artifacts or synthesises the shapes they
have. Nothing calls a model, a judge or the network, and nothing renders.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


numbers = _load_script("iter11_paper_numbers")

ARMS = numbers.ARMS
FILED = numbers.OUT_PATH


def _filed() -> dict:
    assert FILED.is_file(), f"{FILED} is committed evidence; run the stage"
    return json.loads(FILED.read_text(encoding="utf-8"))


def _built() -> dict:
    return numbers.build()


def _verify(tmp_path: Path, doc: dict) -> tuple[int, str, list[str]]:
    """File ``doc`` somewhere and verify it. The pin that actually holds.

    ``check_the_numbers`` is internal agreement; ``verify`` is the document
    against a re-derivation from the artifacts. A claim whose numerals are
    generated rather than typed is pinned by the second, because the first can
    only compare a document with itself.
    """
    target = tmp_path / "numbers.json"
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return numbers.verify(target)


# ---------------------------------------------------------------------------
# The document is a pure function of the filed artifacts
# ---------------------------------------------------------------------------

class TestTheDocumentIsAPureFunctionOfTheArtifacts:
    def test_building_twice_yields_the_same_bytes(self):
        first = json.dumps(_built(), indent=2, sort_keys=True, ensure_ascii=False)
        second = json.dumps(_built(), indent=2, sort_keys=True, ensure_ascii=False)
        assert first == second

    def test_nothing_in_it_records_when_or_where_it_was_written(self):
        banned = {"generated_at", "timestamp", "git_commit", "git_dirty",
                  "code_commit", "python", "host", "generated_from_commit",
                  "code_tree_dirty", "untracked_code_paths"}

        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key not in banned, f"{path}/{key}"
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(_built(), "")

    def test_the_kind_and_producer_are_filed(self):
        doc = _built()
        assert doc["kind"] == numbers.KIND
        assert doc["produced_by"] == "scripts/iter11_paper_numbers.py"

    def test_it_says_what_it_is_not(self):
        doc = _built()
        not_this = doc["what_this_file_is_not"]
        assert set(not_this) == {"not_rendered", "not_prose", "not_figure_bytes",
                                 "not_a_re_analysis"}
        assert "deliberately not bound" in not_this["not_figure_bytes"]


# ---------------------------------------------------------------------------
# Every input is named, hashed and tracked
# ---------------------------------------------------------------------------

class TestEveryNumberHasAnArtifactItCameFrom:
    def test_the_inputs_are_the_declared_sources(self):
        inputs = _built()["inputs"]
        expected = numbers.source_paths()
        assert inputs["paths"] == expected
        assert inputs["n_inputs"] == len(expected) == len(set(expected.values()))
        assert inputs["missing_inputs"] == []
        assert set(inputs["sha256"]) == set(expected)

    def test_every_input_is_hashed_and_the_hash_is_of_the_file(self):
        inputs = _built()["inputs"]
        for name, rel in inputs["paths"].items():
            path = ROOT / rel
            assert path.is_file(), rel
            assert inputs["sha256"][name] == numbers.sha256_file(path), name
            assert len(inputs["sha256"][name]) == 64

    def test_every_input_is_committed(self):
        """An input nobody can resolve makes every number downstream unverifiable."""
        for rel in _built()["inputs"]["paths"].values():
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel],
                cwd=ROOT, capture_output=True, text=True)
            assert tracked.returncode == 0, (
                f"{rel} is an input to the paper's numbers and is not tracked: "
                f"{tracked.stderr.strip()}")

    def test_the_per_arm_sources_expand_to_one_file_per_arm(self):
        paths = set(numbers.source_paths().values())
        for family, pattern in numbers.PER_ARM_SOURCES.items():
            for arm in ARMS:
                rel = numbers._rel(numbers.per_arm_path(pattern, arm))
                assert rel in paths, (family, arm, rel)

    def test_a_missing_input_is_an_absence_and_not_a_silent_gap(self, monkeypatch,
                                                                tmp_path):
        monkeypatch.setattr(numbers, "IT11", tmp_path)
        with pytest.raises(SystemExit) as exc:
            numbers.build()
        assert exc.value.code == 2

    def test_a_moved_input_hash_is_a_disagreement(self):
        doc = _built()
        first = sorted(doc["inputs"]["sha256"])[0]
        doc["inputs"]["sha256"][first] = "0" * 64
        issues = numbers.check_the_numbers(doc)
        assert issues, "an input hash was edited and nothing noticed"

    def test_a_filed_hash_is_re_hashed_and_not_merely_well_formed(self):
        """A 64-character hex string is what the hash of the wrong file looks like."""
        doc = _built()
        name = sorted(doc["inputs"]["sha256"])[0]
        doc["inputs"]["sha256"][name] = "a" * 64
        issues = numbers.check_the_numbers(doc)
        assert any("the paper would be citing bytes that are not" in i
                   for i in issues), issues

    def test_an_input_that_is_not_a_file_here_is_a_finding(self, monkeypatch,
                                                          tmp_path):
        doc = _built()
        monkeypatch.setattr(numbers, "REPO_ROOT", tmp_path)
        issues = numbers.check_the_numbers(doc)
        assert any("is not a file here" in i for i in issues), issues


class TestTheCloseoutManifestIsNotAnInput:
    """The one artifact this stage must not bind, and the reason is a cycle.

    The closeout manifest binds this file: ``paper/`` is one of its bound trees
    and this document is one of its entry points. Quoting the manifest's roll-up
    here would mean two documents each carrying the other's hash, and there would
    be no fixed point to reach -- rewriting the manifest moves the hash this file
    files, regenerating this file moves the bytes the manifest binds, and
    rewriting the manifest again moves the hash once more. A document cannot carry
    its own hash inside itself, and for the same reason it cannot carry the hash
    of a document that carries its own.

    This was found by doing it: the manifest was re-filed, this file's hash of it
    went stale, and regenerating this file invalidated the manifest that had just
    been written.
    """

    def test_it_is_not_in_the_input_list(self):
        paths = _built()["inputs"]["paths"]
        assert "evidence_manifest" not in paths
        assert not any("iteration_11_evidence_manifest" in rel
                       for rel in paths.values()), sorted(paths.values())

    def test_no_table_row_is_sourced_from_it(self):
        for name, block in _built()["tables"].items():
            for row in block["rows"]:
                assert row.get("where") != "evidence_manifest", (name, row)

    def test_sourcing_a_row_from_it_again_is_refused(self):
        doc = _built()
        doc["tables"]["evidence_binding"]["rows"].append(
            {"claim": "roll-up sha256 of the bound set", "value": "b" * 64,
             "format": "sha8", "where": "evidence_manifest"})
        issues = numbers.check_the_numbers(doc)
        assert any("have no fixed point" in i for i in issues), issues

    def test_adding_it_back_to_the_input_list_is_refused(self):
        doc = _built()
        doc["inputs"]["paths"]["evidence_manifest"] = \
            "outputs/iteration_11/closeout/iteration_11_evidence_manifest.json"
        doc["inputs"]["sha256"]["evidence_manifest"] = "c" * 64
        doc["inputs"]["n_inputs"] = len(doc["inputs"]["paths"])
        issues = numbers.check_the_numbers(doc)
        assert any("the input list has the same cycle" in i for i in issues), issues

    def test_the_paper_cites_the_command_that_re_derives_the_closeout(self):
        """A procedure is stable; a count of what was bound is not."""
        block = _built()["tables"]["evidence_binding"]
        rows = {row["claim"]: row for row in block["rows"]}
        assert "--verify" in rows["what re-derives the closeout manifest"]["value"]
        assert "--deep" in rows[
            "what executes every gate the closeout points at"]["value"]
        assert "--verify" in rows[
            "what re-derives every number in this table"]["value"]

    def test_the_omission_is_explied_where_a_reader_would_notice_it(self):
        rows = {row["claim"]: row
                for row in _built()["tables"]["evidence_binding"]["rows"]}
        why = rows["why this table quotes no count from the closeout manifest"]
        assert "no fixed point" in why["value"]
        assert why["where"] == "this stage's own input list"

    def test_the_check_requires_both_the_command_and_the_explanation(self):
        doc = _built()
        block = doc["tables"]["evidence_binding"]
        block["rows"] = [row for row in block["rows"]
                         if not row["claim"].startswith(
                             ("what re-derives the closeout", "why this table"))]
        issues = numbers.check_the_numbers(doc)
        assert any("cites no command that re-derives" in i for i in issues), issues
        assert any("reads as an oversight" in i for i in issues), issues

    def test_the_media_are_still_bound_by_a_count_a_total_and_a_roll_up(self):
        rows = {row["claim"]: row
                for row in _built()["tables"]["evidence_binding"]["rows"]}
        assert rows["media files bound by hash"]["value"] > 0
        assert rows["media bytes bound by hash"]["value"] > 0
        assert len(rows["media roll-up sha256"]["value"]) == 64
        assert rows["panel images referenced but absent from the manifest"][
            "value"] == 0


# ---------------------------------------------------------------------------
# Rendering specs are a closed set, so two renderers cannot guess differently
# ---------------------------------------------------------------------------

class TestEveryColumnNamesARenderingSpecSomebodyHasDefined:
    def test_every_column_format_is_in_the_whitelist(self):
        for name, block in _built()["tables"].items():
            for col in block["columns"]:
                assert col["format"] in numbers.FORMATS, (name, col)

    def test_an_unknown_spec_is_refused_at_build_time(self):
        with pytest.raises(SystemExit) as exc:
            numbers.column("mean", "Mean", "float3ish")
        assert exc.value.code == 1

    def test_every_row_carries_the_keys_its_columns_name(self):
        """Rows may carry companion fields a renderer ignores; columns may not gap."""
        for name, block in _built()["tables"].items():
            keys = {col["key"] for col in block["columns"]}
            assert keys, name
            for index, row in enumerate(block["rows"]):
                assert keys <= set(row), (name, index, keys - set(row))

    def test_the_table_labels_are_unique_and_are_citable(self):
        labels = [block["label"] for block in _built()["tables"].values()]
        assert len(labels) == len(set(labels))
        for label in labels:
            assert label.startswith("tab:"), label

    def test_a_cell_that_contradicts_its_own_spec_is_refused(self):
        doc = _built()
        block = doc["tables"]["sign_transport"]
        row = next(r for r in block["rows"] if r["raw_p"] is not None)
        row["raw_p"] = 1.5
        issues = numbers.check_the_numbers(doc)
        assert any("outside (0, 1]" in i for i in issues), issues

    def test_a_confidence_interval_whose_ends_are_reversed_is_refused(self):
        doc = _built()
        block = doc["tables"]["sign_transport"]
        low, high = block["rows"][0]["ci"]
        block["rows"][0]["ci"] = [high, low]
        issues = numbers.check_the_numbers(doc)
        assert any("exceeds its upper" in i for i in issues), issues

    def test_a_sha_abbreviated_before_render_time_is_refused(self):
        """The renderer abbreviates; the document carries the whole hash."""
        issues: list[str] = []
        numbers._check_cell("probe", 0, {"digest": "abcd1234"}, "digest", "sha8",
                            {}, issues)
        assert any("not a full sha256" in i for i in issues), issues
        issues = []
        numbers._check_cell("probe", 0, {"digest": "a" * 64}, "digest", "sha8",
                            {}, issues)
        assert issues == []

    def test_an_absent_cell_under_a_non_optional_spec_is_refused(self):
        issues: list[str] = []
        numbers._check_cell("probe", 0, {"mean": None}, "mean", "float4", {}, issues)
        assert any("non-optional" in i for i in issues), issues
        issues = []
        numbers._check_cell("probe", 0, {"mean": None}, "mean", "optional_float4",
                            {}, issues)
        assert issues == []

    def test_a_count_that_is_really_a_boolean_is_refused(self):
        """`isinstance(True, int)` is True, and a share is not a count."""
        issues: list[str] = []
        numbers._check_cell("probe", 0, {"n_items": True}, "n_items", "int",
                            {}, issues)
        assert any("not a count" in i for i in issues), issues


# ---------------------------------------------------------------------------
# The Holm-Bonferroni step-down is re-derived, not transcribed
# ---------------------------------------------------------------------------

class TestTheCorrectionIsRecomputedFromTheRawPColumn:
    def test_the_step_down_on_a_known_set(self):
        out = numbers.holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.6}, 0.05)
        assert out["order"] == ["a", "b", "c", "d"]
        assert out["n_tests"] == 4
        assert out["critical_values"] == {"a": 0.05 / 4, "b": 0.05 / 3,
                                          "c": 0.05 / 2, "d": 0.05}
        assert out["adjusted_p"]["a"] == pytest.approx(0.004)
        assert out["adjusted_p"]["b"] == pytest.approx(0.06)
        # The running maximum, not 2 * 0.03: an adjusted p that fell as the
        # step-down advanced would let a later test look stronger than an
        # earlier one that passed.
        assert out["adjusted_p"]["c"] == pytest.approx(0.06)
        assert out["adjusted_p"]["d"] == pytest.approx(0.6)
        assert list(out["adjusted_p"][key] for key in out["order"]) == sorted(
            out["adjusted_p"][key] for key in out["order"])
        assert out["rejected"] == ["a"]
        assert out["retained"] == ["b", "c", "d"]

    def test_the_adjusted_p_is_capped_at_one_and_never_decreases(self):
        out = numbers.holm({"a": 0.9, "b": 0.95}, 0.05)
        assert out["adjusted_p"]["a"] == 1.0
        assert out["adjusted_p"]["b"] == 1.0
        assert out["rejected"] == []

    def test_it_stops_at_the_first_failure_and_does_not_resume(self):
        out = numbers.holm({"a": 0.001, "b": 0.4, "c": 0.004}, 0.05)
        assert out["order"] == ["a", "c", "b"]
        assert out["rejected"] == ["a", "c"]
        out = numbers.holm({"a": 0.2, "b": 0.001}, 0.05)
        assert out["order"] == ["b", "a"]
        assert out["rejected"] == ["b"], (
            "Holm is a step-DOWN: once a p fails its critical value every later "
            "one is retained whatever it is")

    def test_ties_are_ordered_by_name_so_the_derivation_is_deterministic(self):
        first = numbers.holm({"b": 0.01, "a": 0.01}, 0.05)
        assert first["order"] == ["a", "b"]

    def test_the_filed_table_agrees_with_a_fresh_step_down(self):
        doc = _built()
        assert numbers.check_the_numbers(doc) == []

    def test_a_moved_adjusted_p_is_refused(self):
        doc = _built()
        block = doc["tables"]["sign_transport"]
        row = next(r for r in block["rows"]
                   if r["raw_p"] is not None and r["adjusted_p"] is not None)
        row["adjusted_p"] = 0.5
        issues = numbers.check_the_numbers(doc)
        assert any("Holm" in i or "adjusted" in i for i in issues), issues

    def test_a_moved_raw_p_moves_the_correction_with_it(self):
        """The adjusted column is derived, so it cannot be edited on its own."""
        doc = _built()
        block = doc["tables"]["sign_transport"]
        row = next(r for r in block["rows"] if r["raw_p"] is not None)
        row["raw_p"] = 0.4
        issues = numbers.check_the_numbers(doc)
        assert issues, "a raw p was edited and the adjusted column still agreed"

    def test_a_moved_critical_value_is_refused(self):
        doc = _built()
        block = doc["tables"]["bound_configurations"]
        block["rows"][0]["raw_p"] = 0.9
        issues = numbers.check_the_numbers(doc)
        assert issues


# ---------------------------------------------------------------------------
# A bootstrap p has a floor; an exact p does not
# ---------------------------------------------------------------------------

class TestThePFloorIsAppliedToTheColumnsItBelongsTo:
    def test_the_floor_is_one_over_the_resamples(self):
        floor = _built()["the_p_floor"]
        assert floor["the_smallest_p_it_can_report"] == pytest.approx(
            1.0 / floor["n_resamples"])
        assert floor["n_resamples"] == 5000

    def test_no_bootstrap_p_in_any_table_is_below_the_floor(self):
        doc = _built()
        floor = doc["the_p_floor"]["the_smallest_p_it_can_report"]
        for name, block in doc["tables"].items():
            for index, row in enumerate(block["rows"]):
                for key, value in row.items():
                    if key in numbers.BOOTSTRAP_P_COLUMNS and value is not None:
                        assert value >= floor, (name, index, key, value)

    def test_a_bootstrap_p_below_the_floor_is_refused(self):
        doc = _built()
        block = doc["tables"]["sign_transport"]
        raw = next(col["key"] for col in block["columns"]
                   if col["key"] == "raw_p")
        block["rows"][0][raw] = 1e-9
        issues = numbers.check_the_numbers(doc)
        assert any("below the floor" in i for i in issues), issues

    def test_an_exact_p_is_not_held_to_the_bootstrap_floor(self):
        """A sign test can report a p no resampling scheme could reach."""
        assert numbers.EXACT_P_COLUMNS.isdisjoint(numbers.BOOTSTRAP_P_COLUMNS)
        doc = _built()
        found = False
        for name, block in doc["tables"].items():
            for row in block["rows"]:
                for key in numbers.EXACT_P_COLUMNS:
                    if row.get(key) is not None:
                        found = True
                        assert 0.0 < row[key] <= 1.0, (name, key, row[key])
        assert found, "no exact p is filed anywhere, so the exemption is untested"


# ---------------------------------------------------------------------------
# The panel arithmetic is derived from what each arm excluded
# ---------------------------------------------------------------------------

class TestThePanelCountsComeFromTheExclusionsEachArmFiled:
    def test_the_arithmetic_is_the_one_the_evidence_supports(self):
        panel = _built()["panel"]
        assert panel["n_families_in_the_panel"] == 100
        assert panel["n_cells_no_judge_could_label"] == 3
        assert panel["n_families_dropped"] == 2
        assert panel["n_families_in_the_confirmatory_panel"] == 98
        assert panel["n_families_the_bound_restores_to"] == 99
        assert panel["n_records_in_the_panel"] == 600
        assert panel["n_records_analysed_per_arm"] == 588
        assert panel["family_the_bound_still_cannot_restore"] == "CMST_795308"

    def test_the_counts_add_up_rather_than_merely_being_filed(self):
        panel = _built()["panel"]
        assert panel["n_families_in_the_panel"] - panel["n_families_dropped"] == \
            panel["n_families_in_the_confirmatory_panel"]
        assert panel["n_families_in_the_confirmatory_panel"] + 1 == \
            panel["n_families_the_bound_restores_to"]
        assert panel["n_families_dropped"] == len(panel["families_dropped"])
        assert panel["n_cells_no_judge_could_label"] == \
            len(panel["cells_no_judge_could_label"])
        assert panel["n_records_analysed_per_arm"] == \
            panel["n_families_in_the_confirmatory_panel"] * 6

    def test_the_differential_cell_is_the_one_exactly_one_arm_lost(self):
        panel = _built()["panel"]
        assert panel["differential_cells"] == ["CMST_456921/text_only"]
        assert panel["the_differential_cell"] == "CMST_456921/text_only"
        assert panel["the_arm_that_lost_it"] == "ministral3_3b"
        assert panel["cells_missing_in_every_arm"] == [
            "CMST_795308/cross_modal", "CMST_795308/shuffle"]
        assert set(panel["differential_cells"]) | set(
            panel["cells_missing_in_every_arm"]) == set(
            panel["cells_no_judge_could_label"])

    def test_the_counts_agree_with_each_other(self):
        doc = _built()
        assert numbers.check_the_numbers(doc) == []

    def test_a_panel_count_that_disagrees_with_the_exclusions_is_refused(self):
        doc = _built()
        doc["panel"]["n_families_in_the_confirmatory_panel"] = 99
        issues = numbers.check_the_numbers(doc)
        assert issues, "the panel block was edited and nothing noticed"

    def test_the_exclusions_table_is_derived_arm_independently(self):
        """`exclusion_origin` is arm-relative; `refused_in_targets` is not.

        Read from the arm that lost the cell, `exclusion_origin` says
        `this_target`, so a table built from it calls nothing differential. The
        first draft of this stage did exactly that and filed an empty list.
        """
        block = _built()["tables"]["panel_exclusions"]
        assert len(block["rows"]) == 3
        differential = [row for row in block["rows"] if row["differential"]]
        assert len(differential) == 1
        assert differential[0]["cell"] == "CMST_456921/text_only"
        assert differential[0]["arms_that_lost_it"] == "ministral3_3b"
        assert differential[0]["n_arms_that_lost_it_at_seal"] == 1

    def test_the_exclusions_table_agrees_with_the_reprobe(self):
        block = _built()["tables"]["panel_exclusions"]
        for row in block["rows"]:
            assert row["the_reprobe_reproduces_the_seal"] is True, row
            assert row["n_arms_refusing_it_on_the_reprobe"] == \
                row["n_arms_that_lost_it_at_seal"], row
            assert row["n_arms_refusing_it_on_the_reprobe"] \
                + row["n_arms_served_it_on_the_reprobe"] == len(ARMS), row
            assert row["differential"] == (row["n_arms_that_lost_it_at_seal"] == 1)

    def test_a_reprobe_that_does_not_reproduce_the_seal_is_refused(self):
        doc = _built()
        row = doc["tables"]["panel_exclusions"]["rows"][0]
        row["n_arms_refusing_it_on_the_reprobe"] = 0
        row["n_arms_served_it_on_the_reprobe"] = len(ARMS)
        issues = numbers.check_the_numbers(doc)
        assert issues, "the reprobe stopped agreeing with the seal and nothing said"


# ---------------------------------------------------------------------------
# The bound's two facts are kept apart in the tables too
# ---------------------------------------------------------------------------

class TestTheBoundIsRenderedWithoutInvertingItsOwnFields:
    @staticmethod
    def _rows(doc: dict) -> dict[str, dict]:
        return {row["arm_key"]: row
                for row in doc["tables"]["censoring_bound"]["rows"]}

    def test_the_table_covers_the_four_targets_and_only_them(self):
        assert sorted(self._rows(_built())) == sorted(ARMS)

    def test_a_range_on_one_side_of_zero_entails_that_the_sign_cannot_flip(self):
        for arm, row in self._rows(_built()).items():
            low, high = row["mean_range_over_the_whole_rubric"]
            one_side = low > 0 or high < 0
            assert row["sign_can_flip"] is (not one_side), (arm, low, high)

    def test_survives_is_not_a_column_about_matching_the_reference(self):
        """The trap this stage exists to keep out of the paper.

        All four targets' verdicts survive the bound -- including the three that
        reverse. Reading `survives` as "held up like the reference" would report
        three reversals as three failures to survive, which is the opposite of
        what the evidence says: the reversal is what survived.
        """
        doc = _built()
        bound = self._rows(doc)
        sign = {row["arm_key"]: row
                for row in doc["tables"]["sign_transport"]["rows"]}
        assert all(bound[arm]["survives"] is True for arm in ARMS)
        matching = sorted(arm for arm in ARMS
                          if sign[arm]["sign_matches_reference"])
        assert matching == ["qwen35_4b"]
        assert len(matching) < len(ARMS), (
            "if every target matched, `survives` and `matches` would coincide and "
            "this test would not be able to tell them apart")

    def test_the_reversals_hold_at_every_admissible_score(self):
        doc = _built()
        bound = self._rows(doc)
        sign = {row["arm_key"]: row
                for row in doc["tables"]["sign_transport"]["rows"]}
        reversing = [arm for arm in ARMS if not sign[arm]["sign_matches_reference"]]
        assert len(reversing) == 3
        for arm in reversing:
            row = bound[arm]
            assert row["sign_can_flip"] is False, arm
            assert row["verdict_at_99"] == row["verdict_filed"] == "refuted", arm
            low, high = row["mean_range_over_the_whole_rubric"]
            assert high < 0, (arm, low, high)

    def test_the_bound_in_counts_agrees_with_the_table(self):
        doc = _built()
        counts = doc["the_bound_in_counts"]
        rows = doc["tables"]["censoring_bound"]["rows"]
        assert counts["n_hypotheses"] == len(rows) == len(ARMS)
        assert counts["n_surviving_the_genuine_worst_case"] == \
            sum(1 for row in rows if row["survives"])
        assert counts["n_surviving_the_fully_adversarial_bound"] == \
            sum(1 for row in rows if row["survives_the_unlicensed_bound"])
        assert counts["the_cell_ranged_over"] == \
            doc["panel"]["the_differential_cell"]
        assert counts["the_score_range_bounded"] == [0.0, 1.0]

    def test_the_configurations_table_carries_every_arm_and_configuration(self):
        block = _built()["tables"]["bound_configurations"]
        pairs = {(row["configuration"], row["arm"]) for row in block["rows"]}
        assert len(block["rows"]) == len(pairs), "a duplicated cell"
        assert {arm for _, arm in pairs} == set(ARMS)
        assert len({config for config, _ in pairs}) * len(ARMS) == len(pairs)


# ---------------------------------------------------------------------------
# The figures are the data they plot, and the tables' data
# ---------------------------------------------------------------------------

class TestTheFiguresCarryDataAndNotBytes:
    def test_no_figure_specifies_a_file_or_a_hash(self):
        for name, spec in _built()["figures"].items():
            rendered = json.dumps(spec)
            assert "sha256" not in rendered, name
            assert not str(spec.get("file", "")).endswith((".png", ".pdf", ".svg")), \
                name
            assert spec["file_stem"], name

    def test_the_forest_plot_plots_the_sign_table(self):
        doc = _built()
        forest = doc["figures"]["sign_forest"]
        rows = {row["arm_key"]: row
                for row in doc["tables"]["sign_transport"]["rows"]}
        assert forest["source_table"] == doc["tables"]["sign_transport"]["label"]
        assert [series["key"] for series in forest["series"]] == forest["order"]
        assert len(forest["series"]) == len(rows)
        for series in forest["series"]:
            row = rows[series["key"]]
            assert series["mean"] == row["bootstrap_mean"], series["key"]
            assert series["ci"] == row["ci"], series["key"]
            assert series["sign"] == row["sign"]
            assert series["verdict"] == row["verdict"]

    def test_the_forest_plot_marks_exactly_one_reference(self):
        forest = _built()["figures"]["sign_forest"]
        references = [s["key"] for s in forest["series"] if s["is_the_reference"]]
        assert references == ["reference_9b"]
        assert forest["zero_line"] == 0.0

    def test_the_bound_range_plot_plots_the_bound_table(self):
        doc = _built()
        spec = doc["figures"]["bound_ranges"]
        rows = {row["arm_key"]: row
                for row in doc["tables"]["censoring_bound"]["rows"]}
        assert spec["source_table"] == doc["tables"]["censoring_bound"]["label"]
        assert sorted(series["key"] for series in spec["series"]) == sorted(ARMS)
        for series in spec["series"]:
            row = rows[series["key"]]
            assert series["mean_range_over_the_whole_rubric"] == \
                row["mean_range_over_the_whole_rubric"], series["key"]
            assert series["mean_at_98"] == row["mean_at_98"], series["key"]
            assert series["sign_can_flip"] == row["sign_can_flip"], series["key"]
            assert series["the_point_mean_lies_inside_the_range"] == \
                (row["mean_range_over_the_whole_rubric"][0] <= row["mean_at_98"]
                 <= row["mean_range_over_the_whole_rubric"][1])

    def test_the_non_nesting_count_is_derived_and_explained(self):
        spec = _built()["figures"]["bound_ranges"]
        outside = [series["key"] for series in spec["series"]
                   if not series["the_point_mean_lies_inside_the_range"]]
        assert spec["n_targets_whose_point_mean_lies_outside_its_range"] == \
            len(outside)
        assert spec["what_the_filed_decision_says_about_the_same_count"] is not None
        assert spec["why_that_is_not_a_contradiction"]

    def test_a_figure_that_disagrees_with_its_table_is_refused(self):
        doc = _built()
        doc["figures"]["sign_forest"]["series"][0]["mean"] = 0.0
        issues = numbers.check_the_numbers(doc)
        assert issues, "a figure was edited away from the table it plots"


# ---------------------------------------------------------------------------
# Every sentence with a derived numeral carries the list it counts
# ---------------------------------------------------------------------------

class TestTheProseCountsAreGeneratedNotTyped:
    @staticmethod
    def _claims(doc: dict) -> dict[str, dict]:
        return {name: claim for name, claim in doc["claims_in_words"].items()
                if isinstance(claim, dict) and "sentence" in claim}

    def test_every_claim_states_what_it_rests_on(self):
        """A sentence, its derived numerals, and no count written out in words.

        The lists are filed where a numeral counts a SET THE PAPER NAMES -- the
        three reversing targets, the two dropped families -- so a reader can see
        which items a count covers. A bulk count like 3,034 media files has no
        such list and does not need one; what it needs is that the numeral is
        derived rather than typed, which is the re-derivation's job and is
        tested below.
        """
        claims = self._claims(_built())
        assert len(claims) >= 8
        for name, claim in claims.items():
            sentence = claim["sentence"]
            assert sentence.strip(), name
            assert isinstance(claim["numerals"], dict), name
            assert isinstance(claim["the_lists_those_numerals_count"], dict), name
            for numeral, value in claim["numerals"].items():
                assert str(value) in sentence, (
                    f"{name}: {numeral}={value} does not appear in its own "
                    f"sentence, so the sentence would survive the count "
                    f"changing")

    def test_a_numeral_edited_out_of_its_own_sentence_is_refused(self):
        doc = _built()
        claim = doc["claims_in_words"]["the_media_claim"]
        claim["numerals"]["n_media_files"] += 1
        issues = numbers.check_the_numbers(doc)
        assert any("does not appear in its own sentence" in i for i in issues), \
            issues

    def test_a_bulk_count_is_still_pinned_by_the_re_derivation(self, tmp_path):
        """No list behind it, so the re-derivation is the only thing holding it."""
        doc = _built()
        claim = doc["claims_in_words"]["the_media_claim"]
        original = claim["numerals"]["n_media_files"]
        claim["numerals"]["n_media_files"] = original - 1
        claim["sentence"] = claim["sentence"].replace(str(original),
                                                     str(original - 1))
        assert numbers.check_the_numbers(doc) == [], (
            "internally consistent, and still wrong")
        code, _, issues = _verify(tmp_path, doc)
        assert code == 1, issues

    def test_every_filed_list_is_counted_by_a_numeral_in_its_sentence(self):
        for name, claim in self._claims(_built()).items():
            stated = set(claim["numerals"].values())
            for listed, values in claim[
                    "the_lists_those_numerals_count"].items():
                assert len(values) in stated, (
                    f"{name}: the list {listed} has {len(values)} entries and no "
                    f"numeral in the sentence counts it, so the sentence does "
                    f"not depend on it")

    def test_a_list_length_disagreeing_with_its_numeral_is_refused(self):
        doc = _built()
        claim = doc["claims_in_words"]["the_bound_claim"]
        claim["the_lists_those_numerals_count"][
            "surviving_the_unlicensed_bound"] = ["qwen35_4b"]
        assert numbers.check_the_numbers(doc), (
            "a list behind a numeral was shortened and nothing noticed")

    def test_no_claim_pairs_a_spelled_count_with_a_derived_numeral(self):
        """'three of four' beside a 3 that could move reads as prose until it does."""
        for name, claim in self._claims(_built()).items():
            spelled = numbers.SPELLED_COUNT.findall(claim["sentence"])
            assert not spelled, (
                f"{name} spells {spelled} beside the derived numerals "
                f"{sorted(claim['numerals'])}")

    def test_the_quoted_decision_agrees_with_the_derived_numbers(self):
        doc = _built()
        claim = doc["claims_in_words"]["the_transportability_claim"]
        assert claim["the_filed_sentence_it_quotes"]
        must = claim["must_agree_with"]
        assert must["artifact"] == "transportability_decision"
        assert must["filed"] == must["derived_here"], must
        for value in must["derived_here"].values():
            if isinstance(value, int):
                assert str(value) in claim["the_filed_sentence_it_quotes"], value

    def test_a_claim_whose_numeral_moves_without_its_sentence_is_refused(
            self, tmp_path):
        """The re-derivation is the pin: internal agreement cannot see this.

        4 already appears in the sentence as the denominator, so a numerator
        edited to 4 is internally consistent and still wrong.
        """
        doc = _built()
        claim = doc["claims_in_words"]["the_transportability_claim"]
        claim["numerals"]["n_reversing_it"] += 1
        assert numbers.check_the_numbers(doc) == [], (
            "this test is about the edit internal agreement cannot see")
        code, _, issues = _verify(tmp_path, doc)
        assert code == 1, issues
        assert any("claims_in_words" in i for i in issues), issues

    def test_a_claim_whose_list_moves_is_refused(self, tmp_path):
        doc = _built()
        claim = doc["claims_in_words"]["the_panel_claim"]
        claim["the_lists_those_numerals_count"]["families_dropped"] = [
            "CMST_795308"]
        code, _, issues = _verify(tmp_path, doc)
        assert code == 1, issues
        assert any("claims_in_words" in i for i in issues), issues

    def test_a_sentence_retyped_by_hand_is_refused(self, tmp_path):
        """The failure this whole stage exists to prevent."""
        doc = _built()
        doc["claims_in_words"]["the_panel_claim"]["sentence"] = \
            "The panel is 100 families; nothing was excluded."
        code, _, issues = _verify(tmp_path, doc)
        assert code == 1, issues

    def test_the_confirmatory_rule_is_read_out_of_the_analysis(self):
        rule = _built()["the_confirmatory_rule"]
        assert rule["alpha"] == 0.05
        assert rule["n_confirmatory_tests"] == len(ARMS)
        assert rule["family_wise_correction"] == "Holm-Bonferroni"
        assert rule["primary_estimand"] == "Delta_TV"
        assert "step-down" in rule["holm_rule"]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_a_freshly_built_document_verifies(self, tmp_path):
        target = tmp_path / "numbers.json"
        target.write_text(json.dumps(_built(), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        code, conclusion, issues = numbers.verify(target)
        assert code == 0, (conclusion, issues)
        assert conclusion == "reproduced_exactly"

    def test_verify_writes_nothing(self, tmp_path):
        target = tmp_path / "numbers.json"
        target.write_text(json.dumps(_built(), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        before = target.read_bytes()
        numbers.verify(target)
        assert target.read_bytes() == before

    def test_a_hand_edited_number_is_caught(self, tmp_path):
        """The whole reason this stage exists."""
        doc = _built()
        doc["tables"]["sign_transport"]["rows"][0]["mean"] = 0.1234
        target = tmp_path / "numbers.json"
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        code, conclusion, issues = numbers.verify(target)
        assert code == 1, (conclusion, issues)
        assert any("tables" in i for i in issues), issues

    def test_an_unreadable_file_is_a_contradiction_not_an_absence(self, tmp_path):
        target = tmp_path / "numbers.json"
        target.write_text("{not json", encoding="utf-8")
        code, conclusion, issues = numbers.verify(target)
        assert code == 1 and conclusion == "unreadable", (code, issues)

    def test_the_committed_numbers_verify(self):
        code, conclusion, issues = numbers.verify()
        assert code == 0, (conclusion, issues)

    def test_the_committed_document_agrees_with_itself(self):
        assert numbers.check_the_numbers(_filed()) == []


# ---------------------------------------------------------------------------
# Exit codes: one test per documented code
# ---------------------------------------------------------------------------

class TestTheExitCodes:
    def test_zero_when_the_filed_numbers_re_derive(self, tmp_path):
        target = tmp_path / "numbers.json"
        target.write_text(json.dumps(_built(), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        assert numbers.main(["--verify", "--out", str(target)]) == 0

    def test_one_when_they_do_not(self, tmp_path):
        doc = _built()
        doc["panel"]["n_families_confirmatory"] = 99
        target = tmp_path / "numbers.json"
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        assert numbers.main(["--verify", "--out", str(target)]) == 1

    def test_two_when_nothing_is_filed_to_verify_against(self, tmp_path):
        assert numbers.main(
            ["--verify", "--out", str(tmp_path / "absent.json")]) == 2

    def test_two_when_an_input_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(numbers, "IT11", tmp_path)
        with pytest.raises(SystemExit) as exc:
            numbers.build()
        assert exc.value.code == 2

    def test_fatal_exits_with_the_code_it_is_given(self):
        with pytest.raises(SystemExit) as exc:
            numbers.fatal("a missing input", 2)
        assert exc.value.code == 2

    def test_three_is_documented_as_unreachable_here(self):
        """Every input is committed, so no checkout that can run this lacks one."""
        assert "3  not reachable for this stage" in numbers.__doc__
        for rel in _built()["inputs"]["paths"].values():
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel],
                cwd=ROOT, capture_output=True, text=True)
            assert tracked.returncode == 0, rel

    def test_write_and_verify_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            numbers.main(["--verify", "--write"])
        assert exc.value.code == 2

    def test_write_refuses_a_document_that_disagrees_with_itself(self, tmp_path,
                                                                  monkeypatch):
        """The gate runs before the write, so a bad filing never reaches disk."""
        target = tmp_path / "numbers.json"
        doc = _built()
        doc["inputs"]["sha256"][sorted(doc["inputs"]["sha256"])[0]] = "0" * 64
        monkeypatch.setattr(numbers, "build", lambda: doc)
        assert numbers.main(["--write", "--out", str(target)]) == 1
        assert not target.exists()

    def test_write_produces_a_document_that_verifies(self, tmp_path):
        target = tmp_path / "numbers.json"
        assert numbers.main(["--write", "--out", str(target)]) == 0
        assert target.is_file()
        code, conclusion, issues = numbers.verify(target)
        assert code == 0, (conclusion, issues)

    def test_the_documented_codes_are_the_ones_the_script_uses(self):
        source = (ROOT / "scripts" / "iter11_paper_numbers.py").read_text(
            encoding="utf-8")
        for code in ("0", "1", "2", "3"):
            assert f"\n    {code}  " in source, (
                f"exit {code} is not documented in the module docstring")


# ---------------------------------------------------------------------------
# Display names come from the blinding audit's declared ids, not from prose
# ---------------------------------------------------------------------------

class TestTheNamesAreTheOnesTheBlindingAuditDeclared:
    def test_every_arm_has_a_display_name(self):
        names = _built()["display_names"]
        assert set(names) == set(ARMS)
        for arm in ARMS:
            assert names[arm].strip()

    def test_the_names_are_used_in_the_tables_that_carry_them(self):
        doc = _built()
        names = set(doc["display_names"].values())
        rows = {row["arm_key"]: row["row"]
                for row in doc["tables"]["sign_transport"]["rows"]}
        for arm in ARMS:
            assert rows[arm] in names, (arm, rows[arm])

    def test_no_table_leaks_the_internal_arm_key_as_its_display_name(self):
        doc = _built()
        for arm, name in doc["display_names"].items():
            assert name != arm, (
                f"{arm} is rendered as its own key, so the blinding audit's "
                f"declared model_id is not what the paper would print")

    def test_the_figures_use_the_same_names_as_the_tables(self):
        doc = _built()
        names = set(doc["display_names"].values())
        for series in doc["figures"]["sign_forest"]["series"]:
            if series["is_the_reference"]:
                continue
            assert series["label"] in names, series
        for series in doc["figures"]["bound_ranges"]["series"]:
            assert series["label"] in names, series

    def test_the_sealed_reference_is_named_as_sealed(self):
        """It was not re-run here, and the row has to say so."""
        doc = _built()
        rows = {row["arm_key"]: row for row in doc["tables"]["sign_transport"]["rows"]}
        reference = rows["reference_9b"]
        assert reference["raw_p"] is None
        assert reference["sign_matches_reference"] is None
        assert reference["verdict"] == "sealed"
        assert "sealed" in reference["what_this_row_is"]
        assert set(ARMS) == set(rows) - {"reference_9b"}
