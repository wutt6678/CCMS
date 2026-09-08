"""Iteration 11: the exclusion is label-blind, and it is NOT outcome-independent.

One key used to carry two claims:

    "outcome_independent": "which cells a provider refuses is a function of the
     request bytes, so the surviving family set was fixed before any label
     existed"

The second half is true and is the reason the exclusion is defensible at all.
The first half is false, and filing both as one sentence let the true half carry
the false one -- "a function of the request bytes" reads as "not a function of
what the model said", and the request a provider moderates CARRIES the evaluated
response. The probe in
``outputs/iteration_11/diagnostics/judge_moderation/`` settles it by measurement
rather than argument: ``CMST_456921/text_only`` was refused in the
``ministral3_3b`` arm alone and served in the other three at the same moment,
while both ``CMST_795308`` cells were refused in all four. A refusal that tracks
the arm tracks that arm's reply.

So the metadata is now two fields -- ``label_blind`` and ``response_dependent``
-- from one module both producers import, because two copies of a correction
drift apart the same way two copies of the original claim did.

The 21 artifacts already committed are enumerated and corrected BESIDE
themselves rather than rewritten, and this file pins the enforcement in both
directions: a new carrier of the old key fails, and a carrier whose bytes moved
fails, so the correction cannot be "applied" by quietly editing the evidence it
corrects.

CI-safe: the enforcement is exercised on a miniature repository under
``tmp_path``, so a forged carrier, a rewritten seal and a regressed producer can
all be manufactured rather than waited for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from causal_mllm.evaluation.censoring import (
    CORRECTION_ARTIFACT,
    DIFFERENTIAL_CELL,
    DIFFERENTIAL_TARGET,
    EVIDENCE_ARTIFACT,
    SUPERSEDED_KEY,
    UNIFORM_CELLS,
    exclusion_metadata,
    label_blind,
    response_dependent,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


corrector = _load_script("iter11_correct_exclusion_metadata")


# ---------------------------------------------------------------------------
# The two superseded sentences, verbatim, as the committed artifacts hold them
# ---------------------------------------------------------------------------
# Quoted here rather than paraphrased: the correction has to name the exact
# string it replaces, or a consumer holding one of the 21 artifacts cannot look
# up which correction applies to the sentence in front of them.

JUDGE_SCOPE_TEXT = (
    "a provider refusal is a function of the request bytes alone, so this set "
    "was fixed before any label existed")
PANEL_SCOPE_TEXT = (
    "which cells a provider refuses is a function of the request bytes, so the "
    "surviving family set was fixed before any label existed")

ARMS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")

#: The probe's own measurement, copied so the miniature tree exercises the
#: correction with the real numbers rather than invented ones.
FULL_PAYLOAD_STATUS = {
    "CMST_456921/text_only": {"ministral3_3b": 400, "phi4_mm": 200,
                              "qwen35_2b": 200, "qwen35_4b": 200},
    "CMST_795308/cross_modal": dict.fromkeys(ARMS, 400),
    "CMST_795308/shuffle": dict.fromkeys(ARMS, 400),
}

CLEAN_PRODUCER = (
    "from causal_mllm.evaluation.censoring import exclusion_metadata\n"
    "COVERAGE = {**exclusion_metadata('the excluded set')}\n")
REGRESSED_PRODUCER = (
    "COVERAGE = {\"" + SUPERSEDED_KEY + "\": \"" + JUDGE_SCOPE_TEXT + "\"}\n")


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _carrier(payload: dict) -> str:
    return json.dumps(payload, indent=2) + "\n"


@pytest.fixture
def tree(tmp_path):
    """A miniature repository holding three carriers, the probe and both
    producers.

    Three carriers rather than one so that the enumeration has a set to disagree
    with: one carrying the judge-coverage sentence, one carrying the
    panel-restriction sentence, and one carrying both, which is what a
    ``final_evaluation_report.json`` does.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "outputs/iteration_11/judge/qwen35_2b/judge_coverage.json",
           _carrier({"n_excluded": 3, SUPERSEDED_KEY: JUDGE_SCOPE_TEXT}))
    _write(root, "outputs/iteration_11/analysis/cross_model/"
                 "cross_model_analysis.json",
           _carrier({"per_target": [
               {"panel_restriction": {SUPERSEDED_KEY: PANEL_SCOPE_TEXT}},
               {"panel_restriction": {SUPERSEDED_KEY: PANEL_SCOPE_TEXT}}]}))
    _write(root, "outputs/iteration_11/judge/qwen35_2b/"
                 "final_evaluation_report.json",
           _carrier({"panel_restriction": {SUPERSEDED_KEY: PANEL_SCOPE_TEXT},
                     "judge_coverage": {SUPERSEDED_KEY: JUDGE_SCOPE_TEXT}}))
    _write(root, EVIDENCE_ARTIFACT, _carrier({
        "full_payload_status": FULL_PAYLOAD_STATUS,
        "arms_that_disagree_now": [DIFFERENTIAL_CELL],
        "arms_unmeasured_now": [],
        "probed_at": "2026-09-06T14:02:11+00:00",
        "n_requests": 84}))
    for rel in corrector.PRODUCERS:
        _write(root, rel, CLEAN_PRODUCER)
    return root


def _filed(tree, out=None):
    """Write the correction at the path it is written at in the repository.

    Not an arbitrary name: the scan excludes this stage's own output BY PATH,
    because the artifact has to quote the key and the sentence it replaces. A
    test that filed it anywhere else would be testing a tree in which the
    exclusion cannot work, and would then report the correction as a carrier of
    the claim it corrects.
    """
    out = tree / CORRECTION_ARTIFACT if out is None else out
    assert corrector.main(["--out", str(out), "--root", str(tree)]) == 0
    return out, json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The correction keeps the two claims apart
# ---------------------------------------------------------------------------

class TestTheCorrectionSplitsTheClaimTheOldKeyConflated:
    def test_the_true_half_is_kept_and_says_which_key_it_replaces(self):
        text = label_blind("the excluded set")
        assert "before any judge has scored anything" in text
        assert SUPERSEDED_KEY in text, \
            "the surviving half must name the key it was extracted from, or a " \
            "reader of one of the 21 artifacts cannot tell this is its " \
            "replacement"
        assert "is true and is kept" in text

    def test_the_false_half_is_denied_outright(self):
        text = response_dependent("the excluded set")
        assert "NOT independent of the outcome" in text
        assert "the key that said so was wrong" in text

    def test_it_gives_the_mechanism_not_just_the_verdict(self):
        # The reason is the thing that was missed: the moderated request is not
        # the prompt, it is the prompt AND the reply being scored.
        text = response_dependent("the excluded set")
        assert "carries the evaluated response" in text

    def test_it_names_the_measurement_and_where_it_lives(self):
        text = response_dependent("the excluded set")
        assert "Measured, not argued" in text
        assert DIFFERENTIAL_CELL in text
        assert DIFFERENTIAL_TARGET in text
        assert EVIDENCE_ARTIFACT in text
        for cell in UNIFORM_CELLS:
            assert cell in text

    def test_the_uniform_pair_is_called_consistent_and_not_independent(self):
        # Uniform censoring is the easy case, and the easy case is where an
        # overclaim hides: consistent with a request-bytes cause is not the same
        # claim as independent of the response.
        text = response_dependent("the excluded set")
        assert "consistent with a request-bytes cause" in text
        assert "consistency is not independence" in text

    def test_it_points_at_the_correction_for_the_sealed_artifacts(self):
        assert CORRECTION_ARTIFACT in response_dependent("the excluded set")
        assert "rather than rewritten" in response_dependent("the excluded set")
        assert "sealed judge and evaluation evidence" \
            in response_dependent("the excluded set")

    def test_exclusion_metadata_files_exactly_two_fields(self):
        assert list(exclusion_metadata("the excluded set")) == [
            "label_blind", "response_dependent"]
        assert SUPERSEDED_KEY not in exclusion_metadata("the excluded set")

    def test_neither_field_carries_the_other_s_claim(self):
        # The defect being fixed was one sentence asserting two things, so the
        # repair has to be checked for the same shape: the label-blind half must
        # not deny outcome-dependence, and the response-dependent half must not
        # assert independence.
        meta = exclusion_metadata("the excluded set")
        assert "NOT independent" not in meta["label_blind"]
        assert "is a function of the request bytes alone" \
            not in meta["response_dependent"]

    @pytest.mark.parametrize("scope", ["the excluded set",
                                       "the surviving family set"])
    def test_the_scope_is_threaded_into_both_halves(self, scope):
        # Two producers wrote the same claim about two DIFFERENT sets. A
        # correction that does not say which set it corrects reads as a generic
        # disclaimer attached to everything.
        meta = exclusion_metadata(scope)
        assert scope in meta["label_blind"]
        assert scope in meta["response_dependent"]

    def test_two_scopes_give_two_wordings(self):
        assert exclusion_metadata("the excluded set") != \
            exclusion_metadata("the surviving family set")

    def test_there_is_one_wording_not_a_copy_per_producer(self):
        # Both producers must import it. Two transcriptions of a correction
        # drift apart exactly as the two transcriptions of the original claim
        # did -- and they had already drifted, which is why the judge-coverage
        # sentence says "alone" and the panel sentence does not.
        for rel in corrector.PRODUCERS:
            source = (ROOT / rel).read_text(encoding="utf-8")
            assert "from causal_mllm.evaluation.censoring import " \
                "exclusion_metadata" in source, rel
            assert SUPERSEDED_KEY not in source, \
                f"{rel} still writes the superseded key"


# ---------------------------------------------------------------------------
# The enumeration is load-bearing, not decorative
# ---------------------------------------------------------------------------

class TestTheCorrectionEnumeratesWhatItRefusesToRewrite:
    def test_it_finds_every_carrier_and_only_carriers(self, tree):
        found = corrector.scan(tree)
        assert sorted(found) == [
            "outputs/iteration_11/analysis/cross_model/"
            "cross_model_analysis.json",
            "outputs/iteration_11/judge/qwen35_2b/final_evaluation_report.json",
            "outputs/iteration_11/judge/qwen35_2b/judge_coverage.json"]

    def test_it_counts_occurrences_not_files(self, tree):
        found = corrector.scan(tree)
        assert found["outputs/iteration_11/judge/qwen35_2b/"
                     "judge_coverage.json"]["n_occurrences"] == 1
        assert found["outputs/iteration_11/analysis/cross_model/"
                     "cross_model_analysis.json"]["n_occurrences"] == 2
        assert found["outputs/iteration_11/judge/qwen35_2b/"
                     "final_evaluation_report.json"]["n_occurrences"] == 2

    def test_it_quotes_the_superseded_sentence_verbatim(self, tree):
        out, doc = _filed(tree)
        assert doc["distinct_superseded_values"] == [
            JUDGE_SCOPE_TEXT, PANEL_SCOPE_TEXT]
        entry = doc["artifacts"]["outputs/iteration_11/judge/qwen35_2b/"
                                 "judge_coverage.json"]
        assert entry["superseded_values"] == [JUDGE_SCOPE_TEXT]

    def test_a_carrier_of_both_sentences_is_paired_with_both_corrections(
            self, tree):
        _, doc = _filed(tree)
        both = "outputs/iteration_11/judge/qwen35_2b/final_evaluation_report.json"
        for correction in doc["corrections"]:
            assert both in correction["written_by"]

    def test_each_correction_names_the_set_it_corrects(self, tree):
        _, doc = _filed(tree)
        scopes = {c["superseded_value"]: c["scope"]
                  for c in doc["corrections"]}
        assert scopes[JUDGE_SCOPE_TEXT] == "the excluded set"
        assert scopes[PANEL_SCOPE_TEXT] == "the surviving family set"

    def test_it_says_why_the_sealed_artifacts_survive_unchanged(self, tree):
        _, doc = _filed(tree)
        why = doc["why_the_artifacts_are_not_rewritten"]
        assert "re-running a pipeline whose adjudication pass re-calls the " \
            "adjudicator" in why
        assert "delete the record that the claim was ever made" in why

    def test_it_carries_the_measurement_rather_than_restating_it(self, tree):
        _, doc = _filed(tree)
        measured = doc["measurement_behind_the_correction"]
        assert measured["full_payload_status"] == FULL_PAYLOAD_STATUS
        assert measured["arms_that_disagree_now"] == [DIFFERENTIAL_CELL]
        assert measured["differential_target"] == DIFFERENTIAL_TARGET
        assert measured["refused_in_every_arm"] == list(UNIFORM_CELLS)

    def test_the_measurement_is_reduced_to_the_thing_it_shows(self, tree):
        # One arm refusing against four is the whole argument, and it is a
        # derived number: leaving a reader to count 400s across a status table
        # is how the original overclaim survived a read.
        _, doc = _filed(tree)
        shows = doc["measurement_behind_the_correction"]["what_it_shows"]
        assert shows[DIFFERENTIAL_CELL]["n_arms_refusing"] == 1
        for cell in UNIFORM_CELLS:
            assert shows[cell]["n_arms_refusing"] == 4

    def test_a_path_outside_the_repository_is_not_called_untracked(self, tree):
        # ``git ls-files`` runs with the repository as its working directory, so
        # asking it about a scanned tree's relative path asks about a DIFFERENT
        # file that shares the name. None is "git cannot answer", and claiming
        # False would be a false statement about evidence.
        _, doc = _filed(tree)
        assert doc["n_carriers_git_cannot_answer_for"] == 3
        assert doc["n_untracked_carriers"] == 0
        assert all(e["tracked_in_git"] is None
                   for e in doc["artifacts"].values())

    def test_it_records_the_producers_it_checked(self, tree):
        _, doc = _filed(tree)
        assert sorted(doc["producers"]) == sorted(corrector.PRODUCERS)
        for state in doc["producers"].values():
            assert state["exists"] is True
            assert state["emits_superseded_key"] is False
            assert state["imports_exclusion_metadata"] is True

    def test_a_written_correction_verifies_clean(self, tree):
        out, _ = _filed(tree)
        code, issues = corrector.verify(out, root=tree)
        assert code == 0, issues
        assert issues == []


class TestTheRepairIsNotItselfACarrier:
    """The rule matches the CLAIM, not the token -- and this is why.

    The corrected wording has to name the wording it replaces, so it contains
    the superseded key as a quoted string. A scan for the bare token therefore
    flags every artifact the fixed producers write, and the forward enforcement
    becomes a gate that fails on the repair it exists to protect. This was found
    by ``test_removing_the_key_from_a_sealed_artifact_is_the_defect`` above,
    which failed because the corrected artifact still matched the scan -- not by
    reading the code.
    """

    def test_the_corrected_wording_does_not_match_the_claim_rule(self):
        text = json.dumps(exclusion_metadata("the excluded set"), indent=2)
        assert SUPERSEDED_KEY in text, \
            "the correction names the key it replaces, so the token IS present"
        assert corrector.carries_the_claim(text) == (0, 0)

    def test_a_bare_token_scan_would_have_failed_the_repair(self):
        # The obvious implementation, spelled out so the reason the rule looks
        # roundabout survives the next reader.
        text = json.dumps(exclusion_metadata("the surviving family set"))
        assert text.count(SUPERSEDED_KEY) >= 1
        assert corrector.KEY_USE_RE.search(text) is None
        assert corrector.PROSE_CLAIM_RE.search(text) is None

    def test_an_artifact_a_fixed_producer_wrote_verifies_clean(self, tree):
        out, _ = _filed(tree)
        _write(tree, "outputs/iteration_11/judge/phi4_mm/judge_coverage.json",
               _carrier({"n_excluded": 3,
                         **exclusion_metadata("the excluded set")}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 0, issues

    def test_the_correction_is_not_enumerated_in_itself(self, tree):
        # It quotes the key and the sentence, so it would match a sloppier rule.
        # It is excluded by path, and the exclusion is why ``_filed`` writes it
        # where the real run writes it.
        out, doc = _filed(tree)
        text = out.read_text(encoding="utf-8")
        assert SUPERSEDED_KEY in text
        assert JUDGE_SCOPE_TEXT in text and PANEL_SCOPE_TEXT in text
        assert doc["artifacts"], "the enumeration is not empty"
        assert corrector.scan(tree).get(CORRECTION_ARTIFACT) is None
        assert corrector.verify(out, root=tree)[0] == 0

    def test_the_two_forms_of_the_claim_are_both_matched(self):
        assert corrector.carries_the_claim(
            '{"' + SUPERSEDED_KEY + '": "x"}') == (1, 0)
        assert corrector.carries_the_claim(
            "the exclusion is outcome" + "-independent") == (0, 1)
        assert corrector.carries_the_claim(
            "no claim here at all") == (0, 0)


class TestANewCarrierFails:
    def test_an_artifact_filed_with_the_old_key_is_caught(self, tree):
        out, _ = _filed(tree)
        _write(tree, "outputs/iteration_11/judge/phi4_mm/judge_coverage.json",
               _carrier({SUPERSEDED_KEY: JUDGE_SCOPE_TEXT}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("phi4_mm/judge_coverage.json" in i and "not enumerated" in i
                   for i in issues), issues

    def test_it_says_which_of_the_two_causes_it_is(self, tree):
        out, _ = _filed(tree)
        _write(tree, "outputs/new_evidence.json",
               _carrier({SUPERSEDED_KEY: JUDGE_SCOPE_TEXT}))
        _, issues = corrector.verify(out, root=tree)
        assert any("a producer grew the superseded key back" in i
                   for i in issues), issues

    def test_a_carrier_in_the_readme_is_caught_too(self, tree):
        # The scan covers the prose as well as the artifacts: a claim corrected
        # in every JSON file but still asserted in the README is not corrected.
        # The prose form joins the two words with a hyphen, which is how the
        # judge_moderation README asserted it.
        out, _ = _filed(tree)
        _write(tree, "README.md",
               "The exclusion is outcome" + "-independent.\n")
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any(i.startswith("README.md") for i in issues), issues

    def test_a_third_superseded_wording_is_caught(self, tree):
        # Enumeration by file is not enough: the correction quotes the sentence
        # it replaces, so a new SENTENCE has to be quoted too.
        out, _ = _filed(tree)
        _write(tree, "outputs/iteration_11/judge/qwen35_2b/"
                     "judge_coverage.json",
               _carrier({SUPERSEDED_KEY: "refusals are deterministic in the "
                                         "request"}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("quoted" in i and "found" in i for i in issues), issues


class TestARewrittenSealFails:
    def test_removing_the_key_from_a_sealed_artifact_is_the_defect(self, tree):
        out, _ = _filed(tree)
        rel = "outputs/iteration_11/judge/qwen35_2b/judge_coverage.json"
        _write(tree, rel, _carrier({"n_excluded": 3,
                                    **exclusion_metadata("the excluded set")}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any(rel in i and "sealed evidence was rewritten" in i
                   for i in issues), issues

    def test_editing_a_sealed_artifact_in_place_is_caught_by_its_digest(
            self, tree):
        # The "apply the correction" move: keep the key, change the sentence.
        # The file still carries the superseded key, so only the sha256 shows it.
        out, doc = _filed(tree)
        rel = "outputs/iteration_11/judge/qwen35_2b/judge_coverage.json"
        _write(tree, rel, _carrier({
            "n_excluded": 3,
            SUPERSEDED_KEY: JUDGE_SCOPE_TEXT + " (corrected: label-blind)"}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("changed under its own correction" in i for i in issues), \
            issues
        assert doc["artifacts"][rel]["sha256"] != \
            corrector.scan(tree)[rel]["sha256"]

    def test_a_carrier_that_disappeared_entirely_is_caught(self, tree):
        out, _ = _filed(tree)
        (tree / "outputs/iteration_11/judge/qwen35_2b/"
                "judge_coverage.json").unlink()
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("no longer carries" in i for i in issues), issues

    def test_a_moved_measurement_invalidates_the_correction(self, tree):
        # The correction asserts what the probe measured. If the probe is
        # re-run and says something else, the correction has to be re-derived
        # rather than left standing on evidence it no longer has.
        out, _ = _filed(tree)
        status = {**FULL_PAYLOAD_STATUS,
                  DIFFERENTIAL_CELL: dict.fromkeys(ARMS, 400)}
        _write(tree, EVIDENCE_ARTIFACT, _carrier({
            "full_payload_status": status,
            "arms_that_disagree_now": [],
            "arms_unmeasured_now": [],
            "probed_at": "2026-09-06T14:02:11+00:00",
            "n_requests": 84}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("the measurement behind the correction has moved" in i
                   for i in issues), issues
        assert any("arms_that_disagree_now" in i for i in issues), issues

    def test_absent_evidence_is_said_so_rather_than_skipped(self, tree):
        out, _ = _filed(tree)
        (tree / EVIDENCE_ARTIFACT).unlink()
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("cannot be re-read" in i for i in issues), issues


class TestARegressedProducerFails:
    def test_a_producer_that_grows_the_key_back_fails_verification(self, tree):
        out, _ = _filed(tree)
        _write(tree, "src/causal_mllm/evaluation/runner.py",
               REGRESSED_PRODUCER)
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("runner.py" in i and "emits" in i for i in issues), issues

    def test_a_producer_that_stops_importing_the_wording_fails(self, tree):
        # Not emitting the old key is not the same as emitting the new one: a
        # producer that drops the metadata entirely has also stopped making the
        # claim it is required to make.
        out, _ = _filed(tree)
        _write(tree, "scripts/run_llm_judge_pipeline.py", "COVERAGE = {}\n")
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("does not import exclusion_metadata" in i
                   for i in issues), issues

    def test_filing_a_correction_over_a_dirty_producer_is_refused(self, tree):
        # Order matters: fix the producer first, then correct beside the output
        # it already wrote. The other order files a correction that the next run
        # contradicts.
        _write(tree, "src/causal_mllm/evaluation/runner.py",
               REGRESSED_PRODUCER)
        out = tree / "correction.json"
        assert corrector.main(["--out", str(out), "--root", str(tree)]) == 1
        assert not out.exists()

    def test_a_producer_that_no_longer_exists_is_a_finding(self, tree):
        out, _ = _filed(tree)
        (tree / "src/causal_mllm/evaluation/runner.py").unlink()
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("no longer exists" in i for i in issues), issues


class TestTheRefusalsToMeasureNothing:
    def test_no_carriers_is_not_a_passing_correction(self, tmp_path):
        root = tmp_path / "empty"
        (root / "outputs").mkdir(parents=True)
        _write(root, EVIDENCE_ARTIFACT, _carrier({
            "full_payload_status": FULL_PAYLOAD_STATUS,
            "arms_that_disagree_now": [], "arms_unmeasured_now": [],
            "probed_at": None, "n_requests": 0}))
        for rel in corrector.PRODUCERS:
            _write(root, rel, CLEAN_PRODUCER)
        out = root / CORRECTION_ARTIFACT
        assert corrector.main(["--out", str(out), "--root", str(root)]) == 2
        assert not out.exists(), \
            "an empty enumeration hashes and reads like a correction that " \
            "passed over the tree"

    def test_no_correction_to_verify_against_is_its_own_answer(self, tree):
        code, issues = corrector.verify(tree / "absent.json", root=tree)
        assert code == 2
        assert issues and "run this script without --verify" in issues[0]

    def test_a_correction_enumerating_nothing_fails(self, tree):
        out = tree / CORRECTION_ARTIFACT
        _write(out.parent, out.name, json.dumps({"artifacts": {}}))
        code, issues = corrector.verify(out, root=tree)
        assert code == 1
        assert any("enumerates no artifact at all" in i for i in issues), issues

    def test_verify_writes_nothing(self, tree):
        out, _ = _filed(tree)
        before = out.read_bytes()
        _write(tree, "outputs/new_carrier.json",
               _carrier({SUPERSEDED_KEY: JUDGE_SCOPE_TEXT}))
        assert corrector.verify(out, root=tree)[0] == 1
        assert out.read_bytes() == before

    def test_the_two_modes_are_separate_flags(self):
        # --verify is the read. Generation is the default here, as it is for the
        # media manifest, and neither one writes the other's file.
        import inspect
        source = inspect.getsource(corrector.main)
        assert "--verify" in source
        assert source.count("write_text") == 1
        assert "if args.verify:" in source


# ---------------------------------------------------------------------------
# The claim must be gone from the prose too, not only from the emitted fields
# ---------------------------------------------------------------------------

SOURCE_DIRS = ("scripts", "src", "tests")

#: Source files permitted to NAME the superseded claim, because naming it is
#: their job: one defines the rule that detects the claim and quotes the
#: sentence it replaces, the other pins that quotation against the tree. Every
#: other source file must be free of both forms.
#:
#: This is not a stylistic rule. The first pass of the rename fixed the two
#: emitted fields and left THREE docstrings still asserting the old claim -- in
#: the judge pipeline's coverage builder, in the probe script that measured the
#: differential refusal, and in the moderation test module -- because a docstring
#: is not an artifact and no gate read it. A claim that survives in the prose
#: beside the code that stopped making it is the same defect, and it is the one
#: a reader meets first.
ALLOWED_TO_NAME_THE_CLAIM = frozenset({
    "scripts/iter11_correct_exclusion_metadata.py",
    "tests/unit/test_iter11_exclusion_metadata.py",
})

#: The three the first pass left behind, named individually so a regression
#: points at the file that grew the claim back rather than at a set difference.
LEFT_BEHIND_BY_THE_FIRST_PASS = (
    "scripts/run_llm_judge_pipeline.py",
    "scripts/iter11_probe_judge_moderation.py",
    "tests/unit/test_iter11_judge_moderation.py",
)


def _source_carriers() -> dict[str, tuple[int, int]]:
    found = {}
    for name in SOURCE_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            key_uses, prose = corrector.carries_the_claim(
                path.read_text(encoding="utf-8"))
            if key_uses or prose:
                found[str(path.relative_to(ROOT))] = (key_uses, prose)
    return found


def _flat(rel: str) -> str:
    """A source file's text with the line wrapping taken out.

    Docstrings wrap mid-sentence, so a claim stated across two lines is not
    findable by substring without this -- and the whole point of the check is
    what the prose SAYS.
    """
    return " ".join((ROOT / rel).read_text(encoding="utf-8").split())


class TestNoSourceFileStillAssertsTheClaim:
    def test_the_allowlist_is_exactly_the_files_that_must_quote_it(self):
        # Both directions. A file on the list that no longer names the claim is
        # a list that has stopped being checked, and a file off it that does is
        # the defect.
        found = set(_source_carriers())
        assert found == ALLOWED_TO_NAME_THE_CLAIM, (
            f"unexpected {sorted(found - ALLOWED_TO_NAME_THE_CLAIM)}, "
            f"stale {sorted(ALLOWED_TO_NAME_THE_CLAIM - found)}")

    def test_the_two_producers_are_free_of_both_forms(self):
        found = _source_carriers()
        for rel in corrector.PRODUCERS:
            assert rel not in found, rel

    @pytest.mark.parametrize("rel", LEFT_BEHIND_BY_THE_FIRST_PASS)
    def test_the_docstrings_the_first_pass_left_behind_are_fixed(self, rel):
        assert rel not in _source_carriers()

    def test_the_coverage_builder_states_both_halves(self):
        text = _flat("scripts/run_llm_judge_pipeline.py")
        assert "The exclusion is LABEL-BLIND" in text
        assert "It is NOT independent of the outcome" in text
        assert "this docstring used to say it was" in text

    def test_the_probe_script_says_what_it_does_and_does_not_establish(self):
        # It establishes label-blindness by measurement. It never established
        # independence from the outcome, and claiming it did was the overclaim.
        text = _flat("scripts/iter11_probe_judge_moderation.py")
        assert "establishes that the exclusion is LABEL-BLIND" in text
        assert "It does NOT establish that the exclusion is independent of " \
            "the outcome" in text
        assert CORRECTION_ARTIFACT in text

    def test_the_moderation_test_module_no_longer_asserts_it(self):
        text = _flat("tests/unit/test_iter11_judge_moderation.py")
        assert "the exclusion is LABEL-BLIND" in text
        assert "It is not independent of the OUTCOME" in text

    def test_the_readme_under_outputs_states_the_correction(self):
        # Scanned by the corrector as well, since it lives under ``outputs/``:
        # its two prose assertions of the claim were carriers of it.
        readme = (ROOT / "outputs" / "iteration_11" / "diagnostics"
                  / "judge_moderation" / "README.md")
        text = " ".join(readme.read_text(encoding="utf-8").split())
        assert "The exclusion is **label-blind**" in text
        assert "It is **not independent of the outcome**" in text
        assert "consistency is not independence" in text
        assert corrector.carries_the_claim(
            readme.read_text(encoding="utf-8")) == (0, 0), \
            "the README that corrects the claim must not be a carrier of it"


# ---------------------------------------------------------------------------
# The superseded sentences quoted in the tests are the ones on disk
# ---------------------------------------------------------------------------

class TestTheQuotedSentencesAreTheRealOnes:
    """Guard against the test drifting from the evidence it describes.

    Every other test in this file works on a miniature tree, so nothing in them
    would notice if these two strings stopped being what the committed
    artifacts actually say.
    """

    def _live_values(self):
        values = set()
        for path in (ROOT / "outputs").rglob("*.json"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if SUPERSEDED_KEY not in text:
                continue
            values.update(m.group(1)
                          for m in corrector._VALUE_RE.finditer(text))
        return values

    def test_both_sentences_are_quoted_exactly(self):
        live = self._live_values()
        assert JUDGE_SCOPE_TEXT in live
        assert PANEL_SCOPE_TEXT in live

    def test_there_are_exactly_two_of_them(self):
        assert len(self._live_values()) == 2

    def test_the_scopes_are_the_two_the_producers_wrote(self):
        assert corrector.scope_of(JUDGE_SCOPE_TEXT) == "the excluded set"
        assert corrector.scope_of(PANEL_SCOPE_TEXT) == \
            "the surviving family set"
