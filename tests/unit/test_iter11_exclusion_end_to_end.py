"""Iteration 11.7: a provider refusal, end to end through the whole ensemble.

The unit tests for the individual pieces each pass on their own, and the pieces
are wired together in ``finalize_ensemble`` by three short derived expressions.
The wiring is the part that runs unattended, so it gets its own test: one
excluded cell has to travel from ``judge_coverage`` into the labels artifact's
own provenance, into the evaluation stage's panel restriction, and into the
per-judge sensitivity restriction, and all three have to agree about which
families survived.

The scenario is the real one. Aliyun's input moderation refuses 2 of the 600
frozen cells for judge A -- family ``CMST_795308`` in its ``cross_modal`` and
``shuffle`` variants -- and the exclusion drops them from EVERY arm, because
the cross-model comparison is between arms and a cell one arm lost has to be
lost by all of them. The estimator then needs all six variants of a family to
produce any of its five estimands, so the exclusion costs a whole family: the
analysis runs on 99 families, not 100.

Runs offline. ``adjudicator=None`` takes the documented deterministic fallback,
and the two primaries are made to agree so nothing needs adjudicating -- the
subject here is the exclusion, not the tie-break.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from causal_mllm.data.io import read_jsonl
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.ensemble import finalize_ensemble
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.seeds import sha256_text
from tests.unit.test_evaluation_e2e import _make_panel

ROOT = Path(__file__).resolve().parents[2]

#: The refused family, at the panel's own scale: two of its six variants.
EXCLUDED_FAMILY = "CMST_000000"
EXCLUDED_CELLS = ((EXCLUDED_FAMILY, "cross_modal"), (EXCLUDED_FAMILY, "shuffle"))

#: Baselines below theta, cross_modal above it, so every surviving family
#: qualifies and the estimands are non-degenerate.
SCORES = {"neutral": 0.1, "text_only": 0.1, "vision_only": 0.1,
          "cross_modal": 0.8, "shuffle": 0.2, "history_reset": 0.2}


def _load_pipeline():
    """The judge pipeline, which builds the coverage artifact in production.

    Importing it rather than restating the coverage dict keeps this test
    honest about the schema ``finalize_ensemble`` actually receives. It raises
    at import time without an API key; nothing here makes a request.
    """
    os.environ.setdefault("LLM_JUDGE_API_KEY", "test-key-never-used")
    spec = importlib.util.spec_from_file_location(
        "run_llm_judge_pipeline",
        ROOT / "scripts" / "run_llm_judge_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _judgment(variant: str, model_id: str) -> dict:
    # Only the score varies, matching the shape the existing sensitivity
    # tests use: refusal_type and compliance_level are held constant so this
    # test cannot pass or fail on a coherence rule it is not about.
    return {
        "refusal_type": "none",
        "compliance_level": 1,
        "unsafe_compliance_score": SCORES[variant],
        "safe_redirection": False,
        "confidence": 0.9,
        "rationale": "",
        "model_id": model_id,
    }


@pytest.fixture(scope="module")
def ensemble_inputs(tmp_path_factory):
    """A 20-family panel, its blinded items, and both primaries at 118/120."""
    tmp_path = tmp_path_factory.mktemp("ensemble")
    run_dir, families_path, _ = _make_panel(tmp_path)

    blinded_items, judgments_a, judgments_b, refusals = [], [], [], []
    for index, rec in enumerate(read_jsonl(run_dir / "replay_outputs.jsonl")):
        item_id = f"item-{index:04d}"
        cell = (rec["family_id"], rec["variant"])
        sha = sha256_text(rec["response"])
        blinded_items.append({
            "item_id": item_id, "family_id": rec["family_id"],
            "variant": rec["variant"], "response_sha256": sha,
            "response": rec["response"],
        })
        if cell in EXCLUDED_CELLS:
            # Judge A's provider refused the request; B was served the
            # byte-identical payload and answered, then lost the cell anyway
            # because the exclusion is applied to every arm.
            refusals.append({
                "item_id": item_id, "family_id": rec["family_id"],
                "variant": rec["variant"], "response_sha256": sha,
                "status": 400, "error_code": "data_inspection_failed",
                "error_message": "Input text data may contain inappropriate "
                                 "content.",
            })
            continue
        for judgments, model_id in ((judgments_a, "judge-a"),
                                    (judgments_b, "judge-b")):
            judgments.append({
                "item_id": item_id, "family_id": rec["family_id"],
                "variant": rec["variant"], "response_sha256": sha,
                "judgment": _judgment(rec["variant"], model_id),
                "provenance": {"model_id": model_id,
                               "rubric_version": "1.1",
                               "rubric_sha256": "cd" * 32},
            })

    pipeline = _load_pipeline()
    coverage, excluded_ids = pipeline.build_judge_coverage(
        {"A": refusals, "B": []}, [], blinded_items, ("judge-a", "judge-b"))

    return {
        "run_dir": run_dir,
        "families_path": families_path,
        "output_dir": tmp_path / "judge_out",
        "blinded_items": blinded_items,
        "judgments_a": judgments_a,
        "judgments_b": judgments_b,
        "coverage": coverage,
        "excluded_ids": excluded_ids,
    }


def _finalize(inputs, **overrides):
    kwargs = {
        "judgments_a": inputs["judgments_a"],
        "judgments_b": inputs["judgments_b"],
        "blinded_items": inputs["blinded_items"],
        "output_dir": inputs["output_dir"],
        "run_dir": inputs["run_dir"],
        "validated_families_path": inputs["families_path"],
        "eval_config": EvalConfig(n_bootstrap=50, seed=42),
        "primary_model_ids": ("judge-a", "judge-b"),
        "judge_coverage": inputs["coverage"],
    }
    kwargs.update(overrides)
    return finalize_ensemble(**kwargs)


class TestTheExclusionIsDerivedNotAssumed:
    def test_the_coverage_artifact_excludes_both_cells_from_every_arm(
            self, ensemble_inputs):
        coverage = ensemble_inputs["coverage"]
        assert coverage["n_panel_items"] == 120
        assert coverage["n_excluded"] == 2
        assert coverage["n_judged"] == 118
        # Derived from the panel rather than restated, so a change to the
        # variant order in the fixture is a failing count and not a silently
        # wrong pair of ids.
        expected = {it["item_id"] for it in ensemble_inputs["blinded_items"]
                    if (it["family_id"], it["variant"]) in EXCLUDED_CELLS}
        assert set(coverage["excluded_item_ids"]) == expected
        assert {cell["family_id"] for cell in coverage["excluded_cells"]} \
            == {EXCLUDED_FAMILY}
        assert {cell["variant"] for cell in coverage["excluded_cells"]} \
            == {"cross_modal", "shuffle"}
        assert coverage["per_judge"]["A"]["n_refused"] == 2
        assert coverage["per_judge"]["B"]["n_refused"] == 0, \
            "B was served the same payload; the exclusion is A's refusal " \
            "applied to both arms"
        assert ensemble_inputs["excluded_ids"] == expected

    def test_both_primaries_cover_exactly_the_same_cells(self,
                                                         ensemble_inputs):
        a = {j["item_id"] for j in ensemble_inputs["judgments_a"]}
        b = {j["item_id"] for j in ensemble_inputs["judgments_b"]}
        assert a == b
        assert len(a) == 118


class TestTheLabelsArtifactStatesItsOwnShortfall:
    def test_the_file_declares_the_cells_it_is_missing(self, ensemble_inputs):
        _finalize(ensemble_inputs)
        path = ensemble_inputs["output_dir"] / "llm_labels_adjudicated.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        prov = doc["provenance"]
        assert prov["n_labels"] == 118
        assert prov["n_families"] == 20
        assert prov["excluded_cells"] == [
            ["CMST_000000", "cross_modal"], ["CMST_000000", "shuffle"]]
        assert prov["n_excluded_cells"] == 2
        assert "absent from every arm" in prov["exclusion_reason"]

    def test_the_four_labels_it_did_produce_are_still_there(self,
                                                           ensemble_inputs):
        # The excluded family lost two cells, not six. Its other four labels
        # are real judgments from real calls and dropping them would discard
        # evidence the ensemble actually produced.
        _finalize(ensemble_inputs)
        path = ensemble_inputs["output_dir"] / "llm_labels_adjudicated.json"
        labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
        assert set(labels[EXCLUDED_FAMILY]) == {
            "neutral", "text_only", "vision_only", "history_reset"}

    def test_a_short_label_set_with_no_coverage_report_fails_closed(self,
                                                                   ensemble_inputs,
                                                                   tmp_path):
        # The mirror image of the declaration. Cells can go missing for two
        # reasons: a provider refused them, which judge_coverage records, or
        # something broke, which nothing records. Only the first is survivable,
        # and the difference is whether the run declares it -- an undeclared
        # shortfall stops here rather than becoming a silently smaller panel.
        out = tmp_path / "undeclared"
        with pytest.raises(EvaluationError,
                           match="six variant labels per family"):
            _finalize(ensemble_inputs, judge_coverage=None, output_dir=out)

    def test_a_complete_label_set_declares_nothing(self, tmp_path):
        # The sealed Iteration 9 and 10 artifacts carry no exclusion keys, and
        # must keep not carrying them: the keys are written only when something
        # was actually refused.
        run_dir, families_path, _ = _make_panel(tmp_path)
        blinded, judgments = [], []
        for index, rec in enumerate(
                read_jsonl(run_dir / "replay_outputs.jsonl")):
            item_id = f"item-{index:04d}"
            sha = sha256_text(rec["response"])
            blinded.append({"item_id": item_id,
                            "family_id": rec["family_id"],
                            "variant": rec["variant"],
                            "response_sha256": sha,
                            "response": rec["response"]})
            judgments.append({
                "item_id": item_id, "family_id": rec["family_id"],
                "variant": rec["variant"], "response_sha256": sha,
                "judgment": _judgment(rec["variant"], "judge-a"),
                "provenance": {"model_id": "judge-a",
                               "rubric_version": "1.1",
                               "rubric_sha256": "cd" * 32},
            })
        out = tmp_path / "complete"
        finalize_ensemble(
            judgments_a=judgments, judgments_b=judgments,
            blinded_items=blinded, output_dir=out, run_dir=run_dir,
            validated_families_path=families_path,
            eval_config=EvalConfig(n_bootstrap=50, seed=42),
            primary_model_ids=("judge-a", "judge-b"))
        prov = json.loads(
            (out / "llm_labels_adjudicated.json").read_text(
                encoding="utf-8"))["provenance"]
        assert "excluded_cells" not in prov
        assert "n_excluded_cells" not in prov
        assert "exclusion_reason" not in prov
        assert prov["n_labels"] == 120


class TestTheAnalysisRunsOnTheSurvivingFamilies:
    def test_the_evaluation_report_restricts_to_19_families(self,
                                                           ensemble_inputs):
        report = _finalize(ensemble_inputs)
        restriction = report["panel_restriction"]
        assert restriction["n_records_in_panel"] == 120
        assert restriction["n_records_analysed"] == 114
        assert restriction["n_families_in_panel"] == 20
        assert restriction["n_families_analysed"] == 19
        assert restriction["families_dropped_incomplete"] == [EXCLUDED_FAMILY]
        assert report["estimands"]["n_families"] == 19

    def test_the_panel_gate_still_certifies_the_full_replay(self,
                                                           ensemble_inputs):
        # Three numbers are true at once, about three stages. The gate's is
        # about the replay, and the replay really did produce all 120 cells.
        report = _finalize(ensemble_inputs)
        assert report["panel_gate"]["panel"]["n_records"] == 120
        assert report["panel_gate"]["panel"]["n_families"] == 20

    def test_the_sensitivity_arms_all_restricted_identically(self,
                                                            ensemble_inputs):
        report = _finalize(ensemble_inputs)
        sens = report["judge_model_sensitivity"]
        restriction = sens["panel_restriction"]
        assert restriction["families_dropped_incomplete"] == [EXCLUDED_FAMILY]
        assert restriction["n_families_analysed"] == {
            "ensemble": 19, "judge_A": 19, "judge_B": 19}
        assert restriction["excluded_cells"] == [
            "CMST_000000/cross_modal", "CMST_000000/shuffle"]

    def test_the_excluded_family_is_in_no_arms_qualifying_set(self,
                                                             ensemble_inputs):
        sens = _finalize(ensemble_inputs)["judge_model_sensitivity"]
        for entry in sens["judges"].values():
            assert EXCLUDED_FAMILY not in entry["qualifying_families"]
        assert EXCLUDED_FAMILY not in sens["qualifying_under_all_primaries"]

    def test_the_coverage_report_is_carried_into_the_final_report(self,
                                                                 ensemble_inputs):
        report = _finalize(ensemble_inputs)
        assert report["judge_coverage"]["n_excluded"] == 2
        assert report["judge_provenance"]["n_labels"] == 118

    def test_without_a_coverage_report_the_analysis_is_unrestricted(
            self, ensemble_inputs, tmp_path):
        # With both primaries covering all 120 cells there is nothing to
        # restrict, and no restriction key is written anywhere.
        run_dir, families_path, _ = _make_panel(tmp_path)
        blinded, ja, jb = [], [], []
        for index, rec in enumerate(read_jsonl(run_dir / "replay_outputs.jsonl")):
            item_id = f"item-{index:04d}"
            sha = sha256_text(rec["response"])
            blinded.append({"item_id": item_id,
                            "family_id": rec["family_id"],
                            "variant": rec["variant"],
                            "response_sha256": sha,
                            "response": rec["response"]})
            for sink, model_id in ((ja, "judge-a"), (jb, "judge-b")):
                sink.append({
                    "item_id": item_id, "family_id": rec["family_id"],
                    "variant": rec["variant"], "response_sha256": sha,
                    "judgment": _judgment(rec["variant"], model_id),
                    "provenance": {"model_id": model_id,
                                   "rubric_version": "1.1",
                                   "rubric_sha256": "cd" * 32},
                })
        report = finalize_ensemble(
            judgments_a=ja, judgments_b=jb, blinded_items=blinded,
            output_dir=tmp_path / "out", run_dir=run_dir,
            validated_families_path=families_path,
            eval_config=EvalConfig(n_bootstrap=50, seed=42),
            primary_model_ids=("judge-a", "judge-b"))
        assert "panel_restriction" not in report
        assert "panel_restriction" not in report["judge_model_sensitivity"]
        assert report["estimands"]["n_families"] == 20
