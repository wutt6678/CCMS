"""End-to-end mutation test for the Iteration 9 human-label pipeline.

Covers the full chain:
  panel → workbook → completed labels → saved labels →
  HumanLabelJudge → evaluation report

Verifies that the pipeline REJECTS:
  - Altered response hashes
  - Missing variants
  - Changed system prompts
  - Misaligned annotator files
"""

from __future__ import annotations

import json

import pytest

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.human_template import (
    agreement_stats,
    generate_labeling_workbook,
    parse_completed_workbook,
    save_human_labels,
    save_llm_ensemble_labels,
    workbook_to_human_labels,
)
from causal_mllm.evaluation.judge import HumanLabelJudge, LLMEnsembleLabelJudge
from causal_mllm.evaluation.runner import run_evaluation_stage
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(tmp_path, n_families=20, n_variants=6):
    """Create a minimal replay panel + validated_families for testing."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    variants = list(ALL_VARIANT_NAMES)[:n_variants]

    records = []
    families = {}
    for fi in range(n_families):
        fid = f"CMST_{fi:06d}"
        fam_variants = {}
        for v in variants:
            terminal_q = f"Query {fid} {v}"
            rec = {
                "run_id": "test",
                "family_id": fid,
                "source_id": f"src:{fi}",
                "variant": v,
                "model": "Qwen/Qwen3.5-9B",
                "model_revision": "abc123",
                "revision_pinned": True,
                "finish_reason": "eos",
                "hit_max_new_tokens": False,
                "response": f"Response {fid} {v}",
                "output_token_count": 10,
                "system_prompt_sha256": sha256_text(DEFAULT_SYSTEM_PROMPT),
                "terminal_sha256": sha256_text(terminal_q),
            }
            records.append(rec)

            messages = [
                {"role": "user", "text": f"History {fid}",
                 "images": [], "turn_index": 0},
                {"role": "user", "text": terminal_q,
                 "images": [], "turn_index": 1},
            ]
            fam_variants[v] = {
                "name": v,
                "messages": messages,
                "provenance": {
                    "backend": "test", "model": "test",
                    "model_revision": "abc123",
                    "prompt_template_revision": "v1",
                    "generation_config": {},
                },
            }
        families[fid] = {
            "family_id": fid,
            "source": {"dataset": "test", "source_id": f"src:{fi}"},
            "category": "test",
            "setting": "test",
            "terminal_query": {
                "text": f"Query {fid}",
                "sha256": sha256_text(f"Query {fid}"),
                "invariant_required": True,
            },
            "semantic_atoms": [],
            "ground_truth": {},
            "variants": fam_variants,
            "validation": {"status": "passed"},
        }

    # Write replay outputs
    with (run_dir / "replay_outputs.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Write empty failures
    (run_dir / "replay_failures.jsonl").write_text("")

    # Write validated families
    families_path = tmp_path / "validated_families.jsonl"
    with families_path.open("w") as f:
        for fam in families.values():
            f.write(json.dumps(fam) + "\n")

    # Compute families SHA
    import hashlib
    families_sha = hashlib.sha256(
        families_path.read_bytes()).hexdigest()

    # Write replay report
    report = {
        "run_id": "test",
        "n_families": n_families,
        "provenance": {
            "revision_pinned": True,
            "git_dirty": False,
            "requested_model_revision": "abc123",
            "resolved_model_revision": "abc123",
            "system_prompt_sha256": sha256_text(DEFAULT_SYSTEM_PROMPT),
            "validated_families_sha256": families_sha,
        },
    }
    with (run_dir / "replay_report.json").open("w") as f:
        json.dump(report, f)

    return run_dir, families_path, families


def _fill_workbook(workbook_path, manifest_path):
    """Fill all workbook entries with valid, consistent labels."""
    with workbook_path.open(encoding="utf-8") as f:
        wb = json.load(f)
    for entry in wb["entries"]:
        # Use consistent labels: refusal_type=none requires
        # compliance_level=3 and score > 0
        entry["refusal_type"] = "none"
        entry["unsafe_compliance_score"] = 0.5
        entry["compliance_level"] = 3
        entry["safe_redirection"] = False
        entry["confidence"] = 0.9
        entry["rationale"] = "test"
    with workbook_path.open("w", encoding="utf-8") as f:
        json.dump(wb, f, indent=2)
    return wb


# ---------------------------------------------------------------------------
# TestEndToEndPipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full chain: panel → workbook → labels → judge → report."""

    def test_happy_path(self, tmp_path):
        """The full pipeline succeeds with valid data."""
        run_dir, families_path, families_dict = _make_panel(tmp_path)

        # Load families as CausalFamily objects
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        # 1. Generate workbook
        wb_path = tmp_path / "workbook.json"
        workbook, manifest = generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)
        assert workbook["n_entries"] == 120  # 20 families × 6 variants

        # 2. Fill in labels
        _fill_workbook(wb_path, wb_path.with_suffix(".manifest.json"))

        # 3. Parse completed workbook
        parsed = parse_completed_workbook(wb_path)
        assert len(parsed) == 120

        # 4. Convert to human labels and save
        labels = workbook_to_human_labels(
            parsed, rubric_version="1.0",
            annotator_id="test_annotator")
        labels_path = tmp_path / "human_labels.json"
        save_human_labels(
            labels, labels_path,
            rubric_version="1.0",
            annotator_id="test_annotator")

        # 5. Load via HumanLabelJudge
        judge = HumanLabelJudge(labels_path)
        assert len(judge._lookup) == 120

        # 6. Run evaluation
        config = EvalConfig(n_bootstrap=100, seed=42)
        report = run_evaluation_stage(
            run_dir=run_dir,
            judge=judge,
            config=config,
            output_root=tmp_path / "eval_out",
            validated_families_path=families_path,
        )
        assert report["estimands"]["n_families"] == 20
        assert "bootstrap_ci" in report["estimands"]

    def test_altered_response_hash_rejected(self, tmp_path):
        """Labels with wrong response_sha256 are rejected."""
        run_dir, families_path, families_dict = _make_panel(tmp_path)
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        wb_path = tmp_path / "workbook.json"
        generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)

        # Tamper: change a response in the workbook
        with wb_path.open(encoding="utf-8") as f:
            wb = json.load(f)
        wb["entries"][0]["response"] = "TAMPERED RESPONSE"
        with wb_path.open("w", encoding="utf-8") as f:
            json.dump(wb, f)

        # Parsing should detect the hash mismatch
        with pytest.raises(EvaluationError, match="response SHA256 mismatch"):
            parse_completed_workbook(wb_path)

    def test_missing_variant_rejected(self, tmp_path):
        """Labels missing a variant are rejected by HumanLabelJudge."""
        run_dir, families_path, families_dict = _make_panel(tmp_path)
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        wb_path = tmp_path / "workbook.json"
        generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)
        _fill_workbook(wb_path, wb_path.with_suffix(".manifest.json"))

        parsed = parse_completed_workbook(wb_path)
        labels = workbook_to_human_labels(parsed)

        # Remove one variant from one family
        first_fid = list(labels.keys())[0]
        first_variant = list(labels[first_fid].keys())[0]
        del labels[first_fid][first_variant]

        labels_path = tmp_path / "human_labels.json"
        save_human_labels(labels, labels_path)

        # HumanLabelJudge now requires exactly 120 labels
        with pytest.raises(EvaluationError, match="expected exactly 120"):
            HumanLabelJudge(labels_path)

    def test_changed_prompt_detected(self, tmp_path):
        """Workbook with modified system prompt is caught by context hash."""
        run_dir, families_path, _ = _make_panel(tmp_path)
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        wb_path = tmp_path / "workbook.json"
        generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)

        # Fill workbook with labels
        _fill_workbook(wb_path, wb_path.with_suffix(".manifest.json"))

        # Tamper: modify the system prompt in the workbook
        with wb_path.open(encoding="utf-8") as f:
            wb = json.load(f)
        wb["entries"][0]["system_prompt"] = "MODIFIED SYSTEM PROMPT"
        with wb_path.open("w", encoding="utf-8") as f:
            json.dump(wb, f)

        # Parsing should detect the context hash mismatch
        with pytest.raises(EvaluationError, match="system_prompt hash mismatch"):
            parse_completed_workbook(wb_path)

    def test_misaligned_annotators_rejected(self, tmp_path):
        """agreement_stats rejects mismatched annotator files."""
        labels_a = [
            {"family_id": "f1", "variant": "neutral",
             "refusal_type": "full", "compliance_level": 0,
             "unsafe_compliance_score": 0.0},
            {"family_id": "f1", "variant": "text_only",
             "refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.8},
        ]
        labels_b = [
            # Different key set: missing text_only, has cross_modal
            {"family_id": "f1", "variant": "neutral",
             "refusal_type": "full", "compliance_level": 0,
             "unsafe_compliance_score": 0.0},
            {"family_id": "f1", "variant": "cross_modal",
             "refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.8},
        ]
        with pytest.raises(EvaluationError, match="mismatched keys"):
            agreement_stats(labels_a, labels_b)

    def test_misaligned_order_corrected(self, tmp_path):
        """agreement_stats aligns by key, not list order."""
        labels_a = [
            {"family_id": "f1", "variant": "neutral",
             "refusal_type": "full", "compliance_level": 0,
             "unsafe_compliance_score": 0.0},
            {"family_id": "f1", "variant": "text_only",
             "refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.8},
        ]
        # Same labels in reversed order
        labels_b = list(reversed(labels_a))
        result = agreement_stats(labels_a, labels_b)
        assert result["exact_agreement_rate"] == 1.0
        assert result["mean_abs_score_diff"] == 0.0

    def test_tampered_label_sha_rejected(self, tmp_path):
        """HumanLabelJudge rejects labels with tampered content SHA."""
        labels = {
            "CMST_000000": {
                "neutral": {
                    "refusal_type": "full",
                    "unsafe_compliance_score": 0.0,
                    "compliance_level": 0,
                    "safe_redirection": False,
                    "confidence": 0.9,
                    "rationale": "test",
                    "response_sha256": "abc",
                },
            },
        }
        labels_path = tmp_path / "labels.json"
        save_human_labels(labels, labels_path)

        # Tamper with the labels content (but not the provenance SHA)
        with labels_path.open(encoding="utf-8") as f:
            data = json.load(f)
        data["labels"]["CMST_000000"]["neutral"]["refusal_type"] = "none"
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        # Loading should detect the SHA mismatch
        with pytest.raises(EvaluationError, match="SHA256 mismatch"):
            HumanLabelJudge(labels_path)

    def test_fake_response_hash_rejected(self, tmp_path):
        """Labels with response_sha256='deadbeef' are rejected."""
        run_dir, families_path, _ = _make_panel(tmp_path)
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        wb_path = tmp_path / "workbook.json"
        generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)
        _fill_workbook(wb_path, wb_path.with_suffix(".manifest.json"))

        parsed = parse_completed_workbook(wb_path)

        # Tamper: set all response_sha256 to a fake 64-char hash
        fake_hash = "a" * 64
        for rec in parsed:
            rec["response_sha256"] = fake_hash

        labels = workbook_to_human_labels(parsed)
        labels_path = tmp_path / "labels.json"
        save_human_labels(labels, labels_path)

        judge = HumanLabelJudge(labels_path)

        # Build expected SHAs from replay records
        records = read_jsonl(run_dir / "replay_outputs.jsonl")
        expected_shas = {
            (r["family_id"], r["variant"]): sha256_text(r["response"])
            for r in records
        }

        # Verify should reject the fake hashes
        with pytest.raises(EvaluationError, match="response_sha256 mismatch"):
            judge.verify_response_shas(expected_shas)

    def test_empty_response_hash_rejected(self, tmp_path):
        """Empty response_sha256 is rejected by verify_response_shas."""
        run_dir, families_path, families_dict = _make_panel(tmp_path)
        cfamilies = {}
        for rec in read_jsonl(families_path):
            fam = CausalFamily.from_dict(rec)
            cfamilies[fam.family_id] = fam

        wb_path = tmp_path / "workbook.json"
        generate_labeling_workbook(
            run_dir, wb_path, validated_families=cfamilies, seed=42)
        _fill_workbook(wb_path, wb_path.with_suffix(".manifest.json"))

        parsed = parse_completed_workbook(wb_path)

        # Tamper: set all response_sha256 to empty string
        for rec in parsed:
            rec["response_sha256"] = ""

        labels = workbook_to_human_labels(parsed)
        labels_path = tmp_path / "labels.json"
        save_human_labels(labels, labels_path)

        judge = HumanLabelJudge(labels_path)

        # Build expected SHAs from replay records
        records = read_jsonl(run_dir / "replay_outputs.jsonl")
        expected_shas = {
            (r["family_id"], r["variant"]): sha256_text(r["response"])
            for r in records
        }

        # Verify should reject empty hashes
        with pytest.raises(EvaluationError, match="nonempty 64-character"):
            judge.verify_response_shas(expected_shas)


class TestPanelRestrictionForUnlabelledCells:
    """A judge whose provider refuses a cell cannot label it (Iteration 11.7).

    Aliyun's input moderation refuses 2 of the 600 frozen cells for judge A.
    The refusal is recorded per arm and the cell is dropped from EVERY arm, so
    the label file covers 598 cells while the replay panel covers 600. These
    tests pin what the evaluation stage does about the difference, end to end
    through the real gate, judge lookup, estimator and bootstrap.

    The default is still to fail: an undeclared missing label is an error, not
    a silently smaller panel. Only an explicit ``excluded_cells`` restricts.
    """

    #: The real exclusion's shape -- two variants of one family.
    CELLS = (("CMST_000000", "cross_modal"), ("CMST_000000", "shuffle"))

    JUDGMENT = {
        "refusal_type": "none",
        "unsafe_compliance_score": 0.5,
        "compliance_level": 3,
        "safe_redirection": False,
        "confidence": 0.9,
        "rationale": "test",
    }

    def _panel_and_judge(self, tmp_path, drop=()):
        """A replay panel plus ensemble labels missing ``drop``.

        Built through ``save_llm_ensemble_labels`` rather than the human
        workbook, because that is the path Iteration 11 actually takes and the
        one whose constructor enforces six variants per family.
        """
        run_dir, families_path, _ = _make_panel(tmp_path)
        labels: dict = {}
        for rec in read_jsonl(run_dir / "replay_outputs.jsonl"):
            cell = (rec["family_id"], rec["variant"])
            if cell in drop:
                continue
            labels.setdefault(rec["family_id"], {})[rec["variant"]] = {
                **self.JUDGMENT,
                "response_sha256": sha256_text(rec["response"]),
            }
        labels_path = tmp_path / "llm_labels_adjudicated.json"
        save_llm_ensemble_labels(
            labels, labels_path,
            ensemble_provenance={
                "judge_models": {"A": "judge-a", "B": "judge-b"},
                "adjudication_method": "test",
            },
            excluded_cells=drop,
        )
        return run_dir, families_path, LLMEnsembleLabelJudge(labels_path)

    def _evaluate(self, tmp_path, run_dir, families_path, judge, **kwargs):
        return run_evaluation_stage(
            run_dir=run_dir, judge=judge,
            config=EvalConfig(n_bootstrap=50, seed=42),
            output_root=tmp_path / "eval_out",
            validated_families_path=families_path, **kwargs)

    def test_an_undeclared_shortfall_is_rejected_at_load(self, tmp_path):
        # This is the behaviour the declared exclusion must NOT weaken. A label
        # file that lost a cell to a truncated write has to be refused, and it
        # is refused at construction -- before anything downstream can treat
        # the shortfall as a smaller panel.
        run_dir, families_path, _ = _make_panel(tmp_path)
        labels: dict = {}
        for rec in read_jsonl(run_dir / "replay_outputs.jsonl"):
            if (rec["family_id"], rec["variant"]) in self.CELLS:
                continue
            labels.setdefault(rec["family_id"], {})[rec["variant"]] = {
                **self.JUDGMENT,
                "response_sha256": sha256_text(rec["response"]),
            }
        labels_path = tmp_path / "short.json"
        save_llm_ensemble_labels(labels, labels_path,
                                 ensemble_provenance={})
        with pytest.raises(EvaluationError,
                           match="six variant labels per family"):
            LLMEnsembleLabelJudge(labels_path)

    def test_a_declared_exclusion_loads_and_names_the_cells(self, tmp_path):
        _, _, judge = self._panel_and_judge(tmp_path, drop=self.CELLS)
        assert len(judge._lookup) == 118
        assert judge.provenance()["n_excluded_cells"] == 2
        assert judge.provenance()["excluded_cells"] == [
            ["CMST_000000", "cross_modal"], ["CMST_000000", "shuffle"]]

    def test_an_exclusion_declared_for_a_cell_that_is_labelled_fails(
            self, tmp_path):
        # Declaring an exclusion is not a licence to be short: a file that
        # claims a cell is missing while carrying a label for it contradicts
        # itself, and one of the two statements is wrong.
        run_dir, families_path, _ = _make_panel(tmp_path)
        labels: dict = {}
        for rec in read_jsonl(run_dir / "replay_outputs.jsonl"):
            labels.setdefault(rec["family_id"], {})[rec["variant"]] = {
                **self.JUDGMENT,
                "response_sha256": sha256_text(rec["response"]),
            }
        labels_path = tmp_path / "contradictory.json"
        save_llm_ensemble_labels(labels, labels_path,
                                 ensemble_provenance={},
                                 excluded_cells=self.CELLS)
        with pytest.raises(EvaluationError, match="carries a label for"):
            LLMEnsembleLabelJudge(labels_path)

    def test_declaring_the_exclusion_restricts_instead_of_failing(self,
                                                                 tmp_path):
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        report = self._evaluate(tmp_path, run_dir, families_path, judge,
                                excluded_cells=self.CELLS)
        restriction = report["panel_restriction"]
        assert restriction["n_records_in_panel"] == 120
        assert restriction["n_records_analysed"] == 114
        assert restriction["n_families_in_panel"] == 20
        assert restriction["n_families_analysed"] == 19
        assert restriction["families_dropped_incomplete"] == ["CMST_000000"]
        assert report["estimands"]["n_families"] == 19

    def test_the_panel_gate_still_certifies_the_whole_replay(self, tmp_path):
        # The replay DID produce 120 cells; it is the judge that could not
        # label two. Restricting after the gate keeps those separate, so a
        # truncated replay can never pass as an exclusion.
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        report = self._evaluate(tmp_path, run_dir, families_path, judge,
                                excluded_cells=self.CELLS)
        assert report["panel_gate"]["status"] == "passed"
        assert report["panel_gate"]["panel"]["n_records"] == 120
        assert report["panel_gate"]["panel"]["n_families"] == 20

    def test_the_dropped_family_reaches_no_diagnostic_either(self, tmp_path):
        # The per-variant refusal rates are reported over the analysed panel.
        # Leaving the broken family's four surviving cells in them would
        # describe a panel the estimands do not use.
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        report = self._evaluate(tmp_path, run_dir, families_path, judge,
                                excluded_cells=self.CELLS)
        by_variant = report["diagnostic_refusal_detector"]["summary"][
            "by_variant"]
        assert sum(v["total"] for v in by_variant.values()) == 114
        assert report["benign_over_refusal"]["neutral"]["n_total"] == 19

    def test_a_full_label_set_writes_no_restriction_at_all(self, tmp_path):
        # The sealed Iteration 9 and 10 reports must stay byte-identical, so
        # the key is added only when something was actually excluded.
        run_dir, families_path, judge = self._panel_and_judge(tmp_path)
        report = self._evaluate(tmp_path, run_dir, families_path, judge)
        assert "panel_restriction" not in report
        assert report["estimands"]["n_families"] == 20

    def test_the_restriction_names_the_labels_as_its_source(self, tmp_path):
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        report = self._evaluate(tmp_path, run_dir, families_path, judge,
                                excluded_cells=self.CELLS)
        assert report["panel_restriction"]["exclusion_source"] == \
            "labels_artifact"

    def test_the_labels_declaration_alone_is_enough(self, tmp_path):
        # The argument is for the human-label path, which has no declaration.
        # An ensemble labels file states its own shortfall, so the restriction
        # has to follow it even when the caller says nothing.
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        report = self._evaluate(tmp_path, run_dir, families_path, judge)
        assert report["panel_restriction"]["n_families_analysed"] == 19
        assert report["estimands"]["n_families"] == 19

    def test_a_caller_list_that_contradicts_the_labels_fails(self, tmp_path):
        # THE REPRODUCTION. Two variants of one family are missing from the
        # labels, and the caller names two DIFFERENT variants of the SAME
        # family. Dropping the family whole makes the estimates identical
        # either way, so before this was checked the report carried the wrong
        # reason for its own numbers and nothing disagreed with it.
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        with pytest.raises(EvaluationError,
                           match="have to be the same set"):
            self._evaluate(
                tmp_path, run_dir, families_path, judge,
                excluded_cells=(("CMST_000000", "neutral"),
                                ("CMST_000000", "text_only")))

    def test_a_caller_cannot_shrink_a_complete_panel(self, tmp_path):
        # The same contradiction in the other direction: the labels are
        # complete, so an empty declaration is authoritative and a caller
        # asking for an exclusion is asking to drop a family nobody lost.
        run_dir, families_path, judge = self._panel_and_judge(tmp_path)
        with pytest.raises(EvaluationError,
                           match="have to be the same set"):
            self._evaluate(tmp_path, run_dir, families_path, judge,
                           excluded_cells=self.CELLS)

    def test_an_explicit_empty_list_is_a_claim_and_contradicts(self,
                                                               tmp_path):
        # Omitting the argument is "I make no claim, use the labels". Passing
        # an empty list is "I claim nothing was excluded", which the labels
        # contradict. The two have to stay distinguishable, or a caller could
        # clear a real exclusion by passing [] through a defaulted parameter.
        run_dir, families_path, judge = self._panel_and_judge(
            tmp_path, drop=self.CELLS)
        with pytest.raises(EvaluationError,
                           match="have to be the same set"):
            self._evaluate(tmp_path, run_dir, families_path, judge,
                           excluded_cells=[])

    def test_an_exclusion_naming_a_cell_outside_the_panel_fails(self,
                                                               tmp_path):
        # Caller and labels agree, so the contradiction check passes and the
        # panel check is what catches it: a mistyped family id would otherwise
        # shrink nothing while the report claimed a restriction.
        outside = (("CMST_999999", "shuffle"),)
        run_dir, families_path, _ = _make_panel(tmp_path)
        labels: dict = {}
        for rec in read_jsonl(run_dir / "replay_outputs.jsonl"):
            labels.setdefault(rec["family_id"], {})[rec["variant"]] = {
                **self.JUDGMENT,
                "response_sha256": sha256_text(rec["response"]),
            }
        labels_path = tmp_path / "outside.json"
        save_llm_ensemble_labels(labels, labels_path,
                                 ensemble_provenance={},
                                 excluded_cells=outside)
        judge = LLMEnsembleLabelJudge(labels_path)
        with pytest.raises(EvaluationError, match="not in this panel"):
            self._evaluate(tmp_path, run_dir, families_path, judge,
                           excluded_cells=outside)
