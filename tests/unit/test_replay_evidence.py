"""Pinned evidence for Iteration 8 frozen replay runs (Scale B).

Pins the COMMITTED real-model replay artifacts:

  * smoke run: 5 families x 6 variants = 30 trajectories
  * full run: 20 families x 6 variants = 120 trajectories

Every record must carry complete provenance (run_id, family_id,
source_id, variant, model, model_revision, prompt/template revision,
generation config, response, error), deterministic settings
(temperature 0, do_sample False, max_new_tokens 256), and the token
diagnostics (input token counts; visual-token metadata > 0 exactly in
the vision-bearing variants). Failures stay in replay_failures.jsonl
and never masquerade as responses.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_mllm.data.io import read_jsonl

REPLAY_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "replay_runs"
)
SMOKE_RUN = REPLAY_ROOT / "smoke-2026-08-25-qwen35-9b"
FULL_RUN = REPLAY_ROOT / "scale-b-2026-08-25-qwen35-9b"
# Iteration-9 primary panel: the 256-token cap truncated
# condition-dependently (cross_modal 85% vs text_only 40%
# mid-sentence); escalation evidence: 512 truncated 8/27, 768 4/30,
# 1024 3/30 smoke completions (longest complete 847), so the panel
# was generated at 1536 tokens with output diagnostics
# (finish_reason / hit_max_new_tokens).
PANEL_RUN = REPLAY_ROOT / "scale-b-2026-08-27-t1536-qwen35-9b"
# v0.8.2 provenance repair: revision explicitly passed to every
# replay command. The smoke reproduces v0.8.1 byte-for-byte.
PINNED_SMOKE_RUN = REPLAY_ROOT / "smoke-2026-08-28-pinned-qwen35-9b"
PINNED_PANEL_RUN = REPLAY_ROOT / "scale-b-2026-08-28-pinned-qwen35-9b"
CAP_ESCALATION_SUMMARY = REPLAY_ROOT / "cap_escalation_summary.json"
# v0.8.3 clean-tree rerun: git_commit matches the actual code tree
# (not the base commit), git_dirty=False, expanded fingerprint
# includes prompt_template_revision + torch/CUDA versions.
FINAL_SMOKE_RUN = REPLAY_ROOT / "smoke-2026-08-28-t1536-final-qwen35-9b"
FINAL_PANEL_RUN = REPLAY_ROOT / "scale-b-2026-08-28-t1536-final-qwen35-9b"

VARIANTS = ("neutral", "text_only", "vision_only",
            "cross_modal", "shuffle", "history_reset")
VISION_VARIANTS = {"vision_only", "cross_modal", "shuffle"}
SCALE_B_FAMILIES = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "families" / "scale_b_smoke" / "validated_families.jsonl"
)

REQUIRED_FIELDS = (
    "run_id", "family_id", "source_id", "variant", "model",
    "model_revision", "prompt_template_revision", "system_prompt_sha256",
    "generation_config", "terminal_sha256", "n_images",
    "input_token_count", "image_token_count", "response", "error",
)


def _load(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    outputs = read_jsonl(run_dir / "replay_outputs.jsonl")
    failures = read_jsonl(run_dir / "replay_failures.jsonl")
    report = json.loads((run_dir / "replay_report.json")
                        .read_text(encoding="utf-8"))
    return outputs, failures, report


def _assert_records(run_dir: Path, n_families: int):
    outputs, failures, report = _load(run_dir)
    expected = n_families * 6
    assert report["expected_attempts"] == expected
    assert report["n_attempted"] == expected
    assert report["n_succeeded"] == expected
    assert report["n_failed"] == 0
    assert report["missing_variants"] == []
    assert failures == []
    assert len(outputs) == expected

    # Complete (family, variant) coverage — no missing variant
    pairs = {(r["family_id"], r["variant"]) for r in outputs}
    assert len(pairs) == expected

    for record in outputs:
        for field in REQUIRED_FIELDS:
            assert field in record, (record["variant"], field)
        assert record["error"] is None
        assert isinstance(record["response"], str)
        assert record["response"] != ""
        assert record["run_id"] == run_dir.name
        assert record["model"] == "Qwen/Qwen3.5-9B"
        assert record["model_revision"]
        assert record["prompt_template_revision"] == "v1"
        # Deterministic frozen settings, identical for every variant
        assert record["generation_config"] == {
            "temperature": 0.0, "top_p": 1.0, "do_sample": False,
            "max_new_tokens": 256, "seed": 42,
        }
        # Token diagnostics from the ACTUAL target tokenizer
        assert record["input_token_count"] > 0
        if record["variant"] in VISION_VARIANTS:
            assert record["image_token_count"] > 0
            assert record["n_images"] == 1
        else:
            assert record["image_token_count"] == 0
            assert record["n_images"] == 0

    # One system prompt / one config fingerprint for the whole run
    assert len({r["system_prompt_sha256"] for r in outputs}) == 1
    prov = report["provenance"]
    assert prov["generation_config"]["temperature"] == 0.0
    assert prov["config_sha256"]
    assert report["token_stats"]["total_input_tokens"] > 0
    return outputs, report


def _validated_source_ids() -> set:
    return {
        rec["source"]["source_id"]
        for rec in read_jsonl(SCALE_B_FAMILIES)
    }


class TestCommittedSmokeRun:
    """5-family / 30-generation real-model smoke evidence."""

    def test_smoke_run_complete(self):
        assert SMOKE_RUN.exists(), "committed smoke replay run missing"
        outputs, report = _assert_records(SMOKE_RUN, n_families=5)
        assert report["dataset"] == "scale_b_smoke"
        # The smoke run replays the FIRST five validated families
        validated = read_jsonl(SCALE_B_FAMILIES)
        expected_ids = {rec["family_id"] for rec in validated[:5]}
        assert {r["family_id"] for r in outputs} == expected_ids
        assert {r["source_id"] for r in outputs} <= _validated_source_ids()


class TestCommittedFullRun:
    """20-family / 120-generation real-model run (Iteration 8 gate)."""

    def test_full_run_120_trajectories(self):
        assert FULL_RUN.exists(), "committed full replay run missing"
        outputs, report = _assert_records(FULL_RUN, n_families=20)
        assert report["dataset"] == "scale_b_smoke"
        assert {r["source_id"] for r in outputs} == _validated_source_ids()

    def test_replay_inputs_only_from_validated_artifact(self):
        """Replay must not touch raw families.jsonl: every replayed
        family_id exists in validated_families.jsonl."""
        outputs, _, _ = _load(FULL_RUN)
        validated_ids = {
            rec["family_id"] for rec in read_jsonl(SCALE_B_FAMILIES)
        }
        assert {r["family_id"] for r in outputs} <= validated_ids


class TestCommittedIteration9Panel:
    """20-family / 120-generation panel at max_new_tokens=1536 with
    output diagnostics; the primary evidence base for Iteration 9.

    Requirement: approximately zero finish_reason=length — a truncated
    compliant answer vs a complete refusal would bias Delta_V,
    Delta_TV and Delta_order.
    """

    def test_panel_complete_with_output_diagnostics(self):
        assert PANEL_RUN.exists(), "committed 1536-token panel missing"
        outputs, failures, report = _load(PANEL_RUN)
        assert report["expected_attempts"] == 120
        assert report["n_succeeded"] == 120 and report["n_failed"] == 0
        assert failures == [] and len(outputs) == 120
        for record in outputs:
            assert record["generation_config"]["max_new_tokens"] == 1536
            assert record["generation_config"]["temperature"] == 0.0
            assert record["generation_config"]["do_sample"] is False
            assert record["output_token_count"] is not None
            assert record["finish_reason"] in {"eos", "stop", "length"}
            assert record["hit_max_new_tokens"] is not None
            assert record["model_revision"] == \
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        prov = report["provenance"]
        assert prov["resolved_sha256"]
        assert prov["config_sha256"]
        assert {r["source_id"] for r in outputs} == _validated_source_ids()

    def test_panel_truncation_near_zero_by_variant(self):
        _, _, report = _load(PANEL_RUN)
        truncation = report["truncation"]
        assert truncation["n_truncated"] == 0, truncation
        for variant, stats in truncation["by_variant"].items():
            assert stats["n"] == 20, variant
            assert stats["n_truncated"] == 0, (variant, stats)
            assert stats["truncation_rate"] == 0.0, variant


class TestV082PinnedSmoke:
    """5-family smoke with --model-revision passed explicitly.
    Must reproduce the v0.8.1 smoke byte-for-byte (same model
    weights, same prompt, same greedy decoding)."""

    def test_pinned_smoke_complete(self):
        assert PINNED_SMOKE_RUN.exists(), \
            "pinned v0.8.2 smoke run missing"
        outputs, failures, report = _load(PINNED_SMOKE_RUN)
        assert report["expected_attempts"] == 30
        assert report["n_succeeded"] == 30 and report["n_failed"] == 0
        assert failures == [] and len(outputs) == 30

    def test_pinned_smoke_revision_pinned(self):
        _, _, report = _load(PINNED_SMOKE_RUN)
        prov = report["provenance"]
        assert prov["revision_pinned"] is True
        assert prov["requested_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert prov["resolved_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        for record in read_jsonl(PINNED_SMOKE_RUN / "replay_outputs.jsonl"):
            assert record["requested_model_revision"] == \
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            assert record["resolved_model_revision"] == \
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            assert record["revision_pinned"] is True

    def test_pinned_smoke_reproduces_v081_byte_for_byte(self):
        """Same model weights + same prompt + greedy == same text."""
        pinned = read_jsonl(PINNED_SMOKE_RUN / "replay_outputs.jsonl")
        v081 = read_jsonl(
            REPLAY_ROOT / "smoke-2026-08-27-t1536-qwen35-9b" /
            "replay_outputs.jsonl")
        assert len(pinned) == len(v081) == 30
        for pinned_rec, v081_rec in zip(
                sorted(pinned, key=lambda r: (r["family_id"], r["variant"])),
                sorted(v081, key=lambda r: (r["family_id"], r["variant"]))):
            assert pinned_rec["family_id"] == v081_rec["family_id"]
            assert pinned_rec["variant"] == v081_rec["variant"]
            assert pinned_rec["response"] == v081_rec["response"], (
                f"response diverged for "
                f"{pinned_rec['family_id']}:{pinned_rec['variant']}")
            assert pinned_rec["input_token_count"] == \
                v081_rec["input_token_count"]
            assert pinned_rec["output_token_count"] == \
                v081_rec["output_token_count"]


class TestV082PinnedPanel:
    """120-response panel with --model-revision passed explicitly.
    New run id (does NOT overwrite the v0.8.1 panel)."""

    def test_pinned_panel_complete(self):
        assert PINNED_PANEL_RUN.exists(), \
            "pinned v0.8.2 panel run missing"
        outputs, failures, report = _load(PINNED_PANEL_RUN)
        assert report["expected_attempts"] == 120
        assert report["n_succeeded"] == 120 and report["n_failed"] == 0
        assert failures == [] and len(outputs) == 120

    def test_pinned_panel_revision_pinned_and_zero_truncation(self):
        outputs, _, report = _load(PINNED_PANEL_RUN)
        prov = report["provenance"]
        assert prov["revision_pinned"] is True
        assert prov["requested_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert prov["resolved_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert prov["validated_families_sha256"]
        assert prov["transformers_version"]
        assert prov["git_commit"]
        assert prov["resolved_sha256"]
        # Expanded fingerprint fields
        assert prov["processor_revision"]
        assert prov["enable_thinking"] is False
        assert prov["torch_dtype"] == "bfloat16"
        assert report["truncation"]["n_truncated"] == 0
        for record in outputs:
            assert record["revision_pinned"] is True
            assert record["finish_reason"] in {"eos", "stop"}
            assert record["hit_max_new_tokens"] is False

    def test_all_120_pinned_outputs_match_v081_panel(self):
        """All 120 pinned panel outputs must match the v0.8.1 panel
        byte-for-byte (same weights, same prompt, greedy decoding).
        This is the full-panel analogue of the 30-response smoke
        byte-for-byte check."""
        pinned = read_jsonl(PINNED_PANEL_RUN / "replay_outputs.jsonl")
        v081 = read_jsonl(PANEL_RUN / "replay_outputs.jsonl")
        assert len(pinned) == len(v081) == 120
        pinned_sorted = sorted(
            pinned, key=lambda r: (r["family_id"], r["variant"]))
        v081_sorted = sorted(
            v081, key=lambda r: (r["family_id"], r["variant"]))
        divergences = 0
        for pinned_rec, v081_rec in zip(pinned_sorted, v081_sorted):
            assert pinned_rec["family_id"] == v081_rec["family_id"]
            assert pinned_rec["variant"] == v081_rec["variant"]
            if pinned_rec["response"] != v081_rec["response"]:
                divergences += 1
        assert divergences == 0, \
            f"{divergences} response(s) differ between pinned and v0.8.1"


class TestCapEscalationSummary:
    """Machine-readable cap escalation summary."""

    def test_summary_exists_and_valid(self):
        assert CAP_ESCALATION_SUMMARY.exists(), \
            "cap escalation summary missing"
        summary = json.loads(CAP_ESCALATION_SUMMARY.read_text(
            encoding="utf-8"))
        assert summary["selected_cap"] == 1536
        ladder = summary["escalation_ladder"]
        assert len(ladder) >= 4
        # Final step must show zero truncation
        final = ladder[-1]
        assert final["max_new_tokens"] == 1536
        assert final["n_truncated"] == 0
        assert final["truncation_rate"] == 0.0


class TestV083FinalCleanTreePanel:
    """120-response panel from a CLEAN git tree. The git_commit field
    now identifies the EXACT code tree that executed (not a base
    commit from an uncommitted working tree). git_dirty=False."""

    def test_final_panel_exists_and_complete(self):
        assert FINAL_PANEL_RUN.exists(), \
            "final clean-tree panel missing — run the t1536 chain script"
        outputs, failures, report = _load(FINAL_PANEL_RUN)
        assert report["expected_attempts"] == 120
        assert report["n_succeeded"] == 120 and report["n_failed"] == 0
        assert failures == [] and len(outputs) == 120

    def test_final_panel_clean_tree_provenance(self):
        """git_dirty=False and git_commit identifies the actual code."""
        _, _, report = _load(FINAL_PANEL_RUN)
        prov = report["provenance"]
        assert prov["revision_pinned"] is True
        assert prov["requested_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert prov["resolved_model_revision"] == \
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert prov["git_dirty"] is False, \
            "final panel must come from a clean tree"
        assert prov["git_commit"]
        # Expanded fingerprint includes prompt_template_revision +
        # torch/CUDA versions
        assert prov["prompt_template_revision"] == "v1"
        assert prov["torch_version"]
        assert prov["cuda_version"]
        assert prov["resolved_sha256"]

    def test_final_panel_zero_truncation(self):
        _, _, report = _load(FINAL_PANEL_RUN)
        assert report["truncation"]["n_truncated"] == 0

    def test_final_panel_outputs_match_v081(self):
        """Same model, same prompt, greedy == same text."""
        final = read_jsonl(FINAL_PANEL_RUN / "replay_outputs.jsonl")
        v081 = read_jsonl(PANEL_RUN / "replay_outputs.jsonl")
        assert len(final) == len(v081) == 120
        final_sorted = sorted(
            final, key=lambda r: (r["family_id"], r["variant"]))
        v081_sorted = sorted(
            v081, key=lambda r: (r["family_id"], r["variant"]))
        for f_rec, v_rec in zip(final_sorted, v081_sorted):
            assert f_rec["family_id"] == v_rec["family_id"]
            assert f_rec["variant"] == v_rec["variant"]
            assert f_rec["response"] == v_rec["response"], (
                f"response diverged for "
                f"{f_rec['family_id']}:{f_rec['variant']}")
