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
