"""Unit tests for Iteration 8: frozen replay runner (stub backend).

Real-model runs live outside CI; these tests pin the STAGE CONTRACT
with a deterministic CallableBackend: validated-only input gate,
exact history replay, identical settings across variants, full
(family, variant) coverage, failure isolation, media verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.replay import (
    CallableBackend,
    ReplayConfig,
    ReplayError,
    resolved_fingerprint,
    run_replay_stage,
    verify_family_media,
)
from causal_mllm.replay.backend import HFLocalBackend
from tests.unit.test_grounding import CLEAN_Q, _built_family


def _write_validated(tmp_path, *families):
    write_jsonl(tmp_path / "validated_families.jsonl",
                [f.to_dict() for f in families])


def _echo_backend():
    def fn(chat_messages):
        last = chat_messages[-1]["content"]
        text = next(p["text"] for p in last
                    if isinstance(p, dict) and p["type"] == "text")
        n_images = sum(
            1 for m in chat_messages
            if isinstance(m["content"], list)
            for p in m["content"] if p.get("type") == "image")
        return {"response": f"answer to: {text}",
                "input_token_count": 100 + n_images,
                "image_token_count": 10 * n_images}
    return CallableBackend(fn, model_name="stub-model",
                           model_revision="stub-rev")


class TestInputGate:
    def test_rejects_missing_validated_families(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        # Only the RAW artifact exists: replay must refuse outright.
        write_jsonl(tmp_path / "families.jsonl", [family.to_dict()])
        with pytest.raises(ReplayError, match="validated_families"):
            run_replay_stage(tmp_path, tmp_path / "runs",
                             backend=_echo_backend())

    def test_no_media_verification_bypass(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        assert verify_family_media(family) == []


class TestReplayStage:
    def _run(self, tmp_path, backend=None, n_families=2, **kwargs):
        families = [_built_family(tmp_path, CLEAN_Q, fill_grounding=True)
                    for _ in range(n_families)]
        # Unique family ids for coverage assertions
        for i, family in enumerate(families):
            family.family_id = f"fam{i:03d}"
        _write_validated(tmp_path, *families)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs",
            backend=backend or _echo_backend(), run_id="test-run",
            **kwargs)
        run_dir = tmp_path / "runs" / "test-run"
        outputs = read_jsonl(run_dir / "replay_outputs.jsonl")
        failures = read_jsonl(run_dir / "replay_failures.jsonl")
        return report, outputs, failures

    def test_full_coverage_and_counts(self, tmp_path):
        report, outputs, failures = self._run(tmp_path, n_families=2)
        assert report["n_attempted"] == 12  # 2 families x 6 variants
        assert report["expected_attempts"] == 12
        assert report["n_succeeded"] == 12 and report["n_failed"] == 0
        assert report["missing_variants"] == []
        assert len(outputs) == 12 and failures == []
        variants_per_family = {
            (r["family_id"], r["variant"]) for r in outputs}
        assert len(variants_per_family) == 12

    def test_identical_provenance_across_variants(self, tmp_path):
        _, outputs, _ = self._run(tmp_path, n_families=1)
        keys = ("model", "model_revision", "prompt_template_revision",
                "system_prompt_sha256")
        for key in keys:
            assert len({r[key] for r in outputs}) == 1
        settings = {json.dumps(r["generation_config"], sort_keys=True)
                    for r in outputs}
        assert len(settings) == 1
        config = json.loads(settings.pop())
        assert config == {"temperature": 0.0, "top_p": 1.0,
                          "do_sample": False, "max_new_tokens": 256,
                          "seed": 42}
        for record in outputs:
            assert record["run_id"] == "test-run"
            assert record["source_id"] is not None
            assert record["error"] is None
            assert record["response"] is not None

    def test_exact_history_replay_no_rewriting(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        _write_validated(tmp_path, family)
        seen = []

        def fn(chat_messages):
            seen.append(chat_messages)
            return "ok"

        run_replay_stage(tmp_path, tmp_path / "runs",
                         backend=CallableBackend(fn), run_id="r")
        assert len(seen) == 6

        # Each replay must reproduce one stored variant VERBATIM:
        # system prompt first, stored turns unchanged, canonical q*
        # last and identical across all six replays.
        stored_variants = [
            [(m.role, m.text, list(m.images))
             for m in family.variants[name].messages]
            for name in family.variants
        ]
        terminals = set()
        for chat in seen:
            assert chat[0]["role"] == "system"
            replayed = []
            for message in chat[1:]:
                text = next(
                    (p["text"] for p in message["content"]
                     if p["type"] == "text"), None)
                images = [p["image"] for p in message["content"]
                          if p["type"] == "image"]
                replayed.append((message["role"], text, images))
            assert replayed in stored_variants
            terminals.add(replayed[-1][1])
        assert terminals == {CLEAN_Q}

    def test_failure_isolated_not_a_response(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        # Force ONLY cross_modal to raise: discriminate by its exact
        # stored message sequence (shuffle shares the same content).
        cross_seq = [(m.role, m.text, list(m.images))
                     for m in family.variants["cross_modal"].messages]

        def selective(chat_messages):
            replayed = []
            for message in chat_messages[1:]:
                text = next(
                    (p["text"] for p in message["content"]
                     if p["type"] == "text"), None)
                images = [p["image"] for p in message["content"]
                          if p["type"] == "image"]
                replayed.append((message["role"], text, images))
            if replayed == cross_seq:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            return {"response": "ok", "input_token_count": 5,
                    "image_token_count": 0}

        _write_validated(tmp_path, family)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs",
            backend=CallableBackend(selective), run_id="test-run")
        failures = read_jsonl(
            tmp_path / "runs" / "test-run" / "replay_failures.jsonl")
        outputs = read_jsonl(
            tmp_path / "runs" / "test-run" / "replay_outputs.jsonl")
        assert report["n_attempted"] == 6
        assert report["n_succeeded"] == 5 and report["n_failed"] == 1
        assert len(outputs) == 5 and len(failures) == 1
        failed = failures[0]
        assert failed["variant"] == "cross_modal"
        assert failed["response"] is None
        assert failed["error"]["category"] == "oom"
        assert failed["error"]["type"] == "RuntimeError"

    def test_media_missing_fails_every_variant_of_family(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        media_path = next(
            media["path"]
            for atom in family.semantic_atoms
            for media in atom.source_media)
        Path(media_path).unlink()  # corrupt the media store
        _write_validated(tmp_path, family)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", backend=_echo_backend(),
            run_id="media-run")
        failures = read_jsonl(
            tmp_path / "runs" / "media-run" / "replay_failures.jsonl")
        assert report["n_attempted"] == 6  # coverage preserved
        assert report["n_failed"] == 6
        assert len(failures) == 6
        for record in failures:
            assert record["error"]["category"] == "media"
            assert record["response"] is None

    def test_token_stats_recorded(self, tmp_path):
        report, outputs, _ = self._run(tmp_path, n_families=1)
        assert report["token_stats"]["total_input_tokens"] > 0
        assert report["token_stats"]["total_image_tokens"] > 0
        vision = [r for r in outputs
                  if r["variant"] in ("vision_only", "cross_modal",
                                      "shuffle")]
        text = [r for r in outputs
                if r["variant"] in ("neutral", "text_only",
                                    "history_reset")]
        assert all(r["image_token_count"] > 0 for r in vision)
        assert all(r["image_token_count"] == 0 for r in text)

    def test_provenance_block_complete(self, tmp_path):
        report, _, _ = self._run(tmp_path, n_families=1)
        prov = report["provenance"]
        assert prov["model"] == "stub-model"
        assert prov["model_revision"] == "stub-rev"
        assert prov["requested_model_revision"] is None
        assert prov["resolved_model_revision"] == "stub-rev"
        assert prov["revision_pinned"] is False
        assert prov["processor_revision"] == "stub-rev"
        assert prov["prompt_template_revision"] == "v1"
        assert prov["system_prompt_sha256"]
        assert prov["config_sha256"]
        assert prov["generation_config"]["temperature"] == 0.0
        assert prov["enable_thinking"] is False
        assert prov["torch_dtype"] == "bfloat16"
        assert prov["validated_families_sha256"]
        assert prov["resolved_sha256"]
        assert prov["git_commit"]  # running inside git repo


class TestOutputDiagnostics:
    """P0: every record must say HOW generation stopped, and the report
    must expose truncation BY VARIANT (a global rate can hide the
    refusal-short / compliance-long imbalance)."""

    def _cross_modal_seq(self, family):
        return [(m.role, m.text, list(m.images))
                for m in family.variants["cross_modal"].messages]

    def _replayed(self, chat_messages):
        return [
            (message["role"],
             next((p["text"] for p in message["content"]
                   if p["type"] == "text"), None),
             [p["image"] for p in message["content"]
              if p["type"] == "image"])
            for message in chat_messages[1:]
        ]

    def test_truncation_reported_by_variant(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        cross_seq = self._cross_modal_seq(family)

        def fn(chat_messages):
            truncated = self._replayed(chat_messages) == cross_seq
            return {
                "response": "x",
                "input_token_count": 10,
                "image_token_count": 0,
                "output_token_count": 256 if truncated else 83,
                "finish_reason": "length" if truncated else "eos",
                "hit_max_new_tokens": truncated,
            }

        _write_validated(tmp_path, family)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs",
            backend=CallableBackend(fn), run_id="trunc")
        by_variant = report["truncation"]["by_variant"]
        assert by_variant["cross_modal"] == {
            "n": 1, "n_truncated": 1, "truncation_rate": 1.0}
        for variant in ("neutral", "text_only", "vision_only",
                        "shuffle", "history_reset"):
            assert by_variant[variant]["n_truncated"] == 0
            assert by_variant[variant]["truncation_rate"] == 0.0
        assert report["truncation"]["n_truncated"] == 1
        assert report["truncation"]["truncation_rate"] == 1 / 6

        outputs = read_jsonl(
            tmp_path / "runs" / "trunc" / "replay_outputs.jsonl")
        truncated = next(r for r in outputs
                         if r["variant"] == "cross_modal")
        assert truncated["finish_reason"] == "length"
        assert truncated["hit_max_new_tokens"] is True
        assert truncated["output_token_count"] == 256
        completed = next(r for r in outputs if r["variant"] == "neutral")
        assert completed["finish_reason"] == "eos"
        assert completed["hit_max_new_tokens"] is False
        assert completed["output_token_count"] == 83

    def test_stub_backend_defaults_are_complete(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        _write_validated(tmp_path, family)
        run_replay_stage(tmp_path, tmp_path / "runs",
                         backend=CallableBackend(lambda c: "ok"),
                         run_id="d")
        outputs = read_jsonl(tmp_path / "runs" / "d" / "replay_outputs.jsonl")
        for record in outputs:
            assert record["output_token_count"] is not None
            assert record["finish_reason"] == "eos"
            assert record["hit_max_new_tokens"] is False


class TestReproducibilityPinning:
    def test_revision_kwarg_passed_to_load(self):
        pinned = ReplayConfig(model_revision="abc123")
        backend = HFLocalBackend(pinned)
        assert backend._pretrained_kwargs() == {"revision": "abc123"}
        unpinned = HFLocalBackend(ReplayConfig())
        assert unpinned._pretrained_kwargs() == {}

    def test_resolved_fingerprint_binds_revision(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        _write_validated(tmp_path, family)
        backend = CallableBackend(lambda c: "ok", model_revision="rev-A")
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", backend=backend, run_id="fp")
        prov = report["provenance"]
        assert prov["resolved_sha256"]
        assert prov["resolved_sha256"] == resolved_fingerprint(
            backend, ReplayConfig(), tmp_path)
        # A different resolved revision changes the fingerprint while
        # the requested-config fingerprint stays put.
        backend_b = CallableBackend(lambda c: "ok", model_revision="rev-B")
        assert resolved_fingerprint(backend_b, ReplayConfig(), tmp_path) \
            != prov["resolved_sha256"]
        assert backend_b.model_revision() != backend.model_revision()

    def test_output_token_stats_in_report(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"

        def fn(chat_messages):
            return {"response": "ok", "input_token_count": 10,
                    "image_token_count": 0, "output_token_count": 42,
                    "finish_reason": "eos", "hit_max_new_tokens": False}

        _write_validated(tmp_path, family)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs",
            backend=CallableBackend(fn), run_id="tok")
        assert report["token_stats"]["total_output_tokens"] == 42 * 6
        assert report["token_stats"]["mean_output_tokens"] == 42

    def test_revision_pinned_records_requested_and_resolved(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        _write_validated(tmp_path, family)
        config = ReplayConfig(model_revision="abc123")
        backend = CallableBackend(
            lambda c: "ok", model_revision="abc123",
            model_name="stub-model")
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config,
            backend=backend, run_id="pinned")
        prov = report["provenance"]
        assert prov["requested_model_revision"] == "abc123"
        assert prov["resolved_model_revision"] == "abc123"
        assert prov["revision_pinned"] is True
        outputs = read_jsonl(
            tmp_path / "runs" / "pinned" / "replay_outputs.jsonl")
        for record in outputs:
            assert record["requested_model_revision"] == "abc123"
            assert record["resolved_model_revision"] == "abc123"
            assert record["revision_pinned"] is True
            assert record["model_revision"] == "abc123"

    def test_revision_mismatch_fails_run(self, tmp_path):
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        _write_validated(tmp_path, family)
        config = ReplayConfig(model_revision="requested-sha")
        # Backend resolves to a DIFFERENT revision
        backend = CallableBackend(
            lambda c: "ok", model_revision="different-sha",
            model_name="stub-model")
        with pytest.raises(ReplayError, match="revision mismatch"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", config=config,
                backend=backend, run_id="mismatch")
