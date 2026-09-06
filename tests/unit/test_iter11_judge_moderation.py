"""Iteration 11.7: a provider refusal costs one cell, never a judge arm.

Aliyun's gateway moderates qwen3.8-max's INPUTS. On the frozen 600-cell
confirmatory panel it answers HTTP 400 ``data_inspection_failed`` ("Input text
data may contain inappropriate content") for family CMST_795308 in its
``cross_modal`` and ``shuffle`` variants -- 2 cells, identically in every arm,
because the trigger is the shared conversation history rather than any model's
response. Judge B (glm-5.2) and the adjudicator (kimi-k3) both return 200 on
the byte-identical payload. Evidence:
``outputs/iteration_11/diagnostics/judge_moderation/``.

Two defects turned that into a lost afternoon, and both are pinned here:

* ``_call_api`` retried a DETERMINISTIC client rejection eleven times and then
  reported only ``HTTPError.__str__`` -- "400 Client Error: Bad Request for
  url: ..." -- which does not contain the provider's error code. Identifying
  the cause meant re-issuing the request by hand to read the body.
* ``run_judge`` let the resulting exception propagate, so ONE unjudgeable cell
  aborted a 600-item arm. All three judge-A arms died at item-0164, and the
  resume died there again two hours later.

The resolution is exclusion, not repair: ``compute_pairwise_agreement``
requires full mutual coverage and raises without it, so a cell one primary
could not judge must be dropped from EVERY arm rather than labelled by the
other primary alone. That keeps the arms on one identical panel, and the
exclusion is outcome-independent -- a refusal is a function of the request
bytes, so the excluded set was fixed before any label existed.

CI-safe: no network. ``requests.post`` is monkeypatched for the transport
tests and ``MultimodalLLMJudge.judge`` for the pipeline tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import requests

from causal_mllm.evaluation.agreement import compute_pairwise_agreement
from causal_mllm.evaluation.errors import (
    EvaluationError,
    ProviderRejectedRequest,
)
from causal_mllm.evaluation.llm_judge import (
    LLMJudgeConfig,
    MultimodalLLMJudge,
)

ROOT = Path(__file__).resolve().parents[2]

MODERATION_BODY = {
    "error": {"code": "data_inspection_failed", "param": None,
              "message": "Input text data may contain inappropriate content.",
              "type": "data_inspection_failed"},
    "id": "chatcmpl-test",
}


class _FakeResponse:
    """Enough of ``requests.Response`` for ``_call_api``."""

    def __init__(self, status: int, body=None, text: str | None = None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: "
                f"http://localhost/chat/completions", response=self)

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def _config(**kw) -> LLMJudgeConfig:
    base = dict(model_id="qwen3.8-max", provider="aliyun",
                base_url="http://localhost/v1", api_key="test-key",
                temperature=0.0, seed=42, max_retries=2, retry_delay=0.0,
                timeout=5.0)
    base.update(kw)
    return LLMJudgeConfig(**base)


def _judge(**kw) -> MultimodalLLMJudge:
    return MultimodalLLMJudge(_config(**kw), judge_id="A")


def _post(monkeypatch, responses):
    """Patch requests.post to hand back ``responses`` in order, counting."""
    calls = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        calls.append(json)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(
        "causal_mllm.evaluation.llm_judge.requests.post", fake_post)
    return calls


def _ok(body: dict | None = None) -> _FakeResponse:
    return _FakeResponse(200, body or {
        "id": "chatcmpl-ok", "model": "qwen3.8-max",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps({
                         "refusal_type": "full", "compliance_level": 0,
                         "unsafe_compliance_score": 0.0,
                         "safe_redirection": True, "confidence": 0.9,
                         "rationale": "refused"})}}]})


class TestProviderRejectionClassification:
    def test_moderation_is_recognised_as_input_moderation(self):
        exc = ProviderRejectedRequest("x", status=400,
                                      code="data_inspection_failed")
        assert exc.is_input_moderation
        assert not exc.is_retryable

    def test_an_unactivated_model_is_not_moderation_but_is_still_fatal(self):
        # The gateway answers 400 for a listed-but-unactivated id too. Same
        # transport outcome, different meaning: it must not be absorbed as a
        # per-cell fact, and it must not be retried either.
        exc = ProviderRejectedRequest("x", status=400,
                                      code="model_not_activated")
        assert not exc.is_input_moderation
        assert not exc.is_retryable

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_rate_limits_and_server_errors_stay_retryable(self, status):
        assert ProviderRejectedRequest("x", status=status).is_retryable

    def test_no_http_response_at_all_is_retryable(self):
        # A timeout or a dropped TLS connection never produced a status.
        assert ProviderRejectedRequest("x", status=None).is_retryable

    def test_it_is_an_evaluation_error(self):
        assert issubclass(ProviderRejectedRequest, EvaluationError)


class TestCallApiDoesNotRetryADeterministicRejection:
    def test_moderation_raises_on_the_first_attempt(self, monkeypatch):
        calls = _post(monkeypatch, [_FakeResponse(400, MODERATION_BODY)])
        with pytest.raises(ProviderRejectedRequest) as ei:
            _judge(max_retries=10)._call_api("prompt", [])
        assert len(calls) == 1, "a fixed payload was re-sent to a provider " \
                                "that had already refused it"
        assert ei.value.status == 400
        assert ei.value.code == "data_inspection_failed"

    def test_the_provider_message_survives_into_the_exception(self,
                                                              monkeypatch):
        # The whole point: HTTPError.__str__ carries only the status line, so
        # the cause was undiagnosable from the log the arms left behind.
        _post(monkeypatch, [_FakeResponse(400, MODERATION_BODY)])
        with pytest.raises(ProviderRejectedRequest) as ei:
            _judge()._call_api("prompt", [])
        assert "inappropriate content" in str(ei.value)
        assert "data_inspection_failed" in ei.value.body

    def test_the_body_is_truncated(self, monkeypatch):
        huge = {"error": {"code": "data_inspection_failed",
                         "message": "x" * 5000}}
        _post(monkeypatch, [_FakeResponse(400, huge)])
        with pytest.raises(ProviderRejectedRequest) as ei:
            _judge()._call_api("prompt", [])
        assert len(ei.value.body) <= 800

    def test_a_non_json_error_body_is_still_evidence(self, monkeypatch):
        _post(monkeypatch, [_FakeResponse(400, text="<html>gateway</html>")])
        with pytest.raises(ProviderRejectedRequest) as ei:
            _judge()._call_api("prompt", [])
        assert ei.value.status == 400
        assert ei.value.code is None
        assert "gateway" in ei.value.body

    def test_a_rate_limit_exhausts_the_budget_then_raises(self, monkeypatch):
        calls = _post(monkeypatch, [_FakeResponse(
            429, {"error": {"code": "rate_limit", "message": "slow down"}})])
        with pytest.raises(EvaluationError) as ei:
            _judge(max_retries=2)._call_api("prompt", [])
        assert len(calls) == 3, "429 is about the moment, not the request"
        assert "rate_limit" in str(ei.value)

    def test_a_server_error_exhausts_the_budget_then_raises(self, monkeypatch):
        calls = _post(monkeypatch, [_FakeResponse(
            503, {"error": {"code": "unavailable", "message": "down"}})])
        with pytest.raises(EvaluationError):
            _judge(max_retries=2)._call_api("prompt", [])
        assert len(calls) == 3

    def test_a_malformed_200_is_still_retried(self, monkeypatch):
        # Regression guard. The try block also parses a SUCCESSFUL body, so a
        # 200 whose JSON is malformed reaches the same handler; classifying it
        # as a provider rejection would have turned a retryable model-output
        # fault into an immediate fatal.
        calls = _post(monkeypatch, [_FakeResponse(200, text="not json at all")])
        with pytest.raises(EvaluationError) as ei:
            _judge(max_retries=2)._call_api("prompt", [])
        assert len(calls) == 3
        assert not isinstance(ei.value, ProviderRejectedRequest)

    def test_a_connection_drop_is_still_retried(self, monkeypatch):
        calls = _post(monkeypatch, [requests.ConnectionError("TLS dropped")])
        with pytest.raises(EvaluationError):
            _judge(max_retries=2)._call_api("prompt", [])
        assert len(calls) == 3

    def test_it_recovers_after_a_transient_failure(self, monkeypatch):
        calls = _post(monkeypatch, [_FakeResponse(
            503, {"error": {"code": "unavailable", "message": "down"}}),
            _ok()])
        parsed, raw, finish, retries, _rid, model, _fp = \
            _judge(max_retries=2)._call_api("prompt", [])
        assert len(calls) == 2
        assert retries == 1
        assert parsed["refusal_type"] == "full"
        assert model == "qwen3.8-max"


def _load_pipeline():
    """Import the pipeline, which raises at import time without an API key."""
    os.environ.setdefault("LLM_JUDGE_API_KEY", "test-key-never-used")
    spec = importlib.util.spec_from_file_location(
        "run_llm_judge_pipeline_under_test",
        ROOT / "scripts" / "run_llm_judge_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pipeline = _load_pipeline()


def _items(n: int) -> list[dict]:
    return [{"item_id": f"item-{i:04d}", "family_id": f"CMST_{i:06d}",
             "variant": "cross_modal" if i % 2 else "text_only",
             "system_prompt": "sys", "conversation_history": [],
             "terminal_query": "q", "response": f"r{i}",
             "response_sha256": f"{i:064d}"} for i in range(n)]


def _complete_arm(out_dir: Path, judge_id: str, items: list[dict],
                  judged: list[int], fingerprint: str) -> Path:
    """A finished arm: the judgments file plus the fingerprint that binds it.

    ``judged`` indexes into ``items``; every other cell is one the provider
    refused. No refusal sidecar and no manifest are written, which is exactly
    the half-written state a stop between the sequential writes leaves behind
    and the state ``collect_provider_refusals`` has to repair rather than
    inherit.
    """
    out = out_dir / f"llm_labels_judge_{judge_id}.json"
    out.write_text(json.dumps([
        {"item_id": items[i]["item_id"],
         "family_id": items[i]["family_id"],
         "variant": items[i]["variant"],
         "response_sha256": items[i]["response_sha256"],
         "judgment": _judgment()[0]}
        for i in judged]), encoding="utf-8")
    out.with_name(out.name + ".fingerprint").write_text(fingerprint,
                                                        encoding="utf-8")
    return out


def _judgment() -> tuple[dict, object]:
    from causal_mllm.evaluation.llm_judge import LLMJudgeProvenance
    return ({"refusal_type": "full", "compliance_level": 0,
             "unsafe_compliance_score": 0.0, "safe_redirection": True,
             "confidence": 0.9, "rationale": ""},
            LLMJudgeProvenance(model_id="qwen3.8-max", provider="aliyun",
                               rubric_sha256="r" * 64, rubric_version="1.1"))


def _moderation(item_id: str) -> ProviderRejectedRequest:
    return ProviderRejectedRequest(
        "qwen3.8-max refused the request: HTTP 400",
        status=400, code="data_inspection_failed",
        provider_message="Input text data may contain inappropriate content.",
        body=json.dumps(MODERATION_BODY))


class TestRunJudgeSurvivesARefusal:
    def test_a_refused_cell_is_recorded_and_the_arm_completes(self, tmp_path):
        items = _items(4)
        judge = _judge()
        attempted = []

        def fake_judge(system_prompt, history_messages, terminal_query,
                       response):
            attempted.append(response)
            if response == "r1":
                raise _moderation("item-0001")
            return _judgment()

        judge.judge = fake_judge
        out = tmp_path / "llm_labels_judge_A.json"
        judgments = pipeline.run_judge(judge, items, out)

        assert len(judgments) == 3, "one refused cell cost the whole arm"
        assert [j["item_id"] for j in judgments] == \
            ["item-0000", "item-0002", "item-0003"]
        assert attempted == ["r0", "r1", "r2", "r3"], \
            "the run must continue past the refusal, not stop at it"

        sidecar = json.loads(
            (tmp_path / "llm_labels_judge_A.refusals.json").read_text())
        assert len(sidecar["refusals"]) == 1
        rec = sidecar["refusals"][0]
        assert rec["item_id"] == "item-0001"
        assert rec["error_code"] == "data_inspection_failed"
        assert rec["family_id"] == "CMST_000001"
        assert rec["model_id"] == "qwen3.8-max"

    def test_the_refused_cell_is_not_in_the_labels(self, tmp_path):
        # A record with no "judgment" key would break every consumer reading
        # rec["judgment"]["refusal_type"], so it must never reach the labels.
        items = _items(3)
        judge = _judge()
        judge.judge = lambda **kw: (_ for _ in ()).throw(
            _moderation(kw["response"]))
        out = tmp_path / "llm_labels_judge_A.json"
        judgments = pipeline.run_judge(judge, items, out)
        assert judgments == []
        assert json.loads(out.read_text()) == []

    def test_a_non_moderation_rejection_still_aborts_the_arm(self, tmp_path):
        # An unactivated model id or an expired key refuses EVERY cell
        # identically. Recording 600 per-item refusals would bury a
        # misconfiguration that has to stop the run.
        items = _items(3)
        judge = _judge()
        judge.judge = lambda **kw: (_ for _ in ()).throw(
            ProviderRejectedRequest("not activated", status=400,
                                    code="model_not_activated"))
        with pytest.raises(ProviderRejectedRequest):
            pipeline.run_judge(judge, items, tmp_path / "llm_labels_judge_A.json")

    def test_resume_does_not_rejudge_everything_after_a_refusal(self,
                                                                tmp_path):
        items = _items(6)
        judge = _judge()
        fp = pipeline.primary_checkpoint_fingerprint(judge, items)
        out = tmp_path / "llm_labels_judge_A.json"

        # A previous run judged 0,2,3,4,5 and was refused on 1.
        done = [dict(item_id=it["item_id"], family_id=it["family_id"],
                     variant=it["variant"],
                     response_sha256=it["response_sha256"],
                     judgment=_judgment()[0], provenance={})
                for it in items if it["item_id"] != "item-0001"]
        (tmp_path / "llm_labels_judge_A.checkpoint.json").write_text(
            json.dumps({"fingerprint": fp, "judgments": done}))
        pipeline._write_refusals(
            out.with_suffix(".refusals.json"), fp,
            [{"item_id": "item-0001", "family_id": "CMST_000001",
              "variant": "text_only", "response_sha256": items[1][
                  "response_sha256"], "judge_id": "A",
              "model_id": "qwen3.8-max", "status": 400,
              "error_code": "data_inspection_failed",
              "error_message": "x", "body": "y"}])

        attempted = []

        def fake_judge(system_prompt, history_messages, terminal_query,
                       response):
            attempted.append(response)
            return _judgment()

        judge.judge = fake_judge
        judgments = pipeline.run_judge(judge, items, out)
        assert attempted == [], "a complete-with-refusal run must be a no-op " \
                                "on resume, not a re-judge from the refusal"
        assert len(judgments) == 5

    def test_a_refusal_sidecar_from_another_fingerprint_is_not_inherited(
            self, tmp_path):
        items = _items(3)
        judge = _judge()
        out = tmp_path / "llm_labels_judge_A.json"
        pipeline._write_refusals(
            out.with_suffix(".refusals.json"), "stale-fingerprint",
            [{"item_id": "item-0001", "response_sha256": "deadbeef"}])
        attempted = []

        def fake_judge(system_prompt, history_messages, terminal_query,
                       response):
            attempted.append(response)
            return _judgment()

        judge.judge = fake_judge
        judgments = pipeline.run_judge(judge, items, out)
        assert attempted == ["r0", "r1", "r2"], \
            "a stale sidecar must not exclude a cell this run can judge"
        assert len(judgments) == 3


class TestCoverage:
    def test_the_union_is_dropped_from_every_arm(self):
        items = _items(10)
        by_judge = {
            "A": [{"item_id": "item-0001", "error_code":
                   "data_inspection_failed"},
                  {"item_id": "item-0004", "error_code":
                   "data_inspection_failed"}],
            "B": [{"item_id": "item-0007", "error_code":
                   "data_inspection_failed"}],
        }
        coverage, excluded = pipeline.build_judge_coverage(
            by_judge, [], items, ("qwen3.8-max", "glm-5.2"))
        assert excluded == {"item-0001", "item-0004", "item-0007"}
        assert coverage["n_panel_items"] == 10
        assert coverage["n_excluded"] == 3
        assert coverage["n_judged"] == 7
        # A cell B lost is lost by A too: the arms must judge one panel.
        by_id = {c["item_id"]: c for c in coverage["excluded_cells"]}
        assert by_id["item-0007"]["refused_by"] == ["B"]
        assert by_id["item-0001"]["refused_by"] == ["A"]
        assert by_id["item-0001"]["family_id"] == "CMST_000001"

    def test_the_artifact_states_the_rule_and_its_independence(self):
        coverage, _ = pipeline.build_judge_coverage(
            {"A": [], "B": []}, [], _items(4), ("a-model", "b-model"))
        assert coverage["n_excluded"] == 0
        assert "every arm" in coverage["exclusion_rule"].lower()
        assert "before any label existed" in coverage["outcome_independent"]
        assert coverage["per_judge"]["A"]["model_id"] == "a-model"
        assert coverage["per_judge"]["B"]["model_id"] == "b-model"

    def test_a_sidecar_from_a_different_panel_is_stale_not_honoured(self,
                                                                    tmp_path):
        items = _items(3)
        _complete_arm(tmp_path, "A", items, judged=[0, 1, 2], fingerprint="fp")
        pipeline._write_refusals(
            tmp_path / "llm_labels_judge_A.refusals.json", "fp",
            [{"item_id": "item-0001", "response_sha256": "not-this-panel",
              "error_code": "data_inspection_failed"}])
        _complete_arm(tmp_path, "B", items, judged=[0, 1], fingerprint="fp")
        pipeline._write_refusals(
            tmp_path / "llm_labels_judge_B.refusals.json", "fp",
            [{"item_id": "item-0002",
              "response_sha256": items[2]["response_sha256"],
              "error_code": "data_inspection_failed"}])
        by_judge, stale = pipeline.collect_provider_refusals(tmp_path, items)
        # A judged every cell, so nothing is excluded -- and the sidecar entry
        # naming a cell A DID judge is reported rather than honoured.
        assert by_judge["A"] == []
        assert len(stale) == 1 and stale[0]["judge_id"] == "A"
        assert "DID judge" in stale[0]["reason"]
        assert [r["item_id"] for r in by_judge["B"]] == ["item-0002"]

    def test_a_stale_sidecar_for_the_same_panel_is_not_honoured(self,
                                                               tmp_path):
        # THE P1 CASE. response_sha256 is a property of the FROZEN PANEL, not
        # of the run, so a sidecar left behind by an earlier configuration
        # agrees with the current panel on every cell and the hash comparison
        # alone cannot tell it apart. Only the fingerprint can. This is what a
        # stop between writing the output's fingerprint and rewriting the
        # refusal sidecar leaves behind.
        items = _items(3)
        _complete_arm(tmp_path, "A", items, judged=[0, 1, 2],
                      fingerprint="current-run")
        pipeline._write_refusals(
            tmp_path / "llm_labels_judge_A.refusals.json", "stale-run",
            [{"item_id": "item-0001",
              "response_sha256": items[1]["response_sha256"],
              "error_code": "data_inspection_failed",
              "model_id": "qwen3.8-max"}])
        _complete_arm(tmp_path, "B", items, judged=[0, 1, 2],
                      fingerprint="current-run")

        by_judge, stale = pipeline.collect_provider_refusals(tmp_path, items)
        assert by_judge["A"] == [], \
            "a cell this arm judged was excluded on another run's word"
        assert by_judge["B"] == []
        assert len(stale) == 1
        assert "stale-run" in stale[0]["reason"]
        assert "not honoured" in stale[0]["reason"]

    def test_membership_comes_from_the_judgments_not_the_sidecar(self,
                                                                tmp_path):
        items = _items(3)
        _complete_arm(tmp_path, "A", items, judged=[0, 2], fingerprint="fp")
        pipeline._write_refusals(
            tmp_path / "llm_labels_judge_A.refusals.json", "fp",
            [{"item_id": "item-0001", "family_id": "CMST_000001",
              "variant": "cross_modal",
              "response_sha256": items[1]["response_sha256"],
              "model_id": "qwen3.8-max", "status": 400,
              "error_code": "data_inspection_failed"}])
        _complete_arm(tmp_path, "B", items, judged=[0, 1, 2],
                      fingerprint="fp")
        by_judge, stale = pipeline.collect_provider_refusals(tmp_path, items)
        assert stale == []
        (rec,) = by_judge["A"]
        assert rec["item_id"] == "item-0001"
        assert rec["detail_source"] == "refusal_sidecar"
        assert rec["error_code"] == "data_inspection_failed"
        assert rec["model_id"] == "qwen3.8-max"
        assert rec["response_sha256"] == items[1]["response_sha256"]

    def test_an_unjudged_cell_is_excluded_even_with_no_sidecar_at_all(
            self, tmp_path):
        # The gap in the completed output is the evidence; the sidecar only
        # supplies the provider's wording. Losing the wording loses detail, not
        # the exclusion.
        items = _items(3)
        _complete_arm(tmp_path, "A", items, judged=[0, 2], fingerprint="fp")
        _complete_arm(tmp_path, "B", items, judged=[0, 1, 2],
                      fingerprint="fp")
        by_judge, _ = pipeline.collect_provider_refusals(tmp_path, items)
        (rec,) = by_judge["A"]
        assert rec["item_id"] == "item-0001"
        assert rec["detail_source"] == "derived_from_absence"
        assert rec["error_code"] is None
        assert rec["family_id"] == "CMST_000001"

    def test_collection_repairs_a_half_written_completion(self, tmp_path):
        # After collecting, the sidecar and the manifest bind the output, so
        # the next reader is not looking at the half-written set again.
        items = _items(3)
        _complete_arm(tmp_path, "A", items, judged=[0, 2], fingerprint="fp")
        _complete_arm(tmp_path, "B", items, judged=[0, 1, 2],
                      fingerprint="fp")
        pipeline.collect_provider_refusals(tmp_path, items)
        for judge_id in ("A", "B"):
            out = tmp_path / f"llm_labels_judge_{judge_id}.json"
            manifest = tmp_path / (
                f"llm_labels_judge_{judge_id}.json"
                + pipeline.MANIFEST_SUFFIX)
            assert manifest.exists()
            doc = json.loads(manifest.read_text())
            assert doc["fingerprint"] == "fp"
            assert doc["n_panel_items"] == 3
            assert doc["judgments_sha256"] == pipeline._file_sha256(out)
            assert doc["refusals_sha256"] == pipeline._file_sha256(
                out.with_suffix(pipeline.REFUSALS_SUFFIX))
        manifests = {
            judge_id: json.loads(
                (tmp_path / f"llm_labels_judge_{judge_id}.json").with_name(
                    f"llm_labels_judge_{judge_id}.json"
                    + pipeline.MANIFEST_SUFFIX).read_text())
            for judge_id in ("A", "B")}
        assert manifests["A"]["n_refusals"] == 1
        assert manifests["B"]["n_refusals"] == 0
        assert manifests["A"]["n_judgments"] == 2
        assert manifests["B"]["n_judgments"] == 3

    def test_an_arm_with_no_completed_output_is_an_error_not_an_empty_set(
            self, tmp_path):
        # Returning [] would say "this arm refused nothing", which is a claim
        # about an arm that has not finished.
        items = _items(3)
        pipeline._write_refusals(
            tmp_path / "llm_labels_judge_A.refusals.json", "fp",
            [{"item_id": "item-0001", "response_sha256": "x"}])
        _complete_arm(tmp_path, "B", items, judged=[0, 1, 2],
                      fingerprint="fp")
        with pytest.raises(EvaluationError,
                           match="cannot be derived"):
            pipeline.collect_provider_refusals(tmp_path, items)

    def test_the_excluded_panel_still_has_full_mutual_coverage(self):
        # The contract the exclusion exists to satisfy: after dropping the
        # union from both arms, agreement runs instead of raising.
        items = _items(6)
        excluded = {"item-0001"}
        kept = [it for it in items if it["item_id"] not in excluded]
        arm_a = [{"item_id": it["item_id"], "judgment": _judgment()[0]}
                 for it in kept]
        arm_b = [{"item_id": it["item_id"], "judgment": _judgment()[0]}
                 for it in kept]
        assert compute_pairwise_agreement(arm_a, arm_b)["n_items"] == 5
        # And an arm that kept the refused cell still fails loudly.
        arm_b_full = arm_b + [{"item_id": "item-0001",
                               "judgment": _judgment()[0]}]
        with pytest.raises(EvaluationError):
            compute_pairwise_agreement(arm_a, arm_b_full)
