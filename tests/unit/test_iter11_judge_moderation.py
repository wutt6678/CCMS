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

The confirmatory run then showed that "the request bytes" include the target's
own reply: judge A additionally refused CMST_456921/text_only in ONE arm and
served it in three, so the excluded set is the UNION of three cells -- two
uniform history triggers and one differential reply trigger. Both readings are
pinned against the committed artifacts below, not only against fakes.

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


def _load_probe():
    """Import the moderation probe, which resolves credentials at import.

    Resolution is env-first and never fatal at import, so the pure parts of the
    probe are testable in CI where only the ``.example`` credential template is
    committed. No request is made here: ``requests.post`` is monkeypatched.
    """
    os.environ.setdefault("LLM_JUDGE_API_KEY", "test-key-never-used")
    os.environ.setdefault("LLM_JUDGE_BASE_URL", "http://localhost/v1")
    spec = importlib.util.spec_from_file_location(
        "iter11_probe_judge_moderation_under_test",
        ROOT / "scripts" / "iter11_probe_judge_moderation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


def _verdict(status: int, code: str | None = None,
             body: int = 10_000) -> dict:
    """One ``_post`` result, in the shape the probe returns."""
    transport = status < 0
    return {"status": status, "error_code": code,
            "error_message": "Input text data may contain inappropriate "
                             "content." if code else None,
            "request_bytes": body, "transport_attempts": 1,
            "transport_error": "ConnectionError: reset by peer" if transport
            else None,
            "prompt_sha256": "0" * 64}


class TestAScanSeparatesAVerdictFromADroppedConnection:
    """A transport failure is not a provider refusal.

    The first full scan of the frozen panel reported FIVE refusals in
    ``qwen35_2b``'s arm: the two real ``data_inspection_failed`` cells the
    production run had also recorded, plus three cells whose request never got
    a response at all. Filed together, the three transport failures took the
    refusal rate from 2/600 to 5/600, gave the ``shuffle`` variant four
    refusals when one was real — which reads as a claim about which KIND of
    cell gets moderated — put a 164,991-byte body into a size comparison whose
    smallest genuinely refused body was 1,546,338 bytes, and spent seven
    bisection requests on each of three cells that had no trigger to find. All
    four bisections of all three came back 200, which is how they were
    identified.
    """

    def test_the_three_categories_are_disjoint_and_exhaustive(self):
        results = [_verdict(200), _verdict(400, "data_inspection_failed"),
                   _verdict(-1)]
        accepted, refused, unmeasured = probe.classify(results)
        assert [r["status"] for r in accepted] == [200]
        assert [r["status"] for r in refused] == [400]
        assert [r["status"] for r in unmeasured] == [-1]
        assert len(accepted) + len(refused) + len(unmeasured) == len(results)

    def test_a_dropped_connection_is_unmeasured_not_refused(self):
        _accepted, refused, unmeasured = probe.classify(
            [_verdict(-1), _verdict(-1)])
        assert refused == []
        assert len(unmeasured) == 2

    def test_a_gateway_error_is_a_verdict_about_the_request(self):
        # A 5xx came FROM the gateway, so it is a statement about the request
        # even though it is not moderation. Merging it with cells that were
        # never measured at all would hide the difference between "the provider
        # objected" and "we never heard back".
        _accepted, refused, unmeasured = probe.classify(
            [_verdict(500, "internal_error"), _verdict(-1)])
        assert [r["status"] for r in refused] == [500]
        assert [r["status"] for r in unmeasured] == [-1]

    def test_a_transport_failure_is_retried_and_can_recover(self, monkeypatch):
        calls = []

        def fake_post(url, headers=None, data=None, timeout=None, **kw):
            calls.append(data)
            if len(calls) < 3:
                raise ConnectionError("reset by peer")
            return _FakeResponse(200, {"ok": True})

        monkeypatch.setattr(probe.requests, "post", fake_post)
        out = probe._post("A", "prompt", [])
        assert out["status"] == 200
        assert out["transport_attempts"] == 3
        assert out["transport_error"] is None
        assert len(calls) == 3
        # The retry re-sends the SAME bytes, which is why it cannot bias the
        # measurement: moderation is a function of the request, not of the
        # number of times it was sent.
        assert len(set(calls)) == 1

    def test_a_provider_verdict_is_never_retried(self, monkeypatch):
        calls = []

        def fake_post(url, headers=None, data=None, timeout=None, **kw):
            calls.append(data)
            return _FakeResponse(400, MODERATION_BODY)

        monkeypatch.setattr(probe.requests, "post", fake_post)
        out = probe._post("A", "prompt", [])
        assert out["status"] == 400
        assert out["error_code"] == "data_inspection_failed"
        assert out["transport_attempts"] == 1
        assert len(calls) == 1

    def test_the_retry_budget_is_bounded_and_the_failure_is_kept(
            self, monkeypatch):
        calls = []

        def fake_post(url, headers=None, data=None, timeout=None, **kw):
            calls.append(data)
            raise TimeoutError("timed out")

        monkeypatch.setattr(probe.requests, "post", fake_post)
        out = probe._post("A", "prompt", [])
        assert out["status"] == probe.TRANSPORT_FAILURE
        assert out["transport_attempts"] == probe.TRANSPORT_RETRIES + 1
        assert out["transport_error"].startswith("TimeoutError")
        assert len(calls) == probe.TRANSPORT_RETRIES + 1


class TestScanAggregation:
    """``scan_target``'s counts, over fakes: no network, no panel read."""

    @pytest.fixture
    def arm(self, monkeypatch):
        """Four cells: accepted, refused, unmeasured, accepted.

        The refused cell carries the real 1,546,338-byte body and the
        unmeasured one the real 164,991-byte body from the 2026-09-07 scan, so
        the size comparison is asserted on the numbers that exposed the defect.
        """
        items = _items(4)
        scan = [_verdict(200, body=9_712),
                _verdict(400, "data_inspection_failed", body=1_546_338),
                _verdict(-1, body=164_991),
                _verdict(200, body=2_961_488)]
        calls = []

        def fake_post(label, prompt, images, max_tokens=1, **kw):
            calls.append((label, prompt))
            # After the four scan requests every bisection is served, so the
            # bisection tally is a function of which cells were bisected and
            # not of what the gateway said.
            return scan[len(calls) - 1] if len(calls) <= len(scan) \
                else _verdict(200)

        monkeypatch.setattr(probe, "_blinded_items", lambda key: items)
        monkeypatch.setattr(probe, "_render",
                            lambda label, item, **kw: ("prompt", []))
        monkeypatch.setattr(probe, "_post", fake_post)
        monkeypatch.setattr(
            probe, "_panel_binding",
            lambda key, n: {"model_key": key, "path": "blinded_items.json",
                            "sha256": "a" * 64, "n_items_scanned": n})
        return items, calls

    def test_the_categories_are_reported_separately(self, arm):
        out = probe.scan_target("qwen35_2b", None, 1, 0)
        assert out["n_items"] == 4
        assert out["n_accepted"] == 2
        assert out["n_refused"] == 1
        assert out["n_unmeasured"] == 1
        assert out["refusal_rate"] == 0.25
        assert out["error_codes"] == {"data_inspection_failed": 1}
        assert out["transport_errors"] == {"ConnectionError": 1}
        assert [c["item_id"] for c in out["unmeasured_cells"]] == \
            ["item-0002"]

    def test_the_variant_tally_counts_verdicts_only(self, arm):
        out = probe.scan_target("qwen35_2b", None, 1, 0)
        # item-0001 (cross_modal) was refused; item-0002 (text_only) was never
        # measured. Under the old counting BOTH landed in "refused", which is
        # how a dropped connection becomes a claim about a variant.
        assert out["by_variant"]["cross_modal"] == {
            "n": 2, "refused": 1, "unmeasured": 0}
        assert out["by_variant"]["text_only"] == {
            "n": 2, "refused": 0, "unmeasured": 1}

    def test_the_size_comparison_is_over_real_refusals(self, arm):
        out = probe.scan_target("qwen35_2b", None, 1, 0)
        assert out["request_bytes"]["refused_min"] == 1_546_338
        assert out["request_bytes"]["refused_max"] == 1_546_338
        assert out["request_bytes"]["accepted_max"] == 2_961_488
        # Refuted on the honest numbers: an ACCEPTED body was larger than the
        # refused one, so size does not separate them.
        assert out["size_hypothesis_refuted"] is True

    def test_only_a_provider_verdict_is_bisected(self, arm):
        _items_, calls = arm
        out = probe.scan_target("qwen35_2b", None, 1, 40)
        assert out["n_bisected"] == 1
        assert [b["item_id"] for b in out["bisected"]] == ["item-0001"]
        # Four bisections plus the two other identities, for ONE cell: the
        # unmeasured cell cost nothing beyond its own retries.
        assert len(calls) == 4 + len(probe.BISECTIONS) + 2
        assert out["bisection_tally"] == {
            name: 0 for name in
            list(probe.BISECTIONS) + ["full_B", "full_ADJUDICATOR"]}

    def test_the_rule_is_written_into_the_artifact(self, arm):
        out = probe.scan_target("qwen35_2b", None, 1, 0)
        assert out["transport_retries"] == probe.TRANSPORT_RETRIES
        rule = out["classification_rule"]
        assert "unmeasured" in rule and "n_refused" in rule

    def test_the_scan_names_the_panel_it_scanned(self, arm):
        out = probe.scan_target("qwen35_2b", None, 1, 0)
        assert out["panel"] == {"model_key": "qwen35_2b",
                                "path": "blinded_items.json",
                                "sha256": "a" * 64, "n_items_scanned": 4}


class TestScanProvenance:
    """The artifact has to name the code that produced it.

    The counting rule is the artifact's whole meaning: the same five non-200
    responses read as "five cells were refused" under one version of this
    script and as "two were refused and three were never measured" under the
    next. An artifact that does not say which version wrote it cannot be
    interpreted, only quoted.
    """

    def test_it_names_the_producer_and_the_commit(self):
        prov = probe.provenance()
        assert prov["produced_by"] == \
            "scripts/iter11_probe_judge_moderation.py"
        assert prov["kind"] == "iteration_11_judge_moderation_scan_v1"
        commit = prov["code_commit"]
        assert commit is None or len(commit) == 40
        for key in ("git_dirty", "code_dirty_paths", "untracked_code_paths",
                    "excluded_own_outputs", "excluded_cache_paths"):
            assert key in prov

    def test_the_exclusion_is_exactly_this_stages_own_output_tree(self):
        # Narrow by construction: the only thing a scan may make invisible is
        # the artifact the scan itself is writing. If this ever widens, the
        # dirtiness it records stops meaning anything.
        own = probe.OUT_DIR.relative_to(probe.REPO_ROOT).as_posix() + "/"
        assert probe.OWN_OUTPUT_PREFIXES == (own,)

    def test_a_dirty_tree_is_recorded_rather_than_hidden(self, monkeypatch):
        monkeypatch.setattr(probe, "code_tree_status", lambda **kw: {
            "dirty": True, "dirty_paths": ["scripts/x.py"],
            "untracked_paths": [], "code_dirty_paths": ["scripts/x.py"],
            "excluded_own_outputs": list(kw["exclude_prefixes"]),
            "excluded_cache_paths": []})
        prov = probe.provenance()
        assert prov["git_dirty"] is True
        assert prov["code_dirty_paths"] == ["scripts/x.py"]
        assert prov["excluded_own_outputs"] == list(probe.OWN_OUTPUT_PREFIXES)


EVIDENCE_DIR = ROOT / "outputs" / "iteration_11" / "diagnostics" \
    / "judge_moderation"
SCAN_ARTIFACT = EVIDENCE_DIR / "judge_a_moderation_qwen35_2b.json"
UNION_PROBE_ARTIFACT = EVIDENCE_DIR \
    / "cell_probe_exclusion_union_reprobe.json"

#: The 2026-09-07 re-run, taken after all four arms were finalized.
SCANNED_ITEMS = 600
SCANNED_ACCEPTED = 598
SCANNED_REFUSED = 2
SCANNED_REFUSAL_RATE = 0.0033

#: ``(item_id, family, variant, request_bytes)`` -- the same two cells the
#: production run recorded in this arm's ``.refusals.json`` sidecar.
REFUSED_CELLS = [
    ("item-0164", "CMST_795308", "cross_modal", 1_546_338),
    ("item-0410", "CMST_795308", "shuffle", 1_547_881),
]

#: The largest request the provider accepted in the same scan. The size
#: hypothesis is refuted by this being larger than every refusal.
LARGEST_ACCEPTED_REQUEST = 2_961_488

#: ``refused_min`` as the pre-fix script computed it, from a body that was
#: never refused -- a dropped connection on CMST_232475/shuffle. Quoted so the
#: regression is visible as a number rather than as a prose claim.
PRE_FIX_REFUSED_MIN = 164_991

HISTORY_TRIGGERED_CELLS = ("CMST_795308/cross_modal", "CMST_795308/shuffle")
REPLY_TRIGGERED_CELL = "CMST_456921/text_only"
REPLY_TRIGGERED_ARM = "ministral3_3b"
ARMS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")


def _load_evidence(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scan_artifact() -> dict:
    return _load_evidence(SCAN_ARTIFACT)


@pytest.fixture(scope="module")
def scan(scan_artifact) -> dict:
    return scan_artifact["targets"]["qwen35_2b"]


@pytest.fixture(scope="module")
def union_probe() -> dict:
    return _load_evidence(UNION_PROBE_ARTIFACT)


class TestTheCommittedScanCountedVerdictsNotConnections:
    """Pin the artifact, not just the code that writes it.

    The first re-run reported five refusals where two were real, because a
    request that never arrived was filed under ``n_refused``. The unit tests
    above pin the classification rule against fakes; these pin the committed
    numbers, so a regeneration that quietly restores the old rule -- or that
    picks up three more dropped connections and calls them censoring -- fails
    here with the real counts in the message.
    """

    def test_it_was_produced_by_committed_code_from_a_clean_tree(
            self, scan_artifact):
        assert scan_artifact["produced_by"] == \
            "scripts/iter11_probe_judge_moderation.py"
        assert len(scan_artifact["code_commit"]) == 40
        assert scan_artifact["git_dirty"] is False
        assert scan_artifact["code_dirty_paths"] == []
        assert scan_artifact["untracked_code_paths"] == []

    def test_two_refusals_and_nothing_unmeasured(self, scan):
        assert scan["n_items"] == SCANNED_ITEMS
        assert scan["n_accepted"] == SCANNED_ACCEPTED
        assert scan["n_refused"] == SCANNED_REFUSED
        assert scan["n_unmeasured"] == 0
        assert scan["refusal_rate"] == SCANNED_REFUSAL_RATE
        assert scan["unmeasured_cells"] == []
        assert scan["transport_errors"] is None
        # The pre-fix sentinel is absent from the codes entirely: had one cell
        # come back with no response, it would be here rather than hidden.
        assert "http_-1" not in scan["error_codes"]
        assert scan["error_codes"] == {"data_inspection_failed": 2}

    def test_the_refused_cells_are_the_two_the_run_recorded(self, scan):
        got = [(c["item_id"], c["family_id"], c["variant"],
                c["request_bytes"]) for c in scan["refused_cells"]]
        assert got == REFUSED_CELLS
        # Every one is a provider verdict obtained on the first attempt, so no
        # refusal in this list is a retry that happened to fail.
        assert all(c["status"] == 400 for c in scan["refused_cells"])
        assert all(c["transport_attempts"] == 1
                   for c in scan["refused_cells"])
        assert all(c["transport_error"] is None
                   for c in scan["refused_cells"])

    def test_the_variant_tally_is_one_and_one(self, scan):
        by_variant = scan["by_variant"]
        assert sum(v["n"] for v in by_variant.values()) == SCANNED_ITEMS
        refused = {k: v["refused"] for k, v in by_variant.items()
                   if v["refused"]}
        assert refused == {"cross_modal": 1, "shuffle": 1}
        # The pre-fix tally read {"shuffle": 4, "cross_modal": 1}, which is the
        # shape of a variant-specific trigger. Every slot also carries its own
        # unmeasured count, so a dropped connection cannot hide in ``n``.
        assert all(v["unmeasured"] == 0 for v in by_variant.values())

    def test_the_size_comparison_is_over_real_refusals(self, scan):
        sizes = scan["request_bytes"]
        assert sizes["refused_min"] == REFUSED_CELLS[0][3]
        assert sizes["refused_min"] != PRE_FIX_REFUSED_MIN
        assert sizes["accepted_max"] == LARGEST_ACCEPTED_REQUEST
        assert sizes["accepted_max"] > sizes["refused_min"]
        assert scan["size_hypothesis_refuted"] is True

    def test_the_size_ladder_is_all_accepted(self, scan_artifact):
        ladder = scan_artifact["size_ladder"]
        assert len(ladder) == 5
        assert all(step["status"] == 200 for step in ladder)
        assert all(step["error_code"] is None for step in ladder)
        # One image is above the downscale threshold and is sent reduced, so
        # the ladder covers the base64 path as well as the raw one.
        downscaled = [s for s in ladder if s["downscaled"]]
        assert len(downscaled) == 1
        assert downscaled[0]["transmitted_bytes"] < downscaled[0]["raw_bytes"]

    def test_the_bisection_still_isolates_the_history(self, scan):
        assert scan["bisection_tally"] == {
            "neutral_response": 2, "no_history": 0, "context_only": 2,
            "terminal_only": 0, "full_B": 0, "full_ADJUDICATOR": 0}
        assert set(scan["bisection_unmeasured"]) == \
            set(scan["bisection_tally"])
        assert sum(scan["bisection_unmeasured"].values()) == 0
        assert scan["n_bisected"] == SCANNED_REFUSED

    def test_it_scanned_the_panel_this_arm_was_judged_on(self, scan):
        panel = scan["panel"]
        assert panel["n_items_scanned"] == SCANNED_ITEMS
        path = ROOT / panel["path"]
        if not path.exists():
            pytest.skip(f"{panel['path']} not committed")
        from causal_mllm.validation.relations import _file_sha256
        # Recomputed rather than trusted: the binding is only worth having if
        # it still holds against the file in the tree.
        assert _file_sha256(path) == panel["sha256"]

    def test_the_scan_found_what_the_production_run_recorded(self, scan):
        """Two runs that share nothing but the payloads agree on the cells.

        The sidecar is written by the judging run that was actually refused;
        the scan is a re-measurement of the same bytes hours later, from
        another process, with ``max_tokens=1`` and nothing else in common.
        That they name the same cells under the same provider code is the
        evidence that the exclusion set is a property of the payloads and not
        of one run's circumstances -- which is what makes the exclusion
        outcome-independent rather than merely asserted to be.
        """
        sidecar = ROOT / "outputs" / "iteration_11" / "judge" / "qwen35_2b" \
            / "llm_labels_judge_A.refusals.json"
        if not sidecar.exists():
            pytest.skip("refusal sidecar not committed")
        recorded = json.loads(
            sidecar.read_text(encoding="utf-8"))["refusals"]
        assert [(r["item_id"], r["family_id"], r["variant"])
                for r in recorded] == \
            [(c["item_id"], c["family_id"], c["variant"])
             for c in scan["refused_cells"]]
        assert {r["error_code"] for r in recorded} == set(scan["error_codes"])
        assert all(r["judge_id"] == "A" for r in recorded)
        assert all(r["status"] == 400 for r in recorded)


class TestTheUnionReprobeSeparatesTheTwoReadings:
    """One artifact holding both readings, distinguished by a stated test.

    A cell refused in every arm at the same time is a function of the shared
    history; a cell refused in one arm and served in three at the same time is
    a function of that arm's reply. Measuring all three excluded cells across
    all four arms in one artifact is what makes the distinction observable
    rather than inferred from two probes taken five hours apart.
    """

    def test_the_history_cells_are_refused_in_every_arm(self, union_probe):
        status = union_probe["full_payload_status"]
        for cell in HISTORY_TRIGGERED_CELLS:
            assert status[cell] == {arm: 400 for arm in ARMS}

    def test_the_reply_cell_is_refused_in_one_arm_only(self, union_probe):
        status = union_probe["full_payload_status"][REPLY_TRIGGERED_CELL]
        assert status[REPLY_TRIGGERED_ARM] == 400
        assert [arm for arm in ARMS if status[arm] == 200] == \
            [a for a in ARMS if a != REPLY_TRIGGERED_ARM]
        assert union_probe["arms_that_disagree_now"] == [REPLY_TRIGGERED_CELL]

    def test_the_differential_trigger_is_the_reply(self, union_probe):
        cell = union_probe["per_cell"][REPLY_TRIGGERED_CELL][REPLY_TRIGGERED_ARM]
        # The history alone is accepted and the reply alone is refused -- the
        # exact inverse of the shared-history signature.
        assert cell["full"]["status"] == 400
        assert cell["no_history"]["status"] == 400
        assert cell["neutral_response"]["status"] == 200
        assert cell["context_only"]["status"] == 200
        assert cell["terminal_only"]["status"] == 200

    def test_the_uniform_trigger_is_the_history_in_every_arm(self,
                                                            union_probe):
        for cell in HISTORY_TRIGGERED_CELLS:
            for arm in ARMS:
                probes = union_probe["per_cell"][cell][arm]
                assert probes["full"]["status"] == 400, (cell, arm)
                # Reply replaced and reply removed: still refused.
                assert probes["neutral_response"]["status"] == 400, (cell, arm)
                assert probes["context_only"]["status"] == 400, (cell, arm)
                # History dropped: served.
                assert probes["no_history"]["status"] == 200, (cell, arm)
                assert probes["terminal_only"]["status"] == 200, (cell, arm)
                # The other two frozen identities accept the full payload.
                assert probes["full_B"]["status"] == 200, (cell, arm)
                assert probes["full_ADJUDICATOR"]["status"] == 200, (cell, arm)

    def test_no_arm_was_left_without_a_verdict(self, union_probe):
        assert union_probe["arms_unmeasured_now"] == []
        assert sorted(union_probe["cells"]) == sorted(
            list(HISTORY_TRIGGERED_CELLS) + [REPLY_TRIGGERED_CELL])
        assert tuple(union_probe["targets"]) == ARMS

    def test_it_excluded_the_scan_written_beside_it(self, union_probe):
        # The narrow exclusion working in the wild: the scan artifact was
        # sitting uncommitted in this directory when the probe ran, and it is
        # the ONLY path the probe reports as excluded.
        assert union_probe["excluded_own_outputs"] == \
            [str(SCAN_ARTIFACT.relative_to(ROOT))]
        assert union_probe["git_dirty"] is False
        assert union_probe["code_dirty_paths"] == []
        assert union_probe["untracked_code_paths"] == []

    def test_both_artifacts_bind_the_same_panel(self, union_probe, scan):
        assert union_probe["panels"]["qwen35_2b"]["sha256"] == \
            scan["panel"]["sha256"]
        assert all(union_probe["panels"][arm]["sha256"] for arm in ARMS)


class TestAProbeArtifactIsWrittenWhereItCanBeFound:
    """A cell is named ``FAMILY/VARIANT`` everywhere in this project.

    Interpolating that into a default output path puts the artifact in a
    nested directory named after the family, where nobody looking in the
    evidence directory will find it.
    """

    def test_the_separator_does_not_become_a_directory(self):
        stem = probe._probe_out_stem(
            ["CMST_456921/text_only", "CMST_795308/cross_modal"])
        assert "/" not in stem
        assert "\\" not in stem
        assert Path(stem).name == stem

    def test_every_probed_cell_is_still_named(self):
        cells = list(HISTORY_TRIGGERED_CELLS) + [REPLY_TRIGGERED_CELL]
        stem = probe._probe_out_stem(cells)
        for cell in cells:
            assert cell.replace("/", "_") in stem

    def test_a_spec_without_a_separator_is_unchanged(self):
        assert probe._probe_out_stem(["CMST_795308"]) == "CMST_795308"
