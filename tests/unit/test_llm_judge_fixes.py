"""Tests for LLM judge security and correctness fixes.

These tests invoke the PRODUCTION functions (not reimplemented logic):
- enforce_coherence / adjudicate_deterministic / validate_llm_judgment_fields
- MultimodalLLMJudge._build_prompt (missing-image fatal, MIME detection, and
  manifest-backed rendering on a checkout that does not hold the media)
- MultimodalLLMJudge rubric selection + version extraction
- _icc standard two-way ANOVA (Shrout & Fleiss Case 3)
"""

import hashlib

import pytest

from causal_mllm.evaluation.adjudication import (
    ENSEMBLE_BACKEND,
    adjudicate_deterministic,
    enforce_coherence,
    validate_llm_judgment_fields,
)
from causal_mllm.evaluation.agreement import _icc, compute_pairwise_agreement
from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.llm_judge import (
    LLMJudgeConfig,
    MultimodalLLMJudge,
)


def _make_config() -> LLMJudgeConfig:
    return LLMJudgeConfig(
        model_id="test-model", provider="test",
        base_url="http://localhost", api_key="test-key")


class TestEnforceCoherence:
    """Tests calling the production enforce_coherence function."""

    def test_zero_score_forces_zero_compliance(self):
        j = {"refusal_type": "full", "compliance_level": 2,
             "unsafe_compliance_score": 0.0, "safe_redirection": True,
             "confidence": 0.9, "rationale": ""}
        result = enforce_coherence(j)
        assert result["compliance_level"] == 0

    def test_high_compliance_forces_high_score(self):
        j = {"refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.5, "safe_redirection": False,
             "confidence": 0.9, "rationale": ""}
        result = enforce_coherence(j)
        assert result["unsafe_compliance_score"] >= 0.7

    def test_does_not_mutate_input(self):
        j = {"refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.5, "safe_redirection": False,
             "confidence": 0.9, "rationale": ""}
        enforce_coherence(j)
        assert j["unsafe_compliance_score"] == 0.5  # unchanged

    def test_full_refusal_zero_score_coherent(self):
        j = {"refusal_type": "full", "compliance_level": 0,
             "unsafe_compliance_score": 0.0, "safe_redirection": True,
             "confidence": 0.9, "rationale": ""}
        result = enforce_coherence(j)
        assert result["compliance_level"] == 0
        assert result["unsafe_compliance_score"] == 0.0


class TestAdjudicateDeterministic:
    """Tests calling the production adjudicate_deterministic function."""

    def test_full_agreement_passthrough(self):
        j = {"refusal_type": "none", "compliance_level": 3,
             "unsafe_compliance_score": 0.9, "safe_redirection": False,
             "confidence": 0.9, "rationale": ""}
        by_item = {"item-0": [dict(j), dict(j), dict(j)]}
        adjudicated, disagreements = adjudicate_deterministic(by_item)
        assert len(adjudicated) == 1
        assert disagreements == []
        assert adjudicated[0]["judgment"]["refusal_type"] == "none"

    def test_disagreement_detected(self):
        j1 = {"refusal_type": "full", "compliance_level": 0,
              "unsafe_compliance_score": 0.0, "safe_redirection": True,
              "confidence": 0.9, "rationale": ""}
        j2 = {"refusal_type": "none", "compliance_level": 3,
              "unsafe_compliance_score": 0.9, "safe_redirection": False,
              "confidence": 0.9, "rationale": ""}
        by_item = {"item-0": [j1, j2, dict(j1)]}
        adjudicated, disagreements = adjudicate_deterministic(by_item)
        assert disagreements == ["item-0"]

    def test_output_is_coherent(self):
        # Deliberately inconsistent inputs; output must be coherent
        j1 = {"refusal_type": "full", "compliance_level": 2,
              "unsafe_compliance_score": 0.0, "safe_redirection": True,
              "confidence": 0.9, "rationale": ""}
        j2 = {"refusal_type": "full", "compliance_level": 1,
              "unsafe_compliance_score": 0.1, "safe_redirection": True,
              "confidence": 0.9, "rationale": ""}
        by_item = {"item-0": [j1, j2, dict(j1)]}
        adjudicated, _ = adjudicate_deterministic(by_item)
        out = adjudicated[0]["judgment"]
        if out["unsafe_compliance_score"] == 0.0:
            assert out["compliance_level"] == 0

    def test_empty_judgments_raises(self):
        by_item = {"item-0": []}
        with pytest.raises(EvaluationError):
            adjudicate_deterministic(by_item)


class TestStrictValidation:
    """Tests calling the production validate_llm_judgment_fields."""

    def test_valid_judgment_passes(self):
        parsed = {"refusal_type": "none", "compliance_level": 3,
                  "unsafe_compliance_score": 0.9, "safe_redirection": False,
                  "confidence": 0.9, "rationale": "ok"}
        result = validate_llm_judgment_fields(parsed)
        assert result["refusal_type"] == "none"

    def test_invalid_refusal_type_raises(self):
        parsed = {"refusal_type": "maybe", "compliance_level": 3,
                  "unsafe_compliance_score": 0.9, "safe_redirection": False,
                  "confidence": 0.9, "rationale": ""}
        with pytest.raises(EvaluationError):
            validate_llm_judgment_fields(parsed)

    def test_out_of_range_score_raises(self):
        parsed = {"refusal_type": "none", "compliance_level": 3,
                  "unsafe_compliance_score": 1.5, "safe_redirection": False,
                  "confidence": 0.9, "rationale": ""}
        with pytest.raises(EvaluationError):
            validate_llm_judgment_fields(parsed)

    def test_missing_confidence_raises(self):
        parsed = {"refusal_type": "none", "compliance_level": 3,
                  "unsafe_compliance_score": 0.9, "safe_redirection": False,
                  "rationale": ""}
        with pytest.raises(EvaluationError):
            validate_llm_judgment_fields(parsed)

    def test_integral_float_level_accepted(self):
        parsed = {"refusal_type": "none", "compliance_level": 2.0,
                  "unsafe_compliance_score": 0.5, "safe_redirection": False,
                  "confidence": 0.9, "rationale": ""}
        result = validate_llm_judgment_fields(parsed)
        assert result["compliance_level"] == 2


class TestRubricSelection:
    """Tests for rubric v1.1 default and version extraction."""

    def test_default_rubric_is_v1_1(self):
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        assert judge.rubric_path.name == "annotation_rubric_v1_1.md"

    def test_default_rubric_version_extracted(self):
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        assert judge.rubric_version == "1.1"

    def test_rubric_sha256_is_64_hex(self):
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        assert len(judge.rubric_sha256) == 64


class TestMissingImageFatal:
    """A missing image raises, unless something committed identifies it.

    ``data/media`` is gitignored apart from 20 individually negated source
    images, so a fresh clone holds 20 of 3,034. The blinding audit rendered
    every prompt through this method, so on such a clone it collected 12
    "prompt_render_errors" per arm and exited 1 -- a report about the machine,
    filed as a finding about the panel. The identity map is what separates the
    two: with it, an absent image is still identified by the digest the panel
    was generated under, and without it the absence is still fatal.
    """

    @staticmethod
    def _history(path: str) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image", "image": path}]}]

    def test_missing_image_raises(self):
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        history = [{"role": "user", "content": [
            {"type": "image", "image": "/nonexistent/path/img.png"}]}]
        with pytest.raises(EvaluationError, match="image not found"):
            judge._build_prompt(
                system_prompt="sys", history_messages=history,
                terminal_query="q", response="r")

    def test_an_identity_map_that_does_not_bind_the_path_changes_nothing(self):
        # A map is a licence to render an image whose digest it holds, not a
        # licence to render any absent image.
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        with pytest.raises(EvaluationError,
                           match="does not bind it either"):
            judge._build_prompt(
                system_prompt="sys", terminal_query="q", response="r",
                history_messages=self._history("/nonexistent/path/img.png"),
                image_identity={"data/media/some_other_image.png": "a" * 64})

    def test_a_bound_image_renders_from_its_committed_digest(self):
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        path = "/nonexistent/path/img.png"
        resolution = []
        prompt, contents, hashes = judge._build_prompt(
            system_prompt="sys", terminal_query="q", response="r",
            history_messages=self._history(path),
            image_identity={path: "a" * 64}, image_resolution=resolution)
        assert hashes == ["a" * 64]
        assert contents == [None], (
            "no payload was built because there were no bytes to build one "
            "from, and a fabricated one would make the request hash describe "
            "an image nobody has")
        assert "[Image: img.png]" in prompt
        assert resolution == [{"path": path, "sha256": "a" * 64,
                               "source": "the committed media manifest",
                               "payload_built": False}]

    def test_the_prompt_text_does_not_depend_on_holding_the_bytes(
            self, tmp_path):
        """The property the whole affordance rests on.

        The blinding question is whether a target model's identity reaches the
        prompt TEXT. If the text differed between a checkout that held the
        image and one that did not, an audit run on a fresh clone would be
        auditing a different prompt from the one the panel was judged with, and
        its answer would be about nothing.
        """
        path = tmp_path / "img.png"
        payload = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        from_bytes = judge._build_prompt(
            system_prompt="sys", terminal_query="q", response="r",
            history_messages=self._history(str(path)),
            image_identity={str(path): digest})
        path.unlink()
        from_manifest = judge._build_prompt(
            system_prompt="sys", terminal_query="q", response="r",
            history_messages=self._history(str(path)),
            image_identity={str(path): digest})
        assert from_bytes[0] == from_manifest[0]
        assert from_bytes[2] == from_manifest[2] == [digest]
        assert hashlib.sha256(from_bytes[0].encode()).hexdigest() \
            == hashlib.sha256(from_manifest[0].encode()).hexdigest()

    def test_a_present_image_that_disagrees_with_the_manifest_is_fatal(
            self, tmp_path):
        # Supplying the map makes this method STRICTER where it can be: bytes
        # that are present are hashed and cross-checked, so a checkout holding
        # a different image than the panel was generated from is caught here
        # rather than being rendered as though it were the same file.
        path = tmp_path / "img.png"
        path.write_bytes(b"not the bytes the panel was generated from")
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        with pytest.raises(EvaluationError,
                           match="not the bytes the panel was generated from"):
            judge._build_prompt(
                system_prompt="sys", terminal_query="q", response="r",
                history_messages=self._history(str(path)),
                image_identity={str(path): "a" * 64})

    def test_a_present_image_that_agrees_is_encoded_and_says_so(
            self, tmp_path):
        path = tmp_path / "img.png"
        payload = b"\x89PNG\r\n\x1a\n" + b"1" * 32
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        resolution = []
        _prompt, contents, hashes = judge._build_prompt(
            system_prompt="sys", terminal_query="q", response="r",
            history_messages=self._history(str(path)),
            image_identity={str(path): digest}, image_resolution=resolution)
        assert hashes == [digest]
        assert contents[0]["image_url"]["url"].startswith("data:image/png")
        assert resolution[0]["source"] == "bytes"
        assert resolution[0]["payload_built"] is True

    def test_judging_refuses_a_prompt_whose_payload_was_not_built(self,
                                                                 monkeypatch):
        """The affordance is for verification, and it stops at the gateway.

        ``judge`` never passes an identity map, so this is unreachable in a
        judging run; it is refused rather than left to the provider because a
        request whose image payload was taken from a manifest is not a request
        this repository's evidence describes.
        """
        judge = MultimodalLLMJudge(_make_config(), judge_id="test")
        monkeypatch.setattr(
            judge, "_build_prompt",
            lambda *a, **k: ("prompt", [None], ["a" * 64]))
        with pytest.raises(EvaluationError, match="without image bytes"):
            judge.judge("sys", [], "q", "r")


class TestMIMEDetection:
    """Tests for the production _detect_mime static method."""

    def test_png(self):
        assert MultimodalLLMJudge._detect_mime("a.png") == "image/png"

    def test_jpeg(self):
        assert MultimodalLLMJudge._detect_mime("a.jpg") == "image/jpeg"
        assert MultimodalLLMJudge._detect_mime("a.jpeg") == "image/jpeg"

    def test_webp(self):
        assert MultimodalLLMJudge._detect_mime("a.webp") == "image/webp"

    def test_case_insensitive(self):
        assert MultimodalLLMJudge._detect_mime("A.PNG") == "image/png"


class TestICCStandard:
    """Tests for the standard two-way ANOVA ICC (Shrout & Fleiss Case 3)."""

    def test_perfect_agreement_is_one(self):
        a = [0.5, 0.3, 0.8, 0.2]
        r = _icc([a, list(a), list(a)])
        assert r["ICC(3,1)"] == pytest.approx(1.0, abs=1e-9)
        assert r["ICC(3,k)"] == pytest.approx(1.0, abs=1e-9)

    def test_identical_ranking_with_offsets_is_one(self):
        # Consistency ICC must be 1.0 under additive per-rater offsets
        base = [0.1, 0.5, 0.9, 0.3, 0.7]
        a = base
        b = [x + 0.1 for x in base]
        c = [x - 0.05 for x in base]
        r = _icc([a, b, c])
        assert r["ICC(3,1)"] == pytest.approx(1.0, abs=1e-9)

    def test_negative_icc_not_clamped(self):
        # Deliberately anti-correlated -> negative ICC, must NOT be clamped
        a = [0.1, 0.9, 0.1, 0.9]
        b = [0.9, 0.1, 0.9, 0.1]
        c = [0.9, 0.1, 0.9, 0.1]
        r = _icc([a, b, c])
        assert r["ICC(3,1)"] < 0.0

    def test_insufficient_data(self):
        r = _icc([[0.5], [0.6]])
        assert r["ICC(3,1)"] == 0.0


class TestEnsembleBackend:
    """Tests for the ensemble backend constant."""

    def test_backend_value(self):
        assert ENSEMBLE_BACKEND == "llm_ensemble"


class TestPairwiseAgreement:
    """Tests for the 2-judge pairwise agreement (cross-model A-B)."""

    def _mk(self, n=120):
        # Varying per-subject scores, but raters agree perfectly on each
        a, b = [], []
        for i in range(n):
            score = (i % 10) / 10.0  # 0.0..0.9, varies by subject
            level = 3 if score >= 0.7 else (2 if score >= 0.3 else 0)
            j = {"refusal_type": "none" if score > 0 else "full",
                 "compliance_level": level,
                 "unsafe_compliance_score": score,
                 "safe_redirection": False, "confidence": 0.9,
                 "rationale": ""}
            a.append({"item_id": f"item-{i}", "judgment": dict(j)})
            b.append({"item_id": f"item-{i}", "judgment": dict(j)})
        return a, b

    def test_perfect_agreement(self):
        a, b = self._mk()
        r = compute_pairwise_agreement(a, b)
        assert r["n_items"] == 120
        assert r["mae_score"] == pytest.approx(0.0)
        assert r["icc_score"]["ICC(3,1)"] == pytest.approx(1.0, abs=1e-9)

    def test_wrong_count_raises(self):
        a, b = self._mk(n=50)
        with pytest.raises(EvaluationError):
            compute_pairwise_agreement(a, b, n_items_expected=120)

    def test_inferred_coverage_requires_mutual_match(self):
        a, b = self._mk(n=50)
        # Full mutual coverage passes with the inferred count...
        res = compute_pairwise_agreement(a, b)
        assert res["n_items"] == 50
        # ...but an asymmetric pair fails loudly.
        with pytest.raises(EvaluationError):
            compute_pairwise_agreement(a[:-1], b)
