"""Iteration 11.7: which judges can see, and what each arm therefore claims.

The frozen ensemble's primary B (``glm-5.2``) cannot see images on this
gateway. Measured with the SAME instruction sent with and without the image
attached, its ``prompt_tokens`` is 27 both ways and it reports no
``image_tokens``, while ``qwen3.8-max`` goes 76 -> 300 with
``image_tokens=262`` and ``kimi-k3`` goes 100 -> 476 with ``image_tokens=371``.
Evidence: ``outputs/iteration_11/diagnostics/judge_vision/
identity_vision_capability.json``.

That makes the per-judge sensitivity analysis's ``judge_B`` arm a
VISION-ABLATION arm rather than a model-choice arm. The two claim different
things: a model-choice arm is evidence about what another model would judge,
while a vision-ablated arm is evidence about what the ensemble would judge
with the family media held out. Labelling the second as the first would report
a comparison nobody made.

The label is derived from the measurement rather than asserted against a model
id, so it follows the evidence and cannot go stale silently when the gateway's
models change -- and an identity that was never probed is recorded as
``unmeasured`` with no arm at all, because guessing either way would be worse
than saying so.

CI-safe: no network. ``assess_vision`` is a pure function over two recorded
probe responses, and the artifact test skips when the measurement is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from causal_mllm.evaluation.ensemble import build_judge_arm_meta

ROOT = Path(__file__).resolve().parents[2]
VISION_ARTIFACT = ROOT / "outputs" / "iteration_11" / "diagnostics" \
    / "judge_vision" / "identity_vision_capability.json"

# The frozen identities, restated rather than parsed out of the credentials
# file so that a repointed credential is a test failure to read.
PRIMARY_A = "qwen3.8-max"
PRIMARY_B = "glm-5.2"
ADJUDICATOR = "kimi-k3"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_script("iter11_probe_judge_gateway")


def _pipeline():
    # The pipeline raises at import time without an API key; the value is never
    # used here because nothing in these tests makes a request.
    os.environ.setdefault("LLM_JUDGE_API_KEY", "test-key-never-used")
    return _load_script("run_llm_judge_pipeline")


def _resp(prompt_tokens: int | None, image_tokens: int | None = None,
          ok: bool = True, head: str = "a fluent description") -> dict:
    details = {"cached_tokens": 0}
    if image_tokens is not None:
        details["image_tokens"] = image_tokens
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": 900,
             "completion_tokens": 100, "prompt_tokens_details": details}
    return {"ok": ok, "status": 200 if ok else 400, "usage": usage,
            "response_head": head}


class TestAssessVision:
    def test_image_tokens_alone_are_sufficient(self):
        v = probe.assess_vision(_resp(100), _resp(476, image_tokens=371))
        assert v["verdict"] == "sees_image"
        assert v["sees_image"] is True
        assert v["image_tokens"] == 371

    def test_a_token_delta_alone_is_sufficient(self):
        # A provider that charges for the pixels without itemizing them is
        # still taking them in.
        v = probe.assess_vision(_resp(76), _resp(300))
        assert v["verdict"] == "sees_image"
        assert v["prompt_tokens_delta"] == 224

    def test_identical_counts_and_no_image_tokens_is_blind(self):
        v = probe.assess_vision(_resp(27), _resp(27))
        assert v["verdict"] == "blind"
        assert v["sees_image"] is False
        assert v["prompt_tokens_delta"] == 0
        assert v["image_tokens"] is None

    def test_a_fluent_description_does_not_make_it_sighted(self):
        # The decisive case. A blind model asked to describe an image either
        # declines or confabulates something plausible, so what comes BACK is
        # never evidence about what went IN -- only the provider's own token
        # accounting is.
        v = probe.assess_vision(
            _resp(27, head="I cannot see an image."),
            _resp(27, head="The image shows a tidy desk with a closed "
                           "notebook and several pens."))
        assert v["verdict"] == "blind"
        assert "tidy desk" in v["image_response_head"], \
            "the confabulation is recorded, it just does not decide anything"

    def test_a_failed_request_is_unmeasurable_not_blind(self):
        # Reporting "blind" for a request that never landed would be a
        # conclusion drawn from an absence of evidence.
        v = probe.assess_vision(_resp(27), _resp(None, ok=False))
        assert v["verdict"] == "unmeasurable"
        assert v["sees_image"] is None

    def test_missing_usage_is_unmeasurable_not_blind(self):
        v = probe.assess_vision({"ok": True, "usage": None},
                                {"ok": True, "usage": None})
        assert v["verdict"] == "unmeasurable"


class TestArmLabelling:
    def _vision(self):
        return {
            PRIMARY_A: {"verdict": "sees_image", "image_tokens": 262},
            PRIMARY_B: {"verdict": "blind", "image_tokens": None},
            ADJUDICATOR: {"verdict": "sees_image", "image_tokens": 371},
        }

    def test_a_blind_primary_gets_the_vision_ablation_arm(self):
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            self._vision(), adjudicator_present=True)
        assert meta["judge_B"]["arm"] == "vision-ablated"
        assert "ABSENCE OF VISION" in meta["judge_B"]["arm_note"]
        assert PRIMARY_B in meta["judge_B"]["arm_note"]

    def test_a_sighted_primary_gets_the_model_choice_arm(self):
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            self._vision(), adjudicator_present=True)
        assert meta["judge_A"]["arm"] == "model-choice"
        assert "arm_note" not in meta["judge_A"]

    def test_an_unprobed_identity_gets_no_arm_at_all(self):
        # Guessing "model-choice" would claim an unsupported comparison;
        # guessing "vision-ablated" would disparage an arm that may be fine.
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            None, adjudicator_present=True)
        assert meta["judge_A"]["vision"] == {"verdict": "unmeasured"}
        assert meta["judge_B"]["vision"] == {"verdict": "unmeasured"}
        assert "arm" not in meta["judge_A"]
        assert "arm" not in meta["judge_B"]

    def test_an_unmeasurable_verdict_also_gets_no_arm(self):
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            {PRIMARY_A: {"verdict": "unmeasurable"}}, adjudicator_present=True)
        assert "arm" not in meta["judge_A"]

    def test_the_adjudicators_vision_is_recorded_on_the_ensemble(self):
        # The adjudicator sees every disagreement, including the image-bearing
        # ones, so whether IT can see is what makes the tie-break vision-informed.
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            self._vision(), adjudicator_present=True)
        assert meta["ensemble"]["adjudicator_model"] == ADJUDICATOR
        assert meta["ensemble"]["adjudicator_vision"]["verdict"] == "sees_image"

    @pytest.mark.parametrize("present,expected", [
        (True, f"ensemble({PRIMARY_A}, {PRIMARY_B}, "
               f"adjudicator={ADJUDICATOR})"),
        (False, f"ensemble({PRIMARY_A}, {PRIMARY_B})"),
    ])
    def test_the_ensemble_spelling_matches_the_sealed_artifacts(
            self, present, expected):
        # Iteration 9 and 10 committed reports carry this exact string. The
        # vision labels are additive; reformatting a sealed field is not.
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            self._vision(), adjudicator_present=present)
        assert meta["ensemble"]["model_id"] == expected
        assert meta["ensemble"]["adjudicator_model"] == (
            ADJUDICATOR if present else None)


class TestPipelineReadsTheMeasurement:
    def test_a_missing_artifact_is_empty_not_fatal(self, monkeypatch,
                                                    tmp_path):
        pipeline = _pipeline()
        monkeypatch.setattr(pipeline, "VISION_ARTIFACT",
                            tmp_path / "absent.json")
        assert pipeline.load_identity_vision() == {}

    def test_an_unreadable_artifact_is_empty_not_fatal(self, monkeypatch,
                                                       tmp_path):
        pipeline = _pipeline()
        bad = tmp_path / "identity_vision_capability.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(pipeline, "VISION_ARTIFACT", bad)
        assert pipeline.load_identity_vision() == {}

    def test_it_returns_the_identities_keyed_by_model_id(self, monkeypatch,
                                                         tmp_path):
        pipeline = _pipeline()
        good = tmp_path / "identity_vision_capability.json"
        good.write_text(json.dumps({"identities": {
            PRIMARY_B: {"verdict": "blind"}}}), encoding="utf-8")
        monkeypatch.setattr(pipeline, "VISION_ARTIFACT", good)
        assert pipeline.load_identity_vision() == {
            PRIMARY_B: {"verdict": "blind"}}


@pytest.fixture(scope="module")
def measured():
    """The committed measurement. Only requested by the class below, which
    skips when the probe has not been run."""
    return json.loads(VISION_ARTIFACT.read_text(encoding="utf-8"))


class TestReferenceIdentitiesAreNotLoadBearing:
    """``--also-probe`` measures a model nobody froze. It must not be able to
    decide anything about an arm that somebody did."""

    def test_a_reference_identity_in_the_dict_labels_no_arm(self):
        # The lookup is by model id, so the only thing standing between a
        # reference measurement and a frozen arm's label is that the frozen
        # ids are the ones asked for. A stray key must be inert.
        vision = {
            PRIMARY_A: {"verdict": "sees_image", "image_tokens": 262},
            PRIMARY_B: {"verdict": "blind", "image_tokens": None},
            "deepseek-v3.2": {"verdict": "blind", "image_tokens": None},
        }
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            vision, adjudicator_present=True)
        assert set(meta) == {"judge_A", "judge_B", "ensemble"}
        assert meta["judge_A"]["arm"] == "model-choice"
        assert meta["judge_B"]["arm"] == "vision-ablated"

    def test_a_blind_reference_identity_cannot_make_a_frozen_arm_blind(self):
        # The failure this guards against: a reference measurement keyed to an
        # id that happens to equal a frozen primary's would relabel that arm.
        # The probe excludes ensemble ids from --also-probe, and this pins the
        # consumer side of the same rule.
        vision = {PRIMARY_A: {"verdict": "sees_image"},
                  PRIMARY_B: {"verdict": "sees_image"}}
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            vision, adjudicator_present=True)
        assert meta["judge_A"]["arm"] == "model-choice"
        assert meta["judge_B"]["arm"] == "model-choice"
        assert all("vision-ablated" not in json.dumps(entry)
                   for entry in meta.values())


@pytest.mark.skipif(not VISION_ARTIFACT.exists(),
                    reason="the gateway probe has not been run")
class TestTheFrozenEnsembleAsMeasured:
    """Pins the measured reality of the identities this iteration froze."""

    def test_the_frozen_primary_a_sees_the_family_media(self, measured):
        assert measured["identities"][PRIMARY_A]["verdict"] == "sees_image"
        assert measured["identities"][PRIMARY_A]["image_tokens"] > 0

    def test_the_frozen_adjudicator_sees_the_family_media(self, measured):
        # Decisive for the labels: every A/B disagreement is resolved by an
        # adjudicator that received the image, so no adjudicated label is
        # vision-blind.
        assert measured["identities"][ADJUDICATOR]["verdict"] == "sees_image"
        assert measured["identities"][ADJUDICATOR]["image_tokens"] > 0

    def test_the_frozen_primary_b_is_blind(self, measured):
        b = measured["identities"][PRIMARY_B]
        assert b["verdict"] == "blind"
        assert b["sees_image"] is False
        assert b["prompt_tokens_delta"] == 0
        assert b["image_tokens"] is None

    def test_so_the_primary_b_arm_is_recorded_as_vision_ablated(self,
                                                               measured):
        meta = build_judge_arm_meta(
            (PRIMARY_A, PRIMARY_B), ADJUDICATOR, "llm_adjudicator",
            measured["identities"], adjudicator_present=True)
        assert meta["judge_B"]["arm"] == "vision-ablated"
        assert meta["judge_A"]["arm"] == "model-choice"

    def test_the_verdict_is_backed_by_a_same_text_control(self, measured):
        # The measurement is only meaningful because the two requests carried
        # identical text; without that a token difference could be the wording.
        assert "same instruction sent with and without the image" \
            in measured["method"]
        assert measured["probe_image"]["sha256"]

    def test_the_identities_section_holds_exactly_the_frozen_three(
            self, measured):
        # The pipeline reads this section and nothing else, so a reference
        # identity leaking into it would be able to label a frozen arm.
        assert set(measured["identities"]) == {PRIMARY_A, PRIMARY_B,
                                               ADJUDICATOR}

    def test_a_measured_confabulator_is_kept_out_of_the_ensemble_section(
            self, measured):
        reference = measured.get("reference_identities") or {}
        if not reference:
            pytest.skip("the probe was run without --also-probe")
        assert not (set(reference) & set(measured["identities"]))
        assert "NOT members of the frozen ensemble" \
            in measured["reference_note"]

    def test_blindness_presents_both_ways_and_neither_is_a_signal(
            self, measured):
        """The reason ``assess_vision`` ignores response text entirely.

        Two blind identities, opposite behaviour. One declines; the other
        writes a fluent, specific, confident description of an image it never
        received -- the same sentence with and without the image, because at
        temperature 0.0 the pixels contribute nothing to a deterministic
        completion. Reading either response as evidence about sight gets one
        of the two wrong.
        """
        reference = measured.get("reference_identities") or {}
        if not reference:
            pytest.skip("the probe was run without --also-probe")
        confabulator = next(
            (mid for mid, v in reference.items()
             if v["verdict"] == "blind" and v["image_response_head"]
             and v["image_response_head"] == v["text_response_head"]
             and "no image" not in v["image_response_head"].lower()), None)
        assert confabulator is not None, \
            "expected a blind identity that describes an image it never saw"
        v = reference[confabulator]
        assert v["prompt_tokens_delta"] == 0 and v["image_tokens"] is None
        assert measured["identities"][PRIMARY_B]["image_response_head"]
        assert "no image" in measured["identities"][PRIMARY_B][
            "image_response_head"].lower(), \
            "the frozen blind primary declines instead; both are blind"
