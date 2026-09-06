"""Iteration 11.7: judge scale profiles and model-identity blinding.

Two things can silently corrupt the cross-model judging, and neither is
visible in a judge's output:

* a profile that points two targets at one ``output_dir``, or at another
  target's run directory.  The pipeline resolves everything from
  ``CCMS_SCALE``, so a copy-paste slip in ``scale_profiles.json`` merges
  two models' labels under one artifact and every downstream number is
  still computed happily from the mixture.
* a target-model identity reaching the judge.  The frozen protocol requires
  "target-model identity is blinded in judge prompts".  Iteration 10 had one
  target, so that clause was vacuous; Iteration 11 has four, which makes it
  a testable claim with three channels -- the payload, the shared context,
  and the model naming itself inside its own response.

CI-safe: the profiles are read as JSON and the blinding checks use the
production context helpers over a sample of the committed journals. The
judge pipeline itself is NOT imported, because it raises at import time
without ``LLM_JUDGE_API_KEY``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    _extract_conversation_context,
)
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

ROOT = Path(__file__).resolve().parents[2]
PROFILES = json.loads(
    (ROOT / "configs" / "evaluation" / "scale_profiles.json").read_text(
        encoding="utf-8"))
PROTOCOL = json.loads(
    (ROOT / "outputs" / "iteration_11" / "protocol"
     / "iteration_11_protocol.json").read_text(encoding="utf-8"))
GENERATIONS = ROOT / "outputs" / "iteration_11" / "generations"

# The matrix has FIVE entries. qwen35_9b is the frozen Iteration 10
# upper-Qwen reference: its confirmatory panel is the Scale-C run that
# Iteration 10 already replayed and judged under the scale_c profile, so
# 11.6 replays and 11.7 judges only the four new targets, and the
# cross-model analysis compares them against that existing baseline.
# Stated explicitly rather than parsed out of the role string, so the test
# pins the arrangement instead of following the artifact.
BASELINE = "qwen35_9b"
TARGETS = [e["model_key"] for e in PROTOCOL["model_matrix"]
           if e["model_key"] != BASELINE]
ITER11 = {k: v for k, v in PROFILES.items() if k.startswith("iteration_11_")}

# Sampled rather than exhaustive: the journals hold 600 records each and a
# unit test should not read 2400 of them to pin a property that is either
# structurally present or structurally absent.
SAMPLE = 24

# Identity vocabulary per target, deliberately over-inclusive. A false
# positive costs a look; a false negative silently de-blinds the
# confirmatory judging. The judge's OWN model ids are checked too -- a
# prompt that names the judge is a different defect, but still one.
IDENTITY_TERMS = {
    "qwen35_2b": [r"\bqwen35_2b\b", r"Qwen/Qwen3\.5-2B", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b2B\b"],
    "qwen35_4b": [r"\bqwen35_4b\b", r"Qwen/Qwen3\.5-4B", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b4B\b"],
    "ministral3_3b": [r"\bministral3_3b\b", r"\bministral\b",
                      r"\bmistral\b", r"\b3B\b"],
    "phi4_mm": [r"\bphi4_mm\b", r"\bphi-4\b", r"\bphi4\b", r"\bmicrosoft\b"],
    # The baseline is an arm of the analysis too, so its name must not
    # appear in another arm's payload any more than a sibling's may.
    "qwen35_9b": [r"\bqwen35_9b\b", r"Qwen/Qwen3\.5-9B", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b9B\b"],
}
JUDGE_TERMS = [r"\bqwen3\.8-max\b", r"\bglm-5\.2\b", r"\bkimi-k3\b"]


def _journal(model_key: str) -> Path:
    return (GENERATIONS / model_key / f"confirmatory-100f-t1536-{model_key}"
            / "replay_outputs.jsonl")


def _sample(model_key: str) -> list[dict]:
    path = _journal(model_key)
    if not path.exists():
        pytest.skip(f"{model_key} has no 11.6 journal yet")
    records = list(read_jsonl(path))
    return random.Random(42).sample(records, min(SAMPLE, len(records)))


def _families() -> dict:
    panel = ITER11[f"iteration_11_{TARGETS[0]}"]["validated_families"]
    out = {}
    for rec in read_jsonl(ROOT / panel):
        fam = CausalFamily.from_dict(rec)
        out[fam.family_id] = fam
    return out


def _blinded_payload(record: dict, family: CausalFamily) -> str:
    """What the judge is shown, serialized.

    Mirrors run_llm_judge_pipeline.prepare_blinded_items, minus the
    ``family_id`` and ``variant`` keys it keeps for provenance and never
    shows. Reproduced rather than imported because that module raises at
    import time without an API key.
    """
    _, history, terminal = _extract_conversation_context(
        family, record["variant"])
    shown = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "conversation_history": history,
        "terminal_query": terminal,
        "response": record.get("response", ""),
    }
    return json.dumps(shown, ensure_ascii=False)


class TestIteration11JudgeProfiles:
    def test_there_is_one_profile_per_replayed_target(self):
        assert sorted(k[len("iteration_11_"):] for k in ITER11) == \
            sorted(TARGETS)

    def test_the_fifth_matrix_entry_is_the_frozen_iteration_10_baseline(self):
        # Pinned so that a change to the matrix is a test failure to read,
        # not a silently altered set of judged arms.
        matrix = [e["model_key"] for e in PROTOCOL["model_matrix"]]
        assert sorted(matrix) == sorted(TARGETS + [BASELINE])
        entry = next(e for e in PROTOCOL["model_matrix"]
                     if e["model_key"] == BASELINE)
        assert "frozen Iteration 10" in entry["role"]
        assert f"iteration_11_{BASELINE}" not in PROFILES

    def test_no_two_targets_share_an_output_directory(self):
        # The hazard this exists for: one shared output_dir and the second
        # session overwrites the first target's labels.
        dirs = [v["output_dir"] for v in ITER11.values()]
        assert len(set(dirs)) == len(dirs), dirs

    def test_each_output_directory_names_its_own_target(self):
        for key, profile in ITER11.items():
            target = key[len("iteration_11_"):]
            assert profile["output_dir"].rstrip("/").endswith(target), key

    def test_each_replay_run_is_that_targets_own_run(self):
        for key, profile in ITER11.items():
            target = key[len("iteration_11_"):]
            run = Path(profile["replay_run"])
            assert run.parent.name == target, key
            assert run.name == f"confirmatory-100f-t1536-{target}", key

    def test_no_two_targets_share_a_replay_run(self):
        runs = [v["replay_run"] for v in ITER11.values()]
        assert len(set(runs)) == len(runs), runs

    def test_all_four_share_the_one_frozen_panel(self):
        # Sharing the panel is the point of the cross-model arm; sharing the
        # output directory would be a bug. The two must not be conflated.
        panels = {v["validated_families"] for v in ITER11.values()}
        assert len(panels) == 1, panels
        assert panels == {PROFILES["scale_c"]["validated_families"]}

    def test_a_completed_runs_profile_agrees_with_its_report(self):
        checked = 0
        for key, profile in ITER11.items():
            report = Path(profile["replay_run"]) / "replay_report.json"
            if not (ROOT / report).exists():
                continue
            data = json.loads((ROOT / report).read_text(encoding="utf-8"))
            assert data["model_key"] == key[len("iteration_11_"):], key
            assert data["run_id"] == Path(profile["replay_run"]).name, key
            checked += 1
        assert checked, "no completed 11.6 run to cross-check the profiles"

    def test_the_legacy_profiles_are_untouched(self):
        # Iteration 9 and 10 evidence is immutable; adding Iteration 11
        # profiles must not move theirs.
        assert PROFILES["scale_b"]["replay_run"] == \
            "outputs/replay_runs/scale-b-2026-08-28-t1536-final-qwen35-9b"
        assert PROFILES["scale_b"]["output_dir"] == \
            "outputs/llm_judge_artifacts"
        assert PROFILES["scale_c"]["replay_run"] == \
            "outputs/scale_c/replay_runs/scale-c-100-t1536-qwen35-9b"
        assert PROFILES["scale_c"]["output_dir"] == \
            "outputs/scale_c/llm_judge_artifacts"

    def test_an_iteration_11_profile_never_reuses_a_legacy_directory(self):
        legacy = {PROFILES["scale_b"]["output_dir"],
                  PROFILES["scale_c"]["output_dir"]}
        assert not (legacy & {v["output_dir"] for v in ITER11.values()})


class TestModelIdentityBlinding:
    """The frozen clause, now that more than one target exists."""

    def test_the_protocol_clause_is_what_these_tests_read(self):
        clause = PROTOCOL["frozen_inputs"]["judging"]["output_blinding"]
        assert "target-model identity is blinded" in clause

    @pytest.mark.parametrize("model_key", TARGETS)
    def test_the_own_target_identity_never_reaches_the_payload(
            self, model_key):
        families = _families()
        leaks = []
        for record in _sample(model_key):
            payload = _blinded_payload(record, families[record["family_id"]])
            for term in IDENTITY_TERMS[model_key]:
                if re.search(term, payload, re.IGNORECASE):
                    leaks.append((record["family_id"], record["variant"],
                                  term))
        assert not leaks, leaks[:5]

    @pytest.mark.parametrize("model_key", TARGETS)
    def test_no_other_targets_identity_leaks_either(self, model_key):
        # A judge that could tell the arms apart by a stray mention of a
        # DIFFERENT model would be just as unblinded.
        families = _families()
        others = [t for t in TARGETS + [BASELINE] if t != model_key]
        leaks = []
        for record in _sample(model_key):
            payload = _blinded_payload(record, families[record["family_id"]])
            for other in others:
                for term in IDENTITY_TERMS[other]:
                    if re.search(term, payload, re.IGNORECASE):
                        leaks.append((other, record["family_id"], term))
        assert not leaks, leaks[:5]

    @pytest.mark.parametrize("model_key", TARGETS)
    def test_the_judge_identities_are_not_in_the_payload(self, model_key):
        families = _families()
        leaks = []
        for record in _sample(model_key):
            payload = _blinded_payload(record, families[record["family_id"]])
            for term in JUDGE_TERMS:
                if re.search(term, payload, re.IGNORECASE):
                    leaks.append((record["family_id"], term))
        assert not leaks, leaks[:5]

    @pytest.mark.parametrize("model_key", TARGETS)
    def test_variant_and_family_metadata_stay_out_of_the_payload(
            self, model_key):
        # The Iteration 10 half of the clause: variant/family blinded.
        families = _families()
        leaks = []
        for record in _sample(model_key):
            payload = _blinded_payload(record, families[record["family_id"]])
            for forbidden in (record["family_id"], record["variant"]):
                if forbidden and forbidden in payload:
                    leaks.append((record["family_id"], forbidden))
        assert not leaks, leaks[:5]

    def test_the_shared_context_cannot_separate_the_arms(self):
        # Channel 2: if the system prompt differed per target, a judge could
        # tell the arms apart with no name appearing anywhere. All four
        # targets record one system_prompt_sha256, and it is this one.
        expected = sha256_text(DEFAULT_SYSTEM_PROMPT)
        assert expected == (
            "e51b41e6a82264406aa184050eb0552cce8653ff097db9225e775a20b1bf7d9c")
        for key in ITER11:
            target = key[len("iteration_11_"):]
            path = _journal(target)
            if not path.exists():
                continue
            shas = {r.get("system_prompt_sha256")
                    for r in read_jsonl(path)}
            assert shas == {expected}, (target, shas)

    def test_variant_labels_are_anonymized_consistently_across_targets(self):
        # One seed, one map: the anonymization must not differ per target,
        # or the same variant would wear different labels in different arms.
        maps = [_build_anonymization_map(42) for _ in TARGETS]
        assert all(m == maps[0] for m in maps)
        assert sorted(maps[0].values()) == ["A", "B", "C", "D", "E", "F"]
