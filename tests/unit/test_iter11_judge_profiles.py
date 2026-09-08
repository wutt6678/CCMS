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

The audit that measures those channels is pinned here too, on the question of
what it reports when a checkout cannot measure part of it: ``data/media`` is
gitignored apart from 20 individually negated source images, so a fresh clone
holds 20 of 3,034 files, and an audit that treated the missing bytes as a
blinding failure reported one on a machine that had done nothing to the
evidence.

CI-safe: the profiles are read as JSON, the blinding checks use the production
context helpers over a sample of the committed journals, and the audit's own
report is manufactured under ``tmp_path``. The judge module is imported but
never called: ``iter11_blinding_audit`` only assembles prompt text, and no test
here reaches a gateway.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import random
import re
import sys
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


# ---------------------------------------------------------------------------
# The audit's own report: what it measured, and what this checkout could not
# ---------------------------------------------------------------------------

def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


blinding = _load_script("iter11_blinding_audit")

MANIFEST = ROOT / "outputs" / "iteration_11" / "media_manifest.json"


def _resolutions(*records) -> dict:
    """One arm's worth of image resolutions, as ``_build_prompt`` reports them."""
    return {"item-0001": [dict(path=p, sha256=s, source=src,
                                payload_built=built)
                          for p, s, src, built in records]}


def _target(**overrides) -> dict:
    entry = {
        "run_dir": "outputs/iteration_11/generations/x/replay",
        "complete": True, "n_records": 100,
        "payload_identity_hits": [], "payload_other_arm_identity_hits": [],
        "prompt_identity_hits": [], "prompt_judge_identity_hits": [],
        "iteration10_blinding_violations": [], "prompt_render_errors": [],
        "prompts_rendered": 12, "mirror_crosscheck": {"present": True,
                                                      "identical": True},
        "response_self_identification": {"n_items": 0, "rate": 0.0},
    }
    entry.update(overrides)
    return entry


def _report(*, media_verifiable=True, targets=None, **overrides) -> dict:
    rendered = 12
    from_manifest = 0 if media_verifiable else rendered
    report = {
        "targets": targets if targets is not None else {"target_x": _target()},
        "cross_target": {
            "shared_context": {"uniform": True,
                               "n_distinct_system_prompts": 1},
            "item_id_alignment": {"status": "aligned"},
        },
        "media_identity_here": {
            "manifest_present": True,
            "manifest_path": "outputs/iteration_11/media_manifest.json",
            "manifest_rollup_sha256": "a" * 64,
            "manifest_n_files": 3034,
            "n_images_rendered": rendered,
            "n_hashes_from_bytes": rendered - from_manifest,
            "n_hashes_from_the_manifest": from_manifest,
            "n_distinct_paths_identified_from_the_manifest":
                4 if from_manifest else 0,
            "media_bytes_verifiable_here": media_verifiable,
        },
    }
    report.update(overrides)
    return report


def _run_audit(monkeypatch, argv, report, capsys=None):
    monkeypatch.setattr(blinding, "audit", lambda: report)
    monkeypatch.setattr(sys, "argv", ["iter11_blinding_audit.py", *argv])
    return blinding.main()


class TestAbsentMediaIsAStatementAboutTheCheckout:
    """The finding: a fresh clone produced 12 render errors per arm and exit 1.

    ``data/media`` is gitignored apart from 20 individually negated source
    images, so ``audit_target`` could not build a prompt that referenced any of
    the other 3,014, and a machine that had done nothing to the evidence
    reported a BLINDING FAIL. The prompt text carries an image's file name and
    never its contents, so the blinding question is answerable without the
    bytes; what is not answerable without them is the byte identity of the
    media, and that is filed as an incomplete run rather than as a finding.
    """

    def test_a_run_that_held_every_image_is_complete(self):
        statement = blinding.media_identity_statement(
            _resolutions(("data/media/a.png", "b" * 64, "bytes", True),
                         ("data/media/b.png", "c" * 64, "bytes", True)),
            {"present": True, "path": "m.json", "rollup_sha256": "a" * 64,
             "n_files": 2})
        assert statement["media_bytes_verifiable_here"] is True
        assert statement["n_hashes_from_bytes"] == 2
        assert statement["n_hashes_from_the_manifest"] == 0
        assert "note" not in statement, (
            "a note on a complete run would be compared by --verify on a "
            "machine that has nothing to explain")

    def test_a_run_that_identified_images_from_the_manifest_says_so(self):
        statement = blinding.media_identity_statement(
            _resolutions(("data/media/a.png", "b" * 64,
                           "the committed media manifest", False),
                         ("data/media/b.png", "c" * 64, "bytes", True)),
            {"present": True, "path": "m.json", "rollup_sha256": "a" * 64,
             "n_files": 2})
        assert statement["media_bytes_verifiable_here"] is False
        assert statement["n_hashes_from_the_manifest"] == 1
        assert statement["n_hashes_from_bytes"] == 1
        assert "m.json" in statement["note"]
        assert "prompt text" in statement["note"], (
            "the note has to say what the fallback does NOT cost, or a reader "
            "cannot tell an incomplete byte check from an unmeasured clause")

    def test_the_distinct_path_count_is_paths_and_not_images(self):
        statement = blinding.media_identity_statement(
            _resolutions(*[("data/media/a.png", "b" * 64, "manifest", False),
                           ("data/media/a.png", "b" * 64, "manifest", False),
                           ("data/media/b.png", "c" * 64, "manifest", False)]),
            {"present": True, "path": "m.json"})
        assert statement["n_images_rendered"] == 3
        assert statement[
            "n_distinct_paths_identified_from_the_manifest"] == 2

    def test_the_committed_manifest_binds_every_image_it_lists(self):
        if not MANIFEST.exists():
            pytest.skip(f"no committed media manifest at {MANIFEST}")
        identity, meta = blinding.media_identity()
        assert meta["present"] is True
        assert identity, "an empty identity map would make every absent image " \
                         "fatal again, which is the behaviour being replaced"
        assert len(identity) == meta["n_files"]
        assert meta["n_digests_usable"] == len(identity)
        assert re.fullmatch(r"[0-9a-f]{64}", meta["rollup_sha256"])
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in identity.values())

    def test_no_manifest_is_an_empty_map_and_not_an_error(self, monkeypatch,
                                                          tmp_path):
        monkeypatch.setattr(blinding, "MEDIA_MANIFEST_PATH",
                            tmp_path / "absent.json")
        identity, meta = blinding.media_identity()
        assert identity == {}
        assert meta["present"] is False
        assert meta["rollup_sha256"] is None

    def test_an_unreadable_manifest_is_said_to_be_unreadable(self, monkeypatch,
                                                             tmp_path):
        path = tmp_path / "media_manifest.json"
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(blinding, "MEDIA_MANIFEST_PATH", path)
        identity, meta = blinding.media_identity()
        assert identity == {}
        assert meta["present"] is False
        assert "JSONDecodeError" in meta["unreadable"]

    def test_the_render_is_given_the_manifest_identity(self):
        # Structural, because the behavioural alternative is a test that needs
        # 3,014 images to be absent. What the kwarg DOES is pinned in
        # test_llm_judge_fixes; what is pinned here is that the audit threads it
        # all the way down, and by name -- a positional pass would keep working
        # right up to the day somebody reorders audit_target's parameters.
        assert "image_identity" in inspect.signature(
            blinding.audit_target).parameters
        audit_source = inspect.getsource(blinding.audit)
        assert "media_identity()" in audit_source
        assert "image_identity=image_identity" in audit_source
        target_source = inspect.getsource(blinding.audit_target)
        assert "image_identity=image_identity" in target_source
        assert "image_resolution=" in target_source, (
            "without the resolution record the report cannot say how much of "
            "the media this checkout held, and 3 would be indistinguishable "
            "from 0 on a guess")


class TestTheAuditKeepsThreeAnswersApart:
    def test_clean_and_complete_is_zero(self, monkeypatch, capsys):
        assert _run_audit(monkeypatch, ["--json"], _report()) == 0
        assert "BLINDING PASS" in capsys.readouterr().out

    def test_clean_but_not_measurable_here_is_three(self, monkeypatch, capsys):
        code = _run_audit(monkeypatch, ["--json"],
                          _report(media_verifiable=False))
        out = capsys.readouterr().out
        assert code == 3
        assert "BLINDING PASS" in out, (
            "nothing was found wrong, and 3 is not a soft way of saying 1")

    def test_a_finding_outranks_incompleteness(self, monkeypatch, capsys):
        report = _report(media_verifiable=False, targets={
            "target_x": _target(prompt_identity_hits=[{"term": "qwen"}])})
        assert _run_audit(monkeypatch, ["--json"], report) == 1
        assert "BLINDING FAIL" in capsys.readouterr().out

    def test_a_render_error_is_still_a_finding_where_the_bytes_were_held(
            self, monkeypatch):
        # The fix is a fallback for absent images, not a licence to swallow a
        # render that failed on a checkout holding the media.
        report = _report(targets={"target_x": _target(
            prompt_render_errors=["EvaluationError: image not found"])})
        assert _run_audit(monkeypatch, ["--json"], report) == 1

    def test_verify_with_nothing_filed_is_two(self, monkeypatch, tmp_path):
        code = _run_audit(monkeypatch,
                          ["--verify", "--out", str(tmp_path / "absent.json")],
                          _report())
        assert code == 2

    def test_verify_ignores_the_machine_statement_on_both_sides(
            self, monkeypatch, tmp_path, capsys):
        """The other half of the finding.

        The committed artifact predates ``media_identity_here``, and a checkout
        that held every image would file a different one from a checkout that
        held none. Neither is a statement about the blinding, so neither may
        decide the comparison -- and the advice printed on a real mismatch must
        not send a reader to re-run for a reason that is not the reason.
        """
        filed = _report(media_verifiable=True)
        out = tmp_path / "blinding.json"
        out.write_text(json.dumps(filed, indent=2), encoding="utf-8")
        fresh = _report(media_verifiable=False)
        code = _run_audit(monkeypatch, ["--verify", "--out", str(out)], fresh)
        text = capsys.readouterr().out
        assert "does not match a fresh derivation" not in text
        assert code == 3, (
            "the derivation matched on every comparable key, and this checkout "
            "could not measure the byte identity of the media")

    def test_verify_still_catches_a_real_disagreement(
            self, monkeypatch, tmp_path, capsys):
        filed = _report()
        out = tmp_path / "blinding.json"
        out.write_text(json.dumps(filed, indent=2), encoding="utf-8")
        fresh = _report(targets={"target_x": _target(n_records=101)})
        code = _run_audit(monkeypatch, ["--verify", "--out", str(out)], fresh)
        text = capsys.readouterr().out
        assert code == 1
        assert "does not match a fresh derivation" in text
        assert "differing top-level key(s): ['targets']" in text

    def test_the_machine_statement_keys_are_only_the_media_statement(self):
        assert blinding.MACHINE_STATEMENT_KEYS == frozenset(
            {"media_identity_here"})
        for key in ("targets", "cross_target"):
            assert key not in blinding.MACHINE_STATEMENT_KEYS, (
                f"skipping {key} would skip the audit's findings")
