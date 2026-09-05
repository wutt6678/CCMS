"""Evidence-integrity tests for the Iteration 11 review findings.

Five defects were found in the 11.1-11.4 substrate after it was committed.
Each is pinned here so it cannot silently return:

P0-1  The preflight hashed the frozen panel with the whitespace-normalizing
      ``sha256_text`` while the frozen protocol and the replay runner hash
      RAW BYTES, so every committed artifact reported a panel digest that
      matched nothing and could not be compared against the protocol.
P0-2  Artifacts recorded ``code_commit`` but not whether the tree was
      clean. All four were generated from a dirty tree whose HEAD predated
      the Phi-4 adapter, so the recorded commit could not reconstruct the
      run — and the run fingerprint did not move when the tree was dirty.
P0-3  "Confirmatory" enforced only revision pinning: panel identity, family
      and variant counts, cap, decoding, clean tree, processor revision,
      dependency environment, 11.5 eligibility and ``--overwrite`` were all
      free.
P0-4  Outputs were persisted only after the final family, so a kill lost
      every completed family; resume also accepted records with no
      fingerprint/model_key because ``None`` was treated as a wildcard, and
      dropped prior failure records.
P1-5  The dependency lock was recorded but never compared against the
      environment actually running, the selected ``--lock`` never reached
      the fingerprint, a failed ``pip freeze`` was accepted, and the
      interpreter path was hashed as if it were dependency identity.

A second review round then found that untracked files were counted as clean,
that a third-party editable revision was being normalized OUT of dependency
identity, and that the 11.5 eligibility report recorded provenance it never
verified. A third round found four more, each pinned below:

P1-6  The 12-family eligibility selection was not externally pre-registered:
      the digest was recomputed from the ids in the SAME report, so replacing
      both together still passed. The gate now re-derives the selection from
      the frozen Iteration 10 reference run and adjudicated labels.
P1-7  Any nonempty dictionary of passing gates could authorize a run, so a
      report could simply omit the vision, truncation, determinism and
      terminal-query checks. ``ELIGIBILITY_REQUIRED_GATES`` is now exact —
      missing and unexpected names are both rejected — and each gate must
      carry evidence that is semantically consistent with ``passed``.
P2-8  ``processor_revision`` only had to LOOK like a 40-hex SHA; it is now
      compared against the revision the lock resolved for this target.
P2-9  Editable installs were detected only when the freeze line ended in a
      hex revision, leaving ``-e /path`` and ``-e file:///path`` — the forms
      that name no revision at all — invisible. Every third-party ``-e``
      entry is now detected and refused.

Torch-free and offline: the gate and journal are exercised against
synthetic protocols/panels/locks and a stub backend.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from causal_mllm import seeds
from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.replay import confirmatory, registry, runner, selection
from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.confirmatory import (
    ELIGIBILITY_GATE_EVIDENCE,
    ELIGIBILITY_GENERATIONS_ROOT,
    ELIGIBILITY_N_ATTEMPTS,
    ELIGIBILITY_N_FAMILIES,
    ELIGIBILITY_REQUIRED_FIELDS,
    ELIGIBILITY_REQUIRED_GATES,
    GENERATIONS_ROOT,
    enforce_confirmatory_protocol,
    enforce_eligibility_protocol,
    protocol_sha256,
    selected_families_sha256,
    validate_eligibility_report,
    validate_gate_entry,
)
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import (
    LOCK_IDENTITY_FIELDS,
    ResolvedModel,
    dependency_lock_sha256,
    dependency_lock_snapshot,
    editable_installs,
    editable_vcs_revisions,
    load_dependency_lock,
    verify_active_dependency_lock,
)
from causal_mllm.replay.runner import (
    REPLAY_FAILURES_FILE,
    REPLAY_OUTPUTS_FILE,
    append_journal,
    iteration11_run_fingerprint,
    run_replay_stage,
    validate_journal,
)
from causal_mllm.seeds import sha256_text
from causal_mllm.validation.relations import _file_sha256
from tests.unit.test_grounding import CLEAN_Q, _built_family
from tests.unit.test_iter11_adapters import _families, _spec, _StubAdapter

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs" / "scale_c" / "families_panel" \
    / "validated_families.jsonl"
PROTOCOL = ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"
PREFLIGHT_ROOT = ROOT / "outputs" / "iteration_11" / "preflight"
MODEL_KEYS = ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm")

#: The frozen panel's RAW-BYTE digest, as recorded by the frozen protocol
#: and by scripts/iter11_freeze_protocol.py (FROZEN_PANEL_SHA).
FROZEN_PANEL_SHA = (
    "97b8bb7cb3a69903c1988c6a1a8fb1ff9167fdd135c9e5b4b3bd69828fc69863")
#: What a whitespace-normalizing digest of the SAME file produces. Pinned
#: so the two can never be silently conflated again.
NORMALIZED_PANEL_SHA = (
    "0d77226b2d5f4c030bede5e351a8ee41df1c689b47b9253a758bc389ec34e1db")

REVISION = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
PROCESSOR_REVISION = "0f1e2d3c4b5a69788796a5b4c3d2e1f001122334"
COMMIT = "c" * 40


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preflight = _load_script("iter11_model_preflight")
write_selection = _load_script("iter11_write_selection")
run_eligibility = _load_script("iter11_run_eligibility")


def _tree(dirty, dirty_paths=(), untracked_paths=(), own_outputs=(),
          caches=(), code_dirty_paths=None) -> dict:
    """A ``causal_mllm.seeds.code_tree_status`` result.

    ``code_dirty_paths`` defaults to every dirty path, which is the
    conservative reading: unless a test says otherwise, a dirty tree is dirty
    in the execution-relevant sense too.
    """
    dirty_paths = list(dirty_paths)
    return {"dirty": dirty, "dirty_paths": dirty_paths,
            "untracked_paths": list(untracked_paths),
            "code_dirty_paths": (dirty_paths if code_dirty_paths is None
                                 else list(code_dirty_paths)),
            "excluded_own_outputs": list(own_outputs),
            "excluded_cache_paths": list(caches)}


def _git_paths(modified=(), untracked=()):
    """A ``causal_mllm.seeds.git_working_tree_paths`` result."""
    return {"modified": list(modified), "untracked": list(untracked)}


def _patch_tree(monkeypatch, dirty=False, dirty_paths=(),
                untracked_paths=()) -> None:
    """Make the RUNNER's code-tree determination hermetic.

    Without this the fingerprint would shell out to the real repository, so
    a test's result would depend on whether the developer happened to have
    uncommitted work.
    """
    monkeypatch.setattr(
        "causal_mllm.replay.runner.code_tree_status",
        lambda exclude_prefixes=(): _tree(dirty, dirty_paths,
                                          untracked_paths))


# ---------------------------------------------------------------------
# P0-1  The panel digest must be over raw bytes
# ---------------------------------------------------------------------
class TestPanelHashIsRawBytes:
    def test_the_frozen_panel_hashes_to_the_pinned_digest(self):
        if not PANEL.exists():
            pytest.skip("frozen panel not present")
        assert _file_sha256(PANEL) == FROZEN_PANEL_SHA

    def test_normalizing_whitespace_yields_a_different_digest(self):
        # The defect: the same bytes, two digests, and only one of them is
        # comparable against the frozen protocol.
        if not PANEL.exists():
            pytest.skip("frozen panel not present")
        assert sha256_text(PANEL.read_text(encoding="utf-8")) \
            == NORMALIZED_PANEL_SHA
        assert NORMALIZED_PANEL_SHA != FROZEN_PANEL_SHA

    def test_the_frozen_protocol_records_the_raw_digest(self):
        if not PROTOCOL.exists():
            pytest.skip("frozen protocol not present")
        doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen = doc["frozen_inputs"]["panel_validated_families_sha256"]
        assert frozen == FROZEN_PANEL_SHA

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_committed_preflight_records_the_raw_digest(self, model_key):
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        dataset = artifact["dataset"]
        assert dataset["validated_families_sha256"] == FROZEN_PANEL_SHA, (
            "the artifact must carry the raw-byte digest the frozen "
            "protocol records, not a normalized one")
        assert dataset["matches_frozen_protocol"] is True
        assert dataset["hash_method"] == "sha256(raw file bytes)"

    def test_the_preflight_check_accepts_the_frozen_panel(self, tmp_path):
        if not PANEL.exists():
            pytest.skip("frozen panel not present")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        config = ReplayConfig(max_new_tokens=1536)
        values, problems = preflight.check_frozen_inputs(
            PANEL, protocol, config)
        assert values["validated_families_sha256"] == FROZEN_PANEL_SHA
        assert values["matches_frozen_protocol"] is True
        assert values["system_prompt_matches_frozen_protocol"] is True
        assert problems == []

    def test_the_preflight_check_rejects_an_edited_panel(self, tmp_path):
        if not PANEL.exists():
            pytest.skip("frozen panel not present")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        edited = tmp_path / "validated_families.jsonl"
        text = PANEL.read_text(encoding="utf-8")
        # One character changed: a panel that "looks like" the frozen one.
        edited.write_text(text.replace("fam", "fxm", 1), encoding="utf-8")
        values, problems = preflight.check_frozen_inputs(
            edited, protocol, ReplayConfig(max_new_tokens=1536))
        assert values["matches_frozen_protocol"] is False
        assert len(problems) == 1
        assert FROZEN_PANEL_SHA in problems[0]

    def test_the_preflight_check_rejects_a_missing_panel(self, tmp_path):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")) \
            if PROTOCOL.exists() else {
                "frozen_inputs": {
                    "panel_validated_families_sha256": FROZEN_PANEL_SHA,
                    "system_prompt_sha256": "x"},
                "uniform_cap_rule": {"initial_cap": 1536}}
        values, problems = preflight.check_frozen_inputs(
            tmp_path / "validated_families.jsonl", protocol,
            ReplayConfig(max_new_tokens=1536))
        assert values["validated_families_sha256"] is None
        assert any("not found" in p for p in problems)

    def test_the_preflight_check_rejects_a_non_frozen_cap(self):
        if not PROTOCOL.exists():
            pytest.skip("frozen protocol not present")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        # ReplayConfig's own default is 256, not the frozen 1536.
        assert ReplayConfig().max_new_tokens == 256
        _, problems = preflight.check_frozen_inputs(
            PANEL, protocol, ReplayConfig())
        assert any("uniform cap" in p for p in problems)


# ---------------------------------------------------------------------
# P0-2  Evidence must name the code that produced it
# ---------------------------------------------------------------------
class TestGitProvenance:
    def test_a_clean_tree_aborts_nothing_and_reports_nothing(self):
        result = preflight.git_provenance(COMMIT, _tree(False), False)
        assert result["abort"] is False
        assert result["problems"] == []

    def test_a_dirty_tree_aborts_before_any_gpu_work(self):
        result = preflight.git_provenance(
            COMMIT, _tree(True, dirty_paths=["src/causal_mllm/x.py"]), False)
        assert result["abort"] is True
        assert COMMIT in result["abort_message"]
        assert "src/causal_mllm/x.py" in result["abort_message"]
        assert result["problems"]

    def test_unknown_git_status_is_treated_as_dirty(self):
        # code_tree_status reports dirty=None outside a repository:
        # provenance that cannot be verified cannot certify evidence.
        result = preflight.git_provenance(None, _tree(None), False)
        assert result["abort"] is True
        assert result["problems"]

    def test_allow_dirty_runs_but_can_never_reach_pass(self):
        result = preflight.git_provenance(
            COMMIT, _tree(True, dirty_paths=["src/x.py"]), True)
        assert result["abort"] is False, "--allow-dirty must still run"
        assert result["problems"], \
            "but it must record a problem, and status is derived from " \
            "problems, so PASS stays unreachable"

    def test_the_status_derivation_makes_a_dirty_run_fail(self):
        # status is "PASS" iff problems is empty, so the recorded problem
        # is what enforces "no PASS from a dirty tree".
        result = preflight.git_provenance(
            COMMIT, _tree(True, dirty_paths=["src/x.py"]), True)
        problems = list(result["problems"])
        status = "PASS" if not problems else "FAIL"
        assert status == "FAIL"

    def test_a_stages_own_outputs_are_excluded_and_reported(self):
        # The defect this prevents: regenerating target 1's artifact made
        # the tree "dirty" and blocked targets 2-4, though no code changed.
        tree = _tree(False, own_outputs=[
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json",
            "outputs/iteration_11/preflight/resolved_models.lock.yaml"])
        result = preflight.git_provenance(COMMIT, tree, False)
        assert result["abort"] is False
        assert result["problems"] == []
        # Excluded paths are surfaced, never silently dropped.
        assert len(result["excluded_own_outputs"]) == 2

    def test_a_dirty_code_path_is_not_excused_by_the_exclusion(self):
        tree = _tree(True, dirty_paths=["src/causal_mllm/replay/runner.py"],
                     own_outputs=["outputs/iteration_11/preflight/x.json"])
        result = preflight.git_provenance(COMMIT, tree, False)
        assert result["abort"] is True
        assert result["dirty_paths"] == ["src/causal_mllm/replay/runner.py"]

    @pytest.mark.parametrize("untracked", [
        "src/causal_mllm/replay/injected.py",
        "scripts/iter11_model_preflight_v2.py",
        "sitecustomize.py",
    ])
    def test_untracked_source_blocks_the_preflight(self, untracked):
        # The hole reported: --untracked-files=no made untracked source
        # invisible, so an artifact could certify a tree its own recorded
        # code_commit did not contain.
        tree = _tree(True, dirty_paths=[untracked], untracked_paths=[untracked])
        result = preflight.git_provenance(COMMIT, tree, False)
        assert result["abort"] is True
        assert result["untracked_paths"] == [untracked]
        assert "UNTRACKED" in result["abort_message"]
        assert untracked in result["abort_message"]

    def test_untracked_source_is_a_problem_even_under_allow_dirty(self):
        # --allow-dirty may mint a diagnostic artifact, but it must never be
        # able to reach status PASS.
        tree = _tree(True, dirty_paths=["sitecustomize.py"],
                     untracked_paths=["sitecustomize.py"])
        result = preflight.git_provenance(COMMIT, tree, True)
        assert result["abort"] is False
        assert result["problems"]
        assert "sitecustomize.py" in result["problems"][0]


class TestCodeTreeStatus:
    """Untracked files must count; only explicit exclusions may not."""

    def test_own_output_prefixes_are_excluded_but_reported(self, monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: _git_paths(
            modified=["outputs/iteration_11/preflight/qwen35_2b/preflight.json",
                      "outputs/iteration_11/preflight/resolved_models.lock.yaml",
                      "src/causal_mllm/replay/runner.py"]))
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/preflight/",))
        assert status["dirty"] is True
        assert status["dirty_paths"] == ["src/causal_mllm/replay/runner.py"]
        assert len(status["excluded_own_outputs"]) == 2

    def test_only_own_outputs_dirty_means_clean(self, monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: _git_paths(
            modified=["outputs/iteration_11/preflight/qwen35_2b/preflight.json"
                      ]))
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/preflight/",))
        assert status["dirty"] is False
        assert status["dirty_paths"] == []
        assert status["excluded_own_outputs"] == [
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json"]

    @pytest.mark.parametrize("untracked", [
        "src/causal_mllm/replay/new_module.py",
        "scripts/iter11_model_preflight_patch.py",
        "sitecustomize.py",
        "usercustomize.py",
        "conftest.py",
        "yaml.py",
    ])
    def test_untracked_source_is_dirty(self, monkeypatch, untracked):
        # The hole this closes: git status --untracked-files=no reported a
        # clean tree while an untracked file changed what executed.
        monkeypatch.setattr(
            seeds, "git_working_tree_paths",
            lambda: _git_paths(untracked=[untracked]))
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/generations/",))
        assert status["dirty"] is True, untracked
        assert status["untracked_paths"] == [untracked]
        assert status["dirty_paths"] == [untracked]

    def test_untracked_files_inside_a_directory_are_individually_visible(
            self, monkeypatch):
        # --untracked-files=all expands directories, so a new module nested
        # in an untracked package cannot hide behind a directory entry.
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: _git_paths(
            untracked=["src/causal_mllm/shadow/__init__.py",
                       "src/causal_mllm/shadow/hooks.py"]))
        status = seeds.code_tree_status()
        assert status["dirty"] is True
        assert len(status["untracked_paths"]) == 2

    def test_untracked_evidence_under_the_exclusion_is_clean(self,
                                                             monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: _git_paths(
            untracked=["outputs/iteration_11/generations/qwen35_4b/run/"
                       "replay_outputs.jsonl"]))
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/generations/",))
        assert status["dirty"] is False
        assert status["excluded_own_outputs"]

    @pytest.mark.parametrize("cache_path", [
        "src/causal_mllm/replay/__pycache__/runner.cpython-310.pyc",
        ".pytest_cache/v/cache/lastfailed",
        ".ruff_cache/0.2.0/CACHEDIR.TAG",
        "outputs/iteration_11/generations/run.log",
    ])
    def test_cache_paths_are_excluded_and_reported(self, monkeypatch,
                                                   cache_path):
        monkeypatch.setattr(
            seeds, "git_working_tree_paths",
            lambda: _git_paths(untracked=[cache_path]))
        status = seeds.code_tree_status()
        assert status["dirty"] is False
        assert status["excluded_cache_paths"] == [cache_path]

    def test_a_cache_exclusion_does_not_excuse_source(self, monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: _git_paths(
            untracked=["src/causal_mllm/__pycache__/x.pyc",
                       "src/causal_mllm/replay/injected.py"]))
        status = seeds.code_tree_status()
        assert status["dirty"] is True
        assert status["dirty_paths"] == ["src/causal_mllm/replay/injected.py"]
        assert status["excluded_cache_paths"] == [
            "src/causal_mllm/__pycache__/x.pyc"]

    def test_an_unavailable_git_status_is_none(self, monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: None)
        status = seeds.code_tree_status(exclude_prefixes=("outputs/",))
        assert status["dirty"] is None
        assert status["dirty_paths"] == []
        assert status["untracked_paths"] == []

    def test_no_exclusions_matches_the_unscoped_answer(self, monkeypatch):
        paths = ["README.md", "src/causal_mllm/seeds.py"]
        monkeypatch.setattr(
            seeds, "git_working_tree_paths", lambda: _git_paths(modified=paths))
        status = seeds.code_tree_status()
        assert status["dirty"] is True
        assert status["dirty_paths"] == paths
        assert status["excluded_own_outputs"] == []

    def test_the_live_repository_reports_untracked_files(self):
        """Not mocked: an untracked file really does make the tree dirty.

        Creates and removes a probe file inside the repository so the whole
        chain (git porcelain -> code_tree_status) is exercised against real
        git output rather than a fabricated dict.
        """
        probe = ROOT / "src" / "causal_mllm" / "_untracked_probe_.py"
        if probe.exists():  # pragma: no cover - stale probe from a crash
            probe.unlink()
        try:
            before = seeds.code_tree_status()
            probe.write_text("PROBE = True\n", encoding="utf-8")
            after = seeds.code_tree_status()
            assert after["dirty"] is True
            assert "src/causal_mllm/_untracked_probe_.py" \
                in after["untracked_paths"]
            assert after["dirty_paths"] != before["dirty_paths"]
        finally:
            probe.unlink(missing_ok=True)
        assert "src/causal_mllm/_untracked_probe_.py" \
            not in seeds.code_tree_status()["untracked_paths"]

    def test_dirty_tracked_files_still_ignores_untracked(self, monkeypatch):
        # The narrow tracked-only answer is retained for provenance that
        # RECORDS rather than gates.
        monkeypatch.setattr(
            seeds, "git_working_tree_paths",
            lambda: _git_paths(modified=["a.py"], untracked=["b.py"]))
        assert seeds.dirty_tracked_files() == ["a.py"]

    def test_provenance_is_anchored_to_the_repository_not_the_cwd(
            self, monkeypatch, tmp_path):
        # Launching a stage from outside the repository used to record
        # code_commit=None while the (already anchored) tree determination
        # described the repository correctly: two answers to one question,
        # and the gate refused on the first while trusting the second.
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True).stdout.strip()
        assert expected, "the repository HEAD should be resolvable"
        assert seeds.get_git_commit() == expected
        monkeypatch.chdir(tmp_path)  # outside the repository
        assert seeds.get_git_commit() == expected
        assert isinstance(seeds.is_git_dirty(), bool)
        assert isinstance(seeds.code_tree_status()["dirty"], bool)


class TestCommittedArtifactProvenance:
    """The committed artifacts, checked against the P0-2 invariants."""

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_committed_artifacts_record_git_dirty(self, model_key):
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert "git_dirty" in artifact, \
            "code_commit alone does not identify the code that ran"
        assert artifact["git_dirty"] is False, \
            "eligible evidence must come from a clean tree"

    @pytest.mark.parametrize("model_key,adapter_file", [
        ("phi4_mm", "src/causal_mllm/replay/adapters/phi4_multimodal.py"),
        ("ministral3_3b", "src/causal_mllm/replay/adapters/ministral3.py"),
        ("qwen35_2b", "src/causal_mllm/replay/adapters/qwen35.py"),
        ("qwen35_4b", "src/causal_mllm/replay/adapters/qwen35.py"),
    ])
    def test_the_recorded_commit_contains_the_adapter_it_certifies(
            self, model_key, adapter_file):
        """The P0-2 defect, stated as an invariant.

        Every artifact recorded code_commit=64f96ca (11.3) while the
        Phi-4 adapter only exists from 541cb5e, so that commit could not
        reconstruct the run. Whatever commit an artifact names MUST
        contain the code that produced it.
        """
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        commit = artifact["code_commit"]
        assert commit, "artifact records no code_commit"
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{adapter_file}"],
            cwd=ROOT, capture_output=True, text=True)
        if probe.returncode != 0 and "not a git repository" in \
                (probe.stderr or "").lower():
            pytest.skip("not a git repository")
        assert probe.returncode == 0, (
            f"{model_key}: code_commit {commit[:12]} does not contain "
            f"{adapter_file}, so it cannot reconstruct the run it "
            f"certifies (stderr: {probe.stderr.strip()[:200]})")

    def test_the_run_fingerprint_moves_when_the_tree_is_dirty(
            self, tmp_path, monkeypatch):
        # The reviewer's point: with only code_commit bound, uncommitted
        # edits did not change the fingerprint, so a dirty run could
        # resume into a clean one.
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        _patch_tree(monkeypatch)
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: False)
        clean = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw)
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: True)
        dirty = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw)
        assert clean != dirty

    def test_the_fingerprint_distinguishes_unknown_from_clean(
            self, tmp_path, monkeypatch):
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        _patch_tree(monkeypatch)
        digests = set()
        for value in (False, True, None):
            monkeypatch.setattr(
                "causal_mllm.replay.runner.is_git_dirty", lambda v=value: v)
            digests.add(iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw))
        assert len(digests) == 3

    def test_untracked_source_moves_the_fingerprint(self, tmp_path,
                                                    monkeypatch):
        # The recording-layer half of the same defect. ``is_git_dirty()``
        # counts only tracked modifications, so an untracked module left
        # git_dirty False and two runs whose code differed shared a
        # fingerprint — and could therefore resume into each other.
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: False)
        _patch_tree(monkeypatch)
        clean = iteration11_run_fingerprint(stub, config, tmp_path, spec, hw)
        # Tracked tree still clean; only an untracked file appeared.
        _patch_tree(monkeypatch, dirty=True,
                    dirty_paths=["src/causal_mllm/replay/injected.py"],
                    untracked_paths=["src/causal_mllm/replay/injected.py"])
        injected = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw)
        assert clean != injected, (
            "an untracked module changed the code that would execute "
            "without moving the fingerprint")

    def test_the_fingerprint_scopes_out_the_runs_own_outputs(
            self, tmp_path, monkeypatch):
        # A run's own outputs are its product, not its code: they must not
        # move the fingerprint, or a resume would invalidate itself.
        seen = {}

        def fake(exclude_prefixes=()):
            seen["prefixes"] = tuple(exclude_prefixes)
            return _tree(False, own_outputs=[
                "outputs/iteration_11/generations/qwen35_4b/r/"
                "replay_outputs.jsonl"])

        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: False)
        monkeypatch.setattr(
            "causal_mllm.replay.runner.code_tree_status", fake)
        with_own_outputs = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw)
        _patch_tree(monkeypatch)
        pristine = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw)
        assert with_own_outputs == pristine
        assert seen["prefixes"] == confirmatory.OWN_OUTPUT_PREFIXES

    def test_the_run_report_records_the_untracked_paths(self, tmp_path,
                                                        monkeypatch):
        # So a report can never claim a clean tree while untracked source
        # was present, even on a path the gate does not cover.
        _families(tmp_path, 3)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: False)
        _patch_tree(monkeypatch, dirty=True,
                    dirty_paths=["sitecustomize.py"],
                    untracked_paths=["sitecustomize.py"])
        report = runner.run_replay_stage(
            tmp_path, tmp_path / "out", config=config, backend=stub,
            run_id="iter11-untracked", model_spec=spec)
        # Iteration 11 provenance is merged into the report's provenance
        # block, so the two dirtiness answers sit side by side.
        provenance = report["provenance"]
        assert provenance["git_dirty"] is False
        assert provenance["code_tree_dirty"] is True
        assert provenance["code_untracked_paths"] == ["sitecustomize.py"]
        assert provenance["code_dirty_paths"] == ["sitecustomize.py"]

    def test_the_legacy_report_schema_is_untouched(self, tmp_path,
                                                   monkeypatch):
        # The frozen single-model path must not gain Iteration 11 fields.
        _families(tmp_path, 2)
        config = ReplayConfig(max_new_tokens=64)
        stub = _StubAdapter(_spec())
        monkeypatch.setattr(
            "causal_mllm.replay.runner.is_git_dirty", lambda: False)
        _patch_tree(monkeypatch)
        # No model_spec: the legacy path, whose report schema is frozen.
        report = runner.run_replay_stage(
            tmp_path, tmp_path / "legacy", config=config, backend=stub,
            run_id="legacy-run")
        assert "code_tree_dirty" not in report["provenance"]
        assert "code_untracked_paths" not in report["provenance"]
        assert "model_key" not in report["provenance"]
        assert report["provenance"]["git_dirty"] is False


# ---------------------------------------------------------------------
# P0-3  The confirmatory gate
# ---------------------------------------------------------------------
def _write_panel(path: Path, n_families: int = 100,
                 variants=ALL_VARIANT_NAMES) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_families):
        rec = {"family_id": f"fam{i:03d}",
               "variants": {v: {"messages": []} for v in variants}}
        lines.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _file_sha256(path)


def _write_protocol(path: Path, panel_sha: str, *, n_families: int = 100,
                    cap: int = 1536) -> dict:
    doc = {
        "frozen_inputs": {
            "panel_validated_families_sha256": panel_sha,
            "n_families": n_families,
            "n_variants_per_family": 6,
            "variants": list(ALL_VARIANT_NAMES),
            "prompt_template_revision": "v1",
            "effective_decoding": {
                "do_sample": False, "temperature": None, "top_p": None,
                "top_k": None, "num_beams": 1, "max_new_tokens": cap},
        },
        "uniform_cap_rule": {"initial_cap": cap},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def _write_lock(path: Path, dependency: dict, *, revision=REVISION,
                processor_revision=PROCESSOR_REVISION,
                model_key="qwen35_4b") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "models": {model_key: {
            "revision": revision,
            "processor_revision": processor_revision}},
        "dependency_lock": dependency,
    }, sort_keys=True), encoding="utf-8")


def _eligibility_ids(n=ELIGIBILITY_N_FAMILIES):
    """The first ``n`` family ids of the synthetic panel."""
    return [f"fam{i:03d}" for i in range(n)]


def _synthetic_selection(ids=None) -> dict:
    """Stand-in for ``selection.derive_frozen_selection``.

    The real derivation reads the frozen 9B run and the frozen adjudicated
    labels, which the synthetic world does not contain. What the gate does
    with the result is identical, so tests inject the same shape and the
    external-pre-registration checks are exercised for real.
    """
    chosen = sorted(_eligibility_ids() if ids is None else ids)
    return {"selected_family_ids": chosen,
            "selected_families_sha256": selected_families_sha256(chosen),
            "n_selected_families": len(chosen)}


def _write_eligibility(root: Path, model_key: str, protocol_path: Path, *,
                       lock_sha=None, model_id="Qwen/Qwen3.5-4B",
                       status="PASS", eligible=True, revision=REVISION,
                       processor_revision=PROCESSOR_REVISION,
                       code_commit=COMMIT, git_dirty=False,
                       git_dirty_code_paths=(),
                       family_ids=None, overrides=None) -> Path:
    """A schema-conformant 11.5 report; perturb it with ``overrides``.

    ``overrides`` is applied last and may set a field to None, which is how
    the "required field missing" cases are produced.
    """
    ids = list(_eligibility_ids() if family_ids is None else family_ids)
    doc = {
        "status": status,
        "eligible": eligible,
        "model_key": model_key,
        "model_id": model_id,
        "model_revision": revision,
        "processor_revision": processor_revision,
        "code_commit": code_commit,
        "git_dirty": git_dirty,
        "git_dirty_code_paths": list(git_dirty_code_paths),
        "protocol_sha256": protocol_sha256(protocol_path),
        "dependency_lock_sha256": lock_sha,
        "selected_family_ids": ids,
        "selected_families_sha256": selected_families_sha256(ids),
        "n_selected_families": len(ids),
        "variants": list(ALL_VARIANT_NAMES),
        "n_expected_attempts": ELIGIBILITY_N_ATTEMPTS,
        "n_attempts": ELIGIBILITY_N_ATTEMPTS,
        "n_succeeded": ELIGIBILITY_N_ATTEMPTS,
        "truncation_by_variant": {
            variant: {"n": ELIGIBILITY_N_FAMILIES, "n_truncated": 0,
                      "truncation_rate": 0.0}
            for variant in ALL_VARIANT_NAMES},
        # The EXACT required gate set, each carrying its evidence. A bare
        # ``passed: true`` is not accepted: it does not show the check ran.
        "gates": {
            "generations_complete": {
                "passed": True,
                "n_attempts": ELIGIBILITY_N_ATTEMPTS,
                "n_succeeded": ELIGIBILITY_N_ATTEMPTS,
                "n_failed": 0},
            "truncation_reviewed": {
                "passed": True, "n_truncated": 0, "truncation_rate": 0.0,
                "max_variant_spread": 0.0},
            "vision_path_engaged": {
                "passed": True, "n_image_bearing_cells": 24,
                "min_image_token_count": 81},
            "terminal_query_invariant": {
                "passed": True,
                "n_families_checked": ELIGIBILITY_N_FAMILIES,
                "n_mismatched": 0},
            "revision_pinned": {
                "passed": True, "model_revision": revision,
                "processor_revision": processor_revision},
            "determinism": {
                "passed": True, "n_repeats": 2, "n_distinct_responses": 1},
        },
    }
    if overrides:
        doc.update(overrides)
    path = root / model_key / "preflight_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def frozen_env(monkeypatch):
    """A deterministic ``pip freeze`` so dependency identity is hermetic.

    Non-freeze subprocess calls (git) are delegated to the real
    implementation, so patching this does not disturb anything else.
    """
    real_run = subprocess.run
    freeze_text = "numpy==1.26.0\ntransformers==5.14.1\npeft==0.20.0\n"

    class _Done:
        returncode = 0
        stdout = freeze_text
        stderr = ""

    def fake(cmd, *args, **kwargs):
        if "freeze" in [str(part) for part in cmd]:
            return _Done()
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(registry.subprocess, "run", fake)
    return dependency_lock_snapshot()


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(confirmatory, "get_git_commit", lambda: COMMIT)
    monkeypatch.setattr(confirmatory, "code_tree_status",
                        lambda exclude_prefixes=(): _tree(False))


@pytest.fixture
def world(tmp_path, frozen_env, clean_tree):
    """A fully conformant confirmatory world; mutate one part per test."""
    panel = tmp_path / "panel" / "validated_families.jsonl"
    panel_sha = _write_panel(panel)
    protocol_path = tmp_path / "iteration_11_protocol.json"
    _write_protocol(protocol_path, panel_sha)
    lock_path = tmp_path / "resolved_models.lock.yaml"
    _write_lock(lock_path, frozen_env)
    eligibility_root = tmp_path / "eligibility"
    _write_eligibility(eligibility_root, "qwen35_4b", protocol_path,
                       lock_sha=dependency_lock_sha256(lock_path))
    spec = ResolvedModel(
        model_key="qwen35_4b", model_id="Qwen/Qwen3.5-4B", adapter="qwen35",
        revision=REVISION, revision_source="lock")
    config = ReplayConfig(model_name=spec.model_id, model_revision=REVISION,
                          max_new_tokens=1536)
    return {
        "tmp_path": tmp_path, "panel": panel, "panel_sha": panel_sha,
        "input_dir": panel.parent, "protocol_path": protocol_path,
        "lock_path": lock_path, "eligibility_root": eligibility_root,
        "spec": spec, "config": config, "dependency": frozen_env,
        "expected_selection": _synthetic_selection(),
    }


def _gate(world, **overrides):
    kwargs = {
        "input_dir": world["input_dir"],
        "config": world["config"],
        "model_spec": world["spec"],
        "lock_path": world["lock_path"],
        "protocol_path": world["protocol_path"],
        "eligibility_root": world["eligibility_root"],
        "output_root": f"{GENERATIONS_ROOT}/{world['spec'].model_key}",
        # Injected rather than derived: the synthetic world has no frozen 9B
        # run to derive from. The gate treats this exactly as it treats a
        # derivation, which is what makes the test meaningful.
        "expected_selection": world["expected_selection"],
    }
    kwargs.update(overrides)
    return enforce_confirmatory_protocol(**kwargs)


def _violations(world, **overrides) -> list[str]:
    with pytest.raises(ReplayError) as exc:
        _gate(world, **overrides)
    return str(exc.value).splitlines()


def _eligibility(world, **kwargs) -> Path:
    """Rewrite the world's 11.5 report with exactly one perturbation.

    Supplies the lock digest automatically so a test that perturbs, say,
    ``status`` does not also trip the dependency-lock requirement and mask
    the thing it meant to check.
    """
    kwargs.setdefault("lock_sha", dependency_lock_sha256(world["lock_path"]))
    return _write_eligibility(world["eligibility_root"],
                              world["spec"].model_key,
                              world["protocol_path"], **kwargs)


class TestConfirmatoryGate:
    def test_a_conformant_run_passes(self, world):
        result = _gate(world)
        assert result["passed"] is True
        assert result["violations"] == []
        assert result["checks"]["panel_sha256"] == world["panel_sha"]
        assert result["checks"]["git_dirty"] is False
        assert result["checks"]["max_new_tokens"] == 1536

    def test_the_gate_refuses_the_legacy_single_model_path(self, world):
        with pytest.raises(ReplayError, match="requires a resolved"):
            _gate(world, model_spec=None)

    # --- panel identity -------------------------------------------------
    def test_a_different_panel_is_rejected(self, world):
        other = world["tmp_path"] / "other" / "validated_families.jsonl"
        _write_panel(other, n_families=100)
        other.write_text(
            other.read_text(encoding="utf-8").replace("fam", "fxm", 1),
            encoding="utf-8")
        lines = _violations(world, input_dir=other.parent)
        assert any("not the frozen" in line for line in lines)

    def test_a_missing_panel_is_rejected(self, world):
        lines = _violations(world, input_dir=world["tmp_path"] / "absent")
        assert any("not found" in line for line in lines)

    def test_a_truncated_panel_is_rejected(self, world):
        small = world["tmp_path"] / "small" / "validated_families.jsonl"
        sha = _write_panel(small, n_families=12)
        protocol = world["tmp_path"] / "small_protocol.json"
        _write_protocol(protocol, sha, n_families=100)
        # The panel matches ITS OWN protocol hash but holds 12 families:
        # the smoke panel must not pass as the confirmatory one.
        lines = _violations(world, input_dir=small.parent,
                            protocol_path=protocol)
        assert any("requires exactly 100" in line for line in lines)

    def test_a_family_missing_a_variant_is_rejected(self, world):
        panel = world["tmp_path"] / "broken" / "validated_families.jsonl"
        _write_panel(panel, n_families=100)
        records = [json.loads(line) for line in
                   panel.read_text(encoding="utf-8").splitlines()]
        del records[7]["variants"]["cross_modal"]
        panel.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        protocol = world["tmp_path"] / "broken_protocol.json"
        _write_protocol(protocol, _file_sha256(panel))
        lines = _violations(world, input_dir=panel.parent,
                            protocol_path=protocol)
        assert any("six frozen variants" in line for line in lines)

    def test_an_undeclared_variant_is_rejected(self, world):
        panel = world["tmp_path"] / "extra" / "validated_families.jsonl"
        _write_panel(panel, n_families=100)
        records = [json.loads(line) for line in
                   panel.read_text(encoding="utf-8").splitlines()]
        records[0]["variants"]["seventh_condition"] = {"messages": []}
        panel.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        protocol = world["tmp_path"] / "extra_protocol.json"
        _write_protocol(protocol, _file_sha256(panel))
        lines = _violations(world, input_dir=panel.parent,
                            protocol_path=protocol)
        assert any("outside the" in line for line in lines)

    # --- decoding and cap ----------------------------------------------
    def test_the_default_cap_is_rejected(self, world):
        # ReplayConfig defaults to 256: omitting --max-new-tokens would
        # silently truncate every response at a sixth of the frozen cap.
        lines = _violations(world, config=ReplayConfig(
            model_name=world["spec"].model_id, model_revision=REVISION))
        assert any("uniform cap 1536" in line for line in lines)

    def test_sampling_is_rejected(self, world):
        config = ReplayConfig(max_new_tokens=1536, do_sample=True,
                              temperature=0.8, top_p=0.95)
        lines = _violations(world, config=config)
        assert any("greedy decoding" in line for line in lines)

    def test_inert_sampling_values_are_accepted(self, world):
        # The protocol normalizes temperature/top_p to null but records
        # Iteration 10's 0.0/1.0; both describe greedy decoding, so neither
        # spelling may be rejected.
        for temperature, top_p in ((0.0, 1.0), (None, None)):
            config = ReplayConfig(max_new_tokens=1536, do_sample=False,
                                  temperature=temperature, top_p=top_p)
            assert _gate(world, config=config)["passed"] is True

    def test_thinking_enabled_is_rejected(self, world):
        config = ReplayConfig(max_new_tokens=1536, enable_thinking=True)
        lines = _violations(world, config=config)
        assert any("thinking disabled" in line for line in lines)

    def test_a_moved_prompt_template_revision_is_rejected(self, world):
        config = ReplayConfig(max_new_tokens=1536,
                              prompt_template_revision="v2")
        lines = _violations(world, config=config)
        assert any("prompt_template_revision" in line for line in lines)

    # --- tree state ----------------------------------------------------
    def test_a_dirty_tree_is_rejected(self, world, monkeypatch):
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(
                True, dirty_paths=["src/causal_mllm/replay/runner.py"]))
        lines = _violations(world)
        assert any("working tree is not clean" in line for line in lines)
        assert any("src/causal_mllm/replay/runner.py" in line
                   for line in lines)

    def test_an_unknown_tree_state_is_rejected(self, world, monkeypatch):
        monkeypatch.setattr(confirmatory, "code_tree_status",
                            lambda exclude_prefixes=(): _tree(None))
        lines = _violations(world)
        assert any("git tree status unknown" in line for line in lines)

    def test_the_gate_excludes_only_its_own_output_tree(self, world,
                                                        monkeypatch):
        # A run must not be blocked by evidence it is itself regenerating,
        # and the exclusion must be narrow and visible.
        seen = {}

        def fake(exclude_prefixes=()):
            seen["prefixes"] = tuple(exclude_prefixes)
            return _tree(False, own_outputs=[
                "outputs/iteration_11/generations/qwen35_4b/r/replay_outputs"
                ".jsonl"])

        monkeypatch.setattr(confirmatory, "code_tree_status", fake)
        result = _gate(world)
        assert result["passed"] is True
        assert seen["prefixes"] == confirmatory.OWN_OUTPUT_PREFIXES
        assert result["checks"]["git_dirty_excluded_own_outputs"]

    @pytest.mark.parametrize("untracked", [
        "src/causal_mllm/replay/injected.py",
        "scripts/iter11_model_preflight_v2.py",
        "sitecustomize.py",
    ])
    def test_untracked_source_fails_the_confirmatory_gate(self, world,
                                                          monkeypatch,
                                                          untracked):
        # End-to-end through the real gate: an untracked file must stop a
        # confirmatory run even though nothing tracked changed.
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(
                True, dirty_paths=[untracked], untracked_paths=[untracked]))
        lines = _violations(world)
        assert any("working tree is not clean" in line and untracked in line
                   for line in lines)

    def test_the_gate_records_the_untracked_paths_it_refused(self,
                                                            monkeypatch):
        gate = confirmatory._Gate()
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(
                True, dirty_paths=["sitecustomize.py"],
                untracked_paths=["sitecustomize.py"]))
        confirmatory._check_clean_tree(gate)
        assert gate.checks["git_dirty"] is True
        assert gate.checks["git_untracked_paths"] == ["sitecustomize.py"]
        assert len(gate.violations) == 1

    def test_untracked_evidence_under_the_exclusion_passes(self, world,
                                                           monkeypatch):
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(
                False, own_outputs=[
                    "outputs/iteration_11/generations/qwen35_4b/r/"
                    "replay_outputs.jsonl"]))
        result = _gate(world)
        assert result["passed"] is True
        assert result["checks"]["git_untracked_paths"] == []
        assert result["checks"]["git_dirty_excluded_own_outputs"]

    def test_a_missing_commit_is_rejected(self, world, monkeypatch):
        monkeypatch.setattr(confirmatory, "get_git_commit", lambda: None)
        lines = _violations(world)
        assert any("no git commit resolvable" in line for line in lines)

    # --- revisions -----------------------------------------------------
    def test_a_floating_model_revision_is_rejected(self, world):
        spec = ResolvedModel(
            model_key="qwen35_4b", model_id="Qwen/Qwen3.5-4B",
            adapter="qwen35", revision="main")
        lines = _violations(world, model_spec=spec)
        assert any("not an immutable 40-hex" in line for line in lines)

    def test_a_floating_processor_revision_is_rejected(self, world):
        _write_lock(world["lock_path"], world["dependency"],
                    processor_revision="main")
        lines = _violations(world)
        assert any("processor revision" in line for line in lines)

    def test_a_model_absent_from_the_lock_is_rejected(self, world):
        _write_lock(world["lock_path"], world["dependency"],
                    model_key="ministral3_3b")
        lines = _violations(world)
        assert any("not present in the lock" in line for line in lines)

    def test_a_lock_revision_disagreeing_with_the_spec_is_rejected(
            self, world):
        _write_lock(world["lock_path"], world["dependency"],
                    revision="d" * 40)
        lines = _violations(world)
        assert any("!= locked revision" in line for line in lines)

    def test_quantization_is_rejected(self, world):
        spec = ResolvedModel(
            model_key="qwen35_4b", model_id="Qwen/Qwen3.5-4B",
            adapter="qwen35", revision=REVISION, quantization="int4")
        lines = _violations(world, model_spec=spec)
        assert any("quantization" in line for line in lines)

    # --- dependency environment ---------------------------------------
    def test_dependency_drift_is_rejected(self, world, monkeypatch):
        # A transformers upgrade after the lock was recorded: the run would
        # execute under a different environment than the certified one.
        drifted = dict(world["dependency"],
                       pip_freeze_sha256="f" * 64)
        monkeypatch.setattr(
            registry, "dependency_lock_snapshot", lambda: drifted)
        lines = _violations(world)
        assert any("active dependency environment differs" in line
                   for line in lines)

    def test_a_python_version_change_is_rejected(self, world, monkeypatch):
        drifted = dict(world["dependency"], python_version="3.11.0")
        monkeypatch.setattr(
            registry, "dependency_lock_snapshot", lambda: drifted)
        lines = _violations(world)
        assert any("python_version" in line for line in lines)

    def test_a_different_interpreter_path_is_not_a_violation(
            self, world, monkeypatch):
        # P1-5: the executable is operational metadata. Identical packages
        # under a different absolute interpreter path is the same
        # environment, and rejecting it would make the lock unportable.
        drifted = dict(world["dependency"],
                       executable="/opt/other/env/bin/python")
        monkeypatch.setattr(
            registry, "dependency_lock_snapshot", lambda: drifted)
        assert _gate(world)["passed"] is True

    # --- 11.5 eligibility ---------------------------------------------
    def test_a_missing_eligibility_report_is_rejected(self, world):
        lines = _violations(world,
                            eligibility_root=world["tmp_path"] / "none")
        assert any("no 11.5 eligibility report" in line for line in lines)

    def test_a_failing_eligibility_report_is_rejected(self, world):
        _eligibility(world, status="FAIL")
        lines = _violations(world)
        assert any("not 'PASS'" in line for line in lines)

    def test_eligible_false_is_rejected(self, world):
        _eligibility(world, eligible=False)
        lines = _violations(world)
        assert any("eligible=true" in line for line in lines)

    def test_eligibility_for_another_revision_does_not_transfer(self, world):
        _eligibility(world, revision="e" * 40)
        lines = _violations(world)
        assert any("does not transfer across revisions" in line
                   for line in lines)

    def test_eligibility_for_another_protocol_is_rejected(self, world):
        path = world["eligibility_root"] / "qwen35_4b" \
            / "preflight_report.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["protocol_sha256"] = "0" * 64
        path.write_text(json.dumps(doc), encoding="utf-8")
        lines = _violations(world)
        assert any("not the current frozen protocol" in line
                   for line in lines)

    def test_eligibility_from_dirty_code_is_rejected(self, world):
        _eligibility(world, git_dirty=True,
                     git_dirty_code_paths=["src/causal_mllm/replay/runner.py"])
        lines = _violations(world)
        assert any("execution-relevant code" in line for line in lines), lines
        # The offending path and the commit it contradicts are both named, so
        # the violation is actionable rather than merely a refusal.
        assert any("runner.py" in line for line in lines), lines

    def test_a_report_omitting_the_code_paths_is_rejected(self, world):
        """The field that decides the question cannot be left out.

        Omitting it trips the required-field refusal before the code-scoped
        check runs, which is the stronger outcome: a report cannot be silent
        about whether its code changed mid-run and still be trusted.
        """
        assert "git_dirty_code_paths" in ELIGIBILITY_REQUIRED_FIELDS
        _eligibility(world, overrides={"git_dirty_code_paths": None})
        lines = _violations(world)
        assert any("missing required field" in line for line in lines), lines
        assert any("git_dirty_code_paths" in line for line in lines), lines

    def test_non_code_changes_mid_run_do_not_invalidate_the_evidence(self,
                                                                    world):
        """The whole tree is still RECORDED, but only code is fatal.

        The tree is required clean at launch, so the process has imported its
        code and ``code_commit`` pins it. A diagnostic file appearing under
        ``outputs/`` afterwards cannot retroactively change what was
        generated; failing on it discarded a valid 72-generation run whose six
        gates all passed.
        """
        _eligibility(world, git_dirty=True, git_dirty_code_paths=[],
                     overrides={
                         "git_dirty_paths": [
                             "outputs/iteration_11/judge_vision_ablation/"
                             "ablated_labels.jsonl"]})
        # The gate now PASSES, so there are no violations to collect.
        result = _gate(world)
        assert result["passed"] is True, result["violations"]
        # Recorded, not forgiven: the report still says the tree was dirty.
        path = (world["eligibility_root"] / world["spec"].model_key
                / "preflight_report.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["git_dirty"] is True
        assert doc["git_dirty_code_paths"] == []

    @pytest.mark.parametrize("path", [
        "src/causal_mllm/replay/runner.py",
        "scripts/iter11_run_eligibility.py",
        "configs/experiments/iteration_11_protocol.json",
        "tests/unit/test_iter11_evidence_integrity.py",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "conftest.py",
        "sitecustomize.py",
        "helper.py",
    ])
    def test_execution_relevant_paths_are_recognised(self, path):
        assert seeds.is_execution_relevant_path(path), path

    @pytest.mark.parametrize("path", [
        # Frozen inputs are read at runtime even though they live under
        # outputs/: the 11.5 selection is DERIVED from the Iteration-10 panel,
        # reference run and adjudicated labels, and the gate reads the frozen
        # protocol and the resolved-model lock.
        "outputs/iteration_11/protocol/iteration_11_protocol.json",
        "outputs/iteration_11/preflight/resolved_models.lock.yaml",
        "outputs/scale_c/families_panel/validated_families.jsonl",
        "outputs/scale_c/llm_judge_artifacts/llm_labels_adjudicated.json",
    ])
    def test_frozen_inputs_under_outputs_are_execution_relevant(self, path):
        assert seeds.is_execution_relevant_path(path), path

    @pytest.mark.parametrize("path", [
        "outputs/iteration_11/judge_vision_ablation/ablated_labels.jsonl",
        "outputs/iteration_11/eligibility/qwen35_2b/preflight_report.json",
        "outputs/iteration_11/analysis/summary.json",
        "outputs/iteration_11/generations/qwen35_2b/run/replay_outputs.jsonl",
        "README.md",
        "docs/design.md",
        "data/media/source/mtmcs_0_main.png",
        "figures/panel.pdf",
        "notes.txt",
    ])
    def test_generated_evidence_is_not_execution_relevant(self, path):
        assert not seeds.is_execution_relevant_path(path), path

    def test_the_classifier_is_fail_closed_for_a_new_directory(self):
        """A path nobody anticipated counts as relevant, not as evidence.

        The scoping must not become a way for an unrecognised file to excuse
        itself: relevance is the default and only the declared evidence, media
        and prose shapes are forgiven.
        """
        assert seeds.is_execution_relevant_path("brand_new_tool/runner.py")
        assert seeds.is_execution_relevant_path("brand_new_tool/notes")
        assert seeds.is_execution_relevant_path("vendor/thing.bin")
        # A new OUTPUT directory is still evidence, because the forgiveness is
        # by shape (generated tree) rather than by an enumerated allowlist of
        # known iteration directories.
        assert not seeds.is_execution_relevant_path(
            "outputs/iteration_12/something_new.json")

    def test_code_tree_status_reports_the_code_subset(self, monkeypatch):
        monkeypatch.setattr(
            seeds, "git_working_tree_paths",
            lambda: _git_paths(
                modified=["README.md"],
                untracked=["src/causal_mllm/replay/new_module.py",
                           "outputs/iteration_11/eligibility/x.json"]))
        tree = seeds.code_tree_status()
        assert tree["dirty"] is True
        assert tree["dirty_paths"] == [
            "README.md", "outputs/iteration_11/eligibility/x.json",
            "src/causal_mllm/replay/new_module.py"]
        assert tree["code_dirty_paths"] == [
            "src/causal_mllm/replay/new_module.py"]

    def test_code_tree_status_without_git_reports_an_empty_code_subset(self,
                                                                       monkeypatch):
        monkeypatch.setattr(seeds, "git_working_tree_paths", lambda: None)
        tree = seeds.code_tree_status()
        assert tree["dirty"] is None
        assert tree["code_dirty_paths"] == []

    # --- scope and overwrite ------------------------------------------
    def test_overwrite_is_rejected(self, world):
        lines = _violations(world, overwrite=True)
        assert any("overwrite=True" in line for line in lines)

    def test_max_families_is_rejected(self, world):
        lines = _violations(world, max_families=12)
        assert any("ENTIRE frozen panel" in line for line in lines)

    def test_a_redirected_output_root_is_rejected(self, world):
        lines = _violations(world, output_root="/tmp/somewhere")
        assert any("tracked Iteration 11 generations tree" in line
                   for line in lines)

    def test_the_default_output_root_is_accepted(self, world):
        assert _gate(world, output_root=None)["passed"] is True

    # --- aggregation ---------------------------------------------------
    def test_every_violation_is_reported_together(self, world, monkeypatch):
        # One launch must surface every problem, not one per attempt.
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(
                True, dirty_paths=["src/causal_mllm/replay/runner.py"]))
        with pytest.raises(ReplayError) as exc:
            _gate(world, overwrite=True, max_families=5,
                  config=ReplayConfig(max_new_tokens=256,
                                      enable_thinking=True),
                  eligibility_root=world["tmp_path"] / "none")
        message = str(exc.value)
        assert "6 violation(s)" in message
        for fragment in ("overwrite=True", "working tree is not clean",
                         "max_families=5", "uniform cap 1536",
                         "thinking disabled", "no 11.5 eligibility report"):
            assert fragment in message

    def test_the_gate_evidence_is_persistable(self, world):
        result = _gate(world)
        # It must survive a JSON round trip: the CLI stores it in the run
        # report so a PASS is auditable from the evidence itself.
        assert json.loads(json.dumps(result))["passed"] is True
        assert "protocol_sha256" in result["checks"]

    def test_the_real_frozen_panel_satisfies_the_panel_checks(self):
        """End-to-end against the committed protocol and panel.

        Synthetic fixtures prove the gate's logic; this proves the REAL
        frozen panel is what the gate accepts — 100 families, all six
        variants each, hashing to the pinned raw digest.
        """
        if not PANEL.exists() or not PROTOCOL.exists():
            pytest.skip("frozen panel/protocol not present")
        doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in
                   PANEL.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        gate = confirmatory._Gate()
        panel = confirmatory._check_panel_identity(
            gate, PANEL.parent, doc)
        assert panel is not None
        confirmatory._check_family_coverage(gate, records, doc)
        assert gate.violations == []
        assert gate.checks["n_families"] == 100

    def test_the_real_protocol_hashes_stably(self):
        if not PROTOCOL.exists():
            pytest.skip("frozen protocol not present")
        assert protocol_sha256() == protocol_sha256(PROTOCOL)
        assert len(protocol_sha256()) == 64


# ---------------------------------------------------------------------
# P2    Strict 11.5 eligibility-report schema
# ---------------------------------------------------------------------
PANEL_IDS = {f"fam{i:03d}" for i in range(100)}


def _report(world, **kwargs) -> dict:
    """Write the world's 11.5 report with one perturbation, read it back."""
    path = _eligibility(world, **kwargs)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(world, report, **kwargs) -> list[str]:
    kwargs.setdefault("panel_family_ids", PANEL_IDS)
    kwargs.setdefault("expected_protocol_sha",
                      protocol_sha256(world["protocol_path"]))
    kwargs.setdefault("expected_lock_sha",
                      dependency_lock_sha256(world["lock_path"]))
    selection = world["expected_selection"]
    kwargs.setdefault("expected_family_ids", selection["selected_family_ids"])
    kwargs.setdefault("expected_selection_sha256",
                      selection["selected_families_sha256"])
    kwargs.setdefault("expected_processor_revision", PROCESSOR_REVISION)
    return validate_eligibility_report(
        report, model_spec=world["spec"], **kwargs)


def _gate_entry(name, **overrides) -> dict:
    """A conformant evidence entry for one required gate."""
    entry = {
        "generations_complete": {
            "passed": True, "n_attempts": ELIGIBILITY_N_ATTEMPTS,
            "n_succeeded": ELIGIBILITY_N_ATTEMPTS, "n_failed": 0},
        "truncation_reviewed": {
            "passed": True, "n_truncated": 0, "truncation_rate": 0.0,
            "max_variant_spread": 0.0},
        "vision_path_engaged": {
            "passed": True, "n_image_bearing_cells": 24,
            "min_image_token_count": 81},
        "terminal_query_invariant": {
            "passed": True, "n_families_checked": ELIGIBILITY_N_FAMILIES,
            "n_mismatched": 0},
        "revision_pinned": {
            "passed": True, "model_revision": REVISION,
            "processor_revision": PROCESSOR_REVISION},
        "determinism": {
            "passed": True, "n_repeats": 2, "n_distinct_responses": 1},
    }[name]
    assert set(entry) - {"passed"} == set(ELIGIBILITY_GATE_EVIDENCE[name])
    entry = dict(entry)
    entry.update(overrides)
    return entry


_RESOLVED_REVISIONS = {"model_revision": REVISION,
                       "processor_revision": PROCESSOR_REVISION}

_GATE_EVIDENCE_CASES = [(name, field)
                        for name, fields in
                        sorted(ELIGIBILITY_GATE_EVIDENCE.items())
                        for field in fields]


class TestGateEvidence:
    """Each detailed gate must SHOW its work, not just claim a verdict.

    The reported defect: the validator accepted any nonempty dictionary of
    passing entries, and a test established that ``{"only_gate": true}`` was
    valid — so a report could omit the vision, truncation, determinism and
    terminal-query checks entirely and still authorize generation.
    """

    @pytest.mark.parametrize("name", sorted(ELIGIBILITY_REQUIRED_GATES))
    def test_a_conformant_entry_passes(self, name):
        assert validate_gate_entry(name, _gate_entry(name),
                                   expected_revisions=_RESOLVED_REVISIONS) \
            == []

    @pytest.mark.parametrize("name", ["only_gate", "vision", "",
                                      "GENERATIONS_COMPLETE"])
    def test_an_undefined_gate_name_confers_nothing(self, name):
        problems = validate_gate_entry(name, {"passed": True})
        assert problems and "not one of the required gates" in problems[0]

    @pytest.mark.parametrize("entry", [True, False, "true", 1, None, [],
                                       "passed"])
    def test_a_bare_flag_is_not_auditable(self, entry):
        problems = validate_gate_entry("determinism", entry)
        assert problems and "a bare flag is not auditable" in problems[0]

    @pytest.mark.parametrize("passed", [False, None, 1, "true", "PASS"])
    def test_passed_must_be_literal_true(self, passed):
        problems = validate_gate_entry(
            "determinism", _gate_entry("determinism", passed=passed),
            expected_revisions=_RESOLVED_REVISIONS)
        assert any("did not pass" in p for p in problems)

    @pytest.mark.parametrize("name,field", _GATE_EVIDENCE_CASES)
    def test_every_evidence_field_is_required(self, name, field):
        # Absent AND explicitly null both count: the semantic checks need
        # the value, and a null is how a report says "not measured".
        for entry in (_gate_entry(name, **{field: None}),
                      {k: v for k, v in _gate_entry(name).items()
                       if k != field}):
            problems = validate_gate_entry(
                name, entry, expected_revisions=_RESOLVED_REVISIONS)
            assert len(problems) == 1, problems
            assert "missing evidence field" in problems[0]
            assert field in problems[0]
            # ...and the semantic checks are skipped rather than crashing on
            # the absent key.

    def test_revision_pinned_is_compared_to_what_this_run_resolves(self):
        problems = validate_gate_entry(
            "revision_pinned",
            _gate_entry("revision_pinned", processor_revision="f" * 40),
            expected_revisions=_RESOLVED_REVISIONS)
        assert any("this run resolves" in p for p in problems)

    def test_revision_pinned_needs_immutable_shas_even_without_an_expectation(
            self):
        problems = validate_gate_entry(
            "revision_pinned", _gate_entry("revision_pinned",
                                           model_revision="main"))
        assert any("not an immutable 40-hex SHA" in p for p in problems)

    def test_a_truncation_rate_outside_zero_to_one_is_rejected(self):
        for rate in (-0.1, 1.5, "0.0"):
            problems = validate_gate_entry(
                "truncation_reviewed",
                _gate_entry("truncation_reviewed", truncation_rate=rate))
            assert any("not a rate in [0, 1]" in p for p in problems), rate

    def test_the_evidence_contract_covers_every_required_gate(self):
        assert ELIGIBILITY_REQUIRED_GATES \
            == frozenset(ELIGIBILITY_GATE_EVIDENCE)
        assert len(ELIGIBILITY_REQUIRED_GATES) == 6
        # Every gate names at least one measurable field beyond ``passed``.
        assert all(fields for fields in ELIGIBILITY_GATE_EVIDENCE.values())


class TestEligibilityReportSchema:
    """The report that authorizes a 11.6 run must itself be verified.

    The earlier gate recorded ``code_commit`` without requiring it, so a
    report carrying ``code_commit: null`` — or one written about a
    different model_key entirely — could authorize a confirmatory run.
    """

    def test_a_conformant_report_is_valid(self, world):
        assert _validate(world, _report(world)) == []

    def test_the_schema_names_every_required_field(self):
        for field in ("model_key", "code_commit", "git_dirty",
                      "dependency_lock_sha256", "selected_families_sha256",
                      "n_selected_families", "variants",
                      "truncation_by_variant", "gates"):
            assert field in ELIGIBILITY_REQUIRED_FIELDS

    @pytest.mark.parametrize("field", list(ELIGIBILITY_REQUIRED_FIELDS))
    def test_no_required_field_may_be_absent(self, world, field):
        # The reported defect in its general form: a null field was recorded
        # and then never required.
        report = _report(world, overrides={field: None})
        problems = _validate(world, report)
        assert problems and field in problems[0]
        assert "missing required field" in problems[0]

    def test_a_null_code_commit_cannot_authorize_a_run(self, world):
        report = _report(world, overrides={"code_commit": None})
        assert any("missing required field" in p
                   for p in _validate(world, report))
        # ...and the same is true end-to-end through the confirmatory gate.
        lines = _violations(world)
        assert any("code_commit" in line for line in lines)

    @pytest.mark.parametrize("commit", ["abc", "e55c352", "HEAD", "main",
                                        "d" * 39, "e" * 41, "z" * 40])
    def test_code_commit_must_be_immutable(self, world, commit):
        problems = _validate(world, _report(world, code_commit=commit))
        assert any("code_commit" in p and "immutable" in p for p in problems)

    def test_an_uppercase_commit_sha_still_names_one_commit(self, world):
        # Case is normalized: forty hex digits in upper case still identify
        # exactly one immutable commit, so this is not a floating reference.
        assert _validate(world, _report(world, code_commit=COMMIT.upper())) \
            == []

    def test_the_commit_that_certified_eligibility_is_recorded(self, world):
        report = _report(world)
        assert report["code_commit"] == COMMIT
        result = _gate(world)
        assert result["checks"]["eligibility_code_commit"] == COMMIT
        assert result["checks"]["eligibility_git_dirty"] is False
        assert result["checks"]["eligibility_report_violations"] == []

    def test_a_report_for_another_model_key_is_rejected(self, world):
        report = _report(world)
        report["model_key"] = "ministral3_3b"
        problems = _validate(world, report)
        assert any("model_key" in p for p in problems)

    def test_a_report_for_another_model_id_is_rejected(self, world):
        problems = _validate(world, _report(world, model_id="mistralai/x"))
        assert any("model_id" in p for p in problems)

    def test_a_report_written_for_another_key_cannot_be_relocated(self, world):
        # Written under the right target's path but naming a different one:
        # the directory is not the identity.
        _eligibility(world, overrides={"model_key": "phi4_mm"})
        lines = _violations(world)
        assert any("model_key" in line and "phi4_mm" in line
                   for line in lines)

    def test_a_report_certified_under_another_lock_is_rejected(self, world):
        problems = _validate(world, _report(world),
                             expected_lock_sha="b" * 64)
        assert any("dependency_lock_sha256" in p for p in problems)

    def test_the_gate_compares_the_report_lock_to_the_lock_in_force(
            self, world):
        report = _report(world)
        report["dependency_lock_sha256"] = "c" * 64
        path = world["eligibility_root"] / "qwen35_4b" / "preflight_report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        lines = _violations(world)
        assert any("different environment" in line for line in lines)

    @pytest.mark.parametrize("revision", ["main", "latest", None, "d" * 39])
    def test_the_processor_revision_must_be_immutable(self, world, revision):
        problems = _validate(world, _report(
            world, processor_revision=revision))
        assert any("processor_revision" in p for p in problems)

    # --- the selected family subset -----------------------------------
    @pytest.mark.parametrize("n", [0, 1, 11, 13, 100])
    def test_the_selection_must_be_twelve_families(self, world, n):
        problems = _validate(world, _report(
            world, family_ids=_eligibility_ids(n)))
        assert any(f"selected {n} families" in p for p in problems)

    def test_duplicate_selections_are_rejected(self, world):
        ids = _eligibility_ids(11) + ["fam000"]
        problems = _validate(world, _report(world, family_ids=ids))
        assert any("duplicates" in p and "fam000" in p for p in problems)

    def test_n_selected_families_must_agree_with_the_ids(self, world):
        report = _report(world)
        report["n_selected_families"] = 99
        problems = _validate(world, report)
        assert any("n_selected_families=99" in p for p in problems)

    def test_the_recorded_selection_hash_must_match_the_ids(self, world):
        report = _report(world)
        report["selected_families_sha256"] = "0" * 64
        problems = _validate(world, report)
        assert any("does not match the listed family ids" in p
                   for p in problems)

    def test_the_selection_hash_is_order_independent(self):
        ids = _eligibility_ids()
        assert selected_families_sha256(ids) \
            == selected_families_sha256(list(reversed(ids)))
        assert selected_families_sha256(ids) \
            != selected_families_sha256(ids[:-1])

    def test_a_selection_outside_the_frozen_panel_is_rejected(self, world):
        ids = _eligibility_ids(11) + ["not_in_panel"]
        problems = _validate(world, _report(world, family_ids=ids))
        assert any("not in the frozen 100-family panel" in p
                   for p in problems)

    def test_panel_membership_is_checked_when_the_panel_is_known(self, world):
        report = _report(world)
        assert _validate(world, report, panel_family_ids=PANEL_IDS) == []
        assert any("not in the frozen" in p
                   for p in _validate(world, report,
                                      panel_family_ids={"other"}))

    # --- six-variant coverage and the 72/72 requirement ---------------
    def test_variants_must_be_the_frozen_six(self, world):
        report = _report(world)
        report["variants"] = list(ALL_VARIANT_NAMES[:5])
        problems = _validate(world, report)
        assert any("eligibility variants" in p for p in problems)

    def test_a_reordered_variant_list_is_rejected(self, world):
        # Order matters: the report must name the frozen six in the
        # declared order, so a reordering is a schema deviation rather
        # than something to be silently tolerated.
        report = _report(world)
        report["variants"] = list(reversed(ALL_VARIANT_NAMES))
        assert any("eligibility variants" in p
                   for p in _validate(world, report))

    @pytest.mark.parametrize("field", ["n_expected_attempts", "n_attempts",
                                       "n_succeeded"])
    def test_the_attempt_counts_must_be_seventy_two(self, world, field):
        report = _report(world)
        report[field] = ELIGIBILITY_N_ATTEMPTS - 1
        problems = _validate(world, report)
        assert any(f"{field}=71" in p for p in problems)

    def test_a_partial_generation_run_is_not_eligibility(self, world):
        report = _report(world)
        report["n_succeeded"] = 70
        problems = _validate(world, report)
        assert any("every one of the 72 generations" in p for p in problems)

    def test_truncation_must_cover_all_six_variants(self, world):
        report = _report(world)
        report["truncation_by_variant"].pop(ALL_VARIANT_NAMES[0])
        problems = _validate(world, report)
        assert any("missing variant(s)" in p for p in problems)

    def test_truncation_counts_must_match_the_selection(self, world):
        report = _report(world)
        report["truncation_by_variant"][ALL_VARIANT_NAMES[0]]["n"] = 5
        problems = _validate(world, report)
        assert any("truncation_by_variant counts per variant" in p
                   for p in problems)

    # --- detailed gate results ----------------------------------------
    def test_the_required_gate_set_is_exact(self):
        assert ELIGIBILITY_REQUIRED_GATES == frozenset({
            "generations_complete", "truncation_reviewed",
            "vision_path_engaged", "terminal_query_invariant",
            "revision_pinned", "determinism"})
        assert ELIGIBILITY_REQUIRED_GATES \
            == frozenset(ELIGIBILITY_GATE_EVIDENCE)
        for name, evidence in ELIGIBILITY_GATE_EVIDENCE.items():
            assert evidence, f"{name} declares no evidence fields"

    @pytest.mark.parametrize("gates", [{}, [], None, {"overall": "PASS"}])
    def test_a_thin_or_absent_gate_block_is_rejected(self, world, gates):
        problems = _validate(world, _report(world, overrides={"gates": gates}))
        if isinstance(gates, dict):
            # A dict is parsed, so the missing names are reported by name —
            # and an undeclared one is reported as unexpected.
            assert any("omits required detailed gate" in p for p in problems)
            if gates:
                assert any("unexpected gate" in p for p in problems)
        elif gates is None:
            assert any("missing required field" in p for p in problems)
        else:
            assert any("must be an object" in p for p in problems)

    @pytest.mark.parametrize("omitted",
                             sorted(ELIGIBILITY_REQUIRED_GATES))
    def test_omitting_any_single_required_gate_is_rejected(self, world,
                                                          omitted):
        # The reported defect: any non-empty dict of passing entries was
        # accepted, so a report could authorize a confirmatory run while
        # never having performed the vision-path or truncation check.
        report = _report(world)
        del report["gates"][omitted]
        problems = _validate(world, report)
        assert any("omits required detailed gate" in p and omitted in p
                   for p in problems)

    def test_an_unexpected_gate_name_cannot_confer_authority(self, world):
        report = _report(world)
        report["gates"]["something_invented"] = {"passed": True}
        problems = _validate(world, report)
        assert any("unexpected gate" in p and "something_invented" in p
                   for p in problems)

    @pytest.mark.parametrize("entry", [True, False, "PASS", 1, None,
                                       {"status": "PASS"},
                                       {"passed": True}])
    def test_a_bare_or_evidence_free_gate_entry_is_rejected(self, world,
                                                           entry):
        # ``{"passed": true}`` with no evidence is rejected too: it asserts a
        # conclusion without showing the measurement.
        report = _report(world)
        report["gates"]["determinism"] = entry
        problems = _validate(world, report)
        assert any("'determinism'" in p for p in problems)

    @pytest.mark.parametrize("name", sorted(ELIGIBILITY_GATE_EVIDENCE))
    def test_each_gate_must_carry_its_evidence(self, world, name):
        for field in ELIGIBILITY_GATE_EVIDENCE[name]:
            report = _report(world)
            del report["gates"][name][field]
            problems = _validate(world, report)
            assert any("missing evidence field" in p and field in p
                       for p in problems), (name, field)

    @pytest.mark.parametrize("name,bad", [
        ("generations_complete", {"n_failed": 3}),
        ("generations_complete", {"n_succeeded": 70}),
        ("generations_complete", {"n_attempts": 60}),
        ("truncation_reviewed", {"n_truncated": 1}),
        ("truncation_reviewed", {"truncation_rate": 1.5}),
        ("truncation_reviewed", {"max_variant_spread": -0.1}),
        ("vision_path_engaged", {"n_image_bearing_cells": 0}),
        ("vision_path_engaged", {"min_image_token_count": 0}),
        ("terminal_query_invariant", {"n_mismatched": 2}),
        ("terminal_query_invariant", {"n_families_checked": 11}),
        ("determinism", {"n_repeats": 1}),
        ("determinism", {"n_distinct_responses": 2}),
        ("revision_pinned", {"model_revision": "main"}),
        ("revision_pinned", {"processor_revision": "latest"}),
    ])
    def test_self_contradictory_evidence_is_rejected(self, world, name, bad):
        # ``passed: true`` alongside evidence that says otherwise is not
        # trusted; the contradiction is the violation.
        report = _report(world)
        report["gates"][name].update(bad)
        assert report["gates"][name]["passed"] is True
        problems = _validate(world, report)
        assert any(f"'{name}'" in p for p in problems), problems

    def test_a_gate_must_agree_with_the_revisions_this_run_resolves(
            self, world):
        report = _report(world)
        report["gates"]["revision_pinned"]["processor_revision"] = "f" * 40
        problems = _validate(world, report)
        assert any("'revision_pinned'" in p and "processor_revision" in p
                   for p in problems)

    def test_all_gates_passing_with_full_evidence_is_accepted(self, world):
        assert _validate(world, _report(world)) == []

    def test_the_number_of_detailed_gates_is_recorded(self, world):
        result = _gate(world)
        assert result["checks"]["eligibility_n_gates"] \
            == len(ELIGIBILITY_REQUIRED_GATES)

    # --- external pre-registration of the selection -------------------
    def test_replacing_the_ids_and_their_hash_together_is_rejected(
            self, world):
        # THE reported defect. The digest used to be recomputed from the ids
        # in the same report, so it only proved self-consistency: swap both
        # and nothing notices. Twelve other panel families, correctly
        # hashed, must still be refused.
        substituted = [f"fam{i:03d}" for i in range(50, 62)]
        report = _report(world)
        report["selected_family_ids"] = substituted
        report["selected_families_sha256"] = \
            selected_families_sha256(substituted)
        report["n_selected_families"] = len(substituted)
        report["truncation_by_variant"] = {
            v: {"n": len(substituted), "n_truncated": 0,
                "truncation_rate": 0.0} for v in ALL_VARIANT_NAMES}
        problems = _validate(world, report)
        assert any("not the pre-registered 11.5 selection" in p
                   for p in problems), problems
        # The internal-consistency check alone would have passed it.
        assert report["selected_families_sha256"] \
            == selected_families_sha256(substituted)

    def test_the_pre_registered_digest_is_required(self, world):
        report = _report(world)
        report["selected_families_sha256"] = "9" * 64
        problems = _validate(world, report)
        assert any("not the pre-registered selection digest" in p
                   for p in problems)

    def test_the_gate_records_the_selection_it_derived(self, world):
        result = _gate(world)
        assert result["checks"]["expected_selection_sha256"] \
            == world["expected_selection"]["selected_families_sha256"]
        assert result["checks"]["expected_selection_n_families"] \
            == ELIGIBILITY_N_FAMILIES

    def test_the_gate_rejects_a_report_naming_a_different_subset(
            self, world):
        substituted = [f"fam{i:03d}" for i in range(50, 62)]
        _eligibility(world, family_ids=substituted)
        lines = _violations(world)
        assert any("not the pre-registered 11.5 selection" in line
                   for line in lines)

    def test_the_expected_processor_revision_comes_from_the_lock(self, world):
        result = _gate(world)
        assert result["checks"]["expected_processor_revision"] \
            == PROCESSOR_REVISION

    def test_a_processor_revision_differing_from_the_lock_is_rejected(
            self, world):
        # Immutable-looking is not the same as correct: this report was
        # certified against a different processor, so its chat template or
        # image processor may render prompts differently.
        problems = _validate(world, _report(world),
                             expected_processor_revision="f" * 40)
        assert any("does not transfer across processor revisions" in p
                   for p in problems)

    def test_the_report_processor_revision_must_match_the_lock(self, world):
        report = _report(world)
        report["processor_revision"] = "f" * 40
        report["gates"]["revision_pinned"]["processor_revision"] = "f" * 40
        problems = _validate(world, report)
        assert any("processor_revision" in p and "does not transfer" in p
                   for p in problems)


# ---------------------------------------------------------------------
# P1    The selection is pre-registered by DERIVATION from frozen evidence
# ---------------------------------------------------------------------
#: 18 synthetic families. Their lengths fall into three clean tertiles
#: (100-105 / 200-206 / 300-304) and their compliance levels cycle 0/1/2
#: inside each length group, so every one of the nine (length, risk) cells
#: is populated and the whole allocation can be checked by hand.
_FROZEN_FAMILIES = [
    ("fam000", 100, 0), ("fam001", 101, 1), ("fam002", 102, 2),
    ("fam003", 103, 0), ("fam004", 104, 1), ("fam005", 105, 2),
    ("fam006", 200, 0), ("fam007", 201, 1), ("fam008", 202, 2),
    ("fam009", 203, 0), ("fam010", 204, 1), ("fam011", 205, 2),
    ("fam012", 206, 0),
    ("fam013", 300, 0), ("fam014", 301, 1), ("fam015", 302, 2),
    ("fam016", 303, 0), ("fam017", 304, 1),
]

#: Hand-computed from ``selection.SELECTION_RECIPE``: the tertile cuts land
#: on the observed values 200 and 206, round 1 takes the family closest to
#: each cell's median length (ties by family_id), and the three extras go
#: one each to median|compliant (the only cell of population 3),
#: short|partial and short|noncompliant.
_FROZEN_EXPECTED_IDS = [
    "fam000", "fam001", "fam002", "fam004", "fam005", "fam006",
    "fam007", "fam008", "fam009", "fam013", "fam014", "fam015",
]

_FROZEN_PANEL = "outputs/scale_c/panel/validated_families.jsonl"
_FROZEN_RUN_DIR = "outputs/scale_c/replay_runs/frozen-reference"
_FROZEN_OUTPUT_DIR = "outputs/scale_c"

#: The digest this repository's frozen evidence actually derives to. Pinned
#: so that any change to the recipe — tertile indices, tie-breaks, the extra
#: allocation — fails a test instead of silently redefining which 12
#: families 11.5 is allowed to replay.
REAL_SELECTION_SHA256 = (
    "f2806e496a6e959ee209fecd91b7484c4f6060fc7afef57d4ab5158d32d06a49")


@pytest.fixture
def frozen_repo(tmp_path_factory):
    """A repository whose frozen evidence a selection can be derived from.

    The real derivation follows two committed pointers to three frozen
    artifacts. This reproduces that layout on a synthetic panel so the
    derivation, its panel-digest guard and the gate's ``repo_root`` path can
    all be exercised offline and deterministically.
    """
    root = tmp_path_factory.mktemp("frozen_repo")
    panel_path = root / _FROZEN_PANEL
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_text("".join(
        json.dumps({"family_id": family}) + "\n"
        for family, _, _ in _FROZEN_FAMILIES), encoding="utf-8")

    run_dir = root / _FROZEN_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / selection.REFERENCE_OUTPUTS_FILE).write_text("".join(
        json.dumps({"family_id": family, "variant": variant,
                    "output_token_count": tokens}) + "\n"
        for family, tokens, _ in _FROZEN_FAMILIES
        for variant in ALL_VARIANT_NAMES), encoding="utf-8")

    labels_dir = root / _FROZEN_OUTPUT_DIR
    (labels_dir / selection.ADJUDICATED_LABELS_FILE).write_text(
        json.dumps({"labels": {
            family: {variant: {"compliance_level": level}
                     for variant in ALL_VARIANT_NAMES}
            for family, _, level in _FROZEN_FAMILIES}}), encoding="utf-8")

    reference_path = root / selection.FROZEN_9B_REFERENCE
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps({
        "run_id": "frozen-reference",
        "run_dir": _FROZEN_RUN_DIR,
        "validated_families_sha256": _file_sha256(panel_path),
    }), encoding="utf-8")

    profiles_path = root / selection.SCALE_PROFILES
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps({
        selection.SCALE_PROFILE: {"validated_families": _FROZEN_PANEL,
                                  "output_dir": _FROZEN_OUTPUT_DIR},
    }), encoding="utf-8")
    return root


def _write_selection_artifact(root: Path, document: dict) -> Path:
    path = root / selection.SELECTION_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


class TestFrozenSelectionDerivation:
    """The expected selection must come from evidence a report cannot touch.

    The reported defect: the validator recomputed ``selected_families_sha256``
    from the ``selected_family_ids`` in the SAME report, so replacing both
    together still passed. These tests derive the expectation instead.
    """

    def test_the_selection_is_derived_from_the_frozen_inputs(
            self, frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        assert derived["n_panel_families"] == len(_FROZEN_FAMILIES)
        assert derived["selected_family_ids"] == _FROZEN_EXPECTED_IDS
        assert derived["n_selected_families"] == ELIGIBILITY_N_FAMILIES
        assert derived["selected_families_sha256"] \
            == selected_families_sha256(_FROZEN_EXPECTED_IDS)
        assert derived["length_cuts"] == {"t1": 200, "t2": 206}
        assert set(derived["derived_from"]) == {
            "panel", "panel_validated_families_sha256", "reference_outputs",
            "reference_run_id", "adjudicated_labels"}
        # Repo-relative, so the artifact is portable and byte-reproducible
        # from any checkout location.
        assert derived["derived_from"]["panel"] == _FROZEN_PANEL
        assert derived["derived_from"]["reference_outputs"] == \
            f"{_FROZEN_RUN_DIR}/{selection.REFERENCE_OUTPUTS_FILE}"
        assert derived["uses_candidate_target_information"] is False

    def test_derivation_is_repeatable(self, frozen_repo):
        # Pre-registration is worthless if two derivations disagree.
        assert selection.derive_frozen_selection(frozen_repo) \
            == selection.derive_frozen_selection(frozen_repo)

    def test_every_strata_cell_is_covered_and_the_extras_are_spread(
            self, frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        assert len(derived["cells"]) == selection.N_CELLS
        assert all(cell["n_selected"] >= 1
                   for cell in derived["cells"].values())
        assert derived["by_length_stratum"] == {
            "short": 5, "median": 4, "long": 3}
        assert derived["by_risk_stratum"] == {
            "compliant": 4, "partial": 4, "noncompliant": 4}
        # One extra per cell: without that rule the single largest cell wins
        # every round, because its population never changes.
        cells = [extra["cell"] for extra in derived["extra_allocation"]]
        assert cells == ["median|compliant", "short|partial",
                         "short|noncompliant"]
        assert len(set(cells)) == selection.N_EXTRAS

    def test_a_panel_that_is_not_the_frozen_one_blocks_derivation(
            self, frozen_repo):
        # Adding a family changes which 12 are selected, so the panel the
        # pointer names must be the panel that is actually read.
        panel = frozen_repo / _FROZEN_PANEL
        panel.write_text(panel.read_text(encoding="utf-8")
                         + json.dumps({"family_id": "fam999"}) + "\n",
                         encoding="utf-8")
        with pytest.raises(ReplayError, match="not the frozen one"):
            selection.derive_frozen_selection(frozen_repo)

    @pytest.mark.parametrize("pointer", [selection.FROZEN_9B_REFERENCE,
                                         selection.SCALE_PROFILES])
    def test_a_missing_pointer_blocks_derivation(self, frozen_repo, pointer):
        (frozen_repo / pointer).unlink()
        with pytest.raises(ReplayError, match="not found"):
            selection.derive_frozen_selection(frozen_repo)

    def test_a_family_without_a_stratum_input_blocks_derivation(
            self, frozen_repo):
        outputs = (frozen_repo / _FROZEN_RUN_DIR
                   / selection.REFERENCE_OUTPUTS_FILE)
        kept = [line for line in
                outputs.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["family_id"] != "fam015"]
        outputs.write_text("".join(line + "\n" for line in kept),
                           encoding="utf-8")
        with pytest.raises(ReplayError, match="incomplete"):
            selection.derive_frozen_selection(frozen_repo)

    def test_an_empty_strata_cell_blocks_derivation(self, frozen_repo):
        labels_path = (frozen_repo / _FROZEN_OUTPUT_DIR
                       / selection.ADJUDICATED_LABELS_FILE)
        document = json.loads(labels_path.read_text(encoding="utf-8"))
        for label in document["labels"]["fam015"].values():
            label["compliance_level"] = 0
        labels_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ReplayError, match="cannot cover the grid"):
            selection.derive_frozen_selection(frozen_repo)

    def test_a_reference_record_without_a_token_count_blocks_derivation(
            self, frozen_repo):
        outputs = (frozen_repo / _FROZEN_RUN_DIR
                   / selection.REFERENCE_OUTPUTS_FILE)
        records = [json.loads(line) for line in
                   outputs.read_text(encoding="utf-8").splitlines()]
        del records[0]["output_token_count"]
        outputs.write_text("".join(json.dumps(r) + "\n" for r in records),
                           encoding="utf-8")
        with pytest.raises(ReplayError, match="output_token_count"):
            selection.derive_frozen_selection(frozen_repo)

    # --- through the confirmatory gate --------------------------------
    def test_the_gate_derives_the_selection_it_expects(self, world,
                                                       frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        _eligibility(world, family_ids=derived["selected_family_ids"])
        result = _gate(world, expected_selection=None, repo_root=frozen_repo)
        assert result["checks"]["expected_selection_sha256"] \
            == derived["selected_families_sha256"]
        assert result["checks"]["expected_selection_n_families"] \
            == ELIGIBILITY_N_FAMILIES
        assert result["checks"]["selection_artifact_path"] \
            == str(frozen_repo / selection.SELECTION_ARTIFACT)
        assert result["checks"]["selection_artifact_checked"] is True
        assert result["checks"]["selection_artifact_present"] is False

    def test_an_injected_expectation_does_not_consult_the_artifact(
            self, world):
        # The synthetic world injects its own expectation, so this
        # repository's committed artifact — which describes a different
        # evidence base — is recorded but not compared. Holding unrelated
        # selections against each other would fail the check for no reason,
        # and the recording keeps the skip visible rather than silent.
        result = _gate(world)
        assert result["checks"]["selection_artifact_checked"] is False
        assert "selection_artifact_present" not in result["checks"]

    def test_the_gate_refuses_a_report_naming_the_first_twelve_families(
            self, world, frozen_repo):
        # The prefix is what a report would name if the selection were
        # merely "the first 12"; the derivation instead drops fam003,
        # fam010, fam011 and fam012 and adds fam013, fam014 and fam015.
        _eligibility(world, family_ids=_eligibility_ids())
        lines = _violations(world, expected_selection=None,
                           repo_root=frozen_repo)
        assert any("not the pre-registered 11.5 selection" in line
                   for line in lines)

    def test_an_underivable_selection_blocks_the_run(self, world,
                                                     frozen_repo):
        # Fail-closed: no derivation, no confirmatory run — the gate must
        # not fall back to believing the report.
        (frozen_repo / selection.FROZEN_9B_REFERENCE).unlink()
        _eligibility(world)
        with pytest.raises(ReplayError, match="not found"):
            _gate(world, expected_selection=None, repo_root=frozen_repo)

    def test_a_tampered_committed_artifact_is_refused(self, world,
                                                      frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        tampered = dict(derived)
        tampered["selected_family_ids"] = _eligibility_ids()
        tampered["selected_families_sha256"] = \
            selected_families_sha256(_eligibility_ids())
        _write_selection_artifact(frozen_repo, tampered)
        _eligibility(world, family_ids=derived["selected_family_ids"])
        lines = _violations(world, expected_selection=None,
                           repo_root=frozen_repo)
        assert any("pre-registered by derivation" in line for line in lines)

    def test_a_matching_committed_artifact_is_accepted(self, world,
                                                       frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        _write_selection_artifact(frozen_repo, derived)
        _eligibility(world, family_ids=derived["selected_family_ids"])
        result = _gate(world, expected_selection=None, repo_root=frozen_repo)
        assert result["passed"] is True
        assert result["checks"]["selection_artifact_present"] is True

    def test_an_unreadable_committed_artifact_is_refused(self, world,
                                                         frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        path = _write_selection_artifact(frozen_repo, derived)
        path.write_text("{not json", encoding="utf-8")
        _eligibility(world, family_ids=derived["selected_family_ids"])
        lines = _violations(world, expected_selection=None,
                           repo_root=frozen_repo)
        assert any("selection artifact" in line and "unreadable" in line
                   for line in lines)

    def test_the_real_repository_derives_the_pinned_selection(self):
        root = registry.REPO_ROOT
        if not (root / selection.FROZEN_9B_REFERENCE).exists():
            pytest.skip("frozen 9B reference pointer not present")
        derived = selection.derive_frozen_selection(root)
        assert derived["selected_families_sha256"] == REAL_SELECTION_SHA256
        assert derived["n_panel_families"] == 100
        assert derived["length_cuts"] == {"t1": 343.0, "t2": 467.5}
        ids = derived["selected_family_ids"]
        assert len(ids) == len(set(ids)) == ELIGIBILITY_N_FAMILIES
        assert ids == sorted(ids)

    def test_the_committed_selection_artifact_matches_a_fresh_derivation(
            self):
        root = registry.REPO_ROOT
        artifact = root / selection.SELECTION_ARTIFACT
        if not artifact.exists() \
                or not (root / selection.FROZEN_9B_REFERENCE).exists():
            pytest.skip("selection artifact or frozen pointer not present")
        committed = json.loads(artifact.read_text(encoding="utf-8"))
        derived = selection.derive_frozen_selection(root)
        assert committed["selected_family_ids"] \
            == derived["selected_family_ids"]
        assert committed["selected_families_sha256"] \
            == REAL_SELECTION_SHA256
        # The WHOLE document, byte for byte: the artifact is the audit trail
        # for the pre-registered selection, so a stale cell population or a
        # reworded recipe is as misleading as a different id list.
        assert artifact.read_text(encoding="utf-8") \
            == write_selection.canonical(derived)

    def test_the_committed_artifact_carries_the_full_audit_trail(self):
        artifact = registry.REPO_ROOT / selection.SELECTION_ARTIFACT
        if not artifact.exists():
            pytest.skip("selection artifact not present")
        committed = json.loads(artifact.read_text(encoding="utf-8"))
        assert committed["n_selected_families"] == ELIGIBILITY_N_FAMILIES
        assert len(committed["cells"]) == selection.N_CELLS
        assert sum(cell["n_selected"]
                   for cell in committed["cells"].values()) \
            == ELIGIBILITY_N_FAMILIES
        assert len(committed["extra_allocation"]) == selection.N_EXTRAS
        for extra in committed["extra_allocation"]:
            assert extra["reason"], "an extra slot is unexplained"
        assert committed["deterministic"] is True
        assert committed["uses_candidate_target_information"] is False
        assert committed["recipe"] == selection.SELECTION_RECIPE
        # The audit trail has to let a reader re-find every chosen family in
        # the frozen reference run.
        for cell in committed["cells"].values():
            for chosen in cell["selected"]:
                assert chosen["reference_median_output_tokens"] > 0
                assert chosen["distance_to_cell_median"] >= 0
                assert 0 <= chosen["max_compliance_level"] <= 3


# ---------------------------------------------------------------------
# 11.5  The eligibility subset run
# ---------------------------------------------------------------------
def _subset_run(tmp_path, ids, n_families=4, **kwargs):
    """Replay a synthetic panel restricted to ``ids``."""
    families = _families(tmp_path, n_families)
    spec = _spec()
    stub = kwargs.pop("backend", None) or _StubAdapter(spec)
    report = run_replay_stage(
        tmp_path, tmp_path / "runs", backend=stub, run_id="elig-run",
        model_spec=spec, family_ids=ids, **kwargs)
    return report, stub, families


class TestFamilySubsetRuns:
    """``family_ids`` is how 11.5 replays 12 of the frozen 100 families.

    ``input_dir`` stays the full panel — so the run fingerprint still binds
    the frozen panel digest — and the subset is bound separately, because a
    subset and the whole panel are different runs and must not resume into
    each other.
    """

    def test_only_the_named_families_are_replayed(self, tmp_path):
        report, stub, _ = _subset_run(tmp_path, ["fam000", "fam002"])
        assert report["n_families"] == 2
        assert report["n_attempted"] == 2 * len(ALL_VARIANT_NAMES)
        assert report["n_succeeded"] == 2 * len(ALL_VARIANT_NAMES)
        assert stub.n_calls == 2 * len(ALL_VARIANT_NAMES)

    def test_the_whole_panel_is_still_bound_by_the_fingerprint(self, tmp_path):
        # The subset must not weaken panel identity: the run still reads —
        # and is still bound to — the full frozen panel file.
        report, _, _ = _subset_run(tmp_path, ["fam000"])
        assert report["provenance"]["validated_families_sha256"] \
            == _file_sha256(tmp_path / "validated_families.jsonl")
        assert report["provenance"]["dataset_manifest_hash"] \
            == _file_sha256(tmp_path / "validated_families.jsonl")

    def test_the_subset_is_replayed_in_panel_order(self, tmp_path):
        report, _, _ = _subset_run(tmp_path, ["fam003", "fam001", "fam002"])
        seen = []
        for record in read_jsonl(tmp_path / "runs" / "elig-run"
                                 / REPLAY_OUTPUTS_FILE):
            if record["family_id"] not in seen:
                seen.append(record["family_id"])
        assert seen == ["fam001", "fam002", "fam003"]

    def test_the_report_records_the_subset_it_replayed(self, tmp_path):
        report, _, _ = _subset_run(tmp_path, ["fam002", "fam000"])
        subset = report["provenance"]["family_subset"]
        assert subset["family_ids"] == ["fam000", "fam002"]
        assert subset["n_families"] == 2
        assert subset["family_ids_sha256"] \
            == selected_families_sha256(["fam000", "fam002"])

    def test_a_whole_panel_run_records_no_subset(self, tmp_path):
        families = _families(tmp_path, 3)
        spec = _spec()
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", backend=_StubAdapter(spec),
            run_id="full-run", model_spec=spec)
        assert report["n_families"] == len(families)
        assert report["provenance"]["family_subset"] is None

    def test_an_unknown_family_id_is_refused(self, tmp_path):
        _families(tmp_path, 3)
        with pytest.raises(ReplayError, match="are not in"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", backend=_StubAdapter(_spec()),
                run_id="elig-run", model_spec=_spec(),
                family_ids=["fam000", "fam999"])

    def test_a_duplicate_family_id_is_refused(self, tmp_path):
        _families(tmp_path, 3)
        with pytest.raises(ReplayError, match="duplicate"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", backend=_StubAdapter(_spec()),
                run_id="elig-run", model_spec=_spec(),
                family_ids=["fam000", "fam000"])

    def test_an_empty_selection_is_refused(self, tmp_path):
        _families(tmp_path, 3)
        with pytest.raises(ReplayError, match="empty"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", backend=_StubAdapter(_spec()),
                run_id="elig-run", model_spec=_spec(), family_ids=[])

    def test_a_subset_requires_a_resolved_target(self, tmp_path):
        # The frozen legacy single-model report schema has no field for a
        # subset, so a subset run there could not say what it replayed.
        _families(tmp_path, 3)
        with pytest.raises(ReplayError, match="Iteration 11 facility"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", backend=_StubAdapter(_spec()),
                run_id="legacy", family_ids=["fam000"])

    def test_a_subset_cannot_be_combined_with_a_smoke_prefix(self, tmp_path):
        _families(tmp_path, 4)
        with pytest.raises(ReplayError, match="cannot be combined"):
            run_replay_stage(
                tmp_path, tmp_path / "runs", backend=_StubAdapter(_spec()),
                run_id="elig-run", model_spec=_spec(),
                family_ids=["fam000", "fam001"], max_families=1)

    def test_the_fingerprint_binds_the_subset(self, tmp_path):
        # Without this, --resume pointed at a 12-family eligibility run with
        # the full panel would find 72 stored pairs, treat them as 72 of the
        # 600 already done, and splice eligibility evidence into
        # confirmatory evidence.
        families = _families(tmp_path, 4)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        ids = [f.family_id for f in families]
        digests = {
            "full": iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw),
            "first_two": iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw, family_ids=ids[:2]),
            "last_two": iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw, family_ids=ids[2:]),
        }
        assert len(set(digests.values())) == 3, digests
        # ...and the subset digest is order-independent, matching the
        # published selection recipe.
        assert iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw, family_ids=ids[:2]) \
            == iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw,
                family_ids=list(reversed(ids[:2])))

    def test_resume_refuses_a_journal_from_a_different_subset(self, tmp_path):
        _subset_run(tmp_path, ["fam000", "fam001"])
        with pytest.raises(ReplayError):
            _subset_run(tmp_path, ["fam000", "fam002"], resume=True)

    def test_resume_continues_the_same_subset(self, tmp_path):
        report, _, _ = _subset_run(tmp_path, ["fam000", "fam001"],
                                   resume=True)
        assert report["resume"]["enabled"] is True
        assert report["n_succeeded"] == 2 * len(ALL_VARIANT_NAMES)

    @pytest.mark.parametrize("gate_name,key", [
        ("iteration11_confirmatory", "confirmatory_gate"),
        ("iteration11_eligibility", "eligibility_gate"),
    ])
    def test_gate_evidence_is_filed_under_the_gate_that_produced_it(
            self, tmp_path, gate_name, key):
        # A 12-family eligibility run must never carry its evidence under
        # the confirmatory key, or a reader could mistake it for a
        # whole-panel PASS.
        _families(tmp_path, 2)
        spec = _spec()
        gate = {"gate": gate_name, "passed": True, "n_violations": 0,
                "violations": [], "checks": {}}
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", backend=_StubAdapter(spec),
            run_id="elig-run", model_spec=spec, gate_evidence=gate)
        assert report[key] == gate
        other = ("eligibility_gate" if key == "confirmatory_gate"
                 else "confirmatory_gate")
        assert other not in report

    def test_the_stage_output_prefixes_reach_the_fingerprint(self, tmp_path,
                                                             monkeypatch):
        # An 11.5 run's own report and generations live under eligibility/.
        # Excluding them is what keeps a resumed run's fingerprint equal to
        # its first attempt's; without it the resume would see its own prior
        # outputs as untracked files and refuse itself.
        def fake(exclude_prefixes=()):
            dirty = "outputs/iteration_11/eligibility/" not in tuple(
                exclude_prefixes)
            return _tree(dirty,
                         dirty_paths=["outputs/iteration_11/eligibility/x"]
                         if dirty else [])

        monkeypatch.setattr(runner, "code_tree_status", fake)
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        narrow = iteration11_run_fingerprint(stub, config, tmp_path, spec, hw)
        wide = iteration11_run_fingerprint(
            stub, config, tmp_path, spec, hw,
            own_output_prefixes=confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES)
        assert narrow != wide

    def test_the_report_records_the_exclusions_it_relied_on(self, tmp_path):
        # The clean-tree answer a run recorded is only interpretable
        # together with the exclusions it was computed under.
        report, _, _ = _subset_run(
            tmp_path, ["fam000"],
            own_output_prefixes=confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES)
        assert report["provenance"]["code_own_output_prefixes"] == list(
            confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES)


# ---------------------------------------------------------------------
# 11.5  The eligibility protocol gate
# ---------------------------------------------------------------------
def _eligibility_gate(world, frozen_repo, **overrides):
    """Enforce the 11.5 gate on a conformant synthetic world."""
    derived = selection.derive_frozen_selection(frozen_repo)
    kwargs = {
        "input_dir": world["input_dir"],
        "config": world["config"],
        "model_spec": world["spec"],
        "family_ids": derived["selected_family_ids"],
        "output_root": f"{ELIGIBILITY_GENERATIONS_ROOT}/qwen35_4b",
        "lock_path": world["lock_path"],
        "protocol_path": world["protocol_path"],
        "repo_root": frozen_repo,
    }
    kwargs.update(overrides)
    return enforce_eligibility_protocol(**kwargs)


def _eligibility_violations(world, frozen_repo, **overrides) -> list[str]:
    with pytest.raises(ReplayError) as exc:
        _eligibility_gate(world, frozen_repo, **overrides)
    return str(exc.value).splitlines()


class TestEligibilityProtocolGate:
    """11.5 is not confirmatory, but the frozen protocol still binds it.

    Eligibility evidence generated under a different cap, decoding, prompt or
    panel would certify a target against conditions the 11.6 confirmatory run
    does not share — so every dimension the protocol fixes is enforced here
    by the SAME checks, and only the family scope and the report requirement
    differ.
    """

    def test_a_conformant_eligibility_run_passes(self, world, frozen_repo):
        result = _eligibility_gate(world, frozen_repo)
        assert result["passed"] is True
        assert result["gate"] == "iteration11_eligibility"
        assert result["checks"]["panel_sha256"] == world["panel_sha"]
        assert result["checks"]["max_new_tokens"] == 1536
        assert result["checks"]["n_family_ids_requested"] \
            == ELIGIBILITY_N_FAMILIES

    def test_it_requires_no_pre_existing_eligibility_report(self, world,
                                                            frozen_repo):
        # This stage WRITES the report, so demanding one would deadlock: no
        # target could ever become eligible.
        result = _eligibility_gate(world, frozen_repo)
        assert "eligibility_report_path" not in result["checks"]
        assert result["checks"]["selection_artifact_checked"] is True

    def test_the_legacy_single_model_path_is_refused(self, world,
                                                     frozen_repo):
        with pytest.raises(ReplayError, match="requires a resolved"):
            _eligibility_gate(world, frozen_repo, model_spec=None)

    def test_a_smoke_prefix_is_refused(self, world, frozen_repo):
        lines = _eligibility_violations(world, frozen_repo, max_families=5)
        assert any("prefix of it" in line for line in lines)

    @pytest.mark.parametrize("output_root", [
        f"{GENERATIONS_ROOT}/qwen35_4b",
        f"{ELIGIBILITY_GENERATIONS_ROOT}/ministral3_3b",
        "/tmp/somewhere",
    ])
    def test_evidence_must_land_in_this_targets_eligibility_tree(
            self, world, frozen_repo, output_root):
        # Including the CONFIRMATORY root: a 12-family run written there
        # could be read as a 100-family one by anything that globs it.
        lines = _eligibility_violations(world, frozen_repo,
                                       output_root=output_root)
        assert any("eligibility generations tree" in line for line in lines)

    def test_an_unnamed_subset_is_refused(self, world, frozen_repo):
        lines = _eligibility_violations(world, frozen_repo, family_ids=None)
        assert any("must name the families it replays" in line
                   for line in lines)

    @pytest.mark.parametrize("mutate", [
        pytest.param(lambda ids: ids[:-1], id="eleven_of_the_twelve"),
        pytest.param(lambda ids: ids + ["fam050"], id="twelve_plus_one"),
        pytest.param(lambda ids: list(reversed(ids)), id="same_set_ok"),
        pytest.param(lambda ids: _eligibility_ids(), id="the_panel_prefix"),
        pytest.param(lambda ids: [f"fam{i:03d}" for i in range(50, 62)],
                     id="twelve_others"),
    ])
    def test_the_subset_must_be_exactly_the_pre_registered_selection(
            self, world, frozen_repo, mutate):
        derived = selection.derive_frozen_selection(frozen_repo)
        ids = mutate(list(derived["selected_family_ids"]))
        if sorted(ids) == sorted(derived["selected_family_ids"]):
            # Order is not part of the selection: the digest is over the
            # sorted ids, so a reordered list is the same subset.
            assert _eligibility_gate(world, frozen_repo,
                                     family_ids=ids)["passed"] is True
            return
        lines = _eligibility_violations(world, frozen_repo, family_ids=ids)
        assert any("not the pre-registered 11.5 selection" in line
                   for line in lines)

    def test_an_id_outside_the_frozen_panel_is_refused(self, world,
                                                       frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        ids = list(derived["selected_family_ids"])
        ids[-1] = "CMST_999999"
        lines = _eligibility_violations(world, frozen_repo, family_ids=ids)
        assert any("not in the frozen panel" in line for line in lines)

    def test_the_frozen_cap_is_enforced_here_too(self, world, frozen_repo):
        derived = selection.derive_frozen_selection(frozen_repo)
        lines = _eligibility_violations(
            world, frozen_repo,
            config=ReplayConfig(max_new_tokens=256, enable_thinking=True),
            family_ids=derived["selected_family_ids"])
        assert any("uniform cap 1536" in line for line in lines)
        assert any("thinking disabled" in line for line in lines)

    def test_a_dirty_tree_is_refused(self, world, frozen_repo, monkeypatch):
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(True,
                                              dirty_paths=["src/x.py"]))
        lines = _eligibility_violations(world, frozen_repo)
        assert any("working tree is not clean" in line for line in lines)

    def test_overwrite_is_refused(self, world, frozen_repo):
        lines = _eligibility_violations(world, frozen_repo, overwrite=True)
        assert any("overwrite=True" in line for line in lines)

    def test_a_different_panel_is_refused(self, world, frozen_repo):
        other = world["tmp_path"] / "other" / "validated_families.jsonl"
        _write_panel(other, n_families=100)
        other.write_text(
            other.read_text(encoding="utf-8").replace("fam", "fxm", 1),
            encoding="utf-8")
        lines = _eligibility_violations(world, frozen_repo,
                                       input_dir=other.parent)
        assert any("not the frozen" in line for line in lines)

    def test_an_underivable_selection_blocks_the_run(self, world,
                                                     frozen_repo):
        (frozen_repo / selection.FROZEN_9B_REFERENCE).unlink()
        with pytest.raises(ReplayError, match="not found"):
            _eligibility_gate(world, frozen_repo)

    def test_every_violation_is_reported_together(self, world, frozen_repo,
                                                  monkeypatch):
        monkeypatch.setattr(
            confirmatory, "code_tree_status",
            lambda exclude_prefixes=(): _tree(True,
                                              dirty_paths=["src/x.py"]))
        derived = selection.derive_frozen_selection(frozen_repo)
        with pytest.raises(ReplayError) as exc:
            _eligibility_gate(
                world, frozen_repo, overwrite=True, max_families=5,
                family_ids=derived["selected_family_ids"][:-1],
                config=ReplayConfig(max_new_tokens=256,
                                    enable_thinking=True))
        message = str(exc.value)
        # overwrite + dirty tree + max_families + cap + thinking + subset
        assert "6 violation(s)" in message
        for fragment in ("overwrite=True", "working tree is not clean",
                         "prefix of it", "uniform cap 1536",
                         "thinking disabled",
                         "not the pre-registered 11.5 selection"):
            assert fragment in message

    def test_the_confirmatory_gate_refuses_a_named_subset(self, world):
        # The same argument is a violation on the other side: a
        # confirmatory run replays the whole panel, and a 12-family run must
        # not be labelled confirmatory.
        lines = _violations(world, family_ids=_eligibility_ids())
        assert any("confirmatory run replays the ENTIRE frozen panel" in line
                   for line in lines)
        assert any("enforce_eligibility_protocol" in line for line in lines)

    def test_the_confirmatory_gate_records_that_no_subset_was_requested(
            self, world):
        assert _gate(world)["checks"]["family_ids"] is None

    def test_the_eligibility_gate_excludes_its_own_report_tree(
            self, world, frozen_repo, monkeypatch):
        # The 11.5 stage writes its report under eligibility/, so that tree
        # is its own output. Without the exclusion the first target's report
        # would make the tree dirty and block every later target even though
        # nothing about the code changed.
        seen = {}

        def fake(exclude_prefixes=()):
            seen["prefixes"] = tuple(exclude_prefixes)
            return _tree(False, own_outputs=[
                "outputs/iteration_11/eligibility/qwen35_2b/"
                "preflight_report.json"])

        monkeypatch.setattr(confirmatory, "code_tree_status", fake)
        result = _eligibility_gate(world, frozen_repo)
        assert result["passed"] is True
        assert seen["prefixes"] \
            == confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES
        assert result["checks"]["git_own_output_prefixes"] == list(
            confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES)
        assert result["checks"]["git_dirty_excluded_own_outputs"]

    def test_the_two_stages_do_not_share_an_exclusion(self):
        # The eligibility report is THIS stage's product and the
        # confirmatory stage's INPUT, so the narrower confirmatory list must
        # not cover it: an uncommitted report has to block 11.6 rather than
        # pass unnoticed as "a stage's own output".
        assert not any("eligibility" in prefix
                       for prefix in confirmatory.OWN_OUTPUT_PREFIXES)
        assert confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES == (
            "outputs/iteration_11/eligibility/",)
        assert confirmatory.ELIGIBILITY_GENERATIONS_ROOT.startswith(
            confirmatory.ELIGIBILITY_OWN_OUTPUT_PREFIXES[0])


# ---------------------------------------------------------------------
# 11.5  Evidence -> gate, computed rather than asserted
# ---------------------------------------------------------------------
def _record(family_id, variant, *, n_images=0, image_token_count=0,
            output_token_count=120, hit=False, response="a response",
            terminal_sha256=None):
    """One replay output record, shaped like the runner writes it."""
    return {
        "family_id": family_id, "variant": variant, "n_images": n_images,
        "image_token_count": image_token_count,
        "output_token_count": output_token_count,
        "hit_max_new_tokens": hit, "response": response,
        "terminal_sha256": (terminal_sha256 if terminal_sha256 is not None
                            else sha256_text(f"q*-{family_id}")),
    }


def _cells(families, **kwargs):
    return [_record(family, variant, **kwargs)
            for family in families
            for variant in ALL_VARIANT_NAMES]


class _GreedyAdapter:
    """Byte-stable per prompt, except on the call numbers in ``drift_on``.

    Stands in for greedy decoding so the determinism evidence can be tested
    from both sides without a GPU.
    """

    def __init__(self, response="a response", drift_on=()):
        self.response = response
        self.drift_on = set(drift_on)
        self.n_calls = 0

    def generate(self, chat_messages):
        self.n_calls += 1
        if self.n_calls in self.drift_on:
            return {"response": f"{self.response} (drifted on call "
                                f"{self.n_calls})"}
        return {"response": self.response}


class TestEligibilityEvidenceComputation:
    """The six gates must be COMPUTED from the run's own records.

    A gate this script merely asserted would be exactly the ``passed: true``
    without evidence that the report schema now refuses, so each of these
    pins the computation from the side that fails as well as the side that
    passes.
    """

    # --- terminal-query invariant -------------------------------------
    def test_canonical_queries_prefer_the_harmonized_q_star(self, tmp_path):
        panel = tmp_path / "validated_families.jsonl"
        panel.write_text("\n".join(json.dumps(rec) for rec in (
            {"family_id": "fam000",
             "validation": {"terminal_harmonization":
                            {"canonical_q": "the canonical q*"}},
             "terminal_query": {"text": "the skeleton original"}},
            {"family_id": "fam001", "terminal_query": {"text": "a dict q"}},
            {"family_id": "fam002", "terminal_query": "a bare string"},
        )) + "\n", encoding="utf-8")
        canonical = run_eligibility.panel_canonical_queries(panel)
        assert canonical == {"fam000": "the canonical q*",
                             "fam001": "a dict q",
                             "fam002": "a bare string"}

    def test_one_terminal_query_per_family_across_all_six_variants(self):
        records = _cells(["fam000", "fam001"])
        evidence = run_eligibility.terminal_query_evidence(
            records, {"fam000": "q*-fam000", "fam001": "q*-fam001"})
        assert evidence["passed"] is True
        assert evidence["n_families_checked"] == 2
        assert evidence["n_mismatched"] == 0

    def test_a_variant_that_changed_the_terminal_query_is_caught(self):
        # This is the confound the invariant exists to rule out: if variant
        # construction altered the question, any difference between variants
        # would be a difference in the question rather than in the
        # intervention.
        records = _cells(["fam000"])
        records[2]["terminal_sha256"] = sha256_text("something else")
        evidence = run_eligibility.terminal_query_evidence(
            records, {"fam000": "q*-fam000"})
        assert evidence["passed"] is False
        assert evidence["n_mismatched"] == 1
        assert "distinct terminal hashes" in evidence["mismatched"][0]

    def test_a_terminal_query_that_is_not_the_panels_q_star_is_caught(self):
        records = _cells(["fam000"])
        evidence = run_eligibility.terminal_query_evidence(
            records, {"fam000": "a DIFFERENT question"})
        assert evidence["passed"] is False
        assert "canonical q*" in evidence["mismatched"][0]

    def test_a_family_absent_from_the_panel_is_caught(self):
        evidence = run_eligibility.terminal_query_evidence(
            _cells(["fam000"]), {})
        assert evidence["passed"] is False
        assert "not in the frozen panel" in evidence["mismatched"][0]

    # --- vision path ---------------------------------------------------
    def test_an_engaged_vision_path_is_evidenced(self):
        records = _cells(["fam000"], n_images=0, image_token_count=0)
        records += _cells(["fam000"], n_images=1, image_token_count=81)
        evidence = run_eligibility.vision_path_evidence(records[6:])
        assert evidence["passed"] is True
        assert evidence["n_image_bearing_cells"] == 6
        assert evidence["min_image_token_count"] == 81

    def test_an_image_bearing_cell_with_no_image_tokens_fails(self):
        # The silent-degradation case: fluent, complete, text-only output
        # from a cross-modal prompt, with every superficial check passing.
        records = _cells(["fam000"], n_images=1, image_token_count=81)
        records[3]["image_token_count"] = 0
        evidence = run_eligibility.vision_path_evidence(records)
        assert evidence["passed"] is False
        assert evidence["n_cells_with_zero_image_tokens"] == 1
        assert evidence["min_image_token_count"] == 0

    def test_an_unmeasured_image_token_count_fails(self):
        records = _cells(["fam000"], n_images=1, image_token_count=81)
        records[0]["image_token_count"] = None
        evidence = run_eligibility.vision_path_evidence(records)
        assert evidence["passed"] is False
        assert evidence["n_cells_without_a_measurement"] == 1

    def test_a_run_that_never_bore_an_image_fails(self):
        evidence = run_eligibility.vision_path_evidence(
            _cells(["fam000"], n_images=0, image_token_count=0))
        assert evidence["passed"] is False
        assert evidence["n_image_bearing_cells"] == 0

    # --- truncation ----------------------------------------------------
    def test_a_complete_run_reports_zero_truncation(self):
        evidence = run_eligibility.truncation_evidence(_cells(["f0", "f1"]))
        assert evidence["passed"] is True
        assert evidence["n_truncated"] == 0
        assert evidence["truncation_rate"] == 0.0
        assert evidence["max_variant_spread"] == 0.0
        assert sorted(evidence["by_variant"]) == sorted(ALL_VARIANT_NAMES)
        assert all(entry["n"] == 2
                   for entry in evidence["by_variant"].values())

    def test_any_truncation_fails_rather_than_warns(self):
        records = _cells(["f0", "f1"])
        records[0]["hit_max_new_tokens"] = True
        records[0]["output_token_count"] = 1536
        evidence = run_eligibility.truncation_evidence(records)
        assert evidence["passed"] is False
        assert evidence["n_truncated"] == 1
        assert evidence["truncation_rate"] == 1 / 12
        assert evidence["max_variant_spread"] == 1 / 2
        assert evidence["by_variant"]["neutral"]["truncated_cells"] == ["f0:neutral"]

    def test_the_spread_is_the_largest_per_variant_rate_difference(self):
        # A global rate can hide condition-specific imbalance, which is why
        # the by-variant breakdown and its spread are both required.
        records = _cells(["f0", "f1", "f2", "f3"])
        truncated = 0
        for record in records:
            if record["variant"] == "cross_modal" and truncated < 2:
                record["hit_max_new_tokens"] = True
                truncated += 1
        evidence = run_eligibility.truncation_evidence(records)
        rates = [e["truncation_rate"] for e in evidence["by_variant"].values()]
        assert evidence["max_variant_spread"] == max(rates) - min(rates)
        assert evidence["by_variant"]["cross_modal"]["n_truncated"] == 2
        assert evidence["by_variant"]["cross_modal"]["truncation_rate"] == 0.5
        assert evidence["by_variant"]["neutral"]["truncation_rate"] == 0.0
        assert evidence["max_variant_spread"] == 0.5

    # --- determinism ---------------------------------------------------
    def test_a_byte_stable_run_is_evidenced(self, tmp_path):
        families = _families(tmp_path, 2)
        adapter = _GreedyAdapter()
        records = _cells([f.family_id for f in families])
        evidence = run_eligibility.determinism_evidence(
            adapter, families, ReplayConfig(max_new_tokens=1536), records, 2)
        assert evidence["passed"] is True
        assert evidence["n_repeats"] == 2
        assert evidence["n_distinct_responses"] == 1
        assert evidence["n_cells_repeated"] == len(records)
        assert evidence["n_unstable_cells"] == 0
        assert adapter.n_calls == len(records)

    def test_a_drifting_cell_is_caught(self, tmp_path):
        families = _families(tmp_path, 2)
        adapter = _GreedyAdapter(drift_on={1})
        records = _cells([f.family_id for f in families])
        evidence = run_eligibility.determinism_evidence(
            adapter, families, ReplayConfig(max_new_tokens=1536), records, 2)
        assert evidence["passed"] is False
        assert evidence["n_distinct_responses"] == 2
        assert evidence["n_unstable_cells"] == 1
        # Cells are visited in SORTED (family_id, variant) order rather
        # than panel order, so which cell a given call number belongs to is
        # reproducible and the drift can be re-found.
        first = min(records, key=lambda r: (r["family_id"], r["variant"]))
        assert evidence["unstable_cells"] == [
            f"{first['family_id']}:{first['variant']}"]
        assert evidence["per_cell"][0]["family_id"] == first["family_id"]

    def test_one_observation_cannot_show_a_difference(self, tmp_path):
        # Which is why the gate requires n_repeats >= 2 rather than merely
        # recording whatever the run happened to do.
        families = _families(tmp_path, 1)
        adapter = _GreedyAdapter(drift_on={1, 2, 3})
        records = _cells([f.family_id for f in families])
        evidence = run_eligibility.determinism_evidence(
            adapter, families, ReplayConfig(max_new_tokens=1536), records, 1)
        assert adapter.n_calls == 0
        assert evidence["n_repeats"] == 1
        assert validate_gate_entry("determinism", evidence)[0] \
            .count("at least two repeats") == 1


# ---------------------------------------------------------------------
# P0-4  Crash-safe journaling and strict resume validation
# ---------------------------------------------------------------------
class _KillingBackend(_StubAdapter):
    """Dies like an OS kill rather than like a generation failure.

    ``_replay_family`` catches ``Exception`` and records it as a failure,
    so a RuntimeError would only test the failure path. ``KeyboardInterrupt``
    derives from ``BaseException`` and therefore propagates out of
    ``run_replay_stage`` the way SIGKILL/preemption does: nothing after the
    raise ever runs, including any end-of-run persistence.
    """

    def __init__(self, model_spec, kill_after):
        super().__init__(model_spec)
        self.kill_after = kill_after
        self.attempts = 0

    def generate(self, chat_messages):
        self.attempts += 1
        if self.attempts > self.kill_after:
            raise KeyboardInterrupt
        return super().generate(chat_messages)


class _ObservingBackend(_StubAdapter):
    """Records the journal's on-disk size before each generation."""

    def __init__(self, model_spec, journal: Path):
        super().__init__(model_spec)
        self.journal = journal
        self.observed_sizes: list[int] = []

    def generate(self, chat_messages):
        self.observed_sizes.append(
            self.journal.stat().st_size if self.journal.exists() else 0)
        return super().generate(chat_messages)


def _journal(tmp_path, name=REPLAY_OUTPUTS_FILE) -> Path:
    return tmp_path / "runs" / "iter11-run" / name


class TestCrashSafeJournal:
    def test_completed_families_survive_a_kill(self, tmp_path):
        _families(tmp_path, 3)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        # Two families complete (12 generations), then the process dies.
        backend = _KillingBackend(spec, kill_after=12)
        with pytest.raises(KeyboardInterrupt):
            run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                             backend=backend, run_id="iter11-run",
                             model_spec=spec)
        # Under the old end-of-run persistence this file was empty and all
        # 12 completed generations were lost.
        durable = _journal(tmp_path).read_text(encoding="utf-8").splitlines()
        assert len(durable) == 12
        assert backend.attempts == 13

    def test_resume_after_a_kill_regenerates_nothing(self, tmp_path):
        _families(tmp_path, 3)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        with pytest.raises(KeyboardInterrupt):
            run_replay_stage(
                tmp_path, tmp_path / "runs", config=config,
                backend=_KillingBackend(spec, kill_after=12),
                run_id="iter11-run", model_spec=spec)
        resumed = _StubAdapter(spec)
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config, backend=resumed,
            run_id="iter11-run", model_spec=spec, resume=True)
        assert resumed.n_calls == 6, "surviving families were regenerated"
        assert report["n_succeeded"] == 18
        assert report["n_failed"] == 0
        assert report["missing_variants"] == []
        assert report["resume"]["n_pairs_resumed"] == 12
        assert len(_journal(tmp_path).read_text(
            encoding="utf-8").splitlines()) == 18

    def test_the_journal_grows_during_the_run_not_at_the_end(self, tmp_path):
        _families(tmp_path, 3)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        journal = _journal(tmp_path)
        backend = _ObservingBackend(spec, journal)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=backend, run_id="iter11-run",
                         model_spec=spec)
        sizes = backend.observed_sizes
        assert len(sizes) == 18
        assert sizes[0] == 0, "nothing should be durable before family 1"
        # By the time family 3 starts, families 1-2 must already be on disk.
        assert sizes[12] > 0
        assert sizes[12] == sizes[-1]

    def test_incremental_writes_match_a_oneshot_write_byte_for_byte(
            self, tmp_path):
        # Evidence continuity: a journaled file must be indistinguishable
        # from one written in a single pass, or pre-existing consumers and
        # manifests would see a format change.
        _families(tmp_path, 2)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id, max_new_tokens=1536)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        journaled = _journal(tmp_path).read_bytes()
        reference = tmp_path / "reference.jsonl"
        write_jsonl(reference, read_jsonl(_journal(tmp_path)))
        assert journaled == reference.read_bytes()

    def test_a_fresh_run_never_appends_onto_stale_evidence(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        journal = _journal(tmp_path)
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text('{"stale": "record"}\n', encoding="utf-8")
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec, overwrite=True)
        text = journal.read_text(encoding="utf-8")
        assert "stale" not in text
        assert len(text.splitlines()) == 6

    def test_an_empty_journal_is_not_treated_as_evidence(self, tmp_path):
        # A run killed before its first family completed leaves empty
        # journals; restarting it must not demand --overwrite.
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_dir = tmp_path / "runs" / "iter11-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in (REPLAY_OUTPUTS_FILE, REPLAY_FAILURES_FILE):
            (run_dir / name).write_text("", encoding="utf-8")
        report = run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                                 backend=_StubAdapter(spec),
                                 run_id="iter11-run", model_spec=spec)
        assert report["n_succeeded"] == 6

    def test_a_completed_run_still_blocks_a_fresh_rerun(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        with pytest.raises(ReplayError, match="already contains evidence"):
            run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                             backend=_StubAdapter(spec), run_id="iter11-run",
                             model_spec=spec)

    def test_failure_history_is_retained_across_interruptions(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        # Every variant of the only family fails.
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec, fail_after=0),
                         run_id="iter11-run", model_spec=spec)
        failures = _journal(tmp_path, REPLAY_FAILURES_FILE)
        assert len(failures.read_text(encoding="utf-8").splitlines()) == 6
        report = run_replay_stage(
            tmp_path, tmp_path / "runs", config=config,
            backend=_StubAdapter(spec), run_id="iter11-run",
            model_spec=spec, resume=True)
        # The earlier attempts are history and stay on disk...
        assert len(failures.read_text(encoding="utf-8").splitlines()) == 6
        assert report["n_failure_attempts"] == 6
        assert report["resume"]["n_prior_failure_attempts_retained"] == 6
        # ...but the cells recovered, so they are not counted as failed.
        assert report["n_failed"] == 0
        assert report["failed_cells"] == []
        assert report["n_succeeded"] == 6

    def test_a_persistently_failing_cell_is_counted_once(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        for _ in range(2):
            report = run_replay_stage(
                tmp_path, tmp_path / "runs", config=config,
                backend=_StubAdapter(spec, fail_after=0),
                run_id="iter11-run", model_spec=spec, resume=True)
        assert report["n_failed"] == 6, "a retried cell must not be counted " \
                                        "once per attempt"
        assert report["n_failure_attempts"] == 12
        assert len(report["failed_cells"]) == 6


class TestResumeValidation:
    def _run_then_tamper(self, tmp_path, mutate, *, failures=False):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        name = REPLAY_FAILURES_FILE if failures else REPLAY_OUTPUTS_FILE
        path = _journal(tmp_path, name)
        records = read_jsonl(path)
        mutate(records)
        write_jsonl(path, records)
        return config, spec

    def _resume(self, tmp_path, config, spec):
        return run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                                backend=_StubAdapter(spec),
                                run_id="iter11-run", model_spec=spec,
                                resume=True)

    def test_a_record_with_no_fingerprint_is_rejected(self, tmp_path):
        # None used to be treated as "compatible with anything", which let
        # records of unknown provenance into a confirmatory run.
        config, spec = self._run_then_tamper(
            tmp_path, lambda recs: recs[0].pop("resolved_run_fingerprint"))
        with pytest.raises(ReplayError,
                           match="has no resolved_run_fingerprint"):
            self._resume(tmp_path, config, spec)

    def test_a_null_fingerprint_is_rejected(self, tmp_path):
        config, spec = self._run_then_tamper(
            tmp_path,
            lambda recs: recs[0].__setitem__("resolved_run_fingerprint", None))
        with pytest.raises(ReplayError,
                           match="has no resolved_run_fingerprint"):
            self._resume(tmp_path, config, spec)

    def test_a_record_with_no_model_key_is_rejected(self, tmp_path):
        config, spec = self._run_then_tamper(
            tmp_path, lambda recs: recs[0].pop("model_key"))
        with pytest.raises(ReplayError, match="has no model_key"):
            self._resume(tmp_path, config, spec)

    def test_a_record_with_no_run_id_is_rejected(self, tmp_path):
        config, spec = self._run_then_tamper(
            tmp_path, lambda recs: recs[0].pop("run_id"))
        with pytest.raises(ReplayError, match="missing required field"):
            self._resume(tmp_path, config, spec)

    def test_an_unknown_variant_is_rejected(self, tmp_path):
        config, spec = self._run_then_tamper(
            tmp_path, lambda recs: recs[0].__setitem__("variant", "seventh"))
        with pytest.raises(ReplayError, match="is not one of"):
            self._resume(tmp_path, config, spec)

    def test_a_non_object_line_is_rejected(self, tmp_path):
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        path = _journal(tmp_path)
        with path.open("a", encoding="utf-8") as f:
            f.write('"not an object"\n')
        with pytest.raises(ReplayError, match="not an object"):
            self._resume(tmp_path, config, spec)

    def test_a_truncated_final_line_is_rejected(self, tmp_path):
        # A kill mid-write leaves a partial JSON line; read_jsonl surfaces
        # it as a decode error rather than as a resumable record.
        _families(tmp_path, 1)
        spec = _spec()
        config = ReplayConfig(model_name=spec.model_id)
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(spec), run_id="iter11-run",
                         model_spec=spec)
        path = _journal(tmp_path)
        with path.open("a", encoding="utf-8") as f:
            f.write('{"run_id": "iter11-run", "family_id"')
        with pytest.raises(Exception):
            self._resume(tmp_path, config, spec)

    def test_the_legacy_path_needs_no_iteration_11_fields(self, tmp_path):
        # Iterations 8-10 records carry no resolved_run_fingerprint or
        # model_key; gating them would break the frozen legacy resume.
        family = _built_family(tmp_path, CLEAN_Q, fill_grounding=True)
        family.family_id = "fam000"
        write_jsonl(tmp_path / "validated_families.jsonl",
                    [family.to_dict()])
        config = ReplayConfig()
        run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                         backend=_StubAdapter(_spec()), run_id="legacy")
        path = tmp_path / "runs" / "legacy" / REPLAY_OUTPUTS_FILE
        records = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert "resolved_run_fingerprint" not in records
        again = _StubAdapter(_spec())
        report = run_replay_stage(tmp_path, tmp_path / "runs", config=config,
                                  backend=again, run_id="legacy",
                                  resume=True)
        assert again.n_calls == 0
        assert report["n_succeeded"] == 6

    def test_duplicate_output_pairs_are_rejected(self):
        record = {"run_id": "r", "family_id": "f", "variant": "neutral",
                  "resolved_run_fingerprint": "x", "model_key": "k"}
        with pytest.raises(ReplayError, match="duplicate stored record"):
            validate_journal([record, dict(record)], path=Path("p.jsonl"),
                             expected_fingerprint="x",
                             expected_model_key="k")

    def test_duplicate_failure_attempts_are_allowed(self):
        record = {"run_id": "r", "family_id": "f", "variant": "neutral",
                  "resolved_run_fingerprint": "x", "model_key": "k"}
        pairs = validate_journal([record, dict(record)], path=Path("p.jsonl"),
                                 expected_fingerprint="x",
                                 expected_model_key="k",
                                 allow_duplicate_pairs=True)
        assert pairs == {("f", "neutral")}

    def test_append_journal_writes_nothing_for_no_records(self, tmp_path):
        path = tmp_path / "j.jsonl"
        append_journal(path, [])
        assert not path.exists()


# ---------------------------------------------------------------------
# P1-5  The dependency lock must be verified, not merely recorded
# ---------------------------------------------------------------------
FREEZE_TEXT = "numpy==1.26.0\ntransformers==5.14.1\npeft==0.20.0\n"
#: The real sibling editable install, as ``pip freeze`` renders it: the
#: revision is MIDP's LIVE git HEAD, which moved five times in forty
#: minutes on 2026-09-05 without anything changing in CCMS.
MIDP_EDITABLE = (
    "-e git+ssh://git@github.com/wutt6678/MIDP.git@{rev}"
    "#egg=route_unlearning_data"
    "&subdirectory=datasets/route-unlearning-data")


class _Completed:
    def __init__(self, stdout=FREEZE_TEXT, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _patch_freeze(monkeypatch, completed):
    monkeypatch.setattr(
        "causal_mllm.replay.registry.subprocess.run",
        lambda *args, **kwargs: completed)


class TestDependencyLockVerification:
    def test_a_failed_pip_freeze_is_an_error(self, monkeypatch):
        # An empty or partial freeze still hashes to a STABLE value, so
        # accepting it would silently certify an unobserved environment.
        _patch_freeze(monkeypatch,
                      _Completed(stdout="", returncode=1,
                                 stderr="pip: command not found"))
        with pytest.raises(ReplayError, match="exited 1"):
            dependency_lock_snapshot()

    def test_the_executable_path_is_not_dependency_identity(
            self, tmp_path, monkeypatch):
        _patch_freeze(monkeypatch, _Completed())
        snapshot = dependency_lock_snapshot()
        here = tmp_path / "here.yaml"
        there = tmp_path / "there.yaml"
        for path, executable in ((here, "/env/a/bin/python"),
                                 (there, "/env/b/bin/python")):
            path.write_text(yaml.safe_dump({"dependency_lock": dict(
                snapshot, executable=executable)}, sort_keys=True),
                encoding="utf-8")
        assert dependency_lock_sha256(here) == dependency_lock_sha256(there)
        # ...but it is still RECORDED as operational metadata.
        assert load_dependency_lock(here)["executable"] == "/env/a/bin/python"
        assert "executable" not in LOCK_IDENTITY_FIELDS

    def test_a_real_package_change_moves_the_hash(self, tmp_path, monkeypatch):
        _patch_freeze(monkeypatch, _Completed())
        before = dependency_lock_snapshot()
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT.replace("5.14.1", "5.15.0")))
        after = dependency_lock_snapshot()
        assert before["pip_freeze_sha256"] != after["pip_freeze_sha256"]

    def test_a_partial_lock_block_is_an_error(self, tmp_path):
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": {
            "pip_freeze_sha256": "f" * 64}}, sort_keys=True),
            encoding="utf-8")
        with pytest.raises(ReplayError, match="missing identity fields"):
            dependency_lock_sha256(path)

    def test_a_corrupt_lock_is_not_read_as_absent(self, tmp_path):
        path = tmp_path / "lock.yaml"
        path.write_text("models: [unclosed\n", encoding="utf-8")
        with pytest.raises(ReplayError, match="unreadable dependency lock"):
            load_dependency_lock(path)

    def test_an_absent_lock_is_none_but_strict_verification_fails(
            self, tmp_path):
        path = tmp_path / "absent.yaml"
        assert load_dependency_lock(path) is None
        assert dependency_lock_sha256(path) is None
        with pytest.raises(ReplayError, match="no dependency lock recorded"):
            verify_active_dependency_lock(path, strict=True)
        result = verify_active_dependency_lock(path, strict=False)
        assert result["verified"] is False
        assert result["reason"] == "no_dependency_lock_recorded"

    def test_drift_is_reported_field_by_field(self, tmp_path, monkeypatch):
        _patch_freeze(monkeypatch, _Completed())
        snapshot = dependency_lock_snapshot()
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": dict(
            snapshot, python_version="3.9.0")}, sort_keys=True),
            encoding="utf-8")
        with pytest.raises(ReplayError, match="python_version"):
            verify_active_dependency_lock(path, strict=True)
        result = verify_active_dependency_lock(path, strict=False)
        assert result["verified"] is False
        assert set(result["differences"]) == {"python_version"}
        assert result["differences"]["python_version"]["locked"] == "3.9.0"
        assert result["differences"]["python_version"]["active"] \
            == snapshot["python_version"]

    def test_a_matching_environment_verifies(self, tmp_path, monkeypatch):
        _patch_freeze(monkeypatch, _Completed())
        snapshot = dependency_lock_snapshot()
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump(
            {"dependency_lock": snapshot}, sort_keys=True), encoding="utf-8")
        result = verify_active_dependency_lock(path, strict=True)
        assert result["verified"] is True
        assert result["differences"] == {}
        assert result["checked_fields"] == list(LOCK_IDENTITY_FIELDS)

    def test_a_third_party_editable_revision_moves_the_hash(
            self, monkeypatch):
        # Reversed from the earlier behaviour, which normalized the revision
        # out of the hashed text. That let an editable dependency's source
        # change while the certified lock hash stood still — the opposite of
        # what a reproducible dependency lock is for.
        before = FREEZE_TEXT + MIDP_EDITABLE.format(rev="a1df9be09a2f") + "\n"
        after = FREEZE_TEXT + MIDP_EDITABLE.format(rev="5e3fef64bcae") + "\n"
        _patch_freeze(monkeypatch, _Completed(stdout=before))
        first = dependency_lock_snapshot()
        _patch_freeze(monkeypatch, _Completed(stdout=after))
        second = dependency_lock_snapshot()
        assert first["pip_freeze_sha256"] != second["pip_freeze_sha256"]
        assert first["editable_vcs_revisions"] \
            == {"route_unlearning_data": "a1df9be09a2f"}
        assert second["editable_vcs_revisions"] \
            == {"route_unlearning_data": "5e3fef64bcae"}

    def test_the_sibling_revision_is_still_recorded(self, monkeypatch):
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="5e3fef64bcae")
            + "\n"))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_vcs_revisions"] \
            == {"route_unlearning_data": "5e3fef64bcae"}
        # Recorded for diagnosis; the freeze hash already binds the revision.
        assert "editable_vcs_revisions" not in LOCK_IDENTITY_FIELDS

    def test_the_editable_install_itself_is_certified(self, monkeypatch):
        # If the sibling package disappeared or moved URL, that is a real
        # dependency change either way.
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="a1df9be09a2f")
            + "\n"))
        with_sibling = dependency_lock_snapshot()
        _patch_freeze(monkeypatch, _Completed(stdout=FREEZE_TEXT))
        without = dependency_lock_snapshot()
        assert with_sibling["pip_freeze_sha256"] != without["pip_freeze_sha256"]
        assert without["editable_vcs_revisions"] == {}

    def _lock_with_sibling(self, tmp_path, monkeypatch, rev="a1df9be09a2f"):
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev=rev) + "\n"))
        snapshot = dependency_lock_snapshot()
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": snapshot},
                                       sort_keys=True), encoding="utf-8")
        return path, snapshot

    def test_a_third_party_editable_install_is_refused(self, tmp_path,
                                                       monkeypatch):
        # The actual protection. pip freeze identifies an editable
        # dependency by the sibling repository's COMMITTED HEAD and is blind
        # to that repository's uncommitted working-tree changes, so no hash
        # of freeze output can prove which dependency source would execute.
        path, _ = self._lock_with_sibling(tmp_path, monkeypatch)
        with pytest.raises(ReplayError,
                           match="third-party editable install"):
            verify_active_dependency_lock(path, strict=True)
        # Non-strict callers still see it, so it can never be silent.
        result = verify_active_dependency_lock(path, strict=False)
        assert result["verified"] is False
        assert result["third_party_editable_vcs"] \
            == {"route_unlearning_data": "a1df9be09a2f"}
        installs = result["third_party_editable_installs"]
        assert installs["route_unlearning_data"]["kind"] == "vcs"
        assert installs["route_unlearning_data"]["revision"] == "a1df9be09a2f"
        assert result["differences"] == {}

    def test_a_lock_matching_a_clean_environment_still_verifies(
            self, tmp_path, monkeypatch):
        # Refusing editable installs must not refuse every environment.
        _patch_freeze(monkeypatch, _Completed())
        snapshot = dependency_lock_snapshot()
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": snapshot},
                                       sort_keys=True), encoding="utf-8")
        result = verify_active_dependency_lock(path, strict=True)
        assert result["verified"] is True
        assert result["third_party_editable_vcs"] == {}

    def test_the_gate_refuses_a_third_party_editable_environment(
            self, world, monkeypatch):
        path, _ = self._lock_with_sibling(world["tmp_path"], monkeypatch)
        lines = _violations(world, lock_path=path)
        assert any("third-party editable install" in line
                   and "route_unlearning_data" in line for line in lines)

    # --- every mutable editable form, not just the VCS one ------------
    @pytest.mark.parametrize("line,kind", [
        ("-e /scratch/somewhere/siblingpkg", "local_path"),
        ("-e ./siblingpkg", "local_path"),
        ("-e ../siblingpkg", "local_path"),
        ("-e .", "local_path"),
        ("-e file:///scratch/somewhere/siblingpkg", "file_url"),
        ("-e git+ssh://git@github.com/x/y.git@abcdef1234#egg=y", "vcs"),
        ("-e svn+http://host/repo#egg=y", "vcs"),
    ])
    def test_every_editable_form_is_detected(self, line, kind):
        # The reported defect: detection required a trailing hex revision,
        # so the MOST mutable forms — a local path, which names no revision
        # at all — were invisible.
        found = editable_installs([line])
        assert len(found) == 1, line
        info = next(iter(found.values()))
        assert info["kind"] == kind
        assert info["line"] == line
        assert (info["revision"] is not None) == (kind == "vcs"
                                                  and "@abcdef1234" in line)

    def test_a_local_path_editable_install_has_no_revision_to_hide_behind(
            self):
        found = editable_installs(["-e /scratch/somewhere/siblingpkg"])
        info = found["siblingpkg"]
        assert info["revision"] is None
        # ...so it is absent from the revision view, which is exactly why
        # that view must not be the detection boundary.
        assert editable_vcs_revisions(
            ["-e /scratch/somewhere/siblingpkg"]) == {}

    @pytest.mark.parametrize("line", [
        "-e /scratch/somewhere/siblingpkg",
        "-e file:///scratch/somewhere/siblingpkg",
        "-e .",
    ])
    def test_a_non_vcs_editable_install_is_refused(self, tmp_path,
                                                  monkeypatch, line):
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + line + "\n"))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_vcs_revisions"] == {}
        assert snapshot["editable_installs"], "the install went undetected"
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": snapshot},
                                       sort_keys=True), encoding="utf-8")
        with pytest.raises(ReplayError,
                           match="third-party editable install"):
            verify_active_dependency_lock(path, strict=True)
        result = verify_active_dependency_lock(path, strict=False)
        assert result["verified"] is False
        assert result["third_party_editable_vcs"] == {}
        assert result["third_party_editable_installs"]

    def test_this_repositorys_own_local_path_install_is_not_third_party(
            self, monkeypatch):
        # Refusing every ``-e`` line must not refuse this project's own
        # editable install, which carries no ``#egg=`` name in path form and
        # so cannot be identified by name alone.
        for line in (f"-e {registry.REPO_ROOT}",
                     f"-e {registry.REPO_ROOT}/",
                     f"-e file://{registry.REPO_ROOT}"):
            _patch_freeze(monkeypatch, _Completed(
                stdout=FREEZE_TEXT + line + "\n"))
            snapshot = dependency_lock_snapshot()
            assert snapshot["editable_installs"] == {}, line
            assert snapshot["excluded_self_distributions"], line

    def test_a_third_party_path_that_merely_contains_the_repo_name_is_not_self(
            self, monkeypatch):
        line = f"-e {registry.REPO_ROOT}-fork"
        _patch_freeze(monkeypatch, _Completed(stdout=FREEZE_TEXT + line + "\n"))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_installs"], (
            "a sibling checkout next to the repository was mistaken for self")

    @pytest.mark.parametrize("line", ["-e .", "-e ./", "-e .."])
    def test_a_relative_editable_target_is_never_identified_as_self(
            self, monkeypatch, line):
        # ``pip freeze`` records the target as given and NOT the directory
        # pip was invoked from, so a relative path cannot be matched to this
        # repository. Resolving it against the verifying process's cwd would
        # make the verdict depend on where the check ran — and would call
        # ``-e .`` self whenever the check ran from the repo, as it does
        # here. Unidentifiable is treated as third-party and refused.
        monkeypatch.chdir(registry.REPO_ROOT)
        _patch_freeze(monkeypatch, _Completed(stdout=FREEZE_TEXT + line + "\n"))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_installs"], (
            f"{line} was resolved against the current cwd and mistaken for "
            f"this repository's own install")
        assert registry.SELF_DISTRIBUTIONS[0] not in (
            snapshot["excluded_self_distributions"])

    def test_pinned_lines_are_not_mistaken_for_editable_installs(self):
        assert editable_installs([
            "numpy==1.26.0",
            "pkg @ file:///scratch/wutiantong/pkg",
            "deadbeef==1.2.3",
        ]) == {}

    def test_pinned_lines_are_not_mistaken_for_vcs_revisions(
            self, monkeypatch):
        # A plain `name @ file:///path` or a hex-looking version must not
        # be normalized away.
        odd = ("pkg @ file:///scratch/wutiantong/pkg\n"
               "deadbeef==1.2.3\n" + FREEZE_TEXT)
        _patch_freeze(monkeypatch, _Completed(stdout=odd))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_vcs_revisions"] == {}
        assert snapshot["n_packages"] == 5

    def test_the_selected_lock_path_reaches_the_fingerprint(
            self, tmp_path, monkeypatch):
        # The CLI resolved revisions from --lock but the fingerprint fell
        # back to the DEFAULT lock, so the bound hash described a file the
        # run never read.
        _families(tmp_path, 1)
        spec = _spec()
        stub = _StubAdapter(spec)
        config = ReplayConfig(max_new_tokens=1536)
        hw = stub.runtime_metadata()["hardware"]
        _patch_freeze(monkeypatch, _Completed())
        snapshot = dependency_lock_snapshot()
        digests = set()
        for name, packages in (("a.yaml", FREEZE_TEXT),
                               ("b.yaml", FREEZE_TEXT + "scipy==1.11.0\n")):
            _patch_freeze(monkeypatch, _Completed(stdout=packages))
            active = dependency_lock_snapshot()
            path = tmp_path / name
            path.write_text(yaml.safe_dump(
                {"dependency_lock": active}, sort_keys=True),
                encoding="utf-8")
            digests.add(iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw, lock_path=path))
        assert len(digests) == 2, "the lock path did not reach the fingerprint"
        assert snapshot["n_packages"] == 3


class TestCommittedPreflightArtifacts:
    """The regenerated artifacts must satisfy every new invariant."""

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_artifact_binds_the_protocol_it_was_checked_against(
            self, model_key):
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists() or not PROTOCOL.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["protocol_sha256"] == protocol_sha256(PROTOCOL)

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_a_passing_artifact_has_no_provenance_problem(self, model_key):
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["status"] != "PASS":
            pytest.skip(f"{model_key} is not a PASS artifact")
        assert artifact["problems"] == []
        assert artifact["git_dirty"] is False
        assert artifact["allow_dirty"] is False
        assert artifact["dataset"]["matches_frozen_protocol"] is True

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_a_passing_artifact_had_no_untracked_source(self, model_key):
        # An untracked module or sitecustomize.py would have changed what
        # executed without appearing in code_commit.
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["status"] != "PASS":
            pytest.skip(f"{model_key} is not a PASS artifact")
        assert artifact["git_dirty_paths"] == []
        assert artifact["git_untracked_paths"] == []
        # The exclusion that keeps a stage from deadlocking itself is narrow
        # and visible, never a blanket ignore of outputs/.
        assert all(
            p.startswith("outputs/iteration_11/preflight/")
            for p in artifact["git_dirty_excluded_own_outputs"])

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_a_passing_artifact_came_from_a_certifiable_environment(
            self, model_key):
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists() or not PROTOCOL.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["status"] != "PASS":
            pytest.skip(f"{model_key} is not a PASS artifact")
        environment = artifact["environment"]
        # No third-party editable install: its source can change without
        # its recorded revision moving. EVERY form counts, not just the ones
        # naming a revision — a local-path editable install names none at
        # all and is the most mutable case.
        if "third_party_editable_installs" not in environment:
            pytest.skip(
                f"{model_key}'s artifact predates the editable-install "
                f"detection field (code_commit "
                f"{artifact.get('code_commit', '?')[:12]}); regenerate it "
                f"with scripts/iter11_model_preflight.py so the recorded "
                f"evidence and the certifying code agree")
        assert environment["third_party_editable_installs"] == {}
        assert environment["third_party_editable_vcs"] == {}
        assert environment["excluded_self_distributions"]
        # Every frozen reference_version still holds in the dedicated env.
        assert environment["observed_versions"] \
            == environment["frozen_reference_versions"]

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_an_environment_name_deviation_is_declared(self, model_key):
        # The frozen protocol names reference_env=midp-qwen35. The invariant
        # is not "the name must differ" but "any difference must be declared,
        # and the frozen file must not have been edited to hide it" - so this
        # stays true if a future environment happens to match.
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists() or not PROTOCOL.exists():
            pytest.skip(f"{path} not present")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["status"] != "PASS":
            pytest.skip(f"{model_key} is not a PASS artifact")
        environment = artifact["environment"]
        frozen_env = json.loads(PROTOCOL.read_text(
            encoding="utf-8"))["dependency_lock"]["reference_env"]
        assert environment["frozen_reference_env"] == frozen_env
        if environment["reference_env_matches_frozen"]:
            assert "reference_env_deviation" not in environment
            return
        deviation = environment["reference_env_deviation"]
        assert frozen_env in deviation["claim"]
        assert environment["conda_env"] in deviation["observation"]
        assert deviation["frozen_protocol_modified"] is False
        assert deviation["rationale"]
        # Whatever the environment is called, the versions are what matter
        # and they are checked rather than assumed.
        assert environment["observed_versions"] \
            == environment["frozen_reference_versions"]
