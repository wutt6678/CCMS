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
from causal_mllm.replay import confirmatory, registry
from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.confirmatory import (
    GENERATIONS_ROOT,
    enforce_confirmatory_protocol,
    protocol_sha256,
)
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import (
    LOCK_IDENTITY_FIELDS,
    ResolvedModel,
    dependency_lock_sha256,
    dependency_lock_snapshot,
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


def _tree(dirty, dirty_paths=(), excluded_paths=()) -> dict:
    """A ``causal_mllm.seeds.code_tree_status`` result."""
    return {"dirty": dirty, "dirty_paths": list(dirty_paths),
            "excluded_paths": list(excluded_paths)}


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
        tree = _tree(False, excluded_paths=[
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json",
            "outputs/iteration_11/preflight/resolved_models.lock.yaml"])
        result = preflight.git_provenance(COMMIT, tree, False)
        assert result["abort"] is False
        assert result["problems"] == []
        # Excluded paths are surfaced, never silently dropped.
        assert len(result["excluded_paths"]) == 2

    def test_a_dirty_code_path_is_not_excused_by_the_exclusion(self):
        tree = _tree(True, dirty_paths=["src/causal_mllm/replay/runner.py"],
                     excluded_paths=["outputs/iteration_11/preflight/x.json"])
        result = preflight.git_provenance(COMMIT, tree, False)
        assert result["abort"] is True
        assert result["dirty_paths"] == ["src/causal_mllm/replay/runner.py"]


class TestCodeTreeStatus:
    def test_own_output_prefixes_are_excluded_but_reported(self, monkeypatch):
        monkeypatch.setattr(seeds, "dirty_tracked_files", lambda: [
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json",
            "outputs/iteration_11/preflight/resolved_models.lock.yaml",
            "src/causal_mllm/replay/runner.py",
        ])
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/preflight/",))
        assert status["dirty"] is True
        assert status["dirty_paths"] == ["src/causal_mllm/replay/runner.py"]
        assert len(status["excluded_paths"]) == 2

    def test_only_own_outputs_dirty_means_clean(self, monkeypatch):
        monkeypatch.setattr(seeds, "dirty_tracked_files", lambda: [
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json"])
        status = seeds.code_tree_status(
            exclude_prefixes=("outputs/iteration_11/preflight/",))
        assert status["dirty"] is False
        assert status["dirty_paths"] == []
        assert status["excluded_paths"] == [
            "outputs/iteration_11/preflight/qwen35_2b/preflight.json"]

    def test_an_unavailable_git_status_is_none(self, monkeypatch):
        monkeypatch.setattr(seeds, "dirty_tracked_files", lambda: None)
        status = seeds.code_tree_status(exclude_prefixes=("outputs/",))
        assert status["dirty"] is None
        assert status["dirty_paths"] == []

    def test_no_exclusions_matches_the_unscoped_answer(self, monkeypatch):
        paths = ["README.md", "src/causal_mllm/seeds.py"]
        monkeypatch.setattr(seeds, "dirty_tracked_files", lambda: paths)
        status = seeds.code_tree_status()
        assert status["dirty"] is True
        assert status["dirty_paths"] == paths
        assert status["excluded_paths"] == []

    def test_the_live_repository_reports_its_own_state(self):
        # Not mocked: proves the porcelain parsing works against real git
        # output in this repository.
        files = seeds.dirty_tracked_files()
        assert files is None or isinstance(files, list)
        status = seeds.code_tree_status()
        assert status["dirty"] in (True, False, None)
        if files is not None:
            assert status["dirty"] == bool(files)


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
        digests = set()
        for value in (False, True, None):
            monkeypatch.setattr(
                "causal_mllm.replay.runner.is_git_dirty", lambda v=value: v)
            digests.add(iteration11_run_fingerprint(
                stub, config, tmp_path, spec, hw))
        assert len(digests) == 3


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


def _write_eligibility(root: Path, model_key: str, protocol_path: Path, *,
                       status="PASS", eligible=True, revision=REVISION,
                       git_dirty=False) -> Path:
    path = root / model_key / "preflight_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": status, "eligible": eligible, "model_key": model_key,
        "model_revision": revision,
        "protocol_sha256": protocol_sha256(protocol_path),
        "code_commit": COMMIT, "git_dirty": git_dirty,
    }, indent=2), encoding="utf-8")
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
    _write_eligibility(eligibility_root, "qwen35_4b", protocol_path)
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
    }
    kwargs.update(overrides)
    return enforce_confirmatory_protocol(**kwargs)


def _violations(world, **overrides) -> list[str]:
    with pytest.raises(ReplayError) as exc:
        _gate(world, **overrides)
    return str(exc.value).splitlines()


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
        assert any("working tree is dirty" in line for line in lines)
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
            return _tree(False, excluded_paths=[
                "outputs/iteration_11/generations/qwen35_4b/r/replay_outputs"
                ".jsonl"])

        monkeypatch.setattr(confirmatory, "code_tree_status", fake)
        result = _gate(world)
        assert result["passed"] is True
        assert seen["prefixes"] == confirmatory.OWN_OUTPUT_PREFIXES
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
        _write_eligibility(world["eligibility_root"], "qwen35_4b",
                           world["protocol_path"], status="FAIL")
        lines = _violations(world)
        assert any("not 'PASS'" in line for line in lines)

    def test_eligible_false_is_rejected(self, world):
        _write_eligibility(world["eligibility_root"], "qwen35_4b",
                           world["protocol_path"], eligible=False)
        lines = _violations(world)
        assert any("eligible=true" in line for line in lines)

    def test_eligibility_for_another_revision_does_not_transfer(self, world):
        _write_eligibility(world["eligibility_root"], "qwen35_4b",
                           world["protocol_path"], revision="e" * 40)
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

    def test_eligibility_from_a_dirty_tree_is_rejected(self, world):
        _write_eligibility(world["eligibility_root"], "qwen35_4b",
                           world["protocol_path"], git_dirty=True)
        lines = _violations(world)
        assert any("not produced from a clean tree" in line
                   for line in lines)

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
        for fragment in ("overwrite=True", "working tree is dirty",
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

    def test_a_sibling_editable_revision_does_not_move_the_hash(
            self, monkeypatch):
        # MIDP moved five commits in forty minutes while nothing about CCMS
        # inference changed; each move silently changed pip_freeze_sha256
        # and, with the gate enforcing it, would have blocked every
        # confirmatory run for a reason unrelated to the experiment.
        before = FREEZE_TEXT + MIDP_EDITABLE.format(rev="a1df9be09a2f") + "\n"
        after = FREEZE_TEXT + MIDP_EDITABLE.format(rev="5e3fef64bcae") + "\n"
        _patch_freeze(monkeypatch, _Completed(stdout=before))
        first = dependency_lock_snapshot()
        _patch_freeze(monkeypatch, _Completed(stdout=after))
        second = dependency_lock_snapshot()
        assert first["pip_freeze_sha256"] == second["pip_freeze_sha256"]
        assert first["n_packages"] == second["n_packages"] == 4

    def test_the_sibling_revision_is_still_recorded(self, monkeypatch):
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="5e3fef64bcae")
            + "\n"))
        snapshot = dependency_lock_snapshot()
        assert snapshot["editable_vcs_revisions"] \
            == {"route_unlearning_data": "5e3fef64bcae"}
        # Recorded as operational metadata, not as identity.
        assert "editable_vcs_revisions" not in LOCK_IDENTITY_FIELDS

    def test_the_editable_install_itself_is_still_certified(self, monkeypatch):
        # Normalizing the REVISION must not normalize away the install: if
        # the sibling package disappeared or moved URL, that is a real
        # dependency change.
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="a1df9be09a2f")
            + "\n"))
        with_sibling = dependency_lock_snapshot()
        _patch_freeze(monkeypatch, _Completed(stdout=FREEZE_TEXT))
        without = dependency_lock_snapshot()
        assert with_sibling["pip_freeze_sha256"] != without["pip_freeze_sha256"]

    def test_operational_drift_is_reported_but_not_fatal(self, tmp_path,
                                                         monkeypatch):
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="a1df9be09a2f")
            + "\n"))
        snapshot = dependency_lock_snapshot()
        path = tmp_path / "lock.yaml"
        path.write_text(yaml.safe_dump({"dependency_lock": snapshot},
                                       sort_keys=True), encoding="utf-8")
        # The sibling repository moves on; the identity does not.
        _patch_freeze(monkeypatch, _Completed(
            stdout=FREEZE_TEXT + MIDP_EDITABLE.format(rev="5e3fef64bcae")
            + "\n"))
        result = verify_active_dependency_lock(path, strict=True)
        assert result["verified"] is True
        assert result["differences"] == {}
        assert set(result["informational_differences"]) \
            == {"editable_vcs_revisions"}

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
