"""Iteration 11 closeout: the evidence manifest binds hashes, not history.

Scale-C's closeout manifest binds each artifact by SHA-256 AND by the commit that
last touched it. That second binding is a property of the checkout that wrote the
manifest, so a manifest carrying it cannot be re-derived exactly anywhere else,
and on a fresh clone the recorded commits may not be present at all -- leaving a
verifier to choose between failing a correct checkout for its history and
silently dropping the claim.

Iteration 11's binds ``path``, ``sha256`` and ``bytes`` and nothing else, so
``--verify`` REBUILDS the whole document from the files on disk and compares it
against what is filed rather than walking a list of fields somebody remembered to
exclude. Committed-ness is still checked, but as a separate claim against the
object store, so that a checkout with the files and no history -- an export, a
tarball, the anonymous reproducibility package -- reports exit 3, "sound but not
checkable here", instead of exit 1.

What is pinned here:

* the bound set is DISCOVERED from what is committed, not listed, and the
  discovery is repeated at verification time; a file committed under a bound tree
  and left out of the manifest is a finding;
* the document is a pure function of a path list and the bytes on disk -- no
  timestamp, no commit, no tree state, and building it twice gives the same bytes;
* every entry point resolves inside the bound set and names a verifier that
  really implements ``--verify``, because a pointer to a verifier with no verify
  mode is the same claim as a citation to a commit nobody can reach;
* the manifest cannot bind itself, and says why;
* the three-way split between a contradiction (1), an absence (2) and a claim
  this checkout cannot make (3), including that a genuine contradiction still
  outranks the absence of an object store.

CI-safe: no test calls a model or the network. The git questions are answered
against the repository the tests run in, and the blob comparisons read
``HEAD:<path>`` rather than the working tree, so a dirty checkout does not make
them wrong.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


manifest = _load_script("iter11_closeout_evidence_manifest")

OUT = manifest.OUT_PATH
SELF = manifest._rel(OUT)

#: A small bound set of files that are committed whatever else is going on, so
#: the tests that need a manifest on disk do not need the real one.
SMALL = ("README.md", "pyproject.toml", "src/causal_mllm/seeds.py")


#: The manifest binds its own generator, so it cannot be filed until that
#: generator is committed and the tree is clean. The two land in consecutive
#: commits pushed together, and until the second one exists these tests skip
#: rather than pass vacuously.
FILED_LATER = (
    "filed in the commit that follows the one carrying this test: the manifest "
    "binds its own generator, so it cannot be filed until that generator is "
    "committed and the tree is clean")


@pytest.fixture
def filed() -> dict:
    if not OUT.is_file():
        pytest.skip(FILED_LATER)
    return json.loads(OUT.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    done = subprocess.run(("git", *args), cwd=ROOT, capture_output=True)
    assert done.returncode == 0, done.stderr.decode()
    return done.stdout.decode("utf-8", "surrogateescape")


def _write(tmp_path: Path, doc: dict, name: str = "manifest.json") -> Path:
    target = tmp_path / name
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


def _small(monkeypatch) -> dict:
    """A manifest over ``SMALL``, isolated from the state of the working tree.

    The committed-ness claim compares the disk against HEAD. These tests are
    about the document rather than about whether somebody happens to have an
    uncommitted edit to the README, so HEAD is answered from the disk here; the
    real comparison against the object store is what
    ``test_the_filed_manifest_re_derives_exactly`` exercises, and that manifest
    is generated from a clean tree by construction.
    """
    monkeypatch.setattr(manifest, "discover", lambda: list(SMALL))
    doc = manifest.build(list(SMALL), entry_points=[])
    monkeypatch.setattr(
        manifest, "committed_hashes",
        lambda paths: {entry["path"]: entry["sha256"]
                       for entry in doc["bound"] if entry["path"] in paths})
    return doc


def _paths(doc: dict) -> list[str]:
    return [entry["path"] for entry in doc["bound"]]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestTheBoundSetIsDiscoveredRatherThanListed:
    def test_every_tracked_output_is_bound(self):
        paths = set(manifest.discover())
        tracked = {line for line in
                   _git("ls-files", "--", "outputs/iteration_11/").splitlines()
                   if line}
        assert tracked <= paths, sorted(tracked - paths)[:10]

    def test_every_iteration_11_script_and_test_is_bound(self):
        paths = set(manifest.discover())
        for pattern in manifest.BOUND_GLOBS:
            tracked = {line for line in _git("ls-files", "--", pattern)
                       .splitlines() if line}
            assert tracked, f"{pattern} matched nothing, so the pattern is stale"
            assert tracked <= paths, sorted(tracked - paths)[:10]

    def test_the_library_modules_the_verifiers_import_are_bound(self):
        paths = set(manifest.discover())
        assert set(manifest.BOUND_PATHS) <= paths

    def test_nothing_discovered_is_untracked(self):
        for rel in manifest.discover():
            done = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", rel], cwd=ROOT,
                capture_output=True, text=True)
            assert done.returncode == 0, f"{rel} is not tracked"

    def test_the_manifest_does_not_bind_itself(self):
        assert SELF not in manifest.discover()

    def test_discovery_is_sorted_and_duplicate_free(self):
        paths = manifest.discover()
        assert paths == sorted(set(paths))

    def test_an_untracked_file_under_a_bound_tree_is_not_evidence(self, tmp_path):
        """Working-tree bytes are not evidence until they are committed."""
        stray = ROOT / "outputs" / "iteration_11" / "closeout" / "_stray_discovery.json"
        stray.write_text("{}", encoding="utf-8")
        try:
            assert manifest._rel(stray) not in manifest.discover()
        finally:
            stray.unlink()

    def test_an_untracked_named_path_is_refused(self, monkeypatch):
        monkeypatch.setattr(manifest, "BOUND_PATHS", ("no/such/module.py",))
        with pytest.raises(SystemExit) as exc:
            manifest.discover()
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# The document is a pure function
# ---------------------------------------------------------------------------

class TestTheDocumentIsAPureFunctionOfThePathsAndTheDisk:
    def test_building_twice_yields_the_same_bytes(self):
        first = json.dumps(manifest.build(list(SMALL)), indent=2, sort_keys=True)
        second = json.dumps(manifest.build(list(SMALL)), indent=2, sort_keys=True)
        assert first == second

    def test_nothing_in_it_records_when_or_where_it_was_written(self):
        banned = {"generated_at", "timestamp", "git_commit", "git_dirty",
                  "code_commit", "python", "host", "generated_from_commit",
                  "code_tree_dirty"}

        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key not in banned, f"{path}/{key}"
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(manifest.build(list(SMALL)), "")

    def test_the_counts_and_the_roll_up_follow_the_entries(self):
        doc = manifest.build(list(SMALL))
        assert doc["n_bound"] == len(doc["bound"])
        assert doc["total_bytes"] == sum(e["bytes"] for e in doc["bound"])
        assert doc["rollup_sha256"] == manifest.roll_up(doc["bound"])
        counts: dict[str, int] = {}
        for entry in doc["bound"]:
            counts[entry["role"]] = counts.get(entry["role"], 0) + 1
        assert doc["n_by_role"] == dict(sorted(counts.items()))

    def test_every_bound_hash_is_the_hash_of_the_bytes_on_disk(self):
        for entry in manifest.build(list(SMALL))["bound"]:
            path = ROOT / entry["path"]
            assert entry["sha256"] == hashlib.sha256(
                path.read_bytes()).hexdigest()
            assert entry["bytes"] == path.stat().st_size

    def test_a_bound_path_that_is_not_a_file_is_an_absence_not_a_mismatch(self):
        with pytest.raises(SystemExit) as exc:
            manifest.build([*SMALL, "no/such/file.json"])
        assert exc.value.code == 2

    def test_the_roll_up_is_over_path_hash_and_size_of_every_entry(self):
        doc = manifest.build(list(SMALL))
        expected = hashlib.sha256()
        for entry in doc["bound"]:
            expected.update(f"{entry['path']}\n{entry['sha256']}\n"
                            f"{entry['bytes']}\n".encode("utf-8"))
        assert doc["rollup_sha256"] == expected.hexdigest()

    def test_a_roll_up_that_does_not_summarise_its_entries_is_refused(self):
        doc = manifest.build(list(SMALL))
        doc["rollup_sha256"] = "0" * 64
        assert any("rollup_sha256" in i for i in
                   manifest.check_the_manifest(doc))

    def test_a_forged_entry_hash_moves_the_roll_up(self):
        doc = manifest.build(list(SMALL))
        forged = copy.deepcopy(doc)
        forged["bound"][0]["sha256"] = "f" * 64
        assert manifest.roll_up(forged["bound"]) != doc["rollup_sha256"], (
            "a roll-up that does not move when an entry moves binds nothing")


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class TestARoleIsDerivedFromWhereAFileLives:
    @pytest.mark.parametrize("rel,role", [
        ("outputs/iteration_11/protocol/iteration_11_protocol.json",
         "frozen_protocol"),
        ("outputs/iteration_11/preflight/ministral3_3b/preflight.json",
         "environment_and_model_preflight"),
        ("outputs/iteration_11/preflight/dependency_lock_reconstruction.json",
         "environment_reconstruction_claim"),
        ("outputs/iteration_11/eligibility/selection.json", "eligibility_gate"),
        ("outputs/iteration_11/generations/phi4_mm/replay_report.json",
         "model_generation_metadata"),
        ("outputs/iteration_11/judge/ministral3_3b/llm_labels_judge_A.json",
         "judge_labels_and_provenance"),
        ("outputs/iteration_11/judge_vision_ablation/ablated_labels.jsonl",
         "judge_vision_ablation"),
        ("outputs/iteration_11/analysis/cross_model/cross_model_analysis.json",
         "sealed_analysis"),
        ("outputs/iteration_11/analysis/differential_censoring/"
         "differential_censoring_bound.json",
         "differential_censoring_sensitivity"),
        ("outputs/iteration_11/closeout/"
         "iteration_11_transportability_decision.json", "closeout"),
        ("outputs/iteration_11/diagnostics/truncation/truncation_evidence.json",
         "diagnostics"),
        ("outputs/iteration_11/reports/.gitkeep", "placeholder"),
        ("outputs/iteration_11/media_manifest.json",
         "binding_for_the_untracked_media"),
        ("scripts/iter11_replay_checks.py", "verifier_or_generator"),
        ("tests/unit/test_iter11_protocol.py", "test"),
        ("src/causal_mllm/replay/reproduction.py",
         "library_the_verifiers_import"),
        ("somewhere/else.json", "evidence"),
    ])
    def test_the_role_follows_the_path(self, rel, role):
        assert manifest.role_for(rel) == role

    def test_every_bound_entry_carries_its_own_paths_role(self):
        doc = manifest.build(manifest.discover(), entry_points=[])
        assert manifest.check_the_manifest(doc) == []

    def test_a_role_that_disagrees_with_its_path_is_refused(self):
        doc = manifest.build(list(SMALL))
        doc["bound"][0]["role"] = "frozen_protocol"
        issues = manifest.check_the_manifest(doc)
        assert any("filed under role" in i for i in issues), issues

    def test_a_role_count_that_disagrees_with_the_entries_is_refused(self):
        doc = manifest.build(list(SMALL))
        doc["n_by_role"] = {"evidence": 99}
        assert any("n_by_role" in i for i in manifest.check_the_manifest(doc))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

class TestAnEntryPointResolvesAndNamesAVerifierThatCanVerify:
    def test_the_filed_entry_points_are_all_bound(self, filed):
        bound = set(_paths(filed))
        assert filed["where_a_reviewer_starts"], (
            "a closeout that points nowhere is a pile of hashes")
        for point in filed["where_a_reviewer_starts"]:
            assert point["path"] in bound, point["path"]
            assert point["verified_by"] in bound, point["verified_by"]
            assert point["what_it_establishes"].strip()

    def test_every_named_verifier_really_implements_verify(self):
        for point in manifest.ENTRY_POINTS:
            source = (ROOT / point["verified_by"]).read_text(encoding="utf-8")
            assert '"--verify"' in source, (
                f"{point['verified_by']} is named as what verifies "
                f"{point['path']} and has no --verify mode")

    def test_an_entry_point_outside_the_bound_set_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = [{
            "path": "outputs/iteration_11/nothing_here.json",
            "what_it_establishes": "nothing",
            "verified_by": "scripts/iter11_replay_checks.py"}]
        issues = manifest.check_the_manifest(doc)
        assert any("leaves the manifest's guarantee" in i for i in issues), issues

    def test_an_unbound_verifier_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = [{
            "path": "README.md", "what_it_establishes": "nothing",
            "verified_by": "scripts/iter11_replay_checks.py"}]
        issues = manifest.check_the_manifest(doc)
        assert any("that script is not bound" in i for i in issues), issues

    def test_a_verifier_with_no_verify_mode_is_refused(self, monkeypatch):
        """The failure mode this check exists for: a pointer that goes nowhere."""
        doc = _small(monkeypatch)
        doc["bound"].append({
            "path": "scripts/iter11_freeze_protocol.py",
            "sha256": manifest.sha256_file(
                ROOT / "scripts" / "iter11_freeze_protocol.py"),
            "bytes": (ROOT / "scripts" / "iter11_freeze_protocol.py")
            .stat().st_size,
            "role": "verifier_or_generator"})
        doc["n_bound"] = len(doc["bound"])
        doc["where_a_reviewer_starts"] = [{
            "path": "README.md", "what_it_establishes": "nothing",
            "verified_by": "scripts/iter11_freeze_protocol.py"}]
        issues = manifest.check_the_manifest(doc)
        assert any("implements no --verify mode" in i for i in issues), issues


# ---------------------------------------------------------------------------
# Committed-ness, read out of the object store
# ---------------------------------------------------------------------------

class TestCommittednessIsASeparateClaim:
    def test_the_head_tree_maps_paths_to_blob_ids(self):
        tree = manifest._head_tree()
        assert tree, "no object store, so this test cannot say anything"
        assert "README.md" in tree
        assert len(tree["README.md"]) == 40

    def test_a_committed_file_hashes_the_same_from_its_blob(self):
        paths = ["README.md", "pyproject.toml"]
        at_head = manifest.committed_hashes(paths)
        for rel in paths:
            content = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                                     capture_output=True).stdout
            assert at_head[rel] == hashlib.sha256(content).hexdigest(), rel

    def test_a_path_head_does_not_hold_is_none_not_an_error(self):
        at_head = manifest.committed_hashes(["README.md", "no/such/file.json"])
        assert at_head["no/such/file.json"] is None
        assert at_head["README.md"] is not None

    def test_a_missing_object_id_is_parsed_as_missing(self):
        assert manifest._blob_sha256s(["0" * 40]) == {"0" * 40: None}

    def test_an_empty_batch_makes_no_call(self):
        assert manifest._blob_sha256s([]) == {}

    def test_two_blobs_in_one_batch_are_not_confused(self):
        """``--batch`` is length-prefixed, so an off-by-one loses the second."""
        tree = manifest._head_tree()
        pair = [rel for rel in ("README.md", "pyproject.toml") if rel in tree]
        oids = [tree[rel] for rel in pair]
        hashed = manifest._blob_sha256s(oids)
        assert len(hashed) == len(pair)
        for rel, oid in zip(pair, oids):
            content = subprocess.run(["git", "cat-file", "blob", oid], cwd=ROOT,
                                     capture_output=True).stdout
            assert hashed[oid] == hashlib.sha256(content).hexdigest(), rel

    def test_the_object_store_is_absent_where_there_is_no_git(self, tmp_path,
                                                              monkeypatch):
        monkeypatch.setattr(manifest, "REPO_ROOT", tmp_path)
        assert manifest.object_store_available() is False

    def test_discovery_without_an_object_store_says_so(self, monkeypatch):
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        with pytest.raises(SystemExit) as exc:
            manifest.discover()
        assert exc.value.code == 3, (
            "not being able to enumerate what is committed is a limit of the "
            "checkout, not a defect in the evidence")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_a_freshly_built_manifest_verifies(self, tmp_path, monkeypatch):
        doc = _small(monkeypatch)
        target = _write(tmp_path, doc)
        code, conclusion, issues, unverifiable = manifest.verify(target)
        assert code == 0, (issues, unverifiable)
        assert conclusion == "re_derived_exactly_and_every_file_is_committed"
        assert issues == [] and unverifiable == []

    def test_a_bound_file_that_moves_is_a_contradiction(self, tmp_path,
                                                        monkeypatch):
        doc = _small(monkeypatch)
        doc["bound"][0]["sha256"] = "0" * 64
        code, _, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 1
        assert any("does not re-derive" in i or "sha256" in i or "bound" in i
                   for i in issues), issues

    def test_a_bound_file_that_disappears_is_an_absence(self, tmp_path,
                                                        monkeypatch):
        doc = _small(monkeypatch)
        doc["bound"] = [entry for entry in doc["bound"]
                        if entry["path"] != "README.md"]
        doc["bound"].append({"path": "no/such/file.json", "sha256": "0" * 64,
                             "bytes": 1, "role": "evidence"})
        doc["n_bound"] = len(doc["bound"])
        code, conclusion, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 2, (conclusion, issues)
        assert conclusion == "bound_files_absent"

    def test_nothing_filed_to_verify_against_is_its_own_answer(self, tmp_path):
        code, conclusion, issues, _ = manifest.verify(tmp_path / "absent.json")
        assert code == 2 and conclusion == "not_filed"
        assert "--write" in issues[0]

    def test_a_committed_file_left_out_of_the_manifest_is_a_finding(
            self, tmp_path, monkeypatch):
        doc = _small(monkeypatch)
        doc["bound"] = [entry for entry in doc["bound"]
                        if entry["path"] != "pyproject.toml"]
        doc["n_bound"] = len(doc["bound"])
        doc["total_bytes"] = sum(e["bytes"] for e in doc["bound"])
        doc["rollup_sha256"] = manifest.roll_up(doc["bound"])
        counts: dict[str, int] = {}
        for entry in doc["bound"]:
            counts[entry["role"]] = counts.get(entry["role"], 0) + 1
        doc["n_by_role"] = dict(sorted(counts.items()))
        code, _, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 1
        assert any("are not in this manifest" in i for i in issues), issues

    def test_a_bound_file_that_is_not_tracked_is_a_finding(self, tmp_path,
                                                           monkeypatch):
        stray = ROOT / "outputs" / "iteration_11" / "closeout" / "_stray_binding.json"
        stray.write_text("{}", encoding="utf-8")
        try:
            doc = _small(monkeypatch)
            rel = manifest._rel(stray)
            doc["bound"].append({
                "path": rel, "sha256": manifest.sha256_file(stray),
                "bytes": stray.stat().st_size, "role": "closeout"})
            doc["n_bound"] = len(doc["bound"])
            doc["total_bytes"] = sum(e["bytes"] for e in doc["bound"])
            doc["rollup_sha256"] = manifest.roll_up(doc["bound"])
            counts: dict[str, int] = {}
            for entry in doc["bound"]:
                counts[entry["role"]] = counts.get(entry["role"], 0) + 1
            doc["n_by_role"] = dict(sorted(counts.items()))
            code, _, issues, _ = manifest.verify(_write(tmp_path, doc))
            assert code == 1
            assert any("not tracked" in i or "are not in HEAD" in i
                       for i in issues), issues
        finally:
            stray.unlink()

    def test_verify_writes_nothing(self, tmp_path, monkeypatch):
        doc = _small(monkeypatch)
        target = _write(tmp_path, doc)
        before = target.read_bytes()
        manifest.verify(target)
        assert target.read_bytes() == before

    def test_the_filed_manifest_re_derives_exactly(self, filed):
        code, conclusion, issues, unverifiable = manifest.verify()
        assert code == 0, (conclusion, issues, unverifiable)
        assert conclusion == "re_derived_exactly_and_every_file_is_committed"
        assert filed["n_bound"] > 100, (
            "the bound set is discovered from what is committed under "
            "outputs/iteration_11 plus every iteration-11 script and test; a "
            "manifest binding fewer than that has stopped discovering")

    def test_the_filed_manifest_agrees_with_itself(self, filed):
        assert manifest.check_the_manifest(filed) == []

    def test_the_filed_manifest_binds_the_transportability_decision(self, filed):
        bound = set(_paths(filed))
        assert "outputs/iteration_11/closeout/" \
               "iteration_11_transportability_decision.json" in bound
        assert "scripts/iter11_transportability_decision.py" in bound
        assert "scripts/iter11_closeout_evidence_manifest.py" in bound, (
            "the manifest binds its own generator: a document whose producer is "
            "not pinned cannot be re-derived by a reviewer who has only what "
            "the manifest vouches for")
        assert SELF not in bound


# ---------------------------------------------------------------------------
# The exit-3 lane: sound evidence, checkout that cannot say so
# ---------------------------------------------------------------------------

class TestACheckoutWithoutHistoryIsIncompleteRatherThanWrong:
    def test_the_hashes_still_check_and_the_answer_is_three(self, tmp_path,
                                                            monkeypatch):
        doc = _small(monkeypatch)
        target = _write(tmp_path, doc)
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        code, conclusion, issues, unverifiable = manifest.verify(target)
        assert code == 3, (conclusion, issues)
        assert conclusion == "hashes_sound_committedness_uncheckable"
        assert issues == [], (
            "a checkout without an object store has contradicted nothing")
        assert len(unverifiable) == 1
        assert "no git object store" in unverifiable[0]
        assert "blob at HEAD" in unverifiable[0]
        assert "left out of this manifest" in unverifiable[0]

    def test_three_names_both_claims_it_could_not_make(self, tmp_path,
                                                       monkeypatch):
        doc = _small(monkeypatch)
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        _, _, _, unverifiable = manifest.verify(_write(tmp_path, doc))
        note = unverifiable[0]
        assert "Every hash on disk checks out" in note

    def test_a_contradiction_still_outranks_the_missing_history(self, tmp_path,
                                                                monkeypatch):
        doc = _small(monkeypatch)
        doc["n_bound"] = len(doc["bound"]) + 1
        target = _write(tmp_path, doc)
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        code, _, issues, unverifiable = manifest.verify(target)
        assert code == 1, (
            "a finding outranks incompleteness: reporting 3 here would let a "
            "wrong manifest hide behind a checkout that happens to lack history")
        assert any("n_bound" in i for i in issues), issues
        assert unverifiable, "what could not be checked is still reported"

    def test_a_forged_hash_is_a_contradiction_even_without_history(
            self, tmp_path, monkeypatch):
        doc = _small(monkeypatch)
        doc["bound"][0]["sha256"] = "e" * 64
        doc["rollup_sha256"] = manifest.roll_up(doc["bound"])
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        code, _, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 1
        assert issues


# ---------------------------------------------------------------------------
# The clean-tree precondition, and the exit codes
# ---------------------------------------------------------------------------

class TestThePreconditionsAndTheExitCodes:
    def test_fatal_exits_with_the_code_it_is_given(self):
        with pytest.raises(SystemExit) as exc:
            manifest.fatal("nothing is filed", 2)
        assert exc.value.code == 2, (
            "raise SystemExit('text') exits 1 whatever the text says, and 1 "
            "means a contradiction in this script")

    def test_a_dirty_tree_refuses_to_file(self, monkeypatch):
        monkeypatch.setattr(
            manifest, "_git",
            lambda *a: subprocess.CompletedProcess(
                a, 0, " M README.md\n?? outputs/iteration_11/stray.json\n", ""))
        with pytest.raises(SystemExit) as exc:
            manifest._assert_clean_tree()
        assert exc.value.code == 1

    def test_its_own_unwritten_output_does_not_make_the_tree_dirty(
            self, monkeypatch):
        monkeypatch.setattr(
            manifest, "_git",
            lambda *a: subprocess.CompletedProcess(
                a, 0, f"?? {SELF}\n", ""))
        manifest._assert_clean_tree()

    def test_a_dirty_file_beside_it_does(self, monkeypatch):
        """Excluding the whole closeout directory would hide a moved decision."""
        monkeypatch.setattr(
            manifest, "_git",
            lambda *a: subprocess.CompletedProcess(
                a, 0, " M outputs/iteration_11/closeout/"
                      "iteration_11_transportability_decision.json\n", ""))
        with pytest.raises(SystemExit) as exc:
            manifest._assert_clean_tree()
        assert exc.value.code == 1

    def test_a_failed_git_status_is_not_a_clean_tree(self, monkeypatch):
        monkeypatch.setattr(
            manifest, "_git",
            lambda *a: subprocess.CompletedProcess(a, 128, "", "not a repository"))
        with pytest.raises(SystemExit) as exc:
            manifest._assert_clean_tree()
        assert exc.value.code == 1

    def test_zero_is_the_answer_for_a_sound_manifest(self, tmp_path, monkeypatch):
        assert manifest.main(
            ["--verify", "--out", str(_write(tmp_path, _small(monkeypatch)))]) == 0

    def test_one_is_the_answer_for_a_contradiction(self, tmp_path, monkeypatch):
        doc = _small(monkeypatch)
        doc["n_by_role"] = {}
        assert manifest.main(
            ["--verify", "--out", str(_write(tmp_path, doc))]) == 1

    def test_two_is_the_answer_for_an_absence(self, tmp_path):
        assert manifest.main(
            ["--verify", "--out", str(tmp_path / "absent.json")]) == 2

    def test_three_is_the_answer_for_a_checkout_without_history(
            self, tmp_path, monkeypatch):
        target = _write(tmp_path, _small(monkeypatch))
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        assert manifest.main(["--verify", "--out", str(target)]) == 3

    def test_write_and_verify_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            manifest.main(["--verify", "--write"])
        assert exc.value.code == 2

    def test_the_documented_codes_are_the_ones_the_script_uses(self):
        source = (ROOT / "scripts" / "iter11_closeout_evidence_manifest.py") \
            .read_text(encoding="utf-8")
        for code in ("0", "1", "2", "3"):
            assert f"\n    {code}  " in source, (
                f"exit {code} is not documented in the module docstring")

    def test_what_is_not_bound_is_said_rather_than_left_implicit(self):
        doc = manifest.build(list(SMALL))
        not_bound = doc["what_this_manifest_does_not_bind"]
        assert not_bound["itself"]["path"] == SELF
        assert not_bound["the_media"]["bound_indirectly_by"] == \
            "outputs/iteration_11/media_manifest.json"
        assert "no generated_at" in not_bound["commit_and_tree_state"]["why"]

    def test_the_media_are_bound_by_the_manifest_that_is_bound_here(self):
        media = manifest.media_binding()
        assert media["readable_here"] is True
        assert media["n_files_it_binds"] > 0
        assert media["its_rollup_sha256"]
        assert media["panel_referenced_but_absent_from_the_manifest"] == []
