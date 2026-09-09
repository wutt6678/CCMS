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
from types import SimpleNamespace

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


def _a_complete_small_bound_set() -> list[str]:
    """``SMALL``, plus everything a manifest has to bind to be a valid one.

    Since the entry-point comparison became exact rather than a re-derivation of
    whatever the artifact filed, a document that binds a SUBSET of the entry
    points is refused by design -- that refusal is the fix, not an obstacle. So a
    fixture that wants a manifest ``check_the_manifest`` accepts has to carry all
    eight entry points, the verifiers they name, and the import closure those
    verifiers reach. Forty-odd small files: still fast to hash, and now the
    fixture exercises a document that could actually have been filed.
    """
    base = set(SMALL)
    base.update(point["path"] for point in manifest.ENTRY_POINTS)
    base.update(point["verified_by"] for point in manifest.ENTRY_POINTS)
    closure = manifest.dependency_closure(
        sorted(path for path in base if path.endswith(".py")))
    return sorted(base | set(closure["modules"]))


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
    """A valid manifest over a small bound set, isolated from the working tree.

    The committed-ness claim compares the disk against HEAD. These tests are
    about the document rather than about whether somebody happens to have an
    uncommitted edit to the README, so HEAD is answered from the disk here; the
    real comparison against the object store is what
    ``test_the_filed_manifest_re_derives_exactly`` exercises, and that manifest
    is generated from a clean tree by construction.
    """
    paths = _a_complete_small_bound_set()
    monkeypatch.setattr(manifest, "discover", lambda: paths)
    doc = manifest.build(paths, manifest.ENTRY_POINTS)

    def answered_from_disk(wanted: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel in wanted:
            candidate = Path(rel)
            # Absolute for the manifest under test, which lives in tmp_path and
            # is compared against its own blob as part of verification.
            on_disk = candidate if candidate.is_absolute() else ROOT / rel
            if on_disk.is_file():
                out[rel] = manifest.sha256_file(on_disk)
        return out

    monkeypatch.setattr(manifest, "committed_hashes", answered_from_disk)
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
        assert sorted(tracked - paths) == [SELF], (
            "the manifest binds its own path in nothing, so it is the only "
            f"tracked output it may leave out; left out: "
            f"{sorted(tracked - paths)}")
        assert tracked - {SELF} <= paths

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
        doc = manifest.build(manifest.discover(), manifest.ENTRY_POINTS)
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

    def test_the_papers_printed_tables_are_a_gate_of_their_own(self):
        """The renderings are the last derived thing in the chain.

        A table is what a reader actually looks at, so that it was derived rather
        than typed is worth an entry point of its own: the numbers file being right
        does not by itself make the print right, and a renderer that rounded
        differently from the spec it was given would leave every upstream gate
        passing.
        """
        points = {point["path"]: point for point in manifest.ENTRY_POINTS}
        assert "paper/tables/renderings.json" in points
        assert points["paper/tables/renderings.json"]["verified_by"] == \
            "scripts/iter11_paper_tables.py"
        assert "paper/numbers/iteration_11_paper_numbers.json" in points

    def test_the_paper_chain_binds_one_way_so_it_has_a_fixed_point(self):
        """renderings bind the numbers, and the numbers bind no rendering.

        Two documents carrying each other's hash have no fixed point, and neither
        can be re-filed without invalidating the other. That is the same shape as
        the cycle the closeout manifest and the numbers file had, checked here
        because this manifest binds both of them and is the document that would
        silently start requiring the impossible.
        """
        numbers_doc = json.loads(
            (ROOT / "paper" / "numbers" / "iteration_11_paper_numbers.json")
            .read_text(encoding="utf-8"))
        renderings_doc = json.loads(
            (ROOT / "paper" / "tables" / "renderings.json").read_text(
                encoding="utf-8"))
        assert renderings_doc["inputs"]["sha256"], (
            "the renderings do not say which bytes of the numbers file they were "
            "rendered from, so 'derived from the filed numbers' is a claim and not "
            "something a reviewer can check")
        assert renderings_doc["inputs"]["path"].endswith(
            "iteration_11_paper_numbers.json")
        bound_by_numbers = set(numbers_doc["inputs"]["paths"].values())
        assert not any(path.startswith("paper/") for path in bound_by_numbers), (
            f"the numbers file binds {sorted(bound_by_numbers)}, which is downstream "
            f"of it; the paper's chain has to run one way")

    def test_an_entry_point_outside_the_bound_set_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = [{
            "path": "outputs/iteration_11/nothing_here.json",
            "what_it_establishes": "nothing",
            "verified_by": "scripts/iter11_replay_checks.py"}]
        issues = manifest.check_the_manifest(doc)
        assert any("leaves the manifest's guarantee" in i for i in issues), issues

    def test_a_verifier_dropped_from_the_bound_set_is_refused(self, monkeypatch):
        """The pointer stays and the thing it points at is quietly unbound.

        The realistic weakening, and the one the old shape of this test could not
        express: a manifest that keeps telling the reviewer where to start while
        dropping the guarantee under one of those pointers.
        """
        doc = _small(monkeypatch)
        dropped = doc["where_a_reviewer_starts"][0]["verified_by"]
        assert dropped in _paths(doc)
        doc["bound"] = [entry for entry in doc["bound"]
                        if entry["path"] != dropped]
        doc["n_bound"] = len(doc["bound"])
        issues = manifest.check_the_manifest(doc)
        assert any("that script is not bound" in i for i in issues), issues
        assert any(dropped in i for i in issues), issues

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


# ---------------------------------------------------------------------------
# A weakened manifest is refused rather than re-derived
# ---------------------------------------------------------------------------

class TestAWeakenedManifestDoesNotReDeriveItself:
    """The defect these tests exist because of.

    ``verify`` used to read the entry points out of the artifact and pass them
    back into ``build`` as the expected side of the comparison. Deleting all
    eight then produced zero validation issues and a document that re-derived
    exactly, because the thing being compared and the thing it was compared
    against were the same list. A pin compared against a re-derivation of
    itself enforces nothing.

    The expected side now lives outside the artifact, in ``ENTRY_POINTS``, and
    an empty list is refused outright rather than validating cleanly on the
    grounds that there was nothing to validate.
    """

    def test_echoing_the_artifact_back_would_have_passed(self, monkeypatch):
        """The reproduction, kept as a test so the fix cannot be undone quietly.

        Building over the entry points the document itself files agrees with
        that document whatever it files. This is not a behaviour anybody wants;
        it is the demonstration that the comparison has to come from somewhere
        else, and if ``build`` ever stops taking the parameter this fails loudly
        instead of silently weakening ``verify``.
        """
        paths = _a_complete_small_bound_set()
        monkeypatch.setattr(manifest, "discover", lambda: paths)
        for points in ([], list(manifest.ENTRY_POINTS)[:3],
                       [dict(manifest.ENTRY_POINTS[0],
                             what_it_establishes="something else")]):
            echoed = manifest.build(paths, points)
            assert echoed["where_a_reviewer_starts"] == [
                dict(point) for point in points]
            assert manifest.check_the_manifest(echoed) != [], (
                "check_the_manifest is the only thing standing between an "
                "echoed-back entry list and a clean bill of health")

    def test_deleting_every_entry_point_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        assert manifest.check_the_manifest(doc) == []
        doc["where_a_reviewer_starts"] = []
        issues = manifest.check_the_manifest(doc)
        assert any("where_a_reviewer_starts is empty" in i for i in issues), issues
        assert any("weakened rather than simplified" in i for i in issues), issues

    def test_thinning_the_entry_points_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = list(manifest.ENTRY_POINTS)[:3]
        issues = manifest.check_the_manifest(doc)
        assert any("is not the ENTRY_POINTS this module files" in i
                   for i in issues), issues

    def test_rewording_what_an_entry_point_establishes_is_refused(self, monkeypatch):
        """The prose is part of the pointer, so it is part of the pin."""
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"][0]["what_it_establishes"] = "something"
        assert any("is not the ENTRY_POINTS this module files" in i
                   for i in manifest.check_the_manifest(doc))

    def test_building_does_not_alias_the_entry_point_constant(self, monkeypatch):
        """Editing the document must not edit the thing it is checked against.

        ``list(entry_points)`` copies the list and shares the dicts, so a caller
        that rewords one pointer in a built manifest rewords it in ENTRY_POINTS
        too and the comparison agrees with itself again. Found by the test above
        passing when it should not have.
        """
        before = copy.deepcopy(manifest.ENTRY_POINTS)
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"][0]["what_it_establishes"] = "something"
        doc["where_a_reviewer_starts"][0]["verified_by"] = "scripts/nothing.py"
        assert [dict(p) for p in manifest.ENTRY_POINTS] == \
            [dict(p) for p in before], (
                "build() handed out references to ENTRY_POINTS, so the expected "
                "side of the comparison is editable through the artifact")

    def test_reordering_the_entry_points_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = list(reversed(
            [dict(point) for point in manifest.ENTRY_POINTS]))
        assert any("is not the ENTRY_POINTS this module files" in i
                   for i in manifest.check_the_manifest(doc))

    def test_a_weakened_manifest_fails_verification_and_not_re_derivation(
            self, tmp_path, monkeypatch):
        """End to end: the code a reviewer gets, not just the issue list."""
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = []
        code, conclusion, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 1, (conclusion, issues)
        assert any("where_a_reviewer_starts" in i for i in issues), issues

    def test_verify_passes_the_constant_and_not_the_filed_list(self):
        """Read out of the source, because it is one argument on one line."""
        source = (ROOT / "scripts" / "iter11_closeout_evidence_manifest.py") \
            .read_text(encoding="utf-8")
        body = source.split("def verify(", 1)[1].split("\ndef ", 1)[0]
        assert "ENTRY_POINTS)" in body, (
            "verify() has stopped passing the module constant as the expected "
            "entry points, which is the one change that makes a thinned "
            "manifest re-derive itself exactly")
        assert 'filed.get("where_a_reviewer_starts")' not in body


# ---------------------------------------------------------------------------
# The manifest is compared with its own committed bytes
# ---------------------------------------------------------------------------

class TestTheManifestIsComparedWithTheBlobItWasCommittedAs:
    """A document cannot carry its own hash inside itself.

    So the only way to notice that the manifest on disk is not the manifest that
    was committed is to ask the object store. Without that question an edit in
    place is invisible, and it is invisible in the worst direction: the
    re-derivation is fed the edited document, so the edited document agrees.
    """

    @staticmethod
    def _head_holding(target: Path, self_sha256: str | None):
        """``committed_hashes`` answering HEAD with a specific blob for the manifest.

        ``None`` is what the object store returns for a path HEAD does not hold,
        and the manifest's own path is answered with it to test that branch.
        """
        def answered(wanted: list[str]) -> dict[str, str | None]:
            out: dict[str, str | None] = {}
            for rel in wanted:
                if rel == str(target):
                    out[rel] = self_sha256
                    continue
                candidate = Path(rel)
                on_disk = candidate if candidate.is_absolute() else ROOT / rel
                if on_disk.is_file():
                    out[rel] = manifest.sha256_file(on_disk)
            return out
        return answered

    def test_an_untouched_manifest_agrees_with_its_own_blob(self, tmp_path,
                                                            monkeypatch):
        target = _write(tmp_path, _small(monkeypatch))
        code, conclusion, issues, _ = manifest.verify(target)
        assert code == 0, (conclusion, issues)

    def test_a_manifest_edited_after_it_was_committed_is_caught(self, tmp_path,
                                                                monkeypatch):
        doc = _small(monkeypatch)
        target = _write(tmp_path, doc)
        as_committed = manifest.sha256_file(target)
        doc["an_edit_nobody_committed"] = True
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        monkeypatch.setattr(
            manifest, "committed_hashes",
            self._head_holding(target, as_committed))
        code, _, issues, _ = manifest.verify(target)
        assert code == 1
        assert any("edited after it was committed" in i for i in issues), issues
        assert any("the re-derivation above is fed the edited document" in i
                   for i in issues), issues

    def test_the_edit_can_be_one_that_re_derivation_would_not_notice(self,
                                                                    tmp_path,
                                                                    monkeypatch):
        """Deleting the entry points and re-writing every dependent field.

        The strongest form of the attack: a manifest that is internally
        consistent, re-derives exactly, and differs from what was committed only
        in that it no longer points anywhere.
        """
        target = _write(tmp_path, _small(monkeypatch))
        as_committed = manifest.sha256_file(target)
        doc = json.loads(target.read_text(encoding="utf-8"))
        doc.pop("where_a_reviewer_starts")
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        monkeypatch.setattr(
            manifest, "committed_hashes",
            self._head_holding(target, as_committed))
        code, _, issues, _ = manifest.verify(target)
        assert code == 1
        assert any("is not the manifest that was committed" in i
                   or "edited after it was committed" in i for i in issues), issues

    def test_a_manifest_that_is_not_in_head_at_all_is_caught(self, tmp_path,
                                                            monkeypatch):
        target = _write(tmp_path, _small(monkeypatch))
        # HEAD holds no blob for the manifest itself, and the disk copy is real.
        monkeypatch.setattr(
            manifest, "committed_hashes", self._head_holding(target, None))
        code, _, issues, _ = manifest.verify(target)
        assert code == 1
        assert any("is not in HEAD at all" in i for i in issues), issues

    def test_the_comparison_is_named_unavailable_where_there_is_no_history(
            self, tmp_path, monkeypatch):
        """Not silently skipped: a checkout that cannot ask says it cannot ask."""
        target = _write(tmp_path, _small(monkeypatch))
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        code, _, issues, unverifiable = manifest.verify(target)
        assert code == 3, (issues, unverifiable)
        assert "rather than one edited in place" in unverifiable[0], unverifiable

    def test_the_manifest_still_does_not_bind_itself(self, monkeypatch):
        """Binding itself is impossible; being compared with HEAD is not the same."""
        doc = _small(monkeypatch)
        assert SELF not in _paths(doc)
        assert doc["what_this_manifest_does_not_bind"]["itself"]["path"] == SELF


# ---------------------------------------------------------------------------
# The bound set is the dependency closure, not eight modules somebody listed
# ---------------------------------------------------------------------------

#: The modules a review of the manifest named as omitted. They are the ones a
#: reader could spot by looking; the point of walking the imports is the ones
#: nobody could.
THE_MODULES_A_REVIEWER_NAMED = (
    "src/causal_mllm/evaluation/estimands.py",
    "src/causal_mllm/evaluation/hypotheses.py",
    "src/causal_mllm/evaluation/adjudication.py",
    "src/causal_mllm/evaluation/schema.py",
    "src/causal_mllm/replay/selection.py",
    "src/causal_mllm/replay/truncation.py",
    "src/causal_mllm/construction/readiness.py",
)


class TestTheBoundSetIsTheImportClosure:
    def test_the_named_omissions_are_now_bound(self):
        paths = manifest.discover()
        bound = set(paths)
        for module in THE_MODULES_A_REVIEWER_NAMED:
            assert module in bound, module

    def test_the_closure_reaches_far_more_than_anybody_listed(self):
        """The reason it is walked rather than extended by hand."""
        closure = manifest.closure_block(manifest.discover())
        hand_listed = [p for p in manifest.BOUND_PATHS
                       if p.startswith(f"{manifest.SRC_ROOT}/")]
        unlisted = sorted(set(closure["modules"]) - set(hand_listed))
        assert len(unlisted) > len(hand_listed), (
            f"{len(unlisted)} modules reached against {len(hand_listed)} listed "
            f"by hand: a closure that only finds what was already written down "
            f"is not walking anything")
        for module in THE_MODULES_A_REVIEWER_NAMED:
            assert module in closure["modules"]

    def test_every_closure_module_is_bound_and_committed(self):
        paths = set(manifest.discover())
        closure = manifest.closure_block(sorted(paths))
        for module in closure["modules"]:
            assert module in paths, module
            assert (ROOT / module).is_file(), module

    def test_the_closure_is_a_pure_function_of_the_path_list(self):
        paths = _a_complete_small_bound_set()
        first = manifest.closure_block(paths)
        second = manifest.closure_block(paths)
        assert first == second
        assert first["n_modules"] == len(first["modules"])
        assert first["n_roots_walked"] == len(first["roots"])

    def test_the_walk_follows_a_deferred_import(self):
        """Two of the four arms' adapters are imported inside a function body.

        A module-level-only walk leaves the code that produced those generations
        unbound, and nothing about the manifest would say so.
        """
        closure = manifest.closure_block(manifest.discover())
        adapters = [module for module in closure["modules"]
                    if "/replay/adapters/" in module]
        assert len(adapters) >= 3, adapters
        assert "src/causal_mllm/replay/adapters/__init__.py" in closure["modules"]

    def test_a_deferred_import_of_an_absent_module_is_filed_with_its_site(self):
        """The repository has exactly one, and it is a real latent defect.

        ``build_adapter`` imports ``causal_mllm.replay.adapters.gemma3`` for the
        protocol's never-invoked fallback, and that module does not exist, so the
        branch would raise ModuleNotFoundError instead of the ReplayError it
        means to raise. Filed rather than fatal because it does not stop the
        package importing; not filed as harmless either.
        """
        closure = manifest.closure_block(manifest.discover())
        absent = closure["deferred_imports_of_modules_that_do_not_exist"]
        site = absent["src/causal_mllm/replay/adapters/__init__.py"]
        assert list(site) == ["build_adapter"], site
        assert any("gemma3" in name for names in site.values() for name in names)
        assert closure["modules_that_cannot_be_imported"] == {}, (
            "a module-level import of an absent module means the package does "
            "not import at all")

    def test_an_import_of_an_absent_module_is_a_gap_and_not_a_symbol(self,
                                                                    tmp_path):
        """``import X.Y`` can only name a module, so an unresolvable one is a gap.

        Without the symbol-position flag this was discarded because its parent
        ``causal_mllm`` resolves, and the walker walked a smaller graph than the
        interpreter would import.
        """
        target = tmp_path / "probe.py"
        target.write_text(
            "import causal_mllm.seeds\n"
            "import causal_mllm.module_that_does_not_exist\n"
            "from causal_mllm.seeds import sha256_bytes\n",
            encoding="utf-8")
        resolved, at_import_time, deferred, relative = \
            manifest._imported_modules(target)
        assert "src/causal_mllm/seeds.py" in resolved
        assert at_import_time == {
            "<module>": ["causal_mllm.module_that_does_not_exist"]}, at_import_time
        assert deferred == {}
        assert relative == []

    def test_a_symbol_after_a_resolving_module_is_not_a_gap(self, tmp_path):
        target = tmp_path / "probe.py"
        target.write_text(
            "from causal_mllm.seeds import sha256_bytes, sha256_file\n",
            encoding="utf-8")
        resolved, at_import_time, _, _ = manifest._imported_modules(target)
        assert "src/causal_mllm/seeds.py" in resolved
        assert at_import_time == {}, (
            "every attribute of a module that resolves was reported as a "
            "missing module, which is a check that cries wolf and gets "
            "switched off")

    def test_a_deferred_gap_is_not_reported_as_an_import_time_one(self, tmp_path):
        target = tmp_path / "probe.py"
        target.write_text(
            "def build():\n"
            "    from causal_mllm.never_written import Thing\n"
            "    return Thing\n",
            encoding="utf-8")
        _, at_import_time, deferred, _ = manifest._imported_modules(target)
        assert at_import_time == {}
        assert deferred == {"build": ["causal_mllm.never_written",
                                      "causal_mllm.never_written.Thing"]}

    def test_a_relative_import_is_counted_rather_than_skipped(self, tmp_path):
        """This repository has none; the closure says so instead of assuming it."""
        target = tmp_path / "probe.py"
        target.write_text(
            "from . import sibling\n"
            "from ..other import thing\n"
            "import causal_mllm.seeds\n",
            encoding="utf-8")
        resolved, _, _, relative = manifest._imported_modules(target)
        assert "src/causal_mllm/seeds.py" in resolved
        assert relative == [". in <module>", "..other in <module>"], relative
        closure = manifest.closure_block(manifest.discover())
        assert closure["n_relative_imports_the_walk_could_not_follow"] == 0
        assert closure["relative_imports_the_walk_could_not_follow"] == {}

    def test_a_closure_module_dropped_from_the_bound_set_is_a_finding(
            self, monkeypatch):
        """The dependency stays declared; only the guarantee is quietly removed."""
        doc = _small(monkeypatch)
        dropped = doc["scientific_dependency_closure"]["modules"][0]
        doc["bound"] = [entry for entry in doc["bound"]
                        if entry["path"] != dropped]
        doc["n_bound"] = len(doc["bound"])
        issues = manifest.check_the_manifest(doc)
        assert any("does not vouch for" in i for i in issues), issues

    def test_a_module_pretended_not_to_be_a_dependency_is_caught_by_re_derivation(
            self, tmp_path, monkeypatch):
        """The sneakier edit: take it out of the closure list as well.

        The document is then internally consistent and ``check_the_manifest``
        cannot see anything wrong with it, so the re-derivation has to -- which
        recomputes the closure from the bound Python instead of trusting the
        filed list. A module that is not hand-listed is chosen so that the two
        checks are not confused with each other.
        """
        doc = _small(monkeypatch)
        closure = doc["scientific_dependency_closure"]
        hand_listed = {p for p in manifest.BOUND_PATHS
                       if p.startswith(f"{manifest.SRC_ROOT}/")}
        dropped = next(module for module in closure["modules"]
                       if module not in hand_listed)
        closure["modules"] = [m for m in closure["modules"] if m != dropped]
        closure["n_modules"] = len(closure["modules"])
        assert manifest.check_the_manifest(doc) == [], (
            "this test is about the edit internal agreement cannot see")
        code, _, issues, _ = manifest.verify(_write(tmp_path, doc))
        assert code == 1, issues
        assert any("scientific_dependency_closure" in i for i in issues), issues

    def test_a_hand_listed_module_the_walk_misses_is_reported_not_raised(
            self, monkeypatch):
        """``build`` stays total so ``verify`` can report instead of dying.

        A build that refuses to produce a document turns a diagnosable
        contradiction into "could not re-derive", and then the filed manifest
        cannot be compared with anything either.
        """
        monkeypatch.setattr(manifest, "discover", lambda: list(SMALL))
        doc = manifest.build(list(SMALL), manifest.ENTRY_POINTS)
        missed = doc["scientific_dependency_closure"][
            "hand_listed_modules_the_closure_does_not_reach"]
        assert "src/causal_mllm/evaluation/bootstrap.py" in missed
        issues = manifest.check_the_manifest(doc)
        assert any("the walk is broken or the list is stale" in i for i in issues)
        assert any("dropping the module would be neither" in i for i in issues)

    def test_the_filed_field_has_to_describe_the_closure_beside_it(self,
                                                                  monkeypatch):
        doc = _small(monkeypatch)
        closure = doc["scientific_dependency_closure"]
        assert closure["hand_listed_modules_the_closure_does_not_reach"] == []
        closure["hand_listed_modules_the_closure_does_not_reach"] = [
            "src/causal_mllm/seeds.py"]
        issues = manifest.check_the_manifest(doc)
        assert any("the field does not describe the list it sits in" in i
                   for i in issues), issues

    def test_an_empty_closure_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["scientific_dependency_closure"] = {"n_modules": 0, "modules": []}
        issues = manifest.check_the_manifest(doc)
        assert any("no library module is bound" in i for i in issues), issues

    def test_the_document_says_what_the_closure_does_not_cover(self, monkeypatch):
        doc = _small(monkeypatch)
        closure = doc["scientific_dependency_closure"]
        assert "third-party" in closure["what_it_does_not_cover"]
        assert "dependency_lock_reconstruction.json" in \
            closure["what_it_does_not_cover"]


# ---------------------------------------------------------------------------
# The deep closeout executes the gates instead of grepping for --verify
# ---------------------------------------------------------------------------

class TestTheDeepCloseoutExecutesTheGatesItPointsAt:
    """``--verify`` hashes a verifier and looks for a string in its source.

    That is a check on the pointers, not on the evidence, and the top-level
    closeout could pass it while a scientific verifier behind one of those
    pointers failed. ``--deep`` runs them.
    """

    @pytest.fixture
    def gates(self, monkeypatch):
        """Every subprocess the deep closeout starts. Git is passed through."""
        real = subprocess.run
        state = SimpleNamespace(calls=[], cwds=[], codes={}, default=0)

        def fake(command, **kwargs):
            if command and command[0] == "git":
                return real(command, **kwargs)
            state.calls.append(list(command))
            state.cwds.append(str(kwargs.get("cwd")))
            name = Path(command[1]).name
            code = state.codes.get(name, state.default)
            if code == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 1))
            return subprocess.CompletedProcess(
                command, code,
                f"GATE {name}: {'OK' if code == 0 else 'NOT OK'}\nsecond line\n",
                "" if code != 2 else "a traceback would be here")

        monkeypatch.setattr(manifest.subprocess, "run", fake)
        return state

    def test_every_entry_point_verifier_is_executed(self, tmp_path, monkeypatch,
                                                    gates):
        target = _write(tmp_path, _small(monkeypatch))
        code, report = manifest.deep_closeout(target)
        assert code == 0, report
        expected = sorted({point["verified_by"]
                           for point in manifest.ENTRY_POINTS})
        assert report["verifiers_it_executes"] == expected
        assert report["n_verifiers"] == len(expected)
        assert sorted({Path(call[1]).name for call in gates.calls}) == \
            sorted({Path(verifier).name for verifier in expected})
        assert report["n_verified"] == len(gates.calls)
        assert report["n_failed"] == 0 and report["n_incomplete_here"] == 0

    def test_the_argv_is_the_one_filed_and_not_a_bare_verify(self):
        """A verifier run with the wrong arguments can exit 0 having checked less."""
        replay = "scripts/iter11_replay_checks.py"
        assert manifest.deep_invocations_for(replay) == (("--all", "--verify"),), (
            "without --all that gate checks one arm and not the panel, and it "
            "still exits 0")
        media = "scripts/iter11_write_media_manifest.py"
        assert manifest.deep_invocations_for(media) == (
            ("--verify",), ("--verify", "--panel-only"))
        assert manifest.deep_invocations_for(
            "scripts/iter11_cross_model_analysis.py") == \
            manifest.DEFAULT_DEEP_INVOCATION

    def test_the_filed_argv_is_what_actually_runs(self, tmp_path, monkeypatch,
                                                  gates):
        target = _write(tmp_path, _small(monkeypatch))
        manifest.deep_closeout(target)
        by_verifier: dict[str, list[list[str]]] = {}
        for call in gates.calls:
            by_verifier.setdefault(Path(call[1]).as_posix(), []).append(call[2:])
        for verifier in {point["verified_by"] for point in manifest.ENTRY_POINTS}:
            assert sorted(by_verifier[verifier]) == sorted(
                [list(argv) for argv in manifest.deep_invocations_for(verifier)])

    def test_exit_three_is_counted_as_incomplete_here(self, tmp_path, monkeypatch,
                                                      gates):
        """The lane that made a fresh checkout report failure for being fresh."""
        gates.codes["iter11_write_media_manifest.py"] = 3
        target = _write(tmp_path, _small(monkeypatch))
        code, report = manifest.deep_closeout(target)
        assert code == 3, report
        assert report["n_incomplete_here"] == 2  # both media invocations
        assert report["n_failed"] == 0
        assert report["failed"] == []
        incomplete = [entry for entry in report["invocations"]
                      if entry["code"] == 3]
        assert incomplete and all(
            entry["verifier"] == "scripts/iter11_write_media_manifest.py"
            for entry in incomplete)

    @pytest.mark.parametrize("code", [1, 2, 4, 139, -11])
    def test_anything_that_is_not_zero_or_three_fails_it(self, tmp_path,
                                                         monkeypatch, gates,
                                                         code):
        """Exit 1 or 2 as specified, and a crash too: it did not verify."""
        gates.codes["iter11_cross_model_analysis.py"] = code
        target = _write(tmp_path, _small(monkeypatch))
        out, report = manifest.deep_closeout(target)
        assert out == 1, (code, report)
        assert report["n_failed"] == 1
        assert report["failed"][0]["code"] == code
        assert report["failed"][0]["verifier"] == \
            "scripts/iter11_cross_model_analysis.py"

    def test_a_timeout_is_a_failure_and_says_so(self, tmp_path, monkeypatch,
                                                gates):
        gates.codes["iter11_replay_checks.py"] = "timeout"
        target = _write(tmp_path, _small(monkeypatch))
        code, report = manifest.deep_closeout(target, timeout=7)
        assert code == 1
        assert report["timeout_seconds"] == 7
        assert report["failed"][0]["code"] == "timeout"
        assert "did not verify" in report["failed"][0]["stderr"]

    def test_the_gates_are_not_run_when_the_manifest_itself_fails(self, tmp_path,
                                                                  monkeypatch,
                                                                  gates):
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = []
        code, report = manifest.deep_closeout(_write(tmp_path, doc))
        assert code == 1
        assert gates.calls == [], (
            "the gates ran against a manifest whose binding is already in doubt")
        assert "why_the_gates_were_not_run" in report

    def test_the_gates_it_runs_come_from_the_constant_not_the_artifact(
            self, tmp_path, monkeypatch, gates):
        """A weakened manifest must not get to weaken its own audit."""
        doc = _small(monkeypatch)
        doc["where_a_reviewer_starts"] = doc["where_a_reviewer_starts"][:1]
        target = _write(tmp_path, doc)
        monkeypatch.setattr(
            manifest, "verify", lambda path=None: (0, "assumed_sound", [], []))
        code, report = manifest.deep_closeout(target)
        assert code == 0, report
        expected = {point["verified_by"] for point in manifest.ENTRY_POINTS}
        assert {call[1] for call in gates.calls} == expected
        assert report["n_verifiers"] == len(expected)

    def test_a_shallow_verify_that_is_incomplete_makes_the_deep_one_too(
            self, tmp_path, monkeypatch, gates):
        target = _write(tmp_path, _small(monkeypatch))
        monkeypatch.setattr(manifest, "object_store_available", lambda: False)
        code, report = manifest.deep_closeout(target)
        assert code == 3, report
        assert report["shallow_verify"]["code"] == 3
        assert report["n_failed"] == 0
        assert report["n_verified"] == len(gates.calls), (
            "every gate that could run ran, and the answer is still 3 because "
            "the manifest's own committedness could not be checked")

    def test_deep_invocations_names_no_verifier_the_entry_points_do_not(self):
        named = {point["verified_by"] for point in manifest.ENTRY_POINTS}
        assert set(manifest.DEEP_INVOCATIONS) <= named
        doc = manifest.build(_a_complete_small_bound_set(), manifest.ENTRY_POINTS)
        assert manifest.check_the_manifest(doc) == []

    def test_an_argv_for_a_verifier_nobody_points_at_is_refused(self, monkeypatch):
        doc = _small(monkeypatch)
        monkeypatch.setitem(manifest.DEEP_INVOCATIONS,
                            "scripts/iter11_freeze_protocol.py",
                            (("--verify",),))
        issues = manifest.check_the_manifest(doc)
        assert any("a claim about a gate that does not exist" in i for i in issues), \
            issues

    def test_verify_alone_executes_nothing(self, tmp_path, monkeypatch, gates):
        """The distinction the document has to state, tested rather than claimed."""
        target = _write(tmp_path, _small(monkeypatch))
        assert manifest.verify(target)[0] == 0
        assert gates.calls == []

    def test_the_document_says_what_verify_alone_does_not_do(self, monkeypatch):
        doc = _small(monkeypatch)
        deep = doc["deep_closeout"]
        assert deep["command"].endswith("--deep")
        assert "what_verify_alone_does_not_do" in deep
        assert set(deep["how_exit_codes_aggregate"]) >= {"0", "1", "2", "3"}
        assert "incomplete here" in deep["how_exit_codes_aggregate"]["3"]

    def test_the_aggregation_rule_is_checked_not_just_filed(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["deep_closeout"]["how_exit_codes_aggregate"] = {}
        assert any("no defined meaning" in i
                   for i in manifest.check_the_manifest(doc))

    def test_the_verifiers_listed_have_to_be_the_ones_pointed_at(self, monkeypatch):
        doc = _small(monkeypatch)
        doc["deep_closeout"]["verifiers_it_executes"] = \
            doc["deep_closeout"]["verifiers_it_executes"][:1]
        doc["deep_closeout"]["n_verifiers"] = 1
        issues = manifest.check_the_manifest(doc)
        assert any("not a subset of them" in i for i in issues), issues

    def test_the_deep_exit_codes_reach_the_command_line(self, tmp_path,
                                                       monkeypatch, gates,
                                                       capsys):
        target = _write(tmp_path, _small(monkeypatch))
        assert manifest.main(["--deep", "--out", str(target)]) == 0
        out = capsys.readouterr().out
        assert "DEEP CLOSEOUT: PASS" in out
        assert "executing" in out and "entry-point verifier" in out

        gates.codes["iter11_write_media_manifest.py"] = 3
        assert manifest.main(["--deep", "--out", str(target)]) == 3
        assert "INCOMPLETE HERE" in capsys.readouterr().out

        gates.codes["iter11_cross_model_analysis.py"] = 1
        assert manifest.main(["--deep", "--out", str(target)]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_deep_verify_and_write_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            manifest.main(["--deep", "--verify"])
        assert exc.value.code == 2

    def test_a_gate_is_run_in_the_repository_and_not_the_caller(self, tmp_path,
                                                               monkeypatch,
                                                               gates):
        """Each verifier resolves its own inputs relative to its repo root."""
        target = _write(tmp_path, _small(monkeypatch))
        manifest.deep_closeout(target)
        assert gates.cwds and all(cwd == str(ROOT) for cwd in gates.cwds), \
            gates.cwds
