"""Iteration 11.6: a verification command may not rewrite the record it checks.

Two findings are pinned here, and they are the same finding seen from two
sides.

The documented contract. ``README.md`` introduced the completion gate under
"Re-running any of it, read-only:", and the first command in that block was
``iter11_replay_checks.py --all`` -- which wrote
``iteration_11_replay_checks.json`` into every run directory on every run,
unconditionally. A command advertised as a read was a mutation of committed
evidence. Generation is now behind ``--write-report`` and comparison behind
``--verify``; the default mode writes nothing.

The verdict that mutation produced. ``data/media`` is gitignored apart from 20
individually negated source images, so a fresh checkout holds 20 of the 3,034
files and none of the 100 the panel references. The media check re-hashed those
100 and reported every absence as a FAILURE, so running the documented
"read-only" command on a clean machine turned all four committed PASS reports
into FAIL. Nothing was wrong with the panel; the statement was about the
machine. Absence is now recorded as ``verifiable_here: false``, which makes the
verdict ``PASS_WITH_UNVERIFIED_SECTIONS`` -- not PASS, because a check that
could not run did not pass, and not FAIL, because nothing was found wrong --
and the media's identity is bound by a committed manifest so "not present here"
can be told apart from "present and different".

CI-safe: every fixture builds a miniature repository under ``tmp_path``, so a
checkout limitation, a tampered image and an unbound file can all be
manufactured rather than waited for.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
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


checks = _load_script("iter11_replay_checks")
manifest_producer = _load_script("iter11_write_media_manifest")


# ---------------------------------------------------------------------------
# A miniature repository: two families, one source image each
# ---------------------------------------------------------------------------
# One distinct image per family is what the real panel has -- 100 families, 100
# distinct images -- and the shared-identity check assumes it, so the fixture
# has to be built that way for the check to be exercised rather than tripped.

IMAGES = {
    "data/media/CMST_000001.png": b"the-first-family-source-image",
    "data/media/CMST_000002.png": b"the-second-family-source-image",
}
FAMILY_IDS = ("CMST_000001", "CMST_000002")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _families() -> dict[str, dict]:
    """The panel definition: which images each variant carries."""
    out = {}
    for family_id in FAMILY_IDS:
        image = f"data/media/{family_id}.png"
        out[family_id] = {"variants": {
            "cross_modal": {"messages": [{"images": [image]}, {"images": []}]},
            "vision_only": {"messages": [{"images": [image]}]},
            "text_only": {"messages": [{"images": []}]}}}
    return out


def _by_family(n_images: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """The replayed records: how many images each response was actually fed."""
    n_images = n_images or {}
    return {family_id: [
        {"variant": "cross_modal",
         "n_images": n_images.get((family_id, "cross_modal"), 1)},
        {"variant": "vision_only",
         "n_images": n_images.get((family_id, "vision_only"), 1)},
        {"variant": "text_only",
         "n_images": n_images.get((family_id, "text_only"), 0)},
    ] for family_id in FAMILY_IDS}


def _world(tmp_path: Path, *, on_disk=None, tampered=()) -> Path:
    """A checkout holding some of the media.

    ``on_disk`` defaults to every image. Pass a subset to model the fresh
    checkout the finding is about, and ``tampered`` to model the thing that is
    NOT a checkout limitation: the bytes differing from what was bound.
    """
    placed = IMAGES if on_disk is None else {
        rel: IMAGES[rel] for rel in on_disk}
    for rel, payload in placed.items():
        if rel in tampered:
            payload = payload + b"-not-the-bytes-the-panel-was-made-from"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return tmp_path


def _manifest(files: dict | None = None) -> dict:
    bound = IMAGES if files is None else files
    entries = {rel: {"sha256": _digest(payload), "bytes": len(payload)}
               for rel, payload in bound.items()}
    return {
        "scope": "whole_media_tree",
        "n_files": len(entries),
        "files": entries,
        "rollup_sha256": manifest_producer.rollup_sha256(entries),
    }


def _section(root: Path, *, manifest=_manifest, by_family=None, families=None):
    return checks.media_section(
        by_family if by_family is not None else _by_family(),
        families if families is not None else _families(),
        manifest() if callable(manifest) else manifest,
        repo_root=root,
        manifest_path=root / "outputs" / "iteration_11" / "media_manifest.json")


class TestAbsentMediaIsALimitationOfTheCheckoutNotOfThePanel:
    def test_every_image_present_and_matching_passes_and_says_it_was_checked(
            self, tmp_path):
        root = _world(tmp_path)
        block, notes = _section(root)
        assert block["ok"] is True
        assert block["issues"] == []
        assert block["verifiable_here"] is True
        assert block["n_referenced_images"] == 2
        assert block["n_hashed_here"] == 2
        assert block["n_absent_here"] == 0
        assert notes == []

    def test_a_fresh_checkout_holding_none_of_the_media_is_not_a_failure(
            self, tmp_path):
        # THE finding: this used to be two failures per family, and rewriting
        # the four committed reports with them turned PASS into FAIL.
        block, notes = _section(_world(tmp_path, on_disk=()))
        assert block["ok"] is True
        assert block["issues"] == []
        assert block["n_absent_here"] == 2
        assert block["n_hashed_here"] == 0
        assert block["verifiable_here"] is False

    def test_the_unverifiable_section_is_reported_rather_than_dropped(
            self, tmp_path):
        block, notes = _section(_world(tmp_path, on_disk=()))
        assert len(notes) == 1
        assert "UNVERIFIED" in notes[0]
        assert "2 of 2" in notes[0]
        assert block["manifest_rollup_sha256"] == _manifest()["rollup_sha256"]

    def test_it_names_the_families_it_could_not_image_check(self, tmp_path):
        # "Checked and identical" and "never checked" are different claims, and
        # only the naming keeps them apart.
        root = _world(tmp_path, on_disk=("data/media/CMST_000001.png",))
        block, _ = _section(root)
        assert block["families_whose_images_could_not_be_hashed_here"] == [
            "CMST_000002"]
        assert block["n_families_whose_images_could_not_be_hashed_here"] == 1
        assert block["n_hashed_here"] == 1

    def test_the_image_count_is_checked_even_without_the_bytes(self, tmp_path):
        # n_images comes from the panel definition and the replayed record, so
        # it needs no media on disk. Keeping it live is what stops "the media
        # are absent" from becoming "nothing about the media was checked".
        root = _world(tmp_path, on_disk=())
        block, _ = _section(
            root, by_family=_by_family({("CMST_000001", "text_only"): 5}))
        assert block["ok"] is False
        assert any("CMST_000001/text_only: n_images 5 != 0" in i
                   for i in block["issues"])

    def test_a_tampered_image_is_a_finding_and_names_both_digests(
            self, tmp_path):
        root = _world(tmp_path, tampered=("data/media/CMST_000002.png",))
        block, notes = _section(root)
        assert block["ok"] is False
        assert block["verifiable_here"] is True
        assert notes == []
        actual = _digest(IMAGES["data/media/CMST_000002.png"]
                         + b"-not-the-bytes-the-panel-was-made-from")
        issue = next(i for i in block["issues"] if "CMST_000002" in i)
        assert actual in issue
        assert _digest(IMAGES["data/media/CMST_000002.png"]) in issue

    def test_an_image_the_manifest_never_bound_is_a_finding_about_the_manifest(
            self, tmp_path):
        root = _world(tmp_path)
        partial = _manifest({"data/media/CMST_000001.png":
                             IMAGES["data/media/CMST_000001.png"]})
        block, _ = _section(root, manifest=partial)
        assert block["ok"] is False
        assert block["n_referenced_but_unbound"] == 1
        assert any("does not bind it" in i for i in block["issues"])

    def test_no_manifest_at_all_is_reported_with_the_command_that_writes_one(
            self, tmp_path):
        root = _world(tmp_path)
        block, _ = _section(root, manifest=None)
        assert block["ok"] is False
        assert block["manifest_path"] is None
        assert block["verifiable_here"] is False
        assert any("scripts/iter11_write_media_manifest.py" in i
                   for i in block["issues"])

    def test_a_family_whose_variants_disagree_on_the_bytes_still_fails(
            self, tmp_path):
        root = _world(tmp_path)
        families = _families()
        families["CMST_000001"]["variants"]["vision_only"]["messages"] = [
            {"images": ["data/media/CMST_000002.png"]}]
        block, _ = _section(root, families=families)
        assert block["ok"] is False
        assert any("2 distinct image hashes" in i for i in block["issues"])


class TestTheVerdictKeepsThreeAnswersApart:
    def test_a_failure_is_a_failure(self):
        verdict, unverifiable = checks.verdict_for({"failures": ["media: x"]})
        assert verdict == "FAIL"
        assert unverifiable == []

    def test_everything_checked_and_clean_is_a_pass(self):
        verdict, unverifiable = checks.verdict_for(
            {"failures": [], "media": {"ok": True, "verifiable_here": True}})
        assert verdict == "PASS"
        assert unverifiable == []

    def test_clean_but_partly_uncheckable_is_neither_of_those(self):
        result = {"failures": [],
                  "media": {"ok": True, "verifiable_here": False}}
        verdict, unverifiable = checks.verdict_for(result)
        assert verdict == "PASS_WITH_UNVERIFIED_SECTIONS"
        assert unverifiable == ["media"]

    def test_a_failure_outranks_an_unverified_section(self):
        # Otherwise "could not check" would become a way to soften a finding.
        result = {"failures": ["coverage: 99 families"],
                  "media": {"ok": True, "verifiable_here": False}}
        verdict, unverifiable = checks.verdict_for(result)
        assert verdict == "FAIL"
        assert unverifiable == ["media"]

    def test_the_third_state_is_opt_in_not_a_default(self):
        # A section that never considered the question must not be counted as
        # one that could not answer it.
        verdict, unverifiable = checks.verdict_for(
            {"failures": [], "coverage": {"ok": True},
             "media": {"ok": True, "verifiable_here": True}})
        assert verdict == "PASS"
        assert unverifiable == []


class TestVerifyComparesWithoutWriting:
    def test_identical_reports_differ_on_nothing(self):
        report = {"verdict": "PASS", "media": {"ok": True}}
        assert checks.diff_reports(report, dict(report)) == ([], [])

    def test_a_field_that_moved_is_named(self):
        fresh = {"verdict": "PASS", "coverage": {"n_families": 100}}
        filed = {"verdict": "PASS", "coverage": {"n_families": 99}}
        differing, skipped = checks.diff_reports(fresh, filed)
        assert differing == ["coverage"]
        assert skipped == []

    def test_a_section_this_checkout_cannot_verify_is_skipped_and_named(self):
        # The whole point of the mode. On a fresh checkout the media block
        # legitimately differs, and comparing it would report the machine as a
        # change in the panel.
        fresh = {"verdict": "PASS_WITH_UNVERIFIED_SECTIONS",
                 "unverifiable_sections": ["media"],
                 "media": {"ok": True, "verifiable_here": False,
                           "n_hashed_here": 0},
                 "coverage": {"n_families": 100}}
        filed = {"verdict": "PASS", "media": {"ok": True, "n_hashed_here": 100},
                 "coverage": {"n_families": 100}}
        differing, skipped = checks.diff_reports(fresh, filed)
        assert differing == []
        assert skipped == ["media"]

    def test_a_genuine_media_failure_is_still_caught_through_the_verdict(self):
        # Skipping the section must not skip the finding: a mismatched image
        # lands in the top-level verdict and failures, which are never skipped.
        fresh = {"verdict": "FAIL", "unverifiable_sections": [],
                 "failures": ["media: CMST_000001/cross_modal: image differs"],
                 "media": {"ok": False, "verifiable_here": True}}
        filed = {"verdict": "PASS", "failures": [],
                 "media": {"ok": True, "verifiable_here": True}}
        differing, _ = checks.diff_reports(fresh, filed)
        assert "verdict" in differing
        assert "failures" in differing

    def test_a_field_only_one_side_has_is_a_difference(self):
        differing, _ = checks.diff_reports(
            {"verdict": "PASS", "truncation": {"overall_rate": 0.0}},
            {"verdict": "PASS"})
        assert differing == ["truncation"]

    def test_a_checkout_blind_spot_is_not_a_report_that_stopped_reproducing(
            self):
        # THE second half of the finding. The filed report says PASS because the
        # machine that filed it held the media; this one says
        # PASS_WITH_UNVERIFIED_SECTIONS because it does not. Those are the same
        # answer about the panel, and reporting them as a disagreement is how a
        # fresh clone learns that four good reports are broken.
        fresh = {"verdict": "PASS_WITH_UNVERIFIED_SECTIONS",
                 "unverifiable_sections": ["media"], "failures": [],
                 "media": {"ok": True, "verifiable_here": False}}
        filed = {"verdict": "PASS", "failures": [],
                 "media": {"ok": True, "verifiable_here": True}}
        differing, skipped = checks.diff_reports(fresh, filed)
        assert differing == []
        assert skipped == ["media"]

    def test_the_blind_spot_never_softens_a_filed_failure(self):
        fresh = {"verdict": "PASS_WITH_UNVERIFIED_SECTIONS",
                 "unverifiable_sections": ["media"], "failures": [],
                 "media": {"ok": True, "verifiable_here": False}}
        filed = {"verdict": "FAIL", "failures": ["media: image differs"],
                 "media": {"ok": False, "verifiable_here": True}}
        differing, _ = checks.diff_reports(fresh, filed)
        assert "verdict" in differing
        assert "failures" in differing

    def test_the_blind_spot_is_symmetrical(self):
        fresh = {"verdict": "PASS", "failures": [],
                 "media": {"ok": True, "verifiable_here": True}}
        filed = {"verdict": "PASS_WITH_UNVERIFIED_SECTIONS",
                 "unverifiable_sections": ["media"], "failures": [],
                 "media": {"ok": True, "verifiable_here": False}}
        differing, skipped = checks.diff_reports(fresh, filed)
        assert differing == []
        assert skipped == ["media"]

    def test_comparable_verdict_collapses_only_the_blind_spot(self):
        assert checks.comparable_verdict(
            {"verdict": "PASS_WITH_UNVERIFIED_SECTIONS"}) == "PASS"
        assert checks.comparable_verdict({"verdict": "PASS"}) == "PASS"
        assert checks.comparable_verdict({"verdict": "FAIL"}) == "FAIL"


# ---------------------------------------------------------------------------
# The CLI contract, against a run directory that holds nothing real
# ---------------------------------------------------------------------------

SECTIONS = ("coverage", "full_panel", "provenance", "confirmatory_gate",
            "eligibility", "terminal_query_equality", "media", "truncation")


def _canned(mod, *, media_verifiable=True, failures=(), warnings=()):
    result = {"model_key": "target_x", "run_dir": "outputs/target_x/run",
              "lock_path": "outputs/lock.yaml"}
    for name in SECTIONS:
        result[name] = {"issues": [], "ok": not failures}
    result["truncation"].update({"overall_rate": 0.0, "variant_spread": 0.0})
    if not media_verifiable:
        result["media"]["verifiable_here"] = False
    result["failures"] = [f"media: {i}" for i in failures]
    result["warnings"] = list(warnings)
    result["verdict"], result["unverifiable_sections"] = mod.verdict_for(result)
    return result


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """The completion gate, pointed at a run directory that holds nothing."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(checks, "find_run_dir", lambda key: run_dir)
    monkeypatch.setattr(checks, "check_prompt_uniformity", lambda keys: [])
    return run_dir


def _run(monkeypatch, gate, argv, result=None):
    monkeypatch.setattr(checks, "check",
                        lambda key, rd, lp: result or _canned(checks))
    monkeypatch.setattr(sys, "argv", ["iter11_replay_checks.py", *argv])
    return checks.main()


class TestTheDocumentedReadOnlyCommandWritesNothing:
    def test_the_default_mode_writes_nothing(self, monkeypatch, gate):
        code = _run(monkeypatch, gate, ["--model-key", "target_x"])
        assert code == 0
        assert not (gate / checks.CHECKS_FILE).exists()

    def test_verify_writes_nothing(self, monkeypatch, gate):
        (gate / checks.CHECKS_FILE).write_text(
            json.dumps(_canned(checks), indent=2) + "\n", encoding="utf-8")
        before = (gate / checks.CHECKS_FILE).read_bytes()
        code = _run(monkeypatch, gate, ["--model-key", "target_x", "--verify"])
        assert code == 0
        assert (gate / checks.CHECKS_FILE).read_bytes() == before

    def test_verify_writes_nothing_even_when_the_filed_report_disagrees(
            self, monkeypatch, gate):
        # The regression the finding describes: a differing report used to be
        # silently replaced, so the disagreement could never be observed twice.
        filed = _canned(checks)
        filed["coverage"]["n_families"] = 99
        (gate / checks.CHECKS_FILE).write_text(
            json.dumps(filed, indent=2) + "\n", encoding="utf-8")
        before = (gate / checks.CHECKS_FILE).read_bytes()
        code = _run(monkeypatch, gate, ["--model-key", "target_x", "--verify"])
        assert code == 1
        assert (gate / checks.CHECKS_FILE).read_bytes() == before

    def test_write_report_is_the_flag_that_generates(self, monkeypatch, gate):
        result = _canned(checks)
        code = _run(monkeypatch, gate,
                    ["--model-key", "target_x", "--write-report"], result)
        assert code == 0
        written = json.loads(
            (gate / checks.CHECKS_FILE).read_text(encoding="utf-8"))
        assert written == result

    def test_the_two_modes_are_mutually_exclusive(self, monkeypatch, gate):
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, gate,
                 ["--model-key", "target_x", "--verify", "--write-report"])
        assert excinfo.value.code == 2
        assert not (gate / checks.CHECKS_FILE).exists()

    def test_no_committed_report_to_compare_against_is_said_so(
            self, monkeypatch, gate):
        code = _run(monkeypatch, gate, ["--model-key", "target_x", "--verify"])
        assert code == 2

    def test_an_unverifiable_section_is_incomplete_in_every_mode(
            self, monkeypatch, gate):
        result = _canned(checks, media_verifiable=False)
        assert result["verdict"] == "PASS_WITH_UNVERIFIED_SECTIONS"
        assert _run(monkeypatch, gate, ["--model-key", "target_x"],
                    result) == 3
        assert _run(monkeypatch, gate,
                    ["--model-key", "target_x", "--write-report"], result) == 3
        assert _run(monkeypatch, gate,
                    ["--model-key", "target_x", "--verify"], result) == 3
        assert (gate / checks.CHECKS_FILE).exists()

    def test_a_report_filed_by_a_blind_checkout_is_not_a_disagreement(
            self, monkeypatch, gate):
        # The mirror image: filed without the media, re-read by a machine that
        # has it. This checkout checked strictly more and found nothing wrong,
        # so it reproduces the report rather than contradicting it.
        filed = _canned(checks, media_verifiable=False)
        (gate / checks.CHECKS_FILE).write_text(
            json.dumps(filed, indent=2) + "\n", encoding="utf-8")
        before = (gate / checks.CHECKS_FILE).read_bytes()
        code = _run(monkeypatch, gate, ["--model-key", "target_x", "--verify"],
                    _canned(checks))
        assert code == 0
        assert (gate / checks.CHECKS_FILE).read_bytes() == before

    def test_a_failure_exits_nonzero_whatever_the_mode(self, monkeypatch, gate):
        result = _canned(checks, failures=("image differs",))
        assert result["verdict"] == "FAIL"
        for argv in (["--model-key", "target_x"],
                     ["--model-key", "target_x", "--verify"],
                     ["--model-key", "target_x", "--write-report"]):
            (gate / checks.CHECKS_FILE).unlink(missing_ok=True)
            assert _run(monkeypatch, gate, argv, result) == 1


class TestTheWriteIsStructurallyBehindTheFlag:
    """A behavioural test can miss a second write path; this cannot."""

    def test_main_writes_the_report_in_exactly_one_place_under_one_flag(self):
        tree = ast.parse(
            (ROOT / "scripts" / "iter11_replay_checks.py").read_text(
                encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        writes = [node for node in ast.walk(main)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "write_text"]
        assert len(writes) == 1, \
            f"main() has {len(writes)} write_text call(s); the report must be " \
            f"written from one place only"
        guard = next(node for node in ast.walk(main)
                     if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Attribute)
                     and node.test.attr == "write_report")
        guarded_lines = {line for stmt in guard.body
                         for line in range(stmt.lineno,
                                           (stmt.end_lineno or stmt.lineno) + 1)}
        assert writes[0].lineno in guarded_lines, \
            "the report is written outside the --write-report branch"

    def test_the_readme_no_longer_advertises_the_writing_invocation(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        invocations = [line for line in readme.splitlines()
                       if "iter11_replay_checks.py" in line
                       and "python3" in line]
        assert invocations, "the README no longer shows how to run the gate"
        for line in invocations:
            assert "--verify" in line or "--write-report" in line, \
                f"the README invokes the gate with no mode flag, which is the " \
                f"undocumented write: {line.strip()}"


# ---------------------------------------------------------------------------
# The manifest producer
# ---------------------------------------------------------------------------

@pytest.fixture
def producer(monkeypatch, tmp_path):
    """The producer, with every path it touches pointed at a tmp repository."""
    media = tmp_path / "data" / "media"
    media.mkdir(parents=True)
    panel = tmp_path / "panel.jsonl"
    panel.write_text("".join(
        json.dumps({"family_id": family_id, "variants": {
            variant["name"]: {"messages": variant["messages"]}
            for variant in (
                {"name": "cross_modal",
                 "messages": [{"images": [f"data/media/{family_id}.png"]}]},
                {"name": "text_only", "messages": [{"images": []}]})}})
        + "\n" for family_id in FAMILY_IDS), encoding="utf-8")
    for name in ("REPO_ROOT", "MEDIA_ROOT", "PANEL_PATH", "MANIFEST_PATH"):
        monkeypatch.setattr(manifest_producer, name, {
            "REPO_ROOT": tmp_path, "MEDIA_ROOT": media, "PANEL_PATH": panel,
            "MANIFEST_PATH": tmp_path / "media_manifest.json"}[name])
    monkeypatch.setattr(manifest_producer, "provenance",
                        lambda: {"code_commit": "0" * 40, "git_dirty": False})
    return tmp_path


class TestARollUpOverNothingIsNotAMeasurement:
    def test_an_empty_file_map_has_no_roll_up(self):
        # Every empty map hashes identically, so returning a digest here would
        # let two checkouts holding none of the media "agree".
        assert manifest_producer.rollup_sha256({}) is None

    def test_the_roll_up_does_not_depend_on_insertion_order(self):
        entries = {rel: {"sha256": _digest(payload), "bytes": len(payload)}
                   for rel, payload in IMAGES.items()}
        forward = manifest_producer.rollup_sha256(entries)
        backward = manifest_producer.rollup_sha256(
            {rel: entries[rel] for rel in reversed(sorted(entries))})
        assert forward == backward
        assert forward is not None

    def test_the_roll_up_binds_the_size_as_well_as_the_digest(self):
        entries = {rel: {"sha256": _digest(payload), "bytes": len(payload)}
                   for rel, payload in IMAGES.items()}
        shrunk = json.loads(json.dumps(entries))
        shrunk["data/media/CMST_000001.png"]["bytes"] -= 1
        assert manifest_producer.rollup_sha256(entries) \
            != manifest_producer.rollup_sha256(shrunk)

    def test_the_roll_up_moves_when_a_file_is_added(self):
        entries = {rel: {"sha256": _digest(payload), "bytes": len(payload)}
                   for rel, payload in IMAGES.items()}
        one = dict(entries)
        one.pop("data/media/CMST_000002.png")
        assert manifest_producer.rollup_sha256(entries) \
            != manifest_producer.rollup_sha256(one)


class TestTheManifestBindsTheTree:
    def test_it_binds_every_file_with_its_digest_and_size(self, producer):
        _world(producer)
        manifest = manifest_producer.build()
        assert manifest["scope"] == "whole_media_tree"
        assert manifest["n_files"] == 2
        assert sorted(manifest["files"]) == sorted(IMAGES)
        for rel, payload in IMAGES.items():
            assert manifest["files"][rel] == {
                "sha256": _digest(payload), "bytes": len(payload)}
        assert manifest["rollup_sha256"] \
            == manifest_producer.rollup_sha256(manifest["files"])

    def test_it_records_which_panel_images_it_does_not_bind(self, producer):
        _world(producer, on_disk=("data/media/CMST_000001.png",))
        manifest = manifest_producer.build()
        assert manifest["n_panel_referenced_images"] == 2
        assert manifest["panel_referenced_but_absent_from_the_manifest"] == [
            "data/media/CMST_000002.png"]

    def test_panel_only_binds_the_referenced_images_and_nothing_else(
            self, producer):
        _world(producer)
        (producer / "data" / "media" / "unreferenced.png").write_bytes(b"x")
        manifest = manifest_producer.build(panel_only=True)
        assert manifest["scope"] == "panel_referenced"
        assert sorted(manifest["files"]) == sorted(IMAGES)

    def test_it_refuses_to_write_a_manifest_over_an_empty_tree(
            self, monkeypatch, producer):
        manifest_path = producer / "media_manifest.json"
        assert manifest_producer.main([]) == 2
        assert not manifest_path.exists()

    def test_a_written_manifest_verifies_clean(self, producer):
        _world(producer)
        assert manifest_producer.main([]) == 0
        assert manifest_producer.main(["--verify"]) == 0

    def test_a_tampered_file_fails_verification(self, producer):
        _world(producer)
        manifest_producer.main([])
        _world(producer, tampered=("data/media/CMST_000001.png",))
        code, issues = manifest_producer.verify(producer / "media_manifest.json")
        assert code == 1
        assert any("data/media/CMST_000001.png" in i and "sha256" in i
                   for i in issues)

    def test_media_the_manifest_never_bound_fails_verification(self, producer):
        _world(producer)
        manifest_producer.main([])
        (producer / "data" / "media" / "extra.png").write_bytes(b"unbound")
        code, issues = manifest_producer.verify(producer / "media_manifest.json")
        assert code == 1
        assert any("does not bind it" in i for i in issues)

    def test_a_missing_file_is_incomplete_not_a_disagreement(self, producer,
                                                            capsys):
        _world(producer)
        manifest_producer.main([])
        (producer / "data" / "media" / "CMST_000002.png").unlink()
        code, issues = manifest_producer.verify(producer / "media_manifest.json")
        assert code == 3
        assert "absent" in issues[0]
        said = capsys.readouterr().out
        assert "NOT COMPARABLE" in said
        assert "SUBSET" in said
        # A subset roll-up must not be offered as the bound value's rival.
        assert "!= the bound" not in said

    def test_no_manifest_to_verify_against_is_its_own_answer(self, producer):
        code, issues = manifest_producer.verify(
            producer / "media_manifest.json")
        assert code == 2
        assert "no media manifest" in issues[0]

    def test_a_manifest_binding_nothing_is_not_a_measurement(self, producer):
        path = producer / "media_manifest.json"
        path.write_text(json.dumps({"files": {}, "rollup_sha256": None}),
                        encoding="utf-8")
        code, issues = manifest_producer.verify(path)
        assert code == 1
        assert "not a measurement of anything" in issues[0]
