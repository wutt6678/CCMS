#!/usr/bin/env python3
"""Iteration 11 closeout: the evidence manifest, and a verifier that re-derives it.

Scale-C closed Iteration 10 with ``scripts/scale_c_closeout_manifest.py``, which
binds each artifact by SHA-256 AND by the commit that last touched it. That is a
stronger binding than a hash alone and a weaker document than this one needs to
be: ``git_commit`` and ``generated_from_commit`` are properties of the checkout
that produced the manifest, so a manifest carrying them cannot be re-derived
exactly anywhere else, and on a fresh clone -- the case the anonymous
reproducibility package is built for -- the recorded commits may not be present
at all. A verifier that cannot resolve them has to choose between failing and
skipping, and both of those answers are wrong: failing punishes a correct
checkout for its history, and skipping silently weakens the claim.

So this manifest binds three things and only three: ``path``, ``sha256`` and
``bytes``. It carries no timestamp, no commit and no tree state. The consequence
is that ``--verify`` REBUILDS the whole document from the files on disk and
compares it byte for byte against what is filed, rather than walking a list of
fields somebody remembered to exclude from the comparison. Committed-ness is
still checked -- every bound file is resolved as a blob at HEAD and hashed there
too -- but it is checked as a SEPARATE claim, so that a checkout without an
object store can report the hashes as sound and the committed-ness as not
checkable here, which is exit 3 rather than exit 1.

Nothing here is transcribed. The bound set is DISCOVERED from the repository at
generation time -- every tracked file under ``outputs/iteration_11/``, every
``scripts/iter11_*.py``, every ``tests/unit/test_iter11_*.py``, plus the library
modules those verifiers import -- and the discovery is repeated at verification
time wherever a git object store exists, so a file committed and left out of the
manifest is a finding rather than a gap nobody notices.

Exit codes, and each one is tested:
    0  filed, or verified with every bound file present, hashing correctly, and
       resolving out of the object store
    1  a contradiction: a bound file does not hash to what is filed, the filed
       set and the committed set disagree, the document does not re-derive, or
       the tree was dirty when it was generated
    2  nothing filed to verify against, or a bound file is absent from disk
    3  every hash checks out but committed-ness is not checkable here -- no git
       object store, so this is an export or a tarball rather than a checkout.
       The evidence is sound and one claim about it cannot be made from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "outputs" / "iteration_11" / "closeout" \
    / "iteration_11_evidence_manifest.json"

#: Everything tracked under here is bound. The manifest itself lives in this
#: tree and is excluded by name, because a document cannot carry its own hash.
BOUND_TREES = ("outputs/iteration_11/",)

#: Discovered by pattern rather than listed, so a new stage in this iteration is
#: bound by being committed and does not have to be remembered here.
BOUND_GLOBS = ("scripts/iter11_*.py", "tests/unit/test_iter11_*.py")

#: The library modules the bound verifiers import. Listed rather than globbed:
#: binding all of ``src/`` would make the manifest move on every unrelated
#: change, and binding none of it would leave the tolerance rules, the
#: environment certification and the frozen bootstrap outside the evidence.
BOUND_PATHS = (
    "src/causal_mllm/seeds.py",
    "src/causal_mllm/evaluation/bootstrap.py",
    "src/causal_mllm/evaluation/censoring.py",
    "src/causal_mllm/evaluation/llm_judge.py",
    "src/causal_mllm/replay/confirmatory.py",
    "src/causal_mllm/replay/registry.py",
    "src/causal_mllm/replay/reproduction.py",
    "src/causal_mllm/replay/runner.py",
    "tests/unit/test_environment_reconstructibility.py",
    "tests/unit/test_llm_judge_fixes.py",
    "tests/unit/test_replay.py",
    "tests/unit/test_replay_evidence.py",
)

#: Where a reviewer starts, and what verifies each of those places. Every
#: ``verified_by`` here is checked to exist, to be bound, and to actually
#: implement ``--verify``: a pointer to a verifier that has no verify mode is
#: the same kind of claim as a citation to a commit nobody can reach.
ENTRY_POINTS = (
    {
        "path": "outputs/iteration_11/protocol/iteration_11_protocol.json",
        "what_it_establishes":
            "the frozen protocol: the 100-family panel, the six variants, the "
            "hypotheses and the decision rule, sealed before any target was "
            "replayed",
        "verified_by": "scripts/iter11_replay_checks.py",
    },
    {
        "path": "outputs/iteration_11/media_manifest.json",
        "what_it_establishes":
            "the bytes of the images the panel and the judges were shown, bound "
            "by hash because data/media is gitignored and a fresh checkout does "
            "not have them",
        "verified_by": "scripts/iter11_write_media_manifest.py",
    },
    {
        "path": "outputs/iteration_11/preflight/dependency_lock_reconstruction.json",
        "what_it_establishes":
            "what the certified environment can and cannot be rebuilt from: the "
            "freeze's preimage is authenticated, and the freeze format cannot "
            "express the builds that produced it",
        "verified_by": "scripts/iter11_write_dependency_lock.py",
    },
    {
        "path": "outputs/iteration_11/analysis/cross_model/cross_model_analysis.json",
        "what_it_establishes":
            "the confirmatory analysis on the 98-family common panel: four "
            "sign-transport hypotheses, Holm-Bonferroni corrected, with every "
            "null and every reversal retained",
        "verified_by": "scripts/iter11_cross_model_analysis.py",
    },
    {
        "path": "outputs/iteration_11/analysis/differential_censoring/"
                  "differential_censoring_bound.json",
        "what_it_establishes":
            "an exact range rather than a point estimate for the one "
            "differentially missing label: each resample mean is affine in the "
            "score that label could have taken, so the verdicts are enumerated "
            "over the rubric's whole 0.0-1.0 instead of sampled",
        "verified_by": "scripts/iter11_differential_censoring_bound.py",
    },
    {
        "path": "outputs/iteration_11/analysis/differential_censoring/"
                "labels_adjudicated.sensitivity_99f.json",
        "what_it_establishes":
            "the 99-family sensitivity labels the bound ranges over, re-filed "
            "without spending a single new adjudicator call",
        "verified_by": "scripts/iter11_adjudicate_sensitivity_cell.py",
    },
    {
        "path": "outputs/iteration_11/analysis/differential_censoring/"
                "call_receipts.sensitivity_99f.json",
        "what_it_establishes":
            "the preserved evidence for those two reused calls: request hashes, "
            "provider response ids and the labels they produced, committed so "
            "that a citation to them resolves out of the object store instead "
            "of pointing at a commit nobody can reach",
        "verified_by": "scripts/iter11_adjudicate_sensitivity_cell.py",
    },
    {
        "path": "outputs/iteration_11/closeout/"
                "iteration_11_transportability_decision.json",
        "what_it_establishes":
            "the transportability call, derived from the two analyses above "
            "rather than written down beside them",
        "verified_by": "scripts/iter11_transportability_decision.py",
    },
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    ``raise SystemExit("text")`` exits 1 whatever the text says, and 1 is this
    script's code for a CONTRADICTION. A manifest that has not been filed yet
    reported as a contradiction sends whoever reads the exit code to look for a
    disagreement between evidence that was never bound.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", *args), cwd=REPO_ROOT, capture_output=True,
                          text=True)


def object_store_available() -> bool:
    """Can this checkout answer a question about what HEAD holds?

    A tarball, an export and the anonymous reproducibility package all have the
    files and none of the history. That is a statement about the checkout and
    not about the evidence, so it gets its own exit code rather than borrowing
    the one that means something is wrong.
    """
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return False
    head = _git("rev-parse", "--verify", "HEAD")
    return head.returncode == 0


def discover() -> list[str]:
    """The bound set, as the repository records it. Requires an object store."""
    if not object_store_available():
        fatal("no git object store here, so the bound set cannot be discovered "
              "from what is committed. --verify still checks every path the "
              "filed manifest names; generating a new one needs the history", 3)
    paths: set[str] = set()
    for tree in BOUND_TREES:
        found = _git("ls-files", "--", tree)
        if found.returncode != 0:
            fatal(f"git ls-files failed for {tree}: {found.stderr.strip()}", 1)
        paths.update(line for line in found.stdout.splitlines() if line)
    for pattern in BOUND_GLOBS:
        found = _git("ls-files", "--", pattern)
        if found.returncode != 0:
            fatal(f"git ls-files failed for {pattern}: {found.stderr.strip()}", 1)
        paths.update(line for line in found.stdout.splitlines() if line)
    for rel in BOUND_PATHS:
        tracked = _git("ls-files", "--error-unmatch", "--", rel)
        if tracked.returncode != 0:
            fatal(f"{rel} is named in BOUND_PATHS but is not tracked, so it "
                  f"cannot be part of the committed evidence", 1)
        paths.add(rel)
    paths.discard(_rel(OUT_PATH))
    return sorted(paths)


def role_for(rel: str) -> str:
    """What kind of thing a bound path is, derived from where it lives."""
    if rel == "outputs/iteration_11/media_manifest.json":
        return "binding_for_the_untracked_media"
    if rel == "outputs/iteration_11/preflight/dependency_lock_reconstruction.json":
        return "environment_reconstruction_claim"
    for prefix, role in (
            ("outputs/iteration_11/protocol/", "frozen_protocol"),
            ("outputs/iteration_11/preflight/", "environment_and_model_preflight"),
            ("outputs/iteration_11/eligibility/", "eligibility_gate"),
            ("outputs/iteration_11/generations/", "model_generation_metadata"),
            ("outputs/iteration_11/judge_vision_ablation/", "judge_vision_ablation"),
            ("outputs/iteration_11/judge/", "judge_labels_and_provenance"),
            ("outputs/iteration_11/analysis/differential_censoring/",
             "differential_censoring_sensitivity"),
            ("outputs/iteration_11/analysis/", "sealed_analysis"),
            ("outputs/iteration_11/closeout/", "closeout"),
            ("outputs/iteration_11/diagnostics/", "diagnostics"),
            ("outputs/iteration_11/judgments/", "placeholder"),
            ("outputs/iteration_11/reports/", "placeholder"),
            ("scripts/", "verifier_or_generator"),
            ("tests/", "test"),
            ("src/", "library_the_verifiers_import")):
        if rel.startswith(prefix):
            return role
    return "evidence"


def roll_up(entries: list[dict]) -> str:
    """One hash over the whole binding, so a reviewer can compare one number."""
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\n{entry['sha256']}\n{entry['bytes']}\n"
                      .encode("utf-8"))
    return digest.hexdigest()


def media_binding() -> dict:
    """What the bound media manifest itself binds, read out of it.

    The manifest is the only file in the bound set that stands for bytes this
    repository does not track, so the count and the roll-up it files are read
    here rather than quoted, and a manifest that cannot be read is a finding.
    """
    path = REPO_ROOT / "outputs" / "iteration_11" / "media_manifest.json"
    if not path.is_file():
        return {"readable_here": False}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        "readable_here": True,
        "n_files_it_binds": doc.get("n_files"),
        "total_bytes_it_binds": doc.get("total_bytes"),
        "its_rollup_sha256": doc.get("rollup_sha256"),
        "n_panel_referenced_images": doc.get("n_panel_referenced_images"),
        "panel_referenced_but_absent_from_the_manifest":
            doc.get("panel_referenced_but_absent_from_the_manifest"),
        "media_root": doc.get("media_root"),
    }


def build(paths: list[str],
          entry_points: tuple[dict, ...] | list[dict] = ENTRY_POINTS) -> dict:
    """The manifest, as a pure function of a path list, its entry points, and disk.

    ``entry_points`` is a parameter rather than a constant read from the module
    because the list is part of the document's design and is not re-derivable
    from the files: verification passes back the one that is filed, so the
    comparison that follows is about the hashes and not about whether a reviewer
    was pointed at the same eight artifacts. What the entry points claim IS
    checked -- by :func:`check_the_manifest`, against the bound set and against
    a verifier that really implements ``--verify``.
    """
    entries = []
    for rel in paths:
        path = REPO_ROOT / rel
        if not path.is_file():
            fatal(f"{rel} is bound by this manifest but is not a file on disk "
                  f"here", 2)
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "role": role_for(rel)})
    roles: dict[str, int] = {}
    for entry in entries:
        roles[entry["role"]] = roles.get(entry["role"], 0) + 1
    media = media_binding()
    return {
        "kind": "iteration_11_evidence_manifest_v1",
        "produced_by": "scripts/iter11_closeout_evidence_manifest.py",
        "question": "what is the evidence for Iteration 11, by hash, and can "
                    "this checkout say so",
        "n_bound": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "rollup_sha256": roll_up(entries),
        "n_by_role": dict(sorted(roles.items())),
        "bound": entries,
        "where_a_reviewer_starts": list(entry_points),
        "bound_by_discovery_not_by_a_list": {
            "trees": list(BOUND_TREES),
            "patterns": list(BOUND_GLOBS),
            "named_paths": list(BOUND_PATHS),
            "what_that_buys": (
                "a file committed under one of these and left out of the "
                "manifest is a finding at verification time rather than a gap "
                "nobody notices, and a new stage in this iteration is bound by "
                "being committed rather than by being remembered here"),
        },
        "what_this_manifest_does_not_bind": {
            "itself": {
                "path": _rel(OUT_PATH),
                "why": "a document cannot carry its own hash. Its integrity is "
                       "the re-derivation: --verify rebuilds the whole thing "
                       "from the files on disk and compares, so tampering with "
                       "the manifest is tampering with a claim that the bound "
                       "files then contradict",
            },
            "the_media": {
                "n_files": media.get("n_files_it_binds"),
                "bound_indirectly_by": "outputs/iteration_11/media_manifest.json",
                "why": (
                    "data/media is gitignored apart from twenty individually "
                    "negated source images, so the bytes are not in this "
                    "repository and no manifest committed here can bind them "
                    "directly. The media manifest binds them by hash instead, "
                    "and is itself bound here -- which is why a checkout "
                    "without the images can still verify everything committed "
                    "and reports the media section as not verifiable here "
                    "rather than as a failure"),
                "the_manifest_says": media,
            },
            "commit_and_tree_state": {
                "why": (
                    "no generated_at, no code_commit and no git_dirty appears in "
                    "this document. Those are properties of the checkout that "
                    "wrote it, so carrying them would make exact re-derivation "
                    "impossible anywhere else and would reduce --verify to "
                    "comparing a list of fields somebody had to remember to "
                    "keep short. Committed-ness is checked as a separate claim, "
                    "against the object store, at verification time"),
            },
        },
    }


def _head_tree() -> dict[str, str]:
    """HEAD's tree as ``{path: blob oid}``, empty if there is no object store.

    One ``ls-tree`` for the whole repository rather than one ``rev-parse`` per
    bound path: the bound set is a couple of hundred files and a verifier that
    takes ten seconds to run is a verifier somebody starts skipping.
    """
    listing = subprocess.run(("git", "ls-tree", "-r", "-z", "HEAD"),
                             cwd=REPO_ROOT, capture_output=True)
    if listing.returncode != 0:
        return {}
    tree: dict[str, str] = {}
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        meta, _, rel = record.partition(b"\t")
        parts = meta.split()
        if len(parts) != 3:
            continue
        tree[rel.decode("utf-8", "surrogateescape")] = parts[2].decode("ascii")
    return tree


def _blob_sha256s(oids: list[str]) -> dict[str, str | None]:
    """The SHA-256 of each blob's CONTENT, in one ``git cat-file --batch``.

    Not the blob's git object id, which is a SHA-1 over a length-prefixed
    header and the bytes: the manifest binds the bytes a reviewer would hash on
    disk, so the comparison has to be against the same function of the same
    bytes.
    """
    if not oids:
        return {}
    proc = subprocess.run(("git", "cat-file", "--batch"), cwd=REPO_ROOT,
                          input="\n".join(oids).encode("ascii"),
                          capture_output=True)
    if proc.returncode != 0:
        return dict.fromkeys(oids)
    stream, out, pos = proc.stdout, {}, 0
    for oid in oids:
        newline = stream.find(b"\n", pos)
        if newline < 0:
            out[oid] = None
            continue
        parts = stream[pos:newline].split()
        pos = newline + 1
        if len(parts) == 2 and parts[1] == b"missing":
            out[oid] = None
            continue
        if len(parts) != 3:
            out[oid] = None
            continue
        size = int(parts[2])
        out[oid] = hashlib.sha256(stream[pos:pos + size]).hexdigest()
        pos += size + 1  # the trailing newline --batch writes after the content
    return out


def committed_hashes(paths: list[str]) -> dict[str, str | None]:
    """``{path: sha256 of HEAD's blob at path}``, None where HEAD has no such path."""
    tree = _head_tree()
    oids = {path: tree[path] for path in paths if path in tree}
    hashed = _blob_sha256s(sorted(set(oids.values())))
    return {path: hashed.get(oid) for path, oid in oids.items()} | \
        {path: None for path in paths if path not in oids}


def check_the_manifest(doc: dict) -> list[str]:
    """Does the filed document agree with itself?

    Structural checks only: the counts against the entries, the roll-up against
    the entries, the roles against the paths, and every entry point against the
    bound set and against a verifier that really has a verify mode. A manifest
    is the artifact whose whole value is that it does not drift, so its internal
    agreement is checked rather than assumed.
    """
    issues: list[str] = []
    entries = doc.get("bound") or []
    paths = [entry.get("path") for entry in entries]

    if doc.get("n_bound") != len(entries):
        issues.append(f"n_bound is {doc.get('n_bound')} and {len(entries)} "
                      f"files are bound")
    if doc.get("total_bytes") != sum(entry.get("bytes") or 0 for entry in entries):
        issues.append(f"total_bytes is {doc.get('total_bytes')} and the bound "
                      f"entries sum to "
                      f"{sum(entry.get('bytes') or 0 for entry in entries)}")
    if paths != sorted(paths):
        issues.append("the bound entries are not sorted by path, so two "
                      "manifests binding the same set would not be identical")
    if len(set(paths)) != len(paths):
        duplicates = sorted({p for p in paths if paths.count(p) > 1})
        issues.append(f"bound more than once: {duplicates}")
    if doc.get("rollup_sha256") != roll_up(entries):
        issues.append("rollup_sha256 does not hash to the bound entries it "
                      "summarises")

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.get("role")] = counts.get(entry.get("role"), 0) + 1
    if doc.get("n_by_role") != dict(sorted(counts.items())):
        issues.append(f"n_by_role is {doc.get('n_by_role')} and the entries say "
                      f"{dict(sorted(counts.items()))}")
    for entry in entries:
        if entry.get("role") != role_for(entry.get("path", "")):
            issues.append(
                f"{entry.get('path')}: filed under role "
                f"{entry.get('role')!r} and its path puts it under "
                f"{role_for(entry.get('path', ''))!r}")

    if _rel(OUT_PATH) in paths:
        issues.append(f"this manifest binds {_rel(OUT_PATH)}, itself, which no "
                      f"document can do by hash")

    bound = set(paths)
    for point in doc.get("where_a_reviewer_starts") or ():
        target = point.get("path")
        if target not in bound:
            issues.append(f"the entry point {target} is not among the bound "
                          f"files, so a reviewer following it leaves the "
                          f"manifest's guarantee")
        verifier = point.get("verified_by")
        if verifier not in bound:
            issues.append(f"the entry point {target} names {verifier} as its "
                          f"verifier and that script is not bound")
            continue
        source = (REPO_ROOT / verifier).read_text(encoding="utf-8") \
            if (REPO_ROOT / verifier).is_file() else ""
        if '"--verify"' not in source:
            issues.append(
                f"{verifier} is named as what verifies {target} and implements "
                f"no --verify mode: a pointer to a verifier that cannot verify "
                f"is the same claim as a citation nobody can resolve")
    return issues


def verify(path: Path | None = None) -> tuple[int, str, list[str], list[str]]:
    """``(code, conclusion, issues, unverifiable_here)``. Writes nothing.

    ``issues`` are contradictions and ``unverifiable_here`` are claims this
    checkout cannot make. They are kept in separate lists because conflating
    them is what made a fresh checkout without the media report a failure
    instead of a true statement about itself.
    """
    path = OUT_PATH if path is None else path
    if not path.is_file():
        return 2, "not_filed", [
            f"no evidence manifest at {_rel(path)}; run this script with "
            f"--write to file one"], []

    filed = json.loads(path.read_text(encoding="utf-8"))
    issues = check_the_manifest(filed)

    entries = filed.get("bound") or []
    absent = [entry["path"] for entry in entries
              if not (REPO_ROOT / entry["path"]).is_file()]
    if absent:
        return 2, "bound_files_absent", issues + [
            f"{len(absent)} bound file(s) are absent from disk here: "
            f"{', '.join(absent[:8])}"
            + (" ..." if len(absent) > 8 else "")], []

    try:
        fresh = build([entry["path"] for entry in entries],
                      filed.get("where_a_reviewer_starts") or [])
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), \
            "could_not_rederive", issues + [
                f"the manifest could not be re-derived from the files on disk: "
                f"exit {exc.code}"], []

    for key in sorted(set(filed) | set(fresh)):
        if filed.get(key) != fresh.get(key):
            issues.append(
                f"{key}: filed "
                f"{json.dumps(filed.get(key), sort_keys=True)[:200]} != "
                f"re-derived {json.dumps(fresh.get(key), sort_keys=True)[:200]}")

    unverifiable: list[str] = []
    if not object_store_available():
        unverifiable.append(
            "no git object store here, so two claims could not be made: that "
            "every bound file resolves as a blob at HEAD and hashes the same "
            "there, and that nothing committed under the bound trees or "
            "patterns was left out of this manifest. Every hash on disk "
            "checks out; what cannot be said from here is that these files are "
            "the committed ones")
        if issues:
            return 1, "differs", issues, unverifiable
        return 3, "hashes_sound_committedness_uncheckable", [], unverifiable

    mismatched, uncommitted = [], []
    paths = [entry["path"] for entry in entries]
    at_head = committed_hashes(paths)
    for entry in entries:
        blob = at_head.get(entry["path"])
        if blob is None:
            uncommitted.append(entry["path"])
        elif blob != entry["sha256"]:
            mismatched.append(
                f"{entry['path']}: disk {entry['sha256'][:12]} HEAD blob "
                f"{blob[:12]}")
    if mismatched:
        issues.append(f"{len(mismatched)} bound file(s) differ from what HEAD "
                      f"holds: " + "; ".join(mismatched[:8])
                      + (" ..." if len(mismatched) > 8 else ""))
    if uncommitted:
        issues.append(f"{len(uncommitted)} bound file(s) are not in HEAD at all: "
                      f"{', '.join(uncommitted[:8])}"
                      + (" ..." if len(uncommitted) > 8 else ""))

    committed = set(discover())
    bound = {entry["path"] for entry in entries}
    left_out = sorted(committed - bound)
    bound_but_uncommitted = sorted(bound - committed)
    if left_out:
        issues.append(f"{len(left_out)} file(s) are committed under a bound tree "
                      f"or pattern and are not in this manifest, so the "
                      f"manifest is not the closeout it claims to be: "
                      f"{', '.join(left_out[:8])}"
                      + (" ..." if len(left_out) > 8 else "")
                      + ". Re-run with --write")
    if bound_but_uncommitted:
        issues.append(f"{len(bound_but_uncommitted)} bound file(s) are not "
                      f"tracked: {', '.join(bound_but_uncommitted[:8])}")

    if issues:
        return 1, "differs", issues, unverifiable
    return 0, "re_derived_exactly_and_every_file_is_committed", [], []


def _assert_clean_tree() -> None:
    """Refuse to file a manifest from a tree that is not what HEAD holds.

    The document records no commit, so this is the only place the precondition
    can be enforced: binding working-tree bytes that are never committed
    produces a manifest that fails on the next checkout for a reason that looks
    like tampering. Only the manifest's own path is excluded, and nothing else
    -- excluding the directory it lives in would let a modified decision
    artifact beside it be bound from the working tree while this function still
    reported the tree as clean.
    """
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        fatal(f"git status failed: {status.stderr.strip()}", 1)
    self_rel = _rel(OUT_PATH)
    dirty = [line for line in status.stdout.splitlines()
             if line[3:].strip().strip('"') != self_rel]
    if dirty:
        fatal("the working tree is not what HEAD holds, so a manifest filed "
              "now would bind bytes that are not committed: "
              + "; ".join(dirty[:8]) + (" ..." if len(dirty) > 8 else ""), 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="re-derive and compare with the filed manifest, "
                           "writing nothing")
    mode.add_argument("--write", action="store_true",
                      help="file the manifest from a clean tree. Explicit, "
                           "because it overwrites a committed artifact")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.verify or not args.write:
        code, conclusion, issues, unverifiable = verify(args.out)
        if code == 2:
            print(f"EVIDENCE MANIFEST: NOT VERIFIABLE -- {issues[0]}")
            return 2
        if code == 3:
            print(f"EVIDENCE MANIFEST: INCOMPLETE HERE -- "
                  f"{conclusion.replace('_', ' ')}")
            filed = json.loads(args.out.read_text(encoding="utf-8"))
            print(f"  bound      {filed['n_bound']} files, "
                  f"{filed['total_bytes']} bytes")
            print(f"  roll-up    {filed['rollup_sha256']}")
            for note in unverifiable:
                print(f"  not checkable here: {note}")
            return 3
        if code == 1:
            print(f"EVIDENCE MANIFEST: FAIL ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            for note in unverifiable:
                print(f"  not checkable here: {note}")
            return 1
        filed = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"EVIDENCE MANIFEST: VERIFIED -- {conclusion.replace('_', ' ')}")
        print(f"  bound      {filed['n_bound']} files, "
              f"{filed['total_bytes']} bytes, in "
              f"{len(filed['n_by_role'])} roles")
        print(f"  roll-up    {filed['rollup_sha256']}")
        for role, count in filed["n_by_role"].items():
            print(f"    {count:4d}  {role}")
        print(f"  entry points {len(filed['where_a_reviewer_starts'])}, each "
              f"bound and each with a verifier that implements --verify")
        return 0

    _assert_clean_tree()
    paths = discover()
    doc = build(paths)
    problems = check_the_manifest(doc)
    if problems:
        print("FAIL: refusing to file a manifest that disagrees with itself",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(args.out)}")
    print(f"  bound      {doc['n_bound']} files, {doc['total_bytes']} bytes")
    print(f"  roll-up    {doc['rollup_sha256']}")
    for role, count in doc["n_by_role"].items():
        print(f"    {count:4d}  {role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
