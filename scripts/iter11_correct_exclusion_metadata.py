#!/usr/bin/env python3
"""Correct the exclusion metadata beside the 21 sealed artifacts that carry it.

The judge and evaluation artifacts filed this claim under one key:

    "outcome_independent": "which cells a provider refuses is a function of the
     request bytes, so the surviving family set was fixed before any label
     existed"

The second half is true. The first half is false, and the two producers that
wrote it now file them as two fields -- ``label_blind`` and
``response_dependent`` -- from ``causal_mllm.evaluation.censoring``.

That fixes every artifact generated from here on. It does nothing for the 21
already committed, and they are NOT rewritten:

  * they are sealed judge and evaluation evidence. Regenerating a
    ``final_evaluation_report.json`` means re-running a pipeline whose
    adjudication pass re-calls the adjudicator, so the "same" artifact would
    come back with different bytes and a different provenance, and the seal
    that makes it evidence would be gone;
  * rewriting them would delete the record that the claim was ever made. A
    correction that erases the thing it corrects cannot be checked.

So this script enumerates them instead: every file under ``outputs/`` carrying
the superseded key, with its SHA-256, its occurrence count, and the superseded
sentence quoted verbatim, each paired with the wording that replaces it. The
measurement that settles the correction is copied out of the probe artifact
rather than restated, so this file stands alone.

``--verify`` then makes the enumeration load-bearing rather than decorative:

  * a NEW artifact carrying the old key fails -- the rename is enforced forward;
  * an enumerated artifact whose SHA-256 moved fails -- the seal is enforced
    backward, so the correction cannot be "applied" by quietly rewriting the
    evidence it corrects;
  * a producer source that emits the old key again fails.

Usage:
    python3 scripts/iter11_correct_exclusion_metadata.py            # write it
    python3 scripts/iter11_correct_exclusion_metadata.py --verify   # no write

Exit codes: 0 the enumeration matches the tree and the producers are clean;
1 a carrier appeared, disappeared, changed, or a producer regressed; 2 no
correction artifact to verify against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.censoring import (  # noqa: E402
    CORRECTION_ARTIFACT,
    DIFFERENTIAL_CELL,
    DIFFERENTIAL_TARGET,
    EVIDENCE_ARTIFACT,
    SUPERSEDED_KEY,
    UNIFORM_CELLS,
    exclusion_metadata,
)
from causal_mllm.seeds import code_tree_status, get_git_commit  # noqa: E402

CORRECTION_PATH = REPO_ROOT / CORRECTION_ARTIFACT
EVIDENCE_PATH = REPO_ROOT / EVIDENCE_ARTIFACT

#: Where the superseded key is looked for. ``outputs/`` holds the evidence,
#: ``README.md`` and ``docs/`` hold the prose that describes it: a claim
#: corrected in the artifacts but still asserted in the README is not corrected.
SCAN_DIRS = ("outputs", "docs")
SCAN_FILES = ("README.md",)

#: The two sources that used to write the key. Both now import
#: ``exclusion_metadata``, and ``--verify`` fails if either grows it back.
PRODUCERS = ("scripts/run_llm_judge_pipeline.py",
             "src/causal_mllm/evaluation/runner.py")

#: This stage's own output, excluded from the tree dirtiness it RECORDS.
OWN_OUTPUT_PREFIXES = (CORRECTION_ARTIFACT,)

#: The key used as a JSON field, and the same claim asserted in prose. Both
#: count, and NEITHER matches the correction -- which quotes the key inside a
#: string value as ``'outcome_independent'`` and states the denial as "NOT
#: independent of the outcome".
#:
#: Matching the bare token instead is the obvious implementation and it is
#: wrong: the corrected wording has to name the wording it replaces, so every
#: artifact the fixed producers write would be flagged as a carrier, and the
#: forward enforcement would be a gate that fails on the repair it exists to
#: protect.
KEY_USE_RE = re.compile(r'"' + re.escape(SUPERSEDED_KEY) + r'"\s*:')
PROSE_CLAIM_RE = re.compile(r"outcome-independent")

#: The key's JSON value, matched in raw bytes. A regex over the text rather than
#: a JSON parse because the carriers include a 4.7 MB JSONL whose 588
#: occurrences sit in per-record metadata: parsing it to quote them would cost
#: more than the quotation is worth, and a parse failure on one malformed line
#: would hide the other 587.
_VALUE_RE = re.compile(
    r'"' + re.escape(SUPERSEDED_KEY) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"')


def carries_the_claim(text: str) -> tuple[int, int]:
    """How many times ``text`` makes the superseded claim, as a field and as
    prose.

    Counted rather than merely detected, because the enumeration is checked
    against the tree on every ``--verify`` and a carrier whose claim count moved
    is a carrier that was edited.
    """
    return (len(KEY_USE_RE.findall(text)), len(PROSE_CLAIM_RE.findall(text)))


def _rel(path: Path | str, base: Path | None = None) -> str:
    """Repo-relative, or relative to the tree being scanned.

    Relative to the scanned root rather than always to the repository so that a
    test's miniature tree enumerates ``outputs/x.json`` and not an absolute path
    under ``/tmp``: the enumeration's KEYS are part of what ``--verify``
    compares, and they must mean the same thing in both trees.
    """
    path = Path(path)
    base = REPO_ROOT if base is None else base
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_tracked(path: Path) -> bool | None:
    """Whether git has this path committed -- None when git cannot answer.

    Asked of git rather than inferred from a directory convention, because the
    distinction is the whole argument: these files are unrewritable BECAUSE they
    are committed evidence.

    None, not False, for a path outside the repository. ``git ls-files`` is run
    with the repository as its working directory, so handing it a scanned tree's
    relative path asks about a DIFFERENT file that happens to share the name --
    and "untracked" would then be a false claim about evidence rather than an
    admission that the question does not apply. A test's miniature tree gets
    None; the real scan gets a real answer for all 21.
    """
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    rel = _rel(path)
    proc = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode == 0


def _iter_scan_paths(root: Path | None = None) -> list[Path]:
    """Every file to look at, minus this stage's own output.

    The exclusion is not cosmetic. The correction artifact has to QUOTE the key
    and the sentence it replaces, so scanning it would find a carrier of the
    claim inside the file that corrects the claim, and every ``--verify`` would
    demand the correction be enumerated in itself. Excluding it is reported in
    the artifact rather than done silently, so the reader knows one path under
    ``outputs/`` was not looked at and why.
    """
    base = REPO_ROOT if root is None else root
    paths = [base / name for name in SCAN_FILES if (base / name).is_file()]
    for name in SCAN_DIRS:
        directory = base / name
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    return [p for p in paths
            if _rel(p, base) not in OWN_OUTPUT_PREFIXES]


def scan(root: Path | None = None) -> dict[str, dict]:
    """Every file making the superseded claim, by path relative to ``root``."""
    base = REPO_ROOT if root is None else root
    carriers: dict[str, dict] = {}
    for path in _iter_scan_paths(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        key_uses, prose_claims = carries_the_claim(text)
        if not key_uses and not prose_claims:
            continue
        rel = _rel(path, base)
        values = sorted({m.group(1) for m in _VALUE_RE.finditer(text)})
        carriers[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "tracked_in_git": is_tracked(path),
            "n_key_uses": key_uses,
            "n_prose_claims": prose_claims,
            "n_occurrences": key_uses + prose_claims,
            "n_quoted_values": len(values),
            "superseded_values": values,
        }
    return carriers


def scope_of(value: str) -> str:
    """Which exclusion set a given superseded sentence was describing.

    Both producers wrote the same claim about two different sets, and the
    correction has to name the set it is correcting or it reads as a generic
    disclaimer attached to everything.
    """
    return ("the surviving family set" if "surviving family set" in value
            else "the excluded set")


def measurement(evidence_path: Path | None = None,
                root: Path | None = None) -> dict:
    """The differential/uniform split, copied from the probe that measured it.

    Copied rather than restated: the correction asserts that the exclusion
    tracks the arm, and that assertion is only as good as the artifact behind
    it. ``--verify`` re-reads the probe and fails if these fields moved, so a
    correction cannot outlive its evidence.
    """
    evidence_path = _evidence_path(root) if evidence_path is None \
        else evidence_path
    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    status = doc["full_payload_status"]
    return {
        "source": _rel(evidence_path, root),
        "source_sha256": sha256_file(evidence_path),
        "probed_at": doc.get("probed_at"),
        "n_requests": doc.get("n_requests"),
        "full_payload_status": status,
        "arms_that_disagree_now": doc.get("arms_that_disagree_now"),
        "arms_unmeasured_now": doc.get("arms_unmeasured_now"),
        "differential_cell": DIFFERENTIAL_CELL,
        "differential_target": DIFFERENTIAL_TARGET,
        "refused_in_every_arm": list(UNIFORM_CELLS),
        "what_it_shows": {
            cell: {"n_arms_refusing": sum(1 for code in arms.values()
                                          if code != 200),
                   "arms": arms}
            for cell, arms in sorted(status.items())
        },
    }


def provenance() -> dict:
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    return {
        "produced_by": "scripts/iter11_correct_exclusion_metadata.py",
        "kind": "iteration_11_exclusion_metadata_correction_v1",
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
    }


def _evidence_path(root: Path | None = None) -> Path:
    """The probe artifact, resolved against the scanned tree.

    Resolved against ``root`` rather than always against the repository so that
    a test scanning a miniature tree checks a miniature tree's evidence, and so
    that ``--verify`` on a checkout without the probe says "absent" instead of
    quietly reading a different copy.
    """
    return EVIDENCE_PATH if root is None else root / EVIDENCE_ARTIFACT


def producer_state(root: Path | None = None) -> dict[str, dict]:
    """Whether the two producers still emit the superseded key."""
    base = REPO_ROOT if root is None else root
    state = {}
    for rel in PRODUCERS:
        path = base / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        state[rel] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "emits_superseded_key": bool(sum(carries_the_claim(text))),
            "imports_exclusion_metadata": "exclusion_metadata" in text,
        }
    return state


def build(root: Path | None = None) -> dict:
    carriers = scan(root)
    values = sorted({value for entry in carriers.values()
                     for value in entry["superseded_values"]})
    measured = measurement(root=root)
    n_tracked = sum(1 for e in carriers.values()
                    if e["tracked_in_git"] is True)
    return {
        "question": "the exclusion metadata claimed the exclusion was "
                    "outcome-independent; which committed artifacts still say "
                    "so, and what replaces it",
        **provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "superseded_key": SUPERSEDED_KEY,
        "replaced_by": ["label_blind", "response_dependent"],
        "single_source_of_the_correction":
            "src/causal_mllm/evaluation/censoring.py",
        "n_artifacts_carrying_the_superseded_key": len(carriers),
        "n_occurrences": sum(e["n_occurrences"] for e in carriers.values()),
        "n_untracked_carriers": sum(1 for e in carriers.values()
                                    if e["tracked_in_git"] is False),
        "n_carriers_git_cannot_answer_for": sum(
            1 for e in carriers.values() if e["tracked_in_git"] is None),
        "n_tracked_carriers": n_tracked,
        "artifacts": carriers,
        "distinct_superseded_values": values,
        "corrections": [
            {"superseded_value": value,
             "scope": scope_of(value),
             "written_by": sorted(
                 rel for rel, entry in carriers.items()
                 if value in entry["superseded_values"]),
             **exclusion_metadata(scope_of(value))}
            for value in values
        ],
        "measurement_behind_the_correction": measured,
        "producers": producer_state(root),
        "why_the_artifacts_are_not_rewritten": (
            f"{n_tracked} of {len(carriers)} artifact(s) carrying "
            f"'{SUPERSEDED_KEY}' are committed to git, and are enumerated here "
            "rather than edited. "
            "They are sealed judge and evaluation evidence: regenerating one "
            "means re-running a pipeline whose adjudication pass re-calls the "
            "adjudicator, so the artifact would come back with different bytes "
            "and different provenance and the seal that makes it evidence "
            "would be gone. Rewriting them would also delete the record that "
            "the claim was ever made, and a correction that erases the thing it "
            "corrects cannot be checked. --verify enforces both directions: a "
            "new carrier fails, and a carrier whose sha256 moved fails"),
    }


def verify(path: Path | None = None,
           root: Path | None = None) -> tuple[int, list[str]]:
    """Compare the committed correction against the tree, writing nothing."""
    path = CORRECTION_PATH if path is None else path
    if not path.exists():
        return 2, [f"no correction artifact at {_rel(path)}; run this script "
                   f"without --verify to write one"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []

    enumerated = doc.get("artifacts") or {}
    if not enumerated:
        issues.append(f"{_rel(path)} enumerates no artifact at all, so it "
                      f"corrects nothing it can be checked against")
    found = scan(root)

    for rel in sorted(set(found) - set(enumerated)):
        issues.append(
            f"{rel} carries '{SUPERSEDED_KEY}' "
            f"({found[rel]['n_occurrences']} occurrence(s)) and is not "
            f"enumerated in {_rel(path)}: either a producer grew the "
            f"superseded key back, or new evidence was filed with it")
    for rel in sorted(set(enumerated) - set(found)):
        issues.append(
            f"{rel} is enumerated but no longer carries "
            f"'{SUPERSEDED_KEY}': sealed evidence was rewritten, which is what "
            f"this artifact exists to prevent -- restore it and correct beside "
            f"it")
    for rel in sorted(set(enumerated) & set(found)):
        before, after = enumerated[rel], found[rel]
        if before.get("sha256") != after["sha256"]:
            issues.append(
                f"{rel} has sha256 {after['sha256']} but was enumerated as "
                f"{before.get('sha256')}: the sealed artifact changed under "
                f"its own correction")
        if before.get("n_occurrences") != after["n_occurrences"]:
            issues.append(
                f"{rel} now carries '{SUPERSEDED_KEY}' "
                f"{after['n_occurrences']} time(s), enumerated as "
                f"{before.get('n_occurrences')}")

    if doc.get("distinct_superseded_values") != found_values(found):
        issues.append(
            "the superseded sentences quoted in the correction are not the "
            f"ones on disk: quoted {doc.get('distinct_superseded_values')}, "
            f"found {found_values(found)}")

    corrections = doc.get("corrections") or []
    if not corrections:
        issues.append("the correction artifact carries no corrections")
    for entry in corrections:
        for field in ("label_blind", "response_dependent", "scope",
                      "superseded_value"):
            if not entry.get(field):
                issues.append(f"correction for "
                              f"{str(entry.get('superseded_value'))[:40]!r} has "
                              f"no {field}")
        if DIFFERENTIAL_CELL not in str(entry.get("response_dependent", "")):
            issues.append("response_dependent does not name the cell that "
                          f"makes it response-dependent ({DIFFERENTIAL_CELL})")
        if DIFFERENTIAL_TARGET not in str(entry.get("response_dependent", "")):
            issues.append("response_dependent does not name the arm whose "
                          f"reply triggered the refusal ({DIFFERENTIAL_TARGET})")

    filed = doc.get("measurement_behind_the_correction") or {}
    evidence_path = _evidence_path(root)
    if not filed:
        issues.append("the correction does not carry the measurement behind it")
    elif not evidence_path.exists():
        issues.append(f"the evidence artifact {_rel(evidence_path, root)} is "
                      f"absent, so the correction's measurement cannot be "
                      f"re-read")
    else:
        current = measurement(root=root)
        for field in ("full_payload_status", "arms_that_disagree_now",
                      "arms_unmeasured_now", "source_sha256"):
            if filed.get(field) != current[field]:
                issues.append(
                    f"the measurement behind the correction has moved: "
                    f"{field} is {current[field]!r} in "
                    f"{_rel(evidence_path, root)} but the artifact filed "
                    f"{filed.get(field)!r}")

    producers_now = producer_state(root)
    for rel, state in sorted((doc.get("producers") or {}).items()):
        now = producers_now.get(rel, {})
        if not now.get("exists"):
            issues.append(f"producer {rel} no longer exists")
            continue
        if now.get("emits_superseded_key"):
            issues.append(f"producer {rel} emits '{SUPERSEDED_KEY}' again; it "
                          f"must file label_blind and response_dependent from "
                          f"causal_mllm.evaluation.censoring")
        if not now.get("imports_exclusion_metadata"):
            issues.append(f"producer {rel} does not import "
                          f"exclusion_metadata, so it is not writing the "
                          f"corrected wording")
        if state.get("sha256") and now.get("sha256") != state["sha256"]:
            # Not a failure on its own: a producer is code and is allowed to
            # move. Recorded so the artifact does not silently describe a
            # source tree it no longer matches.
            print(f"note  producer {rel} moved since the correction was filed "
                  f"({state['sha256'][:12]} -> {now['sha256'][:12]}) and is "
                  f"still clean")

    print(f"correction    {_rel(path)}")
    print(f"carriers      enumerated {len(enumerated)}   found {len(found)}")
    print(f"occurrences   {sum(e['n_occurrences'] for e in found.values())}")
    print(f"producers     {len(doc.get('producers') or {})} checked")
    if issues:
        return 1, issues
    return 0, []


def found_values(found: dict) -> list[str]:
    return sorted({value for entry in found.values()
                   for value in entry["superseded_values"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="compare the tree against the committed "
                             "correction and write nothing")
    parser.add_argument("--out", type=Path, default=CORRECTION_PATH)
    parser.add_argument("--root", type=Path, default=None,
                        help="scan this directory instead of the repository "
                             "root (tests build a miniature one)")
    args = parser.parse_args(argv)

    if args.verify:
        code, issues = verify(args.out, root=args.root)
        if code == 0:
            print("\nEXCLUSION METADATA: CORRECTION VERIFIED — every carrier "
                  "is enumerated and sealed, and both producers are clean")
            return 0
        if code == 2:
            print(f"\nEXCLUSION METADATA: NO CORRECTION — {issues[0]}")
            return 2
        print(f"\nEXCLUSION METADATA: FAIL ({len(issues)} issue(s))")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    doc = build(root=args.root)
    if not doc["artifacts"]:
        print("FAIL: nothing under the scan roots carries "
              f"'{SUPERSEDED_KEY}', so there is nothing to correct; refusing "
              "to file an empty correction, which would read as a check that "
              "passed over zero artifacts", file=sys.stderr)
        return 2
    for rel, state in sorted(doc["producers"].items()):
        if state["emits_superseded_key"]:
            print(f"FAIL: producer {rel} still emits '{SUPERSEDED_KEY}'; fix "
                  f"the producer before filing a correction beside its output",
                  file=sys.stderr)
            return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {_rel(args.out)}")
    print(f"  carriers       {doc['n_artifacts_carrying_the_superseded_key']}")
    print(f"  occurrences    {doc['n_occurrences']}")
    print(f"  tracked        {doc['n_tracked_carriers']} committed, "
          f"{doc['n_untracked_carriers']} untracked, "
          f"{doc['n_carriers_git_cannot_answer_for']} outside this repository")
    print(f"  superseded     {len(doc['distinct_superseded_values'])} "
          f"distinct sentence(s), each paired with its correction")
    print(f"  differential   {DIFFERENTIAL_CELL} in {DIFFERENTIAL_TARGET} "
          f"alone")
    print(f"  code_commit    {doc['code_commit']}")
    print(f"  git_dirty      {doc['git_dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
