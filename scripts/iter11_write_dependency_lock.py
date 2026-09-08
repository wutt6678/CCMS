#!/usr/bin/env python3
"""Commit the environment the Iteration 11 evidence was certified against.

``dependency_lock_snapshot()`` records ``pip_freeze_sha256`` and ``n_packages``.
That is enough to find out whether a machine already possesses the certified
environment and not enough to give it one: a hash with no committed preimage can
be compared against but never rebuilt. So another machine could test the
environment and could not recreate it, and the only exact-equality verifiers in
the repository were unusable anywhere but here.

This files the preimage. Two artifacts, because the freeze text has to be
byte-exact and therefore cannot carry a word of its own explanation:

* ``dependency_freeze.lock.txt`` -- exactly the 100 lines whose SHA-256 is the
  ``pip_freeze_sha256`` every preflight artifact and every resolved run
  fingerprint already binds. Sorted, comment-free, no trailing newline, and with
  this project's own editable distribution removed BY NAME only, because
  ``pip freeze`` renders that install in a form that embeds this repository's
  live HEAD and would make the file -- and the hash -- move on every commit.
* ``dependency_lock_reconstruction.json`` -- what a reader needs and the text
  cannot say: which script produced it, from which commit, whether the bytes
  still hash to the recorded value, the Python version, and the versions of the
  packages whose floating-point behaviour actually moves a result (numpy, scipy,
  torch, pandas).

REFUSES TO FILE AN ENVIRONMENT THAT IS NOT THE CERTIFIED ONE. If the active
``pip freeze`` does not hash to the recorded ``pip_freeze_sha256`` this script
writes nothing: filing a different package list beside a hash that does not
describe it would put two contradictory statements about the environment in the
same directory. Re-capture the lock with the model preflight's ``--update-lock``
first, deliberately, and then file it.

Usage:
    python3 scripts/iter11_write_dependency_lock.py             # write both
    python3 scripts/iter11_write_dependency_lock.py --verify    # compare only

Exit codes: 0 the committed freeze hashes to the recorded lock identity; 1 it
does not, or a package the lock should name is missing; 2 nothing is committed
yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.replay.registry import (  # noqa: E402
    DEFAULT_LOCK,
    NUMERICALLY_CONSEQUENTIAL_PACKAGES,
    dependency_lock_sha256,
    dependency_lock_snapshot,
    freeze_text,
    load_dependency_lock,
    normalized_freeze_lines,
    pinned_versions,
    verify_committed_freeze,
)
from causal_mllm.seeds import code_tree_status, get_git_commit  # noqa: E402

PREFLIGHT_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "preflight"
FREEZE_PATH = PREFLIGHT_ROOT / "dependency_freeze.lock.txt"
REPORT_PATH = PREFLIGHT_ROOT / "dependency_lock_reconstruction.json"
MODEL_KEYS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")

#: This stage's own outputs, excluded from the tree dirtiness it RECORDS so
#: that filing the freeze cannot invalidate the report that files it. Narrow,
#: and reported rather than silently dropped.
OWN_OUTPUT_PREFIXES = (
    "outputs/iteration_11/preflight/dependency_freeze.lock.txt",
    "outputs/iteration_11/preflight/dependency_lock_reconstruction.json",
)

RECREATE_WITH = (
    "conda create -n ccms-iter11 python={python} -y && "
    "conda activate ccms-iter11 && "
    "pip install -r {freeze} && pip install -e .")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def observed_versions() -> dict[str, str]:
    """What the four filed preflight artifacts observed, per package.

    Read out of committed evidence rather than imported: importing torch to ask
    it its own version takes seconds and answers about THIS process, while the
    preflights answer about the process that produced the evidence.
    """
    seen: dict[str, str] = {}
    for model_key in MODEL_KEYS:
        path = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not path.exists():
            continue
        environment = (json.loads(path.read_text(encoding="utf-8"))
                       .get("environment") or {})
        for name, version in (environment.get("observed_versions")
                              or {}).items():
            seen.setdefault(str(name).lower(), str(version))
    return seen


def local_version_gaps(numeric: dict, observed: dict | None = None) -> dict:
    """Packages whose freeze line cannot reinstall what actually ran.

    ``pip freeze`` reports a distribution's version WITHOUT its local segment,
    so an environment holding torch 2.8.0+cu128 freezes as ``torch==2.8.0`` and
    ``pip install -r`` of that line installs the PyPI build -- a different CUDA
    toolchain, for the one package in this lock whose build decides whether a
    checkpoint loads at all. Recorded rather than papered over: the freeze is
    exactly what ``pip freeze`` says, and an honest statement about
    reconstructibility has to carry the exception.
    """
    observed = observed_versions() if observed is None else observed
    gaps = {}
    for name, frozen in sorted(numeric.items()):
        seen = observed.get(name)
        if frozen and seen and seen != frozen \
                and seen.startswith(f"{frozen}+"):
            gaps[name] = {
                "freeze": frozen,
                "observed_by_the_preflight": seen,
                "what_pip_install_r_would_fetch":
                    f"the default-index build of {name}=={frozen}, not the "
                    f"{seen} the preflight observed"}
    return gaps


#: Both defaults below resolve at CALL time. A default argument is evaluated
#: once, when the module is imported, so binding ``DEFAULT_LOCK`` into the
#: signature would keep pointing at the world lock no matter what a caller did
#: to the module afterwards.
def build(lock_path: str | Path | None = None) -> dict:
    """Both artifacts, or a refusal naming the disagreement.

    Returns ``{"freeze_text", "report"}`` on success and ``{"issues": [...]}``
    when the active environment is not the certified one.
    """
    lock_path = Path(DEFAULT_LOCK) if lock_path is None else Path(lock_path)
    locked = load_dependency_lock(lock_path)
    if locked is None:
        return {"issues": [
            f"{_rel(Path(lock_path))} records no dependency_lock block, so "
            f"there is no certified environment to file; run the model "
            f"preflight with --update-lock first"]}
    lines, excluded = normalized_freeze_lines()
    text = freeze_text(lines)
    snapshot = dependency_lock_snapshot()
    issues = []
    if snapshot["pip_freeze_sha256"] != locked.get("pip_freeze_sha256"):
        issues.append(
            f"the active environment freezes to "
            f"{snapshot['pip_freeze_sha256']} but the lock records "
            f"{locked.get('pip_freeze_sha256')}: this is not the certified "
            f"environment, so filing its package list would put a second, "
            f"contradictory statement about the environment beside the hash "
            f"the artifacts are bound to. Re-capture the lock with the model "
            f"preflight's --update-lock, deliberately, and then file it")
    if snapshot["python_version"] != locked.get("python_version"):
        issues.append(
            f"the active interpreter is {snapshot['python_version']} but the "
            f"lock records {locked.get('python_version')}, and the interpreter "
            f"is part of what decides the last bit of a floating-point sum")
    if issues:
        return {"issues": issues}

    numeric = pinned_versions(lines)
    gaps = local_version_gaps(numeric)
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    report = {
        "question": "which environment was this evidence certified against, "
                    "and could another machine build it",
        "produced_by": "scripts/iter11_write_dependency_lock.py",
        "kind": "iteration_11_dependency_lock_reconstruction_v1",
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "lock_path": _rel(Path(lock_path)),
        "dependency_lock_sha256": dependency_lock_sha256(lock_path),
        "freeze_path": _rel(FREEZE_PATH),
        "freeze_sha256": snapshot["pip_freeze_sha256"],
        "recorded_pip_freeze_sha256": locked.get("pip_freeze_sha256"),
        "freeze_is_the_preimage_of_the_recorded_hash":
            snapshot["pip_freeze_sha256"] == locked.get("pip_freeze_sha256"),
        "n_packages": len(lines),
        "recorded_n_packages": locked.get("n_packages"),
        "python_version": snapshot["python_version"],
        "recorded_python_version": locked.get("python_version"),
        "pyproject_sha256": snapshot["pyproject_sha256"],
        "excluded_self_distributions": excluded,
        "numeric_packages": numeric,
        "numeric_packages_named": list(
            NUMERICALLY_CONSEQUENTIAL_PACKAGES),
        "recreate_with": RECREATE_WITH.format(
            python=snapshot["python_version"], freeze=_rel(FREEZE_PATH)),
        "packages_whose_freeze_line_omits_a_local_version_segment": gaps,
        "reconstructible_from_the_freeze_alone": not gaps,
        "recreate_caveat": (
            "pip install -r this freeze installs the DEFAULT-INDEX build of "
            + ", ".join(
                f"{name} ({info['freeze']}, where the preflight observed "
                f"{info['observed_by_the_preflight']})"
                for name, info in sorted(gaps.items()))
            + ". pip freeze drops a version's local segment, so the CUDA build "
              "the evidence was actually produced under is not expressible in "
              "this file and has to be installed from the index it came from "
              "before the rest of the freeze is applied")
        if gaps else None,
        "why": (
            "a lock that stores only a hash can be compared against but never "
            "rebuilt, so another machine could test whether it somehow "
            "possessed the certified environment and could not create it. That "
            "mattered: the exact-equality verifiers are unusable anywhere else, "
            "and the floating-point differences they report are a property of "
            "the interpreter that summed them. Committing the preimage of the "
            "already-bound hash makes the same environment reconstructible "
            "without moving any hash"),
    }
    return {"freeze_text": text, "report": report}


def verify(lock_path: str | Path | None = None) -> int:
    """Compare both committed artifacts against the lock, writing nothing."""
    lock_path = Path(DEFAULT_LOCK) if lock_path is None else Path(lock_path)
    result = verify_committed_freeze(FREEZE_PATH, lock_path)
    print(f"freeze        {_rel(FREEZE_PATH)}")
    if not result["exists"]:
        for issue in result["issues"]:
            print(f"  - {issue}")
        return 2
    print(f"  bytes hash  {result['sha256']}")
    print(f"  recorded    {result['recorded_pip_freeze_sha256']}")
    print(f"  packages    {result['n_lines']} "
          f"(lock records {result['recorded_n_packages']})")
    print(f"  python      {result['recorded_python_version']}")
    for name in NUMERICALLY_CONSEQUENTIAL_PACKAGES:
        print(f"  {name:11s} {result['numeric_packages'].get(name)}")
    if not REPORT_PATH.exists():
        print(f"  - no reconstruction report at {_rel(REPORT_PATH)}, so "
              f"nothing in the repository says which script filed the freeze "
              f"or how to recreate the environment from it")
        return 1
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("freeze_sha256") != result["sha256"]:
        result["issues"].append(
            f"{_rel(REPORT_PATH)} names freeze_sha256 "
            f"{report.get('freeze_sha256')} but the committed freeze hashes "
            f"to {result['sha256']}")
    if report.get("produced_by") != "scripts/iter11_write_dependency_lock.py":
        result["issues"].append(
            f"{_rel(REPORT_PATH)} does not name the script that produced it")
    for issue in result["issues"]:
        print(f"  - {issue}")
    if result["issues"]:
        return 1
    print("\nDEPENDENCY LOCK: the committed freeze is the preimage of the "
          "recorded hash, so the certified environment is reconstructible")
    print(f"  {report.get('recreate_with')}")
    if report.get("recreate_caveat"):
        print(f"  CAVEAT {report['recreate_caveat']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="compare the committed freeze and report against "
                             "the lock, and write nothing")
    parser.add_argument("--lock", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.lock)

    built = build(args.lock)
    if built.get("issues"):
        print("FAIL: refusing to file this environment", file=sys.stderr)
        for issue in built["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # No trailing newline: the file has to BE the hashed text, and a final
    # newline would make its own bytes hash to something else.
    FREEZE_PATH.write_text(built["freeze_text"], encoding="utf-8")
    REPORT_PATH.write_text(
        json.dumps(built["report"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {_rel(FREEZE_PATH)}")
    print(f"wrote {_rel(REPORT_PATH)}")
    report = built["report"]
    print(f"  packages      {report['n_packages']}")
    print(f"  freeze sha256 {report['freeze_sha256']}")
    print(f"  python        {report['python_version']}")
    for name in NUMERICALLY_CONSEQUENTIAL_PACKAGES:
        print(f"  {name:13s} {report['numeric_packages'][name]}")
    if report["recreate_caveat"]:
        print(f"  CAVEAT        {report['recreate_caveat']}")
    print(f"  code_commit   {report['code_commit']}")
    print(f"  git_dirty     {report['git_dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
