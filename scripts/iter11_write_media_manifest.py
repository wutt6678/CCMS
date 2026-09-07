#!/usr/bin/env python3
"""Iteration 11 media manifest — bind the images the evidence was made from.

``data/media`` is not in the repository: ``.gitignore`` excludes the tree and
negates only 20 individually committed source images, so a fresh checkout holds
20 of the 3,034 files the panel and the human-audit packs were built from. That
has a consequence which was found the expensive way. ``iter11_replay_checks.py``
re-hashes every image the 100-family confirmatory panel references and reports a
missing one as a FAILURE, so running the documented verification command on a
fresh checkout rewrote all four committed PASS reports as FAIL — a verdict about
the panel that was really a statement about the machine it ran on.

This manifest is the fix's other half. It binds every media file by path,
SHA-256 and size, so:

  * the images the evidence was generated from have a committed identity even
    though their bytes are not committed;
  * a consumer can tell "absent here" from "present and different". The first is
    a limitation of the checkout and is reported as one; the second is a finding
    about the evidence and fails;
  * a file present on disk but NOT in the manifest is also a finding, because it
    means the tree holds media this project never bound.

The media are not deterministically regenerable — they are source photographs
and rendered composites carried in from the dataset releases — so binding is the
available option and materialization is not.

NOT A MEASUREMENT WHEN NOTHING WAS HASHED. A roll-up hash over an empty file map
is an ordinary-looking sha256, and every empty map hashes to the same one, so
"the manifest matches" would be true on a checkout holding none of the media.
:func:`rollup_sha256` therefore returns None on an empty map, ``--verify``
exits 3 rather than 0 when any bound file is absent, and the roll-up comparison
is only performed when every file was actually hashed.

Usage:
    python3 scripts/iter11_write_media_manifest.py            # write it
    python3 scripts/iter11_write_media_manifest.py --verify   # compare, no write
    python3 scripts/iter11_write_media_manifest.py --verify --panel-only

Exit codes: 0 every bound file was hashed and matched; 1 a file differs, or the
tree holds media the manifest does not bind; 2 no manifest to verify against;
3 some bound files are absent, so this checkout cannot verify the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.seeds import code_tree_status, get_git_commit  # noqa: E402

MEDIA_ROOT = REPO_ROOT / "data" / "media"
MANIFEST_PATH = REPO_ROOT / "outputs" / "iteration_11" / "media_manifest.json"
PANEL_PATH = REPO_ROOT / "outputs" / "scale_c" / "families_panel" \
    / "validated_families.jsonl"

#: This stage's own output, excluded from the tree dirtiness it RECORDS so that
#: writing the manifest cannot invalidate it. Narrow, and reported.
OWN_OUTPUT_PREFIXES = ("outputs/iteration_11/media_manifest.json",)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollup_sha256(files: dict) -> str | None:
    """One digest over the whole bound set — or None when nothing was hashed.

    Returning a digest for an empty map is the failure this guards against: two
    checkouts holding none of the media would produce equal roll-ups and the
    comparison would read as an agreement between two measurements that were
    never made.
    """
    if not files:
        return None
    blob = "".join(f"{path}\t{files[path]['sha256']}\t{files[path]['bytes']}\n"
                   for path in sorted(files))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


#: The three path defaults below are resolved at CALL time rather than bound
#: into the signature. A default argument is evaluated once, when the module is
#: imported, so ``def scan_media(root: Path = MEDIA_ROOT)`` would keep pointing
#: at the real 3.7 GB media tree no matter what a caller -- or a test building a
#: miniature repository -- did to the module afterwards.
def scan_media(root: Path | None = None,
               only: set[str] | None = None) -> dict:
    """Every regular file under the media root, by repo-relative path."""
    root = MEDIA_ROOT if root is None else root
    files: dict[str, dict] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = _rel(path)
        if only is not None and rel not in only:
            continue
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def panel_images(panel_path: Path | None = None) -> list[str]:
    """The distinct images the frozen 100-family panel references.

    Read out of the panel rather than re-derived: these are the paths the
    generations were built from, and the manifest's job is to bind what the
    evidence actually used.
    """
    panel_path = PANEL_PATH if panel_path is None else panel_path
    if not panel_path.exists():
        return []
    paths: set[str] = set()
    with panel_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            family = json.loads(line)
            for variant in (family.get("variants") or {}).values():
                for message in variant.get("messages") or []:
                    paths.update(message.get("images") or [])
    return sorted(paths)


def provenance() -> dict:
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    return {
        "produced_by": "scripts/iter11_write_media_manifest.py",
        "kind": "iteration_11_media_manifest_v1",
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
    }


def build(panel_only: bool = False) -> dict:
    referenced = panel_images()
    only = set(referenced) if panel_only else None
    files = scan_media(only=only)
    manifest = {
        "question": "which images does this project's evidence rest on, and "
                    "what are their bytes",
        **provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "media_root": _rel(MEDIA_ROOT),
        "scope": "panel_referenced" if panel_only else "whole_media_tree",
        "n_files": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "files": files,
        "rollup_sha256": rollup_sha256(files),
        "n_panel_referenced_images": len(referenced),
        "panel_referenced_but_absent_from_the_manifest": sorted(
            set(referenced) - set(files)),
        "why": (
            "data/media is gitignored apart from 20 individually negated source "
            "images, so a fresh checkout cannot re-run any check that hashes "
            "the media. Binding them by path, SHA-256 and size lets a consumer "
            "distinguish 'not present here' from 'present and different', "
            "which is the difference between a limitation of the checkout and a "
            "finding about the evidence"),
    }
    return manifest


def verify(manifest_path: Path | None = None,
           panel_only: bool = False) -> tuple[int, list[str]]:
    """Compare the committed manifest against the tree, without writing.

    Returns ``(exit_code, issues)``. Absence is not a mismatch and is not
    silently ignored either: it makes the verification incomplete, which is a
    different answer from both PASS and FAIL.
    """
    manifest_path = MANIFEST_PATH if manifest_path is None else manifest_path
    if not manifest_path.exists():
        return 2, [f"no media manifest at {_rel(manifest_path)}; run this "
                   f"script without --verify to write one"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bound = manifest.get("files") or {}
    if not bound:
        return 1, [f"{_rel(manifest_path)} binds no files at all, so it is not "
                   f"a measurement of anything"]

    scope = set(bound)
    if panel_only:
        scope &= set(panel_images())
        if not scope:
            return 1, ["--panel-only intersected with the manifest is empty, "
                       "so there is nothing to verify"]
    present = {path for path in scope if (REPO_ROOT / path).exists()}
    absent = sorted(scope - present)

    issues: list[str] = []
    hashed = {}
    for path in sorted(present):
        actual = (REPO_ROOT / path)
        digest = sha256_file(actual)
        size = actual.stat().st_size
        hashed[path] = {"sha256": digest, "bytes": size}
        expected = bound[path]
        if digest != expected["sha256"]:
            issues.append(
                f"{path}: sha256 {digest} != the bound {expected['sha256']}")
        if size != expected["bytes"]:
            issues.append(
                f"{path}: {size} bytes != the bound {expected['bytes']}")

    # Media the tree holds but the manifest never bound. Only meaningful when
    # verifying the whole tree: under --panel-only everything else is out of
    # scope by construction.
    if not panel_only:
        unbound = sorted(set(_rel(p) for p in MEDIA_ROOT.rglob("*")
                             if p.is_file()) - set(bound))
        for path in unbound[:20]:
            issues.append(f"{path} is present in {_rel(MEDIA_ROOT)} but the "
                          f"manifest does not bind it")
        if len(unbound) > 20:
            issues.append(f"... and {len(unbound) - 20} more unbound file(s)")

    referenced_missing = manifest.get(
        "panel_referenced_but_absent_from_the_manifest") or []
    if referenced_missing:
        issues.append(
            f"the frozen panel references {len(referenced_missing)} image(s) "
            f"the manifest does not bind: {referenced_missing[:5]}")

    print(f"manifest      {_rel(manifest_path)}")
    print(f"scope         {manifest.get('scope')} "
          f"({len(scope)} file(s) in scope)")
    print(f"hashed here   {len(hashed)}   absent {len(absent)}")
    print(f"roll-up       bound {manifest.get('rollup_sha256')}")
    if absent:
        print(f"              NOT COMPARABLE: {len(absent)} bound file(s) are "
              f"absent from this checkout, so the roll-up cannot be "
              f"recomputed. Hashing the {len(hashed)} that are present gives "
              f"{rollup_sha256(hashed)} over a SUBSET, which is not the bound "
              f"value and is not a disagreement.")
    elif len(scope) == len(bound):
        recomputed = rollup_sha256(hashed)
        print(f"              recomputed {recomputed}")
        if recomputed != manifest.get("rollup_sha256"):
            issues.append(
                f"roll-up {recomputed} != the bound "
                f"{manifest.get('rollup_sha256')} even though every file "
                f"matched individually")

    if issues:
        return 1, issues
    if absent:
        return 3, [f"{len(absent)} of {len(scope)} bound file(s) are absent "
                   f"from this checkout, so the manifest is unverified here "
                   f"rather than confirmed"]
    return 0, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="compare the tree against the committed manifest "
                             "and write nothing")
    parser.add_argument("--panel-only", action="store_true",
                        help="restrict to the images the frozen 100-family "
                             "panel references (100 files, ~139 MB) instead "
                             "of the whole media tree")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    if args.verify:
        code, issues = verify(args.out, panel_only=args.panel_only)
        if code == 0:
            print("\nMEDIA MANIFEST: VERIFIED — every bound file was hashed "
                  "and matched")
            return 0
        if code == 3:
            print(f"\nMEDIA MANIFEST: NOT VERIFIABLE HERE — {issues[0]}")
            return 3
        print(f"\nMEDIA MANIFEST: FAIL ({len(issues)} issue(s))")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    if not MEDIA_ROOT.exists():
        print(f"FAIL: {_rel(MEDIA_ROOT)} does not exist, so there is nothing "
              f"to bind; refusing to write a manifest over an empty tree, "
              f"which would hash like a measurement of nothing",
              file=sys.stderr)
        return 2
    manifest = build(panel_only=args.panel_only)
    if not manifest["files"]:
        print(f"FAIL: {_rel(MEDIA_ROOT)} holds no files; refusing to write an "
              f"empty manifest", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {_rel(args.out)}")
    print(f"  scope          {manifest['scope']}")
    print(f"  files          {manifest['n_files']}")
    print(f"  bytes          {manifest['total_bytes']:,}")
    print(f"  roll-up        {manifest['rollup_sha256']}")
    print(f"  panel images   {manifest['n_panel_referenced_images']} "
          f"referenced, "
          f"{len(manifest['panel_referenced_but_absent_from_the_manifest'])} "
          f"unbound")
    print(f"  code_commit    {manifest['code_commit']}")
    print(f"  git_dirty      {manifest['git_dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
