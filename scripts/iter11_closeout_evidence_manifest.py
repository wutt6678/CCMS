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
modules those files import -- and the discovery is repeated at verification time
wherever a git object store exists, so a file committed and left out of the
manifest is a finding rather than a gap nobody notices.

The library modules are not a list anybody maintains. They are the IMPORT
CLOSURE of the bound Python, walked with ``ast`` from every bound script and test
and closed under ``causal_mllm`` imports, so a module the verifiers reach is bound
whether or not anybody remembered it. That matters because the closure is what
makes the numbers in the artifacts mean what they say: the tolerance rules, the
frozen bootstrap, the estimand definitions, the hypothesis tests, the adjudicator
routing and the truncation classifier are all library code, and a manifest that
bound the artifacts and eight modules while the analysis imported thirty-three
would be binding a claim and not the thing that computes it. Third-party packages
are deliberately outside the closure -- they are bound by the dependency lock,
which is itself bound here.

Two commands, because hashing a verifier is not running it:

``--verify`` re-derives the document and resolves every bound file at HEAD. It
reads the entry points' source to confirm each names a ``--verify`` mode, and it
never executes one, so it can pass while a scientific verifier fails.

``--deep`` runs every entry-point verifier as a subprocess and aggregates what
comes back: exit 0 is verified, exit 3 is incomplete-but-not-contradicted and is
reported as such rather than silently folded into success, and exit 1 or 2 fails
the deep closeout. That is the command which answers "does the closeout still
hold", and ``--verify`` alone does not.

Exit codes, and each one is tested:
    0  filed, or verified with every bound file present, hashing correctly, and
       resolving out of the object store; under ``--deep``, every verifier
       returned 0 or 3
    1  a contradiction: a bound file does not hash to what is filed, the filed
       set and the committed set disagree, the document does not re-derive, the
       entry points are not the ones this module files, the manifest on disk is
       not the blob HEAD holds, the tree was dirty when it was generated, or a
       deep verifier returned 1 or 2
    2  nothing filed to verify against, or a bound file is absent from disk
    3  every hash checks out but committed-ness is not checkable here -- no git
       object store, so this is an export or a tarball rather than a checkout.
       The evidence is sound and one claim about it cannot be made from here.
"""

from __future__ import annotations

import argparse
import ast
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

#: The library modules the bound verifiers import, listed by hand. The import
#: closure computed below is a superset of the ``src/`` entries here and is what
#: actually binds them; this list is kept as a floor, so that if the closure
#: walker is ever broken the modules the analysis is known to depend on are still
#: bound, and ``check_the_manifest`` requires every ``src/`` path named here to
#: appear in the computed closure -- a walker that loses one is a finding rather
#: than a silently smaller manifest. Binding all of ``src/`` would make the
#: manifest move on every unrelated change, and binding none of it would leave the
#: tolerance rules, the environment certification and the frozen bootstrap outside
#: the evidence.
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


PACKAGE = "causal_mllm"
SRC_ROOT = "src"


def _module_file(module: str) -> str | None:
    """Resolve a dotted ``causal_mllm`` module name to a path under ``src/``."""
    if module != PACKAGE and not module.startswith(f"{PACKAGE}."):
        return None
    rel = module.replace(".", "/")
    for candidate in (f"{SRC_ROOT}/{rel}.py", f"{SRC_ROOT}/{rel}/__init__.py"):
        if (REPO_ROOT / candidate).is_file():
            return candidate
    return None


def _import_sites(path: Path) -> tuple[list[tuple[str, bool, str, bool]],
                                       list[str]]:
    """Every import in a file, as absolute sites and as relative-site notes.

    An absolute site is ``(dotted name, deferred, where, symbol_position)``.
    ``deferred`` is True when the import sits inside a function or lambda body
    rather than at import time. The distinction is load-bearing: a module-level
    import of a module that does not exist means the package cannot be imported
    at all, while a deferred one is a branch that may legitimately not have
    landed yet. Treating the two the same way makes an optional adapter into a
    fatal finding, and a check that cries wolf on intentional code gets switched
    off -- which is how a manifest ends up binding less than it claims to.

    ``symbol_position`` is True only for the names after ``from X import``, which
    are the ones that may be attributes rather than modules. Names in a plain
    ``import X.Y`` are always modules, so an unresolvable one is a gap and not a
    symbol; without this flag ``import causal_mllm.renamed`` is discarded because
    its parent ``causal_mllm`` resolves, and a walker that discards it walks a
    smaller graph than the interpreter would import.

    Relative imports cannot be resolved from a file path alone -- ``from .x
    import y`` depends on where the file sits in the package -- so they are
    returned separately and COUNTED rather than silently skipped. This
    repository has none; if it ever adopts them the closure says how many edges
    it could not follow instead of binding less and reporting the same success.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: list[tuple[str, bool, str, bool]] = []
    relative: list[str] = []

    def names_of(node) -> list[tuple[str, bool]]:
        if isinstance(node, ast.Import):
            return [(alias.name, False) for alias in node.names]
        if not node.module:
            return []
        return [(node.module, False)] \
            + [(f"{node.module}.{alias.name}", True) for alias in node.names]

    def walk(node, deferred: bool, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom) and child.level:
                relative.append(
                    f"{'.' * child.level}{child.module or ''} in {where}")
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                sites.extend((name, deferred, where, symbol)
                             for name, symbol in names_of(child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Lambda)):
                walk(child, True, getattr(child, "name", "<lambda>"))
            else:
                walk(child, deferred, where)

    walk(tree, False, "<module>")
    return sites, relative


def _imported_modules(path: Path) -> tuple[set[str], dict, dict, list[str]]:
    """What a file imports, read with ``ast`` rather than matched with a regex.

    Returns the names that resolve to a module under ``src/``, the ones that do
    not and are imported at module level, the ones that do not and are imported
    inside a function body, and the relative import sites that could not be
    resolved at all. An unresolvable name in a SYMBOL position is normally just a
    symbol -- ``from causal_mllm.data.io import read_jsonl`` yields the module and
    the symbol both -- so it is only a gap when its parent does not resolve
    either, which is what separates ``read_jsonl`` from a module that has been
    renamed, moved or never written.
    """
    resolved: set[str] = set()
    at_import_time: dict[str, list[str]] = {}
    deferred: dict[str, list[str]] = {}
    sites, relative = _import_sites(path)
    for name, is_deferred, where, symbol_position in sites:
        target = _module_file(name)
        if target is not None:
            resolved.add(target)
            continue
        if not name.startswith(PACKAGE):
            continue
        parent = name.rsplit(".", 1)[0]
        if symbol_position and (parent == name or _module_file(parent) is not None):
            continue
        bucket = deferred if is_deferred else at_import_time
        names = bucket.setdefault(where, [])
        if name not in names:
            names.append(name)
    for names in (at_import_time, deferred):
        for where in names:
            names[where] = sorted(names[where])
    return resolved, at_import_time, deferred, sorted(relative)


def dependency_closure(roots: list[str]) -> dict:
    """The ``causal_mllm`` modules the bound Python reaches, with the edges.

    Walked rather than listed, so the manifest cannot fall behind the code: a
    module a verifier starts importing is bound the next time this runs, and one
    it stops importing leaves the bound set on its own. The edges are filed
    because "why is this module part of the evidence" is a question a reviewer
    should be able to answer from the document instead of by reading imports.

    Deferred imports are walked as well as module-level ones, and that is not a
    formality: the model adapters for two of the four arms are imported inside
    ``build_adapter`` and nowhere else, so a module-level-only walk would leave
    the code that produced those generations unbound.

    Third-party imports are out of scope by construction -- ``_module_file``
    resolves only ``causal_mllm`` names -- and are bound instead by the dependency
    lock, which is itself in the bound set. Relative imports are out of scope by
    necessity rather than by choice, and are counted where they occur so that the
    omission is visible in the document instead of silent in the walker.
    """
    edges: dict[str, list[str]] = {}
    absent_at_import_time: dict[str, dict] = {}
    absent_deferred: dict[str, dict] = {}
    relative_sites: dict[str, list[str]] = {}
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in edges:
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            fatal(f"{rel} is bound by this manifest but is not a file on disk "
                  f"here, so its imports cannot be walked", 2)
        try:
            resolved, at_import_time, deferred, relative = _imported_modules(path)
        except SyntaxError as exc:
            fatal(f"{rel} does not parse ({exc}); a bound file whose imports "
                  f"cannot be walked cannot have its dependency closure computed, "
                  f"and a closure that quietly stops at a syntax error is a "
                  f"smaller manifest pretending to be the same one", 1)
            raise  # unreachable; keeps the return type honest
        edges[rel] = sorted(resolved)
        if at_import_time:
            absent_at_import_time[rel] = at_import_time
        if deferred:
            absent_deferred[rel] = deferred
        if relative:
            relative_sites[rel] = relative
        stack.extend(resolved)
    return {
        "modules": sorted({dep for deps in edges.values() for dep in deps}),
        "edges": dict(sorted(edges.items())),
        "imports_that_resolve_to_no_module": dict(sorted(
            absent_at_import_time.items())),
        "deferred_imports_that_resolve_to_no_module": dict(sorted(
            absent_deferred.items())),
        "relative_imports_that_could_not_be_resolved": dict(sorted(
            relative_sites.items())),
    }


def closure_block(paths: list[str]) -> dict:
    """The filed statement of what the bound code depends on.

    Total: it always returns a document. The two things that can be wrong with
    a closure are filed as fields rather than raised, because ``build`` refusing
    to produce a document turns a diagnosable contradiction into "could not
    re-derive", and a manifest that cannot be built cannot be compared with the
    one that was filed either. :func:`check_the_manifest` requires both fields to
    be empty, so nothing is weakened by filing instead of raising -- the failure
    just arrives where it can be reported alongside the rest.
    """
    roots = sorted(path for path in paths if path.endswith(".py"))
    closure = dependency_closure(roots)
    hand_listed = sorted(module for module in BOUND_PATHS
                         if module.startswith(f"{SRC_ROOT}/"))
    return {
        "n_modules": len(closure["modules"]),
        "n_roots_walked": len(roots),
        "modules": closure["modules"],
        "edges": closure["edges"],
        "roots": roots,
        "modules_that_cannot_be_imported": dict(sorted(
            closure["imports_that_resolve_to_no_module"].items())),
        "what_an_unimportable_module_means": (
            f"a bound file imports a {PACKAGE} module at MODULE LEVEL that does "
            f"not exist, so the package cannot be imported and nothing "
            f"downstream of it can have produced the evidence this manifest "
            f"binds. Empty is the only acceptable value"),
        "hand_listed_modules_the_closure_does_not_reach": sorted(
            module for module in hand_listed
            if module not in closure["modules"]),
        "what_that_field_checks": (
            "the walker against a hand-written expectation. This file names "
            f"{len(hand_listed)} library modules as dependencies of the bound "
            "verifiers; if the walk does not reach one of them then either the "
            "walker is broken or that module is no longer imported by anything "
            "bound. Both are findings, and neither is a reason to bind less, so "
            "the field is filed and required to be empty rather than being "
            "resolved by dropping the module from the list"),
        "deferred_imports_of_modules_that_do_not_exist":
            closure["deferred_imports_that_resolve_to_no_module"],
        "relative_imports_the_walk_could_not_follow":
            closure["relative_imports_that_could_not_be_resolved"],
        "n_relative_imports_the_walk_could_not_follow": sum(
            len(sites) for sites in
            closure["relative_imports_that_could_not_be_resolved"].values()),
        "what_an_unfollowed_relative_import_would_mean": (
            "an edge this closure did not follow, so the module on the other "
            "side of it is not bound. Zero is the value this repository "
            "produces, and it is filed rather than assumed so that adopting "
            "relative imports turns into a number in the manifest instead of a "
            "quietly smaller one"),
        "what_a_deferred_import_of_an_absent_module_is": (
            "a branch that would raise ModuleNotFoundError if it were ever taken. "
            "It is filed rather than fatal because it does not stop the package "
            "importing and it may name a module this iteration never needed; it "
            "is not filed as harmless either, because a registry that advertises "
            "a kind whose module is absent will fail with an import error instead "
            "of the error it meant to raise. Anything imported AT MODULE LEVEL "
            "and absent is fatal, and is not listed here"),
        "how_it_is_computed": (
            f"every bound .py file is parsed with ast, its {PACKAGE} imports are "
            f"resolved to files under {SRC_ROOT}/, and the walk repeats until "
            f"nothing new is reached. Imports inside function and lambda bodies "
            f"are walked too and marked deferred. A name that resolves to no "
            f"module is a symbol and is ignored; a name whose PARENT resolves to "
            f"no module is a gap, recorded per import site"),
        "what_it_does_not_cover": (
            "third-party packages, which no walk of this repository can pin. They "
            "are bound by outputs/iteration_11/preflight/dependency_lock_"
            "reconstruction.json and the freeze it authenticates, both of which "
            "are in the bound set, and the environment a verifier runs in is "
            "certified against that lock at verification time rather than assumed "
            "here"),
        "why_it_is_walked_and_not_listed": (
            "a hand-maintained list of the modules that matter is a claim about "
            "the code that goes stale silently, and the failure it produces is a "
            "manifest that still verifies while an artifact's numbers are computed "
            "by a module nobody bound. Walking means the bound set follows the "
            "imports"),
    }


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
    closure = dependency_closure(
        sorted(path for path in paths if path.endswith(".py")))
    if closure["imports_that_resolve_to_no_module"]:
        offenders = closure["imports_that_resolve_to_no_module"]
        fatal(f"{len(offenders)} bound file(s) import a {PACKAGE} module that "
              f"does not exist at import time: "
              + "; ".join(f"{key} -> {json.dumps(value, sort_keys=True)}"
                          for key, value in sorted(offenders.items())[:6]), 1)
    for module in closure["modules"]:
        tracked = _git("ls-files", "--error-unmatch", "--", module)
        if tracked.returncode != 0:
            fatal(f"{module} is imported by the bound code but is not tracked, so "
                  f"the evidence depends on a file this repository does not hold", 1)
        paths.add(module)
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

    ``entry_points`` is a parameter so that a test can build a hermetic manifest
    over a three-file set without inheriting sixteen entry-point complaints about
    files that are not in it. It is NOT a licence for verification to pass back
    the list that is filed: ``verify`` passes ``ENTRY_POINTS``, the constant this
    module files, so a manifest whose pointers have been thinned out fails to
    re-derive instead of re-deriving itself. A value compared against a
    re-derivation of the same value enforces nothing, and an echoed-back entry
    list is exactly that.
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
    verifiers = sorted({point["verified_by"] for point in entry_points})
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
        # Copied per point, not just as a list. ``list(entry_points)`` shares the
        # dicts with the module constant, so a caller that edits the built
        # document edits ENTRY_POINTS with it -- and in ``verify`` that makes the
        # expected side of the entry-point comparison move with the side being
        # checked, which is the same self-agreement a pin exists to prevent.
        "where_a_reviewer_starts": [dict(point) for point in entry_points],
        "scientific_dependency_closure": closure_block(paths),
        "deep_closeout": {
            "command": ("python scripts/iter11_closeout_evidence_manifest.py "
                        "--deep"),
            "verifiers_it_executes": verifiers,
            "n_verifiers": len(verifiers),
            "how_exit_codes_aggregate": {
                "0": "that verifier's evidence re-derives; counted as verified",
                "3": "incomplete here rather than contradicted -- a section of "
                     "the evidence this checkout does not have, such as the "
                     "media in a fresh clone. Counted, named, and never folded "
                     "into the success total",
                "1": "a contradiction. Fails the deep closeout",
                "2": "nothing filed for that verifier to check, or a bound file "
                     "absent from disk. Fails the deep closeout",
            },
            "what_verify_alone_does_not_do": (
                "--verify hashes the verifiers and reads their source to confirm "
                "each implements a --verify mode. It does not run one, so a "
                "manifest can verify exactly while a scientific verifier behind "
                "it fails. --deep is the command that executes them, and a "
                "closeout that has not been run deep has been hashed, not "
                "checked"),
        },
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
                "why": "a document cannot carry its own hash, so no sha256 of this "
                       "file appears inside it. That is not the same as its bytes "
                       "going unchecked: --verify rebuilds the whole document from "
                       "the bound files and compares, AND resolves this file as a "
                       "blob at HEAD and compares its hash there, so an edited "
                       "manifest that was never committed is a finding rather than "
                       "a document that quietly describes a different evidence set",
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
    points = doc.get("where_a_reviewer_starts")
    if not points:
        issues.append(
            "where_a_reviewer_starts is empty. An empty list validates cleanly, "
            "because there is then nothing to validate, which is why it is "
            "refused outright: the entry points are the part of this manifest a "
            "reviewer actually uses, and a manifest with none has been weakened "
            "rather than simplified")
    elif [dict(point) for point in points] \
            != [dict(point) for point in ENTRY_POINTS]:
        issues.append(
            "where_a_reviewer_starts is not the ENTRY_POINTS this module files: "
            f"{len(points)} pointer(s) filed, {[point.get('path') for point in points]}, "
            f"against {len(ENTRY_POINTS)} in the module. The comparison has to be "
            "against the constant and not against the artifact, or a manifest "
            "that has dropped its pointers re-derives itself exactly and reports "
            "no difference")
    for point in points or ():
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

    closure = doc.get("scientific_dependency_closure") or {}
    modules = closure.get("modules") or []
    if closure.get("n_modules") != len(modules):
        issues.append(f"the dependency closure files n_modules "
                      f"{closure.get('n_modules')} and lists {len(modules)}")
    if not modules:
        issues.append(
            "the dependency closure is empty, so no library module is bound and "
            "the artifacts are bound without the code that computes them")
    for module in modules:
        if module not in bound:
            issues.append(f"the dependency closure reaches {module} and it is not "
                          f"among the bound files, so the evidence depends on a "
                          f"module this manifest does not vouch for")
    for root in closure.get("roots") or []:
        if root not in bound:
            issues.append(f"the closure was walked from {root}, which is not among "
                          f"the bound files")

    # Both of these fields are recomputed here from the closure the document
    # files, and both are required to be empty. Filing them rather than raising
    # in build() is what lets verify() report them next to everything else
    # instead of dying with "could not re-derive".
    missed = closure.get("hand_listed_modules_the_closure_does_not_reach")
    expected_missed = sorted(module for module in BOUND_PATHS
                             if module.startswith(f"{SRC_ROOT}/")
                             and module not in modules)
    if missed != expected_missed:
        issues.append(
            f"hand_listed_modules_the_closure_does_not_reach is {missed} and the "
            f"closure filed beside it reaches all but {expected_missed}: the "
            f"field does not describe the list it sits in")
    for module in expected_missed:
        issues.append(f"{module} is listed by hand as a dependency of the "
                      f"bound code and the computed closure does not reach "
                      f"it, so either the walk is broken or the list is stale "
                      f"-- and dropping the module would be neither")
    unimportable = closure.get("modules_that_cannot_be_imported")
    if unimportable:
        issues.append(
            "the bound code imports a module this repository does not have at "
            "import time, so the package cannot be imported and nothing "
            "downstream of it can have produced this evidence: "
            f"{json.dumps(unimportable, sort_keys=True)[:300]}")
    relative = closure.get("relative_imports_the_walk_could_not_follow") or {}
    n_relative = sum(len(sites) for sites in relative.values())
    if closure.get("n_relative_imports_the_walk_could_not_follow") != n_relative:
        issues.append(
            "n_relative_imports_the_walk_could_not_follow is "
            f"{closure.get('n_relative_imports_the_walk_could_not_follow')} and "
            f"{n_relative} sites are listed beside it")
    if n_relative:
        issues.append(
            f"{n_relative} relative import site(s) could not be followed, so the "
            f"closure is missing whatever they reach: "
            f"{json.dumps(relative, sort_keys=True)[:300]}")

    deep = doc.get("deep_closeout") or {}
    expected_verifiers = sorted({point.get("verified_by")
                                 for point in points or ()})
    for extra in sorted(set(DEEP_INVOCATIONS) - set(expected_verifiers)):
        issues.append(
            f"DEEP_INVOCATIONS files an argv for {extra}, which no entry point "
            f"names as a verifier, so --deep would never run it and the argv is "
            f"a claim about a gate that does not exist")
    if deep.get("verifiers_it_executes") != expected_verifiers:
        issues.append(
            "deep_closeout.verifiers_it_executes is "
            f"{deep.get('verifiers_it_executes')} and the entry points name "
            f"{expected_verifiers}; the deep command must run the verifiers the "
            f"document points at, not a subset of them")
    if deep.get("n_verifiers") != len(expected_verifiers):
        issues.append(f"deep_closeout.n_verifiers is {deep.get('n_verifiers')} and "
                      f"{len(expected_verifiers)} verifiers are named")
    if not deep.get("how_exit_codes_aggregate"):
        issues.append("deep_closeout files no aggregation rule, so an exit 3 from "
                      "a verifier has no defined meaning and would be read as "
                      "either success or failure by whoever runs it")
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

    # Re-derive from DISCOVERY wherever this checkout can discover, and from the
    # filed path list only where it cannot. Re-deriving from the filed list alone
    # is the weaker check: a manifest that dropped files would be rebuilt over
    # the same smaller set and agree with itself, so discovery is what makes the
    # comparison independent of the document being verified.
    discovery = None
    if object_store_available():
        try:
            discovery = discover()
        except SystemExit as exc:
            return (exc.code if isinstance(exc.code, int) else 1), \
                "could_not_discover", issues + [
                    f"the bound set could not be discovered from what is "
                    f"committed: exit {exc.code}"], []

    try:
        fresh = build(discovery if discovery is not None
                      else [entry["path"] for entry in entries], ENTRY_POINTS)
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
            "no git object store here, so three claims could not be made: that "
            "every bound file resolves as a blob at HEAD and hashes the same "
            "there, that nothing committed under the bound trees or patterns was "
            "left out of this manifest, and that this manifest is the one that "
            "was committed rather than one edited in place. Every hash on disk "
            "checks out; what cannot be said from here is that these files are "
            "the committed ones")
        if issues:
            return 1, "differs", issues, unverifiable
        return 3, "hashes_sound_committedness_uncheckable", [], unverifiable

    mismatched, uncommitted = [], []
    paths = [entry["path"] for entry in entries]
    self_rel = _rel(path)
    on_disk = sha256_file(path)
    at_head = committed_hashes(paths + [self_rel])
    self_blob = at_head.get(self_rel)
    if self_blob is None:
        issues.append(f"{self_rel} is not in HEAD at all, so the manifest being "
                      f"verified here is not the manifest that was committed")
    elif self_blob != on_disk:
        issues.append(
            f"{self_rel}: the manifest on disk hashes to {on_disk[:12]} and HEAD "
            f"holds {self_blob[:12]}, so this file was edited after it was "
            f"committed. A document cannot carry its own hash inside itself, "
            f"which is why its bytes are compared against the object store "
            f"instead -- without that comparison an edit here is invisible, "
            f"because the re-derivation above is fed the edited document")
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


#: How each entry-point verifier is invoked by ``--deep``. Most take ``--verify``
#: and nothing else. The argv is filed here rather than assumed because a
#: verifier run with the wrong arguments can exit 0 having checked less than
#: everything: ``iter11_replay_checks.py`` without ``--all`` checks one arm and
#: not the panel, and the media manifest has a second mode that checks the
#: panel-referenced subset on its own.
DEEP_INVOCATIONS = {
    "scripts/iter11_replay_checks.py": (("--all", "--verify"),),
    "scripts/iter11_write_media_manifest.py": (("--verify",),
                                               ("--verify", "--panel-only")),
}

DEFAULT_DEEP_INVOCATION = (("--verify",),)


def deep_invocations_for(verifier: str) -> tuple[tuple[str, ...], ...]:
    return DEEP_INVOCATIONS.get(verifier, DEFAULT_DEEP_INVOCATION)


def deep_closeout(path: Path | None = None,
                  timeout: int = 1800) -> tuple[int, dict]:
    """Run every entry-point verifier and aggregate what comes back.

    ``--verify`` hashes the verifiers and reads their source for a ``--verify``
    string. That is a check on the pointers and not on the evidence: the
    top-level closeout can pass it while a scientific verifier behind one of
    those pointers fails. This function executes them.

    Aggregation is the part that has to be decided in advance, because the gates
    in this iteration use four exit codes and two of them are not failure. Exit 3
    means "this checkout does not have the section of the evidence that gate
    checks" -- the media in a fresh clone, an object store in a tarball -- and is
    counted and named rather than folded into the success total, which is what
    made a fresh checkout report failure for being a fresh checkout. Exit 1 or 2
    fails the deep closeout, and so does anything else, including a traceback,
    because a gate that crashed did not verify.

    The shallow verify runs first and its result gates the rest: executing eight
    verifiers against a manifest that does not re-derive would report on evidence
    whose binding is already in doubt.
    """
    path = OUT_PATH if path is None else path
    code, conclusion, issues, unverifiable = verify(path)
    verifiers = sorted({point["verified_by"] for point in ENTRY_POINTS})
    report: dict = {
        "shallow_verify": {
            "code": code,
            "conclusion": conclusion,
            "issues": issues,
            "unverifiable_here": unverifiable,
        },
        "verifiers_it_executes": verifiers,
        "n_verifiers": len(verifiers),
        "how_each_was_invoked": {
            verifier: [list(argv) for argv in deep_invocations_for(verifier)]
            for verifier in verifiers},
        "timeout_seconds": timeout,
        "invocations": [],
        "n_verified": 0,
        "n_incomplete_here": 0,
        "n_failed": 0,
    }
    if code in (1, 2):
        report["why_the_gates_were_not_run"] = (
            f"the manifest itself came back {code} ({conclusion}), so the "
            f"evidence these gates check is not the evidence this manifest "
            f"binds; running them would report on a binding already in doubt")
        return code, report

    # From the constant, not from the artifact. verify() has already proved the
    # filed entry points equal ENTRY_POINTS, so the two are the same list here --
    # but a deep closeout that decided WHICH GATES TO RUN by reading the document
    # it was checking would let a weakened manifest weaken its own audit.
    failed: list[dict] = []
    for verifier in verifiers:
        for argv in deep_invocations_for(verifier):
            command = [sys.executable, verifier, *argv]
            try:
                proc = subprocess.run(command, cwd=REPO_ROOT, text=True,
                                      capture_output=True, timeout=timeout)
                returned, stdout, stderr = (
                    proc.returncode, proc.stdout, proc.stderr)
            except subprocess.TimeoutExpired:
                returned, stdout, stderr = (
                    "timeout", "",
                    f"no result within {timeout}s; a gate that does not finish "
                    f"did not verify")
            lines = [line for line in stdout.strip().splitlines() if line.strip()]
            entry = {
                "verifier": verifier,
                "argv": list(argv),
                "code": returned,
                "what_it_said": lines[0] if lines else "",
                "stderr": stderr.strip()[:400],
            }
            report["invocations"].append(entry)
            if returned == 0:
                report["n_verified"] += 1
            elif returned == 3:
                report["n_incomplete_here"] += 1
            else:
                report["n_failed"] += 1
                failed.append(entry)
    report["failed"] = failed
    if failed:
        return 1, report
    if code == 3 or report["n_incomplete_here"]:
        return 3, report
    return 0, report


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
    mode.add_argument("--deep", action="store_true",
                      help="verify, then EXECUTE every entry-point verifier and "
                           "aggregate: exit 3 is counted as incomplete here, "
                           "exit 1 or 2 fails the closeout")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds to allow each gate under --deep")
    args = parser.parse_args(argv)

    if args.deep:
        code, report = deep_closeout(args.out, timeout=args.timeout)
        shallow = report["shallow_verify"]
        print(f"DEEP CLOSEOUT: manifest {shallow['conclusion'].replace('_', ' ')} "
              f"(exit {shallow['code']})")
        for issue in shallow["issues"]:
            print(f"  - {issue}")
        for note in shallow["unverifiable_here"]:
            print(f"  not checkable here: {note}")
        if "why_the_gates_were_not_run" in report:
            print(f"  gates not run: {report['why_the_gates_were_not_run']}")
            return code
        print(f"  executing {len(report['invocations'])} invocation(s) of "
              f"{report['n_verifiers']} entry-point verifier(s)")
        for entry in report["invocations"]:
            mark = {0: "verified", 3: "incomplete here"}.get(
                entry["code"], f"FAILED ({entry['code']})")
            print(f"  [{mark:>17s}] {entry['verifier']} "
                  f"{' '.join(entry['argv'])}")
            if entry["what_it_said"]:
                print(f"                     {entry['what_it_said'][:110]}")
            if entry["stderr"]:
                print(f"                     stderr: {entry['stderr'][:110]}")
        print(f"  verified {report['n_verified']}, incomplete here "
              f"{report['n_incomplete_here']}, failed {report['n_failed']}")
        if code == 1:
            print("DEEP CLOSEOUT: FAIL -- a gate contradicted its own evidence")
        elif code == 3:
            print("DEEP CLOSEOUT: INCOMPLETE HERE -- every gate that could run "
                  "ran, and none contradicted anything; the ones that could not "
                  "run say so")
        else:
            print("DEEP CLOSEOUT: PASS -- every entry-point verifier ran and "
                  "none contradicted its evidence")
        return code

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
        closure = filed["scientific_dependency_closure"]
        print(f"  closure    {closure['n_modules']} library modules walked from "
              f"{closure['n_roots_walked']} bound .py files")
        print(f"  entry points {len(filed['where_a_reviewer_starts'])}, each "
              f"bound and each with a verifier that implements --verify")
        deep = filed["deep_closeout"]
        print(f"  deep       {deep['n_verifiers']} verifier(s) executed by "
              f"--deep; --verify hashes them and runs none")
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
    closure = doc["scientific_dependency_closure"]
    print(f"  closure    {closure['n_modules']} library modules walked from "
          f"{closure['n_roots_walked']} bound .py files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
