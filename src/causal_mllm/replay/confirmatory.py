"""Confirmatory-run gate for Iteration 11 cross-model targets.

Revision pinning was already enforced (``resolve_model(confirmatory=True)``
rejects a null/branch revision), but pinning the WEIGHTS is only one of the
dimensions the frozen protocol fixes. A confirmatory run could still be
launched against a different panel, a truncated family subset, a smaller
generation cap, sampling instead of greedy decoding, a dirty working tree,
or a dependency environment other than the one certified at preflight — and
every one of those produces a complete-looking artifact whose numbers are
not comparable to the frozen Qwen3.5-9B reference.

This module makes those dimensions explicit and fail-closed. It collects
EVERY violation and reports them together, because a run that fails three
checks should not have to be relaunched three times to discover that.

Scope: Iteration 11 targets only (``model_spec`` is required). The frozen
legacy single-model path (Iterations 8-10, ``--model-key`` absent) is left
untouched so its evidence stays byte-for-byte reproducible.

The gate is deliberately conservative about what counts as a violation. It
does NOT re-derive the science; it checks the dimensions the frozen
protocol states numerically, and refuses to guess where the protocol is
silent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import (
    DEFAULT_LOCK,
    ResolvedModel,
    dependency_lock_sha256,
    is_immutable_revision,
    load_lock,
    verify_active_dependency_lock,
)
from causal_mllm.seeds import code_tree_status, get_git_commit
from causal_mllm.validation.relations import _file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The frozen Iteration 11 protocol. Immutable: never written by this
#: module, only read and hashed.
FROZEN_PROTOCOL = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"

#: Root of the 11.5 eligibility reports (one directory per model_key).
ELIGIBILITY_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "eligibility"

#: Canonical root for confirmatory generations (must match the CLI's
#: default so a default run passes and a redirected one does not).
GENERATIONS_ROOT = "outputs/iteration_11/generations"

#: 11.5 artifact name (per the frozen plan: a signed ``preflight_report.json``
#: whose ``eligible=true`` is required to start the full run).
ELIGIBILITY_REPORT_FILE = "preflight_report.json"

VALIDATED_FAMILIES_FILE = "validated_families.jsonl"

#: Repo-root-relative prefixes a confirmatory run writes into. Excluded
#: from the clean-tree determination because a run must not be blocked by
#: its own (or a sibling target's) regenerated evidence; every other
#: tracked modification still fails the gate. Excluded paths are recorded
#: in the gate evidence rather than silently dropped.
OWN_OUTPUT_PREFIXES = ("outputs/iteration_11/generations/",)

#: Sampling values that are INERT under greedy decoding. The frozen
#: protocol records Iteration 10's temperature=0.0/top_p=1.0 and
#: normalizes them to null, noting that with ``do_sample=false`` they are
#: meaningless; both spellings therefore describe the same decoding, and
#: rejecting either would be a false positive.
_INERT_TEMPERATURES = (None, 0.0)
_INERT_TOP_P = (None, 1.0)


def protocol_sha256(protocol_path: str | Path | None = None) -> str:
    """Raw-byte SHA-256 of the frozen protocol file.

    This is the "protocol fingerprint" an eligibility report must bind to:
    it changes if and only if the frozen protocol file changes, which it
    must not.
    """
    path = Path(protocol_path) if protocol_path else FROZEN_PROTOCOL
    if not path.exists():
        raise ReplayError(f"frozen protocol not found: {path}")
    digest = _file_sha256(path)
    if digest is None:
        raise ReplayError(f"frozen protocol unreadable: {path}")
    return digest


def load_protocol(protocol_path: str | Path | None = None) -> dict:
    """The frozen protocol document."""
    path = Path(protocol_path) if protocol_path else FROZEN_PROTOCOL
    if not path.exists():
        raise ReplayError(f"frozen protocol not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{path}: unparseable frozen protocol: {exc}") \
            from exc
    if not isinstance(doc, dict):
        raise ReplayError(f"{path}: frozen protocol is not an object")
    return doc


def eligibility_report_path(
    model_key: str,
    root: str | Path | None = None,
) -> Path:
    """Where the 11.5 eligibility report for ``model_key`` must live."""
    base = Path(root) if root else ELIGIBILITY_ROOT
    return base / model_key / ELIGIBILITY_REPORT_FILE


def load_eligibility_report(
    model_key: str,
    root: str | Path | None = None,
) -> dict | None:
    """The 11.5 eligibility report, or None when it does not exist."""
    path = eligibility_report_path(model_key, root)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayError(
            f"{path}: unparseable eligibility report: {exc}") from exc
    return doc if isinstance(doc, dict) else None


class _Gate:
    """Accumulates violations and the evidence each check produced."""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self.checks: dict[str, Any] = {}

    def fail(self, message: str) -> None:
        self.violations.append(message)

    def record(self, name: str, value: Any) -> None:
        self.checks[name] = value


def _check_panel_identity(gate: _Gate, input_dir: Path,
                          protocol: dict) -> str | None:
    """The input panel must BE the frozen 100-family Scale-C panel.

    Checked by raw-byte hash, not by path: a byte-identical panel in
    another location is the same panel, and a same-named file with one
    edited record is not. The hash is over raw bytes because that is what
    the frozen protocol and the replay runner both record; a
    whitespace-normalized digest of the same file is a DIFFERENT value and
    cannot be compared against either.
    """
    frozen = protocol["frozen_inputs"]["panel_validated_families_sha256"]
    gate.record("frozen_panel_sha256", frozen)
    panel = input_dir / VALIDATED_FAMILIES_FILE
    gate.record("panel_path", str(panel))
    if not panel.exists():
        gate.fail(f"{panel} not found — a confirmatory run consumes the "
                  f"frozen validated_families.jsonl only")
        return None
    actual = _file_sha256(panel)
    gate.record("panel_sha256", actual)
    if actual != frozen:
        gate.fail(
            f"panel is not the frozen Iteration 11 panel: raw-byte SHA-256 "
            f"{actual} != frozen {frozen} ({panel}). The estimands are "
            f"defined over the frozen 100-family Scale-C panel; a "
            f"different or edited panel is a different experiment.")
        return None
    return str(panel)


def _check_family_coverage(gate: _Gate, records: list[dict],
                           protocol: dict) -> None:
    """Exactly the frozen family count, each with all six variants."""
    frozen_inputs = protocol["frozen_inputs"]
    expected_families = frozen_inputs["n_families"]
    expected_variants = list(frozen_inputs["variants"])
    gate.record("n_families", len(records))
    gate.record("expected_n_families", expected_families)

    if expected_variants != list(ALL_VARIANT_NAMES):
        # Not a run-level violation: the frozen protocol disagrees with the
        # code's variant vocabulary, which must be resolved by a human.
        raise ReplayError(
            f"frozen protocol variants {expected_variants} do not match "
            f"ALL_VARIANT_NAMES {list(ALL_VARIANT_NAMES)}; the gate cannot "
            f"certify variant coverage")
    if len(records) != expected_families:
        gate.fail(
            f"panel holds {len(records)} families; the frozen protocol "
            f"requires exactly {expected_families}. A subset (e.g. a smoke "
            f"or --max-families run) is not a confirmatory panel and must "
            f"not be labelled one.")

    incomplete: list[str] = []
    extra: list[str] = []
    for rec in records:
        family_id = rec.get("family_id")
        variants = rec.get("variants")
        if not isinstance(variants, dict):
            incomplete.append(f"{family_id}:<no variants block>")
            continue
        have = set(variants)
        if have != set(expected_variants):
            if set(expected_variants) - have:
                incomplete.append(
                    f"{family_id}:missing "
                    f"{sorted(set(expected_variants) - have)}")
            if have - set(expected_variants):
                extra.append(
                    f"{family_id}:unexpected {sorted(have - set(expected_variants))}")
    gate.record("families_with_incomplete_variants", incomplete[:20])
    gate.record("n_families_with_incomplete_variants", len(incomplete))
    if incomplete:
        gate.fail(
            f"{len(incomplete)} family/families lack all six frozen "
            f"variants, e.g. {incomplete[:5]} — the factorial contrast "
            f"(Delta_T / Delta_V / Delta_TV) is undefined for them")
    if extra:
        gate.fail(
            f"{len(extra)} family/families carry variants outside the "
            f"frozen six, e.g. {extra[:5]} — an undeclared condition would "
            f"enter the analysis")


def _check_decoding(gate: _Gate, config: ReplayConfig, protocol: dict) -> None:
    """Greedy decoding, thinking suppressed, and the frozen uniform cap."""
    cap = protocol["uniform_cap_rule"]["initial_cap"]
    effective = protocol["frozen_inputs"]["effective_decoding"]
    gate.record("max_new_tokens", config.max_new_tokens)
    gate.record("frozen_cap", cap)
    gate.record("do_sample", config.do_sample)
    gate.record("temperature", config.temperature)
    gate.record("top_p", config.top_p)
    gate.record("enable_thinking", config.enable_thinking)
    gate.record("frozen_effective_decoding", effective)

    if config.max_new_tokens != cap:
        gate.fail(
            f"max_new_tokens={config.max_new_tokens} != the frozen uniform "
            f"cap {cap}. ReplayConfig's default is 256, so omitting "
            f"--max-new-tokens silently truncates every response; the "
            f"uniform_cap_rule forbids a per-target cap and requires a "
            f"five-model replay if the cap must change.")
    if config.do_sample is not False:
        gate.fail(
            f"do_sample={config.do_sample!r}; the frozen protocol requires "
            f"greedy decoding (do_sample=false)")
    if config.temperature not in _INERT_TEMPERATURES:
        gate.fail(
            f"temperature={config.temperature!r} is not an inert greedy "
            f"value {list(_INERT_TEMPERATURES)}")
    if config.top_p not in _INERT_TOP_P:
        gate.fail(
            f"top_p={config.top_p!r} is not an inert greedy value "
            f"{list(_INERT_TOP_P)}")
    if config.enable_thinking is not False:
        gate.fail(
            f"enable_thinking={config.enable_thinking!r}; the frozen "
            f"protocol requires thinking disabled so the judged response is "
            f"the answer itself")
    if config.prompt_template_revision != \
            protocol["frozen_inputs"]["prompt_template_revision"]:
        gate.fail(
            f"prompt_template_revision={config.prompt_template_revision!r} "
            f"!= frozen "
            f"{protocol['frozen_inputs']['prompt_template_revision']!r}")


def _check_clean_tree(gate: _Gate) -> None:
    """The tree must be clean enough for ``code_commit`` to reconstruct it.

    Untracked files count. Ignoring them was a hole: an untracked
    ``sitecustomize.py`` or ``conftest.py``, a top-level module shadowing an
    installed package, or a new module that tracked code imports all change
    what executes while ``code_commit`` still points at a tree that cannot
    reproduce it — and the run would have recorded ``git_dirty: false``.

    Only :data:`OWN_OUTPUT_PREFIXES` and cache/transient paths are excluded,
    and both exclusions are recorded in the gate evidence rather than
    applied silently.

    ``code_tree_status`` reports ``dirty=None`` outside a git repository,
    which is treated as a violation: unknown provenance cannot be
    certified, and accepting it would make a source tarball
    indistinguishable from a committed tree.
    """
    commit = get_git_commit()
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    dirty = tree["dirty"]
    gate.record("code_commit", commit)
    gate.record("git_dirty", dirty)
    gate.record("git_dirty_paths", tree["dirty_paths"])
    gate.record("git_untracked_paths", tree["untracked_paths"])
    gate.record("git_dirty_excluded_own_outputs",
                tree["excluded_own_outputs"])
    gate.record("git_dirty_excluded_cache_paths",
                tree["excluded_cache_paths"])
    if commit is None:
        gate.fail("no git commit resolvable — code provenance is unknown, so "
                  "this run cannot be reconstructed from evidence")
    if dirty is None:
        gate.fail("git tree status unknown (not a git repository?) — "
                  "confirmatory runs require a verified clean tree")
    elif dirty is True:
        untracked = tree["untracked_paths"]
        kind = ("untracked files" if untracked and
                len(untracked) == len(tree["dirty_paths"])
                else "uncommitted changes")
        gate.fail(
            f"working tree is not clean at commit {commit}: {kind} at "
            f"{tree['dirty_paths']}. The code that would execute is NOT the "
            f"code that commit contains — an untracked module, "
            f"sitecustomize.py or shadowing top-level file changes execution "
            f"without being recorded anywhere — so the recorded code_commit "
            f"could not reconstruct this run. Commit or remove them first.")


def _check_revisions(gate: _Gate, model_spec: ResolvedModel,
                     lock_path: str | Path | None) -> None:
    """Model AND processor revisions must be immutable 40-hex SHAs.

    The processor is checked separately because it is resolved from its own
    lock entry: pinning the weights while the tokenizer/chat-template or
    image processor floats would change prompt rendering without moving the
    model revision.
    """
    path = Path(lock_path) if lock_path else DEFAULT_LOCK
    gate.record("lock_path", str(path))
    gate.record("model_revision", model_spec.revision)
    gate.record("revision_source", model_spec.revision_source)
    if not is_immutable_revision(model_spec.revision):
        gate.fail(
            f"{model_spec.model_key}: model revision "
            f"{model_spec.revision!r} is not an immutable 40-hex SHA")
    lock = load_lock(path)
    entry = lock.get(model_spec.model_key)
    if not isinstance(entry, dict):
        gate.fail(
            f"{model_spec.model_key}: not present in the lock at {path}; a "
            f"confirmatory run requires a locked revision")
        return
    locked_rev = entry.get("revision")
    proc_rev = entry.get("processor_revision")
    gate.record("locked_revision", locked_rev)
    gate.record("locked_processor_revision", proc_rev)
    if locked_rev != model_spec.revision:
        gate.fail(
            f"{model_spec.model_key}: resolved revision "
            f"{model_spec.revision!r} != locked revision {locked_rev!r}")
    if not is_immutable_revision(proc_rev):
        gate.fail(
            f"{model_spec.model_key}: processor revision {proc_rev!r} is not "
            f"an immutable 40-hex SHA")
    if model_spec.quantization not in (None, "none"):
        gate.fail(
            f"{model_spec.model_key}: quantization "
            f"{model_spec.quantization!r}; the frozen protocol forbids "
            f"quantization")


def _check_dependencies(gate: _Gate, lock_path: str | Path | None) -> None:
    """The environment RUNNING must be the environment that was locked.

    The fingerprint binds a hash read from the lock FILE, which on its own
    proves nothing about the live interpreter. This compares the two, and
    separately refuses any third-party editable install, whose source can
    change without the recorded revision moving.
    """
    check = verify_active_dependency_lock(lock_path, strict=False)
    gate.record("dependency_lock_check", check)
    offenders = check.get("third_party_editable_vcs") or {}
    if offenders:
        gate.fail(
            f"the active environment has third-party editable VCS install(s) "
            f"{sorted(offenders)}; an editable dependency's source can change "
            f"without its recorded revision moving, so this environment "
            f"cannot be certified reproducible. Use a dedicated Iteration 11 "
            f"environment with no third-party editable installs.")
    if check.get("reason"):
        gate.fail(f"dependency environment unverified: {check['reason']}")
        return
    differences = check.get("differences") or {}
    if differences:
        detail = "; ".join(
            f"{f}: locked={v['locked']!r} active={v['active']!r}"
            for f, v in sorted(differences.items()))
        gate.fail(
            f"active dependency environment differs from the recorded "
            f"lock — {detail}")


#: 11.5 selects a FIXED stratified subset: 12 families x 6 variants per
#: model, giving 72 generations per target and 288 across the four. The
#: count comes from the approved Iteration 11 plan rather than from the
#: frozen protocol document, which does not state it, so it is declared here
#: and cited rather than read from a file that does not contain it.
ELIGIBILITY_N_FAMILIES = 12
ELIGIBILITY_N_VARIANTS = len(ALL_VARIANT_NAMES)
ELIGIBILITY_N_ATTEMPTS = ELIGIBILITY_N_FAMILIES * ELIGIBILITY_N_VARIANTS

#: Fields every 11.5 eligibility report must carry. A report missing one is
#: rejected rather than partially trusted: each field is what lets a later
#: reader tie the eligibility decision to a specific model, revision, code
#: tree, environment and family selection.
ELIGIBILITY_REQUIRED_FIELDS = (
    "status",
    "eligible",
    "model_key",
    "model_id",
    "model_revision",
    "processor_revision",
    "code_commit",
    "git_dirty",
    "protocol_sha256",
    "dependency_lock_sha256",
    "selected_family_ids",
    "selected_families_sha256",
    "n_selected_families",
    "variants",
    "n_expected_attempts",
    "n_attempts",
    "n_succeeded",
    "truncation_by_variant",
    "gates",
)


def selected_families_sha256(family_ids) -> str:
    """Canonical digest of a family selection.

    Sorted and newline-joined so the hash depends on the SET selected and
    not on the order a report happened to list it in. Making the recipe
    explicit is what lets the gate recompute and verify the recorded value
    instead of merely checking that one is present.
    """
    canonical = "\n".join(sorted(str(f) for f in family_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gate_entry_passed(entry) -> bool:
    """Did one detailed gate result pass?

    Accepts ``True``, ``{"passed": true, ...}`` and ``{"status": "PASS"}``
    so 11.5 can record either a bare flag or a richer result, while
    anything ambiguous counts as not passed.
    """
    if entry is True:
        return True
    if isinstance(entry, dict):
        if "passed" in entry:
            return entry["passed"] is True
        return entry.get("status") == "PASS"
    return False


def validate_eligibility_report(
    report: dict,
    *,
    model_spec: ResolvedModel,
    expected_protocol_sha: str,
    expected_lock_sha: str | None,
    panel_family_ids: set | None = None,
) -> list[str]:
    """Strict schema + content validation of an 11.5 eligibility report.

    Returns the list of violations (empty means valid). The earlier version
    of this gate recorded ``code_commit`` without requiring it, so a report
    with ``code_commit: null`` — or one written about a different
    ``model_key`` entirely — could pass. Every field below is now required
    AND checked against the run it is being used to authorize.

    Args:
        report: The parsed eligibility report.
        model_spec: The target this run resolves.
        expected_protocol_sha: Digest of the frozen protocol in force.
        expected_lock_sha: Digest of the dependency lock in force; the
            report must have been produced under the same one.
        panel_family_ids: The frozen panel's family ids, when known, so the
            selected subset can be confirmed to come from that panel.
    """
    problems: list[str] = []
    missing = [f for f in ELIGIBILITY_REQUIRED_FIELDS
               if report.get(f) is None]
    if missing:
        problems.append(
            f"eligibility report is missing required field(s) {missing}")
        # Field-level checks below cannot run without the fields.
        return problems

    if report.get("status") != "PASS":
        problems.append(
            f"eligibility status is {report.get('status')!r}, not 'PASS'")
    if report.get("eligible") is not True:
        problems.append(
            f"eligibility report does not assert eligible=true "
            f"(got {report.get('eligible')!r})")

    # --- identity: this report must be about THIS target ----------------
    if report.get("model_key") != model_spec.model_key:
        problems.append(
            f"eligibility report is for model_key "
            f"{report.get('model_key')!r}, not {model_spec.model_key!r}")
    if report.get("model_id") != model_spec.model_id:
        problems.append(
            f"eligibility report is for model_id {report.get('model_id')!r}, "
            f"not {model_spec.model_id!r}")
    if report.get("model_revision") != model_spec.revision:
        problems.append(
            f"eligibility was certified for revision "
            f"{report.get('model_revision')!r} but this run resolves "
            f"{model_spec.revision!r} — eligibility does not transfer "
            f"across revisions")
    if not is_immutable_revision(report.get("processor_revision")):
        problems.append(
            f"eligibility processor_revision "
            f"{report.get('processor_revision')!r} is not an immutable "
            f"40-hex SHA")

    # --- code provenance of the eligibility run itself ------------------
    if not is_immutable_revision(report.get("code_commit")):
        problems.append(
            f"eligibility code_commit {report.get('code_commit')!r} is not "
            f"an immutable 40-hex SHA, so the code that certified "
            f"eligibility cannot be reconstructed")
    if report.get("git_dirty") is not False:
        problems.append(
            f"eligibility report was not produced from a clean tree "
            f"(git_dirty={report.get('git_dirty')!r})")

    # --- what it was checked against ------------------------------------
    if report.get("protocol_sha256") != expected_protocol_sha:
        problems.append(
            f"eligibility report binds protocol sha256 "
            f"{report.get('protocol_sha256')!r}, not the current frozen "
            f"protocol {expected_protocol_sha!r}")
    if expected_lock_sha is not None and \
            report.get("dependency_lock_sha256") != expected_lock_sha:
        problems.append(
            f"eligibility report binds dependency_lock_sha256 "
            f"{report.get('dependency_lock_sha256')!r}, not the lock in "
            f"force {expected_lock_sha!r} — eligibility was certified "
            f"under a different environment")

    # --- the selected family subset -------------------------------------
    ids = report.get("selected_family_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        problems.append("selected_family_ids must be a list of family ids")
        ids = []
    if len(ids) != ELIGIBILITY_N_FAMILIES:
        problems.append(
            f"selected {len(ids)} families; 11.5 requires exactly "
            f"{ELIGIBILITY_N_FAMILIES}")
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"selected_family_ids contains duplicates {duplicates}")
    if report.get("n_selected_families") != len(ids):
        problems.append(
            f"n_selected_families={report.get('n_selected_families')} "
            f"disagrees with the {len(ids)} ids listed")
    expected_hash = selected_families_sha256(ids) if ids else None
    if ids and report.get("selected_families_sha256") != expected_hash:
        problems.append(
            f"selected_families_sha256 "
            f"{report.get('selected_families_sha256')!r} does not match the "
            f"listed family ids (recomputed {expected_hash!r})")
    if panel_family_ids is not None and ids:
        outside = sorted(set(ids) - set(panel_family_ids))
        if outside:
            problems.append(
                f"selected families {outside} are not in the frozen "
                f"100-family panel — eligibility must be established on a "
                f"subset of the panel that will actually be replayed")

    # --- six-variant coverage and the 72/72 requirement -----------------
    if list(report.get("variants") or []) != list(ALL_VARIANT_NAMES):
        problems.append(
            f"eligibility variants {report.get('variants')!r} != the frozen "
            f"six {list(ALL_VARIANT_NAMES)}")
    if report.get("n_expected_attempts") != ELIGIBILITY_N_ATTEMPTS:
        problems.append(
            f"n_expected_attempts={report.get('n_expected_attempts')} != "
            f"{ELIGIBILITY_N_ATTEMPTS} "
            f"({ELIGIBILITY_N_FAMILIES} families x "
            f"{ELIGIBILITY_N_VARIANTS} variants)")
    if report.get("n_attempts") != ELIGIBILITY_N_ATTEMPTS:
        problems.append(
            f"n_attempts={report.get('n_attempts')} != "
            f"{ELIGIBILITY_N_ATTEMPTS}")
    if report.get("n_succeeded") != ELIGIBILITY_N_ATTEMPTS:
        problems.append(
            f"n_succeeded={report.get('n_succeeded')} != "
            f"{ELIGIBILITY_N_ATTEMPTS}; 11.5 requires every one of the "
            f"{ELIGIBILITY_N_ATTEMPTS} generations to succeed")
    truncation = report.get("truncation_by_variant")
    if not isinstance(truncation, dict):
        problems.append("truncation_by_variant must be an object")
    else:
        absent = [v for v in ALL_VARIANT_NAMES if v not in truncation]
        if absent:
            problems.append(
                f"truncation_by_variant is missing variant(s) {absent}")
        wrong = {v: truncation[v].get("n")
                 for v in ALL_VARIANT_NAMES
                 if isinstance(truncation.get(v), dict)
                 and truncation[v].get("n") != ELIGIBILITY_N_FAMILIES}
        if wrong:
            problems.append(
                f"truncation_by_variant counts per variant must be "
                f"{ELIGIBILITY_N_FAMILIES}; got {wrong}")

    # --- detailed gate results -----------------------------------------
    gates = report.get("gates")
    if not isinstance(gates, dict) or not gates:
        problems.append(
            "gates must be a non-empty object of detailed per-gate results; "
            "a bare overall status is not auditable")
    else:
        failed = sorted(name for name, entry in gates.items()
                        if not _gate_entry_passed(entry))
        if failed:
            problems.append(f"eligibility gate(s) failed: {failed}")
    return problems


def _check_eligibility(gate: _Gate, model_spec: ResolvedModel,
                       protocol_path: str | Path | None,
                       eligibility_root: str | Path | None,
                       lock_path: str | Path | None,
                       panel_family_ids: set | None = None) -> None:
    """A PASSING, fully-specified 11.5 eligibility report for this target.

    Technical eligibility (the target can represent the same semantic role
    and image structure) is established at 11.5 and must precede the full
    run; without this check a model could be generated for and analysed
    despite having failed eligibility, which is exactly the selection
    freedom the frozen protocol removes.
    """
    path = eligibility_report_path(model_spec.model_key, eligibility_root)
    gate.record("eligibility_report_path", str(path))
    expected_protocol_sha = protocol_sha256(protocol_path)
    gate.record("protocol_sha256", expected_protocol_sha)
    expected_lock_sha = dependency_lock_sha256(lock_path)
    gate.record("expected_dependency_lock_sha256", expected_lock_sha)
    report = load_eligibility_report(model_spec.model_key, eligibility_root)
    if report is None:
        gate.fail(
            f"{model_spec.model_key}: no 11.5 eligibility report at {path}; "
            f"a confirmatory run may only start once technical eligibility "
            f"has been signed off for this target")
        return
    problems = validate_eligibility_report(
        report, model_spec=model_spec,
        expected_protocol_sha=expected_protocol_sha,
        expected_lock_sha=expected_lock_sha,
        panel_family_ids=panel_family_ids)
    gate.record("eligibility_report_violations", problems)
    gate.record("eligibility_code_commit", report.get("code_commit"))
    gate.record("eligibility_git_dirty", report.get("git_dirty"))
    gate.record("eligibility_n_gates",
                len(report.get("gates") or {})
                if isinstance(report.get("gates"), dict) else None)
    for problem in problems:
        gate.fail(f"{model_spec.model_key}: {problem}")


def _check_no_overwrite(gate: _Gate, overwrite: bool) -> None:
    """A confirmatory run may never overwrite retained evidence."""
    gate.record("overwrite", overwrite)
    if overwrite:
        gate.fail(
            "overwrite=True; the frozen stop-conditions forbid overwriting "
            "retained evidence. Use a new run_id, or --resume to continue "
            "an interrupted run.")


def _check_scope(gate: _Gate, max_families: int | None,
                 output_root: str | Path | None, model_key: str) -> None:
    """The run must cover the whole panel and land in the tracked tree.

    ``--max-families`` is a smoke facility. A run that replays a prefix of
    a 100-family panel still writes a complete-looking report, so it is
    rejected here rather than detected later by a missing-cell count.
    ``--output-root`` likewise: evidence written outside the Iteration 11
    generations tree is not where the analysis or the manifest looks for
    it, and may be gitignored.
    """
    gate.record("max_families", max_families)
    if max_families is not None:
        gate.fail(
            f"max_families={max_families}; a confirmatory run replays the "
            f"ENTIRE frozen panel. --max-families is a smoke facility and "
            f"a partial run must not be labelled confirmatory.")
    expected_root = f"{GENERATIONS_ROOT}/{model_key}"
    gate.record("output_root", str(output_root) if output_root else None)
    gate.record("expected_output_root", expected_root)
    if output_root is not None and \
            str(output_root).rstrip("/") != expected_root:
        gate.fail(
            f"output_root={str(output_root)!r} != {expected_root!r}; "
            f"confirmatory evidence must land in the tracked Iteration 11 "
            f"generations tree for this model_key")


def enforce_confirmatory_protocol(
    *,
    input_dir: str | Path,
    config: ReplayConfig,
    model_spec: ResolvedModel | None,
    overwrite: bool = False,
    max_families: int | None = None,
    output_root: str | Path | None = None,
    lock_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    eligibility_root: str | Path | None = None,
) -> dict:
    """Enforce the frozen protocol on a confirmatory Iteration 11 run.

    Args:
        input_dir: Dataset directory holding validated_families.jsonl.
        config: The replay configuration about to be used.
        model_spec: The resolved target. Required — the legacy
            single-model path is out of scope and must not be gated here.
        overwrite: The ``--overwrite`` flag value.
        max_families: The ``--max-families`` value (must be None).
        output_root: The resolved output root for this run.
        lock_path: Lock file supplying revisions and the dependency lock.
        protocol_path: Frozen protocol (override for tests).
        eligibility_root: 11.5 report root (override for tests).

    Returns:
        The gate evidence: every value checked. Callers should persist it
        so a PASS is itself auditable.

    Raises:
        ReplayError: listing ALL violations found, so one launch surfaces
            every problem rather than one per attempt.
    """
    if model_spec is None:
        raise ReplayError(
            "the confirmatory gate requires a resolved Iteration 11 target "
            "(model_spec); the frozen legacy single-model path is out of "
            "scope and must not be routed through this gate")
    protocol = load_protocol(protocol_path)
    gate = _Gate()
    gate.record("model_key", model_spec.model_key)
    gate.record("model_id", model_spec.model_id)
    gate.record("adapter", model_spec.adapter)

    _check_no_overwrite(gate, overwrite)
    _check_scope(gate, max_families, output_root, model_spec.model_key)
    _check_clean_tree(gate)
    _check_decoding(gate, config, protocol)
    _check_revisions(gate, model_spec, lock_path)
    _check_dependencies(gate, lock_path)

    # The panel is read BEFORE eligibility so the selected 12-family subset
    # can be confirmed to come from the panel that will actually be
    # replayed, rather than from some other selection.
    panel_family_ids: set | None = None
    panel = _check_panel_identity(gate, Path(input_dir), protocol)
    if panel is not None:
        records = [
            json.loads(line)
            for line in Path(panel).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _check_family_coverage(gate, records, protocol)
        panel_family_ids = {r.get("family_id") for r in records}

    _check_eligibility(gate, model_spec, protocol_path, eligibility_root,
                       lock_path, panel_family_ids)

    result = {
        "gate": "iteration11_confirmatory",
        "passed": not gate.violations,
        "n_violations": len(gate.violations),
        "violations": gate.violations,
        "checks": gate.checks,
    }
    if gate.violations:
        listing = "\n".join(f"  - {v}" for v in gate.violations)
        raise ReplayError(
            f"confirmatory protocol gate FAILED for "
            f"{model_spec.model_key} with {len(gate.violations)} "
            f"violation(s):\n{listing}\n"
            f"No output was generated. The frozen protocol fixes these "
            f"dimensions; a run that departs from any of them is not "
            f"comparable to the Qwen3.5-9B reference.")
    return result
