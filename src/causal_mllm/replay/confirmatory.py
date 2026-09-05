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

import json
from pathlib import Path
from typing import Any, Sequence

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
from causal_mllm.replay.selection import (
    N_ELIGIBILITY_ATTEMPTS,
    N_ELIGIBILITY_FAMILIES,
    N_ELIGIBILITY_VARIANTS,
    SELECTION_ARTIFACT,
    derive_frozen_selection,
    selected_families_sha256,
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

#: Canonical root for the 11.5 eligibility generations. Kept separate from
#: :data:`GENERATIONS_ROOT` so a 12-family eligibility run can never be
#: mistaken for confirmatory evidence by anything that globs the latter.
ELIGIBILITY_GENERATIONS_ROOT = "outputs/iteration_11/eligibility/generations"

#: 11.5 artifact name (per the frozen plan: a signed ``preflight_report.json``
#: whose ``eligible=true`` is required to start the full run).
ELIGIBILITY_REPORT_FILE = "preflight_report.json"

VALIDATED_FAMILIES_FILE = "validated_families.jsonl"

#: Repo-root-relative prefixes whose contents are a replay run's PRODUCTS.
#: Excluded from the clean-tree determination because a run must not be
#: blocked by its own (or a sibling target's) regenerated evidence — and
#: because a resumed run would otherwise see its own prior outputs as
#: untracked files, compute a dirtier tree than its first attempt did, and
#: so bind a DIFFERENT fingerprint that refuses the resume. Every other
#: tracked modification still fails the gate, and excluded paths are
#: recorded in the gate evidence rather than silently dropped.
#:
#: The exclusions are STAGE-SCOPED, because "this stage's own output" is not
#: the same set for both stages:
#:
#: * a CONFIRMATORY run writes only under ``generations/``. The whole of
#:   ``eligibility/`` stays unexcluded, so ``selection.json`` and the 11.5
#:   ``preflight_report.json`` must be committed before 11.6 starts. Both
#:   are INPUTS that authorize the run, and an uncommitted — therefore
#:   unreproducible — document must never be what authorized generation.
#: * an 11.5 ELIGIBILITY run writes its generations *and* its report under
#:   ``eligibility/``, so that whole tree is its own output (see
#:   :data:`ELIGIBILITY_OWN_OUTPUT_PREFIXES`). Without it, the first
#:   target's report would make the tree dirty and block every subsequent
#:   target even though nothing about the code changed.
OWN_OUTPUT_PREFIXES = (f"{GENERATIONS_ROOT}/",)

#: The 11.5 stage's own output tree: its generations and its report.
ELIGIBILITY_OWN_OUTPUT_PREFIXES = ("outputs/iteration_11/eligibility/",)

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


def _check_clean_tree(gate: _Gate,
                      own_output_prefixes: Sequence[str] = OWN_OUTPUT_PREFIXES,
                      ) -> None:
    """The tree must be clean enough for ``code_commit`` to reconstruct it.

    Untracked files count. Ignoring them was a hole: an untracked
    ``sitecustomize.py`` or ``conftest.py``, a top-level module shadowing an
    installed package, or a new module that tracked code imports all change
    what executes while ``code_commit`` still points at a tree that cannot
    reproduce it — and the run would have recorded ``git_dirty: false``.

    Only ``own_output_prefixes`` and cache/transient paths are excluded, and
    both exclusions are recorded in the gate evidence rather than applied
    silently. The prefix list is stage-scoped: each stage excludes what IT
    writes, so an 11.5 run excludes its own report while a confirmatory run
    does not (that report is an input which authorizes it).

    ``code_tree_status`` reports ``dirty=None`` outside a git repository,
    which is treated as a violation: unknown provenance cannot be
    certified, and accepting it would make a source tarball
    indistinguishable from a committed tree.
    """
    commit = get_git_commit()
    tree = code_tree_status(exclude_prefixes=own_output_prefixes)
    dirty = tree["dirty"]
    gate.record("code_commit", commit)
    gate.record("git_dirty", dirty)
    gate.record("git_own_output_prefixes", list(own_output_prefixes))
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
    offenders = check.get("third_party_editable_installs") or {}
    if offenders:
        detail = "; ".join(
            f"{name} ({info.get('kind')}: {info.get('target')})"
            for name, info in sorted(offenders.items()))
        gate.fail(
            f"the active environment has third-party editable install(s): "
            f"{detail}. An editable dependency's source can change without "
            f"anything in a `pip freeze` hash moving, so this environment "
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
#: counts and the selection recipe live in
#: :mod:`causal_mllm.replay.selection` so that the gate and the 11.5
#: producer cannot each define their own; they are re-exported here under the
#: names this module's contract already uses.
ELIGIBILITY_N_FAMILIES = N_ELIGIBILITY_FAMILIES
ELIGIBILITY_N_VARIANTS = N_ELIGIBILITY_VARIANTS
ELIGIBILITY_N_ATTEMPTS = N_ELIGIBILITY_ATTEMPTS

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
    "git_dirty_code_paths",
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


#: The EXACT set of detailed gates an 11.5 report must carry, and the
#: evidence each must supply. ``passed: true`` on its own is not evidence:
#: any non-empty dict of passing entries used to be accepted, so a report
#: could authorize a confirmatory run while omitting the vision-path,
#: truncation, determinism, terminal-query and revision checks entirely —
#: and a test asserted that ``{"only_gate": true}`` was valid.
#:
#: The set is closed in both directions: a missing gate means the check was
#: never performed, and an unexpected one means the report is claiming
#: authority from something this contract does not define.
ELIGIBILITY_GATE_EVIDENCE = {
    # All 72 generations completed. Technical eligibility has no meaning if
    # some cells silently failed.
    "generations_complete": ("n_attempts", "n_succeeded", "n_failed"),
    # Truncation reviewed per variant. Non-zero is a protocol-level STOP
    # rather than a warning: raising the cap would require a uniform
    # five-model replay including the frozen 9B reference.
    "truncation_reviewed": ("n_truncated", "truncation_rate",
                            "max_variant_spread"),
    # The image path actually engaged. A target that silently degrades to
    # text-only would still produce fluent, complete responses.
    "vision_path_engaged": ("n_image_bearing_cells", "min_image_token_count"),
    # The shared terminal query survived variant construction, which is what
    # makes the six variants comparable within a family.
    "terminal_query_invariant": ("n_families_checked", "n_mismatched"),
    # Weights AND processor pinned to immutable revisions: pinning the
    # weights while the chat template or image processor floats changes
    # prompt rendering without moving the model revision.
    "revision_pinned": ("model_revision", "processor_revision"),
    # Repeated generation is byte-identical, so a difference between targets
    # cannot be sampling noise.
    "determinism": ("n_repeats", "n_distinct_responses"),
}

ELIGIBILITY_REQUIRED_GATES = frozenset(ELIGIBILITY_GATE_EVIDENCE)


def validate_gate_entry(name: str, entry, *,
                        expected_revisions: dict | None = None) -> list[str]:
    """Violations for one detailed gate result.

    Requires the evidence fields, then checks that the evidence is
    internally consistent with the ``passed`` claim — a report asserting
    ``passed: true`` alongside ``n_failed: 3`` is self-contradictory and is
    rejected rather than trusted.
    """
    problems: list[str] = []
    if name not in ELIGIBILITY_GATE_EVIDENCE:
        # Fail closed rather than KeyError: an undefined gate has no
        # evidence contract, so nothing about it can be audited.
        return [f"gate {name!r} is not one of the required gates "
                f"{sorted(ELIGIBILITY_REQUIRED_GATES)}; it is undefined, so "
                f"it cannot confer authority"]
    if not isinstance(entry, dict):
        return [f"gate {name!r} must be an object carrying its evidence, "
                f"not {type(entry).__name__}; a bare flag is not auditable"]
    if entry.get("passed") is not True:
        problems.append(f"gate {name!r} did not pass "
                        f"(passed={entry.get('passed')!r})")
    required = ELIGIBILITY_GATE_EVIDENCE[name]
    missing = [f for f in required if entry.get(f) is None]
    if missing:
        problems.append(f"gate {name!r} is missing evidence field(s) "
                        f"{missing}")
        # The semantic checks below need those fields.
        return problems

    if name == "generations_complete":
        if entry["n_attempts"] != ELIGIBILITY_N_ATTEMPTS:
            problems.append(
                f"gate 'generations_complete' reports n_attempts="
                f"{entry['n_attempts']}, expected {ELIGIBILITY_N_ATTEMPTS}")
        if entry["n_succeeded"] != entry["n_attempts"]:
            problems.append(
                f"gate 'generations_complete' reports n_succeeded="
                f"{entry['n_succeeded']} of n_attempts="
                f"{entry['n_attempts']}")
        if entry["n_failed"] != 0:
            problems.append(
                f"gate 'generations_complete' reports n_failed="
                f"{entry['n_failed']} while claiming passed=true")
    elif name == "truncation_reviewed":
        if entry["n_truncated"] != 0:
            problems.append(
                f"gate 'truncation_reviewed' reports n_truncated="
                f"{entry['n_truncated']}; any truncation is a protocol-level "
                f"STOP (raising the cap would require a uniform five-model "
                f"replay including the frozen 9B reference), not an "
                f"eligibility warning")
        rate = entry["truncation_rate"]
        if not isinstance(rate, (int, float)) or not 0.0 <= rate <= 1.0:
            problems.append(
                f"gate 'truncation_reviewed' reports truncation_rate="
                f"{rate!r}, which is not a rate in [0, 1]")
        spread = entry["max_variant_spread"]
        if not isinstance(spread, (int, float)) or spread < 0:
            problems.append(
                f"gate 'truncation_reviewed' reports max_variant_spread="
                f"{spread!r}, which cannot be negative")
    elif name == "vision_path_engaged":
        if entry["n_image_bearing_cells"] <= 0:
            problems.append(
                "gate 'vision_path_engaged' reports no image-bearing cells, "
                "so the image path was never exercised")
        if entry["min_image_token_count"] <= 0:
            problems.append(
                f"gate 'vision_path_engaged' reports min_image_token_count="
                f"{entry['min_image_token_count']}; an image-bearing cell "
                f"with no image tokens means the image was silently dropped")
    elif name == "terminal_query_invariant":
        if entry["n_families_checked"] != ELIGIBILITY_N_FAMILIES:
            problems.append(
                f"gate 'terminal_query_invariant' checked "
                f"n_families_checked={entry['n_families_checked']}, expected "
                f"{ELIGIBILITY_N_FAMILIES}")
        if entry["n_mismatched"] != 0:
            problems.append(
                f"gate 'terminal_query_invariant' reports n_mismatched="
                f"{entry['n_mismatched']} while claiming passed=true")
    elif name == "revision_pinned":
        for field in ("model_revision", "processor_revision"):
            if not is_immutable_revision(entry[field]):
                problems.append(
                    f"gate 'revision_pinned' reports {field}="
                    f"{entry[field]!r}, which is not an immutable 40-hex SHA")
        for field, want in (expected_revisions or {}).items():
            if want is not None and entry.get(field) != want:
                problems.append(
                    f"gate 'revision_pinned' reports {field}="
                    f"{entry.get(field)!r} but this run resolves {want!r}")
    elif name == "determinism":
        if entry["n_repeats"] < 2:
            problems.append(
                f"gate 'determinism' used n_repeats={entry['n_repeats']}; "
                f"at least two repeats are needed to observe a difference")
        if entry["n_distinct_responses"] != 1:
            problems.append(
                f"gate 'determinism' observed n_distinct_responses="
                f"{entry['n_distinct_responses']} across "
                f"{entry['n_repeats']} repeats; greedy decoding must be "
                f"byte-identical")
    return problems


def validate_eligibility_report(
    report: dict,
    *,
    model_spec: ResolvedModel,
    expected_protocol_sha: str,
    expected_lock_sha: str | None,
    expected_processor_revision: str | None = None,
    expected_family_ids: Sequence[str] | None = None,
    expected_selection_sha256: str | None = None,
    panel_family_ids: set | None = None,
) -> list[str]:
    """Strict schema + content validation of an 11.5 eligibility report.

    Returns the list of violations (empty means valid). The earlier version
    of this gate recorded ``code_commit`` without requiring it, so a report
    with ``code_commit: null`` — or one written about a different
    ``model_key`` entirely — could pass. Every field below is now required
    AND checked against something OUTSIDE the report.

    That distinction is the point of the ``expected_*`` arguments. Checking a
    report's fields only against each other proves self-consistency, not
    correctness: a selection hash recomputed from the ids in the same report
    still passes when both are replaced together, and ``passed: true`` still
    passes when the check was never run. Each expectation below is therefore
    supplied by the caller from evidence the report does not control.

    Args:
        report: The parsed eligibility report.
        model_spec: The target this run resolves.
        expected_protocol_sha: Digest of the frozen protocol in force.
        expected_lock_sha: Digest of the dependency lock in force; the
            report must have been produced under the same one.
        expected_processor_revision: The processor revision this run
            resolves from the lock. The report's own value must equal it,
            not merely look like a SHA.
        expected_family_ids: The pre-registered 12-family selection, derived
            by the caller from frozen Iteration 10 evidence.
        expected_selection_sha256: Digest of that pre-registered selection.
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
    elif (expected_processor_revision is not None
          and report.get("processor_revision") != expected_processor_revision):
        # Looking like a SHA is not the same as being THE right one: a
        # report certified against a different processor revision would
        # authorize a run whose chat template or image processor differs,
        # which changes prompt rendering without moving the model revision.
        problems.append(
            f"eligibility was certified for processor_revision "
            f"{report.get('processor_revision')!r} but this run resolves "
            f"{expected_processor_revision!r} from the lock — eligibility "
            f"does not transfer across processor revisions")

    # --- code provenance of the eligibility run itself ------------------
    if not is_immutable_revision(report.get("code_commit")):
        problems.append(
            f"eligibility code_commit {report.get('code_commit')!r} is not "
            f"an immutable 40-hex SHA, so the code that certified "
            f"eligibility cannot be reconstructed")
    # The report is failed on the CODE subset, not on the whole tree. The
    # whole tree was already required clean at LAUNCH, so the process has
    # imported its code and code_commit pins it; a file appearing under
    # outputs/ afterwards cannot retroactively change what was generated,
    # while one appearing under src/ can, because Python imports lazily and
    # the determinism pass runs after the generations. Failing on the whole
    # tree here discarded a valid 72-generation run whose six gates all passed
    # because an unrelated diagnostic file appeared mid-run. git_dirty stays
    # required and recorded so the forgiven changes remain visible, and
    # git_dirty_code_paths is required so a report cannot omit the field that
    # decides the question.
    code_dirty = report.get("git_dirty_code_paths")
    if code_dirty is None:
        problems.append(
            "eligibility report does not record git_dirty_code_paths, so "
            "whether the code that produced it changed mid-run cannot be "
            "determined")
    elif code_dirty:
        shown = ", ".join(str(p) for p in code_dirty[:8])
        problems.append(
            f"eligibility report was produced while execution-relevant code "
            f"was dirty ({len(code_dirty)} path(s): {shown}"
            f"{' …' if len(code_dirty) > 8 else ''}); the code that generated "
            f"this evidence is not the code at code_commit "
            f"{str(report.get('code_commit'))[:12]}")

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
    # Internal consistency: the recorded digest must be the digest of the
    # recorded ids. On its own this proves nothing about WHICH families were
    # chosen — replacing both together used to pass — so it is only the
    # first of two checks.
    recomputed = selected_families_sha256(ids) if ids else None
    if ids and report.get("selected_families_sha256") != recomputed:
        problems.append(
            f"selected_families_sha256 "
            f"{report.get('selected_families_sha256')!r} does not match the "
            f"listed family ids (recomputed {recomputed!r})")
    # External pre-registration: the ids themselves must be the selection
    # derived from frozen, committed Iteration 10 evidence — which the
    # report does not control and cannot rewrite.
    if expected_family_ids is not None:
        want = sorted(expected_family_ids)
        got = sorted(ids)
        if got != want:
            extra = sorted(set(got) - set(want))
            absent = sorted(set(want) - set(got))
            problems.append(
                f"selected_family_ids are not the pre-registered 11.5 "
                f"selection: not in the selection {extra}, missing from the "
                f"report {absent}")
    if expected_selection_sha256 is not None and \
            report.get("selected_families_sha256") \
            != expected_selection_sha256:
        problems.append(
            f"selected_families_sha256 "
            f"{report.get('selected_families_sha256')!r} is not the "
            f"pre-registered selection digest "
            f"{expected_selection_sha256!r}")
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
    if not isinstance(gates, dict):
        problems.append(
            "gates must be an object of detailed per-gate results; a bare "
            "overall status is not auditable")
        return problems
    names = set(gates)
    omitted = sorted(ELIGIBILITY_REQUIRED_GATES - names)
    unexpected = sorted(names - ELIGIBILITY_REQUIRED_GATES)
    if omitted:
        problems.append(
            f"eligibility report omits required detailed gate(s) {omitted}; "
            f"an omitted gate means the check was never performed, and the "
            f"required set is exactly "
            f"{sorted(ELIGIBILITY_REQUIRED_GATES)}")
    if unexpected:
        problems.append(
            f"eligibility report carries unexpected gate(s) {unexpected}; "
            f"only {sorted(ELIGIBILITY_REQUIRED_GATES)} are defined, so an "
            f"extra name cannot confer authority")
    expected_revisions = {
        "model_revision": model_spec.revision,
        "processor_revision": expected_processor_revision,
    }
    for name in sorted(names & ELIGIBILITY_REQUIRED_GATES):
        problems.extend(validate_gate_entry(
            name, gates[name], expected_revisions=expected_revisions))
    return problems


def _check_selection_artifact(gate: _Gate, root: Path, derived: dict) -> None:
    """The committed selection artifact must agree with a fresh derivation.

    ``outputs/iteration_11/eligibility/selection.json`` is the human-readable
    audit trail for the pre-registered 12 families. It is NOT what the gate
    believes: the expectation comes from :func:`derive_frozen_selection`, so
    editing the artifact changes nothing about which families a run may
    replay. Comparing the two is what makes a stale or tampered artifact
    visible instead of silently authoritative.

    Only called when the gate derived the selection from ``root`` itself. An
    injected expectation describes some other evidence base, so holding this
    repository's artifact against it would be comparing unrelated things.
    """
    artifact_path = root / SELECTION_ARTIFACT
    if not artifact_path.exists():
        # Absent is not a violation — the derivation is the registration —
        # but it is recorded, because the artifact is what a reviewer reads.
        gate.record("selection_artifact_present", False)
        return
    gate.record("selection_artifact_present", True)
    try:
        committed = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        gate.fail(f"the committed 11.5 selection artifact at "
                  f"{artifact_path} is unreadable: {exc}")
        return
    committed_ids = sorted(committed.get("selected_family_ids") or [])
    if committed.get("selected_families_sha256") \
            != derived["selected_families_sha256"] \
            or committed_ids != sorted(derived["selected_family_ids"]):
        gate.fail(
            f"the committed 11.5 selection artifact at {artifact_path} does "
            f"not match a fresh derivation from the frozen evidence "
            f"(artifact sha256 "
            f"{committed.get('selected_families_sha256')!r} vs derived "
            f"{derived['selected_families_sha256']!r}); the selection is "
            f"pre-registered by derivation, not by that file. Regenerate it "
            f"with scripts/iter11_write_selection.py and review the change.")


def _check_eligibility(gate: _Gate, model_spec: ResolvedModel,
                       protocol_path: str | Path | None,
                       eligibility_root: str | Path | None,
                       lock_path: str | Path | None,
                       panel_family_ids: set | None = None,
                       expected_selection: dict | None = None,
                       repo_root: str | Path | None = None) -> None:
    """A PASSING, fully-specified 11.5 eligibility report for this target.

    Technical eligibility (the target can represent the same semantic role
    and image structure) is established at 11.5 and must precede the full
    run; without this check a model could be generated for and analysed
    despite having failed eligibility, which is exactly the selection
    freedom the frozen protocol removes.

    Nothing the report says is trusted on its own terms. The expected
    processor revision comes from the lock, and the expected family
    selection is RE-DERIVED here from frozen, committed Iteration 10
    evidence, so a report cannot authorize a different subset by recording
    matching ids and digest of its own choosing.
    """
    path = eligibility_report_path(model_spec.model_key, eligibility_root)
    gate.record("eligibility_report_path", str(path))
    expected_protocol_sha = protocol_sha256(protocol_path)
    gate.record("protocol_sha256", expected_protocol_sha)
    expected_lock_sha = dependency_lock_sha256(lock_path)
    gate.record("expected_dependency_lock_sha256", expected_lock_sha)

    lock = load_lock(Path(lock_path) if lock_path else DEFAULT_LOCK)
    entry = lock.get(model_spec.model_key)
    expected_processor_revision = (
        entry.get("processor_revision") if isinstance(entry, dict) else None)
    gate.record("expected_processor_revision", expected_processor_revision)

    expected = _resolve_expected_selection(gate, repo_root,
                                           expected_selection)
    expected_ids = list(expected["selected_family_ids"])
    expected_selection_sha = expected["selected_families_sha256"]

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
        expected_processor_revision=expected_processor_revision,
        expected_family_ids=expected_ids,
        expected_selection_sha256=expected_selection_sha,
        panel_family_ids=panel_family_ids)
    gate.record("eligibility_report_violations", problems)
    gate.record("eligibility_code_commit", report.get("code_commit"))
    gate.record("eligibility_git_dirty", report.get("git_dirty"))
    gate.record("eligibility_git_dirty_code_paths",
                report.get("git_dirty_code_paths"))
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
                 output_root: str | Path | None, model_key: str,
                 family_ids: Sequence[str] | None = None) -> None:
    """The run must cover the whole panel and land in the tracked tree.

    ``--max-families`` is a smoke facility. A run that replays a prefix of
    a 100-family panel still writes a complete-looking report, so it is
    rejected here rather than detected later by a missing-cell count.
    ``--output-root`` likewise: evidence written outside the Iteration 11
    generations tree is not where the analysis or the manifest looks for
    it, and may be gitignored.

    A named ``family_ids`` subset is rejected for the same reason as
    ``max_families``: it is the 11.5 eligibility facility, and a
    12-family run must not be labelled confirmatory. The subset a
    confirmatory run may replay is the whole panel.
    """
    gate.record("max_families", max_families)
    if max_families is not None:
        gate.fail(
            f"max_families={max_families}; a confirmatory run replays the "
            f"ENTIRE frozen panel. --max-families is a smoke facility and "
            f"a partial run must not be labelled confirmatory.")
    gate.record("family_ids", sorted({str(f) for f in family_ids})
                if family_ids else None)
    if family_ids is not None:
        gate.fail(
            f"family_ids names {len(set(family_ids))} families; a "
            f"confirmatory run replays the ENTIRE frozen panel. A named "
            f"subset is the 11.5 eligibility facility and its evidence is "
            f"not confirmatory — route it through "
            f"enforce_eligibility_protocol instead.")
    expected_root = f"{GENERATIONS_ROOT}/{model_key}"
    gate.record("output_root", str(output_root) if output_root else None)
    gate.record("expected_output_root", expected_root)
    if output_root is not None and \
            str(output_root).rstrip("/") != expected_root:
        gate.fail(
            f"output_root={str(output_root)!r} != {expected_root!r}; "
            f"confirmatory evidence must land in the tracked Iteration 11 "
            f"generations tree for this model_key")


def _resolve_expected_selection(gate: _Gate,
                                repo_root: str | Path | None,
                                expected_selection: dict | None) -> dict:
    """The pre-registered 11.5 selection, derived rather than trusted.

    Shared by the confirmatory gate (which compares a report against it)
    and the 11.5 gate (which compares the families about to be replayed
    against it), so both stages hold the same expectation from the same
    derivation. Fail-closed: an underivable selection raises instead of
    falling back to whatever a caller or a report claims.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    derived_here = expected_selection is None
    if derived_here:
        expected_selection = derive_frozen_selection(root)
    gate.record("expected_selection_sha256",
                expected_selection["selected_families_sha256"])
    gate.record("expected_selection_n_families",
                expected_selection.get("n_selected_families"))
    gate.record("selection_artifact_path", str(root / SELECTION_ARTIFACT))
    gate.record("selection_artifact_checked", derived_here)
    if derived_here:
        _check_selection_artifact(gate, root, expected_selection)
    return expected_selection


def _check_selection_scope(gate: _Gate,
                           family_ids: Sequence[str] | None,
                           expected_selection: dict) -> None:
    """An 11.5 run must replay EXACTLY the pre-registered 12 families.

    Not "at least" and not "a subset of": the selection exists so that
    which families were replayed is fixed before any target is measured, so
    replaying fewer (or others) is the selection freedom the frozen
    protocol removes.
    """
    want = sorted(str(f) for f in expected_selection["selected_family_ids"])
    got = sorted({str(f) for f in family_ids}) if family_ids else []
    gate.record("n_family_ids_requested", len(got))
    gate.record("family_ids_sha256",
                selected_families_sha256(got) if got else None)
    if not got:
        gate.fail(
            "an 11.5 eligibility run must name the families it replays; "
            f"the pre-registered selection is {want}")
        return
    if got != want:
        extra = sorted(set(got) - set(want))
        absent = sorted(set(want) - set(got))
        gate.fail(
            f"family_ids are not the pre-registered 11.5 selection: not in "
            f"the selection {extra}, missing from the run {absent}. "
            f"Eligibility is established on exactly the "
            f"{len(want)} families the frozen recipe selects, so that which "
            f"families were measured is fixed before any target is.")


def enforce_confirmatory_protocol(
    *,
    input_dir: str | Path,
    config: ReplayConfig,
    model_spec: ResolvedModel | None,
    overwrite: bool = False,
    max_families: int | None = None,
    family_ids: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    lock_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    eligibility_root: str | Path | None = None,
    expected_selection: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Enforce the frozen protocol on a confirmatory Iteration 11 run.

    Args:
        input_dir: Dataset directory holding validated_families.jsonl.
        config: The replay configuration about to be used.
        model_spec: The resolved target. Required — the legacy
            single-model path is out of scope and must not be gated here.
        overwrite: The ``--overwrite`` flag value.
        max_families: The ``--max-families`` value (must be None).
        family_ids: The ``family_ids`` subset value (must be None). A named
            subset is the 11.5 eligibility facility; a confirmatory run
            replays the whole panel and its subset is that panel.
        output_root: The resolved output root for this run.
        lock_path: Lock file supplying revisions and the dependency lock.
        protocol_path: Frozen protocol (override for tests).
        eligibility_root: 11.5 report root (override for tests).
        expected_selection: The pre-registered 11.5 family selection. By
            default it is RE-DERIVED from the frozen evidence under
            ``repo_root``; supplying it is for tests and for callers that
            have already derived it. It is never taken from the report.
            The committed selection artifact is cross-checked against the
            derivation ONLY when the gate derived it here, since an injected
            expectation may describe a different evidence base.
        repo_root: Repository root used to locate the frozen evidence and
            the committed selection artifact.

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
    _check_scope(gate, max_families, output_root, model_spec.model_key,
                 family_ids)
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
                       lock_path, panel_family_ids, expected_selection,
                       repo_root)

    return _finalize_gate(gate, "iteration11_confirmatory",
                          model_spec.model_key)


def enforce_eligibility_protocol(
    *,
    input_dir: str | Path,
    config: ReplayConfig,
    model_spec: ResolvedModel | None,
    family_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    max_families: int | None = None,
    output_root: str | Path | None = None,
    lock_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    expected_selection: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Enforce the frozen protocol on an Iteration 11.5 eligibility run.

    The 11.5 run is NOT confirmatory — it replays 12 families, not 100, and
    it is what produces the eligibility report the confirmatory gate later
    requires — but every dimension the frozen protocol fixes is enforced
    identically here, using the same checks. Eligibility evidence generated
    under a different cap, decoding, prompt or panel would certify a target
    against conditions the confirmatory run does not share.

    Two things differ from :func:`enforce_confirmatory_protocol`, both by
    necessity:

    * the run MUST name a family subset, and it must be exactly the
      pre-registered 12 — derived here from the frozen Iteration 10
      evidence, never taken from the caller's word alone;
    * no eligibility report is required, because this stage writes it.

    Args:
        input_dir: The FULL frozen panel directory. The subset is applied
            by the runner; reading the whole panel is what binds the frozen
            panel digest into the run fingerprint.
        config: The replay configuration about to be used.
        model_spec: The resolved target. Required.
        family_ids: The families about to be replayed. Must equal the
            pre-registered selection.
        overwrite: The ``--overwrite`` flag value.
        max_families: The ``--max-families`` value (must be None).
        output_root: The resolved output root for this run.
        lock_path: Lock file supplying revisions and the dependency lock.
        protocol_path: Frozen protocol (override for tests).
        expected_selection: Injected pre-registered selection (tests, or a
            caller that has already derived it). Derived when None.
        repo_root: Repository root used to locate the frozen evidence.

    Returns:
        The gate evidence: every value checked. Callers should persist it.

    Raises:
        ReplayError: listing ALL violations found.
    """
    if model_spec is None:
        raise ReplayError(
            "the 11.5 eligibility gate requires a resolved Iteration 11 "
            "target (model_spec); the frozen legacy single-model path is "
            "out of scope and must not be routed through this gate")
    protocol = load_protocol(protocol_path)
    gate = _Gate()
    gate.record("model_key", model_spec.model_key)
    gate.record("model_id", model_spec.model_id)
    gate.record("adapter", model_spec.adapter)

    _check_no_overwrite(gate, overwrite)
    if max_families is not None:
        gate.record("max_families", max_families)
        gate.fail(
            f"max_families={max_families}; an 11.5 run replays exactly the "
            f"pre-registered selection, and --max-families would replay a "
            f"prefix of it while the gate certified the whole selection.")
    expected_root = f"{ELIGIBILITY_GENERATIONS_ROOT}/{model_spec.model_key}"
    gate.record("output_root", str(output_root) if output_root else None)
    gate.record("expected_output_root", expected_root)
    if output_root is not None and \
            str(output_root).rstrip("/") != expected_root:
        gate.fail(
            f"output_root={str(output_root)!r} != {expected_root!r}; 11.5 "
            f"evidence must land in the eligibility generations tree for "
            f"this model_key, kept separate from confirmatory evidence so "
            f"a 12-family run can never be read as a 100-family one.")

    _check_clean_tree(gate, ELIGIBILITY_OWN_OUTPUT_PREFIXES)
    _check_decoding(gate, config, protocol)
    _check_revisions(gate, model_spec, lock_path)
    _check_dependencies(gate, lock_path)

    # The whole frozen panel is read and checked, exactly as confirmatory
    # does: the subset is drawn FROM it, so a panel that is not the frozen
    # one would make the selection meaningless even if the 12 ids matched.
    panel = _check_panel_identity(gate, Path(input_dir), protocol)
    if panel is not None:
        records = [
            json.loads(line)
            for line in Path(panel).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _check_family_coverage(gate, records, protocol)
        absent = sorted(set(str(f) for f in (family_ids or []))
                        - {r.get("family_id") for r in records})
        if absent:
            gate.fail(
                f"family_ids {absent} are not in the frozen panel at "
                f"{panel}; the pre-registered selection is drawn from that "
                f"panel, so an id outside it cannot be part of it")

    expected = _resolve_expected_selection(gate, repo_root,
                                           expected_selection)
    _check_selection_scope(gate, family_ids, expected)

    return _finalize_gate(gate, "iteration11_eligibility",
                          model_spec.model_key)


def _finalize_gate(gate: _Gate, name: str, model_key: str) -> dict:
    """The gate evidence, or a ReplayError listing EVERY violation.

    Collecting all of them rather than raising on the first is deliberate:
    one launch surfaces every problem, so an operator fixing a misconfigured
    run does not discover the next violation only after fixing this one.
    """
    result = {
        "gate": name,
        "passed": not gate.violations,
        "n_violations": len(gate.violations),
        "violations": gate.violations,
        "checks": gate.checks,
    }
    if gate.violations:
        listing = "\n".join(f"  - {v}" for v in gate.violations)
        raise ReplayError(
            f"{name} gate FAILED for {model_key} with "
            f"{len(gate.violations)} violation(s):\n{listing}\n"
            f"No output was generated. The frozen protocol fixes these "
            f"dimensions; a run that departs from any of them is not "
            f"comparable to the Qwen3.5-9B reference.")
    return result
