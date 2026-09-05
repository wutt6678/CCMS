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
from typing import Any

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.errors import ReplayError
from causal_mllm.replay.registry import (
    DEFAULT_LOCK,
    ResolvedModel,
    is_immutable_revision,
    load_lock,
    verify_active_dependency_lock,
)
from causal_mllm.seeds import get_git_commit, is_git_dirty
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
    """The tree must be clean, so ``code_commit`` identifies the code run.

    ``is_git_dirty()`` ignores untracked files, so the run's own outputs do
    not trip this. It returns None outside a git repository, which is
    treated as a violation: unknown provenance cannot be certified, and
    accepting it would make a source tarball indistinguishable from a
    committed tree.
    """
    commit = get_git_commit()
    dirty = is_git_dirty()
    gate.record("code_commit", commit)
    gate.record("git_dirty", dirty)
    if commit is None:
        gate.fail("no git commit resolvable — code provenance is unknown, so "
                  "this run cannot be reconstructed from evidence")
    if dirty is None:
        gate.fail("git tree status unknown (not a git repository?) — "
                  "confirmatory runs require a verified clean tree")
    elif dirty is True:
        gate.fail(
            f"working tree is dirty at commit {commit}: the code that would "
            f"execute is NOT the code that commit contains, so the recorded "
            f"code_commit could not reconstruct this run. Commit or stash "
            f"first.")


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
    proves nothing about the live interpreter. This compares the two.
    """
    check = verify_active_dependency_lock(lock_path, strict=False)
    gate.record("dependency_lock_check", check)
    if not check.get("verified"):
        differences = check.get("differences") or {}
        reason = check.get("reason")
        if reason:
            gate.fail(f"dependency environment unverified: {reason}")
        else:
            detail = "; ".join(
                f"{f}: locked={v['locked']!r} active={v['active']!r}"
                for f, v in sorted(differences.items()))
            gate.fail(
                f"active dependency environment differs from the recorded "
                f"lock — {detail}")


def _check_eligibility(gate: _Gate, model_spec: ResolvedModel,
                       protocol_path: str | Path | None,
                       eligibility_root: str | Path | None) -> None:
    """A PASSING 11.5 eligibility report for this revision and protocol.

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
    report = load_eligibility_report(model_spec.model_key, eligibility_root)
    if report is None:
        gate.fail(
            f"{model_spec.model_key}: no 11.5 eligibility report at {path}; "
            f"a confirmatory run may only start once technical eligibility "
            f"has been signed off for this target")
        return
    gate.record("eligibility_status", report.get("status"))
    gate.record("eligibility_eligible", report.get("eligible"))
    gate.record("eligibility_model_revision", report.get("model_revision"))
    gate.record("eligibility_protocol_sha256",
                report.get("protocol_sha256"))
    gate.record("eligibility_code_commit", report.get("code_commit"))
    gate.record("eligibility_git_dirty", report.get("git_dirty"))
    if report.get("status") != "PASS":
        gate.fail(
            f"{model_spec.model_key}: eligibility report status is "
            f"{report.get('status')!r}, not 'PASS'")
    if report.get("eligible") is not True:
        gate.fail(
            f"{model_spec.model_key}: eligibility report does not assert "
            f"eligible=true (got {report.get('eligible')!r})")
    if report.get("model_revision") != model_spec.revision:
        gate.fail(
            f"{model_spec.model_key}: eligibility was certified for "
            f"revision {report.get('model_revision')!r} but this run "
            f"resolves {model_spec.revision!r} — eligibility does not "
            f"transfer across revisions")
    if report.get("protocol_sha256") != expected_protocol_sha:
        gate.fail(
            f"{model_spec.model_key}: eligibility report binds protocol "
            f"sha256 {report.get('protocol_sha256')!r}, not the current "
            f"frozen protocol {expected_protocol_sha!r}")
    if report.get("git_dirty") is not False:
        gate.fail(
            f"{model_spec.model_key}: eligibility report was not produced "
            f"from a clean tree (git_dirty="
            f"{report.get('git_dirty')!r})")


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
    _check_eligibility(gate, model_spec, protocol_path, eligibility_root)

    input_path = Path(input_dir)
    panel = _check_panel_identity(gate, input_path, protocol)
    if panel is not None:
        records = [
            json.loads(line)
            for line in Path(panel).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _check_family_coverage(gate, records, protocol)

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
