#!/usr/bin/env python3
"""Iteration 11.6 completion gate — verify a confirmatory run after the fact.

``enforce_confirmatory_protocol`` decides whether a run MAY start. This decides
whether what it produced IS the evidence the frozen protocol asked for. Those
are different questions, and only the second one can see the outputs: a run can
pass every pre-condition and still come back short of cells, truncated on one
variant, or with a processor that quietly reordered a family's images.

Mirrors ``scripts/scale_c_replay_checks.py`` — the Iteration 10 analogue — with
the differences the cross-model setting forces:

  * the expected revision and processor revision come from the target's entry in
    ``resolved_models.lock.yaml``, not from one frozen 9B constant, and the
    system prompt is required to be the SAME across targets because the
    protocol's whole claim is one shared pipeline with thin adapters;
  * a run must cover the ENTIRE 100-family panel and must NOT be a subset.
    ``provenance.family_subset`` and the fingerprint's
    ``family_subset_sha256`` are both required to be null, so a 12-family 11.5
    eligibility journal can never be passed off as confirmatory evidence;
  * the recorded ``confirmatory_gate`` evidence must be present and passed, so
    the pre-run verdict is auditable from the artifact rather than from a
    stdout line nobody kept;
  * the 11.5 eligibility report is RE-VALIDATED here, at completion time,
    against the current frozen protocol, lock and derived selection. Trusting
    the pre-run gate alone would not notice an eligibility report replaced
    after generation.

Reads by default, and says so. This gate used to write
``iteration_11_replay_checks.json`` into every run directory on every
invocation -- including the one the README listed under "Re-running any of it,
read-only", so the documented verification command rewrote four committed PASS
reports. On a checkout that did not hold the gitignored media it rewrote them as
FAIL, which is a verdict about the machine rather than about the panel.
Generation is now behind ``--write-report``, and ``--verify`` compares without
writing.

The media check is bound to ``outputs/iteration_11/media_manifest.json`` (written
by ``scripts/iter11_write_media_manifest.py``) so that an image missing from THIS
checkout is reported as unverifiable-here while an image that is present and
differs from its bound digest is a failure. A run whose media could not be hashed
here takes the verdict ``PASS_WITH_UNVERIFIED_SECTIONS`` and exit code 3 -- never
a bare PASS, because a check that could not run is not a check that passed.

Usage:
    python3 scripts/iter11_replay_checks.py --all --verify        # compare, write nothing
    python3 scripts/iter11_replay_checks.py --model-key qwen35_2b # check, write nothing
    python3 scripts/iter11_replay_checks.py --model-key phi4_mm RUN_DIR
    python3 scripts/iter11_replay_checks.py --all --write-report  # regenerate evidence

Exit codes: 0 every check passed and every section was verifiable here; 1 a check
failed, or ``--verify`` found the committed report no longer reproduces; 2 there
is no committed report to verify against; 3 some section could not be verified in
this checkout, so the answer is incomplete rather than clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES  # noqa: E402
from causal_mllm.replay.confirmatory import (  # noqa: E402
    ELIGIBILITY_ROOT,
    eligibility_report_path,
    protocol_sha256,
    validate_eligibility_report,
)
from causal_mllm.replay.registry import (  # noqa: E402
    dependency_lock_sha256,
    is_immutable_revision,
    load_lock,
    resolve_model,
)
from causal_mllm.replay.selection import derive_frozen_selection  # noqa: E402
from causal_mllm.replay.truncation import (  # noqa: E402
    MAX_TRUNCATION_RATE,
    MAX_VARIANT_SPREAD,
    is_truncated,
)

PROTOCOL_PATH = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"
PANEL_PATH = REPO_ROOT / "outputs" / "scale_c" / "families_panel" \
    / "validated_families.jsonl"
DEFAULT_LOCK = REPO_ROOT / "outputs" / "iteration_11" / "preflight" \
    / "resolved_models.lock.yaml"
PREFLIGHT_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "preflight"
GENERATIONS_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "generations"

CHECKS_FILE = "iteration_11_replay_checks.json"
OUTPUTS_FILE = "replay_outputs.jsonl"
REPORT_FILE = "replay_report.json"

#: Committed identity of ``data/media``, which is gitignored apart from 20
#: individually negated source images. Without it the media check can only ask
#: "is this file on this machine", and a fresh checkout answers no for 3,014 of
#: 3,034 -- which used to be reported as 3,014 failures of the PANEL.
MEDIA_MANIFEST_PATH = REPO_ROOT / "outputs" / "iteration_11" \
    / "media_manifest.json"

#: Sections whose evidence lives outside the repository and so may be
#: unverifiable in a given checkout. Reported, never silently dropped.
MEDIA_SECTION = "media"


def load_media_manifest(path: Path = MEDIA_MANIFEST_PATH) -> dict | None:
    """The bound media identity, or None when there is none to check against.

    A missing manifest is reported by the caller rather than treated as "no
    media to verify": the difference is between having no expectation and having
    an expectation this checkout cannot meet.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

#: The truncation tolerance is NOT defined here. It lives in
#: ``causal_mllm.replay.truncation`` and the evaluation panel gate imports the
#: same two constants, so one standard governs a panel from the moment it is
#: replayed to the moment it is evaluated. This file used to carry its own
#: copy of 0.02/0.05 while ``evaluation/gate.py`` required zero; the frozen 9B
#: reference truncated nothing so the two never disagreed, and Iteration 11's
#: smaller checkpoints were the first panels to expose the contradiction.
#: "Near-zero" overall, and no material difference BETWEEN variants —
#: condition-dependent truncation is the failure that would let one variant's
#: responses be systematically cut short.

EXPECTED_N_FAMILIES = 100


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def panel_canonical_queries() -> dict[str, str]:
    """family_id -> the canonical terminal query every variant must carry.

    Same definition as ``scale_c_replay_checks.py`` and the 11.5 producer:
    prefer the harmonized ``canonical_q``, fall back to the terminal query's
    text, then to the bare string.
    """
    queries: dict[str, str] = {}
    for line in PANEL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        family = json.loads(line)
        harmonization = (family.get("validation") or {}) \
            .get("terminal_harmonization") or {}
        text = harmonization.get("canonical_q")
        if not text:
            terminal = family["terminal_query"]
            text = terminal["text"] if isinstance(terminal, dict) else terminal
        queries[family["family_id"]] = text
    return queries


def panel_families() -> dict[str, dict]:
    families = {}
    for line in PANEL_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            family = json.loads(line)
            families[family["family_id"]] = family
    return families


def display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    ``relative_to`` raises on a path outside the repo, and a verification tool
    that crashes on the argument it was given reports nothing at all.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def find_run_dir(model_key: str) -> Path:
    """The single confirmatory run directory for ``model_key``."""
    root = GENERATIONS_ROOT / model_key
    if not root.exists():
        raise SystemExit(f"no 11.6 generations for {model_key}: {root} absent")
    candidates = sorted(p for p in root.iterdir()
                        if p.is_dir() and (p / OUTPUTS_FILE).exists())
    if not candidates:
        raise SystemExit(f"{root} holds no run directory with {OUTPUTS_FILE}")
    if len(candidates) > 1:
        raise SystemExit(
            f"{root} holds {len(candidates)} runs "
            f"({[p.name for p in candidates]}); name the intended one "
            f"explicitly rather than letting this script choose which "
            f"evidence to certify")
    return candidates[0]


def media_section(by_family: dict[str, list[dict]],
                  families: dict[str, dict],
                  manifest: dict | None,
                  repo_root: Path = REPO_ROOT,
                  manifest_path: Path = MEDIA_MANIFEST_PATH
                  ) -> tuple[dict, list[str]]:
    """Media identity, bound against the committed manifest.

    Returns ``(block, notes)``. Three different situations used to produce one
    message, and conflating them is what made a checkout limitation look like a
    finding about the panel:

    * an image PRESENT AND DIFFERENT from its bound digest is a finding about
      the evidence, and fails;
    * an image the panel references that the manifest never bound is a finding
      about the manifest, and fails;
    * an image simply NOT ON THIS MACHINE is neither. ``data/media`` is
      gitignored apart from 20 individually negated source images, so a fresh
      checkout holds 20 of 3,034 and cannot hash the panel's 100. Reporting
      that as a failure is how four committed PASS reports became FAIL on a
      machine that had done nothing to the panel. It is recorded as
      ``verifiable_here: false`` instead, which makes the run's verdict
      ``PASS_WITH_UNVERIFIED_SECTIONS`` -- not PASS, because a check that could
      not run is not a check that passed, and not FAIL, because nothing was
      found wrong.

    ``n_images`` against the panel definition does not need the bytes, so it is
    checked even when the media are absent.
    """
    media_issues: list[str] = []
    bound = (manifest or {}).get("files") or {}
    referenced: set[str] = set()
    hashed: set[str] = set()
    absent: set[str] = set()
    unbound: set[str] = set()
    unhashable_families: list[str] = []
    where = display_path(manifest_path)
    for family_id, family in sorted(families.items()):
        records = by_family.get(family_id)
        if not records:
            continue
        expected_counts, shared = {}, set()
        family_paths: set[str] = set()
        for variant_name, variant in (family.get("variants") or {}).items():
            paths = [p for message in variant.get("messages") or []
                     for p in (message.get("images") or [])]
            expected_counts[variant_name] = len(paths)
            for path in paths:
                referenced.add(path)
                family_paths.add(path)
                resolved = repo_root / path
                if not resolved.exists():
                    absent.add(path)
                    continue
                digest = sha256_file(resolved)
                hashed.add(path)
                shared.add(digest)
                if path not in bound:
                    media_issues.append(
                        f"{family_id}/{variant_name}: image {path} is present "
                        f"but {where} does not bind it, so nothing says "
                        f"whether these are the bytes the panel was generated "
                        f"from")
                    unbound.add(path)
                elif bound[path]["sha256"] != digest:
                    media_issues.append(
                        f"{family_id}/{variant_name}: image {path} is {digest} "
                        f"but the committed media manifest binds "
                        f"{bound[path]['sha256']}")
        if family_paths and not (family_paths & hashed):
            # None of this family's images could be hashed here, so the
            # byte-identity check below never ran for it. Recording that is the
            # difference between "checked and identical" and "never checked".
            unhashable_families.append(family_id)
        for record in records:
            want = expected_counts.get(record["variant"])
            if record.get("n_images") != want:
                media_issues.append(
                    f"{family_id}/{record['variant']}: n_images "
                    f"{record.get('n_images')} != {want}")
        if len(shared) > 1:
            media_issues.append(f"{family_id}: {len(shared)} distinct image "
                                f"hashes across variants (shared images must "
                                f"be byte-identical)")
    if manifest is None:
        media_issues.append(
            f"no media manifest at {where}, so the images this panel was "
            f"generated from have no committed identity to be checked against; "
            f"run scripts/iter11_write_media_manifest.py")
    verifiable = not absent and manifest is not None
    block = {
        "issues": media_issues[:40],
        "n_issues": len(media_issues),
        "ok": not media_issues,
        "manifest_path": where if manifest else None,
        "manifest_rollup_sha256": (manifest or {}).get("rollup_sha256"),
        "manifest_scope": (manifest or {}).get("scope"),
        "n_referenced_images": len(referenced),
        "n_hashed_here": len(hashed),
        "n_absent_here": len(absent),
        "absent_images": sorted(absent)[:10],
        "n_referenced_but_unbound": len(unbound),
        "families_whose_images_could_not_be_hashed_here":
            unhashable_families[:10],
        "n_families_whose_images_could_not_be_hashed_here":
            len(unhashable_families),
        "verifiable_here": verifiable,
    }
    notes = []
    if not verifiable and not media_issues:
        notes.append(
            f"media: {len(absent)} of {len(referenced)} referenced image(s) "
            f"are absent from this checkout and {len(unhashable_families)} "
            f"family(ies) could not be image-checked here, so that section is "
            f"UNVERIFIED rather than passed; the committed manifest binds "
            f"{(manifest or {}).get('n_files')} file(s) with roll-up "
            f"{(manifest or {}).get('rollup_sha256')}")
    return block, notes


def verdict_for(result: dict) -> tuple[str, list[str]]:
    """The run's verdict, and the sections that could not be checked here.

    Three answers, kept apart because they are three different facts: the panel
    is wrong (``FAIL``); the panel is right and every section was checked
    (``PASS``); and nothing was wrong but part of it could not be checked on
    THIS machine (``PASS_WITH_UNVERIFIED_SECTIONS``). Collapsing the third into
    the second is how a checkout limitation becomes a certification, and
    collapsing it into the first is how four committed PASS reports became FAIL
    on a machine that had done nothing to the panel.

    Any section may opt in by recording ``verifiable_here: false``, so this does
    not hard-code the media.
    """
    unverifiable = sorted(
        name for name, block in result.items()
        if isinstance(block, dict) and block.get("verifiable_here") is False)
    if result.get("failures"):
        return "FAIL", unverifiable
    if unverifiable:
        return "PASS_WITH_UNVERIFIED_SECTIONS", unverifiable
    return "PASS", unverifiable


def check(model_key: str, run_dir: Path, lock_path: Path) -> dict:
    """Run every completion check for one target; return the verdict dict."""
    failures: list[str] = []
    warnings: list[str] = []
    result: dict = {"model_key": model_key, "run_dir": display_path(run_dir),
                    "lock_path": display_path(lock_path)}

    outputs_path, report_path = run_dir / OUTPUTS_FILE, run_dir / REPORT_FILE
    for required in (outputs_path, report_path):
        if not required.exists():
            raise SystemExit(f"{required} absent; cannot verify {model_key}")
    outputs = read_jsonl(outputs_path)
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    prov = run_report.get("provenance") or {}

    lock = load_lock(lock_path)
    if model_key not in lock:
        raise SystemExit(f"{model_key} is not in {lock_path.name}")
    locked = lock[model_key]
    locked_revision = locked.get("revision")
    locked_processor = locked.get("processor_revision")

    # ---- coverage -------------------------------------------------------
    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in outputs:
        by_family[record["family_id"]].append(record)
    issues = []
    if len(by_family) != EXPECTED_N_FAMILIES:
        issues.append(f"expected {EXPECTED_N_FAMILIES} families, found "
                      f"{len(by_family)} — a confirmatory run replays the "
                      f"ENTIRE frozen panel")
    pairs = {(r["family_id"], r["variant"]) for r in outputs}
    if len(pairs) != len(outputs):
        issues.append(f"{len(outputs) - len(pairs)} duplicate "
                      f"(family_id, variant) record(s)")
    for family_id, records in sorted(by_family.items()):
        got = sorted(r["variant"] for r in records)
        if got != sorted(ALL_VARIANT_NAMES):
            issues.append(f"{family_id}: variants {got} != "
                          f"{sorted(ALL_VARIANT_NAMES)}")
    errors = [r for r in outputs if r.get("error")]
    if errors:
        issues.append(f"{len(errors)} record(s) carry an error")
    expected_attempts = EXPECTED_N_FAMILIES * len(ALL_VARIANT_NAMES)
    if run_report.get("expected_attempts") != expected_attempts:
        issues.append(f"run report expected_attempts="
                      f"{run_report.get('expected_attempts')} != "
                      f"{expected_attempts}")
    if run_report.get("n_succeeded") != expected_attempts:
        issues.append(f"n_succeeded={run_report.get('n_succeeded')} != "
                      f"{expected_attempts}")
    if run_report.get("n_failed"):
        issues.append(f"n_failed={run_report.get('n_failed')}")
    if run_report.get("missing_variants"):
        issues.append(f"missing_variants="
                      f"{run_report.get('missing_variants')}")
    if run_report.get("failed_cells"):
        issues.append(f"failed_cells={run_report.get('failed_cells')}")
    result["coverage"] = {
        "n_families": len(by_family), "n_records": len(outputs),
        "expected_attempts": expected_attempts,
        "n_succeeded": run_report.get("n_succeeded"),
        "n_error_records": len(errors), "issues": issues, "ok": not issues}
    failures.extend(f"coverage: {i}" for i in issues)

    # ---- not a subset ---------------------------------------------------
    # A 12-family eligibility journal is a SCREENING artifact. If one were
    # passed off as confirmatory evidence the coverage check above would
    # usually catch it, but the subset markers are checked explicitly so the
    # refusal names the actual problem.
    subset_issues = []
    if prov.get("family_subset") is not None:
        subset_issues.append(
            f"provenance.family_subset is set "
            f"({len((prov.get('family_subset') or {}).get('family_ids') or [])}"
            f" families); a confirmatory run replays the whole panel, so this "
            f"is a screening run")
    fingerprint = prov.get("resolved_run_fingerprint")
    if isinstance(fingerprint, dict) \
            and fingerprint.get("family_subset_sha256") is not None:
        subset_issues.append(
            "the resolved run fingerprint binds a family_subset_sha256, so "
            "this run was keyed to a subset")
    result["full_panel"] = {"issues": subset_issues,
                            "ok": not subset_issues}
    failures.extend(f"full_panel: {i}" for i in subset_issues)

    # ---- provenance against the lock ------------------------------------
    prov_issues = []
    for field, expected in (("requested_model_revision", locked_revision),
                            ("resolved_model_revision", locked_revision),
                            ("model_revision", locked_revision),
                            ("processor_revision", locked_processor)):
        if prov.get(field) != expected:
            prov_issues.append(f"{field}={str(prov.get(field))[:16]!r} != "
                               f"locked {str(expected)[:16]!r}")
    if prov.get("revision_pinned") is not True:
        prov_issues.append("revision_pinned is not True")
    if not is_immutable_revision(prov.get("code_commit")):
        prov_issues.append(f"code_commit {str(prov.get('code_commit'))!r} is "
                           f"not an immutable 40-hex SHA")
    if prov.get("git_dirty") is not False:
        prov_issues.append(f"git_dirty={prov.get('git_dirty')!r}")
    # Code-scoped, matching what the eligibility report is failed on: the
    # whole-tree answer is recorded for honesty, the code subset decides.
    code_dirty = prov.get("code_dirty_paths")
    if code_dirty is None:
        prov_issues.append("provenance does not record code_dirty_paths")
    elif code_dirty:
        prov_issues.append(f"execution-relevant code was dirty during the run "
                           f"({len(code_dirty)} path(s): "
                           f"{', '.join(map(str, code_dirty[:6]))})")
    if not prov.get("resolved_sha256") and not isinstance(fingerprint, str):
        prov_issues.append("no resolved fingerprint recorded")

    # The system prompt must be the target's 11.2 preflight value AND the same
    # across every target: one shared pipeline with thin adapters is the
    # protocol's central claim, and a target-specific prompt is a
    # stop-condition.
    prompt_sha = prov.get("system_prompt_sha256")
    preflight_path = PREFLIGHT_ROOT / model_key / "preflight.json"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        expected_prompt = preflight.get("system_prompt_sha256")
        if prompt_sha != expected_prompt:
            prov_issues.append(f"system_prompt_sha256 {str(prompt_sha)[:16]!r}"
                               f" != the 11.2 preflight's "
                               f"{str(expected_prompt)[:16]!r}")
    else:
        prov_issues.append(f"no 11.2 preflight artifact at {preflight_path}")
    if prov.get("prompt_template_revision") != "v1":
        prov_issues.append(f"prompt_template_revision="
                           f"{prov.get('prompt_template_revision')!r} != 'v1'")

    per_record = 0
    for record in outputs:
        if (record.get("resolved_model_revision") != locked_revision
                or record.get("system_prompt_sha256") != prompt_sha
                or record.get("revision_pinned") is not True):
            per_record += 1
    if per_record:
        prov_issues.append(f"{per_record} record(s) whose revision/prompt "
                           f"fingerprints disagree with the run")
    result["provenance"] = {
        "locked_revision": locked_revision,
        "locked_processor_revision": locked_processor,
        "system_prompt_sha256": prompt_sha,
        "code_commit": prov.get("code_commit"),
        "issues": prov_issues, "ok": not prov_issues}
    failures.extend(f"provenance: {i}" for i in prov_issues)

    # ---- the pre-run gate must be recorded, and must have passed --------
    gate = run_report.get("confirmatory_gate")
    gate_issues = []
    if not isinstance(gate, dict):
        gate_issues.append(
            "the run report carries no confirmatory_gate evidence, so the "
            "pre-run verdict is not auditable from the artifact")
    else:
        if gate.get("passed") is not True:
            gate_issues.append(f"confirmatory_gate.passed="
                               f"{gate.get('passed')!r}")
        if gate.get("gate") != "iteration11_confirmatory":
            gate_issues.append(f"gate name {gate.get('gate')!r} is not "
                               f"'iteration11_confirmatory' — an eligibility "
                               f"gate cannot authorize confirmatory evidence")
        violations = (gate.get("violations") or [])
        if violations:
            gate_issues.append(f"{len(violations)} recorded violation(s)")
    result["confirmatory_gate"] = {
        "present": isinstance(gate, dict),
        "passed": (gate or {}).get("passed"),
        "n_checks": len((gate or {}).get("checks") or {}),
        "issues": gate_issues, "ok": not gate_issues}
    failures.extend(f"confirmatory_gate: {i}" for i in gate_issues)

    # ---- re-validate the 11.5 eligibility report at completion ----------
    elig_issues = []
    elig_path = eligibility_report_path(model_key, ELIGIBILITY_ROOT)
    if not elig_path.exists():
        elig_issues.append(f"no 11.5 eligibility report at {elig_path}")
    else:
        elig = json.loads(elig_path.read_text(encoding="utf-8"))
        if elig.get("status") != "PASS" or elig.get("eligible") is not True:
            elig_issues.append(f"eligibility status={elig.get('status')!r} "
                               f"eligible={elig.get('eligible')!r}")
        selection = derive_frozen_selection(REPO_ROOT)
        panel_ids = set(panel_families())
        try:
            spec = resolve_model(model_key, confirmatory=True,
                                 lock_path=lock_path)
        except Exception as exc:
            spec = None
            elig_issues.append(f"cannot resolve {model_key}: "
                               f"{type(exc).__name__}: {exc}")
        if spec is not None:
            violations = validate_eligibility_report(
                elig, model_spec=spec,
                expected_protocol_sha=protocol_sha256(PROTOCOL_PATH),
                expected_lock_sha=dependency_lock_sha256(lock_path),
                expected_processor_revision=locked_processor,
                expected_family_ids=selection["selected_family_ids"],
                expected_selection_sha256=selection["selected_families_sha256"],
                panel_family_ids=panel_ids)
            elig_issues.extend(violations)
    result["eligibility"] = {
        "report": display_path(elig_path) if elig_path.exists() else None,
        "issues": elig_issues, "ok": not elig_issues}
    failures.extend(f"eligibility: {i}" for i in elig_issues)

    # ---- terminal-query invariance --------------------------------------
    canonical = panel_canonical_queries()
    term_issues = []
    for family_id, records in sorted(by_family.items()):
        shas = {r.get("terminal_sha256") for r in records}
        if len(shas) != 1:
            term_issues.append(f"{family_id}: {len(shas)} distinct terminal "
                               f"hashes across its variants")
            continue
        expected_text = canonical.get(family_id)
        if expected_text is None:
            term_issues.append(f"{family_id}: not in the frozen panel")
            continue
        expected = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
        if next(iter(shas)) != expected:
            term_issues.append(f"{family_id}: terminal hash != the panel's "
                               f"canonical q*")
    result["terminal_query_equality"] = {
        "n_families_checked": len(by_family), "issues": term_issues,
        "ok": not term_issues}
    failures.extend(f"terminal: {i}" for i in term_issues)

    # ---- media identity -------------------------------------------------
    result[MEDIA_SECTION], media_notes = media_section(
        by_family, panel_families(), load_media_manifest())
    failures.extend(f"media: {i}" for i in result[MEDIA_SECTION]["issues"])
    warnings.extend(media_notes)

    # ---- truncation -----------------------------------------------------
    truncated_by_variant: dict[str, int] = defaultdict(int)
    n_by_variant: dict[str, int] = defaultdict(int)
    tokens_by_variant: dict[str, list[int]] = defaultdict(list)
    for record in outputs:
        variant = record["variant"]
        n_by_variant[variant] += 1
        tokens_by_variant[variant].append(
            record.get("output_token_count") or 0)
        if is_truncated(record):
            truncated_by_variant[variant] += 1
    rates = {v: truncated_by_variant[v] / n_by_variant[v]
             for v in n_by_variant if n_by_variant[v]}
    n_records = max(1, len(outputs))
    overall = sum(truncated_by_variant.values()) / n_records
    spread = (max(rates.values()) - min(rates.values())) if rates else 0.0
    trunc_issues = []
    if overall > MAX_TRUNCATION_RATE:
        trunc_issues.append(f"truncation rate {overall:.4f} > "
                            f"{MAX_TRUNCATION_RATE}")
    if spread > MAX_VARIANT_SPREAD:
        trunc_issues.append(f"truncation spread across variants {spread:.4f} > "
                            f"{MAX_VARIANT_SPREAD} (condition-dependent "
                            f"truncation)")
    result["truncation"] = {
        "overall_rate": overall, "variant_spread": spread,
        "per_variant_rate": rates,
        "per_variant_truncated": dict(truncated_by_variant),
        "per_variant_mean_output_tokens": {
            v: round(sum(t) / len(t), 1)
            for v, t in tokens_by_variant.items() if t},
        "per_variant_max_output_tokens": {
            v: max(t) for v, t in tokens_by_variant.items() if t},
        "thresholds": {"max_rate": MAX_TRUNCATION_RATE,
                       "max_variant_spread": MAX_VARIANT_SPREAD},
        "issues": trunc_issues, "ok": not trunc_issues}
    failures.extend(f"truncation: {i}" for i in trunc_issues)
    if trunc_issues:
        # A registered threshold was exceeded, so the uniform-cap remedy fires.
        # It has to be uniform: raising the cap for one variant or one
        # checkpoint would make the arms incomparable.
        warnings.append(
            f"{sum(truncated_by_variant.values())} truncated record(s) exceed "
            f"a registered threshold; the frozen uniform_cap_rule forbids "
            f"raising the cap for one variant or checkpoint — choose a new "
            f"UNIFORM cap and rerun ALL FIVE targets including Qwen3.5-9B, "
            f"retaining this evidence")
    elif overall > 0:
        # Within tolerance, so the remedy does NOT fire. Saying so explicitly
        # matters: this warning used to recommend a five-model rerun whenever a
        # single record was truncated, which is not what the registered
        # thresholds say, and under greedy decoding a repetition loop runs to
        # any cap so "escalate until nothing truncates" has no termination
        # criterion. The cells are still named for the record.
        warnings.append(
            f"{sum(truncated_by_variant.values())} truncated record(s), "
            f"within the registered tolerance (rate {overall:.4f} <= "
            f"{MAX_TRUNCATION_RATE}, variant spread {spread:.4f} <= "
            f"{MAX_VARIANT_SPREAD}); uniform-cap escalation is NOT triggered "
            f"and the panel is accepted with all its cells")

    # ---- verdict --------------------------------------------------------
    result["failures"] = failures
    result["warnings"] = warnings
    result["verdict"], result["unverifiable_sections"] = verdict_for(result)
    return result


def check_prompt_uniformity(model_keys: list[str]) -> list[str]:
    """The system prompt must be identical across every target.

    A target-specific prompt is one of the frozen stop-conditions, and it is
    only visible when the targets are compared with each other — no single
    run's evidence can show it.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for model_key in model_keys:
        preflight = PREFLIGHT_ROOT / model_key / "preflight.json"
        if not preflight.exists():
            continue
        doc = json.loads(preflight.read_text(encoding="utf-8"))
        seen[str(doc.get("system_prompt_sha256"))].append(model_key)
    if len(seen) > 1:
        return [f"system_prompt_sha256 differs across targets: "
                f"{ {k[:16]: v for k, v in seen.items()} }"]
    return []


def comparable_verdict(result: dict) -> str:
    """The verdict with this machine's blind spots factored out.

    ``PASS_WITH_UNVERIFIED_SECTIONS`` and ``PASS`` give the same answer about
    the panel -- nothing was found wrong -- and differ only in how much of it
    this checkout could look at. Compared literally, a fresh clone would report
    that four committed PASS reports "no longer reproduce" for want of 100
    images it never had, which is the exact confusion ``--verify`` exists to
    prevent. ``FAIL`` is never collapsed, so a filed PASS this checkout
    contradicts is still a difference.
    """
    verdict = result.get("verdict")
    return "PASS" if verdict == "PASS_WITH_UNVERIFIED_SECTIONS" else verdict


def diff_reports(fresh: dict, committed: dict) -> tuple[list[str], list[str]]:
    """Which fields a recomputed report no longer agrees with the filed one on.

    Returns ``(differing, skipped)``. A section this checkout could not verify is
    skipped and named, not compared: comparing it would report the absence of the
    media as a change in the panel, which is precisely the confusion this mode
    exists to prevent. A genuine media mismatch is still caught, because it lands
    in the top-level ``verdict`` and ``failures``, which are never skipped --
    ``verdict`` being compared through :func:`comparable_verdict` so that only a
    real disagreement counts.

    ``unverifiable_sections`` is the one key compared by neither rule. It is a
    statement about the machine, so it differs between checkouts by design, and
    listing it as a difference would put the blind spot back into the answer.
    Whichever side declares the blind spot, the section is not comparable: a
    report filed by a checkout that could not hash the media and re-read by one
    that can is the mirror image of the usual case, and the panel is no more
    contradicted by it.
    """
    unverifiable = (set(fresh.get("unverifiable_sections") or [])
                    | set(committed.get("unverifiable_sections") or []))
    differing, skipped = [], []
    for key in sorted(set(fresh) | set(committed)):
        if key in unverifiable:
            skipped.append(key)
        elif key == "unverifiable_sections":
            continue
        elif key == "verdict":
            if comparable_verdict(fresh) != comparable_verdict(committed):
                differing.append(key)
        elif fresh.get(key) != committed.get(key):
            differing.append(key)
    return differing, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="run directory (default: the single run under "
                             "outputs/iteration_11/generations/<model-key>)")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--all", action="store_true",
                        help="verify every target that has generations")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--verify", action="store_true",
                        help="recompute and compare against the committed "
                             f"{CHECKS_FILE}, writing nothing")
    parser.add_argument("--write-report", action="store_true",
                        help=f"write {CHECKS_FILE} into each run directory. "
                             "Generation is explicit because this file is "
                             "committed evidence: a verification command that "
                             "rewrites it turns a read into a mutation of the "
                             "record it was asked to check")
    args = parser.parse_args()

    if args.verify and args.write_report:
        parser.error("--verify compares without writing; --write-report "
                     "generates. Pick one.")
        return 2

    lock_path = Path(args.lock)

    if args.all:
        if not GENERATIONS_ROOT.exists():
            raise SystemExit(f"{GENERATIONS_ROOT} absent; nothing to verify")
        keys = sorted(p.name for p in GENERATIONS_ROOT.iterdir()
                      if p.is_dir() and any(
                          (p / r / "replay_outputs.jsonl").exists()
                          for r in p.iterdir() if r.is_dir()))
        if not keys:
            raise SystemExit(f"no 11.6 runs under {GENERATIONS_ROOT}")
    elif args.model_key:
        keys = [args.model_key]
    else:
        parser.error("give --model-key, or --all")
        return 2

    results, all_failures = [], []
    codes: list[int] = []
    incomplete: list[str] = []
    for model_key in keys:
        run_dir = Path(args.run_dir).resolve() \
            if (args.run_dir and len(keys) == 1) else find_run_dir(model_key)
        print(f"\n################ {model_key} ################")
        print(f"  run_dir {display_path(run_dir)}")
        result = check(model_key, run_dir, lock_path)
        for section in ("coverage", "full_panel", "provenance",
                        "confirmatory_gate", "eligibility",
                        "terminal_query_equality", "media", "truncation"):
            block = result[section]
            state = "ok" if block["ok"] else "FAIL"
            if block["ok"] and block.get("verifiable_here") is False:
                state = "UNVERIFIED"
            print(f"  {section:26s} {state}"
                  + (f"  ({len(block['issues'])} issue(s))"
                     if block.get("issues") else ""))
            for issue in block.get("issues", [])[:6]:
                print(f"      - {issue}")
        trunc = result["truncation"]
        print(f"  truncation rate={trunc['overall_rate']:.4f} "
              f"spread={trunc['variant_spread']:.4f}")
        print(f"  VERDICT: {result['verdict']} "
              f"({len(result['failures'])} failure(s))")
        for warning in result["warnings"]:
            print(f"  note: {warning[:300]}")

        out_path = run_dir / CHECKS_FILE
        if args.verify:
            if not out_path.exists():
                print(f"  no committed {CHECKS_FILE} to compare against")
                codes.append(2)
            else:
                committed = json.loads(out_path.read_text(encoding="utf-8"))
                differing, skipped = diff_reports(result, committed)
                blind = result["unverifiable_sections"]
                if differing:
                    print(f"  DIFFERS from the committed report: "
                          f"{differing}")
                    for key in differing[:6]:
                        print(f"      {key}: filed "
                              f"{json.dumps(committed.get(key))[:160]}")
                        print(f"      {key}: now   "
                              f"{json.dumps(result.get(key))[:160]}")
                    codes.append(1)
                elif skipped:
                    print(f"  reproduces the committed report on every section "
                          f"comparable here; NOT compared: {skipped}"
                          + ("" if blind else
                             " -- the FILED report is the one that could not "
                             "check them, this checkout did"))
                    if blind:
                        codes.append(3)
                        incomplete.append(model_key)
                    else:
                        codes.append(0)
                else:
                    print("  reproduces the committed report exactly")
                    codes.append(0)
            print("  wrote nothing (--verify)")
        elif args.write_report:
            out_path.write_text(json.dumps(result, indent=2,
                                           ensure_ascii=False) + "\n",
                                encoding="utf-8")
            print(f"  wrote {display_path(out_path)}")
            codes.append(3 if result["unverifiable_sections"] else 0)
            if result["unverifiable_sections"]:
                incomplete.append(model_key)
        else:
            print("  wrote nothing (pass --write-report to generate)")
            codes.append(3 if result["unverifiable_sections"] else 0)
            if result["unverifiable_sections"]:
                incomplete.append(model_key)

        results.append(result)
        all_failures.extend(f"{model_key}: {f}" for f in result["failures"])

    uniformity = check_prompt_uniformity(keys)
    if uniformity:
        all_failures.extend(f"cross-target: {u}" for u in uniformity)

    print(f"\n=== {len(results)} target(s) verified ===")
    for result in results:
        print(f"  {result['model_key']:14s} {result['verdict']}")
    for problem in uniformity:
        print(f"  {problem}")
    if all_failures:
        print(f"\nFAIL ({len(all_failures)} problem(s))")
        return 1
    if 1 in codes:
        print("\nFAIL: the committed report(s) no longer reproduce")
        return 1
    if 2 in codes:
        print("\nNO COMMITTED REPORT to verify against")
        return 2
    if 3 in codes or incomplete:
        print(f"\nINCOMPLETE: nothing failed, but {len(incomplete)} "
              f"target(s) had a section this checkout cannot verify "
              f"({', '.join(incomplete)}). That is not a PASS -- the media "
              f"identity is bound by "
              f"{display_path(MEDIA_MANIFEST_PATH)} and can be checked on any "
              f"machine holding data/media")
        return 3
    print("\nPASS: every target replayed the whole frozen panel under its "
          "locked revisions with an invariant terminal query, media matching "
          "its committed manifest, and acceptable truncation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
