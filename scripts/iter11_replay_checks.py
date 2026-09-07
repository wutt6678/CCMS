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

Writes ``iteration_11_replay_checks.json`` into the run directory and exits
non-zero on any failure.

Usage:
    python3 scripts/iter11_replay_checks.py --model-key qwen35_2b
    python3 scripts/iter11_replay_checks.py --model-key phi4_mm RUN_DIR
    python3 scripts/iter11_replay_checks.py --all
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
    media_issues = []
    families = panel_families()
    for family_id, family in sorted(families.items()):
        records = by_family.get(family_id)
        if not records:
            continue
        expected_counts, shared = {}, set()
        for variant_name, variant in (family.get("variants") or {}).items():
            paths = [p for message in variant.get("messages") or []
                     for p in (message.get("images") or [])]
            expected_counts[variant_name] = len(paths)
            for path in paths:
                resolved = REPO_ROOT / path
                if resolved.exists():
                    shared.add(sha256_file(resolved))
                else:
                    media_issues.append(f"{family_id}/{variant_name}: image "
                                        f"{path} is missing from the repo")
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
    result["media"] = {"issues": media_issues[:40],
                       "n_issues": len(media_issues),
                       "ok": not media_issues}
    failures.extend(f"media: {i}" for i in media_issues[:40])

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

    # ---- cross-target prompt uniformity (reported, not per-target fatal) --
    result["verdict"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures
    result["warnings"] = warnings
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="run directory (default: the single run under "
                             "outputs/iteration_11/generations/<model-key>)")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--all", action="store_true",
                        help="verify every target that has generations")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = parser.parse_args()

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
            print(f"  {section:26s} {'ok' if block['ok'] else 'FAIL'}"
                  + (f"  ({len(block['issues'])} issue(s))"
                     if block.get("issues") else ""))
            for issue in block.get("issues", [])[:6]:
                print(f"      - {issue}")
        trunc = result["truncation"]
        print(f"  truncation rate={trunc['overall_rate']:.4f} "
              f"spread={trunc['variant_spread']:.4f}")
        print(f"  VERDICT: {result['verdict']} "
              f"({len(result['failures'])} failure(s))")
        out_path = run_dir / CHECKS_FILE
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)
                            + "\n", encoding="utf-8")
        print(f"  wrote {display_path(out_path)}")
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
    print("\nPASS: every target replayed the whole frozen panel under its "
          "locked revisions with an invariant terminal query, identical "
          "shared media, and acceptable truncation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
