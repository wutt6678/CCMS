#!/usr/bin/env python3
"""Scale-C replay verification — preflight (Phase 5) and full gates (Phase 6).

Checks a replay run directory against the FROZEN Scale-C protocol
(configs/experiments/scale_c_protocol.json). Emits a verdict JSON into
the run directory and exits non-zero on any FAIL.

Gates:
  - coverage: every family has exactly the six protocol variants,
    unique records, zero missing/error records
  - provenance: pinned revision == frozen revision, system-prompt and
    template revisions match, clean-tree fingerprint (git_dirty=False),
    resolved_sha256 present
  - terminal-query equality: identical terminal_sha256 across all six
    variants within each family AND equal to the panel canonical q*
  - media: per-variant image counts match the family definition and
    shared-image byte identity (sha256) across variants
  - truncation: near-zero hit_max_new_tokens overall and no material
    difference between variants
  - output-token diagnostics per variant

Usage:
    python3 scripts/scale_c_replay_checks.py RUN_DIR [--expect-n-families 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROTOCOL = json.loads(
    (Path(__file__).parent.parent
     / "configs/experiments/scale_c_protocol.json").read_text())
FROZEN_REVISION = PROTOCOL["replay"]["model_revision"]
FROZEN_PROMPT_SHA = PROTOCOL["replay"]["system_prompt_sha256"]
FROZEN_VARIANTS = PROTOCOL["dataset"]["variants"]
PANEL_PATH = Path("outputs/scale_c/families_panel/validated_families.jsonl")

# Truncation tolerance: "near-zero". Preflight uses the same rule.
MAX_TRUNCATION_RATE = 0.02
# Largest acceptable absolute truncation-rate spread across variants.
MAX_VARIANT_SPREAD = 0.05


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--expect-n-families", type=int, default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    failures: list[str] = []
    warnings: list[str] = []
    report: dict = {"run_dir": str(run_dir)}

    outputs = [json.loads(l) for l in
               (run_dir / "replay_outputs.jsonl").read_text().splitlines()
               if l.strip()]
    run_report = json.loads((run_dir / "replay_report.json").read_text())

    # ---- coverage -------------------------------------------------------
    by_family: dict[str, list[dict]] = defaultdict(list)
    for rec in outputs:
        by_family[rec["family_id"]].append(rec)
    n_families = len(by_family)
    if args.expect_n_families is not None and n_families != args.expect_n_families:
        failures.append(
            f"expected {args.expect_n_families} families, found {n_families}")
    n_records = len(outputs)
    keys = {(r["family_id"], r["variant"]) for r in outputs}
    if len(keys) != n_records:
        failures.append("duplicate (family_id, variant) records")
    for fid, recs in by_family.items():
        got = sorted(r["variant"] for r in recs)
        if got != sorted(FROZEN_VARIANTS):
            failures.append(f"{fid}: variants {got}")
    errs = [r for r in outputs if r.get("error")]
    if errs:
        failures.append(f"{len(errs)} error records")
    report["coverage"] = {
        "n_families": n_families, "n_records": n_records,
        "n_error_records": len(errs),
        "ok": not failures,
    }

    # ---- provenance -----------------------------------------------------
    prov = run_report.get("provenance", {})
    prov_issues = []
    if prov.get("requested_model_revision") != FROZEN_REVISION:
        prov_issues.append("requested revision != frozen")
    if prov.get("resolved_model_revision") != FROZEN_REVISION:
        prov_issues.append("resolved revision != frozen")
    if prov.get("revision_pinned") is not True:
        prov_issues.append("revision not pinned")
    if prov.get("system_prompt_sha256") != FROZEN_PROMPT_SHA:
        prov_issues.append("system prompt sha mismatch")
    if prov.get("prompt_template_revision") != "v1":
        prov_issues.append("prompt template revision != v1")
    if prov.get("git_dirty") is not False:
        prov_issues.append("git tree dirty at run time")
    if not prov.get("resolved_sha256"):
        prov_issues.append("missing resolved_sha256 fingerprint")
    per_rec_issues = 0
    for r in outputs:
        if (r.get("resolved_model_revision") != FROZEN_REVISION
                or r.get("system_prompt_sha256") != FROZEN_PROMPT_SHA):
            per_rec_issues += 1
    if per_rec_issues:
        prov_issues.append(f"{per_rec_issues} records with wrong "
                           "revision/prompt fingerprints")
    if prov_issues:
        failures.extend([f"provenance: {x}" for x in prov_issues])
    report["provenance"] = {"issues": prov_issues, "ok": not prov_issues}

    # ---- terminal-query equality ----------------------------------------
    panel = {}
    for line in PANEL_PATH.read_text().splitlines():
        if line.strip():
            fam = json.loads(line)
            tq = fam["terminal_query"]
            text = tq["text"] if isinstance(tq, dict) else tq
            panel[fam["family_id"]] = text
    term_issues = []
    for fid, recs in by_family.items():
        shas = {r.get("terminal_sha256") for r in recs}
        if len(shas) != 1:
            term_issues.append(f"{fid}: {len(shas)} distinct terminal hashes")
            continue
        expected = hashlib.sha256(
            panel[fid].encode("utf-8")).hexdigest() if fid in panel else None
        if expected and next(iter(shas)) != expected:
            term_issues.append(f"{fid}: terminal hash != panel canonical q*")
    if term_issues:
        failures.extend([f"terminal: {x}" for x in term_issues])
    report["terminal_query_equality"] = {
        "issues": term_issues, "ok": not term_issues}

    # ---- media identity ---------------------------------------------------
    media_issues = []
    panel_fams = {}
    for line in PANEL_PATH.read_text().splitlines():
        if line.strip():
            fam = json.loads(line)
            panel_fams[fam["family_id"]] = fam
    for fid, fam in panel_fams.items():
        if fid not in by_family:
            continue
        shared_hashes = set()
        expected_counts = {}
        for vname, v in fam["variants"].items():
            paths = [p for m in v["messages"] for p in (m.get("images") or [])]
            expected_counts[vname] = len(paths)
            for p in paths:
                shared_hashes.add(sha256_file(p))
        for r in by_family[fid]:
            if r.get("n_images") != expected_counts.get(r["variant"], 0):
                media_issues.append(
                    f"{fid}/{r['variant']}: n_images "
                    f"{r.get('n_images')} != {expected_counts.get(r['variant'])}")
        # Shared-image identity: every image referenced by any variant of
        # the family must be byte-identical (hash set already collapses
        # duplicates; verify each path resolves to the single shared hash).
        if len(shared_hashes) > 1:
            media_issues.append(
                f"{fid}: {len(shared_hashes)} distinct image hashes "
                "across variants (shared images must be identical)")
    if media_issues:
        failures.extend([f"media: {x}" for x in media_issues])
    report["media"] = {"issues": media_issues, "ok": not media_issues}

    # ---- truncation + token diagnostics ----------------------------------
    trunc_by_variant: dict[str, int] = defaultdict(int)
    n_by_variant: dict[str, int] = defaultdict(int)
    out_tokens_by_variant: dict[str, list[int]] = defaultdict(list)
    for r in outputs:
        v = r["variant"]
        n_by_variant[v] += 1
        out_tokens_by_variant[v].append(r.get("output_token_count", 0))
        if r.get("hit_max_new_tokens"):
            trunc_by_variant[v] += 1
    rates = {v: trunc_by_variant[v] / n_by_variant[v]
             for v in n_by_variant if n_by_variant[v]}
    overall_rate = sum(trunc_by_variant.values()) / max(1, n_records)
    spread = (max(rates.values()) - min(rates.values())) if rates else 0.0
    if overall_rate > MAX_TRUNCATION_RATE:
        failures.append(
            f"truncation rate {overall_rate:.3f} > {MAX_TRUNCATION_RATE}")
    elif spread > MAX_VARIANT_SPREAD:
        failures.append(
            f"truncation spread across variants {spread:.3f} > "
            f"{MAX_VARIANT_SPREAD} (condition-dependent truncation)")
    report["truncation"] = {
        "overall_rate": overall_rate,
        "per_variant_rate": rates,
        "per_variant_truncated": dict(trunc_by_variant),
        "per_variant_mean_output_tokens": {
            v: round(sum(t) / len(t), 1)
            for v, t in out_tokens_by_variant.items() if t},
        "per_variant_max_output_tokens": {
            v: max(t) for v, t in out_tokens_by_variant.items() if t},
        "ok": overall_rate <= MAX_TRUNCATION_RATE
        and spread <= MAX_VARIANT_SPREAD,
    }
    if overall_rate > 0:
        warnings.append(
            f"{sum(trunc_by_variant.values())} truncated record(s); "
            "frozen policy: raise cap uniformly and rerun the ENTIRE "
            "panel if truncation is material or variant-dependent")

    verdict = "PASS" if not failures else "FAIL"
    report["verdict"] = verdict
    report["failures"] = failures
    report["warnings"] = warnings
    out_path = run_dir / "scale_c_replay_checks.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nVERDICT: {verdict} -> {out_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
