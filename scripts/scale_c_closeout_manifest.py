#!/usr/bin/env python3
"""Scale-C (Iteration 10) closeout: build/verify the frozen evidence manifest.

Scale-C counterpart of
outputs/iteration_9_closeout/scale_b_evidence_manifest.json.

EVIDENCE-CLOSED generation: the script independently re-verifies the
frozen facts before binding anything, and FAILS on any mismatch:
  - decision field exists and matches the frozen decision rule
    re-derived from final_evaluation_report.json;
  - replay checks verdict PASS with zero failures;
  - 100 families, 600 replay records, zero failures, zero truncation;
  - pinned model revision == protocol revision, clean-tree provenance;
  - adjudicator records exactly cover ALL recomputed A/B disagreements;
  - panel sha256 == replay-provenance binding; blinded items cover
    100 families x 6 variants; ensemble labels_sha256 recomputes.

REPRODUCIBILITY: generation requires a CLEAN tree for every artifact
(disk content == HEAD blob), and the timestamp is the HEAD commit date
— so running from the same commit yields a byte-identical manifest
modulo the HEAD-commit-derived fields. Two-commit flow:
  1. commit scripts/report/notes/tests/docs;
  2. run this script from that clean commit and commit the manifest.

VERIFY mode (no rewrite):
    python3 scripts/scale_c_closeout_manifest.py --verify
checks, for every bound artifact, that BOTH the on-disk file and the
git blob at the recorded commit:path hash to the stored SHA-256, and
that the sealed-manifest parent blob matches. Exits nonzero on any
mismatch.

Output: outputs/iteration_10_closeout/scale_c_evidence_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "iteration_10_closeout"
OUT = OUT_DIR / "scale_c_evidence_manifest.json"

REPLAY_DIR = "outputs/scale_c/replay_runs/scale-c-100-t1536-qwen35-9b"
ART = "outputs/scale_c/llm_judge_artifacts"
AUD = "outputs/scale_c/human_audit"
PROTOCOL_PATH = "configs/experiments/scale_c_protocol.json"

SEALED_MANIFEST_COMMIT = "9d6ba0a46d8f68fa47890510a36dcf1fdc804e16"
SEALED_MANIFEST_PATH = f"{AUD}/audit_sample_manifest.json"

JUDGE_FIELDS = ("refusal_type", "compliance_level", "safe_redirection",
                "unsafe_compliance_score")

# (path, role) — every file must exist AND be committed; fail-closed.
ARTIFACTS: list[tuple[str, str]] = [
    (PROTOCOL_PATH, "frozen_preregistered_protocol"),
    ("outputs/scale_c/families_panel/validated_families.jsonl",
     "finalized_100_family_panel"),
    (f"{REPLAY_DIR}/replay_outputs.jsonl", "replay_outputs_600"),
    (f"{REPLAY_DIR}/replay_report.json", "replay_report"),
    (f"{REPLAY_DIR}/replay_failures.jsonl", "replay_failures_empty"),
    (f"{REPLAY_DIR}/scale_c_replay_checks.json", "replay_checks"),
    ("src/causal_mllm/evaluation/annotation_rubric_v1_1.md",
     "rubric_v1_1"),
    (f"{ART}/blinded_items.json", "blinded_items_600"),
    (f"{ART}/llm_labels_judge_A.json", "judge_A_labels"),
    (f"{ART}/llm_labels_judge_A.json.fingerprint",
     "judge_A_binding_fingerprint"),
    (f"{ART}/llm_labels_judge_B.json", "judge_B_labels"),
    (f"{ART}/llm_labels_judge_B.json.fingerprint",
     "judge_B_binding_fingerprint"),
    (f"{ART}/llm_labels_adjudicator.json",
     "kimi_adjudications_per_call_provenance"),
    (f"{ART}/llm_labels_adjudicated.json", "final_ensemble_labels"),
    (f"{ART}/judge_agreement.json", "pairwise_agreement"),
    (f"{ART}/judge_sensitivity.json", "per_judge_sensitivity"),
    (f"{ART}/final_evaluation_report.json", "final_evaluation_report"),
    (f"{ART}/scale_c_decision_report.json", "frozen_decision_rule_report"),
    ("outputs/scale_c/HUMAN_AUDIT_PROTOCOL.md",
     "audit_protocol_frozen_pre_results"),
    (f"{AUD}/audit_worksheet.json", "audit_worksheet_blinded_unfilled"),
    (SEALED_MANIFEST_PATH, "audit_manifest_current_unsealed"),
    (f"{AUD}/gpt_audit_pack.zip", "external_scorer_pack"),
    (f"{AUD}/gpt_audit_pack_manifest.json", "external_pack_image_hashes"),
    (f"{AUD}/answers.json", "external_scorer_answers_45"),
    (f"{AUD}/audit_report.json", "audit_comparison_report"),
    (f"{AUD}/AUDIT_NOTES.md", "audit_notes"),
    # Analysis/generation code bound with the evidence it produced.
    ("scripts/scale_c_audit_sample.py", "script_audit_sampler"),
    ("scripts/scale_c_audit_pack.py", "script_audit_pack_builder"),
    ("scripts/scale_c_audit_report.py", "script_audit_comparison"),
    ("scripts/scale_c_decision_report.py", "script_decision_rule"),
    ("scripts/scale_c_closeout_manifest.py", "script_closeout_generator"),
    ("scripts/audit_annotation_server.py", "script_annotation_server"),
    ("tests/unit/test_scale_c_closeout.py", "tests_closeout_and_validation"),
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _derive_decision(dtv: dict | None, dt: dict | None,
                     hist: dict | None) -> str:
    """Re-implement the FROZEN Scale-C decision rule."""
    def above_zero(ci):
        return ci is not None and ci["CI_lower"] > 0.0

    def includes_zero(ci):
        return ci is not None and ci["CI_lower"] <= 0.0 <= ci["CI_upper"]

    if above_zero(dtv):
        return "POPULATION_INTERACTION"
    if includes_zero(dtv) and above_zero(dt) and above_zero(hist):
        return "TEXT_OR_HISTORY_ONLY"
    return "SAMPLING_SENSITIVE"


def _disagreement_ids(records_a: list[dict],
                      records_b: list[dict]) -> set[str]:
    """Item ids where judges A and B differ on ANY label field."""
    ja = {r["item_id"]: r["judgment"] for r in records_a}
    jb = {r["item_id"]: r["judgment"] for r in records_b}
    if set(ja) != set(jb):
        raise SystemExit("judge A/B item_id sets differ")
    return {
        iid for iid in ja
        if any(ja[iid].get(f) != jb[iid].get(f) for f in JUDGE_FIELDS)}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                       check=True)
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def _head_commit_date() -> str:
    """Deterministic timestamp: committer date of HEAD."""
    return _git("log", "-1", "--format=%cI")


def _head_commit() -> str:
    return _git("rev-parse", "HEAD")


def _blob_at(commit: str, path: str) -> bytes:
    return _git("show", f"{commit}:{path}", binary=True)


def _last_commit_for(path: str) -> str | None:
    out = _git("log", "-1", "--format=%h", "--", path)
    return out or None


def _assert_clean_tree(paths: list[str]) -> None:
    """Every path must be committed with disk content == HEAD blob."""
    dirty = _git("status", "--porcelain", "--", *paths)
    if dirty:
        raise SystemExit(
            "closeout requires a CLEAN tree for all artifacts; commit "
            "first (two-commit flow). Dirty:\n" + dirty)
    mismatched = []
    for rel in paths:
        disk = _sha256_file(ROOT / rel)
        try:
            head = _sha256_bytes(_blob_at("HEAD", rel))
        except subprocess.CalledProcessError:
            mismatched.append(f"{rel} (not in HEAD)")
            continue
        if disk != head:
            mismatched.append(rel)
    if mismatched:
        raise SystemExit(
            "disk != HEAD blob for: " + ", ".join(mismatched))


# ---------------------------------------------------------------------------
# Evidence-closed verification (F2)
# ---------------------------------------------------------------------------

def _verify_evidence() -> dict:
    """Independently re-verify the frozen facts; fail-closed.

    Returns the verified facts dict embedded in the manifest.
    """
    def load(rel):
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))

    protocol = load(PROTOCOL_PATH)
    pinned_rev = protocol["replay"]["model_revision"]
    theta = protocol["analysis"]["primary_threshold_theta"]

    # --- panel -------------------------------------------------------------
    panel_path = ROOT / "outputs/scale_c/families_panel/validated_families.jsonl"
    panel_bytes = panel_path.read_bytes()
    n_families = sum(1 for ln in panel_bytes.splitlines() if ln.strip())
    if n_families != 100:
        raise SystemExit(f"panel: expected 100 families, got {n_families}")
    panel_sha = _sha256_bytes(panel_bytes)

    # --- replay report / failures / checks ---------------------------------
    report = load(f"{REPLAY_DIR}/replay_report.json")
    prov = report["provenance"]
    checks = load(f"{REPLAY_DIR}/scale_c_replay_checks.json")
    failures = (ROOT / f"{REPLAY_DIR}/replay_failures.jsonl").read_bytes()
    n_failures_lines = sum(1 for ln in failures.splitlines() if ln.strip())
    n_outputs = sum(
        1 for ln in (ROOT / f"{REPLAY_DIR}/replay_outputs.jsonl")
        .read_bytes().splitlines() if ln.strip())

    problems = []
    if checks.get("verdict") != "PASS":
        problems.append(f"replay checks verdict={checks.get('verdict')!r}")
    if checks.get("failures"):
        problems.append(f"replay checks failures={checks['failures']}")
    if report.get("n_families") != 100:
        problems.append(f"replay n_families={report.get('n_families')}")
    if n_outputs != 600 or report.get("n_succeeded") != 600:
        problems.append(
            f"replay records={n_outputs}, n_succeeded={report.get('n_succeeded')}")
    if report.get("n_failed") != 0 or n_failures_lines != 0:
        problems.append(
            f"replay failures n_failed={report.get('n_failed')}, "
            f"file lines={n_failures_lines}")
    trunc = report.get("truncation", {})
    if trunc.get("n_truncated") != 0:
        problems.append(f"truncation n_truncated={trunc.get('n_truncated')}")
    if prov.get("revision_pinned") is not True:
        problems.append("revision_pinned is not True")
    if prov.get("git_dirty") is not False:
        problems.append(f"git_dirty={prov.get('git_dirty')!r}")
    if prov.get("resolved_model_revision") != pinned_rev \
            or prov.get("requested_model_revision") != pinned_rev:
        problems.append(
            f"revision mismatch: requested="
            f"{prov.get('requested_model_revision')}, resolved="
            f"{prov.get('resolved_model_revision')}, protocol={pinned_rev}")
    if prov.get("validated_families_sha256") != panel_sha:
        problems.append("panel sha256 != replay-provenance binding")
    if problems:
        raise SystemExit(
            "evidence verification FAILED (replay):\n  "
            + "\n  ".join(problems))

    # --- judges, disagreements, adjudication coverage -----------------------
    judge_a = load(f"{ART}/llm_labels_judge_A.json")
    judge_b = load(f"{ART}/llm_labels_judge_B.json")
    adjudicator = load(f"{ART}/llm_labels_adjudicator.json")
    if len(judge_a) != 600 or len(judge_b) != 600:
        raise SystemExit(
            f"judge records: A={len(judge_a)}, B={len(judge_b)} "
            f"(expected 600 each)")
    disagreements = _disagreement_ids(judge_a, judge_b)
    adj_ids = {r["item_id"] for r in adjudicator["items"]}
    if adj_ids != disagreements:
        raise SystemExit(
            f"adjudication coverage gap: adjudicated={len(adj_ids)}, "
            f"disagreements={len(disagreements)}, symmetric difference="
            f"{sorted(adj_ids ^ disagreements)[:5]}")

    # --- blinded items: 600, full 100x6 coverage -----------------------------
    blinded = load(f"{ART}/blinded_items.json")
    pairs = {(it["family_id"], it["variant"]) for it in blinded}
    fams = {it["family_id"] for it in blinded}
    if len(blinded) != 600 or len(fams) != 100 or len(pairs) != 600:
        raise SystemExit(
            f"blinded items: {len(blinded)} items, {len(fams)} families, "
            f"{len(pairs)} unique pairs (expected 600/100/600)")

    # --- ensemble labels integrity -------------------------------------------
    labels = load(f"{ART}/llm_labels_adjudicated.json")
    lprov = labels["provenance"]
    recomputed = _sha256_bytes(
        json.dumps(labels["labels"], sort_keys=True,
                   ensure_ascii=False).encode("utf-8"))
    if recomputed != lprov.get("labels_sha256"):
        raise SystemExit("ensemble labels_sha256 mismatch on recompute")
    if lprov.get("n_families") != 100 or lprov.get("n_labels") != 600:
        raise SystemExit(
            f"ensemble labels provenance: n_families="
            f"{lprov.get('n_families')}, n_labels={lprov.get('n_labels')}")

    # --- decision rule re-derivation ------------------------------------------
    final_report = load(f"{ART}/final_evaluation_report.json")
    decision_report = load(f"{ART}/scale_c_decision_report.json")
    ci = final_report["estimands"]["bootstrap_ci"]
    derived = _derive_decision(ci.get("Delta_TV"), ci.get("Delta_T"),
                               ci.get("history_effect"))
    committed_decision = decision_report.get("decision")
    if not committed_decision:
        raise SystemExit("decision report has no 'decision' field")
    if committed_decision != derived:
        raise SystemExit(
            f"decision mismatch: report={committed_decision!r}, "
            f"re-derived from frozen rule={derived!r}")

    sens = load(f"{ART}/judge_sensitivity.json")

    return {
        "panel": {"n_families": n_families, "sha256": panel_sha},
        "replay": {
            "n_records": n_outputs,
            "n_succeeded": report["n_succeeded"],
            "n_failed": report["n_failed"],
            "n_truncated": trunc["n_truncated"],
            "checks_verdict": checks["verdict"],
            "revision_pinned": prov["revision_pinned"],
            "resolved_model_revision": prov["resolved_model_revision"],
            "requested_equals_protocol_revision": True,
            "git_dirty": prov["git_dirty"],
            "replay_git_commit": prov.get("git_commit"),
            "panel_sha_matches_replay_provenance": True,
        },
        "judging": {
            "judge_A_records": len(judge_a),
            "judge_B_records": len(judge_b),
            "disagreements_recomputed": len(disagreements),
            "adjudicated_records": len(adj_ids),
            "adjudication_covers_all_disagreements": True,
            "blinded_items": len(blinded),
            "blinded_unique_family_variant_pairs": len(pairs),
            "ensemble_labels_sha256_recomputed": recomputed,
        },
        "decision": {
            "committed": committed_decision,
            "re_derived_from_frozen_rule": derived,
            "theta": theta,
            "delta_tv_ci": ci.get("Delta_TV"),
            "delta_tv_ci_judge_A":
                sens["judges"]["judge_A"]["bootstrap_ci"].get("Delta_TV"),
            "delta_tv_ci_judge_B":
                sens["judges"]["judge_B"]["bootstrap_ci"].get("Delta_TV"),
        },
    }


# ---------------------------------------------------------------------------
# Generate / verify
# ---------------------------------------------------------------------------

def generate() -> None:
    paths = [rel for rel, _ in ARTIFACTS]
    _assert_clean_tree(paths)
    facts = _verify_evidence()

    artifacts = {}
    for rel, role in ARTIFACTS:
        p = ROOT / rel
        artifacts[role] = {
            "path": rel,
            "sha256": _sha256_file(p),
            "bytes": p.stat().st_size,
            "git_commit": _last_commit_for(rel),
        }

    sealed_blob = _blob_at(SEALED_MANIFEST_COMMIT, SEALED_MANIFEST_PATH)
    manifest_cur = json.loads(
        (ROOT / SEALED_MANIFEST_PATH).read_text("utf-8"))
    artifacts["audit_manifest_sealed_parent"] = {
        "sealed_commit": SEALED_MANIFEST_COMMIT[:7],
        "sealed_blob_sha256": _sha256_bytes(sealed_blob),
        "unsealed_at": manifest_cur.get("unsealed_at"),
        "worksheet_sha256_binding": manifest_cur.get("worksheet_sha256"),
        "note": "sealed at draw time (pre-scoring); unsealed only after "
                "all 45 external answers were returned",
    }

    agreement = json.loads(
        (ROOT / f"{ART}/judge_agreement.json").read_text("utf-8"))
    decision_report = json.loads(
        (ROOT / f"{ART}/scale_c_decision_report.json").read_text("utf-8"))

    manifest = {
        "closeout": {
            "iteration": "10",
            "scale": "C (100-family preliminary)",
            "status": "COMPLETE — frozen",
            # Deterministic: HEAD commit date, not wall clock.
            "timestamp": _head_commit_date(),
            "generated_from_commit": _head_commit(),
            "protocol_freeze_commit": "679c4b8",
            "replay_freeze_commit": "0944de5",
            "phase_9_kind": "external-model confirmation (GPT-family "
                            "scorer over 45 blinded outcome-stratified "
                            "items); NOT a completed human audit — the "
                            "drawn worksheet remains unfilled. A human "
                            "audit is required only if the paper claims "
                            "human validation.",
        },
        "evidence_checks": facts,
        "primary_results": {
            "panel_gate": "passed (verified at generation: "
                          f"{facts['replay']['n_records']}/600 records, "
                          f"{facts['replay']['n_failed']} failures, "
                          f"{facts['replay']['n_truncated']} truncation, "
                          "revision pinned, clean tree)",
            "primary_threshold_theta": facts["decision"]["theta"],
            "decision": facts["decision"]["committed"],
            "decision_rule_re_derived":
                facts["decision"]["re_derived_from_frozen_rule"],
            "delta_tv": facts["decision"]["delta_tv_ci"],
            "delta_tv_judge_A": facts["decision"]["delta_tv_ci_judge_A"],
            "delta_tv_judge_B": facts["decision"]["delta_tv_ci_judge_B"],
            "qualifiers_at_theta":
                decision_report["qualifiers_at_theta"]["counts"],
            "adjudication": "distinct_model_adjudication_on_all_"
                            f"disagreements ({facts['judging']['adjudicated_records']}"
                            f"/{facts['judging']['disagreements_recomputed']}, kimi-k3)",
            "cross_model_kappa_refusal": agreement.get("kappa_refusal"),
            "replay_model": "Qwen/Qwen3.5-9B",
            "resolved_model_revision":
                facts["replay"]["resolved_model_revision"],
        },
        "external_confirmation_summary": {
            "n_items": 45,
            "answers_sha256":
                artifacts["external_scorer_answers_45"]["sha256"],
            "conclusion": "consistent with and does not contradict the "
                          "primary decision; cannot independently "
                          "estimate population Delta_TV (outcome-"
                          "stratified sample, 2 complete families)",
            "population_claim_basis": "preregistered 600-item ensemble "
                                      "analysis (delta_tv CIs above)",
            "details": f"{AUD}/audit_report.json + {AUD}/AUDIT_NOTES.md",
        },
        "artifacts": artifacts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  artifacts bound: {len(artifacts)}")
    print(f"  generated_from_commit: {manifest['closeout']['generated_from_commit'][:7]}")
    print(f"  evidence checks: ALL PASS (decision re-derived: "
          f"{facts['decision']['re_derived_from_frozen_rule']})")


def verify() -> int:
    """Check every claimed commit:path blob against the stored SHA-256.

    Never rewrites the manifest. Returns process exit code.
    """
    if not OUT.exists():
        print(f"FAIL: manifest missing: {OUT}")
        return 1
    manifest = json.loads(OUT.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    failures = []
    for role, a in artifacts.items():
        if role == "audit_manifest_sealed_parent":
            blob = _blob_at(a["sealed_commit"], SEALED_MANIFEST_PATH)
            if _sha256_bytes(blob) != a["sealed_blob_sha256"]:
                failures.append(f"{role}: sealed blob sha mismatch")
            continue
        rel, stored, commit = a["path"], a["sha256"], a.get("git_commit")
        disk = ROOT / rel
        if not disk.is_file():
            failures.append(f"{role}: file missing on disk ({rel})")
        elif _sha256_file(disk) != stored:
            failures.append(f"{role}: DISK sha mismatch ({rel})")
        if not commit:
            failures.append(f"{role}: no git_commit recorded ({rel})")
            continue
        try:
            blob_sha = _sha256_bytes(_blob_at(commit, rel))
        except subprocess.CalledProcessError:
            failures.append(
                f"{role}: blob {commit}:{rel} not found in git")
            continue
        if blob_sha != stored:
            failures.append(
                f"{role}: COMMITTED blob sha mismatch at {commit}:{rel} "
                f"(stored {stored[:12]}, blob {blob_sha[:12]})")
    if failures:
        print(f"VERIFY FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"VERIFY PASS: {len(artifacts)} artifacts; disk AND "
          f"commit:path blobs match every stored SHA-256")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify the committed manifest without rewriting")
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify())
    generate()


if __name__ == "__main__":
    main()
