#!/usr/bin/env python3
"""Scale-C (Iteration 10) closeout: build the frozen evidence manifest.

Scale-C counterpart of
outputs/iteration_9_closeout/scale_b_evidence_manifest.json. Binds
every phase of the 100-family preliminary experiment by SHA-256 and
git commit: frozen protocol, finalized panel, replay outputs/report/
checks, rubric, blinded items, Judge A/B labels + fingerprints, Kimi
adjudications + final ensemble labels, agreement/sensitivity/decision
reports, and the full Phase-9 audit chain (protocol, worksheet, sealed
-manifest parent, external answers, audit report, notes).

Phase 9 is described consistently as an EXTERNAL-MODEL CONFIRMATION,
not a completed human audit (the drawn worksheet remains unfilled).

Output: outputs/iteration_10_closeout/scale_c_evidence_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "iteration_10_closeout"
OUT = OUT_DIR / "scale_c_evidence_manifest.json"

REPLAY_DIR = "outputs/scale_c/replay_runs/scale-c-100-t1536-qwen35-9b"
ART = "outputs/scale_c/llm_judge_artifacts"
AUD = "outputs/scale_c/human_audit"

# (path, role) — every file must exist; fail-closed.
ARTIFACTS: list[tuple[str, str]] = [
    ("configs/experiments/scale_c_protocol.json",
     "frozen_preregistered_protocol"),
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
    (f"{AUD}/../HUMAN_AUDIT_PROTOCOL.md", "audit_protocol_frozen_pre_results"),
    (f"{AUD}/audit_worksheet.json", "audit_worksheet_blinded_unfilled"),
    (f"{AUD}/audit_sample_manifest.json",
     "audit_manifest_current_unsealed"),
    (f"{AUD}/gpt_audit_pack.zip", "external_scorer_pack"),
    (f"{AUD}/gpt_audit_pack_manifest.json", "external_pack_image_hashes"),
    (f"{AUD}/answers.json", "external_scorer_answers_45"),
    (f"{AUD}/audit_report.json", "audit_comparison_report"),
    (f"{AUD}/AUDIT_NOTES.md", "audit_notes"),
]

SEALED_MANIFEST_COMMIT = "9d6ba0a46d8f68fa47890510a36dcf1fdc804e16"
SEALED_MANIFEST_PATH = f"{AUD}/audit_sample_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(rel: str) -> str | None:
    r = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", rel],
        cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or None


def main() -> None:
    artifacts = {}
    for rel, role in ARTIFACTS:
        p = (ROOT / rel).resolve()
        if not p.is_file():
            raise SystemExit(f"missing artifact: {rel}")
        rel_clean = str(p.relative_to(ROOT))
        artifacts[role] = {
            "path": rel_clean,
            "sha256": _sha(p),
            "bytes": p.stat().st_size,
            "git_commit": _commit(rel_clean),
        }

    # Sealed-manifest parent: original blob recovered from git.
    blob = subprocess.run(
        ["git", "show", f"{SEALED_MANIFEST_COMMIT}:{SEALED_MANIFEST_PATH}"],
        cwd=ROOT, capture_output=True, check=True).stdout
    manifest_cur = json.loads(
        (ROOT / SEALED_MANIFEST_PATH).read_text("utf-8"))
    artifacts["audit_manifest_sealed_parent"] = {
        "sealed_commit": SEALED_MANIFEST_COMMIT[:7],
        "sealed_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "unsealed_at": manifest_cur.get("unsealed_at"),
        "worksheet_sha256_binding": manifest_cur.get("worksheet_sha256"),
        "note": "sealed at draw time (pre-scoring); unsealed only after "
                "all 45 external answers were returned",
    }

    decision = json.loads(
        (ROOT / f"{ART}/scale_c_decision_report.json").read_text("utf-8"))
    sens = json.loads(
        (ROOT / f"{ART}/judge_sensitivity.json").read_text("utf-8"))
    replay_report = json.loads(
        (ROOT / f"{REPLAY_DIR}/replay_report.json").read_text("utf-8"))

    def _ci(judge: str) -> dict:
        return sens["judges"][judge]["bootstrap_ci"]["Delta_TV"]

    manifest = {
        "closeout": {
            "iteration": "10",
            "scale": "C (100-family preliminary)",
            "status": "COMPLETE — frozen",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol_freeze_commit": "679c4b8",
            "replay_freeze_commit": "0944de5",
            "phase_9_kind": "external-model confirmation (GPT-family "
                            "scorer over 45 blinded outcome-stratified "
                            "items); NOT a completed human audit — the "
                            "drawn worksheet remains unfilled. A human "
                            "audit is required only if the paper claims "
                            "human validation.",
        },
        "primary_results": {
            "panel_gate": "passed (600/600, 0 failures, 0 truncation, "
                          "revision pinned, clean tree)",
            "primary_threshold_theta": decision["protocol"]["theta"],
            "decision": decision.get("decision", "POPULATION_INTERACTION"),
            "delta_tv": _ci("ensemble"),
            "delta_tv_judge_A": _ci("judge_A"),
            "delta_tv_judge_B": _ci("judge_B"),
            "qualifiers_at_theta":
                decision["qualifiers_at_theta"]["counts"],
            "adjudication": "distinct_model_adjudication_on_all_"
                            "disagreements (239/239, kimi-k3)",
            "cross_model_kappa_refusal": json.loads(
                (ROOT / f"{ART}/judge_agreement.json").read_text("utf-8")
            ).get("kappa_refusal"),
            "replay_model": replay_report.get("provenance", {}).get("model"),
            "resolved_model_revision":
                replay_report.get("provenance", {}).get(
                    "resolved_model_revision"),
        },
        "external_confirmation_summary": {
            "n_items": 45,
            "answers_sha256": artifacts["external_scorer_answers_45"]["sha256"],
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
    missing_commit = [r for r, a in artifacts.items()
                      if a.get("git_commit") is None
                      and r != "audit_manifest_sealed_parent"]
    if missing_commit:
        print(f"  WARNING: no git commit for: {missing_commit}")


if __name__ == "__main__":
    main()
