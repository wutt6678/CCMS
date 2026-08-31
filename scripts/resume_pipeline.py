#!/usr/bin/env python3
"""Resume pipeline from saved judge outputs.

This script picks up after the 3 judges have completed, computing
agreement, adjudicating, and running the causal evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from causal_mllm.evaluation.agreement import compute_judge_agreement
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.human_template import save_human_labels
from causal_mllm.evaluation.judge import HumanLabelJudge
from causal_mllm.evaluation.runner import run_evaluation_stage

# Paths
FINAL_PANEL_RUN = Path(
    "outputs/replay_runs/scale-b-2026-08-28-t1536-final-qwen35-9b")
VALIDATED_FAMILIES_PATH = Path(
    "outputs/families/scale_b_smoke/validated_families.jsonl")
OUTPUT_DIR = Path("outputs/llm_judge_artifacts")


def main():
    """Resume from saved judge outputs."""
    print("Loading judge outputs...")
    all_judgments = {}
    for judge_id in ["A", "B", "C"]:
        path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        with path.open(encoding="utf-8") as f:
            all_judgments[judge_id] = json.load(f)
        print(f"  Judge {judge_id}: {len(all_judgments[judge_id])} items")

    # Compute agreement
    print("\nComputing inter-judge agreement...")
    agreement = compute_judge_agreement(
        all_judgments["A"],
        all_judgments["B"],
        all_judgments["C"],
    )

    agreement_path = OUTPUT_DIR / "judge_agreement.json"
    with agreement_path.open("w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)
    print(f"  Saved agreement: {agreement_path}")

    print(f"  Cohen's κ (refusal) mean: "
          f"{agreement['kappa_refusal']['mean']:.4f}")
    print(f"  Weighted κ (compliance) mean: "
          f"{agreement['kappa_compliance_weighted']['mean']:.4f}")
    print(f"  MAE (score) mean: {agreement['mae_score']['mean']:.4f}")
    print(f"  ICC (score): {agreement['icc_score']:.4f}")
    print(f"  Spearman ρ (score) mean: "
          f"{agreement['spearman_rho']['mean']:.4f}")

    # Adjudicate
    print("\nAdjudicating disagreements...")

    # Build lookups
    lookup_a = {j["item_id"]: j for j in all_judgments["A"]}
    lookup_b = {j["item_id"]: j for j in all_judgments["B"]}
    lookup_c = {j["item_id"]: j for j in all_judgments["C"]}

    adjudicated = []
    for item_id in sorted(lookup_a.keys()):
        ja = lookup_a[item_id]["judgment"]
        jb = lookup_b[item_id]["judgment"]
        jc = lookup_c[item_id]["judgment"]

        # Majority voting for refusal_type
        votes = [ja["refusal_type"], jb["refusal_type"], jc["refusal_type"]]
        refusal_type = max(set(votes), key=votes.count)

        # Majority voting for compliance_level
        votes = [ja["compliance_level"], jb["compliance_level"],
                 jc["compliance_level"]]
        compliance_level = max(set(votes), key=votes.count)

        # Average scores
        scores = [ja["unsafe_compliance_score"],
                  jb["unsafe_compliance_score"],
                  jc["unsafe_compliance_score"]]
        avg_score = sum(scores) / len(scores)

        # Average confidence
        confidences = [ja["confidence"], jb["confidence"], jc["confidence"]]
        avg_confidence = sum(confidences) / len(confidences)

        # Safe redirection: majority vote
        votes = [ja["safe_redirection"], jb["safe_redirection"],
                 jc["safe_redirection"]]
        safe_redirection = sum(votes) >= 2

        # Combine rationales
        rat_a = ja["rationale"][:50]
        rat_b = jb["rationale"][:50]
        rat_c = jc["rationale"][:50]
        rationale = (f"Adjudicated from 3 judges. "
                     f"A: {rat_a}... B: {rat_b}... C: {rat_c}...")

        adjudicated.append({
            "item_id": item_id,
            "family_id": lookup_a[item_id]["family_id"],
            "variant": lookup_a[item_id]["variant"],
            "response_sha256": lookup_a[item_id]["response_sha256"],
            "judgment": {
                "refusal_type": refusal_type,
                "compliance_level": compliance_level,
                "unsafe_compliance_score": avg_score,
                "safe_redirection": safe_redirection,
                "confidence": avg_confidence,
                "rationale": rationale,
            },
            "provenance": {
                "backend": "multimodal_llm_judge_adjudicated",
                "judge_a_model": lookup_a[item_id][
                    "provenance"]["model_id"],
                "judge_b_model": lookup_b[item_id][
                    "provenance"]["model_id"],
                "judge_c_model": lookup_c[item_id][
                    "provenance"]["model_id"],
                "adjudication_method": (
                    "majority_vote_categorical_mean_score"),
            },
        })

    # Convert to HumanLabelJudge format
    adjudicated_labels = {}
    for rec in adjudicated:
        family_id = rec["family_id"]
        variant = rec["variant"]
        if family_id not in adjudicated_labels:
            adjudicated_labels[family_id] = {}
        adjudicated_labels[family_id][variant] = {
            **rec["judgment"],
            "response_sha256": rec["response_sha256"],
            "rubric_version": "1.0",
            "annotator_id": "llm_judge_adjudicated",
            "adjudicated": True,
            "item_id": rec["item_id"],
        }

    # Save adjudicated labels
    labels_path = OUTPUT_DIR / "llm_labels_adjudicated.json"
    save_human_labels(
        adjudicated_labels,
        labels_path,
        annotator_id="llm_judge_adjudicated",
        adjudicated=True,
    )
    print(f"  Saved adjudicated labels: {labels_path}")

    # Run causal evaluation
    print("\nRunning causal evaluation with adjudicated labels...")
    judge = HumanLabelJudge(labels_path)

    eval_config = EvalConfig(
        n_bootstrap=5000,
        seed=42,
    )

    report = run_evaluation_stage(
        run_dir=FINAL_PANEL_RUN,
        judge=judge,
        config=eval_config,
        output_root=OUTPUT_DIR / "evaluation_results",
        validated_families_path=VALIDATED_FAMILIES_PATH,
    )

    # Save final report
    report_path = OUTPUT_DIR / "final_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved final report: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("LLM JUDGE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print("  - blinded_items.json")
    print("  - llm_labels_judge_A.json")
    print("  - llm_labels_judge_B.json")
    print("  - llm_labels_judge_C.json")
    print("  - judge_agreement.json")
    print("  - llm_labels_adjudicated.json")
    print("  - final_evaluation_report.json")

    print("\n" + "=" * 60)
    print("RESEARCH VALIDITY NOTE")
    print("=" * 60)
    print("""
For Iteration 9 smoke evidence, two independent LLM judges plus
adjudication is acceptable. For a paper-quality final claim, manually
audit ~20-30 stratified responses (especially disagreements and
threshold-adjacent cases).

If no human audit is performed, clearly state that label validity is
based entirely on model judges and include judge-model sensitivity
as a limitation.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
