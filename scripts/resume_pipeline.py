#!/usr/bin/env python3
"""Resume pipeline from saved judge outputs.

This script picks up after the 3 judges have completed, computing
agreement, adjudicating, and running the causal evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from causal_mllm.evaluation.adjudication import (
    ENSEMBLE_BACKEND,
    adjudicate_deterministic,
)
from causal_mllm.evaluation.agreement import compute_judge_agreement
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.human_template import save_llm_ensemble_labels
from causal_mllm.evaluation.judge import LLMEnsembleLabelJudge
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
    print(f"  ICC(3,1) (score): {agreement['icc_score']['ICC(3,1)']:.4f}")
    print(f"  ICC(3,k) (score): {agreement['icc_score']['ICC(3,k)']:.4f}")
    print(f"  Spearman ρ (score) mean: "
          f"{agreement['spearman_rho']['mean']:.4f}")

    # Adjudicate using the SHARED deterministic fallback (same logic as
    # run_llm_judge_pipeline.py so both entry points give identical results).
    print("\nAdjudicating disagreements (shared deterministic fallback)...")

    lookup_a = {j["item_id"]: j for j in all_judgments["A"]}

    # Extract the rubric version/hash from judge provenance (not hardcoded)
    rubric_version = all_judgments["A"][0]["provenance"].get(
        "rubric_version", "1.1")
    rubric_sha256 = all_judgments["A"][0]["provenance"].get(
        "rubric_sha256", "")

    # Group judgments by item_id for the shared adjudicator
    judgments_by_item: dict[str, list[dict]] = {}
    for judge_key in ("A", "B", "C"):
        for rec in all_judgments[judge_key]:
            judgments_by_item.setdefault(rec["item_id"], []).append(
                rec["judgment"])

    adjudicated_core, disagreement_ids = adjudicate_deterministic(
        judgments_by_item)
    print(f"  Found {len(disagreement_ids)} disagreements "
          f"out of {len(adjudicated_core)} items")

    adjudicated = []
    for rec in adjudicated_core:
        item_id = rec["item_id"]
        source = lookup_a[item_id]
        adjudicated.append({
            "item_id": item_id,
            "family_id": source["family_id"],
            "variant": source["variant"],
            "response_sha256": source["response_sha256"],
            "judgment": rec["judgment"],
            "provenance": {
                "backend": ENSEMBLE_BACKEND,
                "judge_a_model": all_judgments["A"][0][
                    "provenance"]["model_id"],
                "judge_b_model": all_judgments["B"][0][
                    "provenance"]["model_id"],
                "judge_c_model": all_judgments["C"][0][
                    "provenance"]["model_id"],
                "adjudication_method": (
                    "deterministic_fallback_majority_vote_coherence"),
                "is_disagreement": rec["is_disagreement"],
                "note": "Fallback adjudication; not a distinct-model review.",
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
            "rubric_version": rubric_version,
            "annotator_id": "llm_judge_adjudicated",
            "adjudicated": True,
            "item_id": rec["item_id"],
        }

    # Build ensemble provenance for the label file
    ensemble_provenance = {
        "backend": ENSEMBLE_BACKEND,
        "judge_models": {
            "A": all_judgments["A"][0]["provenance"]["model_id"],
            "B": all_judgments["B"][0]["provenance"]["model_id"],
            "C": all_judgments["C"][0]["provenance"]["model_id"],
        },
        "adjudication_method": (
            "deterministic_fallback_majority_vote_coherence"),
        "n_disagreements": len(disagreement_ids),
        "note": ("Fallback adjudication (majority vote + coherence repair). "
                 "A and C share the Qwen model; for research-grade results "
                 "use a distinct adjudicator model on disagreements."),
    }

    # Save adjudicated labels with LLM-ensemble provenance
    labels_path = OUTPUT_DIR / "llm_labels_adjudicated.json"
    save_llm_ensemble_labels(
        adjudicated_labels,
        labels_path,
        ensemble_provenance=ensemble_provenance,
        rubric_version=rubric_version,
        rubric_sha256=rubric_sha256,
    )
    print(f"  Saved adjudicated labels: {labels_path}")

    # Run causal evaluation with the LLM-ensemble judge
    print("\nRunning causal evaluation with adjudicated labels...")
    judge = LLMEnsembleLabelJudge(labels_path)

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
