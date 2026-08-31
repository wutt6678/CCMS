#!/usr/bin/env python3
"""LLM Judge Pipeline for Iteration 9 Evaluation.

This script runs three independent LLM judges (A, B, C) on the frozen
final panel, computes inter-judge agreement, adjudicates disagreements,
and runs the causal evaluation with adjudicated labels.

Judges:
- Judge A: Qwen3.8-Max
- Judge B: GLM-5.2
- Judge C: Qwen3.8-Max (second independent run with different seed)

All judges receive freshly randomized, blinded payloads with no
variant/family metadata visible.
"""

import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.agreement import compute_judge_agreement
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    _extract_conversation_context,
    save_human_labels,
)
from causal_mllm.evaluation.llm_judge import LLMJudgeConfig, MultimodalLLMJudge
from causal_mllm.evaluation.runner import run_evaluation_stage
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

# Paths
FINAL_PANEL_RUN = Path(
    "outputs/replay_runs/scale-b-2026-08-28-t1536-final-qwen35-9b")
VALIDATED_FAMILIES_PATH = Path(
    "outputs/families/scale_b_smoke/validated_families.jsonl")
OUTPUT_DIR = Path("outputs/llm_judge_artifacts")

# Judge configurations
JUDGE_CONFIGS = {
    "A": LLMJudgeConfig(
        model_id="qwen3.8-max",
        provider="aliyun",
        base_url="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        api_key="***REMOVED***",
        temperature=0.0,
        seed=42,
        max_retries=10,
        retry_delay=5.0,
        timeout=300.0,
    ),
    "B": LLMJudgeConfig(
        model_id="glm-5.2",
        provider="aliyun",
        base_url="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        api_key="***REMOVED***",
        temperature=0.0,
        seed=43,
        max_retries=10,
        retry_delay=5.0,
        timeout=300.0,
    ),
    "C": LLMJudgeConfig(
        model_id="qwen3.8-max",
        provider="aliyun",
        base_url="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        api_key="***REMOVED***",
        temperature=0.0,
        seed=44,
        max_retries=10,
        retry_delay=5.0,
        timeout=300.0,
    ),
}


def load_families():
    """Load validated families."""
    families = {}
    for rec in read_jsonl(VALIDATED_FAMILIES_PATH):
        fam = CausalFamily.from_dict(rec)
        families[fam.family_id] = fam
    return families


def prepare_blinded_items(families: dict, seed: int = 42) -> list[dict]:
    """Prepare blinded items for judging.

    Randomizes item order and hides variant/family metadata.
    Each item contains only the conversation context and response.

    Returns:
        List of blinded item dicts with opaque IDs.
    """
    records = read_jsonl(FINAL_PANEL_RUN / "replay_outputs.jsonl")

    # Build anonymization map
    _build_anonymization_map(seed)

    # Randomize order
    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    blinded_items = []
    for idx, rec in enumerate(shuffled_records):
        variant = rec.get("variant")
        family_id = rec.get("family_id")

        # Get family for context extraction
        family = families.get(family_id)
        if family is None:
            raise ValueError(f"family {family_id} not found")

        # Extract conversation context
        # Note: _extract_conversation_context returns empty system_prompt,
        # so we use DEFAULT_SYSTEM_PROMPT directly
        _, history_msgs, terminal_q = _extract_conversation_context(
            family, variant)
        system_prompt = DEFAULT_SYSTEM_PROMPT

        # Create opaque item ID (no family/variant info)
        item_id = f"item-{idx:04d}"

        blinded_items.append({
            "item_id": item_id,
            "family_id": family_id,  # Kept for provenance, not shown to judge
            "variant": variant,  # Kept for provenance, not shown to judge
            "system_prompt": system_prompt,
            "conversation_history": history_msgs,
            "terminal_query": terminal_q,
            "response": rec.get("response", ""),
            "response_sha256": sha256_text(rec.get("response", "")),
        })

    return blinded_items


def run_judge(
    judge: MultimodalLLMJudge,
    blinded_items: list[dict],
    output_path: Path,
) -> list[dict]:
    """Run a judge on all blinded items with checkpointing.

    Supports resuming from a previous run if interrupted.
    Saves progress after each item.

    Returns:
        List of judgment records with provenance.
    """
    # Load checkpoint if exists
    judgments = []
    start_idx = 0
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            judgments = json.load(f)
        completed_ids = {j["item_id"] for j in judgments}
        # Find where to resume
        for i, item in enumerate(blinded_items):
            if item["item_id"] not in completed_ids:
                start_idx = i
                break
        else:
            start_idx = len(blinded_items)
        print(f"  Resuming Judge {judge.judge_id} from item {start_idx+1} "
              f"({len(judgments)} already done)")

    print(f"  Running Judge {judge.judge_id} ({judge.config.model_id})...")

    for i in range(start_idx, len(blinded_items)):
        item = blinded_items[i]
        print(f"    [{i+1}/{len(blinded_items)}] {item['item_id']}", end="\r",
              flush=True)

        try:
            judgment, provenance = judge.judge(
                system_prompt=item["system_prompt"],
                history_messages=item["conversation_history"],
                terminal_query=item["terminal_query"],
                response=item["response"],
            )
        except Exception as e:
            print(f"\n    ERROR on {item['item_id']}: {e}")
            # Save checkpoint before exiting
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(judgments, f, indent=2, ensure_ascii=False)
            raise

        # Build judgment record
        rec = {
            "item_id": item["item_id"],
            "family_id": item["family_id"],
            "variant": item["variant"],
            "response_sha256": item["response_sha256"],
            "judgment": judgment,
            "provenance": {
                "backend": provenance.backend,
                "model_id": provenance.model_id,
                "provider": provenance.provider,
                "prompt_sha256": provenance.prompt_sha256,
                "rubric_sha256": provenance.rubric_sha256,
                "rubric_version": provenance.rubric_version,
                "temperature": provenance.temperature,
                "seed": provenance.seed,
                "response_hash": provenance.response_hash,
                "finish_reason": provenance.finish_reason,
                "retries": provenance.retries,
                "timestamp": provenance.timestamp,
            },
        }
        judgments.append(rec)

        # Checkpoint after each item
        if (i + 1) % 5 == 0 or i == len(blinded_items) - 1:
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(judgments, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Judge {judge.judge_id} complete ({len(judgments)} items)")

    # Save final outputs
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return judgments


def adjudicate_labels(
    judgments_a: list[dict],
    judgments_b: list[dict],
    judgments_c: list[dict],
) -> list[dict]:
    """Adjudicate disagreements between three judges.

    Uses majority voting for categorical labels and averaging for scores.

    Returns:
        List of adjudicated judgment records.
    """
    # Build lookups by item_id
    lookup_a = {j["item_id"]: j for j in judgments_a}
    lookup_b = {j["item_id"]: j for j in judgments_b}
    lookup_c = {j["item_id"]: j for j in judgments_c}

    adjudicated = []

    for item_id in sorted(lookup_a.keys()):
        ja = lookup_a[item_id]["judgment"]
        jb = lookup_b[item_id]["judgment"]
        jc = lookup_c[item_id]["judgment"]

        # Majority voting for refusal_type
        votes = [ja["refusal_type"], jb["refusal_type"], jc["refusal_type"]]
        refusal_type = max(set(votes), key=votes.count)

        # Majority voting for compliance_level
        votes = [ja["compliance_level"], jb["compliance_level"], jc["compliance_level"]]
        compliance_level = max(set(votes), key=votes.count)

        # Average scores
        scores = [
            ja["unsafe_compliance_score"],
            jb["unsafe_compliance_score"],
            jc["unsafe_compliance_score"],
        ]
        avg_score = sum(scores) / len(scores)

        # Average confidence
        confidences = [ja["confidence"], jb["confidence"], jc["confidence"]]
        avg_confidence = sum(confidences) / len(confidences)

        # Safe redirection: majority vote
        votes = [ja["safe_redirection"], jb["safe_redirection"], jc["safe_redirection"]]
        safe_redirection = sum(votes) >= 2

        # Combine rationales
        rat_a = ja["rationale"][:50]
        rat_b = jb["rationale"][:50]
        rat_c = jc["rationale"][:50]
        rationale = f"Adjudicated from 3 judges. A: {rat_a}... B: {rat_b}... C: {rat_c}..."

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
                "judge_a_model": lookup_a[item_id]["provenance"]["model_id"],
                "judge_b_model": lookup_b[item_id]["provenance"]["model_id"],
                "judge_c_model": lookup_c[item_id]["provenance"]["model_id"],
                "adjudication_method": "majority_vote_categorical_mean_score",
            },
        })

    return adjudicated


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validated families...")
    families = load_families()
    print(f"  Loaded {len(families)} families")

    # Prepare blinded items
    print("\nPreparing blinded items...")
    blinded_items = prepare_blinded_items(families, seed=42)
    print(f"  Prepared {len(blinded_items)} blinded items")

    # Save blinded items (for reproducibility)
    blinded_path = OUTPUT_DIR / "blinded_items.json"
    with blinded_path.open("w", encoding="utf-8") as f:
        json.dump(blinded_items, f, indent=2, ensure_ascii=False)
    print(f"  Saved blinded items: {blinded_path}")

    # Run three judges
    print("\nRunning LLM judges...")
    judges = {}
    all_judgments = {}

    for judge_id, config in JUDGE_CONFIGS.items():
        judge = MultimodalLLMJudge(config, judge_id=judge_id)
        judges[judge_id] = judge

        output_path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        judgments = run_judge(judge, blinded_items, output_path)
        all_judgments[judge_id] = judgments

        print(f"  Saved Judge {judge_id} labels: {output_path}")

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
    adjudicated = adjudicate_labels(
        all_judgments["A"],
        all_judgments["B"],
        all_judgments["C"],
    )

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
    from causal_mllm.evaluation.judge import HumanLabelJudge

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
    print("  - blinded_items.json: Randomized items (no variant/family metadata)")
    print("  - llm_labels_judge_A.json: Raw Judge A outputs")
    print("  - llm_labels_judge_B.json: Raw Judge B outputs")
    print("  - llm_labels_judge_C.json: Raw Judge C outputs")
    print("  - judge_agreement.json: Inter-judge agreement metrics")
    print("  - llm_labels_adjudicated.json: Adjudicated labels")
    print("  - final_evaluation_report.json: Causal evaluation results")

    print("\n" + "=" * 60)
    print("RESEARCH VALIDITY NOTE")
    print("=" * 60)
    print("""
For Iteration 9 smoke evidence, two independent LLM judges plus adjudication
is acceptable. For a paper-quality final claim, manually audit ~20-30
stratified responses (especially disagreements and threshold-adjacent cases).

If no human audit is performed, clearly state that label validity is based
entirely on model judges and include judge-model sensitivity as a limitation.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
