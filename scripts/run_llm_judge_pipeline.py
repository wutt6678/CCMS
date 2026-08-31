#!/usr/bin/env python3
"""LLM Judge Pipeline for Iteration 9 Evaluation.

This script runs two DISTINCT primary LLM judges (A, B) on the frozen
final panel, computes cross-model agreement, adjudicates disagreements
with a THIRD DISTINCT adjudicator model, and runs the causal evaluation
with the adjudicated labels.

Judges:
- Judge A (primary): qwen3.8-max
- Judge B (primary): glm-5.2
- Adjudicator: kimi-k3, reviews only the A/B disagreements from
  the original blinded context.

All judges receive freshly randomized, blinded payloads with no
variant/family metadata visible.
"""

import json
import os
import random
import sys
from pathlib import Path

# Add repo src/ to path (script lives in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_mllm.data.io import read_jsonl
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.evaluation.adjudication import (
    ENSEMBLE_BACKEND,
    LLMAdjudicator,
)
from causal_mllm.evaluation.ensemble import finalize_ensemble
from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    _extract_conversation_context,
)
from causal_mllm.evaluation.llm_judge import LLMJudgeConfig, MultimodalLLMJudge
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

# Paths
FINAL_PANEL_RUN = Path(
    "outputs/replay_runs/scale-b-2026-08-28-t1536-final-qwen35-9b")
VALIDATED_FAMILIES_PATH = Path(
    "outputs/families/scale_b_smoke/validated_families.jsonl")
OUTPUT_DIR = Path("outputs/llm_judge_artifacts")

# Path to the gitignored credentials file. See the .example template.
CREDENTIALS_FILE = (
    Path(__file__).parent.parent
    / "configs" / "evaluation" / "llm_judge_credentials.conf")


def _load_credentials_file() -> dict:
    """Load KEY=VALUE pairs from the gitignored credentials conf file.

    Returns an empty dict if the file does not exist. Lines starting with
    '#' and blank lines are ignored.
    """
    if not CREDENTIALS_FILE.exists():
        return {}
    values = {}
    for line in CREDENTIALS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


_FILE_CONFIG = _load_credentials_file()


def _cfg(name: str, default: str = "") -> str:
    """Resolve a config value: environment overrides the conf file."""
    return os.environ.get(name) or _FILE_CONFIG.get(name) or default


# API credentials: environment overrides the gitignored conf file.
# SECURITY: Never hardcode API keys. The key comes from the environment or
# the gitignored configs/evaluation/llm_judge_credentials.conf.
API_KEY = _cfg("LLM_JUDGE_API_KEY")
BASE_URL = _cfg(
    "LLM_JUDGE_BASE_URL",
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")

if not API_KEY or API_KEY == "REPLACE_WITH_ROTATED_KEY":
    raise EnvironmentError(
        "LLM_JUDGE_API_KEY is required. Set the environment variable "
        "LLM_JUDGE_API_KEY, or copy configs/evaluation/"
        "llm_judge_credentials.conf.example to llm_judge_credentials.conf "
        "and fill in the rotated key.")

# Judge architecture: TWO DISTINCT primary judges + a THIRD DISTINCT
# adjudicator model. Model IDs are configurable via environment so the
# adjudicator can be a model different from both primaries.
#
# - Primary A and Primary B must be different model families.
# - The Adjudicator reviews ONLY the items where A and B disagree, from
#   the original blinded context, and must return one coherent judgment.
# - If the adjudicator model is not distinct from both primaries, the
#   pipeline falls back to deterministic adjudication (documented as a
#   fallback, not true adjudication).
PRIMARY_A_MODEL = _cfg("LLM_JUDGE_PRIMARY_A_MODEL", "qwen3.8-max")
PRIMARY_B_MODEL = _cfg("LLM_JUDGE_PRIMARY_B_MODEL", "glm-5.2")
# Distinct adjudicator (kimi-k3 differs from both qwen3.8-max and
# glm-5.2). NOTE: the gateway also lists "kimi/kimi-k3" under /models,
# but that product is not activated (HTTP 400); use the activated
# "kimi-k3" ID.
ADJUDICATOR_MODEL = _cfg("LLM_ADJUDICATOR_MODEL", "kimi-k3")


def _make_config(model_id: str, seed: int) -> LLMJudgeConfig:
    return LLMJudgeConfig(
        model_id=model_id,
        provider="aliyun",
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.0,
        seed=seed,
        max_retries=10,
        retry_delay=5.0,
        timeout=300.0,
    )


# Two distinct primary judges
PRIMARY_JUDGE_CONFIGS = {
    "A": _make_config(PRIMARY_A_MODEL, seed=42),
    "B": _make_config(PRIMARY_B_MODEL, seed=43),
}

# Distinct adjudicator (only used on disagreements). May be None if no
# distinct model is configured, in which case deterministic fallback is used.
ADJUDICATOR_CONFIG = (
    _make_config(ADJUDICATOR_MODEL, seed=99) if ADJUDICATOR_MODEL else None)


def adjudicator_is_distinct() -> bool:
    """Return True if the adjudicator model differs from both primaries."""
    if ADJUDICATOR_CONFIG is None:
        return False
    return (ADJUDICATOR_CONFIG.model_id != PRIMARY_A_MODEL
            and ADJUDICATOR_CONFIG.model_id != PRIMARY_B_MODEL)


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
                "image_hashes": provenance.image_hashes,
                "provider_response_id": provenance.provider_response_id,
                "request_hash": provenance.request_hash,
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

    # Run the two DISTINCT primary judges
    print("\nRunning primary LLM judges (A, B)...")
    print(f"  Primary A model: {PRIMARY_A_MODEL}")
    print(f"  Primary B model: {PRIMARY_B_MODEL}")
    all_judgments = {}
    for judge_id, config in PRIMARY_JUDGE_CONFIGS.items():
        judge = MultimodalLLMJudge(config, judge_id=judge_id)
        output_path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        all_judgments[judge_id] = run_judge(judge, blinded_items, output_path)
        print(f"  Saved Judge {judge_id} labels: {output_path}")

    # Build the distinct adjudicator (or fall back deterministically).
    # The shared finalize_ensemble() drives agreement, adjudication of
    # ALL disagreements, labels, evaluation, and per-judge sensitivity.
    adjudicator = None
    if adjudicator_is_distinct():
        print(f"\nUsing distinct adjudicator model: {ADJUDICATOR_MODEL}")
        adjudicator = LLMAdjudicator(
            MultimodalLLMJudge(ADJUDICATOR_CONFIG, judge_id="ADJ"), seed=0)
    else:
        print("\nNo distinct adjudicator model configured "
              "(set LLM_ADJUDICATOR_MODEL). Using deterministic fallback.")

    print("\nFinalizing ensemble (agreement -> adjudication -> "
          "evaluation -> sensitivity)...")
    report = finalize_ensemble(
        judgments_a=all_judgments["A"],
        judgments_b=all_judgments["B"],
        blinded_items=blinded_items,
        output_dir=OUTPUT_DIR,
        run_dir=FINAL_PANEL_RUN,
        validated_families_path=VALIDATED_FAMILIES_PATH,
        adjudicator=adjudicator,
        adjudicator_model_id=ADJUDICATOR_MODEL,
        primary_model_ids=(PRIMARY_A_MODEL, PRIMARY_B_MODEL),
    )

    # Print summary
    adj = report["adjudication"]
    sens = report["judge_model_sensitivity"]
    print("\n" + "=" * 60)
    print("LLM JUDGE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print(f"Backend: {ENSEMBLE_BACKEND}")
    print(f"Adjudication: {adj['method']}")
    print(f"Disagreements adjudicated: {adj['n_disagreements']} "
          f"(field counts: {adj['disagreement_field_counts']})")
    print("Per-judge strict qualifiers at theta="
          f"{sens['theta']}: "
          f"A={sens['judges']['judge_A']['n_qualifying']}, "
          f"B={sens['judges']['judge_B']['n_qualifying']}, "
          f"ensemble={sens['judges']['ensemble']['n_qualifying']}")
    print("Qualifying under BOTH primaries: "
          f"{sens.get('qualifying_under_all_primaries', [])}")
    print("\nKey files:")
    print("  - blinded_items.json: Randomized items (no variant/family metadata)")
    print("  - llm_labels_judge_A.json: Raw primary Judge A outputs")
    print("  - llm_labels_judge_B.json: Raw primary Judge B outputs")
    print("  - judge_agreement.json: Cross-model A-B agreement metrics")
    print("  - llm_labels_adjudicator.json: Per-call adjudicator provenance")
    print("  - llm_labels_adjudicated.json: Adjudicated labels")
    print("  - judge_sensitivity.json: Per-judge causal sensitivity")
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
