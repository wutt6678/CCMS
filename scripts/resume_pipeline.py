#!/usr/bin/env python3
"""Resume the ensemble pipeline from SAVED primary-judge outputs.

This script reuses the committed Judge A/B outputs and blinded items
(it does NOT re-run the primaries) and then runs the SAME shared
post-judge workflow as ``run_llm_judge_pipeline.py``
(``causal_mllm.evaluation.ensemble.finalize_ensemble``):

1. Cross-model agreement (A-B).
2. Adjudication of ALL A/B disagreements (any categorical or score
   difference) by the distinct adjudicator model. Previously persisted
   adjudicator calls (llm_labels_adjudicator.json) are reused so an
   interrupted adjudication phase can resume.
3. Adjudicated labels (llm_ensemble backend) + causal evaluation with
   fail-closed response-SHA verification.
4. Per-judge causal sensitivity committed alongside the ensemble result.

Both entry points share one implementation, so they always produce
identical evidence.
"""

import json
import sys
from pathlib import Path

# Add repo src/ and scripts/ to path (script lives in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the pipeline's credential/config resolution (single source).
from run_llm_judge_pipeline import (  # noqa: E402
    ADJUDICATOR_CONFIG,
    ADJUDICATOR_MODEL,
    FINAL_PANEL_RUN,
    OUTPUT_DIR,
    PRIMARY_A_MODEL,
    PRIMARY_B_MODEL,
    VALIDATED_FAMILIES_PATH,
    adjudicator_is_distinct,
)

from causal_mllm.evaluation.adjudication import LLMAdjudicator
from causal_mllm.evaluation.ensemble import finalize_ensemble
from causal_mllm.evaluation.llm_judge import MultimodalLLMJudge

EXPECTED_ITEMS = 120


def _load_judge_outputs() -> tuple[list[dict], list[dict]]:
    """Load the saved primary judge outputs (A and B only)."""
    outputs = {}
    for judge_id in ("A", "B"):
        path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing saved judge output: {path}. Run "
                f"run_llm_judge_pipeline.py first.")
        with path.open(encoding="utf-8") as f:
            outputs[judge_id] = json.load(f)
        if len(outputs[judge_id]) != EXPECTED_ITEMS:
            raise ValueError(
                f"judge {judge_id} has {len(outputs[judge_id])} items, "
                f"expected {EXPECTED_ITEMS}")
        print(f"  Loaded Judge {judge_id}: {len(outputs[judge_id])} items "
              f"({outputs[judge_id][0]['provenance']['model_id']})")
    return outputs["A"], outputs["B"]


def _load_blinded_items() -> list[dict]:
    """Load the committed blinded items (original adjudication context)."""
    path = OUTPUT_DIR / "blinded_items.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing blinded items: {path}. Run "
            f"run_llm_judge_pipeline.py first.")
    with path.open(encoding="utf-8") as f:
        items = json.load(f)
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(
            f"blinded items has {len(items)} items, "
            f"expected {EXPECTED_ITEMS}")
    return items


def main():
    """Resume from saved judge outputs."""
    print("Loading saved primary judge outputs (A, B)...")
    judgments_a, judgments_b = _load_judge_outputs()
    blinded_items = _load_blinded_items()
    print(f"  Loaded blinded items: {len(blinded_items)}")

    # Build the distinct adjudicator (or fall back deterministically).
    adjudicator = None
    if adjudicator_is_distinct():
        print(f"Using distinct adjudicator model: {ADJUDICATOR_MODEL}")
        adjudicator = LLMAdjudicator(
            MultimodalLLMJudge(ADJUDICATOR_CONFIG, judge_id="ADJ"), seed=0)
    else:
        print("No distinct adjudicator model configured "
              "(set LLM_ADJUDICATOR_MODEL). Using deterministic fallback.")

    print("\nFinalizing ensemble (agreement -> adjudication -> "
          "evaluation -> sensitivity)...")
    report = finalize_ensemble(
        judgments_a=judgments_a,
        judgments_b=judgments_b,
        blinded_items=blinded_items,
        output_dir=OUTPUT_DIR,
        run_dir=FINAL_PANEL_RUN,
        validated_families_path=VALIDATED_FAMILIES_PATH,
        adjudicator=adjudicator,
        adjudicator_model_id=ADJUDICATOR_MODEL,
        primary_model_ids=(PRIMARY_A_MODEL, PRIMARY_B_MODEL),
    )

    adj = report["adjudication"]
    sens = report["judge_model_sensitivity"]
    print("\n" + "=" * 60)
    print("RESUMED ENSEMBLE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
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


if __name__ == "__main__":
    main()
