"""Shared ensemble finalization for the Iteration 9 LLM-judge pipeline.

This module is the SINGLE post-judge workflow used by both
``scripts/run_llm_judge_pipeline.py`` (fresh judge runs) and
``scripts/resume_pipeline.py`` (resume from saved primary-judge
outputs). Keeping the logic here guarantees both entry points produce
identical evidence:

1. Cross-model agreement (primary A vs primary B).
2. Adjudication of ALL disagreements (any categorical difference or
   any score difference) by a distinct model, with per-call
   provenance persisted to ``llm_labels_adjudicator.json``; or the
   documented deterministic fallback if no adjudicator is configured.
3. Adjudicated labels with ``llm_ensemble`` provenance.
4. Causal evaluation via ``LLMEnsembleLabelJudge`` (response-SHA
   verification is fail-closed in the runner).
5. Per-judge causal sensitivity (each primary judge's raw labels
   scored independently) committed alongside the ensemble result.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from causal_mllm.evaluation.adjudication import (
    ENSEMBLE_BACKEND,
    adjudicate_deterministic,
    adjudicate_pairwise_with_model,
)
from causal_mllm.evaluation.agreement import compute_pairwise_agreement
from causal_mllm.evaluation.config import EvalConfig
from causal_mllm.evaluation.human_template import save_llm_ensemble_labels
from causal_mllm.evaluation.judge import LLMEnsembleLabelJudge
from causal_mllm.evaluation.runner import run_evaluation_stage
from causal_mllm.evaluation.sensitivity import judge_model_sensitivity

ADJUDICATOR_ARTIFACT = "llm_labels_adjudicator.json"
ADJUDICATED_LABELS_ARTIFACT = "llm_labels_adjudicated.json"
AGREEMENT_ARTIFACT = "judge_agreement.json"
SENSITIVITY_ARTIFACT = "judge_sensitivity.json"
FINAL_REPORT_ARTIFACT = "final_evaluation_report.json"

DISTINCT_MODEL_METHOD = "distinct_model_adjudication_on_all_disagreements"
FALLBACK_METHOD = "deterministic_fallback_majority_vote_coherence"


def load_adjudicator_resume(output_dir: Path) -> dict[str, dict]:
    """Load previously persisted adjudicator records for resume.

    Returns:
        Dict mapping item_id to the persisted record; empty if the
        artifact does not exist.
    """
    path = Path(output_dir) / ADJUDICATOR_ARTIFACT
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {rec["item_id"]: rec for rec in data.get("items", [])}


def _disagreement_field_counts(adjudicated: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in adjudicated:
        for field in rec.get("disagreement_fields", []):
            counts[field] = counts.get(field, 0) + 1
    return dict(sorted(counts.items()))


def _write_adjudicator_artifact(
    output_dir: Path,
    adjudicator,
    adjudicator_model_id: str,
    rubric_version: str,
    rubric_sha256: str,
    records: list[dict],
    n_resumed: int,
) -> None:
    """Persist llm_labels_adjudicator.json with per-call provenance.

    Written incrementally (after every adjudicator call) so progress
    survives interruptions. Each record carries the FULL call
    provenance: request hash, provider response ID, image hashes,
    finish reason, retries, timestamps.
    """
    judge_cfg = adjudicator.judge.config
    artifact = {
        "provenance": {
            "backend": adjudicator.judge.provenance()["backend"],
            "adjudicator_model": adjudicator_model_id,
            "provider": judge_cfg.provider,
            "temperature": judge_cfg.temperature,
            "seed": judge_cfg.seed,
            "rubric_version": rubric_version,
            "rubric_sha256": rubric_sha256,
            "adjudication_method": DISTINCT_MODEL_METHOD,
            "n_items_adjudicated": len(records),
            "n_items_resumed": n_resumed,
            "disagreement_field_counts": _disagreement_field_counts(
                records),
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        },
        "items": records,
    }
    with (Path(output_dir) / ADJUDICATOR_ARTIFACT).open(
            "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)


def finalize_ensemble(
    judgments_a: list[dict],
    judgments_b: list[dict],
    blinded_items: list[dict],
    output_dir: Path,
    run_dir: Path,
    validated_families_path: Path,
    adjudicator=None,
    adjudicator_model_id: str = "",
    primary_model_ids: tuple[str, str] = ("", ""),
    eval_config: EvalConfig | None = None,
) -> dict:
    """Run the complete post-judge ensemble workflow.

    Args:
        judgments_a, judgments_b: Primary judge records (item_id,
            family_id, variant, response_sha256, judgment, provenance).
        blinded_items: Blinded item dicts keyed by item_id (original
            context for the adjudicator).
        output_dir: Artifact directory (outputs/llm_judge_artifacts).
        run_dir: Frozen replay run directory.
        validated_families_path: Path to validated_families.jsonl.
        adjudicator: Optional LLMAdjudicator whose model differs from
            both primaries. If None, the deterministic fallback is used.
        adjudicator_model_id: Model ID of the adjudicator (provenance).
        primary_model_ids: Tuple of (primary A, primary B) model IDs.
        eval_config: Evaluation config (defaults to standard Iteration 9).

    Returns:
        The final evaluation report dict (with judge_model_sensitivity).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_config is None:
        eval_config = EvalConfig(n_bootstrap=5000, seed=42)

    # 1. Cross-model agreement (A-B)
    agreement = compute_pairwise_agreement(judgments_a, judgments_b)
    with (output_dir / AGREEMENT_ARTIFACT).open(
            "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)

    # Rubric provenance comes from the primary judges (not hardcoded).
    rubric_version = judgments_a[0]["provenance"].get(
        "rubric_version", "1.1")
    rubric_sha256 = judgments_a[0]["provenance"].get(
        "rubric_sha256", "")

    # 2. Adjudicate ALL disagreements
    items_by_id = {it["item_id"]: it for it in blinded_items}
    lookup_a = {j["item_id"]: j for j in judgments_a}

    if adjudicator is not None:
        resume_records = load_adjudicator_resume(output_dir)

        # Checkpoint the adjudicator artifact after EVERY disagreement
        # record so an interrupted run never loses completed calls.
        _live: dict = {"items": [], "resumed": 0}

        def _on_record(record: dict, resumed: bool) -> None:
            _live["items"].append(record)
            if resumed:
                _live["resumed"] += 1
            _write_adjudicator_artifact(
                output_dir, adjudicator, adjudicator_model_id,
                rubric_version, rubric_sha256,
                _live["items"], _live["resumed"])

        adjudicated, disagreement_ids, adjudicator_records = \
            adjudicate_pairwise_with_model(
                adjudicator, judgments_a, judgments_b, items_by_id,
                resume_records=resume_records, on_record=_on_record)
        adjudication_method = DISTINCT_MODEL_METHOD

        # Final artifact write (idempotent; matches the last checkpoint).
        _write_adjudicator_artifact(
            output_dir, adjudicator, adjudicator_model_id,
            rubric_version, rubric_sha256, adjudicator_records,
            _live["resumed"])
    else:
        # Documented deterministic fallback (no distinct model).
        judgments_by_item: dict[str, list[dict]] = {}
        for rec in judgments_a + judgments_b:
            judgments_by_item.setdefault(
                rec["item_id"], []).append(rec["judgment"])
        core, disagreement_ids = adjudicate_deterministic(
            judgments_by_item)
        adjudicated = []
        for rec in core:
            source = lookup_a[rec["item_id"]]
            adjudicated.append({
                "item_id": rec["item_id"],
                "family_id": source["family_id"],
                "variant": source["variant"],
                "response_sha256": source["response_sha256"],
                "judgment": rec["judgment"],
                "is_disagreement": rec["is_disagreement"],
                "disagreement_fields": [],
                "adjudicated_by": "deterministic_fallback",
            })
        adjudication_method = FALLBACK_METHOD

    field_counts = _disagreement_field_counts(adjudicated)

    # 3. Labels keyed by family/variant
    adjudicated_labels = {}
    for rec in adjudicated:
        adjudicated_labels.setdefault(
            rec["family_id"], {})[rec["variant"]] = {
            **rec["judgment"],
            "response_sha256": rec["response_sha256"],
            "rubric_version": rubric_version,
            "annotator_id": "llm_ensemble",
            "adjudicated": True,
            "item_id": rec["item_id"],
        }

    ensemble_provenance = {
        "backend": ENSEMBLE_BACKEND,
        "judge_models": {
            "A": primary_model_ids[0],
            "B": primary_model_ids[1],
        },
        "adjudicator_model": (
            adjudicator_model_id if adjudicator is not None else None),
        "adjudication_method": adjudication_method,
        "n_disagreements": len(disagreement_ids),
        "disagreement_field_counts": field_counts,
        "adjudicator_artifact": (
            ADJUDICATOR_ARTIFACT if adjudicator is not None else None),
        "note": ("" if adjudicator is not None else
                 "Fallback adjudication; no distinct adjudicator "
                 "model was configured. Results are provisional."),
    }

    labels_path = output_dir / ADJUDICATED_LABELS_ARTIFACT
    save_llm_ensemble_labels(
        adjudicated_labels,
        labels_path,
        ensemble_provenance=ensemble_provenance,
        rubric_version=rubric_version,
        rubric_sha256=rubric_sha256,
    )

    # 4. Causal evaluation (response-SHA gate is fail-closed inside)
    judge = LLMEnsembleLabelJudge(labels_path)
    report = run_evaluation_stage(
        run_dir=Path(run_dir),
        judge=judge,
        config=eval_config,
        output_root=output_dir / "evaluation_results",
        validated_families_path=Path(validated_families_path),
    )

    # 5. Per-judge causal sensitivity
    sensitivity = judge_model_sensitivity(
        {
            "judge_A": judgments_a,
            "judge_B": judgments_b,
            "ensemble": adjudicated,
        },
        theta=eval_config.theta,
        judge_meta={
            "judge_A": {"model_id": primary_model_ids[0]},
            "judge_B": {"model_id": primary_model_ids[1]},
            "ensemble": {
                "model_id": (
                    f"ensemble({primary_model_ids[0]}, "
                    f"{primary_model_ids[1]}"
                    + (f", adjudicator={adjudicator_model_id})"
                       if adjudicator is not None else ")")),
                "adjudication_method": adjudication_method,
            },
        },
        primary_judge_ids=("judge_A", "judge_B"),
    )
    with (output_dir / SENSITIVITY_ARTIFACT).open(
            "w", encoding="utf-8") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)

    report["judge_model_sensitivity"] = sensitivity
    report["adjudication"] = {
        "method": adjudication_method,
        "n_disagreements": len(disagreement_ids),
        "disagreement_item_ids": disagreement_ids,
        "disagreement_field_counts": field_counts,
        "adjudicator_model": (
            adjudicator_model_id if adjudicator is not None else None),
    }

    report_path = output_dir / FINAL_REPORT_ARTIFACT
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
