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

import hashlib
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
from causal_mllm.evaluation.ensemble import (
    finalize_ensemble,
    primary_checkpoint_fingerprint,
)
from causal_mllm.evaluation.errors import ProviderRejectedRequest
from causal_mllm.evaluation.human_template import (
    _build_anonymization_map,
    _extract_conversation_context,
)
from causal_mllm.evaluation.llm_judge import LLMJudgeConfig, MultimodalLLMJudge
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT
from causal_mllm.seeds import sha256_text

# Paths — resolved from the scale profile (configs/evaluation/
# scale_profiles.json). Scale-B and Scale-C NEVER share paths; select
# with CCMS_SCALE=scale_b|scale_c (default: scale_b, the frozen panel).
SCALE = os.environ.get("CCMS_SCALE", "scale_b")
SCALE_PROFILES_FILE = (
    Path(__file__).parent.parent
    / "configs" / "evaluation" / "scale_profiles.json")


def _load_scale_profile(scale: str) -> dict:
    profiles = json.loads(SCALE_PROFILES_FILE.read_text(encoding="utf-8"))
    if scale not in profiles:
        raise EnvironmentError(
            f"unknown CCMS_SCALE={scale!r}; expected one of "
            f"{sorted(profiles)}")
    return profiles[scale]


_SCALE_PROFILE = _load_scale_profile(SCALE)
FINAL_PANEL_RUN = Path(_SCALE_PROFILE["replay_run"])
VALIDATED_FAMILIES_PATH = Path(_SCALE_PROFILE["validated_families"])
OUTPUT_DIR = Path(_SCALE_PROFILE["output_dir"])

# Path to the gitignored credentials file. See the .example template.
CREDENTIALS_FILE = (
    Path(__file__).parent.parent
    / "configs" / "evaluation" / "llm_judge_credentials.conf")

#: Measured per-identity vision capability, written by
#: ``scripts/iter11_probe_judge_gateway.py``. Read so the sensitivity artifact
#: can label a blind primary's arm as a VISION ABLATION rather than as a
#: model-choice arm -- the two claim different things, and only one of them is
#: supported when the provider discards the image.
VISION_ARTIFACT = (
    Path(__file__).parent.parent
    / "outputs" / "iteration_11" / "diagnostics" / "judge_vision"
    / "identity_vision_capability.json")


def load_identity_vision() -> dict:
    """``{model_id: measured vision verdict}``, empty when never probed.

    Absent stays absent rather than being defaulted in either direction.
    Assuming an unmeasured identity CAN see would label a blind judge's arm
    "model-choice" and claim a comparison it cannot make; assuming it cannot
    would disparage an arm that may be fine. ``finalize_ensemble`` records
    "unmeasured" and leaves the reader with the truth.
    """
    if not VISION_ARTIFACT.exists():
        print(f"  NOTE: no vision measurement at {VISION_ARTIFACT}; run "
              f"scripts/iter11_probe_judge_gateway.py so each judge arm can "
              f"be labelled on evidence rather than by assumption")
        return {}
    try:
        return json.loads(
            VISION_ARTIFACT.read_text(encoding="utf-8")).get("identities", {})
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  NOTE: unreadable vision measurement ({exc}); judge arms "
              f"will be recorded as unmeasured")
        return {}


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


def _write_checkpoint(checkpoint_path: Path, fingerprint: str,
                      judgments: list[dict]) -> None:
    """Write a fingerprint-bound primary-judge checkpoint."""
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "judgments": judgments},
                  f, indent=2, ensure_ascii=False)


#: Sidecar holding the cells a provider refused to judge. Kept out of the
#: labels because a record with no ``judgment`` key would break every consumer
#: that reads ``rec["judgment"]["refusal_type"]``.
REFUSALS_SUFFIX = ".refusals.json"

#: Written once both primaries are complete: what the ensemble actually
#: judged, and what it could not.
COVERAGE_ARTIFACT = "judge_coverage.json"


def _read_refusals(refusals_path: Path, fingerprint: str) -> list[dict]:
    """Refusals a previous run recorded under THIS fingerprint, else none.

    Fingerprint-bound exactly like the checkpoint: a refusal recorded against a
    different panel, rubric or model configuration says nothing about this one,
    and inheriting it would exclude a cell the current identity might judge
    perfectly happily.
    """
    if not refusals_path.exists():
        return []
    try:
        with refusals_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
        return []
    return list(data.get("refusals", []))


def _write_refusals(refusals_path: Path, fingerprint: str,
                    refusals: list[dict]) -> None:
    """Write the fingerprint-bound refusal sidecar."""
    with refusals_path.open("w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "refusals": refusals},
                  f, indent=2, ensure_ascii=False)


def collect_provider_refusals(output_dir: Path,
                              blinded_items: list[dict],
                              judge_ids=("A", "B"),
                              ) -> tuple[dict, list]:
    """Which cells each primary's provider refused, bound to THIS panel.

    Returns ``(by_judge, stale)``. A recorded refusal is only honoured when its
    ``response_sha256`` matches the current blinded item of the same id, so a
    sidecar left over from a different panel cannot silently shrink this one.
    """
    by_sha = {it["item_id"]: it["response_sha256"] for it in blinded_items}
    by_judge, stale = {}, []
    for judge_id in judge_ids:
        path = (output_dir / f"llm_labels_judge_{judge_id}.json"
                ).with_suffix(REFUSALS_SUFFIX)
        if not path.exists():
            by_judge[judge_id] = []
            continue
        try:
            with path.open(encoding="utf-8") as f:
                recorded = json.load(f).get("refusals", [])
        except (json.JSONDecodeError, OSError):
            recorded = []
        current = []
        for r in recorded:
            if by_sha.get(r.get("item_id")) == r.get("response_sha256"):
                current.append(r)
            else:
                stale.append({"judge_id": judge_id, **r})
        by_judge[judge_id] = current
    return by_judge, stale


def build_judge_coverage(by_judge: dict, stale: list,
                         blinded_items: list[dict],
                         primary_model_ids: tuple[str, str]) -> tuple[dict, set]:
    """The exclusion, and the evidence that states it.

    The union over BOTH primaries is dropped from BOTH primaries. Excluding a
    cell from one arm only would leave the arms judging different panels, and
    the cross-model comparison in 11.8 is between arms, so a cell one arm lost
    has to be lost by all of them.

    The exclusion is outcome-independent: a provider refusal is a function of
    the request bytes alone, so the excluded set was fixed before any label
    existed and cannot have been chosen by what the cells turned out to say.
    """
    excluded = sorted({r["item_id"] for rs in by_judge.values()
                       for r in rs})
    items_by_id = {it["item_id"]: it for it in blinded_items}
    detail = []
    for item_id in excluded:
        it = items_by_id[item_id]
        detail.append({
            "item_id": item_id,
            "family_id": it["family_id"],
            "variant": it["variant"],
            "response_sha256": it["response_sha256"],
            "refused_by": sorted(j for j, rs in by_judge.items()
                                 if any(r["item_id"] == item_id for r in rs)),
            "reason": next((r["error_code"] for rs in by_judge.values()
                            for r in rs if r["item_id"] == item_id), None),
        })
    coverage = {
        "n_panel_items": len(blinded_items),
        "n_excluded": len(excluded),
        "n_judged": len(blinded_items) - len(excluded),
        "excluded_item_ids": excluded,
        "excluded_cells": detail,
        "exclusion_rule": (
            "the union of cells any primary's provider refused is dropped "
            "from EVERY arm, so all arms judge one identical panel"),
        "outcome_independent": (
            "a provider refusal is a function of the request bytes alone, so "
            "this set was fixed before any label existed"),
        "per_judge": {
            j: {"n_refused": len(rs),
                "model_id": (primary_model_ids[0] if j == "A"
                             else primary_model_ids[1]),
                "refusals": rs}
            for j, rs in sorted(by_judge.items())},
        "stale_refusals_ignored": stale,
    }
    return coverage, set(excluded)


def run_judge(
    judge: MultimodalLLMJudge,
    blinded_items: list[dict],
    output_path: Path,
    dataset_sha256: str | None = None,
) -> list[dict]:
    """Run a judge on all blinded items with fingerprint-bound checkpoints.

    The checkpoint is bound to a fingerprint of the validated dataset
    (when provided), the panel (blinded items), rubric, and model
    configuration. A checkpoint whose fingerprint does not match the
    current run (or that predates the wrapped format) is DISCARDED and
    judging restarts from scratch — stale resume state can never
    contaminate evidence.

    A cell the provider REFUSES is recorded and skipped, not raised. Aliyun's
    input moderation rejects some of this dataset's unsafe conversation
    histories outright (HTTP 400 ``data_inspection_failed``), and the refusal
    is a property of the payload and of that model's policy: judge A refused 2
    of the frozen 600 cells while judge B and the adjudicator both accepted
    the byte-identical payloads. Letting that propagate aborted three whole
    600-item arms on one cell each. Refusals go to a
    ``.refusals.json`` sidecar rather than into the labels, because a record
    with no ``judgment`` would break every consumer that reads
    ``rec["judgment"]["refusal_type"]``. Only an input-moderation refusal is
    tolerated: an unactivated model id or an expired key refuses every cell
    identically, and recording 600 per-item refusals would bury a
    misconfiguration that must stop the run.

    Returns:
        List of judgment records with provenance. Cells the provider refused
        are absent here and present in the sidecar; see
        :func:`collect_provider_refusals`.
    """
    fingerprint = primary_checkpoint_fingerprint(
        judge, blinded_items, dataset_sha256=dataset_sha256)

    # Fast path: a completed output whose sidecar fingerprint matches
    # the current run is valid evidence — skip re-judging entirely.
    sidecar = output_path.with_name(output_path.name + ".fingerprint")
    if output_path.exists() and sidecar.exists():
        if sidecar.read_text(encoding="utf-8").strip() == fingerprint:
            with output_path.open(encoding="utf-8") as f:
                judgments = json.load(f)
            print(f"  Judge {judge.judge_id}: complete output matches "
                  f"the current fingerprint; skipping "
                  f"({len(judgments)} judgments)")
            return judgments
        print(f"  Judge {judge.judge_id}: existing output fingerprint "
              f"mismatch; re-judging from scratch")

    # Load checkpoint if it exists AND matches the current fingerprint
    judgments = []
    start_idx = 0
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    refusals_path = output_path.with_suffix(".refusals.json")
    refusals = _read_refusals(refusals_path, fingerprint)
    refusal_ids = {r["item_id"] for r in refusals}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if (isinstance(data, dict)
                and data.get("fingerprint") == fingerprint):
            judgments = data.get("judgments", [])
            # A refused cell is handled, just not judged. Counting only
            # judgments here would make a resume restart at the first refusal
            # and re-judge every cell after it.
            completed_ids = ({j["item_id"] for j in judgments} | refusal_ids)
            # Find where to resume
            for i, item in enumerate(blinded_items):
                if item["item_id"] not in completed_ids:
                    start_idx = i
                    break
            else:
                start_idx = len(blinded_items)
            print(f"  Resuming Judge {judge.judge_id} from item "
                  f"{start_idx+1} ({len(judgments)} already done"
                  f"{f', {len(refusals)} refused' if refusals else ''})")
        else:
            print(f"  Stale/mismatched checkpoint for Judge "
                  f"{judge.judge_id} ignored (fingerprint mismatch); "
                  f"restarting from item 1")

    print(f"  Running Judge {judge.judge_id} ({judge.config.model_id})...")

    for i in range(start_idx, len(blinded_items)):
        item = blinded_items[i]
        if item["item_id"] in refusal_ids:
            continue
        print(f"    [{i+1}/{len(blinded_items)}] {item['item_id']}", end="\r",
              flush=True)

        try:
            judgment, provenance = judge.judge(
                system_prompt=item["system_prompt"],
                history_messages=item["conversation_history"],
                terminal_query=item["terminal_query"],
                response=item["response"],
            )
        except ProviderRejectedRequest as e:
            if not e.is_input_moderation:
                # Not a fact about this cell, so it is not this loop's to
                # absorb: every cell would be "refused" identically.
                print(f"\n    FATAL on {item['item_id']}: {e}")
                _write_checkpoint(checkpoint_path, fingerprint, judgments)
                raise
            print(f"\n    REFUSED {item['item_id']} "
                  f"({item['family_id']}/{item['variant']}) by "
                  f"{judge.config.model_id}: HTTP {e.status} code={e.code!r} "
                  f"{e.provider_message or ''}")
            refusals.append({
                "item_id": item["item_id"],
                "family_id": item["family_id"],
                "variant": item["variant"],
                "response_sha256": item["response_sha256"],
                "judge_id": judge.judge_id,
                "model_id": judge.config.model_id,
                "status": e.status,
                "error_code": e.code,
                "error_message": e.provider_message,
                "body": e.body,
            })
            refusal_ids.add(item["item_id"])
            _write_refusals(refusals_path, fingerprint, refusals)
            continue
        except Exception as e:
            print(f"\n    ERROR on {item['item_id']}: {e}")
            # Save checkpoint before exiting
            _write_checkpoint(checkpoint_path, fingerprint, judgments)
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
                "provider_returned_model": (
                    provenance.provider_returned_model),
                "provider_system_fingerprint": (
                    provenance.provider_system_fingerprint),
                # Gateway-hosted API models cannot be revision-pinned.
                # The provider-returned model ID / system fingerprint
                # are the ONLY revision evidence available.
                "revision_pinned": False,
                "revision_note": (
                    "API-served model; exact revision cannot be pinned. "
                    "provider_returned_model and "
                    "provider_system_fingerprint are the only "
                    "revision identifiers."),
            },
        }
        judgments.append(rec)

        # Checkpoint after each item (fingerprint-bound)
        if (i + 1) % 5 == 0 or i == len(blinded_items) - 1:
            _write_checkpoint(checkpoint_path, fingerprint, judgments)

    print(f"    ✓ Judge {judge.judge_id} complete ({len(judgments)} items"
          f"{f', {len(refusals)} refused by the provider' if refusals else ''})")

    # Save final outputs + fingerprint sidecar (enables the skip path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)
    sidecar.write_text(fingerprint, encoding="utf-8")
    _write_refusals(refusals_path, fingerprint, refusals)

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

    # Run the two DISTINCT primary judges. CCMS_JUDGES (default "A,B")
    # selects a subset so A and B can run as parallel processes; the
    # fingerprint sidecar makes completed judges skip-safe on re-runs.
    selected = [j.strip().upper() for j in
                os.environ.get("CCMS_JUDGES", "A,B").split(",") if j.strip()]
    print("\nRunning primary LLM judges...")
    print(f"  Scale: {SCALE}")
    print(f"  Primary A model: {PRIMARY_A_MODEL}")
    print(f"  Primary B model: {PRIMARY_B_MODEL}")
    print(f"  Judges selected this process: {selected}")
    dataset_sha = None
    if VALIDATED_FAMILIES_PATH.exists():
        dataset_sha = hashlib.sha256(
            VALIDATED_FAMILIES_PATH.read_bytes()).hexdigest()
        print(f"  Dataset sha256: {dataset_sha[:16]}...")
    all_judgments = {}
    for judge_id, config in PRIMARY_JUDGE_CONFIGS.items():
        if judge_id not in selected:
            print(f"  Judge {judge_id} skipped (CCMS_JUDGES filter)")
            continue
        judge = MultimodalLLMJudge(config, judge_id=judge_id)
        output_path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        all_judgments[judge_id] = run_judge(
            judge, blinded_items, output_path,
            dataset_sha256=dataset_sha)
        print(f"  Saved Judge {judge_id} labels: {output_path}")

    # In split runs, finalize only when BOTH primary outputs exist.
    # Split-mode processes (CCMS_JUDGES != "A,B") NEVER finalize — the
    # final consolidation is run once, in full mode, after both judges
    # complete, to avoid concurrent adjudication/writes.
    split_mode = sorted(selected) != ["A", "B"]
    if split_mode:
        print("\nSplit mode: judging done; run the full pipeline "
              "(CCMS_JUDGES unset) to finalize once both judges are "
              "complete.")
        return
    for judge_id in PRIMARY_JUDGE_CONFIGS:
        labels_path = OUTPUT_DIR / f"llm_labels_judge_{judge_id}.json"
        if judge_id not in all_judgments:
            if not labels_path.exists():
                print(f"\nJudge {judge_id} labels missing "
                      f"({labels_path}); run that judge before "
                      f"finalizing. Stopping here.")
                return
            with labels_path.open(encoding="utf-8") as f:
                all_judgments[judge_id] = json.load(f)

    # --- provider refusals: the panel the ensemble can actually judge -------
    # compute_pairwise_agreement requires FULL mutual coverage and raises
    # without it, which is the right contract: a cell one primary could not
    # judge must not become a label from the other primary alone. So a refusal
    # is resolved by dropping that cell from every arm, not by filling it in.
    by_judge, stale = collect_provider_refusals(OUTPUT_DIR, blinded_items)
    coverage, excluded_ids = build_judge_coverage(
        by_judge, stale, blinded_items, (PRIMARY_A_MODEL, PRIMARY_B_MODEL))
    with (OUTPUT_DIR / COVERAGE_ARTIFACT).open("w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"\nJudge coverage: {coverage['n_judged']}"
          f"/{coverage['n_panel_items']} cells "
          f"({OUTPUT_DIR / COVERAGE_ARTIFACT})")
    if excluded_ids:
        cells = ", ".join(
            f"{c['item_id']} ({c['family_id']}/{c['variant']}, "
            f"refused by {'+'.join(c['refused_by'])})"
            for c in coverage["excluded_cells"])
        print(f"  excluding {len(excluded_ids)} cell(s) the provider "
              f"refused, from EVERY arm: {cells}")
        if stale:
            print(f"  ignored {len(stale)} stale refusal(s) whose "
                  f"response_sha256 is not in this panel")
        blinded_items = [it for it in blinded_items
                         if it["item_id"] not in excluded_ids]
        for judge_id in PRIMARY_JUDGE_CONFIGS:
            before = len(all_judgments[judge_id])
            all_judgments[judge_id] = [
                j for j in all_judgments[judge_id]
                if j["item_id"] not in excluded_ids]
            print(f"  judge {judge_id}: {before} -> "
                  f"{len(all_judgments[judge_id])} judgments")

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
    identity_vision = load_identity_vision()
    for model_id, measured in sorted(identity_vision.items()):
        print(f"  vision {model_id:16s} {measured.get('verdict')}")
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
        judge_coverage=coverage,
        judge_vision=identity_vision,
    )

    # Print summary
    adj = report["adjudication"]
    sens = report["judge_model_sensitivity"]
    print("\n" + "=" * 60)
    print("LLM JUDGE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print(f"Backend: {ENSEMBLE_BACKEND}")
    print(f"Cells judged: {coverage['n_judged']}"
          f"/{coverage['n_panel_items']} "
          f"({coverage['n_excluded']} refused by a provider)")
    print(f"Adjudication: {adj['method']}")
    print(f"Disagreements adjudicated: {adj['n_disagreements']} "
          f"(field counts: {adj['disagreement_field_counts']})")
    print("Per-judge strict qualifiers at theta="
          f"{sens['theta']}: "
          f"A={sens['judges']['judge_A']['n_qualifying']}, "
          f"B={sens['judges']['judge_B']['n_qualifying']}, "
          f"ensemble={sens['judges']['ensemble']['n_qualifying']}")
    for label in ("judge_A", "judge_B"):
        # judge_model_sensitivity merges judge_meta FLAT into each entry.
        entry = sens["judges"][label]
        if entry.get("arm") == "vision-ablated":
            print(f"  {label} is a VISION-ABLATION arm, not a model-choice "
                  f"arm: {entry.get('model_id')} never sees the family media")
    print("Qualifying under BOTH primaries: "
          f"{sens.get('qualifying_under_all_primaries', [])}")
    print("\nKey files:")
    print("  - blinded_items.json: Randomized items (no variant/family metadata)")
    print("  - llm_labels_judge_A.json: Raw primary Judge A outputs")
    print("  - llm_labels_judge_B.json: Raw primary Judge B outputs")
    print("  - judge_coverage.json: cells judged, and cells a provider "
          "refused (excluded from every arm)")
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
