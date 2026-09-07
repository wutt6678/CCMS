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
from causal_mllm.evaluation.errors import (
    EvaluationError,
    ProviderRejectedRequest,
)
from causal_mllm.evaluation.gate import validate_panel
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

#: Directory whose subdirectories are the arms of ONE cross-model comparison.
#: The four Iteration 11 profiles declare it; scale_b and scale_c do not, and
#: a profile that does not is a single arm whose exclusions are its own. See
#: :func:`load_cross_arm_panel`.
CROSS_ARM_GROUP = _SCALE_PROFILE.get("cross_arm_group")

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

#: Written by ``scripts/iter11_common_panel.py`` into ``CROSS_ARM_GROUP``:
#: the union of the cells every arm of one cross-model comparison lost.
COMMON_PANEL_ARTIFACT = "common_panel.json"

#: Written LAST on completion, binding the judgment file, the refusal sidecar
#: and the fingerprint they were both produced under. The two sidecars alone
#: cannot express "this arm finished": they are written in sequence, so a stop
#: between them leaves a completed output beside a refusal file from an earlier
#: configuration -- and since ``response_sha256`` is a property of the frozen
#: panel rather than of the run, nothing about a stale refusal file disagrees
#: with the current panel. The manifest is the one artifact whose presence
#: means the whole set was written.
MANIFEST_SUFFIX = ".manifest.json"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_completion(output_path: Path, refusals_path: Path,
                     blinded_items: list[dict], judgments: list[dict],
                     fingerprint: str,
                     sidecar_detail: dict | None = None) -> list[dict]:
    """Write the refusal sidecar and the manifest that binds it, last.

    Membership is DERIVED: a cell is unjudged iff this arm finished and has no
    judgment for it. That cannot be stale, because it is read from the very
    file the fingerprint sidecar binds. ``run_judge`` judges or refuses every
    item it is given and does nothing else with it, so absence is refusal.

    The provider's own detail -- status, code, message, body -- is carried over
    from ``sidecar_detail`` only for cells that are genuinely unjudged, and
    only when the caller established that the sidecar's fingerprint matches.
    A detail record for a cell this arm DID judge is not carried over: it
    describes a different run.

    Returns the refusal records, so the caller and the fast path agree on one
    list rather than each deriving their own.
    """
    judged_ids = {j["item_id"] for j in judgments}
    by_sha = {it["item_id"]: it["response_sha256"] for it in blinded_items}
    detail = sidecar_detail or {}
    refusals = []
    for item in blinded_items:
        item_id = item["item_id"]
        if item_id in judged_ids:
            continue
        carried = dict(detail.get(item_id) or {})
        # Defaults first so a derived record still carries the keys the
        # coverage artifact reads, then everything the run itself recorded,
        # then the fields this panel is authoritative about.
        record = {
            "status": None, "error_code": None, "error_message": None,
            "body": None,
            **carried,
            "item_id": item_id,
            "family_id": item.get("family_id"),
            "variant": item.get("variant"),
            "response_sha256": by_sha[item_id],
            "detail_source": ("refusal_sidecar" if item_id in detail
                              else "derived_from_absence"),
        }
        refusals.append(record)
    _write_refusals(refusals_path, fingerprint, refusals)
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "fingerprint": fingerprint,
            "judgments_file": output_path.name,
            "judgments_sha256": _file_sha256(output_path),
            "n_judgments": len(judgments),
            "refusals_file": refusals_path.name,
            "refusals_sha256": _file_sha256(refusals_path),
            "n_refusals": len(refusals),
            "n_panel_items": len(blinded_items),
            "rule": ("written last, after the judgment file, its fingerprint "
                     "sidecar and the refusal sidecar; its presence is what "
                     "means the set is complete rather than half-written"),
        }, f, indent=2, ensure_ascii=False)
    return refusals


def _manifest_binds(manifest_path: Path, fingerprint: str,
                    output_path: Path, refusals_path: Path) -> bool:
    """Does the manifest bind THESE files, under THIS fingerprint?"""
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(manifest, dict):
        return False
    return (manifest.get("fingerprint") == fingerprint
            and manifest.get("judgments_sha256") == _file_sha256(output_path)
            and manifest.get("refusals_sha256") == _file_sha256(refusals_path))


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
    """Which cells each primary's provider refused, derived from the run itself.

    Returns ``(by_judge, stale)``.

    MEMBERSHIP is derived, never inherited. A cell is unjudged iff the arm's
    completed output file -- the one its fingerprint sidecar binds -- carries
    no judgment for it. Reading it from the sidecar instead would trust a file
    written in sequence after the output, so a stop between the two leaves a
    completed arm beside a refusal sidecar from an earlier configuration. The
    old ``response_sha256`` comparison cannot catch that: the hash is a
    property of the FROZEN PANEL, so a stale sidecar for the same panel agrees
    with it on every cell and looks current.

    DETAIL -- the provider's status, code and message -- still comes from the
    sidecar, but only when the sidecar's stored fingerprint equals the
    completed output's. Under any other fingerprint the detail describes a
    different rubric, model or panel, and is reported in ``stale`` rather than
    used: the cell is still excluded, because this arm has no judgment for it,
    but the reason recorded is this run's own absence rather than another run's
    words.

    A sidecar entry for a cell the arm DID judge is stale by construction and
    is reported as such. Honouring it would exclude a valid cell.

    This is not read-only. Deriving membership rewrites the refusal sidecar
    under the completed output's own fingerprint and writes the completion
    manifest, so a half-written set repairs itself here rather than being
    carried into the coverage artifact. The judgments file is never touched.
    """
    by_judge, stale = {}, []
    for judge_id in judge_ids:
        output_path = output_dir / f"llm_labels_judge_{judge_id}.json"
        fingerprint_sidecar = output_path.with_name(
            output_path.name + ".fingerprint")
        refusals_path = output_path.with_suffix(REFUSALS_SUFFIX)
        if not output_path.exists() or not fingerprint_sidecar.exists():
            raise EvaluationError(
                f"judge {judge_id} has no completed output bound by a "
                f"fingerprint sidecar in {output_dir}, so which cells it "
                f"refused cannot be derived; refusing to guess")
        fingerprint = fingerprint_sidecar.read_text(
            encoding="utf-8").strip()
        with output_path.open(encoding="utf-8") as f:
            judgments = json.load(f)

        detail, sidecar_fingerprint = {}, None
        if refusals_path.exists():
            try:
                with refusals_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
            if isinstance(data, dict):
                sidecar_fingerprint = data.get("fingerprint")
                recorded = data.get("refusals", [])
                if sidecar_fingerprint == fingerprint:
                    detail = {r.get("item_id"): r for r in recorded
                              if isinstance(r, dict)}
                else:
                    stale.append({
                        "judge_id": judge_id,
                        "reason": (
                            f"the refusal sidecar records fingerprint "
                            f"{str(sidecar_fingerprint)[:16]!r} but the "
                            f"completed judgments were produced under "
                            f"{fingerprint[:16]!r}, so its contents describe "
                            f"a different run; its {len(recorded)} refusal(s) "
                            f"were not honoured"),
                        "n_refusals": len(recorded),
                    })

        by_judge[judge_id] = _bind_completion(
            output_path, refusals_path, blinded_items, judgments, fingerprint,
            sidecar_detail=detail)

        judged_ids = {j["item_id"] for j in judgments}
        for item_id in sorted(set(detail) - judged_ids - {
                r["item_id"] for r in by_judge[judge_id]}):
            stale.append({
                "judge_id": judge_id,
                "reason": "the sidecar records a refusal for a cell this arm "
                          "has no judgment for and the panel does not contain",
                **detail[item_id],
            })
        for item_id in sorted(judged_ids & set(detail)):
            stale.append({
                "judge_id": judge_id,
                "reason": "the sidecar records a refusal for a cell this arm "
                          "DID judge, so the refusal belongs to an earlier "
                          "run; the judgment stands and the cell is kept",
                **detail[item_id],
            })
    return by_judge, stale


def load_cross_arm_panel(group: str | None, output_dir: Path,
                         blinded_items: list[dict],
                         own_cells: set[tuple[str, str]]) -> dict | None:
    """The cells every arm of this comparison lost, or None for a single arm.

    ``build_judge_coverage`` unions judges A and B for ONE target session,
    which is the right rule inside a session and the wrong one across
    sessions. 11.8 places four models side by side, and a provider's
    moderation verdict is a function of the request bytes and of its policy at
    the moment of the call: the four arms reach a given cell minutes apart and
    can lose different ones. Each still reports a Delta_TV, the four are put
    in one table, and nothing in any of the artifacts says two of the numbers
    describe different families.

    So a profile that declares ``cross_arm_group`` is one arm of a comparison
    and is finalized on the union. Missing, pending or stale is an error
    rather than a warning, because finalizing anyway is exactly what produces
    four analyses that look comparable and are not -- and re-finalizing later
    is not free: the adjudicator's binding fingerprint covers the restricted
    panel, so changing it re-calls the adjudicator on every disagreement.
    """
    if not group:
        return None
    artifact_path = Path(group) / COMMON_PANEL_ARTIFACT
    if not artifact_path.exists():
        raise EvaluationError(
            f"{output_dir.name} is one arm of the cross-model comparison in "
            f"{group}, but {artifact_path} does not exist; run "
            f"scripts/iter11_common_panel.py first so this arm is restricted "
            f"to the panel every arm shares rather than to its own")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(
            f"{artifact_path} is unreadable ({exc})") from exc
    if artifact.get("status") != "derived":
        raise EvaluationError(
            f"{artifact_path} has status {artifact.get('status')!r}, not "
            f"'derived': {artifact.get('pending_reason')}. The union over the "
            f"arms is not known yet, so this arm cannot be restricted to it")
    entry = (artifact.get("per_target") or {}).get(output_dir.name)
    if entry is None:
        raise EvaluationError(
            f"{artifact_path} does not list {output_dir.name}, so it was "
            f"derived over a different set of arms than this profile "
            f"declares")
    recorded = {tuple(cell.split("/", 1))
                for cell in entry.get("excluded_cells", [])}
    if recorded != own_cells:
        raise EvaluationError(
            f"{artifact_path} was derived when {output_dir.name} had lost "
            f"{sorted('/'.join(c) for c in recorded)}, but its completed "
            f"outputs now give "
            f"{sorted('/'.join(c) for c in own_cells)}; re-run "
            f"scripts/iter11_common_panel.py so the union covers the arms as "
            f"they actually are")
    union = {tuple(cell.split("/", 1))
             for cell in artifact.get("union_excluded_cells", [])}
    if not union >= own_cells:
        raise EvaluationError(
            f"{artifact_path} unions to {sorted(union)}, which does not "
            f"contain this arm's own exclusions {sorted(own_cells)}")
    return {"cells": union, "artifact": artifact_path,
            "artifact_data": artifact}


def build_judge_coverage(by_judge: dict, stale: list,
                         blinded_items: list[dict],
                         primary_model_ids: tuple[str, str],
                         cross_arm: dict | None = None) -> tuple[dict, set]:
    """The exclusion, and the evidence that states it.

    The union over BOTH primaries is dropped from BOTH primaries. Excluding a
    cell from one arm only would leave the arms judging different panels, and
    the cross-model comparison in 11.8 is between arms, so a cell one arm lost
    has to be lost by all of them.

    ``cross_arm`` extends that rule across TARGETS: the same argument one
    level up. It is the union over every arm of the comparison, derived by
    ``scripts/iter11_common_panel.py``, and a cell only ANOTHER target lost is
    dropped here too even though this target's providers both judged it. That
    is the point, not a defect -- and the judgment is not destroyed: it stays
    in this arm's completed ``llm_labels_judge_*.json``, which the exclusion
    filter never touches. Each excluded cell records which of the two it was.

    The exclusion is outcome-independent: a provider refusal is a function of
    the request bytes alone, so the excluded set was fixed before any label
    existed and cannot have been chosen by what the cells turned out to say.
    """
    items_by_id = {it["item_id"]: it for it in blinded_items}
    cell_of_item = {it["item_id"]: (it["family_id"], it["variant"])
                    for it in blinded_items}
    item_of_cell = {cell: item_id for item_id, cell in cell_of_item.items()}
    own_ids = {r["item_id"] for rs in by_judge.values() for r in rs}
    cross_ids: set = set()
    if cross_arm is not None:
        cells_in_panel = set(cell_of_item.values())
        outside = sorted("/".join(cell) for cell in cross_arm["cells"]
                         if cell not in cells_in_panel)
        if outside:
            raise EvaluationError(
                f"the common panel excludes {outside}, which is not a cell of "
                f"this target's {len(cells_in_panel)}-cell panel; the arms are "
                f"not holding one panel, so they cannot be compared")
        cross_ids = {item_of_cell[cell] for cell in cross_arm["cells"]}
    excluded = sorted(own_ids | cross_ids)
    refused_in: dict = {}
    if cross_arm is not None:
        for name, rec in sorted(
                (cross_arm["artifact_data"].get("per_target") or {}).items()):
            for cell in rec.get("excluded_cells", []):
                refused_in.setdefault(cell, []).append(name)
    detail = []
    for item_id in excluded:
        it = items_by_id[item_id]
        entry = {
            "item_id": item_id,
            "family_id": it["family_id"],
            "variant": it["variant"],
            "response_sha256": it["response_sha256"],
            "refused_by": sorted(j for j, rs in by_judge.items()
                                 if any(r["item_id"] == item_id for r in rs)),
            "reason": next((r["error_code"] for rs in by_judge.values()
                            for r in rs if r["item_id"] == item_id), None),
        }
        if cross_arm is not None:
            cell = f"{it['family_id']}/{it['variant']}"
            entry["exclusion_origin"] = (
                "this_target" if item_id in own_ids else "another_arm")
            entry["refused_in_targets"] = sorted(refused_in.get(cell, []))
        detail.append(entry)
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
    if cross_arm is not None:
        # Added after the literal, and only on this path, so a single-arm
        # profile's coverage artifact is unchanged key for key.
        data = cross_arm["artifact_data"]
        coverage["cross_arm"] = {
            "artifact": str(cross_arm["artifact"]),
            "group": data.get("group"),
            "n_targets": data.get("n_targets"),
            "targets": data.get("targets"),
            "excluded_cells_per_target": {
                name: rec.get("excluded_cells", [])
                for name, rec in sorted(
                    (data.get("per_target") or {}).items())},
            "identical_across_targets": data.get("identical_across_targets"),
            "union_excluded_cells": sorted("/".join(cell)
                                           for cell in cross_arm["cells"]),
            "n_cells_this_target_lost": len(
                {cell_of_item[i] for i in own_ids}),
            "n_cells_added_by_the_union": len(cross_ids - own_ids),
            "rule": (
                "a cell any arm of this comparison lost is dropped from every "
                "arm, so the four per-model analyses are computed over one "
                "identical panel and can be placed side by side"),
        }
        coverage["exclusion_rule"] = (
            "the union of cells any primary of ANY TARGET in this cross-model "
            "comparison refused is dropped from EVERY arm of every target, so "
            "all of them judge one identical panel")
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
    # the current run is valid evidence — skip re-judging entirely. The
    # refusal sidecar and the manifest have to agree with it too, because the
    # three are written in sequence and a stop between them leaves a complete
    # arm beside a refusal record from an earlier configuration.
    sidecar = output_path.with_name(output_path.name + ".fingerprint")
    refusals_path = output_path.with_suffix(REFUSALS_SUFFIX)
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    if output_path.exists() and sidecar.exists():
        if sidecar.read_text(encoding="utf-8").strip() == fingerprint:
            with output_path.open(encoding="utf-8") as f:
                judgments = json.load(f)
            if not _manifest_binds(manifest_path, fingerprint, output_path,
                                   refusals_path):
                # The judgments are this run's -- the fingerprint that binds
                # them matches -- so what is missing is the binding, not the
                # evidence. Re-derive it from the gap rather than re-judging
                # 600 cells that are already on disk.
                print(f"  Judge {judge.judge_id}: complete output matches the "
                      f"current fingerprint but its refusal sidecar or "
                      f"manifest does not bind it; re-deriving the refusal "
                      f"set from the {len(judgments)} judgments present")
                detail = {}
                recorded = _read_refusals(refusals_path, fingerprint)
                if recorded:
                    detail = {r.get("item_id"): r for r in recorded}
                refusals = _bind_completion(
                    output_path, refusals_path, blinded_items, judgments,
                    fingerprint, sidecar_detail=detail)
                print(f"    rebound: {len(judgments)} judgments + "
                      f"{len(refusals)} refused = {len(blinded_items)} "
                      f"panel items")
            else:
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

    # Save final outputs, then bind them. Order matters: the manifest is
    # written LAST, so its presence is what distinguishes a finished arm from
    # one stopped between the judgment file and its refusal sidecar.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)
    sidecar.write_text(fingerprint, encoding="utf-8")
    _bind_completion(output_path, refusals_path, blinded_items, judgments,
                     fingerprint,
                     sidecar_detail={r["item_id"]: r for r in refusals})

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

    # Gate the panel BEFORE anything is spent on it.
    #
    # gate.py's contract is that "a panel that fails the gate is NEVER judged",
    # but its only call site used to be inside run_evaluation_stage, which runs
    # at the END of the ensemble. So the contract was false in practice: three
    # Iteration 11 arms each paid for two primaries and a full adjudication
    # pass -- roughly nine hours of gateway budget apiece -- and were then
    # refused at evaluation. Validating here costs one read of the replay
    # outputs and makes the contract true, which matters most in split mode,
    # where the process that would have failed is the one holding the budget.
    print("\nValidating the replay panel before any judging...")
    try:
        panel, _ = validate_panel(FINAL_PANEL_RUN,
                                  expected_n_families=len(families))
    except EvaluationError as exc:
        print(f"\nFATAL: the replay panel failed the gate, so no judge is "
              f"called and nothing is spent:\n{exc}", file=sys.stderr)
        raise SystemExit(1)
    truncation = panel.truncation
    limits = truncation["thresholds"]
    print(f"  panel gate PASSED: {panel.n_records} records over "
          f"{panel.n_families} families")
    print(f"  truncation: {truncation['n_truncated']} of "
          f"{truncation['n_records']} cell(s) reached the cap "
          f"(rate {truncation['overall_rate']:.4f} <= "
          f"{limits['max_overall_rate']}, variant spread "
          f"{truncation['max_variant_spread']:.4f} <= "
          f"{limits['max_variant_spread']})")
    for cell in truncation["cells"]:
        # Named at the point of acceptance rather than discovered later: a
        # capped cell is kept as an observation of the model, not dropped, so
        # the log has to say which ones were kept.
        print(f"    capped, kept: {cell['family_id']}/{cell['variant']} "
              f"finish_reason={cell['finish_reason']} "
              f"output_tokens={cell['output_token_count']}")

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
    items_by_id = {it["item_id"]: it for it in blinded_items}
    own_cells = {(items_by_id[r["item_id"]]["family_id"],
                  items_by_id[r["item_id"]]["variant"])
                 for rs in by_judge.values() for r in rs}
    cross_arm = load_cross_arm_panel(
        CROSS_ARM_GROUP, OUTPUT_DIR, blinded_items, own_cells)
    coverage, excluded_ids = build_judge_coverage(
        by_judge, stale, blinded_items, (PRIMARY_A_MODEL, PRIMARY_B_MODEL),
        cross_arm=cross_arm)
    with (OUTPUT_DIR / COVERAGE_ARTIFACT).open("w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"\nJudge coverage: {coverage['n_judged']}"
          f"/{coverage['n_panel_items']} cells "
          f"({OUTPUT_DIR / COVERAGE_ARTIFACT})")
    if cross_arm is not None:
        print(f"  cross-arm group {CROSS_ARM_GROUP}: this target lost "
              f"{len(own_cells)} cell(s), the union over "
              f"{cross_arm['artifact_data'].get('n_targets')} arms is "
              f"{coverage['cross_arm']['union_excluded_cells']}, adding "
              f"{coverage['cross_arm']['n_cells_added_by_the_union']} "
              f"({cross_arm['artifact']})")
    if excluded_ids:
        cells = ", ".join(
            f"{c['item_id']} ({c['family_id']}/{c['variant']}, "
            f"refused by {'+'.join(c['refused_by']) or 'ANOTHER ARM'})"
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
