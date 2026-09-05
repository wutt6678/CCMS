#!/usr/bin/env python3
"""Iteration 11.7 — measure judge B's blindness instead of assuming its effect.

Probed on 2026-09-05: glm-5.2, the frozen primary judge B, cannot see images
on this gateway. It answers NO_IMAGE on three different images, its
prompt_tokens are identical with and without an image attached, and it reports
no image_tokens — while qwen3.8-max (judge A) and kimi-k3 (adjudicator) both
describe the images correctly and report image_tokens. Every other GLM id is
blind too, and deepseek-v3.2 is blind but CONFABULATES plausible descriptions,
so silence is not evidence of sight.

An audit of the frozen Iteration 10 labels established that this did not
contaminate the adjudicated labels: of 600 items, 361 agreements resolve to
judge A's judgment and all 239 disagreements resolve to the adjudicator's, and
B never unilaterally determines a label. The adjudicator received the image on
all 151 image-bearing disagreements. So the reference labels the 11.5
stratification is derived from are vision-informed throughout.

What remains unmeasured is WHY A and B disagreed on 50.3% of image-bearing
cells against 29.3% of image-free ones. Two explanations predict different
things about 11.7:

  * blindness — B dissented because it was missing an input A had. Then a
    vision-capable B would agree with A far more often, the ensemble would
    need less adjudication, and the judge_B sensitivity arm is really a
    vision-ablation arm mislabelled as a model-choice arm;
  * genuine model difference — B would dissent just as often sighted. Then the
    frozen ensemble is behaving as designed and only its provenance wording
    needs correcting.

This script separates them by holding the model IDENTITY fixed and removing
only the image: it re-judges the 300 image-bearing frozen items with judge A's
exact identity (qwen3.8-max, rubric v1.1, temperature 0.0, seed 42) and the
image payload withheld. The text prompt is byte-identical to the one A
received, including the " [Image: <name>]" markers — which is precisely the
information set B had — and the script ASSERTS that per item by comparing
prompt_sha256 against the frozen A record. Only request_hash differs, because
it binds the image hashes.

Writes only under outputs/iteration_11/judge_vision_ablation/. It never
modifies outputs/scale_c, whose evidence is sealed by
scale_c_closeout_manifest.py. Resumable: completed items are skipped.

Usage:
    python3 scripts/iter11_judge_vision_ablation.py --limit 5   # trial
    python3 scripts/iter11_judge_vision_ablation.py             # all 300
    python3 scripts/iter11_judge_vision_ablation.py --analyse-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.adjudication import judgments_disagree  # noqa: E402
from causal_mllm.evaluation.llm_judge import LLMJudgeConfig, MultimodalLLMJudge  # noqa: E402

SCALE_C = REPO_ROOT / "outputs" / "scale_c" / "llm_judge_artifacts"
OUT_DIR = REPO_ROOT / "outputs" / "iteration_11" / "judge_vision_ablation"
RUBRIC = REPO_ROOT / "src" / "causal_mllm" / "evaluation" \
    / "annotation_rubric_v1_1.md"

#: The frozen Iteration 10 rubric and judge-A decoding parameters. The
#: ablation must match these exactly or it measures the wrong difference.
FROZEN_RUBRIC_SHA256 = ("ce6c20057b3b84d3c868908062c605383cfb42a9"
                        "eac5243e529aacb2d0bfbd0c")
FROZEN_JUDGE_A_MODEL = "qwen3.8-max"
FROZEN_JUDGE_B_MODEL = "glm-5.2"
FROZEN_TEMPERATURE = 0.0
FROZEN_SEED = 42

CREDENTIALS = REPO_ROOT / "configs" / "evaluation" \
    / "llm_judge_credentials.conf"
DEFAULT_BASE_URL = (
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com"
    "/compatible-mode/v1")

MAX_WORKERS = 4

#: A 400 from the gateway is a client error: the request itself was refused,
#: so the judge's four retries cannot succeed and the item is recorded as an
#: exclusion instead. Matched against the rendered exception because the judge
#: propagates ``raise_for_status`` text, which carries the status and URL but
#: not the response body.
HTTP_400 = re.compile(r"\b400\b")


class VisionAblatedJudge(MultimodalLLMJudge):
    """The production judge with the image payload withheld.

    ``_build_prompt`` is inherited unchanged, so the text prompt — rubric,
    system prompt, rendered history including the " [Image: <name>]" markers,
    terminal query and response — is byte-identical to what judge A received.
    Only the returned image blocks and hashes are emptied. That reproduces
    judge B's information set under judge A's model identity, which is the
    contrast that separates blindness from model difference.

    The markers are deliberately KEPT. Removing them would produce a cleaner
    looking ablation that no longer matches what B actually saw: B was told an
    image was present and was not given one.
    """

    def _build_prompt(self, system_prompt, history_messages, terminal_query,
                      response):
        prompt, _images, _hashes = super()._build_prompt(
            system_prompt, history_messages, terminal_query, response)
        return prompt, [], []


def load_credentials() -> dict:
    """Environment overrides the gitignored conf file, as in the pipeline."""
    import os
    values: dict = {}
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")

    def cfg(name, default=""):
        return os.environ.get(name) or values.get(name) or default

    api_key = cfg("LLM_JUDGE_API_KEY")
    if not api_key or api_key == "REPLACE_WITH_ROTATED_KEY":
        raise SystemExit(f"no LLM_JUDGE_API_KEY resolved (environment or "
                         f"{CREDENTIALS})")
    return {"api_key": api_key,
            "base_url": cfg("LLM_JUDGE_BASE_URL", DEFAULT_BASE_URL)}


def frozen_labels():
    """Judge A, judge B and the frozen blinded items, keyed by item_id."""
    a_list = json.loads((SCALE_C / "llm_labels_judge_A.json")
                        .read_text(encoding="utf-8"))
    b_list = json.loads((SCALE_C / "llm_labels_judge_B.json")
                        .read_text(encoding="utf-8"))
    items = json.loads((SCALE_C / "blinded_items.json")
                       .read_text(encoding="utf-8"))
    return ({r["item_id"]: r for r in a_list},
            {r["item_id"]: r for r in b_list},
            {r["item_id"]: r for r in items})


def image_bearing(a_records) -> list[str]:
    """Item ids whose frozen judge-A request carried at least one image."""
    return sorted(i for i, r in a_records.items()
                  if r["provenance"].get("image_hashes"))


def load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A torn final line from an interrupted run is not fatal: the
            # item is simply re-judged.
            continue
        done[record["item_id"]] = record
    return done


def run_ablation(limit: int | None) -> int:
    rubric_sha = hashlib.sha256(RUBRIC.read_bytes()).hexdigest()
    if rubric_sha != FROZEN_RUBRIC_SHA256:
        raise SystemExit(
            f"rubric digest {rubric_sha} is not the frozen Iteration 10 "
            f"rubric {FROZEN_RUBRIC_SHA256}; the ablation would not be "
            f"comparable to the labels it is contrasted against")

    a_records, b_records, items = frozen_labels()
    # The frozen judge-A records must themselves name the expected identity
    # and decoding parameters, or "hold identity fixed" is an assumption.
    identities = {r["provenance"]["model_id"] for r in a_records.values()}
    if identities != {FROZEN_JUDGE_A_MODEL}:
        raise SystemExit(f"frozen judge A model ids are {identities}, "
                         f"expected {{{FROZEN_JUDGE_A_MODEL!r}}}")
    seeds = {r["provenance"]["seed"] for r in a_records.values()}
    temps = {r["provenance"]["temperature"] for r in a_records.values()}
    if seeds != {FROZEN_SEED} or temps != {FROZEN_TEMPERATURE}:
        raise SystemExit(f"frozen judge A decoding is seeds={seeds} "
                         f"temperatures={temps}, expected "
                         f"seed={FROZEN_SEED} temperature={FROZEN_TEMPERATURE}")

    targets = image_bearing(a_records)
    if limit is not None:
        targets = targets[:limit]
    print(f"rubric           v1.1 {rubric_sha[:16]}… (matches frozen)")
    print(f"ablated identity {FROZEN_JUDGE_A_MODEL} temp={FROZEN_TEMPERATURE} "
          f"seed={FROZEN_SEED}")
    print(f"image-bearing frozen items: {len(image_bearing(a_records))}"
          f"; running {len(targets)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    journal = OUT_DIR / "ablated_labels.jsonl"
    done = load_done(journal)
    todo = [i for i in targets if i not in done]
    print(f"already judged   : {len(done)}")
    print(f"to judge         : {len(todo)}")

    if todo:
        credentials = load_credentials()
        judge = VisionAblatedJudge(
            LLMJudgeConfig(
                model_id=FROZEN_JUDGE_A_MODEL, provider="aliyun",
                base_url=credentials["base_url"],
                api_key=credentials["api_key"],
                temperature=FROZEN_TEMPERATURE, seed=FROZEN_SEED,
                timeout=120.0),
            rubric_path=RUBRIC, judge_id="A_blind")

        lock = threading.Lock()
        written = Counter()
        errors: list[str] = []
        exclusions: list[str] = []

        def judge_one(item_id: str) -> None:
            item = items[item_id]
            try:
                judgment, provenance = judge.judge(
                    system_prompt=item["system_prompt"],
                    history_messages=item["conversation_history"],
                    terminal_query=item["terminal_query"],
                    response=item["response"])
            except Exception as exc:
                # The gateway moderates the INPUT text, and its verdict is not
                # invariant to whether an image is attached: item-0410 was
                # judged successfully by frozen judge A with its image, and is
                # rejected with 400 data_inspection_failed once the image is
                # withheld. A 400 is a client error, so retrying it four times
                # only burns the budget and cannot succeed; record it as an
                # exclusion and analyse the rest. Anything else (transport,
                # 5xx, malformed output) stays retryable.
                detail = f"{type(exc).__name__}: {exc}"
                moderation = "data_inspection_failed" in detail
                if not (moderation or HTTP_400.search(detail)):
                    with lock:
                        errors.append(f"{item_id}: {detail}")
                    return
                tombstone = {
                    "item_id": item_id,
                    "family_id": item["family_id"],
                    "variant": item["variant"],
                    "response_sha256": item["response_sha256"],
                    "judgment": None,
                    "ablation": "image_payload_withheld",
                    "excluded": ("provider_input_moderation" if moderation
                                 else "provider_rejected_request_http_400"),
                    "exclusion_detail": detail[:400],
                    "frozen_judge_a_prompt_sha256":
                        a_records[item_id]["provenance"]["prompt_sha256"],
                }
                with lock:
                    with journal.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(tombstone,
                                                ensure_ascii=False) + "\n")
                    exclusions.append(item_id)
                return
            frozen_prompt = a_records[item_id]["provenance"]["prompt_sha256"]
            record = {
                "item_id": item_id,
                "family_id": item["family_id"],
                "variant": item["variant"],
                "response_sha256": item["response_sha256"],
                "judgment": judgment,
                "ablation": "image_payload_withheld",
                "prompt_sha256_matches_frozen_judge_a":
                    provenance.prompt_sha256 == frozen_prompt,
                "frozen_judge_a_prompt_sha256": frozen_prompt,
                "n_images_withheld":
                    len(a_records[item_id]["provenance"]["image_hashes"]),
                "call_provenance": {
                    "model_id": provenance.model_id,
                    "provider_returned_model":
                        provenance.provider_returned_model,
                    "provider_system_fingerprint":
                        provenance.provider_system_fingerprint,
                    "prompt_sha256": provenance.prompt_sha256,
                    "request_hash": provenance.request_hash,
                    "rubric_sha256": provenance.rubric_sha256,
                    "rubric_version": provenance.rubric_version,
                    "temperature": provenance.temperature,
                    "seed": provenance.seed,
                    "image_hashes": provenance.image_hashes,
                    "finish_reason": provenance.finish_reason,
                    "retries": provenance.retries,
                    "response_hash": provenance.response_hash,
                    "provider_response_id": provenance.provider_response_id,
                    "timestamp": provenance.timestamp,
                },
            }
            with lock:
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written["n"] += 1
                if written["n"] % 10 == 0:
                    print(f"    {written['n']}/{len(todo)} judged", flush=True)

        print(f"judging with {MAX_WORKERS} workers …")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(judge_one, todo))
        if errors:
            print(f"\n{len(errors)} item(s) failed:")
            for error in errors[:10]:
                print(f"  - {error}")
            print("re-run to retry them; completed items are skipped")
            return 1
        if exclusions:
            print(f"\n{len(exclusions)} item(s) excluded by provider input "
                  f"moderation (not retryable): {exclusions}")

    done = load_done(journal)
    print(f"\njournal holds {len(done)} ablated judgments")
    return analyse(a_records, b_records, done, targets)


def analyse(a_records, b_records, ablated, targets) -> int:
    """Compare the three pairs of label sets on the same items."""
    # Items the provider's input moderation refused once the image was
    # withheld carry no judgment. They are dropped from EVERY arm so all three
    # rates share one denominator; reporting the frozen A/B rate over 300
    # items against an ablated rate over 299 would not be a comparison.
    excluded = sorted(i for i, r in ablated.items()
                      if r.get("judgment") is None)
    excluded_set = set(excluded)
    common = [i for i in targets if i in ablated and i not in excluded_set]
    if not common:
        print("nothing to analyse yet")
        return 0

    # The ablation is only a vision ablation if the text prompt is provably
    # unchanged. Refuse to report attribution numbers otherwise.
    bad_prompt = [i for i in common
                  if not ablated[i]["prompt_sha256_matches_frozen_judge_a"]]
    served = Counter(ablated[i]["call_provenance"]["provider_returned_model"]
                     for i in common)
    withheld = Counter(ablated[i]["call_provenance"]["image_hashes"] == []
                       for i in common)
    print("\n=== ablation integrity ===")
    print(f"  items analysed                 : {len(common)}")
    print(f"  excluded by input moderation   : {len(excluded)}"
          + (f" {excluded}" if excluded else ""))
    print(f"  prompt_sha256 == frozen judge A : "
          f"{len(common) - len(bad_prompt)} / {len(common)}")
    print(f"  served as                      : {dict(served)}")
    print(f"  image_hashes empty (withheld)  : {dict(withheld)}")
    if bad_prompt:
        print(f"  !! {len(bad_prompt)} item(s) did NOT reproduce judge A's "
              f"text prompt, so the contrast is not vision-only: "
              f"{bad_prompt[:10]}")
        return 1
    if served != {FROZEN_JUDGE_A_MODEL: len(common)}:
        print(f"  !! the gateway served {dict(served)}, not "
              f"{FROZEN_JUDGE_A_MODEL} throughout")

    # The production routing predicate, not a hand-rolled field comparison:
    # the disagreement counts this reports must be computed the same way the
    # frozen 239 adjudications were, or the rates are not comparable.
    def differing(x, y):
        return judgments_disagree([x["judgment"], y["judgment"]])

    pairs = {
        "A vs B (frozen, B blind)":
            lambda i: differing(a_records[i], b_records[i]),
        "A_blind vs B (both blind)":
            lambda i: differing(ablated[i], b_records[i]),
        "A_blind vs A (vision effect on A)":
            lambda i: differing(ablated[i], a_records[i]),
    }

    print("\n=== disagreement on the image-bearing items ===")
    rates = {}
    for label, fn in pairs.items():
        dis = [i for i in common if fn(i)]
        fields = Counter(f for i in dis for f in fn(i))
        rates[label] = len(dis) / len(common)
        print(f"  {label:38s} {len(dis):3d}/{len(common)} "
              f"= {len(dis) / len(common):.3f}  fields={dict(fields)}")

    print("\n  per variant:")
    by_variant = defaultdict(list)
    for i in common:
        by_variant[a_records[i]["variant"]].append(i)
    for variant in sorted(by_variant):
        group = by_variant[variant]
        row = "  ".join(
            f"{label.split(' (')[0]}="
            f"{sum(1 for i in group if fn(i)) / len(group):.2f}"
            for label, fn in pairs.items())
        print(f"    {variant:15s} n={len(group):3d}  {row}")

    print("\n  compliance_level shift caused by withholding the image "
          "(A_blind - A):")
    deltas = Counter(ablated[i]["judgment"]["compliance_level"]
                     - a_records[i]["judgment"]["compliance_level"]
                     for i in common)
    for delta in sorted(deltas):
        print(f"    {delta:+d}: {deltas[delta]}")
    mean = (sum(k * v for k, v in deltas.items()) / (sum(deltas.values()) or 1))
    print(f"    mean {mean:+.3f}")

    # Attribution: how much of the frozen A/B disagreement survives once A is
    # blind too? What survives is model difference; what vanishes was vision.
    d_ab = sum(1 for i in common if pairs["A vs B (frozen, B blind)"](i))
    d_bb = sum(1 for i in common
               if pairs["A_blind vs B (both blind)"](i))
    d_va = sum(1 for i in common
               if pairs["A_blind vs A (vision effect on A)"](i))
    print("\n=== attribution ===")
    print(f"  frozen A/B disagreements        : {d_ab}")
    print(f"  disagreements once A is blind   : {d_bb}")
    if d_ab:
        print(f"  share removed by blinding A     : "
              f"{(d_ab - d_bb) / d_ab:+.3f}  (positive => blindness was "
              f"creating disagreement)")
    print(f"  A's own labels moved by vision  : {d_va} "
          f"({d_va / len(common):.3f}) — the ceiling on any vision effect")
    print("\n  reading: if 'both blind' agreement rises toward the image-free "
          "rate\n  (0.293 disagreement) then B's dissent was mostly a missing "
          "input,\n  and the judge_B arm of judge_sensitivity.json is a "
          "vision-ablation\n  arm rather than a model-choice arm.")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ablation": "image_payload_withheld",
        "ablated_model_id": FROZEN_JUDGE_A_MODEL,
        "rubric_sha256": FROZEN_RUBRIC_SHA256,
        "temperature": FROZEN_TEMPERATURE,
        "seed": FROZEN_SEED,
        "n_items": len(common),
        "n_excluded_by_provider_input_moderation": len(excluded),
        "excluded_item_ids": excluded,
        "exclusion_note": (
            "The gateway moderates the input TEXT and its verdict is not "
            "invariant to whether an image is attached: these items were "
            "judged successfully by frozen judge A with the image and are "
            "refused with 400 data_inspection_failed once it is withheld. "
            "They are dropped from every arm so the three rates share one "
            "denominator." if excluded else ""),
        "prompt_sha256_matches_frozen_judge_a": len(common) - len(bad_prompt),
        "served_as": dict(served),
        "disagreement_rates": rates,
        "counts": {"a_vs_b": d_ab, "ablind_vs_b": d_bb,
                   "ablind_vs_a": d_va},
        "compliance_shift_ablind_minus_a": {str(k): v for k, v in
                                            sorted(deltas.items())},
        "frozen_reference": {
            "image_bearing_disagreement_rate": 0.503,
            "image_free_disagreement_rate": 0.293,
        },
    }
    out = OUT_DIR / "vision_ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="judge only the first N image-bearing items")
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip judging; re-analyse the existing journal")
    args = parser.parse_args()

    a_records, b_records, _items = frozen_labels()
    targets = image_bearing(a_records)
    if args.analyse_only:
        done = load_done(OUT_DIR / "ablated_labels.jsonl")
        print(f"journal holds {len(done)} ablated judgments")
        return analyse(a_records, b_records, done, targets)
    return run_ablation(args.limit)


if __name__ == "__main__":
    sys.exit(main())
