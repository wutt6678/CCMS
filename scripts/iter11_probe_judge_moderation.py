#!/usr/bin/env python3
"""Iteration 11.7 — measure what a judge's provider refuses to look at.

WHY THIS EXISTS SEPARATELY FROM THE GATEWAY PROBE
:mod:`scripts.iter11_probe_judge_gateway` asks whether an identity is
reachable and whether it accepts an image, using ONE small image
(``mtmcs_type_b_0_main.png``, 533,849 bytes). It reported PROBE PASS, and
eleven hours later all three judge-A arms were dead at item-0164. The probe
had never sent the gateway anything it might object to on CONTENT, and this
is a safety dataset: its whole subject matter is unsafe multi-turn requests,
so a provider that moderates inputs will object to some of them.

WHAT IT FOUND
Judge A (qwen3.8-max) answers HTTP 400 ``data_inspection_failed`` — "Input
text data may contain inappropriate content." — for family CMST_795308 in its
``cross_modal`` and ``shuffle`` variants: 2 of the frozen 600 cells (0.33%).
Judge B (glm-5.2) and the adjudicator (kimi-k3) both return 200 on the
byte-identical payload, so moderation here is per-model, not per-gateway.

Two hypotheses were tested and rejected on the way, and both rejections are
recorded in the artifact because both were plausible:

* payload SIZE. The image is 1,152,138 bytes, under
  ``_JUDGE_PAYLOAD_MAX_BYTES``, so it is transmitted undownscaled and base64
  inflates it to ~1.54 MB of body. But a 2.96 MB body is ACCEPTED by the same
  identity, and a size ladder of real dataset images returns 200 all the way
  up. Size does not separate the accepted from the refused.
* the target's RESPONSE. Bisecting the payload settles it: replacing the
  response with "OK." still 400s, while dropping the conversation history
  returns 200. The trigger is the shared HISTORY — the unsafe turn every
  target was shown — so the same cells drop out of all four arms. That is
  uniform censoring, which costs coverage. Had it been the response, each arm
  would have lost DIFFERENT cells, which would bias the very quantity 11.8
  compares.

The consequence for the run is handled in
:mod:`scripts.run_llm_judge_pipeline`: a refusal is recorded per cell and the
cell is excluded from EVERY arm. This script is what establishes the exclusion
is uniform and outcome-independent rather than merely asserting it.

``max_tokens=1`` keeps the cost near zero; only the INPUT is moderated, so the
verdict does not depend on the completion. Concurrency defaults to 8, which
the gateway probe measured clean.

Usage:
    python3 scripts/iter11_probe_judge_moderation.py                  # one arm
    python3 scripts/iter11_probe_judge_moderation.py --target phi4_mm
    python3 scripts/iter11_probe_judge_moderation.py --limit 40       # trial
    python3 scripts/iter11_probe_judge_moderation.py --all-targets
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.llm_judge import (  # noqa: E402
    LLMJudgeConfig,
    MultimodalLLMJudge,
    _payload_image,
)

PROFILES_FILE = REPO_ROOT / "configs" / "evaluation" / "scale_profiles.json"
CREDENTIALS_FILE = REPO_ROOT / "configs" / "evaluation" \
    / "llm_judge_credentials.conf"
OUT_DIR = REPO_ROOT / "outputs" / "iteration_11" / "diagnostics" \
    / "judge_moderation"

#: Stands in for the target's reply when bisecting, so the only thing that
#: changes is whether the shared context is present.
NEUTRAL_RESPONSE = "OK."

#: Bisection payloads. Each isolates one part of the request, so whichever one
#: still 400s localizes the trigger. The history holds the unsafe turn and is
#: identical across targets; the terminal query and the response are not.
BISECTIONS = {
    "neutral_response": "the cell with this target's reply removed",
    "no_history": "the reply alone, with no multi-turn context",
    "context_only": "the shared context alone, with no reply",
    "terminal_only": "the terminal query alone",
}


def _load_credentials() -> dict:
    if not CREDENTIALS_FILE.exists():
        raise SystemExit(
            f"{CREDENTIALS_FILE} not found; copy the .example template and "
            "fill in the rotated key")
    values = {}
    for line in CREDENTIALS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


CREDS = _load_credentials()
BASE_URL = CREDS["LLM_JUDGE_BASE_URL"]
API_KEY = CREDS["LLM_JUDGE_API_KEY"]
IDENTITIES = {
    "A": CREDS["LLM_JUDGE_PRIMARY_A_MODEL"],
    "B": CREDS["LLM_JUDGE_PRIMARY_B_MODEL"],
    "ADJUDICATOR": CREDS["LLM_ADJUDICATOR_MODEL"],
}

#: Judge A's frozen seed is 42 and B's is 43. The seed cannot affect input
#: moderation, but matching production exactly is cheaper than arguing it.
_SEEDS = {"A": 42, "B": 43, "ADJUDICATOR": 99}
_JUDGES = {
    label: MultimodalLLMJudge(
        LLMJudgeConfig(model_id=model_id, provider="aliyun",
                       base_url=BASE_URL, api_key=API_KEY, temperature=0.0,
                       seed=_SEEDS[label], max_retries=0, timeout=180.0),
        judge_id=label)
    for label, model_id in IDENTITIES.items()}

_lock = threading.Lock()
_issued = 0


def _blinded_items(model_key: str) -> list[dict]:
    """The items the judge pipeline actually built for this target.

    Read rather than re-derived: this measures the payloads that were sent,
    not a reconstruction of them.
    """
    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    profile = profiles[f"iteration_11_{model_key}"]
    path = REPO_ROOT / profile["output_dir"] / "blinded_items.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run the judge pipeline for {model_key} first "
            "(it writes the blinded panel before it judges anything)")
    return json.loads(path.read_text(encoding="utf-8"))


def _render(label: str, item: dict, *, response: str | None = None,
            history: list | None = None) -> tuple[str, list]:
    judge = _JUDGES[label]
    prompt, images, _hashes = judge._build_prompt(
        item["system_prompt"],
        item["conversation_history"] if history is None else history,
        item["terminal_query"],
        item["response"] if response is None else response)
    return prompt, images


def _post(label: str, prompt: str, image_contents: list,
          max_tokens: int = 1) -> dict:
    """One request in the production body shape, with no retries."""
    global _issued
    judge = _JUDGES[label]
    content = [{"type": "text", "text": prompt}]
    content.extend(image_contents)
    payload = {"model": judge.config.model_id,
               "messages": [{"role": "user", "content": content}],
               "temperature": judge.config.temperature,
               "seed": judge.config.seed,
               "max_tokens": max_tokens}
    body = json.dumps(payload).encode("utf-8")
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
            data=body, timeout=180)
        status, text = resp.status_code, resp.text
    except Exception as exc:                            # noqa: BLE001
        status, text = -1, f"{type(exc).__name__}: {exc}"
    code = message = None
    try:
        err = (json.loads(text) or {}).get("error") or {}
        code, message = err.get("code"), err.get("message")
    except Exception:                                   # noqa: BLE001
        pass
    with _lock:
        _issued += 1
        if _issued % 25 == 0:
            print(f"    ... {_issued} requests", flush=True)
    return {"status": status, "error_code": code, "error_message": message,
            "request_bytes": len(body),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest()}


def scan_target(model_key: str, limit: int | None, concurrency: int,
                bisect_limit: int) -> dict:
    items = _blinded_items(model_key)
    if limit:
        items = items[:limit]
    print(f"\n=== {model_key}: {len(items)} items as {IDENTITIES['A']} "
          f"at concurrency {concurrency} ===")

    def one(arg):
        n, item = arg
        prompt, images = _render("A", item)
        res = _post("A", prompt, images)
        return {"index": n, "item_id": item["item_id"],
                "family_id": item["family_id"], "variant": item["variant"],
                "n_images": len(images), "prompt_chars": len(prompt),
                **res}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, enumerate(items)))

    refused = [r for r in results if r["status"] != 200]
    accepted = [r for r in results if r["status"] == 200]
    codes: dict = {}
    for r in refused:
        key = r["error_code"] or f"http_{r['status']}"
        codes[key] = codes.get(key, 0) + 1
    print(f"  accepted {len(accepted)}/{len(results)}   "
          f"refused {len(refused)}   codes={codes}")

    by_variant: dict = {}
    for r in results:
        slot = by_variant.setdefault(r["variant"], {"n": 0, "refused": 0})
        slot["n"] += 1
        slot["refused"] += int(r["status"] != 200)

    sizes = {
        "accepted_min": min((r["request_bytes"] for r in accepted),
                            default=None),
        "accepted_max": max((r["request_bytes"] for r in accepted),
                            default=None),
        "refused_min": min((r["request_bytes"] for r in refused),
                           default=None),
        "refused_max": max((r["request_bytes"] for r in refused),
                           default=None),
    }
    # The size hypothesis is only refuted if an ACCEPTED body was larger than a
    # refused one. Stated as a finding rather than left to be inferred.
    size_refutes = (sizes["accepted_max"] is not None
                    and sizes["refused_min"] is not None
                    and sizes["accepted_max"] > sizes["refused_min"])

    bisected = []
    if refused and bisect_limit:
        sample = refused[:bisect_limit]
        by_id = {it["item_id"]: it for it in items}
        print(f"  bisecting {len(sample)} refused cell(s) "
              f"({(len(BISECTIONS) + 2) * len(sample)} requests)...")

        def bisect_one(rec):
            item = by_id[rec["item_id"]]
            out = {"item_id": rec["item_id"], "family_id": rec["family_id"],
                   "variant": rec["variant"]}
            kwargs = {
                "neutral_response": {"response": NEUTRAL_RESPONSE},
                "no_history": {"history": [], "response": item["response"]},
                "context_only": {"response": NEUTRAL_RESPONSE},
                "terminal_only": {"history": [],
                                  "response": NEUTRAL_RESPONSE},
            }
            for name, kw in kwargs.items():
                prompt, images = _render("A", item, **kw)
                res = _post("A", prompt, images)
                out[name] = {"status": res["status"],
                             "error_code": res["error_code"],
                             "prompt_chars": len(prompt)}
            # Does any OTHER frozen identity accept the full payload? If one
            # does, the cell is judgeable and only primary A is censored.
            for label in ("B", "ADJUDICATOR"):
                prompt, images = _render(label, item)
                res = _post(label, prompt, images)
                out[f"full_{label}"] = {"status": res["status"],
                                        "error_code": res["error_code"]}
            return out

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            bisected = list(ex.map(bisect_one, sample))

    tally = {}
    if bisected:
        names = list(BISECTIONS) + ["full_B", "full_ADJUDICATOR"]
        tally = {name: sum(1 for b in bisected if b[name]["status"] != 200)
                 for name in names}
        print("  bisection tally (still refused / sampled):")
        for name, n in tally.items():
            print(f"    {name:20s} {n}/{len(bisected)}")

    return {
        "model_key": model_key,
        "identity": IDENTITIES["A"],
        "n_items": len(results),
        "n_accepted": len(accepted),
        "n_refused": len(refused),
        "refusal_rate": round(len(refused) / len(results), 4) if results
        else None,
        "error_codes": codes,
        "by_variant": by_variant,
        "request_bytes": sizes,
        "size_hypothesis_refuted": size_refutes,
        "bisections": {k: v for k, v in BISECTIONS.items()},
        "bisection_tally": tally or None,
        "n_bisected": len(bisected),
        "bisected": bisected,
        "refused_cells": refused,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def size_ladder(model_key: str) -> list[dict]:
    """One prompt, real images of increasing size, so bytes are the only
    variable. This is what refutes the payload-size hypothesis directly rather
    than by coincidence of which cells happened to be refused."""
    items = _blinded_items(model_key)
    prompt, _ignore = _render("A", items[0])
    media = sorted((REPO_ROOT / "data" / "media" / "source").glob("*.png"),
                   key=lambda p: p.stat().st_size)
    if not media:
        return []
    picks = sorted({0, len(media) // 4, len(media) // 2,
                    3 * len(media) // 4, len(media) - 1})
    ladder = []
    for i in picks:
        path = media[i]
        raw = path.read_bytes()
        sent, mime_override = _payload_image(raw)
        mime = mime_override or "image/png"
        b64 = base64.b64encode(sent).decode("utf-8")
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}}]
        res = _post("A", prompt, content)
        ladder.append({"image": path.name, "raw_bytes": len(raw),
                       "transmitted_bytes": len(sent),
                       "downscaled": len(sent) != len(raw),
                       "base64_bytes": len(b64),
                       "request_bytes": res["request_bytes"],
                       "status": res["status"],
                       "error_code": res["error_code"]})
        print(f"  {path.name:38s} raw={len(raw):>9,} sent={len(sent):>9,} "
              f"req={res['request_bytes']:>10,} status={res['status']}")
    return ladder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="qwen35_2b")
    ap.add_argument("--all-targets", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bisect-limit", type=int, default=40,
                    help="refused cells to bisect (6 requests each)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-size-ladder", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    targets = (sorted(k[len("iteration_11_"):] for k in profiles
                      if k.startswith("iteration_11_"))
               if args.all_targets else [args.target])

    print(f"gateway    {BASE_URL}")
    print(f"identities {IDENTITIES}")
    print(f"targets    {targets}")

    report = {
        "question": "which cells does a judge's provider refuse to moderate, "
                    "and is the trigger the shared cell or the target's reply",
        "gateway": BASE_URL,
        "identities": IDENTITIES,
        "max_tokens": 1,
        "concurrency": args.concurrency,
        "targets": {},
        "size_ladder": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in targets:
        report["targets"][key] = scan_target(
            key, args.limit, args.concurrency, args.bisect_limit)
    if not args.skip_size_ladder:
        print("\n=== size ladder: one prompt, real images of increasing "
              "size ===")
        report["size_ladder"] = size_ladder(targets[0])

    out = Path(args.json_out) if args.json_out else \
        OUT_DIR / f"judge_a_moderation_{'_'.join(targets)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwrote {out}")

    total_refused = sum(t["n_refused"] for t in report["targets"].values())
    print(f"total refused: {total_refused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
