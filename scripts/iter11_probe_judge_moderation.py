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
cell is excluded from EVERY arm. This script is what establishes that the
exclusion is outcome-independent rather than merely asserting it.

UNIFORMITY DID NOT SURVIVE THE CONFIRMATORY RUN, and the paragraph above is
kept as the 2026-09-06 reading rather than rewritten over: judge A additionally
refused ``CMST_456921/text_only`` in ONE arm and served it in three, and
``--cells`` re-probing localizes that trigger to the arm's own REPLY
(``cell_probe_CMST_456921_text_only.json``). So two cells are uniform censoring
and one is differential, the union of all three is dropped from every arm, and
``outputs/iteration_11/diagnostics/judge_moderation/README.md`` carries both
readings with the dates that separate them.

WHAT THE REGENERATED SCAN FOUND (2026-09-07, once judging was no longer
competing for the gateway) is the same two ``data_inspection_failed`` cells the
production run recorded for this arm, bisecting the same way:
``neutral_response`` and ``context_only`` still 400 while ``no_history`` and
``terminal_only`` return 200, and both other identities accept the full
payload. It also found a defect in this script. Three cells came back with no
HTTP response at all — a transport failure, which is not a verdict — and the
first version filed them under ``n_refused``. That made the rate 5/600 instead
of 2/600, put four refusals in the ``shuffle`` variant's tally when one was
real, dragged a 165 KB body into the size comparison whose whole job is to
refute the size hypothesis (still refuted — 1.55 MB refused against 2.96 MB
accepted — but on the wrong numbers), and spent seven bisection requests per
cell localizing a trigger that did not exist. All four bisections of all three
cells returned 200, which is how they were identified as transport rather than
moderation. Transport failures are now retried (:data:`TRANSPORT_RETRIES`),
classified as ``unmeasured`` instead of refused, never bisected, and they make
the exit code 3, so an incomplete scan cannot be filed as a complete one.

``max_tokens=1`` keeps the cost near zero; only the INPUT is moderated, so the
verdict does not depend on the completion. Concurrency defaults to 8, which
the gateway probe measured clean.

Usage:
    python3 scripts/iter11_probe_judge_moderation.py                  # one arm
    python3 scripts/iter11_probe_judge_moderation.py --target phi4_mm
    python3 scripts/iter11_probe_judge_moderation.py --limit 40       # trial
    python3 scripts/iter11_probe_judge_moderation.py --all-targets

Exit codes: 0 every cell obtained a verdict; 2 no credentials resolved; 3 the
artifact was written but some cells are ``unmeasured``, so re-run it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
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
from causal_mllm.seeds import code_tree_status, get_git_commit  # noqa: E402
from causal_mllm.validation.relations import _file_sha256  # noqa: E402

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

#: ``status`` when no HTTP response arrived at all — a timeout, a reset, a TLS
#: failure. This is NOT a verdict about the cell, and treating it as one is the
#: defect documented under "What the first committed scan found" below.
TRANSPORT_FAILURE = -1

#: Transport failures are retried; provider verdicts are not. Retrying cannot
#: bias the measurement because the request bytes are identical and only the
#: provider's answer is read, while re-asking a question the gateway has
#: already answered spends requests and proves nothing — the production
#: pipeline established that a 400 here is stable over eleven attempts.
TRANSPORT_RETRIES = 3

#: This stage's own output tree, excluded from the code-dirtiness it RECORDS so
#: that a scan cannot invalidate itself by writing its artifact. Narrow and
#: reported, never a blanket ignore of ``outputs/`` — the same per-stage scoping
#: the preflight and the confirmatory gate use.
OWN_OUTPUT_PREFIXES = (
    "outputs/iteration_11/diagnostics/judge_moderation/",
)


def provenance() -> dict:
    """The code, and the state of the code tree, that produced this artifact.

    A diagnostic that costs ~600 requests per arm and is cited as evidence has
    to name the code that produced it. Without that, a reader holding the
    artifact cannot tell which counting rule the numbers came from — which is
    exactly the question the 2026-09-07 regeneration turned on, when three
    transport failures turned out to have been counted as refusals. Run it from
    a dirty tree and the artifact says so rather than looking reconstructible.
    """
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    return {
        "produced_by": "scripts/iter11_probe_judge_moderation.py",
        "kind": "iteration_11_judge_moderation_scan_v1",
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
    }


def _load_credentials() -> dict:
    """Environment first, then the credentials file, then empty strings.

    Resolved at import rather than enforced at import: how a scan classifies
    its own results is pure logic and is unit-tested offline, and a CI checkout
    holds only ``llm_judge_credentials.conf.example``. A missing key is fatal
    where it matters — when :func:`main` is about to spend a request.
    """
    values = {}
    if CREDENTIALS_FILE.exists():
        for line in CREDENTIALS_FILE.read_text(
                encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return {key: os.environ.get(key) or values.get(key) or ""
            for key in ("LLM_JUDGE_BASE_URL", "LLM_JUDGE_API_KEY",
                        "LLM_JUDGE_PRIMARY_A_MODEL",
                        "LLM_JUDGE_PRIMARY_B_MODEL",
                        "LLM_ADJUDICATOR_MODEL")}


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


def _panel_path(model_key: str) -> Path:
    """The blinded panel this target's judge sessions were built from."""
    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    profile = profiles[f"iteration_11_{model_key}"]
    return REPO_ROOT / profile["output_dir"] / "blinded_items.json"


def _blinded_items(model_key: str) -> list[dict]:
    """The items the judge pipeline actually built for this target.

    Read rather than re-derived: this measures the payloads that were sent,
    not a reconstruction of them.
    """
    path = _panel_path(model_key)
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run the judge pipeline for {model_key} first "
            "(it writes the blinded panel before it judges anything)")
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_binding(model_key: str, n_scanned: int) -> dict:
    """What was scanned, by hash, so the artifact cannot be read as a scan of
    some other panel than the one the four arms were judged on."""
    path = _panel_path(model_key)
    return {
        "model_key": model_key,
        "path": str(path.relative_to(REPO_ROOT)) if path.exists() else None,
        "sha256": _file_sha256(path) if path.exists() else None,
        "n_items_scanned": n_scanned,
    }


def _probe_out_stem(cell_specs: list[str]) -> str:
    """A filename stem for a ``--cells`` probe, from specs holding a separator.

    ``FAMILY/VARIANT`` is how a cell is named everywhere else in this project,
    and interpolating one into a default output path would silently place the
    artifact in a nested directory named after the family. Evidence that has to
    be searched for is evidence that goes unread, so the separator becomes an
    underscore and the artifact lands in :data:`OUT_DIR` beside the scan it
    explains.
    """
    return "_".join(spec.replace("/", "_") for spec in cell_specs)


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
          max_tokens: int = 1, retries: int = TRANSPORT_RETRIES) -> dict:
    """One request in the production body shape.

    A transport failure is retried up to ``retries`` times and a provider
    verdict is returned on the first attempt. ``status`` is the HTTP status, or
    :data:`TRANSPORT_FAILURE` when no response arrived at all even after the
    retries; ``transport_attempts`` records how many it took, so a cell that
    needed three tries is visible rather than being folded silently into a 200,
    and ``transport_error`` keeps the exception text of the last failure.
    """
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
    status, text = TRANSPORT_FAILURE, "no attempt made"
    attempts = 0
    while attempts <= retries:
        attempts += 1
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                data=body, timeout=180)
            status, text = resp.status_code, resp.text
        except Exception as exc:                        # noqa: BLE001
            status, text = TRANSPORT_FAILURE, f"{type(exc).__name__}: {exc}"
        with _lock:
            _issued += 1
            if _issued % 25 == 0:
                print(f"    ... {_issued} requests", flush=True)
        if status != TRANSPORT_FAILURE:
            break
    code = message = None
    try:
        err = (json.loads(text) or {}).get("error") or {}
        code, message = err.get("code"), err.get("message")
    except Exception:                                   # noqa: BLE001
        pass
    return {"status": status, "error_code": code, "error_message": message,
            "request_bytes": len(body),
            "transport_attempts": attempts,
            "transport_error": text if status == TRANSPORT_FAILURE else None,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest()}


def classify(results: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split one arm's requests into accepted, refused and unmeasured.

    ``200`` is acceptance and an HTTP ``4xx``/``5xx`` is the provider's own
    verdict. A negative status is this script's transport failing: it says
    nothing about the cell, and filing it as a refusal is not a harmless
    rounding — it inflates the refusal rate, puts a cell into the wrong
    variant's tally (which reads as a claim about which kinds of cell get
    moderated), drags an unrelated body size into the size comparison that is
    supposed to refute the size hypothesis, and spends seven bisection requests
    localizing a trigger that does not exist. So unmeasured cells are reported
    as their own category, with the denominator stated beside the rate.
    """
    accepted = [r for r in results if r["status"] == 200]
    refused = [r for r in results if r["status"] >= 400]
    unmeasured = [r for r in results if r["status"] < 0]
    return accepted, refused, unmeasured


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

    accepted, refused, unmeasured = classify(results)
    codes: dict = {}
    for r in refused:
        key = r["error_code"] or f"http_{r['status']}"
        codes[key] = codes.get(key, 0) + 1
    transport_codes: dict = {}
    for r in unmeasured:
        key = (r.get("transport_error") or "unknown").split(":")[0]
        transport_codes[key] = transport_codes.get(key, 0) + 1
    print(f"  accepted {len(accepted)}/{len(results)}   "
          f"refused {len(refused)}   codes={codes}")
    if unmeasured:
        print(f"  UNMEASURED {len(unmeasured)} (transport failed after "
              f"{TRANSPORT_RETRIES} retries, so no verdict was obtained): "
              f"{transport_codes}")

    by_variant: dict = {}
    for r in results:
        slot = by_variant.setdefault(r["variant"],
                                     {"n": 0, "refused": 0, "unmeasured": 0})
        slot["n"] += 1
        slot["refused"] += int(r["status"] >= 400)
        slot["unmeasured"] += int(r["status"] < 0)

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
            for name, kw in _bisection_kwargs(item).items():
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
    unmeasured_tally = {}
    if bisected:
        names = list(BISECTIONS) + ["full_B", "full_ADJUDICATOR"]
        # Counted over provider verdicts only. A transport failure inside a
        # bisection would otherwise be tallied as "still refused", which is
        # how a network blip becomes a claim about the payload.
        tally = {name: sum(1 for b in bisected if b[name]["status"] >= 400)
                 for name in names}
        unmeasured_tally = {
            name: sum(1 for b in bisected if b[name]["status"] < 0)
            for name in names}
        print("  bisection tally (still refused / sampled):")
        for name, n in tally.items():
            extra = f"   ({unmeasured_tally[name]} unmeasured)" \
                if unmeasured_tally[name] else ""
            print(f"    {name:20s} {n}/{len(bisected)}{extra}")

    return {
        "model_key": model_key,
        "identity": IDENTITIES["A"],
        "panel": _panel_binding(model_key, len(results)),
        "n_items": len(results),
        "n_accepted": len(accepted),
        "n_refused": len(refused),
        "n_unmeasured": len(unmeasured),
        "refusal_rate": round(len(refused) / len(results), 4) if results
        else None,
        "classification_rule": (
            "status 200 is accepted; an HTTP 4xx/5xx is the provider's refusal "
            "and is the only thing counted in n_refused, refusal_rate, "
            "by_variant.refused and the size comparison; a negative status "
            f"means no response arrived after {TRANSPORT_RETRIES} transport "
            "retries, so the cell is unmeasured rather than refused and is "
            "never bisected"),
        "transport_retries": TRANSPORT_RETRIES,
        "error_codes": codes,
        "transport_errors": transport_codes or None,
        "by_variant": by_variant,
        "request_bytes": sizes,
        "size_hypothesis_refuted": size_refutes,
        "bisections": {k: v for k, v in BISECTIONS.items()},
        "bisection_tally": tally or None,
        "bisection_unmeasured": unmeasured_tally or None,
        "n_bisected": len(bisected),
        "bisected": bisected,
        "refused_cells": refused,
        "unmeasured_cells": unmeasured,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def _bisection_kwargs(item: dict) -> dict:
    """The four payload variants that isolate one part of the request."""
    return {
        "neutral_response": {"response": NEUTRAL_RESPONSE},
        "no_history": {"history": [], "response": item["response"]},
        "context_only": {"response": NEUTRAL_RESPONSE},
        "terminal_only": {"history": [], "response": NEUTRAL_RESPONSE},
    }


def probe_cell(model_key: str, item: dict, label: str = "A") -> dict:
    """One cell in one arm: the full payload, every bisection, both others.

    A scan answers "which cells are refused". This answers the question a scan
    cannot: a cell refused in ONE arm and served in three is either a function
    of that arm's reply or a function of when the request was made, and only
    re-sending the payload now, in every arm, separates the two. If it still
    400s in one arm and 200s in the others, the reply matters; if it now 400s
    everywhere or nowhere, the verdict moved.
    """
    out = {"model_key": model_key, "item_id": item["item_id"],
           "family_id": item["family_id"], "variant": item["variant"],
           "response_sha256": item["response_sha256"],
           "response_chars": len(item["response"]),
           "identity": IDENTITIES[label]}
    prompt, images = _render(label, item)
    full = _post(label, prompt, images)
    out["full"] = {"status": full["status"], "error_code": full["error_code"],
                   "prompt_chars": len(prompt), "n_images": len(images),
                   "request_bytes": full["request_bytes"]}
    for name, kw in _bisection_kwargs(item).items():
        prompt, images = _render(label, item, **kw)
        res = _post(label, prompt, images)
        out[name] = {"status": res["status"], "error_code": res["error_code"],
                     "prompt_chars": len(prompt)}
    for other in ("B", "ADJUDICATOR"):
        if other == label:
            continue
        prompt, images = _render(other, item)
        res = _post(other, prompt, images)
        out[f"full_{other}"] = {"status": res["status"],
                                "error_code": res["error_code"],
                                "identity": IDENTITIES[other]}
    out["probed_at"] = datetime.now(timezone.utc).isoformat()
    return out


def probe_cells(targets: list[str], cells: list[str], label: str,
                concurrency: int) -> dict:
    """Named cells across named arms, so a divergence can be localized."""
    wanted = []
    for spec in cells:
        family, _, variant = spec.partition("/")
        if not family or not variant:
            raise SystemExit(
                f"--cells takes FAMILY/VARIANT pairs, got {spec!r}")
        wanted.append((family, variant))
    print(f"\n=== probing {len(wanted)} cell(s) across {len(targets)} arm(s) "
          f"as {IDENTITIES[label]} "
          f"({(len(BISECTIONS) + 3) * len(wanted) * len(targets)} requests) ===")

    jobs = []
    for key in targets:
        by_cell = {(it["family_id"], it["variant"]): it
                   for it in _blinded_items(key)}
        for cell in wanted:
            item = by_cell.get(cell)
            if item is None:
                raise SystemExit(
                    f"{key}: {'/'.join(cell)} is not a cell of that arm's "
                    f"panel")
            jobs.append((key, item))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda job: probe_cell(job[0], job[1], label),
                              jobs))

    per_cell = {}
    for rec in results:
        name = f"{rec['family_id']}/{rec['variant']}"
        slot = per_cell.setdefault(name, {})
        slot[rec["model_key"]] = rec
        marks = " ".join(
            f"{k}={v['status']}" for k, v in rec.items()
            if isinstance(v, dict) and "status" in v)
        print(f"  {name:28s} {rec['model_key']:14s} {marks}")

    full_statuses: dict = {}
    for rec in results:
        name = f"{rec['family_id']}/{rec['variant']}"
        full_statuses.setdefault(name, {})[rec["model_key"]] = \
            rec["full"]["status"]
    # A divergence is a difference between two VERDICTS. An arm whose response
    # never arrived has no verdict to differ with, so counting it as a
    # disagreement would report a network blip as a finding about that arm's
    # reply -- precisely the claim this probe exists to make or refuse.
    divergent = sorted(
        name for name, per in full_statuses.items()
        if len({status for status in per.values() if status >= 0}) > 1)
    unmeasured_now = sorted(
        f"{name} ({arm})" for name, per in full_statuses.items()
        for arm, status in per.items() if status < 0)
    if unmeasured_now:
        print(f"  UNMEASURED (transport failed after {TRANSPORT_RETRIES} "
              f"retries, so these arms answered nothing): {unmeasured_now}")
    return {
        "identity": IDENTITIES[label],
        "cells": sorted({f"{r['family_id']}/{r['variant']}" for r in results}),
        "targets": list(targets),
        "n_requests": len(results) * (len(BISECTIONS) + 3),
        "per_cell": per_cell,
        "full_payload_status": full_statuses,
        "arms_that_disagree_now": divergent,
        "arms_unmeasured_now": unmeasured_now,
        "reading": (
            "a cell whose full payload is refused in some arms and accepted in "
            "others AT THE SAME TIME is a function of that arm's reply, not of "
            "the shared history; a cell refused or accepted in all arms now "
            "but not during the run is a verdict that moved; an arm listed as "
            "unmeasured obtained no verdict at all and is evidence about the "
            "network, not about the cell"),
        "probed_at": datetime.now(timezone.utc).isoformat(),
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
    ap.add_argument("--cells", default=None,
                    help="comma-separated FAMILY/VARIANT cells to re-probe in "
                         "every target arm, instead of scanning the panel; "
                         "this is what separates a refusal caused by one "
                         "arm's reply from a verdict that moved")
    ap.add_argument("--identity", default="A", choices=sorted(IDENTITIES),
                    help="which frozen identity to probe as (default A, the "
                         "one whose provider moderates inputs)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not BASE_URL or not API_KEY:
        print("FAIL: no gateway credentials resolved from the environment or "
              f"{CREDENTIALS_FILE}; this probe spends real requests",
              file=sys.stderr)
        return 2

    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    targets = (sorted(k[len("iteration_11_"):] for k in profiles
                      if k.startswith("iteration_11_"))
               if args.all_targets else [args.target])

    print(f"gateway    {BASE_URL}")
    print(f"identities {IDENTITIES}")
    print(f"targets    {targets}")

    if args.cells:
        cell_specs = [c.strip() for c in args.cells.split(",") if c.strip()]
        probe = probe_cells(targets, cell_specs, args.identity,
                            args.concurrency)
        probe["gateway"] = BASE_URL
        probe["identities"] = IDENTITIES
        probe.update(provenance())
        probe["panels"] = {key: _panel_binding(key, len(cell_specs))
                           for key in targets}
        probe["question"] = (
            "is a cell refused in one arm and served in another a function of "
            "that arm's reply, or of when the request was made")
        out = Path(args.json_out) if args.json_out else \
            OUT_DIR / f"cell_probe_{_probe_out_stem(probe['cells'])}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(probe, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nwrote {out}")
        print(f"arms that disagree on the full payload right now: "
              f"{probe['arms_that_disagree_now'] or 'none'}")
        return 0

    report = {
        "question": "which cells does a judge's provider refuse to moderate, "
                    "and is the trigger the shared cell or the target's reply",
        "gateway": BASE_URL,
        "identities": IDENTITIES,
        **provenance(),
        "max_tokens": 1,
        "concurrency": args.concurrency,
        "transport_retries": TRANSPORT_RETRIES,
        "classification": (
            "a provider refusal is an HTTP 4xx/5xx carrying the gateway's own "
            "error code; a cell whose request never arrived is unmeasured, is "
            "reported separately, and is not counted as a refusal"),
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
    total_unmeasured = sum(t["n_unmeasured"]
                           for t in report["targets"].values())
    print(f"total refused: {total_refused}   "
          f"total unmeasured: {total_unmeasured}")
    if total_unmeasured:
        # The artifact is written either way, but a scan that did not obtain a
        # verdict for every cell has not measured the whole panel, and the exit
        # code is what tells the operator to re-run it.
        print("INCOMPLETE: some cells obtained no verdict after "
              f"{TRANSPORT_RETRIES} transport retries; they are listed under "
              "unmeasured_cells and are NOT refusals", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
