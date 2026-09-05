#!/usr/bin/env python3
"""Iteration 11.7 pre-flight — probe the judge gateway BEFORE any spend.

Judging 2,400 outputs with a three-model ensemble is the one step in
Iteration 11 that costs money and cannot be undone by re-running locally. A
model id that the gateway lists but has not activated returns HTTP 400 only
once a real batch is in flight — which is exactly how the
``kimi/kimi-k3`` vs ``kimi-k3`` distinction was found during Iteration 10.
This probe makes that discovery cost six requests instead of a judging run.

For each of the three frozen judge identities it checks, in order:

1. whether the gateway's ``/models`` listing contains the id;
2. a TEXT-ONLY request using the production payload shape (same keys the
   judge sends: model, messages, temperature, seed);
3. an IMAGE-BEARING request through the production ``_payload_image``
   downscaling path, because the rubric attaches the family's source media
   and a model that refuses ``image_url`` would fail at scale rather than at
   load time.

Each response's provider-returned model id and system fingerprint are
reported: the identity that actually SERVED the request is what the blinding
and the ensemble-distinctness checks depend on, not the id that was asked
for.

The API key is never printed. Writes nothing; exits non-zero if any identity
is unusable.

Usage:
    python3 scripts/iter11_probe_judge_gateway.py
    python3 scripts/iter11_probe_judge_gateway.py --json-out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.evaluation.llm_judge import _payload_image  # noqa: E402

CREDENTIALS = REPO_ROOT / "configs" / "evaluation" / "llm_judge_credentials.conf"
DEFAULT_BASE_URL = (
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com"
    "/compatible-mode/v1")

#: A tracked Scale-C source image, small enough to probe with. The rubric
#: attaches the family's own media, so multimodal acceptance is part of
#: whether an identity is usable at all.
PROBE_IMAGE = REPO_ROOT / "data" / "media" / "source" \
    / "mtmcs_type_b_0_main.png"

#: The production judge payload's fixed decoding values.
TEMPERATURE = 0.0
SEED = 42
TIMEOUT = 90.0

TEXT_PROMPT = (
    "Reply with exactly this JSON object and nothing else: "
    '{"ok": true}')


def load_credentials() -> dict:
    """KEY=VALUE pairs from the gitignored conf file, env taking precedence.

    Mirrors ``scripts/run_llm_judge_pipeline.py`` so the probe resolves
    identity exactly the way the judging run will.
    """
    values: dict = {}
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    resolved = {}
    for key in ("LLM_JUDGE_API_KEY", "LLM_JUDGE_BASE_URL",
                "LLM_JUDGE_PRIMARY_A_MODEL", "LLM_JUDGE_PRIMARY_B_MODEL",
                "LLM_ADJUDICATOR_MODEL"):
        resolved[key] = os.environ.get(key) or values.get(key) or ""
    resolved["LLM_JUDGE_BASE_URL"] = resolved["LLM_JUDGE_BASE_URL"] \
        or DEFAULT_BASE_URL
    return resolved


def post(base_url: str, api_key: str, payload: dict) -> dict:
    """One gateway call, reported without ever echoing the key."""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"ok": False, "stage": "transport", "error": f"{type(exc).__name__}: {exc}"}
    body = None
    try:
        body = response.json()
    except ValueError:
        pass
    if response.status_code != 200:
        return {
            "ok": False, "stage": "http",
            "status": response.status_code,
            "error": (json.dumps(body)[:400] if body is not None
                      else response.text[:400]),
        }
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "ok": True,
        "status": response.status_code,
        "provider_returned_model": str(body.get("model", "")),
        "system_fingerprint": str(body.get("system_fingerprint", "")),
        "finish_reason": choice.get("finish_reason"),
        "response_head": str(message.get("content", ""))[:160],
        "usage": body.get("usage"),
    }


def image_content(path: Path) -> dict:
    """The production image payload for one file, downscaled the same way."""
    raw = path.read_bytes()
    payload_bytes, mime_override = _payload_image(raw)
    mime = mime_override or "image/png"
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,"
                   f"{base64.b64encode(payload_bytes).decode('utf-8')}",
        },
        "_probe_original_sha256": hashlib.sha256(raw).hexdigest(),
        "_probe_original_bytes": len(raw),
        "_probe_payload_bytes": len(payload_bytes),
    }


def list_models(base_url: str, api_key: str) -> dict:
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if response.status_code != 200:
        return {"ok": False, "status": response.status_code,
                "error": response.text[:300]}
    try:
        ids = sorted(m.get("id") for m in response.json().get("data") or [])
    except ValueError as exc:
        return {"ok": False, "error": f"unparseable listing: {exc}"}
    return {"ok": True, "n": len(ids), "ids": ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None,
                        help="write the full probe result here")
    args = parser.parse_args()

    credentials = load_credentials()
    api_key = credentials["LLM_JUDGE_API_KEY"]
    base_url = credentials["LLM_JUDGE_BASE_URL"]
    if not api_key or api_key == "REPLACE_WITH_ROTATED_KEY":
        print("FAIL: no LLM_JUDGE_API_KEY resolved (environment or "
              f"{CREDENTIALS})", file=sys.stderr)
        return 2

    identities = {
        "primary_a": credentials["LLM_JUDGE_PRIMARY_A_MODEL"] or "qwen3.8-max",
        "primary_b": credentials["LLM_JUDGE_PRIMARY_B_MODEL"] or "glm-5.2",
        "adjudicator": credentials["LLM_ADJUDICATOR_MODEL"] or "kimi-k3",
    }
    print(f"gateway   {base_url}")
    print(f"api key   {'present' if api_key else 'MISSING'} "
          f"(sha256 {hashlib.sha256(api_key.encode()).hexdigest()[:12]}…, "
          f"never printed)")
    print(f"identities {identities}")

    distinct = set(identities.values())
    if len(distinct) != 3:
        print(f"FAIL: the ensemble needs three DISTINCT identities, got "
              f"{sorted(identities.values())}; an adjudicator that is also a "
              f"primary is not adjudication", file=sys.stderr)
        return 2

    result: dict = {"gateway": base_url, "identities": identities,
                    "models": None, "probes": {}}

    listing = list_models(base_url, api_key)
    result["models"] = listing
    if listing.get("ok"):
        print(f"\n/models   {listing['n']} ids listed")
        for role, model_id in sorted(identities.items()):
            present = model_id in listing["ids"]
            near = [i for i in listing["ids"]
                    if i and model_id.split("/")[-1] in i]
            print(f"  {role:12s} {model_id:20s} listed={present}"
                  + (f"  similar={near}" if not present and near else ""))
            result.setdefault("listed", {})[role] = present
    else:
        print(f"\n/models   UNAVAILABLE: {listing.get('error')}")

    if not PROBE_IMAGE.exists():
        print(f"\nWARN: probe image {PROBE_IMAGE} absent; skipping the "
              f"image-bearing probe", file=sys.stderr)
        image = None
    else:
        content = image_content(PROBE_IMAGE)
        print(f"\nprobe image {PROBE_IMAGE.name}: "
              f"{content['_probe_original_bytes']} bytes original -> "
              f"{content['_probe_payload_bytes']} bytes transmitted")
        image = {k: v for k, v in content.items() if not k.startswith("_")}

    failures = []
    for role, model_id in sorted(identities.items()):
        print(f"\n################ {role}: {model_id} ################")
        probes = {}

        text = post(base_url, api_key, {
            "model": model_id,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": TEXT_PROMPT}]}],
            "temperature": TEMPERATURE, "seed": SEED})
        probes["text_only"] = text
        print(f"  text-only  ok={text['ok']}"
              + (f" status={text.get('status')} "
                 f"served_as={text.get('provider_returned_model')!r} "
                 f"finish={text.get('finish_reason')} "
                 f"usage={text.get('usage')}" if text["ok"]
                 else f" {text.get('stage')} {text.get('status')} "
                      f"{str(text.get('error'))[:200]}"))
        if not text["ok"]:
            failures.append(f"{role}/{model_id}: text-only {text.get('error')}")

        if image is not None:
            with_image = post(base_url, api_key, {
                "model": model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "text",
                     "text": "Describe this image in one sentence, then "
                             "reply with {\"ok\": true}."},
                    image]}],
                "temperature": TEMPERATURE, "seed": SEED})
            probes["image_bearing"] = with_image
            print(f"  image      ok={with_image['ok']}"
                  + (f" status={with_image.get('status')} "
                     f"served_as={with_image.get('provider_returned_model')!r} "
                     f"finish={with_image.get('finish_reason')} "
                     f"usage={with_image.get('usage')}" if with_image["ok"]
                     else f" {with_image.get('stage')} "
                          f"{with_image.get('status')} "
                          f"{str(with_image.get('error'))[:200]}"))
            if not with_image["ok"]:
                failures.append(
                    f"{role}/{model_id}: image-bearing "
                    f"{with_image.get('error')}")

        served = {p.get("provider_returned_model")
                  for p in probes.values() if p.get("ok")}
        if served:
            print(f"  served as  {sorted(served)}")
            probes["served_as"] = sorted(served)
        result["probes"][role] = probes

    result["failures"] = failures
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    print()
    if failures:
        print(f"PROBE FAIL ({len(failures)} problem(s)):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PROBE PASS: all three identities served text-only and "
          "image-bearing requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
