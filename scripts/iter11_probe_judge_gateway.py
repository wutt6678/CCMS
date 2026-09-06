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

#: The image-bearing probe's instruction. Hoisted to a constant because the
#: vision verdict needs the SAME text sent with and without the image: only
#: then is a prompt_tokens difference attributable to the image rather than to
#: the wording. The two requests this probe originally sent used different
#: text, so their token counts were not comparable and the probe could report
#: an identity as usable while carrying no evidence about whether it could
#: see -- which is how a blind primary judge reached a frozen ensemble.
IMAGE_PROMPT = ("Describe this image in one sentence, then reply with "
                '{"ok": true}.')

#: Where the per-identity vision verdict is written. The judge pipeline reads
#: this so the sensitivity artifact can label a blind primary's arm as
#: vision-ablated on measured grounds instead of on an assertion in source
#: that would go stale the moment the gateway's models change.
VISION_ARTIFACT = REPO_ROOT / "outputs" / "iteration_11" / "diagnostics" \
    / "judge_vision" / "identity_vision_capability.json"


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


def _usage_int(probe: dict, *keys) -> int | None:
    """A numeric field out of the provider's usage breakdown, or None."""
    node = probe.get("usage") or {}
    for key in keys[:-1]:
        node = node.get(key) or {}
    value = node.get(keys[-1])
    return int(value) if isinstance(value, (int, float)) else None


def assess_vision(control: dict, with_image: dict) -> dict:
    """Did this identity actually take the image in?

    Two independent signals, because neither is safe alone:

    * ``prompt_tokens_details.image_tokens`` from the provider's own usage
      breakdown. Present and positive means the provider tokenized pixels.
    * the ``prompt_tokens`` difference between the SAME text sent with and
      without the image. A model that discards the image reports the same
      count both ways.

    What the response SAYS is recorded but never decisive. A blind model asked
    to describe an image either declines or confabulates a plausible
    description, so a fluent answer is not evidence of sight and a refusal is
    not evidence of blindness -- only the accounting the provider does on its
    own input is.
    """
    image_tokens = _usage_int(with_image, "prompt_tokens_details",
                              "image_tokens")
    t_text = _usage_int(control, "prompt_tokens")
    t_image = _usage_int(with_image, "prompt_tokens")
    delta = (t_image - t_text) if None not in (t_text, t_image) else None
    out = {
        "image_tokens": image_tokens,
        "prompt_tokens_text_only": t_text,
        "prompt_tokens_with_image": t_image,
        "prompt_tokens_delta": delta,
        "text_response_head": str(control.get("response_head", ""))[:160],
        "image_response_head": str(with_image.get("response_head", ""))[:160],
    }
    if not (control.get("ok") and with_image.get("ok")):
        out.update(sees_image=None, verdict="unmeasurable",
                   reason="one of the two probe requests did not succeed, so "
                          "there is no pair of token counts to compare")
        return out
    if image_tokens is None and delta is None:
        # No accounting to decide from. Reporting "blind" here would be a
        # conclusion drawn from an absence of evidence, and would label an arm
        # vision-ablated on the strength of a provider that simply does not
        # itemize its usage.
        out.update(sees_image=None, verdict="unmeasurable",
                   reason="the provider reported neither image_tokens nor a "
                          "prompt_tokens count for both requests, so there is "
                          "nothing to decide from")
        return out
    signals = []
    if image_tokens:
        signals.append(f"image_tokens={image_tokens}")
    if delta:
        signals.append(f"prompt_tokens rose by {delta} when the same text "
                       f"was sent with the image attached")
    if signals:
        out.update(sees_image=True, verdict="sees_image",
                   reason="; ".join(signals))
    else:
        out.update(
            sees_image=False, verdict="blind",
            reason="no image_tokens reported and prompt_tokens is identical "
                   f"({t_text}) with and without the image, so the pixels were "
                   "discarded before the model saw them")
    return out


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
    parser.add_argument("--also-probe", action="append", default=[],
                        metavar="MODEL_ID",
                        help="measure vision for an identity OUTSIDE the frozen "
                             "ensemble, recorded under reference_identities. "
                             "It never affects the exit code or the ensemble "
                             "verdicts; it exists so a claim about how blind "
                             "models behave can rest on a measurement instead "
                             "of an anecdote")
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
    # The frozen three first, then any reference identities. A reference
    # identity is measured the same way but is held outside every invariant:
    # the three-distinct-identity check, the failure list and the artifact's
    # `identities` section that the judge pipeline reads. Letting one of them
    # into those would let a model nobody froze decide how an arm is labelled.
    reference = sorted({m for m in args.also_probe
                        if m and m not in set(identities.values())})
    skipped_reference = sorted({m for m in args.also_probe
                                if m in set(identities.values())})
    if skipped_reference:
        print(f"\nNOTE --also-probe {skipped_reference} is already a frozen "
              f"ensemble identity; measured once, as itself")
    probe_targets = [(role, mid, True)
                     for role, mid in sorted(identities.items())]
    probe_targets += [(mid, mid, False) for mid in reference]

    for role, model_id, is_ensemble in probe_targets:
        if is_ensemble:
            print(f"\n################ {role}: {model_id} ################")
        else:
            print(f"\n######## REFERENCE, not in the frozen ensemble: "
                  f"{model_id} ########")
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
            if is_ensemble:
                failures.append(
                    f"{role}/{model_id}: text-only {text.get('error')}")
            else:
                print(f"  WARN       a reference identity is unreachable, "
                      f"which fails nothing: {str(text.get('error'))[:160]}")

        if image is not None:
            # The control: the SAME instruction with no image attached. Its
            # prompt_tokens is the baseline the image-bearing count is read
            # against, so the difference is attributable to the pixels alone.
            control = post(base_url, api_key, {
                "model": model_id,
                "messages": [{"role": "user",
                              "content": [{"type": "text",
                                           "text": IMAGE_PROMPT}]}],
                "temperature": TEMPERATURE, "seed": SEED})
            probes["text_control_for_image"] = control
            with_image = post(base_url, api_key, {
                "model": model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
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
                if is_ensemble:
                    failures.append(
                        f"{role}/{model_id}: image-bearing "
                        f"{with_image.get('error')}")
                else:
                    print(f"  WARN       image-bearing request refused, which "
                          f"fails nothing: "
                          f"{str(with_image.get('error'))[:160]}")
            vision = assess_vision(control, with_image)
            probes["vision"] = vision
            print(f"  VISION     {vision['verdict']}  "
                  f"image_tokens={vision['image_tokens']} "
                  f"prompt_tokens {vision['prompt_tokens_text_only']} -> "
                  f"{vision['prompt_tokens_with_image']} "
                  f"(delta {vision['prompt_tokens_delta']})")
            print(f"             {vision['reason']}")
            if vision["verdict"] == "blind":
                # Not a reachability failure: the identity serves requests and
                # returns parseable output. It is a statement about what the
                # ensemble can measure, and it belongs in the evidence rather
                # than in the exit code.
                print(f"  NOTE       {model_id} cannot see the family media; "
                      f"any arm of the analysis that rests on this identity "
                      f"alone is a VISION-ABLATION arm, not a model-choice "
                      f"arm")

        served = {p.get("provider_returned_model")
                  for p in probes.values() if p.get("ok")}
        if served:
            print(f"  served as  {sorted(served)}")
            probes["served_as"] = sorted(served)
        result["probes"][role] = probes

    result["failures"] = failures

    # The vision verdict, keyed by MODEL ID rather than by role: the pipeline
    # resolves its identities from the same credentials file and looks them up
    # by id, and a role key would silently mislabel an identity if the
    # credentials were ever repointed.
    vision_by_identity = {}
    for role, model_id in sorted(identities.items()):
        vision_by_identity[model_id] = (
            result["probes"].get(role, {}).get("vision")
            or {"verdict": "unmeasured", "reason": "no image probe was run"})
    result["vision"] = vision_by_identity

    # Reference identities get their own section, kept out of `identities` so
    # the judge pipeline -- which looks up arms by model id -- can never pick
    # up a model nobody froze.
    reference_vision = {}
    for model_id in reference:
        reference_vision[model_id] = (
            result["probes"].get(model_id, {}).get("vision")
            or {"verdict": "unmeasured", "reason": "no image probe was run"})
    result["reference_vision"] = reference_vision

    VISION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    VISION_ARTIFACT.write_text(json.dumps({
        "produced_by": "scripts/iter11_probe_judge_gateway.py",
        "gateway": base_url,
        "probe_image": {
            "path": str(PROBE_IMAGE.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(
                PROBE_IMAGE.read_bytes()).hexdigest() if PROBE_IMAGE.exists()
            else None,
        },
        "roles": identities,
        "method": "the same instruction sent with and without the image; the "
                  "verdict comes from the provider's own token accounting, "
                  "never from what the response says",
        "identities": vision_by_identity,
        "reference_identities": reference_vision,
        "reference_note": (
            "measured with --also-probe to document how blindness presents, "
            "NOT members of the frozen ensemble: nothing in the judge pipeline "
            "reads this section and no arm is labelled from it")
        if reference_vision else None,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {VISION_ARTIFACT}")

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
