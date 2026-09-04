"""Iteration 11 per-model technical preflight (11.2 / 11.3 / 11.4).

For one registry ``--model-key`` this records, BEFORE any full-panel
generation:

  * the resolved immutable revision (registry/lock value, or the value
    resolved at load time in preflight mode);
  * declared checkpoint size metadata read from safetensors headers
    (never inferred from a response);
  * an opt-in GPU smoke: one image-bearing and one text-only generation
    on the FIRST frozen Scale-C family, repeated to check greedy
    determinism, with prompt/new-token slicing and image-token
    accounting asserted.

A single-family technical smoke is not a causal result: the frozen
fallback rule still requires any model substitution to happen before a
full-panel causal estimate is inspected.

Usage::

    python scripts/iter11_model_preflight.py --model-key qwen35_2b \
        --gpu-smoke            # defaults to cuda:3
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.data.io import read_jsonl  # noqa: E402
from causal_mllm.data.schemas import CausalFamily  # noqa: E402
from causal_mllm.replay.checkpoint_size import (  # noqa: E402
    checkpoint_size_metadata)
from causal_mllm.replay.config import ReplayConfig  # noqa: E402
from causal_mllm.replay.registry import (  # noqa: E402
    is_immutable_revision, resolve_model)
from causal_mllm.replay.runner import (  # noqa: E402
    build_chat_messages, verify_family_media)
from causal_mllm.seeds import get_git_commit, sha256_text  # noqa: E402

FROZEN_PANEL = REPO_ROOT / "outputs" / "scale_c" / "families_panel"
PREFLIGHT_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "preflight"
FROZEN_CAP = 1536
VISION_VARIANT = "cross_modal"
TEXT_VARIANT = "text_only"

# Runtime conditions observed in the frozen reference environment. They
# are recorded rather than "fixed", because changing them would break
# parity with the immutable Iteration 10 Qwen3.5-9B panel.
PARITY_NOTES = [
    "transformers 5.14.1 warns that `torch_dtype` is deprecated in favour "
    "of `dtype`; `torch_dtype` is retained deliberately so the load path "
    "matches the frozen HFLocalBackend that produced the 9B reference "
    "panel.",
    "Qwen3.5 reports 'fast path is not available' (flash-linear-attention "
    "/ causal-conv1d absent) and falls back to the torch implementation. "
    "The same environment produced the frozen 9B panel, so the fallback "
    "is common to every arm rather than a per-model difference.",
    "torch.use_deterministic_algorithms(True) is NOT enabled: the frozen "
    "Iteration 10 configuration never enabled it, and enabling it could "
    "perturb parity with the 9B reference. Repeat-stability is therefore "
    "established EMPIRICALLY per model (greedy decoding, batch size 1, "
    "fixed seed) and reported separately from the global flag.",
]


def _smoke_generation(adapter, family, config, variant, repeats):
    """Generate one variant ``repeats`` times; check determinism."""
    chat = build_chat_messages(family, variant, config)
    attempts = []
    for _ in range(repeats):
        result = adapter.generate(chat)
        attempts.append({
            "response_sha256": sha256_text(result["response"]),
            "response_chars": len(result["response"]),
            "response_head": result["response"][:200],
            "input_token_count": result["input_token_count"],
            "image_token_count": result["image_token_count"],
            "output_token_count": result["output_token_count"],
            "finish_reason": result["finish_reason"],
            "hit_max_new_tokens": result["hit_max_new_tokens"],
            "serialized_prompt_hash": result["serialized_prompt_hash"],
            "semantic_prompt_hash": result["semantic_prompt_hash"],
            "ordered_image_hashes": result["ordered_image_hashes"],
        })
    responses = {a["response_sha256"] for a in attempts}
    return {
        "variant": variant,
        "repeats": repeats,
        "attempts": attempts,
        "deterministic": len(responses) == 1,
        "n_distinct_responses": len(responses),
    }


def _check_smoke(smoke, size_meta) -> list[str]:
    """Technical eligibility assertions; returns a list of failures."""
    problems: list[str] = []
    vision = next(s for s in smoke if s["variant"] == VISION_VARIANT)
    text = next(s for s in smoke if s["variant"] == TEXT_VARIANT)
    first = vision["attempts"][0]

    if not first["response_chars"]:
        problems.append("vision-bearing generation returned an empty string")
    if first["input_token_count"] <= 0:
        problems.append("input_token_count must be positive")
    if first["output_token_count"] <= 0:
        problems.append("output_token_count must be positive")
    if first["finish_reason"] not in {"eos", "stop", "length"}:
        problems.append(f"unexpected finish_reason {first['finish_reason']!r}")
    if first["image_token_count"] <= 0:
        problems.append(
            "image_token_count must be > 0 for the image-bearing variant")
    if len(first["ordered_image_hashes"]) != 1:
        problems.append("expected exactly one ordered image hash")
    if text["attempts"][0]["image_token_count"] != 0:
        problems.append("text-only variant must report 0 image tokens")
    if text["attempts"][0]["ordered_image_hashes"]:
        problems.append("text-only variant must reference no images")
    for entry in smoke:
        if not entry["deterministic"]:
            problems.append(
                f"greedy decoding is not deterministic for "
                f"{entry['variant']}: {entry['n_distinct_responses']} "
                f"distinct responses over {entry['repeats']} repeats")
    if size_meta.get("unclassified_parameters"):
        problems.append(
            f"{size_meta['unclassified_parameters']} checkpoint parameters "
            f"could not be attributed to language/vision/auxiliary: "
            f"{size_meta['unclassified_prefixes']}")
    if not size_meta.get("vision_parameters"):
        problems.append(
            "checkpoint reports no vision parameters — not a multimodal "
            "model eligible for the common prompt")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iteration 11 per-model technical preflight")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--device", default="cuda:3",
                        help="GPU slot (standing instruction: cuda:3). The "
                             "slot is NOT part of the run fingerprint; the "
                             "hardware class is.")
    parser.add_argument("--input-dir", default=str(FROZEN_PANEL))
    parser.add_argument("--max-new-tokens", type=int, default=FROZEN_CAP)
    parser.add_argument("--repeats", type=int, default=2,
                        help="Greedy determinism repeats per variant")
    parser.add_argument("--gpu-smoke", action="store_true", default=False,
                        help="Load the checkpoint and generate (needs a GPU)")
    parser.add_argument("--n-families", type=int, default=1,
                        help="Smoke families (technical check only)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    spec = resolve_model(args.model_key, confirmatory=False)
    if spec.quantization != "none":
        print(f"FAIL {spec.model_key}: quantization {spec.quantization!r}")
        return 2
    config = ReplayConfig(
        model_name=spec.model_id,
        model_revision=spec.revision,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        enable_thinking=spec.thinking_mode)

    panel = Path(args.input_dir) / "validated_families.jsonl"
    report: dict = {
        "iteration": "11",
        "stage": "model_preflight",
        "model_key": spec.model_key,
        "model_spec": spec.to_dict(),
        "registry_revision": spec.revision,
        "revision_is_immutable": is_immutable_revision(spec.revision),
        "generation_config": config.generation_settings(),
        "effective_decoding": {
            "do_sample": config.do_sample, "temperature": None,
            "top_p": None, "top_k": None, "num_beams": 1,
            "max_new_tokens": config.max_new_tokens},
        "system_prompt_sha256": sha256_text(config.system_prompt),
        "prompt_template_revision": config.prompt_template_revision,
        "dataset": {
            "input_dir": str(Path(args.input_dir).resolve()),
            "validated_families_sha256": (sha256_text(
                panel.read_text(encoding="utf-8"))
                if panel.exists() else None),
            "n_families_smoked": args.n_families,
        },
        "device": args.device,
        "code_commit": get_git_commit(),
        "gpu_smoke": None,
        "runtime_metadata": None,
        "determinism": None,
        "parity_notes": PARITY_NOTES,
        "resolved_revision": None,
        "processor_revision": None,
        "size_metadata": None,
        "problems": [],
        "status": "PENDING",
    }

    # Declared size: header-only, no weights, no GPU.
    try:
        report["size_metadata"] = checkpoint_size_metadata(
            spec.model_id, revision=spec.revision)
    except Exception as exc:
        report["problems"].append(f"size metadata unavailable: {exc}")

    if args.gpu_smoke:
        from causal_mllm.replay.adapters import build_adapter
        families = [CausalFamily.from_dict(rec)
                    for rec in read_jsonl(panel)][:args.n_families]
        for family in families:
            report["problems"].extend(
                f"media: {p}" for p in verify_family_media(family))
        adapter = build_adapter(spec, config, device=args.device)
        adapter.load()
        report["resolved_revision"] = adapter.model_revision()
        report["processor_revision"] = adapter.processor_revision()
        report["runtime_metadata"] = adapter.runtime_metadata()
        report["gpu_smoke"] = [
            _smoke_generation(adapter, families[0], config, variant,
                              args.repeats)
            for variant in (VISION_VARIANT, TEXT_VARIANT)]
        report["problems"].extend(
            _check_smoke(report["gpu_smoke"], report["size_metadata"] or {}))
        report["determinism"] = {
            "torch_deterministic_algorithms_enabled": bool(
                report["runtime_metadata"].get("deterministic_algorithms")),
            "greedy_decoding": config.do_sample is False,
            "batch_size": 1,
            "requested_seed": config.seed,
            "repeats_per_variant": args.repeats,
            "empirical_repeat_stability": {
                entry["variant"]: entry["deterministic"]
                for entry in report["gpu_smoke"]},
            "all_variants_repeat_stable": all(
                entry["deterministic"] for entry in report["gpu_smoke"]),
            "note": "The global torch determinism flag is reported as-is "
                    "and is intentionally left disabled for parity with "
                    "the frozen Iteration 10 9B reference; empirical "
                    "repeat-stability is the operative evidence.",
        }
        if not is_immutable_revision(report["resolved_revision"]):
            report["problems"].append(
                f"resolved revision {report['resolved_revision']!r} is not "
                f"an immutable 40-hex SHA — cannot be locked")
        elif spec.revision is not None and \
                report["resolved_revision"] != spec.revision:
            report["problems"].append(
                f"revision mismatch: requested {spec.revision} but loaded "
                f"{report['resolved_revision']}")

    report["status"] = "PASS" if not report["problems"] else "FAIL"
    report["timestamp"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")

    out = Path(args.out) if args.out else \
        PREFLIGHT_ROOT / spec.model_key / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    size = report["size_metadata"] or {}
    print(f"model_key: {spec.model_key} ({spec.model_id})")
    print(f"status: {report['status']}")
    print(f"registry_revision: {spec.revision}")
    print(f"resolved_revision: {report['resolved_revision']}")
    if size:
        print("checkpoint_parameter_count: "
              f"{size.get('checkpoint_parameter_count'):,}")
        print(f"  language: {size.get('language_parameters'):,}  "
              f"vision: {size.get('vision_parameters'):,}  "
              f"auxiliary: {size.get('auxiliary_parameters'):,}  "
              f"unclassified: {size.get('unclassified_parameters'):,}")
        print(f"  stored dtypes: {size.get('stored_dtype_histogram')}")
    for entry in report["gpu_smoke"] or []:
        attempt = entry["attempts"][0]
        print(f"smoke[{entry['variant']}]: deterministic="
              f"{entry['deterministic']} in={attempt['input_token_count']} "
              f"img={attempt['image_token_count']} "
              f"out={attempt['output_token_count']} "
              f"finish={attempt['finish_reason']}")
    det = report["determinism"]
    if det:
        print(f"determinism: repeat_stable="
              f"{det['all_variants_repeat_stable']} "
              f"(torch_flag={det['torch_deterministic_algorithms_enabled']}, "
              f"greedy={det['greedy_decoding']}, seed={det['requested_seed']}, "
              f"repeats={det['repeats_per_variant']})")
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}")
    print(f"wrote {out}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
