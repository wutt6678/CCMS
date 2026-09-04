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
    "torch.use_deterministic_algorithms(True) is NOT enabled: the frozen "
    "Iteration 10 configuration never enabled it, and enabling it could "
    "perturb parity with the 9B reference. Repeat-stability is therefore "
    "established EMPIRICALLY per model (greedy decoding, batch size 1, "
    "fixed seed) and reported separately from the global flag.",
]

FAMILY_PARITY_NOTES = {
    "qwen35": [
        "Qwen3.5 reports 'fast path is not available' (flash-linear-"
        "attention / causal-conv1d absent) and falls back to the torch "
        "implementation. The same environment produced the frozen 9B "
        "panel, so the fallback is common to every Qwen arm rather than a "
        "per-checkpoint difference.",
    ],
    "ministral3": [
        "The checkpoint ships both HF shards and a Mistral-native "
        "consolidated.safetensors duplicate; only the sharded HF weights "
        "referenced by model.safetensors.index.json were downloaded, so "
        "there is no ambiguity about which file was loaded.",
        "The vendor default system prompt (SYSTEM_PROMPT.txt) is "
        "deliberately suppressed: the frozen CCMS system prompt is always "
        "messages[0], and every generation records "
        "vendor_default_system_prompt_injected plus the frozen prompt's "
        "verbatim presence so the suppression is verified, not assumed.",
        "transformers 5.14.1 warns that this tokenizer may need "
        "`fix_mistral_regex=True` or tokenization will be incorrect. That "
        "was tested rather than assumed: for the pinned revision the "
        "rendered prompt tokenizes to IDENTICAL ids with the flag unset, "
        "True and False (the pre-tokenizer is never replaced), so the "
        "warning is cosmetic here and the flag is left unset. The observed "
        "`tokenizer.fix_mistral_regex` value is recorded per generation.",
    ],
    "phi4_multimodal": [],
    "gemma3": [],
}


def parity_notes(adapter_name: str) -> list[str]:
    return PARITY_NOTES + FAMILY_PARITY_NOTES.get(adapter_name, [])


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
            "adapter_diagnostics": result.get("adapter_diagnostics"),
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
    # Prompt integrity: the frozen CCMS system prompt must be the ONLY
    # instruction reaching the model. A vendor default (e.g. Ministral-3's
    # Le Chat prompt) leaking in would mean this family was evaluated
    # under different instructions than the Qwen arm.
    for entry in smoke:
        for attempt in entry["attempts"]:
            diag = attempt.get("adapter_diagnostics") or {}
            if diag.get("vendor_default_system_prompt_injected"):
                problems.append(
                    f"{entry['variant']}: a vendor default system prompt was "
                    f"injected into the rendered prompt (markers "
                    f"{diag.get('vendor_default_markers_found')})")
            if diag.get("frozen_system_prompt_present_verbatim") is False:
                problems.append(
                    f"{entry['variant']}: the frozen CCMS system prompt is "
                    f"NOT present verbatim in the rendered prompt")
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
    parser.add_argument("--lock", default=None,
                        help="resolved_models.lock.yaml path (default: "
                             "outputs/iteration_11/preflight/"
                             "resolved_models.lock.yaml)")
    parser.add_argument("--update-lock", action="store_true", default=False,
                        help="Record the resolved immutable revision (and a "
                             "hashed pip-freeze dependency lock) so "
                             "confirmatory runs can pin it")
    parser.add_argument("--force-lock", action="store_true", default=False,
                        help="Allow deliberately re-pinning a model_key that "
                             "is already locked to a different revision")
    args = parser.parse_args()

    spec = resolve_model(args.model_key, confirmatory=False,
                         lock_path=args.lock)
    if spec.quantization != "none":
        print(f"FAIL {spec.model_key}: quantization {spec.quantization!r}")
        return 2
    out = Path(args.out) if args.out else \
        PREFLIGHT_ROOT / spec.model_key / "preflight.json"
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
        "parity_notes": parity_notes(spec.adapter),
        "lock": None,
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

    report["timestamp"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")

    if args.update_lock:
        from causal_mllm.replay.registry import (
            dependency_lock_snapshot, update_lock)
        if not is_immutable_revision(report["resolved_revision"]):
            report["problems"].append(
                "--update-lock needs a GPU smoke that resolved an immutable "
                "revision; re-run with --gpu-smoke")
        elif report["problems"]:
            report["problems"].append(
                "--update-lock refused: this preflight reported problems, so "
                "the revision is not eligible to be locked")
        else:
            dependency = dependency_lock_snapshot()
            size = report["size_metadata"] or {}
            measured = {
                key: size.get(key) for key in (
                    "checkpoint_parameter_count", "language_parameters",
                    "vision_parameters", "auxiliary_parameters",
                    "unclassified_parameters", "revision_used",
                    "stored_dtype_histogram", "n_shards", "n_tensors")
            } if size else None
            lock_path = update_lock(
                spec.model_key,
                revision=report["resolved_revision"],
                processor_revision=report["processor_revision"],
                evidence=str(out),
                resolved_at=report["timestamp"],
                measured_size=measured,
                dependency_lock=dependency,
                allow_change=args.force_lock,
                lock_path=args.lock)
            report["lock"] = {
                "path": str(lock_path),
                "revision": report["resolved_revision"],
                "processor_revision": report["processor_revision"],
                "dependency_lock": dependency,
                "forced": bool(args.force_lock),
            }

    report["status"] = "PASS" if not report["problems"] else "FAIL"
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
    if report["lock"]:
        dep = report["lock"]["dependency_lock"]
        print(f"lock: {report['lock']['path']}")
        print(f"  pip_freeze_sha256={dep['pip_freeze_sha256'][:16]} "
              f"({dep['n_packages']} packages, python {dep['python_version']})")
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}")
    print(f"wrote {out}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
