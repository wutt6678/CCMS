"""Frozen replay stage (Iteration 8).

Replays the stored variant histories of VALIDATED families through a
frozen model and stores raw trajectory -> response records, strictly
separated from the dataset artifacts.

Hard gates:

  * Input is ``validated_families.jsonl`` ONLY. A directory without
    it is rejected outright (never raw families.jsonl).
  * Histories are replayed EXACTLY as persisted: no attacker, no
    interactive regeneration of intermediate turns.
  * Identical system prompt and generation settings for every variant
    (one frozen ``ReplayConfig`` for the whole run).
  * All referenced media are verified (exist + hash-match against
    ``source_media``) immediately before inference; problems become
    classified media failures, never responses.
  * Every (family, variant) pair is ATTEMPTED exactly once; the run
    fails loudly if any variant is missing. Generation failures are
    recorded separately with an error category — an OOM/media/context
    error never becomes a safe/refusal label.

Output layout (separate from dataset artifacts)::

    <output_root>/<run_id>/replay_outputs.jsonl
    <output_root>/<run_id>/replay_failures.jsonl
    <output_root>/<run_id>/replay_report.json

This iteration produces trajectory -> raw response ONLY. Judging and
the causal estimands (Delta_T, Delta_V, Delta_TV, reset/order effects)
belong to Iteration 9.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl, write_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.replay.backend import HFLocalBackend, ReplayBackend
from causal_mllm.replay.config import ReplayConfig
from causal_mllm.replay.errors import ReplayError, ReplayMediaError, classify_error
from causal_mllm.seeds import get_git_commit, is_git_dirty, sha256_text
from causal_mllm.validation.relations import _file_sha256

log = get_logger(__name__)


def _backend_model_name(backend: ReplayBackend, config: ReplayConfig) -> str:
    """Prefer the backend's own model identity (injectable stubs)."""
    name = getattr(backend, "model_name", None)
    if callable(name):
        return name()
    return config.model_name


VALIDATED_FAMILIES_FILE = "validated_families.jsonl"
REPLAY_OUTPUTS_FILE = "replay_outputs.jsonl"
REPLAY_FAILURES_FILE = "replay_failures.jsonl"
REPLAY_REPORT_FILE = "replay_report.json"


def resolved_fingerprint(backend: ReplayBackend,
                         config: ReplayConfig,
                         input_dir: str | Path | None = None) -> str:
    """One hash identifying what ACTUALLY produced the responses.

    Binds: backend, model + revision, processor revision,
    enable_thinking, torch_dtype, generation settings, prompt template
    revision, system-prompt hash, validated_families.jsonl SHA256
    (when input_dir is given), transformers version, torch version,
    CUDA version, and repository commit.  The config fingerprint may
    contain ``model_revision=None`` (resolved at load time); this
    fingerprint uses the RESOLVED values.
    """
    validated_families_sha256 = None
    if input_dir is not None:
        vf_path = Path(input_dir) / VALIDATED_FAMILIES_FILE
        if vf_path.exists():
            validated_families_sha256 = _file_sha256(vf_path)
    payload = json.dumps({
        "backend": config.backend,
        "model": _backend_model_name(backend, config),
        "model_revision": backend.model_revision(),
        "processor_revision": backend.processor_revision(),
        "enable_thinking": config.enable_thinking,
        "torch_dtype": config.torch_dtype,
        "generation_config": config.generation_settings(),
        "prompt_template_revision": config.prompt_template_revision,
        "system_prompt_sha256": sha256_text(config.system_prompt),
        "validated_families_sha256": validated_families_sha256,
        "transformers_version": backend.transformers_version(),
        "torch_version": backend.torch_version(),
        "cuda_version": backend.cuda_version(),
        "git_commit": get_git_commit(),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Hardware keys that identify the SCHEDULING SLOT rather than the
#: hardware class. They are recorded for operational honesty but must
#: not enter the run fingerprint.
HARDWARE_SCHEDULING_KEYS = frozenset(
    {"device_index", "requested_device", "device"})


def fingerprint_hardware(hardware: dict | None) -> dict | None:
    """Hardware identity that determines output behaviour.

    The scheduling slot (``device_index`` / ``requested_device``) is
    deliberately EXCLUDED: the frozen ``resolved_fingerprint`` never
    bound the device either, all four local GPUs are the same model, and
    cross-slot generation was verified byte-identical during the 11.2
    preflight — so moving a run between slots must not invalidate
    resume.  The hardware CLASS (name, compute capability, memory) is
    retained, because that can change numerics.
    """
    if not isinstance(hardware, dict):
        return None
    return {key: value for key, value in sorted(hardware.items())
            if key not in HARDWARE_SCHEDULING_KEYS}


def iteration11_run_fingerprint(
    backend: ReplayBackend,
    config: ReplayConfig,
    input_dir: Path,
    model_spec: "ResolvedModel | None" = None,
    hardware: dict | None = None,
) -> str:
    """Iteration 11 resolved-run fingerprint.

    Extends the frozen ``resolved_fingerprint`` payload with the model
    dimension (model_key, adapter, dtype, quantization, hardware class)
    so the 9B reference and each new target are separately identified,
    while ``resolved_fingerprint`` itself is left unchanged for the
    legacy single-model path.  Hashed identically (sha256 over sort_keys
    JSON).

    Like the frozen fingerprint this binds what DETERMINES the output,
    not the scheduling slot: ``config.fingerprint()`` is intentionally
    NOT included because it serializes ``device``.
    """
    validated_path = input_dir / "validated_families.jsonl"
    payload: dict[str, Any] = {
        "backend": config.backend,
        "model": backend.model_name(),
        "requested_model_revision": config.model_revision,
        "resolved_model_revision": backend.model_revision(),
        "processor_revision": backend.processor_revision(),
        "prompt_template_revision": config.prompt_template_revision,
        "system_prompt_sha256": sha256_text(config.system_prompt),
        "generation_config": config.generation_settings(),
        "enable_thinking": config.enable_thinking,
        "torch_dtype": config.torch_dtype,
        "validated_families_sha256": _file_sha256(validated_path),
        "transformers_version": backend.transformers_version(),
        "torch_version": backend.torch_version(),
        "cuda_version": backend.cuda_version(),
        "code_commit": get_git_commit(),
        "hardware": fingerprint_hardware(hardware),
    }
    if model_spec is not None:
        payload["model_key"] = model_spec.model_key
        payload["adapter"] = model_spec.adapter
        payload["dtype"] = model_spec.dtype
        payload["quantization"] = model_spec.quantization
        payload["trust_remote_code"] = model_spec.trust_remote_code
        payload["thinking_mode"] = model_spec.thinking_mode
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def default_run_id(config: ReplayConfig) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    model = config.model_name.replace("/", "-").lower()
    return f"{stamp}-{model}-{config.fingerprint()[:8]}"


def verify_family_media(family: CausalFamily) -> list[str]:
    """Existence + hash verification of every referenced media file."""
    problems: list[str] = []
    recorded = {
        media["path"]: media["sha256"]
        for atom in family.semantic_atoms
        for media in atom.source_media
    }
    referenced = {
        path
        for variant in family.variants.values()
        for message in variant.messages
        for path in message.images
    }
    for path in sorted(referenced):
        sha = recorded.get(path)
        if sha is None:
            problems.append(f"media {path}: not recorded in source_media")
            continue
        file_path = Path(path)
        if not file_path.exists():
            problems.append(f"media {path}: file missing")
            continue
        if _file_sha256(file_path) != sha:
            problems.append(
                f"media {path}: hash differs from recorded source_media")
    return problems


def build_chat_messages(family: CausalFamily, variant_name: str,
                        config: ReplayConfig) -> list[dict]:
    """System prompt + stored history + terminal q*, exactly as built.

    No attacker, no rewriting: turn texts and image paths come from
    the persisted variant verbatim.
    """
    variant = family.variants[variant_name]
    chat: list[dict] = [{
        "role": "system",
        "content": [{"type": "text", "text": config.system_prompt}],
    }]
    for message in variant.messages:
        content: list[dict] = []
        for image_path in message.images:
            content.append({"type": "image", "image": image_path})
        if message.text is not None:
            content.append({"type": "text", "text": message.text})
        chat.append({"role": message.role, "content": content})
    return chat


def _base_record(run_id: str, family: CausalFamily, variant: str,
                 config: ReplayConfig, backend: ReplayBackend,
                 model_spec: "ResolvedModel | None" = None,
                 run_prov: dict | None = None) -> dict:
    terminal = family.variants[variant].messages[-1]
    n_images = sum(len(m.images) for m in family.variants[variant].messages)
    record = {
        "run_id": run_id,
        "family_id": family.family_id,
        "source_id": family.source.get("source_id"),
        "variant": variant,
        "model": config.model_name,
        "requested_model_revision": config.model_revision,
        "resolved_model_revision": backend.model_revision(),
        "revision_pinned": config.model_revision is not None,
        # Legacy alias for backward compatibility
        "model_revision": backend.model_revision(),
        "prompt_template_revision": config.prompt_template_revision,
        "system_prompt_sha256": sha256_text(config.system_prompt),
        "generation_config": config.generation_settings(),
        "terminal_sha256": sha256_text(terminal.text or ""),
        "n_images": n_images,
        "input_token_count": None,
        "image_token_count": None,
        "output_token_count": None,
        "finish_reason": None,
        "hit_max_new_tokens": None,
        "response": None,
        "error": None,
    }
    if model_spec is not None:
        # Iteration 11 additive provenance (spec Section 8). Legacy
        # single-model records (model_spec=None) keep the exact shape
        # above, so frozen Iteration 8-10 evidence is unaffected.
        rp = run_prov or {}
        record.update({
            "model_key": model_spec.model_key,
            "model_id": model_spec.model_id,
            "adapter": model_spec.adapter,
            "dtype": model_spec.dtype,
            "quantization": model_spec.quantization,
            "sample_id": family.source.get("source_id"),
            "variant_id": variant,
            "code_commit": rp.get("code_commit"),
            "dataset_manifest_hash": rp.get("dataset_manifest_hash"),
            "resolved_run_fingerprint": rp.get("resolved_run_fingerprint"),
            "semantic_prompt_hash": None,
            "serialized_prompt_hash": None,
            "ordered_image_hashes": None,
            "requested_seed": config.seed,
            "effective_seed": config.seed,
            "deterministic_algorithms": rp.get("deterministic_algorithms"),
            "runtime_versions": rp.get("runtime_versions"),
            "hardware": rp.get("hardware"),
            "truncated": None,
        })
    return record


def _replay_family(run_id: str, family: CausalFamily, config: ReplayConfig,
                   backend: ReplayBackend,
                   model_spec: "ResolvedModel | None" = None,
                   run_prov: dict | None = None,
                   skip_variants: set | None = None
                   ) -> tuple[list[dict], list[dict]]:
    outputs: list[dict] = []
    failures: list[dict] = []
    skip = skip_variants or set()

    # Verify ALL media immediately before inference; fail loudly and
    # never let a missing/corrupt file masquerade as a response.
    try:
        problems = verify_family_media(family)
        if problems:
            raise ReplayMediaError(
                f"{family.family_id}: " + "; ".join(problems))
    except Exception as exc:  # media problems are per-family
        error = classify_error(exc)
        for variant in ALL_VARIANT_NAMES:
            if variant in skip:
                continue
            record = _base_record(run_id, family, variant, config, backend,
                                  model_spec, run_prov)
            record["error"] = error
            failures.append(record)
        return outputs, failures

    for variant in ALL_VARIANT_NAMES:
        if variant in skip:
            continue
        record = _base_record(run_id, family, variant, config, backend,
                              model_spec, run_prov)
        try:
            result = backend.generate(
                build_chat_messages(family, variant, config))
        except Exception as exc:
            record["error"] = classify_error(exc)
            failures.append(record)
            continue
        record["response"] = result["response"]
        record["input_token_count"] = result.get("input_token_count")
        record["image_token_count"] = result.get("image_token_count")
        record["output_token_count"] = result.get("output_token_count")
        record["finish_reason"] = result.get("finish_reason")
        record["hit_max_new_tokens"] = result.get("hit_max_new_tokens")
        if model_spec is not None:
            record["semantic_prompt_hash"] = result.get("semantic_prompt_hash")
            record["serialized_prompt_hash"] = result.get(
                "serialized_prompt_hash")
            record["ordered_image_hashes"] = result.get("ordered_image_hashes")
            record["effective_decoding"] = result.get("effective_decoding")
            record["truncated"] = result.get("finish_reason") == "length"
        outputs.append(record)
    return outputs, failures



def run_replay_stage(
    input_dir: str | Path,
    output_root: str | Path,
    config: ReplayConfig | None = None,
    backend: ReplayBackend | None = None,
    max_families: int | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    model_spec: "ResolvedModel | None" = None,
    resume: bool = False,
) -> dict:
    """Replay validated families; persist outputs/failures/report.

    Args:
        input_dir: Dataset dir containing validated_families.jsonl.
        output_root: Root for replay runs (separate from datasets).
        config: Frozen replay settings (defaults: Qwen3.5-9B, greedy).
        backend: Injectable backend; defaults per config.backend.
        max_families: Optional limit (smoke runs).
        run_id: Override the generated run id.
        overwrite: If False (default), fail when the run directory
            already contains evidence files.  This prevents accidental
            overwriting of retained evidence.

    Raises:
        ReplayError: On missing validated_families.jsonl, missing
            (family, variant) coverage, or an existing run directory
            when overwrite is False.
    """
    config = config or ReplayConfig()
    input_dir = Path(input_dir)
    source_path = input_dir / VALIDATED_FAMILIES_FILE
    if not source_path.exists():
        raise ReplayError(
            f"{source_path} not found — Iteration 8 consumes "
            f"validated_families.jsonl ONLY; run the validation stage "
            f"first (raw families.jsonl is never replayed)")
    families = [CausalFamily.from_dict(rec)
                for rec in read_jsonl(source_path)]
    if max_families is not None:
        families = families[:max_families]

    if backend is None:
        if model_spec is not None:
            from causal_mllm.replay.adapters import build_adapter
            backend = build_adapter(model_spec, config)
            backend.load()
        elif config.backend == "hf_local":
            backend = HFLocalBackend(config).load()
        else:
            raise ReplayError(f"Unknown backend '{config.backend}'")

    # Fail loudly if the requested revision doesn't match what was
    # actually loaded — provenance integrity.
    if (config.model_revision is not None
            and backend.model_revision() != config.model_revision):
        raise ReplayError(
            f"revision mismatch: requested "
            f"{config.model_revision!r} but loaded "
            f"{backend.model_revision()!r}")

    # Iteration 11 run-level provenance (empty for the legacy path, so
    # single-model records/reports are unchanged).
    run_prov: dict = {}
    if model_spec is not None:
        rt = (backend.runtime_metadata()
              if hasattr(backend, "runtime_metadata") else {})
        hardware = rt.get("hardware")
        run_prov = {
            "model_key": model_spec.model_key,
            "adapter": model_spec.adapter,
            "code_commit": get_git_commit(),
            "dataset_manifest_hash": _file_sha256(source_path),
            "resolved_run_fingerprint": iteration11_run_fingerprint(
                backend, config, input_dir, model_spec, hardware),
            "deterministic_algorithms": rt.get("deterministic_algorithms"),
            "runtime_versions": {
                "transformers": backend.transformers_version(),
                "torch": backend.torch_version(),
                "cuda": backend.cuda_version(),
            },
            "hardware": hardware,
        }

    run_id = run_id or default_run_id(config)
    run_dir = Path(output_root) / run_id

    # Resume: continue an interrupted run at (family_id, variant)
    # granularity, refusing to mix records from a different resolved run
    # fingerprint / model_key.  Only successful outputs count as
    # complete; failed variants are retried.
    existing_outputs: list[dict] = []
    done: set[tuple[str, str]] = set()
    if resume:
        if (run_dir / REPLAY_OUTPUTS_FILE).exists():
            existing_outputs = read_jsonl(run_dir / REPLAY_OUTPUTS_FILE)
            cur_fp = run_prov.get("resolved_run_fingerprint")
            cur_key = model_spec.model_key if model_spec else None
            for rec in existing_outputs:
                prior_fp = rec.get("resolved_run_fingerprint")
                if cur_fp is not None and prior_fp not in (None, cur_fp):
                    raise ReplayError(
                        f"resume mismatch: stored resolved_run_fingerprint "
                        f"{prior_fp!r} != current {cur_fp!r}; refusing to "
                        f"append to a run with different provenance")
                prior_key = rec.get("model_key")
                if cur_key is not None and prior_key not in (None, cur_key):
                    raise ReplayError(
                        f"resume mismatch: stored model_key {prior_key!r} != "
                        f"current {cur_key!r}")
                pair = (rec["family_id"], rec["variant"])
                if pair in done:
                    raise ReplayError(
                        f"resume: duplicate stored record for "
                        f"{pair[0]}:{pair[1]}")
                done.add(pair)
            log.info("Resume %s: %d (family, variant) pairs already done",
                     run_id, len(done))
    elif run_dir.exists() and not overwrite:
        # Evidence-protection guard: refuse to overwrite existing evidence
        # unless the caller explicitly opts in.
        evidence_files = [REPLAY_OUTPUTS_FILE, REPLAY_FAILURES_FILE,
                         REPLAY_REPORT_FILE]
        existing = [f for f in evidence_files
                    if (run_dir / f).exists()]
        if existing:
            raise ReplayError(
                f"run directory {run_dir} already contains evidence "
                f"{existing}; pass overwrite=True or use a new run_id "
                f"to avoid overwriting retained evidence")
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = list(existing_outputs)
    failures: list[dict] = []
    for family in families:
        skip = {variant for (fid, variant) in done
                if fid == family.family_id}
        if len(skip) == len(ALL_VARIANT_NAMES):
            log.info("Replay %s: already complete, skipping",
                     family.family_id)
            continue
        family_outputs, family_failures = _replay_family(
            run_id, family, config, backend, model_spec, run_prov,
            skip_variants=skip)
        outputs.extend(family_outputs)
        failures.extend(family_failures)
        log.info("Replay %s: %d outputs, %d failures",
                 family.family_id, len(family_outputs),
                 len(family_failures))

    expected = len(families) * len(ALL_VARIANT_NAMES)
    attempted = len(outputs) + len(failures)
    covered = {(r["family_id"], r["variant"])
               for r in outputs + failures}
    missing = [
        f"{family.family_id}:{variant}"
        for family in families
        for variant in ALL_VARIANT_NAMES
        if (family.family_id, variant) not in covered
    ]
    if missing or attempted != expected:
        raise ReplayError(
            f"replay coverage broken: attempted {attempted} != expected "
            f"{expected}; missing: {missing}")

    write_jsonl(run_dir / REPLAY_OUTPUTS_FILE, outputs)
    write_jsonl(run_dir / REPLAY_FAILURES_FILE, failures)

    input_tokens = [r["input_token_count"] for r in outputs
                    if r["input_token_count"] is not None]
    image_tokens = [r["image_token_count"] for r in outputs
                    if r["image_token_count"] is not None]
    output_tokens = [r["output_token_count"] for r in outputs
                     if r["output_token_count"] is not None]

    # Truncation BY VARIANT: a global rate can hide condition-specific
    # imbalance (refusals are short, compliant answers are long), so
    # P(truncated | H11) vs P(truncated | H10) must be visible.
    truncation_by_variant: dict[str, dict] = {}
    for variant in ALL_VARIANT_NAMES:
        records = [r for r in outputs if r["variant"] == variant]
        truncated = [r for r in records if r["hit_max_new_tokens"] is True]
        truncation_by_variant[variant] = {
            "n": len(records),
            "n_truncated": len(truncated),
            "truncation_rate": (len(truncated) / len(records)
                                if records else None),
        }
    n_truncated = sum(v["n_truncated"]
                      for v in truncation_by_variant.values())

    report = {
        "iteration": "8",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": input_dir.name,
        "n_families": len(families),
        "expected_attempts": expected,
        "n_attempted": attempted,
        "n_succeeded": len(outputs),
        "n_failed": len(failures),
        "missing_variants": missing,
        "provenance": {
            "backend": config.backend,
            "model": _backend_model_name(backend, config),
            "requested_model_revision": config.model_revision,
            "resolved_model_revision": backend.model_revision(),
            "revision_pinned": config.model_revision is not None,
            # Legacy alias for backward compatibility
            "model_revision": backend.model_revision(),
            "processor_revision": backend.processor_revision(),
            "prompt_template_revision": config.prompt_template_revision,
            "system_prompt_sha256": sha256_text(config.system_prompt),
            "generation_config": config.generation_settings(),
            "enable_thinking": config.enable_thinking,
            "torch_dtype": config.torch_dtype,
            "validated_families_sha256": _file_sha256(source_path),
            "transformers_version": backend.transformers_version(),
            "torch_version": backend.torch_version(),
            "cuda_version": backend.cuda_version(),
            "git_commit": get_git_commit(),
            "git_dirty": is_git_dirty(),
            "config_sha256": config.fingerprint(),
            "resolved_sha256": resolved_fingerprint(
                backend, config, input_dir),
        },
        "token_stats": {
            "total_input_tokens": sum(input_tokens),
            "mean_input_tokens": (sum(input_tokens) / len(input_tokens)
                                  if input_tokens else None),
            "total_image_tokens": sum(image_tokens),
            "mean_image_tokens": (sum(image_tokens) / len(image_tokens)
                                  if image_tokens else None),
            "total_output_tokens": sum(output_tokens),
            "mean_output_tokens": (sum(output_tokens) / len(output_tokens)
                                   if output_tokens else None),
        },
        "truncation": {
            "n_truncated": n_truncated,
            "truncation_rate": (n_truncated / len(outputs)
                                if outputs else None),
            "by_variant": truncation_by_variant,
        },
    }
    if model_spec is not None:
        # Iteration 11 report dimension. The legacy single-model report
        # above is emitted unchanged when model_spec is None.
        report["iteration"] = "11"
        report["model_key"] = model_spec.model_key
        report["adapter"] = model_spec.adapter
        report["model_spec"] = model_spec.to_dict()
        report["provenance"].update(run_prov)
        report["resume"] = {
            "enabled": bool(resume), "n_pairs_resumed": len(done),
        }
    with (run_dir / REPLAY_REPORT_FILE).open(
            "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Replay %s: %d/%d succeeded (%d failed) -> %s",
             run_id, len(outputs), attempted, len(failures), run_dir)
    return report
