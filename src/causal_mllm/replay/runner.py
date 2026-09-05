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
import os
from pathlib import Path
from typing import Any, Sequence

from causal_mllm.construction.readiness import ALL_VARIANT_NAMES
from causal_mllm.data.io import read_jsonl
from causal_mllm.data.logging import get_logger
from causal_mllm.data.schemas import CausalFamily
from causal_mllm.replay.backend import HFLocalBackend, ReplayBackend
from causal_mllm.replay.config import ReplayConfig

# OWN_OUTPUT_PREFIXES is defined by the confirmatory gate so that the
# fingerprint and the gate scope "this stage's own outputs" identically;
# importing it keeps one definition of the output layout rather than two
# constants that must be kept in agreement by hand.
from causal_mllm.replay.confirmatory import OWN_OUTPUT_PREFIXES
from causal_mllm.replay.errors import ReplayError, ReplayMediaError, classify_error
from causal_mllm.replay.registry import (
    ResolvedModel,
    dependency_lock_sha256,
    verify_active_dependency_lock,
)

# The canonical digest over a family selection. Imported from the selection
# contract rather than re-implemented, so the subset bound into a run
# fingerprint is the same value the confirmatory gate compares against.
from causal_mllm.replay.selection import selected_families_sha256
from causal_mllm.seeds import (
    code_tree_status,
    get_git_commit,
    is_git_dirty,
    sha256_text,
)
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
    lock_path: str | Path | None = None,
    family_ids: Sequence[str] | None = None,
    own_output_prefixes: Sequence[str] | None = None,
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
        # Which families were replayed, when the run is a subset of the
        # panel. ``validated_families_sha256`` above binds the PANEL FILE
        # and is identical for a 12-family eligibility run and the 100-
        # family confirmatory run that reads the same frozen panel, so on
        # its own it would let one resume into the other: pointing
        # --resume at an eligibility run_dir with the full panel would find
        # 72 stored pairs, treat them as 72 of the 600 already done, and
        # quietly splice eligibility evidence into confirmatory evidence.
        # Binding the subset makes the stored records' fingerprint differ,
        # so validate_journal refuses them. None for a whole-panel run.
        "family_subset_sha256": (selected_families_sha256(family_ids)
                                 if family_ids else None),
        "transformers_version": backend.transformers_version(),
        "torch_version": backend.torch_version(),
        "cuda_version": backend.cuda_version(),
        "code_commit": get_git_commit(),
        # ``code_commit`` alone does not identify the code that ran: a
        # dirty tree executes whatever is on disk, not what the commit
        # contains, and two runs from the same commit with different
        # uncommitted edits would otherwise share a fingerprint (and so
        # would be allowed to resume into each other). None (not a git
        # repo) is bound as-is: unknown provenance must not collide with a
        # known-clean run.
        "git_dirty": is_git_dirty(),
        # ``git_dirty`` counts only TRACKED modifications, so it stayed
        # False while an untracked module, an untracked sitecustomize.py or
        # a top-level file shadowing an installed package changed what
        # executed — and the fingerprint would then have certified a run its
        # own ``code_commit`` could not reconstruct. This binds the
        # reconstruction-relevant answer: modified OR untracked, minus cache
        # paths and this stage's own output tree, because a run's own
        # outputs are its product rather than its code.
        "code_tree_dirty": code_tree_status(
            exclude_prefixes=(own_output_prefixes
                              if own_output_prefixes is not None
                              else OWN_OUTPUT_PREFIXES))["dirty"],
        "hardware": fingerprint_hardware(hardware),
        # The frozen protocol requires the pip-freeze dependency lock hash
        # to be bound into every resolved run fingerprint.
        "dependency_lock_sha256": dependency_lock_sha256(lock_path),
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
            "adapter_diagnostics": None,
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
            record["adapter_diagnostics"] = result.get(
                "adapter_diagnostics")
            record["truncated"] = result.get("finish_reason") == "length"
        outputs.append(record)
    return outputs, failures


#: Fields every stored journal record must carry before it may be trusted
#: on resume. A record missing one of these cannot be attributed to a
#: (family, variant) cell, so accepting it would let a truncated or
#: foreign file silently satisfy coverage.
REQUIRED_JOURNAL_FIELDS = ("run_id", "family_id", "variant")


def append_journal(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL journal, flushed and fsync'd to disk.

    Crash-safety is the whole point. The frozen protocol requires a
    600-output run per model to be resumable, but a runner that persists
    only after the final family would lose every completed family when
    the process is killed (preemption, OOM, node failure) — the exact
    situation ``--resume`` exists for. Each family is therefore durable
    before the next one is started.

    Line formatting is byte-identical to :func:`write_jsonl`, so an
    incrementally journaled file is indistinguishable from a one-shot
    write of the same records in the same order.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        # flush() only hands the bytes to the OS; fsync() is what survives
        # a machine-level interruption.
        os.fsync(f.fileno())


def validate_journal(
    records: list[dict],
    *,
    path: Path,
    expected_fingerprint: str | None,
    expected_model_key: str | None,
    allowed_variants: tuple[str, ...] = tuple(ALL_VARIANT_NAMES),
    allow_duplicate_pairs: bool = False,
) -> set[tuple[str, str]]:
    """Validate every stored record; return the (family, variant) pairs.

    Fail-closed on provenance: for an Iteration 11 run
    (``expected_fingerprint``/``expected_model_key`` not None) a record
    whose ``resolved_run_fingerprint`` or ``model_key`` is MISSING is
    rejected, not treated as compatible. Treating an absent value as a
    wildcard would let records produced by different code, a different
    dependency environment or a different target model be resumed into
    one run — the resulting file would look complete while mixing
    provenance that the estimands assume to be uniform.

    Args:
        allow_duplicate_pairs: the OUTPUTS journal must hold each pair
            exactly once (a duplicate means two responses for one cell and
            the analysis could not pick between them). The FAILURES journal
            is append-only history, so the same pair may legitimately
            appear once per interrupted attempt.
    """
    pairs: set[tuple[str, str]] = set()
    for index, rec in enumerate(records):
        where = f"{path} line {index + 1}"
        if not isinstance(rec, dict):
            raise ReplayError(f"{where}: stored record is not an object")
        missing = [f for f in REQUIRED_JOURNAL_FIELDS if rec.get(f) is None]
        if missing:
            raise ReplayError(
                f"{where}: stored record is missing required field(s) "
                f"{missing}; refusing to resume onto a record that cannot "
                f"be attributed to a (family, variant) cell")
        variant = rec["variant"]
        if variant not in allowed_variants:
            raise ReplayError(
                f"{where}: stored variant {variant!r} is not one of "
                f"{list(allowed_variants)}")
        if expected_fingerprint is not None:
            prior_fp = rec.get("resolved_run_fingerprint")
            if prior_fp is None:
                raise ReplayError(
                    f"{where}: stored record has no resolved_run_fingerprint; "
                    f"refusing to resume — its provenance cannot be "
                    f"confirmed against the current run "
                    f"({expected_fingerprint[:16]}…)")
            if prior_fp != expected_fingerprint:
                raise ReplayError(
                    f"{where}: resume mismatch — stored "
                    f"resolved_run_fingerprint {prior_fp!r} != current "
                    f"{expected_fingerprint!r}; refusing to append to a run "
                    f"with different provenance")
        if expected_model_key is not None:
            prior_key = rec.get("model_key")
            if prior_key is None:
                raise ReplayError(
                    f"{where}: stored record has no model_key; refusing to "
                    f"resume — it cannot be confirmed to belong to "
                    f"{expected_model_key!r}")
            if prior_key != expected_model_key:
                raise ReplayError(
                    f"{where}: resume mismatch — stored model_key "
                    f"{prior_key!r} != current {expected_model_key!r}")
        pair = (rec["family_id"], variant)
        if pair in pairs and not allow_duplicate_pairs:
            raise ReplayError(
                f"{where}: duplicate stored record for {pair[0]}:{pair[1]} — "
                f"two responses exist for one (family, variant) cell and "
                f"the analysis could not choose between them")
        pairs.add(pair)
    return pairs


def _select_families(families: list[CausalFamily],
                     family_ids: Sequence[str], source_path: Path,
                     model_spec: "ResolvedModel | None") -> list[CausalFamily]:
    """The named subset of ``families``, in PANEL order.

    Fails closed on an id the panel does not contain and on a duplicate:
    silently dropping one would produce a run that looks complete while
    having replayed fewer families than its selection named, which is
    exactly the discrepancy the 11.5 gates exist to catch.
    """
    if model_spec is None:
        raise ReplayError(
            "family_ids is an Iteration 11 facility and requires a resolved "
            "model_spec; the frozen legacy single-model report schema "
            "records no subset, so a subset run there could not say which "
            "families it replayed")
    names = [str(f) for f in family_ids]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ReplayError(
            f"family_ids contains duplicate(s) {duplicates}; a family is "
            f"replayed once per run, and a duplicate would make the attempt "
            f"count disagree with the selection it came from")
    if not names:
        raise ReplayError(
            "family_ids is empty; a subset run must name the families it "
            "replays. Omit the argument to replay the whole panel.")
    absent = sorted(set(names) - {f.family_id for f in families})
    if absent:
        shown = absent[:8] + (["..."] if len(absent) > 8 else [])
        raise ReplayError(
            f"family_ids {shown} ({len(absent)} of {len(names)}) are not in "
            f"{source_path}; a subset run may only select families the panel "
            f"actually contains")
    selected = set(names)
    # Panel order, not the caller's, so the evidence does not depend on the
    # order a selection happened to be listed in.
    subset = [family for family in families if family.family_id in selected]
    log.info("Family subset: replaying %d of %d families from %s",
             len(subset), len(families), source_path.name)
    return subset


def run_replay_stage(
    input_dir: str | Path,
    output_root: str | Path,
    config: ReplayConfig | None = None,
    backend: ReplayBackend | None = None,
    max_families: int | None = None,
    family_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    model_spec: "ResolvedModel | None" = None,
    resume: bool = False,
    lock_path: str | Path | None = None,
    gate_evidence: dict | None = None,
    own_output_prefixes: Sequence[str] | None = None,
) -> dict:
    """Replay validated families; persist outputs/failures/report.

    Outputs and failures are journaled append-only with flush+fsync after
    EVERY family, so an interrupted run keeps the families it finished
    and ``--resume`` continues from them instead of regenerating.

    Args:
        input_dir: Dataset dir containing validated_families.jsonl.
        output_root: Root for replay runs (separate from datasets).
        config: Frozen replay settings (defaults: Qwen3.5-9B, greedy).
        backend: Injectable backend; defaults per config.backend.
        max_families: Optional limit (smoke runs).
        family_ids: Replay ONLY these families, by id. The Iteration 11.5
            eligibility facility: ``input_dir`` stays the full frozen panel
            — so the fingerprint still binds the frozen panel digest — and
            the subset is bound into the fingerprint separately. Every id
            must exist in the panel. Order is taken from the panel, not
            from this argument, so the evidence does not depend on the
            order a selection happened to be listed in. Iteration 11 only
            (``model_spec`` required): the frozen legacy single-model
            report schema has no place to record a subset, and a run whose
            own report cannot say which families it replayed is not
            auditable.
        run_id: Override the generated run id.
        overwrite: If False (default), fail when the run directory
            already contains evidence files.  This prevents accidental
            overwriting of retained evidence.
        model_spec: Resolved Iteration 11 target (None = frozen legacy
            single-model path, whose records are unchanged).
        resume: Continue an interrupted run in the same run_dir, skipping
            (family, variant) pairs already recorded under the SAME
            resolved run fingerprint and model_key.
        lock_path: ``resolved_models.lock.yaml`` supplying the dependency
            lock bound into the resolved run fingerprint. Must be the same
            path used for revision resolution, otherwise the fingerprint
            silently binds the DEFAULT lock instead of the selected one.
        gate_evidence: The protocol gate result (from
            ``enforce_confirmatory_protocol`` or
            ``enforce_eligibility_protocol``) to persist with the run, so a
            PASS is auditable from the evidence itself rather than only
            from the launching terminal's stdout. Stored under
            ``confirmatory_gate`` or ``eligibility_gate`` according to the
            gate that produced it.
        own_output_prefixes: Repo-root-relative prefixes holding THIS
            stage's own outputs, excluded from the clean-tree determination
            that ``code_tree_dirty`` records and the fingerprint binds.
            Defaults to the confirmatory stage's tree; an 11.5 eligibility
            run passes its own, which additionally covers the report it
            writes. Without this a resumed run would see its own prior
            outputs as untracked files and bind a different fingerprint
            than its first attempt did, refusing the resume.

    Raises:
        ReplayError: On missing validated_families.jsonl, missing
            (family, variant) coverage, an existing run directory when
            overwrite is False, or a stored journal record whose
            provenance (fingerprint / model_key / required fields) cannot
            be confirmed identical to the current run.
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
    if family_ids is not None:
        if max_families is not None:
            raise ReplayError(
                f"family_ids cannot be combined with "
                f"max_families={max_families}: a named subset is a "
                f"SELECTION and --max-families is a smoke PREFIX, so "
                f"applying both would replay a prefix of the selection "
                f"while the fingerprint bound all of it")
        families = _select_families(families, family_ids, source_path,
                                    model_spec)
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
        try:
            dependency_check = verify_active_dependency_lock(
                lock_path, strict=False)
        except ReplayError as exc:
            # Recorded, not fatal: enforcement is the confirmatory gate's
            # job, and a run must not be lost because `pip freeze` could
            # not be executed. The reason stays in the evidence.
            dependency_check = {"verified": False, "reason": str(exc)}
        # Recorded alongside ``git_dirty``, which counts only tracked
        # modifications: a report claiming a clean tree while an untracked
        # module or sitecustomize.py changed what executed is the same
        # defect the confirmatory gate now refuses.
        prefixes = (list(own_output_prefixes)
                    if own_output_prefixes is not None
                    else list(OWN_OUTPUT_PREFIXES))
        code_tree = code_tree_status(exclude_prefixes=prefixes)
        run_prov = {
            "model_key": model_spec.model_key,
            "adapter": model_spec.adapter,
            "code_commit": get_git_commit(),
            "git_dirty": is_git_dirty(),
            "code_tree_dirty": code_tree["dirty"],
            "code_dirty_paths": code_tree["dirty_paths"],
            "code_untracked_paths": code_tree["untracked_paths"],
            # Recorded so the exclusion this run's clean-tree answer relied
            # on is auditable rather than implied by which stage launched it.
            "code_own_output_prefixes": prefixes,
            "dataset_manifest_hash": _file_sha256(source_path),
            # Recorded so the evidence itself says which families this run
            # replayed. ``dataset_manifest_hash`` above is the WHOLE panel,
            # because a subset run still reads — and must still be bound to
            # — the frozen panel file.
            "family_subset": ({
                "family_ids": [f.family_id for f in families],
                "n_families": len(families),
                "family_ids_sha256": selected_families_sha256(
                    f.family_id for f in families),
            } if family_ids is not None else None),
            "resolved_run_fingerprint": iteration11_run_fingerprint(
                backend, config, input_dir, model_spec, hardware,
                lock_path=lock_path, family_ids=family_ids,
                own_output_prefixes=own_output_prefixes),
            # Recorded, not merely assumed: the lock hash bound above is
            # read from the lock FILE, so on its own it does not prove the
            # environment actually running inference is the one certified
            # at preflight. strict=False here because enforcement belongs
            # to the confirmatory gate (a technical preflight must still be
            # able to run and REPORT a drift); the comparison is stored so
            # any drift is visible in the evidence either way.
            "dependency_lock_check": dependency_check,
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
    outputs_path = run_dir / REPLAY_OUTPUTS_FILE
    failures_path = run_dir / REPLAY_FAILURES_FILE
    cur_fp = run_prov.get("resolved_run_fingerprint")
    cur_key = model_spec.model_key if model_spec else None
    existing_outputs: list[dict] = []
    prior_failures: list[dict] = []
    done: set[tuple[str, str]] = set()
    if resume:
        if outputs_path.exists():
            existing_outputs = read_jsonl(outputs_path)
        if failures_path.exists():
            prior_failures = read_jsonl(failures_path)
        done = validate_journal(
            existing_outputs, path=outputs_path,
            expected_fingerprint=cur_fp, expected_model_key=cur_key)
        # The failure journal is append-only HISTORY retained across
        # interruptions. Its pairs are deliberately NOT added to ``done``,
        # so they are retried; a pair that later succeeds is excluded from
        # the failure count below instead of being double-counted.
        validate_journal(
            prior_failures, path=failures_path,
            expected_fingerprint=cur_fp, expected_model_key=cur_key,
            allow_duplicate_pairs=True)
        log.info("Resume %s: %d (family, variant) pairs already done, "
                 "%d prior failure attempt(s) retained",
                 run_id, len(done), len(prior_failures))
    elif run_dir.exists() and not overwrite:
        # Evidence-protection guard: refuse to overwrite existing evidence
        # unless the caller explicitly opts in. An EMPTY journal is not
        # evidence — it is a run interrupted before its first family
        # completed — so restarting over one must not demand --overwrite.
        def _holds_evidence(name: str) -> bool:
            p = run_dir / name
            if not p.exists():
                return False
            if name == REPLAY_REPORT_FILE:
                return True  # only written once a run completed
            return p.stat().st_size > 0

        existing = [f for f in (REPLAY_OUTPUTS_FILE, REPLAY_FAILURES_FILE,
                                REPLAY_REPORT_FILE) if _holds_evidence(f)]
        if existing:
            raise ReplayError(
                f"run directory {run_dir} already contains evidence "
                f"{existing}; pass overwrite=True or use a new run_id "
                f"to avoid overwriting retained evidence")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        # A fresh run starts from empty journals even under overwrite=True:
        # append-only journaling must never splice new records onto
        # retained evidence from an earlier attempt.
        for p in (outputs_path, failures_path):
            with p.open("w", encoding="utf-8"):
                pass

    outputs: list[dict] = list(existing_outputs)
    failures: list[dict] = list(prior_failures)
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
        # Durable BEFORE the next family starts: if the process is killed
        # mid-run, everything already journaled survives and --resume
        # continues from it instead of regenerating it.
        append_journal(outputs_path, family_outputs)
        append_journal(failures_path, family_failures)
        outputs.extend(family_outputs)
        failures.extend(family_failures)
        log.info("Replay %s: %d outputs, %d failures",
                 family.family_id, len(family_outputs),
                 len(family_failures))

    expected = len(families) * len(ALL_VARIANT_NAMES)
    succeeded = {(r["family_id"], r["variant"]) for r in outputs}
    # A pair whose LATEST outcome failed counts once as failed; earlier
    # attempts kept in the journal are history, not additional failures.
    failed = {(r["family_id"], r["variant"]) for r in failures} - succeeded
    attempted = len(succeeded) + len(failed)
    covered = succeeded | failed
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

    # The journals are already complete on disk, written incrementally
    # above; rewriting them here would defeat the crash-safety guarantee.

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
        # Cells whose LATEST attempt failed. The failure journal itself is
        # append-only history and may hold more lines than this across
        # interruptions; n_failure_attempts reports that total so a
        # retried-and-recovered cell is visible rather than silently erased.
        "n_failed": len(failed),
        "n_failure_attempts": len(failures),
        "failed_cells": sorted(f"{fid}:{variant}" for fid, variant in failed),
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
            "n_prior_failure_attempts_retained": len(prior_failures),
        }
    if gate_evidence is not None:
        # Persisted with the run so the PASS is auditable from the evidence
        # itself rather than only from the launching terminal's stdout. The
        # key names the gate that produced it, so a 12-family eligibility
        # run can never carry its evidence under the confirmatory key.
        report["confirmatory_gate"
               if gate_evidence.get("gate") == "iteration11_confirmatory"
               else "eligibility_gate"] = gate_evidence
    with (run_dir / REPLAY_REPORT_FILE).open(
            "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Replay %s: %d/%d succeeded (%d failed, %d failure attempt(s) "
             "journaled) -> %s",
             run_id, len(outputs), attempted, len(failed), len(failures),
             run_dir)
    return report
