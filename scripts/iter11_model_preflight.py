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
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.data.io import read_jsonl  # noqa: E402
from causal_mllm.data.schemas import CausalFamily  # noqa: E402
from causal_mllm.replay.checkpoint_size import checkpoint_size_metadata  # noqa: E402
from causal_mllm.replay.config import ReplayConfig  # noqa: E402
from causal_mllm.replay.registry import is_immutable_revision, resolve_model  # noqa: E402
from causal_mllm.replay.runner import build_chat_messages, verify_family_media  # noqa: E402
from causal_mllm.seeds import (  # noqa: E402
    code_tree_status,
    get_git_commit,
    sha256_text,
)
from causal_mllm.validation.relations import _file_sha256  # noqa: E402

FROZEN_PANEL = REPO_ROOT / "outputs" / "scale_c" / "families_panel"
FROZEN_PROTOCOL = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"
PREFLIGHT_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "preflight"
FROZEN_CAP = 1536
VISION_VARIANT = "cross_modal"
TEXT_VARIANT = "text_only"

#: Repo-root-relative prefixes THIS stage writes. Regenerating the first
#: target's artifact (and the shared lock) must not make the tree count as
#: dirty for the second target: nothing about the code changed, and
#: ``code_commit`` still reconstructs it. Excluded paths are recorded in the
#: artifact rather than silently dropped. ``eligibility/`` is included
#: because the 11.5 eligibility report is this stage family's output too.
OWN_OUTPUT_PREFIXES = (
    "outputs/iteration_11/preflight/",
    "outputs/iteration_11/eligibility/",
)


def load_frozen_protocol() -> dict:
    """The frozen protocol this preflight must agree with.

    Read-only: the preflight checks itself against the frozen values, it
    never restates them from memory or from a config default.
    """
    if not FROZEN_PROTOCOL.exists():
        raise SystemExit(f"frozen protocol not found: {FROZEN_PROTOCOL}")
    return json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))


def check_frozen_inputs(panel: Path, protocol: dict,
                        config: ReplayConfig) -> tuple[dict, list[str]]:
    """Compare the preflight's own inputs against the frozen protocol.

    Returns the recorded values plus any violations. The panel is hashed
    over RAW BYTES: the frozen protocol and the replay runner both use a
    raw-byte digest, so a whitespace-normalized digest of the same file is
    a DIFFERENT number that matches neither, and an artifact carrying it
    silently asserts a panel nobody can verify.
    """
    frozen_inputs = protocol["frozen_inputs"]
    frozen_panel_sha = frozen_inputs["panel_validated_families_sha256"]
    panel_sha = _file_sha256(panel) if panel.exists() else None
    prompt_sha = sha256_text(config.system_prompt)
    frozen_prompt_sha = frozen_inputs["system_prompt_sha256"]
    problems: list[str] = []
    if panel_sha is None:
        problems.append(f"frozen panel not found: {panel}")
    elif panel_sha != frozen_panel_sha:
        problems.append(
            f"panel is not the frozen Iteration 11 panel: raw-byte SHA-256 "
            f"{panel_sha} != frozen {frozen_panel_sha} ({panel})")
    if prompt_sha != frozen_prompt_sha:
        problems.append(
            f"system prompt SHA-256 {prompt_sha} != frozen "
            f"{frozen_prompt_sha}")
    if config.max_new_tokens != protocol["uniform_cap_rule"]["initial_cap"]:
        problems.append(
            f"max_new_tokens={config.max_new_tokens} != the frozen uniform "
            f"cap {protocol['uniform_cap_rule']['initial_cap']}")
    values = {
        "input_dir": str(panel.parent.resolve()),
        "validated_families_sha256": panel_sha,
        "hash_method": "sha256(raw file bytes)",
        "frozen_panel_sha256": frozen_panel_sha,
        "matches_frozen_protocol": panel_sha == frozen_panel_sha,
        "frozen_system_prompt_sha256": frozen_prompt_sha,
        "system_prompt_matches_frozen_protocol": prompt_sha == frozen_prompt_sha,
    }
    return values, problems


def git_provenance(code_commit: str | None, tree: dict,
                   allow_dirty: bool) -> dict:
    """Decide what a dirty/unknown tree means for this run.

    ``tree`` is a :func:`causal_mllm.seeds.code_tree_status` result, i.e.
    dirtiness measured over CODE paths with this stage's own outputs
    excluded (and reported).

    Returns ``abort`` (stop before doing any GPU work), ``abort_message``,
    ``problems`` (recorded in the artifact, which makes status PASS
    unreachable because the status is derived from problems), and the two
    path lists so the exclusion is auditable.

    ``code_tree_status`` reports ``dirty=None`` when git is unavailable.
    That is treated like a dirty tree: provenance that cannot be verified
    cannot certify evidence.
    """
    dirty = tree.get("dirty")
    dirty_paths = list(tree.get("dirty_paths") or [])
    untracked_paths = list(tree.get("untracked_paths") or [])
    own_outputs = list(tree.get("excluded_own_outputs") or [])
    cache_paths = list(tree.get("excluded_cache_paths") or [])
    problems: list[str] = []
    abort = False
    abort_message = None
    if dirty is not False:
        if dirty is None:
            reason = "git status unavailable (not a repository?)"
        elif untracked_paths and len(untracked_paths) == len(dirty_paths):
            reason = f"UNTRACKED files at {untracked_paths}"
        else:
            reason = f"uncommitted changes to {dirty_paths}"
        problems.append(
            f"working tree was not clean at code_commit {code_commit} "
            f"({reason}); this artifact cannot certify the code that "
            f"produced it and is diagnostic only")
        if not allow_dirty:
            abort = True
            abort_message = (
                f"working tree is not clean ({reason}) at code_commit "
                f"{code_commit}. The code that would execute is not the "
                f"code that commit contains — an untracked module, "
                f"sitecustomize.py or shadowing top-level file changes "
                f"execution without being recorded anywhere — so the "
                f"artifact could not be reconstructed from its own recorded "
                f"provenance. Commit or remove them first; --allow-dirty "
                f"runs diagnostics but can never produce status PASS.")
    return {"abort": abort, "abort_message": abort_message,
            "problems": problems, "dirty_paths": dirty_paths,
            "untracked_paths": untracked_paths,
            "excluded_own_outputs": own_outputs,
            "excluded_cache_paths": cache_paths}


def check_environment(protocol: dict) -> tuple[dict, list[str]]:
    """Certify the environment this preflight is about to run in.

    Three separate questions:

    1. Does the environment hold a third-party editable install? Fatal.
       ``pip freeze`` identifies an editable dependency by the sibling
       repository's COMMITTED HEAD and is blind to that repository's
       uncommitted working-tree changes, so no hash of freeze output can
       prove which dependency source would execute. A technical preflight
       mints evidence that 11.5/11.6 rely on, so it must refuse here rather
       than let the problem surface at analysis time.
    2. Do the runtime versions match the frozen ``reference_versions``?
       Fatal on mismatch — these are what determine model behaviour.
    3. Is the conda environment NAME the frozen ``reference_env``? Recorded
       as an explicit deviation, NOT fatal: the name labels where the frozen
       versions were observed, and a dedicated clone carrying byte-identical
       versions preserves every scientific property while removing the
       editable install that question 1 forbids. The frozen protocol file is
       never edited to accommodate this.

    Returns the recorded environment identity plus any violations.
    """
    from causal_mllm.replay.registry import dependency_lock_snapshot
    snapshot = dependency_lock_snapshot()
    offenders = dict(snapshot.get("editable_vcs_revisions") or {})
    problems: list[str] = []
    if offenders:
        problems.append(
            f"environment holds third-party editable VCS install(s) "
            f"{sorted(offenders)} at revisions {offenders}; an editable "
            f"dependency's source can change without its recorded revision "
            f"moving, so this environment cannot be certified reproducible. "
            f"Use a dedicated Iteration 11 environment with no third-party "
            f"editable installs.")

    frozen_lock = protocol.get("dependency_lock") or {}
    frozen_versions = dict(frozen_lock.get("reference_versions") or {})
    observed: dict = {}
    try:
        import torch
        import transformers
        observed = {"transformers": transformers.__version__,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda}
    except ImportError as exc:  # pragma: no cover - env without inference
        problems.append(f"cannot read runtime versions: {exc}")
    mismatched = {
        key: {"frozen": frozen_versions.get(key), "observed": observed.get(key)}
        for key in sorted(frozen_versions)
        if observed.get(key) != frozen_versions.get(key)
    }
    if mismatched:
        detail = "; ".join(
            f"{k}: frozen={v['frozen']!r} observed={v['observed']!r}"
            for k, v in mismatched.items())
        problems.append(
            f"runtime versions do not match the frozen reference_versions — "
            f"{detail}")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    frozen_env = frozen_lock.get("reference_env")
    values = {
        "python_version": snapshot.get("python_version"),
        "executable": snapshot.get("executable"),
        "conda_env": conda_env,
        "n_packages": snapshot.get("n_packages"),
        "pip_freeze_sha256": snapshot.get("pip_freeze_sha256"),
        "pyproject_sha256": snapshot.get("pyproject_sha256"),
        "excluded_self_distributions":
            snapshot.get("excluded_self_distributions"),
        "third_party_editable_vcs": offenders,
        "observed_versions": observed,
        "frozen_reference_versions": frozen_versions,
        "frozen_reference_env": frozen_env,
        "reference_env_matches_frozen": conda_env == frozen_env,
    }
    if conda_env != frozen_env:
        values["reference_env_deviation"] = {
            "claim": f"the frozen protocol names reference_env={frozen_env!r}",
            "observation": f"this preflight ran in conda env {conda_env!r}",
            "rationale": (
                "a dedicated clone of that environment with the third-party "
                "editable install removed; every frozen reference_version "
                "matches exactly, so only the environment NAME differs. The "
                "clone is required because the shared environment carries an "
                "editable sibling install whose source can change without its "
                "recorded revision moving, which is not certifiable. The "
                "frozen protocol file is immutable and was not edited."),
            "frozen_protocol_modified": False,
        }
    return values, problems


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
    "phi4_multimodal": [
        "This is a transformers-4.x remote-code checkpoint loaded on "
        "transformers 5.14.1 under the frozen protocol's "
        "`shim_in_shared_env` decision. Every shim actually applied is "
        "listed verbatim in runtime_metadata.phi4_shims, and the direct "
        "checkpoint load is verified fail-closed in "
        "runtime_metadata.phi4_load_report (lm_head tied to the input "
        "embedding, zero parameters left on the meta device, zero "
        "unmatched tensors). A silently untied output projection or one "
        "randomly initialised parameter would emit fluent garbage while "
        "every superficial check still passed, so both are asserted "
        "rather than assumed.",
        "config.json hard-codes flash_attention_2, which is unavailable "
        "for this model here; sdpa is forced on the config and every "
        "nested sub-config, which selects the vendor's own "
        "Phi4MMSdpaAttention. The vendored SigLIP tower has a separate "
        "_flash_attention_forward hook that never consults the config, so "
        "it is redirected to sdpa as well.",
        "The checkpoint ships bundled PEFT vision/speech LoRA adapters, "
        "and Phi4MMForCausalLM.forward activates them from `input_mode`, "
        "which the processor derives from the supplied modalities. Every "
        "generation records input_mode and the LoRA adapters that were "
        "active, so it stays auditable that the vision arm ran the vision "
        "adapter and the text-only arm ran none.",
        "generation_config.json declares eos_token_id [200020, 199999] "
        "while config.json declares only 199999. 200020 is `<|end|>`, the "
        "token the chat template uses to close every message, so the "
        "shipped generation config is loaded explicitly; deriving it from "
        "config.json would drop the model's real stop token and drive "
        "every response to the 1536-token cap.",
        "`num_logits_to_keep=1` is passed explicitly. transformers only "
        "sets `logits_to_keep` itself when forward advertises that exact "
        "name and this model names it `num_logits_to_keep`; greedy "
        "decoding consumes only the last position's logits either way, so "
        "this matches the other families instead of deviating from them.",
        "The frozen protocol records `audio_tower_initialized: false`. "
        "That is inaccurate for this checkpoint: "
        "Phi4MMImageAudioEmbedding builds the audio tower "
        "unconditionally, the checkpoint ships its weights, and the VISION "
        "path itself routes through audio_embed.audio_projection.vision. "
        "The tower is therefore fully initialised; what is false is that "
        "any audio INPUT is supplied. This is reported per record in "
        "runtime_metadata.phi4_audio_tower as an explicit deviation "
        "rather than being silently contradicted.",
        "The chat template concatenates `content` as a string, so the "
        "multimodal part list is flattened: each image becomes one "
        "`<|image_k|>` placeholder in message order followed by the turn "
        "text (the vendor's own documented form). The processor "
        "regex-normalises that to `<|endoftext10|>` and expands it to the "
        "image's token count, asserting exactly one placeholder per "
        "supplied image; both counts are recorded per generation.",
    ],
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


def _check_input_mode_matches_variant(smoke) -> list[str]:
    """The modality the model selected must be the one the variant implies.

    Families whose processor reports an ``input_mode`` (Phi-4) get this
    checked; families that report none are skipped.  A cross-modal prompt
    that silently degraded to language-only would otherwise look like a
    legitimate null result.
    """
    from causal_mllm.replay.adapters.phi4_multimodal import INPUT_MODE_LANGUAGE, INPUT_MODE_VISION

    expected = {VISION_VARIANT: INPUT_MODE_VISION,
                TEXT_VARIANT: INPUT_MODE_LANGUAGE}
    problems = []
    for entry in smoke:
        want = expected.get(entry["variant"])
        if want is None:
            continue
        for attempt in entry["attempts"]:
            diag = attempt.get("adapter_diagnostics") or {}
            got = diag.get("input_mode")
            if got is not None and got != want:
                problems.append(
                    f"{entry['variant']}: processor selected input_mode "
                    f"{got} but the variant requires {want}")
            lora = diag.get("active_lora") or {}
            if entry["variant"] == VISION_VARIANT and lora.get("available") \
                    and "vision" not in (lora.get("active_adapters") or []):
                problems.append(
                    f"{entry['variant']}: the bundled vision LoRA was not "
                    f"active (adapters={lora.get('active_adapters')}, "
                    f"disabled={lora.get('adapters_disabled')}) — the "
                    f"vision arm would run unadapted weights")
            # The vendor disables every LoRA for LANGUAGE input, so the
            # text-only arm runs base weights. Asserting it documents that
            # the two arms differ by the image AND by the adapter the
            # vendor intends for it, rather than leaving that implicit.
            if entry["variant"] == TEXT_VARIANT and lora.get("available") \
                    and lora.get("adapters_disabled") is not True:
                problems.append(
                    f"{entry['variant']}: LoRA adapters were not disabled "
                    f"for language-only input "
                    f"(adapters={lora.get('active_adapters')}, "
                    f"disabled={lora.get('adapters_disabled')})")
    return problems


def _check_load_report(runtime_metadata, size_meta) -> list[str]:
    """Verify a directly-loaded checkpoint actually received its weights.

    Only families that report a ``phi4_load_report`` (i.e. those loaded
    outside ``from_pretrained``, where transformers' own safety nets do
    not apply) are checked.
    """
    report = (runtime_metadata or {}).get("phi4_load_report") or {}
    if not report:
        return []
    problems = []
    if report.get("lm_head_tied_to_embed_tokens") is not True:
        problems.append(
            "direct load: lm_head.weight is not tied to the input "
            "embedding — the output projection would be random")
    if report.get("parameters_left_on_meta"):
        problems.append(
            f"direct load: {report['parameters_left_on_meta']} "
            f"parameter(s) were never materialised (still on meta)")
    if report.get("unexpected_keys"):
        problems.append(
            f"direct load: {len(report['unexpected_keys'])} checkpoint "
            f"tensor(s) matched no parameter: {report['unexpected_keys'][:5]}")
    if report.get("missing_keys") not in ([], ["lm_head.weight"]):
        problems.append(
            f"direct load: unexpected missing keys "
            f"{report['missing_keys'][:5]}")
    if report.get("attn_implementation") != "sdpa":
        problems.append(
            f"direct load: attention implementation is "
            f"{report.get('attn_implementation')!r}, expected 'sdpa'")
    histogram = report.get("checkpoint_dtype_histogram") or {}
    if set(histogram) - {"BF16"}:
        problems.append(
            f"direct load: checkpoint is not uniformly bf16: {histogram}")
    # Independent cross-check: the adapter's header-derived count must
    # agree with the declared-size machinery's header-derived count.
    adapter_count = report.get("checkpoint_parameter_count")
    declared_count = size_meta.get("checkpoint_parameter_count")
    if adapter_count is not None and declared_count is not None \
            and adapter_count != declared_count:
        problems.append(
            f"direct load counted {adapter_count} checkpoint parameters "
            f"but the declared-size pass counted {declared_count}")
    return problems


def _check_rope_headroom(smoke, runtime_metadata, max_new_tokens) -> list:
    """Generation must not cross the longrope switching point.

    Phi-4 uses longrope: once a position passes
    ``original_max_position_embeddings`` the vendor swaps rope factors AND
    discards the KV cache mid-generation.  That is a family-specific
    perturbation the Qwen and Ministral arms do not share, so a prompt
    plus cap that reached it would not be comparable.  Only checked for
    families that report the switch point.
    """
    report = (runtime_metadata or {}).get("phi4_load_report") or {}
    switch = report.get("rope_switch_position")
    if not switch or not max_new_tokens:
        return []
    problems = []
    for entry in smoke:
        for attempt in entry["attempts"]:
            prompt_tokens = attempt["input_token_count"]
            worst = prompt_tokens + max_new_tokens
            if worst > switch:
                problems.append(
                    f"{entry['variant']}: prompt ({prompt_tokens}) + cap "
                    f"({max_new_tokens}) = {worst} exceeds the longrope "
                    f"switch point ({switch}); generation would change "
                    f"rope factors and invalidate the KV cache mid-sequence")
    return problems


def _check_smoke(smoke, size_meta, runtime_metadata=None,
                 max_new_tokens=None) -> list[str]:
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
            # Family-reported serialization invariants. Each is checked
            # only when the adapter actually reports the key, so the
            # checker stays shared across families instead of branching
            # on adapter_name.
            if diag.get("placeholders_match_supplied_images") is False:
                problems.append(
                    f"{entry['variant']}: image placeholders in the "
                    f"rendered prompt "
                    f"({diag.get('image_placeholders_in_rendered_text')}) do "
                    f"not match the images supplied "
                    f"({diag.get('images_supplied')})")
            if diag.get("audio_special_token_present") is True:
                problems.append(
                    f"{entry['variant']}: an audio placeholder reached the "
                    f"prompt — the frozen protocol is vision-only")
            if diag.get("input_mode_is_vision_or_language") is False:
                problems.append(
                    f"{entry['variant']}: input_mode "
                    f"{diag.get('input_mode')!r} is neither VISION nor "
                    f"LANGUAGE — an out-of-protocol modality was selected")
            if diag.get("end_token_present") is False:
                problems.append(
                    f"{entry['variant']}: the chat template's <|end|> "
                    f"message terminator is absent from the rendered "
                    f"prompt — the template did not render as documented")
    problems.extend(_check_input_mode_matches_variant(smoke))
    problems.extend(_check_load_report(runtime_metadata, size_meta))
    problems.extend(_check_rope_headroom(smoke, runtime_metadata,
                                         max_new_tokens))
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
    parser.add_argument("--allow-dirty", action="store_true", default=False,
                        help="Diagnostics ONLY: proceed with uncommitted "
                             "changes. The run still records git_dirty and "
                             "still cannot reach status PASS, because "
                             "evidence produced from a dirty tree cannot be "
                             "reconstructed from its recorded code_commit.")
    args = parser.parse_args()

    protocol = load_frozen_protocol()

    # Environment first: unlike a dirty tree, an environment holding a
    # third-party editable install cannot produce certifiable evidence at
    # all, so there is nothing to gain by loading a checkpoint. No escape
    # hatch — the environment has to be fixed.
    environment, environment_problems = check_environment(protocol)
    if environment_problems:
        for problem in environment_problems:
            print(f"FAIL {args.model_key}: {problem}")
        return 2

    # Captured BEFORE anything is written, and measured over CODE paths:
    # this stage regenerates its own committed artifacts and the shared
    # lock, so an unscoped "is anything tracked modified?" check would let
    # the first target's output block every subsequent target even though
    # nothing about the code changed. Excluded paths are recorded below.
    code_commit = get_git_commit()
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    provenance = git_provenance(code_commit, tree, args.allow_dirty)
    if provenance["abort"]:
        print(f"FAIL {args.model_key}: {provenance['abort_message']}")
        return 2

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
    dataset_values, frozen_problems = check_frozen_inputs(
        panel, protocol, config)
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
        # Binds this artifact to the exact frozen protocol document it was
        # checked against, so a later reader can tell which protocol
        # version certified the run.
        "protocol_path": str(FROZEN_PROTOCOL),
        "protocol_sha256": _file_sha256(FROZEN_PROTOCOL),
        # The environment that produced this artifact, certified above:
        # identity, observed vs frozen runtime versions, and any deviation
        # from the frozen reference_env recorded rather than absorbed.
        "environment": environment,
        "dataset": {**dataset_values, "n_families_smoked": args.n_families},
        "device": args.device,
        "code_commit": code_commit,
        # Recorded alongside code_commit: a commit hash alone does not
        # identify the code that ran when the tree is dirty.
        "git_dirty": tree["dirty"],
        "git_dirty_paths": provenance["dirty_paths"],
        # Untracked files COUNT: an untracked module, sitecustomize.py or a
        # top-level file shadowing an installed package all change what
        # executes while leaving code_commit unable to reproduce it.
        "git_untracked_paths": provenance["untracked_paths"],
        # This stage's own regenerated artifacts and lock: excluded from
        # the determination above, and reported so the exclusion is
        # auditable rather than silent.
        "git_dirty_excluded_own_outputs": provenance["excluded_own_outputs"],
        "git_dirty_excluded_cache_paths": provenance["excluded_cache_paths"],
        "allow_dirty": bool(args.allow_dirty),
        "gpu_smoke": None,
        "runtime_metadata": None,
        "determinism": None,
        "parity_notes": parity_notes(spec.adapter),
        "lock": None,
        "resolved_revision": None,
        "processor_revision": None,
        "size_metadata": None,
        # Provenance problems are listed first: --allow-dirty may RUN, but
        # it may never mint PASS evidence, because the status is derived
        # from problems and a non-clean tree is always one of them.
        "problems": list(provenance["problems"]) + list(frozen_problems),
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
        report["problems"].extend(_check_smoke(
            report["gpu_smoke"], report["size_metadata"] or {},
            report["runtime_metadata"], config.max_new_tokens))
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
        from causal_mllm.replay.registry import dependency_lock_snapshot, update_lock
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
