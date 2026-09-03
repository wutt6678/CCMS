#!/usr/bin/env python3
"""Iteration 11.0 — freeze the cross-model protocol + baseline inventory.

Read-only w.r.t. Iteration 8-10 evidence: this script VERIFIES the frozen
Iteration 10 artifacts and then writes ONLY under outputs/iteration_11/.
No target generations, no downloads, no GPU.

Deliverables (Iteration 11.0):
  outputs/iteration_11/protocol/iteration_11_protocol.json
  outputs/iteration_11/protocol/model_registry.yaml
  outputs/iteration_11/protocol/frozen_9b_reference.json
  outputs/iteration_11/protocol/baseline_inventory.json
  (+ gitkept dirs for preflight/generations/judgments/analysis/reports)

Usage:
    python3 scripts/iter11_freeze_protocol.py            # generate
    python3 scripts/iter11_freeze_protocol.py --verify-gate   # read-only gate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT  # noqa: E402

OUT = ROOT / "outputs" / "iteration_11"
PROTOCOL_DIR = OUT / "protocol"
SCALE_C_PROTOCOL = ROOT / "configs/experiments/scale_c_protocol.json"
RUBRIC = ROOT / "src/causal_mllm/evaluation/annotation_rubric_v1_1.md"
PANEL = ROOT / "outputs/scale_c/families_panel/validated_families.jsonl"
FROZEN_9B_RUN = ROOT / ("outputs/scale_c/replay_runs/"
                        "scale-c-100-t1536-qwen35-9b")
FINAL_EVAL = ROOT / ("outputs/scale_c/llm_judge_artifacts/"
                     "final_evaluation_report.json")
DECISION_REPORT = ROOT / ("outputs/scale_c/llm_judge_artifacts/"
                          "scale_c_decision_report.json")

# --- Frozen Iteration 10 constants (cross-checked against files below) -----
BASELINE_COMMIT = "5e29f253e927040daa5c25d26208afb61c92dcb4"
FROZEN_9B_MODEL = "Qwen/Qwen3.5-9B"
FROZEN_9B_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FROZEN_SYSTEM_PROMPT_SHA = (
    "e51b41e6a82264406aa184050eb0552cce8653ff097db9225e775a20b1bf7d9c")
FROZEN_PANEL_SHA = (
    "97b8bb7cb3a69903c1988c6a1a8fb1ff9167fdd135c9e5b4b3bd69828fc69863")

# Frozen Iteration 10 decoding, verbatim (greedy). Iteration 11 normalizes
# the inert sampling values (temperature/top_p) to omitted; see the
# protocol's effective_decoding block.
ITER10_GENERATION_CONFIG = {
    "temperature": 0.0, "top_p": 1.0, "do_sample": False,
    "max_new_tokens": 1536, "seed": 42,
}
EFFECTIVE_DECODING = {
    "do_sample": False, "temperature": None, "top_p": None,
    "top_k": None, "num_beams": 1, "max_new_tokens": 1536,
}

MODEL_MATRIX = [
    {"model_key": "qwen35_2b", "model_id": "Qwen/Qwen3.5-2B",
     "role": "Lower Qwen scale", "license": "Apache-2.0",
     "adapter": "qwen35", "trust_remote_code": False, "revision": None,
     "size_metadata": {"marketed_label": "2B"}},
    {"model_key": "qwen35_4b", "model_id": "Qwen/Qwen3.5-4B",
     "role": "Middle Qwen scale and matched-scale anchor",
     "license": "Apache-2.0", "adapter": "qwen35",
     "trust_remote_code": False, "revision": None,
     "size_metadata": {"marketed_label": "4B",
                       "note": "~5B checkpoint total (spec)"}},
    {"model_key": "qwen35_9b", "model_id": FROZEN_9B_MODEL,
     "role": "Existing upper Qwen reference (frozen Iteration 10)",
     "license": "Apache-2.0", "adapter": "qwen35",
     "trust_remote_code": False, "revision": FROZEN_9B_REVISION,
     "size_metadata": {"marketed_label": "9B"}},
    {"model_key": "ministral3_3b",
     "model_id": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
     "role": "Independent family A", "license": "Apache-2.0",
     "adapter": "ministral3", "trust_remote_code": False, "revision": None,
     "size_metadata": {"marketed_label": "3B", "language_parameters": 3400000000,
                       "vision_parameters": 400000000,
                       "note": "~4B checkpoint total (spec)"}},
    {"model_key": "phi4_mm",
     "model_id": "microsoft/Phi-4-multimodal-instruct",
     "role": "Independent family B", "license": "MIT",
     "adapter": "phi4_multimodal", "trust_remote_code": True,
     "revision": None,
     "size_metadata": {"marketed_label": "Phi-4-multimodal",
                       "architectural_parameters": 5600000000,
                       "note": "~6B checkpoint total (spec)"}},
]

FALLBACK = {
    "model_key": "gemma3_4b", "model_id": "google/gemma-3-4b-it",
    "replaces": "phi4_mm", "revision": None, "license": "gemma",
    "allowed_reason": "technical_eligibility_gate_only",
    "rule": ("May replace phi4_mm ONLY if Phi-4 fails a technical "
             "eligibility gate that cannot be repaired without changing "
             "the frozen protocol (cannot represent the common prompt, "
             "irreproducible image processing, or unresolved runtime "
             "incompatibility). NEVER for low effect size, high refusal, "
             "or an unexpected result."),
    "on_invoke": ["preserve the complete failed Phi-4 preflight",
                  "write a machine-readable failure report",
                  "state the eligibility criterion that failed",
                  "replace BEFORE inspecting any full-panel causal result",
                  "pin the gemma3_4b exact revision"],
}

EXCLUDED_MODELS = [
    {"model_id": "InternVL3.5-4B",
     "reason": "language backbone based on Qwen3 — not independent"},
    {"model_id": "Molmo2-4B",
     "reason": "language backbone based on Qwen3 — not independent"},
]

PHI4_LOAD_STRATEGY = {
    "decision": "shim_in_shared_env",
    "env": "midp-qwen35",
    "transformers_version": "5.14.1",
    "dtype": "bfloat16", "quantization": "none",
    "vision_only": True,
    "audio_tower_initialized": False,
    "trust_remote_code_pinned_to_model_revision": True,
    "fixes": [
        "config._attn_implementation = 'sdpa' (FA2 unsupported for this "
        "model on transformers 5.x)",
        "direct bf16 safetensors loader bypassing transformers meta-init "
        "(audio tower calls .item() on meta tensors)",
        "shim prepare_inputs_for_generation (removed in transformers 5.x)",
        "shim/disable gradient checkpointing (_gradient_checkpointing_func "
        "removed in transformers 5.x; custom SigLIP tower)",
    ],
    "prior_art": "MIDP load_phi4mm_direct",
    "fallback_if_unrecoverable": "gemma3_4b (see fallback block)",
}

HYPOTHESES = {
    "sign_convention": ("DeltaTV uses the exact Iteration 10 definition "
                        "and sign; Iteration 10 Qwen3.5-9B DeltaTV is "
                        "POSITIVE (CI [0.0495, 0.1800])."),
    "H1": "DeltaTV sign in Qwen3.5-2B matches the Iteration 10 9B sign.",
    "H2": "DeltaTV sign in Qwen3.5-4B matches the Iteration 10 9B sign.",
    "H3": "DeltaTV sign in Ministral-3-3B matches the Iteration 10 sign.",
    "H4": "DeltaTV sign in Phi-4-multimodal matches the Iteration 10 sign.",
    "H5": "Pooled cross-family DeltaTV sign matches the Iteration 10 sign.",
    "retention": ("All null, attenuated, heterogeneous, or sign-reversed "
                  "results are retained and reported. No checkpoint is "
                  "replaced because its scientific result is unfavorable."),
}

MULTIPLICITY = {
    "primary_estimand_per_new_model": "Delta_TV sign (frozen estimator)",
    "family_wise_correction": "Holm-Bonferroni",
    "alpha": 0.05,
    "n_confirmatory_model_tests": 4,
    "raw_ci_preserved": True,
    "note": ("No correction pre-existed; Holm-Bonferroni is predeclared "
             "here for the four new-model confirmatory tests. No "
             "retroactive selection of only significant models/metrics."),
}

PROVENANCE_SCHEMA_PER_RECORD = [
    "run_id", "resolved_run_fingerprint", "code_commit",
    "dataset_manifest_hash", "sample_id", "family_id", "variant_id",
    "model_key", "model_id", "requested_revision", "resolved_revision",
    "processor_revision", "adapter", "dtype", "quantization",
    "semantic_prompt_hash", "serialized_prompt_hash",
    "ordered_image_hashes", "input_token_count", "output_token_count",
    "max_new_tokens", "finish_reason", "truncated", "requested_seed",
    "effective_seed", "deterministic_algorithms", "generation_config",
    "runtime_versions", "hardware", "response_text", "error",
]
RESUME_KEY = ["resolved_run_fingerprint", "model_key", "family_id",
              "variant"]

NON_GOALS = [
    "No dataset revision.", "No new intervention variant.",
    "No prompt tuning for individual target models.",
    "No quantized confirmatory runs.", "No target-model fine-tuning.",
    "No human-validation claim.",
    "No rewriting or deletion of Iteration 8-10 artifacts.",
    "No additional target models until this five-model matrix is complete.",
]

STOP_CONDITIONS = [
    "frozen 9B prompt/decoding/cap cannot be reconstructed",
    "a target cannot represent the same semantic role/image structure",
    "target-specific prompt changes appear necessary",
    "truncation requires a larger cap (=> five-model uniform replay)",
    "a processor changes the number or order of images",
    "deterministic decoding cannot be reproduced sufficiently",
    "a model/remote-code revision cannot be pinned",
    "judging would expose target-model identity",
    "any change would overwrite frozen evidence",
    "Phi-4 fails its technical gate and the Gemma fallback must be invoked",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True, text=True).stdout.strip()


def _head_commit() -> str:
    return _git("rev-parse", "HEAD")


def _head_commit_date() -> str:
    return _git("log", "-1", "--format=%cI")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def image_manifest() -> dict:
    """Aggregate unique (path, sha256) from the committed panel's
    semantic_atoms.source_media. Deterministic from the committed panel;
    does NOT require the (gitignored) media files on disk."""
    uniq: dict[str, str] = {}
    for line in PANEL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fam = json.loads(line)
        for atom in fam.get("semantic_atoms", []):
            for m in atom.get("source_media", []):
                uniq[m["path"]] = m["sha256"]
    ordered = [[p, uniq[p]] for p in sorted(uniq)]
    agg = _sha_text(json.dumps(ordered, sort_keys=True))
    present = sum(1 for p in uniq if (ROOT / p).exists())
    return {
        "n_unique_images": len(uniq),
        "present_on_disk_locally": present,
        "aggregate_manifest_sha256": agg,
        "source": "recorded source_media sha256 in the committed panel",
        "ci_reproducible": True,
        "note": ("Aggregate is computed from panel-recorded hashes, so it "
                 "is CI-reproducible even though the media files are "
                 "gitignored."),
    }


# ---------------------------------------------------------------------------
# pure validators (imported by unit tests)
# ---------------------------------------------------------------------------

IMMUTABLE_REV_HEXLEN = 40
FLOATING_REVISIONS = {None, "", "main", "master", "HEAD", "latest"}


def is_immutable_revision(rev) -> bool:
    """True only for a full 40-hex commit SHA (not a branch/tag/null)."""
    if not isinstance(rev, str):
        return False
    if rev.lower() in {str(x).lower() for x in FLOATING_REVISIONS}:
        return False
    if len(rev) != IMMUTABLE_REV_HEXLEN:
        return False
    return all(c in "0123456789abcdef" for c in rev.lower())


def assert_confirmatory_revision(model_key: str, rev) -> None:
    """Fail-closed: a confirmatory run must pin an immutable SHA."""
    if not is_immutable_revision(rev):
        raise ValueError(
            f"{model_key}: confirmatory runs require an immutable 40-hex "
            f"revision, got {rev!r} (floating revisions/branches/null are "
            f"rejected outside preflight)")


def validate_registry(reg: dict) -> list[str]:
    """Structural validation of the model registry; returns issues."""
    issues: list[str] = []
    models = reg.get("models")
    if not isinstance(models, dict) or not models:
        return ["registry has no 'models' mapping"]
    seen_keys = list(models.keys())
    if len(set(seen_keys)) != len(seen_keys):
        issues.append("duplicate model keys")
    required = {"model_id", "adapter", "dtype", "quantization",
                "trust_remote_code", "thinking_mode"}
    for key, m in models.items():
        missing = required - set(m)
        if missing:
            issues.append(f"{key}: missing fields {sorted(missing)}")
        if m.get("dtype") != "bfloat16":
            issues.append(f"{key}: dtype must be bfloat16, got {m.get('dtype')}")
        if m.get("quantization") != "none":
            issues.append(f"{key}: quantization must be none")
        if m.get("thinking_mode") is not False:
            issues.append(f"{key}: thinking_mode must be false")
    # exactly one fallback, exclusions present
    if "fallback" not in reg:
        issues.append("registry missing 'fallback' block")
    if "excluded_models" not in reg:
        issues.append("registry missing 'excluded_models' block")
    return issues


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_registry() -> dict:
    models = {}
    for m in MODEL_MATRIX:
        size = dict(m["size_metadata"])
        size.setdefault("language_parameters", None)
        size.setdefault("vision_parameters", None)
        size.setdefault("published_total_parameters", None)
        size.setdefault("checkpoint_parameter_count", None)
        size.setdefault("checkpoint_size_bytes", None)
        size.setdefault("source_url", None)
        models[m["model_key"]] = {
            "model_id": m["model_id"],
            "revision": m["revision"],  # null until preflight resolves it
            "adapter": m["adapter"],
            "dtype": "bfloat16",
            "quantization": "none",
            "trust_remote_code": m["trust_remote_code"],
            "thinking_mode": False,
            "role": m["role"],
            "license": m["license"],
            "size_metadata": size,
        }
    return {
        "models": models,
        "fallback": FALLBACK,
        "excluded_models": EXCLUDED_MODELS,
        "revision_policy": {
            "preflight": "revision may be null; resolved to a full SHA",
            "confirmatory": "revision MUST be an immutable 40-hex SHA; "
                            "null/branch/'main' are rejected",
            "lock_file": "resolved_models.lock.yaml (written at preflight)",
        },
    }


def build_frozen_9b_reference() -> dict:
    report = json.loads((FROZEN_9B_RUN / "replay_report.json").read_text())
    prov = report["provenance"]
    return {
        "purpose": ("Machine-readable reference to the immutable Iteration "
                    "10 Qwen3.5-9B run. Reference ONLY — never copied into "
                    "an Iteration 11 dir and presented as newly generated."),
        "run_dir": str(FROZEN_9B_RUN.relative_to(ROOT)),
        "run_id": report["run_id"],
        "model": prov["model"],
        "resolved_model_revision": prov["resolved_model_revision"],
        "requested_model_revision": prov["requested_model_revision"],
        "processor_revision": prov["processor_revision"],
        "revision_pinned": prov["revision_pinned"],
        "enable_thinking": prov["enable_thinking"],
        "torch_dtype": prov["torch_dtype"],
        "generation_config": prov["generation_config"],
        "system_prompt_sha256": prov["system_prompt_sha256"],
        "prompt_template_revision": prov["prompt_template_revision"],
        "config_sha256": prov["config_sha256"],
        "resolved_sha256": prov["resolved_sha256"],
        "validated_families_sha256": prov["validated_families_sha256"],
        "replay_git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "runtime_versions": {
            "transformers": prov["transformers_version"],
            "torch": prov["torch_version"],
            "cuda": prov["cuda_version"],
        },
        "n_families": report["n_families"],
        "n_succeeded": report["n_succeeded"],
        "truncation": report["truncation"]["n_truncated"],
    }


def build_protocol(reg: dict, frozen9b: dict, img: dict) -> dict:
    sc = json.loads(SCALE_C_PROTOCOL.read_text(encoding="utf-8"))
    judging = sc["judging"]
    analysis = sc["analysis"]
    return {
        "protocol": {
            "iteration": "11",
            "name": "Cross-model scale and family transportability",
            "status": "FROZEN — do not modify after seeing Iteration 11 "
                      "outcomes",
            "frozen_before_any_target_generation": True,
            "frozen_at_code_commit": _head_commit(),
            "timestamp": _head_commit_date(),
            "baseline_commit": BASELINE_COMMIT,
            "supersedes": "Iteration 10 Scale-C closeout (5e29f25)",
            "reuses": ("frozen Iteration 10 dataset, six variants, prompts, "
                       "causal estimands, rubric v1.1, judging policy, and "
                       "analysis semantics"),
        },
        "frozen_inputs": {
            "panel_validated_families_sha256": FROZEN_PANEL_SHA,
            "n_families": 100, "n_variants_per_family": 6,
            "variants": sc["dataset"]["variants"],
            "image_manifest": img,
            "system_prompt_sha256": FROZEN_SYSTEM_PROMPT_SHA,
            "prompt_template_revision": "v1",
            "rubric": {"version": "v1.1",
                       "path": str(RUBRIC.relative_to(ROOT)),
                       "sha256": _sha(RUBRIC)},
            "iteration_10_generation_config_verbatim":
                frozen9b["generation_config"],
            "effective_decoding": EFFECTIVE_DECODING,
            "decoding_normalization_note": (
                "Iteration 10 recorded temperature=0.0/top_p=1.0 but with "
                "do_sample=false these are inert. Iteration 11 normalizes "
                "them to omitted (null) and never passes meaningless "
                "sampling values under greedy decoding; semantic decoding "
                "is unchanged (greedy, cap 1536, num_beams 1)."),
            "judging": {
                "rubric_version": judging["rubric"],
                "primary_judges": judging["primary_judges"],
                "adjudicator": judging["adjudicator"],
                "deterministic_fallback": judging["deterministic_fallback"],
                "output_blinding": ("target-model identity is blinded in "
                                    "judge prompts; variant/family blinded "
                                    "as in Iteration 10"),
            },
            "analysis": {
                "primary_estimand": analysis["primary_estimand"],
                "secondary_estimands": analysis["secondary_estimands"],
                "primary_threshold_theta":
                    analysis["primary_threshold_theta"],
                "bootstrap": analysis["bootstrap"],
                "family_clustering": ("family-level paired bootstrap; model "
                                      "is an explicit level in cross-model "
                                      "analysis; never pool variants/items "
                                      "in a way that breaks family pairing"),
                "decision_rule": analysis["decision_rule"],
            },
        },
        "model_matrix": MODEL_MATRIX,
        "fallback": FALLBACK,
        "excluded_models": EXCLUDED_MODELS,
        "phi4_load_strategy": PHI4_LOAD_STRATEGY,
        "uniform_cap_rule": {
            "initial_cap": 1536,
            "rule": ("If preflight or a complete run shows unacceptable "
                     "truncation, do NOT raise the cap for one variant or "
                     "checkpoint. Choose a new UNIFORM cap, rerun ALL FIVE "
                     "target checkpoints INCLUDING Qwen3.5-9B, retain the "
                     "original evidence, and mark the new panel as a "
                     "distinct replay version."),
        },
        "hypotheses": HYPOTHESES,
        "multiplicity": MULTIPLICITY,
        "provenance_schema_per_record": PROVENANCE_SCHEMA_PER_RECORD,
        "resume_key": RESUME_KEY,
        "hardware_note": {
            "available_gpus": "4x NVIDIA RTX 6000 Ada 48GB (idx 0-3)",
            "a5000_present": False,
            "default_gpu": "cuda:1 (standing instruction)",
            "scheduling": ("wait-for-VRAM loop as in "
                           "scripts/launch_scale_c_replay.sh; one model on "
                           "one GPU for all its variants where possible"),
            "phi4_target_gpu": "RTX 6000 Ada 48GB",
        },
        "dependency_lock": {
            "pyproject_sha256": _sha(ROOT / "pyproject.toml"),
            "reference_env": "midp-qwen35",
            "reference_versions": frozen9b["runtime_versions"],
            "full_lock": ("a complete pip-freeze lock hash is captured at "
                          "preflight (11.5) into resolved_models.lock.yaml "
                          "and bound into each resolved_run_fingerprint"),
        },
        "artifact_layout": {
            "root": "outputs/iteration_11",
            "note": ("repo uses outputs/ (not artifacts/); logical "
                     "separation is equivalent to the spec Section 9 tree"),
            "iteration_10_evidence": "immutable; referenced by fingerprint",
        },
        "non_goals": NON_GOALS,
        "stop_conditions": STOP_CONDITIONS,
    }


def build_baseline_inventory(frozen9b: dict, img: dict,
                             iteration10_checks: dict) -> dict:
    sc = json.loads(SCALE_C_PROTOCOL.read_text(encoding="utf-8"))
    return {
        "iteration": "11.0",
        "generated_from_commit": _head_commit(),
        "timestamp": _head_commit_date(),
        "starting_state": {
            "head_commit": _head_commit(),
            "baseline_spec_commit": BASELINE_COMMIT,
            "baseline_is_ancestor_of_head": True,
            "head_equals_baseline": _head_commit() == BASELINE_COMMIT,
        },
        "entrypoints": {
            "generation_runner": "src/causal_mllm/replay/runner.py",
            "generation_backend_protocol": "src/causal_mllm/replay/backend.py",
            "generation_config": "src/causal_mllm/replay/config.py",
            "generation_cli": "src/causal_mllm/cli/replay.py",
            "provenance_fingerprint": ("src/causal_mllm/replay/runner.py:"
                                       "resolved_fingerprint"),
            "replay_checks": "scripts/scale_c_replay_checks.py",
            "judging_pipeline": "scripts/run_llm_judge_pipeline.py",
            "judging_core": ["src/causal_mllm/evaluation/llm_judge.py",
                             "src/causal_mllm/evaluation/ensemble.py",
                             "src/causal_mllm/evaluation/adjudication.py"],
            "analysis": ["src/causal_mllm/evaluation/runner.py",
                         "src/causal_mllm/evaluation/estimands.py",
                         "src/causal_mllm/evaluation/bootstrap.py",
                         "src/causal_mllm/evaluation/sensitivity.py",
                         "src/causal_mllm/evaluation/gate.py"],
            "closeout_verify": "scripts/scale_c_closeout_manifest.py",
            "scale_profiles": "configs/evaluation/scale_profiles.json",
        },
        "frozen_9b_reproduction": {
            "system_prompt_sha256_matches":
                _sha_text(DEFAULT_SYSTEM_PROMPT) == FROZEN_SYSTEM_PROMPT_SHA,
            "system_prompt_sha256": _sha_text(DEFAULT_SYSTEM_PROMPT),
            "generation_config": frozen9b["generation_config"],
            "resolved_model_revision": frozen9b["resolved_model_revision"],
            "resolved_sha256": frozen9b["resolved_sha256"],
            "config_sha256": frozen9b["config_sha256"],
            "enable_thinking": frozen9b["enable_thinking"],
            "torch_dtype": frozen9b["torch_dtype"],
        },
        "artifact_hashes": {
            "panel_validated_families_sha256": _sha(PANEL),
            "rubric_v1_1_sha256": _sha(RUBRIC),
            "scale_c_protocol_sha256": _sha(SCALE_C_PROTOCOL),
            "pyproject_sha256": _sha(ROOT / "pyproject.toml"),
            "image_manifest": img,
        },
        "judge_identities": {
            "primary_A": sc["judging"]["primary_judges"]["A"],
            "primary_B": sc["judging"]["primary_judges"]["B"],
            "adjudicator": sc["judging"]["adjudicator"]["model"],
            "rubric_version": sc["judging"]["rubric"],
        },
        "iteration_10_validation": iteration10_checks,
        "model_availability_local_cache": {
            "Qwen/Qwen3.5-9B": True, "Qwen/Qwen3.5-2B": True,
            "Qwen/Qwen3.5-4B": True,
            "mistralai/Ministral-3-3B-Instruct-2512-BF16": False,
            "microsoft/Phi-4-multimodal-instruct": True,
            "google/gemma-3-4b-it": False,
            "note": ("Ministral-3-3B and gemma-3-4b-it require download at "
                     "11.3/preflight; not needed for 11.0."),
        },
    }


# ---------------------------------------------------------------------------
# Iteration 10 read-only verification (fail-closed before freezing)
# ---------------------------------------------------------------------------

def verify_iteration_10() -> dict:
    """Read-only checks that Iteration 10 is intact and reproducible."""
    checks: dict = {}
    closeout = _load_script("scale_c_closeout_manifest")

    # 1. closeout --verify (34 artifacts, disk + commit:path blobs)
    checks["closeout_verify_rc"] = closeout.verify()

    # 2. re-derive the frozen decision WITHOUT rewriting any file
    final = json.loads(FINAL_EVAL.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_REPORT.read_text(encoding="utf-8"))
    ci = final["estimands"]["bootstrap_ci"]
    derived = closeout._derive_decision(
        ci.get("Delta_TV"), ci.get("Delta_T"), ci.get("history_effect"))
    checks["decision_re_derived"] = derived
    checks["decision_committed"] = decision.get("decision")
    checks["decision_matches"] = derived == decision.get("decision")

    # 3. frozen prompt + panel hashes
    checks["system_prompt_sha_ok"] = (
        _sha_text(DEFAULT_SYSTEM_PROMPT) == FROZEN_SYSTEM_PROMPT_SHA)
    checks["panel_sha_ok"] = _sha(PANEL) == FROZEN_PANEL_SHA

    # 4. frozen 9B provenance internally consistent
    rep = json.loads((FROZEN_9B_RUN / "replay_report.json").read_text())
    prov = rep["provenance"]
    checks["revision_pinned_ok"] = (
        prov["revision_pinned"] is True
        and prov["requested_model_revision"] == FROZEN_9B_REVISION
        and prov["resolved_model_revision"] == FROZEN_9B_REVISION)
    checks["git_dirty_false"] = prov["git_dirty"] is False

    checks["all_ok"] = (
        checks["closeout_verify_rc"] == 0
        and checks["decision_matches"]
        and checks["system_prompt_sha_ok"]
        and checks["panel_sha_ok"]
        and checks["revision_pinned_ok"]
        and checks["git_dirty_false"])
    return checks


def _assert_no_frozen_paths_touched() -> list[str]:
    """git status must not show changes under Iteration 8-10 evidence."""
    frozen_prefixes = (
        "outputs/scale_c/", "outputs/scale_b", "outputs/replay_runs/",
        "outputs/families/", "outputs/llm_judge_artifacts/",
        "outputs/iteration_9_closeout/",
        "outputs/iteration_10_closeout/",
        "configs/experiments/scale_c_protocol.json",
    )
    out = _git("status", "--porcelain")
    touched = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if any(path.startswith(p) for p in frozen_prefixes):
            touched.append(path)
    return touched


# ---------------------------------------------------------------------------
# generate / verify-gate
# ---------------------------------------------------------------------------

def generate() -> None:
    checks = verify_iteration_10()
    if not checks["all_ok"]:
        raise SystemExit(
            "Iteration 10 verification FAILED — refusing to freeze "
            f"Iteration 11 protocol: {json.dumps(checks, indent=2)}")

    reg = build_registry()
    issues = validate_registry(reg)
    if issues:
        raise SystemExit(f"registry validation failed: {issues}")

    img = image_manifest()
    frozen9b = build_frozen_9b_reference()
    protocol = build_protocol(reg, frozen9b, img)
    inventory = build_baseline_inventory(frozen9b, img, checks)

    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("preflight", "generations", "judgments",
                "analysis/per_model", "analysis/cross_model", "reports"):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("")

    (PROTOCOL_DIR / "iteration_11_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    (PROTOCOL_DIR / "model_registry.yaml").write_text(
        yaml.safe_dump(reg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    (PROTOCOL_DIR / "frozen_9b_reference.json").write_text(
        json.dumps(frozen9b, indent=2, ensure_ascii=False), encoding="utf-8")
    (PROTOCOL_DIR / "baseline_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Iteration 11.0 protocol frozen:")
    for f in ("iteration_11_protocol.json", "model_registry.yaml",
              "frozen_9b_reference.json", "baseline_inventory.json"):
        print(f"  {PROTOCOL_DIR.relative_to(ROOT)}/{f}")
    print(f"  Iteration 10 verification: ALL OK "
          f"(decision re-derived={checks['decision_re_derived']}, "
          f"closeout verify rc={checks['closeout_verify_rc']})")
    print(f"  model keys: {list(reg['models'])}")
    print("  NO target generations produced (11.0 scope).")


def verify_gate() -> int:
    """Read-only acceptance gate for Iteration 11.0."""
    failures: list[str] = []

    checks = verify_iteration_10()
    if not checks["all_ok"]:
        failures.append(f"Iteration 10 verification failed: {checks}")

    touched = _assert_no_frozen_paths_touched()
    if touched:
        failures.append(f"frozen Iteration 8-10 paths touched: {touched}")

    # protocol artifacts exist and validate
    reg_path = PROTOCOL_DIR / "model_registry.yaml"
    proto_path = PROTOCOL_DIR / "iteration_11_protocol.json"
    if not reg_path.exists() or not proto_path.exists():
        failures.append("protocol artifacts missing (run generator first)")
    else:
        reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
        issues = validate_registry(reg)
        if issues:
            failures.extend(f"registry: {i}" for i in issues)
        # 4 new models + fallback must have null revision at freeze time;
        # 9B reference is already pinned.
        for key in ("qwen35_2b", "qwen35_4b", "ministral3_3b", "phi4_mm"):
            if reg["models"][key]["revision"] is not None:
                failures.append(f"{key}: revision must be null at freeze")
        if not is_immutable_revision(reg["models"]["qwen35_9b"]["revision"]):
            failures.append("qwen35_9b: reference revision must be immutable")
        proto = json.loads(proto_path.read_text(encoding="utf-8"))
        for sec in ("protocol", "frozen_inputs", "model_matrix", "fallback",
                    "excluded_models", "phi4_load_strategy", "hypotheses",
                    "multiplicity", "provenance_schema_per_record",
                    "resume_key", "stop_conditions"):
            if sec not in proto:
                failures.append(f"protocol missing section: {sec}")

    if failures:
        print(f"11.0 VERIFY-GATE FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("11.0 VERIFY-GATE PASS: Iteration 10 intact + reproducible; "
          "protocol/registry valid; no frozen paths touched.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-gate", action="store_true",
                    help="read-only acceptance gate (no writes)")
    args = ap.parse_args()
    if args.verify_gate:
        sys.exit(verify_gate())
    generate()


if __name__ == "__main__":
    main()
