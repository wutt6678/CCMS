"""Declared checkpoint size metadata (Iteration 11).

Checkpoint size is read from the weights themselves — safetensors header
metadata (dtype + shape per tensor) — and NEVER inferred from a model
response, a marketing label, or ``total_size / bytes_per_param`` (the
Qwen3.5 checkpoints store a mixed BF16/F32 tensor population, so that
shortcut is simply wrong).

Only headers are parsed, so this costs no GPU memory and loads no
weights: it is safe to run before deciding whether a model is eligible.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

from causal_mllm.replay.errors import ReplayError

# Ordered: a key is assigned to the FIRST matching component. Auxiliary
# heads (e.g. Qwen3.5 multi-token-prediction) are matched first so they
# are never silently folded into the language backbone.
AUXILIARY_MARKERS = ("mtp", "multi_token_prediction")
VISION_MARKERS = (
    "visual", "vision_model", "vision_tower", "vision_encoder",
    "image_model", "image_newline", "vit", "multi_modal_projector",
)
LANGUAGE_MARKERS = (
    "language_model", "lm_head", "embed_tokens", "model.layers",
    "model.norm", "tokenizer", "wte", "wpe",
)


def classify_tensor_key(key: str) -> str:
    """Assign one tensor key to language / vision / auxiliary / other."""
    lowered = key.lower()
    if any(marker in lowered for marker in AUXILIARY_MARKERS):
        return "auxiliary"
    if any(marker in lowered for marker in VISION_MARKERS):
        return "vision"
    if any(marker in lowered for marker in LANGUAGE_MARKERS):
        return "language"
    return "other"


def _safetensors_header(path: Path) -> dict:
    """Parse the JSON header of a safetensors file (no tensor data)."""
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) < 8:
            raise ReplayError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", length_bytes)[0]
        header = handle.read(header_len)
        if len(header) < header_len:
            raise ReplayError(f"{path}: truncated safetensors header")
    try:
        return json.loads(header)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{path}: unreadable safetensors header: {exc}") \
            from exc


def resolve_snapshot_dir(model_id: str, revision: str | None = None,
                         local_files_only: bool = True) -> Path:
    """Local snapshot directory for a hub checkpoint (no download)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - env guarantees this
        raise ReplayError(
            "huggingface_hub is required to resolve checkpoint size") from exc
    try:
        path = snapshot_download(
            model_id, revision=revision, local_files_only=local_files_only)
    except Exception as exc:
        raise ReplayError(
            f"{model_id}: checkpoint not available locally "
            f"(revision={revision!r}); download it during preflight "
            f"before recording declared size metadata: {exc}") from exc
    return Path(path)


def _shard_files(snapshot: Path) -> list[Path]:
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(
            index.read_text(encoding="utf-8")).get("weight_map", {})
        if not weight_map:
            raise ReplayError(f"{index}: empty weight_map")
        return [snapshot / name for name in sorted(set(weight_map.values()))]
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise ReplayError(
            f"{snapshot}: no safetensors shards and no index — cannot read "
            f"declared checkpoint size")
    return shards


def checkpoint_size_metadata(model_id: str, revision: str | None = None,
                             local_files_only: bool = True) -> dict:
    """Exact declared size metadata for one checkpoint.

    Returns parameter counts split by component, the stored dtype
    histogram, shard inventory and byte size — all derived from the
    checkpoint, with ``inferred_from_response: False`` recorded so the
    provenance of the number is unambiguous.
    """
    snapshot = resolve_snapshot_dir(
        model_id, revision=revision, local_files_only=local_files_only)
    shards = _shard_files(snapshot)

    per_component: Counter = Counter()
    dtype_histogram: Counter = Counter()
    other_prefixes: Counter = Counter()
    n_tensors = 0
    for shard in shards:
        if not shard.exists():
            raise ReplayError(f"{shard}: shard listed in index but missing")
        for key, tensor in _safetensors_header(shard).items():
            if key == "__metadata__":
                continue
            shape = tensor.get("shape") or []
            count = 1
            for dim in shape:
                count *= int(dim)
            component = classify_tensor_key(key)
            per_component[component] += count
            dtype_histogram[tensor.get("dtype", "unknown")] += 1
            n_tensors += 1
            if component == "other":
                other_prefixes[".".join(key.split(".")[:3])] += count

    total = sum(per_component.values())
    index = snapshot / "model.safetensors.index.json"
    indexed_total_size = None
    if index.exists():
        indexed_total_size = json.loads(
            index.read_text(encoding="utf-8")).get("metadata", {}).get(
                "total_size")
    return {
        "model_id": model_id,
        "revision_requested": revision,
        "snapshot_path": str(snapshot),
        "n_shards": len(shards),
        "shard_bytes": sum(shard.stat().st_size for shard in shards),
        "index_total_size_bytes": indexed_total_size,
        "n_tensors": n_tensors,
        "checkpoint_parameter_count": total,
        "language_parameters": per_component.get("language", 0),
        "vision_parameters": per_component.get("vision", 0),
        "auxiliary_parameters": per_component.get("auxiliary", 0),
        "unclassified_parameters": per_component.get("other", 0),
        "unclassified_prefixes": dict(other_prefixes.most_common(10)),
        "stored_dtype_histogram": dict(dtype_histogram),
        "size_source": "safetensors_header_shapes",
        "inferred_from_response": False,
    }
