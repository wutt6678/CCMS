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
import os
import struct
from collections import Counter
from pathlib import Path

from causal_mllm.replay.errors import ReplayError

# Ordered: a key is assigned to the FIRST matching component. Auxiliary
# heads (e.g. Qwen3.5 multi-token-prediction) are matched first so they
# are never silently folded into the language backbone. Modality towers
# that the frozen protocol never exercises (Phi-4's bundled audio encoder,
# which is dead weight under a vision-only run) are auxiliary for the same
# reason. Vision precedes language because composite key paths can contain
# both: Phi-4 stores its image tower under
# `model.embed_tokens_extend.image_embed.*`, which the `embed_tokens`
# language marker would otherwise swallow whole.
AUXILIARY_MARKERS = (
    "mtp", "multi_token_prediction", "audio_embed", "audio_projection",
)
VISION_MARKERS = (
    "visual", "vision_model", "vision_tower", "vision_encoder",
    "image_model", "image_newline", "vit", "multi_modal_projector",
    "image_embed", "img_processor", "img_projection",
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


def _hub_cache_dir() -> Path:
    cache_root = os.environ.get("HF_HOME")
    return Path(cache_root) / "hub" if cache_root else \
        Path.home() / ".cache" / "huggingface" / "hub"


def _cached_snapshots(model_id: str) -> list[Path]:
    """Snapshot directories already present for a repo (newest name last)."""
    if "/" not in model_id:
        return []
    snapshots = _hub_cache_dir() / (
        "models--" + model_id.replace("/", "--")) / "snapshots"
    if not snapshots.exists():
        return []
    return sorted((path for path in snapshots.iterdir() if path.is_dir()),
                  key=lambda p: p.name)


def _revision_from_snapshot(snapshot: Path) -> str | None:
    """The revision a hub snapshot directory name denotes, if any."""
    name = snapshot.name
    if len(name) == 40 and all(c in "0123456789abcdef" for c in name.lower()):
        return name
    return None


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
        return Path(path)
    except Exception as exc:
        cached = _cached_snapshots(model_id)
        if revision is not None:
            match = [path for path in cached if path.name == revision]
            if len(match) == 1:
                # The hub's completeness check demands EVERY repo file,
                # including ones this project deliberately does not fetch
                # (Ministral-3 ships a redundant 7.7GB
                # consolidated.safetensors alongside the HF shards). The
                # guarantee that actually matters here is that every shard
                # referenced by model.safetensors.index.json is present,
                # which _shard_files() enforces loudly.
                return match[0]
            raise ReplayError(
                f"{model_id}: checkpoint not available locally "
                f"(revision={revision!r}); download it during preflight "
                f"before recording declared size metadata: {exc}") from exc
        # An unpinned lookup needs refs/main, which is absent when the
        # repo was fetched by explicit SHA. Fall back to the cached
        # snapshot only when it is UNAMBIGUOUS - guessing between two
        # revisions would silently misreport the declared size.
        if len(cached) == 1:
            return cached[0]
        if len(cached) > 1:
            raise ReplayError(
                f"{model_id}: {len(cached)} cached snapshots "
                f"({[p.name for p in cached]}) and no pinned revision - "
                f"lock the revision before measuring declared size") from exc
        raise ReplayError(
            f"{model_id}: checkpoint not available locally "
            f"(revision={revision!r}); download it during preflight "
            f"before recording declared size metadata: {exc}") from exc


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
        "revision_used": _revision_from_snapshot(snapshot),
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
