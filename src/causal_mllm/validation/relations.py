"""Independent factorial-relations firewall (Iteration-6 hardening).

The variant generators are tested at construction time, but Iteration
6 is the final integrity firewall over the PERSISTED artifact: it must
catch a corrupted families.jsonl even when the generators are correct.
``validate_factorial_relations`` re-derives every structural relation
between the six variants from the artifact alone:

  * H00 / H10 / history_reset carry NO images; H01 / H11 / shuffle do.
  * H01, H11 and shuffle reference the same source-image hashes.
  * Every referenced media path is recorded in source_media, exists,
    decodes (PNG/JPEG magic bytes), and hashes to the recorded sha256.
  * H11 and shuffle contain exactly the same history multiset of
    (role, text, image hashes); only the order differs, and the order
    matches shuffle_permutation applied to the H11 history.
  * shuffle_permutation is a valid, non-identity permutation.
  * All six variants end with the identical canonical terminal hash.

``validate_factorial_semantic_eligibility`` additionally re-derives
the Iteration-5 SEMANTIC eligibility from the persisted annotations
(equivalent / relevant / required_for_joint_interpretation==True per
variant), reusing the exact gate tables from construction.readiness.

Cross-cell TEXT equality is deliberately NOT checked literally: H00/H10
draw unimodal surface forms while H01/H11 draw multimodal forms, whose
equivalence is a semantic annotation, not a string identity. The
factorial cell labels (T, V) are recorded explicitly by the validation
stage instead.

Media file checks are skipped with ``check_media_files=False`` for
environments without the (git-ignored) media store, e.g. the offline
CI unit job reading the committed Scale-B artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from causal_mllm.construction.readiness import (
    _VARIANT_REQUIREMENTS,
    _variant_semantic_reasons,
    harmonization_gaps,
)
from causal_mllm.data.schemas import CausalFamily, VariantData
from causal_mllm.seeds import sha256_text

# Explicit factorial cell labels: (T = unsafe text history, V = image)
FACTORIAL_CELLS = {
    "neutral": (0, 0),
    "text_only": (1, 0),
    "vision_only": (0, 1),
    "cross_modal": (1, 1),
}

_TEXT_ONLY_CONDITIONS = ("neutral", "text_only", "history_reset")
_IMAGE_CONDITIONS = ("vision_only", "cross_modal", "shuffle")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decodes_as_image(data: bytes) -> bool:
    return data.startswith(_PNG_MAGIC) or data.startswith(_JPEG_MAGIC)


def _history(variant: VariantData) -> list:
    return variant.messages[:-1]


def _terminal(variant: VariantData):
    return variant.messages[-1]


def _msg_key(message) -> tuple:
    return (message.role, message.text, tuple(sorted(message.images)))


def validate_factorial_relations(family: CausalFamily, *,
                                 check_media_files: bool = True) -> list[str]:
    """Re-verify the factorial structure of one persisted family."""
    errors: list[str] = []
    variants = family.variants
    required = set(FACTORIAL_CELLS) | {"shuffle", "history_reset"}
    missing = required - set(variants)
    if missing:
        return [f"missing variants: {sorted(missing)}"]

    # 1. Image placement per condition
    for name in _TEXT_ONLY_CONDITIONS:
        for message in variants[name].messages:
            if message.images:
                errors.append(
                    f"{name}: turn {message.turn_index} carries images in "
                    f"a text-only condition"
                )
    for name in _IMAGE_CONDITIONS:
        if not any(m.images for m in variants[name].messages):
            errors.append(f"{name}: condition carries no image")

    # Recorded media hashes from the vision atoms
    recorded: dict[str, str] = {}
    for atom in family.semantic_atoms:
        for media in atom.source_media:
            recorded[media["path"]] = media["sha256"]

    # 2. H01 / H11 / shuffle reference the same source-image hashes
    def image_hashes(name: str) -> list[str]:
        hashes = []
        for message in variants[name].messages:
            for path in message.images:
                hashes.append(recorded.get(path, f"unrecorded:{path}"))
        return sorted(hashes)

    hash_sets = {name: image_hashes(name) for name in _IMAGE_CONDITIONS}
    if not hash_sets["vision_only"]:
        errors.append("vision_only: no recorded image hashes")
    elif not (hash_sets["vision_only"] == hash_sets["cross_modal"]
              == hash_sets["shuffle"]):
        errors.append(
            f"image hashes differ across vision_only/cross_modal/"
            f"shuffle: {hash_sets}"
        )

    # 3. Referenced media: recorded, existing, decodable, hash-matching
    referenced = {
        path
        for variant in variants.values()
        for message in variant.messages
        for path in message.images
    }
    for path in sorted(referenced):
        sha = recorded.get(path)
        if sha is None:
            errors.append(f"media {path}: not recorded in source_media")
            continue
        if not check_media_files:
            continue
        file_path = Path(path)
        if not file_path.exists():
            errors.append(f"media {path}: file missing")
            continue
        data = file_path.read_bytes()
        if not _decodes_as_image(data):
            errors.append(f"media {path}: does not decode as PNG/JPEG")
        if _file_sha256(file_path) != sha:
            errors.append(
                f"media {path}: content hash differs from recorded "
                f"source_media sha256"
            )

    # 4. H11 vs shuffle: identical history multiset, order via permutation
    h11 = [_msg_key(m) for m in _history(variants["cross_modal"])]
    h_shuffle = [_msg_key(m) for m in _history(variants["shuffle"])]
    if sorted(h11) != sorted(h_shuffle):
        errors.append(
            "shuffle history content differs from cross_modal (H11); "
            "only order may change"
        )

    # 5. Permutation valid, non-identity, and consistent with the order
    perm = variants["shuffle"].shuffle_permutation
    if perm is None:
        errors.append("shuffle: missing shuffle_permutation")
    else:
        if sorted(perm) != list(range(len(h11))):
            errors.append(
                f"shuffle: permutation {perm} is not a permutation of "
                f"{len(h11)} history turns"
            )
        elif perm == list(range(len(h11))):
            errors.append("shuffle: permutation is the identity")
        elif [h11[i] for i in perm] != h_shuffle:
            errors.append(
                "shuffle: message order does not match "
                "shuffle_permutation applied to the H11 history"
            )

    # 6. Identical canonical terminal hash across all six variants
    terminal_hashes = {
        sha256_text(_terminal(variant).text)
        for variant in variants.values()
    }
    if len(terminal_hashes) != 1:
        errors.append(
            f"terminal hashes differ across variants: {terminal_hashes}"
        )
    canonical = ((family.validation or {})
                 .get("terminal_harmonization", {})
                 .get("canonical_sha256"))
    if canonical is not None and terminal_hashes != {canonical}:
        errors.append(
            "terminal hash does not match the recorded canonical q* hash"
        )

    return errors


def validate_factorial_semantic_eligibility(family: CausalFamily) -> list[str]:
    """Re-derive Iteration-5 semantic eligibility from the artifact.

    ``validate_causal_family`` is structural: it permits semantically
    valid-but-negative states such as ``not_equivalent`` or
    ``irrelevant``. A persisted family that CARRIES the vision-bearing
    variants (H01/H11/shuffle) must still hold the POSITIVE evidence
    that justified building them:

      * multimodal_vs_unimodal == "equivalent" on every atom that
        crosses modalities,
      * risk_relevance == "relevant" on the vision atom,
      * required_for_joint_interpretation == True for cross_modal /
        shuffle.

    The exact gate tables are REUSED from ``construction.readiness``
    so the firewall and the build-time gates can never drift apart.
    This must catch an artifact whose annotations were edited after
    construction (or a persisted negative control mistakenly built).
    """
    errors: list[str] = []
    errors.extend(
        f"L2: {gap}" for gap in harmonization_gaps(family)
    )
    for name in family.variants:
        requirements = _VARIANT_REQUIREMENTS.get(name)
        if requirements is None:
            errors.append(f"unknown variant '{name}' in persisted artifact")
            continue
        reasons = _variant_semantic_reasons(family, requirements)
        errors.extend(f"{name}: semantic eligibility: {r}" for r in reasons)
    return errors
