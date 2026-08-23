"""Schema validation for canonical source examples and causal families."""

from __future__ import annotations

from typing import Any

from causal_mllm.data.schemas import (
    ALL_VARIANTS,
    CanonicalSourceExample,
    CausalFamily,
    VariantName,
)


class SchemaValidationError(Exception):
    """Raised when a record fails schema validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Schema validation failed with {len(errors)} error(s):\n" +
                         "\n".join(f"  - {e}" for e in errors))


def validate_source_example(record: dict) -> list[str]:
    """Validate a raw dictionary against the canonical source example schema.

    Returns a list of error messages (empty list means valid).
    """
    errors: list[str] = []

    # Required top-level fields
    required_str = ["source_dataset", "source_id", "label", "terminal_query"]
    for field_name in required_str:
        if field_name not in record or not isinstance(record[field_name], str):
            errors.append(f"Missing or invalid string field: '{field_name}'")

    if "messages" not in record or not isinstance(record["messages"], list):
        errors.append("Missing or invalid 'messages' (must be a list)")
    else:
        if len(record["messages"]) == 0:
            errors.append("'messages' must not be empty")
        for i, msg in enumerate(record["messages"]):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}] is not a dict")
                continue
            if "turn_index" not in msg:
                errors.append(f"messages[{i}] missing 'turn_index'")
            if "role" not in msg or msg["role"] not in ("user", "assistant", "system"):
                errors.append(f"messages[{i}] missing or invalid 'role'")
            if "images" in msg and not isinstance(msg["images"], list):
                errors.append(f"messages[{i}]: 'images' must be a list")

    if "terminal_turn_index" not in record or not isinstance(record["terminal_turn_index"], int):
        errors.append("Missing or invalid 'terminal_turn_index' (must be int)")

    if not errors:
        # Cross-field checks
        term_idx = record["terminal_turn_index"]
        msg_indices = {m.get("turn_index") for m in record["messages"]}
        if term_idx not in msg_indices:
            errors.append(
                f"terminal_turn_index={term_idx} not found in message turn_indices={msg_indices}"
            )

    return errors


def validate_causal_family(record: dict) -> list[str]:
    """Validate a raw dictionary against the causal family schema.

    Returns a list of error messages (empty list means valid).
    """
    errors: list[str] = []

    # Required fields
    if "family_id" not in record or not isinstance(record["family_id"], str):
        errors.append("Missing or invalid 'family_id'")

    if "source" not in record or not isinstance(record["source"], dict):
        errors.append("Missing or invalid 'source' (must be a dict)")
    else:
        if "dataset" not in record["source"]:
            errors.append("source missing 'dataset'")
        if "source_id" not in record["source"]:
            errors.append("source missing 'source_id'")

    if "terminal_query" not in record or not isinstance(record["terminal_query"], dict):
        errors.append("Missing or invalid 'terminal_query'")
    else:
        tq = record["terminal_query"]
        if "text" not in tq or not isinstance(tq["text"], str):
            errors.append("terminal_query missing 'text'")
        if "sha256" not in tq or not isinstance(tq["sha256"], str):
            errors.append("terminal_query missing 'sha256'")

    if "semantic_atoms" not in record or not isinstance(record["semantic_atoms"], list):
        errors.append("Missing or invalid 'semantic_atoms'")
    else:
        atom_ids = set()
        for i, atom in enumerate(record["semantic_atoms"]):
            if not isinstance(atom, dict):
                errors.append(f"semantic_atoms[{i}] is not a dict")
                continue
            aid = atom.get("atom_id")
            if aid is None:
                errors.append(f"semantic_atoms[{i}] missing 'atom_id'")
            elif aid in atom_ids:
                errors.append(f"Duplicate atom_id: '{aid}'")
            else:
                atom_ids.add(aid)
            if "source_turns" not in atom or not isinstance(atom["source_turns"], list):
                errors.append(f"semantic_atoms[{i}] missing or invalid 'source_turns'")

    # Variants
    if "variants" not in record or not isinstance(record["variants"], dict):
        errors.append("Missing or invalid 'variants' (must be a dict)")
    else:
        required_names = {v.value for v in ALL_VARIANTS}
        present_names = set(record["variants"].keys())
        missing = required_names - present_names
        if missing:
            errors.append(f"Missing required variants: {sorted(missing)}")

    return errors


def validate_source_strict(record: dict) -> CanonicalSourceExample:
    """Validate and construct a CanonicalSourceExample, raising on error."""
    errors = validate_source_example(record)
    if errors:
        raise SchemaValidationError(errors)
    return CanonicalSourceExample.from_dict(record)


def validate_family_strict(record: dict) -> CausalFamily:
    """Validate and construct a CausalFamily, raising on error."""
    errors = validate_causal_family(record)
    if errors:
        raise SchemaValidationError(errors)
    return CausalFamily.from_dict(record)
