"""Comprehensive schema inspection for all source datasets.

Generates machine-readable reports for MTMCS-Bench, CoSafe, and MTID.
Run from the project root:
    python -m causal_mllm.adapters.inspect_datasets outputs/schema

MTMCS dialogue semantics (verified against upstream inference code):
  All r*/safe_*/unsafe_* fields are USER turns, not assistant responses.

  TYPE A:
    unsafe: user(r1) → user(r2) → user(unsafe_r3)
    safe:   user(r1) → user(r2) → user(safe_r3)
    Divergence at the terminal turn.

  TYPE B:
    unsafe: user(unsafe_r1) → user(r2) → user(r3)
    safe:   user(safe_r1)   → user(r2) → user(r3)
    Divergence at the opening turn. Terminal query r3 is shared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _sanitize_field_value(val: Any, max_len: int = 40) -> str:
    """Produce a structural descriptor instead of raw content.

    For strings: "<string:length=N>" or truncated preview for short metadata.
    For images: "<PIL.Image:size=WxH>"
    For lists: "<list[N]>"
    For dicts: "<dict[N keys]>"
    """
    if val is None:
        return "<null>"
    if isinstance(val, str):
        if not val.strip():
            return "<empty>"
        return f"<string:length={len(val)}>"
    if isinstance(val, list):
        return f"<list[{len(val)}]>"
    if isinstance(val, dict):
        return f"<dict[{len(val)} keys]>"
    # PIL Images
    if hasattr(val, "size") and hasattr(val, "mode"):
        return f"<PIL.Image:size={val.size[0]}x{val.size[1]},mode={val.mode}>"
    return f"<{type(val).__name__}:{str(val)[:max_len]}>"


def inspect_mtmcs(cache_dir: str | None = None, n: int = 20) -> dict[str, Any]:
    """Inspect ND-25/MCS-bench (MTMCS-Bench) schema for both splits."""
    from datasets import load_dataset, load_dataset_builder

    builder = load_dataset_builder("ND-25/MCS-bench")
    splits = list(builder.info.splits.keys()) if builder.info.splits else []
    features = {k: str(v) for k, v in builder.info.features.items()} if builder.info.features else {}

    report: dict[str, Any] = {
        "source": "mtmcs",
        "huggingface_id": "ND-25/MCS-bench",
        "paper": "https://aclanthology.org/2026.findings-acl.96/",
        "repository": "https://github.com/franciscoliu/MTMCS-Bench",
        "available_splits": splits,
        "features": features,
        "fields": {},
        "example_values": [],
        "schema_analysis": {
            "description": (
                "Multi-turn multimodal contextual safety benchmark with "
                "escalation-based risk. Each scenario provides paired "
                "safe/unsafe user trajectories in both multimodal and "
                "unimodal conditions."
            ),
            "critical_semantics": (
                "ALL r1/unsafe_r1/safe_r1/r2/r3/unsafe_r3/safe_r3 are USER turns. "
                "The upstream evaluation generates assistant responses DYNAMICALLY "
                "after each user turn. They are NOT pre-written assistant responses."
            ),
            "type_a_trajectories": {
                "description": "Divergence at the TERMINAL turn",
                "unsafe": "user(r1) → user(r2) → user(unsafe_r3)",
                "safe":   "user(r1) → user(r2) → user(safe_r3)",
                "shared_fields": ["r1", "r2"],
                "divergent_fields": ["unsafe_r3", "safe_r3"],
                "terminal_query_varies": True,
                "causal_note": (
                    "The final query changes between safe/unsafe. "
                    "Useful for escalation studies but NOT a fixed-terminal-query pair."
                ),
            },
            "type_b_trajectories": {
                "description": "Divergence at the OPENING turn",
                "unsafe": "user(unsafe_r1) → user(r2) → user(r3)",
                "safe":   "user(safe_r1)   → user(r2) → user(r3)",
                "shared_fields": ["r2", "r3"],
                "divergent_fields": ["unsafe_r1", "safe_r1"],
                "terminal_query_varies": False,
                "causal_note": (
                    "Terminal query r3 is IDENTICAL across safe/unsafe. "
                    "Only the opening history differs. "
                    "This is almost exactly (H_safe, q*) vs (H_unsafe, q*)."
                ),
            },
            "key_observations": [
                "type_a and type_b splits, 752 rows each",
                "Each row has multimodal AND unimodal dialogue variants",
                "main image is a real photograph; variant_images are 3 derived images",
                "unsafe_intent describes the harmful goal abstractly",
                "MCQ and T/F questions provide structured evaluation signals",
                "safe_r1/unsafe_r1 and safe_r3/unsafe_r3 are USER turns, not assistant responses",
                "type_b is the most valuable subset for fixed-terminal-query causal experiments",
            ],
            "causal_family_suitability": {
                "strengths": [
                    "Built-in safe/unsafe paired trajectories per scenario",
                    "Both multimodal and unimodal conditions exist",
                    "type_b provides fixed-terminal-query causal pairs",
                    "Real images with diverse safety categories",
                ],
                "limitations": [
                    "type_a terminal query differs between safe/unsafe (not a fixed-q* pair)",
                    "Only 3 user turns per trajectory (short dialogues)",
                    "variant_images purpose needs further investigation",
                ],
                "recommended_path": (
                    "PRIMARY: Use type_b rows for fixed-terminal-query causal families. "
                    "SECONDARY: Use type_a rows for escalation-based history studies."
                ),
            },
        },
    }

    # Load and inspect BOTH splits
    report["total_rows_by_split"] = {}
    split_field_stats: dict[str, dict] = {}

    for split_name in splits:
        ds = load_dataset("ND-25/MCS-bench", split=split_name, cache_dir=cache_dir)
        report["total_rows_by_split"][split_name] = len(ds)

        # Field population analysis for this split
        fields = ["r1", "unsafe_r1", "safe_r1", "r2", "r3", "unsafe_r3", "safe_r3"]
        field_counts = {f: {"nonempty_mm": 0, "empty_mm": 0,
                            "nonempty_uni": 0, "empty_uni": 0}
                        for f in fields}

        for row in ds:
            for modality, dlg_key in [("mm", "multimodal_dialogue"),
                                       ("uni", "unimodal_dialogue")]:
                dlg = row[dlg_key]
                for f in fields:
                    v = dlg.get(f, "")
                    suffix = f"nonempty_{modality}" if (v and v.strip()) else f"empty_{modality}"
                    field_counts[f][suffix] += 1

        split_field_stats[split_name] = field_counts

    report["field_population_by_split"] = split_field_stats

    # Sanitized example values from BOTH splits
    for split_name in splits:
        ds = load_dataset("ND-25/MCS-bench", split=split_name, cache_dir=cache_dir)
        sample_size = min(n, len(ds))
        for i in range(sample_size):
            row = ds[i]
            example: dict[str, Any] = {
                "split": split_name,
                "id": row["id"],
            }
            for key, val in row.items():
                if key in ("image", "variant_images"):
                    example[key] = _sanitize_field_value(val)
                elif isinstance(val, dict):
                    # Sanitize dialogue fields
                    example[key] = {
                        k: _sanitize_field_value(v) for k, v in val.items()
                    }
                elif isinstance(val, list):
                    example[key] = _sanitize_field_value(val)
                else:
                    example[key] = _sanitize_field_value(val)

                # Top-level field analysis
                if key not in report["fields"]:
                    report["fields"][key] = {
                        "type": type(val).__name__,
                        "description": "",
                    }

            report["example_values"].append(example)

    return report


def inspect_cosafe(data_dir: str | None = None, n: int = 20) -> dict[str, Any]:
    """Inspect CoSafe dataset schema from cloned repository."""
    if data_dir is None:
        data_dir = "data/raw/cosafe/CoSafe-Dataset/CoSafe datasets"

    cosafe_dir = Path(data_dir)
    report: dict[str, Any] = {
        "source": "cosafe",
        "repository": "https://github.com/ErxinYu/CoSafe-Dataset",
        "paper": "https://aclanthology.org/2024.emnlp-main.968/",
        "format": "JSONL (one JSON per line, each a list of messages)",
        "available_files": [],
        "fields": {},
        "example_values": [],
        "schema_analysis": {
            "description": (
                "Text-only multi-turn dialogue safety dataset. Conversations "
                "are organized by safety category. Each record is a conversation "
                "trajectory (list of user/assistant messages). The last user "
                "message is the terminal query."
            ),
            "dialogue_structure": {
                "format": "List of {role: user|assistant, content: str}",
                "typical_turns": "5 messages (3 user, 2 assistant)",
                "terminal_query": "Last user message in the trajectory",
            },
            "key_observations": [
                "Text-only (no images)",
                "Categories encoded in filename (comma-separated safety topics)",
                "14 category files, ~100 records each",
                "Alternating user/assistant role structure",
                "Useful as structural template for coreference-based risk",
            ],
            "causal_family_suitability": {
                "strengths": [
                    "Clear multi-turn escalation with coreference",
                    "Diverse safety categories",
                    "Clean JSONL format",
                ],
                "limitations": [
                    "No images — cannot support cross-modal experiments",
                    "Text-only limits modality ablation",
                ],
                "recommended_path": "Structural template for coreference patterns only.",
            },
        },
    }

    if not cosafe_dir.exists():
        report["error"] = f"CoSafe directory not found: {cosafe_dir}"
        return report

    json_files = sorted(cosafe_dir.glob("*.json"))
    total_records = 0
    categories = []

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        records = [json.loads(l) for l in lines if l.strip()]
        category = jf.stem
        categories.append({"file": jf.name, "category": category, "count": len(records)})
        total_records += len(records)
        report["available_files"].append({"file": jf.name, "category": category,
                                          "records": len(records)})

    report["total_categories"] = len(json_files)
    report["total_records"] = total_records
    report["categories"] = categories

    # Sanitized sample records
    if json_files:
        with json_files[0].open("r", encoding="utf-8") as f:
            lines = f.readlines()
        records = [json.loads(l) for l in lines if l.strip()]
        sample_size = min(n, len(records))
        for i in range(sample_size):
            rec = records[i]
            example: dict[str, Any] = {
                "source_file": json_files[0].name,
                "num_messages": len(rec),
                "messages": [],
            }
            for j, msg in enumerate(rec):
                example["messages"].append({
                    "index": j,
                    "role": msg.get("role", "unknown"),
                    "content": f"<string:length={len(msg.get('content', ''))}>",
                })
                if j == 0:
                    report["fields"] = {
                        "record_type": {"type": "list", "description": "List of message dicts"},
                        "message_keys": {"type": "list", "values": list(msg.keys())},
                        "role_values": {"type": "str", "values": ["user", "assistant"]},
                    }
            report["example_values"].append(example)

    return report


def inspect_mtid(cache_dir: str | None = None, n: int = 20) -> dict[str, Any]:
    """Inspect Graph-COM/MTID schema via direct JSONL download."""
    from huggingface_hub import hf_hub_download

    report: dict[str, Any] = {
        "source": "mtid",
        "huggingface_id": "Graph-COM/MTID",
        "repository": "https://github.com/Graph-COM/TurnGate",
        "format": "JSONL (direct download; HF datasets has Json feature incompatibility)",
        "available_files": [
            "harmful_test.jsonl", "harmful_train.jsonl", "harmful_valid.jsonl",
            "benign_test.jsonl", "benign_train.jsonl", "benign_valid.jsonl",
        ],
        "fields": {},
        "example_values": [],
        "schema_analysis": {
            "description": (
                "Multi-turn intervention dataset with 800 unique samples, "
                "20 rollouts per sample, 16000 trajectories. Includes both "
                "harmful and benign conversations with closure-turn annotations."
            ),
            "key_fields": {
                "sample_index": "Unique scenario identifier",
                "rollout_id": "Rollout variant (1-20 per scenario)",
                "target_turn": "Turn where safety failure occurs (closure turn)",
                "target_confidence": "Confidence for target turn label",
                "meta_intent": "The underlying harmful/benign intent",
                "conversation": "List of {turn_id, role, content, hidden_rationale}",
                "asr_classification": "Attack success rate classification",
            },
            "causal_family_suitability": {
                "strengths": [
                    "Explicit closure-turn labels",
                    "Multiple rollouts per scenario",
                    "hidden_rationale provides semantic decomposition",
                ],
                "limitations": [
                    "Text-only (no images)",
                    "Generated dialogues",
                ],
                "recommended_path": "Reference for trajectory metadata and closure-turn design.",
            },
        },
    }

    file_counts = {}
    for fname in ["harmful_test.jsonl", "benign_test.jsonl"]:
        try:
            path = hf_hub_download("Graph-COM/MTID", fname, repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            records = [json.loads(l) for l in lines if l.strip()]
            file_counts[fname] = len(records)

            if "harmful_test" in fname:
                sample_size = min(n, len(records))
                for i in range(sample_size):
                    rec = records[i]
                    example: dict[str, Any] = {}
                    for k, v in rec.items():
                        if k == "conversation":
                            example[k] = f"<list[{len(v)} turns]>"
                            if v:
                                example["conversation_turn_keys"] = list(v[0].keys())
                        else:
                            example[k] = _sanitize_field_value(v)
                        if k not in report["fields"]:
                            report["fields"][k] = {
                                "type": type(v).__name__,
                                "description": "",
                            }
                    report["example_values"].append(example)
        except Exception as e:
            file_counts[fname] = f"error: {e}"

    report["total_rows_by_file"] = file_counts

    # Count all files
    try:
        total = 0
        for fname in report["available_files"]:
            path = hf_hub_download("Graph-COM/MTID", fname, repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                count = sum(1 for l in f if l.strip())
            total += count
        report["total_records"] = total
    except Exception:
        pass

    return report


def save_report(report: dict, output_dir: str | Path) -> Path:
    """Save a schema report to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report['source']}_schema.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples_path = examples_dir / f"{report['source']}_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(report.get("example_values", []), f, indent=2, ensure_ascii=False, default=str)

    return path


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/schema"

    print("Inspecting MTMCS-Bench (both splits)...")
    mtmcs_report = inspect_mtmcs(n=20)
    path = save_report(mtmcs_report, output_dir)
    print(f"  Report saved to {path}")
    print(f"  Total rows: {mtmcs_report.get('total_rows_by_split', {})}")

    print("\nInspecting CoSafe...")
    cosafe_report = inspect_cosafe(n=20)
    path = save_report(cosafe_report, output_dir)
    print(f"  Report saved to {path}")
    print(f"  Total records: {cosafe_report.get('total_records', 'unknown')}")

    print("\nInspecting MTID...")
    mtid_report = inspect_mtid(n=20)
    path = save_report(mtid_report, output_dir)
    print(f"  Report saved to {path}")
    print(f"  Total records: {mtid_report.get('total_records', 'unknown')}")

    print("\nAll inspections complete.")
