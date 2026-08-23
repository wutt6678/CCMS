"""Comprehensive schema inspection for all source datasets.

Generates machine-readable reports for MTMCS-Bench, CoSafe, and MTID.
Run from the project root:
    python -m causal_mllm.adapters.inspect_datasets outputs/schema
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def inspect_mtmcs(cache_dir: str | None = None, n: int = 20) -> dict[str, Any]:
    """Inspect ND-25/MCS-bench (MTMCS-Bench) schema."""
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
                "escalation-based risk. Each scenario has a main image, "
                "3 variant images, an unsafe_intent description, and "
                "paired safe/unsafe dialogues in both multimodal and "
                "unimodal (text-only) conditions."
            ),
            "total_rows": "752 per split",
            "dialogue_structure": {
                "r1": "Round 1 user query (benign, image-referencing for multimodal)",
                "unsafe_r1": "Unsafe assistant response for r1 (often empty)",
                "safe_r1": "Safe assistant response for r1 (often empty)",
                "r2": "Round 2 user query (escalation, more targeted)",
                "r3": "Round 3 user query (usually empty; terminal query is in safe/unsafe_r3)",
                "unsafe_r3": "Unsafe variant response to terminal query",
                "safe_r3": "Safe variant response to terminal query",
            },
            "key_observations": [
                "type_a and type_b splits with 752 rows each",
                "Each row has multimodal AND unimodal dialogue variants",
                "Main image is a real photograph; variant_images are 3 derived images",
                "unsafe_intent field describes the harmful goal abstractly",
                "Escalation: r1 (benign) -> r2 (narrowing) -> r3 (unsafe/safe diverge)",
                "MCQ and T/F questions provide structured evaluation signals",
                "r1/r2 are user queries; safe_r3/unsafe_r3 encode terminal query variants",
            ],
            "causal_family_suitability": {
                "strengths": [
                    "Built-in safe/unsafe pairing per scenario",
                    "Both multimodal and unimodal conditions exist",
                    "Clear escalation structure suitable for history manipulation",
                    "Real images with diverse safety categories",
                ],
                "limitations": [
                    "Terminal query is not byte-identical between safe/unsafe conditions",
                    "Only 2-3 dialogue rounds per scenario",
                    "variant_images purpose needs further investigation",
                    "safe_r3/unsafe_r3 encode both query and response in one field",
                ],
                "recommended_path": (
                    "Path B: supports history experiments but strict 2x2 "
                    "modality intervention requires careful terminal query extraction"
                ),
            },
        },
    }

    # Load actual rows
    ds = load_dataset("ND-25/MCS-bench", split=splits[0] if splits else "type_a",
                      cache_dir=cache_dir)
    report["total_rows_by_split"] = {}
    for s in splits:
        ds_s = load_dataset("ND-25/MCS-bench", split=s, cache_dir=cache_dir)
        report["total_rows_by_split"][s] = len(ds_s)

    sample_size = min(n, len(ds))
    for i in range(sample_size):
        row = ds[i]
        example: dict[str, Any] = {}
        for key, val in row.items():
            if key == "image":
                example[key] = f"PIL.Image(size={val.size}, mode={val.mode})" if val else None
            elif key == "variant_images":
                example[key] = [
                    f"PIL.Image(size={v.size}, mode={v.mode})" if v else None
                    for v in (val or [])
                ]
            elif isinstance(val, dict):
                example[key] = {k: str(v)[:150] for k, v in val.items()}
            elif isinstance(val, list):
                example[key] = f"list[{len(val)} items]"
                if val and len(val) > 0:
                    example[f"{key}_sample"] = json.dumps(val[0], default=str)[:200]
            else:
                example[key] = str(val)[:200]

            if key not in report["fields"]:
                report["fields"][key] = {
                    "type": type(val).__name__,
                    "sample_values": [],
                    "null_count": 0,
                }
            if val is None or (isinstance(val, str) and val.strip() == ""):
                report["fields"][key]["null_count"] += 1
            elif len(report["fields"][key]["sample_values"]) < 3:
                sv = str(val)[:200] if not isinstance(val, (list, dict)) else val
                report["fields"][key]["sample_values"].append(sv)

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
                "message is the terminal query. Risk often depends on references "
                "to earlier dialogue (coreference)."
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
                "Useful as construction template for coreference-based risk",
                "Not suitable as primary multimodal data source",
            ],
            "causal_family_suitability": {
                "strengths": [
                    "Clear multi-turn escalation with coreference",
                    "Diverse safety categories",
                    "Clean JSONL format, easy to parse",
                ],
                "limitations": [
                    "No images - cannot directly support cross-modal experiments",
                    "Text-only limits modality ablation possibilities",
                    "Useful as structural template only",
                ],
                "recommended_path": (
                    "Use as structural template for coreference patterns. "
                    "Not suitable as primary data source for cross-modal families."
                ),
            },
        },
    }

    if not cosafe_dir.exists():
        report["error"] = f"CoSafe directory not found: {cosafe_dir}"
        return report

    # Inspect each category file
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

    # Inspect sample records from the first file
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
                msg_summary = {
                    "index": j,
                    "role": msg.get("role", "unknown"),
                    "content_preview": str(msg.get("content", ""))[:150],
                }
                example["messages"].append(msg_summary)

                if j == 0:
                    report["fields"] = {
                        "record_type": {"type": "list", "description": "List of message dicts"},
                        "message_keys": {"type": "list", "values": list(msg.keys())},
                        "role_values": {"type": "str", "values": ["user", "assistant"]},
                        "content": {"type": "str", "description": "Message text content"},
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
        "format": "JSONL (direct download; HuggingFace datasets has Json feature incompatibility)",
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
                "harmful and benign conversations with response-aware safety "
                "labels and closure-turn annotations."
            ),
            "dialogue_structure": {
                "format": "List of {turn_id, role, content, hidden_rationale}",
                "roles": ["user", "assistant"],
                "hidden_rationale": "Decomposed subgoal description for each turn",
            },
            "key_fields": {
                "sample_index": "Unique identifier for the source scenario",
                "rollout_id": "Rollout variant (1-20 per scenario)",
                "dataset_key": "Source generator identifier",
                "target_turn": "Turn where safety failure occurs (closure turn)",
                "target_confidence": "Confidence score for the target turn label",
                "target_reasoning": "Explanation for why this is the target turn",
                "meta_intent": "The underlying harmful/benign intent",
                "conversation": "List of conversation turns",
                "asr_classification": "Attack success rate classification (C/D/etc.)",
            },
            "key_observations": [
                "harmful splits: 5600 train + 1200 test records",
                "benign splits: 5600 train + 1200 test records",
                "Each conversation has turn_id, role, content, hidden_rationale",
                "target_turn serves as closure-turn annotation",
                "meta_intent describes the underlying goal",
                "20 rollouts per unique sample",
            ],
            "causal_family_suitability": {
                "strengths": [
                    "Explicit closure-turn labels (target_turn)",
                    "Multiple rollouts per scenario",
                    "hidden_rationale provides semantic decomposition",
                    "Both harmful and benign trajectories available",
                ],
                "limitations": [
                    "Text-only (no images)",
                    "Generated dialogues (not organic conversations)",
                    "Json feature type incompatibility with HF datasets library",
                ],
                "recommended_path": (
                    "Use for trajectory metadata patterns and closure-turn "
                    "annotation design. Not suitable as primary multimodal source."
                ),
            },
        },
    }

    # Download and inspect files
    file_counts = {}
    for fname in ["harmful_test.jsonl", "benign_test.jsonl"]:
        try:
            path = hf_hub_download("Graph-COM/MTID", fname, repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            records = [json.loads(l) for l in lines if l.strip()]
            file_counts[fname] = len(records)

            # Inspect first n records from harmful_test
            if "harmful_test" in fname:
                sample_size = min(n, len(records))
                for i in range(sample_size):
                    rec = records[i]
                    example: dict[str, Any] = {}
                    for k, v in rec.items():
                        if k == "conversation":
                            example[k] = f"list[{len(v)} turns]"
                            if v:
                                example["conversation_sample"] = json.dumps(v[0], default=str)[:200]
                        else:
                            example[k] = str(v)[:200]
                        if k not in report["fields"]:
                            report["fields"][k] = {
                                "type": type(v).__name__,
                                "sample_values": [],
                            }
                        if len(report["fields"][k]["sample_values"]) < 3:
                            report["fields"][k]["sample_values"].append(str(v)[:100])
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

    # Also save examples separately
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples_path = examples_dir / f"{report['source']}_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(report.get("example_values", []), f, indent=2, ensure_ascii=False, default=str)

    return path


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/schema"

    print("Inspecting MTMCS-Bench...")
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
