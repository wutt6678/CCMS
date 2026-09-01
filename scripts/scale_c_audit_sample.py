#!/usr/bin/env python3
"""Scale-C Phase 9: draw the stratified human-audit sample.

Implements the FROZEN protocol (outputs/scale_c/HUMAN_AUDIT_PROTOCOL.md,
committed pre-results at a114a6a). The LLM ensemble remains the primary
labeling system; this sample only estimates label validity / judge bias.

Strata (deduplicate in order 1 -> 4, seed 42 throughout):
  1. A/B disagreements: min(15, n) uniform.
  2. Threshold-adjacent: ensemble score within +/-0.05 of theta=0.5, 10.
  3. Strict qualifiers under BOTH primaries at theta=0.5: all six
     variants of up to 2 families (interpretation note in AUDIT_NOTES).
  4. Random exact A/B agreements: 10 uniform.

Outputs:
  - human_audit/audit_worksheet.json   (BLINDED: payload + blank scores)
  - human_audit/audit_sample_manifest.json (strata + judge labels,
    sealed until all human scores are entered)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "outputs" / "scale_c" / "llm_judge_artifacts"
OUT = ROOT / "outputs" / "scale_c" / "human_audit"
SEED = 42
THETA = 0.5
JUDGE_FIELDS = (
    "refusal_type", "compliance_level", "safe_redirection",
    "unsafe_compliance_score",
)


def _flat_records(path: Path) -> dict[str, dict]:
    """Load a raw judge file (list of records) -> {item_id: judgment}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rec["item_id"]: rec["judgment"] for rec in data}


def _flat_ensemble(path: Path) -> dict[tuple[str, str], dict]:
    """Load ensemble labels -> {(family_id, variant): judgment}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["labels"] if "labels" in data else data
    flat = {}
    for fid, variants in labels.items():
        for variant, judgment in variants.items():
            flat[(fid, variant)] = judgment
    return flat


def _differs(a: dict, b: dict) -> bool:
    return any(a.get(f) != b.get(f) for f in JUDGE_FIELDS)


def main() -> None:
    items = json.loads((ART / "blinded_items.json").read_text("utf-8"))
    labels_a = _flat_records(ART / "llm_labels_judge_A.json")
    labels_b = _flat_records(ART / "llm_labels_judge_B.json")
    labels_e = _flat_ensemble(ART / "llm_labels_adjudicated.json")
    decision = json.loads(
        (ART / "scale_c_decision_report.json").read_text("utf-8"))

    by_id = {it["item_id"]: it for it in items}
    assert len(by_id) == 600, f"expected 600 blinded items, got {len(by_id)}"

    def key(it: dict) -> tuple[str, str]:
        return (it["family_id"], it["variant"])

    # ---- Stratum 1: A/B disagreements ----------------------------------
    disagreements = sorted(
        iid for iid, it in by_id.items()
        if _differs(labels_a[iid], labels_b[iid]))
    rng = random.Random(SEED)
    s1 = sorted(rng.sample(disagreements, min(15, len(disagreements))))

    # ---- Stratum 2: threshold-adjacent (ensemble) ----------------------
    adjacent = sorted(
        iid for iid, it in by_id.items()
        if abs(labels_e[key(it)]["unsafe_compliance_score"] - THETA) <= 0.05)
    rng = random.Random(SEED)
    s2 = sorted(rng.sample(adjacent, min(10, len(adjacent))))

    # ---- Stratum 3: strict qualifiers under BOTH primaries -------------
    both_families = sorted(
        decision["qualifiers_at_theta"]["both_primaries"])
    rng = random.Random(SEED)
    picked_fams = rng.sample(both_families, min(2, len(both_families)))
    s3_candidates = sorted(
        iid for iid, it in by_id.items()
        if it["family_id"] in picked_fams)
    if len(s3_candidates) < 10:
        # top up from remaining both-qualifier families, seed-42 order
        extra_fams = sorted(set(both_families) - set(picked_fams))
        rng2 = random.Random(SEED)
        rng2.shuffle(extra_fams)
        for fam in extra_fams:
            if len(s3_candidates) >= 10:
                break
            s3_candidates.extend(sorted(
                iid for iid, it in by_id.items()
                if it["family_id"] == fam))
    s3 = sorted(s3_candidates)

    # ---- Stratum 4: random exact agreements -----------------------------
    agreements = sorted(
        iid for iid, it in by_id.items()
        if not _differs(labels_a[iid], labels_b[iid]))
    rng = random.Random(SEED)
    s4 = sorted(rng.sample(agreements, 10))

    # ---- Deduplicate in stratum order 1 -> 4 ----------------------------
    chosen: dict[str, str] = {}
    counts = {}
    for name, pool in (("1_disagreement", s1), ("2_threshold_adjacent", s2),
                       ("3_qualifier_both", s3), ("4_random_agreement", s4)):
        added = [iid for iid in pool if iid not in chosen]
        for iid in added:
            chosen[iid] = name
        counts[name] = len(added)

    # ---- Blinded worksheet (no judge labels, no family/variant) ---------
    worksheet = {
        "protocol": "outputs/scale_c/HUMAN_AUDIT_PROTOCOL.md",
        "protocol_commit": "a114a6a",
        "rubric": "src/causal_mllm/evaluation/annotation_rubric_v1_1.md",
        "theta": THETA,
        "seed": SEED,
        "n_items": len(chosen),
        "items": [],
    }
    manifest = {
        "seed": SEED,
        "theta": THETA,
        "stratum_counts": counts,
        "n_disagreements_pool": len(disagreements),
        "n_adjacent_pool": len(adjacent),
        "both_primaries_families": both_families,
        "worksheet_sha256": "",
        "sealed": True,
        "items": {},
    }
    for iid in sorted(chosen):
        it = by_id[iid]
        k = key(it)
        worksheet["items"].append({
            "item_id": iid,
            "system_prompt": it["system_prompt"],
            "conversation_history": it["conversation_history"],
            "terminal_query": it["terminal_query"],
            "response": it["response"],
            "response_sha256": it["response_sha256"],
            "human_scores": {
                "refusal_type": None,
                "compliance_level": None,
                "safe_redirection": None,
                "unsafe_compliance_score": None,
                "rationale": None,
                "would_need_rubric_change": None,
            },
        })
        manifest["items"][iid] = {
            "stratum": chosen[iid],
            "family_id": it["family_id"],
            "variant": it["variant"],
            "response_sha256": it["response_sha256"],
            "judge_A": labels_a[iid],
            "judge_B": labels_b[iid],
            "ensemble": labels_e[k],
        }

    ws_bytes = json.dumps(
        worksheet, ensure_ascii=False, indent=2).encode("utf-8")
    manifest["worksheet_sha256"] = hashlib.sha256(ws_bytes).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_worksheet.json").write_bytes(ws_bytes)
    (OUT / "audit_sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"Sampled {len(chosen)} items "
          f"(pool: {len(disagreements)} disagreements, "
          f"{len(adjacent)} threshold-adjacent, "
          f"{len(agreements)} agreements)")
    for name, n in counts.items():
        print(f"  stratum {name}: {n}")
    print(f"Worksheet : {OUT / 'audit_worksheet.json'}")
    print(f"Manifest  : {OUT / 'audit_sample_manifest.json'} (SEALED)")


if __name__ == "__main__":
    main()
