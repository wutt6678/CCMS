#!/usr/bin/env python3
"""Annotator calibration audit: run the Scale-C calibrated LLM annotator
prompt on the Scale-B rows (0-60) and compare against the Scale-B HUMAN
verdicts.

Rows 0-60 are EXCLUDED from the Scale-C panel (anti-preferential);
they are used here ONLY as calibration material with known human
labels. If the calibrated prompt reproduces the human standard, the
full-pool Scale-C yield is the honest maximum. If not, the prompt must
be fixed before any panel decision.

Usage:
    python3 scripts/scale_c_annotator_calibration_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import scale_c_build as scb

SCALE_B = Path("outputs/families/scale_b_smoke")
OUT_REPORT = Path("outputs/scale_c/calibration_audit_round2.json")


def _state(v) -> str:
    return v.get("state") if isinstance(v, dict) else v


def main() -> None:
    conf = scb._load_credentials()
    skels = [json.loads(l) for l in
             (SCALE_B / "family_skeletons.jsonl").read_text().splitlines()
             if l.strip()]
    human = json.loads((SCALE_B / "review/annotations.json").read_text())
    cands = {}
    for line in (SCALE_B / "candidates.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cands[(r["metadata"]["pair_id"], r["metadata"]["modality"],
                   r["metadata"]["safety"])] = r

    pairs = sorted(human.keys())
    matrix_tq = Counter()
    matrix_img = Counter()
    detail = []
    for pair in pairs:
        sk = next(s for s in skels if s["source"]["source_id"] == pair)
        fam_like = {"category": sk.get("category", ""),
                    "semantic_atoms": sk["semantic_atoms"]}
        for atom in sk["semantic_atoms"]:
            role = atom.get("structural_role")
            if role not in ("terminal_query", "shared_image"):
                continue
            prompt = scb._ATOM_PROMPT.format(
                category=fam_like["category"],
                atom_id=atom["atom_id"],
                structural_role=role,
                divergence=atom.get("divergence", ""),
                modalities=",".join(atom.get("source_modalities") or []),
                surface_forms=scb._atom_context(fam_like, atom, {}),
                image_note="")
            parsed, _raw, prov = scb._call_gateway(
                conf, scb.ANNOTATOR_MODEL, prompt)
            llm = parsed["semantic_equivalence"][
                "multimodal_vs_unimodal"]["state"]
            hum = _state(human[pair][atom["atom_id"]][
                "semantic_equivalence"]["multimodal_vs_unimodal"])
            m = matrix_tq if role == "terminal_query" else matrix_img
            m[(hum, llm)] += 1
            detail.append({"pair": pair, "role": role,
                           "human": hum, "llm": llm})
        print(f"audited {pair}", flush=True)

    def summarize(m: Counter) -> dict:
        total = sum(m.values())
        agree = sum(v for (h, l), v in m.items() if h == l)
        return {"confusion_human_x_llm": {
            f"{h}->{l}": v for (h, l), v in sorted(m.items())},
            "agreement": agree / max(1, total), "n": total}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "annotator_model": scb.ANNOTATOR_MODEL,
        "prompt": "Scale-C calibrated _ATOM_PROMPT (round 2)",
        "material": "Scale-B rows 0-60 human review (excluded from the "
                    "Scale-C panel; calibration use only)",
        "terminal_query": summarize(matrix_tq),
        "shared_image": summarize(matrix_img),
        "detail": detail,
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "detail"},
                     indent=2))


if __name__ == "__main__":
    main()
