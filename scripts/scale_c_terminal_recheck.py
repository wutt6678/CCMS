#!/usr/bin/env python3
"""Scale-C round-3 terminal-equivalence recheck.

Re-annotates ONLY the terminal_query atom of every scoped Scale-C
family with the calibrated prompt (round-2 anchors) and merges the
fresh verdict into review_llm/annotations.json. Round-2 annotations
are preserved as annotations_round2_raw.json. Checkpointed by pair_id
(resume-safe).

Usage:
    python3 scripts/scale_c_terminal_recheck.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import scale_c_build as scb

OUT = Path("outputs/scale_c/families")
REVIEW = OUT / "review_llm"
CHECKPOINT = REVIEW / "terminal_recheck.checkpoint.json"


def main() -> None:
    conf = scb._load_credentials()
    ann = json.loads((REVIEW / "annotations.json").read_text())
    skels = [json.loads(l) for l in
             (OUT / "family_skeletons.jsonl").read_text().splitlines()
             if l.strip()]
    done = {}
    if CHECKPOINT.exists():
        done = json.loads(CHECKPOINT.read_text())
        print(f"Resuming: {len(done)} families already rechecked")

    pending = [s for s in skels
               if s["source"]["source_id"] not in done]
    print(f"Rechecking terminal atoms: {len(pending)} pending")

    lock = Lock()
    state = {"n": 0}

    def _recheck(sk: dict) -> tuple[str, dict]:
        pair = sk["source"]["source_id"]
        atom = next(a for a in sk["semantic_atoms"]
                    if a.get("structural_role") == "terminal_query")
        prompt = scb._ATOM_PROMPT.format(
            category=sk.get("category", ""),
            atom_id=atom["atom_id"],
            structural_role="terminal_query",
            divergence=atom.get("divergence", ""),
            modalities=",".join(atom.get("source_modalities") or []),
            surface_forms=scb._atom_context(sk, atom, {}),
            image_note="")
        parsed, _raw, prov = scb._call_gateway(
            conf, scb.ANNOTATOR_MODEL, prompt)
        return pair, {
            "atom_id": atom["atom_id"],
            "payload": {
                "semantic_type": parsed["semantic_type"],
                "semantic_description": parsed["semantic_description"],
                "risk_relevance": parsed["risk_relevance"],
                "required_for_joint_interpretation": bool(
                    parsed["required_for_joint_interpretation"]),
                "semantic_equivalence": parsed["semantic_equivalence"],
                "semantic_validation": "llm",
                "annotation_provenance": {"backend": "llm", **prov},
            },
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        for pair, res in ex.map(_recheck, pending):
            with lock:
                done[pair] = res
                state["n"] += 1
                if state["n"] % 10 == 0 or state["n"] == len(pending):
                    CHECKPOINT.write_text(json.dumps(done))
                    print(f"[{state['n']}/{len(pending)}]", flush=True)

    # Merge into annotations.json (replace the terminal atom payload).
    merged = 0
    for pair, res in done.items():
        if pair in ann and res["atom_id"] in ann[pair]:
            scb._validate_annotation_payload(res["payload"])
            ann[pair][res["atom_id"]] = res["payload"]
            merged += 1
    (REVIEW / "annotations.json").write_text(
        json.dumps(ann, indent=2, ensure_ascii=False))
    CHECKPOINT.unlink()
    print(f"Merged {merged} fresh terminal verdicts into annotations.json")


if __name__ == "__main__":
    main()
