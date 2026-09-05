#!/usr/bin/env python3
"""Iteration 11.5 — write the pre-registered eligibility selection.

Emits ``outputs/iteration_11/eligibility/selection.json``: the 12 families
the 11.5 eligibility replay is allowed to use, together with the full audit
trail that produced them (tertile cut points, every strata cell's population
and median, which family was chosen in each and why, and where the three
remaining slots went).

Read-only w.r.t. Iteration 8-10 evidence: the inputs are the frozen 9B
reference run and the frozen adjudicated labels, both located by following
the committed pointers in ``outputs/iteration_11/protocol/
frozen_9b_reference.json`` and ``configs/evaluation/scale_profiles.json``.
No target generations, no downloads, no GPU.

WHY THIS FILE IS NOT THE PRE-REGISTRATION. The confirmatory gate re-derives
the selection itself and only then compares, so editing this artifact cannot
change what a run is authorized to replay; ``--verify`` below performs the
same comparison and is what keeps the committed copy honest. The artifact
exists so a human can read the selection and its reasoning without running
anything.

Usage:
    python3 scripts/iter11_write_selection.py            # derive and write
    python3 scripts/iter11_write_selection.py --verify    # read-only gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_mllm.replay.errors import ReplayError  # noqa: E402
from causal_mllm.replay.selection import (  # noqa: E402
    N_ELIGIBILITY_ATTEMPTS,
    N_ELIGIBILITY_FAMILIES,
    SELECTION_ARTIFACT,
    derive_frozen_selection,
)


def canonical(document: dict) -> str:
    """Byte-stable rendering, so the artifact is reproducible.

    Sorted keys and a fixed indent: the document carries no timestamp, no
    hostname and no absolute path, so two derivations from the same frozen
    evidence are byte-identical and ``git diff`` stays quiet.
    """
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and compare; never write")
    ap.add_argument("--force", action="store_true",
                    help="rewrite an artifact whose contents would change")
    args = ap.parse_args()

    path = ROOT / SELECTION_ARTIFACT
    derived = derive_frozen_selection(ROOT)
    text = canonical(derived)

    ids = derived["selected_family_ids"]
    print(f"derived {derived['n_selected_families']} families from "
          f"{derived['n_panel_families']} panel families")
    print(f"  length cuts      t1={derived['length_cuts']['t1']} "
          f"t2={derived['length_cuts']['t2']}")
    print(f"  by length        {derived['by_length_stratum']}")
    print(f"  by risk          {derived['by_risk_stratum']}")
    print(f"  extras           "
          f"{[e['cell'] for e in derived['extra_allocation']]}")
    print(f"  selection sha256 {derived['selected_families_sha256']}")
    print(f"  families         {', '.join(ids)}")
    print(f"  -> {N_ELIGIBILITY_FAMILIES} families x 6 variants = "
          f"{N_ELIGIBILITY_ATTEMPTS} generations per target, "
          f"{4 * N_ELIGIBILITY_ATTEMPTS} across the four targets")

    if args.verify:
        if not path.exists():
            raise ReplayError(
                f"no committed selection artifact at {path}; run "
                f"`scripts/iter11_write_selection.py` and commit it, so the "
                f"pre-registered selection is auditable from the tree")
        committed = path.read_text(encoding="utf-8")
        if committed != text:
            document = json.loads(committed)
            raise ReplayError(
                f"the committed selection artifact at {path} does not match "
                f"a fresh derivation from the frozen evidence "
                f"(committed sha256 "
                f"{document.get('selected_families_sha256')!r} vs derived "
                f"{derived['selected_families_sha256']!r}); either the "
                f"recipe or the frozen inputs have moved, and the "
                f"pre-registered selection must not be edited silently")
        print(f"PASS: {path} matches a fresh derivation")
        return

    if path.exists() and path.read_text(encoding="utf-8") != text \
            and not args.force:
        raise ReplayError(
            f"{path} already exists with different contents; refusing to "
            f"silently redefine the pre-registered selection. Inspect the "
            f"difference, then re-run with --force if the change is "
            f"intended and reviewed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except ReplayError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
