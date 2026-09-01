#!/usr/bin/env python3
"""Scale-C (Iteration 10) primary-hypothesis decision report.

Applies the FROZEN decision rule from
configs/experiments/scale_c_protocol.json to the ensemble evaluation
artifacts. Run AFTER the judge ensemble completes:

    CCMS_SCALE=scale_c python3 scripts/scale_c_decision_report.py

Decision rule (frozen, primary estimand Delta_TV):
  1. Delta_TV 95% CI entirely above zero ->
       POPULATION_INTERACTION: evidence for a population-level
       cross-modal interaction.
  2. Delta_TV CI includes zero but Delta_T and history CIs stay
     above zero -> TEXT_OR_HISTORY_ONLY: evidence supports
       textual/history-dependent degradation, not general
       cross-modal synergy.
  3. Effects disappear -> SAMPLING_SENSITIVE: Scale-B results were
       likely sampling-sensitive.

Qualifier sets at theta=0.5 are reported separately for ensemble,
Judge A, Judge B, and the intersection supported by BOTH primaries.
The qualifier count alone is never treated as proof of an average
causal interaction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ART = Path("outputs/scale_c/llm_judge_artifacts")
PROTOCOL = json.loads(
    (Path(__file__).parent.parent
     / "configs/experiments/scale_c_protocol.json").read_text())


def _ci_block(estimands: dict, name: str) -> dict | None:
    ci = estimands.get("bootstrap_ci") or {}
    return ci.get(name)


def _above_zero(ci: dict | None) -> bool | None:
    if ci is None:
        return None
    return ci["CI_lower"] > 0.0


def _includes_zero(ci: dict | None) -> bool | None:
    if ci is None:
        return None
    return ci["CI_lower"] <= 0.0 <= ci["CI_upper"]


def main() -> None:
    report = json.loads(
        (ART / "final_evaluation_report.json").read_text())
    sens = json.loads((ART / "judge_sensitivity.json").read_text())

    est = report["estimands"]
    dtv = _ci_block(est, "Delta_TV")
    dt = _ci_block(est, "Delta_T")
    dv = _ci_block(est, "Delta_V")
    hist = _ci_block(est, "history_effect")
    order = _ci_block(est, "order_effect")

    # ---- frozen decision rule ------------------------------------------
    if _above_zero(dtv):
        decision = "POPULATION_INTERACTION"
        interpretation = PROTOCOL["analysis"]["decision_rule"][
            "population_interaction"]
    elif _includes_zero(dtv) and _above_zero(dt) and _above_zero(hist):
        decision = "TEXT_OR_HISTORY_ONLY"
        interpretation = PROTOCOL["analysis"]["decision_rule"][
            "text_or_history_only"]
    else:
        decision = "SAMPLING_SENSITIVE"
        interpretation = PROTOCOL["analysis"]["decision_rule"][
            "sampling_sensitive"]

    # ---- qualifier sets at theta = 0.5 ----------------------------------
    theta = report["primary_threshold"]["theta"]
    ensemble_q = report["primary_threshold"]["qualifying_families"]
    judge_a = sens["judges"]["judge_A"]
    judge_b = sens["judges"]["judge_B"]
    both = sens.get("qualifying_under_all_primaries", [])

    out = {
        "iteration": "10",
        "scale": "scale_c",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "file": "configs/experiments/scale_c_protocol.json",
            "theta": PROTOCOL["analysis"]["primary_threshold_theta"],
            "bootstrap": PROTOCOL["analysis"]["bootstrap"],
            "primary_estimand": PROTOCOL["analysis"]["primary_estimand"],
        },
        "estimands_ci": {
            "Delta_TV": dtv, "Delta_T": dt, "Delta_V": dv,
            "history_effect": hist, "order_effect": order,
        },
        "decision": decision,
        "interpretation": interpretation,
        "qualifiers_at_theta": {
            "theta": theta,
            "ensemble": ensemble_q,
            "judge_A": judge_a["qualifying_families"],
            "judge_B": judge_b["qualifying_families"],
            "both_primaries": both,
            "counts": {
                "ensemble": len(ensemble_q),
                "judge_A": judge_a["n_qualifying"],
                "judge_B": judge_b["n_qualifying"],
                "both_primaries": len(both),
                "n_families": report["primary_threshold"]["n_families"],
            },
        },
        "per_judge_ci": {
            "judge_A": judge_a.get("bootstrap_ci"),
            "judge_B": judge_b.get("bootstrap_ci"),
        },
        "adjudication": report.get("adjudication"),
        "caveats": [
            "Qualifier count alone is not proof of an average causal "
            "interaction.",
            "Gateway-hosted judges cannot be revision-pinned; "
            "provider_returned_model / system fingerprint are the only "
            "revision evidence.",
            "Deterministic-fallback outputs (if present) are diagnostics "
            "only, never the primary result.",
        ],
    }
    out_path = ART / "scale_c_decision_report.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 64)
    print(f"Scale-C primary decision: {decision}")
    print("=" * 64)
    print(f"Delta_TV: mean={dtv['mean']:.4f} "
          f"CI=[{dtv['CI_lower']:.4f}, {dtv['CI_upper']:.4f}]")
    print(f"Delta_T : mean={dt['mean']:.4f} "
          f"CI=[{dt['CI_lower']:.4f}, {dt['CI_upper']:.4f}]")
    if hist:
        print(f"history : mean={hist['mean']:.4f} "
              f"CI=[{hist['CI_lower']:.4f}, {hist['CI_upper']:.4f}]")
    print(f"Interpretation: {interpretation}")
    c = out["qualifiers_at_theta"]["counts"]
    print(f"\nStrict qualifiers @ theta={theta}: "
          f"ensemble={c['ensemble']}, A={c['judge_A']}, B={c['judge_B']}, "
          f"both={c['both_primaries']} (of {c['n_families']} families)")
    print(f"\nReport -> {out_path}")
    if decision == "SAMPLING_SENSITIVE":
        print("\nNOTE: 'effects disappear' branch — verify whether CIs "
              "are wide (underpowered) vs genuinely null before the "
              "final write-up.", file=sys.stderr)


if __name__ == "__main__":
    main()
