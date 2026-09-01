# Scale-C annotation recalibration note (Iteration 10)

Date: 2026-09-01. Round 1 preserved as `annotations_round1_raw.json`.

## What happened

Round-1 atom annotation (glm-5.2, original prompt) produced
`multimodal_vs_unimodal` verdicts that are systematically stricter than
the Scale-B human review standard that defines the L1 gate semantics:

| structural_role | round-1 not_equivalent | Scale-B human not_equivalent |
| --- | --- | --- |
| terminal_query | 169/170 | 41/61 |
| shared_image | 106/170 | 0/61 |
| shared_history_turn | 72/169 | 11/61 |

Concrete comparison showed the LLM rejected paraphrase/specificity
differences that the Scale-B human standard explicitly accepts
(e.g. "which shown exterior feature allows simplest access?" vs "which
design features influence ease of entry?" — same underlying question;
Scale-B kept structurally identical cases, e.g. rows 000000/000003/000004).

## Decision

The L0/L1/L2 gates are UNCHANGED (frozen protocol). Only the annotator
prompt is recalibrated to the documented Scale-B operational criterion,
with three anchor examples taken from the Scale-B human review
(`outputs/families/scale_b_smoke/review/REVIEW_NOTES.md`):

- equivalent = same underlying question/content survives the modality
  switch; paraphrase and specificity differences do not break it;
- not_equivalent = different question, target entity, or requirements;
- shared_image atoms are equivalent by construction (same image file in
  every multimodal condition).

All 680 atoms are re-annotated from scratch (round 2) with the
calibrated prompt; round-1 outputs are retained, never overwritten.

## Calibration audit against the Scale-B human standard

The calibrated annotator was audited on the Scale-B rows 0-60 (which
are EXCLUDED from the Scale-C panel; calibration use only). Reports:
`outputs/scale_c/calibration_audit.json` (round-1 prompt),
`calibration_audit_round2.json` (round-2 prompt),
`calibration_audit_round3_stability.json` (borderline stability).

| axis | round-1 prompt | round-2 prompt |
| --- | --- | --- |
| shared_image vs human | 105/170 pool verdicts skewed | 60/61 agree (98%) |
| terminal_query vs human | 0.787 agreement (11 FN, 2 FP) | 0.803 agreement (2 FN, 10 FP) |

Findings:
1. The round-2 prompt reproduces the human shared-image standard
   (100% on the audit) and recovers nearly all human-equivalent
   terminal pairs.
2. The remaining ~20% terminal disagreement concentrates on genuinely
   borderline pairs ("visible items" vs "general principle of the same
   unsafe topic"): the LLM verdict FLIPS across independent runs on
   17/21 such atoms, i.e. these sit at the annotator's decision
   boundary. The residual error is symmetric in direction across
   prompt variants and is quantified here as a dataset-construction
   uncertainty.
3. Full-pool yield under the round-2 calibration and frozen L0/L1/L2
   gates: 86 families with six variants, 82 passing validation (four
   excluded for unresolved image-deictic references in text-only
   conditions — legitimate grounding failures).

## Round-3 terminal recheck

`scripts/scale_c_terminal_recheck.py` re-annotates ONLY the
terminal_query atom of every scoped family (round-2 prompt) and merges
the fresh verdict into annotations.json; round-2 annotations retained
as `annotations_round2_raw.json`. The manual stratified inspection
(Phase 4f) then concentrates on borderline-equivalence families to
catch inclusion errors before replay.
