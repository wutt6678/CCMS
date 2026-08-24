# Scale-A smoke build (5 families × 6 variants = 30 trajectories)

Rebuilt 2026-08-25 (Iteration-5 repair) from HUMAN review inputs in
[`review/`](review/REVIEW_NOTES.md): `annotations.json` and
`harmonization.json` were written by a reviewer who read all four
condition dialogues and inspected the image for each of the first 10
Type-B rows. The variants stage built the 5 factorially eligible
families (000000, 000003, 000004, 000006, 000008) and routed the 5
ineligible rows (000001, 000002, 000005, 000007, 000009) to
`negative_controls.jsonl` with explicit reasons.

Evidence status:

- pipeline smoke evidence: PASS (30 trajectories, exact canonical-q
  hash invariant, per-variant structural checks)
- research-data smoke evidence: PASS for the five built families,
  subject to the residual caveats in the review notes (terminal-atom
  equivalence confidences 0.80–0.85; cross_modal remains a candidate
  until Iteration-6 behavioral validation)

## Files

| File | Content |
|------|---------|
| `families.jsonl` | 5 complete families, six variants each |
| `negative_controls.jsonl` | 5 ineligible families + gate reasons |
| `variants_report.json` | Stage-5C report (30 trajectories, 5 controls) |
| `annotation_report.json` | Per-family readiness after human annotation |
| `review/` | Human review inputs + notes (committed evidence) |
