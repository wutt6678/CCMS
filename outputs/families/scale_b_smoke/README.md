# Scale-B research smoke build (20 families × 6 variants = 120 trajectories)

Built 2026-08-25 (Iteration 7) from the human review in
[`review/`](review/REVIEW_NOTES.md): rows 0-60 of MTMCS Type-B were
read in full (four condition dialogues each), images inspected, and
20 factorially eligible families received human annotations and
deictic-free canonical q* with grounding judgments. `max_rows=61`
cuts exactly at the 20th eligible family (row 60).

Evidence status:

- research-data smoke evidence: 20 families × 6 variants = 120
  trajectories, all passing the Iteration-6 automatic validation
  (schema, trajectory, grounding, shared-part leakage) with zero
  exclusions
- 41 reviewed rows are negative controls: their multimodal/text
  histories ask different questions (decided `not_equivalent` by the
  reviewer), so modality would be confounded with wording; they are
  stored in `negative_controls.jsonl` with explicit reasons
- strict causal subset: empty — Risk(q*), Risk(T), Risk(V), Risk(T,V)
  require the frozen-replay model judge (Iteration 8+)

## Files

| File | Content |
|------|---------|
| `families.jsonl` | 20 complete families, six variants each |
| `validated_families.jsonl` | Iteration-6 automatic validation: 20/20 pass |
| `validation_report.json` | Automatic checks; strict subset empty (no judge yet) |
| `negative_controls.jsonl` | 41 decided-but-ineligible families + reasons |
| `review/` | Human review inputs, notes, and generator reference |
