# Scale-A smoke build (5 families × 6 variants = 30 trajectories)

Generated 2026-08-25 with commit `152956c` (Iteration 5) as evidence
for the acceptance gate: 5 real MTMCS Type-B families, all six gated
variants, exact canonical-q hash invariant across every trajectory.

## Status: PLACEHOLDER annotations and harmonizations

The `annotation_provenance` entries record
`model: placeholder-vlm` / `placeholder-llm`: the annotations and
canonical q* here were produced by deterministic test backends so the
full stage chain could be exercised end-to-end. They are NOT human
review.

Before this build counts toward the acceptance gate item
*"human inspection of the five families finds no obvious semantic
leakage"*, the following must be redone with human (or real LLM/VLM)
judgment:

1. `annotated_skeletons.jsonl` — re-annotate via
   `ManualFileAnnotator` / a real `CallableAnnotator`
   (causal atom semantic types, vision risk relevance,
   multimodal_vs_unimodal equivalence).
2. `harmonized_families.jsonl` — re-harmonize via
   `ManualHarmonizer` / a real `CallableHarmonizer`: the placeholder
   simply adopts the multimodal terminal query as canonical q*.
3. Re-run the variants stage and inspect the 30 trajectories.

## Files

| File | Content |
|------|---------|
| `families.jsonl` | Complete families with all six variants |
| `variants_report.json` | Stage-5C report (30 trajectories) |
| `annotation_report.json` | Per-family readiness after annotation |
| `selection_report.json`, `atoms_report.json` | Earlier stages |
