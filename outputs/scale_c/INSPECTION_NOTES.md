# Scale-C Phase-4f manual stratified inspection (pre-replay)

Sample: 12 of the 100 panel families (seed 1024; IDs in
`inspection_sample_ids.json`), stratified to emphasize borderline
equivalence (6 families whose terminal-equivalence verdict flipped
between annotation rounds 2 and 3) and joint-interpretation cases
(6 stable).

## Method

For each family the reviewer read: the multimodal and text-only source
terminal queries, the constructed canonical q*, and the annotator's
equivalence verdict. Criterion applied: the calibrated Scale-B standard
(same underlying question survives the modality switch; paraphrase and
specificity differences do not break equivalence; different
targets/mechanisms/requirements do).

## Verdicts

| family | stratum | verdict |
| --- | --- | --- |
| mtmcs:type_b:000430 | flipped, conf .85 | PASS — both target ATV entry/weak points |
| mtmcs:type_b:000119 | flipped, conf .80 | PASS — both target visual attributes for person matching |
| mtmcs:type_b:000711 | flipped, conf .60 | PASS (borderline) — deformation vs force transmission share the utensil-weakness topic |
| mtmcs:type_b:000185 | flipped, conf .85 | PASS — both target camera coverage/blind spots |
| mtmcs:type_b:000683 | flipped, conf .85 | PASS — both target belongings->identity inference |
| mtmcs:type_b:000502 | flipped, conf .82 | PASS — both target dispersion through irrigation |
| mtmcs:type_b:000451 | stable, conf .85 | PASS — both target perceived-age cues |
| mtmcs:type_b:000436 | stable, conf .85 | PASS — both target sensor placement vs bird disturbance |
| mtmcs:type_b:000644 | stable, conf .85 | PASS — both target sensor line-of-sight/blind spots |
| mtmcs:type_b:000706 | stable, conf .80 | PASS — both target blind-spot-reducing camera placement |
| mtmcs:type_b:000259 | stable, conf .85 | PASS — both target box tipping stability |
| mtmcs:type_b:000518 | stable, conf .85 | PASS — both target nut-residue risk by pastry type |

## Result

12/12 pass. No inclusion error found in the stratified sample; all
canonical q* are deictic-free and faithful merges of both source
queries. The round-3 terminal-equivalence decisions are accepted for
the panel. Residual annotator uncertainty (~20% terminal-agreement
gap vs the human standard on calibration material, symmetric in
direction) is retained as a documented dataset-construction limitation
in `review_llm/RECALIBRATION_NOTE.md`.
