# Scale-C human audit — sampling notes (Phase 9)

Drawn by `scripts/scale_c_audit_sample.py` under the FROZEN protocol
(`outputs/scale_c/HUMAN_AUDIT_PROTOCOL.md`, commit a114a6a, written
before any results were seen). Seed 42 throughout; verified
deterministic across reruns (worksheet sha256
78b99ca33d26dbe7caaba1db8dfb83be9f08175108cd680615f5dd45e0b49fae).

## Pools

- A/B disagreements (adjudicator-reviewed): 239 of 600
- Threshold-adjacent (ensemble score within ±0.05 of θ=0.5): 19
- Exact A/B agreements: 361
- Strict qualifiers under BOTH primaries at θ=0.5: 11 families

## Draw (45 items, within the frozen 30–50 range)

| Stratum | Target | Drawn | Note |
|---|---|---|---|
| 1 A/B disagreement | 15 | 15 | uniform, seed 42 |
| 2 threshold-adjacent | 10 | 9 | 10 drawn, 1 collided with stratum 1 (dedup order 1→4) |
| 3 qualifier-both families | 10 | 11 | see interpretation note |
| 4 random agreement | 10 | 10 | uniform, seed 42, post-dedup |

## Interpretation notes (documented, protocol text preserved)

- Stratum 3: the protocol says "sample 10 items drawn from families
  qualifying under BOTH primaries (all six variants of at most
  2 families + 2 extras)". All-six-variants blocks come in units of 6,
  so we took all six variants of each of 2 seed-42-sampled families
  (12 candidates, 11 after dedup against earlier strata). This keeps
  within-family completeness — needed to audit the qualifier verdict
  as a family-level claim — at a cost of +1 item over the nominal 10,
  still inside the frozen 30–50 range.
- Stratum 2 shortfall (9 vs 10) is due to the frozen dedup order;
  the pool itself has 19 eligible items.

## Status

- `audit_worksheet.json`: BLINDED scoring sheet (judge-visible payload
  only; no judge labels, no family/variant shown to the reviewer).
- `audit_sample_manifest.json`: SEALED — strata, family/variant map,
  and judge A/B/ensemble labels. Open only after all 45 human scores
  are entered (anti-anchoring per protocol).
- Worksheet sha256 recorded inside the manifest for integrity.

## External confirmation scoring pack (added 2026-09-02, user directive)

Per user instruction, the 45 blinded items are also packed for scoring
by an external GPT-family evaluator, used as a CONFIRMATION layer over
the LLM-ensemble results:

- Pack: `gpt_audit_pack.zip` (11.0 MB, built by
  `scripts/scale_c_audit_pack.py` from the frozen worksheet; source
  worksheet sha256 recorded inside `gpt_audit_pack_manifest.json`).
- Blinding verified: opaque item_ids only; no family/variant/stratum
  metadata or judge labels anywhere in the pack (automated leak scan
  passed; the single "variant" string hit is in rubric v1.1's own
  version-history text).
- Images: embedded under the SAME rule the judges saw (long edge
  ≤1568 px else JPEG q88), so the confirmation scorer evaluates the
  identical visual payload; ORIGINAL-file sha256s are recorded per
  image in `gpt_audit_pack_manifest.json`.
- Deviation note: the frozen protocol specifies a HUMAN reviewer. This
  external-model scoring is a distinct-model confirmation, not a human
  audit; conclusions about "label validity" must be reported as
  model-based cross-checks. The LLM ensemble remains the primary
  labeling system either way, and the sealed manifest stays closed
  until the external `answers.json` is returned.

## Next step

Human reviewer scores the 45 items under rubric v1.1 (blinded), then
the analyses in the protocol (label validity, directional judge bias,
robustness note) produce `audit_report.json`. The external scorer's
`answers.json` (schema in the pack README) feeds the same analyses as
a confirmation layer.

## Findings — external confirmation scoring (unsealed 2026-09-02)

All 45 external scores returned; schema + rubric cross-field
consistency validated (0 issues, 0 would-need-rubric-change flags).
Worksheet sha256 verified against the sealed manifest before unsealing
(`78b99ca…`). Manifest unsealed 2026-09-02T04:05:55Z. Full numbers in
`audit_report.json` (script: `scripts/scale_c_audit_report.py`).

### Label validity vs the adjudicated ensemble (n=45)

| Field | Agreement |
|---|---|
| refusal_type | exact match **0.978** |
| safe_redirection | exact match **0.911** |
| compliance_level | exact match 0.667 (weakest field) |
| unsafe_compliance_score | within ±0.1: **0.622**, mean\|Δ\| 0.152 |
| θ=0.5 flip rate | 0.244 |

vs individual judges: Judge B is closer to the external scorer
(within ±0.1: 0.733, bias +0.069) than Judge A (0.533, +0.162).

### Stratified pattern (contested items are oversampled BY DESIGN)

- **4_random_agreement** (drawn from the 361/600 items where the two
  judges agreed): within ±0.1 = **1.000**, bias **−0.010** — external
  scores align closely with the ensemble in the agreement region.
- **1_disagreement** / **2_threshold_adjacent**: within ±0.1 = 0.467 /
  0.333 with positive bias (+0.234 / +0.156) — divergence concentrates
  on the contested items the sample deliberately oversampled.
- **3_qualifier_both**: refusal match 1.000, bias +0.086 — external
  scores for these items are equal or higher than the ensemble's.

Because the sample is outcome-stratified, these per-stratum rates
describe the sampled items only; they are not population estimates
without stratum reweighting.

### Directional bias and the primary decision (corrected after review)

- Overall external − ensemble signed bias is **+0.128** (external is
  harsher on this contested-oversampled sample). Per-variant bias
  values (cross_modal +0.271, shuffle +0.230, text_only +0.122,
  neutral −0.044, vision_only 0.000, history_reset 0.000; n = 3–10 per
  variant) are **DESCRIPTIVE ONLY**: the audit sample is
  outcome-stratified, so variant-specific biases cannot be generalized
  to the 100-family population.
- **The external confirmation sample is consistent with and does not
  contradict the primary Scale-C decision. It cannot independently
  estimate the population ΔTV because the audit sample is
  outcome-stratified and contains only two complete families** (both
  drawn from successful qualifiers; their within-family contrast
  replication is anecdotal at n = 2 families, not confirmatory).
- **The population claim itself remains supported by the
  preregistered 600-item ensemble analysis: ensemble ΔTV CI
  [0.0495, 0.1800], Judge A [0.0220, 0.1590], and Judge B
  [0.1075, 0.2625]** — all entirely above zero at 95% (5000 paired
  bootstrap resamples, seed 42; `judge_sensitivity.json`).

### Robustness note (protocol §3, corrected after review)

No material validity failure was detected in the sampled items, but
the confirmation has clear limits:

1. The confirmation layer is a distinct MODEL, not a human reviewer —
   Phase 9 is an **external-model confirmation, not a completed human
   audit**. The original drawn worksheet remains unfilled. An actual
   human audit is needed only if the paper intends to claim human
   validation.
2. compliance_level banding is the weakest field (0.667 exact match).
3. Contested-item scores carry a +0.2 external harshness bias;
   absolute per-item scores on contested items — not the causal
   contrasts — are the uncertain quantity.
4. Single external scorer, n = 45, outcome-stratified: overall
   agreement rates neither generalize to the population nor
   independently verify ΔTV.

Directional-bias detail per categorical field (refusal-type and
compliance-level confusion matrices with ordinal signed differences,
safe-redirection rate differences, per stratum and per variant
[descriptive]) is in `audit_report.json`.
