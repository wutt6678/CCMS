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

## Next step

Human reviewer scores the 45 items under rubric v1.1 (blinded), then
the analyses in the protocol (label validity, directional judge bias,
robustness note) produce `audit_report.json`.
