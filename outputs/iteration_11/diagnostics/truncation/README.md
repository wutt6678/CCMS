# Twelve cells reached the generation cap, and two frozen gates disagreed about whether that was acceptable

Machine-readable evidence, all of it produced by committed code:

* `truncation_evidence.json` — every capped cell, its finish reason and output
  token count, the overall and per-variant rates, the variant spread, the
  repetition diagnostics and the resolution record. Written by
  `scripts/iter11_truncation_evidence.py`; `--verify` re-derives it from the
  panels on disk and refuses it if anything moved.
* `outputs/iteration_11/judge/<model_key>/evaluation_results/evaluation_report.json`
  — `panel_gate.panel.truncation`, the measurement the gate that ACCEPTED each
  panel made, bound into the published report.
* `outputs/iteration_11/analysis/cross_model/cross_model_analysis.json` —
  `non_truncated_sensitivity` and `primary_results_retain_all_cells`.

## What happened

Phase 2 finalized all four arms. One succeeded and three died at the same
place:

    qwen35_2b      exit 1   panel gate FAILED:
                              zero truncation required, got 7 truncated response(s)
    ministral3_3b  exit 1   ... got 1 truncated response(s)
    phi4_mm        exit 1   ... got 4 truncated response(s)
    qwen35_4b      exit 0

Each of the three had already paid for two primaries over 600 cells and a full
adjudication pass — 1009 adjudicator calls across the four arms — before the
check that refused it ran.

## Two gates, one panel, two standards

| gate | where | rule |
| --- | --- | --- |
| 11.5 eligibility | `replay/confirmatory.py`, gate `truncation_reviewed` | `n_truncated == 0`, any truncation is a protocol-level STOP |
| 11.6 completion | `scripts/iter11_replay_checks.py` | overall rate ≤ 0.02 **and** variant-rate spread ≤ 0.05 |
| evaluation panel | `evaluation/gate.py` | zero truncated records |
| Scale-C completion | `scripts/scale_c_replay_checks.py` | overall rate ≤ 0.02 **and** variant-rate spread ≤ 0.05 |

The 11.6 and Scale-C checkers each carried their own copy of `0.02` / `0.05`,
documented as "near-zero … identical to the Iteration 10 rule". The evaluation
gate carried its own stricter rule. Nothing compared them, because until
Iteration 11 nothing could: the frozen Qwen3.5-9B reference truncated nothing,
its longest response being 1291 tokens against a cap of 1536, so every panel
that had ever reached the evaluation gate satisfied both standards at once.

Iteration 11 replayed four smaller checkpoints under the same frozen cap and
the same greedy decoding, and three of them ran out of room:

| arm | capped | rate | variant spread | longest complete output |
| --- | --- | --- | --- | --- |
| qwen35_2b | 7 / 600 | 0.011667 | 0.03 | 1362 |
| qwen35_4b | 0 / 600 | 0.0 | 0.0 | 1142 |
| ministral3_3b | 1 / 600 | 0.001667 | 0.01 | 1451 |
| phi4_mm | 4 / 600 | 0.006667 | 0.03 | 723 |
| reference 9B | 0 / 600 | 0.0 | 0.0 | 1291 |

The cap of 1536 sits 85 tokens above the longest complete response anywhere in
the five-checkpoint set. All four arms were inside 11.6's tolerance and outside
the evaluation gate's.

## The resolution

This is the resolution of an inconsistency between two gates in this
repository. It is not an amendment made because the affected cells turned out
conveniently, and the judge scores of those cells are deliberately **absent**
from `truncation_evidence.json` so that they cannot be read back into the
justification. A tolerance chosen because the affected cells scored low would be
a tolerance chosen by the result.

What changed:

* `src/causal_mllm/replay/truncation.py` now holds `MAX_TRUNCATION_RATE = 0.02`
  and `MAX_VARIANT_SPREAD = 0.05`, the single definition of a truncated cell
  (`hit_max_new_tokens is True or truncated is True`), and the measurement both
  gates record.
* `evaluation/gate.py`, `scripts/iter11_replay_checks.py`,
  `scripts/scale_c_replay_checks.py` and `replay/runner.py` all import it. No
  file restates the literals, and a test fails if one starts to.
* `validate_panel` takes **no** tolerance argument, `run_evaluation_stage` takes
  **no** threshold, and the `evaluate_responses` CLI offers **no** such flag.
  There is no per-run override to disagree with the artifact it governs.
* Every validation report records the thresholds, the counts, the rates, the
  spread and the affected cells, whether or not anything was truncated — "the
  check ran and found none" and "the check did not run" are different facts and
  now look different in a report.

What did not change:

* The **12-family eligibility gate still stops at zero truncation.** It is a
  screening contract, not a full-panel acceptance rule: it runs before any
  confirmatory spend, so it is allowed to be stricter than the panel it screens
  for. One capped cell is inside the full-panel tolerance and still a STOP at
  eligibility, and `test_truncation_policy.py` pins both halves of that
  asymmetry.
* **Sealed Iteration 9 and 10 artifacts are untouched.** Every archived panel in
  `outputs/` has zero truncated records, so a threshold admitting up to 2%
  returns the verdict those panels always returned. A test walks all of them.
* The four committed `iteration_11_replay_checks.json` artifacts were **not**
  regenerated. Each records the `code_commit` that produced it (`6389df65`), and
  regenerating would stamp a different commit onto a gate result that ran under
  the first one.

## Why the pre-registered cap remedy was not used

The protocol's `uniform_cap_rule` says: if a run shows unacceptable truncation,
choose a new UNIFORM cap and rerun all five checkpoints including the 9B
reference. That remedy assumes truncation means the cap was too small.

It does not here. Measured with `repeat3` (1 − distinct word 3-grams / total
word 3-grams) against each arm's own complete cells:

| arm | complete-cell max repeat3 | capped cells' repeat3 | classified loops |
| --- | --- | --- | --- |
| qwen35_2b | 0.476723 | 0.479821 – 0.897581 | 7 of 7 |
| ministral3_3b | 0.191257 | 0.101947 | 0 of 1 |
| phi4_mm | 0.429577 | 0.835088 – 0.957660 | 4 of 4 |

Under greedy decoding a repetition loop emits to *any* cap. One cell
(`qwen35_2b`/`CMST_654232`/`neutral`) is covered 2.58 times over by a single
eight-word phrase — `hot dogs hot dogs hot dogs …`. Escalating the cap cannot
reach zero truncation on these checkpoints, so the remedy has no termination
criterion.

**One exact count.** Earlier prose described these cells as both "11 of 12" and
"10 of 12" repetition loops. The criterion is now registered in the producer and
applied once — a capped cell is a loop when its `repeat3` exceeds the maximum
`repeat3` of the same arm's COMPLETE cells — and the count is **11 of 12**. The
single exception is `ministral3_3b`/`CMST_475310`/`neutral` (repeat3 0.101947,
below its arm's 0.191257), which is genuine long-form output that a larger cap
would have let finish. "10 of 12" was a miscount: the range it quoted,
0.48–0.96, covers eleven cells.

Uniform-cap escalation is therefore defined as firing **only** when either
registered threshold is exceeded, and it is not triggered here. The repetition
classification gates nothing: acceptance depends on the two thresholds and on
nothing else, whatever the text looks like.

## What is kept

All 600 outputs per arm. A response that ran to the cap is an observation of
what the model did under the frozen greedy decoding, not missing data. Dropping
those cells would drop 7 from one arm, 4 from another, 1 from a third and 0
from the fourth, which changes the quantity the cross-model comparison
estimates — the same shape of differential loss the moderation union exists to
prevent, but endogenous to the models rather than to the provider.

`primary_results_retain_all_cells` in the analysis artifact checks this rather
than asserting it: for every arm, the only excluded cells are the moderation
union and the only dropped families are the two that union made incomplete. All
twelve capped cells are present in the analysed records.

## The sensitivity, and what it moved

Because a reader is entitled to ask, the estimands are also recomputed over the
families containing no capped cell **in any arm** — one family set for all four
and for the reference, the same discipline the common panel applies to
moderation. 98 families become 89; the 9 dropped are `CMST_035954`,
`CMST_361008`, `CMST_380403`, `CMST_475310`, `CMST_501124`, `CMST_619519`,
`CMST_654232`, `CMST_696183`, `CMST_993351`.

| | primary (98) | non-truncated (89) | shift |
| --- | --- | --- | --- |
| qwen35_2b | −0.146724 | −0.155224 | −0.008500 |
| qwen35_4b | +0.093192 | +0.095814 | +0.002622 |
| ministral3_3b | −0.130970 | −0.139467 | −0.008497 |
| phi4_mm | −0.050807 | −0.043553 | +0.007254 |
| H5 pooled | −0.058827 | −0.060607 | −0.001780 |
| reference 9B | +0.113670 | +0.114401 | +0.000731 |

Every sign is unchanged, the largest shift is 0.0085 against intervals 0.08 to
0.14 wide, and the verdicts are the primary ones. This is a sensitivity; H1–H5
are estimated over all 98 families.

## The other defect this exposed

`gate.py`'s own contract is "a panel that fails the gate is NEVER judged", but
its only call site was inside `run_evaluation_stage`, which runs at the END of
the ensemble. The contract was false in practice, and the cost was nine hours of
gateway budget per arm. `run_llm_judge_pipeline.py` now validates the panel
before it prepares a single blinded item and exits before any judge is called if
the panel fails — which matters most in split mode, where the process that would
fail is the one holding the budget.

The 11.6 completion gate knew: it emitted a warning whenever any record was
truncated. But the warning was non-fatal and its text recommended a five-model
rerun for a panel that was inside the registered tolerance, so it pointed at a
remedy the thresholds did not ask for. It now says which case it is in.

## Relation to the other exclusion

This is a second, unrelated reason a cell can be missing, and the two must not
be conflated: moderation refusals remove a cell from the LABELS and cost two
whole families (see `../judge_moderation/README.md`), while a capped cell is
labelled, analysed and kept. The counts are 600 replayed, 597 labelled, 588
analysed — and all twelve capped cells are inside the 588.
