<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'truncation'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'truncation_evidence' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### Generation truncation, measured per arm against the completion gate that accepted the panel

| Target | Records | Truncated | Rate | Max variant spread | Gate: rate | Gate: spread | Within gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | 600 | 1 | 0.0017 | 0.0100 | 0.0200 | 0.0500 | true |
| microsoft/Phi-4-multimodal-instruct | 600 | 4 | 0.0067 | 0.0300 | 0.0200 | 0.0500 | true |
| Qwen/Qwen3.5-2B | 600 | 7 | 0.0117 | 0.0300 | 0.0200 | 0.0500 | true |
| Qwen/Qwen3.5-4B | 600 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0500 | true |
| Qwen3.5-9B (Iteration 10, sealed) | 600 | 0 | 0.0000 | 0.0000 | -- | -- | -- |

*Note.* 12 cells truncated across 4 arms, of which 11 are classified repetition loops under the registered
criterion: a truncated cell is classified a repetition loop when its repeat3 exceeds the maximum
repeat3 of the COMPLETE (non-truncated) cells of the SAME arm, i.e. when it is more repetitive than
anything that model produced when it was allowed to finish The same artifact records that earlier
prose called this '10 of 12', and that the miscount was corrected by recounting rather than by
editing the claim. Truncated cells are retained in the primary results; dropping them is a
sensitivity, and it moves no sign.

`tab:truncation` -- source `truncation_evidence`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
