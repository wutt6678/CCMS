<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'vision_ablation'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'vision_ablation' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### The vision ablation of judge A, and what withholding the image did to its agreement

| Claim | Value | Filed in |
| :--- | :--- | :--- |
| items judged | 299 | vision_ablation |
| ablation applied | image_payload_withheld | vision_ablation |
| ablated judge served as | qwen3.8-max | vision_ablation |
| prompts whose sha256 matches frozen judge A | 299 | vision_ablation |
| items excluded by provider input moderation | 1 | vision_ablation |
| item ids excluded | item-0410 | vision_ablation |
| rubric sha256 | ce6c2005 | vision_ablation |
| sampling temperature | 0.00 | vision_ablation |
| seed | 42 | vision_ablation |
| disagreement rate: A vs B (frozen, B blind) | 50.2% | vision_ablation |
| disagreement rate: A_blind vs A (vision effect on A) | 49.5% | vision_ablation |
| disagreement rate: A_blind vs B (both blind) | 58.9% | vision_ablation |
| frozen Iteration 10 rate: image_bearing_disagreement_rate | 50.3% | vision_ablation |
| frozen Iteration 10 rate: image_free_disagreement_rate | 29.3% | vision_ablation |
| disagreeing items: a_vs_b | 150 | vision_ablation |
| disagreeing items: ablind_vs_a | 148 | vision_ablation |
| disagreeing items: ablind_vs_b | 176 | vision_ablation |

*Note.* The gateway moderates the input TEXT and its verdict is not invariant to whether an image is
attached: these items were judged successfully by frozen judge A with the image and are refused
with 400 data_inspection_failed once it is withheld. They are dropped from every arm so the three
rates share one denominator. The ablated judge is the same model id served under the same rubric
and seed, so the shift below is the image and not the judge.

`tab:vision_ablation` -- source `vision_ablation`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
