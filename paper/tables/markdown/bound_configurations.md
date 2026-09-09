<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'bound_configurations'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'censoring_bound' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### The bound's four label configurations, and which of them the evidence licenses

| Configuration | What it assumes | Families | Licensed | H | Arm | $p$ | Holm $p$ | Matches ref. |
| :--- | :--- | :--- | :---: | :--- | :--- | ---: | ---: | :---: |
| a_committed_98_families | the confirmatory analysis as filed | 98 | true | H1 | qwen35_2b | 0.0002 | 0.0008 | false |
| a_committed_98_families | the confirmatory analysis as filed | 98 | true | H2 | qwen35_4b | 0.0256 | 0.0272 | true |
| a_committed_98_families | the confirmatory analysis as filed | 98 | true | H3 | ministral3_3b | 0.0002 | 0.0008 | false |
| a_committed_98_families | the confirmatory analysis as filed | 98 | true | H4 | phi4_mm | 0.0136 | 0.0272 | false |
| b_99_families_at_the_labels_the_frozen_rule_gives | the family restored where a label exists | 99 | true | H1 | qwen35_2b | 0.0002 | 0.0008 | false |
| b_99_families_at_the_labels_the_frozen_rule_gives | the family restored where a label exists | 99 | true | H2 | qwen35_4b | 0.0224 | 0.0224 | true |
| b_99_families_at_the_labels_the_frozen_rule_gives | the family restored where a label exists | 99 | true | H3 | ministral3_3b | 0.0002 | 0.0008 | false |
| b_99_families_at_the_labels_the_frozen_rule_gives | the family restored where a label exists | 99 | true | H4 | phi4_mm | 0.0088 | 0.0176 | false |
| c_genuine_worst_case | the missing Ministral label ranged over the whole rubric | 99 | true | H1 | qwen35_2b | 0.0002 | 0.0008 | false |
| c_genuine_worst_case | the missing Ministral label ranged over the whole rubric | 99 | true | H2 | qwen35_4b | 0.0224 | 0.0224 | true |
| c_genuine_worst_case | the missing Ministral label ranged over the whole rubric | 99 | true | H3 | ministral3_3b | 0.0002 | 0.0008 | false |
| c_genuine_worst_case | the missing Ministral label ranged over the whole rubric | 99 | true | H4 | phi4_mm | 0.0088 | 0.0176 | false |
| d_fully_adversarial | every arm's cell ranged over the whole rubric | 99 | false | H1 | qwen35_2b | 0.0002 | 0.0008 | false |
| d_fully_adversarial | every arm's cell ranged over the whole rubric | 99 | false | H2 | qwen35_4b | 0.0528 | 0.0528 | true |
| d_fully_adversarial | every arm's cell ranged over the whole rubric | 99 | false | H3 | ministral3_3b | 0.0002 | 0.0008 | false |
| d_fully_adversarial | every arm's cell ranged over the whole rubric | 99 | false | H4 | phi4_mm | 0.0112 | 0.0224 | false |

*Note.* Read down a hypothesis: (a) is the analysis as filed, (b) restores the family where a label exists,
(c) lets the one genuinely unknown label be anything the rubric permits, and (d) does that to all
four arms at once. Only (d) is unlicensed, and it is printed because it is the bound a hostile
reader will ask for.

`tab:bound_configurations` -- source `censoring_bound`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
