<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'selection'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'selection' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### The eligibility subsample, and the fact that it was chosen without looking at any target

| Claim | Value | Filed in |
| :--- | :--- | :--- |
| families in the panel | 100 | selection |
| families selected for the eligibility gate | 12 | selection |
| grid cells the selection is stratified over | 9 | selection |
| extra allocations beyond one per cell | 3 | selection |
| length strata | short, median, long | selection |
| risk strata | compliant, partial, noncompliant | selection |
| selected by length stratum | {"long": 4, "median": 4, "short": 4} | selection |
| selected by risk stratum | {"compliant": 3, "noncompliant": 4, "partial": 5} | selection |
| length tertile cuts | {"t1": 343.0, "t2": 467.5} | selection |
| risk rule | max adjudicated compliance_level per family; compliant=0, partial=1, noncompliant=2 or 3 | selection |
| selected families sha256 | f2806e49 | selection |
| deterministic | true | selection |
| uses candidate target information | false | selection |
| selected family ids | CMST_009475, CMST_058501, CMST_103738, CMST_283869, CMST_444027, CMST_692701, CMST_737461, CMST_820699, CMST_874595, CMST_941962, CMST_958802, CMST_989079 | selection |

*Note.* The last boolean is the one that matters: the selection is a function of the frozen Iteration 10
reference and the sealed adjudicated labels, not of any Iteration 11 target, so it cannot have been
tuned toward the result it gates.

`tab:selection` -- source `selection`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
