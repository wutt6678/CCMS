<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'environment'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'dependency_lock' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### The environment this evidence was certified against, and the gap between authenticating it and rebuilding it

| Claim | Value | Filed in |
| :--- | :--- | :--- |
| certified Python | 3.10.20 | dependency_lock |
| recorded Python | 3.10.20 | dependency_lock |
| packages in the freeze | 100 | dependency_lock |
| recorded package count | 100 | dependency_lock |
| pip freeze sha256 | c03a5800 | dependency_lock |
| recorded pip freeze sha256 | c03a5800 | dependency_lock |
| dependency lock sha256 | a5669f63 | dependency_lock |
| the freeze is the preimage of the recorded hash | true | dependency_lock |
| reconstructible from the freeze alone | false | dependency_lock |
| torch in the freeze line | 2.8.0 | dependency_lock |
| torch the preflight observed | 2.8.0+cu128 | dependency_lock |
| what pip install -r would fetch for torch | the default-index build of torch==2.8.0, not the 2.8.0+cu128 the preflight observed | dependency_lock |
| numeric packages the analysis depends on | numpy==2.2.6, scipy==1.15.3, torch==2.8.0, pandas==2.3.3 | dependency_lock |
| recreate the certified environment with | conda create -n ccms-iter11 python=3.10.20 -y && conda activate ccms-iter11 && pip install -r outputs/iteration_11/preflight/dependency_freeze.lock.txt && pip install -e . | dependency_lock |
| the caveat that goes with it | pip install -r this freeze installs the DEFAULT-INDEX build of torch (2.8.0, where the preflight observed 2.8.0+cu128). pip freeze drops a version's local segment, so the CUDA build the evidence was actually produced under is not expressible in this file and has to be installed from the index it came from before the rest of the freeze is applied | dependency_lock |
| frozen reference model | Qwen/Qwen3.5-9B | frozen_reference |
| frozen reference revision pinned | true | frozen_reference |
| resolved model revision | c2022362 | frozen_reference |
| reference runtime transformers | 5.14.1 | frozen_reference |
| reference runtime torch | 2.8.0+cu128 | frozen_reference |
| reference runtime CUDA | 12.8 | frozen_reference |
| reference generation config | {"do_sample": false, "max_new_tokens": 1536, "seed": 42, "temperature": 0.0, "top_p": 1.0} | frozen_reference |

*Note.* Two claims sit next to each other in this table and are not the same claim. The freeze IS the
preimage of the hash the lock records, so a checkout can prove which environment produced this
evidence and detect any drift from it. The environment is NOT reconstructible from that freeze
alone, because pip freeze drops a version's local segment and the freeze line for torch names the
default-index build rather than the CUDA one the preflight observed. The paper claims the first and
states the second as a limitation; claiming the second would be false, and claiming only the first
without this row beside it would be misleading.

`tab:environment` -- source `dependency_lock`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
