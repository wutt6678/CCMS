# CCMS — Causal Cross-Modal Safety-State Dataset

[![CI](https://github.com/wutt6678/CCMS/actions/workflows/ci.yml/badge.svg)](https://github.com/wutt6678/CCMS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

A research dataset and evaluation pipeline for testing whether **multimodal conversational history causally changes an MLLM's effective safety behavior**.

## Research Question

> Does cross-modal conversational history causally move an MLLM across an effective safety boundary, beyond effects attributable to text alone, vision alone, context length, or the final query?

For a fixed terminal query \(q^*\), we compare model behavior under four controlled history conditions:

| Condition | Description |
|-----------|-------------|
| \(H_{00}\) | Matched neutral history |
| \(H_{10}\) | Relevant textual history only |
| \(H_{01}\) | Relevant visual history only |
| \(H_{11}\) | Complementary text + visual history |

The primary causal estimand is the **cross-modal interaction**:

\[ \Delta_{TV} = Y_{11} - Y_{10} - Y_{01} + Y_{00} \]

where \(Y\) is a safety-behavior outcome (unsafe compliance probability, refusal rate, etc.).

## Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H1 | History causality | Changing preceding history (with fixed query) changes safety behavior |
| H2 | Cross-modal interaction | Joint text+vision effect ≠ sum of text-only + vision-only effects |
| H3 | Temporal-order dependence | Same content, different order → different safety behavior |
| H4 | History-reset recovery | Removing history restores neutral-condition safety behavior |
| H5 | Benign specificity | Topic-matched benign trajectories don't show the same pattern |

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Conda (recommended)
- GPU with ≥ 24 GB VRAM (for inference phase)

### Installation

```bash
git clone https://github.com/wutt6678/CCMS.git
cd CCMS

# Create or use an existing conda environment with PyTorch + CUDA
conda activate <your-env>

# Install the package in development mode
pip install -e ".[dev]"
```

### Run Tests

```bash
# Unit tests (no network required — fast)
python -m pytest -q -m "not slow and not integration"

# All tests including integration (requires network + datasets)
python -m pytest -q

# Integration tests only
python -m pytest -q -m "integration"
```

## Project Structure

```
CCMS/
├── configs/                    # YAML configurations
│   ├── datasets/               #   Source dataset configs (mtmcs, cosafe, mtid)
│   ├── generation/             #   Family building config (mvp.yaml)
│   ├── models/                 #   Model configs (qwen_mllm.yaml)
│   └── evaluation/             #   Evaluation config (default.yaml)
│
├── src/causal_mllm/            # Main Python package
│   ├── data/                   #   Schemas, I/O, validation, logging
│   ├── adapters/               #   Source dataset adapters (MTMCS, CoSafe, MTID)
│   ├── construction/           #   Family/variant builders
│   ├── validation/             #   Family validation checks
│   ├── models/                 #   Model backends (HF, API)
│   ├── inference/              #   Frozen replay inference
│   ├── evaluation/             #   Safety judge + causal metrics
│   ├── cli/                    #   CLI entry points
│   └── seeds.py                #   Deterministic seed utilities
│
├── scripts/                    # Shell scripts for experiment workflows
├── tests/                      # Unit + integration tests
│   ├── unit/                   #   No-network tests (schemas, I/O, seeds, CLI)
│   └── integration/            #   Real-dataset adapter tests
│
├── data/                       # Data directory (gitignored except schemas)
│   ├── raw/                    #   Downloaded source datasets
│   ├── normalized/             #   Canonical normalized examples
│   ├── families/               #   Generated causal families
│   ├── media/                  #   Source/generated images
│   └── splits/                 #   Train/dev/test splits
│
├── outputs/                    # Experiment outputs
│   ├── schema/                 #   Dataset schema reports (tracked)
│   ├── inference/              #   Model responses
│   └── evaluation/             #   Metrics + reports
│
├── pyproject.toml              # Package configuration
├── LICENSE                     # MIT License
└── README.md
```

## Source Datasets

### MTMCS-Bench — Primary Multimodal Source

| Property | Value |
|----------|-------|
| Paper | Liu et al. (2026), Findings of ACL |
| Dataset | [ND-25/MCS-bench](https://huggingface.co/datasets/ND-25/MCS-bench) |
| Size | 752 rows × 2 splits (type_a, type_b) |
| Modalities | Image + multi-turn dialogue |
| Structure | Escalation-based risk with paired safe/unsafe conditions |
| Suitability | **Path B** — history experiments; 2×2 modality intervention needs careful terminal query extraction |

### CoSafe — Structural Template

| Property | Value |
|----------|-------|
| Paper | Yu et al. (2024), EMNLP |
| Repository | [CoSafe-Dataset](https://github.com/ErxinYu/CoSafe-Dataset) |
| Size | ~1,400 records × 14 safety categories |
| Modalities | Text only |
| Structure | Coreference-based risk across multi-turn dialogues |
| Suitability | Structural template for coreference patterns |

### MTID — Trajectory Metadata Reference

| Property | Value |
|----------|-------|
| Paper | TurnGate / MTID |
| Dataset | [Graph-COM/MTID](https://huggingface.co/datasets/Graph-COM/MTID) |
| Size | ~16,000 trajectories (800 samples × 20 rollouts) |
| Modalities | Text only |
| Structure | Closure-turn labels with rollout variants |
| Suitability | Reference for trajectory metadata design |

## CLI Usage

### Inspect Source Datasets

```bash
python -m causal_mllm.cli.inspect_source --dataset mtmcs --n 20
python -m causal_mllm.cli.inspect_source --dataset cosafe --n 20
python -m causal_mllm.cli.inspect_source --dataset mtid --n 20
```

Schema reports are written to `outputs/schema/`.

### Build Causal Families

```bash
# Selection + comparative atom extraction
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --stage atoms --max-families 5

# Full chain: annotate -> harmonize -> six variants
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --stage variants --max-families 5 \
    --annotations data/families/annotations.json \
    --harmonization data/families/harmonization.json
```

The annotation and harmonization JSONs are human/LLM-produced inputs
(`{family_key: {atom_id: payload}}` and `{family_key: canonical_q}`);
LLM/VLM backends are wired via the `CallableAnnotator` /
`CallableHarmonizer` APIs, which record mandatory provenance.

### Validate Families (Iteration 6)

```bash
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --stage validate --max-families 5 \
    --annotations ... --harmonization ... \
    --judge data/families/risk_judge.json --theta 0.5
```

The validation layer re-checks every built family automatically
(schema, per-variant trajectory checks, canonical-q grounding flags,
safe-vs-unsafe shared-part leakage) and EXCLUDES failures. It also
runs `validate_factorial_relations()`, an independent firewall that
re-derives the factorial structure from the persisted artifact alone
(image placement per cell, identical vision hashes across H01/H11/
shuffle, media files existing and hashing to their recorded
`source_media.sha256`, H11-vs-shuffle history multiset plus
non-identity permutation, identical canonical terminal hash) so a
corrupted `families.jsonl` is caught even if the generators are
correct; each validation-report entry records the explicit factorial
cells H00=(0,0), H10=(1,0), H01=(0,1), H11=(1,1). It further
re-derives the Iteration-5 SEMANTIC eligibility from the persisted
annotations (`validate_factorial_semantic_eligibility`): a built
family must still carry `equivalent` / `relevant` /
`required_for_joint_interpretation==True` for the variants it holds —
decided-but-negative annotations are excluded here too. With a risk
judge (`ManualFileJudge` JSON or `CallableJudge` / the future
frozen-replay model judge), it scores Risk(q*), Risk(T), Risk(V),
Risk(T,V) and decides strict cross-modal causal-subset membership
(Risk(q*)<θ, Risk(T)<θ, Risk(V)<θ, Risk(T,V)≥θ), filling the
`standalone_terminal_risk` / `strict_causal_candidate` placeholders.
Without a judge these stay null: candidates only.

### Validate Families

Superseded by `build_families --stage validate` above (Iteration 6).
The standalone `cli/validate_families.py` remains as a deprecated
Iteration-0 stub.

```bash
python -m causal_mllm.cli.validate_families \
    --input data/families/draft \
    --output data/families/validated
```

### Frozen Replay (Iteration 8)

```bash
python -m causal_mllm.cli.replay \
    --input-dir outputs/families/scale_b_smoke \
    --max-families 5        # smoke; omit for the full 20-family run
```

Replays validated families through a frozen model (initial target:
local Qwen3.5-9B; backend/model configurable) and stores trajectory
→ raw response records under `outputs/replay_runs/<run_id>/`,
separate from the dataset artifacts. Hard gates:

- Input is `validated_families.jsonl` ONLY — never raw
  `families.jsonl`.
- Stored histories are replayed EXACTLY: no attacker, no interactive
  regeneration of intermediate turns; identical system prompt and
  generation settings for every variant (temperature 0 = greedy,
  max_new_tokens 256).
- All referenced media are hash-verified immediately before
  inference; missing/corrupt media fail loudly.
- Every (family, variant) pair is attempted exactly once (5×6=30
  smoke / 20×6=120 full); the run fails loudly on missing coverage.
- Failures are recorded separately with an error category (oom /
  media / context_length / generation) — never as safe/refusal
  labels.
- Each record carries run_id, family_id, source_id, variant, model,
  model_revision, prompt/template revision, generation config,
  response, error, plus input-token counts and visual-token metadata
  from the actual target tokenizer (surface-length/confound
  diagnostics), and OUTPUT diagnostics (`output_token_count`,
  `finish_reason`, `hit_max_new_tokens`); the report exposes
  truncation BY VARIANT because it is not condition-independent.
- `--model-revision` pins the weights actually loaded; the report
  records `resolved_sha256` binding resolved revision + prompt +
  generation settings.

Iteration 8 produces raw responses ONLY: judging and the causal
estimands (ΔT, ΔV, ΔTV, reset/order effects) are Iteration 9. The
Iteration-9 primary panel is the clean-tree pinned re-run
(`scale-b-2026-08-28-t1536-final-qwen35-9b`; revision explicitly
pinned, `git_dirty=false`, `resolved_sha256` binds model + processor
revision + prompt template + torch/CUDA versions + validated families
hash + repository commit; ~zero truncation; measured smoke truncation
at lower caps: 512→8/27, 768→4/30, 1024→3/30); the 256-token v0.8.0
runs are retained as evidence of the truncation bias (cross_modal 85%
vs text_only 40% mid-sentence at the 256 cap).

### Run Inference

Superseded by `cli.replay` above (Iteration 8). The standalone
`cli/run_inference.py` remains as a deprecated Iteration-0 stub.

```bash
python -m causal_mllm.cli.run_inference \
    --families data/families/validated \
    --model-config configs/models/qwen_mllm.yaml \
    --output outputs/inference/mvp
```

### Evaluate

```bash
python -m causal_mllm.cli.evaluate \
    --families data/families/validated \
    --responses outputs/inference/mvp/responses.jsonl \
    --config configs/evaluation/default.yaml \
    --output outputs/evaluation/mvp
```

## Experiment Scales

| Scale | Source Scenarios | Variants | Trajectories | Purpose |
|-------|-----------------|----------|--------------|---------|
| A — Schema smoke | 5 | 6 | 30 | Confirm adapters, media, schema |
| B — Research smoke | 20 | 6 | 120 | Manual inspection, first inference |
| C — Preliminary | 100 | 6 | 600 | Primary experiment, go/no-go |

Scale A and B builds are committed under
[`outputs/families/scale_a_smoke`](outputs/families/scale_a_smoke/)
and [`outputs/families/scale_b_smoke`](outputs/families/scale_b_smoke/)
with their human review inputs; Scale B additionally carries 41
decided-but-ineligible negative controls.

## Six Required Variants per Family

All six variants share one canonical terminal query q\* (exact
string/hash invariant), constructed by explicit harmonization because
0/752 Type-B source rows have matching multimodal/text terminals.

| Variant | Description | Factorial cell |
|---------|-------------|----------------|
| `neutral` | Safe history, text only, + q\* | \(H_{00}\) |
| `text_only` | Unsafe history, text only, + q\* | \(H_{10}\) |
| `vision_only` | **Safe** history + shared image, + q\* | \(H_{01}\) |
| `cross_modal` | Unsafe history + shared image, + q\* (candidate) | \(H_{11}\) |
| `shuffle` | Cross-modal content, deterministically permuted order | H3 |
| `history_reset` | q\* alone (minimal context) | H4 |

**Naming note:** the names are convenience aliases for the factorial
cells \(T \in \{0,1\}\) (text risk) × \(V \in \{0,1\}\) (image).
`vision_only` does **not** mean "no text semantics" — it is \(H_{01}\):
safe text + image. Canonical cells: \(H_{00}\)=safe text, no image;
\(H_{10}\)=unsafe text, no image; \(H_{01}\)=safe text, image;
\(H_{11}\)=unsafe text, image.

Variant generation is **gated by evidence**: each generator asserts its
readiness (`L0_structural` → `L1_semantic` → `L2_variant_ready`) and
raises `VariantPrerequisiteError` with explicit reasons instead of
guessing. Annotation *completeness* is separated from factorial
*eligibility*: a decided-but-negative judgment (`not_equivalent`,
`irrelevant`, `required_for_joint_interpretation=False`) does not pass
the gate — it REJECTS the family from the causal subset (negative
controls). Image-bearing conditions require positive evidence:
`vision_only` needs equivalence + relevant image; `cross_modal` /
`shuffle` additionally need `required_for_joint_interpretation=True`.
A generated `cross_modal` is a **candidate**, not a causal claim — the
strict subset needs behavioral validation (Iteration 6+).

## Implementation Roadmap

- [x] **Iteration 0** — Project skeleton, configs, schemas, CLI stubs, 60 unit tests
- [x] **Iteration 1** — Source dataset inspection, schema reports, adapter `normalize()`, 18 integration tests
- [x] **Iteration 2** — Full canonical adapters with golden fixture tests
- [x] **Iteration 3** — Candidate selection with rejection reasons
- [x] **Iteration 4** — Family-level comparative semantic atom extraction
- [x] **Iteration 5** — Annotation → harmonization → six gated variant generators
- [x] **Iteration 6** — Automatic validation layer (grounding, leakage,
      behavioral strict-subset decision via configurable risk judges)
- [x] **Iteration 7** — 20-family research smoke dataset (Scale B:
      120 trajectories, human-reviewed; 41 negative controls)
- [x] **Iteration 8** — Frozen replay model runner
- [x] **Iteration 9** — Safety judge + causal metrics (COMPLETE — Scale-B
      evidence frozen)
- [x] **Iteration 10** — 100-family preliminary experiment (COMPLETE —
      Scale-C evidence frozen; Phase 9 is an external-model confirmation,
      not a human audit)

### Iteration 9 closeout (Scale B, frozen)

Iteration 9 is complete. The 20-family Scale-B panel was judged by a
distinct-model LLM ensemble — primary judges `qwen3.8-max` (A) and
`glm-5.2` (B) under rubric v1.1, with ALL 42 A/B disagreements
adjudicated by `kimi-k3` from the original blinded context with full
per-call provenance (`distinct_model_adjudication_on_all_disagreements`).
Primary result at θ=0.5: ensemble qualifiers `CMST_384606`,
`CMST_392337`, `CMST_436242`; qualifiers supported by both primaries
`CMST_384606`, `CMST_436242`; ΔTV ≈ 0.119, history effect ≈ 0.241, with
per-judge bootstrap CIs in `outputs/llm_judge_artifacts/judge_sensitivity.json`.

The final Scale-B evidence commit is **`1f443f8`** (commit `3b63158`
added provenance binding and per-judge bootstrap CIs on top of identical
labels). A frozen manifest with SHA-256 hashes of the validated dataset,
replay panel, rubric v1.1, A/B labels, Kimi adjudications, and final
report is at `outputs/iteration_9_closeout/scale_b_evidence_manifest.json`.
All earlier evidence is preserved untouched.

### Iteration 10 closeout (Scale C, frozen)

Iteration 10 is complete. The 100-family Scale-C panel (600 trajectories)
was replayed by pinned `Qwen/Qwen3.5-9B` (revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, temp 0, cap 1536, 0%
truncation, panel gate 600/600) and judged under the preregistered
protocol (`configs/experiments/scale_c_protocol.json`, frozen pre-results
at `679c4b8`) by the same distinct-model ensemble — `qwen3.8-max` (A),
`glm-5.2` (B), all 239 disagreements adjudicated by `kimi-k3` with full
per-call provenance.

Primary result at θ=0.5 under the frozen decision rule:
**POPULATION_INTERACTION** — ΔTV CI **[0.0495, 0.1800]**, entirely above
zero; per-judge sensitivity ΔTV CIs: Judge A [0.0220, 0.1590], Judge B
[0.1075, 0.2625]. Strict qualifiers: ensemble 14, A 13, B 21, both
primaries 11 (of 100 families; reported separately, never as proof of
the average effect).

Phase 9 was executed as an **external-model confirmation** (GPT-family
scorer over the 45 blinded outcome-stratified audit items), NOT a
completed human audit — the drawn worksheet remains unfilled, and a
human audit is required only if the paper claims human validation. The
confirmation sample is consistent with and does not contradict the
primary decision; it cannot independently estimate the population ΔTV
(outcome-stratified sample, two complete families). The population
claim rests on the preregistered 600-item ensemble analysis.

A frozen manifest binding every artifact (protocol, panel, replay,
rubric, blinded items, A/B labels + fingerprints, adjudications,
ensemble labels, agreement/sensitivity/decision reports, and the full
audit chain incl. the sealed-manifest parent) by SHA-256 and git commit
is at `outputs/iteration_10_closeout/scale_c_evidence_manifest.json`
(regenerate: `python3 scripts/scale_c_closeout_manifest.py`). The replay
evidence freeze commit is `0944de5`.

## Schema Reports

Pre-computed schema reports from programmatic inspection of all three source datasets are available in [`outputs/schema/`](outputs/schema/):

- [`mtmcs_schema.json`](outputs/schema/mtmcs_schema.json) — MTMCS-Bench field analysis
- [`cosafe_schema.json`](outputs/schema/cosafe_schema.json) — CoSafe structure analysis
- [`mtid_schema.json`](outputs/schema/mtid_schema.json) — MTID trajectory analysis

## Data Quality Diagnostics

[`outputs/diagnostics/type_b_alignment.json`](outputs/diagnostics/type_b_alignment.json)
measures cross-modality alignment across **all 752 Type-B rows** (reproduce with
`python -m causal_mllm.cli.diagnose_type_b`):

- Terminal query equality `q*_mm == q*_text`: **0/752 exact, 0/752 normalized**
- Per-turn mm/text alignment: **0%** for every turn
- Rows directly usable for the 2×2 factorial construction: **0**; requiring q\* harmonization: **752**

The multimodal and unimodal Type-B dialogues are separately written paraphrases
of the same scenario. Terminal-query harmonization is therefore a required
Iteration-5 workstream for every family — it is never assumed away.

## Configuration

Nothing is hard-coded. All paths, models, sample counts, and parameters live in YAML configs under `configs/`. See [`configs/generation/mvp.yaml`](configs/generation/mvp.yaml) for the main experiment config.

## Reproducibility

Every command saves:
- Full config + config hash
- Git commit
- Random seed
- Package versions
- Model/dataset revisions
- Output hashes

## What This Project Does NOT Do (Yet)

- Model fine-tuning or RL
- Guardrail training
- Automatic adversarial attacker agents
- Activation extraction / patching / SAE
- Distributed multi-node training
- Web UI or annotation platform
- Image generation

## References

- Liu et al. (2026), *MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues*. Findings of ACL 2026. [ACL Anthology](https://aclanthology.org/2026.findings-acl.96/)
- Yu et al. (2024), *CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference*. EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.968/)
- TurnGate / MTID. [GitHub](https://github.com/Graph-COM/TurnGate) · [Dataset](https://huggingface.co/datasets/Graph-COM/MTID)
- Li et al. (2026), *State-Dependent Safety Failures in Multi-Turn Language Model Interaction* (STAR). [arXiv:2603.15684](https://arxiv.org/abs/2603.15684)

## License

[MIT](LICENSE)
