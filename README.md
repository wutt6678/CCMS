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
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --max-families 5
```

### Validate Families

```bash
python -m causal_mllm.cli.validate_families \
    --input data/families/draft \
    --output data/families/validated
```

### Run Inference

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

## Six Required Variants per Family

| Variant | Description |
|---------|-------------|
| `neutral` | Topic-matched neutral history |
| `text_only` | All relevant semantics in text |
| `vision_only` | Relevant evidence in images |
| `cross_modal` | Complementary text + vision (main treatment) |
| `shuffle` | Same content, permuted order |
| `history_reset` | Terminal query with minimal context |

## Implementation Roadmap

- [x] **Iteration 0** — Project skeleton, configs, schemas, CLI stubs, 60 unit tests
- [x] **Iteration 1** — Source dataset inspection, schema reports, adapter `normalize()`, 18 integration tests
- [x] **Iteration 2** — Full canonical adapters with golden fixture tests
- [x] **Iteration 3** — Candidate selection with rejection reasons
- [ ] **Iteration 4** — Semantic atom extraction
- [ ] **Iteration 5** — Six independent variant generators
- [ ] **Iteration 6** — Automatic validation layer
- [ ] **Iteration 7** — 20-family research smoke dataset
- [ ] **Iteration 8** — Frozen replay model runner
- [ ] **Iteration 9** — Safety judge + causal metrics
- [ ] **Iteration 10** — 100-family preliminary experiment

## Schema Reports

Pre-computed schema reports from programmatic inspection of all three source datasets are available in [`outputs/schema/`](outputs/schema/):

- [`mtmcs_schema.json`](outputs/schema/mtmcs_schema.json) — MTMCS-Bench field analysis
- [`cosafe_schema.json`](outputs/schema/cosafe_schema.json) — CoSafe structure analysis
- [`mtid_schema.json`](outputs/schema/mtid_schema.json) — MTID trajectory analysis

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
