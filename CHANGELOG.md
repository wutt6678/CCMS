# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-23

### Added — Iteration 0: Project Skeleton

- Full Python package structure under `src/causal_mllm/`
- `pyproject.toml` with editable install support
- Typed dataset schemas (`data/schemas.py`): `CanonicalSourceExample`, `CausalFamily`, `SemanticAtom`, `VariantData`, `TerminalQuery`, `InferenceOutput`, `SafetyJudgeLabel`
- JSONL read/write helpers and YAML config loading (`data/io.py`)
- Schema validation with named checks (`data/validate_schema.py`)
- Structured logging (`data/logging.py`)
- Deterministic seed utilities: family ID generation, SHA-256 hashing, config hashing (`seeds.py`)
- CLI entry points: `inspect_source`, `build_families`, `validate_families`, `run_inference`, `evaluate`
- YAML configurations for 3 datasets, generation, models, and evaluation
- Synthetic test fixture with one text turn and one image reference
- 60 unit tests: package imports, JSONL roundtrip, config loading, schema validation, seed determinism, CLI `--help`, media path validation
- Shell scripts: `smoke_source.sh`, `smoke_build.sh`, `smoke_inference.sh`, `preliminary_100.sh`

### Added — Iteration 1: Source Dataset Inspection

- Programmatic schema inspection for all three source datasets
- **MTMCS-Bench** adapter (`adapters/mtmcs.py`): loads ND-25/MCS-bench, normalizes multimodal dialogue into canonical schema with image saving
- **CoSafe** adapter (`adapters/cosafe.py`): loads from cloned GitHub repo, normalizes text-only dialogues
- **MTID** adapter (`adapters/mtid.py`): downloads JSONL via `huggingface_hub` (workaround for `Json` feature type incompatibility), normalizes trajectory conversations
- Machine-readable schema reports in `outputs/schema/` (mtmcs, cosafe, mtid)
- 18 integration tests verifying source ID retention, message order, terminal query, image references, label survival, and schema validation on real data
- Comprehensive dataset suitability analysis documenting Path A/B/C recommendations

### Dataset Statistics

| Dataset | Records | Modalities | Key Feature |
|---------|---------|------------|-------------|
| MTMCS-Bench | 752 × 2 splits | Image + text | Escalation-based safe/unsafe pairs |
| CoSafe | ~1,400 | Text only | Coreference-based risk |
| MTID | ~16,000 | Text only | Closure-turn labels |
