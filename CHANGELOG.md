# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.1] — 2026-08-23

### Fixed — Selection report granularity

- The selection report's single `reason_counts` was record-level, so one
  rejected MTMCS family counted four times. Replaced with both
  `rejected_records_by_reason` and `rejected_families_by_reason` (plus
  `n_families_rejected`) so family-level reports are not misleading

### Added — Balance reporting

- Selection report now includes `accepted_by_category`, `accepted_by_label`,
  `families_by_category`, and `families_by_safety` distributions
- `balance_warnings`: flags extreme concentration (>= 80% of >= 5 families)
  in a single source category or safety category. Report + warn only — no
  stratified resampling yet. MTMCS pairs are safe+unsafe mixed by
  construction and never trigger the safety warning
- `build_families --stage select` logs family-level rejection counts,
  category/safety distributions, and balance warnings

### Added — GitHub Actions CI

- `.github/workflows/ci.yml`: `unit` job (offline: unit tests, offline golden
  fixtures, ruff on Iteration 3+ modules) and `integration` job (real
  datasets). Test evidence is now attached to commits independently of local
  runs

## [0.3.0] — 2026-08-23

### Added — Iteration 3: Candidate Selection with Rejection Reasons

- Group-aware candidate selection (`construction/select.py`): MTMCS records are
  selected atomically per `pair_id` — all four conditions accepted together or
  all four rejected together
- Causal invariant checks during selection: type_b groups must share the
  terminal query across safe/unsafe (`terminal_query_invariant_violated`),
  type_a groups must diverge (`terminal_query_not_divergent`)
- Machine-readable rejection manifest (`SelectionRejection`) with stable reason
  codes: dataset_excluded, setting_excluded, too_few_turns, too_many_turns,
  text_too_long, terminal_query_too_short, no_images, group_incomplete,
  not_sampled
- Accounting invariant enforced: `n_input == n_accepted + n_rejected`, asserted
  before any result is returned
- Deterministic seeded family sampling (`max_families` + `seed`); unsampled
  eligible units are recorded as `not_sampled`, never silently dropped
- Pass-through guarantee: selection never mutates source records — accepted
  examples are the identical input objects (source trajectory ≠ experimental
  frozen trajectory; no synthetic assistant responses at this stage)
- Selection pipeline stage (`construction/pipeline.py`): adapter →
  normalization with rejection recording → selection → `candidates.jsonl`,
  `normalization_rejections.jsonl`, `selection_rejections.jsonl`,
  `selection_report.json` (with config hash, git commit, reason counts)
- Adapter factory `causal_mllm.adapters.get_adapter()`
- `build_families --stage select` CLI implemented, with `--max-rows` and
  `--max-families` overrides
- `SelectionConfig.from_config()` for YAML-driven criteria (rejects unknown keys)
- 42 new tests (31 unit + 11 integration): criteria, atomicity, invariants,
  accounting, determinism, pass-through, artifact persistence, CLI end-to-end

### Changed

- `configs/generation/mvp.yaml`: corrected MTMCS split (`type_b`, the causal
  gold standard), added `selection` section, `source.max_rows` replaces the
  record-level limit

## [0.2.0] — 2026-08-23

### Added — Iteration 2: Canonical Adapters with Golden Fixture Tests

- Golden fixture JSONL files pinned from real source rows (MTMCS 28, CoSafe 5,
  MTID 5) with terminal query hashes
- Enhanced canonical validator (`validate_canonical_example`) with 9 structural
  and semantic checks
- Schema-change guards (`schema_guards.json`) failing loudly on upstream drift
- Type guard (`assert_canonical`) ensuring the family builder consumes only
  `CanonicalSourceExample`

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
