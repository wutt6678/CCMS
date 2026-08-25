# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] — 2026-08-25

### Added — Iteration 7: 20-family research smoke dataset (Scale B)

- Human review extended to rows 0-60 of MTMCS Type-B: all four
  condition dialogues read per row, images inspected, cross-modality
  equivalence judged per atom, one deictic-free canonical q* per
  eligible family. Review inputs committed under
  `outputs/families/scale_b_smoke/review/` with the generator
  (`scripts/gen_scale_b_review.py`) as provenance.
- 20 factorially eligible families (rows 0,3,4,6,8,17,20,22,23,34,36,
  40,42,44,47,52,55,57,59,60) x 6 variants = 120 trajectories; all
  pass the Iteration-6 automatic validation with zero exclusions.
- 41 reviewed rows stored as DECIDED negative controls
  (`not_equivalent` terminals or shared turns), keeping the causal
  subset clean while documenting why most of MTMCS Type-B cannot
  serve as modality counterfactuals.
- `configs/generation/scale_b.yaml` (max_rows=61 cuts exactly at the
  20th eligible family); pinned evidence tests for the committed
  Scale-B build (20 families, 120 trajectories flag-free, 41 decided
  controls disjoint from the research set).
- Strict causal subset remains empty until the frozen-replay model
  judge (Iteration 8+).

## [0.6.0] — 2026-08-25

### Added — Iteration 6: automatic validation layer

- `validation/judges.py`: `RiskJudge` interface with `ManualFileJudge`
  (JSON {family_key: {variant: score}}) and `CallableJudge` (any
  LLM/VLM/model callable; mandatory model provenance). Scores are
  validated fail-loud in [0, 1]; missing required scores raise
  `JudgeError`. The frozen-replay model judge (Iteration 8+) plugs in
  through the same interface.
- `validation/stage.py`: `run_validation_stage()` re-checks every
  built family automatically — full schema, per-variant trajectory
  checks, canonical-q grounding flags, and safe-vs-unsafe shared-part
  leakage (the `safe_vs_unsafe_shared_parts` axis is auto-resolved at
  confidence 1.0 for shared atoms; the causal atom and the vision
  atom's image channel are compared on the right content). Failures
  are EXCLUDED (excluded_families.jsonl).
- Behavioral strict-subset decision with a judge:
  Risk(q*)<θ and Risk(T)<θ and Risk(V)<θ and Risk(T,V)>=θ fills the
  Iteration-3 placeholders (`standalone_terminal_risk`,
  `strict_causal_candidate`, `requires_standalone_risk_validation`)
  and upgrades `cross_modal_candidate` -> `cross_modal_required`.
  Without a judge these stay null: candidates only.
- CLI: `--stage validate`, `--judge`, `--theta`.
- Evidence: the committed human-reviewed Scale-A build validates 5/5
  automatically with zero exclusions; strict subset empty until a
  behavioral judge exists. Placeholder builds are fully excluded by
  the grounding checks (integration test).
- 330+ tests (227 unit + integration).

## [0.5.2] — 2026-08-25

### Added — persistent canonical-q grounding validator

- `construction/grounding.py`: `flag_grounding_issues(family)` scans
  the canonical q* and every text-only condition (neutral / text_only /
  history_reset) for image-deictic references ("these", "this photo",
  "shown here", ...) that no image can resolve in those conditions,
  and flags built families whose grounding validation targets are
  still null.
- Pinned evidence tests: the committed human-reviewed Scale-A build
  (5 families x 6 variants) must be flag-free with human/manual
  harmonization provenance; negative controls must stay disjoint from
  the research set.
- Integration test proving the flagger catches the placeholder
  harmonization (verbatim mm terminal with "these helmets") —
  placeholder evidence remains mechanically separable from
  research-valid evidence.
- 314 tests (213 unit + 101 integration).

## [0.5.1] — 2026-08-25

### Fixed — eligibility gates confused with annotation completeness

- P0-1: `not_equivalent` no longer passes the equivalence gate.
  `ANNOTATED_*` states (a decision exists) are now separated from
  `FACTORIAL_*` eligibility (`equivalent` only). Constructing a
  modality counterfactual from an explicit S(T_mm)!~S(T_text)
  judgment raises `VariantPrerequisiteError` with a 'belongs with
  negative controls' reason.
- P0-2: `irrelevant` no longer passes the visual-relevance gate, and
  `required_for_joint_interpretation=False` now blocks cross_modal /
  shuffle. Requirements table: neutral/text_only/history_reset =
  canonical q* only; vision_only = + equivalence + relevant;
  cross_modal/shuffle = + joint-interpretation True.
- vision_only now requires the equivalence gate (the H00->H01
  contrast changes both image presence and text wording).
- `run_variants_stage` routes decided-but-negative (or unresolved)
  families to `negative_controls.jsonl` with explicit reasons instead
  of failing the whole stage; per-family generators still fail loudly.

### Added

- Canonical-q grounding VALIDATION TARGETS on the harmonization block
  (`canonical_q_grounding_valid`,
  `canonical_q_no_unintended_modality_dependency`,
  `canonical_q_semantically_preserves_mm_source`,
  `canonical_q_semantically_preserves_text_source`), null until
  human/Iteration-6 review; `ManualHarmonizer` accepts a dict entry
  carrying the reviewer's judgments.
- Naming documented: variant names are aliases for factorial cells
  H00/H10/H01/H11; vision_only = safe text + image, NOT 'no text'.
- Negative regression tests for not_equivalent / irrelevant /
  joint=False, and for stage-level negative-control routing.

### Evidence — Scale-A rebuilt from human review

- `outputs/families/scale_a_smoke/review/` commits the human review
  inputs: per-atom annotations and canonical q* with grounding
  judgments for the first 10 Type-B rows (all four dialogues + images
  read by the reviewer).
- Result: 5 eligible families (000000/000003/000004/000006/000008) x
  6 variants = 30 trajectories; 5 rows routed to negative controls
  with explicit not_equivalent reasons. Canonical q* re-phrased to be
  resolvable without the image (no deictic references), removing the
  reference-resolvability confound in text_only/history_reset.
- Research-data smoke evidence: PASS for the five built families
  (caveats in review/REVIEW_NOTES.md).

## [0.5.0] — 2026-08-25

### Added — Iteration 5: annotate → harmonize → construct variants

Iteration 4 measured 0/752 Type-B rows with cross-modality terminal
equality, so Iteration 5 does NOT map multimodal→cross_modal and
unimodal→text_only directly (that would confound modality with
wording). Instead the pipeline now runs three internal stages:

**5A — annotation resolution**
- `run_annotation_stage()` integrates annotators into the pipeline
  (CLI `--stage annotate` with `--annotations` JSON for manual
  review; `CallableAnnotator` for LLM/VLM backends).
- Minimum-required annotation gate: only what variant generators
  actually consume must be resolved (causal atom semantic_type,
  vision risk_relevance + required_for_joint_interpretation,
  mm/text equivalence) — no exhaustive annotation mandate.
- Mandatory `annotation_provenance` on llm-backed annotations
  (backend, model, model_revision, prompt_version, temperature,
  seed); validator rejects llm payloads without it. 'An LLM did it'
  is not provenance.

**5B — terminal-query harmonization**
- `construction/harmonize.py`: `ManualHarmonizer` (JSON) and
  `CallableHarmonizer` (LLM, mandatory model provenance) construct
  one canonical q* per family.
- `validation.terminal_harmonization = {required, canonical_q,
  canonical_sha256, source_mm_q, source_text_q, method, validation,
  provenance}` — additive; original skeleton terminal queries are
  never overwritten. Missing required harmonizations fail loudly.

**5C — six gated variant generators**
- `construction/variants.py`: neutral (H00), text_only (H10),
  vision_only (H01), cross_modal (H11), shuffle (deterministic
  permutation), history_reset — each builds from resolved atom
  surface forms, never raw MTMCS field names; each carries
  rule-generator provenance with transformation lists.
- Readiness levels L0_structural / L1_semantic / L2_variant_ready
  (`construction/readiness.py`) and `assert_variant_ready()` raising
  `VariantPrerequisiteError` with explicit reasons. Gates are
  per-variant: cross_modal requires resolved equivalence AND risk
  relevance; history_reset is mostly structural.
- Exact canonical-q hash invariant: all six variants end with the
  identical q* string/sha256, enforced by
  `validate_variant_trajectory()`.
- `cross_modal_candidate = true` is recorded; strict cross-modal
  causality (Risk(T)<θ, Risk(V)<θ, Risk(T,V)>=θ) remains a behavioral
  question for Iteration 6+.

### Evidence

- Scale-A smoke build on 5 real MTMCS Type-B families: 5 families ×
  6 variants = 30 trajectories, all passing structural checks with
  exact canonical-q hash invariance; every family correctly flagged
  `requires_terminal_harmonization: true`.
- Unresolved semantics fail loudly on real data: text_only /
  vision_only / cross_modal raise VariantPrerequisiteError while
  structural variants remain constructible.
- 47 new tests (203 unit + 99 integration = 302 total).

## [0.4.2] — 2026-08-24

### Added — Semantic annotation scaffolding (Iteration-4 review, round 2)

- P0-1 cross-modal semantic equivalence: atoms carry
  `semantic_equivalence` with axes `multimodal_vs_unimodal` and
  `safe_vs_unsafe_shared_parts`, defaulting to 'pending'. S(T_mm) ~
  S(T_text) must be ESTABLISHED by annotation before differently worded
  surface forms are treated as modality counterfactuals. Annotated form:
  `{state: equivalent, confidence: 0.94}`.
- P0-2 visual risk relevance: atoms carry `risk_relevance`
  (pending|relevant|irrelevant|uncertain) and
  `required_for_joint_interpretation` (null until evidenced), separating
  'image is present' from 'image supplies information required to
  interpret the risky trajectory'. The strict subset eventually needs
  Risk(T)<θ, Risk(V)<θ, Risk(T,V)>=θ (Iteration 6 behaviorally).
- Annotation module (`construction/annotation.py`):
  `ManualFileAnnotator` (JSON keyed by family_key -> atom_id, for the
  manual check of the first 10-20 families) and `CallableAnnotator`
  (wraps any LLM/VLM callable, recorded as semantic_validation='llm').
  Annotation payloads are validated fail-loud (`AnnotationError`) and
  applied to copies — skeletons are never mutated.

### Changed

- P0-3 atom_type is now an EXACT ALIAS of semantic_type, validator-
  enforced. The extractor no longer emits entity_or_scene/intent
  guesses for vision/terminal atoms — structure lives in
  `structural_role`, meaning stays 'unknown' until annotated. Iteration
  5 must read `semantic_type`.
- P1 skeleton validation now REQUIRES non-null `sha256` in every
  `source_media` entry — missing media never yields a valid skeleton.
  Test factories gained `image_path` for pointing at real hashable files.
- 18 new tests (173 unit + 91 integration = 264 total)

## [0.4.1] — 2026-08-24

### Added — Iteration-4 review: four-condition atoms, terminal alignment,
structure/meaning separation, media references

- Atoms now record ALL FOUR condition surface forms per turn
  (`surface_forms`: multimodal_safe/unimodal_.../... each {text, images}).
  The MTMCS mm and text dialogues are separately written fields; their
  equivalence is now explicit data, never assumed. `safe_text`/
  `unsafe_text` retained as the multimodal pair convenience.
- Cross-modality terminal-query alignment diagnostics on every family:
  `terminal_alignment` (mm_safe_vs_mm_unsafe, text_safe_vs_text_unsafe,
  multimodal_vs_unimodal) and `requires_terminal_harmonization` in the
  skeleton ground truth. The factorial design needs one q* across
  neutral/text_only/vision_only/cross_modal.
- Structure vs meaning separated in `SemanticAtom`: `structural_role`
  (divergent_history_turn, shared_history_turn, terminal_query,
  shared_image, assistant_context) vs `semantic_type` (default
  'unknown'), `semantic_description` (null) and `semantic_validation`
  ('pending'). Semantic type is NEVER inferred from turn position —
  an opening divergence may encode intent, relation, constraint,
  reference, attribute/state, or scene framing.
- Vision atoms carry explicit `source_media` ({path, sha256}) so
  Iteration 5 never infers which image an atom refers to.
- New diagnostic: `python -m causal_mllm.cli.diagnose_type_b` with pure
  `diagnose_type_b_rows()` (7 unit tests). Reports within-condition
  fixed-q validity, mm/text terminal equality (exact + normalized),
  per-turn alignment, and directly-usable vs rewrite-needed counts.
- Skeleton validator extended: structural_role required and valid,
  semantic_validation state valid, causal atoms need all four surface
  forms, vision atoms need source_media paths.

### Diagnostic result — all 752 Type-B rows

`outputs/diagnostics/type_b_alignment.json`:

- n_type_b = 752 (all complete)
- Cross-modality terminal q* equality: **0/752 exact, 0/752 normalized**
- Per-turn mm/text alignment (safe_r1, unsafe_r1, r2, r3): **0%** each
- Directly usable for 2x2 construction: **0**; requiring rewriting: **752**

Conclusion: the multimodal and unimodal Type-B dialogues are separately
written paraphrases on the same scenario. q* harmonization is a required
workstream for Iteration 5 on every family (no row is directly usable).

## [0.4.0] — 2026-08-23

### Added — Iteration 4: Family-Level Comparative Atom Extraction

- Core architectural decision: atoms are extracted at the FAMILY level,
  not per record. A Type-B family is decomposed comparatively —
  H_safe vs H_unsafe with shared q* — so extraction identifies which
  semantic content differs CAUSALLY between the histories
- `construction/atoms.py`: deterministic rule-based extractor producing
  `shared` atoms (identical across conditions), `causal` atoms (differing
  turns, carrying both `safe_text`/`unsafe_text` surface forms), and
  `not_applicable` atoms for singletons without a safe/unsafe pair
- Integrity enforced loudly: divergent-turn sets must agree between the
  multimodal and text-only pairs, image paths must match across
  conditions, and turns must align 1:1 (`AtomExtractionError`)
- `construction/families.py`: family skeletons binding deterministic
  family_id + atoms + invariant q* (sha256) + ground-truth provenance
  (divergent turns, condition labels, causal atom IDs); variants remain
  empty until Iteration 5; standalone-risk placeholders carried forward
- `SemanticAtom` gains `divergence`, `safe_text`, `unsafe_text` fields
- `validate_family_skeleton()`: hash integrity, unique atom IDs, valid
  types/divergences, and MTMCS families MUST contain >= 1 causal atom
- `run_atoms_stage()` + `build_families --stage atoms`; persists
  `family_skeletons.jsonl` and `atoms_report.json`
- 36 new tests (32 unit + 4 integration), incl. real-data checks that
  each type_b family has exactly one causal divergence at turn 0

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
  runs. Integration results uploaded as a JUnit XML artifact
- Pinned `datasets>=4.2,<5.0`: 5.x is an untested breaking major release

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
