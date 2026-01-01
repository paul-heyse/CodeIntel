# Schema Consolidation Implementation Plan (Design Phase, Sharp Cutover)

## Summary

This plan consolidates schema resolution, Arrow metadata handling, observation
emission, and validation into a single coherent architecture. It targets a
sharp cutover with explicit deletion of legacy code paths, aligning with the
inference-first Hamilton + PyArrow design outlined in
`docs/hamilton_inference_first_implementation_plan.md`. The work is organized
into design-lock and implementation phases with strict acceptance gates.

## Goals

- One authoritative schema resolution path: observed -> override -> declared.
- One Arrow metadata codec for schema-level and field-level contract metadata.
- One observation emission pipeline for all materializers.
- One validation engine that covers Arrow tables, readers, Parquet, and JSONL.
- Hamilton + PyArrow first-class execution boundaries (RecordBatchReader and
  dataset scanners), with no schema authority at serve time beyond DuckDB.
- Sharp cutover with decommissioning and deletion of legacy helpers and
  duplicative providers.

## Non-goals

- Incremental migration with long-lived dual paths or feature flags.
- Schema pinning, version gating, or backward-compat enforcement beyond
  observed schema visibility.
- Large-scale behavior changes unrelated to schema consolidation.

## Guiding Principles

- Observed schemas are the primary contract for inferable outputs.
- Hints are additive only and must not override observed types.
- Streaming-first always (RecordBatchReader, scan_batches).
- Parquet datasets are the boundary between build and serving.
- Sharp cutover: one authoritative code path, legacy code deleted.

## Scope and Boundaries

- Build-time: observation emission, schema registry persistence, Arrow metadata.
- Runtime: schema resolution, alignment, validation, and serving inventory.
- Serving: DuckDB catalog and metadata tables as sole schema sources.

## Target Architecture (Post-Cutover)

Hamilton outputs -> RecordBatchReader -> Observation pipeline -> Registry
-> Arrow schema metadata -> Parquet datasets -> DuckDB import
-> SQLGlot planning uses DuckDB catalog only

## Consolidation Workstreams

### 1) Canonical Schema Resolution

Problem: Multiple helper functions load schemas and observations with
inconsistent precedence and error handling.

Design outcome:
- One resolver that implements the authority chain:
  observed -> override -> declared.
- Single API for consumers across build, storage, serving, exports.

Proposed new module (design):
- `src/codeintel/core/schemas/resolution.py`
  - `SchemaResolutionResult` (schema, source, derivation, observation)
  - `resolve_schema(table_key, *, gateway=None, allow_inference=True)`
  - `resolve_observation(table_key, *, gateway=None)`
  - `resolve_arrow_schema(table_key, *, metadata_policy=...)`

Consumers to migrate:
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/storage/schema/registry_provider.py`
- `src/codeintel/storage/validation/columnar.py`
- `src/codeintel/serving/semantic/inventory.py`

Deliverables:
- Resolver module + docs for authority chain.
- All consumers depend on the resolver, not direct storage helpers.

Acceptance:
- No remaining calls to `gateway.schemas.load_table_schema` outside resolver.
- All schema resolution tests pass with observed-precedence semantics.

### 2) Arrow Metadata Codec

Problem: Metadata encoding and extras policy handling are duplicated across
core and build layers.

Design outcome:
- A single codec for Arrow schema/field metadata:
  - encode/decode `codeintel.*` metadata keys
  - validate metadata types
  - attach provenance and schema hashes

Proposed new module (design):
- `src/codeintel/core/columnar/schema_metadata.py`
  - `SchemaMetadataCodec`
  - `encode_schema_metadata(...)`
  - `decode_schema_metadata(...)`
  - `merge_field_metadata(...)`

Consumers to migrate:
- `src/codeintel/core/columnar/schema_alignment.py`
- `src/codeintel/core/schemas/arrow_gen.py`
- `src/codeintel/core/schemas/arrow_polars.py`
- `src/codeintel/build/schemas/observations.py`

Deliverables:
- Single encoder/decoder used in all Arrow boundary code.
- Metadata validation centralized and unit tested.

Acceptance:
- No direct metadata dict reads outside codec module.
- Arrow schema metadata keys in outputs match plan contract.

### 3) Observation Emission Pipeline

Problem: Arrow dataset saver and DuckDB relation saver duplicate observation
logic.

Design outcome:
- Unified observation pipeline with explicit inputs:
  schema hints, derived settings, drift summary, registry persistence.

Proposed new module (design):
- `src/codeintel/build/schemas/observation_pipeline.py`
  - `observe_reader(...) -> SchemaObservationBundle`
  - `persist_observation_bundle(...)` (existing logic moved here)
  - `annotate_arrow_schema(...)` (via metadata codec)

Consumers to migrate:
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`

Deliverables:
- Materializers call the same observation pipeline.
- One place to compute stats, extras policy, and derived settings.

Acceptance:
- Observation records are identical across materializer types.
- No duplicate observation logic remains in saver modules.

### 4) Unified Validation Engine

Problem: Export validation and runtime Arrow validation compute constraints
independently.

Design outcome:
- A shared constraint engine that can:
  - derive constraints from observed stats
  - apply them to Arrow tables, readers, Parquet files, and JSONL
  - respect extras policy from Arrow metadata

Proposed new module (design):
- `src/codeintel/core/validation/schema_constraints.py`
  - `ConstraintSet` (schema, types, nullability, stats)
  - `constraints_from_schema(...)`
  - `constraints_from_observation(...)`
  - `validate_table(...)`
  - `validate_reader(...)`
  - `validate_parquet_path(...)`

Consumers to migrate:
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/storage/validation/columnar.py`

Deliverables:
- Single validation engine with consistent error messaging.
- Export validation becomes a thin adapter around the core engine.

Acceptance:
- One set of validation tests covers both export and runtime paths.
- No duplicate constraint derivation code remains.

### 5) Provider Chain Consolidation

Problem: Schema providers are split across build, storage, and serving with
partial overlap.

Design outcome:
- One SchemaAuthority / SchemaService entrypoint with explicit observed
  precedence, used everywhere.
- Storage and serving use the same provider without importing build-only code.

Proposed changes (design):
- `SchemaService` becomes the single gateway for table and Arrow schemas.
- `RegistrySchemaProvider` becomes an implementation detail of resolver.
- Storage provider facade delegates to `SchemaService` or resolver.

Consumers to migrate:
- `src/codeintel/storage/contracts/schema_provider.py`
- `src/codeintel/serving/semantic/inventory.py`
- `src/codeintel/config/datasets/contracts.py`

Deliverables:
- Storage and serving no longer depend on registry provider directly.
- Registry provider only used inside resolver.

Acceptance:
- `get_schema_provider()` always uses the canonical service path.
- No provider duplication in storage or serving layers.

### 6) Hamilton + PyArrow Alignment Utilities

Problem: Seeding, scanning, and inference utilities are split and partially
duplicate the same Arrow dataset behaviors.

Design outcome:
- Shared dataset scanning utilities that are streaming-first and schema-aware.
- Clear boundary utilities for RecordBatchReader alignment and projection.

Proposed new module (design):
- `src/codeintel/core/columnar/dataset_scanner.py`
  - `scan_dataset_reader(...)`
  - `project_reader(...)`
  - `empty_reader_from_schema(...)`

Consumers to migrate:
- `src/codeintel/build/schemas/seed_harness.py`
- `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`
- `src/codeintel/core/columnar/schema_alignment.py`

Deliverables:
- One scanner path for seed harness and ingestion.
- Streaming-first behavior guaranteed in all inference paths.

Acceptance:
- No seed harness path materializes full tables for inference.
- q__ inputs use dataset scanners with projection-only reads.

## Phased Implementation Plan (Sharp Cutover)

### Phase 0: Design Lock (Current Phase)

Goals:
- Finalize the consolidation design and module boundaries.
- Produce API sketches and dependency graphs.
- Identify exact legacy deletions.

Tasks:
- Write API sketches for new modules listed above.
- Map all call sites for schema resolution, metadata, observations, validation.
- Confirm schema authority chain and metadata keys with existing plan.
- Draft deletion list and update plan docs.

Deliverables:
- Design document with API signatures and module boundaries.
- Call-site inventory with migration targets.
- Deletion manifest (files and functions).

Acceptance:
- Design reviewed and approved.
- Migration targets and deletion list are unambiguous.

### Phase 1: Implement Canonical Modules

Goals:
- Build the new canonical modules in isolation.

Tasks:
- Implement schema resolver.
- Implement Arrow metadata codec.
- Implement observation pipeline.
- Implement unified validation engine.
- Implement dataset scanner utilities.

Deliverables:
- New core/build modules with unit tests.

Acceptance:
- Ruff, pyright, pyrefly clean for new modules.
- Unit tests cover: resolver precedence, metadata encoding, stats constraints.

### Phase 2: Single Cutover of All Consumers

Goals:
- Switch all call sites to the new canonical modules in one change set.

Tasks:
- Replace resolution logic in analytics, exports, storage, serving.
- Update materializers to use observation pipeline.
- Update validation paths to use unified engine.
- Update seed harness and ingestion utils to use dataset scanner.

Deliverables:
- One commit series that removes all legacy use from call sites.

Acceptance:
- No remaining imports of legacy helpers in migrated files.
- All affected integration tests pass.

### Phase 3: Decommission and Delete Legacy Code

Goals:
- Delete legacy modules and helpers now unused.

Tasks:
- Remove deprecated helpers and duplicated logic.
- Remove unused providers and registry access paths.
- Update documentation to reference new modules only.

Deliverables:
- Legacy code deleted, references removed, docs updated.

Acceptance:
- `rg` shows no references to deleted APIs.
- No dead code or unused modules remain.

### Phase 4: Alignment Verification

Goals:
- Validate the new architecture against the inference-first plan.

Tasks:
- Re-run schema inference end-to-end with streaming only.
- Validate Parquet metadata and DuckDB catalog import.
- Verify no registry calls are made at serving query time.

Deliverables:
- Validation report and updated plan status.

Acceptance:
- Observed schemas drive serving inventory and validation.
- All schema consumers reference the canonical resolver.

## Legacy Decommission List (Initial Draft)

The following are slated for deletion or reduction after cutover:

- `_load_inferred_schema` and `_load_latest_observation` helpers in
  `src/codeintel/build/analytics/utilities/datasets.py`.
- `_get_inferred_schema` and `_get_latest_observation` in
  `src/codeintel/build/exports/validation.py`.
- Direct registry access paths in
  `src/codeintel/storage/schema/registry_provider.py` (to be internalized).
- Duplicate metadata parsing in
  `src/codeintel/core/columnar/schema_alignment.py` and
  `src/codeintel/core/schemas/arrow_polars.py`.
- Duplicated observation logic in
  `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py` and
  `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`.
- Duplicate constraint derivation logic in
  `src/codeintel/build/exports/validation.py` and
  `src/codeintel/storage/validation/columnar.py`.
- Seed harness scan logic in `src/codeintel/build/schemas/seed_harness.py`
  that materializes tables (replaced by dataset scanner).

Note: the final deletion manifest will be refined in Phase 0 based on the
call-site inventory.

## Testing and Quality Gates

- Run: `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted tests for build + serving, then segmented pytest runs.
- Add unit tests for resolver precedence, metadata codec, observation pipeline,
  and validation engine.
- Add an integration test for streaming inference that asserts no full table
  materialization.

## Risks and Mitigations

- Risk: schema resolution changes affect serving inventory.
  Mitigation: resolver integration tests and serving snapshot tests.
- Risk: metadata codec breaks existing Parquet metadata expectations.
  Mitigation: read/write round-trip tests with real datasets.
- Risk: sharp cutover causes missed call sites.
  Mitigation: exhaustive `rg` inventory and one-shot migration checklist.

## Open Questions (Design Phase)

- Which tables remain explicit overrides vs inferred relations after cutover?
- What is the authoritative source for view schemas in Phase 8 of the larger
  inference plan?
- Should observation-derived constraints be applied at serving-time only or
  also during build-time validation?

