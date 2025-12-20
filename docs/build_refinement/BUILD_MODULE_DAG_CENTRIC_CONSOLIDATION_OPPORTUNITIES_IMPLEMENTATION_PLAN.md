# BUILD MODULE - DAG-CENTRIC CONSOLIDATION OPPORTUNITIES (IMPLEMENTATION PLAN)

## Context and goals
The build module has matured into a Hamilton-first, DAG-centric architecture, but several
capabilities are implemented multiple times across parallel modules. This plan consolidates
duplicated functionality into canonical services so we can:

- reduce drift between planning and execution
- harden schema and contract guarantees
- simplify extensibility for new targets, exports, and serving artifacts
- keep behavior stable while improving maintainability

This plan operationalizes the consolidation opportunities identified in the build review and
should be executed incrementally to preserve stability.

## Success criteria (definition of done)
- A single canonical schema service resolves TableSchema, Pandera schema, JSON Schema, row
  bindings, and digests with no conflicting sources of truth.
- Target metadata, dependencies, and output inventories are derived once and reused everywhere.
- Export and serving artifact pipelines use a single export engine and shared serialization
  utilities.
- Configuration resolution and options hashing are driven by one effective config stack.
- Planning and execution share one skip/hash evaluator and shared materializer base logic.
- Utility layers (lazy import, table key validation, tag normalization, error envelopes) are
  centralized and reused across modules.
- All quality gates pass:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q`

## Current status summary (as of this update)
- Core schema service, target metadata service, tag index, and export writer utilities exist.
- BuildRunContext and ExecutionPolicy are in place but are not yet the sole source of config and
  runtime policy across build/plan/execute.
- Materializer base helpers are extracted, but skip/hash evaluation is still split across
  planning, run records, and state computation.
- Legacy schema sources (TABLE_SCHEMAS and contract providers) remain in use and must be removed
  once ContractService and schema inference are fully consolidated.
- Export/serving pipelines are partially consolidated; Hamilton export targets still re-implement
  parts of the export flow, and serving artifacts are not yet fully routed through a single
  compiler.
- Quality gates are not yet green; remaining lint/docstring/type parity issues must be resolved
  after consolidation changes are finalized.

## Scope inventory (all items in this plan)

### Schema and contracts
- SCHEMA-1: Unify schema source of truth (TableSchema provider vs Pandera DatasetSchema).
- SCHEMA-2: Collapse contract metadata duplication (OutputContract, DatasetContract, DatasetSchema).
- SCHEMA-3: Centralize schema inference across provider_hamilton and schemas/compile.
- SCHEMA-4: Consolidate JSON Schema generation and digest computation.
- SCHEMA-5: Standardize row bindings and column order; remove hand-maintained column tuples.

### Targets and DAG
- TD-1: Consolidate target metadata construction into one TargetMetadata service.
- TD-2: Unify output inventory derivation (introspect, native/outputs, buildspec).
- TD-3: Route target spec creation through canonical schema provider (avoid TABLE_SCHEMAS drift).
- TD-4: Centralize Hamilton tag discovery and parsing in a shared TagIndex.
- TD-5: Consolidate target lookup by table/artifact using TargetSystem indexes.

### Exports and serving
- EX-1: Consolidate export pipelines (exports engine vs Hamilton-native export targets).
- EX-2: Centralize JSON serialization and streaming loops for export formats.
- EX-3: Use a single schema-manifest compiler for serving and CLI export paths.
- EX-4: Consolidate semantic registry compilation and tag discovery.
- EX-5: Unify manifest and marker models across exports, serving, and schema manifests.

### Config and options
- CO-1: Unify BuildConfig, BuildRunConfig, and execution options into BuildRunContext.
- CO-2: Standardize target option parsing (from_parameters or typed config layer).
- CO-3: Consolidate TargetExecution and run-level execution options into one policy model.
- CO-4: Centralize default parameter definitions and eliminate divergent defaults.
- CO-5: Ensure options hash reflects the effective config stack (profiles + overrides).

### Execution and hashing
- EH-1: Extract a shared materializer base for DuckDB and artifact savers.
- EH-2: Centralize skip/hash evaluation for planning and execution.
- EH-3: De-duplicate options hash computation logic.
- EH-4: Unify schema and asset fingerprint semantics across subsystems.
- EH-5: Consolidate expected output references and inventory (datasets/artifacts).

### Infra and utilities
- IU-1: Replace scattered lazy import patterns with a shared loader or module re-org.
- IU-2: Centralize table key parsing and validation.
- IU-3: Centralize tag value normalization.
- IU-4: Unify BuildError and ExportError into a single error envelope.

## Constraints and guiding principles
- Preserve behavior and data output semantics unless explicitly improved.
- Ship changes in small, reviewable increments with compatibility shims.
- Keep Hamilton DAG as the authoritative source for dependencies and I/O.
- Prefer immutable, typed service boundaries to avoid drift and side effects.

## Plan overview (phases)
Each phase lists deliverables, scope coverage, and acceptance gates.

### Phase 0 - Inventory, design, and guardrails
Deliverables:
- A lightweight consolidation map doc listing all Scope IDs and current owners.
- Architecture tests (report-only initially) that detect duplicate schema sources, target metadata
  duplication, and export pipeline divergence.
- Baseline parity tests for schemas, buildspec, exports, and serving artifacts.

Scope coverage:
- Foundation for all scope items.

Acceptance gates:
- Guardrail tests compile and run locally (can be report-only until later phases).
- Baseline snapshots recorded for schema/buildspec/export outputs.

### Phase 1 - Canonical schema and contract services
Deliverables:
- A canonical SchemaService that resolves:
  - TableSchema
  - Pandera schema
  - JSON Schema and digest
  - row model and binding metadata
- A ContractService that exposes output contracts and dataset contracts from a single source.
- A unified SchemaInferenceService that supports:
  - single table inference
  - batch inference
  - caching and deterministic error policy
- A ColumnOrder/RowBinding helper that derives ordered columns from the canonical schema.

Scope coverage:
- SCHEMA-1, SCHEMA-2, SCHEMA-3, SCHEMA-4, SCHEMA-5

Implementation steps:
1) Define SchemaService interfaces and adapters that wrap:
   - existing TableSchema registry
   - Pandera DatasetSchema registry
   - JSON Schema generators
2) Add an internal canonical schema record type that aggregates all schema forms and hashes.
3) Route schema consumers to the new service via adapters:
   - `codeintel.build.schemas.registry`
   - `codeintel.build.schemas.json_schema_registry`
   - `codeintel.build.schemas.row_registry`
4) Replace manual column tuple usage in build targets with schema-derived column order helpers.
5) Add compatibility shims to preserve old provider APIs.

Status update:
- Completed: SchemaService exists in `codeintel.core.schemas`, build wiring in
  `codeintel.build.schemas.service`, and adapters in registry/json/row registries.
- Remaining: ContractService not implemented; SchemaInferenceService is still split between
  provider_hamilton and schemas/compile. Column order helpers are not yet centralized. TABLE_SCHEMAS
  and contract provider shims still active.

Acceptance gates:
- Schema resolution parity checks pass (TableSchema, JSON Schema, row bindings).
- Export validation results unchanged for a representative dataset set.

### Phase 2 - Target metadata, DAG inventory, and tag index consolidation
Deliverables:
- A TargetMetadataService that constructs:
  - TargetGraph
  - TargetSystem indexes (by name, table, artifact)
  - Output inventories (datasets and artifacts)
- A shared TagIndex for Hamilton tags (targets, datasets, views, semantic tags).
- A single OutputInventoryService used by:
  - buildspec compilation
  - expected outputs
  - schema manifests
  - asset tracking

Scope coverage:
- TD-1, TD-2, TD-3, TD-4, TD-5, EH-5

Implementation steps:
1) Create TargetMetadataService with deterministic build ordering and caching.
2) Migrate `target_system`, `introspect`, and `spec/compile` to use this service.
3) Implement TagIndex that:
   - standardizes tag normalization
   - provides discovery for targets and semantic views
4) Replace direct TABLE_SCHEMAS use in target specs with SchemaService lookups.
5) Deprecate redundant lookup helpers in provider_unified and contract_provider.

Status update:
- Completed: TargetMetadataService and TagIndex exist. Buildspec compilation uses the metadata
  service. Output inventory is available as a service.
- Remaining: Direct `load_target_system()` usage still exists in contract and enforcement paths.
  OutputInventory is not yet authoritative in expected output helpers. TABLE_SCHEMAS is still used
  in target spec helpers and declared schema provider.

Acceptance gates:
- TargetGraph/TargetSystem parity with existing behavior for all targets.
- Buildspec output hash matches baseline for a representative repo.

### Phase 3 - Export and serving consolidation
Deliverables:
- A single ExportEngine with format-specific adapters for JSONL and Parquet.
- Shared serializer and streaming writer utilities.
- A single SchemaManifest compiler used by serving and CLI pipelines.
- A unified semantic registry compiler that uses TagIndex.
- Shared manifest/marker base classes with deterministic serialization.

Scope coverage:
- EX-1, EX-2, EX-3, EX-4, EX-5

Implementation steps:
1) Extract a shared export pipeline that all export entry points call.
2) Move JSON serializer and record streaming into common utilities.
3) Update Hamilton export targets to call ExportEngine (not re-implement logic).
4) Replace serving artifact schema manifest generation with the shared compiler.
5) Consolidate semantic registry compilation to use TagIndex and SchemaService.
6) Introduce a ManifestBase helper for export and serving metadata serialization.

Status update:
- Completed: Export engine exists, shared writers utility exists, schema manifest compiler is
  shared and used by serving artifacts.
- Remaining: Hamilton export targets still implement their own export flow. Semantic registry
  compilation still uses its own tag parsing helpers and does not consistently reuse TagIndex
  across all entry points. Export/serving manifest types remain duplicated.

Acceptance gates:
- Export artifacts are byte-identical (or semantically equivalent) for a known snapshot.
- Serving artifacts are stable and deterministic across runs.

### Phase 4 - Config and options unification
Deliverables:
- BuildRunContext that merges:
  - BuildConfig (TOML)
  - BuildRunConfig (profiles and overrides)
  - BuildExecutionOptions (runtime behavior)
- A consistent options loading system with:
  - `from_parameters` support
  - typed validation helpers
- A single execution policy model covering resource and runtime hints.

Scope coverage:
- CO-1, CO-2, CO-3, CO-4, CO-5

Implementation steps:
1) Define BuildRunContext and map current config usage to it.
2) Update options hashing to use the effective config stack.
3) Replace custom option parsers with typed config helpers where feasible.
4) Align TargetResources and TargetExecution with run-level options.
5) Remove redundant default parameter definitions.

Status update:
- Completed: BuildRunContext exists and ExecutionPolicy is defined.
- Remaining: Options parsing is still split between typed helpers and per-target logic.
  BuildRunContext is not yet the single entry point for all build flows. Default parameters and
  profile overlays still live in multiple places. ExecutionPolicy is not yet used for scheduling.

Acceptance gates:
- Options hash stability verified across old and new stacks.
- Target option loading produces identical values for existing configs.

### Phase 5 - Execution, hashing, and materializers
Deliverables:
- A shared materializer base for:
  - DuckDB Ibis saver
  - DuckDB rows saver
  - File artifact saver
- A unified skip and hash evaluator used by:
  - StateComputer
  - native run records
- A consolidated fingerprint service for schema and asset versioning.

Scope coverage:
- EH-1, EH-2, EH-3, EH-4

Implementation steps:
1) Extract a core materializer workflow with pluggable write strategies.
2) Replace per-materializer skip/hash logic with the shared evaluator.
3) Consolidate options hash calls into a single helper.
4) Align schema hash usage across asset fingerprints and buildspec datasets.

Status update:
- Completed: Shared materializer base helpers exist and are used by DuckDB and artifact savers.
- Remaining: Skip/hash evaluation is still spread across planning, run records, and state
  computation. Options hashing is still invoked via multiple helpers. Fingerprinting alignment is
  still incomplete.

Acceptance gates:
- Skip behavior identical in state computation and execution.
- Materializer error handling unchanged (status and metadata parity).

### Phase 6 - Infra and error modeling cleanup
Deliverables:
- A shared LazyImport helper (or re-org) to reduce circular import patterns.
- Central table key validation utilities used by all entry points.
- Shared tag normalization utilities used by TagIndex and semantic registry.
- A unified error envelope that replaces ExportError with BuildError or ProblemDetail.

Scope coverage:
- IU-1, IU-2, IU-3, IU-4

Implementation steps:
1) Introduce a lazy-loader utility and migrate repeated patterns to it.
2) Replace ad-hoc table key validation with a single helper.
3) Standardize tag normalization in TagIndex and semantic pipelines.
4) Replace ExportError with a consistent ProblemDetail-based error type.

Status update:
- Completed: Shared lazy import helpers exist and are used in build and schemas.
  Tag normalization is centralized in TagIndex, but not all semantic compilers use it.
  BuildProblemError exists as a unified problem envelope.
- Remaining: Central table-key validation is not yet wired everywhere. ExportError is still the
  primary export-facing type and should be collapsed into BuildProblemError. Semantic compilation
  still has tag parsing logic outside TagIndex.

Acceptance gates:
- Error outputs are consistent and structured across build and export flows.
- No new circular import issues are introduced.

## Dependency and sequencing notes
- Phase 1 (SchemaService) should precede Phase 2 and Phase 3 to avoid repeated schema migrations.
- Phase 2 (TargetMetadataService and TagIndex) enables Phase 3 (exports/serving) to reuse inventories.
- Phase 4 (BuildRunContext) should be in place before Phase 5 (skip/hash unification) to ensure
  hashing uses the effective config stack.

## Migration and compatibility strategy
- Add compatibility shims that preserve existing public APIs during migration.
- Use deprecation warnings with a removal window of at least one release cycle.
- Maintain deterministic outputs for buildspec, schema manifests, and exports until explicitly
  planned to change.

## Testing and validation plan
- Unit tests for new services (SchemaService, TargetMetadataService, ExportEngine).
- Parity tests that compare:
  - table schemas and JSON schemas
  - buildspec hash
  - export artifact checksums
  - serving artifact manifests
- Integration tests that exercise a representative snapshot end-to-end.

## Risk register and mitigations
- Risk: schema drift during migration
  - Mitigation: parity tests and dual-write/dual-read adapters during Phase 1.
- Risk: target inventory mismatch affecting buildspec or asset tracking
  - Mitigation: compare derived inventories to current outputs in Phase 2.
- Risk: export behavior divergence
  - Mitigation: byte-level diff of export outputs in Phase 3.
- Risk: configuration stack mismatch affecting options hash
  - Mitigation: hash parity checks in Phase 4, controlled rollout.

## Rollout strategy
- Deliver in incremental PRs per phase with explicit acceptance gates.
- Keep feature flags for any behavioral changes until parity is confirmed.
- Update documentation and playbooks after each phase completes.

## Open questions (to resolve before implementation)
Resolved decisions:
- SchemaService lives under `codeintel.core.schemas`.
- BuildRunContext is a factory that produces BuildEnv + execution options.
- Unified error envelope is a BuildError subclass using ProblemDetail.

Remaining decisions:
- ContractService final API and ownership (core vs build).
- Canonical manifest base class location and serialization format (exports/serving/schemas).

## Scope ID mapping (status, remaining work, decommission targets)
Each scope ID includes its status, the intended final shape, and legacy/compat code to remove
once the scope item is complete.

### Schema and contracts
- SCHEMA-1 (partial): SchemaService exists and is wired to build adapters. Finalize all consumers
  to use SchemaService records (table schema, dataset schema, JSON schema, row binding).
  Decommission: `codeintel.config.datasets.declared_schemas.TABLE_SCHEMAS` usage in
  `codeintel.build.schemas.provider_declared`, `codeintel.build.hamilton.native.target_spec_helpers`.
- SCHEMA-2 (remaining): ContractService not yet implemented. Final design should expose
  OutputContract + DatasetContract views via a single service.
  Decommission: `codeintel.build.schemas.contract_provider` once ContractService is canonical.
- SCHEMA-3 (partial): Inference still split across provider_hamilton and schemas/compile. Final
  service should expose batch + per-table inference and deterministic fallback semantics.
  Decommission: inference entry points in `codeintel.build.schemas.provider_hamilton`.
- SCHEMA-4 (partial): JSON schema/digest flows routed through SchemaService, but not all callers
  use the canonical digest API. Finalize callers and remove redundant helpers if any appear.
- SCHEMA-5 (remaining): Column order/row binding helpers not centralized. Finalize a helper that
  derives ordered columns from SchemaService records and replace manual column tuples.
  Decommission: any manual column tuple definitions in target specs.

### Targets and DAG
- TD-1 (partial): TargetMetadataService exists. Remaining is to route all target system access
  through it and eliminate ad-hoc `load_target_system()` callers.
  Decommission: direct `load_target_system()` usage in `codeintel.build.schemas.contract_provider`,
  `codeintel.build.hamilton.contracts.enforcement`, and schema plugin constraints.
- TD-2 (partial): OutputInventory exists, but is not authoritative in all expected outputs and
  buildspec paths. Finalize output inventory usage everywhere.
  Decommission: any direct contract table_key lists where inventory should be canonical.
- TD-3 (remaining): Target specs still resolve schemas from TABLE_SCHEMAS. Replace with
  SchemaService lookups via canonical provider.
  Decommission: TABLE_SCHEMAS usage in `codeintel.build.hamilton.native.target_spec_helpers`.
- TD-4 (partial): TagIndex exists and normalizes tags. Ensure semantic registry compilation and
  view discovery always use TagIndex.
  Decommission: tag parsing helpers that bypass TagIndex in serving compilation.
- TD-5 (partial): Target lookup by table/artifact exists on TargetSystem. Ensure all lookup paths
  route through TargetMetadataService for consistency.

### Exports and serving
- EX-1 (partial): ExportEngine exists. Hamilton export targets still implement custom flow.
  Decommission: export logic in `codeintel.build.hamilton.native.export.export_targets` once
  it is routed through ExportEngine.
- EX-2 (completed): Shared serializer and writer utilities exist in `codeintel.build.exports.writers`.
  Remaining is to make all export paths use the shared utilities consistently.
- EX-3 (partial): SchemaManifest compiler is shared; serving artifacts use it. Ensure CLI/export
  paths are also routed through the shared compiler.
- EX-4 (partial): Semantic registry compilation still uses bespoke tag parsing. Finalize TagIndex
  usage and SchemaService-backed column resolution.
  Decommission: ad-hoc tag parsing in `codeintel.build.serving.semantic_compile`.
- EX-5 (remaining): Manifest/marker models remain duplicated across exports, serving, and schema
  manifests. Introduce a ManifestBase and migrate current models to it.
  Decommission: duplicate manifest dataclasses in `codeintel.build.exports.manifest`,
  `codeintel.build.serving.manifest`, `codeintel.build.schemas.manifest`.

### Config and options
- CO-1 (partial): BuildRunContext exists. Migrate all entry points to use it as the canonical
  factory for BuildEnv and execution options.
  Decommission: ad-hoc BuildEnv construction outside BuildRunContext.
- CO-2 (remaining): Standardize target option parsing via typed options helpers. Replace custom
  parsing in target modules.
- CO-3 (partial): ExecutionPolicy exists but is not yet integrated into scheduling/execution.
  Finalize usage across plan/execute.
- CO-4 (remaining): Default parameters are still duplicated; centralize them and remove divergent
  defaults.
- CO-5 (partial): Options hash reflects effective config in some flows; standardize across plan,
  execution, and state computation.

### Execution and hashing
- EH-1 (completed): Shared materializer base exists and is used by DuckDB and artifact savers.
- EH-2 (remaining): Skip/hash evaluation still split across plan/run/state. Introduce a shared
  evaluator and route all flows through it.
- EH-3 (partial): Options hash helper exists, but multiple call sites compute hashes differently.
  Standardize on a single helper and remove local re-implementations.
- EH-4 (remaining): Fingerprint semantics are still split between assets and schemas. Align
  schema hash usage across asset fingerprints and buildspec datasets.
- EH-5 (partial): Output inventory exists but is not authoritative everywhere. Route expected
  outputs and asset tracking through the inventory service.

### Infra and utilities
- IU-1 (completed): Lazy import helper exists and is used across build and schemas.
- IU-2 (remaining): Table key parsing/validation is still ad-hoc in some modules. Introduce a
  single helper and update callers.
- IU-3 (partial): Tag normalization exists in TagIndex; ensure all semantic tag pipelines use it.
- IU-4 (partial): BuildProblemError exists, but ExportError is still used directly. Replace
  ExportError with the unified BuildProblemError envelope and remove export-specific error types.
