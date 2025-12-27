# Build Consolidation Plan (DAG-First Unification)

## Purpose
Create a single, authoritative, DAG-first build architecture by consolidating schema
specification, target metadata, execution/materialization, and validation into shared
APIs and common data classes. This plan targets `src/codeintel/build` as the primary
scope and aligns with advanced Hamilton capabilities (decorators, caching, runtime
builder, hooks, and telemetry).

## Guiding Principles
- DAG is the single source of truth for outputs, schemas, and dependencies.
- All schema and contract decisions flow through one authority.
- Execution, materialization, and validation share consistent state, metadata, and
  error surfaces.
- Use Hamilton modifiers and DriverBuilder to reduce custom orchestration code.
- Prefer composable data classes and minimal glue over duplicated utilities.

## Consolidation Outcomes (End State)
- One SchemaAuthority that owns table schemas, contract overlays, and derivations.
- One TargetCatalog/BuildSpec derived from the DAG, reused by CLI, storage, and serving.
- One execution pipeline that standardizes skip/missing/failure semantics.
- One materialization metadata format for tables and artifacts.
- One validation report format used across build, exports, storage, and serving.
- One row/JSON serialization path, with JSON native in DuckDB and encoded only at
  export boundaries.

## Codebase-Wide Consolidation Scope
This plan now targets shared architecture across `src/codeintel/build`,
`src/codeintel/cli`, `src/codeintel/storage`, and `src/codeintel/serving`.

## Hamilton Advanced Features Integration (By Phase)
Each phase should explicitly leverage Hamilton advanced capabilities to reduce
custom orchestration and improve observability.

- **Phase 1 (SchemaAuthority)**: use DriverBuilder to extract DAG schema
  lineage; use `@check_output_custom` to enforce schema alignment at the
  materialization boundary.
- **Phase 2 (TargetCatalog)**: use DriverBuilder + TagSpec to build the catalog
  and propagate `TargetIO` metadata; use `@parameterize` to remove repeated
  target patterns.
- **Phase 3 (Execution)**: use Hamilton lifecycle hooks for start/end tracking;
  align caching decisions with Hamilton’s caching metadata for deterministic
  skip behavior.
- **Phase 4 (Materialization)**: standardize saver patterns via shared
  decorators/modifiers (e.g., `@save_rows`, `@save_artifact`) and emit
  MaterializationResult uniformly.
- **Phase 5 (Validation)**: push contract checks into Hamilton modifiers
  (`@check_output_custom`) to avoid ad-hoc validation logic.
- **Phase 6 (Serialization)**: consolidate row encoding/decoding at DAG
  boundaries; prefer Hamilton modifiers for pre/post-processing.
- **Phase 7 (Cleanup)**: remove legacy orchestration paths and align logging
  and telemetry hooks to Hamilton’s structured logging patterns.

### Cross-Cutting Consolidations
1) **SchemaAuthority (global schema + contract source)**
   - Central module: `src/codeintel/core/schemas/authority.py`.
   - Build: replace `src/codeintel/build/schemas/*` and
     `src/codeintel/build/hamilton/contracts/schemas/*` with SchemaAuthority.
   - Storage: validation and view schemas in `src/codeintel/storage/validation/*`
     and `src/codeintel/storage/views/*` must resolve via SchemaAuthority.
   - Serving: semantic compile and serving manifests resolve via SchemaAuthority.
   - CLI: dataset commands resolve table schemas via SchemaAuthority.

2) **TargetCatalog / BuildSpec (global DAG inventory)**
   - Central module: `src/codeintel/core/targets/catalog.py`.
   - Build: `src/codeintel/build/target_metadata.py` and `src/codeintel/build/spec/*`
     become serialization views of TargetCatalog.
   - CLI: `build.targets` and registry commands use TargetCatalog directly.
   - Storage/Serving: metadata tables and manifests derive from TargetCatalog.

3) **ExecutionOutcome + MaterializationResult (global run metadata)**
   - Central module: `src/codeintel/core/execution/*`.
   - Build: unify `src/codeintel/build/hamilton/executor.py`,
     `src/codeintel/build/hamilton/native/executor.py`, and
     `src/codeintel/build/hamilton/run_records.py`.
   - Storage: tracking tables consume ExecutionOutcome/MaterializationResult.
   - Serving: publishing and semantic compile consume the same metadata.
   - CLI: build status/run output renders from ExecutionOutcome.

4) **ValidationReport (global validation surface)**
   - Central module: `src/codeintel/core/validation/*`.
   - Build: consolidate `src/codeintel/build/hamilton/validators/*` and
     `src/codeintel/build/exports/validation.py` into a single validator.
   - Storage: conformance checks use ValidationReport.
   - Serving/CLI: export and semantic validation surfaces use ValidationReport.

5) **Row/JSON Codec (global serialization path)**
   - Central module: `src/codeintel/core/serialization/row_codec.py`.
   - Build: `src/codeintel/build/hamilton/native/ibis_helpers.py` and
     `src/codeintel/build/hamilton/io/*` use RowCodec.
   - Storage: insert/upsert helpers use RowCodec.
   - Serving/Exports: JSON encoding only at export boundaries.

6) **Optional Inputs + Skip Semantics (global policy)**
   - Target optional inputs live in TargetCatalog.TargetIO.
   - Remove local optional input registries and use a shared policy module.
   - CLI status and serving manifests report skipped/blocked based on TargetIO.

7) **Tool Registry (global tooling inventory)**
   - Central module: `src/codeintel/core/tools/*`.
   - CLI tooling listings, ingestion runners, and build preflight share the same
     tool definitions and aliasing (e.g., scip-python).

8) **Observability Hooks (global instrumentation contract)**
   - One observability policy and hook surface for build, CLI, and serving.
   - Hamilton lifecycle hooks emit standardized execution metadata.

## Detailed Scope Specifications (Per Consolidation Item)

### 1) SchemaAuthority (global schema + contract source)
**Phase-by-phase file map**
- Phase 1: add `src/codeintel/core/schemas/authority.py`.
- Phase 1: update `src/codeintel/build/schemas/provider_unified.py`,
  `src/codeintel/build/schemas/service.py`, and
  `src/codeintel/build/hamilton/contracts/schemas/*` to delegate to SchemaAuthority.
- Phase 5: replace validation schema sources in `src/codeintel/build/exports/validation.py`
  and `src/codeintel/build/hamilton/validators/*` to use SchemaAuthority.
- Phase 7: delete redundant build schema registry and contract service modules.

**New data class APIs**
- `SchemaAuthority`: `get(table_key)`, `require(table_key)`, `iter()`,
  `derivation(table_key)`.
- `SchemaDerivation`: `table_key`, `source_kind`, `source_ref`, `schema_hash`.

**DAG integration steps**
- DriverBuilder builds the DAG and TagSpec, then SchemaAuthority derives schemas
  from DAG outputs and TagIndex.
- `@check_output_custom` enforces that materialized tables match SchemaAuthority.

**Migration steps per subsystem**
- Build: replace schema lookup in `src/codeintel/build/schemas/*` and
  `src/codeintel/build/hamilton/contracts/schemas/*`.
- CLI: update dataset commands to resolve schemas via SchemaAuthority.
- Storage: update `src/codeintel/storage/validation/*` and `src/codeintel/storage/views/*`
  to use SchemaAuthority.
- Serving: update `src/codeintel/build/serving/*` and `src/codeintel/serving/*`
  to resolve schemas via SchemaAuthority.

**Acceptance criteria**
- DAG-derived schema overrides declared schema when both exist.
- Declared schema is used only when DAG schema is absent.
- Schema derivation lineage is visible for every output table.

**Test checkpoints**
- Schema snapshot tests (contract counts and table schemas).
- Build validation tests that assert DAG schema precedence.

**Data compatibility rules**
- Precedence order: DAG schema > declared schema > seed schema.
- Any mismatch between materialized schema and SchemaAuthority is a validation error.

**Execution semantics**
- If SchemaAuthority lacks a schema for a produced table, target fails unless
  explicitly marked as seed or declared-only.

### 2) TargetCatalog / BuildSpec (global DAG inventory)
**Phase-by-phase file map**
- Phase 2: add `src/codeintel/core/targets/catalog.py`.
- Phase 2: update `src/codeintel/build/target_metadata.py` and
  `src/codeintel/build/spec/*` to serialize TargetCatalog.
- Phase 2: regenerate `src/codeintel/core/registry/dag_output_inventory.yaml`
  from TargetCatalog.
- Phase 7: remove any remaining inventory drift or duplicate target specs.

**New data class APIs**
- `TargetCatalog`: `targets()`, `get(name)`, `by_table_key(table_key)`.
- `TargetSpec`: `name`, `domain`, `impl_kind`, `io`, `tags`.
- `TargetIO`: `outputs`, `artifacts`, `optional_inputs`, `required_inputs`.

**DAG integration steps**
- DriverBuilder compiles TargetCatalog using TagSpec and DAG outputs.
- TargetIO is derived from DAG edges and declared output contracts.

**Migration steps per subsystem**
- Build: `src/codeintel/build/target_metadata.py` becomes a wrapper.
- CLI: `build.targets` and registry commands pull from TargetCatalog.
- Storage: metadata tables use TargetCatalog for target listings.
- Serving: manifests and publish flows read TargetCatalog.

**Acceptance criteria**
- TargetCatalog matches DAG outputs (table keys and artifacts).
- No command reads the YAML inventory directly.

**Test checkpoints**
- CLI `build.targets` and registry output tests.
- Target catalog snapshot tests (counts and target names).

**Data compatibility rules**
- TargetCatalog is the single source of truth for targets and outputs.

**Execution semantics**
- Optional inputs and output tables are resolved only from TargetCatalog.TargetIO.

### 3) ExecutionOutcome + MaterializationResult (global run metadata)
**Phase-by-phase file map**
- Phase 3: add `src/codeintel/core/execution/*` and merge executor logic into it.
- Phase 4: add `src/codeintel/core/execution/materialization.py`.
- Phase 4: update `src/codeintel/build/hamilton/materializers/*` and
  `src/codeintel/build/hamilton/native/materialization_records.py`.
- Phase 7: remove duplicate per-target run record constructors.

**New data class APIs**
- `ExecutionOutcome`: `status`, `reason`, `row_counts`, `artifacts`, `duration_ms`.
- `MaterializationResult`: `status`, `table_key`, `row_count`, `artifact_name`,
  `path`, `error`.

**DAG integration steps**
- Hamilton lifecycle hooks write ExecutionOutcome and materialization metadata.
- Caching decisions are driven by Hamilton cache metadata and input hashes.

**Migration steps per subsystem**
- Build: replace `run_records.py` with ExecutionOutcome-based recording.
- CLI: build status/run output from ExecutionOutcome.
- Storage: tracking tables ingest MaterializationResult.
- Serving: publish and manifest flows read ExecutionOutcome and materializations.

**Acceptance criteria**
- All targets emit consistent status, row counts, and artifacts.
- Skip/failed outcomes are represented uniformly across build, CLI, and serving.

**Test checkpoints**
- Build status JSON structure tests.
- Run tracking tests for row_counts and artifacts.

**Data compatibility rules**
- ExecutionOutcome is the canonical status payload for targets.
- MaterializationResult is the canonical output record for tables/artifacts.

**Execution semantics**
- `succeeded`: all required outputs materialized.
- `skipped`: optional inputs missing or cached skip, with explicit reason.
- `failed`: required input missing or tool error, with error message recorded.

### 4) ValidationReport (global validation surface)
**Phase-by-phase file map**
- Phase 5: add `src/codeintel/core/validation/report.py`.
- Phase 5: consolidate `src/codeintel/build/hamilton/validators/*` and
  `src/codeintel/build/exports/validation.py`.
- Phase 5: update storage conformance checks and serving validation.

**New data class APIs**
- `ValidationReport`: `status`, `issues`, `summary`.
- `ValidationIssue`: `scope`, `table_key`, `target`, `severity`, `message`, `details`.

**DAG integration steps**
- Use `@check_output_custom` to emit ValidationReport from materializers.

**Migration steps per subsystem**
- Build: all validation paths emit ValidationReport.
- CLI: datasets and exports render ValidationReport for errors/warnings.
- Storage: conformance checks emit ValidationReport.
- Serving: export validation consumes ValidationReport directly.

**Acceptance criteria**
- Validation issues are reported uniformly in build, CLI, storage, and serving.
- Export validation does not re-implement schema checks.

**Test checkpoints**
- Export validation tests (JSONL and parquet).
- Storage conformance tests.

**Data compatibility rules**
- ValidationReport is the only validation payload persisted or rendered.

**Execution semantics**
- Validation errors set target status to failed unless configured as warn-only.

### 5) Row/JSON Codec (global serialization path)
**Phase-by-phase file map**
- Phase 6: add `src/codeintel/core/serialization/row_codec.py`.
- Phase 6: update `src/codeintel/build/hamilton/native/ibis_helpers.py` and
  `src/codeintel/build/hamilton/io/*` to use RowCodec.
- Phase 6: update `src/codeintel/storage/duckdb_policy_backend.py` and helpers.
- Phase 6: update `src/codeintel/build/exports/*` to encode JSON only at export.

**New data class APIs**
- `RowCodec`: `encode_row`, `decode_row`, `normalize_value`.
- `JsonBoundaryPolicy`: `encode_on_export: bool`, `native_in_db: bool`.

**DAG integration steps**
- Hamilton modifiers use RowCodec as pre/post processing for row outputs.

**Migration steps per subsystem**
- Build: Ibis helpers and materializers call RowCodec.
- Storage: insert/upsert helpers call RowCodec for normalization and row_hash.
- Serving: export encoders perform JSON serialization at the boundary.
- CLI: dataset snapshots render JSON via export encoders only.

**Acceptance criteria**
- JSON columns stored as native JSON in DuckDB.
- JSON encoding occurs only at export boundaries.
- row_hash is injected consistently when required.

**Test checkpoints**
- Ingestion row serialization tests.
- Module index tests that require row_hash.

**Data compatibility rules**
- JSON values are stored as dict/list in DB and encoded only on export.

**Execution semantics**
- RowCodec must not silently coerce schema types outside JSON normalization.

### 6) Optional Inputs + Skip Semantics (global policy)
**Phase-by-phase file map**
- Phase 3: move optional input registry into TargetIO.
- Phase 3: update `src/codeintel/build/hamilton/executor.py` to use TargetIO.
- Phase 7: remove `src/codeintel/build/hamilton/optional_inputs.py`.

**New data class APIs**
- TargetIO fields `optional_inputs` and `required_inputs` are authoritative.

**DAG integration steps**
- TargetIO derived from DAG edges + declared optional input annotations.

**Migration steps per subsystem**
- Build: preflight uses TargetIO optional inputs.
- CLI: status output reports skipped/blocked based on TargetIO.
- Serving: manifest includes skipped/blocked reasoning from TargetIO.

**Acceptance criteria**
- Missing optional inputs => skipped, not success.
- Missing required inputs => failed/blocked, with clear reason.

**Test checkpoints**
- Graph target skip propagation tests.
- Docs view and graph validation tests.

**Data compatibility rules**
- Optional/required inputs are defined in TargetCatalog only.

**Execution semantics**
- Optional missing => skipped with reason.
- Required missing => failed and dependent targets blocked.

### 7) Tool Registry (global tooling inventory)
**Phase-by-phase file map**
- Phase 2: normalize tool registry to `src/codeintel/core/tools/*`.
- Phase 2: align tooling inventory in `src/codeintel/core/registry/*`.
- Phase 7: remove any tool-specific duplicates elsewhere.

**New data class APIs**
- `ToolSpec`: `name`, `aliases`, `config_key`, `version_probe`, `required_by`.
- `ToolResolution`: `resolved_path`, `origin`, `status`.

**DAG integration steps**
- Target resources reference tools by canonical name only.

**Migration steps per subsystem**
- Build: preflight tooling checks use ToolRegistry.
- CLI: `registry.tools` uses ToolRegistry and aliasing rules.
- Ingestion: runner uses ToolRegistry for binary resolution.
- Serving: preflight tool checks use ToolRegistry.

**Acceptance criteria**
- Alias resolution is consistent (e.g., scip-python).
- No tool names are hard-coded outside ToolRegistry.

**Test checkpoints**
- Tool resolution tests (missing tools, alias mapping).

**Data compatibility rules**
- Tool names and aliases are defined in a single registry.

**Execution semantics**
- Missing required tool yields failed preflight with structured error.

### 8) Observability Hooks (global instrumentation contract)
**Phase-by-phase file map**
- Phase 3: unify execution telemetry in `src/codeintel/build/hamilton/observability.py`.
- Phase 5: update validation to emit observability signals.
- Phase 7: remove legacy telemetry hooks.

**New data class APIs**
- `ExecutionTelemetry`: `run_id`, `target`, `status`, `duration_ms`, `attributes`.

**DAG integration steps**
- Use Hamilton lifecycle hooks to emit standardized telemetry for every target.

**Migration steps per subsystem**
- Build: emit ExecutionTelemetry for targets and materializations.
- CLI: surface telemetry summaries in status/run output when enabled.
- Serving: publish telemetry for build and serving snapshot operations.

**Acceptance criteria**
- Telemetry emission is consistent across build, CLI, and serving.
- All targets emit a standardized telemetry envelope.

**Test checkpoints**
- Observability smoke tests for span emission.
- Runtime manager telemetry tests.

**Data compatibility rules**
- Observability uses a single attribute policy and normalization rules.

**Execution semantics**
- Telemetry is emitted for succeeded, skipped, and failed targets with status.

## New Shared Data Classes (Draft API)
These become the canonical API surface. Each should live in a shared build module
and be imported elsewhere instead of local copies.

1) SchemaAuthority
- `SchemaAuthority.get(table_key) -> TableSchema | None`
- `SchemaAuthority.require(table_key) -> TableSchema`
- `SchemaAuthority.iter() -> Iterable[TableSchema]`
- `SchemaAuthority.derivation(table_key) -> SchemaDerivation | None`

2) SchemaDerivation
- `table_key: str`
- `source_kind: Literal["dag", "declared", "seed"]`
- `source_ref: str` (node/target name or registry ref)
- `schema_hash: str`

3) TargetSpec / TargetIO
- `TargetSpec.name: str`
- `TargetSpec.domain: str`
- `TargetSpec.impl_kind: str`
- `TargetSpec.io: TargetIO`
- `TargetSpec.tags: frozenset[str]`

- `TargetIO.outputs: tuple[str, ...]` (table keys)
- `TargetIO.artifacts: tuple[ArtifactSpec, ...]`
- `TargetIO.optional_inputs: tuple[str, ...]`
- `TargetIO.required_inputs: tuple[str, ...]`

4) ExecutionOutcome
- `status: Literal["succeeded", "skipped", "failed"]`
- `reason: str | None`
- `row_counts: dict[str, int] | None`
- `artifacts: dict[str, str] | None`
- `duration_ms: float`

5) MaterializationResult
- `status: Literal["succeeded", "skipped", "failed"]`
- `table_key: str | None`
- `row_count: int | None`
- `artifact_name: str | None`
- `path: str | None`
- `error: str | None`

6) ValidationReport
- `status: Literal["ok", "warn", "error"]`
- `issues: tuple[ValidationIssue, ...]`
- `summary: dict[str, int]`

7) ValidationIssue
- `scope: Literal["table", "target", "artifact"]`
- `table_key: str | None`
- `target: str | None`
- `severity: Literal["warn", "error"]`
- `message: str`
- `details: dict[str, object] | None`

## Phase 0: Inventory and Baseline (1-2 days)
Goal: establish ground truth, test baselines, and migration map.

File-level tasks
- Add a consolidated inventory doc of schema sources and contract overlays.
- Map duplicate metadata sources (BuildSpec, registry inventory, target metadata).
- Map storage/serving schema and contract consumers that must be unified.

Tests/checkpoints
- Run targeted tests for schema registry and build status output.
- Baseline performance: build status and build run on a small repo.

## Phase 1: SchemaAuthority Consolidation (Highest Priority)
Goal: unify schema specification and contract overlays in one authority.

New modules
- `src/codeintel/core/schemas/authority.py`
  - `SchemaAuthority`, `SchemaDerivation`, `SchemaSelection`

File-level refactors
- Replace provider fan-out:
  - `src/codeintel/build/schemas/provider_unified.py` -> thin wrapper over
    `SchemaAuthority`.
- Merge/bridge contract sources:
  - `src/codeintel/build/hamilton/contracts/schemas/*` -> consumed by
    `SchemaAuthority` as overlays, not a parallel registry.
- Update SchemaService wiring:
  - `src/codeintel/build/schemas/service.py` -> uses SchemaAuthority directly.
- Remove redundant schema lookup:
  - `src/codeintel/build/schemas/contract_service.py` -> moved into authority.
- Storage: update `src/codeintel/storage/validation/*` and
  `src/codeintel/storage/views/*` to use SchemaAuthority.
- Serving: update `src/codeintel/build/serving/*` and `src/codeintel/serving/*`
  to resolve schemas via SchemaAuthority.
- CLI: update dataset commands to resolve schemas via SchemaAuthority.

Hamilton integration
- Use Hamilton tag index to drive derivation lineage.
- Prefer `@check_output_custom` for schema validation hooks.

Tests/checkpoints
- New tests:
  - DAG schema overrides declared schema.
  - Declared schema is used only when DAG schema is missing.
  - Schema derivation stored for each output table.
- Update schema snapshot tests if needed.

## Phase 2: Target Catalog + BuildSpec Unification
Goal: single canonical target metadata derived from the DAG.

New modules
- `src/codeintel/core/targets/catalog.py`
  - `TargetSpec`, `TargetIO`, `TargetCatalog`.

File-level refactors
- `src/codeintel/build/target_metadata.py` -> delegates to TargetCatalog.
- `src/codeintel/build/spec/*` -> becomes thin serialization for TargetCatalog
  (or replaced by TargetCatalog exports).
- `src/codeintel/core/registry/dag_output_inventory.yaml` -> regenerated from
  TargetCatalog (no manual drift).
- Storage: metadata tables and inventory readers use TargetCatalog.
- Serving: manifests and publish flows use TargetCatalog.
- CLI: `build.targets` and registry commands use TargetCatalog directly.

Hamilton integration
- Build TargetCatalog via DriverBuilder and TagSpec.

Tests/checkpoints
- TargetCatalog equals DAG outputs (table keys + artifacts).
- CLI `build.targets` uses TargetCatalog (no drift with registry inventory).

## Phase 3: Unified Execution Pipeline
Goal: one execution path for all targets with standard outcomes.

New modules
- `src/codeintel/core/execution/*`
  - `ExecutionOutcome`, `ExecutionContext`, `ExecutionPolicy`.

File-level refactors
- Merge `src/codeintel/build/hamilton/executor.py` and
  `src/codeintel/build/hamilton/native/executor.py` into `execution.py`.
- Replace per-target skip logic with shared policy in `ExecutionPolicy`.
- Update build, serving, and tracking to consume `ExecutionOutcome`.

Hamilton integration
- Use Driver hooks for start/end instrumentation and caching.

Tests/checkpoints
- Skip/blocked propagation tests for graphs and ingestion.
- Build run status tests for computed/skipped/failed.

## Phase 4: Materialization Unification
Goal: one metadata format for rows and artifacts, consistent across savers.

New modules
- `src/codeintel/core/execution/materialization.py`
  - `MaterializationResult`.

File-level refactors
- Consolidate `src/codeintel/build/hamilton/materializers/*` and
  `src/codeintel/build/hamilton/native/materialization_records.py`.
- Update storage tracking to read MaterializationResult.
- Update serving publish/manifest flows to read MaterializationResult.
- Standardize row_counts and artifact path reporting.

Hamilton integration
- Use shared saver templates (`save_rows`, `save_artifact`) to emit
  MaterializationResult consistently.

Tests/checkpoints
- Materialization metadata round-trip in build tracking.
- Table + artifact output results consistent across targets.

## Phase 5: Unified Validation Pipeline
Goal: consistent validation and reporting with Hamilton-driven hooks.

New modules
- `src/codeintel/core/validation/report.py`
  - `ValidationReport`, `ValidationIssue`.

File-level refactors
- Consolidate `src/codeintel/build/hamilton/validators/*` and
  `src/codeintel/build/exports/validation.py` into shared validation module.
- Update storage conformance checks to emit ValidationReport.
- Update serving/CLI export validation to emit ValidationReport.
- Replace scattered contract enforcement with a single ValidationHook.

Hamilton integration
- Use `@check_output_custom` to enforce validation contract at materialization
  boundaries.

Tests/checkpoints
- Validation produces consistent issue structure for errors and warnings.
- Export validation consumes ValidationReport directly.

## Phase 6: Row/JSON Serialization Consolidation
Goal: unify JSON handling, row hashing, and Ibis conversion.

New modules
- `src/codeintel/core/serialization/row_codec.py`
  - `RowCodec`, `JsonBoundaryPolicy`.

File-level refactors
- `src/codeintel/build/hamilton/native/ibis_helpers.py` -> uses RowCodec.
- `src/codeintel/build/hamilton/io/*` -> uses RowCodec and SchemaAuthority.
- Storage insert/upsert helpers use RowCodec.
- Serving/export JSON encoding happens only at export boundaries.
- Align insert helpers so JSON stays native in DuckDB.

Tests/checkpoints
- JSON column storage test (native JSON in DuckDB).
- Row hash injected consistently for core tables.

## Phase 7: Clean-up and Deprecation Removal
Goal: delete legacy/duplicate modules, simplify build surface.

File-level refactors
- Remove unused schema registries and legacy contract layers.
- Remove duplicated execution/util modules after migration.
- Update docs: architecture and DAG migration plans.

Tests/checkpoints
- Full quality report + targeted pytest slices for build/graphs/ingestion/export.
- Add targeted CLI/serving tests for status and validation payloads.

## Test Checkpoint Matrix (By Phase)
- Phase 1: schema authority + schema snapshot tests.
- Phase 2: registry inventory + build.targets CLI.
- Phase 3: build status/run + skip propagation tests.
- Phase 4: materialization metadata tests.
- Phase 5: validation pipeline tests (exports + contract).
- Phase 6: row serialization + JSON storage tests.
- Phase 7: full quality report + top-level smoke suite.

## Implementation Notes
- Prefer file moves with deprecations removed immediately (design phase allows
  breaking changes).
- Use Hamilton modifiers to remove custom pipeline orchestration where possible.
- Keep new dataclasses in build-owned modules to avoid cross-layer drift.

## Deliverables
- New shared API modules (SchemaAuthority, TargetCatalog, ExecutionOutcome,
  MaterializationResult, ValidationReport).
- Updated build runtime that only reads schemas and targets from DAG.
- Reduced surface area and duplication across `src/codeintel/build`.
- CLI/storage/serving updated to consume the shared APIs instead of local copies.
