---
title: "Polars/Arrow-First Compute with SQL Serving Extension Plan"
status: "design"
scope: "hamilton compute + mcp serving"
---

# Polars/Arrow-First Compute with SQL Serving Extension Plan

This plan implements a Polars/Arrow-first compute substrate (Hamilton) and a serving layer that
defaults to Polars/Arrow but can route complex queries to DuckDB + SQLGlot as an extension.
It is designed for columnar analytics workloads with optional SQL power where needed, without
forcing SQL into the core compute pipeline.

Note: dataset-manifest references in this plan are superseded by Iceberg metadata and per-dataset
export manifests plus export summary markers.

## Decisions (Current)

1. **Dataset store & manifests**
   - Use Hive-style partitioning (e.g., `repo=<repo>/commit=<commit>` or `snapshot_id=<id>`).
   - Provide scanner-first reads (batch sizing + fragment readahead); avoid `to_table()` footguns.
   - Keep manifests Iceberg-shaped (manifest list + file stats + partition specs) for future adoption.

2. **Materialization**
   - Prefer Polars `sink_parquet` for LazyFrames.
   - Use `collect_all()` for shared subplans to reduce duplicate scans.
   - Do not serialize LazyFrames; persist Parquet/IPC only.

3. **Serving engine**
   - Replace SQL view maps with a Polars `ViewSpec` registry (pure functions + tags).
   - Compile `FilterSpec` → `pl.Expr` directly.
   - Inject row indices at scan-time when stable IDs are required.
   - Make streaming (`RecordBatchReader`) and cancellation/timeout first-class.
   - Use an engine plugin dispatch layer (Polars first; DuckDB relations for complex queries).

4. **Complex queries**
   - Prefer DuckDB relational API as the primary complex-query path.
   - Use SQLGlot AST builders when relations are insufficient (no Ibis fallback).
   - Avoid raw SQL strings; keep DuckDB optional and fed by Arrow scanners.

5. **Validation + cleanup**
   - Remove pandas-based validation; use Arrow validation or Pandera-on-Polars.
   - Use as-of join rewrites for range joins (coverage) to avoid row explosion.
   - Decommission legacy/compat paths as soon as new systems are authoritative.

---

## Schema Migration Detail (Arrow/Polars-First)

This section pins the exact derivation flow, type mapping expectations, and decommissioning
gates required to move schema orchestration out of DuckDB and into the Arrow/Polars pipeline,
while keeping `meta.duckdb` as the authoritative registry.

### Current-state inventory (DuckDB-dependent schema derivation)

- View schema inference: `src/codeintel/build/schemas/infer_duckdb.py` (`infer_view_schema`).
- DAG output schema inference: `src/codeintel/build/schemas/inference_service.py` uses
  `coerce_to_relation(...)` + DuckDB `DESCRIBE` via `infer_table_schema_from_relation(...)`.
- Seed harness: `src/codeintel/build/schemas/seed_harness.py` creates empty DuckDB tables and
  produces q__ relations for inference execution.
- Manifest compilation: `src/codeintel/build/schemas/compile.py` uses DuckDB for view inference
  when `include_views=True`.

### Target derivation flow (Arrow/Polars-first)

1. **Materialize Arrow outputs** (Hamilton nodes):
   - Each table output yields `pa.Table` / `pa.RecordBatchReader` / `pl.LazyFrame`.
2. **Derive schema from Arrow/Polars**:
   - Convert `pa.Schema` (or `pl.Schema`) to `TableSchema`.
   - Embed metadata in Arrow schema (schema_hash/digest, snapshot_id, writer_version).
3. **Compile SchemaManifest from Arrow-derived schemas**:
   - Replace DuckDB `DESCRIBE` paths for views/tables.
   - Capture provenance (derivation kind/source, inference status).
4. **Persist to meta registry**:
   - `SchemaCatalogTracking.persist_schema_manifest(...)` remains authoritative, now fed by
     Arrow-derived schemas instead of DuckDB introspection.

### Arrow/Polars → TableSchema mapping (spec)

- **Types**: map Arrow/Polars types to `ColumnType` using a deterministic table (documented
  in code) with explicit handling for decimals, timestamps, and large binaries.
- **Nullability**: use Arrow field nullability (or Polars schema nullability when available).
- **Schema hash**: use `TableSchema` → `schema_hash` (not DuckDB type normalization).
- **Metadata propagation**: include `schema_hash`, `schema_digest`, `snapshot_id`,
  `writer_version`, and provenance fields in Arrow schema metadata.

### Manifest compilation updates

- Manifest compilation must read:
  - Table schemas from Arrow/Polars derivation (materializer outputs).
  - View schemas from ViewSpec/Polars derivation (no DuckDB `DESCRIBE`).
- Provenance fields must reflect:
  - `derivation_kind`: `inferred_relation` for DAG outputs, `declared_source` for inputs,
    `view_inferred` for ViewSpec outputs.
  - `derivation_source`: module/target identifier or ViewSpec name.

### Inference strategy changes (near-term)

- **Primary**: infer from Arrow/Polars outputs emitted by compute nodes (no DuckDB relation
  coercion for schema inference).
- **Fallback**: declared schemas for sources; override registry for inference failures.
- **q__ inputs**: replace DuckDB seed harness with Arrow/Polars seed tables built from declared
  schemas (empty `pa.Table` / `pl.DataFrame`).

### Validation gates (must pass)

- All schemas in the manifest are derived from Arrow/Polars outputs when available.
- No DuckDB `DESCRIBE` calls are used for schema derivation.
- `meta.duckdb` registry rows match Arrow-derived schema_hash/digest.
- `SchemaCatalogTracking` persist/refresh remains deterministic across identical DAG runs.

### Decommission checklist (explicit deletes)

- Remove `src/codeintel/build/schemas/infer_duckdb.py` once view inference is Arrow/Polars-based.
- Replace `seed_harness.py` with Arrow/Polars seeding and delete DuckDB-only seeding paths.
- Remove DuckDB relation coercion for schema inference in `inference_service.py`.
- Remove any remaining DuckDB-view schema inference in `compile.py`.

### Tests to add/update

- Arrow/Polars → TableSchema mapping tests (type + nullability + metadata).
- Manifest compilation uses Arrow/Polars schemas (no DuckDB dependency).
- Meta registry rows reflect Arrow-derived schema hashes and provenance.

## Goals

- Make Hamilton compute outputs canonical Arrow datasets (Parquet/Arrow IPC), not DuckDB tables.
- Default serving queries to Polars/Arrow for columnar speed and programmatic APIs.
- Provide an optional SQL extension path (DuckDB + SQLGlot) for complex relational queries.
- Keep a single data contract (dataset manifests + schema metadata) shared by compute and serving.
- Preserve "no build_driver" constraints (composition stays in `compose_runtime(...)`).

## Non-Goals

- Preserve Ibis adapters or maintain transition shims.
- Require SQL for serving queries that are expressible in Polars/Arrow.
- Support dual-write or dual-read compatibility layers.

## End-State Architecture (Summary)

1. **Compute (Hamilton)**
   - Nodes output `pl.LazyFrame`, `pl.DataFrame`, `pa.Table`, or `pa.RecordBatchReader`.
   - Materialization writes Arrow datasets (partitioned Parquet) plus a manifest.
   - No SQL inside the compute path except where explicitly unavoidable.

2. **Canonical Storage**
   - Arrow dataset per table_key (partitioned by repo/commit or snapshot_id).
   - Manifest records dataset version, schema hash, partition layout, and file list.
   - A snapshot pointer references the active dataset manifests.

3. **Serving**
   - Default engine: Polars/Arrow (expression builder + lazy execution).
   - SQL extension: DuckDB attaches Arrow datasets; SQLGlot builds safe SQL AST.
   - Query router selects engine based on capability or request preference.

## Core Contracts

- **Dataset layout**: `<dataset_root>/<table_key>/snapshot_id=<id>/...` (partitioned Parquet).
- **Manifest**: JSON or Arrow schema metadata with dataset_id, snapshot_id, schema hash, stats,
  partition spec, and file list (plus optional lineage metadata).
- **Pointer**: `current.json` (or equivalent) includes dataset manifest path(s) for the active
  serving snapshot.

---

# Phased Implementation Plan

## Phase 0: Configuration + Contract Lock

Objective: define configuration surfaces and dataset contract schema.

Tasks:
- Add dataset root configuration (likely derived from `BuildPaths.document_output_dir`).
- Define dataset manifest schema in a dedicated module (JSON schema or dataclass + serializer).
- Extend `ServingSettings` to include query engine selection (`polars`, `duckdb`, `auto`).
- Update snapshot pointer schema to include dataset manifest paths (no new build_driver usage).

Deliverables:
- Dataset manifest schema module.
- Config fields wired into runtime settings.
- Pointer schema updated for dataset references.

Acceptance:
- Dataset root resolved deterministically in build and serving runtime.
- Pointer contains both repo/commit identity and manifest location(s).

---

## Phase 1: Arrow Dataset Store + Registry

Objective: implement a canonical Arrow dataset store and registry.

Tasks:
- Add a dataset store module (e.g., `src/codeintel/storage/datasets/arrow_store.py`) with:
  - `write_dataset(table_key, snapshot_id, data, partition_spec)`.
  - `scan_dataset(table_key, snapshot_id)` returning `pyarrow.dataset.Dataset`.
  - `dataset_stats(...)` for row counts + basic stats.
- Adopt Hive-style partitioning (e.g., `repo=<repo>/commit=<commit>` or `snapshot_id=<id>`) and
  persist partition layout in the manifest for pruning.
- Provide scanner-first read helpers (batch sizing, fragment readahead) to avoid accidental
  `to_table()` full materialization.
- Extend `src/codeintel/storage/datasets/registry.py` to track Arrow dataset roots + manifests.
- Add manifest persistence under `src/codeintel/storage/metadata` or `src/codeintel/storage/datasets`.
- Add schema metadata helpers (Arrow schema + schema hash) in `src/codeintel/storage/schema`.
- Design manifest schema to be **Iceberg-shaped** (manifest list + file entries + stats + partition
  specs) so we can adopt Iceberg/Delta later without a redesign.
- Add Arrow/Polars schema → `TableSchema` converters (Arrow schema is the new derivation source).

Deliverables:
- Arrow dataset store with manifest IO and partition-aware scanning.
- Registry methods to load dataset manifests by table_key and snapshot_id.

Acceptance:
- Dataset manifest round-trips for at least one dataset.
- Registry resolves table_key -> dataset path + schema hash without DuckDB.

---

## Phase 2: Hamilton Materializers for Arrow Datasets

Objective: persist Hamilton outputs directly as Arrow datasets.

Tasks:
- Add `ArrowDatasetSaver` (e.g., `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`)
  supporting `pl.LazyFrame`, `pl.DataFrame`, `pa.Table`, `pa.RecordBatchReader`.
- Implement partitioned writes using `pyarrow.dataset.write_dataset(...)` or
  `pl.LazyFrame.sink_parquet(...)` with `partition_by`.
- Prefer Polars native sinks (`sink_parquet`) for LazyFrames; avoid serializing LazyFrames.
- Use `collect_all()` when multiple outputs share subplans to reduce duplicate scans.
- Emit dataset manifest metadata (schema hash, row counts, partition spec).
- Emit `TableSchema` + provenance from Arrow outputs at materialization boundaries.
- Add a `save_dataset(...)` helper in `src/codeintel/build/hamilton/native/patterns`.
- Update `serving_artifacts` and materialization records to include dataset manifests.

Deliverables:
- Arrow dataset saver with manifest emission.
- Helper decorators for dataset materialization.

Acceptance:
- At least one Hamilton target writes Arrow datasets and registers manifests.
- No pandas conversions inside saver paths.

---

## Phase 3: Polars/Arrow-First Serving Engine

Objective: replace legacy query paths with Polars/Arrow query plans.

Tasks:
- Introduce a `SemanticQuerySpec` (or similar) independent of backend.
- Implement a Polars query builder that converts filters/sorts into Polars expressions.
- Create a serving engine (e.g., `src/codeintel/serving/semantic/engines/polars_engine.py`) that:
  - Loads Arrow datasets via `pl.scan_parquet(...)` or `pyarrow.dataset.Dataset`.
  - Applies filters/projections/joins in Polars lazy.
  - Emits Arrow `RecordBatchReader` or `pa.Table` for responses.
- Add a ViewSpec-style registry (Polars view functions + tags) to replace SQL view maps.
- Prefer scan-time row index injection when stable row IDs are required.
- Update `src/codeintel/serving/semantic/kernel.py` to use the Polars engine by default.
- Replace legacy template types in `templates.py` with backend-neutral specs.
- Add engine-plugin dispatch contract:
  - `QueryEngine.can_run(spec)`, `compile(spec) -> ExecutablePlan`, `ExecutablePlan.to_reader()`.
  - Register Polars and DuckDB engines via a registry (avoid if/else routing).
- Treat streaming + cancellation as first-class:
  - Prefer `RecordBatchReader` end-to-end.
  - Add cancellation/timeout hooks for MCP + HTTP query lifecycles.
- Replace DuckDB-based view schema inference with ViewSpec/Polars-derived schemas.

Deliverables:
- Polars query engine wired into the semantic kernel.
- Query builder that validates columns and builds Polars expressions.

Acceptance:
- Semantic queries execute via Polars/Arrow in the default path.
- MCP responses stream Arrow with consistent schema and null handling.

---

## Phase 4: Complex Query Engine (DuckDB Relations + SQLGlot)

Objective: add a complex-query engine without raw SQL strings, preferring DuckDB relations and
using SQLGlot only when the relational API is insufficient.

Tasks:
- Implement a DuckDB serving engine that attaches Arrow datasets via `read_parquet` or
  `parquet_scan` and executes relational API query plans.
- Use DuckDB relational API as the default complex-query path.
- Use SQLGlot to build/validate SQL when the relational API cannot express the query.
- Add a query router in `serving/semantic` that selects Polars vs DuckDB based on:
  - explicit request preference, or
  - capability detection (window functions, complex joins, CTEs).
- Preserve SQL ingress policies (`SqlIngressPolicy`) and add SQLGlot allowlists.
- Keep DuckDB strictly optional; prefer Arrow scanner integration and relation plans over SQL strings.

Deliverables:
- SQL extension engine with dataset attachment.
- SQLGlot query builder or adapter from `SemanticQuerySpec`.

Acceptance:
- Complex queries route to DuckDB without changing dataset storage.
- SQL executed only through validated SQLGlot AST (no raw string concatenation).

---

## Phase 5: Snapshot Publish + Pointer Integration

Objective: ensure serving can swap snapshots based on dataset manifests.

Tasks:
- Make publishing **transactional**: pointer references a single `snapshot_root` and
  a `snapshot_manifest.json` (root manifest), not per-table manifest paths.
- Add a root manifest that lists table_key → table manifest + schema hash + stats + partition spec.
- Keep per-table manifests as internal artifacts, but serving resolves everything from the root manifest.
- Extend `ServingDBManager` to load dataset manifests alongside the registry/schema manifest.
- Add a dataset validation step in `storage/serving/snapshot_service.py` (exists, schema hash).
- Keep DuckDB snapshot support for SQL extension (attach datasets rather than copy).
- Split data plane vs metadata/index plane:
  - Introduce a small per-snapshot DuckDB metastore for search/stats/lineage (derived only).
  - Treat it as rebuildable from the snapshot manifest (not canonical).
- Add dataset maintenance utilities (may live under storage/serving or build tooling):
  - partition rewrite, compaction, vacuum/GC, verify.
- Ensure `meta.duckdb` continues as the authoritative registry, but schema derivation
  now originates from Arrow/Polars outputs (not DuckDB DESCRIBE).

Deliverables:
- Pointer includes dataset manifest metadata.
- Serving manager loads both dataset and semantic registry.

Acceptance:
- Hot-swap works with dataset-backed snapshots.
- Serving boot fails fast if dataset manifests are missing or mismatched.

---

## Phase 6: Decommission Legacy Paths (Remove Ibis Fully)

Objective: remove Ibis entirely and rely on DuckDB relations + SQLGlot for complex queries,
keeping raw SQL out of the serving layer.

Tasks:
- Delete all remaining Ibis usage across compute/serving paths.
- Remove the Ibis query builder and related shims in serving.
- Ensure complex-query execution uses DuckDB relations first, SQLGlot AST fallback second.
- Update docs to clarify the relation-first preference and SQLGlot-only fallback.

Deliverables:
- Codebase no longer imports or depends on Ibis (dependencies + runtime).
- Serving and compute paths fully Polars/Arrow + DuckDB (optional).

Acceptance:
- Ruff/Pyright/Pyrefly report zero Ibis imports anywhere.
- All non-complex semantic queries execute via Polars/Arrow engine.

---

# Validation Gates

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run targeted tests for `serving/semantic` and any modified storage modules.
- Verify Arrow dataset manifests and pointer updates on a sample build.

# Suggested Execution Order

1) Phase 0 (contracts + config)
2) Phase 1 (dataset store + manifest)
3) Phase 2 (Hamilton materializers)
4) Phase 3 (Polars serving engine)
5) Phase 4 (SQL extension)
6) Phase 5 (snapshot integration)
7) Phase 6 (decommission Ibis)

---

# Phase 0 Implementation Notes (Completed)

## Findings

- Configuration now exposes a dataset root directory (default: `document_output_dir / "datasets"`),
  alongside the existing Document Output path.
- Serving settings now include a `query_engine` selector (`auto|polars|duckdb`) without changing
  the existing `result_engine` output formatting.
- Snapshot pointers no longer rely on dataset manifest paths; per-dataset export manifests and
  Iceberg metadata cover serving snapshot needs.

## Outputs

- **Snapshot pointer upgrade**:
  - JSON schema updated: `src/codeintel/config/schemas/serving/current.json`
  - Pointer updated: `codeintel.serving.db.pointer.ServingSnapshotPointer`,
    `codeintel.core.manifests.ServingSnapshotManifest`
- **Config surface updates**:
  - Build paths include `dataset_root_dir` (`codeintel.config.primitives.BuildPaths`)
  - CLI model includes `dataset_root_dir` (`codeintel.config.models.CliPathsInput`)
  - Serving settings include `query_engine` (`codeintel.core.config.settings.ServingSettings`)

## Code Touchpoints

- `src/codeintel/config/primitives.py`
- `src/codeintel/config/models.py`
- `src/codeintel/core/config/settings.py`
- `src/codeintel/core/runtime/loader.py`
- `src/codeintel/core/manifests.py`
- `src/codeintel/config/schemas/serving/current.json`
- `src/codeintel/serving/db/pointer.py`
- `src/codeintel/serving/operations/protocols.py`
- `src/codeintel/build/serving/publisher.py`

---

# Legacy/Compatibility Decommissioning Scope (Tracked Across Phases)

This is a **required** workstream: any compatibility code or legacy paths introduced during
phased rollout must be removed as soon as they are no longer needed. If it is safer to remove
them only after the full system is in place, they must be explicitly tracked and deleted as part
of Phase 6 (no “left behind” shims).

## Policy

- **No permanent compatibility shims**: every shim has an explicit removal phase.
- **Deprecation is a task, not a note**: each shim has an owner phase and acceptance criteria.
- **Delete over migrate**: once the new path is authoritative, remove legacy code immediately.

## Required Tracking Artifacts (add as implemented)

Create/update a short “decommission registry” table in this doc during implementation:

| Shim/Legacy Path | Location | Introduced In Phase | Removal Phase | Removal Criteria |
| --- | --- | --- | --- | --- |
| Ibis query builder + temp table staging | `src/codeintel/serving/semantic/query_builder.py` | legacy | P6 | Removed; serving query tests pass with relation + SQLGlot fallback. |
| Ibis gateway facade | `src/codeintel/storage/gateway/ibis_facade.py` | legacy | P6 | Removed; no Ibis imports in codebase. |
| Ibis analytics helpers | `src/codeintel/analytics/compute/ibis_utils.py`, `src/codeintel/core/ibis_typing.py` | legacy | P6 | Removed; analytics profiles use DuckDB relations. |
| Ibis dependency | `pyproject.toml` | legacy | P6 | Dependency removed; tooling metadata updated. |

## Phase-Specific Decommissioning Scope

### Phase 1 (Dataset Store + Registry)
- Remove any old dataset mapping or manifest helper that only exists to bridge
  to DuckDB tables once Arrow dataset scanning is authoritative.

### Phase 2 (Hamilton Materializers)
- Remove any materializers that only exist to output DuckDB tables for intermediate compute
  once Arrow dataset materializers are stable for target outputs.

### Phase 3 (Polars/Arrow Serving Engine)
- Remove Ibis semantic query builder/templates once Polars query engine is default.
- Delete query compatibility branches that attempt Ibis fallback.

### Phase 4 (SQL Extension)
- Remove any duplicated SQL templating paths once SQLGlot AST builders are live.
- Remove raw SQL f-string paths for serving queries.

### Phase 5 (Snapshot Integration)
- Remove snapshot publishers that copy DuckDB tables for serving once dataset-backed snapshots
  are authoritative (retain DuckDB snapshot only if explicitly required for SQL extension).

### Phase 6 (Final Cleanup)
- Delete any remaining legacy/compat paths referenced in the decommission registry.
- Gate on: no Ibis imports, no unused compatibility helpers, and no references to old
  “df__*/q__*” loader patterns for serving/compute.
- Remove DuckDB-based schema inference and view schema inference once Arrow/Polars paths are authoritative.

## Additional refactor targets (high ROI)

- Replace pandas-based validation with Arrow or Pandera-on-Polars validation.
- Port SQL + Python-loop analytics to Polars aggregations or Arrow scanner batch pipelines.
- Use as-of join patterns to replace range joins where feasible (coverage joins).

---

# Phase 1 Implementation Notes (Completed)

## Findings

- Added an Arrow dataset store abstraction that writes/reads snapshot-scoped datasets and emits
  dataset manifests with schema hash + file lists.
- Introduced manifest persistence helpers and dataset path helpers to keep dataset layout stable.
- Extended dataset registry structures to carry dataset root and manifest metadata.

## Outputs

- **Dataset store**: `src/codeintel/storage/datasets/arrow_store.py`
- **Dataset path helpers**: `src/codeintel/storage/datasets/paths.py`
- **Dataset manifest helpers**: `src/codeintel/storage/datasets/manifests.py`
- **Registry extensions**: `src/codeintel/storage/datasets/registry.py`
- **Arrow schema metadata helpers**: `src/codeintel/storage/schema/arrow_schema.py`
