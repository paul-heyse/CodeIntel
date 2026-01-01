# Polars, PyArrow, and Arrow IPC First Implementation Plan for src/codeintel

## Intent

Re-architect src/codeintel/build (and adjacent serving/exports) so all data
operations are columnar by default, using Polars (LazyFrame-first) and PyArrow
(RecordBatchReader-first) as the canonical in-process types. This plan is now
aligned to the Arrow IPC response contract described in
docs/polars_arrow_ipc_rearchitecture_implementation_plan.md and assumes
breaking changes are allowed to simplify the codebase, remove pandas and
row-tuple flows, and unlock streaming, schema-driven, and maintainable
pipelines.

## Guiding Principles

- Make columnar types the only supported in-process table representation.
- Keep data lazy and streaming as long as possible.
- Prefer RecordBatchReader at boundaries to enable incremental downstream work.
- Use Arrow schemas as the contract boundary between components.
- Minimize conversions, especially to Python dicts or lists.
- Prefer explicit, typed conversions over implicit "magic" behavior.
- Remove pandas and row-tuple pipelines from build.
- Make Arrow IPC streams the default serving response format.
- Require schema metadata on IPC streams for traceability.
- Use ColumnarStream-style adapters to unify streaming and lazy inputs.

## Target Architecture

### Canonical Types

Define a single set of canonical types and use them everywhere in build:

- pl.LazyFrame for in-process transformations and "compute DAG" outputs.
- pa.RecordBatchReader for streaming handoff between compute and persistence.
- pa.Table only for small, bounded in-memory data or when needed by an API.
- DuckDB relations only at IO boundaries (ingest from DuckDB, materialize to it).

### Arrow IPC Response Contract (Aligned)

- Serving responses default to Arrow IPC streams
  (`application/vnd.apache.arrow.stream`).
- IPC schema metadata includes `codeintel.table_key`, `codeintel.schema_hash`,
  `codeintel.snapshot_id`, and optional `codeintel.query_hash`.
- Batch sizes follow `DEFAULT_ARROW_BATCH_SIZE` unless overridden.

### Columnar Pipeline Stages

1. Ingestion (tool outputs) -> Arrow/Polars
2. Cleaning + normalization -> Polars expressions
3. Dedupe and enrichment -> Polars expressions
4. Materialization -> DuckDB relation or Arrow dataset
5. Export -> Arrow-native JSONL/Parquet streaming
6. Serving -> Arrow IPC stream by default

## Workstreams and Detailed Changes

### A) Canonical Tabular Types and Conversion Utilities

Objective: Make Arrow and Polars the only data-frame types in build.

Changes:
- Update `src/codeintel/build/tabular/types.py` to remove pandas and redefine:
  - TabularInput = DuckDBRelation | pa.RecordBatchReader | pl.LazyFrame | pa.Table
  - InferableTabularInput = pa.RecordBatchReader | pl.LazyFrame | pa.Table
  - Add TabularFrame alias = pl.LazyFrame.
- Add a new module `src/codeintel/build/tabular/conversion.py`:
  - relation_to_arrow_reader(relation) -> pa.RecordBatchReader
  - relation_to_polars_lazy(relation) -> pl.LazyFrame
  - arrow_reader_to_lazyframe(reader) -> pl.LazyFrame
  - table_to_lazyframe(table) -> pl.LazyFrame
- Update `src/codeintel/build/tabular/duckdb_relation.py`:
  - Avoid `reader.read_all()` in `relation_to_polars`.
  - Prefer Arrow reader conversion to Polars with minimal copies.

Acceptance criteria:
- No pandas references in `src/codeintel/build/tabular`.
- All conversions are explicit and live in a single conversion module.

### B) Ingestion Pipeline: Arrow/Polars First

Objective: Replace tuple-row payloads with columnar tables in ingestion steps.

Changes:
- Update tool output dataclasses in:
  - `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
  - `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
  to emit `pl.LazyFrame` or `pa.RecordBatchReader` instead of row tuples.
- Replace `TRowsByTable` in
  `src/codeintel/build/hamilton/native/patterns/tool_target.py` with a
  mapping that accepts columnar outputs:
  - Mapping[str, TabularInput]
- Replace row-based cleaning in
  `src/codeintel/build/hamilton/native/ingestion/pipelines.py` with
  Polars-based cleaning:
  - drop nulls in required columns with `LazyFrame.drop_nulls`
  - enforce required columns and types via schema
- Replace `_dedupe_rows_for_table` in
  `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py` with
  a Polars-based dedupe:
  - `frame.unique(subset=key_columns, keep="last")`
  - apply "prefer_columns" logic via sort keys and `group_by` if needed

Acceptance criteria:
- Ingestion tool outputs are columnar end-to-end.
- No tuple-row payloads in ingestion modules.

### C) Materialization: Remove Row Tuples and Standardize on Columnar

Objective: Remove row-tuple materialization and route all table writes through
DuckDB relation or Arrow dataset paths.

Changes:
- Deprecate `DuckDBRowsSaver` in
  `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`.
  Replace with a columnar saver that accepts:
  - pl.LazyFrame
  - pa.RecordBatchReader
  - pa.Table
- Update `src/codeintel/build/hamilton/native/patterns/savers.py`:
  - Replace `save_rows` with `save_relation_table` or `save_dataset` usage.
  - Keep a compatibility shim only if required during migration.
- Update `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
  to accept Arrow and Polars as first-class inputs without temp conversions.
- Align `src/codeintel/storage/warehouse.py` to accept Arrow readers or Tables
  for materialization (if still required by build flows).

Acceptance criteria:
- No path writes data via row tuples.
- All table materialization uses relation or columnar inputs.

### D) Analytics Compute: LazyFrame and RecordBatchReader by Default

Objective: Ensure all analytics compute outputs are LazyFrame or
RecordBatchReader, and remove any Python row assembly.

Changes:
- Standardize on `relation_to_polars(...)` returning LazyFrame for base frames in:
  - `src/codeintel/build/hamilton/native/analytics/tables_functions.py`
  - `src/codeintel/build/hamilton/native/analytics/tables_modules.py`
  - `src/codeintel/build/hamilton/native/analytics/tables_risk.py`

Acceptance criteria:
- Analytics targets return `pl.LazyFrame` or `pa.RecordBatchReader`.
- No analytics output is a tuple of rows.

### E) Exports and Validation: Arrow Streaming

Objective: Keep exports streaming and Arrow-native; avoid Python dict loops.

Changes:
- Update `src/codeintel/build/exports/writers.py`:
  - JSONL writer should stream via Arrow batches (or Polars `write_ndjson`)
  - Parquet writer should stream RecordBatchReader -> ParquetWriter
- Update `src/codeintel/build/exports/jsonl.py` to use Arrow/Polars writers
  for repo_map export instead of `to_pydict`.
- Update `src/codeintel/build/exports/validation.py`:
  - Parquet validation should use `ParquetFile.iter_batches()` to avoid
    `table.to_pylist()` for large files.

Acceptance criteria:
- No usage of `to_pydict` or `to_pylist` in export paths.
- All export writers accept Arrow readers or Polars frames.

### F) Schema, Contracts, and Constraints

Objective: Make schema enforcement columnar and remove pandas dependencies.

Changes:
- Replace `pandera.pandas` usage in:
  - `src/codeintel/build/schemas/service.py`
  - `src/codeintel/build/schemas/constraints.py`
  with a Polars- or Arrow-native validation layer.
- Update `src/codeintel/build/schemas/inference_service.py` to accept the
  new canonical types and avoid eager conversions.
- Ensure `TableSchema` to Arrow schema conversion is the primary contract
  boundary for validation and export.

Acceptance criteria:
- No pandas or pandera.pandas in build.
- Schema checks operate on Arrow schema or Polars schema.

### G) Configuration and Runtime Defaults

Objective: Make "polars_lazy" the only supported backend in build.

Changes:
- Remove pandas backend branches in:
  - `src/codeintel/build/hamilton/transforms/decorators.py`
  - `src/codeintel/build/hamilton/transforms/with_columns_backend.py`
  - `src/codeintel/build/hamilton/transforms/tabular_steps.py`
- Update configuration defaults (where defined) to only allow:
  - df_backend = "polars_lazy"
  - clean_mode, null_policy remain but apply to Polars only

Acceptance criteria:
- No `df_backend == "pandas"` paths remain in build.
- Config schemas and defaults only reference Polars backends.

### H) Observability and Performance

Objective: Make performance and pipeline behavior visible with columnar metrics.

Changes:
- Use Polars `profile()` and `collect_schema()` during debug or diagnostics.
- Add Arrow dataset write metrics (rows, bytes, row groups) to materialization
  metadata where available.
- Add optional "streaming" mode for exports and validation using Arrow readers.

Acceptance criteria:
- Materialization records capture columnar-specific metrics.
- Exports support streaming metrics without loading full tables in memory.

### I) Arrow IPC Response Contract (Serving + Exports)

Objective: Make Arrow IPC the default serving contract and unify stream helpers.

Changes:
- Add ColumnarStream protocol and Arrow IPC writer utility.
- Stream IPC responses with cancellation checks and schema metadata injection.
- Align export writer protocols with Arrow batch access patterns.

Acceptance criteria:
- Serving defaults to Arrow IPC streams when requested.
- IPC schema metadata includes table and schema identifiers.
- Streaming writers avoid `to_pylist`/`to_pydict` in serving paths.

## Breaking Changes Summary

- Row tuple outputs and `save_rows` are removed from build.
- pandas support is removed from build.
- All ingestion and analytics targets return Polars or Arrow types.
- Schema validation shifts from pandas-centric to Arrow/Polars.
- Export writers accept columnar inputs only.
- Serving defaults to Arrow IPC streams; JSONL is opt-in for exports only.

## Migration Strategy (Phased, IPC-aligned)

### Phase 0: Contract and API scaffolding
- Add ColumnarStream protocol and Arrow IPC writer utility.
- Define schema metadata keys and default IPC response headers.
- Add cancellation checks for streaming iterators.

### Phase 1: Streaming conversion utilities
- Ensure conversions are streaming-first and avoid eager read_all paths.
- Add explicit eager conversions only when required.

### Phase 2: Ingestion outputs are columnar
- Tool outputs emit LazyFrame or RecordBatchReader only.
- Polars-based dedupe and cleaning replace row tuple flows.

### Phase 3: Dataset writes are streaming-first
- LazyFrame writes avoid full collect unless required by partitioning.
- RecordBatchReader writes are direct with manifest stats.

### Phase 4: Serving emits Arrow IPC streams
- Polars and DuckDB engines emit IPC via readers.
- Kernel writes IPC stream with metadata and cancellation support.

### Phase 5: Validation via Arrow/Polars (no pandas)
- Replace pandas/pandera validation with Arrow/Polars checks.
- Parquet validation uses batch iteration.

### Phase 6: Cleanup and hardening
- Remove remaining row tuple shims and pandas references.
- Update tests and docs to match IPC streaming defaults.

## IPC-Aligned Remaining Tasks Checklist

### Phase 0: Contract and API scaffolding
- Add ColumnarStream protocol and adapters plus Arrow IPC writer utility; files:
  `src/codeintel/core/columnar/stream.py`, `src/codeintel/core/exports/arrow_ipc.py`.
- Inject IPC schema metadata and response headers; files:
  `src/codeintel/serving/semantic/kernel.py`, `src/codeintel/serving/http/streaming.py`.
- Add cancellation checks between IPC batches; files:
  `src/codeintel/core/columnar/stream.py`, `src/codeintel/serving/semantic/kernel.py`.

### Phase 1: Streaming conversion utilities
- Ensure conversions are streaming-first and avoid eager read_all; files:
  `src/codeintel/build/tabular/conversion.py`, `src/codeintel/build/tabular/duckdb_relation.py`.
- Add explicit eager conversions only when required; files:
  `src/codeintel/build/tabular/conversion.py`.

### Phase 2: Ingestion outputs are columnar
- Confirm tool outputs and ingestion steps return LazyFrame or RecordBatchReader; files:
  `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`,
  `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`,
  `src/codeintel/build/hamilton/native/ingestion/scip.py`,
  `src/codeintel/build/hamilton/native/ingestion/pipelines.py`,
  `src/codeintel/build/hamilton/native/patterns/tool_target.py`.
- Replace any remaining row-based dedupe/cleaning with Polars expressions; files:
  `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`,
  `src/codeintel/build/hamilton/native/ingestion/pipelines.py`.

### Phase 3: Dataset writes are streaming-first
- Stream LazyFrame writes and avoid full collect except when partitioning requires it; files:
  `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`.
- Write RecordBatchReader datasets directly and enrich manifest stats; files:
  `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/storage/datasets/manifests.py`.

### Phase 4: Serving emits Arrow IPC streams
- Stream IPC from Polars and DuckDB engines; files:
  `src/codeintel/serving/semantic/engines/polars_engine.py`,
  `src/codeintel/serving/semantic/engines/duckdb_engine.py`.
- Write IPC responses with metadata and cancellation checks; files:
  `src/codeintel/serving/semantic/kernel.py`, `src/codeintel/core/exports/arrow_ipc.py`.
- Align export protocols with Arrow batch access patterns; files:
  `src/codeintel/storage/protocols/export.py`,
  `src/codeintel/build/exports/writers.py`,
  `src/codeintel/build/exports/jsonl.py`.

### Phase 5: Validation via Arrow/Polars (no pandas)
- Replace pandas/pandera validation with Arrow/Polars checks; files:
  `src/codeintel/build/exports/validation.py`,
  `src/codeintel/storage/validation/pandera_df.py`,
  `src/codeintel/build/schemas/constraints.py`,
  `src/codeintel/core/schemas/row_models.py`.

### Phase 6: Cleanup and hardening
- Remove row tuple materializers and compatibility shims; files:
  `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`,
  `src/codeintel/build/hamilton/native/patterns/savers.py`,
  `src/codeintel/build/hamilton/native/patterns/__init__.py`.
- Update tests to reflect columnar and IPC defaults; files:
  `tests/build/hamilton/test_materializer.py`,
  `tests/build/hamilton/test_dag_catalog_compiler.py`,
  `tests/build/hamilton/test_saver_declared_output_inventory.py`,
  `tests/ingestion/test_scip_ingest_result.py`.

## Short POC Plan (IPC-first)

- POC 1: Arrow IPC streaming response for a semantic query with metadata and
  cancellation checks; files: `src/codeintel/serving/semantic/kernel.py`,
  `src/codeintel/serving/semantic/engines/polars_engine.py`,
  `tests/serving/semantic/test_ipc_stream.py`.
- POC 2: Streaming dataset write from LazyFrame with partitioning plus manifest
  stats; files: `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`,
  `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/storage/datasets/manifests.py`,
  `tests/build/hamilton/materializers/test_arrow_dataset_streaming.py`.
- POC 3: SCIP ingestion end-to-end columnar pipeline; files:
  `src/codeintel/build/hamilton/native/ingestion/scip.py`,
  `tests/ingestion/test_scip_ingest_result.py`.

## Acceptance Criteria

- `rg -n "save_rows|DuckDBRowsSaver|pandas|to_pylist|to_pydict" src/codeintel/build src/codeintel/serving`
  returns no results.
- Default serving responses use Arrow IPC streams with required schema metadata.
- All ingestion and analytics outputs are columnar (Polars or Arrow).
- Export and validation pipelines are streaming and Arrow-native.
- All materializations route through columnar-aware savers.

## Testing and Benchmark Plan

- Unit tests for conversion utilities (Arrow reader -> Polars LazyFrame).
- Integration tests for ingestion targets that verify:
  - Schema correctness
  - Dedupe correctness
  - No materialization to Python dicts
- Export tests that validate JSONL/Parquet output using Arrow streaming.
- Serving tests that validate IPC stream metadata and cancellation behavior.
- Performance benchmarks comparing current row-tuple flows vs columnar flows:
  - ingestion throughput
  - memory peak during export
  - time to materialize in DuckDB

## Open Questions

- Where should we emit RecordBatchReader versus LazyFrame at boundaries?
- Which Arrow JSON writer should be the canonical JSONL output path?
- Should Arrow datasets be the default storage format for internal artifacts?
- How should we encode list/struct/map columns for export validation?
- Do we require `codeintel.query_hash` on every IPC response or only for queries?

## Immediate Next Steps

1. Implement IPC Phase 0 scaffolding (ColumnarStream + IPC writer + metadata).
2. Complete POC 1 (IPC streaming response) and POC 2 (streaming dataset write).
3. Align export protocols with Arrow batch access patterns.
4. Remove remaining row-tuple shims and update tests.

## Remaining Scope Checklist (Sequenced, File-Level)

### 1) Ingestion: remove row tuples and shift to columnar builders
- [ ] Convert SCIP row builders to produce columnar structures directly (no tuple lists):
  `src/codeintel/ingestion/scip/rows.py`
- [ ] Replace tuple-based assembly in SCIP ingestion with Polars-first construction:
  `src/codeintel/build/hamilton/native/ingestion/scip.py`
- [ ] Convert ingest target outputs to `pl.LazyFrame` / `pa.RecordBatchReader`:
  `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`,
  `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- [ ] Replace row-based cleaning/dedupe with Polars expressions:
  `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- [ ] Update ingestion payload mapping types to columnar inputs:
  `src/codeintel/build/hamilton/native/patterns/tool_target.py`

### 2) Analytics: LazyFrame/RecordBatchReader outputs only
- [ ] Ensure analytics table functions source from LazyFrame and return LazyFrame:
  `src/codeintel/build/hamilton/native/analytics/tables_functions.py`,
  `src/codeintel/build/hamilton/native/analytics/tables_modules.py`,
  `src/codeintel/build/hamilton/native/analytics/tables_risk.py`

### 3) Exports: Arrow/Polars streaming writers
- [ ] Stream JSONL via Arrow batches or Polars writer (no per-row dict loops):
  `src/codeintel/build/exports/writers.py`,
  `src/codeintel/build/exports/jsonl.py`
- [ ] Ensure Parquet writer uses RecordBatchReader with no eager materialization:
  `src/codeintel/build/exports/writers.py`
- [ ] Audit validation to be batch-based for Parquet:
  `src/codeintel/build/exports/validation.py`

### 4) Schema/constraints: Arrow/Polars-native enforcement
- [ ] Replace Pandera-based constraints with Arrow/Polars checks:
  `src/codeintel/build/schemas/constraints.py`,
  `src/codeintel/build/schemas/service.py`
- [ ] Remove remaining Pandera/pandas schema utilities in core:
  `src/codeintel/core/schemas/pandera_gen.py`,
  `src/codeintel/core/schemas/pandera_types.py`,
  `src/codeintel/core/schemas/json_schema_gen.py`

### 5) Serving/MCP parity: IPC-first across transports
- [ ] Add IPC-first query path for MCP (optional tool or format flag):
  `src/codeintel/serving/mcp/tools/query.py`,
  `src/codeintel/serving/mcp/models/semantic.py`,
  `src/codeintel/serving/mcp/resources/exports.py`
- [ ] Align MCP docs/prompts with IPC defaults:
  `src/codeintel/serving/mcp/prompts.py`

### 6) Remove legacy row materialization paths
- [ ] Remove or strictly isolate row-based materialization entry points:
  `src/codeintel/storage/warehouse.py`
- [ ] Sweep for row tuple usage in analytics/materialization:
  `src/codeintel/analytics/compute/data_models/usage.py`

### 7) Tests and coverage for new columnar flows
- [ ] Add/extend ingestion tests that assert columnar payloads end-to-end:
  `tests/ingestion/test_scip_ingest.py`,
  `tests/ingestion/test_scip_ingest_result.py`
- [ ] Add export tests validating IPC-first defaults and batch-based writers:
  `tests/serving/http/test_export.py`,
  `tests/serving/mcp/test_resources.py`,
  `tests/serving/export/test_formats.py`
- [ ] Add analytics tests that accept LazyFrame outputs:
  `tests/analytics/*` (targeted to the refactored modules above)
