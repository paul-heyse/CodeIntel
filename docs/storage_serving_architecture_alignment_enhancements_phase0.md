# Phase 0 Audit: Storage + Serving Consolidation Opportunities

## Scope

Phase 0 of `docs/storage_serving_architecture_alignment_enhancements_plan.md`
covered an inventory of duplicated logic across `src/codeintel/storage` and
`src/codeintel/serving`, plus a map of consolidation points and a list of
legacy code that can be decommissioned once the consolidation is complete.

## Inventory Summary (shared primitives and duplication)

### Dataset scanning and Arrow metadata

**Current modules**
- `src/codeintel/storage/datasets/scanning.py`
- `src/codeintel/storage/datasets/arrow_store.py`
- `src/codeintel/storage/datasets/parquet_metadata.py`
- `src/codeintel/serving/semantic/datasets.py`
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- `src/codeintel/storage/serving/snapshot_service.py`
- `src/codeintel/serving/semantic/duckdb_scan_adapter.py`

**Observed duplication**
- Scan option normalization, schema unification, and fragment filtering exist
  both in storage and serving paths.
- Dataset registration into DuckDB (via `from_arrow`) appears in serving and
  snapshot preparation paths with similar alignment logic.
- Parquet metadata decode is storage-centric, but serving re-derives schema and
  metadata independently.

**Consolidation opportunity**
- Create a shared scan pipeline in storage (single entrypoint) and import it
  from serving/snapshot codepaths.
- Prefer dataset factories with `_metadata/_common_metadata` for schema
  stability (PyArrow advanced dataset factories + metadata plan).
- Centralize dataset-to-DuckDB registration in one module.

---

### Filter compilation and operator semantics

**Current modules**
- `src/codeintel/serving/semantic/filter_ops.py`
- `src/codeintel/serving/semantic/filter_compiler.py`
- `src/codeintel/storage/queries/expressions.py`
- `src/codeintel/storage/queries/safe.py`

**Observed duplication**
- Filter semantics and operator validation live in serving.
- Storage has separate expression helpers and safe query utilities with
  partially overlapping logic.

**Consolidation opportunity**
- Move filter compiler into a shared storage module and re-export in serving.
- Use one canonical operator set and type checks across both query paths.

---

### Contract/schema resolution and Arrow schema helpers

**Current modules**
- `src/codeintel/storage/schema/duckdb_contracts.py`
- `src/codeintel/serving/semantic/duckdb_contracts.py`
- `src/codeintel/storage/schema/arrow_schema.py`
- `src/codeintel/storage/datasets/parquet_metadata.py`

**Observed duplication**
- Arrow schema resolution is wrapped in multiple layers.
- Arrow schema helpers in storage are now thin wrappers around DuckDB-backed
  contract logic, duplicating APIs.

**Consolidation opportunity**
- Keep `duckdb_contracts.py` as the sole contract resolver.
- Fold Arrow schema helpers into a single canonical location and re-export.

---

### Arrow-to-row conversion and export serialization

**Current modules**
- `src/codeintel/storage/query_results.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/export/ndjson.py`
- `src/codeintel/serving/http/streaming.py`

**Observed duplication**
- Row conversion from Arrow batches exists in both storage and serving.
- Export serialization and streaming are split between HTTP and MCP paths.

**Consolidation opportunity**
- Create a shared Arrow-to-row conversion utility with consistent normalization.
- Share NDJSON and Arrow IPC encoding paths between HTTP and MCP exports.

---

### SQLGlot AST normalization and projection derivation

**Current modules**
- `src/codeintel/serving/semantic/sqlglot_query_builder.py`
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- `src/codeintel/storage/sqlglot_tools.py`

**Observed duplication**
- AST canonicalization and metadata extraction are distributed across files.
- Projection/pushdown derivation is partially coupled to the relation builder.

**Consolidation opportunity**
- Centralize AST normalization and metadata extraction in `storage/sqlglot_tools.py`.
- Ensure serving uses normalized AST for all plan construction.

---

### DuckDB relation building and snapshot registration

**Current modules**
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- `src/codeintel/storage/serving/snapshot_service.py`
- `src/codeintel/storage/warehouse.py`

**Observed duplication**
- Relation creation and alignment are repeated between serving and snapshot
  preparation.

**Consolidation opportunity**
- Create a shared relation-registration entrypoint for datasets and views.

---

### FastAPI/FastMCP transport behavior

**Current modules**
- `src/codeintel/serving/http/streaming.py`
- `src/codeintel/serving/http/routes/v1/*.py`
- `src/codeintel/serving/mcp/tools/*.py`
- `src/codeintel/serving/mcp/resource_store.py`

**Observed duplication**
- Streaming export behavior and cancellation logic are implemented separately
  for HTTP and MCP.

**Consolidation opportunity**
- Centralize export streaming and cancellation handling in a shared layer.

## Advanced Capability Hooks (from library references)

- **DuckDB**: prepared statements for repeated metadata queries, relation API
  chaining to enforce query semantics intrinsically, and profiling hooks for
  explain/metrics capture.
- **SQLGlot**: AST normalization and metadata extraction to drive pushdown and
  intrinsic query validation.
- **PyArrow**: dataset factories, unified schemas, row-group pruning, and IPC
  metadata control for stable schemas and efficient streaming.
- **Polars**: controlled streaming execution (`collect_batches` / `sink_batches`)
  and explicit optimizer flags to avoid hidden materialization.
- **FastAPI/FastMCP**: explicit streaming response controls, cancellation hooks,
  and typed response models to make output shape intrinsic.

## Legacy Code Decommission Candidates (post-consolidation)

These items become redundant once the consolidation tasks are complete.
Decommission only after shared replacements are in place.

### 1) Arrow schema wrappers
- `src/codeintel/storage/schema/arrow_schema.py`  
  **Reason**: thin wrapper around DuckDB-backed contract resolver.
  **Replacement**: `src/codeintel/storage/schema/duckdb_contracts.py`

### 2) Local dataset scanning wrappers
- `src/codeintel/serving/semantic/datasets.py` (most helpers)  
  **Reason**: dataset scan + metadata logic should move to shared storage layer.
  **Replacement**: shared scan entrypoint in `storage/datasets`.

### 3) Duplicate Arrow-to-row conversion
- `src/codeintel/serving/semantic/kernel.py` (batch-to-row helpers)  
  **Reason**: already duplicated in `storage/query_results.py`.
  **Replacement**: shared Arrow row conversion utility.

### 4) Redundant filter expression helpers
- `src/codeintel/storage/queries/expressions.py` (non-snapshot helpers)  
  **Reason**: canonical filter compiler should provide both DuckDB/Arrow/Polars
  expressions in one place.
  **Replacement**: shared filter compiler module (re-exported in serving).

### 5) Export serialization fragments
- `src/codeintel/serving/http/streaming.py` (NDJSON assembly paths)  
  **Reason**: should reuse shared export serialization utilities used by MCP.
  **Replacement**: shared export encoder + streaming adapter.

### 6) Snapshot-specific dataset registration paths
- `src/codeintel/storage/serving/snapshot_service.py` (registration pieces)  
  **Reason**: dataset registration should use the same shared scan pipeline as
  serving to avoid divergent semantics.
  **Replacement**: shared scan/registration module used by both.

## Deliverable

This audit provides the Phase 0 map of duplicated logic and consolidation
opportunities, plus a conditional decommission list for legacy code. It should
be used to drive Phase 1 refactoring and consolidation work.
