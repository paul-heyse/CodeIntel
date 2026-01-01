# Storage + Serving Architecture Enhancements Plan

## Summary

This plan captures follow-on enhancements beyond
`docs/storage_serving_architecture_alignment_plan.md`, focused on consolidating
shared functionality, reducing divergence between storage and serving, and
making correctness intrinsic through design. It emphasizes advanced capabilities
in DuckDB, SQLGlot, PyArrow, Polars, FastAPI, and FastMCP.

## Objectives

- Consolidate dataset scanning, schema resolution, and query planning so
  storage and serving share a single execution and metadata model.
- Make correctness intrinsic (invalid states become unrepresentable).
- Prefer streaming, pushdown, and metadata-driven planning over eager
  materialization.
- Improve robustness, observability, and extensibility with minimal new
  configuration surface.

## Guiding Principles

- **Single source of truth** for filters, schemas, and query semantics.
- **Intrinsic correctness** via compiler-validated plans, not post-hoc checks.
- **Streaming-first** execution paths wherever possible.
- **Shared primitives** between storage and serving (avoid forked logic).
- **Metadata-driven planning** (Parquet/Arrow metadata + DuckDB catalog).

## Scope

### In scope

- Shared dataset scanning and metadata handling (Arrow/Parquet).
- Shared filter compiler, schema alignment, and projection/pushdown extraction.
- SQLGlot AST normalization and capability checks.
- Streaming export and response hardening for HTTP + MCP.
- Consolidation of Arrow->dict conversion and export serialization.
- Polars engine as a downstream tool with explicit execution controls.

### Out of scope

- Replacing DuckDB as the canonical execution engine.
- Reintroducing Arrow schema resolution in serving beyond zero-copy transport.
- Large-scale API redesigns of public endpoints.

## Workstreams and Phases

### Phase 0: Inventory and shared primitives audit

**Goals**
- Identify duplicated logic between `src/codeintel/storage` and
  `src/codeintel/serving`.
- Enumerate current scan, filter, and schema paths.

**Tasks**
- Inventory dataset scanning paths:
  - `src/codeintel/storage/datasets/scanning.py`
  - `src/codeintel/storage/datasets/arrow_store.py`
  - `src/codeintel/serving/semantic/datasets.py`
  - `src/codeintel/serving/semantic/duckdb_relation_builder.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
- Inventory filter compilation and operator semantics:
  - `src/codeintel/serving/semantic/filter_compiler.py`
  - `src/codeintel/serving/semantic/filter_ops.py`
  - `src/codeintel/storage/queries/expressions.py`
- Inventory Arrow IPC streaming and export paths:
  - `src/codeintel/serving/http/streaming.py`
  - `src/codeintel/serving/export/ndjson.py`
  - `src/codeintel/serving/semantic/kernel.py`

**Acceptance**
- A map of duplicated logic and candidate consolidation points.

**Phase 0 Findings (integrated)**

Dataset scanning and Arrow metadata
- Duplicate scan option plumbing and schema unification in:
  `src/codeintel/storage/datasets/scanning.py`,
  `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/serving/semantic/datasets.py`,
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`,
  `src/codeintel/storage/serving/snapshot_service.py`.
- Parquet metadata decoding lives in storage while serving re-derives schema.
- Consolidation opportunity: single scan pipeline + dataset factory usage with
  `_metadata/_common_metadata` and shared DuckDB registration.

Filter compilation and operator semantics
- Canonical operator set is in serving (`filter_ops.py` + `filter_compiler.py`).
- Storage has separate expression helpers (`storage/queries/expressions.py`).
- Consolidation opportunity: shared filter compiler used by both paths.

Contract/schema resolution
- `src/codeintel/serving/semantic/duckdb_contracts.py` is a wrapper around
  `src/codeintel/storage/schema/duckdb_contracts.py`.
- `src/codeintel/storage/schema/arrow_schema.py` is a thin wrapper around
  the same resolver.
- Consolidation opportunity: keep one resolver and re-export where needed.

Arrow-to-row conversion and export serialization
- Arrow-to-row normalization exists in:
  `src/codeintel/storage/query_results.py` and
  `src/codeintel/serving/semantic/kernel.py`.
- Export serialization is split across HTTP and MCP codepaths.
- Consolidation opportunity: one Arrow-to-row utility and shared export encoder.

SQLGlot AST normalization and projection derivation
- AST normalization is distributed across serving/storage.
- Projection derivation in the relation builder is local and ad hoc.
- Consolidation opportunity: centralized AST normalization + metadata extraction.

DuckDB relation building and snapshot registration
- Relation registration logic appears in serving and snapshot preparation.
- Consolidation opportunity: single dataset registration path with consistent
  alignment and scan options.

FastAPI/FastMCP transport behavior
- Streaming and cancellation are implemented separately for HTTP and MCP.
- Consolidation opportunity: shared streaming and cancellation helpers.

**Legacy decommission candidates (post-consolidation)**
- `src/codeintel/storage/schema/arrow_schema.py`: thin wrapper around DuckDB
  contracts. Replacement: `src/codeintel/storage/schema/duckdb_contracts.py`.
- `src/codeintel/serving/semantic/datasets.py` (scan helpers): redundant once
  shared scan pipeline is introduced. Replacement: shared storage scan helpers.
- `src/codeintel/serving/semantic/kernel.py` (Arrow row conversion helpers):
  duplicate of `src/codeintel/storage/query_results.py`. Replacement: shared
  Arrow-to-row utility.
- `src/codeintel/storage/queries/expressions.py` (non-snapshot helpers):
  overlaps with unified filter compiler. Replacement: shared filter compiler.
- `src/codeintel/serving/http/streaming.py` (NDJSON assembly paths):
  duplicate export serialization logic. Replacement: shared export encoder.
- `src/codeintel/storage/serving/snapshot_service.py` (dataset registration
  fragments): redundant once registration is consolidated. Replacement: shared
  scan/registration entrypoint.

---

### Phase 1: Unified dataset scanning + metadata consolidation

**Goals**
- Use a single scan pipeline with explicit options for Arrow datasets.
- Move metadata planning to dataset factories for deterministic schemas.

**Tasks**
- Introduce a shared scan entrypoint (storage + serving):
  - `src/codeintel/storage/datasets/scanning.py`
  - `src/codeintel/serving/semantic/datasets.py`
- Adopt dataset factory pattern when metadata is present:
  - `pyarrow.dataset.FileSystemDatasetFactory` with `_metadata` and
    `_common_metadata` support.
- Add a single helper to resolve:
  - partitioning, schema unification, and row-group pruning.
- Ensure dataset scan options are passed through:
  - batch size, readahead, memory pool, and filter pushdown.

**Acceptance**
- Storage snapshot preparation and serving scans share one scan configuration
  surface.
- Schema unification and partitioning are deterministic across both paths.

---

### Phase 2: Unified filter compiler and operator model

**Goals**
- Remove redundant filter logic between storage and serving.
- Centralize filter operator semantics and validation.

**Tasks**
- Move filter compiler into a shared package (or re-export as shared):
  - Candidate location: `src/codeintel/storage/queries/filters.py`
- Replace storage-only filter helpers with shared compiler output:
  - `src/codeintel/storage/queries/expressions.py`
  - `src/codeintel/storage/queries/safe.py`
- Ensure operator constraints align with column types from:
  - `codeintel.core.schemas.primitives`

**Acceptance**
- One canonical filter predicate pipeline used by both storage and serving.
- Invalid operator/type combinations fail at compile time.

---

### Phase 3: SQLGlot AST canonicalization and capability enforcement

**Goals**
- Make query semantics intrinsic through AST normalization and capability
  envelopes.
- Improve projection/pushdown derivation deterministically.

**Tasks**
- Add a canonical AST normalization pass:
  - `src/codeintel/storage/sqlglot_tools.py`
  - `src/codeintel/serving/semantic/sqlglot_query_builder.py`
- Centralize projection and pushdown extraction:
  - `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- Add AST metadata extraction for:
  - tables, columns, and function usage (SQLGlot advanced metadata).

**Acceptance**
- All queries produce a normalized AST before execution.
- Projection columns and filter pushdown are derived from AST, not ad hoc logic.

---

### Phase 4: Streaming hardening (HTTP + MCP)

**Goals**
- Make streaming exports robust, cancellable, and metadata-rich.
- Avoid eager materialization.

**Tasks**
- Expand IPC streaming options and metadata control:
  - `src/codeintel/serving/http/streaming.py`
  - `src/codeintel/core/exports`
- Ensure dataset scan streaming is used for export:
  - `src/codeintel/serving/semantic/kernel.py`
- Add cancellation hooks and limits on batch sizes:
  - `src/codeintel/serving/http/routes/v1/export.py`
  - `src/codeintel/serving/mcp/tools/export.py`

**Acceptance**
- Export streaming uses Arrow batch readers; no full-table materialization.
- Metadata and cancellations are supported for HTTP and MCP exports.

---

### Phase 5: Polars execution control as a downstream tool

**Goals**
- Use Polars only as a downstream tool with explicit optimizer and engine
  controls.

**Tasks**
- Centralize Polars execution settings:
  - `src/codeintel/serving/semantic/engines/polars_engine.py`
- Prefer `collect_batches` or `sink_*` when streaming is enabled.
- Capture plan inspection artifacts (`profile`, `explain`) in diagnostics.

**Acceptance**
- Polars execution is deterministic and does not bypass DuckDB.
- Streaming and optimization choices are explicitly configured.

---

### Phase 6: Arrow conversion and export serialization consolidation

**Goals**
- Ensure a single source of truth for Arrow->row conversion and export encoding.

**Tasks**
- Move batch->row conversion into shared utility:
  - `src/codeintel/storage/query_results.py`
  - `src/codeintel/serving/semantic/kernel.py`
- Ensure export serialization uses shared row coercion:
  - `src/codeintel/serving/export/ndjson.py`

**Acceptance**
- Row conversion and export serialization are consistent across adapters.

---

### Phase 7: FastAPI + FastMCP advanced capabilities

**Goals**
- Make transport-level behavior intrinsic (DI, lifecycle, background tasks).

**Tasks**
- Use `CurrentContext()` injection consistently in MCP tools where possible.
- Add background task orchestration for long-running exports and queries.
- Tighten response models and streaming responses for HTTP routes.

**Acceptance**
- FastAPI and FastMCP behavior is consistent, cancellable, and typed.

## Cross-Cutting Enhancements

- **Observability**: include structured query plan metadata, dataset stats,
  and timing across storage + serving.
- **Robustness**: prefer schema/metadata-driven planning (Parquet metadata +
  DuckDB catalog) to eliminate divergent interpretations.
- **Extensibility**: move shared logic into clearly named modules and re-export
  where needed rather than duplicating logic.

## Risks and Mitigations

- **Risk**: Consolidation causes regressions in existing queries.  
  **Mitigation**: Add integration tests on normalized AST + scan plans and
  compare query fingerprints before/after changes.

- **Risk**: New streaming paths increase complexity in cancellation behavior.  
  **Mitigation**: Add explicit cancel hooks and thread-safe guards in export
  dispatch.

## Suggested Sequencing

1. Phase 0 (inventory + shared primitives)  
2. Phase 1 (dataset scanning + metadata consolidation)  
3. Phase 2 (filter compiler unification)  
4. Phase 3 (AST canonicalization + pushdown)  
5. Phase 4 (streaming hardening)  
6. Phase 5 (Polars controls)  
7. Phase 6 (conversion + export consolidation)  
8. Phase 7 (FastAPI/FastMCP advanced controls)

## Completion Criteria

- Shared scanning, filtering, and schema paths are consolidated.
- AST normalization and pushdown are intrinsic and deterministic.
- Streaming and export paths are bounded-memory and cancellation-aware.
- Storage and serving adhere to a single architecture surface, with minimal
  duplicate logic.
