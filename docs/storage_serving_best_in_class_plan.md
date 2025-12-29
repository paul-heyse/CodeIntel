# Storage + Serving Best-in-Class Enhancement Plan

## Context and alignment

- Aligns with `docs/hamilton_inference_first_alignment_plan.md` for inference-first schemas.
- Aligns with `docs/Make_Hamilton_graph_authoritative_partE.md` for module-based targets.
- Uses SQLGlot for canonical ASTs, Polars + PyArrow for primary columnar execution, and
  DuckDB relational API for complex fallback queries.
- Uses FastMCP as the primary serving surface for structured, streaming responses.

## Goals

- Make SQLGlot AST the canonical query representation across storage and serving.
- Maximize PyArrow and Polars usage for storage scans and serving execution paths.
- Keep complex queries fully programmatic without raw SQL templates.
- Improve observability, guardrails, and debuggability for query execution and data reads.
- Maintain end-to-end alignment with Hamilton-derived schemas and module provenance.

## Non-goals

- No static SQL templates for serving paths (DDL exceptions only).
- No eager materialization in serving flows unless explicitly required by contracts.
- No hard blocking on schema drift (observe and report instead).

## Decisions

- SQLGlot AST is the single source of truth for query semantics.
- Polars is the default execution engine when AST fits its envelope.
- DuckDB uses the relational API only and is a secondary fallback engine.
- Arrow schema metadata carries view_id, repo, commit, and schema_hash where relevant.
- FastMCP responses stream Arrow IPC where feasible and return structured metadata.

## Phase 0: Baseline inventory and boundaries

Deliverables
- Inventory of current storage and serving flows and their engine usage.
- Explicit feature envelope definitions for Polars vs DuckDB.
- Deprecation list for any raw-SQL view paths or unused helpers.

Work items
- Map current call paths in:
  - `src/codeintel/serving/semantic/*`
  - `src/codeintel/storage/views/*`
  - `src/codeintel/storage/datasets/*`
  - `src/codeintel/serving/mcp/*`
- Document current AST usage in:
  - `src/codeintel/serving/semantic/query_ast.py`
  - `src/codeintel/storage/sqlglot_tools.py`
- Record current query routing logic in:
  - `src/codeintel/serving/semantic/routing.py`

Acceptance criteria
- Inventory doc embedded in this plan under a short "Current State" section.
- Clear list of subsystems to be updated in later phases.

## Phase 1: Canonical SQLGlot AST pipeline

Deliverables
- Single canonical AST representation for every semantic query.
- Standardized AST fingerprint, diff, and lineage tooling in serving.

Work items
- Route all query planning through `ServingQuery` AST:
  - `src/codeintel/serving/semantic/query_ast.py`
  - `src/codeintel/serving/semantic/planner.py`
- Extend AST normalization and hashing to use:
  - `src/codeintel/storage/sqlglot_tools.py`
  - `src/codeintel/serving/semantic/fingerprints.py`
- Add semantic diff support for observability:
  - Use `semantic_diff_sql_duckdb` from `src/codeintel/storage/sqlglot_tools.py`
- Add AST lineage extraction for view dependencies:
  - Use `extract_column_lineage_duckdb` and `extract_table_keys_duckdb`

Acceptance criteria
- All serving paths produce a SQLGlot AST and a canonical fingerprint.
- AST diffs are visible in logs for query changes.

## Phase 2: Polars execution compiler and envelope

Deliverables
- AST-to-Polars compiler is the primary execution path.
- Expanded Polars feature envelope with explicit guardrails.

Work items
- Make AST the input to Polars compilation:
  - `src/codeintel/serving/semantic/polars_query_builder.py`
  - `src/codeintel/serving/semantic/engines/polars_engine.py`
- Expand AST support coverage:
  - Filters, ordering, limits, projections, basic functions
  - Row index injection via `with_row_count` where needed
- Add query envelope checks that are AST-driven:
  - `src/codeintel/serving/semantic/routing.py`
- Use Polars lazy scan APIs:
  - `scan_parquet` and `scan_ipc` with pushdown when available
  - `collect_all` for shared subplans
  - `sink_parquet` for materialization paths

Acceptance criteria
- Polars executes all ASTs that fit the defined envelope.
- Eager materialization is detected and logged when it occurs.

## Phase 3: DuckDB relational fallback (no raw SQL)

Deliverables
- DuckDB fallback runs exclusively via the relational API.
- Arrow-first data registration and streaming reads.

Work items
- Update relational builder to accept SQLGlot AST:
  - `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- Map AST constructs to relation operations where possible:
  - select, filter, project, join, limit, order
- Register Arrow datasets and RecordBatchReaders directly:
  - `src/codeintel/storage/duckdb/context.py`
  - `src/codeintel/storage/duckdb/catalog.py`
- Use `fetch_arrow_reader` or `arrow()` for streaming:
  - avoid `fetchdf` or full materialization

Acceptance criteria
- DuckDB paths operate without raw SQL templates.
- DuckDB output can be consumed as Arrow IPC streams.

## Phase 4: Schema and metadata discipline

Deliverables
- Arrow schema metadata is consistently propagated.
- Explicit mapping for complex types across Arrow, Polars, DuckDB.

Work items
- Centralize schema mapping rules:
  - `src/codeintel/storage/schema/arrow_schema.py`
  - `src/codeintel/storage/duckdb_types.py`
- Ensure schema unification for merges and joins:
  - Use `pyarrow.unify_schemas` where appropriate
- Embed and round-trip metadata:
  - repo, commit, view_id, schema_hash in `pa.Schema.metadata`
- Extend TableSchema conversion helpers:
  - `src/codeintel/storage/schema_roundtrip.py`

Acceptance criteria
- Metadata round-trips for Arrow, Polars, and DuckDB.
- Complex types (struct, list, map) are stable across engines.

## Phase 5: Storage scan and IPC streaming improvements

Deliverables
- Scanner-first dataset reads with pushdown and batch control.
- IPC streaming defaults that preserve schema metadata.

Work items
- Add scanner options and helpers:
  - `src/codeintel/storage/datasets/arrow_store.py`
  - `src/codeintel/storage/datasets/manifests.py`
- Propagate scan options through serving:
  - `src/codeintel/serving/semantic/datasets.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
- Standardize IPC streaming options:
  - `src/codeintel/core/columnar/stream.py`
  - `src/codeintel/serving/http/streaming.py`

Acceptance criteria
- Dataset scans never require `to_table()` in serving paths.
- IPC streaming preserves schema metadata and batch sizing.

## Phase 6: Serving and FastMCP enhancements

Deliverables
- Structured, typed MCP responses with consistent resources.
- Streaming and cancellation behavior aligned with dataset scanning.

Work items
- Add structured MCP models for query responses:
  - `src/codeintel/serving/mcp/models/*`
- Add typed tools and resources:
  - `src/codeintel/serving/mcp/tools/*`
  - `src/codeintel/serving/mcp/resources/*`
- Integrate middleware for observability and error handling:
  - `src/codeintel/serving/mcp/middleware_stack.py`
  - `src/codeintel/serving/mcp/middleware_errors.py`
- Ensure streaming endpoints use IPC readers:
  - `src/codeintel/serving/mcp/runtime.py`

Acceptance criteria
- MCP streaming returns Arrow IPC with metadata.
- Structured outputs include query fingerprint and engine choice.

## Phase 7: Observability and guardrails

Deliverables
- Engine routing telemetry and scan performance metrics.
- Guardrails for query complexity and materialization risks.

Work items
- Add routing metrics and logging:
  - `src/codeintel/serving/semantic/guardrails.py`
  - `src/codeintel/serving/metrics.py`
- Emit scan and batch telemetry:
  - `src/codeintel/storage/datasets/arrow_store.py`
  - `src/codeintel/serving/semantic/datasets.py`
- Add guardrails to detect eager collection:
  - Polars engine and DuckDB fallback

Acceptance criteria
- Logs include engine decision, batch sizes, scan time, and row counts.
- Guardrails warn on eager materialization.

## Phase 8: Tests and validation

Deliverables
- Focused unit tests for AST routing and engine compilation.
- Integration tests for IPC streaming and metadata preservation.

Work items
- AST routing tests:
  - `tests/serving/semantic/test_routing.py`
- Polars and DuckDB compilation tests:
  - `tests/serving/semantic/test_polars_engine.py`
  - `tests/serving/semantic/test_duckdb_engine.py`
- IPC streaming tests:
  - `tests/serving/semantic/test_ipc_stream.py`
  - `tests/_helpers/columnar_streams.py`

Acceptance criteria
- All serving tests pass under `tests/serving`.
- Metadata keys are verified for view_id, repo, commit, schema_hash.

## Phase 9: Decommissioning and cleanup

Deliverables
- Removal of raw-SQL view paths and deprecated helpers.
- Simplified query execution paths with a single AST pipeline.

Work items
- Remove unused SQL view map or legacy adapters:
  - `src/codeintel/storage/views/view_sql_map.json`
  - `src/codeintel/storage/views/sqlglot_views.py` (if replaced)
- Remove redundant spec-based query paths if AST is canonical:
  - `src/codeintel/serving/semantic/polars_query_builder.py`

Acceptance criteria
- No raw SQL view templating remains in serving paths.
- Query compilation path is single-source and test-covered.

## Checklist

- [ ] Phase 0 inventory and current state notes added to this doc.
- [ ] AST pipeline fully canonicalized and fingerprinted.
- [ ] Polars compiler supports the defined envelope.
- [ ] DuckDB relational fallback wired and tested.
- [ ] Schema metadata and complex types validated end-to-end.
- [ ] Scanner-first storage reads in serving and MCP.
- [ ] MCP structured streaming with IPC and metadata.
- [ ] Observability and guardrails in place.
- [ ] Tests updated and passing for serving coverage.
- [ ] Legacy SQL view paths fully removed.
