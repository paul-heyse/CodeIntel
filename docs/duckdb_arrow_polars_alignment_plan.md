# DuckDB + Arrow + Polars Alignment Plan (Serving-First)

## Purpose
Deliver a pyarrow + polars–first data plane while preserving (and improving) complex
query serving. DuckDB remains the complex-query engine, but is fed by Arrow/Polars
objects and returns Arrow IPC streams. All query compilation is programmatic; no
hand-written SQL in serving paths.

## Decisions (Accepted)
- Arrow is the canonical interchange format (schemas + IPC + Parquet).
- Polars is the primary compute engine for simple/medium queries and columnar work.
- DuckDB is the complex-query engine and always operates via the relation API.
- SQLGlot is the canonical query AST for routing and compilation.
- Streaming IPC is the serving boundary; avoid eager `to_table()` / `rel.arrow()` for
  large results.

## Non-Goals
- Re-introducing raw SQL as a primary interface.
- Building a parallel query engine in Python without Polars or DuckDB.
- Full SQL coverage in Polars (DuckDB handles the complex subset).

## Target Architecture
### Data Plane
- **Storage**: Parquet/IPC datasets with Arrow schemas and metadata.
- **Schema**: TableSchema is the source of truth; Arrow schema mirrors TableSchema.
- **Ingestion**: Produce Arrow/Polars directly; avoid pandas conversions.

### Query Plane
- **AST**: SQLGlot AST is the canonical query shape.
- **Router**: Decide Polars vs DuckDB based on AST features and cost heuristics.
- **Compiler**:
  - Polars: AST → `pl.Expr` + `LazyFrame` plan.
  - DuckDB: AST → relation API calls; SQLGlot SQL only for features not exposed
    by relation API (last resort, still generated).

### Serving Plane
- **IPC streaming**: All query results stream as Arrow IPC.
- **Metadata**: Per-batch metadata includes repo/commit, schema hash, and view id.

## Capability Matrix (Routing)
### Route to Polars by Default
- Projections, filters, aggregations (groupby), simple joins, simple sort/limit.
- Pure columnar transforms and window-less analytics.
- Queries that can stay within a single dataset or simple multi-dataset joins.

### Route to DuckDB
- Complex joins (multi-join graphs with non-equi predicates).
- Window functions, subqueries, CTEs, correlated filters.
- Queries that require complex SQL semantics not available in Polars.

## Implementation Plan
### Phase 0: Shared Infrastructure
- **Query AST**: Define a SQLGlot-backed `ServingQuery` AST structure used across
  serving and tests.
- **Router**: Implement a deterministic routing function with a capability matrix.
- **Schema plumbing**: Ensure TableSchema ↔ Arrow schema conversion is authoritative.

### Phase 1: Arrow-First Data Plane
- **Dataset store**: Ensure all datasets can be represented as Arrow scanners and
  Polars lazy sources.
- **Partitioning**: Standardize Hive-style partitions (repo/commit/target).
- **Schema metadata**: Persist Arrow schema metadata (schema hash, column order).

### Phase 2: Polars Query Engine
- **Compiler**: SQLGlot AST → Polars expressions + LazyFrame plan.
- **Execution**: Use `scan_*` sources to maximize predicate/projection pushdown.
- **Streaming**:
  - For IPC: `collect_batches()` → RecordBatch stream.
  - For storage: `sink_parquet` / `sink_ipc` for out-of-core writes.
- **Reuse**: Use `collect_all` for multi-output queries to enable CSE.

### Phase 3: DuckDB Query Engine
- **Source relations**:
  - Arrow: `con.register("t", scanner)` or `con.from_arrow(table)`.
  - Polars: `con.register("t", lazyframe)` only at explicit boundaries.
- **Relational API**:
  - `.filter`, `.project`, `.join`, `.aggregate`, `.order`.
  - Materialize only for persistence, not serving.
- **Streaming**: `rel.fetch_arrow_reader()` → Arrow IPC stream.
- **UDFs**: Use Arrow-vectorized UDFs (`type="arrow"`) for custom compute when needed.

### Phase 4: IPC Streaming Unification
- **IPC writer**: Standardize on `pyarrow.ipc.new_stream` with per-batch metadata.
- **Reader compatibility**: Use `RecordBatchReader.from_stream` for inbound IPC.
- **Memory control**: Prefer buffered stream readers over full table materialization.

### Phase 5: Schema + Type Discipline
- **Arrow schema unification**: `pyarrow.unify_schemas` on merge paths.
- **DuckDB types**: Explicit `duckdb.sqltypes` for DECIMAL/TIMESTAMP/STRUCT/LIST/MAP.
- **Polars dtypes**: Use `collect_schema()` at plan boundaries where required.
- **Metadata round-trip**: Maintain schema hash + version in IPC metadata.

### Phase 6: Observability and Guardrails
- **Profiling**: DuckDB `EXPLAIN ANALYZE`, Polars `profile`/`show_graph`.
- **Metrics**: Track route decisions (Polars vs DuckDB), scan sizes, batch sizes.
- **Guardrails**: Detect eager materialization in serving code paths.

## Integration Points (Current Code)
- Serving kernel and query compiler layers.
- Columnar IPC streaming (`src/codeintel/core/columnar/stream.py`).
- Dataset store + schema services (TableSchema, Arrow conversions).
- View and schema inference paths (ensure Arrow/Polars-first derivation).

## Testing and Validation
- **Unit**: AST → Polars/DuckDB compilers, router decisions, schema conversions.
- **Contract**: Ensure view schemas derived from Arrow/Polars match TableSchema.
- **IPC**: Round-trip tests for streaming with metadata.
- **Parity**: Golden queries executed via Polars and DuckDB produce identical results.

## Acceptance Criteria
- All serving queries are compiled from SQLGlot AST; no hand-written SQL.
- Polars path covers simple/medium queries with streaming IPC output.
- DuckDB path handles complex queries and streams IPC without eager materialization.
- TableSchema ↔ Arrow ↔ DuckDB types align with no drift across pipelines.
- Observability captures route decisions and batch-level performance metrics.

## Risks and Mitigations
- **Polars gaps**: router falls back to DuckDB with deterministic reasoning.
- **Type drift**: enforce explicit type mapping and Arrow schema unification.
- **Memory pressure**: prefer stream readers/sinks and avoid `to_table()` in serving.

