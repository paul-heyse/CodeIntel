# Storage + Serving Additional Opportunities Implementation Plan

## Context

This plan captures new opportunities identified after completing
`docs/storage_serving_architecture_alignment_plan.md` and
`docs/storage_serving_architecture_alignment_enhancements_plan.md`.
It focuses on deeper consolidation, intrinsic correctness, and advanced
capability usage across DuckDB, SQLGlot, PyArrow, Polars, FastAPI, and FastMCP.

## Goals

- Make correctness intrinsic (invalid states are unrepresentable).
- Remove remaining raw SQL construction and string interpolation.
- Centralize type handling across DuckDB, Arrow, and Polars.
- Use streaming-first execution across HTTP and MCP.
- Reduce duplication by sharing query planning, scanning, and transport
  primitives.

## Guiding principles

- Prefer SQLGlot AST and DuckDB Expression API over SQL strings.
- Use DuckDB Types API for complex/nested types.
- Do not materialize large results in memory.
- Keep storage and serving on a single shared planning surface.

## Workstreams and phases

### Phase 1: Intrinsic SQL construction and DDL unification

**Item 1.1: Replace remaining f-string SQL with AST or parameterized SQL**

- **Pattern**: SQLGlot AST + `render_sql_duckdb` with placeholders.

```python
expr = (
    exp.select(exp.Literal.number(1))
    .from_(table_expr_from_ref("metadata.table_schema_registry"))
    .where(exp.EQ(this=exp.column("schema_name"), expression=exp.Placeholder()))
)
con.execute(render_sql_duckdb(expr), [schema_name])
```

- **Targets**: `src/codeintel/storage/schema/sqlglot_ddl.py`,
  `src/codeintel/storage/gateway/factory.py`.
- **Acceptance**: no new SQL strings with interpolated identifiers.

**Item 1.2: Use DuckDB Relation API for view/materialization/export**

- **Pattern**: Relation-first operations, no SQL rendering.

```python
relation = con.table("metadata.schema_validation_runs")
(
    relation.filter("status = 'failed'")
    .project("repo", "commit", "created_at")
    .create_view("metadata.v_schema_validation_failures")
)
```

- **Targets**: `src/codeintel/storage/metadata/views.py`,
  `src/codeintel/storage/metadata/ddl.py`,
  `src/codeintel/storage/duckdb_policy_backend.py`.
- **Acceptance**: views and materializations use relation API where possible.

**Item 1.3: Parameterized SQL for unavoidable raw SQL**

- **Pattern**: DB-API placeholders only.

```python
con.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = ?", [schema])
```

- **Targets**: `src/codeintel/storage/duckdb/catalog.py`,
  `src/codeintel/storage/metadata/meta_catalog.py`.
- **Acceptance**: no formatted SQL strings with user-provided values.

---

### Phase 2: Type system unification and complex types

**Item 2.1: Extend complex type mapping to use DuckDB Types API**

- **Pattern**: `duckdb.sqltypes` + constructors for LIST/MAP/STRUCT/DECIMAL.

```python
duckdb.list_type(duckdb.sqltypes.INTEGER)
duckdb.struct_type({"x": duckdb.sqltypes.INTEGER, "y": duckdb.sqltypes.VARCHAR})
duckdb.map_type(duckdb.sqltypes.VARCHAR, duckdb.sqltypes.BIGINT)
duckdb.decimal_type(18, 4)
```

- **Targets**: `src/codeintel/storage/duckdb_types.py`,
  `src/codeintel/core/schemas/type_mappings.py`.
- **Acceptance**: complex types are represented by DuckDBPyType objects,
  not string literals.

**Item 2.2: Use type objects in DDL and projections**

- **Pattern**: pass `DuckDBPyType` values to DDL and cast expressions.

```python
dtype = duckdb.decimal_type(18, 4)
con.execute("CREATE TABLE t(price $1)", [dtype])
expr = FunctionExpression("cast", ColumnExpression("price"), ConstantExpression(dtype))
```

- **Targets**: `src/codeintel/storage/schema/sqlglot_ddl.py`,
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`.
- **Acceptance**: complex type casts and DDL use type objects.

**Item 2.3: Standardize nested type handling across Arrow and Polars**

- **Pattern**: map Arrow nested types to Polars using a single mapping table.

```python
mapping = complex_type_mapping(column_type)
arrow_type = mapping.arrow_type
polars_type = mapping.polars_type
```

- **Targets**: `src/codeintel/core/schemas/type_mappings.py`,
  `src/codeintel/core/schemas/arrow_gen.py`.
- **Acceptance**: nested types remain consistent end-to-end.

---

### Phase 3: Arrow dataset scanning and metadata

**Item 3.1: Persist richer scan metadata into manifests**

- **Pattern**: read Parquet metadata and store row-group stats and
  dictionary encoding in manifest extras.

```python
parquet_file = pq.ParquetFile(path)
row_groups = parquet_file.metadata.num_row_groups
```

- **Targets**: `src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/storage/datasets/manifest_index.py`.
- **Acceptance**: manifests include row-group stats and encoding hints.

**Item 3.2: Use fragment-level pruning and scanner from fragments**

- **Pattern**: `fragment.subset` + `Scanner.from_fragments`.

```python
fragments = tuple(dataset.get_fragments(filter=filter_expr))
pruned = tuple(frag.subset(filter_expr) for frag in fragments if frag is not None)
scanner = ds.Scanner.from_fragments(pruned, schema=schema, **scan_kwargs)
```

- **Targets**: `src/codeintel/storage/datasets/scanning.py`.
- **Acceptance**: filters prune row-groups before scanning.

**Item 3.3: Dataset factory for _metadata/_common_metadata**

- **Pattern**: prefer dataset factories when metadata exists.

```python
ds.parquet_dataset(str(metadata_path), partitioning=partitioning)
```

- **Targets**: `src/codeintel/storage/datasets/scanning.py`.
- **Acceptance**: schema resolution is deterministic across storage and serving.

---

### Phase 4: SQLGlot semantic tooling

**Item 4.1: Type-aware SQLGlot optimization**

- **Pattern**: run `optimize` with schema derived from DuckDB contracts.

```python
optimized = optimize(expr, schema={"core.table": {"col": "VARCHAR"}})
```

- **Targets**: `src/codeintel/storage/sqlglot_tools.py`,
  `src/codeintel/serving/semantic/sqlglot_query_builder.py`.
- **Acceptance**: canonicalization is stable and type-aware.

**Item 4.2: Semantic diff for view/query evolution**

- **Pattern**: `sqlglot.diff` on normalized ASTs.

```python
diff = semantic_diff(old_expr, new_expr)
```

- **Targets**: `src/codeintel/storage/tracking/schema_catalog.py`.
- **Acceptance**: diffs are recorded structurally, not by SQL text.

**Item 4.3: DuckDB-specific dialect for allowlists**

- **Pattern**: custom SQLGlot dialect generator for allowed constructs only.

```python
class DuckDBSafe(Dialect):
    class Generator(Dialect.Generator):
        TRANSFORMS = {...}
```

- **Targets**: `src/codeintel/storage/sqlglot_tools.py`.
- **Acceptance**: unsupported SQL forms are unrepresentable by design.

**Status (completed)**

- Wired type-aware canonicalization using schema mappings in
  `src/codeintel/storage/sqlglot_tools.py`,
  `src/codeintel/serving/semantic/sqlglot_query_builder.py`,
  `src/codeintel/serving/semantic/query_ast.py`.
- Added DuckDB-safe dialect rendering to make unsupported constructs intrinsic.
- Emitted view SQL maps and structural diffs in
  `src/codeintel/storage/tracking/schema_catalog.py`,
  `src/codeintel/build/hamilton/executor.py`,
  `src/codeintel/build/hamilton/native/export/serving_artifacts.py`.

---

### Phase 5: Polars streaming and execution control

**Item 5.1: Use Polars streaming primitives for NDJSON exports**

- **Pattern**: `collect_batches` or `sink_batches` to avoid full materialization.

```python
for batch in lazy_frame.collect_batches(streaming=True):
    yield from iter_ndjson_bytes(batch.to_dicts())
```

- **Targets**: `src/codeintel/serving/export/ndjson.py`,
  `src/codeintel/serving/semantic/engines/polars_engine.py`.
- **Acceptance**: NDJSON export never materializes full results in memory.

**Item 5.2: Expose QueryOptFlags and determinism controls**

- **Pattern**: pass optimizer flags from settings into Polars plans.

```python
lazy_frame.collect(optimizations=query_opt_flags, maintain_order=True)
```

- **Targets**: `src/codeintel/serving/settings.py`,
  `src/codeintel/serving/semantic/engines/polars_engine.py`.
- **Acceptance**: deterministic behavior is configuration-driven.

**Status (completed)**

- Added batch-level NDJSON streaming and record-batch export surface in
  `src/codeintel/serving/export/ndjson.py`,
  `src/codeintel/serving/http/streaming.py`,
  `src/codeintel/serving/http/export_dispatch.py`,
  `src/codeintel/serving/semantic/kernel.py`.
- Exposed `polars_maintain_order` settings and applied it to batch collection in
  `src/codeintel/core/config/settings.py`,
  `src/codeintel/core/runtime/loader.py`,
  `src/codeintel/serving/semantic/engines/polars_engine.py`.

---

### Phase 6: FastAPI + FastMCP transport hardening

**Item 6.1: Use Annotated dependencies with request scope for streaming**

- **Pattern**: `Annotated` + `Depends(..., scope="request")`.

```python
from typing import Annotated

SessionDep = Annotated[Session, Depends(get_session, scope="request")]
```

- **Targets**: `src/codeintel/serving/http/dependencies.py`,
  `src/codeintel/serving/http/routes/v1/*.py`.
- **Acceptance**: resources live through streaming response completion.

**Item 6.2: FastMCP prompts with PromptResult and typed args**

- **Pattern**: prompt templates for query guidance.

```python
@mcp.prompt(name="semantic_query_help")
def semantic_query_help(view_id: str) -> PromptResult:
    return PromptResult(messages=[Message(f"Query view {view_id}")])
```

- **Targets**: `src/codeintel/serving/mcp/prompts.py`.
- **Acceptance**: prompts provide structured, typed guidance for clients.

**Item 6.3: Standardized progress and cancellation for MCP tools**

- **Pattern**: `TaskConfig` + `ctx.report_progress` + cancel token.

```python
await ctx.report_progress(progress=50, total=100)
cancel_token.raise_if_cancelled()
```

- **Targets**: `src/codeintel/serving/mcp/tools/*.py`.
- **Acceptance**: all long-running MCP tools share a unified protocol.

**Status (completed)**

- Applied request-scoped FastAPI dependencies in
  `src/codeintel/serving/http/dependencies.py` and
  `src/codeintel/serving/http/routes/v1/*.py`.
- Added typed prompt guidance (PromptResult) including `semantic_query_help` in
  `src/codeintel/serving/mcp/prompts.py`.
- Standardized MCP tool task/progress/cancel patterns in
  `src/codeintel/serving/mcp/tools/catalog.py`,
  `src/codeintel/serving/mcp/tools/describe.py`,
  `src/codeintel/serving/mcp/tools/explain.py`,
  `src/codeintel/serving/mcp/tools/search.py`,
  `src/codeintel/serving/mcp/tools/meta.py`.

---

### Phase 7: Intrinsic correctness surfaces

**Item 7.1: SafeRelation / ResultStream interface**

- **Pattern**: expose only readers, not `fetchall` or eager methods.

```python
class ResultStream(Protocol):
    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader: ...
```

- **Targets**: `src/codeintel/storage/protocols/duckdb_export.py`,
  `src/codeintel/serving/semantic/engines/duckdb_engine.py`.
- **Acceptance**: eager materialization is unrepresentable in the API surface.

**Item 7.2: Contract-typed projections in query construction**

- **Pattern**: cast to contract types in the AST before execution.

```python
expr = exp.Cast(this=exp.column("amount"), to=exp.DataType.build("DECIMAL(18,2)"))
```

- **Targets**: `src/codeintel/serving/semantic/duckdb_relation_builder.py`,
  `src/codeintel/serving/semantic/kernel.py`.
- **Acceptance**: output schema always matches contract types.

**Item 7.3: Shared QueryPlanSpec for storage and serving**

- **Pattern**: one plan object with scan options, filters, and projections.

```python
@dataclass(frozen=True, slots=True)
class QueryPlanSpec:
    table_key: str
    columns: tuple[str, ...]
    filter_expression: ds.Expression | None
```

- **Targets**: `src/codeintel/storage/datasets/scanning.py`,
  `src/codeintel/serving/semantic/query_ast.py`.
- **Acceptance**: storage and serving share a single planning surface.

**Item 7.4: Schema-driven operator allowlists**

- **Pattern**: operators derived from column types, not hard-coded lists.

```python
allowed_ops = ops_for_column_type(column.type)
```

- **Targets**: `src/codeintel/core/filters.py`,
  `src/codeintel/storage/queries/filter_compiler.py`.
- **Acceptance**: invalid operator/type pairs are unrepresentable.

---

## Sequencing and dependencies

1. Phase 1 (intrinsic SQL + DDL) to eliminate unsafe SQL surfaces.
2. Phase 2 (type system) to enable contract-typed projections and complex types.
3. Phase 3 (scan metadata + pruning) for deterministic performance.
4. Phase 4 (SQLGlot optimizer + diff) for semantic consistency.
5. Phase 5 (Polars streaming) for bounded-memory exports.
6. Phase 6 (FastAPI/FastMCP) for transport hardening.
7. Phase 7 (intrinsic correctness) to lock in the final interfaces.

## Acceptance criteria

- No new raw SQL strings with interpolated identifiers or values.
- Complex types are constructed with DuckDB Types API, not string literals.
- Query planning and scanning share a single API surface across storage/serving.
- Streaming responses never materialize full results in memory.
- FastAPI and FastMCP share consistent cancellation and progress semantics.

## Decommission notes

- Remove any remaining helpers that allow ad hoc SQL string construction.
- Decommission any eager result APIs that bypass streaming readers.
- Remove legacy operator lists once schema-driven allowlists are in place.
**Status (completed)**: All legacy files listed below have been decommissioned and removed,
and imports were updated to their replacements.

## Legacy and deletion targets (file-level, decommissioned)

These files become legacy once the corresponding migration step is complete.
Delete only after verifying no remaining imports/usages.

- `src/codeintel/serving/semantic/filter_compiler.py`
  - **Replacement**: import directly from
    `src/codeintel/storage/queries/filter_compiler.py`.
  - **Status**: deleted; serving modules use the storage compiler directly.
- `src/codeintel/serving/semantic/filter_ops.py`
  - **Replacement**: schema-driven allowlists in
    `src/codeintel/core/filters.py` and
    `src/codeintel/storage/queries/filter_compiler.py`.
  - **Status**: deleted; no callers rely on serving-local operator validation.
- `src/codeintel/storage/protocols/duckdb_export.py`
  - **Replacement**: `ResultStream`/`SafeRelation` protocol from Phase 7
    (`src/codeintel/storage/protocols/duckdb_relation.py`).
  - **Status**: deleted; export/query paths use streaming readers only.
- `src/codeintel/serving/semantic/datasets.py`
  - **Replacement**: direct imports from
    `src/codeintel/storage/datasets/manifest_index.py` (or new shared module).
  - **Status**: deleted; serving modules import the storage manifest index directly.
- `src/codeintel/storage/schema/ddl.py`
  - **Replacement**: relation-first schema creation or a single canonical DDL
    module post-Phase 1.
  - **Status**: deleted; no call sites rely on SQL-rendered schema DDL.
- `src/codeintel/storage/schema/sqlglot_ddl.py`
  - **Replacement**: relation-first schema/index creation or consolidated DDL
    builder using DuckDB Types API.
  - **Status**: deleted; schema/index DDL is fully centralized elsewhere.
- `src/codeintel/storage/metadata/views.py`
  - **Replacement**: relation-first view builder (Phase 1).
  - **Status**: deleted; metadata view creation is handled by the new
    relation-based implementation.
