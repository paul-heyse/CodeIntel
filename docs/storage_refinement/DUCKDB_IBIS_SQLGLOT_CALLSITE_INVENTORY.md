# DuckDB / Ibis / SQLGlot Callsite Inventory

This inventory captures the primary SQL compilation and execution callsites
across storage and serving, with recommended canonical surfaces.

## Canonical SQL pipeline usage (SQLGlot)

- `src/codeintel/storage/sqlglot_tools.py`
  - Canonical parse/normalize/qualify/optimize/render pipeline.
  - Stable fingerprints and table/column lineage extraction.
- `src/codeintel/storage/views/diff.py`
  - Routes diffing through canonical SQL pipeline.
- `src/codeintel/storage/views/materialization.py`
  - Uses SQLGlot lineage helpers to persist derived lineage metadata.

**Recommendation**: new SQLGlot usage should enter via `sqlglot_tools` rather
than inline parsing/optimization.

## Safe SQL ingress (select-only perimeter)

- `src/codeintel/storage/queries/safe.py`
  - `assert_select_perimeter` + AST policy enforcement.
- `src/codeintel/serving/semantic/kernel.py`
  - Validates raw SQL execution (compiled semantic queries).
- `src/codeintel/serving/semantic/planner.py`
  - Validates compiled SQL for semantic plans.
- `src/codeintel/serving/mcp/resources/meta.py`
  - Validates view SQL artifacts before returning them.
- `src/codeintel/storage/views/ibis_views.py`
  - Validates raw SQL in view helpers.

**Recommendation**: any new raw SQL ingress should pass through
`assert_select_perimeter` with an explicit `SqlIngressPolicy`.

## Ibis query compilation/execution

- `src/codeintel/serving/semantic/query_builder.py`
  - Parameterized query construction; IN-list values staged via memtables.
- `src/codeintel/serving/semantic/templates.py`
  - Query templates with param binding for compile and execution.
- `src/codeintel/storage/ibis_adapter.py`
  - SQLGlot-backed writes (INSERT/UPSERT) for Ibis expressions, DataFrames,
    and tuple batches.
- `src/codeintel/storage/repositories/base.py`
  - Ibis expressions executed and converted to DataFrames.

**Recommendation**: prefer Ibis expressions + `QueryTemplate`/`BoundQuery` for
semantic querying and avoid string-based SQL assembly for filters or values.

## DuckDB direct SQL execution (parameterized)

- `src/codeintel/storage/duckdb_policy_backend.py`
  - DDL, DELETE, and mutation operations (SQLGlot-generated + parameterized).
- `src/codeintel/storage/gateway/accessors.py`
  - `execute(sql, params)` pass-through for gateway operations.
- `src/codeintel/storage/metadata/*`
  - Metadata bootstrap/sync SQL uses parameterized queries.
- `src/codeintel/storage/tracking/*`
  - Tracking tables use parameterized SQL for reads/writes.

**Recommendation**: continue using parameterized `execute` with explicit
placeholders; avoid string interpolation of values.

## Build/serving export paths

- `src/codeintel/build/exports/*`
  - SQL for export helpers (relation compilation + parameterized SQL where
    applicable).
- `src/codeintel/serving/search/engine.py`
  - DB-API templates with bound parameters for search queries.

**Recommendation**: keep DB-API usage in `DbApiTemplate` and avoid ad-hoc
SQL templating.
