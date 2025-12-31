# Raw SQL Deprecation Map and Migration Guide

This document lists raw SQL usage scheduled for deprecation and provides
migration guidance for downstream consumers. The goal is to eliminate
hand-written SQL in query paths in favor of SQLGlot ASTs, DuckDB relation
operations, and the DuckDB Expression API.

## Deprecation policy

- DDL and bootstrap SQL are allowed to remain (schema creation, metadata
  tables, PRAGMA, extension installation).
- Query templates and parameter interpolation are deprecated and should
  be replaced with programmatic builders.
- SQLGlot-generated SQL strings are acceptable only for DDL until
  equivalent relation APIs are available.

## Deprecation map (query paths)

### Serving

- `src/codeintel/serving/search/engine.py`
  - Pattern: SQL templates for FTS and LIKE search.
  - Replacement: SQLGlot AST + DuckDB Expression API relations.
  - Notes: Move to a `SearchQueryBuilder` that returns a relation.

- `src/codeintel/serving/semantic/kernel.py`
  - Pattern: AST rendered to SQL for explain output only.
  - Replacement: Keep for explain output, but do not execute.

### Storage

- `src/codeintel/storage/helpers/sql_params.py`
  - Pattern: manual `$param` interpolation.
  - Replacement: remove in favor of relation APIs or SQLGlot AST builders.

- `src/codeintel/storage/repositories/data_models.py`
  - Pattern: `render_sql` with manual interpolation.
  - Replacement: build relation using Expression API or relation filters.

- `src/codeintel/storage/exports/service.py`
  - Pattern: raw SQL strings for export relation and audit insert.
  - Replacement: use DuckDBPolicyBackend for insert; use relation builders
    for export relations.

- `src/codeintel/storage/queries/safe.py`
  - Pattern: `gateway.con.execute(sql)` with raw SQL strings.
  - Replacement: SQLGlot AST to relation plan or Expression API relations.

## Allowed raw SQL (DDL and bootstrap only)

The following categories are allowed until relation APIs exist:

- Schema creation and bootstrap DDL (metadata tables, schema sync).
- Extension install/load, PRAGMA, and attach/detach operations.
- Database export/import commands.

These should remain confined to:

- `src/codeintel/storage/backend/duckdb_session.py`
- `src/codeintel/storage/metadata/bootstrap.py`
- `src/codeintel/storage/metadata/ddl.py`
- `src/codeintel/storage/metadata/meta_catalog.py`

## Migration guide for downstream consumers

### 1) Replace `con.sql("SELECT ...")` with relation builders

Before

```python
relation = con.sql("SELECT * FROM core.modules WHERE repo = ?", [repo])
```

After

```python
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression

relation = con.table("core.modules")
relation = relation.filter(ColumnExpression("repo") == ConstantExpression(repo))
```

### 2) Replace manual SQL interpolation with relation filters

Before

```python
sql = "SELECT * FROM core.goids WHERE repo = $repo"
relation = con.sql(render_sql(sql, {"repo": repo}))
```

After

```python
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression

relation = con.table("core.goids")
relation = relation.filter(ColumnExpression("repo") == ConstantExpression(repo))
```

### 3) Prefer policy backends for inserts and updates

Before

```python
con.execute("INSERT INTO metadata.export_audit (...) VALUES (?, ?, ?)", params)
```

After

```python
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

backend = DuckDBPolicyBackend(gateway)
backend.bulk_insert("metadata.export_audit", rows, columns=[...])
```

### 4) Use semantic AST builder for query execution

Before

```python
sql = "SELECT ..."  # ad hoc
relation = con.sql(sql)
```

After

```python
from codeintel.serving.semantic.query_ast import build_serving_query

serving_query = build_serving_query(spec=spec)
relation = build_relation_plan(con=con, spec=spec, ast=serving_query.ast, context=context)
```

## Timeline

- Phase 1: Deprecate manual SQL interpolation and query templates in
  serving and storage.
- Phase 2: Replace remaining query templates with relation builders.
- Phase 3: Remove deprecated helpers and update downstream callers.

## Support

If you rely on deprecated query templates, open a migration issue with:

- The module and function you are calling.
- The SQL pattern you rely on.
- The target relation or AST builder you want to use.
