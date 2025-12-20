# Storage SQL Pipeline, Lineage, and Export/Import

This document captures the new canonical SQL pipeline, lineage exposure, and
export/import surfaces introduced by the advanced DuckDB/Ibis/SQLGlot work.

## Canonical SQL pipeline

Use the storage-owned SQLGlot helpers to keep parsing/qualification/optimization
behavior consistent across storage and serving:

- `codeintel.storage.sqlglot_tools.parse_one_duckdb`
- `codeintel.storage.sqlglot_tools.canonicalize_expression_duckdb`
- `codeintel.storage.sqlglot_tools.render_sql_duckdb`
- `codeintel.storage.sqlglot_tools.canonical_sql_duckdb`
- `codeintel.storage.sqlglot_tools.fingerprint_sql_duckdb`

These helpers ensure stable SQL rendering and stable query fingerprints for
diffing and cache keys. Prefer them to ad-hoc SQLGlot usage.

## Lineage exposure

Column lineage is now computed during view materialization and recorded in
`metadata.derived_lineage_columns`. Serving exposes lineage on view descriptions.

- Materialization writes lineage:
  - `codeintel.storage.views.materialization.materialize_registered_views`
  - `codeintel.storage.sqlglot_tools.extract_column_lineage_duckdb`
  - `codeintel.storage.metadata.sync.sync_derived_lineage_columns`
- Serving reads lineage:
  - `codeintel.serving.semantic.kernel.SemanticQueryKernel.describe`
  - Response model: `codeintel.serving.semantic.models.SemanticViewDescriptionResponse.lineage`

Lineage is best-effort for complex SQL. It should never block serving response
construction; callers must handle missing or partial lineage gracefully.

## Export/import APIs

Storage now exposes export/import on the public gateway API and CLI:

- Public API:
  - `StorageGateway.export_database(directory=...)`
  - `StorageGateway.import_database(directory=...)`
- CLI commands:
  - `codeintel storage export-db --db-path <db> --output-dir <dir>`
  - `codeintel storage import-db --db-path <db> --input-dir <dir>`

These commands and APIs use DuckDB `EXPORT DATABASE` and `IMPORT DATABASE`
semantics. Import is blocked when the gateway is read-only.
