# SQLAlchemy-Free DuckDB Iceberg Catalog Implementation Plan

## Intent

Adopt a SQLAlchemy-free PyIceberg catalog backed directly by DuckDB. Replace the current `SqlCatalog` usage with a custom catalog implementation loaded via `py-catalog-impl`, and standardize catalog SQL through SQLGlot for deterministic, safe, dialect-aware metadata operations.

## Decision Summary

- Use a custom `DuckDBCatalog` (PyIceberg `Catalog` subclass) that talks to DuckDB directly.
- Store Iceberg catalog metadata in a DuckDB file with the same tables as `SqlCatalog` (`iceberg_tables`, `iceberg_namespace_properties`).
- Use SQLGlot AST builders for all catalog SQL (DDL/DML). No SQLAlchemy.
- Keep PyIceberg table semantics intact: optimistic concurrency, snapshot refs, transactions, name mapping.

## Design Overview

### Catalog Storage Model

Keep the canonical catalog schema compatible with PyIceberg SQL semantics:

- `iceberg_tables`
  - `catalog_name VARCHAR`
  - `table_namespace VARCHAR`
  - `table_name VARCHAR`
  - `metadata_location VARCHAR`
  - `previous_metadata_location VARCHAR`
  - Primary key: `(catalog_name, table_namespace, table_name)`
- `iceberg_namespace_properties`
  - `catalog_name VARCHAR`
  - `namespace VARCHAR`
  - `property_key VARCHAR`
  - `property_value VARCHAR`
  - Primary key: `(catalog_name, namespace, property_key)`

### Concurrency + Commit Semantics (No SQLAlchemy)

- Use DuckDB transactions with optimistic concurrency by matching on `metadata_location`:
  - UPDATE path: `UPDATE iceberg_tables SET metadata_location = ?, previous_metadata_location = ? WHERE catalog_name = ? AND table_namespace = ? AND table_name = ? AND metadata_location = ?`.
  - If `rowcount == 0`, raise `CommitFailedException` (matches PyIceberg semantics).
- For create operations:
  - Use `INSERT INTO ...` and handle primary key conflicts by raising `TableAlreadyExistsError`.
- Avoid `SELECT ... FOR UPDATE` entirely (DuckDB does not support it).

### SQLGlot Standardization

- Build catalog queries using SQLGlot AST (insert/update/delete/select).
- Use `exp.Table`, `exp.Column`, `exp.EQ`, `exp.And`, etc. for portable SQL generation.
- Use `sql(dialect="duckdb")` for final SQL generation.
- Optional: run `optimize()` in debug mode to normalize SQL before hashing/logging.

### Configuration / Loading

- Replace `IcebergCatalogProvider.load()` to use `py-catalog-impl` for the custom catalog.
- New catalog properties:
  - `py-catalog-impl = codeintel.storage.iceberg.duckdb_catalog.DuckDBCatalog`
  - `uri = duckdb:///path/to/catalog.duckdb` (reused from `IcebergSettings`)
  - `warehouse = ...` as currently configured
- Keep `IcebergSettings` as the single source of truth.

## Implementation Plan (Phased)

### Phase 0 - Finalize Design Decisions

- Confirm catalog file location and naming (e.g., `iceberg_catalog.duckdb`).
- Confirm catalog schema names (still `metadata.iceberg_*` for cache, `iceberg_*` for catalog).
- Decide if the catalog file should be read-only on serving connections.

Acceptance:
- Catalog schema layout and storage location confirmed.
- `IcebergSettings` fields needed for custom catalog validated.

### Phase 1 - Catalog Core Implementation (DuckDB + SQLGlot)

Create the custom catalog module:

- New module: `src/codeintel/storage/iceberg/duckdb_catalog.py`.
- Implement required `Catalog` methods (min set):
  - `create_namespace`, `list_namespaces`, `load_namespace_properties`, `update_namespace_properties`
  - `create_table`, `create_table_transaction`, `load_table`, `table_exists`, `drop_table`, `rename_table`
  - `commit_table` with optimistic concurrency
  - `list_tables`
- Add a `DuckDBCatalogSession` helper:
  - Connection management
  - Schema creation / migrations for `iceberg_*` tables
  - Transaction helper (`BEGIN`, `COMMIT`, `ROLLBACK`)
- Use SQLGlot for all SQL:
  - Store SQL builders for catalog table operations in one place.
  - Keep SQLGlot dialect set to `duckdb`.

Acceptance:
- DuckDB catalog tables created if missing.
- Catalog ops support concurrent-safe commits without SQLAlchemy.
- No SQLAlchemy imports remain in catalog path.

### Phase 2 - Integration with IcebergSettings + Provider

- Update `src/codeintel/core/iceberg/catalog.py`:
  - Set `py-catalog-impl` in `_catalog_properties()` based on settings.
  - Remove any SQLAlchemy-only assumptions (e.g., dialect patching).
- Ensure settings are read exclusively via `SettingsView`.

Acceptance:
- `IcebergCatalogProvider.load()` returns the custom DuckDB catalog.
- Tests using `IcebergCatalogProvider` no longer import/trigger SQLAlchemy.

### Phase 3 - Validation + Compatibility

- Add unit tests for the new catalog:
  - Namespace create/list/update.
  - Table create/load/exists.
  - Commit conflict (metadata_location mismatch).
  - Commit success path.
- Update existing Iceberg tests:
  - Use the custom catalog by default (no SQLAlchemy).
  - Validate commit/run snapshot refs are created.
- Update CLI tests that previously failed on SQLAlchemy locking.

Acceptance:
- Targeted Iceberg tests pass without SQLAlchemy.
- No SQLAlchemy warnings or SQL `FOR UPDATE` appears in test logs.

### Phase 4 - Cleanup + Dependency Removal

- Remove SQLAlchemy from dependencies in `pyproject.toml` if only used for PyIceberg SQL catalog.
- Update docs to reflect the new catalog implementation (include in Iceberg docs or runtime env docs).

Acceptance:
- `sqlalchemy` not imported anywhere in runtime paths.
- `uv sync` succeeds without SQLAlchemy.

## Testing Plan

Focus on the cutover tests already defined for Iceberg:

- `tests/build/hamilton/test_materializer.py` (write, snapshot metadata, refs)
- `tests/storage/test_iceberg_cache.py` (cache refresh)
- `tests/cli/test_iceberg_cli.py` (inspect/refs/manage-snapshots)

Add catalog-specific tests:

- `tests/storage/test_duckdb_catalog.py` (new): create, list, commit, conflict.

Run with:

- `ICEBERG_READ_ENABLED=true ICEBERG_WRITE_ENABLED=true uv run pytest -q ...`

## Risks and Mitigations

- Risk: Commit conflict logic diverges from PyIceberg expectations.
  - Mitigation: Match PyIceberg update conditions on `metadata_location` and raise `CommitFailedException` when `rowcount == 0`.
- Risk: Namespace operations missing in custom catalog.
  - Mitigation: Implement namespace table exactly as SqlCatalog expects.
- Risk: SQL drift across dialects.
  - Mitigation: Use SQLGlot with `duckdb` dialect for all queries.

## Acceptance Criteria

- All Iceberg paths operate without SQLAlchemy.
- DuckDB catalog commits succeed and enforce optimistic concurrency.
- Existing Iceberg tests pass with the custom catalog.
- No `FOR UPDATE` usage remains anywhere in the catalog path.

