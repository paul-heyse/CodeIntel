# Runtime Environment Variables

This document lists environment variables that control runtime behavior. These are parsed by
`src/codeintel/core/runtime/loader.py` and override defaults in code.

## Iceberg

Defaults:
- When `CODEINTEL_DEPLOYMENT_ENVIRONMENT` is `prod`/`production`, Iceberg read/write/tombstones default to false.
- In non-prod, Iceberg read/write/tombstones default to true when catalog configuration is provided.
- Iceberg read fallback defaults to false in prod and true in non-prod.

- `CODEINTEL_DEPLOYMENT_ENVIRONMENT` (str): Deployment environment name (`prod`, `production`, etc.).
- `CODEINTEL_ICEBERG_READ_ENABLED` (bool): Enable Iceberg reads.
- `CODEINTEL_ICEBERG_READ_FALLBACK_ENABLED` (bool): Allow DuckDB fallback when Iceberg scans fail.
- `CODEINTEL_ICEBERG_WRITE_ENABLED` (bool): Enable Iceberg writes.
- `CODEINTEL_ICEBERG_TOMBSTONES_ENABLED` (bool): Enable tombstone filtering.
- `CODEINTEL_ICEBERG_FLIGHT_ENABLED` (bool): Enable Arrow Flight serving (when supported).
- `CODEINTEL_ICEBERG_READ_REF` (str): Override snapshot ref name to read (e.g., `main` or `run/<id>`).
- `CODEINTEL_ICEBERG_ENFORCE_PREFIXES` (csv): Table key prefixes that must resolve to Iceberg.
- `CODEINTEL_ICEBERG_CATALOG_NAME` (str): Catalog name passed to PyIceberg.
- `CODEINTEL_ICEBERG_CATALOG_TYPE` (str): Optional catalog type for non-DuckDB catalogs.
- `CODEINTEL_ICEBERG_CATALOG_PROPERTIES` (csv `key=value`): Extra catalog properties, including
  `py-catalog-impl` to override the default DuckDB catalog implementation.
- `CODEINTEL_ICEBERG_CATALOG_URI` (str): Catalog connection URI.
- `CODEINTEL_ICEBERG_CATALOG_WAREHOUSE` (str): Warehouse path for table data.
- `CODEINTEL_ICEBERG_CONFIG_PATH` (path): Optional `.pyiceberg.yaml` path override.
- `CODEINTEL_ICEBERG_IO_IMPL` (str): FileIO implementation class.
- `CODEINTEL_ICEBERG_IO_OPTIONS` (csv `key=value`): FileIO option overrides.
- `CODEINTEL_ICEBERG_LOCATION_PROVIDER_IMPL` (str): PyIceberg location provider class.
- `CODEINTEL_ICEBERG_WRITE_DATA_PATH` (str): Override Iceberg write data path (`write.data.path`).
- `CODEINTEL_ICEBERG_WRITE_METADATA_PATH` (str): Override Iceberg metadata path (`write.metadata.path`).
- `CODEINTEL_ICEBERG_OBJECT_STORE_PARTITIONED_PATHS` (bool): Enable object-store partitioned paths.
