# Runtime Environment Variables

This document lists environment variables that control runtime behavior. These are parsed by
`src/codeintel/core/runtime/loader.py` and override defaults in code.

## Iceberg

- `CODEINTEL_ICEBERG_READ_ENABLED` (bool): Enable Iceberg reads.
- `CODEINTEL_ICEBERG_WRITE_ENABLED` (bool): Enable Iceberg writes.
- `CODEINTEL_ICEBERG_TOMBSTONES_ENABLED` (bool): Enable tombstone filtering.
- `CODEINTEL_ICEBERG_FLIGHT_ENABLED` (bool): Enable Arrow Flight serving (when supported).
- `CODEINTEL_ICEBERG_READ_REF` (str): Override snapshot ref name to read (e.g., `main` or `run/<id>`).
- `CODEINTEL_ICEBERG_ENFORCE_PREFIXES` (csv): Table key prefixes that must resolve to Iceberg.
- `CODEINTEL_ICEBERG_CATALOG_NAME` (str): Catalog name passed to PyIceberg.
- `CODEINTEL_ICEBERG_CATALOG_TYPE` (str): Catalog type (e.g., `sql`).
- `CODEINTEL_ICEBERG_CATALOG_URI` (str): Catalog connection URI.
- `CODEINTEL_ICEBERG_CATALOG_WAREHOUSE` (str): Warehouse path for table data.
- `CODEINTEL_ICEBERG_CATALOG_PROPERTIES` (csv `key=value`): Extra catalog properties.
- `CODEINTEL_ICEBERG_CONFIG_PATH` (path): Optional `.pyiceberg.yaml` path override.
- `CODEINTEL_ICEBERG_IO_IMPL` (str): FileIO implementation class.
- `CODEINTEL_ICEBERG_IO_OPTIONS` (csv `key=value`): FileIO option overrides.
