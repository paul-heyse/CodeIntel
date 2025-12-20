## MODIFIED Requirements
### Requirement: DuckDB isolation to storage layer
DuckDB-specific imports, connection types, and relation APIs SHALL be confined to storage
modules. Non-storage modules SHALL NOT import duckdb at runtime, and any protocols consumed
outside storage SHALL avoid runtime duckdb imports (type-only usage is allowed).

#### Scenario: Build modules do not import duckdb
- **WHEN** build and export modules are imported
- **THEN** no duckdb symbols are imported outside storage modules

### Requirement: Storage protocol interfaces
Build, export, and serving modules SHALL interact with storage through duckdb-agnostic
protocols (e.g., ExportRelation, RecordBatchReader, storage export services) and SHALL NOT
call `gateway.con` or DuckDB relation APIs directly.

#### Scenario: Export uses protocol interface
- **WHEN** build code exports a dataset
- **THEN** it requests an export relation from a storage-owned service and never calls
  `DuckDBPyConnection` APIs
