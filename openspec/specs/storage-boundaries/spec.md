# storage-boundaries Specification

## Purpose
TBD - created by archiving change refactor-contracts-storage-boundaries. Update Purpose after archive.
## Requirements
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

### Requirement: Canonical storage error surfaces
Storage error types SHALL be imported from canonical modules and SHALL NOT be
re-exported through compatibility shims.

#### Scenario: Consumers import canonical errors
- **WHEN** a non-storage module needs storage errors
- **THEN** it imports from codeintel.core.errors.storage and canonical DuckDB types
  instead of compatibility modules

### Requirement: DuckDB is required for storage protocols
Storage gateway protocol modules SHALL assume DuckDB is available at runtime and
SHALL NOT define fallback DuckDB exception stubs.

#### Scenario: DuckDB dependency is required
- **WHEN** storage gateway protocols are imported in runtime environments
- **THEN** DuckDB types resolve directly without fallback stub definitions

### Requirement: Versioned asset catalog is canonical
Storage SHALL treat build.asset_versions as the canonical, immutable asset catalog and SHALL
store run-scoped metadata via build.asset_version_events. Legacy build.assets storage SHALL
NOT be supported.

#### Scenario: Asset versions are immutable and run-scoped metadata is separated
- **WHEN** an asset version is recorded
- **THEN** immutable version metadata is stored in build.asset_versions and run metadata is
  stored in build.asset_version_events

### Requirement: External compatibility normalization is removed
Storage boundaries SHALL NOT normalize numpy scalar inputs, and callers SHALL pass values to
DuckDB/Ibis without explicit scalar conversion helpers.

#### Scenario: No normalization helpers are used
- **WHEN** rows are written via analytics or build helpers
- **THEN** no numpy scalar normalization helpers are invoked

