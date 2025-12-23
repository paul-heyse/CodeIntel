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
NOT be supported, and legacy dataset catalog generators SHALL NOT be used.

#### Scenario: Asset versions are immutable and run-scoped metadata is separated
- **WHEN** an asset version is recorded
- **THEN** immutable version metadata is stored in build.asset_versions and run metadata is
  stored in build.asset_version_events

#### Scenario: Legacy dataset catalog helpers are absent
- **WHEN** dataset catalog utilities are enumerated
- **THEN** storage.datasets.catalog is not present and docs use versioned catalog tables

### Requirement: External compatibility normalization is removed
Storage boundaries SHALL NOT normalize numpy scalar inputs, and callers SHALL pass values to
DuckDB/Ibis without explicit scalar conversion helpers.

#### Scenario: No normalization helpers are used
- **WHEN** rows are written via analytics or build helpers
- **THEN** no numpy scalar normalization helpers are invoked

### Requirement: Ephemeral storage gateways are not shipped
Storage gateways SHALL be created through configured StorageConfig and gateway factories,
and in-memory ephemeral gateway helpers SHALL NOT be part of runtime packages.

#### Scenario: Runtime gateways exclude ephemeral helpers
- **WHEN** storage gateway helpers are enumerated
- **THEN** no ephemeral gateway helper is present and schema compilation uses standard
  gateway configuration

### Requirement: Canonical catalogs are stored in metadata schema
Storage SHALL persist canonical contract and target catalogs in a single metadata table keyed by
catalog kind and global catalog hash, and the catalogs SHALL be stored and served from DuckDB
only.

#### Scenario: Catalog lookup uses metadata table
- **WHEN** a caller requests contract or target catalog data
- **THEN** the data is loaded from the metadata catalog table in DuckDB

### Requirement: Hamilton IO delegates to storage Ibis interfaces
Hamilton IO adapters SHALL delegate Ibis table reads and writes through storage-owned gateway or
warehouse APIs, and duplicate Ibis adapter implementations SHALL NOT be used.

#### Scenario: Hamilton IO uses storage gateway
- **WHEN** Hamilton reads or writes a dataset via Ibis
- **THEN** the operation routes through storage gateway/warehouse APIs

### Requirement: Storage-owned Ibis connections only
Non-storage modules SHALL obtain Ibis connections and table expressions via storage-owned
Ibis gateways or facades, and SHALL NOT call ibis.duckdb.from_connection or construct Ibis
backends directly.

#### Scenario: Analytics uses storage Ibis gateway
- **WHEN** analytics modules query DuckDB via Ibis
- **THEN** they use the storage Ibis gateway/facade and do not call ibis.duckdb.from_connection

### Requirement: Contract-backed analytics writer is canonical
Analytics persistence outside Hamilton materializers SHALL use a shared, contract-backed
writer that validates rows via the schema registry and performs snapshot-scoped deletes,
and ad-hoc Pandera validation or direct SQL writes SHALL NOT be used.

#### Scenario: Analytics writes use the contract writer
- **WHEN** analytics rows are persisted outside Hamilton materializers
- **THEN** the shared contract-backed writer is used and no module-specific validation or
  direct SQL insert helpers are invoked

### Requirement: Canonical SQL fingerprinting toolkit
The system SHALL centralize DuckDB SQL canonicalization and fingerprinting in storage
SQLGlot tools, and serving SHALL use the same pipeline for sql_fingerprint computation with
safe fallback hashing on parse failures.

#### Scenario: Serving uses canonical fingerprinting
- **WHEN** compiled SQL is fingerprinted for a semantic query
- **THEN** storage SQLGlot canonicalization is used and raw SQL hashing is used on parse
  failures

### Requirement: Semantic SQL diff is available for upgrade gates
Storage SQL tooling SHALL provide semantic diff output for canonicalized DuckDB SQL strings
to aid upgrade diagnostics and test failure analysis.

#### Scenario: Upgrade gate reports semantic diff
- **WHEN** canonical SQL output changes in an upgrade gate test
- **THEN** a semantic diff action list is available for diagnostics

### Requirement: StorageFacade is the non-storage entrypoint
Non-storage modules SHALL access storage via a single StorageFacade that exposes
read, write, and export capabilities. Direct use of gateways, repositories, or
view builders outside storage SHALL NOT be permitted.

#### Scenario: Non-storage code uses the facade
- **WHEN** analytics or serving code needs storage access
- **THEN** it uses StorageFacade APIs instead of gateway or repository classes

