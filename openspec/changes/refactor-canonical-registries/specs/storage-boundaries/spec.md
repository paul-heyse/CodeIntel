## ADDED Requirements
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

## Implementation Status
- Done: canonical catalogs persist in the metadata table and Hamilton IO delegates to storage
  Ibis interfaces.
