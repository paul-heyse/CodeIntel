## ADDED Requirements

### Requirement: DatasetCatalog is canonical contract registry
The system SHALL emit a DatasetCatalog artifact that is the canonical registry for dataset
contracts, including table_key, dataset name, schema_hash, schema_version, export specs, row
binding metadata, dependencies, validation_profile, ownership, tags, and deprecation fields.

#### Scenario: Catalog entry completeness
- **WHEN** a dataset is included in the DAG contract set
- **THEN** its DatasetCatalog entry includes table_key, name, schema_hash, and validation_profile

#### Scenario: Deprecated dataset metadata
- **WHEN** a dataset is marked deprecated
- **THEN** the DatasetCatalog entry includes deprecated=true and a deprecation_message

### Requirement: Metadata tables are derived
DuckDB metadata tables (including metadata.datasets) SHALL be derived from DatasetCatalog and
MUST NOT be treated as the source of truth.

#### Scenario: Metadata bootstrap
- **WHEN** storage bootstrap runs
- **THEN** metadata.datasets is populated from DatasetCatalog entries

#### Scenario: Catalog update propagation
- **WHEN** DatasetCatalog changes
- **THEN** derived metadata tables reflect the updated catalog after bootstrap or sync

### Requirement: DAG-derived dependencies
DatasetCatalog SHALL include upstream_dependencies derived from the global DAG and view lineage.

#### Scenario: Dependency derivation
- **WHEN** a dataset depends on upstream tables in the DAG
- **THEN** those dependencies appear in upstream_dependencies in the catalog

#### Scenario: View lineage
- **WHEN** a view is produced from source tables
- **THEN** upstream_dependencies reflect the view's source table keys

### Requirement: Contract lookup via catalog
Storage and serving SHALL resolve dataset contracts exclusively via DatasetCatalog.

#### Scenario: Lookup by table_key
- **WHEN** a runtime component requests a contract for a table_key
- **THEN** it is returned from DatasetCatalog without querying build registries
