# schema-contracts Specification

## Purpose
TBD - created by archiving change refactor-contracts-storage-boundaries. Update Purpose after archive.
## Requirements
### Requirement: Canonical contract policy
The system SHALL use a single shared contract factory for DatasetContract derivation (schema
IDs, default export filenames, view detection, owner package mapping, tags, and row bindings),
and both build and storage providers SHALL use that factory with their respective metadata.

#### Scenario: Build and storage agree on dataset contracts
- **WHEN** build and storage derive a DatasetContract for the same table key and metadata
- **THEN** the resulting json_schema_id, filenames, tags, owner_package, and row_binding match

### Requirement: Schema IDs are independent of export policy
The system SHALL derive schema IDs from contract content and SHALL NOT change schema IDs
based on exportability configuration.

#### Scenario: Exportability changes do not change schema IDs
- **WHEN** an exportability flag changes for a contract
- **THEN** the schema ID for that contract remains unchanged

### Requirement: Deterministic schema mapping
The system SHALL produce deterministic schema ID maps and DatasetContract fields for identical
contract inputs.

#### Scenario: Deterministic mapping across runs
- **WHEN** the same contract inputs are processed multiple times
- **THEN** the schema ID map and derived DatasetContract values are stable and ordered

### Requirement: Schema-generated row bindings only
Dataset contracts SHALL use schema-generated row bindings sourced from the unified DAG-first
schema registry with provenance metadata, and legacy RowBinding and row migration APIs SHALL
NOT be supported.

#### Scenario: Row binding includes provenance
- **WHEN** a contract requires a row binding for a table key
- **THEN** the binding includes row_model, serializer, table_key, schema_hash, and derivation
  metadata

#### Scenario: DAG output row binding uses the unified registry
- **WHEN** a row binding is requested for a DAG-produced table key
- **THEN** the binding is generated from the DAG-first schema provider rather than
  declared_schemas

#### Scenario: Legacy row migration API is absent
- **WHEN** callers attempt to use the row migration compatibility API
- **THEN** no compatibility module is available and callers must use the schema registry

### Requirement: Schema compilation is native-only
Schema compilation SHALL consider only native targets and SHALL NOT expose compatibility
flags such as --only-native.

#### Scenario: Schema diff CLI omits only-native
- **WHEN** the schema diff command help is displayed
- **THEN** no only-native flag is exposed

### Requirement: Structured schema diff is the only output
Schema diff SHALL emit structured summaries with breaking-change detection and SHALL NOT
provide legacy unified diff output or a toggle to enable it.

#### Scenario: Schema diff uses structured output
- **WHEN** schema diff detects changes
- **THEN** the output includes a structured summary rather than a unified diff

### Requirement: DAG-first schema registry
The system SHALL resolve schemas for DAG-produced datasets via the Hamilton DAG schema index,
with production inference enabled and derivation provenance recorded for every schema.

#### Scenario: DAG output schema uses DAG-first resolution
- **WHEN** a DAG-produced table key is requested from the schema provider
- **THEN** the schema is resolved from the SchemaIndex with provenance indicating inferred
  versus explicit override

### Requirement: Declared schemas are source-only
Declared schema registries SHALL be restricted to non-DAG source tables and explicit overrides,
and DAG outputs SHALL NOT require static declarations in declared_schemas.

#### Scenario: Source-only provider excludes DAG outputs
- **WHEN** the declared source-only provider enumerates table schemas
- **THEN** DAG-produced table keys are excluded from the enumeration

### Requirement: Canonical row serialization from schema registry
Row serialization SHALL use schema registry row models and column ordering, and ad-hoc
row serialization helpers, re-export shims, or static column list modules SHALL NOT be the
authoritative source of column order.

#### Scenario: Row serialization uses schema registry ordering
- **WHEN** rows are serialized for a dataset write
- **THEN** the column order is derived from the schema registry row model

#### Scenario: Compatibility row serialization helpers are absent
- **WHEN** schema serialization helpers are enumerated
- **THEN** build.hamilton.row_serialization, ingestion.row_serialization, and analytics
  cfg/dfg column list modules are not present

### Requirement: Constraint enforcement order
Schema constraints SHALL be enforced in order: Hamilton checks first, then Pandera checks, and
Pydantic constraints only when required.

#### Scenario: Constraints follow the enforcement order
- **WHEN** dataset rows are validated
- **THEN** Hamilton checks run before Pandera checks, and Pydantic is used only if needed

### Requirement: DAG-derived refresh for backups and seeds
Any backup or seed dataset definition SHALL include tooling to refresh from DAG-produced
outputs so the DAG remains the primary source of truth.

#### Scenario: Backup dataset has DAG refresh tooling
- **WHEN** a dataset is designated as a backup or seed
- **THEN** a DAG-driven refresh path is available for that dataset

### Requirement: Legacy schema export and migration utilities are removed
Schema contract APIs SHALL NOT expose legacy export, lineage, schema-doc, or migration
utilities, and callers MUST rely on the canonical schema registry and storage metadata.

#### Scenario: Legacy schema utilities are absent
- **WHEN** schema tooling is enumerated
- **THEN** contracts.schemas.export, contracts.schemas.lineage, schema_docs, and
  validators.migration helpers are not present

