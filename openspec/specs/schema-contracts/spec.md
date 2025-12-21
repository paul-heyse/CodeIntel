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
Dataset contracts SHALL use schema-generated row bindings with provenance metadata,
and legacy RowBinding and row migration APIs SHALL NOT be supported.

#### Scenario: Row binding includes provenance
- **WHEN** a contract requires a row binding for a table key
- **THEN** the binding includes row_model, serializer, table_key, and schema_hash

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

