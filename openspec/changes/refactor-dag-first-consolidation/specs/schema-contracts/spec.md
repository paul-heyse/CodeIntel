## MODIFIED Requirements
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
