## ADDED Requirements

### Requirement: ValidationRunner is canonical
All validation in build, storage, and serving SHALL be executed via the core ValidationRunner with
domain-specific check sets.

#### Scenario: Build validation
- **WHEN** build validation is invoked
- **THEN** it uses ValidationRunner with the build check set

#### Scenario: Storage validation
- **WHEN** storage validation runs against DuckDB state
- **THEN** it uses ValidationRunner with the storage check set

### Requirement: Validation profiles are enforced
Each dataset in DatasetCatalog SHALL declare a validation_profile, and all validation surfaces
MUST honor that profile.

#### Scenario: Strict profile
- **WHEN** a dataset has validation_profile="strict"
- **THEN** validation errors raise and fail the operation

#### Scenario: Lenient profile
- **WHEN** a dataset has validation_profile="lenient"
- **THEN** validation errors are reported but do not fail the operation

### Requirement: Pandera validation uses SchemaService
Pandera validation SHALL use schemas resolved from SchemaService (manifest-backed), not parallel
schema registries.

#### Scenario: Pandera schema resolution
- **WHEN** a DataFrame validation is requested for a table_key
- **THEN** the Pandera schema is generated from SchemaService's TableSchema
