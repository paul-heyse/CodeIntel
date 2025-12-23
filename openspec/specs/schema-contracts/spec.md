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
Declared schema registries SHALL be limited to non-DAG source tables and explicit overrides,
and canonical contract enumeration SHALL ignore declared_schemas for DAG outputs whenever a
canonical catalog entry is available. Explicit overrides SHALL be sourced from Hamilton
registry metadata.

#### Scenario: Source-only provider excludes DAG outputs
- **WHEN** the declared source-only provider enumerates table schemas
- **THEN** DAG-produced table keys are excluded from the enumeration

#### Scenario: Canonical enumeration ignores declared schemas
- **WHEN** a DAG-produced table key is resolved from the canonical catalog
- **THEN** the contract is not sourced from declared_schemas

### Requirement: Canonical row serialization from schema registry
Row serialization SHALL be performed by a single schema-registry-backed serializer service with
caching, and build, ingestion, and validation paths SHALL use that service instead of ad-hoc
serializer helpers.

#### Scenario: Build and ingestion share the serializer
- **WHEN** build and ingestion serialize rows for the same table key
- **THEN** both use the centralized serializer and produce identical column ordering

#### Scenario: Serializer cache avoids repeated binding work
- **WHEN** row serialization is requested repeatedly for the same table key
- **THEN** the serializer reuses cached row bindings rather than rebuilding them

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

### Requirement: Contract-derived JSON Schema is canonical
The system SHALL derive export JSON Schemas from DatasetContract/Pandera schemas in the
canonical registry and SHALL NOT use TypedDict-based export schema generation for exports.

#### Scenario: Export schema uses contract registry
- **WHEN** an export schema is requested for a dataset
- **THEN** the schema is generated from the canonical contract registry

### Requirement: Canonical contract validation service
The system SHALL provide a shared contract validation service built on the Hamilton schema
registry and SHALL use it in build, storage, and serving validation flows.

#### Scenario: Build and serving use shared validator
- **WHEN** build and serving validate a DatasetContract
- **THEN** both invoke the same shared validator implementation

### Requirement: JSON columns are excluded from numeric non-negative checks
Pandera constraint generation SHALL apply non-negative checks only to numeric columns and
SHALL NOT apply them to JSON columns such as functions_covered. Corresponding count columns
(e.g., functions_covered_count) SHALL continue to enforce non-negative constraints.

#### Scenario: Test profile JSON columns pass validation
- **WHEN** analytics.test_profile includes functions_covered as a JSON list and
  functions_covered_count as 1
- **THEN** Pandera validation succeeds and only the count column is evaluated for
  non-negative checks

### Requirement: ContractService is the single contract pipeline
The system SHALL provide a ContractService that compiles Pandera schemas, JSON Schema,
row serializers, and validation policies from DatasetContract definitions. Build,
storage, and serving layers SHALL rely on this service and SHALL NOT maintain parallel
schema compilation or serialization pipelines.

#### Scenario: Build and serving share contract compilation
- **WHEN** a dataset contract is compiled for build and serving
- **THEN** both layers use ContractService outputs with identical schemas and
  serialization behavior

