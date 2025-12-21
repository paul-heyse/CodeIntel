## MODIFIED Requirements
### Requirement: Declared schemas are source-only
Declared schema registries SHALL be limited to non-DAG source tables and explicit overrides, and
canonical contract enumeration SHALL ignore declared_schemas for DAG outputs whenever a
canonical catalog entry is available.

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

## ADDED Requirements
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

## Implementation Status
- Done: centralized row serialization, contract-derived JSON Schema generation, and shared
  contract validation are implemented.
- Remaining: enforce source-only declared schemas by removing declared overrides for DAG outputs
  in target metadata and schema resolution paths.
