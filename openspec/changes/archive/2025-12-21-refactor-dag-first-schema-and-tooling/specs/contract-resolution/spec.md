## ADDED Requirements
### Requirement: DAG-first schema provider with production inference
The system SHALL expose a DAG-first schema provider that resolves DAG outputs via SchemaIndex,
with production inference enabled and derivation provenance recorded; callers SHALL be able to
disable inference explicitly.

#### Scenario: Inference resolves a DAG output schema
- **WHEN** a schema is requested for an inferable DAG-produced table key
- **THEN** the provider returns the inferred schema with provenance marked as inferred

#### Scenario: Inference can be disabled
- **WHEN** a caller requests schemas with inference disabled
- **THEN** the provider returns explicit overrides or None without triggering inference

### Requirement: Inference failures are hard errors when no alternative exists
The system SHALL raise a hard error when DAG schema inference fails and no viable non-DAG
alternative schema is available for the dataset.

#### Scenario: Inference failure without fallback is fatal
- **WHEN** schema inference fails for a DAG-produced table key without a fallback override
- **THEN** schema resolution fails with a hard error

### Requirement: Contract resolution defaults to DAG-first outputs
Contract resolution SHALL default to the DAG-first provider with target metadata and output
overrides, and SHALL provide an explicit declared-only mode for DAG-free enumeration.

#### Scenario: Default contract resolution includes DAG outputs
- **WHEN** a caller resolves contracts without specifying a resolution mode
- **THEN** DAG-produced table keys are included with target metadata overrides applied

#### Scenario: Declared-only mode is DAG-free
- **WHEN** a caller requests declared-only contract resolution
- **THEN** DAG-produced table keys are excluded and the Hamilton DAG is not initialized

## MODIFIED Requirements
### Requirement: Schema-only contract enumeration is DAG-free
Schema-only contract enumeration SHALL use the declared-only provider and SHALL NOT initialize
the Hamilton DAG. DAG-first providers MAY initialize the DAG only when explicitly requested.

#### Scenario: Schema-only enumeration avoids DAG initialization
- **WHEN** schema-only contracts are enumerated or default validation schemas are requested
- **THEN** the Hamilton DAG is not constructed

### Requirement: Build contract resolution uses source-only providers
Build-layer contract resolution SHALL expose a declared-only provider for schema-only
enumeration and a DAG-first provider for execution, and SHALL NOT expose a full declared
schema provider from build APIs.

#### Scenario: Schema-only enumeration excludes DAG outputs
- **WHEN** build contract enumeration runs in schema-only mode
- **THEN** DAG-produced table keys are excluded and no full provider is available from build
