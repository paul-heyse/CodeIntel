## ADDED Requirements
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
row serialization helpers SHALL NOT be the authoritative source of column order.

#### Scenario: Row serialization uses schema registry ordering
- **WHEN** rows are serialized for a dataset write
- **THEN** the column order is derived from the schema registry row model

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

## MODIFIED Requirements
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
