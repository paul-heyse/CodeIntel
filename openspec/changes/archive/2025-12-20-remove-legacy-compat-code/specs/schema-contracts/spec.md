## ADDED Requirements
### Requirement: Schema-generated row bindings only
Dataset contracts SHALL use schema-generated row bindings with provenance metadata,
and legacy RowBinding and row migration APIs SHALL NOT be supported.

#### Scenario: Row binding includes provenance
- **WHEN** a contract requires a row binding for a table key
- **THEN** the binding includes row_model, serializer, table_key, and schema_hash

#### Scenario: Legacy row migration API is absent
- **WHEN** callers attempt to use the row migration compatibility API
- **THEN** no compatibility module is available and callers must use the schema registry
