## MODIFIED Requirements
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

## ADDED Requirements
### Requirement: Legacy schema export and migration utilities are removed
Schema contract APIs SHALL NOT expose legacy export, lineage, schema-doc, or migration
utilities, and callers MUST rely on the canonical schema registry and storage metadata.

#### Scenario: Legacy schema utilities are absent
- **WHEN** schema tooling is enumerated
- **THEN** contracts.schemas.export, contracts.schemas.lineage, schema_docs, and
  validators.migration helpers are not present
