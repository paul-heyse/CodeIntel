## ADDED Requirements
### Requirement: Versioned asset catalog is canonical
Storage SHALL treat build.asset_versions as the canonical, immutable asset catalog and SHALL
store run-scoped metadata via build.asset_version_events. Legacy build.assets storage SHALL
NOT be supported.

#### Scenario: Asset versions are immutable and run-scoped metadata is separated
- **WHEN** an asset version is recorded
- **THEN** immutable version metadata is stored in build.asset_versions and run metadata is
  stored in build.asset_version_events

### Requirement: External compatibility normalization is removed
Storage boundaries SHALL NOT normalize numpy scalar inputs, and callers SHALL pass values to
DuckDB/Ibis without explicit scalar conversion helpers.

#### Scenario: No normalization helpers are used
- **WHEN** rows are written via analytics or build helpers
- **THEN** no numpy scalar normalization helpers are invoked
