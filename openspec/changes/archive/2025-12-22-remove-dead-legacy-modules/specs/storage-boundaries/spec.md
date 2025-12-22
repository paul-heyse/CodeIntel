## MODIFIED Requirements
### Requirement: Versioned asset catalog is canonical
Storage SHALL treat build.asset_versions as the canonical, immutable asset catalog and SHALL
store run-scoped metadata via build.asset_version_events. Legacy build.assets storage SHALL
NOT be supported, and legacy dataset catalog generators SHALL NOT be used.

#### Scenario: Asset versions are immutable and run-scoped metadata is separated
- **WHEN** an asset version is recorded
- **THEN** immutable version metadata is stored in build.asset_versions and run metadata is
  stored in build.asset_version_events

#### Scenario: Legacy dataset catalog helpers are absent
- **WHEN** dataset catalog utilities are enumerated
- **THEN** storage.datasets.catalog is not present and docs use versioned catalog tables

## ADDED Requirements
### Requirement: Ephemeral storage gateways are not shipped
Storage gateways SHALL be created through configured StorageConfig and gateway factories,
and in-memory ephemeral gateway helpers SHALL NOT be part of runtime packages.

#### Scenario: Runtime gateways exclude ephemeral helpers
- **WHEN** storage gateway helpers are enumerated
- **THEN** no ephemeral gateway helper is present and schema compilation uses standard
  gateway configuration
