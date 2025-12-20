## ADDED Requirements
### Requirement: Canonical contract policy
The system SHALL use a single shared contract policy for schema ID derivation and
exportability, and both build and storage providers SHALL use that policy.

#### Scenario: Build and storage agree on schema IDs
- **WHEN** build and storage enumerate schema IDs from the same contract set
- **THEN** the schema ID map is identical in both layers

### Requirement: Schema IDs are independent of export policy
The system SHALL derive schema IDs from contract content and SHALL NOT change schema IDs
based on exportability configuration.

#### Scenario: Exportability changes do not change schema IDs
- **WHEN** an exportability flag changes for a contract
- **THEN** the schema ID for that contract remains unchanged

### Requirement: Deterministic schema mapping
The system SHALL produce deterministic schema ID maps for identical contract inputs.

#### Scenario: Deterministic mapping across runs
- **WHEN** the same contract inputs are processed multiple times
- **THEN** the schema ID map is stable and ordered consistently
