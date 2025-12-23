## ADDED Requirements
### Requirement: Serving uses the canonical registry service
Serving SHALL source semantic catalogs, export metadata, and schema summaries from the
canonical RegistryService and shared manifests. Serving SHALL NOT compile or maintain
local registry copies.

#### Scenario: Serving catalog derives from registry service
- **WHEN** a serving catalog response is generated
- **THEN** it uses RegistryService outputs backed by the canonical metadata tables
