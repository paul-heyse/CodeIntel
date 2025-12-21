## ADDED Requirements
### Requirement: Native-only build target implementations
The build system SHALL execute targets using native Hamilton modules only, and
wrapper/template implementations or allowlists SHALL NOT be used.

#### Scenario: Plan entries are native
- **WHEN** a build plan is computed
- **THEN** each target is classified as native and no wrapper allowlist warnings occur
