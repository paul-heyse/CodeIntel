## ADDED Requirements
### Requirement: Build and serving errors share core taxonomy
Build and serving error types SHALL map to the core ProblemDetail taxonomy and shared error
codes, and layer-specific error catalogs SHALL NOT redefine or shadow core error definitions.

#### Scenario: Build errors use core taxonomy
- **WHEN** a build error is raised and rendered for CLI or API output
- **THEN** the ProblemDetail payload uses core taxonomy codes and extensions

## Implementation Status
- Done: build and serving errors now map through core ProblemDetail/ErrorCode taxonomy.
