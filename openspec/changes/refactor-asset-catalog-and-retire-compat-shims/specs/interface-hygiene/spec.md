## ADDED Requirements
### Requirement: Public interfaces exclude compatibility-only parameters
Public APIs SHALL NOT include unused compatibility-only parameters or methods. Parameters
MUST exist only when required for behavior or documented interfaces.

#### Scenario: Compatibility parameters are removed
- **WHEN** public APIs are reviewed for unused parameters
- **THEN** compatibility-only parameters are removed and call sites are updated

### Requirement: Deprecated or no-op commands are not exposed
CLI surfaces SHALL NOT expose deprecated or no-op commands such as storage.generate_macros.

#### Scenario: Deprecated storage command is absent
- **WHEN** storage CLI commands are listed
- **THEN** storage.generate_macros is not present
