## ADDED Requirements
### Requirement: Canonical storage error surfaces
Storage error types SHALL be imported from canonical modules and SHALL NOT be
re-exported through compatibility shims.

#### Scenario: Consumers import canonical errors
- **WHEN** a non-storage module needs storage errors
- **THEN** it imports from codeintel.core.errors.storage and canonical DuckDB types
  instead of compatibility modules

### Requirement: DuckDB is required for storage protocols
Storage gateway protocol modules SHALL assume DuckDB is available at runtime and
SHALL NOT define fallback DuckDB exception stubs.

#### Scenario: DuckDB dependency is required
- **WHEN** storage gateway protocols are imported in runtime environments
- **THEN** DuckDB types resolve directly without fallback stub definitions
