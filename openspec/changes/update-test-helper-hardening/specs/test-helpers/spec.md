## ADDED Requirements
### Requirement: Idempotent production-coupled schema seeding helper
Tests that require production schemas (core/graph/analytics/docs) SHALL call a shared helper
that creates schemas using the production schema provider and storage policy backend. Tests
SHALL NOT issue ad-hoc `CREATE SCHEMA` statements for production schemas.

#### Scenario: Docs schema is seeded for a test snapshot
- **WHEN** a test prepares a DuckDB database for docs.* tables or views
- **THEN** it calls the shared schema-seeding helper and no ad-hoc schema DDL appears in the test.

#### Scenario: Schema seeding is idempotent
- **WHEN** the schema-seeding helper is called multiple times on the same database
- **THEN** it completes without raising and existing schemas remain intact.

### Requirement: Harness errors surface build failures
Hamilton test harnesses SHALL raise diagnostic errors that include the underlying build error
and target status context when a requested target record is unavailable.

#### Scenario: Target record missing after execution
- **WHEN** a test requests a target record and the Hamilton run did not produce it
- **THEN** the raised error includes the build error message (if any) and lists failed or
  missing targets.
