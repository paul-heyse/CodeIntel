## Context
Serving and semantic tests create temporary DuckDB snapshots, but some still issue ad-hoc
`CREATE SCHEMA docs` statements. Recent helper changes apply production schemas by default,
so these ad-hoc statements now fail. Separately, the Hamilton test harness surfaces missing
TargetRunRecord errors without the underlying Hamilton execution failure, which hides the
root cause of build issues.

## Goals / Non-Goals
- Goals:
  - Centralize schema creation in a shared, idempotent helper that uses production schema
    definitions.
  - Eliminate ad-hoc schema DDL in tests for production schemas.
  - Surface Hamilton build errors and target status context when harness record lookup fails.
- Non-Goals:
  - Changing production schema DDL logic or build execution semantics.
  - Adding new test helpers beyond the schema seeding helper and harness error reporting.

## Decisions
- Decision: Introduce a test-only schema seeding helper that calls the production schema
  provider via DuckDBPolicyBackend helpers (e.g., `create_schemas` plus any required
  metadata DDL) and is safe to call multiple times.
- Decision: The helper always seeds all production schemas; no scoped subsets are allowed.
- Decision: Require tests to call the shared helper instead of issuing `CREATE SCHEMA` for
  production schemas (docs/core/graph/analytics).
- Decision: Extend HamiltonBuildHarness record retrieval to include build error context
  (result.error, failed_targets, skipped_targets, missing mappings) in raised exceptions.

## Risks / Trade-offs
- Broader schema seeding may create more schemas than a test strictly needs.
  Mitigation: Keep helper focused on schema creation only and avoid table materialization.
- Existing tests that rely on minimal setup may need minor updates to use the helper.

## Migration Plan
1. Add the schema seeding helper and update tests to call it.
2. Remove direct `CREATE SCHEMA` statements for production schemas.
3. Update harness record retrieval to emit diagnostic errors and adjust tests accordingly.

## Open Questions
None.
