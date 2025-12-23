## ADDED Requirements

### Requirement: SCIP harness parity coverage
SCIP ingestion tests SHALL include a harness-based path that can execute with real tools when
available, while preserving deterministic artifact-based execution when tools are absent.

#### Scenario: Real tool execution
- **WHEN** SCIP tooling is available on PATH
- **THEN** a harness-based SCIP test executes and validates row counts and artifacts.

#### Scenario: Deterministic artifact execution
- **WHEN** SCIP tooling is unavailable
- **THEN** a harness-based SCIP test uses pre-seeded artifacts to validate ingestion behavior.

### Requirement: Modules-first seeding in helper utilities
Helper utilities that seed module inventories SHALL use modules-first expectations derived from
`modules_expected_from_repo_tree(...)` or `module_paths_expected_from_repo_tree(...)`.

#### Scenario: Helper inventory seeding
- **WHEN** a helper seeds core.modules/core.repo_map from repo contents
- **THEN** module paths are derived from the modules-first helpers instead of raw rglob lists.

### Requirement: Graph inventory consistency assertions
Graph tests that rely on module catalogs SHALL assert repo_map/modules consistency using
`ModulesAssertions.inventory_consistent()`.

#### Scenario: Graph loader assumes module inventory
- **WHEN** a graph test depends on module catalog completeness
- **THEN** it validates repo_map/modules consistency via ModulesAssertions.

### Requirement: Golden diffs for analytics module maps
Tests that compare module/path mappings outside ingestion/storage SHALL use golden diff helpers
for diagnostics.

#### Scenario: Analytics module map mismatch
- **WHEN** a module/path mapping comparison fails in analytics tests
- **THEN** failures use `format_missing_extra(...)` or `format_module_map_diff(...)` to report diffs.

### Requirement: Shared Hamilton runtime for fixtures
Hamilton execution fixtures SHALL reuse the shared runtime/harness path to minimize duplicated
BuildEnv wiring and keep skip/manifest behavior consistent.

#### Scenario: Fixture execution environment
- **WHEN** a fixture constructs a Hamilton execution context
- **THEN** it reuses the shared runtime or harness helpers rather than building a new runtime.
