# test-helpers Specification

## Purpose
TBD - created by archiving change add-test-helper-rollout. Update Purpose after archive.
## Requirements
### Requirement: Modules-first expectations for module inventory tests
Module inventory tests SHALL use the production scanning pipeline through
`modules_expected_from_repo_tree(...)` or `modules_expected_from_env(...)`
to derive expected module inventories.

#### Scenario: Expected inventory derived from repo tree
- **WHEN** a test asserts expected module inventory
- **THEN** it uses `modules_expected_from_repo_tree(...)` or `modules_expected_from_env(...)`
  to compute expectations instead of hard-coded path lists.

### Requirement: High-signal module inventory diffs
Module inventory tests SHALL use golden diff helpers for module list or module map comparisons.

#### Scenario: Path list mismatch
- **WHEN** an assertion compares module path lists
- **THEN** failures use `format_missing_extra(...)` with a contextual label.

#### Scenario: Module map mismatch
- **WHEN** an assertion compares module -> path mappings
- **THEN** failures use `format_module_map_diff(...)` (inverting path -> module maps via
  `module_map_from_path_map(...)` as needed).

### Requirement: Modules inventory consistency helpers
Tests that validate consistency between `core.modules` and `core.repo_map` SHALL use
`ModulesAssertions` helpers.

#### Scenario: Repo map consistency
- **WHEN** a test validates repo map alignment with core modules
- **THEN** it calls `ModulesAssertions(...).inventory_consistent()`.

### Requirement: Standardized Hamilton build harness usage
Tests that execute Hamilton targets or validate build outputs SHALL use `HamiltonBuildHarness`
instead of ad-hoc `BuildEnv` construction.

#### Scenario: Build target execution
- **WHEN** a test executes a Hamilton target and inspects outputs
- **THEN** it uses `HamiltonBuildHarness.open(...)` and associated harness APIs.

### Requirement: Manifest priming and artifacts helpers
Tests that need seeded manifests or artifact paths SHALL use `ManifestPriming` and
`HarnessArtifacts` utilities.

#### Scenario: Manifest-dependent checks
- **WHEN** a test checks manifest data without running a full build
- **THEN** it uses `ManifestPriming` to seed the manifest index.

### Requirement: Orchestration helper standardization
Orchestration test utilities SHALL use `HamiltonBuildHarness` and shared helper utilities
for environment setup and target execution.

#### Scenario: Orchestration setup
- **WHEN** orchestration helpers create build environments
- **THEN** they use harness utilities instead of constructing `BuildEnv` directly.

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

