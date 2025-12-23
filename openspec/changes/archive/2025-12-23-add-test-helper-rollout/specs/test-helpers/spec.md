## ADDED Requirements

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
