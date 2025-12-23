## ADDED Requirements

### Requirement: Hamilton DAG Execution Harness
The system SHALL provide a test harness that executes Hamilton targets through the
production HamiltonBuildExecutor, supports multi-target runs, and returns per-target
TargetRunRecord results. The harness SHALL allow BuildEnv overrides for profile,
force_targets, strict_contracts, validate_outputs, and gateway mode (memory or on-disk).

#### Scenario: Execute multiple targets through the DAG
- **WHEN** a test calls the harness to run targets ["modules", "goids"]
- **THEN** the harness executes the full dependency closure once and returns records
  for both targets.

### Requirement: Tool Execution Realism
The system SHALL provide a deterministic tool execution layer for tests that can
materialize tool outputs (files and stdout) for pytest, coverage, scip, pyright, and
pyrefly. The tool layer SHALL support per-tool payloads, return codes, and error
conditions, and SHALL be injectable via ToolsConfig or Providers.

#### Scenario: Simulate SCIP and pytest tool outputs
- **WHEN** a test configures tool payloads for scip and pytest
- **THEN** the tool layer writes the expected output files and stdout for the plugin
  to parse, and the target completes without requiring real binaries.

### Requirement: Target-Class Harness Wrappers
The system SHALL provide harness wrappers for graph, analytics, and serving target
families with default target sets, convenience assertions, and on-disk gateway support
for parallel execution.

#### Scenario: Run graph targets with a graph harness
- **WHEN** a test uses the graph harness to run ["call_graph", "import_graph"]
- **THEN** the harness executes those targets via the base Hamilton harness and
  exposes helpers to assert graph dataset outputs.

### Requirement: Manifest Lifecycle Helpers
The system SHALL provide helpers to load manifests, prime manifests with computed
input hashes, and assert skip or recompute behavior across multiple runs.

#### Scenario: Verify skip on second run
- **WHEN** a target is run twice with no input changes
- **THEN** the helper asserts that the second run produces a skipped or cached
  TargetRunRecord with a matching manifest.

### Requirement: Target Record and Dataset Assertions
The system SHALL provide assertion helpers for TargetRunRecord status, dataset and
artifact presence, row_counts consistency, and schema validation against the schema
registry for produced datasets.

#### Scenario: Validate target outputs
- **WHEN** a test receives a TargetRunRecord for an analytics target
- **THEN** the helper asserts success status, expected dataset refs, and schema
  validity for produced tables.

### Requirement: Deterministic Repo Fixture Library
The system SHALL provide deterministic repo fixture writers that generate test repos
including multi-language monorepos, generated file noise, large files, and scoped path
layouts, with stable expected module inventories.

#### Scenario: Build a monorepo fixture with generated files
- **WHEN** a test creates a monorepo fixture with generated files enabled
- **THEN** generated files are excluded from module inventory expectations and the
  fixture exposes deterministic expected paths.

### Requirement: Table Snapshot Utilities
The system SHALL provide deterministic table snapshot utilities that export sorted
rows for selected tables and produce readable diffs for regression tests.

#### Scenario: Snapshot an analytics table
- **WHEN** a test snapshots `analytics.function_metrics`
- **THEN** the snapshot output is stable across runs and diffs highlight row changes.

### Requirement: Upstream Status Guard Helpers
The system SHALL provide helpers that normalize dependency status checks for
downstream targets, with configurable rules that can treat `skipped` as cached
success when appropriate.

#### Scenario: Allow skipped upstream target
- **WHEN** a downstream test calls the upstream guard helper with `allow_skipped=True`
- **THEN** a `skipped` upstream record is accepted and the test proceeds.

### Requirement: Generalized Manifest Priming
The system SHALL provide target-agnostic manifest priming helpers that compute
dependency-aware input hashes for any target and persist manifests for skip tests.

#### Scenario: Prime a manifest for a non-modules target
- **WHEN** a test primes a manifest for `call_graph` with dependency manifests
- **THEN** the saved manifest input hash matches the computed dependency hash.

### Requirement: Build Plan and Status Harness
The system SHALL provide helpers to compute build plans and build status for a
harness and format deterministic diffs for compute/skip/blocked decisions.

#### Scenario: Validate blocked plan entries
- **WHEN** a test computes a plan for `function_metrics` without required upstreams
- **THEN** the plan helper reports `blocked` with a readable diff of dependencies.

### Requirement: Tool Payload Fixture Builders
The system SHALL provide payload fixtures that emit minimal valid payloads for
pytest, coverage, and scip that pass plugin parsing in tests.

#### Scenario: Generate a pytest JSON payload
- **WHEN** a test requests a pytest payload fixture
- **THEN** the payload parses into a non-empty TestReport without errors.

### Requirement: Config Override Helpers
The system SHALL provide helpers to write build config sections and reload the
BuildConfig into a harness for options-hash tests.

#### Scenario: Update build config and reload
- **WHEN** a test writes a target options override and reloads the BuildConfig
- **THEN** the target options hash changes for subsequent runs.

### Requirement: Repo Fixture Registry
The system SHALL provide a registry that maps fixture tags to repo writers and
expected module inventories so tests can request fixtures by intent.

#### Scenario: Resolve a fixture by tag
- **WHEN** a test asks for a fixture tagged `monorepo`
- **THEN** the registry returns a repo writer and expected module inventory.

### Requirement: Failure Scenario Helpers
The system SHALL provide assertion helpers for partial and failed TargetRunRecord
bundles, including expected failed/succeeded/skipped sets.

#### Scenario: Assert partial failure bundle
- **WHEN** a test receives a mixed TargetRunRecord bundle
- **THEN** the helper asserts the expected failure and success sets.

### Requirement: Snapshot Diff Helpers for Large Tables
The system SHALL provide snapshot diff helpers that support column subsets and
stable row hashing for large tables to keep diffs readable.

#### Scenario: Diff large table snapshot with row hashing
- **WHEN** a test snapshots a large table with hashing enabled
- **THEN** the diff output highlights changed rows without full table dumps.
