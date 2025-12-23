# Change: Expand production-parity test helpers beyond modules

## Why
Current modules-first helpers improve realism for ingestion, but the broader pipeline
(graphs, analytics, serving, and tool-driven ingestion) still lacks production-parity
harnesses, tool outputs, and manifest lifecycle utilities. This change standardizes the
helper layer so realistic execution can be exercised deterministically across all
target families with minimal boilerplate.

## Goals
- Execute Hamilton targets in tests via the same DAG and executor as production.
- Provide deterministic tool outputs and failure modes without requiring real binaries.
- Provide target-family harnesses and assertions for graph, analytics, and serving.
- Make skip/recompute behavior testable via manifest lifecycle helpers.
- Provide deterministic repo fixtures and table snapshot utilities for regression tests.
- Provide guardrails for upstream dependency status and failure assertions.
- Provide plan/status and config override helpers for deterministic build behavior tests.

## Non-Goals
- No changes to production target logic or tool plugins.
- No changes to CLI behavior or runtime configuration.
- No new external dependencies beyond what tests already use.

## What Changes
- Base Hamilton harness:
  - `tests/_helpers/harnesses/hamilton_build.py` executes targets via
    `HamiltonBuildExecutor` with multi-target runs and returns `TargetRunRecord`s.
  - Supports `BuildEnv` overrides for profile, force_targets, strict_contracts,
    validate_outputs, and gateway mode (memory or on-disk).
- Tool realism layer:
  - Extend `tests/_helpers/fakes/tools.py` so `FakeToolRunner` always writes
    deterministic outputs for pytest, coverage, scip, pyright, pyrefly and supports
    per-tool payloads, return codes, and error conditions.
  - Add `tests/_helpers/tool_sandbox.py` to install stub executables in a temp
    `bin/` and exercise the real ToolRunner subprocess path for integration tests.
- Target-family harness wrappers:
  - `tests/_helpers/harnesses/graph_harness.py` with default graph targets and
    dataset assertions.
  - `tests/_helpers/harnesses/analytics_harness.py` with default analytics targets
    and snapshot assertions.
  - `tests/_helpers/harnesses/serving_harness.py` for serving snapshot publish and
    search index artifacts.
- Manifest lifecycle helpers:
  - `tests/_helpers/manifests.py` to load manifests, prime manifests with computed
    input hashes, and assert skip/recompute behavior across runs.
- Assertion helpers:
  - `tests/_helpers/assertions/target_records.py` for status, row_counts, datasets,
    artifacts, and schema validation of produced tables.
- Deterministic repo fixtures:
  - Extend `tests/_helpers/orchestration/repo_writers.py` with monorepo layouts,
    generated file noise, large file filtering, and scoped path cases.
- Table snapshot utilities:
  - `tests/_helpers/snapshots/tables.py` to dump sorted rows and produce readable
    diffs for regression tests.
- Upstream status guard helpers:
  - Normalize downstream dependency checks with configurable rules for skipped targets.
- Generalized manifest priming:
  - Target-agnostic priming that computes dependency-aware input hashes.
- Build plan/status harness helpers:
  - Deterministic assertions for compute/skip/blocked decisions with readable diffs.
- Tool payload fixture builders:
  - Valid minimal payloads for pytest, scip, and coverage to pass plugin parsing.
- Config override helpers:
  - Write build config sections and reload BuildConfig into a harness.
- Repo fixture registry:
  - Request fixtures by tag/intent with expected module inventories.
- Failure scenario helpers:
  - Consistent assertions for partial/failed TargetRunRecord bundles.
- Fixture wiring and docs updates:
  - `tests/conftest.py` to expose harness fixtures and tool sandbox.
  - `docs/tests_refinement/` updated to document new helpers and migration patterns.

## Success Criteria
- Tests can run `modules`, `call_graph`, `function_metrics`, and `serving_artifacts`
  through the base harness and return `TargetRunRecord`s.
- Fake tool runner produces deterministic output files and stdout payloads for
  tool-driven targets without relying on real binaries.
- Manifest helpers can assert skip behavior on a second run and recompute when inputs
  change, using real stored manifests.
- Table snapshot utilities produce stable output with diffs that highlight row changes.
- Upstream-guard helpers standardize dependency status checks across tests.
- Manifest priming can generate valid input hashes for any target with dependencies.
- Plan/status harness helpers produce deterministic, readable diffs.
- Tool payload fixtures cover pytest/scip/coverage and pass plugin parsing.
- Config override helpers reliably update options hashes for targets under test.
- Repo fixture registry resolves fixtures by tag and documents expectations.
- Failure scenario helpers make partial-failure assertions consistent across tests.

## Impact
- Affected specs: `test-harnesses` (new capability)
- Affected code:
  - `tests/_helpers/harnesses/`
  - `tests/_helpers/fakes/tools.py`
  - `tests/_helpers/manifests.py`
  - `tests/_helpers/assertions/`
  - `tests/_helpers/orchestration/repo_writers.py`
  - `tests/_helpers/snapshots/`
  - `tests/conftest.py`
  - `docs/tests_refinement/`
