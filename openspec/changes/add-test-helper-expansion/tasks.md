## 1. Implementation
- [x] 1.1 Base Hamilton harness
  - Add `tests/_helpers/harnesses/hamilton_build.py` with `open(...)`, `wrap(...)`,
    `run_targets(...)`, `record(...)`, and `with_*` env overrides.
  - Ensure multi-target execution uses a single DAG run via `HamiltonBuildExecutor`.
  - Support memory and on-disk gateway modes; on-disk uses `GatewayFactory` with
    `StorageConfig.for_ingest(...)`.
  - Re-export in `tests/_helpers/__init__.py`.
- [x] 1.2 Tool realism (FakeToolRunner)
  - Extend `tests/_helpers/fakes/tools.py` to always create `options.output_path`
    files when present, with per-tool payloads for pytest, coverage, scip, pyright,
    pyrefly.
  - Add `stdout_payloads`, `returncodes`, `raise_on`, and `not_found` controls.
  - Provide default payloads that pass plugin parsing for each tool.
- [x] 1.3 Tool sandbox (integration realism)
  - Add `tests/_helpers/tool_sandbox.py` to install stub executables in a temp
    `bin/` and expose `tools_config()` and PATH overrides.
  - Provide stubs for `pytest`, `scip-python`, `scip`, `pyright`, `pyrefly`.
- [x] 1.4 Manifest lifecycle helpers
  - Add `tests/_helpers/manifests.py` with:
    - `load_manifest_index(gateway, snapshot)`
    - `prime_manifest(...)` with computed input hash and options hash
    - `run_twice_and_assert_skip(harness, target)`
    - `assert_skipped(record)` and `assert_succeeded(record)`
  - Include helpers to compute input hashes using existing hashing utilities.
- [x] 1.5 Target record assertions
  - Add `tests/_helpers/assertions/target_records.py` with helpers for status,
    datasets, artifacts, row_counts, and schema validation against
    `SCHEMA_REGISTRY`.
  - Re-export in `tests/_helpers/assertions/__init__.py`.
- [x] 1.6 Deterministic repo fixtures
  - Extend `tests/_helpers/orchestration/repo_writers.py` with fixture writers for:
    - multi-language monorepo layouts
    - generated file noise and ignore cases
    - large file and max-size filtering
    - scope-path filtering scenarios
  - Each writer returns expected module maps for assertion helpers.
- [x] 1.7 Target-family harness wrappers
  - Add `tests/_helpers/harnesses/graph_harness.py` for graph target sets and
    dataset assertions.
  - Add `tests/_helpers/harnesses/analytics_harness.py` for analytics target sets
    and snapshot helpers.
  - Add `tests/_helpers/harnesses/serving_harness.py` for serving snapshot publish
    and search index artifacts.
- [x] 1.8 Table snapshot utilities
  - Add `tests/_helpers/snapshots/tables.py` to dump sorted rows for tables and
    generate diffs for regression tests.
- [x] 1.9 Upstream status guard helpers
  - Add a shared helper to normalize upstream status checks (treat skipped as
    cached success when configured).
- [x] 1.10 Build plan/status harness helpers
  - Add helpers to compute build plans/status and format diffs for regression tests.
- [x] 1.11 Tool payload fixture builders
  - Add payload builders for pytest, scip, and coverage that pass plugin parsing.
- [x] 1.12 Config override helpers
  - Add helpers to write build config sections and reload BuildConfig into a harness.
- [x] 1.13 Repo fixture registry
  - Add a fixture registry mapping tags to repo writers and expected inventories.
- [x] 1.14 Failure scenario helpers
  - Add assertion helpers for partial/failed TargetRunRecord bundles.
- [x] 1.15 Fixture wiring
  - Update `tests/conftest.py` with fixtures for the base harness, tool sandbox,
    graph/analytics/serving harness wrappers, and session-scoped Hamilton runtime.
- [x] 1.16 Documentation updates
  - Update `docs/tests_refinement/` with new helper usage, migration guidance,
    and integration vs fast test guidance.
- [x] 1.17 Helper validation tests
  - Add focused tests that exercise:
    - ingestion via the base harness (modules or typing)
    - graph targets (call_graph or import_graph)
    - analytics target (function_metrics)
    - serving snapshot publish (serving_artifacts)
  - Include at least one ToolSandbox integration test.

## 2. Validation
- [ ] 2.1 Run `openspec validate add-test-helper-expansion --strict`.
- [ ] 2.2 Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- [ ] 2.3 Run targeted pytest subsets for updated areas, then segment by major dirs
  (ingestion/graph/analytics/serving) as described in `AGENTS.md`.
