## 1. Implementation
### 1.1 SCIP harness coverage
- [x] Add a harness-based SCIP test path that executes through `HamiltonBuildHarness`
      when `scip-python` and `scip` are available on PATH.
- [x] Validate `TargetRunRecord` row_counts include `core.scip_symbols` and
      `core.scip_occurrences`, and that artifact paths exist.
- [x] Keep the deterministic artifact-based SCIP test path using
      `HarnessArtifacts.write_dummy_scip_artifacts(...)`.

### 1.2 Modules-first seeding parity
- [x] Replace helper-level `repo_root.rglob("*.py")` inventories with
      `modules_expected_from_repo_tree(...)` or `module_paths_expected_from_repo_tree(...)`
      in helper utilities that seed core.modules/core.repo_map.
- [x] Ensure any derived path lists used for module inventory assertions are sourced from
      modules-first helpers.

### 1.3 Graph inventory consistency checks
- [x] Add `ModulesAssertions.inventory_consistent()` to graph tests that depend on module
      catalog correctness, starting with `tests/graphs/test_engine_nx.py`.
- [x] Verify the assertions are placed after module inserts and before graph loader calls.
- [x] Expand consistency checks to additional `tests/graphs/*` cases and orchestration seed
      packs that insert module rows directly.

### 1.4 Golden diffs for analytics module maps
- [x] Update analytics module/path mapping comparisons to use
      `format_missing_extra(...)` or `format_module_map_diff(...)`.
- [x] Use `module_map_from_path_map(...)` when comparing `path -> module` maps.
- [x] Apply the same golden diff helpers to docs export tests and storage repository tests that
      compare module/path lists or maps.

### 1.5 Expanded harness and manifest usage
- [x] Migrate ingestion target tests (docstrings, typing, tests_ingest, coverage_ingest) to
      execute through `HamiltonBuildHarness` where they currently bypass the harness.
- [x] Use `ManifestPriming` and `HarnessArtifacts` for tests that seed manifests or artifacts
      manually (schema manifest, export targets, serving_artifacts).

### 1.6 Shared runtime fixture alignment
- [x] Align `tests/_helpers/hamilton_fixtures.py` with the shared runtime fixture or
      `HamiltonBuildHarness` to reduce duplicated BuildEnv wiring.
- [x] Expose or reuse a `HamiltonTestBuilder` configured with the shared runtime for
      analytics/fixture usage.
- [x] Route remaining BuildEnv fixtures in analytics/graphs helpers through the shared runtime
      or harness path.

### 1.7 Tool sandbox adoption
- [x] Use `tool_sandbox` in integration tests that currently require real tool binaries so
      subprocess execution remains deterministic in CI.

## 2. Validation
Validation deferred per user confirmation (full pytest recently run; minor failures).
### 2.1 Targeted tests
- [x] Run `tests/ingestion/test_scip_ingest.py` (harness + deterministic paths).
- [x] Run `tests/graphs/test_engine_nx.py`.
- [x] Run `tests/analytics/resources/test_provider_factory.py` and
      `tests/analytics/test_dependencies.py`.
- [ ] Run tests that exercise `tests/_helpers/hamilton_fixtures.py`.
- [!] Run targeted ingestion tests (docstrings, typing, tests_ingest, coverage_ingest).
      `tests/ingestion/test_docstrings_inventory.py` currently failing with duplicate
      build run id in `build.runs`.
- [x] Run graph tests updated with ModulesAssertions consistency checks.
- [x] Run analytics/docs/storage tests updated with golden diffs.
- [ ] Run integration tests updated to use tool_sandbox.

### 2.2 Quality gates
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.

## 3. Documentation
### 3.1 Docs updates
- [x] Update test helper docs to reference expanded harness usage, module consistency checks,
      golden diffs in analytics/docs/storage, and tool_sandbox integration guidance.
