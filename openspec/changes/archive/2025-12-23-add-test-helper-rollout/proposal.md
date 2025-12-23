# Change: Roll out test helper toolsets across module-related tests

## Status
Archived with validation deferred per user confirmation (full pytest recently run; minor failures).

## Why
Module inventory and Hamilton build tests currently use ad-hoc setup and low-signal diffs.
Standardizing on the new helpers improves consistency, reduces boilerplate, and yields clearer
failures while keeping tests aligned with production behavior.

## What Changes
- Adopt modules-first expectations (`modules_expected_from_repo_tree` / `modules_expected_from_env`)
  for module inventory tests.
- Replace ad-hoc module inventory diffs with `ModulesAssertions` and golden diff formatters.
- Migrate build/hamilton tests that construct `BuildEnv` directly to `HamiltonBuildHarness` where
  they execute targets or validate build outputs.
- Standardize orchestration test setup with `HamiltonBuildHarness`, `ManifestPriming`, and
  `HarnessArtifacts`.
- Keep storage DB-level tests intact while upgrading assertion messaging when module inventory
  comparisons exist.

## Scope Inventory (Step 1)
The rollout focuses on tests that either:
- compare module inventories (paths or module maps), or
- execute Hamilton targets with direct `BuildEnv` construction, or
- duplicate orchestration/build setup logic.

### Ingestion
- `tests/ingestion/test_module_inventory.py`: replaces hard-coded path lists and manual diffs.
- `tests/ingestion/test_docstrings_inventory.py`: align inventory consistency assertions.
- Keep DB/query behavior tests (`test_runner_plumbing.py`, `test_row_serialization.py`,
  `test_db_queries.py`) largely unchanged unless they compare inventories.
 - Optional: `tests/ingestion/test_scip_ingest.py` can use `HamiltonBuildHarness` plus
   `HarnessArtifacts.write_dummy_scip_artifacts(...)` for deterministic SCIP outputs, keeping
   the current integration test for real binaries.

### Storage
- `tests/storage/test_module_index.py`: module map behavior and diff clarity.
- Repo map/modules integration tests (`test_db_helpers.py`, `test_snapshot_scoping.py`,
  `test_gateway_factory.py`, `test_gateway_helpers.py`) only change diff messaging if needed.

### Graphs
- `tests/graphs/test_graph_validation_catalog.py`: module map consistency when core.modules is
  empty.

### Build / Hamilton
Targets with direct `BuildEnv` setup and output validation:
- `tests/build/hamilton/test_graph_targets.py`
- `tests/build/hamilton/test_materializer.py`
- `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
- `tests/build/hamilton/test_metrics_targets.py`
- `tests/build/hamilton/test_multi_table_pipeline_template.py`
- `tests/build/hamilton/test_executor_pipeline_template.py`
- `tests/build/hamilton/test_pr09_planner.py`
- `tests/build/hamilton/test_pr10_manifest_index.py`
- `tests/build/hamilton/test_coverage_targets.py`
- `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py`

### Orchestration helpers
- `tests/_helpers/orchestration/provisioning.py`
- `tests/_helpers/orchestration/graph_orchestration.py`

### Helper alignment (optional)
- `tests/_helpers/hamilton_execution.py` and `tests/analytics/conftest.py` can be aligned to
  `HamiltonBuildHarness` or a shared `HamiltonRuntime` to avoid repeated DAG builds and keep
  skip/manifest behavior consistent across helpers.

### Analytics (optional)
- `tests/analytics/conftest.py` uses `HamiltonTestBuilder`; migrate only if tests depend on
  module inventory or benefit from harness alignment.

## Adoption Patterns (Step 2)
### Ingestion tests
- Use `modules_expected_from_repo_tree(...)` for expected inventories.
- Use `ModulesAssertions` for repo_map/modules consistency checks.
- Use `format_missing_extra(...)` for path list diffs and `format_module_map_diff(...)` for
  module map diffs.
- Invert `path -> module` maps via `module_map_from_path_map(...)` before diffing.
- For SCIP ingest tests, use `HamiltonBuildHarness` + `HarnessArtifacts.write_dummy_scip_artifacts(...)`
  when a deterministic, no-toolchain path is desired; keep a separate integration test that
  still relies on real binaries.

### Storage tests
- Keep DB-level tests intact.
- Use `format_module_map_diff(...)` where module inventory comparisons exist.
- Prefer `load_modules_module_map(...)` / `load_repo_map_modules(...)` over ad-hoc SQL parsing.

### Build/Hamilton tests
- Replace direct `BuildEnv` construction with `HamiltonBuildHarness` in tests that execute
  targets or validate build outputs.
- Use `ManifestPriming` for manifest-related tests that should not run full builds.
- Use `HarnessArtifacts` for deterministic artifact access.

### Graphs tests
- Use `ModulesAssertions` where module inventories affect graph validation behavior.

### Orchestration helpers/tests
- Standardize on `HamiltonBuildHarness` and `run_targets(...)` for setup and execution.
- Use `ManifestPriming` and `HarnessArtifacts` for pre-seeded manifests and artifact paths.

### Helper alignment (optional)
- Wrap `HamiltonTestBuilder` around `HamiltonBuildHarness` or use a shared `HamiltonRuntime`
  fixture to reduce repeated DAG builds and align runtime configuration.

### Analytics tests
- Migrate only when module inventory is part of the assertions.
- Keep `HamiltonTestBuilder` otherwise.

## Migration Checklists (Step 2)
### Ingestion
- Replace hard-coded path lists or `rglob("*.py")` with
  `modules_expected_from_repo_tree(...)`.
- Replace `load_module_map(...)` comparisons with
  `module_map_from_path_map(...)` + `format_module_map_diff(...)`.
- Use `ModulesAssertions(...).inventory_consistent()` for repo_map/modules parity.
 - For SCIP ingest tests, add a harness-based path that uses `HarnessArtifacts` to stage
   SCIP artifacts without invoking real binaries.

### Storage
- Use `format_module_map_diff(...)` or `format_missing_extra(...)` for module inventory diffs.
- Use `load_modules_module_map(...)` / `load_repo_map_modules(...)` for normalized maps.

### Build/Hamilton
- Use `HamiltonBuildHarness.open(...)` and harness APIs where targets are executed.
- Replace direct `BuildEnv(...)` construction in these cases.
- Use `ManifestPriming` for manifest-only assertions.

### Orchestration
- Replace ad-hoc env/gateway setup with `HamiltonBuildHarness`.
- Standardize target execution via `execute_hamilton_targets(...)`.

### Graphs
- Adopt `ModulesAssertions` for tests validating module map behavior.

## Non-goals
- Do not refactor tests that only validate DB/query behavior or schema wiring unless module
  inventory comparisons are present.
- Do not rewrite analytics tests unless module inventory is a primary concern.

## Risks and Mitigations
- Risk: Harness migrations obscure test intent. Mitigation: keep assertions close to old
  behavior and preserve named contexts.
- Risk: Inventory diffs hide ordering changes. Mitigation: diff helpers explicitly call out
  missing/extra and path changes.

## Impact
- Affected specs: test-helpers (new)
- Affected code: ingestion, graphs, build/hamilton, orchestration, and storage tests touching
  module inventory or build harness setup.
