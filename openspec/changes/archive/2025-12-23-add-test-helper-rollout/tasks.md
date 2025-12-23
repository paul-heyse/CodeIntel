## 1. Implementation
### 1.1 Ingestion migrations
- [x] Confirm `tests/ingestion/test_module_inventory.py` uses
      `modules_expected_from_repo_tree(...)` and golden diff helpers.
- [x] Migrate `tests/ingestion/test_docstrings_inventory.py` to materialize repo scan rows and
      use `ModulesAssertions.inventory_consistent()` for repo_map/modules parity.
- [x] Review `tests/ingestion/test_runner_plumbing.py` and `tests/ingestion/test_row_serialization.py`
      for any module inventory comparisons; no diff upgrades needed.

### 1.2 Storage migrations
- [x] Update `tests/storage/test_module_index.py` to use `format_module_map_diff(...)` for
      comparison failures.
- [x] Review `tests/storage/test_db_helpers.py`, `tests/storage/test_snapshot_scoping.py`,
      `tests/storage/test_gateway_factory.py`, and `tests/storage/test_gateway_helpers.py` for
      module inventory comparisons; no diff upgrades needed.

### 1.3 Graphs migrations
- [x] Update `tests/graphs/test_graph_validation_catalog.py` to use `ModulesAssertions`
      when validating module map behavior.

### 1.4 Build/Hamilton harness migrations
- [x] Migrate tests that execute targets or validate outputs to `HamiltonBuildHarness`:
      `tests/build/hamilton/test_graph_targets.py`
      `tests/build/hamilton/test_materializer.py`
      `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
      `tests/build/hamilton/test_metrics_targets.py`
      `tests/build/hamilton/test_multi_table_pipeline_template.py`
      `tests/build/hamilton/test_executor_pipeline_template.py`
      `tests/build/hamilton/test_coverage_targets.py`
      `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py`
- [x] Review `tests/build/hamilton/test_pr09_planner.py` and
      `tests/build/hamilton/test_pr10_manifest_index.py`; keep direct `BuildEnv` setup for
      FakeGateway unit testing.
- [x] Use `ManifestPriming` for manifest seeding without full builds
      (primed modules manifest in the call_graph fixture).
- [x] Use `HarnessArtifacts` for deterministic access to build output paths
      (writes pytest report in the call_graph fixture).

### 1.5 Orchestration helper migrations
- [x] Update `tests/_helpers/orchestration/provisioning.py` to use
      `HamiltonBuildHarness` + `ManifestPriming` + `HarnessArtifacts`.
- [x] Update `tests/_helpers/orchestration/graph_orchestration.py` to use
      `HamiltonBuildHarness` and `run_targets(...)`.

### 1.6 Analytics (optional)
- [x] Only migrate analytics tests if module inventory comparisons are present (none found).

### 1.7 Additional opportunities (optional)
- [x] Add a deterministic SCIP ingest test path using `HamiltonBuildHarness` +
      `HarnessArtifacts.write_dummy_scip_artifacts(...)`, keeping the current integration
      test for real SCIP binaries.
- [x] Align `tests/_helpers/hamilton_execution.py` / `tests/analytics/conftest.py` with
      `HamiltonBuildHarness` or a shared `HamiltonRuntime` fixture to avoid repeated DAG builds.
- [x] Replace any remaining `repo_root.rglob("*.py")` module inventories in helpers with
      `module_paths_expected_from_repo_tree(...)` where module filtering parity is needed.

## 2. Validation
Validation deferred per user confirmation (full pytest recently run; minor failures).
### 2.1 Targeted tests
- [ ] Run `tests/ingestion/test_module_inventory.py` and
      `tests/ingestion/test_docstrings_inventory.py`.
- [ ] Run `tests/graphs/test_graph_validation_catalog.py`.
- [ ] Run the migrated build/hamilton tests listed above.
- [ ] Run orchestration helper tests that exercise provisioning/graph orchestration.

### 2.2 Quality gates
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.

## 3. Documentation
### 3.1 Docs updates
- [x] Update internal test helper docs to reflect modules-first expectations, golden diffs,
      and Hamilton harness usage patterns.
