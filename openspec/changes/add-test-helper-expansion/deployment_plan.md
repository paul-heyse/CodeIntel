# Deployment Plan: Add-Test-Helper-Expansion

## Scope
This plan identifies where the new test helpers should be deployed and provides an
execution-ready sequence for rolling the changes into target tests.

## Helper Inventory (Created in add-test-helper-expansion)
- `tests/_helpers/harnesses/hamilton_build.py`
- `tests/_helpers/harnesses/graph_harness.py`
- `tests/_helpers/harnesses/analytics_harness.py`
- `tests/_helpers/harnesses/serving_harness.py`
- `tests/_helpers/harnesses/plan_status.py`
- `tests/_helpers/manifests.py`
- `tests/_helpers/hamilton_harness_artifacts.py`
- `tests/_helpers/tool_sandbox.py`
- `tests/_helpers/tool_payloads.py`
- `tests/_helpers/build_config_overrides.py`
- `tests/_helpers/assertions/target_record_assertions.py`
- `tests/_helpers/assertions/target_failures.py`
- `tests/_helpers/assertions/dependencies.py`
- `tests/_helpers/snapshots/tables.py`
- `tests/_helpers/orchestration/repo_writers.py`
- `tests/_helpers/orchestration/repo_registry.py`

## Target Matrix (Where to Deploy Helpers)
**Harness adoption and runtime alignment**
- `tests/analytics/conftest.py` — replace `HamiltonTestBuilder` usage with
  `HamiltonBuildHarness` or `AnalyticsTargetHarness`; keep session runtime reuse.
- `tests/_helpers/hamilton_execution.py` — deprecate in favor of
  `HamiltonBuildHarness`, or rewrap its logic around the new harness.
- `tests/_helpers/hamilton_fixtures.py` — align with harness config/reuse and
  eliminate duplicate BuildEnv wiring.

**Graph/analytics/serving target harnesses**
- `tests/build/hamilton/test_graph_targets.py` — use `GraphTargetHarness` for
  target execution and dataset assertions.
- `tests/build/hamilton/test_metrics_targets.py` — use `AnalyticsTargetHarness`
  to run `function_metrics` and related targets.
- `tests/build/hamilton/test_coverage_targets.py` — use `AnalyticsTargetHarness`
  for coverage targets and row count assertions.
- `tests/build/serving/test_publisher.py` — use `ServingTargetHarness` to publish
  snapshots and validate search artifacts.
- `tests/build/serving/test_pr90_search_index_builds.py` — use
  `ServingTargetHarness` for search index artifacts and snapshot validation.
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py` — align serving
  target execution with harness path.

**Tool realism and payload fixtures**
- `tests/ingestion/test_runner_plumbing.py` — replace inline JSON payloads with
  `pytest_report_payload`, `coverage_json_payload`, `scip_json_payload`.
- `tests/ingestion/test_scip_ingest.py` — use `HarnessArtifacts` for stub SCIP
  artifacts and `ToolSandbox` for integration path.
- `tests/ingestion/test_tools.py` — replace ad-hoc payloads with payload helpers
  and use `ToolSandbox` for deterministic subprocess paths.
- `tests/tools/test_tool_sandbox_integration.py` — use
  `ToolSandbox.install_default_stubs()` plus payload builders for content checks.

**Manifest lifecycle helpers**
- `tests/build/hamilton/native/test_skip_logic.py` — implement skipped/force/
  changed-input integration tests using `run_twice_and_assert_skip` and
  `prime_manifest`.
- `tests/build/hamilton/test_pr10_manifest_index.py` — adopt
  `load_manifest_index`/`prime_manifest` to reduce manual setup.
- `tests/build/hamilton/test_pr72_manifest_v2.py` — use manifest helpers for
  consistent priming and hash assertions.

**Target record assertions and failure helpers**
- `tests/build/hamilton/test_executor_pipeline_template.py` — replace inline
  status/row count checks with `assert_target_ok` and row count assertions.
- `tests/build/hamilton/test_multi_table_pipeline_template.py` — use
  `assert_partial_failure` for partial/failed run checks.
- `tests/build/hamilton/test_materializer.py` — apply schema/table assertions
  via `assert_table_schema_valid`/`assert_record_schemas_valid`.

**Plan/status summaries**
- `tests/build/hamilton/test_pr09_planner.py` — use `compute_plan_summary` and
  `format_plan_diff` for deterministic comparisons.
- `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py` — replace
  manual plan assertions with plan summary diffing.

**Table snapshot utilities**
- `tests/ingestion/test_docstrings_inventory.py` — replace manual SELECT +
  list comparisons with `snapshot_table` and `diff_table_snapshot`.
- `tests/ingestion/test_module_inventory.py` — use table snapshot utilities for
  stable module inventory comparisons.
- `tests/storage/test_module_index.py` — use `snapshot_table` for stable repo_map
  and modules table regression snapshots where appropriate.

**Repo fixtures + registry**
- `tests/_helpers/orchestration/graph_orchestration.py` — replace direct writers
  with registry-tagged fixtures for monorepo, generated noise, and scoped paths.
- `tests/_helpers/orchestration/provisioning.py` — route fixture selection via
  `repo_registry` for deterministic inventories.
- `tests/graphs/test_engine_nx.py` — use registry fixtures for repo_map seeding
  and inventory expectations.

**Config override helpers**
- `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py` — write config
  sections via `write_build_config_sections` and reload with `reload_build_config`.
- `tests/build/hamilton/test_pr10_manifest_index.py` — use config helper to
  mutate options hash inputs deterministically.

## Execution Plan (Phased Rollout)
**Phase 1 — Harness alignment**
- Convert `tests/analytics/conftest.py` and `tests/_helpers/hamilton_fixtures.py`
  to the base harness or analytics harness.
- Update `tests/_helpers/hamilton_execution.py` to forward to the new harness or
  mark for removal in follow-up cleanup.
- Acceptance: targeted analytics tests still pass with shared runtime reuse.

**Phase 2 — Tool realism**
- Convert ingestion tool tests to use payload builders and ToolSandbox stubs.
- Adopt `HarnessArtifacts` for scip/pytest/coverage artifacts.
- Acceptance: tool-driven ingestion tests pass without real binaries.

**Phase 3 — Target harness wrappers**
- Migrate graph, analytics, and serving target tests to their harness wrappers.
- Add dataset and artifact assertions via target record helpers.
- Acceptance: target tests assert row counts, datasets, and artifacts uniformly.

**Phase 4 — Manifest lifecycle**
- Implement skip/force/recompute integration tests with manifest helpers.
- Replace manual manifest priming with helper APIs.
- Acceptance: skip logic tests are deterministic and do not rely on ad-hoc state.

**Phase 5 — Repo fixtures and snapshots**
- Replace direct repo writers with registry lookups in orchestration helpers.
- Use table snapshot utilities in module/ingestion tests.
- Acceptance: repo fixtures produce consistent inventories across tests.

**Phase 6 — Plan/status helpers and config overrides**
- Update plan tests to use `plan_status` helpers for diffs and summaries.
- Apply config override helpers where option hashes are part of the test logic.
- Acceptance: plan tests remain deterministic with clearer diffs.

## Detailed Execution Planning (Per Target)
Each target listed in the matrix should be converted using a short, consistent
worksheet:
- Target file and test function name.
- Current behavior and pain point.
- Helper(s) to adopt.
- Expected change in assertions.
- Acceptance criteria and pytest command for the target.

This worksheet should be prepared before making code changes for each batch.
