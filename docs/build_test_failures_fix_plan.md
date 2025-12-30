# Build Test Failures Fix Plan (Updated)

## Scope

Target: `tests/build` (38 failures in `build/test-results/junit.xml`).
This plan captures the agreed fixes and exact file-level changes to bring the
suite back to green while preserving the current architecture.

## Execution Order (Recommended)

1. Fix core production regressions that cascade into many tests.
2. Update tests that are now misaligned with the new behavior.
3. Re-run focused subsets, then the full `tests/build` scope.

## Workstreams And Detailed Fixes

### 1. Iceberg materialization failures (10 tests, multiple harness failures)

**Symptoms**
- `tests/build/hamilton/test_materializer.py` expects `status == "succeeded"` but
  receives `"failed"` for Iceberg writes and validation flows.
- Downstream harness tests fail with `target_status` failed (graph, serving,
  analytics harnesses), likely because table materialization fails upstream.

**Root cause hypothesis**
- The Iceberg saver now routes contract validation through
  `resolve_validation_policy`, which depends on schema/contract services being
  configured. In test harness scenarios, those services are not always
  configured or the policy resolves to contract checks without an initialized
  provider.
- Validation failures (or missing declared schema) mark materialization as
  failed, causing the cascaded `target_status` failures.

**Fixes**
1. Ensure contract/schema services are configured for harness-driven materializer
   tests before invoking `IcebergDatasetSaver.save_data`.
   - Inject a configured runtime bundle and call
     `configure_schema_service()` / `configure_contract_service()` in
     `tests/build/hamilton/test_materializer.py` setup (or the shared harness
     fixture) so the policy can resolve declared schemas.
   - Alternatively, update `HamiltonBuildHarness.build_env()` or fixture setup
     to call `compose_runtime` and configure services once.

2. Add test-time introspection when a materialization failure occurs:
   - Update tests in `tests/build/hamilton/test_materializer.py` to assert
     `meta["status"]` then include `meta.get("error")` in failure messages to
     keep regressions diagnosable without re-running locally.

3. If validation policy resolution is still producing failures in tests that are
   not validating contract behavior, set `output_role="internal"` or
   `validate_outputs=False` in those specific tests (only where validation is
   out of scope), but keep strict validation tests unchanged.

4. Re-run `tests/build/hamilton/test_materializer.py` and
   `tests/build/hamilton/test_target_harness_wrappers.py` to confirm the
   cascading harness failures resolve.

**Files to change**
- `tests/build/hamilton/test_materializer.py`
- Possibly shared harness fixtures in `tests/_helpers/harnesses/hamilton_build.py`
  or session setup in `tests/conftest.py`

---

### 2. Contract service not configured (2 tests)

**Symptoms**
- `RuntimeError: ContractService has not been configured` in
  `tests/build/hamilton/test_import_time_schema_safety.py` and
  `tests/build/test_contract_resolution_seams.py`.

**Root cause**
- `iter_contracts()` and `iter_contracts_by_table_key()` depend on the build
  contract service, which is not always configured in test runs that construct
  runtimes manually.

**Fixes**
1. In tests that directly call `iter_contracts*`, explicitly initialize runtime
   configuration before the call:
   - Create a `RuntimeBundle` via `compose_runtime(env=..., config=...)`.
   - Call `configure_schema_service(runtime=runtime)` and
     `configure_contract_service(runtime=runtime)`.
2. For tests that verify lazy target metadata loading, preserve the existing
   semantics by resetting and then checking the provider state after
   `iter_contracts()`.

**Files to change**
- `tests/build/hamilton/test_import_time_schema_safety.py`
- `tests/build/test_contract_resolution_seams.py`

---

### 3. Planner recursion (4 tests)

**Symptoms**
- `RecursionError: maximum recursion depth exceeded` in
  `tests/build/hamilton/test_pr09_planner.py` and
  `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py`.

**Root cause hypothesis**
- Planning runtime composition includes planning nodes (`ci_plan`) and plan
  materialization saver nodes in the same graph. When the planner executes,
  the graph traversal picks up nodes that create a self-referential dependency
  (plan nodes referencing plan outputs), triggering DFS recursion.

**Fixes**
1. Exclude `ci_plan` materialization nodes from the plan execution graph when
   `materialize=False`:
   - In `src/codeintel/build/hamilton/planner.py`, restrict `final_vars` to
     `"plan"` only when `materialize=False` (already partially implemented) and
     **also** ensure the planning runtime is built with
     `ci.enable_planning_nodes=True` but without saver/materializer nodes that
     depend on the plan output.
2. If saver nodes are injected via `TargetSpecDescriptor` / `save_relation_table`,
   add a planning-only config flag (e.g., `ci.plan_materialization=false`) that
   disables plan saver nodes during `compute_plan` calls.
3. Update tests to pass `materialize=False` and to use the planning-only
   config flag (once implemented).

**Files to change**
- `src/codeintel/build/hamilton/planner.py`
- `src/codeintel/runtime/compose.py` (config propagation)
- Potentially `src/codeintel/build/hamilton/native/planning/plan_savers.py`
- Tests:
  - `tests/build/hamilton/test_pr09_planner.py`
  - `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py`

---

### 4. Saver graph validation errors (2 tests)

**Symptoms**
- `TargetSpecError: Contract DataSaver node not connected to target materialize
  node` in `tests/build/hamilton/test_dag_catalog_compiler.py` and
  `tests/build/hamilton/test_saver_declared_output_inventory.py`.

**Root cause**
- The tests create saver nodes that are not wired into the target anchor
  dependencies, so the compiler correctly flags the saver nodes as unanchored.

**Fixes**
1. In both tests, pass saver metadata outputs into the `t__alpha` anchor
   signature so they become direct dependencies of the target anchor.
2. Keep the saver-node naming stable; the intention is to keep validation strict
   (no change to production code required).

**Files to change**
- `tests/build/hamilton/test_dag_catalog_compiler.py`
- `tests/build/hamilton/test_saver_declared_output_inventory.py`

---

### 5. Loader node configuration and tag coverage (2 tests)

**Symptoms**
- `test_loader_nodes_disabled_by_config` expects query nodes to become external
  inputs when `ci_support_include_loader_nodes=False`, but the config is
  overridden by plugin config merges.
- `test_pr64_loader_tags_are_canonical` finds q__ nodes without loader tags for
  two analytics test tables.

**Root causes**
- `_merge_hamilton_config()` currently overwrites the base config with plugin
  config values, ignoring `ci_support_include_*` overrides.
- The two q__ nodes are declared by function parameters but no loader function
  is generated or tagged for those tables, so they appear as untagged external
  inputs.

**Fixes**
1. Preserve explicit caller overrides for `ci_support_include_*` flags when
   merging configs:
   - In `src/codeintel/runtime/compose.py` `_merge_hamilton_config`, only apply
     plugin config values for `ci_support_include_*` when not present in the
     base config.
2. For loader tag coverage:
   - Either generate loader nodes for
     `analytics.test_coverage_edges` and `analytics.test_graph_metrics_tests` by
     adding them to the loader registry, **or** update the test to tolerate
     external input q__ nodes without loader tags when no loader spec exists.
   - Preferred: add loader specs and tag them to keep tag coverage consistent.

**Files to change**
- `src/codeintel/runtime/compose.py`
- Loader registry or dataset config (where loader specs are declared)
- `tests/build/hamilton/test_pr12_loader_nodes.py`
- `tests/build/hamilton/test_pr64_loader_tags_are_canonical.py`

---

### 6. Extraction dataset tags (1 test)

**Symptoms**
- `tests/build/hamilton/test_extraction_target_tags.py` reports
  `ast__node_rows` missing `node_type=dataset` tag.

**Root cause**
- The extraction target nodes are generated dynamically; tagging uses
  `tagged_attach_node`, but the tag metadata may be lost if the decorator is
  overwritten or the node name does not map to the generated saver node.

**Fixes**
1. Confirm the `TableOutputSpec(node_name=...)` is used as the actual node name.
2. If tagging is stripped, ensure `tagged_attach_node()` sees the tag decorators
   for the generated rows node by attaching tags *after* the saver decorator is
   applied.
3. Add a regression check in the template code (optional) to assert tag metadata
   on the generated node during attachment.

**Files to change**
- `src/codeintel/build/hamilton/native/patterns/target_builder.py`
- (If necessary) `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

---

### 7. Contract provider parity (2 tests)

**Symptoms**
- Cache test expects `get_contract_for_table_key` to return the cached instance.
- Validation profile defaults to strict but is now `lenient`.

**Root cause**
- `get_contract_for_table_key()` bypasses `_get_enriched_contract_for_table_key`
  when `settings is None`, so LRU cache is unused.
- `BuildEnv.validation_mode` defaults to `LENIENT`, which influences validation
  profile resolution in `resolve_validation_policy` and contract defaults.

**Fixes**
1. Update `get_contract_for_table_key()` to use `_get_enriched_contract_for_table_key`
   when `settings is None`.
2. Decide on canonical default validation profile for tests:
   - If strict is required, set `validation_mode=STRICT` in test envs or adjust
     the expectation to `lenient` where appropriate.

**Files to change**
- `src/codeintel/build/schemas/contract_service.py`
- `tests/build/hamilton/test_pr68_contract_provider_parity.py`

---

### 8. JSON schema mapping for DECIMAL(38,0) (1 test)

**Symptoms**
- `test_decimal38_maps_to_integer` expects integer, actual mapping returns number.

**Root cause hypothesis**
- `_json_schema_type_for_column_type()` should return integer when scale=0.
  If not, the incoming column type may not be normalized as `DECIMAL(38,0)` or
  the regex match fails on the normalized value.

**Fixes**
1. Normalize decimal type before mapping:
   - In `src/codeintel/core/schemas/json_schema_gen.py`, ensure the normalized
     string is compacted (upper + no spaces) and compare using the decimal regex.
2. Add a unit test to validate `DECIMAL(38,0)` and `DECIMAL(10,0)` both map to
   integer, while non-zero scale maps to number.

**Files to change**
- `src/codeintel/core/schemas/json_schema_gen.py`
- `tests/build/hamilton/test_pr73_json_schema_generation.py`

---

### 9. Serving snapshot search index failures (4 tests + harness failures)

**Symptoms**
- `Search index build failed` for publisher tests and serving harness runs.
- Missing lineage tables failure message does not match in one test.

**Root cause**
- `ServingSnapshotService._build_search_index()` requires
  `docs.search_documents` to be non-empty. Tests that build the snapshot do not
  always seed this table before publishing.
- Missing lineage tests run after search index failure, so the error message is
  for the search index, not the lineage check.

**Fixes**
1. Seed search documents in publisher setup:
   - In `tests/build/serving/test_publisher.py`, call
     `build_search_documents_table()` and insert at least one row (use the
     helper `_ensure_search_documents()` from
     `tests/_helpers/serving_snapshot_factory.py`).
2. For the missing lineage test, ensure search docs are present so the lineage
   check is the first failure and the error message matches.
3. For serving harness tests, ensure the harness seeds search docs or relax the
   harness to skip search index construction when not relevant.

**Files to change**
- `tests/build/serving/test_publisher.py`
- `tests/_helpers/serving_snapshot_factory.py` (if adding a reusable helper)
- Possibly `tests/build/serving/test_pr90_search_index_builds.py` to reuse helper

---

### 10. Error message mismatch (1 test)

**Symptoms**
- `tests/build/test_errors.py` expects "Add schema" but message is
  "Register schema for 'core.ast_nodes' in the table registry".

**Fix**
- Update the expected message to the new wording, or align the production error
  text to the previous phrasing if that is still the canonical message.

**Files to change**
- `tests/build/test_errors.py` (or the error source if reverting the message)

---

### 11. State missing/blocked expectation (1 test)

**Symptoms**
- `tests/build/test_state.py::test_empty_state_is_missing` expects
  `("modules", "typing")` but now only `("modules",)` is missing.

**Root cause**
- State propagation now treats downstream targets as blocked rather than missing.

**Fix**
- Update expected missing to `("modules",)` and add `"typing"` to blocked (or
  update logic if behavior should revert).

**Files to change**
- `tests/build/test_state.py`

---

## Validation Checklist

1. Focused runs:
   - `uv run pytest tests/build/hamilton/test_materializer.py`
   - `uv run pytest tests/build/hamilton/test_pr09_planner.py`
   - `uv run pytest tests/build/serving/test_publisher.py`
2. Full scope:
   - `uv run pytest tests/build`

## Notes

- Where fixes require `python` execution for inspection or for generating
  artifacts, use `uv run` exclusively.
- Avoid weakening validation in production; only relax validation in tests where
  validation behavior is not under test.
