# Phase 3 Implementation Plan: Target Harness Wrappers

## Goal
Migrate graph, analytics, and serving target tests to the new harness wrappers
and standardize dataset/artifact assertions using the new helper APIs.

## Scope (Phase 3 Targets)
**Graph harness usage**
- `tests/build/hamilton/test_graph_targets.py`
- `tests/graphs/test_engine_nx.py` (only if harness execution paths are added)

**Analytics harness usage**
- `tests/build/hamilton/test_metrics_targets.py`
- `tests/build/hamilton/test_coverage_targets.py`
- `tests/analytics/resources/test_provider_factory.py` (if it constructs build runs)

**Serving harness usage**
- `tests/build/serving/test_publisher.py`
- `tests/build/serving/test_pr90_search_index_builds.py`
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`

**Assertion standardization**
- `tests/_helpers/assertions/target_record_assertions.py`
- `tests/_helpers/assertions/target_failures.py`
- `tests/_helpers/assertions/dependencies.py`

## Prerequisites
- Phase 1 harness alignment completed.
- Phase 2 tool realism completed.
- `GraphTargetHarness`, `AnalyticsTargetHarness`, `ServingTargetHarness` exported
  in `tests/_helpers/harnesses/__init__.py`.

## Execution Plan (Detailed, Execution-Ready)
### 1) Graph target test migration
**Targets**
- `tests/build/hamilton/test_graph_targets.py`

**Actions**
- Replace manual `HamiltonBuildHarness` setup with `graph_target_harness` fixture
  (from `tests/conftest.py`) or `GraphTargetHarness.open(...)`.
- Use harness wrapper convenience methods to run graph targets:
  - `run_graph_targets()` or `run(...)` depending on wrapper API.
- Replace manual row count/status checks with:
  - `assert_target_ok(...)`
  - `assert_record_row_counts(...)`
  - `assert_record_has_datasets(...)` as appropriate.
- Where graph tables are validated, use `GraphTargetHarness.assert_graph_tables(...)`
  or `TargetRecordAssertions` for dataset keys.

**Acceptance Criteria**
- No change in expected row counts or dataset keys.
- All target executions routed through `GraphTargetHarness`.

### 2) Analytics target test migration
**Targets**
- `tests/build/hamilton/test_metrics_targets.py`
- `tests/build/hamilton/test_coverage_targets.py`

**Actions**
- Replace ad-hoc harness setup with `analytics_target_harness` fixture or
  `AnalyticsTargetHarness.open(...)`.
- Use wrapper methods to execute analytics targets:
  - `run_metrics_targets()` for function metrics/risk factors.
  - Explicit `run(...)` for coverage targets as needed.
- Replace manual status/row count assertions with:
  - `assert_target_ok(...)`
  - `assert_record_row_counts(...)`
  - `assert_record_has_datasets(...)`.
- Where analytics tables are validated, use `assert_table_schema_valid(...)`
  or `assert_record_schemas_valid(...)` if the test is schema-driven.

**Acceptance Criteria**
- All analytics targets executed via `AnalyticsTargetHarness`.
- No behavioral change in row count expectations.

### 3) Serving target test migration
**Targets**
- `tests/build/serving/test_publisher.py`
- `tests/build/serving/test_pr90_search_index_builds.py`
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`

**Actions**
- Replace manual harness setup with `serving_target_harness` fixture or
  `ServingTargetHarness.open(...)`.
- Use wrapper methods to publish snapshots and assert artifacts:
  - `publish_snapshot(...)`
  - `assert_search_index(...)`
- Replace manual artifact checks with `assert_record_has_artifacts(...)`.
- If schema manifests are involved, use `HarnessArtifacts.write_schema_manifest(...)`.

**Acceptance Criteria**
- All serving targets executed via `ServingTargetHarness`.
- Artifact assertions use shared helper APIs.

### 4) Assertion normalization pass
**Targets**
- Tests touched in steps 1–3.

**Actions**
- Replace direct `record.status` comparisons with `assert_target_ok(...)`.
- Replace manual tuple/row count checks with `assert_record_row_counts(...)`.
- Replace manual dataset key checks with `assert_record_has_datasets(...)`.
- Replace manual artifact path checks with `assert_record_has_artifacts(...)`.
- For partial failures, use `assert_partial_failure(...)`.

**Acceptance Criteria**
- Assertions are consistent and centralized.
- No direct, ad-hoc status checks remain in migrated tests.

## Implementation Details (Per File Checklist)
**`tests/build/hamilton/test_graph_targets.py`**
- Convert any direct `HamiltonBuildHarness` usage to `graph_target_harness`.
- Replace row count/status checks with target record assertions.

**`tests/build/hamilton/test_metrics_targets.py`**
- Use `analytics_target_harness` fixture for execution.
- Replace row count/status checks with assertion helpers.

**`tests/build/hamilton/test_coverage_targets.py`**
- Use `analytics_target_harness` fixture.
- Replace manual coverage row count checks with helper assertions.

**`tests/build/serving/test_publisher.py`**
- Use `serving_target_harness` fixture.
- Use `assert_record_has_artifacts` for publish outputs.

**`tests/build/serving/test_pr90_search_index_builds.py`**
- Use `serving_target_harness` fixture.
- Use `ServingTargetHarness.assert_search_index()` for index assertions.

**`tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`**
- Use `serving_target_harness` fixture.
- Use target record assertions for result validation.

## Validation Plan
Run targeted tests in phases:
1) Graph targets:
   - `pytest -q tests/build/hamilton/test_graph_targets.py`
2) Analytics targets:
   - `pytest -q tests/build/hamilton/test_metrics_targets.py`
   - `pytest -q tests/build/hamilton/test_coverage_targets.py`
3) Serving targets:
   - `pytest -q tests/build/serving/test_publisher.py`
   - `pytest -q tests/build/serving/test_pr90_search_index_builds.py`
   - `pytest -q tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`

## Rollback Strategy
If a migration causes unexpected assertion failures:
- Revert the specific file to the previous harness usage.
- Add a follow-up task to adapt the harness wrapper API or expand assertion helpers.
