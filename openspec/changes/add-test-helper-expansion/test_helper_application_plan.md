# Test Helper Application Plan

## Goal
Apply the new test helper standards from `openspec/changes/add-test-helper-expansion` across all identified test locations in a single, coordinated change set.

## Scope Summary
Targeted helper usage:
- `assert_target_ok` for TargetRunRecord status validation.
- `assert_record_row_counts` for exact row count checks on TargetRunRecord.
- `assert_row_count` for minimum/maximum row count checks.
- `assert_record_has_artifacts` and `assert_record_has_datasets` for artifact/dataset membership checks.
- Harness-driven patterns where appropriate for build/graph/serving targets.

## Execution Phases

### Phase 0: Baseline inventory and mapping
1. Confirm all target files listed below still contain manual TargetRunRecord checks (status/row_counts/artifacts/datasets).
2. For each file, define the replacement helper call(s) and note any checks that must remain manual (e.g., explicit length/tuple defaults).

Acceptance:
- Inventory list matches all expected files.
- Each file has a clear helper replacement mapping.

### Phase 1: TargetRunRecord status normalization
Replace manual status checks with `assert_target_ok` where status is the only assertion, while keeping any explicit error/field checks.

Targets:
- `tests/build/hamilton/test_executor_pipeline_template.py`
- `tests/build/hamilton/test_multi_table_pipeline_template.py`
- `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
- `tests/build/hamilton/test_pr12_loader_nodes.py`
- `tests/build/hamilton/native/test_skip_logic.py`
- `tests/build/test_hamilton_phase1.py`
- `tests/_helpers/manifests.py`
- `tests/_helpers/orchestration/graph_orchestration.py`
- `tests/_helpers/orchestration/provisioning.py`

Procedure:
1. Replace `expect_equal(record.status, ...)` and `if record.status != ...` with `assert_target_ok(record, expected_status=...)`.
2. Keep any additional validation (error message substrings, explicit `record.success`/`record.skipped`) intact.
3. For orchestration functions that currently raise `RuntimeError`, wrap `assert_target_ok` in try/except and re-raise `RuntimeError` with the existing message.

Acceptance:
- No manual TargetRunRecord status checks remain in these files.
- Error message assertions remain unchanged.

### Phase 2: Row count standardization
Replace row-count checks with helper assertions based on intent.

Targets:
- `tests/build/hamilton/test_executor_pipeline_template.py`
- `tests/build/hamilton/test_multi_table_pipeline_template.py`
- `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
- `tests/build/hamilton/native/test_skip_logic.py`
- `tests/ingestion/test_scip_ingest.py`

Procedure:
1. For exact row counts, use `assert_record_row_counts(record, {...})`.
2. For minimum counts or “must be non-empty”, use `assert_row_count(record.row_counts, table, min_rows=1)`.
3. Remove patterns that compare `record.row_counts.get(...)` to itself.

Acceptance:
- All row count checks use helper functions.
- No `row_counts.get(...)` “self-compare” patterns remain.

### Phase 3: Artifact and dataset membership helpers
Use helper functions for artifact/dataset membership tests.

Targets:
- `tests/build/hamilton/test_pr11_datasetref_v2.py`

Procedure:
1. Replace direct artifact name access checks with `assert_record_has_artifacts`.
2. Replace dataset existence checks with `assert_record_has_datasets`.
3. Keep length/tuple default checks as-is if they are explicitly testing defaults.

Acceptance:
- Artifact/dataset membership is asserted via helpers.
- Default tuple semantics are still explicitly verified where required.

### Phase 4: Docstring/example alignment
Ensure documented examples use the new helper conventions.

Targets:
- `tests/analytics/conftest.py`
- `tests/_helpers/hamilton_execution.py`

Procedure:
1. Update examples to import and use `assert_target_ok` instead of manual `record.status` comparisons.

Acceptance:
- Documentation matches helper standards.

## Validation
- Run a focused pytest subset covering the edited files after changes are complete.
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json` if requested or if type/lint issues are suspected.

## Change Management
- Apply all updates in a single batch to avoid drift between helper usage.
- Avoid changes outside the target files unless a helper import is required.
- Keep all existing test intent intact; only standardize the assertion style.

## Rollback Strategy
- If a helper replacement breaks a test’s intent, revert only that assertion and document why it cannot be standardized.

