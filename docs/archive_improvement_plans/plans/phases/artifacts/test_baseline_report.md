# Test Baseline Report

> **Generated:** December 10, 2024  
> **Purpose:** Capture test state before CLI architecture migration

## Summary

| Metric | Value |
|--------|-------|
| Total tests collected | 362 |
| Passed | 352 |
| Failed | 9 |
| Errors | 1 |
| Warnings | 10 |
| Test duration | 32.01s |

## Coverage Summary

| Metric | Value |
|--------|-------|
| CLI module coverage | See `coverage_baseline.json` |
| Coverage report location | `docs/plans/phases/artifacts/htmlcov_baseline/` |
| Coverage JSON | `docs/plans/phases/artifacts/coverage_baseline.json` |

## Test Distribution

| Test Category | Count |
|---------------|-------|
| `tests/cli/config/` | Configuration tests |
| `tests/cli/contract/` | CLI result contract tests |
| `tests/cli/golden/` | Golden output tests |
| `tests/cli/handlers/` | Handler unit tests |
| `tests/cli/integration/` | Integration tests |
| `tests/cli/performance/` | Performance tests |
| `tests/cli/property/` | Property-based tests |

## Failing Tests

The following tests are failing in the baseline (9 failures, 1 error):

### Context Resolution Failures (7 tests)

These tests fail because they require a `codeintel.yaml` file but none is present:

1. `tests/cli/golden/test_golden_output.py::test_build_status_json_structure`
2. `tests/cli/golden/test_golden_output.py::test_build_status_text_output`
3. `tests/cli/integration/test_full_pipeline.py::test_json_output_is_valid_json`
4. `tests/cli/integration/test_full_pipeline.py::test_config_affects_output_format`
5. `tests/cli/integration/test_full_pipeline.py::test_telemetry_middleware_creates_spans`
6. `tests/cli/integration/test_full_pipeline.py::test_text_output_is_human_readable`
7. `tests/cli/integration/test_full_pipeline.py::test_env_override_takes_precedence`

**Error Message:** `CommandContextError: No codeintel.yaml found. Provide --repo and --commit explicitly, or create a project file.`

### Validation Logic Failure (1 test)

8. `tests/cli/test_build_cli.py::TestBuildRunValidation::test_run_conflicting_selection_flags`

**Issue:** Test expects validation error message but gets context resolution error first.

### Exit Code Mismatch (1 test)

9. `tests/cli/test_datasets_command.py::test_datasets_scaffold_existing_name`

**Issue:** Expected exit code 1, got exit code 2.

### Import Error (1 error)

10. `tests/cli/test_op_dynamic_cli.py`

**Error:** `ImportError: cannot import name 'cyclopts_ops' from 'codeintel.cli'`

## Baseline Timestamp

- **Test run started:** December 10, 2024
- **Test run completed:** December 10, 2024
- **Git SHA:** (current working tree)

## Notes

- All failures are pre-existing and not related to the migration
- These failures should be tracked but do not block the migration
- Coverage baseline JSON is available for regression tracking
