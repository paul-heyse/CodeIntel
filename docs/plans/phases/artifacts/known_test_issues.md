# Known Test Issues at Migration Start

> **Generated:** December 10, 2024  
> **Purpose:** Document pre-existing test failures to avoid confusion during migration

## Summary

| Category | Count |
|----------|-------|
| Total tests collected | 362 |
| Passing tests | 352 |
| Failing tests | 9 |
| Errors (collection) | 1 |
| Known issues (not migration-related) | 10 |
| Blocking issues | 0 |

## Failing Tests

### 1. Context Resolution Failures

**Issue:** Tests fail because they require a `codeintel.yaml` file but none is present in the test environment.

**Error Message:**
```
CommandContextError: No codeintel.yaml found. 
Provide --repo and --commit explicitly, or create a project file.
```

**Affected Tests (7):**

| Test | File | Status |
|------|------|--------|
| `test_build_status_json_structure` | `tests/cli/golden/test_golden_output.py` | FAILED |
| `test_build_status_text_output` | `tests/cli/golden/test_golden_output.py` | FAILED |
| `test_json_output_is_valid_json` | `tests/cli/integration/test_full_pipeline.py` | FAILED |
| `test_config_affects_output_format` | `tests/cli/integration/test_full_pipeline.py` | FAILED |
| `test_telemetry_middleware_creates_spans` | `tests/cli/integration/test_full_pipeline.py` | FAILED |
| `test_text_output_is_human_readable` | `tests/cli/integration/test_full_pipeline.py` | FAILED |
| `test_env_override_takes_precedence` | `tests/cli/integration/test_full_pipeline.py` | FAILED |

**Migration Impact:** None - these are environmental test setup issues.

**Potential Fix:** Tests should create a temporary `codeintel.yaml` or use explicit `--repo` and `--commit` flags.

---

### 2. Validation Logic Masked by Context Error

**Test:** `tests/cli/test_build_cli.py::TestBuildRunValidation::test_run_conflicting_selection_flags`

**Issue:** Test expects a validation error message about conflicting flags, but the context resolution error occurs first.

**Expected:** `'Provide exactly one of targets, --module, or --all.'`

**Actual:** `'Error: No codeintel.yaml found...'`

**Migration Impact:** None - this is a test ordering issue where context setup happens before parameter validation.

**Potential Fix:** Either fix context setup in test or adjust test to provide required context.

---

### 3. Exit Code Mismatch

**Test:** `tests/cli/test_datasets_command.py::test_datasets_scaffold_existing_name`

**Issue:** Expected exit code 1, but got exit code 2.

**Migration Impact:** None - this is a behavioral regression unrelated to migration.

**Potential Fix:** Investigate what changed in exit code handling.

---

### 4. Import Error

**Test:** `tests/cli/test_op_dynamic_cli.py`

**Error:**
```
ImportError: cannot import name 'cyclopts_ops' from 'codeintel.cli'
```

**Issue:** Test tries to import `cyclopts_ops` which doesn't exist in `codeintel.cli.__init__.py`.

**Migration Impact:** None - this is a missing module/export issue.

**Potential Fix:** Either add the export or update the test to use correct import path.

---

## Skipped Tests

No tests are explicitly skipped (`@pytest.mark.skip`).

---

## Action Items

### Pre-Migration (Optional)

- [ ] Consider fixing context resolution in integration tests
- [ ] Investigate `cyclopts_ops` import error

### During Migration

- [ ] Track that these failures don't increase
- [ ] Update tests if migration changes error handling order

### Post-Migration

- [ ] Review if any failures are resolved by new architecture
- [ ] Fix remaining issues unrelated to migration

---

## Notes

- All 10 issues are **pre-existing** and **do not block** the migration
- The 352 passing tests provide a solid regression baseline
- Any new failures during migration should be investigated immediately
- If migration fixes any of these issues, document in migration tracking
