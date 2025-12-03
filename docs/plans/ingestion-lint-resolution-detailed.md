# Ingestion Test Lint Resolution - Detailed Implementation Plan

**Document Version**: 1.0
**Created**: 2025-12-03
**Current State**: 142 ruff errors, 0 pyright errors
**Target State**: Zero errors through causal resolution (no suppression)

---

## Executive Summary

This plan details the systematic resolution of 142 remaining ruff linting errors in `tests/ingestion/`. All pyright errors have been resolved in prior work. The errors are distributed across 8 files and 13 error codes. The resolution strategy prioritizes structural changes over suppression.

### Error Distribution by Code

| Code | Count | Description |
|------|-------|-------------|
| PLR6301 | 94 | Method could be a function, class method, or static method |
| PLR2004 | 17 | Magic value used in comparison |
| PLC0415 | 9 | Import should be at top-level of file |
| DOC201 | 5 | Return not documented in docstring |
| RUF043 | 5 | Regex pattern contains metacharacters but is not raw |
| PLC2701 | 3 | Private name import from external module |
| DOC501 | 2 | Raised exception missing from docstring |
| SLF001 | 2 | Private member accessed |
| F841 | 1 | Local variable assigned but never used |
| F811 | 1 | Redefinition of unused name |
| FBT001 | 1 | Boolean-typed positional argument |
| FBT002 | 1 | Boolean default positional argument |
| E402 | 1 | Module level import not at top of file |

### Error Distribution by File

| File | Errors | Codes |
|------|--------|-------|
| `test_plugin_registry.py` | 32 | PLR6301:26, DOC201:2, PLR2004:2, PLC0415:1, SLF001:1 |
| `test_workers.py` | 29 | PLR6301:26, PLR2004:3 |
| `test_ingest_runs.py` | 29 | PLR6301:21, PLR2004:7, DOC501:1 |
| `test_scip_resolver.py` | 29 | PLR6301:19, RUF043:5, PLC2701:3, PLR2004:2 |
| `test_tools.py` | 9 | PLC0415:8, F841:1 |
| `test_core.py` | 7 | PLR2004:3, DOC201:2, PLR6301:2 |
| `test_resources.py` | 6 | DOC201:1, DOC501:1, E402:1, FBT001:1, FBT002:1, SLF001:1 |
| `test_module_inventory.py` | 1 | F811:1 |

---

## Test Helper Usage (graph / pipeline / coverage)

- Graph spans: use `tests._helpers.graph_env.create_span_test_env` then `build_span_graph_components` and `collect_span_snapshot` for call graph + CFG + symbol-use assertions without inline seeding.
- Pipeline graphs: use `tests._helpers.pipeline_env.create_pipeline_env` and `build_graph_and_symbols` to populate call graph/CFG/symbol uses, then `generate_pipeline_coverage` for coverage artifacts.
- Coverage edges: use `tests._helpers.coverage_env.create_coverage_edge_env` + `generate_coverage_artifact` + `compute_coverage_edges` for analytics coverage tests; `assert_single_edge` checks success.
- Prefer fixtures wired in `tests/conftest.py` when available (e.g., `span_env`, `pipeline_env`); add new fixtures for helper variants as they spread.
- When adding new tests, centralize repo/GOID seeding in helpers first; only special-case inline seeds when behavior under test requires bespoke schemas or invalid data.

## Phase 1: Convert Test Classes to Standalone Functions (PLR6301)

**Objective**: Eliminate 94 PLR6301 errors by converting test classes to module-level functions.

**Rationale**: Test methods that don't use `self` are better expressed as standalone functions. This is more idiomatic for pytest and eliminates the structural violation.

### 1.1 Convert `test_workers.py` (26 errors)

**File**: `tests/ingestion/test_workers.py`

**Classes to Convert**:
1. `TestWorkerConfig` → `test_worker_config_*` functions
2. `TestResolveWorkerCount` → `test_resolve_worker_count_*` functions
3. `TestCreateExecutor` → `test_create_executor_*` functions
4. `TestWorkerPool` → `test_worker_pool_*` functions
5. `TestExecutorFactory` → `test_executor_factory_*` functions
6. `TestWorkerConfigPresets` → `test_worker_config_presets_*` functions
7. `TestWorkerDefaults` → `test_worker_defaults_*` functions

**Transformation Pattern**:
```python
# BEFORE:
class TestWorkerConfig:
    def test_create_with_defaults(self) -> None:
        config = WorkerConfig()
        assert config.max_workers == 4

# AFTER:
@pytest.mark.worker_config
def test_worker_config_create_with_defaults() -> None:
    """Test WorkerConfig creation with default values."""
    config = WorkerConfig()
    assert config.max_workers == DEFAULT_MAX_WORKERS
```

**Steps**:
1. Read file and identify all test classes
2. For each class:
   - Extract class name (e.g., `TestWorkerConfig`)
   - Convert to marker name (e.g., `worker_config`)
   - For each method in class:
     - Remove `self` parameter
     - Prefix function name with class concept (e.g., `test_create_with_defaults` → `test_worker_config_create_with_defaults`)
     - Add `@pytest.mark.<marker>` decorator
     - Add docstring if missing
3. Remove class wrapper
4. Run `uv run ruff check tests/ingestion/test_workers.py --select PLR6301` to verify

---

### 1.2 Convert `test_plugin_registry.py` (26 errors)

**File**: `tests/ingestion/test_plugin_registry.py`

**Classes to Convert**:
1. `TestPluginRegistry` → `test_plugin_registry_*` functions
2. `TestCapabilityIndex` → `test_capability_index_*` functions
3. `TestExecutionPlanner` → `test_execution_planner_*` functions
4. `TestRegistryCleanup` → `test_registry_cleanup_*` functions
5. `MockPlugin` (internal class - keep but convert methods)

**Special Handling**:
- `MockPlugin` is used as a fixture within tests; keep as a helper class but mark methods with `@staticmethod` where appropriate
- The `MockPlugin.execute()` and `MockPlugin.validate_inputs()` methods have DOC201 errors that need docstring fixes

**Steps**:
1. Convert all `Test*` classes to standalone functions
2. Keep `MockPlugin` as a helper class but add `@staticmethod` decorators
3. Fix DOC201 errors in `MockPlugin` methods by adding Returns sections

---

### 1.3 Convert `test_ingest_runs.py` (21 errors)

**File**: `tests/ingestion/test_ingest_runs.py`

**Classes to Convert**:
1. `TestIngestRunStatus` → `test_ingest_run_status_*` functions
2. `TestIngestRunMode` → `test_ingest_run_mode_*` functions
3. `TestIngestRun` → `test_ingest_run_*` functions
4. `TestClassifyError` → `test_classify_error_*` functions
5. `TestFileSink` → `test_file_sink_*` functions
6. `TestDatabaseSink` → `test_database_sink_*` functions
7. `TestSinkFanout` → `test_sink_fanout_*` functions

**Special Handling**:
- Line 392-393: `FailingSink.record()` method has both PLR6301 and DOC501 errors
- Convert `FailingSink` to a helper class with `@staticmethod` decorator

---

### 1.4 Convert `test_scip_resolver.py` (19 errors)

**File**: `tests/ingestion/test_scip_resolver.py`

**Classes to Convert**:
1. `TestScipResolverInput` → `test_scip_resolver_input_*` functions
2. `TestScipToolOutput` → `test_scip_tool_output_*` functions
3. `TestResolveScipInputs` → `test_resolve_scip_inputs_*` functions
4. `TestModuleRecord` → `test_module_record_*` functions

---

### 1.5 Convert Remaining `test_core.py` (2 errors)

**File**: `tests/ingestion/test_core.py`

**Methods to Fix**:
- Lines 47-48: `compute` method in `DummyPlugin` class
- Lines 63-64: `compute` method in `TrackerPlugin` class

**Solution**: These are helper plugin classes, not test classes. Add `@staticmethod` decorator since they don't use `self`:

```python
# BEFORE:
class DummyPlugin(BaseIngestPlugin):
    def compute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
        ...

# AFTER:
class DummyPlugin(BaseIngestPlugin):
    @staticmethod
    def compute(ctx: IngestExecutionContext) -> IngestPluginResult:
        ...
```

---

## Phase 2: Define Constants for Magic Values (PLR2004)

**Objective**: Eliminate 17 PLR2004 errors by replacing magic values with named constants.

### 2.1 `test_ingest_runs.py` (7 errors)

**Magic Values to Replace**:

| Line | Value | Constant Name |
|------|-------|---------------|
| 96 | `50` | `TEST_TOTAL_MODULES = 50` |
| 138 | `100` | `TEST_TOTAL_FILES = 100` |
| 139 | `10` | `TEST_PROCESSED_FILES = 10` |
| 140 | `2` | `TEST_FAILED_FILES = 2` |
| 141 | `0.1` | `TEST_ELAPSED_SECONDS = 0.1` |
| 142 | `0.02` | `TEST_AVERAGE_TIME = 0.02` |
| 296 | `2` | `EXPECTED_LINE_COUNT = 2` |

**Implementation**:
```python
# Add at module level, after imports:
# Test constants
TEST_TOTAL_MODULES = 50
TEST_TOTAL_FILES = 100
TEST_PROCESSED_FILES = 10
TEST_FAILED_FILES = 2
TEST_ELAPSED_SECONDS = 0.1
TEST_AVERAGE_TIME = 0.02
EXPECTED_LINE_COUNT = 2
```

---

### 2.2 `test_workers.py` (3 errors)

**Magic Values to Replace**:

| Line | Value | Constant Name |
|------|-------|---------------|
| 211 | `42` | `TEST_WORKER_COUNT = 42` |
| 281 | `16` | `DEFAULT_MAX_WORKERS = 16` |
| 286 | `2` | `DEFAULT_MIN_WORKERS = 2` |

---

### 2.3 `test_core.py` (3 errors)

**Magic Values to Replace**:

| Line | Value | Constant Name |
|------|-------|---------------|
| 189 | `42` | `TEST_VALUE = 42` |
| 385 | `1024` | `TEST_MEMORY_MB = 1024` |
| 386 | `60000` | `TEST_TIMEOUT_MS = 60000` |

---

### 2.4 `test_plugin_registry.py` (2 errors)

**Magic Values to Replace**:

| Line | Value | Constant Name |
|------|-------|---------------|
| 196 | `2` | `EXPECTED_PLUGIN_COUNT = 2` |
| 267 | `2` | `EXPECTED_TABLE_COUNT = 2` |

---

### 2.5 `test_scip_resolver.py` (2 errors)

**Magic Values to Replace**:

| Line | Value | Constant Name |
|------|-------|---------------|
| 412 | `5` | `EXPECTED_START_LINE = 5` |
| 413 | `10` | `EXPECTED_END_LINE = 10` |

---

## Phase 3: Fix Import Organization (PLC0415, E402)

**Objective**: Eliminate 10 import-related errors by moving imports to the top of files.

### 3.1 `test_tools.py` (8 PLC0415 errors)

**Lines with Issues**: 1203, 1225, 1245, 1265, 1292, 1327, 1345, 1365

**Problem**: Imports are inside test functions to avoid import-time errors when optional dependencies are missing.

**Solution**: Move all imports to the top of the file under appropriate guards:

```python
# At top of file, after existing imports:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Heavy imports used only for type hints
    pass

# Runtime imports that may fail if optional deps missing
try:
    from codeintel.ingestion.tools.pyright import PyrightRunner
    from codeintel.ingestion.tools.ruff import RuffRunner
    # ... other optional imports
    _OPTIONAL_DEPS_AVAILABLE = True
except ImportError:
    _OPTIONAL_DEPS_AVAILABLE = False

# Then in tests that need these:
@pytest.mark.skipif(not _OPTIONAL_DEPS_AVAILABLE, reason="Optional deps not installed")
def test_pyright_runner():
    ...
```

**Alternative Solution** (if deps are always available in tests):
Simply move all imports to the top unconditionally since the test environment has all dependencies installed.

---

### 3.2 `test_plugin_registry.py` (1 PLC0415 error)

**Line**: 67

**Problem**: Import inside `validate_inputs()` method of `MockPlugin`.

**Solution**: Move import to top of file:
```python
# Move to top of file:
from codeintel.ingestion.core.base import ValidationResult
```

---

### 3.3 `test_resources.py` (1 E402 error)

**Line**: 755

**Problem**: Module-level import appears after code.

**Solution**: Reorganize file so all imports appear before any executable code.

---

## Phase 4: Fix Regex Pattern Issues (RUF043)

**Objective**: Eliminate 5 RUF043 errors in `test_scip_resolver.py`.

**Lines**: 300, 314, 328, 342, 356

**Problem**: Regex patterns passed to `pytest.raises(match=...)` contain metacharacters but are not escaped or raw strings.

**Solution**: Convert patterns to raw strings:

```python
# BEFORE:
with pytest.raises(ValueError, match="repo.*required"):
    ...

# AFTER:
with pytest.raises(ValueError, match=r"repo.*required"):
    ...
```

**Specific Fixes**:
| Line | Current | Fixed |
|------|---------|-------|
| 300 | `match="..."` | `match=r"..."` |
| 314 | `match="..."` | `match=r"..."` |
| 328 | `match="..."` | `match=r"..."` |
| 342 | `match="..."` | `match=r"..."` |
| 356 | `match="..."` | `match=r"..."` |

---

## Phase 5: Fix Private Import Issues (PLC2701)

**Objective**: Eliminate 3 PLC2701 errors in `test_scip_resolver.py`.

**Lines**: 14, 15, 16

**Problem**: Importing private module `_scip_resolver` from external package.

**Current Code**:
```python
from codeintel.ingestion.infrastructure_utilities._scip_resolver import (
    ScipResolverInput,
    ScipToolOutput,
    resolve_scip_inputs,
)
```

**Solution Options**:

**Option A (Preferred)**: Re-export from public module
1. Add exports to `codeintel/ingestion/infrastructure_utilities/__init__.py`:
   ```python
   from codeintel.ingestion.infrastructure_utilities._scip_resolver import (
       ScipResolverInput,
       ScipToolOutput,
       resolve_scip_inputs,
   )
   __all__ = ["ScipResolverInput", "ScipToolOutput", "resolve_scip_inputs"]
   ```
2. Update test imports:
   ```python
   from codeintel.ingestion.infrastructure_utilities import (
       ScipResolverInput,
       ScipToolOutput,
       resolve_scip_inputs,
   )
   ```

**Option B**: Use `# noqa: PLC2701` if the private module is intentionally being tested
- Only use this if Option A is not feasible due to architectural constraints

---

## Phase 6: Fix Docstring Issues (DOC201, DOC501)

**Objective**: Eliminate 7 docstring errors.

### 6.1 DOC201 - Missing Return Documentation (5 errors)

**Files and Lines**:
- `test_core.py`: 48, 64 (DummyPlugin.compute, TrackerPlugin.compute)
- `test_plugin_registry.py`: 62, 66 (MockPlugin.execute, MockPlugin.validate_inputs)
- `test_resources.py`: 770 (helper function)

**Fix Pattern**:
```python
# BEFORE:
def compute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
    """Execute plugin computation."""
    return IngestPluginResult.success({})

# AFTER:
def compute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
    """Execute plugin computation.

    Returns
    -------
    IngestPluginResult
        Success result with empty data.
    """
    return IngestPluginResult.success({})
```

---

### 6.2 DOC501 - Missing Raises Documentation (2 errors)

**Files and Lines**:
- `test_ingest_runs.py`: 393 (FailingSink.record raises RuntimeError)
- `test_resources.py`: 770 (function raises ValueError)

**Fix Pattern**:
```python
# BEFORE:
def record(self, run: IngestRun) -> None:
    """Record an ingest run."""
    raise RuntimeError("Test failure")

# AFTER:
def record(self, run: IngestRun) -> None:
    """Record an ingest run.

    Raises
    ------
    RuntimeError
        Always raised to simulate sink failure.
    """
    raise RuntimeError("Test failure")
```

---

## Phase 7: Fix Remaining Minor Issues

### 7.1 SLF001 - Private Member Access (2 errors)

**Locations**:
- `test_plugin_registry.py`: Line 126 - `_entrypoints_loaded`
- `test_resources.py`: Line 704 - `_config`

**Solutions**:
- If testing internal state is necessary, add `# noqa: SLF001` with justification comment
- If possible, test via public interface instead

---

### 7.2 F811 - Redefinition (1 error)

**File**: `test_module_inventory.py`, Line 18

**Problem**: `FilesystemDiscoveryAdapter` imported twice.

**Solution**: Remove duplicate import.

---

### 7.3 FBT001/FBT002 - Boolean Arguments (2 errors)

**File**: `test_resources.py`, Line 763

**Problem**: Boolean-typed positional argument with default value.

**Solution**: Convert to keyword-only argument:
```python
# BEFORE:
def helper_func(value: str, flag: bool = False) -> str:
    ...

# AFTER:
def helper_func(value: str, *, flag: bool = False) -> str:
    ...
```

---

### 7.4 F841 - Unused Variable (1 error)

**File**: `test_tools.py`, Line 1390

**Problem**: Variable `run` assigned but never used.

**Solution**: Either:
1. Remove the assignment if not needed
2. Prefix with underscore: `_run = ...`
3. Use the variable in an assertion

---

## Implementation Order

Execute phases in this order to minimize conflicts:

| Order | Phase | Files Affected | Errors Fixed | Cumulative Remaining |
|-------|-------|----------------|--------------|---------------------|
| 1 | Phase 1.1 | test_workers.py | 26 | 116 |
| 2 | Phase 1.2 | test_plugin_registry.py | 26 | 90 |
| 3 | Phase 1.3 | test_ingest_runs.py | 21 | 69 |
| 4 | Phase 1.4 | test_scip_resolver.py | 19 | 50 |
| 5 | Phase 1.5 | test_core.py | 2 | 48 |
| 6 | Phase 2 | Multiple files | 17 | 31 |
| 7 | Phase 3 | test_tools.py, test_plugin_registry.py, test_resources.py | 10 | 21 |
| 8 | Phase 4 | test_scip_resolver.py | 5 | 16 |
| 9 | Phase 5 | test_scip_resolver.py | 3 | 13 |
| 10 | Phase 6 | Multiple files | 7 | 6 |
| 11 | Phase 7 | Multiple files | 6 | 0 |

---

## Verification Commands

After each phase, run verification:

```bash
# Check specific file
uv run ruff check tests/ingestion/test_<name>.py

# Check all ingestion tests
uv run ruff check tests/ingestion/

# Run tests to ensure no regressions
uv run pytest tests/ingestion/ -q

# Full quality check
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

---

## Success Criteria

1. **Zero ruff errors**: `uv run ruff check tests/ingestion/` returns exit code 0
2. **All tests pass**: `uv run pytest tests/ingestion/ -q` shows all tests passing
3. **No suppressions**: No new `# noqa` comments added (except justified SLF001)
4. **Maintained coverage**: Coverage percentage unchanged or improved

---

## Appendix A: File-by-File Error Reference

### `test_workers.py` (29 errors total)

```
 30:9  PLR6301 Method `test_create_with_defaults` could be a function...
 39:9  PLR6301 Method `test_create_with_custom_values` could be a function...
 55:9  PLR6301 Method `test_frozen_dataclass` could be a function...
 66:9  PLR6301 Method `test_explicit_count_takes_precedence` could be a function...
 76:9  PLR6301 Method `test_explicit_zero_is_ignored` could be a function...
 88:9  PLR6301 Method `test_negative_explicit_is_ignored` could be a function...
100:9  PLR6301 Method `test_env_var_takes_precedence_over_default` could be a function...
108:9  PLR6301 Method `test_invalid_env_var_is_ignored` could be a function...
117:9  PLR6301 Method `test_zero_env_var_is_ignored` could be a function...
126:9  PLR6301 Method `test_default_calculation` could be a function...
134:9  PLR6301 Method `test_custom_max_workers` could be a function...
146:9  PLR6301 Method `test_custom_min_workers` could be a function...
162:9  PLR6301 Method `test_create_process_executor` could be a function...
172:9  PLR6301 Method `test_create_thread_executor` could be a function...
182:9  PLR6301 Method `test_unknown_kind_defaults_to_thread` could be a function...
196:9  PLR6301 Method `test_worker_pool_process` could be a function...
201:9  PLR6301 Method `test_worker_pool_thread` could be a function...
206:9  PLR6301 Method `test_worker_pool_shutdown_on_exit` could be a function...
211:39 PLR2004 Magic value used in comparison, consider replacing `42`...
220:9  PLR6301 Method `test_factory_returns_callable` could be a function...
226:9  PLR6301 Method `test_factory_creates_thread_executor` could be a function...
236:9  PLR6301 Method `test_factory_creates_process_executor` could be a function...
246:9  PLR6301 Method `test_factory_creates_new_instance_each_call` could be a function...
262:9  PLR6301 Method `test_ast_worker_config` could be a function...
268:9  PLR6301 Method `test_cst_worker_config` could be a function...
278:9  PLR6301 Method `test_default_max_workers` could be a function...
281:39 PLR2004 Magic value used in comparison, consider replacing `16`...
283:9  PLR6301 Method `test_default_min_workers` could be a function...
286:39 PLR2004 Magic value used in comparison, consider replacing `2`...
```

### `test_plugin_registry.py` (32 errors total)

```
 61:9  PLR6301 Method `execute` could be a function...
 62:9  DOC201 `return` is not documented in docstring
 65:9  PLR6301 Method `validate_inputs` could be a function...
 66:9  DOC201 `return` is not documented in docstring
 67:9  PLC0415 `import` should be at the top-level of a file
 75:9  PLR6301 Method `test_register_plugin` could be a function...
 84:9  PLR6301 Method `test_register_duplicate_raises` could be a function...
 95:9  PLR6301 Method `test_unregister_plugin` could be a function...
105:9  PLR6301 Method `test_unregister_nonexistent_is_noop` could be a function...
112:9  PLR6301 Method `test_get_plugin` could be a function...
122:9  PLR6301 Method `test_get_nonexistent_raises` could be a function...
126:9  SLF001 Private member accessed: `_entrypoints_loaded`
131:9  PLR6301 Method `test_contains` could be a function...
140:9  PLR6301 Method `test_list_all` could be a function...
156:9  PLR6301 Method `test_list_names` could be a function...
174:9  PLR6301 Method `test_list_providing` could be a function...
185:9  PLR6301 Method `test_list_providing_multiple` could be a function...
196:31 PLR2004 Magic value used in comparison, consider replacing `2`...
201:9  PLR6301 Method `test_list_providing_empty` could be a function...
213:9  PLR6301 Method `test_list_by_stage` could be a function...
224:9  PLR6301 Method `test_list_by_stage_multiple` could be a function...
243:9  PLR6301 Method `test_list_by_table` could be a function...
254:9  PLR6301 Method `test_list_by_table_multiple` could be a function...
267:31 PLR2004 Magic value used in comparison, consider replacing `2`...
273:9  PLR6301 Method `test_defaults` could be a function...
284:9  PLR6301 Method `test_with_values` could be a function...
306:9  PLR6301 Method `test_plan_with_custom_plugins` could be a function...
319:9  PLR6301 Method `test_plan_respects_dependencies` could be a function...
337:9  PLR6301 Method `test_plan_with_disabled` could be a function...
359:9  PLR6301 Method `test_unregister_removes_from_capability_index` could be a function...
372:9  PLR6301 Method `test_unregister_removes_from_stage_index` could be a function...
385:9  PLR6301 Method `test_unregister_removes_from_table_index` could be a function...
```

### `test_ingest_runs.py` (29 errors total)

```
 33:9  PLR6301 Method `test_status_values` could be a function...
 43:9  PLR6301 Method `test_mode_values` could be a function...
 53:9  PLR6301 Method `test_create_minimal` could be a function...
 77:9  PLR6301 Method `test_create_with_metrics` could be a function...
 96:37 PLR2004 Magic value used in comparison, consider replacing `50`...
 99:9  PLR6301 Method `test_create_with_error` could be a function...
119:9  PLR6301 Method `test_create_with_incremental_metrics` could be a function...
138:37 PLR2004 Magic value used in comparison, consider replacing `100`...
139:39 PLR2004 Magic value used in comparison, consider replacing `10`...
140:39 PLR2004 Magic value used in comparison, consider replacing `2`...
141:45 PLR2004 Magic value used in comparison, consider replacing `0.1`...
142:45 PLR2004 Magic value used in comparison, consider replacing `0.02`...
145:9  PLR6301 Method `test_to_row_tuple` could be a function...
167:9  PLR6301 Method `test_tool_not_found_error` could be a function...
179:9  PLR6301 Method `test_tool_execution_error` could be a function...
199:9  PLR6301 Method `test_tool_execution_timeout` could be a function...
219:9  PLR6301 Method `test_duckdb_error` could be a function...
227:9  PLR6301 Method `test_value_error` could be a function...
235:9  PLR6301 Method `test_other_error` could be a function...
247:9  PLR6301 Method `test_record_creates_file` could be a function...
267:9  PLR6301 Method `test_record_appends_jsonl` could be a function...
296:30 PLR2004 Magic value used in comparison, consider replacing `2`...
304:9  PLR6301 Method `test_record_includes_timestamps` could be a function...
333:9  PLR6301 Method `test_record_inserts_row` could be a function...
361:9  PLR6301 Method `test_record_fans_out` could be a function...
386:9  PLR6301 Method `test_record_continues_on_sink_failure` could be a function...
392:17 PLR6301 Method `record` could be a function...
393:17 DOC501 Raised exception `RuntimeError` missing from docstring
421:9  PLR6301 Method `test_empty_sinks` could be a function...
```

### `test_scip_resolver.py` (29 errors total)

```
 14:5  PLC2701 Private name import `_scip_resolver` from external module...
 15:5  PLC2701 Private name import `_scip_resolver` from external module...
 16:5  PLC2701 Private name import `_scip_resolver` from external module...
 24:9  PLR6301 Method `test_create_minimal` could be a function...
 55:9  PLR6301 Method `test_create_with_modules` could be a function...
 86:9  PLR6301 Method `test_frozen_dataclass` could be a function...
108:9  PLR6301 Method `test_create_empty` could be a function...
122:9  PLR6301 Method `test_create_with_explicit_params` could be a function...
146:9  PLR6301 Method `test_create_with_modules` could be a function...
169:9  PLR6301 Method `test_frozen_dataclass` could be a function...
180:9  PLR6301 Method `test_resolve_with_explicit_params` could be a function...
208:9  PLR6301 Method `test_resolve_with_scip_resolver_input` could be a function...
239:9  PLR6301 Method `test_resolve_with_modules_sequence` could be a function...
267:9  PLR6301 Method `test_resolve_with_modules_kwarg` could be a function...
296:9  PLR6301 Method `test_resolve_missing_repo_raises_value_error` could be a function...
300:46 RUF043 Pattern passed to `match=` contains metacharacters...
310:9  PLR6301 Method `test_resolve_missing_commit_raises_value_error` could be a function...
314:46 RUF043 Pattern passed to `match=` contains metacharacters...
324:9  PLR6301 Method `test_resolve_missing_repo_root_raises_value_error` could be a function...
328:46 RUF043 Pattern passed to `match=` contains metacharacters...
338:9  PLR6301 Method `test_resolve_missing_build_dir_raises_value_error` could be a function...
342:46 RUF043 Pattern passed to `match=` contains metacharacters...
352:9  PLR6301 Method `test_resolve_missing_document_output_dir_raises_value_error` could be a function...
356:46 RUF043 Pattern passed to `match=` contains metacharacters...
366:9  PLR6301 Method `test_resolve_inputs_override_kwargs` could be a function...
397:9  PLR6301 Method `test_create_module_record` could be a function...
412:32 PLR2004 Magic value used in comparison, consider replacing `5`...
413:32 PLR2004 Magic value used in comparison, consider replacing `10`...
415:9  PLR6301 Method `test_module_record_frozen` could be a function...
```

---

## Appendix B: Helper Classes Reference

These helper classes need special handling (not converted to functions, but methods may need `@staticmethod`):

| File | Class | Purpose | Handling |
|------|-------|---------|----------|
| `test_core.py` | `DummyPlugin` | Mock plugin for testing | Add `@staticmethod` to `compute()` |
| `test_core.py` | `TrackerPlugin` | Mock plugin for testing | Add `@staticmethod` to `compute()` |
| `test_plugin_registry.py` | `MockPlugin` | Mock plugin for registry tests | Add `@staticmethod` to methods, fix docstrings |
| `test_ingest_runs.py` | `FailingSink` | Mock sink that raises | Add `@staticmethod` to `record()`, fix docstring |

---

## Appendix C: Pytest Markers to Add

Add these markers to `pytest.ini` for the converted test functions:

```ini
[pytest]
markers =
    worker_config: Tests for WorkerConfig
    resolve_worker_count: Tests for resolve_worker_count()
    create_executor: Tests for create_executor()
    worker_pool: Tests for worker_pool context manager
    executor_factory: Tests for executor factory
    worker_config_presets: Tests for worker config presets
    worker_defaults: Tests for worker defaults
    plugin_registry: Tests for PluginRegistry
    capability_index: Tests for capability indexing
    execution_planner: Tests for execution planning
    registry_cleanup: Tests for registry cleanup
    ingest_run_status: Tests for IngestRunStatus enum
    ingest_run_mode: Tests for IngestRunMode enum
    ingest_run: Tests for IngestRun dataclass
    classify_error: Tests for error classification
    file_sink: Tests for file sink
    database_sink: Tests for database sink
    sink_fanout: Tests for sink fanout
    scip_resolver_input: Tests for ScipResolverInput
    scip_tool_output: Tests for ScipToolOutput
    resolve_scip_inputs: Tests for resolve_scip_inputs()
    module_record: Tests for ModuleRecord
```
