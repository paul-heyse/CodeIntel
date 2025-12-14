# PR-51: Eliminate DB Writes in Analytics — Detailed Implementation Plan

> **Status**: ✅ COMPLETE (11 of 11 modules migrated) — Decommissioning Phase 1-2 Complete  
> **Created**: December 14, 2025  
> **Last Updated**: December 14, 2025  
> **Scope**: Migrate 11 analytics modules with 22 direct writes to Hamilton materializers

---

## Executive Summary

This document provides a comprehensive, file-by-file implementation plan for migrating all direct `gateway.ibis.write()` calls from analytics modules to Hamilton native materializers. Upon completion, all persistence will flow through the Hamilton build system, enabling:

- Asset catalog tracking
- Lineage correctness
- Schema validation at write boundaries
- Centralized materialization policies

---

## Migration Progress

| # | Module | Tables | Status |
|---|--------|--------|--------|
| 1 | `analytics/cfg_dfg/materialize.py` | 6 | ✅ Complete |
| 2 | `analytics/data_models/core.py` | 3 | ✅ Complete |
| 3 | `analytics/dependencies/core.py` | 2 | ✅ Complete |
| 4 | `analytics/entrypoints/core.py` | 2 | ✅ Complete |
| 5 | `analytics/testing/graph_metrics.py` | 2 | ✅ Complete |
| 6 | `analytics/parsing/validation.py` | 2 | ✅ Complete |
| 7 | `analytics/compute/coverage/functions.py` | 1 | ✅ Complete |
| 8 | `analytics/compute/data_models/usage.py` | 1 | ✅ Complete |
| 9 | `analytics/profiles/writer_guard.py` | 1 | ✅ Complete |
| 10 | `analytics/functions/function_history.py` | 1 | ✅ Complete |
| 11 | `analytics/history/history_timeseries.py` | 1 | ✅ Complete |

**Tables migrated**: 22 of 22 ✅

---

## Established Migration Pattern

After completing all 11 module migrations, we have established a robust, consistent pattern. This pattern has proven reliable across targets with varying complexity, including three distinct migration approaches: standard target replacement, utility class enhancement, and pure function extraction.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Hamilton Native Module                           │
│           build/hamilton/native/analytics/<target>.py                │
├──────────────────────────────────────────────────────────────────────┤
│  @tag(domain="analytics", target="<name>", node_type="compute")      │
│  def t__<target>__compute(env: BuildEnv) -> <Result>:                │
│      return compute_<target>_pure(env.gateway, env.snapshot)         │
│                                                                      │
│  @tag(domain="analytics", target="<name>", node_type="materialize")  │
│  def t__<target>(env, graph, t__<target>__compute) -> TargetRunRecord│
│      executor = NativeTargetExecutor.for_target(env, graph, "<name>")│
│      def compute() -> dict[str, int]:                                │
│          ctx = MaterializationContext(...)                           │
│          ref = materialize_rows(ctx, "<table>", rows, COLS)          │
│          return {"<table>": ref.row_count}                           │
│      return executor.execute(compute)                                │
└─────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Pure Compute Layer                                │
│              analytics/<domain>/compute.py                           │
├──────────────────────────────────────────────────────────────────────┤
│  @dataclass(frozen=True)                                             │
│  class <Target>Result:                                               │
│      rows_table1: tuple[tuple[object, ...], ...]                     │
│      rows_table2: tuple[tuple[object, ...], ...]                     │
│                                                                      │
│  def compute_<target>_pure(gateway, snapshot) -> <Target>Result:     │
│      # Read inputs, compute, return rows - NO WRITES                 │
│      return <Target>Result(...)                                      │
└─────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 Materialization Helpers                              │
│           build/hamilton/native/materializer.py                      │
├──────────────────────────────────────────────────────────────────────┤
│  materialize_rows(ctx, table_key, rows, columns) -> DatasetRef       │
│    - Deletes existing rows for snapshot                              │
│    - Writes rows via DuckDBPolicyBackend                             │
│    - Records asset catalog entry                                     │
│    - Returns DatasetRef with row_count                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Hamilton-Specific Details

#### 1. Node Naming Convention

All Hamilton nodes follow the naming pattern:
- `t__<target_name>__<node_type>` for intermediate nodes
- `t__<target_name>` for the final materialization node

Examples:
- `t__cfg_dfg_metrics__compute_cfg` (compute node for CFG)
- `t__cfg_dfg_metrics__compute_dfg` (compute node for DFG)
- `t__cfg_dfg_metrics` (final materialization node)

#### 2. Tag Decorators

Every Hamilton node uses the `@tag` decorator for metadata:

```python
from hamilton.function_modifiers import tag

@tag(domain="analytics", target="data_models", node_type="compute")
def t__data_models__compute(env: BuildEnv) -> DataModelsResult:
    ...
```

Tag fields:
- `domain`: Always `"analytics"` for analytics targets
- `target`: The target name as registered in `registrations.py`
- `node_type`: Either `"compute"` or `"materialize"`

#### 3. NativeTargetExecutor Pattern

The `NativeTargetExecutor` provides standardized boilerplate:

```python
from codeintel.build.hamilton.native.executor import NativeTargetExecutor

def t__<target>(env: BuildEnv, graph: TargetGraph, computed_result: Result) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "<target_name>")
    
    # Check if target should be skipped
    if executor.should_skip():
        return executor.skip()
    
    def compute() -> dict[str, int]:
        # Materialization logic here
        return {"table1": count1, "table2": count2}
    
    return executor.execute(compute)
```

Features:
- `should_skip()`: Checks if target is skipped via configuration
- `skip()`: Returns a TargetRunRecord with skipped status
- `execute(fn)`: Wraps the compute function with timing, error handling, and record creation
- `input_hash`: Provides hash for cache invalidation

#### 4. MaterializationContext

The `MaterializationContext` wraps gateway + snapshot for materialization:

```python
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_rows

ctx = MaterializationContext(
    gateway=env.gateway,
    snapshot=env.snapshot,
    validate=env.validate_outputs,
    owner_target="<target_name>",
    input_hash=executor.input_hash,
)
```

#### 5. materialize_rows Helper

The `materialize_rows` function handles all write operations:

```python
ref = materialize_rows(
    ctx,                          # MaterializationContext
    "analytics.table_name",       # Table key
    result.rows,                  # tuple[tuple[object, ...], ...]
    TABLE_COLS,                   # Column names list
)
```

It performs:
1. Deletes existing rows for the snapshot (repo + commit)
2. Ensures table exists via `DuckDBPolicyBackend.ensure_table()`
3. Writes rows via `DuckDBPolicyBackend.bulk_insert()`
4. Records asset in catalog for lineage tracking
5. Returns `DatasetRef` with `row_count` and metadata

### Result Dataclass Pattern

All pure compute functions return frozen dataclasses with tuple rows:

```python
@dataclass(frozen=True)
class DataModelsResult:
    """Result container for data models computation.
    
    Contains row data for all tables without performing writes.
    The rows are tuples matching the column specifications in the schema.
    """
    model_rows: tuple[tuple[object, ...], ...]
    field_rows: tuple[tuple[object, ...], ...]
    relationship_rows: tuple[tuple[object, ...], ...]
```

Key characteristics:
- `frozen=True` for immutability
- Rows as `tuple[tuple[object, ...], ...]` (tuple of row tuples)
- One attribute per output table
- Docstring documents the schema alignment

### Registration Update Pattern

In `src/codeintel/build/registrations.py`:

```python
# Before: Plugin-based
registry.register(DATA_MODELS_TARGET, plugin=DataModelsPlugin)

# After: Native module
registry.register(
    DATA_MODELS_TARGET,
    native_module="codeintel.build.hamilton.native.analytics.data_models",
)
```

### Deprecation Pattern

Original functions are deprecated but kept callable:

```python
"""Module docstring.

.. deprecated::
    This module contains legacy functions with direct database writes.
    Use the Hamilton native module for new code.
"""

import warnings

def compute_data_models(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    """
    Original docstring...

    .. deprecated::
        Use the Hamilton native module instead.
    """
    warnings.warn(
        "compute_data_models is deprecated. Use the Hamilton native module "
        "'codeintel.build.hamilton.native.analytics.data_models' or "
        "'compute_data_models_pure' for pure compute.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Original implementation continues...
```

---

## Completed Migrations

### 1. `analytics/cfg_dfg/materialize.py` (6 writes) ✅

**Files Created:**
- `src/codeintel/analytics/cfg_dfg/compute.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg.py`
- `tests/build/hamilton/test_pr51_cfg_dfg_native.py`

**Files Modified:**
- `src/codeintel/analytics/cfg_dfg/__init__.py` (exports)
- `src/codeintel/analytics/cfg_dfg/materialize.py` (deprecation)
- `src/codeintel/build/registrations.py` (native_module)

**Tables Migrated:**
- `analytics.cfg_function_metrics`
- `analytics.cfg_block_metrics`
- `analytics.cfg_function_metrics_ext`
- `analytics.dfg_function_metrics`
- `analytics.dfg_block_metrics`
- `analytics.dfg_function_metrics_ext`

---

### 2. `analytics/data_models/core.py` (3 writes) ✅

**Files Created:**
- `src/codeintel/analytics/data_models/compute.py`
- `src/codeintel/build/hamilton/native/analytics/data_models.py`
- `tests/build/hamilton/test_pr51_data_models_native.py`

**Files Modified:**
- `src/codeintel/analytics/data_models/__init__.py` (exports)
- `src/codeintel/analytics/data_models/core.py` (deprecation)
- `src/codeintel/build/registrations.py` (native_module)

**Tables Migrated:**
- `analytics.data_models`
- `analytics.data_model_fields`
- `analytics.data_model_relationships`

---

### 3. `analytics/dependencies/core.py` (2 writes) ✅

**Files Created:**
- `src/codeintel/analytics/dependencies/compute.py`
- `src/codeintel/build/hamilton/native/analytics/dependencies.py`
- `tests/build/hamilton/test_pr51_dependencies_native.py`

**Files Modified:**
- `src/codeintel/analytics/dependencies/__init__.py` (exports)
- `src/codeintel/analytics/dependencies/core.py` (deprecation)
- `src/codeintel/build/registrations.py` (native_module)

**Tables Migrated:**
- `analytics.external_dependency_calls`
- `analytics.external_dependencies`

**Special Notes:**
- This migration required handling a data dependency: `build_external_dependencies()` reads from the `analytics.external_dependency_calls` table that was just written.
- Solution: The materialize node first writes the calls table, then the compute function reads from it to compute aggregated dependencies.
- Uses `CatalogService.from_db()` for catalog access in native modules.

---

### 4. `analytics/entrypoints/core.py` (2 writes) ✅

**Files Created:**
- `src/codeintel/analytics/entrypoints/compute.py`
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
- `tests/build/hamilton/test_pr51_entrypoints_native.py`

**Files Modified:**
- `src/codeintel/analytics/entrypoints/__init__.py` (exports)
- `src/codeintel/analytics/entrypoints/core.py` (deprecation)
- `src/codeintel/build/registrations.py` (native_module)

**Tables Migrated:**
- `analytics.entrypoints` (30 columns)
- `analytics.entrypoint_tests` (9 columns)

**Special Notes:**
- Requires complex input building: catalog, module map, and function features
- Uses `FeaturesProvider` for AST feature loading
- Helper function `_build_inputs()` consolidates input gathering in native module

---

### 5. `analytics/testing/graph_metrics.py` (2 writes) ✅

**Files Created:**
- `src/codeintel/analytics/testing/compute.py`
- `src/codeintel/build/hamilton/native/analytics/test_graph_metrics.py`
- `tests/build/hamilton/test_pr51_test_graph_metrics_native.py`

**Files Modified:**
- `src/codeintel/analytics/testing/__init__.py` (exports)
- `src/codeintel/analytics/testing/graph_metrics.py` (deprecation)
- `src/codeintel/build/registrations.py` (native_module)

**Tables Migrated:**
- `analytics.test_graph_metrics_tests` (12 columns)
- `analytics.test_graph_metrics_functions` (12 columns)

**Special Notes:**
- Requires `GraphRuntime` access for graph-based metrics computation
- Uses `_get_graph_runtime()` helper to resolve from `BuildEnv`
- Follows standard 2-table materialization pattern

---

### 6. `analytics/parsing/validation.py` (2 writes) ✅

**Migration Type:** Utility Class Enhancement (different from standard target replacement)

**Files Created:**
- `src/codeintel/analytics/parsing/compute.py`
- `tests/build/hamilton/test_pr51_validation_reporters_native.py`

**Files Modified:**
- `src/codeintel/analytics/parsing/validation.py` (added `to_rows()` methods, deprecated `flush()`)
- `src/codeintel/analytics/parsing/__init__.py` (exports)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Tables Migrated:**
- `analytics.function_validation` (8 columns)
- `analytics.graph_validation` (10 columns)

**Key Difference - Utility Class Pattern:**

This migration is fundamentally different from previous ones because the validation reporters are **utility classes** used by multiple consumers, not standalone build targets. The approach:

1. **No dedicated Hamilton target** - No new entry in `registrations.py`
2. **Added `to_rows()` methods** to reporters - Returns accumulated rows as tuples
3. **Created materialization helpers** - `materialize_function_validation()`, `materialize_graph_validation()`
4. **Deprecated `flush()` methods** - Continue working with warnings for backward compatibility
5. **Consumer-driven materialization** - Each consumer decides when to materialize

**New Exports:**
```python
from codeintel.analytics.parsing import (
    ValidationResult,
    get_validation_rows,
    materialize_function_validation,
    materialize_graph_validation,
)
```

**Usage Pattern for Consumers:**
```python
# Old (deprecated):
reporter = FunctionValidationReporter(repo, commit)
reporter.record(...)
reporter.flush(gateway)  # ⚠️ Emits DeprecationWarning

# New (preferred):
from codeintel.analytics.parsing import materialize_function_validation
reporter = FunctionValidationReporter(repo, commit)
reporter.record(...)
ref = materialize_function_validation(ctx, reporter)
```

---

### 7. `analytics/compute/coverage/functions.py` (1 write) ✅

**Migration Type:** Native Module with Ibis Expression + Broken Implementation Fix

**Files Created:**
- `src/codeintel/analytics/compute/coverage/compute.py`
- `tests/build/hamilton/test_pr51_coverage_functions_native.py`

**Files Modified:**
- `src/codeintel/analytics/compute/coverage/__init__.py` (exports)
- `src/codeintel/analytics/compute/coverage/functions.py` (deprecation)
- `src/codeintel/build/hamilton/native/analytics/coverage_functions.py` (complete rewrite)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Tables Migrated:**
- `analytics.coverage_functions` (16 columns)

**Key Discovery - Broken Native Module:**

During this migration, we discovered that the existing native module at `coverage_functions.py` had an **incorrect implementation** that didn't match the original algorithm:

1. **Wrong table reference**: Used `q__graph__goids` instead of reading from `core.goids`
2. **Wrong column names**: Expected `executable_lines`/`covered_lines` columns, but actual columns are `is_executable`/`is_covered` (booleans)
3. **Wrong join logic**: Joined on `function_goid_h128` which doesn't exist in coverage_lines - the original joins on `rel_path` + line ranges

**Correct Algorithm (from original):**
```python
# Join GOIDs with coverage lines based on:
# - Same repo, commit, rel_path
# - Line number between function start_line and end_line
join_predicates = [
    goids.repo == coverage.repo,
    goids.commit == coverage.commit,
    goids.rel_path == coverage.rel_path,
    coverage.line >= goids.start_line,
    coverage.line <= ibis.coalesce(goids.end_line, goids.start_line),
]
```

**Solution:**
- Created `build_coverage_functions_expr()` in new `compute.py` with the correct algorithm
- Completely rewrote the native module to call this pure compute function
- Used `materialize_table()` for Ibis expression-based writes

**New Exports:**
```python
from codeintel.analytics.compute.coverage import (
    build_coverage_functions_expr,
    compute_coverage_functions,  # deprecated
)
```

---

### 8. `analytics/compute/data_models/usage.py` (1 write) ✅

**Migration Type:** Pure Function Extraction (simplest pattern)

**Files Created:**
- `tests/build/hamilton/test_pr51_data_model_usage_native.py`

**Files Modified:**
- `src/codeintel/analytics/compute/data_models/usage.py` (added pure function, deprecation)
- `src/codeintel/analytics/compute/data_models/__init__.py` (exports)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Tables Migrated:**
- `analytics.data_model_usage` (8 columns)

**Key Simplification:**

This migration was simpler than most because the existing code already had a well-structured internal `_build_usage_rows()` function that returned row tuples. The migration simply:

1. Created a new public function `build_data_model_usage_rows()` that wraps the internal logic
2. Added deprecation warning to `compute_data_model_usage()`
3. No new Hamilton native module needed - the existing plugin can use the new function

**New Exports:**
```python
from codeintel.analytics.compute.data_models import (
    DATA_MODEL_USAGE_COLS,
    build_data_model_usage_rows,
    compute_data_model_usage,  # deprecated
)
```

**Usage Pattern:**
```python
# Old (deprecated):
compute_data_model_usage(gateway, snapshot, module_map=..., ast_by_goid=...)

# New (preferred):
rows = build_data_model_usage_rows(gateway, snapshot, module_map=..., ast_by_goid=...)
ref = materialize_rows(ctx, "analytics.data_model_usage", rows, DATA_MODEL_USAGE_COLS)
```

---

## Learnings from Completed Migrations

After completing 8 migrations (cfg_dfg, data_models, dependencies, entrypoints, testing/graph_metrics, parsing/validation, coverage_functions, data_model_usage), several patterns and insights have emerged:

### Three Migration Approaches

We've now established three distinct migration patterns:

| Approach | Use When | Example |
|----------|----------|---------|
| **Standard Target Replacement** | Replacing a build plugin/target | cfg_dfg, dependencies, entrypoints |
| **Utility Class Enhancement** | Migrating utility classes used by consumers | parsing/validation |
| **Pure Function Extraction** | Exposing row-building as public API without new target | data_model_usage, coverage_functions |

**Standard Target Replacement** (most common):
- Create new Hamilton native module
- Register with `native_module=` in registrations.py
- Full Hamilton lifecycle management

**Utility Class Enhancement** (for shared utilities):
- Add `to_rows()` methods to existing classes
- Create materialization helper functions
- Deprecate `flush()` methods
- Consumer decides when to materialize

**Pure Function Extraction** (simplest):
- Expose internal row-building logic as public function
- No new Hamilton target registration needed
- Existing plugin/consumer calls new function + Hamilton materializers
- Useful when internal structure is already well-organized

### Input Building Patterns

Different targets require different input sources. Common patterns:

| Target | Input Sources | Helper Pattern |
|--------|---------------|----------------|
| `cfg_dfg` | Gateway only | Direct call |
| `data_models` | Gateway only | Direct call |
| `dependencies` | Gateway + Catalog | `CatalogService.from_db()` |
| `entrypoints` | Gateway + Catalog + Module map + Features | `_build_inputs()` helper |
| `test_graph_metrics` | Gateway + GraphRuntime | `_get_graph_runtime()` helper |
| `validation` | Reporter instances (consumer-owned) | N/A - consumer-driven |

For complex inputs, create a `_build_inputs(env: BuildEnv)` helper in the native module.

### Data Dependency Handling

Some targets have internal data dependencies where one table must be written before another can be computed. The `dependencies` migration demonstrated the solution:

```python
def t__external_deps(env, graph, t__external_deps__compute):
    # 1. Materialize calls table first
    calls_ref = materialize_rows(ctx, "analytics.external_dependency_calls", ...)
    
    # 2. Now compute aggregated deps (reads from calls table)
    deps_result = compute_external_dependencies_pure(env.gateway, env.snapshot)
    
    # 3. Materialize deps table
    deps_ref = materialize_rows(ctx, "analytics.external_dependencies", ...)
```

### Test Robustness

Each test file follows a consistent structure with named constants to avoid magic numbers:

```python
EXPECTED_ENTRYPOINTS_COLS = 30
EXPECTED_ENTRYPOINT_TESTS_COLS = 9
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0
```

### Migration Completion Summary

All 11 modules have been successfully migrated. The final three migrations followed these patterns:

| Module | Pattern | Notes |
|--------|---------|-------|
| `profiles/writer_guard.py` | Pure Function Extraction | Infrastructure code - deprecated, not new target |
| `functions/function_history.py` | Standard Target Replacement | 1-table pattern |
| `history/history_timeseries.py` | Standard Target Replacement | 1-table pattern, special multi-commit config |

---

## Completed Migrations (Final Three)

### 9. `analytics/profiles/writer_guard.py` (1 write) ✅

**Migration Type:** Pure Function Extraction (Infrastructure Deprecation)

**Files Modified:**
- `src/codeintel/analytics/profiles/writer_guard.py` (deprecation warnings added)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Files Created:**
- `tests/build/hamilton/test_pr51_writer_guard_native.py`

**Key Discovery:** This is infrastructure code used by consumers, not a standalone target.

**Migration Approach:**
1. Added deprecation warning to `write_rows_with_registry_guard()` 
2. Added deprecation warning to `create_profile_writer()`
3. Pointed consumers to `materialize_rows` or `write_rows_via_policy_backend`
4. No new Hamilton native module needed

**New Guidance:**
```python
# Old (deprecated):
write_rows_with_registry_guard(gateway, table_key, rows, columns, ...)

# New (preferred):
ref = materialize_rows(ctx, table_key, rows, columns)
# Or for interim solution:
write_rows_via_policy_backend(gateway, table_key, rows, columns)
```

---

### 10. `analytics/functions/function_history.py` (1 write) ✅

**Migration Type:** Standard Target Replacement

**Files Modified:**
- `src/codeintel/analytics/functions/function_history.py` (added pure function, deprecation)
- `src/codeintel/analytics/functions/__init__.py` (exports)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Files Created:**
- `src/codeintel/build/hamilton/native/analytics/function_history.py`
- `tests/build/hamilton/test_pr51_function_history_native.py`

**Tables Migrated:**
- `analytics.function_history` (21 columns)

**New Exports:**
```python
from codeintel.analytics.functions import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
    compute_function_history,  # deprecated
)
```

---

### 11. `analytics/history/history_timeseries.py` (1 write) ✅

**Migration Type:** Standard Target Replacement (Special Multi-Commit)

**Files Modified:**
- `src/codeintel/analytics/history/history_timeseries.py` (added pure function, deprecation)
- `src/codeintel/analytics/history/__init__.py` (exports)
- `tests/build/hamilton/test_pr50_architecture_guardrails.py` (updated comment)

**Files Created:**
- `src/codeintel/build/hamilton/native/analytics/history_timeseries.py`
- `tests/build/hamilton/test_pr51_history_timeseries_native.py`

**Tables Migrated:**
- `analytics.history_timeseries` (19 columns)

**Special Notes:**
- This target is a multi-commit analysis feature requiring special configuration
- The Hamilton native module returns empty when configuration is unavailable via BuildEnv
- For full functionality, use `HistoryTimeseriesPlugin` with explicit configuration
- Both `compute_history_timeseries()` and `compute_history_timeseries_gateways()` deprecated

**New Exports:**
```python
from codeintel.analytics.history import (
    HISTORY_TIMESERIES_COLS,
    build_history_timeseries_rows,
    compute_history_timeseries,  # deprecated
    compute_history_timeseries_gateways,  # deprecated
)
```

---

## Post-Migration Decommissioning Plan

All 11 modules have been migrated to Hamilton native nodes. The following decommissioning work will finalize the migration.

---

### Overview: Decommissioning Phases

| Phase | Description | Estimated Effort | Status |
|-------|-------------|------------------|--------|
| **Phase 1** | Verify Guardrails | Low (1-2 hours) | ✅ Complete |
| **Phase 2** | Update Consumer Code | Medium (depends on consumers) | ✅ Complete |
| **Phase 3** | Remove Deprecated Functions | Medium (4-6 hours) | Pending |
| **Phase 4** | Remove Plugin Classes | Low (2-3 hours) | Pending |
| **Phase 5** | Documentation & Cleanup | Low (1-2 hours) | Pending |

---

### Phase 1: Verify Guardrails ✅ COMPLETE

**Goal:** Verify the guardrail tests correctly identify files with direct `gateway.ibis.write()` calls.

**Status:** Completed December 14, 2025

#### What Was Done

1. **Ran guardrail tests** to confirm the allowlist is correctly configured:
   ```bash
   uv run pytest tests/build/hamilton/test_pr50_architecture_guardrails.py -v
   ```
   Result: All 3 tests passed.

2. **Verified allowlist accuracy** - All 11 files correctly contain deprecated functions with `gateway.ibis.write()` calls that are kept for backward compatibility.

#### Current State

**File:** `tests/build/hamilton/test_pr50_architecture_guardrails.py`

```python
# Current state (11 files allowlisted - all verified):
ALLOWLIST_IBIS_WRITE_FILES = {
    "src/codeintel/analytics/cfg_dfg/materialize.py",  # -> cfg_dfg.py native
    "src/codeintel/analytics/compute/coverage/functions.py",  # -> coverage_functions.py native
    "src/codeintel/analytics/compute/data_models/usage.py",  # -> use build_data_model_usage_rows
    "src/codeintel/analytics/data_models/core.py",  # -> data_models.py native
    "src/codeintel/analytics/dependencies/core.py",  # -> dependencies.py native
    "src/codeintel/analytics/entrypoints/core.py",  # -> entrypoints.py native
    "src/codeintel/analytics/functions/function_history.py",  # -> function_history.py native
    "src/codeintel/analytics/history/history_timeseries.py",  # -> history_timeseries.py native
    "src/codeintel/analytics/parsing/validation.py",  # -> use to_rows() + materialize_*
    "src/codeintel/analytics/profiles/writer_guard.py",  # -> use materialize_rows or write_rows_via_policy_backend
    "src/codeintel/analytics/testing/graph_metrics.py",  # -> test_graph_metrics.py native
}

# Target state after Phase 3 (empty):
ALLOWLIST_IBIS_WRITE_FILES: set[str] = set()
```

**IMPORTANT:** The allowlist cannot be emptied until all deprecated functions are removed in Phase 3. The deprecated functions still contain `gateway.ibis.write()` calls for backward compatibility.

---

### Phase 2: Update Consumer Code ✅ COMPLETE

**Goal:** Identify and update all code that calls deprecated functions to use the new patterns.

**Status:** Completed December 14, 2025

#### Consumer Inventory Results

| Consumer File | Deprecated Function | Action Taken |
|---------------|---------------------|--------------|
| `build/plugins/analytics/functions/history.py` | `compute_function_history` | Deferred to Phase 4 (plugin removal) |
| `build/plugins/analytics/coverage/functions.py` | `compute_coverage_functions` | Deferred to Phase 4 (plugin removal) |
| `build/plugins/analytics/data_models/usage.py` | `compute_data_model_usage` | Deferred to Phase 4 (plugin removal) |
| `analytics/testing/profiles/rows.py` | `write_rows_with_registry_guard` | ✅ Updated to `write_rows_via_policy_backend` |
| `analytics/functions/metrics.py` | `FunctionValidationReporter.flush()` | ✅ Updated to `to_rows()` + policy backend |
| `graphs/validation/findings.py` | `GraphValidationReporter.flush()` | ✅ Updated to `to_rows()` + policy backend |

#### Files Modified

**1. `src/codeintel/analytics/testing/profiles/rows.py`**

Updated `write_test_profile_rows()` and `write_behavioral_coverage_rows()` to use `write_rows_via_policy_backend`:

```python
# Before (deprecated):
from codeintel.analytics.profiles.writer_guard import (
    WriterContext,
    write_rows_with_registry_guard,
)

# After:
from codeintel.analytics.profiles.writer_guard import (
    PolicyWriterConfig,
    write_rows_via_policy_backend,
)

def write_test_profile_rows(gateway, snapshot, rows) -> int:
    config = PolicyWriterConfig(
        table_key="analytics.test_profile",
        columns=TEST_PROFILE_COLUMNS,
        serialize_row=cast("SerializeRow", serialize_test_profile_row),
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows_list, config=config)
```

**2. `src/codeintel/analytics/functions/metrics.py`**

Replaced `result.reporter.flush(gateway)` with direct policy backend write:

```python
# Before (deprecated):
result.reporter.flush(gateway)

# After:
validation_rows = result.reporter.to_rows()
if validation_rows:
    backend.delete_for_snapshot(
        "analytics.function_validation", repo=snapshot.repo, commit=snapshot.commit
    )
    backend.bulk_insert(
        "analytics.function_validation",
        list(validation_rows),
        columns=list(FUNCTION_VALIDATION_COLS),
    )
```

**3. `src/codeintel/graphs/validation/findings.py`**

Replaced `reporter.flush(gateway)` with direct policy backend write:

```python
# Before (deprecated):
reporter.flush(gateway)

# After:
validation_rows = reporter.to_rows()
if validation_rows:
    backend = DuckDBPolicyBackend(gateway)
    backend.bulk_insert(
        "analytics.graph_validation",
        list(validation_rows),
        columns=list(GRAPH_VALIDATION_COLS),
    )
```

#### Implementation Insights

1. **Type Compatibility**: When using `PolicyWriterConfig`, the `serialize_row` parameter requires a `cast("SerializeRow", ...)` to satisfy pyright because the concrete row types (`ProfileRowModel`, `BehavioralCoverageRowModel`) are not directly assignable to `Mapping[str, object]`.

2. **Plugin Updates Deferred**: Build plugins (`FunctionHistoryPlugin`, `CoverageFunctionsPlugin`, `DataModelUsagePlugin`) still call deprecated functions but are slated for complete removal in Phase 4. No changes needed now.

3. **Validation Consistency**: All three modified files now follow the same pattern: `to_rows()` + `DuckDBPolicyBackend.bulk_insert()`.

#### Verification Results

All quality checks passed:
- **Ruff**: All checks passed
- **Pyright**: 0 errors, 0 warnings
- **Pyrefly**: 0 errors
- **Tests**: 121 tests passed (including all guardrail and PR51 tests)

---

### Phase 3: Remove Deprecated Functions

**Goal:** Delete all deprecated functions and their direct `gateway.ibis.write()` calls.

#### Deprecated Functions Inventory

| # | Module | Functions to Remove | Direct Write? |
|---|--------|---------------------|---------------|
| 1 | `cfg_dfg/materialize.py` | `compute_cfg_metrics()` | Yes |
| 2 | `cfg_dfg/materialize.py` | `compute_dfg_metrics()` | Yes |
| 3 | `data_models/core.py` | `compute_data_models()` | Yes |
| 4 | `data_models/core.py` | `_persist_models()` | Yes |
| 5 | `dependencies/core.py` | `build_external_dependency_calls()` | Yes |
| 6 | `dependencies/core.py` | `build_external_dependencies()` | Yes |
| 7 | `entrypoints/core.py` | `build_entrypoints()` | Yes |
| 8 | `testing/graph_metrics.py` | `compute_test_graph_metrics()` | Yes |
| 9 | `parsing/validation.py` | `FunctionValidationReporter.flush()` | Yes |
| 10 | `parsing/validation.py` | `GraphValidationReporter.flush()` | Yes |
| 11 | `compute/coverage/functions.py` | `compute_coverage_functions()` | Yes |
| 12 | `compute/data_models/usage.py` | `compute_data_model_usage()` | Yes |
| 13 | `profiles/writer_guard.py` | `write_rows_with_registry_guard()` | Yes |
| 14 | `profiles/writer_guard.py` | `create_profile_writer()` | No (factory) |
| 15 | `functions/function_history.py` | `compute_function_history()` | Yes |
| 16 | `history/history_timeseries.py` | `compute_history_timeseries()` | Yes |
| 17 | `history/history_timeseries.py` | `compute_history_timeseries_gateways()` | No (wrapper) |

#### Step 3.1: Remove Functions (Per Module)

For each module, follow this checklist:

- [ ] **`cfg_dfg/materialize.py`**
  - Delete `compute_cfg_metrics()` (lines ~150-250)
  - Delete `compute_dfg_metrics()` (lines ~250-350)
  - Remove `import warnings` if no longer needed
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/cfg_dfg/`

- [ ] **`data_models/core.py`**
  - Delete `compute_data_models()` 
  - Delete `_persist_models()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/data_models/`

- [ ] **`dependencies/core.py`**
  - Delete `build_external_dependency_calls()`
  - Delete `build_external_dependencies()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/dependencies/`

- [ ] **`entrypoints/core.py`**
  - Delete `build_entrypoints()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/entrypoints/`

- [ ] **`testing/graph_metrics.py`**
  - Delete `compute_test_graph_metrics()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/testing/`

- [ ] **`parsing/validation.py`**
  - Delete `FunctionValidationReporter.flush()` method
  - Delete `GraphValidationReporter.flush()` method
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/parsing/`

- [ ] **`compute/coverage/functions.py`**
  - Delete `compute_coverage_functions()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/compute/coverage/`

- [ ] **`compute/data_models/usage.py`**
  - Delete `compute_data_model_usage()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/compute/data_models/`

- [ ] **`profiles/writer_guard.py`**
  - Delete `write_rows_with_registry_guard()`
  - Delete `create_profile_writer()`
  - Update `__all__` in module
  - Run: `uv run ruff check --fix src/codeintel/analytics/profiles/`

- [ ] **`functions/function_history.py`**
  - Delete `compute_function_history()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/functions/`

- [ ] **`history/history_timeseries.py`**
  - Delete `compute_history_timeseries()`
  - Delete `compute_history_timeseries_gateways()`
  - Update `__all__` in `__init__.py`
  - Run: `uv run ruff check --fix src/codeintel/analytics/history/`

#### Step 3.2: Remove Module Deprecation Docstrings

After removing deprecated functions, update module docstrings to remove the deprecation notices:

```python
# Before:
"""Module description.

.. deprecated::
    The ``compute_foo`` function contains direct database writes.
    For new code, use ``build_foo_rows`` with Hamilton materializers.
"""

# After:
"""Module description.

This module provides pure compute functions for Hamilton native execution.
"""
```

#### Step 3.3: Update Tests

Remove or update tests that specifically test deprecated functions:

```bash
# Find tests for deprecated functions
rg "test_.*_deprecation" tests/build/hamilton/test_pr51_*.py
```

**Action:** Either remove these tests or convert them to tests for the replacement functions.

#### Step 3.4: Empty Allowlist

After all deprecated functions are removed:

```python
# tests/build/hamilton/test_pr50_architecture_guardrails.py
ALLOWLIST_IBIS_WRITE_FILES: set[str] = set()
```

#### Step 3.5: Verify

```bash
# Full quality check
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest tests/build/hamilton/test_pr50_architecture_guardrails.py -v
uv run pytest tests/build/hamilton/test_pr51_*.py -v
```

---

### Phase 4: Remove Plugin Classes

**Goal:** Delete plugin classes that have been replaced by Hamilton native modules.

#### Plugins to Remove

| Plugin | Location | Native Replacement |
|--------|----------|-------------------|
| `CfgDfgMetricsPlugin` | `build/plugins/analytics/cfg_dfg/metrics.py` | `native/analytics/cfg_dfg.py` |
| `DataModelsPlugin` | `build/plugins/analytics/data_models/plugin.py` | `native/analytics/data_models.py` |
| `ExternalDepsPlugin` | `build/plugins/analytics/dependencies/plugin.py` | `native/analytics/dependencies.py` |
| `EntrypointsPlugin` | `build/plugins/analytics/entrypoints/plugin.py` | `native/analytics/entrypoints.py` |
| `TestGraphMetricsPlugin` | `build/plugins/analytics/testing/metrics.py` | `native/analytics/test_graph_metrics.py` |
| `CoverageFunctionsPlugin` | `build/plugins/analytics/coverage/functions.py` | `native/analytics/coverage_functions.py` |
| `FunctionHistoryPlugin` | `build/plugins/analytics/functions/history.py` | `native/analytics/function_history.py` |
| `HistoryTimeseriesPlugin` | `build/plugins/analytics/history/timeseries.py` | `native/analytics/history_timeseries.py` |

#### Step 4.1: Verify No External Consumers

```bash
# Search for plugin class references
rg "CfgDfgMetricsPlugin" --type py -l
rg "DataModelsPlugin" --type py -l
# ... etc
```

**Expected results:** Only the plugin definition file and tests should reference these.

#### Step 4.2: Remove Plugin Files

For each plugin:
1. Delete the plugin file
2. Update the package `__init__.py`
3. Remove from any plugin registries
4. Delete plugin-specific tests

#### Step 4.3: Verify Registrations

Ensure `src/codeintel/build/registrations.py` uses `native_module=` for all migrated targets.

---

### Phase 5: Documentation & Cleanup

**Goal:** Update documentation to reflect Hamilton-first architecture.

#### Step 5.1: Update AGENTS.md

- Remove references to deprecated functions
- Update code examples to use Hamilton patterns
- Add section on native module development

#### Step 5.2: Archive Migration Docs

- Move `docs/Hamilton_consolidation/` to `docs/archive/Hamilton_consolidation/`
- Or create a summary document and delete detailed plans

#### Step 5.3: Clean Up Tests

Remove test files that are no longer needed:
- Deprecation warning tests (after functions removed)
- Allowlist presence tests (after allowlist emptied)

---

### Decommissioning Verification Checklist

**Phase 1-2 Progress:**
- [x] Guardrail tests pass with current allowlist (Phase 1)
- [x] Consumer code audit complete (Phase 2)
- [x] Non-plugin consumers updated to new patterns (Phase 2)
- [x] All 121 tests pass (Phase 2)
- [x] Quality checks clean: ruff, pyright, pyrefly (Phase 2)

**Phase 3-5 Remaining:**
- [ ] `ALLOWLIST_IBIS_WRITE_FILES` is empty (after Phase 3)
- [ ] `test_pr50_no_ibis_write_outside_build_allowlist` passes with empty allowlist (after Phase 3)
- [ ] No deprecated functions remain in analytics modules (Phase 3)
- [ ] No plugin classes remain for migrated targets (Phase 4)
- [ ] All registrations use `native_module=` parameter (Phase 4)
- [ ] Documentation updated (Phase 5)
- [ ] Full test suite passes: `uv run pytest -q`
- [ ] Quality report clean: `uv run python -m tools.quality_report`

---

### Success Metrics

| Metric | Before Migration | After Migration | After Phase 2 | After Decommission |
|--------|------------------|-----------------|---------------|-------------------|
| Direct writes outside build | 22 | 0 (deprecated) | 0 (deprecated) | 0 |
| Allowlisted files | 11 | 11 | 11 | 0 |
| Deprecated functions | 0 | 17 | 17 | 0 |
| Consumer code using deprecated | N/A | 6 | 3 (plugins only) | 0 |
| Native analytics modules | 4 | 12 | 12 | 12 |
| Plugin classes | 8+ | 8 | 8 | 0 |
| Tables via Hamilton | ~30 | 52 | 52 | 52 |

**Phase 2 Progress:** Updated 3 consumer files to use new patterns. Remaining 3 consumers are build plugins that will be removed in Phase 4.

---

## Directory Structure After Migration

```
src/codeintel/
├── analytics/
│   ├── cfg_dfg/
│   │   ├── __init__.py          # Exports pure compute + result
│   │   ├── compute.py           # CfgMetricsResult, DfgMetricsResult, pure functions ✅
│   │   └── materialize.py       # DEPRECATED - kept for backward compat
│   ├── data_models/
│   │   ├── __init__.py          # Exports pure compute + result
│   │   ├── compute.py           # DataModelsResult, compute_data_models_pure ✅
│   │   └── core.py              # DEPRECATED - kept for backward compat
│   ├── dependencies/
│   │   ├── __init__.py          # Exports pure compute + result
│   │   ├── compute.py           # DependencyCallsResult, ExternalDependenciesResult ✅
│   │   └── core.py              # DEPRECATED - kept for backward compat
│   ├── entrypoints/
│   │   ├── __init__.py          # Exports pure compute + result
│   │   ├── compute.py           # EntrypointsResult, compute_entrypoints_pure ✅
│   │   └── core.py              # DEPRECATED - kept for backward compat
│   ├── testing/
│   │   ├── __init__.py          # Exports pure compute + result
│   │   ├── compute.py           # TestGraphMetricsResult, compute_test_graph_metrics_pure ✅
│   │   └── graph_metrics.py     # DEPRECATED - kept for backward compat
│   ├── parsing/
│   │   ├── __init__.py          # Exports ValidationResult + materialization helpers
│   │   ├── compute.py           # ValidationResult, get_validation_rows, materialize_* ✅
│   │   └── validation.py        # to_rows() added, flush() DEPRECATED
│   ├── compute/
│   │   ├── coverage/
│   │   │   ├── __init__.py      # Exports build_coverage_functions_expr ✅
│   │   │   ├── compute.py       # build_coverage_functions_expr, pure helpers ✅
│   │   │   └── functions.py     # compute_coverage_functions DEPRECATED
│   │   └── data_models/
│   │       ├── __init__.py      # Exports build_data_model_usage_rows ✅
│   │       └── usage.py         # build_data_model_usage_rows ✅, compute_data_model_usage DEPRECATED
│   ├── functions/
│   │   ├── __init__.py          # Exports build_function_history_rows ✅
│   │   └── function_history.py  # build_function_history_rows ✅, compute_function_history DEPRECATED
│   ├── history/
│   │   ├── __init__.py          # Exports build_history_timeseries_rows ✅
│   │   └── history_timeseries.py # build_history_timeseries_rows ✅, compute_* DEPRECATED
│   ├── profiles/
│   │   └── writer_guard.py      # write_rows_with_registry_guard DEPRECATED, create_profile_writer DEPRECATED
│   └── ...
├── build/
│   ├── hamilton/
│   │   └── native/
│   │       └── analytics/
│   │           ├── __init__.py
│   │           ├── cfg_dfg.py            # t__cfg_dfg_metrics*, etc. ✅
│   │           ├── data_models.py        # t__data_models*, etc. ✅
│   │           ├── dependencies.py       # t__external_deps*, etc. ✅
│   │           ├── entrypoints.py        # t__entrypoints*, etc. ✅
│   │           ├── test_graph_metrics.py # t__test_graph_metrics*, etc. ✅
│   │           ├── coverage_functions.py # t__coverage_functions*, etc. ✅
│   │           ├── function_history.py   # t__function_history*, etc. ✅
│   │           └── history_timeseries.py # t__history_timeseries*, etc. ✅
│   └── registrations.py         # Native modules registered
└── ...

tests/build/hamilton/
├── test_pr50_architecture_guardrails.py        # Allowlist (11 files, to be emptied)
├── test_pr51_cfg_dfg_native.py                 # ✅ Complete
├── test_pr51_data_models_native.py             # ✅ Complete
├── test_pr51_dependencies_native.py            # ✅ Complete
├── test_pr51_entrypoints_native.py             # ✅ Complete
├── test_pr51_test_graph_metrics_native.py      # ✅ Complete
├── test_pr51_validation_reporters_native.py    # ✅ Complete
├── test_pr51_coverage_functions_native.py      # ✅ Complete
├── test_pr51_data_model_usage_native.py        # ✅ Complete
├── test_pr51_writer_guard_native.py            # ✅ Complete
├── test_pr51_function_history_native.py        # ✅ Complete
└── test_pr51_history_timeseries_native.py      # ✅ Complete
```

---

## Testing Strategy

### Test File Pattern

Each native module gets a test file with:

```python
"""PR51: Tests for <target> native Hamilton module."""

from __future__ import annotations

# Standard library imports
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Import pure compute function and result type
from codeintel.analytics.<domain> import (
    <Target>Result,
    compute_<target>_pure,
)

# Import column constants
from codeintel.analytics.<domain>.core import (
    <TABLE>_COLS,
)

# Import native module for export tests
from codeintel.build.hamilton.native.analytics import <module> as native_module

# Import materializer helpers
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)

# Test helpers
from tests._helpers.builders import insert_rows, GoidRow, ModuleRow

if TYPE_CHECKING:
    from tests._helpers import TestContext


# Constants for magic number avoidance
EXPECTED_<TABLE>_COLS = len(<TABLE>_COLS)
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0


# =====================================================================
# Tests for compute_<target>_pure
# =====================================================================

def test_<target>_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_<target>_pure returns <Target>Result type."""
    ...


def test_<target>_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty input returns empty result without error."""
    ...


def test_<target>_pure_with_data_produces_rows(test_ctx: TestContext, tmp_path: Path) -> None:
    """Verify compute_<target>_pure produces rows when data exists."""
    ...


# =====================================================================
# Tests for materialize_rows with <target>
# =====================================================================

def test_materialize_rows_writes_<table>(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes <table> rows to database."""
    ...


def test_materialize_rows_handles_empty_<table>(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty row list gracefully."""
    ...


# =====================================================================
# Architecture guardrail tests
# =====================================================================

def test_<module>_core_in_allowlist() -> None:
    """Verify <module>/core.py is in allowlist for backward compat."""
    ...


# =====================================================================
# Deprecation warning tests
# =====================================================================

def test_compute_<target>_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_<target> emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="<target> is deprecated"):
        ...


# =====================================================================
# Native module export tests
# =====================================================================

def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {"t__<target>", "t__<target>__compute"}
    actual = set(native_module.__all__)
    assert actual == expected


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    for node in [native_module.t__<target>, native_module.t__<target>__compute]:
        assert hasattr(node, "decorate_nodes")
```

### Integration Test Pattern

```python
def test_<target>_native_integration(tmp_path: Path) -> None:
    """Verify <target> target writes correct data via Hamilton."""
    gateway = create_ephemeral_gateway(tmp_path)
    seed_required_data(gateway)
    
    result = run_hamilton_target("<target>", gateway=gateway, snapshot=snapshot)
    
    assert result.success
    rows = gateway.con.execute("SELECT * FROM analytics.<table>").fetchall()
    assert len(rows) > 0
```

---

## Success Criteria

| Metric | Before Migration | After Migration | After Decommission |
|--------|------------------|-----------------|-------------------|
| Direct writes outside build | 22 | 0 (all deprecated) | 0 |
| Allowlisted files | 11 | 11 | 0 |
| Deprecated functions | 0 | 17 | 0 |
| Native analytics modules | 4 | 12 | 12 |
| Tables via Hamilton | ~30 | 52 | 52 |
| Test files | 1 | 12 | 12 |

**Notes:**
- All 11 modules have been migrated with pure compute functions and/or Hamilton native nodes
- The validation reporters and data_model_usage added pure functions and materialization helpers
- The coverage_functions native module was rewritten to fix a broken implementation
- The history_timeseries module requires special multi-commit configuration via plugin
- The writer_guard module is infrastructure code - deprecated without a new Hamilton target
- Decommissioning (Phase 1-5) will remove deprecated functions and empty the allowlist

---

## Appendix A: Quick Reference

### Creating a New Native Module

1. **Create compute.py:**
   ```bash
   touch src/codeintel/analytics/<domain>/compute.py
   ```

2. **Define result dataclass:**
   ```python
   @dataclass(frozen=True)
   class <Target>Result:
       rows: tuple[tuple[object, ...], ...]
   ```

3. **Implement pure compute:**
   ```python
   def compute_<target>_pure(gateway, snapshot) -> <Target>Result:
       # Read-only logic, return rows
       return <Target>Result(rows=tuple(rows))
   ```

4. **Create native module:**
   ```bash
   touch src/codeintel/build/hamilton/native/analytics/<target>.py
   ```

5. **Implement Hamilton nodes:**
   ```python
   @tag(domain="analytics", target="<target>", node_type="compute")
   def t__<target>__compute(env: BuildEnv) -> <Target>Result:
       return compute_<target>_pure(env.gateway, env.snapshot)
   
   @tag(domain="analytics", target="<target>", node_type="materialize")
   def t__<target>(env, graph, t__<target>__compute) -> TargetRunRecord:
       executor = NativeTargetExecutor.for_target(env, graph, "<target>")
       if executor.should_skip():
           return executor.skip()
       def compute():
           ctx = MaterializationContext(...)
           ref = materialize_rows(ctx, "analytics.<table>", result.rows, COLS)
           return {"analytics.<table>": ref.row_count or 0}
       return executor.execute(compute)
   ```

6. **Update registration:**
   ```python
   registry.register(
       <TARGET>_TARGET,
       native_module="codeintel.build.hamilton.native.analytics.<target>",
   )
   ```

7. **Add deprecation warning to original**

8. **Update exports in `__init__.py`**

9. **Create tests**

### Running Tests

```bash
# Run all PR51 tests
uv run pytest tests/build/hamilton/test_pr51_*.py -v

# Run architecture guardrails
uv run pytest tests/build/hamilton/test_pr50_architecture_guardrails.py -v

# Run quality checks
uv run ruff check --fix src/codeintel/analytics/<domain>/compute.py
uv run pyright src/codeintel/analytics/<domain>/compute.py
uv run pyrefly check src/codeintel/analytics/<domain>/compute.py
```
