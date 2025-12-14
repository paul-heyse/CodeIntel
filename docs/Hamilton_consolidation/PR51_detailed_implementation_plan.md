# PR-51: Eliminate DB Writes in Analytics — Detailed Implementation Plan

> **Status**: In Progress (4 of 11 modules migrated)  
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
| 5 | `analytics/testing/graph_metrics.py` | 2 | ⏳ Pending |
| 6 | `analytics/parsing/validation.py` | 2 | ⏳ Pending |
| 7 | `analytics/compute/coverage/functions.py` | 1 | ⏳ Pending |
| 8 | `analytics/compute/data_models/usage.py` | 1 | ⏳ Pending |
| 9 | `analytics/profiles/writer_guard.py` | 1 | ⏳ Pending |
| 10 | `analytics/functions/function_history.py` | 1 | ⏳ Pending |
| 11 | `analytics/history/history_timeseries.py` | 1 | ⏳ Pending |

**Tables migrated**: 13 of 22

---

## Established Migration Pattern

After completing cfg_dfg, data_models, dependencies, and entrypoints migrations (4 targets), we have established a robust, consistent pattern that all subsequent migrations will follow. This pattern has proven reliable across targets with varying complexity.

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

## Learnings from Completed Migrations

After completing 4 migrations (cfg_dfg, data_models, dependencies, entrypoints), several patterns and insights have emerged:

### Input Building Patterns

Different targets require different input sources. Common patterns:

| Target | Input Sources | Helper Pattern |
|--------|---------------|----------------|
| `cfg_dfg` | Gateway only | Direct call |
| `data_models` | Gateway only | Direct call |
| `dependencies` | Gateway + Catalog | `CatalogService.from_db()` |
| `entrypoints` | Gateway + Catalog + Module map + Features | `_build_inputs()` helper |

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

### Remaining Work Complexity Assessment

Based on completed migrations, remaining work is categorized:

| Complexity | Target | Notes |
|------------|--------|-------|
| **Low** | `profiles/writer_guard.py` | Infrastructure, just deprecate |
| **Medium** | `testing/graph_metrics.py` | Standard 2-table pattern |
| **Medium** | `parsing/validation.py` | Reporter class pattern |
| **Medium** | `compute/data_models/usage.py` | Standard 1-table pattern |
| **Medium** | `functions/function_history.py` | Standard 1-table pattern |
| **Medium** | `history/history_timeseries.py` | Standard 1-table pattern |
| **High** | `compute/coverage/functions.py` | Ibis expression-based, needs different materializer |

---

## Remaining Migrations

### 5. `analytics/testing/graph_metrics.py` (2 writes)

**Current State:**
- `compute_test_graph_metrics()` writes:
  - `analytics.test_graph_metrics_tests`
  - `analytics.test_graph_metrics_functions`

**Tables & Columns:**
```python
TEST_GRAPH_METRICS_TESTS_COLS = [
    "test_id", "repo", "commit", "degree", "weighted_degree", "degree_centrality",
    "proj_degree", "proj_weight", "proj_clustering", "proj_betweenness",
    "risk_weighted_degree", "created_at",
]
TEST_GRAPH_METRICS_FUNCTIONS_COLS = [
    "function_goid_h128", "repo", "commit", "tests_degree", "tests_weighted_degree",
    "tests_degree_centrality", "proj_degree", "proj_weight", "proj_clustering",
    "proj_betweenness", "tests_risk_weighted_degree", "created_at",
]
```

**Migration Steps:**

1. Create `src/codeintel/analytics/testing/compute.py`:
   ```python
   @dataclass(frozen=True)
   class TestGraphMetricsResult:
       test_rows: tuple[tuple[object, ...], ...]
       function_rows: tuple[tuple[object, ...], ...]
   
   def compute_test_graph_metrics_pure(...) -> TestGraphMetricsResult:
       ...
   ```

2. Create `src/codeintel/build/hamilton/native/analytics/test_graph_metrics.py`

3. Update registration, add deprecation, create tests

---

### 6. `analytics/parsing/validation.py` (2 writes)

**Current State:**
- `FunctionValidationReporter.flush()` → `analytics.function_validation`
- `GraphValidationReporter.flush()` → `analytics.graph_validation`

**Special Consideration:** Reporter classes accumulate rows and flush.

**Tables & Columns:**
```python
FUNCTION_VALIDATION_COLS = [
    "repo", "commit", "function_goid_h128", "rel_path", "qualname",
    "issue", "detail", "created_at",
]
GRAPH_VALIDATION_COLS = [
    "repo", "commit", "graph_name", "entity_id", "issue", "severity",
    "rel_path", "detail", "metadata", "created_at",
]
```

**Migration Steps:**

1. Add `to_rows()` method to reporters:
   ```python
   class FunctionValidationReporter:
       def to_rows(self) -> tuple[tuple[object, ...], ...]:
           """Return accumulated rows without writing."""
           return tuple(self._rows)
   ```

2. Create `src/codeintel/build/hamilton/native/analytics/validation.py`

3. Update registration, add deprecation, create tests

---

### 7. `analytics/compute/coverage/functions.py` (1 write)

**Current State:**
- `compute_coverage_functions()` uses Ibis expressions
- `_write_coverage_results()` writes `analytics.coverage_functions`

**Special Consideration:** Uses Ibis expression-based writes, not row tuples.

**Migration Steps:**

1. Extract expression builder:
   ```python
   def build_coverage_functions_expr(
       gateway: StorageGateway,
       snapshot: SnapshotRef,
   ) -> ibis.Table:
       """Build Ibis expression for coverage functions."""
       ...
   ```

2. Create native node with Ibis materializer:
   ```python
   def t__coverage_functions(ctx, expr: ibis.Table) -> DatasetRef:
       return materialize_ibis_expr(ctx, "analytics.coverage_functions", expr)
   ```

---

### 8. `analytics/compute/data_models/usage.py` (1 write)

**Current State:**
- `compute_data_model_usage()` → `analytics.data_model_usage`

**Tables & Columns:**
```python
DATA_MODEL_USAGE_COLS = [
    "repo", "commit", "model_id", "function_goid_h128", "usage_kinds_json",
    "evidence_json", "context_json", "created_at",
]
```

**Migration Steps:**
Standard pattern - extract pure compute, create native node.

---

### 9. `analytics/profiles/writer_guard.py` (1 write)

**Current State:**
- `write_rows_with_registry_guard()` is a generic writer utility

**Special Consideration:** Infrastructure code, not a specific analytics target.

**Migration Steps:**

1. Mark `write_rows_with_registry_guard()` as deprecated
2. Add warning pointing to Hamilton materializers
3. Keep for backward compatibility

---

### 10. `analytics/functions/function_history.py` (1 write)

**Current State:**
- `compute_function_history()` → `analytics.function_history`

**Tables & Columns:**
```python
FUNCTION_HISTORY_COLS = [
    "repo", "commit", "function_goid_h128", "urn", "rel_path", "module",
    "qualname", "created_in_commit", "created_at", "last_modified_commit",
    "last_modified_at", "age_days", "commit_count", "author_count",
    "lines_added", "lines_deleted", "churn_score", "stability_bucket",
    "history_window_start", "history_window_end", "created_at_row",
]
```

**Migration Steps:**
Standard pattern - extract pure compute, create native node.

---

### 11. `analytics/history/history_timeseries.py` (1 write)

**Current State:**
- `compute_history_timeseries()` → `analytics.history_timeseries`

**Tables & Columns:**
```python
HISTORY_TIMESERIES_COLS = [
    "repo", "entity_kind", "entity_stable_id", "function_goid_h128", "module",
    "rel_path", "language", "qualname", "commit", "commit_ts", "loc",
    "cyclomatic_complexity", "coverage_ratio", "static_error_count",
    "typedness_bucket", "risk_score", "risk_level", "bucket_label",
    "created_at_row",
]
```

**Migration Steps:**
Standard pattern - extract pure compute, create native node.

---

## Post-Migration Deprecation Work

After all 11 modules have been migrated to Hamilton native nodes, the following cleanup work will be performed:

### Phase 1: Allowlist Reduction

1. **Remove files from `ALLOWLIST_IBIS_WRITE_FILES`:**
   ```python
   # test_pr50_architecture_guardrails.py
   
   # Before:
   ALLOWLIST_IBIS_WRITE_FILES: set[str] = {
       "src/codeintel/analytics/cfg_dfg/materialize.py",
       "src/codeintel/analytics/data_models/core.py",
       # ... remaining files
   }
   
   # After (target state):
   ALLOWLIST_IBIS_WRITE_FILES: set[str] = set()  # Empty!
   ```

2. **Verify guardrail test passes with empty allowlist:**
   ```bash
   uv run pytest tests/build/hamilton/test_pr50_architecture_guardrails.py -v
   ```

### Phase 2: Remove Deprecated Functions

After a deprecation period (suggested: 2 release cycles), remove the deprecated functions:

| Module | Functions to Remove | Status |
|--------|---------------------|--------|
| `cfg_dfg/materialize.py` | `compute_cfg_metrics()`, `compute_dfg_metrics()` | ⚠️ Deprecated |
| `data_models/core.py` | `compute_data_models()`, `_persist_models()` | ⚠️ Deprecated |
| `dependencies/core.py` | `build_external_dependency_calls()`, `build_external_dependencies()` | ⚠️ Deprecated |
| `entrypoints/core.py` | `build_entrypoints()` | ⚠️ Deprecated |
| `testing/graph_metrics.py` | `compute_test_graph_metrics()` | ⏳ Pending |
| `parsing/validation.py` | `FunctionValidationReporter.flush()`, `GraphValidationReporter.flush()` | ⏳ Pending |
| `compute/coverage/functions.py` | `_write_coverage_results()` | ⏳ Pending |
| `compute/data_models/usage.py` | `compute_data_model_usage()` | ⏳ Pending |
| `profiles/writer_guard.py` | `write_rows_with_registry_guard()` | ⏳ Pending |
| `functions/function_history.py` | `compute_function_history()` | ⏳ Pending |
| `history/history_timeseries.py` | `compute_history_timeseries()` | ⏳ Pending |

### Phase 3: Plugin Removal

After native modules are stable:

1. **Remove unused plugin classes:**
   - `CfgDfgMetricsPlugin`
   - `DataModelsPlugin`
   - `ExternalDepsPlugin`
   - `EntrypointsPlugin`
   - ... etc.

2. **Clean up plugin imports in `__init__.py` files**

3. **Remove plugin-specific test files** (if not testing backward compat)

### Phase 4: Documentation Update

1. Update `AGENTS.md` to reflect Hamilton-first architecture
2. Update developer guides with new patterns
3. Archive migration documentation

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
│   └── ...
├── build/
│   ├── hamilton/
│   │   └── native/
│   │       └── analytics/
│   │           ├── __init__.py
│   │           ├── cfg_dfg.py         # t__cfg_dfg_metrics*, etc. ✅
│   │           ├── data_models.py     # t__data_models*, etc. ✅
│   │           ├── dependencies.py    # t__external_deps*, etc. ✅
│   │           ├── entrypoints.py     # t__entrypoints*, etc. ✅
│   │           ├── test_graph.py      # PENDING
│   │           ├── validation.py      # PENDING
│   │           ├── coverage.py        # PENDING
│   │           ├── model_usage.py     # PENDING
│   │           ├── function_history.py# PENDING
│   │           └── timeseries.py      # PENDING
│   └── registrations.py         # 4 of 11 use native_module=
└── ...

tests/build/hamilton/
├── test_pr50_architecture_guardrails.py  # Allowlist shrinking
├── test_pr51_cfg_dfg_native.py           # ✅ Complete
├── test_pr51_data_models_native.py       # ✅ Complete
├── test_pr51_dependencies_native.py      # ✅ Complete
├── test_pr51_entrypoints_native.py       # ✅ Complete
├── test_pr51_test_graph_metrics_native.py# PENDING
├── test_pr51_validation_native.py        # PENDING
├── test_pr51_coverage_functions_native.py# PENDING
├── test_pr51_model_usage_native.py       # PENDING
├── test_pr51_function_history_native.py  # PENDING
└── test_pr51_history_timeseries_native.py# PENDING
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

| Metric | Before | Current | Target |
|--------|--------|---------|--------|
| Direct writes outside build | 22 | 9 | 0 |
| Allowlisted files | 11 | 9 | 0 |
| Native analytics modules | 4 | 8 | 14 |
| Tables via Hamilton | ~30 | 43 | 52 |
| Test files | 1 | 5 | 12 |

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
