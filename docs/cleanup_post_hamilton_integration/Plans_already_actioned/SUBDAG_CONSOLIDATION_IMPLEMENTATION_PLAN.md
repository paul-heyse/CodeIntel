# Subdag-Based Native Module Consolidation: Implementation Plan

> **Status**: Phase 1 ✓ | Phase 2 ✓ | Phase 3-5 Ready  
> **Author**: AI Assistant  
> **Date**: 2025-12-17 (Updated: 2025-12-17)  
> **Scope**: Consolidate 38 native module files → 15-18 files using Hamilton @subdag/@parameterize  
> **Impact**: ~5,000-7,000 lines reduced; unified patterns; easier maintenance

---

## Phase Status Tracker

| Phase | Status | Deliverables | Notes |
|-------|--------|--------------|-------|
| **Phase 1: Foundation** | ✅ Complete | `executor_pipeline.py`, `multi_table_pipeline.py`, row helpers | 29 tests passing |
| **Phase 2: Analytics** | ✅ Complete | `coverage_targets.py`, `metrics_targets.py` | 9 targets consolidated, 3 files deleted, 593 tests passing |
| **Phase 3: Graphs** | 🔜 Ready | Consolidate support, metrics targets | |
| **Phase 4: Ingestion** | ⏳ Pending | Simplify ingest_targets.py | |
| **Phase 5: Export** | ⏳ Pending | Parameterize export formats | |

---

## Executive Summary

This plan provides a detailed, actionable blueprint for consolidating the native Hamilton modules using Hamilton's `@subdag` and `@parameterize` decorators. The consolidation leverages shape-based pipeline patterns identified through comprehensive codebase analysis.

### Key Insight

The 38 native module files implement **6 distinct execution patterns**. By creating parameterized pipeline templates for each pattern, we can:

1. Reduce code duplication by 60-70%
2. Ensure consistent behavior across targets
3. Simplify testing and maintenance
4. Enable rapid addition of new targets via configuration

### Target Metrics

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| Native module files | 38 | 15-18 | ~55% |
| Total native LoC | 12,679 | ~5,500 | ~57% |
| Unique targets | 45 | 45 | 0% (no functionality loss) |
| Lines per target (avg) | 282 | ~122 | ~57% |

---

## Table of Contents

1. [Pattern Analysis](#1-pattern-analysis)
2. [Consolidation Architecture](#2-consolidation-architecture)
3. [Implementation Phase 1: Foundation Templates](#3-implementation-phase-1-foundation-templates)
4. [Implementation Phase 2: Analytics Domain](#4-implementation-phase-2-analytics-domain)
5. [Implementation Phase 3: Graphs Domain](#5-implementation-phase-3-graphs-domain)
6. [Implementation Phase 4: Ingestion Domain](#6-implementation-phase-4-ingestion-domain)
7. [Implementation Phase 5: Export Domain](#7-implementation-phase-5-export-domain)
8. [File Structure Changes](#8-file-structure-changes)
9. [Migration Strategy](#9-migration-strategy)
10. [Testing Strategy](#10-testing-strategy)
11. [Risk Assessment](#11-risk-assessment)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Pattern Analysis

### 1.1 Identified Execution Patterns

Through comprehensive analysis of all 38 native module files, **6 distinct execution patterns** were identified:

| Pattern | Description | Current Files | Targets | LoC |
|---------|-------------|---------------|---------|-----|
| **A: Ibis→DuckDB** | Pure Ibis compute → DuckDBIbisTableSaver | 5 | 7 | ~1,800 |
| **B: Rows→DuckDB** | Row tuples → DuckDBRowsSaver | 4 | 8 | ~1,200 |
| **C: Multi-Table Rows** | Result dataclass → multiple row extracts → multiple savers | 4 | 6 | ~2,400 |
| **D: NativeExecutor** | Result dataclass → persist internally → NativeTargetExecutor | 8 | 12 | ~3,800 |
| **E: Tool Invocation** | External tool → Result → NativeTargetExecutor | 3 | 6 | ~1,600 |
| **F: File Artifact** | Compute → FileArtifactSaver | 2 | 3 | ~600 |

### 1.2 Pattern A: Ibis→DuckDB (Pure Ibis Compute)

**Signature**: Compute returns `ir.Table`, materialized via `DuckDBIbisTableSaver`.

**Current Implementations**:
- `coverage_functions` (coverage_pipeline.py:79)
- `risk_factors` (risk_factors.py:262) ← Already uses @pipe_input
- `hotspots` (hotspots.py:239) ← Already uses @pipe_input
- `subsystems` (subsystem_targets.py:190) ← Already uses @pipe_input
- `subsystem_agreement` (subsystem_targets.py:285)
- Graph metrics targets (graph_metrics_pipeline.py)
- Export data queries (export_targets.py)

**Common Structure**:
```python
@SaveToDecorator([DuckDBIbisTableSaver], ...)
@tag(domain="...", target="...", node_type="compute")
def t__TARGET__compute(...) -> ir.Table:
    return ibis_expr

@tag(domain="...", target="...", node_type="materialize")
def t__TARGET(env, graph, m__TABLE) -> TargetRunRecord:
    return record_from_duckdb_materialization(...)
```

**Consolidation Opportunity**: HIGH - Can use @subdag with @parameterize for target/table configs.

### 1.3 Pattern B: Rows→DuckDB (Simple Row Materialization)

**Signature**: Compute returns `tuple[tuple[...], ...] | None`, materialized via `DuckDBRowsSaver`.

**Current Implementations**:
- `function_history` (history_targets.py:52)
- `history_timeseries` (history_targets.py:146)
- Several ingestion targets

**Common Structure**:
```python
@SaveToDecorator([DuckDBRowsSaver], columns=value(COLS))
@tag(domain="...", target="...", node_type="compute")
def t__TARGET__compute(env, graph) -> tuple[tuple[...], ...] | None:
    if should_skip(...): return None
    return compute_rows(...)

@tag(domain="...", target="...", node_type="materialize")
def t__TARGET(env, graph, m__TABLE) -> TargetRunRecord:
    return record_from_duckdb_materialization(...)
```

**Consolidation Opportunity**: HIGH - Near-identical structure across targets.

### 1.4 Pattern C: Multi-Table Row Materialization

**Signature**: Compute returns a Result dataclass with multiple row sets, each extracted and materialized separately.

**Current Implementations**:
- `function_metrics` (function_metrics.py) → 3 tables
- `config_data_flow` (config_graph_targets.py) → 5+ tables
- `cfg_dfg_metrics` (config_graph_targets.py) → 6 tables
- `external_deps` (dependency_targets.py) → 2+ tables
- `data_models` (metadata_targets.py) → 2 tables

**Common Structure**:
```python
@dataclass
class TargetResult:
    table1_rows: list[Row1]
    table2_rows: list[Row2]
    ...

def t__TARGET__compute(...) -> TargetResult | None:
    ...

@SaveToDecorator([DuckDBRowsSaver], ...)
def TARGET__table1_rows(compute_result) -> tuple | None:
    if compute_result is None: return None
    return tuple(...)

@SaveToDecorator([DuckDBRowsSaver], ...)
def TARGET__table2_rows(compute_result) -> tuple | None:
    ...

def t__TARGET(env, graph, m__table1, m__table2) -> TargetRunRecord:
    return record_from_duckdb_materializations(env, graph, target_name, materializations={...})
```

**Consolidation Opportunity**: MEDIUM - Can template the extract/materialize pattern, but compute logic varies.

### 1.5 Pattern D: NativeTargetExecutor Pattern

**Signature**: Compute returns a Result dataclass, materialize node uses `NativeTargetExecutor` for skip/execute.

**Current Implementations**:
- `goids` (support_targets.py:325)
- `symbol_uses` (support_targets.py:652)
- `call_graph_views` (support_targets.py:931)
- `call_graph` (call_graph.py:509)
- `import_graph` (import_graph.py:138)
- `cfg`, `dfg` (cfg_dfg.py)
- `coverage_test_edges`, `behavioral_coverage` (coverage_pipeline.py)

**Common Structure**:
```python
@dataclass
class ExtractResult:
    success: bool
    table_counts: dict[str, int]
    error: str | None = None

@tag(node_type="tool")
def t__TARGET__extract(env, upstream_deps) -> ExtractResult:
    # Compute + persist internally
    ...

@tag(node_type="materialize")
def t__TARGET(env, graph, compute_result) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "TARGET")
    if executor.should_skip(): return executor.skip()
    if not compute_result.success:
        return executor.fail(RuntimeError(compute_result.error))
    def compute(): return compute_result.table_counts
    return executor.execute(compute)
```

**Consolidation Opportunity**: HIGH - The materialize pattern is completely templatable.

### 1.6 Pattern E: Tool Invocation

**Signature**: External tool invocation with async execution.

**Current Implementations**:
- `scip` (scip.py)
- `modules` (ingest_targets.py)
- `typing` (ingest_targets.py)
- `coverage_ingest`, `tests_ingest`, `config_ingest` (ingest_targets.py)

**Common Structure**:
```python
@tag(node_type="tool")
def t__TARGET__run(env) -> ToolResult:
    # Run external tool
    ...

def t__TARGET(env, graph, tool_result) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "TARGET")
    ...
```

**Consolidation Opportunity**: MEDIUM - Tool logic varies, but executor pattern is common.

### 1.7 Pattern F: File Artifact Export

**Signature**: Compute produces data, materialized via `FileArtifactSaver`.

**Current Implementations**:
- `export_jsonl` (export_targets.py:61)
- `export_parquet` (export_targets.py:196)
- `serving_artifacts` (serving_artifacts.py)

**Common Structure**:
```python
@SaveToDecorator([FileArtifactSaver], ...)
def TARGET__content(compute_result) -> bytes | str | None:
    ...

def t__TARGET(env, graph, artifact_metadata) -> TargetRunRecord:
    return record_from_file_artifact_materialization(...)
```

**Consolidation Opportunity**: HIGH - Can parameterize export format and content generation.

---

## 2. Consolidation Architecture

### 2.1 Template Module Structure

Create shape-based template modules in `hamilton/templates/`:

```
hamilton/templates/
├── __init__.py                    # Exports all templates
├── all_targets.py                 # Existing: support node generation
├── ibis_pipeline.py               # Existing: Pattern A
├── rows_pipeline.py               # Existing: Pattern B
├── tool_pipeline.py               # Existing: Pattern F
├── multi_table_pipeline.py        # NEW: Pattern C
├── executor_pipeline.py           # NEW: Pattern D
├── parameterized/                 # NEW: @parameterize configs
│   ├── __init__.py
│   ├── analytics_ibis.py          # Pattern A analytics targets
│   ├── analytics_rows.py          # Pattern B analytics targets
│   ├── analytics_executor.py      # Pattern D analytics targets
│   ├── graphs_executor.py         # Pattern D graphs targets
│   └── export_artifacts.py        # Pattern F export targets
```

### 2.2 Core Template Design Principles

1. **Shape-First**: Templates are organized by data shape (Ibis, rows, multi-table, artifact), not domain.

2. **Config-Driven**: Target-specific parameters are provided via `@parameterize` or config dicts.

3. **Composition over Inheritance**: Templates compose with native modules via `@subdag`.

4. **Override-Friendly**: Critical targets can remain as full native modules and override templates.

### 2.3 Parameterization Strategy

Use Hamilton's `@parameterize` for single-function variations:

```python
from hamilton.function_modifiers import parameterize, source, value

SIMPLE_IBIS_TARGETS = {
    "coverage_functions": {
        "table_key": "analytics.coverage_functions",
        "compute_fn": build_coverage_functions_expr_from_tables,
        "upstream": ("q__core__goids", "q__analytics__coverage_lines"),
    },
    "subsystem_agreement": {
        "table_key": "analytics.subsystem_agreement",
        "compute_fn": compute_subsystem_agreement,
        "upstream": ("q__analytics__subsystems",),
    },
}

@parameterize(
    **{
        f"t__{name}__compute": {
            "table_key": value(cfg["table_key"]),
            "compute_fn": value(cfg["compute_fn"]),
        }
        for name, cfg in SIMPLE_IBIS_TARGETS.items()
    }
)
def _simple_ibis_compute(
    env: BuildEnv,
    table_key: str,
    compute_fn: Callable,
    **upstream_tables,
) -> ir.Table:
    """Parameterized compute for simple Ibis targets."""
    return compute_fn(env, **upstream_tables)
```

### 2.4 Subdag Strategy

Use `@subdag` for multi-node pipeline patterns:

```python
from hamilton.function_modifiers import subdag, source, value

# Define reusable pipeline functions
def _compute(env: BuildEnv, compute_fn: Callable, **inputs) -> ir.Table:
    return compute_fn(env, **inputs)

def _materialize(env: BuildEnv, graph: TargetGraph, ...) -> TargetRunRecord:
    return record_from_duckdb_materialization(...)

# Stamp out per-target
@subdag(
    _compute, _materialize,
    inputs={"compute_fn": value(build_coverage_functions_expr)},
    config={"table_key": "analytics.coverage_functions"},
)
def t__coverage_functions(env: BuildEnv, graph: TargetGraph) -> TargetRunRecord:
    ...
```

---

## 3. Implementation Phase 1: Foundation Templates

> **Status**: ✅ COMPLETE  
> **Completed**: 2025-12-17  
> **Test Coverage**: 29 tests passing

### 3.1 Deliverables Summary

| File | Status | Tests | Key Exports |
|------|--------|-------|-------------|
| `executor_pipeline.py` | ✅ Created | 6 tests | `executor_materialize`, `record`, `ComputeResult` |
| `multi_table_pipeline.py` | ✅ Created | 12 tests | `multi_table_record`, `record`, `create_row_extractor` |
| `rows_pipeline.py` | ✅ Extended | 11 tests | `row_to_tuple`, `rows_to_tuples` |
| `templates/__init__.py` | ✅ Updated | - | All new exports |

### 3.2 `executor_pipeline.py` - Actual Implementation

**File**: `src/codeintel/build/hamilton/templates/executor_pipeline.py`

**Key Difference from Plan**: The `ComputeResult` type is defined as `Any` instead of a `Protocol`. This is required because:
1. Python 3.13's `Protocol` doesn't support `issubclass()` for protocols with non-method members (data-only protocols)
2. Hamilton internally uses `issubclass()` for type matching during DAG construction
3. Using `@runtime_checkable` with data-only protocols raises `TypeError` in Python 3.13

**Actual Implementation**:

```python
from typing import Any

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

# Note: Using Any instead of Protocol for compute_result parameter because:
# 1. Python 3.13 Protocol doesn't support issubclass() for data-only protocols
# 2. Hamilton internally uses issubclass() for type matching
# The expected interface is:
#   - success: bool
#   - table_counts: dict[str, int]
#   - error: str | None
ComputeResult = Any


@tag(node_type="materialize")
def executor_materialize(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ComputeResult,
) -> TargetRunRecord:
    """Materialize using NativeTargetExecutor pattern."""
    executor = NativeTargetExecutor.for_target(env, graph, target_name)

    if executor.should_skip():
        return executor.skip()

    if not compute_result.success:
        error_msg = compute_result.error or f"{target_name} computation failed"
        return executor.fail(RuntimeError(error_msg))

    def compute() -> dict[str, int]:
        return dict(compute_result.table_counts)

    return executor.execute(compute)


@tag(node_type="materialize")
def record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ComputeResult,
) -> TargetRunRecord:
    """Alias for executor_materialize for subDAG composition."""
    return executor_materialize(env, graph, target_name, compute_result)
```

### 3.3 `multi_table_pipeline.py` - Actual Implementation

**File**: `src/codeintel/build/hamilton/templates/multi_table_pipeline.py`

**Key Additions Beyond Plan**:
- Added `create_row_extractor()` factory function (moved here from rows_pipeline)
- Added `record()` alias for subDAG composition
- Types imported at runtime (not in TYPE_CHECKING) for Hamilton compatibility

**Actual Exports**:
- `multi_table_record()` - Combine multiple materializations into single TargetRunRecord
- `record()` - Alias for subDAG composition
- `create_row_extractor()` - Factory for row extraction from Result dataclasses

### 3.4 `rows_pipeline.py` - Row Conversion Helpers

**File**: `src/codeintel/build/hamilton/templates/rows_pipeline.py`

**Added Functions**:

```python
def row_to_tuple(row: Mapping[str, object], columns: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.
    
    Missing columns produce None values.
    """
    return tuple(row.get(col) for col in columns)


def rows_to_tuples(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    """Convert a sequence of mapping rows to a tuple of tuples in column order."""
    return tuple(row_to_tuple(row, columns) for row in rows)
```

### 3.5 Updated `templates/__init__.py` Exports

```python
__all__ = [
    "ComputeResult",
    "ModuleType",
    "create_row_extractor",
    "executor_materialize",
    "executor_record",
    "get_template_module",
    "multi_table_record",
    "row_to_tuple",
    "rows_record",
    "rows_to_save",
    "rows_to_tuples",
    "tool_output_to_save",
    "tool_record",
]
```

### 3.6 Critical Learnings for Future Phases

#### 3.6.1 Hamilton Type Resolution Requirements

**CRITICAL**: When using templates with `@subdag`, Hamilton must resolve type hints at runtime using `typing.get_type_hints()`. This means:

1. **Types MUST be imported at runtime**, not inside `if TYPE_CHECKING:` blocks
2. Types like `BuildEnv`, `TargetGraph`, `TargetRunRecord` must be regular imports
3. This increases import time but is unavoidable for `@subdag` usage

**Pattern to Follow**:
```python
# ✅ CORRECT - Types available at runtime for Hamilton
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.targets import TargetGraph

# ❌ WRONG - Types not available at runtime
if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv  # Hamilton can't resolve!
```

#### 3.6.2 Protocol Limitations with Hamilton

Python 3.13's stricter `Protocol` implementation doesn't support `issubclass()` for protocols with non-method members. Since Hamilton uses `issubclass()` internally for type matching, we must:

1. Use `Any` type alias for duck-typed result parameters
2. Document the expected interface in comments
3. Rely on structural typing at runtime (duck typing)

**Recommended Pattern for Result Types**:
```python
# Define expected interface in docstring/comment
# Expected interface:
#   - success: bool
#   - table_counts: dict[str, int]  
#   - error: str | None
ComputeResult = Any

def my_function(compute_result: ComputeResult) -> TargetRunRecord:
    # Duck typing works at runtime
    if not compute_result.success:
        ...
```

#### 3.6.3 Test Patterns for Templates

Effective test patterns discovered:

1. **Direct function tests**: Test `executor_materialize()` directly with mock results
2. **Subdag integration tests**: Build ephemeral Hamilton modules using `@subdag` and execute via driver
3. **Type narrowing without assert**: Use `if x is None: return` instead of `assert x is not None` (S101 lint rule)

**Example Test Module Pattern**:
```python
def _build_subdag_module(compute_result: MockComputeResult) -> ModuleType:
    """Build an ephemeral Hamilton module using executor_pipeline via @subdag."""
    mod = ModuleType("tests.build.hamilton._executor_pipeline_case")
    sys.modules[mod.__name__] = mod
    
    captured_result = compute_result  # Capture in closure
    
    @tag(domain="graphs", target="goids", node_type="tool")
    def t__goids__extract(env: BuildEnv) -> MockComputeResult:
        _ = env  # Use env to satisfy Hamilton
        return captured_result
    
    @subdag(
        executor_pipeline,
        inputs={
            "env": source("env"),
            "graph": source("graph"),
            "target_name": value("goids"),
            "compute_result": source("t__goids__extract"),
        },
    )
    def t__goids(record: TargetRunRecord) -> TargetRunRecord:
        return record
    
    # Set module ownership for Hamilton discovery
    t__goids__extract.__module__ = mod.__name__
    t__goids.__module__ = mod.__name__
    
    # Attach to module namespace
    mod.t__goids__extract = t__goids__extract
    mod.t__goids = t__goids
    return mod
```

---

## 4. Implementation Phase 2: Analytics Domain

> **Status**: ✅ COMPLETE  
> **Completed**: 2025-12-17  
> **Test Coverage**: 593 tests passing (20 new tests for Phase 2)

### 4.0 Phase 2 Completion Summary

#### 4.0.0 Deliverables

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/codeintel/build/hamilton/native/analytics/coverage_targets.py` | ✅ Created | 369 | Consolidates coverage_functions, coverage_test_edges, behavioral_coverage |
| `src/codeintel/build/hamilton/native/analytics/metrics_targets.py` | ✅ Created | 384 | Consolidates function_history, history_timeseries, subsystem_agreement, subsystem/symbol/test graph metrics |
| `tests/build/hamilton/test_coverage_targets.py` | ✅ Created | 278 | 10 tests for coverage targets |
| `tests/build/hamilton/test_metrics_targets.py` | ✅ Created | 384 | 10 tests for metrics targets |

**Files Deleted**:
- `history_targets.py` (193 lines) → consolidated into `metrics_targets.py`
- `coverage_pipeline.py` (329 lines) → consolidated into `coverage_targets.py`
- `graph_metrics_pipeline.py` (473 lines) → consolidated into `metrics_targets.py`

**Files Modified**:
- `subsystem_targets.py` - Removed `subsystem_agreement` target (~90 lines)
- `registry.py` - Updated module paths
- `analytics/__init__.py` - Updated imports/exports

**Net Line Reduction**: ~995 lines deleted

#### 4.0.1 Phase 2 Learnings for Future Phases

**Critical Discoveries**:

1. **`@tag` with `target_=` parameter is REQUIRED when combined with `@SaveToDecorator`**:
   
   When using `@SaveToDecorator`, Hamilton creates additional nodes. Without the `target_=` parameter, tags may not be applied to the intended node:
   
   ```python
   # ❌ WRONG - tag may not apply to the correct node
   @SaveToDecorator([DuckDBIbisTableSaver], ...)
   @tag(domain="analytics", target="coverage_functions", node_type="compute")
   def t__coverage_functions__compute(...): ...
   
   # ✅ CORRECT - explicitly target the node
   @SaveToDecorator([DuckDBIbisTableSaver], ...)
   @tag(
       domain="analytics",
       target="coverage_functions", 
       node_type="compute",
       target_="t__coverage_functions__compute",  # REQUIRED!
   )
   def t__coverage_functions__compute(...): ...
   ```
   
   This was discovered via the `test_pr64_all_nodes_have_node_type_tag` test failure.

2. **Test fixture naming: Use `fake_gateway`, not `memory_gateway`**:
   
   The shared test fixtures are in `tests/build/hamilton/conftest.py`. Key fixtures:
   - `fake_gateway` - In-memory gateway with `FakeBuildAccessor`
   - `fresh_gateway` - Real DuckDB gateway for integration tests
   - `minimal_target_graph` / `diamond_target_graph` - Test graph fixtures

3. **`FakeBuildAccessor` now has `save_manifest()` method**:
   
   Added during Phase 2 to support `NativeTargetExecutor.execute()` which calls `save_manifest()`.

4. **Ibis types (`ir.Table`) require runtime imports for Hamilton**:
   
   Already known from Phase 1, but reinforced: `import ibis.expr.types as ir` must be at module level, not in `TYPE_CHECKING` block.

5. **Pattern D (Executor) targets typically have 3 components**:
   - Result dataclass with `success`, `table_counts`, `error` fields
   - Compute function returning the result dataclass
   - Materialize function using `executor_materialize()`

6. **Multi-table targets with Ibis use `@SaveToDecorator` directly**:
   
   For Ibis targets that write to a single table, use `@SaveToDecorator` on the compute function and `ibis_pipeline.record` for materialization.

7. **Test file organization pattern**:
   
   Tests should be organized by consolidated module:
   - `test_coverage_targets.py` - Tests for all targets in `coverage_targets.py`
   - `test_metrics_targets.py` - Tests for all targets in `metrics_targets.py`
   
   Each test file should have:
   - Result dataclass construction tests
   - Direct materialize function tests (success and failure)
   - Named constants for magic numbers (e.g., `MAX_COVERAGE_FUNCTIONS_COUNT = 25`)

### 4.0.2 Original Phase 2 Implementation Notes (Based on Phase 1 Learnings)

#### Critical Technical Requirements

Based on Phase 1 implementation, the following requirements apply to all Phase 2 work:

1. **Runtime Type Imports**: All types used in function signatures that will be wired via `@subdag` must be imported at runtime (not in `TYPE_CHECKING` blocks). This includes:
   - `BuildEnv`
   - `TargetGraph` 
   - `TargetRunRecord`

2. **Avoid Protocol for Result Types**: Use `Any` type alias with documented interface for compute result parameters. Hamilton's internal `issubclass()` checks don't work with Python 3.13's stricter Protocol implementation.

3. **Test Structure**: Each consolidated target should have:
   - Direct function call tests (fast, isolated)
   - `@subdag` integration test (validates Hamilton wiring)
   - Output parity test (if migrating existing target)

#### 4.0.2 Recommended Implementation Order for Phase 2

Start with the simplest targets that have minimal dependencies:

1. **Wave 2.1 - Simple Row Targets** (Day 1):
   - `function_history` → Use `rows_pipeline` + `row_to_tuple`
   - `history_timeseries` → Use `rows_pipeline` + `row_to_tuple`
   
2. **Wave 2.2 - Simple Ibis Targets** (Day 1-2):
   - `subsystem_agreement` → Use `ibis_pipeline` (already templated)
   - `coverage_functions` → Use `ibis_pipeline`

3. **Wave 2.3 - Executor Targets** (Day 2-3):
   - Graph metrics (3 targets) → Use `executor_pipeline`
   - `coverage_test_edges` → Use `executor_pipeline`
   - `behavioral_coverage` → Use `executor_pipeline`

4. **Wave 2.4 - Multi-Table Targets** (Day 3-4):
   - `data_models` → Use `multi_table_pipeline` + `create_row_extractor`
   - `function_metrics` → Use `multi_table_pipeline` (3 tables)

#### 4.0.3 Template Usage Patterns for Phase 2

**Pattern for Simple Row Targets** (history_targets.py consolidation):
```python
from codeintel.build.hamilton.templates.rows_pipeline import (
    row_to_tuple,
    rows_to_tuples,
)

# Use row helpers to convert from dict rows to tuple format
def compute_rows(env: BuildEnv) -> tuple[tuple[object, ...], ...] | None:
    raw_rows = fetch_history_data(env.gateway, env.snapshot)
    if not raw_rows:
        return None
    return rows_to_tuples(raw_rows, HISTORY_COLUMNS)
```

**Pattern for Executor Targets** (coverage_pipeline.py consolidation):
```python
from codeintel.build.hamilton.templates import executor_materialize

# Keep existing compute/extract node
@tag(node_type="tool")
def t__coverage_test_edges__extract(env: BuildEnv, ...) -> ExtractResult:
    # Existing complex logic unchanged
    ...

# Use template for materialize
@tag(node_type="materialize")
def t__coverage_test_edges(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_test_edges__extract: ExtractResult,
) -> TargetRunRecord:
    return executor_materialize(env, graph, "coverage_test_edges", t__coverage_test_edges__extract)
```

**Pattern for Multi-Table Targets** (function_metrics.py consolidation):
```python
from codeintel.build.hamilton.templates.multi_table_pipeline import (
    create_row_extractor,
    multi_table_record,
)

# Create extractors for each table
extract_metrics = create_row_extractor(
    "metrics_rows",
    columns=METRICS_COLUMNS,
    row_converter=lambda r: tuple(r.values()),
)
extract_types = create_row_extractor("types_rows", columns=TYPES_COLUMNS)

# Use multi_table_record for final materialize
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_metrics: dict,
    m__analytics__function_types: dict,
    m__analytics__function_validation: dict,
) -> TargetRunRecord:
    return multi_table_record(
        env, graph, "function_metrics",
        {
            "analytics.function_metrics": m__analytics__function_metrics,
            "analytics.function_types": m__analytics__function_types,
            "analytics.function_validation": m__analytics__function_validation,
        },
    )
```

### 4.1 Analytics Consolidation Summary

Current analytics files (12 files, ~4,800 LoC):
- `classification_targets.py` (435 lines, 2 targets)
- `config_graph_targets.py` (769 lines, 2 targets)
- `coverage_pipeline.py` (329 lines, 3 targets)
- `dependency_targets.py` (545 lines, 2 targets)
- `function_detail_targets.py` (471 lines, 2 targets)
- `function_metrics.py` (347 lines, 1 target)
- `graph_metrics_pipeline.py` (473 lines, 3 targets)
- `history_targets.py` (193 lines, 2 targets)
- `hotspots.py` (313 lines, 1 target)
- `metadata_targets.py` (561 lines, 4 targets)
- `risk_factors.py` (335 lines, 1 target)
- `subsystem_targets.py` (364 lines, 2 targets)

Target analytics files (7-8 files, ~2,200 LoC):
- `classification_targets.py` (KEEP - complex compute logic)
- `config_graph_targets.py` (CONSOLIDATE - Pattern C)
- `coverage_targets.py` (NEW - consolidate coverage_pipeline.py)
- `dependency_targets.py` (KEEP - unique external deps logic)
- `function_analytics.py` (NEW - consolidate function_metrics + function_detail)
- `graph_metrics_targets.py` (CONSOLIDATE - graph_metrics_pipeline.py)
- `metrics_targets.py` (NEW - consolidate history, hotspots, risk_factors, subsystems)
- `metadata_targets.py` (SIMPLIFY - use templates)

### 4.2 Create `analytics/metrics_targets.py`

Consolidate simple metrics targets using @parameterize:

```python
"""Consolidated metrics analytics targets.

This module uses @parameterize to consolidate targets with similar patterns:
- risk_factors (Pattern A with @pipe_input - KEEP SEPARATE)
- hotspots (Pattern A with @pipe_input - KEEP SEPARATE)
- subsystems (Pattern A with @pipe_input - KEEP SEPARATE)
- subsystem_agreement (Pattern A - CAN CONSOLIDATE)
- function_history (Pattern B)
- history_timeseries (Pattern B)
- graph_metrics (subsystem, symbol, test) (Pattern D)

Targets with @pipe_input are kept in their own files due to complexity.
"""

from __future__ import annotations

from typing import Any

from hamilton.function_modifiers import parameterize, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
from codeintel.analytics.history.history_timeseries import HISTORY_TIMESERIES_COLS
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

# Configuration for simple row-based targets
SIMPLE_ROWS_TARGETS = {
    "function_history": {
        "table_key": "analytics.function_history",
        "columns": FUNCTION_HISTORY_COLS,
        "compute_fn": build_function_history_rows,
    },
    "history_timeseries": {
        "table_key": "analytics.history_timeseries",
        "columns": HISTORY_TIMESERIES_COLS,
        "compute_fn": lambda gw, snap: (),  # Stubbed
    },
}


def _skip_check(env: BuildEnv, graph: TargetGraph, target_name: str) -> bool:
    """Check if target should be skipped based on manifest."""
    target = graph.get(target_name)
    if target is None:
        return False
    input_hash = compute_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        options_hash=None,
        manifests=env.manifest_index,
    )
    return should_skip_native_target(env, target, input_hash)


# Generate compute nodes via @parameterize
@parameterize(
    **{
        f"t__{name}__compute": {
            "target_name": value(name),
            "table_key": value(cfg["table_key"]),
            "columns": value(cfg["columns"]),
            "compute_fn": value(cfg["compute_fn"]),
        }
        for name, cfg in SIMPLE_ROWS_TARGETS.items()
    }
)
@tag(node_type="compute")
def _simple_rows_compute(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    table_key: str,
    columns: tuple[str, ...],
    compute_fn: Any,
) -> tuple[tuple[object, ...], ...] | None:
    """Parameterized compute for simple row targets."""
    if _skip_check(env, graph, target_name):
        return None
    return compute_fn(env.gateway, env.snapshot)


# Generate SaveToDecorator + materialize nodes
# ... (continue with materialize pattern)
```

### 4.3 Targets to Keep Separate (Complex Logic)

These targets have complex compute logic that doesn't fit simple templates:

| Target | File | Reason |
|--------|------|--------|
| `risk_factors` | risk_factors.py | Uses @pipe_input with 5 transformation steps |
| `hotspots` | hotspots.py | Uses @pipe_input with complex scoring |
| `subsystems` | subsystem_targets.py | Uses @pipe_input with clustering |
| `semantic_roles` | classification_targets.py | Complex classification logic |
| `test_profile` | classification_targets.py | Multi-step test analysis |
| `function_contracts` | function_detail_targets.py | AST-based contract extraction |
| `external_deps` | dependency_targets.py | Complex call graph analysis |

**Action**: Keep these in their original files but extract common materialize patterns.

### 4.4 Targets to Consolidate via Templates

| Target | Current File | Template | Config |
|--------|-------------|----------|--------|
| `function_history` | history_targets.py | rows_pipeline | Simple |
| `history_timeseries` | history_targets.py | rows_pipeline | Simple |
| `coverage_functions` | coverage_pipeline.py | ibis_pipeline | Simple |
| `subsystem_agreement` | subsystem_targets.py | ibis_pipeline | Simple |
| `subsystem_graph_metrics` | graph_metrics_pipeline.py | executor_pipeline | Simple |
| `symbol_graph_metrics` | graph_metrics_pipeline.py | executor_pipeline | Simple |
| `test_graph_metrics` | graph_metrics_pipeline.py | executor_pipeline | Simple |
| `coverage_test_edges` | coverage_pipeline.py | executor_pipeline | Medium |
| `behavioral_coverage` | coverage_pipeline.py | executor_pipeline | Medium |
| `data_models` | metadata_targets.py | multi_table_pipeline | Medium |
| `data_model_usage` | metadata_targets.py | executor_pipeline | Simple |
| `function_ast_features` | metadata_targets.py | rows_pipeline | Simple |
| `profiles` | metadata_targets.py | executor_pipeline | Simple |
| `config_data_flow` | config_graph_targets.py | multi_table_pipeline | Complex |
| `cfg_dfg_metrics` | config_graph_targets.py | multi_table_pipeline | Complex |
| `entrypoints` | dependency_targets.py | executor_pipeline | Medium |

---

## 5. Implementation Phase 3: Graphs Domain

> **Status**: 🔜 Ready to Implement  
> **Prerequisites**: Phase 2 complete ✓  
> **Estimated Effort**: 2-3 days

### 5.0 Phase 3 Learnings from Phase 2

Based on Phase 2 experience, these patterns apply to graphs domain consolidation:

1. **Pattern D targets (NativeExecutor) can use `executor_materialize` directly**:
   - Create a Result dataclass with `success`, `table_counts`, `error` fields
   - Compute function returns the Result dataclass
   - Materialize function calls `executor_materialize(env, graph, target_name, compute_result)`

2. **Required `@tag` parameters for decorator stacking**:
   - Always include `target_="function_name"` when combining `@SaveToDecorator` with `@tag`
   - Use `node_type="tool"` for compute nodes that do internal persistence
   - Use `node_type="materialize"` for final record nodes

3. **Test file structure recommendation**:
   - Create `test_graph_targets.py` for consolidated targets
   - Use `fake_gateway` fixture for unit tests
   - Use `fresh_gateway` fixture for integration tests

### 5.1 Graphs Consolidation Summary

**Current graphs files** (5 files, ~3,200 LoC):
| File | Lines | Targets | Pattern |
|------|-------|---------|---------|
| `call_graph.py` | 656 | 1 (call_graph) | D |
| `cfg_dfg.py` | 583 | 2 (cfg, dfg) | D |
| `import_graph.py` | 288 | 1 (import_graph) | D |
| `metrics_targets.py` | 474 | 2 (graph_metrics, graph_validation) | D |
| `support_targets.py` | 968 | 3 (goids, symbol_uses, call_graph_views) | D |

**Target graphs files** (3-4 files, ~1,800 LoC):
- `call_graph.py` (SIMPLIFY - use `executor_materialize` for materialize node)
- `cfg_dfg.py` (SIMPLIFY - use `executor_materialize` for both materialize nodes)
- `import_graph.py` (SIMPLIFY - use `executor_materialize`)
- `graph_targets.py` (NEW - consolidate support_targets.py + metrics_targets.py)

**Estimated Reduction**: ~1,400 lines (~44%)

### 5.2 Detailed Consolidation Plan

#### 5.2.1 Wave 3.1: Consolidate support_targets.py + metrics_targets.py → graph_targets.py

The 5 targets in `support_targets.py` and `metrics_targets.py` all use Pattern D:

| Target | Result Dataclass | Compute Node | Tables |
|--------|-----------------|--------------|--------|
| `goids` | `GoidsExtractResult` | `t__goids__extract` | `core.goids` |
| `symbol_uses` | `SymbolUsesExtractResult` | `t__symbol_uses__extract` | `graph.symbol_uses` |
| `call_graph_views` | `CallGraphViewsResult` | `t__call_graph_views__extract` | `graph.call_graph_*` |
| `graph_metrics` | `GraphMetricsResult` | `t__graph_metrics__compute` | `analytics.graph_metrics` |
| `graph_validation` | `GraphValidationResult` | `t__graph_validation__check` | `analytics.graph_validation` |

**Implementation Pattern** (based on Phase 2 success):

```python
"""Consolidated graph domain targets.

Consolidates:
- goids (Pattern D)
- symbol_uses (Pattern D)
- call_graph_views (Pattern D)
- graph_metrics (Pattern D)
- graph_validation (Pattern D)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hamilton.function_modifiers import source, tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)


# Result dataclasses for each target (keep existing structure)
@dataclass(frozen=True)
class GoidsExtractResult:
    """Result from GOID extraction."""
    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ... other Result dataclasses ...


# Compute nodes (move from support_targets.py with minimal changes)
@tag(domain="graphs", target="goids", node_type="tool")
def t__goids__extract(env: BuildEnv, t__scip: TargetRunRecord) -> GoidsExtractResult:
    """Extract GOIDs from SCIP index."""
    # Existing compute logic unchanged
    ...


# Materialize nodes using executor_materialize
@tag(domain="graphs", target="goids", node_type="materialize")
def t__goids(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids__extract: GoidsExtractResult,
) -> TargetRunRecord:
    """Materialize goids target."""
    return executor_materialize(env, graph, "goids", t__goids__extract)


# Repeat for other 4 targets...
```

#### 5.2.2 Wave 3.2: Simplify call_graph.py, cfg_dfg.py, import_graph.py

These files have complex compute logic but can use `executor_materialize` for their materialize nodes:

**call_graph.py changes**:
```python
# BEFORE (full custom materialize)
@tag(domain="graphs", target="call_graph", node_type="materialize")
def t__call_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph__compute: CallGraphResult,
) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "call_graph")
    if executor.should_skip():
        return executor.skip()
    if not t__call_graph__compute.success:
        return executor.fail(RuntimeError(t__call_graph__compute.error))
    def compute():
        return t__call_graph__compute.table_counts
    return executor.execute(compute)

# AFTER (use template)
@tag(domain="graphs", target="call_graph", node_type="materialize")
def t__call_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph__compute: CallGraphResult,
) -> TargetRunRecord:
    return executor_materialize(env, graph, "call_graph", t__call_graph__compute)
```

**Estimated line reduction per file**: ~30-50 lines

### 5.3 Files to Delete After Phase 3

| File | Lines | Reason |
|------|-------|--------|
| `support_targets.py` | 968 | Consolidated into `graph_targets.py` |
| `metrics_targets.py` (graphs) | 474 | Consolidated into `graph_targets.py` |

### 5.4 Testing Strategy for Phase 3

Create `tests/build/hamilton/test_graph_targets.py`:

```python
"""Tests for consolidated graph_targets.py."""

from codeintel.build.hamilton.native.graphs.graph_targets import (
    GoidsExtractResult,
    SymbolUsesExtractResult,
    t__goids,
    t__symbol_uses,
)

# Test constants to avoid magic numbers
MAX_GOIDS_COUNT = 50
MAX_SYMBOL_USES_COUNT = 100


def test_goids_result_success() -> None:
    """Verify GoidsExtractResult success case."""
    result = GoidsExtractResult(
        success=True,
        table_counts={"core.goids": 10},
    )
    expect_true(result.success)
    expect_equal(result.table_counts.get("core.goids"), expected=10)


def test_goids_materialize_success(fake_gateway, tmp_path) -> None:
    """Verify t__goids returns success record."""
    # ... test implementation ...
```

---

## 6. Implementation Phase 4: Ingestion Domain

> **Status**: ⏳ Pending  
> **Prerequisites**: Phase 3 complete  
> **Estimated Effort**: 1-2 days

### 6.0 Phase 4 Strategy (Based on Phase 2-3 Learnings)

The ingestion domain is already well-structured. Focus on:

1. **Simplify materialize nodes** using `executor_materialize` where applicable
2. **Avoid over-consolidation** - ingestion targets have unique scan/persist logic
3. **Preserve `@cache` patterns** in `extraction_targets.py`

### 6.1 Ingestion Consolidation Summary

**Current ingestion files** (3 files, ~1,340 LoC):
| File | Lines | Targets | Pattern | Action |
|------|-------|---------|---------|--------|
| `extraction_targets.py` | 318 | 3 (ast, cst, docstrings) | E (cached) | KEEP - already consolidated |
| `ingest_targets.py` | 811 | 6 (modules, config, coverage, tests, typing, goids_ingest) | E | SIMPLIFY materialize |
| `scip.py` | 214 | 1 (scip) | E | KEEP - unique tool |

**Target ingestion files** (same 3 files, ~600-700 LoC):
- `extraction_targets.py` (KEEP AS-IS - well-structured with `@cache`)
- `ingest_targets.py` (SIMPLIFY - use `executor_materialize` for materialize nodes)
- `scip.py` (KEEP AS-IS - unique async tool execution)

**Estimated Reduction**: ~200-300 lines (~20%)

### 6.2 Ingestion Already Well-Structured

`extraction_targets.py` already uses `@cache` for pure-Python operations:

```python
# Already optimal - DO NOT CHANGE
@cache(cache_key=repo_cache_key)
@tag(domain="ingestion", target="ast", node_type="cached")
def t__ast__cached(env: BuildEnv) -> ASTExtractionResult:
    """Cached AST extraction."""
    ...
```

**Key insight**: The `@cache` decorator handles skip logic automatically. These targets don't need `executor_materialize`.

### 6.3 Simplify `ingest_targets.py` Materialize Nodes

The 6 targets in `ingest_targets.py` follow Pattern E (Tool Invocation). Each target has:
1. A `_scan` or `_ingest` function (compute)
2. A `_persist` function (internal persistence)
3. A `t__TARGET` materialize node

**Current structure** (~120-150 lines per target):
```python
def t__modules__scan(env: BuildEnv) -> ModuleScanResult:
    """Scan repository for modules."""
    # Complex scan logic
    ...

def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ModuleScanResult,
) -> TargetRunRecord:
    """Materialize modules target."""
    executor = NativeTargetExecutor.for_target(env, graph, "modules")
    if executor.should_skip():
        return executor.skip()
    if not t__modules__scan.success:
        return executor.fail(RuntimeError(t__modules__scan.error))
    def compute():
        return t__modules__scan.table_counts
    return executor.execute(compute)
```

**Simplified with `executor_materialize`** (~90-110 lines per target):
```python
def t__modules__scan(env: BuildEnv) -> ModuleScanResult:
    """Scan repository for modules."""
    # Complex scan logic unchanged
    ...

@tag(domain="ingestion", target="modules", node_type="materialize")
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ModuleScanResult,
) -> TargetRunRecord:
    """Materialize modules target."""
    return executor_materialize(env, graph, "modules", t__modules__scan)
```

**Estimated line reduction**: ~30-40 lines per target × 6 targets = ~180-240 lines

### 6.4 Targets to Simplify in Phase 4

| Target | Current Lines | After Simplification | Reduction |
|--------|---------------|----------------------|-----------|
| `modules` | ~150 | ~110 | ~40 |
| `config_ingest` | ~120 | ~90 | ~30 |
| `coverage_ingest` | ~100 | ~70 | ~30 |
| `tests_ingest` | ~130 | ~100 | ~30 |
| `typing` | ~100 | ~70 | ~30 |
| `goids_ingest` | ~100 | ~70 | ~30 |
| **Total** | ~800 | ~510 | ~290 |

### 6.5 Testing Strategy for Phase 4

No new test file needed - existing tests should continue to pass after simplification.

**Validation checklist**:
- [ ] All existing `test_ingest_targets.py` tests pass
- [ ] No DAG structure changes (same node names)
- [ ] Manifest hashes unchanged

---

## 7. Implementation Phase 5: Export Domain

> **Status**: ⏳ Pending  
> **Prerequisites**: Phase 4 complete  
> **Estimated Effort**: 0.5-1 day

### 7.0 Phase 5 Strategy (Based on Phase 2-4 Learnings)

The export domain is small and already uses Pattern F (File Artifact). Focus on:

1. **Use `@parameterize` for format variations** (JSONL vs Parquet)
2. **Keep `serving_artifacts.py` separate** - unique deployment logic
3. **Use `tool_pipeline` template** from existing templates

### 7.1 Export Consolidation Summary

**Current export files** (2 files, ~500 LoC):
| File | Lines | Targets | Pattern | Action |
|------|-------|---------|---------|--------|
| `export_targets.py` | 283 | 2 (export_jsonl, export_parquet) | F | PARAMETERIZE |
| `serving_artifacts.py` | 216 | 1 (serving_artifacts) | F | KEEP |

**Target export files** (same 2 files, ~350 LoC):
- `export_targets.py` (SIMPLIFY - parameterize JSONL/Parquet)
- `serving_artifacts.py` (KEEP AS-IS - unique serving artifact logic)

**Estimated Reduction**: ~150 lines (~30%)

### 7.2 Analysis of export_targets.py

The current `export_targets.py` has two very similar targets:

| Target | Format | Content Function | Artifact Path |
|--------|--------|------------------|---------------|
| `export_jsonl` | JSONL | `build_jsonl_export_content()` | `{snapshot}/exports/export.jsonl` |
| `export_parquet` | Parquet | `build_parquet_export_content()` | `{snapshot}/exports/export.parquet` |

Both follow the same structure:
1. Query tables via Ibis
2. Build content in format-specific way
3. Save via `FileArtifactSaver`
4. Return `TargetRunRecord`

### 7.3 Parameterized Export Implementation

```python
"""Export targets using @parameterize for format variations."""

from __future__ import annotations

from hamilton.function_modifiers import parameterize, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.targets import TargetGraph

# Configuration for export formats
EXPORT_FORMATS = {
    "export_jsonl": {
        "format": "jsonl",
        "content_fn": "build_jsonl_export_content",  # Use string to avoid import
        "artifact_name": "export.jsonl",
    },
    "export_parquet": {
        "format": "parquet", 
        "content_fn": "build_parquet_export_content",
        "artifact_name": "export.parquet",
    },
}


def _get_content_fn(format_type: str):
    """Get content builder function for format type."""
    from codeintel.build.exports.content import (
        build_jsonl_export_content,
        build_parquet_export_content,
    )
    return {
        "jsonl": build_jsonl_export_content,
        "parquet": build_parquet_export_content,
    }[format_type]


@parameterize(
    **{
        f"t__{name}__compute": {
            "target_name": value(name),
            "format_type": value(cfg["format"]),
            "artifact_name": value(cfg["artifact_name"]),
        }
        for name, cfg in EXPORT_FORMATS.items()
    }
)
@tag(domain="export", node_type="compute")
def _export_compute(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    format_type: str,
    artifact_name: str,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> bytes | None:
    """Parameterized compute for export targets."""
    # Skip check via NativeTargetExecutor
    from codeintel.build.hamilton.native.executor import NativeTargetExecutor
    executor = NativeTargetExecutor.for_target(env, graph, target_name)
    if executor.should_skip():
        return None
    
    # Build content using format-specific function
    content_fn = _get_content_fn(format_type)
    return content_fn(env.gateway, env.snapshot)


@parameterize(
    **{
        f"t__{name}": {
            "target_name": value(name),
            "artifact_name": value(cfg["artifact_name"]),
        }
        for name, cfg in EXPORT_FORMATS.items()
    }
)
@tag(domain="export", node_type="materialize")
def _export_materialize(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    artifact_name: str,
    content: bytes | None,
    materialization: dict,
) -> TargetRunRecord:
    """Parameterized materialize for export targets."""
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name=target_name,
        artifact_name=artifact_name,
        materialization=materialization,
    )
```

### 7.4 Files to Keep Separate

| File | Reason |
|------|--------|
| `serving_artifacts.py` | Unique deployment logic, FAISS index handling |

### 7.5 Testing Strategy for Phase 5

**Validation checklist**:
- [ ] Existing export tests pass
- [ ] Both JSONL and Parquet exports produce identical content
- [ ] Artifact paths unchanged
- [ ] Skip logic works correctly

### 7.6 Alternative: Minimal Changes

If `@parameterize` introduces complexity, an alternative is to simply:
1. Extract common logic into shared helper functions
2. Keep two separate targets but with less code duplication

```python
def _build_export_content(env: BuildEnv, format_type: str) -> bytes:
    """Shared export content builder."""
    ...

def t__export_jsonl__compute(env: BuildEnv) -> bytes | None:
    return _build_export_content(env, "jsonl")

def t__export_parquet__compute(env: BuildEnv) -> bytes | None:
    return _build_export_content(env, "parquet")
```

This approach is simpler but achieves less consolidation (~50 lines vs ~150 lines).
def _export_compute(
    env: BuildEnv,
    graph: TargetGraph,
    format_name: str,
    format_type: str,
    content_fn: Callable,
    q__core__modules: ir.Table,
    q__analytics__function_metrics: ir.Table,
) -> ExportComputeResult | None:
    """Parameterized compute for export targets."""
    ...
```

---

## 8. File Structure Changes

### 8.1 Before (38 files)

```
hamilton/native/
├── __init__.py
├── executor.py
├── ibis_helpers.py
├── materialization_records.py
├── options/
│   ├── __init__.py
│   ├── graphs.py
│   └── ingestion.py
├── outputs.py
├── registry.py
├── runner.py
├── tools/
│   ├── __init__.py
│   └── executor.py
├── analytics/
│   ├── __init__.py
│   ├── classification_targets.py      # 435 lines
│   ├── config_graph_targets.py        # 769 lines
│   ├── coverage_pipeline.py           # 329 lines
│   ├── dependency_targets.py          # 545 lines
│   ├── function_detail_targets.py     # 471 lines
│   ├── function_metrics.py            # 347 lines
│   ├── graph_metrics_pipeline.py      # 473 lines
│   ├── history_targets.py             # 193 lines
│   ├── hotspots.py                    # 313 lines
│   ├── metadata_targets.py            # 561 lines
│   ├── risk_factors.py                # 335 lines
│   └── subsystem_targets.py           # 364 lines
├── export/
│   ├── __init__.py
│   ├── export_targets.py              # 283 lines
│   └── serving_artifacts.py           # 216 lines
├── graphs/
│   ├── __init__.py
│   ├── call_graph.py                  # 656 lines
│   ├── cfg_dfg.py                     # 583 lines
│   ├── import_graph.py                # 288 lines
│   ├── metrics_targets.py             # 474 lines
│   └── support_targets.py             # 968 lines
└── ingestion/
    ├── __init__.py
    ├── extraction_targets.py          # 318 lines
    ├── ingest_targets.py              # 811 lines
    └── scip.py                        # 214 lines
```

### 8.2 After (15-18 files)

```
hamilton/native/
├── __init__.py
├── executor.py                        # KEEP
├── ibis_helpers.py                    # KEEP
├── materialization_records.py         # KEEP
├── options/                           # KEEP
├── outputs.py                         # KEEP
├── registry.py                        # UPDATE
├── runner.py                          # KEEP
├── tools/                             # KEEP
├── analytics/
│   ├── __init__.py
│   ├── classification_targets.py      # KEEP (semantic_roles, test_profile)
│   ├── config_targets.py              # NEW: consolidate config_graph
│   ├── coverage_targets.py            # NEW: consolidate coverage_pipeline
│   ├── dependency_targets.py          # SIMPLIFY (keep external_deps, entrypoints)
│   ├── function_targets.py            # NEW: consolidate function_metrics + detail
│   ├── hotspots.py                    # KEEP (@pipe_input complexity)
│   ├── metadata_targets.py            # SIMPLIFY
│   ├── metrics_targets.py             # NEW: consolidate history + graph_metrics
│   ├── risk_factors.py                # KEEP (@pipe_input complexity)
│   └── subsystem_targets.py           # KEEP (@pipe_input complexity)
├── export/
│   ├── __init__.py
│   ├── export_targets.py              # SIMPLIFY with @parameterize
│   └── serving_artifacts.py           # KEEP
├── graphs/
│   ├── __init__.py
│   ├── call_graph.py                  # SIMPLIFY
│   ├── cfg_dfg.py                     # SIMPLIFY
│   ├── graph_targets.py               # NEW: consolidate support + metrics
│   └── import_graph.py                # SIMPLIFY
└── ingestion/
    ├── __init__.py
    ├── extraction_targets.py          # KEEP (already consolidated)
    ├── ingest_targets.py              # SIMPLIFY with templates
    └── scip.py                        # KEEP

hamilton/templates/
├── __init__.py
├── all_targets.py                     # EXISTS
├── ibis_pipeline.py                   # EXISTS
├── rows_pipeline.py                   # EXISTS
├── tool_pipeline.py                   # EXISTS
├── executor_pipeline.py               # NEW
├── multi_table_pipeline.py            # NEW
└── parameterized/                     # NEW
    ├── __init__.py
    ├── analytics_simple.py            # NEW
    └── graphs_simple.py               # NEW
```

### 8.3 Files Deleted/To Delete

#### Phase 2 (COMPLETED)

| File | Lines | Targets | Consolidated Into | Status |
|------|-------|---------|------------------|--------|
| `history_targets.py` | 193 | 2 | `metrics_targets.py` | ✅ Deleted |
| `coverage_pipeline.py` | 329 | 3 | `coverage_targets.py` | ✅ Deleted |
| `graph_metrics_pipeline.py` | 473 | 3 | `metrics_targets.py` | ✅ Deleted |

**Phase 2 Total**: 3 files deleted, ~995 lines removed

#### Phase 3 (PLANNED)

| File | Lines | Targets | Consolidated Into | Status |
|------|-------|---------|------------------|--------|
| `support_targets.py` | 968 | 3 | `graph_targets.py` | ⏳ Planned |
| `metrics_targets.py` (graphs) | 474 | 2 | `graph_targets.py` | ⏳ Planned |

**Phase 3 Planned**: 2 files deleted, ~1,442 lines to remove

#### Phase 4-5 (No file deletions planned)

Phases 4 and 5 focus on simplification, not file deletion.

**Grand Total**: 5 files deleted  
**Net line reduction**: ~2,400 lines

---

## 9. Migration Strategy

### 9.1 Migration Principles

1. **One Target at a Time**: Migrate targets individually, not files
2. **Test-First**: Ensure tests pass before and after each migration
3. **Preserve Node Names**: Keep `t__TARGET` and `t__TARGET__compute` names stable
4. **Backward Compatible**: Templates work alongside existing code during transition
5. **Feature Flaggable**: Use config flag to switch between old/new implementations

### 9.2 Migration Order

**Wave 1: Low-Risk Simple Targets** (Week 1-2)
1. `function_history` → rows_pipeline template
2. `history_timeseries` → rows_pipeline template
3. `subsystem_agreement` → ibis_pipeline template
4. Graph metrics (3 targets) → executor_pipeline template

**Wave 2: Medium Complexity** (Week 2-3)
5. `coverage_functions` → ibis_pipeline (already close)
6. `coverage_test_edges` → executor_pipeline
7. `behavioral_coverage` → executor_pipeline
8. `data_model_usage` → executor_pipeline
9. `function_ast_features` → rows_pipeline

**Wave 3: Multi-Table Targets** (Week 3-4)
10. `data_models` → multi_table_pipeline
11. `config_data_flow` → multi_table_pipeline
12. `cfg_dfg_metrics` → multi_table_pipeline
13. `function_metrics` → multi_table_pipeline (3 tables)

**Wave 4: Graphs Consolidation** (Week 4-5)
14. `goids` → executor_pipeline
15. `symbol_uses` → executor_pipeline
16. `call_graph_views` → executor_pipeline
17. `graph_metrics` → executor_pipeline
18. `graph_validation` → executor_pipeline

**Wave 5: Ingestion/Export** (Week 5-6)
19. Export targets → parameterize
20. Remaining ingestion targets → tool_pipeline

### 9.3 Migration Checklist (Per Target)

```markdown
## Migration: [TARGET_NAME]

### Pre-Migration
- [ ] Existing tests pass
- [ ] Document current node names
- [ ] Document current dependencies
- [ ] Identify pattern type

### Implementation
- [ ] Create template-based implementation
- [ ] Verify node names match original
- [ ] Verify DAG structure unchanged
- [ ] Add migration feature flag (optional)

### Validation
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E build test passes
- [ ] Manifest hashes stable (if applicable)

### Post-Migration
- [ ] Remove old implementation
- [ ] Update imports in registry.py
- [ ] Update __init__.py exports
- [ ] Remove feature flag
```

---

## 10. Testing Strategy

### 10.1 Test Coverage Requirements

| Level | Requirement |
|-------|-------------|
| Unit | Each template function has unit tests |
| Integration | Each consolidated target has integration test |
| DAG Parity | DAG structure matches before/after |
| Output Parity | Row counts and content match |
| Manifest Parity | Input hashes unchanged |

### 10.2 DAG Parity Test

```python
def test_dag_parity_after_consolidation():
    """Verify DAG structure unchanged after consolidation."""
    # Build driver with old modules
    old_runtime = build_driver(modules=OLD_MODULES)
    old_nodes = set(old_runtime.dr.list_available_variables())
    
    # Build driver with new modules
    new_runtime = build_driver(modules=NEW_MODULES)
    new_nodes = set(new_runtime.dr.list_available_variables())
    
    # Target nodes must match exactly
    old_targets = {n for n in old_nodes if n.startswith("t__")}
    new_targets = {n for n in new_nodes if n.startswith("t__")}
    assert old_targets == new_targets, f"Missing: {old_targets - new_targets}"
```

### 10.3 Output Parity Test

```python
@pytest.mark.parametrize("target_name", MIGRATED_TARGETS)
def test_output_parity(target_name, test_db, test_snapshot):
    """Verify migrated target produces same output."""
    # Run old implementation
    old_result = run_target_old(target_name, test_db, test_snapshot)
    
    # Run new implementation
    new_result = run_target_new(target_name, test_db, test_snapshot)
    
    # Compare
    assert old_result.status == new_result.status
    assert old_result.row_counts == new_result.row_counts
```

---

## 11. Risk Assessment

### 11.1 High Risk

| Risk | Mitigation | Status |
|------|------------|--------|
| DAG structure changes break manifest hashing | Test manifest hash stability per target | Ongoing |
| Node name changes break driver.execute() calls | Keep exact node names via decorator naming | Ongoing |
| Compute logic regression | Output parity tests for each target | Ongoing |
| **Python 3.13 Protocol incompatibility** | Use `Any` type alias instead of Protocol | ✅ Resolved in Phase 1 |

### 11.2 Medium Risk

| Risk | Mitigation | Status |
|------|------------|--------|
| @parameterize quirks with Hamilton | Spike test @parameterize behavior first | Phase 2 |
| Template composition complexity | Start with simple templates, iterate | ✅ Templates validated |
| Import cycles with templates | Keep templates dependency-free | ✅ No cycles |
| **TYPE_CHECKING imports break @subdag** | Import types at runtime in templates | ✅ Resolved in Phase 1 |

### 11.3 Low Risk

| Risk | Mitigation | Status |
|------|------------|--------|
| Slower execution from added abstraction | Benchmark critical paths | Phase 2+ |
| Developer learning curve | Document patterns with examples | ✅ Patterns documented |

### 11.4 Risks Discovered During Phase 1

| Risk | Discovery | Resolution |
|------|-----------|------------|
| Hamilton uses `issubclass()` for type matching | Hamilton DAG construction calls `issubclass()` on parameter types | Use `Any` for duck-typed parameters |
| Python 3.13 `Protocol` strict mode | `@runtime_checkable` protocols with data members don't support `issubclass()` | Document expected interface in comments, use `Any` |
| `get_type_hints()` requires runtime types | Hamilton's `Node.from_fn()` uses `typing.get_type_hints()` | Move type imports out of `TYPE_CHECKING` blocks |
| Assert statements in tests | S101 lint rule disallows `assert` even for type narrowing | Use `if x is None: return` pattern instead |

---

## 12. Implementation Roadmap

### 12.1 Timeline

```
Phase 1: Foundation ✅ COMPLETE (2025-12-17)
├── ✅ Create executor_pipeline.py template (ComputeResult, executor_materialize, record)
├── ✅ Create multi_table_pipeline.py template (multi_table_record, create_row_extractor)
├── ✅ Extend rows_pipeline.py with helpers (row_to_tuple, rows_to_tuples)
├── ✅ Update templates/__init__.py exports
└── ✅ Create 29 comprehensive tests

Phase 2: Analytics Domain ✅ COMPLETE (2025-12-17)
├── ✅ Create coverage_targets.py (3 targets: coverage_functions, coverage_test_edges, behavioral_coverage)
├── ✅ Create metrics_targets.py (6 targets: function_history, history_timeseries, 
│       subsystem_agreement, subsystem/symbol/test graph metrics)
├── ✅ Delete history_targets.py (193 lines)
├── ✅ Delete coverage_pipeline.py (329 lines)
├── ✅ Delete graph_metrics_pipeline.py (473 lines)
├── ✅ Update subsystem_targets.py (remove subsystem_agreement)
├── ✅ Update registry.py module paths
├── ✅ Create test_coverage_targets.py (10 tests)
├── ✅ Create test_metrics_targets.py (10 tests)
└── ✅ All 593 Hamilton tests passing

Phase 3: Graphs Domain (Ready - 2-3 days)
├── Consolidate support_targets.py + metrics_targets.py (graphs) → graph_targets.py
│   ├── goids → executor_materialize
│   ├── symbol_uses → executor_materialize
│   ├── call_graph_views → executor_materialize
│   ├── graph_metrics → executor_materialize
│   └── graph_validation → executor_materialize
├── Simplify call_graph.py, cfg_dfg.py, import_graph.py materialize nodes
└── Delete support_targets.py (~968 lines) + metrics_targets.py (~474 lines)

Phase 4: Ingestion Domain (After Phase 3 - 1-2 days)
├── Simplify ingest_targets.py materialize nodes with executor_materialize
│   ├── modules → executor_materialize
│   ├── config_ingest → executor_materialize
│   ├── coverage_ingest → executor_materialize
│   ├── tests_ingest → executor_materialize
│   └── typing → executor_materialize
└── Keep extraction_targets.py (@cache), scip.py (unique) AS-IS

Phase 5: Export Domain (After Phase 4 - 0.5-1 day)
├── Parameterize export_targets.py for JSONL/Parquet formats
└── Keep serving_artifacts.py (unique deployment logic) AS-IS

Phase 6: Cleanup & Polish (Final - 0.5 day)
├── Update any remaining imports
└── Final documentation updates
```

### 12.2 Success Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Native files | ≤18 | `find native -name "*.py" \| wc -l` |
| Native LoC | ≤6,000 | `wc -l native/**/*.py` |
| Test pass rate | 100% | `pytest` |
| DAG parity | 100% | Parity test |
| Build time | ≤110% of baseline | Benchmark |

### 12.3 Rollback Plan

Each wave can be rolled back independently:

1. Revert template changes
2. Restore original native module
3. Update registry.py imports
4. Verify tests pass

Keep old modules in a `_deprecated/` directory for 1 sprint after migration.

---

## Appendix A: Pattern Reference

### A.1 Complete Pattern Signatures

```python
# Pattern A: Ibis → DuckDB
@SaveToDecorator([DuckDBIbisTableSaver], output_name_=materialize_node(TABLE_KEY), ...)
def t__TARGET__compute(...) -> ir.Table: ...
def t__TARGET(env, graph, m__TABLE) -> TargetRunRecord: ...

# Pattern B: Rows → DuckDB
@SaveToDecorator([DuckDBRowsSaver], columns=value(COLS), ...)
def t__TARGET__compute(...) -> tuple[tuple[...], ...] | None: ...
def t__TARGET(env, graph, m__TABLE) -> TargetRunRecord: ...

# Pattern C: Multi-Table Rows
def t__TARGET__compute(...) -> ResultDataclass | None: ...
@SaveToDecorator([DuckDBRowsSaver], ...) 
def TARGET__table1_rows(compute_result) -> tuple | None: ...
def t__TARGET(env, graph, m__table1, m__table2, ...) -> TargetRunRecord: ...

# Pattern D: NativeExecutor
def t__TARGET__extract(...) -> ExtractResult: ...
def t__TARGET(env, graph, extract_result) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "TARGET")
    ...

# Pattern E: Tool Invocation
def t__TARGET__run(...) -> ToolResult: ...
def t__TARGET(env, graph, tool_result) -> TargetRunRecord: ...

# Pattern F: File Artifact
@SaveToDecorator([FileArtifactSaver], ...)
def TARGET__content(compute_result) -> bytes | str | None: ...
def t__TARGET(env, graph, artifact_metadata) -> TargetRunRecord: ...
```

### A.2 Target-to-Pattern Mapping

#### Phase 2 Completed Targets (9 targets consolidated)

| Target | Pattern | Original File | New File | Status |
|--------|---------|---------------|----------|--------|
| `function_history` | B | history_targets.py | metrics_targets.py | ✅ Complete |
| `history_timeseries` | B | history_targets.py | metrics_targets.py | ✅ Complete |
| `coverage_functions` | A | coverage_pipeline.py | coverage_targets.py | ✅ Complete |
| `coverage_test_edges` | D | coverage_pipeline.py | coverage_targets.py | ✅ Complete |
| `behavioral_coverage` | D | coverage_pipeline.py | coverage_targets.py | ✅ Complete |
| `subsystem_agreement` | D | subsystem_targets.py | metrics_targets.py | ✅ Complete |
| `subsystem_graph_metrics` | D | graph_metrics_pipeline.py | metrics_targets.py | ✅ Complete |
| `symbol_graph_metrics` | D | graph_metrics_pipeline.py | metrics_targets.py | ✅ Complete |
| `test_graph_metrics` | C | graph_metrics_pipeline.py | metrics_targets.py | ✅ Complete |

#### Phase 3 Targets (5 targets to consolidate)

| Target | Pattern | Current File | Template |
|--------|---------|--------------|----------|
| `goids` | D | support_targets.py | executor_materialize |
| `symbol_uses` | D | support_targets.py | executor_materialize |
| `call_graph_views` | D | support_targets.py | executor_materialize |
| `graph_metrics` | D | metrics_targets.py (graphs) | executor_materialize |
| `graph_validation` | D | metrics_targets.py (graphs) | executor_materialize |

#### Phase 4-5 Targets (simplify, no consolidation)

| Target | Pattern | Current File | Action |
|--------|---------|--------------|--------|
| `modules` | E | ingest_targets.py | Use executor_materialize |
| `config_ingest` | E | ingest_targets.py | Use executor_materialize |
| `coverage_ingest` | E | ingest_targets.py | Use executor_materialize |
| `tests_ingest` | E | ingest_targets.py | Use executor_materialize |
| `typing` | E | ingest_targets.py | Use executor_materialize |
| `export_jsonl` | F | export_targets.py | @parameterize |
| `export_parquet` | F | export_targets.py | @parameterize |

#### Targets to KEEP AS-IS (complex logic)

| Target | Pattern | Current File | Reason |
|--------|---------|--------------|--------|
| `function_metrics` | C | function_metrics.py | Multi-table with complex compute |
| `function_contracts` | C | function_detail_targets.py | AST analysis |
| `function_effects` | C | function_detail_targets.py | Effect analysis |
| `risk_factors` | A | risk_factors.py | @pipe_input with 5 steps |
| `hotspots` | A | hotspots.py | @pipe_input complexity |
| `subsystems` | A | subsystem_targets.py | @pipe_input clustering |
| `semantic_roles` | C | classification_targets.py | ML classification |
| `test_profile` | D | classification_targets.py | Test analysis |
| `config_data_flow` | C | config_graph_targets.py | Graph traversal |
| `cfg_dfg_metrics` | C | config_graph_targets.py | Control/data flow |
| `external_deps` | C | dependency_targets.py | Call graph analysis |
| `entrypoints` | D | dependency_targets.py | Entry detection |
| `data_models` | C | metadata_targets.py | Model extraction |
| `data_model_usage` | D | metadata_targets.py | Usage tracking |
| `function_ast_features` | B | metadata_targets.py | AST features |
| `profiles` | D | metadata_targets.py | Profile generation |
| `call_graph` | D | call_graph.py | Complex edge collection |
| `import_graph` | D | import_graph.py | AST + edge resolution |
| `cfg` | D | cfg_dfg.py | Control flow extraction |
| `dfg` | D | cfg_dfg.py | Data flow extraction |
| `ast` | E | extraction_targets.py | @cache |
| `cst` | E | extraction_targets.py | @cache |
| `docstrings` | E | extraction_targets.py | @cache |
| `scip` | E | scip.py | Unique async tool |
| `serving_artifacts` | F | serving_artifacts.py | FAISS deployment |
| `goids` | D | support_targets.py | executor_pipeline |
| `symbol_uses` | D | support_targets.py | executor_pipeline |
| `call_graph_views` | D | support_targets.py | executor_pipeline |
| `graph_metrics` | D | metrics_targets.py | executor_pipeline |
| `graph_validation` | D | metrics_targets.py | executor_pipeline |
| `call_graph` | D | call_graph.py | executor_pipeline |
| `import_graph` | D | import_graph.py | executor_pipeline |
| `cfg` | D | cfg_dfg.py | executor_pipeline |
| `dfg` | D | cfg_dfg.py | executor_pipeline |
| `modules` | E | ingest_targets.py | tool_pipeline |
| `config_ingest` | E | ingest_targets.py | tool_pipeline |
| `coverage_ingest` | E | ingest_targets.py | tool_pipeline |
| `tests_ingest` | E | ingest_targets.py | tool_pipeline |
| `typing` | E | ingest_targets.py | tool_pipeline |
| `ast` | E | extraction_targets.py | KEEP (cached) |
| `cst` | E | extraction_targets.py | KEEP (cached) |
| `docstrings` | E | extraction_targets.py | KEEP (cached) |
| `scip` | E | scip.py | KEEP (unique) |
| `export_jsonl` | F | export_targets.py | tool_pipeline |
| `export_parquet` | F | export_targets.py | tool_pipeline |
| `serving_artifacts` | F | serving_artifacts.py | KEEP (unique) |

---

## Appendix B: Phase 1 Deliverables

### B.1 Files Created/Modified

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/codeintel/build/hamilton/templates/executor_pipeline.py` | ✅ Created | 148 | Pattern D template |
| `src/codeintel/build/hamilton/templates/multi_table_pipeline.py` | ✅ Created | 245 | Pattern C template |
| `src/codeintel/build/hamilton/templates/rows_pipeline.py` | ✅ Extended | +80 | Row conversion helpers |
| `src/codeintel/build/hamilton/templates/__init__.py` | ✅ Updated | 68 | New exports |
| `tests/build/hamilton/test_executor_pipeline_template.py` | ✅ Created | 302 | 6 tests |
| `tests/build/hamilton/test_multi_table_pipeline_template.py` | ✅ Created | 411 | 12 tests |
| `tests/build/hamilton/test_rows_pipeline_helpers.py` | ✅ Created | 190 | 11 tests |

### B.2 Test Summary

**Total Tests**: 29 passing

| Test Category | Count | Description |
|---------------|-------|-------------|
| executor_pipeline direct | 4 | Success, failure, default error, protocol check |
| executor_pipeline @subdag | 2 | Integration via Hamilton driver |
| multi_table_pipeline direct | 4 | Success, partial failure, all skipped, mixed |
| multi_table_pipeline extractors | 8 | Factory patterns, edge cases |
| rows_pipeline helpers | 11 | row_to_tuple, rows_to_tuples variations |

### B.3 Exported Symbols

**From `codeintel.build.hamilton.templates`**:

```python
# Pattern D (NativeExecutor) Template
ComputeResult         # Type alias for duck-typed compute results
executor_materialize  # Main materialize function
executor_record       # Alias for subDAG composition

# Pattern C (Multi-Table) Template  
multi_table_record    # Combine multiple materializations
create_row_extractor  # Factory for row extraction functions

# Pattern B (Rows) Helpers
row_to_tuple          # Convert mapping to tuple
rows_to_tuples        # Batch convert mappings to tuples
rows_record           # Existing record function
rows_to_save          # Existing saver decorator

# Pattern F (Tool/Artifact) Template
tool_record           # Existing tool record function
tool_output_to_save   # Existing saver decorator

# Infrastructure
get_template_module   # Template module factory
ModuleType            # Re-exported for convenience
```

### B.4 Key Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Use `Any` instead of `Protocol` for `ComputeResult` | Python 3.13 Protocol doesn't support `issubclass()` for data-only protocols; Hamilton uses `issubclass()` internally |
| Import types at runtime (not TYPE_CHECKING) | Hamilton's `get_type_hints()` requires types available at runtime |
| Create `record()` aliases in templates | Cleaner node naming when used with `@subdag` |
| Place `create_row_extractor` in multi_table_pipeline | Logically grouped with multi-table patterns |
| Add `row_to_tuple`/`rows_to_tuples` to rows_pipeline | General utility for Pattern B and C targets |

---

## Appendix C: Phase 2 Deliverables

### C.1 Files Created/Modified

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/codeintel/build/hamilton/native/analytics/coverage_targets.py` | ✅ Created | 369 | Consolidates coverage_functions, coverage_test_edges, behavioral_coverage |
| `src/codeintel/build/hamilton/native/analytics/metrics_targets.py` | ✅ Created | 384 | Consolidates history + graph metrics + subsystem_agreement |
| `src/codeintel/build/hamilton/native/analytics/__init__.py` | ✅ Updated | 236 | Updated imports/exports |
| `src/codeintel/build/hamilton/native/registry.py` | ✅ Updated | 117 | Updated module paths |
| `src/codeintel/build/hamilton/native/analytics/subsystem_targets.py` | ✅ Trimmed | 274 | Removed subsystem_agreement |
| `src/codeintel/build/hamilton/templates/ibis_pipeline.py` | ✅ Fixed | 103 | Runtime type imports for Hamilton |
| `tests/build/hamilton/test_coverage_targets.py` | ✅ Created | 278 | 10 tests |
| `tests/build/hamilton/test_metrics_targets.py` | ✅ Created | 384 | 10 tests |
| `tests/build/hamilton/conftest.py` | ✅ Updated | 365 | Added save_manifest to FakeBuildAccessor |
| `tests/build/hamilton/test_pr52_no_legacy_orchestrators.py` | ✅ Updated | 162 | Reflect new module structure |
| `tests/build/hamilton/test_phase2_ibis_pipeline_template.py` | ✅ Fixed | 170 | Type injection for Hamilton |

### C.2 Files Deleted

| File | Lines | Targets | Reason |
|------|-------|---------|--------|
| `history_targets.py` | 193 | function_history, history_timeseries | Consolidated into metrics_targets.py |
| `coverage_pipeline.py` | 329 | coverage_functions, coverage_test_edges, behavioral_coverage | Consolidated into coverage_targets.py |
| `graph_metrics_pipeline.py` | 473 | subsystem/symbol/test_graph_metrics | Consolidated into metrics_targets.py |

**Total lines deleted**: ~995 lines

### C.3 Test Summary

**Total Tests**: 593 passing (20 new Phase 2 tests)

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_coverage_targets.py | 10 | Coverage target Result classes and materialize functions |
| test_metrics_targets.py | 10 | Metrics target Result classes and materialize functions |

### C.4 Key Learnings Applied

| Learning | Application |
|----------|-------------|
| `@tag(target_="fn_name")` required with `@SaveToDecorator` | Applied to `t__coverage_functions__compute` |
| Use `fake_gateway` fixture, not `memory_gateway` | All new tests use correct fixture |
| Add `save_manifest()` to `FakeBuildAccessor` | Method added to `conftest.py` |
| Runtime type imports for Hamilton | Fixed `ibis_pipeline.py` and test files |
| Named constants for magic numbers | Added `MAX_*_COUNT` constants in tests |

### C.5 Consolidated Target Summary

| New Module | Targets | Pattern | Source Files |
|------------|---------|---------|--------------|
| `coverage_targets.py` | 3 | A, D, D | coverage_pipeline.py |
| `metrics_targets.py` | 6 | B, B, D, D, D, C | history_targets.py, graph_metrics_pipeline.py, subsystem_targets.py |

---

*This document is the authoritative implementation plan for subdag-based consolidation. Last updated: 2025-12-17 after Phase 2 completion.*

