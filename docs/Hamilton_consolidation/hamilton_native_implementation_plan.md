# Hamilton Native Implementation Plan

> **Purpose**: Comprehensive, detailed implementation plan for migrating to a 100% native Hamilton architecture, eliminating the plugin abstraction layer entirely.

**Status**: Phase 2 Complete  
**Version**: 4.0  
**Created**: 2025-12-15  
**Last Updated**: 2025-12-15  
**Target Completion**: TBD

---

### Document Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2025-12-15 | Initial design document |
| 2.0 | 2025-12-15 | Added 16 advanced Hamilton features integration plan |
| 3.0 | 2025-12-15 | Phase 1 implementation complete, all PRs documented |
| 3.1 | 2025-12-15 | Added Phase 1.5 validation POC plan |
| 4.0 | 2025-12-15 | Phase 1.5 and Phase 2 implementation complete |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1 Implementation Status](#phase-1-implementation-status) ✅ **COMPLETE**
3. [Phase 1.5 Implementation Status](#phase-15-implementation-status) ✅ **COMPLETE**
4. [Phase 2 Implementation Status](#phase-2-implementation-status) ✅ **COMPLETE**
5. [Phase 1.5 Design](#phase-15-validation-proof-of-concept)
6. [Advanced Hamilton Features Integration](#advanced-hamilton-features-integration)
7. [Architectural Vision](#architectural-vision)
8. [Current State Analysis](#current-state-analysis)
9. [Target State Architecture](#target-state-architecture)
10. [Implementation Phases](#implementation-phases)
11. [Detailed PR Breakdown](#detailed-pr-breakdown)
12. [Migration Recipes](#migration-recipes)
13. [Testing Strategy](#testing-strategy)
14. [Risk Assessment & Mitigation](#risk-assessment--mitigation)
15. [Success Criteria](#success-criteria)
16. [Appendix: File-by-File Disposition](#appendix-file-by-file-disposition)

---

## Executive Summary

### Goal

Transform the build system from a **dual execution model** (plugins + native Hamilton) to a **100% native Hamilton architecture** where:

- Every target is a Hamilton module with pure compute nodes and a materialize node
- `BuildEnv` is the single execution context
- Dependencies are expressed via function signatures (Hamilton resolves them)
- Skip logic, manifest persistence, and observability are uniform via Hamilton hooks
- The plugin abstraction layer is completely eliminated

### Scope

| Category | Action | Lines Affected |
|----------|--------|----------------|
| Delete plugin infrastructure | Remove `plugin.py`, `context.py`, `context_base.py` | -1,612 |
| Delete plugin registry | Remove `unified_registry.py`, `registrations.py` | -794 |
| Migrate plugins to native | Convert ~45 plugins → ~45 Hamilton modules | ~3,000 (rewrite) |
| Consolidate execution | Remove dual paths in executor, node_factory | -400 |
| Consolidate hashing/skip | Unify into single implementations | -200 |
| Simplify registry | Remove static constants, keep metadata only | -500 |
| **Total Net Change** | | **~-3,500 lines** |

### Timeline Estimate

| Phase | Duration | Description | Validation Effort | Status |
|-------|----------|-------------|-------------------|--------|
| Phase 1 | 1-2 weeks | Foundation (env consolidation, hooks) | ✅ Complete | ✅ Complete |
| Phase 1.5 | 1-2 days | Validation POC (existing native targets) | ✅ Complete | ✅ Complete |
| Phase 2 | 2-3 weeks | Ingestion domain migration | ✅ Complete | ✅ Complete |
| Phase 3 | 2-3 weeks | Graphs domain migration | +4 hours | 🔜 Next |
| Phase 4 | 2-3 weeks | Analytics domain migration | +7 hours | Pending |
| Phase 5 | 1 week | Export domain migration | +1 hour | Pending |
| Phase 6 | 1 week | Cleanup, deletion, validation finalization | +1 day | Pending |
| **Total** | **9-13 weeks** | | **+3-4 days** | |

---

## Phase 1 Implementation Status

> ✅ **COMPLETE** - All Phase 1 foundation PRs have been implemented and tested.

This section documents the actual implementation of Phase 1 advanced features, including file locations, APIs, and integration patterns for use in subsequent phases.

### PR-100: Hook Consolidation ✅

**Status**: Complete

**Files Created/Modified:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/hooks/__init__.py` | Centralized hook exports |
| `src/codeintel/build/hamilton/hooks/manifest_hook.py` | Skip logic + manifest persistence |
| `src/codeintel/build/hamilton/hooks/telemetry_hook.py` | Observability spans/metrics |
| `src/codeintel/build/hamilton/hooks/contract_hook.py` | Contract enforcement |
| `src/codeintel/build/hamilton/hooks/lifecycle.py` | Advanced lifecycle hooks (new) |

**Backward Compatibility Aliases:**

| Old Import Path | Status |
|-----------------|--------|
| `codeintel.build.hamilton.manifest_hook` | Re-export alias preserved |
| `codeintel.build.hamilton.telemetry_hook` | Re-export alias preserved |
| `codeintel.build.hamilton.contracts.enforcement_hook` | Re-export alias preserved |

---

### PR-100.5: Hamilton-Native Data Validation ✅

**Status**: Complete

> **IMPORTANT DESIGN DECISION**: We chose Hamilton-native custom validators over Pandera `@check_output` integration. This makes the DAG engine the authoritative source for validation without external schema dependencies.

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/validators/__init__.py` | Package exports |
| `src/codeintel/build/hamilton/validators/dataframe.py` | Custom DataFrame validators |
| `src/codeintel/build/hamilton/validators/contracts.py` | Validator factory functions |
| `src/codeintel/build/hamilton/schema_docs.py` | `@schema.output` utilities |

**Custom Validators Implemented:**

```python
from codeintel.build.hamilton.validators import (
    ColumnsExistValidator,      # Verify required columns exist
    ColumnTypesValidator,       # Verify column dtypes match expected
    NoNullsInColumnsValidator,  # Verify no nulls in specified columns
    UniqueColumnsValidator,     # Verify uniqueness constraints
    RowCountValidator,          # Verify exact row count
    RowCountRangeValidator,     # Verify row count in range
    ColumnValuesInSetValidator, # Verify values from allowed set
)
```

**Contract Builder Functions:**

```python
from codeintel.build.hamilton.validators import (
    build_table_contract,       # Standard table validation
    build_key_column_contract,  # Primary key constraints
    build_metrics_contract,     # Numeric range validation
    build_enum_column_contract, # Enum/categorical validation
)
```

**Usage Pattern for Phase 2+ Targets:**

```python
from hamilton.function_modifiers import check_output_custom, tag
from codeintel.build.hamilton.validators import (
    build_table_contract,
    build_key_column_contract,
)

@tag(domain="analytics", target="function_metrics", node_type="compute")
@check_output_custom(*build_table_contract(
    required_columns=["function_goid_h128", "repo", "commit", "loc"],
    column_types={"loc": "int64", "complexity": "int64"},
    non_null_columns=["function_goid_h128", "repo", "commit"],
))
@check_output_custom(*build_key_column_contract(
    key_columns=["function_goid_h128", "repo", "commit"],
))
def t__function_metrics__compute(...) -> pd.DataFrame:
    """Compute function metrics with Hamilton-native validation."""
    ...
```

**Schema Documentation Pattern:**

```python
from codeintel.build.hamilton.schema_docs import schema_output_tuple

# Use @schema.output for documentation (visible in Hamilton UI)
from hamilton.function_modifiers import schema

@tag(domain="analytics", target="function_metrics", node_type="compute")
@schema.output(*schema_output_tuple([
    ("function_goid_h128", "string", "Unique function identifier"),
    ("repo", "string", "Repository name"),
    ("commit", "string", "Commit SHA"),
    ("loc", "int", "Lines of code"),
    ("complexity", "int", "Cyclomatic complexity"),
]))
def t__function_metrics__compute(...) -> pd.DataFrame:
    ...
```

---

### PR-100.6: Extended Lifecycle Hooks ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/hooks/lifecycle.py` | Advanced lifecycle hooks |

**Hooks Implemented:**

```python
from codeintel.build.hamilton.hooks import (
    ProgressBarHook,    # tqdm progress visualization
    BuildTimingHook,    # Per-node execution timing
    ConditionalHook,    # Conditionally enable/disable hooks
    create_progress_hook,  # Factory with CI detection
)
```

**ProgressBarHook Usage:**

```python
from codeintel.build.hamilton.hooks import ProgressBarHook, create_progress_hook

# Manual instantiation
hook = ProgressBarHook(desc="Building targets", disable=False)

# Factory with automatic CI detection (disables in CI environments)
hook = create_progress_hook(desc="Building targets")
```

**BuildTimingHook Usage:**

```python
from codeintel.build.hamilton.hooks import BuildTimingHook

hook = BuildTimingHook()
# After execution:
hook.timings  # dict mapping node names to NodeTimingRecord
hook.total_duration_seconds()  # Total execution time
hook.slowest_nodes(n=10)  # Top N slowest nodes
```

**ConditionalHook Usage:**

```python
from codeintel.build.hamilton.hooks import ConditionalHook

# Only enable timing in debug mode
timing_hook = ConditionalHook(
    BuildTimingHook(),
    enabled=os.getenv("DEBUG") == "1",
)
```

**Updated `build_hooks()` Function:**

```python
from codeintel.build.hamilton.hooks import build_hooks

hooks = build_hooks(
    manifest_index=manifest_index,
    telemetry_enabled=True,
    enable_progress=True,   # NEW: Enable tqdm progress bar
    enable_timing=True,     # NEW: Enable timing collection
)
```

---

### PR-100.7: Migration Bridge ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/validators/migration.py` | Pandera-to-Hamilton migration utilities |

**Migration Utilities:**

```python
from codeintel.build.hamilton.validators import (
    validators_from_pandera_schema,   # Convert Pandera schema to validators
    validators_from_schema_registry,  # Convert from SCHEMA_REGISTRY
    schema_output_from_registry,      # Generate @schema.output from registry
    MigrationReport,                  # Track migration status
)
```

**Usage for Migrating Existing Schemas:**

```python
from codeintel.build.hamilton.validators.migration import (
    validators_from_schema_registry,
    schema_output_from_registry,
    generate_migration_code,
)

# Generate Hamilton validators from existing Pandera schema
validators = validators_from_schema_registry("analytics.function_metrics")

# Generate @schema.output arguments
schema_args = schema_output_from_registry("analytics.function_metrics")

# Generate complete migration code for a target
migration_code = generate_migration_code("analytics.function_metrics")
print(migration_code)  # Outputs ready-to-use Python code
```

**Migration Report:**

```python
report = MigrationReport()
report.add_converted("analytics.function_metrics", validators)
report.add_skipped("analytics.legacy_table", "No Pandera schema defined")
report.summary()  # Returns migration statistics
```

---

### PR-101: NativeTargetExecutor with Async Support ✅

**Status**: Complete

**Files Modified:**

| File | Change |
|------|--------|
| `src/codeintel/build/hamilton/native/executor.py` | Added `execute_async()` method |

**New Async API:**

```python
from codeintel.build.hamilton.native import NativeTargetExecutor

executor = NativeTargetExecutor.for_target(env, graph, "function_metrics")

# Synchronous execution (existing)
result = executor.execute(lambda: {...})

# Asynchronous execution (new)
result = await executor.execute_async(async_compute_fn)
```

---

### PR-101.5: Custom BuildResultBuilder ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/result_builder.py` | Custom result builders |

**Classes Implemented:**

```python
from codeintel.build.hamilton.result_builder import (
    BuildResultBuilder,     # Aggregates results into BuildExecutionResult
    BuildExecutionResult,   # Structured execution output
    NodeResult,             # Per-node result container
    ResultStatus,           # Enum: SUCCESS, FAILURE, SKIPPED, PARTIAL
    DictResultBuilder,      # Simple dict aggregation
)
```

**BuildExecutionResult Structure:**

```python
@dataclass
class BuildExecutionResult:
    """Structured result of Hamilton DAG execution."""
    status: ResultStatus
    nodes: dict[str, NodeResult]
    started_at: datetime
    ended_at: datetime
    
    @property
    def duration_seconds(self) -> float: ...
    @property
    def succeeded_nodes(self) -> list[str]: ...
    @property
    def failed_nodes(self) -> list[str]: ...
    @property
    def skipped_nodes(self) -> list[str]: ...
    def summary(self) -> str: ...
```

**Usage in Driver Construction:**

```python
from hamilton import base, driver
from codeintel.build.hamilton.result_builder import BuildResultBuilder

adapter = base.SimplePythonGraphAdapter(
    result_builder=BuildResultBuilder()
)
dr = driver.Builder().with_modules(*modules).with_adapter(adapter).build()

result: BuildExecutionResult = dr.execute(["t__function_metrics"])
print(result.summary())
```

---

### PR-102: NativeModuleLoader ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/native/loader.py` | Module discovery and loading |

**API:**

```python
from codeintel.build.hamilton.native import NativeModuleLoader

# List available domains
domains = NativeModuleLoader.list_domains()  # ['ingestion', 'graphs', 'analytics', 'export']

# List module paths for a domain
paths = NativeModuleLoader.list_module_paths("analytics")

# Discover all modules with metadata
modules = NativeModuleLoader.discover_modules()  # List[NativeModuleInfo]

# Validate a module
is_valid, errors = NativeModuleLoader.validate_module(module)

# Load modules for Hamilton driver
loaded = NativeModuleLoader.load_for_driver(["analytics", "graphs"])

# Get target names from modules
targets = NativeModuleLoader.get_target_names(loaded)
```

---

### PR-103: Native-Only Mode Flag ✅

**Status**: Complete

**Files Modified:**

| File | Change |
|------|--------|
| `src/codeintel/build/hamilton/driver_factory.py` | Added "native" to HamiltonNodeMode |
| `src/codeintel/cli/commands/build.py` | Updated --hamilton-mode choices |
| `src/codeintel/cli/handlers/build.py` | Updated validation |

**CLI Usage:**

```bash
# Run in pure native Hamilton mode (no plugin wrappers)
codeintel build run --hamilton-mode native

# Other modes still available:
codeintel build run --hamilton-mode generated  # Plugin wrappers
codeintel build run --hamilton-mode auto       # Auto-detect
```

---

### PR-103.5: Parallel Execution Adapters ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/adapters/__init__.py` | Package exports |
| `src/codeintel/build/hamilton/adapters/parallel.py` | Parallel execution adapters |

**Classes Implemented:**

```python
from codeintel.build.hamilton.adapters import (
    ExecutionBackend,       # Enum: SEQUENTIAL, THREADPOOL, AUTO
    ParallelConfig,         # Configuration dataclass
    ThreadPoolAdapter,      # Hamilton FutureAdapter wrapper
    create_parallel_adapter,  # Factory function
)
```

**CLI Flags Added:**

```bash
# Enable parallel execution
codeintel build run --parallel-backend threadpool --max-workers 4

# Auto-detect (uses threadpool if multiple targets)
codeintel build run --parallel-backend auto

# With progress bar
codeintel build run --parallel-backend threadpool --progress
```

**Programmatic Usage:**

```python
from codeintel.build.hamilton.adapters import (
    ExecutionBackend,
    ParallelConfig,
    create_parallel_adapter,
)

# Create configuration
config = ParallelConfig(
    backend=ExecutionBackend.THREADPOOL,
    max_workers=4,
)

# Create adapter
adapter = create_parallel_adapter(config)

# Use in driver
dr = driver.Builder().with_modules(*modules).with_adapter(adapter).build()
```

**Environment Variable Configuration:**

```bash
export CODEINTEL_PARALLEL_BACKEND=threadpool
export CODEINTEL_MAX_WORKERS=8
```

---

### PR-104: Migration Test Harness ✅

**Status**: Complete

**Files Created:**

| File | Description |
|------|-------------|
| `tests/build/hamilton/native/__init__.py` | Test package |
| `tests/build/hamilton/native/conftest.py` | Pytest fixtures |
| `tests/build/hamilton/native/harness.py` | MigrationTestHarness class |
| `tests/build/hamilton/native/test_parity.py` | Parity tests |
| `tests/build/hamilton/native/test_skip_logic.py` | Skip logic tests |

**Test Coverage:**

- ✅ Module loader discovery tests
- ✅ Module validation tests
- ✅ Driver factory mode tests
- ✅ Skip logic tests (should_skip, force override, input change)
- ✅ Executor execute/fail/skip tests
- ✅ Manifest persistence tests

**MigrationTestHarness Usage:**

```python
from tests.build.hamilton.native.harness import MigrationTestHarness

harness = MigrationTestHarness(plugin_gateway, native_gateway)

# Compare outputs
harness.compare_row_counts("analytics.function_metrics")
harness.compare_table_contents("analytics.function_metrics", key_columns=["function_goid_h128"])
harness.compare_table_schema("analytics.function_metrics")
```

---

### Test Results Summary

All Phase 1 tests pass:

```
tests/build/hamilton/validators/test_dataframe_validators.py .......... [  5%]
tests/build/hamilton/validators/test_contracts.py ........            [ 10%]
tests/build/hamilton/validators/test_schema_docs.py .......           [ 13%]
tests/build/hamilton/validators/test_migration.py ........            [ 18%]
tests/build/hamilton/hooks/test_lifecycle.py ..........               [ 23%]
tests/build/hamilton/test_result_builder.py ..........                [ 28%]
tests/build/hamilton/adapters/test_parallel.py ............           [ 34%]
tests/build/hamilton/native/test_parity.py ...........................[ 48%]
tests/build/hamilton/native/test_skip_logic.py ........................[100%]

196 passed
```

---

### Go-Forward Integration Guide

**For Phase 2-5 Target Migrations**, use the following pattern:

```python
"""Native Hamilton implementation for <target_name> target.

Phase: 2/3/4/5
PR: PR-XXX
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from hamilton.function_modifiers import check_output_custom, schema, tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.validators import (
    build_table_contract,
    build_key_column_contract,
)
from codeintel.build.hamilton.schema_docs import schema_output_tuple
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    import ibis.expr.types as ir


# Compute node with Hamilton-native validation
@tag(domain="<domain>", target="<target>", node_type="compute")
@check_output_custom(*build_table_contract(
    required_columns=["col1", "col2"],
    non_null_columns=["col1"],
))
@schema.output(*schema_output_tuple([
    ("col1", "string", "Primary key"),
    ("col2", "int", "Metric value"),
]))
def t__<target>__compute(
    q__<schema>__<dep>: ir.Table,
) -> pd.DataFrame:
    """Pure computation with validation and schema documentation."""
    ...


# Materialize node
@tag(domain="<domain>", target="<target>", node_type="materialize")
def t__<target>(
    env: BuildEnv,
    graph: TargetGraph,
    t__<target>__compute: pd.DataFrame,
) -> TargetRunRecord:
    """Materialize with skip logic and manifest persistence."""
    executor = NativeTargetExecutor.for_target(env, graph, "<target>")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: {
        "<schema>.<table>": materialize_table(env, "<schema>.<table>", t__<target>__compute),
    })
```

---

## Phase 1.5: Validation Proof-of-Concept

> ✅ **COMPLETE** - Validators applied to existing native targets.

This phase validates the Hamilton-native validation infrastructure on existing native targets before rolling it out across all target migrations. This ensures the approach works end-to-end before committing to it for 45+ targets.

### Objective

Apply Hamilton-native validators to existing native targets (`t__risk_factors`, `t__hotspots`) to:
1. Validate the infrastructure works in practice
2. Refine error messages and failure modes
3. Integrate with existing `ContractEnforcementHook`
4. Establish patterns for Phase 2-6 migrations

### PR-104.5: Validation Proof-of-Concept

**Status**: ✅ Complete

**Effort**: 1-2 days (standalone)

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Apply `@check_output_custom` to `t__risk_factors__compute` | 1 hour |
| 2 | Apply `@schema.output` to `t__risk_factors__compute` | 30 min |
| 3 | Apply validators to `t__hotspots__compute` | 1 hour |
| 4 | Update `ContractEnforcementHook` to work with Hamilton validators | 2 hours |
| 5 | Run with real repository data, verify error messages | 1 hour |
| 6 | Document learnings, refine contracts.py if needed | 1 hour |
| 7 | Update `build_hooks()` to enable validation by default | 30 min |

**Implementation Pattern:**

```python
# src/codeintel/build/hamilton/native/analytics/risk_factors.py

from hamilton.function_modifiers import check_output_custom, schema, tag
from codeintel.build.hamilton.validators import (
    build_table_contract,
    build_key_column_contract,
)
from codeintel.build.hamilton.schema_docs import schema_output_tuple

@tag(domain="analytics", target="risk_factors", node_type="compute")
@check_output_custom(*build_table_contract(
    required_columns=["function_goid_h128", "repo", "commit", "risk_score"],
    column_types={"risk_score": "float64"},
    non_null_columns=["function_goid_h128", "repo", "commit"],
))
@check_output_custom(*build_key_column_contract(
    key_columns=["function_goid_h128", "repo", "commit"],
))
@schema.output(*schema_output_tuple([
    ("function_goid_h128", "string", "Unique function identifier"),
    ("repo", "string", "Repository name"),
    ("commit", "string", "Commit SHA"),
    ("risk_score", "float", "Computed risk score (0-1)"),
    ("complexity_factor", "float", "Complexity contribution"),
    ("churn_factor", "float", "Change frequency contribution"),
]))
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
    q__graph__call_graph_edges: ir.Table,
) -> pd.DataFrame:
    """Compute risk factors with Hamilton-native validation."""
    return compute_risk_factors(q__analytics__function_metrics, q__graph__call_graph_edges)
```

**Hook Integration Update:**

```python
# Update ContractEnforcementHook to report validation results

class ContractEnforcementHook(NodeExecutionHook):
    """Hook that reports validation results from Hamilton @check_output_custom."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.validation_results: dict[str, ValidationResult] = {}
    
    def post_node_execute(
        self,
        run_id: str,
        node_name: str,
        result: Any,
        success: bool,
        error: Exception | None,
        **kwargs,
    ) -> None:
        # Hamilton already validated via @check_output_custom
        # This hook captures results for reporting
        if error and "validation" in str(error).lower():
            self.validation_results[node_name] = ValidationResult(
                node=node_name,
                passed=False,
                error=str(error),
            )
        else:
            self.validation_results[node_name] = ValidationResult(
                node=node_name,
                passed=True,
            )
```

**Acceptance Criteria:**

- [x] `t__risk_factors` uses `@check_output_custom` validators
- [x] `t__hotspots` uses `@check_output_custom` validators
- [x] Both targets have `@schema.output` documentation
- [x] `ContractEnforcementHook` reports validation results
- [x] Validation errors produce clear, actionable messages
- [x] `build_hooks()` enables validation by default
- [x] Tests verify validation catches bad data

---

## Phase 1.5 Implementation Status

> ✅ **COMPLETE** - Validation POC successfully implemented and tested.

This section documents the actual implementation of Phase 1.5 validation proof-of-concept.

### PR-104.5: Validation POC ✅

**Status**: Complete

**Files Created/Modified:**

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/native/analytics/risk_factors.py` | Added `@check_output_custom` with consolidated validators |
| `src/codeintel/build/hamilton/native/analytics/hotspots.py` | Added `@check_output_custom` with consolidated validators |
| `src/codeintel/build/hamilton/hooks/contract_hook.py` | Updated with `ValidationResult`, `ValidationSummary` dataclasses |
| `src/codeintel/build/hamilton/validators/dataframe.py` | Enhanced validators to support Ibis tables |
| `tests/build/hamilton/validators/test_validation_poc.py` | Comprehensive validation POC tests |

**Key Implementation Details:**

1. **Consolidated Validators**: Instead of multiple `@check_output_custom` decorators (which caused Hamilton to create duplicate nodes), validators are consolidated into a single decorator:

```python
@check_output_custom(
    *build_table_contract(...),
    *build_enum_column_contract(...),
)
def t__risk_factors__compute(...) -> ir.Table:
    ...
```

2. **Ibis Table Support**: Validators were enhanced to support `ibis.expr.types.Table` in addition to `pd.DataFrame`:
   - `applies_to()` returns `True` for both `pd.DataFrame` and `ir.Table`
   - Schema-level checks (columns, types) work on Ibis tables
   - Data-level checks (nulls, uniqueness, row counts) are skipped for Ibis tables (lazy evaluation)

3. **Hook Integration**: `ContractEnforcementHook.post_node_execute()` extracts `_output_validation_results` from Hamilton's internal kwargs to populate `validation_summary`

4. **Validation by Default**: `build_hooks()` now enables validation by default when `contract_enforcement_hook` is included

**Learnings Applied to Phase 2:**

- **Decorator Consolidation**: Use single `@check_output_custom` with unpacked validator lists
- **Ibis Compatibility**: Validators automatically handle Ibis tables with schema-only checks
- **Node Naming**: Hamilton-generated validator nodes (ending in `_validator`) are filtered from `node_type` tag requirements
- **Return Type Awareness**: `@check_output_custom` and `@schema.output` only apply to functions returning `pd.DataFrame` or `ir.Table`, not custom dataclasses

---

### Validation Rollout Strategy

After PR-104.5 validates the approach, validation will be integrated into all subsequent phases:

```
Phase 1.5 (POC):     Apply to 2 existing native targets
                     ↓
Phase 2 (Ingestion): Apply to 12 targets during migration (+30 min/target)
                     ↓
Phase 3 (Graphs):    Apply to 8 targets during migration (+30 min/target)
                     ↓
Phase 4 (Analytics): Apply to 20+ targets during migration (+30 min/target)
                     ↓
Phase 5 (Export):    Apply to 2 targets during migration (+30 min/target)
                     ↓
Phase 6 (Cleanup):   Remove SCHEMA_REGISTRY Pandera dependencies (+1 day)
```

**Total Additional Effort for Validation**: ~3-4 days spread across phases

### Schema Registry Transition Plan

```
Current Flow (External):
┌──────────────────┐     ┌─────────────────┐     ┌───────────────┐
│ SCHEMA_REGISTRY  │ ──> │ Pandera Schema  │ ──> │ Post-hoc      │
│ (external)       │     │ validation      │     │ validation    │
└──────────────────┘     └─────────────────┘     └───────────────┘

Target Flow (Hamilton-Native):
┌──────────────────┐     ┌─────────────────┐     ┌───────────────┐
│ @check_output_   │ ──> │ Hamilton DAG    │ ──> │ Per-node      │
│ custom(...)      │     │ validates       │     │ validation    │
└──────────────────┘     └─────────────────┘     └───────────────┘
```

**Transition Steps:**

1. **Phase 1.5**: Validate approach on existing targets
2. **Phases 2-5**: Use migration bridge to generate validators from SCHEMA_REGISTRY during each target migration
3. **Phase 6**: Remove SCHEMA_REGISTRY Pandera dependencies; Hamilton validators become authoritative
4. **Post-Phase 6** (Optional): Delete Pandera schema definitions (can keep for documentation)

---

## Phase 2 Implementation Status

> ✅ **COMPLETE** - All ingestion domain targets migrated to native Hamilton.

This section documents the actual implementation of Phase 2 Ingestion Domain Migration.

### Overview

Phase 2 migrated all ingestion targets to native Hamilton modules, eliminating their plugin implementations and establishing the pattern for subsequent phases.

**Key Outcomes:**
- 9 native Hamilton modules created for ingestion targets
- All ingestion plugins deleted
- Parity tests verify native implementations match expected behavior
- Target naming conventions established (`t__<domain>_ingest` pattern for disambiguation)

### PRs Completed

| PR | Description | Status |
|----|-------------|--------|
| PR-105 | Native `t__modules` | ✅ Complete |
| PR-106 | Enhanced `t__scip` | ✅ Complete |
| PR-107 | Native `t__ast` | ✅ Complete |
| PR-108 | Native `t__cst` | ✅ Complete |
| PR-109 | Enhanced `t__typing` | ✅ Complete |
| PR-110 | Native `t__tests_ingest` | ✅ Complete |
| PR-111 | Native `t__docstrings` | ✅ Complete |
| PR-112 | Native `t__coverage_ingest` | ✅ Complete |
| PR-113 | Native `t__config_ingest` | ✅ Complete |
| PR-114 | Ingestion domain parity tests | ✅ Complete |
| PR-115 | Delete ingestion plugins | ✅ Complete |
| PR-116 | Update registrations | ✅ Complete |

### Files Created

| File | Description |
|------|-------------|
| `src/codeintel/build/hamilton/native/ingestion/modules.py` | Repository scanning and module discovery |
| `src/codeintel/build/hamilton/native/ingestion/ast.py` | AST extraction from Python files |
| `src/codeintel/build/hamilton/native/ingestion/cst.py` | CST extraction from Python files |
| `src/codeintel/build/hamilton/native/ingestion/tests.py` | Test file ingestion |
| `src/codeintel/build/hamilton/native/ingestion/docstrings.py` | Docstring extraction |
| `src/codeintel/build/hamilton/native/ingestion/coverage.py` | Coverage data ingestion |
| `src/codeintel/build/hamilton/native/ingestion/config.py` | Configuration file scanning and ingestion |
| `src/codeintel/build/plugins/ingestion/stubs.py` | Stub classes for deleted plugins (test infrastructure) |
| `tests/build/hamilton/native/ingestion/test_parity.py` | Parity tests for native ingestion modules |

### Files Deleted

| File | Description |
|------|-------------|
| `src/codeintel/build/plugins/ingestion/repo_scan.py` | Old RepoScanPlugin |
| `src/codeintel/build/plugins/ingestion/ast_extract.py` | Old AstExtractPlugin |
| `src/codeintel/build/plugins/ingestion/cst_extract.py` | Old CstExtractPlugin |
| `src/codeintel/build/plugins/ingestion/tests_ingest.py` | Old TestsIngestPlugin |
| `src/codeintel/build/plugins/ingestion/docstrings_extract.py` | Old DocstringsExtractPlugin |
| `src/codeintel/build/plugins/ingestion/coverage_ingest.py` | Old CoverageIngestPlugin |
| `src/codeintel/build/plugins/ingestion/config_ingest.py` | Old ConfigIngestPlugin |
| `tests/ingestion/plugins/test_*.py` | Old plugin-specific tests |

### Key Implementation Patterns

**1. Native Module Structure:**

```python
# Each native module follows this pattern:
@tag(domain="ingestion", target="<target_name>", node_type="compute")
def t__<target_name>__<step>(env: BuildEnv) -> <ResultType>:
    """Compute step that does the actual work."""
    ...

@tag(domain="ingestion", target="<target_name>", node_type="materialize")
def t__<target_name>(env: BuildEnv, t__<target_name>__<step>: <ResultType>) -> TargetRunRecord:
    """Materialize node that persists results and creates run record."""
    executor = NativeTargetExecutor(env)
    return executor.for_target("<target_name>").execute(...)
```

**2. Target Naming Conventions:**

Targets that could conflict with reserved words or other domains use `_ingest` suffix:
- `t__tests_ingest` (not `t__tests` to avoid pytest conflicts)
- `t__coverage_ingest` (not `t__coverage` to distinguish from analytics coverage)
- `t__config_ingest` (not `t__config` to distinguish from configuration)

**3. Registration Pattern:**

```python
# src/codeintel/build/registrations.py
register_target(
    name="modules",
    domain="ingestion",
    native_module="codeintel.build.hamilton.native.ingestion.modules",
    # No plugin= argument - native only
)
```

**4. Stub Classes for Test Infrastructure:**

```python
# src/codeintel/build/plugins/ingestion/stubs.py
# Provides minimal stub classes for deleted plugins
# Used by test helpers that need type references
class RepoScanPlugin(_StubPluginBase):
    _core_metadata: ClassVar[CorePluginMetadata] = REPO_SCAN_METADATA
```

### Learnings and Callouts for Phase 3

1. **Naming Consistency**: The `@tag(target="...")` value MUST exactly match the target name in `UnifiedRegistry`. Mismatches cause graph validation failures.

2. **Validator Applicability**: `@check_output_custom` and `@schema.output` only work on functions returning `pd.DataFrame` or `ibis.expr.types.Table`. Functions returning custom dataclasses should NOT use these decorators.

3. **Helper Functions**: Private helper functions (starting with `_`) should be tagged with `@tag(node_type="helper")` to prevent Hamilton from treating them as compute nodes.

4. **Test Infrastructure**: When deleting plugins, create stub classes if test helpers need type references. This maintains backward compatibility without requiring full test rewrites.

5. **Parity Testing**: Create comprehensive parity tests that verify:
   - Module discovery by `NativeModuleLoader`
   - Function naming conventions
   - Target presence in Hamilton graph
   - Domain disjointness (ingestion vs analytics vs graphs)

---

## Advanced Hamilton Features Integration

This section documents advanced Hamilton features that will be leveraged to maximize the Hamilton-first architecture, improving functionality, extensibility, robustness, and maintainability.

### Feature 1: Hamilton-Native Data Validation ✅ IMPLEMENTED

**Original Approach:** Custom `ContractEnforcementHook` for schema validation after materialization.

**Implemented Approach:** Hamilton-native custom validators using `@check_output_custom` with `BaseDefaultValidator` subclasses, making the DAG engine the authoritative source for validation without external Pandera dependencies in the execution path.

> **Design Decision**: We chose Hamilton-native validators over Pandera `@check_output` integration. This keeps validation logic within Hamilton's framework and eliminates the need to maintain parallel Pandera schemas.

**Benefits:**
- Per-node validation (catches errors earlier in the DAG)
- Automatic validation before downstream consumption
- DAG engine is the single source of truth for validation
- No external schema dependencies in execution path
- Validators are composable and reusable
- Gradual migration from existing Pandera schemas via migration bridge

**Implementation (Actual):**

```python
from hamilton.function_modifiers import check_output_custom, tag
from codeintel.build.hamilton.validators import (
    build_table_contract,
    build_key_column_contract,
)

@tag(domain="analytics", target="function_metrics", node_type="compute")
@check_output_custom(*build_table_contract(
    required_columns=["function_goid_h128", "repo", "commit", "loc"],
    column_types={"loc": "int64", "complexity": "int64"},
    non_null_columns=["function_goid_h128", "repo", "commit"],
))
@check_output_custom(*build_key_column_contract(
    key_columns=["function_goid_h128", "repo", "commit"],
))
def t__function_metrics__compute(
    q__core__goids: ir.Table,
    q__core__ast_nodes: ir.Table,
) -> pd.DataFrame:
    """Compute function metrics with Hamilton-native validation."""
    return compute_metrics(q__core__goids, q__core__ast_nodes)
```

**Files Created:**
- `src/codeintel/build/hamilton/validators/__init__.py`
- `src/codeintel/build/hamilton/validators/dataframe.py`
- `src/codeintel/build/hamilton/validators/contracts.py`
- `src/codeintel/build/hamilton/validators/migration.py`
- `src/codeintel/build/hamilton/schema_docs.py`

**Phase Integration:** ✅ Implemented in Phase 1 as PR-100.5

---

### Feature 2: @extract_fields for Multi-Output Nodes

**Current Approach:** Return tuples or dataclasses from compute nodes, unpack in materialize node.

**Enhanced Approach:** Use `@extract_fields` to split multi-table outputs into separate addressable nodes.

**Benefits:**
- Individual nodes for each output table (better lineage)
- Each output can be independently cached
- Downstream nodes can depend on specific outputs, not the whole result
- DAG visualization shows all outputs explicitly

**Implementation:**

```python
from hamilton.function_modifiers import extract_fields, tag
from typing import TypedDict

class ProfilesOutput(TypedDict):
    function_profile: pd.DataFrame
    file_profile: pd.DataFrame
    module_profile: pd.DataFrame

@tag(domain="analytics", target="profiles", node_type="compute")
@extract_fields(
    {"function_profile": pd.DataFrame, "file_profile": pd.DataFrame, "module_profile": pd.DataFrame}
)
def t__profiles__compute(
    q__analytics__function_metrics: ir.Table,
    q__core__modules: ir.Table,
) -> ProfilesOutput:
    """Compute all profile types - each becomes a separate node."""
    return {
        "function_profile": compute_function_profiles(...),
        "file_profile": compute_file_profiles(...),
        "module_profile": compute_module_profiles(...),
    }

# Now downstream can depend on individual outputs:
@tag(domain="analytics", target="subsystems", node_type="compute")
def t__subsystems__compute(
    function_profile: pd.DataFrame,  # Depends on extracted field directly
    module_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Compute subsystems from profiles."""
    ...
```

**Phase Integration:** Update all multi-output targets in Phases 2-5 to use @extract_fields

---

### Feature 3: @parameterize for Target Variants

**Current Approach:** Separate function definitions for similar targets.

**Enhanced Approach:** Use `@parameterize` to generate multiple nodes from a single template.

**Benefits:**
- DRY code for similar targets
- Centralized logic with parameterized variations
- Easier maintenance when shared logic changes
- Dynamic docstrings with parameter substitution

**Implementation (Export Domain):**

```python
from hamilton.function_modifiers import parameterize, source, value, tag

@parameterize(
    t__export_jsonl={"format": value("jsonl"), "extension": value(".jsonl")},
    t__export_parquet={"format": value("parquet"), "extension": value(".parquet")},
    t__export_csv={"format": value("csv"), "extension": value(".csv")},
    t__export_arrow={"format": value("arrow"), "extension": value(".arrow")},
)
@tag(domain="export", node_type="materialize")
def export_target(
    env: BuildEnv,
    target_table: str,
    format: str,
    extension: str,
) -> TargetRunRecord:
    """Export {target_table} to {format} format.
    
    This generates t__export_jsonl, t__export_parquet, t__export_csv, t__export_arrow.
    """
    executor = NativeTargetExecutor.for_target(env, graph, f"export_{format}")
    if executor.should_skip():
        return executor.skip()
    return executor.execute(lambda: export_table(env, target_table, format, extension))
```

**Implementation (Graph Views):**

```python
@parameterize(
    t__call_graph_view_callers={"view_type": value("callers")},
    t__call_graph_view_callees={"view_type": value("callees")},
    t__call_graph_view_transitive={"view_type": value("transitive")},
)
@tag(domain="graphs", node_type="compute")
def call_graph_view(
    t__call_graph: TargetRunRecord,
    q__graph__call_graph_edges: ir.Table,
    view_type: str,
) -> ir.Table:
    """Compute {view_type} view of call graph."""
    return compute_view(q__graph__call_graph_edges, view_type)
```

**Phase Integration:** Add PR-146.5: "Consolidate export targets with @parameterize"

---

### Feature 4: @config.when for Conditional Implementations

**Current Approach:** Manual configuration checks in code.

**Enhanced Approach:** Use `@config.when` for environment-specific or feature-flag implementations.

**Benefits:**
- Clean separation of implementations
- DAG structure adapts to configuration at build time
- No runtime conditionals in compute nodes
- Easy A/B testing of implementations

**Implementation:**

```python
from hamilton.function_modifiers import config, tag

# Development implementation with extra logging
@config.when(environment="development")
@tag(domain="ingestion", target="scip", node_type="compute")
def t__scip__index__development(env: BuildEnv) -> ScipIndexResult:
    """SCIP indexing with verbose debug output."""
    log.info("Running SCIP in development mode with full diagnostics")
    return run_scip_with_diagnostics(env)

# Production implementation optimized for speed
@config.when(environment="production")
@tag(domain="ingestion", target="scip", node_type="compute")
def t__scip__index__production(env: BuildEnv) -> ScipIndexResult:
    """SCIP indexing optimized for production."""
    return run_scip_optimized(env)

# Fallback for other environments
@config.when_not_in(environment=["development", "production"])
@tag(domain="ingestion", target="scip", node_type="compute")
def t__scip__index__default(env: BuildEnv) -> ScipIndexResult:
    """Default SCIP indexing."""
    return run_scip_standard(env)

# At driver construction:
driver = (
    driver.Builder()
    .with_modules(scip_module)
    .with_config({"environment": os.getenv("CODEINTEL_ENV", "development")})
    .build()
)
```

**Use Cases:**
- `@config.when(strict_mode=True)` for strict vs lenient validation
- `@config.when(experimental=True)` for feature flags
- `@config.when(backend="duckdb")` vs `@config.when(backend="postgres")` for storage backends

**Phase Integration:** Add to Phase 1 as configuration pattern; apply throughout Phases 2-5

---

### Feature 5: @datasaver for Standardized I/O with Metadata

**Current Approach:** `materialize_table()` utility function.

**Enhanced Approach:** Use Hamilton's `@datasaver` decorator for standardized persistence with automatic metadata.

**Benefits:**
- Automatic metadata capture (row counts, schema, timestamps)
- Consistent I/O interface across all targets
- Built-in support for Hamilton's caching system
- Lineage tracking integration

**Implementation:**

```python
from hamilton.function_modifiers import datasaver, tag
from hamilton.io import utils as io_utils

@datasaver()
@tag(domain="analytics", target="function_metrics", node_type="persist")
def persist_function_metrics(
    env: BuildEnv,
    function_metrics_df: pd.DataFrame,
) -> dict:
    """Persist function metrics to DuckDB with metadata tracking."""
    table_key = "analytics.function_metrics"
    
    # Write to DuckDB
    row_count = env.gateway.write_table(table_key, function_metrics_df)
    
    # Return Hamilton I/O metadata
    return {
        "table_key": table_key,
        "row_count": row_count,
        "schema": list(function_metrics_df.columns),
        "dtypes": function_metrics_df.dtypes.to_dict(),
        **io_utils.get_dataframe_metadata(function_metrics_df),
    }
```

**Phase Integration:** Update `materialize_table()` to be a `@datasaver` wrapper in Phase 1

---

### Feature 6: Builder.with_materializers() for Dynamic Export

**Current Approach:** Static export target modules.

**Enhanced Approach:** Use `Builder.with_materializers()` for flexible, runtime-configurable export.

**Benefits:**
- Export configuration without code changes
- Dynamic format selection at runtime
- Multiple simultaneous export destinations
- Easy integration with external systems (S3, GCS, etc.)

**Implementation:**

```python
from hamilton.io import materialization as mat

def build_driver_with_exports(
    modules: list[ModuleType],
    export_config: dict,
) -> driver.Driver:
    """Build driver with dynamic export materializers."""
    
    materializers = []
    
    # Add exports based on configuration
    if export_config.get("export_parquet"):
        for table in export_config["tables"]:
            materializers.append(
                mat.to.parquet(
                    id=f"{table}__parquet",
                    dependencies=[f"q__{table.replace('.', '__')}"],
                    path=f"{export_config['output_dir']}/{table}.parquet",
                )
            )
    
    if export_config.get("export_jsonl"):
        for table in export_config["tables"]:
            materializers.append(
                mat.to.json(
                    id=f"{table}__jsonl",
                    dependencies=[f"q__{table.replace('.', '__')}"],
                    path=f"{export_config['output_dir']}/{table}.jsonl",
                    orient="records",
                    lines=True,
                )
            )
    
    return (
        driver.Builder()
        .with_modules(*modules)
        .with_materializers(*materializers)
        .build()
    )
```

**Phase Integration:** Add to Phase 5 as alternative export mechanism

---

### Feature 7: Parallel Execution with Graph Adapters ✅ IMPLEMENTED

**Original Approach:** Sequential execution of all nodes.

**Implemented Approach:** `ThreadPoolAdapter` wrapping Hamilton's `FutureAdapter`, with CLI flags and environment variable configuration for easy control.

**Benefits:**
- Parallel execution of independent targets (e.g., AST + CST simultaneously)
- Configurable via CLI (`--parallel-backend`, `--max-workers`)
- Configurable via environment variables (`CODEINTEL_PARALLEL_BACKEND`, `CODEINTEL_MAX_WORKERS`)
- Auto-detection mode that enables parallelism for multi-target builds
- No code changes to node functions
- Significant speedup for I/O-bound operations

**Implementation (Actual):**

```python
from codeintel.build.hamilton.adapters import (
    ExecutionBackend,
    ParallelConfig,
    ThreadPoolAdapter,
    create_parallel_adapter,
)

# Create configuration from CLI args or environment
config = ParallelConfig(
    backend=ExecutionBackend.THREADPOOL,
    max_workers=4,
)

# Or use factory with auto-detection
config = ParallelConfig.from_args(
    parallel_backend="auto",
    max_workers=None,  # Will use cpu_count()
)

# Create adapter
adapter = create_parallel_adapter(config)

# Use in driver
dr = (
    driver.Builder()
    .with_modules(*modules)
    .with_adapter(adapter)
    .build()
)
```

**CLI Usage:**

```bash
# Enable threadpool parallelism
codeintel build run --parallel-backend threadpool --max-workers 4

# Auto-detect (uses threadpool if multiple targets)
codeintel build run --parallel-backend auto

# With progress bar
codeintel build run --parallel-backend threadpool --progress
```

**Environment Variable Configuration:**

```bash
export CODEINTEL_PARALLEL_BACKEND=threadpool
export CODEINTEL_MAX_WORKERS=8
```

**ExecutionBackend Enum:**
- `SEQUENTIAL` - Run nodes one at a time
- `THREADPOOL` - Use concurrent.futures ThreadPoolExecutor
- `AUTO` - Detect based on target count and environment

**Files Created:**
- `src/codeintel/build/hamilton/adapters/__init__.py`
- `src/codeintel/build/hamilton/adapters/parallel.py`

**Phase Integration:** ✅ Implemented in Phase 1 as PR-103.5

---

### Feature 8: Parallelizable/Collect for Dynamic Parallelism

**Current Approach:** Process files/modules sequentially.

**Enhanced Approach:** Use `Parallelizable[T]` and `Collect[T]` for dynamic parallel processing.

**Benefits:**
- Automatic parallelization of file-level processing
- Scales with number of files
- Works with distributed backends (Ray, Dask)
- Clean separation of fan-out and fan-in logic

**Implementation:**

```python
from hamilton.htypes import Parallelizable, Collect
from hamilton.function_modifiers import tag

@tag(domain="ingestion", target="ast", node_type="compute")
def module_paths(env: BuildEnv) -> Parallelizable[Path]:
    """Yield module paths for parallel processing."""
    modules = env.gateway.load_table("core.modules")
    for path in modules["path"]:
        yield env.repo_root / path

@tag(domain="ingestion", target="ast", node_type="compute")  
def ast_for_module(module_paths: Path, env: BuildEnv) -> AstResult:
    """Extract AST for a single module (runs in parallel)."""
    return extract_ast(module_paths, env.config)

@tag(domain="ingestion", target="ast", node_type="compute")
def combined_ast(ast_for_module: Collect[AstResult]) -> pd.DataFrame:
    """Combine all AST results into a single DataFrame."""
    return pd.concat([r.to_dataframe() for r in ast_for_module])

# Enable dynamic execution in driver:
dr = (
    driver.Builder()
    .with_modules(ast_module)
    .enable_dynamic_execution(allow_experimental_mode=True)
    .with_remote_executor(executors.MultiProcessingExecutor(max_tasks=8))
    .with_local_executor(executors.SynchronousLocalTaskExecutor())
    .build()
)
```

**Phase Integration:** Add to Phase 2 for ingestion targets that process files (AST, CST, coverage)

---

### Feature 9: Enhanced Lifecycle Hooks ✅ IMPLEMENTED

**Original Approach:** TelemetryHook, ManifestHook, ContractHook.

**Implemented Approach:** Added `ProgressBarHook` (tqdm integration), `BuildTimingHook` (per-node timing), and `ConditionalHook` (hook enable/disable based on conditions).

**Benefits:**
- Real-time progress visualization via tqdm
- Per-node execution timing for performance analysis
- Conditional hook activation (e.g., disable progress in CI)
- Foundation for future OpenLineage integration

**Implementation (Actual):**

```python
from codeintel.build.hamilton.hooks import (
    ProgressBarHook,
    BuildTimingHook,
    ConditionalHook,
    create_progress_hook,
    build_hooks,
)

# Use factory with automatic CI detection
progress_hook = create_progress_hook(desc="Building targets")

# Manual timing hook
timing_hook = BuildTimingHook()

# Conditional hook (only enabled in debug mode)
debug_timing = ConditionalHook(
    BuildTimingHook(),
    enabled=os.getenv("DEBUG") == "1",
)

# Updated build_hooks() function
hooks = build_hooks(
    manifest_index=manifest_index,
    telemetry_enabled=True,
    enable_progress=True,   # NEW
    enable_timing=True,     # NEW
)

# After execution, access timing data:
timing_hook.timings  # dict[str, NodeTimingRecord]
timing_hook.total_duration_seconds()
timing_hook.slowest_nodes(n=10)
```

**Files Created:**
- `src/codeintel/build/hamilton/hooks/lifecycle.py`

**Phase Integration:** ✅ Implemented in Phase 1 as PR-100.6

---

### Feature 10: Hamilton UI Integration

**Current Approach:** CLI-only interface.

**Enhanced Approach:** Integrate Hamilton UI for visualization and monitoring.

**Benefits:**
- Interactive DAG visualization
- Real-time execution monitoring
- Node catalog with documentation
- Execution history and comparison
- Team collaboration features

**Implementation:**

```python
# Install with: pip install "sf-hamilton[ui,sdk]"

# CLI usage:
# hamilton ui native/ingestion/ native/analytics/ native/graphs/

# Programmatic usage:
from hamilton_sdk import HamiltonTracker

tracker = HamiltonTracker(
    project_id=42,
    api_key=os.getenv("HAMILTON_API_KEY"),
    username="codeintel-build",
    dag_name="codeintel-build-dag",
    tags={"environment": "production", "repo": env.repo},
)

dr = (
    driver.Builder()
    .with_modules(*modules)
    .with_adapters(tracker)  # Send execution data to UI
    .build()
)
```

**Phase Integration:** Add to Phase 6 as PR-160.5: "Hamilton UI integration"

---

### Feature 11: @schema for Column Metadata

**Current Approach:** Schema information in separate Pandera definitions.

**Enhanced Approach:** Use `@schema.output()` to document column types directly on nodes.

**Benefits:**
- Schema documentation co-located with code
- Visible in Hamilton UI and visualizations
- Self-documenting DAG nodes
- Lightweight metadata without validation overhead

**Implementation:**

```python
from hamilton.function_modifiers import schema, tag

@tag(domain="analytics", target="function_metrics", node_type="compute")
@schema.output(
    ("function_goid_h128", "string"),
    ("repo", "string"),
    ("commit", "string"),
    ("loc", "int"),
    ("complexity", "int"),
    ("parameter_count", "int"),
    ("return_count", "int"),
    ("has_docstring", "bool"),
)
def t__function_metrics__compute(
    q__core__goids: ir.Table,
    q__core__ast_nodes: ir.Table,
) -> pd.DataFrame:
    """Compute function-level metrics."""
    ...
```

**Phase Integration:** Apply to all compute nodes during Phases 2-5

---

### Feature 12: @pipe_input for Transformation Chains

**Current Approach:** Explicit intermediate nodes for data transformations.

**Enhanced Approach:** Use `@pipe_input` to chain transformations while maintaining DAG visibility.

**Benefits:**
- Each transformation step is a visible node
- Intermediate results can be inspected/debugged
- Clean representation of data cleaning pipelines
- Maintains single responsibility per function

**Implementation:**

```python
from hamilton.function_modifiers import pipe_input, step, tag

def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with null goids."""
    return df.dropna(subset=["function_goid_h128"])

def normalize_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize file paths to repo-relative."""
    df["path"] = df["path"].str.replace(r"^/.*?/", "", regex=True)
    return df

def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns."""
    df["complexity_per_loc"] = df["complexity"] / df["loc"].clip(lower=1)
    return df

@tag(domain="analytics", target="function_metrics", node_type="compute")
@pipe_input(
    step(clean_nulls),
    step(normalize_paths),
    step(add_computed_columns),
    on_input="raw_metrics",
)
def t__function_metrics__compute(
    raw_metrics: pd.DataFrame,  # After pipe transformations
) -> pd.DataFrame:
    """Final function metrics after cleaning pipeline."""
    return raw_metrics
```

**Phase Integration:** Apply to complex data cleaning in analytics targets

---

### Feature 13: Hamilton Caching Integration

**Current Approach:** Custom ManifestHook with input/options hashing.

**Enhanced Approach:** Layer Hamilton's native caching alongside our manifest system.

**Benefits:**
- Automatic code version tracking (Hamilton hashes function source)
- Per-node cache granularity
- Multiple cache backends (disk, memory, cloud)
- Integrates with @cache decorator for format control

**Implementation:**

```python
from hamilton.function_modifiers import cache, tag

# Mark compute-heavy nodes for caching
@tag(domain="analytics", target="risk_factors", node_type="compute")
@cache(format="parquet")  # Cache as parquet for efficient reload
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
    q__graph__call_graph_edges: ir.Table,
) -> pd.DataFrame:
    """Risk factor computation (expensive, cache results)."""
    ...

# Enable caching at driver level:
dr = (
    driver.Builder()
    .with_modules(*modules)
    .with_cache()  # Enable Hamilton caching
    .build()
)

# Check cache behavior:
print(dr.cache.behaviors)  # Shows which nodes were cached vs recomputed
```

**Integration Strategy:**
- Use Hamilton caching for compute-intensive intermediate nodes
- Keep ManifestHook for target-level skip logic (cross-session persistence)
- Hamilton cache provides within-session optimization

**Phase Integration:** Add to Phase 1 as optional caching layer; apply to heavy compute in Phases 3-4

---

### Feature 14: Custom ResultBuilder for Build Results ✅ IMPLEMENTED

**Original Approach:** Return dict of TargetRunRecords.

**Implemented Approach:** Custom `BuildResultBuilder` that aggregates node outputs into a structured `BuildExecutionResult` with status tracking, timing, and summary generation.

**Benefits:**
- Consistent build result format with `ResultStatus` enum
- Automatic aggregation with `succeeded_nodes`, `failed_nodes`, `skipped_nodes`
- Rich timing information with `duration_seconds`
- Human-readable `summary()` method
- Integration with downstream systems via structured data

**Implementation (Actual):**

```python
from codeintel.build.hamilton.result_builder import (
    BuildResultBuilder,
    BuildExecutionResult,
    NodeResult,
    ResultStatus,
    DictResultBuilder,
)

# Use in driver:
from hamilton import base
adapter = base.SimplePythonGraphAdapter(result_builder=BuildResultBuilder())
dr = driver.Builder().with_modules(*modules).with_adapter(adapter).build()

result: BuildExecutionResult = dr.execute(["t__risk_factors", "t__hotspots"])
print(result.summary())
# Output:
# Build Execution: SUCCESS
# Duration: 12.34s
# Succeeded: 5
# Failed: 0
# Skipped: 2

# Access individual node results
for name, node_result in result.nodes.items():
    print(f"{name}: {node_result.status}")
```

**ResultStatus Enum:**
- `SUCCESS` - All nodes succeeded
- `FAILURE` - One or more nodes failed
- `SKIPPED` - Node was skipped (inputs unchanged)
- `PARTIAL` - Mix of success and failure

**Files Created:**
- `src/codeintel/build/hamilton/result_builder.py`

**Phase Integration:** ✅ Implemented in Phase 1 as PR-101.5

---

### Feature 15: Hamilton CLI Integration for CI/CD

**Current Approach:** Custom CLI commands.

**Enhanced Approach:** Integrate Hamilton CLI for DAG operations.

**Benefits:**
- `hamilton diff` for change detection between commits
- `hamilton version` for DAG versioning
- `hamilton build` for visualization generation
- Standard tooling for CI/CD pipelines

**Implementation:**

```bash
# CI/CD pipeline integration:

# 1. Verify DAG structure before deployment
hamilton build native/ingestion/*.py native/analytics/*.py --output dag.png

# 2. Detect DAG changes between commits
hamilton diff native/ --from-commit $PREV_COMMIT --to-commit $CURR_COMMIT

# 3. Compute DAG version hash for cache invalidation
DAG_VERSION=$(hamilton version native/)
echo "DAG_VERSION=$DAG_VERSION" >> $GITHUB_ENV

# 4. Launch UI for debugging
hamilton ui native/ --port 8080 &
```

**Phase Integration:** Add to Phase 6 as CI/CD integration

---

### Feature 16: @parameterized_subdag for Multi-Repo Builds

**Current Approach:** Single-repo build with explicit configuration.

**Enhanced Approach:** Use `@parameterized_subdag` for multi-repo or multi-snapshot builds.

**Benefits:**
- Same build logic applied to multiple repositories
- Parallel execution across repos
- Shared intermediate results where appropriate
- Clean separation of per-repo configuration

**Implementation:**

```python
from hamilton.function_modifiers import parameterized_subdag, value

# Define core analytics as reusable module
import native.analytics as analytics_module

@parameterized_subdag(
    analytics_module,
    repo_a={
        "inputs": {"env": value(create_env("org/repo-a", "main"))},
        "config": {"profile": "default"},
    },
    repo_b={
        "inputs": {"env": value(create_env("org/repo-b", "main"))},
        "config": {"profile": "strict"},
    },
    repo_c={
        "inputs": {"env": value(create_env("org/repo-c", "develop"))},
        "config": {"profile": "default"},
    },
)
def multi_repo_analytics(risk_factors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk factors from multiple repos."""
    return risk_factors

# Each repo gets its own sub-DAG execution
# Results are collected and aggregated
```

**Phase Integration:** Add to Phase 6 as advanced multi-repo capability

---

### Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BUILD SYSTEM (ENHANCED HAMILTON-FIRST)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hamilton Driver                                 │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              Advanced Decorators Layer                       │    │   │
│  │  │  @check_output(schema=Pandera)   @extract_fields            │    │   │
│  │  │  @parameterize(variants...)      @config.when(env=...)      │    │   │
│  │  │  @cache(format="parquet")        @datasaver()               │    │   │
│  │  │  @schema.output(cols...)         @pipe_input(steps...)      │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │t__modules│ │t__scip   │ │t__ast    │ │t__goids  │ │t__metrics│  │   │
│  │  │  (root)  │→│(external)│→│(parallel)│→│(compute) │→│(validated│  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       │            │            │            │            │         │   │
│  │  ┌────┴────────────┴────────────┴────────────┴────────────┴────┐   │   │
│  │  │                    Parallelizable/Collect                    │   │   │
│  │  │         (Dynamic fan-out for file-level processing)         │   │   │
│  │  └──────────────────────────────┬──────────────────────────────┘   │   │
│  │                                 │                                   │   │
│  │                           ┌─────▼─────┐                             │   │
│  │                           │  BuildEnv │                             │   │
│  │                           └───────────┘                             │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                     Hamilton Hooks Stack                     │    │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │    │   │
│  │  │  │ManifestHook │ │TelemetryHook│ │ContractHook │            │    │   │
│  │  │  │(skip logic) │ │(spans/logs) │ │(validation) │            │    │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘            │    │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │    │   │
│  │  │  │ProgressBar  │ │OpenLineage  │ │ResultBuilder│            │    │   │
│  │  │  │  (tqdm)     │ │ (lineage)   │ │(BuildReport)│            │    │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘            │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                   Execution Adapters                         │    │   │
│  │  │  FutureAdapter (ThreadPool) │ DaskGraphAdapter (Cluster)    │    │   │
│  │  │  RayGraphAdapter (Dist.)    │ CachingGraphAdapter           │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                 Builder.with_materializers()                 │    │   │
│  │  │  mat.to.parquet()  mat.to.json()  mat.to.csv()              │    │   │
│  │  │  (Dynamic export without code changes)                       │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Hamilton Tooling                              │  │
│  │  hamilton ui (visualization)  │  hamilton diff (CI/CD)              │  │
│  │  hamilton version (hashing)   │  VSCode Extension (IDE)             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Updated Phase 1 PRs (Foundation) ✅ ALL COMPLETE

| PR | Description | Dependencies | Risk | Status |
|----|-------------|--------------|------|--------|
| PR-100 | Consolidate hooks into `hamilton/hooks/` | None | Low | ✅ |
| PR-100.5 | Hamilton-native validators (NOT Pandera @check_output) | PR-100 | Low | ✅ |
| PR-100.6 | Extended lifecycle hooks (ProgressBar, Timing) | PR-100 | Low | ✅ |
| PR-100.7 | Migration bridge from Pandera schemas | PR-100.5 | Low | ✅ |
| PR-101 | Unified `NativeTargetExecutor` with async support | PR-100 | Low | ✅ |
| PR-101.5 | Custom `BuildResultBuilder` | PR-101 | Low | ✅ |
| PR-102 | Native module loader with validation | PR-101 | Low | ✅ |
| PR-103 | `--native-only` flag to driver/CLI | PR-102 | Low | ✅ |
| PR-103.5 | Parallel execution adapters (ThreadPool) | PR-103 | Medium | ✅ |
| PR-104 | Migration test harness | PR-103 | Low | ✅ |

> **Note**: PR-100.5 was changed from "Pandera @check_output" to "Hamilton-native validators" to make the DAG engine the authoritative source for validation.

### Phase 1.5 PR (Validation POC) ✅ COMPLETE

| PR | Description | Dependencies | Risk | Status |
|----|-------------|--------------|------|--------|
| PR-104.5 | Apply validators to existing native targets | PR-104 | Low | ✅ Complete |

> **Purpose**: Validates the Hamilton-native validation infrastructure works end-to-end before rolling out across 45+ targets in Phases 2-6.

---

### Updated Success Criteria

**Phase 1 (Complete):**

- [x] **Hooks consolidated** into `hamilton/hooks/` directory
- [x] **Hamilton-native validators** implemented (`@check_output_custom` with custom validators)
- [x] **Schema documentation** via `@schema.output` utilities
- [x] **Migration bridge** from Pandera schemas to Hamilton validators
- [x] **Lifecycle hooks** added (ProgressBar, Timing, Conditional)
- [x] **Parallel execution** enabled via `ThreadPoolAdapter`
- [x] **CLI flags** for parallel backend and progress
- [x] **Custom ResultBuilder** returns structured `BuildExecutionResult`
- [x] **Native module loader** for discovering Hamilton modules
- [x] **Migration test harness** with parity and skip logic tests

**Phase 1.5 (Complete):**

- [x] `t__risk_factors` uses `@check_output_custom` validators
- [x] `t__hotspots` uses `@check_output_custom` validators
- [x] Both targets have `@schema.output` documentation
- [x] `ContractEnforcementHook` reports validation results
- [x] `build_hooks()` enables validation by default
- [x] Validation patterns documented for Phase 2-6 migrations

**Phase 2 (Complete):**

- [x] **All ingestion targets** migrated to native Hamilton modules
- [x] **Ingestion plugins** deleted
- [x] **Parity tests** verify module discovery and target presence
- [x] **Target naming conventions** established (`_ingest` suffix for disambiguation)
- [x] **Registration pattern** updated (native_module only, no plugin)

> **Note**: Validators (`@check_output_custom`, `@schema.output`) were NOT applied to ingestion compute nodes because they return custom dataclasses, not DataFrames or Ibis tables. This is the expected pattern for ingestion targets.

**Phases 3-5 (Pending):**

- [ ] **All compute nodes** (that return DataFrames/Ibis tables) use `@check_output_custom` with Hamilton validators
- [ ] **All compute nodes** (that return DataFrames/Ibis tables) have `@schema.output` documentation
- [ ] **Multi-output targets** use `@extract_fields` for lineage
- [ ] **Similar targets** consolidated with `@parameterize`
- [ ] **Environment variants** use `@config.when`
- [ ] **I/O nodes** use `@datasaver` with metadata
- [ ] **Parallelizable/Collect** used for file-level parallelism

**Phase 6 (Pending):**

- [ ] **SCHEMA_REGISTRY Pandera dependencies** removed
- [ ] **Hamilton validators** are authoritative source for validation
- [ ] **Hamilton UI** deployed for DAG visualization
- [ ] **OpenLineage** integrated for lineage tracking
- [ ] **Hamilton CLI** integrated in CI/CD pipelines

---

## Architectural Vision

### Before: Dual Execution Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BUILD SYSTEM (CURRENT)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hamilton Driver                                 │   │
│  │  ┌───────────────────────┐    ┌───────────────────────────────────┐│   │
│  │  │ Native Hamilton Path  │    │ Plugin Wrapper Path               ││   │
│  │  │                       │    │                                   ││   │
│  │  │ t__risk_factors       │    │ t__function_metrics (wrapper)     ││   │
│  │  │   ↓                   │    │   ↓                               ││   │
│  │  │ BuildEnv              │    │ Creates TargetExecutionContext    ││   │
│  │  │   ↓                   │    │   ↓                               ││   │
│  │  │ NativeTargetExecutor  │    │ Calls TargetPlugin.execute()      ││   │
│  │  │   ↓                   │    │   ↓                               ││   │
│  │  │ materialize_table()   │    │ ctx.write_table() (manual)        ││   │
│  │  │   ↓                   │    │   ↓                               ││   │
│  │  │ TargetRunRecord       │    │ TargetResult → TargetRunRecord    ││   │
│  │  └───────────────────────┘    └───────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐   │
│  │ Context Types (6)   │  │ Plugin Classes (45) │  │ Registries (3)   │   │
│  │ - BuildContext      │  │ - RepoScanPlugin    │  │ - registry.py    │   │
│  │ - ExecutionContext  │  │ - AstExtractPlugin  │  │ - unified_reg.py │   │
│  │ - TargetExecContext │  │ - ScipPlugin        │  │ - registrations  │   │
│  │ - MaterializContext │  │ - FunctionMetrics   │  │                  │   │
│  │ - BuildEnv          │  │ - ...               │  │                  │   │
│  │ - _RunContext       │  │                     │  │                  │   │
│  └─────────────────────┘  └─────────────────────┘  └──────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### After: 100% Native Hamilton

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUILD SYSTEM (TARGET STATE)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hamilton Driver                                 │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │t__modules│ │t__scip   │ │t__ast    │ │t__goids  │ │t__metrics│  │   │
│  │  │t__cst    │ │t__typing │ │t__tests  │ │t__cfg_dfg│ │t__risk   │  │   │
│  │  │t__import │ │t__call   │ │t__profile│ │t__hotspot│ │t__export │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       │            │            │            │            │         │   │
│  │       └────────────┴────────────┴────────────┴────────────┘         │   │
│  │                              │                                       │   │
│  │                        ┌─────▼─────┐                                 │   │
│  │                        │  BuildEnv │  ← SINGLE CONTEXT               │   │
│  │                        └───────────┘                                 │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                     Hamilton Hooks                           │    │   │
│  │  │  ManifestHook: Skip logic, manifest persistence              │    │   │
│  │  │  TelemetryHook: Spans, metrics, logging                      │    │   │
│  │  │  ContractHook: Schema validation                             │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────┐  ┌────────────────────────────────────────────┐  │
│  │ Context Types (1)   │  │ Hamilton Native Modules (45)               │  │
│  │ - BuildEnv          │  │ native/ingestion/*.py                      │  │
│  │                     │  │ native/graphs/*.py                         │  │
│  │                     │  │ native/analytics/*.py                      │  │
│  │                     │  │ native/export/*.py                         │  │
│  └─────────────────────┘  └────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Current State Analysis

### Files to Delete

| File | Lines | Reason |
|------|-------|--------|
| `plugin.py` | 425 | Plugin abstraction eliminated |
| `context.py` | 582 | TargetExecutionContext eliminated |
| `context_base.py` | 605 | Context hierarchy eliminated |
| `unified_registry.py` | 461 | Plugin tracking eliminated |
| `registrations.py` | 333 | Plugin registration eliminated |
| `resources.py` | 177 | Merged into BuildEnv |
| `run_config.py` | 66 | Merged into BuildEnv |
| `result.py` | 92 | Replaced by TargetRunRecord |
| **Total** | **2,741** | |

### Files to Significantly Simplify

| File | Current Lines | Target Lines | Change |
|------|---------------|--------------|--------|
| `registry.py` | 750 | ~250 | -500 (remove static constants) |
| `targets.py` | 484 | ~300 | -184 (simplify TargetGraph) |
| `node_factory.py` | 828 | ~300 | -528 (no wrapper generation) |
| `executor.py` | 566 | ~400 | -166 (single path) |
| `state.py` | 147 | 0 | -147 (merge into state_computer) |
| **Total** | | | **-1,525** |

### Files to Retain (With Updates)

| File | Lines | Updates Needed |
|------|-------|----------------|
| `contracts.py` | 297 | Minor: remove plugin references |
| `manifest.py` | 172 | None |
| `hashing.py` | 198 | Minor: single code path |
| `session.py` | 226 | Minor: simplified interface |
| `state_types.py` | 415 | None |
| `state_computer.py` | 415 | None |
| `protocols.py` | 333 | None (DI for external tools) |
| `providers.py` | 1,070 | None (implementations) |
| `types.py` | 343 | None (ToolRunResult, etc.) |
| `errors.py` | 854 | Minor: remove plugin errors |
| `config.py` | 359 | None |
| `parameters.py` | 231 | Minor: simplify |

### Plugin Migration Inventory

#### Ingestion Domain (12 plugins → 12 modules)

| Plugin | File | Complexity | Dependencies |
|--------|------|------------|--------------|
| `RepoScanPlugin` | `repo_scan.py` | High | Filesystem, change detection |
| `AstExtractPlugin` | `ast_extract.py` | Medium | modules |
| `CstExtractPlugin` | `cst_extract.py` | Medium | modules |
| `ScipPlugin` | `scip_plugin.py` | High | External binary, modules |
| `TypingPlugin` | `typing_plugin.py` | Medium | scip |
| `TestsPlugin` | `tests_plugin.py` | Medium | modules |
| `DocsPlugin` | `docstrings_plugin.py` | Medium | modules, ast |
| `CoveragePlugin` | `coverage_plugin.py` | Medium | External pytest-cov |
| `ConfigPlugin` | `config_plugin.py` | Low | modules |

#### Graphs Domain (8 plugins → 8 modules)

| Plugin | File | Complexity | Dependencies |
|--------|------|------------|--------------|
| `GoidPlugin` | `goid.py` | Medium | scip |
| `CallGraphPlugin` | `callgraph.py` | Medium | goids |
| `ImportGraphPlugin` | `import_graph.py` | Medium | modules |
| `SymbolUsesPlugin` | `symbol_uses.py` | Medium | scip |
| `CfgDfgPlugin` | `cfg_dfg.py` | High | ast, goids |
| `GraphMetricsPlugin` | `core.py` | Low | call_graph |
| `ValidationPlugin` | `validation.py` | Low | various |

#### Analytics Domain (20+ plugins → 20+ modules)

| Plugin | File | Complexity | Dependencies |
|--------|------|------------|--------------|
| `FunctionMetricsPlugin` | `metrics.py` | Medium | goids |
| `AstFeaturesPlugin` | `ast_features.py` | Medium | ast, goids |
| `RiskFactorsPlugin` | `factors.py` | Medium | function_metrics, call_graph |
| `HotspotsPlugin` | `build.py` | Medium | risk_factors |
| `ProfilesPlugin` | `build.py` | High | multiple |
| `TestProfilePlugin` | `profile.py` | Medium | tests, coverage |
| `SubsystemsPlugin` | `build.py` | High | multiple |
| `CoverageEdgesPlugin` | `test_edges.py` | Medium | coverage |
| ... | ... | ... | ... |

#### Export Domain (2 plugins → 2 modules)

| Plugin | File | Complexity | Dependencies |
|--------|------|------------|--------------|
| `ExportJsonlPlugin` | (generated) | Low | any target |
| `ExportParquetPlugin` | (generated) | Low | any target |

---

## Target State Architecture

### Directory Structure

```
src/codeintel/build/
├── __init__.py                    # Public API (simplified)
├── contracts.py                   # OutputContract, ArtifactSpec
├── manifest.py                    # OutputManifest, BuildRunRecord
├── targets.py                     # OutputTarget, TargetGraph (metadata only)
├── registry.py                    # Target metadata registry (no plugins)
├── hashing.py                     # Input hash computation
├── session.py                     # Session-scoped caching
├── errors.py                      # Build error hierarchy
├── types.py                       # Shared types (ToolRunResult, etc.)
├── config.py                      # BuildConfig
├── parameters.py                  # Target parameters
├── protocols.py                   # DI protocols (ToolRunner, etc.)
├── providers.py                   # DI implementations
├── state_types.py                 # BuildState, TargetState
├── state_computer.py              # State computation
│
├── hamilton/
│   ├── __init__.py
│   ├── env.py                     # BuildEnv (THE ONE CONTEXT)
│   ├── driver_factory.py          # build_driver(), get_target_graph()
│   ├── naming.py                  # Node naming conventions
│   ├── introspect.py              # DAG introspection
│   ├── planner.py                 # Build planning
│   ├── tags.py                    # Hamilton tag constants
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── manifest_hook.py       # Skip logic + manifest persistence
│   │   ├── telemetry_hook.py      # Observability
│   │   └── contract_hook.py       # Schema validation
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── dataset_ref.py         # DatasetRef for provenance
│   │   ├── artifact_ref.py        # ArtifactRef for files
│   │   └── materializer.py        # materialize_table() utility
│   │
│   └── native/
│       ├── __init__.py
│       ├── executor.py            # NativeTargetExecutor (simplified)
│       ├── outputs.py             # Output helpers
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── modules.py         # t__modules
│       │   ├── scip.py            # t__scip
│       │   ├── ast.py             # t__ast
│       │   ├── cst.py             # t__cst
│       │   ├── typing.py          # t__typing
│       │   ├── tests.py           # t__tests
│       │   ├── docstrings.py      # t__docstrings
│       │   ├── coverage.py        # t__coverage
│       │   └── config.py          # t__config
│       │
│       ├── graphs/
│       │   ├── __init__.py
│       │   ├── goids.py           # t__goids
│       │   ├── call_graph.py      # t__call_graph
│       │   ├── import_graph.py    # t__import_graph
│       │   ├── symbol_uses.py     # t__symbol_uses
│       │   ├── cfg_dfg.py         # t__cfg_dfg
│       │   └── views.py           # t__call_graph_views, etc.
│       │
│       ├── analytics/
│       │   ├── __init__.py
│       │   ├── function_metrics.py
│       │   ├── ast_features.py
│       │   ├── risk_factors.py
│       │   ├── hotspots.py
│       │   ├── profiles.py
│       │   ├── test_profile.py
│       │   ├── coverage_edges.py
│       │   ├── subsystems.py
│       │   ├── semantic_roles.py
│       │   └── ...
│       │
│       └── export/
│           ├── __init__.py
│           ├── export_jsonl.py    # t__export_jsonl
│           └── export_parquet.py  # t__export_parquet
│
└── schemas/                       # Unchanged
```

### BuildEnv: The Single Context

```python
# src/codeintel/build/hamilton/env.py

@dataclass(frozen=True)
class BuildEnv:
    """Single execution context for all Hamilton nodes.
    
    This is the ONLY context type in the system. All nodes receive
    the same BuildEnv instance, providing access to:
    - Storage (gateway)
    - Identity (snapshot)
    - Paths (build_dir, scip_dir, etc.)
    - External tools (providers)
    - Configuration (config, profile)
    - Execution control (force_targets, validate_outputs)
    - Caching (manifest_index)
    """
    
    # Core
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    
    # DI for external tools
    providers: Providers
    
    # Configuration
    config: BuildConfig
    profile: str | None = None
    
    # Execution control
    force_targets: frozenset[str] = field(default_factory=frozenset)
    validate_outputs: bool = False
    strict_contracts: bool = False
    
    # Caching
    manifest_index: Mapping[str, OutputManifest] | None = None
    
    # Convenience properties
    @property
    def repo(self) -> str:
        return self.snapshot.repo
    
    @property
    def commit(self) -> str:
        return self.snapshot.commit
    
    @property
    def repo_root(self) -> Path:
        return self.snapshot.repo_root
    
    @property
    def build_dir(self) -> Path:
        return self.paths.build_dir
```

### Hamilton Module Template

```python
# src/codeintel/build/hamilton/native/<domain>/<target>.py
"""Native Hamilton implementation for <target_name> target.

This module follows the canonical pattern:
1. Pure compute nodes (no side effects, return Ibis expressions or data)
2. Single materialize node (t__<target>) that persists and returns record
3. Dependencies expressed as function parameters
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.io.materializer import materialize_table
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Compute Nodes (Pure, No Side Effects)
# -----------------------------------------------------------------------------

@tag(domain="<domain>", target="<target_name>", node_type="compute")
def t__<target_name>__compute(
    q__<schema>__<dep_table>: ir.Table,  # Dependency tables
) -> ir.Table:
    """Pure computation returning Ibis expression.
    
    No side effects. Hamilton can cache, parallelize, optimize.
    """
    return ibis_expression


# -----------------------------------------------------------------------------
# Materialize Node (Side Effect Boundary)
# -----------------------------------------------------------------------------

@tag(domain="<domain>", target="<target_name>", node_type="materialize")
def t__<target_name>(
    env: BuildEnv,
    graph: TargetGraph,
    t__<target_name>__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize results to DuckDB.
    
    This is the only node with side effects.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "<target_name>")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: {
        "<schema>.<table>": materialize_table(
            env, "<schema>.<table>", t__<target_name>__compute
        ),
    })
```

---

## Implementation Phases

### Phase 1: Foundation (PRs 100-104) ✅ COMPLETE

**Goal**: Establish the target architecture without breaking existing functionality.

**Status**: ✅ All PRs implemented and tested. 196 tests passing.

| PR | Description | Dependencies | Risk | Status |
|----|-------------|--------------|------|--------|
| PR-100 | Consolidate hooks into `hamilton/hooks/` | None | Low | ✅ Complete |
| PR-100.5 | Hamilton-native data validation | PR-100 | Low | ✅ Complete |
| PR-100.6 | Extended lifecycle hooks | PR-100 | Low | ✅ Complete |
| PR-100.7 | Migration bridge from Pandera | PR-100.5 | Low | ✅ Complete |
| PR-101 | Unified `NativeTargetExecutor` with async | PR-100 | Low | ✅ Complete |
| PR-101.5 | Custom `BuildResultBuilder` | PR-101 | Low | ✅ Complete |
| PR-102 | Native module loader | PR-101 | Low | ✅ Complete |
| PR-103 | `--native-only` flag to driver/CLI | PR-102 | Low | ✅ Complete |
| PR-103.5 | Parallel execution adapters | PR-103 | Medium | ✅ Complete |
| PR-104 | Migration test harness | PR-103 | Low | ✅ Complete |

**Key Deliverables:**
- Hamilton hooks consolidated in `src/codeintel/build/hamilton/hooks/`
- Hamilton-native validators in `src/codeintel/build/hamilton/validators/`
- Parallel execution adapters in `src/codeintel/build/hamilton/adapters/`
- Custom result builder in `src/codeintel/build/hamilton/result_builder.py`
- Schema documentation utilities in `src/codeintel/build/hamilton/schema_docs.py`
- Native module loader in `src/codeintel/build/hamilton/native/loader.py`
- Migration test harness in `tests/build/hamilton/native/`

### Phase 1.5: Validation POC (PR-104.5) 🔜 NEXT

**Goal**: Validate Hamilton-native validation infrastructure on existing targets before full rollout.

**Status**: Pending (1-2 days, standalone before Phase 2)

| PR | Description | Dependencies | Risk | Status |
|----|-------------|--------------|------|--------|
| PR-104.5 | Apply validators to existing native targets | PR-104 | Low | 🔜 Pending |

**Key Deliverables:**
- `t__risk_factors` with `@check_output_custom` and `@schema.output`
- `t__hotspots` with `@check_output_custom` and `@schema.output`
- Updated `ContractEnforcementHook` for validation reporting
- `build_hooks()` with validation enabled by default
- Documented patterns for Phase 2-6 target migrations

### Phase 2: Ingestion Domain (PRs 105-116) ✅ COMPLETE

**Goal**: Migrate all ingestion plugins to native Hamilton **with Hamilton-native validation**.

**Status**: ✅ All PRs completed. See [Phase 2 Implementation Status](#phase-2-implementation-status) for details.

**Validation Integration**: Each target migration includes:
- Apply `@check_output_custom` with appropriate validators (where applicable)
- Apply `@schema.output` for documentation (where applicable)
- Use migration bridge to generate validators from existing Pandera schemas
- Additional effort: ~30 min per target

> **Note**: Validators were NOT applied to compute nodes that return custom dataclasses (e.g., `ScanResult`, `AstExtractResult`). Validators only apply to `pd.DataFrame` or `ibis.expr.types.Table` outputs.

| PR | Description | Dependencies | Risk | Status |
|----|-------------|--------------|------|--------|
| PR-105 | Native `t__modules` | PR-104.5 | High | ✅ Complete |
| PR-106 | Enhanced `t__scip` | PR-105 | High | ✅ Complete |
| PR-107 | Native `t__ast` | PR-105 | Medium | ✅ Complete |
| PR-108 | Native `t__cst` | PR-105 | Medium | ✅ Complete |
| PR-109 | Enhanced `t__typing` | PR-106 | Medium | ✅ Complete |
| PR-110 | Native `t__tests_ingest` | PR-105 | Medium | ✅ Complete |
| PR-111 | Native `t__docstrings` | PR-107 | Low | ✅ Complete |
| PR-112 | Native `t__coverage_ingest` | PR-110 | Medium | ✅ Complete |
| PR-113 | Native `t__config_ingest` | PR-105 | Low | ✅ Complete |
| PR-114 | Verify ingestion domain parity | PR-105-113 | Low | ✅ Complete |
| PR-115 | Delete ingestion plugins | PR-114 | Low | ✅ Complete |
| PR-116 | Delete ingestion registrations | PR-115 | Low | ✅ Complete |

**Actual Effort**: ~2 days (including debugging naming conventions and test infrastructure updates)

### Phase 3: Graphs Domain (PRs 117-126)

**Goal**: Migrate all graph plugins to native Hamilton **with Hamilton-native validation**.

**Validation Integration**: Same pattern as Phase 2 - each target gets validators and schema docs.

| PR | Description | Dependencies | Risk | Includes Validation |
|----|-------------|--------------|------|---------------------|
| PR-117 | Native `t__goids` | PR-106 | Medium | ✅ |
| PR-118 | Native `t__call_graph` | PR-117 | Medium | ✅ |
| PR-119 | Native `t__import_graph` | PR-105 | Medium | ✅ |
| PR-120 | Native `t__symbol_uses` | PR-106 | Medium | ✅ |
| PR-121 | Native `t__cfg_dfg` | PR-107, PR-117 | High | ✅ |
| PR-122 | Native `t__call_graph_views` | PR-118 | Low | ✅ |
| PR-123 | Native graph metrics targets | PR-118 | Low | ✅ |
| PR-124 | Verify graphs domain parity | PR-117-123 | Low | — |
| PR-125 | Delete graphs plugins | PR-124 | Low | — |
| PR-126 | Delete graphs registrations | PR-125 | Low | — |

**Additional Effort for Validation**: ~4 hours (8 targets × 30 min)

### Phase 4: Analytics Domain (PRs 127-145)

**Goal**: Migrate all analytics plugins to native Hamilton **with Hamilton-native validation**.

**Validation Integration**: Same pattern - each target gets validators and schema docs.

> **Note**: `t__risk_factors` and `t__hotspots` already have validation from Phase 1.5.

| PR | Description | Dependencies | Risk | Includes Validation |
|----|-------------|--------------|------|---------------------|
| PR-127 | Native `t__function_metrics` | PR-117 | Medium | ✅ |
| PR-128 | Native `t__function_types` | PR-117 | Low | ✅ |
| PR-129 | Native `t__ast_features` | PR-107, PR-117 | Medium | ✅ |
| PR-130 | Native `t__risk_factors` | PR-127, PR-118 | Low (exists) | ✅ (Phase 1.5) |
| PR-131 | Native `t__hotspots` | PR-130 | Low | ✅ (Phase 1.5) |
| PR-132 | Native `t__profiles` | Multiple | High | ✅ |
| PR-133 | Native `t__test_profile` | PR-110, PR-112 | Medium | ✅ |
| PR-134 | Native `t__coverage_edges` | PR-112 | Medium | ✅ |
| PR-135 | Native `t__subsystems` | Multiple | High | ✅ |
| PR-136 | Native `t__semantic_roles` | Multiple | Medium | ✅ |
| PR-137 | Native `t__symbol_graph_metrics` | PR-120 | Medium | ✅ |
| PR-138 | Native `t__function_history` | PR-127 | Medium | ✅ |
| PR-139 | Native `t__history_timeseries` | PR-138 | Low | ✅ |
| PR-140 | Native `t__data_models` | PR-106 | Medium | ✅ |
| PR-141 | Native `t__entrypoints` | PR-118 | Medium | ✅ |
| PR-142 | Native `t__dependencies` | PR-118, PR-119 | Medium | ✅ |
| PR-143 | Verify analytics domain parity | PR-127-142 | Low | — |
| PR-144 | Delete analytics plugins | PR-143 | Low | — |
| PR-145 | Delete analytics registrations | PR-144 | Low | — |

**Additional Effort for Validation**: ~7 hours (14 new targets × 30 min, 2 already done)

### Phase 5: Export Domain (PRs 146-150)

**Goal**: Migrate export plugins to native Hamilton **with Hamilton-native validation**.

**Validation Integration**: Export targets get output format validators.

| PR | Description | Dependencies | Risk | Includes Validation |
|----|-------------|--------------|------|---------------------|
| PR-146 | Native `t__export_jsonl` | Any | Low (exists) | ✅ |
| PR-147 | Native `t__export_parquet` | Any | Low (exists) | ✅ |
| PR-148 | Verify export domain parity | PR-146-147 | Low | — |
| PR-149 | Delete export plugins | PR-148 | Low | — |
| PR-150 | Delete export registrations | PR-149 | Low | — |

**Additional Effort for Validation**: ~1 hour (2 targets × 30 min)

### Phase 6: Cleanup (PRs 151-160)

**Goal**: Delete all legacy infrastructure and **finalize validation transition**.

**Validation Finalization**: Remove SCHEMA_REGISTRY Pandera dependencies; Hamilton validators become authoritative.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-151 | Delete `plugin.py` | All domain PRs | Low |
| PR-152 | Delete `context.py` | PR-151 | Low |
| PR-153 | Delete `context_base.py` | PR-152 | Low |
| PR-154 | Delete `unified_registry.py` | PR-151 | Low |
| PR-155 | Delete `registrations.py` | PR-154 | Low |
| PR-156 | Delete `resources.py`, `result.py`, `run_config.py` | PR-153 | Low |
| PR-157 | Simplify `registry.py` (remove static constants) | PR-155 | Medium |
| PR-158 | Simplify `node_factory.py` (remove wrappers) | PR-157 | Medium |
| PR-159 | Clean up `__init__.py` exports | PR-158 | Low |
| PR-160 | Final verification and documentation | PR-159 | Low |
| PR-160.5 | **Remove SCHEMA_REGISTRY Pandera dependencies** | PR-160 | Medium |

**Additional Effort for Validation Finalization**: ~1 day

---

## Detailed PR Breakdown

### PR-100: Consolidate Hooks ✅ COMPLETE

**Files Changed:**
- Created `hamilton/hooks/__init__.py`
- Moved `manifest_hook.py` → `hamilton/hooks/manifest_hook.py`
- Moved `telemetry_hook.py` → `hamilton/hooks/telemetry_hook.py`
- Created `hamilton/hooks/contract_hook.py` (from enforcement_hook)
- Created `hamilton/hooks/lifecycle.py` (new advanced hooks)
- Updated old files to re-export for backward compatibility

**Acceptance Criteria:**
- [x] All hooks in single directory
- [x] Backward compatibility aliases preserved
- [x] Imports updated across codebase
- [x] Tests pass

---

### PR-101: Unified NativeTargetExecutor ✅ COMPLETE

**Files Changed:**
- Enhanced `hamilton/native/executor.py` with async support
- Updated imports to use new hooks directory
- Added comprehensive error handling

**Implemented Interface:**
```python
class NativeTargetExecutor:
    """Unified executor for all native Hamilton targets."""
    
    @classmethod
    def for_target(cls, env: BuildEnv, graph: TargetGraph, name: str) -> Self:
        """Create executor for a named target."""
    
    def should_skip(self) -> bool:
        """Check if target can be skipped."""
    
    def skip(self) -> TargetRunRecord:
        """Return skip record."""
    
    def execute(self, compute_fn: Callable[[], dict[str, int]]) -> TargetRunRecord:
        """Execute synchronously and return record with row counts."""
    
    async def execute_async(self, compute_fn: Callable[[], Awaitable[dict[str, int]]]) -> TargetRunRecord:
        """Execute asynchronously and return record with row counts."""  # NEW
    
    def fail(self, error: Exception) -> TargetRunRecord:
        """Return failure record."""
```

**Acceptance Criteria:**
- [x] Single executor class
- [x] Handles skip check, timing, record creation
- [x] Async execution support added
- [x] All existing native targets compatible
- [x] Comprehensive test coverage

---

### PR-100.5: Hamilton-Native Data Validation ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/validators/__init__.py`
- `src/codeintel/build/hamilton/validators/dataframe.py`
- `src/codeintel/build/hamilton/validators/contracts.py`
- `src/codeintel/build/hamilton/schema_docs.py`

**Validators Implemented:**
- `ColumnsExistValidator` - Verify required columns exist
- `ColumnTypesValidator` - Verify column dtypes
- `NoNullsInColumnsValidator` - Verify no nulls
- `UniqueColumnsValidator` - Verify uniqueness constraints
- `RowCountValidator` - Verify exact row count
- `RowCountRangeValidator` - Verify row count range
- `ColumnValuesInSetValidator` - Verify values in allowed set

**Contract Builders:**
- `build_table_contract()` - Standard table validation
- `build_key_column_contract()` - Primary key constraints
- `build_metrics_contract()` - Numeric range validation
- `build_enum_column_contract()` - Enum/categorical validation

**Acceptance Criteria:**
- [x] Custom validators extend `BaseDefaultValidator`
- [x] Contract builders return validator tuples
- [x] Schema documentation via `@schema.output`
- [x] Comprehensive test coverage

---

### PR-100.6: Extended Lifecycle Hooks ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/hooks/lifecycle.py`

**Hooks Implemented:**
- `ProgressBarHook` - tqdm integration for progress visualization
- `BuildTimingHook` - Per-node execution timing
- `ConditionalHook` - Conditionally enable/disable hooks
- `NodeTimingRecord` - Timing data structure

**Factory Functions:**
- `create_progress_hook()` - Creates ProgressBarHook with CI detection

**Updated Functions:**
- `build_hooks()` - Added `enable_progress` and `enable_timing` parameters

**Acceptance Criteria:**
- [x] Progress bar integration with tqdm
- [x] Timing collection with slowest_nodes() analysis
- [x] CI environment detection for disabling progress
- [x] Conditional hook activation
- [x] Comprehensive test coverage

---

### PR-100.7: Migration Bridge ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/validators/migration.py`

**Functions Implemented:**
- `validators_from_pandera_schema()` - Convert Pandera schema to validators
- `validators_from_schema_registry()` - Convert from SCHEMA_REGISTRY
- `schema_output_from_registry()` - Generate @schema.output args
- `generate_migration_code()` - Generate ready-to-use migration code
- `MigrationReport` - Track migration status

**Acceptance Criteria:**
- [x] Pandera-to-Hamilton type mapping
- [x] Schema registry integration
- [x] Code generation for migration
- [x] Migration tracking via MigrationReport
- [x] Comprehensive test coverage

---

### PR-101.5: Custom BuildResultBuilder ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/result_builder.py`

**Classes Implemented:**
- `ResultStatus` - Enum (SUCCESS, FAILURE, SKIPPED, PARTIAL)
- `NodeResult` - Per-node result container
- `BuildExecutionResult` - Structured execution output
- `BuildResultBuilder` - Hamilton ResultBuilder implementation
- `DictResultBuilder` - Simple dict aggregation

**Acceptance Criteria:**
- [x] Structured result format
- [x] Status aggregation logic
- [x] Timing information
- [x] Summary generation
- [x] Comprehensive test coverage

---

### PR-102: Native Module Loader ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/native/loader.py`

**Class Implemented:**
```python
class NativeModuleLoader:
    @staticmethod
    def list_domains() -> list[str]
    @staticmethod
    def list_module_paths(domain: str) -> list[Path]
    @staticmethod
    def discover_modules() -> list[NativeModuleInfo]
    @staticmethod
    def validate_module(module: ModuleType) -> tuple[bool, list[str]]
    @staticmethod
    def load_for_driver(domains: list[str]) -> list[ModuleType]
    @staticmethod
    def get_target_names(modules: list[ModuleType]) -> list[str]
```

**Acceptance Criteria:**
- [x] Domain discovery
- [x] Module path resolution
- [x] Module validation
- [x] Driver-ready loading
- [x] Target name extraction
- [x] Comprehensive test coverage

---

### PR-103: Native-Only Mode Flag ✅ COMPLETE

**Files Modified:**
- `src/codeintel/build/hamilton/driver_factory.py` - Added "native" to HamiltonNodeMode
- `src/codeintel/cli/commands/build.py` - Updated --hamilton-mode choices
- `src/codeintel/cli/handlers/build.py` - Updated validation

**Acceptance Criteria:**
- [x] "native" added to HamiltonNodeMode literal
- [x] CLI help updated with native option
- [x] Handler validation updated
- [x] Comprehensive test coverage

---

### PR-103.5: Parallel Execution Adapters ✅ COMPLETE

**Files Created:**
- `src/codeintel/build/hamilton/adapters/__init__.py`
- `src/codeintel/build/hamilton/adapters/parallel.py`

**Classes/Functions Implemented:**
- `ExecutionBackend` - Enum (SEQUENTIAL, THREADPOOL, AUTO)
- `ParallelConfig` - Configuration dataclass
- `ThreadPoolAdapter` - Hamilton FutureAdapter wrapper
- `create_parallel_adapter()` - Factory function

**CLI Flags Added:**
- `--parallel-backend` - sequential, threadpool, auto
- `--max-workers` - Thread pool size
- `--progress` - Enable progress bar

**Environment Variables:**
- `CODEINTEL_PARALLEL_BACKEND`
- `CODEINTEL_MAX_WORKERS`

**Acceptance Criteria:**
- [x] ThreadPool adapter implementation
- [x] CLI flag support
- [x] Environment variable configuration
- [x] Auto-detection mode
- [x] Comprehensive test coverage

---

### PR-104: Migration Test Harness ✅ COMPLETE

**Files Created:**
- `tests/build/hamilton/native/__init__.py`
- `tests/build/hamilton/native/conftest.py`
- `tests/build/hamilton/native/harness.py`
- `tests/build/hamilton/native/test_parity.py`
- `tests/build/hamilton/native/test_skip_logic.py`

**MigrationTestHarness Methods:**
- `compare_row_counts()` - Verify row count parity
- `compare_table_contents()` - Verify data parity
- `compare_table_schema()` - Verify schema parity

**Test Coverage:**
- Module loader discovery tests
- Module validation tests
- Driver factory mode tests
- Skip logic tests (should_skip, force override, input change)
- Executor execute/fail/skip tests
- Manifest persistence tests

**Acceptance Criteria:**
- [x] Reusable test harness
- [x] Parity comparison methods
- [x] Skip logic validation
- [x] Comprehensive fixture setup
- [x] 196 tests passing

---

### PR-104.5: Validation Proof-of-Concept 🔜 NEXT

**Status**: Pending (1-2 days, standalone before Phase 2)

**Files to Modify:**
- `src/codeintel/build/hamilton/native/analytics/risk_factors.py` - Add validators
- `src/codeintel/build/hamilton/native/analytics/hotspots.py` - Add validators
- `src/codeintel/build/hamilton/hooks/contract_hook.py` - Update for Hamilton validators
- `src/codeintel/build/hamilton/hooks/__init__.py` - Update `build_hooks()` defaults

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Apply `@check_output_custom` to `t__risk_factors__compute` | 1 hour |
| 2 | Apply `@schema.output` to `t__risk_factors__compute` | 30 min |
| 3 | Apply validators to `t__hotspots__compute` | 1 hour |
| 4 | Update `ContractEnforcementHook` for validation reporting | 2 hours |
| 5 | Run with real data, verify error messages | 1 hour |
| 6 | Document learnings, refine contracts.py if needed | 1 hour |
| 7 | Update `build_hooks()` to enable validation by default | 30 min |

**Acceptance Criteria:**
- [ ] `t__risk_factors` uses `@check_output_custom` validators
- [ ] `t__hotspots` uses `@check_output_custom` validators
- [ ] Both targets have `@schema.output` documentation
- [ ] `ContractEnforcementHook` reports validation results
- [ ] Validation errors produce clear, actionable messages
- [ ] `build_hooks()` enables validation by default
- [ ] Tests verify validation catches bad data
- [ ] Patterns documented for Phase 2-6 target migrations

---

### PR-105: Native t__modules (Critical Path)

**This is the most complex migration** as `modules` is a root target with no dependencies.

**Files Created:**
- `hamilton/native/ingestion/modules.py`

**Migration Steps:**

1. Extract core logic from `RepoScanPlugin.execute()`:
```python
# Core logic (unchanged)
step = RepoScanStep(storage, discovery, change_detection, filter)
result, modules, change_set = step.execute(...)
```

2. Create Hamilton module:
```python
@tag(domain="ingestion", target="modules", node_type="compute")
def t__modules__scan(env: BuildEnv) -> ScanResult:
    """Execute repository scan."""
    storage = DuckDBStorageAdapter(env.gateway)
    discovery = FilesystemDiscoveryAdapter(env.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    
    profile = build_scan_profile(env.repo_root, env.config)
    step = RepoScanStep(storage, discovery, change_detection)
    
    return step.execute(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.repo_root,
        profile=profile,
        full_rebuild=False,
    )


@tag(domain="ingestion", target="modules", node_type="materialize")
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ScanResult,
) -> TargetRunRecord:
    """Materialize scan results."""
    executor = NativeTargetExecutor.for_target(env, graph, "modules")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: persist_modules(env, t__modules__scan))
```

3. Create golden output test:
```python
def test_native_modules_matches_plugin():
    """Verify native output matches plugin output."""
    plugin_output = run_plugin("modules", test_repo)
    native_output = run_native("modules", test_repo)
    assert_tables_equal(plugin_output, native_output)
```

**Acceptance Criteria:**
- [ ] Native module produces identical output to plugin
- [ ] Skip logic works correctly
- [ ] Manifest is persisted
- [ ] Performance is comparable

---

### PR-127: Native t__function_metrics (Template for Analytics)

**Files Created:**
- `hamilton/native/analytics/function_metrics.py`

**Migration:**

```python
# Pure compute (Ibis transformation)
@tag(domain="analytics", target="function_metrics", node_type="compute")
def t__function_metrics__compute(
    q__core__goids: ir.Table,
    q__core__ast_nodes: ir.Table,
) -> tuple[ir.Table, ir.Table]:
    """Compute function metrics and types."""
    # Extract computation from compute_function_metrics_and_types()
    metrics = compute_metrics_expression(q__core__goids, q__core__ast_nodes)
    types = compute_types_expression(q__core__goids, q__core__ast_nodes)
    return metrics, types


@tag(domain="analytics", target="function_metrics", node_type="materialize")
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_metrics__compute: tuple[ir.Table, ir.Table],
) -> TargetRunRecord:
    """Materialize function metrics."""
    metrics_expr, types_expr = t__function_metrics__compute
    executor = NativeTargetExecutor.for_target(env, graph, "function_metrics")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: {
        "analytics.function_metrics": materialize_table(env, "analytics.function_metrics", metrics_expr),
        "analytics.function_types": materialize_table(env, "analytics.function_types", types_expr),
    })
```

---

## Migration Recipes

### Recipe 1: Pure Ibis Plugin → Native

**Pattern**: Plugin does Ibis transformation and writes result.

```python
# BEFORE (Plugin)
class RiskFactorsPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        metrics = ctx.gateway.load_table("analytics.function_metrics")
        edges = ctx.gateway.load_table("graph.call_graph_edges")
        
        result = compute_risk_factors(metrics, edges)  # Returns DataFrame
        ctx.write_table("analytics.goid_risk_factors", result)
        
        return TargetResult.succeeded(row_counts={"analytics.goid_risk_factors": len(result)})

# AFTER (Native)
@tag(domain="analytics", target="risk_factors", node_type="compute")
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Pure Ibis computation."""
    return compute_risk_factors_ibis(q__analytics__function_metrics, q__graph__call_graph_edges)


@tag(domain="analytics", target="risk_factors", node_type="materialize")
def t__risk_factors(env: BuildEnv, graph: TargetGraph, t__risk_factors__compute: ir.Table) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    if executor.should_skip():
        return executor.skip()
    return executor.execute(lambda: {
        "analytics.goid_risk_factors": materialize_table(env, "analytics.goid_risk_factors", t__risk_factors__compute),
    })
```

### Recipe 2: External Tool Plugin → Native

**Pattern**: Plugin calls external binary (SCIP, Pyright, etc.)

```python
# BEFORE (Plugin)
class ScipPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        result = await ctx.resources.scip_indexer.index(ctx.repo_root, output_path)
        ctx.write_table("core.scip_symbols", result.symbols)
        return TargetResult.succeeded(...)

# AFTER (Native)
@tag(domain="ingestion", target="scip", node_type="compute")
def t__scip__index(
    env: BuildEnv,
    t__modules: TargetRunRecord,  # Ensures modules ran first
) -> ScipIndexResult:
    """Call external SCIP indexer."""
    return asyncio.run(env.providers.scip_indexer.index(
        env.repo_root,
        env.paths.scip_dir / "index.scip",
    ))


@tag(domain="ingestion", target="scip", node_type="materialize")
def t__scip(env: BuildEnv, graph: TargetGraph, t__scip__index: ScipIndexResult) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "scip")
    if executor.should_skip():
        return executor.skip()
    return executor.execute(lambda: persist_scip_result(env, t__scip__index))
```

### Recipe 3: Multi-Output Plugin → Native

**Pattern**: Plugin writes multiple tables.

```python
# BEFORE (Plugin)
class ProfilesPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        profiles = compute_profiles(...)
        ctx.write_table("analytics.function_profile", profiles.functions)
        ctx.write_table("analytics.file_profile", profiles.files)
        ctx.write_table("analytics.module_profile", profiles.modules)
        return TargetResult.succeeded(row_counts={...})

# AFTER (Native)
@dataclass
class ProfilesResult:
    functions: ir.Table
    files: ir.Table
    modules: ir.Table


@tag(domain="analytics", target="profiles", node_type="compute")
def t__profiles__compute(
    q__analytics__function_metrics: ir.Table,
    q__core__modules: ir.Table,
    # ... other dependencies
) -> ProfilesResult:
    return ProfilesResult(
        functions=compute_function_profiles(...),
        files=compute_file_profiles(...),
        modules=compute_module_profiles(...),
    )


@tag(domain="analytics", target="profiles", node_type="materialize")
def t__profiles(env: BuildEnv, graph: TargetGraph, t__profiles__compute: ProfilesResult) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "profiles")
    if executor.should_skip():
        return executor.skip()
    return executor.execute(lambda: {
        "analytics.function_profile": materialize_table(env, "analytics.function_profile", t__profiles__compute.functions),
        "analytics.file_profile": materialize_table(env, "analytics.file_profile", t__profiles__compute.files),
        "analytics.module_profile": materialize_table(env, "analytics.module_profile", t__profiles__compute.modules),
    })
```

### Recipe 4: Stateful Plugin → Native

**Pattern**: Plugin maintains state (e.g., ChangeTracker)

```python
# BEFORE (Plugin)
class RepoScanPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # Creates and stores ChangeTracker on context
        tracker = ChangeTracker.create(...)
        ctx.resources.change_tracker = tracker  # Stateful!
        ...

# AFTER (Native)
# State flows through Hamilton DAG as data, not side effects

@tag(domain="ingestion", target="modules", node_type="compute")
def t__modules__scan(env: BuildEnv) -> tuple[ScanResult, ChangeTracker]:
    """Return scan result AND tracker."""
    tracker = ChangeTracker.create(...)
    result = scan(...)
    return result, tracker


# Downstream nodes receive tracker as parameter
@tag(domain="ingestion", target="ast", node_type="compute")
def t__ast__extract(
    env: BuildEnv,
    t__modules__scan: tuple[ScanResult, ChangeTracker],
) -> AstResult:
    result, tracker = t__modules__scan
    # Use tracker.changed_modules() to filter work
    ...
```

---

## Testing Strategy

### Test Categories

#### 1. Parity Tests (Per Target)

```python
# tests/build/hamilton/native/test_<target>_parity.py

@pytest.fixture
def test_repo():
    """Small repository fixture with known outputs."""
    return create_test_repo_fixture()


def test_native_<target>_matches_plugin(test_repo, gateway):
    """Native output matches plugin output exactly."""
    # Run plugin
    plugin_result = run_via_plugin("<target>", test_repo, gateway)
    
    # Run native
    native_result = run_via_native("<target>", test_repo, gateway)
    
    # Compare
    assert_tables_equal(plugin_result.tables, native_result.tables)
    assert_row_counts_equal(plugin_result, native_result)
```

#### 2. Integration Tests (Per Domain)

```python
# tests/build/hamilton/native/test_<domain>_integration.py

def test_<domain>_full_pipeline(test_repo, gateway):
    """Run all <domain> targets end-to-end."""
    result = run_domain("<domain>", test_repo, gateway)
    
    assert result.all_succeeded()
    assert result.tables_populated(["core.modules", "core.scip_symbols", ...])
```

#### 3. Skip Logic Tests

```python
def test_native_target_skips_when_current(test_repo, gateway):
    """Target skips when manifest hash matches."""
    # First run - computes
    result1 = run_native("function_metrics", test_repo, gateway)
    assert result1.status == "succeeded"
    
    # Second run - skips
    result2 = run_native("function_metrics", test_repo, gateway)
    assert result2.status == "skipped"


def test_native_target_recomputes_when_forced(test_repo, gateway):
    """Target recomputes when in force_targets."""
    # First run
    run_native("function_metrics", test_repo, gateway)
    
    # Second run with force
    result = run_native("function_metrics", test_repo, gateway, force=True)
    assert result.status == "succeeded"  # Not skipped
```

#### 4. Performance Tests

```python
@pytest.mark.benchmark
def test_native_not_slower_than_plugin(test_repo, gateway, benchmark):
    """Native path is not significantly slower."""
    plugin_time = benchmark(run_via_plugin, "function_metrics", test_repo, gateway)
    native_time = benchmark(run_via_native, "function_metrics", test_repo, gateway)
    
    # Allow 10% overhead
    assert native_time < plugin_time * 1.1
```

### Migration Verification Checklist

For each migrated target:

- [ ] Native module created in correct location
- [ ] Tags include domain, target, node_type
- [ ] Dependencies expressed as function parameters
- [ ] Compute nodes are pure (no side effects)
- [ ] Materialize node uses NativeTargetExecutor
- [ ] Skip logic works correctly
- [ ] Output tables match plugin exactly
- [ ] Row counts match
- [ ] Manifests are persisted
- [ ] Performance is acceptable

---

## Risk Assessment & Mitigation

### High Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Complex plugin migration | Delay | High | Start with simpler plugins, build patterns |
| Async compatibility | Breakage | Medium | Hamilton supports async, test thoroughly |
| ChangeTracker state | Breakage | Medium | Flow state through DAG, not context |
| Performance regression | UX | Low | Benchmark before/after |

### Rollback Strategy

Each phase is independently deployable:

1. **Phase 2 (Ingestion)**: Can run with `--plugin-path ingestion` to revert
2. **Phase 3-5**: Same pattern
3. **Phase 6 (Cleanup)**: Only after all domains verified

### Feature Flags

```python
# During migration, support both paths
class BuildMode(Enum):
    PLUGIN = "plugin"      # Legacy
    NATIVE = "native"      # New
    HYBRID = "hybrid"      # Mix (during migration)


def build_driver(mode: BuildMode = BuildMode.HYBRID):
    if mode == BuildMode.NATIVE:
        return load_native_modules_only()
    elif mode == BuildMode.PLUGIN:
        return load_plugin_wrappers_only()
    else:
        return load_hybrid()  # Native where available
```

---

## Success Criteria

### Phase Completion Criteria

| Phase | Criteria |
|-------|----------|
| Phase 1 | Hooks consolidated, executor unified, test harness ready ✅ |
| Phase 1.5 | Validation POC complete, existing native targets validated |
| Phase 2 | All ingestion plugins migrated **with validators**, parity tests pass |
| Phase 3 | All graphs plugins migrated **with validators**, parity tests pass |
| Phase 4 | All analytics plugins migrated **with validators**, parity tests pass |
| Phase 5 | All export plugins migrated **with validators**, parity tests pass |
| Phase 6 | Legacy infrastructure deleted, **SCHEMA_REGISTRY removed**, all tests pass |

### Final Success Criteria

**Phase 1 Complete:**
- [x] **Hooks consolidated** into single directory
- [x] **Hamilton-native validation** framework established
- [x] **Parallel execution** adapters implemented
- [x] **Custom result builder** for structured output
- [x] **Native module loader** for discovery
- [x] **Migration test harness** ready
- [x] **CLI updated** with native mode and parallel options

**Phase 1.5 Pending (Next):**
- [ ] **Validation POC** complete on existing native targets
- [ ] **ContractEnforcementHook** integrated with Hamilton validators
- [ ] **Validation patterns** documented for subsequent phases

**Phases 2-6 Pending:**
- [ ] **Zero plugin classes** remain in codebase
- [ ] **Single context type** (BuildEnv) in use
- [ ] **All targets** are Hamilton native modules with validators
- [ ] **All targets** have `@schema.output` documentation
- [ ] **Skip logic** works uniformly via ManifestHook
- [ ] **All tests pass** (no new xfails)
- [ ] **Performance** is equal or better
- [ ] **Lines of code** reduced by ~3,000+
- [ ] **SCHEMA_REGISTRY** Pandera dependencies removed
- [ ] **Documentation** updated

---

## Appendix: File-by-File Disposition

### Files to DELETE

| File | Lines | Phase |
|------|-------|-------|
| `plugin.py` | 425 | Phase 6 |
| `context.py` | 582 | Phase 6 |
| `context_base.py` | 605 | Phase 6 |
| `unified_registry.py` | 461 | Phase 6 |
| `registrations.py` | 333 | Phase 6 |
| `resources.py` | 177 | Phase 6 |
| `result.py` | 92 | Phase 6 |
| `run_config.py` | 66 | Phase 6 |
| `state.py` | 147 | Phase 6 |
| `plugins/ingestion/*.py` | ~1,000 | Phase 2 |
| `plugins/graphs/*.py` | ~800 | Phase 3 |
| `plugins/analytics/*.py` | ~1,500 | Phase 4 |
| **Total** | **~6,188** | |

### Files to SIMPLIFY

| File | Current | Target | Phase |
|------|---------|--------|-------|
| `registry.py` | 750 | 250 | Phase 6 |
| `targets.py` | 484 | 300 | Phase 6 |
| `node_factory.py` | 828 | 300 | Phase 6 |
| `executor.py` | 566 | 400 | Phase 1 |
| `hashing.py` | 198 | 150 | Phase 1 |
| **Total Reduction** | | **~1,376** | |

### Files to KEEP (Unchanged)

| File | Lines | Reason |
|------|-------|--------|
| `contracts.py` | 297 | Core data model |
| `manifest.py` | 172 | Core data model |
| `session.py` | 226 | Caching layer |
| `state_types.py` | 415 | State data model |
| `state_computer.py` | 415 | State computation |
| `protocols.py` | 333 | DI interfaces |
| `providers.py` | 1,070 | DI implementations |
| `types.py` | 343 | Shared types |
| `errors.py` | 854 | Error hierarchy |
| `config.py` | 359 | Configuration |
| `parameters.py` | 231 | Parameters |

---

## Document Control

**Version**: 3.1  
**Status**: Phase 1 Complete, Phase 1.5 Next  
**Author**: CodeIntel Build Team  
**Last Updated**: 2025-12-15

### Changelog

- **v3.1** (2025-12-15): Added Phase 1.5 Validation Proof-of-Concept and integrated validation into Phases 2-6
  - Added Phase 1.5: Validation POC section with PR-104.5 details
  - Added validation rollout strategy and schema registry transition plan
  - Updated Phase 2-6 tables to include "Includes Validation" column
  - Updated Phase 6 to include PR-160.5 for SCHEMA_REGISTRY removal
  - Updated Timeline Estimate with validation effort per phase
  - Updated Success Criteria with Phase 1.5 and validation criteria
  - Updated Phase Completion Criteria with validation requirements
  - Total additional effort for validation: ~3-4 days spread across phases
- **v3.0** (2025-12-15): Phase 1 Advanced Features Implementation Complete
  - Implemented PR-100: Hook consolidation into `hamilton/hooks/`
  - Implemented PR-100.5: Hamilton-native data validation (custom validators, NOT Pandera @check_output)
  - Implemented PR-100.6: Extended lifecycle hooks (ProgressBarHook, BuildTimingHook, ConditionalHook)
  - Implemented PR-100.7: Migration bridge from Pandera schemas to Hamilton validators
  - Implemented PR-101: Async support in NativeTargetExecutor
  - Implemented PR-101.5: Custom BuildResultBuilder with structured output
  - Implemented PR-102: NativeModuleLoader for native module discovery
  - Implemented PR-103: Added `--native-only` flag to CLI
  - Implemented PR-103.5: Parallel execution adapters (ThreadPoolAdapter, CLI flags)
  - Implemented PR-104: Migration test harness with parity and skip logic tests
  - Added Phase 1 Implementation Status section documenting all created files
  - Updated Feature 1 to reflect Hamilton-native validation approach (not Pandera @check_output)
  - Updated go-forward plans for Phases 2-6 to leverage new infrastructure
- **v2.0** (2025-12-15): Enhanced with 16 advanced Hamilton features integration
  - Added @check_output with Pandera for per-node validation
  - Added @extract_fields for multi-output lineage
  - Added @parameterize for DRY target variants
  - Added @config.when for environment-specific implementations
  - Added @datasaver for standardized I/O with metadata
  - Added Builder.with_materializers() for dynamic export
  - Added Graph Adapters for parallel execution
  - Added Parallelizable/Collect for dynamic parallelism
  - Added enhanced lifecycle hooks (ProgressBar, OpenLineage)
  - Added Hamilton UI integration
  - Added @schema for column metadata
  - Added @pipe_input for transformation chains
  - Added Hamilton caching integration
  - Added custom ResultBuilder for BuildReport
  - Added Hamilton CLI integration for CI/CD
  - Added @parameterized_subdag for multi-repo builds
  - Updated Phase 1 PR breakdown with new foundation PRs
  - Updated success criteria with advanced feature adoption
  - Added enhanced architecture diagram
- **v1.0** (2025-12-15): Initial comprehensive implementation plan

