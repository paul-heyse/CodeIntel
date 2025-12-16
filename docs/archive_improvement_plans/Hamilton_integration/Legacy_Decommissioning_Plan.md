# Legacy System Decommissioning Plan

## Overview

This document validates the proposed legacy system removal and identifies additional candidates for consolidation now that Hamilton is the primary build orchestrator.

---

## Implementation Status

**Completed: 2024-12-13**

### Phase A: CLI Cleanup ✅
- [x] Removed `--engine` flag from CLI build command
- [x] Removed `--hamilton-mode phase0` from CLI (only "generated" and "auto" remain)
- [x] Updated CLI handlers to always use Hamilton

### Phase B: Core Legacy Removal ✅
- [x] Migrated `serving/auto_pipeline.py` to use `HamiltonBuildExecutor`
- [x] Deleted `build/executor.py` (1,193 lines removed)
- [x] Deleted `build/plan.py` (503 lines removed)  
- [x] Deleted `build/resolver.py` (736 lines removed)

### Phase C: Hamilton Scaffolding Cleanup ✅
- [x] Deleted `build/hamilton/nodes/targets_phase0.py` (652 lines removed)
- [x] Updated `driver_factory.py` to remove phase0 support
- [x] Moved `_run_target` execution logic to `node_factory.py`
- [x] Updated `HamiltonNodeMode` type to only allow "generated" | "auto"

### Phase D: Graphs Plugin Consolidation ✅
- [x] Migrate `cli/commands/graphs.py` and `cli/handlers/graphs.py`
- [x] Migrate `serving/mcp/architecture_tools.py`
- [x] Rewrite `graphs/plugins/__init__.py` registration
- [x] Delete `graphs/core/`
- [x] Delete `graphs/runtime/`
- [x] Delete legacy test files (16 files removed)
- [x] Update test helpers to remove legacy graph plugin infrastructure

---

---

## 1. Validation of Proposed Changes

### 1.1 Remove Legacy Build Engine (VALIDATED)

**Proposed deletions:**
- `src/codeintel/build/executor.py` (BuildExecutor)
- `src/codeintel/build/plan.py` (PlanGenerator)
- `src/codeintel/build/resolver.py` (BuildResolver)

**Current import analysis:**

| File | Import | Usage |
|------|--------|-------|
| `cli/handlers/build.py` | `BuildExecutor, ExecutorEnv` | Used in `_execute_build()` when `engine=="legacy"` |
| `cli/handlers/build.py` | `PlanGenerator` | Used in `build_plan_handler` and `build_explain_handler` |
| `cli/handlers/build.py` | `BuildResolver` | Used in `build_status_handler` |
| `serving/auto_pipeline.py` | `BuildExecutor, ExecutorEnv` | Used in `run_pipeline()` |
| `serving/auto_pipeline.py` | `PlanGenerator` | Used in `run_pipeline()` |
| `serving/auto_pipeline.py` | `BuildResolver` | Used in `run_pipeline()` |
| `build/__init__.py` | All three | Public API exports |

**Migration requirements:**

1. **CLI `build run` handler**: Remove `--engine` flag entirely. Route all execution through `HamiltonBuildExecutor`.

2. **CLI `build plan/explain` handlers**: Replace `PlanGenerator` with `codeintel.build.hamilton.planner.compute_plan()` and `explain_plan()` which already exist.

3. **CLI `build status` handler**: The `StateValidator` and `DatabaseState` classes in `state.py` can remain (they are orthogonal to the executor). The `BuildResolver` is only used for the legacy status display - Hamilton's `compute_plan()` can provide equivalent info.

4. **`serving/auto_pipeline.py`**: This is a critical migration - must be updated to use `HamiltonBuildExecutor` instead of `BuildExecutor`.

**Verdict: Safe to remove AFTER migration of call sites**

---

### 1.2 Remove Phase 0 Mode Scaffolding (PARTIAL VALIDATION)

**Proposed changes:**
- Remove `src/codeintel/build/hamilton/nodes/targets_phase0.py`
- Remove `--hamilton-mode` CLI flag
- Simplify `HamiltonNodeMode` to just `"generated"` or `"auto"`

**Analysis:**

`targets_phase0.py` (652 lines) contains:
- Explicit Hamilton node definitions for the original Phase 0 chain
- `TARGET_TO_NODE` mapping
- Manual `t__modules`, `t__scip`, `t__ast`, `t__goids`, `t__call_graph`, `t__function_metrics`, `t__risk_factors` nodes

This was scaffolding for initial Hamilton integration. Now that:
- Native modules exist (`build/hamilton/native/`)
- Generated wrappers work (`build/hamilton/nodes/node_factory.py`)
- "auto" mode composes native + generated

**The phase0 module is truly legacy.**

However, `driver_factory.py` still references it:
```python
from codeintel.build.hamilton.nodes import targets_phase0

if mode == "phase0":
    dr = driver.Driver(config or {}, targets_phase0)
```

**Verdict: Safe to remove if:**
1. Remove `--hamilton-mode phase0` from CLI
2. Remove `"phase0"` from `HamiltonNodeMode` literal
3. Delete `targets_phase0.py`
4. Update `driver_factory.py` to remove phase0 branch

**Recommendation:** Keep `HamiltonNodeMode = Literal["generated", "auto"]` for now since "auto" mode (native + generated composition) is the more performant path for certain targets.

---

### 1.3 Remove Duplicate Graph Plugin System (VALIDATED)

**Proposed deletions:**
- `src/codeintel/graphs/core/` (registry, protocol, adapters, context)
- `src/codeintel/graphs/runtime/` (graph_executor, manifest, planning)

**Current import analysis:**

| Module | Consumers | Usage |
|--------|-----------|-------|
| `graphs.core.registry` | `cli/commands/graphs.py`, `cli/handlers/graphs.py`, `serving/mcp/architecture_tools.py`, `graphs/plugins/__init__.py` | Graph plugin discovery/listing |
| `graphs.core.protocol` | `graphs/runtime/*`, `graphs/core/*` | `GraphPluginProtocol` definition |
| `graphs.core.adapters` | `graphs/plugins/__init__.py` | `TargetPluginAdapter` for wrapping |
| `graphs.runtime.planning` | `build/executor.py` | `plan_graph_plugin_run()` |
| `graphs.runtime.graph_executor` | `build/executor.py` | `GraphPluginExecutor` |

**Key insight:** The only non-graph consumer of `graphs.runtime` is `build/executor.py` (the legacy executor). Once the legacy executor is removed, `graphs.runtime` becomes orphaned.

**Migration path:**

1. `cli/commands/graphs.py` and `cli/handlers/graphs.py` commands like `graphs plugins-list` should become thin views over:
   - `codeintel.build.registry.get_target_graph()` filtered to `module="graphs"`
   - Hamilton's DAG export for plugin ordering

2. `serving/mcp/architecture_tools.py` uses `get_graph_registry()` for MCP tool introspection - this should migrate to the build registry.

3. `graphs/plugins/__init__.py` uses `TargetPluginAdapter` to register graph plugins - this should be replaced by direct target plugin registration.

**Verdict: Safe to remove AFTER:**
1. Legacy `build/executor.py` is removed (breaks the only runtime consumer)
2. CLI handlers are migrated to use build registry
3. MCP architecture tools are migrated

---

## 2. Additional Legacy Files to Remove

### 2.1 In `build/` Directory

| File | Lines | Reason | Action |
|------|-------|--------|--------|
| `executor.py` | 1,193 | Legacy BuildExecutor | **Remove** |
| `plan.py` | 503 | Legacy PlanGenerator | **Remove** |
| `resolver.py` | 736 | Legacy BuildResolver | **Remove** |
| `state.py` | 605 | StateValidator | **Keep** - Used by Hamilton planner for staleness checks |
| `readiness.py` | 810 | DatabaseReadinessView | **Keep** - Still useful for serving prerequisites |
| `plugin_registry.py` | 398 | Plugin lookup | **Keep** - Used by Hamilton nodes and ingestion |
| `plugins.py` | 242 | Base plugin class | **Review** - May be superseded by hamilton/native system |

### 2.2 In `build/hamilton/nodes/` Directory

| File | Lines | Reason | Action |
|------|-------|--------|--------|
| `targets_phase0.py` | 652 | Phase 0 debug scaffolding | **Remove** |
| `dataset_nodes.py` | ~200 | Manual dataset nodes | **Review** - May overlap with node_factory.py |
| `node_factory.py` | ~500 | Dynamic node generation | **Keep** - Core Hamilton infrastructure |

### 2.3 In `graphs/` Directory

| Module | Reason | Action |
|--------|--------|--------|
| `graphs/core/` | Duplicate registry/protocol | **Remove** (after executor.py) |
| `graphs/runtime/` | Only used by legacy executor | **Remove** (after executor.py) |
| `graphs/plugins/__init__.py` | Registration via duplicate system | **Rewrite** to use build registry |

---

## 3. Recommended Execution Order

### Phase A: CLI Cleanup (Low Risk)
1. Remove `--engine legacy` from CLI
2. Remove `--hamilton-mode phase0` from CLI
3. Update CLI handlers to always use Hamilton

### Phase B: Core Legacy Removal (Medium Risk)
1. Migrate `serving/auto_pipeline.py` to use `HamiltonBuildExecutor`
2. Delete `build/executor.py`
3. Delete `build/plan.py`
4. Delete `build/resolver.py`
5. Update `build/__init__.py` exports

### Phase C: Hamilton Scaffolding Cleanup (Low Risk)
1. Delete `build/hamilton/nodes/targets_phase0.py`
2. Update `driver_factory.py` to remove phase0 support
3. Simplify `HamiltonNodeMode` type

### Phase D: Graphs Plugin Consolidation (Medium Risk)
1. Migrate `cli/commands/graphs.py` and `cli/handlers/graphs.py`
2. Migrate `serving/mcp/architecture_tools.py`
3. Rewrite `graphs/plugins/__init__.py` registration
4. Delete `graphs/core/`
5. Delete `graphs/runtime/`

---

## 4. Files That Should NOT Be Removed

These are still needed even with Hamilton-only execution:

| File | Reason |
|------|--------|
| `state.py` | `StateValidator` and `DatabaseState` used by Hamilton planner |
| `hashing.py` | `compute_input_hash` used by Hamilton manifest_hook |
| `manifest.py` | `OutputManifest` and `BuildRunRecord` used everywhere |
| `context.py` | `TargetExecutionContext` used by all plugins |
| `contracts.py` | `OutputContract` is SSOT for target outputs |
| `registry.py` | `TargetGraph` is canonical dependency graph |
| `targets.py` | `OutputTarget` definitions |
| `plugin.py` | `TargetPlugin` base class |
| `plugin_registry.py` | Plugin lookup for Hamilton nodes |
| `readiness.py` | Serving layer prerequisites |
| `operations.py` | Operation to target mapping |
| `providers.py` | DI for SCIP indexer, type checker, etc. |
| `protocols.py` | Tool runner protocols |

---

## 5. Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Files in `build/` | 23 | 20 (-3) |
| Lines in `build/` | ~10,000 | ~7,500 (-25%) |
| Files in `graphs/` | 40+ | 35 (-5) |
| CLI flags | `--engine`, `--hamilton-mode` | None |
| Build engines | 2 (legacy + Hamilton) | 1 (Hamilton) |
| Graph plugin systems | 2 | 1 |

---

## 6. Test Updates Required

Tests that will need updates:

```
tests/build/test_executor.py          # Delete or migrate
tests/build/test_plan.py              # Delete or migrate
tests/build/test_resolver.py          # Delete or migrate
tests/cli/test_help_rendering.py      # Remove --engine assertions
tests/serving/test_auto_pipeline.py   # Update to Hamilton
tests/graphs/test_runtime_*.py        # Delete
tests/graphs/test_core_*.py           # Delete
```

---

## 7. Summary

The proposed decommissioning is **valid and recommended**. The analysis confirms:

1. **Legacy executor removal is safe** - Hamilton executor is fully capable
2. **Phase 0 mode removal is safe** - "auto" and "generated" modes are production-ready
3. **Graphs plugin consolidation is beneficial** - Removes ~1,500 lines of duplicate infrastructure

**Additional candidates identified:**
- `targets_phase0.py` (confirmed redundant)
- Potentially `dataset_nodes.py` (needs review vs node_factory overlap)

**Key dependencies to respect:**
- Remove executor.py BEFORE graphs/runtime/ (executor is the only consumer)
- Migrate auto_pipeline.py BEFORE removing executor.py
- Migrate CLI handlers BEFORE removing plan.py/resolver.py

