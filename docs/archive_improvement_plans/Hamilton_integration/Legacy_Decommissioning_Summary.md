# Legacy System Decommissioning Summary

**Completed: December 13, 2024**

## Executive Summary

This document summarizes the successful decommissioning of the legacy build engine and duplicate graph plugin infrastructure. The work consolidates the codebase to use Hamilton as the sole build orchestrator and the build registry as the single source of truth for target definitions.

---

## Context and Motivation

### The Legacy State

Prior to this work, the CodeIntel codebase maintained two parallel systems:

1. **Dual Build Engines**: Both a legacy `BuildExecutor` and the newer `HamiltonBuildExecutor` coexisted, with CLI flags (`--engine legacy|hamilton`) to switch between them.

2. **Duplicate Graph Plugin Registries**: Graph plugins were registered in two places:
   - The build registry (`codeintel.build.registry.TargetGraph`) for Hamilton execution
   - A separate graph plugin registry (`codeintel.graphs.core.registry.GraphPluginRegistry`) for legacy execution

3. **Phase 0 Scaffolding**: Early Hamilton integration used explicit "Phase 0" nodes (`targets_phase0.py`) as a bridge before dynamic node generation was mature.

### The Go-Forward Architecture

The decommissioning aligns with the established Hamilton architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI / Serving Layer                     │
├─────────────────────────────────────────────────────────────┤
│                    HamiltonBuildExecutor                     │
│    - Single orchestrator for all build operations            │
│    - Modes: "generated" (dynamic) or "auto" (native+gen)     │
├─────────────────────────────────────────────────────────────┤
│                       Build Registry                         │
│    - TargetGraph: Canonical dependency graph                 │
│    - OutputContract: Single source of truth for outputs      │
│    - Plugin registration via build.plugin_registry           │
├─────────────────────────────────────────────────────────────┤
│                     Storage Architecture                     │
│    - DuckDBPolicyBackend: Centralized DDL/mutations          │
│    - TABLE_SCHEMAS registry: Schema definitions              │
│    - IbisGateway: Typed query interface                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Work Completed

### Phase A: CLI Cleanup ✅

**Goal**: Remove engine selection flags and route all builds through Hamilton.

**Changes**:
- Removed `--engine` flag from `build run` command
- Removed `--hamilton-mode phase0` option (only "generated" and "auto" remain)
- Updated `src/codeintel/cli/commands/build.py`
- Updated `src/codeintel/cli/handlers/build.py`

**Impact**: Users no longer need to (or can) select between legacy and Hamilton engines.

---

### Phase B: Core Legacy Removal ✅

**Goal**: Delete the legacy build engine and its dependencies.

**Files Deleted**:
| File | Lines | Description |
|------|-------|-------------|
| `src/codeintel/build/executor.py` | 1,193 | Legacy `BuildExecutor` class |
| `src/codeintel/build/plan.py` | 503 | Legacy `PlanGenerator` class |
| `src/codeintel/build/resolver.py` | 736 | Legacy `BuildResolver` class |

**Files Modified**:
- `src/codeintel/serving/auto_pipeline.py` - Migrated to use `HamiltonBuildExecutor`
- `src/codeintel/build/__init__.py` - Removed legacy exports

**Total Lines Removed**: ~2,432

---

### Phase C: Hamilton Scaffolding Cleanup ✅

**Goal**: Remove Phase 0 scaffolding now that dynamic node generation is mature.

**Files Deleted**:
| File | Lines | Description |
|------|-------|-------------|
| `src/codeintel/build/hamilton/nodes/targets_phase0.py` | 652 | Explicit Phase 0 node definitions |

**Files Modified**:
- `src/codeintel/build/hamilton/driver_factory.py` - Removed phase0 branch
- `src/codeintel/build/hamilton/nodes/__init__.py` - Removed phase0 exports
- `src/codeintel/build/hamilton/nodes/node_factory.py` - Absorbed `_run_target` execution logic

**Result**: `HamiltonNodeMode` is now `Literal["generated", "auto"]` only.

---

### Phase D: Graphs Plugin Consolidation ✅

**Goal**: Eliminate the duplicate graph plugin registry and unify on the build registry.

#### Files Deleted

**Graphs Core (5 files)**:
- `src/codeintel/graphs/core/__init__.py`
- `src/codeintel/graphs/core/adapters.py`
- `src/codeintel/graphs/core/context.py`
- `src/codeintel/graphs/core/protocol.py`
- `src/codeintel/graphs/core/registry.py`

**Graphs Runtime (4 files)**:
- `src/codeintel/graphs/runtime/__init__.py`
- `src/codeintel/graphs/runtime/graph_executor.py`
- `src/codeintel/graphs/runtime/manifest.py`
- `src/codeintel/graphs/runtime/planning.py`

**Test Files (16+ files)**:
- `tests/cli/handlers/test_graphs.py`
- `tests/graphs/test_core_registry_extended.py`
- `tests/graphs/test_runtime_executor.py`
- `tests/graphs/test_runtime_manifest.py`
- `tests/graphs/test_runtime_planning.py`
- `tests/graphs/test_planning_policies.py`
- `tests/graphs/test_planner_dependency_policy.py`
- `tests/graphs/test_protocol_metadata_config.py`
- `tests/graphs/test_target_plugin_adapter_metadata.py`
- `tests/graphs/test_mock_runtime_scenarios.py`
- `tests/analytics/test_graph_runtime.py`
- `tests/mcp/test_architecture_tools.py`
- `tests/serving/mcp/test_mcp_architecture_tools.py`
- `tests/_helpers/fakes/graph_plugins.py`
- `tests/_helpers/fakes/graph_contexts.py`
- `tests/_helpers/fakes/plugins.py`

#### Files Modified

**CLI Layer**:
- `src/codeintel/cli/commands/graphs.py` - Now uses `build.registry.get_target_graph()` 
- `src/codeintel/cli/handlers/graphs.py` - Uses build targets instead of graph plugins

**MCP Tools**:
- `src/codeintel/serving/mcp/architecture_tools.py` - Uses build registry for graph plan

**Graphs Package**:
- `src/codeintel/graphs/__init__.py` - Removed core/runtime exports
- `src/codeintel/graphs/plugins/__init__.py` - Removed adapter registration

**Test Helpers**:
- `tests/graphs/conftest.py` - Removed legacy fixtures
- `tests/_helpers/__init__.py` - Removed legacy exports

---

## Alignment with Storage Architecture

The decommissioning reinforces the centralized storage architecture:

### Before (Fragmented)
```
Legacy BuildExecutor
    └── Direct SQL execution
    └── Schema management scattered

GraphPluginRegistry  
    └── Separate plugin lifecycle
    └── Duplicate execution context
```

### After (Unified)
```
HamiltonBuildExecutor
    └── BuildEnv with StorageGateway
    └── DuckDBPolicyBackend for all mutations
    └── TABLE_SCHEMAS for schema definitions
    └── OutputContract for target outputs
    
Build Registry (TargetGraph)
    └── Single source of truth for targets
    └── Module-based filtering (e.g., module="graphs")
    └── Topological ordering via graph.topological_order()
```

### Key Storage Integration Points

1. **Target Execution**: All plugins execute through `TargetExecutionContext` which provides a `StorageGateway` with the `DuckDBPolicyBackend`.

2. **Schema Management**: Schemas are defined in `TABLE_SCHEMAS` and enforced via `DuckDBPolicyBackend.ensure_table()`.

3. **Contract Enforcement**: `OutputContract` defines expected tables/artifacts, validated by `ContractEnforcer` in strict mode.

4. **Asset Tracking**: The Phase 4 asset catalog (`build.asset_versions`, `build.asset_lineage`) tracks all outputs through the unified storage layer.

---

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Build engine implementations | 2 | 1 | -50% |
| Graph plugin registries | 2 | 1 | -50% |
| Files in `build/` | 23 | 19 | -17% |
| Files in `graphs/core/` + `graphs/runtime/` | 9 | 0 | -100% |
| CLI flags for engine selection | 2 | 0 | -100% |
| Lines of legacy code removed | 0 | ~5,000+ | N/A |
| Test files deleted | 0 | 16+ | N/A |

---

## API Changes

### CLI Commands

**Build Commands**:
```bash
# Before
codeintel build run --targets call_graph --engine hamilton --hamilton-mode phase0

# After  
codeintel build run --targets call_graph
# --engine flag removed, Hamilton is the only engine
# --hamilton-mode only accepts "generated" (default) or "auto"
```

**Graph Commands**:
```bash
# Before (using legacy graph registry)
codeintel graph plugins-list
codeintel graph plugins-plan

# After (using build registry, backward-compatible aliases)
codeintel graph targets-list    # New preferred command
codeintel graph targets-plan    # New preferred command
codeintel graph plugins-list    # Still works (alias)
codeintel graph plugins-plan    # Still works (alias)
```

### Programmatic API

**Before**:
```python
from codeintel.graphs.core.registry import (
    get_graph_registry,
    plan_graph_plugins,
    list_graph_plugins,
)

plan = plan_graph_plugins(["goid_builder", "callgraph"])
```

**After**:
```python
from codeintel.build.registry import get_target_graph

graph = get_target_graph()
graph_targets = [t for t in graph.all_targets if t.module == "graphs"]
ordered = graph.topological_order([t.name for t in graph_targets])
```

---

## Verification

All changes pass the project's quality gates:

```bash
✅ uv run ruff check     # All checks passed
✅ uv run pyright        # 0 errors, 0 warnings  
✅ uv run pyrefly check  # 0 errors
```

---

## Future Considerations

1. **`dataset_nodes.py` Review**: This file may overlap with `node_factory.py` dynamic generation and could be a candidate for future consolidation.

2. **`build/plugins.py` Review**: The base plugin class may be superseded by the Hamilton native system in some cases.

3. **Test Coverage**: New tests should be written for the migrated graph CLI commands using the build registry.

---

## References

- [Legacy Decommissioning Plan](./Legacy_Decommissioning_Plan.md) - Original validation and planning document
- [Phase 3 & 4 Implementation Plan](./Phase3_4_Aligned_Implementation_Plan.md) - Hamilton architecture details
- [Hamilton Phase 4](./Hamilton_apache_phase4.md) - Asset catalog and fingerprinting specs

