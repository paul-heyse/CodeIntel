# Phase 5: Step Configuration Migration - Complete Implementation Plan

> **Status**: In Progress  
> **Last Updated**: 2025-01-13  
> **Estimated Effort**: ~4-6 hours implementation

## Executive Summary

This document provides a comprehensive, detailed implementation plan for completing the migration from deprecated `XXXStepConfig` dataclasses to the Hamilton-native pattern using `SnapshotRef` directly. This is the final phase of the step configuration deprecation effort.

### Goal
Remove all dependencies on `steps_analytics.py` and `steps_graphs.py`, enabling deletion of these deprecated modules and completing the migration to the go-forward architecture.

### Pattern Applied
All compute functions are migrated from:
```python
def compute_xxx(gateway: StorageGateway, cfg: XXXStepConfig) -> None:
    # Uses cfg.repo, cfg.commit, cfg.repo_root
```

To:
```python
def compute_xxx(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    optional_param1: type = default,
    optional_param2: type = default,
) -> None:
    # Uses snapshot.repo, snapshot.commit, snapshot.repo_root
```

---

## Phase 5.2: In-Progress Migrations (Testing Layer)

### 5.2.1 TestCoverageStepConfig Migration

**Status**: 🔄 In Progress

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/testing/coverage/edges.py` | Update function signatures |
| `src/codeintel/build/plugins/analytics/coverage/test_edges.py` | Update plugin to pass `snapshot` |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class TestCoverageStepConfig:
    snapshot: SnapshotRef
    coverage_file: Path | None = None
    coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None
```

**Migration Tasks**:

1. **Create `TestCoverageOptions` dataclass** in `edges.py`:
   ```python
   @dataclass(frozen=True)
   class TestCoverageOptions:
       coverage_file: Path | None = None
       coverage_loader: Callable[[SnapshotRef, Path | None], Coverage | None] | None = None
   ```

2. **Update `EdgeContext`** to accept `snapshot: SnapshotRef` instead of `cfg: TestCoverageStepConfig`

3. **Refactor core functions**:
   - `_load_coverage_data(snapshot, coverage_file)` 
   - `load_coverage_data(snapshot, coverage_file)`
   - `_functions_by_path(gateway, snapshot, catalog_provider)`
   - `_backfill_test_goids(gateway, snapshot)`
   - `backfill_test_goids_for_catalog(gateway, snapshot)`
   - `_test_status_and_meta(gateway, snapshot)`
   - `compute_test_coverage_edges(gateway, snapshot, *, options)`

4. **Update build plugin** to construct options and pass `ctx.snapshot`

---

### 5.2.2 TestProfileStepConfig Migration

**Status**: 🔄 In Progress

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/testing/profiles/builder.py` | Main profile builder |
| `src/codeintel/analytics/testing/profiles/rows.py` | Row building utilities |
| `src/codeintel/analytics/testing/profiles/types.py` | Type definitions |
| `src/codeintel/analytics/testing/coverage/inputs.py` | Input data structures |
| `src/codeintel/build/plugins/analytics/tests/profile.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class TestProfileStepConfig:
    snapshot: SnapshotRef
    slow_test_threshold_ms: float = 2000.0
    io_spec: dict[str, object] | None = None
    refresh_subsystem_cache: bool = True
    benchmark_subsystem_cache: bool = False
```

**Migration Tasks**:

1. **Create `TestProfileOptions` dataclass**:
   ```python
   @dataclass(frozen=True)
   class TestProfileOptions:
       slow_test_threshold_ms: float = 2000.0
       io_spec: dict[str, object] | None = None
       refresh_subsystem_cache: bool = True
       benchmark_subsystem_cache: bool = False
   ```

2. **Update `build_test_profile`**:
   ```python
   def build_test_profile(
       gateway: StorageGateway,
       snapshot: SnapshotRef,
       *,
       options: TestProfileOptions | None = None,
   ) -> None:
   ```

3. **Update `build_test_profile_context`** in `rows.py`:
   - Change signature from `cfg: TestProfileStepConfig` to `snapshot: SnapshotRef`
   - Pass options separately

4. **Update `write_test_profile_rows`** in `rows.py`:
   - Change signature from `cfg: TestProfileStepConfig` to `snapshot: SnapshotRef`

5. **Update build plugin** to construct options and pass `ctx.snapshot`

---

## Phase 5.3: Graph & History Migrations

### 5.3.1 HistoryTimeseriesStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/history/history_timeseries.py` | Core computation |
| `src/codeintel/build/plugins/analytics/history/timeseries.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class HistoryTimeseriesStepConfig:
    snapshot: SnapshotRef
    commits: tuple[str, ...]
    entity_kind: str = "function"
    max_entities: int = 500
    selection_strategy: str = "risk_score"
```

**Migration Tasks**:

1. **Create `HistoryTimeseriesOptions`**:
   ```python
   @dataclass(frozen=True)
   class HistoryTimeseriesOptions:
       commits: tuple[str, ...]
       entity_kind: str = "function"
       max_entities: int = 500
       selection_strategy: str = "risk_score"
   ```

2. **Update functions**:
   - `compute_history_timeseries(history_gateway, snapshot, db_resolver, *, options, runner)`
   - `compute_history_timeseries_gateways(history_gateway, snapshot, snapshot_resolver, *, options, runner)`
   - `_select_entities(snapshot, options, db_resolver)`
   - `_select_top_functions(con, snapshot, options, commit)`
   - `_select_top_modules(con, snapshot, options, commit)`
   - `_collect_function_rows_for_commit(snapshot, options, con_ci, *, commit_ctx, selection)`
   - `_collect_module_rows_for_commit(snapshot, options, con_ci, *, commit_ctx, selection)`

---

### 5.3.2 FunctionEffectsStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/functions/function_effects.py` | Core computation |
| `src/codeintel/build/plugins/analytics/functions/effects.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class FunctionEffectsStepConfig:
    snapshot: SnapshotRef
    max_call_depth: int = 3
    require_all_callees_pure: bool = True
    io_apis: dict[str, list[str]] = field(default_factory=dict)
    db_apis: dict[str, list[str]] = field(default_factory=dict)
    time_apis: dict[str, list[str]] = field(default_factory=dict)
    random_apis: dict[str, list[str]] = field(default_factory=dict)
    threading_apis: dict[str, list[str]] = field(default_factory=dict)
```

**Migration Tasks**:

1. **Create `FunctionEffectsOptions`**:
   ```python
   @dataclass(frozen=True)
   class FunctionEffectsOptions:
       max_call_depth: int = 3
       require_all_callees_pure: bool = True
       io_apis: dict[str, list[str]] = field(default_factory=dict)
       db_apis: dict[str, list[str]] = field(default_factory=dict)
       time_apis: dict[str, list[str]] = field(default_factory=dict)
       random_apis: dict[str, list[str]] = field(default_factory=dict)
       threading_apis: dict[str, list[str]] = field(default_factory=dict)
   ```

2. **Update `compute_function_effects`**:
   ```python
   def compute_function_effects(
       gateway: StorageGateway,
       snapshot: SnapshotRef,
       *,
       options: FunctionEffectsOptions | None = None,
   ) -> None:
   ```

---

### 5.3.3 BehavioralCoverageStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/testing/profiles/builder.py` | `build_behavioral_coverage` |
| `src/codeintel/analytics/testing/profiles/rows.py` | Row utilities |
| `src/codeintel/build/plugins/analytics/tests/behavioral_coverage.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class BehavioralCoverageStepConfig:
    snapshot: SnapshotRef
    heuristic_version: str = "v1"
    enable_llm: bool = False
    llm_model: str | None = None
```

**Migration Tasks**:

1. **Create `BehavioralCoverageOptions`**:
   ```python
   @dataclass(frozen=True)
   class BehavioralCoverageOptions:
       heuristic_version: str = "v1"
       enable_llm: bool = False
       llm_model: str | None = None
   ```

2. **Update `build_behavioral_coverage`**:
   ```python
   def build_behavioral_coverage(
       gateway: StorageGateway,
       snapshot: SnapshotRef,
       *,
       options: BehavioralCoverageOptions | None = None,
       llm_runner: BehavioralLLMRunner | None = None,
   ) -> None:
   ```

3. **Update supporting functions** in `behavioral/tags.py`:
   - `build_behavior_rows(gateway, snapshot, *, options, llm_runner)`

4. **Update `write_behavioral_coverage_rows`** in `rows.py`

---

### 5.3.4 CallGraphStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/build/plugins/graphs/builders/callgraph.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class CallGraphStepConfig:
    snapshot: SnapshotRef
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
```

**Note**: This step config is already only used in the plugin layer. The migration involves:

1. **Update `CallGraphPlugin.execute`**:
   - Remove `CallGraphStepConfig` construction
   - Access `ctx.snapshot.repo`, `ctx.snapshot.commit`, `ctx.snapshot.repo_root` directly
   - Pass collectors via `ctx.parameters` if needed

---

### 5.3.5 CFGBuilderStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/build/plugins/graphs/builders/cfg_dfg.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class CFGBuilderStepConfig:
    snapshot: SnapshotRef
    cfg_builder: Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None = None
```

**Migration Tasks**:
- Remove `CFGBuilderStepConfig` construction in plugin
- Access `ctx.snapshot` directly
- Pass builder callable via `ctx.parameters` if needed

---

### 5.3.6 SymbolUsesStepConfig Migration

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/build/plugins/graphs/builders/symbol_uses.py` | Build plugin |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class SymbolUsesStepConfig:
    snapshot: SnapshotRef
    paths: BuildPaths
    scip_json_path: Path | None = None
```

**Migration Tasks**:

1. **Create `SymbolUsesOptions`**:
   ```python
   @dataclass(frozen=True)
   class SymbolUsesOptions:
       scip_json_path: Path | None = None
   ```

2. **Update plugin**:
   - Access `ctx.snapshot` and `ctx.paths` directly
   - Pass `scip_json_path` via `ctx.parameters` or options

---

## Phase 5.4: GraphMetricsStepConfig (Special Case)

**Status**: ⚠️ Complex - Requires careful refactoring

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/analytics/graphs/graph_metrics.py` | Core metrics computation |
| `src/codeintel/analytics/graphs/graph_metrics_ext.py` | Extended metrics |
| `src/codeintel/analytics/graphs/module_graph_metrics_ext.py` | Module-level metrics |
| `src/codeintel/analytics/graphs/subsystem_graph_metrics.py` | Subsystem metrics |
| `src/codeintel/build/plugins/graphs/metrics/core.py` | Core metrics plugin |
| `src/codeintel/build/plugins/graphs/metrics/secondary.py` | Secondary metrics plugin |
| `src/codeintel/build/plugins/graphs/validation.py` | Validation utilities |
| `src/codeintel/analytics/runtime/context.py` | Runtime context |
| `tests/_helpers/graph_runtime_harness.py` | Test harness |

**Step Config Structure**:
```python
@dataclass(frozen=True)
class GraphMetricsStepConfig:
    snapshot: SnapshotRef | None = None
    repo: str = ""
    commit: str = ""
    repo_root: Path | None = None
    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    seed: int = 0
    enabled_plugins: tuple[str, ...] = ()
    disabled_plugins: tuple[str, ...] = ()
    plugin_options: dict[str, dict[str, object]] = field(default_factory=dict)
    plugin_policy: GraphPluginPolicy = field(default_factory=GraphPluginPolicy)
    scope: GraphRunScope = field(default_factory=GraphRunScope)
```

**Migration Strategy**:

This is the most complex step config due to its:
- Multiple nested config objects (`GraphPluginPolicy`, `GraphRunScope`, `GraphMetricWeights`, etc.)
- Special `__post_init__` normalization logic
- Extensive use across 9+ files

**Proposed Approach**:

1. **Create composite options structure**:
   ```python
   @dataclass(frozen=True)
   class GraphMetricsTuningOptions:
       max_betweenness_sample: int | None = 200
       eigen_max_iter: int = 200
       seed: int = 0

   @dataclass(frozen=True)
   class GraphMetricsWeightOptions:
       pagerank_weight: str | None = "weight"
       betweenness_weight: str | None = "weight"

   @dataclass(frozen=True)
   class GraphMetricsPluginOptions:
       enabled_plugins: tuple[str, ...] = ()
       disabled_plugins: tuple[str, ...] = ()
       plugin_options: dict[str, dict[str, object]] = field(default_factory=dict)
       plugin_policy: GraphPluginPolicy = field(default_factory=GraphPluginPolicy)

   @dataclass(frozen=True)
   class GraphMetricsOptions:
       tuning: GraphMetricsTuningOptions = field(default_factory=GraphMetricsTuningOptions)
       weights: GraphMetricsWeightOptions = field(default_factory=GraphMetricsWeightOptions)
       plugins: GraphMetricsPluginOptions = field(default_factory=GraphMetricsPluginOptions)
       scope: GraphRunScope = field(default_factory=GraphRunScope)
   ```

2. **Keep `GraphRunScope` and `GraphPluginPolicy`** as they are used by the serving layer

3. **Update all compute functions** to accept `snapshot: SnapshotRef` + `options: GraphMetricsOptions`

4. **Update runtime context** to use new options structure

---

## Phase 5.5: Public API Updates

**File**: `src/codeintel/analytics/functions/__init__.py`

**Migration Tasks**:

1. **Update all exported function signatures** to use `SnapshotRef` pattern
2. **Ensure backward compatibility** by keeping old function names but with new signatures
3. **Update docstrings** to reflect new API
4. **Update `__all__` exports**

---

## Phase 5.6: Module Deletion & Config Cleanup

**Files to Delete**:
| File | Reason |
|------|--------|
| `src/codeintel/config/steps_analytics.py` | All step configs migrated |
| `src/codeintel/config/steps_graphs.py` | All step configs migrated |

**Files to Update**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/config/__init__.py` | Remove step config imports and exports |
| `src/codeintel/config/builder.py` | Remove step builder methods |

**Migration Tasks**:

1. **Remove imports from `config/__init__.py`**:
   ```python
   # Remove these imports
   from codeintel.config.steps_analytics import (
       AnalyticsStepBuilder,
       BehavioralCoverageStepConfig,
       CoverageAnalyticsStepConfig,
       ...
   )
   from codeintel.config.steps_graphs import (
       CallGraphStepConfig,
       CFGBuilderStepConfig,
       ...
   )
   ```

2. **Update `__all__` exports**

3. **Remove `AnalyticsStepBuilder` and `GraphStepBuilder`** from `config/builder.py`

4. **Delete the deprecated modules**

---

## Phase 5.7: Serving Layer Updates

**Files to Modify**:
| File | Changes Required |
|------|------------------|
| `src/codeintel/serving/backend/query_api.py` | Update GraphRunScope usage |
| `src/codeintel/serving/backend/profile_backend.py` | Update scope handling |
| `src/codeintel/serving/backend/function_backend.py` | Update scope handling |
| `src/codeintel/serving/mcp/models.py` | Ensure GraphScopePayload is primary |

**Strategy**:

The serving layer currently uses `GraphRunScope` from `config/steps_graphs.py`. This class should be:

1. **Relocated** to a more appropriate location (e.g., `serving/types.py` or `graphs/runtime/scope.py`)
2. **Kept as-is** since it's a pure data structure with no step config dependency
3. **Updated imports** in all serving files

**Recommended Location**: `src/codeintel/graphs/runtime/scope.py`

---

## Phase 5.8: Final Verification

### Quality Gates

```bash
# 1. Run quality report
uv run python -m tools.quality_report --output build/quality-results/quality_report.json

# 2. Run full test suite
uv run pytest -q

# 3. Verify no remaining step config imports
grep -r "steps_analytics\|steps_graphs" src/ --include="*.py" | grep -v "__pycache__"
grep -r "StepConfig" src/ --include="*.py" | grep -v "__pycache__" | grep -v "# Migration"

# 4. Verify serving layer still works
uv run pytest tests/serving/ -q
```

### Checklist

- [ ] All `XXXStepConfig` classes removed from imports
- [ ] `steps_analytics.py` deleted
- [ ] `steps_graphs.py` deleted
- [ ] `config/__init__.py` cleaned up
- [ ] `config/builder.py` cleaned up
- [ ] All tests passing
- [ ] No pyright errors
- [ ] No ruff errors
- [ ] No pyrefly errors

---

## Implementation Order

### Wave 1: Complete In-Progress (2 tasks)
1. `phase5-2-test-coverage` - TestCoverageStepConfig
2. `phase5-2-test-profile` - TestProfileStepConfig

### Wave 2: History & Effects (3 tasks)
3. `phase5-3-history-timeseries` - HistoryTimeseriesStepConfig
4. `phase5-3-function-effects` - FunctionEffectsStepConfig
5. `phase5-3-behavioral-coverage` - BehavioralCoverageStepConfig

### Wave 3: Graph Builders (3 tasks)
6. `phase5-3-call-graph` - CallGraphStepConfig
7. `phase5-3-cfg-dfg` - CFGBuilderStepConfig
8. `phase5-3-symbol-uses` - SymbolUsesStepConfig

### Wave 4: Complex Migration (1 task)
9. `phase5-4-graph-metrics` - GraphMetricsStepConfig (most complex)

### Wave 5: Cleanup (4 tasks)
10. `phase5-5-public-apis` - Update public API exports
11. `phase5-6-delete-modules` - Delete deprecated modules
12. `phase5-7-serving` - Update serving layer
13. `phase5-verification` - Final verification

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking serving layer | Keep `GraphRunScope` as stable interface; relocate only |
| Test failures | Run tests after each migration; fix immediately |
| Type errors | Run pyright after each file change |
| Missing imports | Use grep to find all usages before deletion |

---

## Success Criteria

1. **Zero** remaining `XXXStepConfig` imports in `src/`
2. **Both** `steps_analytics.py` and `steps_graphs.py` deleted
3. **All** tests passing (including serving layer tests)
4. **All** quality gates passing (ruff, pyright, pyrefly)
5. **No** deprecation warnings in normal operation

---

## Appendix: File-by-File Migration Reference

### Complete Migration Status

| Step Config | Status | Files |
|-------------|--------|-------|
| `CoverageAnalyticsStepConfig` | ✅ Complete | 2 files |
| `DataModelsStepConfig` | ✅ Complete | 2 files |
| `DataModelUsageStepConfig` | ✅ Complete | 2 files |
| `EntryPointsStepConfig` | ✅ Complete | 2 files |
| `ExternalDependenciesStepConfig` | ✅ Complete | 2 files |
| `FunctionAnalyticsStepConfig` | ✅ Complete | 3 files |
| `FunctionContractsStepConfig` | ✅ Complete | 2 files |
| `FunctionHistoryStepConfig` | ✅ Complete | 2 files |
| `GoidBuilderStepConfig` | ✅ Complete | 1 file |
| `HotspotsStepConfig` | ✅ Complete | 2 files |
| `ImportGraphStepConfig` | ✅ Complete | 1 file |
| `ProfilesAnalyticsStepConfig` | ✅ Complete | 5 files |
| `SemanticRolesStepConfig` | ✅ Complete | 2 files |
| `SubsystemsStepConfig` | ✅ Complete | 5 files |
| `ConfigDataFlowStepConfig` | ✅ Complete | 2 files |
| `TestCoverageStepConfig` | 🔄 In Progress | 2 files |
| `TestProfileStepConfig` | 🔄 In Progress | 5 files |
| `HistoryTimeseriesStepConfig` | ⏳ Pending | 2 files |
| `FunctionEffectsStepConfig` | ⏳ Pending | 2 files |
| `BehavioralCoverageStepConfig` | ⏳ Pending | 3 files |
| `CallGraphStepConfig` | ⏳ Pending | 1 file |
| `CFGBuilderStepConfig` | ⏳ Pending | 1 file |
| `SymbolUsesStepConfig` | ⏳ Pending | 1 file |
| `GraphMetricsStepConfig` | ⏳ Pending | 9 files |

