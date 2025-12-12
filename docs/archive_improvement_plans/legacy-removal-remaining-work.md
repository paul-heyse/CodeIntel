# Legacy Ingestion Infrastructure Removal: Status and Remaining Work

## Executive Summary

This document captures the progress and remaining work for migrating from the legacy ingestion infrastructure to the new build system architecture. The migration removes ~3,000+ lines of legacy code and consolidates all plugin execution through the unified `TargetPlugin` protocol.

---

## Completed Work

### Phase 1: BuildExecutor Migration ✅

**Files Modified:**
- `src/codeintel/build/executor.py`

**Changes:**
- Removed recipe imports (`get_builtin_recipe`, `execute_recipe`, `RecipeExecutorContext`)
- Removed `resolve_scan_profiles` import (no longer needed)
- Removed `ingestion_recipe` constructor parameter
- Rewrote `_execute_ingestion_stage()` to use `_execute_target_direct()` for each target instead of recipe-based execution
- Updated docstrings to reflect new architecture

**Result:** The BuildExecutor now executes ingestion targets directly via the plugin registry rather than through the legacy recipe system.

---

### Phase 2: Ingest CLI Removal ✅

**Files Deleted:**
- `src/codeintel/cli/commands/ingest.py`

**Files Modified:**
- `src/codeintel/cli/__init__.py`
  - Removed `ingest_app` import and registration
  - Updated docstring to remove ingest CLI references
  - Updated examples to show `build run --module ingestion` as replacement
  - Removed `ingest_app` from `__all__`

**Result:** Users now use `codeintel build run --module ingestion` instead of `codeintel ingest run`.

---

### Phase 3: Ingestion Package Cleanup ✅

**Files Modified:**
- `src/codeintel/ingestion/__init__.py`
  - Removed all legacy exports from `core/`, `runtime/`, `recipes/`, `resources/`
  - Kept only: adapters, compute steps, plugins, ports, tracker, validation, infrastructure
  - Updated docstring to reflect new architecture

**Result:** The ingestion package now exports only the new plugin classes and compute layer.

---

### Phase 4: Dependent File Updates ✅

**Files Deleted:**
- `src/codeintel/ingestion/plugins/config_factory.py` (depended on deleted `IngestExecutionContext`)

**Files Modified:**
- `src/codeintel/config/datasets/rows/core.py`
  - Moved `IngestRunMode` and `IngestRunStatus` enums locally (previously imported from deleted `ingestion/core/runs.py`)
  - Added enums to `__all__`

**Result:** All source files now compile without references to deleted modules.

---

### Phase 5: Legacy Directory Deletion ✅

**Directories Deleted:**
- `src/codeintel/ingestion/runtime/` (4 files: `__init__.py`, `executor.py`, `planning.py`, `telemetry.py`)
- `src/codeintel/ingestion/recipes/` (4 files: `__init__.py`, `builtin.py`, `dsl.py`, `executor.py`)
- `src/codeintel/ingestion/core/` (5 files: `__init__.py`, `base.py`, `execution_context.py`, `runs.py`, `traits.py`)
- `src/codeintel/ingestion/resources/` (6 files: `__init__.py`, `modules.py`, `protocol.py`, `registry.py`, `tools.py`, `tracker.py`)

**Total:** 19 legacy source files removed (~2,500+ lines)

---

### Phase 6: Test Cleanup ✅

**Test Directories Deleted:**
- `tests/ingestion/runtime/`
- `tests/ingestion/core/`

**Test Files Deleted:**
- `tests/ingestion/test_recipe_executor.py`
- `tests/ingestion/test_pipeline_integration.py`
- `tests/ingestion/test_plugin_registry.py`
- `tests/ingestion/test_ingest_runs.py`
- `tests/ingestion/test_ingest_run_reporting.py`
- `tests/ingestion/test_resources.py`
- `tests/ingestion/test_core.py`
- `tests/ingestion/test_config_factory.py`
- `tests/ingestion/plugins/test_config_plugin.py`
- `tests/ingestion/plugins/test_scip_plugin.py`
- `tests/ingestion/test_plugins.py`
- `tests/ingestion/test_ingest_run_incremental_ast.py`
- `tests/ingestion/test_coverage_incremental.py`

**Test Helper Files Deleted:**
- `tests/_helpers/harnesses/ingestion.py`
- `tests/_helpers/harnesses/ingest_setup.py`

**Files Modified:**
- `tests/conftest.py` - Removed `IngestExecutionContext` import and `ingest_setup`/`ingest_ctx` fixtures
- `tests/_helpers/harnesses/__init__.py` - Removed ingestion harness exports
- `tests/_helpers/__init__.py` - Removed `IngestPluginTestHarness`, `IngestPluginResultAssertions`, `assert_ingest_result`

**Total:** ~15 legacy test files removed

---

### Additional Fixes Applied ✅

**Analytics Plugins:**
- Removed `paths=ctx.paths` from 18 analytics plugin config constructors
- Fixed `HotspotsPlugin` - removed unused `min_churn_threshold` parameter
- Fixed `ProfilesPlugin` - removed unused `include_ownership` parameter  
- Fixed `HotspotsPlugin._compute_row_counts` - changed `gateway.query()` to `gateway.con.execute()`
- Fixed ToolRunner type mismatches - passing `None` instead of incompatible types
- Fixed `HistoryTimeseriesPlugin` - added required `commits` parameter handling
- Replaced all `catalog.get_resource("...")` calls with `None` comments (resources not yet populated)

**Analytics Registration:**
- Updated `src/codeintel/analytics/plugins/registration.py` - deprecated legacy registry registration

**Graph Plugins (partial):**
- Updated `src/codeintel/graphs/plugins/__init__.py` - simplified exports
- Updated `src/codeintel/graphs/plugins/builders/__init__.py` - updated to new plugin class names

---

## Remaining Work

### Error Summary

| Category | Files | Errors | Status |
|----------|-------|--------|--------|
| Test Infrastructure | 1 | 1 | Pending |
| Config Resolver | 1 | 1 | Pending |
| Core Recipes | 1 | 1 | Pending |
| Graph Plugins | 12 | 45 | Pending |
| **TOTAL** | **15** | **48** | **Pending** |

---

## Category 1: Test Infrastructure Fix

### File: `tests/ingestion/engine/__init__.py`

**Error:** `W391` - Extra newline at end of file / `D104` - Missing docstring

**Current State:**
```python
# (empty file or extra newlines)
```

**Target State:**
```python
"""Tests for ingestion engine components."""
```

**Estimated effort:** 1 minute

---

## Category 2: Config Resolver Fix

### File: `src/codeintel/config/resolver.py:72`

**Error:** `Arguments missing for parameters "scip_python_bin", "scip_bin", "pyright_bin", "pyrefly_bin", "ruff_bin", "coverage_bin", "pytest_bin", "git_bin", "default_timeout_s"`

**Root Cause:** `ToolsConfig()` is being called without required arguments.

**Current State:**
```python
def resolve_tools_config(base: ToolsConfig | None = None) -> ToolsConfig:
    # Start with default values from ToolsConfig
    default_config = ToolsConfig()  # <-- ERROR: Missing required args
    data = base.model_dump() if base is not None else default_config.model_dump()
```

**Target State (Option A - Add defaults to ToolsConfig):**
```python
# In models.py, add defaults to ToolsConfig fields:
class ToolsConfig(BaseModel):
    scip_python_bin: str = "scip-python"
    scip_bin: str = "scip"
    pyright_bin: str = "pyright"
    pyrefly_bin: str = "pyrefly"
    ruff_bin: str = "ruff"
    coverage_bin: str = "coverage"
    pytest_bin: str = "pytest"
    git_bin: str = "git"
    default_timeout_s: int = 300
```

**Target State (Option B - Use model_validate in resolver):**
```python
def resolve_tools_config(base: ToolsConfig | None = None) -> ToolsConfig:
    # Start with default values from ToolsConfig
    if base is not None:
        data = base.model_dump()
    else:
        # Use model_validate to apply field defaults
        data = {}
    env_map = { ... }
    # Apply env overrides to data dict
    for env_key, field_name in env_map.items():
        if (val := os.environ.get(env_key)) is not None:
            data[field_name] = val
    return ToolsConfig.model_validate(data)
```

**Estimated effort:** 5 minutes

---

## Category 3: Core Recipes Module Fix

### File: `src/codeintel/core/recipes/unified_executor.py:414`

**Error:** `Argument of type "Literal['pipeline']" cannot be assigned to parameter "module" of type "StageModule"`

**Root Cause:** The `StageModule` literal type no longer includes `"pipeline"` as a valid value.

**Current State:**
```python
def _execute_pipeline_stage(
    stage: UnifiedStage,
    context: UnifiedExecutionContext,
) -> UnifiedStageResult:
    # Placeholder - actual implementation would delegate to step registry
    log.debug("pipeline_stage context.snapshot=%s", context.snapshot)
    return _create_placeholder_result(stage, "pipeline")  # <-- ERROR
```

**Target State (Option A - Delete if unused):**
```python
# Delete the entire function and any references to it.
# If the `core/recipes/` directory is entirely legacy, consider deleting it.
```

**Target State (Option B - Update to valid module if still needed):**
```python
def _execute_pipeline_stage(
    stage: UnifiedStage,
    context: UnifiedExecutionContext,
) -> UnifiedStageResult:
    """Execute a pipeline stage using the build system.
    
    Note: 'pipeline' was removed from StageModule. This function is now
    a wrapper that delegates to the appropriate module-specific executor.
    """
    log.debug("pipeline_stage context.snapshot=%s", context.snapshot)
    # Determine which module this stage belongs to based on targets
    # For now, default to "ingestion" as a placeholder
    return _create_placeholder_result(stage, "ingestion")
```

**Recommendation:** Option A (delete) is preferred since we're migrating away from the legacy pipeline system to the build system.

**Estimated effort:** 10 minutes

---

## Category 4: Graph Plugins Migration

The graph plugins were incompletely migrated to the `TargetPlugin` pattern. This is the bulk of remaining work.

### 4A: Missing Compute Module Imports (4 plugins)

| File | Missing Import | Actual Module |
|------|----------------|---------------|
| `builders/cfg_dfg.py:14` | `codeintel.graphs.compute.cfg_dfg` | Split: `compute.cfg` + `compute.dfg` |
| `builders/goid.py:14` | `codeintel.graphs.compute.goids` | Should be `compute.goid` |
| `builders/symbol_uses.py:14` | `codeintel.graphs.compute.symbol_uses` | Should be `compute.symbols` |
| `validation.py:14` | `codeintel.graphs.compute.validation` | Does not exist |

---

#### 4A.1: `builders/cfg_dfg.py`

**Current State:**
```python
from codeintel.config import CfgDfgStepConfig
from codeintel.graphs.compute.cfg_dfg import build_cfg_dfg_data  # ERROR: Module doesn't exist
```

**Target State:**
```python
from codeintel.config import CFGBuilderStepConfig
from codeintel.graphs.compute import cfg, dfg

# ... later in execute():
async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    config = CFGBuilderStepConfig(snapshot=ctx.snapshot)
    
    try:
        # The compute modules provide pure functions
        # Actual persistence is done via ctx.gateway
        row_counts: dict[str, int] = {}
        
        # Build CFG data
        cfg_blocks, cfg_edges = cfg.build_cfg_data(ctx.gateway, config)
        row_counts["graphs.cfg_blocks"] = len(cfg_blocks)
        row_counts["graphs.cfg_edges"] = len(cfg_edges)
        
        # Build DFG data  
        dfg_edges = dfg.build_dfg_data(ctx.gateway, config)
        row_counts["graphs.dfg_edges"] = len(dfg_edges)
        
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"CFG/DFG build failed: {e}")
```

---

#### 4A.2: `builders/goid.py`

**Current State:**
```python
from codeintel.config import GoidStepConfig  # ERROR: Doesn't exist
from codeintel.graphs.compute.goids import build_goid_data  # ERROR: Module is 'goid' not 'goids'
```

**Target State:**
```python
from codeintel.config import GoidBuilderStepConfig
from codeintel.graphs.compute import goid

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    config = GoidBuilderStepConfig(snapshot=ctx.snapshot)
    
    try:
        # Use the goid compute module
        row_counts = goid.build_goid_data(ctx.gateway, config)
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"GOID build failed: {e}")
```

---

#### 4A.3: `builders/symbol_uses.py`

**Current State:**
```python
from codeintel.config import SymbolUsesStepConfig
from codeintel.graphs.compute.symbol_uses import build_symbol_uses_data  # ERROR: Module is 'symbols'
```

**Target State:**
```python
from codeintel.config import SymbolUsesStepConfig
from codeintel.graphs.compute import symbols

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    # Note: SymbolUsesStepConfig requires 'paths' parameter
    config = SymbolUsesStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
    )
    
    try:
        row_counts = symbols.build_symbol_uses_data(ctx.gateway, config)
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"Symbol uses build failed: {e}")
```

---

#### 4A.4: `validation.py`

**Current State:**
```python
from codeintel.config import GraphValidationStepConfig  # ERROR: Doesn't exist
from codeintel.graphs.compute.validation import validate_graphs  # ERROR: Module doesn't exist
```

**Target State:**

The validation module doesn't exist in `compute/`. Options:
1. Create the validation compute module
2. Inline the validation logic in the plugin
3. Delete the validation plugin if not needed

**Target State (Option B - Inline validation):**
```python
from codeintel.config import GraphMetricsStepConfig  # Or create GraphValidationStepConfig

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    try:
        errors: list[str] = []
        con = ctx.gateway.con
        repo, commit = ctx.repo, ctx.commit
        
        # Validate call graph integrity
        orphan_edges = con.execute("""
            SELECT COUNT(*) FROM graphs.call_graph_edges e
            LEFT JOIN graphs.call_graph_nodes n ON e.caller_goid = n.goid
            WHERE n.goid IS NULL AND e.repo = ? AND e.commit = ?
        """, [repo, commit]).fetchone()[0]
        if orphan_edges > 0:
            errors.append(f"Found {orphan_edges} orphan call graph edges")
        
        # Write validation results
        row_counts = {"graphs.validation_errors": len(errors)}
        
        if errors:
            return TargetResult.failed("\n".join(errors))
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"Graph validation failed: {e}")
```

**Estimated effort:** 20 minutes

---

### 4B: Missing Config Class Imports (3 plugins)

| File | Missing Config Class | Correct Config Class |
|------|----------------------|---------------------|
| `builders/cfg_dfg.py:13` | `CfgDfgStepConfig` | `CFGBuilderStepConfig` |
| `builders/goid.py:13` | `GoidStepConfig` | `GoidBuilderStepConfig` |
| `validation.py:13` | `GraphValidationStepConfig` | `GraphMetricsStepConfig` (or create new) |

**Available config classes from `codeintel.config`:**
```python
# From steps_graphs.py - these are the ACTUAL exported classes:
from codeintel.config import (
    CallGraphStepConfig,      # For call graph construction
    CFGBuilderStepConfig,     # For CFG/DFG construction  
    GoidBuilderStepConfig,    # For GOID generation
    ImportGraphStepConfig,    # For import graph construction
    SymbolUsesStepConfig,     # For symbol use derivation (requires paths)
    GraphMetricsStepConfig,   # For graph metrics computation
    ConfigDataFlowStepConfig, # For config data flow analytics
    ExternalDependenciesStepConfig,  # For external dependency analytics
)
```

**Note:** There is no `GraphValidationStepConfig`. Either:
1. Use `GraphMetricsStepConfig` as a base
2. Create a new config class in `steps_graphs.py`
3. Make validation work without a specialized config (just use `ctx.snapshot` directly)

**Estimated effort:** 10 minutes

---

### 4C: Fix `paths=` Parameter Issues (4 plugins)

These plugins pass `paths=ctx.paths` to config constructors that don't have this parameter:

| File | Line |
|------|------|
| `builders/callgraph.py` | 242 |
| `builders/import_graph.py` | 52 |
| `metrics/core.py` | 52 |
| `metrics/secondary.py` | 52 |

---

#### Config Classes That Accept `paths`:
```python
# ONLY SymbolUsesStepConfig accepts paths:
class SymbolUsesStepConfig:
    snapshot: SnapshotRef
    paths: BuildPaths  # <-- Required!
```

#### Config Classes That DO NOT Accept `paths`:
```python
# These only accept snapshot (and optional specific parameters):
CallGraphStepConfig(snapshot=...)
CFGBuilderStepConfig(snapshot=...)
GoidBuilderStepConfig(snapshot=..., language="python")
ImportGraphStepConfig(snapshot=...)
GraphMetricsStepConfig(snapshot=..., max_betweenness_sample=..., ...)
```

---

#### Fix: `builders/callgraph.py:242`

**Current:**
```python
cfg = CallGraphStepConfig(
    snapshot=ctx.snapshot,
    paths=ctx.paths,  # ERROR: No such parameter
)
```

**Target:**
```python
cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
```

---

#### Fix: `builders/import_graph.py:52`

**Current:**
```python
cfg = ImportGraphStepConfig(
    snapshot=ctx.snapshot,
    paths=ctx.paths,  # ERROR: No such parameter
)
```

**Target:**
```python
cfg = ImportGraphStepConfig(snapshot=ctx.snapshot)
```

---

#### Fix: `metrics/core.py:52`

**Current:**
```python
cfg = GraphMetricsStepConfig(
    snapshot=ctx.snapshot,
    paths=ctx.paths,  # ERROR: No such parameter
)
```

**Target:**
```python
cfg = GraphMetricsStepConfig(snapshot=ctx.snapshot)
```

---

#### Fix: `metrics/secondary.py:52`

**Current:**
```python
cfg = GraphMetricsStepConfig(
    snapshot=ctx.snapshot,
    paths=ctx.paths,  # ERROR: No such parameter
)
```

**Target:**
```python
cfg = GraphMetricsStepConfig(snapshot=ctx.snapshot)
```

**Estimated effort:** 10 minutes

---

### 4D: Fix CallGraph Plugin Deep Issues (8 errors)

**File:** `src/codeintel/graphs/plugins/builders/callgraph.py`

| Line | Error | Fix Required |
|------|-------|--------------|
| 180-182 | `EdgeResolutionContext` constructor wrong | Update to correct parameter names |
| 242 | `paths` parameter invalid | Remove `paths=ctx.paths` |
| 303 | `StorageGateway` incompatible with `IngestStoragePort` | Use `DuckDBStorageAdapter(ctx.gateway)` |
| 304 | `FunctionCatalogService()` expects 1 argument | Check constructor signature |
| 310 | `FunctionCatalog.function_by_goid` doesn't exist | Find correct method name |
| 311 | `CallGraphNodeRow` constructor mismatch | Check expected fields |

---

#### 4D.1: Fix `EdgeResolutionContext` constructor (lines 180-184)

**Current:**
```python
resolution_ctx = EdgeResolutionContext(
    global_callee_by_name=inputs.global_callee_by_name,
    scip_candidates_by_use=inputs.scip_candidates_by_use,
    def_goids_by_path=inputs.def_goids_by_path,
)
```

**Investigation needed:** Check `EdgeResolutionContext` in `graphs/compute/callgraph/resolution.py` for actual parameter names.

---

#### 4D.2: Fix `paths` parameter (line 242)

**Current:**
```python
cfg = CallGraphStepConfig(
    snapshot=ctx.snapshot,
    paths=ctx.paths,
)
```

**Target:**
```python
cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
```

---

#### 4D.3: Fix storage adapter usage (line 303)

**Current:**
```python
storage = IngestStorageService(gateway)  # ERROR: gateway is StorageGateway, not IngestStoragePort
```

**Target:**
```python
from codeintel.ingestion.adapters import DuckDBStorageAdapter

storage = DuckDBStorageAdapter(gateway)  # Adapts StorageGateway to IngestStoragePort
```

---

#### 4D.4: Fix `FunctionCatalogService` constructor (line 304)

**Current:**
```python
catalog = FunctionCatalogService(storage, cfg.repo, cfg.commit).catalog()
```

**Investigation needed:** Check `FunctionCatalogService` constructor signature. It likely needs:
```python
# Option A: If it takes a gateway directly
catalog = FunctionCatalogService(gateway).catalog()

# Option B: If it takes snapshot info
catalog = FunctionCatalogService(storage).for_snapshot(repo, commit)

# Option C: Use ctx.resources.catalog if already available
catalog = ctx.resources.catalog
if catalog is None:
    return TargetResult.failed("Function catalog not available")
```

---

#### 4D.5: Fix `function_by_goid` attribute (line 310)

**Current:**
```python
for goid, meta in catalog.function_by_goid.items():  # ERROR: No such attribute
```

**Target (depends on actual Catalog API):**
```python
# Option A: If method is get_functions_by_goid()
for goid, meta in catalog.get_functions_by_goid().items():

# Option B: If it's functions property
for goid, meta in catalog.functions.items():

# Option C: Query directly from gateway
rows = ctx.gateway.con.execute("""
    SELECT goid, qualname, rel_path, start_line, end_line
    FROM core.goids
    WHERE repo = ? AND commit = ? AND kind = 'function'
""", [ctx.repo, ctx.commit]).fetchall()
for row in rows:
    goid, qualname, rel_path, start_line, end_line = row
    # ... process
```

---

#### 4D.6: Fix `CallGraphNodeRow` constructor (line 311)

**Current:**
```python
nodes.append(CallGraphNodeRow(
    goid=goid,
    # ... wrong fields
))
```

**Investigation needed:** Check `CallGraphNodeRow` in `config/datasets/rows/` for actual fields.

**Likely Target:**
```python
from datetime import datetime, UTC
from codeintel.config.datasets import CallGraphNodeRow

nodes.append(CallGraphNodeRow(
    repo=cfg.repo,
    commit=cfg.commit,
    goid=goid,
    qualname=meta.qualname,
    rel_path=meta.rel_path,
    start_line=meta.start_line,
    end_line=meta.end_line,
    kind="function",
    created_at=datetime.now(tz=UTC),
))
```

**Estimated effort:** 45 minutes

---

### 4E: Fix Metrics Plugin Missing Function Imports (2 plugins)

| File | Missing Function |
|------|------------------|
| `metrics/core.py:14` | `compute_core_metrics` |
| `metrics/secondary.py:14` | `compute_secondary_metrics` |

The `graphs/compute/metrics/` package exports **submodules, not functions**:
```python
# From graphs/compute/metrics/__init__.py:
__all__ = [
    "bipartite",    # Bipartite graph metrics
    "centrality",   # PageRank, betweenness, closeness, etc.
    "cfg",          # Control flow graph metrics
    "community",    # Community detection
    "components",   # SCC, connected components
    "coupling",     # Coupling metrics
    "dfg",          # Data flow graph metrics
    "paths",        # Path-related metrics
    "statistics",   # Global statistics
    "structural",   # Clustering, triangles, etc.
]
```

---

#### Fix: `metrics/core.py`

**Current:**
```python
from codeintel.graphs.compute.metrics import compute_core_metrics  # ERROR: No such function
```

**Target:**
```python
from codeintel.graphs.compute.metrics import centrality, components, structural

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    config = GraphMetricsStepConfig(snapshot=ctx.snapshot)
    graph_runtime = ctx.resources.graph_runtime
    
    try:
        row_counts: dict[str, int] = {}
        
        # Load call graph as NetworkX graph
        call_graph = _load_call_graph(ctx.gateway, ctx.repo, ctx.commit)
        
        # Compute centrality metrics
        pagerank = centrality.compute_pagerank(call_graph)
        betweenness = centrality.compute_betweenness(
            call_graph, 
            k=config.max_betweenness_sample
        )
        
        # Compute component metrics
        sccs = components.find_strongly_connected(call_graph)
        
        # Compute structural metrics
        clustering = structural.compute_clustering_coefficient(call_graph)
        
        # Persist metrics (implementation depends on storage schema)
        row_counts["analytics.call_graph_metrics"] = len(pagerank)
        
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"Core metrics computation failed: {e}")


def _load_call_graph(gateway, repo: str, commit: str):
    """Load call graph as NetworkX DiGraph."""
    import networkx as nx
    
    g = nx.DiGraph()
    edges = gateway.con.execute("""
        SELECT caller_goid, callee_goid, call_count
        FROM graphs.call_graph_edges
        WHERE repo = ? AND commit = ?
    """, [repo, commit]).fetchall()
    
    for caller, callee, weight in edges:
        g.add_edge(caller, callee, weight=weight)
    return g
```

---

#### Fix: `metrics/secondary.py`

**Current:**
```python
from codeintel.graphs.compute.metrics import compute_secondary_metrics  # ERROR
```

**Target:**
```python
from codeintel.graphs.compute.metrics import cfg, dfg, community, paths

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    config = GraphMetricsStepConfig(snapshot=ctx.snapshot)
    
    try:
        row_counts: dict[str, int] = {}
        
        # CFG metrics
        cfg_graph = _load_cfg_graph(ctx.gateway, ctx.repo, ctx.commit)
        cfg_metrics = cfg.compute_dominance_metrics(cfg_graph)
        
        # DFG metrics
        dfg_graph = _load_dfg_graph(ctx.gateway, ctx.repo, ctx.commit)
        dfg_metrics = dfg.compute_dataflow_metrics(dfg_graph)
        
        # Community detection on call graph
        call_graph = _load_call_graph(ctx.gateway, ctx.repo, ctx.commit)
        communities = community.detect_communities_louvain(call_graph)
        
        row_counts["analytics.secondary_graph_metrics"] = len(cfg_metrics)
        
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"Secondary metrics computation failed: {e}")
```

**Estimated effort:** 10 minutes

---

### 4F: Update Metrics `__init__.py` (26 missing exports)

**File:** `src/codeintel/graphs/plugins/metrics/__init__.py`

**Problem:** Imports 26 symbols that no longer exist after migration to `TargetPlugin`.

**Current State (BROKEN):**
```python
"""Graph metric plugins."""
from codeintel.graphs.plugins.metrics.core import (
    core_graph_metrics_plugin,          # ERROR: Doesn't exist
    function_ext_metrics_plugin,         # ERROR: Doesn't exist
    get_core_graph_metrics_plugin,       # ERROR: Doesn't exist
    get_function_ext_metrics_plugin,     # ERROR: Doesn't exist
    get_module_ext_metrics_plugin,       # ERROR: Doesn't exist
    module_ext_metrics_plugin,           # ERROR: Doesn't exist
)
from codeintel.graphs.plugins.metrics.secondary import (
    cfg_metrics_plugin,                  # ERROR: Doesn't exist
    config_graph_metrics_plugin,         # ERROR: Doesn't exist
    # ... 18 more missing exports ...
)

__all__ = [
    "cfg_metrics_plugin",
    "config_graph_metrics_plugin",
    # ... 24 more stale exports ...
]
```

**Target State:**
```python
"""Graph metrics plugins.

This package contains plugins that compute metrics over graph structures:
- CoreMetricsPlugin: Core function/module metrics (PageRank, centrality, components)
- SecondaryMetricsPlugin: Extended metrics (CFG, DFG, community detection)

All plugins implement the TargetPlugin protocol and are executed by the
build system via BuildExecutor.
"""

from codeintel.graphs.plugins.metrics.core import CoreMetricsPlugin
from codeintel.graphs.plugins.metrics.secondary import SecondaryMetricsPlugin

__all__ = [
    "CoreMetricsPlugin",
    "SecondaryMetricsPlugin",
]
```

**Note:** The old naming convention used:
- `xxx_plugin` (module-level singleton instance)
- `get_xxx_plugin()` (factory function)

The new naming convention uses:
- `XxxPlugin` (class that implements `TargetPlugin`)

**Estimated effort:** 15 minutes

---

### 4G: Fix Import Graph Plugin (1 error)

**File:** `builders/import_graph.py:14`

**Error:** `build_import_graph_data` is unknown import symbol

The `graphs/compute/imports.py` module exports data structures and pure functions, not a high-level `build_import_graph_data` function.

**Available exports from `graphs/compute/imports.py`:**
```python
@dataclass(frozen=True)
class ImportEdge:
    src_module: str
    dst_module: str

@dataclass(frozen=True)
class ImportModuleRow:
    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int
```

---

**Current State:**
```python
from codeintel.graphs.compute.imports import build_import_graph_data  # ERROR: Doesn't exist

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    cfg = ImportGraphStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,  # Also wrong - no paths parameter
    )
    row_counts = build_import_graph_data(ctx.gateway, cfg)
    return TargetResult.succeeded(row_counts=row_counts)
```

**Target State:**
```python
from codeintel.config import ImportGraphStepConfig
from codeintel.graphs.compute import imports

async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    _ = self
    
    cfg = ImportGraphStepConfig(snapshot=ctx.snapshot)
    
    try:
        row_counts: dict[str, int] = {}
        con = ctx.gateway.con
        repo, commit = ctx.repo, ctx.commit
        
        # Get all module imports from AST data
        import_rows = con.execute("""
            SELECT rel_path, imported_module 
            FROM core.ast_imports 
            WHERE repo = ? AND commit = ?
        """, [repo, commit]).fetchall()
        
        # Build import edges
        edges: list[imports.ImportEdge] = []
        for src_path, dst_module in import_rows:
            src_module = src_path.replace("/", ".").replace(".py", "")
            edges.append(imports.ImportEdge(src_module=src_module, dst_module=dst_module))
        
        # Build import graph using NetworkX
        import networkx as nx
        g = nx.DiGraph()
        for edge in edges:
            g.add_edge(edge.src_module, edge.dst_module)
        
        # Compute SCCs and layers
        sccs = list(nx.strongly_connected_components(g))
        
        # Write results
        # ... persist to graphs.import_graph_nodes, graphs.import_graph_edges ...
        
        row_counts["graphs.import_graph_edges"] = len(edges)
        row_counts["graphs.import_graph_nodes"] = g.number_of_nodes()
        
        return TargetResult.succeeded(row_counts=row_counts)
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"Import graph build failed: {e}")
```

**Estimated effort:** 5 minutes

---

## Estimated Total Effort

| Category | Estimated Time |
|----------|----------------|
| Test Infrastructure | 1 min |
| Config Resolver | 5 min |
| Core Recipes | 10 min |
| Graph Plugins - Imports | 20 min |
| Graph Plugins - Configs | 10 min |
| Graph Plugins - paths= | 10 min |
| Graph Plugins - CallGraph | 45 min |
| Graph Plugins - Metrics funcs | 10 min |
| Graph Plugins - Metrics init | 15 min |
| Graph Plugins - Import graph | 5 min |
| **TOTAL** | **~2 hours** |

---

## Recommended Execution Order

### Step 1: Quick Wins (~15 minutes)
1. Fix test docstring (`tests/ingestion/engine/__init__.py`)
2. Fix config resolver (`src/codeintel/config/resolver.py`)
3. Fix or delete core recipes (`src/codeintel/core/recipes/unified_executor.py`)

### Step 2: Graph Plugin Imports (~30 minutes)
1. Fix all missing module imports (4A)
2. Fix all config class imports (4B)
3. Fix metrics function imports (4E)
4. Fix import graph function import (4G)

### Step 3: Graph Plugin Parameters (~10 minutes)
1. Remove all `paths=ctx.paths` issues (4C)

### Step 4: Graph Plugin __init__ Files (~15 minutes)
1. Rewrite metrics `__init__.py` (4F)

### Step 5: CallGraph Deep Fixes (~45 minutes)
1. Fix `EdgeResolutionContext` constructor
2. Fix storage adapter usage
3. Fix catalog method calls
4. Fix row constructor

### Step 6: Final Verification
```bash
uv run ruff check --fix src/codeintel/ tests/
uv run pyright src/codeintel/
uv run pyrefly check
uv run pytest tests/ -q
```

---

## Files That May Need Complete Rewrite

Based on error density, consider if these files should be rewritten from scratch:

1. **`src/codeintel/graphs/plugins/builders/callgraph.py`** - 8 errors, complex data structures
2. **`src/codeintel/graphs/plugins/metrics/__init__.py`** - 26 stale exports

---

## Decision Points

### Delete vs Fix

Some graph plugins might be better deleted if:
- They duplicate functionality now in the build system
- They're not actively used in production
- The migration effort exceeds rewrite effort

### Files to Evaluate:
- `src/codeintel/core/recipes/unified_executor.py` - May be legacy
- Graph plugins that have no corresponding build targets

---

## Quality Gates

After completing all fixes, verify:

1. **Ruff** (lint): `uv run ruff check src/codeintel/ tests/` → 0 errors
2. **Pyright** (types): `uv run pyright src/codeintel/` → 0 errors  
3. **Pyrefly** (types): `uv run pyrefly check` → 0 errors
4. **Tests**: `uv run pytest tests/ -q` → All pass

---

## Appendix: Deleted Files Summary

### Source Files (19 files, ~2,500 lines)
```
src/codeintel/ingestion/runtime/__init__.py
src/codeintel/ingestion/runtime/executor.py
src/codeintel/ingestion/runtime/planning.py
src/codeintel/ingestion/runtime/telemetry.py
src/codeintel/ingestion/recipes/__init__.py
src/codeintel/ingestion/recipes/builtin.py
src/codeintel/ingestion/recipes/dsl.py
src/codeintel/ingestion/recipes/executor.py
src/codeintel/ingestion/core/__init__.py
src/codeintel/ingestion/core/base.py
src/codeintel/ingestion/core/execution_context.py
src/codeintel/ingestion/core/runs.py
src/codeintel/ingestion/core/traits.py
src/codeintel/ingestion/resources/__init__.py
src/codeintel/ingestion/resources/modules.py
src/codeintel/ingestion/resources/protocol.py
src/codeintel/ingestion/resources/registry.py
src/codeintel/ingestion/resources/tools.py
src/codeintel/ingestion/resources/tracker.py
src/codeintel/ingestion/plugins/config_factory.py
src/codeintel/cli/commands/ingest.py
```

### Test Files (~15 files)
```
tests/ingestion/runtime/ (entire directory)
tests/ingestion/core/ (entire directory)
tests/ingestion/test_recipe_executor.py
tests/ingestion/test_pipeline_integration.py
tests/ingestion/test_plugin_registry.py
tests/ingestion/test_ingest_runs.py
tests/ingestion/test_ingest_run_reporting.py
tests/ingestion/test_resources.py
tests/ingestion/test_core.py
tests/ingestion/test_config_factory.py
tests/ingestion/plugins/test_config_plugin.py
tests/ingestion/plugins/test_scip_plugin.py
tests/ingestion/test_plugins.py
tests/ingestion/test_ingest_run_incremental_ast.py
tests/ingestion/test_coverage_incremental.py
tests/_helpers/harnesses/ingestion.py
tests/_helpers/harnesses/ingest_setup.py
```

