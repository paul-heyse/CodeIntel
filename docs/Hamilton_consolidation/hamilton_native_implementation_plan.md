# Hamilton Native Implementation Plan

> **Purpose**: Comprehensive, detailed implementation plan for migrating to a 100% native Hamilton architecture, eliminating the plugin abstraction layer entirely.

**Status**: Design Document  
**Created**: 2025-12-15  
**Target Completion**: TBD

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Vision](#architectural-vision)
3. [Current State Analysis](#current-state-analysis)
4. [Target State Architecture](#target-state-architecture)
5. [Implementation Phases](#implementation-phases)
6. [Detailed PR Breakdown](#detailed-pr-breakdown)
7. [Migration Recipes](#migration-recipes)
8. [Testing Strategy](#testing-strategy)
9. [Risk Assessment & Mitigation](#risk-assessment--mitigation)
10. [Success Criteria](#success-criteria)
11. [Appendix: File-by-File Disposition](#appendix-file-by-file-disposition)

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

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | 1-2 weeks | Foundation (env consolidation, hooks) |
| Phase 2 | 2-3 weeks | Ingestion domain migration |
| Phase 3 | 2-3 weeks | Graphs domain migration |
| Phase 4 | 2-3 weeks | Analytics domain migration |
| Phase 5 | 1 week | Export domain migration |
| Phase 6 | 1 week | Cleanup and deletion |
| **Total** | **9-13 weeks** | |

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

### Phase 1: Foundation (PRs 100-104)

**Goal**: Establish the target architecture without breaking existing functionality.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-100 | Consolidate hooks into `hamilton/hooks/` | None | Low |
| PR-101 | Create unified `NativeTargetExecutor` | PR-100 | Low |
| PR-102 | Create module loader for native modules | PR-101 | Low |
| PR-103 | Add `--native-only` flag to driver | PR-102 | Low |
| PR-104 | Create migration test harness | PR-103 | Low |

### Phase 2: Ingestion Domain (PRs 105-116)

**Goal**: Migrate all ingestion plugins to native Hamilton.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-105 | Native `t__modules` | PR-104 | High |
| PR-106 | Native `t__scip` | PR-105 | High |
| PR-107 | Native `t__ast` | PR-105 | Medium |
| PR-108 | Native `t__cst` | PR-105 | Medium |
| PR-109 | Native `t__typing` | PR-106 | Medium |
| PR-110 | Native `t__tests` | PR-105 | Medium |
| PR-111 | Native `t__docstrings` | PR-107 | Low |
| PR-112 | Native `t__coverage` | PR-110 | Medium |
| PR-113 | Native `t__config` | PR-105 | Low |
| PR-114 | Verify ingestion domain parity | PR-105-113 | Low |
| PR-115 | Delete ingestion plugins | PR-114 | Low |
| PR-116 | Delete ingestion registrations | PR-115 | Low |

### Phase 3: Graphs Domain (PRs 117-126)

**Goal**: Migrate all graph plugins to native Hamilton.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-117 | Native `t__goids` | PR-106 | Medium |
| PR-118 | Native `t__call_graph` | PR-117 | Medium |
| PR-119 | Native `t__import_graph` | PR-105 | Medium |
| PR-120 | Native `t__symbol_uses` | PR-106 | Medium |
| PR-121 | Native `t__cfg_dfg` | PR-107, PR-117 | High |
| PR-122 | Native `t__call_graph_views` | PR-118 | Low |
| PR-123 | Native graph metrics targets | PR-118 | Low |
| PR-124 | Verify graphs domain parity | PR-117-123 | Low |
| PR-125 | Delete graphs plugins | PR-124 | Low |
| PR-126 | Delete graphs registrations | PR-125 | Low |

### Phase 4: Analytics Domain (PRs 127-145)

**Goal**: Migrate all analytics plugins to native Hamilton.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-127 | Native `t__function_metrics` | PR-117 | Medium |
| PR-128 | Native `t__function_types` | PR-117 | Low |
| PR-129 | Native `t__ast_features` | PR-107, PR-117 | Medium |
| PR-130 | Native `t__risk_factors` | PR-127, PR-118 | Low (exists) |
| PR-131 | Native `t__hotspots` | PR-130 | Low |
| PR-132 | Native `t__profiles` | Multiple | High |
| PR-133 | Native `t__test_profile` | PR-110, PR-112 | Medium |
| PR-134 | Native `t__coverage_edges` | PR-112 | Medium |
| PR-135 | Native `t__subsystems` | Multiple | High |
| PR-136 | Native `t__semantic_roles` | Multiple | Medium |
| PR-137 | Native `t__symbol_graph_metrics` | PR-120 | Medium |
| PR-138 | Native `t__function_history` | PR-127 | Medium |
| PR-139 | Native `t__history_timeseries` | PR-138 | Low |
| PR-140 | Native `t__data_models` | PR-106 | Medium |
| PR-141 | Native `t__entrypoints` | PR-118 | Medium |
| PR-142 | Native `t__dependencies` | PR-118, PR-119 | Medium |
| PR-143 | Verify analytics domain parity | PR-127-142 | Low |
| PR-144 | Delete analytics plugins | PR-143 | Low |
| PR-145 | Delete analytics registrations | PR-144 | Low |

### Phase 5: Export Domain (PRs 146-150)

**Goal**: Migrate export plugins to native Hamilton.

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-146 | Native `t__export_jsonl` | Any | Low (exists) |
| PR-147 | Native `t__export_parquet` | Any | Low (exists) |
| PR-148 | Verify export domain parity | PR-146-147 | Low |
| PR-149 | Delete export plugins | PR-148 | Low |
| PR-150 | Delete export registrations | PR-149 | Low |

### Phase 6: Cleanup (PRs 151-160)

**Goal**: Delete all legacy infrastructure.

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

---

## Detailed PR Breakdown

### PR-100: Consolidate Hooks

**Files Changed:**
- Create `hamilton/hooks/__init__.py`
- Move `manifest_hook.py` → `hamilton/hooks/manifest_hook.py`
- Move `telemetry_hook.py` → `hamilton/hooks/telemetry_hook.py`
- Create `hamilton/hooks/contract_hook.py` (from enforcement)

**Acceptance Criteria:**
- [ ] All hooks in single directory
- [ ] Imports updated across codebase
- [ ] Tests pass

---

### PR-101: Unified NativeTargetExecutor

**Files Changed:**
- Simplify `hamilton/native/executor.py`
- Remove dual-path logic
- Standardize interface

**New Interface:**
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
        """Execute and return record with row counts."""
```

**Acceptance Criteria:**
- [ ] Single executor class
- [ ] Handles skip check, timing, record creation
- [ ] All existing native targets use it

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
| Phase 1 | Hooks consolidated, executor unified, test harness ready |
| Phase 2 | All ingestion plugins migrated, parity tests pass |
| Phase 3 | All graphs plugins migrated, parity tests pass |
| Phase 4 | All analytics plugins migrated, parity tests pass |
| Phase 5 | All export plugins migrated, parity tests pass |
| Phase 6 | Legacy infrastructure deleted, all tests pass |

### Final Success Criteria

- [ ] **Zero plugin classes** remain in codebase
- [ ] **Single context type** (BuildEnv) in use
- [ ] **All targets** are Hamilton native modules
- [ ] **Skip logic** works uniformly via ManifestHook
- [ ] **All tests pass** (no new xfails)
- [ ] **Performance** is equal or better
- [ ] **Lines of code** reduced by ~3,000+
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

**Version**: 1.0  
**Status**: Design Complete  
**Author**: CodeIntel Build Team  
**Last Updated**: 2025-12-15

### Changelog

- **v1.0** (2025-12-15): Initial comprehensive implementation plan

