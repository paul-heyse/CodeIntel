# Remaining Open Items: Build System Enhancement Plan

> **Status**: Ready for Implementation  
> **Author**: AI Assistant  
> **Date**: 2025-12-17  
> **Scope**: Complete remaining phases of Hamilton build consolidation  
> **Estimated Effort**: 5-8 days total

---

## Executive Summary

This document details the remaining open items from the Hamilton build consolidation effort. Four areas require attention:

| Item | Priority | Effort | Status |
|------|----------|--------|--------|
| Phase 7: Registry Unification | High | 2-3 days | Ready |
| Phase 5 Completion: @pipe_input Expansion | Medium | 1-2 days | Ready |
| Phase 9 Completion: Parallel Execution Promotion | Low | 0.5 day | **Already CLI-accessible** |
| Phase 6 Finalization: Context Simplification | Low | Optional | Acceptable as-is |

---

## Table of Contents

1. [Phase 7: Registry Unification](#phase-7-registry-unification)
2. [Phase 5 Completion: @pipe_input Expansion](#phase-5-completion-pipe_input-expansion)
3. [Phase 9 Completion: Parallel Execution Promotion](#phase-9-completion-parallel-execution-promotion)
4. [Phase 6 Finalization: Context Simplification](#phase-6-finalization-context-simplification)
5. [Implementation Roadmap](#implementation-roadmap)

---

## Phase 7: Registry Unification

### Current State

The build system currently maintains **two sources of truth** for target dependencies:

1. **Static OutputTarget declarations** in [`registry.py`](../../src/codeintel/build/registry.py) with hardcoded `dependencies=` tuples
2. **Hamilton DAG** where actual dependencies are derived from function parameters and `@subdag` wiring

The [`introspect.py`](../../src/codeintel/build/hamilton/introspect.py) module provides `derive_target_dependencies()` to extract dependencies from Hamilton, but this is only used at runtime via `target_graph_from_hamilton()`.

### Problem

Static dependencies in `registry.py` can drift from the actual Hamilton DAG:

```python
# registry.py - Static declaration (may become stale)
CALL_GRAPH_TARGET = OutputTarget(
    name="call_graph",
    dependencies=("goids", "scip"),  # Hardcoded
    ...
)

# Hamilton DAG - Actual runtime dependencies
@tag(domain="graphs", target="call_graph", node_type="tool")
def t__call_graph__extract(
    env: BuildEnv,
    t__goids: TargetRunRecord,  # Actual dependency
) -> CallGraphExtractResult:
    ...
```

This causes:
- Incorrect manifest hashing (wrong dependencies → wrong skip decisions)
- Broken closure planning (may miss or include wrong targets)
- Maintenance burden (two places to update)

### Proposed Solution: TargetRegistry

Create a unified `TargetRegistry` that derives dependencies from Hamilton at startup:

```python
# New: src/codeintel/build/target_registry.py

@dataclass
class TargetRegistry:
    """Unified target registry with Hamilton-derived dependencies.
    
    This replaces the static OutputTarget declarations with a registry
    that derives dependencies from the actual Hamilton DAG.
    """
    
    _targets: dict[str, OutputTarget]
    _dependencies: dict[str, tuple[str, ...]]
    
    @classmethod
    def from_hamilton(cls, runtime: HamiltonRuntime) -> TargetRegistry:
        """Build registry from Hamilton DAG introspection."""
        # Use existing derive_target_dependencies()
        deps = derive_target_dependencies(runtime)
        
        # Merge with OutputTarget metadata from registry.py
        targets = {}
        for target in ALL_TARGETS:
            # Replace static dependencies with derived ones
            derived_deps = deps.get(target.name, target.dependencies)
            targets[target.name] = _clone_target_with_dependencies(
                target, deps=derived_deps
            )
        
        return cls(_targets=targets, _dependencies=deps)
    
    def get(self, name: str) -> OutputTarget | None:
        """Get target by name."""
        return self._targets.get(name)
    
    def dependencies(self, name: str) -> tuple[str, ...]:
        """Get direct dependencies for a target."""
        return self._dependencies.get(name, ())
    
    def closure(self, names: Iterable[str]) -> list[str]:
        """Compute transitive closure of targets in topological order."""
        ...
```

### Implementation Steps

#### Step 7.1: Create TargetRegistry Class (1 day)

**File**: `src/codeintel/build/target_registry.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.introspect import derive_target_dependencies
from codeintel.build.targets import OutputTarget, TargetGraph

if TYPE_CHECKING:
    from collections.abc import Iterable
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


@dataclass
class TargetRegistry:
    """Unified target registry with Hamilton-derived dependencies."""
    
    _graph: TargetGraph
    _derived_deps: dict[str, tuple[str, ...]]
    
    @classmethod
    def from_hamilton(
        cls,
        runtime: HamiltonRuntime,
        *,
        base_graph: TargetGraph | None = None,
    ) -> TargetRegistry:
        """Build registry from Hamilton DAG.
        
        Parameters
        ----------
        runtime
            Hamilton runtime with configured driver.
        base_graph
            Optional base TargetGraph for OutputTarget metadata.
            Defaults to runtime.graph.
        
        Returns
        -------
        TargetRegistry
            Registry with Hamilton-derived dependencies.
        """
        derived = derive_target_dependencies(runtime)
        graph = target_graph_from_hamilton(runtime, base_graph=base_graph)
        return cls(_graph=graph, _derived_deps=derived)
    
    def get(self, name: str) -> OutputTarget | None:
        """Get target by name."""
        return self._graph.get(name)
    
    def dependencies(self, name: str) -> tuple[str, ...]:
        """Get direct dependencies for a target."""
        return self._derived_deps.get(name, ())
    
    def closure(self, names: Iterable[str]) -> list[str]:
        """Compute transitive closure in topological order.
        
        Parameters
        ----------
        names
            Target names to compute closure for.
        
        Returns
        -------
        list[str]
            Targets in dependency order (dependencies first).
        """
        visited: set[str] = set()
        result: list[str] = []
        
        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies(name):
                visit(dep)
            result.append(name)
        
        for name in names:
            visit(name)
        
        return result
    
    @property
    def all_targets(self) -> list[OutputTarget]:
        """Return all registered targets."""
        return self._graph.all_targets
```

#### Step 7.2: Integrate with Build Planner (1 day)

Update `HamiltonBuildExecutor` to use `TargetRegistry`:

**File**: `src/codeintel/build/hamilton/executor.py`

```python
# Before:
class HamiltonBuildExecutor:
    def __init__(self, ...):
        self._graph = get_target_graph()  # Static graph
        
# After:
class HamiltonBuildExecutor:
    def __init__(self, ...):
        runtime = build_hamilton_runtime(...)
        self._registry = TargetRegistry.from_hamilton(runtime)
        self._graph = self._registry._graph
```

#### Step 7.3: Update Manifest Hashing (0.5 day)

Ensure `compute_input_hash()` uses derived dependencies:

**File**: `src/codeintel/build/hashing.py`

```python
def compute_input_hash(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None,
    manifests: ManifestIndex,
    # NEW: Use registry for accurate dependencies
    registry: TargetRegistry | None = None,
) -> str:
    # Use registry.dependencies() if available
    deps = (
        registry.dependencies(target.name) 
        if registry 
        else target.dependencies
    )
    ...
```

#### Step 7.4: Deprecate Static Dependencies (0.5 day)

Add deprecation warnings to static `dependencies` in `OutputTarget`:

```python
# registry.py
CALL_GRAPH_TARGET = OutputTarget(
    name="call_graph",
    dependencies=(),  # Empty - derived from Hamilton
    ...
)
```

### Validation

1. **Dependency parity test**: Verify derived dependencies match expected
2. **Manifest hash stability**: Ensure hashes are deterministic
3. **Closure correctness**: Test transitive closure computation

---

## Phase 5 Completion: @pipe_input Expansion

### Current State

`@pipe_input` is used in **3 targets** for DAG-visible multi-step transformations:

| Target | File | Steps |
|--------|------|-------|
| `risk_factors` | `risk_factors.py` | 3 steps (join, score, finalize) |
| `hotspots` | `hotspots.py` | 5 steps (filter, aggregate, join, score, select) |
| `subsystems` | `subsystem_targets.py` | 4 steps (filter, group, assign, build) |

### Candidates for @pipe_input Expansion

Analysis of complex targets with multi-step Ibis transformations:

| Target | File | Current Pattern | Benefit |
|--------|------|-----------------|---------|
| `call_graph` | `call_graph.py` | Sequential helper calls | Medium - Multiple phases (scan, collect, persist) |
| `import_graph` | `import_graph.py` | Sequential helper calls | Low - Simple linear flow |
| `function_contracts` | `function_detail_targets.py` | Complex AST analysis | Medium - Could split parsing/extraction/validation |
| `external_deps` | `dependency_targets.py` | Call graph traversal | Medium - Could split resolution/classification/aggregation |
| `config_data_flow` | `config_graph_targets.py` | Graph traversal | Low - Already multi-table pattern |

### Recommended Candidates

Based on complexity vs. benefit analysis, prioritize:

1. **`external_deps`** - Best candidate
2. **`function_contracts`** - Good candidate
3. **`call_graph`** - Optional (already well-structured)

### Implementation Pattern

For `external_deps` in [`dependency_targets.py`](../../src/codeintel/build/hamilton/native/analytics/dependency_targets.py):

```python
from hamilton.function_modifiers import pipe_input, step, source

# Step functions (pure transformations)
def _deps_resolve_imports(expr: ir.Table, env: BuildEnv) -> ir.Table:
    """Resolve import statements to package references."""
    ...

def _deps_classify_packages(expr: ir.Table) -> ir.Table:
    """Classify packages as stdlib, third-party, or internal."""
    ...

def _deps_aggregate_usage(expr: ir.Table) -> ir.Table:
    """Aggregate usage counts per package."""
    ...

# Main compute with @pipe_input
@SaveToDecorator(...)
@pipe_input(
    step(_deps_resolve_imports, env=source("env")),
    step(_deps_classify_packages),
    step(_deps_aggregate_usage),
    namespace=None,
    on_input="q__graph__import_graph_edges",
)
@tag(domain="analytics", target="external_deps", node_type="compute")
def t__external_deps__compute(
    q__graph__import_graph_edges: ir.Table,
) -> ir.Table:
    """Compute external dependencies with DAG-visible transformations."""
    return q__graph__import_graph_edges
```

### Benefits of @pipe_input

1. **DAG visibility**: Each step appears in Hamilton graph
2. **Debuggability**: Can inspect intermediate results
3. **Testability**: Steps are unit-testable in isolation
4. **Traceability**: Clear data lineage through transformations

### Effort Estimate

- Per target: 2-4 hours
- Testing: 1-2 hours per target
- **Total**: 1-2 days for 2-3 candidates

---

## Phase 9 Completion: Parallel Execution Promotion

### Current State

**Good news**: Parallel execution is **already CLI-accessible**!

The [`build.py`](../../src/codeintel/cli/commands/build.py) CLI already supports:

```bash
# Use threadpool backend with 4 workers
codeintel build run --parallel-backend=threadpool --max-workers=4 function_metrics

# Auto-select backend
codeintel build run --parallel-backend=auto --max-workers=8 --all
```

### Implementation Details

**CLI Flags** (already implemented):
- `--parallel-backend`: `sequential` (default), `threadpool`, `auto`
- `--max-workers` / `--workers`: Number of parallel workers

**Backend** ([`parallel.py`](../../src/codeintel/build/hamilton/adapters/parallel.py)):
- `ThreadPoolAdapter` with global write lock for DuckDB safety
- Per-thread gateway management for isolation
- Future support stubs for Ray/Dask

### Remaining Work

Only documentation and testing improvements needed:

#### 9.1 Add CLI Help Documentation

Update `--help` output with usage examples:

```python
# In build.py
parallel_backend: Annotated[
    str,
    Parameter(
        name=["--parallel-backend"],
        help="""Parallel execution backend.

Options:
  sequential  - Single-threaded (default, safest)
  threadpool  - Multi-threaded with write lock
  auto        - Auto-select best backend

Example: --parallel-backend=threadpool --max-workers=4""",
    ),
] = "sequential"
```

#### 9.2 Add Integration Tests

```python
# tests/cli/test_build_parallel.py
def test_build_run_with_threadpool_backend():
    """Verify threadpool backend executes without errors."""
    result = cli_runner.invoke([
        "build", "run",
        "--parallel-backend=threadpool",
        "--max-workers=2",
        "function_metrics",
    ])
    assert result.exit_code == 0

def test_max_workers_implies_threadpool():
    """Verify --max-workers implies threadpool backend."""
    # This is already implemented in _extract_build_run_params
    ...
```

#### 9.3 Performance Benchmarks (Optional)

Add benchmarks to quantify parallel speedup:

```python
# benchmarks/test_parallel_speedup.py
@pytest.mark.benchmark
def test_parallel_vs_sequential_speedup(benchmark):
    """Measure parallel execution speedup."""
    ...
```

### Effort Estimate

- Documentation: 1-2 hours
- Integration tests: 2-3 hours
- **Total**: 0.5 day

---

## Phase 6 Finalization: Context Simplification

### Current State

The context hierarchy is:

```
BuildContext (context_base.py)
├── snapshot: SnapshotRef
├── gateway: StorageGateway
├── paths: BuildPaths
├── providers: Providers
└── options: MaterializationOptions

TargetExecutionContext (context.py)
├── build: BuildContext  # Composition
├── target: OutputTarget
├── contract: OutputContract
├── resources: ContextResources
└── parameters: TargetParameters
```

### Assessment

**Current state is acceptable** because:

1. `BuildContext` and `TargetExecutionContext` have clear responsibilities
2. `TargetExecutionContext` composes `BuildContext` rather than inheriting
3. No code duplication or confusion

### Optional Improvement

If desired, merge `ContextResources` into `TargetExecutionContext`:

```python
# Before:
@dataclass
class TargetExecutionContext:
    build: BuildContext
    resources: ContextResources
    ...

# After:
@dataclass
class TargetExecutionContext:
    build: BuildContext
    # Resources inlined
    tracker: ChangeTracker | None = None
    modules: list[ModuleRecord] | None = None
    git_history: GitHistoryProvider | None = None
    ...
```

### Recommendation

**Leave as-is** - The current design is clean and maintainable. The separation of `ContextResources` provides clarity about what resources are available during execution.

---

## Implementation Roadmap

### Priority Order

```
Week 1: Phase 7 (Registry Unification) - HIGH PRIORITY
├── Day 1: Create TargetRegistry class
├── Day 2: Integrate with build executor
└── Day 3: Update hashing, add tests

Week 2: Phase 5 (@pipe_input Expansion) - MEDIUM PRIORITY
├── Day 1: Convert external_deps
└── Day 2: Convert function_contracts (optional)

Week 2 (cont): Phase 9 (Documentation) - LOW PRIORITY
└── Day 3: Add CLI docs and integration tests
```

### Success Criteria

| Phase | Success Criteria |
|-------|------------------|
| 7 | Dependencies derived from Hamilton; manifest hashes stable |
| 5 | 2+ additional targets use @pipe_input; steps visible in DAG |
| 9 | CLI help documents parallel flags; integration tests pass |

### Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 7 | Manifest hash changes break incremental builds | Add migration path; keep static deps as fallback |
| 5 | @pipe_input increases DAG complexity | Only apply to genuinely multi-step targets |
| 9 | Thread safety issues | Already handled by write lock; add stress tests |

---

## Appendix: File Reference

### Phase 7 Files

| File | Purpose |
|------|---------|
| `src/codeintel/build/registry.py` | Static OutputTarget declarations |
| `src/codeintel/build/hamilton/introspect.py` | Hamilton DAG introspection |
| `src/codeintel/build/hashing.py` | Input hash computation |
| `src/codeintel/build/target_registry.py` | NEW: Unified registry |

### Phase 5 Files

| File | Purpose |
|------|---------|
| `src/codeintel/build/hamilton/native/analytics/dependency_targets.py` | external_deps candidate |
| `src/codeintel/build/hamilton/native/analytics/function_detail_targets.py` | function_contracts candidate |

### Phase 9 Files

| File | Purpose |
|------|---------|
| `src/codeintel/build/hamilton/adapters/parallel.py` | ThreadPoolAdapter |
| `src/codeintel/cli/commands/build.py` | CLI flags |
| `src/codeintel/cli/handlers/build.py` | CLI handler |

---

*This document is the planning reference for remaining Hamilton consolidation work. Last updated: 2025-12-17*

