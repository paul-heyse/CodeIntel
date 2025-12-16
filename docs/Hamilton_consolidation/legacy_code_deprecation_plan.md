# Legacy Code Deprecation Plan (Post-Phase 5)

> **Purpose**: This document identifies and details all legacy code functionality that becomes redundant once Phase 5 (PR-74 through PR-80) is complete. It provides clear rationale for each deprecation and a recommended deletion order to minimize disruption.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 5 Context](#phase-5-context)
3. [Legacy Components for Deprecation](#legacy-components-for-deprecation)
   - [1. Legacy Dataset Contract Shims](#1-legacy-dataset-contract-shims)
   - [2. Legacy TargetGraph as Dependency Source](#2-legacy-targetgraph-as-dependency-source)
   - [3. Legacy Target Registry Constants](#3-legacy-target-registry-constants)
   - [4. Legacy Plan Generator & Resolver](#4-legacy-plan-generator--resolver)
   - [5. Legacy State Validation Types](#5-legacy-state-validation-types)
   - [6. Legacy graph_source="targetgraph" Code Paths](#6-legacy-graph_sourcetargetgraph-code-paths)
   - [7. Contract Provider Legacy Fallbacks](#7-contract-provider-legacy-fallbacks)
4. [Files Summary](#files-summary)
5. [Recommended Deletion Order](#recommended-deletion-order)
6. [Migration Guidance](#migration-guidance)
7. [Risk Assessment](#risk-assessment)
8. [Verification Checklist](#verification-checklist)
9. [Appendix: Related Phase 5 PRs](#appendix-related-phase-5-prs)
10. [Appendix B: Build Directory File Assessment](#appendix-b-build-directory-file-assessment)

---

## Executive Summary

Phase 5 of the Hamilton consolidation establishes a **DAG-first architecture** where:

- **Hamilton is the single source of truth** for target dependencies, outputs, and artifacts
- **BuildSpec** is the deterministic, stable compiled contract of the Hamilton DAG
- **Validator gates** ensure correctness and detect drift automatically

This architectural shift makes several legacy components redundant:

| Category | Estimated Lines | Risk Level |
|----------|----------------|------------|
| Dataset contract shims | ~200 lines | Low |
| TargetGraph dependency source | ~280 lines | Medium |
| Static target registry | ~500 lines | Medium |
| ~~Legacy plan/resolver~~ | ~~~660 lines~~ | ✅ **DELETED** |
| Legacy state adapter types | ~200 lines | Low |
| graph_source fallbacks | ~50 lines | Low |
| Contract provider fallbacks | ~100 lines | Low |
| ~~contracts_validation (dead)~~ | ~~~82 lines~~ | ✅ **DELETED** |
| ~~readiness.py~~ | ~~~400 lines~~ | ✅ **DELETED** |
| **Total Remaining** | **~1,330 lines** | |
| ~~Already Deleted~~ | ~~1,148 lines~~ | ✅ |

**Key Finding**: A detailed code analysis revealed that `plan.py`, `resolver.py`, `contracts_validation.py`, and `readiness.py` were legacy/dead code.

> ✅ **COMPLETED (2025-12-15)**: These files and their associated tests have been deleted. See [Appendix B](#appendix-b-build-directory-file-assessment) for details.

---

## Phase 5 Context

Phase 5 introduces the following changes that enable legacy deprecation:

| PR | Change | What It Enables |
|----|--------|-----------------|
| **PR-74** | Auto mode generates `d__/q__/a__` helpers for native targets | Native targets can participate in composition; no special-casing needed |
| **PR-75** | BuildSpec primitives + deterministic JSON + hashing | Stable, serializable contract for CI gates and serving |
| **PR-76** | BuildSpec compiler from Hamilton DAG | Dependencies/outputs derived from actual graph, not parallel registry |
| **PR-77** | BuildSpec CLI (`build spec compile`) | Human/CI interface for inspecting compiled contracts |
| **PR-78** | Hamilton graph validator gate | Automatic enforcement of invariants; catches drift immediately |
| **PR-79** | Default `graph_source="hamilton"` | TargetGraph becomes legacy fallback mode |
| **PR-80** | Batch schema inference | Fast, deterministic schema compilation |

**Key Principle**: Once the Hamilton DAG is validated and BuildSpec is compiled from it, any parallel data structure (TargetGraph, static registries, manual mappings) becomes:
1. **Redundant** — duplicates information already in Hamilton
2. **Drift-prone** — can diverge from the actual execution graph
3. **Maintenance burden** — requires manual updates when targets change

---

## Legacy Components for Deprecation

### 1. Legacy Dataset Contract Shims

**Location**: `src/codeintel/config/datasets/contracts.py`

**What It Is**:

Manually maintained dictionaries mapping dataset names to JSON schema IDs and default export filenames:

```python
# ~14 manually maintained entries
_JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    "call_graph_edges": "call_graph_edges",
    # ... etc
}

# ~70+ manually maintained entries each
DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    # ... 70+ entries
}

DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    "core.goids": "goids.parquet",
    # ... 70+ entries
}
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Manual maintenance** | Every new table requires adding 2-3 dictionary entries |
| **Inconsistent naming** | Some use `table_key`, others use `name` portion only |
| **Drift risk** | Entries can become stale when tables are renamed/removed |
| **No validation** | Nothing checks if entries match actual tables |

**Replacement Strategy**:

Derive filenames deterministically from `table_key`:

```python
def default_jsonl_filename(table_key: str) -> str:
    """Derive JSONL filename from table_key."""
    _, name = table_key.rsplit(".", maxsplit=1)
    return f"{name}.jsonl"

def default_parquet_filename(table_key: str) -> str:
    """Derive Parquet filename from table_key."""
    _, name = table_key.rsplit(".", maxsplit=1)
    return f"{name}.parquet"
```

For the rare cases needing custom filenames, use `OutputContract` metadata tags.

**Lines to Remove**: ~160 lines (the three dictionaries)

---

### 2. Legacy TargetGraph as Dependency Source

**Location**: `src/codeintel/build/targets.py`

**What It Is**:

The `TargetGraph` class that maintains its own dependency graph parallel to Hamilton:

```python
@dataclass
class TargetGraph:
    """Complete dependency graph of all output targets."""
    
    _targets: dict[str, OutputTarget] = field(default_factory=dict)
    _dependents: dict[str, set[str]] = field(default_factory=dict)

    def register(self, target: OutputTarget) -> None:
        """Register a target in the graph."""
        # Manually track dependencies
        for dep in target.dependencies:
            self._dependents[dep].add(target.name)

    def transitive_deps(self, name: str) -> frozenset[str]:
        """Return all transitive dependencies."""
        # Computed from manually registered deps

    def topological_order(self, names: Iterable[str]) -> tuple[str, ...]:
        """Sort targets in dependency order."""
        # Uses Kahn's algorithm on the parallel graph
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Parallel graph** | Two dependency graphs must be kept in sync (TargetGraph + Hamilton) |
| **Manual registration** | Every target must be explicitly registered |
| **Drift risk** | TargetGraph deps can diverge from actual Hamilton DAG |
| **Validation bypass** | TargetGraph doesn't enforce Hamilton's node invariants |

**Replacement**:

Use `target_graph_from_hamilton(runtime)` which derives dependencies from the actual Hamilton FunctionGraph:

```python
def target_graph_from_hamilton(
    runtime: HamiltonRuntime,
    *,
    base_graph: TargetGraph | None = None,
    strict: bool = False,
) -> TargetGraph:
    """Build a TargetGraph whose dependency edges are derived from Hamilton."""
    derived_deps = derive_target_dependencies(runtime)
    # Dependencies come from Hamilton, not manual registration
```

**Migration Path**:

1. Phase 5 flips default to `graph_source="hamilton"`
2. TargetGraph becomes a thin wrapper over Hamilton-derived data
3. Eventually remove manual dependency tracking entirely

**Lines Affected**: ~280 lines (most of `TargetGraph` methods)

---

### 3. Legacy Target Registry Constants

**Location**: `src/codeintel/build/registry.py`

**What It Is**:

~45 statically defined `OutputTarget` constants:

```python
MODULES_TARGET = OutputTarget(
    name="modules",
    module="ingestion",
    plugin="repo_scan",
    contract=OutputContract(tables=(...)),
    dependencies=(),
    description="Repository module and file index from scanning.",
)

AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin="ast_extract",
    contract=OutputContract(tables=(...)),
    dependencies=("modules",),  # ← Manual dependency declaration
    description="Python AST extraction and metrics.",
)

# ... 40+ more target definitions

ALL_TARGETS: tuple[OutputTarget, ...] = (
    MODULES_TARGET,
    AST_TARGET,
    CST_TARGET,
    # ... all targets
)
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Duplication** | Target metadata exists in both registry.py AND Hamilton modules |
| **Dependencies declared twice** | Once in `OutputTarget.dependencies`, once in Hamilton node deps |
| **Plugin coupling** | Ties target definition to plugin name strings |
| **Static tuple** | `ALL_TARGETS` must be updated manually for each new target |

**Replacement Strategy**:

With BuildSpec as the compiled contract, target metadata flows:

```
Hamilton DAG (native modules + generated wrappers)
    ↓ compile
BuildSpec (deterministic JSON)
    ↓ serve
Serving layer / MCP inventory
```

The static registry becomes unnecessary because:
- Target names come from Hamilton `t__<target>` nodes
- Dependencies come from Hamilton node dependencies
- Outputs come from Hamilton `d__<table_key>` nodes
- Artifacts come from Hamilton `a__<artifact>` nodes

**Lines to Remove**: ~500 lines (all static target definitions)

---

### 4. Legacy Plan Generator & Resolver

**Location**: `src/codeintel/build/plan.py`, `src/codeintel/build/resolver.py`

**What It Is**:

Hamilton-agnostic planning infrastructure:

```python
# plan.py
class PlanGenerator:
    """Generate a BuildPlan from a ResolutionResult and target graph."""
    
    def __init__(self, graph: TargetGraph) -> None:
        self._graph = graph

    def generate(self, resolution: ResolutionResult) -> BuildPlan:
        """Generate a plan from a resolver output."""
        steps = self._build_steps(resolution.to_compute, reasons=resolution.reasons)
        # Uses TargetGraph for metadata lookup

# resolver.py
class BuildResolver:
    """Resolve the minimal computation set for build goals."""

    def __init__(self, graph: TargetGraph, state: DatabaseState) -> None:
        self._graph = graph
        self._state = state

    def resolve(
        self,
        goals: Sequence[str],
        *,
        force_recompute: Sequence[str] | None = None,
    ) -> ResolutionResult:
        """Resolve goals into to_compute/to_skip/blocked sets."""
        closure = self._graph.topological_order(goals)
        # Uses TargetGraph for dependency traversal
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **TargetGraph dependency** | Tied to the parallel graph, not Hamilton |
| **Separate state model** | Uses `DatabaseState` wrapper types |
| **Duplicate logic** | Plan computation duplicated in Hamilton planner |

**Replacement**:

The Hamilton planner (`codeintel.build.hamilton.planner`) provides:

```python
def compute_plan(
    *,
    env: BuildEnv,
    graph: TargetGraph | None = None,
    requested: tuple[str, ...],
    mode: HamiltonNodeMode = "generated",
    graph_source: GraphSource = "hamilton",  # ← Uses Hamilton deps
) -> HamiltonBuildPlan:
    """Compute build plan for requested targets."""
    if graph_source == "hamilton":
        runtime = build_driver(mode=mode)
        graph = target_graph_from_hamilton(runtime, base_graph=graph)
    # Plan derived from Hamilton DAG
```

**Migration Path**:

1. All callers of `PlanGenerator`/`BuildResolver` switch to `compute_plan()`
2. Legacy modules moved to `_legacy/` or archived
3. Eventually deleted entirely

**Lines to Remove**: ~450 lines total

---

### 5. Legacy State Validation Types

**Location**: `src/codeintel/build/state.py`

**What It Is**:

Adapter types that convert between legacy and unified state representations:

```python
def _unified_to_legacy_status(status: str) -> TargetStatus:
    """Convert unified status to legacy status."""
    status_map: dict[str, TargetStatus] = {
        "current": "computed",
        "stale": "stale",
        "missing": "missing",
        "blocked": "blocked",
    }
    return status_map.get(status, "missing")

@dataclass(frozen=True)
class TargetState:
    """Current state of a single build target."""
    
    @classmethod
    def from_unified(cls, unified: UnifiedTargetState) -> TargetState:
        """Create legacy TargetState from unified TargetState."""
        return cls(
            name=unified.name,
            status=_unified_to_legacy_status(unified.status),
            # ... adaptation logic
        )

@dataclass(frozen=True)
class DatabaseState:
    """Aggregate state of all targets for a repo/commit snapshot."""
    
    @classmethod
    def from_unified(cls, unified: UnifiedBuildState) -> DatabaseState:
        """Create legacy DatabaseState from unified BuildState."""
        legacy_targets = {
            name: TargetState.from_unified(state)
            for name, state in unified.targets.items()
        }
        return cls(repo=unified.repo, commit=unified.commit, targets=legacy_targets)
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Adapter overhead** | Extra conversion step for every state lookup |
| **Type confusion** | Two state type hierarchies to understand |
| **Maintenance burden** | Adapters must be updated when either side changes |

**Replacement**:

Use unified types directly from `codeintel.build.state_types`:

```python
from codeintel.build.state_types import BuildState, TargetState

# Direct usage, no adaptation needed
state = computer.compute_all()  # Returns unified BuildState
```

**Lines to Remove**: ~200 lines (adapter types and conversion functions)

---

### 6. Legacy `graph_source="targetgraph"` Code Paths

**Location**: `src/codeintel/build/hamilton/introspect.py`, CLI commands

**What It Is**:

Support for the `"targetgraph"` option in the `GraphSource` type:

```python
GraphSource = Literal["targetgraph", "hamilton"]

def parse_graph_source(value: str) -> GraphSource:
    """Parse and validate a GraphSource value."""
    if value == "targetgraph":
        return "targetgraph"
    if value == "hamilton":
        return "hamilton"
    msg = f"Unknown graph source: {value}"
    raise ValueError(msg)
```

And CLI parameter defaults:

```python
graph_source: Annotated[
    GraphSource,
    Parameter(
        name=["--graph-source"],
        help="Dependency graph source: hamilton (default) or targetgraph.",
        show_choices=True,
        validator=_graph_source_validator,
    ),
] = "hamilton"  # PR-79 flips this default
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Code complexity** | Every graph_source consumer has conditional logic |
| **Testing burden** | Both code paths must be tested |
| **Drift detection** | When targetgraph diverges, it's not caught |

**Replacement**:

After PR-79:
1. `"hamilton"` becomes the only supported value
2. Remove `"targetgraph"` from `GraphSource` literal
3. Remove conditional branches in planner/CLI handlers

**Lines to Remove**: ~50 lines (option handling, conditional branches)

---

### 7. Contract Provider Legacy Fallbacks

**Location**: `src/codeintel/build/schemas/contract_provider.py`

**What It Is**:

Functions that fall back to the legacy contract dictionaries:

```python
def _get_json_schema_id(table_key: str) -> str | None:
    """Get the JSON schema ID for a table key."""
    contracts_mod = _contracts_module()
    _, name = table_key.split(".", maxsplit=1)
    json_schema_map = getattr(contracts_mod, "_JSON_SCHEMA_BY_DATASET_NAME", {})
    return json_schema_map.get(name)

def _get_jsonl_filename(table_key: str) -> str | None:
    """Get the default JSONL export filename for a table key."""
    contracts_mod = _contracts_module()
    jsonl_filenames = getattr(contracts_mod, "DEFAULT_JSONL_FILENAMES", {})
    return jsonl_filenames.get(table_key)

def _get_parquet_filename(table_key: str) -> str | None:
    """Get the default Parquet export filename for a table key."""
    contracts_mod = _contracts_module()
    parquet_filenames = getattr(contracts_mod, "DEFAULT_PARQUET_FILENAMES", {})
    return parquet_filenames.get(table_key)
```

Also the lazy module loaders:

```python
_contracts_provider: LazyProvider[ModuleType] = LazyProvider(
    lambda: _load_module("codeintel.config.datasets.contracts"),
    name="contracts_module",
)
```

**Why Deprecate**:

| Problem | Impact |
|---------|--------|
| **Circular complexity** | Lazy loading to break import cycles |
| **Hidden dependencies** | Runtime access to legacy module |
| **Fallback masking** | Bugs hidden when fallback provides value |

**Replacement**:

1. Derive filenames deterministically from `table_key`
2. Use `OutputContract` metadata for any custom overrides
3. Remove lazy loading of contracts module

**Lines to Remove**: ~100 lines (fallback functions + lazy providers)

---

## Files Summary

| File | Action | Lines Affected | Risk | Status |
|------|--------|----------------|------|--------|
| `src/codeintel/config/datasets/contracts.py` | **Delete dictionaries** | ~160 | Low | Pending |
| `src/codeintel/build/targets.py` | **Simplify TargetGraph** | ~280 | Medium | Pending |
| `src/codeintel/build/registry.py` | **Delete static targets** | ~500 | Medium | Pending |
| ~~`src/codeintel/build/plan.py`~~ | ~~Archive/delete~~ | ~~334~~ | ~~Low~~ | ✅ **DELETED** |
| ~~`src/codeintel/build/resolver.py`~~ | ~~Archive/delete~~ | ~~328~~ | ~~Low~~ | ✅ **DELETED** |
| ~~`src/codeintel/build/contracts_validation.py`~~ | ~~Delete~~ | ~~82~~ | ~~Low~~ | ✅ **DELETED** |
| `src/codeintel/build/state.py` | **Remove adapters** | ~200 | Low | Pending |
| `src/codeintel/build/hamilton/introspect.py` | **Remove targetgraph** | ~30 | Low | Pending |
| `src/codeintel/build/schemas/contract_provider.py` | **Remove fallbacks** | ~100 | Low | Pending |
| Various CLI commands | **Update defaults** | ~20 | Low | Pending |

**Completed**: 744 lines deleted
**Remaining**: ~1,290 lines

---

## Recommended Deletion Order

The following order minimizes risk by removing components in dependency order:

### Wave 1: Contract Shims (PR-82)

**Scope**: `src/codeintel/config/datasets/contracts.py`

1. Delete `_JSON_SCHEMA_BY_DATASET_NAME`
2. Delete `DEFAULT_JSONL_FILENAMES`
3. Delete `DEFAULT_PARQUET_FILENAMES`
4. Update `contract_provider.py` to use deterministic derivation

**Verification**:
- [ ] All export tests pass
- [ ] No hardcoded filename references in tests
- [ ] JSONL/Parquet exports produce correct filenames

### Wave 2: Remove graph_source="targetgraph" (PR-83+)

**Scope**: `introspect.py`, CLI commands

1. Remove `"targetgraph"` from `GraphSource` literal type
2. Remove conditional branches in `compute_plan()`
3. Update CLI help text
4. Update all snapshot goldens

**Verification**:
- [ ] All CLI snapshot tests pass
- [ ] `build plan/graph/explain` commands work without `--graph-source`
- [ ] Validator gate passes

### Wave 3: Archive Legacy Plan/Resolver (PR-84+)

**Scope**: `plan.py`, `resolver.py`

1. Move files to `src/codeintel/build/_legacy/`
2. Update any remaining imports
3. Add deprecation warnings
4. Eventually delete after one release cycle

**Verification**:
- [ ] No imports from legacy modules in main codebase
- [ ] Hamilton planner handles all use cases
- [ ] Build commands work end-to-end

### Wave 4: Simplify State Types (PR-85+)

**Scope**: `state.py`

1. Remove `_unified_to_legacy_*` functions
2. Remove `TargetState.from_unified()` / `DatabaseState.from_unified()`
3. Update callers to use unified types directly

**Verification**:
- [ ] State validation tests pass
- [ ] Build status commands work correctly

### Wave 5: Remove Static Target Registry (PR-86+)

**Scope**: `registry.py`

1. Remove all `*_TARGET` constants
2. Remove `ALL_TARGETS` tuple
3. Update `build_target_graph()` to derive from Hamilton
4. Keep minimal API surface for compatibility

**Verification**:
- [ ] `get_target_graph()` returns correct graph
- [ ] All targets discoverable via Hamilton DAG
- [ ] BuildSpec compilation works

### Wave 6: Final Cleanup

**Scope**: Various files

1. Remove `TargetGraph` class (replace with thin Hamilton wrapper)
2. Remove unused imports and dead code
3. Update documentation

---

## Migration Guidance

### For Code Calling `PlanGenerator`

**Before**:
```python
from codeintel.build.plan import PlanGenerator
from codeintel.build.resolver import BuildResolver

graph = get_target_graph()
state = validator.validate()
resolver = BuildResolver(graph, state)
resolution = resolver.resolve(goals)
plan = PlanGenerator(graph).generate(resolution)
```

**After**:
```python
from codeintel.build.hamilton.planner import compute_plan

plan = compute_plan(
    env=env,
    requested=tuple(goals),
    graph_source="hamilton",
)
```

### For Code Using Legacy State Types

**Before**:
```python
from codeintel.build.state import DatabaseState, TargetState

db_state = validator.validate()
target_state = db_state.get("function_metrics")
if target_state.status == "computed":
    ...
```

**After**:
```python
from codeintel.build.state_types import BuildState

build_state = computer.compute_all()
target_state = build_state.targets["function_metrics"]
if target_state.status == "current":
    ...
```

### For Code Using Static Target Constants

**Before**:
```python
from codeintel.build.registry import FUNCTION_METRICS_TARGET, ALL_TARGETS

target = FUNCTION_METRICS_TARGET
for t in ALL_TARGETS:
    ...
```

**After**:
```python
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import target_graph_from_hamilton

runtime = build_driver(mode="auto")
graph = target_graph_from_hamilton(runtime)
target = graph.get("function_metrics")
for t in graph.all_targets:
    ...
```

---

## Risk Assessment

| Component | Risk | Mitigation |
|-----------|------|------------|
| Contract shims | **Low** | Deterministic derivation is well-understood |
| TargetGraph | **Medium** | Phase 5 validator catches dependency drift |
| Static registry | **Medium** | BuildSpec provides stable alternative |
| Plan/resolver | **Low** | Hamilton planner is proven in production |
| State types | **Low** | Unified types already exist and are tested |
| graph_source | **Low** | Hamilton path is already default |

### Rollback Strategy

Each wave should be:
1. Feature-flagged initially (environment variable to restore legacy behavior)
2. Monitored for one release cycle
3. Flag removed only after confidence is established

---

## Verification Checklist

### Pre-Deletion Gates

- [ ] Phase 5 DoD gates all pass (PR-74 through PR-80)
- [ ] `build validate` returns zero issues
- [ ] BuildSpec compile is deterministic (hash stable)
- [ ] All CLI snapshot tests pass
- [ ] Integration tests pass with `graph_source="hamilton"`

### Post-Deletion Verification

- [ ] `pytest` passes (no new xfails)
- [ ] Build commands work end-to-end
- [ ] Export produces correct filenames
- [ ] Serving layer can read BuildSpec
- [ ] No import errors from removed modules

### Documentation Updates

- [ ] AGENTS.md updated to reflect new architecture
- [ ] CLI help text reflects Hamilton-only mode
- [ ] Architecture docs updated
- [ ] Migration guide published (this document)

---

## Appendix: Related Phase 5 PRs

| PR | Description | Enables Deprecation Of |
|----|-------------|------------------------|
| PR-74 | Auto mode native helpers | Nothing directly, but enables PR-79 |
| PR-75 | BuildSpec primitives | Static target registry |
| PR-76 | BuildSpec compiler | TargetGraph as source of truth |
| PR-77 | BuildSpec CLI | Manual contract inspection |
| PR-78 | Validator gate | graph_source="targetgraph" |
| PR-79 | Hamilton default | Legacy fallback paths |
| PR-80 | Batch inference | Per-table schema loops |

---

## Appendix B: Build Directory File Assessment

A comprehensive review of all files in `src/codeintel/build/` (excluding subdirectories) was conducted to assess their current status and determine which are actively used vs. legacy/deprecated.

### Actively Used (Retain)

These modules are core infrastructure with active imports and no replacement planned:

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `__init__.py` | 159 | Public API exports and lazy loading | **Active** |
| `contracts.py` | 297 | `OutputContract`, `ArtifactSpec`, `TableSchema` | **Active** |
| `context.py` | 582 | `TargetExecutionContext` for plugin execution | **Active** |
| `context_base.py` | 605 | Base classes for execution context | **Active** |
| `errors.py` | 854 | Build error hierarchy with actionable hints | **Active** |
| `hashing.py` | 198 | Input hash computation for cache invalidation | **Active** |
| `manifest.py` | 172 | `OutputManifest`, `BuildRunRecord` data models | **Active** |
| `parameters.py` | 231 | `TargetParameters` configuration | **Active** |
| `plugin.py` | 425 | `TargetPlugin` protocol and base class | **Active** |
| `protocols.py` | 333 | DI protocols (`ToolRunner`, `ScipIndexer`, etc.) | **Active** |
| `providers.py` | 1070 | Real implementations of DI protocols | **Active** |
| `resources.py` | 177 | `TargetResources`, `TargetExecution` | **Active** |
| `result.py` | 92 | `TargetResult` for plugin execution results | **Active** |
| `run_config.py` | 66 | `BuildRunConfig` tying profiles to options | **Active** |
| `session.py` | 226 | `BuildSession` for session-scoped caching | **Active** |
| `types.py` | 343 | Shared types (`ToolRunResult`, etc.) | **Active** |
| `config.py` | 359 | `BuildConfig` and loading utilities | **Active** |
| `unified_registry.py` | 461 | `UnifiedRegistry` for atomic target registration | **Active** |
| `registrations.py` | 333 | `register_all_targets()` registration functions | **Active** |

### Unified State Infrastructure (Retain, Canonical)

These modules represent the new, unified state architecture:

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `state_types.py` | 415 | **Canonical** unified state types (`BuildState`, `TargetState`) | **Active - Canonical** |
| `state_computer.py` | 415 | `StateComputer` - single source of truth for state computation | **Active - Canonical** |

### Legacy / Compatibility (Deprecate)

These modules are legacy, only used by tests, or provide compatibility shims:

| File | Lines | Purpose | Status | Usage |
|------|-------|---------|--------|-------|
| `state.py` | 481 | Legacy state types with `from_unified()` adapters | **Legacy** | Used by CLI handlers; delegates to `StateComputer` internally |
| ~~`plan.py`~~ | ~~334~~ | ~~Legacy `PlanGenerator` and `BuildPlan` types~~ | ✅ **DELETED** | ~~Only used by tests~~ |
| ~~`resolver.py`~~ | ~~328~~ | ~~Legacy `BuildResolver` for minimal-work resolution~~ | ✅ **DELETED** | ~~Only used by `plan.py` and tests~~ |
| ~~`readiness.py`~~ | ~~834~~ | ~~`DatabaseReadinessView` complex readiness computation~~ | ✅ **DELETED** | ~~Redundant complexity given unified types~~ |
| ~~`contracts_validation.py`~~ | ~~82~~ | ~~`validate_contracts()` for contract validation~~ | ✅ **DELETED** | ~~Only used by 1 test file~~ |
| `targets.py` | 475 | `OutputTarget`, `TargetGraph` with manual dependency tracking | **Partially Legacy** | Active but dependency tracking superseded by Hamilton |
| `registry.py` | 821 | Static `*_TARGET` constants and `ALL_TARGETS` tuple | **Partially Legacy** | Active but static definitions superseded by Hamilton/BuildSpec |

### Dead Code Analysis

> ✅ **All items in this section have been DELETED as of 2025-12-15**

The following modules had **no imports from production code** and were only used by tests:

1. ~~**`plan.py`** (334 lines)~~ ✅ **DELETED**
   - Had zero imports from src/
   - Test files also deleted: `test_plan.py`
   - Tests referencing plan types removed from `test_hashing_plan_targets.py`

2. ~~**`resolver.py`** (328 lines)~~ ✅ **DELETED**
   - Was only imported by `plan.py` (also deleted)
   - Test files also deleted: `test_resolver.py`, `test_resolver_additional.py`
   - Tests referencing resolver removed from `test_readiness_registry_resources_resolver.py`

3. ~~**`contracts_validation.py`** (82 lines)~~ ✅ **DELETED**
   - Had zero imports from src/
   - Test file also deleted: `test_pr16_contract_parity.py`

4. ~~**`readiness.py`** (834 lines)~~ ✅ **DELETED**
   - Provided redundant complexity given unified state types
   - Test file also deleted: `test_readiness.py`
   - Readiness tests removed from `test_readiness_registry_resources_resolver.py`

### Recommended Build Directory Deprecation

#### Immediate Deprecation (Post-Phase 5)

> ✅ **All items in this section have been COMPLETED as of 2025-12-15**

| File | Action | Lines | Status |
|------|--------|-------|--------|
| ~~`plan.py`~~ | ~~Archive to `_legacy/` → Delete~~ | ~~334~~ | ✅ **DELETED** |
| ~~`resolver.py`~~ | ~~Archive to `_legacy/` → Delete~~ | ~~328~~ | ✅ **DELETED** |
| ~~`contracts_validation.py`~~ | ~~Delete~~ | ~~82~~ | ✅ **DELETED** |

**Also deleted:**
- `tests/build/test_plan.py`
- `tests/build/test_resolver.py`
- `tests/build/test_resolver_additional.py`
- `tests/build/hamilton/test_pr16_contract_parity.py`
- Resolver/plan tests removed from `test_hashing_plan_targets.py` and `test_readiness_registry_resources_resolver.py`

#### Near-Term Deprecation (Wave 4-5)

> ✅ **Wave 4 COMPLETED (2025-12-15)**: Registry deprecation

| File | Action | Lines | Risk | Status |
|------|--------|-------|------|--------|
| ~~`state.py`~~ | ~~Remove adapters~~ | ~~200~~ | ~~Medium~~ | ✅ **COMPLETED** (Wave 2) |
| ~~`registry.py`~~ | ~~Deprecate static exports~~ | ~~45 constants~~ | ~~Medium~~ | ✅ **COMPLETED** (Wave 4) |

**Wave 4 Details:**
- Removed 45 individual `*_TARGET` constants from `__all__` (constants still exist for `registrations.py`)
- Updated tests to use `get_target_graph()` API instead of direct constant imports
- Files updated: `test_registry.py`, `test_state.py`, `test_registry_consistency.py`, `test_hamilton_phase0.py`

#### Long-Term Refactoring (Wave 5+)

> ✅ **Wave 5 COMPLETED (2025-12-15)**: TargetGraph documentation updated

| File | Action | Lines | Risk | Status |
|------|--------|-------|------|--------|
| ~~`targets.py`~~ | ~~Document Hamilton-first architecture~~ | ~~docstrings~~ | ~~Low~~ | ✅ **COMPLETED** |

**Wave 5 Details:**
- Updated module docstring to explain Hamilton-first architecture
- Updated `TargetGraph` docstring with Hamilton usage example
- Noted that static `dependencies` field remains for `registrations.py` compatibility
- Hamilton is documented as source of truth for actual execution dependencies

### Import Graph Summary

```
Active Path (Hamilton-first):
  CLI → handlers/build.py → hamilton/planner.py → state_computer.py → session.py → hashing.py

Legacy Path (Still exists but deprecating):
  CLI → handlers/build.py → state.py → state_computer.py (delegation)
                          └→ resolver.py → plan.py (test-only)

Dead End (No src imports):
  tests/* → plan.py → resolver.py
  tests/* → contracts_validation.py
```

### Migration Priority Matrix

| Priority | Files | Reason | Status |
|----------|-------|--------|--------|
| ~~**P0**~~ | ~~`plan.py`, `resolver.py`, `contracts_validation.py`~~ | ~~Zero production usage~~ | ✅ **COMPLETED** |
| ~~**P0.5**~~ | ~~`readiness.py`~~ | ~~Redundant with unified state~~ | ✅ **COMPLETED** |
| ~~**P1**~~ | ~~`state.py` adapter types~~ | ~~Thin wrapper over canonical types~~ | ✅ **COMPLETED** (Wave 2) |
| ~~**P2**~~ | ~~`registry.py` static targets~~ | ~~High usage but superseded~~ | ✅ **COMPLETED** (Wave 4) |
| ~~**P3**~~ | ~~`targets.py` dependency tracking~~ | ~~Core type still needed, only tracking legacy~~ | ✅ **COMPLETED** (Wave 5) |

---

**Document Version**: 1.5
**Last Updated**: 2025-12-15
**Author**: CodeIntel Build Team

### Changelog

- **v1.5** (2025-12-15): Completed Wave 5 - targets.py Hamilton-first documentation:
  - Updated module docstring to explain Hamilton-first architecture
  - Added `target_graph_from_hamilton()` usage example to `TargetGraph` docstring
  - Documented that static `dependencies` field remains for compatibility
  - All priority items (P0-P3) now completed
- **v1.4** (2025-12-15): Completed Wave 4 - registry.py deprecation:
  - Removed individual `*_TARGET` constants from `__all__` (deprecated)
  - Updated 4 test files to use `get_target_graph()` instead of direct imports
  - Tests now use `graph.all_targets` instead of `ALL_TARGETS`
  - Constants still exist for `registrations.py` compatibility
- **v1.3** (2025-12-15): Completed Wave 3 - readiness.py deletion:
  - Deleted `readiness.py` (834 lines)
  - Deleted `test_readiness.py`
  - Updated `test_readiness_registry_resources_resolver.py` to remove readiness tests
  - Updated `__init__.py` to remove readiness import
  - Total deleted: 1,148 lines across all waves
- **v1.2** (2025-12-15): Completed P0 immediate deletions:
  - Deleted `plan.py`, `resolver.py`, `contracts_validation.py` (744 lines)
  - Deleted 4 test files: `test_plan.py`, `test_resolver.py`, `test_resolver_additional.py`, `test_pr16_contract_parity.py`
  - Updated 2 test files to remove references to deleted modules
- **v1.1** (2025-12-15): Added Appendix B with detailed build directory file assessment
- **v1.0** (2025-12-15): Initial document with Phase 5 deprecation plan

