# Build Directory Consolidation Plan

> **Status**: Draft  
> **Author**: AI Assistant  
> **Date**: 2025-12-16  
> **Scope**: `src/codeintel/build/` (~22,000 lines, 177 files)

## Executive Summary

The build system has evolved through multiple architectural phases:
1. **Original plugin-based architecture** with scattered ClassVars
2. **Hamilton-First migration** introducing native modules
3. **Unified registry consolidation** centralizing target-plugin associations

This evolution has left several layers of abstraction that can now be consolidated. This document identifies duplication, proposes consolidation strategies, and outlines a phased implementation plan to achieve a best-in-class architecture.

### Key Metrics

| Category | Current | After Consolidation |
|----------|---------|---------------------|
| Context types | 7+ | 2-3 |
| Registry systems | 4+ | 1-2 |
| Schema providers | 5+ | 1 |
| Lines of code | ~22,000 | ~18,000 (estimated) |

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Context Hierarchy Consolidation](#2-context-hierarchy-consolidation)
3. [Registry Unification](#3-registry-unification)
4. [Schema Provider Simplification](#4-schema-provider-simplification)
5. [Plugin System Deprecation](#5-plugin-system-deprecation)
6. [Dead Code Removal](#6-dead-code-removal)
7. [Recommended Target Architecture](#7-recommended-target-architecture)
8. [Implementation Phases](#8-implementation-phases)
9. [Risk Assessment](#9-risk-assessment)
10. [Appendix: File Inventory](#appendix-file-inventory)

---

## 1. Current Architecture Analysis

### 1.1 Directory Structure Overview

```
src/codeintel/build/
├── __init__.py              # Public API with lazy imports
├── targets.py               # OutputTarget and TargetGraph
├── registry.py              # Target constants and get_target_graph()
├── unified_registry.py      # UnifiedRegistry for target-plugin associations
├── registrations.py         # Registration functions
├── context.py               # TargetExecutionContext
├── context_base.py          # BuildContext, ExecutionContext
├── plugin.py                # TargetPlugin hierarchy
├── contracts.py             # OutputContract, ArtifactSpec
├── resources.py             # TargetResources, TargetExecution
├── parameters.py            # TargetParameters
├── protocols.py             # DI protocols (ToolRunner, etc.)
├── providers.py             # Protocol implementations
├── errors.py                # Error hierarchy
├── types.py                 # Result types (ToolRunResult, etc.)
├── state.py                 # StateValidator
├── state_computer.py        # StateComputer
├── state_types.py           # TargetState, BuildState
├── session.py               # BuildSession caching
├── hashing.py               # Input hash computation
├── config.py                # BuildConfig loading
├── result.py                # TargetResult
├── run_config.py            # BuildRunConfig
├── manifest.py              # Re-exports from storage
├── hamilton/                # Hamilton integration (~100 files)
│   ├── native/              # Native target modules
│   ├── contracts/           # Contract enforcement
│   ├── hooks/               # Lifecycle hooks
│   └── ...
├── schemas/                 # Schema resolution (~15 files)
├── plugins/                 # Legacy plugin infrastructure
├── exports/                 # Export functionality
├── assets/                  # Asset tracking
├── serving/                 # Semantic registry compilation
└── spec/                    # Build specification
```

### 1.2 Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│                    (codeintel build ...)                         │
├─────────────────────────────────────────────────────────────────┤
│                      Execution Layer                             │
│   HamiltonBuildExecutor → Driver → Native Modules                │
├─────────────────────────────────────────────────────────────────┤
│                      Planning Layer                              │
│   StateValidator → StateComputer → BuildSession                  │
├─────────────────────────────────────────────────────────────────┤
│                      Registry Layer                              │
│   UnifiedRegistry ←→ TargetGraph ←→ NativeModuleLoader          │
├─────────────────────────────────────────────────────────────────┤
│                      Context Layer                               │
│   BuildContext → ExecutionContext → TargetExecutionContext       │
├─────────────────────────────────────────────────────────────────┤
│                      Schema Layer                                │
│   UnifiedSchemaProvider → (Hamilton | Target | Declared)         │
├─────────────────────────────────────────────────────────────────┤
│                      Storage Layer                               │
│   StorageGateway → Warehouse → DuckDB                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Context Hierarchy Consolidation

### 2.1 Current State: Context Proliferation

The codebase has evolved to include 7+ context types:

| Context Type | Location | Purpose | Mutability |
|-------------|----------|---------|------------|
| `ContextPropertiesProtocol` | `context_base.py` | Interface for shared properties | Protocol |
| `BuildContext` | `context_base.py` | Base for materialization | Frozen |
| `ExecutionContext` | `context_base.py` | Extended for target execution | Mutable |
| `TargetExecutionContext` | `context.py` | Full plugin execution context | Mutable |
| `MaterializationContext` | `hamilton/native/materializer.py` | **Deprecated** - backward compat | Frozen |
| `ArtifactMaterializationContext` | `hamilton/native/artifact_materializer.py` | Artifact-specific context | Frozen |
| `GoidExtractionContext` | `hamilton/native/graphs/goids.py` | Domain-specific context | Frozen |

Additionally, there are internal context types:
- `_RunContext` in `hamilton/executor.py`
- `_PluginExecContext` in `hamilton/nodes/node_factory.py`
- `_EdgeCollectionContext` in `hamilton/native/graphs/call_graph.py`
- `_VersionContext` in `assets/emitter.py`

### 2.2 Problems Identified

1. **Duplicate Properties**: `ExecutionContext` and `TargetExecutionContext` duplicate `repo`, `commit`, `repo_root`, `build_dir`, `scip_dir`, `artifact_path()`

2. **Deprecated Context Still Used**: `MaterializationContext` is deprecated but `materialize_table()` still accepts `BuildContext | MaterializationContext`

3. **Inconsistent Patterns**: Some contexts use composition, others inheritance; some are frozen, others mutable

4. **PathResolver Duplication**: `PathResolver` exists in `context_base.py` but `artifact_path()` is implemented separately in both `ExecutionContext` and `TargetExecutionContext`

### 2.3 Consolidation Recommendation

#### Target Hierarchy

```python
# context_base.py - Keep as canonical
@runtime_checkable
class BuildContextProtocol(Protocol):
    """Minimal protocol for all build operations."""
    @property
    def gateway(self) -> StorageGateway: ...
    @property
    def snapshot(self) -> SnapshotRef: ...
    @property
    def paths(self) -> BuildPaths: ...

@dataclass(frozen=True)
class BuildContext:
    """Immutable context for materialization and queries."""
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    session: BuildSession | None = None
    validate_schemas: bool = False
    owner_target: str | None = None
    input_hash: str | None = None
    
    # Delegate to PathResolver for all path operations
    @property
    def path_resolver(self) -> PathResolver:
        return PathResolver(paths=self.paths, snapshot=self.snapshot)

# context.py - Simplified TargetExecutionContext
@dataclass
class TargetExecutionContext:
    """Mutable context for target plugin execution."""
    # Composition: include BuildContext rather than duplicate fields
    build_ctx: BuildContext
    target: OutputTarget
    resources: ContextResources
    parameters: TargetParameters = field(default_factory=lambda: EMPTY_PARAMETERS)
    _written_tables: dict[str, WriteRecord] = field(default_factory=dict)
    
    # Delegate common properties to build_ctx
    @property
    def gateway(self) -> StorageGateway:
        return self.build_ctx.gateway
    
    @property
    def snapshot(self) -> SnapshotRef:
        return self.build_ctx.snapshot
    
    # ... other delegations
```

#### Migration Path

1. **Phase 1**: Delete `MaterializationContext` entirely
   - Update `materialize_table()` to accept only `BuildContext`
   - Update all callers (~10 files)

2. **Phase 2**: Merge `ExecutionContext` into `BuildContext`
   - Add optional `target` field to `BuildContext`
   - Deprecate `ExecutionContext` import

3. **Phase 3**: Simplify `TargetExecutionContext`
   - Use composition with `BuildContext` instead of field duplication
   - Keep only execution-specific fields (resources, written_tables)

### 2.4 Files to Modify

| File | Changes |
|------|---------|
| `context_base.py` | Remove `ExecutionContext`, enhance `BuildContext` |
| `context.py` | Simplify `TargetExecutionContext` to use composition |
| `hamilton/native/materializer.py` | Remove `MaterializationContext`, update functions |
| `hamilton/native/artifact_materializer.py` | Use `BuildContext` directly |
| ~10 native modules | Update context usage |

---

## 3. Registry Unification

### 3.1 Current State: Multiple Registries

| Registry | Purpose | Location | Access |
|----------|---------|----------|--------|
| `TargetGraph` | Target dependencies | `targets.py` | `get_target_graph()` |
| `UnifiedRegistry` | Target-plugin associations | `unified_registry.py` | `get_unified_registry()` |
| `NativeModuleLoader` | Native Hamilton modules | `hamilton/native/loader.py` | `get_loader()` |
| `SCHEMA_REGISTRY` | Pandera schemas | `hamilton/contracts/schemas/registry.py` | Direct import |

### 3.2 Problems Identified

1. **Separate Truth Sources**: `TargetGraph` holds dependency info while `UnifiedRegistry` holds implementations
2. **Module List Duplication**: `NativeModuleLoader._NATIVE_MODULE_PACKAGES` is separate from `registrations.py`
3. **Initialization Order**: `get_target_graph()` builds from Hamilton, `get_unified_registry()` builds from registrations
4. **Cache Management**: Each registry has separate `@lru_cache` or `SingletonHolder`

### 3.3 Consolidation Recommendation

#### Unified TargetRegistry

```python
# target_registry.py (NEW)
@dataclass
class TargetRegistry:
    """Single source of truth for targets, implementations, and graph."""
    
    _targets: dict[str, OutputTarget]
    _plugins: dict[str, type[TargetPlugin]]
    _native_modules: dict[str, str]  # target_name -> module_path
    _graph: TargetGraph
    
    def get_target(self, name: str) -> OutputTarget:
        """Get target by name."""
        return self._targets[name]
    
    def get_plugin(self, name: str) -> type[TargetPlugin] | None:
        """Get plugin class for target."""
        return self._plugins.get(name)
    
    def get_native_module_path(self, name: str) -> str | None:
        """Get native module path for target."""
        return self._native_modules.get(name)
    
    def dependencies_of(self, name: str) -> tuple[str, ...]:
        """Get direct dependencies (from Hamilton DAG)."""
        return self._graph.dependencies_of(name)
    
    def topological_order(self, names: Iterable[str]) -> tuple[str, ...]:
        """Sort targets in dependency order."""
        return self._graph.topological_order(names)
    
    def is_native(self, name: str) -> bool:
        """Check if target has native Hamilton implementation."""
        return name in self._native_modules
    
    @classmethod
    def build(cls) -> TargetRegistry:
        """Build from Hamilton DAG and registrations."""
        # 1. Build Hamilton driver
        driver = build_driver(mode="auto")
        
        # 2. Derive graph from Hamilton
        graph = target_graph_from_hamilton(driver)
        
        # 3. Load registrations
        targets = {t.name: t for t in ALL_TARGETS}
        
        # 4. Load native modules
        loader = NativeModuleLoader()
        native_modules = {}
        for target_name in loader.get_target_names():
            # Map target to module path
            native_modules[target_name] = _find_module_for_target(target_name)
        
        return cls(
            _targets=targets,
            _plugins={},  # Populated from registrations
            _native_modules=native_modules,
            _graph=graph,
        )

# Singleton access
_registry_holder = SingletonHolder[TargetRegistry]()

def get_target_registry() -> TargetRegistry:
    """Get the singleton target registry."""
    return _registry_holder.get(TargetRegistry.build)
```

#### Migration Path

1. **Phase 1**: Create `TargetRegistry` combining `UnifiedRegistry` + `TargetGraph` interfaces
2. **Phase 2**: Migrate callers from `get_unified_registry()` to `get_target_registry()`
3. **Phase 3**: Migrate callers from `get_target_graph()` to `get_target_registry()`
4. **Phase 4**: Deprecate old accessors

### 3.4 Files to Modify

| File | Changes |
|------|---------|
| `target_registry.py` (NEW) | Create unified registry |
| `unified_registry.py` | Deprecate, forward to new registry |
| `registry.py` | Deprecate `get_target_graph()`, forward to new registry |
| `registrations.py` | Update to populate new registry |
| ~20 files | Update imports |

---

## 4. Schema Provider Simplification

### 4.1 Current State: Multi-tier Fallback

```
UnifiedSchemaProvider (provider_unified.py)
├── Tier 1: Hamilton-native inference (provider_hamilton.py)
│   └── Infers from q__* Ibis compute nodes
├── Tier 2: Target-declared schemas (OutputContract.tables)
│   └── From registry.py target definitions
└── Tier 3: Raw declared schemas (provider_declared.py)
    └── From declared_schemas.py TABLE_SCHEMAS

Additional registries:
├── SCHEMA_REGISTRY (hamilton/contracts/schemas/registry.py)
│   └── Pandera DataFrameSchema instances
├── ContractProvider (schemas/contract_provider.py)
│   └── Dataset contracts with metadata
├── JSON Schema Registry (schemas/json_schema_registry.py)
│   └── JSON Schema for exports
└── Row Binding Registry (schemas/row_registry.py)
    └── TypedDict row models
```

### 4.2 Problems Identified

1. **15 Files** in `schemas/` directory for related functionality
2. **Multiple Caches**: Each provider has `@lru_cache`, cleared separately
3. **Unclear Authority**: Which provider is authoritative for a given table?
4. **Circular Imports**: Extensive use of lazy imports to avoid cycles

### 4.3 Consolidation Recommendation

#### Pluggable SchemaRegistry

```python
# schemas/registry.py (ENHANCED)
class SchemaResolver(Protocol):
    """Protocol for schema resolution strategies."""
    
    def resolve(self, table_key: str) -> TableSchema | None:
        """Attempt to resolve schema for table_key."""
        ...
    
    def clear_cache(self) -> None:
        """Clear resolver-specific cache."""
        ...
    
    @property
    def priority(self) -> int:
        """Higher priority resolvers are tried first."""
        ...

@dataclass
class SchemaRegistry:
    """Unified schema registry with pluggable resolvers."""
    
    _resolvers: list[SchemaResolver]  # Sorted by priority
    _cache: dict[str, TableSchema] = field(default_factory=dict)
    
    def get(self, table_key: str) -> TableSchema | None:
        """Resolve schema through resolver chain."""
        if table_key in self._cache:
            return self._cache[table_key]
        
        for resolver in self._resolvers:
            if schema := resolver.resolve(table_key):
                self._cache[table_key] = schema
                return schema
        return None
    
    def require(self, table_key: str) -> TableSchema:
        """Get schema or raise KeyError."""
        schema = self.get(table_key)
        if schema is None:
            raise KeyError(f"Unknown schema: {table_key}")
        return schema
    
    def clear_all_caches(self) -> None:
        """Clear all caches with single call."""
        self._cache.clear()
        for resolver in self._resolvers:
            resolver.clear_cache()
    
    @classmethod
    def build_default(cls) -> SchemaRegistry:
        """Build with standard resolver chain."""
        return cls(
            _resolvers=[
                HamiltonInferenceResolver(priority=100),
                TargetContractResolver(priority=50),
                DeclaredSchemaResolver(priority=10),
            ]
        )
```

#### Migration Path

1. **Phase 1**: Create `SchemaResolver` protocol and registry wrapper
2. **Phase 2**: Wrap existing providers as resolvers
3. **Phase 3**: Consolidate `clear_*_cache()` functions into single method
4. **Phase 4**: Deprecate direct provider access

### 4.4 Files to Consolidate

| Current Files | Action |
|--------------|--------|
| `provider_unified.py` | Becomes `HamiltonInferenceResolver` + `TargetContractResolver` |
| `provider_declared.py` | Becomes `DeclaredSchemaResolver` |
| `provider_hamilton.py` | Merge into `HamiltonInferenceResolver` |
| `registry.py` | Enhance with pluggable architecture |
| `contract_provider.py` | Keep separate (different concern) |
| `json_schema_registry.py` | Keep separate (export-specific) |

---

## 5. Plugin System Deprecation

### 5.1 Current State

```python
# plugin.py hierarchy
TargetPluginProtocol (Protocol)
└── TargetPlugin (ABC)
    ├── MetadataPlugin (adds automatic metadata)
    │   └── FactoryPlugin[TStep] (adds factory methods)
    └── Direct implementations
```

### 5.2 Analysis

Since all targets now have native Hamilton implementations:
- Plugin classes are only used during testing
- `MetadataPlugin` and `FactoryPlugin` boilerplate is rarely used
- Factory types (`StorageFactory`, `DiscoveryFactory`) have limited utility

### 5.3 Recommendation

1. **Keep**: `TargetPluginProtocol` as minimal interface for extension points
2. **Deprecate**: `MetadataPlugin` and `FactoryPlugin` with warnings
3. **Remove** (Phase 3): After migration period, remove deprecated classes

```python
# plugin.py - Simplified
import warnings

@runtime_checkable
class TargetPluginProtocol(Protocol):
    """Minimal interface for target plugins."""
    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str]
    plugin_description: ClassVar[str]
    
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...

class TargetPlugin(ABC):
    """Base class for target plugins."""
    # Keep as-is for backward compatibility

class MetadataPlugin(TargetPlugin, ABC):
    """Enhanced plugin base with automatic metadata handling.
    
    .. deprecated:: 2.0
        Use native Hamilton modules instead. This class will be
        removed in a future version.
    """
    
    def __init_subclass__(cls, **kwargs: object) -> None:
        warnings.warn(
            f"{cls.__name__} inherits from deprecated MetadataPlugin. "
            "Migrate to native Hamilton modules.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
```

---

## 6. Dead Code Removal

### 6.1 Immediate Removal Candidates

| Item | Location | Lines | Reason |
|------|----------|-------|--------|
| `plugins/analytics/` | Directory | Empty | All plugins migrated to native |
| `plugins/graphs/` | Directory | Empty | All plugins migrated to native |
| `plugins/graphs/builders/` | Directory | Deleted | Confirmed empty |
| `_analytics_plugins()` | `registrations.py:63-71` | ~10 | References deleted module |
| `_graphs_plugins()` | `registrations.py:74-82` | ~10 | References deleted module |
| `MaterializationContext` | `hamilton/native/materializer.py:70-151` | ~80 | Deprecated, use BuildContext |

### 6.2 Safe Deletion Verification

```bash
# Verify no references to deleted plugin modules
grep -r "from codeintel.build.plugins.analytics" src/ tests/
# Expected: No matches

grep -r "from codeintel.build.plugins.graphs" src/ tests/
# Expected: No matches (after previous cleanup)

# Verify MaterializationContext usage
grep -r "MaterializationContext" src/ --include="*.py" | grep -v "deprecated"
# Review results - should be minimal
```

### 6.3 Migration-Required Removals

| Item | Dependents | Migration Effort |
|------|-----------|------------------|
| `ExecutionContext` | ~15 files | Medium - update to BuildContext |
| `relpath_to_module` | 0 files | Already cleaned up |

---

## 7. Recommended Target Architecture

### 7.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public API                               │
│                                                                  │
│  get_target_registry()   get_schema_registry()   BuildContext   │
└────────────────┬──────────────────┬─────────────────┬───────────┘
                 │                  │                 │
┌────────────────▼──────────────────▼─────────────────▼───────────┐
│                      Core Abstractions                           │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │ TargetRegistry │  │ SchemaRegistry │  │  BuildContext  │     │
│  │                │  │                │  │                │     │
│  │ - targets      │  │ - resolvers    │  │ - gateway      │     │
│  │ - graph        │  │ - cache        │  │ - snapshot     │     │
│  │ - native_mods  │  │                │  │ - paths        │     │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘     │
└───────────┼───────────────────┼───────────────────┼─────────────┘
            │                   │                   │
┌───────────▼───────────────────▼───────────────────▼─────────────┐
│                      Execution Layer                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 HamiltonBuildExecutor                    │    │
│  │                                                          │    │
│  │  Driver → NativeModules → materialize_table()           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────┐
│                       State Layer                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │StateComputer │─▶│ BuildSession │─▶│OutputManifest│           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│                                                                  │
│  StorageGateway → Warehouse → DuckDB                            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Design Principles

1. **Single Registry Pattern**: One `TargetRegistry` for all target-related lookups
2. **Pluggable Schema Resolution**: `SchemaRegistry` with ordered resolver chain
3. **Unified Context**: `BuildContext` as base, `TargetExecutionContext` for execution
4. **Hamilton-First Execution**: All execution through Hamilton driver
5. **Session-Scoped Caching**: `BuildSession` owns all runtime caches

### 7.3 File Structure After Consolidation

```
src/codeintel/build/
├── __init__.py              # Public API
├── target_registry.py       # NEW: Unified TargetRegistry
├── targets.py               # OutputTarget, TargetGraph (internal)
├── context.py               # BuildContext, TargetExecutionContext
├── plugin.py                # TargetPluginProtocol (simplified)
├── contracts.py             # OutputContract, ArtifactSpec
├── resources.py             # TargetResources, TargetExecution
├── parameters.py            # TargetParameters
├── protocols.py             # DI protocols
├── providers.py             # Protocol implementations
├── errors.py                # Error hierarchy
├── types.py                 # Result types
├── state.py                 # StateValidator
├── state_computer.py        # StateComputer
├── state_types.py           # TargetState, BuildState
├── session.py               # BuildSession
├── hashing.py               # Input hash computation
├── config.py                # BuildConfig
├── result.py                # TargetResult
├── hamilton/                # Hamilton integration
│   ├── native/              # Native modules (unchanged)
│   ├── contracts/           # Contract enforcement
│   └── ...
├── schemas/                 # Simplified schema resolution
│   ├── __init__.py
│   ├── registry.py          # SchemaRegistry with resolvers
│   ├── resolvers/           # NEW: Pluggable resolvers
│   └── ...
├── exports/                 # Export functionality
├── assets/                  # Asset tracking
└── serving/                 # Semantic registry
```

---

## 8. Implementation Phases

### Phase 1: Quick Wins (1-2 days)

**Objective**: Remove confirmed dead code

| Task | Files | Est. Lines |
|------|-------|------------|
| Remove empty `plugins/analytics/` directory | 1 dir | 0 |
| Remove empty `plugins/graphs/` directory | 1 dir | 0 |
| Remove `_analytics_plugins()` function | `registrations.py` | ~10 |
| Remove `_graphs_plugins()` function | `registrations.py` | ~10 |
| Delete `MaterializationContext` class | `hamilton/native/materializer.py` | ~80 |
| Update `materialize_*` to accept only `BuildContext` | ~5 files | ~20 |

**Verification**:
```bash
uv run python -m tools.quality_report
uv run pytest -q
```

### Phase 2: Context Consolidation (3-5 days)

**Objective**: Simplify context hierarchy

| Task | Files | Est. Lines |
|------|-------|------------|
| Merge `ExecutionContext` into `BuildContext` | `context_base.py` | ~200 |
| Refactor `TargetExecutionContext` to use composition | `context.py` | ~100 |
| Update all context usages | ~30 files | ~200 |
| Add deprecation warnings | Various | ~20 |

**Verification**:
```bash
# Ensure no direct ExecutionContext imports remain
grep -r "from codeintel.build.context_base import.*ExecutionContext" src/
uv run pytest -q
```

### Phase 3: Registry Unification (5-7 days)

**Objective**: Single TargetRegistry

| Task | Files | Est. Lines |
|------|-------|------------|
| Create `TargetRegistry` class | `target_registry.py` (NEW) | ~200 |
| Implement `SingletonHolder` access | `target_registry.py` | ~30 |
| Add forwarding from `get_unified_registry()` | `unified_registry.py` | ~20 |
| Add forwarding from `get_target_graph()` | `registry.py` | ~20 |
| Migrate callers | ~20 files | ~100 |
| Deprecate old accessors | Various | ~30 |

**Verification**:
```bash
# Ensure registry consistency
uv run pytest tests/build/test_target_registry.py -v
uv run pytest -q
```

### Phase 4: Schema Simplification (3-5 days)

**Objective**: Pluggable SchemaRegistry

| Task | Files | Est. Lines |
|------|-------|------------|
| Create `SchemaResolver` protocol | `schemas/resolvers/protocol.py` | ~40 |
| Implement `HamiltonInferenceResolver` | `schemas/resolvers/hamilton.py` | ~100 |
| Implement `TargetContractResolver` | `schemas/resolvers/target.py` | ~60 |
| Implement `DeclaredSchemaResolver` | `schemas/resolvers/declared.py` | ~60 |
| Enhance `SchemaRegistry` | `schemas/registry.py` | ~100 |
| Consolidate cache clearing | Various | ~30 |

**Verification**:
```bash
uv run pytest tests/build/schemas/ -v
uv run pytest -q
```

### Phase 5: Plugin Deprecation (1-2 days)

**Objective**: Add deprecation warnings

| Task | Files | Est. Lines |
|------|-------|------------|
| Add deprecation to `MetadataPlugin` | `plugin.py` | ~20 |
| Add deprecation to `FactoryPlugin` | `plugin.py` | ~20 |
| Update docstrings | `plugin.py` | ~30 |

**Verification**:
```bash
# Ensure warnings are emitted
python -W default -c "from codeintel.build.plugin import MetadataPlugin"
```

---

## 9. Risk Assessment

### 9.1 High Risk Items

| Risk | Mitigation |
|------|------------|
| Context changes break plugins | Comprehensive test coverage before changes |
| Registry unification breaks imports | Use forwarding functions during transition |
| Schema resolution changes break validation | Keep fallback chain behavior identical |

### 9.2 Medium Risk Items

| Risk | Mitigation |
|------|------------|
| Circular import introduction | Maintain lazy import patterns |
| Cache inconsistency during migration | Clear all caches in tests |
| Hamilton driver initialization changes | Verify `build_driver()` returns consistent results |

### 9.3 Low Risk Items

| Risk | Mitigation |
|------|------------|
| Dead code removal breaks tests | Run full test suite after each deletion |
| Deprecation warnings cause noise | Use appropriate warning categories |

---

## 10. Success Criteria

### 10.1 Quantitative

- [ ] Total build directory lines reduced by ~4,000 (18% reduction)
- [ ] Context types reduced from 7+ to 2-3
- [ ] Registry systems reduced from 4+ to 1-2
- [ ] Schema provider files reduced from 15 to ~8
- [ ] All tests passing
- [ ] Quality gates (ruff, pyright, pyrefly) passing

### 10.2 Qualitative

- [ ] Single import path for registry access
- [ ] Single import path for schema access
- [ ] Clear context hierarchy documentation
- [ ] No deprecated code without warnings
- [ ] Improved import times (no circular import workarounds needed)

---

## Appendix: File Inventory

### A.1 Core Files (~11,000 lines)

| File | Lines | Status |
|------|-------|--------|
| `errors.py` | 854 | Keep |
| `registry.py` | 751 | Refactor (forward to TargetRegistry) |
| `context.py` | 582 | Refactor (composition) |
| `context_base.py` | 605 | Refactor (merge ExecutionContext) |
| `targets.py` | 484 | Keep (internal to TargetRegistry) |
| `unified_registry.py` | 461 | Deprecate (forward to TargetRegistry) |
| `plugin.py` | 425 | Simplify (deprecate subclasses) |
| `state_computer.py` | 415 | Keep |
| `state_types.py` | 415 | Keep |
| `registrations.py` | 396 | Cleanup (remove dead refs) |
| `config.py` | 359 | Keep |
| `types.py` | 343 | Keep |
| `protocols.py` | 333 | Keep |
| `contracts.py` | 297 | Keep |
| `parameters.py` | 231 | Keep |
| `session.py` | 226 | Keep |
| `hashing.py` | 198 | Keep |
| `resources.py` | 177 | Keep |
| `state.py` | 147 | Keep |
| `result.py` | 92 | Keep |
| `run_config.py` | 66 | Keep |
| `manifest.py` | 20 | Keep |

### A.2 Schemas Directory (~2,000 lines)

| File | Lines | Status |
|------|-------|--------|
| `__init__.py` | 162 | Keep |
| `registry.py` | 136 | Enhance |
| `provider_unified.py` | 287 | Refactor into resolvers |
| `provider_declared.py` | ~100 | Refactor into resolver |
| `provider_hamilton.py` | ~200 | Refactor into resolver |
| `contract_provider.py` | ~150 | Keep |
| `json_schema_registry.py` | ~100 | Keep |
| `row_registry.py` | ~100 | Keep |
| `declared_schemas.py` | ~200 | Keep |
| `compile.py` | ~150 | Keep |
| `diff.py` | ~150 | Keep |
| `infer_duckdb.py` | ~100 | Keep |
| `manifest.py` | ~100 | Keep |
| `seed_harness.py` | ~50 | Keep |

### A.3 Hamilton Directory (~9,000+ lines)

Largely unchanged - native modules are well-structured and should remain as-is.

---

## Changelog

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-16 | AI Assistant | Initial draft |

---

*This document is part of the post-Hamilton integration cleanup initiative.*

