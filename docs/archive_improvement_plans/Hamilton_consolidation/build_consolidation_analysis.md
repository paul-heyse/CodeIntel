# Build Directory Consolidation Analysis

> **Purpose**: Identify functional duplication in `src/codeintel/build/` and recommend consolidations aligned with the Hamilton-first architecture from Phase 5.

---

## Part I: 100% Native Hamilton Architecture (Recommended)

This section describes the aggressive, breaking-change approach: **completely eliminating the plugin abstraction layer** and making everything native Hamilton.

### Vision: Pure Hamilton, No Plugins

```
CURRENT (Dual Model):
┌─────────────────────────────────────────────────────────────────────┐
│                     Hamilton Driver                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐│
│  │ Native Hamilton │    │ Plugin Wrapper                          ││
│  │ t__risk_factors │    │ t__function_metrics (wrapper)           ││
│  │   ↓             │    │   ↓                                     ││
│  │ Pure Ibis       │    │ TargetPlugin.execute(ctx)              ││
│  │ compute node    │    │   ↓                                     ││
│  │   ↓             │    │ TargetExecutionContext                  ││
│  │ materialize()   │    │   ↓                                     ││
│  └─────────────────┘    │ ctx.write_table() [manual]              ││
│                         └─────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

PROPOSED (100% Native):
┌─────────────────────────────────────────────────────────────────────┐
│                     Hamilton Driver                                  │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ t__modules      │  │ t__ast          │  │ t__risk_factors │     │
│  │ t__scip         │  │ t__goids        │  │ t__hotspots     │     │
│  │ t__call_graph   │  │ t__profiles     │  │ t__coverage     │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┴────────────────────┘               │
│                               │                                     │
│                        ┌──────▼──────┐                              │
│                        │   BuildEnv   │                              │
│                        │  (only ctx)  │                              │
│                        └─────────────┘                              │
│                                                                      │
│  Dependencies via function signatures (Hamilton resolves)           │
│  Outputs via materialize_table() (uniform)                          │
│  Skip logic via Hamilton hooks (uniform)                            │
│  Manifests via Hamilton hooks (uniform)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### What Gets Deleted

| Component | Lines | Replacement |
|-----------|-------|-------------|
| `plugin.py` (TargetPlugin, TargetPluginProtocol) | 425 | Hamilton node functions |
| `context.py` (TargetExecutionContext) | 582 | `BuildEnv` only |
| `context_base.py` (ExecutionContext hierarchy) | 605 | `BuildEnv` only |
| `plugins/` directory (all plugin classes) | ~3,000+ | Hamilton native modules |
| Plugin wrapper in `node_factory.py` | ~200 | Direct native modules |
| `unified_registry.py` plugin tracking | ~150 | Not needed |
| `registrations.py` plugin registration | 333 | Not needed |

**Total Estimated Deletion: ~5,000+ lines**

### What Remains (Simplified)

```
src/codeintel/build/
├── hamilton/
│   ├── env.py                 # BuildEnv (the ONE context)
│   ├── native/
│   │   ├── ingestion/         # All ingestion targets as Hamilton modules
│   │   ├── graphs/            # All graph targets
│   │   ├── analytics/         # All analytics targets
│   │   └── export/            # All export targets
│   ├── hooks/
│   │   ├── manifest_hook.py   # Uniform manifest persistence
│   │   ├── skip_hook.py       # Uniform skip logic
│   │   └── telemetry_hook.py  # Uniform observability
│   ├── driver_factory.py      # Build the Hamilton driver
│   └── naming.py              # Node naming conventions
├── contracts.py               # OutputContract, TableSchema
├── manifest.py                # OutputManifest
├── targets.py                 # OutputTarget (metadata only)
├── registry.py                # Target metadata (no plugins)
├── hashing.py                 # Input hash computation
└── session.py                 # Caching layer
```

### The Native Hamilton Pattern (Template)

Every target becomes a Hamilton module following this pattern:

```python
# src/codeintel/build/hamilton/native/analytics/function_metrics.py
"""Native Hamilton implementation for function_metrics target.

This module demonstrates the canonical pattern for 100% Hamilton targets:
1. Pure compute nodes (Ibis transformations, no side effects)
2. A single materialize node (t__<target>) that persists results
3. Dependencies expressed as function parameters
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.materializer import materialize_table
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Compute Nodes (Pure, No Side Effects)
# -----------------------------------------------------------------------------

@tag(domain="analytics", target="function_metrics", node_type="compute")
def t__function_metrics__complexity(
    q__core__goids: ir.Table,          # Dependency: goids table
    q__core__ast_nodes: ir.Table,      # Dependency: AST data
) -> ir.Table:
    """Compute complexity metrics from AST data.
    
    This is a PURE Ibis transformation. No side effects, no I/O.
    Hamilton can cache, parallelize, and optimize this freely.
    """
    return (
        q__core__ast_nodes
        .filter(q__core__ast_nodes.node_type == "function")
        .group_by("function_goid_h128")
        .aggregate(
            cyclomatic_complexity=ibis._.complexity.sum(),
            loc=ibis._.loc.sum(),
            parameter_count=ibis._.parameters.count(),
        )
    )


@tag(domain="analytics", target="function_metrics", node_type="compute")
def t__function_metrics__types(
    q__core__goids: ir.Table,
    q__core__type_annotations: ir.Table,
) -> ir.Table:
    """Compute type coverage metrics.
    
    Pure Ibis transformation for type annotation analysis.
    """
    return (
        q__core__type_annotations
        .group_by("function_goid_h128")
        .aggregate(
            has_return_type=ibis._.return_annotation.notnull().any(),
            param_coverage=ibis._.annotated_params.mean(),
        )
    )


@tag(domain="analytics", target="function_metrics", node_type="compute")
def t__function_metrics__combined(
    t__function_metrics__complexity: ir.Table,
    t__function_metrics__types: ir.Table,
) -> ir.Table:
    """Combine complexity and type metrics into final output.
    
    Pure join operation, still no side effects.
    """
    return t__function_metrics__complexity.join(
        t__function_metrics__types,
        "function_goid_h128",
        how="left",
    )


# -----------------------------------------------------------------------------
# Materialize Node (Side Effect Boundary)
# -----------------------------------------------------------------------------

@tag(domain="analytics", target="function_metrics", node_type="materialize")
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_metrics__combined: ir.Table,
) -> TargetRunRecord:
    """Materialize function_metrics to DuckDB.
    
    This is the ONLY node with side effects. It:
    1. Checks if skip is possible (via manifest)
    2. Executes the Ibis expression
    3. Persists to DuckDB
    4. Returns a TargetRunRecord for tracking
    
    The Hamilton driver calls this node. All upstream compute nodes
    are automatically resolved via the DAG.
    """
    from codeintel.build.hamilton.native.executor import NativeTargetExecutor
    
    target = graph.get("function_metrics")
    executor = NativeTargetExecutor.for_target(env, graph, "function_metrics")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: {
        # Materialize produces row counts
        "analytics.function_metrics": materialize_table(
            env, "analytics.function_metrics", t__function_metrics__combined
        ),
    })
```

### Key Architectural Decisions

#### 1. BuildEnv is the ONLY Context

```python
@dataclass(frozen=True)
class BuildEnv:
    """Single context for all Hamilton node execution."""
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers           # DI for external tools
    config: BuildConfig
    force_targets: frozenset[str]
    manifest_index: Mapping[str, OutputManifest] | None
    validate_outputs: bool
```

No more:
- `TargetExecutionContext`
- `ExecutionContext`
- `BuildContext`
- `MaterializationContext`
- `ContextResources`

Just `BuildEnv`.

#### 2. Dependencies via Function Signatures

```python
# BEFORE (Plugin style)
class FunctionMetricsPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # Dependencies declared in OutputTarget.dependencies = ("goids",)
        # Must manually load: goids = ctx.gateway.load_table("core.goids")
        ...

# AFTER (Hamilton native)
def t__function_metrics__compute(
    q__core__goids: ir.Table,      # Hamilton injects this automatically
    q__core__ast_nodes: ir.Table,  # And this too
) -> ir.Table:
    # Dependencies are explicit in the signature
    # Hamilton resolves them from the DAG
    ...
```

#### 3. Outputs via Ibis Expressions

```python
# BEFORE (Plugin style)
rows = compute_metrics(data)
ctx.write_table("analytics.function_metrics", rows)  # Manual write

# AFTER (Hamilton native)
@tag(node_type="compute")
def t__function_metrics__compute(...) -> ir.Table:
    return ibis_expression  # Return expression, not data

@tag(node_type="materialize")  
def t__function_metrics(env, graph, t__function_metrics__compute: ir.Table):
    materialize_table(env, "analytics.function_metrics", t__function_metrics__compute)
```

#### 4. Uniform Skip Logic via Hooks

```python
# All targets use the same skip mechanism:
class ManifestHook(GraphExecutionHook):
    """Hamilton hook that handles skip logic uniformly."""
    
    def pre_node_execute(self, run_id, node_name, kwargs, task_id):
        if node_name.startswith("t__") and "materialize" in node_tags:
            target_name = extract_target_name(node_name)
            if self._should_skip(target_name, self.env):
                return SkipResult(...)
```

### Migration Path for Each Plugin

#### Step 1: Identify Plugin's Core Logic

```python
# From RepoScanPlugin.execute():
step = RepoScanStep(storage, discovery, change_detection, filter)
result, modules, change_set = step.execute(repo, commit, repo_root, profile, full_rebuild)
```

The core logic is `RepoScanStep.execute()`. This doesn't change.

#### Step 2: Create Hamilton Module

```python
# src/codeintel/build/hamilton/native/ingestion/modules.py

@tag(domain="ingestion", target="modules", node_type="compute")
def t__modules__scan(env: BuildEnv) -> ScanResult:
    """Execute repository scan (side-effect: reads filesystem)."""
    storage = DuckDBStorageAdapter(env.gateway)
    discovery = FilesystemDiscoveryAdapter(env.paths.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    
    step = RepoScanStep(storage, discovery, change_detection)
    return step.execute(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.paths.repo_root,
        profile=build_scan_profile(env),
    )


@tag(domain="ingestion", target="modules", node_type="materialize")
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ScanResult,
) -> TargetRunRecord:
    """Materialize scan results to DuckDB."""
    executor = NativeTargetExecutor.for_target(env, graph, "modules")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: persist_scan_result(env, t__modules__scan))
```

#### Step 3: Delete Plugin Class

Once the Hamilton module works, delete:
- `RepoScanPlugin` class
- Registration in `registrations.py`
- Entry in `unified_registry.py`

### What About Complex Plugins?

Some plugins have complex logic that doesn't fit the pure Ibis pattern:

1. **SCIP indexing** - Calls external binaries
2. **AST extraction** - Parses Python files
3. **Type checking** - Runs Pyright

These become Hamilton nodes that:
- Take `env: BuildEnv` for access to providers
- Call external tools via `env.providers.tool_runner`
- Return results that downstream nodes can use

```python
@tag(domain="ingestion", target="scip", node_type="compute")
def t__scip__index(
    env: BuildEnv,
    t__modules: TargetRunRecord,  # Depends on modules being done
) -> ScipIndexResult:
    """Index repository with SCIP (calls external binary)."""
    result = await env.providers.scip_indexer.index(
        repo_root=env.paths.repo_root,
        output_path=env.paths.scip_dir / "index.scip",
    )
    return result


@tag(domain="ingestion", target="scip", node_type="materialize")
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    t__scip__index: ScipIndexResult,
) -> TargetRunRecord:
    """Materialize SCIP index results."""
    executor = NativeTargetExecutor.for_target(env, graph, "scip")
    
    if executor.should_skip():
        return executor.skip()
    
    return executor.execute(lambda: {
        "core.scip_symbols": persist_symbols(env, t__scip__index),
        "core.scip_occurrences": persist_occurrences(env, t__scip__index),
    })
```

### Migration Order (Breaking Changes)

This is a **big-bang migration** since plugins and native Hamilton don't interoperate cleanly. Recommended order:

1. **Create all Hamilton native modules** (in parallel with existing plugins)
2. **Test each native module** against golden outputs
3. **Switch driver_factory to load native modules only**
4. **Delete plugin infrastructure** in one PR

### Lines of Code Impact

| Before | After | Change |
|--------|-------|--------|
| `plugin.py` (425) | 0 | -425 |
| `context.py` (582) | 0 | -582 |
| `context_base.py` (605) | 0 | -605 |
| `plugins/` (~3,000) | Hamilton native (~2,000) | -1,000 |
| `unified_registry.py` (461) | ~200 (simplified) | -261 |
| `registrations.py` (333) | 0 | -333 |
| Node wrappers (~200) | 0 | -200 |
| **Total** | | **~-3,400 lines** |

Plus significant simplification in:
- `executor.py` (no more dual paths)
- `hashing.py` (single path)
- `session.py` (simplified caching)

### Benefits of 100% Native Hamilton

1. **Single execution model** - Everything is a Hamilton node
2. **Explicit dependencies** - In function signatures, not static declarations
3. **Pure compute separation** - Business logic in pure functions, I/O at boundaries
4. **Uniform caching** - Hamilton handles it all
5. **Better parallelization** - Hamilton can optimize the DAG
6. **Simpler testing** - Test pure functions directly
7. **No context confusion** - Just `BuildEnv`
8. **Debuggable** - Hamilton's execution model is transparent

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Big-bang migration | Create native modules in parallel, test thoroughly before switch |
| Async plugin logic | Hamilton supports async nodes via `@tag(async_func=True)` |
| Plugin options system | Migrate to Hamilton `@config.when()` pattern |
| External tool calls | Still work via `env.providers` |
| Large refactor | Can be done incrementally per domain (ingestion → graphs → analytics) |

---

## Part II: Original Analysis (Unified Context Approach)

The build directory has evolved organically, resulting in **parallel systems** for similar concerns. With Hamilton becoming the single source of truth (Phase 5), we can consolidate these into a unified, best-in-class architecture.

**Key Finding**: There are effectively **two execution models** running in parallel:
1. **Plugin Model**: `TargetPlugin.execute()` → `TargetExecutionContext` → manual writes
2. **Native Hamilton Model**: Hamilton nodes → `BuildEnv` → `NativeTargetExecutor` → materializers

This duplication cascades into context types, skip-check logic, manifest handling, and hash computation.

---

## Table of Contents

1. [Duplication Categories](#duplication-categories)
2. [Detailed Analysis](#detailed-analysis)
3. [Consolidation Recommendations](#consolidation-recommendations)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Architectural Principles](#architectural-principles)

---

## Duplication Categories

| Category | Files Involved | Lines of Duplication | Impact |
|----------|---------------|---------------------|--------|
| **Execution Models** | `plugin.py`, `executor.py`, `native/executor.py`, `native/runner.py` | ~800 | High |
| **Context Types** | `context.py`, `context_base.py`, `native/materializer.py`, `env.py` | ~500 | High |
| **Target Registry** | `registry.py`, `unified_registry.py`, `registrations.py` | ~600 | Medium |
| **Hash/Skip Logic** | `hashing.py`, `session.py`, `manifest_hook.py`, `native/runner.py` | ~300 | Medium |
| **State Computation** | `state.py`, `state_types.py`, `state_computer.py` | ~200 | Low (mostly resolved) |

---

## Detailed Analysis

### 1. Dual Execution Model (HIGH PRIORITY)

**Current State:**

```
Plugin Execution Path:
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ TargetPlugin │───>│ TargetExecutionContext │───>│ ctx.write_table │
│   .execute() │    │   (context.py)        │    │   (manual)      │
└──────────────┘    └──────────────────────┘    └─────────────────┘

Native Hamilton Path:
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ t__<target>  │───>│ BuildEnv + NativeTarget│───>│ materialize_    │
│ Hamilton node│    │ Executor (native/)     │    │ table() (auto)  │
└──────────────┘    └──────────────────────┘    └─────────────────┘
```

**Specific Duplication:**

| Concern | Plugin Path | Native Path |
|---------|-------------|-------------|
| Input hash computation | `compute_input_hash()` in context | `compute_target_input_hash_with_deps()` in manifest_hook |
| Skip check | Embedded in executor | `should_skip_native_target()` + `NativeTargetExecutor.should_skip()` |
| Manifest persistence | `ManifestHook` in Hamilton | `save_manifest()` in runner.py |
| Row counting | `WriteRecord.rows` tracking | `row_counts` in `NativeRunInfo` |
| Error handling | `TargetResult.failed()` | `TargetRunRecord.failed()` |

**Code Example - Skip Check Duplication:**

```python
# In native/runner.py
def should_skip_native_target(env, target, input_hash):
    if target.name in env.force_targets:
        return False
    return should_skip(SkipCheckRequest(...))

# In native/executor.py (NativeTargetExecutor)
def should_skip(self) -> bool:
    return should_skip_native_target(self.env, self.target, self.input_hash)

# In manifest_hook.py (general)
def should_skip(request: SkipCheckRequest) -> bool:
    # Core skip logic...
```

Three layers to check "should this skip?" - but they all ultimately call the same function.

---

### 2. Context Type Proliferation (HIGH PRIORITY)

**Current State:**

```
context_base.py:
├── ContextPropertiesProtocol (interface)
├── BuildContext (base class)
└── ExecutionContext (extends BuildContext)

context.py:
├── ContextResources (dataclass)
├── WriteRecord (dataclass)  
└── TargetExecutionContext (extends ExecutionContext)

native/materializer.py:
└── MaterializationContext (DEPRECATED - duplicates BuildContext)

hamilton/env.py:
└── BuildEnv (parallel context for Hamilton)

hamilton/executor.py:
└── _RunContext (internal execution context)
```

**The Problem:**

- `BuildContext` and `MaterializationContext` are nearly identical
- `BuildEnv` duplicates most of what `BuildContext` provides
- `TargetExecutionContext` adds plugin-specific concerns but could be a composition
- `_RunContext` is yet another execution context

**Field Overlap:**

| Field | BuildContext | MaterializationContext | BuildEnv | TargetExecutionContext |
|-------|--------------|----------------------|----------|------------------------|
| gateway | ✓ | ✓ | ✓ | ✓ (via resources) |
| snapshot | ✓ | ✓ | ✓ | ✓ |
| paths | ✓ | - | ✓ | ✓ |
| validate | ✓ | ✓ | - | - |
| session | ✓ | - | - | - |
| target | - | - | - | ✓ |
| parameters | - | - | - | ✓ |

---

### 3. Target Registry Fragmentation (MEDIUM PRIORITY)

**Current State:**

Three interconnected systems define "what targets exist":

```
registry.py (Static):
├── MODULES_TARGET = OutputTarget(...)
├── AST_TARGET = OutputTarget(...)
├── ... (45 constants)
├── ALL_TARGETS = (MODULES_TARGET, ...)
└── get_target_graph() → builds from Hamilton

unified_registry.py (Dynamic):
├── UnifiedRegistry
├── TargetRegistration (target + plugin + native_module)
└── get_unified_registry() → singleton

registrations.py (Bridge):
└── register_all_targets(registry) → uses static constants
```

**The Problem:**

- Static constants in `registry.py` duplicate information that should come from `UnifiedRegistry`
- `registrations.py` imports ALL static constants just to register them
- Dependency declarations exist in BOTH `OutputTarget.dependencies` AND Hamilton DAG
- There are ~600 lines of static target definitions that could be derived

---

### 4. Hash Computation Fragmentation (MEDIUM PRIORITY)

**Current State:**

```python
# hashing.py - Base function
def compute_input_hash(target, snapshot, gateway, options_hash, manifests) -> str:
    ...

# session.py - Cached wrapper
def get_input_hash(self, target, options_hash) -> str:
    hash_value = compute_input_hash(...)
    self._hash_cache[cache_key] = hash_value
    ...

# manifest_hook.py - Different signature, similar purpose
def compute_target_input_hash_with_deps(target, env, graph) -> str:
    # Also computes hash but with different inputs
    ...

# native/runner.py - Uses NativeTargetExecutor which computes hash
# native/executor.py - Stores input_hash as attribute
```

**The Problem:**

- Same concept (input hash) computed via 3+ code paths
- Different function signatures make it unclear which to use
- Native vs plugin paths use different hash computation flows

---

### 5. Manifest/Record Type Duplication (MEDIUM PRIORITY)

**Current State:**

```python
# manifest.py
@dataclass(frozen=True)
class OutputManifest:
    target: str
    input_hash: str
    options_hash: str | None
    ...

@dataclass(frozen=True)
class BuildRunRecord:
    run_id: str
    computed_targets: tuple[str, ...]
    ...

# manifest_hook.py
@dataclass(frozen=True)
class TargetRunRecord:
    target: str
    status: Literal["succeeded", "failed", "skipped"]
    input_hash: str
    ...

# native/runner.py
@dataclass(frozen=True)
class NativeRunInfo:
    input_hash: str
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int] | None
```

**The Problem:**

- `TargetRunRecord` and `NativeRunInfo` capture similar data
- `OutputManifest` vs `TargetRunRecord` have overlapping concerns
- Multiple "record" types for what is conceptually one thing: "what happened when we ran a target"

---

## Consolidation Recommendations

### Recommendation 1: Unified Execution Context (PR-87)

**Goal:** Single context type that works for ALL execution paths.

```python
# Proposed: Unified HamiltonExecutionContext

@dataclass
class HamiltonExecutionContext:
    """Single execution context for all Hamilton nodes.
    
    Whether the node wraps a plugin or is native Hamilton,
    it receives the same context.
    """
    # Core (from BuildContext)
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    session: BuildSession
    
    # Execution (from BuildEnv)
    force_targets: frozenset[str]
    manifest_index: dict[str, OutputManifest]
    
    # Target-specific (from TargetExecutionContext)  
    target: OutputTarget
    parameters: TargetParameters
    
    # Hamilton-specific
    runtime: HamiltonRuntime
    run_id: str
```

**Migration Path:**
1. Create `HamiltonExecutionContext` with all fields
2. Add adapters: `as_build_context()`, `as_target_context()`
3. Update Hamilton nodes to accept unified context
4. Deprecate `MaterializationContext`, `BuildEnv`, `TargetExecutionContext`

**Lines Saved:** ~400

---

### Recommendation 2: Plugin-as-Hamilton-Node Pattern (PR-88)

**Goal:** Plugins become Hamilton nodes, eliminating the dual execution model.

**Current (Plugin):**
```python
class AstExtractPlugin(TargetPlugin):
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # Manual implementation
        ...
```

**Proposed (Hamilton Node):**
```python
# Generated or hand-written Hamilton module
@tag(target="ast", node_type="materialize")
def t__ast(
    ctx: HamiltonExecutionContext,
    t__modules: TargetRunRecord,  # Dependency injection via Hamilton
) -> TargetRunRecord:
    """Execute AST extraction plugin as Hamilton node."""
    plugin = get_plugin_for_target("ast")
    return await run_plugin(plugin, ctx)
```

**Benefits:**
- Single execution model (Hamilton)
- Dependencies are explicit in function signature
- Skip logic handled uniformly by Hamilton hooks
- Manifest persistence via single hook

**Lines Saved:** ~300 (executor.py, native/executor.py consolidation)

---

### Recommendation 3: Target Definition from Registry Only (PR-89)

**Goal:** Remove static `*_TARGET` constants; derive everything from `UnifiedRegistry`.

**Current:**
```python
# registry.py - 45 constants
MODULES_TARGET = OutputTarget(name="modules", ...)
AST_TARGET = OutputTarget(name="ast", dependencies=("modules",), ...)
...

# registrations.py - imports all constants
from codeintel.build.registry import MODULES_TARGET, AST_TARGET, ...

def register_all_targets(registry):
    registry.register(MODULES_TARGET, plugin=ModulesPlugin)
    ...
```

**Proposed:**
```python
# unified_registrations.py - Declarative registration
TARGETS: list[TargetDefinition] = [
    TargetDefinition(
        name="modules",
        module="ingestion",
        contract=OutputContract(tables=(...)),
        plugin="codeintel.build.plugins.ingestion:ModulesPlugin",
    ),
    TargetDefinition(
        name="ast", 
        module="ingestion",
        contract=OutputContract(tables=(...)),
        plugin="codeintel.build.plugins.ingestion:AstExtractPlugin",
        # NOTE: dependencies derived from Hamilton, not declared here
    ),
    ...
]

def register_all_targets(registry: UnifiedRegistry) -> None:
    for defn in TARGETS:
        registry.register_from_definition(defn)
```

**Benefits:**
- No more parallel static constants
- Dependencies come from Hamilton (single source of truth)
- Easier to add new targets (one place)
- Validation at registration time

**Lines Saved:** ~500 (remove static constants from registry.py)

---

### Recommendation 4: Unified Skip/Hash Infrastructure (PR-90)

**Goal:** Single skip-check function, single hash function, single caching layer.

**Proposed Architecture:**

```python
# skip_check.py (NEW)
@dataclass(frozen=True)
class SkipDecision:
    """Unified skip decision for any target."""
    can_skip: bool
    reason: Literal["forced", "no_manifest", "hash_changed", "up_to_date"]
    prior_hash: str | None = None
    current_hash: str | None = None

def check_skip(
    target: OutputTarget,
    session: BuildSession,
    force_targets: frozenset[str],
) -> SkipDecision:
    """Single function for all skip-check logic."""
    if target.name in force_targets:
        return SkipDecision(can_skip=False, reason="forced")
    
    current_hash = session.get_input_hash(target)
    manifest = session.get_manifest(target.name)
    
    if manifest is None:
        return SkipDecision(can_skip=False, reason="no_manifest", current_hash=current_hash)
    
    if manifest.input_hash != current_hash:
        return SkipDecision(
            can_skip=False, 
            reason="hash_changed",
            prior_hash=manifest.input_hash,
            current_hash=current_hash,
        )
    
    return SkipDecision(can_skip=True, reason="up_to_date", current_hash=current_hash)
```

**Benefits:**
- One function to understand
- Consistent behavior across plugin/native
- Clear decision data structure
- Easier to test and debug

**Lines Saved:** ~200 (consolidate skip logic across files)

---

### Recommendation 5: Unified Run Record (PR-91)

**Goal:** Single record type for "what happened when a target ran."

**Proposed:**

```python
# run_record.py (consolidate from manifest.py, manifest_hook.py, native/runner.py)

@dataclass(frozen=True)
class TargetRunRecord:
    """Unified record of a target execution.
    
    Used by Hamilton hooks for both plugin and native targets.
    This is the canonical type for "what happened when we ran X."
    """
    target: str
    status: Literal["succeeded", "failed", "skipped"]
    input_hash: str
    options_hash: str | None
    started_at: datetime
    duration_ms: float
    row_counts: dict[str, int]
    artifact_paths: dict[str, str]
    error_message: str | None = None
    error_type: str | None = None
    
    # Factory methods
    @classmethod
    def succeeded(cls, ...) -> Self: ...
    
    @classmethod
    def failed(cls, error: Exception, ...) -> Self: ...
    
    @classmethod
    def skipped(cls, ...) -> Self: ...
```

**Benefits:**
- One type to understand and document
- Consistent serialization/persistence
- Clear factory methods for common cases
- Aligns with Hamilton's node return pattern

**Lines Saved:** ~150 (consolidate NativeRunInfo, parts of OutputManifest)

---

## Implementation Roadmap

### Phase A: Foundation (PR-87 through PR-89)

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-87 | Unified execution context | None | Medium |
| PR-88 | Plugin-as-Hamilton-node pattern | PR-87 | High |
| PR-89 | Target definition from registry only | PR-88 | Medium |

### Phase B: Infrastructure (PR-90 through PR-91)

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-90 | Unified skip/hash infrastructure | PR-87 | Low |
| PR-91 | Unified run record | PR-87, PR-90 | Low |

### Phase C: Cleanup

| PR | Description | Dependencies | Risk |
|----|-------------|--------------|------|
| PR-92 | Delete `MaterializationContext` | PR-87 | Low |
| PR-93 | Delete `native/executor.py` (use unified) | PR-88 | Low |
| PR-94 | Delete static `*_TARGET` constants | PR-89 | Medium |

---

## Architectural Principles

### Principle 1: Hamilton is the Execution Engine

```
BEFORE: Plugin OR Hamilton OR Native
AFTER:  Everything is a Hamilton node
```

All targets—whether they wrap legacy plugins or are pure Hamilton—execute through Hamilton's `Driver.execute()`. This gives us:
- Unified dependency resolution
- Uniform caching/skip semantics
- Single observability path
- Consistent error handling

### Principle 2: Context Composition over Inheritance

```
BEFORE: ExecutionContext extends BuildContext extends ...
AFTER:  HamiltonExecutionContext composes components
```

Instead of deep inheritance hierarchies, use composition:
- `session: BuildSession` for caching
- `resources: ContextResources` for DI
- `target: OutputTarget` for target metadata

### Principle 3: Derive, Don't Declare

```
BEFORE: OutputTarget(dependencies=("modules", "ast"))
AFTER:  Dependencies derived from Hamilton DAG
```

Static declarations drift. Let Hamilton's FunctionGraph be the source of truth:
- Target dependencies from node signatures
- Output tables from `d__` node tags
- Artifacts from `a__` node tags

### Principle 4: Single Path for Common Concerns

```
BEFORE: 3 ways to compute input hash
AFTER:  1 function, 1 cache, 1 behavior
```

Each concern should have exactly one code path:
- Skip check: `check_skip()`
- Input hash: `session.get_input_hash()`
- Manifest save: `ManifestHook`
- Record creation: `TargetRunRecord.succeeded()`

---

## Metrics Summary

| Metric | Current | After Consolidation | Improvement |
|--------|---------|---------------------|-------------|
| Context types | 6 | 1-2 | -70% |
| Execution paths | 2 (plugin + native) | 1 (Hamilton) | -50% |
| Skip-check implementations | 3 | 1 | -67% |
| Hash computation paths | 3+ | 1 | -67% |
| Target definition locations | 3 | 1 | -67% |
| Total build/ lines | ~12,000 | ~9,500 | -21% |

---

## Verification Checklist

After consolidation, verify:

- [ ] All tests pass (no new xfails)
- [ ] `codeintel build validate` returns zero issues
- [ ] BuildSpec compile is deterministic
- [ ] Skip behavior is identical for plugin and native targets
- [ ] Manifest persistence works uniformly
- [ ] Hash computation is consistent
- [ ] No import cycles introduced
- [ ] Performance is not degraded (measure state computation time)

---

**Document Version**: 1.0
**Created**: 2025-12-15
**Author**: CodeIntel Build Team

