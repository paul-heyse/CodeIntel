# CLI Unified Context - High-Level Implementation Plan

> **Purpose**: Identify the practical migration boundary for implementing the unified context and resolution architecture, distinguishing between what can be done now versus what requires separate design work.

---

## Executive Summary

After analyzing the current codebase against the target architecture, I find **no irreconcilable architectural differences**. The existing code exhibits duplication and drift, but all variations are reconcilable through:
- **Superset unification** (RuntimeCliOptions variations → CommonOptions with all fields)
- **Parameterization** (build_runtime_from_cli variations → single function with options)
- **Adapter patterns** (old handler signatures → ExecutionContext wrappers)

The "cliff" is not architectural complexity—it's **migration volume**. The infrastructure can be built cleanly; handler migration is mechanical but numerous (~40 handlers across 10 files).

### Recommended Boundary

| Phase | Scope | This Plan? | Rationale |
|-------|-------|------------|-----------|
| **1: Infrastructure** | Create resolution/, options/, enhance ExecutionContext | ✅ YES | New code, no breaking changes |
| **2: Adapter Integration** | Wire CycloptsAdapter to use new infrastructure | ✅ YES | Internal change, backward compatible |
| **3: Proof of Concept** | Migrate 1-2 handler groups to validate pattern | ✅ YES | Validates architecture works |
| **4: Full Handler Migration** | Migrate remaining ~35 handlers | ❌ SEPARATE | Volume work, can be incremental |
| **5: Cleanup** | Remove deprecated code | ❌ SEPARATE | Depends on Phase 4 completion |

---

## Analysis: Where Complexity Escalates

### Layer 1: Resolution Package (LOW COMPLEXITY)

**Current State**: 7 implementations of `build_runtime_from_cli` across 8 files

**Differences Analysis**:

| File | Variations | Reconcilable? |
|------|------------|---------------|
| `cyclopts_common.py` | Has `allow_fallback` param | ✅ Add as optional param |
| `common_handlers.py` | No fallback param | ✅ Default behavior |
| `datasets_handlers.py` | Uses full RuntimeCliOptions | ✅ Superset handles |
| `subsystem_handlers.py` | Uses minimal RuntimeCliOptions (project_root only) | ✅ Superset handles |
| `ide_handlers.py` | Uses minimal RuntimeCliOptions | ✅ Superset handles |
| `build_handlers.py` | Returns ProjectRuntime | ✅ Same return type |
| `ops_handlers.py` | Simple wrapper | ✅ Can use resolver |

**Verdict**: All variations can be unified into single `RuntimeResolver` with optional parameters.

**Action**: Create `resolution/` package - **INCLUDE IN THIS PLAN**

---

### Layer 2: Options Consolidation (LOW COMPLEXITY)

**Current State**: `RuntimeCLI` and `OutputFormatCLI` in `cyclopts_common.py`

**Differences Analysis**:

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| Runtime options | `RuntimeCLI` (9 fields) | `CommonOptions` runtime section | Field rename only |
| Output options | `OutputFormatCLI` (2 fields) | `CommonOptions` output section | Combine |
| Backend options | `BackendFlags` in cli_types | `CommonOptions` backend section | Add fields |

**Existing Issue Found**: `docs_handlers.py` defines its own `OutputFormat` enum that shadows `cli_types.OutputFormat`. This is a bug regardless of this migration.

**Verdict**: Straightforward combination. `CommonOptions` is a superset.

**Action**: Create `options/CommonOptions` - **INCLUDE IN THIS PLAN**

---

### Layer 3: ExecutionContext Enhancement (LOW COMPLEXITY)

**Current State**: `ExecutionContext` in `execution/context.py` with basic fields

**Required Changes**:
1. Add `ContextMetadata` dataclass
2. Add `require_runtime()` method
3. Add `require_gateway()` method
4. Add convenience properties

**Backward Compatibility**: 
- Existing code continues to work
- New methods are additive
- No breaking changes

**Verdict**: Purely additive enhancement.

**Action**: Enhance ExecutionContext - **INCLUDE IN THIS PLAN**

---

### Layer 4: Handler Migration (HIGH VOLUME, BUT MECHANICAL)

**Current State**: ~40 handlers with signatures like:
```python
def handler(options: SomeOptions, ctx: SomeContext) -> None:
```

**Target State**:
```python
def handler(ctx: ExecutionContext) -> CliResult[T]:
```

**Migration Pattern** (same for all handlers):
```python
# Before
def build_run_handler(options: BuildRunOptions, ctx: BuildRunContext) -> None:
    runtime = build_runtime_from_cli(ctx.runtime_options)
    # ... logic ...

# After  
def build_run_handler(ctx: ExecutionContext) -> CliResult[BuildRunResult]:
    runtime = ctx.require_runtime()
    targets = ctx.get_param("targets", [])
    # ... logic ...
    return CliResult.success(result)
```

**The "Cliff" Assessment**:

This is NOT an architectural cliff—each migration is:
1. Change signature to accept `ExecutionContext`
2. Replace `build_runtime_from_cli()` with `ctx.require_runtime()`
3. Replace option access with `ctx.get_param()`
4. Return `CliResult` instead of `None`

The complexity is **volume**, not **architecture**.

**Verdict**: Mechanical migrations, can be done incrementally.

**Action**: Migrate 1-2 handler groups as proof of concept - **INCLUDE IN THIS PLAN**
**Action**: Full migration - **SEPARATE PLAN** (incremental, per-group)

---

### Layer 5: Command Simplification (DEPENDS ON HANDLERS)

**Current State**: Commands have ~30 lines of boilerplate
```python
def __call__(self) -> None:
    runtime_opts, verbose, output_format = make_handler_context(...)
    options = BuildRunOptions(...)
    ctx_opts = BuildRunContext(...)
    run_handler(build_run_handler, options, ctx_opts)
```

**Target State**: Commands have ~5 lines
```python
def __call__(self) -> None:
    execute_command("build.run", self)
```

**Dependency**: Commands can only be simplified AFTER their handlers are migrated.

**Verdict**: Commands can be simplified incrementally as handlers are migrated.

**Action**: Simplify commands alongside handler migration - **SEPARATE PLAN**

---

## Identified Non-Issues (Initially Appeared Problematic)

### Non-Issue 1: Different RuntimeCliOptions Field Sets

**Appearance**: Some handlers use 6-field RuntimeCliOptions, others use 1-field

**Reality**: This is a symptom of copy-paste, not an architectural difference. A superset `CommonOptions` with all fields works for all cases. Handlers only access the fields they need via `ctx.get_param()`.

### Non-Issue 2: Handler Return Types

**Appearance**: Some handlers return `None`, others return values

**Reality**: All can be unified to return `CliResult[T]`. Handlers that don't have results return `CliResult[None]` or `CliResult.success(None)`.

### Non-Issue 3: Context Type Variations

**Appearance**: `BuildRunContext`, `ProjectContext`, etc. have different fields

**Reality**: All context information can flow through `ExecutionContext.params`. The typed context classes were just organizing data that params can hold.

### Non-Issue 4: docs_handlers.py OutputFormat Shadow

**Appearance**: Defines own `OutputFormat` enum

**Reality**: This is a bug to fix regardless. Should import from `cli_types`.

---

## Implementation Plan: This Phase

### Phase 1: Resolution Layer (~2-3 hours)

**Goal**: Single source of truth for runtime resolution

**Files to Create**:
```
resolution/
├── __init__.py      # Package exports
├── types.py         # ResolvedRuntime, ResolutionError
├── runtime.py       # RuntimeResolver
└── gateway.py       # GatewayManager
```

**Key Implementation**:
```python
# resolution/types.py
@dataclass(frozen=True)
class ResolvedRuntime:
    root: Path
    project: ProjectConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    config: CodeIntelConfig
    serving: ServingConfig

class ResolutionError(Exception):
    """Raised when runtime cannot be resolved."""

# resolution/runtime.py
class RuntimeResolver:
    def resolve(self, ctx: ExecutionContext) -> ResolvedRuntime:
        # Consolidate ALL build_runtime_from_cli logic here
        ...
```

**Migration Strategy**: 
- New code, no changes to existing handlers
- Existing `build_runtime_from_cli` functions remain (deprecated)
- New handlers use `ctx.require_runtime()`

---

### Phase 2: Options Consolidation (~1-2 hours)

**Goal**: Single option bundle for CLI commands

**Files to Create**:
```
options/
├── __init__.py      # Package exports
└── common.py        # CommonOptions
```

**Key Implementation**:
```python
# options/common.py
@dataclass
class CommonOptions:
    # Runtime (from RuntimeCLI)
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    
    # Output (from OutputFormatCLI)
    output_format: OutputFormat = OutputFormat.TEXT
    json: bool = False
    
    # Execution
    verbose: int = 0
    dry_run: bool = False
    
    # Backend
    use_gpu: bool = False
```

**Migration Strategy**:
- New commands can use `CommonOptions`
- Existing commands continue using `RuntimeCLI` + `OutputFormatCLI`
- `cyclopts_common.py` can delegate to `CommonOptions` for backward compat

---

### Phase 3: ExecutionContext Enhancement (~1-2 hours)

**Goal**: Add lazy resolution capabilities

**Files to Modify**:
- `execution/context.py`

**Key Changes**:
```python
@dataclass
class ContextMetadata:
    config: CliConfig
    verbosity: int = 0
    output_format: OutputFormat = OutputFormat.TEXT
    dry_run: bool = False
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)

@dataclass
class ExecutionContext:
    # ... existing fields ...
    metadata: ContextMetadata  # NEW
    
    def require_runtime(self) -> ResolvedRuntime:
        """Lazy resolution."""
        if self.metadata._runtime is None:
            from codeintel.cli.resolution import resolve_runtime
            self.metadata._runtime = resolve_runtime(self)
        return self.metadata._runtime
    
    def require_gateway(self, *, read_only: bool = True) -> StorageGateway:
        """Lazy gateway opening."""
        ...
```

**Migration Strategy**:
- Additive changes only
- Existing ExecutionContext usage continues to work
- New handlers can use `require_*` methods

---

### Phase 4: Proof of Concept Migration (~2-3 hours)

**Goal**: Validate architecture with real handler migration

**Recommended Groups** (simplest first):
1. **storage_handlers.py** - Only 3 functions, simple logic
2. **ops_handlers.py** - 2 main functions, straightforward

**Migration Per Handler**:
1. Change signature: `def handler(ctx: ExecutionContext) -> CliResult[T]`
2. Replace runtime resolution: `runtime = ctx.require_runtime()`
3. Replace param access: `target = ctx.get_param("target")`
4. Return CliResult: `return CliResult.success(result)`
5. Update corresponding Cyclopts command to use `execute_command()`

**Success Criteria**:
- Migrated handlers work correctly
- Pattern is validated as repeatable
- No unexpected issues discovered

---

## What Remains for Separate Plan

### Handler Migration (Phase 4 Full)

**Scope**: ~35 remaining handlers across:
- `build_handlers.py` (~8 handlers)
- `docs_handlers.py` (~10 handlers)  
- `datasets_handlers.py` (~8 handlers)
- `common_handlers.py` (~6 handlers)
- `subsystem_handlers.py` (~4 handlers)
- `history_handlers.py` (~3 handlers)
- `ide_handlers.py` (~2 handlers)
- `graphs_handlers.py` (~3 handlers)

**Why Separate**:
- Volume work (35 handlers × ~30 min each = ~17 hours)
- Can be done incrementally per-group
- Each group can be its own PR
- Infrastructure must be proven first

### Command Simplification

**Scope**: ~18 `cyclopts_*.py` files

**Why Separate**:
- Depends on handler migration
- Can be done alongside handler migration
- Each command simplified after its handler is migrated

### Cleanup

**Scope**: Remove deprecated code

**Why Separate**:
- Must wait until all handlers migrated
- Includes:
  - Duplicate `RuntimeCliOptions` classes
  - Duplicate `build_runtime_from_cli` functions
  - Old context types (`BuildRunContext`, etc.)
  - `HandlerContext` in `handlers/base.py`

---

## Risk Assessment

### Low Risk (This Plan)
| Risk | Mitigation |
|------|------------|
| New code has bugs | Comprehensive tests for resolution layer |
| ExecutionContext changes break existing code | Changes are additive only |
| Options consolidation misses edge cases | Superset design handles all cases |

### Medium Risk (Future Plan)
| Risk | Mitigation |
|------|------------|
| Handler migration introduces bugs | Migrate one group at a time, full testing |
| Command simplification breaks CLI | Keep old `__call__` pattern available |
| Volume of changes causes merge conflicts | Small PRs per handler group |

### No Architectural Risks Identified
- All variations are reconcilable
- No fundamental redesigns required
- Adapter patterns bridge old → new

---

## Summary: This Plan vs Future Plan

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           THIS IMPLEMENTATION PLAN                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Phase 1: resolution/ package                                              │
│           └── RuntimeResolver, GatewayManager, ResolvedRuntime             │
│                                                                            │
│  Phase 2: options/ package                                                 │
│           └── CommonOptions                                                │
│                                                                            │
│  Phase 3: ExecutionContext enhancement                                     │
│           └── ContextMetadata, require_runtime(), require_gateway()        │
│                                                                            │
│  Phase 4: Proof of concept (storage_handlers, ops_handlers)                │
│           └── 5 handlers migrated, pattern validated                       │
│                                                                            │
│  BOUNDARY LINE ─────────────────────────────────────────────────────────── │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                              FUTURE PLAN                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Handler Migration: 35 remaining handlers (incremental, per-group)         │
│                                                                            │
│  Command Simplification: 18 cyclopts_*.py files (parallel with handlers)   │
│                                                                            │
│  Cleanup: Remove deprecated code (after full migration)                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Resolution | 2-3 hours | None |
| Phase 2: Options | 1-2 hours | None |
| Phase 3: ExecutionContext | 1-2 hours | Phase 1 |
| Phase 4: Proof of Concept | 2-3 hours | Phases 1-3 |
| **Total This Plan** | **6-10 hours** | - |
| Future: Full Migration | ~17 hours | This plan complete |
| Future: Cleanup | ~2 hours | Full migration complete |

---

*Document Version: 1.0*
*Created: 2025-01-09*

