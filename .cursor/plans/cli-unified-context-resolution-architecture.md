# CLI Unified Context and Resolution Architecture

> **Purpose**: This document specifies the target architecture for consolidating handler contexts, runtime resolution, and CLI option bundles into a unified, layered system. It serves as the authoritative reference for implementation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Problems](#current-state-problems)
3. [Design Principles](#design-principles)
4. [Architecture Overview](#architecture-overview)
5. [Layer Specifications](#layer-specifications)
6. [Core Abstractions](#core-abstractions)
7. [Interface Contracts](#interface-contracts)
8. [Data Flow](#data-flow)
9. [Migration Boundaries](#migration-boundaries)
10. [File Inventory](#file-inventory)
11. [Acceptance Criteria](#acceptance-criteria)

---

## Executive Summary

### Problem

The CLI codebase has accumulated significant duplication and drift across three interconnected areas:

| Area | Duplication | Impact |
|------|-------------|--------|
| Handler Contexts | 4 `RuntimeCliOptions` classes, 3+ context types | Inconsistent behavior, maintenance burden |
| Runtime Resolution | ~450 lines across 8 files | Drift in resolution logic, bugs |
| Option Bundles | Manual wiring in every command | Boilerplate, inconsistent defaults |

### Solution

Consolidate into a **four-layer architecture**:

```
┌─────────────────────────────────────────────────┐
│ CLI Layer: Commands + Options                   │
├─────────────────────────────────────────────────┤
│ Execution Layer: Context + Executor + Adapter   │
├─────────────────────────────────────────────────┤
│ Resolution Layer: Runtime + Gateway             │
├─────────────────────────────────────────────────┤
│ Handler Layer: Business Logic                   │
└─────────────────────────────────────────────────┘
```

### Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| `RuntimeCliOptions` definitions | 4 | 1 |
| `build_runtime_from_cli` implementations | 7 | 1 |
| Lines of resolution code | ~450 | ~150 |
| Per-command boilerplate | ~30 lines | ~5 lines |
| Handler context types | 5+ | 1 |

---

## Current State Problems

### Problem 1: Multiple RuntimeCliOptions Definitions

```python
# common_handlers.py - Full definition
RuntimeCliOptions = RuntimeOptions  # Alias to cli_types

# datasets_handlers.py - Own definition
@dataclass
class RuntimeCliOptions:
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None

# subsystem_handlers.py - Minimal definition
@dataclass
class RuntimeCliOptions:
    project_root: Path | None = None

# ide_handlers.py - Minimal definition  
@dataclass
class RuntimeCliOptions:
    project_root: Path | None = None
```

**Impact**: Different handlers have different capabilities based on which definition they use.

### Problem 2: Duplicated Resolution Logic

```python
# cyclopts_common.py:314-402
def build_runtime_from_cli(options, *, allow_fallback=True) -> ProjectRuntime:
    # 90 lines of resolution logic

# common_handlers.py:330-421
def build_runtime_from_cli(options) -> ProjectRuntime:
    # 90 lines of near-identical resolution logic

# datasets_handlers.py:347-400
def build_runtime_from_cli(options) -> ProjectRuntime:
    # 50 lines of similar logic

# ... and 4 more implementations
```

**Impact**: Bug fixes must be applied to 7 locations. Behavior drift is inevitable.

### Problem 3: Multiple Handler Context Types

```python
# handlers/base.py
@dataclass
class HandlerContext:
    config: CliConfig
    execution: ExecutionContext
    project_root: Path | None
    verbosity: int

# build_handlers.py
@dataclass
class BuildRunContext:
    runtime_options: RuntimeCliOptions
    verbose: int
    output_format: OutputFormat

# common_handlers.py
@dataclass
class ProjectContext:
    runtime: ProjectRuntime
    gateway: StorageGateway
    config: CodeIntelConfig
    # ... more fields
```

**Impact**: Each handler defines its own context, leading to inconsistent capabilities.

### Problem 4: Per-Command Boilerplate

Every Cyclopts command repeats:

```python
def __call__(self) -> None:
    # 1. Extract options
    runtime_opts, verbose, output_format = make_handler_context(
        self.runtime or RuntimeCLI(),
        self.output or OutputFormatCLI(),
        default_output=OutputFormat.TEXT,
    )
    # 2. Build handler-specific options
    options = SomeOptions(targets=self.targets, ...)
    # 3. Build handler-specific context
    ctx_opts = SomeContext(runtime_options=runtime_opts, ...)
    # 4. Call handler
    run_handler(handler, options, ctx_opts)
```

**Impact**: ~30 lines of boilerplate per command, inconsistent patterns.

---

## Design Principles

### 1. Single Source of Truth

Each concept is defined exactly once:

| Concept | Single Location |
|---------|-----------------|
| CLI option structure | `options/common.py::CommonOptions` |
| Runtime resolution | `resolution/runtime.py::resolve_runtime()` |
| Gateway management | `resolution/gateway.py::GatewayManager` |
| Handler context | `execution/context.py::ExecutionContext` |

### 2. Lazy Resolution

Resources are resolved only when needed:

```python
# Resolution happens on first access, not at context creation
runtime = ctx.require_runtime()  # Resolves if not already resolved
gateway = ctx.require_gateway()  # Opens if not already open
```

### 3. Composition Over Inheritance

Contexts compose capabilities rather than inheriting:

```python
@dataclass
class ExecutionContext:
    operation_id: str
    params: dict[str, Any]
    metadata: ContextMetadata  # Composed, not inherited
```

### 4. Explicit Dependencies

No global state. Dependencies flow through context:

```python
# Good: Dependencies from context
def handler(ctx: ExecutionContext) -> CliResult:
    runtime = ctx.require_runtime()

# Bad: Global resolution
def handler(options) -> CliResult:
    runtime = get_global_runtime()  # Hidden dependency
```

### 5. Type-Safe Contracts

All boundaries are typed:

```python
def resolve_runtime(ctx: ExecutionContext) -> ResolvedRuntime:
    """Returns ResolvedRuntime or raises ResolutionError."""
    ...
```

---

## Architecture Overview

### Layer Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLI LAYER                                        │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  CommonOptions  │    │  Command Class  │    │  execute_cmd()  │          │
│  │  (dataclass)    │───▶│  (dataclass)    │───▶│  (adapter)      │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                         │                    │
│  Responsibility: Define CLI parameters, invoke adapter  │                    │
└─────────────────────────────────────────────────────────┼────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                      ExecutionContext                                │     │
│  │  ┌──────────────┬────────────────┬──────────────────────────────┐   │     │
│  │  │ operation_id │ params         │ metadata: ContextMetadata    │   │     │
│  │  └──────────────┴────────────────┴──────────────────────────────┘   │     │
│  │                                                                      │     │
│  │  Methods:                                                            │     │
│  │    require_runtime() -> ResolvedRuntime                              │     │
│  │    require_gateway(read_only) -> StorageGateway                      │     │
│  │    get_param(key, default) -> Any                                    │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │OperationExecutor│───▶│ MiddlewareStack │───▶│  ProgressTracker│          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                              │
│  Responsibility: Orchestrate execution, apply cross-cutting concerns         │
└──────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          RESOLUTION LAYER                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ RuntimeResolver                                                      │     │
│  │   resolve(ctx) -> ResolvedRuntime                                    │     │
│  │   - Try project file discovery                                       │     │
│  │   - Fall back to explicit params                                     │     │
│  │   - Cache result in ctx.metadata                                     │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ GatewayManager                                                       │     │
│  │   open(ctx, read_only) -> StorageGateway                             │     │
│  │   close(ctx) -> None                                                 │     │
│  │   - Uses resolved runtime for db_path                                │     │
│  │   - Manages lifecycle                                                │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ ResolvedRuntime (frozen dataclass)                                   │     │
│  │   project: ProjectConfig                                             │     │
│  │   snapshot: SnapshotRef                                              │     │
│  │   paths: BuildPaths                                                  │     │
│  │   config: CodeIntelConfig                                            │     │
│  │   serving: ServingConfig                                             │     │
│  │   root: Path                                                         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Responsibility: Convert params to resolved resources                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           HANDLER LAYER                                       │
│                                                                              │
│  @operation("build.run", category=OperationCategory.BUILD)                   │
│  def build_run_handler(ctx: ExecutionContext) -> CliResult[BuildResult]:     │
│      runtime = ctx.require_runtime()                                         │
│      gateway = ctx.require_gateway(read_only=False)                          │
│      targets = ctx.get_param("targets", [])                                  │
│      # ... pure business logic ...                                           │
│                                                                              │
│  Responsibility: Pure business logic, no resolution or context building      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Specifications

### CLI Layer

**Location**: `src/codeintel/cli/options/`, `src/codeintel/cli/commands/`

**Responsibilities**:
- Define CLI parameters via dataclasses
- Provide type-safe option bundles
- Invoke execution adapter

**Key Components**:

| Component | File | Purpose |
|-----------|------|---------|
| `CommonOptions` | `options/common.py` | Unified option bundle |
| `RuntimeOptions` | `options/runtime.py` | Runtime-specific options |
| `OutputOptions` | `options/output.py` | Output format options |
| `BackendOptions` | `options/backend.py` | GPU/backend options |
| `execute_command()` | `commands/_base.py` | Adapter entry point |

**Boundary Rules**:
- Commands MUST NOT import from handler layer directly
- Commands MUST NOT perform resolution
- Commands MUST delegate to `execute_command()`

### Execution Layer

**Location**: `src/codeintel/cli/execution/`

**Responsibilities**:
- Build `ExecutionContext` from command params
- Orchestrate middleware application
- Manage operation lifecycle

**Key Components**:

| Component | File | Purpose |
|-----------|------|---------|
| `ExecutionContext` | `context.py` | Unified context |
| `ContextMetadata` | `context.py` | Lazy-resolved metadata |
| `OperationExecutor` | `executor.py` | Execution orchestration |
| `MiddlewareStack` | `middleware.py` | Cross-cutting concerns |
| `CycloptsAdapter` | `adapter.py` | Command-to-executor bridge |

**Boundary Rules**:
- Execution layer MUST NOT contain business logic
- Execution layer MAY invoke resolution layer
- `ExecutionContext` is the ONLY context type handlers receive

### Resolution Layer

**Location**: `src/codeintel/cli/resolution/`

**Responsibilities**:
- Resolve project/runtime from params
- Manage gateway lifecycle
- Cache resolved resources

**Key Components**:

| Component | File | Purpose |
|-----------|------|---------|
| `RuntimeResolver` | `runtime.py` | Project/runtime resolution |
| `GatewayManager` | `gateway.py` | Gateway lifecycle |
| `ResolvedRuntime` | `types.py` | Resolution result type |
| `ResolutionError` | `errors.py` | Resolution failures |

**Boundary Rules**:
- Resolution layer MUST be stateless (state in context)
- Resolution layer MUST NOT import from handler layer
- Resolution results MUST be immutable

### Handler Layer

**Location**: `src/codeintel/cli/handlers/`

**Responsibilities**:
- Implement business logic
- Return structured `CliResult`
- Use context for resource access

**Key Components**:

| Component | File | Purpose |
|-----------|------|---------|
| `@operation` decorator | `execution/adapter.py` | Handler registration |
| Handler functions | `handlers/*.py` | Business logic |

**Boundary Rules**:
- Handlers MUST accept `ExecutionContext` as sole context
- Handlers MUST NOT perform resolution (use `ctx.require_*()`)
- Handlers MUST return `CliResult[T]`

---

## Core Abstractions

### CommonOptions

```python
@dataclass
class CommonOptions:
    """Single option bundle for all CLI commands.
    
    This dataclass is designed to be embedded in Cyclopts command classes
    using `field(default_factory=CommonOptions, metadata={"parameter": Parameter(name="*")})`.
    
    Attributes
    ----------
    project_root
        Explicit project root directory. If None, auto-discovery is used.
    repo
        Repository slug (e.g., "org/repo"). Required if no project file.
    commit
        Commit SHA. Required if no project file.
    db_path
        Path to DuckDB database. Defaults to build/db/codeintel.duckdb.
    build_dir
        Build directory. Defaults to build/.
    repo_root
        Repository root. Defaults to current directory.
    document_output_dir
        Override for document output directory.
    output_format
        Output format (text or json).
    json
        Shorthand for --output-format json.
    verbose
        Verbosity level (0=warning, 1=info, 2+=debug).
    dry_run
        If True, show what would be done without doing it.
    use_gpu
        Enable GPU acceleration for graph operations.
    """
    # Runtime selection
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    
    # Output control
    output_format: OutputFormat = OutputFormat.TEXT
    json: bool = False
    
    # Execution control
    verbose: int = 0
    dry_run: bool = False
    
    # Backend control
    use_gpu: bool = False
    
    def to_params(self) -> dict[str, Any]:
        """Convert to parameter dictionary for ExecutionContext."""
        return {
            "project_root": self.project_root,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": self.db_path,
            "build_dir": self.build_dir,
            "repo_root": self.repo_root,
            "document_output_dir": self.document_output_dir,
            "output_format": self.resolve_output_format(),
            "verbose": self.verbose,
            "dry_run": self.dry_run,
            "use_gpu": self.use_gpu,
        }
    
    def resolve_output_format(self) -> OutputFormat:
        """Resolve output format with json flag precedence."""
        return OutputFormat.JSON if self.json else self.output_format
```

### ExecutionContext (Enhanced)

```python
@dataclass
class ContextMetadata:
    """Metadata and lazy-resolved resources for ExecutionContext.
    
    This class holds both static configuration and lazy-resolved
    resources. Resolution happens on first access via the parent
    ExecutionContext's `require_*` methods.
    
    Attributes
    ----------
    config
        CLI configuration loaded from sources.
    verbosity
        Verbosity level from CLI options.
    output_format
        Resolved output format.
    dry_run
        Whether this is a dry-run execution.
    _runtime
        Lazy-resolved runtime (None until resolved).
    _gateway
        Lazy-opened gateway (None until opened).
    """
    config: CliConfig
    verbosity: int = 0
    output_format: OutputFormat = OutputFormat.TEXT
    dry_run: bool = False
    
    # Private lazy-resolved fields
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)


@dataclass
class ExecutionContext:
    """Unified context for all CLI operations.
    
    This is the SINGLE context type that all handlers receive. It provides:
    - Operation identification (operation_id)
    - Parameter access (params, get_param)
    - Lazy resource resolution (require_runtime, require_gateway)
    - Configuration access (config, output_format, etc.)
    - Logging (logger)
    
    Handlers MUST NOT create their own context types. All context
    information flows through ExecutionContext.
    
    Attributes
    ----------
    operation_id
        Unique identifier for this operation (e.g., "build.run").
    params
        Raw parameters from CLI/caller.
    metadata
        Configuration and lazy-resolved resources.
    started_at
        When execution started (for timing).
    
    Examples
    --------
    >>> ctx = ExecutionContext.for_sync("build.run", {"targets": ["all"]})
    >>> runtime = ctx.require_runtime()  # Lazy resolution
    >>> gateway = ctx.require_gateway()  # Lazy opening
    >>> ctx.logger.info("Building targets: %s", ctx.get_param("targets"))
    """
    operation_id: str
    params: dict[str, Any]
    metadata: ContextMetadata
    started_at: float = field(default_factory=time.monotonic)
    
    @classmethod
    def for_sync(
        cls,
        operation_id: str,
        params: dict[str, Any],
        *,
        config: CliConfig | None = None,
    ) -> ExecutionContext:
        """Create context for synchronous operation.
        
        Parameters
        ----------
        operation_id
            Operation identifier.
        params
            Operation parameters.
        config
            Optional pre-loaded config.
        
        Returns
        -------
        ExecutionContext
            New context ready for execution.
        """
        config = config or load_config(validate=False)
        metadata = ContextMetadata(
            config=config,
            verbosity=params.get("verbose", 0),
            output_format=params.get("output_format", OutputFormat.TEXT),
            dry_run=params.get("dry_run", False),
        )
        return cls(operation_id=operation_id, params=params, metadata=metadata)
    
    # --- Resource Access ---
    
    def require_runtime(self) -> ResolvedRuntime:
        """Get resolved runtime, resolving lazily if needed.
        
        Resolution is cached in metadata._runtime after first call.
        
        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime information.
        
        Raises
        ------
        ResolutionError
            If runtime cannot be resolved from params.
        """
        if self.metadata._runtime is None:
            from codeintel.cli.resolution import resolve_runtime
            self.metadata._runtime = resolve_runtime(self)
        return self.metadata._runtime
    
    def require_gateway(self, *, read_only: bool = True) -> StorageGateway:
        """Get gateway, opening lazily if needed.
        
        Gateway is cached in metadata._gateway after first call.
        
        Parameters
        ----------
        read_only
            Whether to open in read-only mode.
        
        Returns
        -------
        StorageGateway
            Open gateway.
        
        Raises
        ------
        ResolutionError
            If runtime cannot be resolved.
        StorageConnectionError
            If gateway cannot be opened.
        """
        if self.metadata._gateway is None:
            from codeintel.cli.resolution import open_gateway_for_context
            self.metadata._gateway = open_gateway_for_context(self, read_only=read_only)
        return self.metadata._gateway
    
    def close(self) -> None:
        """Close any open resources.
        
        Should be called when execution completes, typically by the executor.
        """
        if self.metadata._gateway is not None:
            self.metadata._gateway.close()
            self.metadata._gateway = None
    
    # --- Parameter Access ---
    
    def get_param[T](self, key: str, default: T = None) -> T:
        """Get parameter with default.
        
        Parameters
        ----------
        key
            Parameter name.
        default
            Value if parameter not present.
        
        Returns
        -------
        T
            Parameter value or default.
        """
        return self.params.get(key, default)
    
    def require_param[T](self, key: str, type_: type[T]) -> T:
        """Get required parameter with type check.
        
        Parameters
        ----------
        key
            Parameter name.
        type_
            Expected type.
        
        Returns
        -------
        T
            Parameter value.
        
        Raises
        ------
        ValueError
            If parameter missing or wrong type.
        """
        value = self.params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
        if not isinstance(value, type_):
            msg = f"Parameter '{key}' must be {type_.__name__}, got {type(value).__name__}"
            raise ValueError(msg)
        return value
    
    # --- Convenience Properties ---
    
    @property
    def config(self) -> CliConfig:
        """Get CLI configuration."""
        return self.metadata.config
    
    @property
    def output_format(self) -> OutputFormat:
        """Get resolved output format."""
        return self.metadata.output_format
    
    @property
    def verbosity(self) -> int:
        """Get verbosity level."""
        return self.metadata.verbosity
    
    @property
    def dry_run(self) -> bool:
        """Check if this is a dry-run."""
        return self.metadata.dry_run
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this operation."""
        return logging.getLogger(f"codeintel.cli.{self.operation_id}")
```

### ResolvedRuntime

```python
@dataclass(frozen=True)
class ResolvedRuntime:
    """Fully resolved runtime - the result of runtime resolution.
    
    This immutable dataclass contains all resolved project information.
    It is created by RuntimeResolver and cached in ExecutionContext.
    
    Attributes
    ----------
    root
        Project root directory.
    project
        Project configuration.
    snapshot
        Repository snapshot reference.
    paths
        Resolved build paths.
    config
        Full CodeIntel configuration.
    serving
        Serving configuration.
    """
    root: Path
    project: ProjectConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    config: CodeIntelConfig
    serving: ServingConfig
    
    @property
    def db_path(self) -> Path:
        """Shortcut to database path."""
        return self.paths.db_path
    
    @property
    def repo(self) -> str:
        """Shortcut to repository slug."""
        return self.snapshot.repo
    
    @property
    def commit(self) -> str:
        """Shortcut to commit SHA."""
        return self.snapshot.commit
```

### RuntimeResolver

```python
class RuntimeResolver:
    """Resolve project runtime from ExecutionContext parameters.
    
    Resolution follows this order:
    1. Try project file discovery (codeintel.yaml) from project_root
    2. Fall back to explicit parameters (repo, commit, db_path, etc.)
    
    The resolver is stateless - all state lives in the ExecutionContext.
    
    Examples
    --------
    >>> resolver = RuntimeResolver()
    >>> runtime = resolver.resolve(ctx)
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    """
    
    def resolve(self, ctx: ExecutionContext) -> ResolvedRuntime:
        """Resolve runtime from context parameters.
        
        Parameters
        ----------
        ctx
            Execution context with params.
        
        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime.
        
        Raises
        ------
        ResolutionError
            If resolution fails (no project file and missing required params).
        """
        project_root = ctx.get_param("project_root")
        
        # Try project file discovery first
        try:
            return self._resolve_from_project(project_root)
        except ProjectNotFoundError:
            pass
        
        # Fall back to explicit params
        return self._resolve_from_params(ctx)
    
    def _resolve_from_project(self, project_root: Path | None) -> ResolvedRuntime:
        """Resolve from project file (codeintel.yaml)."""
        # Implementation: load project config, build runtime
        ...
    
    def _resolve_from_params(self, ctx: ExecutionContext) -> ResolvedRuntime:
        """Resolve from explicit parameters."""
        repo = ctx.get_param("repo")
        commit = ctx.get_param("commit")
        
        if repo is None or commit is None:
            raise ResolutionError(
                "No codeintel.yaml found. Provide --repo and --commit explicitly."
            )
        
        # Implementation: build config from params, create runtime
        ...


# Module-level convenience function
_resolver = RuntimeResolver()

def resolve_runtime(ctx: ExecutionContext) -> ResolvedRuntime:
    """Resolve runtime from context (module-level convenience)."""
    return _resolver.resolve(ctx)
```

### GatewayManager

```python
class GatewayManager:
    """Manage gateway lifecycle for ExecutionContext.
    
    The manager opens gateways on demand and tracks them for cleanup.
    It uses the resolved runtime to determine the database path.
    
    Examples
    --------
    >>> manager = GatewayManager()
    >>> gateway = manager.open(ctx, read_only=True)
    >>> # ... use gateway ...
    >>> manager.close(ctx)
    """
    
    def open(
        self,
        ctx: ExecutionContext,
        *,
        read_only: bool = True,
    ) -> StorageGateway:
        """Open gateway for context.
        
        Parameters
        ----------
        ctx
            Execution context (must have resolved runtime).
        read_only
            Whether to open in read-only mode.
        
        Returns
        -------
        StorageGateway
            Open gateway.
        """
        runtime = ctx.require_runtime()
        storage_config = StorageConfig(
            db_path=runtime.db_path,
            read_only=read_only,
        )
        return open_gateway(storage_config)
    
    def close(self, ctx: ExecutionContext) -> None:
        """Close gateway if open."""
        if ctx.metadata._gateway is not None:
            ctx.metadata._gateway.close()
            ctx.metadata._gateway = None


# Module-level convenience function
_gateway_manager = GatewayManager()

def open_gateway_for_context(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open gateway for context (module-level convenience)."""
    return _gateway_manager.open(ctx, read_only=read_only)
```

---

## Interface Contracts

### Handler Contract

All handlers MUST conform to this interface:

```python
# Type signature
Handler = Callable[[ExecutionContext], CliResult[T]]

# Registration pattern
@operation(
    operation_id: str,
    *,
    category: OperationCategory = OperationCategory.READ,
    description: str = "",
    retryable: bool = False,
)
def handler(ctx: ExecutionContext) -> CliResult[ResultType]:
    """Handler docstring becomes operation description if not specified."""
    ...
```

**Contract Rules**:

1. Handlers MUST accept exactly one argument: `ExecutionContext`
2. Handlers MUST return `CliResult[T]` where T is the result type
3. Handlers MUST use `ctx.require_*()` for resources, never global resolution
4. Handlers MUST NOT create their own context types
5. Handlers MAY raise exceptions (converted to error results by executor)

### Command Contract

All Cyclopts commands MUST conform to this pattern:

```python
@app.command(name="command-name")
@dataclass
class CommandCli:
    """Command docstring."""
    
    # Command-specific parameters
    target: str | None = None
    
    # Common options (flattened)
    options: Annotated[CommonOptions, Parameter(name="*")] = field(
        default_factory=CommonOptions
    )
    
    def __call__(self) -> None:
        """Execute command - delegate to adapter."""
        execute_command("category.command", self)
```

**Contract Rules**:

1. Commands MUST embed `CommonOptions` with `Parameter(name="*")`
2. Commands MUST delegate to `execute_command()` in `__call__`
3. Commands MUST NOT perform resolution or handler invocation directly
4. Commands MUST NOT import handler functions (avoid circular imports)

### Resolution Contract

The resolution layer MUST provide:

```python
def resolve_runtime(ctx: ExecutionContext) -> ResolvedRuntime:
    """
    Preconditions:
        - ctx.params may contain: project_root, repo, commit, db_path, etc.
    
    Postconditions:
        - Returns ResolvedRuntime with all fields populated
        - OR raises ResolutionError with descriptive message
    
    Invariants:
        - Resolution is deterministic given same params
        - Resolution does not modify ctx.params
    """

def open_gateway_for_context(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """
    Preconditions:
        - ctx must have resolvable runtime (or already resolved)
    
    Postconditions:
        - Returns open StorageGateway
        - Gateway is attached to ctx.metadata._gateway
    
    Invariants:
        - Same ctx returns same gateway instance (cached)
    """
```

---

## Data Flow

### Command Execution Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Cyclopts Parsing                                                  │
│    - Parse CLI arguments into command dataclass                      │
│    - CommonOptions populated from flags                              │
└────────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Command.__call__()                                                │
│    - Calls execute_command(operation_id, self)                       │
└────────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. execute_command() / CycloptsAdapter                               │
│    - Extract params from command dataclass                           │
│    - Build ExecutionContext                                          │
│    - Setup logging                                                   │
│    - Get OperationSpec from registry                                 │
│    - Call executor.execute(spec, ctx)                                │
└────────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. OperationExecutor.execute()                                       │
│    - Apply middleware stack (logging, metrics, tracing, resilience)  │
│    - Invoke handler with ctx                                         │
│    - Handle errors, convert to CliResult                             │
│    - Close resources (ctx.close())                                   │
└────────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Handler(ctx)                                                      │
│    - ctx.require_runtime() → triggers RuntimeResolver                │
│    - ctx.require_gateway() → triggers GatewayManager                 │
│    - Execute business logic                                          │
│    - Return CliResult                                                │
└────────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Result Rendering                                                  │
│    - Render CliResult based on output_format                         │
│    - Exit with appropriate code                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Resolution Flow (Lazy)

```
Handler calls ctx.require_runtime()
    │
    ▼
┌───────────────────────────────────────┐
│ Check: ctx.metadata._runtime is None? │
│                                       │
│   No ──────────────────────────────────────▶ Return cached runtime
│   │
│   Yes
│   │
└───┼───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ RuntimeResolver.resolve(ctx)          │
│                                       │
│   1. Get project_root from params     │
│   2. Try load codeintel.yaml          │
│      Success ───────────────────────────────▶ Build from project config
│      │
│      Fail (ProjectNotFoundError)
│      │
│   3. Check explicit params            │
│      repo + commit present? ────────────────▶ Build from explicit params
│      │
│      Missing required params
│      │
│   4. Raise ResolutionError            │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ Cache: ctx.metadata._runtime = result │
│ Return result                         │
└───────────────────────────────────────┘
```

---

## Migration Boundaries

### Phase 1: Resolution Layer

**Scope**: Create `resolution/` package, consolidate all resolution logic

**Boundary**: 
- Handlers continue using old context types
- New resolution available via `ctx.require_runtime()`

**Files Created**:
- `resolution/__init__.py`
- `resolution/runtime.py`
- `resolution/gateway.py`
- `resolution/types.py`
- `resolution/errors.py`

### Phase 2: Options Consolidation

**Scope**: Create `options/` package, unify option bundles

**Boundary**:
- Commands can use either old or new options
- Gradual migration of commands

**Files Created**:
- `options/__init__.py`
- `options/common.py`

### Phase 3: ExecutionContext Enhancement

**Scope**: Add lazy resolution to ExecutionContext

**Boundary**:
- Backward compatible (old code still works)
- New handlers can use `ctx.require_*()` methods

**Files Modified**:
- `execution/context.py`

### Phase 4: Handler Migration

**Scope**: Update handlers to accept `ExecutionContext`

**Boundary**:
- One handler group at a time (build, docs, etc.)
- Old context classes remain until all handlers migrated

**Files Modified**:
- All `*_handlers.py` files

### Phase 5: Cleanup

**Scope**: Remove deprecated code

**Boundary**:
- Remove duplicate `RuntimeCliOptions` classes
- Remove old context types
- Remove duplicate `build_runtime_from_cli` functions

---

## File Inventory

### Files to Create

| File | Layer | Purpose |
|------|-------|---------|
| `resolution/__init__.py` | Resolution | Package exports |
| `resolution/runtime.py` | Resolution | RuntimeResolver |
| `resolution/gateway.py` | Resolution | GatewayManager |
| `resolution/types.py` | Resolution | ResolvedRuntime |
| `resolution/errors.py` | Resolution | ResolutionError |
| `options/__init__.py` | CLI | Package exports |
| `options/common.py` | CLI | CommonOptions |

### Files to Modify

| File | Changes |
|------|---------|
| `execution/context.py` | Add ContextMetadata, lazy resolution |
| `execution/adapter.py` | Add execute_command() |
| `handlers/base.py` | Remove HandlerContext (replaced by ExecutionContext) |
| `build_handlers.py` | Remove BuildRunContext, RuntimeCliOptions |
| `common_handlers.py` | Remove duplicate build_runtime_from_cli |
| `cyclopts_common.py` | Delegate to options/common.py |
| All `*_handlers.py` | Update signatures to accept ExecutionContext |
| All `cyclopts_*.py` | Use execute_command() pattern |

### Files to Delete (Phase 5)

| File | Replacement |
|------|-------------|
| N/A (gradual deprecation) | Code removed from existing files |

---

## Acceptance Criteria

### Functional

- [ ] Single `CommonOptions` class used by all commands
- [ ] Single `resolve_runtime()` function in resolution/
- [ ] Single `ExecutionContext` type accepted by all handlers
- [ ] All handlers use `ctx.require_*()` for resources
- [ ] All commands use `execute_command()` pattern
- [ ] Lazy resolution works correctly (cached after first call)
- [ ] Gateway lifecycle managed correctly (closed on completion)

### Quality

- [ ] Zero pyright errors
- [ ] Zero pyrefly errors
- [ ] Zero ruff errors
- [ ] All CLI tests pass
- [ ] No duplicate RuntimeCliOptions definitions
- [ ] No duplicate build_runtime_from_cli implementations

### Performance

- [ ] Resolution only happens when needed (lazy)
- [ ] Gateway only opened when needed (lazy)
- [ ] No regression in command startup time

---

## Appendix: Cyclopts Integration Details

### Option Flattening

Cyclopts supports nested dataclass flattening via `Parameter(name="*")`:

```python
@dataclass
class CommonOptions:
    verbose: int = 0
    output_format: OutputFormat = OutputFormat.TEXT

@app.command
@dataclass
class MyCommand:
    target: str
    options: Annotated[CommonOptions, Parameter(name="*")] = field(
        default_factory=CommonOptions
    )
```

This makes all CommonOptions fields available as top-level flags:
```bash
codeintel mycommand target --verbose --output-format json
```

### Parameter Extraction

The adapter extracts params using dataclass introspection:

```python
def extract_params(command: object) -> dict[str, Any]:
    """Extract all parameters from command dataclass."""
    if not is_dataclass(command):
        return {}
    
    params = {}
    for field in fields(command):
        value = getattr(command, field.name)
        if is_dataclass(value):
            # Flatten nested dataclass
            params.update(extract_params(value))
        else:
            params[field.name] = value
    return params
```

---

*Document Version: 1.0*
*Created: 2025-01-09*

