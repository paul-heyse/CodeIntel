# CLI Configuration, Handlers, and Execution Consolidation

> **Purpose**: This document defines the target architecture for consolidating three parallel subsystems in the CodeIntel CLI: configuration management, handler utilities, and execution wiring. It serves as the authoritative reference for implementation planning and end-state validation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Design Principles](#design-principles)
4. [Target Architecture](#target-architecture)
5. [Component Specifications](#component-specifications)
6. [Interface Definitions](#interface-definitions)
7. [Migration Strategy](#migration-strategy)
8. [File Inventory](#file-inventory)
9. [Acceptance Criteria](#acceptance-criteria)
10. [Implementation Dependencies](#implementation-dependencies)

---

## Executive Summary

### Problem Statement

The CodeIntel CLI has accumulated three areas of architectural drift:

1. **Configuration**: Three parallel systems (flat dataclasses, nested JSON Schema, bespoke validators) that must be manually synchronized
2. **Handler Utilities**: Identical helper functions copy-pasted across 8+ handler modules
3. **Execution Wiring**: A sophisticated execution pipeline (`OperationExecutor`, middleware, resilience) that Cyclopts commands bypass entirely

### Solution

Consolidate into a unified architecture where:

- **One typed model** drives configuration, schema generation, and validation
- **One utility module** provides all shared handler functionality
- **One execution path** routes all commands through the unified executor

### Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Config validation implementations | 3 | 1 |
| Logging setup implementations | 8+ | 1 |
| Commands using executor middleware | ~5% | 100% |
| Lines of duplicated code | ~500+ | ~0 |
| New command boilerplate | ~100 lines | ~20 lines |

---

## Current State Analysis

### 1. Configuration Subsystem

#### Files Involved

| File | Lines | Role |
|------|-------|------|
| `config_loader.py` | ~343 | Flat `ResolvedConfig`, multi-source loading |
| `cli_config_schema.py` | ~1211 | Nested JSON Schema 2020-12, section dataclasses |
| `cli_validation.py` | ~668 | Bespoke validator framework |
| `cli_types.py` | ~200 | Overlapping type definitions |
| `cyclopts_config.py` | ~402 | Config CLI commands |

#### Architectural Issues

**Issue 1: Schema/Model Mismatch**

```python
# config_loader.py - FLAT structure
@dataclass
class ResolvedConfig:
    progress: bool = True           # Single boolean
    progress_threshold: float = 2.0  # Separate field

# cli_config_schema.py - NESTED structure  
CLI_CONFIG_JSON_SCHEMA = {
    "progress": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "threshold": {"type": "number"}
        }
    }
}
```

**Issue 2: Duplicated Defaults**

```python
# config_loader.py
def _get_defaults() -> dict[str, Any]:
    return {
        "output_format": "text",
        "color": True,
        "progress": True,
        "progress_threshold": 2.0,
    }

# cli_config_schema.py - SAME defaults in JSON Schema
"output_format": {
    "type": "string",
    "default": "text",  # Duplicated!
}
```

**Issue 3: Parallel Validation Systems**

```python
# cli_validation.py - Bespoke validators
class StringValidator(Validator):
    def validate(self, value: Any, field: str) -> ValidationResult:
        ...

# cli_config_schema.py - JSON Schema validation
def validate_with_json_schema(config: dict) -> list[ValidationError]:
    ...
```

#### Data Flow (Current)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Config File    │────▶│  config_loader  │────▶│  ResolvedConfig │
│  (YAML/JSON)    │     │  (flattens)     │     │  (flat)         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  cli_config_    │
                        │  schema.py      │
                        │  (validates)    │
                        └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  cli_validation │
                        │  (re-validates) │
                        └─────────────────┘
```

---

### 2. Handler Utilities Subsystem

#### Duplication Evidence

**Logging Setup (8+ identical implementations)**

```python
# common_handlers.py:84-102
def setup_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# build_handlers.py:111-129 - EXACT SAME CODE
def setup_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    # ... identical ...
```

**Files with Duplicated Logging Setup**

| File | Implementation |
|------|----------------|
| `common_handlers.py` | `setup_logging()` function |
| `build_handlers.py` | `setup_logging()` function (duplicate) |
| `storage_handlers.py` | Inline `logging.basicConfig()` |
| `datasets_handlers.py` | Inline `logging.basicConfig()` |
| `docs_handlers.py` | Inline `logging.basicConfig()` |
| `subsystem_handlers.py` | Inline `logging.basicConfig()` |
| `history_handlers.py` | Inline `logging.basicConfig()` |
| `ide_handlers.py` | Inline `logging.basicConfig()` |
| `ops_handlers.py` | Inline `logging.basicConfig()` |
| `graphs_handlers.py` | Logger setup pattern |

#### Other Duplicated Patterns

| Pattern | Occurrences | Files |
|---------|-------------|-------|
| Runtime building from CLI | 6+ | Multiple `*_handlers.py` |
| Gateway open/close | 5+ | `docs_handlers.py`, `subsystem_handlers.py`, etc. |
| Flag resolution | 4+ | Various handlers |
| Error wrapping | 3+ | Via `run_handler()` inconsistently |

---

### 3. Execution Wiring Subsystem

#### Current Execution Paths

**Path A: Through Executor (Minority)**

```
cyclopts_ops.py ──▶ invoke_operation() ──▶ OperationExecutor
                                               │
                                               ├── Middleware
                                               ├── Resilience
                                               ├── Progress
                                               └── Tracing
```

**Path B: Direct Handler Call (Majority)**

```
cyclopts_build.py ──▶ run_handler() ──▶ build_run_handler()
                          │
                          └── Basic error handling only
                              (NO middleware, NO resilience, NO progress)
```

**Path C: Inline Logic (Some Commands)**

```
cyclopts_config.py ──▶ __call__() ──▶ Direct logic
                                       (NO structure at all)
```

#### Usage Analysis

```
grep results for executor usage in cyclopts_*.py:
- get_executor: 0 matches
- OperationRegistry: 0 matches
- OperationExecutor: 0 matches
```

**Conclusion**: The entire execution infrastructure is unused by Cyclopts commands.

---

## Design Principles

### 1. Single Source of Truth

Every concept has exactly one authoritative definition:

- Configuration structure → `config/model.py`
- Validation rules → Derived from model types
- JSON Schema → Generated from model
- Handler context → `handlers/base.py`

### 2. Model-Driven Everything

The typed configuration model drives:

- Default values (from dataclass defaults)
- JSON Schema generation (from type annotations)
- Environment variable mapping (from field paths)
- Validation (from type constraints)

### 3. Decorator-First Registration

Operations are defined where handlers live:

```python
@operation("build.run", category=OperationCategory.BUILD)
def build_run_handler(ctx: HandlerContext, options: BuildOptions) -> CliResult[BuildResult]:
    ...
```

### 4. Unified Execution Path

All commands flow through one executor:

```
Cyclopts Command ──▶ Adapter ──▶ OperationExecutor ──▶ Handler
                                        │
                                        ├── Logging Middleware
                                        ├── Metrics Middleware
                                        ├── Tracing Middleware
                                        ├── Resilience Middleware
                                        ├── Progress Middleware
                                        └── Plugin Middleware
```

### 5. Thin Commands, Rich Handlers

- **Commands**: Parameter definitions only (~20 lines)
- **Handlers**: Pure business logic (~100+ lines)
- **Adapter**: Bridges commands to executor (shared)

### 6. Zero Duplication

Shared functionality lives in exactly one place:

| Functionality | Location |
|---------------|----------|
| Logging setup | `handlers/base.py::setup_logging()` |
| Runtime building | `handlers/base.py::build_handler_context()` |
| Gateway management | `handlers/common.py::gateway_context()` |
| Error wrapping | `execution/adapter.py` (via executor) |

---

## Target Architecture

### Module Structure

```
src/codeintel/cli/
├── config/                              # Configuration subsystem
│   ├── __init__.py                      # Public API exports
│   ├── model.py                         # Single typed config model
│   ├── schema.py                        # JSON Schema 2020-12 generation
│   ├── loader.py                        # Multi-source loading
│   ├── env.py                           # Environment variable parsing
│   └── validation.py                    # Model-driven validation
│
├── execution/                           # Execution subsystem (existing + enhanced)
│   ├── __init__.py                      # Public API exports
│   ├── context.py                       # ExecutionContext
│   ├── executor.py                      # OperationExecutor
│   ├── middleware.py                    # Middleware protocol + implementations
│   ├── progress.py                      # Progress tracking
│   ├── types.py                         # Type definitions
│   └── adapter.py                       # NEW: Cyclopts-to-Executor bridge
│
├── handlers/                            # Handler subsystem (consolidated)
│   ├── __init__.py                      # Public API exports
│   ├── base.py                          # NEW: Base utilities (logging, context)
│   ├── common.py                        # Shared utilities (gateway, project)
│   ├── build.py                         # Build handlers
│   ├── docs.py                          # Docs handlers
│   ├── storage.py                       # Storage handlers
│   ├── datasets.py                      # Dataset handlers
│   ├── graphs.py                        # Graph handlers
│   ├── history.py                       # History handlers
│   ├── ide.py                           # IDE handlers
│   ├── ops.py                           # Operations handlers
│   └── subsystem.py                     # Subsystem handlers
│
├── commands/                            # Command subsystem (renamed from cyclopts_*)
│   ├── __init__.py                      # App registration
│   ├── app.py                           # Main Cyclopts app
│   ├── build.py                         # Build commands (thin)
│   ├── docs.py                          # Docs commands (thin)
│   ├── storage.py                       # Storage commands (thin)
│   ├── datasets.py                      # Dataset commands (thin)
│   ├── graphs.py                        # Graph commands (thin)
│   ├── history.py                       # History commands (thin)
│   ├── ide.py                           # IDE commands (thin)
│   ├── ops.py                           # Operations commands (thin)
│   ├── subsystem.py                     # Subsystem commands (thin)
│   ├── config.py                        # Config commands
│   ├── plugins.py                       # Plugin commands
│   ├── jobs.py                          # Job commands
│   ├── health.py                        # Health commands
│   ├── help.py                          # Help commands
│   └── completions.py                   # Shell completion commands
│
├── plugins/                             # Plugin subsystem (existing)
├── operations/                          # Operation definitions (existing)
├── completions/                         # Shell completions (existing)
├── results.py                           # CliResult (existing)
├── cli_types.py                         # Shared types (reduced)
├── cli_errors.py                        # Error types (existing)
├── error_taxonomy.py                    # Error codes (existing)
├── cli_render.py                        # Output rendering (existing)
├── output.py                            # Output utilities (existing)
├── operation_registry.py                # Operation registry (existing)
├── introspection.py                     # Operation introspection (existing)
├── resilience.py                        # Resilience infrastructure (existing)
├── resilience_middleware.py             # Resilience middleware (existing)
├── observability.py                     # Observability middleware (existing)
├── telemetry.py                         # Telemetry (existing)
├── health.py                            # Health checks (existing)
├── help_system.py                       # Help infrastructure (existing)
├── shell.py                             # Interactive shell (existing)
├── pipelines.py                         # Pipeline execution (existing)
├── job_runner.py                        # Job execution (existing)
├── jobs.py                              # Job management (existing)
├── project.py                           # Project detection (existing)
└── dry_run.py                           # Dry run mode (existing)
```

### Data Flow (Target)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Configuration Loading                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │
│   │ Defaults│ + │  File   │ + │   Env   │ + │   CLI   │                 │
│   │ (model) │   │ (YAML)  │   │  Vars   │   │  Flags  │                 │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                 │
│        │             │             │             │                       │
│        └──────────┬──┴─────────────┴─────────────┘                       │
│                   │                                                      │
│                   ▼                                                      │
│            ┌──────────────┐                                              │
│            │  CliConfig   │  ◀── Single typed model                      │
│            │  (validated) │      (generates JSON Schema)                 │
│            └──────────────┘                                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           Command Execution                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────┐                                                      │
│   │ Cyclopts Cmd  │  ◀── Thin: parameter definitions only                │
│   │ (dataclass)   │                                                      │
│   └───────┬───────┘                                                      │
│           │                                                              │
│           ▼                                                              │
│   ┌───────────────┐                                                      │
│   │CycloptsAdapter│  ◀── Bridges command to executor                     │
│   └───────┬───────┘                                                      │
│           │                                                              │
│           ▼                                                              │
│   ┌───────────────┐                                                      │
│   │ Executor      │  ◀── Applies all middleware                          │
│   │ .execute()    │                                                      │
│   └───────┬───────┘                                                      │
│           │                                                              │
│           ▼                                                              │
│   ┌───────────────────────────────────────────────────────────┐          │
│   │                    Middleware Stack                        │          │
│   ├───────────────────────────────────────────────────────────┤          │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │          │
│   │  │ Logging │▶│ Metrics │▶│ Tracing │▶│Resilience│▶ ...    │          │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │          │
│   └───────────────────────────────────────────────────────────┘          │
│           │                                                              │
│           ▼                                                              │
│   ┌───────────────┐                                                      │
│   │   Handler     │  ◀── Pure business logic                             │
│   │ (decorated)   │      @operation("build.run")                         │
│   └───────┬───────┘                                                      │
│           │                                                              │
│           ▼                                                              │
│   ┌───────────────┐                                                      │
│   │  CliResult    │  ◀── Structured output                               │
│   └───────────────┘                                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Configuration Model (`config/model.py`)

#### Design Goals

1. Single dataclass hierarchy defines all configuration
2. Nested structure matches JSON Schema exactly
3. Defaults defined once (in dataclass fields)
4. Type annotations drive validation
5. Generates JSON Schema 2020-12 programmatically

#### Core Model

```python
@dataclass(frozen=True)
class ProgressConfig:
    """Progress display configuration."""
    enabled: bool = True
    threshold: float = 2.0

@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry and observability configuration."""
    enabled: bool = True
    endpoint: str | None = None
    service_name: str = "codeintel-cli"

@dataclass(frozen=True)
class RetryConfig:
    """Retry policy configuration."""
    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 30.0

@dataclass(frozen=True)
class StorageConfigSection:
    """Storage backend configuration."""
    db_path: Path | None = None
    cache_dir: Path | None = None
    max_connections: int = 5

@dataclass(frozen=True)
class ProjectConfigSection:
    """Project identification configuration."""
    name: str | None = None
    repo: str | None = None
    root: Path | None = None
    commit: str | None = None

@dataclass(frozen=True)
class PluginsConfigSection:
    """Plugin system configuration."""
    directories: tuple[Path, ...] = ()
    disabled: tuple[str, ...] = ()

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
OutputFormat = Literal["text", "json"]

@dataclass(frozen=True)
class CliConfig:
    """Complete CLI configuration - single source of truth."""
    
    # Top-level settings
    output_format: OutputFormat = "text"
    color: bool = True
    log_level: LogLevel = "WARNING"
    
    # Nested sections
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    storage: StorageConfigSection = field(default_factory=StorageConfigSection)
    project: ProjectConfigSection = field(default_factory=ProjectConfigSection)
    plugins: PluginsConfigSection = field(default_factory=PluginsConfigSection)
    
    # Metadata (not in schema)
    _sources: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Generate JSON Schema 2020-12 from this model."""
        ...
    
    @classmethod
    def from_sources(
        cls,
        config_file: Path | None = None,
        env_prefix: str = "CODEINTEL_",
        cli_overrides: dict[str, Any] | None = None,
    ) -> CliConfig:
        """Load config with precedence: defaults < file < env < cli."""
        ...
```

---

### 2. Handler Base (`handlers/base.py`)

#### Single Logging Implementation

```python
_LOGGING_CONFIGURED = False

def setup_logging(
    verbosity: int = 0,
    *,
    config: CliConfig | None = None,
    force: bool = False,
) -> None:
    """Configure logging - SINGLE IMPLEMENTATION for all handlers.
    
    Parameters
    ----------
    verbosity
        0=use config default, 1=INFO, 2+=DEBUG.
    config
        Configuration for default log level.
    force
        Force reconfiguration.
    """
    global _LOGGING_CONFIGURED
    
    if _LOGGING_CONFIGURED and not force:
        return
    
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    elif config is not None:
        level = getattr(logging, config.log_level, logging.WARNING)
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=force,
    )
    _LOGGING_CONFIGURED = True
```

#### Unified Handler Context

```python
@dataclass(frozen=True)
class HandlerContext:
    """Unified context for all handlers.
    
    Replaces scattered RuntimeCliOptions, BuildRunContext, etc.
    """
    config: CliConfig
    execution: ExecutionContext
    project_root: Path | None = None
    verbosity: int = 0
    
    @property
    def operation_id(self) -> str:
        return self.execution.operation_id
    
    @property
    def logger(self) -> logging.Logger:
        return get_handler_logger(self.operation_id)
    
    @property
    def output_format(self) -> str:
        return self.config.output_format

def build_handler_context(
    operation_id: str,
    params: dict[str, Any],
    *,
    config: CliConfig | None = None,
    verbosity: int = 0,
) -> HandlerContext:
    """Build unified handler context."""
    config = config or CliConfig.from_sources()
    setup_logging(verbosity, config=config)
    execution = ExecutionContext.for_sync(operation_id, params)
    return HandlerContext(config=config, execution=execution, verbosity=verbosity)
```

---

### 3. Execution Adapter (`execution/adapter.py`)

#### Operation Decorator

```python
def operation(
    operation_id: str,
    *,
    category: OperationCategory = OperationCategory.READ,
    description: str = "",
    retryable: bool = False,
) -> Callable[[Callable[P, CliResult[T]]], Callable[P, CliResult[T]]]:
    """Register handler as operation and route through executor.
    
    Example
    -------
    @operation("build.run", category=OperationCategory.BUILD)
    def build_run_handler(ctx: HandlerContext) -> CliResult[BuildResult]:
        ...
    """
    def decorator(handler):
        spec = OperationSpec(
            operation_id=operation_id,
            handler=handler,
            category=category,
            description=description or handler.__doc__,
            retryable=retryable,
        )
        get_operation_registry().register(spec)
        
        @wraps(handler)
        def wrapper(*args, **kwargs):
            executor = get_executor()
            params = _merge_args_to_params(args, kwargs)
            return executor.execute(spec, params)
        
        wrapper._original = handler  # For testing
        wrapper._spec = spec
        return wrapper
    return decorator
```

#### Cyclopts Adapter

```python
class CycloptsAdapter:
    """Bridge Cyclopts command classes to operation executor."""
    
    def __init__(self, operation_id: str, handler: Callable) -> None:
        self._operation_id = operation_id
        self._handler = handler
        self._spec = getattr(handler, "_spec", None)
    
    def __call__(self, command: Any) -> None:
        params = self._extract_params(command)
        verbosity = params.pop("verbosity", 0)
        
        ctx = build_handler_context(self._operation_id, params, verbosity=verbosity)
        
        if self._spec:
            result = get_executor().execute(self._spec, params)
        else:
            result = self._handler(ctx, **params)
        
        render_cli_result(result, ctx.output_format)
```

---

### 4. Thin Command Pattern

#### Before (Current)

```python
# cyclopts_build.py - 50+ lines per command
@build_app.command(name="run")
@dataclass
class BuildRunCli:
    targets: Annotated[list[str] | None, Parameter(...)] = None
    module: Annotated[str | None, Parameter(...)] = None
    # ... more params ...
    
    def __call__(self) -> None:
        # Validation logic here
        _validate_build_run_selection(...)
        
        # Manual context building
        runtime_opts, verbose, output_format = make_handler_context(...)
        
        # Logging setup (duplicated)
        setup_logging(verbose)
        
        # Build options manually
        options = BuildRunOptions(...)
        ctx_opts = BuildRunContext(...)
        
        # Call handler through run_handler (bypasses executor)
        run_handler(build_run_handler, options, ctx_opts)
```

#### After (Target)

```python
# commands/build.py - ~15 lines per command
@build_app.command(name="run")
@dataclass
class BuildRunCli:
    targets: Annotated[list[str] | None, Parameter(...)] = None
    module: Annotated[str | None, Parameter(...)] = None
    # ... params only ...
    
    def __call__(self) -> None:
        CycloptsAdapter("build.run", build_run_handler)(self)
```

---

## Migration Strategy

### Phase 1: Configuration Consolidation

**Scope**: Create unified config model, migrate loading

| Step | Action | Risk |
|------|--------|------|
| 1.1 | Create `config/` package | Low |
| 1.2 | Implement `CliConfig` model | Low |
| 1.3 | Implement schema generation | Low |
| 1.4 | Implement loader | Medium |
| 1.5 | Add backward-compatible imports | Low |
| 1.6 | Delete old files | Low |

### Phase 2: Handler Utilities Consolidation

**Scope**: Consolidate logging and shared utilities

| Step | Action | Risk |
|------|--------|------|
| 2.1 | Create `handlers/base.py` | Low |
| 2.2 | Create `handlers/common.py` | Low |
| 2.3 | Update handlers to use shared utilities | Medium |
| 2.4 | Remove duplicate implementations | Medium |
| 2.5 | Move handlers to `handlers/` package | Medium |

### Phase 3: Execution Wiring

**Scope**: Route all commands through executor

| Step | Action | Risk |
|------|--------|------|
| 3.1 | Create `execution/adapter.py` | Low |
| 3.2 | Implement `@operation` decorator | Medium |
| 3.3 | Implement `CycloptsAdapter` | Medium |
| 3.4 | Decorate all handlers | Medium |
| 3.5 | Update all commands | High |
| 3.6 | Rename to `commands/` | Low |

### Phase 4: Cleanup

**Scope**: Remove deprecated code

| Step | Action | Risk |
|------|--------|------|
| 4.1 | Delete old config files | Low |
| 4.2 | Delete old handler files | Low |
| 4.3 | Update imports | Low |
| 4.4 | Full test suite | Low |

---

## File Inventory

### Files to Delete

| File | Replacement | Phase |
|------|-------------|-------|
| `config_loader.py` | `config/loader.py` | 1 |
| `cli_config_schema.py` | `config/model.py` + `config/schema.py` | 1 |
| `cli_validation.py` | `config/validation.py` | 1 |

### Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `config/__init__.py` | Public API | 1 |
| `config/model.py` | Unified config model | 1 |
| `config/schema.py` | JSON Schema generation | 1 |
| `config/loader.py` | Multi-source loading | 1 |
| `config/env.py` | Environment variable parsing | 1 |
| `config/validation.py` | Model-driven validation | 1 |
| `handlers/__init__.py` | Public API | 2 |
| `handlers/base.py` | Shared utilities | 2 |
| `execution/adapter.py` | Cyclopts adapter | 3 |
| `commands/__init__.py` | App registration | 3 |
| `commands/app.py` | Main app | 3 |

### Files to Refactor

| Current | Target | Phase |
|---------|--------|-------|
| `common_handlers.py` | `handlers/common.py` | 2 |
| `build_handlers.py` | `handlers/build.py` | 2 |
| `*_handlers.py` (all) | `handlers/*.py` | 2 |
| `cyclopts_build.py` | `commands/build.py` | 3 |
| `cyclopts_*.py` (all) | `commands/*.py` | 3 |

---

## Acceptance Criteria

### Configuration

- [ ] `CliConfig` is single source of truth
- [ ] JSON Schema generated from model
- [ ] Schema matches model structure (nested)
- [ ] Defaults defined once
- [ ] Environment variables map to nested paths
- [ ] All validation uses model constraints

### Handler Utilities

- [ ] `setup_logging()` has exactly one implementation
- [ ] All handlers use `HandlerContext`
- [ ] Gateway uses shared `gateway_context()`
- [ ] No duplicate utility functions

### Execution Wiring

- [ ] All commands route through `OperationExecutor`
- [ ] All middleware applies to all commands
- [ ] `@operation` decorator auto-registers handlers
- [ ] Direct handler invocation works for testing

### Quality Gates

- [ ] Zero pyright errors
- [ ] Zero pyrefly errors
- [ ] Zero ruff errors
- [ ] All CLI tests pass

---

## Implementation Dependencies

```
Phase 1 (Config)
    │
    ▼
Phase 2 (Handlers) ◀── Depends on config model
    │
    ▼
Phase 3 (Execution) ◀── Depends on handlers
    │
    ▼
Phase 4 (Cleanup) ◀── Depends on all phases
```

---

## Appendix: Environment Variable Mapping

| Environment Variable | Config Path | Type |
|---------------------|-------------|------|
| `CODEINTEL_OUTPUT_FORMAT` | `output_format` | string |
| `CODEINTEL_COLOR` | `color` | bool |
| `CODEINTEL_LOG_LEVEL` | `log_level` | string |
| `CODEINTEL_PROGRESS_ENABLED` | `progress.enabled` | bool |
| `CODEINTEL_PROGRESS_THRESHOLD` | `progress.threshold` | float |
| `CODEINTEL_TELEMETRY_ENABLED` | `telemetry.enabled` | bool |
| `CODEINTEL_TELEMETRY_ENDPOINT` | `telemetry.endpoint` | string |
| `CODEINTEL_RETRY_MAX_ATTEMPTS` | `retry.max_attempts` | int |
| `CODEINTEL_RETRY_INITIAL_DELAY` | `retry.initial_delay` | float |
| `CODEINTEL_STORAGE_DB_PATH` | `storage.db_path` | path |
| `CODEINTEL_STORAGE_CACHE_DIR` | `storage.cache_dir` | path |

---

*Document Version: 1.0*
*Last Updated: 2025-01-09*

