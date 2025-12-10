# CLI Consolidation Architecture

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Date**: 2025-01-10  
> **Scope**: `src/codeintel/cli/`

## Executive Summary

This document specifies the target architecture for consolidating the CodeIntel CLI subsystem. The consolidation addresses three primary concerns:

1. **Runtime Resolution Fragmentation** — Seven parallel implementations of `build_runtime_from_cli` with drifting field sets
2. **Configuration Loading Duplication** — Parallel config loading in Cyclopts modules vs. the `cli/config/` package
3. **Output Rendering Inconsistency** — 85+ direct `sys.stdout.write()` calls bypassing the rendering infrastructure

The target architecture establishes clear **single sources of truth** for each concern while maintaining backward compatibility during migration.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture Overview](#2-target-architecture-overview)
3. [Component Specifications](#3-component-specifications)
   - [3.1 ConfigService](#31-configservice)
   - [3.2 RuntimeResolver](#32-runtimeresolver)
   - [3.3 RenderingService](#33-renderingservice)
   - [3.4 Handler Protocol](#34-handler-protocol)
   - [3.5 Cyclopts Integration](#35-cyclopts-integration)
4. [Data Flow Contracts](#4-data-flow-contracts)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [Directory Structure](#6-directory-structure)
7. [Migration Strategy](#7-migration-strategy)
8. [Appendix: Current Duplication Inventory](#appendix-current-duplication-inventory)

---

## 1. Current State Analysis

### 1.1 Runtime Resolution Fragmentation

| Module | `RuntimeCliOptions` Fields | Notes |
|--------|---------------------------|-------|
| `ide_handlers.py` | `project_root` | Minimal, project discovery only |
| `subsystem_handlers.py` | `project_root` | Identical to ide_handlers |
| `datasets_handlers.py` | 6 fields + nested classes | Extended with repo, commit, paths |
| `build_handlers.py` | Imports from `common_handlers` | Uses canonical version |
| `common_handlers.py` | Alias to `RuntimeOptions` | Full implementation |
| `cyclopts_common.py` | `RuntimeCLI` dataclass | Cyclopts-specific with Parameter annotations |
| `resolution/runtime.py` | `RuntimeResolver` class | Uses ExecutionContext.params dict |

**Problem**: Field drift causes subtle bugs. Simple handlers (ide, subsystem) silently ignore explicit `--repo`, `--commit` flags because their `RuntimeCliOptions` lacks those fields.

### 1.2 Configuration Loading Duplication

```
cyclopts_common.py:
  CONFIG_ENV_PREFIX = "CODEINTEL_"
  _optional_toml_config()  → loads codeintel.toml
  _ENV_CONFIG             → cyclopts.config.Env

cyclopts_config.py:
  CONFIG_ENV_PREFIX = "CODEINTEL_"  (duplicate)
  _resolve_config_path()
  _get_env_overrides()

cli/config/:
  load_config()           → loads ~/.codeintel/config.yaml
  CliConfig model         → typed, validated
  ENV_MAPPINGS           → structured env mapping
```

**Problem**: Different config file paths, no validation on Cyclopts path, CliConfig features unused.

### 1.3 Output Rendering Inconsistency

| Path | Location | Capabilities |
|------|----------|--------------|
| `CliResult.render()` | `results.py` | Format-aware, warnings, metadata |
| `RichRenderer` | `cli_render.py` | Tables, colors, TTY detection |
| `PlainRenderer` | `cli_render.py` | Non-TTY fallback |
| `StreamingRenderer` | `pipelines.py` | JSONL for batch operations |
| Direct `sys.stdout.write()` | 85+ instances | No consistency guarantees |

**Problem**: Direct writes bypass warnings, metadata, format negotiation, and create testing difficulties.

---

## 2. Target Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Points Layer                              │
│                                                                             │
│  cyclopts_app.py ──► cyclopts_*.py commands                                │
│  (Params → ExecutionContext → Handler → RenderingService)                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Execution Infrastructure Layer                           │
│                                                                             │
│  ┌───────────────┐    ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ ConfigService │    │ExecutionContext │    │   RenderingService      │   │
│  │               │    │                 │    │                         │   │
│  │ • load()      │    │ • operation_id  │    │ • render_result()       │   │
│  │ • validate()  │    │ • params        │    │ • render_table()        │   │
│  │ • to_cyclopts │    │ • runtime       │    │ • render_error()        │   │
│  └───────┬───────┘    └────────┬────────┘    └────────────┬────────────┘   │
│          │                     │                          │                 │
└──────────┼─────────────────────┼──────────────────────────┼─────────────────┘
           │                     │                          │
           │                     ▼                          │
           │   ┌─────────────────────────────────────────┐  │
           │   │           Resolution Layer              │  │
           │   │                                         │  │
           │   │  RuntimeResolver ──► ResolvedRuntime   │  │
           │   │  GatewayManager  ──► StorageGateway    │  │
           │   └─────────────────────────────────────────┘  │
           │                     │                          │
           │                     ▼                          │
           │   ┌─────────────────────────────────────────┐  │
           │   │            Handler Layer                │  │
           │   │                                         │  │
           │   │  HandlerProtocol[T]                     │  │
           │   │  • Receives: HandlerContext             │  │
           │   │  • Returns:  CliResult[T]               │  │
           │   │  • NEVER writes to stdout               │  │
           │   └─────────────────────────────────────────┘  │
           │                     │                          │
           └─────────────────────┴──────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    Result Pipeline     │
                    │                        │
                    │  CliResult[T]          │
                    │    ↓                   │
                    │  RenderingService      │
                    │    ↓                   │
                    │  stdout / stderr       │
                    └────────────────────────┘
```

### Architectural Principles

1. **Single Source of Truth** — Each concern has exactly one authoritative implementation
2. **Explicit Dependencies** — Handlers receive typed context, never resolve their own dependencies
3. **Output Separation** — Business logic returns data; rendering is a separate concern
4. **Composable Primitives** — Small, focused types that compose into larger contexts
5. **Testability** — All components can be tested without I/O mocking

---

## 3. Component Specifications

### 3.1 ConfigService

**Location**: `cli/config/service.py`

**Responsibility**: Single source of truth for all CLI configuration loading, validation, and access.

#### Interface

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from cyclopts import App

from codeintel.cli.config.model import CliConfig


@dataclass(frozen=True)
class ConfigService:
    """Unified configuration service.
    
    Precedence (highest to lowest):
    1. CLI flags (explicit overrides)
    2. Environment variables (CODEINTEL_*)
    3. Config file (codeintel.toml or ~/.codeintel/config.yaml)
    4. Built-in defaults from CliConfig
    
    Attributes
    ----------
    config
        The resolved, validated configuration.
    sources
        Ordered list of sources that contributed to the config.
        Example: ("defaults", "file:~/.codeintel/config.yaml", "environment")
    """
    
    config: CliConfig
    sources: tuple[str, ...]
    
    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
        *,
        env_prefix: str = "CODEINTEL_",
        validate: bool = True,
    ) -> ConfigService:
        """Load configuration from all sources with precedence.
        
        Parameters
        ----------
        config_path
            Explicit config file path. If None, searches default locations.
        cli_overrides
            Overrides from CLI flags (highest precedence).
        env_prefix
            Environment variable prefix.
        validate
            If True, validate config and raise ConfigLoadError on failure.
            
        Returns
        -------
        ConfigService
            Service with loaded configuration.
            
        Raises
        ------
        ConfigLoadError
            If validation is enabled and config is invalid.
        """
        ...
    
    def get_cyclopts_config_chain(self) -> list[Callable[[App, tuple[str, ...], Any], Any]]:
        """Return Cyclopts-compatible config callables.
        
        Integrates with Cyclopts' config parameter while maintaining
        our unified precedence. The returned chain:
        1. Applies TOML config if present
        2. Applies environment variable overrides
        
        Returns
        -------
        list
            Config callables for Cyclopts App.config parameter.
        """
        ...
    
    def with_overrides(self, **overrides: Any) -> ConfigService:
        """Create new service with overrides applied.
        
        Useful for testing or command-specific modifications.
        """
        ...
```

#### Integration Point: Cyclopts Root App

```python
# cli/cyclopts_app.py

def make_root_app() -> App:
    """Construct root Cyclopts application with unified config."""
    # ConfigService handles all config loading
    service = ConfigService.load(validate=False)  # Defer validation to command execution
    
    return App(
        name="codeintel",
        help="CodeIntel unified CLI.",
        default_parameter=Parameter(show_default=True),
        config=service.get_cyclopts_config_chain(),
        result_action=["call_if_callable", "return_value"],
        print_error=True,
    )
```

#### Deprecation: Remove from cyclopts_common.py

```python
# DELETE these from cyclopts_common.py:
CONFIG_ENV_PREFIX = "CODEINTEL_"
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("codeintel.toml")
_ENV_CONFIG = cyclopts_config.Env(CONFIG_ENV_PREFIX)

def _resolve_config_path() -> Path: ...
def _optional_toml_config(...) -> object: ...
```

---

### 3.2 RuntimeResolver

**Location**: `cli/resolution/runtime.py`

**Responsibility**: Single source of truth for resolving project runtime from any input source.

#### Core Types

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.config.models import CodeIntelConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.config.serving_models import ServingConfig
    from codeintel.cli.project import ProjectConfig


@dataclass(frozen=True)
class BackendFlags:
    """Graph backend configuration flags.
    
    Attributes
    ----------
    use_gpu
        Whether to attempt GPU acceleration.
    backend
        Backend selection: "auto", "cpu", or "nx-cugraph".
    strict
        Whether to enforce strict backend compatibility.
    """
    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RuntimeParams:
    """Canonical runtime parameters from any input source.
    
    This is THE type for runtime parameters. All other RuntimeCliOptions
    variants are deprecated in favor of this single type.
    
    Attributes
    ----------
    project_root
        Root directory for project file discovery.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit SHA.
    db_path
        Explicit database path.
    build_dir
        Build output directory.
    repo_root
        Repository root path.
    document_output_dir
        Document export directory.
    backend
        Graph backend configuration.
    """
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)
    
    # --- Factory Methods ---
    
    @classmethod
    def from_context(cls, ctx: ExecutionContext) -> RuntimeParams:
        """Extract RuntimeParams from ExecutionContext.params dict.
        
        The context params may contain any subset of fields.
        Missing fields use defaults.
        """
        params = ctx.params
        backend_raw = params.get("backend", {})
        backend = BackendFlags(
            use_gpu=backend_raw.get("use_gpu", False) if isinstance(backend_raw, dict) else False,
            backend=backend_raw.get("backend", "auto") if isinstance(backend_raw, dict) else "auto",
            strict=backend_raw.get("strict", False) if isinstance(backend_raw, dict) else False,
        )
        
        return cls(
            project_root=_to_path(params.get("project_root")),
            repo=_to_str(params.get("repo")),
            commit=_to_str(params.get("commit")),
            db_path=_to_path(params.get("db_path")),
            build_dir=_to_path(params.get("build_dir")),
            repo_root=_to_path(params.get("repo_root")),
            document_output_dir=_to_path(params.get("document_output_dir")),
            backend=backend,
        )
    
    @classmethod
    def from_cyclopts(cls, runtime_cli: RuntimeCLI) -> RuntimeParams:
        """Convert Cyclopts RuntimeCLI to canonical RuntimeParams.
        
        RuntimeCLI is a Cyclopts-specific dataclass with Parameter
        annotations. This method extracts values into the canonical type.
        """
        return cls(
            project_root=runtime_cli.project_root,
            repo=runtime_cli.repo,
            commit=runtime_cli.commit,
            db_path=runtime_cli.db_path,
            build_dir=runtime_cli.build_dir,
            repo_root=runtime_cli.repo_root,
            document_output_dir=runtime_cli.document_output_dir,
            backend=BackendFlags(),  # RuntimeCLI doesn't include backend
        )
    
    @classmethod
    def minimal(cls, project_root: Path | None = None) -> RuntimeParams:
        """Create minimal params for simple commands.
        
        Use for commands that only need project discovery (ide hints, etc).
        """
        return cls(project_root=project_root)


@dataclass(frozen=True)
class ResolvedRuntime:
    """Immutable result of runtime resolution.
    
    Contains all resolved project information. Created by RuntimeResolver
    and consumed by handlers.
    
    Attributes
    ----------
    root
        Project root directory.
    project
        Project configuration (from codeintel.yaml or constructed).
    snapshot
        Repository snapshot reference.
    paths
        Resolved build paths.
    config
        Full CodeIntelConfig.
    serving
        Serving configuration for API operations.
    """
    root: Path
    project: ProjectConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    config: CodeIntelConfig
    serving: ServingConfig
    
    # --- Convenience Properties ---
    
    @property
    def db_path(self) -> Path:
        """Database file path shortcut."""
        return self.paths.db_path
    
    @property
    def repo(self) -> str:
        """Repository slug shortcut."""
        return self.snapshot.repo
    
    @property
    def commit(self) -> str:
        """Commit SHA shortcut."""
        return self.snapshot.commit
```

#### Resolver Class

```python
class RuntimeResolver:
    """Resolve RuntimeParams to ResolvedRuntime.
    
    Resolution Strategy
    -------------------
    1. If project_root is provided, attempt project file discovery
    2. If discovery succeeds, return ResolvedRuntime from project
    3. If discovery fails and allow_fallback=True, use explicit params
    4. If explicit params missing required fields, raise ResolutionError
    
    Thread Safety
    -------------
    RuntimeResolver is stateless and thread-safe. All state is in params/result.
    
    Examples
    --------
    >>> params = RuntimeParams.from_context(ctx)
    >>> runtime = RuntimeResolver.resolve(params)
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    """
    
    @staticmethod
    def resolve(
        params: RuntimeParams,
        *,
        allow_fallback: bool = True,
    ) -> ResolvedRuntime:
        """Resolve runtime from parameters.
        
        Parameters
        ----------
        params
            Canonical runtime parameters.
        allow_fallback
            If True, attempt explicit param fallback on project file miss.
            If False, raise immediately on missing project file.
            
        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime.
            
        Raises
        ------
        ResolutionError
            If resolution fails. Includes structured guidance:
            - missing_params: List of missing required parameters
            - suggestion: Human-readable fix suggestion
        """
        ...
    
    @staticmethod
    def resolve_from_context(
        ctx: ExecutionContext,
        *,
        allow_fallback: bool = True,
    ) -> ResolvedRuntime:
        """Convenience method: extract params from context and resolve.
        
        Equivalent to:
            RuntimeResolver.resolve(RuntimeParams.from_context(ctx), ...)
        """
        return RuntimeResolver.resolve(
            RuntimeParams.from_context(ctx),
            allow_fallback=allow_fallback,
        )
```

#### Gateway Management

```python
# cli/resolution/gateway.py

class GatewayManager:
    """Manage StorageGateway lifecycle.
    
    Provides:
    - Lazy gateway creation
    - Connection caching (optional)
    - Proper cleanup on context exit
    
    Usage
    -----
    Handlers should access gateways through HandlerContext.gateway,
    which delegates to GatewayManager internally.
    """
    
    def __init__(self, runtime: ResolvedRuntime):
        self._runtime = runtime
        self._gateway: StorageGateway | None = None
        self._graph_runtime: GraphRuntime | None = None
    
    @property
    def gateway(self) -> StorageGateway:
        """Get or create storage gateway (lazy).
        
        Gateway is opened on first access and cached for reuse.
        """
        if self._gateway is None:
            self._gateway = self._open_gateway()
        return self._gateway
    
    @property
    def graph_runtime(self) -> GraphRuntime:
        """Get or create graph runtime (lazy)."""
        if self._graph_runtime is None:
            self._graph_runtime = self._build_graph_runtime()
        return self._graph_runtime
    
    def close(self) -> None:
        """Close all managed resources.
        
        Called automatically when ExecutionContext exits.
        """
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None
        self._graph_runtime = None
```

#### Deprecation: Remove RuntimeCliOptions Variants

```python
# DELETE from ide_handlers.py:
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None

# DELETE from subsystem_handlers.py:
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None

# DELETE from datasets_handlers.py:
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None

# KEEP in common_handlers.py as ALIAS (deprecated):
RuntimeCliOptions = RuntimeParams  # Backward compat alias
```

---

### 3.3 RenderingService

**Location**: `cli/rendering/` (new package)

**Responsibility**: Single path for all CLI output with format negotiation, streaming, and metadata.

#### Core Types

```python
# cli/rendering/types.py

from enum import Enum
from dataclasses import dataclass, field
from typing import Literal


class OutputFormat(Enum):
    """Canonical output formats."""
    TEXT = "text"      # Human-readable, may include colors
    JSON = "json"      # Structured JSON object
    JSONL = "jsonl"    # JSON Lines for streaming


@dataclass(frozen=True)
class RenderContext:
    """Context for rendering operations.
    
    Determines how output is formatted based on environment and user preference.
    
    Attributes
    ----------
    format
        Output format preference.
    color
        Whether to use ANSI color codes.
    writer
        Primary output stream (stdout).
    err_writer
        Error/warning stream (stderr).
    is_tty
        Whether output is a terminal (affects defaults).
    """
    format: OutputFormat
    color: bool
    writer: TextIO = field(default_factory=lambda: sys.stdout)
    err_writer: TextIO = field(default_factory=lambda: sys.stderr)
    is_tty: bool = field(default=True)
    
    @classmethod
    def auto_detect(
        cls,
        format_override: OutputFormat | None = None,
        color_override: bool | None = None,
    ) -> RenderContext:
        """Create context with auto-detection.
        
        TTY detection determines color default. JSON format disables color.
        """
        is_tty = sys.stdout.isatty()
        fmt = format_override or (OutputFormat.TEXT if is_tty else OutputFormat.JSON)
        color = color_override if color_override is not None else (is_tty and fmt == OutputFormat.TEXT)
        
        return cls(format=fmt, color=color, is_tty=is_tty)
    
    @classmethod
    def for_testing(cls) -> tuple[RenderContext, StringIO, StringIO]:
        """Create context with captured output for testing."""
        out = StringIO()
        err = StringIO()
        return cls(format=OutputFormat.TEXT, color=False, writer=out, err_writer=err, is_tty=False), out, err


JustifyMethod = Literal["left", "center", "right", "full", "default"]


@dataclass(frozen=True)
class ColumnSpec:
    """Table column specification.
    
    Attributes
    ----------
    key
        Dictionary key to extract from row data.
    header
        Column header text.
    style
        Rich style for the column (ignored in plain mode).
    justify
        Text justification.
    width
        Fixed column width (None for auto).
    """
    key: str
    header: str
    style: str | None = None
    justify: JustifyMethod = "left"
    width: int | None = None


@dataclass(frozen=True)
class TableSpec:
    """Table rendering specification.
    
    Attributes
    ----------
    columns
        Column specifications.
    title
        Optional table title.
    caption
        Optional table caption (footer).
    show_row_numbers
        Whether to show row numbers.
    empty_message
        Message when table has no rows.
    """
    columns: tuple[ColumnSpec, ...]
    title: str | None = None
    caption: str | None = None
    show_row_numbers: bool = False
    empty_message: str = "No data."
```

#### Renderer Protocol and Implementation

```python
# cli/rendering/service.py

from typing import Protocol, TypeVar, Sequence

T = TypeVar("T")


class RenderingService(Protocol):
    """Protocol for CLI output rendering.
    
    All handlers delegate output to this protocol. Implementations handle
    format negotiation, TTY detection, and consistent formatting.
    """
    
    def render_result(self, result: CliResult[T]) -> int:
        """Render a CLI result and return exit code.
        
        Handles:
        - Warning emission to stderr
        - Error rendering with Problem Details
        - Data rendering in appropriate format
        - Metadata inclusion in JSON output
        
        Returns
        -------
        int
            Exit code: 0 for success, non-zero for failure.
        """
        ...
    
    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data.
        
        In TEXT format: Rich table (TTY) or ASCII table (non-TTY)
        In JSON format: Array of row objects
        In JSONL format: One JSON object per line
        """
        ...
    
    def render_error(self, error: ProblemDetail) -> None:
        """Render error with RFC 9457 Problem Details.
        
        Always writes to err_writer (stderr).
        """
        ...
    
    def render_message(self, message: str, *, level: str = "info") -> None:
        """Render a simple message.
        
        Levels: "info", "success", "warning", "error"
        """
        ...
    
    def emit_progress(
        self,
        current: int,
        total: int,
        message: str | None = None,
    ) -> None:
        """Emit progress update.
        
        In TEXT format: Updates progress bar (if TTY)
        In JSONL format: Emits progress JSON object
        In JSON format: No-op (batch result includes progress)
        """
        ...


class UnifiedRenderer:
    """Single implementation of RenderingService.
    
    Consolidates:
    - RichRenderer from cli_render.py
    - PlainRenderer from cli_render.py
    - StreamingRenderer from pipelines.py
    - CliResult.render() logic from results.py
    
    Design Notes
    ------------
    - Handlers NEVER import sys or write to stdout/stderr directly
    - All output flows through this class
    - JSON output includes metadata and warnings for parseability
    - Rich is only used when color=True and format=TEXT
    """
    
    def __init__(self, ctx: RenderContext):
        self._ctx = ctx
        self._console = Console(theme=CODEINTEL_THEME) if ctx.color else None
    
    def render_result(self, result: CliResult[T]) -> int:
        """Primary entry point for handler result rendering."""
        # 1. Emit warnings to stderr
        for warning in result.warnings:
            self._emit_warning(warning)
        
        # 2. Handle failure case
        if not result.success:
            if result.error:
                self.render_error(result.error)
            return 1
        
        # 3. Handle success case
        if result.data is not None:
            self._render_data(result.data, result.metadata)
        
        return 0
    
    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data with format negotiation."""
        if self._ctx.format == OutputFormat.JSON:
            self._write_json([dict(row) for row in rows])
        elif self._ctx.format == OutputFormat.JSONL:
            for row in rows:
                self._write_json(dict(row))
        elif self._console is not None:
            self._render_rich_table(rows, spec)
        else:
            self._render_plain_table(rows, spec)
    
    # ... internal methods ...
```

#### Usage Pattern

```python
# In cyclopts command:

@build_app.command()
def status(runtime: RuntimeCLI = runtime_field(), output: OutputCLI = output_field()) -> int:
    """Show build status."""
    # 1. Resolve runtime
    params = RuntimeParams.from_cyclopts(runtime)
    resolved = RuntimeResolver.resolve(params)
    
    # 2. Create handler context
    ctx = HandlerContext(
        config=ConfigService.load().config,
        runtime=resolved,
        params={},
        logger=LOG,
    )
    
    # 3. Execute handler (returns CliResult)
    result = build_status_handler(ctx)
    
    # 4. Render result
    render_ctx = RenderContext.auto_detect(
        format_override=_resolve_format(output),
    )
    return UnifiedRenderer(render_ctx).render_result(result)


# Handler (NEVER writes to stdout):

def build_status_handler(ctx: HandlerContext) -> CliResult[BuildStatusData]:
    """Get build status.
    
    Returns
    -------
    CliResult[BuildStatusData]
        Status data. Rendering handled by caller.
    """
    try:
        status = _compute_status(ctx.runtime, ctx.gateway)
        return CliResult.ok(status)
    except BuildStateError as exc:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:build/state-error",
                title="Build State Error",
                detail=str(exc),
            )
        )
```

---

### 3.4 Handler Protocol

**Location**: `cli/handlers/protocol.py`

**Responsibility**: Define the contract for all CLI handlers.

#### Protocol Definition

```python
# cli/handlers/protocol.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, Mapping

if TYPE_CHECKING:
    from codeintel.analytics.runtime import GraphRuntime
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.cli.resolution.gateway import GatewayManager
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@dataclass
class HandlerContext:
    """Unified context for all CLI handlers.
    
    Every handler receives this context. It provides:
    - Resolved configuration
    - Resolved runtime (project, paths, snapshot)
    - Operation-specific parameters
    - Lazy access to gateway and graph runtime
    
    Handlers should NOT:
    - Import sys
    - Write to stdout/stderr
    - Open their own gateways
    - Load their own config
    
    Attributes
    ----------
    config
        CLI configuration (from ConfigService).
    runtime
        Resolved project runtime.
    params
        Operation-specific parameters (not runtime params).
    verbosity
        Verbosity level (0=warnings, 1=info, 2+=debug).
    
    Examples
    --------
    >>> def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    ...     ctx.logger.info("Starting operation")
    ...     rows = ctx.gateway.execute("SELECT * FROM table")
    ...     return CliResult.ok(MyData(rows=rows))
    """
    config: CliConfig
    runtime: ResolvedRuntime
    params: Mapping[str, object]
    verbosity: int = 0
    
    _gateway_manager: GatewayManager | None = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this handler."""
        return logging.getLogger(f"codeintel.cli.handlers.{self._operation_name}")
    
    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (lazy).
        
        Gateway is opened on first access. The context manages lifecycle.
        """
        if self._gateway_manager is None:
            self._gateway_manager = GatewayManager(self.runtime)
        return self._gateway_manager.gateway
    
    @property
    def graph_runtime(self) -> GraphRuntime:
        """Get graph runtime (lazy)."""
        if self._gateway_manager is None:
            self._gateway_manager = GatewayManager(self.runtime)
        return self._gateway_manager.graph_runtime
    
    @property
    def db_path(self) -> Path:
        """Shortcut to database path."""
        return self.runtime.db_path
    
    @property
    def output_format(self) -> str:
        """Get output format from config."""
        return self.config.output_format
    
    def close(self) -> None:
        """Close managed resources.
        
        Called automatically by execution infrastructure.
        """
        if self._gateway_manager is not None:
            self._gateway_manager.close()


class HandlerProtocol(Protocol[T]):
    """Protocol for CLI handler functions.
    
    All handlers must:
    1. Accept HandlerContext as their only argument
    2. Return CliResult[T] (never None, never raise for expected errors)
    3. Never write to stdout/stderr directly
    4. Never call sys.exit()
    
    Unexpected exceptions (bugs) may propagate; expected errors
    should return CliResult.fail() with appropriate ProblemDetail.
    """
    
    def __call__(self, ctx: HandlerContext) -> CliResult[T]:
        """Execute the handler.
        
        Parameters
        ----------
        ctx
            Handler context with config, runtime, params.
            
        Returns
        -------
        CliResult[T]
            Success or failure result. Never None.
        """
        ...
```

#### Handler Migration Pattern

Before (old pattern):

```python
# ide_handlers.py (OLD)

@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None

@dataclass(frozen=True)
class IdeHintsOptions:
    rel_path: str
    runtime_options: RuntimeCliOptions
    verbose: int = 0

def ide_hints_handler(options: IdeHintsOptions) -> CliResult[dict[str, Any]]:
    setup_logging(options.verbose)
    runtime = build_runtime_from_cli(options.runtime_options)
    gateway = open_gateway(...)
    # ... business logic ...
    sys.stdout.write(json.dumps(result))  # BAD: direct write
    return CliResult.ok(result)
```

After (new pattern):

```python
# cli/handlers/ide.py (NEW)

def ide_hints_handler(ctx: HandlerContext) -> CliResult[IdeHintsData]:
    """Generate IDE hints for a file.
    
    Parameters
    ----------
    ctx
        Handler context. Expects ctx.params["rel_path"] to be set.
        
    Returns
    -------
    CliResult[IdeHintsData]
        IDE hints data for rendering.
    """
    rel_path = ctx.params.get("rel_path")
    if not rel_path:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="rel_path is required",
            )
        )
    
    # Use ctx.gateway (lazy, managed)
    # Use ctx.graph_runtime (lazy, managed)
    # ctx.logger for logging
    
    hints = _compute_hints(ctx.gateway, rel_path)
    return CliResult.ok(IdeHintsData(hints=hints))


# cyclopts_ide.py (wiring)

@ide_app.command()
def hints(
    path: str,
    runtime: RuntimeCLI = runtime_field(),
    output: OutputCLI = output_field(),
) -> int:
    """Get IDE hints for a file."""
    # Build context
    ctx = _build_handler_context(runtime, output, params={"rel_path": path})
    
    # Execute handler
    result = ide_hints_handler(ctx)
    
    # Render result
    return _render_result(result, output)
```

---

### 3.5 Cyclopts Integration

**Location**: `cli/cyclopts_common.py` (simplified), `cli/cyclopts_app.py`

**Responsibility**: Thin wiring between Cyclopts parameter parsing and unified infrastructure.

#### Cyclopts RuntimeCLI (Keep, Simplified)

```python
# cli/cyclopts_common.py

@dataclass
class RuntimeCLI:
    """Cyclopts runtime selection flags.
    
    This is a Cyclopts-specific dataclass with Parameter annotations.
    It is NOT the canonical runtime params type.
    
    Use RuntimeParams.from_cyclopts(cli) to convert to canonical type.
    """
    project_root: Annotated[Path | None, Parameter(name=["--root", "-r"])] = None
    repo: Annotated[str | None, Parameter(name="--repo")] = None
    commit: Annotated[str | None, Parameter(name="--commit")] = None
    db_path: Annotated[Path | None, Parameter(name="--db-path")] = None
    build_dir: Annotated[Path | None, Parameter(name="--build-dir")] = None
    repo_root: Annotated[Path | None, Parameter(name="--repo-root")] = None
    document_output_dir: Annotated[Path | None, Parameter(name="--document-output-dir")] = None
    verbose: Annotated[int, Parameter(name=["--verbose", "-v"], count=True)] = 0


@dataclass
class OutputCLI:
    """Cyclopts output format flags."""
    output_format: Annotated[OutputFormat, Parameter(name="--output-format")] = OutputFormat.TEXT
    json: Annotated[bool, Parameter(name="--json", negative=())] = False


def runtime_field() -> RuntimeCLI:
    """Create RuntimeCLI field with Cyclopts metadata."""
    return field(default_factory=RuntimeCLI, metadata={"parameter": Parameter(name="*")})


def output_field() -> OutputCLI:
    """Create OutputCLI field with Cyclopts metadata."""
    return field(default_factory=OutputCLI, metadata={"parameter": Parameter(name="*")})
```

#### Standard Command Wiring Pattern

```python
# cli/cyclopts_build.py

from codeintel.cli.config import ConfigService
from codeintel.cli.resolution import RuntimeResolver, RuntimeParams
from codeintel.cli.rendering import UnifiedRenderer, RenderContext
from codeintel.cli.handlers.protocol import HandlerContext
from codeintel.cli.handlers.build import build_run_handler
from codeintel.cli.cyclopts_common import RuntimeCLI, OutputCLI, runtime_field, output_field

build_app = App(name="build", help="Build operations.")


@build_app.command()
def run(
    runtime: RuntimeCLI = runtime_field(),
    output: OutputCLI = output_field(),
    targets: list[str] | None = None,
    force: bool = False,
) -> int:
    """Run build targets."""
    # 1. Load config (once per command)
    config_service = ConfigService.load()
    
    # 2. Resolve runtime
    params = RuntimeParams.from_cyclopts(runtime)
    resolved = RuntimeResolver.resolve(params)
    
    # 3. Build handler context
    ctx = HandlerContext(
        config=config_service.config,
        runtime=resolved,
        params={"targets": targets, "force": force},
        verbosity=runtime.verbose,
    )
    
    # 4. Setup logging based on verbosity
    setup_logging(ctx.verbosity, config=ctx.config)
    
    try:
        # 5. Execute handler (returns CliResult)
        result = build_run_handler(ctx)
    finally:
        # 6. Cleanup resources
        ctx.close()
    
    # 7. Render result
    render_ctx = RenderContext.auto_detect(
        format_override=_resolve_format(output),
    )
    return UnifiedRenderer(render_ctx).render_result(result)


def _resolve_format(output: OutputCLI) -> OutputFormat:
    """Resolve output format with --json flag override."""
    if output.json:
        return OutputFormat.JSON
    return output.output_format
```

#### Shared Wiring Helper (Optional)

```python
# cli/cyclopts_common.py (new helper)

@contextmanager
def command_context(
    runtime: RuntimeCLI,
    output: OutputCLI,
    params: dict[str, object],
) -> Iterator[tuple[HandlerContext, UnifiedRenderer]]:
    """Standard context manager for command wiring.
    
    Handles:
    - Config loading
    - Runtime resolution
    - Logging setup
    - Resource cleanup
    - Renderer creation
    
    Usage
    -----
    @app.command()
    def my_command(runtime: RuntimeCLI, output: OutputCLI, ...) -> int:
        with command_context(runtime, output, {"key": value}) as (ctx, renderer):
            result = my_handler(ctx)
            return renderer.render_result(result)
    """
    config_service = ConfigService.load()
    resolved = RuntimeResolver.resolve(RuntimeParams.from_cyclopts(runtime))
    
    ctx = HandlerContext(
        config=config_service.config,
        runtime=resolved,
        params=params,
        verbosity=runtime.verbose,
    )
    
    setup_logging(ctx.verbosity, config=ctx.config)
    
    render_ctx = RenderContext.auto_detect(
        format_override=OutputFormat.JSON if output.json else output.output_format,
    )
    renderer = UnifiedRenderer(render_ctx)
    
    try:
        yield ctx, renderer
    finally:
        ctx.close()
```

---

## 4. Data Flow Contracts

### 4.1 Command Execution Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Cyclopts Command Function                                       │
│                                                                 │
│  1. Extract Cyclopts params (RuntimeCLI, OutputCLI, etc.)      │
│  2. Convert to canonical types (RuntimeParams)                  │
│  3. Load config (ConfigService.load())                          │
│  4. Resolve runtime (RuntimeResolver.resolve())                 │
│  5. Create HandlerContext                                       │
│  6. Call handler function                                       │
│  7. Render result (UnifiedRenderer)                            │
│  8. Return exit code                                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Handler Function                                                │
│                                                                 │
│  Input:  HandlerContext (config, runtime, params, logger)      │
│  Output: CliResult[T]                                          │
│                                                                 │
│  • Access gateway via ctx.gateway (lazy)                       │
│  • Access graph_runtime via ctx.graph_runtime (lazy)           │
│  • Log via ctx.logger                                          │
│  • Return CliResult.ok(data) or CliResult.fail(error)         │
│  • NEVER write to stdout/stderr                                │
│  • NEVER call sys.exit()                                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ UnifiedRenderer                                                 │
│                                                                 │
│  Input:  CliResult[T], RenderContext                           │
│  Output: int (exit code)                                       │
│                                                                 │
│  • Emit warnings to stderr                                     │
│  • Render error (if failed) with Problem Details               │
│  • Render data in requested format (TEXT/JSON/JSONL)          │
│  • Include metadata in JSON output                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
stdout / stderr
```

### 4.2 Type Flow

```
RuntimeCLI (Cyclopts)
    │
    │ RuntimeParams.from_cyclopts()
    ▼
RuntimeParams (Canonical)
    │
    │ RuntimeResolver.resolve()
    ▼
ResolvedRuntime (Immutable Result)
    │
    │ HandlerContext construction
    ▼
HandlerContext
    │
    │ Handler execution
    ▼
CliResult[T]
    │
    │ UnifiedRenderer.render_result()
    ▼
Exit Code (int)
```

---

## 5. Error Handling Strategy

### 5.1 Error Categories

| Category | Exception Type | Handling |
|----------|---------------|----------|
| Resolution Failure | `ResolutionError` | Convert to CliResult.fail() at command level |
| Config Validation | `ConfigLoadError` | Convert to CliResult.fail() at command level |
| Business Logic Error | Return `CliResult.fail()` | Handler returns, no exception |
| Unexpected Bug | Any exception | Propagate, logged by Cyclopts |

### 5.2 ResolutionError Structure

```python
@dataclass
class ResolutionError(Exception):
    """Runtime resolution failed.
    
    Attributes
    ----------
    message
        Human-readable error message.
    missing_params
        List of missing required parameters (if applicable).
    suggestion
        Actionable fix suggestion.
    """
    message: str
    missing_params: list[str] = field(default_factory=list)
    suggestion: str | None = None
    
    def to_problem_detail(self) -> ProblemDetail:
        """Convert to RFC 9457 Problem Detail for rendering."""
        return ProblemDetail(
            type="urn:codeintel:cli:resolution/failed",
            title="Runtime Resolution Failed",
            detail=self.message,
            extensions={
                "missing_params": self.missing_params,
                "suggestion": self.suggestion,
            },
        )
```

### 5.3 Handler Error Pattern

```python
def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    """Handler example with error handling."""
    
    # Validation errors → CliResult.fail()
    if not ctx.params.get("required_param"):
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="required_param is required",
            )
        )
    
    # Business logic errors → CliResult.fail()
    try:
        data = _do_business_logic(ctx)
    except BusinessRuleViolation as exc:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:my-domain/rule-violation",
                title="Business Rule Violation",
                detail=str(exc),
            )
        )
    
    # Unexpected errors → Let propagate (bug)
    # Don't catch Exception broadly
    
    return CliResult.ok(data)
```

---

## 6. Directory Structure

### 6.1 Target Structure

```
cli/
├── __init__.py                    # Public API exports
├── cyclopts_app.py               # Root app construction
├── cyclopts_common.py            # SIMPLIFIED: RuntimeCLI, OutputCLI, helpers
├── cyclopts_*.py                 # Command wiring (thin)
│
├── config/                       # SINGLE SOURCE: Configuration
│   ├── __init__.py              # Public API
│   ├── service.py               # ConfigService (NEW)
│   ├── model.py                 # CliConfig, nested configs
│   ├── loader.py                # load_config() implementation
│   ├── env.py                   # Environment variable mapping
│   ├── validation.py            # Config validation
│   └── schema.py                # JSON Schema generation
│
├── resolution/                   # SINGLE SOURCE: Runtime resolution
│   ├── __init__.py              # Public API
│   ├── runtime.py               # RuntimeResolver, RuntimeParams
│   ├── gateway.py               # GatewayManager
│   ├── types.py                 # ResolvedRuntime
│   └── errors.py                # ResolutionError
│
├── rendering/                    # SINGLE SOURCE: Output (NEW)
│   ├── __init__.py              # Public API
│   ├── service.py               # UnifiedRenderer
│   ├── types.py                 # RenderContext, OutputFormat
│   ├── table.py                 # TableSpec, ColumnSpec
│   └── progress.py              # Progress bar utilities
│
├── handlers/                     # Business logic
│   ├── __init__.py
│   ├── protocol.py              # HandlerProtocol, HandlerContext
│   ├── base.py                  # setup_logging (KEEP)
│   ├── build.py                 # Build handlers
│   ├── datasets.py              # Dataset handlers
│   ├── docs.py                  # Documentation handlers
│   ├── graphs.py                # Graph handlers
│   ├── ide.py                   # IDE integration handlers
│   ├── ops.py                   # Operation handlers
│   ├── storage.py               # Storage handlers
│   └── subsystem.py             # Subsystem handlers
│
├── execution/                    # Existing: ExecutionContext
├── operations/                   # Existing: Operation definitions
├── plugins/                      # Existing: Plugin system
├── completions/                  # Existing: Shell completions
│
├── results.py                    # CliResult (delegates to rendering)
├── cli_types.py                  # SIMPLIFIED: Remove RuntimeOptions
├── cli_errors.py                 # ProblemDetail, ValidationError
├── error_taxonomy.py             # Error classification
├── pipelines.py                  # REFACTORED: Use rendering/
└── ...                           # Other existing files
```

### 6.2 Deleted/Deprecated Files

| File | Action | Replacement |
|------|--------|-------------|
| `common_handlers.py` | DELETE | `resolution/`, `handlers/protocol.py` |
| `cli_render.py` | DELETE | `rendering/service.py` |
| `*_handlers.py` (top-level) | MIGRATE | `handlers/*.py` |

### 6.3 File Movement Plan

```
# Phase 1: New packages (additive)
NEW: cli/config/service.py
NEW: cli/rendering/__init__.py
NEW: cli/rendering/service.py
NEW: cli/rendering/types.py
NEW: cli/rendering/table.py
NEW: cli/handlers/protocol.py

# Phase 2: Migrate handlers
MOVE: ide_handlers.py       → handlers/ide.py
MOVE: subsystem_handlers.py → handlers/subsystem.py
MOVE: datasets_handlers.py  → handlers/datasets.py
MOVE: build_handlers.py     → handlers/build.py
MOVE: docs_handlers.py      → handlers/docs.py
MOVE: graphs_handlers.py    → handlers/graphs.py
MOVE: ops_handlers.py       → handlers/ops.py
MOVE: storage_handlers.py   → handlers/storage.py

# Phase 3: Cleanup
DELETE: common_handlers.py
DELETE: cli_render.py
SIMPLIFY: cyclopts_common.py (remove config loading)
SIMPLIFY: cli_types.py (RuntimeOptions → RuntimeParams alias)
```

---

## 7. Migration Strategy

### 7.1 Phase Overview

| Phase | Focus | Risk | Deliverable |
|-------|-------|------|-------------|
| 1 | ConfigService | Low | Unified config loading |
| 2 | RuntimeResolver | Medium | Single runtime resolution |
| 3 | RenderingService | Medium | Unified output pipeline |
| 4 | Handler Migration | High | New handler pattern |
| 5 | Cleanup | Low | Remove deprecated code |

### 7.2 Phase 1: ConfigService

**Goal**: Unify configuration loading without breaking existing commands.

**Steps**:
1. Create `cli/config/service.py` with `ConfigService` class
2. Update `make_root_app()` to use `ConfigService.get_cyclopts_config_chain()`
3. Remove duplicate `CONFIG_ENV_PREFIX` from `cyclopts_common.py` and `cyclopts_config.py`
4. Add deprecation warnings to old config loading functions

**Acceptance Criteria**:
- All commands load config through ConfigService
- TOML and YAML config files both work
- Environment variables apply correctly
- Existing tests pass

### 7.3 Phase 2: RuntimeResolver

**Goal**: Consolidate runtime resolution to single implementation.

**Steps**:
1. Enhance `cli/resolution/runtime.py` with `RuntimeParams` type
2. Add factory methods: `from_context()`, `from_cyclopts()`, `minimal()`
3. Update `RuntimeResolver.resolve()` to accept `RuntimeParams`
4. Create `RuntimeCliOptions = RuntimeParams` alias in `common_handlers.py`
5. Migrate handlers one at a time:
   - Start with `ide_handlers.py` (simplest)
   - Then `subsystem_handlers.py`
   - Then `build_handlers.py`
   - Then `datasets_handlers.py` (most complex)
6. Delete per-module `RuntimeCliOptions` classes
7. Delete per-module `build_runtime_from_cli` functions

**Acceptance Criteria**:
- Single `RuntimeParams` type used everywhere
- Single `RuntimeResolver.resolve()` implementation
- All existing tests pass
- New tests for resolution edge cases

### 7.4 Phase 3: RenderingService

**Goal**: Unified output pipeline.

**Steps**:
1. Create `cli/rendering/` package with types and service
2. Implement `UnifiedRenderer` consolidating all rendering logic
3. Migrate handlers to return `CliResult` (if not already)
4. Replace `sys.stdout.write()` calls one file at a time:
   - Start with `cyclopts_plugins.py` (isolated)
   - Then `cyclopts_jobs.py`
   - Then `graphs_handlers.py`
   - Then `docs_handlers.py`
5. Delete `cli_render.py` after all migrations complete
6. Refactor `pipelines.py` to use `UnifiedRenderer` for streaming

**Acceptance Criteria**:
- No direct `sys.stdout.write()` in handlers
- JSON output includes metadata and warnings
- TTY detection works correctly
- Streaming mode works for batch operations

### 7.5 Phase 4: Handler Migration

**Goal**: Move handlers to new pattern with `HandlerContext`.

**Steps**:
1. Create `cli/handlers/protocol.py` with `HandlerContext` and `HandlerProtocol`
2. Create `cli/handlers/*.py` files (empty)
3. Migrate one handler file at a time:
   - Extract pure handler functions
   - Update to accept `HandlerContext`
   - Update to return `CliResult`
   - Update cyclopts command to use new handler
4. Update cyclopts commands to use `command_context()` helper

**Acceptance Criteria**:
- All handlers follow `HandlerProtocol`
- All handlers receive `HandlerContext`
- All handlers return `CliResult`
- No handler imports `sys`

### 7.6 Phase 5: Cleanup

**Goal**: Remove deprecated code.

**Steps**:
1. Delete `common_handlers.py`
2. Remove `RuntimeCliOptions` alias (breaking change, major version)
3. Remove deprecated functions with warnings
4. Update documentation
5. Update CHANGELOG

**Acceptance Criteria**:
- No deprecated code remains
- All imports updated
- Documentation current

---

## 8. Additional Architectural Details

### 8.1 CliResult Contract

The `CliResult[T]` type is the **mandatory return type** for all handlers. It encapsulates success/failure, data, warnings, and metadata.

```python
# cli/results.py (enhanced)

@dataclass
class CliResult[T]:
    """Structured result from a CLI handler.
    
    Invariants
    ----------
    - success=True implies data may be set, error is None
    - success=False implies error is set, data is None
    - warnings can be present in both success and failure cases
    - metadata is always included in JSON output
    
    Attributes
    ----------
    success
        Whether the operation completed successfully.
    data
        Result payload (generic type T). None for failures.
    error
        ProblemDetail for failures. None for success.
    warnings
        Non-fatal warnings to display.
    metadata
        Timing, counts, and other metadata for observability.
    """
    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    
    # --- Factory Methods ---
    
    @classmethod
    def ok(
        cls,
        data: T,
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CliResult[T]:
        """Create successful result."""
        return cls(
            success=True,
            data=data,
            warnings=warnings or [],
            metadata=metadata or {},
        )
    
    @classmethod
    def fail(
        cls,
        error: ProblemDetail,
        *,
        warnings: list[str] | None = None,
    ) -> CliResult[T]:
        """Create failed result."""
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
        )
    
    @classmethod
    def empty(cls) -> CliResult[None]:
        """Create successful result with no data (for void operations)."""
        return cls(success=True, data=None)
    
    # --- Composition ---
    
    def with_warning(self, warning: str) -> CliResult[T]:
        """Add a warning (returns new instance)."""
        return replace(self, warnings=[*self.warnings, warning])
    
    def with_metadata(self, key: str, value: object) -> CliResult[T]:
        """Add metadata entry (returns new instance)."""
        return replace(self, metadata={**self.metadata, key: value})
    
    # --- Serialization ---
    
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.
        
        Structure matches RFC 9457 Problem Details envelope.
        """
        result: dict[str, object] = {"success": self.success}
        
        if self.data is not None:
            result["data"] = self._serialize_data(self.data)
        
        if self.error is not None:
            result["error"] = self.error.to_dict()
        
        if self.warnings:
            result["warnings"] = self.warnings
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
```

### 8.2 Testing Patterns

Handlers are designed for easy testing without I/O mocking.

#### Testing a Handler

```python
# tests/cli/handlers/test_build.py

import pytest
from codeintel.cli.handlers.build import build_status_handler
from codeintel.cli.handlers.protocol import HandlerContext
from tests._helpers.context import TestContext


def test_build_status_success(test_ctx: TestContext) -> None:
    """Test build status with valid project."""
    # Arrange: Use test infrastructure to create context
    test_ctx.require(CORE_PACK)
    
    handler_ctx = HandlerContext(
        config=test_ctx.config,
        runtime=test_ctx.resolved_runtime,
        params={},
        verbosity=0,
    )
    
    # Act: Call handler directly
    result = build_status_handler(handler_ctx)
    
    # Assert: Check CliResult
    assert result.success
    assert result.data is not None
    assert result.data.total_targets > 0


def test_build_status_no_database(tmp_path: Path) -> None:
    """Test build status with missing database."""
    # Arrange: Create context with non-existent DB
    runtime = _create_runtime_with_missing_db(tmp_path)
    handler_ctx = HandlerContext(
        config=CliConfig(),
        runtime=runtime,
        params={},
        verbosity=0,
    )
    
    # Act
    result = build_status_handler(handler_ctx)
    
    # Assert: Expect failure with specific error type
    assert not result.success
    assert result.error is not None
    assert result.error.type == "urn:codeintel:cli:storage/database-not-found"
```

#### Testing Rendering

```python
# tests/cli/rendering/test_service.py

def test_render_table_json_format() -> None:
    """Test table rendering in JSON format."""
    # Arrange
    ctx, out, err = RenderContext.for_testing()
    ctx = replace(ctx, format=OutputFormat.JSON)
    renderer = UnifiedRenderer(ctx)
    
    rows = [{"name": "foo", "count": 10}, {"name": "bar", "count": 20}]
    spec = TableSpec(columns=(
        ColumnSpec("name", "Name"),
        ColumnSpec("count", "Count"),
    ))
    
    # Act
    renderer.render_table(rows, spec)
    
    # Assert
    output = json.loads(out.getvalue())
    assert len(output) == 2
    assert output[0]["name"] == "foo"


def test_render_result_with_warnings() -> None:
    """Test that warnings go to stderr."""
    # Arrange
    ctx, out, err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)
    
    result = CliResult.ok({"key": "value"}).with_warning("Something to note")
    
    # Act
    exit_code = renderer.render_result(result)
    
    # Assert
    assert exit_code == 0
    assert "Something to note" in err.getvalue()
    assert "key" in out.getvalue()
```

### 8.3 Exit Code Mapping

| CliResult State | Exit Code | Meaning |
|-----------------|-----------|---------|
| `success=True` | 0 | Operation succeeded |
| `success=False`, error.status < 500 | 1 | User/input error |
| `success=False`, error.status >= 500 | 2 | Internal error |
| Uncaught exception | 3 | Bug in handler |

```python
# cli/rendering/service.py

def _exit_code_from_result(result: CliResult[object]) -> int:
    """Map CliResult to exit code."""
    if result.success:
        return 0
    if result.error and result.error.status >= 500:
        return 2  # Internal error
    return 1  # User error
```

### 8.4 Progress and Streaming Contract

For long-running operations, handlers can emit progress through metadata or use streaming mode.

#### Batch Mode (Default)

Handler returns single result with progress in metadata:

```python
def build_run_handler(ctx: HandlerContext) -> CliResult[BuildResult]:
    """Run build with progress tracking."""
    targets = ctx.params.get("targets", [])
    results = []
    
    for i, target in enumerate(targets):
        # Progress tracked in metadata (no streaming)
        result = _build_target(ctx, target)
        results.append(result)
    
    return CliResult.ok(
        BuildResult(targets=results),
        metadata={
            "total_targets": len(targets),
            "completed": len(results),
            "duration_seconds": elapsed,
        },
    )
```

#### Streaming Mode (JSONL)

For operations that should emit results incrementally:

```python
# cli/rendering/streaming.py

class StreamingEmitter:
    """Emit results as JSON Lines during execution.
    
    Use for:
    - Batch operations with many items
    - Long-running operations where partial results are useful
    - Piping to other tools (jq, etc.)
    """
    
    def __init__(self, writer: TextIO):
        self._writer = writer
    
    def emit_item(self, item: object) -> None:
        """Emit a single item as JSON line."""
        self._writer.write(json.dumps(item, default=str))
        self._writer.write("\n")
        self._writer.flush()
    
    def emit_progress(self, current: int, total: int, message: str = "") -> None:
        """Emit progress marker."""
        self.emit_item({
            "type": "progress",
            "current": current,
            "total": total,
            "message": message,
        })
    
    def emit_summary(self, summary: dict[str, object]) -> None:
        """Emit final summary."""
        self.emit_item({"type": "summary", **summary})
```

### 8.5 ExecutionContext vs HandlerContext

The codebase has an existing `ExecutionContext` in `cli/execution/context.py`. Here's how it relates to `HandlerContext`:

| Aspect | ExecutionContext | HandlerContext |
|--------|-----------------|----------------|
| Scope | Operation lifecycle | Handler invocation |
| Created by | Execution infrastructure | Command wiring |
| Contains | operation_id, params dict, lifecycle hooks | config, runtime, typed params |
| Gateway access | Via params lookup | Via lazy property |
| Thread safety | Mutable (state machine) | Immutable-ish (lazy init) |

**Relationship**: `ExecutionContext` is lower-level infrastructure. `HandlerContext` wraps it for handler convenience:

```python
# cli/handlers/protocol.py

@dataclass
class HandlerContext:
    """Handler-level context wrapping ExecutionContext."""
    
    config: CliConfig
    runtime: ResolvedRuntime
    params: Mapping[str, object]
    verbosity: int = 0
    
    _execution_ctx: ExecutionContext | None = None  # Optional back-reference
    
    @classmethod
    def from_execution_context(
        cls,
        exec_ctx: ExecutionContext,
        config: CliConfig,
        runtime: ResolvedRuntime,
    ) -> HandlerContext:
        """Create HandlerContext from ExecutionContext."""
        return cls(
            config=config,
            runtime=runtime,
            params=exec_ctx.params,
            _execution_ctx=exec_ctx,
        )
```

### 8.6 Pre-built TableSpecs

Common table specifications are defined centrally for consistency:

```python
# cli/rendering/specs.py

from codeintel.cli.rendering.types import TableSpec, ColumnSpec

# Operations listing
OPERATIONS_TABLE = TableSpec(
    columns=(
        ColumnSpec("id", "Operation ID", style="cyan"),
        ColumnSpec("summary", "Summary"),
        ColumnSpec("tags", "Tags", style="dim"),
    ),
    title="Available Operations",
)

# Dataset listing
DATASETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("table_key", "Table", style="cyan"),
        ColumnSpec("name", "Name"),
        ColumnSpec("row_count", "Rows", justify="right"),
        ColumnSpec("description", "Description", style="dim"),
    ),
    title="Datasets",
)

# Build targets
BUILD_TARGETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Target", style="cyan"),
        ColumnSpec("module", "Module"),
        ColumnSpec("status", "Status"),
        ColumnSpec("duration", "Duration", justify="right"),
    ),
    title="Build Targets",
)

# Plugin listing
PLUGINS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Plugin", style="cyan"),
        ColumnSpec("version", "Version"),
        ColumnSpec("status", "Status"),
        ColumnSpec("capabilities", "Capabilities", style="dim"),
    ),
    title="Installed Plugins",
)

# Jobs listing
JOBS_TABLE = TableSpec(
    columns=(
        ColumnSpec("job_id", "Job ID", style="cyan"),
        ColumnSpec("operation", "Operation"),
        ColumnSpec("status", "Status"),
        ColumnSpec("created_at", "Created", style="dim"),
    ),
    title="Jobs",
)
```

### 8.7 Logging Contract

Handlers use the logger from `HandlerContext` with these conventions:

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Internal state, loop iterations | `ctx.logger.debug("Processing row %d", i)` |
| INFO | Major milestones | `ctx.logger.info("Build started for %d targets", n)` |
| WARNING | Recoverable issues | `ctx.logger.warning("Skipping invalid entry: %s", name)` |
| ERROR | Never in handlers | Use `CliResult.fail()` instead |

```python
def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    """Handler with proper logging."""
    ctx.logger.info("Starting operation")
    
    for item in items:
        ctx.logger.debug("Processing: %s", item.name)
        
        if item.is_invalid:
            ctx.logger.warning("Skipping invalid item: %s", item.name)
            continue
        
        # Process item...
    
    ctx.logger.info("Completed: %d items processed", len(items))
    return CliResult.ok(MyData(count=len(items)))
```

### 8.8 Backward Compatibility Layer

During migration, maintain backward compatibility with deprecation warnings:

```python
# cli/common_handlers.py (during migration)

import warnings
from codeintel.cli.resolution import RuntimeParams, RuntimeResolver

# Alias for backward compatibility
RuntimeCliOptions = RuntimeParams


def build_runtime_from_cli(
    options: RuntimeCliOptions | RuntimeParams,
) -> ProjectRuntime:
    """Build runtime from options.
    
    .. deprecated:: 2.0
        Use RuntimeResolver.resolve(RuntimeParams) instead.
    """
    warnings.warn(
        "build_runtime_from_cli is deprecated. "
        "Use RuntimeResolver.resolve(RuntimeParams) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    if isinstance(options, RuntimeCliOptions):
        params = RuntimeParams(
            project_root=options.project_root,
            repo=getattr(options, "repo", None),
            commit=getattr(options, "commit", None),
            # ... map all fields
        )
    else:
        params = options
    
    resolved = RuntimeResolver.resolve(params)
    
    # Convert ResolvedRuntime back to ProjectRuntime for compat
    return _resolved_to_project_runtime(resolved)
```

### 8.9 Async Considerations

Currently, handlers are synchronous. If async support is needed:

```python
# Future: cli/handlers/protocol.py

class AsyncHandlerProtocol(Protocol[T]):
    """Protocol for async CLI handlers."""
    
    async def __call__(self, ctx: HandlerContext) -> CliResult[T]:
        """Execute the handler asynchronously."""
        ...


# Cyclopts command would use anyio:
@app.command()
def my_async_command(...) -> int:
    import anyio
    
    async def run() -> int:
        ctx = _build_context(...)
        result = await my_async_handler(ctx)
        return _render(result)
    
    return anyio.run(run)
```

For now, all handlers are synchronous. Async can be added later without breaking the architecture.

---

## Appendix: Current Duplication Inventory

### A.1 RuntimeCliOptions Definitions

```python
# ide_handlers.py:53
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None

# subsystem_handlers.py:60
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None

# datasets_handlers.py:130
@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None

# common_handlers.py:61
RuntimeCliOptions = RuntimeOptions  # Alias to cli_types.py

# cli_types.py:42
@dataclass(frozen=True)
class RuntimeOptions:
    # ... full definition with 8 fields
```

### A.2 build_runtime_from_cli Implementations

| File | Lines | Behavior |
|------|-------|----------|
| `ide_handlers.py` | 73-99 | Project discovery only, raises ValidationError |
| `subsystem_handlers.py` | 121-147 | Identical to ide_handlers |
| `datasets_handlers.py` | 350-426 | Full fallback, raises ValidationError |
| `build_handlers.py` | 124-150 | Simple, delegates to find_project_root |
| `common_handlers.py` | 330-421 | Full implementation with RuntimeSelection |
| `cyclopts_common.py` | 314-402 | Full implementation, raises RuntimeCliError |
| `resolution/runtime.py` | 83-118 | Uses ExecutionContext, raises ResolutionError |

### A.3 Config Loading Duplicates

| Location | Code |
|----------|------|
| `cyclopts_common.py:41` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` |
| `cyclopts_config.py:24` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` |
| `cyclopts_common.py:45` | `_ENV_CONFIG = cyclopts_config.Env(CONFIG_ENV_PREFIX)` |
| `cyclopts_common.py:60` | `def _optional_toml_config(...)` |
| `cyclopts_config.py:29` | `def _resolve_config_path() -> Path` |
| `cyclopts_config.py:65` | `def _get_env_overrides() -> dict` |

### A.4 Direct stdout Write Locations

```bash
$ rg "sys\.stdout\.write" src/codeintel/cli --files-with-matches
src/codeintel/cli/graphs_handlers.py
src/codeintel/cli/docs_handlers.py
src/codeintel/cli/cyclopts_plugins.py
src/codeintel/cli/cyclopts_jobs.py
src/codeintel/cli/cyclopts_health.py
src/codeintel/cli/cli_completions.py
src/codeintel/cli/cli_render.py
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-10 | AI Assistant | Initial architecture proposal |

---

## Appendix B: Quick Reference

### B.1 Import Paths After Consolidation

```python
# Configuration
from codeintel.cli.config import ConfigService, CliConfig, load_config

# Resolution
from codeintel.cli.resolution import (
    RuntimeResolver,
    RuntimeParams,
    ResolvedRuntime,
    ResolutionError,
    GatewayManager,
)

# Rendering
from codeintel.cli.rendering import (
    UnifiedRenderer,
    RenderContext,
    OutputFormat,
    TableSpec,
    ColumnSpec,
)
from codeintel.cli.rendering.specs import (
    OPERATIONS_TABLE,
    DATASETS_TABLE,
    BUILD_TARGETS_TABLE,
)

# Handlers
from codeintel.cli.handlers import HandlerContext, HandlerProtocol
from codeintel.cli.handlers.build import build_run_handler, build_status_handler
from codeintel.cli.handlers.datasets import dataset_export_handler

# Results
from codeintel.cli.results import CliResult
from codeintel.cli.cli_errors import ProblemDetail, ValidationError

# Cyclopts helpers
from codeintel.cli.cyclopts_common import (
    RuntimeCLI,
    OutputCLI,
    runtime_field,
    output_field,
    command_context,
)
```

### B.2 Minimal Handler Template

```python
"""Handler module template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import HandlerContext


@dataclass(frozen=True)
class MyHandlerData:
    """Data returned by my_handler."""
    
    count: int
    items: list[str]


def my_handler(ctx: HandlerContext) -> CliResult[MyHandlerData]:
    """Do something useful.
    
    Parameters
    ----------
    ctx
        Handler context. Expects ctx.params["required_key"].
        
    Returns
    -------
    CliResult[MyHandlerData]
        Handler result with data or error.
    """
    # 1. Validate params
    required_key = ctx.params.get("required_key")
    if not required_key:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="required_key is required",
            )
        )
    
    # 2. Business logic (use ctx.gateway, ctx.graph_runtime as needed)
    ctx.logger.info("Processing with key: %s", required_key)
    
    items = _fetch_items(ctx.gateway, str(required_key))
    
    # 3. Return result
    return CliResult.ok(
        MyHandlerData(count=len(items), items=items),
        metadata={"key": required_key},
    )


def _fetch_items(gateway: StorageGateway, key: str) -> list[str]:
    """Internal helper (can be tested separately)."""
    # Implementation...
    return []
```

### B.3 Minimal Cyclopts Command Template

```python
"""Cyclopts command template."""

from __future__ import annotations

from dataclasses import dataclass, field

from cyclopts import App, Parameter

from codeintel.cli.cyclopts_common import (
    OutputCLI,
    RuntimeCLI,
    command_context,
    output_field,
    runtime_field,
)
from codeintel.cli.handlers.my_module import my_handler

my_app = App(name="my", help="My commands.")


@my_app.command()
def do_something(
    required_key: str,
    runtime: RuntimeCLI = runtime_field(),
    output: OutputCLI = output_field(),
    optional_flag: bool = False,
) -> int:
    """Do something useful.
    
    Parameters
    ----------
    required_key
        A required string parameter.
    optional_flag
        An optional boolean flag.
    """
    with command_context(
        runtime,
        output,
        params={"required_key": required_key, "optional_flag": optional_flag},
    ) as (ctx, renderer):
        result = my_handler(ctx)
        return renderer.render_result(result)
```

### B.4 Error Type URN Conventions

Error type URNs follow this pattern:

```
urn:codeintel:cli:<domain>/<error-kind>
```

| Domain | Error Kinds | Example |
|--------|-------------|---------|
| `validation` | `missing-param`, `invalid-value`, `type-error` | `urn:codeintel:cli:validation/missing-param` |
| `resolution` | `project-not-found`, `missing-config` | `urn:codeintel:cli:resolution/project-not-found` |
| `storage` | `database-not-found`, `connection-failed` | `urn:codeintel:cli:storage/database-not-found` |
| `build` | `target-not-found`, `cycle-detected` | `urn:codeintel:cli:build/target-not-found` |
| `export` | `schema-mismatch`, `validation-failed` | `urn:codeintel:cli:export/schema-mismatch` |

### B.5 Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Single `RuntimeParams` type | Eliminate field drift across handlers | Keep per-handler variants with shared base |
| `CliResult[T]` mandatory | Enable consistent rendering and testing | Allow handlers to write directly |
| `HandlerContext` over raw params | Type safety, lazy gateway, unified interface | Pass individual params |
| `ConfigService` singleton pattern | Avoid repeated file I/O, ensure consistency | Load config per command |
| Delete `cli_render.py` | One rendering path reduces confusion | Keep both, document when to use each |
| Handlers return, never raise | Cleaner error flow, easier testing | Raise exceptions, catch at command level |

