# CLI Unified Context - Phase 1 Detailed Implementation Plan

> **Purpose**: Comprehensive, step-by-step implementation plan for creating the resolution layer, options consolidation, ExecutionContext enhancement, and proof-of-concept handler migration.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Resolution Package](#step-1-resolution-package)
4. [Step 2: Options Package](#step-2-options-package)
5. [Step 3: ExecutionContext Enhancement](#step-3-executioncontext-enhancement)
6. [Step 4: Proof of Concept Handler Migration](#step-4-proof-of-concept-handler-migration)
7. [Step 5: Verification](#step-5-verification)
8. [Rollback Plan](#rollback-plan)

---

## Overview

### Scope

| Component | Files | Effort |
|-----------|-------|--------|
| Resolution Package | 5 new files | 2-3 hours |
| Options Package | 2 new files | 1-2 hours |
| ExecutionContext Enhancement | 1 modified file | 1-2 hours |
| Proof of Concept | 2-3 modified files | 2-3 hours |
| **Total** | **10-11 files** | **6-10 hours** |

### Success Criteria

- [ ] `resolution/` package created with RuntimeResolver and GatewayManager
- [ ] `options/CommonOptions` created and functional
- [ ] `ExecutionContext` enhanced with `require_runtime()` and `require_gateway()`
- [ ] 5 handlers migrated as proof of concept
- [ ] All CLI tests pass
- [ ] Zero pyright/pyrefly/ruff errors

---

## Prerequisites

Before starting, verify:

```bash
# Environment is set up
scripts/bootstrap.sh

# Current tests pass
uv run pytest tests/cli/ -q

# Quality checks pass
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
```

---

## Step 1: Resolution Package

### 1.1 Create Package Structure

Create directory and `__init__.py`:

```bash
mkdir -p src/codeintel/cli/resolution
```

**File: `src/codeintel/cli/resolution/__init__.py`**

```python
"""Centralized project and runtime resolution for CLI operations.

This package provides:
- `RuntimeResolver`: Resolve project runtime from CLI parameters
- `GatewayManager`: Manage storage gateway lifecycle
- `ResolvedRuntime`: Immutable result of runtime resolution
- `ResolutionError`: Exception for resolution failures

Example
-------
>>> from codeintel.cli.resolution import resolve_runtime
>>> runtime = resolve_runtime(ctx)
>>> runtime.db_path
PosixPath('build/db/codeintel.duckdb')
"""

from __future__ import annotations

from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.gateway import GatewayManager, open_gateway_for_context
from codeintel.cli.resolution.runtime import RuntimeResolver, resolve_runtime
from codeintel.cli.resolution.types import ResolvedRuntime

__all__ = [
    "GatewayManager",
    "ResolutionError",
    "ResolvedRuntime",
    "RuntimeResolver",
    "open_gateway_for_context",
    "resolve_runtime",
]
```

### 1.2 Create Error Types

**File: `src/codeintel/cli/resolution/errors.py`**

```python
"""Resolution error types.

This module defines exceptions raised during runtime and gateway resolution.
"""

from __future__ import annotations


class ResolutionError(Exception):
    """Raised when runtime or gateway resolution fails.

    This exception indicates that the CLI could not resolve the project
    configuration from either a project file (codeintel.yaml) or explicit
    CLI parameters.

    Parameters
    ----------
    message
        Human-readable error description.
    missing_params
        Optional list of missing required parameters.

    Examples
    --------
    >>> raise ResolutionError("No codeintel.yaml found")
    >>> raise ResolutionError(
    ...     "Missing required parameters",
    ...     missing_params=["repo", "commit"]
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        missing_params: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.missing_params = missing_params or []

    def __str__(self) -> str:
        """Format error message.

        Returns
        -------
        str
            Error message, optionally including missing parameters.
        """
        if self.missing_params:
            params = ", ".join(self.missing_params)
            return f"{self.message}. Missing: {params}"
        return self.message


__all__ = ["ResolutionError"]
```

### 1.3 Create Types Module

**File: `src/codeintel/cli/resolution/types.py`**

```python
"""Resolution result types.

This module defines the immutable result types returned by resolution operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.project import ProjectConfig
    from codeintel.config.models import CodeIntelConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.config.serving_models import ServingConfig


@dataclass(frozen=True)
class ResolvedRuntime:
    """Fully resolved runtime - the immutable result of runtime resolution.

    This dataclass contains all resolved project information needed by handlers.
    It is created by RuntimeResolver and cached in ExecutionContext.

    All fields are required and immutable (frozen=True).

    Parameters
    ----------
    root
        Project root directory.
    project
        Project configuration from codeintel.yaml or constructed from params.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    paths
        Resolved build paths (db_path, build_dir, etc.).
    config
        Full CodeIntel configuration.
    serving
        Serving configuration for API operations.

    Examples
    --------
    >>> runtime = resolve_runtime(ctx)
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    >>> runtime.repo
    'org/repo'
    """

    root: Path
    project: ProjectConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    config: CodeIntelConfig
    serving: ServingConfig

    @property
    def db_path(self) -> Path:
        """Get database path shortcut.

        Returns
        -------
        Path
            Path to DuckDB database file.
        """
        return self.paths.db_path

    @property
    def repo(self) -> str:
        """Get repository slug shortcut.

        Returns
        -------
        str
            Repository slug (e.g., 'org/repo').
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Get commit SHA shortcut.

        Returns
        -------
        str
            Commit SHA.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Get repository root shortcut.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.snapshot.repo_root


__all__ = ["ResolvedRuntime"]
```

### 1.4 Create RuntimeResolver

**File: `src/codeintel/cli/resolution/runtime.py`**

This is the most complex file - it consolidates ALL `build_runtime_from_cli` implementations.

```python
"""Runtime resolution - single source of truth.

This module consolidates all runtime resolution logic previously scattered across:
- cyclopts_common.py:build_runtime_from_cli
- common_handlers.py:build_runtime_from_cli
- datasets_handlers.py:build_runtime_from_cli
- subsystem_handlers.py:build_runtime_from_cli
- ide_handlers.py:build_runtime_from_cli
- build_handlers.py:build_runtime_from_cli
- ops_handlers.py:_build_runtime_or_error
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.models import CliPathsInput, RepoConfig
from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.config.serving_models import ServingConfig

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.config.models import CodeIntelConfig

LOG = logging.getLogger(__name__)


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

    def resolve(
        self,
        ctx: ExecutionContext,
        *,
        allow_fallback: bool = True,
    ) -> ResolvedRuntime:
        """Resolve runtime from context parameters.

        Parameters
        ----------
        ctx
            Execution context with params.
        allow_fallback
            If True, attempt fallback to explicit params when no project file.
            If False, raise immediately when project file not found.

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
            if not allow_fallback:
                raise ResolutionError(
                    "No codeintel.yaml found and fallback disabled"
                ) from None

        # Fall back to explicit params
        return self._resolve_from_params(ctx)

    def _resolve_from_project(self, project_root: Path | None) -> ResolvedRuntime:
        """Resolve from project file (codeintel.yaml).

        Parameters
        ----------
        project_root
            Optional explicit project root. If None, searches from cwd.

        Returns
        -------
        ResolvedRuntime
            Runtime resolved from project file.

        Raises
        ------
        ProjectNotFoundError
            If no project file found.
        """
        # build_project_runtime handles discovery and construction
        project_runtime = build_project_runtime(project_root)

        return ResolvedRuntime(
            root=project_runtime.root,
            project=project_runtime.project,
            snapshot=project_runtime.snapshot,
            paths=project_runtime.paths,
            config=project_runtime.cfg,
            serving=project_runtime.serving,
        )

    def _resolve_from_params(self, ctx: ExecutionContext) -> ResolvedRuntime:
        """Resolve from explicit CLI parameters.

        Parameters
        ----------
        ctx
            Execution context with params.

        Returns
        -------
        ResolvedRuntime
            Runtime resolved from explicit parameters.

        Raises
        ------
        ResolutionError
            If required parameters are missing.
        """
        repo = ctx.get_param("repo")
        commit = ctx.get_param("commit")

        # Check required params
        missing = []
        if repo is None:
            missing.append("repo")
        if commit is None:
            missing.append("commit")

        if missing:
            raise ResolutionError(
                "No codeintel.yaml found. Provide --repo and --commit explicitly",
                missing_params=missing,
            )

        # Get optional params with defaults
        repo_root = ctx.get_param("repo_root") or Path.cwd()
        db_path = ctx.get_param("db_path") or Path("build/db/codeintel.duckdb")
        build_dir = ctx.get_param("build_dir") or Path("build")
        document_output_dir = ctx.get_param("document_output_dir")

        # Build configuration
        config = self._build_config(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            db_path=db_path,
            build_dir=build_dir,
            document_output_dir=document_output_dir,
            use_gpu=ctx.get_param("use_gpu", False),
        )

        # Build snapshot reference
        snapshot = SnapshotRef(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
        )

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Build project config
        project = ProjectConfig(
            repo=repo,
            storage=StorageProjectConfig(db_path=db_path),
        )

        # Build serving config
        serving = ServingConfig(
            mode="local_db",
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            db_path=db_path,
            read_only=True,
        )

        # Build paths
        paths = config.build_paths

        return ResolvedRuntime(
            root=repo_root,
            project=project,
            snapshot=snapshot,
            paths=paths,
            config=config,
            serving=serving,
        )

    def _build_config(
        self,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        db_path: Path,
        build_dir: Path,
        document_output_dir: Path | None,
        use_gpu: bool,
    ) -> CodeIntelConfig:
        """Build CodeIntelConfig from parameters.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.
        repo_root
            Repository root directory.
        db_path
            Database path.
        build_dir
            Build directory.
        document_output_dir
            Optional document output directory override.
        use_gpu
            Whether to enable GPU backend.

        Returns
        -------
        CodeIntelConfig
            Constructed configuration.
        """
        from codeintel.config.models import CodeIntelConfig

        paths_cfg = CliPathsInput(
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path,
            document_output_dir=document_output_dir,
        )

        repo_cfg = RepoConfig(repo=repo, commit=commit)

        backend = GraphBackendConfig(
            use_gpu=use_gpu,
            features=GraphFeatureFlags(),
        )

        return CodeIntelConfig.from_cli(
            repo=repo_cfg,
            paths=paths_cfg,
            backend=backend,
        )


# Module-level singleton for convenience
_resolver = RuntimeResolver()


def resolve_runtime(
    ctx: ExecutionContext,
    *,
    allow_fallback: bool = True,
) -> ResolvedRuntime:
    """Resolve runtime from context (module-level convenience function).

    Parameters
    ----------
    ctx
        Execution context with params.
    allow_fallback
        If True, attempt fallback to explicit params.

    Returns
    -------
    ResolvedRuntime
        Fully resolved runtime.

    Raises
    ------
    ResolutionError
        If resolution fails.
    """
    return _resolver.resolve(ctx, allow_fallback=allow_fallback)


__all__ = [
    "RuntimeResolver",
    "resolve_runtime",
]
```

### 1.5 Create GatewayManager

**File: `src/codeintel/cli/resolution/gateway.py`**

```python
"""Gateway lifecycle management.

This module provides centralized gateway management for CLI operations,
ensuring consistent opening, caching, and cleanup of storage gateways.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


class GatewayManager:
    """Manage gateway lifecycle for ExecutionContext.

    The manager opens gateways on demand using the resolved runtime's db_path.
    Gateways are cached in the ExecutionContext and cleaned up when the
    context is closed.

    This class is stateless - all state is stored in ExecutionContext.

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

        Uses the resolved runtime to determine the database path.
        If the context doesn't have a resolved runtime yet, this will
        trigger resolution.

        Parameters
        ----------
        ctx
            Execution context. Must have resolvable runtime.
        read_only
            Whether to open in read-only mode. Defaults to True for safety.

        Returns
        -------
        StorageGateway
            Open gateway connected to the database.

        Raises
        ------
        ResolutionError
            If runtime cannot be resolved.
        StorageConnectionError
            If gateway cannot be opened.
        """
        # This will resolve runtime if not already resolved
        runtime = ctx.require_runtime()

        LOG.debug(
            "Opening gateway for %s: db_path=%s, read_only=%s",
            ctx.operation_id,
            runtime.db_path,
            read_only,
        )

        storage_config = StorageConfig(
            db_path=runtime.db_path,
            read_only=read_only,
        )
        return open_gateway(storage_config)

    def close(self, ctx: ExecutionContext) -> None:
        """Close gateway if open.

        Safe to call even if no gateway is open.

        Parameters
        ----------
        ctx
            Execution context.
        """
        if ctx.metadata._gateway is not None:
            LOG.debug("Closing gateway for %s", ctx.operation_id)
            ctx.metadata._gateway.close()
            ctx.metadata._gateway = None


# Module-level singleton for convenience
_gateway_manager = GatewayManager()


def open_gateway_for_context(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open gateway for context (module-level convenience function).

    Parameters
    ----------
    ctx
        Execution context.
    read_only
        Whether to open in read-only mode.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    return _gateway_manager.open(ctx, read_only=read_only)


def close_gateway_for_context(ctx: ExecutionContext) -> None:
    """Close gateway for context (module-level convenience function).

    Parameters
    ----------
    ctx
        Execution context.
    """
    _gateway_manager.close(ctx)


__all__ = [
    "GatewayManager",
    "close_gateway_for_context",
    "open_gateway_for_context",
]
```

### 1.6 Verification Checkpoint

After completing Step 1, verify:

```bash
# Lint check
uv run ruff check src/codeintel/cli/resolution/

# Type check
uv run pyright src/codeintel/cli/resolution/

# Pyrefly check
uv run pyrefly check src/codeintel/cli/resolution/

# Import test
python -c "from codeintel.cli.resolution import resolve_runtime, ResolvedRuntime, ResolutionError"
```

---

## Step 2: Options Package

### 2.1 Create Package Structure

```bash
mkdir -p src/codeintel/cli/options
```

**File: `src/codeintel/cli/options/__init__.py`**

```python
"""Unified CLI option bundles.

This package provides consolidated option dataclasses for Cyclopts commands,
replacing scattered RuntimeCLI, OutputFormatCLI, and other option classes.

Example
-------
>>> from codeintel.cli.options import CommonOptions
>>> options = CommonOptions(verbose=2, output_format=OutputFormat.JSON)
>>> params = options.to_params()
"""

from __future__ import annotations

from codeintel.cli.options.common import CommonOptions

__all__ = [
    "CommonOptions",
]
```

### 2.2 Create CommonOptions

**File: `src/codeintel/cli/options/common.py`**

```python
"""Unified option bundle for CLI commands.

This module provides CommonOptions, a single dataclass that combines all
common CLI options:
- Runtime options (project_root, repo, commit, db_path, etc.)
- Output options (output_format, json flag)
- Execution options (verbose, dry_run)
- Backend options (use_gpu)

CommonOptions replaces the scattered:
- RuntimeCLI in cyclopts_common.py
- OutputFormatCLI in cyclopts_common.py
- BackendFlags in cli_types.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import Parameter

from codeintel.cli.cli_types import OutputFormat

if TYPE_CHECKING:
    pass


@dataclass
class CommonOptions:
    """Single option bundle for all CLI commands.

    This dataclass is designed to be embedded in Cyclopts command classes
    using ``field(default_factory=CommonOptions, metadata=COMMON_OPTIONS_METADATA)``.

    Cyclopts will flatten all fields as top-level CLI flags.

    Parameters
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

    Examples
    --------
    In a Cyclopts command:

    >>> @app.command
    ... @dataclass
    ... class MyCommand:
    ...     target: str
    ...     options: Annotated[CommonOptions, Parameter(name="*")] = field(
    ...         default_factory=CommonOptions
    ...     )
    ...
    ...     def __call__(self) -> None:
    ...         params = self.options.to_params()
    ...         params["target"] = self.target
    ...         execute_command("my.command", params)
    """

    # Runtime selection
    project_root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
            help="Explicit project root directory.",
        ),
    ] = None

    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'org/repo'). Uses project config if omitted.",
        ),
    ] = None

    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Uses project config if omitted.",
        ),
    ] = None

    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database. Uses project config if omitted.",
        ),
    ] = None

    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory (default: build/).",
        ),
    ] = None

    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root (default: current directory).",
        ),
    ] = None

    document_output_dir: Annotated[
        Path | None,
        Parameter(
            name="--document-output-dir",
            help="Override document output directory.",
        ),
    ] = None

    # Output control
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format.",
            show_choices=True,
        ),
    ] = OutputFormat.TEXT

    json: Annotated[
        bool,
        Parameter(
            name="--json",
            help="Alias for --output-format json.",
            negative=(),
        ),
    ] = False

    # Execution control
    verbose: Annotated[
        int,
        Parameter(
            name=["--verbose", "-v"],
            help="Increase verbosity (0=warnings, 1=info, 2=debug).",
            count=True,
        ),
    ] = 0

    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run", "-n"],
            help="Show what would be done without doing it.",
            negative=(),
        ),
    ] = False

    # Backend control
    use_gpu: Annotated[
        bool,
        Parameter(
            name="--gpu",
            help="Enable GPU acceleration for graph operations.",
            negative=(),
        ),
    ] = False

    def to_params(self) -> dict[str, Any]:
        """Convert to parameter dictionary for ExecutionContext.

        Returns
        -------
        dict[str, Any]
            All options as a flat dictionary.
        """
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
        """Resolve output format with json flag precedence.

        Returns
        -------
        OutputFormat
            JSON if json flag is True, otherwise output_format.
        """
        return OutputFormat.JSON if self.json else self.output_format


# Metadata for Cyclopts parameter flattening
COMMON_OPTIONS_METADATA: dict[str, Parameter] = {"parameter": Parameter(name="*")}


def common_options_field() -> CommonOptions:
    """Create a field for CommonOptions with Cyclopts metadata.

    Use this in dataclass definitions:

    >>> @dataclass
    ... class MyCommand:
    ...     options: Annotated[CommonOptions, Parameter(name="*")] = field(
    ...         default_factory=CommonOptions
    ...     )

    Returns
    -------
    CommonOptions
        Default CommonOptions instance.
    """
    return field(default_factory=CommonOptions, metadata=COMMON_OPTIONS_METADATA)


__all__ = [
    "COMMON_OPTIONS_METADATA",
    "CommonOptions",
    "common_options_field",
]
```

### 2.3 Verification Checkpoint

```bash
# Lint check
uv run ruff check src/codeintel/cli/options/

# Type check
uv run pyright src/codeintel/cli/options/

# Pyrefly check
uv run pyrefly check src/codeintel/cli/options/

# Import test
python -c "from codeintel.cli.options import CommonOptions; print(CommonOptions())"
```

---

## Step 3: ExecutionContext Enhancement

### 3.1 Read Current ExecutionContext

First, understand the current implementation:

```bash
cat src/codeintel/cli/execution/context.py
```

### 3.2 Modify ExecutionContext

**File: `src/codeintel/cli/execution/context.py`** (modifications)

Add the following to the existing file:

#### 3.2.1 Add Imports

Add to imports section:

```python
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.config import CliConfig, load_config

if TYPE_CHECKING:
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway
```

#### 3.2.2 Add ContextMetadata Class

Add new dataclass before ExecutionContext:

```python
@dataclass
class ContextMetadata:
    """Metadata and lazy-resolved resources for ExecutionContext.

    This class holds both static configuration and lazy-resolved
    resources. Resolution happens on first access via the parent
    ExecutionContext's ``require_*`` methods.

    Parameters
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

    # Private lazy-resolved fields (underscore prefix, not in repr)
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)
```

#### 3.2.3 Enhance ExecutionContext Class

Add/modify the ExecutionContext class:

```python
@dataclass
class ExecutionContext:
    """Unified context for all CLI operations.

    This is the SINGLE context type that all handlers receive. It provides:
    - Operation identification (operation_id)
    - Parameter access (params, get_param, require_param)
    - Lazy resource resolution (require_runtime, require_gateway)
    - Configuration access (config, output_format, etc.)
    - Logging (logger)

    Handlers MUST NOT create their own context types. All context
    information flows through ExecutionContext.

    Parameters
    ----------
    operation_id
        Unique identifier for this operation (e.g., "build.run").
    params
        Raw parameters from CLI/caller.
    metadata
        Configuration and lazy-resolved resources. If not provided,
        a default ContextMetadata is created.
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
    metadata: ContextMetadata | None = None
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            config = load_config(validate=False)
            self.metadata = ContextMetadata(
                config=config,
                verbosity=self.params.get("verbose", 0),
                output_format=self.params.get("output_format", OutputFormat.TEXT),
                dry_run=self.params.get("dry_run", False),
            )

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

    # --- Resource Access (Lazy Resolution) ---

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
        if self.metadata is None:
            msg = "ExecutionContext metadata not initialized"
            raise RuntimeError(msg)

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
        if self.metadata is None:
            msg = "ExecutionContext metadata not initialized"
            raise RuntimeError(msg)

        if self.metadata._gateway is None:
            from codeintel.cli.resolution import open_gateway_for_context

            self.metadata._gateway = open_gateway_for_context(self, read_only=read_only)
        return self.metadata._gateway

    def close(self) -> None:
        """Close any open resources.

        Should be called when execution completes, typically by the executor.
        Safe to call multiple times.
        """
        if self.metadata is not None and self.metadata._gateway is not None:
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

    def require_param(self, key: str) -> Any:
        """Get required parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        Any
            Parameter value.

        Raises
        ------
        ValueError
            If parameter is missing.
        """
        value = self.params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
        return value

    # --- Convenience Properties ---

    @property
    def config(self) -> CliConfig:
        """Get CLI configuration.

        Returns
        -------
        CliConfig
            CLI configuration.

        Raises
        ------
        RuntimeError
            If metadata not initialized.
        """
        if self.metadata is None:
            msg = "ExecutionContext metadata not initialized"
            raise RuntimeError(msg)
        return self.metadata.config

    @property
    def output_format(self) -> OutputFormat:
        """Get resolved output format.

        Returns
        -------
        OutputFormat
            Output format (TEXT or JSON).
        """
        if self.metadata is None:
            return OutputFormat.TEXT
        return self.metadata.output_format

    @property
    def verbosity(self) -> int:
        """Get verbosity level.

        Returns
        -------
        int
            Verbosity level (0-2+).
        """
        if self.metadata is None:
            return 0
        return self.metadata.verbosity

    @property
    def dry_run(self) -> bool:
        """Check if this is a dry-run.

        Returns
        -------
        bool
            True if dry-run mode.
        """
        if self.metadata is None:
            return False
        return self.metadata.dry_run

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this operation.

        Returns
        -------
        logging.Logger
            Logger named for this operation.
        """
        return logging.getLogger(f"codeintel.cli.{self.operation_id}")
```

#### 3.2.4 Update __all__

Update the `__all__` in context.py:

```python
__all__ = [
    "ContextMetadata",
    "ExecutionContext",
    "ExecutionResult",
]
```

### 3.3 Verification Checkpoint

```bash
# Lint check
uv run ruff check src/codeintel/cli/execution/context.py

# Type check
uv run pyright src/codeintel/cli/execution/context.py

# Pyrefly check
uv run pyrefly check src/codeintel/cli/execution/context.py

# Full CLI tests
uv run pytest tests/cli/ -q --tb=short
```

---

## Step 4: Proof of Concept Handler Migration

### 4.1 Target Handlers

Migrate the simplest handlers first:

| File | Handler | Current Signature | Complexity |
|------|---------|-------------------|------------|
| `storage_handlers.py` | `storage_validate_macros` | `(db_path, macro_req, verbose)` | Low |
| `storage_handlers.py` | `generate_macros_for_tables` | `(tables, verbose)` | Low |
| `storage_handlers.py` | `profile_storage_paths` | `(db_path, output_dir, ...)` | Low |
| `ops_handlers.py` | `op_list_handler` | `(category, output_format)` | Low |
| `ops_handlers.py` | `_build_runtime_or_error` | `(project_root)` | Helper |

### 4.2 Migration Pattern

For each handler:

1. **Change signature** to accept `ExecutionContext`
2. **Replace resolution** with `ctx.require_runtime()`
3. **Replace param access** with `ctx.get_param()`
4. **Return CliResult** instead of None/direct output
5. **Update Cyclopts command** to use adapter

### 4.3 Migrate storage_handlers.py

**Current** (`storage_handlers.py`):

```python
def storage_validate_macros(
    db_path: Path,
    macro_requirement: MacroRequirement,
    verbose: int,
) -> None:
    """Validate macro registry hashes and normalized macro schemas."""
    setup_logging(verbose)
    # ... implementation ...
```

**After** (`storage_handlers.py`):

```python
from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.results import CliResult


def storage_validate_macros(ctx: ExecutionContext) -> CliResult[dict[str, Any]]:
    """Validate macro registry hashes and normalized macro schemas.

    Parameters
    ----------
    ctx
        Execution context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - macro_requirement: MacroRequirement enum value
        - verbose: Verbosity level

    Returns
    -------
    CliResult[dict[str, Any]]
        Validation result with status and any issues found.
    """
    from codeintel.cli.handlers.base import setup_logging

    # Setup logging from context
    setup_logging(ctx.verbosity)

    # Get params
    db_path = ctx.get_param("db_path")
    if db_path is None:
        # Use runtime resolution if no explicit db_path
        runtime = ctx.require_runtime()
        db_path = runtime.db_path

    macro_requirement = ctx.get_param("macro_requirement", MacroRequirement.REQUIRE)

    # ... existing implementation ...

    try:
        gateway = open_gateway(StorageConfig.for_readonly(db_path))
    except StorageConnectionError as exc:
        LOG.warning("Falling back to existing database attachment: %s", exc)
        return CliResult.success({"status": "skipped", "reason": str(exc)})

    # ... rest of implementation ...

    return CliResult.success({
        "status": "valid",
        "missing_ingest": missing_ingest,
        "present_ingest": present_ingest,
    })
```

### 4.4 Update Corresponding Cyclopts Command

**Before** (`cyclopts_storage.py`):

```python
@storage_app.command(name="validate-macros")
@dataclass
class ValidateMacrosCli:
    db_path: Path
    macro_requirement: MacroRequirement = MacroRequirement.REQUIRE
    runtime: RuntimeCLI | None = None
    output: OutputFormatCLI | None = None

    def __call__(self) -> None:
        runtime_opts, verbose, output_format = make_handler_context(...)
        storage_validate_macros(
            db_path=self.db_path,
            macro_requirement=self.macro_requirement,
            verbose=verbose,
        )
```

**After** (`cyclopts_storage.py`):

```python
from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.options import CommonOptions


@storage_app.command(name="validate-macros")
@dataclass
class ValidateMacrosCli:
    db_path: Annotated[Path | None, Parameter(...)] = None
    macro_requirement: MacroRequirement = MacroRequirement.REQUIRE
    options: Annotated[CommonOptions, Parameter(name="*")] = field(
        default_factory=CommonOptions
    )

    def __call__(self) -> None:
        CycloptsAdapter("storage.validate_macros", storage_validate_macros)(self)
```

### 4.5 Repeat for Other Handlers

Apply the same pattern to:
- `generate_macros_for_tables`
- `profile_storage_paths`
- `op_list_handler`

### 4.6 Verification Checkpoint

```bash
# Run storage-specific tests
uv run pytest tests/cli/ -k "storage" -v

# Run ops-specific tests
uv run pytest tests/cli/ -k "ops" -v

# Full quality checks
uv run ruff check src/codeintel/cli/storage_handlers.py src/codeintel/cli/ops_handlers.py
uv run pyright src/codeintel/cli/storage_handlers.py src/codeintel/cli/ops_handlers.py
```

---

## Step 5: Verification

### 5.1 Full Quality Suite

```bash
# Lint all CLI code
uv run ruff check src/codeintel/cli/

# Type check all CLI code
uv run pyright src/codeintel/cli/

# Pyrefly check
uv run pyrefly check src/codeintel/cli/

# Full test suite
uv run pytest tests/cli/ -q
```

### 5.2 Import Verification

```bash
# Verify new packages are importable
python -c "
from codeintel.cli.resolution import (
    resolve_runtime,
    ResolvedRuntime,
    ResolutionError,
    GatewayManager,
)
from codeintel.cli.options import CommonOptions
from codeintel.cli.execution.context import ExecutionContext, ContextMetadata
print('All imports successful')
"
```

### 5.3 Integration Test

Create a simple integration test:

```python
# tests/cli/unit/test_resolution_integration.py

def test_resolution_from_params():
    """Test runtime resolution from explicit params."""
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.cli.resolution import ResolutionError

    ctx = ExecutionContext.for_sync(
        "test.op",
        {
            "repo": "test/repo",
            "commit": "abc123",
            "db_path": Path("/tmp/test.duckdb"),
        },
    )

    runtime = ctx.require_runtime()
    assert runtime.repo == "test/repo"
    assert runtime.commit == "abc123"


def test_resolution_missing_params():
    """Test resolution fails with missing params."""
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.cli.resolution import ResolutionError

    ctx = ExecutionContext.for_sync("test.op", {})

    with pytest.raises(ResolutionError) as exc_info:
        ctx.require_runtime()

    assert "repo" in str(exc_info.value)
    assert "commit" in str(exc_info.value)
```

---

## Rollback Plan

If issues are discovered:

### Rollback Level 1: Proof of Concept Only

If handler migration causes issues:
- Revert handler changes
- Keep infrastructure (resolution/, options/, enhanced ExecutionContext)
- Infrastructure is backward compatible

### Rollback Level 2: ExecutionContext

If ExecutionContext changes cause issues:
- Revert context.py changes
- Keep resolution/ and options/ packages
- They can be used standalone

### Rollback Level 3: Full Rollback

If fundamental issues discovered:
```bash
git revert HEAD~N  # Revert all changes
```

Infrastructure is designed to be additive and non-breaking, so partial rollback should be sufficient.

---

## Checklist Summary

### Step 1: Resolution Package
- [ ] Create `resolution/__init__.py`
- [ ] Create `resolution/errors.py`
- [ ] Create `resolution/types.py`
- [ ] Create `resolution/runtime.py`
- [ ] Create `resolution/gateway.py`
- [ ] Verify imports and types

### Step 2: Options Package
- [ ] Create `options/__init__.py`
- [ ] Create `options/common.py`
- [ ] Verify imports and types

### Step 3: ExecutionContext Enhancement
- [ ] Add `ContextMetadata` class
- [ ] Add `require_runtime()` method
- [ ] Add `require_gateway()` method
- [ ] Add convenience properties
- [ ] Update `__all__`
- [ ] Run full test suite

### Step 4: Proof of Concept
- [ ] Migrate `storage_validate_macros`
- [ ] Migrate `generate_macros_for_tables`
- [ ] Migrate `profile_storage_paths`
- [ ] Migrate `op_list_handler`
- [ ] Update corresponding Cyclopts commands
- [ ] Run targeted tests

### Step 5: Verification
- [ ] Full quality suite passes
- [ ] All imports work
- [ ] Integration tests pass

---

*Document Version: 1.0*
*Created: 2025-01-09*

