# CLI Consolidation Phase 3: Handler Migration — Detailed Implementation

> **Status**: Draft  
> **Depends On**: Phase 1 (Foundation Layer), Phase 2 (Config Integration)  
> **Enables**: Phase 4 (Cleanup)  
> **Risk Level**: Medium-High (touches all handler code)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current State Analysis](#2-current-state-analysis)
3. [Foundation Components from Phase 1 & 2](#3-foundation-components-from-phase-1--2)
4. [Wave 1: Simple Handlers (ide, subsystem)](#4-wave-1-simple-handlers-ide-subsystem)
5. [Wave 2: Medium Handlers (build, storage, history, health)](#5-wave-2-medium-handlers-build-storage-history-health)
6. [Wave 3: Complex Handlers (datasets, docs, graphs, ops)](#6-wave-3-complex-handlers-datasets-docs-graphs-ops)
7. [Shared Infrastructure Updates](#7-shared-infrastructure-updates)
8. [Testing Strategy](#8-testing-strategy)
9. [Migration Checklist](#9-migration-checklist)

---

## 1. Overview

Phase 3 migrates all CLI handlers from the legacy pattern to the unified pattern established in Phases 1 and 2. Upon completion:

- **All handlers** accept `EnhancedHandlerContext` as their primary input
- **All handlers** return `CliResult[T]` (never write to stdout directly)
- **All handlers** use `RuntimeParams` via `RuntimeResolver.resolve()`
- **No** per-module `RuntimeCliOptions` variants remain
- **Zero** `sys.stdout.write()` calls in handler files

### Key Components Available from Phase 1 & 2

| Component | Location | Purpose |
|-----------|----------|---------|
| `EnhancedHandlerContext` | `handlers/protocol.py` | Unified context with lazy gateway/runtime |
| `HandlerProtocol` | `handlers/protocol.py` | Contract for all handlers |
| `handler_context()` | `handlers/protocol.py` | Context manager for handler execution |
| `RuntimeParams` | `resolution/params.py` | Canonical runtime parameters |
| `RuntimeResolver` | `resolution/runtime.py` | Single runtime resolution |
| `UnifiedRenderer` | `rendering/service.py` | All output rendering |
| `RenderContext` | `rendering/types.py` | Format/color/stream settings |
| `ConfigService` | `config/service.py` | Unified config loading |
| `TableSpec`, `ColumnSpec` | `rendering/table.py` | Table specifications |

---

## 2. Current State Analysis

### 2.1 Legacy Pattern (Example: ide_handlers.py)

```python
# Current pattern in ide_handlers.py

@dataclass(frozen=True)
class RuntimeCliOptions:
    """Per-module RuntimeCliOptions with limited fields."""
    project_root: Path | None = None

@dataclass(frozen=True)
class IdeHintsOptions:
    rel_path: str
    runtime_options: RuntimeCliOptions
    verbose: int = 0

def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Per-module runtime builder."""
    try:
        project_root = find_project_root(options.project_root)
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise ValidationError(msg) from exc

def ide_hints_ctx(ctx: ExecutionContext) -> CliResult[IdeHintsResult]:
    """Handler using ExecutionContext (older pattern)."""
    setup_logging(ctx.verbosity)
    rel_path = ctx.require_str_param("rel_path")
    # ... direct gateway management, etc.
```

### 2.2 Target Pattern (Using Phase 1 Foundation)

```python
# Target pattern using EnhancedHandlerContext

def ide_hints_handler(ctx: EnhancedHandlerContext) -> CliResult[IdeHintsData]:
    """Generate IDE hints for a file.
    
    Parameters
    ----------
    ctx
        Handler context. Expects ctx.params["rel_path"].
        
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
    
    # Use ctx.gateway (lazy), ctx.graph_runtime (lazy)
    ctx.logger.info("Fetching hints for %s", rel_path)
    hints = _fetch_hints(ctx.gateway, ctx.graph_runtime, str(rel_path))
    
    return CliResult.ok(IdeHintsData(rel_path=str(rel_path), hints=hints))
```

### 2.3 Handler Files to Migrate

| File | Handler Count | `RuntimeCliOptions` | `sys.stdout.write` | Complexity |
|------|---------------|--------------------|--------------------|------------|
| `ide_handlers.py` | 1 | Yes (minimal) | 0 | Low |
| `subsystem_handlers.py` | ~5 | Yes (minimal) | 0 | Low |
| `build_handlers.py` | ~8 | No (uses alias) | 0 | Medium |
| `storage_handlers.py` | ~5 | Yes | 0 | Medium |
| `history_handlers.py` | ~4 | Yes | 0 | Medium |
| `cyclopts_health.py` | ~3 | No | 2 | Medium |
| `datasets_handlers.py` | ~15 | Yes (extended) | 0 | High |
| `docs_handlers.py` | ~10 | Yes | 9 | High |
| `graphs_handlers.py` | ~8 | Yes | 12 | High |
| `ops_handlers.py` | ~6 | No | 0 | Medium |

---

## 3. Foundation Components from Phase 1 & 2

### 3.1 EnhancedHandlerContext (handlers/protocol.py)

Already implemented with:

- `config: CliConfig` — CLI configuration
- `runtime: ResolvedRuntime` — Resolved project runtime
- `params: Mapping[str, object]` — Operation-specific parameters
- `verbosity: int` — Logging verbosity
- `gateway` property — Lazy StorageGateway access
- `graph_runtime` property — Lazy GraphRuntime access
- `logger` property — Handler-specific logger
- `close()` method — Resource cleanup

### 3.2 handler_context() Context Manager (handlers/protocol.py)

Already implemented:

```python
@contextmanager
def handler_context(
    config: CliConfig,
    runtime: ResolvedRuntime,
    params: Mapping[str, object] | None = None,
    *,
    verbosity: int = 0,
    operation_name: str = "handler",
) -> Iterator[EnhancedHandlerContext]:
    """Create handler context with automatic resource cleanup."""
```

### 3.3 RuntimeParams (resolution/params.py)

Already implemented with factory methods:

- `from_cyclopts(runtime_cli: RuntimeCLI)` — From Cyclopts dataclass
- `from_context(ctx: ExecutionContext)` — From execution context
- `from_dict(data: dict)` — From dictionary
- `minimal(project_root)` — For simple commands

### 3.4 ConfigService (config/service.py)

Already implemented in Phase 2:

- `load()` — Load config from all sources
- `get_cyclopts_config_chain()` — Cyclopts integration
- `get_toml_config_path()` — Config path introspection

---

## 4. Wave 1: Simple Handlers (ide, subsystem)

### 4.1 Scope

| Current File | New Location | Handlers |
|--------------|--------------|----------|
| `ide_handlers.py` | `handlers/ide.py` | `ide_hints_handler` |
| `subsystem_handlers.py` | `handlers/subsystem.py` | ~5 handlers |
| `cyclopts_ide.py` | (update) | Command wiring |
| `cyclopts_subsystem.py` | (update) | Command wiring |

### 4.2 Step 1: Create handlers/ide.py

**File**: `src/codeintel/cli/handlers/ide.py`

```python
"""IDE integration handlers.

Handlers for IDE hints and integration features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class IdeHintsData:
    """Data returned by ide_hints_handler.

    Parameters
    ----------
    rel_path
        Relative path that was queried.
    hints
        List of hints for the file.
    meta
        Response metadata.
    """

    rel_path: str
    hints: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rel_path": self.rel_path,
            "hints": self.hints,
            "meta": self.meta,
        }


def ide_hints_handler(ctx: EnhancedHandlerContext) -> CliResult[IdeHintsData]:
    """Generate IDE hints for a file.

    Parameters
    ----------
    ctx
        Handler context. Expects ctx.params["rel_path"].

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

    ctx.logger.info("Fetching hints for %s", rel_path)

    # Build backend resource using context's lazy resources
    resource = build_backend_resource(
        ctx.runtime.serving,
        gateway=ctx.gateway,
        options=BackendResourceOptions(graph_runtime=ctx.graph_runtime),
    )

    response = resource.backend.get_file_hints(rel_path=str(rel_path))
    if not response.found or not response.hints:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:ide/no-hints",
                title="No Hints Found",
                detail=f"No hints found for: {rel_path}",
            )
        )

    return CliResult.ok(
        IdeHintsData(
            rel_path=str(rel_path),
            hints=[hint.model_dump() for hint in response.hints],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "IdeHintsData",
    "ide_hints_handler",
]
```

### 4.3 Step 2: Create command_context Helper

**File**: `src/codeintel/cli/cyclopts_common.py` (add to existing)

```python
from contextlib import contextmanager
from collections.abc import Iterator

from codeintel.cli.config import ConfigService
from codeintel.cli.resolution import RuntimeResolver, RuntimeParams
from codeintel.cli.rendering import UnifiedRenderer, RenderContext, OutputFormat
from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@contextmanager
def command_context(
    runtime: RuntimeCLI,
    output: OutputFormatCLI,
    params: dict[str, object],
) -> Iterator[tuple[EnhancedHandlerContext, UnifiedRenderer]]:
    """Create standard context for command execution.

    Handle config loading, runtime resolution, logging setup,
    resource cleanup, and renderer creation.

    Parameters
    ----------
    runtime
        Cyclopts RuntimeCLI dataclass.
    output
        Cyclopts OutputFormatCLI dataclass.
    params
        Operation-specific parameters.

    Yields
    ------
    tuple[EnhancedHandlerContext, UnifiedRenderer]
        Handler context and renderer for execution.

    Examples
    --------
    >>> @app.command()
    >>> def my_command(runtime: RuntimeCLI, output: OutputFormatCLI, arg: str) -> int:
    ...     with command_context(runtime, output, {"arg": arg}) as (ctx, renderer):
    ...         result = my_handler(ctx)
    ...         return renderer.render_result(result)
    """
    # 1. Load config
    config_service = ConfigService.load(validate=False)

    # 2. Resolve runtime
    runtime_params = RuntimeParams.from_cyclopts(runtime)
    resolved = RuntimeResolver.resolve(runtime_params)

    # 3. Create handler context
    ctx = EnhancedHandlerContext(
        config=config_service.config,
        runtime=resolved,
        params=params,
        verbosity=runtime.verbose,
    )

    # 4. Setup logging
    setup_logging(ctx.verbosity)

    # 5. Create renderer
    render_ctx = RenderContext.auto_detect(
        format_override=OutputFormat.JSON if output.json else output.output_format,
    )
    renderer = UnifiedRenderer(render_ctx)

    try:
        yield ctx, renderer
    finally:
        ctx.close()
```

### 4.4 Step 3: Update cyclopts_ide.py

**File**: `src/codeintel/cli/cyclopts_ide.py`

```python
"""Cyclopts commands for IDE integration."""

from __future__ import annotations

from dataclasses import dataclass

from cyclopts import App, Parameter

from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    command_context,
    output_field,
    runtime_field,
)
from codeintel.cli.handlers.ide import ide_hints_handler

ide_app = App(name="ide", help="IDE integration commands.")


@ide_app.command(name="hints")
@dataclass
class IdeHintsCommand:
    """Get IDE hints for a file path.

    Display module context and subsystem information for IDE tooltips
    and code intelligence features.
    """

    path: str
    runtime: RuntimeCLI = runtime_field()
    output: OutputFormatCLI = output_field()

    def __call__(self) -> int:
        """Execute the IDE hints command."""
        with command_context(
            self.runtime,
            self.output,
            params={"rel_path": self.path},
        ) as (ctx, renderer):
            result = ide_hints_handler(ctx)
            return renderer.render_result(result)


__all__ = ["ide_app"]
```

### 4.5 Step 4: Update ide_handlers.py with Deprecations

**File**: `src/codeintel/cli/ide_handlers.py`

```python
"""Legacy IDE handlers.

.. deprecated:: 2.0
    This module is deprecated. Use codeintel.cli.handlers.ide instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "codeintel.cli.ide_handlers is deprecated. "
    "Use codeintel.cli.handlers.ide instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from codeintel.cli.handlers.ide import IdeHintsData, ide_hints_handler

# Keep legacy types with deprecation for external consumers
from codeintel.cli.handlers.base import setup_logging  # noqa: F401

# Legacy type alias (deprecated)
IdeHintsResult = IdeHintsData  # Backward compat

__all__ = [
    "IdeHintsData",
    "IdeHintsResult",
    "ide_hints_handler",
    "setup_logging",
]
```

### 4.6 Step 5: Create handlers/subsystem.py

Similar pattern to `handlers/ide.py` — migrate all subsystem handlers.

**File**: `src/codeintel/cli/handlers/subsystem.py`

```python
"""Subsystem handlers.

Handlers for subsystem listing, status, and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class SubsystemListData:
    """Data returned by subsystem_list_handler."""

    subsystems: list[dict[str, object]]
    total_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "subsystems": self.subsystems,
            "total_count": self.total_count,
        }


def subsystem_list_handler(ctx: EnhancedHandlerContext) -> CliResult[SubsystemListData]:
    """List all subsystems in the project.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    CliResult[SubsystemListData]
        List of subsystems.
    """
    ctx.logger.info("Listing subsystems")

    # Query subsystems from gateway
    rows = list(ctx.gateway.execute(
        "SELECT * FROM analytics.subsystems ORDER BY name"
    ))

    subsystems = [dict(row) for row in rows]

    return CliResult.ok(
        SubsystemListData(
            subsystems=subsystems,
            total_count=len(subsystems),
        )
    )


# Additional subsystem handlers follow same pattern:
# - subsystem_show_handler
# - subsystem_modules_handler
# - subsystem_metrics_handler
# - subsystem_graph_handler


__all__ = [
    "SubsystemListData",
    "subsystem_list_handler",
]
```

### 4.7 Wave 1 Acceptance Criteria

- [ ] `handlers/ide.py` created with `ide_hints_handler`
- [ ] `handlers/subsystem.py` created with all subsystem handlers
- [ ] `command_context()` helper added to `cyclopts_common.py`
- [ ] `cyclopts_ide.py` updated to use `command_context()`
- [ ] `cyclopts_subsystem.py` updated to use `command_context()`
- [ ] `ide_handlers.py` shows deprecation warning on import
- [ ] `subsystem_handlers.py` shows deprecation warning on import
- [ ] All tests pass
- [ ] `codeintel ide hints <path>` works end-to-end
- [ ] `codeintel subsystem list` works end-to-end

---

## 5. Wave 2: Medium Handlers (build, storage, history, health)

### 5.1 Scope

| Current File | New Location | Key Changes |
|--------------|--------------|-------------|
| `build_handlers.py` | `handlers/build.py` | Already uses alias, minimal changes |
| `storage_handlers.py` | `handlers/storage.py` | Remove RuntimeCliOptions |
| `history_handlers.py` | `handlers/history.py` | Remove RuntimeCliOptions |
| `cyclopts_health.py` | `handlers/health.py` | Remove stdout.write calls |

### 5.2 Step 1: Create handlers/build.py

`build_handlers.py` already imports `RuntimeCliOptions` from `common_handlers.py` as an alias. Migration is straightforward:

```python
"""Build handlers.

Handlers for build operations, status, and target management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class BuildStatusData:
    """Data returned by build_status_handler."""

    targets: list[dict[str, object]]
    total_count: int
    completed_count: int
    failed_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "targets": self.targets,
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
        }


def build_status_handler(ctx: EnhancedHandlerContext) -> CliResult[BuildStatusData]:
    """Get build status.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    CliResult[BuildStatusData]
        Build status data.
    """
    ctx.logger.info("Checking build status")

    # Query build tracking
    targets = _get_build_targets(ctx.gateway)

    completed = [t for t in targets if t.get("status") == "completed"]
    failed = [t for t in targets if t.get("status") == "failed"]

    return CliResult.ok(
        BuildStatusData(
            targets=targets,
            total_count=len(targets),
            completed_count=len(completed),
            failed_count=len(failed),
        )
    )


def _get_build_targets(gateway) -> list[dict[str, object]]:
    """Get build targets from tracking tables."""
    # Implementation
    return []


# Additional build handlers:
# - build_run_handler
# - build_clean_handler
# - build_target_handler
# etc.


__all__ = [
    "BuildStatusData",
    "build_status_handler",
]
```

### 5.3 Step 2: Create handlers/storage.py

```python
"""Storage handlers.

Handlers for storage operations, database management, and queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class StorageInfoData:
    """Data returned by storage_info_handler."""

    db_path: str
    size_bytes: int
    table_count: int
    tables: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "db_path": self.db_path,
            "size_bytes": self.size_bytes,
            "table_count": self.table_count,
            "tables": self.tables,
        }


def storage_info_handler(ctx: EnhancedHandlerContext) -> CliResult[StorageInfoData]:
    """Get storage information.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    CliResult[StorageInfoData]
        Storage information.
    """
    ctx.logger.info("Getting storage info for %s", ctx.db_path)

    # Get database stats
    tables = _get_table_info(ctx.gateway)
    db_size = ctx.db_path.stat().st_size if ctx.db_path.exists() else 0

    return CliResult.ok(
        StorageInfoData(
            db_path=str(ctx.db_path),
            size_bytes=db_size,
            table_count=len(tables),
            tables=tables,
        )
    )


def _get_table_info(gateway) -> list[dict[str, object]]:
    """Get table information from database."""
    # Implementation
    return []


__all__ = [
    "StorageInfoData",
    "storage_info_handler",
]
```

### 5.4 Step 3: Create handlers/health.py

`cyclopts_health.py` has direct `sys.stdout.write()` calls that need conversion:

**Before** (in cyclopts_health.py):

```python
def __call__(self) -> None:
    """Execute health check."""
    results = check_health()
    if self.json:
        sys.stdout.write(json.dumps(results, indent=2))
    else:
        for name, status in results.items():
            sys.stdout.write(f"{name}: {status}\n")
```

**After** (in handlers/health.py):

```python
"""Health check handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class HealthCheckData:
    """Data returned by health_check_handler."""

    checks: dict[str, str]
    all_healthy: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "checks": self.checks,
            "all_healthy": self.all_healthy,
        }


def health_check_handler(ctx: EnhancedHandlerContext) -> CliResult[HealthCheckData]:
    """Run health checks.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    CliResult[HealthCheckData]
        Health check results.
    """
    ctx.logger.info("Running health checks")

    checks = _run_health_checks(ctx)
    all_healthy = all(status == "healthy" for status in checks.values())

    return CliResult.ok(
        HealthCheckData(
            checks=checks,
            all_healthy=all_healthy,
        )
    )


def _run_health_checks(ctx: EnhancedHandlerContext) -> dict[str, str]:
    """Run individual health checks."""
    results = {}

    # Database check
    try:
        ctx.gateway.execute("SELECT 1")
        results["database"] = "healthy"
    except Exception:
        results["database"] = "unhealthy"

    # Add more checks as needed
    results["config"] = "healthy" if ctx.config else "unhealthy"

    return results


__all__ = [
    "HealthCheckData",
    "health_check_handler",
]
```

### 5.5 Wave 2 Acceptance Criteria

- [ ] `handlers/build.py` created with all build handlers
- [ ] `handlers/storage.py` created with all storage handlers
- [ ] `handlers/history.py` created with all history handlers
- [ ] `handlers/health.py` created with all health handlers
- [ ] `cyclopts_build.py` updated to use `command_context()`
- [ ] `cyclopts_storage.py` updated to use `command_context()`
- [ ] `cyclopts_health.py` updated — **zero** `sys.stdout.write()`
- [ ] All legacy handler files have deprecation warnings
- [ ] All tests pass
- [ ] `codeintel build status` works end-to-end
- [ ] `codeintel storage info` works end-to-end
- [ ] `codeintel health check --json` works end-to-end

---

## 6. Wave 3: Complex Handlers (datasets, docs, graphs, ops)

### 6.1 Scope

| Current File | `sys.stdout.write` | Lines | Key Challenges |
|--------------|--------------------|-------|----------------|
| `datasets_handlers.py` | 0 | ~2100 | Largest file, many nested options |
| `docs_handlers.py` | 9 | ~800 | Streaming output needs review |
| `graphs_handlers.py` | 12 | ~600 | Graph data serialization |
| `ops_handlers.py` | 0 | ~400 | Operation execution flow |

### 6.2 datasets_handlers.py Migration Strategy

This is the largest and most complex handler file. Migration approach:

1. **Extract data classes first** — Create typed return dataclasses
2. **Migrate one handler at a time** — Start with simplest
3. **Preserve nested option classes** as `ctx.params` entries
4. **Remove `RuntimeCliOptions`** at the end

**File**: `src/codeintel/cli/handlers/datasets.py`

```python
"""Dataset handlers.

Handlers for dataset export, listing, and schema operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult
from codeintel.cli.rendering.table import TableSpec

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


# --- Data Classes ---


@dataclass(frozen=True)
class DatasetListData:
    """Data returned by dataset_list_handler."""

    datasets: list[dict[str, object]]
    total_count: int
    table_spec: TableSpec = field(default=None)  # For table rendering hint

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "datasets": self.datasets,
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class DatasetExportData:
    """Data returned by dataset_export_handler."""

    table_key: str
    rows_exported: int
    output_path: str
    format: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "table_key": self.table_key,
            "rows_exported": self.rows_exported,
            "output_path": self.output_path,
            "format": self.format,
        }


# --- Handlers ---


def dataset_list_handler(ctx: EnhancedHandlerContext) -> CliResult[DatasetListData]:
    """List available datasets.

    Parameters
    ----------
    ctx
        Handler context. Optional params:
        - category: Filter by category
        - include_empty: Include empty datasets

    Returns
    -------
    CliResult[DatasetListData]
        List of datasets.
    """
    category = ctx.params.get("category")
    include_empty = ctx.params.get("include_empty", False)

    ctx.logger.info("Listing datasets (category=%s)", category)

    datasets = _fetch_datasets(ctx.gateway, category=category, include_empty=include_empty)

    return CliResult.ok(
        DatasetListData(
            datasets=datasets,
            total_count=len(datasets),
        )
    )


def dataset_export_handler(ctx: EnhancedHandlerContext) -> CliResult[DatasetExportData]:
    """Export a dataset to file.

    Parameters
    ----------
    ctx
        Handler context. Required params:
        - table_key: Dataset table key
        - output_path: Output file path
        - format: Export format (json, parquet, csv)

    Returns
    -------
    CliResult[DatasetExportData]
        Export result.
    """
    table_key = ctx.params.get("table_key")
    output_path = ctx.params.get("output_path")
    export_format = ctx.params.get("format", "json")

    if not table_key:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="table_key is required",
            )
        )

    if not output_path:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="output_path is required",
            )
        )

    ctx.logger.info("Exporting %s to %s", table_key, output_path)

    rows_exported = _export_dataset(
        ctx.gateway,
        str(table_key),
        str(output_path),
        str(export_format),
    )

    return CliResult.ok(
        DatasetExportData(
            table_key=str(table_key),
            rows_exported=rows_exported,
            output_path=str(output_path),
            format=str(export_format),
        )
    )


def _fetch_datasets(
    gateway,
    *,
    category: object = None,
    include_empty: object = False,
) -> list[dict[str, object]]:
    """Fetch dataset information from database."""
    # Implementation
    return []


def _export_dataset(
    gateway,
    table_key: str,
    output_path: str,
    format: str,
) -> int:
    """Export dataset to file."""
    # Implementation
    return 0


# Additional dataset handlers:
# - dataset_schema_handler
# - dataset_query_handler
# - dataset_preview_handler
# etc.


__all__ = [
    "DatasetExportData",
    "DatasetListData",
    "dataset_export_handler",
    "dataset_list_handler",
]
```

### 6.3 stdout.write Remediation (graphs_handlers.py, docs_handlers.py)

For files with `sys.stdout.write()` calls, each instance needs review:

**Pattern 1: Direct JSON output** → Replace with `CliResult.ok(data)`

```python
# Before
sys.stdout.write(json.dumps(result))

# After
return CliResult.ok(result)
```

**Pattern 2: Streaming output** → Use `StreamingEmitter` or JSONL format

```python
# Before (in loop)
for item in items:
    sys.stdout.write(json.dumps(item) + "\n")

# After (using JSONL via renderer)
return CliResult.ok(items)  # Renderer handles JSONL if format=JSONL
```

**Pattern 3: Progress output** → Use `renderer.emit_progress()`

```python
# Before
sys.stdout.write(f"Processing {i}/{total}\r")

# After (in command wiring, not handler)
renderer.emit_progress(i, total, "Processing")
```

### 6.4 handlers/graphs.py Example

```python
"""Graph handlers.

Handlers for graph operations, analysis, and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext


@dataclass(frozen=True)
class GraphStatsData:
    """Data returned by graph_stats_handler."""

    node_count: int
    edge_count: int
    graph_type: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "graph_type": self.graph_type,
            "metrics": self.metrics,
        }


def graph_stats_handler(ctx: EnhancedHandlerContext) -> CliResult[GraphStatsData]:
    """Get graph statistics.

    Parameters
    ----------
    ctx
        Handler context. Optional params:
        - graph_type: Type of graph ("call", "import", "dependency")

    Returns
    -------
    CliResult[GraphStatsData]
        Graph statistics.
    """
    graph_type = str(ctx.params.get("graph_type", "call"))

    ctx.logger.info("Getting stats for %s graph", graph_type)

    # Use lazy graph_runtime access
    graph_runtime = ctx.graph_runtime

    stats = _compute_graph_stats(graph_runtime, graph_type)

    return CliResult.ok(
        GraphStatsData(
            node_count=stats["nodes"],
            edge_count=stats["edges"],
            graph_type=graph_type,
            metrics=stats.get("metrics", {}),
        )
    )


def _compute_graph_stats(graph_runtime, graph_type: str) -> dict[str, Any]:
    """Compute graph statistics."""
    # Implementation using graph_runtime
    return {"nodes": 0, "edges": 0}


# Additional graph handlers:
# - graph_export_handler
# - graph_analyze_handler
# - graph_paths_handler
# etc.


__all__ = [
    "GraphStatsData",
    "graph_stats_handler",
]
```

### 6.5 Wave 3 Acceptance Criteria

- [ ] `handlers/datasets.py` created with all dataset handlers
- [ ] `handlers/docs.py` created with all docs handlers
- [ ] `handlers/graphs.py` created with all graph handlers
- [ ] `handlers/ops.py` created with all ops handlers
- [ ] **Zero** `sys.stdout.write()` calls in any handler file
- [ ] All `RuntimeCliOptions` variants removed
- [ ] All cyclopts files updated to use `command_context()`
- [ ] All legacy handler files have deprecation warnings
- [ ] All tests pass
- [ ] `codeintel datasets list` works end-to-end
- [ ] `codeintel graphs stats` works end-to-end
- [ ] `codeintel docs export` works end-to-end

---

## 7. Shared Infrastructure Updates

### 7.1 Update handlers/__init__.py

After all handlers are migrated:

```python
"""Unified CLI handlers package.

This package provides all CLI handler implementations following the
EnhancedHandlerContext → CliResult[T] pattern.
"""

from __future__ import annotations

from codeintel.cli.handlers.base import (
    HandlerContext,
    build_handler_context,
    get_handler_logger,
    open_handler_gateway,
    setup_logging,
)
from codeintel.cli.handlers.protocol import (
    EnhancedHandlerContext,
    HandlerProtocol,
    handler_context,
)

# Domain handlers
from codeintel.cli.handlers.build import (
    BuildStatusData,
    build_status_handler,
)
from codeintel.cli.handlers.datasets import (
    DatasetExportData,
    DatasetListData,
    dataset_export_handler,
    dataset_list_handler,
)
from codeintel.cli.handlers.graphs import (
    GraphStatsData,
    graph_stats_handler,
)
from codeintel.cli.handlers.health import (
    HealthCheckData,
    health_check_handler,
)
from codeintel.cli.handlers.ide import (
    IdeHintsData,
    ide_hints_handler,
)
from codeintel.cli.handlers.subsystem import (
    SubsystemListData,
    subsystem_list_handler,
)

__all__ = [
    # Protocol and context
    "EnhancedHandlerContext",
    "HandlerContext",
    "HandlerProtocol",
    "build_handler_context",
    "get_handler_logger",
    "handler_context",
    "open_handler_gateway",
    "setup_logging",
    # Build handlers
    "BuildStatusData",
    "build_status_handler",
    # Dataset handlers
    "DatasetExportData",
    "DatasetListData",
    "dataset_export_handler",
    "dataset_list_handler",
    # Graph handlers
    "GraphStatsData",
    "graph_stats_handler",
    # Health handlers
    "HealthCheckData",
    "health_check_handler",
    # IDE handlers
    "IdeHintsData",
    "ide_hints_handler",
    # Subsystem handlers
    "SubsystemListData",
    "subsystem_list_handler",
]
```

### 7.2 Update cyclopts_common.py Exports

Add `command_context` to `__all__`:

```python
__all__ = [
    # ... existing exports ...
    "command_context",
]
```

---

## 8. Testing Strategy

### 8.1 Unit Tests per Handler

Each handler gets unit tests in `tests/cli/handlers/test_<domain>.py`:

```python
"""Tests for IDE handlers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.config import CliConfig
from codeintel.cli.handlers.ide import IdeHintsData, ide_hints_handler
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.cli.resolution.types import ResolvedRuntime


def test_ide_hints_handler_missing_param(
    test_config: CliConfig,
    test_runtime: ResolvedRuntime,
) -> None:
    """Verify handler fails with missing rel_path."""
    ctx = EnhancedHandlerContext(
        config=test_config,
        runtime=test_runtime,
        params={},  # Missing rel_path
    )

    result = ide_hints_handler(ctx)

    expect_true(not result.success)
    expect_equal(result.error.type, "urn:codeintel:cli:validation/missing-param")


def test_ide_hints_handler_success(
    test_config: CliConfig,
    test_runtime: ResolvedRuntime,
    seeded_gateway,  # Gateway with test data
) -> None:
    """Verify handler returns hints for valid path."""
    ctx = EnhancedHandlerContext(
        config=test_config,
        runtime=test_runtime,
        params={"rel_path": "src/module.py"},
    )
    # Override gateway for test
    ctx._gateway = seeded_gateway

    result = ide_hints_handler(ctx)

    expect_true(result.success)
    expect_true(isinstance(result.data, IdeHintsData))
```

### 8.2 Integration Tests per Command

```python
"""Integration tests for IDE commands."""

from __future__ import annotations

from pathlib import Path

from codeintel.cli.cyclopts_app import app


def test_ide_hints_command_e2e(
    tmp_path: Path,
    cli_runner,
    seeded_project,
) -> None:
    """Test ide hints command end-to-end."""
    result = cli_runner.invoke(
        app,
        ["ide", "hints", "src/module.py", "--root", str(seeded_project)],
    )

    assert result.exit_code == 0
    assert "hints" in result.output or "hints" in result.stdout
```

### 8.3 Deprecation Warning Tests

```python
"""Tests for deprecation warnings."""

from __future__ import annotations

import warnings


def test_ide_handlers_deprecation_warning() -> None:
    """Verify ide_handlers.py emits deprecation warning on import."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Import deprecated module
        import codeintel.cli.ide_handlers  # noqa: F401

        deprecation_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert "handlers.ide" in str(deprecation_warnings[0].message)
```

---

## 9. Migration Checklist

### Wave 1 Checklist

- [ ] Create `handlers/ide.py` with `ide_hints_handler`
- [ ] Create `handlers/subsystem.py` with all handlers
- [ ] Add `command_context()` to `cyclopts_common.py`
- [ ] Update `cyclopts_ide.py` to use new pattern
- [ ] Update `cyclopts_subsystem.py` to use new pattern
- [ ] Add deprecation warning to `ide_handlers.py`
- [ ] Add deprecation warning to `subsystem_handlers.py`
- [ ] Create tests for new handlers
- [ ] Verify all existing tests pass
- [ ] Verify CLI commands work end-to-end

### Wave 2 Checklist

- [ ] Create `handlers/build.py` with all handlers
- [ ] Create `handlers/storage.py` with all handlers
- [ ] Create `handlers/history.py` with all handlers
- [ ] Create `handlers/health.py` with all handlers
- [ ] Remove `sys.stdout.write()` from `cyclopts_health.py`
- [ ] Update cyclopts files to use new pattern
- [ ] Add deprecation warnings to legacy files
- [ ] Create tests for new handlers
- [ ] Verify all existing tests pass
- [ ] Verify CLI commands work end-to-end

### Wave 3 Checklist

- [ ] Create `handlers/datasets.py` with all handlers
- [ ] Create `handlers/docs.py` with all handlers
- [ ] Create `handlers/graphs.py` with all handlers
- [ ] Create `handlers/ops.py` with all handlers
- [ ] Remove ALL `sys.stdout.write()` from handler files
- [ ] Remove ALL `RuntimeCliOptions` variants
- [ ] Update cyclopts files to use new pattern
- [ ] Add deprecation warnings to legacy files
- [ ] Create tests for new handlers
- [ ] Verify all existing tests pass
- [ ] Verify CLI commands work end-to-end

### Final Verification

- [ ] **Zero** `sys.stdout.write()` in `cli/handlers/`
- [ ] **Zero** per-module `RuntimeCliOptions` definitions
- [ ] All handlers accept `EnhancedHandlerContext`
- [ ] All handlers return `CliResult[T]`
- [ ] All legacy files emit deprecation warnings
- [ ] All tests pass (including new handler tests)
- [ ] All CLI commands work end-to-end
- [ ] Quality checks pass (pyright, pyrefly, ruff)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-10 | AI Assistant | Initial detailed plan based on Phase 1 & 2 implementation |
