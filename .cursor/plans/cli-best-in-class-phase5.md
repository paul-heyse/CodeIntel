# CLI Best-in-Class Implementation Plan (Phase 5)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 4 (Completed)

---

## Executive Summary

Phase 5 represents the **activation and extension** of the CLI infrastructure built in Phases 2-4. While previous phases created foundational components (types, validation, middleware, executor, telemetry, introspection), Phase 5 connects these into a fully operational system and extends it with new capabilities.

The six priorities address:

1. **Full Handler & Command Migration** — Wire all existing handlers through OperationExecutor
2. **Configuration System Activation** — Load and apply configuration at startup
3. **Help System Enhancement** — Rich contextual help with examples and schemas
4. **Async Operation Support** — Background jobs with status tracking
5. **Health Check System** — Verify environment and dependencies
6. **Plugin Architecture** — Extensible operations without core changes

### Why Phase 5 Matters

Without Phase 5, we have excellent infrastructure that isn't fully utilized:

| Component | Created In | Current State | After Phase 5 |
|-----------|------------|---------------|---------------|
| OperationExecutor | Phase 4 | Defined | Used by all commands |
| OperationRegistry | Phase 4 | Empty | Populated with all operations |
| Middleware | Phase 2 | Manual wiring | Automatic for all operations |
| Telemetry | Phase 4 | Opt-in | Active for all operations |
| cli_config_schema | Phase 3 | Defined | Loaded at startup |
| Introspection | Phase 4 | Available | Powers help system |

---

## Table of Contents

1. [Phase 5.1: Full Handler & Command Migration](#phase-51-full-handler--command-migration)
2. [Phase 5.2: Configuration System Activation](#phase-52-configuration-system-activation)
3. [Phase 5.3: Help System Enhancement](#phase-53-help-system-enhancement)
4. [Phase 5.4: Async Operation Support](#phase-54-async-operation-support)
5. [Phase 5.5: Health Check System](#phase-55-health-check-system)
6. [Phase 5.6: Plugin Architecture](#phase-56-plugin-architecture)
7. [Implementation Timeline](#implementation-timeline)
8. [Success Metrics](#success-metrics)
9. [Migration Guide](#migration-guide)

---

## Phase 5.1: Full Handler & Command Migration

### Value Proposition

The `OperationExecutor` and `OperationRegistry` from Phase 4 provide:
- Automatic validation before execution
- Middleware (logging, metrics, tracing) for all operations
- Progress tracking for long operations
- Consistent rendering

Currently, **zero** commands use this infrastructure. This phase wires everything together.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cyclopts Command Entry Point                          │
│                    (e.g., cyclopts_build.py)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OperationSpec (registered)                          │
│  operation_id: "build.status"                                           │
│  handler: build_status_handler_structured                               │
│  category: OperationCategory.BUILD                                      │
│  param_schema: BuildStatusSchema                                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OperationExecutor.execute()                         │
│  1. Validate params against schema                                      │
│  2. Run middleware stack (logging, metrics, tracing)                    │
│  3. Execute handler                                                     │
│  4. Render result                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Create operation specifications for all existing handlers
2. Register specifications in `OperationRegistry` at module load
3. Update cyclopts commands to use `OperationExecutor`
4. Migrate remaining handlers to return `CliResult[T]`
5. Replace ad-hoc error handling with `error_taxonomy` factories

### Implementation

#### File: `src/codeintel/cli/operations/__init__.py`

```python
"""Central operation registration.

This module imports all operation modules to trigger registration
of their OperationSpecs with the global registry.
"""

from __future__ import annotations

# Import modules to trigger registration
from codeintel.cli.operations import (
    build_operations,
    dataset_operations,
    docs_operations,
    graph_operations,
    op_operations,
    storage_operations,
)

__all__ = [
    "build_operations",
    "dataset_operations",
    "docs_operations",
    "graph_operations",
    "op_operations",
    "storage_operations",
]
```

#### File: `src/codeintel/cli/operations/build_operations.py`

```python
"""Build operation specifications."""

from __future__ import annotations

from codeintel.cli.build_handlers import (
    build_history_handler_structured,
    build_run_handler_structured,
    build_status_handler_structured,
)
from codeintel.cli.cli_validation import PathValidator, StringValidator, ValidationSchema
from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import register_operation

# Build Status Operation
_build_status_schema = ValidationSchema()
# No required params - uses runtime discovery

BUILD_STATUS_SPEC = register_operation(
    OperationSpec(
        operation_id="build.status",
        handler=build_status_handler_structured,
        category=OperationCategory.BUILD,
        param_schema=None,  # Runtime discovers project
        requires_progress=False,
        description="Show build target status",
    )
)

# Build Run Operation
_build_run_schema = ValidationSchema()
_build_run_schema.add("targets", StringValidator(min_length=0))

BUILD_RUN_SPEC = register_operation(
    OperationSpec(
        operation_id="build.run",
        handler=build_run_handler_structured,
        category=OperationCategory.BUILD,
        param_schema=_build_run_schema,
        requires_progress=True,
        estimated_duration=30.0,
        retryable=True,
        description="Execute build targets",
    )
)

# Build History Operation
BUILD_HISTORY_SPEC = register_operation(
    OperationSpec(
        operation_id="build.history",
        handler=build_history_handler_structured,
        category=OperationCategory.READ,
        requires_progress=False,
        description="Show build execution history",
    )
)

__all__ = [
    "BUILD_HISTORY_SPEC",
    "BUILD_RUN_SPEC",
    "BUILD_STATUS_SPEC",
]
```

#### File: `src/codeintel/cli/operations/op_operations.py`

```python
"""Op command operation specifications."""

from __future__ import annotations

from codeintel.cli.cli_validation import StringValidator, ValidationSchema
from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import register_operation
from codeintel.cli.ops_handlers import (
    op_call_handler_structured,
    op_list_handler_structured,
)

# Op List Operation
OP_LIST_SPEC = register_operation(
    OperationSpec(
        operation_id="op.list",
        handler=op_list_handler_structured,
        category=OperationCategory.READ,
        requires_progress=False,
        description="List available operations",
    )
)

# Op Call Operation
_op_call_schema = ValidationSchema()
_op_call_schema.add("operation_id", StringValidator(min_length=1, pattern=r"^[\w.]+$"))

OP_CALL_SPEC = register_operation(
    OperationSpec(
        operation_id="op.call",
        handler=op_call_handler_structured,
        category=OperationCategory.COMPUTE,
        param_schema=_op_call_schema,
        requires_progress=True,
        retryable=True,
        description="Call an operation by ID",
    )
)

__all__ = [
    "OP_CALL_SPEC",
    "OP_LIST_SPEC",
]
```

#### Updated Command: `src/codeintel/cli/cyclopts_build.py` (Example)

```python
"""Build commands using OperationExecutor."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.executor import get_executor
from codeintel.cli.operations.build_operations import (
    BUILD_HISTORY_SPEC,
    BUILD_RUN_SPEC,
    BUILD_STATUS_SPEC,
)

build_app = App(name="build", help="Build system commands")


@build_app.command()
def status(
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Show build target status."""
    executor = get_executor()
    result = executor.execute(
        BUILD_STATUS_SPEC,
        {},
        output_format=output_format,
    )
    if not result.result.success:
        raise SystemExit(1)


@build_app.command()
def run(
    targets: Annotated[
        list[str],
        Parameter(help="Targets to build"),
    ] = (),
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Execute build targets."""
    executor = get_executor()
    result = executor.execute(
        BUILD_RUN_SPEC,
        {"targets": list(targets)},
        output_format=output_format,
    )
    if not result.result.success:
        raise SystemExit(1)


@build_app.command()
def history(
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Show build execution history."""
    executor = get_executor()
    result = executor.execute(
        BUILD_HISTORY_SPEC,
        {},
        output_format=output_format,
    )
    if not result.result.success:
        raise SystemExit(1)
```

### Migration Checklist

For each command group, complete these steps:

- [ ] Create `operations/{group}_operations.py` with OperationSpecs
- [ ] Ensure all handlers return `CliResult[T]`
- [ ] Update cyclopts commands to use `get_executor().execute()`
- [ ] Remove manual error handling (let executor handle)
- [ ] Add to `operations/__init__.py` imports
- [ ] Verify middleware (logging, metrics) is active
- [ ] Test with `--dry-run` flag

---

## Phase 5.2: Configuration System Activation

### Value Proposition

Users need to customize CLI behavior without editing code:
- Default output format
- Retry policies
- Progress thresholds
- Telemetry settings
- Project defaults

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Sources                         │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  Command-Line │  Environment  │  Config File  │    Defaults     │
│    Flags      │   Variables   │ ~/.codeintel/ │   (built-in)    │
│   (highest)   │               │   config.yaml │    (lowest)     │
└───────┬───────┴───────┬───────┴───────┬───────┴────────┬────────┘
        │               │               │                │
        └───────────────┴───────────────┴────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   CliConfigManager    │
                    │   (merged config)     │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │  Executor   │ │  Middleware │ │  Renderer   │
        │   Config    │ │   Config    │ │   Config    │
        └─────────────┘ └─────────────┘ └─────────────┘
```

### Implementation

#### File: `src/codeintel/cli/config_loader.py`

```python
"""Configuration loading and application."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.cli_config_schema import CliConfig, CliConfigLoader
from codeintel.cli.cli_resilience import RetryPolicy

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".codeintel" / "config.yaml",
    Path.home() / ".codeintel" / "config.json",
    Path(".codeintel.yaml"),
    Path(".codeintel.json"),
]


@dataclass
class ResolvedConfig:
    """Fully resolved CLI configuration.

    Parameters
    ----------
    output_format
        Default output format.
    color
        Enable colored output.
    progress
        Show progress bars.
    progress_threshold
        Minimum duration (seconds) before showing progress.
    retry_policy
        Default retry policy.
    telemetry_enabled
        Enable telemetry.
    log_level
        Logging level.
    project_root
        Default project root.
    config_sources
        List of sources that contributed to this config.
    """

    output_format: str = "text"
    color: bool = True
    progress: bool = True
    progress_threshold: float = 2.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    telemetry_enabled: bool = True
    log_level: str = "WARNING"
    project_root: Path | None = None
    config_sources: list[str] = field(default_factory=list)


def load_config(
    *,
    config_file: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Load configuration from all sources.

    Parameters
    ----------
    config_file
        Explicit config file path.
    cli_overrides
        Command-line overrides.

    Returns
    -------
    ResolvedConfig
        Merged configuration.
    """
    sources: list[str] = []
    merged: dict[str, Any] = {}

    # 1. Built-in defaults
    merged.update(_get_defaults())
    sources.append("defaults")

    # 2. Config file
    file_config = _load_config_file(config_file)
    if file_config:
        merged.update(file_config)
        sources.append(f"file:{config_file or 'auto-discovered'}")

    # 3. Environment variables
    env_config = _load_env_config()
    if env_config:
        merged.update(env_config)
        sources.append("environment")

    # 4. CLI overrides
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
        sources.append("cli-flags")

    return _build_resolved_config(merged, sources)


def _get_defaults() -> dict[str, Any]:
    """Get built-in default values.

    Returns
    -------
    dict[str, Any]
        Default configuration.
    """
    return {
        "output_format": "text",
        "color": True,
        "progress": True,
        "progress_threshold": 2.0,
        "telemetry_enabled": True,
        "log_level": "WARNING",
    }


def _load_config_file(explicit_path: Path | None) -> dict[str, Any] | None:
    """Load configuration from file.

    Parameters
    ----------
    explicit_path
        Explicit path or None to search defaults.

    Returns
    -------
    dict[str, Any] | None
        Loaded config or None.
    """
    if explicit_path and explicit_path.exists():
        return CliConfigLoader.load_file(explicit_path)

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            LOG.debug("Loading config from %s", path)
            return CliConfigLoader.load_file(path)

    return None


def _load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns
    -------
    dict[str, Any]
        Environment-based config.
    """
    config: dict[str, Any] = {}

    env_mappings = {
        "CODEINTEL_OUTPUT_FORMAT": "output_format",
        "CODEINTEL_COLOR": ("color", _parse_bool),
        "CODEINTEL_PROGRESS": ("progress", _parse_bool),
        "CODEINTEL_TELEMETRY": ("telemetry_enabled", _parse_bool),
        "CODEINTEL_LOG_LEVEL": "log_level",
        "CODEINTEL_PROJECT_ROOT": ("project_root", Path),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if isinstance(mapping, tuple):
                key, converter = mapping
                config[key] = converter(value)
            else:
                config[mapping] = value

    return config


def _parse_bool(value: str) -> bool:
    """Parse boolean from string.

    Parameters
    ----------
    value
        String value.

    Returns
    -------
    bool
        Parsed boolean.
    """
    return value.lower() in ("true", "1", "yes", "on")


def _build_resolved_config(
    merged: dict[str, Any],
    sources: list[str],
) -> ResolvedConfig:
    """Build ResolvedConfig from merged dict.

    Parameters
    ----------
    merged
        Merged configuration dict.
    sources
        Sources that contributed.

    Returns
    -------
    ResolvedConfig
        Resolved configuration.
    """
    retry_config = merged.get("retry", {})
    retry_policy = RetryPolicy(
        max_attempts=retry_config.get("max_attempts", 3),
        initial_delay=retry_config.get("initial_delay", 0.5),
        backoff_factor=retry_config.get("backoff_factor", 2.0),
    )

    project_root = merged.get("project_root")
    if isinstance(project_root, str):
        project_root = Path(project_root)

    return ResolvedConfig(
        output_format=merged.get("output_format", "text"),
        color=merged.get("color", True),
        progress=merged.get("progress", True),
        progress_threshold=merged.get("progress_threshold", 2.0),
        retry_policy=retry_policy,
        telemetry_enabled=merged.get("telemetry_enabled", True),
        log_level=merged.get("log_level", "WARNING"),
        project_root=project_root,
        config_sources=sources,
    )


__all__ = [
    "ResolvedConfig",
    "load_config",
]
```

#### Config Commands: `src/codeintel/cli/cyclopts_config.py` (Enhanced)

```python
"""Configuration management commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.config_loader import DEFAULT_CONFIG_PATHS, load_config

config_app = App(name="config", help="Configuration management")


@config_app.command()
def show(
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Show current configuration with sources."""
    config = load_config()

    if output_format == OutputFormat.JSON:
        data = {
            "output_format": config.output_format,
            "color": config.color,
            "progress": config.progress,
            "progress_threshold": config.progress_threshold,
            "telemetry_enabled": config.telemetry_enabled,
            "log_level": config.log_level,
            "project_root": str(config.project_root) if config.project_root else None,
            "sources": config.config_sources,
        }
        print(json.dumps(data, indent=2))
    else:
        print("Current Configuration")
        print("=" * 40)
        print(f"Output Format:      {config.output_format}")
        print(f"Color:              {config.color}")
        print(f"Progress:           {config.progress}")
        print(f"Progress Threshold: {config.progress_threshold}s")
        print(f"Telemetry:          {config.telemetry_enabled}")
        print(f"Log Level:          {config.log_level}")
        print(f"Project Root:       {config.project_root or '(auto-discover)'}")
        print()
        print("Sources (highest to lowest priority):")
        for source in reversed(config.config_sources):
            print(f"  • {source}")


@config_app.command()
def paths() -> None:
    """Show configuration file search paths."""
    print("Configuration File Search Paths:")
    print()
    for path in DEFAULT_CONFIG_PATHS:
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {path}")


@config_app.command()
def init(
    path: Annotated[
        Path | None,
        Parameter(help="Path for config file"),
    ] = None,
) -> None:
    """Create a default configuration file."""
    target = path or (Path.home() / ".codeintel" / "config.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)

    default_config = '''# CodeIntel CLI Configuration
# https://codeintel.dev/docs/cli/configuration

# Output settings
output_format: text  # text, json
color: true
progress: true
progress_threshold: 2.0  # seconds before showing progress

# Telemetry
telemetry_enabled: true

# Logging
log_level: WARNING  # DEBUG, INFO, WARNING, ERROR

# Retry policy
retry:
  max_attempts: 3
  initial_delay: 0.5
  backoff_factor: 2.0

# Project defaults (optional)
# project_root: /path/to/project
'''

    target.write_text(default_config)
    print(f"Created configuration file: {target}")


@config_app.command()
def validate(
    path: Annotated[
        Path | None,
        Parameter(help="Config file to validate"),
    ] = None,
) -> None:
    """Validate a configuration file."""
    try:
        config = load_config(config_file=path)
        print("✓ Configuration is valid")
        print(f"  Loaded from: {', '.join(config.config_sources)}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        raise SystemExit(1)
```

---

## Phase 5.3: Help System Enhancement

### Value Proposition

Current help is generic cyclopts output. Users need:
- Detailed operation information with examples
- JSON Schema for parameters
- Contextual help based on what they're doing

### Implementation

#### File: `src/codeintel/cli/help_system.py`

```python
"""Enhanced help system with rich contextual output."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from codeintel.cli.introspection import (
    OperationInfo,
    get_operation_info,
    get_operation_schema,
    list_all_operations,
    list_operations_by_category,
    search_operations,
)


@dataclass
class HelpRenderer:
    """Render help content with rich formatting.

    Parameters
    ----------
    console
        Rich console for output.
    """

    console: Console

    def render_operation_detail(self, operation_id: str) -> bool:
        """Render detailed help for an operation.

        Parameters
        ----------
        operation_id
            Operation to describe.

        Returns
        -------
        bool
            True if operation found.
        """
        info = get_operation_info(operation_id)
        if info is None:
            self.console.print(f"[error]Operation not found: {operation_id}[/error]")
            return False

        # Header
        self.console.print()
        self.console.print(Panel(
            f"[bold]{info.operation_id}[/bold]\n\n{info.description}",
            title="Operation",
            border_style="cyan",
        ))

        # Metadata
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Category", info.category)
        table.add_row("Progress", "Yes" if info.requires_progress else "No")
        table.add_row("Retryable", "Yes" if info.retryable else "No")
        self.console.print(table)
        self.console.print()

        # Parameters
        if info.parameters:
            self.console.print("[heading]Parameters[/heading]")
            param_table = Table()
            param_table.add_column("Name", style="cyan")
            param_table.add_column("Type")
            param_table.add_column("Required")
            for param in info.parameters:
                param_table.add_row(
                    str(param.get("name", "")),
                    str(param.get("type", "")),
                    "Yes" if param.get("required") else "No",
                )
            self.console.print(param_table)
            self.console.print()

        # Examples
        if info.examples:
            self.console.print("[heading]Examples[/heading]")
            for example in info.examples:
                self.console.print(f"  [dim]$[/dim] {example}")
            self.console.print()

        return True

    def render_operation_schema(self, operation_id: str) -> bool:
        """Render JSON Schema for operation parameters.

        Parameters
        ----------
        operation_id
            Operation to describe.

        Returns
        -------
        bool
            True if schema found.
        """
        schema = get_operation_schema(operation_id)
        if schema is None:
            info = get_operation_info(operation_id)
            if info is None:
                self.console.print(f"[error]Operation not found: {operation_id}[/error]")
                return False
            self.console.print("[dim]No parameters for this operation[/dim]")
            return True

        self.console.print(json.dumps(schema, indent=2))
        return True

    def render_operation_list(self, *, by_category: bool = False) -> None:
        """Render list of all operations.

        Parameters
        ----------
        by_category
            Group by category.
        """
        if by_category:
            categories = list_operations_by_category()
            for category, op_ids in sorted(categories.items()):
                self.console.print(f"\n[heading]{category.upper()}[/heading]")
                for op_id in sorted(op_ids):
                    info = get_operation_info(op_id)
                    desc = info.description if info else ""
                    self.console.print(f"  [cyan]{op_id}[/cyan] - {desc}")
        else:
            operations = list_all_operations()
            table = Table(title="Available Operations")
            table.add_column("Operation ID", style="cyan")
            table.add_column("Category")
            table.add_column("Description")
            for info in sorted(operations, key=lambda x: x.operation_id):
                table.add_row(info.operation_id, info.category, info.description)
            self.console.print(table)

    def render_search_results(self, query: str) -> None:
        """Render search results.

        Parameters
        ----------
        query
            Search query.
        """
        results = search_operations(query)
        if not results:
            self.console.print(f"[dim]No operations matching: {query}[/dim]")
            return

        self.console.print(f"\n[heading]Operations matching '{query}'[/heading]\n")
        for info in results:
            self.console.print(f"[cyan]{info.operation_id}[/cyan]")
            self.console.print(f"  {info.description}")
            self.console.print()


def get_help_renderer() -> HelpRenderer:
    """Get a help renderer instance.

    Returns
    -------
    HelpRenderer
        Configured renderer.
    """
    from codeintel.cli.cli_render import CODEINTEL_THEME

    console = Console(theme=CODEINTEL_THEME)
    return HelpRenderer(console=console)


__all__ = [
    "HelpRenderer",
    "get_help_renderer",
]
```

#### Help Commands: `src/codeintel/cli/cyclopts_help.py`

```python
"""Enhanced help commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.help_system import get_help_renderer

help_app = App(name="help", help="Get help on operations")


@help_app.command(name="operation")
def operation_help(
    operation_id: Annotated[str, Parameter(help="Operation ID")],
) -> None:
    """Show detailed help for an operation.

    Examples
    --------
    codeintel help operation build.status
    codeintel help operation op.call
    """
    renderer = get_help_renderer()
    if not renderer.render_operation_detail(operation_id):
        raise SystemExit(1)


@help_app.command(name="schema")
def schema_help(
    operation_id: Annotated[str, Parameter(help="Operation ID")],
) -> None:
    """Show JSON Schema for operation parameters.

    Examples
    --------
    codeintel help schema build.run
    """
    renderer = get_help_renderer()
    if not renderer.render_operation_schema(operation_id):
        raise SystemExit(1)


@help_app.command(name="list")
def list_help(
    by_category: Annotated[
        bool,
        Parameter(help="Group by category"),
    ] = False,
) -> None:
    """List all available operations.

    Examples
    --------
    codeintel help list
    codeintel help list --by-category
    """
    renderer = get_help_renderer()
    renderer.render_operation_list(by_category=by_category)


@help_app.command(name="search")
def search_help(
    query: Annotated[str, Parameter(help="Search query")],
) -> None:
    """Search operations by name or description.

    Examples
    --------
    codeintel help search build
    codeintel help search "coverage analysis"
    """
    renderer = get_help_renderer()
    renderer.render_search_results(query)
```

---

## Phase 5.4: Async Operation Support

### Value Proposition

Long-running operations block the terminal. Users need:
- Run operations in background
- Check status without blocking
- Retrieve results when ready
- Cancel running operations

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    codeintel op call --background                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JobManager.submit()                         │
│  1. Generate job ID                                             │
│  2. Serialize operation spec + params                           │
│  3. Fork subprocess or add to queue                             │
│  4. Return job ID immediately                                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌─────────────────┐                           ┌─────────────────┐
│  Job Runner     │                           │   Job Store     │
│  (subprocess)   │                           │  ~/.codeintel/  │
│                 │ ─────writes status────▶   │  jobs/          │
└─────────────────┘                           └─────────────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────┐
                                          │ codeintel jobs      │
                                          │ status/output/cancel│
                                          └─────────────────────┘
```

### Implementation

#### File: `src/codeintel/cli/jobs.py`

```python
"""Background job management."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a background job.

    Parameters
    ----------
    job_id
        Unique job identifier.
    operation_id
        Operation being executed.
    params
        Operation parameters.
    status
        Current status.
    created_at
        Job creation timestamp.
    started_at
        Execution start timestamp.
    completed_at
        Completion timestamp.
    pid
        Process ID if running.
    exit_code
        Exit code if completed.
    error
        Error message if failed.
    """

    job_id: str
    operation_id: str
    params: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobInfo:
        """Create from dictionary.

        Parameters
        ----------
        data
            Dictionary data.

        Returns
        -------
        JobInfo
            Job info instance.
        """
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobStore:
    """Persistent storage for job information.

    Parameters
    ----------
    base_dir
        Base directory for job storage.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize job store."""
        self._base_dir = base_dir or (Path.home() / ".codeintel" / "jobs")
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Get path for job metadata.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        Path
            Metadata file path.
        """
        return self._base_dir / f"{job_id}.json"

    def _output_path(self, job_id: str) -> Path:
        """Get path for job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        Path
            Output file path.
        """
        return self._base_dir / f"{job_id}.output.json"

    def save(self, job: JobInfo) -> None:
        """Save job information.

        Parameters
        ----------
        job
            Job to save.
        """
        path = self._job_path(job.job_id)
        path.write_text(json.dumps(job.to_dict(), indent=2))

    def load(self, job_id: str) -> JobInfo | None:
        """Load job information.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None if not found.
        """
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return JobInfo.from_dict(data)

    def save_output(self, job_id: str, output: dict[str, Any]) -> None:
        """Save job output.

        Parameters
        ----------
        job_id
            Job identifier.
        output
            Output data.
        """
        path = self._output_path(job_id)
        path.write_text(json.dumps(output, indent=2))

    def load_output(self, job_id: str) -> dict[str, Any] | None:
        """Load job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        dict[str, Any] | None
            Output data or None.
        """
        path = self._output_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs.

        Parameters
        ----------
        status
            Filter by status.
        limit
            Maximum jobs to return.

        Returns
        -------
        list[JobInfo]
            Matching jobs.
        """
        jobs = []
        for path in sorted(self._base_dir.glob("*.json"), reverse=True):
            if path.name.endswith(".output.json"):
                continue
            job = JobInfo.from_dict(json.loads(path.read_text()))
            if status is None or job.status == status:
                jobs.append(job)
            if len(jobs) >= limit:
                break
        return jobs

    def delete(self, job_id: str) -> bool:
        """Delete job and its output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        bool
            True if deleted.
        """
        job_path = self._job_path(job_id)
        output_path = self._output_path(job_id)
        deleted = False
        if job_path.exists():
            job_path.unlink()
            deleted = True
        if output_path.exists():
            output_path.unlink()
        return deleted


class JobManager:
    """Manage background job execution.

    Parameters
    ----------
    store
        Job storage backend.
    """

    def __init__(self, store: JobStore | None = None) -> None:
        """Initialize job manager."""
        self._store = store or JobStore()

    def submit(
        self,
        operation_id: str,
        params: dict[str, Any],
    ) -> str:
        """Submit a job for background execution.

        Parameters
        ----------
        operation_id
            Operation to execute.
        params
            Operation parameters.

        Returns
        -------
        str
            Job ID.
        """
        job_id = str(uuid.uuid4())[:8]

        job = JobInfo(
            job_id=job_id,
            operation_id=operation_id,
            params=params,
            status=JobStatus.PENDING,
        )
        self._store.save(job)

        # Start subprocess
        self._start_job_process(job)

        return job_id

    def _start_job_process(self, job: JobInfo) -> None:
        """Start background process for job.

        Parameters
        ----------
        job
            Job to start.
        """
        # Build command to run job
        cmd = [
            sys.executable,
            "-m",
            "codeintel.cli.job_runner",
            "--job-id",
            job.job_id,
        ]

        # Start detached process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Update job with PID
        job.pid = process.pid
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._store.save(job)

    def get_status(self, job_id: str) -> JobInfo | None:
        """Get job status.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None.
        """
        job = self._store.load(job_id)
        if job and job.status == JobStatus.RUNNING and job.pid:
            # Check if process is still running
            if not self._is_process_running(job.pid):
                job.status = JobStatus.FAILED
                job.error = "Process terminated unexpectedly"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                self._store.save(job)
        return job

    def get_output(self, job_id: str) -> dict[str, Any] | None:
        """Get job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        dict[str, Any] | None
            Output data or None.
        """
        return self._store.load_output(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        bool
            True if cancelled.
        """
        job = self._store.load(job_id)
        if job is None:
            return False

        if job.status != JobStatus.RUNNING:
            return False

        if job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._store.save(job)
        return True

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs.

        Parameters
        ----------
        status
            Filter by status.
        limit
            Maximum jobs.

        Returns
        -------
        list[JobInfo]
            Jobs.
        """
        return self._store.list_jobs(status=status, limit=limit)

    def cleanup(self, *, max_age_days: int = 7) -> int:
        """Clean up old completed jobs.

        Parameters
        ----------
        max_age_days
            Maximum age in days.

        Returns
        -------
        int
            Number of jobs cleaned.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        cleaned = 0

        for job in self._store.list_jobs(limit=1000):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    completed_ts = datetime.fromisoformat(job.completed_at).timestamp()
                    if completed_ts < cutoff:
                        self._store.delete(job.job_id)
                        cleaned += 1

        return cleaned

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if process is running.

        Parameters
        ----------
        pid
            Process ID.

        Returns
        -------
        bool
            True if running.
        """
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def get_job_manager() -> JobManager:
    """Get global job manager.

    Returns
    -------
    JobManager
        Job manager instance.
    """
    return JobManager()


__all__ = [
    "JobInfo",
    "JobManager",
    "JobStatus",
    "JobStore",
    "get_job_manager",
]
```

#### Job Runner: `src/codeintel/cli/job_runner.py`

```python
"""Background job runner process."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from codeintel.cli.executor import get_executor
from codeintel.cli.jobs import JobStatus, JobStore
from codeintel.cli.operation_registry import get_operation_registry


def main() -> None:
    """Run a background job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    store = JobStore()
    job = store.load(args.job_id)

    if job is None:
        sys.exit(1)

    registry = get_operation_registry()
    spec = registry.get(job.operation_id)

    if spec is None:
        job.status = JobStatus.FAILED
        job.error = f"Unknown operation: {job.operation_id}"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        store.save(job)
        sys.exit(1)

    executor = get_executor()

    try:
        result = executor.execute(spec, job.params, render=False)

        if result.result.success:
            job.status = JobStatus.COMPLETED
            store.save_output(job.job_id, result.result.to_dict())
        else:
            job.status = JobStatus.FAILED
            job.error = result.result.error.detail if result.result.error else "Unknown error"

        job.exit_code = 0 if result.result.success else 1

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.exit_code = 1

    job.completed_at = datetime.now(timezone.utc).isoformat()
    store.save(job)

    sys.exit(job.exit_code or 0)


if __name__ == "__main__":
    main()
```

#### Jobs Commands: `src/codeintel/cli/cyclopts_jobs.py`

```python
"""Background job management commands."""

from __future__ import annotations

import json
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.jobs import JobStatus, get_job_manager

jobs_app = App(name="jobs", help="Manage background jobs")


@jobs_app.command()
def list_jobs(
    status: Annotated[
        str | None,
        Parameter(help="Filter by status (pending/running/completed/failed/cancelled)"),
    ] = None,
    limit: Annotated[int, Parameter(help="Maximum jobs to show")] = 20,
    output_format: Annotated[OutputFormat, Parameter(help="Output format")] = OutputFormat.TEXT,
) -> None:
    """List background jobs."""
    manager = get_job_manager()
    status_filter = JobStatus(status) if status else None
    jobs = manager.list_jobs(status=status_filter, limit=limit)

    if output_format == OutputFormat.JSON:
        print(json.dumps([j.to_dict() for j in jobs], indent=2))
        return

    console = Console()
    table = Table(title="Background Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Operation")
    table.add_column("Status")
    table.add_column("Created")

    for job in jobs:
        status_style = {
            JobStatus.PENDING: "yellow",
            JobStatus.RUNNING: "blue",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            JobStatus.CANCELLED: "dim",
        }.get(job.status, "")

        table.add_row(
            job.job_id,
            job.operation_id,
            f"[{status_style}]{job.status.value}[/{status_style}]",
            job.created_at[:19],
        )

    console.print(table)


@jobs_app.command()
def status(
    job_id: Annotated[str, Parameter(help="Job ID")],
    output_format: Annotated[OutputFormat, Parameter(help="Output format")] = OutputFormat.TEXT,
) -> None:
    """Get status of a background job."""
    manager = get_job_manager()
    job = manager.get_status(job_id)

    if job is None:
        print(f"Job not found: {job_id}")
        raise SystemExit(1)

    if output_format == OutputFormat.JSON:
        print(json.dumps(job.to_dict(), indent=2))
        return

    console = Console()
    console.print(f"[bold]Job ID:[/bold] {job.job_id}")
    console.print(f"[bold]Operation:[/bold] {job.operation_id}")
    console.print(f"[bold]Status:[/bold] {job.status.value}")
    console.print(f"[bold]Created:[/bold] {job.created_at}")

    if job.started_at:
        console.print(f"[bold]Started:[/bold] {job.started_at}")
    if job.completed_at:
        console.print(f"[bold]Completed:[/bold] {job.completed_at}")
    if job.error:
        console.print(f"[bold red]Error:[/bold red] {job.error}")


@jobs_app.command()
def output(
    job_id: Annotated[str, Parameter(help="Job ID")],
) -> None:
    """Get output of a completed job."""
    manager = get_job_manager()
    job = manager.get_status(job_id)

    if job is None:
        print(f"Job not found: {job_id}")
        raise SystemExit(1)

    if job.status != JobStatus.COMPLETED:
        print(f"Job is not completed (status: {job.status.value})")
        raise SystemExit(1)

    result = manager.get_output(job_id)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("No output available")


@jobs_app.command()
def cancel(
    job_id: Annotated[str, Parameter(help="Job ID")],
) -> None:
    """Cancel a running job."""
    manager = get_job_manager()

    if manager.cancel(job_id):
        print(f"Job {job_id} cancelled")
    else:
        print(f"Could not cancel job {job_id}")
        raise SystemExit(1)


@jobs_app.command()
def cleanup(
    max_age_days: Annotated[int, Parameter(help="Maximum age in days")] = 7,
) -> None:
    """Clean up old completed jobs."""
    manager = get_job_manager()
    cleaned = manager.cleanup(max_age_days=max_age_days)
    print(f"Cleaned up {cleaned} jobs")
```

---

## Phase 5.5: Health Check System

### Value Proposition

Users need to verify their CLI environment is properly configured before running operations.

### Implementation

#### File: `src/codeintel/cli/health.py`

```python
"""Health check system for CLI environment."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Health check status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a health check.

    Parameters
    ----------
    name
        Check name.
    status
        Check status.
    message
        Status message.
    duration_ms
        Check duration in milliseconds.
    details
        Additional details.
    """

    name: str
    status: CheckStatus
    message: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Complete health check report.

    Parameters
    ----------
    checks
        Individual check results.
    overall_status
        Overall health status.
    total_duration_ms
        Total check duration.
    """

    checks: list[CheckResult]
    overall_status: CheckStatus
    total_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "overall_status": self.overall_status.value,
            "total_duration_ms": self.total_duration_ms,
            "checks": [c.to_dict() for c in self.checks],
        }


class HealthChecker:
    """Run health checks on CLI environment."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self._checks: list[tuple[str, callable]] = [
            ("python_version", self._check_python_version),
            ("config_file", self._check_config_file),
            ("storage_connection", self._check_storage),
            ("project_discovery", self._check_project),
            ("operation_registry", self._check_registry),
            ("telemetry", self._check_telemetry),
        ]

    def run_all(self) -> HealthReport:
        """Run all health checks.

        Returns
        -------
        HealthReport
            Complete health report.
        """
        start = time.monotonic()
        results = []

        for name, check_fn in self._checks:
            check_start = time.monotonic()
            try:
                result = check_fn()
                result.duration_ms = (time.monotonic() - check_start) * 1000
            except Exception as e:
                result = CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    message=str(e),
                    duration_ms=(time.monotonic() - check_start) * 1000,
                )
            results.append(result)

        total_duration = (time.monotonic() - start) * 1000

        # Determine overall status
        if any(r.status == CheckStatus.FAIL for r in results):
            overall = CheckStatus.FAIL
        elif any(r.status == CheckStatus.WARN for r in results):
            overall = CheckStatus.WARN
        else:
            overall = CheckStatus.PASS

        return HealthReport(
            checks=results,
            overall_status=overall,
            total_duration_ms=total_duration,
        )

    def _check_python_version(self) -> CheckResult:
        """Check Python version.

        Returns
        -------
        CheckResult
            Check result.
        """
        import sys

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version >= (3, 13):
            return CheckResult(
                name="python_version",
                status=CheckStatus.PASS,
                message=f"Python {version_str}",
                details={"version": version_str},
            )
        elif version >= (3, 11):
            return CheckResult(
                name="python_version",
                status=CheckStatus.WARN,
                message=f"Python {version_str} (3.13+ recommended)",
                details={"version": version_str},
            )
        else:
            return CheckResult(
                name="python_version",
                status=CheckStatus.FAIL,
                message=f"Python {version_str} (3.11+ required)",
                details={"version": version_str},
            )

    def _check_config_file(self) -> CheckResult:
        """Check configuration file.

        Returns
        -------
        CheckResult
            Check result.
        """
        from codeintel.cli.config_loader import DEFAULT_CONFIG_PATHS

        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                return CheckResult(
                    name="config_file",
                    status=CheckStatus.PASS,
                    message=f"Found config: {path}",
                    details={"path": str(path)},
                )

        return CheckResult(
            name="config_file",
            status=CheckStatus.WARN,
            message="No config file found (using defaults)",
            details={"searched": [str(p) for p in DEFAULT_CONFIG_PATHS]},
        )

    def _check_storage(self) -> CheckResult:
        """Check storage connectivity.

        Returns
        -------
        CheckResult
            Check result.
        """
        try:
            # Try to import DuckDB
            import duckdb

            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1").fetchone()
            conn.close()

            return CheckResult(
                name="storage_connection",
                status=CheckStatus.PASS,
                message="DuckDB available",
                details={"engine": "duckdb"},
            )
        except ImportError:
            return CheckResult(
                name="storage_connection",
                status=CheckStatus.FAIL,
                message="DuckDB not installed",
            )
        except Exception as e:
            return CheckResult(
                name="storage_connection",
                status=CheckStatus.FAIL,
                message=f"Storage error: {e}",
            )

    def _check_project(self) -> CheckResult:
        """Check project discovery.

        Returns
        -------
        CheckResult
            Check result.
        """
        # Look for codeintel.yaml in current directory or parents
        cwd = Path.cwd()
        for path in [cwd, *cwd.parents]:
            config_path = path / "codeintel.yaml"
            if config_path.exists():
                return CheckResult(
                    name="project_discovery",
                    status=CheckStatus.PASS,
                    message=f"Project found: {path}",
                    details={"project_root": str(path)},
                )

        return CheckResult(
            name="project_discovery",
            status=CheckStatus.WARN,
            message="No project found in current directory",
        )

    def _check_registry(self) -> CheckResult:
        """Check operation registry.

        Returns
        -------
        CheckResult
            Check result.
        """
        from codeintel.cli.operation_registry import get_operation_registry

        registry = get_operation_registry()
        count = len(registry.operations)

        if count > 0:
            return CheckResult(
                name="operation_registry",
                status=CheckStatus.PASS,
                message=f"{count} operations registered",
                details={"operation_count": count},
            )
        else:
            return CheckResult(
                name="operation_registry",
                status=CheckStatus.WARN,
                message="No operations registered",
            )

    def _check_telemetry(self) -> CheckResult:
        """Check telemetry configuration.

        Returns
        -------
        CheckResult
            Check result.
        """
        from codeintel.cli.telemetry import TelemetryConfig

        config = TelemetryConfig.from_env()

        if config.enabled:
            return CheckResult(
                name="telemetry",
                status=CheckStatus.PASS,
                message="Telemetry enabled",
                details={"service_name": config.service_name},
            )
        else:
            return CheckResult(
                name="telemetry",
                status=CheckStatus.WARN,
                message="Telemetry disabled",
            )


def get_health_checker() -> HealthChecker:
    """Get health checker instance.

    Returns
    -------
    HealthChecker
        Health checker.
    """
    return HealthChecker()


__all__ = [
    "CheckResult",
    "CheckStatus",
    "HealthChecker",
    "HealthReport",
    "get_health_checker",
]
```

#### Health Commands: `src/codeintel/cli/cyclopts_health.py`

```python
"""Health check commands."""

from __future__ import annotations

import json
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.health import CheckStatus, get_health_checker

health_app = App(name="health", help="Check CLI environment health")


@health_app.default
def check_health(
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Run all health checks."""
    checker = get_health_checker()
    report = checker.run_all()

    if output_format == OutputFormat.JSON:
        print(json.dumps(report.to_dict(), indent=2))
        if report.overall_status == CheckStatus.FAIL:
            raise SystemExit(1)
        return

    console = Console()

    # Status symbols
    status_symbols = {
        CheckStatus.PASS: "[green]✓[/green]",
        CheckStatus.WARN: "[yellow]![/yellow]",
        CheckStatus.FAIL: "[red]✗[/red]",
        CheckStatus.SKIP: "[dim]-[/dim]",
    }

    table = Table(title="Health Check Results")
    table.add_column("Status", justify="center")
    table.add_column("Check")
    table.add_column("Message")
    table.add_column("Duration", justify="right")

    for check in report.checks:
        table.add_row(
            status_symbols.get(check.status, "?"),
            check.name,
            check.message,
            f"{check.duration_ms:.1f}ms",
        )

    console.print(table)
    console.print()

    overall_symbol = status_symbols.get(report.overall_status, "?")
    console.print(f"Overall: {overall_symbol} {report.overall_status.value.upper()}")
    console.print(f"Total time: {report.total_duration_ms:.1f}ms")

    if report.overall_status == CheckStatus.FAIL:
        raise SystemExit(1)
```

---

## Phase 5.6: Plugin Architecture

### Value Proposition

Users and teams need to extend CLI without modifying core code:
- Custom analytics operations
- Integration with internal tools
- Domain-specific commands

### Implementation

#### File: `src/codeintel/cli/plugins.py`

```python
"""Plugin architecture for CLI extensions."""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from codeintel.cli.executor import OperationSpec
from codeintel.cli.operation_registry import get_operation_registry

LOG = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIRS = [
    Path.home() / ".codeintel" / "plugins",
    Path("/etc/codeintel/plugins"),
]


class PluginProtocol(Protocol):
    """Protocol for CLI plugins.

    Plugins must implement this interface to be loadable.
    """

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version."""
        ...

    @property
    def description(self) -> str:
        """Plugin description."""
        ...

    def get_operations(self) -> list[OperationSpec[Any]]:
        """Get operations provided by this plugin.

        Returns
        -------
        list[OperationSpec[Any]]
            Operations to register.
        """
        ...

    def initialize(self) -> None:
        """Initialize the plugin.

        Called after loading but before operations are registered.
        """
        ...


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    description
        Plugin description.
    path
        Plugin file path.
    operations
        Number of operations provided.
    enabled
        Whether plugin is enabled.
    """

    name: str
    version: str
    description: str
    path: Path
    operations: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "path": str(self.path),
            "operations": self.operations,
            "enabled": self.enabled,
        }


@dataclass
class PluginManager:
    """Manage CLI plugins.

    Parameters
    ----------
    plugin_dirs
        Directories to search for plugins.
    loaded_plugins
        Currently loaded plugins.
    """

    plugin_dirs: list[Path] = field(default_factory=lambda: list(DEFAULT_PLUGIN_DIRS))
    loaded_plugins: dict[str, PluginInfo] = field(default_factory=dict)

    def discover(self) -> list[Path]:
        """Discover available plugins.

        Returns
        -------
        list[Path]
            Plugin file paths.
        """
        plugins = []
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            for path in plugin_dir.glob("*.py"):
                if not path.name.startswith("_"):
                    plugins.append(path)
        return plugins

    def load_plugin(self, path: Path) -> PluginInfo | None:
        """Load a plugin from file.

        Parameters
        ----------
        path
            Plugin file path.

        Returns
        -------
        PluginInfo | None
            Plugin info or None on failure.
        """
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                LOG.warning("Could not load plugin: %s", path)
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)

            # Get plugin instance
            plugin_factory = getattr(module, "create_plugin", None)
            if plugin_factory is None:
                LOG.warning("Plugin missing create_plugin(): %s", path)
                return None

            plugin = plugin_factory()

            # Initialize
            plugin.initialize()

            # Register operations
            operations = plugin.get_operations()
            registry = get_operation_registry()
            for op_spec in operations:
                registry.register(op_spec)

            info = PluginInfo(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                path=path,
                operations=len(operations),
            )
            self.loaded_plugins[plugin.name] = info

            LOG.info("Loaded plugin: %s v%s (%d operations)", info.name, info.version, info.operations)
            return info

        except Exception as e:
            LOG.exception("Failed to load plugin %s: %s", path, e)
            return None

    def load_all(self) -> list[PluginInfo]:
        """Load all discovered plugins.

        Returns
        -------
        list[PluginInfo]
            Loaded plugin info.
        """
        loaded = []
        for path in self.discover():
            info = self.load_plugin(path)
            if info:
                loaded.append(info)
        return loaded

    def list_plugins(self) -> list[PluginInfo]:
        """List loaded plugins.

        Returns
        -------
        list[PluginInfo]
            All loaded plugins.
        """
        return list(self.loaded_plugins.values())

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get plugin by name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        PluginInfo | None
            Plugin info or None.
        """
        return self.loaded_plugins.get(name)


# Global plugin manager
_PLUGIN_MANAGER: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager.

    Returns
    -------
    PluginManager
        Plugin manager instance.
    """
    global _PLUGIN_MANAGER
    if _PLUGIN_MANAGER is None:
        _PLUGIN_MANAGER = PluginManager()
    return _PLUGIN_MANAGER


def initialize_plugins() -> None:
    """Initialize and load all plugins.

    Called during CLI startup.
    """
    manager = get_plugin_manager()
    manager.load_all()


__all__ = [
    "PluginInfo",
    "PluginManager",
    "PluginProtocol",
    "get_plugin_manager",
    "initialize_plugins",
]
```

#### Plugin Commands: `src/codeintel/cli/cyclopts_plugins.py`

```python
"""Plugin management commands."""

from __future__ import annotations

import json
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.plugins import get_plugin_manager

plugins_app = App(name="plugins", help="Manage CLI plugins")


@plugins_app.command(name="list")
def list_plugins(
    output_format: Annotated[
        OutputFormat,
        Parameter(help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """List installed plugins."""
    manager = get_plugin_manager()
    plugins = manager.list_plugins()

    if output_format == OutputFormat.JSON:
        print(json.dumps([p.to_dict() for p in plugins], indent=2))
        return

    if not plugins:
        print("No plugins installed")
        return

    console = Console()
    table = Table(title="Installed Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Operations", justify="right")
    table.add_column("Description")

    for plugin in plugins:
        table.add_row(
            plugin.name,
            plugin.version,
            str(plugin.operations),
            plugin.description,
        )

    console.print(table)


@plugins_app.command()
def discover() -> None:
    """Discover available plugins."""
    manager = get_plugin_manager()
    paths = manager.discover()

    if not paths:
        print("No plugins found")
        print("\nPlugin directories searched:")
        for d in manager.plugin_dirs:
            print(f"  • {d}")
        return

    print("Available plugins:")
    for path in paths:
        loaded = path.stem in [p.name for p in manager.loaded_plugins.values()]
        status = "✓ loaded" if loaded else "○ available"
        print(f"  {status} {path.name}")


@plugins_app.command()
def info(
    name: Annotated[str, Parameter(help="Plugin name")],
) -> None:
    """Show details about a plugin."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(name)

    if plugin is None:
        print(f"Plugin not found: {name}")
        raise SystemExit(1)

    console = Console()
    console.print(f"[bold]Name:[/bold] {plugin.name}")
    console.print(f"[bold]Version:[/bold] {plugin.version}")
    console.print(f"[bold]Description:[/bold] {plugin.description}")
    console.print(f"[bold]Path:[/bold] {plugin.path}")
    console.print(f"[bold]Operations:[/bold] {plugin.operations}")
    console.print(f"[bold]Enabled:[/bold] {plugin.enabled}")
```

#### Example Plugin Template

```python
"""Example CodeIntel CLI plugin.

Save this file to ~/.codeintel/plugins/my_plugin.py
"""

from __future__ import annotations

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.results import CliResult


class MyPlugin:
    """Example plugin implementation."""

    @property
    def name(self) -> str:
        """Plugin name."""
        return "my-plugin"

    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Plugin description."""
        return "An example plugin for demonstration"

    def initialize(self) -> None:
        """Initialize the plugin."""
        pass

    def get_operations(self) -> list[OperationSpec]:
        """Get operations provided by this plugin."""
        return [
            OperationSpec(
                operation_id="my-plugin.hello",
                handler=self._hello_handler,
                category=OperationCategory.READ,
                description="Say hello from plugin",
            ),
        ]

    @staticmethod
    def _hello_handler() -> CliResult[dict[str, str]]:
        """Handler for hello operation."""
        return CliResult.ok({"message": "Hello from my-plugin!"})


def create_plugin() -> MyPlugin:
    """Factory function to create plugin instance."""
    return MyPlugin()
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority | Effort |
|-------|----------|--------------|----------|--------|
| 5.1 Handler Migration | 5-7 days | None | Critical | High |
| 5.2 Config Activation | 2-3 days | 5.1 | High | Medium |
| 5.3 Help Enhancement | 2-3 days | 5.1 | High | Medium |
| 5.4 Async Operations | 4-5 days | 5.1 | Medium | High |
| 5.5 Health Checks | 2-3 days | 5.2 | Medium | Low |
| 5.6 Plugin Architecture | 3-4 days | 5.1 | Medium | Medium |

**Total estimated time: 18-25 days**

### Recommended Order

```
Week 1-2:     [======== Phase 5.1 ========]
Week 2-3:           [=== 5.2 ===][=== 5.3 ===]
Week 3-4:                 [======= Phase 5.4 =======]
Week 4-5:                       [== 5.5 ==][==== 5.6 ====]
```

---

## Success Metrics

### Technical Quality

- [ ] 100% of handlers return `CliResult[T]`
- [ ] 100% of commands use `OperationExecutor`
- [ ] All operations registered in `OperationRegistry`
- [ ] Config loaded automatically on startup
- [ ] Background jobs complete successfully

### User Experience

- [ ] `codeintel help operation <id>` works for all operations
- [ ] `codeintel config show` displays merged config
- [ ] `codeintel health` passes in clean environment
- [ ] `codeintel jobs list` shows background jobs
- [ ] Plugin example loads and registers operations

### Operational

- [ ] Telemetry active for all operation executions
- [ ] Middleware logging shows for all operations
- [ ] Health checks identify misconfiguration
- [ ] Jobs can be cancelled while running

---

## Migration Guide

### For New Handlers

Always create a structured handler and operation spec:

```python
# 1. Define result type
@dataclass
class MyResult:
    count: int
    items: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "items": self.items}

# 2. Create structured handler
def my_handler_structured(**params) -> CliResult[MyResult]:
    # ... implementation
    return CliResult.ok(MyResult(count=10, items=["a", "b"]))

# 3. Register operation
MY_SPEC = register_operation(
    OperationSpec(
        operation_id="domain.my_operation",
        handler=my_handler_structured,
        category=OperationCategory.READ,
        description="My operation description",
    )
)

# 4. Use in command
@app.command()
def my_command(output_format: OutputFormat = OutputFormat.TEXT) -> None:
    result = get_executor().execute(MY_SPEC, {}, output_format=output_format)
    if not result.result.success:
        raise SystemExit(1)
```

### For Existing Handlers

1. Add `_structured` variant returning `CliResult[T]`
2. Create `OperationSpec` in `operations/` module
3. Update command to use executor
4. Deprecate old handler (keep for compatibility)
5. Remove deprecated handler in next major version

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/cli/operations/__init__.py` | Central operation registration |
| `src/codeintel/cli/operations/build_operations.py` | Build operation specs |
| `src/codeintel/cli/operations/op_operations.py` | Op command specs |
| `src/codeintel/cli/operations/dataset_operations.py` | Dataset operation specs |
| `src/codeintel/cli/operations/docs_operations.py` | Docs operation specs |
| `src/codeintel/cli/operations/graph_operations.py` | Graph operation specs |
| `src/codeintel/cli/operations/storage_operations.py` | Storage operation specs |
| `src/codeintel/cli/config_loader.py` | Configuration loading |
| `src/codeintel/cli/help_system.py` | Enhanced help rendering |
| `src/codeintel/cli/cyclopts_help.py` | Help commands |
| `src/codeintel/cli/jobs.py` | Background job management |
| `src/codeintel/cli/job_runner.py` | Job subprocess runner |
| `src/codeintel/cli/cyclopts_jobs.py` | Jobs commands |
| `src/codeintel/cli/health.py` | Health check system |
| `src/codeintel/cli/cyclopts_health.py` | Health commands |
| `src/codeintel/cli/plugins.py` | Plugin architecture |
| `src/codeintel/cli/cyclopts_plugins.py` | Plugin commands |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/cyclopts_app.py` | Register new command groups, initialize plugins |
| `src/codeintel/cli/cyclopts_build.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_ops.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_datasets.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_docs.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_graphs.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_storage.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_config.py` | Add config commands |

---

*End of Phase 5 Implementation Plan*

